import torch
from timm.models import checkpoint_seq
from torch import nn
from timm.models.vision_transformer import Block, VisionTransformer


class AdaptiveBlock(nn.Module):
    def __init__(self, block: Block, num_patches, dim, eval_threshold=1e-3, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._block = block
        self.num_patches = num_patches

        self.fl = nn.Sequential(nn.Linear(dim, 1, bias=True), nn.Sigmoid()).to(next(self.parameters()).device)
        self.fh = nn.Sequential(nn.Linear(dim, 1, bias=True), nn.Sigmoid()).to(next(self.parameters()).device)

        self.fl[-2].bias.data.normal_(5, 0.1)
        self.fh[-2].bias.data.normal_(-5, 0.1)

        self.last_mask = None
        self.eval_threshold = eval_threshold

    @property
    def base_block(self):
        return self._block

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self._block, item)

    def get_mask(self, x, mask=None):
        a_log = self.fl(x)

        return a_log.sigmoid()

    def forward(self, x, alpha, prev_mask=None):
        if alpha is None:
            return self._block(x), None

        # mask = self.get_mask(x[:, 1:-1])
        mask = self.fl(x[:, 1:-1])

        th = self.fh(x[:, -2:-1])

        # torch.einsum('btd,btd->btt', mask, th)

        # mask = torch.einsum('btd,bod->bt', mask / torch.linalg.norm(mask, 2, -1, keepdim=True), th / torch.linalg.norm(th, 2, -1, keepdim=True)) + 0.5
        # mask = torch.relu(mask - alpha) / (1 - alpha)
        mask = torch.relu(mask - th)

        ones = torch.ones_like(mask[:, :1])
        mask = torch.cat((ones, mask, ones), 1)

        if not self.training:
            mask = torch.where(mask > self.eval_threshold, mask, torch.zeros_like(mask))

        if not self.training and len(x) == 1:
            bmask = mask > 0
            self.last_mask = bmask

            if mask.sum(1) == 2:
                return self._block(x), None

            x = x[bmask.expand_as(x)].view(1, -1, x.shape[-1])
            x = self._block(x)
            return x, None
        else:
            if prev_mask is not None:
                mask = mask * prev_mask

            self.last_mask = mask

            x = self._block(x * mask) * mask

        return x, mask


class SemanticVit(nn.Module):
    def __init__(self,
                 model: VisionTransformer,
                 use_budget_emb=False,
                 freeze_model=False,
                 blocks_to_transform=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._model = model

        if freeze_model:
            for p in self._model.parameters():
                if p.requires_grad:
                    p.requires_grad_(False)

        self.use_budget_emb = use_budget_emb
        if not use_budget_emb:
            self.budget_token_lower = nn.Parameter(torch.randn_like(self.cls_token) * 0.02)
            self.budget_token_upper = nn.Parameter(torch.randn_like(self.cls_token) * 0.02)
        else:
            self.budget_embs = nn.Embedding(100, len(self.cls_token.squeeze())).to(self.cls_token.device)

        if blocks_to_transform is None:
            blocks = self.blocks
        else:
            blocks = self.blocks[:blocks_to_transform]

        for i, b in enumerate(blocks):
            if i == 0:
                continue
            self.blocks[i] = AdaptiveBlock(block=b, dim=self.cls_token.shape[-1],
                                           num_patches=self.patch_embed.num_patches, *args, **kwargs)

        self.current_alpha = None
        # self.blocks[-2] = CustomBlock(block=self.blocks[-2], dim=self.budget_token_lower.shape[-1], *args, **kwargs)

    def _pos_embed(self, x, alpha=0.5):
        x = self._model._pos_embed(x)

        if alpha is None or alpha == 1:
            return x, None

        if self.use_budget_emb:
            if isinstance(alpha, float):
                alpha = torch.tensor([int(alpha * 100)], device=x.device)
                budget_embedding = self.budget_embs(alpha)
            elif isinstance(alpha, tuple):
                a, b = min(alpha), max(alpha)
                delta = b - a
                alpha = torch.linspace(a, b, len(x), device=x.device, dtype=torch.float)
                alpha = torch.empty(len(x), 1, device=x.device, dtype=torch.float).uniform_(a, b)
                alpha = (alpha * 100).long()
                budget_embedding = self.budget_embs(alpha)
            else:
                assert False

            alpha = alpha[..., None]
            alpha = alpha / 100
        else:
            if isinstance(alpha, float):
                budget_embedding = self.budget_token_lower * alpha + self.budget_token_upper * (1 - alpha)
            elif isinstance(alpha, tuple):
                a, b = min(alpha), max(alpha)
                sigma = (b - a) / len(x)
                mean = torch.linspace(a, b, len(x), device=x.device, dtype=torch.float)[..., None, None]
                alpha = torch.normal(mean, sigma).clip(a, b)
                # alpha = torch.empty(len(x), 1, 1, device=x.device, dtype=torch.float).uniform_(a, b)
                budget_embedding = self.budget_token_lower * alpha + self.budget_token_upper * (1 - alpha)
            else:
                assert False

        self.last_alpha = alpha
        budget_embedding = budget_embedding.expand(x.shape[0], -1, -1)

        # x = x + budget_embedding
        x = torch.cat((x, budget_embedding), 1)
        # x = torch.cat((x[:, 0:1], budget_embedding, x[:, 1:]), 1)
        return x, alpha

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self._model, item)

    # def __setattr__(self, name, value):
    #     if hasattr(self._model, name):
    #         self._model.__setattr__(name, value)
    #     else:
    #         self.name = value

    def forward_features(self, x: torch.Tensor, alpha=0.5) -> torch.Tensor:
        x = self.patch_embed(x)
        x, alpha = self._pos_embed(x, alpha=alpha)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            prev_mask = None
            for b in self.blocks:
                if isinstance(b, AdaptiveBlock):
                    x, prev_mask = b(x, prev_mask=prev_mask, alpha=alpha)
                    # prev_mask = None
                else:
                    # TODO: fix when BS = 1 and block is not the last
                    if prev_mask is not None:
                        x = b(x * prev_mask) * prev_mask
                    else:
                        x = b(x)

        x = self.norm(x)
        return x

    def forward(self, x, alpha=None):
        if self.training and alpha is None:
            alpha = (0 + 1e-3, 1 - 1e-3)

        if alpha is None and self.current_alpha is not None:
            alpha = self.current_alpha

        x = self.forward_features(x, alpha=alpha)
        x = self.forward_head(x)

        return x
