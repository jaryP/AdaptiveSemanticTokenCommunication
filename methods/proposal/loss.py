import torch
from torch import nn

from methods.proposal import SemanticVit, AdaptiveBlock


class AdaptiveTokenLoss(nn.Module):
    def __init__(self,
                 model: SemanticVit,
                 inner_flops_w: float = 1,
                 output_flops_w: float = 1,
                 inner_flops_type: str = 'margin',
                 margin: float = 0.01,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.inner_flops_w = inner_flops_w
        self.output_flops_w = output_flops_w
        self.inner_flops_type = inner_flops_type
        self._model = model
        self.margin = margin

        assert inner_flops_type in ['margin', 'l1', 'bml1', 'ml1']

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        alphas = self._model.last_alpha.squeeze()
        masks = []

        for b in [b for b in self._model.blocks if isinstance(b, AdaptiveBlock)]:
            mask = b.last_mask.squeeze()[:, 1:-1]
            masks.append(mask)

        alphas = alphas[:, None]
        masks = torch.stack(masks, 1)
        output_loss = (torch.abs(masks[:, -1].mean(-1) - alphas.squeeze()) - self.margin).relu()

        if self.inner_flops_type == 'ml1':
            m = (torch.abs(masks[:, :-1].mean(-1) - alphas) - self.margin).detach()
            l1 = masks[:, :-1].mean(-1)

            inner_loss = (m * l1).mean(-1)

        elif self.inner_flops_type == 'bml1':
            m = ((torch.abs(masks[:, :-1].mean(-1) - alphas) - self.margin) > 0).float()
            l1 = masks[:, :-1].mean(-1)

            inner_loss = (m * l1).mean(-1)

        elif self.inner_flops_type == 'l1':
            # masks[:, :-1].mean(-1) * (torch.abs(masks[:, :-1].mean(-1) - alphas) - self.margin)
            inner_loss = masks[:, :-1].mean(-1).mean(1)
        else:
            inner_loss = (torch.abs(masks[:, :-1].mean(-1) - alphas) - self.margin).relu().mean(1)

        loss = (nn.functional.cross_entropy(x, y) + output_loss.mean() * self.output_flops_w +
                inner_loss.mean() * self.inner_flops_w)

        return loss
