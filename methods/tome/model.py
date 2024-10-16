from timm.models import VisionTransformer
from torch import nn

from methods import ToMeBlockWrapper


class ToMeVit(nn.Module):
    def __init__(self, model: VisionTransformer, use_budget_emb=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model = model

        for i, b in enumerate(self.blocks):
            if i == 0:
                continue
            self.blocks[i] = ToMeBlockWrapper(block=b, dim=self.cls_token.shape[-1],
                                              num_patches=self.patch_embed.num_patches, *args, **kwargs)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self._model, item)

    def forward(self, x):
        x = self._model.forward(x)

        return x
