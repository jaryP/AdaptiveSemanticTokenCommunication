from typing import Tuple

import torch
from torch import nn

from .utils import bipartite_soft_matching, merge_wavg


class ToMeAttention(nn.Module):
    def __init__(self, attention, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = attention

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.attention, item)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k.mean(1)



class ToMeBlockWrapper(nn.Module):
    def __init__(self, block, r, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = block
        self._r = r
        self.attn = ToMeAttention(block.attn)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.block, item)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, v):
        # assert 0.5 > v
        self._r = v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        # attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x))
        x = x + self.drop_path1(x_attn)

        r = self.r
        if r > 0:
            # Apply ToMe here
            if isinstance(r, float):
                assert 0.5 >= r
                r = int(r * x.shape[1])
            else:
                assert int(0.5 * x.shape[1]) <= r

            merge, _ = bipartite_soft_matching(
                metric,
                r,
                True,
                False,
            )
            # if self._tome_info["trace_source"]:
            #     self._tome_info["source"] = merge_source(
            #         merge, x, self._tome_info["source"]
            #     )
            x, size = merge_wavg(merge, x, None)

        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
