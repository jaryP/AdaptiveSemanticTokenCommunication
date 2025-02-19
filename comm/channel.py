import math
from typing import Union, Tuple, Sequence

import numpy as np
import torch
from torch import nn


class CleanChannel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x

    def set_noise(self, v):
        pass


class OpenChannel(nn.Module):

    def __init__(self, channel, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._channel = channel
        self.size = size
        self._snr = None

    def __enter__(self):
        self._snr = self._channel.get_snr(self.size)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._snr = None

        return

    def forward(self, x):
        r = self._channel(x, snr=self._snr)

        return r

# class SNRChannel(nn.Module):
#     def __init__(self, snr: Union[float, Tuple[float, float]], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         assert snr is not None
#         self._base_snr = snr
#
#     def get_snr(self, size: int, device: Union[str, torch.device] = 'cpu', *args, **kwargs):
#         raise NotImplementedError
#
#     @property
#     def snr(self):
#         return self._snr
#
#     @snr.setter
#     def snr(self, v: Union[float, str]):
#         if isinstance(v, str):
#             assert v == 'random'
#             self._snr = None
#         else:
#             self._snr = v


class GaussianNoiseChannel(CleanChannel):
    def __init__(self,
                 snr: Union[float, Tuple[float, float]],
                 test_snr = None,
                 use_training_snr=True,
                 dims=-1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert snr is not None
        self._base_snr = snr
        self.use_training_snr = use_training_snr

        self._snr = 0
        self._test_snr = snr
        self._train_snr = test_snr if test_snr is not None else snr

        self.dims = dims
        self.symbols = None
        self.channel_input = None

    def get_snr(self, size: int, device: Union[str, torch.device] = 'cpu'):
        if self.training:
            snr = self._train_snr
        else:
            snr = self._test_snr

        if snr is None:
            return snr

        if isinstance(snr, Sequence):
            r1, r2 = snr
            snr = (r1 - r2) * torch.rand(size, device=device) + r2

        # else:
        #     snr = self._base_snr

        return snr

    def calculate_sigma(self, signal_power, snr):
        noise_power = (signal_power / (10 ** (snr / 10)))
        std = torch.sqrt(noise_power)

        return std

    def apply_noise(self, x, signal_power, snr):
        if isinstance(snr, torch.Tensor):
            while len(snr.shape) < len(x.shape):
                snr = snr[..., None]

        noise_power = (signal_power / (10 ** (snr / 10)))
        std = torch.sqrt(noise_power)

        noise = torch.randn_like(x) * std

        return x + noise

    def forward(self, x: torch.Tensor, snr=None, **kwargs):

        self.symbols = x.shape
        self.channel_input = x

        if self.training and not self.use_training_snr:
            return x

        # if self.snr is None:
        #     return x

        if snr is None:
            snr = self.get_snr(len(x), x.device)
            # self.snr = snr
        elif isinstance(snr, torch.Tensor):
            snr = snr.to(x.device)

        if snr is None:
            return x

        signal_power = torch.linalg.norm(x, ord=2, dim=self.dims, keepdim=True)
        size = math.prod([x.size(dim=d) for d in self.dims]) if isinstance(self.dims, Sequence) else x.size(dim=self.dims)
        signal_power = signal_power / size

        # while len(snr.shape) < len(signal_power.shape):
        #     snr = snr[:, None]

        # noise_power = (signal_power / (10 ** (snr / 10)))
        # std = self.calculate_sigma(signal_power, snr)
        #
        # noise = torch.randn_like(x) * std
        #
        # x = x + noise

        return self.apply_noise(x, signal_power, snr)

    @property
    def test_snr(self):
        return self._snr

    @test_snr.setter
    def test_snr(self, v: Union[float, str]):
        if isinstance(v, str):
            assert v == 'random'
            self._test_snr = None
        else:
            self._test_snr = v


class FadingGaussianNoiseChannel(GaussianNoiseChannel):
    def __init__(self,
                 snr: Union[float, Tuple[float, float]],
                 fading_sigma: float,
                 dims=-1,
                 *args, **kwargs):
        super().__init__(snr=snr, dims=dims, *args, **kwargs)

        self._fading_sigma = fading_sigma

    def apply_noise(self, x, signal_power, snr):
        noise_power = (((self._fading_sigma ** 2) * signal_power) / (10 ** (snr / 10)))
        std = torch.sqrt(noise_power)

        noise = torch.randn_like(x) * std
        h = torch.randn_like(signal_power) * self._fading_sigma

        return x * h + noise


# class DigitalNoiseChannel(CleanChannel):
#     def __init__(self, p: float, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.p = p
#
#     def forward(self, x):
#         if isinstance(self.p, Sequence):
#             a, b = self.p
#             p = (a - b) * torch.rand_like(x) + b
#         else:
#             p = torch.full_like(x, self.p)
#
#         b = torch.bernoulli(p)
#         b = b.float()
#
#         return x * b
#
#     def set_noise(self, v):
#         self.p = v


# class MeasurableChannelWrapper(nn.Module):
#     def __init__(self, channel: CleanChannel, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         assert isinstance(channel, (GaussianNoiseChannel))
#
#         self._channel = channel
#         self._snr_set = None
#
#     def measure_snr(self, size: int):
#         snr = self._channel.get_snr(size)
#         self._snr_set = snr
#         return snr
#
#     def __getattr__(self, item):
#         try:
#             return super().__getattr__(item)
#         except AttributeError:
#             return getattr(self._channel, item)
#
#     def forward(self, x):
#         assert self._snr_set is not None and len(x) == len(self._snr_set)
#         # self._channel.snr = self._snr_set.to(x.device)
#
#         r = self._channel(x, snr=self._snr_set)
#         self._snr_set = None
#
#         return r
#
#
# if __name__ == '__main__':
#     channel = GaussianNoiseChannel(snr=(0, 10))
#     measurable_channel = MeasurableChannelWrapper(channel)
#
#     print(measurable_channel.measure_snr(100))
#
#     measurable_channel(torch.randn(100, 1))
