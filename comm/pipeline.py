from copy import deepcopy

import numpy as np
import torch
from torch import nn


def get_layers(input_size, output_size=1.0, n_layers=2, n_copy=1, invert=False, drop_last_activation=False):
    if isinstance(output_size, float):
        output_size = int(input_size * output_size)

    shapes = np.linspace(input_size, output_size, num=n_layers + 1, endpoint=True, dtype=int)

    model = []

    for s in range(len(shapes) - 1):
        model.append(nn.Linear(shapes[s], shapes[s + 1]))
        model.append(nn.ReLU())

    if drop_last_activation:
        model = model[:-1]
    # if invert:
    #     model = model[::-1]

    model = nn.Sequential(*model)

    models = []
    for _ in range(n_copy):
        _model = deepcopy(model)

        for m in _model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        models.append(_model)

    return shapes[-1], models


class BaseRealToComplexNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_layers=2,
                 normalize=True,
                 transpose=False,
                 drop_last_activation=False,
                 sincos=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        output_size, (cc, rr) = get_layers(input_size=input_size, output_size=output_size,
                                           n_layers=n_layers, drop_last_activation=drop_last_activation,
                                           n_copy=2, invert=False)

        self.r_fl, self.c_fl = rr, cc

        self.normalize = normalize
        self.transpose = transpose
        self.sincos = sincos
        self.output_size = output_size

    def forward(self, x, *args, **kwargs):
        if self.transpose:
            x = x.permute(0, 2, 1)

        a, b = self.r_fl(x), self.c_fl(x)

        if self.sincos:
            a, b = torch.cos(a), torch.sin(b)

        x = torch.complex(a, b)

        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        return x


class ABSComplexToRealNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_layers=2,
                 normalize=False,
                 transpose=False,
                 drop_last_activation=True,
                 sincos=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        out_shape, (cc,) = get_layers(input_size=input_size, output_size=output_size,
                                      n_layers=n_layers, drop_last_activation=drop_last_activation,
                                      n_copy=1, invert=False)

        self.d_f = cc
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x=None, *args, **kwargs):
        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        x = self.d_f(x.abs())

        if self.transpose:
            x = x.permute(0, 2, 1)

        return x


class ConcatComplexToRealNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_layers=2,
                 normalize=False,
                 transpose=False,
                 drop_last_activation=True,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        out_shape, (cc,) = get_layers(input_size=input_size * 2, output_size=output_size,
                                      n_layers=n_layers, drop_last_activation=drop_last_activation,
                                      n_copy=1, invert=False)

        self.d_f = cc
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x=None, *args, **kwargs):
        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        x = torch.cat((x.real, x.imag), -1)
        x = self.d_f(x)

        if self.transpose:
            x = x.permute(0, 2, 1)

        return x
