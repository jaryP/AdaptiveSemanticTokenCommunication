from copy import deepcopy

import numpy
import numpy as np
import torch
from torch import nn


def get_layers(input_size, output_size=1.0, n_layers=2, n_copy=1, invert=False, drop_last_activation=False):
    if isinstance(output_size, float):
        output_size = max(int(input_size * output_size), 1)

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


def get_cnn_layers(input_size, output_size=1.0, n_layers=2, n_copy=1, invert=False, drop_last_activation=True):
    def get_sizes(os, ins, inverse=False):
        mn = (np.inf, -1, -1)
        for i in range(1, 12):
            for j in range(1, 12):

                if not inverse:
                    s = np.floor((ins - (i - 1) - 1) / j + 1)
                else:
                    s = np.floor(((ins - 1) * j + (i - 1))) + 1

                d = abs(s - os)

                if d < mn[0]:
                    mn = (d, i, j, s)

        return mn[1:]

    input_channels, w, h = input_size
    out_channels = input_channels

    if isinstance(output_size, tuple):
        out_channels = output_size[0]
        output_size = output_size[1]

    if isinstance(output_size, float):
        output_size = max(int(output_size * w), 1)

    shapes = numpy.stack((np.linspace(w, output_size, num=n_layers + 1, endpoint=True, dtype=int),
                          np.linspace(input_channels, out_channels, num=n_layers + 1, endpoint=True, dtype=int)), 1)

    model = []
    os = None

    for s in range(len(shapes) - 1):

        if shapes[s + 1][0] < shapes[s][0]:
            ks, stride, size = get_sizes(shapes[s + 1][0], shapes[s][0])
            model.append(nn.Conv2d(shapes[s][1], shapes[s+1][1], kernel_size=ks, stride=stride))
            model.append(nn.ReLU())
        else:
            ks, stride, size = get_sizes(shapes[s + 1][0], shapes[s][0], inverse=True)
            model.append(nn.ConvTranspose2d(shapes[s][1], shapes[s+1][1], kernel_size=ks, stride=stride))
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

        if isinstance(input_size, tuple):
            output_size, (cc, rr) = get_cnn_layers(input_size=input_size, output_size=output_size,
                                                   n_layers=n_layers, drop_last_activation=drop_last_activation,
                                                   n_copy=2, invert=False)
        else:
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

        new_x = torch.complex(a, b)

        if self.normalize:
            new_x = new_x / torch.norm(new_x, 2, -1, keepdim=True)

        if len(x) > 0:
            zeros_mask = (x.sum(-1, keepdims=True) != 0).float()
            new_x = new_x * zeros_mask

        return new_x


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

        new_x = self.d_f(x.abs())

        if len(x) > 0:
            zeros_mask = (x.sum(-1, keepdims=True) != 0).float()
            new_x = new_x * zeros_mask

        if self.transpose:
            new_x = new_x.permute(0, 2, 1)

        return new_x


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

        # out_shape, (cc,) = get_layers(input_size=input_size * 2, output_size=output_size,
        #                               n_layers=n_layers, drop_last_activation=drop_last_activation,
        #                               n_copy=1, invert=False)

        if isinstance(input_size, tuple):
            self.cdim = 1
            input_size = (input_size[0]*2, input_size[1], input_size[2])
            output_size, (cc, _) = get_cnn_layers(input_size=input_size, output_size=output_size[:2],
                                                  n_layers=n_layers, drop_last_activation=drop_last_activation,
                                                  n_copy=2, invert=False)
        else:
            self.cdim = -1
            output_size, (cc, _) = get_layers(input_size=input_size * 2, output_size=output_size,
                                              n_layers=n_layers, drop_last_activation=drop_last_activation,
                                              n_copy=2, invert=False)

        self.d_f = cc
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x=None, *args, **kwargs):
        if self.normalize:
            x = x / torch.norm(x, 2, -1, keepdim=True)

        new_x = torch.cat((x.real, x.imag), self.cdim)
        new_x = self.d_f(new_x)

        if len(x) > 0:
            zeros_mask = (x.sum(-1, keepdims=True) != 0).float()
            new_x = new_x * zeros_mask

        if self.transpose:
            new_x = new_x.permute(0, 2, 1)

        return new_x


if __name__ == '__main__':
    _, (encoder, decoder) = get_cnn_layers((24, 32, 32), n_layers=3, output_size=0.5)
    x = torch.rand(1, 24, 32, 32)
    print(encoder(x).shape)
    print(decoder(encoder(x)).shape)
