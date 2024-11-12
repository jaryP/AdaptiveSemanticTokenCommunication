import math
import os
from copy import deepcopy
from functools import lru_cache
from typing import Callable, Tuple

import hydra
import numpy as np
import timm
import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
import tqdm.auto as tqdm


def calculate_cumulative_percentage(v):
    p = 1
    for _v in v:
        p = p * _v
    return p


def do_nothing(x, mode=None):
    return x


def get_merging_percentage1(n, target_p, epsilon=0.01):
    a = [np.linspace(0.0, 0.5, 10, endpoint=True) for _ in range(n)]
    results = np.asarray(np.meshgrid(*a)).T.reshape(-1, n)[1:]

    prod = np.prod(1 - results, 1)
    mask = np.abs(prod - target_p) < epsilon

    results = results[mask]

    return results

@lru_cache(maxsize=None)
def get_merging_percentage_torch(n, target_p, inverse=False):
    i = 0
    p = torch.zeros(n)

    best_p = torch.zeros(n)
    best_d = np.inf

    while i < n:
        for j in np.linspace(0, 0.5, endpoint=True, num=20):
            p[i] = j
            tp = math.prod((item for item in p if item > 0))

            if torch.abs(tp - target_p) < best_d:
                best_p = np.copy(p)
                best_d = np.abs(tp - target_p)

        i += 1

    if inverse:
        best_p = best_p[::-1]

    return best_p


def get_pretrained_model(cfg, model, device):

    to_download = not os.path.exists(cfg.dataset.train.root)

    train_dataset = hydra.utils.instantiate(cfg.dataset.train, download=to_download, _convert_="partial")
    test_dataset = hydra.utils.instantiate(cfg.dataset.test, _convert_="partial")

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    scheduler = None
    if 'scheduler' in cfg:
        scheduler = hydra.utils.instantiate(cfg.scheduler,
                                            optimizer=optimizer)

    datalaoder_wrapper = hydra.utils.instantiate(
        cfg.dataloader, _partial_=True)

    if 'dev_dataloader' in cfg.dataloader:
        test_dataloader = hydra.utils.instantiate(
            cfg.dataloader, dataset=test_dataset)
    else:
        test_dataloader = datalaoder_wrapper(dataset=test_dataset)

    train_dataloader = datalaoder_wrapper(dataset=train_dataset)

    bar = tqdm.tqdm(range(cfg.schema.epochs),
                    leave=False,
                    desc='Pre training model')

    model = model.to(device)
    for _ in bar:
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = nn.functional.cross_entropy(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            t, c = 0, 0

            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                c += (pred.argmax(-1) == y).sum().item()
                t += len(x)

        bar.set_postfix({'Test acc': c / t})

    return model


def get_encoder_decoder(input_size, compression=1, n_layers=2):

    compressions = np.linspace(1, compression, num=n_layers, endpoint=True)
    encoder = []
    decoder = []

    shape = input_size

    for c in compressions:
        out_shape = int(input_size * c)

        encoder.append(nn.Linear(shape, out_shape))
        encoder.append(nn.ReLU())

        decoder.append(nn.Linear(out_shape, shape))
        decoder.append(nn.ReLU())

        shape = out_shape

    decoder = decoder[-2::-1]
    encoder = encoder[:-1]

    encoder, decoder = nn.Sequential(*encoder), nn.Sequential(*decoder)

    encoder1 = deepcopy(encoder)
    for m in encoder1.modules():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    return encoder, encoder1, decoder


class BaseRealToComplex(nn.Module):
    def __init__(self, real_module, complex_module, normalize=True, transpose=False, sincos=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.r_fl, self.c_fl = real_module, complex_module

        self.normalize = normalize
        self.transpose = transpose
        self.sincos = sincos

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


class BaseComplexToReal(nn.Module):
    def __init__(self, decoder, transpose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.d_f = decoder
        self.transpose = transpose

    def forward(self, x=None, *args, **kwargs):
        x = self.d_f(x.abs())

        if self.transpose:
            x = x.permute(0, 2, 1)

        return x


class CommunicationPipeline(nn.Module):
    def __init__(self, encoder, channel, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.channel = channel

    def forward(self, x, snr=None):
        if not self.training:
            mask = (x.sum(dim=-1, keepdim=True) != 0).float()
            x = self.encoder(x)

            if self.channel is not None:
                x = self.channel(x * mask)

            decoded = self.decoder(x) * mask
        else:
            x = self.encoder(x)

            if self.channel is not None:
                x = self.channel(x)

            decoded = self.decoder(x)

        return decoded
