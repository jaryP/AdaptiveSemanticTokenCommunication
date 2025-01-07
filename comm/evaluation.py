import math
from copy import deepcopy
from io import BytesIO
from typing import Sequence
import logging

import numpy as np
import torch
import torchvision.transforms.functional
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
import tqdm.auto as tqdm

from comm.channel import GaussianNoiseChannel
from methods.proposal import SemanticVit


@torch.no_grad()
def gaussian_snr_evaluation(model: SemanticVit,
                            dataset,
                            snr,
                            function: None,
                            monte_carlo_n=1,
                            **kwargs):
    if function is None:
        function = lambda *x: x

    model.eval()

    modules = [m for m in model.modules() if isinstance(m, GaussianNoiseChannel)]
    assert len(modules) == 1

    channel = modules[0]

    if not isinstance(snr, Sequence):
        snr = [snr]

    results = {}

    for n in tqdm.tqdm(range(monte_carlo_n), leave=False):
        _results = {}
        for _snr in tqdm.tqdm(snr, leave=False):
            channel.test_snr = _snr
            _results[_snr] = function(model=model, dataset=dataset, **kwargs)

        results[n] = _results

    return results


@torch.no_grad()
def digital_jpeg(model, dataset, kn, snr, base=10, batch_size=32,
                 previous_results=None):
    class ToJpegAnalogical(torch.nn.Module):

        def __init__(self, max_bits):
            super().__init__()
            self.byts = []
            self.max_bits = max_bits

        def forward(self, img):
            w, h = img.size

            buffer = BytesIO()
            img.save(buffer, "JPEG", quality=1)

            bytes = (buffer.tell() * 8) / (w * h * 3)

            if bytes > self.max_bits:
                self.byts.append(-bytes)
                return img
            else:
                for v in range(95, 0, -1):
                    buffer = BytesIO()
                    img.save(buffer, "JPEG", quality=v)
                    bytes = (buffer.tell() * 8) / (w * h * 3)
                    if bytes < self.max_bits:
                        self.byts.append(bytes)
                        return Image.open(buffer)

            return img

    model.eval()
    device = next(model.parameters()).device

    base_transforms = deepcopy(dataset.transform)
    rr = None
    for t in base_transforms.transforms:
        if isinstance(t, Resize):
            rr = t

    results = {}
    if previous_results is not None and isinstance(previous_results, dict):
        results = previous_results

    for _snr in tqdm.tqdm(snr, leave=False):

        if _snr not in results:
            results[_snr] = {}

        for _kn in kn:
            if _kn in results[_snr]:
                continue

            rmax = _kn * np.log2(1 + (base ** (_snr / base)))

            to_jpeg = ToJpegAnalogical(rmax)
            if rr is not None:
                dataset.transform = Compose([rr, to_jpeg, base_transforms])
            else:
                dataset.transform = Compose([to_jpeg, base_transforms])

            tot = 0
            cor = 0

            for x, y in DataLoader(dataset, batch_size=batch_size):
                x, y = x.to(device), y.to(device)

                pred = model(x).argmax(-1)

                correct = pred.eq(y.view_as(pred))
                bits = np.asarray(to_jpeg.byts)

                correct[bits < 0] = False
                tot += len(x)
                cor += correct.sum().item()

                to_jpeg.byts = []

            results[_snr][_kn] = cor / tot

    dataset.transform = base_transforms

    return results


@torch.no_grad()
def digital_resize(model,
                   dataset,
                   kn,
                   snr,
                   base=10,
                   batch_size=32,
                   previous_results=None):
    model.eval()
    device = next(model.parameters()).device

    base_transforms = deepcopy(dataset.transform)

    results = {}
    if previous_results is not None and isinstance(previous_results, dict):
        results = previous_results

    x = dataset[0][0]
    if isinstance(x, torch.Tensor):
        shape = x.shape[1:]
    elif isinstance(x, Image.Image):
        shape = x.size
    else:
        shape = x.shape[:-1]

    for _snr in tqdm.tqdm(snr, leave=False):

        if _snr not in results:
            results[_snr] = {}

        for _kn in kn:
            if _kn in results[_snr]:
                continue

            L = _kn * np.log2(1 + (base ** (_snr / base)))
            L = np.sqrt(L * np.prod(shape) / 8)
            L = np.rint(L)

            if L < 1:
                results[_snr][_kn] = 0
                continue

            resize = Resize((L, L))
            dataset.transform = Compose([resize, base_transforms])

            tot = 0
            cor = 0

            for x, y in DataLoader(dataset, batch_size=batch_size):
                x, y = x.to(device), y.to(device)

                pred = model(x).argmax(-1)

                correct = pred.eq(y.view_as(pred))

                tot += len(x)
                cor += correct.sum().item()

            results[_snr][_kn] = cor / tot

    dataset.transform = base_transforms

    return results


@torch.no_grad()
def analog_resize(model,
                  dataset,
                  kn,
                  snr,
                  base=10,
                  batch_size=32,
                  previous_results=None):
    # TODO: TEST
    class AnalogNoiseResized(torch.nn.Module):

        def __init__(self, compressions, snr):
            super().__init__()
            self.compressions = compressions
            self.snr = snr

        def forward(self, img):
            w, h = img.size

            img = torchvision.transforms.functional.to_tensor(img)
            img = torchvision.transforms.functional.resize(img, [int(np.rint(w * self.compressions)),
                                                                 int(np.rint(w * self.compressions))])

            signal_power = torch.linalg.norm(x.view(-1), ord=2, dim=None, keepdim=True)
            size = img.numel()
            signal_power = signal_power / size
            noise_power = (signal_power / (10 ** (self.snr / 10)))
            std = torch.sqrt(noise_power)

            noise = torch.randn_like(img) * std
            img = img + noise

            img = torchvision.transforms.functional.resize(img, [w, h])

            return torchvision.transforms.functional.to_pil_image(img)

    model.eval()
    device = next(model.parameters()).device

    base_transforms = deepcopy(dataset.transform)

    results = {}
    if previous_results is not None and isinstance(previous_results, dict):
        results = previous_results

    x = dataset[0][0]
    if isinstance(x, torch.Tensor):
        shape = x.shape[1:]
    elif isinstance(x, Image.Image):
        shape = x.size
    else:
        shape = x.shape[:-1]

    for _snr in tqdm.tqdm(snr, leave=False):

        if _snr not in results:
            results[_snr] = {}

        for _kn in kn:
            if _kn in results[_snr]:
                continue

            if np.rint(shape[0] * _kn) < 1:
                results[_snr][_kn] = 0
                continue

            dataset.transform = Compose([AnalogNoiseResized(_kn, _snr), base_transforms])

            tot = 0
            cor = 0

            for x, y in DataLoader(dataset, batch_size=batch_size):
                x, y = x.to(device), y.to(device)

                pred = model(x).argmax(-1)

                correct = pred.eq(y.view_as(pred))

                tot += len(x)
                cor += correct.sum().item()

            results[_snr][_kn] = cor / tot

    dataset.transform = base_transforms

    return results
