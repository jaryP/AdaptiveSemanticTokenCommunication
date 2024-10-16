import math
import os
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


    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test)

    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params=model.parameters())

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

    train_dataloader = datalaoder_wrapper(dataset=train_dataset, batch_size=cfg.schema.train_batch_size)

    bar = tqdm.tqdm(range(cfg.schema.epochs),
                    leave=False,
                    desc='Pre training model')

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

    print(c, t, c / t)

    return model

    BATCH_SIZE = 128
    EPOCHS = 50

    # device = '0'
    # if torch.cuda.is_available() and device != 'cpu':
    #     device = 'cuda:{}'.format(device)
    #     torch.cuda.set_device(device)
    # else:
    #     device = 'cpu'
    #
    # device = torch.device(device)

    vit = timm.create_model(
        'deit_tiny_patch16_224.fb_in1k',
        pretrained=True,
        num_classes=10).to(device)

    opt = Adam(vit.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    results_path = './results/'

    model_path = os.path.join(results_path, f'pretrained_models/spawc.pt')

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path):
        vit.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    else:
        for _ in range(EPOCHS):
            vit.train()

            for x, y in DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True):
                x, y = x.to(device), y.to(device)

                pred = vit(x)
                loss = nn.functional.cross_entropy(pred, y)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vit.parameters(), 1)

                opt.step()

            scheduler.step()

            vit.eval()
            c, t = 0, 0
            for x, y in DataLoader(testset, batch_size=BATCH_SIZE):
                x, y = x.to(device), y.to(device)

                pred = vit(x)

                c += (pred.argmax(-1) == y).sum().item()
                t += len(x)

            print(c, t, c / t)

        torch.save(vit.state_dict(), model_path)

    return vit


def get_encoder_decoder(input_size, compression, n_layers=2):

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

    return nn.Sequential(*encoder), nn.Sequential(*decoder)
