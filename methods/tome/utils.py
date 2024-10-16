import math
import os
from functools import lru_cache
from typing import Callable, Tuple

import numpy as np
import timm
import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T


def calculate_cumulative_percentage(v):
    p = 1
    for _v in v:
        p = p * _v
    return p



def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


@lru_cache(maxsize=None)
def get_merging_percentage(n, target_p, inverse=False):
    i = 0
    p = np.zeros(n)

    best_p = np.zeros(n)
    best_d = np.inf

    while i < n:
        for j in np.linspace(0, 0.5, endpoint=True, num=20):
            p[i] = j
            tp = math.prod((item for item in p if item > 0))

            if np.abs(tp - target_p) < best_d:
                best_p = np.copy(p)
                best_d = np.abs(tp - target_p)

        i += 1

    if inverse:
        best_p = best_p[::-1]

    return best_p


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


def get_pretrained_model(device):
    test_transform = T.Compose([
        T.Resize((248, 248), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        T.CenterCrop((224, 224)),
        # T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_transform = T.Compose([
        T.RandAugment(num_ops=2, magnitude=9),
        T.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0),
        T.Resize((248, 248), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        T.Resize((224, 224)),
        # T.RandomCrop((224, 224), padding=12),
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.Imagenette(root='./data/imagenette',
                                               split='train',
                                               download=False, transform=train_transform)
    testset = torchvision.datasets.Imagenette(root='./data/imagenette',
                                              split='val',
                                              download=False, transform=test_transform)
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
