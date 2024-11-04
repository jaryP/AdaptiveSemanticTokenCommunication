from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from flops_count import compute_flops
from methods.proposal import AdaptiveBlock, SemanticVit


@torch.no_grad()
def semantic_evaluation(model: SemanticVit, dataset, budgets=None):
    device = next(model.parameters()).device
    model.eval()

    accuracy = {}
    flops = {}
    all_sizes = {}

    if budgets is None:
        budgets = [0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]

    full_flops = None

    for a in budgets:

        c, t = 0, 0
        average_dropping = defaultdict(float)
        a_flops = []

        for x, y in DataLoader(dataset, batch_size=1):
            x, y = x.to(device), y.to(device)

            if full_flops is None:
                full_flops = compute_flops(model, x, verbose=False, print_per_layer_stat=False)[0]

            model.current_alpha = a
            pred = model(x, alpha=a)
            a_flops.append(compute_flops(model, x, verbose=False, print_per_layer_stat=False)[0] / full_flops)
            model.current_alpha = None

            c += (pred.argmax(-1) == y).sum().item()
            t += len(x)

            for i, b in enumerate([b for b in model.blocks if isinstance(b, AdaptiveBlock) if b.last_mask is not None]):
                average_dropping[i] += b.last_mask.shape[1]

        accuracy[a] = c / t
        flops[a] = np.mean(a_flops)
        all_sizes[a] = {k: v / t for k, v in average_dropping.items()}


    return accuracy, flops, all_sizes
