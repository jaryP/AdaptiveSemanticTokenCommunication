from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm.auto as tqdm

from methods.flops_count import compute_flops
from methods.proposal import AdaptiveBlock, SemanticVit


@torch.no_grad()
def semantic_evaluation(model: SemanticVit,
                        dataset,
                        budgets=None,
                        calculate_flops=True,
                        batch_size=1,
                        # full_flops=None,
                        **kwargs):

    device = next(model.parameters()).device
    model.eval()

    accuracy = {}
    flops = {}
    all_sizes = {}

    if budgets is None:
        budgets = [0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9]

    full_flops = None

    for a in tqdm.tqdm(budgets, leave=False):

        c, t = 0, 0
        average_dropping = defaultdict(list)
        a_flops = []

        for x, y in DataLoader(dataset, batch_size=batch_size):
            x, y = x.to(device), y.to(device)

            if full_flops is None and calculate_flops:
                full_flops = compute_flops(model, x, verbose=False, print_per_layer_stat=False)[0]

            pred = model(x, alpha=a)

            if full_flops is not None:
                model.current_alpha = a
                # a_flops.append(compute_flops(model, x, verbose=False, print_per_layer_stat=False)[0] / full_flops)
                a_flops.append(compute_flops(model, x, verbose=False, print_per_layer_stat=False)[0])
                model.current_alpha = None

            c += (pred.argmax(-1) == y).sum().item()
            t += len(x)

            for i, b in enumerate([b for b in model.blocks if isinstance(b, AdaptiveBlock) if b.last_mask is not None]):
                if b.last_mask.shape[1] < b.num_patches:
                    average_dropping[i].append(b.last_mask.shape[1])
                else:
                    average_dropping[i].extend(b.last_mask.mean(1).cpu().detach().tolist())

        accuracy[a] = c / t

        all_sizes[a] = {k: (np.mean(v), np.std(v)) for k, v in average_dropping.items()}

        if len(a_flops) > 0:
            flops[a] = (np.mean(a_flops), np.std(a_flops))

    if len(flops) > 0:
        return {'accuracy': accuracy, 'full_flops': full_flops, 'flops':  flops, 'all_sizes':  all_sizes}
    else:
        return {'accuracy': accuracy, 'all_sizes':  all_sizes}
