import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from comm.channel import GaussianNoiseChannel


@torch.no_grad()
def evaluation(model: nn.Module,
               dataset,
               **kwargs):

    modules = [m for m in model.modules() if isinstance(m, GaussianNoiseChannel)]
    symbols = None

    device = next(model.parameters()).device
    model.eval()

    c, t = 0, 0

    for x, y in DataLoader(dataset, batch_size=1):
        x, y = x.to(device), y.to(device)

        pred = model(x)

        c += (pred.argmax(-1) == y).sum().item()
        t += len(x)

        if len(modules) > 0:
            symbols = np.prod(modules[0].symbols[1:])

    accuracy = c / t

    if symbols is not None:
        return {'accuracy': accuracy, 'symbols': symbols}

    return {'accuracy': accuracy}
