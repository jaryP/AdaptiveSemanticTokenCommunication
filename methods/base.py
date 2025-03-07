import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from comm.channel import GaussianNoiseChannel


@torch.no_grad()
def evaluation(model: nn.Module,
               dataset,
               batch_size=1,
               return_predictions=False,
               **kwargs):

    modules = [m for m in model.modules() if isinstance(m, GaussianNoiseChannel)]
    symbols = None

    device = next(model.parameters()).device
    model.eval()

    c, t = 0, 0

    preds = []
    ys = []

    for x, y in DataLoader(dataset, batch_size=batch_size):
        ys.extend(y.tolist())

        x, y = x.to(device), y.to(device)

        pred = model(x)

        c += (pred.argmax(-1) == y).sum().item()
        t += len(x)

        preds.append(pred.detach().cpu().numpy())

        if len(modules) > 0:
            symbols = np.prod(modules[0].symbols[1:])

    accuracy = c / t

    dd = {'accuracy': accuracy}

    if symbols is not None:
        dd.update({'symbols': symbols})

    if return_predictions:
        dd.update({'predictions': (np.concatenate(preds), ys)})

    return dd
