import torch
from torch.utils.data import DataLoader

from methods.proposal import SemanticVit


@torch.no_grad()
def evaluation(model: SemanticVit,
               dataset,
               **kwargs):

    device = next(model.parameters()).device
    model.eval()

    c, t = 0, 0

    for x, y in DataLoader(dataset, batch_size=1):
        x, y = x.to(device), y.to(device)

        pred = model(x)

        c += (pred.argmax(-1) == y).sum().item()
        t += len(x)

    accuracy = c / t

    return {'accuracy': accuracy}
