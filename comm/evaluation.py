from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from comm.channel import GaussianNoiseChannel
from methods.flops_count import compute_flops
from methods.proposal import AdaptiveBlock, SemanticVit


@torch.no_grad()
def gaussian_snr_evaluation(model: SemanticVit,
                            dataset,
                            snr,
                            function,
                            **kwargs):

    model.eval()

    modules = [m for m in model.modules() if isinstance(m, GaussianNoiseChannel)]
    assert len(modules) == 1

    channel = modules[0]

    if not isinstance(snr, Sequence):
        snr = [snr]

    results = {}
    for _snr in tqdm(snr, leave=False):
        channel.test_snr = _snr
        _results = function(model=model, dataset=dataset, **kwargs)
        results[_snr] = _results

    return results
