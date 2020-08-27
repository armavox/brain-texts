import numpy as np
import torch


def get_split_legths(ds, ratio=(0.7, 0.2, 0.1)):
    assert np.allclose(sum(ratio), 1)
    train_len = int(len(ds) * ratio[0])
    val_len = int((len(ds) - train_len) * ratio[1] / sum(ratio[1:]))
    test_len = len(ds) - train_len - val_len
    return train_len, val_len, test_len


def random_seed_init(random_seed: bool = None, cuda: bool = False):
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
