import numpy as np


def get_split_legths(ds, ratio=(0.7, 0.2, 0.1)):
    assert np.allclose(sum(ratio), 1)
    train_len = int(len(ds) * ratio[0])
    val_len = int((len(ds) - train_len) * ratio[1] / sum(ratio[1:]))
    test_len = len(ds) - train_len - val_len
    return train_len, val_len, test_len
