from torch.utils.data import Dataset
import numpy as np


class ARDataset(Dataset):
    def __init__(self, files_path):
        self.files_path = files_path

    def __getitem__(self, index):
        filepath = self.files_path[index]
        data = np.load(filepath)

        return data[0], data[1]

    def __len__(self):
        return len(self.files_path)
