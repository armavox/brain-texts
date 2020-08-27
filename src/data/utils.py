import os
import numpy as np
import torch
import torchvision as tv
from torchvision.datasets.folder import has_file_allowed_extension


class DatasetFolder(tv.datasets.VisionDataset):
    """Modified torchbision DatasetFolder to work with albumentations"""

    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.__make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + self.root + "\n"
                    "Supported extensions are: " + ",".join(extensions)
                )
            )

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):

        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            if callable(self.transform):
                image = sample.pop("image").transpose(1, 2, 0)
                image = self.transform(image=image)["image"].transpose(2, 0, 1)
                sample = {"image": image, **sample}
            else:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:

            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        return instances


class PadCollate:
    """ A variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0, tensor_name="report", dataset_folder_used=False):
        """
        args:
            dim - the dimension to be padded (dimension of sequence in tensor)
        """
        self.dim = dim
        self.tensor_name = tensor_name
        self.dataset_folder_used = dataset_folder_used

    def __call__(self, batch):
        return self.pad_collate(batch)

    def pad_collate(self, batch):
        if self.dataset_folder_used:
            labels = [x[1] for x in batch]
            batch = [x[0] for x in batch]
        # report_seq = [torch.tensor(sample['report']) for sample in batch]
        # torch.nn.utils.rnn.pad_sequence(seq, batch_first=True)

        # find longest sequence and pad according to max_len
        max_len = max(map(lambda x: x[self.tensor_name].shape[self.dim], batch))
        for sample in batch:
            sample[self.tensor_name] = self._pad_tensor(sample[self.tensor_name], pad=max_len, dim=self.dim)

        # stack all
        _batch = dict()
        for key in batch[0].keys():
            key_stack = tuple(map(lambda x: torch.tensor(x[key]), batch))
            _batch[key] = torch.stack(key_stack, dim=0)

        if self.dataset_folder_used:
            return [_batch, torch.tensor(labels)]
        return _batch

    @staticmethod
    def _pad_tensor(vec, pad, dim):
        """
        args:
            vec - tensor to pad
            pad - the size to pad to
            dim - dimension to pad

        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """

        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.shape[dim]
        if isinstance(vec, torch.Tensor):
            return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
        elif isinstance(vec, np.ndarray):
            return np.concatenate([vec, np.zeros(pad_size, dtype=vec.dtype)], axis=dim)
        else:
            raise TypeError("vec should be instance of [np.ndarray, torch.Tensor]")
