import os
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
