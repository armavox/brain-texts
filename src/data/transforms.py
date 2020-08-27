from abc import ABC, abstractmethod
import skimage


class AbstractTransform(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, sample):
        raise NotImplementedError
        image = sample.pop("image") if "image" in sample else None
        if image is not None:
            # perform some actions with the image
            pass
        return {"image": image, **sample}


class Resize:
    """Rescale the image in a sample to a given size.

    Parameters
    ----------
    output_size : tuple, int
        Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample: dict):
        image = sample.pop("image") if "image" in sample else None
        if image is not None:
            d, h, w = image.shape
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = skimage.transform.resize(image, (d, new_h, new_w), preserve_range=True)

        return {'image': img.transpose(1, 2, 0), **sample}
