import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from segmentation.model.zf_unet_224_model import ZF_UNET_224
from segmentation.model.Unet224Torch import Unet224Torch
from segmentation.eval.mask_helper import fixmask


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class TestNet:
    def __init__(self, weight_path, tester_name, need_fix_mask, device):
        self.weight_path = weight_path
        self.tester_name = tester_name
        self.need_fix_mask = need_fix_mask
        self.device = device

        self.model = Unet224Torch(1)#(weights_path=weight_path)

        self.__setting_model()

    def __setting_model(self):
        self.model.load_state_dict(torch.load(self.weight_path,
                                   map_location=self.device).module.state_dict())
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f'{torch.cuda.device_count()} GPUs used')
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
        self.model.eval()

    def apply_mask(self, origs, masks):
        result = []

        for i in range(len(origs)):
            mask = masks[i, 0, :, :].copy()
            mask = mask.astype(np.uint8)
            # if self.need_fix_mask:
            #     mask = fixmask(mask)

            orig = origs[i][0].copy()
            orig[mask == 0] = 0

            # plt.imshow(orig, cmap="gray")
            # plt.show()
            # plt.imshow(mask, cmap="gray")
            # plt.show()

            result.append(mask)

        return np.array(result)

    def predict(self, x):
        preds = []
        xs = list(chunks(x, 15))
        print(self.tester_name, " is working")
        with torch.no_grad():
            for i in xs:
                i = torch.from_numpy(i)
                idev = i.to(self.device).float()
                pred = self.model(idev)
                pred = torch.sigmoid(pred)

                pred = np.array(pred.cpu())
                pred = (pred > 0.3).astype(np.uint8)
                pred *= 255

                preds.extend(pred)

                del idev

        return self.apply_mask(x, np.array(preds))
