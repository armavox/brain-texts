from segmentation.model.zf_unet_224_model import ZF_UNET_224
import numpy as np
from segmentation.eval.mask_helper import fixmask_0


class TestNet:
    def __init__(self, weight_path, tester_name):
        self.weight_path = weight_path
        self.tester_name = tester_name
        self.model = ZF_UNET_224(weights_path=weight_path)

    @staticmethod
    def apply_mask(origs, masks):
        result = []

        for i in range(len(origs)):
            mask = masks[i,:,:,0].copy()
            mask = mask / (mask.max() / 255)
            mask = mask.astype(np.uint8)
            mask = fixmask_0(mask)

            orig = origs[i].copy()
            orig[mask == 0] = (0,0,0)

            result.append(orig)

        return np.array(result)

    def predict(self, x):
        print(self.tester_name, " is working")
        pred = self.model.predict(x)

        return TestNet.apply_mask(x, pred)



