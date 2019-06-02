from segmentation.model.zf_unet_224_model import ZF_UNET_224
import numpy as np
from segmentation.eval.mask_helper import fixmask


class TestNet:
    def __init__(self, weight_path, tester_name, need_fix_mask):
        self.weight_path = weight_path
        self.tester_name = tester_name
        self.need_fix_mask = need_fix_mask
        self.model = ZF_UNET_224(weights_path=weight_path)

    def apply_mask(self, origs, masks):
        result = []

        for i in range(len(origs)):
            mask = masks[i,:,:,0].copy()
            mask = mask / (mask.max() / 255)
            mask = mask.astype(np.uint8)
            if self.need_fix_mask:
                mask = fixmask(mask)

            orig = origs[i].copy()
            orig[mask == 0] = (0,0,0)

            result.append(orig)

        return np.array(result)

    def predict(self, x):
        print(self.tester_name, " is working")
        pred = self.model.predict(x)

        return self.apply_mask(x, pred)



