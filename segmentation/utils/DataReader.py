import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import cv2
from albumentations import (Compose, HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, Resize, OneOf)


class DataReader:
    def __init__(self, slice_size, is_nii=True):
        self.height = slice_size[0]
        self.width  = slice_size[0]

        self.is_nii = is_nii

        self.resize = Resize(height=self.height, width=self.width, interpolation=cv2.INTER_CUBIC)
        self.aug = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            OneOf([
                ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                GridDistortion(p=0.5),
            ], p=0.5)
        ])

    @staticmethod
    def normalization_array(array):
        mi = array.min()
        if mi < 0:
            return (((array - mi) / array.max()) * 255).astype(np.uint8)

        return ((array / array.max()) * 255).astype(np.uint8)

    @staticmethod
    def read_nii(path):
        img = nib.load(path)
        img = nib.as_closest_canonical(img)
        data = img.get_fdata()

        return DataReader.normalization_array(data)

    def read_patient_nii(self, path_orig, path_mask, count_aug):
        orig, mask = DataReader.read_nii(path_orig), DataReader.read_nii(path_mask)

        origs, masks = [], []

        for i in range(90, orig.shape[2] - 30):
            img, img_mask = orig[:, :, i].T, mask[:, :, i].T
            resized = self.resize(image=img, mask=img_mask)

            orig_aug, mask_aug = self.augmentation(resized['image'], resized['mask'], count_aug)
            origs += orig_aug
            masks += mask_aug

        return DataReader.reshape(origs), DataReader.reshape(masks)

    @staticmethod
    def reshape(images):
        shape = (len(images), images[0].shape[0], images[0].shape[1], 1)
        images = np.array(images).astype(np.uint8)
        images.reshape(shape)

        return images

    def augmentation(self, orig, mask, count_aug):
        origs, masks = [orig], [mask]

        for i in range(count_aug):
            augmented = self.aug(image=orig, mask=mask)
            origs.append(augmented['image'])
            masks.append(augmented['mask'])

        return origs, masks

    def read_patient_mhd(self, path_orig, path_mask, count_aug):
        orig, mask = DataReader.read_mhd(path_orig)[0], DataReader.read_mhd(path_mask)[0]

        mask[np.nonzero(mask)] = 255

        origs, masks = [], []

        for i in range(10, orig.shape[2] - 30):
            img, img_mask = orig[:, :, i].T, mask[:, :, i].T
            resized = self.resize(image=img, mask=img_mask)

            orig_aug, mask_aug = self.augmentation(resized['image'], resized['mask'], count_aug)
            origs += orig_aug
            masks += mask_aug

        return DataReader.reshape(origs), DataReader.reshape(masks)

    @staticmethod
    def read_mhd(filename):
        """This funciton reads a '.mhd' file using SimpleITK
            and return the image array, origin and spacing of the image.
            """
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)

        # Convert the image to a  numpy array first and then
        # shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)

        # Read the origin of the ct_scan, will be used to convert
        # the coordinates from world to voxel and vice versa.
        origin = np.array(list(reversed(itkimage.GetOrigin())))

        # Read the spacing along each dimension
        spacing = np.array(list(reversed(itkimage.GetSpacing())))

        return DataReader.normalization_array(ct_scan), origin, spacing

    def save_to_npy(self, path_to_save, patient_id, path_orig, path_mask, count_aug):
        if self.is_nii:
            orig, mask = self.read_patient_nii(path_orig, path_mask, count_aug)
        else:
            orig, mask = self.read_patient_mhd(path_orig, path_mask, count_aug)

        patient_path = os.path.join(path_to_save, patient_id)
        os.makedirs(patient_path, exist_ok=True)

        for i in range(len(orig)):
            np.save(os.path.join(patient_path, "IM%s" % i), np.array([orig[i], mask[i]]))
