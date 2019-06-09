import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from segmentation.eval.TestNet import TestNet
import argparse
import data_utils
import cv2
import glob
import numpy as np
import os


from segmentation.utils.DataReader import DataReader
import SimpleITK as sitk


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="Path to folder with patients")
    parser.add_argument("-wb", "--weight_brain", type=str,
                        help="Path to weight file for segmentation brain")
    parser.add_argument("-wg", "--weight_gliom", type=str,
                        help="Path to weight file for segmentation gliomas")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to weight file")
    return parser.parse_args()


def get_orig_mask_filenames_from_patient_directory(patient_path):
    mask_filename = ""
    orig_filename = ""

    for i in glob.glob1(patient_path, "*.mhd"):
        lower_i = i.lower()

        if "label" in lower_i:
            mask_filename = os.path.join(patient_path, i)
        if "flair" in lower_i:
            orig_filename = os.path.join(patient_path, i)

    return orig_filename, mask_filename

def load_patient(general_path, patient_id):
    patient_path = os.path.join(general_path, patient_id)
    orig_filename, mask_filename = get_orig_mask_filenames_from_patient_directory(patient_path)
    # x = data_utils.stack_images(general_path, patient_id, False)
    # x = data_utils.load_mgh(general_path, patient_id)
    print(orig_filename, mask_filename)
    x = DataReader.read_mhd(orig_filename)[0]
    y = DataReader.read_mhd(mask_filename)[0]
    plt.imshow(x[:, :, 100])
    plt.show()
    print(y.shape)

    shape = list(x.shape)
    shape[1] = shape[0] = 224

    if len(x.shape) == 3:
        xx = np.empty((shape[2], shape[1], shape[0], 3), dtype = np.uint8)
        for i in range(shape[2]):
            img = x[:, :, i].copy()
            mi = img.min()
            if mi < 0:
                img = (((img - mi) / img.max()) * 255).astype(np.uint8)
            else:
                img = ((img / img.max()) * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (224, 224))
            xx[i,...] = img
        x = xx

    return x

import matplotlib.pyplot as plt

def main(opt):
    input_path = "/data/brain/rs-mhd-dataset"#opt.input
    weight_brain = "/home/armavox/pyprojects/brain-texts/segmentation/weights/weights-43.hdf5" #opt.weight_brain
    # weight_gliom = opt.weight_gliom
    # output_path = opt.output

    patients_id = glob.glob1(input_path, "AR*")

    brain_segm = TestNet(weight_brain, "Brain segmentation", True)
    # gliom_segm = TestNet(weight_gliom, "Gliomas segmentation", False)

    for patient_id in patients_id:
        if patient_id == "AR-5":
            continue
        patient_saving_path = os.path.join(input_path, patient_id, "%s_rs_mask.mhd" % patient_id)

        x = load_patient(input_path, patient_id)
        x = brain_segm.predict(x)
        sitk.WriteImage(sitk.GetImageFromArray(x), patient_saving_path)

        # np.save(os.path.join(output_path, "brain_%s.npy" % patient_id), x)
        # x = gliom_segm.predict(x)

        # np.save(patient_saving_path, x)
        print("Patient's gliomas saved to: ", patient_saving_path)


if __name__ == '__main__':
    opt = arguments()
    main(opt)
