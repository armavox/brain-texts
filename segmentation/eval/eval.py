import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from segmentation.eval.TestNet import TestNet
import argparse
import glob
import numpy as np
import os

import torch


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
        if "flair" in lower_i or 'kda' in lower_i:
            orig_filename = os.path.join(patient_path, i)

    return orig_filename, mask_filename


def load_patient(general_path, patient_id):
    patient_path = os.path.join(general_path, patient_id)

    orig_filename, mask_filename = get_orig_mask_filenames_from_patient_directory(patient_path)


    print('F', orig_filename)
    x = DataReader.read_mhd(orig_filename)[0]
    if 'norma' in orig_filename:
        x = x.transpose(1, 2, 0)
        print('NORMA')
    if mask_filename:
        y = DataReader.read_mhd(mask_filename)[0]

    shape = list(x.shape)

    if len(x.shape) == 3:
        xx = np.empty((shape[2], 1, shape[1], shape[0]), dtype = np.uint8)
        for i in range(shape[2]):
            img = x[:, :, i].copy()
            mi = img.min()
            if mi < 0:
                img = (((img - mi) / img.max()) * 255).astype(np.uint8)
            else:
                img = ((img / img.max()) * 255).astype(np.uint8)
            img = np.array([img])
            img = img[np.newaxis, ...]
            xx[i,...] = img
        x = xx

    return x


def main(opt):
    input_path = "/data/brain/rs-mhd-dataset"#opt.input
    weight_brain = "/data/brain/checkpoints/lr=0.0001_bs=8_dice=0.8_batch_10.pt" #opt.weight_brain
    # weight_gliom = opt.weight_gliom
    output_path = '/data/brain/rs-mhd-dataset/net_out_masks_torch'

    patients_id = glob.glob1(input_path, "AR*")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    brain_segm = TestNet(weight_brain, "Brain segmentation", True, device)
    # gliom_segm = TestNet(weight_gliom, "Gliomas segmentation", False)

    for patient_id in patients_id:
        if patient_id in ["AR-5"]:
            continue

        patient_saving_path = os.path.join(output_path, "%s_rs_mask.mhd" % patient_id)

        x = load_patient(input_path, patient_id)
        x = brain_segm.predict(x)
        sitk.WriteImage(sitk.GetImageFromArray(x), patient_saving_path)

        print("Patient's gliomas saved to: ", patient_saving_path)


if __name__ == '__main__':
    opt = arguments()
    main(opt)
