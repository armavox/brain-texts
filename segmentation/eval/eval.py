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


def load_patient(general_path, patient_id):
    x = data_utils.stack_images(general_path, patient_id, False)

    if len(x.shape) == 3:
        xx = np.empty((x.shape[0], x.shape[1], x.shape[1], 3), dtype = np.uint8)
        for i in range(x.shape[0]):
            img = x[i]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            xx[i,...] = img
        x = xx

    return x


def main(opt):
    input_path = opt.input
    weight_brain = opt.weight_brain
    weight_gliom = opt.weight_gliom
    output_path = opt.output

    patients_id = glob.glob1(input_path, "G*")

    brain_segm = TestNet(weight_brain, "Brain segmentation", True)
    gliom_segm = TestNet(weight_gliom, "Gliomas segmentation", False)

    for patient_id in patients_id:
        patient_saving_path = os.path.join(output_path, "%s.npy" % patient_id)

        x = load_patient(input_path, patient_id)
        x = brain_segm.predict(x)
        np.save(os.path.join(output_path, "brain_%s.npy" % patient_id), x)
        x = gliom_segm.predict(x)

        np.save(patient_saving_path, x)
        print("Patient's gliomas saved to: ", patient_saving_path)


if __name__ == '__main__':
    opt = arguments()
    main(opt)
