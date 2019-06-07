import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from segmentation.utils.DataReader import DataReader
import argparse
import glob
import os


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="Path to folder with patients")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to output folder")
    parser.add_argument("-a", "--aug", type=int, default=2,
                        help="Count augmented images per slice. Default: 2")
    parser.add_argument("-he", "--height", type=int, default=224,
                        help="Height of output slices. Default: 224")
    parser.add_argument("-wi", "--width", type=int, default=224,
                        help="Width of output slices. Default: 224")
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


if __name__ == '__main__':
    opt = arguments()

    datapath = r"/home/anton/Un/brain-texts/data/rs-mhd-dataset" # opt.input
    path_save = r"/home/anton/Un/brain-texts/data/dataset_aug" # opt.output
    patients = glob.glob1(datapath, "**")

    template_orig = 'sub-%s_ses-NFB3_T1w.nii.gz'
    template_mask = 'sub-%s_ses-NFB3_T1w_brainmask.nii.gz'

    height = 224 #opt.height
    width = 224 #opt.width
    aug_size = 3 # opt.aug

    reader = DataReader((height, width), False)

    for patient in patients:
        if patient == "AR-5":
            continue
        patient_path = os.path.join(datapath, patient)

        if not os.path.isdir(patient_path):
            continue

        print("Patient: ", patient)

        orig_filename, mask_filename = get_orig_mask_filenames_from_patient_directory(patient_path)
        reader.save_to_npy(path_save, patient, orig_filename, mask_filename, aug_size)

