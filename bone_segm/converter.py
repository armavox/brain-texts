import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from bone_segm.NiiReader import NiiReader
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


if __name__ == '__main__':
    opt = arguments()

    datapath = opt.input
    path_save = opt.output
    patients = glob.glob1(datapath, "**")

    template_orig = 'sub-%s_ses-NFB3_T1w.nii.gz'
    template_mask = 'sub-%s_ses-NFB3_T1w_brainmask.nii.gz'

    height = opt.height
    width = opt.width
    aug_size = opt.aug

    reader = NiiReader((height, width))

    for patient in patients:
        print("Patient: ", patient)
        orig_filename = os.path.join(datapath, patient, template_orig % patient)
        mask_filename = os.path.join(datapath, patient, template_mask % patient)
        reader.save_to_npy(path_save, patient, orig_filename, mask_filename, aug_size)
