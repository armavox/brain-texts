import glob
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import pydicom
from skimage import exposure, transform
from torch.utils.data import Dataset


class MIMIC_CXR_DICOM_Image:
    def __init__(self, filepath, study_id=None, resize_shape=None):
        self._filepath = filepath
        self._ignored_meta = ["PixelData"]
        self.id = os.path.basename(filepath).strip(".dcm")
        self.study_id = study_id
        self._image = None
        self._read_dcm()
        self._resize = resize_shape

    def __repr__(self):
        return f"""MIMIC_CXR_DICOM_Image filepath: {self._filepath}
    pixel_spacing: {self.pixel_spacing}
    image: {self._image if self._image is not None else "not loaded, call `image` attr to load"}
        """

    def _read_dcm(self):
        dcm = pydicom.dcmread(self._filepath)
        try:
            self.pixel_spacing = dcm.PixelSpacing
        except AttributeError:
            self.pixel_spacing = "undefined"

        tags = [tag for tag in dir(dcm) if tag[0].isupper()]
        self.meta = dict(
            (tag, getattr(dcm, tag))
            for tag in tags
            if (tag not in self._ignored_meta) and not (isinstance(getattr(dcm, tag), pydicom.Sequence))
        )

    @property
    def image(self):
        if self._image is not None:
            return self._image

        dcm = pydicom.dcmread(self._filepath)
        image = dcm.pixel_array
        try:
            image = image * dcm.RescaleSlope + dcm.RescaleIntercept
        except AttributeError as e:
            print(f"image: {self.id}; study_id: {self.study_id}; AtributeError: {e.args[0]}")

        image = self._map_to_01(image)

        # Inverse image if nessesary (air is white).
        if self.meta["PhotometricInterpretation"] == "MONOCHROME2":
            image = 1 - image
        elif self.meta["PhotometricInterpretation"] == "MONOCHROME1":
            image = image
        else:
            raise ValueError(f"Unknown PhotometricInterpretation for DX: {self.meta['PhotometricInterpretation']}")

        # Equalize image histogram (increase contrast).
        image = exposure.equalize_hist(image)

        # Strip redundant areas of the image.
        try:
            image = image[
                max(0, self.meta["CollimatorUpperHorizontalEdge"]) : self.meta["CollimatorLowerHorizontalEdge"],
                max(0, self.meta["CollimatorLeftVerticalEdge"]) : self.meta["CollimatorRightVerticalEdge"],
            ]
        except KeyError as e:
            print(f"image: {self.id}; study_id: {self.study_id}; KeyError: {e.args[0]}")

        # Resize image if resize_shape was provided.
        if self._resize:
            image = transform.resize(self._image, self._resize, preserve_range=True)

        self._image = image
        return self._image

    def _map_to_01(self, image: np.ndarray):
        return (image - image.min()) / (image.max() - image.min())


class MIMIC_CXR_DICOM_Study:
    def __init__(
        self,
        dicom_image_list: List[MIMIC_CXR_DICOM_Image] = None,
        report_txt_path: str = None,
        folderpath: str = None,
        chexpert_csv_path: str = None,
        df_chexpert: pd.DataFrame = None,
        resize_shape: tuple = None,
    ):
        assert (bool(dicom_image_list) and bool(report_txt_path)) ^ bool(
            folderpath
        ), "Provide either (dicom_image_list and report_txt_path) or folderpath."

        self.images = []
        self.id = "undefined"
        self._resize_shape = resize_shape

        if dicom_image_list:
            self.images = dicom_image_list
            self.report = self._read_txt(report_txt_path)

        elif folderpath:
            self._folderpath = folderpath.rstrip("/")
            self.id = int(os.path.basename(self._folderpath).strip("s"))
            self._parse_mimic_cxr_study_folder(self._folderpath)
            self.report = self._read_txt(self._folderpath + ".txt")
            if df_chexpert is not None:
                self.chexpert_labels = self._parse_mimic_cxr_chexpert_csv(df_chexpert=df_chexpert)
            elif chexpert_csv_path is not None:
                self.chexpert_labels = self._parse_mimic_cxr_chexpert_csv(chexpert_csv=chexpert_csv_path)

    def __repr__(self):
        return f"Study id: {self.id}\nImages: {len(self.images)}\nREPORT\n======\n{self.report}"

    def __getitem__(self, i):
        return self.images[i]

    def __len__(self):
        return len(self.images)

    def _parse_mimic_cxr_study_folder(self, folderpath):
        flist = glob.glob(os.path.join(folderpath, "*.dcm"))
        for fpath in flist:
            self.images.append(MIMIC_CXR_DICOM_Image(fpath, self.id, self._resize_shape))

    def _read_txt(self, report_txt_path):
        with open(report_txt_path) as f:
            doc = f.readlines()
        raw_report = " ".join(doc).replace("\n", " ").strip()
        raw_report = " ".join(raw_report.split())
        raw_report = re.sub(r"(\w+:)", r"\n\1", raw_report).strip("FINAL REPORT").strip("\n")
        return raw_report

    def _parse_mimic_cxr_chexpert_csv(self, csv_path: str = None, df_chexpert: pd.DataFrame = None):
        if df_chexpert is None:
            try:
                df_chexpert = pd.read_csv(csv_path)
            except FileNotFoundError:
                raise FileNotFoundError("Provide path to `mimic-cxr-2.0.0-chexpert.csv.gz`")
        chexpert_study = df_chexpert[df_chexpert["study_id"] == self.id]
        chexpert_study = chexpert_study.where(chexpert_study.notna(), None)
        return dict((k.lower().replace(" ", "_"), v) for k, v in chexpert_study.iloc[0].items())


class MIMIC_CXR_DICOM_Subject:
    def __init__(
        self,
        folderpath: str,
        chexpert_csv_path: str = None,
        df_chexpert: pd.DataFrame = None,
        resize_shape=None,
        **kwargs,
    ):
        self._folderpath: str = folderpath
        self._chexpert_csv = chexpert_csv_path
        self._df_chexpert = df_chexpert
        self._resize_shape = resize_shape
        self.id = int(os.path.basename(self._folderpath).strip("p"))

        self.studies: Dict[int, List[MIMIC_CXR_DICOM_Study]] = {}
        self._parse_mimic_cxr_patient_folder(folderpath)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        study_ids = "\n".join(["- " + str(_) for _ in sorted(self.studies.keys())])
        return f"Patient ID: {self.id}\nStudies: {len(self.studies)}\n\n{study_ids}"

    def __getitem__(self, i):
        keys = list(self.studies)
        return self.studies[keys[i]]

    def __len__(self):
        return len(self.studies)

    def _parse_mimic_cxr_patient_folder(self, folderpath):
        for path in glob.glob(os.path.join(folderpath, "*")):
            if os.path.isdir(path):
                study_id = int(os.path.basename(path).strip("s"))
                if self._chexpert_csv is not None:
                    study = MIMIC_CXR_DICOM_Study(
                        folderpath=path,
                        chexpert_csv_path=self._chexpert_csv,
                        resize_shape=self._resize_shape,
                    )
                elif self._df_chexpert is not None:
                    study = MIMIC_CXR_DICOM_Study(
                        folderpath=path,
                        df_chexpert=self._df_chexpert,
                        resize_shape=self._resize_shape,
                    )
                else:
                    study = MIMIC_CXR_DICOM_Study(folderpath=path, resize_shape=self._resize_shape)
                self.studies[study_id] = study


class MIMIC_CXR_Dataset:
    def __init__(self, root_path: str, chexpert_csv_path: str = None, resize_shape=None):

        self.patients = {}
        self._files_folder = os.path.join(root_path, "files")
        self._resize_shape = resize_shape
        self.df_chexpert = None

        if chexpert_csv_path:
            self.df_chexpert = pd.read_csv(chexpert_csv_path)

        for split_folder in glob.glob(os.path.join(self._files_folder, "*")):
            for pat_folder in glob.glob(os.path.join(split_folder, "*")):
                subject_id = int(os.path.basename(pat_folder).strip("p"))
                self.patients[subject_id] = pat_folder

        self.keys = sorted(self.patients.keys())

    def __getitem__(self, i):
        return MIMIC_CXR_DICOM_Subject(
            self.patients[self.keys[i]], df_chexpert=self.df_chexpert, resize_shape=self._resize_shape
        )

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        return MIMIC_CXR_Dataset_Iterator(self)

    def __repr__(self):
        return f"MIMIC-CXR Dataset. Subjects: {len(self.keys)}"

    def get_subject(self, subject_id):
        return MIMIC_CXR_DICOM_Subject(
            self.patients[subject_id], df_chexpert=self.df_chexpert, resize_shape=self._resize_shape
        )

    def get_study(self, study_id, return_subject_id=False):
        try:
            study_df = self.df_chexpert[self.df_chexpert["study_id"] == study_id]
        except AttributeError:
            raise AttributeError(
                "Download chexpert labels `mimic-cxr-2.0.0-chexpert.csv.gz` from MIMIC-CXR-JPG dataset"
            )
        subject_id = study_df["subject_id"].iloc[0]
        subject = MIMIC_CXR_DICOM_Subject(
            self.patients[subject_id], df_chexpert=self.df_chexpert, resize_shape=self._resize_shape
        )
        if return_subject_id:
            return subject.studies[study_id], subject_id
        return subject.studies[study_id]


class MIMIC_CXR_Dataset_Iterator:
    def __init__(self, mimic_dataset):
        self.patients = mimic_dataset.patients
        self.keys = mimic_dataset.keys
        self.df_chexpert = mimic_dataset.df_chexpert
        self.resize_shape = mimic_dataset._resize_shape
        self._i = 0

    def __next__(self):
        try:
            subj = MIMIC_CXR_DICOM_Subject(
                self.patients[self.keys[self._i]], df_chexpert=self.df_chexpert, resize_shape=self.resize_shape
            )
        except IndexError:
            raise StopIteration()
        self._i += 1
        return subj

    def __iter__(self):
        return self
