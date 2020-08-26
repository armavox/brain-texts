import re

import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from transformers import pipeline

from data import mimic_cxr_dataset as cxr


if __name__ == "__main__":

    DATASET_PATH = "/data/hdd2/mimic-cxr-2.0.0/"
    CHEXPERT_CSV_PATH = "/data/hdd2/mimic-cxr-2.0.0/mimic-cxr-2.0.0-chexpert.csv"

    mimic_dataset = cxr.MIMIC_CXR_Dataset(DATASET_PATH, CHEXPERT_CSV_PATH)
    patient = mimic_dataset[0]
    study = patient[0]

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    feature_extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    classifier = pipeline("sentiment-analysis")

    print(classifier(study.report))
    print(feature_extractor(study.report))
