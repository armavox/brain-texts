import os

import albumentations as A
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data.helpers import get_split_legths, random_seed_init
from data.mimic_cxr_dataset import MIMICCXRTorchDataset
from data.utils import PadCollate


if __name__ == "__main__":
    random_seed_init(42, True)

    DATASET_PATH = "/ws/rif-net-ws/data/origin/mimic-cxr-2.0.0/"
    CHEXPERT_CSV_PATH = "/ws/rif-net-ws/data/origin/mimic-cxr-2.0.0/mimic-cxr-2.0.0-chexpert.csv"
    TRAIN_VAL_TEST_RATIO = [0.7, 0.2, 0.1]

    ds = MIMICCXRTorchDataset(
        DATASET_PATH,
        CHEXPERT_CSV_PATH,
        label_name="Pneumonia",
        transform=A.Resize(224, 224, always_apply=True),
        bert_pooling_strategy="none",
    )

    train_len, val_len, test_len = get_split_legths(ds, ratio=TRAIN_VAL_TEST_RATIO)
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [train_len, val_len, test_len])

    dl = DataLoader(train_ds, batch_size=4, collate_fn=PadCollate(0))
    print(next(iter(dl)))

    PT_DATASET_PATH = "/data/hdd2/mimic-cxr-pt/mimic-cxr-pt-dataset-224-bert-reports-no-pool-squeezed"

    for sample in tqdm(train_ds, desc="train"):
        label, study_id = sample["label"], sample["study_id"]
        os.makedirs(f"{PT_DATASET_PATH}/train/{label}", exist_ok=True)
        fpath = os.path.join(f"{PT_DATASET_PATH}/train/{label}/{study_id}.pt")
        torch.save(sample, fpath)

    for sample in tqdm(valid_ds, desc="valid"):
        label, study_id = sample["label"], sample["study_id"]
        os.makedirs(f"{PT_DATASET_PATH}/valid/{label}", exist_ok=True)
        fpath = os.path.join(f"{PT_DATASET_PATH}/valid/{label}/{study_id}.pt")
        torch.save(sample, fpath)

    for sample in tqdm(test_ds, desc="test"):
        label, study_id = sample["label"], sample["study_id"]
        os.makedirs(f"{PT_DATASET_PATH}/test/{label}", exist_ok=True)
        fpath = os.path.join(f"{PT_DATASET_PATH}/test/{label}/{study_id}.pt")
        torch.save(sample, fpath)

    os.makedirs(f"{PT_DATASET_PATH}/done", exist_ok=True)
