from segmentation.model.Unet224Torch import Unet224Torch
from segmentation.utils.ARDataset import ARDataset
from segmentation.utils.metrics import JaccardBCELoss, jaccard_metric
from segmentation.train.TrainerTorchUNET224 import TrainerTorchUNET224
from segmentation.utils.utils import split_train_test_data

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="Path to folder with patients")
    parser.add_argument("-c", "--checkpoints", type=str,
                        help="Path to checkpoints folder")

    parser.add_argument("-l", "--lr", type=float, default=0.001,
                        help="Learning rate. Default: 0.001")
    parser.add_argument("-d", "--dice_weight", type=float, default=0.8,
                        help="Dice weight in error. Between 0 and 1. Default: 0.8")
    parser.add_argument("-bs", "--batch_size", type=int, default=4,
                        help="Batch size. Default: 4")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Count of epochs. Default: 10")
    parser.add_argument("-vs", "--valid_size", type=float, default=0.2,
                        help="Part of data for validation. Default: 0.2")
    return parser.parse_args()


def main(opt):
    checkpoint_path = opt.checkpoints
    dice_weight = opt.dice_weight
    lr = float(opt.lr)
    batch_size = int(opt.batch_size)
    data_path = opt.input
    validation_size = opt.valid_size
    epochs = int(opt.epochs)

    model = Unet224Torch(1)
    criterion = JaccardBCELoss(dice_weight)
    metric = jaccard_metric

    train, val = split_train_test_data(path=data_path, validation_size=validation_size)

    train_dataset = ARDataset(train)
    val_dataset = ARDataset(val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    prefix = "lr=%s_bs=%s_dice=%s" % (lr, batch_size, dice_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = TrainerTorchUNET224(model, train_loader, val_loader, checkpoint_path, criterion, optimizer, prefix, device, epochs)
    trainer.train_model()


if __name__ == '__main__':
    opt = arguments()
    main(opt)
