from segmentation.model.zf_unet_224_model import ZF_UNET_224, DiceBCELoss, dice_coef_loss, dice_coef
import segmentation.model.zf_unet_224_model as model_224
from segmentation.train.TrainerUNET224 import TrainerUNET224
from segmentation.utils.utils import data_generator, split_train_test_data
from keras.optimizers import Adam
import argparse
import numpy as np
np.random.seed(0)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="Path to folder with patients")
    parser.add_argument("-c", "--checkpoints", type=str,
                        help="Path to checkpoints folder")
    parser.add_argument("-w", "--weight", type=str,
                        help="Path to weight file")

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
    weight_path = opt.weight
    dice_weight = opt.dice_weight
    lr = float(opt.lr)
    batch_size = int(opt.batch_size)
    data_path = opt.input
    validation_size = opt.valid_size
    epochs = int(opt.epochs)

    prefix = "lr=%s_bs=%s_dice=%s" % (lr, batch_size, dice_weight)

    model = ZF_UNET_224()
    optimizer = Adam(lr=lr)

    model.compile(optimizer=optimizer, loss=DiceBCELoss(dice_weight), metrics=[dice_coef])

    train, val = split_train_test_data(path=data_path, validation_size=validation_size)

    train_loader = data_generator(train, batch_size)
    val_loader = data_generator(val, batch_size)

    trainer = TrainerUNET224(model=model,
                             train_loader=train_loader,
                             val_loader=val_loader,
                             checkpoint_path=checkpoint_path,
                             epochs=epochs,
                             train_steps_per_epoch=len(train) // batch_size + 1,
                             val_steps_per_epoch=len(val) // batch_size + 1, prefix=prefix)

    trainer.train_model()


if __name__ == '__main__':
    opt = arguments()

    main(opt)
