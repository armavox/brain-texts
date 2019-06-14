import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from torch.utils.data import Subset
import torch.nn.functional as Fn

import utils
from data_utils import BertFeaturesDataset, train_val_holdout_split
from models.unet import UNet
from models.vgg import VGG, VGG11
from models.text_net import BrainLSTM
from models.fuse import EarlyFusion


def main():
    parser = argparse.ArgumentParser()

# Required parameters
# parser.add_argument("--imgs_folder", default=None, type=str, required=True)
# parser.add_argument("--texts_file", default=None, type=str, required=True)
# parser.add_argument("--labels_file", default=None, type=str, required=True)
# parser.add_argument("--bert_model", default=None, type=str, required=True,
#                     help="Bert pre-trained model selected in the list: "
#                          "bert-base-uncased, bert-large-uncased,"
#                          "bert-base-cased, bert-base-multilingual,"
#                          "bert-base-chinese.")

    # Other parameters
    parser.add_argument("-e", "--epochs", default=3, type=int,
                        help="Epochs to train. Default: 3")
    parser.add_argument("-lr", "--lr", type=float, default=0.001,
                        help="Learning rate. Default: 0.001")
    parser.add_argument("-b", "--batch-size", default=4, type=int,
                        help="Batch size for predictions. Default: 4")
    parser.add_argument('--max-seq-length', default=256, type=int,
                        help="Seq size for texts embeddings. Default: 256")
    parser.add_argument('--no-cuda', action='store_true')

    args = parser.parse_args()

    dev = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    n_gpu = torch.cuda.device_count()

    ###
    imgs_folder = '/data/brain/rs-mhd-dataset/net_out_masks_torch/'
    input_text_file = '/data/brain/rs-mhd-dataset/annotations.txt'
    labels_file = '/data/brain/rs-mhd-dataset/brain-labels.csv'
    bert_model = 'bert-base-uncased'
    ###

    data = BertFeaturesDataset(imgs_folder, input_text_file,
                               labels_file, bert_model,
                               max_seq_length=args.max_seq_length,
                               batch_size=args.batch_size,
                               bert_device='cpu',
                               resize_to=64)
    
    np.random.seed(5)  # TODO: saving indices for test phase
    train_inds, val_inds, test_inds = train_val_holdout_split(
        data, ratios=[0.7, 0.3, 0]
    )
    train_inds = [10,  6, 18, 16, 11,  9,  4,  8,  2, 17,  7, 12,  5,  0,  1]
    val_inds = [15, 13,  3, 14]
    print('INDS', train_inds, val_inds, test_inds)
    train_sampler = SubsetRandomSampler(train_inds)
    val_sampler = SubsetRandomSampler(val_inds)
    test_sampler = Subset(data, test_inds)

    train_loader = DataLoader(data, batch_size=args.batch_size,
                              sampler=train_sampler)
    val_loader = DataLoader(data, batch_size=args.batch_size * 2,
                            sampler=val_sampler)
    test_loader = DataLoader(test_sampler)

    vgg = VGG11(combine_dim=2)
    vgg = vgg.to(dev)
    lstm = BrainLSTM(embed_dim=768, hidden_dim=256, num_layers=1,
                      context_size=2, combine_dim=2, dropout=0)
    lstm = lstm.to(dev)

    # model = EarlyFusion(combine_dim=4096)
    # model = VGG11(combine_dim=2)
    # model = model.to(dev)

    model_name = utils.get_model_name(vgg)
    prefix = "%s_lr=%s_bs=%s" % (model_name, args.lr, args.batch_size)

    opt = torch.optim.Adam([
        {'params': lstm.parameters()},
        {'params': vgg.parameters()}
    ], lr=0.001, weight_decay=1e-2)
    schedlr = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)

    loss_func = nn.CrossEntropyLoss()

    # TRAINING
    print(f'\n===== BEGIN TRAINING {model_name} WITH {str(dev).upper()} =====')
    loss_train = []
    loss_val, acc_val = [], []
    for epoch in range(args.epochs):
        lstm.train()
        vgg.train()
        train_loss = 0
        for batch in train_loader:

            labels = batch['label'].long().to(dev).squeeze(1)
            images = batch['image'].to(dev)
            embeddings = batch['embedding'].to(dev)

            out_lstm = lstm(embeddings)#, images)  # embeddings, 
            out_vgg = vgg(images)#, images)  # embeddings,

            loss_lstm = loss_func(out_lstm, labels)
            loss_vgg = loss_func(out_vgg, labels)

            loss = (loss_lstm + loss_vgg) / 2

            loss.backward()

            # utils.plot_grad_flow(model.named_parameters(),
            #                      epoch, 'grad_flow_plots')
            train_loss += loss.cpu().item()# - 1e-1 * l2_reg.cpu().item() - 1e-1 * l1_reg.cpu().item()

            opt.step()
            opt.zero_grad()

        train_loss /= len(train_loader)
        loss_train.append(train_loss)
        if epoch % 1 == 0:
            print('Epoch: %03d Train loss: %.4f' % (epoch, train_loss))

        # VALIDATION
        if epoch % 1 == 0:
            lstm.eval()
            vgg.eval()
            val_loss = 0
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:

                    labels = batch['label'].long().to(dev).squeeze(1)
                    images = batch['image'].to(dev)
                    embeddings = batch['embedding'].to(dev)

                    out_lstm = lstm(embeddings)#, images)  # embeddings, 
                    out_vgg = vgg(images)#, images)  # embeddings,

                    loss_lstm = loss_func(out_lstm, labels)
                    loss_vgg = loss_func(out_vgg, labels)

                    out = (out_lstm + out_vgg) / 2
                    val_loss += (loss_lstm + loss_vgg) / 2

                    print(Fn.softmax(out.data, dim=1))
                    print(labels)
                    out = out.data.max(1)[1]
                    correct += out.eq(labels.data.view_as(out)).cpu().sum()
                    total += labels.size(0)

                val_loss /= len(val_loader)
                loss_val.append(val_loss)
                acc = 100. * correct / total
                acc_val.append(acc)

                print('Epoch: %03d Valid loss: %.4f Acc: %.2f' % (epoch, val_loss.item(), acc))

                # torch.save(
                #     model.state_dict(),
                #     f'/data/brain/checkpoints/{prefix}_ep_{epoch}.pth'
                # )

        schedlr.step()

    plots_path = 'train_plots'
    utils.draw_plots(args.epochs, plots_path, prefix,
                     loss_train, loss_val, acc_val)


if __name__ == "__main__":
    main()
