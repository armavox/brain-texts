import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from torch.utils.data import Subset

from data_utils import BertFeaturesDataset, train_val_holdout_split
from models.unet import UNet
from models.vgg import VGG, VGG11
from models.text_net import BrainLSTM
from models.fuse import EarlyFusion


def plot_grad_flow(named_parameters, epoch):
    """Plots the gradients flowing through different layers
    in the net during training. Can be used for checking for
    possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after 
    loss.backward() as  "plot_grad_flow(self.model.named_parameters())"
    to visualize the gradient flow.
    """

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=45)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title(f"Epoch{epoch}. Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
                ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig(f'epoch_{epoch}_gf.png')

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
    parser.add_argument("--epochs", default=3, type=int,
                        help="Batch size for predictions.")
    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size for predictions.")
    parser.add_argument('--max_seq_length', default=256, type=int,
                        help="Seq size for texts embeddings.")
    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    n_gpu = torch.cuda.device_count()

    ###
    imgs_folder = '/data/brain-skull-stripped/rs/'
    input_text_file = '/data/brain-skull-stripped/dataset/annotations.txt'
    labels_file = '/data/brain-skull-stripped/dataset/brain-labels.csv'
    bert_model = 'bert-base-uncased'
    ###
    data = BertFeaturesDataset(imgs_folder, input_text_file,
                               labels_file, bert_model,
                               max_seq_length=args.max_seq_length,
                               batch_size=args.batch_size,
                               torch_device='cpu')
    
    np.random.seed(0)  # TODO: saving indices for test phase
    train_inds, val_inds, test_inds = train_val_holdout_split(
        data, ratios=[0.6,0.4,0]
    )
    print('INDS', train_inds, val_inds, test_inds)
    train_sampler = SubsetRandomSampler(train_inds)
    val_sampler = SubsetRandomSampler(val_inds)
    test_sampler = Subset(data, test_inds)

    train_loader = DataLoader(data, batch_size=args.batch_size,
                              sampler=train_sampler)
    val_loader = DataLoader(data, batch_size=args.batch_size*2,
                            sampler=val_sampler)
    test_loader = DataLoader(test_sampler)

    # vgg = VGG11(combine_dim=2)
    # vgg = vgg.to(device)
    lstm = BrainLSTM(768, 256, 1, 2, 2)
    lstm = lstm.to(device)

    # model = EarlyFusion(combine_dim=128)
    # model = BrainLSTM(768, 256, 1, 2, 2)
    # model = model.to(device)
    print(f'UNet using {device}')
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {n_gpu} CUDAs")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)
    # vgg.load_state_dict(torch.load('checkpoints/vgg20.pth'))

    optimizer = torch.optim.SGD(lstm.parameters(),
                                lr=0.001, weight_decay=5e-2)
    lambda2 = lambda epoch: 0.95 ** epoch
    schedlr = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=[lambda2])

    # loss_func = nn.MSELoss()
    # loss_func = nn.NLLLoss()
    # softmax = nn.LogSoftmax(dim=1)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        lstm.train()
        train_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            labels = batch['label'].long().to(device).squeeze(1)
            images = batch['image'].to(device)
            embeddings = batch['embedding'].to(device)
            # pred = model(embeddings, images)
            pred = lstm(embeddings)
            loss = loss_func(pred, labels)
            loss.backward()
            plot_grad_flow(lstm.named_parameters(), epoch)
            schedlr.step()
            train_loss += np.sqrt(loss.cpu().item())
        train_loss /= len(train_loader)
        if epoch % 1 == 0:
            print('Epoch: %04d Train loss: %.4f' % (epoch, train_loss))

        # VALIDATE
        if epoch % 10 == 0:
            lstm.eval()
            val_loss = 0
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].long().to(device).squeeze(1)
                    embeddings = batch['embedding'].to(device)
                    # img_pred = vgg(images)
                    pred = lstm(embeddings)
                    # loss = loss_func(pred, labels)
                    val_loss += loss_func(pred, labels)
                    pred = pred.data.max(1)[1]
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
                    total += labels.size(0)
                val_loss /= len(val_loader)
                acc = 100. * correct / total
                print('  Val loss: %.4f Acc: %.2f' % (val_loss.item(), acc))
                torch.save(lstm.state_dict(), f'checkpoints/fuse{epoch}.pth')


if __name__ == "__main__":
    main()
