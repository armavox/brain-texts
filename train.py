import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler

from data_utils import BertFeaturesDataset
from models.unet import UNet
from models.text_net import BrainLSTM

def plot_grad_flow(named_parameters):
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
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
                ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig('asd.png')

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--imgs_folder", default=None, type=str, required=True)
    parser.add_argument("--texts_file", default=None, type=str, required=True)
    parser.add_argument("--labels_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: "
                             "bert-base-uncased, bert-large-uncased,"
                             "bert-base-cased, bert-base-multilingual,"
                             "bert-base-chinese.")

    # Other parameters
    parser.add_argument("--epochs", default=3, type=int,
                        help="Batch size for predictions.")
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size for predictions.")
    parser.add_argument('--max_seq_length', default=256, type=int,
                        help="Seq size for texts embeddings.")
    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    n_gpu = torch.cuda.device_count()

    data = BertFeaturesDataset(args.imgs_folder, args.texts_file,
                               args.labels_file, args.bert_model,
                               max_seq_length=args.max_seq_length,
                               batch_size=args.batch_size,
                               torch_device='cpu')
    sampler = SequentialSampler(data)
    dl = DataLoader(data, args.batch_size, sampler=sampler)
    model = UNet(1)
    model = model.to(device)
    lstm = BrainLSTM(768, 256, 1, 2, 1)
    lstm = lstm.to(device)
    print(f'UNet using {device}')
    if device == 'cuda' and n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.SGD(lstm.parameters(), lr=0.001, weight_decay=5e-4)

    loss_func = nn.MSELoss()
    # model = model.double()
    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        for i, batch in enumerate(dl):
            optimizer.zero_grad()
            images = batch['image'].to(device)
            labels = batch['label'].float().to(device)
            embeddings = batch['embedding'].to(device)
            pred = lstm(embeddings)
            loss = loss_func(pred, labels)
            print(pred.cpu().detach().numpy(), labels.cpu().detach().numpy())
            print('LOSS', np.sqrt(loss.cpu().item()))
            loss.backward()
            # print(lstm.lstm.weight_ih_l0.grad.shape)
            # print(lstm.lstm.weight_hh_l0.grad.shape)
            plot_grad_flow(lstm.named_parameters())
            optimizer.step()
            train_loss += np.sqrt(loss.cpu().item())
        train_loss /= len(dl)
        if epoch % 1 == 0:
            print('Epoch: %04d Train loss: %.4f' % (epoch, train_loss))

if __name__ == "__main__":
    main()
