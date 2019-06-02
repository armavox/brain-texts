import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler

from data_utils import BertFeaturesDataset
from models.unet import UNet
from models.text_net import BrainLSTM


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--imgs_folder", default=None, type=str, required=True)
    parser.add_argument("--texts_file", default=None, type=str, required=True)
    parser.add_argument("--labels_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    # Other parameters
    parser.add_argument("--epochs", default=3, type=int,
                        help="Batch size for predictions.")
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size for predictions.")
    parser.add_argument('--max_seq_length', default=256, type=int,
                        help="Seq size for texts embeddings.")
    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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
    lstm = BrainLSTM(768, 256, 2, 2, 1)
    lstm = lstm.to(device)
    print(f'UNet using {device}')
    if device == 'cuda' and n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.SGD(lstm.parameters(), lr=0.001, weight_decay=5e-4)

    loss_func = nn.MSELoss()
    # model = model.double()
    for epoch in range(args.epochs):
        train_loss = 0
        model.train()
        for i, batch in enumerate(dl):
            optimizer.zero_grad()
            images = batch['image'].to(device)
            labels = batch['label'].float().to(device)
            # pred = model(images)
            pred = lstm(batch['embedding'])
            print(f'batch{i}, pred {pred}')
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().item()
        train_loss /= len(dl)
        if epoch % 1 == 0:
            print('Epoch: %04d Train loss: %.4f' % (epoch, train_loss))

if __name__ == "__main__":
    main()
