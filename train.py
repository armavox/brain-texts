import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler

from data_utils import BertFeaturesDataset
from models.unet import UNet


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
    print('befor dl')
    dl = DataLoader(data, args.batch_size, sampler=sampler)
    print('after dl')
    model = UNet(1)
    from pytorch_modelsize import SizeEstimator
    se = SizeEstimator(model, input_size=(1,1,152,256,256))
    print(se.estimate_size())
    print(f'UNet using {device}')
    model.to(device)
    

    
    # if device == 'cuda' and n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         print(f'{torch.cuda.device_count()} GPUs used')
    #         model = torch.nn.DataParallel(model)
    #     model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4)

    loss_func = nn.MSELoss()
    # model = model.double()
    for epoch in range(args.epochs):
        train_loss = 0
        model.train()
        for i, batch in enumerate(dl):
            optimizer.zero_grad()
            images, labels = batch['image'].to(device), batch['label'].to(device)
            pred = model(images)
            print(f'batch{i}')
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_loss /= len(dl)
        if epoch % 1 != 0:
            print('Epoch: %04d Train loss: %.4f' % (epoch, train_loss.item()))

if __name__ == "__main__":
    main()
