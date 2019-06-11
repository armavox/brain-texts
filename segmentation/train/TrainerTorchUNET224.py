import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


class TrainerTorchUNET224:
    def __init__(self, model, train_loader, val_loader, checkpoint_path, criterion, optimizer, prefix, device='cpu',
                 epochs=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_path = checkpoint_path
        self.plots_path = os.path.join(self.checkpoint_path, "plots")
        self.criterion = criterion

        self.epochs = epochs
        self.prefix = prefix
        self.optimizer = optimizer
        self.device = device

        self.__prepare()

    def __prepare(self):
        os.makedirs(self.plots_path, exist_ok=True)
        self.__setup_model()

    def __setup_model(self):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f'{torch.cuda.device_count()} GPUs used')
                self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

    def train_model(self):
        loss_val, acc_val = [], []
        loss_train, acc_train = [], []

        for i in range(self.epochs):
            self.model.train()

            epoch_loss = 0
            epoch_metric = 0

            for j, (x, target) in enumerate(self.train_loader):
                x = x.to(self.device).float()
                with torch.no_grad():
                    target = target.to(self.device).float()

                output = self.model(x)
                loss, metric = self.criterion(output, target)
                print('Train Epoch: {}. Batch: {}/{} train_loss: {:.8f}'.format(i, j, len(self.train_loader), loss.item()))
                epoch_loss += loss.item()
                epoch_metric += metric
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            loss_train.append(epoch_loss / len(self.train_loader))
            acc_train.append(epoch_metric / len(self.train_loader))

            val_loss, val_acc = self.validate_model()
            loss_val.append(val_loss)
            acc_val.append(val_acc)

            print('Train Epoch: {} train_loss: {:.8f} val_loss: {:.8f} jaccard: {:.8f}'.format(
                    i, epoch_loss / len(self.train_loader), val_loss, val_acc))

            torch.save(self.model, os.path.join(self.checkpoint_path, "%s_batch_%s.pt" % (self.prefix, i)))

        self.draw_plots(loss_train, acc_train, loss_val, acc_val)

    def validate_model(self):
        self.model.eval()
        num_iter = len(self.val_loader)

        with torch.no_grad():
            global_loss = 0
            global_acc = 0

            for i, (x, target) in enumerate(self.val_loader):
                x, target = x.to(self.device).float(), target.to(self.device).float()

                output = self.model(x)

                loss, metric = self.criterion(output, target)
                global_loss += loss.item()
                global_acc += metric

        return global_loss / num_iter, global_acc / num_iter

    def draw_plots(self, loss_train, acc_train, loss_val, acc_val):

        time_str = datetime.now().strftime('%Y-%m-%d%H-%M-%S')

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(1, self.epochs + 1), loss_train, color='blue', label="train_loss")
        plt.plot(np.arange(1, self.epochs + 1), loss_val, color='red', label="test_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig( os.path.join(self.plots_path, "%s_loss_%s.png" % (self.prefix, time_str)))
        plt.close()

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(1, self.epochs + 1), acc_val, color='red', label="test_metric")
        plt.plot(np.arange(1, self.epochs + 1), acc_val, color='blue', label="train_metric")
        plt.title("Metric")
        plt.xlabel("Epoch #")
        plt.ylabel("Jaccard index")
        plt.legend(loc="upper right")
        plt.savefig( os.path.join(self.plots_path, "%s_metric_%s.png" % (self.prefix, time_str)))
        plt.close()
