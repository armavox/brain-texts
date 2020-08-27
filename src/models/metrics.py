import torch
import torch.nn as nn
import pytorch_lightning.metrics as M


class Metrics(nn.Module):
    def __init__(self, num_classes=2, channel_dim=1, prefix="train", reduction="elementwise_mean"):
        super().__init__()
        self.trues = []
        self.preds = []
        self.metrics = [
            M.Precision(num_classes, reduction),
            M.Recall(num_classes, reduction),
            M.Accuracy(num_classes, reduction),
        ]
        self.pref = prefix
        self.ch_dim = channel_dim

    def forward(self, y_pred, y_true):
        self.trues.append(y_true.cpu())
        self.preds.append(y_pred.detach().cpu())

    def reset(self):
        self.trues = []
        self.preds = []

    def get(self):
        trues = torch.cat(self.trues, dim=0)
        preds = torch.cat(self.preds, dim=0)

        results = {}
        if len(trues.unique()) != 1:
            preds = torch.softmax(preds, dim=self.ch_dim)
            for metric in self.metrics:
                results.update({f"{self.pref}_{metric.name}": metric(preds.cpu(), trues.cpu())})
        return results
