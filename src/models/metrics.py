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

    def get_mean(self):
        return dict((k, float(v.mean())) for k, v in self.get().items())


class Performance:
    def __init__(self, y_hat, y_actual):
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0

        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                self.TP += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                self.FP += 1
            if y_actual[i] == y_hat[i] == 0:
                self.TN += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                self.FN += 1

    def __repr__(self):
        return f"TP:{self.TP}, FP:{self.FP}, TN:{self.TN}, FN:{self.FN}"

    @property
    def sensitivity(self):
        return self.TP / (self.TP + self.FN)

    @property
    def specificity(self):
        return self.TN / (self.TN + self.FP)

    @property
    def precision(self):
        return self.TP / (self.TP + self.FP)
