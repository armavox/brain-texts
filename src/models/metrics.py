import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning.metrics as M
import pytorch_lightning.metrics.functional as FM
from sklearn import metrics


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
    def __init__(self, y_hat, y_actual, logits=True):
        self.probs, self.trues = y_hat, y_actual

        if logits:
            self.probs = y_hat.softmax(1)
            y_hat = self.probs.argmax(1)

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
    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    @property
    def sensitivity(self):
        return self.TP / (self.TP + self.FN)

    @property
    def specificity(self):
        return self.TN / (self.TN + self.FP)

    @property
    def precision(self):
        return self.TP / (self.TP + self.FP)

    @property
    def negative_predictive_value(self):
        return self.TN / (self.TN + self.FN)

    @property
    def fpr(self):
        return self.FP / (self.FP + self.TN)

    @property
    def mcc(self):
        num = self.TP * self.TN - self.FP * self.FN
        den = ((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN)) ** 1 / 2
        return num / den

    @property
    def roc(self):
        fpr, tpr, thr = FM.roc(self.preds[:, 1], self.trues)
        fpr, tpr, thr = fpr.numpy(), tpr.numpy(), thr.numpy()
        return fpr, tpr, thr

    @property
    def pr(self):
        pr, rec, thr = FM.precision_recall_curve(self.preds[:, 1], self.trues)
        pr, rec, thr = pr.numpy(), rec.numpy(), thr.numpy()
        return pr, rec, thr

    @property
    def auroc(self):
        fpr, tpr, _ = self.roc
        return metrics.auc(fpr, tpr)

    @property
    def aupr(self):
        pr, rec, _ = self.pr
        return metrics.auc(rec, pr)

    def plot_roc_pr_curves(self):
        fpr, tpr, roc_thr = self.roc
        pr, rec, pr_thr = self.pr

        fig, ax = plt.subplots(1, 2, figsize=(16, 4))
        ax[0].plot(fpr, tpr)
        ax[0].set_title(f"ROC. AUC: {self.auroc:.4f}")
        ax_thr = ax[0].twinx()
        ax_thr.plot(fpr[1:], roc_thr[1:], c='orange')

        ax[1].step(rec, pr)
        ax[1].set_title(f'PR. AUC: {self.aupr:.4f}')
        ax_thr = ax[1].twinx()
        ax_thr.plot(rec[:-1], pr_thr, c='orange')

        return fig
