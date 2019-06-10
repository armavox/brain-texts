import torch
from torch import nn


def jaccard_metric(y_pred, y_true):
    eps = 1e-15
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()

    result = (intersection + eps) / (union - intersection + eps)
    print("jaccard: ", result.item())
    return result


class JaccardBCELoss:
    def __init__(self, jaccard_weight):
        self.bce = nn.BCELoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        y_true = (targets > 0).view(-1).float()
        y_pred = torch.sigmoid(outputs).view(-1).float()

        loss = (1 - self.jaccard_weight) * self.bce(y_pred, y_true)
        print("loss: ", loss.item())

        if self.jaccard_weight:
            jaccard_met = jaccard_metric(y_pred, y_true)
            jaccard = self.jaccard_weight * torch.log(jaccard_met)
            print("log jaccard: ", jaccard.item())
            loss -= jaccard


        return loss, jaccard_met.item()
