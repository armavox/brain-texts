import torch
from torch import nn


def jaccard_metric(y_pred, y_true):
    jaccard_target = (y_true > 0).view(-1).float()
    jaccard_output = torch.sigmoid(y_pred).view(-1).float()

    eps = 1e-15
    intersection = (jaccard_target * jaccard_output).sum()
    union = jaccard_target.sum() + jaccard_output.sum()

    result = (intersection + eps) / (union - intersection + eps)
    print("jaccard: ", result.item())
    return result


class JaccardBCELoss:
    def __init__(self, jaccard_weight):
        self.bce = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.bce(outputs, targets)

        if self.jaccard_weight:
            loss -= self.jaccard_weight * torch.log(jaccard_metric(outputs, targets))

        return loss
