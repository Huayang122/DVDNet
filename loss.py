import torch
import torch.nn as nn


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCEWithLogitsLoss()

    def soft_dice_coeff(self, y_true, y_pred, smooth=0.1):
        smooth = smooth
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, tuple):
            y_true = y_true[0]
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        if y_true.size(1) == 1 and y_pred.size(1) == 2:
            y_true = y_true.expand(-1, 2, -1, -1)

        a = self.bce_loss(y_pred, y_true)
        y_pred_prob = torch.sigmoid(y_pred)
        b = self.soft_dice_loss(y_true, y_pred_prob)
        return a + b