import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy import ndimage


class MutLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    '''
    def __init__(self, eps=1e-6):
        super(MutLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        assert pred.dim() == 4, pred.dim()
        return self._compute(pred, gt, mask)

    def _compute(self, pred, gt, mask):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        # if weights is not None:
        #     assert weights.shape == mask.shape
        #     mask = weights * mask

        # loss = pred.sum()
        try:
            loss = torch.sqrt((pred*(1-gt)*mask).sum()/mask.sum())
        except:
            print(pred*(1-gt).max(), pred*(1-gt).min())


        return loss


