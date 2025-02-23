import torch
import torch.nn as nn
import math
class RegLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    '''
    def __init__(self, eps=1e-6):
        super(RegLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        # assert pred.dim() == 4, pred.dim()
        return self._compute(pred, gt, mask)

    def _compute(self, pred, gt, mask):
        # if pred.dim() == 4:
        #     pred = pred[:, 0, :, :]
        #     gt = gt[:, 0, :, :]
        #print(gt.shape,pred.shape)
        assert pred.shape == gt.shape
        #assert pred.shape == mask.shape
        # if weights is not None:
        #     assert weights.shape == mask.shape
        #     mask = weights * mask
        # print(pred[:, 0, :, :].shape,gt[:, 0, :, :].shape,444)

        maxx = torch.max(pred,gt)
        maxx = maxx*mask+1e-6
        minn = torch.min(pred,gt)*mask+1e-6
        print(maxx.max(),minn.min())
        loss = (torch.log(maxx/minn).sum())/mask.sum()
   
        return loss 