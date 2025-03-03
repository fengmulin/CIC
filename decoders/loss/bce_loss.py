import torch
import torch.nn as nn


class BalanceCrossEntropyLoss_my(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    '''

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss_my, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        
    def _bce_my_loss(self,pred,target):
        
        loss = (target - 1) * torch.log(1 - pred + 1e-37)/torch.exp(1 - pred) - target * torch.log(pred+1e-37)/torch.exp(pred)
        return loss
    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()),
                            int(positive_count * self.negative_ratio))
        # loss = nn.functional.binary_cross_entropy(
        #     pred, gt, reduction='none')[:, 0, :, :]
        # loss1 = nn.functional.binary_cross_entropy(
        #     pred, gt, reduction='none')[:, 0, :, :]
        # loss2 = nn.functional.binary_cross_entropy(
        #     pred, gt, reduction='none')[:, 0, :, :]
        loss = self._bce_my_loss(pred,gt)[:, 0, :, :]
        #print(torch.mean(loss1)-torch.mean(loss))
        # print(loss1== loss)
        # if not loss1.equal(loss):
        #     print(loss1,111)
        #     print(loss,222)
        #     print('******************')
        #     raise
        # print(loss.shape)
        #print(loss.grad_fn, loss1.grad_fn, loss2.grad_fn)
        #print(type(loss),type(loss1),loss.shape,loss1.shape)
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) /\
            (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss
