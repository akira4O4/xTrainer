import torch
from torch import nn
from loss import pytorch_iou
from loss import pytorch_ssim


class IouSsimLoss(nn.Module):
    def __init__(self,**kwargs):
        super(IouSsimLoss, self).__init__()
        self.bce_loss_ = nn.BCELoss(size_average=True)
        self.ssim_loss_ = pytorch_ssim.SSIM(window_size=11, size_average=True)
        self.iou_loss_ = pytorch_iou.IOU(size_average=True)

    def bce_ssim_loss(self,pred, target):
        bce_out = self.bce_loss_(pred, target)
        ssim_out = 1 - self.ssim_loss_(pred, target)
        iou_out = self.iou_loss_(pred, target)
        loss = bce_out + ssim_out * 0.01 + iou_out * 0.01

        return loss

    def forward(self, output,labels_v):
        d0,d1=output
        d0=torch.sigmoid(d0)
        d1=torch.sigmoid(d1)
        loss0 = self.bce_ssim_loss(d0, labels_v)
        loss1 = self.bce_ssim_loss(d1, labels_v)
        loss=loss0+loss1
        return loss
