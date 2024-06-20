
import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self,focal_alpha=2,focal_gamma=2,**kwargs):
        super(FocalLoss, self).__init__()
        self.alpha=focal_alpha
        self.gamma=focal_gamma
    def forward(self,intputs,targets):
        one_hot=F.one_hot(targets).float()
        bce_loss=F.binary_cross_entropy_with_logits(intputs,one_hot,reduce=False)
        pt=torch.exp(-bce_loss)
        focal_loss=self.alpha*(1-pt)**self.gamma*bce_loss
        return torch.mean(focal_loss)

def focalloss(version=0,**kwargs):
    if version==0:
        return FocalLoss(**kwargs)