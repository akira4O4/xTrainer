import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodLoss(nn.Module):
    def __init__(
            self, 
            period_thresh=0.5, 
            period_n_min=1000, 
            period_ignore_lb=-255,
            period_weights=None, 
            *args, 
            **kwargs
        )->None:
        """
        Args:
            thresh: 判断困难像素的样本阈值
            n_min:  用于统计loss的像素数量
            ignore_lb:  忽略不统计loss的label
            weights: 。
            *args:
            **kwargs:
        """
        super(PeriodLoss, self).__init__()

        # self.thresh = -torch.log(torch.tensor(period_thresh, dtype=torch.float)).cuda()
        self.n_min = period_n_min
        self.ignore_lb = period_ignore_lb
        self.device=period_weights.device
        self.criteria = nn.CrossEntropyLoss(ignore_index=period_ignore_lb, reduction='none',weight=period_weights)

    def forward(self, logits, labels):

        # labels_t=labels_t.squeeze(1) 
        # labels_t=labels_t.type(torch.Tensor)
        # labels_t=labels_t.to(self.device)
        # labels_t = torch.cuda.LongTensor(labels_t.cpu().numpy(),device=self.device)

        loss_mean = 0
        for i in range(len(logits)):
            if (logits[i].shape[2] != labels.shape[2]) and (logits[i].shape[2] / labels.shape[2] == logits[i].shape[3] / labels.shape[3]):
                labels_t = F.interpolate(labels, (logits[i].shape[2], logits[i].shape[3]))
            else:
                labels_t = labels.clone().detach()

            # labels_t = torch.cuda.LongTensor(labels_t.cpu().numpy()).squeeze(1)
            labels_t=labels_t.squeeze(1)
            labels_t = torch.LongTensor(labels_t.cpu().numpy()).to(self.device)

            loss = self.criteria(logits[i], labels_t)
            loss = loss.view(-1)
            loss, _ = torch.sort(loss, descending=True)
            n_min_1 = loss.shape[0] * 50 // 100
            loss_mean += torch.mean(loss[:n_min_1])
        return loss_mean / len(logits)