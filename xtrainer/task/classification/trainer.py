import os
from copy import deepcopy
from typing import Union, List

import torch
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from mlflow import log_metric

from xtrainer.core.lr_scheduler import LRSchedulerWrapper
from xtrainer.core.preprocess import (
    ClsImageT,
    ClsTargetT,
    ClsValT,
    SegImageT,
    SegValT
)

from xtrainer import CONFIG, DEFAULT_OPTIMIZER

from xtrainer.core.model import Model
from xtrainer.core.optim import (
    AMPOptimWrapper,
    OptimWrapper,
    build_optimizer_wrapper,
    build_amp_optimizer_wrapper
)

from xtrainer.dataset.segmentation import SegmentationDataSet
from xtrainer.dataset.classification import ClassificationDataset
from xtrainer.utils.labels import Labels
from xtrainer.utils.common import (
    round4,
    round8,
    timer,
    print_of_mt,
    print_of_seg,
    print_of_cls,
)

from xtrainer.core.loss import ClassificationLoss, SegmentationLoss
from xtrainer.utils.task import Task
from xtrainer.utils.perf import (
    topk_accuracy,
    compute_confusion_matrix_classification,
    compute_confusion_matrix_segmentation,
    draw_confusion_matrix,
    compute_iou
)
from xtrainer.utils.tracker import (
    ClsTrainTracker,
    ClsValTracker,
    SegTrainTracker,
    SegValTracker
)
from xtrainer.utils.torch_utils import (
    loss_sum,
    convert_optimizer_state_dict_to_fp16
)

from base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.train_tracker = ClsTrainTracker(topk=np.argmax(CONFIG['topk']))  # noqa
        self.val_tracker = ClsValTracker(topk=np.argmax(CONFIG['topk']))  # noqa
        self.labels = Labels(CONFIG['classification.labels'])

    def init_loss(self) -> None:

        alpha = CONFIG['alpha']
        if alpha == 'auto':
            # alpha = [1] * self.model.num_classes
            alpha = [1] * self.labels.nc

        alpha = torch.tensor(alpha, dtype=torch.float)
        alpha = self.to_device(alpha)

        self.loss = ClassificationLoss(alpha=alpha, gamma=CONFIG['gamma'])
        logger.success('Init Classification Loss.')

    def init_ds_dl(self) -> None:
        wh = tuple(CONFIG['wh'])
        workers: int = CONFIG['workers']
        use_cache: bool = CONFIG['cache']
        bs: int = CONFIG['classification.batch']

        # Build Train Dataset --------------------------------------------------------------------------------------
        self.train_ds = ClassificationDataset(
            root=CONFIG['classification.train'],
            wh=wh,
            labels=self.labels,
            transform=ClsImageT(wh),
            target_transform=ClsTargetT(),
            cache=use_cache
        )
        logger.success('Init classification train dataset.')

        self.val_ds = ClassificationDataset(
            root=CONFIG['classification.val'],
            wh=wh,
            labels=self.labels,
            transform=ClsValT(wh),
            target_transform=ClsTargetT(),
            cache=use_cache
        )
        logger.success('Init classification val dataset.')

        logger.info(f'Classification Train data size: {self.train_ds.real_data_size}.')
        logger.info(f'Classification Val data size: {self.val_ds.real_data_size}.')

        batch_sampler = None
        if bs < self.labels.nc:
            logger.info('Close BalancedBatchSampler.')
        else:
            logger.info('Open BalancedBatchSampler')
            batch_sampler = BalancedBatchSampler(
                self.train_ds.targets,
                batch_size=bs
            )

        # Build Train DataLoader -----------------------------------------------------------------------------------
        self.train_dl = DataLoader(
            dataset=self.train_ds,
            batch_size=bs if bs < self.labels.nc else 1,
            num_workers=workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            shuffle=True if bs < self.labels.nc else False,
            drop_last=False,
            sampler=None
        )
        logger.success('Init classification train dataloader.')

        self.val_dl = DataLoader(
            dataset=self.val_ds,
            batch_size=bs,
            num_workers=workers,
            pin_memory=True,
            shuffle=False
        )
        logger.success('Init classification val dataloader.')

    def train(self) -> None:

        self.model.train()

        datas: tuple
        for curr_step, datas in enumerate(self.train_dl):
            images, targets = datas
            images = self.to_device(images)
            targets = self.to_device(targets)
            loss = self.forward(images, targets)

            with self.optimizer.context() as opt:
                opt.update(loss)

        self.lr_scheduler.update()

    def forward(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with self.optimizer.context():
            outputs = self.model(images)
            loss = self.loss(outputs, targets)  # noqa

        topk: List[float] = topk_accuracy(outputs, targets, CONFIG['topk'])

        maxk = max(CONFIG["topk"])
        maxk_idx = np.argmax(CONFIG["topk"])

        top1_val = topk[0]
        topk_val = topk[maxk_idx]

        self.train_tracker.top1.add(top1_val)
        self.train_tracker.topk.add(topk_val)
        self.train_tracker.loss.add(loss.cpu().detach())

        log_metric('Train Batch Top1', top1_val)
        log_metric(f'Train Batch Top{maxk}', topk_val)

        return loss

    def val(self) -> None:
        self.model.eval()

        maxk: int = max(CONFIG["topk"])
        maxk_idx = np.argmax(CONFIG["topk"])

        confusion_matrix = 0

        for data in self.val_dl:
            images, targets = data
            images = self.to_device(images)
            targets = self.to_device(targets)

            output = self.model(images)  # [[cls1,cls2],[seg1,seg2,...]]

            confusion_matrix += compute_confusion_matrix_classification(output, targets, self.labels.nc)
            topk: List[float] = topk_accuracy(output, targets, CONFIG['topk'])

            top1_val = topk[0]
            topk_val = topk[maxk_idx]

            self.val_tracker.top1.add(top1_val)
            self.val_tracker.topk.add(topk_val)

        draw_confusion_matrix(
            confusion_matrix,
            self.labels.labels,
            os.path.join(CONFIG['experiment_path'], 'cls_confusion_matrix.png')
        )
        total_top1: float = self.val_tracker.top1.avg  # i.e.60%
        total_topk: float = self.val_tracker.topk.avg  # i.e.80%
        log_metric('Val Epoch Top1', total_top1)
        log_metric(f'Val Epoch Top{maxk}', total_topk)

    def save_model(self) -> None:

        save_dict = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict,
            'model_name': self.model.model_name,
            'num_classes': self.labels.nc,
            'optimizer': convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            'lr': self.optimizer.lrs[0]
        }
        save_path = os.path.join(CONFIG['weight_path'], f'epoch{self.epoch}.pth')
        torch.save(save_dict, save_path)
