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
from xtrainer.dataset.classification import ClassificationDataset, BalancedBatchSampler
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


class SegmentationTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.train_tracker = SegTrainTracker()
        self.val_tracker = SegValTracker()
        self.labels = Labels(CONFIG['segmentation.labels'])

    def init_ds_dl(self) -> None:
        wh = tuple(CONFIG['wh'])
        workers: int = CONFIG['workers']
        use_cache: bool = CONFIG['cache']
        bs: int = CONFIG['segmentation.batch']

        self.train_ds = SegmentationDataSet(
            root=CONFIG['segmentation.train'],
            wh=wh,
            labels=self.labels,
            transform=SegImageT(wh),
            cache=use_cache
        )
        logger.success('Init segmentation train dataset.')

        self.val_ds = SegmentationDataSet(
            root=CONFIG['segmentation.val'],
            wh=wh,
            labels=self.labels,
            transform=SegValT(wh),
            cache=use_cache
        )
        logger.success('Init segmentation val dataset.')

        background_size = len(self.train_ds.background_samples)
        logger.info(
            f'Segmentation Train data size: {self.train_ds.real_data_size} (background:{background_size}).')

        self.train_dl = DataLoader(
            dataset=self.train_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        logger.success('Init segmentation train dataloader.')

        self.val_dl = DataLoader(
            dataset=self.val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        logger.success('Init segmentation val dataloader.')
        logger.info(f'Segmentation Val data size: {self.val_ds.real_data_size}.')

    def init_loss(self) -> None:
        self.loss = SegmentationLoss(CONFIG['seg_loss_sum_weights'])
        logger.success('Init segmentation loss.')

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
            # segmentation output=[x1,x2,x3,x4]
            outputs = self.model(images)
            loss1 = 1 * self.loss(outputs[0], targets)  # noqa
            loss2 = 1 * self.loss(outputs[1], targets)  # noqa
            loss3 = 0.5 * self.loss(outputs[2], targets)  # noqa
            loss4 = 0.5 * self.loss(outputs[3], targets)  # noqa

        loss = loss1 + loss2 + loss3 + loss4

        miou: float = compute_iou(outputs[0], targets, self.labels.nc)

        self.train_tracker.miou.add(miou)
        self.train_tracker.loss.add(loss.cpu().detach())  # noqa
        log_metric('Train Batch MIoU', miou)

        return loss

    def val(self) -> None:
        self.model.eval()

        confusion_matrix = 0
        for data in self.val_dl:
            images, targets = data
            images = self.to_device(images)
            targets = self.to_device(targets)  # target.shape=(N,1,H,W)

            output = self.model(images)

            miou: float = compute_iou(output[0], targets, self.labels.nc)
            self.val_tracker.miou.add(miou)

            confusion_matrix += compute_confusion_matrix_segmentation(output[0], targets, self.labels.nc)

        draw_confusion_matrix(
            confusion_matrix,
            self.labels.labels,
            os.path.join(CONFIG['experiment_path'], 'seg_confusion_matrix.png')
        )
        total_miou: float = self.val_tracker.miou.avg
        log_metric('Val Epoch MIoU', total_miou)

    def save_model(self) -> None:

        save_dict = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict,
            'model_name': self.model.model_name,
            'mask_classes': self.labels.nc,
            'optimizer': convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            'lr': self.optimizer.lrs[0]
        }
        save_path = os.path.join(CONFIG['weight_path'], f'epoch{self.epoch}.pth')
        torch.save(save_dict, save_path)
