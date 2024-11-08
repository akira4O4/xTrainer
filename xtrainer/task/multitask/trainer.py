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


class MultiTaskTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.cls_trainer = ClassificationTrainer()
        self.seg_trainer = SegmentationTrainer()
        self.init_ds_dl()
        self.init_loss()

    def init_ds_dl(self) -> None:
        self.cls_trainer.init_ds_dl()
        self.seg_trainer.init_ds_dl()

    def init_loss(self) -> None:
        self.cls_trainer.init_loss()
        self.seg_trainer.init_loss()

    def train(self) -> None:
        self.model.train()

        dataloaders: List[DataLoader] = []
        if self.task.CLS or self.task.MT:
            dataloaders.append(self.cls_trainer.train_dl)
        if self.task.SEG or self.task.MT:
            dataloaders.append(self.seg_trainer.train_dl)

        datas: tuple
        for curr_step, datas in enumerate(zip(*dataloaders)):

            if self.task.MT:
                cls_data, seg_data = datas
            else:
                cls_data = seg_data = datas[0]

            cls_loss = 0
            seg_loss = 0

            if self.task.CLS or self.task.MT:
                images, targets = cls_data
                images = self.to_device(images)
                targets = self.to_device(targets)
                cls_loss = self.cls_trainer.forward(images, targets)

            if self.task.SEG or self.task.MT:
                images, targets = seg_data
                images = self.to_device(images)
                targets = self.to_device(targets)
                seg_loss = self.seg_trainer.forward(images, targets)

            if self.task.MT:
                final_loss = loss_sum([cls_loss, seg_loss], CONFIG['loss_sum_weights'])
            else:
                final_loss = cls_loss + seg_loss

            with self.optimizer.context() as opt:
                # self.optimizer.update(final_loss)
                opt.update(final_loss)

        self.lr_scheduler.update()

    def val(self) -> None:
        self.model.eval()
        if self.task.SEG or self.task.MT:
            self.seg_trainer.val()

        if self.task.CLS or self.task.MT:
            self.cls_trainer.val()

    def save_model(self) -> None:
        save_dict = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict,
            'model_name': self.model.model_name,
            'num_classes': self.cls_trainer.labels.nc,
            'mask_classes': self.seg_trainer.labels.nc,
            'optimizer': convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            'lr': self.optimizer.lrs[0]
        }
        save_path = os.path.join(CONFIG['weight_path'], f'epoch{self.epoch}.pth')
        torch.save(save_dict, save_path)


class Trainer:
    def __init__(self):

        if CONFIG['task'].lower() == 'classification':
            self.trainer = ClassificationTrainer()
        elif CONFIG['task'].lower() == 'segmentation':
            self.trainer = SegmentationTrainer()
        elif CONFIG['task'].lower() == 'multitask':
            self.trainer = MultiTaskTrainer()

        self.trainer.init_model()
        self.trainer.init_loss()
        self.trainer.init_ds_dl()
        self.trainer.init_optimizer()
        self.trainer.init_lr_scheduler()

    def run(self) -> None:
        while self.trainer.epoch < CONFIG['epochs']:
            for mode in ['train', 'val']:

                if mode == 'train':
                    self.trainer.train()
                    self.trainer.epoch += 1

                    if self.trainer.epoch % CONFIG['save_period'] == 0:
                        self.trainer.save_model()
                    log_metric('Epoch', self.trainer.epoch)

                else:
                    if CONFIG['not_val'] is True:
                        continue
                    self.trainer.val()

                lr: float = round8(self.trainer.optimizer.lrs[0]) if mode == 'train' else None

                # Display info
                if self.trainer.task.MT:
                    cls_loss: float = round4(self.trainer.cls_trainer.loss.avg) if mode == 'train' else None
                    seg_loss: float = round4(self.trainer.seg_trainer.loss.avg) if mode == 'train' else None
                    top1 = round4(
                        self.trainer.cls_trainer.train_tracker.top1.avg if mode == 'train' else self.trainer.val_tracker.top1.avg)
                    topk = round4(
                        self.trainer.cls_trainer.train_tracker.topk.avg if mode == 'train' else self.trainer.val_tracker.topk.avg)
                    miou = round4(
                        self.trainer.seg_trainer.train_tracker.miou.avg if mode == 'train' else self.trainer.val_tracker.miou.avg)

                    print_of_mt(mode, 'MT', self.trainer.epoch, CONFIG['epochs'], cls_loss, seg_loss, lr, top1, topk,
                                miou)

                elif self.trainer.task.CLS:
                    cls_loss: float = round4(self.trainer.train_tracker.loss.avg) if mode == 'train' else None
                    top1 = round4(
                        self.trainer.train_tracker.top1.avg if mode == 'train' else self.trainer.val_tracker.top1.avg)
                    topk = round4(
                        self.trainer.train_tracker.topk.avg if mode == 'train' else self.trainer.val_tracker.topk.avg)

                    print_of_cls(mode, 'CLS', self.trainer.epoch, CONFIG['epochs'], cls_loss, lr, top1, topk, )

                elif self.trainer.task.SEG:
                    seg_loss: float = round4(self.trainer.train_tracker.loss.avg) if mode == 'train' else None
                    miou = round4(
                        self.trainer.train_tracker.miou.avg if mode == 'train' else self.trainer.val_tracker.miou.avg)

                    print_of_seg(mode, 'SEG', self.trainer.epoch, CONFIG['epochs'], seg_loss, lr, miou)

                self.trainer.train_tracker.reset()
                self.trainer.val_tracker.reset()
