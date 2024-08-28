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


class BaseTrainer:
    def __init__(self):

        self.epoch: int = 0
        self.task = Task(CONFIG('task'))

        # Instance
        self.model: Model = None  # noqa
        self.label: Labels = None  # noqa

        self.loss: Union[ClassificationLoss, SegmentationLoss] = None  # noqa
        self.optimizer: Union[OptimWrapper, AMPOptimWrapper] = None  # noqa
        self.lr_scheduler: LRSchedulerWrapper = None  # noqa

        self.train_ds: Union[ClassificationDataset, SegmentationDataSet] = None  # noqa
        self.val_ds: Union[ClassificationDataset, SegmentationDataSet] = None  # noqa

        self.train_dl: DataLoader = None  # noqa
        self.val_dl: DataLoader = None  # noqa

        self.train_tracker: Union[ClsTrainTracker, SegTrainTracker] = None  # noqa
        self.val_tracker: Union[ClsValTracker, SegValTracker] = None  # noqa

    def init_model(self) -> None:
        num_classes = 0
        mask_classes = 0
        if self.task.CLS or self.task.MT:
            num_classes = len(CONFIG('classification.labels'))
        if self.task.SEG or self.task.MT:
            mask_classes = len(CONFIG('segmentation.labels'))

        self.model = Model(
            model_name=CONFIG('model'),
            num_classes=num_classes,
            mask_classes=mask_classes,
            pretrained=CONFIG("pretrained"),
            weight=CONFIG('weight'),
            device=CONFIG('device')
        )
        self.model.init()

    def init_optimizer(self) -> None:
        name: str = CONFIG("optimizer")

        if name.upper() == 'AUTO':
            name = DEFAULT_OPTIMIZER

        args = {
            "params": [{
                'params': self.model.parameters,
                'initial_lr': CONFIG('lr0')
            }],
            "lr": CONFIG("lr0"),
        }

        if name in ["Adam", "Adamax", "AdamW", "NAdam", "RAdam"]:
            args.update({
                'betas': (CONFIG('momentum'), 0.999),
                'weight_decay': 0.0
            })
        elif name == "RMSProp":
            args.update({
                'momentum': CONFIG('momentum')
            })
        elif name == "SGD":
            args.update({
                'momentum': CONFIG('momentum'),
                'nesterov': True
            })

        if CONFIG("amp"):
            self.optimizer = build_amp_optimizer_wrapper(name, **args)
            logger.info('AMP: Open Automatic Mixed Precision(AMP)')
        else:
            self.optimizer = build_optimizer_wrapper(name, **args)

        logger.info(f'Build Optim: {name}.')

    def init_lr_scheduler(self) -> None:
        self.lr_scheduler = LRSchedulerWrapper(
            self.optimizer.optimizer,
            lrf=CONFIG('lrf'),
            epochs=CONFIG('epochs'),
            cos_lr=CONFIG('cos_lr')
        )

    def to_device(self, data: torch.Tensor) -> torch.Tensor:
        if self.model.is_gpu:
            return data.cuda(self.model.device, non_blocking=True)
        else:
            return data


class ClassificationTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.train_tracker = ClsTrainTracker(topk=np.argmax(CONFIG('topk')))  # noqa
        self.val_tracker = ClsValTracker(topk=np.argmax(CONFIG('topk')))  # noqa
        self.labels = Labels(CONFIG('classification.labels'))

    def init_loss(self) -> None:

        alpha = CONFIG('alpha')
        if alpha == 'auto':
            # alpha = [1] * self.model.num_classes
            alpha = [1] * self.labels.size

        alpha = torch.tensor(alpha, dtype=torch.float)
        alpha = self.to_device(alpha)

        self.loss = ClassificationLoss(alpha=alpha, gamma=CONFIG('gamma'))
        logger.success('Init Classification Loss.')

    def init_ds_dl(self) -> None:
        wh = tuple(CONFIG('wh'))
        workers: int = CONFIG('workers')
        use_cache: bool = CONFIG('cache')
        bs: int = CONFIG('classification.batch')

        # Build Train Dataset --------------------------------------------------------------------------------------
        self.train_ds = ClassificationDataset(
            root=CONFIG('classification.train'),
            wh=wh,
            labels=self.labels,
            transform=ClsImageT(wh),
            target_transform=ClsTargetT(),
            cache=use_cache
        )
        logger.success('Init classification train dataset.')

        self.val_ds = ClassificationDataset(
            root=CONFIG('classification.val'),
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
        if bs < self.labels.size:
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
            batch_size=bs if bs < self.labels.size else 1,
            num_workers=workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            shuffle=True if bs < self.labels.size else False,
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

        topk: List[float] = topk_accuracy(outputs, targets, CONFIG('topk'))

        maxk = max(CONFIG("topk"))
        maxk_idx = np.argmax(CONFIG("topk"))

        top1_val = topk[0]
        topk_val = topk[maxk_idx]

        self.train_tracker.top1.add(top1_val)
        self.train_tracker.topk.add(topk_val)
        self.train_tracker.loss.add(loss.cpu().detach())

        log_metric('Train Batch Top1', top1_val)
        log_metric(f'Train Batch Top{maxk}', topk_val)

        return loss

    def val(self) -> None:

        maxk: int = max(CONFIG("topk"))
        maxk_idx = np.argmax(CONFIG("topk"))

        confusion_matrix = 0

        for data in self.val_dl:
            images, targets = data
            images = self.to_device(images)
            targets = self.to_device(targets)

            output = self.model(images)  # [[cls1,cls2],[seg1,seg2,...]]

            confusion_matrix += compute_confusion_matrix_classification(output, targets, self.labels.size)
            topk: List[float] = topk_accuracy(output, targets, CONFIG('topk'))

            top1_val = topk[0]
            topk_val = topk[maxk_idx]

            self.val_tracker.top1.add(top1_val)
            self.val_tracker.topk.add(topk_val)

        draw_confusion_matrix(
            confusion_matrix,
            self.labels.labels,
            os.path.join(CONFIG('experiment_path'), 'cls_confusion_matrix.png')
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
            'num_classes': self.labels.size,
            'optimizer': convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            'lr': self.optimizer.lrs[0]
        }
        save_path = os.path.join(CONFIG('weight_path'), f'epoch{self.epoch}.pth')
        torch.save(save_dict, save_path)


class SegmentationTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.train_tracker = SegTrainTracker()
        self.val_tracker = SegValTracker()
        self.labels = Labels(CONFIG('segmentation.labels'))

    def init_ds_dl(self) -> None:
        wh = tuple(CONFIG('wh'))
        workers: int = CONFIG('workers')
        use_cache: bool = CONFIG('cache')
        bs: int = CONFIG('segmentation.batch')

        self.train_ds = SegmentationDataSet(
            root=CONFIG('segmentation.train'),
            wh=wh,
            labels=self.labels,
            transform=SegImageT(wh),
            cache=use_cache
        )
        logger.success('Init segmentation train dataset.')

        self.val_ds = SegmentationDataSet(
            root=CONFIG('segmentation.val'),
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
        self.loss = SegmentationLoss(CONFIG('seg_loss_sum_weights'))
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

        miou: float = compute_iou(outputs[0], targets, self.labels.size)

        self.train_tracker.miou.add(miou)
        self.train_tracker.loss.add(loss.cpu().detach())  # noqa
        log_metric('Train Batch MIoU', miou)

        return loss

    def val(self) -> None:
        confusion_matrix = 0

        for data in self.val_dl:
            images, targets = data
            images = self.to_device(images)
            targets = self.to_device(targets)  # target.shape=(N,1,H,W)

            output = self.model(images)

            miou: float = compute_iou(output[0], targets, self.labels.size)
            self.val_tracker.miou.add(miou)

            confusion_matrix += compute_confusion_matrix_segmentation(output[0], targets, self.labels.size)

        draw_confusion_matrix(
            confusion_matrix,
            self.labels.labels,
            os.path.join(CONFIG('experiment_path'), 'seg_confusion_matrix.png')
        )
        total_miou: float = self.val_tracker.miou.avg
        log_metric('Val Epoch MIoU', total_miou)

    def save_model(self) -> None:

        save_dict = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict,
            'model_name': self.model.model_name,
            'mask_classes': self.labels.size,
            'optimizer': convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            'lr': self.optimizer.lrs[0]
        }
        save_path = os.path.join(CONFIG('weight_path'), f'epoch{self.epoch}.pth')
        torch.save(save_dict, save_path)


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
                final_loss = loss_sum([cls_loss, seg_loss], CONFIG('loss_sum_weights'))
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
            'num_classes': self.cls_trainer.labels.size,
            'mask_classes': self.seg_trainer.labels.size,
            'optimizer': convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            'lr': self.optimizer.lrs[0]
        }
        save_path = os.path.join(CONFIG('weight_path'), f'epoch{self.epoch}.pth')
        torch.save(save_dict, save_path)


class Trainer:
    def __init__(self):

        if CONFIG('task').lower() == 'classification':
            self.trainer = ClassificationTrainer()
        elif CONFIG('task').lower() == 'segmentation':
            self.trainer = SegmentationTrainer()
        elif CONFIG('task').lower() == 'multitask':
            self.trainer = MultiTaskTrainer()

        self.trainer.init_model()
        self.trainer.init_loss()
        self.trainer.init_ds_dl()
        self.trainer.init_optimizer()
        self.trainer.init_lr_scheduler()

    def run(self) -> None:
        while self.trainer.epoch < CONFIG('epochs'):
            for mode in ['train', 'val']:

                if mode == 'train':
                    self.trainer.train()
                    self.trainer.epoch += 1

                    if self.trainer.epoch % CONFIG('save_period') == 0:
                        self.trainer.save_model()
                    log_metric('Epoch', self.trainer.epoch)

                else:
                    if CONFIG('not_val') is True:
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

                    print_of_mt(mode, 'MT', self.trainer.epoch, CONFIG('epochs'), cls_loss, seg_loss, lr, top1, topk,
                                miou)

                elif self.trainer.task.CLS:
                    cls_loss: float = round4(self.trainer.train_tracker.loss.avg) if mode == 'train' else None
                    top1 = round4(
                        self.trainer.train_tracker.top1.avg if mode == 'train' else self.trainer.val_tracker.top1.avg)
                    topk = round4(
                        self.trainer.train_tracker.topk.avg if mode == 'train' else self.trainer.val_tracker.topk.avg)

                    print_of_cls(mode, 'CLS', self.trainer.epoch, CONFIG('epochs'), cls_loss, lr, top1, topk, )

                elif self.trainer.task.SEG:
                    seg_loss: float = round4(self.trainer.train_tracker.loss.avg) if mode == 'train' else None
                    miou = round4(
                        self.trainer.train_tracker.miou.avg if mode == 'train' else self.trainer.val_tracker.miou.avg)

                    print_of_seg(mode, 'SEG', self.trainer.epoch, CONFIG('epochs'), seg_loss, lr, miou)

                self.trainer.train_tracker.reset()
                self.trainer.val_tracker.reset()
