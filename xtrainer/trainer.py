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
    align_size,
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
    TrainTracker,
    ValTracker,
    LossTracker
)
from xtrainer.utils.torch_utils import (
    init_seeds,
    init_backends_cudnn,
    loss_sum,
    convert_optimizer_state_dict_to_fp16
)


class ClassificationTrainer:
    def __init__(self):
        ...


class SegmentationTrainer:
    def __init__(self):
        ...


class MultiTaskTrainer:
    def __init__(self):
        ...


class Trainer:
    def __init__(self):

        self.epoch = 0

        # Init Data Logger ---------------------------------------------------------------------------------------------
        self.train_tracker = TrainTracker(topk=np.argmax(CONFIG('topk')))  # noqa
        self.val_tracker = ValTracker(topk=np.argmax(CONFIG('topk')))  # noqa
        self.loss_tracker = LossTracker()

        # Init work env ------------------------------------------------------------------------------------------------
        init_seeds(CONFIG('seed'))
        logger.info(f'Init seed: {CONFIG("seed")}.')

        init_backends_cudnn(CONFIG('deterministic'))
        logger.info(f'Init deterministic: {CONFIG("deterministic")}.')
        logger.info(f'Init benchmark: {not CONFIG("deterministic")}.')

        self.task = Task(CONFIG('task'))
        logger.info(f"Task: {self.task}")

        # Init dataset and dataloader ---------------------------------------------------------------------------------
        self.cls_labels: Labels = None  # noqa
        self.seg_labels: Labels = None  # noqa

        if self.task.CLS or self.task.MT:
            self.cls_train_ds: ClassificationDataset = None  # noqa
            self.cls_train_dl: DataLoader = None  # noqa

            self.cls_val_ds: ClassificationDataset = None  # noqa
            self.cls_val_dl: DataLoader = None  # noqa

            self.cls_labels = Labels(CONFIG('classification.labels'))
            self.build_classification_ds_dl()

        if self.task.SEG or self.task.MT:
            self.seg_train_ds: SegmentationDataSet = None  # noqa
            self.seg_train_dl: DataLoader = None  # noqa

            self.seg_val_ds: SegmentationDataSet = None  # noqa
            self.seg_val_dl: DataLoader = None  # noqa

            self.seg_labels = Labels(CONFIG('segmentation.labels'))
            self.build_segmentation_ds_dl()

        # Expand dataset -----------------------------------------------------------------------------------------------
        if self.task.MT:
            rate1, rate2 = align_size(self.cls_train_ds.real_data_size, self.seg_train_ds.real_data_size)
            self.cls_train_ds.expand_data(rate1)
            self.seg_train_ds.expand_data(rate2)
            logger.info(f'Expanding classification dataset to: {self.cls_train_ds.real_data_size}x{rate1}')
            logger.info(f'Expanding segmentation dataset to: {self.seg_train_ds.real_data_size}x{rate2}')

        if self.task.MT:
            self.total_step = min(len(self.cls_train_dl), len(self.seg_train_dl))
        elif self.task.CLS:
            self.total_step = len(self.cls_train_dl)
        elif self.task.SEG:
            self.total_step = len(self.seg_train_dl)

        # Init Model --------------------------------------------------------------------------------------------------
        self.model: Model = None  # noqa
        self.init_model()

        # Init Optimizer ----------------------------------------------------------------------------------------------
        self.optimizer: Union[OptimWrapper, AMPOptimWrapper] = None  # noqa
        self.init_optimizer()

        # Init Lr Scheduler -------------------------------------------------------------------------------------------
        self.curr_lr = 0
        self.lr_scheduler: LRSchedulerWrapper = None  # noqa
        self.scheduler_step_in_batch = False
        self.init_lr_scheduler()

        # Init loss ---------------------------------------------------------------------------------------------------
        self.classification_loss = None
        self.segmentation_loss = None
        self.init_loss()

        logger.info('请使用MLFlow UI进行训练数据观察 -> [Terminal]: mlflow ui')

    def init_model(self) -> None:
        self.model = Model(
            model_name=CONFIG('model'),
            num_classes=self.cls_labels.size if self.cls_labels is not None else 0,
            mask_classes=self.seg_labels.size if self.seg_labels is not None else 0,
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

    def build_classification_ds_dl(self) -> None:
        wh = tuple(CONFIG('wh'))
        workers: int = CONFIG('workers')
        use_cache: bool = CONFIG('cache')
        bs: int = CONFIG('classification.batch')

        # Build Train Dataset --------------------------------------------------------------------------------------
        self.cls_train_ds = ClassificationDataset(
            root=CONFIG('classification.train'),
            wh=wh,
            labels=self.cls_labels,
            transform=ClsImageT(wh),
            target_transform=ClsTargetT(),
            cache=use_cache
        )
        self.cls_val_ds = ClassificationDataset(
            root=CONFIG('classification.val'),
            wh=wh,
            labels=self.cls_labels,
            transform=ClsValT(wh),
            target_transform=ClsTargetT(),
            cache=use_cache
        )

        logger.info(f'Classification Train data size: {self.cls_train_ds.real_data_size}.')
        logger.info(f'Classification Val data size: {self.cls_val_ds.real_data_size}.')

        batch_sampler = None
        if bs < self.cls_labels.size:
            logger.info('Close BalancedBatchSampler.')
        else:
            logger.info('Open BalancedBatchSampler')
            batch_sampler = BalancedBatchSampler(
                self.cls_train_ds.targets,
                batch_size=bs
            )

        # Build Train DataLoader -----------------------------------------------------------------------------------
        self.cls_train_dl = DataLoader(
            dataset=self.cls_train_ds,
            batch_size=bs if bs < self.cls_labels.size else 1,
            num_workers=workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            shuffle=True if bs < self.cls_labels.size else False,
            drop_last=False,
            sampler=None
        )
        self.cls_val_dl = DataLoader(
            dataset=self.cls_val_ds,
            batch_size=bs,
            num_workers=workers,
            pin_memory=True,
            shuffle=False
        )

    def build_segmentation_ds_dl(self) -> None:
        wh = tuple(CONFIG('wh'))
        workers: int = CONFIG('workers')
        use_cache: bool = CONFIG('cache')
        bs: int = CONFIG('segmentation.batch')

        self.seg_train_ds = SegmentationDataSet(
            root=CONFIG('segmentation.train'),
            wh=wh,
            labels=self.seg_labels,
            transform=SegImageT(wh),
            cache=use_cache
        )
        self.seg_val_ds = SegmentationDataSet(
            root=CONFIG('segmentation.val'),
            wh=wh,
            labels=self.seg_labels,
            transform=SegValT(wh),
            cache=use_cache
        )

        background_size = len(self.seg_train_ds.background_samples)
        logger.info(
            f'Segmentation Train data size: {self.seg_train_ds.real_data_size} (background:{background_size}).')

        self.seg_train_dl = DataLoader(
            dataset=self.seg_train_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        self.seg_val_dl = DataLoader(
            dataset=self.seg_val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )

        logger.info(f'Segmentation Val data size: {self.seg_val_ds.real_data_size}.')

    def init_loss(self) -> None:

        if self.task.CLS or self.task.MT:
            alpha = CONFIG('alpha')
            if alpha == 'auto':
                # alpha = [1] * self.model.num_classes
                alpha = [1] * self.cls_labels.size

            alpha = torch.tensor(alpha, dtype=torch.float)
            alpha = self.to_device(alpha)

            self.classification_loss = ClassificationLoss(alpha=alpha, gamma=CONFIG('gamma'))
            logger.info('Build Classification Loss.')

        if self.task.SEG or self.task.MT:
            self.segmentation_loss = SegmentationLoss(CONFIG('seg_loss_sum_weights'))
            logger.info('Build Segmentation Loss.')

    def to_device(self, data: torch.Tensor) -> torch.Tensor:
        if self.model.is_gpu:
            return data.cuda(self.model.device, non_blocking=True)
        else:
            return data

    def run(self) -> None:
        while self.epoch < CONFIG('epochs'):
            for mode in ['train', 'val']:
                run_one_epoch = getattr(self, mode)

                run_one_epoch()

                if mode == 'train':
                    self.epoch += 1
                    if self.epoch % CONFIG('save_period') == 0:
                        self.save_model()

                    log_metric('Epoch', self.epoch)

                else:
                    if CONFIG('not_val') is True:
                        continue

                lr: float = round8(self.optimizer.lrs[0]) if mode == 'train' else None

                # Display info
                if self.task.MT:
                    cls_loss: float = round4(self.loss_tracker.classification.avg) if mode == 'train' else None
                    seg_loss: float = round4(self.loss_tracker.segmentation.avg) if mode == 'train' else None
                    top1 = round4(self.train_tracker.top1.avg if mode == 'train' else self.val_tracker.top1.avg)
                    topk = round4(self.train_tracker.topk.avg if mode == 'train' else self.val_tracker.topk.avg)
                    miou = round4(self.train_tracker.miou.avg if mode == 'train' else self.val_tracker.miou.avg)

                    print_of_mt(mode, 'MT', self.epoch, CONFIG('epochs'), cls_loss, seg_loss, lr, top1, topk,
                                miou)

                elif self.task.CLS:
                    cls_loss: float = round4(self.loss_tracker.classification.avg) if mode == 'train' else None
                    top1 = round4(self.train_tracker.top1.avg if mode == 'train' else self.val_tracker.top1.avg)
                    topk = round4(self.train_tracker.topk.avg if mode == 'train' else self.val_tracker.topk.avg)

                    print_of_cls(mode, 'CLS', self.epoch, CONFIG('epochs'), cls_loss, lr, top1, topk, )

                elif self.task.SEG:
                    seg_loss: float = round4(self.loss_tracker.segmentation.avg) if mode == 'train' else None
                    miou = round4(self.train_tracker.miou.avg if mode == 'train' else self.val_tracker.miou.avg)

                    print_of_seg(mode, 'SEG', self.epoch, CONFIG('epochs'), seg_loss, lr, miou)

                self.train_tracker.reset()
                self.val_tracker.reset()
                self.loss_tracker.reset()

    @timer
    def train(self) -> None:

        self.model.train()

        dataloaders: List[DataLoader] = []
        if self.task.CLS or self.task.MT:
            dataloaders.append(self.cls_train_dl)
        if self.task.SEG or self.task.MT:
            dataloaders.append(self.seg_train_dl)

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
                cls_loss = self._classification_train(images, targets)

            if self.task.SEG or self.task.MT:
                images, targets = seg_data
                images = self.to_device(images)
                targets = self.to_device(targets)
                seg_loss = self._segmentation_train(images, targets)

            if self.task.MT:
                final_loss = loss_sum([cls_loss, seg_loss], CONFIG('loss_sum_weights'))
            else:
                final_loss = cls_loss + seg_loss

            with self.optimizer.context() as opt:
                # self.optimizer.update(final_loss)
                opt.update(final_loss)

        self.lr_scheduler.update()

    def _classification_train(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0
        with self.optimizer.context():

            # multitask output=[[x1,x2],[x1,x2,x3,x4]]
            # classification output=x
            outputs = self.model(images)

            if self.task.MT:
                preds = outputs[0]  # [[cls1,cls2],[seg1,seg2,...]]
                for pred in preds:
                    loss += 0.5 * self.classification_loss(pred, targets)  # noqa
            else:
                loss = self.classification_loss(outputs, targets)  # noqa

        if self.task.MT:
            pred = outputs[0][0]
        else:
            pred = outputs

        topk: List[float] = topk_accuracy(pred, targets, CONFIG('topk'))

        maxk = max(CONFIG("topk"))
        maxk_idx = np.argmax(CONFIG("topk"))

        top1_val = topk[0]
        topk_val = topk[maxk_idx]

        self.train_tracker.top1.add(top1_val)
        self.train_tracker.topk.add(topk_val)
        self.loss_tracker.classification.add(loss.cpu().detach())

        log_metric('Train Batch Top1', top1_val)
        log_metric(f'Train Batch Top{maxk}', topk_val)

        return loss

    def _segmentation_train(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with self.optimizer.context():
            # multitask output=[[x1,x2],[x1,x2,x3,x4]]
            # segmentation output=[x1,x2,x3,x4]
            outputs = self.model(images)

            preds = outputs[1] if self.task.MT else outputs

            loss1 = 1 * self.segmentation_loss(preds[0], targets)  # noqa
            loss2 = 1 * self.segmentation_loss(preds[1], targets)  # noqa
            loss3 = 0.5 * self.segmentation_loss(preds[2], targets)  # noqa
            loss4 = 0.5 * self.segmentation_loss(preds[3], targets)  # noqa

        loss = loss1 + loss2 + loss3 + loss4

        pred = outputs[1][0] if self.task.MT else outputs[0]

        # miou: float = compute_iou(pred, targets, self.model.mask_classes)
        miou: float = compute_iou(pred, targets, self.seg_labels.size)

        self.train_tracker.miou.add(miou)
        self.loss_tracker.segmentation.add(loss.cpu().detach())  # noqa
        log_metric('Train Batch MIoU', miou)

        return loss

    # @timer
    def val(self) -> None:
        self.model.eval()

        if self.task.SEG or self.task.MT:
            self._segmentation_val()

        if self.task.CLS or self.task.MT:
            self._classification_val()

    def _classification_val(self) -> None:

        maxk: int = max(CONFIG("topk"))
        maxk_idx = np.argmax(CONFIG("topk"))

        confusion_matrix = 0

        for data in self.cls_val_dl:
            images, targets = data
            images = self.to_device(images)
            targets = self.to_device(targets)

            output = self.model(images)  # [[cls1,cls2],[seg1,seg2,...]]

            pred = output[0][0] if self.task.MT else output

            # confusion_matrix += compute_confusion_matrix_classification(pred, targets, self.model.num_classes)
            confusion_matrix += compute_confusion_matrix_classification(pred, targets, self.cls_labels.size)
            topk: List[float] = topk_accuracy(pred, targets, CONFIG('topk'))

            top1_val = topk[0]
            topk_val = topk[maxk_idx]

            self.val_tracker.top1.add(top1_val)
            self.val_tracker.topk.add(topk_val)

        draw_confusion_matrix(
            confusion_matrix,
            self.cls_labels.labels,
            os.path.join(CONFIG('experiment_path'), 'cls_confusion_matrix.png')
        )
        total_top1: float = self.val_tracker.top1.avg  # i.e.60%
        total_topk: float = self.val_tracker.topk.avg  # i.e.80%
        log_metric('Val Epoch Top1', total_top1)
        log_metric(f'Val Epoch Top{maxk}', total_topk)

    def _segmentation_val(self) -> None:
        confusion_matrix = 0

        for data in self.seg_val_dl:
            images, targets = data
            images = self.to_device(images)
            targets = self.to_device(targets)  # target.shape=(N,1,H,W)

            output = self.model(images)

            # pred.shpae=(N,C,H,W)
            pred = output[1][0] if self.task.MT else output[0]

            # miou: float = compute_iou(pred, targets, self.model.mask_classes)
            miou: float = compute_iou(pred, targets, self.seg_labels.size)
            self.val_tracker.miou.add(miou)

            # confusion_matrix += compute_confusion_matrix_segmentation(pred, targets, self.model.mask_classes)
            confusion_matrix += compute_confusion_matrix_segmentation(pred, targets, self.seg_labels.size)

        draw_confusion_matrix(
            confusion_matrix,
            self.seg_labels.labels,
            os.path.join(CONFIG('experiment_path'), 'seg_confusion_matrix.png')
        )
        total_miou: float = self.val_tracker.miou.avg
        log_metric('Val Epoch MIoU', total_miou)

    def save_model(self) -> None:

        save_dict = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict,
            'model_name': self.model.model_name,
            'num_classes': self.cls_labels.size if self.cls_labels is not None else 0,
            'mask_classes': self.seg_labels.size if self.seg_labels is not None else 0,
            'optimizer': convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
            'lr': self.optimizer.lrs[0]
        }
        save_path = os.path.join(CONFIG('weight_path'), f'epoch{self.epoch}.pth')
        torch.save(save_dict, save_path)
