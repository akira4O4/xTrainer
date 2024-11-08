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
        self.task = Task(CONFIG['task'])

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
            num_classes = len(CONFIG['classification.labels'])
        if self.task.SEG or self.task.MT:
            mask_classes = len(CONFIG['segmentation.labels'])

        self.model = Model(
            model_name=CONFIG['model'],
            num_classes=num_classes,
            mask_classes=mask_classes,
            pretrained=CONFIG["pretrained"],
            weight=CONFIG['weight'],
            device=CONFIG['device']
        )
        self.model.init()

    def init_optimizer(self) -> None:
        name: str = CONFIG["optimizer"]

        if name.upper() == 'AUTO':
            name = DEFAULT_OPTIMIZER

        args = {
            "params": [{
                'params': self.model.parameters,
                'initial_lr': CONFIG['lr0']
            }],
            "lr": CONFIG["lr0"],
        }

        if name in ["Adam", "Adamax", "AdamW", "NAdam", "RAdam"]:
            args.update({
                'betas': (CONFIG['momentum'], 0.999),
                'weight_decay': 0.0
            })
        elif name == "RMSProp":
            args.update({
                'momentum': CONFIG['momentum']
            })
        elif name == "SGD":
            args.update({
                'momentum': CONFIG['momentum'],
                'nesterov': True
            })

        if CONFIG["amp"]:
            self.optimizer = build_amp_optimizer_wrapper(name, **args)
            logger.info('AMP: Open Automatic Mixed Precision(AMP)')
        else:
            self.optimizer = build_optimizer_wrapper(name, **args)

        logger.info(f'Build Optim: {name}.')

    def init_lr_scheduler(self) -> None:
        self.lr_scheduler = LRSchedulerWrapper(
            self.optimizer.optimizer,
            lrf=CONFIG['lrf'],
            epochs=CONFIG['epochs'],
            cos_lr=CONFIG['cos_lr']
        )

    def to_device(self, data: torch.Tensor) -> torch.Tensor:
        if self.model.is_gpu:
            return data.cuda(self.model.device, non_blocking=True)
        else:
            return data
