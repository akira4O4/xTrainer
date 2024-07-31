import math
import os
from typing import Union, List, Optional, Any
from copy import deepcopy
import numpy as np
import torch
from loguru import logger
from mlflow import log_metric, set_experiment
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainerx.core.lr_scheduler import LRSchedulerWrapper
from trainerx.core.preprocess import (
    ClsImageT,
    ClsTargetT,
    ClsValT,
    SegImageT,
    SegValT
)

from trainerx import (
    CONFIG,
    DEFAULT_OPTIMIZER
)

from trainerx.core.model import Model
from trainerx.core.optim import (
    AMPOptimWrapper,
    OptimWrapper,
    build_optimizer_wrapper,
    build_amp_optimizer_wrapper
)
from trainerx.dataset.classification import ClassificationDataset, BalancedBatchSampler
from trainerx.dataset.segmentation import SegmentationDataSet
from trainerx.utils.common import (
    save_yaml,
    error_exit,
    round4,
    round8,
    timer,
    check_dir,
    align_size,
    get_time,
    print_of_mt,
    print_of_seg,
    print_of_cls
)
from trainerx.utils.data_tracker import (
    TrainTracker,
    ValTracker,
    LossTracker
)
# from trainerx.utils.perf import calc_performance
from trainerx.utils.task import Task
from trainerx.core.loss import ClassificationLoss, SegmentationLoss
from trainerx.utils.torch_utils import (
    init_seeds,
    init_backends_cudnn,
    convert_optimizer_state_dict_to_fp16
)
from trainerx.utils.perf import (
    topk_accuracy,
    mean_iou_v1
)


class Trainer:
    def __init__(self):

        self.epoch = 0

        self.weight_path: str = ''  # project/weight
        self.experiment_path: str = ''  # project/experiment
        self.init_workspace()

        # Init Data Logger ---------------------------------------------------------------------------------------------
        self.train_tracker = TrainTracker(topk=np.argmax(CONFIG('topk')))  # noqa
        self.val_tracker = ValTracker(topk=np.argmax(CONFIG('topk')))  # noqa
        self.loss_tracker = LossTracker()

        # Init work env ------------------------------------------------------------------------------------------------
        init_seeds(CONFIG('seed'))
        logger.info(f'Init seed:{CONFIG("seed")}.')

        init_backends_cudnn(CONFIG('deterministic'))
        logger.info(f'Init deterministic:{CONFIG("deterministic")}.')
        logger.info(f'Init benchmark:{not CONFIG("deterministic")}.')

        self.task = Task(CONFIG('task'))
        logger.info(f"Task: {self.task}")

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
        self.loss_weights: List[int] = CONFIG('loss_weights')
        self.init_loss()

        # Init dataset and dataloader ---------------------------------------------------------------------------------
        if self.task.CLS or self.task.MT:
            self.cls_train_ds: ClassificationDataset = None  # noqa
            self.cls_train_dl: DataLoader = None  # noqa

            self.cls_val_ds: ClassificationDataset = None  # noqa
            self.cls_val_dl: DataLoader = None  # noqa

            if not os.path.exists(CONFIG('classification')['train']): ...
            if not os.path.exists(CONFIG('classification')['val']): ...

            self.build_classification_ds_dl()

            if CONFIG('classification')['classes'] != self.cls_train_ds.num_of_label:
                logger.error('classification num of classes setting error.')
                error_exit()

        if self.task.SEG or self.task.MT:

            self.seg_train_ds: SegmentationDataSet = None  # noqa
            self.seg_train_dl: DataLoader = None  # noqa

            self.seg_val_ds: SegmentationDataSet = None  # noqa
            self.seg_val_dl: DataLoader = None  # noqa

            self.build_segmentation_ds_dl()

            if CONFIG('segmentation')['classes'] != self.seg_train_ds.num_of_label:
                logger.error('segmentation num of classes setting error.')
                error_exit()

        # Expand dataset -----------------------------------------------------------------------------------------------
        if self.task.MT:
            rate1, rate2 = align_size(self.cls_train_ds.real_data_size, self.seg_train_ds.real_data_size)
            self.cls_train_ds.expanding_data(rate1)
            self.seg_train_ds.expanding_data(rate2)
            logger.info(f'Expanding classification dataset to: {self.cls_train_ds.real_data_size}x{rate1}')
            logger.info(f'Expanding segmentation dataset to: {self.seg_train_ds.real_data_size}x{rate2}')

        if self.task.MT:
            self.total_step = min(len(self.cls_train_dl), len(self.seg_train_dl))
        elif self.task.CLS:
            self.total_step = len(self.cls_train_dl)
        elif self.task.SEG:
            self.total_step = len(self.seg_train_dl)

        # Init MLFlow  -------------------------------------------------------------------------------------------------
        self.init_mlflow()
        logger.info('请使用MLFlow UI进行训练数据观察 -> [Terminal]: mlflow ui')

    def init_workspace(self) -> None:
        """
        project
            - experiment1
                - cls_labels.txt
                - weights
            - experiment2
                - seg_labels.txt
                - weights
        """
        # assert os.path.isdir(CONFIG('project')) is True, 'args.project must be dir.'

        check_dir(CONFIG('project'))

        self.experiment_path = os.path.join(CONFIG('project'), CONFIG('experiment'))
        if os.path.exists(self.experiment_path):
            self.experiment_path = os.path.join(CONFIG('project'), CONFIG('experiment') + get_time())

        check_dir(self.experiment_path)

        self.weight_path = os.path.join(self.experiment_path, 'weights')
        check_dir(self.weight_path)

    @staticmethod
    def init_mlflow() -> None:
        if CONFIG('mlflow_experiment_name') == '':
            logger.info(f'MLFlow Experiment Name: Default.')
        else:
            set_experiment(CONFIG('mlflow_experiment_name'))
            logger.info(f'MLFlow Experiment Name:{CONFIG("exp_name")}.')

    def init_model(self) -> None:
        num_classes: int = CONFIG('classification')['classes']
        mask_classes: int = CONFIG('segmentation')['classes']

        if num_classes == mask_classes == 0:
            logger.error("num_classes == mask_classes == 0")
            error_exit()

        self.model = Model(
            CONFIG('model'),
            num_classes,
            mask_classes,
            CONFIG("pretrained"),
            CONFIG('weight'),
            CONFIG('device')
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

        logger.success(f'Build Optim: {name}.')

    def init_lr_scheduler(self) -> None:
        self.lr_scheduler = LRSchedulerWrapper(
            self.optimizer.optimizer,
            lrf=CONFIG('lrf'),
            epochs=CONFIG('epochs'),
            cos_lr=CONFIG('cos_lr')
        )

    def build_classification_ds_dl(self) -> None:

        # Build Train Dataset --------------------------------------------------------------------------------------
        self.cls_train_ds = ClassificationDataset(
            root=CONFIG('classification')['train'],
            wh=CONFIG('wh'),
            transform=ClsImageT(tuple(CONFIG('wh'))),
            target_transform=ClsTargetT(),
        )

        save_yaml(self.cls_train_ds.labels, os.path.join(self.experiment_path, 'cls_labels.yaml'))

        # Build BalancedBatchSampler -------------------------------------------------------------------------------
        balanced_batch_sampler = None
        if CONFIG('balanced_batch_sampler'):
            balanced_batch_sampler = BalancedBatchSampler(
                torch.tensor(self.cls_train_ds.targets),
                n_classes=self.model.num_classes,
                n_samples=math.ceil(CONFIG('classification')['batch'] / self.model.num_classes)
            )

        # Build Train DataLoader -----------------------------------------------------------------------------------
        self.cls_train_dl = DataLoader(
            dataset=self.cls_train_ds,
            batch_size=1 if CONFIG('balanced_batch_sampler') else CONFIG('classification')['batch'],
            num_workers=CONFIG('workers'),
            pin_memory=True,
            batch_sampler=balanced_batch_sampler,
            shuffle=False if CONFIG('balanced_batch_sampler') else True,
            drop_last=False,
            sampler=None
        )
        logger.info(f'Classification num of labels: {self.cls_train_ds.num_of_label}.')
        logger.info(f'Classification Train data size: {self.cls_train_ds.real_data_size}.')

        # Build Val Dataset --------------------------------------------------------------------------------------------
        self.cls_val_ds = ClassificationDataset(
            root=CONFIG('classification')['val'],
            wh=CONFIG('wh'),
            transform=ClsValT(tuple(CONFIG('wh'))),
            target_transform=ClsTargetT(),
        )
        logger.info(f'Classification Val data size: {self.cls_val_ds.real_data_size}.')

        # Build Val DataLoader -----------------------------------------------------------------------------------------
        self.cls_val_dl = DataLoader(
            dataset=self.cls_val_ds,
            batch_size=CONFIG('classification')['batch'],
            num_workers=CONFIG('workers'),
            pin_memory=True,
            shuffle=False
        )

    def build_segmentation_ds_dl(self) -> None:

        self.seg_train_ds = SegmentationDataSet(
            root=CONFIG('segmentation')['train'],
            wh=CONFIG('wh'),
            transform=SegImageT(tuple(CONFIG('wh'))),
        )

        background_size = len(self.seg_val_ds.background_samples)

        save_yaml(
            self.seg_train_ds.labels,
            os.path.join(self.experiment_path, 'seg_labels.yaml')
        )
        self.seg_train_dl = DataLoader(
            dataset=self.seg_train_ds,
            batch_size=CONFIG('batch'),
            shuffle=False,
            num_workers=CONFIG('workers'),
            pin_memory=True,
        )

        logger.info(f'Segmentation num of labels: {self.seg_train_ds.num_of_label}.')
        logger.info(
            f'Segmentation Train data size: {self.seg_train_ds.real_data_size} (background:{background_size}).')

        self.seg_val_ds = SegmentationDataSet(
            root=CONFIG('segmentation')['val'],
            wh=CONFIG('wh'),
            transform=SegValT(tuple(CONFIG('wh'))),
        )
        self.seg_val_dl = DataLoader(
            dataset=self.seg_val_ds,
            batch_size=CONFIG('batch'),
            shuffle=False,
            num_workers=CONFIG('workers'),
            pin_memory=True,
        )

        logger.info(f'Segmentation Val data size: {self.seg_val_ds.real_data_size}.')

    def init_loss(self) -> None:

        if self.task.CLS or self.task.MT:
            alpha = CONFIG('alpha') if CONFIG('alpha') is not None else [1] * self.model.num_classes
            alpha = torch.tensor(alpha, dtype=torch.float)
            alpha = self.to_device(alpha)
            self.classification_loss = ClassificationLoss(alpha=alpha, gamma=CONFIG('gamma'))
            logger.info('Build Classification Loss.')

        if self.task.SEG or self.task.MT:
            self.segmentation_loss = SegmentationLoss()
            logger.info('Build Segmentation Loss.')

    def to_device(self, data: torch.Tensor) -> torch.Tensor:
        if self.model.is_gpu:
            return data.cuda(self.model.device, non_blocking=True)
        else:
            return data

    def run(self) -> None:
        while self.epoch < CONFIG('epochs'):

            # n*train -> k*val -> n*train->...
            for wf in CONFIG('workflow'):
                mode, times = wf
                run_one_epoch = getattr(self, mode)

                for _ in range(times):
                    if self.epoch >= CONFIG('epochs'):
                        break

                    run_one_epoch()  # train() or val()

                    if mode == 'train':
                        self.epoch += 1
                        log_metric('Epoch', self.epoch)
                        if self.task.MT:
                            print_of_mt(
                                self.epoch,
                                CONFIG('epochs'),
                                round4(self.loss_tracker.classification.avg),
                                round4(self.loss_tracker.segmentation.avg),
                                round4(self.train_tracker.top1.avg),
                                round4(self.train_tracker.topk.avg),
                                round4(self.train_tracker.miou.avg)
                            )
                        elif self.task.CLS:
                            print_of_cls(
                                self.epoch,
                                CONFIG('epochs'),
                                round4(self.loss_tracker.classification.avg),
                                round4(self.train_tracker.top1.avg),
                                round4(self.train_tracker.topk.avg),
                            )
                        elif self.task.SEG:
                            print_of_seg(
                                self.epoch,
                                CONFIG('epochs'),
                                round4(self.loss_tracker.segmentation.avg),
                                round4(self.train_tracker.miou.avg)
                            )

                        self.train_tracker.reset()


                    elif mode == 'val':
                        ...

    # @timer
    def train(self) -> None:

        self.model.train()

        # self.curr_lr = round8(sum(self.optimizer.lrs) / len(self.optimizer.lrs))
        # log_metric('Lr', self.curr_lr)

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

            final_loss = self.loss_sum([cls_loss, seg_loss])

            # 1.loss backward
            # 2.optimizer step
            # 3.optimizer zero_grad

            with self.optimizer.context() as opt:
                # self.optimizer.update(final_loss)
                opt.update(final_loss)

        self.lr_scheduler.update()

            # if self.scheduler_step_in_batch is True:
            #     self.lr_scheduler.step(CONFIG('epochs') + curr_step / self.total_step)

    def _classification_train(self, images: torch.Tensor, targets: torch.Tensor):
        with self.optimizer.context():
            pred = self.model(images)
            if self.task.MT:
                pred = pred[0][0]  # [[cls1,cls2],[seg1,seg2,...]]
            loss = self.classification_loss(pred, targets)  # noqa

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

    def _segmentation_train(self, images: torch.Tensor, targets: torch.Tensor):
        with self.optimizer.context():
            pred = self.model(images)

            if self.task.MT:
                pred = pred[1][0]  # [[cls1,cls2],[seg1,seg2,...]]
            elif self.task.SEG:
                pred = pred[0]  # [seg1,seg2,...]

            loss = self.segmentation_loss(pred, targets)  # noqa

        miou: float = mean_iou_v1(pred, targets, self.model.mask_classes)
        self.train_tracker.miou.add(miou)
        self.loss_tracker.segmentation.add(loss.cpu().detach())

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

        for data in tqdm(self.cls_val_dl):
            images, targets = data
            images = self.to_device(images)
            targets = self.to_device(targets)

            pred = self.model(images)

            if self.task.MT:
                pred = pred[0][0]  # [[cls1,cls2],[seg1,seg2,...]]

            topk: List[float] = topk_accuracy(pred, targets, CONFIG('topk'))

            top1_val = topk[0]
            topk_val = topk[maxk_idx]

            self.val_tracker.top1.add(top1_val)
            self.val_tracker.topk.add(topk_val)

        total_top1: float = self.val_tracker.top1.avg  # i.e.60%
        total_topk: float = self.val_tracker.topk.avg  # i.e.80%
        log_metric('Val Epoch Top1', total_top1)
        log_metric(f'Val Epoch Top{maxk}', total_topk)

    def _segmentation_val(self) -> None:
        for data in tqdm(self.seg_val_dl):
            images, targets = data
            images = self.to_device(images)
            targets = self.to_device(targets)

            pred = self.model(images)

            if self.task.MT:
                pred = pred[1][0]  # [[cls1,cls2],[seg1,seg2,...]]
            elif self.task.SEG:
                pred = pred[0]  # [seg1,seg2,...]

            miou: float = mean_iou_v1(pred, targets, self.model.mask_classes)
            self.val_tracker.miou.add(miou)

        total_miou: float = self.val_tracker.miou.avg
        log_metric('Val Epoch MIoU', total_miou)

    def save_model(self, save_path: str) -> None:

        save_dict = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict,
            'model_name': self.model.model_name,
            'num_classes': self.model.num_classes,
            'mask_classes': self.model.mask_classes,
            'optimizer': convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict()))
        }
        torch.save(save_dict, save_path)
        logger.success(f'👍 Save weight to: {save_path}.')

    def loss_sum(self, losses: List[torch.Tensor]) -> torch.Tensor:

        if len(losses) != len(self.loss_weights):
            logger.error('len(loss_result)!=len(self.losses_weights)')

        ret = torch.tensor(0.0, dtype=losses[0].dtype, device=losses[0].device)

        for loss, weight in zip(losses, self.loss_weights):
            ret += loss * weight

        return ret  # noqa
