import os
import math
from typing import Union, List, Optional
import shutil
import csv

import numpy as np
import torch
from loguru import logger
from dataclasses import asdict
from torch.utils.data import DataLoader, Dataset

from utils.util import load_yaml, get_num_of_images, timer, get_time
from .task import Task, task_convert
from .args import ProjectArgs, TrainArgs, ModelArgs
from .builder import build_dir, init_seeds, init_backends_cudnn
from .builder import build_model, build_amp_optimizer_wrapper, build_loss, build_lr_scheduler
from .balanced_batch_sampler import BalancedBatchSampler
from .dataset import ClassificationDataset, SegmentationDataSet
from .transforms import ClassificationTransform, SegmentationTransform
from .model import Model
from .optim import AmpOptimWrapper
from .loss_forward import *
from .meter import Logger
from .performance import calc_performance


class Pipline:
    def __init__(self, config_path: str):

        # Check config path

        self.config_path = config_path
        if not os.path.exists(config_path):
            logger.error(f'Can`t found the {config_path}.')
            exit()

        # Init Args
        self.config = load_yaml(config_path)
        self.project_args = ProjectArgs(**self.config['project_config'])
        self.train_args = TrainArgs(**self.config['train_config'])
        self.model_args = ModelArgs(**self.config['model_config'])
        self.optimizer_args = self.config['optimizer_config']
        self.lr_args = self.config['lr_config']
        self.loss_args = self.config['loss_config']

        # Init workspace Env
        self.curr_exp_path: str = ''
        self.logs_dir = ''

        self.cls_loss_log_path = ''
        self.seg_loss_log_path = ''
        self.top1_log_path = ''
        self.miou_log_path = ''  # noqa
        self.weights_dir = ''
        self.performance_log_path = ''
        self.loss_log_path = ''
        self.performance_table_name = []
        self.loss_table_name = []
        self.init_workspace()

        init_seeds(self.train_args.seed)
        init_backends_cudnn(self.train_args.deterministic)
        self.task = task_convert(self.project_args.task)

        # Build Model
        self.model: Model = None  # noqa
        self.init_model()

        # Build Optimizer
        self.amp_optimizer_wrapper: AmpOptimWrapper = None  # noqa
        self.init_optimizer()

        # Build Lr Scheduler
        self.lr_scheduler = None
        self.scheduler_step_in_batch = False
        self.init_lr_scheduler()

        # Init loss
        self.losses = {}
        self.loss_results = {
            'classification': [],
            'segmentation': []
        }
        self.losses_weights = []
        self.init_loss()

        self.classification_data_args: dict | None = self.config.get('classification_data_config')
        self.segmentation_data_args: dict | None = self.config.get('segmentation_data_config')

        # Init Classification And Segmentation Expand Rate
        self.cls_expanding_rate = 0
        self.seg_expanding_rate = 0
        self.init_expand_rate()

        self.step = 0
        # Init Classification Dataset And Dataloader
        if self.task in [Task.MultiTask, Task.CLS]:
            self.cls_train_dataset: ClassificationDataset = None  # noqa
            self.cls_val_dataset: ClassificationDataset = None  # noqa
            self.cls_train_dataloader: DataLoader = None  # noqa
            self.cls_val_dataloader: DataLoader = None  # noqa

            self.classification_data_args['dataset']['train']['expanding_rate'] = self.cls_expanding_rate
            self.build_classification_dataset_and_dataloader()
            logger.success('Init Classification Dataset And Dataloader Done.')

        # Init Segmentation Dataset And Dataloader
        if self.task in [Task.MultiTask, Task.SEG]:
            self.seg_train_dataset: SegmentationDataSet = None  # noqa
            self.seg_val_dataset: SegmentationDataSet = None  # noqa
            self.seg_train_dataloader: DataLoader = None  # noqa
            self.seg_val_dataloader: DataLoader = None  # noqa

            self.segmentation_data_args['dataset']['train']['expanding_rate'] = self.seg_expanding_rate
            self.build_segmentation_dataset_and_dataloader()
            logger.success('Init Segmentation Dataset And Dataloader Done.')

        if self.task == Task.MultiTask:
            self.step = min(len(self.cls_train_dataloader), len(self.seg_train_dataloader))
        elif self.task == Task.CLS:
            self.step = len(self.cls_train_dataloader)
        elif self.task == Task.SEG:
            self.step = len(self.seg_train_dataloader)

        self.check_num_of_classes()
        self.backup_config()

        self.train_logger = None
        self.val_logger = None
        self.init_logger()

    def backup_config(self) -> None:
        shutil.copy(self.config_path, self.curr_exp_path)

    def init_workspace(self) -> None:
        time = get_time()

        self.curr_exp_path = os.path.join(self.project_args.work_dir, 'runs', time)
        self.weights_dir = os.path.join(self.curr_exp_path, 'weights')
        self.logs_dir = os.path.join(self.curr_exp_path, 'logs')

        log_type = 'csv'
        self.loss_log_path = os.path.join(self.logs_dir, f'loss_log.{log_type}')
        self.performance_log_path = os.path.join(self.logs_dir, f'performance_log.{log_type}')

        build_dir(self.project_args.work_dir)
        build_dir(self.curr_exp_path)
        build_dir(self.logs_dir)
        build_dir(self.weights_dir)

    def init_logger(self):
        self.train_logger = Logger(
            task=self.task,
            total_step=self.step,
            prefix=f"🚀 [Train:{self.task.value}]"
        )
        self.val_logger = Logger(
            task=self.task,
            prefix="🚀 [Val]"
        )

        self.loss_table_name = []
        for loss_type, loss_item in self.loss_args.items():
            for loss_name in loss_item.keys():
                self.loss_table_name.append(loss_name)
        self.write_csv(self.loss_log_path, self.loss_table_name)

        self.performance_table_name = []
        if self.task in [Task.CLS, Task.MultiTask]:
            self.performance_table_name.append('ACC1')
            self.performance_table_name.append(f'ACC{self.train_args.topk}')
        if self.task in [Task.SEG, Task.MultiTask]:
            self.performance_table_name.append('MIoU')

        self.write_csv(self.performance_log_path, self.performance_table_name)

    def init_model(self) -> None:
        self.model = build_model(asdict(self.model_args))

    def init_optimizer(self) -> None:
        self.optimizer_args['params'] = self.model.parameters
        self.amp_optimizer_wrapper = build_amp_optimizer_wrapper(**self.optimizer_args)

    def init_lr_scheduler(self) -> None:
        self.scheduler_step_in_batch = self.lr_args.pop('scheduler_step_in_batch')
        if self.lr_args['name'] == 'LambdaLR':
            self.lr_args.update({
                'optimizer': self.amp_optimizer_wrapper.optimizer,
                'lr_lambda': lambda epoch: 1 / (epoch / 4 + 1),
                'last_epoch': -1,
                'verbose': False
            })

        elif self.lr_args['name'] == 'CosineAnnealingWarmRestarts':
            ...
        else:
            ...

        self.lr_scheduler = build_lr_scheduler(**self.lr_args)

    def init_expand_rate(self) -> None:
        if self.task == Task.MultiTask:
            self.cls_expanding_rate, self.seg_expanding_rate = self.calc_expand_rate()
            logger.info(f'cls dataset expanding rate: x{self.cls_expanding_rate}')
            logger.info(f"seg dataset expanding rate: x{self.seg_expanding_rate}")

    def calc_expand_rate(self) -> tuple[int, int]:
        # expanding data
        cls_train_num_of_images = get_num_of_images(self.classification_data_args['dataset']['train']['root'])
        seg_train_num_of_images = get_num_of_images(self.segmentation_data_args['dataset']['train']['root'])

        cls_expanding_rate = 1
        seg_expanding_rate = 1
        if cls_train_num_of_images > seg_train_num_of_images:
            difference = cls_train_num_of_images - seg_train_num_of_images
            cls_expanding_rate = 0
        else:
            difference = seg_train_num_of_images - cls_train_num_of_images
            seg_expanding_rate = 0

        cls_expanding_rate *= math.ceil(difference / cls_train_num_of_images)
        seg_expanding_rate *= math.ceil(difference / seg_train_num_of_images)
        return cls_expanding_rate, seg_expanding_rate

    def build_classification_dataset_and_dataloader(self) -> None:
        # Add transform
        # Train transform
        classification_transform = ClassificationTransform()
        self.classification_data_args['dataset']['train'][
            'transform'] = classification_transform.image_transform
        self.classification_data_args['dataset']['train'][
            'target_transform'] = classification_transform.target_transform

        # Val transform
        self.classification_data_args['dataset']['val'][
            'transform'] = classification_transform.normalize_transform
        self.classification_data_args['dataset']['val'][
            'target_transform'] = classification_transform.target_transform

        # Build Dataset
        self.cls_train_dataset = ClassificationDataset(**self.classification_data_args['dataset']['train'])
        self.cls_val_dataset = ClassificationDataset(**self.classification_data_args['dataset']['val'])
        self.cls_train_dataset.save_label_to_id_map(
            os.path.join(self.curr_exp_path, 'cls_id_to_label.txt'),
            self.cls_train_dataset.labels_to_idx
        )

        # BalancedBatchSampler
        if self.classification_data_args['dataloader']['train']['batch_sampler'] == 'BalancedBatchSampler':
            batch_sampler = BalancedBatchSampler(
                torch.tensor(self.cls_train_dataset.targets),
                n_classes=self.model.num_classes,
                n_samples=math.ceil(
                    self.classification_data_args['dataloader']['train']['batch_size'] / self.model.num_classes
                )
            )
            self.classification_data_args['dataloader']['train'].update({
                'shuffle': False,
                'batch_size': 1,
                'drop_last': False,
                'sampler': None,
                'batch_sampler': batch_sampler
            })

        # Build Dataloader
        self.cls_train_dataloader = DataLoader(
            dataset=self.cls_train_dataset,
            **self.classification_data_args['dataloader']['train']
        )
        self.cls_val_dataloader = DataLoader(
            dataset=self.cls_val_dataset,
            **self.classification_data_args['dataloader']['val']
        )

    def build_segmentation_dataset_and_dataloader(self) -> None:
        # Add transform
        segmentation_transform = None
        if self.segmentation_data_args['dataset']['train'].get('wh'):
            segmentation_transform = SegmentationTransform(
                resize_wh=self.segmentation_data_args['dataset']['train']['wh']
            )

        # Train transform
        self.segmentation_data_args['dataset']['train'][
            'transform'] = segmentation_transform.image_transform
        self.segmentation_data_args['dataset']['train'][
            'target_transform'] = segmentation_transform.target_transform

        # Val transform
        self.classification_data_args['dataset']['val'][
            'transform'] = segmentation_transform.image_transform
        self.classification_data_args['dataset']['val'][
            'target_transform'] = segmentation_transform.target_transform

        # Build Dataset
        self.seg_train_dataset = SegmentationDataSet(**self.segmentation_data_args['dataset']['train'])
        self.seg_val_dataset = SegmentationDataSet(**self.segmentation_data_args['dataset']['val'])
        self.seg_train_dataset.save_label_to_id_map(
            os.path.join(self.curr_exp_path, 'seg_id_to_label.txt'),
            self.seg_train_dataset.labels_to_idx
        )

        # Build Dataloader
        self.seg_train_dataloader = DataLoader(
            dataset=self.seg_train_dataset,
            **self.segmentation_data_args['dataloader']['train']
        )
        self.seg_val_dataloader = DataLoader(
            dataset=self.seg_val_dataset,
            **self.segmentation_data_args['dataloader']['val']
        )

    def check_num_of_classes(self) -> None:
        if self.task in [Task.MultiTask, Task.CLS]:
            if self.model.num_classes != len(self.cls_train_dataset.labels):
                logger.error(
                    f'model num_classes:{self.model.num_classes}!=num of dataset labels:{len(self.cls_train_dataset.labels)}')
                exit()
        if self.task in [Task.MultiTask, Task.SEG]:
            if self.model.mask_classes != len(self.seg_train_dataset.labels):
                logger.error(
                    f'model mask_classes:{self.model.mask_classes}!=num of dataset labels:{len(self.seg_train_dataset.labels)}')
                exit()

    def init_loss(self):

        if self.task in [Task.MultiTask, Task.CLS]:
            self.losses.update({'classification': []})  # noqa

        if self.task in [Task.MultiTask, Task.SEG]:
            self.losses.update({'segmentation': []})  # noqa

        self.losses_weights = self.loss_args.pop('losses_weights')

        args: dict
        for loss_type, loss_config in self.loss_args.items():
            for name, args in loss_config.items():
                if name == 'PeriodLoss':
                    args.update({
                        'device': self.model.device
                    })
                self.losses[loss_type].append(build_loss(name, **args))

    def forward_with_train(self, images: torch.Tensor, ) -> Union[torch.Tensor, List[torch.Tensor]]:

        if images is None:
            logger.error('Images is None')
            raise

        if not self.model.training:
            self.model.train()

        model_output = self.model(images)

        return model_output

    def forward_with_val(self, images: torch.Tensor, ) -> Union[torch.Tensor, List[torch.Tensor]]:

        if images is None:
            logger.error('Images is None')
            raise

        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            model_output = self.model(images)

        return model_output

    def move_to_device(self, data: torch.Tensor) -> torch.Tensor:
        if self.model.device != 'cpu':
            return data.cuda(self.model.device, non_blocking=True)
        return data

    def run(self):

        while self.train_args.epoch < self.train_args.max_epoch:
            # n*train -> k*val -> n*train->...
            for i, flow in enumerate(self.train_args.workflow):
                mode, epochs = flow
                run_one_epoch = getattr(self, mode)

                for _ in range(epochs):
                    if self.train_args.epoch >= self.train_args.max_epoch:
                        break

                    run_one_epoch()  # train() or val()

                    if mode == 'train':
                        self.train_args.epoch += 1
                    elif mode == 'val':
                        ...

    @timer
    def train(self):

        self.model.train()

        self.amp_optimizer_wrapper.step()
        self.amp_optimizer_wrapper.show_lr()

        dataloaders = []
        if self.task in [Task.CLS, Task.MultiTask]:
            dataloaders.append(self.cls_train_dataloader)
        if self.task in [Task.SEG, Task.MultiTask]:
            dataloaders.append(self.seg_train_dataloader)

        # Update lr
        if self.scheduler_step_in_batch is False:
            self.lr_scheduler.step()

        datas: tuple
        for i, datas in enumerate(zip(*dataloaders)):
            if self.task == Task.MultiTask:
                cls_data, seg_data = datas
            else:
                cls_data = seg_data = datas[0]  # cls or seg training

            input_data = None
            loss_results = []  # [loss_res1,loss_res2,...]
            performances = []  # {acc1:a,accn:b,miou:c}
            curr_bs: int = 1

            # curr_task=classification or segmentation
            for curr_task in self.loss_args.keys():

                # Get the classification data or segmentation data
                if curr_task == Task.CLS.value:
                    input_data = cls_data
                elif curr_task == Task.SEG.value:
                    input_data = seg_data

                curr_task = task_convert(curr_task)

                # Move to device
                images, targets = input_data
                images = self.move_to_device(images)
                targets = self.move_to_device(targets)

                with self.amp_optimizer_wrapper.optim_context():

                    # Model forward
                    model_output = self.forward_with_train(images)

                    # Loss forward
                    loss: BaseLossForward
                    for loss in self.losses[curr_task.value]:  # loss1->loss2->...
                        loss.model_output = model_output
                        loss.targets = targets

                        curr_bs = targets.shape[0]

                        # self.loss_results[curr_task.value].append(loss.forward())
                        loss_results.append(loss.forward())

                curr_performance: dict = calc_performance(
                    curr_task,
                    self.train_args.topk,
                    self.model_args.mask_classes,
                    model_output,
                    targets
                )
                for k, v in curr_performance.items():
                    performances.append(v)

            loss_sum = self.loss_sum(loss_results)
            self.update_loss_log(loss_results, curr_bs)

            # Update optimizer
            self.amp_optimizer_wrapper.update_params(loss_sum)

            # Update lr
            if self.scheduler_step_in_batch is True:
                self.lr_scheduler.step(self.train_args.epoch + i / self.step)

    def val(self):
        self.model.eval()

    def update_loss_log(
            self,
            losses: Union[np.ndarray, List[torch.Tensor]],
            batch_size: Optional[int] = 1
    ) -> None:

        losses = [round(self.to_constant(item), 8) for item in losses]

        self.write_csv(self.loss_log_path, losses)

        total_cls_loss = 0
        total_seg_loss = 0

        # losses = self.to_constant(losses)
        new_losses = []
        for weight, loss in zip(self.losses_weights, losses):
            new_losses.append(weight * loss)

        for loss_type, loss_items in self.loss_args.items():  # Classification or Segmentation
            for i in range(len(loss_items.items())):
                if loss_type == Task.CLS.value:
                    total_cls_loss += new_losses.pop(0)

                elif loss_type == Task.SEG.value:
                    total_seg_loss += new_losses.pop(0)

        if self.task in [Task.CLS, Task.MultiTask]:
            self.train_logger.cls_loss.update(total_cls_loss, batch_size)

        if self.task in [Task.SEG, Task.MultiTask]:
            self.train_logger.seg_loss.update(total_seg_loss, batch_size)

    def update_performance_log(
            self,
            performances: list,
            batch_size: Optional[int] = 1
    ) -> None:
        for loss_type, loss_items in self.loss_args.items():
            if task in [Task.CLS, Task.MultiTask]:
                self.train_logger.top1.update(performances[0], batch_size)
                self.train_logger.topn.update(performances[1], batch_size)

            if task == Task.SEG:
                self.train_logger.MIoU.update(performances[0], batch_size)

            if task == Task.MultiTask:
                self.train_logger.MIoU.update(performances[2], batch_size)

    def loss_sum(self, loss_result: List[torch.Tensor]) -> torch.Tensor:

        if len(loss_result) != len(self.losses_weights):
            logger.error('len(loss_result)!=len(self.losses_weights)')

        ret = 0
        for loss_val, weight in zip(loss_result, self.losses_weights):
            ret += loss_val * weight
        return ret  # noqa

    @staticmethod
    def write_csv(save_path: str, data: list) -> None:

        with open(save_path, "a", encoding="utf-8", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(data)
            f.close()

    @staticmethod
    def to_constant(x: Union[np.float32, np.float64, torch.Tensor, np.ndarray]):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, np.float64) or isinstance(x, np.float32):
            return x.item()
        elif isinstance(x, torch.Tensor):
            return x.tolist()
        else:
            return x
