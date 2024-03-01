import os
import math
from typing import Union, List
import shutil

import torch
from loguru import logger
from dataclasses import asdict
from torch.utils.data import DataLoader

from utils.util import load_yaml, get_num_of_images, timer
from task import Task, task_convert
from args import ProjectArgs, TrainArgs, ModelArgs
from builder import build_workspace, build_exp, init_seeds, init_backends_cudnn
from builder import build_model, build_amp_optimizer_wrapper, build_loss, build_lr_scheduler
from balanced_batch_sampler import BalancedBatchSampler
from dataset import ClassificationDataset, SegmentationDataSet
from transforms import ClassificationTransform, SegmentationTransform
from model import Model
from optim import AmpOptimWrapper
from loss_forward import *


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
        self.losses_weights = []
        self.init_loss()

        self.classification_data_args: dict | None = self.config.get('classification_data_config')
        self.segmentation_data_args: dict | None = self.config.get('segmentation_data_config')

        # Init Classification And Segmentation Expand Rate
        self.cls_expanding_rate = 0
        self.seg_expanding_rate = 0
        self.init_expand_rate()

        # Init Classification Dataset And Dataloader
        if self.task in [Task.MultiTask, Task.CLS]:
            self.cls_train_dataset = None
            self.cls_train_dataloader = None
            self.cls_val_dataset = None
            self.cls_val_dataloader = None
            self.classification_data_args['dataset']['train']['expanding_rate'] = self.cls_expanding_rate
            self.build_classification_dataset_and_dataloader()
            logger.success('Init Classification Dataset And Dataloader Done.')

        # Init Segmentation Dataset And Dataloader
        if self.task in [Task.MultiTask, Task.SEG]:
            self.seg_train_dataset = None
            self.seg_train_dataloader = None
            self.seg_val_dataset = None
            self.seg_val_dataloader = None
            self.segmentation_data_args['dataset']['train']['expanding_rate'] = self.seg_expanding_rate
            self.build_segmentation_dataset_and_dataloader()
            logger.success('Init Segmentation Dataset And Dataloader Done.')

        self.check_num_of_classes()
        self.backup_config()

    def backup_config(self) -> None:
        shutil.copy(self.config_path, self.curr_exp_path)

    def init_workspace(self) -> None:
        build_workspace(self.project_args.work_dir)
        self.curr_exp_path = build_exp(self.project_args.work_dir)

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

    def forward_with_train(
            self,
            images: torch.Tensor,
            targets: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        if images is None or targets is None:
            logger.error('train_data is None')
            raise
        with self.amp_optimizer_wrapper.optim_context():
            model_output = self.model(images)

        return model_output

    def forward_with_val(
            self,
            images: torch.Tensor,
            targets: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        if images is None or targets is None:
            logger.error('train_data is None')
            raise
        with torch.no_grad():
            model_output = self.model(images)

        return model_output

    def move_to_device(self, data: torch.Tensor) -> torch.Tensor:
        if self.model.device != 'cpu':
            return data.cuda(self.model.device, non_blocking=True)
        return data

    def run(self):
        curr_epoch = 0

        while self.train_args.epoch < self.train_args.max_epoch:
            # n*train -> k*val -> n*train->...
            for i, flow in enumerate(self.train_args.workflow):
                mode, epochs = flow
                run_one_epoch = getattr(self, mode)

                for _ in range(epochs):
                    if curr_epoch >= self.train_args.max_epoch:
                        break

                    run_one_epoch()  # train() or val()

                    if mode == 'train':
                        curr_epoch += 1
                    elif mode == 'val':
                        ...

    def train(self):

        self.model.train()

        self.amp_optimizer_wrapper.step()
        self.amp_optimizer_wrapper.show_lr()

        dataloaders = []
        if self.task in [Task.CLS, Task.MultiTask]:
            dataloaders.append(self.cls_train_dataloader)
        if self.task in [Task.SEG, Task.MultiTask]:
            dataloaders.append(self.seg_train_dataloader)

        if self.scheduler_step_in_batch is False:
            self.lr_scheduler.step()  # update lr

        datas: tuple
        for i, datas in enumerate(zip(*dataloaders)):
            if self.task == Task.MultiTask:
                cls_data, seg_data = datas
            else:
                cls_data = seg_data = datas[0]  # cls or seg training

            # Update lr
            if self.scheduler_step_in_batch is True:
                self.lr_scheduler.step()

            loss_results = []
            input_data = None
            for curr_task in self.loss_args.keys():
                if curr_task == Task.CLS.value:
                    curr_task = Task.CLS
                    input_data = cls_data

                elif curr_task == Task.SEG.value:
                    curr_task = Task.SEG
                    input_data = seg_data

                images, targets = input_data
                images = self.move_to_device(images)
                targets = self.move_to_device(targets)

                # Model Infer
                model_output = self.forward_with_train(images, targets)

                # Loss forward
                loss: BaseLossForward
                for loss in self.losses[curr_task.value]:
                    loss.model_output = model_output
                    loss.targets = targets
                    loss_results.append(loss.forward())

    def val(self):
        self.model.eval()


if __name__ == '__main__':
    config_path = r'D:\llf\code\pytorch-lab\configs\test_mt.yml'
    pipline = Pipline(config_path)
    pipline.run()
