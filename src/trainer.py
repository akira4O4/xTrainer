import os
import math
import shutil
from typing import Union, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from dataclasses import asdict
from torch.utils.data import DataLoader
from mlflow import log_metric, log_param, set_experiment

from utils.util import load_yaml, get_num_of_images, timer, get_time
from .task import Task, task_convert
from .args import ProjectArgs, TrainArgs, ModelArgs
from .builder import build_dir, init_seeds, init_backends_cudnn
from .builder import build_model, build_optimizer_wrapper, build_amp_optimizer_wrapper, build_loss, build_lr_scheduler
from .balanced_batch_sampler import BalancedBatchSampler
from .dataset import ClassificationDataset, SegmentationDataSet
from .transforms import ClassificationTransform, SegmentationTransform
from .model import Model
from .optim import AmpOptimWrapper, OptimWrapper
from .loss_forward import *
from .performance import calc_performance
from .data_logger import DataLogger


class Trainer:
    def __init__(self, config_path: str):

        self.config_path = config_path
        if not os.path.exists(config_path):
            logger.error(f'Can`t found the {config_path}.')
            exit()

        self.round_float_4 = lambda num: round(float(num), 4)
        self.round_float_8 = lambda num: round(float(num), 8)

        # Init Args
        self.config = load_yaml(config_path)
        self.project_args = ProjectArgs(**self.config['project_config'])
        self.train_args = TrainArgs(**self.config['train_config'])
        self.model_args = ModelArgs(**self.config['model_config'])
        self.optimizer_args = self.config['optimizer_config']
        self.lr_args = self.config['lr_config']
        self.loss_args = self.config['loss_config']

        if self.model_args.gpu == -1:
            logger.error('Current Is Not Support CPU Training.')
            exit()

        if self.train_args.accumulation_steps != 0:
            self.is_accumulation = True
        else:
            self.is_accumulation = False

        # Init workspace Env
        self.curr_exp_path: str = ''
        self.weights_dir = ''

        self.init_workspace()

        init_seeds(self.train_args.seed)
        init_backends_cudnn(self.train_args.deterministic)
        self.task = task_convert(self.project_args.task)

        # Build Model
        self.model: Model = None  # noqa
        self.init_model()

        # Build Optimizer
        # self.amp_optimizer_wrapper: AmpOptimWrapper = None  # noqa
        self.optimizer_wrapper: Union[OptimWrapper, AmpOptimWrapper] = None  # noqa
        self.init_optimizer()

        # Build Lr Scheduler
        self.curr_lr = 0
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

        self.classification_data_args: dict = self.config.get('classification_data_config')
        self.segmentation_data_args: dict = self.config.get('segmentation_data_config')

        # Init Classification And Segmentation Expand Rate
        self.cls_expanding_rate = 0
        self.seg_expanding_rate = 0
        self.init_expand_rate()

        self.total_step = 0
        # Init Classification Dataset And Dataloader
        if self.task == Task.MultiTask:
            self.classification_data_args['dataset']['train']['expanding_rate'] = self.cls_expanding_rate
            self.segmentation_data_args['dataset']['train']['expanding_rate'] = self.seg_expanding_rate

        if self.task in [Task.MultiTask, Task.CLS]:
            self.cls_train_dataset: ClassificationDataset = None  # noqa
            self.cls_val_dataset: ClassificationDataset = None  # noqa
            self.cls_train_dataloader: DataLoader = None  # noqa
            self.cls_val_dataloader: DataLoader = None  # noqa
            self.build_classification_dataset_and_dataloader()
            logger.success('Init Classification Dataset And Dataloader Done.')

        # Init Segmentation Dataset And Dataloader
        if self.task in [Task.MultiTask, Task.SEG]:
            self.seg_train_dataset: SegmentationDataSet = None  # noqa
            self.seg_val_dataset: SegmentationDataSet = None  # noqa
            self.seg_train_dataloader: DataLoader = None  # noqa
            self.seg_val_dataloader: DataLoader = None  # noqa
            self.build_segmentation_dataset_and_dataloader()
            logger.success('Init Segmentation Dataset And Dataloader Done.')

        if self.task == Task.MultiTask:
            self.total_step = min(len(self.cls_train_dataloader), len(self.seg_train_dataloader))
        elif self.task == Task.CLS:
            self.total_step = len(self.cls_train_dataloader)
        elif self.task == Task.SEG:
            self.total_step = len(self.seg_train_dataloader)

        self.check_num_of_classes()
        self.backup_config()

        self.train_logger = None
        self.val_logger = None

        self.training_top1_data_logger: DataLogger = None  # noqa
        self.training_topk_data_logger: DataLogger = None  # noqa
        self.training_miou_data_logger: DataLogger = None  # noqa

        self.val_top1_data_logger: DataLogger = None  # noqa
        self.val_topk_data_logger: DataLogger = None  # noqa
        self.val_miou_data_logger: DataLogger = None  # noqa

        self.classification_loss_data_logger: DataLogger = None  # noqa
        self.segmentation_loss_data_logger: DataLogger = None  # noqa

        self.init_logger()

        self.init_mlflow()
        logger.info('请使用MLFlow UI进行训练数据观察 -> [Terminal]: mlflow ui')

    def backup_config(self) -> None:
        shutil.copy(self.config_path, self.curr_exp_path)

    def init_mlflow(self) -> None:
        if self.project_args.mlflow_experiment_name == '':
            logger.info(f'MLFlow Experiment Name: Default.')
        else:
            set_experiment(self.project_args.mlflow_experiment_name)
            logger.info(f'MLFlow Experiment Name:{self.project_args.mlflow_experiment_name}.')

    def init_workspace(self) -> None:
        time = get_time()

        self.curr_exp_path = os.path.join(self.project_args.work_dir, 'runs', time)
        self.weights_dir = os.path.join(self.curr_exp_path, 'weights')

        build_dir(self.project_args.work_dir)
        build_dir(self.curr_exp_path)
        build_dir(self.weights_dir)

    def init_logger(self):
        self.training_top1_data_logger = DataLogger('Training Top1')
        self.training_topk_data_logger = DataLogger(f'Training Top{self.train_args.topk}')
        self.training_miou_data_logger = DataLogger('Training MIoU')

        self.val_top1_data_logger = DataLogger('Val Top1')
        self.val_topk_data_logger = DataLogger(f'Val Top{self.train_args.topk}')
        self.val_miou_data_logger = DataLogger('Val MIoU')

        self.classification_loss_data_logger = DataLogger('Classification Loss')
        self.segmentation_loss_data_logger = DataLogger('Segmentation Loss')

    def init_model(self) -> None:
        self.model = build_model(asdict(self.model_args))

    def init_optimizer(self) -> None:

        self.optimizer_args['params'] = self.model.parameters

        if self.train_args.amp:
            self.optimizer_wrapper = build_amp_optimizer_wrapper(**self.optimizer_args)
            logger.info('AMP is open.')
        else:
            self.optimizer_wrapper = build_optimizer_wrapper(**self.optimizer_args)
            logger.info('AMP is close')

    def init_lr_scheduler(self) -> None:
        self.scheduler_step_in_batch = self.lr_args.pop('scheduler_step_in_batch')
        if self.lr_args['name'] == 'LambdaLR':
            self.lr_args.update({
                'optimizer': self.optimizer_wrapper.optimizer,
                'lr_lambda': lambda epoch: 1 / (epoch / 4 + 1),
                'last_epoch': -1,
                'verbose': False
            })

        elif self.lr_args['name'] == 'CosineAnnealingWarmRestarts':
            self.lr_args.update({
                'optimizer': self.optimizer_wrapper.optimizer,
            })
        else:
            ...

        self.lr_scheduler = build_lr_scheduler(**self.lr_args)

    def init_expand_rate(self) -> None:
        if self.task == Task.MultiTask:
            self.cls_expanding_rate, self.seg_expanding_rate = self.calc_expand_rate()
            logger.info(f'cls dataset expanding rate: x{self.cls_expanding_rate}')
            logger.info(f"seg dataset expanding rate: x{self.seg_expanding_rate}")

    def calc_expand_rate(self) -> Tuple[int, int]:
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

        log_param('cls data expanding rate', cls_expanding_rate)
        log_param('seg data expanding rate', seg_expanding_rate)

        return cls_expanding_rate, seg_expanding_rate

    def build_classification_dataset_and_dataloader(self) -> None:
        # Add transform
        # Train transform

        classification_transform = ClassificationTransform()

        # classification_transform = None
        # if self.classification_data_args['dataset']['train']['transform_resize']:
        #     classification_transform = ClassificationTransform(
        #         resize_wh=self.classification_data_args['dataset']['train']['wh']
        #     )
        # else:
        #     classification_transform = ClassificationTransform()

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
        if self.segmentation_data_args['dataset']['train'].get('transform_resize') is not None:
            segmentation_transform = SegmentationTransform(
                resize_wh=self.segmentation_data_args['dataset']['train']['wh']
            )
        else:
            segmentation_transform = SegmentationTransform()

        # Train transform
        self.segmentation_data_args['dataset']['train'][
            'transform'] = segmentation_transform.image_transform
        self.segmentation_data_args['dataset']['train'][
            'target_transform'] = segmentation_transform.target_transform

        # Val transform
        self.segmentation_data_args['dataset']['val'][
            'transform'] = segmentation_transform.image_transform
        self.segmentation_data_args['dataset']['val'][
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

        self.losses_weights = self.loss_args.pop('loss_weights')

        args: dict
        for loss_type, loss_config in self.loss_args.items():
            for name, args in loss_config.items():
                if name == 'PeriodLoss':
                    args.update({
                        'device': self.model.device
                    })
                self.losses[loss_type].append(build_loss(name, **args))

        if not self.losses:
            logger.error('Loss is empty.')
            return

    def forward_with_train(self, images: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:

        if images is None:
            logger.error('Images is None')
            raise

        if not self.model.training:
            self.model.train()

        model_output = self.model(images)

        return model_output

    def forward_with_val(self, images: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:

        if images is None:
            logger.error('Images is None')
            raise

        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            model_output = self.model(images)

        return model_output

    def move_to_device(self, data: torch.Tensor) -> torch.Tensor:
        if self.model.device != torch.device('cpu'):
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
                        log_metric('Epoch', self.train_args.epoch)

                    elif mode == 'val':
                        best_weight_info = {}

                        if self.task in [Task.MultiTask, Task.CLS]:
                            best_weight_info.update({
                                'Top1#': self.round_float_4(self.val_top1_data_logger.avg)
                            })

                        if self.task in [Task.MultiTask, Task.SEG]:
                            best_weight_info.update({
                                'MIoU#': self.round_float_4(self.val_miou_data_logger.avg)
                            })

                        self.model.save_checkpoint(
                            save_path=self.weights_dir,
                            epoch=self.train_args.epoch,
                            lr=self.curr_lr,
                            optimizer_state_dict=self.optimizer_wrapper.state_dict(),
                            model_info=best_weight_info
                        )

                        self.val_top1_data_logger.reset()
                        self.val_topk_data_logger.reset()
                        self.val_miou_data_logger.reset()

    @timer
    def train(self):

        if not self.model.training:
            self.model.train()

        self.optimizer_wrapper.step()
        self.curr_lr = self.round_float_8(self.optimizer_wrapper.lr[0])
        log_metric('Lr', self.curr_lr)

        dataloaders = []
        if self.task in [Task.CLS, Task.MultiTask]:
            dataloaders.append(self.cls_train_dataloader)
        if self.task in [Task.SEG, Task.MultiTask]:
            dataloaders.append(self.seg_train_dataloader)

        # Update lr
        if self.scheduler_step_in_batch is False:
            self.lr_scheduler.step()

        datas: tuple
        for curr_step, datas in enumerate(zip(*dataloaders)):
            if self.task == Task.MultiTask:
                cls_data, seg_data = datas
            else:
                cls_data = seg_data = datas[0]  # cls or seg training

            input_data = None
            loss_results = {}  # [loss_res1,loss_res2,...]

            # curr_task=classification or segmentation
            for loss_type in self.loss_args.keys():

                # Get the classification data or segmentation data
                if loss_type == Task.CLS.value:
                    input_data = cls_data
                elif loss_type == Task.SEG.value:
                    input_data = seg_data

                loss_type = task_convert(loss_type)

                # Move to device
                images, targets = input_data
                images = self.move_to_device(images)
                targets = self.move_to_device(targets)

                # AMP training
                if self.train_args.amp:
                    with self.optimizer_wrapper.optim_context():
                        model_output = self.forward_with_train(images)

                # Loss forward
                # loss1->loss2->...
                loss_runner: BaseLossRunner
                for loss_runner in self.losses[loss_type.value]:
                    loss_runner.model_output = model_output
                    loss_runner.targets = targets

                    if self.train_args.amp:
                        with self.optimizer_wrapper.optim_context():
                            loss_val = loss_runner.forward()  # noqa

                    assert not torch.any(torch.isnan(loss_val))
                    loss_results.update({loss_runner.loss_name: loss_val})

                training_performance: dict = calc_performance(
                    loss_type,
                    self.train_args.topk,
                    self.model_args.mask_classes,
                    model_output,
                    targets
                )
                if loss_type == Task.CLS:
                    self.training_top1_data_logger.update(training_performance['top1'])
                    self.training_topk_data_logger.update(training_performance['topk'])
                if loss_type == Task.SEG:
                    self.training_miou_data_logger.update(training_performance['miou'])

            self.update_training_performance_to_mlflow('batch_size')

            # Update optimizer
            loss_sum = self.loss_sum(loss_results)

            if self.is_accumulation:
                loss_sum = loss_sum / self.train_args.accumulation_steps
                self.optimizer_wrapper.loss_backward(loss_sum)

                if (curr_step + 1) % self.train_args.accumulation_steps == 0:
                    self.optimizer_wrapper.step_update_zero()
            else:
                self.optimizer_wrapper.backward_step_update_zero(loss_sum)

            self.update_training_loss_to_mlflow(loss_results, 'batch_size')
            log_metric('Sum of Loss', self.to_constant(loss_sum))

            # Update lr
            if self.scheduler_step_in_batch is True:
                self.lr_scheduler.step(self.train_args.epoch + curr_step / self.total_step)

            # Easy info display
            if curr_step % self.train_args.print_freq == 0:
                print(
                    f'🚀[Training] Epoch:[{self.train_args.epoch}/{self.train_args.max_epoch}] '
                    f'Step:[{curr_step}/{self.total_step}]...'
                )

        self.update_training_performance_to_mlflow('epoch')
        self.update_training_loss_to_mlflow(batch_or_epoch='epoch')
        self.training_top1_data_logger.reset()
        self.training_topk_data_logger.reset()
        self.training_miou_data_logger.reset()

    @timer
    def val(self):

        if self.model.training:
            self.model.eval()

        for task in self.loss_args.keys():
            task = task_convert(task)

            if task == Task.SEG:
                self.segmentation_val()
                print(f'Segmentation Val MIoU(Avg): {self.round_float_4(self.val_miou_data_logger.avg)}')

            elif task == Task.CLS:
                self.classification_val()
                print(
                    f'Classification Val Top1(Avg): {self.round_float_4(self.val_top1_data_logger.avg)} '
                    f'Top{self.train_args.topk}(Avg): {self.round_float_4(self.val_topk_data_logger.avg)}'
                )

    def classification_val(self) -> None:

        for data in tqdm(self.cls_val_dataloader):
            images, targets = data
            images = self.move_to_device(images)
            targets = self.move_to_device(targets)

            model_output = self.forward_with_val(images)
            performance: dict = calc_performance(
                task=Task.CLS,
                topk=self.train_args.topk,
                model_output=model_output,
                targets=targets
            )
            self.val_top1_data_logger.update(performance['top1'])
            self.val_topk_data_logger.update(performance['topk'])

        log_metric('Val Epoch Top1', self.val_top1_data_logger.avg)
        log_metric(f'Val Epoch Top{self.train_args.topk}', self.val_topk_data_logger.avg)

    def segmentation_val(self) -> None:
        for data in tqdm(self.seg_val_dataloader):
            images, targets = data
            images = self.move_to_device(images)
            targets = self.move_to_device(targets)

            model_output = self.forward_with_val(images)
            performance: dict = calc_performance(
                task=Task.SEG,
                model_output=model_output,
                targets=targets,
                mask_classes=self.model.mask_classes
            )
            self.val_miou_data_logger.update(performance['miou'])

        log_metric('Val Epoch MIoU', self.val_miou_data_logger.avg)

    def update_training_loss_to_mlflow(
            self,
            loss_results: Optional[dict] = None,
            batch_or_epoch: Optional[str] = None
    ) -> None:
        if batch_or_epoch == 'batch_size':

            for k, v in loss_results.items():
                loss_results[k] = self.round_float_8(self.to_constant(v))

            for kv, weight in zip(loss_results.items(), self.losses_weights):
                loss_name, loss_val = kv
                loss_results[loss_name] = loss_val * weight

            for loss_type, loss_item in self.loss_args.items():
                for loss_name, loss_params in self.loss_args[loss_type].items():

                    if task_convert(loss_type) == Task.CLS:
                        self.classification_loss_data_logger.update(loss_results[loss_name])
                    elif task_convert(loss_type) == Task.SEG:
                        self.segmentation_loss_data_logger.update(loss_results[loss_name])

                    log_metric(f'Batch {loss_name}', loss_results[loss_name])

        elif batch_or_epoch == 'epoch':
            log_metric('Classification Total Loss', self.classification_loss_data_logger.avg)
            log_metric('Segmentation Total Loss', self.segmentation_loss_data_logger.avg)

    def update_training_performance_to_mlflow(self, batch_size_or_epoch: str = 'batch_size') -> None:
        if batch_size_or_epoch == 'batch_size':
            log_metric('Training Batch Top1', self.training_top1_data_logger.curr_val)
            log_metric(f'Training Batch Top{self.train_args.topk}', self.training_topk_data_logger.curr_val)
            log_metric('Training Batch MIoU', self.training_miou_data_logger.curr_val)
        else:
            log_metric('Training Epoch Top1', self.training_top1_data_logger.avg)
            log_metric(f'Training Epoch Top{self.train_args.topk}', self.training_topk_data_logger.avg)
            log_metric('Training Epoch MIoU', self.training_miou_data_logger.avg)

    def loss_sum(self, loss_results: dict) -> torch.Tensor:

        if len(loss_results) != len(self.losses_weights):
            logger.error('len(loss_result)!=len(self.losses_weights)')

        ret = 0
        for kv, weight in zip(loss_results.items(), self.losses_weights):
            loss_name, loss_val = kv
            ret += loss_val * weight

        return ret  # noqa

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
