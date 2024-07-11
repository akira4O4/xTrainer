import os
import math
from typing import Union, List, Optional, Tuple, Any

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from mlflow import log_metric, set_experiment

from trainerx import (
    CONFIG,
    DEFAULT_WORKSPACE,
    DEFAULT_OPTIMIZER
)
from trainerx.utils.task import Task
from trainerx.core.model import Model
from trainerx.core.loss_forward import BaseLossForward
from trainerx.utils.performance import calc_performance
from trainerx.core.optim import AmpOptimWrapper, OptimWrapper
from trainerx.utils.data_logger import TrainLogger, ValLogger, LossLogger
from trainerx.core.balanced_batch_sampler import BalancedBatchSampler
from trainerx.dataset.segmentation_dataset import SegmentationDataSet
from trainerx.dataset.classification_dataset import ClassificationDataset
from trainerx.core.transforms import (
    ValTransform,
    ClsImageTransform,
    ClsTargetTransform,
    SegTargetTransform,
    SegImageTransform
)
from trainerx.core.builder import (
    build_loss_forward,
    build_optimizer_wrapper,
    build_amp_optimizer_wrapper
)
from trainerx.utils.torch_utils import (
    init_seeds,
    init_backends_cudnn,
)
from trainerx.utils.common import (
    save_yaml,
    error_exit,
    round4,
    round8,
    timer,
    check_dir
)


def init_workspace(root: str) -> tuple:
    if os.path.isdir(root):
        project_root = root
    else:  # ./workspace/project
        project_root = os.path.join(DEFAULT_WORKSPACE, CONFIG('project'))

    check_dir(project_root)

    weight_path = os.path.join(project_root, 'weights')
    check_dir(weight_path)

    output_path = os.path.join(project_root, 'temp')
    check_dir(output_path)

    return project_root, weight_path, output_path


def init_mlflow(exp_name: str) -> None:
    if exp_name == '':
        logger.info(f'MLFlow Experiment Name: Default.')
    else:
        set_experiment(exp_name)
        logger.info(f'MLFlow Experiment Name:{exp_name}.')


def align_data_size(data1: int, data2: int) -> Tuple[int, int]:
    assert data1 != 0
    assert data2 != 0

    expanding_rate1 = 1
    expanding_rate2 = 1

    if data1 > data2:
        difference = data1 - data2
        expanding_rate1 = 0
    else:
        difference = data2 - data1
        expanding_rate2 = 0

    expanding_rate1 *= math.ceil(difference / data1)
    expanding_rate2 *= math.ceil(difference / data2)

    return expanding_rate1, expanding_rate2


class Trainer:
    def __init__(self):

        self.project_root_path, self.weight_path, self.output_path = init_workspace(CONFIG('project'))
        self.is_classification: bool = False
        self.is_segmentation: bool = False
        self.is_multi_task: bool = False

        # Init Data Logger ---------------------------------------------------------------------------------------------
        self.train_logger = TrainLogger(topk=CONFIG('topk'))  # noqa
        self.val_logger = ValLogger(topk=CONFIG('topk'))  # noqa
        self.loss_logger = LossLogger()

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
        self.optimizer_wrapper: Union[OptimWrapper, AmpOptimWrapper] = None  # noqa
        self.init_optimizer()

        # Init Lr Scheduler -------------------------------------------------------------------------------------------
        self.curr_lr = 0
        self.lr_scheduler = None
        self.scheduler_step_in_batch = False
        self.init_lr_scheduler()

        # Init loss ---------------------------------------------------------------------------------------------------
        self.loss_forwards: List[BaseLossForward] = []
        self.loss_weights: List[float] = []
        self.init_loss()

        # Init dataset and dataloader ---------------------------------------------------------------------------------
        if self.task.CLS:
            self.cls_train_dataset: ClassificationDataset = None  # noqa
            self.cls_val_dataset: ClassificationDataset = None  # noqa
            self.cls_train_dataloader: DataLoader = None  # noqa
            self.cls_val_dataloader: DataLoader = None  # noqa
            self.build_classification_ds_dl()

            if CONFIG('classification')['classes'] != self.cls_train_dataset.num_of_label:
                logger.error('classification num of classes setting error.')
                error_exit()

        if self.task.SEG:
            self.seg_train_dataset: SegmentationDataSet = None  # noqa
            self.seg_val_dataset: SegmentationDataSet = None  # noqa
            self.seg_train_dataloader: DataLoader = None  # noqa
            self.seg_val_dataloader: DataLoader = None  # noqa
            self.build_segmentation_ds_dl()

            if CONFIG('segmentation')['classes'] != self.seg_train_dataset.num_of_label:
                logger.error('segmentation num of classes setting error.')
                error_exit()

        # Expand dataset -----------------------------------------------------------------------------------------------
        if self.task.MT:
            rate1, rate2 = align_data_size(self.cls_train_dataset.data_size, self.seg_train_dataset.data_size)
            self.cls_train_dataset.expanding_data(rate1)
            self.seg_train_dataset.expanding_data(rate2)

            logger.info(f'Expanding classification dataset to: {self.cls_train_dataset.data_size} x{rate1}')
            logger.info(f'Expanding segmentation dataset to: {self.seg_train_dataset.data_size} x{rate2}')

        if self.task.MT:
            self.total_step = min(len(self.cls_train_dataloader), len(self.seg_train_dataloader))
        elif self.task.CLS:
            self.total_step = len(self.cls_train_dataloader)
        elif self.task.SEG:
            self.total_step = len(self.seg_train_dataloader)

        # Init MLFlow  -------------------------------------------------------------------------------------------------
        init_mlflow(CONFIG("mlflow_experiment_name"))
        logger.info('请使用MLFlow UI进行训练数据观察 -> [Terminal]: mlflow ui')

    def init_model(self) -> None:
        num_classes: int = CONFIG('classification')['classes']
        mask_classes: int = CONFIG('segmentation')['classes']

        if num_classes == mask_classes == 0:
            logger.error("num_classes == mask_classes == 0")
            error_exit()

        if not os.path.exists(CONFIG('weight')):
            logger.warning(f'{CONFIG("weight")} is not exists.')

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
            "params": self.model.parameters,
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
            self.optimizer_wrapper = build_amp_optimizer_wrapper(name, **args)
            logger.info('AMP: Open Automatic Mixed Precision(AMP)')
        else:
            self.optimizer_wrapper = build_optimizer_wrapper(name, **args)

        logger.success(f'Build Optim: {name}.')

    @staticmethod
    def _cos_lr_lambda(y1: float = 0.0, y2: float = 1.0, steps: int = 100):
        return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

    @staticmethod
    def _linear_lr_lambda(epochs: int, lrf: float):
        return lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf  # linear

    @staticmethod
    def _easy_lr_lambda():
        return lambda x: 1 / (x / 4 + 1),

    def init_lr_scheduler(self) -> None:

        if CONFIG('cos_lr'):
            lr_lambda = self._cos_lr_lambda(1, CONFIG('lrf'), CONFIG('epochs'))
        else:
            # lr_lambda = self._easy_lr_lambda()
            lr_lambda = self._linear_lr_lambda(CONFIG('epochs'), CONFIG('lrf')),

        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer_wrapper.optimizer, lr_lambda=lr_lambda)

    def build_classification_ds_dl(self) -> None:

        image_t = ClsImageTransform()
        target_t = ClsTargetTransform()
        val_t = ValTransform()

        if not os.path.exists(CONFIG('classification')['train']):
            logger.warning(f"Don`t found the {CONFIG('classification')['train']}.")
        else:

            # Build Train Dataset
            self.cls_train_dataset = ClassificationDataset(
                root=CONFIG('classification')['train'],
                wh=CONFIG('wh'),
                transform=image_t,
                target_transform=target_t,
                letterbox=CONFIG('letterbox')
            )
            save_yaml(self.cls_train_dataset.labels, os.path.join(self.output_path, 'cls_labels.yaml'))

            # Build BalancedBatchSampler
            balanced_batch_sampler = None
            if CONFIG('balanced_batch_sampler'):
                balanced_batch_sampler = BalancedBatchSampler(
                    torch.tensor(self.cls_train_dataset.targets),
                    n_classes=self.model.num_classes,
                    n_samples=math.ceil(CONFIG('classification')['batch'] / self.model.num_classes)
                )

            # Build Train DataLoader
            self.cls_train_dataloader = DataLoader(
                dataset=self.cls_train_dataset,
                batch_size=1 if CONFIG('balanced_batch_sampler') else CONFIG('batch'),
                num_workers=CONFIG('workers'),
                pin_memory=CONFIG('pin_memory'),
                batch_sampler=balanced_batch_sampler,
                shuffle=False if CONFIG('balanced_batch_sampler') else True,
                drop_last=False,
                sampler=None
            )
            logger.info(f'Classification num of labels: {self.cls_train_dataset.num_of_label}.')
            logger.info(f'Classification Train data size: {self.cls_train_dataset.data_size}.')

        if not os.path.exists(CONFIG('classification')['val']):
            logger.warning(f"Don`t found the {CONFIG('classification')['val']}.")
        else:
            # Build Val Dataset
            self.cls_val_dataset = ClassificationDataset(
                root=CONFIG('classification')['val'],
                wh=CONFIG('wh'),
                transform=val_t,
                target_transform=target_t,
                letterbox=CONFIG('letterbox')
            )
            logger.info(f'Classification Val data size: {self.cls_val_dataset.data_size}.')

            # Build Val DataLoader
            self.cls_val_dataloader = DataLoader(
                dataset=self.cls_val_dataset,
                batch_size=CONFIG('batch'),
                num_workers=CONFIG('workers'),
                pin_memory=CONFIG('pin_memory'),
                shuffle=False
            )

    def build_segmentation_ds_dl(self) -> None:
        image_t = SegImageTransform()
        target_t = SegTargetTransform()
        val_t = ValTransform()

        if not os.path.exists(CONFIG('segmentation')['val']):
            logger.warning(f"Don`t found the {CONFIG('segmentation')['train']}.")
        else:
            self.seg_train_dataset = SegmentationDataSet(
                root=CONFIG('segmentation')['train'],
                transform=image_t,
                target_transform=target_t,
            )
            background_size = len(self.seg_val_dataset.background_samples)

            save_yaml(
                self.seg_train_dataset.labels,
                os.path.join(self.output_path, 'seg_labels.yaml')
            )
            self.seg_train_dataloader = DataLoader(
                dataset=self.seg_train_dataset,
                batch_size=CONFIG('batch'),
                shuffle=False,
                num_workers=CONFIG('workers'),
                pin_memory=CONFIG('pin_memory'),
            )

            logger.info(f'Segmentation num of labels: {self.seg_train_dataset.num_of_label}.')
            logger.info(
                f'Segmentation Train data size: {self.seg_train_dataset.data_size} (background:{background_size}).')

        if not os.path.exists(CONFIG('segmentation')['val']):
            logger.warning(f"Don`t found the {CONFIG('segmentation')['val']}.")
        else:
            self.seg_val_dataset = SegmentationDataSet(
                root=CONFIG('segmentation')['val'],
                transform=val_t,
                target_transform=target_t,
            )
            self.seg_val_dataloader = DataLoader(
                dataset=self.seg_val_dataset,
                batch_size=CONFIG('batch'),
                shuffle=False,
                num_workers=CONFIG('workers'),
                pin_memory=CONFIG('pin_memory'),
            )

            logger.info(f'Segmentation Val data size: {self.seg_val_dataset.data_size}.')

    def init_loss(self) -> None:
        if self.task.CLS:
            self.loss_weights = [1]
            self.loss_forwards.append(build_loss_forward('CrossEntropyLoss'))

        if self.task.SEG:
            self.loss_weights.extend([1, 0.5])
            self.loss_forwards.extend(
                [
                    build_loss_forward('PeriodLoss', weight=[1] * self.model.mask_classes, device=self.model.device),
                    build_loss_forward('DiceLoss')
                ]
            )

        assert len(self.loss_weights) == len(self.loss_forwards)
        logger.info(f'loss weights: {self.loss_weights}.')

    def forward_with_train(self, images: torch.Tensor) -> Any:

        if images is None:
            logger.error('Images is None')
            error_exit()

        if not self.model.training:
            self.model.train()

        model_output = self.model(images)

        return model_output

    def forward_with_val(self, images: torch.Tensor) -> Any:

        if images is None:
            logger.error('Images is None')
            error_exit()

        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            model_output = self.model(images)

        return model_output

    def move_to_device(self, data: torch.Tensor) -> torch.Tensor:
        if self.model.is_cpu:
            return data
        else:
            return data.cuda(self.model.device, non_blocking=True)

    def run(self) -> None:
        curr_epoch = 0
        while curr_epoch < CONFIG('epochs'):

            # n*train -> k*val -> n*train->...
            flow: dict
            for flow in CONFIG('workflow'):
                mode, times = flow.items()
                run_one_epoch = getattr(self, mode)

                for _ in range(times):
                    if curr_epoch >= CONFIG('epochs'):
                        break

                    run_one_epoch()  # train() or val()

                    if mode == 'train':
                        curr_epoch += 1
                        log_metric('Epoch', curr_epoch)

                    elif mode == 'val':
                        best_weight_info = {}

                        if self.task.MT:
                            best_weight_info.update({'Top1#': round4(self.val_top1_data_logger.avg)})

                        if self.task.MT:
                            best_weight_info.update({'MIoU#': round4(self.val_miou_data_logger.avg)})
                        self.val_top1_data_logger.reset()
                        self.val_topk_data_logger.reset()
                        self.val_miou_data_logger.reset()

                    # self.model.save_checkpoint(
                    #     save_path=self.weights_dir,
                    #     epoch=curr_epoch,
                    # )

    @timer
    def train(self) -> None:

        if not self.model.training:
            self.model.train()

        self.optimizer_wrapper.step()
        self.curr_lr = round8(self.optimizer_wrapper.lr[0])
        log_metric('Lr', self.curr_lr)

        dataloaders: List[DataLoader] = []
        if self.task.CLS:
            dataloaders.append(self.cls_train_dataloader)
        if self.task.SEG:
            dataloaders.append(self.seg_train_dataloader)

        # Update lr
        if self.scheduler_step_in_batch is False:
            self.lr_scheduler.step()

        datas: tuple
        for curr_step, datas in enumerate(zip(*dataloaders)):
            if self.task.MT:
                cls_data, seg_data = datas
            else:
                cls_data = seg_data = datas[0]  # cls or seg training

            input_data = None
            # loss_results = {}  # [loss_res1,loss_res2,...]
            if self.task.CLS:
                input_data = cls_data

            if self.task.SEG:
                input_data = seg_data

            images, targets = input_data
            images = self.move_to_device(images)
            targets = self.move_to_device(targets)

            # Train ----------------------------------------------------------------------------------------------------
            if CONFIG('amp'):
                with self.optimizer_wrapper.optim_context():
                    model_output = self.forward_with_train(images)
            else:
                model_output = self.forward_with_train(images)

            # Calc loss ------------------------------------------------------------------------------------------------
            loss_result = {}
            for loss_forward in self.loss_forwards:
                loss_forward.set_model_output(model_output)
                loss_forward.set_targets(targets)

                if CONFIG('amp'):
                    with self.optimizer_wrapper.optim_context():
                        loss_val = forward.run()  # noqa
                else:
                    loss_val = forward.run()  # noqa

                assert not torch.any(torch.isnan(loss_val))
                loss_result.update({loss_forward.loss_name: loss_val})

            # curr_task=classification or segmentation
            # for loss_type in self.loss_args.keys():
            #
            #     # Get the classification data or segmentation data
            #     if loss_type == Task.CLS.value:
            #         input_data = cls_data
            #     elif loss_type == Task.SEG.value:
            #         input_data = seg_data
            #
            #     loss_type = task_convert(loss_type)
            #
            #     # Move to device
            #     images, targets = input_data
            #     images = self.move_to_device(images)
            #     targets = self.move_to_device(targets)
            #
            #     # AMP training
            #     if self.train_args.amp:
            #         with self.optimizer_wrapper.optim_context():
            #             model_output = self.forward_with_train(images)
            #     else:
            #         model_output = self.forward_with_train(images)
            #
            #     # Loss forward
            #     # loss1->loss2->...
            #     loss_runner: BaseLossRunner
            #     for loss_runner in self.losses[loss_type.value]:
            #         loss_runner.model_output = model_output
            #         loss_runner.targets = targets
            #
            #         if self.train_args.amp:
            #             with self.optimizer_wrapper.optim_context():
            #                 loss_val = loss_runner.forward()  # noqa
            #         else:
            #             loss_val = loss_runner.forward()  # noqa
            #
            #         assert not torch.any(torch.isnan(loss_val))
            #         loss_results.update({loss_runner.loss_name: loss_val})
            #
            #     training_performance: dict = calc_performance(
            #         loss_type,
            #         self.train_args.topk,
            #         self.model_args.mask_classes,
            #         model_output,
            #         targets
            #     )
            #     if loss_type == Task.CLS:
            #         self.training_top1_data_logger.update(training_performance['top1'])
            #         self.training_topk_data_logger.update(training_performance['topk'])
            #     if loss_type == Task.SEG:
            #         self.training_miou_data_logger.update(training_performance['miou'])

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
                self.lr_scheduler.step(CONFIG('epochs') + curr_step / self.total_step)

            # Easy info display
            if curr_step % 20 == 0:
                print(
                    f'🚀[Training] Epoch:[{self.train_args.epoch}/{CONFIG("epochs")}] '
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

        if len(loss_results) != len(self.loss_weights):
            logger.error('len(loss_result)!=len(self.losses_weights)')

        ret = 0
        for kv, weight in zip(loss_results.items(), self.loss_weights):
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