import os
import time
import inspect
import random
import warnings
import os.path as osp
from math import ceil
from typing import Optional, List, Union

from tqdm import tqdm
import numpy as np
import torch.cuda
import torch.backends.cudnn
import torch.optim.lr_scheduler as torch_lr_scheduler
from loguru import logger
from torch.utils.data.distributed import DistributedSampler

import loss
import hooks
from helper.model import Model
from helper.DDP import DDPWrapper

from helper.dataloader import DataLoaderWrapper, BalancedBatchSampler

from helper.optim_wrapper import AmpOptimWrapper
from helper.dataset import ClassificationDataset, SegmentationDataSet
from helper.precision import data_precision

from augment.transforms import ClsTransform, SegTransform
import lr_scheduler.lr_adjustment as lr_adjustment
from register import Register, LR_PARAMS, LOSS

from utils.meter import Logger
from utils.draw_utils import draw_confusion_matrix
from utils.util import load_yaml, get_num_of_images, split_line, accuracy, generate_matrix, accl_miou, timer, Task
from utils.util import task_convert, join, get_time

warnings.filterwarnings("ignore")

Tensor = torch.Tensor
ListTensor = List[Tensor]


class Trainer:
    def __init__(
            self,
            project_config: dict,
            model_config: dict,
            train_config: dict,
            optimizer_config: dict,
            lr_config: dict,
            loss_config: dict,
            ddp_config: dict,
            classification_data_config: Optional[dict] = None,
            segmentation_data_config: Optional[dict] = None,
            **kwargs
    ) -> None:
        self.project_config = project_config
        self.train_config = train_config
        self.classification_data_config = classification_data_config
        self.segmentation_data_config = segmentation_data_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.lr_config = lr_config
        self.loss_config = loss_config
        self.ddp_config = ddp_config

        # DataSet ------------------------------------------------
        self.cls_train_dataset = None
        self.cls_val_dataset = None
        self.seg_train_dataset = None
        self.seg_val_dataset = None

        # Dataloader ------------------------------------------------
        self.cls_train_dataloader_wrapper = None
        self.cls_val_dataloader_wrapper = None
        self.seg_train_dataloader_wrapper = None
        self.seg_val_dataloader_wrapper = None
        self.cls_train_sampler = None
        self.cls_train_sampler = None
        self.cls_val_sampler = None
        self.seg_val_sampler = None
        self.dataloaders: list = []

        # Project Config ------------------------------------------------
        self.task = task_convert(task=self.project_config.get('task'))

        self.work_dir: str = self.project_config.get('work_dir')
        self.timestamp: str = ''
        self.topk: int = self.project_config['topk']
        if self.task != Task.MultiTask and (len(self.loss_config.keys()) > 2):
            logger.error(
                'Please check your loss_config: self.task != "multitask" but (len(self.loss_config.keys()) > 2)')
            raise

            # Training Config ------------------------------------------------
        self.seed: int = self.train_config.get('seed')
        self.epoch: int = self.train_config.get('epoch')
        self.max_epoch: int = self.train_config.get('max_epoch')
        self.print_freq: int = self.train_config.get('print_freq')
        self.workflow: list = self.train_config.get('workflow')
        self.deterministic: list = self.train_config.get('deterministic')

        # Model Config ------------------------------------------------
        self.model: Model = None  # type=Model
        self.dataloader_wrapper_set = {}
        self.optimizer = None
        self.optimizer_wrapper = None  #
        self.lr_scheduler = None
        self.losses = None
        self.hooks: List[Register] = [LR_PARAMS, LOSS]

        if self.task in [Task.CLS, Task.MultiTask]:
            self.cls_transforms = ClsTransform()

        if self.task in [Task.SEG, Task.MultiTask]:
            self.seg_transforms = SegTransform(
                wh=self.segmentation_data_config['train']['dataset_params']['wh'])

        self.step: int = 0
        self.draw_loss_data = {'x': [], 'y': []}
        self.ddp_wrapper = None
        self.use_ddp_mode = True if self.ddp_config['flag'] is True else False

        self.init_seeds()
        self.init_env()
        self.build_model()

        if self.use_ddp_mode:
            self.init_ddp()

        self.build_dataloader()
        self.build_optimizer()
        self.build_lr_scheduler()

        self.train_logger = Logger(
            task=self.task,
            total_step=self.step,
            prefix=f"🚀 [Train:{self.task.value}]"
        )
        self.val_logger = Logger(
            task=self.task,
            prefix="🚀 [Val]"
        )
        self.is_master_device: bool = (self.model.gpu == 0) or (not self.use_ddp_mode)

    def init_env(self):

        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        self.timestamp = time.strftime('%Y%m%d_%H%M%S',
                                       time.localtime(timestamp.item()))

        if osp.exists(self.work_dir) is False:
            logger.info(f'Create work dir:{self.work_dir}')
            os.makedirs(self.work_dir)

        logger.info('Init ENV done.')
        split_line()

    def init_ddp(self):
        self.ddp_wrapper = DDPWrapper(**self.ddp_config)
        self.ddp_wrapper.init_distributed_mode()

        self.model.set_gpu(self.ddp_wrapper.gpu)
        self.model.move_to_device()
        self.model.ddp_mode(self.ddp_wrapper.sync_bn)
        self.seed = self.ddp_wrapper.rank

    def build_model(self):
        self.model = Model(**self.model_config)
        self.model.init_model()  # building model and loading weight
        self.model.move_to_device()
        split_line()

    def _build_balanced_batch_sampler(self):
        n_classes = self.model.num_classes
        n_samples = ceil(
            self.classification_data_config['train']['dataloader_params']['batch_size'] / n_classes)

        self.classification_data_config['train']['dataloader_params']['shuffle'] = None
        self.classification_data_config['train']['dataloader_params']['batch_size'] = 1
        self.classification_data_config['train']['dataloader_params']['sampler'] = None
        self.classification_data_config['train']['dataloader_params']['drop_last'] = False
        batch_sampler = BalancedBatchSampler(
            torch.tensor(self.cls_train_dataset.targets),
            n_classes=n_classes,
            n_samples=n_samples
        )
        return batch_sampler

    def _build_ddp_dataloader(self, data_config: dict, train_sampler, val_sampler):
        data_config['train']['dataloader_params']['batch_sampler'] = None

        data_config['train']['dataloader_params']['sampler'] = train_sampler
        data_config['val']['dataloader_params']['sampler'] = val_sampler

        data_config['train']['dataloader_params']['shuffle'] = False

        data_config['train']['dataloader_params']['batch_size'] = self.ddp_wrapper.batch_size
        data_config['val']['dataloader_params']['batch_size'] = self.ddp_wrapper.batch_size

        data_config['train']['dataloader_params']['num_workers'] = self.ddp_wrapper.num_workers
        data_config['val']['dataloader_params']['num_workers'] = self.ddp_wrapper.num_workers

    # Building classification dataset and dataloader
    def _build_classification_ds_dl(self):
        self.classification_data_config['train']['dataset_params']['transform'] = self.cls_transforms.cls_image_trans
        self.classification_data_config['train']['dataset_params'][
            'target_transform'] = self.cls_transforms.cls_target_trans
        self.classification_data_config['val']['dataset_params']['transform'] = self.cls_transforms.val_trans

        self.cls_val_dataset = ClassificationDataset(
            **self.classification_data_config['val']['dataset_params']
        )
        self.cls_train_dataset = ClassificationDataset(
            **self.classification_data_config['train']['dataset_params']
        )

        if self.use_ddp_mode:
            self.cls_train_sampler = self.ddp_wrapper.get_distributed_sampler(
                self.cls_train_dataset)
            self.cls_val_sampler = self.ddp_wrapper.get_distributed_sampler(
                self.cls_val_dataset)
            self._build_ddp_dataloader(
                self.classification_data_config, self.cls_train_sampler, self.cls_val_sampler)
        else:
            if self.classification_data_config['train']['dataloader_params'].get('batch_sampler') is not None:
                if self.classification_data_config['train']['dataloader_params'][
                    'batch_sampler'] == 'BalancedBatchSampler':
                    self.classification_data_config['train']['dataloader_params'][
                        'batch_sampler'] = self._build_balanced_batch_sampler(
                    )

        self.cls_train_dataloader_wrapper = DataLoaderWrapper(
            dataset=self.cls_train_dataset,
            **self.classification_data_config['train']['dataloader_params'])
        self.cls_val_dataloader_wrapper = DataLoaderWrapper(
            dataset=self.cls_val_dataset,
            **self.classification_data_config['val']['dataloader_params'])
        _class_id_map_path = join(self.work_dir,
                                  'cls_class_id_map.txt')
        self.cls_train_dataset.save_class_id_map(
            _class_id_map_path, self.cls_train_dataset.class_to_idx)

        logger.info(
            f'Classification Dataset len:{len(self.cls_train_dataset)}')

    # Building segmentation dataset and dataloader
    def _build_segmentation_ds_dl(self):
        self.segmentation_data_config['train']['dataset_params']['transform'] = self.seg_transforms.seg_image_trans
        self.segmentation_data_config['train']['dataset_params'][
            'target_transform'] = self.seg_transforms.seg_target_trans
        self.segmentation_data_config['val']['dataset_params']['transform'] = self.seg_transforms.seg_image_trans
        self.segmentation_data_config['val']['dataset_params'][
            'target_transform'] = self.seg_transforms.seg_target_trans

        self.seg_train_dataset = SegmentationDataSet(
            **self.segmentation_data_config['train']['dataset_params'])
        self.seg_val_dataset = SegmentationDataSet(
            **self.segmentation_data_config['val']['dataset_params'])

        if self.use_ddp_mode:
            self.seg_train_sampler = self.ddp_wrapper.get_distributed_sampler(
                self.seg_train_dataset)
            self.seg_val_sampler = self.ddp_wrapper.get_distributed_sampler(
                self.seg_val_dataset)

            self._build_ddp_dataloader(
                self.segmentation_data_config, self.seg_train_sampler, self.seg_val_sampler)

        self.seg_train_dataloader_wrapper = DataLoaderWrapper(
            dataset=self.seg_train_dataset,
            **self.segmentation_data_config['train']['dataloader_params'])
        self.seg_val_dataloader_wrapper = DataLoaderWrapper(
            dataset=self.seg_val_dataset,
            **self.segmentation_data_config['val']['dataloader_params'])
        _class_id_map_path = join(self.work_dir,
                                  'seg_class_id_map.txt')
        self.seg_train_dataset.save_class_id_map(
            _class_id_map_path, self.seg_train_dataset.class_to_idx)

        logger.info(f'Segmentation Dataset len:{len(self.seg_train_dataset)}')

    def calc_expand_rate(self):
        # expanding data
        train_cls_num_of_images = get_num_of_images(
            self.classification_data_config['train']['dataset_params']['root'])
        train_seg_num_of_images = get_num_of_images(
            self.segmentation_data_config['train']['dataset_params']['root'])

        cls_expanding_rate = 1
        seg_expanding_rate = 1
        if train_cls_num_of_images > train_seg_num_of_images:
            difference = train_cls_num_of_images - train_seg_num_of_images
            cls_expanding_rate = 0
        else:
            difference = train_seg_num_of_images - train_cls_num_of_images
            seg_expanding_rate = 0

        cls_expanding_rate *= ceil(difference / train_cls_num_of_images)
        seg_expanding_rate *= ceil(difference / train_seg_num_of_images)
        return cls_expanding_rate, seg_expanding_rate

    def build_dataloader(self):
        if self.task not in [Task.MultiTask, Task.SEG, Task.CLS]:
            logger.error(f'Don`t support this task type:{self.task}')
            raise

        # Calc expand rate
        if self.task == Task.MultiTask:
            cls_expanding_rate, seg_expanding_rate = self.calc_expand_rate()
            self.classification_data_config['train']['dataset_params']['expanding_rate'] = cls_expanding_rate
            self.segmentation_data_config['train']['dataset_params']['expanding_rate'] = seg_expanding_rate
            logger.info(f'cls dataset expanding rate: x{cls_expanding_rate}')
            logger.info(f"seg dataset expanding rate: x{seg_expanding_rate}")

        # Build dataset and dataloader
        if self.task in [Task.CLS, Task.MultiTask]:
            self._build_classification_ds_dl()
        if self.task in [Task.SEG, Task.MultiTask]:
            self._build_segmentation_ds_dl()

        if self.task == Task.MultiTask:
            self.step = min(self.cls_train_dataloader_wrapper.step,
                            self.seg_train_dataloader_wrapper.step)
        elif self.task == Task.CLS:
            self.step = self.cls_train_dataloader_wrapper.step
        elif self.task == Task.SEG:
            self.step = self.seg_train_dataloader_wrapper.step

        logger.info('Build Dataloader')
        split_line()

    def init_seeds(self) -> None:

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # if you are using multi-GPU.
        np.random.seed(self.seed)  # Numpy module.
        random.seed(self.seed)  # Python random module.
        torch.manual_seed(self.seed)

        # random.seed(self.seed)
        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logger.info(f'Init seed:{self.seed} '
                    f'deterministic:{torch.backends.cudnn.deterministic} '
                    f'benchmark:{torch.backends.cudnn.benchmark}')
        split_line()

    def build_optimizer(self) -> None:
        optim = torch.optim.__dict__.get(self.optimizer_config['name'])
        if optim is None:
            logger.error(
                f'Do not get the {self.optimizer_config["name"]} optimizer from torch.optim.'
            )
            exit()

        self.optimizer = optim(self.model.parameters,
                               **self.optimizer_config['params'])
        self.optimizer_wrapper = AmpOptimWrapper(optimizer=self.optimizer)
        logger.info(f'Build optimizer: {self.optimizer_config["name"]}')
        split_line()

    def build_lr_scheduler(self) -> None:
        lr_name = self.lr_config['name']
        lr_scheduler = lr_adjustment.__dict__.get(lr_name)

        if lr_scheduler is None:
            lr_scheduler = torch_lr_scheduler.__dict__.get(lr_name)

        if lr_scheduler is None:
            logger.error(f'Do not get the {lr_name} lr_scheduler.')
            exit()

        if lr_name in LR_PARAMS.keys():
            res = self.call_hook(lr_name)
            if res is not None:
                self.lr_config['params'].update(res)

        self.lr_scheduler = lr_scheduler(self.optimizer,
                                         **self.lr_config['params'])
        logger.info(f'Build lr scheduler: {lr_name}')
        split_line()

    @timer
    def run(self):
        is_best = False
        best_top1 = 0.0
        best_MIoU = 0.0
        best_epoch = 0
        curr_top1 = 0.0
        curr_MIoU = 0.0
        best_weight_info = {}

        while self.epoch < self.max_epoch:
            # n*train -> k*val -> n*train->...
            for i, flow in enumerate(self.workflow):
                mode, epochs = flow
                run_one_epoch = getattr(self, mode)

                for _ in range(epochs):
                    if self.epoch >= self.max_epoch:
                        break

                    run_one_epoch()  # train() or val()

                    if mode == 'train':
                        self.epoch += 1
                    elif mode == 'val':
                        if self.is_master_device:

                            if self.task in [Task.CLS, Task.MultiTask]:
                                curr_top1 = self.val_logger.top1.avg

                            if self.task in [Task.SEG, Task.MultiTask]:
                                curr_MIoU = self.val_logger.MIoU.avg

                            # if (curr_top1 + curr_MIoU) > best_top1 + best_MIoU:
                            is_best = True
                            best_epoch = self.epoch
                            # best_top1 = curr_top1 if curr_top1 > best_top1 else best_top1
                            # best_MIoU = curr_MIoU if curr_MIoU > best_MIoU else best_MIoU

                            if self.task in [Task.CLS, Task.MultiTask]:
                                if is_best:
                                    best_weight_info['Acc'] = round(float(curr_top1), data_precision.Medium)
                                    print(f"[Best Top1]: {round(best_top1, data_precision.Medium)}")
                                print(f"[Curr Top1]: {round(curr_top1, data_precision.Medium)}")

                            if self.task in [Task.SEG, Task.MultiTask]:
                                if is_best:
                                    best_weight_info['MIoU'] = round(float(curr_MIoU), data_precision.Medium)
                                    print(f"[Best MIoU]: {round(best_MIoU, data_precision.Medium)}")
                                print(f"[Curr MIoU]: {round(curr_MIoU, data_precision.Medium)}")
                            best_weight_info['lr'] = float(round(self.optimizer_wrapper.lr[0], data_precision.High))

                            self.model.save_checkpoint(
                                save_path=join(self.work_dir, 'weights'),
                                epoch=self.epoch,
                                is_best=is_best,
                                **best_weight_info
                            )

                            best_weight_info = {}
                            is_best = False

    @timer
    def train(self):
        self.model.train()
        self.train_logger.clear()
        dataloaders = []
        if self.task in [Task.CLS, Task.MultiTask]:
            dataloaders.append(self.cls_train_dataloader_wrapper())
        if self.task in [Task.SEG, Task.MultiTask]:
            dataloaders.append(self.seg_train_dataloader_wrapper())

        if self.use_ddp_mode:

            if self.task in [Task.CLS, Task.MultiTask]:
                self.cls_train_sampler.set_epoch(self.epoch)

            if self.task in [Task.SEG, Task.MultiTask]:
                self.seg_train_sampler.set_epoch(self.epoch)

        if self.train_config['scheduler_step_batch'] is False:
            self.lr_scheduler.step()

        self.optimizer_wrapper.show_lr()

        # base_step = self.epoch * self.step
        datas: tuple
        for i, datas in enumerate(zip(*dataloaders)):
            if self.task == Task.MultiTask:
                cls_data, seg_data = datas  # datas.shape=tuple(data1,data2)
            else:
                cls_data = seg_data = datas[0]  # datas.shape=tuple(data1)

            losses = []
            performance_data = []
            # classification or segmentation
            for curr_task in self.loss_config.keys():
                if curr_task == Task.CLS.value:
                    curr_task = Task.CLS
                    input_data = cls_data

                elif curr_task == Task.SEG.value:
                    curr_task = Task.SEG
                    input_data = seg_data

                else:
                    continue

                images, targets = input_data
                images = self.move_to_device(images)
                targets = self.move_to_device(targets)

                model_output = self.forward_with_train(images, targets)
                losses.extend(self.calc_loss(curr_task, model_output, targets))
                performance: dict = self.calc_performance(curr_task, model_output, targets)

                if curr_task == Task.CLS:
                    performance_data += [performance['acc1'],
                                         performance['accn']]
                elif curr_task == Task.SEG:
                    performance_data += [performance['miou']]

            # Backward
            loss_sum = self.loss_sum(losses, self.loss_config.get('loss_weights'))
            self.optimizer_wrapper.update_params(loss_sum)

            if self.train_config['scheduler_step_batch'] is True:
                self.lr_scheduler.step(self.epoch + i / self.step)

            # Sync loss
            if self.use_ddp_mode:
                losses = self.ddp_wrapper.sync_tensor(
                    losses, self.model.device)
                losses = np.array(losses) / self.ddp_wrapper.world_size

                performance_data = self.ddp_wrapper.sync_tensor(
                    performance_data, self.model.device)
                performance_data = np.array(
                    performance_data) / self.ddp_wrapper.world_size

            if self.is_master_device:
                curr_bs = targets.shape[0]
                self.update_loss_info(losses, curr_bs)
                self.update_performance_info(
                    self.task, self.train_logger, performance_data, curr_bs)

                if i % self.print_freq == 0:
                    self.train_logger.display(self.task, i, self.epoch)

            # self.draw_loss_data['y'].append(loss_sum.item())
            # self.draw_loss_data['x'].append(base_step + i)
            # draw_loss(
            #     self.draw_loss_data['x'],
            #     self.draw_loss_data['y'],
            #     save_path=join(self.project_config['work_dir'], 'loss.png')
            # )
        # time.sleep(1)

    @timer
    def val(self):
        self.model.eval()
        self.val_logger.clear()
        # print(self.loss_config.keys())
        for task in self.loss_config.keys():

            task = task_convert(task)
            if task == Task.SEG:
                self._val_impl(task,
                               self.seg_val_dataloader_wrapper,
                               self.seg_val_sampler,
                               )
                if self.is_master_device:
                    avg_miou = round(self.val_logger.MIoU.avg, data_precision.Medium)
                    print(
                        f'Segmentation Val MIoU(Avg): {avg_miou}'
                    )
                    with open(os.path.join(self.project_config['work_dir'], 'Val_MIoU.txt'), 'a+') as f:
                        f.writelines(f'{get_time()}\tEpoch: {self.epoch}\tMiou: {avg_miou}\n')


            elif task == Task.CLS:
                confusion_matrix = self._val_impl(task,
                                                  self.cls_val_dataloader_wrapper,
                                                  self.cls_val_sampler,
                                                  )

                if self.is_master_device:
                    top1 = round(self.val_logger.top1.avg, data_precision.Medium)
                    print(
                        f'Classification Val Top1(Avg): {top1} '
                        f'Top{self.topk}(Avg): {round(self.val_logger.topn.avg, data_precision.Medium)}')
                    with open(os.path.join(self.project_config['work_dir'], 'Val_Top1.txt'), 'a+') as f:
                        f.writelines(f'{get_time()}\tEpoch: {self.epoch}\tTop1: {top1}\n')
                    # draw_confusion_matrix(numclasses=self.model.num_classes,
                    #                       labels=self.cls_train_dataset.labels,
                    #                       matrix=confusion_matrix,
                    #                       save_path=join(self.project_config['work_dir'], 'cls_confusion_matrix.png'))
            else:
                continue
        time.sleep(1)

    def _val_impl(
            self,
            task: Task,
            val_dataloader_wrapper: DataLoaderWrapper,
            distributed_sampler: Optional[DistributedSampler] = None,
    ) -> Optional[np.ndarray]:

        self.val_logger.set_total_step(val_dataloader_wrapper.step)

        if self.use_ddp_mode:
            distributed_sampler.set_epoch(self.epoch)

        confusion_matrix = None
        if task == Task.CLS:
            confusion_matrix = np.zeros((self.model.num_classes,
                                         self.model.num_classes))
        val_performance_data = {
            'acc1': 0,
            'accn': 0,
            'miou': 0,
        }
        for data in tqdm(val_dataloader_wrapper()):
            images, targets = data
            # print(targets)
            images = self.move_to_device(images)
            targets = self.move_to_device(targets)

            model_output = self.forward_with_val(images, targets)
            performance: dict = self.calc_performance(task,
                                                      model_output,
                                                      targets)
            if task == Task.CLS:
                val_performance_data['acc1'] += performance['acc1']
                val_performance_data['accn'] += performance['accn']
            elif task == Task.SEG:
                val_performance_data['miou'] += performance['miou']

        performance_data = []
        if self.use_ddp_mode:
            logger.info(f'Sync val performance data')
            if task == Task.CLS:
                sync_data = [val_performance_data['acc1'],
                             val_performance_data['accn']]
            elif task == Task.SEG:
                sync_data = [val_performance_data['miou']]

            performance_data = self.ddp_wrapper.sync_tensor(
                sync_data, self.model.device)
            performance_data = np.array(
                performance_data) / self.ddp_wrapper.world_size / val_dataloader_wrapper.step
        else:
            if task == Task.CLS:
                performance_data = [
                    val_performance_data['acc1'] / val_dataloader_wrapper.step,
                    val_performance_data['accn'] / val_dataloader_wrapper.step
                ]
            elif task == Task.SEG:
                performance_data = [
                    val_performance_data['miou'] / val_dataloader_wrapper.step
                ]

        if self.is_master_device:
            self.update_performance_info(
                task, self.val_logger, performance_data, targets.shape[0])

            if task == Task.CLS:

                if isinstance(model_output, list):
                    model_output = model_output[0]
                    if isinstance(model_output, list):
                        model_output = model_output[0]

                model_output = model_output.argmax(1)
                confusion_matrix += generate_matrix(
                    self.model_config['num_classes'], model_output, targets)
        return confusion_matrix

    def forward_with_val(self, images: Tensor, targets: Tensor) -> Union[Tensor, ListTensor]:
        if images is None or targets is None:
            logger.error('train_data is None')
            raise

        with torch.no_grad():
            model_output = self.model(images)

        return model_output

    def forward_with_train(self, images: Tensor, targets: Tensor) -> Union[Tensor, ListTensor]:
        if images is None or targets is None:
            logger.error('train_data is None')
            raise
        with self.optimizer_wrapper.optim_context():
            model_output = self.model(images)

        return model_output

    def calc_loss(self, task: Task, model_output: Union[Tensor, ListTensor], targets: Tensor) -> ListTensor:

        loss_pipline = {}
        loss_pipline.update(self.loss_config[task.value])  # loss1->loss2->...
        default_params = {
            'model_output': model_output,
            'targets': targets,
            'device': self.model.device
        }
        losses = []
        for loss_name, params in loss_pipline.items():
            params.update(default_params)
            with self.optimizer_wrapper.optim_context():
                loss_obj = self.call_hook(loss_name, **params)
                losses.append(loss_obj.forward())

        return losses

    def calc_performance(self, task: Task, model_output, targets: Tensor) -> dict:
        performance = {
            'acc1': 0,
            'accn': 0,
            'miou': 0,
        }

        if task == Task.SEG:
            miou = self.calc_miou(model_output, targets)
            performance['miou'] = miou.item()

        elif task == Task.CLS:
            acc1, accn = self.calc_accuracy(model_output, targets)
            performance['acc1'] = acc1.item()
            performance['accn'] = accn.item()

        # elif task == Task.SEG:
        #     miou = self.calc_miou(model_output, targets)
        #     performance['miou'] = miou.item()

        # if task == Task.CLS:
        #     acc1, accn = self.calc_accuracy(model_output, targets)
        #     performance['acc1'] = acc1.item()
        #     performance['accn'] = accn.item()
        #
        # elif task == Task.SEG:
        #     miou = self.calc_miou(model_output, targets)
        #     performance['miou'] = miou.item()

        return performance

    def update_performance_info(self, task: Task, logger: Logger, performance: Union[list, np.ndarray],
                                batch_size: int):
        performance = self.to_constant(performance)
        if task in [Task.CLS, Task.MultiTask]:
            logger.top1.update(performance[0], batch_size)
            logger.topn.update(performance[1], batch_size)

        if task == Task.SEG:
            logger.MIoU.update(performance[0], batch_size)

        if task == Task.MultiTask:
            logger.MIoU.update(performance[2], batch_size)

    def update_loss_info(self, losses: Union[np.ndarray, ListTensor], batch_size: int):
        cls_losses = 0
        seg_losses = 0
        losses = self.to_constant(losses)
        for loss_type in self.loss_config.keys():
            if loss_type not in [Task.CLS.value, Task.SEG.value]:
                continue
            elif loss_type == Task.CLS.value:
                cls_losses += losses.pop(0)

            elif loss_type == Task.SEG.value:
                seg_losses += losses.pop(0)
        if self.task in [Task.CLS, Task.MultiTask]:
            self.train_logger.loss_cls.update(cls_losses, batch_size)
        if self.task in [Task.SEG, Task.MultiTask]:
            self.train_logger.loss_seg.update(seg_losses, batch_size)

    def move_to_device(self, data: Tensor) -> Tensor:
        if self.model.device != 'cpu':
            return data.cuda(self.model.gpu, non_blocking=True)
        return data

    def call_hook(self, key: str, **kwargs):
        get = False
        for hook in self.hooks:
            if inspect.isfunction(hook.get(key)):  # return function result
                func = hook.get(key)
                get = True
                return func(**kwargs)

            elif inspect.isclass(hook.get(key)):  # return class instance(obj)
                obj = hook.get(key)(**kwargs)
                get = True
                return obj

        if not get:
            return None
            # logger.error(f'Don`t found this fn_name: {key}')
            # raise

    def calc_accuracy(self, model_output, targets: Tensor) -> tuple:
        if isinstance(model_output, list):

            model_output = model_output[0]
            if isinstance(model_output, list):
                model_output = model_output[0]

        acc1, acc_n = accuracy(model_output, targets, topk=(1, self.topk))
        return acc1, acc_n

    def calc_miou(self, model_output, targets: Tensor) -> np.ndarray:
        if isinstance(model_output, list):

            model_output = model_output[1]
            if isinstance(model_output, list):
                model_output = model_output[0]

        output = model_output.argmax(1)  # [bs,cls,h,w]->[bs,h,w]
        mask_target_seg = targets.squeeze(1)  # [bs,1,h,w]->[bs,h,w]

        confusion_matrix = generate_matrix(self.model_config['mask_classes'],
                                           output, mask_target_seg)
        iou, miou = accl_miou(confusion_matrix)
        return miou

    @staticmethod
    def loss_sum(losses: ListTensor, loss_weights: Optional[list] = None) -> Tensor:

        if loss_weights is None:
            loss_weights = [1]
            loss_weights = loss_weights * len(losses)

        if len(loss_weights) < len(losses):
            tmp_weights = [1]
            tmp_weights = tmp_weights * (len(losses) - losses(loss_weights))
            loss_weights.extend(tmp_weights)

        if len(losses) != len(loss_weights):
            logger.error('len(losses)!=len(weights)')
            raise
        ret = 0
        for loss_val, weight in zip(losses, loss_weights):
            ret += loss_val * weight
        return ret

    @staticmethod
    def to_constant(x: Union[np.float32, np.float64, Tensor, np.ndarray]):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, np.float64) or isinstance(x, np.float32):
            return x.item()
        elif isinstance(x, Tensor):
            return x.tolist()
        else:
            return x
