import os
import shutil
from copy import deepcopy
from typing import Union, List, Any

import cv2
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from mlflow import log_metric

from xtrainer.core.lr_scheduler import LRSchedulerWrapper
from xtrainer.core.preprocess import (
    ClsImageT,
    ClsTargetT,
    ClsValT,
    SegImageT,
    SegValT,
    InferT
)

from xtrainer import CONFIG, DEFAULT_OPTIMIZER, COLOR_LIST

from xtrainer.core.model import Model
from xtrainer.core.optim import (
    AMPOptimWrapper,
    OptimWrapper,
    build_optimizer_wrapper,
    build_amp_optimizer_wrapper
)

from xtrainer.dataset.segmentation import SegmentationDataSet
from xtrainer.dataset.classification import ClassificationDataset, BalancedBatchSampler
from xtrainer.utils.common import (
    save_yaml,
    error_exit,
    round4,
    round8,
    timer,
    align_size,
    get_images,
    get_time,
    print_of_mt,
    print_of_seg,
    print_of_cls,
    Colors,
    safe_imread,
    load_yaml,
    check_dir,
    save_json
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
    convert_optimizer_state_dict_to_fp16,
    ToDevice
)


class Predictor:
    def __init__(self):
        self.task = Task(CONFIG('task'))
        logger.info(f"Task: {Colors.BLUE}{self.task}{Colors.ENDC}")

        # Init Model --------------------------------------------------------------------------------------------------
        self.model: Model = None  # noqa
        self.init_model()

        # Init Model --------------------------------------------------------------------------------------------------
        self.transform = InferT(tuple(CONFIG('wh')))
        self.to_device = ToDevice(CONFIG('device'))

        # Init label --------------------------------------------------------------------------------------------------
        self.cls_label: List[str] = []
        self.seg_label: List[str] = []
        self.load_label()

        # Init output dir ----------------------------------------------------------------------------------------------
        self.cls_save = ''  # project/runs/classification
        self.seg_save = ''  # project/runs/segmentation
        self.seg_image_output = ''  # project/runs/segmentation/images
        self.seg_result_output = ''  # project/runs/segmentation/results
        self.init_output_dir()

    def init_output_dir(self) -> None:
        if self.task.CLS or self.task.MT:
            self.cls_save = os.path.join(CONFIG('project'), 'runs', f'classification.{get_time()}')
            check_dir(self.cls_save)

        if self.task.SEG or self.task.MT:
            self.seg_save = os.path.join(CONFIG('project'), 'runs', f'segmentation.{get_time()}')
            check_dir(self.seg_save)

            self.seg_image_output = os.path.join(self.seg_save, 'images')
            check_dir(self.seg_image_output)

            self.seg_result_output = os.path.join(self.seg_save, 'results')
            check_dir(self.seg_result_output)

    def load_label(self) -> None:
        if self.task.CLS or self.task.MT:
            self.cls_label = load_yaml(CONFIG('cls_label'))
            self.cls_label.sort()
        if self.task.SEG or self.task.MT:
            self.seg_label = load_yaml(CONFIG('seg_label'))
            self.seg_label.sort()

    def run(self) -> None:
        images: List[str] = get_images(CONFIG('source'))

        for image in tqdm(images, desc='Predict: '):

            if not os.path.exists(image):
                logger.error(f'Can`t open image: {image}.')
                continue

            im: torch.Tensor = self.preprocess(image)
            output = self.infer(im)

            if self.task.CLS:
                self.classification(output, image)
            elif self.task.SEG:
                self.segmentation(output[0], image)
            elif self.task.MT:
                self.multitask(output, image)

    @staticmethod
    def imread(path: str) -> np.ndarray:
        im = safe_imread(path)
        if im is None:
            raise FileNotFoundError(f'Don`t open image: {path}')

        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif im.ndim == 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

        return im

    def preprocess(self, image: str) -> torch.Tensor:
        im = self.imread(image)
        im = self.transform(im)
        im = im.unsqueeze(0)
        im = self.to_device(im)
        return im

    def infer(self, image: torch.Tensor) -> Any:
        outputs = self.model(image)
        return outputs

    def classification(self, output, image: str) -> None:
        output = output.squeeze(0)  # (1,C)->(C,)
        output = output.softmax(0)
        output = output.cpu().numpy()

        sort_idx = np.argsort(-np.array(output))  # down-sort

        no_result = True
        score = 0.0
        label = 'NoResult'

        for idx in sort_idx:
            score = float(output[idx])
            thr = CONFIG('cls_thr')[idx]
            if score >= thr:
                label: str = self.cls_label[idx]
                save_path = os.path.join(self.cls_save, label)
                check_dir(save_path)
                shutil.copy(image, save_path)
                no_result = False
                break

        if no_result:
            save_path = os.path.join(self.cls_save, 'no_result')
            check_dir(save_path)
            shutil.copy(image, save_path)

        with open(os.path.join(self.cls_save, 'result.txt'), 'a') as f:
            f.write(
                f'[Image]:{image}\t'
                f'[Score]:{round8(score)}\t'
                f'[Label]:{label}\n'
            )

    def segmentation(self, output, image: str) -> None:
        # Decode mask
        mask = output.squeeze(0)  # (1,C,H,W)-(C,H,W)
        mask = mask.cpu().numpy()
        mask = mask.argmax(0)  # (C,H,W)->(1,H,W)

        # Prepare draw image
        im = safe_imread(image)
        ih, iw, ic = im.shape
        mask = mask.astype(np.uint8)  # 0-255
        draw_mask = np.zeros((ih, iw, 3)) + [0, 0, 0]  # black image

        no_result = True
        record = {'image': image}
        nc = CONFIG('segmentation.classes') + 1

        for label_idx in range(1, nc):
            thr = CONFIG('seg_thr')[label_idx - 1]
            label = self.seg_label[label_idx - 1]

            color = tuple(map(int, COLOR_LIST[label_idx]))

            if thr == -1:  # just ignore
                record[label] = -1
                continue

            index = np.where(mask == label_idx)
            num_of_pixel = len(index[0])

            if CONFIG('sum_method'):
                record[label] = num_of_pixel
                if num_of_pixel >= thr:
                    draw_mask[index[0], index[1], :] = color
                    no_result = False

            # TODO: BUG
            else:
                w, h = CONFIG('wh')
                mask_index = np.zeros((h, w), dtype=np.uint8)
                mask_index[index[0], index[1]] = 255
                cnts, _ = cv2.findContours(mask_index, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                draw_cns = []
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area >= thr:
                        draw_cns.append(cnt)
                cv2.drawContours(draw_mask, draw_cns, -1, color, -1)

        basename = os.path.basename(image)
        name, suffix = os.path.splitext(basename)

        if not no_result:
            # Save draw image
            draw_mask = draw_mask.astype(np.uint8)
            draw_mask = cv2.addWeighted(im, 0.3, draw_mask, 0.5, 0)

            cv2.imwrite(os.path.join(self.seg_image_output, basename), im)
            cv2.imwrite(os.path.join(self.seg_image_output, name + '_mask.png'), draw_mask)  # image_mask.png

        # Save result json
        save_json(record, os.path.join(self.seg_result_output, basename.replace(suffix, '.json')))

    def multitask(self, output, image: str) -> None:
        cls_output = output[0][0]
        seg_output = output[1][0]

        self.classification(cls_output, image)
        self.segmentation(seg_output, image)

    def init_model(self) -> None:
        num_classes: int = CONFIG('classification.classes')
        mask_classes: int = CONFIG('segmentation.classes') + 1

        if num_classes == mask_classes == 0:
            logger.error("num_classes == mask_classes == 0")
            error_exit()

        self.model = Model(
            CONFIG('model'),
            num_classes,
            mask_classes,
            CONFIG("pretrained"),
            CONFIG('test_weight'),
            CONFIG('device')
        )
        self.model.init()
        self.model.eval()
