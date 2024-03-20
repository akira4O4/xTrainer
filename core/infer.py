import os
import os.path as osp
import shutil
from abc import ABC
from typing import Callable, Optional, Union, List

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from bunch import Bunch

from loguru import logger
from helper.model import Model
from helper.precision import data_precision
from utils.util import get_images, recursion_bunch, check_exists, color_list, Task
from utils.util import task_convert, join
from augment.transforms import ClsTransform, SegTransform
from utils.util import timer
import time

__all__ = ['Infer']


class Infer:
    def __init__(
            self,
            project_config: dict,
            model_config: dict,
            test_config: dict,
            classification_data_config: Optional[dict] = None,
            segmentation_data_config: Optional[dict] = None,
            **kwargs
    ):
        self.model = None
        self.processing_funcs = {
            Task.MultiTask: self.multi_task,
            Task.SEG: self.segmentation_postprocessing,
            Task.CLS: self.classification_postprocessing
        }
        self.project_config = project_config
        self.classification_data_config = classification_data_config
        self.segmentation_data_config = segmentation_data_config

        self.task = task_convert(self.project_config.get('task'))
        if self.task in [Task.CLS, Task.MultiTask]:
            self.wh = self.classification_data_config['train']['dataset_params']['wh']
        if self.task in [Task.SEG, Task.MultiTask]:
            self.wh = self.segmentation_data_config['train']['dataset_params']['wh']

        self.val_transforms = None
        self.model_config = model_config
        self.cls_id2class = None
        self.seg_id2class = None
        self.classify_output_path = None
        self.segmentation_output_path = None
        self.cls_transforms = None
        self.seg_transforms = None
        self.test_config = test_config
        # self.test_dir =

    def init(self):
        self.init_id_map()
        self.build_model()
        self.init_transform()
        self.init_output_dir()

    def init_transform(self):
        if self.task in [Task.CLS, Task.MultiTask]:
            self.cls_transforms = ClsTransform()
            self.val_transforms = self.cls_transforms.val_trans

        if self.task in [Task.SEG, Task.MultiTask]:
            self.seg_transforms = SegTransform(
                wh=self.segmentation_data_config['train']['dataset_params']['wh'])
            self.val_transforms = self.seg_transforms.val_trans

    def init_output_dir(self):
        if self.project_config['work_dir'] is None:
            logger.error(f'Please setting project dir.')
            logger.error(f'[Exit]')
            exit()
        if check_exists(self.project_config['work_dir']) is False:
            os.makedirs(self.project_config['work_dir'])

        save_path = osp.join(self.project_config['work_dir'], 'temp')
        self.classify_output_path = osp.join(
            save_path, Task.CLS.value)  # temp/classify
        self.segmentation_output_path = osp.join(
            save_path, Task.SEG.value)  # temp/segmentation

        need_mk_dir = [
            self.classify_output_path,
            self.segmentation_output_path
        ]

        for item in need_mk_dir:
            if check_exists(item):
                shutil.rmtree(item)
                logger.info(f'[Clean]:{item}')
            os.makedirs(item)
        logger.success('Init output dir done.')

    def build_model(self):
        self.model = Model(**self.model_config)
        self.model.setting_weight(self.test_config['weight'])
        self.model.init_model()  # building model and loading weight
        self.model.move_to_device()

    def init_id_map(self):
        seg_class_id_map_path = osp.join(
            self.project_config['work_dir'], 'seg_class_id_map.txt')
        cls_class_id_map_path = osp.join(
            self.project_config['work_dir'], 'cls_class_id_map.txt')

        if check_exists(cls_class_id_map_path):
            self.cls_id2class = self.load_class_id_map(cls_class_id_map_path)
            logger.success('Loading cls_class_id_map.txt')
        if check_exists(seg_class_id_map_path):
            self.seg_id2class = self.load_class_id_map(seg_class_id_map_path)
            logger.success('Loading seg_class_id_map.txt')

    @staticmethod
    def copy_image(image: str, root: str, save_path: str):
        image_save_path = image.replace(root, save_path)
        image_name = osp.basename(image)
        image_save_dir = image_save_path.split(image_name)[0]
        os.makedirs(image_save_dir, exist_ok=True)  # check output dir
        shutil.copy(image, image_save_dir)

    def preprocess(self, image: str, transforms: Callable):
        img = Image.open(image)
        img_w, img_h = img.size
        if img_w != self.wh[0] or img_h != self.wh[1]:
            img = img.resize(self.wh, Image.BILINEAR)
        img = img.convert('RGB')
        img = transforms(img)

        return img

    def infer_core(self, image: str, transforms: Callable):

        self.model.eval()
        with torch.no_grad():
            img = self.preprocess(image, transforms)
            if img is None:
                return None

            img = img.unsqueeze(0).cuda(self.model.gpu)
            predict = self.model(img)

        return predict

    def run(self):
        postprocessing_func = self.processing_funcs.get(self.task)
        logger.info(f'Test Dir: {self.test_config["test_dir"]}')
        images = get_images(self.test_config['test_dir'])
        for image in tqdm(images):
            model_output = self.infer_core(
                image, self.val_transforms)
            postprocessing_func(
                image=image,
                model_output=model_output,
                **self.test_config)

    def classification_postprocessing(
            self,
            image: str,
            model_output: Union[torch.Tensor, List[torch.Tensor]],
            cls_threshold: Optional[List[int]] = None,
            **kwargs
    ) -> None:

        if isinstance(model_output, list):
            model_output = model_output[0]
        model_output = model_output.squeeze(0).softmax(
            0).cpu().numpy()  # [score1,score2,...]
        # adjust th len
        if cls_threshold is None:
            cls_threshold = []
        if len(cls_threshold) < len(model_output):
            _t_th = [cls_threshold[0]]
            _t_th = _t_th * (len(model_output) - len(cls_threshold))
            cls_threshold.extend(_t_th)
            logger.info(f'cls threshold:{cls_threshold}')
        good_cnt = 0
        ng_cnt = 0
        good_idx = self.test_config['good_idx']
        good_score = model_output[good_idx]
        good_score = np.round(float(good_score), data_precision.Medium)
        is_good_flag = False
        if good_score >= cls_threshold[good_idx]:
            good_cnt += 1
            good_label = self.cls_id2class.get(good_idx)
            if good_label is None:
                good_label = str(good_idx)
            save_path = osp.join(self.classify_output_path, good_label)
            self.copy_image(
                image, root=self.test_config['test_dir'], save_path=save_path)
            is_good_flag = True

        ng_max_score_idx = model_output[1:].argmax(0) + 1
        ng_max_score = model_output[ng_max_score_idx]
        ng_max_score = np.round(float(ng_max_score), data_precision.Medium)
        ng_pred_label = self.cls_id2class.get(ng_max_score_idx)

        if ng_pred_label is None:
            ng_pred_label = str(ng_max_score_idx)

        if is_good_flag is False:
            if ng_max_score > cls_threshold[ng_max_score_idx]:
                save_path = osp.join(self.classify_output_path, ng_pred_label)
                self.copy_image(image, root=self.test_config['test_dir'], save_path=save_path)
            else:
                save_path = osp.join(self.classify_output_path, 're-good')
                self.copy_image(image, root=self.test_config['test_dir'], save_path=save_path)

        with open(osp.join(self.classify_output_path, 'classification_output.txt'), 'a') as f:
            f.write(
                f'[Good_Th]:{cls_threshold[good_idx]}    '
                f'[Image]:{image}    '
                f'[GS]:{good_score}    '
                f'[Max_NGS]:{ng_max_score}    '
                f'[NG_Label]:{ng_pred_label}\n'
            )

    @timer
    def segmentation_postprocessing(
            self,
            image: str,
            model_output: Union[torch.Tensor, List[torch.Tensor]],
            sum_method: Optional[bool] = True,
            seg_threshold: Optional[List[int]] = None,
            **kwargs
    ):
        # output:Tensor or [Tensor,...]
        if isinstance(model_output, list):
            model_output = model_output[0]

        img = cv2.imread(image)
        h, w, c = img.shape

        if h != self.wh[1] or w != self.wh[0]:
            img = cv2.resize(img, self.wh, interpolation=cv2.INTER_NEAREST)
        img = img.copy().astype(np.uint8)

        mask = model_output.squeeze(0).cpu().numpy().argmax(0)
        vis_predict = mask.copy().astype(np.uint8)  # 0-255
        vis_predict_seg = np.zeros((img.shape[0], img.shape[1], 3)) + [0, 0, 0]
        cls_and_area = {}

        if seg_threshold is None:
            seg_threshold = []
        # adjust threshold
        if len(seg_threshold) < self.model.mask_classes:
            extend_zero = [0 for i in range(
                self.model.mask_classes - len(seg_threshold))]
            seg_threshold.extend(extend_zero)

        is_good = True
        for curr_cls in range(0, self.model.mask_classes):

            index = np.where(vis_predict == curr_cls)

            if (len(index[0]) * len(index[1])) == (img.shape[0] * img.shape[1]):
                continue

            if sum_method is True:
                t = seg_threshold[curr_cls]
                num_of_bad = len(index[0])
                if num_of_bad >= t:

                    if curr_cls > 0:
                        is_good = False

                    vis_predict_seg[index[0], index[1], :] = color_list[curr_cls]
                    label = self.seg_id2class.get(curr_cls)

                    if label is None:
                        label = curr_cls
                    cls_and_area.update({
                        label: len(index[0])
                    })


            else:
                mask_index = np.zeros(
                    (img.shape[0], img.shape[1]), dtype=np.uint8)
                mask_index[index[0], index[1]] = 255
                cnts, _ = cv2.findContours(
                    mask_index, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                draw_cns = []
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area >= seg_threshold[curr_cls]:
                        draw_cns.append(cnt)

                cv2.drawContours(vis_predict_seg, draw_cns, -1,
                                 tuple(color_list[curr_cls]), -1)

        if not is_good:
            with open(osp.join(self.segmentation_output_path, 'seg_area.txt'), 'a') as f:
                f.write(
                    f'[image]:{os.path.basename(image)}]\t'
                    f'[data]:{cls_and_area}\n'
                )
            # vis_predict_color = vis_predict_seg.astype(np.uint8)
            vis_predict_seg = vis_predict_seg.astype(np.uint8)
            vis_addweight = cv2.addWeighted(img, 0.3, vis_predict_seg, 0.5, 0)

            image_basename = osp.basename(image)
            file_name, ext = osp.splitext(image_basename)

            cv2.imwrite(osp.join(self.segmentation_output_path, image_basename), img)
            cv2.imwrite(osp.join(self.segmentation_output_path, file_name + '_seg.png'), vis_addweight)

    def multi_task(
            self,
            image: str,
            model_output: Union[torch.Tensor, List[torch.Tensor]],
            sum_method: Optional[bool] = True,
            need_segment: Optional[bool] = True,
            cls_threshold: Optional[List[int]] = None,
            seg_threshold: Optional[List[int]] = None,
            **kwargs
    ):
        predict_cls = []
        predict_seg = []
        # multi_task.output:[cls,mask] or [[cls,...],[mask,..]]
        if isinstance(model_output, list):
            predict_cls = model_output[0][0] if isinstance(
                model_output[0], list) else model_output[0]
            predict_seg = model_output[1][0] if isinstance(
                model_output[1], list) else model_output[1]

        if need_segment:
            self.segmentation_postprocessing(
                image, predict_seg, sum_method, seg_threshold)

        self.classification_postprocessing(image, predict_cls, cls_threshold)

    @staticmethod
    def load_class_id_map(class_id_map_path: str) -> dict:

        if osp.exists(class_id_map_path) is False:
            logger.warning(f'Don`t found file:{class_id_map_path}.')
            return {}

        class_id_map = {}
        with open(class_id_map_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                # k=0,1,2,...
                # v=classes1,classes2,...
                k, v = line.split("x", 1)
                # class_id_map={0:"classes1",1:"classes2",...}
                class_id_map[int(k)] = v
        return class_id_map
