import os
import shutil
from typing import Callable, Optional, Union, List

import cv2
import torch
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm
from dataclasses import asdict

from utils.util import load_yaml, color_list, get_images
from .task import Task, task_convert
from .args import ProjectArgs, TrainArgs, ModelArgs, TestArgs
from .transforms import ClassificationTransform, SegmentationTransform
from .model import Model

__all__ = ['Infer']


class Infer:
    def __init__(
            self,
            config_path: str
    ):
        self.config_path = config_path
        if not os.path.exists(config_path):
            logger.error(f'Can`t found the {config_path}.')
            exit()

        # Init Args
        self.config = load_yaml(config_path)
        self.project_args = ProjectArgs(**self.config['project_config'])
        self.train_args = TrainArgs(**self.config['train_config'])
        self.model_args = ModelArgs(**self.config['model_config'])
        self.test_args = TestArgs(**self.config['test_config'])

        self.task = task_convert(self.project_args.task)

        if self.task in [Task.MultiTask, Task.CLS]:
            self.classification_data_args: dict | None = self.config.get('classification_data_config')
            self.wh = self.classification_data_args['dataset']['train']['wh']

        if self.task in [Task.MultiTask, Task.SEG]:
            self.segmentation_data_args: dict | None = self.config.get('segmentation_data_config')
            self.wh = self.segmentation_data_args['dataset']['train']['wh']

        # Build Model
        self.model: Model = None  # noqa
        self.init_model()

        self.processing_funcs = {
            Task.MultiTask: self.multi_task,
            Task.SEG: self.segmentation_postprocessing,
            Task.CLS: self.classification_postprocessing
        }

        self.val_transforms = None
        self.cls_id2class = None
        self.seg_id2class = None
        self.classification_output_path = None
        self.segmentation_output_path = None
        self.classification_transforms = None
        self.segmentation_transforms = None

        self.result_output_path = ''

        self.init()

    def init(self):
        self.init_id_map()
        self.init_transform()
        self.init_output_dir()

    def init_transform(self):
        if self.task in [Task.CLS, Task.MultiTask]:
            if self.test_args.need_resize:
                self.classification_transforms = ClassificationTransform(resize_wh=self.wh)
            else:
                self.classification_transforms = ClassificationTransform()

            self.val_transforms = self.classification_transforms.normalize_transform

        if self.task in [Task.SEG, Task.MultiTask]:
            if self.test_args.need_resize:
                self.segmentation_transforms = SegmentationTransform(resize_wh=self.wh)
            else:
                self.segmentation_transforms = SegmentationTransform()

            self.val_transforms = self.segmentation_transforms.normalize_transform

    def init_output_dir(self) -> None:

        if os.path.exists(self.project_args.work_dir) is False:
            os.makedirs(self.project_args.work_dir)

        self.result_output_path = os.path.join(self.project_args.work_dir, 'temp')
        self.classification_output_path = os.path.join(self.result_output_path, Task.CLS.value)  # temp/classify
        self.segmentation_output_path = os.path.join(self.result_output_path, Task.SEG.value)  # temp/segmentation

        need_mk_dir = [
            self.classification_output_path,
            self.segmentation_output_path
        ]

        for item in need_mk_dir:
            if os.path.exists(item):
                shutil.rmtree(item)
                logger.info(f'[Clean]:{item}')
            os.makedirs(item)
        logger.success('Init output dir done.')

    def init_model(self) -> None:
        self.model = Model(**asdict(self.model_args))
        self.model.set_model_path(self.test_args.weight)
        self.model.init_model()  # building model and loading weight
        self.model.move_to_device()

    def init_id_map(self):
        experiment_path = os.path.join(self.project_args.work_dir, 'runs', self.test_args.experiment_time)
        seg_class_id_map_path = os.path.join(experiment_path, 'seg_id_to_label.txt')
        cls_class_id_map_path = os.path.join(experiment_path, 'cls_id_to_label.txt')

        if self.task in [Task.MultiTask, Task.CLS]:
            if os.path.exists(cls_class_id_map_path):
                self.cls_id2class = self.load_class_id_map(cls_class_id_map_path)
                logger.success('Loading cls_id_to_label.txt')
            else:
                logger.error(f'Don`t found the cls_id_to_label .txt')
                exit()

        if self.task in [Task.MultiTask, Task.SEG]:
            if os.path.exists(seg_class_id_map_path):
                self.seg_id2class = self.load_class_id_map(seg_class_id_map_path)
                logger.success('Loading seg_id_to_label.txt')
            else:
                logger.error(f'Don`t found the seg_class_id_map.txt')
                exit()

    @staticmethod
    def copy_image(image: str, root: str, save_path: str):
        image_save_path = image.replace(root, save_path)
        image_name = os.path.basename(image)
        image_save_dir = image_save_path.split(image_name)[0]
        os.makedirs(image_save_dir, exist_ok=True)  # check output dir
        shutil.copy(image, image_save_dir)

    def preprocess(self, image: str, transforms: Callable) -> torch.Tensor:
        img = Image.open(image)
        img_w, img_h = img.size
        if img_w != self.wh[0] or img_h != self.wh[1]:
            img = img.resize(self.wh, Image.BILINEAR)
        img = img.convert('RGB')
        img = transforms(img)

        return img

    def infer_core(self, image: str, transforms: Callable) -> Optional[torch.Tensor]:
        if self.model.training:
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
        logger.info(f'Test Dir: {self.test_args.test_dir}')
        images = get_images(self.test_args.test_dir)

        for image in tqdm(images):
            model_output = self.infer_core(image, self.val_transforms)

            postprocessing_func(
                image=image,
                model_output=model_output,
                **asdict(self.test_args)
            )

    def classification_postprocessing(
            self,
            image: str,
            model_output: Union[torch.Tensor, List[torch.Tensor]],
            cls_threshold: Optional[List[int]] = None,
            **kwargs
    ) -> None:

        if isinstance(model_output, list):
            model_output = model_output[0]
        model_output = model_output.squeeze(0).softmax(0).cpu().numpy()  # [score1,score2,...]

        round_float_4 = lambda num: round(float(num), 4)

        # Std Post-Processing
        # max_score_id = model_output.argmax(0)
        # max_score = round(float(model_output[max_score_id]), 4)
        # max_score_label = self.cls_id2class[max_score_id]
        #
        # save_path = os.path.join(self.classification_output_path, max_score_label)
        # self.copy_image(image, root=self.test_args.test_dir, save_path=save_path)
        # with open(os.path.join(self.result_output_path, 'classification_output.txt'), 'a') as f:
        #     f.write(
        #         f'[Image]:{image}\t'
        #         f'[Max Score Label]:{max_score_label}\t'
        #         f'[Max Score]:{max_score}\n'
        #     )
        # =====================================================================================

        # Project post-processing
        if cls_threshold is None:
            cls_threshold = [0.5] * self.model_args.num_classes

        if len(cls_threshold) < len(model_output):
            logger.error('len(cls_threshold) < len(model_output)')
            exit()

        good_idx = self.test_args.good_idx
        # good_score = np.round(float(model_output[good_idx]), 4)
        good_score = round_float_4(model_output[good_idx])

        is_good_flag = False
        if good_score >= cls_threshold[good_idx]:

            good_label = self.cls_id2class.get(good_idx)
            if good_label is None:
                logger.error('Don`t found the label.')
                exit()

            save_path = os.path.join(self.classification_output_path, good_label)
            self.copy_image(image, root=self.test_args.test_dir, save_path=save_path)
            is_good_flag = True

        ng_max_score_idx = model_output[1:].argmax(0) + 1
        # ng_max_score = np.round(float(model_output[ng_max_score_idx]), 4)
        ng_max_score = round_float_4(model_output[ng_max_score_idx])
        ng_max_score_label = self.cls_id2class.get(ng_max_score_idx)

        if ng_max_score_label is None:
            logger.error('Don`t found the label.')
            exit()

        if is_good_flag is False:

            if ng_max_score >= cls_threshold[ng_max_score_idx]:
                save_path = os.path.join(self.classification_output_path, ng_max_score_label)
            else:
                save_path = os.path.join(self.classification_output_path, 'return-good')

            self.copy_image(image, root=self.test_args.test_dir, save_path=save_path)

        with open(os.path.join(self.result_output_path, 'classification_output.txt'), 'a') as f:
            f.write(
                f'[Good_Th]:{cls_threshold[good_idx]}\t'
                f'[Image]:{image}\t'
                f'[GS]:{good_score}\t'
                f'[Max_NGS]:{ng_max_score}\t'
                f'[NG_Label]:{ng_max_score_label}\n'
            )

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

        if not seg_threshold:
            seg_threshold = [0] * self.model_args.mask_classes
        if len(seg_threshold) < self.model.mask_classes:
            logger.error('len(seg_threshold) < self.model.mask_classes')
            exit()

        is_good = True
        for curr_label_idx in range(0, self.model.mask_classes):

            index = np.where(vis_predict == curr_label_idx)

            if (len(index[0]) * len(index[1])) == (img.shape[0] * img.shape[1]):
                continue

            if sum_method is True:
                label_th = seg_threshold[curr_label_idx]
                num_of_pixel = len(index[0])
                if num_of_pixel >= label_th:

                    if curr_label_idx > 0:
                        is_good = False

                    vis_predict_seg[index[0], index[1], :] = color_list[curr_label_idx]
                    label = self.seg_id2class.get(curr_label_idx)

                    if label is None:
                        logger.error('label is None')
                        exit()
                    cls_and_area.update({
                        label: len(index[0])
                    })

            else:
                mask_index = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                mask_index[index[0], index[1]] = 255
                cnts, _ = cv2.findContours(mask_index, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                draw_cns = []

                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area >= seg_threshold[curr_label_idx]:
                        draw_cns.append(cnt)

                cv2.drawContours(vis_predict_seg, draw_cns, -1, tuple(color_list[curr_label_idx]), -1)

        if not is_good:
            with open(os.path.join(self.result_output_path, 'seg_area.txt'), 'a') as f:
                f.write(
                    f'[image]:{os.path.basename(image)}]\t'
                    f'[data]:{cls_and_area}\n'
                )
            # vis_predict_color = vis_predict_seg.astype(np.uint8)
            vis_predict_seg = vis_predict_seg.astype(np.uint8)
            vis_addweight = cv2.addWeighted(img, 0.3, vis_predict_seg, 0.5, 0)

            image_basename = os.path.basename(image)
            file_name, ext = os.path.splitext(image_basename)

            cv2.imwrite(os.path.join(self.segmentation_output_path, image_basename), img)
            cv2.imwrite(os.path.join(self.segmentation_output_path, file_name + '_seg.png'), vis_addweight)

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
            self.segmentation_postprocessing(image, predict_seg, sum_method, seg_threshold)

        self.classification_postprocessing(image, predict_cls, cls_threshold)

    @staticmethod
    def load_class_id_map(class_id_map_path: str) -> dict:

        if os.path.exists(class_id_map_path) is False:
            logger.warning(f'Don`t found file:{class_id_map_path}.')
            return {}

        class_id_map = {}
        with open(class_id_map_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                # k=0,1,2,...
                # v=classes1,classes2,...
                k, v = line.split(":", 1)
                # class_id_map={0:"classes1",1:"classes2",...}
                class_id_map[int(k)] = v
        return class_id_map
