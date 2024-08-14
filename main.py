import os
import argparse

from loguru import logger
from mlflow import set_experiment

from xtrainer import CONFIG, OS, VERSION, CUDA
from xtrainer.trainer import Trainer
from xtrainer.predict import Predictor
from xtrainer.utils.common import check_dir, get_time


def init_workspace() -> None:
    """
    project
        - experiment1
            - cls_labels.txt
            - seg_labels.txt
            - weights(dir)
    """

    check_dir(CONFIG('project'))

    experiment_path = os.path.join(CONFIG('project'), CONFIG('experiment'))
    if os.path.exists(experiment_path):
        experiment_path = os.path.join(CONFIG('project'), CONFIG('experiment') + '.' + get_time())
    check_dir(experiment_path)

    weight_path = os.path.join(experiment_path, 'weights')
    check_dir(weight_path)

    CONFIG.update({"experiment_path": experiment_path})
    CONFIG.update({"weight_path": weight_path})


def init_mlflow() -> None:
    if CONFIG('mlflow_experiment_name') == '':
        logger.info(f'MLFlow Experiment Name: Default.')
    else:
        exp_name = CONFIG('mlflow_experiment_name')
        set_experiment(exp_name)
        logger.info(f'MLFlow Experiment Name :{exp_name}.')


def check_args() -> None:
    if CONFIG('mode').lower() not in ['train', 'predict']:
        raise KeyError("Model must be in ['train', 'predict']")

    if CONFIG('task').lower() not in ['classification', 'segmentation', 'multitask']:
        raise KeyError("Model must be in ['classification', 'segmentation', 'multitask']")

    if not CUDA and CONFIG('device') >= 0:
        logger.error('CUDA is not available')
        exit(-1)

    if CONFIG('save_period') < 1:
        raise ValueError('save period must be >= 1')

    if CONFIG('mode') == 'predict':
        if os.path.exists(CONFIG('source')) is False:
            raise FileNotFoundError('Don`t test source')

        if os.path.exists(CONFIG('test_weight')) is False:
            raise FileNotFoundError('Don`t found weight')

        if CONFIG('task') in ['classification', 'multitask']:
            if os.path.exists(CONFIG('cls_label')) is False:
                raise FileNotFoundError('Don`t found label file')

            if CONFIG('classification.classes') < len(CONFIG('cls_thr')):
                raise EOFError('nc!=len(thr)')

        if CONFIG('task') in ['segmentation', 'multitask']:
            if os.path.exists(CONFIG('seg_label')) is False:
                raise FileNotFoundError('Don`t found label file')

            if CONFIG('segmentation.classes') < len(CONFIG('seg_thr')):
                raise EOFError('nc!=len(thr)')


if __name__ == '__main__':
    logger.info(f'OS: {OS}')
    logger.info(f'Version: {VERSION}')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=r'configs\default.yaml',
        help='CONFIG path'
    )
    args = parser.parse_args()

    CONFIG.set_path(args.config)
    CONFIG.load()

    check_args()

    if CONFIG('mode').lower() == 'train':
        init_workspace()
        init_mlflow()
        trainer = Trainer()
        trainer.run()

    elif CONFIG('mode').lower() == 'predict':
        predictor = Predictor()
        predictor.run()
