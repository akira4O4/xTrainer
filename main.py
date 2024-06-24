import argparse
from loguru import logger
from src import CONFIG, OS
from src.core.trainer_v3 import Trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cfg',
        '--config',
        type=str,
        # default=r'amat\configs\default\classification.yaml',
        default=r'configs\new.yaml',
        # default=r'amat\configs\default\detection.yaml',
        help='CONFIG path'
    )
    logger.info(f'OS: {OS}')
    args = parser.parse_args()
    CONFIG.load(args.config)
    trainer = Trainer()
