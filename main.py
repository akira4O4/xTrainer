import argparse
from loguru import logger
from src import CONFIG, OS
from src.trainer_v3 import Trainer

if __name__ == '__main__':
    logger.info(f'OS: {OS}')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cfg',
        '--config',
        type=str,
        default=r'configs\new.yaml',
        help='CONFIG path'
    )
    args = parser.parse_args()

    CONFIG.set_path(args.config)
    CONFIG.load()

    trainer = Trainer()
