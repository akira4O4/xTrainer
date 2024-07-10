import argparse
from loguru import logger
from trainerx import CONFIG, OS, VERSION
from trainerx.trainer_v3 import Trainer

if __name__ == '__main__':
    logger.info(f'OS: {OS}')
    logger.info(f'Version: {VERSION}')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cfg',
        '--config',
        type=str,
        default=r'configs\default.yaml',
        help='CONFIG path'
    )
    args = parser.parse_args()

    CONFIG.set_path(args.config)
    CONFIG.load()

    trainer = Trainer()
