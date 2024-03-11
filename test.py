import argparse

from core.infer import Infer
from utils.util import load_config, timer


@timer
def test(args):
    infer = Infer(**args)
    infer.init()
    infer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project')
    parser.add_argument('--config',
                        # default=r"configs/gz_sapa.yml",
                        # default=r"configs/guoyao_kela.yml",
                        # default=r"configs/gz_sapa_duanmian.yml",
                        # default=r"configs/gz_sapa_front.yml",
                        # default=r"configs/gz_sapa_xiansao.yml",
                        # default=r"configs/hrsf_zuo.yml",
                        # default=r"configs/danyang_E_mt.yml",
                        # default=r"configs/danyang_E_mt_exp2.yml",
                        # default=r"configs/danyang_F_seg_exp2.yml",
                        # default=r"configs/danyang_G.yml",
                        default=r"configs/danyang_C2.yml",
                        # default=r"configs/danyang_x_seg_exp1.yml",
                        help='your project config json.')
    args = parser.parse_args()
    config = load_config(args.config)
    test(config)
