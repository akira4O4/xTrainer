import argparse
from core.trainer import Trainer
from utils.util import load_yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project')
    parser.add_argument('--config',
                        # default=r"configs/gz_sapa.yml",
                        # default=r"configs/suzou_sapa_zifu.yml",
                        # default=r"configs/default/cls.yml",
                        # default=r"configs/anbuping.yml",
                        # default=r"configs/guoyao_aqi.yml",
                        # default=r"configs/guoyao_kela.yml",
                        # default=r"configs/anbuping.yml",

                        # default=r"configs/gz_sapa_xiansao.yml",
                        # default=r"configs/gz_sapa_duanmian.yml",
                        # default=r"configs/gz_sapa_front.yml",

                        # default=r"configs/hrsf_zuo.yml",
                        # default=r"configs/danyang_C2.yml",
                        # default=r"configs/danyang_E_mt.yml",
                        # default=r"configs/danyang_G.yml",
                        # default=r"configs/danyang_F_seg_exp2.yml",
                        # default=r"configs/danyang_E_mt_exp2.yml",
                        default=r"configs/cat_dog.yml",
                        help='your project config json.')
    args = parser.parse_args()
    input_config = load_yaml(args.config)

    trainer = Trainer(**input_config)
    trainer.run()
