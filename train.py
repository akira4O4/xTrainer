import argparse
from src.trainer_v3 import Trainer

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

                        default=r"configs/default/classification.yml",
                        # default=r"configs/default/segmentation.yml",
                        # default=r"configs/default/multi_task.yml",
                        help='your project config json.')
    args = parser.parse_args()
    trainer = Trainer(args.config)
    trainer.run()
