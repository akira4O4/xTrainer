import time

import torch
from torchvision import models

from network import create_network
from config.parser_config import config_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
时间测试总结：
input:[1,3,544,544]
model_multi: 9ms
model_origin: 7ms

input:[20,3,544,544]
model_multi: 99ms
model_multi: 30ms
"""


def test_time(model_multi, model_origin):
    model_multi = model_multi.cuda()
    model_origin = model_origin.cuda()
    model_multi.eval()
    model_origin.eval()
    input = torch.rand((1, 3, 320, 320)).cuda()
    with torch.no_grad():
        for i in range(200):
            start_time = time.time()
            output = model_origin(input)
            print("model_origin 耗时：{}".format(time.time() - start_time))

        for i in range(200):
            start_time = time.time()
            output = model_multi(input)
            print("model_multi 耗时:{}".format(time.time() - start_time))


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    args.resume = "temp/model_best_landmark_.pth.tar"
    # args.resume="temp/model_best_cls_.pth.tar"
    root_path = "temp/test"
    output_save_path = "temp/output"
    model = create_network.create_model(args)
    model_origin = models.shufflenet_v2_x1_0(pretrained=True)
    test_time(model, model_origin)
