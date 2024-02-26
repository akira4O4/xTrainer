import os

from torchvision import models
import torch
import network as network


def create_model(args):
    model = network.__dict__[args.model_names](pretrained=True, multi_task=args.multi_task, num_classes=args.classes,
                                               input_w=args.input_w, input_h=args.input_h)
    # resume operate
    if args.resume:
        if os.path.isfile(args.resume):
            try:
                print("==> loading checkpoint '{}'".format(args.resume))
                loc = 'cuda:{}'.format(0)
                checkpoint = torch.load(args.resume, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                static_dict = checkpoint['state_dict']
                new_static_dict = {}
                # for k,val in static_dict.items():
                #     这里的名称是shufflenet里的变量名称
                #     if "mask_branch" not in k:
                #         print("删除mask_branch分支权重")
                #         new_static_dict[k]=val
                if new_static_dict:
                    model.load_state_dict(new_static_dict, strict=False)
                else:
                    # static_dict.pop('landmark_branch.coord_conv1.0.weight')
                    # static_dict.pop('landmark_branch.coord_conv2.0.weight')
                    # static_dict.pop('landmark_branch.coord_conv3.0.weight')
                    # static_dict.pop('landmark_branch.coord_fc.weight')
                    # static_dict.pop('landmark_branch.coord_fc.bias')
                    model.load_state_dict(static_dict, strict=True)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                args.best_acc1 = best_acc1
            except Exception as e:
                print('\n 定义的网络跟resume的权重不匹配，可能是multi_task参数没有设置正确。')
                raise
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    return model


def create_model_origin(pretrained):
    model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    return model
