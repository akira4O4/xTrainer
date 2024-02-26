from imgaug import augmenters as iaa

__all__ = ['iaa_augment_list']


def iaa_augment_list():
    # 进行模型初始化
    # sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    iaa_aug_seq = iaa.Sequential([
        # iaa.SomeOf((0, 1), [
        # iaa.OneOf([
        #     sometimes(iaa.Affine(scale=(0.99, 1.01))),
        #     sometimes(iaa.Affine(translate_percent=(-0.01, 0.01))),
        # ]),
        #
        # iaa.SomeOf((1, 2), [
        #     iaa.Affine(shear=(-1, 1)),
        #     iaa.Affine(rotate=(-1, 1))
        # ]),

        # iaa.SomeOf((0, 1), [
        #     iaa.Flipud(0.5),
        #     iaa.Fliplr(0.5)
        # ]),
        # ]),

        iaa.SomeOf((0, 1), [iaa.Flipud(0.7), iaa.Fliplr(0.7)]),
        iaa.MultiplyHue((0.9, 1.1)),
        iaa.MultiplySaturation((0.9, 1.1)),
        iaa.SomeOf((0, 1), [iaa.Add((-10, 20)), iaa.Multiply((0.8, 1.2))]),
        # iaa.SomeOf((0, 1), [iaa.WithBrightnessChannels(iaa.Add((-10, 10))), iaa.GammaContrast((0.9, 1.1))]),
    ])

    return iaa_aug_seq
