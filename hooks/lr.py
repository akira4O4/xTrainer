from register import LR_PARAMS


@LR_PARAMS.registered
def LambdaLR(**kwargs) -> dict:
    args = {
        'lr_lambda': lambda epoch: 1 / (epoch / 4 + 1),
        'last_epoch': -1,
        'verbose': False
    }
    args.update(kwargs)
    return args
