# from torch.nn import CrossEntropyLoss

SUPPORT_LOSS = {"CrossEntropyLoss",
                "FocalLoss",
                "IouSsimLoss",
                "DiceLoss",
                "PeriodLoss"
                }


def build_loss(loss_name, **kwargs):
    assert loss_name in SUPPORT_LOSS, f"all support loss is {SUPPORT_LOSS}"
    criterion = eval(loss_name)(**kwargs)
    return criterion
