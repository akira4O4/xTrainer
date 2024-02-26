from .register import Register

LR_PARAMS = Register('lr_params')
LOSS = Register('loss')

__all__ = [
    'Register',
    'LR_PARAMS',
    'LOSS',
]
