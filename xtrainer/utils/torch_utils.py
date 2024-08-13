import random
from typing import List
import numpy as np
import torch
import torch.backends.cudnn


def init_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def init_backends_cudnn(deterministic: bool = False) -> None:
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def convert_optimizer_state_dict_to_fp16(state_dict) -> dict:
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict


def loss_sum(losses: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    assert len(losses) == len(weights), 'len(loss)!=len(weights)'

    ret = torch.tensor(0.0, dtype=losses[0].dtype, device=losses[0].device)

    for loss, weight in zip(losses, weights):
        ret += loss * weight

    return ret  # noqa
