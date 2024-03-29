import math
from typing import Tuple, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# [NOTE]: copied from https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/tri_stage_lr_scheduler.py
def get_tri_stage_scheduler_with_warmup(
    optimizer: Optimizer,
    num_training_steps: int,
    final_lr: float,
    num_warmup_steps: Union[int, float],
    num_hold_steps: Union[int, float],
    num_decay_steps: Union[int, float],
    last_epoch: int = -1,
) -> LambdaLR:
    """doc"""
    warmup_steps = num_warmup_steps if num_warmup_steps >= 1 else (num_warmup_steps * num_training_steps)
    hold_steps = num_hold_steps if num_hold_steps >= 1 else (num_hold_steps * num_training_steps)
    decay_steps = num_decay_steps if num_decay_steps >= 1 else (num_decay_steps * num_training_steps)

    if not (warmup_steps + hold_steps + decay_steps) <= num_training_steps:
        raise ValueError(
            f"""must don't exceed max_steps or epoch. but lr steps exceed max_step, please setting again
            num_training_steps: {num_training_steps}
            warmup_steps: {warmup_steps}
            hold_steps: {hold_steps}
            decay_steps: {decay_steps}
            """
        )

    default_lr = optimizer.defaults["lr"]

    warm_up_compensator = (default_lr - default_lr) + (default_lr / default_lr)
    warmup_factor = warm_up_compensator / warmup_steps
    decay_factor = -math.log(final_lr) / decay_steps

    def _decide_stage(step: int) -> Tuple[int, int]:
        # [NOTE]: warmup(rampup) stage
        if step < warmup_steps:
            return "warm", step
        offset = warmup_steps

        # [NOTE]: hold stage
        if step < (offset + hold_steps):
            return "hold", (step - offset)
        offset += hold_steps

        # [NOTE]: decay stage
        if step <= (offset + decay_steps):
            return "decay", (step - offset)

        # [NOTE]: over stage
        return "over", (step - offset)

    def lr_lambda(current_step: int) -> float:
        stage, step = _decide_stage(current_step)
        if "warm" == stage:
            learning_rate = warmup_factor * step
        elif "hold" == stage:
            compensator = default_lr
            learning_rate = math.ceil(default_lr**compensator)
        elif "decay" == stage:
            compensator = default_lr
            learning_rate = math.ceil(default_lr**compensator) * math.exp(-decay_factor * step)
        elif "over" == stage:
            learning_rate = final_lr

        return learning_rate

    return LambdaLR(optimizer, lr_lambda, last_epoch)
