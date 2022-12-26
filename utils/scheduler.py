import math

from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from typing import List, Union, Callable, Dict, Tuple


class NoCalcStepLambdaLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Union[Callable[[int], float], List[Callable[[int], float]]],
        last_epoch: int = ...,
    ) -> None:
        super().__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self) -> float:
        super().get_lr()
        # [NOTE]: deleted ``base_lr * lmbda(self.last_epoch)``
        #         The ``base_lr * lmbda(self.last_epoch)`` part was deleted to match the value of fairseq's tri-stage scheduler.
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


# [NOTE]: copied from https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/tri_stage_lr_scheduler.py
class TriStageLRScheduler:
    """TriStageLRScheduler

    낮은 러닝레이트(init_learning_rate로 계산)에서 리니어처럼 증가하여, hold 까지 증가하며,
    final_learning_rate까지 log 감쇠하는 러닝레이트 스케쥴러이다.
    step으로 각각의 stage 수를 조절할 수 있으며, 좀만 응용하면 cosine과 같이 만들 수도 있다.
    """

    def __init__(self, args: Dict, max_steps: int, learning_rate: float = 1e-5) -> None:
        if learning_rate > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with tri-stage lr." " Consider --lr-scheduler=fixed instead."
            )

        assert (args.warmup_step_ratio + args.hold_step_ratio + args.decay_step_ratio) == 1, "ratio의 합이 1이 되도록 설정해 주세요!"

        self.peak_learning_rate = learning_rate
        self.init_learning_rate = args.init_learning_rate * learning_rate
        self.final_learning_rate = args.final_learning_rate * learning_rate

        self.warmup_steps = int(max_steps * args.warmup_step_ratio)
        self.hold_steps = int(max_steps * args.hold_step_ratio)
        self.decay_steps = int(max_steps * args.decay_step_ratio)

        self.warmup_rate = (
            (self.peak_learning_rate - self.init_learning_rate) / self.warmup_steps if self.warmup_steps != 0 else 0
        )
        self.decay_factor = -math.log(args.final_learning_rate) / self.decay_steps

    def _decide_stage(self, step: int) -> Tuple[int, int]:
        if step < self.warmup_steps:  # warmup state (or ramped up)
            return 0, step
        offset = self.warmup_steps

        if step < offset + self.hold_steps:  # hold stage
            return 1, step - offset
        offset += self.hold_steps

        if step <= offset + self.decay_steps:  # decay stage
            return 2, step - offset
        offset += self.decay_steps

        return 3, step - offset

    def get_tri_stage_scheduler(self, optimizer: object, last_epoch: int = -1) -> NoCalcStepLambdaLR:
        def lr_lambda(current_step: int):
            stage, step = self._decide_stage(current_step)
            if stage == 0:
                learning_rate = self.init_learning_rate + self.warmup_rate * step
            elif stage == 1:
                learning_rate = self.peak_learning_rate
            elif stage == 2:
                learning_rate = self.peak_learning_rate * math.exp(-self.decay_factor * step)
            elif stage == 3:
                learning_rate = self.final_learning_rate

            return learning_rate

        return NoCalcStepLambdaLR(optimizer, lr_lambda, last_epoch)
