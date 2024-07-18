import importlib

from callbacks import EmptyCacheCallback, GaussianNoiseCallback

from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.utils import ExplicitEnum

from .args import TransformerTransducerArguments
from .optimization import get_tri_stage_schedule_with_warmup_lr_lambda, set_scheduler
from .preprocessor import (
    centi_meter_regex,
    default_sentence_norm,
    double_space_regex,
    kilo_meter_regex,
    librosa_silence_filter,
    meter_regex,
    noise_filter_regex,
    noise_mark_delete,
    normal_dual_bracket_regex,
    normal_dual_transcript_extractor,
    percentage_regex,
    space_norm,
    special_char_norm,
    special_char_regex,
    unit_system_normalize,
    unnormal_dual_bracket_regex,
    unnormal_dual_transcript_extractor,
)


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
    COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    WARMUP_STABLE_DECAY = "warmup_stable_decay"
    TRI_STAGE = "tri_stage"  # 추가됨


NEW_TYPE_TO_SCHEDULER_FUNCTION = TYPE_TO_SCHEDULER_FUNCTION
NEW_TYPE_TO_SCHEDULER_FUNCTION.update({SchedulerType.TRI_STAGE: get_tri_stage_schedule_with_warmup_lr_lambda})

# NOTE: 빼놓고 추가하지 않은 곳이 있으면 정상동작 안할 가능성이 존재함. 확인 필요
module = importlib.import_module("transformers.optimization")
setattr(module, "TYPE_TO_SCHEDULER_FUNCTION", NEW_TYPE_TO_SCHEDULER_FUNCTION)

module = importlib.import_module("transformers.trainer_utils")
setattr(module, "SchedulerType", SchedulerType)

module = importlib.import_module("transformers.training_args")
setattr(module, "SchedulerType", SchedulerType)

module = importlib.import_module("transformers.optimization")
setattr(module, "SchedulerType", SchedulerType)
