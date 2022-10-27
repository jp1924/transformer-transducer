from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class TrainArguments:
    cache: str = field(
        default=None,
        metadata={"help": "허브로 부터 데이터들을 불러올 때 저장할 공간의 경로를 지정합니다."},
    )
    output_dir: str = field(
        default=None,
        metadata={"help": ""},
    )
    train_batch_size: int = field(
        default=None,
        metadata={"help": ""},
    )
    valid_batch_size: int = field(
        default=None,
        metadata={"help": ""},
    )
    train_accumulate: int = field(
        default=1,
        metadata={"help": ""},
    )
    valid_accumulate: int = field(
        default=1,
        metadata={"help": ""},
    )
    local_rank: int = field(default=-1)
    train_epochs: int = field(
        default=None,
        metadata={"help": ""},
    )
    train_max_step: int = field(
        default=None,
        metadata={"help": ""},
    )
    valid_step: int = field(
        default=5000,
        metadata={"help": ""},
    )
    warmup_step: int = field(
        default=None,
        metadata={"help": ""},
    )
    weight_decay: float = field(
        default=None,
        metadata={"help": ""},
    )
    learning_rate: float = field(
        default=None,
        metadata={"help": ""},
    )
