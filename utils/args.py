from dataclasses import dataclass, field
from typing import List, Union

from transformers import TrainingArguments

from .optimization import SchedulerType


@dataclass
class TransformerTransducerArguments(TrainingArguments):
    # 이걸 해야 스케줄러가 정상적으로 적용됨.
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    # data
    dataset_repo_ls: List[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batched: bool = field(
        default=True,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'audio'"},
    )
    sentence_column_name: str = field(
        default="sentence",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'sentence'"},
    )
    min_duration_in_seconds: float = field(
        default=3584.0,
        metadata={"help": "Filter out audio files that are longer than `min_duration_in_seconds` seconds"},
    )
    max_duration_in_seconds: float = field(
        default=448512.0,
        metadata={"help": "Filter out audio files that are shorter than `max_duration_in_seconds` seconds"},
    )
    train_dataset_prefix: List[str] = field(
        default="train",
        metadata={"help": ""},
    )
    valid_dataset_prefix: List[str] = field(
        default="validation",
        metadata={"help": ""},
    )
    test_dataset_prefix: List[str] = field(
        default="eval_other",
        metadata={"help": ""},
    )
    cache_file_name: str = field(
        default=None,
        metadata={"help": "Path to cached file name"},
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    sampling_rate: str = field(
        default=16000,
        metadata={"help": ""},
    )

    # model
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )

    pad_to_multiple_of: int = field(
        default=None,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        },
    )

    attn_implementation: str = field(
        default="eager",
        metadata={"help": ""},
    )

    wandb_code_log_dir: str = field(
        default="",
        metadata={"help": ""},
    )
