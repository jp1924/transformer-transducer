from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments


@dataclass
class TransducerTrainArgument(Seq2SeqTrainingArguments):
    mel_max_length: int = field(
        default=400,
        metadata={"help": "mel의 최대 길이르 지정합니다. 이 Transducer는 무조건 일정한 크기의 데이터만 들어가야 합니다."},
    )
    vocab_path: str = field(
        default=None,
        metadata={"help": "이건 나중에 삭제될 거임"},
    )
