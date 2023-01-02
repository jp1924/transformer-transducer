from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments


@dataclass
class TransducerTrainArgument(Seq2SeqTrainingArguments):
    mel_max_length: int = field(
        default=400,
        metadata={"help": "mel의 최대 길이르 지정합니다. 이 Transducer는 무조건 일정한 크기의 데이터만 들어가야 합니다."},
    )
    mel_stack: int = field(
        default=4,
        metadata={"help": "mel을 windowing 시킨 다음 그 값을 적재할 크기를 지정합니다."},
    )
    window_stride: int = field(
        default=3,
        metadata={"help": "windowing을 할 값을 지정합니다. mel_stack 값보다 적은 값이여 합니다. 그렇지 않으면 음성간의 연관성이 사라질 수 있습니다!"},
    )
    vocab_path: str = field(
        default=None,
        metadata={"help": "이건 나중에 삭제될 거임"},
    )
