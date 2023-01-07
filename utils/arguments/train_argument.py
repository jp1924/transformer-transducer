from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments


@dataclass
class TransducerTrainArgument(Seq2SeqTrainingArguments):
    vocab_path: str = field(
        default=None,
        metadata={"help": "이건 나중에 삭제될 거임"},
    )
    do_fist_predict: bool = field(
        default=False,
        metadata={"help": "만약 model을 checkpoint로 부터 불러오거나 재개 했을 때 모델의 성능을 측정하기 위해 사용하는 값"},
    )
