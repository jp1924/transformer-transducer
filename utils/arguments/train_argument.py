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
    noise_step: int = field(
        default=-1,
        metadata={"help": "gausian_noise를 추가하는 스탭을 선택합니다."},
    )
    noise_mean: float = field(
        default=0.0,
        metadata={"help": "gaussian noise의 평균값을 설정합니다."},
    )
    noise_std: float = field(
        default=0.01,
        metadata={"help": "gaussian noise의 분산값을 설정합니다."},
    )
