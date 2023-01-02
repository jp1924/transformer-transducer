from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "사전학습된 모델의 local 경로 혹은 hub의 이름을 설정합니다."},
    )
    cache_dir: str = field(
        default="./cache",
        metadata={"help": "모델 혹은 데이터를 저장할 캐쉬의 경로를 지정합니다."},
    )
    ramp_up_step_ratio: float = field(
        default=0.0,  # 0.02
        metadata={"help": "전체 스텝에서 warm_up or ramped up step의 비율을 설정합니다. "},
    )
    hold_step_ratio: float = field(
        default=0.0,  # 0.15
        metadata={"help": "tri-stage의 hold step의 비율을 설정합니다."},
    )
    decay_step_ratio: float = field(
        default=0.0,
        metadata={"help": "tri-stage의 decay step의 비율을 설정합니다."},
    )
    init_learning_rate: float = field(
        default=0.0,
        metadata={"help": "tri-stage의 시작 lr값을 설정합니다."},
    )
    final_learning_rate: float = field(
        default=0.0,
        metadata={"help": "tri-stage의 끛 lr값을 설정합니다."},
    )
