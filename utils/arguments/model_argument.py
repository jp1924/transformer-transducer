from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    cache_dir: str = field(
        default="./cache",
        metadata={"help": "모델 혹은 데이터를 저장할 캐쉬의 경로를 지정합니다."},
    )
