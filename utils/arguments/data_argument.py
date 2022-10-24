from dataclasses import dataclass, field


@dataclass
class DataArguments:
    data_name: str = field(
        default=None,
        metadata={"help": "데이터으 허브 이름 혹은 데이터의 로컬 경로를 포함합니다."},
    )
