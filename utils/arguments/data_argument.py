from dataclasses import dataclass, field


@dataclass
class DataArguments:
    data_name: str = field(
        default="librispeech_asr",
        metadata={"help": "데이터의 이름 혹은 데이터의 로컬 경로를 포함합니다."},
    )
    num_proc: int = field(
        default=1,
        metadata={"help": "전처리에서 사용할 CPU 프로세서의 개수를 설정합니다."},
    )
