from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    num_labels: int = field(
        default=1,
        metadata={"help": "classification시 분류를 할 label의 개수를 설정합니다."},
    )
