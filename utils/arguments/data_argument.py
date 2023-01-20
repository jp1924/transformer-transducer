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
    mel_stack: int = field(
        default=4,
        metadata={"help": "mel을 windowing 시킨 다음 그 값을 적재할 크기를 지정합니다."},
    )
    window_stride: int = field(
        default=3,
        metadata={"help": "windowing을 할 값을 지정합니다. mel_stack 값보다 적은 값이여 합니다. 그렇지 않으면 음성간의 연관성이 사라질 수 있습니다!"},
    )
    num_fourier: int = field(
        default=512,
        metadata={"help": "Fast-Fourier-Transform을 적용시킬 음성의 양을 지정한다."},
    )
    mel_shape: int = field(
        default=128,
        metadata={"help": "생성될 mel의 크기를 설정한다."},
    )
    hop_length: int = field(
        default=128,
        metadata={"help": "windowing시 옆으로 옮겨갈 크기를 결정한다."},
    )
    sampling_rate: int = field(
        default=16000,
        metadata={"help": ""},
    )
