from transformers import ProcessorMixin


class TransducerProcessor(ProcessorMixin):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
