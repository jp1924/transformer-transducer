from transformers import SequenceFeatureExtractor


class TransducerFeatureExtractor(SequenceFeatureExtractor):
    def __init__(self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs) -> None:
        super().__init__(feature_size, sampling_rate, padding_value, **kwargs)
