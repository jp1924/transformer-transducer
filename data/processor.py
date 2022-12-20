import warnings
from transformers import ProcessorMixin
from .extractor import TransducerFeatureExtractor
from .tokenizer import TransducerTokenizer
from contextlib import contextmanager


class TransducerProcessor(ProcessorMixin):
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.

    [`Wav2Vec2Processor`] offers all the functionalities of [`Wav2Vec2FeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~Wav2Vec2Processor.__call__`] and [`~Wav2Vec2Processor.decode`] for more information.

    Args:
        feature_extractor (`Wav2Vec2FeatureExtractor`):
            An instance of [`Wav2Vec2FeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'Wav2Vec2CTCTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            feature_extractor = TransducerFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = TransducerTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.__call__`] and returns its output. If used in the context
        [`~Wav2Vec2Processor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's
        [`~PreTrainedTokenizer.__call__`]. Please refer to the docstring of the above two methods for more information.
        """
        return self.current_processor(*args, **kwargs)

    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.pad`] and returns its output. If used in the context
        [`~Wav2Vec2Processor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's
        [`~PreTrainedTokenizer.pad`]. Please refer to the docstring of the above two methods for more information.
        """
        return self.current_processor.pad(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
