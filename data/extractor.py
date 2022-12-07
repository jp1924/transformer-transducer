import math
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch

from transformers.file_utils import is_torchaudio_available


if is_torchaudio_available():
    from torchaudio.transforms import MelSpectrogram

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)
DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)


class TransducerFeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int,
        sampling_rate: int,
        padding_value: float,
        stack: int = None,
        stride: int = None,
        **kwargs,
    ) -> None:
        super().__init__(feature_size, sampling_rate, padding_value, **kwargs)

        # [NOTE]: 굳이 stride, stack를 init에 추가한 이유,
        #         init에 있는 값들은 한번 지정하면 이후에 변경하는 일이 없다. stride나 stack도 한번 설정하면 변경할 일이 없기 때문에 init에 놔뒀다.
        if stride and stack:
            assert stack > stride, "stride must be small stack_size, please set correct value"
            self.stack = stack
            self.stride = stride

    @staticmethod
    def piecewise_linear_log(features: torch.FloatTensor):
        features[features > math.e] = torch.log(features[features > math.e])
        features[features <= math.e] = features[features <= math.e] / math.e
        return features

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract the mel spectrogram features and normalize them
        """
        waveform = torch.tensor(waveform)
        features = self.mel_transform(waveform)
        features = features.transpose(1, 0)
        features = self.piecewise_linear_log(features * GAIN)
        features = (features - self.global_mean) * self.global_invstddev
        features = features.numpy()

        return features

    # def mask_chunking(self, attention_mask)

    def compress_features(self, features: Union[List[np.ndarray], List["mel"], np.ndarray]) -> List[Any]:
        if not isinstance(features, list):
            features = [features]

        return_list = list()
        for mel in features:
            compressed_feature = self._compress_features(mel)
            return_list.append(compressed_feature)

        return return_list

    def _compress_features(self, mel_feature: np.ndarray) -> np.ndarray:
        """mel을 windowing을 적용시켜 값을 압축시킨다."""
        # [NOTE]: 여기서 각각의 멜을 windowing + padding한 뒤 나머지 compress_mel을 padding하는 방식으로 진행해야 할 듯 하다.
        #         compress_features에서 padding을 처리하지 않고 하기에는 self._pad를 overriding해서 새로 만들어야 함.
        time_steps, _ = mel_feature.shape
        expected_len = math.ceil(time_steps / self.stride)

        mel_store = list()
        # [TODO]: left, right context하고 data전처리 간의 어떤 연관이 있는지 모르겠다.
        for step in range(self.stack):
            indices = [step + idx for idx in range(0, (time_steps - step), self.stride)]
            features = mel_feature[indices]
            pad_width = ((0, expected_len - features.shape[0]), (0, 0))
            features = np.pad(features, pad_width)

            mel_store.append(features)

        padded_feature = np.concatenate(mel_store, axis=1)
        return padded_feature

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        windowing: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s). sequences.

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For Speech2TextTransoformer models, `attention_mask` should alwys be passed for batched inference, to
                avoid subtle bugs.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            padding_value (`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
        """
        """
        wav2vec2는 단순 audio만 받으면 됐지만 transformer-transducer와 같은 streamming모델들은 extractor를 다르게 만들어야 한다.
        raw_audio, mel_spectrogram이 두게가 들어오는걸 상정하고 만들어야 한다.
        extractor


        np.ndarray: mini_batch 사이즈 만큼 들어올 때
        size(batch, audio_dim) or size(batch, mel_features, time_seq)
        
        List[float]: 그냥 raw_audio만 들어올때
        size(audio_dim)
        
        List[np.ndarray]: raw_audio가 batch_size만큼 들어올 때 ndarray.ver
        size(batch, audio_dim.ndarray_ver) or size(batch, mel_features.ndarray_ver, time_seq.ndarray_ver)

        List[List[float]: raw_audio가 batch_size만큼 들어올 때 list.ver
        size(batch, audio_dim.list_ver) or size(mel_features.list_ver, time_seq.list_ver)
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_sequential = isinstance(raw_speech, (list, tuple))
        inner_type = isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list))
        is_batched: bool = is_sequential and inner_type
        # list인데 내부에 ndarray나 list가 들어가 있는 경우를 찾는 듯
        # batched는 아마 list형식의 내부에 ndarray, list같은 값이 들어가 있는 경우를 뜻하는듯 함.

        if is_batched:  # List[np.ndarray], List[List[float]]인 경우
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):  # List[float]인 경우
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):  # np.ndarray인 경우
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched and raw_speech.ndim != 3:
            raw_speech = [raw_speech]
        # 여기 이후부터는 list 형식임.

        # extractor의 call은 값의 추출과 추출한 값들을 padding시키는 매소드다.
        # raw_speech.ndim != 3를 한 이유는 mel이 들어오더라도 2차원 값이 아닌 무조건 3차원 값이여야 하기 때문에 저런식으로 코드를 작성했다.

        features = [self.extract_features(waveform) for waveform in raw_speech]  # [bsz, raw_aduio]
        features = [self.compress_features(mel) for mel in features]  # [bsz, mel, raw_aduio]

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_features": features})
        # [TODO]: max_length를 여기서 지정해서 알아서 잘라내도록 할 수 있지 않을까?
        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        # make sure list is in array format
        input_features = padded_inputs.get("input_features")
        if isinstance(input_features[0], list):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]

        attention_mask = padded_inputs.get("attention_mask")
        if attention_mask is not None:
            padded_inputs["attention_mask"] = [np.asarray(array, dtype=np.int32) for array in attention_mask]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs
