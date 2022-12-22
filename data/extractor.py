import math
from typing import Dict, List, Optional, Union, Any, Callable

import numpy as np
from numpy.fft import fft
import torch

from transformers.file_utils import is_torchaudio_available


if is_torchaudio_available():
    from torchaudio.transforms import MelSpectrogram, Spectrogram
    import torchaudio.functional as F

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)

DECIBEL = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
GAIN = pow(10, 0.05 * DECIBEL)

# [NOTE]: copied from whisper extractor
class TransducerFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        power: int = None,
        window_fn: Callable = None,
        min_frequency: float = None,
        max_frequency: float = None,
        stack: Optional[int] = 4,  # value from transformer-transducer paper
        stride: Optional[int] = 3,  # value from transformer-transducer paper
        padding_value: float = 0.0,
        return_attention_mask: bool = False,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        # [NOTE]: for Log-MelSpectrogram
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.mel_filters = self.get_mel_filters(sampling_rate, n_fft, n_mels=feature_size)

        # np.hanning(self.n_fft + 1)[:-1]애서 (self.n_fft + 1)[:-1]를 하는 이유를 모르겠음.
        # 만약 n_fft가 400이라 했을 때 + 1 를 하는 건 window_length를 401개의 window값을 생성한다는 소리
        # 근데 거기서 다시 [:-1]를 하는게 이해가 안됨. 값의 scale를 낮추는 역할을 하는 건가/

        # window = np.hanning(self.n_fft + 1)[:-1]
        self.window = np.hanning(self.n_fft)

        if is_torchaudio_available():
            self.window_fn: torch.Tensor = lambda x: torch.tensor(np.hanning(x), dtype=torch.float32)

            self.mel_transform = MelSpectrogram(
                n_fft=self.n_fft,
                n_mels=self.feature_size,
                sample_rate=self.sampling_rate,
                hop_length=self.hop_length,
                window_fn=self.window_fn,
                pad_mode="reflect",
                center=True,
            )
            self.spectrogram = Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window_fn=self.window_fn,
                pad_mode="reflect",
                center=True,
            )

            F.melscale_fbanks

        # [NOTE]: for compression
        self.stack = stack
        self.stride = stride

        assert self.stack is not None or self.stride is not None, "windowing이 정상적으로 작동하지 않습니다!"

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
        for stack_num in range(self.stack):
            idx_iter = range(0, (time_steps - stack_num), self.stride)
            indices = [stack_num + idx for idx in idx_iter]

            features = mel_feature[indices]
            pad_width = ((0, expected_len - features.shape[0]), (0, 0))
            # 더 간단한 방법이 있지만 그렇게 하기에는 너무 대충한 느낌이 든다.
            features = np.pad(features, pad_width)

            mel_store.append(features)

        padded_feature = np.concatenate(mel_store, axis=1)
        return padded_feature

    def piecewise_linear_log(features):
        features[features > math.e] = torch.log(features[features > math.e])
        features[features <= math.e] = features[features <= math.e] / math.e
        return features

    def extract_features(self, waveform: torch.Tensor) -> np.ndarray:
        """
        Extract the mel spectrogram features and normalize them
        """
        features = self.mel_transform(waveform)
        features = features.transpose(1, 0)

        features = self.piecewise_linear_log((features * GAIN))
        features = (features - self.global_mean) * self.global_invstddev
        features = features.numpy()

        return features

    def get_mel_filters(self, sr, n_fft, n_mels=128, dtype=np.float32):
        # Initialize the weights

        # htk가 아닌 slaney로 진행함.

        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = np.linspace(min_mel, max_mel, n_mels + 2)
        mels = np.asanyarray(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        mel_f = freqs

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

        return weights

    def fram_wave(self, waveform, center=True):
        """
        Transform a raw waveform into a list of smaller waveforms. The window length defines how much of the signal is
        contain in each frame (smalle waveform), while the hope length defines the step between the beginning of each
        new frame.

        Centering is done by reflecting the waveform which is first centered around `frame_idx * hop_length`.
        """
        frames = []
        for i in range(0, waveform.shape[0] + 1, self.hop_length):
            half_window = (self.n_fft - 1) // 2 + 1
            # half_window = (self.n_fft) // 2 + 1

            if center:
                start = i - half_window if i > half_window else 0
                end = i + half_window if i < waveform.shape[0] - half_window else waveform.shape[0]

                frame = waveform[start:end]

                if start == 0:
                    padd_width = (-i + half_window, 0)
                    frame = np.pad(frame, pad_width=padd_width, mode="reflect")

                elif end == waveform.shape[0]:
                    padd_width = (0, (i - waveform.shape[0] + half_window))
                    frame = np.pad(frame, pad_width=padd_width, mode="reflect")

            else:
                frame = waveform[i : i + self.n_fft]
                frame_width = frame.shape[0]
                if frame_width < waveform.shape[0]:
                    frame = np.lib.pad(
                        frame, pad_width=(0, self.n_fft - frame_width), mode="constant", constant_values=0
                    )

            frames.append(frame)
        return np.stack(frames, 0)

    def stft(self, frames, window):
        """
        Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same
        results as `torch.stft`.
        """
        frame_size = frames.shape[1]
        fft_size = self.n_fft

        if fft_size is None:
            fft_size = frame_size

        if fft_size < frame_size:
            raise ValueError("FFT size must greater or equal the frame size")
        # number of FFT bins to store
        num_fft_bins = (fft_size >> 1) + 1

        data = np.empty((len(frames), num_fft_bins), dtype=np.complex64)
        fft_signal = np.zeros(fft_size)

        for f, frame in enumerate(frames):
            if window is not None:
                np.multiply(frame, window, out=fft_signal[:frame_size])
            else:
                fft_signal[:frame_size] = frame
            data[f] = fft(fft_signal, axis=0)[:num_fft_bins]
        return data.T

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-Mel spectrogram of the provided audio, gives similar results whisper's original torch
        implementation with 1e-5 tolerance.
        """
        filters = self.mel_filters
        window = np.hanning(self.n_fft + 1)[:-1]

        # [NOTE]: Mel-Spectrogram
        frames = self.fram_wave(waveform)
        stft = self.stft(frames, window=window)
        magnitudes = np.abs(stft[:, :-1]) ** 2
        mel_spec = filters @ magnitudes

        # [NOTE]: Mel-Filter
        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)

        # [NOTE]: log scale
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For WhisperTransoformer models, `attention_mask` should alwys be passed for batched inference, to avoid
                subtle bugs.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            padding_value (`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
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

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = BatchFeature({"input_features": raw_speech})

        # convert into correct format for padding

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length else self.n_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            **kwargs,
        )
        # make sure list is in array format
        input_features = padded_inputs.get("input_features").transpose(2, 0, 1)

        # 내 생각에는 torchaudio가 설치되어 있지 않으면 python구현 mel-spectrogram으로 가고,
        # 만약 있으면 torchaudio melspectrogram으로 진행할려 했지만 그걸 구현하지 않은 것 같다

        filters = self.mel_filters
        window = self.window
        # [NOTE]: Mel-Spectrogram
        frames = self.fram_wave(input_features[0][0], center=True)
        stft = self.stft(frames, window=window)
        # magnitudes = np.abs(stft[:, :-1]) ** 2
        magnitudes = np.abs(stft) ** 2
        mel_spec = filters @ magnitudes

        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)

        # [NOTE]: log scale
        test_1 = (log_spec + 4.0) / 4.0

        test_2 = self.mel_transform(torch.tensor(input_features[0][0]))[:, :-1].numpy()

        if is_torchaudio_available():
            # device가 있는 상태로 들어오면 어떻게 하지?
            input_features = []
        else:
            input_features = [self._np_extract_fbank_features(waveform) for waveform in input_features[0]]

        if isinstance(input_features[0], List):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]
        else:
            padded_inputs["input_features"] = input_features

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs
