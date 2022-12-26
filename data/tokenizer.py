# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization class for the Emformer model."""


import json
import os
import sys
import warnings
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import PaddingStrategy, TensorType, add_end_docstrings, logging

logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json",
    },
    "tokenizer_config_file": {
        "facebook/wav2vec2-base-960h": (
            "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/tokenizer_config.json"
        ),
    },
}

# Wav2Vec2 has no max input length
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/wav2vec2-base-960h": sys.maxsize}

WAV2VEC2_KWARGS_DOCSTRING = r"""
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.
                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
"""

# [NOTE]: RNN-T, like CTC, predicts speech as phonemes. so Sentence Piece Tokenizer should not be used.
#         But I'm not sure if this is the right Tokenizer for RNN-T.
#         It may be modified later!


# [NOTE]: copied from Wav2Vec2Tokenizer
class TransducerTokenizer(PreTrainedTokenizer):
    """
    Constructs a Wav2Vec2 tokenizer.
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        word_delimiter_token (`str`, *optional*, defaults to `"|"`):
            The token used for defining the end of a word.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the output when decoding.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models, *e.g.*,
            [wav2vec2-lv60](https://huggingface.co/models?search=lv60).
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~Wav2Vec2Tokenizer.__call__`] should return `attention_mask`.
            <Tip>
            Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as
            [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), have **not** been trained using
            `attention_mask`. For such models, `input_values` should simply be padded with 0 and no `attention_mask`
            should be passed.
            For Wav2Vec2 models that have set `config.feat_extract_norm == "layer"`, such as
            [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self), `attention_mask` should be
            passed for batched inference.
            </Tip>
        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = {
        "vocab_file": {
            "facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json"
        },
        "tokenizer_config_file": {
            "facebook/wav2vec2-base-960h": (
                "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/tokenizer.json"
            ),
        },
    }
    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        do_normalize=False,
        return_attention_mask=False,
        **kwargs,
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            do_normalize=do_normalize,
            return_attention_mask=return_attention_mask,
            word_delimiter_token=word_delimiter_token,
            **kwargs,
        )

        warnings.warn(
            "The class `Wav2Vec2Tokenizer` is deprecated and will be removed in version 5 of Transformers. Please use"
            " `Wav2Vec2Processor` or `Wav2Vec2CTCTokenizer` instead.",
            FutureWarning,
        )

        self._word_delimiter_token = word_delimiter_token

        self.do_lower_case = do_lower_case
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def word_delimiter_token(self) -> str:
        """
        `str`: Padding token. Log an error if used while not having been set.
        """
        if self._word_delimiter_token is None and self.verbose:
            logger.error("Using word_delimiter_token, but it is not set yet.")
            return None
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        self._word_delimiter_token = value

    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    @add_end_docstrings(WAV2VEC2_KWARGS_DOCSTRING)
    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.
        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrayr or a list of list of float values.
        """

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        # make sure input is in list format
        if is_batched and not isinstance(raw_speech[0], np.ndarray):
            raw_speech = [np.asarray(speech) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # zero-mean and unit-variance normalization
        if self.do_normalize:
            raw_speech = [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in raw_speech]

        # convert into correct format for padding
        encoded_inputs = BatchEncoding({"input_values": raw_speech})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return padded_inputs

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.unk_token)
        return result

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        # group same tokens into non-repeating tokens in CTC style decoding
        grouped_tokens = [token_group[0] for token_group in groupby(tokens)]

        # filter self.pad_token which is used as CTC-blank token
        filtered_tokens = list(filter(lambda token: token != self.pad_token, grouped_tokens))

        # replace delimiter token
        string = "".join([" " if token == self.word_delimiter_token else token for token in filtered_tokens]).strip()

        if self.do_lower_case:
            string = string.lower()

        return string

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> str:
        """
        special _decode function is needed for Wav2Vec2Tokenizer because added tokens should be treated exactly the
        same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be called on
        the whole token list and not individually on added tokens
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        text = self.convert_tokens_to_string(result)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)
