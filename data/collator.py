from transformers import PreTrainedTokenizer
import torch
from typing import List, Dict, Any
import numpy as np
import math
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor


class TransducerCollator:
    max_length: int = 400

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        max_length=None,
        stack=None,
        stride=None,
        extractor: SequenceFeatureExtractor = None,
        blank_id=0,
    ) -> None:
        """DataLoader로 부터 건내받은 불규칙한 길이의 데이터를 일정한 길이오 만든 뒤 model에 건내주는 역할을 합니다.

        Transformer Transducer에서 처리하기 위해 Audio데이터를 처리합니다.

        **dataset의 형식**
            "input_values": numpy.2darray, shape(1, chnnel, time)
            "input_ids": numpy.1darray
            "audio_len": int
            "label_len": int

        현재 데이터에서 syllable이 오타가 있어서 syllabel이 되었음
        데이터는 다른 곳에서 이미 전처리가 된 상태로 만들어짐
        그래서 collator에서 512로 짜를 수 밖에 없다.


        Args:
            tokenizer (PreTrainedTokenizer, optional): _description_. Defaults to None.
        """
        self.tokenizer = tokenizer
        self.stack = stack
        self.stride = stride
        self.max_length = max_length
        self.extractor = extractor
        self.blank_id = blank_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        feature_select: str = lambda key: [feature[key] for feature in features]
        input_values: list = feature_select("input_features")
        labels: list = feature_select("labels")

        input_values = self.extractor.compress_features(input_values)
        input_values = [{"input_features": features} for features in input_values]
        labels = [{"input_values": np.insert(label, 0, self.blank_id)[:512]} for label in labels]

        batch = self.extractor.pad(
            input_values,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        labels = self.tokenizer.pad(
            labels,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch["labels"] = labels["input_values"]
        batch["decoder_attention_mask"] = labels["attention_mask"]

        return batch
