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
        max_length=512,
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
        self.max_length = max_length
        self.extractor = extractor
        self.blank_id = blank_id

    def __call__(self, batch_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        add_blank: np.ndarray = lambda label: np.insert(label, 0, self.blank_id)[: self.max_length]

        features: list = [dataset["input_features"] for dataset in batch_dataset]
        labels: list = [dataset["labels"] for dataset in batch_dataset]

        features = [{"input_features": mel} for mel in features]
        labels = [{"input_ids": add_blank(label)} for label in labels]

        batch = self.extractor.pad(
            features,
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

        batch["labels"] = labels["input_ids"]
        batch["decoder_attention_mask"] = labels["attention_mask"]

        return batch
