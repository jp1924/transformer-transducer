from transformers import PreTrainedTokenizer
import torch
from typing import List, Dict, Any
import numpy as np
import math
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor


class TransducerCollator:

    max_length: int = 504

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        max_length=None,
        stack=None,
        stride=None,
        extractor: SequenceFeatureExtractor = None,
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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        feature_select: str = lambda key: [feature[key] for feature in features]
        input_values: list = feature_select("input_features")
        labels: list = feature_select("labels")

        input_values = self.extractor.compress_features(input_values)
        input_values = [{"input_features": features} for features in input_values]
        labels = [{"input_values": np.insert(label, 0, 2)[:512]} for label in labels]

        batch = self.extractor.pad(
            input_values,
            padding=True,
            max_length=400,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        batch["audio_attention_mask"] = batch.pop("attention_mask")
        # batch = self._audio_pad_2(input_values)
        labels = self.tokenizer.pad(
            labels,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch["labels"] = labels["input_values"]
        batch["label_attention_mask"] = labels["attention_mask"]

        return batch

    def _audio_pad_1(self, input_values: List[torch.Tensor]) -> torch.Tensor:
        chennel_size = input_values[0].shape[1]

        # [NOTE]: cutting by max_length
        input_values = [value[:, :, : self.max_length] for value in input_values]
        max_size = 504  # max([value.shape[2] for value in input_values])

        difference = [max_size - value.shape[2] for value in input_values]

        attention_masks = list()
        padded_values = list()
        for pad_size, value in zip(difference, input_values):

            value_size = value.shape[2]
            ones = torch.ones([1, value_size])

            pad = torch.zeros([1, chennel_size, pad_size], dtype=torch.float64)

            value = torch.cat([value, pad], dim=2)
            mask = torch.cat([ones, pad[:, 0, :]], dim=1)

            padded_values.append(value)
            attention_masks.append(mask)
        input_values = torch.cat(padded_values, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0)

        result = {
            "input_values": input_values.to(torch.float32),
            "audio_attention_mask": attention_mask.to(torch.float32),
        }

        return result

    def _audio_pad_2(self, input_values: List[torch.Tensor]) -> torch.Tensor:

        value_list = list()
        for mel_feature in input_values:
            time_steps, features_dim = mel_feature.shape
            expected_len = math.ceil(time_steps / self.stride)
            mel_store = list()
            for step in range(self.stack):
                indices = [step + idx for idx in range(0, (time_steps - step), self.stride)]
                features = mel_feature[indices]

                # [TODO]: 이 부분은 extractor의 pad 기능이 하는게 맞지만 임시로 이렇게 만든다.
                pad_width = ((0, expected_len - features.shape[0]), (0, 0))
                features = np.pad(features, pad_width)

                mel_store.append(features)

            padded_feature = np.concatenate(mel_store, axis=1)
            value_list.append(torch.tensor(padded_feature))

        attention_result = list()
        for audio in value_list:
            time_seq, mel_seq = audio.shape
            mask = np.ones([mel_seq, mel_seq])
            up = np.triu(mask, k=2 + 1)
            down = np.tril(mask, k=-10 - 1)
            attention_mask = up + down
            attention_result.append(attention_mask)
        attention_result = torch.tensor(attention_result)

        value_result = torch.stack(value_list)

        result = {
            "input_values": value_result.transpose(2, 1),
            "audio_attention_mask": attention_result.to(torch.float32),
        }

        return result
