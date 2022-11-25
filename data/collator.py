from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import List, Dict, Any
import numpy as np


class TorchCollator:
    def __init__(self, pad: int = None, device: int = None) -> None:
        self.pad = pad
        self.device = "cpu" if device < 0 else device

    def __call__(self, features) -> None:
        token_type_ids = [torch.tensor(dataset["token_type_ids"]) for dataset in features]
        input_values = [torch.tensor(dataset["input_values"]) for dataset in features]
        labels = [dataset["label"] for dataset in features]

        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.pad)
        input_values = pad_sequence(input_values, batch_first=True, padding_value=self.pad)
        labels = torch.tensor(labels)

        batch = {
            "input_values": input_values,
            "attention_mask": (input_values != 0).type(torch.long),
            "labels": labels,
            "token_type_ids": token_type_ids,
        }

        return batch


class TransducerCollator:

    max_length: int = 504

    def __init__(self, tokenizer: PreTrainedTokenizer = None) -> None:
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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        feature_select: str = lambda key: [feature[key] for feature in features]
        input_values: list = feature_select("input_values")
        labels: list = feature_select("labels")

        labels = [{"input_values": np.insert(label, 0, 0)} for label in labels]

        batch = self._audio_pad_2(input_values)
        labels = self.tokenizer.pad(
            labels,
            return_attention_mask=True,
            max_length=self.max_length,
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
        max_length = 512
        input_values = [value[:max_length, :] for value in input_values]

        pad_result = list()
        for audio in input_values:
            time_seq, mel_seq = audio.shape
            pad_zeros_mat = np.zeros([max_length - time_seq, mel_seq])
            padded_inputs = np.row_stack([audio, pad_zeros_mat])
            pad_result.append(padded_inputs)

        pad_result = torch.tensor(pad_result)

        attention_result = list()
        for audio in input_values:
            time_seq, mel_seq = audio.shape
            mask = np.ones([mel_seq, mel_seq])
            up = np.triu(mask, k=2 + 1)
            down = np.tril(mask, k=-10 - 1)
            attention_mask = up + down
            attention_result.append(attention_mask)
        attention_result = torch.tensor(attention_result)

        result = {
            "input_values": pad_result.to(torch.float32),
            "audio_attention_mask": attention_result.to(torch.float32),
        }

        return result
