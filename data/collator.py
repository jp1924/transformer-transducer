from transformers import PreTrainedTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch
from typing import List, Dict, Any


class TorchCollator:
    def __init__(self, pad: int = None, device: int = None) -> None:
        self.pad = pad
        self.device = "cpu" if device < 0 else device

    def __call__(self, features) -> None:
        token_type_ids = [torch.tensor(dataset["token_type_ids"]) for dataset in features]
        input_ids = [torch.tensor(dataset["input_ids"]) for dataset in features]
        labels = [dataset["label"] for dataset in features]

        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.pad)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad)
        labels = torch.tensor(labels)

        batch = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != 0).type(torch.long),
            "labels": labels,
            "token_type_ids": token_type_ids,
        }

        return batch


class TransducerCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer = None) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:

        print

        return
