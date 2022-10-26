from torch.nn.utils.rnn import pad_sequence
import torch


class TorchCollator:
    def __init__(self, pad: int = None) -> None:
        self.pad = pad

    def __call__(self, features) -> None:
        token_type_ids = [torch.tensor(dataset["token_type_ids"]) for dataset in features]
        input_ids = [torch.tensor(dataset["input_ids"]) for dataset in features]
        labels = [dataset["label"] for dataset in features]

        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.pad)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad)
        labels = torch.tensor(labels)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "token_type_ids": token_type_ids,
        }

        return batch
