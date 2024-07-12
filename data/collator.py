import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from transformers.data.data_collator import DataCollatorMixin


# from ..models import TransformerTransducerProcessor


@dataclass
class DataCollatorRNNTWithPadding(DataCollatorMixin):
    processor: "TransformerTransducerProcessor"
    sampling_rate: int = 16000
    padding: Union[bool, str] = "longest"
    return_tensors: str = "pt"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def torch_call(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_featurse": x["audio"]["array"]} for x in features]
        # labels = [{"input_ids": feature["sentence"]} for feature in features]
        labels = [
            {"input_ids": self.processor(text=f"""<blank>{feature["sentence"]}""")["input_ids"]}
            for feature in features
        ]

        feature_ls = list()
        mask_ls = list()
        check_ls = list()
        for feature in features:
            # input_features = feature["input_features"]
            input_features = feature["audio"]["array"]
            chunk_num = math.ceil(len(input_features) / self.sampling_rate) * self.sampling_rate

            chunk_idxer = range(0, chunk_num, self.sampling_rate)
            chunk_audio_ls = [input_features[i : i + self.sampling_rate] for i in chunk_idxer]

            # torch input 입력하는 경우 error가 발생 함.
            pro_outputs = self.processor(
                audio=chunk_audio_ls,
                sampling_rate=16000,
                return_tensors="pt",
            )
            check_ls.append(pro_outputs["attention_mask"].sum(-1))

            flatten_input_features = np.vstack(pro_outputs["input_features"])
            flatten_attention_mask = np.hstack(pro_outputs["attention_mask"])

            feature_ls.append({"input_features": flatten_input_features})
            mask_ls.append({"input_ids": flatten_attention_mask})

        batch = self.processor.pad(input_features=feature_ls, return_tensors=self.return_tensors)
        batch["attention_mask"] = self.processor.pad(labels=mask_ls, return_tensors=self.return_tensors)["input_ids"]

        max_shape = batch["attention_mask"].sum(-1).max()
        batch["attention_mask"] = batch["attention_mask"][:, :max_shape]
        batch["input_features"] = batch["input_features"][:, :max_shape]

        _labels = self.processor.pad(labels=labels, return_tensors=self.return_tensors)

        batch["labels"] = _labels["input_ids"]
        batch["decoder_attention_mask"] = _labels["attention_mask"]

        return batch
