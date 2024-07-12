import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch

from transformers.data.data_collator import DataCollatorMixin

from ..models import TransformerTransducerProcessor


@dataclass
class DataCollatorRNNTWithPadding(DataCollatorMixin):
    processor: TransformerTransducerProcessor
    sampling_rate: int = 16000
    padding: Union[bool, str] = "longest"
    return_tensors: str = "pt"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    feature_extractor_input_name: Optional[str] = "input_values"

    def torch_call(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {self.feature_extractor_input_name: feature[self.feature_extractor_input_name][0]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        feature_ls = list()
        mask_ls = list()
        for feature in features:
            input_features = feature["input_features"]
            chunk_num = math.ceil(len(input_features) / self.sampling_rate) * self.sampling_rate

            chunk_idxer = range(0, chunk_num, self.sampling_rate)
            chunk_audio_ls = [input_features[i : i + self.sampling_rate] for i in chunk_idxer]

            pro_outputs = self.processor(raw_speech=chunk_audio_ls)

            flatten_input_features = torch.vstack(pro_outputs["input_features"])
            flatten_attention_mask = torch.hstack(pro_outputs["attention_mask"])

            feature_ls.append({"input_features": flatten_input_features})
            mask_ls.append({"input_features": flatten_attention_mask})

        batch = self.processor.pad(input_features=feature_ls)

        return batch
