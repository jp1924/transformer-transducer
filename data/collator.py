import math
from dataclasses import dataclass
from typing import Dict, List, Union

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

    def get_audio_chunk_ls(self, input_features):
        chunk_num = math.ceil(len(input_features) / self.sampling_rate) * self.sampling_rate

        chunk_idxer = range(0, chunk_num, self.sampling_rate)
        chunk_audio_ls = list()
        for i in chunk_idxer:
            chunk_audio = input_features[i : i + self.sampling_rate]

            # mel로 변환할 때 음성의 길이가 너무 짧으면 processor에서 error가 발생 함.
            if chunk_audio.shape[0] < self.processor.feature_extractor.n_fft:
                padded_array = np.zeros(self.processor.feature_extractor.n_fft)
                padded_array[: chunk_audio.shape[0]] = chunk_audio
                chunk_audio = padded_array

            chunk_audio_ls.append(chunk_audio)
        # chunk_audio_ls = [input_features[i : i + self.sampling_rate] for i in chunk_idxer]
        return chunk_audio_ls

    def torch_call(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        labels = [{"input_ids": feature["labels"]} for feature in features]

        feature_ls = list()
        mask_ls = list()
        check_ls = list()
        for feature in features:
            input_features = feature["input_features"].numpy()
            chunk_audio_ls = self.get_audio_chunk_ls(input_features)

            # torch input 입력하는 경우 error가 발생 함.
            pro_outputs = self.processor(
                audio=chunk_audio_ls,
                sampling_rate=self.sampling_rate,
                return_tensors=self.return_tensors,
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

        batch = {k: v.to(torch.float32) for k, v in batch.items()}
        batch["labels"] = batch["labels"].to(torch.long)

        return batch
