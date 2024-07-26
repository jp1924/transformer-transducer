import sys


sys.path.insert(0, "/root/workspace")


import json
from typing import List

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from models import TransformerTransducerForRNNT, TransformerTransducerProcessor


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        model_name_or_path = model_config["parameters"]["model_name_or_path"]["string_value"]
        self.device = torch.device(f"""cuda:{args["model_instance_device_id"]}""")

        self.model = TransformerTransducerForRNNT.from_pretrained(model_name_or_path)
        self.processor = TransformerTransducerProcessor.from_pretrained(model_name_or_path)
        self.model = self.model.eval().to(self.device)

    @torch.no_grad()
    def execute(self, request_ls: list):
        response_ls = []
        for request in request_ls:
            text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy().tolist()
            audio = pb_utils.get_input_tensor_by_name(request, "audio").as_numpy()
            cache = pb_utils.get_input_tensor_by_name(request, "cache").as_numpy()

            input_params = self.processor(audio=audio, sample_rate=16000, return_tensors="pt")
            input_params = {k: v.to(self.device) for k, v in input_params.items()}
            input_params["mems"] = torch.tensor(cache, device=self.device)
            audio_features, mems = self.model.get_audio_features(**input_params)
            mems = torch.stack(mems)

            text = self.greedy_search(audio_features, text)

            outputs = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("text", np.array(text, dtype=np.int32)),
                    pb_utils.Tensor("cache", mems.detach().cpu().numpy()),
                ]
            )
            response_ls.append(outputs)
        return response_ls

    def greedy_search(
        self,
        audio_features: torch.Tensor,
        hyps_ls=None,
    ) -> List[int]:
        audio_features = audio_features[0]
        blank_id = self.model.config.blk_token_ids
        hyps_ls = hyps_ls if hyps_ls else [blank_id]

        decoder_out = self.model.get_text_features(torch.tensor([hyps_ls], device=self.device))[0][0]

        max_frame_num = audio_features.shape[0]
        cur_frame_num = 0

        max_utt_num = 10
        cur_utt_num = 0

        while cur_frame_num < max_frame_num:
            if cur_utt_num >= max_utt_num:
                cur_frame_num += 1
                continue

            frame = audio_features[cur_frame_num]
            joint_out = self.model.joint_network(frame + decoder_out)
            pred = joint_out.log_softmax(-1).argmax(-1).item()

            if pred != blank_id:
                hyps_ls.append(pred)
                decoder_out = self.model.get_text_features(torch.tensor([hyps_ls], device=self.device))[0][-1]
                cur_utt_num += 1
            else:
                cur_frame_num += 1
                cur_utt_num = 0

        return hyps_ls

    def finalize(self):
        pass
