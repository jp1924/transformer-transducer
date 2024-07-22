import torch
import torch.nn as nn

from transformers.trainer_callback import TrainerCallback


# [NOTE]: copied https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
class GaussianNoiseCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.model: nn.Module = model

        self.noise_step = 10000

        self.noise_flag = self.noise_step < 1
        self.std = 0.01
        self.mean = 0.0

    @torch.no_grad()
    def on_step_begin(self, args, state, control, **kwargs) -> None:
        # [NOTE]: it's temp, it's will modify soon
        if self.noise_flag:
            return control

        if state.global_step < self.noise_step:
            return control

        # [NOTE]: copied from https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829/2
        model_parameters = zip(self.model.parameters(), self.model.named_parameters())
        for parameter, name in model_parameters:
            if "weight" in name:
                nosied_parameter = self.get_noise(parameter)
                parameter.add_(nosied_parameter)

        return control

    def get_noise(self, parameter: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(parameter.size(), device=parameter.device)
        parameter = (parameter + noise) * (self.std + self.mean)
        return parameter


class EmptyCacheCallback(TrainerCallback):
    def on_prediction_step(self, args, state, control, logs=None, **kwargs):
        torch.cuda.empty_cache()

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        torch.cuda.empty_cache()

    def on_step_begin(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 100 == 0:
            torch.cuda.empty_cache()
