import torch
import torch.nn as nn


# [NOTE]: copied https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
class GaussianNoise(object):
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.std = std
        self.mean = mean

    @torch.no_grad()
    def __call__(self, model: nn.Module) -> None:
        # [NOTE]: copied from https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829/2
        model_parameters = zip(model.parameters(), model.named_parameters())
        for parameter, name in model_parameters:
            if "weight" in name:
                nosied_parameter = self.get_noise(parameter)
                parameter.add_(nosied_parameter)

    def get_noise(self, parameter: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(parameter.size(), device=parameter.device)
        parameter = (parameter + noise) * (self.std + self.mean)
        return parameter

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)
