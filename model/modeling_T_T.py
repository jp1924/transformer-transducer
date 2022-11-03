from winreg import QueryValue
import torch
import torch.nn as nn

nn.TransformerEncoderLayer


class SelfAttention(nn.Module):
    def __init__(self, config: dict) -> None:
        self.query_ = nn.Linear()
        self.key = nn.Linear()
        self.value = nn.Linear()

        self.attention_dropout = nn.Dropout(0.06)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        query = input_ids
        key = input_ids
        value = input_ids

        key = key.transpose(1, 2)

        attention = torch.matmul(query, key)

        k_d = key.shape(2)
        scaling_factor = torch.sqrt(k_d)
        scaled_attention = attention / scaling_factor

        scaled_attention *= attention_mask
        attention_score = scaled_attention.softmax(dim=-1)

        attention_score = self.attention_dropout(attention_score)

        attention_value = torch.matmul(attention_score, value)

        return attention_value


class FeedForwardNetwork(nn.Module):
    def __init__(self) -> None:
        pass


class Encoder(nn.Module):
    def __init__(self) -> None:
        pass


class LabelEncoder(nn.Module):
    def __init__(self) -> None:
        pass


class AudioEncoder(nn.Module):
    def __init__(self) -> None:
        pass


class TransformerTranducer(nn.Module):
    def __init__(self):
        pass
