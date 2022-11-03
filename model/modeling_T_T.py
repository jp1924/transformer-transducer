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
    def __init__(self, config) -> None:
        self.dense = nn.Linear()
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear()
        self.dropout = nn.Dropout()

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        input_values = self.dense(input_values)
        input_values = self.activation(input_values)
        input_values = self.linear_2(input_values)
        return input_values


class Encoder(nn.Module):
    def __init__(self, config) -> None:
        self.attn = SelfAttention(config)
        self.ffn = FeedForwardNetwork(config)
        self.norm = nn.LayerNorm()
        self.attn_dropout = nn.Dropout()
        self.ffn_dropout = nn.Dropout()

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # [NOTE]: computing attentions
        attn_output = self.attn(hidden_state, attention_mask)
        attn_output = self.attn_dropout(attn_output)
        hidden_state = hidden_state + attn_output
        hidden_state = self.norm(hidden_state)

        # [NOTE]: computing Feed_Forward Network
        ffn_output = self.ffn(hidden_state)
        ffn_output = self.ffn_dropout(ffn_output)
        hidden_state = hidden_state + ffn_output
        hidden_state = self.norm(hidden_state)

        return hidden_state


class LabelEncoder(nn.Module):
    def __init__(self) -> None:
        pass


class AudioEncoder(nn.Module):
    def __init__(self) -> None:
        pass


class TransformerTranducer(nn.Module):
    def __init__(self):
        pass
