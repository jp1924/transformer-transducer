from winreg import QueryValue
import torch
import torch.nn as nn
import math
from config import TransformerTransducerConfig

nn.TransformerEncoderLayer


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        # self.query_ = nn.Linear()
        # self.key = nn.Linear()
        # self.value = nn.Linear()

        self.score_dropout = nn.Dropout(config.score_dropout)

    def forward(
        self,
        hidden_size: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        query = hidden_size
        key = hidden_size
        value = hidden_size

        key = key.transpose(1, 2)
        attention = torch.matmul(query, key)

        k_d = key.shape(2)
        scaling_factor = torch.sqrt(k_d)
        scaled_attention = attention / scaling_factor

        scaled_attention += attention_mask
        attention_score = scaled_attention.softmax(dim=-1)

        attention_score = self.score_dropout(attention_score)

        attention_value = torch.matmul(attention_score, value)

        return attention_value


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        self.dense = nn.Linear(config.hidden_size, config.ffn_size)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(config.ffn_size, config.hidden_size)

    def forward(self, hidden_size: torch.Tensor) -> torch.Tensor:
        hidden_size = self.dense(hidden_size)
        hidden_size = self.activation(hidden_size)
        hidden_size = self.linear_2(hidden_size)
        return hidden_size


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        self.attn = SelfAttention(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.attn_norm_eps)
        self.attn_dropout = nn.Dropout(config.attn_dropout)

        self.ffn = FeedForwardNetwork(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.ffn_norm_eps)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout)

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # [NOTE]: computing attentions
        attn_output = self.attn(hidden_state, attention_mask)
        attn_output = self.attn_dropout(attn_output)
        hidden_state = hidden_state + attn_output
        hidden_state = self.attn_norm(hidden_state)

        # [NOTE]: computing Feed_Forward Network
        ffn_output = self.ffn(hidden_state)
        ffn_output = self.ffn_dropout(ffn_output)
        hidden_state = hidden_state + ffn_output
        hidden_state = self.ffn_norm(hidden_state)

        return hidden_state


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, config: TransformerTransducerConfig, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(config.embed_dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LabelEncoder(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:

        layer_list = [EncoderLayer(config) for _ in range(config.label_layers)]
        self.encoder = nn.ModuleList(layer_list)
        self.embeddings = nn.Embedding(config.position_embed_size, config.hidden_size)

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        for layer in self.encoder:
            hidden_state = layer(hidden_state, attention_mask)

        return hidden_state


class AudioEncoder(nn.Module):
    def __init__(self) -> None:
        pass


class TransformerTranducer(nn.Module):
    def __init__(self):
        pass
