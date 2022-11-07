import torch
import torch.nn as nn
import math
from .config import TransformerTransducerConfig

nn.TransformerEncoderLayer


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super(SelfAttention, self).__init__()
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

        k_d = key.shape[2]

        key = key.transpose(1, 2)
        attention = torch.matmul(query, key)

        scaling_factor = torch.sqrt(torch.tensor(k_d))
        scaled_attention = attention / scaling_factor
        shape = attention_mask.shape
        scaled_attention += attention_mask.view(shape[0], 1, shape[1])
        attention_score = scaled_attention.softmax(dim=-1)

        attention_score = self.score_dropout(attention_score)

        attention_value = torch.matmul(attention_score, value)

        return attention_value


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super(FeedForwardNetwork, self).__init__()
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
        super(EncoderLayer, self).__init__()
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
        hidden_state = self.attn_norm(hidden_state)  # 논문에서는 이렇게 구현되어 있지 않음

        # [NOTE]: computing Feed_Forward Network
        ffn_output = self.ffn(hidden_state)
        ffn_output = self.ffn_dropout(ffn_output)
        hidden_state = hidden_state + ffn_output
        hidden_state = self.ffn_norm(hidden_state)  # 논문에서는 이렇게 구현되어 있지 않음

        return hidden_state


class TestEncoder(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super(TestEncoder, self).__init__()
        layer_list = [EncoderLayer(config) for _ in range(config.label_layers)]
        self.encoder = nn.ModuleList(layer_list)
        self.position_embeddings = nn.Embedding(config.position_embed_size, config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        self.dense_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler = nn.Tanh()
        # [NOTE]: 테스트용 레이어, NSMC를 이용해 loss가 정상적으로 떨어지는 지 확인.
        self.test_dropout = nn.Dropout(0.02)
        self.test_classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor, _) -> torch.Tensor:
        position = torch.arange(input_values.shape[1])

        pos_embed = self.position_embeddings(position)
        word_embed = self.word_embeddings(input_values)
        hidden_state = pos_embed + word_embed

        for layer in self.encoder:
            hidden_state = layer(hidden_state, attention_mask)

        hidden_state = hidden_state[:, 0]
        hidden_state = self.dense_layer(hidden_state)
        hidden_state = self.pooler(hidden_state)

        hidden_state = self.test_dropout(hidden_state)
        logits = self.test_classifier(hidden_state)

        return logits


class LabelEncoder(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(config.position_embed_size, config.hidden_size)
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.position_ids = torch.arange(config.position_embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.head_size,
            dim_feedforward=config.ffn_size,
            dropout=config.ffn_dropout,
            layer_norm_eps=config.ffn_norm_eps,
            batch_first=False,
            norm_first=False,
            device=None,
            dtype=None,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, config.label_layers)

    def forward(
        self,
        label_data: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = self.position_ids[label_data.shape[1]]
        position = self.position_embedding(position_ids)
        word = self.word_embedding(label_data)
        input_embeddings = position + word

        outputs = self.encoder(input_embeddings, attention_mask)

        return outputs


class AudioEncoder(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(config.position_embed_size, config.hidden_size)

        self.position_ids = torch.arange(config.position_embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.head_size,
            dim_feedforward=config.ffn_size,
            dropout=config.ffn_dropout,
            layer_norm_eps=config.ffn_norm_eps,
            batch_first=False,
            norm_first=False,
            device=None,
            dtype=None,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, config.audio_layers)

    def forward(
        self,
        audio_data: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = self.position_ids[audio_data.shape[1]]
        input_embeddings = self.position_embedding(position_ids)

        outputs = self.encoder(input_embeddings, attention_mask)

        return outputs


class JointNetwork(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super().__init__()
        joint_input_size = config.hidden_size * 2

        self.dense_1 = nn.Linear(joint_input_size, config.ffn_size)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(config.joint_dropout_1)
        self.dense_2 = nn.Linear(config.ffn_size, config.hidden_size)
        self.dropout_2 = nn.Dropout(config.joint_dropout_2)

    def forward(
        self,
        audio_state: torch.Tensor,
        label_state: torch.Tensor,
    ) -> torch.Tensor:
        joint_state = torch.cat([audio_state, label_state], dim=-1)

        hideen_state = self.dense_1(joint_state)
        hideen_state = self.relu(hideen_state)
        hideen_state = self.dropout_1(hideen_state)
        hideen_state = self.dense_2(hideen_state)
        hideen_state = self.dropout_2(hideen_state)

        return hideen_state


class TransformerTranducer(nn.Module):
    def __init__(self, config: TransformerTransducerConfig):
        super().__init__()

        self.label_encoder = LabelEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.joint_network = JointNetwork(config)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self) -> torch.Tensor:
        return
