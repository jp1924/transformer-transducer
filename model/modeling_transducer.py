import torch
import torch.nn as nn
from .config import TestConfig, TransformerTransducerConfig
from torchaudio.functional import rnnt_loss
from transformers.utils import ModelOutput


"""
    [NOTE]
    스스로 재작한 Transformer Encoder를 검증하기 위한 실험,
    torch.nn의 TransformerEncoderLayer와 TransformerEncoder를 이용해 Encoder를 만듬
    이후 Transducer를 전부 만들고 난 뒤 모델이 정상적으로 작동하는지 확인한다.
    민약 정상적으로 loss가 떨어진다면 이후 스스로 재작한 Encoder를 넣어서 테스트 하는 방식으로 진행

"""


class TransducerOuput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class SelfAttention(nn.Module):
    def __init__(self, config: TestConfig) -> None:
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
    def __init__(self, config: TestConfig) -> None:
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
    def __init__(self, config: TestConfig) -> None:
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
    def __init__(self, config: TestConfig) -> None:
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


# [NOTE]: 여기 위 부터는 테스트 용으로 만든 Transformer Encoder


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

        attention_mask = attention_mask.transpose(1, 0)

        position_ids = self.position_ids[: label_data.shape[1]]
        position_ids = position_ids.to(label_data.device)
        position_embed = self.position_embedding(position_ids)
        word_embed = self.word_embedding(label_data)

        input_embed = position_embed + word_embed

        outputs = self.encoder(input_embed, src_key_padding_mask=attention_mask)

        return outputs


class AudioEncoder(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super().__init__()
        # self.position_embedding = PositionalEncoding(config.hidden_size)

        self.test_embedding = nn.Embedding(config.position_embed_size, config.hidden_size)
        self.test_ids = torch.arange(config.position_embed_size)

        self.linear = nn.Linear(80, config.hidden_size)

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
        audio_inputs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        audio_inputs = audio_inputs.transpose(1, 2)
        attention_mask = attention_mask.transpose(1, 0)

        seq_len = audio_inputs.shape[1]
        length_ids = self.test_ids[:seq_len].to(audio_inputs.device)
        position_embed = self.test_embedding(length_ids)

        position_embed = position_embed.unsqueeze(0)
        audio_inputs = self.linear(audio_inputs)

        audio_inputs = audio_inputs + position_embed
        outputs = self.encoder(audio_inputs, src_key_padding_mask=attention_mask)

        return outputs


class JointNetwork(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super().__init__()
        concat_size = config.hidden_size * 2
        self.dense = nn.Linear(concat_size, config.ffn_size)

    def forward(
        self,
        audio_vector: torch.Tensor,
        label_vector: torch.Tensor,
    ) -> torch.Tensor:

        audio_len = audio_vector.shape[1]
        label_len = label_vector.shape[1]

        audio_vector = audio_vector.unsqueeze(2)
        label_vector = label_vector.unsqueeze(1)
        # [NOTE]: logits (Tensor) – Tensor of dimension (batch, max seq length, max target ""length + 1"", class) containing output from joiner
        audio_vector = audio_vector.repeat(1, 1, label_len, 1)
        label_vector = label_vector.repeat(1, audio_len, 1, 1)

        hidden_state = torch.cat([audio_vector, label_vector], dim=-1)
        hidden_state = self.dense(hidden_state)

        return hidden_state


class TransformerTranducer(nn.Module):
    def __init__(self, config: TransformerTransducerConfig):
        super().__init__()
        self.config = config

        self.audio_encoder = AudioEncoder(config)
        self.label_encoder = LabelEncoder(config)
        self.joint_network = JointNetwork(config)

        self.tanh = nn.Tanh()
        self.dense = nn.Linear(config.ffn_size, config.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.reduction = config.loss_reduction
        self.blank_id = config.blank_id

    def forward(
        self,
        input_values: torch.Tensor,
        labels: torch.Tensor,
        audio_attention_mask: torch.Tensor,
        label_attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        audio_vector = self.audio_encoder(input_values, audio_attention_mask)
        label_vector = self.label_encoder(labels, label_attention_mask)
        joint_vector = self.joint_network(audio_vector, label_vector)

        hidden_state = self.tanh(joint_vector)
        hidden_state = self.dense(hidden_state)
        logits = self.log_softmax(hidden_state)

        label_len = torch.IntTensor([torch.masked_select(tensor, tensor != 0).shape[0] for tensor in labels])
        non_blank_labels = torch.stack([tensor[1:] for tensor in labels]).to(torch.int32)
        audio_len = torch.IntTensor(
            [torch.masked_select(tensor[0], tensor[0] != 0.0).shape[0] for tensor in input_values]
        )

        label_len = label_len.to(labels.device)
        audio_len = audio_len.to(input_values.device)
        non_blank_labels = non_blank_labels.to(labels.device)

        loss = rnnt_loss(
            logits=logits,
            targets=non_blank_labels,
            logit_lengths=audio_len,
            target_lengths=label_len,
            blank=self.blank_id,
            clamp=-1,
            reduction=self.reduction,
        )

        return TransducerOuput(loss=loss, logits=logits)
