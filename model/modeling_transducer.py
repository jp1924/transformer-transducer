import torch
import torch.nn as nn
from torchaudio.functional import rnnt_loss
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
import math
from typing import Optional, Tuple

from .config import TransformerTransducerConfig

# from accelerate import Accelerator
# accelerator = Accelerator()


class TransducerOuput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class TransducerSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class TransducerFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)

        return hidden_states


class TransducerEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.attention = TransducerSelfAttention(config)
        self.feed_forward = TransducerFeedForward(config)

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_residual = hidden_states
        hidden_states = self.attention(hidden_states, attention_mask=attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        """
            [NOTE]: 이 부분은 한번 확인할 것
                    일단은 변수를 줄이기 위해 wav2vec2의 FFN을 따라함.
            ffn_residual = hidden_states
            hidden_states = ffn_residual + self.feed_forward(hidden_states)
            hidden_states = self.final_layer_norm(hidden_states)
        """

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class LabelEncoder(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super().__init__()
        self.position_embedding = nn.Embedding(config.position_embed_size, config.hidden_size)
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_ids = torch.arange(config.position_embed_size)

        encoder_layers = [TransducerEncoderLayer(config) for _ in range(config.label_layers)]
        self.label_layers = nn.ModuleList(encoder_layers)

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

        hidden_state = word_embed + position_embed

        for layer in self.audio_layers:
            hidden_state = layer(hidden_state, attention_mask)

        return hidden_state


class AudioEncoder(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super().__init__()
        # self.position_embedding = PositionalEncoding(config.hidden_size)

        self.test_embedding = nn.Embedding(config.position_embed_size, config.hidden_size)
        self.test_ids = torch.arange(config.position_embed_size)

        self.linear = nn.Linear(80, config.hidden_size)

        encoder_layers = [TransducerEncoderLayer(config) for _ in range(config.audio_layers)]
        self.audio_layers = nn.ModuleList(encoder_layers)

    def forward(
        self,
        audio_inputs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        audio_inputs = audio_inputs.transpose(1, 2)

        hidden_state = self.linear(audio_inputs)

        for layer in self.audio_layers:
            hidden_state = layer(hidden_state, attention_mask)

        return hidden_state


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


class TransducerModel(nn.Moduel):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        self.audio_encoder = AudioEncoder(config)
        self.label_encoder = LabelEncoder(config)
        self.joint_network = JointNetwork(config)

        self.tanh = nn.Tanh()
        self.dense = nn.Linear(config.ffn_size, config.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None,
        label_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_state: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        audio_outputs = self.audio_encoder(input_values, audio_attention_mask)
        label_outputs = self.label_encoder(labels, label_attention_mask)
        joint_output = self.joint_network(audio_outputs, label_outputs)

        hidden_state = self.tanh(joint_output)
        hidden_state = self.dense(hidden_state)
        logits = self.log_softmax(hidden_state)

        return logits


class TransformerTranducerForRNNT(nn.Module):
    def __init__(self, config: TransformerTransducerConfig):
        super().__init__()
        self.config = config

        self.transducer = TransducerModel(config)

        self.reduction = config.loss_reduction
        self.blank_id = config.blank_id

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None,
        label_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_state: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        logits = self.transducer(
            input_values,
            audio_attention_mask,
            label_attention_mask,
            output_attentions,
            output_hidden_state,
            return_dict,
            labels,
        )

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

        test_1 = accelerator.pad_across_processes(logits, 1)
        test_2 = accelerator.pad_across_processes(test_1, 2)

        return TransducerOuput(loss=loss, logits=logits)
