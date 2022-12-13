import copy
import heapq
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchaudio.functional import rnnt_loss
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from .config import TransformerTransducerConfig


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


@dataclass
class GreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GreedySearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]


@dataclass
class EncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor
    hidden_states: torch.Tensor = None
    attentions: torch.Tensor = None


@dataclass
class JoinerOutput(ModelOutput):
    logtis: torch.Tensor
    audio_hidden: torch.Tensor
    label_hidden: torch.Tensor


@dataclass
class TransducerBaseModelOutput(ModelOutput):
    logits: torch.Tensor

    audio_last_hidden_state: torch.Tensor = None
    audio_hidden_states: torch.Tensor = None
    audio_attentions: torch.Tensor = None

    label_last_hidden_state: torch.Tensor = None
    label_hidden_states: torch.Tensor = None
    label_attentions: torch.Tensor = None


@dataclass
class RNNTBaseOutput(ModelOutput):
    loss: torch.Tensor = None
    logits: torch.Tensor = None

    audio_hidden_states: torch.Tensor = None
    audio_attentions: torch.Tensor = None

    label_hidden_states: torch.Tensor = None
    label_attentions: torch.Tensor = None


class TransducerPretrainedModel(PreTrainedModel):
    config_class = TransformerTransducerConfig
    base_model_prefix = "transformertransducer"
    main_input_name = "input_features"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TransducerEncoderLayer):
            module.gradient_checkpointing = value


class TransducerSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, position_embedding_type=None):
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
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        self.ffn_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.ffn_dense(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)

        return hidden_states


class TransducerEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
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
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        attention_residual = hidden_states
        hidden_states = self.attention(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = self.dropout(hidden_states[0])
        hidden_states = attention_residual + hidden_states
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


# ======================================================


class TransducerDecoder(TransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super(TransducerPretrainedModel, self).__init__(config)
        self.config = config
        self.is_decoder = True

        if True:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.activation_dropout,
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                norm_first=False,
                device=None,
                dtype=None,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, config.label_layers)
            self.head_size = config.num_attention_heads
        else:
            encoder_layers = [TransducerEncoderLayer(config) for _ in range(config.label_layers)]
            self.layers = nn.ModuleList(encoder_layers)

        self.position_ids = torch.arange(config.position_embed_size, device=self.device)
        self.position_embeddings = nn.Embedding(config.position_embed_size, config.hidden_size)
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int = 0,
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[EncoderOutput, Tuple[Any]]:
        input_len = labels.shape[1] if labels.dim() == 2 else labels.shape[0]

        # [XXX]: 이 부분은 나중에 확인 필요
        #         코드가 너무 길어서 나중에 수정될 수 있음
        label_shape = labels.shape
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, label_shape) if self.training else None

        # [XXX]: 나중에 따로 모듈화 해야할 수도 있음
        position_ids = self.position_ids[:input_len]
        position_embed = self.position_embeddings(position_ids)
        word_embed = self.word_embedding(labels)

        hidden_state = word_embed + position_embed

        if True:
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze(1).bool()
                attention_mask = attention_mask.repeat(self.head_size, 1, 1)
                attention_mask = ~attention_mask
            hidden_state = self.encoder(hidden_state, attention_mask)
        else:
            for layer in self.layers:
                hidden_state = layer(hidden_state, attention_mask)

        if not return_dict:
            return None

        return EncoderOutput(hidden_state)


class TransducerEncoder(TransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super(TransducerPretrainedModel, self).__init__(config)
        self.config = config

        if True:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.activation_dropout,
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                norm_first=False,
                device=None,
                dtype=None,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, config.audio_layers)
            self.head_size = config.num_attention_heads

        else:
            self.test_embedding = nn.Embedding(config.position_embed_size, config.hidden_size)
            self.test_ids = torch.arange(config.position_embed_size)

            self.linear = nn.Linear(80, config.hidden_size)

            encoder_layers = [TransducerEncoderLayer(config) for _ in range(config.audio_layers)]
            self.layers = nn.ModuleList(encoder_layers)
        self.position_ids = torch.arange(512, device=self.device)
        self.position_embeddings = nn.Embedding(512, config.hidden_size)

    def _prepare_encoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        dtype: Optional[torch.float] = None,
    ) -> torch.Tensor:
        # [NOTE]: opt model의 _prepare_decoder_attention_mask 에서 따왔다.
        #         huggingface의 encoder, deocder형식을 가진 seq2seq모델의 경우 model이 아닌 decoder에 casual mask를 생성하는
        #         _prepare_decoder_attention_mask가 존재함. 하지만 Transducer의 경우 Encoder에서 별도의 mask를 생성해야 하기 때문에
        #         _prepare_encoder_attention_mask로 이름을 바꿔서 TransducerEncoder에 집어넣었음

        if dtype is None:
            dtype = self.dtype

        if len(input_shape) == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
            return extended_attention_mask.to(dtype)

        mask_type = self.config.mask_type
        if mask_type == "chunk-wise":
            extended_attention_mask = self._create_chunk_attention_mask(
                attention_mask,
                input_shape,
                chunk_size=3,
            )
        elif mask_type == "diagonal":
            extended_attention_mask = self._create_diag_attention_mask(
                attention_mask,
                input_shape,
                left_context=10,
                right_context=3,
            )
            # [TODO]: 나중에 huggingface encoder 사용할 때 제거할 것!!!
            extended_attention_mask = extended_attention_mask == 0
        elif mask_type == "full":
            extended_attention_mask = attention_mask[:, None, :]
            # extended_attention_mask[:, None, :, :] 여기에서 차원이 추가되서 4차원이 됨
        else:
            raise ValueError("diagonal, chunk-wise, full 중 하나를 선택해 주세요!")

        extended_attention_mask = extended_attention_mask[:, None, :, :]
        return extended_attention_mask.to(dtype)

    def _create_chunk_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        chunk_size: int,
    ) -> torch.Tensor:

        # [NOTE]: batch_size, mel_seq값은 사용하지 않지만 굳이 만든 이유는 input되는 audio_features의 차원 값을 알려줄 수 있기 때문에 명시함
        batch_size, time_seq, mel_seq = input_shape
        chunk_mask = attention_mask[0].diag()
        mask = torch.ones([(chunk_size * 2), chunk_size])

        mask_x_dim = mask.shape[1]
        mask_y_dim = mask.shape[0]

        for mask_idx in range(0, time_seq, chunk_size):
            mask_x_pos = mask_idx + mask_x_dim
            mask_y_pos = mask_idx + mask_y_dim

            if mask_y_pos > time_seq:  # for y, 이 부분은 mask가 끝 부분과 맞지 않을 때 일부러 짤라내는 부분이다.
                truncate_size = mask_y_dim - (mask_y_pos - time_seq)
                mask = mask[:truncate_size, :]

            if mask_x_pos > time_seq:  # for x
                truncate_size = mask_x_dim - (mask_x_pos - time_seq)
                mask = mask[:, :truncate_size]

            chunk_mask[mask_idx:mask_y_pos, mask_idx:mask_x_pos] = mask

        # [TODO]: audio_size만큼 padding 하는 기능 추가

        chunk_attention_mask = torch.stack([chunk_mask for _ in range(batch_size)])
        return chunk_attention_mask

    def _create_diag_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        left_context: int = 0,
        right_context: int = 0,
    ) -> torch.Tensor:
        batch_size, time_seq, mel_seq = input_shape
        diag_mask = attention_mask[0].new_ones(time_seq, time_seq)

        right_mask = diag_mask.triu(right_context)
        left_mask = diag_mask.tril(-left_context)

        diag_mask = left_mask + right_mask

        # 모든 베치에 일괄적으로 적용한다.
        diag_attention_mask = torch.stack([diag_mask for _ in range(batch_size)])
        return diag_attention_mask

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[EncoderOutput, Tuple[Any]]:

        # [TODO]: attention_mask에 dtype 설정되도록 하기
        feature_shape = input_features.size()
        attention_mask = self._prepare_encoder_attention_mask(attention_mask, feature_shape)

        if True:
            time_seq = input_features.shape[1]
            position_vector = self.position_embeddings(self.position_ids[:time_seq])
            hidden_state = input_features + position_vector

            if attention_mask is not None:
                # pytorch transformer를 위해 만든 곳, pytorch transformer는 bsz * head_size임
                attention_mask = attention_mask.squeeze(1)
                attention_mask = attention_mask.repeat(self.head_size, 1, 1)
                # [NOTE]: if full attention일 경우
                # attention_mask = attention_mask.repeat(1, attention_mask.shape[2], 1)

            hidden_state = self.encoder(hidden_state, attention_mask.bool())
        else:
            for layer in self.layers:
                hidden_state = layer(hidden_state, attention_mask)

        if not return_dict:
            return None

        return EncoderOutput(hidden_state)


class TransducerJoiner(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.audio_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.label_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(config.intermediate_size, config.vocab_size)

    def forward(
        self,
        audio_hiddens: torch.Tensor,
        label_hiddens: torch.Tensor,
        return_dict: Optional[bool] = True,
    ) -> Union[JoinerOutput, Tuple[Any]]:

        # [TODO]: 나중에 concat test하기
        if audio_hiddens.dim() == 3 and label_hiddens.dim() == 3:
            audio_hiddens = audio_hiddens[:, :, None, :]
            label_hiddens = label_hiddens[:, None, :, :]
            # [NOTE]: huggingface의 attention_mask 생성에서 위와 같은 방법이 사용됨
            #         그리고 차원이 몇 차원인지 간접적으로 알려줄 수 있을 거라 생각해 사용함

        audio_hiddens = self.audio_linear(audio_hiddens)
        label_hiddens = self.label_linear(label_hiddens)

        concat_hiddens = audio_hiddens + label_hiddens
        concat_hiddens = self.tanh(concat_hiddens)
        concat_hiddens = self.dense(concat_hiddens)

        if not return_dict:
            return (audio_hiddens, label_hiddens, concat_hiddens)

        return JoinerOutput(
            logits=concat_hiddens,
            label_hidden=label_hiddens,
            audio_hidden=audio_hiddens,
        )


class TransducerModel(TransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super(TransducerPretrainedModel, self).__init__(config)
        self.config = config

        self.encoder = TransducerEncoder(config)
        self.decoder = TransducerDecoder(config)
        self.joiner = TransducerJoiner(config)

    def get_encoder(self) -> nn.Module:
        return self.encoder

    def get_decoder(self) -> nn.Module:
        return self.decoder

    def get_joiner(self) -> nn.Module:
        return self.joiner

    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TransducerBaseModelOutput, Tuple[Any]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_features,
            attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        audio_hiddens = encoder_outputs.last_hidden_state

        decoder_outputs = self.decoder(
            labels,
            decoder_attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        label_hiddens = decoder_outputs.last_hidden_state

        joiner_outputs = self.joiner(audio_hiddens, label_hiddens, return_dict)
        hidden_states = joiner_outputs.logtis

        if not return_dict:
            return (
                (hidden_states, audio_hiddens, label_hiddens)
                + encoder_outputs[1:]
                + decoder_outputs[1:]
                + joiner_outputs[1:]
            )
        return TransducerBaseModelOutput(
            logits=hidden_states,
            audio_last_hidden_state=audio_hiddens,
            label_last_hidden_state=label_hiddens,
            audio_attentions=None,
            label_attentions=None,
        )


class TransformerTranducerForRNNT(TransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.config = config

        self.transducer = TransducerModel(config)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.reduction = config.loss_reduction
        self.blank_id = config.blank_id
        self.loss_clamp = config.clamp

        self.dtype = torch.int32

        self.post_init()

    def get_encoder(self) -> nn.Module:
        return self.transducer.get_encoder()

    def get_decoder(self) -> nn.Module:
        return self.transducer.get_decoder()

    def get_joiner(self) -> nn.Module:
        return self.transducer.get_joiner()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        feature_lengths: Optional[torch.Tensor] = None,
    ) -> Union[RNNTBaseOutput, Tuple[Any]]:

        # [NOTE]: attention_mask가 3차원으로 들어와도 정상적으로 작동하도록 고려함
        if attention_mask is not None:
            if attention_mask.dim() == 3 and feature_lengths == None:
                # [TODO]: 나중에 영어로 수정할 것
                raise ValueError("attention_mask가 3차원 일때 무조건 feature_lengths가 입력되어야 합니다!")
        if decoder_attention_mask is not None:
            if decoder_attention_mask.dim() == 3 and label_lengths == None:
                raise ValueError("decoder_attention_mask가 3차원 일 때 무조건 label_length가 입력되어야 합니다!")

        transducer_outputs = self.transducer(
            input_features,
            labels,
            attention_mask,
            decoder_attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
        )
        logits = transducer_outputs.logits
        log_prob = self.log_softmax(logits)

        # [NOTE]: label에 붙어져 있는 blank token 제거
        non_blank_labels = labels[1:]

        # [XXX]: torchaudio의 rnn-t loss는 int32의 정수형만 받아들임
        #        이 if문은 부분은 나중에 수정될 수 있음
        feature_lengths = feature_lengths if feature_lengths else attention_mask.sum(-1, self.dtype)
        label_lengths = label_lengths if label_lengths else decoder_attention_mask.sum(-1, self.dtype)

        loss = rnnt_loss(
            logits=log_prob,
            targets=non_blank_labels,
            logit_lengths=feature_lengths,
            target_lengths=label_lengths,
            blank=self.blank_id,
            clamp=self.loss_clamp,
            reduction=self.reduction,
        )

        # [TODO]: beamsearch 테스트를 위해 넣어둔 코드, 나중에 삭제할 것!
        self.test_beam_search(transducer_outputs.audio_last_hidden_state[0])
        # [NOTE]: transformer-transducer는 메모리를 많이 차지하기 때문에 임시로 empty_cache를 진행함
        #         torch profiler를 이용해 값을 확인할 것
        torch.cuda.empty_cache()

        if not return_dict:
            return None

        return RNNTBaseOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        return {
            "input_features": input_features,
            "labels": labels,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
        }

    def greedy_search(
        self,
        input_features: torch.LongTensor,
        encoder_outputs: torch.FloatTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # ====================

        encoder_state = encoder_outputs.last_hidden_state
        blank_id = self.config.blank_id

        decoder = self.get_decoder()
        joiner = self.get_joiner()

        decoder_outputs = decoder(input_features)
        decoder_state = decoder_outputs.last_hidden_state[:, -1, :]

        state_iter = enumerate(zip(encoder_state, decoder_state))
        result_list = list()
        for batch_idx, (audio_logits, decoder_state) in state_iter:
            # decoder_outputs = decoder(start_token)
            # decoder_state = decoder_outputs.last_hidden_state[:, -1, :]
            time_seq = audio_logits.shape[0]
            reuslt_ids = input_features[batch_idx]

            for time_idx in range(time_seq):
                current_state = audio_logits[time_idx]
                # [NOTE]: just output 1d tensor
                joint_outputs = joiner(current_state.view(-1), decoder_state.view(-1))
                joint_state = joint_outputs.hidden_state

                log_prob = joint_state.log_softmax(dim=-1)
                token_id = log_prob.argmax(dim=-0)
                if blank_id not in token_id:
                    token_id = token_id.unsqueeze(0)
                    concat_frame = [reuslt_ids, token_id]
                    reuslt_ids = torch.concat(concat_frame, dim=-1)

                    decoder_outputs = decoder(reuslt_ids.unsqueeze(0)[:, 1:])
                    decoder_state = decoder_outputs.last_hidden_state[:, -1, :]
            max_length = 511
            pad = torch.zeros(max_length - len(reuslt_ids[:max_length]), device=reuslt_ids.device)
            reuslt_ids = torch.concat([reuslt_ids[:max_length], pad])

            result_list.append(reuslt_ids)

        return torch.stack(result_list)[:, 1:]

    def test_beam_search(self, encoder_state) -> None:
        beam_width = 5
        decoder = self.get_decoder()
        joiner = self.get_joiner()
        lengths = encoder_state.shape[0]

        first = True
        device = torch.device("cuda" if encoder_state.is_cuda else "cpu")

        token_list = []  # len=beam_width
        probability = np.zeros((beam_width,), dtype=float)
        token_child_list = []  # len=beam_width**2
        probability_child = np.zeros((beam_width, beam_width), dtype=float)

        # [NOTE]: beam_width 만큼의 list를 만든다음 그 리스트에 값들을 차례대로 넣어가며 예측한다.
        for i in range(beam_width):
            token_list.append([0])
        for i in range(beam_width):
            token_child_list.append([])
            for j in range(beam_width):
                token_child_list[i].append([0])

        # 1 배치 단위의 데이터 들어와야 함
        for t in range(lengths):
            max_index = probability.argmax()  # 첫 시작에는 0이 출력됨
            token = torch.tensor([token_list[max_index]], dtype=torch.long).to(device)  # max_index를 왜 출력하는지 이해안됨

            decoder_outputs = decoder(token)
            decoder_state = decoder_outputs.last_hidden_state[:, -1, :]

            joiner_outputs = joiner(encoder_state[t].view(-1), decoder_state.view(-1))
            logits = joiner_outputs.hidden_state
            out = logits.softmax(dim=0).detach()

            pred_max = torch.argmax(out, dim=0)
            pred_max = int(pred_max.item())

            if pred_max != 0:
                for token_index in range(len(token_list)):
                    token = torch.tensor([token_list[token_index]], dtype=torch.long).to(device)

                    decoder_outputs = decoder(token)
                    decoder_state = decoder_outputs.last_hidden_state[:, -1, :]

                    joiner_outputs = joiner(encoder_state[t].view(-1), decoder_state.view(-1))
                    logits = joiner_outputs.hidden_state
                    out = logits.softmax(dim=0).detach()  # 1차원 배

                    values, indices = torch.topk(out, k=beam_width + 1, dim=0)
                    values = values.tolist()
                    indices = indices.tolist()

                    if 0 in indices:  # 첫 시작 떄 token에 있는 blank 값을 삭제함.
                        zero_index = indices.index(0)
                        indices.pop(zero_index)
                        values.pop(zero_index)
                    else:
                        indices.pop(-1)
                        values.pop(-1)
                    if first:
                        for i in range(len(indices)):
                            token_child_list[i][token_index].append(indices[i])
                            probability_child[:, token_index] = np.log(values)
                    else:
                        for i in range(len(indices)):
                            token_child_list[token_index][i].append(indices[i])
                            probability_child[token_index] = probability[token_index] + np.log(values)
                if first:
                    first = False
                    for i in range(beam_width):
                        token_list[i] = copy.deepcopy(token_child_list[i][0])
                        probability[i] = copy.deepcopy(probability_child[i, 0])
                else:
                    top_k_index = heapq.nlargest(beam_width, range(beam_width**2), probability_child.take)
                    for i in range(len(top_k_index)):
                        index = top_k_index[i]
                        probability[i] = copy.deepcopy(probability_child[index // beam_width, index % beam_width])
                        token_list[i] = copy.deepcopy(token_child_list[index // beam_width][index % beam_width])

        max_index = probability.argmax()
        token_list = token_list[max_index]
        return token_list[1:]
