import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import rnnt_loss
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation_utils import (
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
    GreedySearchOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .config import TransformerTransducerConfig

logger = logging.get_logger(__name__)


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
class DecoderOutput(ModelOutput):
    last_hidden_states: torch.Tensor
    decoder_attentions: Optional[torch.Tensor] = None
    decoder_hidden_states: Optional[torch.Tensor] = None


@dataclass
class EncoderOutput(ModelOutput):
    last_hidden_states: torch.Tensor
    encoder_attentions: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None


@dataclass
class JoinerOutput(ModelOutput):
    logits: torch.Tensor
    encoder_hidden_states: Optional[torch.Tensor] = None
    decoder_hidden_states: Optional[torch.Tensor] = None


@dataclass
class TransducerBaseModelOutput(ModelOutput):
    logits: torch.Tensor

    encoder_last_hidden_states: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    encoder_attentions: Optional[torch.Tensor] = None

    decoder_last_hidden_states: Optional[torch.Tensor] = None
    decoder_hidden_states: Optional[torch.Tensor] = None
    decoder_attentions: Optional[torch.Tensor] = None


@dataclass
class RNNTBaseOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    encoder_attentions: Optional[torch.Tensor] = None
    decoder_attentions: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    decoder_hidden_states: Optional[torch.Tensor] = None


# =====================================================


class TransformerTransducerPretrainedModel(PreTrainedModel):
    config_class = TransformerTransducerConfig
    base_model_prefix = "transformertransducer"
    main_input_name = "input_features"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module) -> None:
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

    def _set_gradient_checkpointing(self, module, value=False) -> None:
        if isinstance(module, TransformerTransducerEncoderLayer):
            module.gradient_checkpointing = value


class TransformerTransducerSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, position_embedding_type=None) -> None:
        super().__init__()
        # [NOTE]: BART attention layer를 참고해서 만들었다.
        #         이유: Wav2Vec2도 BART attention을 참고해서 만들었기 때문에
        self.num_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        # [NOTE]: 일반 BART attention module은 Linear layer의 bias 설정 여부를 매개변수로 설정할 수 있도록 만들어 놨었음
        #         왜 그렇게 만들었는지는 모르겠지만 Transformer 특정상 bias는 무조건 필요하다고 하기 때문에 bias는 삭제함.
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(config.attention_dropout)

        # [XXX]]: bert의 positional encoding을 추가시킴, 하지만 이 positional encoding이 어떤 영향을 미칠 지 모르기 때문에 테스트가 필요함
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.is_decoder = config.is_decoder

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # [NOTE]: BART Attention에 있던 Prefix tuning을 위한 구문은 삭제함.
        #         streming모델에서 사용할 가능성이 보이지 않았기 때문,

        #         decoder에서는 사용될 수 있나??? 그건 잘 모르겠다. 다만
        #         prefix-tuning의 특징 상 streming에 적용시키기 어려울 듯

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # [TODO]: Transducer 계열의 모델들은 encoder, decoder를 다양하게 조합해서 진행하기 때문에
        #         이후 cross-attention이 사용될 가능성이 있음. 나중에 Cross-attention이 가능하도록 코드를 작성할 것
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # [XXX]: Bert positional_embedding, 테스트 해보고 성능에 긍정적인 영향을 미치면 bert로 갈아탈 것
        use_cache = past_key_value is not None
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_states.shape[2], key_states.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_states.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_states, positional_embedding)
                attention_scores = attn_weights + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_states, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_states, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if head_mask is not None:
            if head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is" f" {head_mask.size()}"
                )
            attn_weights = head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        # [NOTE]: 이 층은 아래의 self.dropout으로 바뀜
        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_probs = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class TransformerTransducerFeedForward(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        # [NOTE]: Wav2Vec2를 참고해서 만들었음.
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        self.ffn_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.ffn_dense(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)

        return hidden_states


class TransformerTransducerEncoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.attention = TransformerTransducerSelfAttention(config)
        self.feed_forward = TransformerTransducerFeedForward(config)

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # [NOTE}: wav2vec2 + bert encoder layer를 참고해서 만들었음.
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        attention_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        hidden_states = hidden_states[0]

        hidden_states = self.dropout(hidden_states)
        hidden_states = attention_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# ======================================================


class TransformerTransducerDecoder(TransformerTransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super(TransformerTransducerPretrainedModel, self).__init__(config)
        self.config = config
        self.is_decoder = True

        if False:
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
            self.encoder = nn.TransformerEncoder(encoder_layer, config.decoder_layers)
            self.head_size = config.num_attention_heads
        else:
            decoder_layers = [TransformerTransducerEncoderLayer(config) for _ in range(config.decoder_layers)]
            self.layers = nn.ModuleList(decoder_layers)
            self.layerdrop = config.decoder_layerdrop
        # [TODO]: absolute, relative positional embedding 구현하기
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))  # from bert
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _prepare_decoder_attention_mask(
        self,
        input_shape: Tuple[int],
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # [NOTE]: 원래 attention_mask는 Optional이 아니였음.
        #         무조건 값이 들어오도록 만들었어야 했을 것 같지만
        #         `attention_mask is not None` 로 된 구문을 봤을 때 단순 버그라 생각된다.
        #         _prepare_decoder_attention_mask는 generate시 생성되는 문장 만큼 attention_mask 생성하기 위해
        #         attention_mask가 들어오지 않는 경우에 mask를 생성하는 기능을 추가해 놓은 듯 하다.

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
        head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[EncoderOutput, Tuple[Any]]:

        attentions_flag = output_attentions is not None
        output_attentions = output_attentions if attentions_flag else self.config.output_attentions

        output_hidden_flag = output_hidden_states is not None
        output_hidden_states = output_hidden_states if output_hidden_flag else self.config.output_hidden_states

        return_flag = return_dict is not None
        return_dict = return_dict if return_flag else self.config.use_return_dict

        # [XXX]: 나중에 따로 모듈화 해야할 수도 있음
        seq_length = labels.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        position_embed = self.position_embeddings(position_ids)
        word_embed = self.word_embedding(labels)

        hidden_states = word_embed + position_embed

        label_shape = labels.shape
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=label_shape,
            inputs_embeds=position_embed,
        )

        if False:
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze(1)
                attention_mask = attention_mask.repeat(self.head_size, 1, 1)
                attention_mask = attention_mask.bool()
            hidden_states = self.encoder(hidden_states, attention_mask)
            all_attentions = None
            encoder_states = None
        else:
            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = () if use_cache else None

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            for idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                    layer_outputs = (None, None)
                else:
                    if self.gradient_checkpointing and self.training:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(decoder_layer),
                            hidden_states,
                            attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                        )
                    else:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask,
                            head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                        )

                    hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return DecoderOutput(
            last_hidden_states=hidden_states,
            decoder_attentions=all_attentions,
            decoder_hidden_states=None,
        )


class TransformerTransducerEncoder(TransformerTransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super(TransformerTransducerPretrainedModel, self).__init__(config)
        self.config = config
        self.attention_type = self.config.attention_type

        self.left_context = None
        self.right_context = None
        self.chunk = None

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
            self.encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
            self.head_size = config.num_attention_heads
        else:
            config.position_embedding_type = "no"
            encoder_layers = [TransformerTransducerEncoderLayer(config) for _ in range(config.encoder_layers)]
            self.layers = nn.ModuleList(encoder_layers)
            self.layerdrop = config.encoder_layerdrop
            self.gradient_checkpointing = False
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))  # from bert
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.post_init()

    def _prepare_encoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        dtype: torch.float = None,
    ) -> torch.Tensor:
        # [NOTE]: opt model의 _prepare_decoder_attention_mask 에서 따왔다.
        #         huggingface의 encoder, deocder형식을 가진 seq2seq모델의 경우 model이 아닌 decoder에 casual mask를 생성하는
        #         _prepare_decoder_attention_mask가 존재함. 하지만 Transducer의 경우 Encoder에서 별도의 mask를 생성해야 하기 때문에
        #         _prepare_encoder_attention_mask로 이름을 바꿔서 TransducerEncoder에 집어넣었음

        if dtype is None:
            dtype = self.dtype

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
            return extended_attention_mask.to(dtype)

        if self.attention_type == "chunk-wise":
            extended_attention_mask = self._create_chunk_attention_mask(
                attention_mask,
                input_shape,
                chunk_size=3,
            )
        elif self.attention_type == "diagonal":
            extended_attention_mask = self._create_diag_attention_mask(
                attention_mask,
                input_shape,
                left_context=10,
                right_context=3,
            )
            # [TODO]: 나중에 huggingface encoder 사용할 때 제거할 것!!!
            # extended_attention_mask = extended_attention_mask == 0
        elif self.attention_type == "original_full":  # from BigBird Model
            # extended_attention_mask[:, None, :, :] 여기에서 차원이 추가되서 4차원이 됨
            extended_attention_mask = attention_mask[:, None, :]
        else:
            # [TODO]: 나중에 영어로 작성할 것
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
        head_mask: Optional[torch.Tensor] = None,
    ) -> Union[EncoderOutput, Tuple[Any]]:

        attentions_flag = output_attentions is not None
        output_attentions = output_attentions if attentions_flag else self.config.output_attentions

        output_hidden_flag = output_hidden_states is not None
        output_hidden_states = output_hidden_states if output_hidden_flag else self.config.output_hidden_states

        return_flag = return_dict is not None
        return_dict = return_dict if return_flag else self.config.use_return_dict

        # [TODO]: attention_mask에 dtype 설정되도록 하기
        feature_shape = input_features.size()
        attention_mask = self._prepare_encoder_attention_mask(attention_mask, feature_shape)

        time_seq = input_features.shape[1]
        position_ids = self.position_ids[:, :time_seq]
        position_vector = self.position_embeddings(position_ids)
        hidden_states = input_features + position_vector

        if True:

            if attention_mask is not None:
                # pytorch transformer를 위해 만든 곳, pytorch transformer는 bsz * head_size임
                attention_mask = attention_mask.squeeze(1)
                attention_mask = attention_mask.repeat(self.head_size, 1, 1)
                # [NOTE]: if full attention일 경우
                # attention_mask = attention_mask.repeat(1, attention_mask.shape[2], 1)

            hidden_states = self.encoder(hidden_states, attention_mask.bool())
            attentions = None
        else:
            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None
            if head_mask is not None:
                if head_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                    layer_outputs = (None, None)
                else:
                    if self.gradient_checkpointing and self.training:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                        )
                    else:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                        )

                    hidden_states = layer_outputs[0]

                if output_attentions:
                    attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return None

        return EncoderOutput(
            last_hidden_states=hidden_states,
            encoder_attentions=attentions,
            encoder_hidden_states=None,
        )


class TransformerTransducerJoiner(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.audio_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.label_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(config.intermediate_size, config.vocab_size)

    def forward(
        self,
        encoder_hiddens: torch.Tensor,
        decoder_hiddens: torch.Tensor,
        return_dict: Optional[bool] = True,
    ) -> Union[JoinerOutput, Tuple[Any]]:

        # [TODO]: 나중에 concat test하기
        if encoder_hiddens.dim() == 3 and decoder_hiddens.dim() == 3:
            encoder_hiddens = encoder_hiddens[:, :, None, :]
            decoder_hiddens = decoder_hiddens[:, None, :, :]
            # [NOTE]: huggingface의 attention_mask 생성에서 위와 같은 방법이 사용됨
            #         그리고 차원이 몇 차원인지 간접적으로 알려줄 수 있을 거라 생각해 사용함

        encoder_hiddens = self.audio_linear(encoder_hiddens)
        decoder_hiddens = self.label_linear(decoder_hiddens)

        joint_hiddens = encoder_hiddens + decoder_hiddens
        joint_hiddens = self.tanh(joint_hiddens)
        joint_hiddens = self.dense(joint_hiddens)

        if not return_dict:
            return (encoder_hiddens, decoder_hiddens, joint_hiddens)

        return JoinerOutput(
            logits=joint_hiddens,
            encoder_hidden_states=encoder_hiddens,
            decoder_hidden_states=decoder_hiddens,
        )


class TransducerModel(TransformerTransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super(TransformerTransducerPretrainedModel, self).__init__(config)
        self.config = config

        self.encoder = TransformerTransducerEncoder(config)
        self.decoder = TransformerTransducerDecoder(config)
        self.joiner = TransformerTransducerJoiner(config)

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
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TransducerBaseModelOutput, Tuple[Any]]:
        """
        Transducer는 Seq2Seq와 비슷한 모델이지만 encoder, decoder가 Cross-Attention이 아닌 Self-attention으로 작동함.
        이후 joiner가 cross attention과 비슷한 역할을 수행함. 이는 inference에서도 동일하게 동작함.
        """

        # [XXX]: 단순 indentation이 발생하는 게 싫어서 flag를 선언함. 나중에 수정해도 무관함
        attentions_flag = output_attentions is not None
        output_attentions = output_attentions if attentions_flag else self.config.output_attentions

        output_hidden_flag = output_hidden_states is not None
        output_hidden_states = output_hidden_states if output_hidden_flag else self.config.output_hidden_states

        return_flag = return_dict is not None
        return_dict = return_dict if return_flag else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_features,
            attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        encoder_hiddens = encoder_outputs.last_hidden_states

        decoder_outputs = self.decoder(
            labels,
            decoder_attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        decoder_hiddens = decoder_outputs.last_hidden_states

        joiner_outputs = self.joiner(encoder_hiddens, decoder_hiddens, return_dict)
        hidden_states = joiner_outputs.logits

        if not return_dict:
            return (
                (hidden_states, encoder_hiddens, decoder_hiddens)
                + encoder_outputs[1:]
                + decoder_outputs[1:]
                + joiner_outputs[1:]
            )
        return TransducerBaseModelOutput(
            logits=hidden_states,
            encoder_last_hidden_states=encoder_hiddens,
            encoder_hidden_states=None,
            encoder_attentions=None,
            decoder_attentions=None,
            decoder_last_hidden_states=decoder_hiddens,
            decoder_hidden_states=None,
        )


class TransformerTranducerForRNNT(TransformerTransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.config = config

        self.transducer = TransducerModel(config)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.reduction = config.loss_reduction
        self.blank_id = config.blank_id
        self.loss_clamp = config.clamp

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
        head_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
        feature_lengths: Optional[torch.Tensor] = None,
    ) -> Union[RNNTBaseOutput, Tuple[Any]]:

        # [BUG]: DP에서는 동작할 수 없음. 아마 attention_mask단위로 계산하는 것 때문에 발생하는 문제인 듯 함.
        #        max_length문제는 나중에 해결할 것

        # [NOTE]: attention_mask가 3차원으로 들어와도 정상적으로 작동하도록 고려함
        if attention_mask is not None:
            if attention_mask.dim() == 3 and feature_lengths is None:
                # [TODO]: 나중에 영어로 수정할 것
                raise ValueError("attention_mask가 3차원 일때 무조건 feature_lengths가 입력되어야 합니다!")
        if decoder_attention_mask is not None:
            if decoder_attention_mask.dim() == 3 and label_lengths is None:
                raise ValueError("decoder_attention_mask가 3차원 일 때 무조건 label_length가 입력되어야 합니다!")

        transducer_outputs = self.transducer(
            input_features=input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        logits = transducer_outputs.logits
        log_prob = self.log_softmax(logits)

        # [NOTE]: label에 붙어져 있는 blank token 제거
        non_blank_labels = labels[:, 1:]
        # [XXX]: torchaudio의 rnn-t loss는 int32의 정수형만 받아들임
        #        이 if문은 부분은 나중에 수정될 수 있음
        feature_lengths = feature_lengths if feature_lengths else attention_mask.sum(-1, dtype=torch.int32)
        label_lengths = label_lengths if label_lengths else decoder_attention_mask.sum(-1, dtype=torch.int32)
        label_lengths = label_lengths - 1

        non_blank_labels = non_blank_labels.to(torch.int32)

        loss = rnnt_loss(
            logits=log_prob,
            targets=non_blank_labels,
            target_lengths=label_lengths,
            logit_lengths=feature_lengths,
            blank=self.blank_id,
            clamp=self.loss_clamp,
            reduction=self.reduction,
        )

        # [TODO]: beamsearch 테스트를 위해 넣어둔 코드, 나중에 삭제할 것!
        # self.test_beam_search(transducer_outputs.audio_last_hidden_state[0])
        # [NOTE]: transformer-transducer는 메모리를 많이 차지하기 때문에 임시로 empty_cache를 진행함
        #         torch profiler를 이용해 값을 확인할 것
        # torch.cuda.empty_cache()

        if not return_dict:
            return None

        return RNNTBaseOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # 이 부분은 generate에서 어떻게 동작하는지 확인할 필요가 있음
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:

        # [TODO]: input_features라는 이름은 나중에 바꿀 것

        # [NOTE]: logits_processor
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

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

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_features.new(input_features.shape[0]).fill_(1)

        this_peer_finished = False  # used by synced_gpus only

        # ====================

        decoder = self.get_decoder()
        joiner = self.get_joiner()

        decoder_outputs = decoder(input_features)
        decoder_state = decoder_outputs.last_hidden_states
        encoder_state = encoder_outputs.last_hidden_states
        feature_length = encoder_state.shape[1] - 1
        # [NOTE]: 모든 index는 0번에서 시작하기 때문에 index에 맞추기 위해서는 1을 제거할 필요가 있다.
        repeat_count = 0
        decoding_list = list()
        state_iter = enumerate(zip(encoder_state, decoder_state))
        for batch_idx, (audio_logits, decoder_state) in state_iter:
            time_idx = 0
            repeat_max = 5
            gen_sentence = input_features[batch_idx]

            while True:
                if synced_gpus:
                    # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                    # The following logic allows an early break if all peers finished generating their sequence
                    this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0, device=self.device)
                    # send 0.0 if we finished, 1.0 otherwise
                    dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                    # did all peers finish? the reduced sum will be 0.0 then
                    if this_peer_finished_flag.item() == 0.0:
                        break

                if time_idx == feature_length or len(gen_sentence) == 512:
                    break

                current_state = audio_logits[time_idx].view(-1)
                decoder_state = decoder_state.view(-1)

                joiner_outputs = joiner(current_state, decoder_state)
                next_token_logits = joiner_outputs.logits

                next_tokens_scores = logits_processor(gen_sentence, next_token_logits)
                next_tokens_scores = torch.log_softmax(next_tokens_scores, dim=-1)

                next_token = torch.argmax(next_tokens_scores)
                next_token_score = next_tokens_scores[next_token]

                if next_token == self.blank_id or repeat_count == repeat_max:
                    time_idx += 1
                    repeat_count = 0
                    continue

                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_score,)
                    if output_attentions:
                        decoder_attentions += (
                            (decoder_outputs.decoder_attentions,)
                            if self.config.is_encoder_decoder
                            else (decoder_outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (decoder_outputs.cross_attentions,)
                    if output_hidden_states:
                        decoder_hidden_states += (
                            (decoder_outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (decoder_outputs.hidden_states,)
                        )

                gen_sentence = torch.cat([gen_sentence, next_token[None]], dim=-1)

                decoder_outputs = decoder(gen_sentence[1:].unsqueeze(0))
                decoder_state = decoder_outputs.last_hidden_states[:, -1, :]
                repeat_count += 1
            decoding_list.append(gen_sentence)

        # [NOTE]: 일반적인 generate의 greedy search의 경우 model이 batch_size만큼 하나씩 예측해 나가기 때문에 cocnat및 pad문제에서 자유롭다.
        #         하지만 streaming 모델의 경우 위와 같은 방법으로 예측할 수 없기 때문에 별도로 pad를 붙여줘야 한다.
        inner_max_length = max([len(sentence) for sentence in decoding_list])
        pad_shape = (0, inner_max_length)
        # [NOTE]: 계산의 간소화를 위해 슬라이싱을 이용해 잘라내도록 한다.
        decoding_list = [F.pad(tensor, pad_shape)[:inner_max_length] for tensor in decoding_list]
        input_features = torch.stack(decoding_list)

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_features,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_features,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_features
