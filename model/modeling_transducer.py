import math
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.file_utils import is_torchaudio_available
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

if is_torchaudio_available():
    from torchaudio.transforms import FrequencyMasking, RNNTLoss, TimeMasking
else:
    raise ModuleNotFoundError(
        "Transformer-Transducer must need torchaudio, please intall torchaudio when you use transformer-transducer"
    )

logger = logging.get_logger(__name__)


TRANSFORMER_TRANSDUCER_MODEL_ARCHIVE_LIST = ["jp/transformer-transducer-librispeech"]


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


# [TODO]: it's must be changed huggingface outputs
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
class TransformerTransducerJoinerOutput(ModelOutput):
    logits: torch.Tensor
    encoder_hidden_states: Optional[torch.Tensor] = None
    decoder_hidden_states: Optional[torch.Tensor] = None


@dataclass
class TransducerBaseModelOutput(ModelOutput):
    logits: torch.Tensor

    encoder_hidden_states: Optional[torch.Tensor] = None
    encoder_attentions: Optional[torch.Tensor] = None

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


class TransformerTransducerPretrainedModel(PreTrainedModel):
    config_class = TransformerTransducerConfig
    base_model_prefix = "transformertransducer"
    main_input_name = "input_features"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = False
    # im not sure, model can support gradient_checkpointing. it's need expriments

    def _init_weights(self, module: nn.Module) -> None:
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

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, TransformerTransducerEncoderLayer):
            module.gradient_checkpointing = value


# [NOTE]: copied from BertSelfAttention
class TransformerTransducerAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, position_embedding_type: Optional[str] = None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scale = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.softmax = nn.Softmax(dim=-1)

        # attention_probs_dropout_prob > attention_dropout
        self.dropout = nn.Dropout(config.attention_dropout)
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
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_values is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_values[0]
            value_layer = past_key_values[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_values is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_values[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_values[1], value_layer], dim=2)
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
            # if encoder bi-directional self-attention `past_key_values` is always `None`
            past_key_values = (key_layer, value_layer)

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

        attention_scores = attention_scores / self.scale
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

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
        context_layer = self.out_proj(context_layer)

        # [XXX]: just copied from bart attention output,
        #        It is not intuitive to return to the tuple, so I modified it as below.
        #        it can be modified at any time depending on the situation.
        return context_layer, attention_probs, past_key_values


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

        self.self_attention = TransformerTransducerAttention(config)
        self.feed_forward = TransformerTransducerFeedForward(config)

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.cross_attention = TransformerTransducerAttention(config, position_embedding_type="absolute")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        self_head_mask: Optional[torch.Tensor] = None,
        cross_head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.Tensor]:
        # [NOTE]: when i mask encoder layer, i checked from BERT, BART, Wav2vec2 Encoder layer
        #         espectially cross attention refer to BERT and BART encoder layer
        self_attn_past_key_value = past_key_values[:2] if past_key_values is not None else None

        # self attention
        attention_residual = hidden_states
        hidden_states, attn_weights, past_key_values = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=self_head_mask,
            past_key_values=self_attn_past_key_value,
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = attention_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)

        # [XXX]: I am not sure, Cross-Attention implementation is proper for this model.
        #        because streaming models have many architecture
        #        it's always removed from this code
        present_key_value = None
        cross_attn_weights = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "cross_attention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
            present_key_value = past_key_values[-2:] if past_key_values is not None else None
            hidden_states, cross_attn_weights, present_key_value = self.cross_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=cross_head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=present_key_value,
            )
            hidden_states = self.dropout(hidden_states)
            hidden_states = attention_residual + hidden_states
            hidden_states = self.layer_norm(hidden_states)

            past_key_values = past_key_values + present_key_value

        # FFN
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class TransformerTransducerDecoder(TransformerTransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super(TransformerTransducerPretrainedModel, self).__init__(config)
        self.config = config
        self.layerdrop = config.decoder_layerdrop

        decoder_layers = [TransformerTransducerEncoderLayer(config) for _ in range(config.decoder_layers)]
        self.layers = nn.ModuleList(decoder_layers)

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
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cross_head_mask: Optional[torch.Tensor] = None,
        self_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[EncoderOutput, Tuple[Any]]:
        # [TODO]: change head mask name, cross, self head mask is not inappropriate name --------

        attentions_flag = output_attentions is not None
        output_attentions = output_attentions if attentions_flag else self.config.output_attentions

        output_hidden_flag = output_hidden_states is not None
        output_hidden_states = output_hidden_states if output_hidden_flag else self.config.output_hidden_states

        return_flag = return_dict is not None
        return_dict = return_dict if return_flag else self.config.use_return_dict

        # [XXX]: it's may need functionalization
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

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None
        if self_head_mask is not None:
            if self_head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {self_head_mask.size()[0]}."
                )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
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
                        (self_head_mask[idx] if self_head_mask is not None else None),
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        self_head_mask=(self_head_mask[idx] if self_head_mask is not None else None),
                        cross_head_mask=(cross_head_mask[idx] if cross_head_mask is not None else None),
                        output_attentions=output_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

                if output_attentions:
                    all_attentions += (layer_outputs[1],)

                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return DecoderOutput(
            last_hidden_states=hidden_states,
            decoder_attentions=all_attentions,
            decoder_hidden_states=all_hidden_states,
        )


class TransformerTransducerEncoder(TransformerTransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super(TransformerTransducerPretrainedModel, self).__init__(config)
        self.config = config
        self.attention_type = self.config.attention_type
        self.layerdrop = config.encoder_layerdrop
        self.gradient_checkpointing = False

        self.left_context = None
        self.right_context = None
        self.chunk = None

        encoder_layers = [TransformerTransducerEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layers = nn.ModuleList(encoder_layers)

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
        elif self.attention_type == "original_full":  # from BigBird Model
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

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
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
                        self_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return EncoderOutput(
            last_hidden_states=hidden_states,
            encoder_attentions=all_attentions,
            encoder_hidden_states=all_hidden_states,
        )


class TransformerTransducerJoiner(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.encoder_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.decoder_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(config.intermediate_size, config.vocab_size)

    def forward(
        self,
        encoder_hiddens: torch.Tensor,
        decoder_hiddens: torch.Tensor,
        return_dict: Optional[bool] = True,
    ) -> Union[TransformerTransducerJoinerOutput, Tuple[Any]]:
        if encoder_hiddens.dim() == 3 and decoder_hiddens.dim() == 3:
            encoder_hiddens = encoder_hiddens[:, :, None, :]
            decoder_hiddens = decoder_hiddens[:, None, :, :]

        encoder_hiddens = self.encoder_linear(encoder_hiddens)
        decoder_hiddens = self.decoder_linear(decoder_hiddens)

        # [TODO]: it need to test torch.concat instead of +
        joint_hiddens = encoder_hiddens + decoder_hiddens
        joint_hiddens = self.tanh(joint_hiddens)
        joint_hiddens = self.dense(joint_hiddens)

        if not return_dict:
            return (joint_hiddens, encoder_hiddens, decoder_hiddens)

        return TransformerTransducerJoinerOutput(
            logits=joint_hiddens,
            encoder_hidden_states=encoder_hiddens,
            decoder_hidden_states=decoder_hiddens,
        )


class TransformerTransducerModel(TransformerTransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super(TransformerTransducerPretrainedModel, self).__init__(config)
        self.config = config

        self.encoder = TransformerTransducerEncoder(config)
        self.decoder = TransformerTransducerDecoder(config)
        self.joiner = TransformerTransducerJoiner(config)

        self.freq_masking = FrequencyMasking(config.freq_mask_size)
        self.time_masking = TimeMasking(config.time_mask_size)

    def get_encoder(self) -> nn.Module:
        return self.encoder

    def get_decoder(self) -> nn.Module:
        return self.decoder

    def get_joiner(self) -> nn.Module:
        return self.joiner

    def spec_augment(self, hidden_state) -> torch.Tensor():
        for _ in range(self.config.freq_apply_num):
            hidden_state = self.freq_masking(hidden_state)

        for _ in range(self.config.time_apply_num):
            hidden_state = self.time_masking(hidden_state)

        return hidden_state

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
        # [XXX]: 단순 indentation이 발생하는 게 싫어서 flag를 선언함. 나중에 수정해도 무관함
        # 코드가 길면 보기가 힘들어서
        attentions_flag = output_attentions is not None
        output_attentions = output_attentions if attentions_flag else self.config.output_attentions

        output_hidden_flag = output_hidden_states is not None
        output_hidden_states = output_hidden_states if output_hidden_flag else self.config.output_hidden_states

        return_flag = return_dict is not None
        return_dict = return_dict if return_flag else self.config.use_return_dict

        if self.training:
            input_features = self.spec_augment(input_features)

        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            head_mask=head_mask,
            return_dict=return_dict,
        )
        encoder_hidden_states = encoder_outputs[0]

        decoder_outputs = self.decoder(
            labels=labels,
            attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            self_head_mask=decoder_head_mask,
            return_dict=return_dict,
        )
        decoder_hidden_states = decoder_outputs[0]

        joiner_outputs = self.joiner(encoder_hidden_states, decoder_hidden_states, return_dict)
        hidden_states = joiner_outputs.logits

        if not return_dict:
            return (
                (hidden_states, encoder_outputs[0], decoder_outputs[0])
                + encoder_outputs[1:]
                + decoder_outputs[1:]
                + joiner_outputs[1:]
            )
        # [XXX]: i don't know how make model_outputs,
        #        it's need to add encoder & decoder's last_hidden_states?
        return TransducerBaseModelOutput(
            logits=hidden_states,
            encoder_hidden_states=encoder_outputs[2],
            decoder_hidden_states=decoder_outputs[2],
            encoder_attentions=encoder_outputs[1],
            decoder_attentions=decoder_outputs[1],
        )


class TransformerTransducerForRNNT(TransformerTransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.config = config

        self.transducer = TransformerTransducerModel(config)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.rnnt_loss = RNNTLoss(
            blank=config.blk_token_id,
            clamp=config.clamp,
            reduction=config.loss_reduction,
        )
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
        # [BUG]: !!!!!!!!! it's didn't work at DP !!!!!!!!!!!
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

        non_blank_labels = labels[:, 1:].to(torch.int32)

        feature_lengths = feature_lengths if feature_lengths else attention_mask.sum(-1, dtype=torch.int32)
        label_lengths = label_lengths if label_lengths else decoder_attention_mask.sum(-1, dtype=torch.int32)
        label_lengths = label_lengths - 1

        loss = self.rnnt_loss(
            logits=log_prob,
            targets=non_blank_labels,
            target_lengths=label_lengths,
            logit_lengths=feature_lengths,
        )

        # [NOTE]: my testing environment is rtx1080 * 4, rtx1080's GPU memory is 12GB.
        #         RNN-T models have many space complexity(it's need large capacity)
        #         so i use cuda.empty_cache()
        torch.cuda.empty_cache()

        if not return_dict:
            return None

        # [TODO]: it's need to add more outputs, like encoder & decoder hidden_states, attentions, last_hidden_states
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

                repeat_check = repeat_count == self.config.generate_repeat_max
                if next_token == self.config.blk_token_id or repeat_check:
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
