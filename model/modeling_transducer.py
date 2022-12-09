import torch
from dataclasses import dataclass
import torch.nn as nn
from torchaudio.functional import rnnt_loss
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
import math
from typing import Optional, Tuple
from .config import TransformerTransducerConfig
import torch.distributed as dist
from transformers.trainer_utils import is_main_process
from typing import List, Union, Any, Dict
import numpy as np
from transformers import PretrainedConfig

# head_mask는 multi-head attention에서 head의 사용 여부를 결정하는 mask다.


@dataclass
class EncoderOutput(ModelOutput):
    last_hidden_state: torch.Tensor
    hidden_states: torch.Tensor = None
    attentions: torch.Tensor = None

    # or need audio encoder output, label encoder output


@dataclass
class JointOutput(ModelOutput):
    hidden_state: torch.Tensor


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
    main_input_name = "input_values"
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


class LabelEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super(LabelEncoder, self).__init__()
        self.config = config
        self.test = config.model_test
        if self.test:
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

        self.position_embedding = nn.Embedding(config.position_embed_size, config.hidden_size)
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_ids = torch.arange(config.position_embed_size)

    def forward(
        self,
        label_data: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_state: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> torch.Tensor:
        position_ids = self.position_ids[: label_data.shape[1]]
        position_ids = position_ids.to(label_data.device)
        position_embed = self.position_embedding(position_ids)
        word_embed = self.word_embedding(label_data)

        hidden_state = word_embed + position_embed

        if self.test:
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


class AudioEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super(AudioEncoder, self).__init__()
        self.config = config
        # self.position_embedding = PositionalEncoding(config.hidden_size)
        self.test = config.model_test
        if self.test:
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
            self.position_embedding = nn.Embedding(512, config.hidden_size)

        else:
            self.test_embedding = nn.Embedding(config.position_embed_size, config.hidden_size)
            self.test_ids = torch.arange(config.position_embed_size)

            self.linear = nn.Linear(80, config.hidden_size)

            encoder_layers = [TransducerEncoderLayer(config) for _ in range(config.audio_layers)]
            self.layers = nn.ModuleList(encoder_layers)

    def forward(
        self,
        audio_data: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_state: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> torch.Tensor:
        if self.test:
            time_seq = audio_data.shape[1]
            position_ids = torch.arange(512, device=audio_data.device)
            position_vector = self.position_embedding(position_ids[:time_seq])
            hidden_state = audio_data + position_vector

            if attention_mask is not None:
                attention_mask = attention_mask.repeat(self.head_size, 1, 1)
                attention_mask = attention_mask != 0

            hidden_state = self.encoder(hidden_state, attention_mask)
        else:
            for layer in self.layers:
                hidden_state = layer(hidden_state, attention_mask)
        if not return_dict:
            return None
        return EncoderOutput(hidden_state)


class JointNetwork(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.audio_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.label_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(config.intermediate_size, config.vocab_size)

    def forward(
        self,
        audio_vector: torch.Tensor,
        label_vector: torch.Tensor,
        return_dict: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        논문에서
        Joint = Linear(AudioEncoder_ti(X)) +
                Linear(LabelEncoder(Labels(z_1:(i-1))))
        와 같이 적혀져 있기 때문에 Linear로 합치는 부분 까지를 joint network로 지정한다.
        """
        if audio_vector.dim() == 3 and label_vector.dim() == 3
            audio_vector = audio_vector[:, :, None, :]
            label_vector = label_vector[:, None, :, :]
            # [NOTE]: 굳이 여길 [:, None, :, :]와 같이 한 이유
            #         더 직관적이기 때문에, joint_nec은 값을 4차원으로 확장시킨 다음에 그 값을 합치는 역할을 수행한다.
            #         그렇기 깨문에 계산에 수행되는 값은 4차원 이라는 것을 간접적으로 알려주기 위해서 위와 같은 방법을 사용했다.
            #         물론 .unsqueeze와 같은 방법이 있지만 그러면 차이 어느정도 인지 확인할 수 없기에 이 방법은 제외했다.

        audio_vector = self.audio_linear(audio_vector)
        label_vector = self.label_linear(label_vector)

        concat_vector = audio_vector + label_vector
        concat_vector = self.tanh(concat_vector)
        concat_vector = self.dense(concat_vector)
        if not return_dict:
            return None
        return JointOutput(concat_vector)


class TransducerModel(TransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig) -> None:
        super(TransducerPretrainedModel, self).__init__(config)
        self.audio_encoder = AudioEncoder(config)
        self.label_encoder = LabelEncoder(config)
        self.joint_network = JointNetwork(config)
        self.config = config

    def forward(
        self,
        audios: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        label_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_state: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> torch.Tensor:
        # audio_shape = audios.size()
        # audio_attention_mask = self.get_extended_attention_mask(audio_attention_mask, audio_shape)

        label_shape = labels.size()
        label_attention_mask = self.create_extended_attention_mask_for_decoder(label_shape, label_attention_mask)

        audio_outputs = self.audio_encoder(
            audios,
            audio_attention_mask,
            output_attentions,
            output_hidden_state,
            return_dict,
        )
        audio_hiddens = audio_outputs.last_hidden_state

        label_outputs = self.label_encoder(
            labels,
            label_attention_mask,
            output_attentions,
            output_hidden_state,
            return_dict,
        )
        label_hiddens = label_outputs.last_hidden_state

        joint_outputs = self.joint_network(audio_hiddens, label_hiddens, return_dict)
        hidden_states = joint_outputs.hidden_state

        if not self.training and is_main_process(dist.get_rank()):
            print(self.inference_test_2(audio_outputs.last_hidden_state[0]))

        if not return_dict:
            return (hidden_states, audio_hiddens, label_hiddens) + audio_outputs[1:] + label_outputs[1:]

        # [BUG]: audio_hiddens를 출력할 때 차원이 [:, :, None, :]된 상태의 hidden_state값이 출려된다. 수정할 것!
        #        수정을 해야 하는 이유는 저 audio_hiddens의 값을 valid시 사용할 것이기 때문에

        return TransducerBaseModelOutput(logits=hidden_states, audio_last_hidden_state=audio_hiddens)

    def get_chunk_attetion_mask(self) -> torch.Tensor:
        return

    @torch.no_grad()
    def inference_test_2(self, audio_input: torch.Tensor) -> None:
        token_list = [self.config.blank_id]
        blank_token = torch.tensor([token_list], device=self.device)
        dec_state = self.label_encoder(blank_token)
        dec_state = dec_state.last_hidden_state[:, -1, :]
        lengths = audio_input.shape[0]

        for t in range(lengths):
            joint_outputs = self.joint_network(audio_input[t].view(-1), dec_state.view(-1))
            hidden_state = joint_outputs.hidden_state
            logits = hidden_state.log_softmax(dim=-1)

            pred = logits.argmax(dim=0)
            pred = int(pred.item())

            if pred != self.config.blank_id:
                token_list.append(pred)
                token = torch.tensor([token_list], dtype=torch.long)

                if audio_input.is_cuda:
                    token = token.cuda()
                dec_state = self.label_encoder(token)
                dec_state = dec_state.last_hidden_state[:, -1, :]
        return token_list[1:]


class TransformerTranducerForRNNT(TransducerPretrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config

        self.transducer = TransducerModel(config)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.reduction = config.loss_reduction
        self.blank_id = config.blank_id

        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None,
        label_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_state: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # [BUG]: pad되어 있는 부분에는 chunk_size attention_mask를 적용시키면 안될거 같은데?
        input_length = audio_attention_mask.sum(-1, dtype=torch.int32)
        label_len = (labels != 0).sum(dim=-1) - 1

        attention_result = list()
        for audio in audio_attention_mask:
            time_seq = audio.shape[0]
            mask = np.ones([time_seq, time_seq])
            up = np.triu(mask, k=2 + 1)
            down = np.tril(mask, k=-10 - 1)
            attention_mask = up + down
            attention_result.append(attention_mask)

        audio_attention_mask = torch.tensor(attention_result, device=self.device)

        transducer_outputs = self.transducer(
            input_features,
            labels,
            audio_attention_mask,
            label_attention_mask,
            output_attentions,
            output_hidden_state,
            return_dict,
        )
        logits = transducer_outputs.logits
        log_prob = self.log_softmax(logits)

        # [TODO]: 길이 구하는 기능들 클린 코드 할 것
        # [BUG]: input_length는 실제 mel의 길이값이 들어가야함. 하지만 여긴 padding된 일정한 길이값이 들어가기 때문에 loss가 정상적으로 떨어지지 않음!!
        non_blank_labels = torch.stack([tensor[1:] for tensor in labels]).to(torch.int32)

        loss = rnnt_loss(
            logits=log_prob,
            targets=non_blank_labels,
            logit_lengths=input_length,
            target_lengths=label_len.to(torch.int32),
            blank=self.blank_id,
            clamp=-1,
            reduction=self.reduction,
        )

        torch.cuda.empty_cache()
        return RNNTBaseOutput(loss=loss, logits=logits)

    def get_audio_length(self, audios, labels) -> None:
        return

    @torch.no_grad()
    def test_greedy_search(self, audio_input: torch.Tensor, audio_mask: torch.Tensor) -> None:
        self._audio_encoder
        self._label_encoder
        self._joint_network
        
        predict_list = list()

        return

    def inference_test_2(self, audio_input: torch.Tensor) -> None:
        token_list = [self.config.blank_id]
        blank_token = torch.tensor([token_list], device=self.device)
        dec_state = self.label_encoder(blank_token)
        dec_state = dec_state.last_hidden_state[:, -1, :]
        lengths = audio_input.shape[0]

        for t in range(lengths):
            joint_outputs = self.joint_network(audio_input[t].view(-1), dec_state.view(-1))
            hidden_state = joint_outputs.hidden_state
            logits = hidden_state.log_softmax(dim=-1)

            pred = logits.argmax(dim=0)
            pred = int(pred.item())

            if pred != self.config.blank_id:
                token_list.append(pred)
                token = torch.tensor([token_list], dtype=torch.long)

                if audio_input.is_cuda:
                    token = token.cuda()
                dec_state = self.label_encoder(token)
                dec_state = dec_state.last_hidden_state[:, -1, :]
        return token_list[1:]

    @property
    def _audio_encoder(self, *args, **kwargs) -> ModelOutput:
        # or another method is self.transducer.audio_encoder
        # 그냥 길어지는게 싫어서 이렇게 만들었다.
        return self.transducer.audio_encoder(*args, **kwargs)

    @property
    def _label_encoder(self, *args, **kwargs) -> ModelOutput:
        return self.transducer.label_encoder(*args, **kwargs)

    @property
    def _joint_network(self, *args, **kwargs) -> ModelOutput:
        return self.transducer.joint_network(*args, **kwargs)
