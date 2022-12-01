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

from typing import List, Union, Any, Dict

# head_mask는 multi-head attention에서 head의 사용 여부를 결정하는 mask다.


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[List[List[torch.Tensor]], List[torch.Tensor]]
    y: List[torch.tensor] = None
    lm_state: Union[Dict[str, Any], List[Any]] = None
    lm_scores: torch.Tensor = None


class TransducerOuput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None


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
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        attn_residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states = self.attention(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        hidden_states = self.dropout(hidden_states[0])
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
        self.layers = nn.ModuleList(encoder_layers)

    def forward(
        self,
        label_data: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_ids = self.position_ids[: label_data.shape[1]]
        position_ids = position_ids.to(label_data.device)
        position_embed = self.position_embedding(position_ids)
        word_embed = self.word_embedding(label_data)

        hidden_state = word_embed + position_embed

        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask)

        return hidden_state


class AudioEncoder(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super(AudioEncoder, self).__init__()
        # self.position_embedding = PositionalEncoding(config.hidden_size)

        self.test_embedding = nn.Embedding(config.position_embed_size, config.hidden_size)
        self.test_ids = torch.arange(config.position_embed_size)

        self.linear = nn.Linear(80, config.hidden_size)

        encoder_layers = [TransducerEncoderLayer(config) for _ in range(config.audio_layers)]
        self.layers = nn.ModuleList(encoder_layers)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # audio_inputs = audio_inputs.transpose(1, 2)
        # hidden_state = self.linear(audio_inputs)

        for layer in self.layers:
            hidden_state = layer(hidden_state, attention_mask)

        return hidden_state


class JointNetwork(nn.Module):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super().__init__()
        # [TODO]: 여기 중에서 하나를 선택하기!
        # [NOTE]: case_1
        concat_size = config.hidden_size * 2
        self.dense = nn.Linear(concat_size, config.hidden_size)

        self.audio_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.label_linear = nn.Linear(config.hidden_size, config.intermediate_size)

        # [TODO]: test와 실제 코드와 비교해 결과가 비슷한지 확인하기!
        self.temp = False

    def forward(
        self,
        audio_output: torch.Tensor,
        label_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        논문에서
        Joint = Linear(AudioEncoder_ti(X)) +
                Linear(LabelEncoder(Labels(z_1:(i-1))))
        와 같이 적혀져 있기 때문에 Linear로 합치는 부분 까지를 joint network로 지정한다.
        """

        if self.temp:
            audio_len = audio_output.shape[1]
            label_len = label_output.shape[1]

            audio_output = audio_output.unsqueeze(2)
            label_output = label_output.unsqueeze(1)
            # [NOTE]: logits (Tensor) – Tensor of dimension (batch, max seq length, max target ""length + 1"", class) containing output from joiner
            audio_output = audio_output.repeat(1, 1, label_len, 1)
            label_output = label_output.repeat(1, audio_len, 1, 1)

            hidden_state = torch.cat([audio_output, label_output], dim=-1)
            hidden_state = self.dense(hidden_state)

        else:
            test_1 = audio_output[:, :, None, :]
            test_2 = label_output[:, None, :, :]

            test_1 = self.audio_linear(test_1)
            test_2 = self.label_linear(test_2)

            hidden_state = test_1 + test_2

        return hidden_state


class TransducerModel(TransducerPretrainedModel):
    def __init__(self, config: TransformerTransducerConfig) -> None:
        super(TransducerPretrainedModel, self).__init__(config)
        self.audio_encoder = AudioEncoder(config)
        self.label_encoder = LabelEncoder(config)
        self.joint_network = JointNetwork(config)

        self.tanh = nn.Tanh()
        self.dense = nn.Linear(config.intermediate_size, config.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.after_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        audios: Optional[torch.Tensor],
        audio_attention_mask: Optional[torch.Tensor] = None,
        label_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_state: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        audio_shape = audios.size()
        audio_attention_mask = self.get_extended_attention_mask(audio_attention_mask, audio_shape)

        label_shape = labels.size()
        label_attention_mask = self.create_extended_attention_mask_for_decoder(label_shape, label_attention_mask)
        audio_outputs = self.audio_encoder(audios, audio_attention_mask)
        label_outputs = self.label_encoder(labels, label_attention_mask)
        joint_outputs = self.joint_network(audio_outputs, label_outputs)

        hidden_state = self.tanh(joint_outputs)
        hidden_state = self.dense(hidden_state)
        logits = self.log_softmax(hidden_state)

        return logits

    def inference_test(self, audio_input: torch.Tensor, audio_attention_mask: torch.Tensor) -> None:
        audio_shape = audio_input.size()
        audio_attention_mask = self.get_extended_attention_mask(audio_attention_mask, audio_shape)
        audio_hidden = self.audio_encoder(audio_input, audio_attention_mask)

        self.blank = 2

        dec_state = [None] * len(self.label_encoder.layers)
        hyp = Hypothesis(score=0.0, yseq=[self.blank], dec_state=dec_state)
        cache = {}

        y, state, _ = self.score(hyp, cache, audio_hidden)

        return

    def check_state(self, state, max_len, pad_token):
        """Left pad or trim state according to max_len.

        test code

        Args:
            state (list): list of L decoder states (in_len, dec_dim)
            max_len (int): maximum length authorized
            pad_token (int): padding token id

        Returns:
            final (list): list of L padded decoder states (1, max_len, dec_dim)

        """
        if state is None or max_len < 1 or state[0].size(1) == max_len:
            return state

        # decoder_state의 seq길이를 측정
        curr_len = state[0].size(1)  # label일 가능성이 높음.

        if curr_len > max_len:
            trim_val = int(state[0].size(1) - max_len)

            for i, s in enumerate(state):
                state[i] = s[:, trim_val:, :]
        else:
            layers = len(state)
            ddim = state[0].size(2)

            final_dims = (1, max_len, ddim)
            final = [state[0].data.new(*final_dims).fill_(pad_token) for _ in range(layers)]

            for i, s in enumerate(state):
                final[i][:, (max_len - s.size(1)) : max_len, :] = s

            return final

        return state

    def score(self, hyp: Hypothesis, cache: dict, init_tensor=None) -> None:
        label = torch.tensor(hyp.yseq, device=self.device).unsqueeze(0)
        str_yseq = "".join([str(x) for x in hyp.yseq])

        if str_yseq in cache:
            y, new_state = cache[str_yseq]
        else:
            label_mask = self.create_extended_attention_mask_for_decoder(label.size(), (label != 0).int())

            state = self.check_state(hyp.dec_state, (label.size(1) - 1), self.blank)

            position_ids = self.label_encoder.position_ids[: label.shape[1]].to(self.device)

            position_embed = self.label_encoder.position_embedding(position_ids)
            word_embed = self.label_encoder.word_embedding(label)

            label = word_embed + position_embed

            new_state = list()
            for s, layer in zip(state, self.label_encoder.layers):
                if s is not None:
                    label = label[:, -1:, :]
                    label_mask = label_mask[:, -1:, :]

                label = layer(label, label_mask)

                if s is not None:
                    label = torch.cat([label, s], dim=1)

                new_state.append(label)

            y = self.after_norm(label[:, -1])
            cache[str_yseq] = (y, new_state)

        return y, new_state, "lm_state"


class TransformerTranducerForRNNT(TransducerPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
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

        # [TODO]: 길이 구하는 기능들 클린 코드 할 것
        label_len = (labels != 0).sum(dim=-1) - 1
        non_blank_labels = torch.stack([tensor[1:] for tensor in labels]).to(torch.int32)
        audio_len = torch.IntTensor(
            [torch.masked_select(tensor[1], tensor[1] != 0.0).shape[0] for tensor in input_values],
        )
        audio_len = audio_len.to(input_values.device)
        label_len = label_len.to(labels.device)
        audio_len = audio_len.to(input_values.device)
        non_blank_labels = non_blank_labels.to(labels.device)

        loss = rnnt_loss(
            logits=logits,
            targets=non_blank_labels,
            logit_lengths=torch.tensor([320, 320, 320, 320], device=input_values.device, dtype=torch.int32),
            target_lengths=label_len.to(torch.int32),
            blank=self.blank_id,
            clamp=-1,
            reduction=self.reduction,
        )

        torch.cuda.empty_cache()
        return TransducerOuput(loss=loss, logits=logits)

    def get_lengths(self, audios, labels) -> None:
        return

    def generate(self, audio_input: torch.Tensor, audio_mask: torch.Tensor) -> None:
        self.transducer.audio_encoder(audio_input, audio_mask)
        self.transducer.label_encoder.layers
        self.transducer.joint_network

        return
