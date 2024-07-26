import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torchaudio.transforms import RNNTLoss

from transformers import AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

from .configuration_transformer_transducer import TransformerTransducerConfig, TransfoXLConfig


logger = logging.get_logger(__name__)

try:
    from k2 import do_rnnt_pruning, get_rnnt_prune_ranges, rnnt_loss_pruned, rnnt_loss_smoothed

    logger.info("use k2")

    USE_K2 = True
except:
    logger.info("not use k2")
    USE_K2 = False


class TransformerTransducerFeatureProjection(nn.Module):
    def __init__(self, config: TransfoXLConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.feature_layer_norm_eps)
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-5):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0,
        pre_lnorm=False,
        r_r_bias=None,
        r_w_bias=None,
        layer_norm_epsilon=1e-5,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def _rel_shift(self, x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None, output_attentions=False):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum("ibnd,jbnd->ijbn", (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum("ibnd,jnd->ijbn", (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        mask_value = torch.finfo(attn_score.dtype).min

        # compute attention probability
        if attn_mask is not None and torch.sum(attn_mask).item():
            attn_mask = attn_mask == 1  # Switch to bool
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float().masked_fill(attn_mask[None, :, :, None], mask_value).type_as(attn_score)
                )
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], mask_value).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = nn.functional.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            outputs = [w + attn_out]
        else:
            # residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]

        if output_attentions:
            outputs.append(attn_prob)

        return outputs


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, layer_norm_epsilon=1e-5, **kwargs):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, layer_norm_epsilon=layer_norm_epsilon, **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm"), layer_norm_epsilon=layer_norm_epsilon
        )

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None, head_mask=None, output_attentions=False):
        attn_outputs = self.dec_attn(
            dec_inp,
            r,
            attn_mask=dec_attn_mask,
            mems=mems,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        ff_output = self.pos_ff(attn_outputs[0])

        outputs = [ff_output] + attn_outputs[1:]

        return outputs


class TransfoXLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TransfoXLConfig
    base_model_prefix = "transformer"

    def _init_weight(self, weight):
        if self.config.init == "uniform":
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
        elif self.config.init == "normal":
            nn.init.normal_(weight, 0.0, self.config.init_std)

    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def _init_weights(self, m):
        """Initialize the weights."""
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                self._init_weight(m.weight)
        elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
            if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
                self._init_weight(m.cluster_weight)
            if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
                self._init_bias(m.cluster_bias)
            if hasattr(m, "out_projs"):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 1.0, self.config.init_std)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "r_emb"):
                self._init_weight(m.r_emb)
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "r_bias"):
                self._init_bias(m.r_bias)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, layer: Optional[int] = -1):
        """
        Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size. Take care of tying
        weights embeddings afterwards if the model class has a *tie_weights()* method.

        Arguments:
            new_num_tokens: (*optional*) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at
                the end. Reducing the size will remove vectors from the end. If not provided or None: does nothing and
                just returns a pointer to the input tokens `torch.nn.Embeddings` Module of the model.
            layer: (*optional*) int:
                Layer of the *AdaptiveEmbedding* where the resizing should be done. Per default the last layer will be
                resized. Be aware that when resizing other than the last layer, you have to ensure that the new
                token(s) in the tokenizer are at the corresponding position.

        Return: `torch.nn.Embeddings` Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed

        if new_num_tokens is None:
            return self.get_input_embeddings()

        new_num_tokens_layer, layer = self._get_new_num_tokens_layer(new_num_tokens, layer)
        assert new_num_tokens_layer > 0, "The size of the new embedding layer cannot be 0 or less"
        model_embeds = base_model._resize_token_embeddings(new_num_tokens_layer, layer)

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens
        base_model.n_token = new_num_tokens

        new_embedding_shapes = self._get_embedding_shapes()
        self._resize_cutoffs(new_num_tokens, new_num_tokens_layer, new_embedding_shapes, layer)

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _get_new_num_tokens_layer(self, new_num_tokens, layer):
        embeddings = self.get_input_embeddings()
        if layer == -1:
            layer = len(embeddings.emb_layers) - 1
        assert 0 <= layer <= len(embeddings.emb_layers) - 1

        new_num_tokens_layer = (
            new_num_tokens
            - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[:layer]])
            - sum([emb.weight.shape[0] for emb in embeddings.emb_layers[layer + 1 :]])
        )
        return new_num_tokens_layer, layer

    def _get_embedding_shapes(self):
        embeddings = self.get_input_embeddings()
        return [emb.weight.shape[0] for emb in embeddings.emb_layers]

    def _resize_token_embeddings(self, new_num_tokens, layer=-1):
        embeddings = self.get_input_embeddings()
        if new_num_tokens is None:
            return embeddings
        new_embeddings_layer = self._get_resized_embeddings(embeddings.emb_layers[layer], new_num_tokens)
        embeddings.emb_layers[layer] = new_embeddings_layer

        self.set_input_embeddings(embeddings)

        return self.get_input_embeddings()

    def _resize_cutoffs(self, new_num_tokens, new_emb_size, new_embedding_shapes, layer):
        embeddings = self.get_input_embeddings()

        for i in range(layer, len(embeddings.cutoffs)):
            embeddings.cutoffs[i] = sum(new_embedding_shapes[: i + 1])

        embeddings.cutoff_ends = [0] + embeddings.cutoffs
        embeddings.n_token = new_num_tokens

        self.config.cutoffs = embeddings.cutoffs[:-1]

        return embeddings.cutoffs


@dataclass
class TransfoXLModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor
    mems: List[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TransformerTransducerModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mems: List[torch.FloatTensor] = None
    text_embeds: torch.FloatTensor = None
    audio_embeds: torch.FloatTensor = None
    total_grad: torch.FloatTensor = None


TRANSFO_XL_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TransfoXLConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TRANSFO_XL_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems
            given to this model should not be passed as `input_values` as they have already been computed.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_values` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_values` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    TRANSFO_XL_START_DOCSTRING,
)
class TransfoXLModel(TransfoXLPreTrainedModel):
    def __init__(self, config: TransfoXLConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head

        self.audio_embed = TransformerTransducerFeatureProjection(config)

        self.drop = nn.Dropout(config.dropout)

        self.n_layer = config.n_layer
        self.mem_len = config.mem_len
        self.attn_type = config.attn_type

        if not config.untie_r:
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        self.layers = nn.ModuleList()
        if config.attn_type == 0:  # the default attention
            for i in range(config.n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        config.n_head,
                        config.d_model,
                        config.d_head,
                        config.d_inner,
                        config.dropout,
                        dropatt=config.dropatt,
                        pre_lnorm=config.pre_lnorm,
                        r_w_bias=None if config.untie_r else self.r_w_bias,
                        r_r_bias=None if config.untie_r else self.r_r_bias,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                    )
                )
        else:  # learnable embeddings and absolute embeddings are not used in our pretrained checkpoints
            raise NotImplementedError  # Removed them to avoid maintaining dead code

        self.same_length = config.same_length
        self.clamp_len = config.clamp_len

        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
        else:  # learnable embeddings and absolute embeddings
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.audio_embed

    def set_input_embeddings(self, new_embeddings):
        self.audio_embed = new_embeddings

    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    def _prune_heads(self, heads):
        logger.info("Head pruning is not implemented for Transformer-XL model")
        pass

    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer):
                empty = torch.zeros(self.mem_len, bsz, self.config.d_model, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=None,
        output_type=TransfoXLModelOutput,
        config_class="TransfoXLConfig",
    )
    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        mems: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TransfoXLModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bsz, qlen, _ = input_features.size()
        # the original code for Transformer-XL used shapes [len, bsz] but we want a unified interface in the library
        # so we transpose here from shape [bsz, len] to shape [len, bsz]
        # if input_features is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_features and inputs_embeds at the same time")
        # elif input_features is not None:
        #     input_features = input_features.transpose(0, 1).contiguous()
        #     qlen, bsz = input_features.size()
        # elif inputs_embeds is not None:
        #     inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
        #     qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        # else:
        #     raise ValueError("You have to specify either input_features or inputs_embeds")

        if mems is None:
            mems = self.init_mems(bsz)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        if inputs_embeds is not None:
            audio_embed = inputs_embeds
        else:
            audio_embed, _ = self.audio_embed(input_features)
            audio_embed = audio_embed.transpose(1, 0)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = audio_embed.new_ones((qlen, klen), dtype=torch.bool)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len))[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(audio_embed.new_ones((qlen, klen), dtype=torch.bool), diagonal=1 + mlen)[
                :, :, None
            ]

        hids = []
        attentions = [] if output_attentions else None
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(
                klen - 1, -1, -1.0, device=audio_embed.device, dtype=torch.int64
            )  # .type_as(dtype=audio_embed.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(audio_embed)
            pos_emb = self.drop(pos_emb)

            for i, layer in enumerate(self.layers):
                hids.append(core_out)
                mems_i = None if mems is None else mems[i]
                layer_outputs = layer(
                    core_out,
                    pos_emb,
                    dec_attn_mask=dec_attn_mask,
                    mems=mems_i,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
                core_out = layer_outputs[0]
                if output_attentions:
                    attentions.append(layer_outputs[1])
        else:  # learnable embeddings and absolute embeddings
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        if output_hidden_states:
            # Add last layer and transpose to library standard shape [bsz, len, hidden_dim]
            hids.append(core_out)
            hids = tuple(t.transpose(0, 1).contiguous() for t in hids)
        else:
            hids = None
        if output_attentions:
            # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
        # We transpose back here to shape [bsz, len, hidden_dim]
        core_out = core_out.transpose(0, 1).contiguous()

        if not return_dict:
            return tuple(v for v in [core_out, new_mems, hids, attentions] if v is not None)

        return TransfoXLModelOutput(
            last_hidden_state=core_out,
            mems=new_mems,
            hidden_states=hids,
            attentions=attentions,
        )


# VisionTextDualEncoderModel를 참고함.
class TransformerTransducerForRNNT(PreTrainedModel):
    config_class = TransformerTransducerConfig
    base_model_prefix = "transformer_transducer"

    def __init__(self, config: TransformerTransducerConfig) -> None:
        super().__init__(config)

        self.config = config

        self.audio_model = TransfoXLModel(config.audio_config)
        self.text_model = AutoModel.from_config(config.text_config)

        self.audio_embed_dim = config.audio_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        projection_dim = config.vocab_size if USE_K2 else self.projection_dim
        self.audio_projection = nn.Linear(self.audio_embed_dim, projection_dim)
        self.text_projection = nn.Linear(self.text_embed_dim, projection_dim)

        self.joint_act = ACT2FN[config.joint_act]
        self.vocab_projection = nn.Linear(self.projection_dim, config.vocab_size)

        if USE_K2:
            self.hidden_projection = nn.Linear(self.config.vocab_size, self.projection_dim)

    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> torch.Tensor:
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[0]
        text_features = self.text_projection(text_embeds)

        return text_features

    def get_audio_features(
        self,
        input_features=None,
        attention_mask=None,
        mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        audio_outputs = self.audio_model(
            input_features=input_features,
            attention_mask=attention_mask,
            mems=mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        audio_embeds = audio_outputs[0]  # audio_embeds
        mems = audio_outputs[1]
        audio_features = self.audio_projection(audio_embeds)

        return (audio_features, mems)

    def joint_network(self, hidden_states) -> torch.Tensor:
        if USE_K2:
            hidden_states = self.hidden_projection(hidden_states)
        return self.vocab_projection(self.joint_act(hidden_states))

    def forward(
        self,
        input_features,
        labels,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        mems: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bsz, seq_len, _ = input_features.shape
        chunk_num = math.ceil(seq_len / self.config.one_sec_mel_shape) * self.config.one_sec_mel_shape
        chunk_features_ls = list()
        for idx in range(0, chunk_num, self.config.one_sec_mel_shape):
            chunk_features = input_features[:, idx : idx + self.config.one_sec_mel_shape]
            chunk_mask = attention_mask[:, idx : idx + self.config.one_sec_mel_shape]
            audio_outputs = self.audio_model(
                input_features=chunk_features,
                mems=mems,
                attention_mask=chunk_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            audio_embeds = audio_outputs[0]  # pooler_output
            audio_embeds = self.audio_projection(audio_embeds)

            mems = audio_outputs.mems
            chunk_features_ls.append(audio_embeds)

        text_outputs = self.text_model(
            input_ids=labels,
            attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        audio_embeds = torch.concat(chunk_features_ls, axis=1)

        text_embeds = text_outputs[0]  # pooler_output
        text_embeds = self.text_projection(text_embeds)

        loss = None
        total_grad = None
        if (labels is not None) and (not USE_K2):
            combined_hidden = audio_embeds[:, :, None, :] + text_embeds[:, None, :, :]
            logits = self.joint_network(combined_hidden)

            feature_lengths = attention_mask.sum(-1)
            label_lengths = decoder_attention_mask.sum(-1) - 1  # remove blank length
            non_blank_labels = labels[:, 1:]

            loss_fct = RNNTLoss(
                blank=self.config.blk_token_ids,
                reduction=self.config.reduction,
                fused_log_softmax=True,
            )
            loss = loss_fct(
                logits=logits,
                targets=non_blank_labels.to(torch.int32),
                target_lengths=label_lengths.to(torch.int32),
                logit_lengths=feature_lengths.to(torch.int32),
            )
        elif (labels is not None) and USE_K2:
            boundary = torch.zeros((bsz, 4), dtype=torch.int64, device=self.device)
            boundary[:, 2] = decoder_attention_mask.sum(-1) - 1
            boundary[:, 3] = attention_mask.sum(-1)

            simple_loss, (px_grad, py_grad) = rnnt_loss_smoothed(
                lm=text_embeds,
                am=audio_embeds,
                symbols=labels[:, 1:],
                termination_symbol=self.config.blk_token_ids,
                lm_only_scale=0.0,
                am_only_scale=0.0,
                boundary=boundary,
                reduction=self.config.reduction,
                return_grad=True,
            )
            prune_range = 5
            ranges = get_rnnt_prune_ranges(
                px_grad=px_grad,
                py_grad=py_grad,
                boundary=boundary,
                s_range=prune_range,
            )
            am_pruned, lm_pruned = do_rnnt_pruning(lm=text_embeds, am=audio_embeds, ranges=ranges)

            combined_hidden = am_pruned + lm_pruned
            logits = self.joint_network(combined_hidden)

            pruned_loss = rnnt_loss_pruned(
                logits=logits,
                symbols=labels[:, 1:],
                ranges=ranges,
                termination_symbol=self.config.blk_token_ids,
                boundary=boundary,
                reduction=self.config.reduction,
            )

            loss = (self.config.simple_loss_scale * simple_loss) + pruned_loss

            B = px_grad.size(0)
            S = px_grad.size(1)
            T = px_grad.size(2) - 1
            px_grad_pad = torch.zeros((B, 1, T + 1), dtype=px_grad.dtype, device=self.device)
            py_grad_pad = torch.zeros((B, S + 1, 1), dtype=px_grad.dtype, device=self.device)

            px_grad_padded = torch.cat([px_grad, px_grad_pad], dim=1)
            py_grad_padded = torch.cat([py_grad, py_grad_pad], dim=2)

            total_grad = px_grad_padded + py_grad_padded

        if not return_dict:
            output = (logits, text_embeds, audio_embeds, text_outputs, audio_outputs)
            return ((loss,) + output) if loss is not None else output

        return TransformerTransducerModelOutput(
            loss=loss,
            mems=mems,
            logits=logits,
            total_grad=total_grad,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
        )
