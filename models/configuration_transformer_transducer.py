from copy import deepcopy

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.utils import logging


logger = logging.get_logger(__name__)


class TransfoXLConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`TransfoXLModel`] or a [`TFTransfoXLModel`]. It is
    used to instantiate a Transformer-XL model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the TransfoXL
    [transfo-xl/transfo-xl-wt103](https://huggingface.co/transfo-xl/transfo-xl-wt103) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 267735):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`TransfoXLModel`] or [`TFTransfoXLModel`].
        cutoffs (`List[int]`, *optional*, defaults to `[20000, 40000, 200000]`):
            Cutoffs for the adaptive softmax.
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the model's hidden states.
        d_embed (`int`, *optional*, defaults to 1024):
            Dimensionality of the embeddings
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_head (`int`, *optional*, defaults to 64):
            Dimensionality of the model's heads.
        d_inner (`int`, *optional*, defaults to 4096):
            Inner dimension in FF
        div_val (`int`, *optional*, defaults to 4):
            Divident value for adapative input and softmax
        pre_lnorm (`boolean`, *optional*, defaults to `False`):
            Whether or not to apply LayerNorm to the input instead of the output in the blocks.
        n_layer (`int`, *optional*, defaults to 18):
            Number of hidden layers in the Transformer encoder.
        mem_len (`int`, *optional*, defaults to 1600):
            Length of the retained previous heads.
        clamp_len (`int`, *optional*, defaults to 1000):
            Use the same pos embeddings after clamp_len.
        same_length (`boolean`, *optional*, defaults to `True`):
            Whether or not to use the same attn length for all tokens
        proj_share_all_but_first (`boolean`, *optional*, defaults to `True`):
            True to share all but first projs, False not to share.
        attn_type (`int`, *optional*, defaults to 0):
            Attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
        sample_softmax (`int`, *optional*, defaults to -1):
            Number of samples in the sampled softmax.
        adaptive (`boolean`, *optional*, defaults to `True`):
            Whether or not to use adaptive softmax.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        dropatt (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        untie_r (`boolean`, *optional*, defaults to `True`):
            Whether ot not to untie relative position biases.
        init (`str`, *optional*, defaults to `"normal"`):
            Parameter initializer to use.
        init_range (`float`, *optional*, defaults to 0.01):
            Parameters initialized by U(-init_range, init_range).
        proj_init_std (`float`, *optional*, defaults to 0.01):
            Parameters initialized by N(0, init_std)
        init_std (`float`, *optional*, defaults to 0.02):
            Parameters initialized by N(0, init_std)
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers
        eos_token_id (`int`, *optional*, defaults to 0):
            End of stream token id.

    Examples:

    ```python
    >>> from transformers import TransfoXLConfig, TransfoXLModel

    >>> # Initializing a Transformer XL configuration
    >>> configuration = TransfoXLConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = TransfoXLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "transfo-xl"
    keys_to_ignore_at_inference = ["mems"]
    attribute_map = {
        "n_token": "vocab_size",
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        cutoffs=[20000, 40000, 200000],
        d_model=1024,
        d_embed=1024,
        n_head=16,
        d_head=64,
        d_inner=4096,
        div_val=4,
        pre_lnorm=False,
        n_layer=18,
        mem_len=1600,
        clamp_len=1000,
        same_length=True,
        proj_share_all_but_first=True,
        attn_type=0,
        sample_softmax=-1,
        adaptive=True,
        dropout=0.1,
        dropatt=0.0,
        untie_r=True,
        init="normal",
        init_range=0.01,
        proj_init_std=0.01,
        init_std=0.02,
        layer_norm_epsilon=1e-5,
        eos_token_id=0,
        feature_projection_input_dim=160,
        feat_proj_dropout=0.0,
        **kwargs,
    ):
        self.cutoffs = []
        self.cutoffs.extend(cutoffs)
        if proj_share_all_but_first:
            self.tie_projs = [False] + [True] * len(self.cutoffs)
        else:
            self.tie_projs = [False] + [False] * len(self.cutoffs)
        self.d_model = d_model
        self.d_embed = d_embed
        self.d_head = d_head
        self.d_inner = d_inner
        self.div_val = div_val
        self.pre_lnorm = pre_lnorm
        self.n_layer = n_layer
        self.n_head = n_head
        self.mem_len = mem_len
        self.same_length = same_length
        self.attn_type = attn_type
        self.clamp_len = clamp_len
        self.sample_softmax = sample_softmax
        self.adaptive = adaptive
        self.dropout = dropout
        self.dropatt = dropatt
        self.untie_r = untie_r
        self.init = init
        self.init_range = init_range
        self.proj_init_std = proj_init_std
        self.init_std = init_std
        self.layer_norm_epsilon = layer_norm_epsilon
        self.feature_projection_input_dim = feature_projection_input_dim
        self.feat_proj_dropout = feat_proj_dropout

        super().__init__(eos_token_id=eos_token_id, **kwargs)

    @property
    def max_position_embeddings(self):
        # Message copied from Transformer-XL documentation
        logger.info(f"The model {self.model_type} is one of the few models that has no sequence length limit.")
        return -1

    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        # Message copied from Transformer-XL documentation
        raise NotImplementedError(
            f"The model {self.model_type} is one of the few models that has no sequence length limit."
        )


class TransformerTransducerConfig(PretrainedConfig):
    model_type = "transformer_transducer"
    keys_to_ignore_at_inference = ["mems"]
    attribute_map = {
        "n_token": "vocab_size",
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        audio_config = kwargs.pop("audio_config", {})
        text_config = kwargs.pop("text_config", {})

        self.audio_config = TransfoXLConfig(**audio_config)

        text_model_type = text_config["model_type"]
        text_config_class = CONFIG_MAPPING[text_model_type]
        self.text_config = text_config_class(**text_config)

    @classmethod
    def from_vision_text_configs(
        cls,
        audio_config: PretrainedConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.

        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        """

        return cls(
            audio_config=audio_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = deepcopy(self.__dict__)
        output["audio_config"] = self.audio_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type

        return output
