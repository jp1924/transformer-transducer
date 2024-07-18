from copy import deepcopy

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.utils import logging


logger = logging.get_logger(__name__)


class TransfoXLConfig(PretrainedConfig):
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
        d_model=512,
        d_inner=1024,
        n_head=8,
        d_head=64,
        div_val=4,
        pre_lnorm=False,
        n_layer=12,
        mem_len=512,
        clamp_len=1000,
        same_length=True,
        proj_share_all_but_first=True,
        attn_type=0,
        dropout=0.1,
        dropatt=0.0,
        untie_r=True,
        init="normal",
        init_range=0.01,
        proj_init_std=0.01,
        init_std=0.02,
        layer_norm_epsilon=1e-5,
        feature_projection_input_dim=512,
        feat_proj_dropout=0.0,
        feature_layer_norm_eps=1e-05,
        apply_spec_augment=True,
        mask_time_prob=0.1,
        mask_time_length=30,
        mask_time_min_masks=1,
        mask_feature_prob=0.02,
        mask_feature_length=50,
        mask_feature_min_masks=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoffs = []
        self.cutoffs.extend(cutoffs)
        if proj_share_all_but_first:
            self.tie_projs = [False] + [True] * len(self.cutoffs)
        else:
            self.tie_projs = [False] + [False] * len(self.cutoffs)
        self.d_model = d_model
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
        self.dropout = dropout
        self.dropatt = dropatt
        self.untie_r = untie_r
        self.init = init
        self.init_range = init_range
        self.proj_init_std = proj_init_std
        self.init_std = init_std
        self.layer_norm_epsilon = layer_norm_epsilon
        self.feature_layer_norm_eps = feature_layer_norm_eps
        self.feature_projection_input_dim = feature_projection_input_dim
        self.feat_proj_dropout = feat_proj_dropout

        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

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
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        audio_config = kwargs.pop("audio_config", {})
        text_config = kwargs.pop("text_config", {})

        self.audio_config = TransfoXLConfig(**audio_config)

        text_model_type = text_config["model_type"]
        text_config_class = CONFIG_MAPPING[text_model_type]
        self.text_config = text_config_class(**text_config)

        self.vocab_size = kwargs.pop("vocab_size", self.text_config.vocab_size)
        self.one_sec_mel_shape = kwargs.pop("one_sec_mel_shape", 34)
        self.joint_act = kwargs.pop("joint_act", "tanh")
        self.projection_dim = kwargs.pop("projection_dim", 512)
        self.blk_token_ids = kwargs.pop("blk_token_ids", -1)
        self.reduction = kwargs.pop("reduction", "mean")
        self.simple_loss_scale = kwargs.pop("simple_loss_scale", 0.5)

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
