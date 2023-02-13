from transformers import PretrainedConfig

# [NOTE]: No environment to train the model now, temporarily put wav2vec2
TRANSFORMER_TRANSDUCER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "jp42maru/transformer-transducer-960h": "https://huggingface.co/jp42maru/transformer-transducer-960h/tree/main/config.json",
}


class TransformerTransducerConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=111,
        initializer_range=0.02,
        encoder_layers=18,
        encoder_layerdrop=0.0,
        decoder_layers=2,
        decoder_layerdrop=0.0,
        hidden_size=512,
        hidden_dropout=0.1,
        hidden_act="gelu",
        joiner_act="tanh",
        layer_norm_eps=0.00001,
        max_position_embeddings=512,
        position_embedding_type="relative_key",
        intermediate_size=2048,
        activation_dropout=0.1,
        attention_dropout=0.1,
        num_attention_heads=8,
        attention_type="original_full",
        score_dropout=0.07,
        loss_reduction="mean",
        clamp=-1,
        blk_token_id=0,
        generate_repeat_max=10,
        apply_spec_augment=True,
        mask_time_prob=0.1,
        mask_time_length=30,
        mask_time_min_masks=1,
        mask_feature_prob=0.02,
        mask_feature_length=50,
        mask_feature_min_masks=1,
        mask_chunk_size=3,
        mask_left_context=10,
        mask_right_context=3,
        use_cache=False,
        **kwargs
    ) -> None:
        self.encoder_layers = encoder_layers
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layers = decoder_layers
        self.decoder_layerdrop = decoder_layerdrop

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_dropout = activation_dropout
        self.hidden_dropout = hidden_dropout
        self.hidden_act = hidden_act

        self.joiner_act = joiner_act

        self.attention_dropout = attention_dropout
        self.score_dropout = score_dropout
        self.num_attention_heads = num_attention_heads

        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout = hidden_dropout

        self.use_cache = use_cache

        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.vocab_size = vocab_size

        self.loss_reduction = loss_reduction
        self.clamp = clamp
        self.initializer_range = initializer_range

        self.blk_token_id = blk_token_id

        # attention_mask
        self.mask_chunk_size = mask_chunk_size
        self.mask_left_context = mask_left_context
        self.mask_right_context = mask_right_context

        # spec-augment
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        # for generate
        self.attention_type = attention_type
        self.generate_repeat_max = generate_repeat_max

        super().__init__(**kwargs)
