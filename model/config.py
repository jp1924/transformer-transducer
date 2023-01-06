from transformers import PretrainedConfig


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
        layer_norm_eps=0.00001,
        max_position_embeddings=512,
        position_embedding_type="absolute",
        intermediate_size=2048,
        activation_dropout=0.1,
        attention_dropout=0.1,
        num_attention_heads=8,
        attention_type="original_full",
        score_dropout=0.07,
        loss_reduction="mean",
        clamp=-1,
        decoder_start_token_id=0,
        freq_mask_size=50,
        time_mask_size=30,
        freq_apply_num=2,
        time_apply_num=10,
        bos_token_id=0,
        blk_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        generate_repeat_max=10,
        output_attentions=False,
        output_hidden_states=False,
        is_encoder_decoder=True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.encoder_layers = encoder_layers
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layers = decoder_layers
        self.decoder_layerdrop = decoder_layerdrop

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_dropout = activation_dropout
        self.hidden_dropout = hidden_dropout
        self.hidden_act = hidden_act

        self.attention_dropout = attention_dropout
        self.score_dropout = score_dropout
        self.num_attention_heads = num_attention_heads

        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout = hidden_dropout

        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size

        self.position_embedding_type = position_embedding_type

        self.loss_reduction = loss_reduction
        self.clamp = clamp
        self.initializer_range = initializer_range

        self.blk_token_id = blk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.is_encoder_decoder = is_encoder_decoder

        # spec-augment
        self.freq_mask_size = freq_mask_size
        self.time_mask_size = time_mask_size
        self.freq_apply_num = freq_apply_num
        self.time_apply_num = time_apply_num

        # for generate
        self.generate_repeat_max = generate_repeat_max

        self.pruned_heads = {}
        self.num_return_sequences = 1
        self.early_stopping = True
        self.length_penalty = 1.0

        self.max_length = 512
        self.min_length = 1
        self.do_sample = False
        self.early_stopping = False
        self.num_beams = 1
        self.num_beam_groups = 1
        self.diversity_penalty = 0.0
        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 1.0
        self.typical_p = 1.0
        self.repetition_penalty = 1.0
        self.length_penalty = 1.0
        self.no_repeat_ngram_size = 0
        self.encoder_no_repeat_ngram_size = 0
        self.bad_words_ids = None
        self.num_return_sequences = 1
        self.chunk_size_feed_forward = 0
        self.output_scores = False
        self.return_dict_in_generate = False
        self.forced_bos_token_id = None
        self.forced_eos_token_id = None
        self.remove_invalid_values = False
        self.exponential_decay_length_penalty = None
        self.suppress_tokens = None
        self.begin_suppress_tokens = None

        self.decoder_start_token_id = decoder_start_token_id
        self.attention_type = attention_type
