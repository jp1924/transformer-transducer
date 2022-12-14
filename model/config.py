from transformers import PretrainedConfig


class TransformerTransducerConfig(PretrainedConfig):
    def __init__(self, vocab_size=None) -> None:
        self.label_layers = 2
        self.audio_layers = 18

        self.hidden_size = 320
        self.intermediate_size = 640
        self.activation_dropout = 0.03
        self.hidden_dropout = 0.03
        self.hidden_act = "gelu"

        self.attention_probs_dropout_prob = 0.03
        self.score_dropout = 0.07
        self.num_attention_heads = 10

        self.layer_norm_eps = 0.00001
        # self.attn_norm_eps = 0.00001
        self.hidden_dropout = 0.0002

        self.max_position_embeddings = 512
        self.vocab_size = vocab_size

        self.position_embedding_type = "absolute"

        self.loss_reduction = "mean"
        self.is_decoder = False
        self.blank_id = 0
        self.is_audio = False
        self.mel_size = 80

        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.pruned_heads = False
        self.initializer_range = 0.02
        self.output_attentions = False
        self.output_hidden_states = False
        self.is_encoder_decoder = True

        self.num_return_sequences = 1
        self.early_stopping = True
        self.length_penalty = 1.0

        # for generate
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

        self.decoder_start_token_id = 0
        self.attention_type = "diagonal"
        self.clamp = -1
