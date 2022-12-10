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

        self.position_embed_size = 512
        self.vocab_size = vocab_size

        self.position_embedding_type = "absolute"

        self.loss_reduction = "mean"
        self.blank_id = 0
        self.is_audio = False
        self.mel_size = 80
        self.model_test = True
        self.pruned_heads = False
        self.initializer_range = 0.02
        self.output_attentions = False
        self.output_hidden_states = False
        self.max_length = 512
        self.num_beams = 1
        self.is_encoder_decoder = True
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

        self.num_return_sequences = 1
        self.early_stopping = True
        self.num_beam_groups = 1
        self.length_penalty = 1.0

        self.do_sample = False

        self.output_scores = False
        self.output_attentions = False
        self.output_hidden_states = False
        self.return_dict_in_generate = True
