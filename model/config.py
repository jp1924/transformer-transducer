class TransformerTransducerConfig:
    def __init__(self, vocab_size) -> None:
        self.label_layers = 2
        self.audio_layers = 18

        self.hidden_size = 504
        self.intermediate_size = 1024
        self.activation_dropout = 0.03
        self.hidden_dropout = 0.03
        self.hidden_act = "gelu"

        self.attention_probs_dropout_prob = 0.03
        self.score_dropout = 0.07
        self.num_attention_heads = 12

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
