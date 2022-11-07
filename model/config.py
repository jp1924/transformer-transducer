class TransformerTransducerConfig:
    def __init__(self, vocab_size) -> None:
        self.label_layers = 2
        self.audio_layers = 18

        self.hidden_size = 756
        self.ffn_size = 3072
        self.ffn_dropout = 0.03
        self.ffn_norm_eps = 0.00001

        self.attn_dropout = 0.03
        self.attn_norm_eps = 0.00001
        self.score_dropout = 0.07
        self.head_size = 12

        self.position_embed_size = 512
        self.vocab_size = vocab_size

        self.loss_reduction = "mean"
        self.blank_id = 1
