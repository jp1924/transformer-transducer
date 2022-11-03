class TransformerTransducerConfig:
    def __init__(self) -> None:
        self.label_layers = 8
        self.sudio_layers = 8

        self.hidden_size = 756
        self.ffn_size = 3072
        self.ffn_dropout = 0.03
        self.ffn_norm_eps = 0.00001

        self.attn_dropout = 0.03
        self.attn_norm_eps = 0.00001
        self.score_dropout = 0.07

        self.position_embed_size = 512
