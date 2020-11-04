from transformer import Transformer


class TransformerBuilder:
    def __init__(self):
        self.transformer = Transformer()

    def with_encoder(self, encoder):
        self.transformer.setEncoder(encoder)
        return self

    def with_decoder(self, decoder):
        self.transformer.setDecoder(decoder)
        return self

    def build(self):
        return self.transformer
