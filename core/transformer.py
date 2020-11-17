from encoder import Encoder
from decoder import Decoder


class Transformer:
    def __init__(self):
        self.encoder = None
        self.decoder = None

    def setEncoder(self, encoder):
        self.encoder = encoder

    def setDecoder(self, decoder):
        self.decoder = decoder