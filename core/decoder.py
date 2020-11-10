import torch
from decoderlayer import DecoderLayer


class Decoder(torch.nn.Linear):
    def __init__(self, number_of_layers, dimension_model, number_attention_heads, dimension_feed_forward_layer,
                 target_chords_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.number_of_layers = number_of_layers
        self.dimension_model = dimension_model
        self.embedding = torch.nn.Embedding(target_chords_size, dimension_model)

        self.Decoder_layers = self.dec_layers = [DecoderLayer(dimension_model, number_attention_heads,
                                                              dimension_feed_forward_layer, dropout_rate) for _ in
                                                 range(number_of_layers)]

        self.dropout_layer = torch.nn.Dropout(dropout_rate)

    def create_architecture(self):
        """
            This method is going to link all the layers declared above
            and will return the final layer.
        :return: self.layer based on the other layers
        """
        pass
