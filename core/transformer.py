import torch
from encoder import Encoder


class Transformer(torch.nn.Module):
    def __init__(self, number_of_layers, dimension_model,
                 number_attention_heads, dimension_feed_forward_layer,
                 input_chords_size, target_chords_size,
                 input_chords, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(number_of_layers, dimension_model, number_attention_heads,
                               dimension_feed_forward_layer, input_chords_size, input_chords, dropout_rate)

        self.output_layer = torch.nn.Linear(
            dimension_model, target_chords_size)

    def compute(self, input_data):
        encoder_output = self.encoder.compute(input_data)
        output = self.output_layer(encoder_output)
        return output
