import torch
import numpy as np
from encoder import Encoder
from decoder import Decoder


class Transformer(torch.nn.Module):
    def __init__(self, number_of_layers, dimension_model, number_attention_heads, dimension_feed_forward_layer, input_chords_size, target_chords_size, input_chords, target_chords, dropout_rate=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(number_of_layers, dimension_model, number_attention_heads, dimension_feed_forward_layer, input_chords_size, input_chords, dropout_rate)


        self.decoder = Decoder(number_of_layers, dimension_model, number_attention_heads, dimension_feed_forward_layer, target_chords_size, target_chords, dropout_rate)
        self.output_layer = torch.nn.Linear(dimension_model, target_chords_size)

    def compute(self, input, target, train, enc_padding_mask, ahead_m, dec_padding_mask):
        encoder_output = self.encoder.compute(input, target, enc_padding_mask)
        decoder_output, attention_w = self.decoder.compute(target, encoder_output, train, ahead_m, dec_padding_mask)
        output = self.output_layer(decoder_output)
        return output, attention_w 