import torch
import numpy as np
from decoderlayer import DecoderLayer


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = torch.from_numpy(angle_rads[np.newaxis, ...])
    return pos_encoding.type(torch.FloatTensor)

class Decoder(torch.nn.Module):
    def __init__(self, number_of_layers, dimension_model, number_attention_heads, dimension_feed_forward_layer,
                 target_chords_size, target_max,  dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.number_of_layers = number_of_layers
        self.dimension_model = dimension_model
        self.embedding = torch.nn.Embedding(
            target_chords_size, dimension_model)

        self.pos_encoding = positional_encoding(target_max, self.dimension_model)

        self.decoder_layers = self.dec_layers = [DecoderLayer(dimension_model, number_attention_heads,
                                                              dimension_feed_forward_layer, dropout_rate) for _ in
                                                 range(number_of_layers)]

        self.dropout_layer = torch.nn.Dropout(dropout_rate)

    def compute(self, x, enc_out, train, ahead_m, padd_m):
        dim = x.shape[1]
        attention_weights = {}

        x = self.embedding(x)
        d = torch.tensor(self.dimension_model)
        x *= torch.sqrt(d.type(torch.FloatTensor))
        x += self.pos_encoding[:, :dim, :]
        x = self.dropout_layer(x)

        for i in range(self.number_of_layers):
            x, block_one, block_two = self.decoder_layers[i].compute(x, enc_out, train, ahead_m, padd_m)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block_one
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block_two

        return x, attention_weights