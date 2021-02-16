import torch
import numpy as np
from encoderlayer import EncoderLayer


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


class Encoder(torch.nn.Module):
    def __init__(self, number_of_layers, dimension_model, number_attention_heads, dimension_feed_forward_layer,
                 input_chords, target_max, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.number_of_layers = number_of_layers
        self.dimension_model = dimension_model
        self.embedding = torch.nn.Embedding(input_chords, dimension_model)

        self.pos_encoding = positional_encoding(target_max, dimension_model)

        self.encoder_layers = [EncoderLayer(dimension_model, number_attention_heads,
                                            dimension_feed_forward_layer, dropout_rate) for _ in
                               range(number_of_layers)]

        self.dropout_layer = torch.nn.Dropout(dropout_rate)

    def compute(self, x):
        dim = x.shape[1]

        x = self.embedding(x)
        d = torch.tensor(self.dimension_model)
        x *= torch.sqrt(d.type(torch.FloatTensor))
        x += self.pos_encoding[:, :dim, :]
        x = self.dropout_layer(x)

        for i in range(self.number_of_layers):
            x = self.encoder_layers[i].compute(x)

        return x
