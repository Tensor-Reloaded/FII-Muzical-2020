import torch
import math
from encoderlayer import EncoderLayer


class Encoder(torch.nn.Module):
    def __init__(self, number_of_layers, dimension_model, number_attention_heads, dimension_feed_forward_layer,
                 input_chords, target_max, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.number_of_layers = number_of_layers
        self.dimension_model = dimension_model
        self.embedding = torch.nn.Embedding(input_chords, dimension_model)

        # print(target_max, dimension_model)
        self.pos_encoding = self.positional_encoding(target_max, dimension_model)
        # print(self.pos_encoding)

        self.encoder_layers = [EncoderLayer(dimension_model, number_attention_heads,
                                            dimension_feed_forward_layer, dropout_rate) for _ in
                               range(number_of_layers)]

        self.dropout_layer = torch.nn.Dropout(dropout_rate)

    def positional_encoding(self, target_max, d_model):
        result = torch.zeros(target_max, d_model)
        position = torch.arange(0, target_max, dtype=torch.float).unsqueeze(1)
        angle_rads = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        result[:, 0::2] = torch.sin(position * angle_rads)
        result[:, 1::2] = torch.cos(position * angle_rads)
        return result

    def compute(self, x):
        dim = x.shape[1]

        x = self.embedding(x)
        d = torch.tensor(self.dimension_model)
        x *= torch.sqrt(d.type(torch.FloatTensor))
        x += self.pos_encoding[:dim, :]
        x = self.dropout_layer(x)

        for i in range(self.number_of_layers):
            x = self.encoder_layers[i].compute(x)

        return x