import torch
from multiheadattention import MultiHeadAttention
from pointwisefeedforwardnetwork import PointWiseFeedForwardNetwork


class DecoderLayer(torch.nn.Module):
    def __init__(self, dimension_model, number_attention_heads, dimension_feed_forward_layer, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention1 = MultiHeadAttention(
            dimension_model, number_attention_heads)
        self.multi_head_attention2 = MultiHeadAttention(
            dimension_model, number_attention_heads)
        self.feed_forward_network = PointWiseFeedForwardNetwork(dimension_model,
                                                                dimension_feed_forward_layer)

        # self.layer_1_normalized = torch.nn.LayerNorm(eps=1e-6)  # TODO
        # self.layer_2_normalized = torch.nn.LayerNorm(eps=1e-6)  # TODO
        # self.layer_3_normalized = torch.nn.LayerNorm(eps=1e-6)  # TODO

        self.dropout_layer_1 = torch.nn.Dropout(dropout_rate)
        self.dropout_layer_2 = torch.nn.Dropout(dropout_rate)
        self.dropout_layer_3 = torch.nn.Dropout(dropout_rate)

    def Normalize(self, x):
        norm = torch.nn.LayerNorm(x.size()[1:])
        return norm(x)


    def compute(self, x, encoder_output, train, ahead_m, padd_m):
        attention_output_one, attention_weights_one = self.multi_head_attention1.compute(x, x, x, ahead_m)
        attention_output_one = self.dropout_layer_1(attention_output_one)
        layer_output_1 = self.Normalize(attention_output_one + x)

        attention_output_two, attention_weights_two = self.multi_head_attention2.compute(encoder_output, encoder_output, layer_output_1, padd_m)
        attention_output_two = self.dropout_layer_1(attention_output_two)
        layer_output_2 = self.Normalize(attention_output_two + layer_output_1)

        feed_output = self.feed_forward_network.get_model(layer_output_2)
        feed_output = self.dropout_layer_2(feed_output)
        layer_output_3 = self.Normalize(feed_output + layer_output_2)
        return layer_output_3, attention_weights_one, attention_weights_two
