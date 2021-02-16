import torch
from multiheadattention import MultiHeadAttention
from pointwisefeedforward import PointWiseFeedForwardNetwork


class EncoderLayer(torch.nn.Module):
    def __init__(self, dimension_model, number_attention_heads, dimension_feed_forward_layer, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(dimension_model, number_attention_heads)
        self.feed_forward_network = PointWiseFeedForwardNetwork(dimension_model, dimension_feed_forward_layer)

        self.dropout_layer_1 = torch.nn.Dropout(dropout_rate)
        self.dropout_layer_2 = torch.nn.Dropout(dropout_rate)

    def compute(self, x):
        attention_output, _ = self.multi_head_attention.compute(x, x, x)
        attention_output = self.dropout_layer_1(attention_output)
        
        m = attention_output + x
        norm1 = torch.nn.LayerNorm(m.size()[1:])

        layer_output_1 = norm1(m)
        feed_output = self.feed_forward_network.get_model(layer_output_1)
        feed_output = self.dropout_layer_2(feed_output)
        
        f = feed_output + layer_output_1
        norm2 = torch.nn.LayerNorm(f.size()[1:])
        layer_output_2 = norm2(f)
        
        return layer_output_2
