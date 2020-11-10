import torch


class DecoderLayer(torch.nn.Linear):
    def __init__(self, dimension_model, number_attention_heads, dimension_feed_forward_layer, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention1 = MultiHeadAttention(dimension_model, number_attention_heads)
        self.multi_head_attention2 = MultiHeadAttention(dimension_model, number_attention_heads)
        self.feed_forward_network = PointWiseFeedForwardNetwork(dimension_model,
                                                                dimension_feed_forward_layer).get_model()

        self.layer_1_normalized = torch.nn.LayerNorm(eps=1e-6)  # TODO
        self.layer_2_normalized = torch.nn.LayerNorm(eps=1e-6)  # TODO
        self.layer_3_normalized = torch.nn.LayerNorm(eps=1e-6)  # TODO

        self.dropout_layer_1 = torch.nn.Dropout(dropout_rate)
        self.dropout_layer_2 = torch.nn.Dropout(dropout_rate)
        self.dropout_layer_3 = torch.nn.Dropout(dropout_rate)

    def create_architecture(self, dimension_model, number_attention_heads):
        """
            This method is going to link all the layers declared above
            and will return the final layer.
        :return: self.layer based on the other layers
        """
        pass