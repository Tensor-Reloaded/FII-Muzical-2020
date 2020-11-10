import torch


class MultiHeadAttention(torch.nn.Linear):
    def __init__(self, number_attention_heads, dimension_model):
        super(MultiHeadAttention, self).__init__()
        self.number_attention_heads = number_attention_heads
        self.dimension_model = dimension_model
        self.number_layers = dimension_model // number_attention_heads

        # query layer
        self.weights_query = torch.nn.Linear(dimension_model)
        # key layer
        self.weights_key = torch.nn.Linear(dimension_model)
        # value layer
        self.weights_value = torch.nn.Linear(dimension_model)

        self.layer = torch.nn.Linear(dimension_model)

    def split_heads(self):
        pass

    def create_architecture(self):
        """
            This method is going to link all the layers declared above
            and will return the final layer.
        :return: self.layer based on the other layers
        """
        pass