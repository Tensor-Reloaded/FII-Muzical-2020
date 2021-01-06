import torch
import numpy as np

def dot_product_attention(query, key, value, mask):
    attention = torch.matmul(query, torch.transpose(key, -2, -1)) / np.sqrt(query.size(-1))
    if mask is not None:
        attention += (mask * -1e9)
    w = torch.softmax(attention, axis = -1)
    r = torch.matmul(w, value)
    return r, w

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dimension_model, number_attention_heads):
        super(MultiHeadAttention, self).__init__()
        self.number_attention_heads = number_attention_heads
        self.dimension_model = dimension_model
        self.number_layers = dimension_model // number_attention_heads

        # layers
        self.weights_query = torch.nn.Linear(dimension_model, dimension_model)
        self.weights_key = torch.nn.Linear(dimension_model, dimension_model)
        self.weights_value = torch.nn.Linear(dimension_model, dimension_model)

        self.layer = torch.nn.Linear(dimension_model, dimension_model)


    def split_heads(self, val, size):
        r = val.view(size, -1, self.number_attention_heads, self.number_layers)
        return r.transpose(1,2)

    def compute(self, value, key, query, mask):
        batch_size = query.shape[0]

        query = self.split_heads(self.weights_key(query), batch_size)
        key = self.split_heads(self.weights_key(key), batch_size)
        value = self.split_heads(self.weights_value(value), batch_size)

        scaled_attention, attention_w = dot_product_attention(query, key, value, mask)
        concat_attention = scaled_attention.transpose(1,2).contiguous().view(batch_size, -1, self.dimension_model)
        return self.layer(concat_attention), attention_w


