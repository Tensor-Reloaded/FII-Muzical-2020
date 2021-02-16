import torch
import numpy as np
from torch.nn import functional as F


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dimension_model, number_attention_heads):
        super(MultiHeadAttention, self).__init__()
        self.number_attention_heads = number_attention_heads
        self.dimension_model = dimension_model
        self.number_layers = dimension_model // number_attention_heads

        self.weights_query = torch.nn.Linear(dimension_model, dimension_model)
        self.weights_key = torch.nn.Linear(dimension_model, dimension_model)
        self.weights_value = torch.nn.Linear(dimension_model, dimension_model)
        self.layer_heads = torch.nn.Linear(dimension_model, dimension_model)

    @staticmethod
    def casual_attention_mask(dim_d, dim_s, dim_type):
        i = torch.arange(dim_d)[:, None]
        j = torch.arange(dim_s)
        m = i >= j - dim_s + dim_d
        m = m.type(dim_type)
        return m

    def dot_product_attention(self, query, key, value):
        key = torch.transpose(key, -1, 2)
        attention_score = torch.matmul(query, key)
        scaled_attention = attention_score / np.sqrt(query.size(-1))

        attention_mask = self.casual_attention_mask(scaled_attention.shape[2], scaled_attention.shape[3],
                                                    scaled_attention.dtype)
        attention_mask = torch.reshape(attention_mask, [1, 1, scaled_attention.shape[2], scaled_attention.shape[3]])
        scaled_attention = scaled_attention * attention_mask - 1e4 * (1 - attention_mask)

        w = F.softmax(scaled_attention, dim=-1)
        output = torch.matmul(w, value)
        return output, w

    def split_heads(self, val, size):
        r = val.view(size, -1, self.number_attention_heads, self.number_layers)
        return r.permute(0, 2, 1, 3)

    def compute(self, value, key, query):
        batch_size = query.shape[0]

        query = self.split_heads(self.weights_query(query), batch_size)
        key = self.split_heads(self.weights_key(key), batch_size)
        value = self.split_heads(self.weights_value(value), batch_size)

        scaled_attention, attention_w = self.dot_product_attention(query, key, value)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        concat_attention = scaled_attention.reshape(batch_size, -1, self.dimension_model)
        return self.layer_heads(concat_attention), attention_w
