import math

import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.layer_size = layer_size
        self.query = nn.Linear(layer_size, layer_size, bias=False)
        self.key = nn.Linear(layer_size, layer_size, bias=False)
        self.value = nn.Linear(layer_size, layer_size, bias=False)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        seq_len = x.shape[1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.layer_size)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)

        return torch.matmul(attention_weights, V)

torch.manual_seed(21)

if __name__ == "__main__":
    layers = 16
    batch = 2
    seq_len = 8

    attention = AttentionBlock(layers)
    x = torch.randn(batch, seq_len, layers)
    out = attention(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    if out.shape == x.shape:
        print("Shape test Passed")
    else:
        print("Shape test Failed")
