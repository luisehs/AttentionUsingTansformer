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

    print("\nRunning causal mask verification...")
    torch.manual_seed(0)
    x1 = torch.randn(1, seq_len, layers)
    x2 = x1.clone()
    x2[0, 5:, :] = torch.randn(seq_len - 5, layers)  # perturb positions 5+

    attention.eval()
    with torch.no_grad():
        out1 = attention(x1)
        out2 = attention(x2)

    # Positions 0-4 should be identical; positions 5+ may differ
    assert torch.allclose(out1[0, :5, :], out2[0, :5, :], atol=1e-6), \
        "Causal mask FAILED: early positions changed when future tokens were perturbed."
    print("  Causal mask verified: early positions unaffected by future token changes.")
    print("\nAll tests passed!")
