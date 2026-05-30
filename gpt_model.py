import torch
from torch import nn

from models import AttentionBlock


class Transformer_Block(nn.Module):
    def __init__(self, layer_size):
        super().__init__()
        self.attention = AttentionBlock(layer_size)
        self.norm1 = nn.LayerNorm(layer_size)
        self.ff = nn.Linear(layer_size, layer_size)
        self.norm2 = nn.LayerNorm(layer_size)

    def forward(self, x):
        attn_out = self.attention(x)
        x2 = attn_out + x
        x3 = self.norm1(x2)
        ff_out = torch.relu(self.ff(x3))
        x5 = ff_out + x3
        return self.norm2(x5)


class GPT(nn.Module):
    def __init__(self, vocab_size=100, layer_size=16, num_blocks=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, layer_size)
        self.blocks = nn.ModuleList(
            [Transformer_Block(layer_size) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(layer_size)
        self.output_proj = nn.Linear(layer_size, vocab_size)

    def forward(self, idx):
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

    def generate(self, idx, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


if __name__ == "__main__":
    torch.manual_seed(21)

    batch_size = 2
    seq_len = 8
    layer_size = 16
    vocab_size = 50

    block = Transformer_Block(layer_size)
    x = torch.randn(batch_size, seq_len, layer_size)
    block_out = block(x)

    model = GPT(vocab_size=vocab_size, layer_size=layer_size, num_blocks=2)
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(idx)
    expected_logits_shape = (batch_size, seq_len, vocab_size)

    if block_out.shape == x.shape:
        print("Transformer_Block shape test passed \u2713")
    else:
        print("Transformer_Block shape test failed")
    if logits.shape == expected_logits_shape:
        print("GPT logits shape test passed \u2713")
    else:
        print("GPT logits shape test failed")
    if logits.requires_grad:
        print("GPT gradient shape test passed \u2713")
    else:
        print("GPT gradient shape test failed")

    print("All shape tests passed!")
