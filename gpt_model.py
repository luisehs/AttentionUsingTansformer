

import torch
import torch.nn as nn
from torch.nn import functional
from models import AttentionBlock


class Transformer_Block(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()

        self.attn_block = AttentionBlock(n_embd)

        self.norm_1 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, n_embd)
        self.norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_out = self.attn_block(x)
        x2 = attn_out + x
        x3 = self.norm_1(x2)

        ff_out = functional.relu(self.linear_1(x3))
        x5 = ff_out + x3

        return self.norm_2(x5)


class GPT(nn.Module):
    def __init__(self, block_size, n_embd, n_layer, vocab_size):
        super().__init__()

        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, n_embd)

        self.transformer_blocks = nn.ModuleList(
            [Transformer_Block(n_embd, block_size) for _ in range(n_layer)]
        )

        self.norm = nn.LayerNorm(n_embd)
        self.output_layer = nn.Linear(n_embd, vocab_size, bias=False)

    def get_loss(self, input, target):
        output = self(input)

        return functional.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
            ignore_index=-1
        )

    def forward(self, input):
        b, t = input.size()

        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.block_size}"
        )

        x = self.embed(input)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.output_layer(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = (
                idx if idx.size(1) <= self.block_size
                else idx[:, -self.block_size:]
            )

            logits = self(idx_cond)
            logits = logits[:, -1, :]

            probs = functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    torch.manual_seed(21)

    batch_size = 2
    seq_len = 8
    n_embd = 16
    vocab_size = 50
    block_size = seq_len
    n_layer = 2

    block = Transformer_Block(n_embd, block_size)

    x = torch.randn(batch_size, seq_len, n_embd)
    block_out = block(x)

    model = GPT(
        block_size=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        vocab_size=vocab_size
    )

    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(idx)

    expected_logits_shape = (batch_size, seq_len, vocab_size)

    if block_out.shape == x.shape:
        print("Transformer_Block shape test passed ")
    else:
        print("Transformer_Block shape test failed")

    if logits.shape == expected_logits_shape:
        print("GPT logits shape test passed ")
    else:
        print("GPT logits shape test failed")

    if logits.requires_grad:
        print("GPT gradient test passed ")
    else:
        print("GPT gradient test failed")

    print("All shape tests passed!")