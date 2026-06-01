from datetime import datetime

import torch
import torch.nn.functional as F

from gpt_model import GPT


BLOCK_SIZE = 64
LAYER_SIZE = 64
N_LAYERS = 2
LEARNING_RATE = 3e-4
MAX_ITERS = 3000
BATCH_SIZE = 32

PRINT_EVERY = 300
SAMPLE_TOKENS = 500


def train_and_generate(
    n_layers=N_LAYERS,
    layer_size=LAYER_SIZE,
    max_iters=MAX_ITERS,
    input_file="input.txt",
    log_file="training_log.txt",
):
    torch.manual_seed(21)
    start_dt = datetime.now()
    start_time = start_dt.strftime("%Y-%m-%d %H:%M:%S")

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(indices):
        return "".join(itos[i] for i in indices)

    data = torch.tensor(encode(text), dtype=torch.long)
    model = GPT(block_size=BLOCK_SIZE, vocab_size=vocab_size, n_embd=layer_size, n_layer=n_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    def get_batch():
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
        return x, y

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Training log\n")
        f.write(f"START_TIME\t{start_time}\n")
        f.write(f"INPUT_FILE\t{input_file}\n")
        f.write(f"CHARACTERS\t{len(text)}\n")
        f.write(f"VOCAB_SIZE\t{vocab_size}\n")
        f.write(f"BLOCK_SIZE\t{BLOCK_SIZE}\n")
        f.write(f"LAYER_SIZE\t{layer_size}\n")
        f.write(f"N_LAYERS\t{n_layers}\n")
        f.write(f"LEARNING_RATE\t{LEARNING_RATE}\n")
        f.write(f"MAX_ITERS\t{max_iters}\n")
        f.write(f"BATCH_SIZE\t{BATCH_SIZE}\n\n")
        f.write("step\tloss\n")

    print("step\tloss")
    last_logged_step = None
    for step in range(max_iters):
        display_step = step + 1
        xb, yb = get_batch()
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if display_step == 1 or display_step % PRINT_EVERY == 0:
            loss_value = loss.item()
            print(f"{display_step}\t{loss_value:.4f}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{display_step}\t{loss_value:.4f}\n")
            last_logged_step = display_step

    final_step = max_iters
    if last_logged_step != final_step:
        final_loss = loss.item()
        print(f"{final_step}\t{final_loss:.4f}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{final_step}\t{final_loss:.4f}\n")

    start = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start, max_new_tokens=SAMPLE_TOKENS)
    generated_text = decode(generated[0].tolist())

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\nGenerated sample\n")
        f.write(generated_text + "\n")

    return generated_text


if __name__ == "__main__":
    run_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = f"training_log_{run_datetime}.txt"
    # generated_text = train_and_generate(
    #     n_layers=N_LAYERS,
    #     layer_size=LAYER_SIZE,
    #     max_iters=MAX_ITERS,
    #     input_file="input.txt",
    #     log_file=log_file,
    # )
    # print("\nGenerated sample:")
    # print(generated_text)

    experiments = [
        # {
        #     "n_layers": N_LAYERS + 2,
        #     "layer_size": LAYER_SIZE,
        #     "max_iters": MAX_ITERS,
        # },
        # {
        #     "n_layers": N_LAYERS,
        #     "layer_size": LAYER_SIZE * 2,
        #     "max_iters": MAX_ITERS,
        # },
        # {
        #     "n_layers": N_LAYERS,
        #     "layer_size": LAYER_SIZE,
        #     "max_iters": MAX_ITERS * 2,
        # },
        {
            "n_layers": N_LAYERS + 2,
            "layer_size": LAYER_SIZE * 2,
            "max_iters": MAX_ITERS *2,
        },
    ]

    for experiment in experiments:
        experiment_log = (
            f"training_log_{run_datetime}_"
            f"layers{experiment['n_layers']}_"
            f"size{experiment['layer_size']}_"
            f"iters{experiment['max_iters']}.txt"
        )
        print(
            "\nRunning experiment: "
            f"N_LAYERS={experiment['n_layers']}, "
            f"LAYER_SIZE={experiment['layer_size']}, "
            f"MAX_ITERS={experiment['max_iters']}"
        )
        experiment_text = train_and_generate(
            n_layers=experiment["n_layers"],
            layer_size=experiment["layer_size"],
            max_iters=experiment["max_iters"],
            input_file="input.txt",
            log_file=experiment_log,
        )
        print("\nGenerated sample:")
        print(experiment_text)
