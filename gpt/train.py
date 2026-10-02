import argparse
import dataclasses
import math
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GPTDataset, GPTModel, cfg, text_to_tokenIDS, tokenIDS_to_text


def create_dataloader_v1(
    txt: str,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return data_loader


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    logits = model(input_batch)  # B,T,V
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Mean loss over (up to max_batches of) a loader. Restores train mode."""
    was_training = model.training
    model.eval()

    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        total += calc_loss_batch(x, y, model, device).item()
        n += 1

    model.train(was_training)
    return total / n if n else float("nan")


@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens, context_length):
    """idx: (B, T) token ids. Greedy decoding, no KV cache."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def sample_text(model, tokenizer, device, prompt, max_new_tokens=50):
    was_training = model.training
    model.eval()
    idx = text_to_tokenIDS(prompt, tokenizer).to(device)
    out = generate_greedy(model, idx, max_new_tokens, cfg.context_length)
    model.train(was_training)
    return tokenIDS_to_text(out.cpu(), tokenizer).replace("\n", " ")


def make_optimizer(model, lr, weight_decay):
    # Decay only matrices (Linear / Embedding weights); never biases or LayerNorm params.
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available()
    )


def lr_at(step, max_steps, warmup_steps, max_lr, min_lr):
    """Linear warmup, then cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(
    model,
    train_loader,
    val_loader,
    device,
    tokenizer,
    epochs=10,
    max_lr=4e-4,
    min_lr=4e-5,
    weight_decay=0.1,
    warmup_steps=100,
    grad_accum_steps=1,
    grad_clip=1.0,
    eval_every=100,
    eval_batches=20,
    sample_prompt="Every effort moves you",
    ckpt_path=None,
):
    optimizer = make_optimizer(model, max_lr, weight_decay)
    max_steps = epochs * len(train_loader) // grad_accum_steps

    history = {
        "step": [], "tokens": [], "train_loss": [], "val_loss": [],
        "epoch_train_loss": [], "epoch_val_loss": [],
    }
    best_val = float("inf")
    tokens_seen, step = 0, 0
    t0 = time.perf_counter()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()
        step_loss = torch.zeros((), device=device)  # summed on GPU, no sync per batch
        epoch_loss_sum, epoch_steps = 0.0, 0

        bar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for micro, (x, y) in enumerate(bar):
            loss = calc_loss_batch(x, y, model, device)
            (loss / grad_accum_steps).backward()
            step_loss += loss.detach()
            tokens_seen += x.numel()

            if (micro + 1) % grad_accum_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            lr = lr_at(step, max_steps, warmup_steps, max_lr, min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            epoch_loss_sum += step_loss.item() / grad_accum_steps
            epoch_steps += 1
            step_loss.zero_()
            bar.set_postfix(
                loss=f"{epoch_loss_sum / epoch_steps:.3f}",
                lr=f"{lr:.1e}",
                tok_s=f"{tokens_seen / (time.perf_counter() - t0):,.0f}",
                refresh=False,
            )

            if step % eval_every == 0:
                train_loss = evaluate(model, train_loader, device, eval_batches)
                val_loss = evaluate(model, val_loader, device, eval_batches)
                history["step"].append(step)
                history["tokens"].append(tokens_seen)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                tqdm.write(
                    f"  step {step:5d} | lr {lr:.2e} | "
                    f"train {train_loss:.3f} val {val_loss:.3f}"
                )

                if ckpt_path and val_loss < best_val:
                    best_val = val_loss
                    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "cfg": dataclasses.asdict(cfg),
                            "step": step,
                            "val_loss": val_loss,
                        },
                        ckpt_path,
                    )

        bar.close()

        epoch_train = epoch_loss_sum / max(epoch_steps, 1)
        epoch_val = evaluate(model, val_loader, device, eval_batches)
        history["epoch_train_loss"].append(epoch_train)
        history["epoch_val_loss"].append(epoch_val)
        mins = (time.perf_counter() - epoch_t0) / 60

        tqdm.write(
            f"=== epoch {epoch + 1}/{epochs} done in {mins:.1f} min | "
            f"train {epoch_train:.3f} | val {epoch_val:.3f}"
        )
        tqdm.write(f"    sample: {sample_text(model, tokenizer, device, sample_prompt)}")

    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tinystories_100mb.txt")
    p.add_argument(
        "--val-data",
        default="data/tinystories_valid.txt",
        help="separate validation file; if missing, split --data by --val-frac",
    )
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--ckpt", default="results/train_best.pt")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    text = Path(args.data).read_text(encoding="utf-8")
    if args.val_data and Path(args.val_data).exists():
        train_text = text
        val_text = Path(args.val_data).read_text(encoding="utf-8")
    else:
        split = int(len(text) * (1 - args.val_frac))
        train_text, val_text = text[:split], text[split:]

    train_loader = create_dataloader_v1(
        train_text,
        args.batch_size,
        args.context,
        stride=args.context,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_text,
        args.batch_size,
        args.context,
        stride=args.context,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    print(f"train batches {len(train_loader)} | val batches {len(val_loader)}")

    model = GPTModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.1f}M on {device}")

    train(
        model,
        train_loader,
        val_loader,
        device,
        tokenizer,
        epochs=args.epochs,
        max_lr=args.lr,
        grad_accum_steps=args.grad_accum,
        sample_prompt="Once upon a time",
        ckpt_path=args.ckpt,
    )


if __name__ == "__main__":
    main()
