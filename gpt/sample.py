import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
import torch
from matplotlib.ticker import PercentFormatter

import model as gm
from model import GPTModel, text_to_tokenIDS, tokenIDS_to_text

# Ordinal blue ramp (light -> dark = low -> high temperature), validated light->dark.
TEMP_RAMP = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
MAX_TEMPS = 5
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # The model classes read the module-level cfg when built, so set it first.
    if "cfg" in ckpt:
        for k, v in ckpt["cfg"].items():
            setattr(gm.cfg, k, v)
    model = GPTModel().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.inference_mode()
def generate(
    model, idx, max_new_tokens, temperature=0.6, top_k=20, top_p=1.0, eos_id=None
):
    """idx: (B, T) token ids. temperature=0 is greedy; top_k=0 / top_p=1.0 disable them."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -gm.cfg.context_length :]
        # BATCH, N_TOKEN, VOCAB_SIZE -> BATCH, VOCAB_SIZE
        logits = model(idx_cond)[:, -1, :]

        if temperature == 0:
            # greedy:
            idx_next = logits.argmax(dim=-1, keepdim=True)
        else:
            # temperature scaling
            logits = logits / temperature

            # top-k: keep the k largest logits
            if top_k:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k).values[:, -1, None]  # (B, 1)
                logits = logits.masked_fill(logits < kth, -torch.inf)

            # top-p: keep the smallest set of tokens whose probability reaches top_p
            if top_p is not None and top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mass_before = cumsum - sorted_probs
                remove_sorted = mass_before >= top_p
                remove_sorted[:, 0] = (
                    False  # always keep the top token (top_p=0 -> greedy)
                )
                remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
                logits = logits.masked_fill(remove, -torch.inf)

            # sample after every filter has been applied
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and (idx_next == eos_id).all():
            break
        idx = torch.cat([idx, idx_next], dim=1)
    return idx


@torch.inference_mode()
def next_token_logits(model, idx):
    """Logits for the token after the prompt: (V,) on CPU."""
    return model(idx[:, -gm.cfg.context_length :])[0, -1, :].float().cpu()


def token_label(tokenizer, token_id):
    s = tokenizer.decode([token_id]).replace("\n", "\\n")
    return s.strip() or repr(s)


def ramp_colors(n):
    if n == 1:
        return [TEMP_RAMP[4]]
    return [TEMP_RAMP[round(i * (len(TEMP_RAMP) - 1) / (n - 1))] for i in range(n)]


def plot_temperatures(
    model, tokenizer, prompt, temperatures, top_n, device, save_path=None
):
    """Grouped bars: next-token probability of the top_n tokens at each temperature."""
    idx = text_to_tokenIDS(prompt, tokenizer).to(device)
    logits = next_token_logits(model, idx)

    # Dividing by T > 0 keeps the order of logits, so the top tokens are the
    # same at every temperature; only how much probability they get changes.
    top_ids = torch.topk(logits, top_n).indices
    labels = [token_label(tokenizer, t) for t in top_ids.tolist()]
    probs = {T: torch.softmax(logits / T, dim=-1)[top_ids] for T in temperatures}

    # Terminal table (same numbers as the chart).
    print(f"\nnext token after {prompt!r}")
    print(f"{'token':>14} " + " ".join(f"{f'T={T:g}':>8}" for T in temperatures))
    for j, label in enumerate(labels):
        print(
            f"{label:>14} "
            + " ".join(f"{probs[T][j].item():8.1%}" for T in temperatures)
        )

    x = torch.arange(top_n).numpy()
    group = 0.8
    width = group / len(temperatures)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.75 * top_n), 4.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (T, color) in enumerate(zip(temperatures, ramp_colors(len(temperatures)))):
        p = probs[T].numpy()
        ax.bar(
            x - group / 2 + (i + 0.5) * width,
            p,
            width,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,  # surface gap between bars
            label=f"T = {T:g}   (top {top_n} hold {p.sum():.0%})",
        )

    shown = prompt if len(prompt) <= 60 else "…" + prompt[-59:]
    ax.set_title(f"Next token after “{shown}”", loc="left", color=INK, fontsize=11)
    ax.set_ylabel("Probability", color=INK_2)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE)
        print(f"saved {save_path}")
    plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", help="omit for interactive mode")
    p.add_argument("--ckpt", default="results/tinystories-30m.pt")
    p.add_argument("-n", "--num-samples", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=50, help="0 = no top-k")
    p.add_argument("--top-p", type=float, default=1.0, help="1.0 = no top-p")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--plot",
        action="store_true",
        help="plot next-token probabilities at several temperatures instead of generating",
    )
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--top-n", type=int, default=10, help="tokens shown on the x axis")
    p.add_argument(
        "--no-save",
        action="store_true",
        help="show the plot without saving it to results/",
    )
    args = p.parse_args()

    if args.plot:
        if any(T <= 0 for T in args.temperatures):
            p.error("--temperatures must all be > 0")
        if len(args.temperatures) > MAX_TEMPS:
            p.error(f"at most {MAX_TEMPS} temperatures per plot")
        args.temperatures = sorted(args.temperatures)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    model, ckpt = load_model(args.ckpt, device)
    print(
        f"loaded {args.ckpt} (step {ckpt.get('step')}, val {ckpt.get('val_loss', 0):.3f}) on {device}"
    )

    def run(prompt):
        if args.plot:
            save_path = None
            if not args.no_save:
                slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:40]
                temps = "-".join(f"{T:g}" for T in args.temperatures)
                RESULTS_DIR.mkdir(exist_ok=True)
                save_path = RESULTS_DIR / f"temperatures_{slug}_T{temps}.png"
            plot_temperatures(
                model,
                tokenizer,
                prompt,
                args.temperatures,
                args.top_n,
                device,
                save_path,
            )
            return

        idx = text_to_tokenIDS(prompt, tokenizer).to(device)
        for i in range(args.num_samples):
            out = generate(
                model,
                idx,
                args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_id=tokenizer.eot_token,
            )
            if args.num_samples > 1:
                print(f"--- sample {i + 1}")
            print(tokenIDS_to_text(out.cpu(), tokenizer), "\n")

    if args.prompt:
        run(args.prompt)
        return

    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        run(prompt)


if __name__ == "__main__":
    main()
