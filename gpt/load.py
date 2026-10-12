"""Load OpenAI's pretrained GPT-2 weights into our GPTModel.

  python gpt/load.py                    # gpt2 (124M): load + greedy sample
  python gpt/load.py gpt2-medium "The meaning of life is"

- Linear layers are `Conv1D` with weight shape (in, out), computing x @ W + b.
  nn.Linear stores (out, in) and computes x @ W.T + b, so every matrix is transposed.
- q, k, v are fused into one `c_attn` matrix of shape (d, 3d); we split it.
- `h.N.attn.bias` is the causal-mask buffer, not a weight; skipped.
- There is no separate lm_head: the output layer is tied to `wte`.
"""

import argparse

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from config import GPT2_PRESETS, GptConfig, gpt2_config
from model import GPTModel


def download_gpt2(name: str = "gpt2") -> dict[str, torch.Tensor]:
    """HF state dict for a GPT-2 checkpoint. Downloads once, then reads the HF cache."""
    path = hf_hub_download(repo_id=name, filename="model.safetensors")
    return load_file(path)


def convert_hf_gpt2(
    hf_sd: dict[str, torch.Tensor], cfg: GptConfig
) -> dict[str, torch.Tensor]:
    """Map HF GPT-2 names/layouts onto our module names."""
    sd = {k.removeprefix("transformer."): v for k, v in hf_sd.items()}
    d = cfg.embed_dim

    out = {
        "tok_embd.weight": sd["wte.weight"],
        "pos_embd.weight": sd["wpe.weight"],
        "norm.scale": sd["ln_f.weight"],
        "norm.shift": sd["ln_f.bias"],
    }

    for i in range(cfg.n_layers):
        h, b = f"h.{i}.", f"trf_blocks.{i}."

        # (d, 3d) -> three (d, d) blocks [q | k | v]; biases (3d,) -> three (d,)
        q_w, k_w, v_w = sd[h + "attn.c_attn.weight"].split(d, dim=1)
        q_b, k_b, v_b = sd[h + "attn.c_attn.bias"].split(d, dim=0)

        out |= {
            b + "norm1.scale": sd[h + "ln_1.weight"],
            b + "norm1.shift": sd[h + "ln_1.bias"],
            b + "att.query_proj.weight": q_w.T,
            b + "att.query_proj.bias": q_b,
            b + "att.key_proj.weight": k_w.T,
            b + "att.key_proj.bias": k_b,
            b + "att.value_proj.weight": v_w.T,
            b + "att.value_proj.bias": v_b,
            b + "att.out_proj.weight": sd[h + "attn.c_proj.weight"].T,
            b + "att.out_proj.bias": sd[h + "attn.c_proj.bias"],
            b + "norm2.scale": sd[h + "ln_2.weight"],
            b + "norm2.shift": sd[h + "ln_2.bias"],
            b + "ff.layers.0.weight": sd[h + "mlp.c_fc.weight"].T,
            b + "ff.layers.0.bias": sd[h + "mlp.c_fc.bias"],
            b + "ff.layers.2.weight": sd[h + "mlp.c_proj.weight"].T,
            b + "ff.layers.2.bias": sd[h + "mlp.c_proj.bias"],
        }
    return out


def load_gpt2(name: str = "gpt2", device: str | torch.device = "cpu") -> GPTModel:
    cfg = gpt2_config(name)
    model = GPTModel(cfg)
    sd = convert_hf_gpt2(download_gpt2(name), cfg)

    # load_state_dict checks every shape, so a missing transpose fails loudly here.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # out.weight is tied to tok_embd.weight; the causal masks are built in __init__.
    missing = [k for k in missing if k != "out.weight" and not k.endswith(".att.mask")]
    if missing or unexpected:
        raise RuntimeError(
            f"weight mismatch: missing={missing} unexpected={unexpected}"
        )
    assert model.out.weight is model.tok_embd.weight, "output layer must stay tied"

    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", nargs="?", default="gpt2", choices=list(GPT2_PRESETS))
    p.add_argument("prompt", nargs="?", default="The meaning of life is")
    p.add_argument("--max-new-tokens", type=int, default=30)
    args = p.parse_args()

    import tiktoken

    from model import text_to_tokenIDS, tokenIDS_to_text
    from sample import generate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_gpt2(args.name, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {n_params / 1e6:.1f}M params on {device}")

    tokenizer = tiktoken.get_encoding("gpt2")
    idx = text_to_tokenIDS(args.prompt, tokenizer).to(device)
    out = generate(model, idx, args.max_new_tokens, temperature=0)  # greedy
    print(tokenIDS_to_text(out.cpu(), tokenizer))


if __name__ == "__main__":
    main()
