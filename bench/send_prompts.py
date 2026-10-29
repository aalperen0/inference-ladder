"""Send your own text prompts to a running server all at once, so the engine has a queue to
schedule, and print each completion in the order it finished.

    python3 server/api.py --policy priority --slots 2 --log-events   # terminal 1
    python3 bench/send_prompts.py bench/prompts.txt                   # terminal 2

The prompts file has one prompt per line, written `priority | prompt`. 0 is the highest
priority, and a line without "|" gets 0. Blank lines and lines starting with # are skipped.
Write \\n for a newline inside a prompt.

The requests go out concurrently. One sent only after the previous one finished would
never queue, and the policy would have nothing to decide.
"""

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class Result:
    prompt: str
    priority: int
    done_at: float  # seconds after all were sent
    id: str | None  # cmpl-<n>, as r<n> in the server's --log-events lines
    text: str = ""  # the completion
    error: str | None = None  # why the server refused it (a 400 rejected has an id too)


def read_prompts(path: str | Path) -> list[tuple[str, int]]:
    """(prompt, priority) per line of the file."""
    prompts = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        priority, sep, prompt = line.partition("|")
        if not sep:
            priority, prompt = "0", line
        try:
            prompts.append((prompt.strip().replace("\\n", "\n"), int(priority)))
        except ValueError:
            raise SystemExit(
                f"{path}:{n}: expected `priority | prompt`, with an integer priority; "
                f"got {line!r}"
            )
    return prompts


async def send_all(
    url: str, prompts: list[tuple[str, int]], max_tokens: int, ignore_eos: bool = False
) -> list[Result]:
    """Sends every prompt at once and returns the results in the order they finished."""

    async def complete(client, prompt, priority):
        body = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "priority": priority,
            "ignore_eos": ignore_eos,
        }
        r = await client.post("/v1/completions", json=body)
        done_at = time.perf_counter() - t0
        out = r.json()
        if r.status_code != 200:
            error = (
                out["error"]["message"] if "error" in out else str(out["detail"])
            )  # 422
            return Result(prompt, priority, done_at, out.get("id"), error=error)
        return Result(
            prompt, priority, done_at, out["id"], text=out["choices"][0]["text"]
        )

    limits = httpx.Limits(max_connections=None)
    async with httpx.AsyncClient(base_url=url, timeout=None, limits=limits) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(complete(client, p, prio) for p, prio in prompts)
        )
    return sorted(results, key=lambda r: r.done_at)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "prompts", nargs="?", default=Path(__file__).with_name("prompts.txt")
    )
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument(
        "--ignore-eos", action="store_true", help="always make all --max-tokens"
    )
    args = p.parse_args()

    prompts = read_prompts(args.prompts)
    try:
        results = asyncio.run(
            send_all(args.url, prompts, args.max_tokens, args.ignore_eos)
        )
    except httpx.ConnectError:
        raise SystemExit(f"no server at {args.url}: start python3 server/api.py first")

    print(f"{len(prompts)} prompts sent at once; in the order they finished:\n")
    print(f"{'done at':>8}  {'id':>8}  prio  prompt → completion")
    for r in results:
        if r.error is None:
            shown = f"{oneline(r.prompt)} → {oneline(r.text)}"
        else:
            shown = f"{oneline(r.prompt)}  ✗ {r.error}"
        print(f"{r.done_at:6.2f} s  {r.id or '-':>8}  p{r.priority:<3} {shown}")


def oneline(s: str) -> str:
    return s.replace("\n", "\\n")


if __name__ == "__main__":
    main()
