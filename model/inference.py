from __future__ import annotations

import argparse
import os
import re
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Reuse model definition from training
from train import Configuration, Model, load_checkpoint


DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "out", "best_model1.pt")
DEFAULT_TOKENIZER = os.path.join(
    os.path.dirname(__file__), "..", "bpe", "artifacts", "institutional_books_ocr_gt_90_pre1940_50304", "tokenizer.json"
)

# End of sentence: . ! or ? (optionally followed by a closing quote/bracket)
SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*")

# Hidden context for --frame: primes the model to write short period aphorisms
FRAMING_PREFIX = (
    "MAXIMS AND REFLECTIONS.\n\n"
    "Knowledge is power.\n\n"
    "Friendship is the wine of life.\n\n"
    "Silence is the element in which great things fashion themselves together.\n\n"
    "The unexamined life is not worth living.\n\n"
)


def build_config(ckpt: dict) -> Configuration:
    """Model size from the checkpoint's config dict; the 124M checkpoints have none, so they get the default."""
    return Configuration(**ckpt["config"]) if "config" in ckpt else Configuration()


@torch.no_grad()
def generate_sentence(
    model: Model,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 30,
    eos_id: int | None = None,
    no_repeat_ngram: int = 0,
) -> str:
    """Sample tokens until the continuation ends a sentence; return prompt + that sentence.

    no_repeat_ngram=n bans any n-token sequence from occurring twice in the full context,
    hidden prefix included (so the prefix can't be copied either).
    """
    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    block_size = model.config.block_size
    new_ids: list[int] = []

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)  # (1, 1, vocab)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if no_repeat_ngram > 1:
            # Slide an n-token window: if the last n-1 tokens occurred before,
            # forbid whatever followed them then
            seq = idx[0].tolist()
            n = no_repeat_ngram
            tail = seq[len(seq) - (n - 1) :]
            banned = [seq[i + n - 1] for i in range(len(seq) - n + 1) if seq[i : i + n - 1] == tail]
            logits[0, banned] = float("-inf")

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        if eos_id is not None and next_id.item() == eos_id:
            break
        idx = torch.cat([idx, next_id], dim=1)
        new_ids.append(next_id.item())

        # Only search the generated part, so a period in the prompt doesn't stop us
        match = SENTENCE_END.search(tokenizer.decode(new_ids))
        if match:
            text = tokenizer.decode(prompt_ids + new_ids)
            # A token can carry text past the terminator (e.g. ". The"); cut it off
            overshoot = len(tokenizer.decode(new_ids)) - match.end()
            return text[: len(text) - overshoot].strip()

    # No sentence end (EOS or token cap): cut at the last ; or , in the generated part
    text = tokenizer.decode(prompt_ids + new_ids)
    new_text = tokenizer.decode(new_ids)
    cut = max(new_text.rfind(";"), new_text.rfind(","))
    if cut != -1:
        text = text[: len(text) - len(new_text) + cut]
    return text.strip() + "."


@torch.no_grad()
def score_sentence(model: Model, tokenizer: Tokenizer, prompt_ids: list[int], sentence: str) -> float:
    """Mean log-prob per token the model assigns to the continuation (higher = more fluent)."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(sentence).ids
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    y[:, : len(prompt_ids) - 1] = -1  # ignore_index: don't score the prompt itself
    _, loss = model(x, target=y)
    return -loss.item()


def repeats_phrase(sentence: str, n: int = 3) -> bool:
    """True if any n-word phrase occurs more than once (log-prob rewards such loops)."""
    words = re.findall(r"\w+", sentence.lower())
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(ngrams) != len(set(ngrams))


def main():
    parser = argparse.ArgumentParser(description="Generate one sentence with the trained vintage LLM.")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Path to checkpoint .pt")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, default="We can be very happy by", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Hard cap if no sentence end is produced")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--num-samples", type=int, default=8, help="Sample N sentences, print the highest-scoring")
    parser.add_argument("--verbose", action="store_true", help="Print every candidate with its score to stderr")
    parser.add_argument("--frame", action="store_true", help="Prepend hidden maxims to steer the style")
    parser.add_argument("--no-repeat-ngram", type=int, default=3, help="Ban repeating any n-token sequence (0 = off)")
    parser.add_argument("--max-words", type=int, default=12, help="Keep only continuations this short (0 = off)")
    parser.add_argument("--max-tries", type=int, default=64, help="Give up collecting short candidates after this many")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = load_checkpoint(args.ckpt, device)
    model = Model(build_config(ckpt)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.get_vocab().get("<|endoftext|>")

    prefix = FRAMING_PREFIX if args.frame else ""
    prompt_ids = tokenizer.encode(prefix + args.prompt).ids
    if not args.prompt or not prompt_ids:
        print("empty prompt after tokenization; provide a non-empty --prompt", file=sys.stderr)
        sys.exit(1)

    def is_short(sentence: str) -> bool:
        return not args.max_words or len(sentence.split()) - len(args.prompt.split()) <= args.max_words

    # Long answers wander off topic, so resample until num_samples short ones are collected
    candidates, tries = [], 0
    while sum(is_short(c[1]) for c in candidates) < args.num_samples and tries < args.max_tries:
        tries += 1
        sentence = generate_sentence(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            eos_id=eos_id,
            no_repeat_ngram=args.no_repeat_ngram,
        )
        score = score_sentence(model, tokenizer, prompt_ids, sentence)
        candidates.append((score, sentence.removeprefix(prefix)))

    # Drop long ones and loops before ranking; fall back to the full pool if nothing survives
    kept = [c for c in candidates if is_short(c[1]) and not repeats_phrase(c[1])] or candidates

    if args.verbose:
        for score, sentence in sorted(candidates, reverse=True):
            mark = " " if (score, sentence) in kept else "x"
            print(f"{mark} {score:7.3f}  {sentence}", file=sys.stderr)
    best = " ".join(max(kept)[1].split())  # the sentence may span a paragraph break
    if best.endswith(('"', "”")) and (best.count('"') + best.count("“") + best.count("”")) % 2:  # unopened quote
        best = best[:-1]
    print(best)


if __name__ == "__main__":
    main()
