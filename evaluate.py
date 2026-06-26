"""
evaluate.py — Zero-shot evaluation of a trained Mixtral/TokenFormer model.

Benchmarks:
  1. Perplexity on the Pile validation split
  2. HellaSwag commonsense reasoning accuracy
  3. WinoGrande pronoun/coreference accuracy

Usage:
  python evaluate.py --checkpoint checkpoints/model_base.pt
  python evaluate.py --checkpoint checkpoints/model_tokenformer.pt --tasks hellaswag winogrande
  python evaluate.py --checkpoint checkpoints/model_base.pt \\
      --pile_data_dir data/pile  # use pre-tokenised binary instead of streaming
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from mistral_inference.args import TransformerArgs
from mistral_inference.cache import BufferCache
from mistral_inference.transformer import Transformer

# --------------------------------------------------------------------------- #
#  Core scoring                                                               #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _forward_logprobs(
    model: Transformer,
    device: torch.device,
    token_seqs: List[List[int]],
) -> List[torch.Tensor]:
    """Teacher-forced token log-probs for a batch of (variable-length) sequences.

    Returns, for each input sequence of length L, a 1-D tensor of length L-1
    where element i is  log P(token[i+1] | token[:i+1]).
    """
    seqlens = [len(s) for s in token_seqs]
    flat = torch.tensor([t for s in token_seqs for t in s], device=device, dtype=torch.long)

    cache = BufferCache(
        model.n_local_layers,
        max_batch_size=len(token_seqs),
        max_seq_len=max(seqlens),
        n_kv_heads=model.args.n_kv_heads,
        head_dim=model.args.head_dim,
        sliding_window=model.args.sliding_window,
    )
    cache.to(device=device, dtype=model.dtype)
    cache.reset()

    logits = model.forward(flat, seqlens=seqlens, cache=cache)  # (sum_seqlens, vocab)
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    out: List[torch.Tensor] = []
    offset = 0
    for s, L in zip(token_seqs, seqlens):
        # row offset+i predicts the token at position i+1, for i in [0, L-2]
        targets = torch.tensor(s[1:], device=device, dtype=torch.long)
        rows = logprobs[offset : offset + L - 1]
        out.append(rows.gather(1, targets[:, None]).squeeze(1))
        offset += L
    return out


def _batches(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# --------------------------------------------------------------------------- #
#  Multiple-choice harness (HellaSwag / WinoGrande)                           #
# --------------------------------------------------------------------------- #


def _eval_multiple_choice(
    model: Transformer,
    device: torch.device,
    tokenizer,
    examples: List[Dict],
    batch_size: int,
    normalize: bool,
    desc: str,
) -> float:
    """Each example: {"context": str, "continuations": List[str], "label": int}.

    Scores every continuation by the (optionally length-normalised) sum of its
    token log-probs conditioned on the context, then counts argmax == label.
    """
    # Flatten to one scoring request per (example, continuation).
    seqs: List[List[int]] = []
    meta: List[Tuple[int, int, int]] = []  # (example_idx, choice_idx, n_continuation_tokens)
    for ei, ex in enumerate(examples):
        ctx_tokens = tokenizer.encode(ex["context"])  # includes BOS
        for ci, cont in enumerate(ex["continuations"]):
            cont_tokens = tokenizer.encode(cont, add_special_tokens=False)
            if len(cont_tokens) == 0:
                cont_tokens = [tokenizer.eos_token_id]
            seqs.append(ctx_tokens + cont_tokens)
            meta.append((ei, ci, len(cont_tokens)))

    scores: List[List[float]] = [[0.0] * len(ex["continuations"]) for ex in examples]

    n_batches = math.ceil(len(seqs) / batch_size)
    batch_meta = _batches(meta, batch_size)
    for bseqs, bmeta in tqdm(
        zip(_batches(seqs, batch_size), batch_meta), total=n_batches, desc=desc, leave=False
    ):
        token_lps = _forward_logprobs(model, device, bseqs)
        for (ei, ci, clen), tlp in zip(bmeta, token_lps):
            total = tlp[-clen:].sum().item()
            scores[ei][ci] = total / clen if normalize else total

    correct = sum(
        int(np.argmax(sc)) == ex["label"] for ex, sc in zip(examples, scores)
    )
    return correct / len(examples)


# --------------------------------------------------------------------------- #
#  Task 1: Perplexity on the Pile validation split                            #
# --------------------------------------------------------------------------- #

def eval_pile_perplexity(
    model: Transformer,
    device: torch.device,
    tokenizer,
    seq_len: int = 512,
    max_tokens: int = 2_000_000,
    data_dir: Optional[str] = None,
    pile_dataset: str = "EleutherAI/the_pile_deduplicated",
) -> float:
    # Gather a flat stream of up to `max_tokens` token ids.
    if data_dir is not None:
        path = Path(data_dir) / "val.bin"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (pass --pile_data_dir with a val.bin)")
        data = np.memmap(path, dtype=np.uint16, mode="r")
        tokens = np.asarray(data[:max_tokens], dtype=np.int64)
    else:
        from datasets import load_dataset

        stream = load_dataset(pile_dataset, split="train", streaming=True)
        eos = tokenizer.eos_token_id
        buf: List[int] = []
        for ex in stream:
            buf.extend(tokenizer.encode(ex["text"], add_special_tokens=False))
            buf.append(eos)
            if len(buf) >= max_tokens:
                break
        tokens = np.asarray(buf[:max_tokens], dtype=np.int64)

    # Non-overlapping windows; each is scored as an independent sequence.
    windows = [
        tokens[i : i + seq_len].tolist()
        for i in range(0, len(tokens) - 1, seq_len)
    ]
    windows = [w for w in windows if len(w) >= 2]

    batch_size = max(1, model.args.max_batch_size)
    total_nll = 0.0
    total_count = 0
    for bseqs in tqdm(
        _batches(windows, batch_size),
        total=math.ceil(len(windows) / batch_size),
        desc="pile",
        leave=False,
    ):
        for tlp in _forward_logprobs(model, device, bseqs):
            total_nll += -tlp.sum().item()
            total_count += tlp.numel()

    return math.exp(total_nll / total_count)


# --------------------------------------------------------------------------- #
#  Task 2: HellaSwag                                                           #
# --------------------------------------------------------------------------- #

def eval_hellaswag(
    model: Transformer,
    device: torch.device,
    tokenizer,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
) -> float:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    examples: List[Dict] = []
    for row in ds:
        context = (row["activity_label"] + ": " + row["ctx"]).strip()
        examples.append({
            "context": context,
            "continuations": [" " + e.strip() for e in row["endings"]],
            "label": int(row["label"]),
        })
    # length-normalised score (acc_norm) — continuations vary a lot in length
    return _eval_multiple_choice(model, device, tokenizer, examples, batch_size, normalize=True, desc="hellaswag")


# --------------------------------------------------------------------------- #
#  Task 3: WinoGrande                                                          #
# --------------------------------------------------------------------------- #

def eval_winogrande(
    model: Transformer,
    device: torch.device,
    tokenizer,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
) -> float:
    from datasets import load_dataset

    # winogrande ships as a (now-unsupported) dataset script; load the Hub's
    # auto-generated parquet branch instead.  That branch exposes only a single
    # "default" config, but the validation split is identical across all the
    # size configs (they differ only in training-set size), so it's what we want.
    ds = load_dataset(
        "allenai/winogrande", revision="refs/convert/parquet", split="validation"
    )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # WinoGrande: fill the blank with each option, then compare the likelihood
    # of the (identical) suffix conditioned on each filled prefix.  Because the
    # suffix is shared between options, no length normalisation is needed.
    examples: List[Dict] = []
    for row in ds:
        prefix, suffix = row["sentence"].split("_", 1)
        examples.append({
            "context": (prefix + row["option1"], prefix + row["option2"]),
            "suffix": suffix,
            "label": int(row["answer"]) - 1,  # answer is "1" / "2"
        })

    # Score each option's suffix separately (contexts differ, continuation shared).
    seqs: List[List[int]] = []
    meta: List[Tuple[int, int, int]] = []  # (example_idx, option_idx, n_suffix_tokens)
    for ei, ex in enumerate(examples):
        cont_tokens = tokenizer.encode(ex["suffix"], add_special_tokens=False)
        if len(cont_tokens) == 0:
            cont_tokens = [tokenizer.eos_token_id]
        for oi, ctx in enumerate(ex["context"]):
            seqs.append(tokenizer.encode(ctx) + cont_tokens)
            meta.append((ei, oi, len(cont_tokens)))

    scores = [[0.0, 0.0] for _ in examples]
    n_batches = math.ceil(len(seqs) / batch_size)
    for bseqs, bmeta in tqdm(
        zip(_batches(seqs, batch_size), _batches(meta, batch_size)),
        total=n_batches, desc="winogrande", leave=False,
    ):
        for (ei, oi, clen), tlp in zip(bmeta, _forward_logprobs(model, device, bseqs)):
            scores[ei][oi] = tlp[-clen:].sum().item()

    correct = sum(int(np.argmax(sc)) == ex["label"] for ex, sc in zip(examples, scores))
    return correct / len(examples)


TASK_NAMES = ["pile", "hellaswag", "winogrande"]


# --------------------------------------------------------------------------- #
#  Checkpoint loading + CLI                                                    #
# --------------------------------------------------------------------------- #

def load_model_from_checkpoint(
    checkpoint: str, device: torch.device, max_batch_size: int
) -> Transformer:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args: TransformerArgs = ckpt["args"]
    args.max_batch_size = max(args.max_batch_size, max_batch_size)

    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = Transformer(args)
    torch.set_default_dtype(orig_dtype)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device=device)
    model.eval()
    print(f"Loaded checkpoint {checkpoint} (step {ckpt.get('step', '?')})")
    return model


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of a trained model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=TASK_NAMES)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap examples per multiple-choice task (None = full set)")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Window length for Pile perplexity")
    parser.add_argument("--pile_max_tokens", type=int, default=2_000_000)
    parser.add_argument("--pile_data_dir", type=str, default=None,
                        help="Directory with a pre-tokenised val.bin (else stream from HF)")
    parser.add_argument("--pile_dataset", type=str, default="EleutherAI/the_pile_deduplicated")
    parser.add_argument("--output", type=str, default=None, help="Write results JSON here")
    cfg = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(cfg.checkpoint, device, cfg.batch_size)
    model.args.max_batch_size = cfg.batch_size

    from transformers import AutoTokenizer
    print(f"Loading tokenizer {cfg.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    results: Dict[str, float] = {}
    with torch.no_grad():
        if "pile" in cfg.tasks:
            results["pile_perplexity"] = eval_pile_perplexity(
                model, device, tokenizer,
                seq_len=cfg.seq_len, max_tokens=cfg.pile_max_tokens,
                data_dir=cfg.pile_data_dir, pile_dataset=cfg.pile_dataset,
            )
        if "hellaswag" in cfg.tasks:
            results["hellaswag_accuracy"] = eval_hellaswag(
                model, device, tokenizer, cfg.batch_size, cfg.max_samples)
        if "winogrande" in cfg.tasks:
            results["winogrande_accuracy"] = eval_winogrande(
                model, device, tokenizer, cfg.batch_size, cfg.max_samples)

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"  {k:>22}: {v:.4f}")

    if cfg.output:
        with open(cfg.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {cfg.output}")


if __name__ == "__main__":
    main()
