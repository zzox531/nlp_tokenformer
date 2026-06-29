"""
Streaming dataloader for training.

The training data is consumed straight from the HuggingFace stream — nothing is
downloaded or tokenized to disk ahead of time. Documents are streamed (and, for
the train split, shuffled with a reservoir buffer), tokenized on the fly, and
packed into contiguous `seq_len + 1` windows.

A held-out validation split is carved out deterministically by reserving the
first `val_docs` streamed documents for "val" and skipping them for "train", so
the two never overlap.

The `prepare()` helper below is retained only for producing the pre-tokenized
Pile `val.bin` used by the perplexity eval; it is not needed for training.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import torch


class StreamingTextDataLoader:
    """
    Packs a shuffled stream of documents into (batch_size, seq_len) batches.

    One or more HuggingFace datasets are streamed and (for the train split)
    interleaved with given mixing probabilities — e.g. an OpenWebText + Pile
    mix. Tokens from successive documents are concatenated (with EOS separators)
    into a buffer; each call to get_batch() peels off batch_size * (seq_len + 1)
    tokens and reshapes them into next-token (inputs, targets) pairs. The stream
    is restarted when exhausted so training can run for arbitrarily many steps.

    Each source reserves its first `val_docs` documents for the held-out val
    split; the train split skips them, so train/val never overlap per source.
    """

    def __init__(
        self,
        split: Literal["train", "val"],
        seq_len: int,
        batch_size: int,
        device: torch.device,
        dataset_names: Sequence[str] = ("Skylion007/openwebtext",),
        mix_probs: Optional[Sequence[float]] = None,
        tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
        shuffle_buffer: int = 10_000,
        val_docs: int = 2_000,
        seed: int = 42,
        text_key: str = "text",
    ):
        from transformers import AutoTokenizer

        self.split = split
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.dataset_names = list(dataset_names)
        if mix_probs is not None:
            assert len(mix_probs) == len(self.dataset_names), \
                "mix_probs must have one entry per dataset"
            total = float(sum(mix_probs))
            self.mix_probs: Optional[List[float]] = [p / total for p in mix_probs]
        else:
            self.mix_probs = None
        self.shuffle_buffer = shuffle_buffer
        self.val_docs = val_docs
        self.seed = seed
        self.text_key = text_key

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.eos = self.tokenizer.eos_token_id

        self._token_buffer: List[int] = []
        self._stream = self._make_stream()

        probs = self.mix_probs or [1.0 / len(self.dataset_names)] * len(self.dataset_names)
        mix = ", ".join(f"{n}({p:.2f})" for n, p in zip(self.dataset_names, probs))
        print(f"[StreamingDataLoader] {split}: streaming {mix}")

    def _make_stream(self):
        from datasets import load_dataset, interleave_datasets

        streams = []
        for i, name in enumerate(self.dataset_names):
            ds = load_dataset(name, split="train", streaming=True)
            if self.text_key != "text":
                ds = ds.rename_column(self.text_key, "text")
            # Align schemas across sources so interleave_datasets can mix them.
            ds = ds.select_columns(["text"])
            if self.split == "val":
                # Deterministic held-out set: the first `val_docs` documents.
                ds = ds.take(self.val_docs)
            else:
                # Everything after the val reservation, shuffled on the fly.
                # Per-source seed offset so the streams aren't phase-locked.
                ds = ds.skip(self.val_docs).shuffle(
                    buffer_size=self.shuffle_buffer, seed=self.seed + i
                )
            streams.append(ds)

        if len(streams) == 1:
            combined = streams[0]
        else:
            combined = interleave_datasets(
                streams,
                probabilities=self.mix_probs,
                seed=self.seed,
                stopping_strategy="all_exhausted",
            )
        return iter(combined)

    def _fill(self, n: int) -> None:
        while len(self._token_buffer) < n:
            try:
                ex = next(self._stream)
            except StopIteration:
                # Exhausted (small val set, or a full epoch of train) — restart.
                self._stream = self._make_stream()
                ex = next(self._stream)
            ids = self.tokenizer.encode(ex["text"])
            ids.append(self.eos)
            self._token_buffer.extend(ids)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.seq_len + 1
        need = self.batch_size * chunk
        self._fill(need)

        flat = self._token_buffer[:need]
        self._token_buffer = self._token_buffer[need:]

        buf = torch.tensor(flat, dtype=torch.long).view(self.batch_size, chunk)
        buf = buf.to(self.device)
        return buf[:, :-1].contiguous(), buf[:, 1:].contiguous()


def prepare(split: str, data_dir: str, dataset_name: str = "monology/pile-uncopyrighted",
            tokenizer_name: str = "mistralai/Mistral-7B-v0.1", max_tokens: int = 2_000_000,
            max_docs: Optional[int] = None):
    """Tokenize a streamed dataset into a flat uint16 `{split}.bin` file.

    Retained for building the pre-tokenized Pile `val.bin` consumed by the
    perplexity eval. Training no longer uses this — it streams directly.

    Documents are tokenized exactly as the training loader packs them — each
    `[BOS, ...doc tokens..., EOS]` — so the perplexity eval sees the same
    distribution the model trains on. To keep the eval set disjoint from the
    train stream (which skips the first `val_docs` documents per source), build
    this from the first `max_docs` documents with `max_docs == val_docs`.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{split}.bin"

    if out_path.exists():
        print(f"{out_path} already exists, skipping.")
        return

    print(f"Loading tokenizer from {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eos = tokenizer.eos_token_id

    limit = f"{max_docs:,} docs" if max_docs is not None else f"{max_tokens:,} tokens"
    print(f"Streaming {dataset_name} until {limit} ...")
    stream = load_dataset(dataset_name, split="train", streaming=True)

    total = 0
    with open(out_path, "wb") as f:
        for i, example in enumerate(stream):
            if max_docs is not None and i >= max_docs:
                break
            ids = tokenizer.encode(example["text"])  # adds BOS
            ids.append(eos)
            np.array(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
            if max_docs is None and total >= max_tokens:
                break
            if (i + 1) % 10_000 == 0:
                print(f"  {i+1:,} docs  {total:,} tokens ...")

    print(f"Saved {out_path}  ({total:,} tokens, {out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a pre-tokenized binary (e.g. Pile val.bin).")
    parser.add_argument("--data_dir", type=str, default="data/pile")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--dataset", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--max_tokens", type=int, default=2_000_000)
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Cap documents (use = --val_docs to keep val.bin disjoint from train)")
    args = parser.parse_args()
    prepare(args.split, args.data_dir, args.dataset, args.tokenizer, args.max_tokens, args.max_docs)
