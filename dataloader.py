"""
OpenWebText dataloader for training.

Preparation (run once before training):
    python dataloader.py --data_dir data/owt --split train
    python dataloader.py --data_dir data/owt --split val

This downloads OpenWebText, tokenizes with the Mistral tokenizer, and writes
flat uint16 binary files that the DataLoader mmap-reads during training.
"""

import argparse
import os
import numpy as np
from pathlib import Path
from typing import Literal

import torch


def prepare(split: str, data_dir: str, tokenizer_name: str = "mistralai/Mistral-7B-v0.1", fraction: float = 0.1):
    """Download and tokenize OpenWebText into a flat uint16 binary file.

    Uses streaming mode so the raw dataset is never cached to disk — only the
    final tokenized binary is written, keeping disk usage minimal.
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

    # OpenWebText has ~8,013,769 examples total.  We stream to avoid writing
    # the full Arrow cache (which caused the disk-quota error).
    TOTAL_OWT = 8_013_769
    n_total = int(TOTAL_OWT * fraction)
    n_train = int(n_total * 0.995)
    n_val = n_total - n_train

    print(f"Streaming OpenWebText (fraction={fraction:.3f} → {n_total:,} examples) ...")
    stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    if split == "train":
        dataset = stream.take(n_train)
        n_expected = n_train
    else:
        dataset = stream.skip(n_train).take(n_val)
        n_expected = n_val

    print(f"  {split}: ~{n_expected:,} documents")

    # Tokenize and write tokens directly to the output file one document at a
    # time so peak RAM stays small and nothing large lands in the disk cache.
    total = 0
    with open(out_path, "wb") as f:
        for i, example in enumerate(dataset):
            ids = tokenizer.encode(example["text"])
            ids.append(eos)
            np.array(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
            if (i + 1) % 10_000 == 0:
                print(f"  {i+1:,} / {n_expected:,} docs  {total:,} tokens ...")

    print(f"Saved {out_path}  ({total:,} tokens, {out_path.stat().st_size / 1e9:.2f} GB)")


class OpenWebTextDataLoader:
    """
    Memory-mapped dataloader over a pre-tokenized OpenWebText binary.

    Call get_batch() to get (inputs, targets) of shape (batch_size, seq_len)
    with random starting positions (sampling with replacement each call).
    """

    def __init__(
        self,
        split: Literal["train", "val"],
        data_dir: str,
        seq_len: int,
        batch_size: int,
        device: torch.device,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

        path = Path(data_dir) / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run:\n"
                f"  python dataloader.py --data_dir {data_dir} --split train\n"
                f"  python dataloader.py --data_dir {data_dir} --split val"
            )

        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        print(f"[DataLoader] {split}: {self.n_tokens:,} tokens from {path}")

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.seq_len + 1
        ix = torch.randint(self.n_tokens - chunk, (self.batch_size,))
        buf = torch.stack(
            [torch.from_numpy(self.data[i : i + chunk].astype(np.int64)) for i in ix.tolist()]
        ).to(self.device)
        return buf[:, :-1].contiguous(), buf[:, 1:].contiguous()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OpenWebText binary files for training.")
    parser.add_argument("--data_dir", type=str, default="data/owt")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument(
        "--fraction", type=float, default=0.1,
        help="Fraction of OpenWebText to use (0.0–1.0). Default 0.1 = 10%%."
    )
    args = parser.parse_args()
    prepare(args.split, args.data_dir, args.tokenizer, args.fraction)
