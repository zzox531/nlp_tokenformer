import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import wandb

from mistral_inference.args import TransformerArgs, TokenformerArgs
from mistral_inference.transformer import Transformer
from dataloader import StreamingTextDataLoader
from evaluate import (
    eval_hellaswag,
    eval_pile_perplexity,
    eval_winogrande,
)


def load_args(config_path: str) -> TransformerArgs:
    with open(config_path) as f:
        data = json.load(f)
    tokenformer_data = data.pop("tokenformer", None)
    args = TransformerArgs.from_dict(data)
    if tokenformer_data is not None:
        args.tokenformer = TokenformerArgs(**tokenformer_data)
    return args


def build_model(args: TransformerArgs, device: torch.device) -> Transformer:
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = Transformer(args)
    torch.set_default_dtype(orig_dtype)

    n_params = sum(p.numel() for p in model.parameters())
    model_mem_gib = n_params * 2 / 1024**3  # bfloat16 = 2 bytes
    gpu_free = (torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_reserved(device)) / 1024**3
    print(f"Model params:      {n_params:,}")
    print(f"Model memory (bf16): {model_mem_gib:.2f} GiB")
    print(f"GPU free memory:   {gpu_free:.2f} GiB")
    if model_mem_gib > gpu_free:
        print("WARNING: model is too large for available GPU memory")
    else:
        print("OK: model should fit")

    return model.to(device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mixtral_tokenformer.json")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Fraction of optimizer updates spent in LR warmup")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default="mixtral-tokenformer")
    parser.add_argument("--datasets", nargs="+", default=["Skylion007/openwebtext"],
                        help="HuggingFace dataset(s) streamed and interleaved for training")
    parser.add_argument("--mix_probs", nargs="+", type=float, default=None,
                        help="Interleave probabilities per dataset (must match --datasets; "
                             "normalized automatically). Default: equal weights.")
    parser.add_argument("--shuffle_buffer", type=int, default=10_000,
                        help="Reservoir buffer size for on-the-fly shuffling of the train stream")
    parser.add_argument("--val_docs", type=int, default=2_000,
                        help="Documents reserved (and skipped by train) for the held-out val split")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="Seed for the streaming shuffle")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--no-tokenformer", dest="no_tokenformer", action="store_true",
                        help="Disable tokenformer even if present in config")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Run benchmarks every N steps (0 = disabled)")
    parser.add_argument("--eval_tasks", nargs="+",
                        choices=["pile", "hellaswag", "winogrande"],
                        default=["pile", "hellaswag", "winogrande"],
                        help="Benchmarks to run at each eval interval")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Pairs per forward pass during evaluation")
    parser.add_argument("--eval_max_samples", type=int, default=500,
                        help="Max examples per benchmark during training eval (None = full set)")
    parser.add_argument("--pile_data_dir", type=str, default=None,
                        help="Pre-tokenised Pile val.bin directory (required if 'pile' in eval_tasks)")
    parser.add_argument("--eval_pile_max_tokens", type=int, default=500_000,
                        help="Token budget for Pile perplexity during training")
    cfg = parser.parse_args()

    device = torch.device("cuda")

    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    args = load_args(cfg.config)
    if cfg.no_tokenformer:
        args.tokenformer = None
    args.max_batch_size = cfg.batch_size
    model = build_model(args, device)

    wandb.config.update({
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_layers": args.n_layers,
        "dim": args.dim,
        "tokenformer_qkv_slots": args.tokenformer.qkv_slots if args.tokenformer else None,
        "tokenformer_ffn_slots": args.tokenformer.ffn_slots if args.tokenformer else None,
    })

    eval_tasks = set(cfg.eval_tasks)
    tokenizer: Optional[Any] = None
    if cfg.eval_interval > 0:
        print(f"Loading tokenizer for eval ({cfg.tokenizer}) ...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    eval_log: List[Dict] = []

    def run_eval(step: int) -> None:
        model.eval()
        orig_max_batch = model.args.max_batch_size
        model.args.max_batch_size = cfg.eval_batch_size
        metrics: Dict[str, float] = {"step": step}

        with torch.no_grad():
            if "pile" in eval_tasks:
                metrics["pile_perplexity"] = eval_pile_perplexity(
                    model, device, tokenizer,
                    seq_len=cfg.seq_len,
                    max_tokens=cfg.eval_pile_max_tokens,
                    data_dir=cfg.pile_data_dir,
                )
            if "hellaswag" in eval_tasks:
                metrics["hellaswag_accuracy"] = eval_hellaswag(
                    model, device, tokenizer,
                    batch_size=cfg.eval_batch_size,
                    max_samples=cfg.eval_max_samples,
                )
            if "winogrande" in eval_tasks:
                metrics["winogrande_accuracy"] = eval_winogrande(
                    model, device, tokenizer,
                    batch_size=cfg.eval_batch_size,
                    max_samples=cfg.eval_max_samples,
                )

        model.args.max_batch_size = orig_max_batch
        model.train()

        wandb.log({f"eval/{k}": v for k, v in metrics.items() if k != "step"} | {"step": step})
        print(f"  [eval step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "step"))

        eval_log.append(metrics)
        with open(eval_log_path, "w") as f:
            json.dump(eval_log, f, indent=2)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)
    # The scheduler advances once per optimizer update (every grad_accum_steps
    # micro-steps), so its horizon must be measured in optimizer updates, not
    # raw micro-steps — otherwise the cosine barely moves and the LR stays flat.
    num_updates = cfg.max_steps // cfg.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * num_updates),
        num_training_steps=num_updates,
    )

    if cfg.mix_probs is not None and len(cfg.mix_probs) != len(cfg.datasets):
        parser.error("--mix_probs must have one value per --datasets entry")
    loader_kwargs = dict(
        dataset_names=cfg.datasets,
        mix_probs=cfg.mix_probs,
        tokenizer_name=cfg.tokenizer,
        shuffle_buffer=cfg.shuffle_buffer,
        val_docs=cfg.val_docs,
        seed=cfg.data_seed,
    )
    train_loader = StreamingTextDataLoader("train", cfg.seq_len, cfg.batch_size, device, **loader_kwargs)
    val_loader = StreamingTextDataLoader("val", cfg.seq_len, cfg.batch_size, device, **loader_kwargs)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(exist_ok=True)
    eval_log_path = save_dir / f"eval_log_{'tokenformer' if args.tokenformer else 'base'}.json"

    model.train()
    optimizer.zero_grad()

    for step in range(cfg.max_steps):
        inputs, targets = train_loader.get_batch()

        # mistral_inference forward expects a flat (sum_seqlens,) input
        # and a list of per-sequence lengths
        seqlens = [inputs.shape[1]] * cfg.batch_size
        flat_inputs = inputs.reshape(-1)

        logits = model(flat_inputs, seqlens=seqlens)  # (sum_seqlens, vocab_size)
        flat_targets = targets.reshape(-1)

        loss = nn.functional.cross_entropy(logits, flat_targets)
        loss = loss / cfg.grad_accum_steps
        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
            run_eval(step)

        if step % 50 == 0:
            train_loss = loss.item() * cfg.grad_accum_steps

            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = val_loader.get_batch()
                val_flat = val_inputs.reshape(-1)
                val_flat_targets = val_targets.reshape(-1)
                val_seqlens = [val_inputs.shape[1]] * cfg.batch_size
                val_logits = model(val_flat, seqlens=val_seqlens)
                val_loss = nn.functional.cross_entropy(val_logits, val_flat_targets).item()
                val_acc = (val_logits.argmax(-1) == val_flat_targets).float().mean().item()
            model.train()

            wandb.log({"loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0], "step": step})
            print(f"step {step:>6}  loss {train_loss:.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

    ckpt_path = save_dir / f"model_{"tokenformer" if args.tokenformer else "base"}.pt"
    torch.save({"step": step, "model_state_dict": model.state_dict(), "args": args}, ckpt_path)
    with open(save_dir / f"params_{"tokenformer" if args.tokenformer else "base"}.json", "w") as f:
        json.dump(json.load(open(cfg.config)), f, indent=2)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()