import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from mistral_inference.args import TransformerArgs
from mistral_inference.transformer import Transformer
from dataloader import OpenWebTextDataLoader


def load_args(config_path: str) -> TransformerArgs:
    with open(config_path) as f:
        data = json.load(f)
    args = TransformerArgs.from_dict(data)
    return args


def build_model(args: TransformerArgs, device: torch.device) -> Transformer:
    args.max_batch_size = 1  # set before construction
    with torch.device("meta"):
        model = Transformer(args)
    # materialize all parameters on the target device
    model = model.to_empty(device=device)
    model = model.to(dtype=torch.bfloat16)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mixtral_tokenformer.json")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default="mixtral-tokenformer")
    parser.add_argument("--data_dir", type=str, default="data/owt")
    parser.add_argument("--seq_len", type=int, default=512)
    cfg = parser.parse_args()

    device = torch.device("cuda")

    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    args = load_args(cfg.config)
    args.max_batch_size = cfg.batch_size
    model = build_model(args, device)

    wandb.config.update({
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_layers": args.n_layers,
        "dim": args.dim,
        "tokenformer_qkv_slots": args.tokenformer.qkv_slots if args.tokenformer else None,
        "tokenformer_ffn_slots": args.tokenformer.ffn_slots if args.tokenformer else None,
    })

    optimizer = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps)

    train_loader = OpenWebTextDataLoader("train", cfg.data_dir, cfg.seq_len, cfg.batch_size, device)
    val_loader = OpenWebTextDataLoader("val", cfg.data_dir, cfg.seq_len, cfg.batch_size, device)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(exist_ok=True)

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

        if step % 50 == 0:
            train_loss = loss.item() * cfg.grad_accum_steps

            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = val_loader.get_batch()
                val_flat = val_inputs.reshape(-1)
                val_seqlens = [val_inputs.shape[1]] * cfg.batch_size
                val_logits = model(val_flat, seqlens=val_seqlens)
                val_loss = nn.functional.cross_entropy(val_logits, val_targets.reshape(-1)).item()
            model.train()

            wandb.log({"loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0], "step": step})
            print(f"step {step:>6}  loss {train_loss:.4f}  val_loss {val_loss:.4f}")

        if step % 1000 == 0 and step > 0:
            ckpt_path = save_dir / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": args,
            }, ckpt_path)
            # also save the config so the checkpoint is self-contained
            with open(save_dir / "params.json", "w") as f:
                json.dump(json.load(open(cfg.config)), f, indent=2)
            print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()