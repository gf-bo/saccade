"""Train a ~50M SACCADE language model on FineWeb-Edu in Kaggle.

This is intentionally a standalone entry point: it can be run from a Kaggle
notebook after cloning the repository or with torchrun for multi-GPU DDP.
The default budget is one streamed epoch capped at 600M training tokens and
8,000 packed evaluation sequences.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from saccade import SaccadeConfig, SACCADE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/saccade_fineweb_edu_50m"))
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--train-tokens", type=int, default=600_000_000)
    p.add_argument("--eval-sequences", type=int, default=8_000)
    p.add_argument("--context-length", type=int, choices=(1024, 2048), default=1024)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--mtp-horizon", type=int, default=0)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("This training entry point requires a CUDA Kaggle accelerator.")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    return world, rank, local_rank, device


def tokenizer_and_stream(args: argparse.Namespace):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("The selected tokenizer must define an EOS token.")
    stream = load_dataset(
        args.dataset, name=args.dataset_config, split="train", streaming=True
    )
    return tokenizer, stream


def packed_sequences(
    stream, tokenizer, context_length: int, rank: int, world: int,
    start_document: int = 0,
) -> Iterator[torch.Tensor]:
    """Pack documents into fixed-length causal examples without dropping tails."""
    buffer: list[int] = []
    document_index = 0
    for row in stream:
        if document_index < start_document or document_index % world != rank:
            document_index += 1
            continue
        text = row.get("text")
        if not text:
            document_index += 1
            continue
        buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        buffer.append(tokenizer.eos_token_id)
        while len(buffer) >= context_length + 1:
            values = buffer[: context_length + 1]
            del buffer[:context_length]
            yield torch.tensor(values, dtype=torch.long)
        document_index += 1


def model_config(tokenizer, args: argparse.Namespace) -> SaccadeConfig:
    # GPT-2 vocabulary + d_model=512 produces 47.6M parameters with the
    # current architecture. MTP is opt-in because each extra head is large.
    return SaccadeConfig(
        vocab_size=len(tokenizer),
        d_model=512,
        d_addr=64,
        d_ssm=256,
        n_heads=8,
        w_fine=32,
        w_coarse=256,
        s_max=64,
        l_slot=128,
        top_k=4,
        local_window=32,
        sequence_chunk_size=128,
        mtp_num_tokens=args.mtp_horizon,
        mtp_loss_weight=0.5,
        pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )


def make_batch(iterator, batch_size: int, device: torch.device) -> torch.Tensor:
    rows = [next(iterator) for _ in range(batch_size)]
    return torch.stack(rows).to(device, non_blocking=True)


def evaluate(model, iterator, count: int, batch_size: int, device: torch.device) -> dict:
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)
    seen = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        while seen < count:
            batch = make_batch(iterator, min(batch_size, count - seen), device)
            logits, _ = model.forward_sequence(batch)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_loss += loss
            total_tokens += batch[:, 1:].numel()
            seen += batch.size(0)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss)
        dist.all_reduce(total_tokens)
    mean_loss = float((total_loss / total_tokens).cpu())
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0)),
            "tokens": int(total_tokens.item())}


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    world, rank, _, device = setup_distributed()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer, stream = tokenizer_and_stream(args)
    config = model_config(tokenizer, args)
    model = SACCADE(config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model = DDP(model, device_ids=[device.index], find_unused_parameters=False) if world > 1 else model
    raw_model = model.module if isinstance(model, DDP) else model
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
            weight_decay=args.weight_decay, fused=True,
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    train_iter = iter(packed_sequences(
        stream, tokenizer, args.context_length, rank, world, start_document=10_000))
    eval_stream = load_eval_stream(args)
    eval_before = evaluate(
        raw_model, packed_sequences(eval_stream, tokenizer, args.context_length, rank, world),
        args.eval_sequences, args.micro_batch_size, device)
    steps = math.ceil(args.train_tokens / (
        args.context_length * args.micro_batch_size * world * args.grad_accumulation))
    warmup = max(1, int(steps * args.warmup_ratio))
    history = []
    seen_tokens = 0
    start_time = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        loss_value = 0.0
        for _ in range(args.grad_accumulation):
            batch = make_batch(train_iter, args.micro_batch_size, device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model.forward_sequence(batch)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                ) / args.grad_accumulation
            scaler.scale(loss).backward()
            loss_value += float(loss.detach()) * args.grad_accumulation
            seen_tokens += batch[:, 1:].numel() * world
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        progress = min(1.0, (step + 1) / max(1, steps))
        lr_scale = progress / (warmup / steps) if step < warmup else 0.5 * (
            1 + math.cos(math.pi * (progress - warmup / steps) / max(1e-8, 1 - warmup / steps)))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * max(0.0, min(1.0, lr_scale))
        if rank == 0 and ((step + 1) % args.eval_every == 0 or step == 0):
            record = {"step": step + 1, "tokens": seen_tokens, "train_loss": loss_value,
                      "lr": optimizer.param_groups[0]["lr"],
                      "tokens_per_second": seen_tokens / max(1e-6, time.perf_counter() - start_time)}
            history.append(record)
            print(json.dumps(record), flush=True)
        if rank == 0 and (step + 1) % args.save_every == 0:
            torch.save(raw_model.state_dict(), args.output / "checkpoint.pt")
    eval_stream = load_eval_stream(args)
    eval_after = evaluate(
        raw_model, packed_sequences(eval_stream, tokenizer, args.context_length, rank, world),
        args.eval_sequences, args.micro_batch_size, device)
    if rank == 0:
        torch.save(raw_model.state_dict(), args.output / "model.pt")
        tokenizer.save_pretrained(args.output / "tokenizer")
        save_json(args.output / "config.json", config.__dict__)
        save_json(args.output / "metrics.json", {
            "parameters": parameter_count, "world_size": world,
            "train_tokens": seen_tokens, "context_length": args.context_length,
            "eval_sequences": args.eval_sequences, "eval_before": eval_before,
            "eval_after": eval_after, "history": history,
        })
        with (args.output / "eval_table.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["stage", "loss", "perplexity", "tokens"])
            writer.writeheader()
            writer.writerow({"stage": "before", **eval_before})
            writer.writerow({"stage": "after", **eval_after})
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_eval_stream(args: argparse.Namespace):
    from datasets import load_dataset
    return load_dataset(args.dataset, name=args.dataset_config, split="train", streaming=True)


if __name__ == "__main__":
    main()
