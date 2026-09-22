"""Deterministic long-context retrieval benchmark for SACCADE."""
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .config import SaccadeConfig
from .model import SACCADE


@dataclass(frozen=True)
class RetrievalExample:
    tokens: tuple[int, ...]
    answer: int
    semantic_answers: tuple[int, ...]
    answer_position: int
    key: int


class SyntheticRetrievalDataset(Dataset):
    """Fixed-length token sequences with a distant key/value needle.

    Each record contains one ``KEY, VALUE`` pair in the first 80% of the
    context and a query near the end.  Values have a deterministic alias,
    allowing semantic accuracy to accept an equivalent token without
    treating unrelated tokens as correct.
    """

    def __init__(self, examples: list[RetrievalExample], split: str):
        self.examples, self.split = examples, split

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor | int]:
        item = self.examples[index]
        tokens = torch.tensor(item.tokens, dtype=torch.long)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
            "answer": item.answer,
            "semantic_answers": item.semantic_answers,
            # Position in labels at which the answer value is predicted.
            "answer_position": item.answer_position - 1,
            "key": item.key,
        }


def generate_retrieval_datasets(total: int = 1000, seq_len: int = 512,
                                seed: int = 1729) -> dict[str, SyntheticRetrievalDataset]:
    if total < 3 or seq_len < 32:
        raise ValueError("total must be >= 3 and seq_len must be >= 32")
    rng = random.Random(seed)
    examples: list[RetrievalExample] = []
    for index in range(total):
        key = 8 + (index % 24)
        value = 32 + ((index * 7 + 3) % 16)
        alias = 48 + ((value - 32) % 16)
        tokens = [1]  # BOS
        # Structured filler makes the task reproducible but non-trivial.
        for position in range(seq_len - 12):
            tokens.append(2 + ((position * 13 + index * 17 + rng.randrange(7)) % 6))
        needle_at = seq_len - 12 - (index % 17)
        tokens[needle_at:needle_at] = [key, value]
        tokens = tokens[:seq_len - 6]
        tokens += [3, 4, key, 5, value, 6]  # query, answer marker, answer
        tokens = tokens[:seq_len]
        # Keep exact length even after the insertion/truncation above.
        tokens += [2] * (seq_len - len(tokens))
        answer_position = tokens.index(value, max(1, needle_at + 2))
        examples.append(RetrievalExample(tuple(tokens), value, (value, alias),
                                         answer_position, key))
    rng.shuffle(examples)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    return {
        "train": SyntheticRetrievalDataset(examples[:n_train], "train"),
        "val": SyntheticRetrievalDataset(examples[n_train:n_train + n_val], "val"),
        "test": SyntheticRetrievalDataset(examples[n_train + n_val:], "test"),
    }


def save_retrieval_datasets(datasets: dict[str, SyntheticRetrievalDataset],
                            path: Path) -> None:
    """Persist the generated corpus and metadata as a portable JSON artifact."""
    payload = {}
    for split, dataset in datasets.items():
        payload[split] = [
            {
                "tokens": list(example.tokens),
                "answer": example.answer,
                "semantic_answers": list(example.semantic_answers),
                "answer_position": example.answer_position,
                "key": example.key,
            }
            for example in dataset.examples
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


class CausalTransformerBaseline(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 4,
                 layers: int = 2, max_length: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                           dropout=0.0, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_sequence(self, tokens: Tensor) -> Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.embedding(tokens) + self.position(positions)[None]
        mask = torch.triu(torch.ones(tokens.size(1), tokens.size(1),
                                     device=tokens.device, dtype=torch.bool), 1)
        return self.lm_head(self.norm(self.encoder(x, mask=mask)))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _batch(dataset: SyntheticRetrievalDataset, indices: list[int], device: str) -> dict:
    rows = [dataset[i] for i in indices]
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if isinstance(values[0], Tensor):
            result[key] = torch.stack(values).to(device)
        elif isinstance(values[0], int):
            result[key] = torch.tensor(values, dtype=torch.long, device=device)
        else:
            result[key] = values
    return result


@torch.no_grad()
def evaluate_model(model: nn.Module, dataset: SyntheticRetrievalDataset,
                   batch_size: int = 4, device: str = "cpu",
                   limit: int | None = None) -> dict[str, float]:
    model.eval()
    losses, exact, semantic, count = [], 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    size = min(len(dataset), limit) if limit else len(dataset)
    for start in range(0, size, batch_size):
        batch = _batch(dataset, list(range(start, min(start + batch_size, size))), device)
        logits = (model.forward_sequence(batch["input_ids"])[0]
                  if isinstance(model, SACCADE) else model.forward_sequence(batch["input_ids"]))
        losses.append(criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1)).item())
        positions = batch["answer_position"]
        predicted = logits[torch.arange(logits.size(0), device=device), positions].argmax(-1)
        answers = batch["answer"]
        exact += int((predicted == answers).sum())
        semantic += sum(int(int(predicted[i]) in batch["semantic_answers"][i])
                        for i in range(len(predicted)))
        count += len(predicted)
    loss = sum(losses) / max(len(losses), 1)
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss))),
            "exact_retrieval_accuracy": exact / max(count, 1),
            "semantic_retrieval_accuracy": semantic / max(count, 1)}


@torch.no_grad()
def evaluate_generation(model: nn.Module, dataset: SyntheticRetrievalDataset,
                        limit: int | None = None, device: str = "cpu") -> dict[str, float]:
    """Autoregressively generate the answer token after the query prefix."""
    model.eval()
    exact = semantic = 0
    rows = dataset.examples[:limit] if limit else dataset.examples
    for item in rows:
        prefix = torch.tensor(item.tokens[:item.answer_position], dtype=torch.long,
                              device=device).unsqueeze(0)
        logits = (model.forward_sequence(prefix)[0] if isinstance(model, SACCADE)
                  else model.forward_sequence(prefix))
        predicted = int(logits[:, -1].argmax(-1).item())
        exact += predicted == item.answer
        semantic += predicted in item.semantic_answers
    total = max(len(rows), 1)
    return {"generation_exact_accuracy": exact / total,
            "generation_semantic_accuracy": semantic / total}


def train_model(model: nn.Module, dataset: SyntheticRetrievalDataset, *,
                epochs: int = 1, steps_per_epoch: int = 8, batch_size: int = 2,
                learning_rate: float = 3e-4, seed: int = 1729, device: str = "cpu") -> list[dict]:
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        order = list(range(len(dataset)))
        random.Random(seed + epoch).shuffle(order)
        for step in range(min(steps_per_epoch, (len(order) + batch_size - 1) // batch_size)):
            batch = _batch(dataset, order[step * batch_size: (step + 1) * batch_size], device)
            logits = (model.forward_sequence(batch["input_ids"])[0]
                      if isinstance(model, SACCADE) else model.forward_sequence(batch["input_ids"]))
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append({"epoch": epoch, "step": step, "loss": float(loss.detach()),
                            "perplexity": float(torch.exp(loss.detach()))})
    return history


def save_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(history, indent=2))
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
