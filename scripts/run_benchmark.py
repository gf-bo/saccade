from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saccade.benchmark import (CausalTransformerBaseline, generate_retrieval_datasets,
                               evaluate_generation, evaluate_model, parameter_count,
                               save_history, save_retrieval_datasets, train_model)
from saccade.config import SaccadeConfig
from saccade.model import SACCADE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-limit", type=int, default=100,
                        help="number of validation/test examples evaluated (dataset remains 1000)")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    datasets = generate_retrieval_datasets(args.total, args.seq_len, args.seed)
    cfg = SaccadeConfig(vocab_size=64, d_model=40, d_addr=10, d_ssm=20, n_heads=4,
                        w_fine=8, w_coarse=32, s_max=8, l_slot=64, top_k=2,
                        local_window=8)
    models = {
        "saccade": SACCADE(cfg),
        # A compact one-block baseline keeps the parameter budget close to the
        # improved SACCADE configuration, including its local FFN.
        "transformer": CausalTransformerBaseline(64, 36, 4, 1, args.seq_len),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_retrieval_datasets(datasets, args.output / "dataset.json")
    results = {"config": vars(args) | {"config": cfg.__dict__}, "models": {}}
    for name, model in models.items():
        count = parameter_count(model)
        if not 25_000 <= count <= 200_000:
            raise RuntimeError(f"{name} has {count} parameters; expected 25k-200k")
        history = train_model(model, datasets["train"], epochs=args.epochs,
                              steps_per_epoch=args.steps, batch_size=args.batch_size,
                              seed=args.seed)
        save_history(args.output / name, history)
        torch.save(model.state_dict(), args.output / f"{name}.pt")
        results["models"][name] = {"parameters": count,
                                   "validation": evaluate_model(model, datasets["val"], args.batch_size, limit=args.eval_limit),
                                   "test": evaluate_model(model, datasets["test"], args.batch_size, limit=args.eval_limit),
                                   "generation": evaluate_generation(model, datasets["test"], args.eval_limit)}
    (args.output / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
