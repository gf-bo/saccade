import torch

from saccade import SACCADE, SaccadeConfig
from saccade.benchmark import (CausalTransformerBaseline, evaluate_model,
                               generate_retrieval_datasets, parameter_count,
                               save_retrieval_datasets,
                               train_model)


def config():
    return SaccadeConfig(vocab_size=64, d_model=40, d_addr=10, d_ssm=20,
                         n_heads=4, w_fine=8, w_coarse=32, s_max=8,
                         l_slot=64, top_k=2, local_window=8)


def test_dataset_is_deterministic_and_split():
    left = generate_retrieval_datasets(100, 64, 7)
    right = generate_retrieval_datasets(100, 64, 7)
    assert [len(left[x]) for x in ("train", "val", "test")] == [80, 10, 10]
    assert left["train"][0]["input_ids"].tolist() == right["train"][0]["input_ids"].tolist()
    assert left["train"][0]["answer_position"] >= 0


def test_dataset_can_be_persisted(tmp_path):
    datasets = generate_retrieval_datasets(10, 64, 7)
    path = tmp_path / "dataset.json"
    save_retrieval_datasets(datasets, path)
    payload = __import__("json").loads(path.read_text())
    assert sum(len(rows) for rows in payload.values()) == 10
    assert len(payload["train"][0]["tokens"]) == 64


def test_parameter_counts_and_sequence_shapes():
    saccade = SACCADE(config())
    baseline = CausalTransformerBaseline(64, 28, 4, 1, 512)
    assert 25_000 <= parameter_count(saccade) <= 200_000
    assert 25_000 <= parameter_count(baseline) <= 200_000
    data = generate_retrieval_datasets(20, 64, 2)["test"]
    batch = torch.stack([data[i]["input_ids"] for i in range(2)])
    assert saccade.forward_sequence(batch)[0].shape == (2, 63, 64)
    assert baseline.forward_sequence(batch).shape == (2, 63, 64)


def test_metric_shapes_and_short_training_smoke():
    data = generate_retrieval_datasets(8, 40, 3)["train"]
    model = SACCADE(config())
    train_model(model, data, epochs=1, steps_per_epoch=1, batch_size=1)
    metrics = evaluate_model(model, data, batch_size=1)
    assert set(metrics) == {"loss", "perplexity", "exact_retrieval_accuracy",
                            "semantic_retrieval_accuracy"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
