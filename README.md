# SACCADE

SACCADE (**Segmented Addressable Causal Chunking with Adaptive Dynamic
Embeddings**) is a PyTorch architecture for autoregressive models that need
long-context memory without global attention.

It combines:

- causal local attention;
- adaptive chunking with a straight-through EMA boundary estimator;
- a bounded, addressable slot memory with causal Top-K routing;
- a selective Mini-SSM for inexpensive context summarization;
- fine attention over selected slots, local context, and the SSM state;
- streaming state that preserves batch isolation, device/dtype, and absolute
  addresses;
- optional Triton kernels for CUDA inference hot paths, with tested PyTorch
  fallbacks.
- optional multi-token prediction (MTP) heads for speculative decoding and
  higher-throughput large-scale training.

SACCADE is an experimental research library, not a claim that global
Transformers are obsolete. Its design goal is to trade dense all-to-all
attention for bounded local computation and explicit long-context memory.

## Installation

```bash
python -m pip install git+https://github.com/gf-bo/saccade.git
```

Install from source with the development dependencies:

```bash
python -m pip install -e ".[dev]"
```

PyTorch is the only runtime dependency. Triton is optional and is only used
when it is installed, CUDA is available, and the input satisfies the kernel
constraints.

## Multi-token prediction

MTP is disabled by default. Enable it by choosing a prediction horizon:

```python
from saccade import SaccadeConfig, create_model, create_trainer

config = SaccadeConfig(
    vocab_size=32_000,
    d_model=256,
    d_addr=64,
    d_ssm=128,
    n_heads=8,
    mtp_num_tokens=3,       # predict offsets +2, +3, and +4
    mtp_loss_weight=0.5,
)
model = create_model(config).cuda()
trainer = create_trainer(model)

tokens = tokens.cuda()
result = trainer.step(tokens, step=0, total_steps=1000)
```

The regular `forward_sequence` API remains unchanged. For MTP outputs use:

```python
output = model.forward_sequence_mtp(tokens)
primary = output.logits
future_logits = output.mtp_logits  # tuple of [batch, time, vocab] tensors
```

`SaccadeTrainer` automatically selects standard LM loss or MTP loss from the
configuration. `mtp_loss` uses full autograd during training. During
inference/evaluation, the optional Triton cross-entropy kernel can reduce the
cost of scoring large MTP batches. This separation is intentional: replacing
training autograd with a forward-only kernel would silently produce incorrect
gradients.

## Quick start

```python
import torch
from saccade import SACCADE, SaccadeConfig, StreamingState

config = SaccadeConfig(
    vocab_size=32_000,
    d_model=256,
    d_addr=64,
    d_ssm=128,
    n_heads=8,
)
model = SACCADE(config)

tokens = torch.randint(0, config.vocab_size, (2, 128))
state = StreamingState.empty(batch=2, d_ssm=config.d_ssm)
output = model(tokens, state)

next_token_logits = output.logits       # [batch, vocab_size]
next_state = output.state               # reusable for the next stream segment
```

For teacher-forced language-model training, use the causal sequence adapter:

```python
logits, state = model.forward_sequence(tokens, state)
# logits: [batch, sequence, vocab_size]
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, config.vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

Detach the state at truncated-backpropagation boundaries:

```python
state = state.detach()
```

## Architecture and guarantees

The regular `forward` method is the streaming endpoint. `forward_sequence`
computes local attention and the SSM over a sequence while writing and
querying slots chronologically, so future tokens cannot affect earlier
outputs. Slot memories are isolated per batch item and capped by `s_max`.
`StreamingState.to(device, dtype)` preserves integer addresses while moving
tensor state safely.

SACCADE does **not** add global attention. The local window, slot capacity,
slot length, and Top-K route are explicit configuration limits:

```python
config = SaccadeConfig(
    vocab_size=32_000,
    d_model=256,
    local_window=16,
    s_max=32,
    l_slot=128,
    top_k=4,
)
```

## Optional Triton kernels

The optional kernels implement the sequential EMA update and slot dot-product
scoring, plus MTP cross-entropy evaluation. They are dispatched only for
compatible contiguous CUDA tensors and inference-only calls. Training keeps
the PyTorch path so autograd remains complete and predictable.

```bash
python scripts/benchmark_triton.py --device cpu
python scripts/benchmark_triton.py --device cuda
```

On a Tesla T4, the measured EMA kernel was 62.76x faster than its PyTorch
reference (0.5605 ms vs. 35.1815 ms) and used 34.30 MiB vs. 38.30 MiB peak
allocation for the reported workload. For only 64 slots, PyTorch `bmm` was
faster than the Triton scoring kernel; the implementation deliberately keeps
the faster path instead of forcing Triton.

The reproducible Kaggle launcher is `scripts/kaggle_two_t4.py`. It supports
NCCL/DDP when two GPUs are actually assigned:

```bash
torchrun --standalone --nproc_per_node=2 scripts/kaggle_two_t4.py
```

## Validation

Run the complete CPU-safe suite:

```bash
python -m pytest
```

The suite covers tensor shapes, gradients, causal behavior, streaming state,
batch slot isolation, routing, empty and one-token sequences, benchmark data,
and Triton fallback behavior. CUDA-specific tests skip cleanly when CUDA or
Triton is unavailable.

## Benchmark

The research benchmark is intentionally kept separate from the installable
runtime package:

```bash
python scripts/run_benchmark.py --output benchmark_results
```

It creates 1,000 synthetic 512-token retrieval examples, trains SACCADE and a
parameter-matched causal Transformer, and records loss, perplexity, exact
retrieval, semantic retrieval, and generation metrics. Generated datasets,
checkpoints, and logs are ignored by Git.

## License

Released under the [MIT License](LICENSE).
