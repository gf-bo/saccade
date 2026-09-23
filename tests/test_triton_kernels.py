import pytest
import torch

from saccade.triton_kernels import (
    causal_local_attention,
    fused_ema_update,
    mtp_cross_entropy,
    slot_scores,
    triton_available,
)


def reference_ema(x, alpha):
    values = [x[:, 0]]
    for i in range(1, x.size(1)):
        values.append(alpha * values[-1] + (1 - alpha) * x[:, i - 1])
    return torch.stack(values, 1)


def test_ema_and_scores_cpu_match_reference_and_have_gradients():
    x = torch.randn(2, 7, 5, requires_grad=True)
    state = torch.randn(2, 5, requires_grad=True)
    keys = torch.randn(2, 4, 5, requires_grad=True)
    ema = fused_ema_update(x, 0.8)
    assert torch.allclose(ema, reference_ema(x, 0.8))
    scores = slot_scores(state, keys)
    assert torch.allclose(scores, (state[:, None] * keys).sum(-1) / 5**0.5)
    (ema.square().mean() + scores.square().mean()).backward()
    assert x.grad is not None and state.grad is not None and keys.grad is not None


def test_mtp_cross_entropy_cpu_matches_torch():
    logits = torch.randn(12, 19, requires_grad=True)
    targets = torch.randint(0, 19, (12,))
    actual = mtp_cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert logits.grad is not None


@pytest.mark.skipif(not triton_available(), reason="CUDA and Triton are unavailable")
def test_cuda_triton_forward_matches_reference():
    x = torch.randn(2, 33, 17, device="cuda")
    state = torch.randn(2, 17, device="cuda")
    keys = torch.randn(2, 9, 17, device="cuda")
    with torch.no_grad():
        actual_ema = fused_ema_update(x, 0.83)
        actual_scores = slot_scores(state, keys)
        actual_attn = causal_local_attention(
            x[:, None], x[:, None], x[:, None], window=7)
    assert torch.allclose(actual_ema, reference_ema(x, 0.83), atol=2e-3, rtol=2e-3)
    assert torch.allclose(actual_scores, (state[:, None] * keys).sum(-1) / 17**0.5,
                          atol=2e-3, rtol=2e-3)
    assert actual_attn.shape == (2, 1, 33, 17)


@pytest.mark.skipif(not triton_available(), reason="CUDA and Triton are unavailable")
def test_cuda_gradients_use_reference_autograd_path():
    x = torch.randn(2, 8, 7, device="cuda", requires_grad=True)
    state = torch.randn(2, 7, device="cuda", requires_grad=True)
    keys = torch.randn(2, 3, 7, device="cuda", requires_grad=True)
    (fused_ema_update(x, 0.7).sum() + slot_scores(state, keys).sum()).backward()
    assert x.grad is not None and state.grad is not None and keys.grad is not None
