import torch
from saccade import SACCADE, SaccadeConfig, StreamingState, train_step, CurriculumConfig
from saccade.losses import saccade_loss

def make():
    return SACCADE(SaccadeConfig(vocab_size=97, d_model=32, d_addr=8, d_ssm=16, n_heads=4, w_fine=4, w_coarse=8, s_max=6, l_slot=8, top_k=2, local_window=4))

def test_shapes_and_gradients():
    m = make(); x = torch.randint(0, 97, (2, 13)); y = m(x)
    assert y.logits.shape == (2, 97)
    assert y.boundary_probs.shape == x.shape
    loss = saccade_loss(y.logits, x[:, -1], y.boundary_probs, y.route_scores, y.route_indices, y.ssm_logits, m.config)["loss"]
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())

def test_streaming_state_and_capacity():
    m = make(); s = StreamingState.empty(1, 16)
    for _ in range(4):
        out = m(torch.randint(0, 97, (1, 9)), s)
        s = out.state
    assert all(len(row) <= m.config.s_max for row in s.slots)
    assert s.next_address.tolist() == [36]
    assert all(slot.tokens.size(0) <= m.config.l_slot for row in s.slots for slot in row)
    for row in s.slots:
        assert all(slot.start < slot.end for slot in row)
        assert all(row[i].end <= row[i + 1].start for i in range(len(row) - 1))

def test_causal_and_topk():
    m = make(); x = torch.randint(0, 97, (1, 10))
    local_prefix = m.local(m.embed(x[:, :6])[0], m.config.w_fine)
    local_extended = m.local(m.embed(x)[0], m.config.w_fine)
    assert torch.allclose(local_prefix, local_extended[:, :6], atol=1e-6)
    assert m(x).route_indices.shape[1] == m.config.top_k

def test_batched_teacher_forcing_is_causal_and_differentiable():
    m = make()
    x = torch.randint(0, 97, (2, 9))
    logits, state = m.forward_sequence(x)
    assert logits.shape == (2, 9, 97)
    assert state.next_address.tolist() == [9, 9]
    prefix, _ = m.forward_sequence(x[:, :5])
    extended, _ = m.forward_sequence(x)
    assert torch.allclose(prefix, extended[:, :5], atol=1e-6)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 97), x.reshape(-1))
    loss.backward()
    assert m.sequence_retrieval_gate.grad is not None
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters())

def test_batch_slots_and_dtype_device_and_empty():
    m = make().double()
    state = StreamingState.empty(2, 16)
    out = m(torch.randint(0, 97, (2, 3)), state)
    assert len(out.state.slots) == 2 and out.state.next_address.tolist() == [3, 3]
    empty = m(torch.empty(2, 0, dtype=torch.long), out.state)
    assert empty.logits.shape == (2, 97)

def test_loss_one_token_is_finite():
    m = make()
    out = m(torch.randint(0, 97, (1, 1)))
    losses = saccade_loss(out.logits, torch.randint(0, 97, (1,)), out.boundary_probs,
                          out.route_scores, out.route_indices, out.ssm_logits, m.config)
    assert all(value.isfinite().item() for value in losses.values())

def test_threshold_has_straight_through_gradient_and_chunk_lengths():
    m = make()
    x = torch.randn(1, 5, m.config.d_model)
    _, _, boundaries, _ = m.chunker(x)
    boundaries.sum().backward()
    assert m.chunker.threshold.grad is not None
    # Boundaries at positions 2 and 4 produce lengths 2, 2, 1.
    probs = torch.tensor([[0.0, 0.1, 0.9, 0.1, 0.9]])
    cfg = m.config
    zero = torch.zeros(1, cfg.vocab_size)
    losses = saccade_loss(zero, torch.zeros(1, dtype=torch.long), probs,
                          torch.empty(1, 0), torch.empty(1, 0, dtype=torch.long), None, cfg, target_chunk=5/3)
    assert losses["chunk"].item() < 1e-8

def test_forward_top_k_override_and_curriculum_schedule():
    m = make()
    schedule = CurriculumConfig(initial_k=1)
    k, threshold, temperature = schedule.values(0, 10, m.config.top_k, m.config.boundary_threshold)
    assert k == 1
    out = m(torch.randint(0, 97, (1, 8)), top_k=k, threshold=threshold, temperature=temperature)
    assert out.route_indices.shape == (1, 1)
    assert out.route_scores.shape == (1, 1)

def test_curriculum_training_step():
    m = make()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    result = train_step(m, opt, torch.randint(0, 97, (2, 8)), torch.randint(0, 97, (2,)), 0, 10)
    assert result.losses["loss"].isfinite()
    assert result.grad_norm.isfinite()
