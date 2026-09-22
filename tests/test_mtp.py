import torch

from saccade import SaccadeConfig, create_model, create_trainer, mtp_loss, mtp_targets


def make_mtp():
    return create_model(
        SaccadeConfig(
            vocab_size=31, d_model=24, d_addr=8, d_ssm=12, n_heads=4,
            w_fine=4, w_coarse=8, s_max=4, l_slot=8, top_k=2,
            local_window=4, mtp_num_tokens=2,
        )
    )


def test_mtp_shapes_and_loss_backward():
    model = make_mtp()
    tokens = torch.randint(0, 31, (2, 9))
    output = model.forward_sequence_mtp(tokens)
    assert output.logits.shape == (2, 9, 31)
    assert len(output.mtp_logits) == 2
    assert all(item.shape == (2, 9, 31) for item in output.mtp_logits)
    losses = mtp_loss(output.logits, output.mtp_logits, tokens)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert any(p.grad is not None for p in model.parameters())


def test_mtp_targets_and_trainer_step():
    tokens = torch.arange(8).reshape(1, 8)
    targets = mtp_targets(tokens, 2, pad_id=0)
    assert targets.shape == (1, 8, 2)
    assert targets[0, 0].tolist() == [2, 3]
    model = make_mtp()
    trainer = create_trainer(model)
    result = trainer.step(torch.randint(0, 31, (2, 8)), 0, 2)
    assert result.mtp_enabled
    assert result.losses["mtp"].isfinite()
