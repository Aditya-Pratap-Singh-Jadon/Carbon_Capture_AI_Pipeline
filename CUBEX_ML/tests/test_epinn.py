"""Unit tests for E-PINN Architecture, Losses, and Training."""

import pytest
import torch
from carbon_capture.epinn.architecture import EPINNMultiHead, PHYSICAL_OUTPUT_NAMES
from carbon_capture.epinn.model import EPINNModel
from carbon_capture.epinn.losses import EPINNLoss
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER

def test_epinn_forward_pass():
    batch_size = 4
    x = torch.randn(batch_size, 26)
    model = EPINNModel(input_dim=26, hidden_dims=[64, 32])
    out = model(x)

    assert len(out) == len(PHYSICAL_OUTPUT_NAMES)
    for name in PHYSICAL_OUTPUT_NAMES:
        assert name in out
        assert out[name].shape == (batch_size,)

def test_epinn_loss_and_gradients():
    batch_size = 4
    x = torch.randn(batch_size, 26)
    model = EPINNModel(input_dim=26, hidden_dims=[64, 32])

    inputs_dict = {
        name: torch.ones(batch_size, requires_grad=False) * 10.0
        for name in CANONICAL_INPUT_ORDER
    }
    inputs_dict["eta_purity"] = torch.full((batch_size,), 0.95)
    inputs_dict["P_CO2"] = torch.full((batch_size,), 0.4397)
    inputs_dict["gamma_slip"] = torch.full((batch_size,), 0.02)
    inputs_dict["eta_boiler"] = torch.full((batch_size,), 0.85)
    inputs_dict["F_valid"] = torch.full((batch_size,), 0.95)
    inputs_dict["F_SME"] = torch.full((batch_size,), 1.05)
    inputs_dict["F_perm"] = torch.full((batch_size,), 0.98)
    inputs_dict["alpha"] = torch.full((batch_size,), 0.90)

    preds = model(x)
    total_loss, breakdown = model.loss_fn(
        preds=preds,
        inputs=inputs_dict,
        lagrangian_manager=model.lagrangian_manager,
    )

    assert total_loss > 0.0
    assert "l_physics" in breakdown
    assert "l_consistency" in breakdown
    assert "l_cc" in breakdown

    # Verify backpropagation
    total_loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and "temporal_encoder" not in name:
            assert param.grad is not None
