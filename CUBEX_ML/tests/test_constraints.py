"""Unit tests for Augmented Lagrangian Inequality Constraint System."""

import pytest
import torch
from carbon_capture.epinn.constraints import AugmentedLagrangianManager
from carbon_capture.input.canonical_order import CANONICAL_INPUT_ORDER

def test_constraint_evaluation_and_dual_ascent():
    manager = AugmentedLagrangianManager(initial_mu=1.0, initial_lambda=0.0)

    # Synthetic violation: purity = 1.30 (exceeds 1.0 by 0.30)
    preds = {"E_reduced": torch.tensor([50.0])}
    inputs = {col: torch.tensor([10.0]) for col in CANONICAL_INPUT_ORDER}
    inputs["eta_purity"] = torch.tensor([1.30])  # Violates eta_purity <= 1.0
    inputs["F_SME"] = torch.tensor([1.05])
    inputs["gamma_slip"] = torch.tensor([0.02])
    inputs["eta_boiler"] = torch.tensor([0.85])
    inputs["F_valid"] = torch.tensor([0.95])
    inputs["F_perm"] = torch.tensor([0.98])
    inputs["alpha"] = torch.tensor([0.90])

    loss, viol_dict = manager.compute_constraint_loss(preds, inputs)
    assert loss > 0.0
    assert viol_dict["c_eta_purity_upper"] > 0.25

    # Run dual ascent
    statuses = manager.update_dual_variables(viol_dict)
    purity_status = next(s for s in statuses if s.name == "c_eta_purity_upper")
    assert purity_status.lambda_val > 0.0
    assert not purity_status.satisfied

    # Second epoch with stagnating violation should increase mu
    statuses_2 = manager.update_dual_variables(viol_dict)
    purity_status_2 = next(s for s in statuses_2 if s.name == "c_eta_purity_upper")
    assert purity_status_2.mu_val > 1.0
