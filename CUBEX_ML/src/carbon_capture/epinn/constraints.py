"""Augmented Lagrangian Constraint Enforcement System (Document 1 Section H, M5, O).

Maintains dual variables lambda_c and penalty multipliers mu_c for each physical
and operational inequality constraint g_c(Y) <= 0.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

# Registered Inequality Constraints g_c <= 0
CONSTRAINT_NAMES = [
    "c_e_reduced_nonneg",    # -E_reduced <= 0
    "c_eta_purity_upper",    # eta_purity - 1 <= 0
    "c_eta_purity_lower",    # -eta_purity <= 0
    "c_gamma_slip_upper",    # gamma_slip - 1 <= 0
    "c_gamma_slip_lower",    # -gamma_slip <= 0
    "c_eta_boiler_upper",    # eta_boiler - 1 <= 0
    "c_eta_boiler_lower",    # -eta_boiler <= 0
    "c_f_valid_upper",       # F_valid - 1 <= 0
    "c_f_valid_lower",       # -F_valid <= 0
    "c_f_perm_upper",        # F_perm - 1 <= 0
    "c_f_perm_lower",        # -F_perm <= 0
    "c_alpha_upper",         # alpha - 1 <= 0
    "c_alpha_lower",         # -alpha <= 0
    "c_f_sme_lower",         # 1 - F_SME <= 0  (F_SME >= 1.0)
    "c_m_captured_nonneg",   # -M_captured <= 0
    "c_m_byproduct_nonneg",  # -M_byproduct <= 0
    "c_m_solvent_nonneg",    # -M_solvent_makeup <= 0
    "c_d_k_nonneg",          # -D_k <= 0
    "c_m_trans_nonneg",      # -M_trans_k <= 0
    "c_t_op_nonneg",         # -t_op <= 0
    "c_p_fan_nonneg",        # -P_fan <= 0
    "c_p_pump_nonneg",       # -P_pump <= 0
    "c_h_recovered_nonneg",  # -H_recovered <= 0
]

@dataclass
class ConstraintStatus:
    name: str
    violation: float
    lambda_val: float
    mu_val: float
    satisfied: bool

class AugmentedLagrangianManager:
    """Manages dual variables, penalties, and loss evaluation for inequality constraints."""

    def __init__(
        self,
        constraint_names: Optional[List[str]] = None,
        initial_mu: float = 1.0,
        initial_lambda: float = 0.0,
        mu_inflation_rate: float = 1.5,
        max_mu: float = 1.0e5,
        violation_tolerance: float = 1.0e-3,
    ):
        self.names = constraint_names or list(CONSTRAINT_NAMES)
        self.initial_mu = initial_mu
        self.mu_inflation_rate = mu_inflation_rate
        self.max_mu = max_mu
        self.violation_tolerance = violation_tolerance

        # Dual variables lambda_c and penalty multipliers mu_c
        self.lambdas: Dict[str, float] = {name: initial_lambda for name in self.names}
        self.mus: Dict[str, float] = {name: initial_mu for name in self.names}
        self.prev_violations: Dict[str, float] = {name: float("inf") for name in self.names}

    def compute_constraint_functions(
        self,
        preds: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Evaluate g_c(Y) <= 0 for all constraints."""
        g = {}

        # 1. Output constraints
        g["c_e_reduced_nonneg"] = -preds["E_reduced"]

        # 2. Multipliers and bounded fractions
        g["c_eta_purity_upper"] = inputs["eta_purity"] - 1.0
        g["c_eta_purity_lower"] = -inputs["eta_purity"]

        g["c_gamma_slip_upper"] = inputs["gamma_slip"] - 1.0
        g["c_gamma_slip_lower"] = -inputs["gamma_slip"]

        g["c_eta_boiler_upper"] = inputs["eta_boiler"] - 1.0
        g["c_eta_boiler_lower"] = -inputs["eta_boiler"]

        f_valid_val = inputs.get("F_valid", torch.ones_like(inputs["alpha"]))
        g["c_f_valid_upper"] = f_valid_val - 1.0
        g["c_f_valid_lower"] = -f_valid_val

        g["c_f_perm_upper"] = inputs["F_perm"] - 1.0
        g["c_f_perm_lower"] = -inputs["F_perm"]

        g["c_alpha_upper"] = inputs["alpha"] - 1.0
        g["c_alpha_lower"] = -inputs["alpha"]

        g["c_f_sme_lower"] = 1.0 - inputs["F_SME"]

        # 3. Non-negativity constraints for physical quantities
        g["c_m_captured_nonneg"] = -inputs["M_captured"]
        g["c_m_byproduct_nonneg"] = -inputs["M_byproduct"]
        g["c_m_solvent_nonneg"] = -inputs["M_solvent_makeup"]
        g["c_d_k_nonneg"] = -inputs["D_k"]
        g["c_m_trans_nonneg"] = -inputs["M_trans_k"]
        g["c_t_op_nonneg"] = -inputs["t_op"]
        g["c_p_fan_nonneg"] = -inputs["P_fan"]
        g["c_p_pump_nonneg"] = -inputs["P_pump"]
        g["c_h_recovered_nonneg"] = -inputs["H_recovered"]

        return g

    def compute_constraint_loss(
        self,
        preds: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate L_constraints = sum_c [ mu_c * max(0, g_c)^2 + lambda_c * g_c ]."""
        g_dict = self.compute_constraint_functions(preds, inputs)
        device = preds["E_reduced"].device

        total_loss = torch.tensor(0.0, device=device)
        violations_dict = {}

        for name in self.names:
            g_c = g_dict[name]
            mu_c = self.mus[name]
            lambda_c = self.lambdas[name]

            # max(0, g_c)
            pos_violation = torch.relu(g_c)
            mean_violation = float(pos_violation.mean().item())
            violations_dict[name] = mean_violation

            # Augmented Lagrangian penalty term
            quad_penalty = mu_c * (pos_violation ** 2).mean()
            linear_term = lambda_c * g_c.mean()

            total_loss = total_loss + quad_penalty + linear_term

        return total_loss, violations_dict

    def update_dual_variables(self, violations_dict: Dict[str, float]) -> List[ConstraintStatus]:
        """Update lambda_c by dual ascent and adapt mu_c on stagnating violation."""
        statuses = []
        for name in self.names:
            viol = violations_dict.get(name, 0.0)
            mu_c = self.mus[name]
            lambda_c = self.lambdas[name]

            # Dual ascent: lambda_c <- max(0, lambda_c + mu_c * viol)
            new_lambda = max(0.0, lambda_c + mu_c * viol)
            self.lambdas[name] = new_lambda

            # Check if violation shrunk by at least 10%
            prev_viol = self.prev_violations[name]
            if viol > self.violation_tolerance and viol >= 0.9 * prev_viol:
                # Inflate penalty mu_c
                self.mus[name] = min(self.max_mu, mu_c * self.mu_inflation_rate)

            self.prev_violations[name] = viol
            is_satisfied = viol <= self.violation_tolerance

            statuses.append(
                ConstraintStatus(
                    name=name,
                    violation=viol,
                    lambda_val=new_lambda,
                    mu_val=self.mus[name],
                    satisfied=is_satisfied,
                )
            )

        return statuses

    def get_status_summary(self) -> Dict[str, dict]:
        """Export current constraint parameters for audit logs."""
        return {
            name: {
                "lambda": self.lambdas[name],
                "mu": self.mus[name],
                "last_violation": self.prev_violations[name],
                "satisfied": self.prev_violations[name] <= self.violation_tolerance,
            }
            for name in self.names
        }
