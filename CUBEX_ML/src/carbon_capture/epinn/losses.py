"""E-PINN Loss Function Framework (Document 1 Section O & P, Document 2 Phase 6-7).

Implements:
L_total = L_data + L_physics + L_consistency + w_CC * L_CC + L_constraints

All residuals are individually inspectable and normalized without arbitrary static weights.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from carbon_capture.epinn.constraints import AugmentedLagrangianManager

class EPINNLoss(nn.Module):
    """Loss module for Extended Physics-Informed Neural Network."""

    def __init__(
        self,
        characteristic_scales: Optional[Dict[str, float]] = None,
        w_cc: float = 1.0,
        beta_hinge: float = 10.0,
        pe_aux_mode: str = "verbatim",
    ):
        super().__init__()
        self.characteristic_scales = characteristic_scales or {}
        self.w_cc = w_cc
        self.beta_hinge = beta_hinge
        self.pe_aux_mode = pe_aux_mode

    def _get_scale(self, name: str, default: float = 10.0) -> float:
        s = self.characteristic_scales.get(name, default)
        return max(float(s), 1e-6)

    def compute_differentiable_residuals(
        self,
        preds: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute all 15 residuals differentiably in PyTorch."""
        res = {}

        # 1. r_actual
        fuel_actual = inputs["FC_actual"] * inputs["NCV"] * inputs["EF_CO2"] * inputs["OF"]
        elec_actual = inputs["EC_actual"] * inputs["EF_grid"]
        res["r_actual"] = preds["E_actual"] - (fuel_actual + elec_actual)

        # 2. r_baseline
        fuel_base = inputs["FC_baseline"] * inputs["NCV"] * inputs["EF_CO2"] * inputs["OF"]
        elec_base = inputs["EC_baseline"] * inputs["EF_grid"]
        res["r_baseline"] = preds["E_baseline"] - (fuel_base + elec_base)

        # 3. r_reduced with smooth softplus relaxation (Document 1 Section R.4)
        diff = preds["E_baseline"] - preds["E_actual"]
        smooth_max = F.softplus(diff, beta=self.beta_hinge)
        res["r_reduced"] = preds["E_reduced"] - smooth_max

        # 4. r_removed
        res["r_removed"] = preds["E_removed"] - (inputs["M_captured"] * inputs["P_CO2"] * inputs["eta_purity"])

        # 5. r_disp_chem
        res["r_disp_chem"] = preds["E_displaced_chem"] - (inputs["M_byproduct"] * inputs["EF_virgin_displace"])

        # 6. r_disp_heat
        denom_heat = torch.clamp(inputs["eta_boiler"] * inputs["NCV_fuel"], min=1e-6)
        eq_disp_heat = (inputs["H_recovered"] / denom_heat) * inputs["EF_CO2_fuel"]
        res["r_disp_heat"] = preds["E_displaced_heat"] - eq_disp_heat

        # 7. r_displaced (compositional)
        res["r_displaced"] = preds["E_displaced"] - (preds["E_displaced_chem"] + preds["E_displaced_heat"])

        # 8. r_EChw
        res["r_EChw"] = preds["EC_hardware"] - ((inputs["P_fan"] + inputs["P_pump"]) * (inputs["t_op"] / 3600.0))

        # 9. r_aux
        if self.pe_aux_mode == "physical":
            eq_pe_aux = preds["EC_hardware"] * inputs["EF_grid"]
        else:
            eq_pe_aux = preds["EC_hardware"] - inputs["EF_grid"]
        res["r_aux"] = preds["PE_aux"] - eq_pe_aux

        # 10. r_lifecycle
        res["r_lifecycle"] = preds["PE_lifecycle"] - (inputs["M_solvent_makeup"] * inputs["EF_solvent_LCA"])

        # 11. r_slip
        res["r_slip"] = preds["L_slip"] - (inputs["M_captured"] * inputs["gamma_slip"])

        # 12. r_transport
        res["r_transport"] = preds["L_transport"] - (inputs["D_k"] * inputs["EF_vehicle_k"] * inputs["M_trans_k"])

        # 13. r_leakage (compositional)
        res["r_leakage"] = preds["L_leakage"] - (preds["L_slip"] + preds["L_transport"])

        # 14. r_mult (compositional)
        f_valid_val = inputs.get("F_valid", torch.ones_like(inputs["alpha"]))
        res["r_mult"] = preds["F_multiplier"] - (f_valid_val * inputs["F_SME"] * inputs["F_perm"] * inputs["alpha"])

        # 15. r_CC (top-level accounting consistency)
        bracket = (
            preds["E_reduced"]
            + preds["E_removed"]
            + preds["E_displaced"]
            - preds["PE_aux"]
            - preds["PE_lifecycle"]
            - preds["L_leakage"]
        )
        res["r_CC"] = preds["CC_T"] - (bracket * preds["F_multiplier"])

        return res

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        uncertainties: Optional[Dict[str, torch.Tensor]] = None,
        lagrangian_manager: Optional[AugmentedLagrangianManager] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate complete composite loss and component breakdowns."""
        device = preds["E_reduced"].device
        residuals = self.compute_differentiable_residuals(preds, inputs)

        # 1. Data Loss (Supervised fidelity where targets exist)
        l_data = torch.tensor(0.0, device=device)
        if targets is not None:
            data_losses = []
            for name, target_val in targets.items():
                if name in preds:
                    diff = preds[name] - target_val
                    if uncertainties is not None and name in uncertainties:
                        sigma = torch.clamp(uncertainties[name], min=1e-4)
                        nll = (diff ** 2) / (2.0 * (sigma ** 2)) + torch.log(sigma)
                        data_losses.append(nll.mean())
                    else:
                        s = self._get_scale(name, 10.0)
                        data_losses.append(((diff / s) ** 2).mean())
            if data_losses:
                l_data = torch.stack(data_losses).sum()

        # 2. First-Principles Physics Loss
        first_principles = [
            "r_actual", "r_baseline", "r_reduced", "r_removed",
            "r_disp_chem", "r_disp_heat", "r_EChw", "r_aux",
            "r_lifecycle", "r_slip", "r_transport",
        ]
        physics_losses = []
        for r_name in first_principles:
            r = residuals[r_name]
            s = self._get_scale(r_name, 10.0)
            physics_losses.append(((r / s) ** 2).mean())
        l_physics = torch.stack(physics_losses).sum()

        # 3. Compositional Consistency Loss
        compositional_losses = []
        # r_displaced: normalize by scale of larger operand
        s_disp = torch.clamp(torch.max(torch.abs(preds["E_displaced"]), torch.abs(preds["E_displaced_chem"] + preds["E_displaced_heat"])), min=1e-4)
        compositional_losses.append(((residuals["r_displaced"] / s_disp) ** 2).mean())

        # r_leakage: normalize by scale of larger operand
        s_leak = torch.clamp(torch.max(torch.abs(preds["L_leakage"]), torch.abs(preds["L_slip"] + preds["L_transport"])), min=1e-4)
        compositional_losses.append(((residuals["r_leakage"] / s_leak) ** 2).mean())

        # r_mult: normalize by max(|F_mult|, 1.0)
        s_mult = torch.clamp(torch.abs(preds["F_multiplier"]), min=1.0)
        compositional_losses.append(((residuals["r_mult"] / s_mult) ** 2).mean())

        l_consistency = torch.stack(compositional_losses).sum()

        # 4. Top-level Carbon Credit Consistency Loss
        s_cc = self._get_scale("CC_T", 100.0)
        l_cc = self.w_cc * ((residuals["r_CC"] / s_cc) ** 2).mean()

        # 5. Augmented Lagrangian Constraints Loss
        l_constraints = torch.tensor(0.0, device=device)
        violations_dict = {}
        if lagrangian_manager is not None:
            l_constraints, violations_dict = lagrangian_manager.compute_constraint_loss(preds, inputs)

        # 6. Temporal Consistency Loss (Phase 8)
        # Configurable placeholder: Disabled by default pending scientific dynamic equation derivation.
        # See docs/temporal_modeling.md for details on why a process-specific equation is required.
        l_temporal = torch.tensor(0.0, device=device)

        # Total Loss
        l_total = l_data + l_physics + l_consistency + l_cc + l_constraints + l_temporal

        loss_breakdown = {
            "l_total": l_total,
            "l_data": l_data,
            "l_physics": l_physics,
            "l_consistency": l_consistency,
            "l_cc": l_cc,
            "l_constraints": l_constraints,
            "l_temporal": l_temporal,
        }

        # Also expose individual normalized residual metrics for diagnostics
        for r_name, r_val in residuals.items():
            loss_breakdown[f"metric_{r_name}"] = torch.abs(r_val).mean()

        return l_total, loss_breakdown
