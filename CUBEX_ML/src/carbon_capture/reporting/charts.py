"""Scientific Analytical Visualization Engine for Carbon Capture Verification."""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from carbon_capture.epinn.inference import EPINNInferenceOutput
from carbon_capture.carbon_credit.calculator import CarbonCreditSummary
from carbon_capture.utils.logging import get_logger

logger = get_logger(__name__)

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

class VerificationChartGenerator:
    """Generates publication-quality analytical plots for verification reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_residual_distributions(
        self,
        epinn_output: EPINNInferenceOutput,
        filename: str = "residuals_distribution.png",
    ) -> Path:
        """Plot horizontal bar chart of mean absolute normalized residuals across all 15 equations."""
        res_names = list(epinn_output.normalized_residuals.keys())
        mean_vals = [float(np.mean(np.abs(epinn_output.normalized_residuals[k]))) for k in res_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2b5c8f" if v <= 1.5 else "#d9534f" for v in mean_vals]
        y_pos = np.arange(len(res_names))

        ax.barh(y_pos, mean_vals, color=colors, alpha=0.85, edgecolor="#1f3c5b")
        ax.axvline(1.5, color="#f0ad4e", linestyle="--", linewidth=1.5, label="Tolerance (1.5 sigma)")
        ax.axvline(3.0, color="#d9534f", linestyle="--", linewidth=1.5, label="Max Violation (3.0 sigma)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(res_names, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Mean Absolute Normalized Residual (|r| / s)", fontsize=11)
        ax.set_title("E-PINN Physics Residual Spectrum (Validation Fidelity)", fontsize=13, pad=12)
        ax.legend(loc="lower right")
        plt.tight_layout()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_carbon_credit_breakdown(
        self,
        credit_summary: CarbonCreditSummary,
        filename: str = "carbon_credit_breakdown.png",
    ) -> Path:
        """Waterfall/bar chart breakdown of benefit terms and penalty deductions."""
        categories = [
            "E_reduced", "E_removed", "E_displaced",
            "PE_aux (-)", "PE_lifecycle (-)", "L_leakage (-)", "CC_T (Net)"
        ]
        values = [
            credit_summary.total_E_reduced,
            credit_summary.total_E_removed,
            credit_summary.total_E_displaced,
            -credit_summary.total_PE_aux,
            -credit_summary.total_PE_lifecycle,
            -credit_summary.total_L_leakage,
            credit_summary.total_CC_T,
        ]
        colors = ["#28a745", "#28a745", "#28a745", "#dc3545", "#dc3545", "#dc3545", "#007bff"]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor="#222")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Metric Tonnes CO2e (tCO2e)", fontsize=11)
        ax.set_title("Authoritative Carbon-Credit Mathematical Accounting Breakdown", fontsize=13, pad=12)
        plt.xticks(rotation=20, ha="right", fontweight="bold")

        # Label bar values
        for bar in bars:
            height = bar.get_height()
            offset = 5 if height >= 0 else -15
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=9,
                fontweight="bold"
            )

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_constraint_violations(
        self,
        epinn_output: EPINNInferenceOutput,
        filename: str = "constraint_violations.png",
    ) -> Path:
        """Bar chart of physical constraint evaluations."""
        c_names = list(epinn_output.constraint_violations.keys())
        c_vals = list(epinn_output.constraint_violations.values())

        fig, ax = plt.subplots(figsize=(11, 6))
        colors = ["#28a745" if v <= 1e-3 else "#dc3545" for v in c_vals]
        y_pos = np.arange(len(c_names))

        ax.barh(y_pos, c_vals, color=colors, alpha=0.85)
        ax.axvline(1e-3, color="#d9534f", linestyle="--", label="Violation Threshold (1e-3)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(c_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Constraint Violation max(0, g_c)", fontsize=11)
        ax.set_title("Augmented Lagrangian Physical Constraint Checks", fontsize=13, pad=12)
        ax.legend(loc="lower right")
        plt.tight_layout()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path
