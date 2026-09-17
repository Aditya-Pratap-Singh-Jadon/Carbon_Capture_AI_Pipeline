"""Extended Physics-Informed Neural Network (E-PINN) package."""

from carbon_capture.epinn.architecture import EPINNMultiHead, PHYSICAL_OUTPUT_NAMES
from carbon_capture.epinn.model import EPINNModel
from carbon_capture.epinn.losses import EPINNLoss
from carbon_capture.epinn.constraints import AugmentedLagrangianManager, ConstraintStatus
from carbon_capture.epinn.trainer import EPINNTrainer
from carbon_capture.epinn.inference import EPINNInferenceEngine, EPINNInferenceOutput
from carbon_capture.epinn.uncertainty import UncertaintyEstimator, UncertaintyResult

__all__ = [
    "EPINNMultiHead",
    "PHYSICAL_OUTPUT_NAMES",
    "EPINNModel",
    "EPINNLoss",
    "AugmentedLagrangianManager",
    "ConstraintStatus",
    "EPINNTrainer",
    "EPINNInferenceEngine",
    "EPINNInferenceOutput",
    "UncertaintyEstimator",
    "UncertaintyResult",
]
