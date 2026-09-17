"""Custom exception hierarchy for the Carbon Capture AI pipeline."""

class CarbonCaptureError(Exception):
    """Base exception for all carbon capture pipeline errors."""
    pass

class SchemaValidationError(CarbonCaptureError):
    """Raised when incoming data violates the canonical schema or column specifications."""
    pass

class PhysicsConsistencyError(CarbonCaptureError):
    """Raised when process telemetry violates fundamental physical or thermodynamic equations."""
    pass

class ConstraintViolationError(CarbonCaptureError):
    """Raised when physical or operational boundaries are violated beyond tolerance."""
    pass

class UnitMismatchError(CarbonCaptureError):
    """Raised when an unrecognized or inconsistent unit is declared."""
    pass

class ModelPersistenceError(CarbonCaptureError):
    """Raised when saving or loading model artifacts and scalers fails."""
    pass
