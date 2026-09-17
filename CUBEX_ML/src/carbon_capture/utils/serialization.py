"""Serialization utilities for models, scalers, and configuration metadata."""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Union
import yaml
import torch
from carbon_capture.utils.exceptions import ModelPersistenceError

def save_artifact(obj: Any, file_path: Union[str, Path], artifact_type: str = "pickle") -> None:
    """Save an object to disk according to artifact type."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if artifact_type == "torch":
            torch.save(obj, path)
        elif artifact_type == "pickle":
            with open(path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif artifact_type == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=str)
        elif artifact_type == "yaml":
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(obj, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
    except Exception as e:
        raise ModelPersistenceError(f"Failed saving {artifact_type} to {file_path}: {e}") from e

def load_artifact(file_path: Union[str, Path], artifact_type: str = "pickle") -> Any:
    """Load an object from disk according to artifact type."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {file_path}")
        
    try:
        if artifact_type == "torch":
            return torch.load(path, map_location="cpu", weights_only=False)
        elif artifact_type == "pickle":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif artifact_type == "json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif artifact_type == "yaml":
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
    except Exception as e:
        raise ModelPersistenceError(f"Failed loading {artifact_type} from {file_path}: {e}") from e
