"""THEMIS - The Parametric Legal Intelligence Engine for Indian Law."""

from .infer import load_model, generate_response, get_inference, ThemisInference
from .config import config

__version__ = "1.0.0"

__all__ = [
    "load_model",
    "generate_response",
    "get_inference",
    "ThemisInference",
    "config",
]
