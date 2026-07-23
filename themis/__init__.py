"""THEMIS - The Parametric Legal Intelligence Engine for Indian Law."""

from .config import config
from .infer import ThemisInference, generate_response, get_inference, load_model

__version__ = "1.0.0"

__all__ = [
    "load_model",
    "generate_response",
    "get_inference",
    "ThemisInference",
    "config",
]
