"""Thermal pipe simulation package split from the legacy monolithic solver file."""

from .config import params, validate_params
from .numerics import HAS_NUMBA

__all__ = ["HAS_NUMBA", "params", "validate_params"]
