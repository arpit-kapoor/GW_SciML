"""
Neural Operator models for learning operators on function spaces.

This module contains implementations of various neural operator architectures
including Fourier Neural Operators (FNO) and Geometry-Informed Neural Operators (GINO).
"""

from .fno import FNO
from .gino import GINO

__all__ = [
    'FNO',
    'GINO',
]
