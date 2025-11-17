"""
pyCurveWave
-----------

A Python library for Curvelet and Wavelet-based image augmentations and blending.

Author: Nehal Mantri
"""

from .core import wavelet_augment, curvelet_augment_color, blend,start_matlab_engine

__all__ = ["wavelet_augment", "curvelet_augment_color", "blend","start_matlab_engine"]
__version__ = "0.1.0"
