"""Alias for module blend.

Deprecated module. Original name for module blend.py. Was changed in 0.2.8.
"""
from __future__ import print_function, division, absolute_import
import warnings
from . import blend

DEPRECATION_WARNING = "Usage of imgaug.augmenters.overlay.%s() is deprecated. " \
                      "Use imgaug.augmenters.blend.%s() instead. " \
                      "(Same interface, simple renaming.)"


def blend_alpha(*args, **kwargs):
    warnings.warn(DeprecationWarning(DEPRECATION_WARNING % ("blend_alpha", "blend_alpha")))
    return blend.blend_alpha(*args, **kwargs)


def Alpha(*args, **kwargs):
    warnings.warn(DeprecationWarning(DEPRECATION_WARNING % ("Alpha", "Alpha")))
    return blend.Alpha(*args, **kwargs)


def AlphaElementwise(*args, **kwargs):
    warnings.warn(DeprecationWarning(DEPRECATION_WARNING % ("AlphaElementwise", "AlphaElementwise")))
    return blend.AlphaElementwise(*args, **kwargs)


def SimplexNoiseAlpha(*args, **kwargs):
    warnings.warn(DeprecationWarning(DEPRECATION_WARNING % ("SimplexNoiseAlpha", "SimplexNoiseAlpha")))
    return blend.SimplexNoiseAlpha(*args, **kwargs)


def FrequencyNoiseAlpha(*args, **kwargs):
    warnings.warn(DeprecationWarning(DEPRECATION_WARNING % ("FrequencyNoiseAlpha", "FrequencyNoiseAlpha")))
    return blend.FrequencyNoiseAlpha(*args, **kwargs)
