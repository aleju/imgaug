"""Alias for module blend.

Deprecated module. Original name for module blend.py. Was changed in 0.2.8.

"""
from __future__ import print_function, division, absolute_import

from . import blend
import imgaug as ia


@ia.deprecated(alt_func="imgaug.augmenters.blend.blend_alpha()",
               comment="It has the exactly same interface.")
def blend_alpha(*args, **kwargs):
    return blend.blend_alpha(*args, **kwargs)


@ia.deprecated(alt_func="imgaug.augmenters.blend.Alpha",
               comment="It has the exactly same interface.")
def Alpha(*args, **kwargs):
    return blend.Alpha(*args, **kwargs)


@ia.deprecated(alt_func="imgaug.augmenters.blend.AlphaElementwise",
               comment="It has the exactly same interface.")
def AlphaElementwise(*args, **kwargs):
    return blend.AlphaElementwise(*args, **kwargs)


@ia.deprecated(alt_func="imgaug.augmenters.blend.SimplexNoiseAlpha",
               comment="It has the exactly same interface.")
def SimplexNoiseAlpha(*args, **kwargs):
    return blend.SimplexNoiseAlpha(*args, **kwargs)


@ia.deprecated(alt_func="imgaug.augmenters.blend.FrequencyNoiseAlpha",
               comment="It has the exactly same interface.")
def FrequencyNoiseAlpha(*args, **kwargs):
    return blend.FrequencyNoiseAlpha(*args, **kwargs)
