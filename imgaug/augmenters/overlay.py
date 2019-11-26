"""Alias for module blend.

Deprecated module. Original name for module blend.py. Was changed in 0.2.8.

"""
from __future__ import print_function, division, absolute_import

import imgaug as ia
from . import blend


@ia.deprecated(alt_func="imgaug.augmenters.blend.blend_alpha()",
               comment="It has the exactly same interface.")
def blend_alpha(*args, **kwargs):
    """See :func:`imgaug.augmenters.blend.blend_alpha`."""
    # pylint: disable=invalid-name
    return blend.blend_alpha(*args, **kwargs)


@ia.deprecated(alt_func="imgaug.augmenters.blend.Alpha",
               comment="It has the exactly same interface.")
def Alpha(*args, **kwargs):
    """See :func:`imgaug.augmenters.blend.Alpha`."""
    # pylint: disable=invalid-name
    return blend.Alpha(*args, **kwargs)


@ia.deprecated(alt_func="imgaug.augmenters.blend.AlphaElementwise",
               comment="It has the exactly same interface.")
def AlphaElementwise(*args, **kwargs):
    """See :func:`imgaug.augmenters.blend.AlphaElementwise`."""
    # pylint: disable=invalid-name
    return blend.AlphaElementwise(*args, **kwargs)


@ia.deprecated(alt_func="imgaug.augmenters.blend.SimplexNoiseAlpha",
               comment="It has the exactly same interface.")
def SimplexNoiseAlpha(*args, **kwargs):
    """See :func:`imgaug.augmenters.blend.SimplexNoiseAlpha`."""
    # pylint: disable=invalid-name
    return blend.SimplexNoiseAlpha(*args, **kwargs)


@ia.deprecated(alt_func="imgaug.augmenters.blend.FrequencyNoiseAlpha",
               comment="It has the exactly same interface.")
def FrequencyNoiseAlpha(*args, **kwargs):
    """See :func:`imgaug.augmenters.blend.FrequencyNoiseAlpha`."""
    # pylint: disable=invalid-name
    return blend.FrequencyNoiseAlpha(*args, **kwargs)
