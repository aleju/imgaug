from __future__ import print_function, division, absolute_import

import warnings
import sys
# unittest only added in 3.4 self.subTest()
if sys.version_info[0] < 3 or sys.version_info[1] < 4:
    import unittest2 as unittest
else:
    import unittest
# unittest.mock is not available in 2.7 (though unittest2 might contain it?)
try:
    import unittest.mock as mock
except ImportError:
    import mock

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np

import imgaug.augmenters as iaa
import imgaug.augmenters.overlay as overlay


class Test_blend_alpha(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        image_fg = np.zeros((1, 1, 3), dtype=np.uint8)
        image_bg = np.copy(image_fg)
        alpha = 1

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = overlay.blend_alpha(image_fg, image_bg, alpha)

        assert len(caught_warnings) == 1
        assert (
            "imgaug.augmenters.blend.blend_alpha"
            in str(caught_warnings[-1].message)
        )


class TestAlpha(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        children_fg = iaa.Identity()
        factor = 1

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = overlay.Alpha(factor, children_fg)

        assert len(caught_warnings) == 1
        assert (
            "imgaug.augmenters.blend.Alpha"
            in str(caught_warnings[-1].message)
        )


class TestAlphaElementwise(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        children_fg = iaa.Identity()
        factor = 1

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = overlay.AlphaElementwise(factor, children_fg)

        assert len(caught_warnings) == 1
        assert (
            "imgaug.augmenters.blend.AlphaElementwise"
            in str(caught_warnings[-1].message)
        )


class TestSimplexNoiseAlpha(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        children_fg = iaa.Identity()

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = overlay.SimplexNoiseAlpha(children_fg)

        assert len(caught_warnings) == 1
        assert (
            "imgaug.augmenters.blend.SimplexNoiseAlpha"
            in str(caught_warnings[-1].message)
        )


class TestFrequencyNoiseAlpha(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        children_fg = iaa.Identity()

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = overlay.FrequencyNoiseAlpha(first=children_fg)

        assert len(caught_warnings) == 1
        assert (
            "imgaug.augmenters.blend.FrequencyNoiseAlpha"
            in str(caught_warnings[-1].message)
        )
