from __future__ import print_function, division, absolute_import

import sys
import warnings
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
import six.moves as sm

from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.testutils import reseed


class TestSuperpixels(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _array_equals_tolerant(cls, a, b, tolerance):
        # TODO isnt this just np.allclose(a, b, rtol=0, atol=tolerance) ?!
        diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
        return np.all(diff <= tolerance)

    @property
    def base_img(self):
        base_img = [
            [255, 255, 255, 0, 0, 0],
            [255, 235, 255, 0, 20, 0],
            [250, 250, 250, 5, 5, 5]
        ]
        base_img = np.tile(
            np.array(base_img, dtype=np.uint8)[..., np.newaxis],
            (1, 1, 3))
        return base_img

    @property
    def base_img_superpixels(self):
        base_img_superpixels = [
            [251, 251, 251, 4, 4, 4],
            [251, 251, 251, 4, 4, 4],
            [251, 251, 251, 4, 4, 4]
        ]
        base_img_superpixels = np.tile(
            np.array(base_img_superpixels, dtype=np.uint8)[..., np.newaxis],
            (1, 1, 3))
        return base_img_superpixels

    @property
    def base_img_superpixels_left(self):
        base_img_superpixels_left = self.base_img_superpixels
        base_img_superpixels_left[:, 3:, :] = self.base_img[:, 3:, :]
        return base_img_superpixels_left

    @property
    def base_img_superpixels_right(self):
        base_img_superpixels_right = self.base_img_superpixels
        base_img_superpixels_right[:, :3, :] = self.base_img[:, :3, :]
        return base_img_superpixels_right

    def test_p_replace_0_n_segments_2(self):
        aug = iaa.Superpixels(p_replace=0, n_segments=2)
        observed = aug.augment_image(self.base_img)
        expected = self.base_img
        assert np.allclose(observed, expected)

    def test_p_replace_1_n_segments_2(self):
        aug = iaa.Superpixels(p_replace=1.0, n_segments=2)
        observed = aug.augment_image(self.base_img)
        expected = self.base_img_superpixels
        assert self._array_equals_tolerant(observed, expected, 2)

    def test_p_replace_1_n_segments_stochastic_parameter(self):
        aug = iaa.Superpixels(p_replace=1.0, n_segments=iap.Deterministic(2))
        observed = aug.augment_image(self.base_img)
        expected = self.base_img_superpixels
        assert self._array_equals_tolerant(observed, expected, 2)

    def test_p_replace_stochastic_parameter_n_segments_2(self):
        aug = iaa.Superpixels(
            p_replace=iap.Binomial(iap.Choice([0.0, 1.0])), n_segments=2)
        observed = aug.augment_image(self.base_img)
        assert (
            np.allclose(observed, self.base_img)
            or self._array_equals_tolerant(
                observed, self.base_img_superpixels, 2)
        )

    def test_p_replace_050_n_segments_2(self):
        aug = iaa.Superpixels(p_replace=0.5, n_segments=2)
        seen = {"none": False, "left": False, "right": False, "both": False}
        for _ in sm.xrange(100):
            observed = aug.augment_image(self.base_img)
            if self._array_equals_tolerant(observed, self.base_img, 2):
                seen["none"] = True
            elif self._array_equals_tolerant(
                    observed, self.base_img_superpixels_left, 2):
                seen["left"] = True
            elif self._array_equals_tolerant(
                    observed, self.base_img_superpixels_right, 2):
                seen["right"] = True
            elif self._array_equals_tolerant(
                    observed, self.base_img_superpixels, 2):
                seen["both"] = True
            else:
                raise Exception(
                    "Generated superpixels image does not match any "
                    "expected image.")
            if all(seen.values()):
                break
        assert np.all(seen.values())

    def test_failure_on_invalid_datatype_for_p_replace(self):
        # note that assertRaisesRegex does not exist in 2.7
        got_exception = False
        try:
            _ = iaa.Superpixels(p_replace="test", n_segments=100)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_failure_on_invalid_datatype_for_n_segments(self):
        # note that assertRaisesRegex does not exist in 2.7
        got_exception = False
        try:
            _ = iaa.Superpixels(p_replace=1, n_segments="test")
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_get_parameters(self):
        aug = iaa.Superpixels(
            p_replace=0.5, n_segments=2, max_size=100, interpolation="nearest")
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Binomial)
        assert isinstance(params[0].p, iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert 0.5 - 1e-4 < params[0].p.value < 0.5 + 1e-4
        assert params[1].value == 2
        assert params[2] == 100
        assert params[3] == "nearest"

    def test_other_dtypes_bool(self):
        aug = iaa.Superpixels(p_replace=1.0, n_segments=2)
        img = np.array([
            [False, False, True, True],
            [False, False, True, True]
        ], dtype=bool)
        img_aug = aug.augment_image(img)
        assert img_aug.dtype == img.dtype
        assert np.all(img_aug == img)

        aug = iaa.Superpixels(p_replace=1.0, n_segments=1)
        img = np.array([
            [True, True, True, True],
            [False, True, True, True]
        ], dtype=bool)
        img_aug = aug.augment_image(img)
        assert img_aug.dtype == img.dtype
        assert np.all(img_aug)

    def test_other_dtypes_uint_int(self):
        for dtype in [np.uint8, np.uint16, np.uint32,
                      np.int8, np.int16, np.int32]:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            if np.dtype(dtype).kind == "i":
                values = [int(center_value), int(0.1 * max_value),
                          int(0.2 * max_value), int(0.5 * max_value),
                          max_value-100]
                values = [((-1)*value, value) for value in values]
            else:
                values = [(0, int(center_value)),
                          (10, int(0.1 * max_value)),
                          (10, int(0.2 * max_value)),
                          (10, int(0.5 * max_value)),
                          (0, max_value),
                          (int(center_value),
                           max_value)]

            for v1, v2 in values:
                aug = iaa.Superpixels(p_replace=1.0, n_segments=2)
                img = np.array([
                    [v1, v1, v2, v2],
                    [v1, v1, v2, v2]
                ], dtype=dtype)
                img_aug = aug.augment_image(img)
                assert img_aug.dtype == np.dtype(dtype)
                assert np.array_equal(img_aug, img)

                aug = iaa.Superpixels(p_replace=1.0, n_segments=1)
                img = np.array([
                    [v2, v2, v2, v2],
                    [v1, v2, v2, v2]
                ], dtype=dtype)
                img_aug = aug.augment_image(img)
                assert img_aug.dtype == np.dtype(dtype)
                assert np.all(img_aug == int(np.round((7/8)*v2 + (1/8)*v1)))

    def test_other_dtypes_float(self):
        # currently, no float dtype is actually accepted
        for dtype in []:
            def _allclose(a, b):
                atol = 1e-4 if dtype == np.float16 else 1e-8
                return np.allclose(a, b, atol=atol, rtol=0)

            isize = np.dtype(dtype).itemsize
            for value in [0, 1.0, 10.0, 1000 ** (isize - 1)]:
                v1 = (-1) * value
                v2 = value

                aug = iaa.Superpixels(p_replace=1.0, n_segments=2)
                img = np.array([
                    [v1, v1, v2, v2],
                    [v1, v1, v2, v2]
                ], dtype=dtype)
                img_aug = aug.augment_image(img)
                assert img_aug.dtype == np.dtype(dtype)
                assert _allclose(img_aug, img)

                aug = iaa.Superpixels(p_replace=1.0, n_segments=1)
                img = np.array([
                    [v2, v2, v2, v2],
                    [v1, v2, v2, v2]
                ], dtype=dtype)
                img_aug = aug.augment_image(img)
                assert img_aug.dtype == np.dtype(dtype)
                assert _allclose(img_aug, (7/8)*v2 + (1/8)*v1)


class Test_segment_voronoi(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_cell_coordinates_is_empty_integrationtest(self):
        image = np.arange(2*2*3).astype(np.uint8).reshape((2, 2, 3))
        cell_coordinates = np.zeros((0, 2), dtype=np.float32)
        replace_mask = np.zeros((0,), dtype=bool)

        image_seg = iaa.segment_voronoi(image, cell_coordinates, replace_mask)

        assert np.array_equal(image, image_seg)

    @classmethod
    def _test_image_n_channels_integrationtest(cls, nb_channels):
        image = np.uint8([
            [0, 1, 200, 201],
            [2, 3, 202, 203]
        ])
        if nb_channels is not None:
            image = np.tile(image[:, :, np.newaxis], (1, 1, nb_channels))
            for c in sm.xrange(nb_channels):
                image[..., c] += c
        cell_coordinates = np.float32([
            [1.0, 1.0],
            [3.0, 1.0]
        ])
        replace_mask = np.array([True, True], dtype=bool)

        image_seg = iaa.segment_voronoi(image, cell_coordinates, replace_mask)

        pixels1 = image[0:2, 0:2]
        pixels2 = image[0:2, 2:4]
        avg_color1 = np.average(pixels1.astype(np.float32), axis=(0, 1))
        avg_color2 = np.average(pixels2.astype(np.float32), axis=(0, 1))
        image_expected = np.uint8([
            [avg_color1, avg_color1, avg_color2, avg_color2],
            [avg_color1, avg_color1, avg_color2, avg_color2],
        ])

        assert np.array_equal(image_seg, image_expected)

    def test_image_has_no_channels_integrationtest(self):
        self._test_image_n_channels_integrationtest(None)

    def test_image_has_one_channel_integrationtest(self):
        self._test_image_n_channels_integrationtest(1)

    def test_image_has_three_channels_integrationtest(self):
        self._test_image_n_channels_integrationtest(3)

    def test_replace_mask_is_all_false_integrationtest(self):
        image = np.uint8([
            [0, 1, 200, 201],
            [2, 3, 202, 203]
        ])
        cell_coordinates = np.float32([
            [1.0, 1.0],
            [3.0, 1.0]
        ])
        replace_mask = np.array([False, False], dtype=bool)

        image_seg = iaa.segment_voronoi(image, cell_coordinates, replace_mask)

        assert np.array_equal(image_seg, image)

    def test_replace_mask_is_mixed_integrationtest(self):
        image = np.uint8([
            [0, 1, 200, 201],
            [2, 3, 202, 203]
        ])
        cell_coordinates = np.float32([
            [1.0, 1.0],
            [3.0, 1.0]
        ])
        replace_mask = np.array([False, True], dtype=bool)

        image_seg = iaa.segment_voronoi(image, cell_coordinates, replace_mask)

        pixels2 = image[0:2, 2:4]
        avg_color2 = np.sum(pixels2).astype(np.float32) / pixels2.size
        image_expected = np.uint8([
            [0, 1, avg_color2, avg_color2],
            [2, 3, avg_color2, avg_color2],
        ])
        assert np.array_equal(image_seg, image_expected)

    def test_replace_mask_is_none_integrationtest(self):
        image = np.uint8([
            [0, 1, 200, 201],
            [2, 3, 202, 203]
        ])
        cell_coordinates = np.float32([
            [1.0, 1.0],
            [3.0, 1.0]
        ])
        replace_mask = None

        image_seg = iaa.segment_voronoi(image, cell_coordinates, replace_mask)

        pixels1 = image[0:2, 0:2]
        pixels2 = image[0:2, 2:4]
        avg_color1 = np.sum(pixels1).astype(np.float32) / pixels1.size
        avg_color2 = np.sum(pixels2).astype(np.float32) / pixels2.size
        image_expected = np.uint8([
            [avg_color1, avg_color1, avg_color2, avg_color2],
            [avg_color1, avg_color1, avg_color2, avg_color2],
        ])
        assert np.array_equal(image_seg, image_expected)

    def test_no_cell_coordinates_provided_and_no_channel_integrationtest(self):
        image = np.uint8([
            [0, 1, 200, 201],
            [2, 3, 202, 203]
        ])
        cell_coordinates = np.zeros((0, 2), dtype=np.float32)
        replace_mask = np.zeros((0,), dtype=bool)

        image_seg = iaa.segment_voronoi(image, cell_coordinates, replace_mask)

        assert np.array_equal(image_seg, image)

    def test_no_cell_coordinates_provided_and_3_channels_integrationtest(self):
        image = np.uint8([
            [0, 1, 200, 201],
            [2, 3, 202, 203]
        ])
        image = np.tile(image[..., np.newaxis], (1, 1, 3))
        cell_coordinates = np.zeros((0, 2), dtype=np.float32)
        replace_mask = np.zeros((0,), dtype=bool)

        image_seg = iaa.segment_voronoi(image, cell_coordinates, replace_mask)

        assert np.array_equal(image_seg, image)

    def test_image_with_zero_height(self):
        image = np.zeros((0, 4, 3), dtype=np.uint8)
        cell_coordinates = np.float32([
            [1.0, 1.0],
            [3.0, 1.0]
        ])
        replace_mask = np.array([True, True], dtype=bool)

        image_seg = iaa.segment_voronoi(image, cell_coordinates, replace_mask)

        assert np.array_equal(image_seg, image)

    def test_image_with_zero_width(self):
        image = np.zeros((4, 0, 3), dtype=np.uint8)
        cell_coordinates = np.float32([
            [1.0, 1.0],
            [3.0, 1.0]
        ])
        replace_mask = np.array([True, True], dtype=bool)

        image_seg = iaa.segment_voronoi(image, cell_coordinates, replace_mask)

        assert np.array_equal(image_seg, image)

    def test_image_with_zero_size(self):
        image = np.zeros((0, 0), dtype=np.uint8)
        cell_coordinates = np.float32([
            [1.0, 1.0],
            [3.0, 1.0]
        ])
        replace_mask = np.array([True, True], dtype=bool)

        image_seg = iaa.segment_voronoi(image, cell_coordinates, replace_mask)

        assert np.array_equal(image_seg, image)
