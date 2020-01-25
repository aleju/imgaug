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
from imgaug import random as iarandom
from imgaug.testutils import reseed, runtest_pickleable_uint8_img


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
            if np.all(seen.values()):
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

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 128, dtype=np.uint8)
                aug = iaa.Superpixels(p_replace=1.0, n_segments=10)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 128, dtype=np.uint8)
                aug = iaa.Superpixels(p_replace=1.0, n_segments=10)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

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

    def test_pickleable(self):
        aug = iaa.Superpixels(p_replace=0.5, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10, shape=(25, 25, 1))


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

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        cell_coordinates = np.float32([
            [1.0, 1.0],
            [3.0, 1.0]
        ])
        replace_mask = np.array([True, True], dtype=bool)

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 128, dtype=np.uint8)

                image_aug = iaa.segment_voronoi(image, cell_coordinates,
                                                replace_mask)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        cell_coordinates = np.float32([
            [1.0, 1.0],
            [3.0, 1.0]
        ])
        replace_mask = np.array([True, True], dtype=bool)

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 128, dtype=np.uint8)

                image_aug = iaa.segment_voronoi(image, cell_coordinates,
                                                replace_mask)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape


class TestVoronoi(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___defaults(self):
        sampler = iaa.RegularGridPointsSampler(1, 1)
        aug = iaa.Voronoi(sampler)
        assert aug.points_sampler is sampler
        assert isinstance(aug.p_replace, iap.Deterministic)
        assert aug.p_replace.value == 1
        assert aug.max_size == 128
        assert aug.interpolation == "linear"

    def test___init___custom_arguments(self):
        sampler = iaa.RegularGridPointsSampler(1, 1)
        aug = iaa.Voronoi(sampler, p_replace=0.5, max_size=None,
                          interpolation="cubic")
        assert aug.points_sampler is sampler
        assert isinstance(aug.p_replace, iap.Binomial)
        assert np.isclose(aug.p_replace.p.value, 0.5)
        assert aug.max_size is None
        assert aug.interpolation == "cubic"

    def test_max_size_is_none(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(1, 1)
        aug = iaa.Voronoi(sampler, max_size=None)

        mock_imresize = mock.MagicMock()
        mock_imresize.return_value = image

        fname = "imgaug.imresize_single_image"
        with mock.patch(fname, mock_imresize):
            _image_aug = aug(image=image)

        assert mock_imresize.call_count == 0

    def test_max_size_is_int_image_not_too_large(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(1, 1)
        aug = iaa.Voronoi(sampler, max_size=100)

        mock_imresize = mock.MagicMock()
        mock_imresize.return_value = image

        fname = "imgaug.imresize_single_image"
        with mock.patch(fname, mock_imresize):
            _image_aug = aug(image=image)

        assert mock_imresize.call_count == 0

    def test_max_size_is_int_image_too_large(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(1, 1)
        aug = iaa.Voronoi(sampler, max_size=10)

        mock_imresize = mock.MagicMock()
        mock_imresize.return_value = image

        fname = "imgaug.imresize_single_image"
        with mock.patch(fname, mock_imresize):
            _image_aug = aug(image=image)

        assert mock_imresize.call_count == 1
        assert mock_imresize.call_args_list[0][0][1] == (5, 10)

    def test_interpolation(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(1, 1)
        aug = iaa.Voronoi(sampler, max_size=10, interpolation="cubic")

        mock_imresize = mock.MagicMock()
        mock_imresize.return_value = image

        fname = "imgaug.imresize_single_image"
        with mock.patch(fname, mock_imresize):
            _image_aug = aug(image=image)

        assert mock_imresize.call_count == 1
        assert mock_imresize.call_args_list[0][1]["interpolation"] == "cubic"

    def test_point_sampler_called(self):
        class LoggedPointSampler(iaa.IPointsSampler):
            def __init__(self, other):
                self.other = other
                self.call_count = 0

            def sample_points(self, images, random_state):
                self.call_count += 1
                return self.other.sample_points(images, random_state)

        image = np.zeros((10, 20, 3), dtype=np.uint8)
        sampler = LoggedPointSampler(iaa.RegularGridPointsSampler(1, 1))
        aug = iaa.Voronoi(sampler)

        _image_aug = aug(image=image)

        assert sampler.call_count == 1

    def test_point_sampler_returns_no_points_integrationtest(self):
        class NoPointsPointSampler(iaa.IPointsSampler):
            def sample_points(self, images, random_state):
                return [np.zeros((0, 2), dtype=np.float32)]

        image = np.zeros((10, 20, 3), dtype=np.uint8)
        sampler = NoPointsPointSampler()
        aug = iaa.Voronoi(sampler)

        image_aug = aug(image=image)

        assert np.array_equal(image_aug, image)

    @classmethod
    def _test_image_with_n_channels(cls, nb_channels):
        image = np.zeros((10, 20), dtype=np.uint8)
        if nb_channels is not None:
            image = image[..., np.newaxis]
            image = np.tile(image, (1, 1, nb_channels))
        sampler = iaa.RegularGridPointsSampler(1, 1)
        aug = iaa.Voronoi(sampler)

        mock_segment_voronoi = mock.MagicMock()
        if nb_channels is None:
            mock_segment_voronoi.return_value = image[..., np.newaxis]
        else:
            mock_segment_voronoi.return_value = image

        fname = "imgaug.augmenters.segmentation.segment_voronoi"
        with mock.patch(fname, mock_segment_voronoi):
            image_aug = aug(image=image)

        assert image_aug.shape == image.shape

    def test_image_with_no_channels(self):
        self._test_image_with_n_channels(None)

    def test_image_with_one_channel(self):
        self._test_image_with_n_channels(1)

    def test_image_with_three_channels(self):
        self._test_image_with_n_channels(3)

    def test_p_replace_is_zero(self):
        image = np.zeros((50, 50), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(50, 50)
        aug = iaa.Voronoi(sampler, p_replace=0.0)

        mock_segment_voronoi = mock.MagicMock()
        mock_segment_voronoi.return_value = image[..., np.newaxis]

        fname = "imgaug.augmenters.segmentation.segment_voronoi"
        with mock.patch(fname, mock_segment_voronoi):
            _image_aug = aug(image=image)

        replace_mask = mock_segment_voronoi.call_args_list[0][0][2]
        assert not np.any(replace_mask)

    def test_p_replace_is_one(self):
        image = np.zeros((50, 50), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(50, 50)
        aug = iaa.Voronoi(sampler, p_replace=1.0)

        mock_segment_voronoi = mock.MagicMock()
        mock_segment_voronoi.return_value = image[..., np.newaxis]

        fname = "imgaug.augmenters.segmentation.segment_voronoi"
        with mock.patch(fname, mock_segment_voronoi):
            _image_aug = aug(image=image)

        replace_mask = mock_segment_voronoi.call_args_list[0][0][2]
        assert np.all(replace_mask)

    def test_p_replace_is_50_percent(self):
        image = np.zeros((200, 200), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(200, 200)
        aug = iaa.Voronoi(sampler, p_replace=0.5)

        mock_segment_voronoi = mock.MagicMock()
        mock_segment_voronoi.return_value = image[..., np.newaxis]

        fname = "imgaug.augmenters.segmentation.segment_voronoi"
        with mock.patch(fname, mock_segment_voronoi):
            _image_aug = aug(image=image)

        replace_mask = mock_segment_voronoi.call_args_list[0][0][2]
        replace_fraction = np.average(replace_mask.astype(np.float32))
        assert 0.4 <= replace_fraction <= 0.6

    def test_determinism_integrationtest(self):
        image = np.arange(10*20).astype(np.uint8).reshape((10, 20, 1))
        image = np.tile(image, (1, 1, 3))
        image[:, :, 1] += 5
        image[:, :, 2] += 10
        sampler = iaa.DropoutPointsSampler(
            iaa.RegularGridPointsSampler((1, 10), (1, 20)),
            0.5
        )
        aug = iaa.Voronoi(sampler, p_replace=(0.0, 1.0))
        aug_det = aug.to_deterministic()

        images_aug_a1 = aug(images=[image] * 50)
        images_aug_a2 = aug(images=[image] * 50)

        images_aug_b1 = aug_det(images=[image] * 50)
        images_aug_b2 = aug_det(images=[image] * 50)

        same_within_a1 = _all_arrays_identical(images_aug_a1)
        same_within_a2 = _all_arrays_identical(images_aug_a2)

        same_within_b1 = _all_arrays_identical(images_aug_b1)
        same_within_b2 = _all_arrays_identical(images_aug_b2)

        same_between_a1_a2 = _array_lists_elementwise_identical(images_aug_a1,
                                                                images_aug_a2)
        same_between_b1_b2 = _array_lists_elementwise_identical(images_aug_b1,
                                                                images_aug_b2)

        assert not same_within_a1
        assert not same_within_a2
        assert not same_within_b1
        assert not same_within_b2

        assert not same_between_a1_a2
        assert same_between_b1_b2

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        sampler = iaa.RegularGridPointsSampler(50, 50)

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 128, dtype=np.uint8)
                aug = iaa.Voronoi(sampler, p_replace=1)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        sampler = iaa.RegularGridPointsSampler(50, 50)

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 128, dtype=np.uint8)
                aug = iaa.Voronoi(sampler, p_replace=1)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_get_parameters(self):
        sampler = iaa.RegularGridPointsSampler(1, 1)
        aug = iaa.Voronoi(sampler, p_replace=0.5, max_size=None,
                          interpolation="cubic")
        params = aug.get_parameters()
        assert params[0] is sampler
        assert isinstance(params[1], iap.Binomial)
        assert np.isclose(params[1].p.value, 0.5)
        assert params[2] is None
        assert params[3] == "cubic"

    def test_pickleable(self):
        sampler = iaa.RegularGridPointsSampler(5, 5)
        aug = iaa.Voronoi(sampler, p_replace=0.5, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10, shape=(25, 25, 1))


def _all_arrays_identical(arrs):
    if len(arrs) == 1:
        return True

    return np.all([np.array_equal(arrs[0], arr_other)
                   for arr_other in arrs[1:]])


def _array_lists_elementwise_identical(arrs1, arrs2):
    return np.all([np.array_equal(arr1, arr2)
                   for arr1, arr2 in zip(arrs1, arrs2)])


class TestUniformVoronoi(unittest.TestCase):
    def test___init___(self):
        rs = iarandom.RNG(10)

        mock_voronoi = mock.MagicMock()
        mock_voronoi.return_value = mock_voronoi
        fname = "imgaug.augmenters.segmentation.Voronoi.__init__"
        with mock.patch(fname, mock_voronoi):
            _ = iaa.UniformVoronoi(
                100,
                p_replace=0.5,
                max_size=5,
                interpolation="cubic",
                seed=rs,
                name="foo"
            )

        assert mock_voronoi.call_count == 1
        assert isinstance(mock_voronoi.call_args_list[0][1]["points_sampler"],
                          iaa.UniformPointsSampler)
        assert np.isclose(mock_voronoi.call_args_list[0][1]["p_replace"],
                          0.5)
        assert mock_voronoi.call_args_list[0][1]["max_size"] == 5
        assert mock_voronoi.call_args_list[0][1]["interpolation"] == "cubic"
        assert mock_voronoi.call_args_list[0][1]["name"] == "foo"
        assert mock_voronoi.call_args_list[0][1]["seed"] is rs

    def test___init___integrationtest(self):
        rs = iarandom.RNG(10)
        aug = iaa.UniformVoronoi(
            100,
            p_replace=0.5,
            max_size=5,
            interpolation="cubic",
            seed=rs,
            name=None
        )
        assert aug.points_sampler.n_points.value == 100
        assert np.isclose(aug.p_replace.p.value, 0.5)
        assert aug.max_size == 5
        assert aug.interpolation == "cubic"
        assert aug.name == "UnnamedUniformVoronoi"
        assert aug.random_state.equals(rs)

    def test_pickleable(self):
        aug = iaa.UniformVoronoi((10, 50), p_replace=0.5, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3, shape=(50, 50, 1))


class TestRegularGridVoronoi(unittest.TestCase):
    def test___init___(self):
        rs = iarandom.RNG(10)

        mock_voronoi = mock.MagicMock()
        mock_voronoi.return_value = mock_voronoi
        fname = "imgaug.augmenters.segmentation.Voronoi.__init__"
        with mock.patch(fname, mock_voronoi):
            _ = iaa.RegularGridVoronoi(
                10,
                20,
                p_drop_points=0.6,
                p_replace=0.5,
                max_size=5,
                interpolation="cubic",
                seed=rs,
                name="foo"
            )

        assert mock_voronoi.call_count == 1
        ps = mock_voronoi.call_args_list[0][1]["points_sampler"]
        assert isinstance(ps, iaa.DropoutPointsSampler)
        assert isinstance(ps.other_points_sampler,
                          iaa.RegularGridPointsSampler)
        assert np.isclose(ps.p_drop.p.value, 1-0.6)
        assert ps.other_points_sampler.n_rows.value == 10
        assert ps.other_points_sampler.n_cols.value == 20
        assert np.isclose(mock_voronoi.call_args_list[0][1]["p_replace"],
                          0.5)
        assert mock_voronoi.call_args_list[0][1]["max_size"] == 5
        assert mock_voronoi.call_args_list[0][1]["interpolation"] == "cubic"
        assert mock_voronoi.call_args_list[0][1]["name"] == "foo"
        assert mock_voronoi.call_args_list[0][1]["seed"] is rs

    def test___init___integrationtest(self):
        rs = iarandom.RNG(10)
        aug = iaa.RegularGridVoronoi(
            10,
            (10, 30),
            p_replace=0.5,
            max_size=5,
            interpolation="cubic",
            seed=rs,
            name=None
        )
        assert aug.points_sampler.other_points_sampler.n_rows.value == 10
        assert isinstance(aug.points_sampler.other_points_sampler.n_cols,
                          iap.DiscreteUniform)
        assert aug.points_sampler.other_points_sampler.n_cols.a.value == 10
        assert aug.points_sampler.other_points_sampler.n_cols.b.value == 30
        assert np.isclose(aug.p_replace.p.value, 0.5)
        assert aug.max_size == 5
        assert aug.interpolation == "cubic"
        assert aug.name == "UnnamedRegularGridVoronoi"
        assert aug.random_state.equals(rs)

    def test_pickleable(self):
        aug = iaa.RegularGridVoronoi((5, 10), (5, 10), p_replace=0.5,
                                     seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3, shape=(50, 50, 1))


class TestRelativeRegularGridVoronoi(unittest.TestCase):
    def test___init___(self):
        rs = iarandom.RNG(10)

        mock_voronoi = mock.MagicMock()
        mock_voronoi.return_value = mock_voronoi
        fname = "imgaug.augmenters.segmentation.Voronoi.__init__"
        with mock.patch(fname, mock_voronoi):
            _ = iaa.RelativeRegularGridVoronoi(
                0.1,
                0.2,
                p_drop_points=0.6,
                p_replace=0.5,
                max_size=5,
                interpolation="cubic",
                seed=rs,
                name="foo"
            )

        assert mock_voronoi.call_count == 1
        ps = mock_voronoi.call_args_list[0][1]["points_sampler"]
        assert isinstance(ps, iaa.DropoutPointsSampler)
        assert isinstance(ps.other_points_sampler,
                          iaa.RelativeRegularGridPointsSampler)
        assert np.isclose(ps.p_drop.p.value, 1-0.6)
        assert np.isclose(ps.other_points_sampler.n_rows_frac.value, 0.1)
        assert np.isclose(ps.other_points_sampler.n_cols_frac.value, 0.2)
        assert np.isclose(mock_voronoi.call_args_list[0][1]["p_replace"],
                          0.5)
        assert mock_voronoi.call_args_list[0][1]["max_size"] == 5
        assert mock_voronoi.call_args_list[0][1]["interpolation"] == "cubic"
        assert mock_voronoi.call_args_list[0][1]["name"] == "foo"
        assert mock_voronoi.call_args_list[0][1]["seed"] is rs

    def test___init___integrationtest(self):
        rs = iarandom.RNG(10)
        aug = iaa.RelativeRegularGridVoronoi(
            0.1,
            (0.1, 0.3),
            p_replace=0.5,
            max_size=5,
            interpolation="cubic",
            seed=rs,
            name=None
        )

        ps = aug.points_sampler
        assert np.isclose(ps.other_points_sampler.n_rows_frac.value, 0.1)
        assert isinstance(ps.other_points_sampler.n_cols_frac, iap.Uniform)
        assert np.isclose(ps.other_points_sampler.n_cols_frac.a.value, 0.1)
        assert np.isclose(ps.other_points_sampler.n_cols_frac.b.value, 0.3)
        assert np.isclose(aug.p_replace.p.value, 0.5)
        assert aug.max_size == 5
        assert aug.interpolation == "cubic"
        assert aug.name == "UnnamedRelativeRegularGridVoronoi"
        assert aug.random_state.equals(rs)

    def test_pickleable(self):
        aug = iaa.RelativeRegularGridVoronoi((0.01, 0.2), (0.01, 0.2),
                                             p_replace=0.5, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3, shape=(50, 50, 1))


# TODO verify behaviours when image height/width is zero
class TestRegularGridPointsSampler(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___(self):
        sampler = iaa.RegularGridPointsSampler((1, 10), 20)
        assert isinstance(sampler.n_rows, iap.DiscreteUniform)
        assert sampler.n_rows.a.value == 1
        assert sampler.n_rows.b.value == 10
        assert sampler.n_cols.value == 20

    def test_sample_single_point(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(1, 1)
        points = sampler.sample_points([image], iarandom.RNG(1))[0]
        assert len(points) == 1
        assert np.allclose(points[0], [10.0, 5.0])

    def test_sample_points(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(2, 2)
        points = sampler.sample_points([image], iarandom.RNG(1))[0]
        assert len(points) == 4
        assert np.allclose(points, [
            [2.5, 2.5],
            [7.5, 2.5],
            [2.5, 7.5],
            [7.5, 7.5]
        ])

    def test_sample_points_stochastic(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(1, iap.Choice([1, 2]))
        points = sampler.sample_points([image], iarandom.RNG(1))[0]

        matches_single_point = np.allclose(points, [
            [5.0, 5.0]
        ])
        matches_two_points = np.allclose(points, [
            [2.5, 5.0],
            [7.5, 5.0]
        ])

        assert len(points) in [1, 2]
        assert matches_single_point or matches_two_points

    def test_sample_points_cols_is_zero(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(iap.Deterministic(0), 1)
        points = sampler.sample_points([image], iarandom.RNG(1))[0]

        matches_single_point = np.allclose(points, [
            [5.0, 5.0]
        ])

        assert len(points) == 1
        assert matches_single_point

    def test_sample_points_rows_is_zero(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(1, iap.Deterministic(0))
        points = sampler.sample_points([image], iarandom.RNG(1))[0]

        matches_single_point = np.allclose(points, [
            [5.0, 5.0]
        ])

        assert len(points) == 1
        assert matches_single_point

    def test_sample_points_rows_is_more_than_image_height(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(2, 1)
        points = sampler.sample_points([image], iarandom.RNG(1))[0]

        matches_single_point = np.allclose(points, [
            [0.5, 0.5]
        ])

        assert len(points) == 1
        assert matches_single_point

    def test_sample_points_cols_is_more_than_image_width(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler(1, 2)
        points = sampler.sample_points([image], iarandom.RNG(1))[0]

        matches_single_point = np.allclose(points, [
            [0.5, 0.5]
        ])

        assert len(points) == 1
        assert matches_single_point

    def test_zero_sized_axes(self):
        shapes = [
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                sampler = iaa.RegularGridPointsSampler(1, 1)

                points = sampler.sample_points([image], iarandom.RNG(1))[0]

                assert len(points) == 1

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                sampler = iaa.RegularGridPointsSampler(1, 1)

                points = sampler.sample_points([image], iarandom.RNG(1))[0]

                assert len(points) == 1

    def test_determinism(self):
        image = np.zeros((500, 500, 1), dtype=np.uint8)
        sampler = iaa.RegularGridPointsSampler((1, 500), (1, 500))
        points_seed1_1 = sampler.sample_points([image], 1)[0]
        points_seed1_2 = sampler.sample_points([image], 1)[0]
        points_seed2_1 = sampler.sample_points([image], 2)[0]

        assert points_seed1_1.shape == points_seed1_2.shape
        assert points_seed1_1.shape != points_seed2_1.shape

    def test_conversion_to_string(self):
        sampler = iaa.RegularGridPointsSampler(10, (10, 30))
        expected = (
            "RegularGridPointsSampler("
            "Deterministic(int 10), "
            "DiscreteUniform(Deterministic(int 10), Deterministic(int 30))"
            ")"
        )
        assert sampler.__str__() == sampler.__repr__() == expected


class TestRelativeRegularGridPointsSampler(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___(self):
        sampler = iaa.RelativeRegularGridPointsSampler((0.1, 0.2), 0.1)
        assert isinstance(sampler.n_rows_frac, iap.Uniform)
        assert np.isclose(sampler.n_rows_frac.a.value, 0.1)
        assert np.isclose(sampler.n_rows_frac.b.value, 0.2)
        assert np.isclose(sampler.n_cols_frac.value, 0.1)

    def test_sample_single_point(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        sampler = iaa.RelativeRegularGridPointsSampler(0.001, 0.001)
        points = sampler.sample_points([image], iarandom.RNG(1))[0]
        assert len(points) == 1
        assert np.allclose(points[0], [10.0, 5.0])

    def test_sample_points(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        sampler = iaa.RelativeRegularGridPointsSampler(0.2, 0.2)
        points = sampler.sample_points([image], iarandom.RNG(1))[0]
        assert len(points) == 4
        assert np.allclose(points, [
            [2.5, 2.5],
            [7.5, 2.5],
            [2.5, 7.5],
            [7.5, 7.5]
        ])

    def test_sample_points_stochastic(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        sampler = iaa.RelativeRegularGridPointsSampler(0.1,
                                                       iap.Choice([0.1, 0.2]))
        points = sampler.sample_points([image], iarandom.RNG(1))[0]

        matches_single_point = np.allclose(points, [
            [5.0, 5.0]
        ])
        matches_two_points = np.allclose(points, [
            [2.5, 5.0],
            [7.5, 5.0]
        ])

        assert len(points) in [1, 2]
        assert matches_single_point or matches_two_points

    def test_sample_points_cols_is_zero(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        sampler = iaa.RelativeRegularGridPointsSampler(iap.Deterministic(0.001),
                                                       0.1)
        points = sampler.sample_points([image], iarandom.RNG(1))[0]

        matches_single_point = np.allclose(points, [
            [5.0, 5.0]
        ])

        assert len(points) == 1
        assert matches_single_point

    def test_sample_points_rows_is_zero(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        sampler = iaa.RelativeRegularGridPointsSampler(0.1,
                                                       iap.Deterministic(0.001))
        points = sampler.sample_points([image], iarandom.RNG(1))[0]

        matches_single_point = np.allclose(points, [
            [5.0, 5.0]
        ])

        assert len(points) == 1
        assert matches_single_point

    def test_determinism(self):
        image = np.zeros((500, 500, 1), dtype=np.uint8)
        sampler = iaa.RelativeRegularGridPointsSampler((0.01, 1.0), (0.1, 1.0))
        points_seed1_1 = sampler.sample_points([image], 1)[0]
        points_seed1_2 = sampler.sample_points([image], 1)[0]
        points_seed2_1 = sampler.sample_points([image], 2)[0]

        assert points_seed1_1.shape == points_seed1_2.shape
        assert points_seed1_1.shape != points_seed2_1.shape

    def test_zero_sized_axes(self):
        shapes = [
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                sampler = iaa.RelativeRegularGridPointsSampler(0.01, 0.01)

                points = sampler.sample_points([image], iarandom.RNG(1))[0]

                assert len(points) == 1

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                sampler = iaa.RelativeRegularGridPointsSampler(0.01, 0.01)

                points = sampler.sample_points([image], iarandom.RNG(1))[0]

                assert len(points) == 1

    def test_conversion_to_string(self):
        sampler = iaa.RelativeRegularGridPointsSampler(0.01, (0.01, 0.05))
        expected = (
            "RelativeRegularGridPointsSampler("
            "Deterministic(float 0.01000000), "
            "Uniform("
            "Deterministic(float 0.01000000), "
            "Deterministic(float 0.05000000)"
            ")"
            ")"
        )
        assert sampler.__str__() == sampler.__repr__() == expected


class _FixedPointsSampler(iaa.IPointsSampler):
    def __init__(self, points):
        self.points = np.float32(np.copy(points))
        self.last_random_state = None

    def sample_points(self, images, random_state):
        self.last_random_state = random_state
        return np.tile(self.points[np.newaxis, ...], (len(images), 1))


class TestDropoutPointsSampler(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        other = iaa.RegularGridPointsSampler(1, 1)
        sampler = iaa.DropoutPointsSampler(other, 0.5)
        assert sampler.other_points_sampler is other
        assert isinstance(sampler.p_drop, iap.Binomial)
        assert np.isclose(sampler.p_drop.p.value, 0.5)

    def test_p_drop_is_0_percent(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        points = np.linspace(0.0, 1000.0, num=100000)
        points = np.stack([points, points], axis=-1)
        other = _FixedPointsSampler(points)
        sampler = iaa.DropoutPointsSampler(other, 0.0)

        observed = sampler.sample_points([image], 1)[0]

        assert np.allclose(observed, points)

    def test_p_drop_is_100_percent(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        points = np.linspace(0.0+0.9, 1000.0-0.9, num=100000)
        points = np.stack([points, points], axis=-1)
        other = _FixedPointsSampler(points)
        sampler = iaa.DropoutPointsSampler(other, 1.0)

        observed = sampler.sample_points([image], 1)[0]

        eps = 1e-4
        assert len(observed) == 1
        assert 0.0 + 0.9 - eps <= observed[0][0] <= 1000.0 - 0.9 + eps
        assert 0.0 + 0.9 - eps <= observed[0][1] <= 1000.0 - 0.9 + eps

    def test_p_drop_is_50_percent(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        points = np.linspace(0.0+0.9, 1000.0-0.9, num=100000)
        points = np.stack([points, points], axis=-1)
        other = _FixedPointsSampler(points)
        sampler = iaa.DropoutPointsSampler(other, 0.5)

        observed = sampler.sample_points([image], 1)[0]

        assert 50000 - 1000 <= len(observed) <= 50000 + 1000

    def test_determinism(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        points = np.linspace(0.0+0.9, 1000.0-0.9, num=100000)
        points = np.stack([points, points], axis=-1)
        other = _FixedPointsSampler(points)
        sampler = iaa.DropoutPointsSampler(other, (0.3, 0.7))

        observed_s1_1 = sampler.sample_points([image], 1)[0]
        observed_s1_2 = sampler.sample_points([image], 1)[0]
        observed_s2_1 = sampler.sample_points([image], 2)[0]

        assert np.allclose(observed_s1_1, observed_s1_2)
        assert (observed_s1_1.shape != observed_s2_1.shape
                or not np.allclose(observed_s1_1, observed_s2_1))

    def test_random_state_propagates(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        points = np.linspace(0.0+0.9, 1000.0-0.9, num=1)
        points = np.stack([points, points], axis=-1)
        other = _FixedPointsSampler(points)
        sampler = iaa.DropoutPointsSampler(other, 0.5)

        _ = sampler.sample_points([image], 1)[0]
        rs_s1_1 = other.last_random_state
        _ = sampler.sample_points([image], 1)[0]
        rs_s1_2 = other.last_random_state
        _ = sampler.sample_points([image], 2)[0]
        rs_s2_1 = other.last_random_state

        assert rs_s1_1.equals(rs_s1_2)
        assert not rs_s1_1.equals(rs_s2_1)

    def test_conversion_to_string(self):
        sampler = iaa.DropoutPointsSampler(
            iaa.RegularGridPointsSampler(10, 20),
            0.2
        )
        expected = (
            "DropoutPointsSampler("
            "RegularGridPointsSampler("
            "Deterministic(int 10), "
            "Deterministic(int 20)"
            "), "
            "Binomial(Deterministic(float 0.80000000))"
            ")"
        )
        assert sampler.__str__() == sampler.__repr__() == expected


class TestUniformPointsSampler(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        sampler = iaa.UniformPointsSampler(100)
        assert isinstance(sampler.n_points, iap.Deterministic)
        assert sampler.n_points.value == 100

    def test_sampled_points_not_identical(self):
        sampler = iaa.UniformPointsSampler(3)
        images = [np.zeros((1000, 1000, 3), dtype=np.uint8)]

        points = sampler.sample_points(images, 1)[0]
        points_tpls = [tuple(point) for point in points]
        n_points = len(points)
        n_points_uq = len(set(points_tpls))

        assert n_points == 3
        assert n_points_uq == 3

    def test_sampled_points_uniformly_distributed_by_quadrants(self):
        # split image into 2x2 quadrants, group all points per quadrant,
        # assume that at least around N_points/(2*2) points are in each
        # quadrant
        sampler = iaa.UniformPointsSampler(10000)
        images = [np.zeros((1000, 3000, 1), dtype=np.uint8)]

        points = sampler.sample_points(images, 1)[0]
        points_rel = points.astype(np.float32)
        points_rel[:, 1] /= 1000
        points_rel[:, 0] /= 3000

        points_quadrants = np.clip(
            np.floor(points_rel * 2),
            0, 1
        ).astype(np.int32)
        n_points_per_quadrant = np.zeros((2, 2), dtype=np.int32)
        np.add.at(
            n_points_per_quadrant,
            (points_quadrants[:, 1], points_quadrants[:, 0]),
            1)

        assert np.all(n_points_per_quadrant > 0.8*(10000/4))

    def test_sampled_points_uniformly_distributed_by_distance_from_origin(self):
        # Sample N points, compute distances from origin each axis,
        # split into B bins, assume that each bin contains at least around
        # N/B points.
        sampler = iaa.UniformPointsSampler(10000)
        images = [np.zeros((1000, 3000, 1), dtype=np.uint8)]

        points = sampler.sample_points(images, 1)[0]
        points_rel = points.astype(np.float32)
        points_rel[:, 1] /= 1000
        points_rel[:, 0] /= 3000

        points_bins = np.clip(
            np.floor(points_rel * 10),
            0, 1
        ).astype(np.int32)

        # Don't use euclidean (2d) distance here, but instead axis-wise (1d)
        # distance. The euclidean distance leads to non-uniform density of
        # distances, because points on the same "circle" have the same
        # distance, and there are less points close/far away from the origin
        # that fall on the same circle.
        points_bincounts_x = np.bincount(points_bins[:, 0])
        points_bincounts_y = np.bincount(points_bins[:, 1])

        assert np.all(points_bincounts_x > 0.8*(10000/10))
        assert np.all(points_bincounts_y > 0.8*(10000/10))

    def test_many_images(self):
        sampler = iaa.UniformPointsSampler(1000)
        images = [
            np.zeros((100, 500, 3), dtype=np.uint8),
            np.zeros((500, 100, 1), dtype=np.uint8)
        ]

        points = sampler.sample_points(images, 1)

        assert len(points) == 2
        assert len(points[0]) == 1000
        assert len(points[1]) == 1000
        assert not np.allclose(points[0], points[1])
        assert np.any(points[0][:, 1] < 20)
        assert np.any(points[0][:, 1] > 0.9*100)
        assert np.any(points[0][:, 0] < 20)
        assert np.any(points[0][:, 0] > 0.9*500)
        assert np.any(points[1][:, 1] < 20)
        assert np.any(points[1][:, 1] > 0.9*500)
        assert np.any(points[1][:, 0] < 20)
        assert np.any(points[1][:, 0] > 0.9*100)

    def test_always_at_least_one_point(self):
        sampler = iaa.UniformPointsSampler(iap.Deterministic(0))
        images = [np.zeros((10, 10, 1), dtype=np.uint8)]

        points = sampler.sample_points(images, 1)[0]

        assert len(points) == 1

    def test_n_points_can_vary_between_calls(self):
        sampler = iaa.UniformPointsSampler(iap.Choice([1, 10]))
        images = [np.zeros((10, 10, 1), dtype=np.uint8)]

        seen = {1: False, 10: False}
        for i in sm.xrange(50):
            points = sampler.sample_points(images, i)[0]
            seen[len(points)] = True
            if np.all(seen.values()):
                break

        assert len(list(seen.keys())) == 2
        assert np.all(seen.values())

    def test_n_points_can_vary_between_images(self):
        sampler = iaa.UniformPointsSampler(iap.Choice([1, 10]))
        images = [
            np.zeros((10, 10, 1), dtype=np.uint8)
            for _ in sm.xrange(50)]

        points = sampler.sample_points(images, 1)
        point_counts = set([len(points_i) for points_i in points])

        assert len(points) == 50
        assert len(list(point_counts)) == 2
        assert 1 in point_counts
        assert 10 in point_counts

    def test_determinism(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        sampler = iaa.UniformPointsSampler(100)

        observed_s1_1 = sampler.sample_points([image], 1)[0]
        observed_s1_2 = sampler.sample_points([image], 1)[0]
        observed_s2_1 = sampler.sample_points([image], 2)[0]

        assert np.allclose(observed_s1_1, observed_s1_2)
        assert not np.allclose(observed_s1_1, observed_s2_1)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                sampler = iaa.UniformPointsSampler(1)

                points = sampler.sample_points([image], iarandom.RNG(1))[0]

                # TODO this is not the same as for
                #      (Relative)RegularGridPointsSampler, which returns in
                #      this case 0 points
                assert len(points) == 1

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                sampler = iaa.UniformPointsSampler(1)

                points = sampler.sample_points([image], iarandom.RNG(1))[0]

                assert len(points) == 1

    def test_conversion_to_string(self):
        sampler = iaa.UniformPointsSampler(10)
        expected = "UniformPointsSampler(Deterministic(int 10))"
        assert sampler.__str__() == sampler.__repr__() == expected


class TestSubsamplingPointsSampler(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        other = iaa.RegularGridPointsSampler(1, 1)
        sampler = iaa.SubsamplingPointsSampler(other, 100)
        assert sampler.other_points_sampler is other
        assert sampler.n_points_max == 100

    def test_max_is_zero(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        other = iaa.RegularGridPointsSampler(2, 2)

        with warnings.catch_warnings(record=True) as caught_warnings:
            sampler = iaa.SubsamplingPointsSampler(other, 0)

        observed = sampler.sample_points([image], 1)[0]

        assert len(observed) == 0
        assert len(caught_warnings) == 1
        assert "n_points_max=0" in str(caught_warnings[-1].message)

    def test_max_is_above_point_count(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        other = iaa.RegularGridPointsSampler(2, 2)
        sampler = iaa.SubsamplingPointsSampler(other, 100)

        observed = sampler.sample_points([image], 1)[0]

        assert len(observed) == 4
        assert np.allclose(observed, [
            [2.5, 2.5],
            [7.5, 2.5],
            [2.5, 7.5],
            [7.5, 7.5]
        ])

    def test_max_is_below_point_count(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        other = iaa.RegularGridPointsSampler(5, 5)
        sampler = iaa.SubsamplingPointsSampler(other, 1000)

        observed = sampler.sample_points([image], 1)[0]

        assert len(observed) == 5*5

    def test_max_is_sometimes_below_point_count(self):
        image = np.zeros((1, 10, 3), dtype=np.uint8)
        other = iaa.RegularGridPointsSampler(1, (9, 11))
        sampler = iaa.SubsamplingPointsSampler(other, 1000)

        observed = sampler.sample_points([image] * 100, 1)
        counts = [len(observed_i) for observed_i in observed]
        counts_uq = set(counts)

        assert 9 in counts_uq
        assert 10 in counts_uq
        assert 11 not in counts_uq

    def test_random_state_propagates(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        points = np.linspace(0.0+0.9, 1000.0-0.9, num=1)
        points = np.stack([points, points], axis=-1)
        other = _FixedPointsSampler(points)
        sampler = iaa.SubsamplingPointsSampler(other, 100)

        _ = sampler.sample_points([image], 1)[0]
        rs_s1_1 = other.last_random_state
        _ = sampler.sample_points([image], 1)[0]
        rs_s1_2 = other.last_random_state
        _ = sampler.sample_points([image], 2)[0]
        rs_s2_1 = other.last_random_state

        assert rs_s1_1.equals(rs_s1_2)
        assert not rs_s1_1.equals(rs_s2_1)

    def test_conversion_to_string(self):
        sampler = iaa.SubsamplingPointsSampler(
            iaa.RegularGridPointsSampler(10, 20),
            10
        )
        expected = (
            "SubsamplingPointsSampler("
            "RegularGridPointsSampler("
            "Deterministic(int 10), "
            "Deterministic(int 20)"
            "), "
            "10"
            ")"
        )
        assert sampler.__str__() == sampler.__repr__() == expected
