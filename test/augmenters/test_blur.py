from __future__ import print_function, division, absolute_import

import warnings
import sys
import itertools
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

import numpy as np
import six.moves as sm
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug import random as iarandom
from imgaug.testutils import keypoints_equal, reseed, runtest_pickleable_uint8_img


class Test_blur_gaussian_(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_integration(self):
        backends = ["auto", "scipy", "cv2"]
        nb_channels_lst = [None, 1, 3, 4, 5, 10]

        gen = itertools.product(backends, nb_channels_lst)
        for backend, nb_channels in gen:
            with self.subTest(backend=backend, nb_channels=nb_channels):
                image = np.zeros((5, 5), dtype=np.uint8)
                if nb_channels is not None:
                    image = np.tile(image[..., np.newaxis], (1, 1, nb_channels))

                image[2, 2] = 255
                mask = image < 255
                observed = iaa.blur_gaussian_(
                    np.copy(image), sigma=5.0, backend=backend)
                assert observed.shape == image.shape
                assert observed.dtype.name == "uint8"
                assert np.all(observed[2, 2] < 255)
                assert np.sum(observed[mask]) > (5*5-1)
                if nb_channels is not None and nb_channels > 1:
                    for c in sm.xrange(1, observed.shape[2]):
                        assert np.array_equal(observed[..., c],
                                              observed[..., 0])

    def test_sigma_zero(self):
        image = np.arange(4*4).astype(np.uint8).reshape((4, 4))
        observed = iaa.blur_gaussian_(np.copy(image), 0)
        assert np.array_equal(observed, image)

        image = np.arange(4*4).astype(np.uint8).reshape((4, 4, 1))
        observed = iaa.blur_gaussian_(np.copy(image), 0)
        assert np.array_equal(observed, image)

        image = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))
        observed = iaa.blur_gaussian_(np.copy(image), 0)
        assert np.array_equal(observed, image)

    def test_eps(self):
        image = np.arange(4*4).astype(np.uint8).reshape((4, 4))
        observed_no_eps = iaa.blur_gaussian_(np.copy(image), 1.0, eps=0)
        observed_with_eps = iaa.blur_gaussian_(np.copy(image), 1.0, eps=1e10)
        assert not np.array_equal(observed_no_eps, observed_with_eps)
        assert np.array_equal(observed_with_eps, image)

    def test_ksize(self):
        def side_effect(image, ksize, sigmaX, sigmaY, borderType):
            return image + 1

        sigmas = [5.0, 5.0]
        ksizes = [None, 3]
        ksizes_expected = [2.6*5.0, 3]
        gen = zip(sigmas, ksizes, ksizes_expected)

        for (sigma, ksize, ksize_expected) in gen:
            with self.subTest(sigma=sigma, ksize=ksize):
                mock_GaussianBlur = mock.Mock(side_effect=side_effect)
                image = np.arange(4*4).astype(np.uint8).reshape((4, 4))
                with mock.patch('cv2.GaussianBlur', mock_GaussianBlur):
                    observed = iaa.blur_gaussian_(
                        np.copy(image),
                        sigma=sigma,
                        ksize=ksize,
                        backend="cv2")
                    assert np.array_equal(observed, image+1)

                cargs = mock_GaussianBlur.call_args
                assert mock_GaussianBlur.call_count == 1
                assert np.array_equal(cargs[0][0], image)
                assert isinstance(cargs[0][1], tuple)
                assert np.allclose(
                    np.float32(cargs[0][1]),
                    np.float32([ksize_expected, ksize_expected]))
                assert np.isclose(cargs[1]["sigmaX"], sigma)
                assert np.isclose(cargs[1]["sigmaY"], sigma)
                assert cargs[1]["borderType"] == cv2.BORDER_REFLECT_101

    def test_more_than_four_channels(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.blur_gaussian_(np.copy(image), 1.0)

                assert image_aug.shape == image.shape

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
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.blur_gaussian_(np.copy(image), 1.0)

                assert image_aug.shape == image.shape

    def test_backends_called(self):
        def side_effect_cv2(image, ksize, sigmaX, sigmaY, borderType):
            return image + 1

        def side_effect_scipy(image, sigma, mode):
            return image + 1

        mock_GaussianBlur = mock.Mock(side_effect=side_effect_cv2)
        mock_gaussian_filter = mock.Mock(side_effect=side_effect_scipy)
        image = np.arange(4*4).astype(np.uint8).reshape((4, 4))
        with mock.patch('cv2.GaussianBlur', mock_GaussianBlur):
            _observed = iaa.blur_gaussian_(
                np.copy(image), sigma=1.0, eps=0, backend="cv2")
        assert mock_GaussianBlur.call_count == 1

        with mock.patch('scipy.ndimage.gaussian_filter', mock_gaussian_filter):
            _observed = iaa.blur_gaussian_(
                np.copy(image), sigma=1.0, eps=0, backend="scipy")
        assert mock_gaussian_filter.call_count == 1

    def test_backends_similar(self):
        with self.subTest(nb_channels=None):
            size = 10
            image = np.arange(
                0, size*size).astype(np.uint8).reshape((size, size))
            image_cv2 = iaa.blur_gaussian_(
                np.copy(image), sigma=3.0, ksize=20, backend="cv2")
            image_scipy = iaa.blur_gaussian_(
                np.copy(image), sigma=3.0, backend="scipy")
            diff = np.abs(image_cv2.astype(np.int32)
                          - image_scipy.astype(np.int32))
            assert np.average(diff) < 0.05 * (size * size)

        with self.subTest(nb_channels=3):
            size = 10
            image = np.arange(
                0, size*size).astype(np.uint8).reshape((size, size))
            image = np.tile(image[..., np.newaxis], (1, 1, 3))
            image[1] += 1
            image[2] += 2
            image_cv2 = iaa.blur_gaussian_(
                np.copy(image), sigma=3.0, ksize=20, backend="cv2")
            image_scipy = iaa.blur_gaussian_(
                np.copy(image), sigma=3.0, backend="scipy")
            diff = np.abs(image_cv2.astype(np.int32)
                          - image_scipy.astype(np.int32))
            assert np.average(diff) < 0.05 * (size * size)
            for c in sm.xrange(3):
                diff = np.abs(image_cv2[..., c].astype(np.int32)
                              - image_scipy[..., c].astype(np.int32))
                assert np.average(diff) < 0.05 * (size * size)

    def test_view(self):
        for backend in ["auto", "scipy", "cv2"]:
            image = np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 255, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1]
            ], dtype=np.uint8)
            image_cp = np.copy(image[0:5, :])

            image_aug = iaa.blur_gaussian_(image[0:5, :], 3.0, backend=backend)

            assert image_aug.shape == (5, 5)
            assert image_aug.dtype.name == "uint8"
            assert np.all(image_aug[image_cp == 0] > 0)
            assert np.all(image_aug[image_cp == 255] < 255)

    def test_non_contiguous(self):
        for backend in ["auto", "scipy", "cv2"]:
            image = np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 255, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=np.uint8, order="F")
            image_cp = np.copy(image)

            image_aug = iaa.blur_gaussian_(image, 3.0, backend=backend)

            assert image_aug.shape == (5, 5)
            assert image_aug.dtype.name == "uint8"
            assert np.all(image_aug[image_cp == 0] > 0)
            assert np.all(image_aug[image_cp == 255] < 255)

    def test_warnings(self):
        # note that self.assertWarningRegex does not exist in python 2.7
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            _ = iaa.blur_gaussian_(
                np.zeros((1, 1), dtype=np.uint32),
                sigma=3.0,
                ksize=11,
                backend="scipy")

            assert len(caught_warnings) == 1
            assert (
                "but also provided 'ksize' argument"
                in str(caught_warnings[-1].message))

    def test_other_dtypes_sigma_0(self):
        try:
            f128 = [np.dtype("float128").name]
        except TypeError:
            f128 = []

        dtypes_to_test_list = [
            ["bool",
             "uint8", "uint16", "uint32", "uint64",
             "int8", "int16", "int32", "int64",
             "float16", "float32", "float64"] + f128,
            ["bool",
             "uint8", "uint16", "uint32", "uint64",
             "int8", "int16", "int32", "int64",
             "float16", "float32", "float64"] + f128
        ]
        gen = zip(["scipy", "cv2"], dtypes_to_test_list)

        for backend, dtypes_to_test in gen:
            # bool
            if "bool" in dtypes_to_test:
                with self.subTest(backend=backend, dtype="bool"):
                    image = np.zeros((3, 3), dtype=bool)
                    image[1, 1] = True
                    image_aug = iaa.blur_gaussian_(
                        np.copy(image), sigma=0, backend=backend)
                    assert image_aug.dtype.name == "bool"
                    assert np.all(image_aug == image)

            # uint, int
            uint_dts = [np.uint8, np.uint16, np.uint32, np.uint64]
            int_dts = [np.int8, np.int16, np.int32, np.int64]
            for dtype in uint_dts + int_dts:
                dtype = np.dtype(dtype)
                if dtype.name in dtypes_to_test:
                    with self.subTest(backend=backend, dtype=dtype.name):
                        _min_value, center_value, _max_value = \
                            iadt.get_value_range_of_dtype(dtype)
                        image = np.zeros((3, 3), dtype=dtype)
                        image[1, 1] = int(center_value)
                        image_aug = iaa.blur_gaussian_(
                            np.copy(image), sigma=0, backend=backend)
                        assert image_aug.dtype.name == dtype.name
                        assert np.all(image_aug == image)

            # float
            float_dts = [np.float16, np.float32, np.float64] + f128
            for dtype in float_dts:
                dtype = np.dtype(dtype)
                if dtype.name in dtypes_to_test:
                    with self.subTest(backend=backend, dtype=dtype.name):
                        _min_value, center_value, _max_value = \
                            iadt.get_value_range_of_dtype(dtype)
                        image = np.zeros((3, 3), dtype=dtype)
                        image[1, 1] = center_value
                        image_aug = iaa.blur_gaussian_(
                            np.copy(image), sigma=0, backend=backend)
                        assert image_aug.dtype.name == dtype.name
                        assert np.allclose(image_aug, image)

    def test_other_dtypes_sigma_075(self):
        # prototype kernel, generated via:
        # mask = np.zeros((5, 5), dtype=np.int32)
        # mask[2, 2] = 1000 * 1000
        # kernel = ndimage.gaussian_filter(mask, 0.75)
        mask = np.float64([
           [   923,   6650,  16163,   6650,    923],
           [  6650,  47896, 116408,  47896,   6650],
           [ 16163, 116408, 282925, 116408,  16163],
           [  6650,  47896, 116408,  47896,   6650],
           [   923,   6650,  16163,   6650,    923]
        ]) / (1000.0 * 1000.0)

        dtypes_to_test_list = [
            # scipy
            ["bool",
             "uint8", "uint16", "uint32", "uint64",
             "int8", "int16", "int32", "int64",
             "float16", "float32", "float64"],
            # cv2
            ["bool",
             "uint8", "uint16",
             "int8", "int16", "int32",
             "float16", "float32", "float64"]
        ]
        gen = zip(["scipy", "cv2"], dtypes_to_test_list)

        for backend, dtypes_to_test in gen:
            # bool
            if "bool" in dtypes_to_test:
                with self.subTest(backend=backend, dtype="bool"):
                    image = np.zeros((5, 5), dtype=bool)
                    image[2, 2] = True
                    image_aug = iaa.blur_gaussian_(
                        np.copy(image), sigma=0.75, backend=backend)
                    assert image_aug.dtype.name == "bool"
                    assert np.all(image_aug == (mask > 0.5))

            # uint, int
            uint_dts = [np.uint8, np.uint16, np.uint32, np.uint64]
            int_dts = [np.int8, np.int16, np.int32, np.int64]
            for dtype in uint_dts + int_dts:
                dtype = np.dtype(dtype)
                if dtype.name in dtypes_to_test:
                    with self.subTest(backend=backend, dtype=dtype.name):
                        min_value, center_value, max_value = \
                            iadt.get_value_range_of_dtype(dtype)
                        dynamic_range = max_value - min_value

                        value = int(center_value + 0.4 * max_value)
                        image = np.zeros((5, 5), dtype=dtype)
                        image[2, 2] = value
                        image_aug = iaa.blur_gaussian_(
                            image, sigma=0.75, backend=backend)
                        expected = (mask * value).astype(dtype)
                        diff = np.abs(image_aug.astype(np.int64)
                                      - expected.astype(np.int64))
                        assert image_aug.shape == mask.shape
                        assert image_aug.dtype.type == dtype
                        if dtype.itemsize <= 1:
                            assert np.max(diff) <= 4
                        else:
                            assert np.max(diff) <= 0.01 * dynamic_range

            # float
            float_dts = [np.float16, np.float32, np.float64]
            values = [5000, 1000**1, 1000**2, 1000**3]
            for dtype, value in zip(float_dts, values):
                dtype = np.dtype(dtype)
                if dtype.name in dtypes_to_test:
                    with self.subTest(backend=backend, dtype=dtype.name):
                        image = np.zeros((5, 5), dtype=dtype)
                        image[2, 2] = value
                        image_aug = iaa.blur_gaussian_(
                            image, sigma=0.75, backend=backend)
                        expected = (mask * value).astype(dtype)
                        diff = np.abs(image_aug.astype(np.float64)
                                      - expected.astype(np.float64))
                        assert image_aug.shape == mask.shape
                        assert image_aug.dtype.type == dtype
                        # accepts difference of 2.0, 4.0, 8.0, 16.0 (at 1,
                        # 2, 4, 8 bytes, i.e. 8, 16, 32, 64 bit)
                        max_diff = (
                            np.dtype(dtype).itemsize
                            * 0.01
                            * np.float64(value)
                        )
                        assert np.max(diff) < max_diff

    def test_other_dtypes_bool_at_sigma_06(self):
        # --
        # blur of bool input at sigma=0.6
        # --
        # here we use a special mask and sigma as otherwise the only values
        # ending up with >0.5 would be the ones that
        # were before the blur already at >0.5
        # prototype kernel, generated via:
        #  mask = np.zeros((5, 5), dtype=np.float64)
        #  mask[1, 0] = 255
        #  mask[2, 0] = 255
        #  mask[2, 2] = 255
        #  mask[2, 4] = 255
        #  mask[3, 0] = 255
        #  mask = ndimage.gaussian_filter(mask, 1.0, mode="mirror")
        mask_bool = np.float64([
           [ 57,  14,   2,   1,   1],
           [142,  42,  29,  14,  28],
           [169,  69, 114,  56, 114],
           [142,  42,  29,  14,  28],
           [ 57,  14,   2,   1,   1]
        ]) / 255.0

        image = np.zeros((5, 5), dtype=bool)
        image[1, 0] = True
        image[2, 0] = True
        image[2, 2] = True
        image[2, 4] = True
        image[3, 0] = True

        for backend in ["scipy", "cv2"]:
            image_aug = iaa.blur_gaussian_(
                np.copy(image), sigma=0.6, backend=backend)
            expected = mask_bool > 0.5
            assert image_aug.shape == mask_bool.shape
            assert image_aug.dtype.type == np.bool_
            assert np.all(image_aug == expected)


class Test_blur_avg_(unittest.TestCase):
    @classmethod
    def _avg(cls, values):
        return int(np.round(np.average(values)))

    def test_kernel_size_is_int(self):
        # reflection padded:
        # [6, 5, 6, 7, 8, 7],
        # [2, 1, 2, 3, 4, 3],
        # [6, 5, 6, 7, 8, 7],
        # [10, 9, 10, 11, 12, 11],
        # [14, 13, 14, 15, 16, 15]
        # [10, 9, 10, 11, 12, 11],
        image = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.uint8)

        image_aug = iaa.blur_avg_(np.copy(image), 3)

        assert image_aug[0, 0] == self._avg([6, 5, 6, 2, 1, 2, 6, 5, 6])
        assert image_aug[0, 1] == self._avg([5, 6, 7, 1, 2, 3, 5, 6, 7])
        assert image_aug[3, 3] == self._avg([11, 12, 11, 15, 16, 15, 11, 12,
                                             11])

    def test_kernel_size_is_tuple(self):
        # reflection padded:
        # [6, 5, 6, 7, 8, 7],
        # [2, 1, 2, 3, 4, 3],
        # [6, 5, 6, 7, 8, 7],
        # [10, 9, 10, 11, 12, 11],
        # [14, 13, 14, 15, 16, 15]
        # [10, 9, 10, 11, 12, 11],
        image = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.uint8)

        image_aug = iaa.blur_avg_(np.copy(image), (3, 1))

        assert image_aug[0, 0] == self._avg([5, 1, 5])
        assert image_aug[0, 1] == self._avg([6, 2, 6])
        assert image_aug[3, 3] == self._avg([12, 16, 12])

    def test_view(self):
        # reflection padded (after crop):
        # [6, 5, 6, 7, 8, 7],
        # [2, 1, 2, 3, 4, 3],
        # [6, 5, 6, 7, 8, 7],
        # [10, 9, 10, 11, 12, 11],
        # [14, 13, 14, 15, 16, 15]
        # [10, 9, 10, 11, 12, 11],
        image = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [0, 0, 0, 0]
        ], dtype=np.uint8)

        image_aug = iaa.blur_avg_(np.copy(image)[0:4, :], 3)

        assert image_aug[0, 0] == self._avg([6, 5, 6, 2, 1, 2, 6, 5, 6])
        assert image_aug[0, 1] == self._avg([5, 6, 7, 1, 2, 3, 5, 6, 7])
        assert image_aug[3, 3] == self._avg([11, 12, 11, 15, 16, 15, 11, 12,
                                             11])

    def test_noncontiguous(self):
        # reflection padded:
        # [6, 5, 6, 7, 8, 7],
        # [2, 1, 2, 3, 4, 3],
        # [6, 5, 6, 7, 8, 7],
        # [10, 9, 10, 11, 12, 11],
        # [14, 13, 14, 15, 16, 15]
        # [10, 9, 10, 11, 12, 11],
        image = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.uint8, order="F")

        image_aug = iaa.blur_avg_(image, 3)

        assert image_aug[0, 0] == self._avg([6, 5, 6, 2, 1, 2, 6, 5, 6])
        assert image_aug[0, 1] == self._avg([5, 6, 7, 1, 2, 3, 5, 6, 7])
        assert image_aug[3, 3] == self._avg([11, 12, 11, 15, 16, 15, 11, 12,
                                             11])


class Test_blur_mean_shift_(unittest.TestCase):
    @property
    def image(self):
        image = [
            [1, 2, 3, 4, 200, 201, 202, 203],
            [1, 2, 3, 4, 200, 201, 202, 203],
            [1, 2, 3, 4, 200, 201, 202, 203],
            [1, 2, 3, 4, 200, 201, 202, 203]
        ]
        image = np.array(image, dtype=np.uint8).reshape((4, 2*4, 1))
        image = np.tile(image, (1, 1, 3))
        return image

    def test_simple_image(self):
        image = self.image

        image_blurred = iaa.blur_mean_shift_(np.copy(image), 0.5, 0.5)

        assert image_blurred.shape == image.shape
        assert image_blurred.dtype.name == "uint8"
        assert not np.array_equal(image_blurred, image)
        assert 0 <= np.average(image[:, 0:4, :]) <= 5
        assert 199 <= np.average(image[:, 4:, :]) <= 203

    def test_hw_image(self):
        image = self.image[:, :, 0]

        image_blurred = iaa.blur_mean_shift_(np.copy(image), 0.5, 0.5)

        assert image_blurred.shape == image.shape
        assert image_blurred.dtype.name == "uint8"
        assert not np.array_equal(image_blurred, image)

    def test_hw1_image(self):
        image = self.image[:, :, 0:1]

        image_blurred = iaa.blur_mean_shift_(np.copy(image), 0.5, 0.5)

        assert image_blurred.ndim == 3
        assert image_blurred.shape == image.shape
        assert image_blurred.dtype.name == "uint8"
        assert not np.array_equal(image_blurred, image)

    def test_non_contiguous_image(self):
        image = self.image
        image_cp = np.copy(np.fliplr(image))
        image = np.fliplr(image)
        assert image.flags["C_CONTIGUOUS"] is False

        image_blurred = iaa.blur_mean_shift_(image, 0.5, 0.5)

        assert image_blurred.shape == image_cp.shape
        assert image_blurred.dtype.name == "uint8"
        assert not np.array_equal(image_blurred, image_cp)

    def test_both_parameters_are_zero(self):
        image = self.image[:, :, 0]

        image_blurred = iaa.blur_mean_shift_(np.copy(image), 0, 0)

        assert image_blurred.shape == image.shape
        assert image_blurred.dtype.name == "uint8"
        assert not np.array_equal(image_blurred, image)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.blur_mean_shift_(np.copy(image), 1.0, 1.0)

                assert image_aug.shape == image.shape


class TestGaussianBlur(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_sigma_is_zero(self):
        # no blur, shouldnt change anything
        base_img = np.array([[0, 0, 0],
                         [0, 255, 0],
                         [0, 0, 0]], dtype=np.uint8)
        base_img = base_img[:, :, np.newaxis]
        images = np.array([base_img])

        aug = iaa.GaussianBlur(sigma=0)

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

    def test_low_sigma(self):
        base_img = np.array([[0, 0, 0],
                         [0, 255, 0],
                         [0, 0, 0]], dtype=np.uint8)
        base_img = base_img[:, :, np.newaxis]

        images = np.array([base_img])
        images_list = [base_img]
        outer_pixels = ([], [])
        for i in sm.xrange(base_img.shape[0]):
            for j in sm.xrange(base_img.shape[1]):
                if i != j:
                    outer_pixels[0].append(i)
                    outer_pixels[1].append(j)

        # weak blur of center pixel
        aug = iaa.GaussianBlur(sigma=0.5)
        aug_det = aug.to_deterministic()

        # images as numpy array
        observed = aug.augment_images(images)
        assert 100 < observed[0][1, 1] < 255
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

        observed = aug_det.augment_images(images)
        assert 100 < observed[0][1, 1] < 255
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

        # images as list
        observed = aug.augment_images(images_list)
        assert 100 < observed[0][1, 1] < 255
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

        observed = aug_det.augment_images(images_list)
        assert 100 < observed[0][1, 1] < 255
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    def test_keypoints_dont_change(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        kpsoi = [ia.KeypointsOnImage(kps, shape=(3, 3, 1))]

        aug = iaa.GaussianBlur(sigma=0.5)
        aug_det = aug.to_deterministic()

        observed = aug.augment_keypoints(kpsoi)
        expected = kpsoi
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(kpsoi)
        expected = kpsoi
        assert keypoints_equal(observed, expected)

    def test_sigma_is_tuple(self):
        # varying blur sigmas
        base_img = np.array([[0, 0, 0],
                         [0, 255, 0],
                         [0, 0, 0]], dtype=np.uint8)
        base_img = base_img[:, :, np.newaxis]
        images = np.array([base_img])

        aug = iaa.GaussianBlur(sigma=(0, 1))
        aug_det = aug.to_deterministic()

        last_aug = None
        last_aug_det = None
        nb_changed_aug = 0
        nb_changed_aug_det = 0
        nb_iterations = 1000
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(images)
            observed_aug_det = aug_det.augment_images(images)
            if i == 0:
                last_aug = observed_aug
                last_aug_det = observed_aug_det
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                if not np.array_equal(observed_aug_det, last_aug_det):
                    nb_changed_aug_det += 1
                last_aug = observed_aug
                last_aug_det = observed_aug_det
        assert nb_changed_aug >= int(nb_iterations * 0.8)
        assert nb_changed_aug_det == 0

    def test_other_dtypes_bool_at_sigma_0(self):
        # bool
        aug = iaa.GaussianBlur(sigma=0)

        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == image)

    def test_other_dtypes_uint_int_at_sigma_0(self):
        aug = iaa.GaussianBlur(sigma=0)
        dts = [np.uint8, np.uint16, np.uint32,
               np.int8, np.int16, np.int32]

        for dtype in dts:
            _min_value, center_value, _max_value = \
                iadt.get_value_range_of_dtype(dtype)
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = int(center_value)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == image)

    def test_other_dtypes_float_at_sigma_0(self):
        aug = iaa.GaussianBlur(sigma=0)
        dts = [np.float16, np.float32, np.float64]

        for dtype in dts:
            _min_value, center_value, _max_value = \
                iadt.get_value_range_of_dtype(dtype)
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = center_value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.allclose(image_aug, image)

    def test_other_dtypes_bool_at_sigma_060(self):
        # --
        # blur of bool input at sigma=0.6
        # --
        # here we use a special mask and sigma as otherwise the only values
        # ending up with >0.5 would be the ones that
        # were before the blur already at >0.5
        # prototype kernel, generated via:
        #  mask = np.zeros((5, 5), dtype=np.float64)
        #  mask[1, 0] = 255
        #  mask[2, 0] = 255
        #  mask[2, 2] = 255
        #  mask[2, 4] = 255
        #  mask[3, 0] = 255
        #  mask = ndimage.gaussian_filter(mask, 1.0, mode="mirror")
        aug = iaa.GaussianBlur(sigma=0.6)

        mask_bool = np.float64([
           [ 57,  14,   2,   1,   1],
           [142,  42,  29,  14,  28],
           [169,  69, 114,  56, 114],
           [142,  42,  29,  14,  28],
           [ 57,  14,   2,   1,   1]
        ]) / 255.0

        image = np.zeros((5, 5), dtype=bool)
        image[1, 0] = True
        image[2, 0] = True
        image[2, 2] = True
        image[2, 4] = True
        image[3, 0] = True
        image_aug = aug.augment_image(image)
        expected = mask_bool > 0.5
        assert image_aug.shape == mask_bool.shape
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == expected)

    def test_other_dtypes_at_sigma_1(self):
        # --
        # blur of various dtypes at sigma=1.0
        # and using an example value of 100 for int/uint/float and True for
        # bool
        # --
        # prototype kernel, generated via:
        #  mask = np.zeros((5, 5), dtype=np.float64)
        #  mask[2, 2] = 100
        #  mask = ndimage.gaussian_filter(mask, 1.0, mode="mirror")
        aug = iaa.GaussianBlur(sigma=1.0)

        mask = np.float64([
            [1, 2, 3, 2, 1],
            [2, 5, 9, 5, 2],
            [4, 9, 15, 9, 4],
            [2, 5, 9, 5, 2],
            [1, 2, 3, 2, 1]
        ])

        # uint, int
        uint_dts = [np.uint8, np.uint16, np.uint32]
        int_dts = [np.int8, np.int16, np.int32]
        for dtype in uint_dts + int_dts:
            image = np.zeros((5, 5), dtype=dtype)
            image[2, 2] = 100
            image_aug = aug.augment_image(image)
            expected = mask.astype(dtype)
            diff = np.abs(image_aug.astype(np.int64)
                          - expected.astype(np.int64))
            assert image_aug.shape == mask.shape
            assert image_aug.dtype.type == dtype
            assert np.max(diff) <= 4
            assert np.average(diff) <= 2

        # float
        float_dts = [np.float16, np.float32, np.float64]
        for dtype in float_dts:
            image = np.zeros((5, 5), dtype=dtype)
            image[2, 2] = 100.0
            image_aug = aug.augment_image(image)
            expected = mask.astype(dtype)
            diff = np.abs(image_aug.astype(np.float64)
                          - expected.astype(np.float64))
            assert image_aug.shape == mask.shape
            assert image_aug.dtype.type == dtype
            assert np.max(diff) < 4
            assert np.average(diff) < 2.0

    def test_other_dtypes_at_sigma_040(self):
        # --
        # blur of various dtypes at sigma=0.4
        # and using an example value of 100 for int/uint/float and True for
        # bool
        # --
        aug = iaa.GaussianBlur(sigma=0.4)

        # prototype kernel, generated via:
        #  mask = np.zeros((5, 5), dtype=np.uint8)
        #  mask[2, 2] = 100
        #  kernel = ndimage.gaussian_filter(mask, 0.4, mode="mirror")
        mask = np.float64([
            [0,  0,  0,  0,  0],
            [0,  0,  3,  0,  0],
            [0,  3, 83,  3,  0],
            [0,  0,  3,  0,  0],
            [0,  0,  0,  0,  0]
        ])

        # uint, int
        uint_dts = [np.uint8, np.uint16, np.uint32]
        int_dts = [np.int8, np.int16, np.int32]
        for dtype in uint_dts + int_dts:
            image = np.zeros((5, 5), dtype=dtype)
            image[2, 2] = 100
            image_aug = aug.augment_image(image)
            expected = mask.astype(dtype)
            diff = np.abs(image_aug.astype(np.int64)
                          - expected.astype(np.int64))
            assert image_aug.shape == mask.shape
            assert image_aug.dtype.type == dtype
            assert np.max(diff) <= 4

        # float
        float_dts = [np.float16, np.float32, np.float64]
        for dtype in float_dts:
            image = np.zeros((5, 5), dtype=dtype)
            image[2, 2] = 100.0
            image_aug = aug.augment_image(image)
            expected = mask.astype(dtype)
            diff = np.abs(image_aug.astype(np.float64)
                          - expected.astype(np.float64))
            assert image_aug.shape == mask.shape
            assert image_aug.dtype.type == dtype
            assert np.max(diff) < 4.0

    def test_other_dtypes_at_sigma_075(self):
        # --
        # blur of various dtypes at sigma=0.75
        # and values being half-way between center and maximum for each dtype
        # The goal of this test is to verify that no major loss of resolution
        # happens for large dtypes.
        # Such inaccuracies appear for float64 if used.
        # --
        aug = iaa.GaussianBlur(sigma=0.75)

        # prototype kernel, generated via:
        # mask = np.zeros((5, 5), dtype=np.int32)
        # mask[2, 2] = 1000 * 1000
        # kernel = ndimage.gaussian_filter(mask, 0.75)
        mask = np.float64([
           [   923,   6650,  16163,   6650,    923],
           [  6650,  47896, 116408,  47896,   6650],
           [ 16163, 116408, 282925, 116408,  16163],
           [  6650,  47896, 116408,  47896,   6650],
           [   923,   6650,  16163,   6650,    923]
        ]) / (1000.0 * 1000.0)

        # uint, int
        uint_dts = [np.uint8, np.uint16, np.uint32]
        int_dts = [np.int8, np.int16, np.int32]
        for dtype in uint_dts + int_dts:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)
            dynamic_range = max_value - min_value

            value = int(center_value + 0.4 * max_value)
            image = np.zeros((5, 5), dtype=dtype)
            image[2, 2] = value
            image_aug = aug.augment_image(image)
            expected = (mask * value).astype(dtype)
            diff = np.abs(image_aug.astype(np.int64)
                          - expected.astype(np.int64))
            assert image_aug.shape == mask.shape
            assert image_aug.dtype.type == dtype
            if np.dtype(dtype).itemsize <= 1:
                assert np.max(diff) <= 4
            else:
                assert np.max(diff) <= 0.01 * dynamic_range

        # float
        float_dts = [np.float16, np.float32, np.float64]
        values = [5000, 1000*1000, 1000*1000*1000]
        for dtype, value in zip(float_dts, values):
            image = np.zeros((5, 5), dtype=dtype)
            image[2, 2] = value
            image_aug = aug.augment_image(image)
            expected = (mask * value).astype(dtype)
            diff = np.abs(image_aug.astype(np.float64)
                          - expected.astype(np.float64))
            assert image_aug.shape == mask.shape
            assert image_aug.dtype.type == dtype
            # accepts difference of 2.0, 4.0, 8.0, 16.0 (at 1, 2, 4, 8 bytes,
            # i.e. 8, 16, 32, 64 bit)
            max_diff = np.dtype(dtype).itemsize * 0.01 * np.float64(value)
            assert np.max(diff) < max_diff

    # float128 is the only unsupported dtype (excluding non-numerics and
    # complex dtypes)
    @unittest.skipIf(
        not hasattr(np, "float128"),
        "Test can only be executed on systems that know numpy.float128"
    )
    def test_failure_on_invalid_dtypes(self):
        # assert failure on invalid dtypes
        aug = iaa.GaussianBlur(sigma=1.0)
        for dt in [np.float128]:
            got_exception = False
            try:
                _ = aug.augment_image(np.zeros((1, 1), dtype=dt))
            except Exception as exc:
                assert "forbidden dtype" in str(exc)
                got_exception = True
            assert got_exception

    def test_pickleable(self):
        aug = iaa.GaussianBlur((0.1, 3.0), seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10)


class TestAverageBlur(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAverageBlur, self).__init__(*args, **kwargs)

        base_img = np.zeros((11, 11, 1), dtype=np.uint8)
        base_img[5, 5, 0] = 200
        base_img[4, 5, 0] = 100
        base_img[6, 5, 0] = 100
        base_img[5, 4, 0] = 100
        base_img[5, 6, 0] = 100

        blur3x3 = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 11, 11, 11, 0, 0, 0, 0],
            [0, 0, 0, 11, 44, 56, 44, 11, 0, 0, 0],
            [0, 0, 0, 11, 56, 67, 56, 11, 0, 0, 0],
            [0, 0, 0, 11, 44, 56, 44, 11, 0, 0, 0],
            [0, 0, 0, 0, 11, 11, 11, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        blur3x3 = np.array(blur3x3, dtype=np.uint8)[..., np.newaxis]

        blur4x4 = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0],
            [0, 0, 0, 6, 25, 31, 31, 25, 6, 0, 0],
            [0, 0, 0, 6, 31, 38, 38, 31, 6, 0, 0],
            [0, 0, 0, 6, 31, 38, 38, 31, 6, 0, 0],
            [0, 0, 0, 6, 25, 31, 31, 25, 6, 0, 0],
            [0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        blur4x4 = np.array(blur4x4, dtype=np.uint8)[..., np.newaxis]

        blur5x5 = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0],
            [0, 0, 4, 16, 20, 20, 20, 16, 4, 0, 0],
            [0, 0, 4, 20, 24, 24, 24, 20, 4, 0, 0],
            [0, 0, 4, 20, 24, 24, 24, 20, 4, 0, 0],
            [0, 0, 4, 20, 24, 24, 24, 20, 4, 0, 0],
            [0, 0, 4, 16, 20, 20, 20, 16, 4, 0, 0],
            [0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        blur5x5 = np.array(blur5x5, dtype=np.uint8)[..., np.newaxis]

        self.base_img = base_img
        self.blur3x3 = blur3x3
        self.blur4x4 = blur4x4
        self.blur5x5 = blur5x5

    def setUp(self):
        reseed()

    def test_kernel_size_0(self):
        # no blur, shouldnt change anything
        aug = iaa.AverageBlur(k=0)
        observed = aug.augment_image(self.base_img)
        assert np.array_equal(observed, self.base_img)

    def test_kernel_size_3(self):
        # k=3
        aug = iaa.AverageBlur(k=3)
        observed = aug.augment_image(self.base_img)
        assert np.array_equal(observed, self.blur3x3)

    def test_kernel_size_5(self):
        # k=5
        aug = iaa.AverageBlur(k=5)
        observed = aug.augment_image(self.base_img)
        assert np.array_equal(observed, self.blur5x5)

    def test_kernel_size_is_tuple(self):
        # k as (3, 4)
        aug = iaa.AverageBlur(k=(3, 4))
        nb_iterations = 100
        nb_seen = [0, 0]
        for i in sm.xrange(nb_iterations):
            observed = aug.augment_image(self.base_img)
            if np.array_equal(observed, self.blur3x3):
                nb_seen[0] += 1
            elif np.array_equal(observed, self.blur4x4):
                nb_seen[1] += 1
            else:
                raise Exception("Unexpected result in AverageBlur@1")
        p_seen = [v/nb_iterations for v in nb_seen]
        assert 0.4 <= p_seen[0] <= 0.6
        assert 0.4 <= p_seen[1] <= 0.6

    def test_kernel_size_is_tuple_with_wider_range(self):
        # k as (3, 5)
        aug = iaa.AverageBlur(k=(3, 5))
        nb_iterations = 200
        nb_seen = [0, 0, 0]
        for i in sm.xrange(nb_iterations):
            observed = aug.augment_image(self.base_img)
            if np.array_equal(observed, self.blur3x3):
                nb_seen[0] += 1
            elif np.array_equal(observed, self.blur4x4):
                nb_seen[1] += 1
            elif np.array_equal(observed, self.blur5x5):
                nb_seen[2] += 1
            else:
                raise Exception("Unexpected result in AverageBlur@2")
        p_seen = [v/nb_iterations for v in nb_seen]
        assert 0.23 <= p_seen[0] <= 0.43
        assert 0.23 <= p_seen[1] <= 0.43
        assert 0.23 <= p_seen[2] <= 0.43

    def test_kernel_size_is_stochastic_parameter(self):
        # k as stochastic parameter
        aug = iaa.AverageBlur(k=iap.Choice([3, 5]))
        nb_iterations = 100
        nb_seen = [0, 0]
        for i in sm.xrange(nb_iterations):
            observed = aug.augment_image(self.base_img)
            if np.array_equal(observed, self.blur3x3):
                nb_seen[0] += 1
            elif np.array_equal(observed, self.blur5x5):
                nb_seen[1] += 1
            else:
                raise Exception("Unexpected result in AverageBlur@3")
        p_seen = [v/nb_iterations for v in nb_seen]
        assert 0.4 <= p_seen[0] <= 0.6
        assert 0.4 <= p_seen[1] <= 0.6

    def test_kernel_size_is_tuple_of_tuples(self):
        # k as ((3, 5), (3, 5))
        aug = iaa.AverageBlur(k=((3, 5), (3, 5)))

        possible = dict()
        for kh in [3, 4, 5]:
            for kw in [3, 4, 5]:
                key = (kh, kw)
                if kh == 0 or kw == 0:
                    possible[key] = np.copy(self.base_img)
                else:
                    possible[key] = cv2.blur(
                        self.base_img, (kh, kw))[..., np.newaxis]

        nb_iterations = 250
        nb_seen = dict([(key, 0) for key, val in possible.items()])
        for i in sm.xrange(nb_iterations):
            observed = aug.augment_image(self.base_img)
            for key, img_aug in possible.items():
                if np.array_equal(observed, img_aug):
                    nb_seen[key] += 1
        # dont check sum here, because 0xX and Xx0 are all the same, i.e. much
        # higher sum than nb_iterations
        assert np.all([v > 0 for v in nb_seen.values()])

    def test_more_than_four_channels(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.AverageBlur(k=3)(image=image)

                assert image_aug.shape == image.shape

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
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.AverageBlur(k=3)(image=image)

                assert image_aug.shape == image.shape

    def test_keypoints_dont_change(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        kpsoi = [ia.KeypointsOnImage(kps, shape=(11, 11, 1))]

        aug = iaa.AverageBlur(k=3)
        aug_det = aug.to_deterministic()
        observed = aug.augment_keypoints(kpsoi)
        expected = kpsoi
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(kpsoi)
        expected = kpsoi
        assert keypoints_equal(observed, expected)

    def test_other_dtypes_k0(self):
        aug = iaa.AverageBlur(k=0)

        # bool
        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True
        image[2, 2] = True
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == image)

        # uint, int
        uint_dts = [np.uint8, np.uint16]
        int_dts = [np.int8, np.int16]

        for dtype in uint_dts + int_dts:
            _min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = int(center_value + 0.4 * max_value)
            image[2, 2] = int(center_value + 0.4 * max_value)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == image)

        # float
        float_dts = [np.float16, np.float32, np.float64]
        values = [5000, 1000*1000, 1000*1000*1000]

        for dtype, value in zip(float_dts, values):
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image[2, 2] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.allclose(image_aug, image)

    def test_other_dtypes_k3_value_100(self):
        # --
        # blur of various dtypes at k=3
        # and using an example value of 100 for int/uint/float and True for
        # bool
        # --
        aug = iaa.AverageBlur(k=3)

        # prototype mask
        # we place values in a 3x3 grid at positions (row=1, col=1) and
        # (row=2, col=2) (beginning with 0)
        # AverageBlur uses cv2.blur(), which uses BORDER_REFLECT_101 as its
        # default padding mode,
        # see https://docs.opencv.org/3.1.0/d2/de8/group__core__array.html
        # the matrix below shows the 3x3 grid and the padded row/col values
        # around it
        # [1, 0, 1, 0, 1]
        # [0, 0, 0, 0, 0]
        # [1, 0, 1, 0, 1]
        # [0, 0, 0, 1, 0]
        # [1, 0, 1, 0, 1]
        mask = np.float64([
            [4/9, 2/9, 4/9],
            [2/9, 2/9, 3/9],
            [4/9, 3/9, 5/9]
        ])

        # bool
        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True
        image[2, 2] = True
        image_aug = aug.augment_image(image)
        expected = mask > 0.5
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == expected)

        # uint, int
        uint_dts = [np.uint8, np.uint16]
        int_dts = [np.int8, np.int16]

        for dtype in uint_dts + int_dts:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = 100
            image[2, 2] = 100
            image_aug = aug.augment_image(image)
            # cv2.blur() applies rounding for int/uint dtypes
            expected = np.round(mask * 100).astype(dtype)
            diff = np.abs(image_aug.astype(np.int64)
                          - expected.astype(np.int64))
            assert image_aug.dtype.type == dtype
            assert np.max(diff) <= 2

        # float
        float_dts = [np.float16, np.float32, np.float64]
        for dtype in float_dts:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = 100.0
            image[2, 2] = 100.0
            image_aug = aug.augment_image(image)
            expected = (mask * 100.0).astype(dtype)
            diff = np.abs(image_aug.astype(np.float64)
                          - expected.astype(np.float64))
            assert image_aug.dtype.type == dtype
            assert np.max(diff) < 1.0

    def test_other_dtypes_k3_dynamic_value(self):
        # --
        # blur of various dtypes at k=3
        # and values being half-way between center and maximum for each
        # dtype (bool is skipped as it doesnt make any sense here)
        # The goal of this test is to verify that no major loss of resolution
        # happens for large dtypes.
        # --
        aug = iaa.AverageBlur(k=3)

        # prototype mask (see above)
        mask = np.float64([
            [4/9, 2/9, 4/9],
            [2/9, 2/9, 3/9],
            [4/9, 3/9, 5/9]
        ])

        # uint, int
        uint_dts = [np.uint8, np.uint16]
        int_dts = [np.int8, np.int16]

        for dtype in uint_dts + int_dts:
            _min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)
            value = int(center_value + 0.4 * max_value)
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image[2, 2] = value
            image_aug = aug.augment_image(image)
            expected = (mask * value).astype(dtype)
            diff = np.abs(image_aug.astype(np.int64)
                          - expected.astype(np.int64))
            assert image_aug.dtype.type == dtype
            # accepts difference of 4, 8, 16 (at 1, 2, 4 bytes, i.e. 8, 16,
            # 32 bit)
            assert np.max(diff) <= 2**(1 + np.dtype(dtype).itemsize)

        # float
        float_dts = [np.float16, np.float32, np.float64]
        values = [5000, 1000*1000, 1000*1000*1000]

        for dtype, value in zip(float_dts, values):
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image[2, 2] = value
            image_aug = aug.augment_image(image)
            expected = (mask * value).astype(dtype)
            diff = np.abs(image_aug.astype(np.float64)
                          - expected.astype(np.float64))
            assert image_aug.dtype.type == dtype
            # accepts difference of 2.0, 4.0, 8.0, 16.0 (at 1, 2, 4, 8 bytes,
            # i.e. 8, 16, 32, 64 bit)
            assert np.max(diff) < 2**(1 + np.dtype(dtype).itemsize)

    def test_failure_on_invalid_dtypes(self):
        # assert failure on invalid dtypes
        aug = iaa.AverageBlur(k=3)
        for dt in [np.uint32, np.uint64, np.int32, np.int64]:
            got_exception = False
            try:
                _ = aug.augment_image(np.zeros((1, 1), dtype=dt))
            except Exception as exc:
                assert "forbidden dtype" in str(exc)
                got_exception = True
            assert got_exception

    def test_pickleable(self):
        aug = iaa.AverageBlur((1, 11), seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10)


class TestMedianBlur(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMedianBlur, self).__init__(*args, **kwargs)

        base_img = np.zeros((11, 11, 1), dtype=np.uint8)
        base_img[3:8, 3:8, 0] = 1
        base_img[4:7, 4:7, 0] = 2
        base_img[5:6, 5:6, 0] = 3

        blur3x3 = np.zeros_like(base_img)
        blur3x3[3:8, 3:8, 0] = 1
        blur3x3[4:7, 4:7, 0] = 2
        blur3x3[4, 4, 0] = 1
        blur3x3[4, 6, 0] = 1
        blur3x3[6, 4, 0] = 1
        blur3x3[6, 6, 0] = 1
        blur3x3[3, 3, 0] = 0
        blur3x3[3, 7, 0] = 0
        blur3x3[7, 3, 0] = 0
        blur3x3[7, 7, 0] = 0

        blur5x5 = np.copy(blur3x3)
        blur5x5[4, 3, 0] = 0
        blur5x5[3, 4, 0] = 0
        blur5x5[6, 3, 0] = 0
        blur5x5[7, 4, 0] = 0
        blur5x5[4, 7, 0] = 0
        blur5x5[3, 6, 0] = 0
        blur5x5[6, 7, 0] = 0
        blur5x5[7, 6, 0] = 0
        blur5x5[blur5x5 > 1] = 1

        self.base_img = base_img
        self.blur3x3 = blur3x3
        self.blur5x5 = blur5x5

    def setUp(self):
        reseed()

    def test_k_is_1(self):
        # no blur, shouldnt change anything
        aug = iaa.MedianBlur(k=1)
        observed = aug.augment_image(self.base_img)
        assert np.array_equal(observed, self.base_img)

    def test_k_is_3(self):
        # k=3
        aug = iaa.MedianBlur(k=3)
        observed = aug.augment_image(self.base_img)
        assert np.array_equal(observed, self.blur3x3)

    def test_k_is_5(self):
        # k=5
        aug = iaa.MedianBlur(k=5)
        observed = aug.augment_image(self.base_img)
        assert np.array_equal(observed, self.blur5x5)

    def test_k_is_tuple(self):
        # k as (3, 5)
        aug = iaa.MedianBlur(k=(3, 5))
        seen = [False, False]
        for i in sm.xrange(100):
            observed = aug.augment_image(self.base_img)
            if np.array_equal(observed, self.blur3x3):
                seen[0] = True
            elif np.array_equal(observed, self.blur5x5):
                seen[1] = True
            else:
                raise Exception("Unexpected result in MedianBlur@1")
            if all(seen):
                break
        assert np.all(seen)

    def test_k_is_stochastic_parameter(self):
        # k as stochastic parameter
        aug = iaa.MedianBlur(k=iap.Choice([3, 5]))
        seen = [False, False]
        for i in sm.xrange(100):
            observed = aug.augment_image(self.base_img)
            if np.array_equal(observed, self.blur3x3):
                seen[0] += True
            elif np.array_equal(observed, self.blur5x5):
                seen[1] += True
            else:
                raise Exception("Unexpected result in MedianBlur@2")
            if all(seen):
                break
        assert np.all(seen)

    def test_more_than_four_channels(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.MedianBlur(k=3)(image=image)

                assert image_aug.shape == image.shape

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
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.MedianBlur(k=3)(image=image)

                assert image_aug.shape == image.shape

    def test_keypoints_not_changed(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        kpsoi = [ia.KeypointsOnImage(kps, shape=(11, 11, 1))]

        aug = iaa.MedianBlur(k=3)
        aug_det = aug.to_deterministic()
        observed = aug.augment_keypoints(kpsoi)
        expected = kpsoi
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(kpsoi)
        expected = kpsoi
        assert keypoints_equal(observed, expected)

    def test_pickleable(self):
        aug = iaa.MedianBlur((1, 11), seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10)


# TODO extend these tests
class TestBilateralBlur(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0, 3),
            (0, 1, 3),
            (1, 0, 3)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.BilateralBlur(3)(image=image)

                assert image_aug.shape == image.shape

    def test_pickleable(self):
        aug = iaa.BilateralBlur((1, 11), seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10)


class TestMotionBlur(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_simple_parameters(self):
        # simple scenario
        aug = iaa.MotionBlur(k=3, angle=0, direction=0.0)
        matrix_func = aug.matrix
        matrices = [
            matrix_func(
                np.zeros((128, 128, 3), dtype=np.uint8),
                3,
                iarandom.RNG(i)
            ) for i in range(10)
        ]
        expected = np.float32([
            [0, 1.0/3, 0],
            [0, 1.0/3, 0],
            [0, 1.0/3, 0]
        ])
        for matrices_image in matrices:
            for matrix_channel in matrices_image:
                assert np.allclose(matrix_channel, expected)

    def test_simple_parameters_angle_is_90(self):
        # 90deg angle
        aug = iaa.MotionBlur(k=3, angle=90, direction=0.0)
        matrix_func = aug.matrix
        matrices = [
            matrix_func(
                np.zeros((128, 128, 3), dtype=np.uint8),
                3,
                iarandom.RNG(i)
            ) for i in range(10)
        ]
        expected = np.float32([
            [0, 0, 0],
            [1.0/3, 1.0/3, 1.0/3],
            [0, 0, 0]
        ])
        for matrices_image in matrices:
            for matrix_channel in matrices_image:
                assert np.allclose(matrix_channel, expected)

    def test_simple_parameters_angle_is_45(self):
        # 45deg angle
        aug = iaa.MotionBlur(k=3, angle=45, direction=0.0, order=0)
        matrix_func = aug.matrix
        matrices = [
            matrix_func(
                np.zeros((128, 128, 3), dtype=np.uint8),
                3,
                iarandom.RNG(i)
            ) for i in range(10)
        ]
        expected = np.float32([
            [0, 0, 1.0/3],
            [0, 1.0/3, 0],
            [1.0/3, 0, 0]
        ])
        for matrices_image in matrices:
            for matrix_channel in matrices_image:
                assert np.allclose(matrix_channel, expected)

    def test_simple_parameters_angle_is_list(self):
        # random angle
        aug = iaa.MotionBlur(k=3, angle=[0, 90], direction=0.0)
        matrix_func = aug.matrix
        matrices = [
            matrix_func(
                np.zeros((128, 128, 3), dtype=np.uint8),
                3,
                iarandom.RNG(i)
            ) for i in range(50)
        ]
        expected1 = np.float32([
            [0, 1.0/3, 0],
            [0, 1.0/3, 0],
            [0, 1.0/3, 0]
        ])
        expected2 = np.float32([
            [0, 0, 0],
            [1.0/3, 1.0/3, 1.0/3],
            [0, 0, 0],
        ])
        nb_seen = [0, 0]
        for matrices_image in matrices:
            assert np.allclose(matrices_image[0], matrices_image[1])
            assert np.allclose(matrices_image[1], matrices_image[2])
            for matrix_channel in matrices_image:
                if np.allclose(matrix_channel, expected1):
                    nb_seen[0] += 1
                elif np.allclose(matrix_channel, expected2):
                    nb_seen[1] += 1
        assert nb_seen[0] > 0
        assert nb_seen[1] > 0

    def test_k_is_5_angle_90(self):
        # 5x5
        aug = iaa.MotionBlur(k=5, angle=90, direction=0.0)
        matrix_func = aug.matrix
        matrices = [
            matrix_func(
                np.zeros((128, 128, 3), dtype=np.uint8),
                3,
                iarandom.RNG(i)
            ) for i in range(10)
        ]
        expected = np.float32([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        for matrices_image in matrices:
            for matrix_channel in matrices_image:
                assert np.allclose(matrix_channel, expected)

    def test_k_is_list_angle_90(self):
        # random k
        aug = iaa.MotionBlur(k=[3, 5], angle=90, direction=0.0)
        matrix_func = aug.matrix
        matrices = [
            matrix_func(
                np.zeros((128, 128, 3), dtype=np.uint8),
                3,
                iarandom.RNG(i)
            ) for i in range(50)
        ]
        expected1 = np.float32([
            [0, 0, 0],
            [1.0/3, 1.0/3, 1.0/3],
            [0, 0, 0],
        ])
        expected2 = np.float32([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        nb_seen = [0, 0]
        for matrices_image in matrices:
            assert np.allclose(matrices_image[0], matrices_image[1])
            assert np.allclose(matrices_image[1], matrices_image[2])
            for matrix_channel in matrices_image:
                if (matrix_channel.shape == expected1.shape
                        and np.allclose(matrix_channel, expected1)):
                    nb_seen[0] += 1
                elif (matrix_channel.shape == expected2.shape
                      and np.allclose(matrix_channel, expected2)):
                    nb_seen[1] += 1
        assert nb_seen[0] > 0
        assert nb_seen[1] > 0

    def test_failure_on_continuous_kernel_sizes(self):
        # k with choice [a, b, c, ...] must error in case of non-discrete
        # values
        got_exception = False
        try:
            _ = iaa.MotionBlur(k=[3, 3.5, 4])
        except Exception as exc:
            assert "to only contain integer" in str(exc)
            got_exception = True
        assert got_exception

    # TODO extend this to test sampled kernel sizes
    def test_k_is_tuple(self):
        # no error in case of (a, b), checks for #215
        aug = iaa.MotionBlur(k=(3, 7))
        for _ in range(10):
            _ = aug.augment_image(np.zeros((11, 11, 3), dtype=np.uint8))

    def test_direction_is_1(self):
        # direction 1.0
        aug = iaa.MotionBlur(k=3, angle=0, direction=1.0)
        matrix_func = aug.matrix
        matrices = [
            matrix_func(
                np.zeros((128, 128, 3), dtype=np.uint8),
                3,
                iarandom.RNG(i)
            ) for i in range(10)
        ]
        expected = np.float32([
            [0, 1.0/1.5, 0],
            [0, 0.5/1.5, 0],
            [0, 0.0/1.5, 0]
        ])
        for matrices_image in matrices:
            for matrix_channel in matrices_image:
                assert np.allclose(matrix_channel, expected, rtol=0, atol=1e-2)

    def test_direction_is_minus_1(self):
        # direction -1.0
        aug = iaa.MotionBlur(k=3, angle=0, direction=-1.0)
        matrix_func = aug.matrix
        matrices = [
            matrix_func(
                np.zeros((128, 128, 3), dtype=np.uint8),
                3,
                iarandom.RNG(i)
            ) for i in range(10)
        ]
        expected = np.float32([
            [0, 0.0/1.5, 0],
            [0, 0.5/1.5, 0],
            [0, 1.0/1.5, 0]
        ])
        for matrices_image in matrices:
            for matrix_channel in matrices_image:
                assert np.allclose(matrix_channel, expected, rtol=0, atol=1e-2)

    def test_direction_is_list(self):
        # random direction
        aug = iaa.MotionBlur(k=3, angle=[0, 90], direction=[-1.0, 1.0])
        matrix_func = aug.matrix
        matrices = [
            matrix_func(
                np.zeros((128, 128, 3), dtype=np.uint8),
                3,
                iarandom.RNG(i)
            ) for i in range(50)
        ]
        expected1 = np.float32([
            [0, 1.0/1.5, 0],
            [0, 0.5/1.5, 0],
            [0, 0.0/1.5, 0]
        ])
        expected2 = np.float32([
            [0, 0.0/1.5, 0],
            [0, 0.5/1.5, 0],
            [0, 1.0/1.5, 0]
        ])
        nb_seen = [0, 0]
        for matrices_image in matrices:
            assert np.allclose(matrices_image[0], matrices_image[1])
            assert np.allclose(matrices_image[1], matrices_image[2])
            for matrix_channel in matrices_image:
                if np.allclose(matrix_channel, expected1, rtol=0, atol=1e-2):
                    nb_seen[0] += 1
                elif np.allclose(matrix_channel, expected2, rtol=0, atol=1e-2):
                    nb_seen[1] += 1
        assert nb_seen[0] > 0
        assert nb_seen[1] > 0

    def test_k_is_3_angle_is_90_verify_results(self):
        # test of actual augmenter
        img = np.zeros((7, 7, 3), dtype=np.uint8)
        img[3-1:3+2, 3-1:3+2, :] = 255
        aug = iaa.MotionBlur(k=3, angle=90, direction=0.0)
        img_aug = aug.augment_image(img)
        v1 = (255*(1/3))
        v2 = (255*(1/3)) * 2
        v3 = (255*(1/3)) * 3
        expected = np.float32([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, v1, v2, v3, v2, v1, 0],
            [0, v1, v2, v3, v2, v1, 0],
            [0, v1, v2, v3, v2, v1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]).astype(np.uint8)
        expected = np.tile(expected[..., np.newaxis], (1, 1, 3))
        assert np.allclose(img_aug, expected)

    def test_pickleable(self):
        aug = iaa.MotionBlur((3, 11), seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10)


class TestMeanShiftBlur(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___defaults(self):
        aug = iaa.MeanShiftBlur()
        assert np.isclose(aug.spatial_window_radius.a.value, 5.0)
        assert np.isclose(aug.spatial_window_radius.b.value, 40.0)
        assert np.isclose(aug.color_window_radius.a.value, 5.0)
        assert np.isclose(aug.color_window_radius.b.value, 40.0)

    def test___init___custom(self):
        aug = iaa.MeanShiftBlur(
            spatial_radius=[1.0, 2.0, 3.0],
            color_radius=iap.Deterministic(5)
        )
        assert np.allclose(aug.spatial_window_radius.a, [1.0, 2.0, 3.0])
        assert aug.color_window_radius.value == 5

    def test_draw_samples(self):
        aug = iaa.MeanShiftBlur(
            spatial_radius=[1.0, 2.0, 3.0],
            color_radius=(1.0, 2.0)
        )
        batch = mock.Mock()
        batch.nb_rows = 100

        samples = aug._draw_samples(batch, iarandom.RNG(0))

        assert np.all(
            np.isclose(samples[0], 1.0)
            | np.isclose(samples[0], 2.0)
            | np.isclose(samples[0], 3.0)
        )
        assert np.all((1.0 <= samples[1]) | (samples[1] <= 2.0))

    @mock.patch("imgaug.augmenters.blur.blur_mean_shift_")
    def test_mocked(self, mock_ms):
        aug = iaa.MeanShiftBlur(
            spatial_radius=1,
            color_radius=2
        )
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        mock_ms.return_value = image

        _image_aug = aug(image=image)

        kwargs = mock_ms.call_args_list[0][1]
        assert mock_ms.call_count == 1
        assert np.isclose(kwargs["spatial_window_radius"], 1.0)
        assert np.isclose(kwargs["color_window_radius"], 2.0)

    def test_batch_without_images(self):
        aug = iaa.MeanShiftBlur()
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(5, 5, 3))

        kps_aug = aug(keypoints=kpsoi)

        assert kps_aug.keypoints[0].x == 0
        assert kps_aug.keypoints[0].y == 1

    def test_get_parameters(self):
        aug = iaa.MeanShiftBlur()
        params = aug.get_parameters()
        assert params[0] is aug.spatial_window_radius
        assert params[1] is aug.color_window_radius

    def test_pickleable(self):
        aug = iaa.MeanShiftBlur(seed=1)
        runtest_pickleable_uint8_img(aug, iterations=5, shape=(40, 40, 3))
