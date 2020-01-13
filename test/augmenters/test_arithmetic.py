from __future__ import print_function, division, absolute_import

import functools
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

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug import random as iarandom
from imgaug.testutils import (array_equal_lists, keypoints_equal, reseed,
                              runtest_pickleable_uint8_img)
import imgaug.augmenters.arithmetic as arithmetic_lib
import imgaug.augmenters.contrast as contrast_lib


class Test_cutout(unittest.TestCase):
    @mock.patch("imgaug.augmenters.arithmetic.cutout_")
    def test_mocked(self, mock_inplace):
        image = np.mod(np.arange(100*100*3), 255).astype(np.uint8).reshape(
            (100, 100, 3))
        mock_inplace.return_value = "foo"

        rng = iarandom.RNG(0)
        image_aug = iaa.cutout(image,
                               x1=10,
                               y1=20,
                               x2=30,
                               y2=40,
                               fill_mode="gaussian",
                               cval=1,
                               fill_per_channel=0.5,
                               seed=rng)

        assert mock_inplace.call_count == 1
        assert image_aug == "foo"

        args = mock_inplace.call_args_list[0][0]
        assert args[0] is not image
        assert np.array_equal(args[0], image)
        assert np.isclose(args[1], 10)
        assert np.isclose(args[2], 20)
        assert np.isclose(args[3], 30)
        assert np.isclose(args[4], 40)
        assert args[5] == "gaussian"
        assert args[6] == 1
        assert np.isclose(args[7], 0.5)
        assert args[8] is rng


class Test_cutout_(unittest.TestCase):
    def test_with_simple_image(self):
        image = np.mod(np.arange(100*100*3), 255).astype(np.uint8).reshape(
            (100, 100, 3))
        image = 1 + image

        image_aug = iaa.cutout_(image,
                                x1=10,
                                y1=20,
                                x2=30,
                                y2=40,
                                fill_mode="constant",
                                cval=0,
                                fill_per_channel=False,
                                seed=None)

        mask = np.zeros(image.shape, dtype=bool)
        mask[20:40, 10:30, :] = True
        overlap_inside = np.sum(image_aug[mask] == 0) / np.sum(mask)
        overlap_outside = np.sum(image_aug[~mask] > 0) / np.sum(~mask)
        assert image_aug is image
        assert overlap_inside >= 1.0 - 1e-4
        assert overlap_outside >= 1.0 - 1e-4

    @mock.patch("imgaug.augmenters.arithmetic._fill_rectangle_constant_")
    def test_fill_mode_constant_mocked(self, mock_fill):
        self._test_with_fill_mode_mocked("constant", mock_fill)

    @mock.patch("imgaug.augmenters.arithmetic._fill_rectangle_gaussian_")
    def test_fill_mode_gaussian_mocked(self, mock_fill):
        self._test_with_fill_mode_mocked("gaussian", mock_fill)

    @classmethod
    def _test_with_fill_mode_mocked(cls, fill_mode, mock_fill):
        image = np.mod(np.arange(100*100*3), 256).astype(np.uint8).reshape(
            (100, 100, 3))
        mock_fill.return_value = image

        seed = iarandom.RNG(0)

        image_aug = iaa.cutout_(image,
                                x1=10,
                                y1=20,
                                x2=30,
                                y2=40,
                                fill_mode=fill_mode,
                                cval=0,
                                fill_per_channel=False,
                                seed=seed)

        assert mock_fill.call_count == 1
        args = mock_fill.call_args_list[0][0]
        kwargs = mock_fill.call_args_list[0][1]
        assert image_aug is image
        assert args[0] is image
        assert kwargs["x1"] == 10
        assert kwargs["y1"] == 20
        assert kwargs["x2"] == 30
        assert kwargs["y2"] == 40
        assert kwargs["cval"] == 0
        assert kwargs["per_channel"] is False
        assert kwargs["random_state"] is seed

    def test_zero_height(self):
        image = np.mod(np.arange(100*100*3), 255).astype(np.uint8).reshape(
            (100, 100, 3))
        image = 1 + image
        image_cp = np.copy(image)

        image_aug = iaa.cutout_(image,
                                x1=10,
                                y1=20,
                                x2=30,
                                y2=20,
                                fill_mode="constant",
                                cval=0,
                                fill_per_channel=False,
                                seed=None)

        assert np.array_equal(image_aug, image_cp)

    def test_zero_height_width(self):
        image = np.mod(np.arange(100*100*3), 255).astype(np.uint8).reshape(
            (100, 100, 3))
        image = 1 + image
        image_cp = np.copy(image)

        image_aug = iaa.cutout_(image,
                                x1=10,
                                y1=20,
                                x2=10,
                                y2=40,
                                fill_mode="constant",
                                cval=0,
                                fill_per_channel=False,
                                seed=None)

        assert np.array_equal(image_aug, image_cp)

    def test_position_outside_of_image_rect_fully_outside(self):
        image = np.mod(np.arange(100*100*3), 255).astype(np.uint8).reshape(
            (100, 100, 3))
        image = 1 + image
        image_cp = np.copy(image)

        image_aug = iaa.cutout_(image,
                                x1=-50,
                                y1=150,
                                x2=-1,
                                y2=200,
                                fill_mode="constant",
                                cval=0,
                                fill_per_channel=False,
                                seed=None)

        assert np.array_equal(image_aug, image_cp)

    def test_position_outside_of_image_rect_partially_inside(self):
        image = np.mod(np.arange(100*100*3), 255).astype(np.uint8).reshape(
            (100, 100, 3))
        image = 1 + image

        image_aug = iaa.cutout_(image,
                                x1=-25,
                                y1=-25,
                                x2=25,
                                y2=25,
                                fill_mode="constant",
                                cval=0,
                                fill_per_channel=False,
                                seed=None)

        assert np.all(image_aug[0:25, 0:25] == 0)
        assert np.all(image_aug[0:25, 25:] > 0)
        assert np.all(image_aug[25:, :] > 0)

    def test_zero_sized_axes(self):
        shapes = [(0, 0, 0),
                  (1, 0, 0),
                  (0, 1, 0),
                  (0, 1, 1),
                  (1, 1, 0),
                  (1, 0, 1),
                  (1, 0),
                  (0, 1),
                  (0, 0)]
        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                image_cp = np.copy(image)

                image_aug = iaa.cutout_(image,
                                        x1=-5,
                                        y1=-5,
                                        x2=5,
                                        y2=5,
                                        fill_mode="constant",
                                        cval=0)

                assert np.array_equal(image_aug, image_cp)


class Test_fill_rectangle_gaussian_(unittest.TestCase):
    def test_simple_image(self):
        image = np.mod(np.arange(100*100*3), 256).astype(np.uint8).reshape(
            (100, 100, 3))
        image_cp = np.copy(image)
        rng = iarandom.RNG(0)

        image_aug = arithmetic_lib._fill_rectangle_gaussian_(
            image,
            x1=10,
            y1=20,
            x2=60,
            y2=70,
            cval=0,
            per_channel=False,
            random_state=rng)

        assert np.array_equal(image_aug[:20, :],
                              image_cp[:20, :])
        assert not np.array_equal(image_aug[20:70, 10:60],
                                  image_cp[20:70, 10:60])
        assert np.isclose(np.average(image_aug[20:70, 10:60]), 127.5,
                          rtol=0, atol=5.0)
        assert np.isclose(np.std(image_aug[20:70, 10:60]), 255.0/2.0/3.0,
                          rtol=0, atol=2.5)

    def test_per_channel(self):
        image = np.uint8([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        image = np.tile(image.reshape((1, 10, 1)), (1, 1, 3))

        image_aug = arithmetic_lib._fill_rectangle_gaussian_(
            np.copy(image),
            x1=0, y1=0, x2=10, y2=1,
            cval=0,
            per_channel=False,
            random_state=iarandom.RNG(0))

        image_aug_pc = arithmetic_lib._fill_rectangle_gaussian_(
            np.copy(image),
            x1=0, y1=0, x2=10, y2=1,
            cval=0,
            per_channel=True,
            random_state=iarandom.RNG(0))

        diff11 = (image_aug[..., 0] != image_aug[..., 1])
        diff12 = (image_aug[..., 0] != image_aug[..., 2])
        diff21 = (image_aug_pc[..., 0] != image_aug_pc[..., 1])
        diff22 = (image_aug_pc[..., 0] != image_aug_pc[..., 2])

        assert not np.any(diff11)
        assert not np.any(diff12)
        assert np.any(diff21)
        assert np.any(diff22)

    def test_deterministic_with_same_seed(self):
        image = np.uint8([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        image = np.tile(image.reshape((1, 10, 1)), (1, 1, 3))

        image_aug_pc1 = arithmetic_lib._fill_rectangle_gaussian_(
            np.copy(image),
            x1=0, y1=0, x2=10, y2=1,
            cval=0,
            per_channel=True,
            random_state=iarandom.RNG(0))

        image_aug_pc2 = arithmetic_lib._fill_rectangle_gaussian_(
            np.copy(image),
            x1=0, y1=0, x2=10, y2=1,
            cval=0,
            per_channel=True,
            random_state=iarandom.RNG(0))

        image_aug_pc3 = arithmetic_lib._fill_rectangle_gaussian_(
            np.copy(image),
            x1=0, y1=0, x2=10, y2=1,
            cval=0,
            per_channel=True,
            random_state=iarandom.RNG(1))

        assert np.array_equal(image_aug_pc1, image_aug_pc2)
        assert not np.array_equal(image_aug_pc2, image_aug_pc3)

    def test_no_channels(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                image = np.uint8([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                image = image.reshape((1, 10))

                image_aug = arithmetic_lib._fill_rectangle_gaussian_(
                    np.copy(image),
                    x1=0, y1=0, x2=10, y2=1,
                    cval=0,
                    per_channel=per_channel,
                    random_state=iarandom.RNG(0))

                assert not np.array_equal(image_aug, image)

    def test_unusual_channel_numbers(self):
        for nb_channels in [1, 2, 3, 4, 5, 511, 512, 513]:
            for per_channel in [False, True]:
                with self.subTest(nb_channels=nb_channels,
                                  per_channel=per_channel):
                    image = np.uint8([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                    image = np.tile(image.reshape((1, 10, 1)),
                                    (1, 1, nb_channels))

                    image_aug = arithmetic_lib._fill_rectangle_gaussian_(
                        np.copy(image),
                        x1=0, y1=0, x2=10, y2=1,
                        cval=0,
                        per_channel=True,
                        random_state=iarandom.RNG(0))

                    assert not np.array_equal(image_aug, image)

    def test_other_dtypes_bool(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                image = np.array([0, 1], dtype=bool)
                image = np.tile(image, (int(3*300*300/2),))
                image = image.reshape((300, 300, 3))
                image_cp = np.copy(image)
                rng = iarandom.RNG(0)

                image_aug = arithmetic_lib._fill_rectangle_gaussian_(
                    image,
                    x1=10,
                    y1=10,
                    x2=300-10,
                    y2=300-10,
                    cval=0,
                    per_channel=per_channel,
                    random_state=rng)

                rect = image_aug[10:-10, 10:-10]
                p_true = np.sum(rect) / rect.size
                assert np.array_equal(image_aug[:10, :], image_cp[:10, :])
                assert not np.array_equal(rect, image_cp[10:-10, 10:-10])
                assert np.isclose(p_true, 0.5, rtol=0, atol=0.1)
                if per_channel:
                    for c in np.arange(1, image.shape[2]):
                        assert not np.array_equal(image_aug[..., 0],
                                                  image_aug[..., c])

    def test_other_dtypes_int_uint(self):
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64"]
        for dtype in dtypes:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)
            dynamic_range = int(max_value) - int(min_value)

            gaussian_min = iarandom.RNG(0).normal(min_value, 0.0001,
                                                  size=(1,))
            gaussian_max = iarandom.RNG(0).normal(max_value, 0.0001,
                                                  size=(1,))
            assert min_value - 1.0 <= gaussian_min <= min_value + 1.0
            assert max_value - 1.0 <= gaussian_max <= max_value + 1.0

            for per_channel in [False, True]:
                with self.subTest(dtype=dtype, per_channel=per_channel):
                    # dont generate image from choice() here, that seems
                    # to not support uint64 (max value not in result)
                    image = np.array([min_value, min_value+1,
                                      int(center_value),
                                      max_value-1, max_value], dtype=dtype)
                    image = np.tile(image, (int(3*300*300/5),))
                    image = image.reshape((300, 300, 3))
                    assert min_value in image
                    assert max_value in image
                    image_cp = np.copy(image)
                    rng = iarandom.RNG(0)

                    image_aug = arithmetic_lib._fill_rectangle_gaussian_(
                        image, x1=10, y1=10, x2=300-10, y2=300-10,
                        cval=0, per_channel=per_channel, random_state=rng)

                    rect = image_aug[10:-10, 10:-10]
                    mean = np.average(np.float128(rect))
                    std = np.std(np.float128(rect) - center_value)
                    assert np.array_equal(image_aug[:10, :], image_cp[:10, :])
                    assert not np.array_equal(rect,
                                              image_cp[10:-10, 10:-10])
                    assert np.isclose(mean, center_value, rtol=0,
                                      atol=0.05*dynamic_range)
                    assert np.isclose(std, dynamic_range/2.0/3.0, rtol=0,
                                      atol=0.05*dynamic_range/2.0/3.0)
                    assert np.min(rect) < min_value + 0.2 * dynamic_range
                    assert np.max(rect) > max_value - 0.2 * dynamic_range

                    if per_channel:
                        for c in np.arange(1, image.shape[2]):
                            assert not np.array_equal(image_aug[..., 0],
                                                      image_aug[..., c])

    def test_other_dtypes_float(self):
        dtypes = ["float16", "float32", "float64", "float128"]
        for dtype in dtypes:
            min_value = 0.0
            center_value = 0.5
            max_value = 1.0
            dynamic_range = np.float128(max_value) - np.float128(min_value)

            gaussian_min = iarandom.RNG(0).normal(min_value, 0.0001,
                                                  size=(1,))
            gaussian_max = iarandom.RNG(0).normal(max_value, 0.0001,
                                                  size=(1,))
            assert min_value - 1.0 <= gaussian_min <= min_value + 1.0
            assert max_value - 1.0 <= gaussian_max <= max_value + 1.0

            for per_channel in [False, True]:
                with self.subTest(dtype=dtype, per_channel=per_channel):
                    # dont generate image from choice() here, that seems
                    # to not support uint64 (max value not in result)
                    image = np.array([min_value, min_value+1,
                                      int(center_value),
                                      max_value-1, max_value], dtype=dtype)
                    image = np.tile(image, (int(3*300*300/5),))
                    image = image.reshape((300, 300, 3))
                    assert np.any(np.isclose(image, min_value,
                                             rtol=0, atol=1e-4))
                    assert np.any(np.isclose(image, max_value,
                                             rtol=0, atol=1e-4))
                    image_cp = np.copy(image)
                    rng = iarandom.RNG(0)

                    image_aug = arithmetic_lib._fill_rectangle_gaussian_(
                        image, x1=10, y1=10, x2=300-10, y2=300-10,
                        cval=0, per_channel=per_channel, random_state=rng)

                    rect = image_aug[10:-10, 10:-10]
                    mean = np.average(np.float128(rect))
                    std = np.std(np.float128(rect) - center_value)
                    assert np.allclose(image_aug[:10, :], image_cp[:10, :],
                                       rtol=0, atol=1e-4)
                    assert not np.allclose(rect, image_cp[10:-10, 10:-10],
                                           rtol=0, atol=1e-4)
                    assert np.isclose(mean, center_value, rtol=0,
                                      atol=0.05*dynamic_range)
                    assert np.isclose(std, dynamic_range/2.0/3.0, rtol=0,
                                      atol=0.05*dynamic_range/2.0/3.0)
                    assert np.min(rect) < min_value + 0.2 * dynamic_range
                    assert np.max(rect) > max_value - 0.2 * dynamic_range

                    if per_channel:
                        for c in np.arange(1, image.shape[2]):
                            assert not np.allclose(image_aug[..., 0],
                                                   image_aug[..., c],
                                                   rtol=0, atol=1e-4)


class Test_fill_rectangle_constant_(unittest.TestCase):
    def test_simple_image(self):
        image = np.mod(np.arange(100*100*3), 256).astype(np.uint8).reshape(
            (100, 100, 3))
        image_cp = np.copy(image)

        image_aug = arithmetic_lib._fill_rectangle_constant_(
            image,
            x1=10, y1=20, x2=60, y2=70,
            cval=17, per_channel=False, random_state=None)

        assert np.array_equal(image_aug[:20, :], image_cp[:20, :])
        assert np.all(image_aug[20:70, 10:60] == 17)

    def test_iterable_cval_but_per_channel_is_false(self):
        image = np.mod(np.arange(100*100*3), 256).astype(np.uint8).reshape(
            (100, 100, 3))
        image_cp = np.copy(image)

        image_aug = arithmetic_lib._fill_rectangle_constant_(
            image,
            x1=10, y1=20, x2=60, y2=70,
            cval=[17, 21, 25], per_channel=False, random_state=None)

        assert np.array_equal(image_aug[:20, :], image_cp[:20, :])
        assert np.all(image_aug[20:70, 10:60] == 17)

    def test_iterable_cval_with_per_channel_is_true(self):
        image = np.mod(np.arange(100*100*3), 256).astype(np.uint8).reshape(
            (100, 100, 3))
        image_cp = np.copy(image)

        image_aug = arithmetic_lib._fill_rectangle_constant_(
            image,
            x1=10, y1=20, x2=60, y2=70,
            cval=[17, 21, 25], per_channel=True, random_state=None)

        assert np.array_equal(image_aug[:20, :], image_cp[:20, :])
        assert np.all(image_aug[20:70, 10:60, 0] == 17)
        assert np.all(image_aug[20:70, 10:60, 1] == 21)
        assert np.all(image_aug[20:70, 10:60, 2] == 25)

    def test_iterable_cval_with_per_channel_is_true_channel_mismatch(self):
        image = np.mod(np.arange(100*100*5), 256).astype(np.uint8).reshape(
            (100, 100, 5))
        image_cp = np.copy(image)

        image_aug = arithmetic_lib._fill_rectangle_constant_(
            image,
            x1=10, y1=20, x2=60, y2=70,
            cval=[17, 21], per_channel=True, random_state=None)

        assert np.array_equal(image_aug[:20, :], image_cp[:20, :])
        assert np.all(image_aug[20:70, 10:60, 0] == 17)
        assert np.all(image_aug[20:70, 10:60, 1] == 21)
        assert np.all(image_aug[20:70, 10:60, 2] == 17)
        assert np.all(image_aug[20:70, 10:60, 3] == 21)
        assert np.all(image_aug[20:70, 10:60, 4] == 17)

    def test_single_cval_with_per_channel_is_true(self):
        image = np.mod(np.arange(100*100*3), 256).astype(np.uint8).reshape(
            (100, 100, 3))
        image_cp = np.copy(image)

        image_aug = arithmetic_lib._fill_rectangle_constant_(
            image,
            x1=10, y1=20, x2=60, y2=70,
            cval=17, per_channel=True, random_state=None)

        assert np.array_equal(image_aug[:20, :], image_cp[:20, :])
        assert np.all(image_aug[20:70, 10:60, 0] == 17)
        assert np.all(image_aug[20:70, 10:60, 1] == 17)
        assert np.all(image_aug[20:70, 10:60, 2] == 17)

    def test_no_channels_single_cval(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                image = np.mod(
                    np.arange(100*100), 256
                ).astype(np.uint8).reshape((100, 100))
                image_cp = np.copy(image)

                image_aug = arithmetic_lib._fill_rectangle_constant_(
                    image,
                    x1=10, y1=20, x2=60, y2=70,
                    cval=17, per_channel=per_channel, random_state=None)

                assert np.array_equal(image_aug[:20, :], image_cp[:20, :])
                assert np.all(image_aug[20:70, 10:60] == 17)

    def test_no_channels_iterable_cval(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                image = np.mod(
                    np.arange(100*100), 256
                ).astype(np.uint8).reshape((100, 100))
                image_cp = np.copy(image)

                image_aug = arithmetic_lib._fill_rectangle_constant_(
                    image,
                    x1=10, y1=20, x2=60, y2=70,
                    cval=[17, 21, 25], per_channel=per_channel,
                    random_state=None)

                assert np.array_equal(image_aug[:20, :], image_cp[:20, :])
                assert np.all(image_aug[20:70, 10:60] == 17)

    def test_unusual_channel_numbers(self):
        for nb_channels in [1, 2, 4, 5, 511, 512, 513]:
            for per_channel in [False, True]:
                with self.subTest(per_channel=per_channel):
                    image = np.mod(
                        np.arange(100*100*nb_channels), 256
                    ).astype(np.uint8).reshape((100, 100, nb_channels))
                    image_cp = np.copy(image)

                    image_aug = arithmetic_lib._fill_rectangle_constant_(
                        image,
                        x1=10, y1=20, x2=60, y2=70,
                        cval=[17, 21], per_channel=per_channel,
                        random_state=None)

                    assert np.array_equal(image_aug[:20, :], image_cp[:20, :])
                    if per_channel:
                        for c in np.arange(nb_channels):
                            val = 17 if c % 2 == 0 else 21
                            assert np.all(image_aug[20:70, 10:60, c] == val)
                    else:
                        assert np.all(image_aug[20:70, 10:60, :] == 17)

    def test_other_dtypes_bool(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                image = np.array([0, 1], dtype=bool)
                image = np.tile(image, (int(3*300*300/2),))
                image = image.reshape((300, 300, 3))
                image_cp = np.copy(image)

                image_aug = arithmetic_lib._fill_rectangle_constant_(
                    image,
                    x1=10, y1=10, x2=300-10, y2=300-10,
                    cval=[0, 1], per_channel=per_channel,
                    random_state=None)

                rect = image_aug[10:-10, 10:-10]
                assert np.array_equal(image_aug[:10, :], image_cp[:10, :])
                if per_channel:
                    assert np.all(image_aug[10:-10, 10:-10, 0] == 0)
                    assert np.all(image_aug[10:-10, 10:-10, 1] == 1)
                    assert np.all(image_aug[10:-10, 10:-10, 2] == 0)
                else:
                    assert np.all(image_aug[20:70, 10:60] == 0)

    def test_other_dtypes_uint_int(self):
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64"]
        for dtype in dtypes:
            for per_channel in [False, True]:
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                with self.subTest(dtype=dtype, per_channel=per_channel):
                    image = np.array([min_value, min_value+1,
                                      int(center_value),
                                      max_value-1, max_value], dtype=dtype)
                    image = np.tile(image, (int(3*300*300/5),))
                    image = image.reshape((300, 300, 3))
                    assert min_value in image
                    assert max_value in image
                    image_cp = np.copy(image)

                    image_aug = arithmetic_lib._fill_rectangle_constant_(
                        image,
                        x1=10, y1=10, x2=300-10, y2=300-10,
                        cval=[min_value, 10, max_value],
                        per_channel=per_channel,
                        random_state=None)

                    assert np.array_equal(image_aug[:10, :], image_cp[:10, :])
                    if per_channel:
                        assert np.all(image_aug[10:-10, 10:-10, 0]
                                      == min_value)
                        assert np.all(image_aug[10:-10, 10:-10, 1]
                                      == 10)
                        assert np.all(image_aug[10:-10, 10:-10, 2]
                                      == max_value)
                    else:
                        assert np.all(image_aug[-10:-10, 10:-10] == min_value)

    def test_other_dtypes_float(self):
        dtypes = ["float16", "float32", "float64", "float128"]
        for dtype in dtypes:
            for per_channel in [False, True]:
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                with self.subTest(dtype=dtype, per_channel=per_channel):
                    image = np.array([min_value, min_value+1,
                                      int(center_value),
                                      max_value-1, max_value], dtype=dtype)
                    image = np.tile(image, (int(3*300*300/5),))
                    image = image.reshape((300, 300, 3))

                    # Use this here instead of any(isclose(...)) because
                    # the latter one leads to overflow warnings.
                    assert image.flat[0] <= np.float128(min_value) + 1.0
                    assert image.flat[4] >= np.float128(max_value) - 1.0

                    image_cp = np.copy(image)

                    image_aug = arithmetic_lib._fill_rectangle_constant_(
                        image,
                        x1=10, y1=10, x2=300-10, y2=300-10,
                        cval=[min_value, 10, max_value],
                        per_channel=per_channel,
                        random_state=None)

                    assert image_aug.dtype.name == dtype
                    assert np.allclose(image_aug[:10, :], image_cp[:10, :],
                                       rtol=0, atol=1e-4)
                    if per_channel:
                        assert np.allclose(image_aug[10:-10, 10:-10, 0],
                                           np.float128(min_value),
                                           rtol=0, atol=1e-4)
                        assert np.allclose(image_aug[10:-10, 10:-10, 1],
                                           np.float128(10),
                                           rtol=0, atol=1e-4)
                        assert np.allclose(image_aug[10:-10, 10:-10, 2],
                                           np.float128(max_value),
                                           rtol=0, atol=1e-4)
                    else:
                        assert np.allclose(image_aug[-10:-10, 10:-10],
                                           np.float128(min_value),
                                           rtol=0, atol=1e-4)


class TestAdd(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _ = iaa.Add(value="test")
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _ = iaa.Add(value=1, per_channel="test")
        except Exception:
            got_exception = True
        assert got_exception

    def test_add_zero(self):
        # no add, shouldnt change anything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Add(value=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

    def test_add_one(self):
        # add > 0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Add(value=1)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images + 1
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [images_list[0] + 1]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images + 1
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [images_list[0] + 1]
        assert array_equal_lists(observed, expected)

    def test_minus_one(self):
        # add < 0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Add(value=-1)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images - 1
        assert np.array_equal(observed, expected)

        observed = aug.augment_images(images_list)
        expected = [images_list[0] - 1]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images - 1
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [images_list[0] - 1]
        assert array_equal_lists(observed, expected)

    def test_uint8_every_possible_value(self):
        # uint8, every possible addition for base value 127
        for value_type in [float, int]:
            for per_channel in [False, True]:
                for value in np.arange(-255, 255+1):
                    aug = iaa.Add(value=value_type(value), per_channel=per_channel)
                    expected = np.clip(127 + value_type(value), 0, 255)

                    img = np.full((1, 1), 127, dtype=np.uint8)
                    img_aug = aug.augment_image(img)
                    assert img_aug.item(0) == expected

                    img = np.full((1, 1, 3), 127, dtype=np.uint8)
                    img_aug = aug.augment_image(img)
                    assert np.all(img_aug == expected)

    def test_add_floats(self):
        # specific tests with floats
        aug = iaa.Add(value=0.75)
        img = np.full((1, 1), 1, dtype=np.uint8)
        img_aug = aug.augment_image(img)
        assert img_aug.item(0) == 2

        img = np.full((1, 1), 1, dtype=np.uint16)
        img_aug = aug.augment_image(img)
        assert img_aug.item(0) == 2

        aug = iaa.Add(value=0.45)
        img = np.full((1, 1), 1, dtype=np.uint8)
        img_aug = aug.augment_image(img)
        assert img_aug.item(0) == 1

        img = np.full((1, 1), 1, dtype=np.uint16)
        img_aug = aug.augment_image(img)
        assert img_aug.item(0) == 1

    def test_stochastic_parameters_as_value(self):
        # test other parameters
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

        aug = iaa.Add(value=iap.DiscreteUniform(1, 10))
        observed = aug.augment_images(images)
        assert 100 + 1 <= np.average(observed) <= 100 + 10

        aug = iaa.Add(value=iap.Uniform(1, 10))
        observed = aug.augment_images(images)
        assert 100 + 1 <= np.average(observed) <= 100 + 10

        aug = iaa.Add(value=iap.Clip(iap.Normal(1, 1), -3, 3))
        observed = aug.augment_images(images)
        assert 100 - 3 <= np.average(observed) <= 100 + 3

        aug = iaa.Add(value=iap.Discretize(iap.Clip(iap.Normal(1, 1), -3, 3)))
        observed = aug.augment_images(images)
        assert 100 - 3 <= np.average(observed) <= 100 + 3

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.Add(value=1)
        aug_det = iaa.Add(value=1).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test_tuple_as_value(self):
        # varying values
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

        aug = iaa.Add(value=(0, 10))
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
        assert nb_changed_aug >= int(nb_iterations * 0.7)
        assert nb_changed_aug_det == 0

    def test_per_channel(self):
        # test channelwise
        aug = iaa.Add(value=iap.Choice([0, 1]), per_channel=True)
        observed = aug.augment_image(np.zeros((1, 1, 100), dtype=np.uint8))
        uq = np.unique(observed)
        assert observed.shape == (1, 1, 100)
        assert 0 in uq
        assert 1 in uq
        assert len(uq) == 2

    def test_per_channel_with_probability(self):
        # test channelwise with probability
        aug = iaa.Add(value=iap.Choice([0, 1]), per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(400):
            observed = aug.augment_image(np.zeros((1, 1, 20), dtype=np.uint8))
            assert observed.shape == (1, 1, 20)

            uq = np.unique(observed)
            per_channel = (len(uq) == 2)
            if per_channel:
                seen[0] += 1
            else:
                seen[1] += 1
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

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
                aug = iaa.Add(1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

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
                aug = iaa.Add(1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.Add(value=1, per_channel=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert params[0].value == 1
        assert params[1].value == 0

    def test_heatmaps(self):
        # test heatmaps (not affected by augmenter)
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        aug = iaa.Add(value=10)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_bool(self):
        image = np.zeros((3, 3), dtype=bool)
        aug = iaa.Add(value=1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.Add(value=1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.Add(value=-1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.Add(value=-2)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

    def test_other_dtypes_uint_int(self):
        for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            image = np.full((3, 3), min_value, dtype=dtype)
            aug = iaa.Add(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == min_value + 1)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.Add(11)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == min_value + 21)

            image = np.full((3, 3), max_value - 2, dtype=dtype)
            aug = iaa.Add(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == max_value - 1)

            image = np.full((3, 3), max_value - 1, dtype=dtype)
            aug = iaa.Add(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == max_value)

            image = np.full((3, 3), max_value - 1, dtype=dtype)
            aug = iaa.Add(2)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == max_value)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.Add(-9)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == min_value + 1)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.Add(-10)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == min_value)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.Add(-11)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == min_value)

            for _ in sm.xrange(10):
                image = np.full((1, 1, 3), 20, dtype=dtype)
                aug = iaa.Add(iap.Uniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                assert len(np.unique(image_aug)) == 1

                image = np.full((1, 1, 100), 20, dtype=dtype)
                aug = iaa.Add(iap.Uniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                assert len(np.unique(image_aug)) > 1

                image = np.full((1, 1, 3), 20, dtype=dtype)
                aug = iaa.Add(iap.DiscreteUniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                assert len(np.unique(image_aug)) == 1

                image = np.full((1, 1, 100), 20, dtype=dtype)
                aug = iaa.Add(iap.DiscreteUniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                assert len(np.unique(image_aug)) > 1

    def test_other_dtypes_float(self):
        # float
        for dtype in [np.float16, np.float32]:
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            if dtype == np.float16:
                atol = 1e-3 * max_value
            else:
                atol = 1e-9 * max_value
            _allclose = functools.partial(np.allclose, atol=atol, rtol=0)

            image = np.full((3, 3), min_value, dtype=dtype)
            aug = iaa.Add(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, min_value + 1)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.Add(11)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, min_value + 21)

            image = np.full((3, 3), max_value - 2, dtype=dtype)
            aug = iaa.Add(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, max_value - 1)

            image = np.full((3, 3), max_value - 1, dtype=dtype)
            aug = iaa.Add(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, max_value)

            image = np.full((3, 3), max_value - 1, dtype=dtype)
            aug = iaa.Add(2)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, max_value)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.Add(-9)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, min_value + 1)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.Add(-10)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, min_value)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.Add(-11)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, min_value)

            for _ in sm.xrange(10):
                image = np.full((50, 1, 3), 0, dtype=dtype)
                aug = iaa.Add(iap.Uniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
                assert np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
                assert np.allclose(image_aug[..., 0], image_aug[..., 1])

                image = np.full((1, 1, 100), 0, dtype=dtype)
                aug = iaa.Add(iap.Uniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
                assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])

                image = np.full((50, 1, 3), 0, dtype=dtype)
                aug = iaa.Add(iap.DiscreteUniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
                assert np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
                assert np.allclose(image_aug[..., 0], image_aug[..., 1])

                image = np.full((1, 1, 100), 0, dtype=dtype)
                aug = iaa.Add(iap.DiscreteUniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
                assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])

    def test_pickleable(self):
        aug = iaa.Add((0, 50), per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10)


class TestAddElementwise(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _aug = iaa.AddElementwise(value="test")
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _aug = iaa.AddElementwise(value=1, per_channel="test")
        except Exception:
            got_exception = True
        assert got_exception

    def test_add_zero(self):
        # no add, shouldnt change anything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.AddElementwise(value=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

    def test_add_one(self):
        # add > 0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.AddElementwise(value=1)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images + 1
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [images_list[0] + 1]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images + 1
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [images_list[0] + 1]
        assert array_equal_lists(observed, expected)

    def test_add_minus_one(self):
        # add < 0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.AddElementwise(value=-1)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images - 1
        assert np.array_equal(observed, expected)

        observed = aug.augment_images(images_list)
        expected = [images_list[0] - 1]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images - 1
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [images_list[0] - 1]
        assert array_equal_lists(observed, expected)

    def test_uint8_every_possible_value(self):
        # uint8, every possible addition for base value 127
        for value_type in [int]:
            for per_channel in [False, True]:
                for value in np.arange(-255, 255+1):
                    aug = iaa.AddElementwise(value=value_type(value), per_channel=per_channel)
                    expected = np.clip(127 + value_type(value), 0, 255)

                    img = np.full((1, 1), 127, dtype=np.uint8)
                    img_aug = aug.augment_image(img)
                    assert img_aug.item(0) == expected

                    img = np.full((1, 1, 3), 127, dtype=np.uint8)
                    img_aug = aug.augment_image(img)
                    assert np.all(img_aug == expected)

    def test_stochastic_parameters_as_value(self):
        # test other parameters
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

        aug = iaa.AddElementwise(value=iap.DiscreteUniform(1, 10))
        observed = aug.augment_images(images)
        assert np.min(observed) >= 100 + 1
        assert np.max(observed) <= 100 + 10

        aug = iaa.AddElementwise(value=iap.Uniform(1, 10))
        observed = aug.augment_images(images)
        assert np.min(observed) >= 100 + 1
        assert np.max(observed) <= 100 + 10

        aug = iaa.AddElementwise(value=iap.Clip(iap.Normal(1, 1), -3, 3))
        observed = aug.augment_images(images)
        assert np.min(observed) >= 100 - 3
        assert np.max(observed) <= 100 + 3

        aug = iaa.AddElementwise(value=iap.Discretize(iap.Clip(iap.Normal(1, 1), -3, 3)))
        observed = aug.augment_images(images)
        assert np.min(observed) >= 100 - 3
        assert np.max(observed) <= 100 + 3

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.AddElementwise(value=1)
        aug_det = iaa.AddElementwise(value=1).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test_tuple_as_value(self):
        # varying values
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

        aug = iaa.AddElementwise(value=(0, 10))
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
        assert nb_changed_aug >= int(nb_iterations * 0.7)
        assert nb_changed_aug_det == 0

    def test_samples_change_by_spatial_location(self):
        # values should change between pixels
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

        aug = iaa.AddElementwise(value=(-50, 50))

        nb_same = 0
        nb_different = 0
        nb_iterations = 1000
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(images)
            observed_aug_flat = observed_aug.flatten()
            last = None
            for j in sm.xrange(observed_aug_flat.size):
                if last is not None:
                    v = observed_aug_flat[j]
                    if v - 0.0001 <= last <= v + 0.0001:
                        nb_same += 1
                    else:
                        nb_different += 1
                last = observed_aug_flat[j]
        assert nb_different > 0.9 * (nb_different + nb_same)

    def test_per_channel(self):
        # test channelwise
        aug = iaa.AddElementwise(value=iap.Choice([0, 1]), per_channel=True)
        observed = aug.augment_image(np.zeros((100, 100, 3), dtype=np.uint8))
        sums = np.sum(observed, axis=2)
        values = np.unique(sums)
        assert all([(value in values) for value in [0, 1, 2, 3]])

    def test_per_channel_with_probability(self):
        # test channelwise with probability
        aug = iaa.AddElementwise(value=iap.Choice([0, 1]), per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(400):
            observed = aug.augment_image(np.zeros((20, 20, 3), dtype=np.uint8))
            sums = np.sum(observed, axis=2)
            values = np.unique(sums)
            all_values_found = all([(value in values) for value in [0, 1, 2, 3]])
            if all_values_found:
                seen[0] += 1
            else:
                seen[1] += 1
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

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
                aug = iaa.AddElementwise(1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

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
                aug = iaa.AddElementwise(1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.AddElementwise(value=1, per_channel=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert params[0].value == 1
        assert params[1].value == 0

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.AddElementwise(value=10)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_bool(self):
        # bool
        image = np.zeros((3, 3), dtype=bool)
        aug = iaa.AddElementwise(value=1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.AddElementwise(value=1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.AddElementwise(value=-1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.AddElementwise(value=-2)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

    def test_other_dtypes_uint_int(self):
        # uint, int
        for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            image = np.full((3, 3), min_value, dtype=dtype)
            aug = iaa.AddElementwise(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == min_value + 1)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.AddElementwise(11)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == min_value + 21)

            image = np.full((3, 3), max_value - 2, dtype=dtype)
            aug = iaa.AddElementwise(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == max_value - 1)

            image = np.full((3, 3), max_value - 1, dtype=dtype)
            aug = iaa.AddElementwise(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == max_value)

            image = np.full((3, 3), max_value - 1, dtype=dtype)
            aug = iaa.AddElementwise(2)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == max_value)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.AddElementwise(-9)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == min_value + 1)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.AddElementwise(-10)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == min_value)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.AddElementwise(-11)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == min_value)

            for _ in sm.xrange(10):
                image = np.full((5, 5, 3), 20, dtype=dtype)
                aug = iaa.AddElementwise(iap.Uniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                assert len(np.unique(image_aug)) > 1
                assert np.all(image_aug[..., 0] == image_aug[..., 1])

                image = np.full((1, 1, 100), 20, dtype=dtype)
                aug = iaa.AddElementwise(iap.Uniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                assert len(np.unique(image_aug)) > 1

                image = np.full((5, 5, 3), 20, dtype=dtype)
                aug = iaa.AddElementwise(iap.DiscreteUniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                assert len(np.unique(image_aug)) > 1
                assert np.all(image_aug[..., 0] == image_aug[..., 1])

                image = np.full((1, 1, 100), 20, dtype=dtype)
                aug = iaa.AddElementwise(iap.DiscreteUniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                assert len(np.unique(image_aug)) > 1

    def test_other_dtypes_float(self):
        # float
        for dtype in [np.float16, np.float32]:
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            if dtype == np.float16:
                atol = 1e-3 * max_value
            else:
                atol = 1e-9 * max_value
            _allclose = functools.partial(np.allclose, atol=atol, rtol=0)

            image = np.full((3, 3), min_value, dtype=dtype)
            aug = iaa.AddElementwise(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, min_value + 1)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.AddElementwise(11)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, min_value + 21)

            image = np.full((3, 3), max_value - 2, dtype=dtype)
            aug = iaa.AddElementwise(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, max_value - 1)

            image = np.full((3, 3), max_value - 1, dtype=dtype)
            aug = iaa.AddElementwise(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, max_value)

            image = np.full((3, 3), max_value - 1, dtype=dtype)
            aug = iaa.AddElementwise(2)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, max_value)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.AddElementwise(-9)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, min_value + 1)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.AddElementwise(-10)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, min_value)

            image = np.full((3, 3), min_value + 10, dtype=dtype)
            aug = iaa.AddElementwise(-11)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, min_value)

            for _ in sm.xrange(10):
                image = np.full((50, 1, 3), 0, dtype=dtype)
                aug = iaa.AddElementwise(iap.Uniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
                assert not np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
                assert np.allclose(image_aug[..., 0], image_aug[..., 1])

                image = np.full((1, 1, 100), 0, dtype=dtype)
                aug = iaa.AddElementwise(iap.Uniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
                assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])

                image = np.full((50, 1, 3), 0, dtype=dtype)
                aug = iaa.AddElementwise(iap.DiscreteUniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
                assert not np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
                assert np.allclose(image_aug[..., 0], image_aug[..., 1])

                image = np.full((1, 1, 100), 0, dtype=dtype)
                aug = iaa.AddElementwise(iap.DiscreteUniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
                assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])

    def test_pickleable(self):
        aug = iaa.AddElementwise((0, 50), per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=2)


class AdditiveGaussianNoise(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_loc_zero_scale_zero(self):
        # no noise, shouldnt change anything
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128
        images = np.array([base_img])

        aug = iaa.AdditiveGaussianNoise(loc=0, scale=0)

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

    def test_loc_zero_scale_nonzero(self):
        # zero-centered noise
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128
        images = np.array([base_img])
        images_list = [base_img]
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.2 * 255)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        assert not np.array_equal(observed, images)

        observed = aug_det.augment_images(images)
        assert not np.array_equal(observed, images)

        observed = aug.augment_images(images_list)
        assert not array_equal_lists(observed, images_list)

        observed = aug_det.augment_images(images_list)
        assert not array_equal_lists(observed, images_list)

        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints)

        observed = aug_det.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints)

    def test_std_dev_of_added_noise_matches_scale(self):
        # std correct?
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

        aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.2 * 255)
        images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
        nb_iterations = 1000
        values = []
        for i in sm.xrange(nb_iterations):
            images_aug = aug.augment_images(images)
            values.append(images_aug[0, 0, 0, 0])
        values = np.array(values)
        assert np.min(values) == 0
        assert 0.1 < np.std(values) / 255.0 < 0.4

    def test_nonzero_loc(self):
        # non-zero loc
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

        aug = iaa.AdditiveGaussianNoise(loc=0.25 * 255, scale=0.01 * 255)
        images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
        nb_iterations = 1000
        values = []
        for i in sm.xrange(nb_iterations):
            images_aug = aug.augment_images(images)
            values.append(images_aug[0, 0, 0, 0] - 128)
        values = np.array(values)
        assert 54 < np.average(values) < 74 # loc=0.25 should be around 255*0.25=64 average

    def test_tuple_as_loc(self):
        # varying locs
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

        aug = iaa.AdditiveGaussianNoise(loc=(0, 0.5 * 255), scale=0.0001 * 255)
        aug_det = aug.to_deterministic()
        images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
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
        assert nb_changed_aug >= int(nb_iterations * 0.95)
        assert nb_changed_aug_det == 0

    def test_stochastic_parameter_as_loc(self):
        # varying locs by stochastic param
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

        aug = iaa.AdditiveGaussianNoise(loc=iap.Choice([-20, 20]), scale=0.0001 * 255)
        images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
        seen = [0, 0]
        for i in sm.xrange(200):
            observed = aug.augment_images(images)
            mean = np.mean(observed)
            diff_m20 = abs(mean - (128-20))
            diff_p20 = abs(mean - (128+20))
            if diff_m20 <= 1:
                seen[0] += 1
            elif diff_p20 <= 1:
                seen[1] += 1
            else:
                assert False
        assert 75 < seen[0] < 125
        assert 75 < seen[1] < 125

    def test_tuple_as_scale(self):
        # varying stds
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

        aug = iaa.AdditiveGaussianNoise(loc=0, scale=(0.01 * 255, 0.2 * 255))
        aug_det = aug.to_deterministic()
        images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
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
        assert nb_changed_aug >= int(nb_iterations * 0.95)
        assert nb_changed_aug_det == 0

    def test_stochastic_parameter_as_scale(self):
        # varying stds by stochastic param
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

        aug = iaa.AdditiveGaussianNoise(loc=0, scale=iap.Choice([1, 20]))
        images = np.ones((1, 20, 20, 1), dtype=np.uint8) * 128
        seen = [0, 0, 0]
        for i in sm.xrange(200):
            observed = aug.augment_images(images)
            std = np.std(observed.astype(np.int32) - 128)
            diff_1 = abs(std - 1)
            diff_20 = abs(std - 20)
            if diff_1 <= 2:
                seen[0] += 1
            elif diff_20 <= 5:
                seen[1] += 1
            else:
                seen[2] += 1
        assert seen[2] <= 5
        assert 75 < seen[0] < 125
        assert 75 < seen[1] < 125

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _ = iaa.AdditiveGaussianNoise(loc="test")
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _ = iaa.AdditiveGaussianNoise(scale="test")
        except Exception:
            got_exception = True
        assert got_exception

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

        aug = iaa.AdditiveGaussianNoise(loc=0.5, scale=10)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_pickleable(self):
        aug = iaa.AdditiveGaussianNoise(scale=(0.1, 10), per_channel=True,
                                        seed=1)
        runtest_pickleable_uint8_img(aug, iterations=2)


class TestCutout(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___defaults(self):
        aug = iaa.Cutout()
        assert aug.nb_iterations.value == 1
        assert isinstance(aug.position[0], iap.Uniform)
        assert isinstance(aug.position[1], iap.Uniform)
        assert np.isclose(aug.size.value, 0.2)
        assert aug.squared.value == 1
        assert aug.fill_mode.value == "constant"
        assert aug.cval.value == 128
        assert aug.fill_per_channel.value == 0

    def test___init___custom(self):
        aug = iaa.Cutout(
            nb_iterations=1,
            position=(0.5, 0.5),
            size=0.1,
            squared=0.6,
            fill_mode=["gaussian", "constant"],
            cval=(0, 255),
            fill_per_channel=0.5
        )
        assert aug.nb_iterations.value == 1
        assert np.isclose(aug.position[0].value, 0.5)
        assert np.isclose(aug.position[1].value, 0.5)
        assert np.isclose(aug.size.value, 0.1)
        assert np.isclose(aug.squared.p.value, 0.6)
        assert aug.fill_mode.a == ["gaussian", "constant"]
        assert np.isclose(aug.cval.a.value, 0)
        assert np.isclose(aug.cval.b.value, 255)
        assert np.isclose(aug.fill_per_channel.p.value, 0.5)

    def test___init___fill_mode_is_stochastic_param(self):
        param = iap.Deterministic("constant")
        aug = iaa.Cutout(fill_mode=param)
        assert aug.fill_mode is param

    @mock.patch("imgaug.augmenters.arithmetic.cutout_")
    def test_mocked__squared_false(self, mock_apply):
        aug = iaa.Cutout(nb_iterations=2,
                         position=(0.5, 0.6),
                         size=iap.DeterministicList([0.1, 0.2]),
                         squared=False,
                         fill_mode="gaussian",
                         cval=1,
                         fill_per_channel=True)
        image = np.zeros((10, 30, 3), dtype=np.uint8)

        # dont return image itself, otherwise the loop below will fail
        # at its second iteration as the method is expected to handle
        # internally a copy of the image and not the image itself
        mock_apply.return_value = np.copy(image)

        _ = aug(image=image)

        assert mock_apply.call_count == 2

        for call_idx in np.arange(2):
            args = mock_apply.call_args_list[call_idx][0]
            kwargs = mock_apply.call_args_list[call_idx][1]
            assert args[0] is not image
            assert np.array_equal(args[0], image)
            assert np.isclose(kwargs["x1"], 0.5*30 - 0.5 * (0.2*30))
            assert np.isclose(kwargs["y1"], 0.6*10 - 0.5 * (0.1*10))
            assert np.isclose(kwargs["x2"], 0.5*30 + 0.5 * (0.2*30))
            assert np.isclose(kwargs["y2"], 0.6*10 + 0.5 * (0.1*10))
            assert kwargs["fill_mode"] == "gaussian"
            assert np.array_equal(kwargs["cval"], [1, 1, 1])
            assert np.isclose(kwargs["fill_per_channel"], 1.0)
            assert isinstance(kwargs["seed"], iarandom.RNG)

    @mock.patch("imgaug.augmenters.arithmetic.cutout_")
    def test_mocked__squared_true(self, mock_apply):
        aug = iaa.Cutout(nb_iterations=2,
                         position=(0.5, 0.6),
                         size=iap.DeterministicList([0.1, 0.2]),
                         squared=True,
                         fill_mode="gaussian",
                         cval=1,
                         fill_per_channel=True)
        image = np.zeros((10, 30, 3), dtype=np.uint8)

        # dont return image itself, otherwise the loop below will fail
        # at its second iteration as the method is expected to handle
        # internally a copy of the image and not the image itself
        mock_apply.return_value = np.copy(image)

        _ = aug(image=image)

        assert mock_apply.call_count == 2

        for call_idx in np.arange(2):
            args = mock_apply.call_args_list[call_idx][0]
            kwargs = mock_apply.call_args_list[call_idx][1]
            assert args[0] is not image
            assert np.array_equal(args[0], image)
            assert np.isclose(kwargs["x1"], 0.5*30 - 0.5 * (0.1*10))
            assert np.isclose(kwargs["y1"], 0.6*10 - 0.5 * (0.1*10))
            assert np.isclose(kwargs["x2"], 0.5*30 + 0.5 * (0.1*10))
            assert np.isclose(kwargs["y2"], 0.6*10 + 0.5 * (0.1*10))
            assert kwargs["fill_mode"] == "gaussian"
            assert np.array_equal(kwargs["cval"], [1, 1, 1])
            assert np.isclose(kwargs["fill_per_channel"], 1.0)
            assert isinstance(kwargs["seed"], iarandom.RNG)

    def test_simple_image(self):
        aug = iaa.Cutout(nb_iterations=2,
                         position=(
                             iap.DeterministicList([0.2, 0.8]),
                             iap.DeterministicList([0.2, 0.8])
                         ),
                         size=0.2,
                         fill_mode="constant",
                         cval=iap.DeterministicList([0, 0, 0, 1, 1, 1]))
        image = np.full((100, 100, 3), 255, dtype=np.uint8)

        for _ in np.arange(3):
            images_aug = aug(images=[image, image])
            for image_aug in images_aug:
                values = np.unique(image_aug)
                assert len(values) == 3
                assert 0 in values
                assert 1 in values
                assert 255 in values

    def test_batch_contains_only_non_image_data(self):
        aug = iaa.Cutout()
        segmap_arr = np.ones((3, 3, 1), dtype=np.int32)
        segmap = ia.SegmentationMapsOnImage(segmap_arr, shape=(3, 3, 3))
        segmap_aug = aug.augment_segmentation_maps(segmap)
        assert np.array_equal(segmap.get_arr(), segmap_aug.get_arr())

    def test_sampling_when_position_is_stochastic_parameter(self):
        # sampling of position works slightly differently when it is a single
        # parameter instead of tuple (paramX, paramY), so we have an extra
        # test for that situation here
        param = iap.DeterministicList([0.5, 0.6])
        aug = iaa.Cutout(position=param)
        samples = aug._draw_samples([
            np.zeros((3, 3, 3), dtype=np.uint8),
            np.zeros((3, 3, 3), dtype=np.uint8)
        ], iarandom.RNG(0))
        assert np.allclose(samples.pos_x, [0.5, 0.5])
        assert np.allclose(samples.pos_y, [0.6, 0.6])

    def test_by_comparison_to_official_implementation(self):
        image = np.ones((10, 8, 2), dtype=np.uint8)
        aug = iaa.Cutout(1, position="uniform", size=0.2, squared=True,
                         cval=0)
        aug_official = _CutoutOfficial(n_holes=1, length=int(10*0.2))

        dropped = np.zeros((10, 8, 2), dtype=np.int32)
        dropped_official = np.copy(dropped)
        height = np.zeros((10, 8, 2), dtype=np.int32)
        width = np.copy(height)
        height_official = np.copy(height)
        width_official = np.copy(width)

        nb_iterations = 3 * 1000

        images_aug = aug(images=[image] * nb_iterations)
        for image_aug in images_aug:
            image_aug_off = aug_official(image)

            mask = (image_aug == 0)
            mask_off = (image_aug_off == 0)

            dropped += mask
            dropped_official += mask_off

            ydrop = np.max(mask, axis=(2, 1))
            xdrop = np.max(mask, axis=(2, 0))
            wx = np.where(xdrop)
            wy = np.where(ydrop)
            x1 = wx[0][0]
            x2 = wx[0][-1]
            y1 = wy[0][0]
            y2 = wy[0][-1]

            ydrop_off = np.max(mask_off, axis=(2, 1))
            xdrop_off = np.max(mask_off, axis=(2, 0))
            wx_off = np.where(xdrop_off)
            wy_off = np.where(ydrop_off)
            x1_off = wx_off[0][0]
            x2_off = wx_off[0][-1]
            y1_off = wy_off[0][0]
            y2_off = wy_off[0][-1]

            height += (
                np.full(height.shape, 1 + (y2 - y1), dtype=np.int32)
                * mask)
            width += (
                np.full(width.shape, 1 + (x2 - x1), dtype=np.int32)
                * mask)
            height_official += (
                np.full(height_official.shape, 1 + (y2_off - y1_off),
                        dtype=np.int32)
                * mask_off)
            width_official += (
                np.full(width_official.shape, 1 + (x2_off - x1_off),
                        dtype=np.int32)
                * mask_off)

        dropped_prob = dropped / nb_iterations
        dropped_prob_off = dropped_official / nb_iterations
        height_avg = height / (dropped + 1e-4)
        height_avg_off = height_official / (dropped_official + 1e-4)
        width_avg = width / (dropped + 1e-4)
        width_avg_off = width_official / (dropped_official + 1e-4)

        prob_max_diff = np.max(np.abs(dropped_prob - dropped_prob_off))
        height_avg_max_diff = np.max(np.abs(height_avg - height_avg_off))
        width_avg_max_diff = np.max(np.abs(width_avg - width_avg_off))

        assert prob_max_diff < 0.04
        assert height_avg_max_diff < 0.3
        assert width_avg_max_diff < 0.3

    def test_determinism(self):
        aug = iaa.Cutout(nb_iterations=(1, 3),
                         size=(0.1, 0.2),
                         fill_mode=["gaussian", "constant"],
                         cval=(0, 255))
        image = np.mod(
            np.arange(100*100*3), 256
        ).reshape((100, 100, 3)).astype(np.uint8)

        sums = []
        for _ in np.arange(10):
            aug_det = aug.to_deterministic()
            image_aug1 = aug_det(image=image)
            image_aug2 = aug_det(image=image)
            assert np.array_equal(image_aug1, image_aug2)
            sums.append(np.sum(image_aug1))
        assert len(np.unique(sums)) > 1

    def test_get_parameters(self):
        aug = iaa.Cutout(
            nb_iterations=1,
            position=(0.5, 0.5),
            size=0.1,
            squared=0.6,
            fill_mode=["gaussian", "constant"],
            cval=(0, 255),
            fill_per_channel=0.5
        )
        params = aug.get_parameters()
        assert params[0] is aug.nb_iterations
        assert params[1] is aug.position
        assert params[2] is aug.size
        assert params[3] is aug.squared
        assert params[4] is aug.fill_mode
        assert params[5] is aug.cval
        assert params[6] is aug.fill_per_channel


# this is mostly copy-pasted cutout code from
# https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
# we use this to compare our implementation against
# we changed some pytorch to numpy stuff
class _CutoutOfficial(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of
            it.
        """
        # h = img.size(1)
        # w = img.size(2)
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            # note that in the paper they normalize to 0-mean,
            # i.e. 0 here is actually not black but grayish pixels
            mask[y1: y2, x1: x2] = 0

        # mask = torch.from_numpy(mask)
        # mask = mask.expand_as(img)
        if img.ndim != 2:
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, img.shape[-1]))
        img = img * mask

        return img


class TestDropout(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_zero(self):
        # no dropout, shouldnt change anything
        base_img = np.ones((512, 512, 1), dtype=np.uint8) * 255
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Dropout(p=0)
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        # 100% dropout, should drop everything
        aug = iaa.Dropout(p=1.0)
        observed = aug.augment_images(images)
        expected = np.zeros((1, 512, 512, 1), dtype=np.uint8)
        assert np.array_equal(observed, expected)

        observed = aug.augment_images(images_list)
        expected = [np.zeros((512, 512, 1), dtype=np.uint8)]
        assert array_equal_lists(observed, expected)

    def test_p_is_50_percent(self):
        # 50% dropout
        base_img = np.ones((512, 512, 1), dtype=np.uint8) * 255
        images = np.array([base_img])
        images_list = [base_img]
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.Dropout(p=0.5)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        assert not np.array_equal(observed, images)
        percent_nonzero = len(observed.flatten().nonzero()[0]) \
                          / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
        assert 0.35 <= (1 - percent_nonzero) <= 0.65

        observed = aug_det.augment_images(images)
        assert not np.array_equal(observed, images)
        percent_nonzero = len(observed.flatten().nonzero()[0]) \
                          / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
        assert 0.35 <= (1 - percent_nonzero) <= 0.65

        observed = aug.augment_images(images_list)
        assert not array_equal_lists(observed, images_list)
        percent_nonzero = len(observed[0].flatten().nonzero()[0]) \
                          / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
        assert 0.35 <= (1 - percent_nonzero) <= 0.65

        observed = aug_det.augment_images(images_list)
        assert not array_equal_lists(observed, images_list)
        percent_nonzero = len(observed[0].flatten().nonzero()[0]) \
                          / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
        assert 0.35 <= (1 - percent_nonzero) <= 0.65

        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints)

        observed = aug_det.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints)

    def test_tuple_as_p(self):
        # varying p
        aug = iaa.Dropout(p=(0.0, 1.0))
        aug_det = aug.to_deterministic()
        images = np.ones((1, 8, 8, 1), dtype=np.uint8) * 255
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
        assert nb_changed_aug >= int(nb_iterations * 0.95)
        assert nb_changed_aug_det == 0

    def test_list_as_p(self):
        aug = iaa.Dropout(p=[0.0, 0.5, 1.0])
        images = np.ones((1, 20, 20, 1), dtype=np.uint8) * 255
        nb_seen = [0, 0, 0, 0]
        nb_iterations = 1000
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(images)

            n_dropped = np.sum(observed_aug == 0)
            p_observed = n_dropped / observed_aug.size
            if 0 <= p_observed <= 0.01:
                nb_seen[0] += 1
            elif 0.5 - 0.05 <= p_observed <= 0.5 + 0.05:
                nb_seen[1] += 1
            elif 1.0-0.01 <= p_observed <= 1.0:
                nb_seen[2] += 1
            else:
                nb_seen[3] += 1
        assert np.allclose(nb_seen[0:3], nb_iterations*0.33, rtol=0, atol=75)
        assert nb_seen[3] < 30

    def test_stochastic_parameter_as_p(self):
        # varying p by stochastic parameter
        aug = iaa.Dropout(p=iap.Binomial(1-iap.Choice([0.0, 0.5])))
        images = np.ones((1, 20, 20, 1), dtype=np.uint8) * 255
        seen = [0, 0, 0]
        for i in sm.xrange(400):
            observed = aug.augment_images(images)
            p = np.mean(observed == 0)
            if 0.4 < p < 0.6:
                seen[0] += 1
            elif p < 0.1:
                seen[1] += 1
            else:
                seen[2] += 1
        assert seen[2] <= 10
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

    def test___init___bad_datatypes(self):
        # test exception for wrong parameter datatype
        got_exception = False
        try:
            _aug = iaa.Dropout(p="test")
        except Exception:
            got_exception = True
        assert got_exception

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.Dropout(p=1.0)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_pickleable(self):
        aug = iaa.Dropout(p=0.5, per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3)


class TestCoarseDropout(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_zero(self):
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 100
        aug = iaa.CoarseDropout(p=0, size_px=4, size_percent=None, per_channel=False, min_size=4)
        observed = aug.augment_image(base_img)
        expected = base_img
        assert np.array_equal(observed, expected)

    def test_p_is_one(self):
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 100
        aug = iaa.CoarseDropout(p=1.0, size_px=4, size_percent=None, per_channel=False, min_size=4)
        observed = aug.augment_image(base_img)
        expected = np.zeros_like(base_img)
        assert np.array_equal(observed, expected)

    def test_p_is_50_percent(self):
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 100
        aug = iaa.CoarseDropout(p=0.5, size_px=1, size_percent=None, per_channel=False, min_size=1)
        averages = []
        for _ in sm.xrange(50):
            observed = aug.augment_image(base_img)
            averages.append(np.average(observed))
        assert all([v in [0, 100] for v in averages])
        assert 50 - 20 < np.average(averages) < 50 + 20

    def test_size_percent(self):
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 100
        aug = iaa.CoarseDropout(p=0.5, size_px=None, size_percent=0.001, per_channel=False, min_size=1)
        averages = []
        for _ in sm.xrange(50):
            observed = aug.augment_image(base_img)
            averages.append(np.average(observed))
        assert all([v in [0, 100] for v in averages])
        assert 50 - 20 < np.average(averages) < 50 + 20

    def test_per_channel(self):
        aug = iaa.CoarseDropout(p=0.5, size_px=1, size_percent=None, per_channel=True, min_size=1)
        base_img = np.ones((4, 4, 3), dtype=np.uint8) * 100
        found = False
        for _ in sm.xrange(100):
            observed = aug.augment_image(base_img)
            avgs = np.average(observed, axis=(0, 1))
            if len(set(avgs)) >= 2:
                found = True
                break
        assert found

    def test_stochastic_parameter_as_p(self):
        # varying p by stochastic parameter
        aug = iaa.CoarseDropout(p=iap.Binomial(1-iap.Choice([0.0, 0.5])), size_px=50)
        images = np.ones((1, 100, 100, 1), dtype=np.uint8) * 255
        seen = [0, 0, 0]
        for i in sm.xrange(400):
            observed = aug.augment_images(images)
            p = np.mean(observed == 0)
            if 0.4 < p < 0.6:
                seen[0] += 1
            elif p < 0.1:
                seen[1] += 1
            else:
                seen[2] += 1
        assert seen[2] <= 10
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

    def test___init___bad_datatypes(self):
        # test exception for bad parameters
        got_exception = False
        try:
            _ = iaa.CoarseDropout(p="test")
        except Exception:
            got_exception = True
        assert got_exception

    def test___init___size_px_and_size_percent_both_none(self):
        got_exception = False
        try:
            _ = iaa.CoarseDropout(p=0.5, size_px=None, size_percent=None)
        except Exception:
            got_exception = True
        assert got_exception

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.CoarseDropout(p=1.0, size_px=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_pickleable(self):
        aug = iaa.CoarseDropout(p=0.5, size_px=10, per_channel=True,
                                seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10, shape=(40, 40, 3))


class TestDropout2d(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___defaults(self):
        aug = iaa.Dropout2d(p=0)
        assert isinstance(aug.p, iap.Binomial)
        assert np.isclose(aug.p.p.value, 1.0)
        assert aug.nb_keep_channels == 1

    def test___init___p_is_float(self):
        aug = iaa.Dropout2d(p=0.7)
        assert isinstance(aug.p, iap.Binomial)
        assert np.isclose(aug.p.p.value, 0.3)
        assert aug.nb_keep_channels == 1

    def test___init___nb_keep_channels_is_int(self):
        aug = iaa.Dropout2d(p=0, nb_keep_channels=2)
        assert isinstance(aug.p, iap.Binomial)
        assert np.isclose(aug.p.p.value, 1.0)
        assert aug.nb_keep_channels == 2

    def test_no_images_in_batch(self):
        aug = iaa.Dropout2d(p=0.0, nb_keep_channels=0)
        heatmaps = np.float32([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        heatmaps = ia.HeatmapsOnImage(heatmaps, shape=(2, 2, 3))

        heatmaps_aug = aug(heatmaps=heatmaps)

        assert np.allclose(heatmaps_aug.arr_0to1, heatmaps.arr_0to1)

    def test_p_is_1(self):
        image = np.full((1, 2, 3), 255, dtype=np.uint8)
        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=0)

        image_aug = aug(image=image)

        assert image_aug.shape == image.shape
        assert image_aug.dtype.name == image.dtype.name
        assert np.sum(image_aug) == 0

    def test_p_is_1_heatmaps(self):
        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=0)
        arr = np.float32([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        hm = ia.HeatmapsOnImage(arr, shape=(2, 2, 3))

        heatmaps_aug = aug(heatmaps=hm)

        assert np.allclose(heatmaps_aug.arr_0to1, 0.0)

    def test_p_is_1_segmentation_maps(self):
        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=0)
        arr = np.int32([
            [0, 1],
            [0, 1]
        ])
        segmaps = ia.SegmentationMapsOnImage(arr, shape=(2, 2, 3))

        segmaps_aug = aug(segmentation_maps=segmaps)

        assert np.allclose(segmaps_aug.arr, 0.0)

    def test_p_is_1_cbaois(self):
        cbaois = [
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(2, 2, 3)),
            ia.BoundingBoxesOnImage([ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)],
                                    shape=(2, 2, 3)),
            ia.PolygonsOnImage([ia.Polygon([(0, 0), (1, 0), (1, 1)])],
                               shape=(2, 2, 3)),
            ia.LineStringsOnImage([ia.LineString([(0, 0), (1, 0)])],
                                  shape=(2, 2, 3))
        ]

        cbaoi_names = ["keypoints", "bounding_boxes", "polygons",
                       "line_strings"]

        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=0)
        for name, cbaoi in zip(cbaoi_names, cbaois):
            with self.subTest(datatype=name):
                cbaoi_aug = aug(**{name: cbaoi})

                assert cbaoi_aug.shape == (2, 2, 3)
                assert cbaoi_aug.items == []

    def test_p_is_1_heatmaps__keep_one_channel(self):
        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=1)
        arr = np.float32([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        hm = ia.HeatmapsOnImage(arr, shape=(2, 2, 3))

        heatmaps_aug = aug(heatmaps=hm)

        assert np.allclose(heatmaps_aug.arr_0to1, hm.arr_0to1)

    def test_p_is_1_segmentation_maps__keep_one_channel(self):
        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=1)
        arr = np.int32([
            [0, 1],
            [0, 1]
        ])
        segmaps = ia.SegmentationMapsOnImage(arr, shape=(2, 2, 3))

        segmaps_aug = aug(segmentation_maps=segmaps)

        assert np.allclose(segmaps_aug.arr, segmaps.arr)

    def test_p_is_1_cbaois__keep_one_channel(self):
        cbaois = [
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(2, 2, 3)),
            ia.BoundingBoxesOnImage([ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)],
                                    shape=(2, 2, 3)),
            ia.PolygonsOnImage([ia.Polygon([(0, 0), (1, 0), (1, 1)])],
                               shape=(2, 2, 3)),
            ia.LineStringsOnImage([ia.LineString([(0, 0), (1, 0)])],
                                  shape=(2, 2, 3))
        ]

        cbaoi_names = ["keypoints", "bounding_boxes", "polygons",
                       "line_strings"]

        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=1)
        for name, cbaoi in zip(cbaoi_names, cbaois):
            with self.subTest(datatype=name):
                cbaoi_aug = aug(**{name: cbaoi})

                assert cbaoi_aug.shape == (2, 2, 3)
                assert np.allclose(
                    cbaoi_aug.items[0].coords,
                    cbaoi.items[0].coords
                )

    def test_p_is_0(self):
        image = np.full((1, 2, 3), 255, dtype=np.uint8)
        aug = iaa.Dropout2d(p=0.0, nb_keep_channels=0)

        image_aug = aug(image=image)

        assert image_aug.shape == image.shape
        assert image_aug.dtype.name == image.dtype.name
        assert np.array_equal(image_aug, image)

    def test_p_is_0_heatmaps(self):
        aug = iaa.Dropout2d(p=0.0, nb_keep_channels=0)
        arr = np.float32([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        hm = ia.HeatmapsOnImage(arr, shape=(2, 2, 3))

        heatmaps_aug = aug(heatmaps=hm)

        assert np.allclose(heatmaps_aug.arr_0to1, hm.arr_0to1)

    def test_p_is_0_segmentation_maps(self):
        aug = iaa.Dropout2d(p=0.0, nb_keep_channels=0)
        arr = np.int32([
            [0, 1],
            [0, 1]
        ])
        segmaps = ia.SegmentationMapsOnImage(arr, shape=(2, 2, 3))

        segmaps_aug = aug(segmentation_maps=segmaps)

        assert np.allclose(segmaps_aug.arr, segmaps.arr)

    def test_p_is_0_cbaois(self):
        cbaois = [
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(2, 2, 3)),
            ia.BoundingBoxesOnImage([ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)],
                                    shape=(2, 2, 3)),
            ia.PolygonsOnImage([ia.Polygon([(0, 0), (1, 0), (1, 1)])],
                               shape=(2, 2, 3)),
            ia.LineStringsOnImage([ia.LineString([(0, 0), (1, 0)])],
                                  shape=(2, 2, 3))
        ]

        cbaoi_names = ["keypoints", "bounding_boxes", "polygons",
                       "line_strings"]

        aug = iaa.Dropout2d(p=0.0, nb_keep_channels=0)
        for name, cbaoi in zip(cbaoi_names, cbaois):
            with self.subTest(datatype=name):
                cbaoi_aug = aug(**{name: cbaoi})

                assert cbaoi_aug.shape == (2, 2, 3)
                assert np.allclose(
                    cbaoi_aug.items[0].coords,
                    cbaoi.items[0].coords
                )

    def test_p_is_075(self):
        image = np.full((1, 1, 3000), 255, dtype=np.uint8)
        aug = iaa.Dropout2d(p=0.75, nb_keep_channels=0)

        image_aug = aug(image=image)

        nb_kept = np.sum(image_aug == 255)
        nb_dropped = image.shape[2] - nb_kept
        assert image_aug.shape == image.shape
        assert image_aug.dtype.name == image.dtype.name
        assert np.isclose(nb_dropped, image.shape[2]*0.75, atol=75)

    def test_force_nb_keep_channels(self):
        image = np.full((1, 1, 3), 255, dtype=np.uint8)
        images = np.array([image] * 1000)
        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=1)

        images_aug = aug(images=images)

        ids_kept = [np.nonzero(image[0, 0, :]) for image in images_aug]
        ids_kept_uq = np.unique(ids_kept)
        nb_kept = np.sum(images_aug == 255)
        nb_dropped = (len(images) * images.shape[3]) - nb_kept

        assert images_aug.shape == images.shape
        assert images_aug.dtype.name == images.dtype.name

        # on average, keep 1 of 3 channels
        # due to p=1.0 we expect to get exactly 2/3 dropped
        assert np.isclose(nb_dropped,
                          (len(images)*images.shape[3])*(2/3), atol=1)

        # every channel dropped at least once, i.e. which one is kept is random
        assert sorted(ids_kept_uq.tolist()) == [0, 1, 2]

    def test_some_images_below_nb_keep_channels(self):
        image_2c = np.full((1, 1, 2), 255, dtype=np.uint8)
        image_3c = np.full((1, 1, 3), 255, dtype=np.uint8)
        images = [image_2c if i % 2 == 0 else image_3c
                  for i in sm.xrange(100)]
        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=2)

        images_aug = aug(images=images)

        for i, image_aug in enumerate(images_aug):
            assert np.sum(image_aug == 255) == 2
            if i % 2 == 0:
                assert np.sum(image_aug == 0) == 0
            else:
                assert np.sum(image_aug == 0) == 1

    def test_all_images_below_nb_keep_channels(self):
        image = np.full((1, 1, 2), 255, dtype=np.uint8)
        images = np.array([image] * 100)
        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=3)

        images_aug = aug(images=images)

        nb_kept = np.sum(images_aug == 255)
        nb_dropped = (len(images) * images.shape[3]) - nb_kept
        assert nb_dropped == 0

    def test_get_parameters(self):
        aug = iaa.Dropout2d(p=0.7, nb_keep_channels=2)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Binomial)
        assert np.isclose(params[0].p.value, 0.3)
        assert params[1] == 2

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
                image = np.full(shape, 255, dtype=np.uint8)
                aug = iaa.Dropout2d(1.0, nb_keep_channels=0)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_other_dtypes_bool(self):
        image = np.full((1, 1, 10), 1, dtype=bool)
        aug = iaa.Dropout2d(p=1.0, nb_keep_channels=3)

        image_aug = aug(image=image)

        assert image_aug.shape == image.shape
        assert image_aug.dtype.name == "bool"
        assert np.sum(image_aug == 1) == 3
        assert np.sum(image_aug == 0) == 7

    def test_other_dtypes_uint_int(self):
        dts = ["uint8", "uint16", "uint32", "uint64",
               "int8", "int16", "int32", "int64"]

        for dt in dts:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dt)
            values = [min_value, int(center_value), max_value]

            for value in values:
                with self.subTest(dtype=dt, value=value):
                    image = np.full((1, 1, 10), value, dtype=dt)
                    aug = iaa.Dropout2d(p=1.0, nb_keep_channels=3)

                    image_aug = aug(image=image)

                    assert image_aug.shape == image.shape
                    assert image_aug.dtype.name == dt
                    if value == 0:
                        assert np.sum(image_aug == value) == 10
                    else:
                        assert np.sum(image_aug == value) == 3
                        assert np.sum(image_aug == 0) == 7

    def test_other_dtypes_float(self):
        dts = ["float16", "float32", "float64", "float128"]

        for dt in dts:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dt)
            values = [min_value, -10.0, center_value, 10.0, max_value]

            atol = 1e-3*max_value if dt == "float16" else 1e-9 * max_value
            _isclose = functools.partial(np.isclose, atol=atol, rtol=0)

            for value in values:
                with self.subTest(dtype=dt, value=value):
                    image = np.full((1, 1, 10), value, dtype=dt)
                    aug = iaa.Dropout2d(p=1.0, nb_keep_channels=3)

                    image_aug = aug(image=image)

                    assert image_aug.shape == image.shape
                    assert image_aug.dtype.name == dt
                    if _isclose(value, 0.0):
                        assert np.sum(_isclose(image_aug, value)) == 10
                    else:
                        assert (
                            np.sum(_isclose(image_aug, np.float128(value)))
                            == 3)
                        assert np.sum(image_aug == 0) == 7

    def test_pickleable(self):
        aug = iaa.Dropout2d(p=0.5, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3, shape=(1, 1, 50))


class TestTotalDropout(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___p(self):
        aug = iaa.TotalDropout(p=0)
        assert isinstance(aug.p, iap.Binomial)
        assert np.isclose(aug.p.p.value, 1.0)

    def test_p_is_1(self):
        image = np.full((1, 2, 3), 255, dtype=np.uint8)
        aug = iaa.TotalDropout(p=1.0)

        image_aug = aug(image=image)

        assert image_aug.shape == image.shape
        assert image_aug.dtype.name == image.dtype.name
        assert np.sum(image_aug) == 0

    def test_p_is_1_multiple_images_list(self):
        image = np.full((1, 2, 3), 255, dtype=np.uint8)
        images = [image, image, image]
        aug = iaa.TotalDropout(p=1.0)

        images_aug = aug(images=images)

        for image_aug, image_ in zip(images_aug, images):
            assert image_aug.shape == image_.shape
            assert image_aug.dtype.name == image_.dtype.name
            assert np.sum(image_aug) == 0

    def test_p_is_1_multiple_images_array(self):
        image = np.full((1, 2, 3), 255, dtype=np.uint8)
        images = np.array([image, image, image], dtype=np.uint8)
        aug = iaa.TotalDropout(p=1.0)

        images_aug = aug(images=images)

        assert images_aug.shape == images.shape
        assert images_aug.dtype.name == images.dtype.name
        assert np.sum(images_aug) == 0

    def test_p_is_1_heatmaps(self):
        aug = iaa.TotalDropout(p=1.0)
        arr = np.float32([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        hm = ia.HeatmapsOnImage(arr, shape=(2, 2, 3))

        heatmaps_aug = aug(heatmaps=hm)

        assert np.allclose(heatmaps_aug.arr_0to1, 0.0)

    def test_p_is_1_segmentation_maps(self):
        aug = iaa.TotalDropout(p=1.0)
        arr = np.int32([
            [0, 1],
            [0, 1]
        ])
        segmaps = ia.SegmentationMapsOnImage(arr, shape=(2, 2, 3))

        segmaps_aug = aug(segmentation_maps=segmaps)

        assert np.allclose(segmaps_aug.arr, 0.0)

    def test_p_is_1_cbaois(self):
        cbaois = [
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(2, 2, 3)),
            ia.BoundingBoxesOnImage([ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)],
                                    shape=(2, 2, 3)),
            ia.PolygonsOnImage([ia.Polygon([(0, 0), (1, 0), (1, 1)])],
                               shape=(2, 2, 3)),
            ia.LineStringsOnImage([ia.LineString([(0, 0), (1, 0)])],
                                  shape=(2, 2, 3))
        ]

        cbaoi_names = ["keypoints", "bounding_boxes", "polygons",
                       "line_strings"]

        aug = iaa.TotalDropout(p=1.0)
        for name, cbaoi in zip(cbaoi_names, cbaois):
            with self.subTest(datatype=name):
                cbaoi_aug = aug(**{name: cbaoi})

                assert cbaoi_aug.shape == (2, 2, 3)
                assert cbaoi_aug.items == []

    def test_p_is_0(self):
        image = np.full((1, 2, 3), 255, dtype=np.uint8)
        aug = iaa.TotalDropout(p=0.0)

        image_aug = aug(image=image)

        assert image_aug.shape == image.shape
        assert image_aug.dtype.name == image.dtype.name
        assert np.array_equal(image_aug, image)

    def test_p_is_0_multiple_images_list(self):
        image = np.full((1, 2, 3), 255, dtype=np.uint8)
        images = [image, image, image]
        aug = iaa.TotalDropout(p=0.0)

        images_aug = aug(images=images)

        for image_aug, image_ in zip(images_aug, images):
            assert image_aug.shape == image_.shape
            assert image_aug.dtype.name == image_.dtype.name
            assert np.array_equal(image_aug, image_)

    def test_p_is_0_multiple_images_array(self):
        image = np.full((1, 2, 3), 255, dtype=np.uint8)
        images = np.array([image, image, image], dtype=np.uint8)
        aug = iaa.TotalDropout(p=0.0)

        images_aug = aug(images=images)

        for image_aug, image_ in zip(images_aug, images):
            assert image_aug.shape == image_.shape
            assert image_aug.dtype.name == image_.dtype.name
            assert np.array_equal(image_aug, image_)

    def test_p_is_0_heatmaps(self):
        aug = iaa.TotalDropout(p=0.0)
        arr = np.float32([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        hm = ia.HeatmapsOnImage(arr, shape=(2, 2, 3))

        heatmaps_aug = aug(heatmaps=hm)

        assert np.allclose(heatmaps_aug.arr_0to1, hm.arr_0to1)

    def test_p_is_0_segmentation_maps(self):
        aug = iaa.TotalDropout(p=0.0)
        arr = np.int32([
            [0, 1],
            [0, 1]
        ])
        segmaps = ia.SegmentationMapsOnImage(arr, shape=(2, 2, 3))

        segmaps_aug = aug(segmentation_maps=segmaps)

        assert np.allclose(segmaps_aug.arr, segmaps.arr)

    def test_p_is_0_cbaois(self):
        cbaois = [
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(2, 2, 3)),
            ia.BoundingBoxesOnImage([ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)],
                                    shape=(2, 2, 3)),
            ia.PolygonsOnImage([ia.Polygon([(0, 0), (1, 0), (1, 1)])],
                               shape=(2, 2, 3)),
            ia.LineStringsOnImage([ia.LineString([(0, 0), (1, 0)])],
                                  shape=(2, 2, 3))
        ]

        cbaoi_names = ["keypoints", "bounding_boxes", "polygons",
                       "line_strings"]

        aug = iaa.TotalDropout(p=0.0)
        for name, cbaoi in zip(cbaoi_names, cbaois):
            with self.subTest(datatype=name):
                cbaoi_aug = aug(**{name: cbaoi})

                assert cbaoi_aug.shape == (2, 2, 3)
                assert np.allclose(
                    cbaoi_aug.items[0].coords,
                    cbaoi.items[0].coords
                )

    def test_p_is_075_multiple_images_list(self):
        images = [np.full((1, 1, 1), 255, dtype=np.uint8)] * 3000
        aug = iaa.TotalDropout(p=0.75)

        images_aug = aug(images=images)

        nb_kept = np.sum([np.sum(image_aug == 255) for image_aug in images_aug])
        nb_dropped = len(images) - nb_kept
        for image_aug in images_aug:
            assert image_aug.shape == images[0].shape
            assert image_aug.dtype.name == images[0].dtype.name
        assert np.isclose(nb_dropped, len(images)*0.75, atol=75)

    def test_p_is_075_multiple_images_array(self):
        images = np.full((3000, 1, 1, 1), 255, dtype=np.uint8)
        aug = iaa.TotalDropout(p=0.75)

        images_aug = aug(images=images)

        nb_kept = np.sum(images_aug == 255)
        nb_dropped = len(images) - nb_kept
        assert images_aug.shape == images.shape
        assert images_aug.dtype.name == images.dtype.name
        assert np.isclose(nb_dropped, len(images)*0.75, atol=75)

    def test_get_parameters(self):
        aug = iaa.TotalDropout(p=0.0)
        params = aug.get_parameters()
        assert params[0] is aug.p

    def test_unusual_channel_numbers(self):
        shapes = [
            (5, 1, 1, 4),
            (5, 1, 1, 5),
            (5, 1, 1, 512),
            (5, 1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                images = np.zeros(shape, dtype=np.uint8)
                aug = iaa.TotalDropout(1.0)

                images_aug = aug(images=images)

                assert np.all(images_aug == 0)
                assert images_aug.dtype.name == "uint8"
                assert images_aug.shape == shape

    def test_zero_sized_axes(self):
        shapes = [
            (5, 0, 0),
            (5, 0, 1),
            (5, 1, 0),
            (5, 0, 1, 0),
            (5, 1, 0, 0),
            (5, 0, 1, 1),
            (5, 1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                images = np.full(shape, 255, dtype=np.uint8)
                aug = iaa.TotalDropout(1.0)

                images_aug = aug(images=images)

                assert images_aug.dtype.name == "uint8"
                assert images_aug.shape == images.shape

    def test_other_dtypes_bool(self):
        image = np.full((1, 1, 10), 1, dtype=bool)
        aug = iaa.TotalDropout(p=1.0)

        image_aug = aug(image=image)

        assert image_aug.shape == image.shape
        assert image_aug.dtype.name == "bool"
        assert np.sum(image_aug == 1) == 0

    def test_other_dtypes_uint_int(self):
        dts = ["uint8", "uint16", "uint32", "uint64",
               "int8", "int16", "int32", "int64"]

        for dt in dts:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dt)
            values = [min_value, int(center_value), max_value]

            for value in values:
                for p in [1.0, 0.0]:
                    with self.subTest(dtype=dt, value=value, p=p):
                        images = np.full((5, 1, 1, 3), value, dtype=dt)
                        aug = iaa.TotalDropout(p=p)

                        images_aug = aug(images=images)

                        assert images_aug.shape == images.shape
                        assert images_aug.dtype.name == dt
                        if np.isclose(p, 1.0) or value == 0:
                            assert np.sum(images_aug == 0) == 5*3
                        else:
                            assert np.sum(images_aug == value) == 5*3

    def test_other_dtypes_float(self):
        dts = ["float16", "float32", "float64", "float128"]

        for dt in dts:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dt)
            values = [min_value, -10.0, center_value, 10.0, max_value]

            atol = 1e-3*max_value if dt == "float16" else 1e-9 * max_value
            _isclose = functools.partial(np.isclose, atol=atol, rtol=0)

            for value in values:
                for p in [1.0, 0.0]:
                    with self.subTest(dtype=dt, value=value, p=p):
                        images = np.full((5, 1, 1, 3), value, dtype=dt)
                        aug = iaa.TotalDropout(p=p)

                        images_aug = aug(images=images)

                        assert images_aug.shape == images.shape
                        assert images_aug.dtype.name == dt
                        if np.isclose(p, 1.0):
                            assert np.sum(_isclose(images_aug, 0.0)) == 5*3
                        else:
                            assert (
                                np.sum(_isclose(images_aug, np.float128(value)))
                                == 5*3)

    def test_pickleable(self):
        aug = iaa.TotalDropout(p=0.5, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=30, shape=(4, 4, 2))


class TestMultiply(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_mul_is_one(self):
        # no multiply, shouldnt change anything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Multiply(mul=1.0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

    def test_mul_is_above_one(self):
        # multiply >1.0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Multiply(mul=1.2)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
        assert array_equal_lists(observed, expected)

    def test_mul_is_below_one(self):
        # multiply <1.0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Multiply(mul=0.8)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
        assert array_equal_lists(observed, expected)

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.Multiply(mul=1.2)
        aug_det = iaa.Multiply(mul=1.2).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test_tuple_as_mul(self):
        # varying multiply factors
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

        aug = iaa.Multiply(mul=(0, 2.0))
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
        assert nb_changed_aug >= int(nb_iterations * 0.95)
        assert nb_changed_aug_det == 0

    def test_per_channel(self):
        aug = iaa.Multiply(mul=iap.Choice([0, 2]), per_channel=True)
        observed = aug.augment_image(np.ones((1, 1, 100), dtype=np.uint8))
        uq = np.unique(observed)
        assert observed.shape == (1, 1, 100)
        assert 0 in uq
        assert 2 in uq
        assert len(uq) == 2

    def test_per_channel_with_probability(self):
        # test channelwise with probability
        aug = iaa.Multiply(mul=iap.Choice([0, 2]), per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(400):
            observed = aug.augment_image(np.ones((1, 1, 20), dtype=np.uint8))
            assert observed.shape == (1, 1, 20)

            uq = np.unique(observed)
            per_channel = (len(uq) == 2)
            if per_channel:
                seen[0] += 1
            else:
                seen[1] += 1
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _ = iaa.Multiply(mul="test")
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _ = iaa.Multiply(mul=1, per_channel="test")
        except Exception:
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
                image = np.ones(shape, dtype=np.uint8)
                aug = iaa.Multiply(1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 2)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.ones(shape, dtype=np.uint8)
                aug = iaa.Multiply(2)

                image_aug = aug(image=image)

                assert np.all(image_aug == 2)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.Multiply(mul=1, per_channel=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert params[0].value == 1
        assert params[1].value == 0

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.Multiply(mul=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_bool(self):
        # bool
        image = np.zeros((3, 3), dtype=bool)
        aug = iaa.Multiply(1.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.Multiply(1.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.Multiply(2.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.Multiply(0.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.Multiply(-1.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

    def test_other_dtypes_uint_int(self):
        # uint, int
        for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            dtype = np.dtype(dtype)
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            image = np.full((3, 3), 10, dtype=dtype)
            aug = iaa.Multiply(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 10)

            image = np.full((3, 3), 10, dtype=dtype)
            aug = iaa.Multiply(10)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 100)

            image = np.full((3, 3), 10, dtype=dtype)
            aug = iaa.Multiply(0.5)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 5)

            image = np.full((3, 3), 0, dtype=dtype)
            aug = iaa.Multiply(0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 0)

            if np.dtype(dtype).kind == "u":
                image = np.full((3, 3), 10, dtype=dtype)
                aug = iaa.Multiply(-1)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(image_aug == 0)
            else:
                image = np.full((3, 3), 10, dtype=dtype)
                aug = iaa.Multiply(-1)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(image_aug == -10)

            image = np.full((3, 3), int(center_value), dtype=dtype)
            aug = iaa.Multiply(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == int(center_value))

            image = np.full((3, 3), int(center_value), dtype=dtype)
            aug = iaa.Multiply(1.2)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == int(1.2 * int(center_value)))

            if np.dtype(dtype).kind == "u":
                image = np.full((3, 3), int(center_value), dtype=dtype)
                aug = iaa.Multiply(100)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(image_aug == max_value)

            image = np.full((3, 3), max_value, dtype=dtype)
            aug = iaa.Multiply(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == max_value)

            # non-uint8 currently don't increase the itemsize
            if dtype.name == "uint8":
                image = np.full((3, 3), max_value, dtype=dtype)
                aug = iaa.Multiply(10)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(image_aug == max_value)

            image = np.full((3, 3), max_value, dtype=dtype)
            aug = iaa.Multiply(0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 0)

            # non-uint8 currently don't increase the itemsize
            if dtype.name == "uint8":
                image = np.full((3, 3), max_value, dtype=dtype)
                aug = iaa.Multiply(-2)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(image_aug == min_value)

            # non-uint8 currently don't increase the itemsize
            if dtype.name == "uint8":
                for _ in sm.xrange(10):
                    image = np.full((1, 1, 3), 10, dtype=dtype)
                    aug = iaa.Multiply(iap.Uniform(0.5, 1.5))
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(np.logical_and(5 <= image_aug, image_aug <= 15))
                    assert len(np.unique(image_aug)) == 1

                    image = np.full((1, 1, 100), 10, dtype=dtype)
                    aug = iaa.Multiply(iap.Uniform(0.5, 1.5), per_channel=True)
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(np.logical_and(5 <= image_aug, image_aug <= 15))
                    assert len(np.unique(image_aug)) > 1

                    image = np.full((1, 1, 3), 10, dtype=dtype)
                    aug = iaa.Multiply(iap.DiscreteUniform(1, 3))
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                    assert len(np.unique(image_aug)) == 1

                    image = np.full((1, 1, 100), 10, dtype=dtype)
                    aug = iaa.Multiply(iap.DiscreteUniform(1, 3), per_channel=True)
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                    assert len(np.unique(image_aug)) > 1

    def test_other_dtypes_float(self):
        # float
        for dtype in [np.float16, np.float32]:
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            if dtype == np.float16:
                atol = 1e-3 * max_value
            else:
                atol = 1e-9 * max_value
            _allclose = functools.partial(np.allclose, atol=atol, rtol=0)

            image = np.full((3, 3), 10.0, dtype=dtype)
            aug = iaa.Multiply(1.0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 10.0)

            image = np.full((3, 3), 10.0, dtype=dtype)
            aug = iaa.Multiply(2.0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 20.0)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), max_value, dtype=dtype)
            # aug = iaa.Multiply(-10)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert _allclose(image_aug, min_value)

            image = np.full((3, 3), max_value, dtype=dtype)
            aug = iaa.Multiply(0.0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 0.0)

            image = np.full((3, 3), max_value, dtype=dtype)
            aug = iaa.Multiply(0.5)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 0.5*max_value)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), min_value, dtype=dtype)
            # aug = iaa.Multiply(-2.0)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert _allclose(image_aug, max_value)

            image = np.full((3, 3), min_value, dtype=dtype)
            aug = iaa.Multiply(0.0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 0.0)

            # using tolerances of -100 - 1e-2 and 100 + 1e-2 is not enough for float16, had to be increased to -/+ 1e-1
            # deactivated, because itemsize increase was deactivated
            """
            for _ in sm.xrange(10):
                image = np.full((1, 1, 3), 10.0, dtype=dtype)
                aug = iaa.Multiply(iap.Uniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
                assert np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
                assert np.allclose(image_aug[..., 0], image_aug[..., 1])

                image = np.full((1, 1, 100), 10.0, dtype=dtype)
                aug = iaa.Multiply(iap.Uniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
                assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])

                image = np.full((1, 1, 3), 10.0, dtype=dtype)
                aug = iaa.Multiply(iap.DiscreteUniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
                assert np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
                assert np.allclose(image_aug[..., 0], image_aug[..., 1])

                image = np.full((1, 1, 100), 10.0, dtype=dtype)
                aug = iaa.Multiply(iap.DiscreteUniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
                assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])
            """

    def test_pickleable(self):
        aug = iaa.Multiply((0.5, 1.5), per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=20)


class TestMultiplyElementwise(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_mul_is_one(self):
        # no multiply, shouldnt change anything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.MultiplyElementwise(mul=1.0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

    def test_mul_is_above_one(self):
        # multiply >1.0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.MultiplyElementwise(mul=1.2)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
        assert array_equal_lists(observed, expected)

    def test_mul_is_below_one(self):
        # multiply <1.0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.MultiplyElementwise(mul=0.8)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
        assert array_equal_lists(observed, expected)

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.MultiplyElementwise(mul=1.2)
        aug_det = iaa.Multiply(mul=1.2).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test_tuple_as_mul(self):
        # varying multiply factors
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

        aug = iaa.MultiplyElementwise(mul=(0, 2.0))
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
        assert nb_changed_aug >= int(nb_iterations * 0.95)
        assert nb_changed_aug_det == 0

    def test_samples_change_by_spatial_location(self):
        # values should change between pixels
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

        aug = iaa.MultiplyElementwise(mul=(0.5, 1.5))

        nb_same = 0
        nb_different = 0
        nb_iterations = 1000
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(images)
            observed_aug_flat = observed_aug.flatten()
            last = None
            for j in sm.xrange(observed_aug_flat.size):
                if last is not None:
                    v = observed_aug_flat[j]
                    if v - 0.0001 <= last <= v + 0.0001:
                        nb_same += 1
                    else:
                        nb_different += 1
                last = observed_aug_flat[j]
        assert nb_different > 0.95 * (nb_different + nb_same)

    def test_per_channel(self):
        # test channelwise
        aug = iaa.MultiplyElementwise(mul=iap.Choice([0, 1]), per_channel=True)
        observed = aug.augment_image(np.ones((100, 100, 3), dtype=np.uint8))
        sums = np.sum(observed, axis=2)
        values = np.unique(sums)
        assert all([(value in values) for value in [0, 1, 2, 3]])
        assert observed.shape == (100, 100, 3)

    def test_per_channel_with_probability(self):
        # test channelwise with probability
        aug = iaa.MultiplyElementwise(mul=iap.Choice([0, 1]), per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(400):
            observed = aug.augment_image(np.ones((20, 20, 3), dtype=np.uint8))
            assert observed.shape == (20, 20, 3)

            sums = np.sum(observed, axis=2)
            values = np.unique(sums)
            all_values_found = all([(value in values) for value in [0, 1, 2, 3]])
            if all_values_found:
                seen[0] += 1
            else:
                seen[1] += 1
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _aug = iaa.MultiplyElementwise(mul="test")
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _aug = iaa.MultiplyElementwise(mul=1, per_channel="test")
        except Exception:
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
                image = np.ones(shape, dtype=np.uint8)
                aug = iaa.MultiplyElementwise(2)

                image_aug = aug(image=image)

                assert np.all(image_aug == 2)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.ones(shape, dtype=np.uint8)
                aug = iaa.MultiplyElementwise(2)

                image_aug = aug(image=image)

                assert np.all(image_aug == 2)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.MultiplyElementwise(mul=1, per_channel=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert params[0].value == 1
        assert params[1].value == 0

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.MultiplyElementwise(mul=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_bool(self):
        # bool
        image = np.zeros((3, 3), dtype=bool)
        aug = iaa.MultiplyElementwise(1.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.MultiplyElementwise(1.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.MultiplyElementwise(2.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.MultiplyElementwise(0.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

        image = np.full((3, 3), True, dtype=bool)
        aug = iaa.MultiplyElementwise(-1.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

    def test_other_dtypes_uint_int(self):
        # uint, int
        for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            dtype = np.dtype(dtype)
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            image = np.full((3, 3), 10, dtype=dtype)
            aug = iaa.MultiplyElementwise(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 10)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), 10, dtype=dtype)
            # aug = iaa.MultiplyElementwise(10)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert np.all(image_aug == 100)

            image = np.full((3, 3), 10, dtype=dtype)
            aug = iaa.MultiplyElementwise(0.5)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 5)

            image = np.full((3, 3), 0, dtype=dtype)
            aug = iaa.MultiplyElementwise(0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 0)

            # partially deactivated, because itemsize increase was deactivated
            if dtype.name == "uint8":
                if dtype.kind == "u":
                    image = np.full((3, 3), 10, dtype=dtype)
                    aug = iaa.MultiplyElementwise(-1)
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(image_aug == 0)
                else:
                    image = np.full((3, 3), 10, dtype=dtype)
                    aug = iaa.MultiplyElementwise(-1)
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(image_aug == -10)

            image = np.full((3, 3), int(center_value), dtype=dtype)
            aug = iaa.MultiplyElementwise(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == int(center_value))

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), int(center_value), dtype=dtype)
            # aug = iaa.MultiplyElementwise(1.2)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert np.all(image_aug == int(1.2 * int(center_value)))

            # deactivated, because itemsize increase was deactivated
            if dtype.name == "uint8":
                if dtype.kind == "u":
                    image = np.full((3, 3), int(center_value), dtype=dtype)
                    aug = iaa.MultiplyElementwise(100)
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(image_aug == max_value)

            image = np.full((3, 3), max_value, dtype=dtype)
            aug = iaa.MultiplyElementwise(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == max_value)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), max_value, dtype=dtype)
            # aug = iaa.MultiplyElementwise(10)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert np.all(image_aug == max_value)

            image = np.full((3, 3), max_value, dtype=dtype)
            aug = iaa.MultiplyElementwise(0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 0)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), max_value, dtype=dtype)
            # aug = iaa.MultiplyElementwise(-2)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert np.all(image_aug == min_value)

            # partially deactivated, because itemsize increase was deactivated
            if dtype.name == "uint8":
                for _ in sm.xrange(10):
                    image = np.full((5, 5, 3), 10, dtype=dtype)
                    aug = iaa.MultiplyElementwise(iap.Uniform(0.5, 1.5))
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(np.logical_and(5 <= image_aug, image_aug <= 15))
                    assert len(np.unique(image_aug)) > 1
                    assert np.all(image_aug[..., 0] == image_aug[..., 1])

                    image = np.full((1, 1, 100), 10, dtype=dtype)
                    aug = iaa.MultiplyElementwise(iap.Uniform(0.5, 1.5), per_channel=True)
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(np.logical_and(5 <= image_aug, image_aug <= 15))
                    assert len(np.unique(image_aug)) > 1

                    image = np.full((5, 5, 3), 10, dtype=dtype)
                    aug = iaa.MultiplyElementwise(iap.DiscreteUniform(1, 3))
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                    assert len(np.unique(image_aug)) > 1
                    assert np.all(image_aug[..., 0] == image_aug[..., 1])

                    image = np.full((1, 1, 100), 10, dtype=dtype)
                    aug = iaa.MultiplyElementwise(iap.DiscreteUniform(1, 3), per_channel=True)
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.type == dtype
                    assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
                    assert len(np.unique(image_aug)) > 1

    def test_other_dtypes_float(self):
        # float
        for dtype in [np.float16, np.float32]:
            dtype = np.dtype(dtype)
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            if dtype == np.float16:
                atol = 1e-3 * max_value
            else:
                atol = 1e-9 * max_value
            _allclose = functools.partial(np.allclose, atol=atol, rtol=0)

            image = np.full((3, 3), 10.0, dtype=dtype)
            aug = iaa.MultiplyElementwise(1.0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 10.0)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), 10.0, dtype=dtype)
            # aug = iaa.MultiplyElementwise(2.0)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert _allclose(image_aug, 20.0)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), max_value, dtype=dtype)
            # aug = iaa.MultiplyElementwise(-10)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert _allclose(image_aug, min_value)

            image = np.full((3, 3), max_value, dtype=dtype)
            aug = iaa.MultiplyElementwise(0.0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 0.0)

            image = np.full((3, 3), max_value, dtype=dtype)
            aug = iaa.MultiplyElementwise(0.5)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 0.5*max_value)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), min_value, dtype=dtype)
            # aug = iaa.MultiplyElementwise(-2.0)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert _allclose(image_aug, max_value)

            image = np.full((3, 3), min_value, dtype=dtype)
            aug = iaa.MultiplyElementwise(0.0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 0.0)

            # using tolerances of -100 - 1e-2 and 100 + 1e-2 is not enough for float16, had to be increased to -/+ 1e-1
            # deactivated, because itemsize increase was deactivated
            """
            for _ in sm.xrange(10):
                image = np.full((50, 1, 3), 10.0, dtype=dtype)
                aug = iaa.MultiplyElementwise(iap.Uniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
                assert not np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
                assert np.allclose(image_aug[..., 0], image_aug[..., 1])

                image = np.full((1, 1, 100), 10.0, dtype=dtype)
                aug = iaa.MultiplyElementwise(iap.Uniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
                assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])

                image = np.full((50, 1, 3), 10.0, dtype=dtype)
                aug = iaa.MultiplyElementwise(iap.DiscreteUniform(-10, 10))
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
                assert not np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
                assert np.allclose(image_aug[..., 0], image_aug[..., 1])

                image = np.full((1, 1, 100), 10, dtype=dtype)
                aug = iaa.MultiplyElementwise(iap.DiscreteUniform(-10, 10), per_channel=True)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
                assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])
            """

    def test_pickleable(self):
        aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=True,
                                      seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3)


class TestReplaceElementwise(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_mask_is_always_zero(self):
        # no replace, shouldnt change anything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) + 99
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.ReplaceElementwise(mask=0, replacement=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

    def test_mask_is_always_one(self):
        # replace at 100 percent prob., should change everything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) + 99
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.ReplaceElementwise(mask=1, replacement=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = np.zeros((1, 3, 3, 1), dtype=np.uint8)
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [np.zeros((3, 3, 1), dtype=np.uint8)]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = np.zeros((1, 3, 3, 1), dtype=np.uint8)
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [np.zeros((3, 3, 1), dtype=np.uint8)]
        assert array_equal_lists(observed, expected)

    def test_mask_is_stochastic_parameter(self):
        # replace half
        aug = iaa.ReplaceElementwise(mask=iap.Binomial(p=0.5), replacement=0)
        img = np.ones((100, 100, 1), dtype=np.uint8)

        nb_iterations = 100
        nb_diff_all = 0
        for i in sm.xrange(nb_iterations):
            observed = aug.augment_image(img)
            nb_diff = np.sum(img != observed)
            nb_diff_all += nb_diff
        p = nb_diff_all / (nb_iterations * 100 * 100)
        assert 0.45 <= p <= 0.55

    def test_mask_is_list(self):
        # mask is list
        aug = iaa.ReplaceElementwise(mask=[0.2, 0.7], replacement=1)
        img = np.zeros((20, 20, 1), dtype=np.uint8)

        seen = [0, 0, 0]
        for i in sm.xrange(400):
            observed = aug.augment_image(img)
            p = np.mean(observed)
            if 0.1 < p < 0.3:
                seen[0] += 1
            elif 0.6 < p < 0.8:
                seen[1] += 1
            else:
                seen[2] += 1
        assert seen[2] <= 10
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        base_img = np.ones((3, 3, 1), dtype=np.uint8) + 99
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.ReplaceElementwise(mask=iap.Binomial(p=0.5), replacement=0)
        aug_det = iaa.ReplaceElementwise(mask=iap.Binomial(p=0.5), replacement=0).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test_replacement_is_stochastic_parameter(self):
        # different replacements
        aug = iaa.ReplaceElementwise(mask=1, replacement=iap.Choice([100, 200]))
        img = np.zeros((1000, 1000, 1), dtype=np.uint8)
        img100 = img + 100
        img200 = img + 200
        observed = aug.augment_image(img)
        nb_diff_100 = np.sum(img100 != observed)
        nb_diff_200 = np.sum(img200 != observed)
        p100 = nb_diff_100 / (1000 * 1000)
        p200 = nb_diff_200 / (1000 * 1000)
        assert 0.45 <= p100 <= 0.55
        assert 0.45 <= p200 <= 0.55
        # test channelwise
        aug = iaa.MultiplyElementwise(mul=iap.Choice([0, 1]), per_channel=True)
        observed = aug.augment_image(np.ones((100, 100, 3), dtype=np.uint8))
        sums = np.sum(observed, axis=2)
        values = np.unique(sums)
        assert all([(value in values) for value in [0, 1, 2, 3]])

    def test_per_channel_with_probability(self):
        # test channelwise with probability
        aug = iaa.ReplaceElementwise(mask=iap.Choice([0, 1]), replacement=1, per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(400):
            observed = aug.augment_image(np.zeros((20, 20, 3), dtype=np.uint8))
            assert observed.shape == (20, 20, 3)

            sums = np.sum(observed, axis=2)
            values = np.unique(sums)
            all_values_found = all([(value in values) for value in [0, 1, 2, 3]])
            if all_values_found:
                seen[0] += 1
            else:
                seen[1] += 1
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _aug = iaa.ReplaceElementwise(mask="test", replacement=1)
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _aug = iaa.ReplaceElementwise(mask=1, replacement=1, per_channel="test")
        except Exception:
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
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.ReplaceElementwise(1.0, 1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

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
                aug = iaa.ReplaceElementwise(1.0, 1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.ReplaceElementwise(mask=0.5, replacement=2, per_channel=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Binomial)
        assert isinstance(params[0].p, iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert isinstance(params[2], iap.Deterministic)
        assert 0.5 - 1e-6 < params[0].p.value < 0.5 + 1e-6
        assert params[1].value == 2
        assert params[2].value == 0

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.ReplaceElementwise(mask=1, replacement=0.5)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_bool(self):
        # bool
        aug = iaa.ReplaceElementwise(mask=1, replacement=0)
        image = np.full((3, 3), False, dtype=bool)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

        aug = iaa.ReplaceElementwise(mask=1, replacement=1)
        image = np.full((3, 3), False, dtype=bool)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        aug = iaa.ReplaceElementwise(mask=1, replacement=0)
        image = np.full((3, 3), True, dtype=bool)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

        aug = iaa.ReplaceElementwise(mask=1, replacement=1)
        image = np.full((3, 3), True, dtype=bool)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        aug = iaa.ReplaceElementwise(mask=1, replacement=0.7)
        image = np.full((3, 3), False, dtype=bool)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 1)

        aug = iaa.ReplaceElementwise(mask=1, replacement=0.2)
        image = np.full((3, 3), False, dtype=bool)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == np.bool_
        assert np.all(image_aug == 0)

    def test_other_dtypes_uint_int(self):
        # uint, int
        for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
            dtype = np.dtype(dtype)
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            aug = iaa.ReplaceElementwise(mask=1, replacement=1)
            image = np.full((3, 3), 0, dtype=dtype)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 1)

            aug = iaa.ReplaceElementwise(mask=1, replacement=2)
            image = np.full((3, 3), 1, dtype=dtype)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 2)

            # deterministic stochastic parameters are by default int32 for
            # any integer value and hence cannot cover the full uint32 value
            # range
            if dtype.name != "uint32":
                aug = iaa.ReplaceElementwise(mask=1, replacement=max_value)
                image = np.full((3, 3), min_value, dtype=dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(image_aug == max_value)

                aug = iaa.ReplaceElementwise(mask=1, replacement=min_value)
                image = np.full((3, 3), max_value, dtype=dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(image_aug == min_value)

            aug = iaa.ReplaceElementwise(mask=1, replacement=iap.Uniform(1.0, 10.0))
            image = np.full((100, 1), 0, dtype=dtype)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(1 <= image_aug, image_aug <= 10))
            assert len(np.unique(image_aug)) > 1

            aug = iaa.ReplaceElementwise(mask=1, replacement=iap.DiscreteUniform(1, 10))
            image = np.full((100, 1), 0, dtype=dtype)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(1 <= image_aug, image_aug <= 10))
            assert len(np.unique(image_aug)) > 1

            aug = iaa.ReplaceElementwise(mask=0.5, replacement=iap.DiscreteUniform(1, 10), per_channel=True)
            image = np.full((1, 1, 100), 0, dtype=dtype)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(0 <= image_aug, image_aug <= 10))
            assert len(np.unique(image_aug)) > 2

    def test_other_dtypes_float(self):
        # float
        for dtype in [np.float16, np.float32, np.float64]:
            dtype = np.dtype(dtype)
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            atol = 1e-3*max_value if dtype == np.float16 else 1e-9 * max_value
            _allclose = functools.partial(np.allclose, atol=atol, rtol=0)

            aug = iaa.ReplaceElementwise(mask=1, replacement=1.0)
            image = np.full((3, 3), 0.0, dtype=dtype)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.allclose(image_aug, 1.0)

            aug = iaa.ReplaceElementwise(mask=1, replacement=2.0)
            image = np.full((3, 3), 1.0, dtype=dtype)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.allclose(image_aug, 2.0)

            # deterministic stochastic parameters are by default float32 for
            # any float value and hence cannot cover the full float64 value
            # range
            if dtype.name != "float64":
                aug = iaa.ReplaceElementwise(mask=1, replacement=max_value)
                image = np.full((3, 3), min_value, dtype=dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert _allclose(image_aug, max_value)

                aug = iaa.ReplaceElementwise(mask=1, replacement=min_value)
                image = np.full((3, 3), max_value, dtype=dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert _allclose(image_aug, min_value)

            aug = iaa.ReplaceElementwise(mask=1, replacement=iap.Uniform(1.0, 10.0))
            image = np.full((100, 1), 0, dtype=dtype)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(1 <= image_aug, image_aug <= 10))
            assert not np.allclose(image_aug[1:, :], image_aug[:-1, :], atol=0.01)

            aug = iaa.ReplaceElementwise(mask=1, replacement=iap.DiscreteUniform(1, 10))
            image = np.full((100, 1), 0, dtype=dtype)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(1 <= image_aug, image_aug <= 10))
            assert not np.allclose(image_aug[1:, :], image_aug[:-1, :], atol=0.01)

            aug = iaa.ReplaceElementwise(mask=0.5, replacement=iap.DiscreteUniform(1, 10), per_channel=True)
            image = np.full((1, 1, 100), 0, dtype=dtype)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(0 <= image_aug, image_aug <= 10))
            assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1], atol=0.01)

    def test_pickleable(self):
        aug = iaa.ReplaceElementwise(mask=0.5, replacement=(0, 255),
                                     per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3)


# not more tests necessary here as SaltAndPepper is just a tiny wrapper around
# ReplaceElementwise
class TestSaltAndPepper(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.SaltAndPepper(p=0.5)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6

    def test_p_is_one(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.SaltAndPepper(p=1.0)
        observed = aug.augment_image(base_img)
        nb_pepper = np.sum(observed < 40)
        nb_salt = np.sum(observed > 255 - 40)
        assert nb_pepper > 200
        assert nb_salt > 200

    def test_pickleable(self):
        aug = iaa.SaltAndPepper(p=0.5, per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3)


class TestCoarseSaltAndPepper(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.CoarseSaltAndPepper(p=0.5, size_px=100)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6

    def test_size_px(self):
        aug1 = iaa.CoarseSaltAndPepper(p=0.5, size_px=100)
        aug2 = iaa.CoarseSaltAndPepper(p=0.5, size_px=10)
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        ps1 = []
        ps2 = []
        for _ in sm.xrange(100):
            observed1 = aug1.augment_image(base_img)
            observed2 = aug2.augment_image(base_img)
            p1 = np.mean(observed1 != 128)
            p2 = np.mean(observed2 != 128)
            ps1.append(p1)
            ps2.append(p2)
        assert 0.4 < np.mean(ps2) < 0.6
        assert np.std(ps1)*1.5 < np.std(ps2)

    def test_p_is_list(self):
        aug = iaa.CoarseSaltAndPepper(p=[0.2, 0.5], size_px=100)
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        seen = [0, 0, 0]
        for _ in sm.xrange(200):
            observed = aug.augment_image(base_img)
            p = np.mean(observed != 128)
            diff_020 = abs(0.2 - p)
            diff_050 = abs(0.5 - p)
            if diff_020 < 0.025:
                seen[0] += 1
            elif diff_050 < 0.025:
                seen[1] += 1
            else:
                seen[2] += 1
        assert seen[2] < 10
        assert 75 < seen[0] < 125
        assert 75 < seen[1] < 125

    def test_p_is_tuple(self):
        aug = iaa.CoarseSaltAndPepper(p=(0.0, 1.0), size_px=50)
        base_img = np.zeros((50, 50, 1), dtype=np.uint8) + 128
        ps = []
        for _ in sm.xrange(200):
            observed = aug.augment_image(base_img)
            p = np.mean(observed != 128)
            ps.append(p)

        nb_bins = 5
        hist, _ = np.histogram(ps, bins=nb_bins, range=(0.0, 1.0), density=False)
        tolerance = 0.05
        for nb_seen in hist:
            density = nb_seen / len(ps)
            assert density - tolerance < density < density + tolerance

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _ = iaa.CoarseSaltAndPepper(p="test", size_px=100)
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _ = iaa.CoarseSaltAndPepper(p=0.5, size_px=None, size_percent=None)
        except Exception:
            got_exception = True
        assert got_exception

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.CoarseSaltAndPepper(p=1.0, size_px=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_pickleable(self):
        aug = iaa.CoarseSaltAndPepper(p=0.5, size_px=(4, 15),
                                      per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=20)


# not more tests necessary here as Salt is just a tiny wrapper around
# ReplaceElementwise
class TestSalt(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.Salt(p=0.5)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6
        # Salt() occasionally replaces with 127, which probably should be the center-point here anyways
        assert np.all(observed >= 127)

    def test_p_is_one(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.Salt(p=1.0)
        observed = aug.augment_image(base_img)
        nb_pepper = np.sum(observed < 40)
        nb_salt = np.sum(observed > 255 - 40)
        assert nb_pepper == 0
        assert nb_salt > 200

    def test_pickleable(self):
        aug = iaa.Salt(p=0.5, per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3)


class TestCoarseSalt(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.CoarseSalt(p=0.5, size_px=100)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6

    def test_size_px(self):
        aug1 = iaa.CoarseSalt(p=0.5, size_px=100)
        aug2 = iaa.CoarseSalt(p=0.5, size_px=10)
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        ps1 = []
        ps2 = []
        for _ in sm.xrange(100):
            observed1 = aug1.augment_image(base_img)
            observed2 = aug2.augment_image(base_img)
            p1 = np.mean(observed1 != 128)
            p2 = np.mean(observed2 != 128)
            ps1.append(p1)
            ps2.append(p2)
        assert 0.4 < np.mean(ps2) < 0.6
        assert np.std(ps1)*1.5 < np.std(ps2)

    def test_p_is_list(self):
        aug = iaa.CoarseSalt(p=[0.2, 0.5], size_px=100)
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        seen = [0, 0, 0]
        for _ in sm.xrange(200):
            observed = aug.augment_image(base_img)
            p = np.mean(observed != 128)
            diff_020 = abs(0.2 - p)
            diff_050 = abs(0.5 - p)
            if diff_020 < 0.025:
                seen[0] += 1
            elif diff_050 < 0.025:
                seen[1] += 1
            else:
                seen[2] += 1
        assert seen[2] < 10
        assert 75 < seen[0] < 125
        assert 75 < seen[1] < 125

    def test_p_is_tuple(self):
        aug = iaa.CoarseSalt(p=(0.0, 1.0), size_px=50)
        base_img = np.zeros((50, 50, 1), dtype=np.uint8) + 128
        ps = []
        for _ in sm.xrange(200):
            observed = aug.augment_image(base_img)
            p = np.mean(observed != 128)
            ps.append(p)

        nb_bins = 5
        hist, _ = np.histogram(ps, bins=nb_bins, range=(0.0, 1.0), density=False)
        tolerance = 0.05
        for nb_seen in hist:
            density = nb_seen / len(ps)
            assert density - tolerance < density < density + tolerance

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _ = iaa.CoarseSalt(p="test", size_px=100)
        except Exception:
            got_exception = True
        assert got_exception

    def test_size_px_or_size_percent_not_none(self):
        got_exception = False
        try:
            _ = iaa.CoarseSalt(p=0.5, size_px=None, size_percent=None)
        except Exception:
            got_exception = True
        assert got_exception

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.CoarseSalt(p=1.0, size_px=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_pickleable(self):
        aug = iaa.CoarseSalt(p=0.5, size_px=(4, 15),
                             per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=20)


# not more tests necessary here as Salt is just a tiny wrapper around
# ReplaceElementwise
class TestPepper(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_probability_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.Pepper(p=0.5)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6
        assert np.all(observed <= 128)

    def test_probability_is_one(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.Pepper(p=1.0)
        observed = aug.augment_image(base_img)
        nb_pepper = np.sum(observed < 40)
        nb_salt = np.sum(observed > 255 - 40)
        assert nb_pepper > 200
        assert nb_salt == 0

    def test_pickleable(self):
        aug = iaa.Pepper(p=0.5, per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3)


class TestCoarsePepper(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.CoarsePepper(p=0.5, size_px=100)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6

    def test_size_px(self):
        aug1 = iaa.CoarsePepper(p=0.5, size_px=100)
        aug2 = iaa.CoarsePepper(p=0.5, size_px=10)
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        ps1 = []
        ps2 = []
        for _ in sm.xrange(100):
            observed1 = aug1.augment_image(base_img)
            observed2 = aug2.augment_image(base_img)
            p1 = np.mean(observed1 != 128)
            p2 = np.mean(observed2 != 128)
            ps1.append(p1)
            ps2.append(p2)
        assert 0.4 < np.mean(ps2) < 0.6
        assert np.std(ps1)*1.5 < np.std(ps2)

    def test_p_is_list(self):
        aug = iaa.CoarsePepper(p=[0.2, 0.5], size_px=100)
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        seen = [0, 0, 0]
        for _ in sm.xrange(200):
            observed = aug.augment_image(base_img)
            p = np.mean(observed != 128)
            diff_020 = abs(0.2 - p)
            diff_050 = abs(0.5 - p)
            if diff_020 < 0.025:
                seen[0] += 1
            elif diff_050 < 0.025:
                seen[1] += 1
            else:
                seen[2] += 1
        assert seen[2] < 10
        assert 75 < seen[0] < 125
        assert 75 < seen[1] < 125

    def test_p_is_tuple(self):
        aug = iaa.CoarsePepper(p=(0.0, 1.0), size_px=50)
        base_img = np.zeros((50, 50, 1), dtype=np.uint8) + 128
        ps = []
        for _ in sm.xrange(200):
            observed = aug.augment_image(base_img)
            p = np.mean(observed != 128)
            ps.append(p)

        nb_bins = 5
        hist, _ = np.histogram(ps, bins=nb_bins, range=(0.0, 1.0), density=False)
        tolerance = 0.05
        for nb_seen in hist:
            density = nb_seen / len(ps)
            assert density - tolerance < density < density + tolerance

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _ = iaa.CoarsePepper(p="test", size_px=100)
        except Exception:
            got_exception = True
        assert got_exception

    def test_size_px_or_size_percent_not_none(self):
        got_exception = False
        try:
            _ = iaa.CoarsePepper(p=0.5, size_px=None, size_percent=None)
        except Exception:
            got_exception = True
        assert got_exception

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.CoarsePepper(p=1.0, size_px=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_pickleable(self):
        aug = iaa.CoarsePepper(p=0.5, size_px=(4, 15),
                               per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=20)


class Test_invert(unittest.TestCase):
    @mock.patch("imgaug.augmenters.arithmetic.invert_")
    def test_mocked_defaults(self, mock_invert):
        mock_invert.return_value = "foo"
        arr = np.zeros((1,), dtype=np.uint8)
        observed = iaa.invert(arr)

        assert observed == "foo"
        args = mock_invert.call_args_list[0]
        assert np.array_equal(mock_invert.call_args_list[0][0][0], arr)
        assert args[1]["min_value"] is None
        assert args[1]["max_value"] is None
        assert args[1]["threshold"] is None
        assert args[1]["invert_above_threshold"] is True

    @mock.patch("imgaug.augmenters.arithmetic.invert_")
    def test_mocked(self, mock_invert):
        mock_invert.return_value = "foo"
        arr = np.zeros((1,), dtype=np.uint8)
        observed = iaa.invert(arr, min_value=1, max_value=10, threshold=5,
                              invert_above_threshold=False)

        assert observed == "foo"
        args = mock_invert.call_args_list[0]
        assert np.array_equal(mock_invert.call_args_list[0][0][0], arr)
        assert args[1]["min_value"] == 1
        assert args[1]["max_value"] == 10
        assert args[1]["threshold"] == 5
        assert args[1]["invert_above_threshold"] is False

    def test_uint8(self):
        values = np.array([0, 20, 45, 60, 128, 255], dtype=np.uint8)
        expected = np.array([
            255,
            255-20,
            255-45,
            255-60,
            255-128,
            255-255
        ], dtype=np.uint8)

        observed = iaa.invert(values)

        assert np.array_equal(observed, expected)
        assert observed is not values


# most parts of this function are tested via Invert
class Test_invert_(unittest.TestCase):
    def test_arr_is_noncontiguous_uint8(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)
        max_vr_flipped = np.fliplr(np.copy(zeros + 255))

        observed = iaa.invert_(max_vr_flipped)
        expected = zeros
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_arr_is_view_uint8(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)
        max_vr_view = np.copy(zeros + 255)[:, :, [0, 2]]

        observed = iaa.invert_(max_vr_view)
        expected = zeros[:, :, [0, 2]]
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_uint(self):
        dtypes = ["uint8", "uint16", "uint32", "uint64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    max_value - 0,
                    max_value - 20,
                    max_value - 45,
                    max_value - 60,
                    max_value - center_value,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values))

                assert np.array_equal(observed, expected)

    def test_uint_with_threshold_50_inv_above(self):
        threshold = 50
        dtypes = ["uint8", "uint16", "uint32", "uint64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    0,
                    20,
                    45,
                    max_value - 60,
                    max_value - center_value,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_uint_with_threshold_0_inv_above(self):
        threshold = 0
        dtypes = ["uint8", "uint16", "uint32", "uint64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    max_value - 0,
                    max_value - 20,
                    max_value - 45,
                    max_value - 60,
                    max_value - center_value,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_uint8_with_threshold_255_inv_above(self):
        threshold = 255
        dtypes = ["uint8"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    0,
                    20,
                    45,
                    60,
                    center_value,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_uint8_with_threshold_256_inv_above(self):
        threshold = 256
        dtypes = ["uint8"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    0,
                    20,
                    45,
                    60,
                    center_value,
                    max_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_uint_with_threshold_50_inv_below(self):
        threshold = 50
        dtypes = ["uint8", "uint16", "uint32", "uint64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    max_value - 0,
                    max_value - 20,
                    max_value - 45,
                    60,
                    center_value,
                    max_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=False)

                assert np.array_equal(observed, expected)

    def test_uint_with_threshold_50_inv_above_with_min_max(self):
        threshold = 50
        # uint64 does not support custom min/max, hence removed it here
        dtypes = ["uint8", "uint16", "uint32"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    0,  # not clipped to 10 as only >thresh affected
                    20,
                    45,
                    100 - 50,
                    100 - 90,
                    100 - 90
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       min_value=10,
                                       max_value=100,
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_int_with_threshold_50_inv_above(self):
        threshold = 50
        dtypes = ["int8", "int16", "int32", "int64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([-45, -20, center_value, 20, 45, max_value],
                                  dtype=dt)
                expected = np.array([
                    -45,
                    -20,
                    center_value,
                    20,
                    45,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_int_with_threshold_50_inv_below(self):
        threshold = 50
        dtypes = ["int8", "int16", "int32", "int64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([-45, -20, center_value, 20, 45, max_value],
                                  dtype=dt)
                expected = np.array([
                    (-1) * (-45) - 1,
                    (-1) * (-20) - 1,
                    (-1) * center_value - 1,
                    (-1) * 20 - 1,
                    (-1) * 45 - 1,
                    max_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=False)

                assert np.array_equal(observed, expected)

    def test_float_with_threshold_50_inv_above(self):
        threshold = 50
        dtypes = ["float16", "float32", "float64", "float128"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = center_value

                values = np.array([-45.5, -20.5, center_value, 20.5, 45.5,
                                   max_value],
                                  dtype=dt)
                expected = np.array([
                    -45.5,
                    -20.5,
                    center_value,
                    20.5,
                    45.5,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.allclose(observed, expected, rtol=0, atol=1e-4)

    def test_float_with_threshold_50_inv_below(self):
        threshold = 50
        dtypes = ["float16", "float32", "float64", "float128"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = center_value

                values = np.array([-45.5, -20.5, center_value, 20.5, 45.5,
                                   max_value],
                                  dtype=dt)
                expected = np.array([
                    (-1) * (-45.5),
                    (-1) * (-20.5),
                    (-1) * center_value,
                    (-1) * 20.5,
                    (-1) * 45.5,
                    max_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=False)

                assert np.allclose(observed, expected, rtol=0, atol=1e-4)


class Test_solarize(unittest.TestCase):
    @mock.patch("imgaug.augmenters.arithmetic.solarize_")
    def test_mocked_defaults(self, mock_sol):
        arr = np.zeros((1,), dtype=np.uint8)
        mock_sol.return_value = "foo"

        observed = iaa.solarize(arr)

        args = mock_sol.call_args_list[0][0]
        kwargs = mock_sol.call_args_list[0][1]
        assert args[0] is not arr
        assert np.array_equal(args[0], arr)
        assert kwargs["threshold"] == 128
        assert observed == "foo"

    @mock.patch("imgaug.augmenters.arithmetic.solarize_")
    def test_mocked(self, mock_sol):
        arr = np.zeros((1,), dtype=np.uint8)
        mock_sol.return_value = "foo"

        observed = iaa.solarize(arr, threshold=5)

        args = mock_sol.call_args_list[0][0]
        kwargs = mock_sol.call_args_list[0][1]
        assert args[0] is not arr
        assert np.array_equal(args[0], arr)
        assert kwargs["threshold"] == 5
        assert observed == "foo"

    def test_uint8(self):
        arr = np.array([0, 10, 50, 150, 200, 255], dtype=np.uint8)
        arr = arr.reshape((2, 3, 1))

        observed = iaa.solarize(arr)

        expected = np.array([0, 10, 50, 255-150, 255-200, 255-255],
                            dtype=np.uint8).reshape((2, 3, 1))
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)


class Test_solarize_(unittest.TestCase):
    @mock.patch("imgaug.augmenters.arithmetic.invert_")
    def test_mocked_defaults(self, mock_sol):
        arr = np.zeros((1,), dtype=np.uint8)
        mock_sol.return_value = "foo"

        observed = iaa.solarize_(arr)

        args = mock_sol.call_args_list[0][0]
        kwargs = mock_sol.call_args_list[0][1]
        assert args[0] is arr
        assert kwargs["threshold"] == 128
        assert observed == "foo"

    @mock.patch("imgaug.augmenters.arithmetic.invert_")
    def test_mocked(self, mock_sol):
        arr = np.zeros((1,), dtype=np.uint8)
        mock_sol.return_value = "foo"

        observed = iaa.solarize_(arr, threshold=5)

        args = mock_sol.call_args_list[0][0]
        kwargs = mock_sol.call_args_list[0][1]
        assert args[0] is arr
        assert kwargs["threshold"] == 5
        assert observed == "foo"


class TestInvert(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_one(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        observed = iaa.Invert(p=1.0).augment_image(zeros + 255)
        expected = zeros
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_p_is_zero(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        observed = iaa.Invert(p=0.0).augment_image(zeros + 255)
        expected = zeros + 255
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_max_value_set(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        observed = iaa.Invert(p=1.0, max_value=200).augment_image(zeros + 200)
        expected = zeros
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_min_value_and_max_value_set(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        observed = iaa.Invert(p=1.0, max_value=200, min_value=100).augment_image(zeros + 200)
        expected = zeros + 100
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

        observed = iaa.Invert(p=1.0, max_value=200, min_value=100).augment_image(zeros + 100)
        expected = zeros + 200
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_min_value_and_max_value_set_with_float_image(self):
        # with min/max and float inputs
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        zeros_f32 = zeros.astype(np.float32)
        observed = iaa.Invert(p=1.0, max_value=200, min_value=100).augment_image(zeros_f32 + 200)
        expected = zeros_f32 + 100
        assert observed.dtype.name == "float32"
        assert np.array_equal(observed, expected)

        observed = iaa.Invert(p=1.0, max_value=200, min_value=100).augment_image(zeros_f32 + 100)
        expected = zeros_f32 + 200
        assert observed.dtype.name == "float32"
        assert np.array_equal(observed, expected)

    def test_p_is_80_percent(self):
        nb_iterations = 1000
        nb_inverted = 0
        aug = iaa.Invert(p=0.8)
        img = np.zeros((1, 1, 1), dtype=np.uint8) + 255
        expected = np.zeros((1, 1, 1), dtype=np.uint8)
        for i in sm.xrange(nb_iterations):
            observed = aug.augment_image(img)
            if np.array_equal(observed, expected):
                nb_inverted += 1
        pinv = nb_inverted / nb_iterations
        assert 0.75 <= pinv <= 0.85

        nb_iterations = 1000
        nb_inverted = 0
        aug = iaa.Invert(p=iap.Binomial(0.8))
        img = np.zeros((1, 1, 1), dtype=np.uint8) + 255
        expected = np.zeros((1, 1, 1), dtype=np.uint8)
        for i in sm.xrange(nb_iterations):
            observed = aug.augment_image(img)
            if np.array_equal(observed, expected):
                nb_inverted += 1
        pinv = nb_inverted / nb_iterations
        assert 0.75 <= pinv <= 0.85

    def test_per_channel(self):
        aug = iaa.Invert(p=0.5, per_channel=True)
        img = np.zeros((1, 1, 100), dtype=np.uint8) + 255
        observed = aug.augment_image(img)
        assert len(np.unique(observed)) == 2

    # TODO split into two tests
    def test_p_is_stochastic_parameter_per_channel_is_probability(self):
        nb_iterations = 1000
        aug = iaa.Invert(p=iap.Binomial(0.8), per_channel=0.7)
        img = np.zeros((1, 1, 20), dtype=np.uint8) + 255
        seen = [0, 0]
        for i in sm.xrange(nb_iterations):
            observed = aug.augment_image(img)
            uq = np.unique(observed)
            if len(uq) == 1:
                seen[0] += 1
            elif len(uq) == 2:
                seen[1] += 1
            else:
                assert False
        assert 300 - 75 < seen[0] < 300 + 75
        assert 700 - 75 < seen[1] < 700 + 75

    def test_threshold(self):
        arr = np.array([0, 10, 50, 150, 200, 255], dtype=np.uint8)
        arr = arr.reshape((2, 3, 1))
        aug = iaa.Invert(p=1.0, threshold=128, invert_above_threshold=True)

        observed = aug.augment_image(arr)

        expected = np.array([0, 10, 50, 255-150, 255-200, 255-255],
                            dtype=np.uint8).reshape((2, 3, 1))
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_threshold_inv_below(self):
        arr = np.array([0, 10, 50, 150, 200, 255], dtype=np.uint8)
        arr = arr.reshape((2, 3, 1))
        aug = iaa.Invert(p=1.0, threshold=128, invert_above_threshold=False)

        observed = aug.augment_image(arr)

        expected = np.array([255-0, 255-10, 255-50, 150, 200, 255],
                            dtype=np.uint8).reshape((2, 3, 1))
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=zeros.shape)]

        aug = iaa.Invert(p=1.0)
        aug_det = iaa.Invert(p=1.0).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _ = iaa.Invert(p="test")
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _ = iaa.Invert(p=0.5, per_channel="test")
        except Exception:
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
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Invert(1.0)

                image_aug = aug(image=image)

                assert np.all(image_aug == 255)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

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
                aug = iaa.Invert(1.0)

                image_aug = aug(image=image)

                assert np.all(image_aug == 255)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.Invert(p=0.5, per_channel=False, min_value=10, max_value=20)
        params = aug.get_parameters()
        assert params[0] is aug.p
        assert params[1] is aug.per_channel
        assert params[2] == 10
        assert params[3] == 20
        assert params[4] is aug.threshold
        assert params[5] is aug.invert_above_threshold

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.Invert(p=1.0)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_p_is_zero(self):
        # with p=0.0
        aug = iaa.Invert(p=0.0)
        dtypes = [bool,
                  np.uint8, np.uint16, np.uint32, np.uint64,
                  np.int8, np.int16, np.int32, np.int64,
                  np.float16, np.float32, np.float64, np.float128]
        for dtype in dtypes:
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
            kind = np.dtype(dtype).kind
            image_min = np.full((3, 3), min_value, dtype=dtype)
            if dtype is not bool:
                image_center = np.full((3, 3), center_value if kind == "f" else int(center_value), dtype=dtype)
            image_max = np.full((3, 3), max_value, dtype=dtype)
            image_min_aug = aug.augment_image(image_min)
            image_center_aug = None
            if dtype is not bool:
                image_center_aug = aug.augment_image(image_center)
            image_max_aug = aug.augment_image(image_max)

            assert image_min_aug.dtype == np.dtype(dtype)
            if image_center_aug is not None:
                assert image_center_aug.dtype == np.dtype(dtype)
            assert image_max_aug.dtype == np.dtype(dtype)

            if dtype is bool:
                assert np.all(image_min_aug == image_min)
                assert np.all(image_max_aug == image_max)
            elif np.dtype(dtype).kind in ["i", "u"]:
                assert np.array_equal(image_min_aug, image_min)
                assert np.array_equal(image_center_aug, image_center)
                assert np.array_equal(image_max_aug, image_max)
            else:
                assert np.allclose(image_min_aug, image_min)
                assert np.allclose(image_center_aug, image_center)
                assert np.allclose(image_max_aug, image_max)

    def test_other_dtypes_p_is_one(self):
        # with p=1.0
        aug = iaa.Invert(p=1.0)
        dtypes = [np.uint8, np.uint16, np.uint32, np.uint64,
                  np.int8, np.int16, np.int32, np.int64,
                  np.float16, np.float32, np.float64, np.float128]
        for dtype in dtypes:
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
            kind = np.dtype(dtype).kind
            image_min = np.full((3, 3), min_value, dtype=dtype)
            if dtype is not bool:
                image_center = np.full((3, 3), center_value if kind == "f" else int(center_value), dtype=dtype)
            image_max = np.full((3, 3), max_value, dtype=dtype)
            image_min_aug = aug.augment_image(image_min)
            image_center_aug = None
            if dtype is not bool:
                image_center_aug = aug.augment_image(image_center)
            image_max_aug = aug.augment_image(image_max)

            assert image_min_aug.dtype == np.dtype(dtype)
            if image_center_aug is not None:
                assert image_center_aug.dtype == np.dtype(dtype)
            assert image_max_aug.dtype == np.dtype(dtype)

            if dtype is bool:
                assert np.all(image_min_aug == image_max)
                assert np.all(image_max_aug == image_min)
            elif np.dtype(dtype).kind in ["i", "u"]:
                assert np.array_equal(image_min_aug, image_max)
                assert np.allclose(image_center_aug, image_center, atol=1.0+1e-4, rtol=0)
                assert np.array_equal(image_max_aug, image_min)
            else:
                assert np.allclose(image_min_aug, image_max)
                assert np.allclose(image_center_aug, image_center)
                assert np.allclose(image_max_aug, image_min)

    def test_other_dtypes_p_is_one_with_min_value(self):
        # with p=1.0 and min_value
        aug = iaa.Invert(p=1.0, min_value=1)
        dtypes = [np.uint8, np.uint16, np.uint32,
                  np.int8, np.int16, np.int32,
                  np.float16, np.float32]
        for dtype in dtypes:
            _min_value, _center_value, max_value = iadt.get_value_range_of_dtype(dtype)
            min_value = 1
            kind = np.dtype(dtype).kind
            center_value = min_value + 0.5 * (max_value - min_value)
            image_min = np.full((3, 3), min_value, dtype=dtype)
            if dtype is not bool:
                image_center = np.full((3, 3), center_value if kind == "f" else int(center_value), dtype=dtype)
            image_max = np.full((3, 3), max_value, dtype=dtype)
            image_min_aug = aug.augment_image(image_min)
            image_center_aug = None
            if dtype is not bool:
                image_center_aug = aug.augment_image(image_center)
            image_max_aug = aug.augment_image(image_max)

            assert image_min_aug.dtype == np.dtype(dtype)
            if image_center_aug is not None:
                assert image_center_aug.dtype == np.dtype(dtype)
            assert image_max_aug.dtype == np.dtype(dtype)

            if dtype is bool:
                assert np.all(image_min_aug == 1)
                assert np.all(image_max_aug == 1)
            elif np.dtype(dtype).kind in ["i", "u"]:
                assert np.array_equal(image_min_aug, image_max)
                assert np.allclose(image_center_aug, image_center, atol=1.0+1e-4, rtol=0)
                assert np.array_equal(image_max_aug, image_min)
            else:
                assert np.allclose(image_min_aug, image_max)
                assert np.allclose(image_center_aug, image_center)
                assert np.allclose(image_max_aug, image_min)

    def test_other_dtypes_p_is_one_with_max_value(self):
        # with p=1.0 and max_value
        aug = iaa.Invert(p=1.0, max_value=16)
        dtypes = [np.uint8, np.uint16, np.uint32,
                  np.int8, np.int16, np.int32,
                  np.float16, np.float32]
        for dtype in dtypes:
            min_value, _center_value, _max_value = iadt.get_value_range_of_dtype(dtype)
            max_value = 16
            kind = np.dtype(dtype).kind
            center_value = min_value + 0.5 * (max_value - min_value)
            image_min = np.full((3, 3), min_value, dtype=dtype)
            if dtype is not bool:
                image_center = np.full((3, 3), center_value if kind == "f" else int(center_value), dtype=dtype)
            image_max = np.full((3, 3), max_value, dtype=dtype)
            image_min_aug = aug.augment_image(image_min)
            image_center_aug = None
            if dtype is not bool:
                image_center_aug = aug.augment_image(image_center)
            image_max_aug = aug.augment_image(image_max)

            assert image_min_aug.dtype == np.dtype(dtype)
            if image_center_aug is not None:
                assert image_center_aug.dtype == np.dtype(dtype)
            assert image_max_aug.dtype == np.dtype(dtype)

            if dtype is bool:
                assert not np.any(image_min_aug == 1)
                assert not np.any(image_max_aug == 1)
            elif np.dtype(dtype).kind in ["i", "u"]:
                assert np.array_equal(image_min_aug, image_max)
                assert np.allclose(image_center_aug, image_center, atol=1.0+1e-4, rtol=0)
                assert np.array_equal(image_max_aug, image_min)
            else:
                assert np.allclose(image_min_aug, image_max)
                if dtype is np.float16:
                    # for float16, this is off by about 10
                    assert np.allclose(image_center_aug, image_center, atol=0.001*np.finfo(dtype).max)
                else:
                    assert np.allclose(image_center_aug, image_center)
                assert np.allclose(image_max_aug, image_min)

    def test_pickleable(self):
        aug = iaa.Invert(p=0.5, per_channel=True, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=20, shape=(2, 2, 5))


class TestSolarize(unittest.TestCase):
    def test_p_is_one(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        observed = iaa.Solarize(p=1.0).augment_image(zeros)

        expected = zeros
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_p_is_one_some_values_above_threshold(self):
        arr = np.array([0, 99, 111, 200]).astype(np.uint8).reshape((2, 2, 1))

        observed = iaa.Solarize(p=1.0, threshold=(100, 110))(image=arr)

        expected = np.array([0, 99, 255-111, 255-200])\
            .astype(np.uint8).reshape((2, 2, 1))
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)


class TestContrastNormalization(unittest.TestCase):
    @unittest.skipIf(sys.version_info[0] <= 2,
                     "Warning is not generated in 2.7 on travis, but locally "
                     "in 2.7 it is?!")
    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            aug = arithmetic_lib.ContrastNormalization((0.9, 1.1))
            assert isinstance(aug, contrast_lib._ContrastFuncWrapper)

        assert len(caught_warnings) == 1
        assert (
            "deprecated"
            in str(caught_warnings[-1].message)
        )


# TODO use this in test_contrast.py or remove it?
"""
def deactivated_test_ContrastNormalization():
    reseed()

    zeros = np.zeros((4, 4, 3), dtype=np.uint8)
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=zeros.shape)]

    # contrast stays the same
    observed = iaa.ContrastNormalization(alpha=1.0).augment_image(zeros + 50)
    expected = zeros + 50
    assert np.array_equal(observed, expected)

    # image with mean intensity (ie 128), contrast cannot be changed
    observed = iaa.ContrastNormalization(alpha=2.0).augment_image(zeros + 128)
    expected = zeros + 128
    assert np.array_equal(observed, expected)

    # increase contrast
    observed = iaa.ContrastNormalization(alpha=2.0).augment_image(zeros + 128 + 10)
    expected = zeros + 128 + 20
    assert np.array_equal(observed, expected)

    observed = iaa.ContrastNormalization(alpha=2.0).augment_image(zeros + 128 - 10)
    expected = zeros + 128 - 20
    assert np.array_equal(observed, expected)

    # decrease contrast
    observed = iaa.ContrastNormalization(alpha=0.5).augment_image(zeros + 128 + 10)
    expected = zeros + 128 + 5
    assert np.array_equal(observed, expected)

    observed = iaa.ContrastNormalization(alpha=0.5).augment_image(zeros + 128 - 10)
    expected = zeros + 128 - 5
    assert np.array_equal(observed, expected)

    # increase contrast by stochastic parameter
    observed = iaa.ContrastNormalization(alpha=iap.Choice([2.0, 3.0])).augment_image(zeros + 128 + 10)
    expected1 = zeros + 128 + 20
    expected2 = zeros + 128 + 30
    assert np.array_equal(observed, expected1) or np.array_equal(observed, expected2)

    # change contrast by tuple
    nb_iterations = 1000
    nb_changed = 0
    last = None
    for i in sm.xrange(nb_iterations):
        observed = iaa.ContrastNormalization(alpha=(0.5, 2.0)).augment_image(zeros + 128 + 40)
        if last is None:
            last = observed
        else:
            if not np.array_equal(observed, last):
                nb_changed += 1
    p_changed = nb_changed / (nb_iterations-1)
    assert p_changed > 0.5

    # per_channel=True
    aug = iaa.ContrastNormalization(alpha=(1.0, 6.0), per_channel=True)
    img = np.zeros((1, 1, 100), dtype=np.uint8) + 128 + 10
    observed = aug.augment_image(img)
    uq = np.unique(observed)
    assert len(uq) > 5

    # per_channel with probability
    aug = iaa.ContrastNormalization(alpha=(1.0, 4.0), per_channel=0.7)
    img = np.zeros((1, 1, 100), dtype=np.uint8) + 128 + 10
    seen = [0, 0]
    for _ in sm.xrange(1000):
        observed = aug.augment_image(img)
        uq = np.unique(observed)
        if len(uq) == 1:
            seen[0] += 1
        elif len(uq) >= 2:
            seen[1] += 1
    assert 300 - 75 < seen[0] < 300 + 75
    assert 700 - 75 < seen[1] < 700 + 75

    # keypoints shouldnt be changed
    aug = iaa.ContrastNormalization(alpha=2.0)
    aug_det = iaa.ContrastNormalization(alpha=2.0).to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        _ = iaa.ContrastNormalization(alpha="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.ContrastNormalization(alpha=1.5, per_channel="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.ContrastNormalization(alpha=1, per_channel=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert params[0].value == 1
    assert params[1].value == 0

    # test heatmaps (not affected by augmenter)
    aug = iaa.ContrastNormalization(alpha=2)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)
"""


class TestJpegCompression(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_compression_is_zero(self):
        # basic test at 0 compression
        img = ia.quokka(extract="square", size=(64, 64))
        aug = iaa.JpegCompression(0)
        img_aug = aug.augment_image(img)
        diff = np.average(np.abs(img.astype(np.float32) - img_aug.astype(np.float32)))
        assert diff < 1.0

    def test_compression_is_90(self):
        # basic test at 90 compression
        img = ia.quokka(extract="square", size=(64, 64))
        aug = iaa.JpegCompression(90)
        img_aug = aug.augment_image(img)
        diff = np.average(np.abs(img.astype(np.float32) - img_aug.astype(np.float32)))
        assert 1.0 < diff < 50.0

    def test___init__(self):
        aug = iaa.JpegCompression([0, 100])
        assert isinstance(aug.compression, iap.Choice)
        assert len(aug.compression.a) == 2
        assert aug.compression.a[0] == 0
        assert aug.compression.a[1] == 100

    def test_get_parameters(self):
        aug = iaa.JpegCompression([0, 100])
        assert len(aug.get_parameters()) == 1
        assert aug.get_parameters()[0] == aug.compression

    def test_compression_is_stochastic_parameter(self):
        # test if stochastic parameters are used by augmentation
        img = ia.quokka(extract="square", size=(64, 64))

        class _TwoValueParam(iap.StochasticParameter):
            def __init__(self, v1, v2):
                super(_TwoValueParam, self).__init__()
                self.v1 = v1
                self.v2 = v2

            def _draw_samples(self, size, random_state):
                arr = np.full(size, self.v1, dtype=np.float32)
                arr[1::2] = self.v2
                return arr

        param = _TwoValueParam(0, 100)
        aug = iaa.JpegCompression(param)
        img_aug_c0 = iaa.JpegCompression(0).augment_image(img)
        img_aug_c100 = iaa.JpegCompression(100).augment_image(img)
        imgs_aug = aug.augment_images([img] * 4)
        assert np.array_equal(imgs_aug[0], img_aug_c0)
        assert np.array_equal(imgs_aug[1], img_aug_c100)
        assert np.array_equal(imgs_aug[2], img_aug_c0)
        assert np.array_equal(imgs_aug[3], img_aug_c100)

    def test_keypoints_dont_change(self):
        # test keypoints (not affected by augmenter)
        aug = iaa.JpegCompression(50)
        kps = ia.quokka_keypoints()
        kps_aug = aug.augment_keypoints([kps])[0]
        for kp, kp_aug in zip(kps.keypoints, kps_aug.keypoints):
            assert np.allclose([kp.x, kp.y], [kp_aug.x, kp_aug.y])

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.JpegCompression(50)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0, 3),
            (0, 1, 3),
            (1, 0, 3)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.JpegCompression(100)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_pickleable(self):
        aug = iaa.JpegCompression((0, 100), seed=1)
        runtest_pickleable_uint8_img(aug, iterations=20)
