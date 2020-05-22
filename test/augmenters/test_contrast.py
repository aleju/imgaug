from __future__ import print_function, division, absolute_import

import itertools
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
import warnings

import numpy as np
import six.moves as sm
import skimage
import skimage.data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.augmenters import contrast as contrast_lib
from imgaug.testutils import (ArgCopyingMagicMock, keypoints_equal, reseed,
                              runtest_pickleable_uint8_img, assertWarns,
                              is_parameter_instance)
from imgaug.augmentables.batches import _BatchInAugmentation


class TestGammaContrast(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___tuple_to_uniform(self):
        aug = iaa.GammaContrast((1, 2))
        assert is_parameter_instance(aug.params1d[0], iap.Uniform)
        assert is_parameter_instance(aug.params1d[0].a, iap.Deterministic)
        assert is_parameter_instance(aug.params1d[0].b, iap.Deterministic)
        assert aug.params1d[0].a.value == 1
        assert aug.params1d[0].b.value == 2

    def test___init___list_to_choice(self):
        aug = iaa.GammaContrast([1, 2])
        assert is_parameter_instance(aug.params1d[0], iap.Choice)
        assert np.all([val in aug.params1d[0].a for val in [1, 2]])

    def test_images_basic_functionality(self):
        img = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        img = np.uint8(img)
        img3d = np.tile(img[:, :, np.newaxis], (1, 1, 3))

        # check basic functionality with gamma=1 or 2 (deterministic) and
        # per_channel on/off (makes
        # no difference due to deterministic gamma)
        for per_channel in [False, 0, 0.0, True, 1, 1.0]:
            for gamma in [1, 2]:
                aug = iaa.GammaContrast(
                    gamma=iap.Deterministic(gamma),
                    per_channel=per_channel)
                img_aug = aug.augment_image(img)
                img3d_aug = aug.augment_image(img3d)
                assert img_aug.dtype.name == "uint8"
                assert img3d_aug.dtype.name == "uint8"
                assert np.array_equal(
                    img_aug,
                    skimage.exposure.adjust_gamma(img, gamma=gamma))
                assert np.array_equal(
                    img3d_aug,
                    skimage.exposure.adjust_gamma(img3d, gamma=gamma))

    def test_per_channel_is_float(self):
        # check that per_channel at 50% prob works
        aug = iaa.GammaContrast((0.5, 2.0), per_channel=0.5)
        seen = [False, False]
        img1000d = np.zeros((1, 1, 1000), dtype=np.uint8) + 128
        for _ in sm.xrange(100):
            img_aug = aug.augment_image(img1000d)
            assert img_aug.dtype.name == "uint8"
            nb_values_uq = len(set(img_aug.flatten().tolist()))
            if nb_values_uq == 1:
                seen[0] = True
            else:
                seen[1] = True
            if np.all(seen):
                break
        assert np.all(seen)

    def test_keypoints_not_changed(self):
        aug = iaa.GammaContrast(gamma=2)
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
        kpsoi_aug = aug.augment_keypoints([kpsoi])
        assert keypoints_equal([kpsoi], kpsoi_aug)

    def test_heatmaps_not_changed(self):
        aug = iaa.GammaContrast(gamma=2)
        heatmaps_arr = np.zeros((3, 3, 1), dtype=np.float32) + 0.5
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))
        heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
        assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

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
                aug = iaa.GammaContrast(0.5)

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
                aug = iaa.GammaContrast(0.5)

                image_aug = aug(image=image)

                assert np.any(image_aug != 128)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_other_dtypes_uint_int(self):
        try:
            high_res_dt = np.float128
            dts = [np.uint8, np.uint16, np.uint32, np.uint64,
                   np.int8, np.int16, np.int32, np.int64]
        except AttributeError:
            # cannot reliably check uint64 and int64 on systems that dont
            # support float128
            high_res_dt = np.float64
            dts = [np.uint8, np.uint16, np.uint32,
                   np.int8, np.int16, np.int32]

        for dtype in dts:
            dtype = np.dtype(dtype)

            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            exps = [1, 2, 3]
            values = [0, 100, int(center_value + 0.1*max_value)]
            tolerances = [0, 0, 1e-8 * max_value
                          if dtype
                          in [np.uint64, np.int64] else 0]

            for exp in exps:
                aug = iaa.GammaContrast(exp)
                for value, tolerance in zip(values, tolerances):
                    with self.subTest(dtype=dtype.name, exp=exp,
                                      nb_channels=None):
                        image = np.full((3, 3), value, dtype=dtype)
                        expected = (
                            (
                                (image.astype(high_res_dt) / max_value)
                                ** exp
                            ) * max_value
                        ).astype(dtype)
                        image_aug = aug.augment_image(image)
                        value_aug = int(image_aug[0, 0])
                        value_expected = int(expected[0, 0])
                        diff = abs(value_aug - value_expected)
                        assert image_aug.dtype.name == dtype.name
                        assert len(np.unique(image_aug)) == 1
                        assert diff <= tolerance

                    # test other channel numbers
                    for nb_channels in [1, 2, 3, 4, 5, 7, 11]:
                        with self.subTest(dtype=dtype.name, exp=exp,
                                          nb_channels=nb_channels):
                            image = np.full((3, 3), value, dtype=dtype)
                            image = np.tile(image[..., np.newaxis],
                                            (1, 1, nb_channels))
                            for c in sm.xrange(nb_channels):
                                image[..., c] += c
                            expected = (
                                (
                                    (image.astype(high_res_dt) / max_value)
                                    ** exp
                                ) * max_value
                            ).astype(dtype)
                            image_aug = aug.augment_image(image)
                            assert image_aug.shape == (3, 3, nb_channels)
                            assert image_aug.dtype.name == dtype.name
                            # can be less than nb_channels when multiple input
                            # values map to the same output value
                            # mapping distribution can behave exponential with
                            # slow start and fast growth at the end
                            assert len(np.unique(image_aug)) <= nb_channels
                            for c in sm.xrange(nb_channels):
                                value_aug = int(image_aug[0, 0, c])
                                value_expected = int(expected[0, 0, c])
                                diff = abs(value_aug - value_expected)
                                assert diff <= tolerance

    def test_other_dtypes_float(self):
        dts = [np.float16, np.float32, np.float64]
        for dtype in dts:
            dtype = np.dtype(dtype)

            def _allclose(a, b):
                atol = 1e-3 if dtype == np.float16 else 1e-8
                return np.allclose(a, b, atol=atol, rtol=0)

            exps = [1, 2]
            isize = np.dtype(dtype).itemsize
            values = [0, 1.0, 50.0, 100 ** (isize - 1)]

            for exp in exps:
                aug = iaa.GammaContrast(exp)
                for value in values:
                    with self.subTest(dtype=dtype.name, exp=exp,
                                      nb_channels=None):
                        image = np.full((3, 3), value, dtype=dtype)
                        expected = (
                            image.astype(np.float64)
                            ** exp
                        ).astype(dtype)
                        image_aug = aug.augment_image(image)
                        assert image_aug.dtype == np.dtype(dtype)
                        assert _allclose(image_aug, expected)

                    # test other channel numbers
                    for nb_channels in [1, 2, 3, 4, 5, 7, 11]:
                        with self.subTest(dtype=dtype.name, exp=exp,
                                          nb_channels=nb_channels):
                            image = np.full((3, 3), value, dtype=dtype)
                            image = np.tile(image[..., np.newaxis],
                                            (1, 1, nb_channels))
                            for c in sm.xrange(nb_channels):
                                image[..., c] += float(c)
                            expected = (
                                image.astype(np.float64)
                                ** exp
                            ).astype(dtype)
                            image_aug = aug.augment_image(image)
                            assert image_aug.shape == (3, 3, nb_channels)
                            assert image_aug.dtype.name == dtype.name
                            for c in sm.xrange(nb_channels):
                                value_aug = image_aug[0, 0, c]
                                value_expected = expected[0, 0, c]
                                assert _allclose(value_aug, value_expected)

    def test_pickleable(self):
        aug = iaa.GammaContrast((0.5, 2.0), seed=1)
        runtest_pickleable_uint8_img(aug, iterations=20)


class TestSigmoidContrast(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___tuple_to_uniform(self):
        # check that tuple to uniform works
        # note that gain and cutoff are saved in inverted order in
        # _ContrastFuncWrapper to match the order of skimage's function
        aug = iaa.SigmoidContrast(gain=(1, 2), cutoff=(0.25, 0.75))
        assert is_parameter_instance(aug.params1d[0], iap.Uniform)
        assert is_parameter_instance(aug.params1d[0].a, iap.Deterministic)
        assert is_parameter_instance(aug.params1d[0].b, iap.Deterministic)
        assert aug.params1d[0].a.value == 1
        assert aug.params1d[0].b.value == 2
        assert is_parameter_instance(aug.params1d[1], iap.Uniform)
        assert is_parameter_instance(aug.params1d[1].a, iap.Deterministic)
        assert is_parameter_instance(aug.params1d[1].b, iap.Deterministic)
        assert np.allclose(aug.params1d[1].a.value, 0.25)
        assert np.allclose(aug.params1d[1].b.value, 0.75)

    def test___init___list_to_choice(self):
        # check that list to choice works
        # note that gain and cutoff are saved in inverted order in
        # _ContrastFuncWrapper to match the order of skimage's function
        aug = iaa.SigmoidContrast(gain=[1, 2], cutoff=[0.25, 0.75])
        assert is_parameter_instance(aug.params1d[0], iap.Choice)
        assert np.all([val in aug.params1d[0].a for val in [1, 2]])
        assert is_parameter_instance(aug.params1d[1], iap.Choice)
        assert np.all([
            np.allclose(val, val_choice)
            for val, val_choice
            in zip([0.25, 0.75], aug.params1d[1].a)])

    def test_images_basic_functionality(self):
        img = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        img = np.uint8(img)
        img3d = np.tile(img[:, :, np.newaxis], (1, 1, 3))

        # check basic functionality with per_chanenl on/off (makes no
        # difference due to deterministic parameters)
        for per_channel in [False, 0, 0.0, True, 1, 1.0]:
            for gain, cutoff in itertools.product([5, 10], [0.25, 0.75]):
                with self.subTest(gain=gain, cutoff=cutoff,
                                  per_channel=per_channel):
                    aug = iaa.SigmoidContrast(
                        gain=iap.Deterministic(gain),
                        cutoff=iap.Deterministic(cutoff),
                        per_channel=per_channel)
                    img_aug = aug.augment_image(img)
                    img3d_aug = aug.augment_image(img3d)
                    assert img_aug.dtype.name == "uint8"
                    assert img3d_aug.dtype.name == "uint8"
                    assert np.array_equal(
                        img_aug,
                        skimage.exposure.adjust_sigmoid(
                            img, gain=gain, cutoff=cutoff))
                    assert np.array_equal(
                        img3d_aug,
                        skimage.exposure.adjust_sigmoid(
                            img3d, gain=gain, cutoff=cutoff))

    def test_per_channel_is_float(self):
        # check that per_channel at 50% prob works
        aug = iaa.SigmoidContrast(gain=(1, 10),
                                  cutoff=(0.25, 0.75),
                                  per_channel=0.5)
        seen = [False, False]
        img1000d = np.zeros((1, 1, 1000), dtype=np.uint8) + 128
        for _ in sm.xrange(100):
            img_aug = aug.augment_image(img1000d)
            assert img_aug.dtype.name == "uint8"
            nb_values_uq = len(set(img_aug.flatten().tolist()))
            if nb_values_uq == 1:
                seen[0] = True
            else:
                seen[1] = True
            if np.all(seen):
                break
        assert np.all(seen)

    def test_keypoints_dont_change(self):
        aug = iaa.SigmoidContrast(gain=10, cutoff=0.5)
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
        kpsoi_aug = aug.augment_keypoints([kpsoi])
        assert keypoints_equal([kpsoi], kpsoi_aug)

    def test_heatmaps_dont_change(self):
        aug = iaa.SigmoidContrast(gain=10, cutoff=0.5)
        heatmaps_arr = np.zeros((3, 3, 1), dtype=np.float32) + 0.5
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))
        heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
        assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

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
                aug = iaa.SigmoidContrast(gain=10, cutoff=0.5)

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
                aug = iaa.SigmoidContrast(gain=10, cutoff=1.0)

                image_aug = aug(image=image)

                assert np.any(image_aug != 128)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_other_dtypes_uint_int(self):
        try:
            high_res_dt = np.float128
            dtypes = [np.uint8, np.uint16, np.uint32, np.uint64,
                      np.int8, np.int16, np.int32, np.int64]
        except AttributeError:
            # cannot reliably check uint64 and int64 on systems that dont
            # support float128
            high_res_dt = np.float64
            dtypes = [np.uint8, np.uint16, np.uint32,
                      np.int8, np.int16, np.int32]

        for dtype in dtypes:
            dtype = np.dtype(dtype)

            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            gains = [5, 20]
            cutoffs = [0.25, 0.75]
            values = [0, 100, int(center_value + 0.1 * max_value)]
            tmax = 1e-8 * max_value if dtype in [np.uint64, np.int64] else 0
            tolerances = [tmax, tmax, tmax]

            for gain, cutoff in itertools.product(gains, cutoffs):
                with self.subTest(dtype=dtype.name, gain=gain, cutoff=cutoff):
                    aug = iaa.SigmoidContrast(gain=gain, cutoff=cutoff)
                    for value, tolerance in zip(values, tolerances):
                        image = np.full((3, 3), value, dtype=dtype)
                        # TODO this looks like the equation commented out
                        #      should actually the correct one, but when using
                        #      it we get a difference between expectation and
                        #      skimage ground truth
                        # 1/(1 + exp(gain*(cutoff - I_ij/max)))
                        expected = (
                            1
                            / (
                                1
                                + np.exp(
                                    gain
                                    * (
                                        cutoff
                                        - image.astype(high_res_dt)/max_value
                                    )
                                )
                            )
                        )
                        expected = (expected * max_value).astype(dtype)
                        # expected = (
                        #   1/(1 + np.exp(gain * (
                        #       cutoff - (
                        #           image.astype(high_res_dt)-min_value
                        #       )/dynamic_range
                        #   ))))
                        # expected = (
                        #   min_value + expected * dynamic_range).astype(dtype)
                        image_aug = aug.augment_image(image)
                        value_aug = int(image_aug[0, 0])
                        value_expected = int(expected[0, 0])
                        diff = abs(value_aug - value_expected)
                        assert image_aug.dtype.name == dtype.name
                        assert len(np.unique(image_aug)) == 1
                        assert diff <= tolerance

    def test_other_dtypes_float(self):
        dtypes = [np.float16, np.float32, np.float64]
        for dtype in dtypes:
            dtype = np.dtype(dtype)

            def _allclose(a, b):
                atol = 1e-3 if dtype == np.float16 else 1e-8
                return np.allclose(a, b, atol=atol, rtol=0)

            gains = [5, 20]
            cutoffs = [0.25, 0.75]
            isize = np.dtype(dtype).itemsize
            values = [0, 1.0, 50.0, 100 ** (isize - 1)]

            for gain, cutoff in itertools.product(gains, cutoffs):
                with self.subTest(dtype=dtype, gain=gain, cutoff=cutoff):
                    aug = iaa.SigmoidContrast(gain=gain, cutoff=cutoff)
                    for value in values:
                        image = np.full((3, 3), value, dtype=dtype)
                        expected = (
                            1
                            / (
                                1
                                + np.exp(
                                    gain
                                    * (
                                        cutoff
                                        - image.astype(np.float64)
                                    )
                                )
                            )
                        ).astype(dtype)
                        image_aug = aug.augment_image(image)
                        assert image_aug.dtype.name == dtype.name
                        assert _allclose(image_aug, expected)

    def test_pickleable(self):
        aug = iaa.SigmoidContrast(gain=(1, 2), seed=1)
        runtest_pickleable_uint8_img(aug, iterations=20)


class TestLogContrast(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_images_basic_functionality(self):
        img = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        img = np.uint8(img)
        img3d = np.tile(img[:, :, np.newaxis], (1, 1, 3))

        # check basic functionality with gain=1 or 2 (deterministic) and
        # per_channel on/off (makes no difference due to deterministic gain)
        for per_channel in [False, 0, 0.0, True, 1, 1.0]:
            for gain in [1, 2]:
                with self.subTest(gain=gain, per_channel=per_channel):
                    aug = iaa.LogContrast(
                        gain=iap.Deterministic(gain),
                        per_channel=per_channel)
                    img_aug = aug.augment_image(img)
                    img3d_aug = aug.augment_image(img3d)
                    assert img_aug.dtype.name == "uint8"
                    assert img3d_aug.dtype.name == "uint8"
                    assert np.array_equal(
                        img_aug,
                        skimage.exposure.adjust_log(img, gain=gain))
                    assert np.array_equal(
                        img3d_aug,
                        skimage.exposure.adjust_log(img3d, gain=gain))

    def test___init___tuple_to_uniform(self):
        aug = iaa.LogContrast((1, 2))
        assert is_parameter_instance(aug.params1d[0], iap.Uniform)
        assert is_parameter_instance(aug.params1d[0].a, iap.Deterministic)
        assert is_parameter_instance(aug.params1d[0].b, iap.Deterministic)
        assert aug.params1d[0].a.value == 1
        assert aug.params1d[0].b.value == 2

    def test___init___list_to_choice(self):
        aug = iaa.LogContrast([1, 2])
        assert is_parameter_instance(aug.params1d[0], iap.Choice)
        assert np.all([val in aug.params1d[0].a for val in [1, 2]])

    def test_per_channel_is_float(self):
        # check that per_channel at 50% prob works
        aug = iaa.LogContrast((0.5, 2.0), per_channel=0.5)
        seen = [False, False]
        img1000d = np.zeros((1, 1, 1000), dtype=np.uint8) + 128
        for _ in sm.xrange(100):
            img_aug = aug.augment_image(img1000d)
            assert img_aug.dtype.name == "uint8"
            nb_values_uq = len(set(img_aug.flatten().tolist()))
            if nb_values_uq == 1:
                seen[0] = True
            else:
                seen[1] = True
            if np.all(seen):
                break
        assert np.all(seen)

    def test_keypoints_not_changed(self):
        aug = iaa.LogContrast(gain=2)
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
        kpsoi_aug = aug.augment_keypoints([kpsoi])
        assert keypoints_equal([kpsoi], kpsoi_aug)

    def test_heatmaps_not_changed(self):
        aug = iaa.LogContrast(gain=2)
        heatmap_arr = np.zeros((3, 3, 1), dtype=np.float32) + 0.5
        heatmaps = ia.HeatmapsOnImage(heatmap_arr, shape=(3, 3, 3))
        heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
        assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

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
                aug = iaa.LogContrast(gain=2)

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
                aug = iaa.LogContrast(gain=2)

                image_aug = aug(image=image)

                assert np.any(image_aug != 128)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_other_dtypes_uint_int(self):
        # support before 1.17:
        #   [np.uint8, np.uint16, np.uint32, np.uint64,
        #    np.int8, np.int16, np.int32, np.int64]
        # support beginning with 1.17:
        #   [np.uint8, np.uint16,
        #    np.int8, np.int16]
        # uint, int
        dtypes = [np.uint8, np.uint16, np.int8, np.int16]

        for dtype in dtypes:
            dtype = np.dtype(dtype)

            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            gains = [0.5, 0.75, 1.0, 1.1]
            values = [0, 100, int(center_value + 0.1 * max_value)]
            tmax = 1e-8 * max_value if dtype in [np.uint64, np.int64] else 0
            tolerances = [0, tmax, tmax]

            for gain in gains:
                aug = iaa.LogContrast(gain)
                for value, tolerance in zip(values, tolerances):
                    with self.subTest(dtype=dtype.name, gain=gain):
                        image = np.full((3, 3), value, dtype=dtype)
                        expected = (
                            gain
                            * np.log2(
                                1 + (image.astype(np.float64)/max_value)
                            )
                        )
                        expected = (expected*max_value).astype(dtype)
                        image_aug = aug.augment_image(image)
                        value_aug = int(image_aug[0, 0])
                        value_expected = int(expected[0, 0])
                        diff = abs(value_aug - value_expected)
                        assert image_aug.dtype.name == dtype.name
                        assert len(np.unique(image_aug)) == 1
                        assert diff <= tolerance

    def test_other_dtypes_float(self):
        dtypes = [np.float16, np.float32, np.float64]

        for dtype in dtypes:
            dtype = np.dtype(dtype)

            def _allclose(a, b):
                # since numpy 1.17 this needs for some reason at least 1e-5 as
                # the tolerance, previously 1e-8 worked
                atol = 1e-2 if dtype == np.float16 else 1e-5
                return np.allclose(a, b, atol=atol, rtol=0)

            gains = [0.5, 0.75, 1.0, 1.1]
            isize = np.dtype(dtype).itemsize
            values = [0, 1.0, 50.0, 100 ** (isize - 1)]

            for gain in gains:
                aug = iaa.LogContrast(gain)
                for value in values:
                    with self.subTest(dtype=dtype.name, gain=gain):
                        image = np.full((3, 3), value, dtype=dtype)
                        expected = (
                            gain
                            * np.log2(
                                1 + image.astype(np.float64)
                            )
                        )
                        expected = expected.astype(dtype)
                        image_aug = aug.augment_image(image)
                        assert image_aug.dtype.name == dtype
                        assert _allclose(image_aug, expected)

    def test_pickleable(self):
        aug = iaa.LogContrast((1, 2), seed=1)
        runtest_pickleable_uint8_img(aug, iterations=20)


class TestLinearContrast(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_images_basic_functionality(self):
        img = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        img = np.uint8(img)
        img3d = np.tile(img[:, :, np.newaxis], (1, 1, 3))

        # check basic functionality with alpha=1 or 2 (deterministic) and
        # per_channel on/off (makes no difference due to deterministic alpha)
        for per_channel in [False, 0, 0.0, True, 1, 1.0]:
            for alpha in [1, 2]:
                with self.subTest(alpha=alpha, per_channel=per_channel):
                    aug = iaa.LinearContrast(
                        alpha=iap.Deterministic(alpha),
                        per_channel=per_channel)
                    img_aug = aug.augment_image(img)
                    img3d_aug = aug.augment_image(img3d)
                    assert img_aug.dtype.name == "uint8"
                    assert img3d_aug.dtype.name == "uint8"
                    assert np.array_equal(
                        img_aug,
                        contrast_lib.adjust_contrast_linear(img, alpha=alpha))
                    assert np.array_equal(
                        img3d_aug,
                        contrast_lib.adjust_contrast_linear(img3d, alpha=alpha))

    def test___init___tuple_to_uniform(self):
        aug = iaa.LinearContrast((1, 2))
        assert is_parameter_instance(aug.params1d[0], iap.Uniform)
        assert is_parameter_instance(aug.params1d[0].a, iap.Deterministic)
        assert is_parameter_instance(aug.params1d[0].b, iap.Deterministic)
        assert aug.params1d[0].a.value == 1
        assert aug.params1d[0].b.value == 2

    def test___init___list_to_choice(self):
        aug = iaa.LinearContrast([1, 2])
        assert is_parameter_instance(aug.params1d[0], iap.Choice)
        assert np.all([val in aug.params1d[0].a for val in [1, 2]])

    def test_float_as_per_channel(self):
        # check that per_channel at 50% prob works
        aug = iaa.LinearContrast((0.5, 2.0), per_channel=0.5)
        seen = [False, False]
        # must not use just value 128 here, otherwise nothing will change as
        # all values would have distance 0 to 128
        img1000d = np.zeros((1, 1, 1000), dtype=np.uint8) + 128 + 20
        for _ in sm.xrange(100):
            img_aug = aug.augment_image(img1000d)
            assert img_aug.dtype.name == "uint8"
            nb_values_uq = len(set(img_aug.flatten().tolist()))
            if nb_values_uq == 1:
                seen[0] = True
            else:
                seen[1] = True
            if np.all(seen):
                break
        assert np.all(seen)

    def test_keypoints_not_changed(self):
        aug = iaa.LinearContrast(alpha=2)
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
        kpsoi_aug = aug.augment_keypoints([kpsoi])
        assert keypoints_equal([kpsoi], kpsoi_aug)

    def test_heatmaps_not_changed(self):
        aug = iaa.LinearContrast(alpha=2)
        heatmaps_arr = np.zeros((3, 3, 1), dtype=np.float32) + 0.5
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))
        heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
        assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

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
                image = np.full(shape, 129, dtype=np.uint8)
                aug = iaa.LinearContrast(alpha=2)

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
                image = np.full(shape, 129, dtype=np.uint8)
                aug = iaa.LinearContrast(alpha=2)

                image_aug = aug(image=image)

                assert np.any(image_aug != 128)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    # test for other dtypes are in Test_adjust_contrast_linear

    def test_pickleable(self):
        aug = iaa.LinearContrast((0.5, 2.0), seed=1)
        runtest_pickleable_uint8_img(aug, iterations=20)


class Test_adjust_contrast_linear(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_other_dtypes(self):
        dtypes = [np.uint8, np.uint16, np.uint32,
                  np.int8, np.int16, np.int32,
                  np.float16, np.float32, np.float64]

        for dtype in dtypes:
            dtype = np.dtype(dtype)

            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)
            cv = center_value
            kind = np.dtype(dtype).kind
            if kind in ["u", "i"]:
                cv = int(cv)

            def _compare(a, b):
                if kind in ["u", "i"]:
                    return np.array_equal(a, b)
                else:
                    assert kind == "f"
                    atol = 1e-4 if dtype == np.float16 else 1e-8
                    return np.allclose(a, b, atol=atol, rtol=0)

            img = [
                [cv-4, cv-3, cv-2],
                [cv-1, cv+0, cv+1],
                [cv+2, cv+3, cv+4]
            ]
            img = np.array(img, dtype=dtype)

            alphas = [0, 1, 2, 4]
            if dtype.name not in ["uint8", "int8", "float16"]:
                alphas.append(100)

            for alpha in alphas:
                expected = [
                    [cv-4*alpha, cv-3*alpha, cv-2*alpha],
                    [cv-1*alpha, cv+0*alpha, cv+1*alpha],
                    [cv+2*alpha, cv+3*alpha, cv+4*alpha]
                ]

                expected = np.array(expected, dtype=dtype)
                observed = contrast_lib.adjust_contrast_linear(
                    img, alpha=alpha)
                assert observed.dtype.name == dtype.name
                assert observed.shape == img.shape
                assert _compare(observed, expected)

    def test_output_values_exceed_uint8_value_range(self):
        cv = 127
        img = [
            [cv-4, cv-3, cv-2],
            [cv-1, cv+0, cv+1],
            [cv+2, cv+3, cv+4]
        ]
        img = np.array(img, dtype=np.uint8)
        observed = contrast_lib.adjust_contrast_linear(img, alpha=255)
        expected = [
            [0, 0, 0],
            [0, cv, 255],
            [255, 255, 255]
        ]
        assert np.array_equal(observed, expected)

    def test_alpha_exceeds_uint8_value_range(self):
        # overflow in alpha for uint8, should not cause issues
        cv = 127
        img = [
            [cv, cv, cv],
            [cv, cv, cv],
            [cv, cv, cv]
        ]
        img = np.array(img, dtype=np.uint8)
        observed = contrast_lib.adjust_contrast_linear(img, alpha=257)
        expected = [
            [cv, cv, cv],
            [cv, cv, cv],
            [cv, cv, cv]
        ]
        assert np.array_equal(observed, expected)


class TestAllChannelsCLAHE(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_init(self):
        aug = iaa.AllChannelsCLAHE(
            clip_limit=10,
            tile_grid_size_px=11,
            tile_grid_size_px_min=4,
            per_channel=True)
        assert is_parameter_instance(aug.clip_limit, iap.Deterministic)
        assert aug.clip_limit.value == 10
        assert is_parameter_instance(aug.tile_grid_size_px[0],
                                     iap.Deterministic)
        assert aug.tile_grid_size_px[0].value == 11
        assert aug.tile_grid_size_px[1] is None
        assert aug.tile_grid_size_px_min == 4
        assert is_parameter_instance(aug.per_channel, iap.Deterministic)
        assert np.isclose(aug.per_channel.value, 1.0)

        aug = iaa.AllChannelsCLAHE(
            clip_limit=(10, 20),
            tile_grid_size_px=(11, 17),
            tile_grid_size_px_min=4,
            per_channel=0.5)
        assert is_parameter_instance(aug.clip_limit, iap.Uniform)
        assert aug.clip_limit.a.value == 10
        assert aug.clip_limit.b.value == 20
        assert is_parameter_instance(aug.tile_grid_size_px[0],
                                     iap.DiscreteUniform)
        assert aug.tile_grid_size_px[0].a.value == 11
        assert aug.tile_grid_size_px[0].b.value == 17
        assert aug.tile_grid_size_px[1] is None
        assert aug.tile_grid_size_px_min == 4
        assert is_parameter_instance(aug.per_channel, iap.Binomial)
        assert np.isclose(aug.per_channel.p.value, 0.5)

        aug = iaa.AllChannelsCLAHE(
            clip_limit=[10, 20, 30],
            tile_grid_size_px=[11, 17, 21])
        assert is_parameter_instance(aug.clip_limit, iap.Choice)
        assert aug.clip_limit.a[0] == 10
        assert aug.clip_limit.a[1] == 20
        assert aug.clip_limit.a[2] == 30
        assert is_parameter_instance(aug.tile_grid_size_px[0], iap.Choice)
        assert aug.tile_grid_size_px[0].a[0] == 11
        assert aug.tile_grid_size_px[0].a[1] == 17
        assert aug.tile_grid_size_px[0].a[2] == 21
        assert aug.tile_grid_size_px[1] is None

        aug = iaa.AllChannelsCLAHE(tile_grid_size_px=((11, 17), [11, 13, 15]))
        assert is_parameter_instance(aug.tile_grid_size_px[0], iap.DiscreteUniform)
        assert aug.tile_grid_size_px[0].a.value == 11
        assert aug.tile_grid_size_px[0].b.value == 17
        assert is_parameter_instance(aug.tile_grid_size_px[1], iap.Choice)
        assert aug.tile_grid_size_px[1].a[0] == 11
        assert aug.tile_grid_size_px[1].a[1] == 13
        assert aug.tile_grid_size_px[1].a[2] == 15

    def test_basic_functionality(self):
        img = [
            [99, 100, 101],
            [99, 100, 101],
            [99, 100, 101]
        ]
        img = np.uint8(img)
        img3d = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        img3d[..., 1] += 1
        img3d[..., 2] += 2

        aug = iaa.AllChannelsCLAHE(clip_limit=20, tile_grid_size_px=17)

        mock_clahe = ArgCopyingMagicMock()
        mock_clahe.apply.return_value = img

        # image with single channel
        with mock.patch('cv2.createCLAHE') as mock_createCLAHE:
            mock_createCLAHE.return_value = mock_clahe
            _ = aug.augment_image(img)

        mock_createCLAHE.assert_called_once_with(
            clipLimit=20, tileGridSize=(17, 17))
        assert np.array_equal(mock_clahe.apply.call_args_list[0][0][0], img)

    def test_basic_functionality_3d(self):
        # image with three channels
        img = [
            [99, 100, 101],
            [99, 100, 101],
            [99, 100, 101]
        ]
        img = np.uint8(img)
        img3d = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        img3d[..., 1] += 1
        img3d[..., 2] += 2

        aug = iaa.AllChannelsCLAHE(clip_limit=20, tile_grid_size_px=17)

        mock_clahe = ArgCopyingMagicMock()
        mock_clahe.apply.return_value = img

        with mock.patch('cv2.createCLAHE') as mock_createCLAHE:
            mock_createCLAHE.return_value = mock_clahe
            _ = aug.augment_image(img3d)

        clist = mock_clahe.apply.call_args_list
        assert np.array_equal(clist[0][0][0], img3d[..., 0])
        assert np.array_equal(clist[1][0][0], img3d[..., 1])
        assert np.array_equal(clist[2][0][0], img3d[..., 2])

    def test_basic_functionality_integrationtest(self):
        img = np.zeros((3, 7), dtype=np.uint8)
        img[0, 0] = 90
        img[0, 1] = 100
        img[0, 2] = 110
        for per_channel in [False, 0, 0.0, True, 1, 1.0]:
            for clip_limit in [4, 6]:
                for tile_grid_size_px in [3, 5, 7]:
                    with self.subTest(per_channel=per_channel,
                                      clip_limit=clip_limit,
                                      tile_grid_size_px=tile_grid_size_px):
                        aug = iaa.AllChannelsCLAHE(
                            clip_limit=clip_limit,
                            tile_grid_size_px=tile_grid_size_px,
                            per_channel=per_channel)
                        img_aug = aug.augment_image(img)
                        assert int(np.max(img_aug)) - int(np.min(img_aug)) > 2

    def test_tile_grid_size_px_min(self):
        img = np.zeros((1, 1), dtype=np.uint8)
        aug = iaa.AllChannelsCLAHE(
            clip_limit=20,
            tile_grid_size_px=iap.Deterministic(-1),
            tile_grid_size_px_min=5)
        mock_clahe = mock.Mock()
        mock_clahe.apply.return_value = img
        mock_createCLAHE = mock.MagicMock(return_value=mock_clahe)
        with mock.patch('cv2.createCLAHE', mock_createCLAHE):
            _ = aug.augment_image(img)
        mock_createCLAHE.assert_called_once_with(
            clipLimit=20, tileGridSize=(5, 5))

    def test_per_channel_integrationtest(self):
        # check that per_channel at 50% prob works
        aug = iaa.AllChannelsCLAHE(
            clip_limit=(1, 200),
            tile_grid_size_px=(3, 8),
            per_channel=0.5)
        seen = [False, False]
        img1000d = np.zeros((3, 7, 1000), dtype=np.uint8)
        img1000d[0, 0, :] = 90
        img1000d[0, 1, :] = 100
        img1000d[0, 2, :] = 110
        for _ in sm.xrange(100):
            with assertWarns(self, iaa.SuspiciousSingleImageShapeWarning):
                img_aug = aug.augment_image(img1000d)
            assert img_aug.dtype.name == "uint8"

            maxs = np.max(img_aug, axis=(0, 1))
            mins = np.min(img_aug, axis=(0, 1))
            diffs = maxs.astype(np.int32) - mins.astype(np.int32)

            nb_diffs_uq = len(set(diffs.flatten().tolist()))
            if nb_diffs_uq == 1:
                seen[0] = True
            else:
                seen[1] = True
            if np.all(seen):
                break
        assert np.all(seen)

    def test_unit_sized_kernels(self):
        img = np.zeros((1, 1), dtype=np.uint8)

        tile_grid_sizes = [0, 0, 0, 1, 1, 1, 3, 3, 3]
        tile_grid_min_sizes = [0, 1, 3, 0, 1, 3, 0, 1, 3]
        nb_calls_expected = [0, 0, 1, 0, 0, 1, 1, 1, 1]

        gen = zip(tile_grid_sizes, tile_grid_min_sizes, nb_calls_expected)
        for tile_grid_size_px, tile_grid_size_px_min, nb_calls_exp_i in gen:
            with self.subTest(tile_grid_size_px=tile_grid_size_px,
                              tile_grid_size_px_min=tile_grid_size_px_min,
                              nb_calls_expected_i=nb_calls_exp_i):
                aug = iaa.AllChannelsCLAHE(
                    clip_limit=20,
                    tile_grid_size_px=tile_grid_size_px,
                    tile_grid_size_px_min=tile_grid_size_px_min)
                mock_clahe = mock.Mock()
                mock_clahe.apply.return_value = img
                mock_createCLAHE = mock.MagicMock(return_value=mock_clahe)
                with mock.patch('cv2.createCLAHE', mock_createCLAHE):
                    _ = aug.augment_image(img)
                assert mock_createCLAHE.call_count == nb_calls_exp_i

    def test_other_dtypes(self):
        aug = iaa.AllChannelsCLAHE(clip_limit=0.01, tile_grid_size_px=3)

        # np.uint32: TypeError: src data type = 6 is not supported
        # np.uint64: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351:
        #            error: (-215:Assertion failed)
        #            src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #            || _src.type() == (((2) & ((1 << 3) - 1))
        #             + (((1)-1) << 3)) in function 'apply'
        # np.int8: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error:
        #          (-215:Assertion failed)
        #          src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #          || _src.type() == (((2) & ((1 << 3) - 1))
        #          + (((1)-1) << 3)) in function 'apply'
        # np.int16: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error:
        #           (-215:Assertion failed)
        #           src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #           || _src.type() == (((2) & ((1 << 3) - 1))
        #           + (((1)-1) << 3)) in function 'apply'
        # np.int32: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error:
        #           (-215:Assertion failed)
        #           src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #           || _src.type() == (((2) & ((1 << 3) - 1))
        #           + (((1)-1) << 3)) in function 'apply'
        # np.int64: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error:
        #           (-215:Assertion failed)
        #           src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #           || _src.type() == (((2) & ((1 << 3) - 1))
        #           + (((1)-1) << 3)) in function 'apply'
        # np.float16: TypeError: src data type = 23 is not supported
        # np.float32: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error:
        #             (-215:Assertion failed)
        #             src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #             || _src.type() == (((2) & ((1 << 3) - 1))
        #             + (((1)-1) << 3)) in function 'apply'
        # np.float64: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error:
        #             (-215:Assertion failed)
        #             src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #             || _src.type() == (((2) & ((1 << 3) - 1))
        #             + (((1)-1) << 3)) in function 'apply'
        # np.float128: TypeError: src data type = 13 is not supported
        for dtype in [np.uint8, np.uint16]:
            with self.subTest(dtype=np.dtype(dtype).name):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                dynamic_range = max_value - min_value

                img = np.zeros((11, 11, 1), dtype=dtype)
                img[:, 0, 0] = min_value
                img[:, 1, 0] = min_value + 30
                img[:, 2, 0] = min_value + 40
                img[:, 3, 0] = min_value + 50
                img[:, 4, 0] = (
                    int(center_value)
                    if np.dtype(dtype).kind != "f"
                    else center_value)
                img[:, 5, 0] = max_value - 50
                img[:, 6, 0] = max_value - 40
                img[:, 7, 0] = max_value - 30
                img[:, 8, 0] = max_value
                img_aug = aug.augment_image(img)

                assert img_aug.dtype.name == np.dtype(dtype).name
                assert (
                    min_value
                    <= np.min(img_aug)
                    <= min_value + 0.2 * dynamic_range)
                assert (
                    max_value - 0.2 * dynamic_range
                    <= np.max(img_aug)
                    <= max_value)

        # TypeError: src data type = 0 is not supported
        """
        with self.subTest("bool"):
            dtype = np.dtype(bool)
            print(dtype)

            img = np.zeros((11, 11, 1), dtype=dtype)
            img[:, 0, 0] = 0
            img[:, 1, 0] = 0
            img[:, 2, 0] = 0
            img[:, 3, 0] = 0
            img[:, 4, 0] = 0
            img[:, 5, 0] = 1
            img[:, 6, 0] = 1
            img[:, 7, 0] = 1
            img[:, 8, 0] = 1
            img_aug = aug.augment_image(img)
            print(img[..., 0])
            print(img_aug[..., 0])

            assert img_aug.dtype.name == np.dtype(dtype).name
            assert np.min(img_aug) == 0
            assert np.max(img_aug) == 1
        """

    def test_keypoints_not_changed(self):
        aug = iaa.AllChannelsCLAHE()
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
        kpsoi_aug = aug.augment_keypoints([kpsoi])
        assert keypoints_equal([kpsoi], kpsoi_aug)

    def test_heatmaps_not_changed(self):
        aug = iaa.AllChannelsCLAHE()
        heatmaps_arr = np.zeros((3, 3, 1), dtype=np.float32) + 0.5
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))
        heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
        assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

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
                image = np.full(shape, 129, dtype=np.uint8)
                aug = iaa.AllChannelsCLAHE()

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
                image = np.full(shape, 129, dtype=np.uint8)
                aug = iaa.AllChannelsCLAHE()

                image_aug = aug(image=image)

                assert np.any(image_aug != 128)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_get_parameters(self):
        aug = iaa.AllChannelsCLAHE(
            clip_limit=1,
            tile_grid_size_px=3,
            tile_grid_size_px_min=2,
            per_channel=True)
        params = aug.get_parameters()
        assert np.all([
            is_parameter_instance(params[i], iap.Deterministic)
            for i
            in [0, 3]])
        assert params[0].value == 1
        assert params[1][0].value == 3
        assert params[1][1] is None
        assert params[2] == 2
        assert params[3].value == 1

    def test_pickleable(self):
        aug = iaa.AllChannelsCLAHE(clip_limit=(30, 50),
                                   tile_grid_size_px=(4, 12),
                                   seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10, shape=(100, 100, 3))


class TestCLAHE(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_init(self):
        clahe = iaa.CLAHE(
            clip_limit=1,
            tile_grid_size_px=3,
            tile_grid_size_px_min=2,
            from_colorspace=iaa.CSPACE_BGR,
            to_colorspace=iaa.CSPACE_HSV)

        assert clahe.all_channel_clahe.clip_limit.value == 1
        assert clahe.all_channel_clahe.tile_grid_size_px[0].value == 3
        assert clahe.all_channel_clahe.tile_grid_size_px[1] is None
        assert clahe.all_channel_clahe.tile_grid_size_px_min == 2

        icba = clahe.intensity_channel_based_applier
        assert icba.from_colorspace == iaa.CSPACE_BGR
        assert icba.to_colorspace == iaa.CSPACE_HSV

    @mock.patch("imgaug.augmenters.color.change_colorspace_")
    def test_single_image_grayscale(self, mock_cs):
        img = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14]
        ]
        img = np.uint8(img)

        mocked_batch = _BatchInAugmentation(
            images=[img[..., np.newaxis] + 2])

        def _side_effect(image, _to_colorspace, _from_colorspace):
            return image + 1

        mock_cs.side_effect = _side_effect

        mock_all_channel_clahe = ArgCopyingMagicMock()
        mock_all_channel_clahe._augment_batch_.return_value = mocked_batch

        clahe = iaa.CLAHE(
            clip_limit=1,
            tile_grid_size_px=3,
            tile_grid_size_px_min=2,
            from_colorspace=iaa.CSPACE_RGB,
            to_colorspace=iaa.CSPACE_Lab)
        clahe.all_channel_clahe = mock_all_channel_clahe

        img_aug = clahe.augment_image(img)
        assert np.array_equal(img_aug, img+2)

        assert mock_cs.call_count == 0
        assert mock_all_channel_clahe._augment_batch_.call_count == 1

    @classmethod
    def _test_single_image_3d_rgb_to_x(cls, to_colorspace, channel_idx):
        fname_cs = "imgaug.augmenters.color.change_colorspace_"
        with mock.patch(fname_cs) as mock_cs:
            img = [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14]
            ]
            img = np.uint8(img)
            img3d = np.tile(img[..., np.newaxis], (1, 1, 3))
            img3d[..., 1] += 10
            img3d[..., 2] += 20

            def side_effect_change_colorspace(image, _to_colorspace,
                                              _from_colorspace):
                return image + 1

            def side_effect_all_channel_clahe(batch_call, _random_state,
                                              _parents, _hooks):
                batch_call = batch_call.deepcopy()
                batch_call.images = [batch_call.images[0] + 2]
                return batch_call

            mock_cs.side_effect = side_effect_change_colorspace

            mock_all_channel_clahe = ArgCopyingMagicMock()
            mock_all_channel_clahe._augment_batch_.side_effect = \
                side_effect_all_channel_clahe

            clahe = iaa.CLAHE(
                clip_limit=1,
                tile_grid_size_px=3,
                tile_grid_size_px_min=2,
                from_colorspace=iaa.CSPACE_RGB,
                to_colorspace=to_colorspace)
            clahe.all_channel_clahe = mock_all_channel_clahe

            img3d_aug = clahe.augment_image(np.copy(img3d))
            expected1 = img3d + 1
            expected2 = np.copy(expected1)
            expected2[..., channel_idx] += 2
            expected3 = np.copy(expected2) + 1
            assert np.array_equal(img3d_aug, expected3)

            assert mock_cs.call_count == 2
            assert mock_all_channel_clahe._augment_batch_.call_count == 1

            # indices: call 0, args, arg 0
            assert np.array_equal(mock_cs.call_args_list[0][0][0], img3d)

            # for some unclear reason, call_args_list here seems to contain the
            # output instead of the input to side_effect_all_channel_clahe, so
            # this assert is deactivated for now
            # cargs = mock_all_channel_clahe.call_args_list
            # print("mock", cargs[0][0][0][0].shape)
            # print("mock", cargs[0][0][0][0][..., 0])
            # print("exp ", expected1[..., channel_idx])
            # assert np.array_equal(
            #     cargs[0][0][0][0],
            #     expected1[..., channel_idx:channel_idx+1]
            # )

            assert np.array_equal(mock_cs.call_args_list[1][0][0], expected2)

    def test_single_image_3d_rgb_to_lab(self):
        self._test_single_image_3d_rgb_to_x(iaa.CSPACE_Lab, 0)

    def test_single_image_3d_rgb_to_hsv(self):
        self._test_single_image_3d_rgb_to_x(iaa.CSPACE_HSV, 2)

    def test_single_image_3d_rgb_to_hls(self):
        self._test_single_image_3d_rgb_to_x(iaa.CSPACE_HLS, 1)

    @mock.patch("imgaug.augmenters.color.change_colorspace_")
    def test_single_image_4d_rgb_to_lab(self, mock_cs):
        channel_idx = 0

        img = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14]
        ]
        img = np.uint8(img)
        img4d = np.tile(img[..., np.newaxis], (1, 1, 4))
        img4d[..., 1] += 10
        img4d[..., 2] += 20
        img4d[..., 3] += 30

        def side_effect_change_colorspace(image, _to_colorspace,
                                          _from_colorspace):
            return image + 1

        def side_effect_all_channel_clahe(batch_call, _random_state, _parents,
                                          _hooks):
            batch_call = batch_call.deepcopy()
            batch_call.images = [batch_call.images[0] + 2]
            return batch_call

        mock_cs.side_effect = side_effect_change_colorspace

        mock_all_channel_clahe = ArgCopyingMagicMock()
        mock_all_channel_clahe._augment_batch_.side_effect = \
            side_effect_all_channel_clahe

        clahe = iaa.CLAHE(
            clip_limit=1,
            tile_grid_size_px=3,
            tile_grid_size_px_min=2,
            from_colorspace=iaa.CSPACE_RGB,
            to_colorspace=iaa.CSPACE_Lab)
        clahe.all_channel_clahe = mock_all_channel_clahe

        img4d_aug = clahe.augment_image(img4d)
        expected1 = img4d[..., 0:3] + 1
        expected2 = np.copy(expected1)
        expected2[..., channel_idx] += 2
        expected3 = np.copy(expected2) + 1
        expected4 = np.dstack((expected3, img4d[..., 3:4]))
        assert np.array_equal(img4d_aug, expected4)

        assert mock_cs.call_count == 2
        assert mock_all_channel_clahe._augment_batch_.call_count == 1

        # indices: call 0, args, arg 0
        assert np.array_equal(mock_cs.call_args_list[0][0][0], img4d[..., 0:3])

        # for some unclear reason, call_args_list here seems to contain the
        # output instead of the input to side_effect_all_channel_clahe, so
        # this assert is deactivated for now
        # assert np.array_equal(
        #     mock_all_channel_clahe.call_args_list[0][0][0][0],
        #     expected1[..., channel_idx:channel_idx+1]
        # )

        assert np.array_equal(mock_cs.call_args_list[1][0][0], expected2)

    @mock.patch("imgaug.augmenters.color.change_colorspace_")
    def test_single_image_5d_rgb_to_lab(self, mock_cs):
        img = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14]
        ]
        img = np.uint8(img)
        img5d = np.tile(img[..., np.newaxis], (1, 1, 5))
        img5d[..., 1] += 10
        img5d[..., 2] += 20
        img5d[..., 3] += 30
        img5d[..., 4] += 40

        def side_effect_change_colorspace(image, _to_colorspace,
                                          _from_colorspace):
            return image + 1

        def side_effect_all_channel_clahe(batch_call, _random_state, _parents,
                                          _hooks):
            batch_call = batch_call.deepcopy()
            batch_call.images = [batch_call.images[0] + 2]
            return batch_call

        mock_cs.side_effect = side_effect_change_colorspace

        mock_all_channel_clahe = ArgCopyingMagicMock()
        mock_all_channel_clahe._augment_batch_.side_effect = \
            side_effect_all_channel_clahe

        clahe = iaa.CLAHE(
            clip_limit=1,
            tile_grid_size_px=3,
            tile_grid_size_px_min=2,
            from_colorspace=iaa.CSPACE_RGB,
            to_colorspace=iaa.CSPACE_Lab,
            name="ExampleCLAHE")
        clahe.all_channel_clahe = mock_all_channel_clahe

        # note that self.assertWarningRegex does not exist in python 2.7
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            img5d_aug = clahe.augment_image(img5d)
            assert len(caught_warnings) == 1
            assert (
                "Got image with 5 channels in _IntensityChannelBasedApplier "
                "(parents: ExampleCLAHE)"
                in str(caught_warnings[-1].message)
            )

        assert np.array_equal(img5d_aug, img5d + 2)

        assert mock_cs.call_count == 0
        assert mock_all_channel_clahe._augment_batch_.call_count == 1

        # indices: call 0, args, arg 0, image 0 in list of images
        assert np.array_equal(
            mock_all_channel_clahe
            ._augment_batch_
            .call_args_list[0][0][0]
            .images[0],
            img5d
        )

    @classmethod
    def _test_many_images_rgb_to_lab_list(cls, with_3d_images):
        fname_cs = "imgaug.augmenters.color.change_colorspace_"
        with mock.patch(fname_cs) as mock_cs:
            img = [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14]
            ]
            img = np.uint8(img)

            n_imgs = 2
            n_3d_imgs = 3 if with_3d_images else 0

            imgs = []
            for i in sm.xrange(n_imgs):
                imgs.append(img + i)
            for i in sm.xrange(n_3d_imgs):
                imgs.append(np.tile(img[..., np.newaxis], (1, 1, 3)) + 2 + i)

            def side_effect_change_colorspace(image, _to_colorspace,
                                              _from_colorspace):
                return image + 1

            def side_effect_all_channel_clahe(batch_call, _random_state,
                                              _parents, _hooks):
                batch_call = batch_call.deepcopy()
                batch_call.images = [image + 2 for image in batch_call.images]
                return batch_call

            mock_cs.side_effect = side_effect_change_colorspace

            mock_all_channel_clahe = ArgCopyingMagicMock()
            mock_all_channel_clahe._augment_batch_.side_effect = \
                side_effect_all_channel_clahe

            clahe = iaa.CLAHE(
                clip_limit=1,
                tile_grid_size_px=3,
                tile_grid_size_px_min=2,
                from_colorspace=iaa.CSPACE_RGB,
                to_colorspace=iaa.CSPACE_Lab)
            clahe.all_channel_clahe = mock_all_channel_clahe

            imgs_aug = clahe.augment_images(imgs)
            assert isinstance(imgs_aug, list)

            assert mock_cs.call_count == (n_3d_imgs*2 if with_3d_images else 0)
            assert (
                mock_all_channel_clahe
                ._augment_batch_
                .call_count == 1)

            # indices: call 0, args, arg 0
            assert isinstance(
                mock_all_channel_clahe
                ._augment_batch_
                .call_args_list[0][0][0],
                _BatchInAugmentation)

            assert (
                len(mock_all_channel_clahe
                    ._augment_batch_
                    .call_args_list[0][0][0]
                    .images)
                == 5 if with_3d_images else 2)

            # indices: call 0, args, arg 0, image i in list of images
            for i in sm.xrange(0, 2):
                expected = imgs[i][..., np.newaxis]
                assert np.array_equal(
                    mock_all_channel_clahe
                    ._augment_batch_
                    .call_args_list[0][0][0]
                    .images[i],
                    expected
                )

            if with_3d_images:
                for i in sm.xrange(2, 5):
                    expected = imgs[i]
                    if expected.shape[2] == 4:
                        expected = expected[..., 0:3]
                    assert np.array_equal(
                        mock_cs.call_args_list[i-2][0][0],
                        expected
                    )

                    # for some unclear reason, call_args_list here seems to
                    # contain the output instead of the input to
                    # side_effect_all_channel_clahe, so this assert is
                    # deactivated for now
                    # assert np.array_equal(
                    #     mock_all_channel_clahe.call_args_list[0][0][0][i],
                    #     (expected + 1)[..., 0:1]
                    # )

                    exp = (expected + 1)
                    exp[..., 0:1] += 2
                    assert np.array_equal(
                        mock_cs.call_args_list[3+i-2][0][0],
                        exp
                    )

    def test_many_images_rgb_to_lab_list_without_3d_images(self):
        self._test_many_images_rgb_to_lab_list(with_3d_images=False)

    def test_many_images_rgb_to_lab_list_with_3d_images(self):
        self._test_many_images_rgb_to_lab_list(with_3d_images=True)

    @classmethod
    def _test_many_images_rgb_to_lab_array(cls, nb_channels, nb_images):
        fname_cs = "imgaug.augmenters.color.change_colorspace_"
        with mock.patch(fname_cs) as mock_cs:
            with_color_conversion = (
                True if nb_channels is not None and nb_channels in [3, 4]
                else False)

            img = [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14]
            ]
            img = np.uint8(img)
            if nb_channels is not None:
                img = np.tile(img[..., np.newaxis], (1, 1, nb_channels))

            imgs = [img] * nb_images
            imgs = np.uint8(imgs)

            def side_effect_change_colorspace(image, _to_colorspace,
                                              _from_colorspace):
                return image + 1

            def side_effect_all_channel_clahe(batch_call, _random_state,
                                              _parents, _hooks):
                batch_call = batch_call.deepcopy()
                batch_call.images = [image + 2 for image in batch_call.images]
                return batch_call

            mock_cs.side_effect = side_effect_change_colorspace

            mock_all_channel_clahe = ArgCopyingMagicMock()
            mock_all_channel_clahe._augment_batch_.side_effect = \
                side_effect_all_channel_clahe

            clahe = iaa.CLAHE(
                clip_limit=1,
                tile_grid_size_px=3,
                tile_grid_size_px_min=2,
                from_colorspace=iaa.CSPACE_RGB,
                to_colorspace=iaa.CSPACE_Lab)
            clahe.all_channel_clahe = mock_all_channel_clahe

            imgs_aug = clahe.augment_images(imgs)
            assert ia.is_np_array(imgs_aug)

            assert mock_cs.call_count == (2*nb_images
                                          if with_color_conversion
                                          else 0)
            assert (
                mock_all_channel_clahe
                ._augment_batch_
                .call_count
                == 1)

            # indices: call 0, args, arg 0
            assert isinstance(
                mock_all_channel_clahe
                ._augment_batch_
                .call_args_list[0][0][0],
                _BatchInAugmentation)

            assert (
                len(
                    mock_all_channel_clahe
                    ._augment_batch_
                    .call_args_list[0][0][0]
                    .images)
                == nb_images)

            # indices: call 0, args, arg 0, image i in list of images
            if not with_color_conversion:
                for i in sm.xrange(nb_images):
                    expected = imgs[i]
                    if expected.ndim == 2:
                        expected = expected[..., np.newaxis]
                    # cant have 4 channels and no color conversion for RGB2Lab

                    assert np.array_equal(
                        mock_all_channel_clahe
                        ._augment_batch_
                        .call_args_list[0][0][0]
                        .images[i],
                        expected
                    )
            else:
                for i in sm.xrange(nb_images):
                    expected = imgs[i]
                    if expected.shape[2] == 4:
                        expected = expected[..., 0:3]
                    # cant have color conversion for RGB2Lab and no channel
                    # axis

                    assert np.array_equal(
                        mock_cs.call_args_list[i][0][0],
                        expected
                    )

                    # for some unclear reason, call_args_list here seems to
                    # contain the output instead of the input to
                    # side_effect_all_channel_clahe, so this assert is
                    # deactivated for now
                    # assert np.array_equal(
                    #     mock_all_channel_clahe.call_args_list[0][0][0][i],
                    #     (expected + 1)[..., 0:1]
                    # )

                    exp = (expected + 1)
                    exp[..., 0:1] += 2
                    assert np.array_equal(
                        mock_cs.call_args_list[nb_images+i][0][0],
                        exp
                    )

    def test_many_images_rgb_to_lab_array(self):
        gen = itertools.product([None, 1, 3, 4], [1, 2, 4])
        for nb_channels, nb_images in gen:
            with self.subTest(nb_channels=nb_channels, nb_images=nb_images):
                self._test_many_images_rgb_to_lab_array(
                    nb_channels=nb_channels,
                    nb_images=nb_images)

    def test_determinism(self):
        clahe = iaa.CLAHE(clip_limit=(1, 100),
                          tile_grid_size_px=(3, 60),
                          tile_grid_size_px_min=2,
                          from_colorspace=iaa.CSPACE_RGB,
                          to_colorspace=iaa.CSPACE_Lab)

        for nb_channels in [None, 1, 3, 4]:
            with self.subTest(nb_channels=nb_channels):
                img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
                if nb_channels is not None:
                    img = np.tile(img[..., np.newaxis], (1, 1, nb_channels))

                all_same = True
                for _ in sm.xrange(10):
                    result1 = clahe.augment_image(img)
                    result2 = clahe.augment_image(img)
                    same = np.array_equal(result1, result2)
                    all_same = all_same and same
                    if not all_same:
                        break
                assert not all_same

                clahe_det = clahe.to_deterministic()
                all_same = True
                for _ in sm.xrange(10):
                    result1 = clahe_det.augment_image(img)
                    result2 = clahe_det.augment_image(img)
                    same = np.array_equal(result1, result2)
                    all_same = all_same and same
                    if not all_same:
                        break
                assert all_same

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0, 3),
            (0, 1, 3),
            (1, 0, 3)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 129, dtype=np.uint8)
                aug = iaa.CLAHE()

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_get_parameters(self):
        clahe = iaa.CLAHE(
            clip_limit=1,
            tile_grid_size_px=3,
            tile_grid_size_px_min=2,
            from_colorspace=iaa.CSPACE_BGR,
            to_colorspace=iaa.CSPACE_HSV)
        params = clahe.get_parameters()
        assert params[0].value == 1
        assert params[1][0].value == 3
        assert params[1][1] is None
        assert params[2] == 2
        assert params[3] == iaa.CSPACE_BGR
        assert params[4] == iaa.CSPACE_HSV

    def test_pickleable(self):
        aug = iaa.CLAHE(clip_limit=(30, 50),
                        tile_grid_size_px=(4, 12),
                        seed=1)
        runtest_pickleable_uint8_img(aug, iterations=10, shape=(100, 100, 3))


class TestAllChannelsHistogramEqualization(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_basic_functionality(self):
        gen = itertools.product([None, 1, 2, 3], [1, 2, 3], [False, True])
        for nb_channels, nb_images, is_array in gen:
            with self.subTest(nb_channels=nb_channels, nb_images=nb_images,
                              is_array=is_array):
                img = [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                    [16, 17, 18, 19]
                ]
                img = np.uint8(img)
                if nb_channels is not None:
                    img = np.tile(img[..., np.newaxis], (1, 1, nb_channels))

                imgs = [img] * nb_images
                if is_array:
                    imgs = np.uint8(imgs)

                def _side_effect(img_call):
                    return img_call + 1

                mock_equalizeHist = mock.MagicMock(side_effect=_side_effect)
                with mock.patch('cv2.equalizeHist', mock_equalizeHist):
                    aug = iaa.AllChannelsHistogramEqualization()
                    imgs_aug = aug.augment_images(imgs)
                if is_array:
                    assert ia.is_np_array(imgs_aug)
                else:
                    assert isinstance(imgs_aug, list)
                assert len(imgs_aug) == nb_images
                for i in sm.xrange(nb_images):
                    assert imgs_aug[i].dtype.name == "uint8"
                    assert np.array_equal(imgs_aug[i], imgs[i] + 1)

    def test_basic_functionality_integrationtest(self):
        nb_channels = 3
        nb_images = 2

        img = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19]
        ]
        img = np.uint8(img)
        img = np.tile(img[..., np.newaxis], (1, 1, nb_channels))

        imgs = [img] * nb_images
        imgs = np.uint8(imgs)
        imgs[1][3:, ...] = 0

        aug = iaa.AllChannelsHistogramEqualization()
        imgs_aug = aug.augment_images(imgs)
        assert imgs_aug.dtype.name == "uint8"
        assert len(imgs_aug) == nb_images
        for i in sm.xrange(nb_images):
            assert imgs_aug[i].shape == img.shape
            assert np.max(imgs_aug[i]) > np.max(img)
        assert len(np.unique(imgs_aug[0])) > len(np.unique(imgs_aug[1]))

    def test_other_dtypes(self):
        aug = iaa.AllChannelsHistogramEqualization()

        # np.uint16: cv2.error: OpenCV(3.4.5) (...)/histogram.cpp:3345:
        #            error: (-215:Assertion failed)
        #            src.type() == CV_8UC1 in function 'equalizeHist'
        # np.uint32: TypeError: src data type = 6 is not supported
        # np.uint64: see np.uint16
        # np.int8: see np.uint16
        # np.int16: see np.uint16
        # np.int32: see np.uint16
        # np.int64: see np.uint16
        # np.float16: TypeError: src data type = 23 is not supported
        # np.float32: see np.uint16
        # np.float64: see np.uint16
        # np.float128: TypeError: src data type = 13 is not supported
        for dtype in [np.uint8]:
            with self.subTest(dtype=np.dtype(dtype).name):
                min_value, _center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                dynamic_range = max_value + abs(min_value)
                if np.dtype(dtype).kind == "f":
                    img = np.zeros((16,), dtype=dtype)
                    for i in sm.xrange(16):
                        img[i] = min_value + i * (0.01 * dynamic_range)
                    img = img.reshape((4, 4))
                else:
                    img = np.arange(
                        min_value, min_value + 16, dtype=dtype).reshape((4, 4))
                img_aug = aug.augment_image(img)
                assert img_aug.dtype.name == np.dtype(dtype).name
                assert img_aug.shape == img.shape
                assert np.min(img_aug) < min_value + 0.1 * dynamic_range
                assert np.max(img_aug) > max_value - 0.1 * dynamic_range

    def test_keypoints_not_changed(self):
        aug = iaa.AllChannelsHistogramEqualization()
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
        kpsoi_aug = aug.augment_keypoints([kpsoi])
        assert keypoints_equal([kpsoi], kpsoi_aug)

    def test_heatmaps_not_changed(self):
        aug = iaa.AllChannelsHistogramEqualization()
        heatmaps_arr = np.zeros((3, 3, 1), dtype=np.float32) + 0.5
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))
        heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
        assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

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
                image = np.full(shape, 129, dtype=np.uint8)
                aug = iaa.AllChannelsHistogramEqualization()

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
                image = np.full(shape, 129, dtype=np.uint8)
                aug = iaa.AllChannelsHistogramEqualization()

                image_aug = aug(image=image)

                assert np.any(image_aug != 128)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_get_parameters(self):
        aug = iaa.AllChannelsHistogramEqualization()
        params = aug.get_parameters()
        assert len(params) == 0

    def test_pickleable(self):
        aug = iaa.AllChannelsHistogramEqualization(seed=1)
        runtest_pickleable_uint8_img(aug, iterations=2, shape=(100, 100, 3))


class TestHistogramEqualization(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_init(self):
        aug = iaa.HistogramEqualization(
            from_colorspace=iaa.CSPACE_BGR,
            to_colorspace=iaa.CSPACE_HSV)
        assert isinstance(
            aug.all_channel_histogram_equalization,
            iaa.AllChannelsHistogramEqualization)

        icba = aug.intensity_channel_based_applier
        assert icba.from_colorspace == iaa.CSPACE_BGR
        assert icba.to_colorspace == iaa.CSPACE_HSV

    def test_basic_functionality_integrationtest(self):
        for nb_channels in [None, 1, 3, 4, 5]:
            with self.subTest(nb_channels=nb_channels):
                img = [
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14]
                ]
                img = np.uint8(img)
                if nb_channels is not None:
                    img = np.tile(img[..., np.newaxis], (1, 1, nb_channels))
                    if nb_channels >= 3:
                        img[..., 1] += 10
                        img[..., 2] += 20

                aug = iaa.HistogramEqualization(
                    from_colorspace=iaa.CSPACE_BGR,
                    to_colorspace=iaa.CSPACE_HSV,
                    name="ExampleHistEq")

                if nb_channels is None or nb_channels != 5:
                    img_aug = aug.augment_image(img)
                else:
                    with warnings.catch_warnings(record=True) as caught_warns:
                        warnings.simplefilter("always")
                        img_aug = aug.augment_image(img)
                        assert len(caught_warns) == 1
                        assert (
                            "Got image with 5 channels in "
                            "_IntensityChannelBasedApplier (parents: "
                            "ExampleHistEq)"
                            in str(caught_warns[-1].message)
                        )

                expected = img
                if nb_channels is None or nb_channels == 1:
                    expected = cv2.equalizeHist(expected)
                    if nb_channels == 1:
                        expected = expected[..., np.newaxis]
                elif nb_channels == 5:
                    for c in sm.xrange(expected.shape[2]):
                        expected[..., c:c+1] = cv2.equalizeHist(
                            expected[..., c]
                        )[..., np.newaxis]
                else:
                    if nb_channels == 4:
                        expected = expected[..., 0:3]
                    expected = cv2.cvtColor(expected, cv2.COLOR_RGB2HSV)
                    expected[..., 2] = cv2.equalizeHist(expected[..., 2])
                    expected = cv2.cvtColor(expected, cv2.COLOR_HSV2RGB)
                    if nb_channels == 4:
                        expected = np.concatenate(
                            (expected, img[..., 3:4]), axis=2)

                assert np.array_equal(img_aug, expected)

    def test_determinism(self):
        aug = iaa.HistogramEqualization(
            from_colorspace=iaa.CSPACE_RGB,
            to_colorspace=iaa.CSPACE_Lab)

        for nb_channels in [None, 1, 3, 4]:
            with self.subTest(nb_channels=nb_channels):
                img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
                if nb_channels is not None:
                    img = np.tile(img[..., np.newaxis], (1, 1, nb_channels))

                aug_det = aug.to_deterministic()
                result1 = aug_det.augment_image(img)
                result2 = aug_det.augment_image(img)
                assert np.array_equal(result1, result2)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0, 3),
            (0, 1, 3),
            (1, 0, 3)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 129, dtype=np.uint8)
                aug = iaa.HistogramEqualization()

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_get_parameters(self):
        aug = iaa.HistogramEqualization(
            from_colorspace=iaa.CSPACE_BGR,
            to_colorspace=iaa.CSPACE_HSV)
        params = aug.get_parameters()
        assert params[0] == iaa.CSPACE_BGR
        assert params[1] == iaa.CSPACE_HSV

    def test_pickleable(self):
        aug = iaa.HistogramEqualization(seed=1)
        runtest_pickleable_uint8_img(aug, iterations=2, shape=(100, 100, 3))
