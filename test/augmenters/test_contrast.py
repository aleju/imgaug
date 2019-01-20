from __future__ import print_function, division, absolute_import

import time
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

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
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
from imgaug.testutils import keypoints_equal, reseed


def main():
    time_start = time.time()

    test_GammaContrast()
    test_SigmoidContrast()
    test_LogContrast()
    test_LinearContrast()
    test_adjust_contrast_linear()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_GammaContrast():
    reseed()

    img = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    img = np.uint8(img)
    img3d = np.tile(img[:, :, np.newaxis], (1, 1, 3))

    # check basic functionality with gamma=1 or 2 (deterministic) and per_chanenl on/off (makes
    # no difference due to deterministic gamma)
    for per_channel in [False, 0, 0.0, True, 1, 1.0]:
        for gamma in [1, 2]:
            aug = iaa.GammaContrast(gamma=iap.Deterministic(gamma), per_channel=per_channel)
            img_aug = aug.augment_image(img)
            img3d_aug = aug.augment_image(img3d)
            assert img_aug.dtype.type == np.uint8
            assert img3d_aug.dtype.type == np.uint8
            assert np.array_equal(img_aug, skimage.exposure.adjust_gamma(img, gamma=gamma))
            assert np.array_equal(img3d_aug, skimage.exposure.adjust_gamma(img3d, gamma=gamma))

    # check that tuple to uniform works
    aug = iaa.GammaContrast((1, 2))
    assert isinstance(aug.params1d[0], iap.Uniform)
    assert isinstance(aug.params1d[0].a, iap.Deterministic)
    assert isinstance(aug.params1d[0].b, iap.Deterministic)
    assert aug.params1d[0].a.value == 1
    assert aug.params1d[0].b.value == 2

    # check that list to choice works
    aug = iaa.GammaContrast([1, 2])
    assert isinstance(aug.params1d[0], iap.Choice)
    assert all([val in aug.params1d[0].a for val in [1, 2]])

    # check that per_channel at 50% prob works
    aug = iaa.GammaContrast((0.5, 2.0), per_channel=0.5)
    seen = [False, False]
    img1000d = np.zeros((1, 1, 1000), dtype=np.uint8) + 128
    for _ in sm.xrange(100):
        img_aug = aug.augment_image(img1000d)
        assert img_aug.dtype.type == np.uint8
        nb_values_uq = len(set(img_aug.flatten().tolist()))
        if nb_values_uq == 1:
            seen[0] = True
        else:
            seen[1] = True
        if all(seen):
            break
    assert all(seen)

    # check that keypoints are not changed
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
    kpsoi_aug = iaa.GammaContrast(gamma=2).augment_keypoints([kpsoi])
    assert keypoints_equal([kpsoi], kpsoi_aug)

    # check that heatmaps are not changed
    heatmaps = ia.HeatmapsOnImage(np.zeros((3, 3, 1), dtype=np.float32) + 0.5, shape=(3, 3, 3))
    heatmaps_aug = iaa.GammaContrast(gamma=2).augment_heatmaps([heatmaps])[0]
    assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

    ###################
    # test other dtypes
    ###################
    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        exps = [1, 2, 3]
        values = [0, 100, int(center_value + 0.1*max_value)]
        tolerances = [0, 0, 1e-8 * max_value if dtype in [np.uint64, np.int64] else 0]

        for exp in exps:
            aug = iaa.GammaContrast(exp)
            for value, tolerance in zip(values, tolerances):
                image = np.full((3, 3), value, dtype=dtype)
                expected = (((image.astype(np.float128) / max_value) ** exp) * max_value).astype(dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype == np.dtype(dtype)
                assert len(np.unique(image_aug)) == 1
                value_aug = int(image_aug[0, 0])
                value_expected = int(expected[0, 0])
                diff = abs(value_aug - value_expected)
                assert diff <= tolerance

                # test other channel numbers
                for nb_channels in [1, 2, 3, 4, 5, 7, 11]:
                    image = np.full((3, 3), value, dtype=dtype)
                    image = np.tile(image[..., np.newaxis], (1, 1, nb_channels))
                    for c in sm.xrange(nb_channels):
                        image[..., c] += c
                    expected = (((image.astype(np.float128) / max_value) ** exp) * max_value).astype(dtype)
                    image_aug = aug.augment_image(image)
                    assert image_aug.shape == (3, 3, nb_channels)
                    assert image_aug.dtype == np.dtype(dtype)
                    # can be less than nb_channels when multiple input values map to the same output value
                    # mapping distribution can behave exponential with slow start and fast growth at the end
                    assert len(np.unique(image_aug)) <= nb_channels
                    for c in sm.xrange(nb_channels):
                        value_aug = int(image_aug[0, 0, c])
                        value_expected = int(expected[0, 0, c])
                        diff = abs(value_aug - value_expected)
                        assert diff <= tolerance

    # float
    for dtype in [np.float16, np.float32, np.float64]:
        def _allclose(a, b):
            atol = 1e-3 if dtype == np.float16 else 1e-8
            return np.allclose(a, b, atol=atol, rtol=0)

        exps = [1, 2]
        isize = np.dtype(dtype).itemsize
        values = [0, 1.0, 50.0, 100 ** (isize - 1)]

        for exp in exps:
            aug = iaa.GammaContrast(exp)
            for value in values:
                image = np.full((3, 3), value, dtype=dtype)
                expected = (image.astype(np.float128) ** exp).astype(dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype == np.dtype(dtype)
                assert _allclose(image_aug, expected)

                # test other channel numbers
                for nb_channels in [1, 2, 3, 4, 5, 7, 11]:
                    image = np.full((3, 3), value, dtype=dtype)
                    image = np.tile(image[..., np.newaxis], (1, 1, nb_channels))
                    for c in sm.xrange(nb_channels):
                        image[..., c] += float(c)
                    expected = (image.astype(np.float128) ** exp).astype(dtype)
                    image_aug = aug.augment_image(image)
                    assert image_aug.shape == (3, 3, nb_channels)
                    assert image_aug.dtype == np.dtype(dtype)
                    for c in sm.xrange(nb_channels):
                        value_aug = image_aug[0, 0, c]
                        value_expected = expected[0, 0, c]
                        assert _allclose(value_aug, value_expected)


def test_SigmoidContrast():
    reseed()

    img = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    img = np.uint8(img)
    img3d = np.tile(img[:, :, np.newaxis], (1, 1, 3))

    # check basic functionality with per_chanenl on/off (makes no difference due to deterministic
    # parameters)
    for per_channel in [False, 0, 0.0, True, 1, 1.0]:
        for gain, cutoff in itertools.product([5, 10], [0.25, 0.75]):
            aug = iaa.SigmoidContrast(gain=iap.Deterministic(gain), cutoff=iap.Deterministic(cutoff), per_channel=per_channel)
            img_aug = aug.augment_image(img)
            img3d_aug = aug.augment_image(img3d)
            assert img_aug.dtype.type == np.uint8
            assert img3d_aug.dtype.type == np.uint8
            assert np.array_equal(img_aug, skimage.exposure.adjust_sigmoid(img, gain=gain, cutoff=cutoff))
            assert np.array_equal(img3d_aug, skimage.exposure.adjust_sigmoid(img3d, gain=gain, cutoff=cutoff))

    # check that tuple to uniform works
    # note that gain and cutoff are saved in inverted order in _ContrastFuncWrapper to match
    # the order of skimage's function
    aug = iaa.SigmoidContrast(gain=(1, 2), cutoff=(0.25, 0.75))
    assert isinstance(aug.params1d[0], iap.Uniform)
    assert isinstance(aug.params1d[0].a, iap.Deterministic)
    assert isinstance(aug.params1d[0].b, iap.Deterministic)
    assert aug.params1d[0].a.value == 1
    assert aug.params1d[0].b.value == 2
    assert isinstance(aug.params1d[1], iap.Uniform)
    assert isinstance(aug.params1d[1].a, iap.Deterministic)
    assert isinstance(aug.params1d[1].b, iap.Deterministic)
    assert np.allclose(aug.params1d[1].a.value, 0.25)
    assert np.allclose(aug.params1d[1].b.value, 0.75)

    # check that list to choice works
    # note that gain and cutoff are saved in inverted order in _ContrastFuncWrapper to match
    # the order of skimage's function
    aug = iaa.SigmoidContrast(gain=[1, 2], cutoff=[0.25, 0.75])
    assert isinstance(aug.params1d[0], iap.Choice)
    assert all([val in aug.params1d[0].a for val in [1, 2]])
    assert isinstance(aug.params1d[1], iap.Choice)
    assert all([np.allclose(val, val_choice) for val, val_choice in zip([0.25, 0.75], aug.params1d[1].a)])

    # check that per_channel at 50% prob works
    aug = iaa.SigmoidContrast(gain=(1, 10), cutoff=(0.25, 0.75), per_channel=0.5)
    seen = [False, False]
    img1000d = np.zeros((1, 1, 1000), dtype=np.uint8) + 128
    for _ in sm.xrange(100):
        img_aug = aug.augment_image(img1000d)
        assert img_aug.dtype.type == np.uint8
        nb_values_uq = len(set(img_aug.flatten().tolist()))
        if nb_values_uq == 1:
            seen[0] = True
        else:
            seen[1] = True
        if all(seen):
            break
    assert all(seen)

    # check that keypoints are not changed
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
    kpsoi_aug = iaa.SigmoidContrast(gain=10, cutoff=0.5).augment_keypoints([kpsoi])
    assert keypoints_equal([kpsoi], kpsoi_aug)

    # check that heatmaps are not changed
    heatmaps = ia.HeatmapsOnImage(np.zeros((3, 3, 1), dtype=np.float32) + 0.5, shape=(3, 3, 3))
    heatmaps_aug = iaa.SigmoidContrast(gain=10, cutoff=0.5).augment_heatmaps([heatmaps])[0]
    assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

    ###################
    # test other dtypes
    ###################
    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
        # dynamic_range = max_value - min_value

        gains = [5, 20]
        cutoffs = [0.25, 0.75]
        values = [0, 100, int(center_value + 0.1 * max_value)]
        tmax = 1e-8 * max_value if dtype in [np.uint64, np.int64] else 0
        tolerances = [tmax, tmax, tmax]

        for gain, cutoff in itertools.product(gains, cutoffs):
            aug = iaa.SigmoidContrast(gain=gain, cutoff=cutoff)
            for value, tolerance in zip(values, tolerances):
                image = np.full((3, 3), value, dtype=dtype)
                # TODO this looks like the equation commented out should acutally the correct one, but when using it
                #      we get a difference between expectation and skimage ground truth
                # 1/(1 + exp(gain*(cutoff - I_ij/max)))
                expected = (1/(1 + np.exp(gain * (cutoff - image.astype(np.float128)/max_value))))
                expected = (expected * max_value).astype(dtype)
                # expected = (1/(1 + np.exp(gain * (cutoff - (image.astype(np.float128)-min_value)/dynamic_range))))
                # expected = (min_value + expected * dynamic_range).astype(dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype == np.dtype(dtype)
                assert len(np.unique(image_aug)) == 1
                value_aug = int(image_aug[0, 0])
                value_expected = int(expected[0, 0])
                diff = abs(value_aug - value_expected)
                assert diff <= tolerance

    # float
    for dtype in [np.float16, np.float32, np.float64]:
        def _allclose(a, b):
            atol = 1e-3 if dtype == np.float16 else 1e-8
            return np.allclose(a, b, atol=atol, rtol=0)

        gains = [5, 20]
        cutoffs = [0.25, 0.75]
        isize = np.dtype(dtype).itemsize
        values = [0, 1.0, 50.0, 100 ** (isize - 1)]

        for gain, cutoff in itertools.product(gains, cutoffs):
            aug = iaa.SigmoidContrast(gain=gain, cutoff=cutoff)
            for value in values:
                image = np.full((3, 3), value, dtype=dtype)
                expected = (1 / (1 + np.exp(gain * (cutoff - image.astype(np.float128))))).astype(dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype == np.dtype(dtype)
                assert _allclose(image_aug, expected)


def test_LogContrast():
    reseed()

    img = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    img = np.uint8(img)
    img3d = np.tile(img[:, :, np.newaxis], (1, 1, 3))

    # check basic functionality with gain=1 or 2 (deterministic) and per_chanenl on/off (makes
    # no difference due to deterministic gain)
    for per_channel in [False, 0, 0.0, True, 1, 1.0]:
        for gain in [1, 2]:
            aug = iaa.LogContrast(gain=iap.Deterministic(gain), per_channel=per_channel)
            img_aug = aug.augment_image(img)
            img3d_aug = aug.augment_image(img3d)
            assert img_aug.dtype.type == np.uint8
            assert img3d_aug.dtype.type == np.uint8
            assert np.array_equal(img_aug, skimage.exposure.adjust_log(img, gain=gain))
            assert np.array_equal(img3d_aug, skimage.exposure.adjust_log(img3d, gain=gain))

    # check that tuple to uniform works
    aug = iaa.LogContrast((1, 2))
    assert isinstance(aug.params1d[0], iap.Uniform)
    assert isinstance(aug.params1d[0].a, iap.Deterministic)
    assert isinstance(aug.params1d[0].b, iap.Deterministic)
    assert aug.params1d[0].a.value == 1
    assert aug.params1d[0].b.value == 2

    # check that list to choice works
    aug = iaa.LogContrast([1, 2])
    assert isinstance(aug.params1d[0], iap.Choice)
    assert all([val in aug.params1d[0].a for val in [1, 2]])

    # check that per_channel at 50% prob works
    aug = iaa.LogContrast((0.5, 2.0), per_channel=0.5)
    seen = [False, False]
    img1000d = np.zeros((1, 1, 1000), dtype=np.uint8) + 128
    for _ in sm.xrange(100):
        img_aug = aug.augment_image(img1000d)
        assert img_aug.dtype.type == np.uint8
        nb_values_uq = len(set(img_aug.flatten().tolist()))
        if nb_values_uq == 1:
            seen[0] = True
        else:
            seen[1] = True
        if all(seen):
            break
    assert all(seen)

    # check that keypoints are not changed
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
    kpsoi_aug = iaa.LogContrast(gain=2).augment_keypoints([kpsoi])
    assert keypoints_equal([kpsoi], kpsoi_aug)

    # check that heatmaps are not changed
    heatmaps = ia.HeatmapsOnImage(np.zeros((3, 3, 1), dtype=np.float32) + 0.5, shape=(3, 3, 3))
    heatmaps_aug = iaa.LogContrast(gain=2).augment_heatmaps([heatmaps])[0]
    assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

    ###################
    # test other dtypes
    ###################
    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        gains = [0.5, 0.75, 1.0, 1.1]
        values = [0, 100, int(center_value + 0.1 * max_value)]
        tmax = 1e-8 * max_value if dtype in [np.uint64, np.int64] else 0
        tolerances = [0, tmax, tmax]

        for gain in gains:
            aug = iaa.LogContrast(gain)
            for value, tolerance in zip(values, tolerances):
                image = np.full((3, 3), value, dtype=dtype)
                expected = gain * np.log2(1 + (image.astype(np.float128)/max_value))
                expected = (expected*max_value).astype(dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype == np.dtype(dtype)
                assert len(np.unique(image_aug)) == 1
                value_aug = int(image_aug[0, 0])
                value_expected = int(expected[0, 0])
                diff = abs(value_aug - value_expected)
                assert diff <= tolerance

    # float
    for dtype in [np.float16, np.float32, np.float64]:
        def _allclose(a, b):
            atol = 1e-2 if dtype == np.float16 else 1e-8
            return np.allclose(a, b, atol=atol, rtol=0)

        gains = [0.5, 0.75, 1.0, 1.1]
        isize = np.dtype(dtype).itemsize
        values = [0, 1.0, 50.0, 100 ** (isize - 1)]

        for gain in gains:
            aug = iaa.LogContrast(gain)
            for value in values:
                image = np.full((3, 3), value, dtype=dtype)
                expected = gain * np.log2(1 + image.astype(np.float128))
                expected = expected.astype(dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype == np.dtype(dtype)
                assert _allclose(image_aug, expected)


def test_LinearContrast():
    reseed()

    img = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    img = np.uint8(img)
    img3d = np.tile(img[:, :, np.newaxis], (1, 1, 3))

    # check basic functionality with alpha=1 or 2 (deterministic) and per_chanenl on/off (makes
    # no difference due to deterministic alpha)
    for per_channel in [False, 0, 0.0, True, 1, 1.0]:
        for alpha in [1, 2]:
            aug = iaa.LinearContrast(alpha=iap.Deterministic(alpha), per_channel=per_channel)
            img_aug = aug.augment_image(img)
            img3d_aug = aug.augment_image(img3d)
            assert img_aug.dtype.type == np.uint8
            assert img3d_aug.dtype.type == np.uint8
            assert np.array_equal(img_aug, contrast_lib.adjust_contrast_linear(img, alpha=alpha))
            assert np.array_equal(img3d_aug, contrast_lib.adjust_contrast_linear(img3d, alpha=alpha))

    # check that tuple to uniform works
    aug = iaa.LinearContrast((1, 2))
    assert isinstance(aug.params1d[0], iap.Uniform)
    assert isinstance(aug.params1d[0].a, iap.Deterministic)
    assert isinstance(aug.params1d[0].b, iap.Deterministic)
    assert aug.params1d[0].a.value == 1
    assert aug.params1d[0].b.value == 2

    # check that list to choice works
    aug = iaa.LinearContrast([1, 2])
    assert isinstance(aug.params1d[0], iap.Choice)
    assert all([val in aug.params1d[0].a for val in [1, 2]])

    # check that per_channel at 50% prob works
    aug = iaa.LinearContrast((0.5, 2.0), per_channel=0.5)
    seen = [False, False]
    # must not use just value 128 here, otherwise nothing will change as all values would have
    # distance 0 to 128
    img1000d = np.zeros((1, 1, 1000), dtype=np.uint8) + 128 + 20
    for _ in sm.xrange(100):
        img_aug = aug.augment_image(img1000d)
        assert img_aug.dtype.type == np.uint8
        nb_values_uq = len(set(img_aug.flatten().tolist()))
        if nb_values_uq == 1:
            seen[0] = True
        else:
            seen[1] = True
        if all(seen):
            break
    assert all(seen)

    # check that keypoints are not changed
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
    kpsoi_aug = iaa.LinearContrast(alpha=2).augment_keypoints([kpsoi])
    assert keypoints_equal([kpsoi], kpsoi_aug)

    # check that heatmaps are not changed
    heatmaps = ia.HeatmapsOnImage(np.zeros((3, 3, 1), dtype=np.float32) + 0.5, shape=(3, 3, 3))
    heatmaps_aug = iaa.LinearContrast(alpha=2).augment_heatmaps([heatmaps])[0]
    assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

    # test for other dtypes are in test_adjust_contrast_linear()


def test_adjust_contrast_linear():
    for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32,
                  np.float16, np.float32, np.float64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
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

        # alphas [0, 1, 2, 4, 100]
        alphas = [0, 1, 2, 4]
        if dtype not in [np.uint8, np.int8, np.float16]:
            alphas.append(100)

        for alpha in alphas:
            expected = [
                [cv-4*alpha, cv-3*alpha, cv-2*alpha],
                [cv-1*alpha, cv+0*alpha, cv+1*alpha],
                [cv+2*alpha, cv+3*alpha, cv+4*alpha]
            ]
            expected = np.array(expected, dtype=dtype)
            observed = contrast_lib.adjust_contrast_linear(img, alpha=alpha)
            assert observed.dtype == np.dtype(dtype)
            assert observed.shape == img.shape
            assert _compare(observed, expected)

    # overflow for uint8
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
        aug = iaa.AllChannelsCLAHE(clip_limit=10, tile_grid_size_px=11, tile_grid_size_px_min=4, per_channel=True)
        assert isinstance(aug.clip_limit, iap.Deterministic)
        assert aug.clip_limit.value == 10
        assert isinstance(aug.tile_grid_size_px[0], iap.Deterministic)
        assert aug.tile_grid_size_px[0].value == 11
        assert aug.tile_grid_size_px[1] is None
        assert aug.tile_grid_size_px_min == 4
        assert isinstance(aug.per_channel, iap.Deterministic)
        assert np.isclose(aug.per_channel.value, 1.0)

        aug = iaa.AllChannelsCLAHE(clip_limit=(10, 20), tile_grid_size_px=(11, 17), tile_grid_size_px_min=4,
                                   per_channel=0.5)
        assert isinstance(aug.clip_limit, iap.Uniform)
        assert aug.clip_limit.a.value == 10
        assert aug.clip_limit.b.value == 20
        assert isinstance(aug.tile_grid_size_px[0], iap.DiscreteUniform)
        assert aug.tile_grid_size_px[0].a.value == 11
        assert aug.tile_grid_size_px[0].b.value == 17
        assert aug.tile_grid_size_px[1] is None
        assert aug.tile_grid_size_px_min == 4
        assert isinstance(aug.per_channel, iap.Binomial)
        assert np.isclose(aug.per_channel.p.value, 0.5)

        aug = iaa.AllChannelsCLAHE(clip_limit=[10, 20, 30], tile_grid_size_px=[11, 17, 21])
        assert isinstance(aug.clip_limit, iap.Choice)
        assert aug.clip_limit.a[0] == 10
        assert aug.clip_limit.a[1] == 20
        assert aug.clip_limit.a[2] == 30
        assert isinstance(aug.tile_grid_size_px[0], iap.Choice)
        assert aug.tile_grid_size_px[0].a[0] == 11
        assert aug.tile_grid_size_px[0].a[1] == 17
        assert aug.tile_grid_size_px[0].a[2] == 21
        assert aug.tile_grid_size_px[1] is None

        aug = iaa.AllChannelsCLAHE(tile_grid_size_px=((11, 17), [11, 13, 15]))
        assert isinstance(aug.tile_grid_size_px[0], iap.DiscreteUniform)
        assert aug.tile_grid_size_px[0].a.value == 11
        assert aug.tile_grid_size_px[0].b.value == 17
        assert isinstance(aug.tile_grid_size_px[1], iap.Choice)
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

        mock_clahe = mock.Mock()
        mock_clahe.apply.return_value = img
        mock_createCLAHE = mock.MagicMock(return_value=mock_clahe)

        # image with single channel
        with mock.patch('cv2.createCLAHE', mock_createCLAHE):
            _ = aug.augment_image(img)

        mock_createCLAHE.assert_called_once_with(clipLimit=20, tileGridSize=(17, 17))
        assert np.array_equal(mock_clahe.apply.call_args_list[0][0][0], img)

        mock_clahe.reset_mock()

        # image with three channels
        with mock.patch('cv2.createCLAHE', mock_createCLAHE):
            _ = aug.augment_image(img3d)
        assert np.array_equal(mock_clahe.apply.call_args_list[0][0][0], img3d[..., 0])
        assert np.array_equal(mock_clahe.apply.call_args_list[1][0][0], img3d[..., 1])
        assert np.array_equal(mock_clahe.apply.call_args_list[2][0][0], img3d[..., 2])

    def test_basic_functionality_integrationtest(self):
        img = np.zeros((3, 7), dtype=np.uint8)
        img[0, 0] = 90
        img[0, 1] = 100
        img[0, 2] = 110
        for per_channel in [False, 0, 0.0, True, 1, 1.0]:
            for clip_limit in [4, 6]:
                for tile_grid_size_px in [3, 5, 7]:
                    with self.subTest(per_channel=per_channel, clip_limit=clip_limit,
                                      tile_grid_size_px=tile_grid_size_px):
                        aug = iaa.AllChannelsCLAHE(clip_limit=clip_limit,
                                                   tile_grid_size_px=tile_grid_size_px,
                                                   per_channel=per_channel)
                        img_aug = aug.augment_image(img)
                        assert int(np.max(img_aug)) - int(np.min(img_aug)) > 2

    def test_tile_grid_size_px_min(self):
        img = np.zeros((1, 1), dtype=np.uint8)
        aug = iaa.AllChannelsCLAHE(clip_limit=20, tile_grid_size_px=iap.Deterministic(-1), tile_grid_size_px_min=5)
        mock_clahe = mock.Mock()
        mock_clahe.apply.return_value = img
        mock_createCLAHE = mock.MagicMock(return_value=mock_clahe)
        with mock.patch('cv2.createCLAHE', mock_createCLAHE):
            _ = aug.augment_image(img)
        mock_createCLAHE.assert_called_once_with(clipLimit=20, tileGridSize=(5, 5))

    def test_per_channel_integrationtest(self):
        # check that per_channel at 50% prob works
        aug = iaa.AllChannelsCLAHE(clip_limit=(1, 200), tile_grid_size_px=(3, 8), per_channel=0.5)
        seen = [False, False]
        img1000d = np.zeros((3, 7, 1000), dtype=np.uint8)
        img1000d[0, 0, :] = 90
        img1000d[0, 1, :] = 100
        img1000d[0, 2, :] = 110
        for _ in sm.xrange(100):
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
            if all(seen):
                break
        assert all(seen)

    def test_unit_sized_kernels(self):
        img = np.zeros((1, 1), dtype=np.uint8)

        tile_grid_sizes = [0, 0, 0, 1, 1, 1, 3, 3, 3]
        tile_grid_min_sizes = [0, 1, 3, 0, 1, 3, 0, 1, 3]
        nb_calls_expected = [0, 0, 1, 0, 0, 1, 1, 1, 1]

        gen = zip(tile_grid_sizes, tile_grid_min_sizes, nb_calls_expected)
        for tile_grid_size_px, tile_grid_size_px_min, nb_calls_expected_i in gen:
            with self.subTest(tile_grid_size_px=tile_grid_size_px,
                              tile_grid_size_px_min=tile_grid_size_px_min,
                              nb_calls_expected_i=nb_calls_expected_i):
                aug = iaa.AllChannelsCLAHE(clip_limit=20,
                                           tile_grid_size_px=tile_grid_size_px,
                                           tile_grid_size_px_min=tile_grid_size_px_min)
                mock_clahe = mock.Mock()
                mock_clahe.apply.return_value = img
                mock_createCLAHE = mock.MagicMock(return_value=mock_clahe)
                with mock.patch('cv2.createCLAHE', mock_createCLAHE):
                    _ = aug.augment_image(img)
                assert mock_createCLAHE.call_count == nb_calls_expected_i

    def test_other_dtypes(self):
        aug = iaa.AllChannelsCLAHE(clip_limit=0.01, tile_grid_size_px=3)

        # np.uint32: TypeError: src data type = 6 is not supported
        # np.uint64: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion failed)
        #            src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #            || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in function 'apply'
        # np.int8: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion failed)
        #          src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #          || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in function 'apply'
        # np.int16: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion failed)
        #           src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #           || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in function 'apply'
        # np.int32: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion failed)
        #           src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #           || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in function 'apply'
        # np.int64: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion failed)
        #           src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #           || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in function 'apply'
        # np.float16: TypeError: src data type = 23 is not supported
        # np.float32: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion failed)
        #             src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #             || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in function 'apply'
        # np.float64: cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion failed)
        #             src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
        #             || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in function 'apply'
        # np.float128: TypeError: src data type = 13 is not supported
        for dtype in [np.uint8, np.uint16]:
            with self.subTest(dtype=np.dtype(dtype).name):
                min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
                dynamic_range = max_value - min_value

                img = np.zeros((11, 11, 1), dtype=dtype)
                img[:, 0, 0] = min_value
                img[:, 1, 0] = min_value + 30
                img[:, 2, 0] = min_value + 40
                img[:, 3, 0] = min_value + 50
                img[:, 4, 0] = int(center_value) if np.dtype(dtype).kind != "f" else center_value
                img[:, 5, 0] = max_value - 50
                img[:, 6, 0] = max_value - 40
                img[:, 7, 0] = max_value - 30
                img[:, 8, 0] = max_value
                img_aug = aug.augment_image(img)

                assert img_aug.dtype.name == np.dtype(dtype).name
                assert min_value <= np.min(img_aug) <= min_value + 0.2 * dynamic_range
                assert max_value - 0.2 * dynamic_range <= np.max(img_aug) <= max_value

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
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
        kpsoi_aug = iaa.AllChannelsCLAHE().augment_keypoints([kpsoi])
        assert keypoints_equal([kpsoi], kpsoi_aug)

    def test_heatmaps_not_changed(self):
        heatmaps = ia.HeatmapsOnImage(np.zeros((3, 3, 1), dtype=np.float32) + 0.5, shape=(3, 3, 3))
        heatmaps_aug = iaa.AllChannelsCLAHE().augment_heatmaps([heatmaps])[0]
        assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

    def test_get_parameters(self):
        aug = iaa.AllChannelsCLAHE(clip_limit=1, tile_grid_size_px=3, tile_grid_size_px_min=2, per_channel=True)
        params = aug.get_parameters()
        assert all([isinstance(params[i], iap.Deterministic) for i in [0, 3]])
        assert params[0].value == 1
        assert params[1][0].value == 3
        assert params[1][1] is None
        assert params[2] == 2
        assert params[3].value == 1


class TestCLAHE(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_init(self):
        clahe = iaa.CLAHE(clip_limit=1, tile_grid_size_px=3, tile_grid_size_px_min=2, from_colorspace=iaa.CLAHE.BGR,
                          to_colorspace=iaa.CLAHE.HSV)
        assert clahe.all_channel_clahe.clip_limit.value == 1
        assert clahe.all_channel_clahe.tile_grid_size_px[0].value == 3
        assert clahe.all_channel_clahe.tile_grid_size_px[1] is None
        assert clahe.all_channel_clahe.tile_grid_size_px_min == 2
        assert clahe.intensity_channel_based_applier.change_colorspace.from_colorspace == iaa.CLAHE.BGR
        assert clahe.intensity_channel_based_applier.change_colorspace.to_colorspace.value == iaa.CLAHE.HSV
        assert clahe.intensity_channel_based_applier.change_colorspace_inv.from_colorspace == iaa.CLAHE.HSV
        assert clahe.intensity_channel_based_applier.change_colorspace_inv.to_colorspace.value == iaa.CLAHE.BGR

    def test_single_image_grayscale(self):
        img = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14]
        ]
        img = np.uint8(img)

        mock_change_colorspace = mock.Mock()
        mock_change_colorspace._augment_images.return_value = [img[..., np.newaxis] + 1]
        mock_all_channel_clahe = mock.Mock()
        mock_all_channel_clahe._augment_images.return_value = [img[..., np.newaxis] + 2]
        mock_change_colorspace_inv = mock.Mock()
        mock_change_colorspace_inv._augment_image.return_value = [img[..., np.newaxis] + 3]

        clahe = iaa.CLAHE(clip_limit=1, tile_grid_size_px=3, tile_grid_size_px_min=2,
                          from_colorspace=iaa.CLAHE.RGB,
                          to_colorspace=iaa.CLAHE.Lab)
        clahe.all_channel_clahe = mock_all_channel_clahe
        clahe.intensity_channel_based_applier.change_colorspace = mock_change_colorspace
        clahe.intensity_channel_based_applier.change_colorspace_inv = mock_change_colorspace_inv

        img_aug = clahe.augment_image(img)
        assert np.array_equal(img_aug, img+2)

        mock_change_colorspace = mock_change_colorspace._augment_images
        mock_all_channel_clahe = mock_all_channel_clahe._augment_images
        mock_change_colorspace_inv = mock_change_colorspace_inv._augment_images

        assert mock_change_colorspace.call_count == 0
        assert mock_all_channel_clahe.call_count == 1
        assert mock_change_colorspace_inv.call_count == 0

    def _test_single_image_3d_rgb_to_x(self, to_colorspace, channel_idx):
        img = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14]
        ]
        img = np.uint8(img)
        img3d = np.tile(img[..., np.newaxis], (1, 1, 3))
        img3d[..., 1] += 10
        img3d[..., 2] += 20

        def side_effect_change_colorspace(imgs_call, _random_state, _parents, _hooks):
            return [imgs_call[0] + 1]

        def side_effect_all_channel_clahe(imgs_call, _random_state, _parents, _hooks):
            return [imgs_call[0] + 2]

        def side_effect_change_colorspace_inv(imgs_call, _random_state, _parents, _hooks):
            return [imgs_call[0] + 3]

        mock_change_colorspace = mock.Mock()
        mock_change_colorspace._augment_images.side_effect = side_effect_change_colorspace
        mock_change_colorspace.to_colorspace = iap.Deterministic(to_colorspace)
        mock_all_channel_clahe = mock.Mock()
        mock_all_channel_clahe._augment_images.side_effect = side_effect_all_channel_clahe
        mock_change_colorspace_inv = mock.Mock()
        mock_change_colorspace_inv._augment_images.side_effect = side_effect_change_colorspace_inv

        clahe = iaa.CLAHE(clip_limit=1, tile_grid_size_px=3, tile_grid_size_px_min=2,
                          from_colorspace=iaa.CLAHE.RGB,
                          to_colorspace=to_colorspace)
        clahe.all_channel_clahe = mock_all_channel_clahe
        clahe.intensity_channel_based_applier.change_colorspace = mock_change_colorspace
        clahe.intensity_channel_based_applier.change_colorspace_inv = mock_change_colorspace_inv

        img3d_aug = clahe.augment_image(np.copy(img3d))
        expected1 = img3d + 1
        expected2 = np.copy(expected1)
        expected2[..., channel_idx] += 2
        expected3 = np.copy(expected2) + 3
        assert np.array_equal(img3d_aug, expected3)

        mock_change_colorspace = mock_change_colorspace._augment_images
        mock_all_channel_clahe = mock_all_channel_clahe._augment_images
        mock_change_colorspace_inv = mock_change_colorspace_inv._augment_images

        assert mock_change_colorspace.call_count == 1
        assert mock_all_channel_clahe.call_count == 1
        assert mock_change_colorspace_inv.call_count == 1

        # indices: call 0, args, arg 0, image 0 in list of images
        assert np.array_equal(mock_change_colorspace.call_args_list[0][0][0][0], img3d)

        # for some unclear reason, call_args_list here seems to contain the output instead of the input
        # to side_effect_all_channel_clahe, so this assert is deactivated for now
        # print("mock", mock_all_channel_clahe.call_args_list[0][0][0][0].shape)
        # print("mock", mock_all_channel_clahe.call_args_list[0][0][0][0][..., 0])
        # print("exp ", expected1[..., channel_idx])
        # assert np.array_equal(
        #     mock_all_channel_clahe.call_args_list[0][0][0][0],
        #     expected1[..., channel_idx:channel_idx+1]
        # )

        assert np.array_equal(
            mock_change_colorspace_inv.call_args_list[0][0][0][0],
            expected2
        )

    def test_single_image_3d_rgb_to_lab(self):
        self._test_single_image_3d_rgb_to_x(iaa.CLAHE.Lab, 0)

    def test_single_image_3d_rgb_to_hsv(self):
        self._test_single_image_3d_rgb_to_x(iaa.CLAHE.HSV, 2)

    def test_single_image_3d_rgb_to_hls(self):
        self._test_single_image_3d_rgb_to_x(iaa.CLAHE.HLS, 1)

    def test_single_image_4d_rgb_to_lab(self):
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

        def side_effect_change_colorspace(imgs_call, _random_state, _parents, _hooks):
            return [imgs_call[0] + 1]

        def side_effect_all_channel_clahe(imgs_call, _random_state, _parents, _hooks):
            return [imgs_call[0] + 2]

        def side_effect_change_colorspace_inv(imgs_call, _random_state, _parents, _hooks):
            return [imgs_call[0] + 3]

        mock_change_colorspace = mock.Mock()
        mock_change_colorspace._augment_images.side_effect = side_effect_change_colorspace
        mock_change_colorspace.to_colorspace = iap.Deterministic(iaa.CLAHE.Lab)
        mock_all_channel_clahe = mock.Mock()
        mock_all_channel_clahe._augment_images.side_effect = side_effect_all_channel_clahe
        mock_change_colorspace_inv = mock.Mock()
        mock_change_colorspace_inv._augment_images.side_effect = side_effect_change_colorspace_inv

        clahe = iaa.CLAHE(clip_limit=1, tile_grid_size_px=3, tile_grid_size_px_min=2,
                          from_colorspace=iaa.CLAHE.RGB,
                          to_colorspace=iaa.CLAHE.Lab)
        clahe.all_channel_clahe = mock_all_channel_clahe
        clahe.intensity_channel_based_applier.change_colorspace = mock_change_colorspace
        clahe.intensity_channel_based_applier.change_colorspace_inv = mock_change_colorspace_inv

        img4d_aug = clahe.augment_image(img4d)
        expected1 = img4d[..., 0:3] + 1
        expected2 = np.copy(expected1)
        expected2[..., channel_idx] += 2
        expected3 = np.copy(expected2) + 3
        expected4 = np.dstack((expected3, img4d[..., 3:4]))
        assert np.array_equal(img4d_aug, expected4)

        mock_change_colorspace = mock_change_colorspace._augment_images
        mock_all_channel_clahe = mock_all_channel_clahe._augment_images
        mock_change_colorspace_inv = mock_change_colorspace_inv._augment_images

        assert mock_change_colorspace.call_count == 1
        assert mock_all_channel_clahe.call_count == 1
        assert mock_change_colorspace_inv.call_count == 1

        # indices: call 0, args, arg 0, image 0 in list of images
        assert np.array_equal(mock_change_colorspace.call_args_list[0][0][0][0], img4d[..., 0:3])

        # for some unclear reason, call_args_list here seems to contain the output instead of the input
        # to side_effect_all_channel_clahe, so this assert is deactivated for now
        # assert np.array_equal(
        #     mock_all_channel_clahe.call_args_list[0][0][0][0],
        #     expected1[..., channel_idx:channel_idx+1]
        # )

        assert np.array_equal(
            mock_change_colorspace_inv.call_args_list[0][0][0][0],
            expected2
        )

    def test_single_image_5d_rgb_to_lab(self):
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

        def side_effect_change_colorspace(imgs_call, _random_state, _parents, _hooks):
            return [imgs_call[0] + 1]

        def side_effect_all_channel_clahe(imgs_call, _random_state, _parents, _hooks):
            return [imgs_call[0] + 2]

        def side_effect_change_colorspace_inv(imgs_call, _random_state, _parents, _hooks):
            return [imgs_call[0] + 3]

        mock_change_colorspace = mock.Mock()
        mock_change_colorspace._augment_images.side_effect = side_effect_change_colorspace
        mock_all_channel_clahe = mock.Mock()
        mock_all_channel_clahe._augment_images.side_effect = side_effect_all_channel_clahe
        mock_change_colorspace_inv = mock.Mock()
        mock_change_colorspace_inv._augment_images.side_effect = side_effect_change_colorspace_inv

        clahe = iaa.CLAHE(clip_limit=1, tile_grid_size_px=3, tile_grid_size_px_min=2,
                          from_colorspace=iaa.CLAHE.RGB,
                          to_colorspace=iaa.CLAHE.Lab,
                          name="ExampleCLAHE")
        clahe.all_channel_clahe = mock_all_channel_clahe
        clahe.intensity_channel_based_applier.change_colorspace = mock_change_colorspace
        clahe.intensity_channel_based_applier.change_colorspace_inv = mock_change_colorspace_inv

        # note that self.assertWarningRegex does not exist in python 2.7
        with warnings.catch_warnings(record=True) as caught_warnings:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            img5d_aug = clahe.augment_image(img5d)
            # Verify
            assert len(caught_warnings) == 1
            assert "Got image with 5 channels in _IntensityChannelBasedApplier (parents: ExampleCLAHE)" \
                   in str(caught_warnings[-1].message)

        assert np.array_equal(img5d_aug, img5d + 2)

        mock_change_colorspace = mock_change_colorspace._augment_images
        mock_all_channel_clahe = mock_all_channel_clahe._augment_images
        mock_change_colorspace_inv = mock_change_colorspace_inv._augment_images

        assert mock_change_colorspace.call_count == 0
        assert mock_all_channel_clahe.call_count == 1
        assert mock_change_colorspace_inv.call_count == 0

        # indices: call 0, args, arg 0, image 0 in list of images
        assert np.array_equal(
            mock_all_channel_clahe.call_args_list[0][0][0][0],
            img5d
        )

    def _test_many_images_rgb_to_lab_list(self, with_3d_images):
        img = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14]
        ]
        img = np.uint8(img)

        imgs = [
            img,
            img + 1
        ]
        if with_3d_images:
            imgs.extend([
                np.tile(img[..., np.newaxis], (1, 1, 3)) + 2,
                np.tile(img[..., np.newaxis], (1, 1, 3)) + 3,
                np.tile(img[..., np.newaxis], (1, 1, 4)) + 4
            ])

        def side_effect_change_colorspace(imgs_call, _random_state, _parents, _hooks):
            return [img + 1 for img in imgs_call]

        def side_effect_all_channel_clahe(imgs_call, _random_state, _parents, _hooks):
            return [img + 2 for img in imgs_call]

        def side_effect_change_colorspace_inv(imgs_call, _random_state, _parents, _hooks):
            return [img + 3 for img in imgs_call]

        mock_change_colorspace = mock.Mock()
        mock_change_colorspace._augment_images.side_effect = side_effect_change_colorspace
        mock_change_colorspace.to_colorspace = iap.Deterministic(iaa.CLAHE.Lab)
        mock_all_channel_clahe = mock.Mock()
        mock_all_channel_clahe._augment_images.side_effect = side_effect_all_channel_clahe
        mock_change_colorspace_inv = mock.Mock()
        mock_change_colorspace_inv._augment_images.side_effect = side_effect_change_colorspace_inv

        clahe = iaa.CLAHE(clip_limit=1, tile_grid_size_px=3, tile_grid_size_px_min=2,
                          from_colorspace=iaa.CLAHE.RGB,
                          to_colorspace=iaa.CLAHE.Lab)
        clahe.all_channel_clahe = mock_all_channel_clahe
        clahe.intensity_channel_based_applier.change_colorspace = mock_change_colorspace
        clahe.intensity_channel_based_applier.change_colorspace_inv = mock_change_colorspace_inv

        imgs_aug = clahe.augment_images(imgs)
        assert isinstance(imgs_aug, list)

        mock_change_colorspace = mock_change_colorspace._augment_images
        mock_all_channel_clahe = mock_all_channel_clahe._augment_images
        mock_change_colorspace_inv = mock_change_colorspace_inv._augment_images

        assert mock_change_colorspace.call_count == (1 if with_3d_images else 0)
        assert mock_all_channel_clahe.call_count == 1
        assert mock_change_colorspace_inv.call_count == (1 if with_3d_images else 0)

        # indices: call 0, args, arg 0
        if with_3d_images:
            assert isinstance(mock_change_colorspace.call_args_list[0][0][0], list)
            assert isinstance(mock_change_colorspace_inv.call_args_list[0][0][0], list)
        assert isinstance(mock_all_channel_clahe.call_args_list[0][0][0], list)

        if with_3d_images:
            assert len(mock_change_colorspace.call_args_list[0][0][0]) == 3
            assert len(mock_change_colorspace_inv.call_args_list[0][0][0]) == 3
        assert len(mock_all_channel_clahe.call_args_list[0][0][0]) == 5 if with_3d_images else 2

        # indices: call 0, args, arg 0, image i in list of images
        for i in sm.xrange(0, 2):
            expected = imgs[i][..., np.newaxis]
            assert np.array_equal(
                mock_all_channel_clahe.call_args_list[0][0][0][i],
                expected
            )

        if with_3d_images:
            for i in sm.xrange(2, 5):
                expected = imgs[i]
                if expected.shape[2] == 4:
                    expected = expected[..., 0:3]
                assert np.array_equal(
                    mock_change_colorspace.call_args_list[0][0][0][i-2],
                    expected
                )

                # for some unclear reason, call_args_list here seems to contain the output instead of the input
                # to side_effect_all_channel_clahe, so this assert is deactivated for now
                # assert np.array_equal(
                #     mock_all_channel_clahe.call_args_list[0][0][0][i],
                #     (expected + 1)[..., 0:1]
                # )

                exp = (expected + 1)
                exp[..., 0:1] += 2
                assert np.array_equal(
                    mock_change_colorspace_inv.call_args_list[0][0][0][i-2],
                    exp
                )

    def test_many_images_rgb_to_lab_list_without_3d_images(self):
        self._test_many_images_rgb_to_lab_list(with_3d_images=False)

    def test_many_images_rgb_to_lab_list_with_3d_images(self):
        self._test_many_images_rgb_to_lab_list(with_3d_images=True)

    def _test_many_images_rgb_to_lab_array(self, nb_channels, nb_images):
        with_color_conversion = True if nb_channels is not None and nb_channels in [3, 4] else False

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

        def side_effect_change_colorspace(imgs_call, _random_state, _parents, _hooks):
            return [img + 1 for img in imgs_call]

        def side_effect_all_channel_clahe(imgs_call, _random_state, _parents, _hooks):
            return [img + 2 for img in imgs_call]

        def side_effect_change_colorspace_inv(imgs_call, _random_state, _parents, _hooks):
            return [img + 3 for img in imgs_call]

        mock_change_colorspace = mock.Mock()
        mock_change_colorspace._augment_images.side_effect = side_effect_change_colorspace
        mock_change_colorspace.to_colorspace = iap.Deterministic(iaa.CLAHE.Lab)
        mock_all_channel_clahe = mock.Mock()
        mock_all_channel_clahe._augment_images.side_effect = side_effect_all_channel_clahe
        mock_change_colorspace_inv = mock.Mock()
        mock_change_colorspace_inv._augment_images.side_effect = side_effect_change_colorspace_inv

        clahe = iaa.CLAHE(clip_limit=1, tile_grid_size_px=3, tile_grid_size_px_min=2,
                          from_colorspace=iaa.CLAHE.RGB,
                          to_colorspace=iaa.CLAHE.Lab)
        clahe.all_channel_clahe = mock_all_channel_clahe
        clahe.intensity_channel_based_applier.change_colorspace = mock_change_colorspace
        clahe.intensity_channel_based_applier.change_colorspace_inv = mock_change_colorspace_inv

        imgs_aug = clahe.augment_images(imgs)
        assert ia.is_np_array(imgs_aug)

        mock_change_colorspace = mock_change_colorspace._augment_images
        mock_all_channel_clahe = mock_all_channel_clahe._augment_images
        mock_change_colorspace_inv = mock_change_colorspace_inv._augment_images

        assert mock_change_colorspace.call_count == (1 if with_color_conversion else 0)
        assert mock_all_channel_clahe.call_count == 1
        assert mock_change_colorspace_inv.call_count == (1 if with_color_conversion else 0)

        # indices: call 0, args, arg 0
        if with_color_conversion:
            assert isinstance(mock_change_colorspace.call_args_list[0][0][0], list)
            assert isinstance(mock_change_colorspace_inv.call_args_list[0][0][0], list)
        assert isinstance(mock_all_channel_clahe.call_args_list[0][0][0], list)

        if with_color_conversion:
            assert len(mock_change_colorspace.call_args_list[0][0][0]) == nb_images
            assert len(mock_change_colorspace_inv.call_args_list[0][0][0]) == nb_images
        assert len(mock_all_channel_clahe.call_args_list[0][0][0]) == nb_images

        # indices: call 0, args, arg 0, image i in list of images
        if not with_color_conversion:
            for i in sm.xrange(nb_images):
                expected = imgs[i]
                if expected.ndim == 2:
                    expected = expected[..., np.newaxis]
                # cant have 4 channels and no color conversion for RGB2Lab

                assert np.array_equal(
                    mock_all_channel_clahe.call_args_list[0][0][0][i],
                    expected
                )
        else:
            for i in sm.xrange(nb_images):
                expected = imgs[i]
                if expected.shape[2] == 4:
                    expected = expected[..., 0:3]
                # cant have color conversion for RGB2Lab and no channel axis

                assert np.array_equal(
                    mock_change_colorspace.call_args_list[0][0][0][i],
                    expected
                )

                # for some unclear reason, call_args_list here seems to contain the output instead of the input
                # to side_effect_all_channel_clahe, so this assert is deactivated for now
                # assert np.array_equal(
                #     mock_all_channel_clahe.call_args_list[0][0][0][i],
                #     (expected + 1)[..., 0:1]
                # )

                exp = (expected + 1)
                exp[..., 0:1] += 2
                assert np.array_equal(
                    mock_change_colorspace_inv.call_args_list[0][0][0][i],
                    exp
                )

    def test_many_images_rgb_to_lab_array(self):
        for nb_channels, nb_images in itertools.product([None, 1, 3, 4], [1, 2, 4]):
            with self.subTest(nb_channels=nb_channels, nb_images=nb_images):
                self._test_many_images_rgb_to_lab_array(nb_channels=nb_channels, nb_images=nb_images)

    def test_determinism(self):
        clahe = iaa.CLAHE(clip_limit=(1, 100), tile_grid_size_px=(3, 60), tile_grid_size_px_min=2,
                          from_colorspace=iaa.CLAHE.RGB, to_colorspace=iaa.CLAHE.Lab)

        for nb_channels in [None, 1, 3, 4]:
            with self.subTest(nb_channels=nb_channels):
                img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
                if nb_channels is not None:
                    img = np.tile(img[..., np.newaxis], (1, 1, nb_channels))

                result1 = clahe.augment_image(img)
                result2 = clahe.augment_image(img)
                assert not np.array_equal(result1, result2)

                clahe_det = clahe.to_deterministic()
                result1 = clahe_det.augment_image(img)
                result2 = clahe_det.augment_image(img)
                assert np.array_equal(result1, result2)

    def test_get_parameters(self):
        clahe = iaa.CLAHE(clip_limit=1, tile_grid_size_px=3, tile_grid_size_px_min=2, from_colorspace=iaa.CLAHE.BGR,
                          to_colorspace=iaa.CLAHE.HSV)
        params = clahe.get_parameters()
        assert params[0].value == 1
        assert params[1][0].value == 3
        assert params[1][1] is None
        assert params[2] == 2
        assert params[3] == iaa.CLAHE.BGR
        assert params[4] == iaa.CLAHE.HSV


class TestAllChannelsHistogramEqualization(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_basic_functionality(self):
        for nb_channels, nb_images, is_array in itertools.product([None, 1, 2, 3], [1, 2, 3], [False, True]):
            with self.subTest(nb_channels=nb_channels, nb_images=nb_images, is_array=is_array):
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

        # np.uint16: cv2.error: OpenCV(3.4.5) (...)/histogram.cpp:3345: error: (-215:Assertion failed)
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
                min_value, _center_value, max_value = iadt.get_value_range_of_dtype(dtype)
                dynamic_range = max_value + abs(min_value)
                if np.dtype(dtype).kind == "f":
                    img = np.zeros((16,), dtype=dtype)
                    for i in sm.xrange(16):
                        img[i] = min_value + i * (0.01 * dynamic_range)
                    img = img.reshape((4, 4))
                else:
                    img = np.arange(min_value, min_value + 16, dtype=dtype).reshape((4, 4))
                img_aug = aug.augment_image(img)
                assert img_aug.dtype.name == np.dtype(dtype).name
                assert img_aug.shape == img.shape
                assert np.min(img_aug) < min_value + 0.1 * dynamic_range
                assert np.max(img_aug) > max_value - 0.1 * dynamic_range

    def test_keypoints_not_changed(self):
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(1, 1)], shape=(3, 3, 3))
        kpsoi_aug = iaa.AllChannelsHistogramEqualization().augment_keypoints([kpsoi])
        assert keypoints_equal([kpsoi], kpsoi_aug)

    def test_heatmaps_not_changed(self):
        heatmaps = ia.HeatmapsOnImage(np.zeros((3, 3, 1), dtype=np.float32) + 0.5, shape=(3, 3, 3))
        heatmaps_aug = iaa.AllChannelsHistogramEqualization().augment_heatmaps([heatmaps])[0]
        assert np.allclose(heatmaps.arr_0to1, heatmaps_aug.arr_0to1)

    def test_get_parameters(self):
        aug = iaa.AllChannelsHistogramEqualization()
        params = aug.get_parameters()
        assert len(params) == 0


class TestHistogramEqualization(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_init(self):
        aug = iaa.HistogramEqualization(from_colorspace=iaa.HistogramEqualization.BGR,
                                        to_colorspace=iaa.HistogramEqualization.HSV)
        assert isinstance(aug.all_channel_histogram_equalization, iaa.AllChannelsHistogramEqualization)

        icba = aug.intensity_channel_based_applier
        assert icba.change_colorspace.from_colorspace == iaa.HistogramEqualization.BGR
        assert icba.change_colorspace.to_colorspace.value == iaa.HistogramEqualization.HSV

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

                aug = iaa.HistogramEqualization(from_colorspace=iaa.HistogramEqualization.BGR,
                                                to_colorspace=iaa.HistogramEqualization.HSV,
                                                name="ExampleHistEq")

                if nb_channels is None or nb_channels != 5:
                    img_aug = aug.augment_image(img)
                else:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        # Cause all warnings to always be triggered.
                        warnings.simplefilter("always")
                        # Trigger a warning.
                        img_aug = aug.augment_image(img)
                        # Verify
                        assert len(caught_warnings) == 1
                        assert "Got image with 5 channels in _IntensityChannelBasedApplier (parents: ExampleHistEq)"\
                               in str(caught_warnings[-1].message)

                expected = img
                if nb_channels is None or nb_channels == 1:
                    expected = cv2.equalizeHist(expected)
                    if nb_channels == 1:
                        expected = expected[..., np.newaxis]
                elif nb_channels == 5:
                    for c in sm.xrange(expected.shape[2]):
                        expected[..., c:c+1] = cv2.equalizeHist(expected[..., c])[..., np.newaxis]
                else:
                    if nb_channels == 4:
                        expected = expected[..., 0:3]
                    expected = cv2.cvtColor(expected, cv2.COLOR_RGB2HSV)
                    expected[..., 2] = cv2.equalizeHist(expected[..., 2])
                    expected = cv2.cvtColor(expected, cv2.COLOR_HSV2RGB)
                    if nb_channels == 4:
                        expected = np.concatenate((expected, img[..., 3:4]), axis=2)

                assert np.array_equal(img_aug, expected)

    def test_determinism(self):
        aug = iaa.HistogramEqualization(from_colorspace=iaa.HistogramEqualization.RGB,
                                        to_colorspace=iaa.HistogramEqualization.Lab)

        for nb_channels in [None, 1, 3, 4]:
            with self.subTest(nb_channels=nb_channels):
                img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
                if nb_channels is not None:
                    img = np.tile(img[..., np.newaxis], (1, 1, nb_channels))

                aug_det = aug.to_deterministic()
                result1 = aug_det.augment_image(img)
                result2 = aug_det.augment_image(img)
                assert np.array_equal(result1, result2)

    def test_get_parameters(self):
        aug = iaa.HistogramEqualization(from_colorspace=iaa.HistogramEqualization.BGR,
                                        to_colorspace=iaa.HistogramEqualization.HSV)
        params = aug.get_parameters()
        assert params[0] == iaa.HistogramEqualization.BGR
        assert params[1] == iaa.HistogramEqualization.HSV


if __name__ == "__main__":
    main()
