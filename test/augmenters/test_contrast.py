from __future__ import print_function, division, absolute_import

import time
import itertools

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm
import skimage
import skimage.data

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmenters import contrast as contrast_lib
from imgaug.augmenters import meta
from imgaug.testutils import keypoints_equal, reseed


def main():
    time_start = time.time()

    test_GammaContrast()
    test_SigmoidContrast()
    test_LogContrast()
    test_LinearContrast()
    test_contrast_adjust_linear()

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
        min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)

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
    assert np.allclose(aug.params1d[0].a.value, 0.25)
    assert np.allclose(aug.params1d[0].b.value, 0.75)
    assert isinstance(aug.params1d[1], iap.Uniform)
    assert isinstance(aug.params1d[1].a, iap.Deterministic)
    assert isinstance(aug.params1d[1].b, iap.Deterministic)
    assert aug.params1d[1].a.value == 1
    assert aug.params1d[1].b.value == 2

    # check that list to choice works
    # note that gain and cutoff are saved in inverted order in _ContrastFuncWrapper to match
    # the order of skimage's function
    aug = iaa.SigmoidContrast(gain=[1, 2], cutoff=[0.25, 0.75])
    assert isinstance(aug.params1d[0], iap.Choice)
    assert all([np.allclose(val, val_choice) for val, val_choice in zip([0.25, 0.75], aug.params1d[0].a)])
    assert isinstance(aug.params1d[1], iap.Choice)
    assert all([val in aug.params1d[1].a for val in [1, 2]])

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
        min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)

        gains = [5, 20]
        cutoffs = [0.25, 0.75]
        values = [0, 100, int(center_value + 0.1 * max_value)]
        tmax = 1e-8 * max_value if dtype in [np.uint64, np.int64] else 0
        tolerances = [tmax, tmax, tmax]

        for gain, cutoff in itertools.product(gains, cutoffs):
            aug = iaa.SigmoidContrast(gain=gain, cutoff=cutoff)
            for value, tolerance in zip(values, tolerances):
                image = np.full((3, 3), value, dtype=dtype)
                # 1/(1 + exp(gain*(cutoff - I_ij/max)))
                expected = (1/(1 + np.exp(gain * (cutoff - image.astype(np.float128)/max_value))))
                expected = (expected * max_value).astype(dtype)
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
        min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)

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
            assert np.array_equal(img_aug, contrast_lib._adjust_linear(img, alpha=alpha))
            assert np.array_equal(img3d_aug, contrast_lib._adjust_linear(img3d, alpha=alpha))

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

    # test for other dtypes are in test_contrast_adjust_linear()


def test_contrast_adjust_linear():
    for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32,
                  np.float16, np.float32, np.float64]:
        min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)
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
            observed = contrast_lib._adjust_linear(img, alpha=alpha)
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
    observed = contrast_lib._adjust_linear(img, alpha=255)
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
    observed = contrast_lib._adjust_linear(img, alpha=257)
    expected = [
        [cv, cv, cv],
        [cv, cv, cv],
        [cv, cv, cv]
    ]
    assert np.array_equal(observed, expected)


if __name__ == "__main__":
    main()
