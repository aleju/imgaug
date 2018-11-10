from __future__ import print_function, division, absolute_import

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
from utils import create_random_images, create_random_keypoints, array_equal_lists, keypoints_equal, reseed


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
        l = len(set(img_aug.flatten().tolist()))
        if l == 1:
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
        l = len(set(img_aug.flatten().tolist()))
        if l == 1:
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
        l = len(set(img_aug.flatten().tolist()))
        if l == 1:
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
        l = len(set(img_aug.flatten().tolist()))
        if l == 1:
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


def test_contrast_adjust_linear():
    img = [
        [124, 125, 126],
        [127, 128, 129],
        [130, 131, 132]
    ]
    img = np.uint8(img)

    # alpha = 1
    observed = contrast_lib._adjust_linear(img, alpha=1)
    assert observed.dtype.type == np.uint8
    assert observed.shape == img.shape
    assert np.array_equal(observed, img)

    # alpha = 2
    expected = [
        [120, 122, 124],
        [126, 128, 130],
        [132, 134, 136]
    ]
    expected = np.uint8(expected)
    observed = contrast_lib._adjust_linear(img, alpha=2)
    assert observed.dtype.type == np.uint8
    assert observed.shape == img.shape
    assert np.array_equal(observed, expected)


if __name__ == "__main__":
    main()
