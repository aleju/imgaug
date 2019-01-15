from __future__ import print_function, division, absolute_import

import time

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.testutils import reseed


def main():
    time_start = time.time()

    test_FastSnowyLandscape()
    test_Clouds()
    test_Fog()
    # test_CloudLayer()
    test_Snowflakes()
    # test_SnowflakesLayer()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_FastSnowyLandscape():
    reseed()

    # check parameters
    aug = iaa.FastSnowyLandscape(lightness_threshold=[100, 200], lightness_multiplier=[1.0, 4.0])
    assert isinstance(aug.lightness_threshold, iap.Choice)
    assert len(aug.lightness_threshold.a) == 2
    assert aug.lightness_threshold.a[0] == 100
    assert aug.lightness_threshold.a[1] == 200

    assert isinstance(aug.lightness_multiplier, iap.Choice)
    assert len(aug.lightness_multiplier.a) == 2
    assert np.allclose(aug.lightness_multiplier.a[0], 1.0)
    assert np.allclose(aug.lightness_multiplier.a[1], 4.0)

    # basic functionality test
    aug = iaa.FastSnowyLandscape(lightness_threshold=100, lightness_multiplier=2.0)
    image = np.arange(0, 6*6*3).reshape((6, 6, 3)).astype(np.uint8)
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    mask = (image_hls[..., 1] < 100)
    expected = np.copy(image_hls).astype(np.float32)
    expected[..., 1][mask] *= 2.0
    expected = np.clip(np.round(expected), 0, 255).astype(np.uint8)
    expected = cv2.cvtColor(expected, cv2.COLOR_HLS2RGB)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, expected)

    # test when varying lightness_threshold between images
    image = np.arange(0, 6*6*3).reshape((6, 6, 3)).astype(np.uint8)
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    class _TwoValueParam(iap.StochasticParameter):
        def __init__(self, v1, v2):
            super(_TwoValueParam, self).__init__()
            self.v1 = v1
            self.v2 = v2

        def _draw_samples(self, size, random_state):
            arr = np.full(size, self.v1, dtype=np.float32)
            arr[1::2] = self.v2
            return arr

    aug = iaa.FastSnowyLandscape(lightness_threshold=_TwoValueParam(75, 125), lightness_multiplier=2.0)

    mask = (image_hls[..., 1] < 75)
    expected1 = np.copy(image_hls).astype(np.float64)
    expected1[..., 1][mask] *= 2.0
    expected1 = np.clip(np.round(expected1), 0, 255).astype(np.uint8)
    expected1 = cv2.cvtColor(expected1, cv2.COLOR_HLS2RGB)

    mask = (image_hls[..., 1] < 125)
    expected2 = np.copy(image_hls).astype(np.float64)
    expected2[..., 1][mask] *= 2.0
    expected2 = np.clip(np.round(expected2), 0, 255).astype(np.uint8)
    expected2 = cv2.cvtColor(expected2, cv2.COLOR_HLS2RGB)

    observed = aug.augment_images([image] * 4)

    assert np.array_equal(observed[0], expected1)
    assert np.array_equal(observed[1], expected2)
    assert np.array_equal(observed[2], expected1)
    assert np.array_equal(observed[3], expected2)

    # test when varying lightness_multiplier between images
    aug = iaa.FastSnowyLandscape(lightness_threshold=100, lightness_multiplier=_TwoValueParam(1.5, 2.0))

    mask = (image_hls[..., 1] < 100)
    expected1 = np.copy(image_hls).astype(np.float64)
    expected1[..., 1][mask] *= 1.5
    expected1 = np.clip(np.round(expected1), 0, 255).astype(np.uint8)
    expected1 = cv2.cvtColor(expected1, cv2.COLOR_HLS2RGB)

    mask = (image_hls[..., 1] < 100)
    expected2 = np.copy(image_hls).astype(np.float64)
    expected2[..., 1][mask] *= 2.0
    expected2 = np.clip(np.round(expected2), 0, 255).astype(np.uint8)
    expected2 = cv2.cvtColor(expected2, cv2.COLOR_HLS2RGB)

    observed = aug.augment_images([image] * 4)

    assert np.array_equal(observed[0], expected1)
    assert np.array_equal(observed[1], expected2)
    assert np.array_equal(observed[2], expected1)
    assert np.array_equal(observed[3], expected2)

    # test BGR colorspace
    aug = iaa.FastSnowyLandscape(lightness_threshold=100, lightness_multiplier=2.0, from_colorspace="BGR")
    image = np.arange(0, 6*6*3).reshape((6, 6, 3)).astype(np.uint8)
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mask = (image_hls[..., 1] < 100)
    expected = np.copy(image_hls).astype(np.float32)
    expected[..., 1][mask] *= 2.0
    expected = np.clip(np.round(expected), 0, 255).astype(np.uint8)
    expected = cv2.cvtColor(expected, cv2.COLOR_HLS2BGR)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, expected)


def test_Clouds():
    # rough test as fairly hard to test more detailed
    reseed()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_aug = iaa.Clouds().augment_image(img)
    assert 20 < np.average(img_aug) < 250
    assert np.max(img_aug) > 150

    grad_x = img_aug[:, 1:].astype(np.float32) - img_aug[:, :-1].astype(np.float32)
    grad_y = img_aug[1:, :].astype(np.float32) - img_aug[:-1, :].astype(np.float32)

    assert np.sum(np.abs(grad_x)) > 5 * img.shape[1]
    assert np.sum(np.abs(grad_y)) > 5 * img.shape[0]


def test_Fog():
    # rough test as fairly hard to test more detailed
    reseed()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_aug = iaa.Clouds().augment_image(img)
    assert 50 < np.average(img_aug) < 255
    assert np.max(img_aug) > 100

    grad_x = img_aug[:, 1:].astype(np.float32) - img_aug[:, :-1].astype(np.float32)
    grad_y = img_aug[1:, :].astype(np.float32) - img_aug[:-1, :].astype(np.float32)

    assert np.sum(np.abs(grad_x)) > 1 * img.shape[1]
    assert np.sum(np.abs(grad_y)) > 1 * img.shape[0]


def test_Snowflakes():
    # rather rough test as fairly hard to test more detailed
    reseed()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_aug = iaa.Snowflakes().augment_image(img)
    assert 0.01 < np.average(img_aug) < 100
    assert np.max(img_aug) > 100

    grad_x = img_aug[:, 1:].astype(np.float32) - img_aug[:, :-1].astype(np.float32)
    grad_y = img_aug[1:, :].astype(np.float32) - img_aug[:-1, :].astype(np.float32)

    assert np.sum(np.abs(grad_x)) > 5 * img.shape[1]
    assert np.sum(np.abs(grad_y)) > 5 * img.shape[0]

    # test density
    img_aug_undense = iaa.Snowflakes(density=0.001, density_uniformity=0.99).augment_image(img)
    img_aug_dense = iaa.Snowflakes(density=0.1, density_uniformity=0.99).augment_image(img)
    assert np.average(img_aug_undense) < np.average(img_aug_dense)

    # test density_uniformity
    img_aug_ununiform = iaa.Snowflakes(density=0.4, density_uniformity=0.1).augment_image(img)
    img_aug_uniform = iaa.Snowflakes(density=0.4, density_uniformity=0.9).augment_image(img)

    def _measure_uniformity(image, patch_size=5, n_patches=100):
        pshalf = (patch_size-1) // 2
        grad_x = image[:, 1:].astype(np.float32) - image[:, :-1].astype(np.float32)
        grad_y = image[1:, :].astype(np.float32) - image[:-1, :].astype(np.float32)
        grad = np.abs(grad_x[1:, :] + grad_y[:, 1:])
        points_y = np.random.randint(0, image.shape[0], size=(n_patches,))
        points_x = np.random.randint(0, image.shape[0], size=(n_patches,))
        stds = []
        for y, x in zip(points_y, points_x):
            bb = ia.BoundingBox(x1=x-pshalf, y1=y-pshalf, x2=x+pshalf, y2=y+pshalf)
            patch = bb.extract_from_image(grad)
            stds.append(np.std(patch))
        return 1 / (1+np.std(stds))

    assert _measure_uniformity(img_aug_ununiform) < _measure_uniformity(img_aug_uniform)
