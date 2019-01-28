from __future__ import print_function, division, absolute_import

import time

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm
import cv2

from imgaug import augmenters as iaa
from imgaug.testutils import reseed


def main():
    time_start = time.time()

    # TODO WithColorspace
    test_AddToHueAndSaturation()
    # TODO ChangeColorspace
    test_Grayscale()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_AddToHueAndSaturation():
    reseed()

    # interestingly, when using this RGB2HSV and HSV2RGB conversion from skimage, the results
    # differ quite a bit from the cv2 ones
    """
    def _add_hue_saturation(img, value):
        img_hsv = color.rgb2hsv(img / 255.0)
        img_hsv[..., 0:2] += (value / 255.0)
        return color.hsv2rgb(img_hsv) * 255
    """

    def _add_hue_saturation(img, value):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv = img_hsv.astype(np.int32)
        img_hsv[..., 0] = np.mod(img_hsv[..., 0] + (value/255.0) * (360/2), 180)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + value, 0, 255)
        img_hsv = img_hsv.astype(np.uint8)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    base_img = np.zeros((2, 2, 3), dtype=np.uint8)
    base_img[..., 0] += 20
    base_img[..., 1] += 40
    base_img[..., 2] += 60

    for per_channel in [False, True]:
        for backend in ["cv2", "numpy"]:
            aug = iaa.AddToHueAndSaturation(0, per_channel=per_channel)
            aug.backend = backend
            observed = aug.augment_image(base_img)
            expected = base_img
            assert np.allclose(observed, expected)

            aug = iaa.AddToHueAndSaturation(30, per_channel=per_channel)
            aug.backend = backend
            observed = aug.augment_image(base_img)
            expected = _add_hue_saturation(base_img, 30)
            diff = np.abs(observed.astype(np.float32) - expected)
            assert np.all(diff <= 1)

            aug = iaa.AddToHueAndSaturation(255, per_channel=per_channel)
            aug.backend = backend
            observed = aug.augment_image(base_img)
            expected = _add_hue_saturation(base_img, 255)
            diff = np.abs(observed.astype(np.float32) - expected)
            assert np.all(diff <= 1)

            aug = iaa.AddToHueAndSaturation(-255, per_channel=per_channel)
            aug.backend = backend
            observed = aug.augment_image(base_img)
            expected = _add_hue_saturation(base_img, -255)
            diff = np.abs(observed.astype(np.float32) - expected)
            assert np.all(diff <= 1)

    aug = iaa.AddToHueAndSaturation([0, 10, 20])
    base_img = base_img[0:1, 0:1, :]
    expected_imgs = [
        iaa.AddToHueAndSaturation(0).augment_image(base_img),
        iaa.AddToHueAndSaturation(10).augment_image(base_img),
        iaa.AddToHueAndSaturation(20).augment_image(base_img)
    ]

    assert not np.array_equal(expected_imgs[0], expected_imgs[1])
    assert not np.array_equal(expected_imgs[1], expected_imgs[2])
    assert not np.array_equal(expected_imgs[0], expected_imgs[2])
    nb_iterations = 300
    seen = dict([(i, 0) for i, _ in enumerate(expected_imgs)])
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        for i, expected_img in enumerate(expected_imgs):
            if np.allclose(observed, expected_img):
                seen[i] += 1
    assert np.sum(list(seen.values())) == nb_iterations
    n_exp = nb_iterations / 3
    n_exp_tol = nb_iterations * 0.1
    assert all([n_exp - n_exp_tol < v < n_exp + n_exp_tol for v in seen.values()])


def test_Grayscale():
    reseed()

    def _compute_luminosity(r, g, b):
        return 0.21 * r + 0.72 * g + 0.07 * b

    base_img = np.zeros((4, 4, 3), dtype=np.uint8)
    base_img[..., 0] += 10
    base_img[..., 1] += 20
    base_img[..., 2] += 30

    aug = iaa.Grayscale(0.0)
    observed = aug.augment_image(base_img)
    expected = np.copy(base_img)
    assert np.allclose(observed, expected)

    aug = iaa.Grayscale(1.0)
    observed = aug.augment_image(base_img)
    luminosity = _compute_luminosity(10, 20, 30)
    expected = np.zeros_like(base_img) + luminosity
    assert np.allclose(observed, expected.astype(np.uint8))

    aug = iaa.Grayscale(0.5)
    observed = aug.augment_image(base_img)
    luminosity = _compute_luminosity(10, 20, 30)
    expected = 0.5 * base_img + 0.5 * luminosity
    assert np.allclose(observed, expected.astype(np.uint8))

    aug = iaa.Grayscale((0.0, 1.0))
    base_img = np.uint8([255, 0, 0]).reshape((1, 1, 3))
    base_img_float = base_img.astype(np.float64) / 255.0
    base_img_gray = iaa.Grayscale(1.0).augment_image(base_img).astype(np.float64) / 255.0
    distance_max = np.linalg.norm(base_img_gray.flatten() - base_img_float.flatten())
    nb_iterations = 1000
    distances = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img).astype(np.float64) / 255.0
        distance = np.linalg.norm(observed.flatten() - base_img_float.flatten()) / distance_max
        distances.append(distance)

    assert 0 - 1e-4 < min(distances) < 0.1
    assert 0.4 < np.average(distances) < 0.6
    assert 0.9 < max(distances) < 1.0 + 1e-4

    nb_bins = 5
    hist, _ = np.histogram(distances, bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / nb_iterations
        assert density_expected - density_tolerance < density < density_expected + density_tolerance


if __name__ == "__main__":
    main()
