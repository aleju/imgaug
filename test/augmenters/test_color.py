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

    # TODO WithColorspace
    # TODO ChangeColorspace
    test_Grayscale()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


class TestAddToHueAndSaturation(unittest.TestCase):
    def setUp(self):
        reseed()

    # interestingly, when using this RGB2HSV and HSV2RGB conversion from
    # skimage, the results differ quite a bit from the cv2 ones
    """
    def _add_hue_saturation(img, value):
        img_hsv = color.rgb2hsv(img / 255.0)
        img_hsv[..., 0:2] += (value / 255.0)
        return color.hsv2rgb(img_hsv) * 255
    """

    @classmethod
    def _add_hue_saturation(cls, img, value=None, value_hue=None,
                            value_saturation=None):
        if value is not None:
            assert value_hue is None and value_saturation is None
        else:
            assert value_hue is not None or value_saturation is not None

        if value is not None:
            value_hue = value
            value_saturation = value
        else:
            value_hue = value_hue or 0
            value_saturation = value_saturation or 0

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv = img_hsv.astype(np.int32)
        img_hsv[..., 0] = np.mod(
            img_hsv[..., 0] + (value_hue/255.0) * (360/2), 180)
        img_hsv[..., 1] = np.clip(
            img_hsv[..., 1] + value_saturation, 0, 255)
        img_hsv = img_hsv.astype(np.uint8)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    def test___init__(self):
        aug = iaa.AddToHueAndSaturation((-20, 20))
        assert isinstance(aug.value, iap.DiscreteUniform)
        assert aug.value.a.value == -20
        assert aug.value.b.value == 20
        assert aug.value_hue is None
        assert aug.value_saturation is None
        assert isinstance(aug.per_channel, iap.Deterministic)
        assert aug.per_channel.value == 0

    def test___init___value_none(self):
        aug = iaa.AddToHueAndSaturation(value_hue=(-20, 20),
                                        value_saturation=[0, 5, 10])
        assert aug.value is None
        assert isinstance(aug.value_hue, iap.DiscreteUniform)
        assert isinstance(aug.value_saturation, iap.Choice)
        assert aug.value_hue.a.value == -20
        assert aug.value_hue.b.value == 20
        assert aug.value_saturation.a == [0, 5, 10]
        assert isinstance(aug.per_channel, iap.Deterministic)
        assert aug.per_channel.value == 0

    def test___init___per_channel(self):
        aug = iaa.AddToHueAndSaturation(per_channel=0.5)
        assert aug.value is None
        assert aug.value_hue is None
        assert aug.value_saturation is None
        assert isinstance(aug.per_channel, iap.Binomial)
        assert np.isclose(aug.per_channel.p.value, 0.5)

    def test_augment_images(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        gen = itertools.product([False, True], ["cv2", "numpy"])
        for per_channel, backend in gen:
            with self.subTest(per_channel=per_channel, backend=backend):
                aug = iaa.AddToHueAndSaturation(0, per_channel=per_channel)
                aug.backend = backend
                observed = aug.augment_image(base_img)
                expected = base_img
                assert np.allclose(observed, expected)

                aug = iaa.AddToHueAndSaturation(30, per_channel=per_channel)
                aug.backend = backend
                observed = aug.augment_image(base_img)
                expected = self._add_hue_saturation(base_img, 30)
                diff = np.abs(observed.astype(np.float32) - expected)
                assert np.all(diff <= 1)

                aug = iaa.AddToHueAndSaturation(255, per_channel=per_channel)
                aug.backend = backend
                observed = aug.augment_image(base_img)
                expected = self._add_hue_saturation(base_img, 255)
                diff = np.abs(observed.astype(np.float32) - expected)
                assert np.all(diff <= 1)

                aug = iaa.AddToHueAndSaturation(-255, per_channel=per_channel)
                aug.backend = backend
                observed = aug.augment_image(base_img)
                expected = self._add_hue_saturation(base_img, -255)
                diff = np.abs(observed.astype(np.float32) - expected)
                assert np.all(diff <= 1)

    def test_augment_images__different_hue_and_saturation__no_per_channel(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                arr = np.float32([10, 20])
                return np.tile(arr[np.newaxis, :], (size[0], 1))

        aug = iaa.AddToHueAndSaturation(value=_DummyParam(), per_channel=False)
        img_expected = self._add_hue_saturation(base_img, value=10)
        img_observed = aug.augment_image(base_img)

        assert np.array_equal(img_observed, img_expected)

    def test_augment_images__different_hue_and_saturation__per_channel(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                arr = np.float32([10, 20])
                return np.tile(arr[np.newaxis, :], (size[0], 1))

        aug = iaa.AddToHueAndSaturation(value=_DummyParam(), per_channel=True)
        img_expected = self._add_hue_saturation(
            base_img, value_hue=10, value_saturation=20)
        img_observed = aug.augment_image(base_img)

        assert np.array_equal(img_observed, img_expected)

    def test_augment_images__different_hue_and_saturation__mixed_perchan(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParamValue(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                arr = np.float32([10, 20])
                return np.tile(arr[np.newaxis, :], (size[0], 1))

        class _DummyParamPerChannel(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                assert size == (3,)
                return np.float32([1.0, 0.0, 1.0])

        aug = iaa.AddToHueAndSaturation(
            value=_DummyParamValue(), per_channel=_DummyParamPerChannel())

        img_expected1 = self._add_hue_saturation(
            base_img, value_hue=10, value_saturation=20)
        img_expected2 = self._add_hue_saturation(
            base_img, value_hue=10, value_saturation=10)
        img_expected3 = self._add_hue_saturation(
            base_img, value_hue=10, value_saturation=20)

        img_observed1, img_observed2, img_observed3, = \
            aug.augment_images([base_img] * 3)

        assert np.array_equal(img_observed1, img_expected1)
        assert np.array_equal(img_observed2, img_expected2)
        assert np.array_equal(img_observed3, img_expected3)

    def test_augment_images__list_as_value(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

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
        assert all([n_exp - n_exp_tol < v < n_exp + n_exp_tol
                    for v in seen.values()])

    def test_augment_images__value_hue(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                return np.float32([10, 20, 30])

        aug = iaa.AddToHueAndSaturation(value_hue=_DummyParam())

        img_expected1 = self._add_hue_saturation(base_img, value_hue=10)
        img_expected2 = self._add_hue_saturation(base_img, value_hue=20)
        img_expected3 = self._add_hue_saturation(base_img, value_hue=30)

        img_observed1, img_observed2, img_observed3 = \
            aug.augment_images([base_img] * 3)

        assert np.array_equal(img_observed1, img_expected1)
        assert np.array_equal(img_observed2, img_expected2)
        assert np.array_equal(img_observed3, img_expected3)

    def test_augment_images__value_saturation(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                return np.float32([10, 20, 30])

        aug = iaa.AddToHueAndSaturation(value_saturation=_DummyParam())

        img_expected1 = self._add_hue_saturation(base_img, value_saturation=10)
        img_expected2 = self._add_hue_saturation(base_img, value_saturation=20)
        img_expected3 = self._add_hue_saturation(base_img, value_saturation=30)

        img_observed1, img_observed2, img_observed3 = \
            aug.augment_images([base_img] * 3)

        assert np.array_equal(img_observed1, img_expected1)
        assert np.array_equal(img_observed2, img_expected2)
        assert np.array_equal(img_observed3, img_expected3)

    def test_augment_images__value_hue_and_value_saturation(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                return np.float32([10, 20, 30])

        aug = iaa.AddToHueAndSaturation(value_hue=_DummyParam(),
                                        value_saturation=_DummyParam()+40)

        img_expected1 = self._add_hue_saturation(base_img, value_hue=10,
                                                 value_saturation=40+10)
        img_expected2 = self._add_hue_saturation(base_img, value_hue=20,
                                                 value_saturation=40+20)
        img_expected3 = self._add_hue_saturation(base_img, value_hue=30,
                                                 value_saturation=40+30)

        img_observed1, img_observed2, img_observed3 = \
            aug.augment_images([base_img] * 3)

        assert np.array_equal(img_observed1, img_expected1)
        assert np.array_equal(img_observed2, img_expected2)
        assert np.array_equal(img_observed3, img_expected3)

    def test_get_parameters(self):
        aug = iaa.AddToHueAndSaturation((-20, 20), per_channel=0.5)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.DiscreteUniform)
        assert params[0].a.value == -20
        assert params[0].b.value == 20
        assert params[1] is None
        assert params[2] is None
        assert isinstance(params[3], iap.Binomial)
        assert np.isclose(params[3].p.value, 0.5)

    def test_get_parameters_value_hue_and_value_saturation(self):
        aug = iaa.AddToHueAndSaturation(value_hue=(-20, 20),
                                        value_saturation=5)
        params = aug.get_parameters()
        assert params[0] is None
        assert isinstance(params[1], iap.DiscreteUniform)
        assert params[1].a.value == -20
        assert params[1].b.value == 20
        assert isinstance(params[2], iap.Deterministic)
        assert params[2].value == 5
        assert isinstance(params[3], iap.Deterministic)
        assert params[3].value == 0


class TestAddToHue(unittest.TestCase):
    def test_returns_correct_class(self):
        aug = iaa.AddToHue((-20, 20))
        assert isinstance(aug, iaa.AddToHueAndSaturation)
        assert isinstance(aug.value_hue, iap.DiscreteUniform)
        assert aug.value_hue.a.value == -20
        assert aug.value_hue.b.value == 20


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
