from __future__ import print_function, division, absolute_import

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

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np

from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.testutils import reseed


class TestAveragePool(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___default_settings(self):
        aug = iaa.AveragePool(2)
        assert len(aug.kernel_size) == 2
        assert isinstance(aug.kernel_size[0], iap.Deterministic)
        assert aug.kernel_size[0].value == 2
        assert aug.kernel_size[1] is None
        assert aug.keep_size is True

    def test___init___custom_settings(self):
        aug = iaa.AveragePool(((2, 4), (5, 6)), keep_size=False)
        assert len(aug.kernel_size) == 2
        assert isinstance(aug.kernel_size[0], iap.DiscreteUniform)
        assert isinstance(aug.kernel_size[1], iap.DiscreteUniform)
        assert aug.kernel_size[0].a.value == 2
        assert aug.kernel_size[0].b.value == 4
        assert aug.kernel_size[1].a.value == 5
        assert aug.kernel_size[1].b.value == 6
        assert aug.keep_size is False

    def test_augment_images__kernel_size_is_zero(self):
        aug = iaa.AveragePool(0)
        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))
        assert np.array_equal(aug.augment_image(image), image)

    def test_augment_images__kernel_size_is_one(self):
        aug = iaa.AveragePool(1)
        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))
        assert np.array_equal(aug.augment_image(image), image)

    def test_augment_images__kernel_size_is_two__full_100s(self):
        aug = iaa.AveragePool(2, keep_size=False)
        image = np.full((6, 6, 3), 100, dtype=np.uint8)
        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - 100)
        assert image_aug.shape == (3, 3, 3)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_is_two__custom_array(self):
        aug = iaa.AveragePool(2, keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50, 120]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.shape == (1, 2, 3)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_is_two__four_channels(self):
        aug = iaa.AveragePool(2, keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 4))

        expected = np.uint8([
            [50, 120]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 4))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.shape == (1, 2, 4)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_differs(self):
        aug = iaa.AveragePool(
            (iap.Deterministic(3), iap.Deterministic(2)),
            keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+2, 120-1],
            [50-5, 50+5, 120-2, 120+1],
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50, 120]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.shape == (1, 2, 3)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_is_two__keep_size(self):
        aug = iaa.AveragePool(2, keep_size=True)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50, 50, 120, 120],
            [50, 50, 120, 120]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.shape == (2, 4, 3)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_is_two__single_channel(self):
        aug = iaa.AveragePool(2, keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = image[:, :, np.newaxis]

        expected = np.uint8([
            [50, 120]
        ])
        expected = expected[:, :, np.newaxis]

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.shape == (1, 2, 1)
        assert np.all(diff <= 1)

    def test_get_parameters(self):
        aug = iaa.AveragePool(2)
        params = aug.get_parameters()
        assert len(params) == 2
        assert len(params[0]) == 2
        assert isinstance(params[0][0], iap.Deterministic)
        assert params[0][0].value == 2
        assert params[0][1] is None


# We don't have many tests here, because MaxPool and AveragePool derive from
# the same base class, i.e. they share most of the methods, which are then
# tested via TestAveragePool.
class TestMaxPool(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_augment_images(self):
        aug = iaa.MaxPool(2, keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50+2, 120+4]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.shape == (1, 2, 3)
        assert np.all(diff <= 1)

    def test_augment_images__different_channels(self):
        aug = iaa.MaxPool((iap.Deterministic(1), iap.Deterministic(4)),
                          keep_size=False)

        c1 = np.arange(start=1, stop=8+1).reshape((1, 8, 1))
        c2 = (100 + np.arange(start=1, stop=8+1)).reshape((1, 8, 1))
        image = np.dstack([c1, c2]).astype(np.uint8)

        c1_expected = np.uint8([4, 8]).reshape((1, 2, 1))
        c2_expected = np.uint8([100+4, 100+8]).reshape((1, 2, 1))
        image_expected = np.dstack([c1_expected, c2_expected])

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - image_expected)
        assert image_aug.shape == (1, 2, 2)
        assert np.all(diff <= 1)


# We don't have many tests here, because MinPool and AveragePool derive from
# the same base class, i.e. they share most of the methods, which are then
# tested via TestAveragePool.
class TestMinPool(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_augment_images(self):
        aug = iaa.MinPool(2, keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50-2, 120-4]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.shape == (1, 2, 3)
        assert np.all(diff <= 1)

    def test_augment_images__different_channels(self):
        aug = iaa.MinPool((iap.Deterministic(1), iap.Deterministic(4)),
                          keep_size=False)

        c1 = np.arange(start=1, stop=8+1).reshape((1, 8, 1))
        c2 = (100 + np.arange(start=1, stop=8+1)).reshape((1, 8, 1))
        image = np.dstack([c1, c2]).astype(np.uint8)

        c1_expected = np.uint8([1, 5]).reshape((1, 2, 1))
        c2_expected = np.uint8([100+1, 100+4]).reshape((1, 2, 1))
        image_expected = np.dstack([c1_expected, c2_expected])

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - image_expected)
        assert image_aug.shape == (1, 2, 2)
        assert np.all(diff <= 1)
