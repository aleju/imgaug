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
from imgaug.testutils import reseed, runtest_pickleable_uint8_img


class TestRandAugment(unittest.TestCase):
    def setUp(self):
        reseed()

    # for some reason these mocks don't work with
    # imgaug.augmenters.collections.(...)
    @mock.patch("imgaug.augmenters.RandAugment._create_initial_augmenters_list")
    @mock.patch("imgaug.augmenters.RandAugment._create_main_augmenters_list")
    def test_n(self, mock_main, mock_initial):
        mock_main.return_value = [iaa.Add(1), iaa.Add(2), iaa.Add(4)]
        mock_initial.return_value = []

        img = np.zeros((1, 1, 3), dtype=np.uint8)
        expected = {
            0: [0],
            1: [1, 2, 4],
            2: [1+1, 1+2, 1+4, 2+2, 2+4, 4+4]
        }

        for n in [0, 1, 2]:
            with self.subTest(n=n):
                aug = iaa.RandAugment(n=n)
                img_aug = aug(image=img)
                assert img_aug[0, 0, 0] in expected[n]

    # for some reason these mocks don't work with
    # imgaug.augmenters.collections.(...)
    @mock.patch("imgaug.augmenters.RandAugment._create_initial_augmenters_list")
    @mock.patch("imgaug.augmenters.RandAugment._create_main_augmenters_list")
    def test_m(self, mock_main, mock_initial):
        def _create_main_list(m, _cval):
            return [iaa.Add(m)]

        mock_main.side_effect = _create_main_list
        mock_initial.return_value = []

        img = np.zeros((1, 1, 3), dtype=np.uint8)

        for m in [0, 1, 2]:
            with self.subTest(m=m):
                aug = iaa.RandAugment(m=m)
                img_aug = aug(image=img)
                assert img_aug[0, 0, 0] == m

    def test_cval(self):
        cval = 200
        aug = iaa.RandAugment(n=1, m=30, cval=cval)
        img = np.zeros((20, 20, 3), dtype=np.uint8)

        x_cval = False
        y_cval = False

        # lots of iterations here, because only in some iterations an affine
        # translation is actually applied
        for _ in np.arange(500):
            img_aug = aug(image=img)

            x_cval = x_cval or np.all(img_aug[:, :1] == cval)
            x_cval = x_cval or np.all(img_aug[:, -1:] == cval)
            y_cval = y_cval or np.all(img_aug[:1, :] == cval)
            y_cval = y_cval or np.all(img_aug[-1:, :] == cval)

            if np.all([x_cval, y_cval]):
                break

        assert np.all([x_cval, y_cval])

    def test_get_parameters(self):
        aug = iaa.RandAugment(n=1, m=30, cval=100)
        params = aug.get_parameters()
        assert params[0] is aug[1].n
        assert params[1] is aug._m
        assert params[2] is aug._cval

    def test_pickleable(self):
        aug = iaa.RandAugment(m=(0, 10), n=(1, 2))
        runtest_pickleable_uint8_img(aug, iterations=50)
