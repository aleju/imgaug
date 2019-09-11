"""
Some utility functions that are only used for unittests.
Placing them in test/ directory seems to be against convention, so they are part of the library.

"""
from __future__ import print_function, division, absolute_import

import random
import copy

import numpy as np
import six.moves as sm
# unittest.mock is not available in 2.7 (though unittest2 might contain it?)
try:
    import unittest.mock as mock
except ImportError:
    import mock

import imgaug as ia
import imgaug.random as iarandom
from imgaug.augmentables.kps import KeypointsOnImage


class ArgCopyingMagicMock(mock.MagicMock):
    """A MagicMock that copies its call args/kwargs before storing the call.

    This is useful for imgaug as many augmentation methods change data
    in-place.

    """

    def _mock_call(_mock_self, *args, **kwargs):
        args_copy = copy.deepcopy(args)
        kwargs_copy = copy.deepcopy(kwargs)
        return super(
            ArgCopyingMagicMock, _mock_self
        )._mock_call(*args_copy, **kwargs_copy)


def create_random_images(size):
    return np.random.uniform(0, 255, size).astype(np.uint8)


def create_random_keypoints(size_images, nb_keypoints_per_img):
    result = []
    for _ in sm.xrange(size_images[0]):
        kps = []
        height, width = size_images[1], size_images[2]
        for _ in sm.xrange(nb_keypoints_per_img):
            x = np.random.randint(0, width-1)
            y = np.random.randint(0, height-1)
            kps.append(ia.Keypoint(x=x, y=y))
        result.append(ia.KeypointsOnImage(kps, shape=size_images[1:]))
    return result


def array_equal_lists(list1, list2):
    assert isinstance(list1, list), (
        "Expected list1 to be a list, got type %s." % (type(list1),))
    assert isinstance(list2, list), (
        "Expected list2 to be a list, got type %s." % (type(list2),))

    if len(list1) != len(list2):
        return False

    for a, b in zip(list1, list2):
        if not np.array_equal(a, b):
            return False

    return True


def keypoints_equal(kps1, kps2, eps=0.001):
    if isinstance(kps1, KeypointsOnImage):
        assert isinstance(kps2, KeypointsOnImage)
        kps1 = [kps1]
        kps2 = [kps2]

    if len(kps1) != len(kps2):
        return False

    for i in sm.xrange(len(kps1)):
        a = kps1[i].keypoints
        b = kps2[i].keypoints
        if len(a) != len(b):
            return False

        for j in sm.xrange(len(a)):
            x_equal = float(b[j].x) - eps <= float(a[j].x) <= float(b[j].x) + eps
            y_equal = float(b[j].y) - eps <= float(a[j].y) <= float(b[j].y) + eps
            if not x_equal or not y_equal:
                return False

    return True


def reseed(seed=0):
    iarandom.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
