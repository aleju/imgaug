"""
Some utility functions that are only used for unittests.
Placing them in test/ directory seems to be against convention, so they are part of the library.

"""
from __future__ import print_function, division, absolute_import

import random
import copy
import warnings

import numpy as np
import six.moves as sm
# unittest.mock is not available in 2.7 (though unittest2 might contain it?)
try:
    import unittest.mock as mock
except ImportError:
    import mock
try:
    import cPickle as pickle
except ImportError:
    import pickle

import imgaug as ia
import imgaug.random as iarandom
from imgaug.augmentables.kps import KeypointsOnImage


class ArgCopyingMagicMock(mock.MagicMock):
    """A MagicMock that copies its call args/kwargs before storing the call.

    This is useful for imgaug as many augmentation methods change data
    in-place.

    Taken from https://stackoverflow.com/a/23264042/3760780

    """

    def _mock_call(self, *args, **kwargs):
        args_copy = copy.deepcopy(args)
        kwargs_copy = copy.deepcopy(kwargs)
        return super(ArgCopyingMagicMock, self)._mock_call(
            *args_copy, **kwargs_copy)


def assert_cbaois_equal(observed, expected, max_distance=1e-4):
    # pylint: disable=unidiomatic-typecheck
    if isinstance(observed, list) or isinstance(expected, list):
        assert isinstance(observed, list)
        assert isinstance(expected, list)
        assert len(observed) == len(expected)
        for observed_i, expected_i in zip(observed, expected):
            assert_cbaois_equal(observed_i, expected_i,
                                max_distance=max_distance)
    else:
        assert type(observed) == type(expected)
        assert len(observed.items) == len(expected.items)
        assert observed.shape == expected.shape
        for item_a, item_b in zip(observed.items, expected.items):
            assert item_a.coords_almost_equals(item_b,
                                               max_distance=max_distance)
        if isinstance(expected, ia.PolygonsOnImage):
            for item_obs, item_exp in zip(observed.items, expected.items):
                if item_exp.is_valid:
                    assert item_obs.is_valid


# TODO remove this function, no longer needed since shift interfaces were
#      standardized
def shift_cbaoi(cbaoi, x=0, y=0, top=0, right=0, bottom=0, left=0):
    return cbaoi.shift(x=x+left-right, y=y+top-bottom)


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

    for arr1, arr2 in zip(list1, list2):
        if not np.array_equal(arr1, arr2):
            return False

    return True


def keypoints_equal(kpsois1, kpsois2, eps=0.001):
    if isinstance(kpsois1, KeypointsOnImage):
        assert isinstance(kpsois2, KeypointsOnImage)
        kpsois1 = [kpsois1]
        kpsois2 = [kpsois2]

    if len(kpsois1) != len(kpsois2):
        return False

    for kpsoi1, kpsoi2 in zip(kpsois1, kpsois2):
        kps1 = kpsoi1.keypoints
        kps2 = kpsoi2.keypoints
        if len(kps1) != len(kps2):
            return False

        for kp1, kp2 in zip(kps1, kps2):
            x_equal = (float(kp2.x) - eps
                       <= float(kp1.x)
                       <= float(kp2.x) + eps)
            y_equal = (float(kp2.y) - eps
                       <= float(kp1.y)
                       <= float(kp2.y) + eps)
            if not x_equal or not y_equal:
                return False

    return True


def reseed(seed=0):
    iarandom.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def runtest_pickleable_uint8_img(augmenter, shape=(15, 15, 3), iterations=3):
    image = np.mod(np.arange(int(np.prod(shape))), 256).astype(np.uint8)
    image = image.reshape(shape)
    augmenter_pkl = pickle.loads(pickle.dumps(augmenter, protocol=-1))

    for _ in np.arange(iterations):
        image_aug = augmenter(image=image)
        image_aug_pkl = augmenter_pkl(image=image)
        assert np.array_equal(image_aug, image_aug_pkl)


def wrap_shift_deprecation(func, *args, **kwargs):
    """Helper for tests of CBA shift() functions."""
    # No deprecated arguments? Just call the functions directly.
    deprecated_kwargs = ["top", "right", "bottom", "left"]
    if not any([kwname in kwargs for kwname in deprecated_kwargs]):
        return func()

    # Deprecated arguments? Log warnings and assume that there was a
    # deprecation warning with expected message.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        result = func()

        assert (
            "These are deprecated. Use `x` and `y` instead."
            in str(caught_warnings[-1].message)
        )

        return result
