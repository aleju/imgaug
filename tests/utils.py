from __future__ import print_function, division, absolute_import

# fix execution of tests involving matplotlib on travis
import matplotlib
matplotlib.use('Agg')

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmenters import contrast as contrast_lib
import numpy as np
import random
import six.moves as sm
import skimage
import skimage.data
import cv2
import time
import scipy
import copy
import warnings
import itertools
import shapely
import shapely.geometry

#from nose.plugins.attrib import attr


def create_random_images(size):
    return np.random.uniform(0, 255, size).astype(np.uint8)


def create_random_keypoints(size_images, nb_keypoints_per_img):
    result = []
    for i in sm.xrange(size_images[0]):
        kps = []
        height, width = size_images[1], size_images[2]
        for i in sm.xrange(nb_keypoints_per_img):
            x = np.random.randint(0, width-1)
            y = np.random.randint(0, height-1)
            kps.append(ia.Keypoint(x=x, y=y))
        result.append(ia.KeypointsOnImage(kps, shape=size_images[1:]))
    return result


def array_equal_lists(list1, list2):
    assert isinstance(list1, list)
    assert isinstance(list2, list)

    if len(list1) != len(list2):
        return False

    for a, b in zip(list1, list2):
        if not np.array_equal(a, b):
            return False

    return True


def keypoints_equal(kps1, kps2, eps=0.001):
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
    ia.seed(seed)
    np.random.seed(seed)
    random.seed(seed)