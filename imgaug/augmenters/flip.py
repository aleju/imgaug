"""
Augmenters that apply mirroring/flipping operations to images.

Do not import directly from this file, as the categorization is not final.
Use instead
    from imgaug import augmenters as iaa
and then e.g.
    seq = iaa.Sequential([
        iaa.Fliplr((0.0, 1.0)),
        iaa.Flipud((0.0, 1.0))
    ])

List of augmenters:
    Fliplr
    Flipud
"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, Binomial, Choice, DiscreteUniform, Normal, Uniform, FromLowerResolution
from .. import parameters as iap
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy as copy_module
import re
import math
from scipy import misc, ndimage
from skimage import transform as tf, segmentation, measure
import itertools
import cv2
import six
import six.moves as sm
import types
import warnings

from .meta import Augmenter

class Fliplr(Augmenter):
    """
    Flip/mirror input images horizontally.

    Parameters
    ----------
    p : int or float or StochasticParameter, optional(default=0)
        Probability of each image to get flipped.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Fliplr(0.5)

    would horizontally flip/mirror 50 percent of all input images.


    >>> aug = iaa.Fliplr(1.0)

    would horizontally flip/mirror all input images.

    """

    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        super(Fliplr, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be int or float or StochasticParameter, got %s." % (type(p),))

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            if samples[i] == 1:
                images[i] = np.fliplr(images[i])
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                width = keypoints_on_image.shape[1]
                for keypoint in keypoints_on_image.keypoints:
                    keypoint.x = (width - 1) - keypoint.x
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]


class Flipud(Augmenter):
    """
    Flip/mirror input images vertically.

    Parameters
    ----------
    p : int or float or StochasticParameter, optional(default=0)
        Probability of each image to get flipped.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Flipud(0.5)

    would vertically flip/mirror 50 percent of all input images.

    >>> aug = iaa.Flipud(1.0)

    would vertically flip/mirror all input images.

    """

    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        super(Flipud, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be int or float or StochasticParameter, got %s." % (type(p),))

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            if samples[i] == 1:
                images[i] = np.flipud(images[i])
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                height = keypoints_on_image.shape[0]
                for keypoint in keypoints_on_image.keypoints:
                    keypoint.y = (height - 1) - keypoint.y
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]
