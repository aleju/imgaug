"""
Augmenters that apply mirroring/flipping operations to images.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([
        iaa.Fliplr((0.0, 1.0)),
        iaa.Flipud((0.0, 1.0))
    ])

List of augmenters:

    * Fliplr
    * Flipud

"""
from __future__ import print_function, division, absolute_import

import numpy as np

from . import meta
from .. import parameters as iap


def HorizontalFlip(*args, **kwargs):
    """Alias for Fliplr."""
    return Fliplr(*args, **kwargs)


def VerticalFlip(*args, **kwargs):
    """Alias for Flipud."""
    return Flipud(*args, **kwargs)


class Fliplr(meta.Augmenter):  # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Flip/mirror input images horizontally.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    p : number or imgaug.parameters.StochasticParameter, optional
        Probability of each image to get flipped.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Fliplr(0.5)

    would horizontally flip/mirror 50 percent of all input images.


    >>> aug = iaa.Fliplr(1.0)

    would horizontally flip/mirror all input images.

    """

    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        super(Fliplr, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.p = iap.handle_probability_param(p, "p")

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, (image, sample) in enumerate(zip(images, samples)):
            if sample > 0.5:
                images[i] = np.fliplr(image)
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        arrs_flipped = self._augment_images(
            [heatmaps_i.arr_0to1 for heatmaps_i in heatmaps],
            random_state=random_state,
            parents=parents,
            hooks=hooks
        )
        for heatmaps_i, arr_flipped in zip(heatmaps, arrs_flipped):
            heatmaps_i.arr_0to1 = arr_flipped
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if not keypoints_on_image.keypoints:
                continue
            elif samples[i] == 1:
                width = keypoints_on_image.shape[1]
                for keypoint in keypoints_on_image.keypoints:
                    # TODO is this still correct with float keypoints? Seems like the -1 should be dropped
                    keypoint.x = (width - 1) - keypoint.x
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]


# TODO merge with Fliplr
class Flipud(meta.Augmenter):  # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Flip/mirror input images vertically.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    p : number or imgaug.parameters.StochasticParameter, optional
        Probability of each image to get flipped.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Flipud(0.5)

    would vertically flip/mirror 50 percent of all input images.

    >>> aug = iaa.Flipud(1.0)

    would vertically flip/mirror all input images.

    """

    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        super(Flipud, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.p = iap.handle_probability_param(p, "p")

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, (image, sample) in enumerate(zip(images, samples)):
            if sample > 0.5:
                images[i] = np.flipud(image)
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        arrs_flipped = self._augment_images(
            [heatmaps_i.arr_0to1 for heatmaps_i in heatmaps],
            random_state=random_state,
            parents=parents,
            hooks=hooks
        )
        for heatmaps_i, arr_flipped in zip(heatmaps, arrs_flipped):
            heatmaps_i.arr_0to1 = arr_flipped
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if not keypoints_on_image.keypoints:
                continue
            elif samples[i] == 1:
                height = keypoints_on_image.shape[0]
                for keypoint in keypoints_on_image.keypoints:
                    # TODO is this still correct with float keypoints? seems like the -1 should be dropped
                    keypoint.y = (height - 1) - keypoint.y
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]
