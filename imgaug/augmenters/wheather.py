"""
Augmenters that create wheather effects.

Do not import directly from this file, as the categorization is not final.
Use instead::

    from imgaug import augmenters as iaa

and then e.g.::

    seq = iaa.Sequential([iaa.Snow(...)])

List of augmenters:

    * FastSnowyLandscape

"""
from __future__ import print_function, division, absolute_import

import numpy as np
import cv2

from . import meta
from .. import imgaug as ia
from .. import parameters as iap


class FastSnowyLandscape(meta.Augmenter):
    """
    Augmenter to convert non-snowy landscapes to snowy ones.

    This expects to get an image that roughly shows a landscape.

    This is based on the method proposed by
    https://medium.freecodecamp.org/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f?gi=bca4a13e634c

    Parameters
    ----------
    lightness_threshold : number or tuple of number or list of number\
                          or imgaug.parameters.StochasticParameter, optional
        All pixels with lightness in HLS colorspace below this value will have their lightness increased by
        `lightness_multiplier`.

            * If an int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the discrete range ``[a .. b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    lightness_multiplier : number or tuple of number or list of number\
                           or imgaug.parameters.StochasticParameter, optional
        Multiplier for pixel's lightness value in HLS colorspace. Affects all pixels selected via `lightness_threshold`.

            * If an int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the discrete range ``[a .. b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.FastSnowyLandscape(lightness_threshold=140, lightness_multiplier=2.5)

    Search for all pixels in the image with a lightness value in HLS colorspace of less than 140 and increase their
    lightness by a factor of 2.5. This is the configuration proposed in the original article (see link above).

    >>> aug = iaa.FastSnowyLandscape(lightness_threshold=[128, 200], lightness_multiplier=(1.5, 3.5))

    Search for all pixels in the image with a lightness value in HLS colorspace of less than 128 or less than 200
    (one of these values is picked per image) and multiply their lightness by a factor of ``x`` with ``x`` being
    sampled from ``uniform(1.5, 3.5)`` (once per image).

    >>> aug = iaa.FastSnowyLandscape(lightness_threshold=(100, 255), lightness_multiplier=(1.0, 4.0))

    Similar to above, but the lightness threshold is sampled from ``uniform(100, 255)`` (per image) and the multiplier
    from ``uniform(1.0, 4.0)`` (per image). This seems to produce good and varied results.

    """

    def __init__(self, lightness_threshold=(100, 255), lightness_multiplier=(1.0, 4.0), name=None, deterministic=False,
                 random_state=None):
        super(FastSnowyLandscape, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.lightness_threshold = iap.handle_continuous_param(lightness_threshold, "lightness_threshold",
                                                               value_range=(0, 255),
                                                               tuple_to_uniform=True,
                                                               list_to_choice=True)
        self.lightness_multiplier = iap.handle_continuous_param(lightness_multiplier, "lightness_multiplier",
                                                                value_range=(0, None), tuple_to_uniform=True,
                                                                list_to_choice=True)

    def _draw_samples(self, augmentables, random_state):
        nb_augmentables = len(augmentables)
        rss = ia.derive_random_states(random_state, 2)
        thresh_samples = self.lightness_threshold.draw_samples((nb_augmentables,), rss[1])
        lmul_samples = self.lightness_multiplier.draw_samples((nb_augmentables,), rss[0])
        return thresh_samples, lmul_samples

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)
        thresh_samples, lmul_samples = self._draw_samples(images, random_state)
        result = []

        for image, input_dtype, thresh, lmul in zip(images, input_dtypes, thresh_samples, lmul_samples):
            image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float64)
            lightness = image_hls[..., 1]

            lightness[lightness < thresh] *= lmul

            image_hls = meta.clip_augmented_image_(image_hls, 0, 255)  # TODO make value range more flexible
            image_hls = meta.restore_augmented_image_dtype_(image_hls, input_dtype)
            image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

            result.append(image_rgb)

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.lightness_threshold, self.lightness_multiplier]

