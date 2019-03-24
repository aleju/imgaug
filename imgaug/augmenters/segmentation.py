"""
Augmenters that apply changes to images based on forms of segmentation.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([
        iaa.Superpixels(...)
    ])

List of augmenters:

    * Superpixels

"""
from __future__ import print_function, division, absolute_import

import numpy as np
from skimage import segmentation, measure
import six.moves as sm

from . import meta
import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt


class Superpixels(meta.Augmenter):
    """
    Completely or partially transform images to their superpixel representation.

    This implementation uses skimage's version of the SLIC algorithm.

    dtype support::

        if (image size <= max_size)::

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested
            * ``uint64``: limited (1)
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested
            * ``int64``: limited (1)
            * ``float16``: no (2)
            * ``float32``: no (2)
            * ``float64``: no (3)
            * ``float128``: no (2)
            * ``bool``: yes; tested

            - (1) Superpixel mean intensity replacement requires computing these means as float64s.
                  This can cause inaccuracies for large integer values.
            - (2) Error in scikit-image.
            - (3) Loss of resolution in scikit-image.

        if (image size > max_size)::

            minimum of (
                ``imgaug.augmenters.segmentation.Superpixels(image size <= max_size)``,
                :func:`imgaug.imgaug.imresize_many_images`
            )

    Parameters
    ----------
    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines the probability of any superpixel area being replaced by the
        superpixel, i.e. by the average pixel color within its area:

            * A probability of 0 would mean, that no superpixel area is replaced by
              its average (image is not changed at all).
            * A probability of 0.5 would mean, that half of all superpixels are
              replaced by their average color.
            * A probability of 1.0 would mean, that all superpixels are replaced
              by their average color (resulting in a standard superpixel image).

        Behaviour based on chosen datatypes for this parameter:

            * If number, then that numbre will always be used.
            * If tuple ``(a, b)``, then a random probability will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If this parameter is a StochasticParameter, it is expected to return
              values between 0 and 1. Values ``>=0.5`` will be interpreted as the command
              to replace a superpixel region with its mean. Recommended to be some
              form of ``Binomial(...)``.

    n_segments : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Target number of superpixels to generate. Lower numbers are faster.

            * If a single int, then that value will always be used as the
              number of segments.
            * If a tuple ``(a, b)``, then a value from the discrete interval ``[a..b]``
              will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then that parameter will be queried to
              draw one value per image.

    max_size : int or None, optional
        Maximum image size at which the superpixels are generated.
        If the width or height of an image exceeds this value, it will be
        downscaled so that the longest side matches `max_size`.
        Though, the final output (superpixel) image has the same size as the
        input image.
        This is done to speed up the superpixel algorithm.
        Use None to apply no downscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`imgaug.imgaug.imresize_single_image`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Superpixels(p_replace=1.0, n_segments=64)

    generates ~64 superpixels per image and replaces all of them with
    their average color (standard superpixel image).

    >>> aug = iaa.Superpixels(p_replace=0.5, n_segments=64)

    generates always ~64 superpixels per image and replaces half of them
    with their average color, while the other half are left unchanged (i.e.
    they still show the input image's content).

    >>> aug = iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128))

    generates between ~16 and ~128 superpixels per image and replaces
    25 to 100 percent of them with their average color.

    """

    def __init__(self, p_replace=0, n_segments=100, max_size=128, interpolation="linear",
                 name=None, deterministic=False, random_state=None):
        super(Superpixels, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.p_replace = iap.handle_probability_param(p_replace, "p_replace", tuple_to_uniform=True,
                                                      list_to_choice=True)
        self.n_segments = iap.handle_discrete_param(n_segments, "n_segments", value_range=(1, None),
                                                    tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.max_size = max_size
        self.interpolation = interpolation

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"],
                         disallowed=["uint128", "uint256", "int128", "int256",
                                     "float16", "float32", "float64", "float96", "float128", "float256"],
                         augmenter=self)

        nb_images = len(images)
        rss = ia.derive_random_states(random_state, 1+nb_images)
        n_segments_samples = self.n_segments.draw_samples((nb_images,), random_state=rss[0])
        for i, (image, rs) in enumerate(zip(images, rss[1:])):
            # TODO this results in an error when n_segments is 0
            replace_samples = self.p_replace.draw_samples((n_segments_samples[i],), random_state=rs)

            if np.max(replace_samples) == 0:
                # not a single superpixel would be replaced by its average color,
                # i.e. the image would not be changed, so just keep it
                pass
            else:
                image = images[i]
                min_value, _center_value, max_value = iadt.get_value_range_of_dtype(image.dtype)

                orig_shape = image.shape
                if self.max_size is not None:
                    size = max(image.shape[0], image.shape[1])
                    if size > self.max_size:
                        resize_factor = self.max_size / size
                        new_height, new_width = int(image.shape[0] * resize_factor), int(image.shape[1] * resize_factor)
                        image = ia.imresize_single_image(image, (new_height, new_width),
                                                         interpolation=self.interpolation)

                image_sp = np.copy(image)
                segments = segmentation.slic(image, n_segments=n_segments_samples[i], compactness=10)
                nb_channels = image.shape[2]
                for c in sm.xrange(nb_channels):
                    # segments+1 here because otherwise regionprops always misses
                    # the last label
                    regions = measure.regionprops(segments+1, intensity_image=image[..., c])
                    for ridx, region in enumerate(regions):
                        # with mod here, because slic can sometimes create more superpixel
                        # than requested. replace_samples then does not have enough
                        # values, so we just start over with the first one again.
                        if replace_samples[ridx % len(replace_samples)] >= 0.5:
                            mean_intensity = region.mean_intensity
                            image_sp_c = image_sp[..., c]
                            if image_sp_c.dtype.kind in ["i", "u", "b"]:
                                # After rounding the value can end up slightly outside of the value_range.
                                # Hence, we need to clip. We do clip via min(max(...)) instead of np.clip
                                # because the latter one does not seem to keep dtypes for dtypes with
                                # large itemsizes (e.g. uint64).
                                value = int(np.round(mean_intensity))
                                value = min(max(value, min_value), max_value)
                                image_sp_c[segments == ridx] = value
                            else:
                                image_sp_c[segments == ridx] = mean_intensity

                if orig_shape != image.shape:
                    image_sp = ia.imresize_single_image(image_sp, orig_shape[0:2], interpolation=self.interpolation)

                images[i] = image_sp
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.p_replace, self.n_segments, self.max_size, self.interpolation]
