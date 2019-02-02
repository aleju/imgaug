"""
Augmenters that blend two images with each other.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([
        iaa.Alpha(0.5, iaa.Add((-5, 5)))
    ])

List of augmenters:

    * Alpha
    * AlphaElementwise
    * SimplexNoiseAlpha
    * FrequencyNoiseAlpha

"""
from __future__ import print_function, division, absolute_import

import numpy as np
import six.moves as sm

from . import meta
from .. import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt


def blend_alpha(image_fg, image_bg, alpha, eps=1e-2):
    """
    Blend two images using an alpha blending.

    In an alpha blending, the two images are naively mixed. Let ``A`` be the foreground image
    and ``B`` the background image and ``a`` is the alpha value. Each pixel intensity is then
    computed as ``a * A_ij + (1-a) * B_ij``.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested (1)
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested (1)
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested (1)
        * ``float128``: no (2)
        * ``bool``: yes; fully tested (2)

        - (1) Tests show that these dtypes work, but a conversion to float128 happens, which only
              has 96 bits of size instead of true 128 bits and hence not twice as much resolution.
              It is possible that these dtypes result in inaccuracies, though the tests did not
              indicate that.
        - (2) Not available due to the input dtype having to be increased to an equivalent float
              dtype with two times the input resolution.
        - (3) Mapped internally to ``float16``.

    Parameters
    ----------
    image_fg : (H,W,C) ndarray
        Foreground image. Channel axis must be provided. Shape and dtype kind must match the one
        of the background image.

    image_bg : (H,W,C) ndarray
        Background image. Channel axis must be provided. Shape and dtype kind must match the one
        of the foreground image.

    alpha : number or iterable of number or ndarray
        The blending factor, between 0.0 and 1.0. Can be interpreted as the opacity of the
        foreground image. Values around 1.0 result in only the foreground image being visible.
        Values around 0.0 result in only the background image being visible.
        Multiple alphas may be provided. In these cases, there must be exactly one alpha per
        channel in the foreground/background image. Alternatively, for ``(H,W,C)`` images,
        either one ``(H,W)`` array or an ``(H,W,C)`` array of alphas may be provided,
        denoting the elementwise alpha value.

    eps : number, optional
        Controls when an alpha is to be interpreted as exactly 1.0 or exactly 0.0, resulting
        in only the foreground/background being visible and skipping the actual computation.

    Returns
    -------
    image_blend : (H,W,C) ndarray
        Blend of foreground and background image.

    """
    assert image_fg.shape == image_bg.shape
    assert image_fg.dtype.kind == image_bg.dtype.kind
    # TODO switch to gate_dtypes()
    assert image_fg.dtype not in [np.float128]
    assert image_bg.dtype not in [np.float128]

    input_was_bool = False
    if image_fg.dtype.kind == "b":
        input_was_bool = True
        # use float32 instead of float16 here because it seems to be faster
        image_fg = image_fg.astype(np.float32)
        image_bg = image_bg.astype(np.float32)

    alpha = np.array(alpha, dtype=np.float64)
    if alpha.size == 1:
        pass
    else:
        if alpha.ndim == 2:
            assert alpha.shape == image_fg.shape[0:2]
            alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        elif alpha.ndim == 3:
            assert alpha.shape == image_fg.shape or alpha.shape == image_fg.shape[0:2] + (1,)
        else:
            alpha = alpha.reshape((1, 1, -1))
        if alpha.shape[2] != image_fg.shape[2]:
            alpha = np.tile(alpha, (1, 1, image_fg.shape[2]))

    if not input_was_bool:
        if np.all(alpha >= 1.0 - eps):
            return np.copy(image_fg)
        elif np.all(alpha <= eps):
            return np.copy(image_bg)

    # for efficiency reaons, only test one value of alpha here, even if alpha is much larger
    assert 0 <= alpha.item(0) <= 1.0

    dt_images = iadt.get_minimal_dtype([image_fg, image_bg])

    # doing this only for non-float images led to inaccuracies for large floats values
    isize = dt_images.itemsize * 2
    isize = max(isize, 4)  # at least 4 bytes (=float32), tends to be faster than float16
    dt_blend = np.dtype("f%d" % (isize,))

    if alpha.dtype != dt_blend:
        alpha = alpha.astype(dt_blend)
    if image_fg.dtype != dt_blend:
        image_fg = image_fg.astype(dt_blend)
    if image_bg.dtype != dt_blend:
        image_bg = image_bg.astype(dt_blend)

    # the following is equivalent to
    #     image_blend = alpha * image_fg + (1 - alpha) * image_bg
    # but supposedly faster
    image_blend = image_bg + alpha * (image_fg - image_bg)

    if input_was_bool:
        image_blend = image_blend > 0.5
    else:
        # skip clip, because alpha is expected to be in range [0.0, 1.0] and both images must have same dtype
        # dont skip round, because otherwise it is very unlikely to hit the image's max possible value
        image_blend = iadt.restore_dtypes_(image_blend, dt_images, clip=False, round=True)

    return image_blend


class Alpha(meta.Augmenter):  # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Augmenter to blend two image sources using an alpha/transparency value.

    The two image sources can be imagined as branches.
    If a source is not given, it is automatically the same as the input.
    Let A be the first branch and B be the second branch.
    Then the result images are defined as ``factor * A + (1-factor) * B``,
    where ``factor`` is an overlay factor.

    For keypoint augmentation this augmenter will pick the keypoints either
    from the first or the second branch. The first one is picked if
    ``factor >= 0.5`` is true (per image). It is recommended to *not* use
    augmenters that change keypoint positions with this class.

    dtype support::

        See :func:`imgaug.augmenters.blend.blend_alpha`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Weighting of the results of the first branch. Values close to 0 mean
        that the results from the second branch (see parameter `second`)
        make up most of the final image.

            * If float, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value from range ``a <= x <= b`` will
              be sampled per image.
            * If a list, then a random value will be picked from that list per
              image.
            * If StochasticParameter, then that parameter will be used to
              sample a value per image.

    first : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the first of the two branches.

            * If None, then the input images will be reused as the output
              of the first branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    second : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the second of the two branches.

            * If None, then the input images will be reused as the output
              of the second branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Alpha(0.5, iaa.Grayscale(1.0))

    Converts each image to grayscale and alpha-blends it by 50 percent with the
    original image, thereby removing about 50 percent of all color. This
    is equivalent to ``iaa.Grayscale(0.5)``.

    >>> aug = iaa.Alpha((0.0, 1.0), iaa.Grayscale(1.0))

    Converts each image to grayscale and alpha-blends it by a random percentage
    (sampled per image) with the original image, thereby removing a random
    percentage of all colors. This is equivalent to ``iaa.Grayscale((0.0, 1.0))``.

    >>> aug = iaa.Alpha((0.0, 1.0), iaa.Affine(rotate=(-20, 20)), per_channel=0.5)

    Rotates each image by a random degree from the range ``[-20, 20]``. Then
    alpha-blends that new image with the original one by a random factor from
    the range ``[0.0, 1.0]``. In 50 percent of all cases, the blending happens
    channel-wise and the factor is sampled independently per channel. As a
    result, e.g. the red channel may look visible rotated (factor near 1.0),
    while the green and blue channels may not look rotated (factors near 0.0).
    NOTE: It is not recommended to use Alpha with augmenters that change the
    positions of pixels if you *also* want to augment keypoints, as it is
    unclear which of the two keypoint results (first or second branch) should
    be used as the final result.

    >>> aug = iaa.Alpha((0.0, 1.0), first=iaa.Add(10), second=iaa.Multiply(0.8))

    (A) Adds 10 to each image and (B) multiplies each image by 0.8. Then per
    image a blending factor is sampled from the range ``[0.0, 1.0]``. If it is
    close to 1.0, the results from (A) are mostly used, otherwise the ones
    from (B). This is equivalent to
    ``iaa.Sequential([iaa.Multiply(0.8), iaa.Alpha((0.0, 1.0), iaa.Add(10))])``.

    >>> aug = iaa.Alpha(iap.Choice([0.25, 0.75]), iaa.MedianBlur((3, 7)))

    Applies a random median blur to each image and alpha-blends the result with
    the original image by either 25 or 75 percent strength.

    """

    # TODO rename first/second to foreground/background?
    def __init__(self, factor=0, first=None, second=None, per_channel=False,
                 name=None, deterministic=False, random_state=None):
        super(Alpha, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.factor = iap.handle_continuous_param(factor, "factor", value_range=(0, 1.0), tuple_to_uniform=True,
                                                  list_to_choice=True)

        ia.do_assert(first is not None or second is not None,
                     "Expected 'first' and/or 'second' to not be None (i.e. at least one Augmenter), "
                     + "but got two None values.")
        self.first = meta.handle_children_list(first, self.name, "first", default=None)
        self.second = meta.handle_children_list(second, self.name, "second", default=None)

        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

        self.epsilon = 1e-2

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        nb_channels = meta.estimate_max_number_of_channels(images)
        rss = ia.derive_random_states(random_state, 2)
        per_channel = self.per_channel.draw_samples(nb_images, random_state=rss[0])
        alphas = self.factor.draw_samples((nb_images, nb_channels), random_state=rss[1])

        if hooks is None or hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
            if self.first is None:
                images_first = images
            else:
                images_first = self.first.augment_images(
                    images=meta.copy_arrays(images),
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is None:
                images_second = images
            else:
                images_second = self.second.augment_images(
                    images=meta.copy_arrays(images),
                    parents=parents + [self],
                    hooks=hooks
                )
        else:
            images_first = images
            images_second = images

        for i, (image_first, image_second) in enumerate(zip(images_first, images_second)):
            if per_channel[i] > 0.5:
                nb_channels_i = image_first.shape[2]
                alphas_i = alphas[i, 0:nb_channels_i]
            else:
                alphas_i = alphas[i, 0]

            result[i] = blend_alpha(image_first, image_second, alphas_i, eps=self.epsilon)
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        result = heatmaps
        nb_heatmaps = len(heatmaps)
        if nb_heatmaps == 0:
            return heatmaps

        nb_channels = meta.estimate_max_number_of_channels(heatmaps)
        rss = ia.derive_random_states(random_state, 2)
        per_channel = self.per_channel.draw_samples(nb_heatmaps, random_state=rss[0])
        alphas = self.factor.draw_samples((nb_heatmaps, nb_channels), random_state=rss[1])

        if hooks is None or hooks.is_propagating(heatmaps, augmenter=self, parents=parents, default=True):
            if self.first is None:
                heatmaps_first = heatmaps
            else:
                heatmaps_first = self.first.augment_heatmaps(
                    [heatmaps_i.deepcopy() for heatmaps_i in heatmaps],
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is None:
                heatmaps_second = heatmaps
            else:
                heatmaps_second = self.second.augment_heatmaps(
                    [heatmaps_i.deepcopy() for heatmaps_i in heatmaps],
                    parents=parents + [self],
                    hooks=hooks
                )
        else:
            heatmaps_first = heatmaps
            heatmaps_second = heatmaps

        for i, (heatmaps_first_i, heatmaps_second_i) in enumerate(zip(heatmaps_first, heatmaps_second)):
            # sample alphas channelwise if necessary and try to use the image's channel number
            # values properly synchronized with the image augmentation
            # per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel[i] > 0.5:
                nb_channels_i = heatmaps[i].shape[2] if len(heatmaps[i].shape) >= 3 else 1
                alpha = np.average(alphas[i, 0:nb_channels_i])
            else:
                alpha = alphas[i, 0]
            ia.do_assert(0 <= alpha <= 1.0)

            if alpha >= 0.5:
                result[i].arr_0to1 = heatmaps_first_i.arr_0to1
            else:
                result[i].arr_0to1 = heatmaps_second_i.arr_0to1

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        if nb_images == 0:
            return keypoints_on_images

        nb_channels = meta.estimate_max_number_of_channels(keypoints_on_images)
        rss = ia.derive_random_states(random_state, 2)
        per_channel = self.per_channel.draw_samples(nb_images, random_state=rss[0])
        alphas = self.factor.draw_samples((nb_images, nb_channels), random_state=rss[1])

        result = keypoints_on_images

        if hooks is None or hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents, default=True):
            if self.first is None:
                kps_ois_first = keypoints_on_images
            else:
                kps_ois_first = self.first.augment_keypoints(
                    keypoints_on_images=[kpsoi_i.deepcopy() for kpsoi_i in keypoints_on_images],
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is None:
                kps_ois_second = keypoints_on_images
            else:
                kps_ois_second = self.second.augment_keypoints(
                    keypoints_on_images=[kpsoi_i.deepcopy() for kpsoi_i in keypoints_on_images],
                    parents=parents + [self],
                    hooks=hooks
                )
        else:
            kps_ois_first = keypoints_on_images
            kps_ois_second = keypoints_on_images

        for i, (kps_oi_first, kps_oi_second) in enumerate(zip(kps_ois_first, kps_ois_second)):
            # keypoint augmentation also works channel-wise, even though
            # keypoints do not have channels, in order to keep the random
            # values properly synchronized with the image augmentation
            # per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel[i] > 0.5:
                nb_channels_i = keypoints_on_images[i].shape[2] if len(keypoints_on_images[i].shape) >= 3 else 1
                alpha = np.average(alphas[i, 0:nb_channels_i])
            else:
                alpha = alphas[i, 0]
            ia.do_assert(0 <= alpha <= 1.0)

            # We cant choose "just a bit" of one keypoint augmentation result
            # without messing up the positions (interpolation doesn't make much
            # sense here),
            # so if the alpha is >= 0.5 (branch A is more visible than
            # branch B), the result of branch A, otherwise branch B.
            if alpha >= 0.5:
                result[i] = kps_oi_first
            else:
                result[i] = kps_oi_second

        return result

    def _to_deterministic(self):
        aug = self.copy()
        aug.first = aug.first.to_deterministic() if aug.first is not None else None
        aug.second = aug.second.to_deterministic() if aug.second is not None else None
        aug.deterministic = True
        aug.random_state = ia.derive_random_state(self.random_state)
        return aug

    def get_parameters(self):
        return [self.factor, self.per_channel]

    def get_children_lists(self):
        return [lst for lst in [self.first, self.second] if lst is not None]

    def __str__(self):
        return "%s(factor=%s, per_channel=%s, name=%s, first=%s, second=%s, deterministic=%s)" % (
            self.__class__.__name__, self.factor, self.per_channel, self.name,
            self.first, self.second, self.deterministic)


# TODO merge this with Alpha
class AlphaElementwise(Alpha):  # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Augmenter to blend two image sources pixelwise alpha/transparency values.

    This is the same as ``Alpha``, except that the transparency factor is
    sampled per pixel instead of once per image (or a few times per image, if
    per_channel is True).

    See ``Alpha`` for more description.

    dtype support::

        See :func:`imgaug.augmenters.blend.blend_alpha`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Weighting of the results of the first branch. Values close to 0 mean
        that the results from the second branch (see parameter `second`)
        make up most of the final image.

            * If float, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value from range ``a <= x <= b`` will
              be sampled per image.
            * If a list, then a random value will be picked from that list per
              image.
            * If StochasticParameter, then that parameter will be used to
              sample a value per image.

    first : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the first of the two branches.

            * If None, then the input images will be reused as the output
              of the first branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    second : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the second of the two branches.

            * If None, then the input images will be reused as the output
              of the second branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.AlphaElementwise(0.5, iaa.Grayscale(1.0))

    Converts each image to grayscale and overlays it by 50 percent with the
    original image, thereby removing about 50 percent of all color. This
    is equivalent to ``iaa.Grayscale(0.5)``. This is also equivalent to
    ``iaa.Alpha(0.5, iaa.Grayscale(1.0))``, as the transparency factor is the
    same for all pixels.

    >>> aug = iaa.AlphaElementwise((0, 1.0), iaa.Grayscale(1.0))

    Converts each image to grayscale and alpha-blends it by a random percentage
    (sampled per pixel) with the original image, thereby removing a random
    percentage of all colors per pixel.

    >>> aug = iaa.AlphaElementwise((0.0, 1.0), iaa.Affine(rotate=(-20, 20)), per_channel=0.5)

    Rotates each image by a random degree from the range ``[-20, 20]``. Then
    alpha-blends that new image with the original one by a random factor from
    the range ``[0.0, 1.0]``, sampled per pixel. In 50 percent of all cases, the
    blending happens channel-wise and the factor is sampled independently per
    channel. As a result, e.g. the red channel may look visible rotated (factor
    near 1.0), while the green and blue channels may not look rotated (factors
    near 0.0). NOTE: It is not recommended to use Alpha with augmenters that
    change the positions of pixels if you *also* want to augment keypoints, as
    it is unclear which of the two keypoint results (first or second branch)
    should be used as the final result.

    >>> aug = iaa.AlphaElementwise((0.0, 1.0), first=iaa.Add(10), second=iaa.Multiply(0.8))

    (A) Adds 10 to each image and (B) multiplies each image by 0.8. Then per
    pixel a blending factor is sampled from the range ``[0.0, 1.0]``. If it is
    close to 1.0, the results from (A) are mostly used, otherwise the ones
    from (B).

    >>> aug = iaa.AlphaElementwise(iap.Choice([0.25, 0.75]), iaa.MedianBlur((3, 7)))

    Applies a random median blur to each image and alpha-blends the result with
    the original image by either 25 or 75 percent strength (sampled per pixel).

    """

    def __init__(self, factor=0, first=None, second=None, per_channel=False,
                 name=None, deterministic=False, random_state=None):
        super(AlphaElementwise, self).__init__(
            factor=factor,
            first=first,
            second=second,
            per_channel=per_channel,
            name=name,
            deterministic=deterministic,
            random_state=random_state
        )

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))

        if hooks is None or hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
            if self.first is None:
                images_first = images
            else:
                images_first = self.first.augment_images(
                    images=meta.copy_arrays(images),
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is None:
                images_second = images
            else:
                images_second = self.second.augment_images(
                    images=meta.copy_arrays(images),
                    parents=parents + [self],
                    hooks=hooks
                )
        else:
            images_first = images
            images_second = images

        # TODO simplify this loop and the ones for heatmaps, keypoints; similar to Alpha
        for i in sm.xrange(nb_images):
            image = images[i]
            h, w, nb_channels = image.shape[0:3]
            image_first = images_first[i]
            image_second = images_second[i]
            per_channel = self.per_channel.draw_sample(random_state=ia.new_random_state(seeds[i]))
            if per_channel > 0.5:
                alphas = []
                for c in sm.xrange(nb_channels):
                    samples_c = self.factor.draw_samples((h, w), random_state=ia.new_random_state(seeds[i]+1+c))
                    ia.do_assert(0 <= samples_c.item(0) <= 1.0) # validate only first value
                    alphas.append(samples_c)
                alphas = np.float64(alphas).transpose((1, 2, 0))
            else:
                alphas = self.factor.draw_samples((h, w), random_state=ia.new_random_state(seeds[i]))
                ia.do_assert(0.0 <= alphas.item(0) <= 1.0)
            result[i] = blend_alpha(image_first, image_second, alphas, eps=self.epsilon)
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        def _sample_factor_mask(h_images, w_images, h_heatmaps, w_heatmaps, seed):
            samples_c = self.factor.draw_samples((h_images, w_images), random_state=ia.new_random_state(seed))
            ia.do_assert(0 <= samples_c.item(0) <= 1.0)  # validate only first value

            if (h_images, w_images) != (h_heatmaps, w_heatmaps):
                samples_c = np.clip(samples_c * 255, 0, 255).astype(np.uint8)
                samples_c = ia.imresize_single_image(samples_c, (h_heatmaps, w_heatmaps), interpolation="cubic")
                samples_c = samples_c.astype(np.float32) / 255.0

            return samples_c

        result = heatmaps
        nb_heatmaps = len(heatmaps)
        seeds = random_state.randint(0, 10**6, (nb_heatmaps,))

        if hooks is None or hooks.is_propagating(heatmaps, augmenter=self, parents=parents, default=True):
            if self.first is None:
                heatmaps_first = heatmaps
            else:
                heatmaps_first = self.first.augment_heatmaps(
                    [heatmaps_i.deepcopy() for heatmaps_i in heatmaps],
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is None:
                heatmaps_second = heatmaps
            else:
                heatmaps_second = self.second.augment_heatmaps(
                    [heatmaps_i.deepcopy() for heatmaps_i in heatmaps],
                    parents=parents + [self],
                    hooks=hooks
                )
        else:
            heatmaps_first = heatmaps
            heatmaps_second = heatmaps

        for i in sm.xrange(nb_heatmaps):
            heatmaps_i = heatmaps[i]
            h_img, w_img = heatmaps_i.shape[0:2]
            h_heatmaps, w_heatmaps = heatmaps_i.arr_0to1.shape[0:2]
            nb_channels_img = heatmaps_i.shape[2] if len(heatmaps_i.shape) >= 3 else 1
            nb_channels_heatmaps = heatmaps_i.arr_0to1.shape[2]
            heatmaps_first_i = heatmaps_first[i]
            heatmaps_second_i = heatmaps_second[i]
            per_channel = self.per_channel.draw_sample(random_state=ia.new_random_state(seeds[i]))
            if per_channel > 0.5:
                samples = []
                for c in sm.xrange(nb_channels_img):
                    # We sample here at the same size as the original image, as some effects
                    # might not scale with image size. We sampled mask is then downscaled to the
                    # heatmap size.
                    samples_c = _sample_factor_mask(h_img, w_img, h_heatmaps, w_heatmaps, seeds[i]+1+c)
                    samples.append(samples_c[..., np.newaxis])
                samples = np.concatenate(samples, axis=2)
                samples_avg = np.average(samples, axis=2)
                samples_tiled = np.tile(samples_avg[..., np.newaxis], (1, 1, nb_channels_heatmaps))
            else:
                samples = _sample_factor_mask(h_img, w_img, h_heatmaps, w_heatmaps, seeds[i])
                samples_tiled = np.tile(samples[..., np.newaxis], (1, 1, nb_channels_heatmaps))

            mask = samples_tiled >= 0.5
            heatmaps_arr_aug = mask * heatmaps_first_i.arr_0to1 + (~mask) * heatmaps_second_i.arr_0to1

            result[i].arr_0to1 = heatmaps_arr_aug

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = keypoints_on_images
        nb_images = len(keypoints_on_images)
        seeds = random_state.randint(0, 10**6, (nb_images,))

        if hooks is None or hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents, default=True):
            if self.first is None:
                kps_ois_first = keypoints_on_images
            else:
                kps_ois_first = self.first.augment_keypoints(
                    keypoints_on_images=[kpsoi_i.deepcopy() for kpsoi_i in keypoints_on_images],
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is None:
                kps_ois_second = keypoints_on_images
            else:
                kps_ois_second = self.second.augment_keypoints(
                    keypoints_on_images=[kpsoi_i.deepcopy() for kpsoi_i in keypoints_on_images],
                    parents=parents + [self],
                    hooks=hooks
                )
        else:
            kps_ois_first = keypoints_on_images
            kps_ois_second = keypoints_on_images

        # FIXME this is essentially the same behaviour as Alpha, requires inclusion of (x, y) coordinates to estimate
        # new keypoint coordinates
        for i in sm.xrange(nb_images):
            kps_oi_first = kps_ois_first[i]
            kps_oi_second = kps_ois_second[i]
            ia.do_assert(
                len(kps_oi_first.shape) == 3,
                ("Keypoint augmentation in AlphaElementwise requires KeypointsOnImage.shape to have channel "
                 + "information (i.e. tuple with 3 entries), which you did not provide (input shape: %s). The"
                   "channels must match the corresponding image channels.") % (kps_oi_first.shape,)
            )
            h, w, nb_channels = kps_oi_first.shape[0:3]

            # keypoint augmentation also works channel-wise, even though
            # keypoints do not have channels, in order to keep the random
            # values properly synchronized with the image augmentation
            per_channel = self.per_channel.draw_sample(random_state=ia.new_random_state(seeds[i]))
            if per_channel > 0.5:
                samples = np.zeros((h, w, nb_channels), dtype=np.float32)
                for c in sm.xrange(nb_channels):
                    samples_c = self.factor.draw_samples((h, w), random_state=ia.new_random_state(seeds[i]+1+c))
                    samples[:, :, c] = samples_c
            else:
                samples = self.factor.draw_samples((h, w), random_state=ia.new_random_state(seeds[i]))
            ia.do_assert(0.0 <= samples.item(0) <= 1.0)
            sample = np.average(samples)

            # We cant choose "just a bit" of one keypoint augmentation result
            # without messing up the positions (interpolation doesn't make much
            # sense here),
            # so if the alpha is >= 0.5 (branch A is more visible than
            # branch B), the result of branch A, otherwise branch B.
            if sample >= 0.5:
                result[i] = kps_oi_first
            else:
                result[i] = kps_oi_second

        return result


def SimplexNoiseAlpha(first=None, second=None, per_channel=False, size_px_max=(2, 16), upscale_method=None,
                      iterations=(1, 3), aggregation_method="max", sigmoid=True, sigmoid_thresh=None,
                      name=None, deterministic=False, random_state=None):
    """
    Augmenter to alpha-blend two image sources using simplex noise alpha masks.

    The alpha masks are sampled using a simplex noise method, roughly creating
    connected blobs of 1s surrounded by 0s. If nearest neighbour upsampling
    is used, these blobs can be rectangular with sharp edges.

    dtype support::

        See ``imgaug.augmenters.blend.AlphaElementwise``.

    Parameters
    ----------
    first : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the first of the two branches.

            * If None, then the input images will be reused as the output
              of the first branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    second : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the second of the two branches.

            * If None, then the input images will be reused as the output
              of the second branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    size_px_max : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        The simplex noise is always generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).

            * If int, then that number will be used as the size for all
              iterations.
            * If tuple of two ints ``(a, b)``, then a value will be sampled
              per iteration from the discrete range ``[a..b]``.
            * If a list of ints, then a value will be picked per iteration at
              random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.

            * If None, then either ``nearest`` or ``linear`` or ``cubic`` is picked.
              Most weight is put on linear, followed by cubic.
            * If ia.ALL, then either ``nearest`` or ``linear`` or ``area`` or ``cubic``
              is picked per iteration (all same probability).
            * If string, then that value will be used as the method (must be
              'nearest' or ``linear`` or ``area`` or ``cubic``).
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If StochasticParameter, then a random value will be sampled
              from that parameter per iteration.

    iterations : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        How often to repeat the simplex noise generation process per image.

            * If int, then that number will be used as the iterations for all
              images.
            * If tuple of two ints ``(a, b)``, then a value will be sampled
              per image from the discrete range ``[a..b]``.
            * If a list of ints, then a value will be picked per image at
              random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per image.

    aggregation_method : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        The noise maps (from each iteration) are combined to one noise map
        using an aggregation process. This parameter defines the method used
        for that process. Valid methods are ``min``, ``max`` or ``avg``,
        where ``min`` combines the noise maps by taking the (elementwise) minimum
        over all iteration's results, ``max`` the (elementwise) maximum and
        ``avg`` the (elementwise) average.

            * If imgaug.ALL, then a random value will be picked per image from the
              valid ones.
            * If a string, then that value will always be used as the method.
            * If a list of string, then a random value will be picked from
              that list per image.
            * If a StochasticParameter, then a random value will be sampled
              from that paramter per image.

    sigmoid : bool or number, optional
        Whether to apply a sigmoid function to the final noise maps, resulting
        in maps that have more extreme values (close to 0.0 or 1.0).

            * If bool, then a sigmoid will always (True) or never (False) be
              applied.
            * If a number ``p`` with ``0<=p<=1``, then a sigmoid will be applied to
              ``p`` percent of all final noise maps.

    sigmoid_thresh : None or number or tuple of number or imgaug.parameters.StochasticParameter, optional
        Threshold of the sigmoid, when applied. Thresholds above zero
        (e.g. 5.0) will move the saddle point towards the right, leading to
        more values close to 0.0.

            * If None, then ``Normal(0, 5.0)`` will be used.
            * If number, then that threshold will be used for all images.
            * If tuple of two numbers ``(a, b)``, then a random value will
              be sampled per image from the range ``[a, b]``.
            * If StochasticParameter, then a random value will be sampled from
              that parameter per image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.SimplexNoiseAlpha(iaa.EdgeDetect(1.0))

    Detects per image all edges, marks them in a black and white image and
    then alpha-blends the result with the original image using simplex noise
    masks.

    >>> aug = iaa.SimplexNoiseAlpha(iaa.EdgeDetect(1.0), upscale_method="linear")

    Same as the first example, but uses only (smooth) linear upscaling to
    scale the simplex noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used, which would result in rectangles with hard
    edges.

    >>> aug = iaa.SimplexNoiseAlpha(iaa.EdgeDetect(1.0), sigmoid_thresh=iap.Normal(10.0, 5.0))

    Same as the first example, but uses a threshold for the sigmoid function
    that is further to the right. This is more conservative, i.e. the generated
    noise masks will be mostly black (values around 0.0), which means that
    most of the original images (parameter/branch `second`) will be kept,
    rather than using the results of the augmentation (parameter/branch
    `first`).

    """
    upscale_method_default = iap.Choice(["nearest", "linear", "cubic"], p=[0.05, 0.6, 0.35])
    sigmoid_thresh_default = iap.Normal(0.0, 5.0)

    noise = iap.SimplexNoise(
        size_px_max=size_px_max,
        upscale_method=upscale_method if upscale_method is not None else upscale_method_default
    )

    if iterations != 1:
        noise = iap.IterativeNoiseAggregator(
            noise,
            iterations=iterations,
            aggregation_method=aggregation_method
        )

    if sigmoid is False or (ia.is_single_number(sigmoid) and sigmoid <= 0.01):
        noise = iap.Sigmoid.create_for_noise(
            noise,
            threshold=sigmoid_thresh if sigmoid_thresh is not None else sigmoid_thresh_default,
            activated=sigmoid
        )

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return AlphaElementwise(
        factor=noise, first=first, second=second, per_channel=per_channel,
        name=name, deterministic=deterministic, random_state=random_state
    )


def FrequencyNoiseAlpha(exponent=(-4, 4), first=None, second=None, per_channel=False,
                        size_px_max=(4, 16), upscale_method=None,
                        iterations=(1, 3), aggregation_method=["avg", "max"],
                        sigmoid=0.5, sigmoid_thresh=None,
                        name=None, deterministic=False, random_state=None):
    """
    Augmenter to alpha-blend two image sources using frequency noise masks.

    The alpha masks are sampled using frequency noise of varying scales,
    which can sometimes create large connected blobs of 1s surrounded by 0s
    and other times results in smaller patterns. If nearest neighbour
    upsampling is used, these blobs can be rectangular with sharp edges.

    dtype support::

        See ``imgaug.augmenters.blend.AlphaElementwise``.

    Parameters
    ----------
    exponent : number or tuple of number of list of number or imgaug.parameters.StochasticParameter, optional
        Exponent to use when scaling in the frequency domain.
        Sane values are in the range -4 (large blobs) to 4 (small patterns).
        To generate cloud-like structures, use roughly -2.

            * If number, then that number will be used as the exponent for all
              iterations.
            * If tuple of two numbers ``(a, b)``, then a value will be sampled
              per iteration from the range ``[a, b]``.
            * If a list of numbers, then a value will be picked per iteration
              at random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per iteration.

    first : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the first of the two branches.

            * If None, then the input images will be reused as the output
              of the first branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    second : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the second of the two branches.

            * If None, then the input images will be reused as the output
              of the second branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    size_px_max : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        The noise is generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).

            * If int, then that number will be used as the size for all
              iterations.
            * If tuple of two ints ``(a, b)``, then a value will be sampled
              per iteration from the discrete range ``[a..b]``.
            * If a list of ints, then a value will be picked per iteration at
              random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.

            * If None, then either ``nearest`` or ``linear`` or ``cubic`` is picked.
              Most weight is put on linear, followed by cubic.
            * If imgaug.ALL, then either ``nearest`` or ``linear`` or ``area`` or ``cubic``
              is picked per iteration (all same probability).
            * If string, then that value will be used as the method (must be
              ``nearest`` or ``linear`` or ``area`` or ``cubic``).
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If StochasticParameter, then a random value will be sampled
              from that parameter per iteration.

    iterations : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        How often to repeat the simplex noise generation process per
        image.

            * If int, then that number will be used as the iterations for all
              images.
            * If tuple of two ints ``(a, b)``, then a value will be sampled
              per image from the discrete range ``[a..b]``.
            * If a list of ints, then a value will be picked per image at
              random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per image.

    aggregation_method : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        The noise maps (from each iteration) are combined to one noise map
        using an aggregation process. This parameter defines the method used
        for that process. Valid methods are ``min``, ``max`` or ``avg``,
        where 'min' combines the noise maps by taking the (elementwise) minimum
        over all iteration's results, ``max`` the (elementwise) maximum and
        ``avg`` the (elementwise) average.

            * If imgaug.ALL, then a random value will be picked per image from the
              valid ones.
            * If a string, then that value will always be used as the method.
            * If a list of string, then a random value will be picked from
              that list per image.
            * If a StochasticParameter, then a random value will be sampled
              from that parameter per image.

    sigmoid : bool or number, optional
        Whether to apply a sigmoid function to the final noise maps, resulting
        in maps that have more extreme values (close to 0.0 or 1.0).

            * If bool, then a sigmoid will always (True) or never (False) be
              applied.
            * If a number ``p`` with ``0<=p<=1``, then a sigmoid will be applied to
              ``p`` percent of all final noise maps.

    sigmoid_thresh : None or number or tuple of number or imgaug.parameters.StochasticParameter, optional
        Threshold of the sigmoid, when applied. Thresholds above zero
        (e.g. 5.0) will move the saddle point towards the right, leading to
        more values close to 0.0.

            * If None, then ``Normal(0, 5.0)`` will be used.
            * If number, then that threshold will be used for all images.
            * If tuple of two numbers ``(a, b)``, then a random value will
              be sampled per image from the range ``[a, b]``.
            * If StochasticParameter, then a random value will be sampled from
              that parameter per image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(1.0))

    Detects per image all edges, marks them in a black and white image and
    then alpha-blends the result with the original image using frequency noise
    masks.

    >>> aug = iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(1.0), upscale_method="linear")

    Same as the first example, but uses only (smooth) linear upscaling to
    scale the frequency noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used, which would result in rectangles with hard
    edges.

    >>> aug = iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(1.0), upscale_method="linear", exponent=-2, sigmoid=False)

    Same as the previous example, but also limits the exponent to -2 and
    deactivates the sigmoid, resulting in cloud-like patterns without sharp
    edges.

    >>> aug = iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(1.0), sigmoid_thresh=iap.Normal(10.0, 5.0))

    Same as the first example, but uses a threshold for the sigmoid function
    that is further to the right. This is more conservative, i.e. the generated
    noise masks will be mostly black (values around 0.0), which means that
    most of the original images (parameter/branch `second`) will be kept,
    rather than using the results of the augmentation (parameter/branch
    `first`).

    """
    # pylint: disable=dangerous-default-value
    upscale_method_default = iap.Choice(["nearest", "linear", "cubic"], p=[0.05, 0.6, 0.35])
    sigmoid_thresh_default = iap.Normal(0.0, 5.0)

    noise = iap.FrequencyNoise(
        exponent=exponent,
        size_px_max=size_px_max,
        upscale_method=upscale_method if upscale_method is not None else upscale_method_default
    )

    if iterations != 1:
        noise = iap.IterativeNoiseAggregator(
            noise,
            iterations=iterations,
            aggregation_method=aggregation_method
        )

    if sigmoid is False or (ia.is_single_number(sigmoid) and sigmoid <= 0.01):
        noise = iap.Sigmoid.create_for_noise(
            noise,
            threshold=sigmoid_thresh if sigmoid_thresh is not None else sigmoid_thresh_default,
            activated=sigmoid
        )

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return AlphaElementwise(
        factor=noise, first=first, second=second, per_channel=per_channel,
        name=name, deterministic=deterministic, random_state=random_state
    )
