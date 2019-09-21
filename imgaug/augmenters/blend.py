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
import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt
from ..augmentables import utils as augm_utils


def blend_alpha(image_fg, image_bg, alpha, eps=1e-2):
    """
    Blend two images using an alpha blending.

    In alpha blending, the two images are naively mixed using a multiplier.
    Let ``A`` be the foreground image and ``B`` the background image and
    ``a`` is the alpha value. Each pixel intensity is then computed as
    ``a * A_ij + (1-a) * B_ij``.

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

        - (1) Tests show that these dtypes work, but a conversion to
              ``float128`` happens, which only has 96 bits of size instead of
              true 128 bits and hence not twice as much resolution. It is
              possible that these dtypes result in inaccuracies, though the
              tests did not indicate that.
        - (2) Not available due to the input dtype having to be increased to
              an equivalent float dtype with two times the input resolution.
        - (3) Mapped internally to ``float16``.

    Parameters
    ----------
    image_fg : (H,W,[C]) ndarray
        Foreground image. Shape and dtype kind must match the one of the
        background image.

    image_bg : (H,W,[C]) ndarray
        Background image. Shape and dtype kind must match the one of the
        foreground image.

    alpha : number or iterable of number or ndarray
        The blending factor, between ``0.0`` and ``1.0``. Can be interpreted
        as the opacity of the foreground image. Values around ``1.0`` result
        in only the foreground image being visible. Values around ``0.0``
        result in only the background image being visible. Multiple alphas
        may be provided. In these cases, there must be exactly one alpha per
        channel in the foreground/background image. Alternatively, for
        ``(H,W,C)`` images, either one ``(H,W)`` array or an ``(H,W,C)``
        array of alphas may be provided, denoting the elementwise alpha value.

    eps : number, optional
        Controls when an alpha is to be interpreted as exactly ``1.0`` or
        exactly ``0.0``, resulting in only the foreground/background being
        visible and skipping the actual computation.

    Returns
    -------
    image_blend : (H,W,C) ndarray
        Blend of foreground and background image.

    """
    assert image_fg.shape == image_bg.shape, (
        "Expected oreground and background images to have the same shape. "
        "Got %s and %s." % (image_fg.shape, image_bg.shape))
    assert image_fg.dtype.kind == image_bg.dtype.kind, (
        "Expected oreground and background images to have the same dtype kind. "
        "Got %s and %s." % (image_fg.dtype.kind, image_bg.dtype.kind))
    # TODO switch to gate_dtypes()
    assert image_fg.dtype.name not in ["float128"], (
        "Foreground image was float128, but blend_alpha() cannot handle that "
        "dtype.")
    assert image_bg.dtype.name not in ["float128"], (
        "Background image was float128, but blend_alpha() cannot handle that "
        "dtype.")

    input_was_2d = (image_fg.ndim == 2)
    if input_was_2d:
        image_fg = image_fg[..., np.newaxis]
        image_bg = image_bg[..., np.newaxis]

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
            assert alpha.shape == image_fg.shape[0:2], (
                "'alpha' given as an array must match the height and width "
                "of the foreground and background image. Got shape %s vs "
                "foreground/background shape %s." % (
                    alpha.shape, image_fg.shape))
            alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        elif alpha.ndim == 3:
            assert (
                alpha.shape == image_fg.shape
                or alpha.shape == image_fg.shape[0:2] + (1,)), (
                "'alpha' given as an array must match the height and width "
                "of the foreground and background image. Got shape %s vs "
                "foreground/background shape %s." % (
                    alpha.shape, image_fg.shape))
        else:
            alpha = alpha.reshape((1, 1, -1))
        if alpha.shape[2] != image_fg.shape[2]:
            alpha = np.tile(alpha, (1, 1, image_fg.shape[2]))

    if not input_was_bool:
        if np.all(alpha >= 1.0 - eps):
            return np.copy(image_fg)
        elif np.all(alpha <= eps):
            return np.copy(image_bg)

    # for efficiency reaons, only test one value of alpha here, even if alpha
    # is much larger
    assert 0 <= alpha.item(0) <= 1.0, (
        "Expected 'alpha' value(s) to be in the interval [0.0, 1.0]. "
        "Got min %.4f and max %.4f." % (np.min(alpha), np.max(alpha)))

    dt_images = iadt.get_minimal_dtype([image_fg, image_bg])

    # doing the below itemsize increase only for non-float images led to
    # inaccuracies for large float values
    # we also use a minimum of 4 bytes (=float32), as float32 tends to be
    # faster than float16
    isize = dt_images.itemsize * 2
    isize = max(isize, 4)
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
        # skip clip, because alpha is expected to be in range [0.0, 1.0] and
        # both images must have same dtype dont skip round, because otherwise
        # it is very unlikely to hit the image's max possible value
        image_blend = iadt.restore_dtypes_(
            image_blend, dt_images, clip=False, round=True)

    if input_was_2d:
        return image_blend[:, :, 0]
    return image_blend


class Alpha(meta.Augmenter):
    """
    Alpha-blend two image sources using an alpha/opacity value.

    The two image sources can be imagined as branches.
    If a source is not given, it is automatically the same as the input.
    Let A be the first branch and B be the second branch.
    Then the result images are defined as ``factor * A + (1-factor) * B``,
    where ``factor`` is an overlay factor.

    .. note::

        It is not recommended to use ``Alpha`` with augmenters
        that change the geometry of images (e.g. horizontal flips, affine
        transformations) if you *also* want to augment coordinates (e.g.
        keypoints, polygons, ...), as it is unclear which of the two
        coordinate results (first or second branch) should be used as the
        coordinates after augmentation.

        Currently, if ``factor >= 0.5`` (per image), the results of the first
        branch are used as the new coordinates, otherwise the results of the
        second branch.

    dtype support::

        See :func:`imgaug.augmenters.blend.blend_alpha`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Weighting of the results of the first branch. Values close to ``0``
        mean that the results from the second branch (see parameter `second`)
        make up most of the final image.

            * If float, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be picked from that list per
              image.
            * If ``StochasticParameter``, then that parameter will be used to
              sample a value per image.

    first : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the first of the two branches.

            * If ``None``, then the input images will be reused as the output
              of the first branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    second : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the second of the two branches.

            * If ``None``, then the input images will be reused as the output
              of the second branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Alpha(0.5, iaa.Grayscale(1.0))

    Convert each image to pure grayscale and alpha-blend the result with the
    original image using an alpha of ``50%``, thereby removing about ``50%`` of
    all color. This is equivalent to ``iaa.Grayscale(0.5)``.

    >>> aug = iaa.Alpha((0.0, 1.0), iaa.Grayscale(1.0))

    Same as in the previous example, but the alpha factor is sampled uniformly
    from the interval ``[0.0, 1.0]`` once per image, thereby removing a random
    fraction of all colors. This is equivalent to
    ``iaa.Grayscale((0.0, 1.0))``.

    >>> aug = iaa.Alpha(
    >>>     (0.0, 1.0),
    >>>     iaa.Affine(rotate=(-20, 20)),
    >>>     per_channel=0.5)

    First, rotate each image by a random degree sampled uniformly from the
    interval ``[-20, 20]``. Then, alpha-blend that new image with the original
    one using a random factor sampled uniformly from the interval
    ``[0.0, 1.0]``. For ``50%`` of all images, the blending happens
    channel-wise and the factor is sampled independently per channel
    (``per_channel=0.5``). As a result, e.g. the red channel may look visibly
    rotated (factor near ``1.0``), while the green and blue channels may not
    look rotated (factors near ``0.0``).

    >>> aug = iaa.Alpha(
    >>>     (0.0, 1.0),
    >>>     first=iaa.Add(100),
    >>>     second=iaa.Multiply(0.2))

    Apply two branches of augmenters -- ``A`` and ``B`` -- *independently*
    to input images and alpha-blend the results of these branches using a
    factor ``f``. Branch ``A`` increases image pixel intensities by ``100``
    and ``B`` multiplies the pixel intensities by ``0.2``. ``f`` is sampled
    uniformly from the interval ``[0.0, 1.0]`` per image. The resulting images
    contain a bit of ``A`` and a bit of ``B``.

    >>> aug = iaa.Alpha([0.25, 0.75], iaa.MedianBlur(13))

    Apply median blur to each image and alpha-blend the result with the
    original image using an alpha factor of either exactly ``0.25`` or
    exactly ``0.75`` (sampled once per image).

    """

    # TODO rename first/second to foreground/background?
    def __init__(self, factor=0, first=None, second=None, per_channel=False,
                 name=None, deterministic=False, random_state=None):
        super(Alpha, self).__init__(name=name, deterministic=deterministic,
                                    random_state=random_state)

        self.factor = iap.handle_continuous_param(
            factor, "factor", value_range=(0, 1.0), tuple_to_uniform=True,
            list_to_choice=True)

        assert first is not None or second is not None, (
            "Expected 'first' and/or 'second' to not be None (i.e. at least "
            "one Augmenter), but got two None values.")
        self.first = meta.handle_children_list(first, self.name, "first",
                                               default=None)
        self.second = meta.handle_children_list(second, self.name, "second",
                                                default=None)

        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")

        self.epsilon = 1e-2

    def _augment_images(self, images, random_state, parents, hooks):
        outputs_first, outputs_second = self._generate_branch_outputs(
            images, "augment_images", hooks, parents)

        nb_images = len(images)
        nb_channels = meta.estimate_max_number_of_channels(images)
        rngs = random_state.duplicate(2)
        per_channel = self.per_channel.draw_samples(nb_images,
                                                    random_state=rngs[0])
        alphas = self.factor.draw_samples((nb_images, nb_channels),
                                          random_state=rngs[1])
        result = images
        gen = enumerate(zip(outputs_first, outputs_second))
        for i, (image_first, image_second) in gen:
            if per_channel[i] > 0.5:
                nb_channels_i = image_first.shape[2]
                alphas_i = alphas[i, 0:nb_channels_i]
            else:
                alphas_i = alphas[i, 0]

            result[i] = blend_alpha(image_first, image_second, alphas_i,
                                    eps=self.epsilon)
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # This currently uses either branch A's results or branch B's results.
        # TODO Make this flexible to allow gradual blending of heatmaps here
        #      as used for images. Heatmaps are probably the only augmentable
        #      where this makes sense.
        return self._augment_nonimages(
            heatmaps, random_state, parents, hooks,
            "augment_heatmaps")

    def _augment_segmentation_maps(self, segmaps, random_state, parents, hooks):
        return self._augment_nonimages(
            segmaps, random_state, parents, hooks,
            "augment_segmentation_maps")

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return self._augment_nonimages(
            keypoints_on_images, random_state, parents, hooks,
            "augment_keypoints")

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        return self._augment_nonimages(
            polygons_on_images, random_state, parents, hooks,
            "augment_polygons")

    def _augment_nonimages(self, augmentables, random_state, parents, hooks,
                           augfunc_name):
        outputs_first, outputs_second = self._generate_branch_outputs(
            augmentables, augfunc_name, hooks, parents)

        nb_images = len(augmentables)
        if nb_images == 0:
            return augmentables

        nb_channels = meta.estimate_max_number_of_channels(augmentables)
        rngs = random_state.duplicate(2)
        per_channel = self.per_channel.draw_samples(nb_images,
                                                    random_state=rngs[0])
        alphas = self.factor.draw_samples((nb_images, nb_channels),
                                          random_state=rngs[1])

        result = augmentables
        gen = enumerate(zip(outputs_first, outputs_second))
        for i, (outputs_first_i, outputs_second_i) in gen:
            # nonimage augmentation also works channel-wise -- even though
            # e.g. keypoints do not have channels -- in order to keep the
            # random samples properly synchronized with the image augmentation
            if per_channel[i] > 0.5:
                nb_channels_i = (
                    augmentables[i].shape[2]
                    if len(augmentables[i].shape) >= 3
                    else 1)
                alpha = np.average(alphas[i, 0:nb_channels_i])
            else:
                alpha = alphas[i, 0]
            assert 0 <= alpha <= 1.0, (
                "Expected 'alpha' to be in the interval [0.0, 1.0]. "
                "Got %.4f." % (alpha,))

            # We decide here for one branch output or another.
            # For coordinate-based augmentables there is really no good other
            # way here as you can't just choose "a bit" of one coordinate
            # (averaging almost never makes sense). For heatmaps we could
            # do some blending, but choose here to not do that to better
            # align with segmentation maps.
            # So if the alpha is >= 0.5 (branch A is more visible than
            # branch B), we pick the results of branch A and only A,
            # otherwise the result of branch B.
            if alpha >= 0.5:
                result[i] = outputs_first_i
            else:
                result[i] = outputs_second_i

        return result

    def _generate_branch_outputs(self, augmentables, augfunc_name, hooks,
                                 parents):
        outputs_first = augmentables
        outputs_second = augmentables

        if self._is_propagating(augmentables, hooks, parents):
            if self.first is not None:
                outputs_first = getattr(self.first, augfunc_name)(
                    augm_utils.copy_augmentables(augmentables),
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is not None:
                outputs_second = getattr(self.second, augfunc_name)(
                    augm_utils.copy_augmentables(augmentables),
                    parents=parents + [self],
                    hooks=hooks
                )

        return outputs_first, outputs_second

    def _is_propagating(self, augmentables, hooks, parents):
        return (hooks is None
                or hooks.is_propagating(augmentables, augmenter=self,
                                        parents=parents, default=True))

    def _to_deterministic(self):
        aug = self.copy()
        aug.first = (
            aug.first.to_deterministic() if aug.first is not None else None)
        aug.second = (
            aug.second.to_deterministic() if aug.second is not None else None)
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self):
        return [self.factor, self.per_channel]

    def get_children_lists(self):
        return [lst for lst in [self.first, self.second] if lst is not None]

    def __str__(self):
        pattern = (
            "%s("
            "factor=%s, per_channel=%s, name=%s, first=%s, second=%s, "
            "deterministic=%s"
            ")"
        )
        return pattern % (
            self.__class__.__name__, self.factor, self.per_channel, self.name,
            self.first, self.second, self.deterministic)


# FIXME the output of the third example makes it look like per_channel isn't
#       working
# TODO switch line strings to either-or approach, like polygons
class AlphaElementwise(Alpha):
    """
    Alpha-blend two image sources using alpha/opacity values sampled per pixel.

    This is the same as :class:`Alpha`, except that the opacity factor is
    sampled once per *pixel* instead of once per *image* (or a few times per
    image, if ``Alpha.per_channel`` is set to ``True``).

    See :class:`Alpha` for more details.

    .. note::

        It is not recommended to use ``AlphaElementwise`` with augmenters
        that change the geometry of images (e.g. horizontal flips, affine
        transformations) if you *also* want to augment coordinates (e.g.
        keypoints, polygons, ...), as it is unclear which of the two
        coordinate results (first or second branch) should be used as the
        coordinates after augmentation.

        Currently, the for keypoints and line strings the results of the
        first and second branch will be mixed. For each coordinate, the
        augmented one from the first or second branch will be picked based
        on the average alpha mask value at the corresponding spatial location.

        For polygons, only all polygons of the first or all of the second
        branch will be used, based on the average over the whole alpha mask.

    dtype support::

        See :func:`imgaug.augmenters.blend.blend_alpha`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Weighting of the results of the first branch. Values close to 0 mean
        that the results from the second branch (see parameter `second`)
        make up most of the final image.

            * If float, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be picked from that list per
              image.
            * If ``StochasticParameter``, then that parameter will be used to
              sample a value per image.

    first : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the first of the two branches.

            * If ``None``, then the input images will be reused as the output
              of the first branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    second : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the second of the two branches.

            * If ``None``, then the input images will be reused as the output
              of the second branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AlphaElementwise(0.5, iaa.Grayscale(1.0))

    Convert each image to pure grayscale and alpha-blend the result with the
    original image using an alpha of ``50%`` for all pixels, thereby removing
    about ``50%`` of all color. This is equivalent to ``iaa.Grayscale(0.5)``.
    This is also equivalent to ``iaa.Alpha(0.5, iaa.Grayscale(1.0))``, as
    the opacity has a fixed value of ``0.5`` and is hence identical for all
    pixels.

    >>> aug = iaa.AlphaElementwise((0, 1.0), iaa.Grayscale(1.0))

    Same as in the previous example, but the alpha factor is sampled uniformly
    from the interval ``[0.0, 1.0]`` once per pixel, thereby removing a random
    fraction of all colors from each pixel. This is equivalent to
    ``iaa.Grayscale((0.0, 1.0))``.

    >>> aug = iaa.AlphaElementwise(
    >>>     (0.0, 1.0),
    >>>     iaa.Affine(rotate=(-20, 20)),
    >>>     per_channel=0.5)

    First, rotate each image by a random degree sampled uniformly from the
    interval ``[-20, 20]``. Then, alpha-blend that new image with the original
    one using a random factor sampled uniformly from the interval
    ``[0.0, 1.0]`` per pixel. For ``50%`` of all images, the blending happens
    channel-wise and the factor is sampled independently per pixel *and*
    channel (``per_channel=0.5``). As a result, e.g. the red channel may look
    visibly rotated (factor near ``1.0``), while the green and blue channels
    may not look rotated (factors near ``0.0``).

    >>> aug = iaa.AlphaElementwise(
    >>>     (0.0, 1.0),
    >>>     first=iaa.Add(100),
    >>>     second=iaa.Multiply(0.2))

    Apply two branches of augmenters -- ``A`` and ``B`` -- *independently*
    to input images and alpha-blend the results of these branches using a
    factor ``f``. Branch ``A`` increases image pixel intensities by ``100``
    and ``B`` multiplies the pixel intensities by ``0.2``. ``f`` is sampled
    uniformly from the interval ``[0.0, 1.0]`` per pixel. The resulting images
    contain a bit of ``A`` and a bit of ``B``.

    >>> aug = iaa.AlphaElementwise([0.25, 0.75], iaa.MedianBlur(13))

    Apply median blur to each image and alpha-blend the result with the
    original image using an alpha factor of either exactly ``0.25`` or
    exactly ``0.75`` (sampled once per pixel).

    """

    # Currently the mode is only used for keypoint augmentation.
    # either or: use all keypoints from first or all from second branch (based
    #   on average of the whole mask).
    # pointwise: decide for each point whether to use the first or secon
    #   branch's keypoint (based on the average mask value at the point's
    #   xy-location).
    _MODE_EITHER_OR = "either-or"
    _MODE_POINTWISE = "pointwise"
    _MODES = [_MODE_POINTWISE, _MODE_EITHER_OR]

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

        # this controls how keypoints and polygons are augmented
        # keypoint mode currently also affects line strings and bounding boxes
        self._keypoints_mode = self._MODE_POINTWISE
        self._polygons_mode = self._MODE_EITHER_OR

    def _augment_images(self, images, random_state, parents, hooks):
        outputs_first, outputs_second = self._generate_branch_outputs(
            images, "augment_images", hooks, parents)

        nb_images = len(images)
        rngs = random_state.duplicate(nb_images+1)
        per_channel = self.per_channel.draw_samples(nb_images,
                                                    random_state=rngs[-1])

        result = images
        gen = enumerate(zip(images, outputs_first, outputs_second))
        for i, (image, image_first, image_second) in gen:
            h, w, nb_channels = image.shape[0:3]
            mask = self._sample_mask(h, w, nb_channels, per_channel[i], rngs[i])

            result[i] = blend_alpha(image_first, image_second, mask,
                                    eps=self.epsilon)

        return result

    def _sample_mask(self, height, width, nb_channels, per_channel, rng):
        if per_channel > 0.5:
            mask = [
                self.factor.draw_samples((height, width), random_state=rng)
                for _ in sm.xrange(nb_channels)]
            mask = np.stack(mask, axis=-1).astype(np.float64)
        else:
            # TODO When this was wrongly sampled directly as (H,W,C) no
            #      test for AlphaElementwise ended up failing. That should not
            #      happen.
            # note that this should not be (H,W,1) as otherwise
            # SimplexNoiseAlpha fails as noise params expected a call of (H,W)
            mask = self.factor.draw_samples((height, width), random_state=rng)
            mask = np.tile(mask[..., np.newaxis], (1, 1, nb_channels))

        assert 0 <= mask.item(0) <= 1.0, (
            "Expected 'factor' samples to be in the interval "
            "[0.0, 1.0]. Got min %.4f and max %.4f." % (
                np.min(mask), np.max(mask),))

        return mask

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return self._augment_hms_and_segmaps(
            heatmaps, random_state, parents, hooks,
            "augment_heatmaps", "arr_0to1")

    def _augment_segmentation_maps(self, segmaps, random_state, parents,
                                   hooks):
        return self._augment_hms_and_segmaps(
            segmaps, random_state, parents, hooks,
            "augment_segmentation_maps", "arr")

    def _augment_hms_and_segmaps(self, augmentables, random_state, parents,
                                 hooks, augfunc_name, arr_attr_name):
        outputs_first, outputs_second = self._generate_branch_outputs(
            augmentables, augfunc_name, hooks, parents)

        result = augmentables
        nb_images = len(augmentables)
        rngs = random_state.duplicate(nb_images+1)
        per_channel = self.per_channel.draw_samples(nb_images,
                                                    random_state=rngs[-1])

        gen = enumerate(zip(augmentables, outputs_first, outputs_second))
        for i, (augmentable_i, outputs_first_i, outputs_second_i) in gen:
            arr_i = getattr(augmentable_i, arr_attr_name)
            arr_first_i = getattr(outputs_first_i, arr_attr_name)
            arr_second_i = getattr(outputs_second_i, arr_attr_name)

            h_img, w_img = augmentable_i.shape[0:2]
            h_arr, w_arr = arr_i.shape[0:2]
            nb_channels_img = (
                augmentable_i.shape[2] if len(augmentable_i.shape) >= 3 else 1)

            per_channel_i = per_channel[i]

            mask_image = self._sample_mask(h_img, w_img, nb_channels_img,
                                           per_channel_i, rngs[i])

            # Average over channels, resize to heatmap/segmap array size
            # (+clip for cubic interpolation). We can use none-NN interpolation
            # for segmaps here as this is just the mask and not the segmap
            # array.
            mask_arr = iadt.clip_(
                ia.imresize_single_image(
                    np.average(np.atleast_3d(mask_image), axis=2),
                    (h_arr, w_arr)
                ),
                0, 1.0)

            mask_arr_binarized = (mask_arr >= 0.5)
            arr_i_aug = blend_alpha(
                arr_first_i, arr_second_i, mask_arr_binarized,
                eps=self.epsilon)

            setattr(result[i], arr_attr_name, arr_i_aug)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        def _get_coords_func(augmentable):
            return [(kp.x, kp.y) for kp in augmentable.keypoints]

        def _assign_coords_func(augmentable, coords):
            kps = [ia.Keypoint(x=x, y=y) for x, y in coords]
            return augmentable.deepcopy(keypoints=kps)

        return self._augment_coordinate_based(
            keypoints_on_images, random_state, parents, hooks,
            "augment_keypoints", _get_coords_func, _assign_coords_func)

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        # This currently uses an either-or approach.
        # Using pointwise augmentation is problematic here, because the order
        # of the points may have changed (e.g. from clockwise to
        # counter-clockwise). For polygons, it is also overall more likely
        # that some child-augmenter added/deleted points and we would need a
        # polygon recoverer.
        # Overall it seems to be the better approach to use all polygons
        # from one branch or the other, which guarantuees their validity.
        # TODO decide the either-or not based on the whole average mask
        #      value but on the average mask value within the polygon's area?
        with _switch_keypoint_mode_temporarily(self, self._polygons_mode):
            return self._augment_polygons_as_keypoints(
                polygons_on_images, random_state, parents, hooks)

    def _augment_coordinate_based(self, inputs, random_state, parents, hooks,
                                  augfunc_name, get_coords_func,
                                  assign_coords_func):
        assert self._keypoints_mode in self._MODES, (
            "Expected _keypoint_mode to be one of: %s. Got: %s." % (
                self._MODES, self._keypoint_mode))

        outputs_first, outputs_second = self._generate_branch_outputs(
            inputs, augfunc_name, hooks, parents)

        result = inputs
        nb_images = len(inputs)
        rngs = random_state.duplicate(nb_images+1)
        per_channel = self.per_channel.draw_samples(nb_images,
                                                    random_state=rngs[-1])

        gen = enumerate(zip(inputs, outputs_first, outputs_second))
        for i, (augmentable_i, outputs_first_i, outputs_second_i) in gen:
            coords = get_coords_func(augmentable_i)
            coords_first = get_coords_func(outputs_first_i)
            coords_second = get_coords_func(outputs_second_i)

            h_img, w_img = augmentable_i.shape[0:2]
            nb_channels_img = (
                augmentable_i.shape[2] if len(augmentable_i.shape) >= 3 else 1)

            per_channel_i = per_channel[i]

            mask_image = self._sample_mask(h_img, w_img, nb_channels_img,
                                           per_channel_i, rngs[i])

            if self._keypoints_mode == self._MODE_POINTWISE:
                # Augment pointwise, i.e. check for each point and its
                # xy-location the average mask value and pick based on that
                # either the point from the first or second branch.
                assert len(coords_first) == len(coords_second), (
                    "Got different numbers of coordinates before/after "
                    "augmentation in AlphaElementwise. The number of "
                    "coordinates is currently not allowed to change for this "
                    "augmenter. Input contained %d coordinates, first branch "
                    "%d, second branch %d." % (
                        len(coords), len(coords_first), len(coords_second)))

                coords_aug = []
                subgen = zip(coords, coords_first, coords_second)
                for coord, coord_first, coord_second in subgen:
                    x_int = int(np.round(coord[0]))
                    y_int = int(np.round(coord[1]))
                    if 0 <= y_int < h_img and 0 <= x_int < w_img:
                        alpha = np.average(mask_image[y_int, x_int, :])
                        if alpha > 0.5:
                            coords_aug.append(coord_first)
                        else:
                            coords_aug.append(coord_second)
                    else:
                        coords_aug.append((x_int, y_int))
            else:
                # Augment with an either-or approach over all points, i.e.
                # based on the average of the whole mask, either all points
                # from the first or all points from the second branch are
                # used.
                # Note that we ensured above that _keypoint_mode must be
                # _MODE_EITHER_OR if it wasn't _MODE_POINTWISE.
                mask_image_avg = np.average(mask_image)
                if mask_image_avg > 0.5:
                    coords_aug = coords_first
                else:
                    coords_aug = coords_second

            result[i] = assign_coords_func(result[i], coords_aug)

        return result


# Helper for AlphaElementwise to temporarily change the augmentation mode
# of keypoints.
class _switch_keypoint_mode_temporarily(object):
    def __init__(self, augmenter, mode):
        self.augmenter = augmenter
        self.mode = mode
        self.mode_orig = None

    def __enter__(self):
        # pylint:disable=protected-access
        self.mode_orig = self.augmenter._keypoints_mode
        self.augmenter._keypoints_mode = self.mode

    def __exit__(self, exception_type, exception_value, traceback):
        self.augmenter._keypoints_mode = self.mode_orig


class SimplexNoiseAlpha(AlphaElementwise):
    """Alpha-blend two image sources using simplex noise alpha masks.

    The alpha masks are sampled using a simplex noise method, roughly creating
    connected blobs of 1s surrounded by 0s. If nearest neighbour
    upsampling is used, these blobs can be rectangular with sharp edges.

    dtype support::

        See ``imgaug.augmenters.blend.AlphaElementwise``.

    Parameters
    ----------
    first : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the first of the two branches.

            * If ``None``, then the input images will be reused as the output
              of the first branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a Sequential and used as the augmenter.

    second : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the second of the two branches.

            * If ``None``, then the input images will be reused as the output
              of the second branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a Sequential and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``, otherwise as ``False``.

    size_px_max : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        The simplex noise is always generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).

            * If int, then that number will be used as the size for all
              iterations.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per iteration from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per iteration
              at random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.

            * If ``None``, then either ``nearest`` or ``linear`` or ``cubic``
              is picked. Most weight is put on ``linear``, followed by
              ``cubic``.
            * If ``imgaug.ALL``, then either ``nearest`` or ``linear`` or
              ``area`` or ``cubic`` is picked per iteration (all same
              probability).
            * If a string, then that value will be used as the method (must be
              ``nearest`` or ``linear`` or ``area`` or ``cubic``).
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per iteration.

    iterations : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        How often to repeat the simplex noise generation process per image.

            * If ``int``, then that number will be used as the iterations for
              all images.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per image from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per image at
              random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per image.

    aggregation_method : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        The noise maps (from each iteration) are combined to one noise map
        using an aggregation process. This parameter defines the method used
        for that process. Valid methods are ``min``, ``max`` or ``avg``,
        where ``min`` combines the noise maps by taking the (elementwise)
        minimum over all iteration's results, ``max`` the (elementwise)
        maximum and ``avg`` the (elementwise) average.

            * If ``imgaug.ALL``, then a random value will be picked per image
              from the valid ones.
            * If a string, then that value will always be used as the method.
            * If a list of string, then a random value will be picked from
              that list per image.
            * If a ``StochasticParameter``, then a random value will be
              sampled from that paramter per image.

    sigmoid : bool or number, optional
        Whether to apply a sigmoid function to the final noise maps, resulting
        in maps that have more extreme values (close to 0.0 or 1.0).

            * If ``bool``, then a sigmoid will always (``True``) or never
              (``False``) be applied.
            * If a number ``p`` with ``0<=p<=1``, then a sigmoid will be
              applied to ``p`` percent of all final noise maps.

    sigmoid_thresh : None or number or tuple of number or imgaug.parameters.StochasticParameter, optional
        Threshold of the sigmoid, when applied. Thresholds above zero
        (e.g. ``5.0``) will move the saddle point towards the right, leading
        to more values close to 0.0.

            * If ``None``, then ``Normal(0, 5.0)`` will be used.
            * If number, then that threshold will be used for all images.
            * If tuple of two numbers ``(a, b)``, then a random value will
              be sampled per image from the interval ``[a, b]``.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.SimplexNoiseAlpha(iaa.EdgeDetect(1.0))

    Detect per image all edges, mark them in a black and white image and
    then alpha-blend the result with the original image using simplex noise
    masks.

    >>> aug = iaa.SimplexNoiseAlpha(
    >>>     iaa.EdgeDetect(1.0),
    >>>     upscale_method="nearest")

    Same as in the previous example, but using only nearest neighbour
    upscaling to scale the simplex noise masks to the final image sizes, i.e.
    no nearest linear upsampling is used. This leads to rectangles with sharp
    edges.

    >>> aug = iaa.SimplexNoiseAlpha(
    >>>     iaa.EdgeDetect(1.0),
    >>>     upscale_method="linear")

    Same as in the previous example, but using only linear upscaling to
    scale the simplex noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used. This leads to rectangles with smooth edges.

    >>> aug = iaa.SimplexNoiseAlpha(
    >>>     iaa.EdgeDetect(1.0),
    >>>     sigmoid_thresh=iap.Normal(10.0, 5.0))

    Same as in the first example, but using a threshold for the sigmoid
    function that is further to the right. This is more conservative, i.e.
    the generated noise masks will be mostly black (values around ``0.0``),
    which means that most of the original images (parameter/branch `second`)
    will be kept, rather than using the results of the augmentation
    (parameter/branch `first`).

    """

    def __init__(self, first=None, second=None, per_channel=False,
                 size_px_max=(2, 16), upscale_method=None,
                 iterations=(1, 3), aggregation_method="max",
                 sigmoid=True, sigmoid_thresh=None,
                 name=None, deterministic=False, random_state=None):
        upscale_method_default = iap.Choice(["nearest", "linear", "cubic"],
                                            p=[0.05, 0.6, 0.35])
        sigmoid_thresh_default = iap.Normal(0.0, 5.0)

        noise = iap.SimplexNoise(
            size_px_max=size_px_max,
            upscale_method=(upscale_method
                            if upscale_method is not None
                            else upscale_method_default)
        )

        if iterations != 1:
            noise = iap.IterativeNoiseAggregator(
                noise,
                iterations=iterations,
                aggregation_method=aggregation_method
            )

        use_sigmoid = (
            sigmoid is True
            or (ia.is_single_number(sigmoid) and sigmoid >= 0.01))
        if use_sigmoid:
            noise = iap.Sigmoid.create_for_noise(
                noise,
                threshold=(sigmoid_thresh
                           if sigmoid_thresh is not None
                           else sigmoid_thresh_default),
                activated=sigmoid
            )

        super(SimplexNoiseAlpha, self).__init__(
            factor=noise, first=first, second=second, per_channel=per_channel,
            name=name, deterministic=deterministic, random_state=random_state
        )


class FrequencyNoiseAlpha(AlphaElementwise):
    """Alpha-blend two image sources using frequency noise masks.

    The alpha masks are sampled using frequency noise of varying scales,
    which can sometimes create large connected blobs of ``1`` s surrounded
    by ``0`` s and other times results in smaller patterns. If nearest
    neighbour upsampling is used, these blobs can be rectangular with sharp
    edges.

    dtype support::

        See ``imgaug.augmenters.blend.AlphaElementwise``.

    Parameters
    ----------
    exponent : number or tuple of number of list of number or imgaug.parameters.StochasticParameter, optional
        Exponent to use when scaling in the frequency domain.
        Sane values are in the range ``-4`` (large blobs) to ``4`` (small
        patterns). To generate cloud-like structures, use roughly ``-2``.

            * If number, then that number will be used as the exponent for all
              iterations.
            * If tuple of two numbers ``(a, b)``, then a value will be sampled
              per iteration from the interval ``[a, b]``.
            * If a list of numbers, then a value will be picked per iteration
              at random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per iteration.

    first : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the first of the two branches.

            * If ``None``, then the input images will be reused as the output
              of the first branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    second : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the second of the two branches.

            * If ``None``, then the input images will be reused as the output
              of the second branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``, otherwise as ``False``.

    size_px_max : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        The noise is generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).

            * If ``int``, then that number will be used as the size for all
              iterations.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per iteration from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per
              iteration at random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.

            * If ``None``, then either ``nearest`` or ``linear`` or ``cubic``
              is picked. Most weight is put on ``linear``, followed by
              ``cubic``.
            * If ``imgaug.ALL``, then either ``nearest`` or ``linear`` or
              ``area`` or ``cubic`` is picked per iteration (all same
              probability).
            * If string, then that value will be used as the method (must be
              ``nearest`` or ``linear`` or ``area`` or ``cubic``).
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per iteration.

    iterations : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        How often to repeat the simplex noise generation process per
        image.

            * If ``int``, then that number will be used as the iterations for
              all images.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per image from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per image at
              random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per image.

    aggregation_method : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        The noise maps (from each iteration) are combined to one noise map
        using an aggregation process. This parameter defines the method used
        for that process. Valid methods are ``min``, ``max`` or ``avg``,
        where 'min' combines the noise maps by taking the (elementwise) minimum
        over all iteration's results, ``max`` the (elementwise) maximum and
        ``avg`` the (elementwise) average.

            * If ``imgaug.ALL``, then a random value will be picked per image
              from the valid ones.
            * If a string, then that value will always be used as the method.
            * If a list of string, then a random value will be picked from
              that list per image.
            * If a ``StochasticParameter``, then a random value will be sampled
              from that parameter per image.

    sigmoid : bool or number, optional
        Whether to apply a sigmoid function to the final noise maps, resulting
        in maps that have more extreme values (close to ``0.0`` or ``1.0``).

            * If ``bool``, then a sigmoid will always (``True``) or never
              (``False``) be applied.
            * If a number ``p`` with ``0<=p<=1``, then a sigmoid will be applied to
              ``p`` percent of all final noise maps.

    sigmoid_thresh : None or number or tuple of number or imgaug.parameters.StochasticParameter, optional
        Threshold of the sigmoid, when applied. Thresholds above zero
        (e.g. ``5.0``) will move the saddle point towards the right, leading to
        more values close to ``0.0``.

            * If ``None``, then ``Normal(0, 5.0)`` will be used.
            * If number, then that threshold will be used for all images.
            * If tuple of two numbers ``(a, b)``, then a random value will
              be sampled per image from the range ``[a, b]``.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(1.0))

    Detect per image all edges, mark them in a black and white image and
    then alpha-blend the result with the original image using frequency noise
    masks.

    >>> aug = iaa.FrequencyNoiseAlpha(
    >>>     first=iaa.EdgeDetect(1.0),
    >>>     upscale_method="nearest")

    Same as the first example, but using only linear upscaling to
    scale the frequency noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used. This results in smooth edges.

    >>> aug = iaa.FrequencyNoiseAlpha(
    >>>     first=iaa.EdgeDetect(1.0),
    >>>     upscale_method="linear")

    Same as the first example, but using only linear upscaling to
    scale the frequency noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used. This results in smooth edges.

    >>> aug = iaa.FrequencyNoiseAlpha(
    >>>     first=iaa.EdgeDetect(1.0),
    >>>     upscale_method="linear",
    >>>     exponent=-2,
    >>>     sigmoid=False)

    Same as in the previous example, but with the exponent set to a constant
    ``-2`` and the sigmoid deactivated, resulting in cloud-like patterns
    without sharp edges.

    >>> aug = iaa.FrequencyNoiseAlpha(
    >>>     first=iaa.EdgeDetect(1.0),
    >>>     sigmoid_thresh=iap.Normal(10.0, 5.0))

    Same as the first example, but using a threshold for the sigmoid function
    that is further to the right. This is more conservative, i.e. the generated
    noise masks will be mostly black (values around ``0.0``), which means that
    most of the original images (parameter/branch `second`) will be kept,
    rather than using the results of the augmentation (parameter/branch
    `first`).

    """

    def __init__(self, exponent=(-4, 4), first=None, second=None,
                 per_channel=False, size_px_max=(4, 16), upscale_method=None,
                 iterations=(1, 3), aggregation_method=["avg", "max"],
                 sigmoid=0.5, sigmoid_thresh=None,
                 name=None, deterministic=False, random_state=None):
        # pylint: disable=dangerous-default-value
        upscale_method_default = iap.Choice(["nearest", "linear", "cubic"],
                                            p=[0.05, 0.6, 0.35])
        sigmoid_thresh_default = iap.Normal(0.0, 5.0)

        noise = iap.FrequencyNoise(
            exponent=exponent,
            size_px_max=size_px_max,
            upscale_method=(upscale_method
                            if upscale_method is not None
                            else upscale_method_default)
        )

        if iterations != 1:
            noise = iap.IterativeNoiseAggregator(
                noise,
                iterations=iterations,
                aggregation_method=aggregation_method
            )

        use_sigmoid = (
            sigmoid is True
            or (ia.is_single_number(sigmoid) and sigmoid >= 0.01))
        if use_sigmoid:
            noise = iap.Sigmoid.create_for_noise(
                noise,
                threshold=(sigmoid_thresh
                           if sigmoid_thresh is not None
                           else sigmoid_thresh_default),
                activated=sigmoid
            )

        super(FrequencyNoiseAlpha, self).__init__(
            factor=noise, first=first, second=second, per_channel=per_channel,
            name=name, deterministic=deterministic, random_state=random_state
        )
