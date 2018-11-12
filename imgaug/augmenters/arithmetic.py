"""
Augmenters that perform simple arithmetic changes.

Do not import directly from this file, as the categorization is not final.
Use instead::

    from imgaug import augmenters as iaa

and then e.g.::

    `seq = iaa.Sequential([iaa.Add((-5, 5)), iaa.Multiply((0.9, 1.1))])`

List of augmenters:

    * Add
    * AddElementwise
    * AdditiveGaussianNoise
    * Multiply
    * MultiplyElementwise
    * Dropout
    * CoarseDropout
    * ReplaceElementwise
    * SaltAndPepper
    * CoarseSaltAndPepper
    * Salt
    * CoarseSalt
    * Pepper
    * CoarsePepper
    * Invert
    * ContrastNormalization
    * JpegCompression

"""
from __future__ import print_function, division, absolute_import

from PIL import Image
import imageio
import tempfile
import numpy as np
import six.moves as sm

from . import meta
from .. import imgaug as ia
from .. import parameters as iap


class Add(meta.Augmenter):
    """
    Add a value to all pixels in an image.

    Parameters
    ----------
    value : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Value to add to all pixels.

            * If an int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the discrete range ``[a .. b]``
              will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image
              from that parameter.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
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
    >>> aug = iaa.Add(10)

    always adds a value of 10 to all pixels in the image.

    >>> aug = iaa.Add((-10, 10))

    adds a value from the discrete range ``[-10 .. 10]`` to all pixels of
    the input images. The exact value is sampled per image.

    >>> aug = iaa.Add((-10, 10), per_channel=True)

    adds a value from the discrete range ``[-10 .. 10]`` to all pixels of
    the input images. The exact value is sampled per image AND channel,
    i.e. to a red-channel it might add 5 while subtracting 7 from the
    blue channel of the same image.

    >>> aug = iaa.Add((-10, 10), per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """

    def __init__(self, value=0, per_channel=False, name=None,
                 deterministic=False, random_state=None):
        super(Add, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.value = iap.handle_discrete_param(value, "value", value_range=(-255, 255), tuple_to_uniform=True,
                                               list_to_choice=True, allow_floats=False)
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)

        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.int32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.value.draw_samples((nb_channels,), random_state=rs_image).astype(image.dtype)
                for c, sample in enumerate(samples):
                    # TODO make value range more flexible
                    ia.do_assert(-255 <= sample <= 255)
                    image[..., c] += sample
            else:
                sample = self.value.draw_sample(random_state=rs_image).astype(image.dtype)
                ia.do_assert(-255 <= sample <= 255) # TODO make value range more flexible
                image += sample

            image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.value, self.per_channel]


class AddElementwise(meta.Augmenter):
    """
    Add values to the pixels of images with possibly different values for neighbouring pixels.

    While the Add Augmenter adds a constant value per image, this one can
    add different values (sampled per pixel).

    Parameters
    ----------
    value : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Value to add to the pixels.

            * If an int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then values from the discrete range ``[a .. b]``
              will be sampled.
            * If a list of integers, a random value will be sampled from the list
              per image.
            * If a StochasticParameter, then values will be sampled per pixel
              (and possibly channel) from that parameter.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
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
    >>> aug = iaa.AddElementwise(10)

    always adds a value of 10 to all pixels in the image.

    >>> aug = iaa.AddElementwise((-10, 10))

    samples per pixel a value from the discrete range ``[-10 .. 10]`` and
    adds that value to the pixel.

    >>> aug = iaa.AddElementwise((-10, 10), per_channel=True)

    samples per pixel *and channel* a value from the discrete
    range ``[-10 .. 10]`` ands adds it to the pixel's value. Therefore,
    added values may differ between channels of the same pixel.

    >>> aug = iaa.AddElementwise((-10, 10), per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """

    def __init__(self, value=0, per_channel=False, name=None, deterministic=False, random_state=None):
        super(AddElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.value = iap.handle_discrete_param(value, "value", value_range=(-255, 255), tuple_to_uniform=True,
                                               list_to_choice=True, allow_floats=False)
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)

        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            seed = seeds[i]
            image = images[i].astype(np.int32)
            height, width, nb_channels = image.shape
            rs_image = ia.new_random_state(seed)
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                samples = self.value\
                    .draw_samples((height, width, nb_channels), random_state=rs_image)\
                    .astype(image.dtype)
            else:
                samples = self.value.draw_samples((height, width, 1), random_state=rs_image).astype(image.dtype)
                samples = np.tile(samples, (1, 1, nb_channels))
            after_add = image + samples

            after_add = meta.clip_augmented_image_(after_add, 0, 255) # TODO make value range more flexible
            after_add = meta.restore_augmented_image_dtype_(after_add, input_dtypes[i])

            result[i] = after_add

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.value, self.per_channel]


def AdditiveGaussianNoise(loc=0, scale=0, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Add gaussian noise (aka white noise) to images.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Mean of the normal distribution that generates the noise.

            * If a number, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the range ``a <= x <= b`` will
              be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the normal distribution that generates the noise.
        Must be ``>= 0``. If 0 then only `loc` will be used.

            * If an int or float, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the range ``a <= x <= b`` will
              be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    per_channel : bool or float, optional
        Whether to use the same noise value per pixel for all channels (False)
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
    >>> aug = iaa.GaussianNoise(scale=0.1*255)

    adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images.

    >>> aug = iaa.GaussianNoise(scale=(0, 0.1*255))

    adds gaussian noise from the distribution ``N(0, s)`` to images,
    where s is sampled per image from the range ``0 <= s <= 0.1*255``.

    >>> aug = iaa.GaussianNoise(scale=0.1*255, per_channel=True)

    adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images,
    where the noise value is different per pixel *and* channel (e.g. a
    different one for red, green and blue channels for the same pixel).

    >>> aug = iaa.GaussianNoise(scale=0.1*255, per_channel=0.5)

    adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images,
    where the noise value is sometimes (50 percent of all cases) the same
    per pixel for all channels and sometimes different (other 50 percent).

    """
    loc2 = iap.handle_continuous_param(loc, "loc", value_range=None, tuple_to_uniform=True, list_to_choice=True)
    scale2 = iap.handle_continuous_param(scale, "scale", value_range=(0, None), tuple_to_uniform=True,
                                         list_to_choice=True)

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return AddElementwise(iap.Normal(loc=loc2, scale=scale2), per_channel=per_channel, name=name,
                          deterministic=deterministic, random_state=random_state)


class Multiply(meta.Augmenter):
    """
    Multiply all pixels in an image with a specific value.

    This augmenter can be used to make images lighter or darker.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The value with which to multiply the pixel values in each image.

            * If a number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the range ``a <= x <= b`` will
              be sampled per image and used for all pixels.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then that parameter will be used to
              sample a new value per image.

    per_channel : bool or float, optional
        Whether to use the same multiplier per pixel for all channels (False)
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
    >>> aug = iaa.Multiply(2.0)

    would multiply all images by a factor of 2, making the images
    significantly brighter.

    >>> aug = iaa.Multiply((0.5, 1.5))

    would multiply images by a random value from the range ``0.5 <= x <= 1.5``,
    making some images darker and others brighter.

    """

    def __init__(self, mul=1.0, per_channel=False, name=None,
                 deterministic=False, random_state=None):
        super(Multiply, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.mul = iap.handle_continuous_param(mul, "mul", value_range=(0, None), tuple_to_uniform=True,
                                               list_to_choice=True)
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)

        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.float32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.mul.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    ia.do_assert(sample >= 0)
                    image[..., c] *= sample
            else:
                sample = self.mul.draw_sample(random_state=rs_image)
                ia.do_assert(sample >= 0)
                image *= sample

            image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul, self.per_channel]


class MultiplyElementwise(meta.Augmenter):
    """
    Multiply values of pixels with possibly different values for neighbouring pixels.

    While the Multiply Augmenter uses a constant multiplier per image,
    this one can use different multipliers per pixel.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The value by which to multiply the pixel values in the image.

            * If a number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the range ``a <= x <= b`` will
              be sampled per image and pixel.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then that parameter will be used to
              sample a new value per image and pixel.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
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
    >>> aug = iaa.MultiplyElementwise(2.0)

    multiply all images by a factor of 2.0, making them significantly
    bighter.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5))

    samples per pixel a value from the range ``0.5 <= x <= 1.5`` and
    multiplies the pixel with that value.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=True)

    samples per pixel *and channel* a value from the range
    ``0.5 <= x <= 1.5`` ands multiplies the pixel by that value. Therefore,
    added multipliers may differ between channels of the same pixel.

    >>> aug = iaa.AddElementwise((0.5, 1.5), per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """

    def __init__(self, mul=1.0, per_channel=False, name=None, deterministic=False, random_state=None):
        super(MultiplyElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.mul = iap.handle_continuous_param(mul, "mul", value_range=(0, None), tuple_to_uniform=True,
                                               list_to_choice=True)
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)

        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            seed = seeds[i]
            image = images[i].astype(np.float32)
            height, width, nb_channels = image.shape
            rs_image = ia.new_random_state(seed)
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                samples = self.mul.draw_samples((height, width, nb_channels), random_state=rs_image)
            else:
                samples = self.mul.draw_samples((height, width, 1), random_state=rs_image)
                samples = np.tile(samples, (1, 1, nb_channels))
            image = image * samples

            image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul, self.per_channel]


def Dropout(p=0, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Augmenter that sets a certain fraction of pixels in images to zero.

    Parameters
    ----------
    p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. set to zero).

            * If a float, then that value will be used for all images. A value
              of 1.0 would mean that all pixels will be dropped and 0.0 that
              no pixels would be dropped. A value of 0.05 corresponds to 5
              percent of all pixels dropped.
            * If a tuple ``(a, b)``, then a value p will be sampled from the
              range ``a <= p <= b`` per image and be used as the pixel's dropout
              probability.
            * If a StochasticParameter, then this parameter will be used to
              determine per pixel whether it should be dropped (sampled value
              of 0) or shouldn't (sampled value of 1).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    per_channel : bool or float, optional
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Dropout(0.02)

    drops 2 percent of all pixels.

    >>> aug = iaa.Dropout((0.0, 0.05))

    drops in each image a random fraction of all pixels, where the fraction
    is in the range ``0.0 <= x <= 0.05``.

    >>> aug = iaa.Dropout(0.02, per_channel=True)

    drops 2 percent of all pixels in a channel-wise fashion, i.e. it is unlikely
    for any pixel to have all channels set to zero (black pixels).

    >>> aug = iaa.Dropout(0.02, per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """
    if ia.is_single_number(p):
        p2 = iap.Binomial(1 - p)
    elif ia.is_iterable(p):
        ia.do_assert(len(p) == 2)
        ia.do_assert(p[0] < p[1])
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        p2 = iap.Binomial(iap.Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, iap.StochasticParameter):
        p2 = p
    else:
        raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return MultiplyElementwise(p2, per_channel=per_channel, name=name, deterministic=deterministic,
                               random_state=random_state)


def CoarseDropout(p=0, size_px=None, size_percent=None, per_channel=False, min_size=4, name=None, deterministic=False,
                  random_state=None):
    """
    Augmenter that sets rectangular areas within images to zero.

    In contrast to Dropout, these areas can have larger sizes.
    (E.g. you might end up with three large black rectangles in an image.)
    Note that the current implementation leads to correlated sizes,
    so when there is one large area that is dropped, there is a high likelihood
    that all other dropped areas are also large.

    This method is implemented by generating the dropout mask at a
    lower resolution (than the image has) and then upsampling the mask
    before dropping the pixels.

    Parameters
    ----------
    p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. set to zero).

            * If a float, then that value will be used for all pixels. A value
              of 1.0 would mean, that all pixels will be dropped. A value of
              0.0 would lead to no pixels being dropped.
            * If a tuple ``(a, b)``, then a value p will be sampled from the
              range ``a <= p <= b`` per image and be used as the pixel's dropout
              probability.
            * If a StochasticParameter, then this parameter will be used to
              determine per pixel whether it should be dropped (sampled value
              of 0) or shouldn't (sampled value of 1).

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the dropout
        mask in absolute pixel dimensions.

            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a ``3x3`` mask, which is then
              upsampled to ``HxW``, where ``H`` is the image size and W the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be sampled from the
              range ``[a..b]`` and the mask will be generated at size ``MxN``, then
              upsampled to ``HxW``.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the dropout
        mask *in percent* of the input image.

            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from ``(p*H)x(p*W)`` and later upsampled
              to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be sampled from the
              interval ``(a, b)`` and used as the percentages, i.e the mask size
              will be ``(m*H)x(n*W)``.
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a ``1x1`` low resolution mask, leading easily
        to the whole image being dropped.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Dropout(0.02, size_percent=0.5)

    drops 2 percent of all pixels on an lower-resolution image that has
    50 percent of the original image's size, leading to dropped areas that
    have roughly 2x2 pixels size.


    >>> aug = iaa.Dropout((0.0, 0.05), size_percent=(0.05, 0.5))

    generates a dropout mask at 5 to 50 percent of image's size. In that mask,
    0 to 5 percent of all pixels are dropped (random per image).

    >>> aug = iaa.Dropout((0.0, 0.05), size_px=(2, 16))

    same as previous example, but the lower resolution image has 2 to 16 pixels
    size.

    >>> aug = iaa.Dropout(0.02, size_percent=0.5, per_channel=True)

    drops 2 percent of all pixels at 50 percent resolution (2x2 sizes)
    in a channel-wise fashion, i.e. it is unlikely
    for any pixel to have all channels set to zero (black pixels).

    >>> aug = iaa.Dropout(0.02, size_percent=0.5, per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """
    if ia.is_single_number(p):
        p2 = iap.Binomial(1 - p)
    elif ia.is_iterable(p):
        ia.do_assert(len(p) == 2)
        ia.do_assert(p[0] < p[1])
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        p2 = iap.Binomial(iap.Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, iap.StochasticParameter):
        p2 = p
    else:
        raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))

    if size_px is not None:
        p3 = iap.FromLowerResolution(other_param=p2, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        p3 = iap.FromLowerResolution(other_param=p2, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return MultiplyElementwise(p3, per_channel=per_channel, name=name, deterministic=deterministic,
                               random_state=random_state)


class ReplaceElementwise(meta.Augmenter):
    """
    Replace pixels in an image with new values.

    Parameters
    ----------
    mask : float or tuple of float or list of float or imgaug.parameters.StochasticParameter
        Mask that indicates the pixels that are supposed to be replaced.
        The mask will be thresholded with 0.5. A value of 1 then indicates a
        pixel that is supposed to be replaced.

            * If this is a float, then that value will be used as the
              probability of being a 1 per pixel.
            * If a tuple ``(a, b)``, then the probability will be sampled per image
              from the range ``a <= x <= b``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used to
              sample a mask.

    replacement : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The replacement to use at all locations that are marked as `1` in the mask.

            * If this is a number, then that value will always be used as the
              replacement.
            * If a tuple ``(a, b)``, then the replacement will be sampled pixelwise
              from the range ``a <= x <= b``.
            * If a list of number, then a random value will be picked from
              that list as the replacement per pixel.
            * If a StochasticParameter, then this parameter will be used sample
              pixelwise replacement values.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
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
    >>> aug = ReplaceElementwise(0.05, [0, 255])

    Replace 5 percent of all pixels in each image by either 0 or 255.

    """

    def __init__(self, mask, replacement, per_channel=False, name=None, deterministic=False, random_state=None):
        super(ReplaceElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.mask = iap.handle_probability_param(mask, "mask", tuple_to_uniform=True, list_to_choice=True)
        self.replacement = iap.handle_continuous_param(replacement, "replacement")
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)

        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            seed = seeds[i]
            image = images[i].astype(np.float32)
            height, width, nb_channels = image.shape
            per_channel = self.per_channel.draw_sample(random_state=ia.new_random_state(seed+1))
            if per_channel == 1:
                mask_samples = self.mask.draw_samples(
                    (height, width, nb_channels),
                    random_state=ia.new_random_state(seed+2)
                )
                replacement_samples = self.replacement.draw_samples(
                    (height, width, nb_channels),
                    random_state=ia.new_random_state(seed+3)
                )
            else:
                mask_samples = self.mask.draw_samples(
                    (height, width, 1),
                    random_state=ia.new_random_state(seed+2)
                )
                mask_samples = np.tile(mask_samples, (1, 1, nb_channels))
                replacement_samples = self.replacement.draw_samples(
                    (height, width, 1),
                    random_state=ia.new_random_state(seed+3)
                )
                replacement_samples = np.tile(replacement_samples, (1, 1, nb_channels))

            mask_thresh = mask_samples > 0.5
            image_repl = image * (~mask_thresh) + replacement_samples * mask_thresh

            image_repl = meta.clip_augmented_image_(image_repl, 0, 255) # TODO make value range more flexible
            image_repl = meta.restore_augmented_image_dtype_(image_repl, input_dtypes[i])

            result[i] = image_repl

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mask, self.replacement, self.per_channel]


def SaltAndPepper(p=0, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Adds salt and pepper noise to an image, i.e. some white-ish and black-ish pixels.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt/pepper noise.

            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled per image
              from the range ``a <= x <= b``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that salt/pepper is to be added
              at that location.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
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
    >>> aug = iaa.SaltAndPepper(0.05)

    Replaces 5 percent of all pixels with salt/pepper.

    """
    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=p,
        replacement=iap.Beta(0.5, 0.5) * 255,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )


def CoarseSaltAndPepper(p=0, size_px=None, size_percent=None, per_channel=False, min_size=4, name=None,
                        deterministic=False, random_state=None):
    """
    Adds coarse salt and pepper noise to an image, i.e. rectangles that contain noisy white-ish and black-ish pixels.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt/pepper noise.

            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled per image
              from the range ``a <= x <= b``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that salt/pepper is to be added
              at that location.

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the noise
        mask in absolute pixel dimensions.

            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a ``3x3`` mask, which is then
              upsampled to ``HxW``, where ``H`` is the image size and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be sampled from the
              range ``[a..b]`` and the mask will be generated at size ``MxN``, then
              upsampled to ``HxW``.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the noise
        mask *in percent* of the input image.

            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from ``(p*H)x(p*W)`` and later upsampled
              to ``HxW.``
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be sampled from the
              interval ``(a, b)`` and used as the percentages, i.e the mask size
              will be ``(m*H)x(n*W)``.
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a 1x1 low resolution mask, leading easily
        to the whole image being replaced.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1))

    Replaces 5 percent of all pixels with salt/pepper in an image that has
    1 to 10 percent of the input image size, then upscales the results
    to the input image size, leading to large rectangular areas being replaced.

    """
    mask = iap.handle_probability_param(p, "p", tuple_to_uniform=True, list_to_choice=True)

    if size_px is not None:
        mask_low = iap.FromLowerResolution(other_param=mask, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        mask_low = iap.FromLowerResolution(other_param=mask, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    replacement = iap.Beta(0.5, 0.5) * 255

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=mask_low,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )


def Salt(p=0, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Adds salt noise to an image, i.e. white-ish pixels.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt noise.

            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled per image
              from the range ``a <= x <= b``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that salt is to be added
              at that location.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
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
    >>> aug = iaa.Salt(0.05)

    Replaces 5 percent of all pixels with salt.

    """

    replacement01 = iap.ForceSign(
        iap.Beta(0.5, 0.5) - 0.5,
        positive=True,
        mode="invert"
    ) + 0.5
    replacement = replacement01 * 255

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(mask=p, replacement=replacement, per_channel=per_channel, name=name,
                              deterministic=deterministic, random_state=random_state)


def CoarseSalt(p=0, size_px=None, size_percent=None, per_channel=False, min_size=4, name=None, deterministic=False,
               random_state=None):
    """
    Adds coarse salt noise to an image, i.e. rectangles containing noisy white-ish pixels.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt noise.

            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled per image
              from the range ``a <= x <= b``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that salt is to be added
              at that location.

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the noise
        mask in absolute pixel dimensions.

            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a ``3x3`` mask, which is then
              upsampled to ``HxW``, where H is the image size and W the image width.
            * If a tuple ``(a, b)``, then two values M, N will be sampled from the
              range ``[a..b]`` and the mask will be generated at size ``MxN``, then
              upsampled to ``HxW``.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the noise
        mask *in percent* of the input image.

            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from ``(p*H)x(p*W)`` and later upsampled
              to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be sampled from the
              interval ``(a, b)`` and used as the percentages, i.e the mask size
              will be ``(m*H)x(n*W)``.
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a ``1x1`` low resolution mask, leading easily
        to the whole image being replaced.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1))

    Replaces 5 percent of all pixels with salt in an image that has
    1 to 10 percent of the input image size, then upscales the results
    to the input image size, leading to large rectangular areas being replaced.

    """
    mask = iap.handle_probability_param(p, "p", tuple_to_uniform=True, list_to_choice=True)

    if size_px is not None:
        mask_low = iap.FromLowerResolution(other_param=mask, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        mask_low = iap.FromLowerResolution(other_param=mask, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    replacement01 = iap.ForceSign(
        iap.Beta(0.5, 0.5) - 0.5,
        positive=True,
        mode="invert"
    ) + 0.5
    replacement = replacement01 * 255

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(mask=mask_low, replacement=replacement, per_channel=per_channel, name=name,
                              deterministic=deterministic, random_state=random_state)


def Pepper(p=0, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Adds pepper noise to an image, i.e. black-ish pixels.
    This is similar to dropout, but slower and the black pixels are not
    uniformly black.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to pepper noise.

            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled per image
              from the range ``a <= x <= b``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that pepper is to be added
              at that location.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
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
    >>> aug = iaa.Pepper(0.05)

    Replaces 5 percent of all pixels with pepper.

    """

    replacement01 = iap.ForceSign(
        iap.Beta(0.5, 0.5) - 0.5,
        positive=False,
        mode="invert"
    ) + 0.5
    replacement = replacement01 * 255

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=p,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )


def CoarsePepper(p=0, size_px=None, size_percent=None, per_channel=False, min_size=4, name=None, deterministic=False,
                 random_state=None):
    """
    Adds coarse pepper noise to an image, i.e. rectangles that contain noisy black-ish pixels.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to pepper noise.

            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled per image
              from the range ``a <= x <= b.``
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that pepper is to be added
              at that location.

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the noise
        mask in absolute pixel dimensions.

            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a ``3x3`` mask, which is then
              upsampled to ``HxW``, where ``H`` is the image size and W the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be sampled from the
              range ``[a..b]`` and the mask will be generated at size ``MxN``, then
              upsampled to ``HxW``.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the noise
        mask *in percent* of the input image.

            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from ``(p*H)x(p*W)`` and later upsampled
              to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be sampled from the
              interval ``(a, b)`` and used as the percentages, i.e the mask size
              will be ``(m*H)x(n*W)``.
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a 1x1 low resolution mask, leading easily
        to the whole image being replaced.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1))

    Replaces 5 percent of all pixels with pepper in an image that has
    1 to 10 percent of the input image size, then upscales the results
    to the input image size, leading to large rectangular areas being replaced.

    """
    mask = iap.handle_probability_param(p, "p", tuple_to_uniform=True, list_to_choice=True)

    if size_px is not None:
        mask_low = iap.FromLowerResolution(other_param=mask, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        mask_low = iap.FromLowerResolution(other_param=mask, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    replacement01 = iap.ForceSign(
        iap.Beta(0.5, 0.5) - 0.5,
        positive=False,
        mode="invert"
    ) + 0.5
    replacement = replacement01 * 255

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=mask_low,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )


class Invert(meta.Augmenter):
    """
    Augmenter that inverts all values in images.

    For the standard value range of 0-255 it converts 0 to 255, 255 to 0
    and 10 to ``(255-10)=245``.

    Let ``M`` be the maximum value possible, ``m`` the minimum value possible,
    ``v`` a value. Then the distance of ``v`` to ``m`` is ``d=abs(v-m)`` and the new value
    is given by ``v'=M-d``.

    Parameters
    ----------
    p : float or imgaug.parameters.StochasticParameter, optional
        The probability of an image to be inverted.

            * If a float, then that probability will be used for all images.
            * If a StochasticParameter, then that parameter will queried per
              image and is expected to return values in the range ``[0.0, 1.0]``,
              where values ``>0.5`` mean that the image/channel is supposed to be
              inverted. Recommended to be some form of ``imgaug.parameters.Binomial``.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_value : int or float, optional
        Minimum of the range of possible pixel values.
        For uint8 (0-255) images, this should be 0.

    max_value : int or float, optional
        Maximum of the range of possible pixel values.
        For uint8 (0-255) images, this should be 255.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Invert(0.1)

    Inverts the colors in 10 percent of all images.

    >>> aug = iaa.Invert(0.1, per_channel=0.5)

    For 50 percent of all images, it inverts all channels with a probability of
    10 percent (same as the first example). For the other 50 percent of all
    images, it inverts each channel individually with a probability of 10
    percent (so some channels of an image may end up inverted, others not).

    """

    def __init__(self, p=0, per_channel=False, min_value=0, max_value=255, name=None, deterministic=False,
                 random_state=None):
        super(Invert, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.p = iap.handle_probability_param(p, "p")
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")
        self.min_value = min_value
        self.max_value = max_value

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)

        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.int32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                p_samples = self.p.draw_samples((nb_channels,), random_state=rs_image)
                for c, p_sample in enumerate(p_samples):
                    ia.do_assert(0 <= p_sample <= 1)
                    if p_sample > 0.5:
                        image_c = image[..., c]
                        distance_from_min = np.abs(image_c - self.min_value) # d=abs(v-m)
                        image[..., c] = -distance_from_min + self.max_value # v'=M-d
            else:
                p_sample = self.p.draw_sample(random_state=rs_image)
                ia.do_assert(0 <= p_sample <= 1.0)
                if p_sample > 0.5:
                    distance_from_min = np.abs(image - self.min_value) # d=abs(v-m)
                    image = -distance_from_min + self.max_value

            image = meta.clip_augmented_image_(image, self.min_value, self.max_value)
            image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.p, self.per_channel, self.min_value, self.max_value]


# TODO merge with contrast.LinearContrast
class ContrastNormalization(meta.Augmenter):
    """
    Augmenter that changes the contrast of images.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Strength of the contrast normalization. Higher values than 1.0
        lead to higher contrast, lower values decrease the contrast.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled per image from
              the range ``a <= x <= b`` and be used as the alpha value.
            * If a list, then a random value will be sampled per image from
              that list.
            * If a StochasticParameter, then this parameter will be used to
              sample the alpha value per image.

    per_channel : bool or float, optional
        Whether to use the same value for all channels (False)
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
    >>> iaa.ContrastNormalization((0.5, 1.5))

    Decreases oder improves contrast per image by a random factor between
    0.5 and 1.5. The factor 0.5 means that any difference from the center value
    (i.e. 128) will be halved, leading to less contrast.

    >>> iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)

    Same as before, but for 50 percent of all images the normalization is done
    independently per channel (i.e. factors can vary per channel for the same
    image). In the other 50 percent of all images, the factor is the same for
    all channels.

    """

    def __init__(self, alpha=1.0, per_channel=False, name=None, deterministic=False, random_state=None):
        super(ContrastNormalization, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.alpha = iap.handle_continuous_param(alpha, "alpha", value_range=(0, None), tuple_to_uniform=True,
                                                 list_to_choice=True)
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)

        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.float32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel:
                nb_channels = images[i].shape[2]
                alphas = self.alpha.draw_samples((nb_channels,), random_state=rs_image)
                for c, alpha in enumerate(alphas):
                    image[..., c] = alpha * (image[..., c] - 128) + 128
            else:
                alpha = self.alpha.draw_sample(random_state=rs_image)
                image = alpha * (image - 128) + 128

            image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.alpha, self.per_channel]


class JpegCompression(meta.Augmenter):
    """
    Degrade image quality by applying JPEG compression to it.

    During JPEG compression, high frequency components (e.g. edges) are removed. With low compression (strength)
    only the highest frequency components are removed, while very high compression (strength) will lead to only the
    lowest frequency components "surviving". This lowers the image quality. For more details,
    see https://en.wikipedia.org/wiki/Compression_artifact.

    Note that this augmenter still returns images as numpy arrays (i.e. saves the images with JPEG compression and
    then reloads them into arrays). It does not return the raw JPEG file content.

    Parameters
    ----------
    compression : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Degree of compression used during jpeg compression within value range ``[0, 100]``. Higher values denote
        stronger compression and will cause low-frequency components to disappear. Standard values used when saving
        images are at around 75 and will usually not degrade image quality very much.

            * If a single number, then that value will be used for the compression degree.
            * If a tuple of two number ``(a, b)``, then the compression will be a
              value sampled from the interval ``[a..b]``.
            * If a list, then a random value will be sampled and used as the
              compression per image.
            * If a StochasticParameter, then ``N`` samples will be drawn from
              that parameter per ``N`` input images, each representing the compression
              for the nth image. Expected to be discrete.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.JpegCompression(compression=(80, 95))

    Removes high frquency components in images based on JPEG compression with a compression strength between
    80 and 95 (randomly sampled per image).

    """
    def __init__(self, compression=50, name=None, deterministic=False, random_state=None):
        super(JpegCompression, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        # will be converted to int during augmentation, which is why we allow floats here
        self.compression = iap.handle_continuous_param(compression, "compression", value_range=(0, 100),
                                                       tuple_to_uniform=True, list_to_choice=True)

        # The value range 1 to 95 is suggested by PIL's save() documentation
        # Values above 95 seem to not make sense (no improvement in visual quality, but large file size)
        # A value of 100 would mostly deactivate jpeg compression
        # A value of 0 would lead to no compression (instead of maximum compression)
        # We use range 1 to 100 here, because this augmenter is about generating images for training
        # and not for saving, hence we do not care about large file sizes
        self.maximum_quality = 100
        self.minimum_quality = 1

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.compression.draw_samples((nb_images,), random_state=random_state)

        for i in sm.xrange(nb_images):
            image = images[i].astype(np.float32)
            nb_channels = image.shape[-1]
            is_single_channel = (nb_channels == 1)
            if is_single_channel:
                image = image[..., 0]
            sample = int(samples[i])
            ia.do_assert(100 >= sample >= 0)
            img = Image.fromarray(image.astype(np.uint8))
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg") as f:
                # Map from compression to quality used by PIL
                # We have valid compressions from 0 to 100, i.e. 101 possible values
                quality = int(np.clip(np.round(
                    self.minimum_quality + (self.maximum_quality - self.minimum_quality) * (1.0 - (sample / 101))
                ), self.minimum_quality, self.maximum_quality))

                img.save(f, quality=quality)
                if nb_channels == 1:
                    image = imageio.imread(f.name, pilmode="L")
                else:
                    image = imageio.imread(f.name, pilmode="RGB")
            if is_single_channel:
                image = image[..., np.newaxis]
            result[i] = image
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.compression]
