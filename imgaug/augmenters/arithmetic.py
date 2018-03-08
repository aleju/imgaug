"""
Augmenters that perform simple arithmetic changes.

Do not import directly from this file, as the categorization is not final.
Use instead
    `from imgaug import augmenters as iaa`
and then e.g.
    `seq = iaa.Sequential([iaa.Add((-5, 5)), iaa.Multiply((0.9, 1.1))])`

List of augmenters:
    * Add
    * AddElementwise
    * AdditiveGaussianNoise
    * Multiply
    * Dropout
    * CoarseDropout
    * Invert
    * ContrastNormalization
"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, Binomial, DiscreteUniform, Normal, Uniform, FromLowerResolution
from .. import parameters as iap
import numpy as np
import six.moves as sm

from . import meta
from .meta import Augmenter

# TODO tests
class Add(Augmenter):
    """
    Add a value to all pixels in an image.

    Parameters
    ----------
    value : int or iterable of two ints or StochasticParameter, optional(default=0)
        Value to add to all
        pixels.
            * If an int, then that value will be used for all images.
            * If a tuple (a, b), then a value from the discrete range [a .. b]
              will be used.
            * If a StochasticParameter, then a value will be sampled per image
              from that parameter.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Add(10)

    always adds a value of 10 to all pixels in the image.

    >>> aug = iaa.Add((-10, 10))

    adds a value from the discrete range [-10 .. 10] to all pixels of
    the input images. The exact value is sampled per image.

    >>> aug = iaa.Add((-10, 10), per_channel=True)

    adds a value from the discrete range [-10 .. 10] to all pixels of
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

        if ia.is_single_integer(value):
            ia.do_assert(-255 <= value <= 255, "Expected value to have range [-255, 255], got value %d." % (value,))
            self.value = Deterministic(value)
        elif ia.is_iterable(value):
            ia.do_assert(len(value) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(value),))
            self.value = DiscreteUniform(value[0], value[1])
        elif isinstance(value, StochasticParameter):
            self.value = value
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(value),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            ia.do_assert(0 <= per_channel <= 1.0, "Expected bool, or number in range [0, 1.0] for per_channel, got %s." % (type(per_channel),))
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

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
                samples = self.value.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    # TODO make value range more flexible
                    ia.do_assert(-255 <= sample <= 255)
                    image[..., c] += sample
            else:
                sample = self.value.draw_sample(random_state=rs_image)
                ia.do_assert(-255 <= sample <= 255) # TODO make value range more flexible
                image += sample

            image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.value]

# TODO tests
class AddElementwise(Augmenter):
    """
    Add values to the pixels of images with possibly different values
    for neighbouring pixels.

    While the Add Augmenter adds a constant value per image, this one can
    add different values (sampled per pixel).

    Parameters
    ----------
    value : int or iterable of two ints or StochasticParameter, optional(default=0)
        Value to add to the
        pixels.
            * If an int, then that value will be used for all images.
            * If a tuple (a, b), then values from the discrete range [a .. b]
              will be sampled.
            * If a StochasticParameter, then values will be sampled per pixel
              (and possibly channel) from that parameter.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.AddElementwise(10)

    always adds a value of 10 to all pixels in the image.

    >>> aug = iaa.AddElementwise((-10, 10))

    samples per pixel a value from the discrete range [-10 .. 10] and
    adds that value to the pixel.

    >>> aug = iaa.AddElementwise((-10, 10), per_channel=True)

    samples per pixel *and channel* a value from the discrete
    range [-10 .. 10] ands adds it to the pixel's value. Therefore,
    added values may differ between channels of the same pixel.

    >>> aug = iaa.AddElementwise((-10, 10), per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """

    def __init__(self, value=0, per_channel=False, name=None, deterministic=False, random_state=None):
        super(AddElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_integer(value):
            ia.do_assert(-255 <= value <= 255, "Expected value to have range [-255, 255], got value %d." % (value,))
            self.value = Deterministic(value)
        elif ia.is_iterable(value):
            ia.do_assert(len(value) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(value),))
            self.value = DiscreteUniform(value[0], value[1])
        elif isinstance(value, StochasticParameter):
            self.value = value
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(value),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            ia.do_assert(0 <= per_channel <= 1.0)
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

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
                samples = self.value.draw_samples((height, width, nb_channels), random_state=rs_image)
            else:
                samples = self.value.draw_samples((height, width, 1), random_state=rs_image)
                samples = np.tile(samples, (1, 1, nb_channels))
            after_add = image + samples

            after_add = meta.clip_augmented_image_(after_add, 0, 255) # TODO make value range more flexible
            after_add = meta.restore_augmented_image_dtype_(after_add, input_dtypes[i])

            result[i] = after_add

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.value]

def AdditiveGaussianNoise(loc=0, scale=0, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Add gaussian noise (aka white noise) to images.

    Parameters
    ----------
    loc : int or float or tupel of two ints/floats or StochasticParameter, optional(default=0)
        Mean of the normal distribution that generates the
        noise.
            * If an int or float, exactly that value will be used.
            * If a tuple (a, b), a random value from the range a <= x <= b will
              be sampled per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    scale : int or float or tupel of two ints/floats or StochasticParameter, optional(default=0)
        Standard deviation of the normal distribution that generates the
        noise. If this value gets too close to zero, the image will not be
        changed.
            * If an int or float, exactly that value will be used.
            * If a tuple (a, b), a random value from the range a <= x <= b will
              be sampled per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    per_channel : bool or float, optional(default=False)
        Whether to use the same noise value per pixel for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.GaussianNoise(scale=0.1*255)

    adds gaussian noise from the distribution N(0, 0.1*255) to images.

    >>> aug = iaa.GaussianNoise(scale=(0, 0.1*255))

    adds gaussian noise from the distribution N(0, s) to images,
    where s is sampled per image from the range 0 <= s <= 0.1*255.

    >>> aug = iaa.GaussianNoise(scale=0.1*255, per_channel=True)

    adds gaussian noise from the distribution N(0, 0.1*255) to images,
    where the noise value is different per pixel *and* channel (e.g. a
    different one for red, green and blue channels for the same pixel).

    >>> aug = iaa.GaussianNoise(scale=0.1*255, per_channel=0.5)

    adds gaussian noise from the distribution N(0, 0.1*255) to images,
    where the noise value is sometimes (50 percent of all cases) the same
    per pixel for all channels and sometimes different (other 50 percent).

    """
    if ia.is_single_number(loc):
        loc2 = Deterministic(loc)
    elif ia.is_iterable(loc):
        ia.do_assert(len(loc) == 2, "Expected tuple/list with 2 entries for argument 'loc', got %d entries." % (len(loc),))
        loc2 = Uniform(loc[0], loc[1])
    elif isinstance(loc, StochasticParameter):
        loc2 = loc
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter for argument 'loc'. Got %s." % (type(loc),))

    if ia.is_single_number(scale):
        scale2 = Deterministic(scale)
    elif ia.is_iterable(scale):
        ia.do_assert(len(scale) == 2, "Expected tuple/list with 2 entries for argument 'scale', got %d entries." % (len(scale),))
        scale2 = Uniform(scale[0], scale[1])
    elif isinstance(scale, StochasticParameter):
        scale2 = scale
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter for argument 'scale'. Got %s." % (type(scale),))

    return AddElementwise(Normal(loc=loc2, scale=scale2), per_channel=per_channel, name=name, deterministic=deterministic, random_state=random_state)

# TODO
#class MultiplicativeGaussianNoise(Augmenter):
#    pass

# TODO
#class ReplacingGaussianNoise(Augmenter):
#    pass


class Multiply(Augmenter):
    """
    Multiply all pixels in an image with a specific value.

    This augmenter can be used to make images lighter or darker.

    Parameters
    ----------
    mul : float or tuple of two floats or StochasticParameter, optional(default=1.0)
        The value with which to multiply the pixel values in each
        image.
            * If a float, then that value will always be used.
            * If a tuple (a, b), then a value from the range a <= x <= b will
              be sampled per image and used for all pixels.
            * If a StochasticParameter, then that parameter will be used to
              sample a new value per image.

    per_channel : bool or float, optional(default=False)
        Whether to use the same multiplier per pixel for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Multiply(2.0)

    would multiply all images by a factor of 2, making the images
    significantly brighter.

    >>> aug = iaa.Multiply((0.5, 1.5))

    would multiply images by a random value from the range 0.5 <= x <= 1.5,
    making some images darker and others brighter.

    """

    def __init__(self, mul=1.0, per_channel=False, name=None,
                 deterministic=False, random_state=None):
        super(Multiply, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(mul):
            ia.do_assert(mul >= 0.0, "Expected multiplier to have range [0, inf), got value %.4f." % (mul,))
            self.mul = Deterministic(mul)
        elif ia.is_iterable(mul):
            ia.do_assert(len(mul) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(mul),))
            self.mul = Uniform(mul[0], mul[1])
        elif isinstance(mul, StochasticParameter):
            self.mul = mul
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(mul),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            ia.do_assert(0 <= per_channel <= 1.0)
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

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

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul]


# TODO tests
class MultiplyElementwise(Augmenter):
    """
    Multiply values of pixels with possibly different values
    for neighbouring pixels.

    While the Multiply Augmenter uses a constant multiplier per image,
    this one can use different multipliers per pixel.

    Parameters
    ----------
    mul : float or iterable of two floats or StochasticParameter, optional(default=1.0)
        The value by which to multiply the pixel values in the
        image.
            * If a float, then that value will always be used.
            * If a tuple (a, b), then a value from the range a <= x <= b will
              be sampled per image and pixel.
            * If a StochasticParameter, then that parameter will be used to
              sample a new value per image and pixel.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.MultiplyElementwise(2.0)

    multiply all images by a factor of 2.0, making them significantly
    bighter.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5))

    samples per pixel a value from the range 0.5 <= x <= 1.5 and
    multiplies the pixel with that value.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=True)

    samples per pixel *and channel* a value from the range
    0.5 <= x <= 1.5 ands multiplies the pixel by that value. Therefore,
    added multipliers may differ between channels of the same pixel.

    >>> aug = iaa.AddElementwise((0.5, 1.5), per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """

    def __init__(self, mul=1.0, per_channel=False, name=None, deterministic=False, random_state=None):
        super(MultiplyElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(mul):
            ia.do_assert(mul >= 0.0, "Expected multiplier to have range [0, inf), got value %.4f." % (mul,))
            self.mul = Deterministic(mul)
        elif ia.is_iterable(mul):
            ia.do_assert(len(mul) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(mul),))
            self.mul = Uniform(mul[0], mul[1])
        elif isinstance(mul, StochasticParameter):
            self.mul = mul
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(mul),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            ia.do_assert(0 <= per_channel <= 1.0)
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

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

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul]

def Dropout(p=0, per_channel=False, name=None, deterministic=False,
            random_state=None):
    """
    Augmenter that sets a certain fraction of pixels in images to zero.

    Parameters
    ----------
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        The probability of any pixel being dropped (i.e. set to
        zero).
            * If a float, then that value will be used for all images. A value
              of 1.0 would mean that all pixels will be dropped and 0.0 that
              no pixels would be dropped. A value of 0.05 corresponds to 5
              percent of all pixels dropped.
            * If a tuple (a, b), then a value p will be sampled from the
              range a <= p <= b per image and be used as the pixel's dropout
              probability.
            * If a StochasticParameter, then this parameter will be used to
              determine per pixel whether it should be dropped (sampled value
              of 0) or shouldn't (sampled value of 1).

    per_channel : bool or float, optional(default=False)
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Dropout(0.02)

    drops 2 percent of all pixels.

    >>> aug = iaa.Dropout((0.0, 0.05))

    drops in each image a random fraction of all pixels, where the fraction
    is in the range 0.0 <= x <= 0.05.

    >>> aug = iaa.Dropout(0.02, per_channel=True)

    drops 2 percent of all pixels in a channel-wise fashion, i.e. it is unlikely
    for any pixel to have all channels set to zero (black pixels).

    >>> aug = iaa.Dropout(0.02, per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """
    if ia.is_single_number(p):
        p2 = Binomial(1 - p)
    elif ia.is_iterable(p):
        ia.do_assert(len(p) == 2)
        ia.do_assert(p[0] < p[1])
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        p2 = Binomial(Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, StochasticParameter):
        p2 = p
    else:
        raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))
    return MultiplyElementwise(p2, per_channel=per_channel, name=name, deterministic=deterministic, random_state=random_state)

def CoarseDropout(p=0, size_px=None, size_percent=None,
                  per_channel=False, min_size=4, name=None, deterministic=False,
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
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        The probability of any pixel being dropped (i.e. set to
        zero).
            * If a float, then that value will be used for all pixels. A value
              of 1.0 would mean, that all pixels will be dropped. A value of
              0.0 would lead to no pixels being dropped.
            * If a tuple (a, b), then a value p will be sampled from the
              range a <= p <= b per image and be used as the pixel's dropout
              probability.
            * If a StochasticParameter, then this parameter will be used to
              determine per pixel whether it should be dropped (sampled value
              of 0) or shouldn't (sampled value of 1).

    size_px : int or tuple of two ints or StochasticParameter, optional(default=None)
        The size of the lower resolution image from which to sample the dropout
        mask in absolute pixel dimensions.
            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a 3x3 mask, which is then
              upsampled to HxW, where H is the image size and W the image width.
            * If a tuple (a, b), then two values M, N will be sampled from the
              range [a..b] and the mask will be generated at size MxN, then
              upsampled to HxW.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of two floats or StochasticParameter, optional(default=None)
        The size of the lower resolution image from which to sample the dropout
        mask *in percent* of the input image.
            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from (p*H)x(p*W) and later upsampled
              to HxW.
            * If a tuple (a, b), then two values m, n will be sampled from the
              interval (a, b) and used as the percentages, i.e the mask size
              will be (m*H)x(n*W).
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional(default=4)
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a 1x1 low resolution mask, leading easily
        to the whole image being dropped.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

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
        p2 = Binomial(1 - p)
    elif ia.is_iterable(p):
        ia.do_assert(len(p) == 2)
        ia.do_assert(p[0] < p[1])
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        p2 = Binomial(Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, StochasticParameter):
        p2 = p
    else:
        raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))

    if size_px is not None:
        p3 = FromLowerResolution(other_param=p2, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        p3 = FromLowerResolution(other_param=p2, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    return MultiplyElementwise(p3, per_channel=per_channel, name=name, deterministic=deterministic, random_state=random_state)

class ReplaceElementwise(Augmenter):
    """
    Replace pixels in an image with new values.

    Parameters
    ----------
    mask : number or tuple of two number or StochasticParameter, optional(default=0)
        Mask that indicates the pixels that are supposed to be replaced.
        The mask will be thresholded with 0.5. A value of 1 then indicates a
        pixel that is supposed to be replaced.
            * If this is a number, then that value will be used as the
              probability of being a 1 per pixel.
            * If a tuple (a, b), then the probability will be sampled per image
              from the range a <= x <= b.
            * If a StochasticParameter, then this parameter will be used to
              sample a mask.

    replacement : number or tuple of two number or list of number or StochasticParameter
        The replacement to use at all locations that are marked as `1` in
        the mask.
            * If this is a number, then that value will always be used as the
              replacement.
            * If a tuple (a, b), then the replacement will be sampled pixelwise
              from the range a <= x <= b.
            * If a list of number, then a random value will be picked from
              that list as the replacement per pixel.
            * If a StochasticParameter, then this parameter will be used sample
              pixelwise replacement values.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = ReplaceElementwise(0.05, [0, 255])

    Replace 5 percent of all pixels in each image by either 0 or 255.

    """

    def __init__(self, mask, replacement, per_channel=False, name=None, deterministic=False, random_state=None):
        super(ReplaceElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(mask):
            self.mask = Binomial(mask)
        elif isinstance(mask, tuple):
            ia.do_assert(len(mask) == 2)
            ia.do_assert(0 <= mask[0] <= 1.0)
            ia.do_assert(0 <= mask[1] <= 1.0)
            self.mask = Binomial(Uniform(mask[0], mask[1]))
        elif ia.is_iterable(mask):
            ia.do_assert(all([0 <= pi <= 1.0 for pi in mask]))
            self.mask = iap.Choice(mask)
        elif isinstance(mask, StochasticParameter):
            self.mask = mask
        else:
            raise Exception("Expected mask to be number or tuple of two number or list of number or StochasticParameter, got %s." % (type(mask),))
        #self.mask = iap.handle_continuous_param(mask, "mask", minval=0.0, maxval=1.0)
        self.replacement = iap.handle_continuous_param(replacement, "replacement")

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            ia.do_assert(0 <= per_channel <= 1.0)
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

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

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mask, self.replacement]

def SaltAndPepper(p=0, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Adds salt and pepper noise to an image, i.e. some white-ish and black-ish
    pixels.

    Parameters
    ----------
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        Probability of changing a pixel to salt/pepper
        noise.
            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple (a, b), then a probability will be sampled per image
              from the range a <= x <= b..
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that salt/pepper is to be added
              at that location.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.SaltAndPepper(0.05)

    Replaces 5 percent of all pixels with salt/pepper.

    """

    return ReplaceElementwise(
        mask=p,
        replacement=iap.Beta(0.5, 0.5) * 255,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )

def CoarseSaltAndPepper(p=0, size_px=None, size_percent=None,
                        per_channel=False, min_size=4, name=None,
                        deterministic=False, random_state=None):
    """
    Adds coarse salt and pepper noise to an image, i.e. rectangles that
    contain noisy white-ish and black-ish pixels.

    Parameters
    ----------
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        Probability of changing a pixel to salt/pepper
        noise.
            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple (a, b), then a probability will be sampled per image
              from the range a <= x <= b..
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that salt/pepper is to be added
              at that location.

    size_px : int or tuple of two ints or StochasticParameter, optional(default=None)
        The size of the lower resolution image from which to sample the noise
        mask in absolute pixel dimensions.
            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a 3x3 mask, which is then
              upsampled to HxW, where H is the image size and W the image width.
            * If a tuple (a, b), then two values M, N will be sampled from the
              range [a..b] and the mask will be generated at size MxN, then
              upsampled to HxW.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of two floats or StochasticParameter, optional(default=None)
        The size of the lower resolution image from which to sample the noise
        mask *in percent* of the input image.
            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from (p*H)x(p*W) and later upsampled
              to HxW.
            * If a tuple (a, b), then two values m, n will be sampled from the
              interval (a, b) and used as the percentages, i.e the mask size
              will be (m*H)x(n*W).
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional(default=4)
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a 1x1 low resolution mask, leading easily
        to the whole image being replaced.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1))

    Replaces 5 percent of all pixels with salt/pepper in an image that has
    1 to 10 percent of the input image size, then upscales the results
    to the input image size, leading to large rectangular areas being replaced.

    """

    if ia.is_single_number(p):
        mask = Binomial(p)
    elif isinstance(p, tuple):
        ia.do_assert(len(p) == 2)
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        mask = Binomial(Uniform(p[0], p[1]))
    elif ia.is_iterable(p):
        ia.do_assert(all([0 <= pi <= 1.0 for pi in p]))
        mask = iap.Choice(p)
    elif isinstance(p, StochasticParameter):
        mask = p
    else:
        raise Exception("Expected p to be number or tuple of two number or list of number or StochasticParameter, got %s." % (type(p),))

    if size_px is not None:
        mask_low = FromLowerResolution(other_param=mask, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        mask_low = FromLowerResolution(other_param=mask, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    replacement = iap.Beta(0.5, 0.5) * 255

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
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        Probability of changing a pixel to salt
        noise.
            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple (a, b), then a probability will be sampled per image
              from the range a <= x <= b..
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that salt is to be added
              at that location.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

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
    return ReplaceElementwise(
        mask=p,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )

def CoarseSalt(p=0, size_px=None, size_percent=None,
               per_channel=False, min_size=4, name=None,
               deterministic=False, random_state=None):
    """
    Adds coarse salt noise to an image, i.e. rectangles containing noisy
    white-ish pixels.

    Parameters
    ----------
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        Probability of changing a pixel to salt
        noise.
            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple (a, b), then a probability will be sampled per image
              from the range a <= x <= b..
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that salt is to be added
              at that location.

    size_px : int or tuple of two ints or StochasticParameter, optional(default=None)
        The size of the lower resolution image from which to sample the noise
        mask in absolute pixel dimensions.
            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a 3x3 mask, which is then
              upsampled to HxW, where H is the image size and W the image width.
            * If a tuple (a, b), then two values M, N will be sampled from the
              range [a..b] and the mask will be generated at size MxN, then
              upsampled to HxW.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of two floats or StochasticParameter, optional(default=None)
        The size of the lower resolution image from which to sample the noise
        mask *in percent* of the input image.
            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from (p*H)x(p*W) and later upsampled
              to HxW.
            * If a tuple (a, b), then two values m, n will be sampled from the
              interval (a, b) and used as the percentages, i.e the mask size
              will be (m*H)x(n*W).
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional(default=4)
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a 1x1 low resolution mask, leading easily
        to the whole image being replaced.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1))

    Replaces 5 percent of all pixels with salt in an image that has
    1 to 10 percent of the input image size, then upscales the results
    to the input image size, leading to large rectangular areas being replaced.

    """

    if ia.is_single_number(p):
        mask = Binomial(p)
    elif isinstance(p, tuple):
        ia.do_assert(len(p) == 2)
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        mask = Binomial(Uniform(p[0], p[1]))
    elif ia.is_iterable(p):
        ia.do_assert(all([0 <= pi <= 1.0 for pi in p]))
        mask = iap.Choice(p)
    elif isinstance(p, StochasticParameter):
        mask = p
    else:
        raise Exception("Expected p to be number or tuple of two number or list of number or StochasticParameter, got %s." % (type(p),))

    if size_px is not None:
        mask_low = FromLowerResolution(other_param=mask, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        mask_low = FromLowerResolution(other_param=mask, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    replacement01 = iap.ForceSign(
        iap.Beta(0.5, 0.5) - 0.5,
        positive=True,
        mode="invert"
    ) + 0.5
    replacement = replacement01 * 255

    return ReplaceElementwise(
        mask=mask_low,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )

def Pepper(p=0, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Adds pepper noise to an image, i.e. black-ish pixels.
    This is similar to dropout, but slower and the black pixels are not
    uniformly black.

    Parameters
    ----------
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        Probability of changing a pixel to pepper
        noise.
            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple (a, b), then a probability will be sampled per image
              from the range a <= x <= b..
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that pepper is to be added
              at that location.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

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
    return ReplaceElementwise(
        mask=p,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )

def CoarsePepper(p=0, size_px=None, size_percent=None,
                 per_channel=False, min_size=4, name=None,
                 deterministic=False, random_state=None):
    """
    Adds coarse pepper noise to an image, i.e. rectangles that contain
    noisy black-ish pixels.

    Parameters
    ----------
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        Probability of changing a pixel to pepper
        noise.
            * If a float, then that value will be used for all images as the
              probability.
            * If a tuple (a, b), then a probability will be sampled per image
              from the range a <= x <= b..
            * If a StochasticParameter, then this parameter will be used as
              the *mask*, i.e. it is expected to contain values between
              0.0 and 1.0, where 1.0 means that pepper is to be added
              at that location.

    size_px : int or tuple of two ints or StochasticParameter, optional(default=None)
        The size of the lower resolution image from which to sample the noise
        mask in absolute pixel dimensions.
            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a 3x3 mask, which is then
              upsampled to HxW, where H is the image size and W the image width.
            * If a tuple (a, b), then two values M, N will be sampled from the
              range [a..b] and the mask will be generated at size MxN, then
              upsampled to HxW.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of two floats or StochasticParameter, optional(default=None)
        The size of the lower resolution image from which to sample the noise
        mask *in percent* of the input image.
            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from (p*H)x(p*W) and later upsampled
              to HxW.
            * If a tuple (a, b), then two values m, n will be sampled from the
              interval (a, b) and used as the percentages, i.e the mask size
              will be (m*H)x(n*W).
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional(default=4)
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a 1x1 low resolution mask, leading easily
        to the whole image being replaced.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1))

    Replaces 5 percent of all pixels with pepper in an image that has
    1 to 10 percent of the input image size, then upscales the results
    to the input image size, leading to large rectangular areas being replaced.

    """

    if ia.is_single_number(p):
        mask = Binomial(p)
    elif isinstance(p, tuple):
        ia.do_assert(len(p) == 2)
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        mask = Binomial(Uniform(p[0], p[1]))
    elif ia.is_iterable(p):
        ia.do_assert(all([0 <= pi <= 1.0 for pi in p]))
        mask = iap.Choice(p)
    elif isinstance(p, StochasticParameter):
        mask = p
    else:
        raise Exception("Expected p to be number or tuple of two number or list of number or StochasticParameter, got %s." % (type(p),))

    if size_px is not None:
        mask_low = FromLowerResolution(other_param=mask, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        mask_low = FromLowerResolution(other_param=mask, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    replacement01 = iap.ForceSign(
        iap.Beta(0.5, 0.5) - 0.5,
        positive=False,
        mode="invert"
    ) + 0.5
    replacement = replacement01 * 255

    return ReplaceElementwise(
        mask=mask_low,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )

# TODO tests
class Invert(Augmenter):
    """
    Augmenter that inverts all values in images.

    For the standard value range of 0-255 it converts 0 to 255, 255 to 0
    and 10 to (255-10)=245.

    Let M be the maximum value possible, m the minimum value possible,
    v a value. Then the distance of v to m is d=abs(v-m) and the new value
    is given by v'=M-d.

    Parameters
    ----------
    p : float or StochasticParameter, optional(default=0)
        The probability of an image to be
        inverted.
            * If a float, then that probability will be used for all images.
            * If a StochasticParameter, then that parameter will queried per
              image and is expected too return values in the range [0.0, 1.0],
              where values >0.5 mean that the image/channel is supposed to be
              inverted.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_value : int or float, optional(default=0)
        Minimum of the range of possible pixel values. For uint8 (0-255)
        images, this should be 0.

    max_value : int or float, optional(default=255)
        Maximum of the range of possible pixel values. For uint8 (0-255)
        images, this should be 255.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

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

    def __init__(self, p=0, per_channel=False, min_value=0, max_value=255, name=None,
                 deterministic=False, random_state=None):
        super(Invert, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be int or float or StochasticParameter, got %s." % (type(p),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            ia.do_assert(0 <= per_channel <= 1.0)
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

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

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.p, self.per_channel, self.min_value, self.max_value]

# TODO tests
class ContrastNormalization(Augmenter):
    """
    Augmenter that changes the contrast of images.

    Parameters
    ----------
    alpha : float or tuple of two floats or StochasticParameter, optional(default=1.0)
        Strength of the contrast normalization. Higher values than 1.0
        lead to higher contrast, lower values decrease the contrast.
            * If a float, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled per image from
              the range a <= x <= b and be used as the alpha value.
            * If a StochasticParameter, then this parameter will be used to
              sample the alpha value per image.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

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

        if ia.is_single_number(alpha):
            ia.do_assert(alpha >= 0.0, "Expected alpha to have range (0, inf), got value %.4f." % (alpha,))
            self.alpha = Deterministic(alpha)
        elif ia.is_iterable(alpha):
            ia.do_assert(len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(alpha),))
            self.alpha = Uniform(alpha[0], alpha[1])
        elif isinstance(alpha, StochasticParameter):
            self.alpha = alpha
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            ia.do_assert(0 <= per_channel <= 1.0)
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

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

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.alpha]
