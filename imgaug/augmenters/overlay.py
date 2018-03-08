"""
Augmenters that overlay two images with each other.

Do not import directly from this file, as the categorization is not final.
Use instead
    `from imgaug import augmenters as iaa`
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
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, Binomial, Uniform
from .. import parameters as iap
import numpy as np
import six.moves as sm

from .meta import Augmenter, Sequential

# TODO tests
class Alpha(Augmenter): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Augmenter to overlay two image sources with each other using an
    alpha/transparency value.

    The image sources can be imagined as branches.
    If a source is not given, it is automatically the same as the input.
    Let A be the first branch and B be the second branch.
    Then the result images are defined as
        factor * A + (1-factor) * B,
    where `factor` is an overlay factor.

    For keypoint augmentation this augmenter will pick the keypoints either
    from the first or the second branch. The first one is picked if
    `factor >= 0.5` is true (per image). It is recommended to *not* use
    augmenters that change keypoint positions with this class.

    Parameters
    ----------
    factor : int or float or iterable of two floats or StochasticParameter, optional(default=0)
        Weighting of the results of the first branch. Values close to 0 mean
        that the results from the second branch (see parameter `second`)
        make up most of the final image.
            * If float, then that value will be used for all images.
            * If tuple (a, b), then a random value from range a <= x <= b will
              be sampled per image.
            * If StochasticParameter, then that parameter will be used to
              sample a value per image.

    first : None or Augmenter or iterable of Augmenter, optional(default=None)
        Augmenter(s) that make up the first of the two
        branches.
            * If None, then the input images will be reused as the output
              of the first branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    second : None or Augmenter or iterable of Augmenter, optional(default=None)
        Augmenter(s) that make up the second of the two
        branches.
            * If None, then the input images will be reused as the output
              of the second branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    per_channel : bool or float, optional(default=False)
        Whether to use the same factor for all channels (False)
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
    >>> aug = iaa.Alpha(0.5, iaa.Grayscale(1.0))

    Converts each image to grayscale and overlays it by 50 percent with the
    original image, thereby removing about 50 percent of all color. This
    is equivalent to iaa.Grayscale(0.5).

    >>> aug = iaa.Alpha((0.0, 1.0), iaa.Grayscale(1.0))

    Converts each image to grayscale and overlays it by a random percentage
    (sampled per image) with the original image, thereby removing a random
    percentage of all colors. This is equivalent to iaa.Grayscale((0.0, 1.0)).

    >>> aug = iaa.Alpha((0.0, 1.0), iaa.Affine(rotate=(-20, 20)), per_channel=0.5)

    Rotates each image by a random degree from the range [-20, 20]. Then
    overlays that new image with the original one by a random factor from the
    range [0.0, 1.0]. In 50 percent of all cases, the overlay happens
    channel-wise and the factor is sampled independently per channel. As a
    result, e.g. the red channel may look visible rotated (factor near 1.0),
    while the green and blue channels may not look rotated (factors near 0.0).
    NOTE: It is not recommended to use Alpha with augmenters that change the
    positions of pixels if you *also* want to augment keypoints, as it is
    unclear which of the two keypoint results (first or second branch) should
    be used as the final result.

    >>> aug = iaa.Alpha((0.0, 1.0), first=iaa.Add(10), second=iaa.Multiply(0.8))

    (A) Adds 10 to each image and (B) multiplies each image by 0.8. Then per
    image an overlay factor is sampled from the range [0.0, 1.0]. If it is
    close to 1.0, the results from (A) are mostly used, otherwise the ones
    from (B). This is equivalent to
    `iaa.Sequential([iaa.Multiply(0.8), iaa.Alpha((0.0, 1.0), iaa.Add(10))])`.

    >>> aug = iaa.Alpha(iap.Choice([0.25, 0.75]), iaa.MedianBlur((3, 7)))

    Applies a random median blur to each image and overlays the result with
    the original image by either 25 or 75 percent strength.

    """

    def __init__(self, factor=0, first=None, second=None, per_channel=False,
                 name=None, deterministic=False, random_state=None):
        super(Alpha, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(factor):
            ia.do_assert(0.0 <= factor <= 1.0, "Expected factor to have range [0, 1.0], got value %.2f." % (factor,))
            self.factor = Deterministic(factor)
        elif ia.is_iterable(factor):
            ia.do_assert(len(factor) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(factor),))
            self.factor = Uniform(factor[0], factor[1])
        elif isinstance(factor, StochasticParameter):
            self.factor = factor
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(factor),))

        ia.do_assert(first is not None or second is not None, "Expected 'first' and/or 'second' to not be None (i.e. at least one Augmenter), but got two None values.")

        if first is None:
            self.first = None
        elif ia.is_iterable(first):
            if isinstance(first, Augmenter):
                self.first = first
            else:
                self.first = Sequential(first, name="%s-first" % (self.name,))
        elif isinstance(first, Augmenter):
            self.first = Sequential([first], name="%s-first" % (self.name,))
        else:
            raise Exception("Expected 'first' to be either None or Augmenter or iterable of Augmenter, got %s." % (type(first),))

        if second is None:
            self.second = None
        elif ia.is_iterable(second):
            if isinstance(second, Augmenter):
                self.second = second
            else:
                self.second = Sequential(second, name="%s-second" % (self.name,))
        elif isinstance(second, Augmenter):
            self.second = Sequential([second], name="%s-second" % (self.name,))
        else:
            raise Exception("Expected 'second' to be either None or Augmenter or iterable of Augmenter, got %s." % (type(second),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            ia.do_assert(0 <= per_channel <= 1.0)
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

        self.epsilon = 0.01

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))

        if hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
            if self.first is None:
                images_first = images
            else:
                images_first = self.first.augment_images(
                    images=images,
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is None:
                images_second = images
            else:
                images_second = self.second.augment_images(
                    images=images,
                    parents=parents + [self],
                    hooks=hooks
                )
        else:
            images_first = images
            images_second = images

        for i in sm.xrange(nb_images):
            image = images[i]
            image_first = images_first[i]
            image_second = images_second[i]
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            input_dtype = image.dtype
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.factor.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    ia.do_assert(0 <= sample <= 1.0)
                    # if the value is nearly 1.0 or 0.0 skip the computation
                    # and just use only the first/second image
                    if sample >= 1.0 - self.epsilon:
                        image[..., c] = image_first[..., c]
                    elif sample <= 0.0 + self.epsilon:
                        image[..., c] = image_second[..., c]
                    else:
                        image[..., c] = sample * image_first[..., c] + (1 - sample) * image_second[..., c]
                # TODO change this to meta.clip_* and meta.restore_*
                np.clip(image, 0, 255, out=image)
                result[i] = image.astype(input_dtype)
            else:
                sample = self.factor.draw_sample(random_state=rs_image)
                ia.do_assert(0 <= sample <= 1.0)
                # if the value is nearly 1.0 or 0.0 skip the computation
                # and just use only the first/second image
                if sample >= 1.0 - self.epsilon:
                    image = image_first
                elif sample <= 0.0 + self.epsilon:
                    image = image_second
                else:
                    image = sample * image_first + (1 - sample) * image_second
                # TODO change this to meta.clip_* and meta.restore_*
                np.clip(image, 0, 255, out=image)
                result[i] = image.astype(input_dtype)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = keypoints_on_images
        nb_images = len(keypoints_on_images)
        seeds = random_state.randint(0, 10**6, (nb_images,))

        if hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents, default=True):
            if self.first is None:
                kps_ois_first = keypoints_on_images
            else:
                kps_ois_first = self.first.augment_keypoints(
                    keypoints_on_images=keypoints_on_images,
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is None:
                kps_ois_second = keypoints_on_images
            else:
                kps_ois_second = self.second.augment_keypoints(
                    keypoints_on_images=keypoints_on_images,
                    parents=parents + [self],
                    hooks=hooks
                )
        else:
            kps_ois_first = keypoints_on_images
            kps_ois_second = keypoints_on_images

        for i in sm.xrange(nb_images):
            kps_oi_first = kps_ois_first[i]
            kps_oi_second = kps_ois_second[i]
            rs_image = ia.new_random_state(seeds[i])
            # keypoint augmentation also works channel-wise, even though
            # keypoints do not have channels, in order to keep the random
            # values properly synchronized with the image augmentation
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = keypoints_on_images[i].shape[2]
                samples = self.factor.draw_samples((nb_channels,), random_state=rs_image)
                sample = np.average(samples)
            else:
                sample = self.factor.draw_sample(random_state=rs_image)
                ia.do_assert(0 <= sample <= 1.0)

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

    def _to_deterministic(self):
        aug = self.copy()
        aug.first = aug.first.to_deterministic() if aug.first is not None else None
        aug.second = aug.second.to_deterministic() if aug.second is not None else None
        aug.deterministic = True
        aug.random_state = ia.new_random_state()
        return aug

    def get_parameters(self):
        return [self.factor, self.first, self.second, self.per_channel]

    def get_children_lists(self):
        result = []
        if self.first is not None:
            result.append(self.first)
        if self.second is not None:
            result.append(self.second)
        return result

class AlphaElementwise(Alpha): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Augmenter to overlay two image sources with each other using pixelwise
    alpha values.

    This is the same as `Alpha`, except that the transparency factor is
    sampled per pixel instead of once per image (or a few times per image, if
    per_channel is True).

    See `Alpha` for more description.

    Parameters
    ----------
    factor : float or iterable of two floats or StochasticParameter, optional(default=0)
        Weighting of the results of the first branch. Values close to 0 mean
        that the results from the second branch (see parameter `second`)
        make up most of the final image.
            * If float, then that value will be used for all images.
            * If tuple (a, b), then a random value from range a <= x <= b will
              be sampled per image.
            * If StochasticParameter, then that parameter will be used to
              sample a value per image.

    first : None or Augmenter or iterable of Augmenter, optional(default=None)
        Augmenter(s) that make up the first of the two
        branches.
            * If None, then the input images will be reused as the output
              of the first branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    second : None or Augmenter or iterable of Augmenter, optional(default=None)
        Augmenter(s) that make up the second of the two
        branches.
            * If None, then the input images will be reused as the output
              of the second branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    per_channel : bool or float, optional(default=False)
        Whether to use the same factor for all channels (False)
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
    >>> aug = iaa.AlphaElementwise(0.5, iaa.Grayscale(1.0))

    Converts each image to grayscale and overlays it by 50 percent with the
    original image, thereby removing about 50 percent of all color. This
    is equivalent to iaa.Grayscale(0.5). This is also equivalent to
    iaa.Alpha(0.5, iaa.Grayscale(1.0)), as the transparency factor is the
    same for all pixels.

    >>> aug = iaa.AlphaElementwise((0, 1.0), iaa.Grayscale(1.0))

    Converts each image to grayscale and overlays it by a random percentage
    (sampled per pixel) with the original image, thereby removing a random
    percentage of all colors per pixel.

    >>> aug = iaa.AlphaElementwise((0.0, 1.0), iaa.Affine(rotate=(-20, 20)), per_channel=0.5)

    Rotates each image by a random degree from the range [-20, 20]. Then
    overlays that new image with the original one by a random factor from the
    range [0.0, 1.0], sampled per pixel. In 50 percent of all cases, the
    overlay happens channel-wise and the factor is sampled independently per
    channel. As a result, e.g. the red channel may look visible rotated (factor
    near 1.0), while the green and blue channels may not look rotated (factors
    near 0.0). NOTE: It is not recommended to use Alpha with augmenters that
    change the positions of pixels if you *also* want to augment keypoints, as
    it is unclear which of the two keypoint results (first or second branch)
    should be used as the final result.

    >>> aug = iaa.AlphaElementwise((0.0, 1.0), first=iaa.Add(10), second=iaa.Multiply(0.8))

    (A) Adds 10 to each image and (B) multiplies each image by 0.8. Then per
    pixel an overlay factor is sampled from the range [0.0, 1.0]. If it is
    close to 1.0, the results from (A) are mostly used, otherwise the ones
    from (B).

    >>> aug = iaa.AlphaElementwise(iap.Choice([0.25, 0.75]), iaa.MedianBlur((3, 7)))

    Applies a random median blur to each image and overlays the result with
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

        if hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
            if self.first is None:
                images_first = images
            else:
                images_first = self.first.augment_images(
                    images=images,
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is None:
                images_second = images
            else:
                images_second = self.second.augment_images(
                    images=images,
                    parents=parents + [self],
                    hooks=hooks
                )
        else:
            images_first = images
            images_second = images

        for i in sm.xrange(nb_images):
            image = images[i]
            h, w, nb_channels = image.shape[0:3]
            image_first = images_first[i]
            image_second = images_second[i]
            per_channel = self.per_channel.draw_sample(random_state=ia.new_random_state(seeds[i]))
            input_dtype = image.dtype
            if per_channel == 1:
                for c in sm.xrange(nb_channels):
                    samples_c = self.factor.draw_samples((h, w), random_state=ia.new_random_state(seeds[i]+1+c))
                    ia.do_assert(0 <= samples_c.item(0) <= 1.0) # validate only first value
                    image[..., c] = samples_c * image_first[..., c] + (1.0 - samples_c) * image_second[..., c]
                # TODO change this to meta.clip_* and meta.restore_*
                np.clip(image, 0, 255, out=image)
                result[i] = image.astype(input_dtype)
            else:
                samples = self.factor.draw_samples((h, w), random_state=ia.new_random_state(seeds[i]))
                samples = np.tile(samples[..., np.newaxis], (1, 1, nb_channels))
                ia.do_assert(0.0 <= samples.item(0) <= 1.0)

                image = samples * image_first + (1.0 - samples) * image_second
                # TODO change this to meta.clip_* and meta.restore_*
                np.clip(image, 0, 255, out=image)
                result[i] = image.astype(input_dtype)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = keypoints_on_images
        nb_images = len(keypoints_on_images)
        seeds = random_state.randint(0, 10**6, (nb_images,))

        if hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents, default=True):
            if self.first is None:
                kps_ois_first = keypoints_on_images
            else:
                kps_ois_first = self.first.augment_keypoints(
                    keypoints_on_images=keypoints_on_images,
                    parents=parents + [self],
                    hooks=hooks
                )

            if self.second is None:
                kps_ois_second = keypoints_on_images
            else:
                kps_ois_second = self.second.augment_keypoints(
                    keypoints_on_images=keypoints_on_images,
                    parents=parents + [self],
                    hooks=hooks
                )
        else:
            kps_ois_first = keypoints_on_images
            kps_ois_second = keypoints_on_images

        for i in sm.xrange(nb_images):
            kps_oi_first = kps_ois_first[i]
            kps_oi_second = kps_ois_second[i]
            #rs_image = ia.new_random_state(seeds[i])
            ia.do_assert(
                len(kps_oi_first.shape) == 3,
                "Keypoint augmentation in AlphaElementwise requires " \
                "KeypointsOnImage.shape to have channel information (i.e. " \
                "tuple with 3 entries), which you did not provide (input " \
                "shape: %s). The channels must match the corresponding " \
                "image channels." % (kps_oi_first.shape,)
            )
            h, w, nb_channels = kps_oi_first.shape[0:3]

            # keypoint augmentation also works channel-wise, even though
            # keypoints do not have channels, in order to keep the random
            # values properly synchronized with the image augmentation
            per_channel = self.per_channel.draw_sample(random_state=ia.new_random_state(seeds[i]))
            if per_channel == 1:
                #samples = self.factor.draw_samples((h, w, nb_channels,), random_state=rs_image)
                samples = np.zeros((h, w, nb_channels), dtype=np.float32)
                for c in nb_channels:
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

def SimplexNoiseAlpha(first=None, second=None, per_channel=False,
                      size_px_max=(2, 16), upscale_method=None,
                      iterations=(1, 3), aggregation_method="max",
                      sigmoid=True, sigmoid_thresh=None,
                      name=None, deterministic=False, random_state=None):
    """
    Augmenter to overlay two image sources with each other using alpha values
    that follow noisy patterns.

    The alpha masks are sampled using a simplex noise method, roughly creating
    connected blobs of 1s surrounded by 0s. If nearest neighbour upsampling
    is used, these blobs can be rectangular with sharp edges.

    Parameters
    ----------
    first : None or Augmenter or iterable of Augmenter, optional(default=None)
        Augmenter(s) that make up the first of the two
        branches.
            * If None, then the input images will be reused as the output
              of the first branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    second : None or Augmenter or iterable of Augmenter, optional(default=None)
        Augmenter(s) that make up the second of the two
        branches.
            * If None, then the input images will be reused as the output
              of the second branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    per_channel : bool or float, optional(default=False)
        Whether to use the same factor for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    size_px_max : int or tuple of ints or list of ints or StochasticParameter, optional(default=(2, 16))
        The simplex noise is always generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).
            * If int, then that number will be used as the size for all
              iterations.
            * If tuple of two ints (a, b), then a value will be sampled
              per iteration from the discrete range [a..b].
            * If a list of ints, then a value will be picked per iteration at
              random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or ia.ALL or string or list of string or StochasticParameter, optional(default=None)
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.
            * If None, then either 'nearest' or 'linear' or 'cubic' is picked.
              Most weight is put on linear, followed by cubic.
            * If ia.ALL, then either 'nearest' or 'linear' or 'area' or 'cubic'
              is picked per iteration (all same probability).
            * If string, then that value will be used as the method (must be
              'nearest' or 'linear' or 'area' or 'cubic').
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If StochasticParameter, then a random value will be sampled
              from that parameter per iteration.

    iterations : int or tuple of ints or list of ints or StochasticParameter, optional(default=(1, 3))
        How often to repeat the simplex noise generation process per
        image.
            * If int, then that number will be used as the iterations for all
              images.
            * If tuple of two ints (a, b), then a value will be sampled
              per image from the discrete range [a..b].
            * If a list of ints, then a value will be picked per image at
              random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per image.

    aggregation_method : ia.ALL or string or list of string or StochasticParameter, optional(default="max")
        The noise maps (from each iteration) are combined to one noise map
        using an aggregation process. This parameter defines the method used
        for that process. Valid methods are 'min', 'max' or 'avg',
        where 'min' combines the noise maps by taking the (elementwise) minimum
        over all iteration's results, 'max' the (elementwise) maximum and
        'avg' the (elemtwise) average.
            * If ia.ALL, then a random value will be picked per image from the
              valid ones.
            * If a string, then that value will always be used as the method.
            * If a list of string, then a random value will be picked from
              that list per image.
            * If a StochasticParameter, then a random value will be sampled
              from that paramter per image.

    sigmoid : bool or number, optional(default=True)
        Whether to apply a sigmoid function to the final noise maps, resulting
        in maps that have more extreme values (close to 0.0 or 1.0).
            * If bool, then a sigmoid will always (True) or never (False) be
              applied.
            * If a number p with 0<=p<=1, then a sigmoid will be applied to
              p percent of all final noise maps.

    sigmoid_thresh : None or number or tuple of number or StochasticParameter, optional(default=None)
        Threshold of the sigmoid, when applied. Thresholds above zero
        (e.g. 5.0) will move the saddle point towards the right, leading to
        more values close to 0.0.
            * If None, then Normal(0, 5.0) will be used.
            * If number, then that threshold will be used for all images.
            * If tuple of two numbers (a, b), then a random value will
              be sampled per image from the range [a, b].
            * If StochasticParameter, then a random value will be sampled from
              that parameter per image.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.SimplexNoiseAlpha(iaa.EdgeDetect(1.0))

    Detects per image all edges, marks them in a black and white image and
    then overlays the result with the original image using simplex noise masks.

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

    if sigmoid != False or (ia.is_single_number(sigmoid) and sigmoid <= 0.01):
        noise = iap.Sigmoid.create_for_noise(
            noise,
            threshold=sigmoid_thresh if sigmoid_thresh is not None else sigmoid_thresh_default,
            activated=sigmoid
        )

    return AlphaElementwise(
        factor=noise, first=first, second=second, per_channel=per_channel,
        name=name, deterministic=deterministic, random_state=random_state
    )

def FrequencyNoiseAlpha(exponent=(-4, 4),
                        first=None, second=None, per_channel=False,
                        size_px_max=(4, 16), upscale_method=None,
                        iterations=(1, 3), aggregation_method=["avg", "max"], # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
                        sigmoid=0.5, sigmoid_thresh=None,
                        name=None, deterministic=False, random_state=None):
    """
    Augmenter to overlay two image sources with each other using alpha values
    that follow noisy patterns.

    The alpha masks are sampled using frequency noise of varying scales,
    which can sometimes create large connected blobs of 1s surrounded by 0s
    and other times results in smaller patterns. If nearest neighbour
    upsampling is used, these blobs can be rectangular with sharp edges.

    Parameters
    ----------
    exponent : number or tuple of numbers of list of numbers or StochasticParameter, optional(default=(-4, 4))
        Exponent to use when scaling in the frequency domain.
        Sane values are in the range -4 (large blobs) to 4 (small patterns).
        To generate cloud-like structures, use roughly -2.
            * If number, then that number will be used as the exponent for all
              iterations.
            * If tuple of two numbers (a, b), then a value will be sampled
              per iteration from the range [a, b].
            * If a list of numbers, then a value will be picked per iteration
              at random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per iteration.

    first : None or Augmenter or iterable of Augmenter, optional(default=None)
        Augmenter(s) that make up the first of the two
        branches.
            * If None, then the input images will be reused as the output
              of the first branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    second : None or Augmenter or iterable of Augmenter, optional(default=None)
        Augmenter(s) that make up the second of the two
        branches.
            * If None, then the input images will be reused as the output
              of the second branch.
            * If Augmenter, then that augmenter will be used as the branch.
            * If iterable of Augmenter, then that iterable will be converted
              into a Sequential and used as the augmenter.

    per_channel : bool or float, optional(default=False)
        Whether to use the same factor for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    size_px_max : int or tuple of ints or list of ints or StochasticParameter, optional(default=(4, 16))
        The noise is generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).
            * If int, then that number will be used as the size for all
              iterations.
            * If tuple of two ints (a, b), then a value will be sampled
              per iteration from the discrete range [a..b].
            * If a list of ints, then a value will be picked per iteration at
              random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or ia.ALL or string or list of string or StochasticParameter, optional(default=None)
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.
            * If None, then either 'nearest' or 'linear' or 'cubic' is picked.
              Most weight is put on linear, followed by cubic.
            * If ia.ALL, then either 'nearest' or 'linear' or 'area' or 'cubic'
              is picked per iteration (all same probability).
            * If string, then that value will be used as the method (must be
              'nearest' or 'linear' or 'area' or 'cubic').
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If StochasticParameter, then a random value will be sampled
              from that parameter per iteration.

    iterations : int or tuple of ints or list of ints or StochasticParameter, optional(default=(1, 3))
        How often to repeat the simplex noise generation process per
        image.
            * If int, then that number will be used as the iterations for all
              images.
            * If tuple of two ints (a, b), then a value will be sampled
              per image from the discrete range [a..b].
            * If a list of ints, then a value will be picked per image at
              random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per image.

    aggregation_method : ia.ALL or string or list of string or StochasticParameter, optional(default=["avg", "max"])
        The noise maps (from each iteration) are combined to one noise map
        using an aggregation process. This parameter defines the method used
        for that process. Valid methods are 'min', 'max' or 'avg',
        where 'min' combines the noise maps by taking the (elementwise) minimum
        over all iteration's results, 'max' the (elementwise) maximum and
        'avg' the (elemtwise) average.
            * If ia.ALL, then a random value will be picked per image from the
              valid ones.
            * If a string, then that value will always be used as the method.
            * If a list of string, then a random value will be picked from
              that list per image.
            * If a StochasticParameter, then a random value will be sampled
              from that paramter per image.

    sigmoid : bool or number, optional(default=0.5)
        Whether to apply a sigmoid function to the final noise maps, resulting
        in maps that have more extreme values (close to 0.0 or 1.0).
            * If bool, then a sigmoid will always (True) or never (False) be
              applied.
            * If a number p with 0<=p<=1, then a sigmoid will be applied to
              p percent of all final noise maps.

    sigmoid_thresh : None or number or tuple of number or StochasticParameter, optional(default=None)
        Threshold of the sigmoid, when applied. Thresholds above zero
        (e.g. 5.0) will move the saddle point towards the right, leading to
        more values close to 0.0.
            * If None, then Normal(0, 5.0) will be used.
            * If number, then that threshold will be used for all images.
            * If tuple of two numbers (a, b), then a random value will
              be sampled per image from the range [a, b].
            * If StochasticParameter, then a random value will be sampled from
              that parameter per image.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(1.0))

    Detects per image all edges, marks them in a black and white image and
    then overlays the result with the original image using frequency noise
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

    if sigmoid != False or (ia.is_single_number(sigmoid) and sigmoid <= 0.01):
        noise = iap.Sigmoid.create_for_noise(
            noise,
            threshold=sigmoid_thresh if sigmoid_thresh is not None else sigmoid_thresh_default,
            activated=sigmoid
        )

    return AlphaElementwise(
        factor=noise, first=first, second=second, per_channel=per_channel,
        name=name, deterministic=deterministic, random_state=random_state
    )
