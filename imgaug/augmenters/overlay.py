"""
Augmenters that overlay two images with each other.

Do not import directly from this file, as the categorization is not final.
Use instead
    from imgaug import augmenters as iaa
and then e.g.
    seq = iaa.Sequential([
        iaa.Alpha(0.5, iaa.Add((-5, 5)))
    ])

List of augmenters:
    Alpha
"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, Binomial, Choice, DiscreteUniform, Normal, Uniform, FromLowerResolution
from .. import parameters as iap
import numpy as np
import cv2
import six
import six.moves as sm
import warnings

from .meta import Augmenter

# TODO tests
class Alpha(Augmenter):
    """
    Augmenter to overlay two image sources with each other using an alpha value.

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
    >>> aug = iaa.Alpha(0.5, iaa.Grayscale(1.0))

    Converts the image to grayscale and overlays it by 50 percent with the
    original image, thereby removing about 50 percent of all color. This
    is equivalent to iaa.Grayscale(0.5).

    >>> aug = iaa.Alpha((0, 1.0), iaa.Grayscale(1.0))

    Converts the image to grayscale and overlays it by a random percentage
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

    """

    def __init__(self, factor=0, first=None, second=None, per_channel=False,
                 name=None, deterministic=False, random_state=None):
        super(Alpha, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_float(factor):
            assert 0 <= factor <= 1.0, "Expected factor to have range [0, 1.0], got value %.2f." % (factor,)
            self.factor = Deterministic(factor)
        elif ia.is_iterable(factor):
            assert len(factor) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(factor),)
            self.factor = Uniform(factor[0], factor[1])
        elif isinstance(factor, StochasticParameter):
            self.factor = factor
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(factor),))

        assert first is not None or second is not None, "Expected 'first' and/or 'second' to not be None (i.e. at least one Augmenter), but got two None values."

        if first is None:
            self.first = None
        elif isinstance(first, Augmenter):
            self.first = first
        elif ia.is_iterable(first):
            self.first = iaa.Sequential(first, name="%s-first" % (self.name,))
        else:
            raise Exception("Expected 'first' to be either None or Augmenter or iterable of Augmenter, got %s." % (type(first),))

        if second is None:
            self.second = None
        elif isinstance(second, Augmenter):
            self.second = second
        elif ia.is_iterable(second):
            self.second = iaa.Sequential(second, name="%s-second" % (self.name,))
        else:
            raise Exception("Expected 'second' to be either None or Augmenter or iterable of Augmenter, got %s." % (type(second),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            assert 0 <= per_channel <= 1.0
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
                    assert 0 <= sample <= 1.0
                    # if the value is nearly 1.0 or 0.0 skip the computation
                    # and just use only the first/second image
                    if sample >= 1.0 - self.epsilon:
                        image[..., c] = image_first[..., c]
                    elif sample <= 0.0 + self.epsilon:
                        image[..., c] = image_second[..., c]
                    else:
                        image[..., c] = sample * image_first[..., c] + (1 - sample) * image_second[..., c]
                np.clip(image, 0, 255, out=image)
                result[i] = image.astype(input_dtype)
            else:
                sample = self.factor.draw_sample(random_state=rs_image)
                assert 0 <= sample <= 1.0
                # if the value is nearly 1.0 or 0.0 skip the computation
                # and just use only the first/second image
                if sample >= 1.0 - self.epsilon:
                    image = image_first
                elif sample <= 0.0 + self.epsilon:
                    image = image_second
                else:
                    image = sample * image_first + (1 - sample) * image_second
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
                assert 0 <= sample <= 1.0

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
