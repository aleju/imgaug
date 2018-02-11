"""
Augmenters that blur images.

Do not import directly from this file, as the categorization is not final.
Use instead
    `from imgaug import augmenters as iaa`
and then e.g. ::

    seq = iaa.Sequential([
        iaa.GaussianBlur((0.0, 3.0)),
        iaa.AverageBlur((2, 5))
    ])

List of augmenters:
    * GaussianBlur
    * AverageBlur
    * MedianBlur
    * BilateralBlur
"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, DiscreteUniform, Uniform
import numpy as np
from scipy import ndimage
import cv2
import six.moves as sm

from .meta import Augmenter

class GaussianBlur(Augmenter): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Augmenter to blur images using gaussian kernels.

    Parameters
    ----------
    sigma : float or tuple of two floats or StochasticParameter
        Standard deviation of the gaussian kernel.
        Values in the range 0.0 (no blur) to 3.0 (strong blur) are common.
            * If a single float, that value will always be used as the standard
              deviation.
            * If a tuple (a, b), then a random value from the range a <= x <= b
              will be picked per image.
            * If a StochasticParameter, then N samples will be drawn from
              that parameter per N input images.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.GaussianBlur(sigma=1.5)

    blurs all images using a gaussian kernel with standard deviation 1.5.

    >>> aug = iaa.GaussianBlur(sigma=(0.0, 3.0))

    blurs images using a gaussian kernel with a random standard deviation
    from the range 0.0 <= x <= 3.0. The value is sampled per image.

    """

    def __init__(self, sigma=0, name=None, deterministic=False, random_state=None):
        super(GaussianBlur, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(sigma):
            self.sigma = Deterministic(sigma)
        elif ia.is_iterable(sigma):
            ia.do_assert(len(sigma) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(sigma),))
            self.sigma = Uniform(sigma[0], sigma[1])
        elif isinstance(sigma, StochasticParameter):
            self.sigma = sigma
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(sigma),))

        self.eps = 0.001 # epsilon value to estimate whether sigma is above 0

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.sigma.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            nb_channels = images[i].shape[2]
            sig = samples[i]
            if sig > 0 + self.eps:
                # note that while gaussian_filter can be applied to all channels
                # at the same time, that should not be done here, because then
                # the blurring would also happen across channels (e.g. red
                # values might be mixed with blue values in RGB)
                for channel in sm.xrange(nb_channels):
                    result[i][:, :, channel] = ndimage.gaussian_filter(result[i][:, :, channel], sig)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.sigma]

class AverageBlur(Augmenter): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Blur an image by computing simple means over neighbourhoods.

    Parameters
    ----------
    k : int or tuple of two ints or tuple of each one/two ints or StochasticParameter or tuple of two StochasticParameter, optional
        Kernel size to use.
            * If a single int, then that value will be used for the height
              and width of the kernel.
            * If a tuple of two ints `(a, b)`, then the kernel size will be
              sampled from the interval `[a..b]`.
            * If a StochasticParameter, then `N` samples will be drawn from
              that parameter per `N` input images, each representing the kernel
              size for the nth image.
            * If a tuple `(a, b)`, where either `a` or `b` is a tuple, then `a`
              and `b` will be treated according to the rules above. This leads
              to different values for height and width of the kernel.

    name : string, optional
        See `Augmenter.__init__()`

    deterministic : bool, optional
        See `Augmenter.__init__()`

    random_state : None or int or np.random.RandomState, optional
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.AverageBlur(k=5)

    Blurs all images using a kernel size of 5x5.

    >>> aug = iaa.AverageBlur(k=(2, 5))

    Blurs images using a varying kernel size per image, which is sampled
    from the interval [2..5].

    >>> aug = iaa.AverageBlur(k=((5, 7), (1, 3)))

    Blurs images using a varying kernel size per image, which's height
    is sampled from the interval [5..7] and which's width is sampled
    from [1..3].

    """

    def __init__(self, k=1, name=None, deterministic=False, random_state=None):
        super(AverageBlur, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.mode = "single"
        if ia.is_single_number(k):
            self.k = Deterministic(int(k))
        elif ia.is_iterable(k):
            ia.do_assert(len(k) == 2)
            if all([ia.is_single_number(ki) for ki in k]):
                self.k = DiscreteUniform(int(k[0]), int(k[1]))
            elif all([isinstance(ki, StochasticParameter) for ki in k]):
                self.mode = "two"
                self.k = (k[0], k[1])
            else:
                k_tuple = [None, None]
                if ia.is_single_number(k[0]):
                    k_tuple[0] = Deterministic(int(k[0]))
                elif ia.is_iterable(k[0]) and all([ia.is_single_number(ki) for ki in k[0]]):
                    k_tuple[0] = DiscreteUniform(int(k[0][0]), int(k[0][1]))
                else:
                    raise Exception("k[0] expected to be int or tuple of two ints, got %s" % (type(k[0]),))

                if ia.is_single_number(k[1]):
                    k_tuple[1] = Deterministic(int(k[1]))
                elif ia.is_iterable(k[1]) and all([ia.is_single_number(ki) for ki in k[1]]):
                    k_tuple[1] = DiscreteUniform(int(k[1][0]), int(k[1][1]))
                else:
                    raise Exception("k[1] expected to be int or tuple of two ints, got %s" % (type(k[1]),))

                self.mode = "two"
                self.k = k_tuple
        elif isinstance(k, StochasticParameter):
            self.k = k
        else:
            raise Exception("Expected int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(k),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        if self.mode == "single":
            samples = self.k.draw_samples((nb_images,), random_state=random_state)
            samples = (samples, samples)
        else:
            samples = (
                self.k[0].draw_samples((nb_images,), random_state=random_state),
                self.k[1].draw_samples((nb_images,), random_state=random_state),
            )
        for i in sm.xrange(nb_images):
            kh, kw = samples[0][i], samples[1][i]
            #print(images.shape, result.shape, result[i].shape)
            kernel_impossible = (kh == 0 or kw == 0)
            kernel_does_nothing = (kh == 1 and kw == 1)
            if not kernel_impossible and not kernel_does_nothing:
                image_aug = cv2.blur(result[i], (kh, kw))
                # cv2.blur() removes channel axis for single-channel images
                if image_aug.ndim == 2:
                    image_aug = image_aug[..., np.newaxis]
                result[i] = image_aug
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.k]

# TODO tests
class MedianBlur(Augmenter): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Blur an image by computing median values over neighbourhoods.

    Median blurring can be used to remove small dirt from images.
    At larger kernel sizes, its effects have some similarity with Superpixels.

    Parameters
    ----------
    k : int or tuple of two ints or StochasticParameter
        Kernel
        size.
            * If a single int, then that value will be used for the height and
              width of the kernel. Must be an odd value.
            * If a tuple of two ints (a, b), then the kernel size will be an
              odd value sampled from the interval [a..b]. a and b must both
              be odd values.
            * If a StochasticParameter, then N samples will be drawn from
              that parameter per N input images, each representing the kernel
              size for the nth image. Expected to be discrete. If a sampled
              value is not odd, then that value will be increased by 1.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.MedianBlur(k=5)

    blurs all images using a kernel size of 5x5.

    >>> aug = iaa.MedianBlur(k=(3, 7))

    blurs images using a varying kernel size per image, which is
    and odd value sampled from the interval [3..7], i.e. 3 or 5 or 7.

    """

    def __init__(self, k=1, name=None, deterministic=False, random_state=None):
        super(MedianBlur, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(k):
            ia.do_assert(k % 2 != 0, "Expected k to be odd, got %d. Add or subtract 1." % (int(k),))
            self.k = Deterministic(int(k))
        elif ia.is_iterable(k):
            ia.do_assert(len(k) == 2)
            ia.do_assert(all([ia.is_single_number(ki) for ki in k]))
            ia.do_assert(k[0] % 2 != 0, "Expected k[0] to be odd, got %d. Add or subtract 1." % (int(k[0]),))
            ia.do_assert(k[1] % 2 != 0, "Expected k[1] to be odd, got %d. Add or subtract 1." % (int(k[1]),))
            self.k = DiscreteUniform(int(k[0]), int(k[1]))
        elif isinstance(k, StochasticParameter):
            self.k = k
        else:
            raise Exception("Expected int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(k),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.k.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            ki = samples[i]
            if ki > 1:
                ki = ki + 1 if ki % 2 == 0 else ki
                image_aug = cv2.medianBlur(result[i], ki)
                # cv2.medianBlur() removes channel axis for single-channel
                # images
                if image_aug.ndim == 2:
                    image_aug = image_aug[..., np.newaxis]
                result[i] = image_aug
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.k]

# TODO tests
class BilateralBlur(Augmenter): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Blur/Denoise an image using a bilateral filter.

    Bilateral filters blur homogenous and textured areas, while trying to
    preserve edges.

    See http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#bilateralfilter
    for more information regarding the parameters.

    Parameters
    ----------
    d : int or tuple of two ints or StochasticParameter
        Diameter of each pixel neighborhood.
        High values for d lead to significantly worse performance. Values
        equal or less than 10 seem to be good.
            * If a single int, then that value will be used for the diameter.
            * If a tuple of two ints (a, b), then the diameter will be a
              value sampled from the interval [a..b].
            * If a StochasticParameter, then N samples will be drawn from
              that parameter per N input images, each representing the diameter
              for the nth image. Expected to be discrete.

    sigma_color : int or tuple of two ints or StochasticParameter
        Filter sigma in the color space. A larger value of the parameter means
        that farther colors within the pixel neighborhood (see sigmaSpace )
        will be mixed together, resulting in larger areas of semi-equal color.
            * If a single int, then that value will be used for the diameter.
            * If a tuple of two ints (a, b), then the diameter will be a
              value sampled from the interval [a..b].
            * If a StochasticParameter, then N samples will be drawn from
              that parameter per N input images, each representing the diameter
              for the nth image. Expected to be discrete.

    sigma_space :
        Filter sigma in the coordinate space. A larger value of the parameter
        means that farther pixels will influence each other as long as their
        colors are close enough (see sigmaColor ).
        * If a single int, then that value will be used for the diameter.
        * If a tuple of two ints (a, b), then the diameter will be a
          value sampled from the interval [a..b].
        * If a StochasticParameter, then N samples will be drawn from
          that parameter per N input images, each representing the diameter
          for the nth image. Expected to be discrete.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))

    blurs all images using a bilateral filter with max distance 3 to 10
    and wide ranges for sigma_color and sigma_space.

    """

    def __init__(self, d=1, sigma_color=(10, 250), sigma_space=(10, 250), name=None, deterministic=False, random_state=None):
        super(BilateralBlur, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        def val(var):
            if ia.is_single_number(var):
                return Deterministic(int(var))
            elif ia.is_iterable(var):
                ia.do_assert(len(var) == 2)
                ia.do_assert(all([ia.is_single_number(var_i) for var_i in var]))
                return DiscreteUniform(int(var[0]), int(var[1]))
            elif isinstance(d, StochasticParameter):
                return var
            else:
                raise Exception("Expected int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(var),))

        self.d = val(d)
        self.sigma_color = val(sigma_color)
        self.sigma_space = val(sigma_space)

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seed = random_state.randint(0, 10**6)
        samples_d = self.d.draw_samples((nb_images,), random_state=ia.new_random_state(seed))
        samples_sigma_color = self.sigma_color.draw_samples((nb_images,), random_state=ia.new_random_state(seed+1))
        samples_sigma_space = self.sigma_space.draw_samples((nb_images,), random_state=ia.new_random_state(seed+2))
        for i in sm.xrange(nb_images):
            ia.do_assert(images[i].shape[2] == 3, "BilateralBlur can currently only be applied to images with 3 channels.")
            di = samples_d[i]
            sigma_color_i = samples_sigma_color[i]
            sigma_space_i = samples_sigma_space[i]

            if di != 1:
                result[i] = cv2.bilateralFilter(images[i], di, sigma_color_i, sigma_space_i)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.d, self.sigma_color, self.sigma_space]
