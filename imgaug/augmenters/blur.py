"""
Augmenters that blur images.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

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
    * MotionBlur

"""
from __future__ import print_function, division, absolute_import

import numpy as np
from scipy import ndimage
import cv2
import six.moves as sm

from . import meta
from . import convolutional as iaa_convolutional
from . import geometric as iaa_geometric
from .. import imgaug as ia
from .. import parameters as iap


class GaussianBlur(meta.Augmenter):  # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Augmenter to blur images using gaussian kernels.

    Parameters
    ----------
    sigma : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the gaussian kernel.
        Values in the range ``0.0`` (no blur) to ``3.0`` (strong blur) are common.

            * If a single float, that value will always be used as the standard
              deviation.
            * If a tuple ``(a, b)``, then a random value from the range ``a <= x <= b``
              will be picked per image.
            * If a list, then a random value will be sampled per image from
              that list.
            * If a StochasticParameter, then ``N`` samples will be drawn from
              that parameter per ``N`` input images.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.GaussianBlur(sigma=1.5)

    blurs all images using a gaussian kernel with standard deviation 1.5.

    >>> aug = iaa.GaussianBlur(sigma=(0.0, 3.0))

    blurs images using a gaussian kernel with a random standard deviation
    from the range ``0.0 <= x <= 3.0``. The value is sampled per image.

    """

    def __init__(self, sigma=0, name=None, deterministic=False, random_state=None):
        super(GaussianBlur, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.sigma = iap.handle_continuous_param(sigma, "sigma", value_range=(0, None), tuple_to_uniform=True,
                                                 list_to_choice=True)
        self.eps = 0.001  # epsilon value to estimate whether sigma is sufficently above 0 to apply the blur

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.sigma.draw_samples((nb_images,), random_state=random_state)
        for i, (image, sig) in enumerate(zip(images, samples)):
            nb_channels = image.shape[2]
            if sig > 0 + self.eps:
                # note that while gaussian_filter can be applied to all channels
                # at the same time, that should not be done here, because then
                # the blurring would also happen across channels (e.g. red
                # values might be mixed with blue values in RGB)
                for channel in sm.xrange(nb_channels):
                    image[:, :, channel] = ndimage.gaussian_filter(image[:, :, channel], sig)
                images[i] = image
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.sigma]


class AverageBlur(meta.Augmenter):  # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Blur an image by computing simple means over neighbourhoods.

    Parameters
    ----------
    k : int or tuple of int or tuple of tuple of int or imgaug.parameters.StochasticParameter\
        or tuple of StochasticParameter, optional
        Kernel size to use.

            * If a single int, then that value will be used for the height
              and width of the kernel.
            * If a tuple of two ints ``(a, b)``, then the kernel size will be
              sampled from the interval ``[a..b]``.
            * If a tuple of two tuples of ints ``((a, b), (c, d))``, then per image
              a random kernel height will be sampled from the interval ``[a..b]``
              and a random kernel width will be sampled from the interval ``[c..d]``.
            * If a StochasticParameter, then ``N`` samples will be drawn from
              that parameter per ``N`` input images, each representing the kernel
              size for the nth image.
            * If a tuple ``(a, b)``, where either ``a`` or ``b`` is a tuple, then ``a``
              and ``b`` will be treated according to the rules above. This leads
              to different values for height and width of the kernel.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.AverageBlur(k=5)

    Blurs all images using a kernel size of ``5x5``.

    >>> aug = iaa.AverageBlur(k=(2, 5))

    Blurs images using a varying kernel size per image, which is sampled
    from the interval ``[2..5]``.

    >>> aug = iaa.AverageBlur(k=((5, 7), (1, 3)))

    Blurs images using a varying kernel size per image, which's height
    is sampled from the interval ``[5..7]`` and which's width is sampled
    from ``[1..3]``.

    """

    def __init__(self, k=1, name=None, deterministic=False, random_state=None):
        super(AverageBlur, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.mode = "single"
        if ia.is_single_number(k):
            self.k = iap.Deterministic(int(k))
        elif ia.is_iterable(k):
            ia.do_assert(len(k) == 2)
            if all([ia.is_single_number(ki) for ki in k]):
                self.k = iap.DiscreteUniform(int(k[0]), int(k[1]))
            elif all([isinstance(ki, iap.StochasticParameter) for ki in k]):
                self.mode = "two"
                self.k = (k[0], k[1])
            else:
                k_tuple = [None, None]
                if ia.is_single_number(k[0]):
                    k_tuple[0] = iap.Deterministic(int(k[0]))
                elif ia.is_iterable(k[0]) and all([ia.is_single_number(ki) for ki in k[0]]):
                    k_tuple[0] = iap.DiscreteUniform(int(k[0][0]), int(k[0][1]))
                else:
                    raise Exception("k[0] expected to be int or tuple of two ints, got %s" % (type(k[0]),))

                if ia.is_single_number(k[1]):
                    k_tuple[1] = iap.Deterministic(int(k[1]))
                elif ia.is_iterable(k[1]) and all([ia.is_single_number(ki) for ki in k[1]]):
                    k_tuple[1] = iap.DiscreteUniform(int(k[1][0]), int(k[1][1]))
                else:
                    raise Exception("k[1] expected to be int or tuple of two ints, got %s" % (type(k[1]),))

                self.mode = "two"
                self.k = k_tuple
        elif isinstance(k, iap.StochasticParameter):
            self.k = k
        else:
            raise Exception("Expected int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(k),))

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        if self.mode == "single":
            samples = self.k.draw_samples((nb_images,), random_state=random_state)
            samples = (samples, samples)
        else:
            rss = ia.derive_random_states(random_state, 2)
            samples = (
                self.k[0].draw_samples((nb_images,), random_state=rss[0]),
                self.k[1].draw_samples((nb_images,), random_state=rss[1]),
            )
        for i, (image, kh, kw) in enumerate(zip(images, samples[0], samples[1])):
            kernel_impossible = (kh == 0 or kw == 0)
            kernel_does_nothing = (kh == 1 and kw == 1)
            if not kernel_impossible and not kernel_does_nothing:
                image_aug = cv2.blur(image, (kh, kw))
                # cv2.blur() removes channel axis for single-channel images
                if image_aug.ndim == 2:
                    image_aug = image_aug[..., np.newaxis]
                images[i] = image_aug
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.k]


class MedianBlur(meta.Augmenter):  # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Blur an image by computing median values over neighbourhoods.

    Median blurring can be used to remove small dirt from images.
    At larger kernel sizes, its effects have some similarity with Superpixels.

    Parameters
    ----------
    k : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Kernel size.

            * If a single int, then that value will be used for the height and
              width of the kernel. Must be an odd value.
            * If a tuple of two ints ``(a, b)``, then the kernel size will be an
              odd value sampled from the interval ``[a..b]``. ``a`` and ``b`` must both
              be odd values.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then ``N`` samples will be drawn from
              that parameter per ``N`` input images, each representing the kernel
              size for the nth image. Expected to be discrete. If a sampled
              value is not odd, then that value will be increased by 1.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.MedianBlur(k=5)

    blurs all images using a kernel size of ``5x5``.

    >>> aug = iaa.MedianBlur(k=(3, 7))

    blurs images using a varying kernel size per image, which is
    and odd value sampled from the interval ``[3..7]``, i.e. 3 or 5 or 7.

    """

    def __init__(self, k=1, name=None, deterministic=False, random_state=None):
        super(MedianBlur, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.k = iap.handle_discrete_param(k, "k", value_range=(1, None), tuple_to_uniform=True, list_to_choice=True,
                                           allow_floats=False)
        if ia.is_single_integer(k):
            ia.do_assert(k % 2 != 0, "Expected k to be odd, got %d. Add or subtract 1." % (int(k),))
        elif ia.is_iterable(k):
            ia.do_assert(all([ki % 2 != 0 for ki in k]),
                         "Expected all values in iterable k to be odd, but at least one was not. "
                         + "Add or subtract 1 to/from that value.")

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.k.draw_samples((nb_images,), random_state=random_state)
        for i, (image, ki) in enumerate(zip(images, samples)):
            if ki > 1:
                ki = ki + 1 if ki % 2 == 0 else ki
                image_aug = cv2.medianBlur(image, ki)
                # cv2.medianBlur() removes channel axis for single-channel
                # images
                if image_aug.ndim == 2:
                    image_aug = image_aug[..., np.newaxis]
                images[i] = image_aug
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.k]


# TODO tests
class BilateralBlur(meta.Augmenter):  # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Blur/Denoise an image using a bilateral filter.

    Bilateral filters blur homogenous and textured areas, while trying to
    preserve edges.

    See http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#bilateralfilter
    for more information regarding the parameters.

    Parameters
    ----------
    d : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Diameter of each pixel neighborhood with value range ``[1 .. inf)``.
        High values for d lead to significantly worse performance. Values
        equal or less than 10 seem to be good. Use ``<5`` for real-time
        applications.

            * If a single int, then that value will be used for the diameter.
            * If a tuple of two ints ``(a, b)``, then the diameter will be a
              value sampled from the interval ``[a..b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then ``N`` samples will be drawn from
              that parameter per ``N`` input images, each representing the diameter
              for the nth image. Expected to be discrete.

    sigma_color : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Filter sigma in the color space with value range [1, inf). A larger value
        of the parameter means that farther colors within the pixel neighborhood
        (see sigma_space) will be mixed together, resulting in larger areas of
        semi-equal color.

            * If a single int, then that value will be used for the diameter.
            * If a tuple of two ints ``(a, b)``, then the diameter will be a
              value sampled from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then ``N`` samples will be drawn from
              that parameter per ``N`` input images, each representing the diameter
              for the nth image. Expected to be discrete.

    sigma_space : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Filter sigma in the coordinate space with value range ``[1, inf)``. A larger value
        of the parameter means that farther pixels will influence each other as long as
        their colors are close enough (see sigma_color).

            * If a single int, then that value will be used for the diameter.
            * If a tuple of two ints ``(a, b)``, then the diameter will be a
              value sampled from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then ``N`` samples will be drawn from
              that parameter per ``N`` input images, each representing the diameter
              for the nth image. Expected to be discrete.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))

    blurs all images using a bilateral filter with max distance 3 to 10
    and wide ranges for sigma_color and sigma_space.

    """

    def __init__(self, d=1, sigma_color=(10, 250), sigma_space=(10, 250), name=None, deterministic=False,
                 random_state=None):
        super(BilateralBlur, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.d = iap.handle_discrete_param(d, "d", value_range=(1, None), tuple_to_uniform=True, list_to_choice=True,
                                           allow_floats=False)
        self.sigma_color = iap.handle_continuous_param(sigma_color, "sigma_color", value_range=(1, None),
                                                       tuple_to_uniform=True, list_to_choice=True)
        self.sigma_space = iap.handle_continuous_param(sigma_space, "sigma_space", value_range=(1, None),
                                                       tuple_to_uniform=True, list_to_choice=True)

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seed = random_state.randint(0, 10**6)
        samples_d = self.d.draw_samples((nb_images,), random_state=ia.new_random_state(seed))
        samples_sigma_color = self.sigma_color.draw_samples((nb_images,), random_state=ia.new_random_state(seed+1))
        samples_sigma_space = self.sigma_space.draw_samples((nb_images,), random_state=ia.new_random_state(seed+2))
        for i in sm.xrange(nb_images):
            ia.do_assert(images[i].shape[2] == 3,
                         "BilateralBlur can currently only be applied to images with 3 channels.")
            di = samples_d[i]
            sigma_color_i = samples_sigma_color[i]
            sigma_space_i = samples_sigma_space[i]

            if di != 1:
                result[i] = cv2.bilateralFilter(images[i], di, sigma_color_i, sigma_space_i)
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.d, self.sigma_color, self.sigma_space]


# TODO add k sizing via float/percentage
def MotionBlur(k=5, angle=(0, 360), direction=(-1.0, 1.0), order=1, name=None, deterministic=False, random_state=None):
    """
    Augmenter that sharpens images and overlays the result with the original image.

    Parameters
    ----------
    k : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Kernel size to use.

            * If a single int, then that value will be used for the height
              and width of the kernel.
            * If a tuple of two ints ``(a, b)``, then the kernel size will be
              sampled from the interval ``[a..b]``.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then ``N`` samples will be drawn from
              that parameter per ``N`` input images, each representing the kernel
              size for the nth image.

    angle : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Angle of the motion blur in degrees (clockwise, relative to top center direction).

            * If a number, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the range ``a <= x <= b`` will
              be sampled per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    direction : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Forward/backward direction of the motion blur. Lower values towards -1.0 will point the motion blur towards
        the back (with angle provided via `angle`). Higher values towards 1.0 will point the motion blur forward.
        A value of 0.0 leads to a uniformly (but still angled) motion blur.

            * If a number, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the range ``a <= x <= b`` will
              be sampled per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use when rotating the kernel according to `angle`.
        See :func:`imgaug.augmenters.geometric.Affine.__init__`.
        Recommended to be ``0`` or ``1``, with ``0`` being faster, but less continuous/smooth as `angle` is changed,
        particularly around multiple of 45 degrees.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.MotionBlur(k=15)

    Create a motion blur augmenter with kernel size of 15x15.

    >>> aug = iaa.MotionBlur(k=15, angle=[-45, 45])

    Create a motion blur augmenter with kernel size of 15x15 and a blur angle of either -45 or 45 degrees (randomly
    picked per image).

    """
    k_param = iap.handle_continuous_param(k, "k", value_range=(3, None), tuple_to_uniform=True, list_to_choice=True)
    angle_param = iap.handle_continuous_param(angle, "angle", value_range=None, tuple_to_uniform=True,
                                              list_to_choice=True)
    direction_param = iap.handle_continuous_param(direction, "direction", value_range=(-1.0-1e-6, 1.0+1e-6),
                                                  tuple_to_uniform=True, list_to_choice=True)

    def create_matrices(image, nb_channels, random_state_func):
        k_sample = k_param.draw_sample(random_state=random_state_func)
        angle_sample = angle_param.draw_sample(random_state=random_state_func)
        direction_sample = direction_param.draw_sample(random_state=random_state_func)

        k_sample = k_sample if k_sample % 2 != 0 else k_sample + 1
        direction_sample = np.clip(direction_sample, -1.0, 1.0)
        direction_sample = (direction_sample + 1.0) / 2.0

        matrix = np.zeros((k_sample, k_sample), dtype=np.float32)
        matrix[:, k_sample//2] = np.linspace(float(direction_sample), 1.0 - float(direction_sample), num=k_sample)
        rot = iaa_geometric.Affine(rotate=angle_sample, order=order)
        matrix = (rot.augment_image((matrix * 255).astype(np.uint8)) / 255.0).astype(np.float32)

        return [matrix/np.sum(matrix)] * nb_channels

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return iaa_convolutional.Convolve(create_matrices, name=name, deterministic=deterministic,
                                      random_state=random_state)
