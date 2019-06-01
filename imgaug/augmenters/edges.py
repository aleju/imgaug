"""
Augmenters that deal with edge detection.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([
        iaa.Canny()
    ])

List of augmenters:

    * Canny

EdgeDetect and DirectedEdgeDetect are currently still in `convolutional.py`.

"""
from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod

import numpy as np
import cv2
import six

from . import meta
from . import blend
import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt


# TODO this should be placed in some other file than edges.py as it could be
#      re-used wherever a binary image is the result
@six.add_metaclass(ABCMeta)
class BinaryImageColorizerIf(object):
    @abstractmethod
    def colorize(self, image_binary, image_original, nth_image, random_state):
        """
        Convert a binary image to a colorized one.

        Parameters
        ----------
        image_binary : ndarray
            Boolean ``(H,W)`` image.

        image_original : ndarray
            Original ``(H,W,C)`` input image.

        nth_image : int
            Index of the image in the batch.

        random_state : numpy.random.RandomState
            Random state to use.

        Returns
        -------
        ndarray
            Colorized form of `image_binary`.

        """


# TODO see above, this should be moved to another file
class RandomColorsBinaryImageColorizer(BinaryImageColorizerIf):
    """
    Colorizer using two randomly sampled foreground/background colors.

    Parameters
    ----------
    color_true : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Color of the foreground, i.e. all pixels in binary images that are
        ``True``. This parameter will be queried once per image to
        generate ``(3,)`` samples denoting the color. (Note that even for
        grayscale images three values will be sampled and converted to
        grayscale according to ``0.299*R + 0.587*G + 0.114*B``. This is the
        same equation that is also used by OpenCV.)

            * If an int, exactly that value will always be used, i.e. every
              color will be ``(v, v, v)`` for value ``v``.
            * If a tuple ``(a, b)``, three random values from the range
              ``a <= x <= b`` will be sampled per image.
            * If a list, then three random values will be sampled from that
              list per image.
            * If a StochasticParameter, three values will be sampled from the
              parameter per image.

    color_false : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Analogous to `color_true`, but denotes the color for all pixels that
        are ``False`` in the binary input image.

    """

    def __init__(self, color_true=(0, 255), color_false=(0, 255)):
        self.color_true = iap.handle_discrete_param(
            color_true,
            "color_true",
            value_range=(0, 255),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False)

        self.color_false = iap.handle_discrete_param(
            color_false,
            "color_false",
            value_range=(0, 255),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False)

    def _draw_samples(self, random_state):
        color_true = self.color_true.draw_samples((3,),
                                                  random_state=random_state)
        color_false = self.color_false.draw_samples((3,),
                                                    random_state=random_state)
        return color_true, color_false

    def colorize(self, image_binary, image_original, nth_image, random_state):
        assert image_binary.ndim == 2
        assert image_binary.dtype.kind == "b"
        assert image_original.ndim == 3
        assert image_original.shape[-1] in [1, 3, 4]
        assert image_original.dtype.name == "uint8"

        color_true, color_false = self._draw_samples(random_state)

        nb_channels = min(image_original.shape[-1], 3)
        image_colorized = np.zeros(
            (image_original.shape[0], image_original.shape[1], nb_channels),
            dtype=image_original.dtype)

        if nb_channels == 1:
            # single channel input image, convert colors to grayscale
            image_colorized[image_binary] = (
                0.299*color_true[0]
                + 0.587*color_true[1]
                + 0.114*color_true[2])
            image_colorized[~image_binary] = (
                0.299*color_false[0]
                + 0.587*color_false[1]
                + 0.114*color_false[2])
        else:
            image_colorized[image_binary] = color_true
            image_colorized[~image_binary] = color_false

        # re-attach alpha channel if it was present in input image
        if image_original.shape[-1] == 4:
            image_colorized = np.dstack(
                [image_colorized, image_original[:, :, 3:4]])

        return image_colorized

    def __str__(self):
        return ("RandomColorsBinaryImageColorizer("
                "color_true=%s, color_false=%s)") % (
            self.color_true, self.color_false)


class Canny(meta.Augmenter):
    """
    Apply a canny edge detector to input images.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: no; not tested
        * ``uint32``: no; not tested
        * ``uint64``: no; not tested
        * ``int8``: no; not tested
        * ``int16``: no; not tested
        * ``int32``: no; not tested
        * ``int64``: no; not tested
        * ``float16``: no; not tested
        * ``float32``: no; not tested
        * ``float64``: no; not tested
        * ``float128``: no; not tested
        * ``bool``: no; not tested

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Blending factor to use in alpha blending.
        A value close to 1.0 means that only the edge image is visible.
        A value close to 0.0 means that only the original image is visible.
        A value close to 0.5 means that the images are merged according to
        `0.5*image + 0.5*edge_image`.
        If a sample from this parameter is 0, no action will be performed for
        the corresponding image.

            * If an int or float, exactly that value will be used.
            * If a tuple ``(a, b)``, a random value from the range
              ``a <= x <= b`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    hysteresis_thresholds : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or tuple of tuple of number or tuple of list of number or tuple of imgaug.parameters.StochasticParameter, optional
        Min and max values to use in hysteresis thresholding.
        (This parameter seems to have not very much effect on the results.)
        Either a single parameter or a tuple of two parameters.
        If a single parameter is provided, the sampling happens once for all
        images with `(N,2)` samples being requested from the parameter,
        where each first value denotes the hysteresis minimum and each second
        the maximum.
        If a tuple of two parameters is provided, one sampling of `(N,)` values
        is independently performed per parameter (first parameter: hysteresis
        minimum, second: hysteresis maximum).

            * If this is a single number, both min and max value will always be
              exactly that value.
            * If this is a tuple of numbers ``(a, b)``, two random values from
              the range ``a <= x <= b`` will be sampled per image.
            * If this is a list, two random values will be sampled from that
              list per image.
            * If this is a StochasticParameter, two random values will be
              sampled from that parameter per image.
            * If this is a tuple ``(min, max)`` with ``min`` and ``max``
              both *not* being numbers, they will be treated according to the
              rules above (i.e. may be a number, tuple, list or
              StochasticParameter). A single value will be sampled per image
              and parameter.

    sobel_kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Kernel size of the sobel operator initially applied to each image.
        This corresponds to ``apertureSize`` in ``cv2.Canny()``.
        If a sample from this parameter is ``<=1``, no action will be performed
        for the corresponding image.
        The maximum for this parameter is ``7`` (inclusive). Higher values are
        not accepted by OpenCV.
        If an even value ``v`` is sampled, it is automatically changed to
        ``v-1``.

            * If this is a single integer, the kernel size always matches that
              value.
            * If this is a tuple of integers ``(a, b)``, a random discrete
              value will be sampled from the range ``a <= x <= b`` per image.
            * If this is a list, a random value will be sampled from that
              list per image.
            * If this is a StochasticParameter, a random value will be sampled
              from that parameter per image.

    colorizer : None or imgaug.augmenters.edges.BinaryImageColorizerIf, optional
        A strategy to convert binary edge images to color images.
        If this is ``None``, an instance of ``RandomColorBinaryImageColorizer``
        is created, which means that each edge image is converted into an
        ``uint8`` image, where edge and non-edge pixels each have a different
        color that was uniformly randomly sampled from the space of all
        ``uint8`` colors.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Canny()

    Creates an augmenter that generates random blends between images and
    their canny edge representations. Apply the augmenter to images using
    e.g. ``images_aug = aug(images=<list of numpy array>)``.

    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Canny(sobel_kernel_size=(0, 7))

    Creates a canny edge augmenter that initially preprocesses images using
    a sobel filter with kernel size ``3x3`` to ``7x7`` and will sometimes
    not modify images at all (if a value ``<=2`` is sampled).

    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Canny(alpha=(0.0, 0.5))

    Creates a canny edge augmenter that generates edge images with a blending
    factor of max 50%, i.e. the original (non-edge) image is always at least
    partially visible.

    """

    def __init__(self,
                 alpha=(0.0, 1.0),
                 hysteresis_thresholds=((100-40, 100+40), (200-40, 200+40)),
                 sobel_kernel_size=(3, 7),
                 colorizer=None,
                 name=None, deterministic=False, random_state=None):
        super(Canny, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.alpha = iap.handle_continuous_param(
            alpha, "alpha", value_range=(0, 1.0), tuple_to_uniform=True,
            list_to_choice=True)

        if isinstance(hysteresis_thresholds, tuple) \
                and len(hysteresis_thresholds) == 2 \
                and not ia.is_single_number(hysteresis_thresholds[0]) \
                and not ia.is_single_number(hysteresis_thresholds[1]):
            self.hysteresis_thresholds = (
                iap.handle_discrete_param(
                    hysteresis_thresholds[0],
                    "hysteresis_thresholds[0]",
                    value_range=(0, 255),
                    tuple_to_uniform=True,
                    list_to_choice=True,
                    allow_floats=True),
                iap.handle_discrete_param(
                    hysteresis_thresholds[1],
                    "hysteresis_thresholds[1]",
                    value_range=(0, 255),
                    tuple_to_uniform=True,
                    list_to_choice=True,
                    allow_floats=True)
            )
        else:
            self.hysteresis_thresholds = iap.handle_discrete_param(
                hysteresis_thresholds,
                "hysteresis_thresholds",
                value_range=(0, 255),
                tuple_to_uniform=True,
                list_to_choice=True,
                allow_floats=True)

        # we don't use handle_discrete_kernel_size_param() here, because
        # cv2.Canny() can't handle independent height/width values, only a
        # single kernel size
        self.sobel_kernel_size = iap.handle_discrete_param(
            sobel_kernel_size,
            "sobel_kernel_size",
            value_range=(0, 7),  # OpenCV only accepts ksize up to 7
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False)

        self.colorizer = (
            colorizer
            if colorizer is not None
            else RandomColorsBinaryImageColorizer()
        )

    def _draw_samples(self, augmentables, random_state):
        nb_images = len(augmentables)
        rss = ia.derive_random_states(random_state, 4)

        alpha_samples = self.alpha.draw_samples((nb_images,), rss[0])

        hthresh = self.hysteresis_thresholds
        if isinstance(hthresh, tuple):
            assert len(hthresh) == 2
            min_values = hthresh[0].draw_samples((nb_images,), rss[1])
            max_values = hthresh[1].draw_samples((nb_images,), rss[2])
            hthresh_samples = np.stack([min_values, max_values], axis=-1)
        else:
            hthresh_samples = hthresh.draw_samples((nb_images, 2), rss[1])

        sobel_samples = self.sobel_kernel_size.draw_samples((nb_images,),
                                                            rss[3])

        # verify for hysteresis thresholds that min_value < max_value everywhere
        invalid = (hthresh_samples[:, 0] > hthresh_samples[:, 1])
        if np.any(invalid):
            hthresh_samples[invalid, :] = hthresh_samples[invalid, :][:, [1, 0]]

        # ensure that sobel kernel sizes are correct
        # note that OpenCV accepts only kernel sizes that are (a) even
        # and (b) <=7
        assert not np.any(sobel_samples < 0), (
            "Sampled a sobel kernel size below 0 in Canny. "
            "Allowed value range is 0 to 7.")
        assert not np.any(sobel_samples > 7), (
            "Sampled a sobel kernel size above 7 in Canny. "
            "Allowed value range is 0 to 7.")
        even_idx = (np.mod(sobel_samples, 2) == 0)
        sobel_samples[even_idx] -= 1

        return alpha_samples, hthresh_samples, sobel_samples

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["uint8"],
                         disallowed=[
                             "bool",
                             "uint16", "uint32", "uint64", "uint128",
                             "uint256",
                             "int8", "int16", "int32", "int64", "int128",
                             "int256",
                             "float32", "float64", "float96", "float128",
                             "float256"],
                         augmenter=self)

        rss = ia.derive_random_states(random_state, len(images))
        samples = self._draw_samples(images, rss[-1])
        alpha_samples = samples[0]
        hthresh_samples = samples[1]
        sobel_samples = samples[2]
        result = images

        gen = enumerate(zip(images, alpha_samples, hthresh_samples,
                            sobel_samples))
        for i, (image, alpha, hthreshs, sobel) in gen:
            assert image.ndim == 3
            assert image.shape[-1] in [1, 3, 4], (
                "Canny edge detector can currently only handle images with "
                "channel numbers that are 1, 3 or 4. Got %d.") % (
                    image.shape[-1],)

            if alpha > 0 and sobel > 1:
                image_canny = cv2.Canny(
                    image[:, :, 0:3],
                    threshold1=hthreshs[0],
                    threshold2=hthreshs[1],
                    apertureSize=sobel,
                    L2gradient=True)
                image_canny = (image_canny > 0)

                # canny returns a boolean (H,W) image, so we change it to
                # (H,W,C) and then uint8
                image_canny_color = self.colorizer.colorize(
                    image_canny, image, nth_image=i, random_state=rss[i])

                result[i] = blend.blend_alpha(image_canny_color, image, alpha)

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        return keypoints_on_images

    def get_parameters(self):
        return [self.alpha, self.hysteresis_thresholds, self.sobel_kernel_size,
                self.colorizer]

    def __str__(self):
        return ("Canny("
                "alpha=%s, "
                "hysteresis_thresholds=%s, "
                "sobel_kernel_size=%s, "
                "colorizer=%s, "
                "name=%s, "
                "deterministic=%s)" % (
                    self.alpha, self.hysteresis_thresholds,
                    self.sobel_kernel_size, self.colorizer,
                    self.name, self.deterministic))
