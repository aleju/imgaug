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

import warnings

import numpy as np
from scipy import ndimage
import cv2
import six.moves as sm

from . import meta
from . import convolutional as iaa_convolutional
import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt


# TODO add border mode, cval
def blur_gaussian_(image, sigma, ksize=None, backend="auto", eps=1e-3):
    """
    Blur an image using gaussian blurring.

    This operation might change the input image in-place.

    dtype support::

        if (backend="auto")::

            * ``uint8``: yes; fully tested (1)
            * ``uint16``: yes; tested (1)
            * ``uint32``: yes; tested (2)
            * ``uint64``: yes; tested (2)
            * ``int8``: yes; tested (1)
            * ``int16``: yes; tested (1)
            * ``int32``: yes; tested (1)
            * ``int64``: yes; tested (2)
            * ``float16``: yes; tested (1)
            * ``float32``: yes; tested (1)
            * ``float64``: yes; tested (1)
            * ``float128``: no
            * ``bool``: yes; tested (1)

            - (1) Handled by ``cv2``. See ``backend="cv2"``.
            - (2) Handled by ``scipy``. See ``backend="scipy"``.

        if (backend="cv2")::

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: no (2)
            * ``uint64``: no (3)
            * ``int8``: yes; tested (4)
            * ``int16``: yes; tested
            * ``int32``: yes; tested (5)
            * ``int64``: no (6)
            * ``float16``: yes; tested (7)
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: no (8)
            * ``bool``: yes; tested (1)

            - (1) Mapped internally to ``float32``. Otherwise causes ``TypeError: src data type = 0 is not supported``.
            - (2) Causes ``TypeError: src data type = 6 is not supported``.
            - (3) Causes ``cv2.error: OpenCV(3.4.5) (...)/filter.cpp:2957: error: (-213:The function/feature is not
                  implemented) Unsupported combination of source format (=4), and buffer format (=5) in function
                  'getLinearRowFilter'``.
            - (4) Mapped internally to ``int16``. Otherwise causes ``cv2.error: OpenCV(3.4.5) (...)/filter.cpp:2957:
                  error: (-213:The function/feature is not implemented) Unsupported combination of source format (=1),
                  and buffer format (=5) in function 'getLinearRowFilter'``.
            - (5) Mapped internally to ``float64``. Otherwise causes ``cv2.error: OpenCV(3.4.5) (...)/filter.cpp:2957:
                  error: (-213:The function/feature is not implemented) Unsupported combination of source format (=4),
                  and buffer format (=5) in function 'getLinearRowFilter'``.
            - (6) Causes ``cv2.error: OpenCV(3.4.5) (...)/filter.cpp:2957: error: (-213:The function/feature is not
                  implemented) Unsupported combination of source format (=4), and buffer format (=5) in function
                  'getLinearRowFilter'``.
            - (7) Mapped internally to ``float32``. Otherwise causes ``TypeError: src data type = 23 is not supported``.
            - (8) Causes ``TypeError: src data type = 13 is not supported``.


        if (backend="scipy")::

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested
            * ``uint64``: yes; tested
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested
            * ``int64``: yes; tested
            * ``float16``: yes; tested (1)
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: no (2)
            * ``bool``: yes; tested (3)

            - (1) Mapped internally to ``float32``. Otherwise causes ``RuntimeError: array type dtype('float16')
                  not supported``.
            - (2) Causes ``RuntimeError: array type dtype('float128') not supported``.
            - (3) Mapped internally to ``float32``. Otherwise too inaccurate.

    Parameters
    ----------
    image : numpy.ndarray
        The image to blur. Expected to be of shape ``(H, W)`` or ``(H, W, C)``.

    sigma : number
        Standard deviation of the gaussian blur. Larger numbers result in more large-scale blurring, which is overall
        slower than small-scale blurring.

    ksize : None or int, optional
        Size in height/width of the gaussian kernel. This argument is only understood by the ``cv2`` backend.
        If it is set to None, an appropriate value for `ksize` will automatically be derived from `sigma`.
        The value is chosen tighter for larger sigmas to avoid as much as possible very large kernel sizes
        and therey improve performance.

    backend : {'auto', 'cv2', 'scipy'}, optional
        Backend library to use. If ``auto``, then the likely best library will be automatically picked per image. That
        is usually equivalent to ``cv2`` (OpenCV) and it will fall back to ``scipy`` for datatypes not supported by
        OpenCV.

    eps : number, optional
        A threshold used to decide whether `sigma` can be considered zero.

    Returns
    -------
    image : numpy.ndarray
        The blurred image. Same shape and dtype as the input.

    """
    if sigma > 0 + eps:
        dtype = image.dtype

        iadt.gate_dtypes(image,
                         allowed=["bool",
                                  "uint8", "uint16", "uint32",
                                  "int8", "int16", "int32", "int64", "uint64",
                                  "float16", "float32", "float64"],
                         disallowed=["uint128", "uint256",
                                     "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=None)

        dts_not_supported_by_cv2 = ["uint32", "uint64", "int64", "float128"]
        backend_to_use = backend
        if backend == "auto":
            backend_to_use = "cv2" if image.dtype.name not in dts_not_supported_by_cv2 else "scipy"
        elif backend == "cv2":
            assert image.dtype.name not in dts_not_supported_by_cv2,\
                ("Requested 'cv2' backend, but provided %s input image, which "
                 + "cannot be handled by that backend. Choose a different backend or "
                 + "set backend to 'auto' or use a different datatype.") % (image.dtype.name,)
        elif backend == "scipy":
            # can handle all dtypes that were allowed in gate_dtypes()
            pass

        if backend_to_use == "scipy":
            if dtype.name == "bool":
                # We convert bool to float32 here, because gaussian_filter() seems to only return True when
                # the underlying value is approximately 1.0, not when it is above 0.5. So we do that here manually.
                # cv2 does not support bool for gaussian blur
                image = image.astype(np.float32, copy=False)
            elif dtype.name == "float16":
                image = image.astype(np.float32, copy=False)

            # gaussian_filter() has no ksize argument
            # TODO it does have a truncate argument that truncates at x standard deviations -- maybe can be used
            #      similarly to ksize
            if ksize is not None:
                warnings.warn("Requested 'scipy' backend or picked it automatically by backend='auto' "
                              "in blur_gaussian_(), but also provided 'ksize' argument, which is not understood by "
                              "that backend and will be ignored.")

            # Note that while gaussian_filter can be applied to all channels at the same time, that should not
            # be done here, because then the blurring would also happen across channels (e.g. red values might
            # be mixed with blue values in RGB)
            if image.ndim == 2:
                image[:, :] = ndimage.gaussian_filter(image[:, :], sigma, mode="mirror")
            else:
                nb_channels = image.shape[2]
                for channel in sm.xrange(nb_channels):
                    image[:, :, channel] = ndimage.gaussian_filter(image[:, :, channel], sigma, mode="mirror")
        else:
            if dtype.name == "bool":
                image = image.astype(np.float32, copy=False)
            elif dtype.name == "float16":
                image = image.astype(np.float32, copy=False)
            elif dtype.name == "int8":
                image = image.astype(np.int16, copy=False)
            elif dtype.name == "int32":
                image = image.astype(np.float64, copy=False)

            # ksize here is derived from the equation to compute sigma based on ksize,
            # see https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html -> cv::getGaussianKernel()
            # example values:
            #   sig = 0.1 -> ksize = -1.666
            #   sig = 0.5 -> ksize = 0.9999
            #   sig = 1.0 -> ksize = 1.0
            #   sig = 2.0 -> ksize = 11.0
            #   sig = 3.0 -> ksize = 17.666
            # ksize = ((sig - 0.8)/0.3 + 1)/0.5 + 1

            if ksize is None:
                if sigma < 3.0:
                    ksize = 3.3 * sigma  # 99% of weight
                elif sigma < 5.0:
                    ksize = 2.9 * sigma  # 97% of weight
                else:
                    ksize = 2.6 * sigma  # 95% of weight

                # we use 5x5 here as the minimum size as that simplifies comparisons with gaussian_filter() in the tests
                # TODO reduce this to 3x3
                ksize = int(max(ksize, 5))
            else:
                assert ia.is_single_integer(ksize), "Expected 'ksize' argument to be a number, got %s." % (type(ksize),)

            ksize = ksize + 1 if ksize % 2 == 0 else ksize

            if ksize > 0:
                image_warped = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                                                borderType=cv2.BORDER_REFLECT_101)

                # re-add channel axis removed by cv2 if input was (H, W, 1)
                image = image_warped[..., np.newaxis] if image.ndim == 3 and image_warped.ndim == 2 else image_warped

        if dtype.name == "bool":
            image = image > 0.5
        elif dtype.name != image.dtype.name:
            image = iadt.restore_dtypes_(image, dtype)

    return image


# TODO offer different values for sigma on x/y-axis, supported by cv2 but not by scipy
# TODO add channelwise flag - channelwise=False would be supported by scipy
class GaussianBlur(meta.Augmenter):  # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Augmenter to blur images using gaussian kernels.

    dtype support::

        See :func:`imgaug.augmenters.blur.blur_gaussian_(backend="auto")`.

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
        self.eps = 1e-3  # epsilon value to estimate whether sigma is sufficently above 0 to apply the blur

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.sigma.draw_samples((nb_images,), random_state=random_state)
        for image, sig in zip(images, samples):
            image[...] = blur_gaussian_(image, sigma=sig, eps=self.eps)
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

    The padding behaviour around the image borders is cv2's ``BORDER_REFLECT_101``.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (3)
        * ``int16``: yes; tested
        * ``int32``: no (4)
        * ``int64``: no (5)
        * ``float16``: yes; tested (6)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no
        * ``bool``: yes; tested (7)

        - (1) rejected by ``cv2.blur()``
        - (2) loss of resolution in ``cv2.blur()`` (result is ``int32``)
        - (3) ``int8`` is mapped internally to ``int16``, ``int8`` itself leads to cv2 error "Unsupported combination
              of source format (=1), and buffer format (=4) in function 'getRowSumFilter'" in ``cv2``
        - (4) results too inaccurate
        - (5) loss of resolution in ``cv2.blur()`` (result is ``int32``)
        - (6) ``float16`` is mapped internally to ``float32``
        - (7) ``bool`` is mapped internally to ``float32``

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

        # TODO replace this by iap.handle_discrete_kernel_size()
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
        iadt.gate_dtypes(images,
                         allowed=["bool", "uint8", "uint16", "int8", "int16", "float16", "float32", "float64"],
                         disallowed=["uint32", "uint64", "uint128", "uint256",
                                     "int32", "int64", "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=self)

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
                input_dtype = image.dtype
                if image.dtype in [np.bool_, np.float16]:
                    image = image.astype(np.float32, copy=False)
                elif image.dtype == np.int8:
                    image = image.astype(np.int16, copy=False)

                image_aug = cv2.blur(image, (kh, kw))
                # cv2.blur() removes channel axis for single-channel images
                if image_aug.ndim == 2:
                    image_aug = image_aug[..., np.newaxis]

                if input_dtype == np.bool_:
                    image_aug = image_aug > 0.5
                elif input_dtype in [np.int8, np.float16]:
                    image_aug = iadt.restore_dtypes_(image_aug, input_dtype)

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

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

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

        # TODO replace this by iap.handle_discrete_kernel_size()
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
                # cv2.medianBlur() removes channel axis for single-channel images
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

    dtype support::

        * ``uint8``: yes; not tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

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
        # Make sure that all images have 3 channels
        ia.do_assert(all([image.shape[2] == 3 for image in images]),
                     ("BilateralBlur can currently only be applied to images with 3 channels."
                      + "Got channels: %s") % ([image.shape[2] for image in images],))

        nb_images = len(images)
        rss = ia.derive_random_states(random_state, 3)
        samples_d = self.d.draw_samples((nb_images,), random_state=rss[0])
        samples_sigma_color = self.sigma_color.draw_samples((nb_images,), random_state=rss[1])
        samples_sigma_space = self.sigma_space.draw_samples((nb_images,), random_state=rss[2])
        gen = enumerate(zip(images, samples_d, samples_sigma_color, samples_sigma_space))
        for i, (image, di, sigma_color_i, sigma_space_i) in gen:
            if di != 1:
                images[i] = cv2.bilateralFilter(image, di, sigma_color_i, sigma_space_i)
        return images

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

    dtype support::

        See ``imgaug.augmenters.convolutional.Convolve``.

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
    # TODO allow (1, None) and set to identity matrix if k == 1
    k_param = iap.handle_discrete_param(k, "k", value_range=(3, None), tuple_to_uniform=True, list_to_choice=True,
                                        allow_floats=False)
    angle_param = iap.handle_continuous_param(angle, "angle", value_range=None, tuple_to_uniform=True,
                                              list_to_choice=True)
    direction_param = iap.handle_continuous_param(direction, "direction", value_range=(-1.0-1e-6, 1.0+1e-6),
                                                  tuple_to_uniform=True, list_to_choice=True)

    def create_matrices(image, nb_channels, random_state_func):
        # avoid cyclic import between blur and geometric
        from . import geometric as iaa_geometric

        # force discrete for k_sample via int() in case of stochastic parameter
        k_sample = int(k_param.draw_sample(random_state=random_state_func))
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
