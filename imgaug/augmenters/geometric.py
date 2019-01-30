"""
Augmenters that apply affine transformations or other similar augmentations.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([
        iaa.Affine(...),
        iaa.PerspectiveTransform(...)
    ])

List of augmenters:
    * Affine
    * AffineCv2
    * PiecewiseAffine
    * PerspectiveTransform
    * ElasticTransformation
    * Rot90

"""
from __future__ import print_function, division, absolute_import

import math
from functools import partial

import numpy as np
from scipy import ndimage
from skimage import transform as tf
import cv2
import six.moves as sm

from . import meta
from . import blur as blur_lib
from .. import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt


class Affine(meta.Augmenter):
    """
    Augmenter to apply affine transformations to images.

    This is mostly a wrapper around skimage's AffineTransform class and warp function.

    Affine transformations involve:

        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a
    defined content, e.g. if the image is translated to the left, pixels
    are created on the right.
    A method has to be defined to deal with these pixel values. The
    parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameter `order`
    deals with the method of interpolation used for this.

    dtype support::

        if (backend="skimage", order in [0, 1])::

            * ``uint8``: yes; tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested (1)
            * ``uint64``: no (2)
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested  (1)
            * ``int64``: no (2)
            * ``float16``: yes; tested
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: no (2)
            * ``bool``: yes; tested

            - (1) scikit-image converts internally to float64, which might affect the accuracy of
                  large integers. In tests this seemed to not be an issue.
            - (2) results too inaccurate

        if (backend="skimage", order in [3, 4])::

            * ``uint8``: yes; tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested (1)
            * ``uint64``: no (2)
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested  (1)
            * ``int64``: no (2)
            * ``float16``: yes; tested
            * ``float32``: yes; tested
            * ``float64``: limited; tested (3)
            * ``float128``: no (2)
            * ``bool``: yes; tested

            - (1) scikit-image converts internally to float64, which might affect the accuracy of
                  large integers. In tests this seemed to not be an issue.
            - (2) results too inaccurate
            - (3) ``NaN`` around minimum and maximum of float64 value range

        if (backend="skimage", order=5])::

                * ``uint8``: yes; tested
                * ``uint16``: yes; tested
                * ``uint32``: yes; tested (1)
                * ``uint64``: no (2)
                * ``int8``: yes; tested
                * ``int16``: yes; tested
                * ``int32``: yes; tested  (1)
                * ``int64``: no (2)
                * ``float16``: yes; tested
                * ``float32``: yes; tested
                * ``float64``: limited; not tested (3)
                * ``float128``: no (2)
                * ``bool``: yes; tested

                - (1) scikit-image converts internally to float64, which might affect the accuracy of
                      large integers. In tests this seemed to not be an issue.
                - (2) results too inaccurate
                - (3) ``NaN`` around minimum and maximum of float64 value range

        if (backend="cv2", order=0)::

            * ``uint8``: yes; tested
            * ``uint16``: yes; tested
            * ``uint32``: no (1)
            * ``uint64``: no (2)
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested
            * ``int64``: no (2)
            * ``float16``: yes; tested (3)
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: no (1)
            * ``bool``: yes; tested (3)

            - (1) rejected by cv2
            - (2) changed to ``int32`` by cv2
            - (3) mapped internally to ``float32``

        if (backend="cv2", order=1):

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: no (1)
            * ``uint64``: no (2)
            * ``int8``: yes; tested (3)
            * ``int16``: yes; tested
            * ``int32``: no (2)
            * ``int64``: no (2)
            * ``float16``: yes; tested (4)
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: no (1)
            * ``bool``: yes; tested (4)

            - (1) rejected by cv2
            - (2) causes cv2 error: ``cv2.error: OpenCV(3.4.4) (...)imgwarp.cpp:1805: error:
                  (-215:Assertion failed) ifunc != 0 in function 'remap'``
            - (3) mapped internally to ``int16``
            - (4) mapped internally to ``float32``

        if (backend="cv2", order=3):

            * ``uint8``: yes; tested
            * ``uint16``: yes; tested
            * ``uint32``: no (1)
            * ``uint64``: no (2)
            * ``int8``: yes; tested (3)
            * ``int16``: yes; tested
            * ``int32``: no (2)
            * ``int64``: no (2)
            * ``float16``: yes; tested (4)
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: no (1)
            * ``bool``: yes; tested (4)

            - (1) rejected by cv2
            - (2) causes cv2 error: ``cv2.error: OpenCV(3.4.4) (...)imgwarp.cpp:1805: error:
                  (-215:Assertion failed) ifunc != 0 in function 'remap'``
            - (3) mapped internally to ``int16``
            - (4) mapped internally to ``float32``


    Parameters
    ----------
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter\
            or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter},\
            optional
        Scaling factor to use,
        where 1.0 represents no change and 0.5 is zoomed out to 50 percent of the original size.

            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled from the range
              ``a <= x <= b`` per image. That value will be used identically for
              both x- and y-axis.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`scale`). Using a dictionary allows to
              set different values for the axis. If they are set to the same
              ranges, different values may still be sampled per axis.

    translate_percent : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter or\
                        dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter},\
                        optional
        Translation in percent relative to the image height/width (x-translation, y-translation) to use,
        where 0 represents no change and 0.5 is half of the image height/width.

            * If None then equivalent to 0 unless translate_px has a non-None value.
            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled from the range
              ``a <= x <= b`` per image. That percent value will be used identically
              for both x- and y-axis.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`translate_percent`).
              Using a dictionary allows to set different values for the axis.
              If they are set to the same ranges, different values may still
              be sampled per axis.

    translate_px : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter or\
                   dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter},\
                   optional
        Translation in pixels.

            * If None then equivalent to 0.0 unless translate_percent has a non-None value.
            * If a single int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled from the discrete
              range ``[a..b]`` per image. That number will be used identically
              for both x- and y-axis.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`translate_px`).
              Using a dictionary allows to set different values for the axis.
              If they are set to the same ranges, different values may still
              be sampled per axis.

    rotate : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Rotation in degrees (_NOT_ radians), i.e. expected value range is
        0 to 360 for positive rotations (may also be negative). Rotation
        happens around the _center_ of the image, not the top left corner
        as in some other frameworks.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled per image from the
              range ``a <= x <= b`` and be used as the rotation value.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used to
              sample the rotation value per image.

    shear : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Shear in degrees (_NOT_ radians), i.e. expected value range is
        0 to 360 for positive shear (may also be negative).

            * If a float/int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled per image from the
              range ``a <= x <= b`` and be used as the rotation value.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used to
              sample the shear value per image.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use. Same meaning as in skimage:

            * ``0``: ``Nearest-neighbor``
            * ``1``: ``Bi-linear`` (default)
            * ``2``: ``Bi-quadratic`` (not recommended by skimage)
            * ``3``: ``Bi-cubic``
            * ``4``: ``Bi-quartic``
            * ``5``: ``Bi-quintic``

        Method 0 and 1 are fast, 3 is a bit slower, 4 and 5 are very slow.
        If the backend is ``cv2``, the mapping to OpenCV's interpolation modes
        is as follows:

            * ``0`` -> ``cv2.INTER_NEAREST``
            * ``1`` -> ``cv2.INTER_LINEAR``
            * ``2`` -> ``cv2.INTER_CUBIC``
            * ``3`` -> ``cv2.INTER_CUBIC``
            * ``4`` -> ``cv2.INTER_CUBIC``

        As datatypes this parameter accepts:

            * If a single int, then that order will be used for all images.
            * If an iterable, then for each image a random value will be sampled
              from that iterable (i.e. list of allowed order values).
            * If imgaug.ALL, then equivalant to list ``[0, 1, 3, 4, 5]`` in case of backend ``skimage``
              and otherwise ``[0, 1, 3]``.
            * If StochasticParameter, then that parameter is queried per image
              to sample the order value to use.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant value used for skimage's transform function.
        This is the value used to fill up pixels in the result image that
        didn't exist in the input image (e.g. when translating to the left,
        some new pixels are created at the right). Such a fill-up with a
        constant value only happens, when `mode` is "constant".
        The expected value range is ``[0, 255]``. It may be a float value.

            * If this is a single number, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple ``(a, b)``, then a random value from the range ``a <= x <= b``
              is picked per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If imgaug.ALL, a value from the discrete range ``[0 .. 255]`` will be
              sampled per image.
            * If a StochasticParameter, a new value will be sampled from the
              parameter per image.

    fit_output : bool, optional
        Whether the image after affine transformation is completely contained in the output image.
        If False, parts of the image may be outside of the image plane or the image might make up only a small
        part of the image plane. Activating this can be useful e.g. for rotations by 45 degrees to avoid that the
        image corners are outside of the image plane.
        Note that activating this will negate translation.
        Note also that activating this may lead to image sizes differing from the input image sizes. To avoid this,
        wrap ``Affine`` in ``KeepSizeByResize``, e.g. ``KeepSizeByResize(Affine(...))``.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Parameter that defines the handling of newly created pixels.
        Same meaning as in skimage (and numpy.pad):

            * ``constant``: Pads with a constant value
            * ``edge``: Pads with the edge values of array
            * ``symmetric``: Pads with the reflection of the vector mirrored
              along the edge of the array.
            * ``reflect``: Pads with the reflection of the vector mirrored on
              the first and last values of the vector along each axis.
            * ``wrap``: Pads with the wrap of the vector along the axis.
              The first values are used to pad the end and the end values
              are used to pad the beginning.

        If ``cv2`` is chosen as the backend the mapping is as follows:

            * ``constant`` -> ``cv2.BORDER_CONSTANT``
            * ``edge`` -> ``cv2.BORDER_REPLICATE``
            * ``symmetric`` -> ``cv2.BORDER_REFLECT``
            * ``reflect`` -> ``cv2.BORDER_REFLECT_101``
            * ``wrap`` -> ``cv2.BORDER_WRAP``

        The datatype of the parameter may be:

            * If a single string, then that mode will be used for all images.
            * If a list of strings, then per image a random mode will be picked
              from that list.
            * If imgaug.ALL, then a random mode from all possible modes will be
              picked.
            * If StochasticParameter, then the mode will be sampled from that
              parameter per image, i.e. it must return only the above mentioned
              strings.

    backend : str, optional
        Framework to use as a backend. Valid values are ``auto``, ``skimage``
        (scikit-image's warp) and ``cv2`` (OpenCV's warp).
        If ``auto`` is used, the augmenter will automatically try
        to use ``cv2`` where possible (order must be in ``[0, 1, 3]`` and
        image's dtype uint8, otherwise skimage is chosen). It will
        silently fall back to skimage if order/dtype is not supported by cv2.
        cv2 is generally faster than skimage. It also supports RGB cvals,
        while skimage will resort to intensity cvals (i.e. 3x the same value
        as RGB). If ``cv2`` is chosen and order is 2 or 4, it will automatically
        fall back to order 3.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Affine(scale=2.0)

    zooms all images by a factor of 2.

    >>> aug = iaa.Affine(translate_px=16)

    translates all images on the x- and y-axis by 16 pixels (to the
    right/top), fills up any new pixels with zero (black values).

    >>> aug = iaa.Affine(translate_percent=0.1)

    translates all images on the x- and y-axis by 10 percent of their
    width/height (to the right/top), fills up any new pixels with zero
    (black values).

    >>> aug = iaa.Affine(rotate=35)

    rotates all images by 35 degrees, fills up any new pixels with zero
    (black values).

    >>> aug = iaa.Affine(shear=15)

    rotates all images by 15 degrees, fills up any new pixels with zero
    (black values).

    >>> aug = iaa.Affine(translate_px=(-16, 16))

    translates all images on the x- and y-axis by a random value
    between -16 and 16 pixels (to the right/top) (same for both axis, i.e.
    sampled once per image), fills up any new pixels with zero (black values).

    >>> aug = iaa.Affine(translate_px={"x": (-16, 16), "y": (-4, 4)})

    translates all images on the x-axis by a random value
    between -16 and 16 pixels (to the right) and on the y-axis by a
    random value between -4 and 4 pixels to the top. Even if both ranges
    were the same, both axis could use different samples.
    Fills up any new pixels with zero (black values).

    >>> aug = iaa.Affine(scale=2.0, order=[0, 1])

    same as previously, but uses (randomly) either nearest neighbour
    interpolation or linear interpolation.

    >>> aug = iaa.Affine(translate_px=16, cval=(0, 255))

    same as previously, but fills up any new pixels with a random
    brightness (same for the whole image).

    >>> aug = iaa.Affine(translate_px=16, mode=["constant", "edge"])

    same as previously, but fills up the new pixels in only 50 percent
    of all images with black values. In the other 50 percent of all cases,
    the value of the nearest edge is used.

    """
    VALID_DTYPES_CV2_ORDER_0 = {"uint8", "uint16", "int8", "int16", "int32",
                                "float16", "float32", "float64",
                                "bool"}
    VALID_DTYPES_CV2_ORDER_NOT_0 = {"uint8", "uint16", "int8", "int16",
                                    "float16", "float32", "float64",
                                    "bool"}

    def __init__(self, scale=1.0, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0, order=1, cval=0,
                 mode="constant", fit_output=False, backend="auto", name=None, deterministic=False, random_state=None):
        super(Affine, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        ia.do_assert(backend in ["auto", "skimage", "cv2"])
        self.backend = backend

        # skimage | cv2
        # 0       | cv2.INTER_NEAREST
        # 1       | cv2.INTER_LINEAR
        # 2       | -
        # 3       | cv2.INTER_CUBIC
        # 4       | -
        self.order_map_skimage_cv2 = {
            0: cv2.INTER_NEAREST,
            1: cv2.INTER_LINEAR,
            2: cv2.INTER_CUBIC,
            3: cv2.INTER_CUBIC,
            4: cv2.INTER_CUBIC
        }
        # Peformance in skimage:
        #  1.0x order 0
        #  1.5x order 1
        #  3.0x order 3
        # 30.0x order 4
        # 60.0x order 5
        # measurement based on 256x256x3 batches, difference is smaller
        # on smaller images (seems to grow more like exponentially with image
        # size)
        if order == ia.ALL:
            if backend == "auto" or backend == "cv2":
                self.order = iap.Choice([0, 1, 3])
            else:
                # dont use order=2 (bi-quadratic) because that is apparently currently not recommended (and throws
                # a warning)
                self.order = iap.Choice([0, 1, 3, 4, 5])
        elif ia.is_single_integer(order):
            ia.do_assert(0 <= order <= 5,
                         "Expected order's integer value to be in range 0 <= x <= 5, got %d." % (order,))
            if backend == "cv2":
                ia.do_assert(order in [0, 1, 3])
            self.order = iap.Deterministic(order)
        elif isinstance(order, list):
            ia.do_assert(all([ia.is_single_integer(val) for val in order]),
                         "Expected order list to only contain integers, got types %s." % (
                             str([type(val) for val in order]),))
            ia.do_assert(all([0 <= val <= 5 for val in order]),
                         "Expected all of order's integer values to be in range 0 <= x <= 5, got %s." % (str(order),))
            if backend == "cv2":
                ia.do_assert(all([val in [0, 1, 3] for val in order]))
            self.order = iap.Choice(order)
        elif isinstance(order, iap.StochasticParameter):
            self.order = order
        else:
            raise Exception("Expected order to be imgaug.ALL, int, list of int or StochasticParameter, got %s." % (
                type(order),))

        if cval == ia.ALL:
            # TODO change this so that it is dynamically created per image (or once per dtype)
            self.cval = iap.Uniform(0, 255)  # skimage transform expects float
        else:
            self.cval = iap.handle_continuous_param(cval, "cval", value_range=None, tuple_to_uniform=True,
                                                    list_to_choice=True)

        # constant, edge, symmetric, reflect, wrap
        # skimage   | cv2
        # constant  | cv2.BORDER_CONSTANT
        # edge      | cv2.BORDER_REPLICATE
        # symmetric | cv2.BORDER_REFLECT
        # reflect   | cv2.BORDER_REFLECT_101
        # wrap      | cv2.BORDER_WRAP
        self.mode_map_skimage_cv2 = {
            "constant": cv2.BORDER_CONSTANT,
            "edge": cv2.BORDER_REPLICATE,
            "symmetric": cv2.BORDER_REFLECT,
            "reflect": cv2.BORDER_REFLECT_101,
            "wrap": cv2.BORDER_WRAP
        }
        if mode == ia.ALL:
            self.mode = iap.Choice(["constant", "edge", "symmetric", "reflect", "wrap"])
        elif ia.is_string(mode):
            self.mode = iap.Deterministic(mode)
        elif isinstance(mode, list):
            ia.do_assert(all([ia.is_string(val) for val in mode]))
            self.mode = iap.Choice(mode)
        elif isinstance(mode, iap.StochasticParameter):
            self.mode = mode
        else:
            raise Exception("Expected mode to be imgaug.ALL, a string, a list of strings or StochasticParameter, "
                            + "got %s." % (type(mode),))

        # scale
        if isinstance(scale, dict):
            ia.do_assert("x" in scale or "y" in scale)
            x = scale.get("x", 1.0)
            y = scale.get("y", 1.0)
            self.scale = (
                iap.handle_continuous_param(x, "scale['x']", value_range=(0+1e-4, None), tuple_to_uniform=True,
                                            list_to_choice=True),
                iap.handle_continuous_param(y, "scale['y']", value_range=(0+1e-4, None), tuple_to_uniform=True,
                                            list_to_choice=True)
            )
        else:
            self.scale = iap.handle_continuous_param(scale, "scale", value_range=(0+1e-4, None), tuple_to_uniform=True,
                                                     list_to_choice=True)

        # translate
        if translate_percent is None and translate_px is None:
            translate_px = 0

        ia.do_assert(translate_percent is None or translate_px is None)

        if translate_percent is not None:
            # translate by percent
            if isinstance(translate_percent, dict):
                ia.do_assert("x" in translate_percent or "y" in translate_percent)
                x = translate_percent.get("x", 0)
                y = translate_percent.get("y", 0)
                self.translate = (
                    iap.handle_continuous_param(x, "translate_percent['x']", value_range=None, tuple_to_uniform=True,
                                                list_to_choice=True),
                    iap.handle_continuous_param(y, "translate_percent['y']", value_range=None, tuple_to_uniform=True,
                                                list_to_choice=True)
                )
            else:
                self.translate = iap.handle_continuous_param(translate_percent, "translate_percent", value_range=None,
                                                             tuple_to_uniform=True, list_to_choice=True)
        else:
            # translate by pixels
            if isinstance(translate_px, dict):
                ia.do_assert("x" in translate_px or "y" in translate_px)
                x = translate_px.get("x", 0)
                y = translate_px.get("y", 0)
                self.translate = (
                    iap.handle_discrete_param(x, "translate_px['x']", value_range=None, tuple_to_uniform=True,
                                              list_to_choice=True, allow_floats=False),
                    iap.handle_discrete_param(y, "translate_px['y']", value_range=None, tuple_to_uniform=True,
                                              list_to_choice=True, allow_floats=False)
                )
            else:
                self.translate = iap.handle_discrete_param(translate_px, "translate_px", value_range=None,
                                                           tuple_to_uniform=True, list_to_choice=True,
                                                           allow_floats=False)

        self.rotate = iap.handle_continuous_param(rotate, "rotate", value_range=None, tuple_to_uniform=True,
                                                  list_to_choice=True)
        self.shear = iap.handle_continuous_param(shear, "shear", value_range=None, tuple_to_uniform=True,
                                                 list_to_choice=True)
        self.fit_output = fit_output

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        scale_samples, translate_samples, rotate_samples, shear_samples, \
            cval_samples, mode_samples, order_samples = self._draw_samples(nb_images, random_state)

        result = self._augment_images_by_samples(images, scale_samples, translate_samples, rotate_samples,
                                                 shear_samples, cval_samples, mode_samples, order_samples)

        return result

    def _augment_images_by_samples(self, images, scale_samples, translate_samples, rotate_samples, shear_samples,
                                   cval_samples, mode_samples, order_samples, return_matrices=False):
        nb_images = len(images)
        input_was_array = ia.is_np_array(images)
        input_dtype = None if not input_was_array else images.dtype
        result = []
        if return_matrices:
            matrices = [None] * nb_images
        for i in sm.xrange(nb_images):
            image = images[i]

            min_value, _center_value, max_value = iadt.get_value_range_of_dtype(image.dtype)

            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x, translate_y = translate_samples[0][i], translate_samples[1][i]
            if ia.is_single_float(translate_y):
                translate_y_px = int(np.round(translate_y * images[i].shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(np.round(translate_x * images[i].shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            cval = cval_samples[i]
            mode = mode_samples[i]
            order = order_samples[i]
            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 \
                    or shear != 0:
                cv2_bad_order = order not in [0, 1, 3]
                if order == 0:
                    cv2_bad_dtype = image.dtype.name not in self.VALID_DTYPES_CV2_ORDER_0
                else:
                    cv2_bad_dtype = image.dtype.name not in self.VALID_DTYPES_CV2_ORDER_NOT_0
                cv2_bad_shape = image.shape[2] > 4
                cv2_impossible = cv2_bad_order or cv2_bad_dtype or cv2_bad_shape
                if self.backend == "skimage" or (self.backend == "auto" and cv2_impossible):
                    # cval contains 3 values as cv2 can handle 3, but skimage only 1
                    cval = cval[0]
                    # skimage does not clip automatically
                    cval = max(min(cval, max_value), min_value)
                    image_warped = self._warp_skimage(
                        image,
                        scale_x, scale_y,
                        translate_x_px, translate_y_px,
                        rotate, shear,
                        cval,
                        mode, order,
                        self.fit_output,
                        return_matrix=return_matrices,
                    )
                else:
                    ia.do_assert(not cv2_bad_dtype,
                                 "cv2 backend can only handle images of dtype uint8, float32 and float64, got %s." % (
                                     image.dtype,))
                    image_warped = self._warp_cv2(
                        image,
                        scale_x, scale_y,
                        translate_x_px, translate_y_px,
                        rotate, shear,
                        tuple([int(v) for v in cval]),
                        self.mode_map_skimage_cv2[mode],
                        self.order_map_skimage_cv2[order],
                        self.fit_output,
                        return_matrix=return_matrices,
                    )
                if return_matrices:
                    image_warped, matrix = image_warped
                    matrices[i] = matrix

                result.append(image_warped)
            else:
                result.append(images[i])

        # the shapes can change due to fit_output, then it may not be possible to return an array, even when the input
        # was an array
        if input_was_array:
            nb_shapes = len(set([image.shape for image in result]))
            if nb_shapes == 1:
                result = np.array(result, input_dtype)

        if return_matrices:
            result = (result, matrices)

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        nb_heatmaps = len(heatmaps)
        scale_samples, translate_samples, rotate_samples, shear_samples, \
            cval_samples, mode_samples, order_samples = self._draw_samples(nb_heatmaps, random_state)
        cval_samples = np.zeros((cval_samples.shape[0], 1), dtype=np.float32)
        mode_samples = ["constant"] * len(mode_samples)

        arrs = [heatmaps_i.arr_0to1 for heatmaps_i in heatmaps]
        arrs_aug, matrices = self._augment_images_by_samples(arrs, scale_samples, translate_samples, rotate_samples,
                                                             shear_samples, cval_samples, mode_samples, order_samples,
                                                             return_matrices=True)
        for heatmaps_i, arr_aug, matrix in zip(heatmaps, arrs_aug, matrices):
            heatmaps_i.arr_0to1 = arr_aug
            if self.fit_output:
                _, output_shape_i = self._tf_to_fit_output(heatmaps_i.shape, matrix)
            else:
                output_shape_i = heatmaps_i.shape
            heatmaps_i.shape = output_shape_i
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        scale_samples, translate_samples, rotate_samples, shear_samples, \
            _cval_samples, _mode_samples, _order_samples = self._draw_samples(nb_images, random_state)

        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if not keypoints_on_image.keypoints:
                result.append(keypoints_on_image)
                continue
            height, width = keypoints_on_image.height, keypoints_on_image.width
            shift_x = width / 2.0 - 0.5
            shift_y = height / 2.0 - 0.5
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x, translate_y = translate_samples[0][i], translate_samples[1][i]
            if ia.is_single_float(translate_y):
                translate_y_px = int(np.round(translate_y * keypoints_on_image.shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(np.round(translate_x * keypoints_on_image.shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 \
                    or shear != 0:
                matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x_px, translate_y_px),
                    rotation=math.radians(rotate),
                    shear=math.radians(shear)
                )
                matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
                matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center)
                if self.fit_output:
                    matrix, output_shape = self._tf_to_fit_output(keypoints_on_image.shape, matrix)
                else:
                    output_shape = keypoints_on_image.shape

                coords = keypoints_on_image.get_coords_array()
                coords_aug = tf.matrix_transform(coords, matrix.params)
                result.append(ia.KeypointsOnImage.from_coords_array(coords_aug, shape=output_shape))
            else:
                result.append(keypoints_on_image)
        return result

    def get_parameters(self):
        return [self.scale, self.translate, self.rotate, self.shear, self.order, self.cval, self.mode, self.backend,
                self.fit_output]

    def _draw_samples(self, nb_samples, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]

        if isinstance(self.scale, tuple):
            scale_samples = (
                self.scale[0].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 10)),
                self.scale[1].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 20)),
            )
        else:
            scale_samples = self.scale.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 30))
            scale_samples = (scale_samples, scale_samples)

        if isinstance(self.translate, tuple):
            translate_samples = (
                self.translate[0].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 40)),
                self.translate[1].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 50)),
            )
        else:
            translate_samples = self.translate.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 60))
            translate_samples = (translate_samples, translate_samples)

        rotate_samples = self.rotate.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 70))
        shear_samples = self.shear.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 80))

        cval_samples = self.cval.draw_samples((nb_samples, 3), random_state=ia.new_random_state(seed + 90))
        mode_samples = self.mode.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 100))
        order_samples = self.order.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 110))

        return scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, mode_samples, \
            order_samples

    @classmethod
    def _tf_to_fit_output(cls, input_shape, matrix):
        height, width = input_shape[:2]
        # determine shape of output image
        corners = np.array([
            [0, 0],
            [0, height - 1],
            [width - 1, height - 1],
            [width - 1, 0]
        ])
        corners = matrix(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_height = maxr - minr + 1
        out_width = maxc - minc + 1
        if len(input_shape) == 3:
            output_shape = np.ceil((out_height, out_width, input_shape[2]))
        else:
            output_shape = np.ceil((out_height, out_width))
        output_shape = tuple([int(v) for v in output_shape.tolist()])
        # fit output image in new shape
        translation = (-minc, -minr)
        matrix_to_fit = tf.SimilarityTransform(translation=translation)
        matrix = matrix + matrix_to_fit
        return matrix, output_shape

    def _warp_skimage(self, image, scale_x, scale_y, translate_x_px, translate_y_px, rotate, shear, cval, mode, order,
                      fit_output, return_matrix=False):
        iadt.gate_dtypes(image,
                         allowed=["bool", "uint8", "uint16", "uint32", "int8", "int16", "int32",
                                  "float16", "float32", "float64"],
                         disallowed=["uint64", "uint128", "uint256", "int64", "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=self)

        input_dtype = image.dtype

        height, width = image.shape[0], image.shape[1]
        shift_x = width / 2.0 - 0.5
        shift_y = height / 2.0 - 0.5

        matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
        matrix_transforms = tf.AffineTransform(
            scale=(scale_x, scale_y),
            translation=(translate_x_px, translate_y_px),
            rotation=math.radians(rotate),
            shear=math.radians(shear)
        )
        matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
        matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center)

        output_shape = None
        if fit_output:
            matrix, output_shape = self._tf_to_fit_output(image.shape, matrix)

        image_warped = tf.warp(
            image,
            matrix.inverse,
            order=order,
            mode=mode,
            cval=cval,
            preserve_range=True,
            output_shape=output_shape,
        )

        # tf.warp changes all dtypes to float64, including uint8
        if input_dtype == np.bool_:
            image_warped = image_warped > 0.5
        else:
            image_warped = iadt.restore_dtypes_(image_warped, input_dtype)

        if return_matrix:
            return image_warped, matrix
        return image_warped

    def _warp_cv2(self, image, scale_x, scale_y, translate_x_px, translate_y_px, rotate, shear, cval, mode, order,
                  fit_output, return_matrix=False):
        iadt.gate_dtypes(image,
                         allowed=["bool", "uint8", "uint16", "int8", "int16", "int32",
                                  "float16", "float32", "float64"],
                         disallowed=["uint32", "uint64", "uint128", "uint256",
                                     "int64", "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=self)
        if order != 0:
            ia.do_assert(image.dtype != np.int32,
                         ("Affine only supports cv2-based transformations of int32 arrays when "
                          + "using order=0, but order was set to %d.") % (order,))

        input_dtype = image.dtype
        if input_dtype in [np.bool_, np.float16]:
            image = image.astype(np.float32)
        elif input_dtype == np.int8 and order != 0:
            image = image.astype(np.int16)

        height, width = image.shape[0], image.shape[1]
        shift_x = width / 2.0 - 0.5
        shift_y = height / 2.0 - 0.5

        matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
        matrix_transforms = tf.AffineTransform(
            scale=(scale_x, scale_y),
            translation=(translate_x_px, translate_y_px),
            rotation=math.radians(rotate),
            shear=math.radians(shear)
        )
        matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
        matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center)

        dsize = (width, height)
        if fit_output:
            matrix, output_shape = self._tf_to_fit_output(image.shape, matrix)
            dsize = (int(np.round(output_shape[1])), int(np.round(output_shape[0])))

        # TODO this uses always a tuple of 3 values for cval, even if #chans != 3, works with 1d but what in other
        # cases?
        image_warped = cv2.warpAffine(
            image,
            matrix.params[:2],
            dsize=dsize,
            flags=order,
            borderMode=mode,
            borderValue=cval
        )

        # cv2 warp drops last axis if shape is (H, W, 1)
        if image_warped.ndim == 2:
            image_warped = image_warped[..., np.newaxis]

        if input_dtype == np.bool_:
            image_warped = image_warped > 0.5
        elif input_dtype in [np.int8, np.float16]:
            image_warped = iadt.restore_dtypes_(image_warped, input_dtype)

        if return_matrix:
            return image_warped, matrix
        return image_warped


class AffineCv2(meta.Augmenter):
    """
    Augmenter to apply affine transformations to images using cv2 (i.e. opencv)
    backend.

    NOTE: This augmenter will likely be removed in the future as Affine() already
    offers a cv2 backend (use `backend="cv2"`).

    Affine transformations
    involve:

        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a
    defined content, e.g. if the image is translated to the left, pixels
    are created on the right.
    A method has to be defined to deal with these pixel values. The
    parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameter `order`
    deals with the method of interpolation used for this.

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
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or\
            dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter},\
            optional
        Scaling factor to use,
        where 1.0 represents no change and 0.5 is zoomed out to 50 percent of the original size.

            * If a single float, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled from the range
              ``a <= x <= b`` per image. That value will be used identically for
              both x- and y-axis.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`scale`). Using a dictionary allows to
              set different values for the axis. If they are set to the same
              ranges, different values may still be sampled per axis.

    translate_percent : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or\
                        dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter},\
                        optional
        Translation in percent relative to the image
        height/width (x-translation, y-translation) to use,
        where 0 represents no change and 0.5 is half of the image
        height/width.

            * If a single float, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled from the range
              ``a <= x <= b`` per image. That percent value will be used identically
              for both x- and y-axis.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`translate_percent`).
              Using a dictionary allows to set different values for the axis.
              If they are set to the same ranges, different values may still
              be sampled per axis.

    translate_px : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or\
                   dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter},\
                   optional
        Translation in pixels.

            * If a single int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled from the discrete
              range ``[a..b]`` per image. That number will be used identically
              for both x- and y-axis.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`translate_px`).
              Using a dictionary allows to set different values for the axis.
              If they are set to the same ranges, different values may still
              be sampled per axis.

    rotate : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Rotation in degrees (_NOT_ radians), i.e. expected value range is
        0 to 360 for positive rotations (may also be negative). Rotation
        happens around the _center_ of the image, not the top left corner
        as in some other frameworks.

            * If a float/int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled per image from the
              range ``a <= x <= b`` and be used as the rotation value.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used to
              sample the rotation value per image.

    shear : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Shear in degrees (NOT radians), i.e. expected value range is
        0 to 360 for positive shear (may also be negative).

            * If a float/int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled per image from the
              range ``a <= x <= b`` and be used as the rotation value.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If a StochasticParameter, then this parameter will be used to
              sample the shear value per image.

    order : int or list of int or str or list of str or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use. Allowed are:

            * ``cv2.INTER_NEAREST`` - a nearest-neighbor interpolation
            * ``cv2.INTER_LINEAR`` - a bilinear interpolation (used by default)
            * ``cv2.INTER_CUBIC`` - a bicubic interpolation over 4x4 pixel neighborhood
            * ``cv2.INTER_LANCZOS4``
            * ``nearest``
            * ``linear``
            * ``cubic``
            * ``lanczos4``

        The first four are OpenCV constants, the other four are strings that
        are automatically replaced by the OpenCV constants.
        INTER_NEAREST (nearest neighbour interpolation) and INTER_NEAREST
        (linear interpolation) are the fastest.

            * If a single int, then that order will be used for all images.
            * If a string, then it must be one of: ``nearest``, ``linear``, ``cubic``,
              ``lanczos4``.
            * If an iterable of int/string, then for each image a random value
              will be sampled from that iterable (i.e. list of allowed order
              values).
            * If imgaug.ALL, then equivalant to list ``[cv2.INTER_NEAREST,
              cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]``.
            * If StochasticParameter, then that parameter is queried per image
              to sample the order value to use.

    cval : number or tuple of number or list of number or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant value used to fill up pixels in the result image that
        didn't exist in the input image (e.g. when translating to the left,
        some new pixels are created at the right). Such a fill-up with a
        constant value only happens, when `mode` is "constant".
        The expected value range is ``[0, 255]``. It may be a float value.

            * If this is a single int or float, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple ``(a, b)``, then a random value from the range ``a <= x <= b``
              is picked per image.
            * If a list, then a random value will eb sampled from that list
              per image.
            * If imgaug.ALL, a value from the discrete range ``[0 .. 255]`` will be
              sampled per image.
            * If a StochasticParameter, a new value will be sampled from the
              parameter per image.

    mode : int or str or list of str or list of int or imgaug.ALL or imgaug.parameters.StochasticParameter,
           optional
        Parameter that defines the handling of newly created pixels.
        Same meaning as in OpenCV's border mode. Let ``abcdefgh`` be an image's
        content and ``|`` be an image boundary, then:

            * ``cv2.BORDER_REPLICATE``: ``aaaaaa|abcdefgh|hhhhhhh``
            * ``cv2.BORDER_REFLECT``: ``fedcba|abcdefgh|hgfedcb``
            * ``cv2.BORDER_REFLECT_101``: ``gfedcb|abcdefgh|gfedcba``
            * ``cv2.BORDER_WRAP``: ``cdefgh|abcdefgh|abcdefg``
            * ``cv2.BORDER_CONSTANT``: ``iiiiii|abcdefgh|iiiiiii``, where ``i`` is the defined cval.
            * ``replicate``: Same as ``cv2.BORDER_REPLICATE``.
            * ``reflect``: Same as ``cv2.BORDER_REFLECT``.
            * ``reflect_101``: Same as ``cv2.BORDER_REFLECT_101``.
            * ``wrap``: Same as ``cv2.BORDER_WRAP``.
            * ``constant``: Same as ``cv2.BORDER_CONSTANT``.

        The datatype of the parameter may be:

            * If a single int, then it must be one of ``cv2.BORDER_*``.
            * If a single string, then it must be one of: ``replicate``,
              ``reflect``, ``reflect_101``, ``wrap``, ``constant``.
            * If a list of ints/strings, then per image a random mode will be
              picked from that list.
            * If imgaug.ALL, then a random mode from all possible modes will be
              picked.
            * If StochasticParameter, then the mode will be sampled from that
              parameter per image, i.e. it must return only the above mentioned
              strings.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.AffineCv2(scale=2.0)

    zooms all images by a factor of 2.

    >>> aug = iaa.AffineCv2(translate_px=16)

    translates all images on the x- and y-axis by 16 pixels (to the
    right/top), fills up any new pixels with zero (black values).

    >>> aug = iaa.AffineCv2(translate_percent=0.1)

    translates all images on the x- and y-axis by 10 percent of their
    width/height (to the right/top), fills up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(rotate=35)

    rotates all images by 35 degrees, fills up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(shear=15)

    rotates all images by 15 degrees, fills up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(translate_px=(-16, 16))

    translates all images on the x- and y-axis by a random value
    between -16 and 16 pixels (to the right/top) (same for both axis, i.e.
    sampled once per image), fills up any new pixels with zero (black values).

    >>> aug = iaa.AffineCv2(translate_px={"x": (-16, 16), "y": (-4, 4)})

    translates all images on the x-axis by a random value
    between -16 and 16 pixels (to the right) and on the y-axis by a
    random value between -4 and 4 pixels to the top. Even if both ranges
    were the same, both axis could use different samples.
    Fills up any new pixels with zero (black values).

    >>> aug = iaa.AffineCv2(scale=2.0, order=[0, 1])

    same as previously, but uses (randomly) either nearest neighbour
    interpolation or linear interpolation.

    >>> aug = iaa.AffineCv2(translate_px=16, cval=(0, 255))

    same as previously, but fills up any new pixels with a random
    brightness (same for the whole image).

    >>> aug = iaa.AffineCv2(translate_px=16, mode=["constant", "replicate"])

    same as previously, but fills up the new pixels in only 50 percent
    of all images with black values. In the other 50 percent of all cases,
    the value of the closest edge is used.

    """

    def __init__(self, scale=1.0, translate_percent=None, translate_px=None,
                 rotate=0.0, shear=0.0, order=cv2.INTER_LINEAR, cval=0, mode=cv2.BORDER_CONSTANT,
                 name=None, deterministic=False, random_state=None):
        super(AffineCv2, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        available_orders = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        available_orders_str = ["nearest", "linear", "cubic", "lanczos4"]

        if order == ia.ALL:
            self.order = iap.Choice(available_orders)
        elif ia.is_single_integer(order):
            ia.do_assert(order in available_orders,
                         "Expected order's integer value to be in %s, got %d." % (str(available_orders), order))
            self.order = iap.Deterministic(order)
        elif ia.is_string(order):
            ia.do_assert(order in available_orders_str,
                         "Expected order to be in %s, got %s." % (str(available_orders_str), order))
            self.order = iap.Deterministic(order)
        elif isinstance(order, list):
            ia.do_assert(all([ia.is_single_integer(val) or ia.is_string(val) for val in order]),
                         "Expected order list to only contain integers/strings, got types %s." % (
                             str([type(val) for val in order]),))
            ia.do_assert(all([val in available_orders + available_orders_str for val in order]),
                         "Expected all order values to be in %s, got %s." % (
                             available_orders + available_orders_str, str(order),))
            self.order = iap.Choice(order)
        elif isinstance(order, iap.StochasticParameter):
            self.order = order
        else:
            raise Exception("Expected order to be imgaug.ALL, int, string, a list of int/string or "
                            + "StochasticParameter, got %s." % (type(order),))

        if cval == ia.ALL:
            self.cval = iap.DiscreteUniform(0, 255)
        else:
            self.cval = iap.handle_discrete_param(cval, "cval", value_range=(0, 255), tuple_to_uniform=True,
                                                  list_to_choice=True, allow_floats=True)

        available_modes = [cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101, cv2.BORDER_WRAP,
                           cv2.BORDER_CONSTANT]
        available_modes_str = ["replicate", "reflect", "reflect_101", "wrap", "constant"]
        if mode == ia.ALL:
            self.mode = iap.Choice(available_modes)
        elif ia.is_single_integer(mode):
            ia.do_assert(mode in available_modes,
                         "Expected mode to be in %s, got %d." % (str(available_modes), mode))
            self.mode = iap.Deterministic(mode)
        elif ia.is_string(mode):
            ia.do_assert(mode in available_modes_str,
                         "Expected mode to be in %s, got %s." % (str(available_modes_str), mode))
            self.mode = iap.Deterministic(mode)
        elif isinstance(mode, list):
            ia.do_assert(all([ia.is_single_integer(val) or ia.is_string(val) for val in mode]),
                         "Expected mode list to only contain integers/strings, got types %s." % (
                             str([type(val) for val in mode]),))
            ia.do_assert(all([val in available_modes + available_modes_str for val in mode]),
                         "Expected all mode values to be in %s, got %s." % (
                             str(available_modes + available_modes_str), str(mode)))
            self.mode = iap.Choice(mode)
        elif isinstance(mode, iap.StochasticParameter):
            self.mode = mode
        else:
            raise Exception("Expected mode to be imgaug.ALL, an int, a string, a list of int/strings or "
                            + "StochasticParameter, got %s." % (type(mode),))

        # scale
        if isinstance(scale, dict):
            ia.do_assert("x" in scale or "y" in scale)
            x = scale.get("x", 1.0)
            y = scale.get("y", 1.0)
            self.scale = (
                iap.handle_continuous_param(x, "scale['x']", value_range=(0+1e-4, None), tuple_to_uniform=True,
                                            list_to_choice=True),
                iap.handle_continuous_param(y, "scale['y']", value_range=(0+1e-4, None), tuple_to_uniform=True,
                                            list_to_choice=True)
            )
        else:
            self.scale = iap.handle_continuous_param(scale, "scale", value_range=(0+1e-4, None), tuple_to_uniform=True,
                                                     list_to_choice=True)

        # translate
        if translate_percent is None and translate_px is None:
            translate_px = 0

        ia.do_assert(translate_percent is None or translate_px is None)

        if translate_percent is not None:
            # translate by percent
            if isinstance(translate_percent, dict):
                ia.do_assert("x" in translate_percent or "y" in translate_percent)
                x = translate_percent.get("x", 0)
                y = translate_percent.get("y", 0)
                self.translate = (
                    iap.handle_continuous_param(x, "translate_percent['x']", value_range=None, tuple_to_uniform=True,
                                                list_to_choice=True),
                    iap.handle_continuous_param(y, "translate_percent['y']", value_range=None, tuple_to_uniform=True,
                                                list_to_choice=True)
                )
            else:
                self.translate = iap.handle_continuous_param(translate_percent, "translate_percent", value_range=None,
                                                             tuple_to_uniform=True, list_to_choice=True)
        else:
            # translate by pixels
            if isinstance(translate_px, dict):
                ia.do_assert("x" in translate_px or "y" in translate_px)
                x = translate_px.get("x", 0)
                y = translate_px.get("y", 0)
                self.translate = (
                    iap.handle_discrete_param(x, "translate_px['x']", value_range=None, tuple_to_uniform=True,
                                              list_to_choice=True, allow_floats=False),
                    iap.handle_discrete_param(y, "translate_px['y']", value_range=None, tuple_to_uniform=True,
                                              list_to_choice=True, allow_floats=False)
                )
            else:
                self.translate = iap.handle_discrete_param(translate_px, "translate_px", value_range=None,
                                                           tuple_to_uniform=True, list_to_choice=True,
                                                           allow_floats=False)

        self.rotate = iap.handle_continuous_param(rotate, "rotate", value_range=None, tuple_to_uniform=True,
                                                  list_to_choice=True)
        self.shear = iap.handle_continuous_param(shear, "shear", value_range=None, tuple_to_uniform=True,
                                                 list_to_choice=True)

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, \
            mode_samples, order_samples = self._draw_samples(nb_images, random_state)
        result = self._augment_images_by_samples(images, scale_samples, translate_samples,  rotate_samples,
                                                 shear_samples, cval_samples, mode_samples, order_samples)
        return result

    @classmethod
    def _augment_images_by_samples(cls, images, scale_samples, translate_samples, rotate_samples, shear_samples,
                                   cval_samples, mode_samples, order_samples):
        # TODO change these to class attributes
        order_str_to_int = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4
        }
        mode_str_to_int = {
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "wrap": cv2.BORDER_WRAP,
            "constant": cv2.BORDER_CONSTANT
        }

        nb_images = len(images)
        result = images
        for i in sm.xrange(nb_images):
            height, width = images[i].shape[0], images[i].shape[1]
            shift_x = width / 2.0 - 0.5
            shift_y = height / 2.0 - 0.5
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x, translate_y = translate_samples[0][i], translate_samples[1][i]
            if ia.is_single_float(translate_y):
                translate_y_px = int(np.round(translate_y * images[i].shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(np.round(translate_x * images[i].shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            cval = cval_samples[i]
            mode = mode_samples[i]
            order = order_samples[i]

            mode = mode if ia.is_single_integer(mode) else mode_str_to_int[mode]
            order = order if ia.is_single_integer(order) else order_str_to_int[order]

            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 \
                    or shear != 0:
                matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x_px, translate_y_px),
                    rotation=math.radians(rotate),
                    shear=math.radians(shear)
                )
                matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
                matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center)

                image_warped = cv2.warpAffine(
                    images[i],
                    matrix.params[:2],
                    dsize=(width, height),
                    flags=order,
                    borderMode=mode,
                    borderValue=tuple([int(v) for v in cval])
                )

                # cv2 warp drops last axis if shape is (H, W, 1)
                if image_warped.ndim == 2:
                    image_warped = image_warped[..., np.newaxis]

                # warp changes uint8 to float64, making this necessary
                result[i] = image_warped
            else:
                result[i] = images[i]

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        nb_images = len(heatmaps)
        scale_samples, translate_samples, rotate_samples, shear_samples, \
            cval_samples, mode_samples, order_samples = self._draw_samples(nb_images, random_state)
        cval_samples = np.zeros((cval_samples.shape[0], 1), dtype=np.float32)
        mode_samples = ["constant"] * len(mode_samples)
        arrs = [heatmap_i.arr_0to1 for heatmap_i in heatmaps]
        arrs_aug = self._augment_images_by_samples(arrs, scale_samples, translate_samples, rotate_samples,
                                                   shear_samples, cval_samples, mode_samples, order_samples)
        for heatmap_i, arr_aug in zip(heatmaps, arrs_aug):
            heatmap_i.arr_0to1 = arr_aug
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        scale_samples, translate_samples, rotate_samples, shear_samples, \
            _cval_samples, _mode_samples, _order_samples = self._draw_samples(nb_images, random_state)

        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if not keypoints_on_image.keypoints:
                result.append(keypoints_on_image)
                continue
            height, width = keypoints_on_image.height, keypoints_on_image.width
            shift_x = width / 2.0 - 0.5
            shift_y = height / 2.0 - 0.5
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x, translate_y = translate_samples[0][i], translate_samples[1][i]
            if ia.is_single_float(translate_y):
                translate_y_px = int(np.round(translate_y * keypoints_on_image.shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(np.round(translate_x * keypoints_on_image.shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 \
                    or shear != 0:
                matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x_px, translate_y_px),
                    rotation=math.radians(rotate),
                    shear=math.radians(shear)
                )
                matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
                matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center)

                coords = keypoints_on_image.get_coords_array()
                coords_aug = tf.matrix_transform(coords, matrix.params)
                result.append(ia.KeypointsOnImage.from_coords_array(coords_aug, shape=keypoints_on_image.shape))
            else:
                result.append(keypoints_on_image)
        return result

    def get_parameters(self):
        return [self.scale, self.translate, self.rotate, self.shear, self.order, self.cval, self.mode]

    def _draw_samples(self, nb_samples, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]

        if isinstance(self.scale, tuple):
            scale_samples = (
                self.scale[0].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 10)),
                self.scale[1].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 20)),
            )
        else:
            scale_samples = self.scale.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 30))
            scale_samples = (scale_samples, scale_samples)

        if isinstance(self.translate, tuple):
            translate_samples = (
                self.translate[0].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 40)),
                self.translate[1].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 50)),
            )
        else:
            translate_samples = self.translate.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 60))
            translate_samples = (translate_samples, translate_samples)

        ia.do_assert(translate_samples[0].dtype in [np.int32, np.int64, np.float32, np.float64])
        ia.do_assert(translate_samples[1].dtype in [np.int32, np.int64, np.float32, np.float64])

        rotate_samples = self.rotate.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 70))
        shear_samples = self.shear.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 80))

        cval_samples = self.cval.draw_samples((nb_samples, 3), random_state=ia.new_random_state(seed + 90))
        mode_samples = self.mode.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 100))
        order_samples = self.order.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 110))

        return scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, mode_samples, \
            order_samples


class PiecewiseAffine(meta.Augmenter):
    """
    Augmenter that places a regular grid of points on an image and randomly
    moves the neighbourhood of these point around via affine transformations.
    This leads to local distortions.

    This is mostly a wrapper around scikit-image's PiecewiseAffine.
    See also the Affine augmenter for a similar technique.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested (1)
        * ``uint32``: yes; tested (1) (2)
        * ``uint64``: no (3)
        * ``int8``: yes; tested (1)
        * ``int16``: yes; tested (1)
        * ``int32``: yes; tested (1) (2)
        * ``int64``: no (3)
        * ``float16``: yes; tested (1)
        * ``float32``: yes; tested (1)
        * ``float64``: yes; tested (1)
        * ``float128``: no (3)
        * ``bool``: yes; tested (1) (4)

        - (1) Only tested with `order` set to 0.
        - (2) scikit-image converts internally to ``float64``, which might introduce inaccuracies.
              Tests showed that these inaccuracies seemed to not be an issue.
        - (3) results too inaccurate
        - (4) mapped internally to ``float64``

    Parameters
    ----------
    scale : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        Each point on the regular grid is moved around via a normal
        distribution. This scale factor is equivalent to the normal
        distribution's sigma. Note that the jitter (how far each point is
        moved in which direction) is multiplied by the height/width of the
        image if ``absolute_scale=False`` (default), so this scale can be
        the same for different sized images.
        Recommended values are in the range 0.01 to 0.05 (weak to strong
        augmentations).

            * If a single float, then that value will always be used as the
              scale.
            * If a tuple ``(a, b)`` of floats, then a random value will be picked
              from the interval ``(a, b)`` (per image).
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then that parameter will be queried to
              draw one value per image.

    nb_rows : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        Number of rows of points that the regular grid should have.
        Must be at least 2. For large images, you might want to pick a
        higher value than 4. You might have to then adjust scale to lower
        values.

            * If a single int, then that value will always be used as the
              number of rows.
            * If a tuple ``(a, b)``, then a value from the discrete interval ``[a..b]``
              will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a StochasticParameter, then that parameter will be queried to
              draw one value per image.

    nb_cols : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        Number of columns. See `nb_rows`.

    order : int or list of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.geometric.Affine.__init__`.

    cval : int or float or tuple of float or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.geometric.Affine.__init__`.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.geometric.Affine.__init__`.

    absolute_scale : bool, optional
        Take `scale` as an absolute value rather than a relative value.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))

    Puts a grid of points on each image and then randomly moves each point
    around by 1 to 5 percent (with respect to the image height/width). Pixels
    between these points will be moved accordingly.

    >>> aug = iaa.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=8, nb_cols=8)

    Same as the previous example, but uses a denser grid of ``8x8`` points (default
    is ``4x4``). This can be useful for large images.

    """

    def __init__(self, scale=0, nb_rows=4, nb_cols=4, order=1, cval=0, mode="constant", absolute_scale=False,
                 name=None, deterministic=False, random_state=None):
        super(PiecewiseAffine, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.scale = iap.handle_continuous_param(scale, "scale", value_range=(0, None), tuple_to_uniform=True,
                                                 list_to_choice=True)
        self.jitter = iap.Normal(loc=0, scale=self.scale)
        self.nb_rows = iap.handle_discrete_param(nb_rows, "nb_rows", value_range=(2, None), tuple_to_uniform=True,
                                                 list_to_choice=True, allow_floats=False)
        self.nb_cols = iap.handle_discrete_param(nb_cols, "nb_cols", value_range=(2, None), tuple_to_uniform=True,
                                                 list_to_choice=True, allow_floats=False)

        # --------------
        # order, mode, cval
        # TODO these are the same as in class Affine, make DRY
        # --------------

        # Peformance:
        #  1.0x order 0
        #  1.5x order 1
        #  3.0x order 3
        # 30.0x order 4
        # 60.0x order 5
        # measurement based on 256x256x3 batches, difference is smaller
        # on smaller images (seems to grow more like exponentially with image
        # size)
        if order == ia.ALL:
            # dont use order=2 (bi-quadratic) because that is apparently currently not recommended (and throws
            # a warning)
            self.order = iap.Choice([0, 1, 3, 4, 5])
        elif ia.is_single_integer(order):
            ia.do_assert(0 <= order <= 5,
                         "Expected order's integer value to be in range 0 <= x <= 5, got %d." % (order,))
            self.order = iap.Deterministic(order)
        elif isinstance(order, list):
            ia.do_assert(all([ia.is_single_integer(val) for val in order]),
                         "Expected order list to only contain integers, got types %s." % (
                             str([type(val) for val in order]),))
            ia.do_assert(all([0 <= val <= 5 for val in order]),
                         "Expected all of order's integer values to be in range 0 <= x <= 5, got %s." % (str(order),))
            self.order = iap.Choice(order)
        elif isinstance(order, iap.StochasticParameter):
            self.order = order
        else:
            raise Exception("Expected order to be imgaug.ALL, int or StochasticParameter, got %s." % (type(order),))

        if cval == ia.ALL:
            # TODO change this so that it is dynamically created per image (or once per dtype)
            self.cval = iap.Uniform(0, 255)
        else:
            self.cval = iap.handle_continuous_param(cval, "cval", value_range=None, tuple_to_uniform=True,
                                                    list_to_choice=True)

        # constant, edge, symmetric, reflect, wrap
        if mode == ia.ALL:
            self.mode = iap.Choice(["constant", "edge", "symmetric", "reflect", "wrap"])
        elif ia.is_string(mode):
            self.mode = iap.Deterministic(mode)
        elif isinstance(mode, list):
            ia.do_assert(all([ia.is_string(val) for val in mode]))
            self.mode = iap.Choice(mode)
        elif isinstance(mode, iap.StochasticParameter):
            self.mode = mode
        else:
            raise Exception("Expected mode to be imgaug.ALL, a string, a list of strings or StochasticParameter, "
                            + "got %s." % (type(mode),))

        self.absolute_scale = absolute_scale

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool", "uint8", "uint16", "uint32", "int8", "int16", "int32",
                                  "float16", "float32", "float64"],
                         disallowed=["uint64", "uint128", "uint256", "int64", "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=self)

        result = images
        nb_images = len(images)

        rss = ia.derive_random_states(random_state, nb_images+5)

        # make sure to sample "order" here at the 3rd position to match the sampling steps
        # in _augment_heatmaps()
        nb_rows_samples = self.nb_rows.draw_samples((nb_images,), random_state=rss[-5])
        nb_cols_samples = self.nb_cols.draw_samples((nb_images,), random_state=rss[-4])
        order_samples = self.order.draw_samples((nb_images,), random_state=rss[-3])
        cval_samples = self.cval.draw_samples((nb_images,), random_state=rss[-2])
        mode_samples = self.mode.draw_samples((nb_images,), random_state=rss[-1])

        for i, image in enumerate(images):
            rs_image = rss[i]
            h, w = image.shape[0:2]

            transformer = self._get_transformer(h, w, nb_rows_samples[i], nb_cols_samples[i], rs_image)

            if transformer is not None:
                input_dtype = image.dtype
                if image.dtype == np.bool_:
                    image = image.astype(np.float64)

                min_value, _center_value, max_value = iadt.get_value_range_of_dtype(image.dtype)
                cval = cval_samples[i]
                cval = max(min(cval, max_value), min_value)

                image_warped = tf.warp(
                    image,
                    transformer,
                    order=order_samples[i],
                    mode=mode_samples[i],
                    cval=cval,
                    preserve_range=True,
                    output_shape=images[i].shape
                )

                if input_dtype == np.bool_:
                    image_warped = image_warped > 0.5
                else:
                    # warp seems to change everything to float64, including uint8,
                    # making this necessary
                    image_warped = iadt.restore_dtypes_(image_warped, input_dtype)

                result[i] = image_warped

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        result = heatmaps
        nb_images = len(heatmaps)

        rss = ia.derive_random_states(random_state, nb_images+3)

        nb_rows_samples = self.nb_rows.draw_samples((nb_images,), random_state=rss[-3])
        nb_cols_samples = self.nb_cols.draw_samples((nb_images,), random_state=rss[-2])
        order_samples = self.order.draw_samples((nb_images,), random_state=rss[-1])

        for i in sm.xrange(nb_images):
            heatmaps_i = heatmaps[i]
            arr_0to1 = heatmaps_i.arr_0to1

            rs_image = rss[i]
            h, w = arr_0to1.shape[0:2]
            transformer = self._get_transformer(h, w, nb_rows_samples[i], nb_cols_samples[i], rs_image)

            if transformer is not None:
                arr_0to1_warped = tf.warp(
                    arr_0to1,
                    transformer,
                    order=order_samples[i],
                    mode="constant",
                    cval=0,
                    preserve_range=True,
                    output_shape=arr_0to1.shape
                )

                # skimage converts to float64
                arr_0to1_warped = arr_0to1_warped.astype(np.float32)

                heatmaps_i.arr_0to1 = arr_0to1_warped

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)

        rss = ia.derive_random_states(random_state, nb_images+2)

        nb_rows_samples = self.nb_rows.draw_samples((nb_images,), random_state=rss[-2])
        nb_cols_samples = self.nb_cols.draw_samples((nb_images,), random_state=rss[-1])

        for i in sm.xrange(nb_images):
            if not keypoints_on_images[i].keypoints:
                result.append(keypoints_on_images[i])
                continue
            rs_image = rss[i]
            kpsoi = keypoints_on_images[i]
            h, w = kpsoi.shape[0:2]
            transformer = self._get_transformer(h, w, nb_rows_samples[i], nb_cols_samples[i], rs_image)

            if transformer is None or len(kpsoi.keypoints) == 0:
                result.append(kpsoi)
            else:
                # Augmentation routine that only modifies keypoint coordinates
                # This is efficient (coordinates of all other locations in the
                # image are ignored). The code below should usually work, but
                # for some reason augmented coordinates are often wildly off
                # for large scale parameters (lots of jitter/distortion).
                # The reason for that is unknown.
                """
                coords = keypoints_on_images[i].get_coords_array()
                coords_aug = transformer.inverse(coords)
                result.append(
                    ia.KeypointsOnImage.from_coords_array(
                        coords_aug,
                        shape=keypoints_on_images[i].shape
                    )
                )
                """

                # Image based augmentation routine. Draws the keypoints on the image plane using distance maps (more
                # accurate than just marking the points),  then augments these images, then searches for the new
                # (visual) location of the keypoints.
                # Much slower than directly augmenting the coordinates, but here the only method that reliably works.
                dist_maps = kpsoi.to_distance_maps(inverted=True)
                dist_maps_warped = tf.warp(
                    dist_maps,
                    transformer,
                    order=1,
                    preserve_range=True,
                    output_shape=(kpsoi.shape[0], kpsoi.shape[1], len(kpsoi.keypoints))
                )

                kps_aug = ia.KeypointsOnImage.from_distance_maps(
                    dist_maps_warped,
                    inverted=True,
                    threshold=0.01,
                    if_not_found_coords={"x": -1, "y": -1},
                    nb_channels=None if len(kpsoi.shape) < 3 else kpsoi.shape[2]
                )

                # TODO is this still necessary after nb_channels was added to from_keypoint_image() ?
                if len(kpsoi.shape) > 2:
                    kps_aug.shape = (
                        kps_aug.shape[0],
                        kps_aug.shape[1],
                        kpsoi.shape[2]
                    )

                # Keypoints that were outside of the image plane before the
                # augmentation will be replaced with (-1, -1) by default (as
                # they can't be drawn on the keypoint images). They are now
                # replaced by their old coordinates values.
                ooi = [not 0 <= kp.x < w or not 0 <= kp.y < h for kp in kpsoi.keypoints]
                for kp_idx in sm.xrange(len(kps_aug.keypoints)):
                    if ooi[kp_idx]:
                        kp_unaug = kpsoi.keypoints[kp_idx]
                        kps_aug.keypoints[kp_idx] = kp_unaug

                result.append(kps_aug)

        return result

    def _get_transformer(self, h, w, nb_rows, nb_cols, random_state):
        # get coords on y and x axis of points to move around
        # these coordinates are supposed to be at the centers of each cell
        # (otherwise the first coordinate would be at (0, 0) and could hardly
        # be moved around before leaving the image),
        # so we use here (half cell height/width to H/W minus half height/width)
        # instead of (0, H/W)

        nb_rows = max(nb_rows, 2)
        nb_cols = max(nb_cols, 2)

        y = np.linspace(0, h, nb_rows)
        x = np.linspace(0, w, nb_cols)

        xx_src, yy_src = np.meshgrid(x, y)  # (H, W) and (H, W) for H=rows, W=cols
        points_src = np.dstack([yy_src.flat, xx_src.flat])[0]  # (1, HW, 2) => (HW, 2) for H=rows, W=cols

        jitter_img = self.jitter.draw_samples(points_src.shape, random_state=random_state)

        nb_nonzero = len(jitter_img.flatten().nonzero()[0])
        if nb_nonzero == 0:
            return None
        else:
            if not self.absolute_scale:
                jitter_img[:, 0] = jitter_img[:, 0] * h
                jitter_img[:, 1] = jitter_img[:, 1] * w
            points_dest = np.copy(points_src)
            points_dest[:, 0] = points_dest[:, 0] + jitter_img[:, 0]
            points_dest[:, 1] = points_dest[:, 1] + jitter_img[:, 1]

            # Restrict all destination points to be inside the image plane.
            # This is necessary, as otherwise keypoints could be augmented
            # outside of the image plane and these would be replaced by
            # (-1, -1), which would not conform with the behaviour of the
            # other augmenters.
            points_dest[:, 0] = np.clip(points_dest[:, 0], 0, h-1)
            points_dest[:, 1] = np.clip(points_dest[:, 1], 0, w-1)

            matrix = tf.PiecewiseAffineTransform()
            matrix.estimate(points_src[:, ::-1], points_dest[:, ::-1])
            return matrix

    def get_parameters(self):
        return [self.scale, self.nb_rows, self.nb_cols, self.order, self.cval, self.mode, self.absolute_scale]


class PerspectiveTransform(meta.Augmenter):
    """
    Augmenter that performs a random four point perspective transform.

    Each of the four points is placed on the image using a random distance from
    its respective corner. The distance is sampled from a normal distribution.
    As a result, most transformations don't change very much, while some
    "focus" on polygons far inside the image.

    The results of this augmenter have some similarity with Crop.

    Code partially from http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ .

    dtype support::

        if (keep_size=False)::

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: no (1)
            * ``uint64``: no (2)
            * ``int8``: yes; tested (3)
            * ``int16``: yes; tested
            * ``int32``: no (2)
            * ``int64``: no (2)
            * ``float16``: yes; tested (4)
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: no (1)
            * ``bool``: yes; tested (4)

            - (1) rejected by opencv
            - (2) leads to opencv error: cv2.error: ``OpenCV(3.4.4) (...)imgwarp.cpp:1805: error:
                  (-215:Assertion failed) ifunc != 0 in function 'remap'``.
            - (3) mapped internally to ``int16``.
            - (4) mapped intenally to ``float32``.

        if (keep_size=True)::

            minimum of (
                ``imgaug.augmenters.geometric.PerspectiveTransform(keep_size=False)``,
                :func:`imgaug.imgaug.imresize_many_images`
            )

    Parameters
    ----------
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the normal distributions. These are used to sample
        the random distances of the subimage's corners from the full image's
        corners. The sampled values reflect percentage values (with respect
        to image height/width). Recommended values are in the range 0.0 to 0.1.

            * If a single number, then that value will always be used as the
              scale.
            * If a tuple ``(a, b)`` of numbers, then a random value will be picked
              from the interval ``(a, b)`` (per image).
            * If a list of values, a random one of the values will be picked
              per image.
            * If a StochasticParameter, then that parameter will be queried to
              draw one value per image.

    keep_size : bool, optional
        Whether to resize image's back to their original size after applying
        the perspective transform. If set to False, the resulting images
        may end up having different shapes and will always be a list, never
        an array.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.PerspectiveTransform(scale=(0.01, 0.10))

    Applies perspective transformations using a random scale between 0.01 and
    0.1 per image, where the scale is roughly a measure of how far the
    perspective transform's corner points may be distanced from the original
    image's corner points.

    """

    def __init__(self, scale=0, keep_size=True, name=None, deterministic=False, random_state=None):
        super(PerspectiveTransform, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.scale = iap.handle_continuous_param(scale, "scale", value_range=(0, None), tuple_to_uniform=True,
                                                 list_to_choice=True)
        self.jitter = iap.Normal(loc=0, scale=self.scale)
        self.keep_size = keep_size

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool", "uint8", "uint16", "int8", "int16",
                                  "float16", "float32", "float64"],
                         disallowed=["uint32", "uint64", "uint128", "uint256", "int32", "int64", "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=self)

        result = images
        if not self.keep_size:
            result = list(result)

        matrices, max_heights, max_widths = self._create_matrices(
            [image.shape for image in images],
            random_state
        )

        for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
            image = images[i]

            # cv2.warpPerspective only supports <=4 channels
            nb_channels = image.shape[2]
            input_dtype = image.dtype
            if input_dtype in [np.int8]:
                image = image.astype(np.int16)
            elif input_dtype in [np.bool_, np.float16]:
                image = image.astype(np.float32)

            if nb_channels <= 4:
                warped = cv2.warpPerspective(image, M, (max_width, max_height))
                if warped.ndim == 2 and images[i].ndim == 3:
                    warped = np.expand_dims(warped, 2)
            else:
                # warp each channel on its own, re-add channel axis, then stack
                # the result from a list of [H, W, 1] to (H, W, C).
                warped = [cv2.warpPerspective(image[..., c], M, (max_width, max_height))
                          for c in sm.xrange(nb_channels)]
                warped = [warped_i[..., np.newaxis] for warped_i in warped]
                warped = np.dstack(warped)

            if self.keep_size:
                h, w = image.shape[0:2]
                warped = ia.imresize_single_image(warped, (h, w), interpolation="cubic")

            if input_dtype == np.bool_:
                warped = warped > 0.5
            elif warped.dtype != input_dtype:
                warped = iadt.restore_dtypes_(warped, input_dtype)

            result[i] = warped

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        result = heatmaps

        matrices, max_heights, max_widths = self._create_matrices(
            [heatmaps_i.arr_0to1.shape for heatmaps_i in heatmaps],
            ia.copy_random_state(random_state)
        )

        # estimate max_heights/max_widths for the underlying images
        # this is only necessary if keep_size is False as then the underlying image sizes
        # change and we need to update them here
        if self.keep_size:
            max_heights_imgs, max_widths_imgs = max_heights, max_widths
        else:
            _, max_heights_imgs, max_widths_imgs = self._create_matrices(
                [heatmaps_i.shape for heatmaps_i in heatmaps],
                ia.copy_random_state(random_state)
            )

        for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
            heatmaps_i = heatmaps[i]

            arr = heatmaps_i.arr_0to1

            nb_channels = arr.shape[2]

            warped = [cv2.warpPerspective(arr[..., c], M, (max_width, max_height)) for c in sm.xrange(nb_channels)]
            warped = [warped_i[..., np.newaxis] for warped_i in warped]
            warped = np.dstack(warped)

            heatmaps_i_aug = ia.HeatmapsOnImage.from_0to1(warped, shape=heatmaps_i.shape,
                                                          min_value=heatmaps_i.min_value,
                                                          max_value=heatmaps_i.max_value)

            if self.keep_size:
                h, w = arr.shape[0:2]
                heatmaps_i_aug = heatmaps_i_aug.resize((h, w))
            else:
                heatmaps_i_aug.shape = (max_heights_imgs[i], max_widths_imgs[i]) + heatmaps_i_aug.shape[2:]

            result[i] = heatmaps_i_aug

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = keypoints_on_images
        matrices, max_heights, max_widths = self._create_matrices(
            [kps.shape for kps in keypoints_on_images],
            random_state
        )

        for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
            keypoints_on_image = keypoints_on_images[i]
            if not keypoints_on_image.keypoints:
                continue
            kps_arr = keypoints_on_image.get_coords_array()

            warped = cv2.perspectiveTransform(np.array([kps_arr], dtype=np.float32), M)
            warped = warped[0]
            warped_kps = ia.KeypointsOnImage.from_coords_array(
                warped,
                shape=(max_height, max_width) + keypoints_on_image.shape[2:]
            )
            if self.keep_size:
                warped_kps = warped_kps.on(keypoints_on_image.shape)
            result[i] = warped_kps

        return result

    def _create_matrices(self, shapes, random_state):
        matrices = []
        max_heights = []
        max_widths = []
        nb_images = len(shapes)
        seeds = ia.copy_random_state(random_state).randint(0, 10**6, (nb_images,))

        for i in sm.xrange(nb_images):
            h, w = shapes[i][0:2]

            points = self.jitter.draw_samples((4, 2), random_state=ia.new_random_state(seeds[i]))
            points = np.mod(np.abs(points), 1)

            # top left
            points[0, 1] = 1.0 - points[0, 1]  # h = 1.0 - jitter

            # top right
            points[1, 0] = 1.0 - points[1, 0]  # w = 1.0 - jitter
            points[1, 1] = 1.0 - points[1, 1]  # h = 1.0 - jitter

            # bottom right
            points[2, 0] = 1.0 - points[2, 0]  # h = 1.0 - jitter

            # bottom left
            # nothing

            points[:, 0] = points[:, 0] * w
            points[:, 1] = points[:, 1] * h

            # obtain a consistent order of the points and unpack them
            # individually
            points = self._order_points(points)
            (tl, tr, br, bl) = points

            # compute the width of the new image, which will be the
            # maximum distance between bottom-right and bottom-left
            # x-coordiates or the top-right and top-left x-coordinates
            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(width_a), int(width_b))

            # compute the height of the new image, which will be the
            # maximum distance between the top-right and bottom-right
            # y-coordinates or the top-left and bottom-left y-coordinates
            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = max(int(height_a), int(height_b))

            # now that we have the dimensions of the new image, construct
            # the set of destination points to obtain a "birds eye view",
            # (i.e. top-down view) of the image, again specifying points
            # in the top-left, top-right, bottom-right, and bottom-left
            # order
            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype="float32")

            # compute the perspective transform matrix and then apply it
            m = cv2.getPerspectiveTransform(points, dst)
            matrices.append(m)
            max_heights.append(max_height)
            max_widths.append(max_width)

        return matrices, max_heights, max_widths

    @classmethod
    def _order_points(cls, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        pts_ordered = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        pts_ordered[0] = pts[np.argmin(s)]
        pts_ordered[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        pts_ordered[1] = pts[np.argmin(diff)]
        pts_ordered[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return pts_ordered

    def get_parameters(self):
        return [self.jitter, self.keep_size]


# code partially from https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
# TODO add independent sigmas for x/y
# TODO add independent alphas for x/y
# TODO add backend arg
class ElasticTransformation(meta.Augmenter):
    """
    Augmenter to transform images by moving pixels locally around using displacement fields.

    The augmenter has the parameters ``alpha`` and ``sigma``. ``alpha`` controls the strength of the
    displacement: higher values mean that pixels are moved further. ``sigma`` controls the
    smoothness of the displacement: higher values lead to smoother patterns -- as if the
    image was below water -- while low values will cause indivdual pixels to be moved very
    differently from their neighbours, leading to noisy and pixelated images.

    A relation of 10:1 seems to be good for ``alpha`` and ``sigma``, e.g. ``alpha=10`` and ``sigma=1`` or
    ``alpha=50``, ``sigma=5``. For ``128x128`` a setting of ``alpha=(0, 70.0)``, ``sigma=(4.0, 6.0)`` may be a
    good choice and will lead to a water-like effect.

    See ::

        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003

    for a detailed explanation.

    dtype support::

        * ``uint8``: yes; fully tested (1)
        * ``uint16``: yes; tested (1)
        * ``uint32``: yes; tested (2)
        * ``uint64``: limited; tested (3)
        * ``int8``: yes; tested (1) (4) (5)
        * ``int16``: yes; tested (4) (6)
        * ``int32``: yes; tested (4) (6)
        * ``int64``: limited; tested (3)
        * ``float16``: yes; tested (1)
        * ``float32``: yes; tested (1)
        * ``float64``: yes; tested (1)
        * ``float128``: no
        * ``bool``: yes; tested (1) (7)

        (1) Always handled by ``cv2``.
        (2) Always handled by ``scipy``.
        (3) Only supported for ``order != 0``. Will fail for ``order=0``.
        (4) Mapped internally to ``float64`` when ``order=1``.
        (5) Mapped internally to ``int16`` when ``order>=2``.
        (6) Handled by ``cv2`` when ``order=0`` or ``order=1``, otherwise by ``scipy``.
        (7) Mapped internally to ``float32``.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Strength of the distortion field. Higher values mean that pixels are moved further
        with respect to the distortion field's direction. Set this to around 10 times the
        value of `sigma` for visible effects.

            * If number, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value from range ``a <= x <= b`` will be
              sampled per image.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If StochasticParameter, then that parameter will be used to sample
              a value per image.

    sigma : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the gaussian kernel used to smooth the distortion
        fields. Higher values (for ``128x128`` images around 5.0) lead to more water-like effects,
        while lower values (for ``128x128`` images around 1.0 and lower) lead to more noisy, pixelated
        images. Set this to around 1/10th of `alpha` for visible effects.

            * If number, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value from range ``a <= x <= b`` will be
              sampled per image.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If StochasticParameter, then that parameter will be used to sample
              a value per image.

    order : int or list of int or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use. Same meaning as in
        :func:`scipy.ndimage.map_coordinates` and may take any integer value
        in the range 0 to 5, where orders close to 0 are faster.

            * If a single int, then that order will be used for all images.
            * If a tuple ``(a, b)``, then a random value from the range ``a <= x <= b``
              is picked per image.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If imgaug.ALL, then equivalant to list ``[0, 1, 2, 3, 4, 5]``.
            * If StochasticParameter, then that parameter is queried per image
              to sample the order value to use.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to "constant".
        For standard uint8 images (value range 0-255), this value may also
        come from the range 0-255. It may be a float value, even for
        integer image dtypes.

            * If this is a single int or float, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple ``(a, b)``, then a random value from the range ``a <= x <= b``
              is picked per image.
            * If a list, then a random value will be picked from that list per
              image.
            * If imgaug.ALL, a value from the discrete range ``[0 .. 255]`` will be
              sampled per image.
            * If a StochasticParameter, a new value will be sampled from the
              parameter per image.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Parameter that defines the handling of newly created pixels.
        May take the same values as in :func:`scipy.ndimage.map_coordinates`,
        i.e. ``constant``, ``nearest``, ``reflect`` or ``wrap``.
        The datatype of the parameter may
        be:

            * If a single string, then that mode will be used for all images.
            * If a list of strings, then per image a random mode will be picked
              from that list.
            * If imgaug.ALL, then a random mode from all possible modes will be
              picked.
            * If StochasticParameter, then the mode will be sampled from that
              parameter per image, i.e. it must return only the above mentioned
              strings.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.ElasticTransformation(alpha=50.0, sigma=5.0)

    apply elastic transformations with a strength/alpha of 50.0 and
    smoothness of 5.0 to all images.


    >>> aug = iaa.ElasticTransformation(alpha=(0.0, 70.0), sigma=5.0)

    apply elastic transformations with a strength/alpha that comes
    from the range ``0.0 <= x <= 70.0`` (randomly picked per image) and
    with a smoothness of 5.0.

    """

    NB_NEIGHBOURING_KEYPOINTS = 3
    NEIGHBOURING_KEYPOINTS_DISTANCE = 1.0
    KEYPOINT_AUG_ALPHA_THRESH = 0.05
    # even at high alphas we don't augment keypoints if the sigma is too low, because then
    # the pixel movements are mostly gaussian noise anyways
    KEYPOINT_AUG_SIGMA_THRESH = 1.0

    _MAPPING_MODE_SCIPY_CV2 = {
        "constant": cv2.BORDER_CONSTANT,
        "nearest": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "wrap": cv2.BORDER_WRAP
    }

    _MAPPING_ORDER_SCIPY_CV2 = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_CUBIC,
        3: cv2.INTER_CUBIC,
        4: cv2.INTER_CUBIC,
        5: cv2.INTER_CUBIC
    }

    def __init__(self, alpha=0, sigma=0, order=3, cval=0, mode="constant",
                 name=None, deterministic=False, random_state=None):
        super(ElasticTransformation, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.alpha = iap.handle_continuous_param(alpha, "alpha", value_range=(0, None), tuple_to_uniform=True,
                                                 list_to_choice=True)
        self.sigma = iap.handle_continuous_param(sigma, "sigma", value_range=(0, None), tuple_to_uniform=True,
                                                 list_to_choice=True)

        if order == ia.ALL:
            self.order = iap.Choice([0, 1, 2, 3, 4, 5])
        else:
            self.order = iap.handle_discrete_param(order, "order", value_range=(0, 5), tuple_to_uniform=True,
                                                   list_to_choice=True, allow_floats=False)

        if cval == ia.ALL:
            # TODO change this so that it is dynamically created per image (or once per dtype)
            self.cval = iap.DiscreteUniform(0, 255)
        else:
            self.cval = iap.handle_discrete_param(cval, "cval", value_range=None, tuple_to_uniform=True,
                                                  list_to_choice=True, allow_floats=True)

        if mode == ia.ALL:
            self.mode = iap.Choice(["constant", "nearest", "reflect", "wrap"])
        elif ia.is_string(mode):
            self.mode = iap.Deterministic(mode)
        elif ia.is_iterable(mode):
            ia.do_assert(all([ia.is_string(val) for val in mode]))
            self.mode = iap.Choice(mode)
        elif isinstance(mode, iap.StochasticParameter):
            self.mode = mode
        else:
            raise Exception("Expected mode to be imgaug.ALL, a string, a list of strings or StochasticParameter, "
                            + "got %s." % (type(mode),))

    def _draw_samples(self, nb_images, random_state):
        # seeds = ia.copy_random_state(random_state).randint(0, 10**6, (nb_images+1,))
        rss = ia.derive_random_states(random_state, nb_images+5)
        alphas = self.alpha.draw_samples((nb_images,), random_state=rss[-5])
        sigmas = self.sigma.draw_samples((nb_images,), random_state=rss[-4])
        orders = self.order.draw_samples((nb_images,), random_state=rss[-3])
        cvals = self.cval.draw_samples((nb_images,), random_state=rss[-2])
        modes = self.mode.draw_samples((nb_images,), random_state=rss[-1])
        return rss[0:-5], alphas, sigmas, orders, cvals, modes

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64",
                                  "float16", "float32", "float64"],
                         disallowed=["uint128", "uint256", "int128", "int256", "float96", "float128", "float256"],
                         augmenter=self)

        result = images
        nb_images = len(images)
        rss, alphas, sigmas, orders, cvals, modes = self._draw_samples(nb_images, random_state)
        assert len(images) == len(rss) == len(alphas) == len(sigmas) == len(orders) == len(cvals) == len(modes)

        for i, image in enumerate(images):
            image = images[i]
            min_value, _center_value, max_value = iadt.get_value_range_of_dtype(image.dtype)
            cval = cvals[i]
            cval = max(min(cval, max_value), min_value)

            input_dtype = image.dtype
            if image.dtype == np.dtype(np.float16):
                image = image.astype(np.float32)

            dx, dy = self.generate_shift_maps(
                image.shape[0:2],
                alpha=alphas[i],
                sigma=sigmas[i],
                random_state=rss[i]
            )
            image_aug = self.map_coordinates(
                image,
                dx,
                dy,
                order=orders[i],
                cval=cval,
                mode=modes[i]
            )

            if image.dtype != input_dtype:
                image_aug = iadt.restore_dtypes_(image_aug, input_dtype)
            result[i] = image_aug

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        nb_heatmaps = len(heatmaps)
        rss, alphas, sigmas, orders, _cvals, _modes = self._draw_samples(nb_heatmaps, random_state)
        for i in sm.xrange(nb_heatmaps):
            heatmaps_i = heatmaps[i]
            if heatmaps_i.arr_0to1.shape[0:2] == heatmaps_i.shape[0:2]:
                dx, dy = self.generate_shift_maps(
                    heatmaps_i.arr_0to1.shape[0:2],
                    alpha=alphas[i],
                    sigma=sigmas[i],
                    random_state=rss[i]
                )

                arr_0to1_warped = self.map_coordinates(
                    heatmaps_i.arr_0to1,
                    dx,
                    dy,
                    order=orders[i],
                    cval=0,
                    mode="constant"
                )

                # interpolation in map_coordinates() can cause some values to be below/above 1.0,
                # so we clip here
                arr_0to1_warped = np.clip(arr_0to1_warped, 0.0, 1.0, out=arr_0to1_warped)

                heatmaps_i.arr_0to1 = arr_0to1_warped
            else:
                # Heatmaps do not have the same size as augmented images.
                # This may result in indices of moved pixels being different.
                # To prevent this, we use the same image size as for the base images, but that
                # requires resizing the heatmaps temporarily to the image sizes.
                height_orig, width_orig = heatmaps_i.arr_0to1.shape[0:2]
                heatmaps_i = heatmaps_i.resize(heatmaps_i.shape[0:2])
                arr_0to1 = heatmaps_i.arr_0to1
                dx, dy = self.generate_shift_maps(
                    arr_0to1.shape[0:2],
                    alpha=alphas[i],
                    sigma=sigmas[i],
                    random_state=rss[i]
                )

                # TODO will it produce similar results to first downscale the shift maps and then remap?
                #      That would make the remap step take less operations and would also mean that the heatmaps
                #      wouldnt have to be scaled up anymore. It would also simplify the code as this branch could
                #      be merged with the one above.
                arr_0to1_warped = self.map_coordinates(
                    arr_0to1,
                    dx,
                    dy,
                    order=orders[i],
                    cval=0,
                    mode="constant"
                )

                # interpolation in map_coordinates() can cause some values to be below/above 1.0,
                # so we clip here
                arr_0to1_warped = np.clip(arr_0to1_warped, 0.0, 1.0, out=arr_0to1_warped)

                heatmaps_i_warped = ia.HeatmapsOnImage.from_0to1(arr_0to1_warped, shape=heatmaps_i.shape,
                                                                 min_value=heatmaps_i.min_value,
                                                                 max_value=heatmaps_i.max_value)
                heatmaps_i_warped = heatmaps_i_warped.resize((height_orig, width_orig))
                heatmaps[i] = heatmaps_i_warped

        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = keypoints_on_images
        nb_images = len(keypoints_on_images)
        rss, alphas, sigmas, _orders, _cvals, _modes = self._draw_samples(nb_images, random_state)
        for i in sm.xrange(nb_images):
            kpsoi = keypoints_on_images[i]
            if not kpsoi.keypoints:
                continue
            h, w = kpsoi.shape[0:2]
            dx, dy = self.generate_shift_maps(
                kpsoi.shape[0:2],
                alpha=alphas[i],
                sigma=sigmas[i],
                random_state=rss[i]
            )

            kps_aug = []
            for kp in kpsoi.keypoints:
                # dont augment keypoints if alpha/sigma are too low or if the keypoint is outside
                # of the image plane
                params_above_thresh = (alphas[i] > self.KEYPOINT_AUG_ALPHA_THRESH
                                       and sigmas[i] > self.KEYPOINT_AUG_SIGMA_THRESH)
                within_image_plane = (0 <= kp.x < w and 0 <= kp.y < h)
                if not params_above_thresh or not within_image_plane:
                    kps_aug.append(kp)
                else:
                    kp_neighborhood = kp.generate_similar_points_manhattan(
                        self.NB_NEIGHBOURING_KEYPOINTS,
                        self.NEIGHBOURING_KEYPOINTS_DISTANCE,
                        return_array=True
                    )

                    # We can clip here, because we made sure above that the keypoint is inside the
                    # image plane. Keypoints at the bottom row or right columns might be rounded
                    # outside the image plane, which we prevent here.
                    # We reduce neighbours to only those within the image plane as only for such
                    # points we know where to move them.
                    xx = np.round(kp_neighborhood[:, 0]).astype(np.int32)
                    yy = np.round(kp_neighborhood[:, 1]).astype(np.int32)
                    inside_image_mask = np.logical_and(
                        np.logical_and(0 <= xx, xx < w),
                        np.logical_and(0 <= yy, yy < h)
                    )
                    xx = xx[inside_image_mask]
                    yy = yy[inside_image_mask]

                    xxyy = np.concatenate([xx[:, np.newaxis], yy[:, np.newaxis]], axis=1)

                    xxyy_aug = np.copy(xxyy).astype(np.float32)
                    xxyy_aug[:, 0] += dx[yy, xx]
                    xxyy_aug[:, 1] += dy[yy, xx]

                    med = ia.compute_geometric_median(xxyy_aug)
                    # med = np.average(xxyy_aug, 0)  # uncomment to use average instead of median
                    kps_aug.append(ia.Keypoint(x=med[0], y=med[1]))

            result[i] = ia.KeypointsOnImage(kps_aug, shape=kpsoi.shape)

        return result

    def get_parameters(self):
        return [self.alpha, self.sigma, self.order, self.cval, self.mode]

    @classmethod
    def generate_shift_maps(cls, shape, alpha, sigma, random_state):
        ia.do_assert(len(shape) == 2)

        # kernel size used for cv2 based blurring, should be copied from gaussian_blur_()
        # TODO make dry
        if sigma < 3.0:
            ksize = 3.3 * sigma  # 99% of weight
        elif sigma < 5.0:
            ksize = 2.9 * sigma  # 97% of weight
        else:
            ksize = 2.6 * sigma  # 95% of weight

        ksize = int(max(ksize, 5))
        ksize = ksize + 1 if ksize % 2 == 0 else ksize

        padding = ksize
        h, w = shape[0:2]
        h_pad = h + 2*padding
        w_pad = w + 2*padding

        # The step of random number generation could be batched, so that random numbers are sampled once for the whole
        # batch. Would get rid of creating many random_states.
        dxdy_unsmoothed = (random_state.rand(2 * h_pad, w_pad) * 2 - 1).astype(np.float32)

        dx_unsmoothed = dxdy_unsmoothed[0:h_pad, :]
        dy_unsmoothed = dxdy_unsmoothed[h_pad:, :]

        # TODO could this also work with an average blur? would probably be faster
        dx = blur_lib.blur_gaussian_(dx_unsmoothed, sigma) * alpha
        dy = blur_lib.blur_gaussian_(dy_unsmoothed, sigma) * alpha

        if padding > 0:
            dx = dx[padding:-padding, padding:-padding]
            dy = dy[padding:-padding, padding:-padding]

        return dx, dy

    @classmethod
    def map_coordinates(cls, image, dx, dy, order=1, cval=0, mode="constant"):
        """

        backend="scipy"

            order=0
                * uint8: yes
                * uint16: yes
                * uint32: yes
                * uint64: no (produces array filled with only 0)
                * int8: yes
                * int16: yes
                * int32: yes
                * int64: no (produces array filled with <min_value> when testing with <max_value>)
                * float16: yes
                * float32: yes
                * float64: yes
                * float128: no (causes: 'data type no supported')
                * bool: yes

            order=1 to 5
                * uint*, int*: yes (rather loose test, to avoid having to re-compute the interpolation)
                * float16 - float64: yes (rather loose test, to avoid having to re-compute the interpolation)
                * float128: no (causes: 'data type no supported')
                * bool: yes

        backend="cv2"

            order=0
                * uint8: yes
                * uint16: yes
                * uint32: no (causes: src data type = 6 is not supported)
                * uint64: no (silently converts to int32)
                * int8: yes
                * int16: yes
                * int32: yes
                * int64: no (silently converts to int32)
                * float16: yes
                * float32: yes
                * float64: yes
                * float128: no (causes: src data type = 13 is not supported)
                * bool: no (causes: src data type = 0 is not supported)

            order=1
                * uint8: yes
                * uint16: yes
                * uint32: no (causes: src data type = 6 is not supported)
                * uint64: no (causes: OpenCV(3.4.5) (...)/imgwarp.cpp:1805: error: (-215:Assertion failed) ifunc != 0
                  in function 'remap')
                * int8: no (causes: OpenCV(3.4.5) (...)/imgwarp.cpp:1805: error: (-215:Assertion failed) ifunc != 0
                  in function 'remap')
                * int16: no (causes: OpenCV(3.4.5) (...)/imgwarp.cpp:1805: error: (-215:Assertion failed) ifunc != 0
                  in function 'remap')
                * int32: no (causes: OpenCV(3.4.5) (...)/imgwarp.cpp:1805: error: (-215:Assertion failed) ifunc != 0
                  in function 'remap')
                * int64: no (causes: OpenCV(3.4.5) (...)/imgwarp.cpp:1805: error: (-215:Assertion failed) ifunc != 0
                  in function 'remap')
                * float16: yes
                * float32: yes
                * float64: yes
                * float128: no (causes: src data type = 13 is not supported)
                * bool: no (causes: src data type = 0 is not supported)

            order=2 to 5:
                as order=1, but int16 supported

        """
        if order == 0 and image.dtype.name in ["uint64", "int64"]:
            raise Exception(("dtypes uint64 and int64 are only supported in ElasticTransformation for order=0, "
                             + "got order=%d with dtype=%s.") % (order, image.dtype.name))

        input_dtype = image.dtype
        if image.dtype.name == "bool":
            image = image.astype(np.float32)
        elif order == 1 and image.dtype.name in ["int8", "int16", "int32"]:
            image = image.astype(np.float64)
        elif order >= 2 and image.dtype.name == "int8":
            image = image.astype(np.int16)
        elif order >= 2 and image.dtype.name == "int32":
            image = image.astype(np.float64)

        shrt_max = 32767
        backend = "cv2"
        if order == 0:
            bad_dtype_cv2 = (image.dtype.name in ["uint32", "uint64", "int64", "float128", "bool"])
        elif order == 1:
            bad_dtype_cv2 = (image.dtype.name in ["uint32", "uint64", "int8", "int16", "int32", "int64", "float128",
                                                  "bool"])
        else:
            bad_dtype_cv2 = (image.dtype.name in ["uint32", "uint64", "int8", "int32", "int64", "float128", "bool"])

        bad_dx_shape_cv2 = (dx.shape[0] >= shrt_max or dx.shape[1] >= shrt_max)
        bad_dy_shape_cv2 = (dy.shape[0] >= shrt_max or dy.shape[1] >= shrt_max)
        if bad_dtype_cv2 or bad_dx_shape_cv2 or bad_dy_shape_cv2:
            backend = "scipy"

        ia.do_assert(image.ndim == 3)
        result = np.copy(image)
        height, width = image.shape[0:2]
        if backend == "scipy":
            h, w = image.shape[0:2]
            y, x = np.meshgrid(np.arange(h).astype(np.float32), np.arange(w).astype(np.float32), indexing='ij')
            x_shifted = x + (-1) * dx
            y_shifted = y + (-1) * dy

            for c in sm.xrange(image.shape[2]):
                remapped_flat = ndimage.interpolation.map_coordinates(
                    image[..., c],
                    (y_shifted.flatten(), x_shifted.flatten()),
                    order=order,
                    cval=cval,
                    mode=mode
                )
                remapped = remapped_flat.reshape((height, width))
                result[..., c] = remapped
        else:
            h, w, nb_channels = image.shape

            y, x = np.meshgrid(np.arange(h).astype(np.float32), np.arange(w).astype(np.float32), indexing='ij')
            x_shifted = x + (-1) * dx
            y_shifted = y + (-1) * dy

            if image.dtype.kind == "f":
                cval = float(cval)
            else:
                cval = int(cval)
            border_mode = cls._MAPPING_MODE_SCIPY_CV2[mode]
            interpolation = cls._MAPPING_ORDER_SCIPY_CV2[order]

            is_nearest_neighbour = (interpolation == cv2.INTER_NEAREST)
            map1, map2 = cv2.convertMaps(x_shifted, y_shifted, cv2.CV_16SC2, nninterpolation=is_nearest_neighbour)
            # remap only supports up to 4 channels
            if nb_channels <= 4:
                result = cv2.remap(image, map1, map2, interpolation=interpolation, borderMode=border_mode, borderValue=cval)
                if image.ndim == 3 and result.ndim == 2:
                    result = result[..., np.newaxis]
            else:
                current_chan_idx = 0
                result = []
                while current_chan_idx < nb_channels:
                    channels = image[..., current_chan_idx:current_chan_idx+4]
                    result_c =  cv2.remap(channels, map1, map2, interpolation=interpolation, borderMode=border_mode,
                                          borderValue=cval)
                    if result_c.ndim == 2:
                        result_c = result_c[..., np.newaxis]
                    result.append(result_c)
                    current_chan_idx += 4
                result = np.concatenate(result, axis=2)

        if result.dtype.name != input_dtype.name:
            result = iadt.restore_dtypes_(result, input_dtype)

        return result


class Rot90(meta.Augmenter):
    """
    Augmenter to rotate images clockwise by multiples of 90 degrees.

    This could also be achieved using ``Affine``, but Rot90 is significantly more efficient.

    dtype support::

        if (keep_size=False)::

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested
            * ``uint64``: yes; tested
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested
            * ``int64``: yes; tested
            * ``float16``: yes; tested
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: yes; tested
            * ``bool``: yes; tested

        if (keep_size=True)::

            minimum of (
                ``imgaug.augmenters.geometric.Rot90(keep_size=False)``,
                :func:`imgaug.imgaug.imresize_many_images`
            )

    Parameters
    ----------
    k : int or list of int or tuple of int or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        How often to rotate clockwise by 90 degrees.

            * If a single int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value from the discrete
              range ``a <= x <= b`` is picked per image.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If imgaug.ALL, then equivalant to list ``[0, 1, 2, 3]``.
            * If StochasticParameter, then that parameter is queried per image
              to sample the value to use.

    keep_size : bool, optional
        After rotation by an odd-valued `k` (e.g. 1 or 3), the resulting image
        may have a different height/width than the original image.
        If this parameter is set to True, then the rotated
        image will be resized to the input image's size. Note that this might also
        cause the augmented image to look distorted.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.


    Examples
    --------
    >>> aug = iaa.Rot90(1)

    Rotates all images by 90 degrees.
    Resizes all images afterwards to keep the size that they had before augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90([1, 3])

    Rotates all images by 90 or 270 degrees.
    Resizes all images afterwards to keep the size that they had before augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90((1, 3))

    Rotates all images by 90, 180 or 270 degrees.
    Resizes all images afterwards to keep the size that they had before augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90((1, 3), keep_size=False)

    Rotates all images by 90, 180 or 270 degrees.
    Does not resize to the original image size afterwards, i.e. each image's size may change.

    """

    def __init__(self, k, keep_size=True, name=None, deterministic=False, random_state=None):
        super(Rot90, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.k = iap.handle_discrete_param(k, "k", value_range=None, tuple_to_uniform=True, list_to_choice=True,
                                           allow_floats=False)
        self.keep_size = keep_size

    def _draw_samples(self, nb_images, random_state):
        return self.k.draw_samples((nb_images,), random_state=random_state)

    def _augment_arrays(self, arrs, random_state, resize_func):
        ks = self._draw_samples(len(arrs), random_state)
        return self._augment_arrays_by_samples(arrs, ks, self.keep_size, resize_func), ks

    @classmethod
    def _augment_arrays_by_samples(cls, arrs, ks, keep_size, resize_func):
        input_was_array = ia.is_np_array(arrs)
        input_dtype = arrs.dtype if input_was_array else None
        arrs_aug = []
        for arr, k_i in zip(arrs, ks):
            arr_aug = np.rot90(arr, k_i, axes=(1, 0))  # adding axes here rotates clock-wise instead of ccw
            if keep_size and arr.shape != arr_aug.shape and resize_func is not None:
                arr_aug = resize_func(arr_aug, arr.shape[0:2])
            arrs_aug.append(arr_aug)
        if keep_size and input_was_array:
            n_shapes = len(set([arr.shape for arr in arrs_aug]))
            if n_shapes == 1:
                arrs_aug = np.array(arrs_aug, dtype=input_dtype)
        return arrs_aug

    def _augment_images(self, images, random_state, parents, hooks):
        resize_func = partial(ia.imresize_single_image, interpolation="cubic")
        images_aug, _ = self._augment_arrays(images, random_state, resize_func)
        return images_aug

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        arrs = [heatmaps_i.arr_0to1 for heatmaps_i in heatmaps]
        arrs_aug, ks = self._augment_arrays(arrs, random_state, None)
        heatmaps_aug = []
        for heatmaps_i, arr_aug, k_i in zip(heatmaps, arrs_aug, ks):
            shape_orig = heatmaps_i.arr_0to1.shape
            heatmaps_i.arr_0to1 = arr_aug
            if self.keep_size:
                heatmaps_i = heatmaps_i.resize(shape_orig[0:2])
            elif k_i % 2 == 1:
                h, w = heatmaps_i.shape[0:2]
                heatmaps_i.shape = tuple([w, h] + list(heatmaps_i.shape[2:]))
            else:
                # keep_size was False, but rotated by a multiple of 2, hence height and width do not change
                pass
            heatmaps_aug.append(heatmaps_i)
        return heatmaps_aug

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        ks = self._draw_samples(nb_images, random_state)
        result = []
        for kpsoi_i, k_i in zip(keypoints_on_images, ks):
            if not kpsoi_i.keypoints:
                result.append(kpsoi_i)
                continue
            elif (k_i % 4) == 0:
                result.append(kpsoi_i)
            else:
                k_i = int(k_i) % 4  # this is also correct when k_i is negative
                kps_aug = []
                h, w = kpsoi_i.shape[0:2]
                h_aug, w_aug = (h, w) if (k_i % 2) == 0 else (w, h)

                for kp in kpsoi_i.keypoints:
                    y, x = kp.y, kp.x
                    yr, xr = y, x
                    wr, hr = w, h
                    for _ in sm.xrange(k_i):
                        # FIXME using ((hr - 1) - yr) here seems wrong for float-based coordinates, should likely be
                        #       (hr - yr). Same problem as in horizontal flipping.
                        xr, yr = (hr - 1) - yr, xr
                        wr, hr = hr, wr
                    kps_aug.append(ia.Keypoint(x=xr, y=yr))

                shape_aug = tuple([h_aug, w_aug] + list(kpsoi_i.shape[2:]))
                kpsoi_i_aug = ia.KeypointsOnImage(kps_aug, shape=shape_aug)
                if self.keep_size and (h, w) != (h_aug, w_aug):
                    kpsoi_i_aug = kpsoi_i_aug.on(kpsoi_i.shape)
                    kpsoi_i_aug.shape = kpsoi_i.shape

                result.append(kpsoi_i_aug)
        return result

    def get_parameters(self):
        return [self.k, self.keep_size]
