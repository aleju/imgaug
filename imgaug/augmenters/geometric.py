"""Augmenters that apply affine or similar transformations.

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
import functools

import numpy as np
from scipy import ndimage
from skimage import transform as tf
import cv2
import six.moves as sm

from . import meta
from . import blur as blur_lib
import imgaug as ia
from imgaug.augmentables.polys import _ConcavePolygonRecoverer
from .. import parameters as iap
from .. import dtypes as iadt


_VALID_DTYPES_CV2_ORDER_0 = {"uint8", "uint16", "int8", "int16", "int32",
                             "float16", "float32", "float64",
                             "bool"}
_VALID_DTYPES_CV2_ORDER_NOT_0 = {"uint8", "uint16", "int8", "int16",
                                 "float16", "float32", "float64",
                                 "bool"}

# skimage | cv2
# 0       | cv2.INTER_NEAREST
# 1       | cv2.INTER_LINEAR
# 2       | -
# 3       | cv2.INTER_CUBIC
# 4       | -
_AFFINE_INTERPOLATION_ORDER_SKIMAGE_TO_CV2 = {
    0: cv2.INTER_NEAREST,
    1: cv2.INTER_LINEAR,
    2: cv2.INTER_CUBIC,
    3: cv2.INTER_CUBIC,
    4: cv2.INTER_CUBIC
}

# constant, edge, symmetric, reflect, wrap
# skimage   | cv2
# constant  | cv2.BORDER_CONSTANT
# edge      | cv2.BORDER_REPLICATE
# symmetric | cv2.BORDER_REFLECT
# reflect   | cv2.BORDER_REFLECT_101
# wrap      | cv2.BORDER_WRAP
_AFFINE_MODE_SKIMAGE_TO_CV2 = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "symmetric": cv2.BORDER_REFLECT,
    "reflect": cv2.BORDER_REFLECT_101,
    "wrap": cv2.BORDER_WRAP
}


def _handle_order_arg(order, backend):
    # Peformance in skimage for Affine:
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
            return iap.Choice([0, 1, 3])
        else:
            # dont use order=2 (bi-quadratic) because that is apparently
            # currently not recommended (and throws a warning)
            return iap.Choice([0, 1, 3, 4, 5])
    elif ia.is_single_integer(order):
        assert 0 <= order <= 5, (
            "Expected order's integer value to be in the interval [0, 5], "
            "got %d." % (order,))
        if backend == "cv2":
            assert order in [0, 1, 3], (
                "Backend \"cv2\" and order=%d was chosen, but cv2 backend "
                "can only handle order 0, 1 or 3." % (order,))
        return iap.Deterministic(order)
    elif isinstance(order, list):
        assert all([ia.is_single_integer(val) for val in order]), (
            "Expected order list to only contain integers, "
            "got types %s." % (str([type(val) for val in order]),))
        assert all([0 <= val <= 5 for val in order]), (
            "Expected all of order's integer values to be in range "
            "0 <= x <= 5, got %s." % (str(order),))
        if backend == "cv2":
            assert all([val in [0, 1, 3] for val in order]), (
                "cv2 backend can only handle order 0, 1 or 3. Got order "
                "list of %s." % (order,))
        return iap.Choice(order)
    elif isinstance(order, iap.StochasticParameter):
        return order
    else:
        raise Exception(
            "Expected order to be imgaug.ALL, int, list of int or "
            "StochasticParameter, got %s." % (type(order),))


def _handle_cval_arg(cval):
    if cval == ia.ALL:
        # TODO change this so that it is dynamically created per image
        #      (or once per dtype)
        return iap.Uniform(0, 255)  # skimage transform expects float
    else:
        return iap.handle_continuous_param(
            cval, "cval", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)


# currently used for Affine and PiecewiseAffine
# TODO use iap.handle_categorical_string_param() here
def _handle_mode_arg(mode):
    if mode == ia.ALL:
        return iap.Choice(["constant", "edge", "symmetric",
                           "reflect", "wrap"])
    elif ia.is_string(mode):
        return iap.Deterministic(mode)
    elif isinstance(mode, list):
        assert all([ia.is_string(val) for val in mode]), (
            "Expected list of modes to only contain strings, got "
            "types %s" % (", ".join([str(type(v)) for v in mode])))
        return iap.Choice(mode)
    elif isinstance(mode, iap.StochasticParameter):
        return mode
    else:
        raise Exception(
            "Expected mode to be imgaug.ALL, a string, a list of strings "
            "or StochasticParameter, got %s." % (type(mode),))


def _warp_affine_arr(arr, matrix, order=1, mode="constant", cval=0,
                     output_shape=None, backend="auto"):
    if ia.is_single_integer(cval):
        cval = [cval] * len(arr.shape[2])

    # no changes to zero-sized arrays
    if arr.size == 0:
        return arr

    min_value, _center_value, max_value = \
        iadt.get_value_range_of_dtype(arr.dtype)

    cv2_bad_order = order not in [0, 1, 3]
    if order == 0:
        cv2_bad_dtype = (
            arr.dtype.name
            not in _VALID_DTYPES_CV2_ORDER_0)
    else:
        cv2_bad_dtype = (
            arr.dtype.name
            not in _VALID_DTYPES_CV2_ORDER_NOT_0
        )
    cv2_impossible = cv2_bad_order or cv2_bad_dtype
    use_skimage = (
        backend == "skimage"
        or (backend == "auto" and cv2_impossible)
    )
    if use_skimage:
        # cval contains 3 values as cv2 can handle 3, but
        # skimage only 1
        cval = cval[0]
        # skimage does not clip automatically
        cval = max(min(cval, max_value), min_value)
        image_warped = _warp_affine_arr_skimage(
            arr,
            matrix,
            cval=cval,
            mode=mode,
            order=order,
            output_shape=output_shape
        )
    else:
        assert not cv2_bad_dtype, (
            not cv2_bad_dtype,
            "cv2 backend in Affine got a dtype %s, which it "
            "cannot handle. Try using a different dtype or set "
            "order=0." % (
                arr.dtype,))
        image_warped = _warp_affine_arr_cv2(
            arr,
            matrix,
            cval=tuple([int(v) for v in cval]),
            mode=mode,
            order=order,
            output_shape=output_shape
        )
    return image_warped


def _warp_affine_arr_skimage(arr, matrix, cval, mode, order, output_shape):
    iadt.gate_dtypes(
        arr,
        allowed=["bool",
                 "uint8", "uint16", "uint32",
                 "int8", "int16", "int32",
                 "float16", "float32", "float64"],
        disallowed=["uint64", "uint128", "uint256",
                    "int64", "int128", "int256",
                    "float96", "float128", "float256"],
        augmenter=None)

    input_dtype = arr.dtype

    image_warped = tf.warp(
        arr,
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

    return image_warped


def _warp_affine_arr_cv2(arr, matrix, cval, mode, order, output_shape):
    iadt.gate_dtypes(
        arr,
        allowed=["bool",
                 "uint8", "uint16",
                 "int8", "int16", "int32",
                 "float16", "float32", "float64"],
        disallowed=["uint32", "uint64", "uint128", "uint256",
                    "int64", "int128", "int256",
                    "float96", "float128", "float256"],
        augmenter=None)

    if order != 0:
        assert arr.dtype.name != "int32", (
            "Affine only supports cv2-based transformations of int32 "
            "arrays when using order=0, but order was set to %d." % (
                order,))

    input_dtype = arr.dtype
    if input_dtype in [np.bool_, np.float16]:
        arr = arr.astype(np.float32)
    elif input_dtype == np.int8 and order != 0:
        arr = arr.astype(np.int16)

    dsize = (
        int(np.round(output_shape[1])),
        int(np.round(output_shape[0]))
    )

    # map key X from skimage to cv2 or fall back to key X
    mode = _AFFINE_MODE_SKIMAGE_TO_CV2.get(mode, mode)
    order = _AFFINE_INTERPOLATION_ORDER_SKIMAGE_TO_CV2.get(order, order)

    # TODO this uses always a tuple of 3 values for cval, even if
    #      #chans != 3, works with 1d but what in other cases?
    nb_channels = arr.shape[-1]
    if nb_channels <= 3:
        # TODO this block can also be when order==0 for any nb_channels,
        #      but was deactivated for now, because cval would always
        #      contain 3 values and not nb_channels values
        image_warped = cv2.warpAffine(
            arr,
            matrix.params[:2],
            dsize=dsize,
            flags=order,
            borderMode=mode,
            borderValue=cval
        )

        # cv2 warp drops last axis if shape is (H, W, 1)
        if image_warped.ndim == 2:
            image_warped = image_warped[..., np.newaxis]
    else:
        # warp each channel on its own, re-add channel axis, then stack
        # the result from a list of [H, W, 1] to (H, W, C).
        image_warped = [
            cv2.warpAffine(
                arr[:, :, c],
                matrix.params[:2],
                dsize=dsize,
                flags=order,
                borderMode=mode,
                borderValue=tuple([cval[0]])
            )
            for c in sm.xrange(nb_channels)]
        image_warped = np.stack(image_warped, axis=-1)

    if input_dtype.name == "bool":
        image_warped = image_warped > 0.5
    elif input_dtype.name in ["int8", "float16"]:
        image_warped = iadt.restore_dtypes_(image_warped, input_dtype)

    return image_warped


def _compute_affine_warp_output_shape(matrix, input_shape):
    height, width = input_shape[:2]

    if height == 0 or width == 0:
        return matrix, input_shape

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


class _AffineSamplingResult(object):
    def __init__(self, scale=None, translate=None, rotate=None, shear=None,
                 cval=None, mode=None, order=None):
        self.scale = scale
        self.translate = translate
        self.rotate = rotate
        self.shear = shear
        self.cval = cval
        self.mode = mode
        self.order = order

    def to_matrix(self, idx, arr_shape, fit_output, shift_add=(0.5, 0.5)):
        height, width = arr_shape[0:2]
        # for images we use additional shifts of (0.5, 0.5) as otherwise
        # we get an ugly black border for 90deg rotations
        shift_y = height / 2.0 - shift_add[0]
        shift_x = width / 2.0 - shift_add[1]
        scale_y = self.scale[1][idx]  # TODO 1 and 0 should be inverted here
        scale_x = self.scale[0][idx]
        translate_y = self.translate[1][idx]  # TODO same as above
        translate_x = self.translate[0][idx]
        if ia.is_single_float(translate_y):
            translate_y_px = int(
                np.round(translate_y * height))
        else:
            translate_y_px = translate_y
        if ia.is_single_float(translate_x):
            translate_x_px = int(
                np.round(translate_x * width))
        else:
            translate_x_px = translate_x

        rotation_deg, shear_deg = np.deg2rad([
            self.rotate[idx], self.shear[idx]])

        matrix_to_topleft = tf.SimilarityTransform(
            translation=[-shift_x, -shift_y])
        matrix_transforms = tf.AffineTransform(
            scale=(scale_x, scale_y),
            translation=(translate_x_px, translate_y_px),
            rotation=rotation_deg,
            shear=shear_deg
        )
        matrix_to_center = tf.SimilarityTransform(
            translation=[shift_x, shift_y])
        matrix = (matrix_to_topleft
                  + matrix_transforms
                  + matrix_to_center)
        if fit_output:
            return _compute_affine_warp_output_shape(matrix, arr_shape)
        return matrix, arr_shape

    def to_matrix_cba(self, idx, arr_shape, fit_output, shift_add=(0.0, 0.0)):
        return self.to_matrix(idx, arr_shape, fit_output, shift_add)

    def copy(self):
        return _AffineSamplingResult(
            scale=self.scale,
            translate=self.translate,
            rotate=self.rotate,
            shear=self.shear,
            cval=self.cval,
            mode=self.mode,
            order=self.order
        )


def _is_identity_matrix(matrix, eps=1e-4):
    identity = np.float32([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    return np.average(np.abs(identity - matrix.params)) <= eps


class Affine(meta.Augmenter):
    """
    Augmenter to apply affine transformations to images.

    This is mostly a wrapper around the corresponding classes and functions
    in OpenCV and skimage..

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

            - (1) scikit-image converts internally to float64, which might
                  affect the accuracy of large integers. In tests this seemed
                  to not be an issue.
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

            - (1) scikit-image converts internally to float64, which might
                  affect the accuracy of large integers. In tests this seemed
                  to not be an issue.
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

                - (1) scikit-image converts internally to ``float64``, which
                      might affect the accuracy of large integers. In tests
                      this seemed to not be an issue.
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
            - (2) causes cv2 error: ``cv2.error: OpenCV(3.4.4)
                  (...)imgwarp.cpp:1805: error:
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
            - (2) causes cv2 error: ``cv2.error: OpenCV(3.4.4)
                  (...)imgwarp.cpp:1805: error:
                  (-215:Assertion failed) ifunc != 0 in function 'remap'``
            - (3) mapped internally to ``int16``
            - (4) mapped internally to ``float32``


    Parameters
    ----------
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        Scaling factor to use, where ``1.0`` denotes "no change" and
        ``0.5`` is zoomed out to ``50`` percent of the original size.

            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``. That value will be
              used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    translate_percent : None or number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        Translation as a fraction of the image height/width (x-translation,
        y-translation), where ``0`` denotes "no change" and ``0.5`` denotes
        "half of the axis size".

            * If ``None`` then equivalent to ``0.0`` unless `translate_px` has
              a value other than ``None``.
            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``. That sampled fraction
              value will be used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    translate_px : None or int or tuple of int or list of int or imgaug.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        Translation in pixels.

            * If ``None`` then equivalent to ``0`` unless `translate_percent`
              has a value other than ``None``.
            * If a single int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the discrete interval ``[a..b]``. That number
              will be used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    rotate : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Rotation in degrees (**NOT** radians), i.e. expected value range is
        around ``[-360, 360]``. Rotation happens around the *center* of the
        image, not the top left corner as in some other frameworks.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]`` and used as the rotation
              value.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the rotation value per image.

    shear : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Shear in degrees (**NOT** radians), i.e. expected value range is
        around ``[-360, 360]``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]`` and be used as the
              rotation value.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then this parameter will be used
              to sample the shear value per image.

    order : int or iterable of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use. Same meaning as in ``skimage``:

            * ``0``: ``Nearest-neighbor``
            * ``1``: ``Bi-linear`` (default)
            * ``2``: ``Bi-quadratic`` (not recommended by skimage)
            * ``3``: ``Bi-cubic``
            * ``4``: ``Bi-quartic``
            * ``5``: ``Bi-quintic``

        Method ``0`` and ``1`` are fast, ``3`` is a bit slower, ``4`` and
        ``5`` are very slow. If the backend is ``cv2``, the mapping to
        OpenCV's interpolation modes is as follows:

            * ``0`` -> ``cv2.INTER_NEAREST``
            * ``1`` -> ``cv2.INTER_LINEAR``
            * ``2`` -> ``cv2.INTER_CUBIC``
            * ``3`` -> ``cv2.INTER_CUBIC``
            * ``4`` -> ``cv2.INTER_CUBIC``

        As datatypes this parameter accepts:

            * If a single ``int``, then that order will be used for all images.
            * If a list, then a random value will be sampled from that list
              per image.
            * If ``imgaug.ALL``, then equivalant to list ``[0, 1, 3, 4, 5]``
              in case of ``backend=skimage`` and otherwise ``[0, 1, 3]``.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the order value to use.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant value to use when filling in newly created pixels.
        (E.g. translating by 1px to the right will create a new 1px-wide
        column of pixels on the left of the image).  The value is only used
        when `mode=constant`. The expected value range is ``[0, 255]`` for
        ``uint8`` images. It may be a float value.

            * If this is a single number, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple ``(a, b)``, then three values (for three image
              channels) will be uniformly sampled per image from the
              interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If ``imgaug.ALL`` then equivalent to tuple ``(0, 255)`.
            * If a ``StochasticParameter``, a new value will be sampled from
              the parameter per image.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Method to use when filling in newly created pixels.
        Same meaning as in ``skimage`` (and :func:`numpy.pad`):

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
            * If a list of strings, then a random mode will be picked
              from that list per image.
            * If ``imgaug.ALL``, then a random mode from all possible modes
              will be picked.
            * If ``StochasticParameter``, then the mode will be sampled from
              that parameter per image, i.e. it must return only the above
              mentioned strings.

    fit_output : bool, optional
        Whether to modify the affine transformation so that the whole output
        image is always contained in the image plane (``True``) or accept
        parts of the image being outside the image plane (``False``).
        This can be thought of as first applying the affine transformation
        and then applying a second transformation to "zoom in" on the new
        image so that it fits the image plane,
        This is useful to avoid corners of the image being outside of the image
        plane after applying rotations. It will however negate translation
        and scaling.
        Note also that activating this may lead to image sizes differing from
        the input image sizes. To avoid this, wrap ``Affine`` in
        :class:`imgaug.augmenters.size.KeepSizeByResize`,
        e.g. ``KeepSizeByResize(Affine(...))``.

    backend : str, optional
        Framework to use as a backend. Valid values are ``auto``, ``skimage``
        (scikit-image's warp) and ``cv2`` (OpenCV's warp).
        If ``auto`` is used, the augmenter will automatically try
        to use ``cv2`` whenever possible (order must be in ``[0, 1, 3]``). It
        will silently fall back to skimage if order/dtype is not supported by
        cv2. cv2 is generally faster than skimage. It also supports RGB cvals,
        while skimage will resort to intensity cvals (i.e. 3x the same value
        as RGB). If ``cv2`` is chosen and order is ``2`` or ``4``, it will
        automatically fall back to order ``3``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Affine(scale=2.0)

    Zoom in on all images by a factor of ``2``.

    >>> aug = iaa.Affine(translate_px=16)

    Translate all images on the x- and y-axis by 16 pixels (towards the
    bottom right) and fill up any new pixels with zero (black values).

    >>> aug = iaa.Affine(translate_percent=0.1)

    Translate all images on the x- and y-axis by ``10`` percent of their
    width/height (towards the bottom right). The pixel values are computed
    per axis based on that axis' size. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.Affine(rotate=35)

    Rotate all images by ``35`` *degrees*. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.Affine(shear=15)

    Shear all images by ``15`` *degrees*. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.Affine(translate_px=(-16, 16))

    Translate all images on the x- and y-axis by a random value
    between ``-16`` and ``16`` pixels (to the bottom right) and fill up any new
    pixels with zero (black values). The translation value is sampled once
    per image and is the same for both axis.

    >>> aug = iaa.Affine(translate_px={"x": (-16, 16), "y": (-4, 4)})

    Translate all images on the x-axis by a random value
    between ``-16`` and ``16`` pixels (to the right) and on the y-axis by a
    random value between ``-4`` and ``4`` pixels to the bottom. The sampling
    happens independently per axis, so even if both intervals were identical,
    the sampled axis-wise values would likely be different.
    This also fills up any new pixels with zero (black values).

    >>> aug = iaa.Affine(scale=2.0, order=[0, 1])

    Same as in the above `scale` example, but uses (randomly) either
    nearest neighbour interpolation or linear interpolation. If `order` is
    not specified, ``order=1`` would be used by default.

    >>> aug = iaa.Affine(translate_px=16, cval=(0, 255))

    Same as in the `translate_px` example above, but newly created pixels
    are now filled with a random color (sampled once per image and the
    same for all newly created pixels within that image).

    >>> aug = iaa.Affine(translate_px=16, mode=["constant", "edge"])

    Similar to the previous example, but the newly created pixels are
    filled with black pixels in half of all images (mode ``constant`` with
    default `cval` being ``0``) and in the other half of all images using
    ``edge`` mode, which repeats the color of the spatially closest pixel
    of the corresponding image edge.

    """

    def __init__(self, scale=1.0, translate_percent=None, translate_px=None,
                 rotate=0.0, shear=0.0, order=1, cval=0, mode="constant",
                 fit_output=False, backend="auto",
                 name=None, deterministic=False, random_state=None):
        super(Affine, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        assert backend in ["auto", "skimage", "cv2"], (
            "Expected 'backend' to be \"auto\", \"skimage\" or \"cv2\", "
            "got %s." % (backend,))
        self.backend = backend
        self.order = _handle_order_arg(order, backend)
        self.cval = _handle_cval_arg(cval)
        self.mode = _handle_mode_arg(mode)
        self.scale = self._handle_scale_arg(scale)
        self.translate = self._handle_translate_arg(
            translate_px, translate_percent)
        self.rotate = iap.handle_continuous_param(
            rotate, "rotate", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        self.shear = iap.handle_continuous_param(
            shear, "shear", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        self.fit_output = fit_output

        # Special order, mode and cval parameters for heatmaps and
        # segmentation maps. These may either be None or a fixed value.
        # Stochastic parameters are currently *not* supported.
        # If set to None, the same values as for images will be used.
        # That is really not recommended for the cval parameter.
        #
        # Segmentation map augmentation by default always pads with a
        # constant value of 0 (background class id), and always uses nearest
        # neighbour interpolation. While other pad modes and BG class ids
        # could be used, the interpolation mode has to be NN as any other
        # mode would lead to averaging class ids, which makes no sense to do.
        self._order_heatmaps = 3
        self._order_segmentation_maps = 0
        self._mode_heatmaps = "constant"
        self._mode_segmentation_maps = "constant"
        self._cval_heatmaps = 0
        self._cval_segmentation_maps = 0

    @classmethod
    def _handle_scale_arg(cls, scale):
        if isinstance(scale, dict):
            assert "x" in scale or "y" in scale, (
                "Expected scale dictionary to contain at least key \"x\" or "
                "key \"y\". Found neither of them.")
            x = scale.get("x", 1.0)
            y = scale.get("y", 1.0)
            return (
                iap.handle_continuous_param(
                    x, "scale['x']", value_range=(0+1e-4, None),
                    tuple_to_uniform=True, list_to_choice=True),
                iap.handle_continuous_param(
                    y, "scale['y']", value_range=(0+1e-4, None),
                    tuple_to_uniform=True, list_to_choice=True)
            )
        else:
            return iap.handle_continuous_param(
                scale, "scale", value_range=(0+1e-4, None),
                tuple_to_uniform=True, list_to_choice=True)

    @classmethod
    def _handle_translate_arg(cls, translate_px, translate_percent):
        if translate_percent is None and translate_px is None:
            translate_px = 0

        assert translate_percent is None or translate_px is None, (
            "Expected either translate_percent or translate_px to be "
            "provided, but neither of them was.")

        if translate_percent is not None:
            # translate by percent
            if isinstance(translate_percent, dict):
                assert "x" in translate_percent or "y" in translate_percent, (
                    "Expected translate_percent dictionary to contain at "
                    "least key \"x\" or key \"y\". Found neither of them.")
                x = translate_percent.get("x", 0)
                y = translate_percent.get("y", 0)
                return (
                    iap.handle_continuous_param(
                        x, "translate_percent['x']", value_range=None,
                        tuple_to_uniform=True, list_to_choice=True),
                    iap.handle_continuous_param(
                        y, "translate_percent['y']", value_range=None,
                        tuple_to_uniform=True, list_to_choice=True)
                )
            else:
                return iap.handle_continuous_param(
                    translate_percent, "translate_percent", value_range=None,
                    tuple_to_uniform=True, list_to_choice=True)
        else:
            # translate by pixels
            if isinstance(translate_px, dict):
                assert "x" in translate_px or "y" in translate_px, (
                    "Expected translate_px dictionary to contain at "
                    "least key \"x\" or key \"y\". Found neither of them.")
                x = translate_px.get("x", 0)
                y = translate_px.get("y", 0)
                return (
                    iap.handle_discrete_param(
                        x, "translate_px['x']", value_range=None,
                        tuple_to_uniform=True, list_to_choice=True,
                        allow_floats=False),
                    iap.handle_discrete_param(
                        y, "translate_px['y']", value_range=None,
                        tuple_to_uniform=True, list_to_choice=True,
                        allow_floats=False)
                )
            else:
                return iap.handle_discrete_param(
                    translate_px, "translate_px", value_range=None,
                    tuple_to_uniform=True, list_to_choice=True,
                    allow_floats=False)

    def _augment_batch(self, batch, random_state, parents, hooks):
        samples = self._draw_samples(batch.nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images,
                                                           samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps, samples, "arr_0to1", self._cval_heatmaps,
                self._mode_heatmaps, self._order_heatmaps, "float32")

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, samples, "arr",
                self._cval_segmentation_maps, self._mode_segmentation_maps,
                self._order_segmentation_maps, "int32")

        for augm_name in ["keypoints", "bounding_boxes", "polygons",
                          "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                for i, cbaoi in enumerate(augm_value):
                    matrix, output_shape = samples.to_matrix_cba(
                        i, cbaoi.shape, self.fit_output)

                    if (not _is_identity_matrix(matrix)
                            and not cbaoi.empty
                            and not (0 in cbaoi.shape[0:2])):
                        coords = cbaoi.to_xy_array()
                        coords_aug = tf.matrix_transform(coords, matrix.params)
                        cbaoi = cbaoi.fill_from_xy_array_(coords_aug)

                    cbaoi.shape = output_shape
                    augm_value[i] = cbaoi

        return batch

    def _augment_images_by_samples(self, images, samples,
                                   return_matrices=False):
        nb_images = len(images)
        input_was_array = ia.is_np_array(images)
        input_dtype = None if not input_was_array else images.dtype
        result = []
        if return_matrices:
            matrices = [None] * nb_images
        for i in sm.xrange(nb_images):
            image = images[i]

            matrix, output_shape = samples.to_matrix(i, image.shape,
                                                     self.fit_output)

            cval = samples.cval[i]
            mode = samples.mode[i]
            order = samples.order[i]

            if not _is_identity_matrix(matrix):
                image_warped = _warp_affine_arr(
                    image, matrix,
                    order=order, mode=mode, cval=cval,
                    output_shape=output_shape, backend=self.backend)

                result.append(image_warped)
            else:
                result.append(image)

            if return_matrices:
                matrices[i] = matrix

        # the shapes can change due to fit_output, then it may not be possible
        # to return an array, even when the input was an array
        if input_was_array:
            nb_shapes = len(set([image.shape for image in result]))
            if nb_shapes == 1:
                result = np.array(result, input_dtype)

        if return_matrices:
            result = (result, matrices)

        return result

    def _augment_maps_by_samples(self, augmentables, samples,
                                 arr_attr_name, cval, mode, order, cval_dtype):
        nb_images = len(augmentables)

        samples = samples.copy()
        if cval is not None:
            samples.cval = np.full((nb_images, 1), cval, dtype=cval_dtype)
        if mode is not None:
            samples.mode = [mode] * nb_images
        if order is not None:
            samples.order = [order] * nb_images

        arrs = [getattr(augmentable, arr_attr_name)
                for augmentable in augmentables]
        arrs_aug, matrices = self._augment_images_by_samples(
            arrs, samples, return_matrices=True)

        gen = zip(augmentables, arrs_aug, matrices, samples.order)
        for augmentable_i, arr_aug, matrix, order_i in gen:
            # skip augmented HM/SM arrs for which the images were not
            # augmented due to being zero-sized
            if 0 in augmentable_i.shape:
                continue

            # order=3 matches cubic interpolation and can cause values to go
            # outside of the range [0.0, 1.0] not clear whether 4+ also do that
            # We don't clip here for Segmentation Maps, because for these
            # the value range isn't clearly limited to [0, 1] (and they should
            # also never use order=3 to begin with).
            # TODO add test for this
            if order_i >= 3 and isinstance(augmentable_i, ia.HeatmapsOnImage):
                arr_aug = np.clip(arr_aug, 0.0, 1.0, out=arr_aug)

            setattr(augmentable_i, arr_attr_name, arr_aug)
            if self.fit_output:
                _, output_shape_i = _compute_affine_warp_output_shape(
                    matrix, augmentable_i.shape)
            else:
                output_shape_i = augmentable_i.shape
            augmentable_i.shape = output_shape_i
        return augmentables

    def _draw_samples(self, nb_samples, random_state):
        rngs = random_state.duplicate(11)

        if isinstance(self.scale, tuple):
            scale_samples = (
                self.scale[0].draw_samples((nb_samples,), random_state=rngs[0]),
                self.scale[1].draw_samples((nb_samples,), random_state=rngs[1]),
            )
        else:
            scale_samples = self.scale.draw_samples((nb_samples,),
                                                    random_state=rngs[2])
            scale_samples = (scale_samples, scale_samples)

        if isinstance(self.translate, tuple):
            translate_samples = (
                self.translate[0].draw_samples((nb_samples,),
                                               random_state=rngs[3]),
                self.translate[1].draw_samples((nb_samples,),
                                               random_state=rngs[4]),
            )
        else:
            translate_samples = self.translate.draw_samples(
                (nb_samples,), random_state=rngs[5])
            translate_samples = (translate_samples, translate_samples)

        rotate_samples = self.rotate.draw_samples((nb_samples,),
                                                  random_state=rngs[6])
        shear_samples = self.shear.draw_samples((nb_samples,),
                                                random_state=rngs[7])

        cval_samples = self.cval.draw_samples((nb_samples, 3),
                                              random_state=rngs[8])
        mode_samples = self.mode.draw_samples((nb_samples,),
                                              random_state=rngs[9])
        order_samples = self.order.draw_samples((nb_samples,),
                                                random_state=rngs[10])

        return _AffineSamplingResult(
            scale=scale_samples,
            translate=translate_samples,
            rotate=rotate_samples,
            shear=shear_samples,
            cval=cval_samples,
            mode=mode_samples,
            order=order_samples)

    def get_parameters(self):
        return [
            self.scale, self.translate, self.rotate, self.shear, self.order,
            self.cval, self.mode, self.backend, self.fit_output]


class AffineCv2(meta.Augmenter):
    """
    Augmenter to apply affine transformations to images using cv2 (i.e. opencv)
    backend.

    .. warning ::

        This augmenter might be removed in the future as ``Affine``
        already offers a cv2 backend (use ``backend="cv2"``).

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
    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        Scaling factor to use, where ``1.0`` denotes \"no change\" and
        ``0.5`` is zoomed out to ``50`` percent of the original size.

            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``. That value will be
              used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    translate_percent : number or tuple of number or list of number or imgaug.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        Translation as a fraction of the image height/width (x-translation,
        y-translation), where ``0`` denotes "no change" and ``0.5`` denotes
        "half of the axis size".

            * If ``None`` then equivalent to ``0.0`` unless `translate_px` has
              a value other than ``None``.
            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``. That sampled fraction
              value will be used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    translate_px : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        Translation in pixels.

            * If ``None`` then equivalent to ``0`` unless `translate_percent`
              has a value other than ``None``.
            * If a single int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the discrete interval ``[a..b]``. That number
              will be used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    rotate : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Rotation in degrees (**NOT** radians), i.e. expected value range is
        around ``[-360, 360]``. Rotation happens around the *center* of the
        image, not the top left corner as in some other frameworks.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]`` and used as the rotation
              value.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the rotation value per image.

    shear : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Shear in degrees (**NOT** radians), i.e. expected value range is
        around ``[-360, 360]``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]`` and be used as the
              rotation value.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then this parameter will be used
              to sample the shear value per image.

    order : int or list of int or str or list of str or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use. Allowed are:

            * ``cv2.INTER_NEAREST`` (nearest-neighbor interpolation)
            * ``cv2.INTER_LINEAR`` (bilinear interpolation, used by default)
            * ``cv2.INTER_CUBIC`` (bicubic interpolation over ``4x4`` pixel
                neighborhood)
            * ``cv2.INTER_LANCZOS4``
            * string ``nearest`` (same as ``cv2.INTER_NEAREST``)
            * string ``linear`` (same as ``cv2.INTER_LINEAR``)
            * string ``cubic`` (same as ``cv2.INTER_CUBIC``)
            * string ``lanczos4`` (same as ``cv2.INTER_LANCZOS``)

        ``INTER_NEAREST`` (nearest neighbour interpolation) and
        ``INTER_NEAREST`` (linear interpolation) are the fastest.

            * If a single ``int``, then that order will be used for all images.
            * If a string, then it must be one of: ``nearest``, ``linear``,
              ``cubic``, ``lanczos4``.
            * If an iterable of ``int``/``str``, then for each image a random
              value will be sampled from that iterable (i.e. list of allowed
              order values).
            * If ``imgaug.ALL``, then equivalant to list ``[cv2.INTER_NEAREST,
              cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]``.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the order value to use.

    cval : number or tuple of number or list of number or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant value to use when filling in newly created pixels.
        (E.g. translating by 1px to the right will create a new 1px-wide
        column of pixels on the left of the image).  The value is only used
        when `mode=constant`. The expected value range is ``[0, 255]`` for
        ``uint8`` images. It may be a float value.

            * If this is a single number, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple ``(a, b)``, then three values (for three image
              channels) will be uniformly sampled per image from the
              interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If ``imgaug.ALL`` then equivalent to tuple ``(0, 255)`.
            * If a ``StochasticParameter``, a new value will be sampled from
              the parameter per image.

    mode : int or str or list of str or list of int or imgaug.ALL or imgaug.parameters.StochasticParameter,
           optional
        Method to use when filling in newly created pixels.
        Same meaning as in OpenCV's border mode. Let ``abcdefgh`` be an image's
        content and ``|`` be an image boundary after which new pixels are
        filled in, then the valid modes and their behaviour are the following:

            * ``cv2.BORDER_REPLICATE``: ``aaaaaa|abcdefgh|hhhhhhh``
            * ``cv2.BORDER_REFLECT``: ``fedcba|abcdefgh|hgfedcb``
            * ``cv2.BORDER_REFLECT_101``: ``gfedcb|abcdefgh|gfedcba``
            * ``cv2.BORDER_WRAP``: ``cdefgh|abcdefgh|abcdefg``
            * ``cv2.BORDER_CONSTANT``: ``iiiiii|abcdefgh|iiiiiii``,
               where ``i`` is the defined cval.
            * ``replicate``: Same as ``cv2.BORDER_REPLICATE``.
            * ``reflect``: Same as ``cv2.BORDER_REFLECT``.
            * ``reflect_101``: Same as ``cv2.BORDER_REFLECT_101``.
            * ``wrap``: Same as ``cv2.BORDER_WRAP``.
            * ``constant``: Same as ``cv2.BORDER_CONSTANT``.

        The datatype of the parameter may be:

            * If a single ``int``, then it must be one of the ``cv2.BORDER_*``
              constants.
            * If a single string, then it must be one of: ``replicate``,
              ``reflect``, ``reflect_101``, ``wrap``, ``constant``.
            * If a list of ``int``/``str``, then per image a random mode will
              be picked from that list.
            * If ``imgaug.ALL``, then a random mode from all possible modes
              will be picked.
            * If ``StochasticParameter``, then the mode will be sampled from
              that parameter per image, i.e. it must return only the above
              mentioned strings.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AffineCv2(scale=2.0)

    Zoom in on all images by a factor of ``2``.

    >>> aug = iaa.AffineCv2(translate_px=16)

    Translate all images on the x- and y-axis by 16 pixels (towards the
    bottom right) and fill up any new pixels with zero (black values).

    >>> aug = iaa.AffineCv2(translate_percent=0.1)

    Translate all images on the x- and y-axis by ``10`` percent of their
    width/height (towards the bottom right). The pixel values are computed
    per axis based on that axis' size. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(rotate=35)

    Rotate all images by ``35`` *degrees*. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(shear=15)

    Shear all images by ``15`` *degrees*. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(translate_px=(-16, 16))

    Translate all images on the x- and y-axis by a random value
    between ``-16`` and ``16`` pixels (to the bottom right) and fill up any new
    pixels with zero (black values). The translation value is sampled once
    per image and is the same for both axis.

    >>> aug = iaa.AffineCv2(translate_px={"x": (-16, 16), "y": (-4, 4)})

    Translate all images on the x-axis by a random value
    between ``-16`` and ``16`` pixels (to the right) and on the y-axis by a
    random value between ``-4`` and ``4`` pixels to the bottom. The sampling
    happens independently per axis, so even if both intervals were identical,
    the sampled axis-wise values would likely be different.
    This also fills up any new pixels with zero (black values).

    >>> aug = iaa.AffineCv2(scale=2.0, order=[0, 1])

    Same as in the above `scale` example, but uses (randomly) either
    nearest neighbour interpolation or linear interpolation. If `order` is
    not specified, ``order=1`` would be used by default.

    >>> aug = iaa.AffineCv2(translate_px=16, cval=(0, 255))

    Same as in the `translate_px` example above, but newly created pixels
    are now filled with a random color (sampled once per image and the
    same for all newly created pixels within that image).

    >>> aug = iaa.AffineCv2(translate_px=16, mode=["constant", "replicate"])

    Similar to the previous example, but the newly created pixels are
    filled with black pixels in half of all images (mode ``constant`` with
    default `cval` being ``0``) and in the other half of all images using
    ``replicate`` mode, which repeats the color of the spatially closest pixel
    of the corresponding image edge.

    """

    def __init__(self, scale=1.0, translate_percent=None, translate_px=None,
                 rotate=0.0, shear=0.0, order=cv2.INTER_LINEAR, cval=0,
                 mode=cv2.BORDER_CONSTANT,
                 name=None, deterministic=False, random_state=None):
        super(AffineCv2, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        available_orders = [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                            cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        available_orders_str = ["nearest", "linear", "cubic", "lanczos4"]

        if order == ia.ALL:
            self.order = iap.Choice(available_orders)
        elif ia.is_single_integer(order):
            assert order in available_orders, (
                "Expected order's integer value to be in %s, got %d." % (
                    str(available_orders), order))
            self.order = iap.Deterministic(order)
        elif ia.is_string(order):
            assert order in available_orders_str, (
                "Expected order to be in %s, got %s." % (
                    str(available_orders_str), order))
            self.order = iap.Deterministic(order)
        elif isinstance(order, list):
            valid_types = all(
                [ia.is_single_integer(val) or ia.is_string(val)
                 for val in order])
            assert valid_types, (
                "Expected order list to only contain integers/strings, got "
                "types %s." % (str([type(val) for val in order]),))
            valid_orders = all(
                [val in available_orders + available_orders_str
                 for val in order])
            assert valid_orders, (
                "Expected all order values to be in %s, got %s." % (
                    available_orders + available_orders_str, str(order),))
            self.order = iap.Choice(order)
        elif isinstance(order, iap.StochasticParameter):
            self.order = order
        else:
            raise Exception(
                "Expected order to be imgaug.ALL, int, string, a list of"
                "int/string or StochasticParameter, got %s." % (type(order),))

        if cval == ia.ALL:
            self.cval = iap.DiscreteUniform(0, 255)
        else:
            self.cval = iap.handle_discrete_param(
                cval, "cval", value_range=(0, 255), tuple_to_uniform=True,
                list_to_choice=True, allow_floats=True)

        available_modes = [cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT,
                           cv2.BORDER_REFLECT_101, cv2.BORDER_WRAP,
                           cv2.BORDER_CONSTANT]
        available_modes_str = ["replicate", "reflect", "reflect_101",
                               "wrap", "constant"]
        if mode == ia.ALL:
            self.mode = iap.Choice(available_modes)
        elif ia.is_single_integer(mode):
            assert mode in available_modes, (
                "Expected mode to be in %s, got %d." % (
                    str(available_modes), mode))
            self.mode = iap.Deterministic(mode)
        elif ia.is_string(mode):
            assert mode in available_modes_str, (
                "Expected mode to be in %s, got %s." % (
                    str(available_modes_str), mode))
            self.mode = iap.Deterministic(mode)
        elif isinstance(mode, list):
            all_valid_types = all([
                ia.is_single_integer(val) or ia.is_string(val) for val in mode])
            assert all_valid_types, (
                "Expected mode list to only contain integers/strings, "
                "got types %s." % (str([type(val) for val in mode]),))
            all_valid_modes = all([
                val in available_modes + available_modes_str for val in mode])
            assert all_valid_modes, (
                "Expected all mode values to be in %s, got %s." % (
                    str(available_modes + available_modes_str), str(mode)))
            self.mode = iap.Choice(mode)
        elif isinstance(mode, iap.StochasticParameter):
            self.mode = mode
        else:
            raise Exception(
                "Expected mode to be imgaug.ALL, an int, a string, a list of "
                "int/strings or StochasticParameter, got %s." % (type(mode),))

        # scale
        if isinstance(scale, dict):
            assert "x" in scale or "y" in scale, (
                "Expected scale dictionary to contain at "
                "least key \"x\" or key \"y\". Found neither of them.")
            x = scale.get("x", 1.0)
            y = scale.get("y", 1.0)
            self.scale = (
                iap.handle_continuous_param(
                    x, "scale['x']", value_range=(0+1e-4, None),
                    tuple_to_uniform=True, list_to_choice=True),
                iap.handle_continuous_param(
                    y, "scale['y']", value_range=(0+1e-4, None),
                    tuple_to_uniform=True, list_to_choice=True)
            )
        else:
            self.scale = iap.handle_continuous_param(
                scale, "scale", value_range=(0+1e-4, None),
                tuple_to_uniform=True, list_to_choice=True)

        # translate
        if translate_percent is None and translate_px is None:
            translate_px = 0

        assert translate_percent is None or translate_px is None, (
            "Expected either translate_percent or translate_px to be "
            "provided, but neither of them was.")

        if translate_percent is not None:
            # translate by percent
            if isinstance(translate_percent, dict):
                assert "x" in translate_percent or "y" in translate_percent, (
                    "Expected translate_percent dictionary to contain at "
                    "least key \"x\" or key \"y\". Found neither of them.")
                x = translate_percent.get("x", 0)
                y = translate_percent.get("y", 0)
                self.translate = (
                    iap.handle_continuous_param(
                        x, "translate_percent['x']", value_range=None,
                        tuple_to_uniform=True, list_to_choice=True),
                    iap.handle_continuous_param(
                        y, "translate_percent['y']", value_range=None,
                        tuple_to_uniform=True, list_to_choice=True)
                )
            else:
                self.translate = iap.handle_continuous_param(
                    translate_percent, "translate_percent", value_range=None,
                    tuple_to_uniform=True, list_to_choice=True)
        else:
            # translate by pixels
            if isinstance(translate_px, dict):
                assert "x" in translate_px or "y" in translate_px, (
                    "Expected translate_px dictionary to contain at "
                    "least key \"x\" or key \"y\". Found neither of them.")
                x = translate_px.get("x", 0)
                y = translate_px.get("y", 0)
                self.translate = (
                    iap.handle_discrete_param(
                        x, "translate_px['x']", value_range=None,
                        tuple_to_uniform=True, list_to_choice=True,
                        allow_floats=False),
                    iap.handle_discrete_param(
                        y, "translate_px['y']", value_range=None,
                        tuple_to_uniform=True, list_to_choice=True,
                        allow_floats=False)
                )
            else:
                self.translate = iap.handle_discrete_param(
                    translate_px, "translate_px", value_range=None,
                    tuple_to_uniform=True, list_to_choice=True,
                    allow_floats=False)

        self.rotate = iap.handle_continuous_param(
            rotate, "rotate", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        self.shear = iap.handle_continuous_param(
            shear, "shear", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        scale_samples, translate_samples, rotate_samples, shear_samples, \
            cval_samples,  mode_samples, order_samples = self._draw_samples(
                nb_images, random_state)
        result = self._augment_images_by_samples(
            images, scale_samples, translate_samples,  rotate_samples,
            shear_samples, cval_samples, mode_samples, order_samples)
        return result

    @classmethod
    def _augment_images_by_samples(cls, images, scale_samples,
                                   translate_samples, rotate_samples,
                                   shear_samples, cval_samples, mode_samples,
                                   order_samples):
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
            translate_x = translate_samples[0][i]
            translate_y = translate_samples[1][i]
            if ia.is_single_float(translate_y):
                translate_y_px = int(
                    np.round(translate_y * images[i].shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(
                    np.round(translate_x * images[i].shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            cval = cval_samples[i]
            mode = mode_samples[i]
            order = order_samples[i]

            mode = (mode
                    if ia.is_single_integer(mode)
                    else mode_str_to_int[mode])
            order = (order
                     if ia.is_single_integer(order)
                     else order_str_to_int[order])

            any_change = (
                scale_x != 1.0 or scale_y != 1.0
                or translate_x_px != 0 or translate_y_px != 0
                or rotate != 0 or shear != 0
            )

            if any_change:
                matrix_to_topleft = tf.SimilarityTransform(
                    translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x_px, translate_y_px),
                    rotation=math.radians(rotate),
                    shear=math.radians(shear)
                )
                matrix_to_center = tf.SimilarityTransform(
                    translation=[shift_x, shift_y])
                matrix = (matrix_to_topleft
                          + matrix_transforms
                          + matrix_to_center)

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
            cval_samples, mode_samples, order_samples = self._draw_samples(
                nb_images, random_state)
        cval_samples = np.zeros((cval_samples.shape[0], 1), dtype=np.float32)
        mode_samples = ["constant"] * len(mode_samples)
        arrs = [heatmap_i.arr_0to1 for heatmap_i in heatmaps]
        arrs_aug = self._augment_images_by_samples(
            arrs, scale_samples, translate_samples, rotate_samples,
            shear_samples, cval_samples, mode_samples, order_samples)
        for heatmap_i, arr_aug in zip(heatmaps, arrs_aug):
            heatmap_i.arr_0to1 = arr_aug
        return heatmaps

    def _augment_segmentation_maps(self, segmaps, random_state, parents, hooks):
        nb_images = len(segmaps)
        scale_samples, translate_samples, rotate_samples, shear_samples, \
            cval_samples, mode_samples, order_samples = self._draw_samples(
                nb_images, random_state)
        cval_samples = np.zeros((cval_samples.shape[0], 1), dtype=np.float32)
        mode_samples = ["constant"] * len(mode_samples)
        order_samples = [0] * len(order_samples)
        arrs = [segmaps_i.arr for segmaps_i in segmaps]
        arrs_aug = self._augment_images_by_samples(
            arrs, scale_samples, translate_samples, rotate_samples,
            shear_samples, cval_samples, mode_samples, order_samples)
        for segmaps_i, arr_aug in zip(segmaps, arrs_aug):
            segmaps_i.arr = arr_aug
        return segmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        result = []
        nb_images = len(keypoints_on_images)
        scale_samples, translate_samples, rotate_samples, shear_samples, \
            _cval_samples, _mode_samples, _order_samples = self._draw_samples(
                nb_images, random_state)

        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if not keypoints_on_image.keypoints:
                # AffineCv2 does not change the image shape, hence we can skip
                # all steps below if there are no keypoints
                result.append(keypoints_on_image)
                continue
            height, width = keypoints_on_image.height, keypoints_on_image.width
            shift_x = width / 2.0 - 0.5
            shift_y = height / 2.0 - 0.5
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x = translate_samples[0][i]
            translate_y = translate_samples[1][i]
            if ia.is_single_float(translate_y):
                translate_y_px = int(
                    np.round(translate_y * keypoints_on_image.shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(
                    np.round(translate_x * keypoints_on_image.shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]

            any_change = (
                scale_x != 1.0 or scale_y != 1.0
                or translate_x_px != 0 or translate_y_px != 0
                or rotate != 0 or shear != 0
            )

            if any_change:
                matrix_to_topleft = tf.SimilarityTransform(
                    translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x_px, translate_y_px),
                    rotation=math.radians(rotate),
                    shear=math.radians(shear)
                )
                matrix_to_center = tf.SimilarityTransform(
                    translation=[shift_x, shift_y])
                matrix = (matrix_to_topleft
                          + matrix_transforms
                          + matrix_to_center)

                coords = keypoints_on_image.to_xy_array()
                coords_aug = tf.matrix_transform(coords, matrix.params)
                kps_new = [kp.deepcopy(x=coords[0], y=coords[1])
                           for kp, coords
                           in zip(keypoints_on_image.keypoints, coords_aug)]
                result.append(keypoints_on_image.deepcopy(
                    keypoints=kps_new,
                    shape=keypoints_on_image.shape
                ))
            else:
                result.append(keypoints_on_image)
        return result

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        return self._augment_polygons_as_keypoints(
            polygons_on_images, random_state, parents, hooks)

    def _augment_line_strings(self, line_strings_on_images, random_state,
                              parents, hooks):
        return self._augment_line_strings_as_keypoints(
            line_strings_on_images, random_state, parents, hooks)

    def _augment_bounding_boxes(self, bounding_boxes_on_images, random_state,
                                parents, hooks):
        return self._augment_bounding_boxes_as_keypoints(
            bounding_boxes_on_images, random_state, parents, hooks)

    def get_parameters(self):
        return [self.scale, self.translate, self.rotate, self.shear,
                self.order, self.cval, self.mode]

    def _draw_samples(self, nb_samples, random_state):
        rngs = random_state.duplicate(11)

        if isinstance(self.scale, tuple):
            scale_samples = (
                self.scale[0].draw_samples((nb_samples,),
                                           random_state=rngs[0]),
                self.scale[1].draw_samples((nb_samples,),
                                           random_state=rngs[1]),
            )
        else:
            scale_samples = self.scale.draw_samples((nb_samples,),
                                                    random_state=rngs[2])
            scale_samples = (scale_samples, scale_samples)

        if isinstance(self.translate, tuple):
            translate_samples = (
                self.translate[0].draw_samples((nb_samples,),
                                               random_state=rngs[3]),
                self.translate[1].draw_samples((nb_samples,),
                                               random_state=rngs[4]),
            )
        else:
            translate_samples = self.translate.draw_samples(
                (nb_samples,), random_state=rngs[5])
            translate_samples = (translate_samples, translate_samples)

        valid_dts = ["int32", "int64", "float32", "float64"]
        for i in sm.xrange(2):
            assert translate_samples[i].dtype.name in valid_dts, (
                "Expected translate_samples to have any dtype of %s. "
                "Got %s." % (str(valid_dts), translate_samples[i].dtype.name,))

        rotate_samples = self.rotate.draw_samples((nb_samples,),
                                                  random_state=rngs[6])
        shear_samples = self.shear.draw_samples((nb_samples,),
                                                random_state=rngs[7])

        cval_samples = self.cval.draw_samples((nb_samples, 3),
                                              random_state=rngs[8])
        mode_samples = self.mode.draw_samples((nb_samples,),
                                              random_state=rngs[9])
        order_samples = self.order.draw_samples((nb_samples,),
                                                random_state=rngs[10])

        return (
            scale_samples, translate_samples, rotate_samples, shear_samples,
            cval_samples, mode_samples, order_samples
        )


class _PiecewiseAffineSamplingResult(object):
    def __init__(self, nb_rows, nb_cols, jitter, order, cval, mode):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.order = order
        self.jitter = jitter
        self.cval = cval
        self.mode = mode

    def get_clipped_cval(self, idx, dtype):
        min_value, _, max_value = iadt.get_value_range_of_dtype(dtype)
        cval = self.cval[idx]
        cval = max(min(cval, max_value), min_value)
        return cval


class PiecewiseAffine(meta.Augmenter):
    """
    Apply affine transformations that differ between local neighbourhoods.

    This augmenter places a regular grid of points on an image and randomly
    moves the neighbourhood of these point around via affine transformations.
    This leads to local distortions.

    This is mostly a wrapper around scikit-image's ``PiecewiseAffine``.
    See also ``Affine`` for a similar technique.

    .. note::

        This augmenter is very slow. See :ref:`performance`.
        Try to use ``ElasticTransformation`` instead, which is at least 10x
        faster.

    .. note::

        For coordinate-based inputs (keypoints, bounding boxes, polygons,
        ...), this augmenter still has to perform an image-based augmentation,
        which will make it significantly slower for such inputs than other
        augmenters. See :ref:`performance`.

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

        - (1) Only tested with `order` set to ``0``.
        - (2) scikit-image converts internally to ``float64``, which might
              introduce inaccuracies. Tests showed that these inaccuracies
              seemed to not be an issue.
        - (3) Results too inaccurate.
        - (4) Mapped internally to ``float64``.

    Parameters
    ----------
    scale : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        Each point on the regular grid is moved around via a normal
        distribution. This scale factor is equivalent to the normal
        distribution's sigma. Note that the jitter (how far each point is
        moved in which direction) is multiplied by the height/width of the
        image if ``absolute_scale=False`` (default), so this scale can be
        the same for different sized images.
        Recommended values are in the range ``0.01`` to ``0.05`` (weak to
        strong augmentations).

            * If a single ``float``, then that value will always be used as
              the scale.
            * If a tuple ``(a, b)`` of ``float`` s, then a random value will
              be uniformly sampled per image from the interval ``[a, b]``.
            * If a list, then a random value will be picked from that list
              per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    nb_rows : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        Number of rows of points that the regular grid should have.
        Must be at least ``2``. For large images, you might want to pick a
        higher value than ``4``. You might have to then adjust scale to lower
        values.

            * If a single ``int``, then that value will always be used as the
              number of rows.
            * If a tuple ``(a, b)``, then a value from the discrete interval
              ``[a..b]`` will be uniformly sampled per image.
            * If a list, then a random value will be picked from that list
              per image.
            * If a StochasticParameter, then that parameter will be queried to
              draw one value per image.

    nb_cols : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        Number of columns. Analogous to `nb_rows`.

    order : int or list of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.geometric.Affine.__init__`.

    cval : int or float or tuple of float or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.geometric.Affine.__init__`.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.geometric.Affine.__init__`.

    absolute_scale : bool, optional
        Take `scale` as an absolute value rather than a relative value.

    polygon_recoverer : 'auto' or None or imgaug.augmentables.polygons._ConcavePolygonRecoverer, optional
        The class to use to repair invalid polygons.
        If ``"auto"``, a new instance of
        :class`imgaug.augmentables.polygons._ConcavePolygonRecoverer`
        will be created.
        If ``None``, no polygon recoverer will be used.
        If an object, then that object will be used and must provide a
        ``recover_from()`` method, similar to
        :class:`imgaug.augmentables.polygons._ConcavePolygonRecoverer`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))

    Place a regular grid of points on each image and then randomly move each
    point around by ``1`` to ``5`` percent (with respect to the image
    height/width). Pixels between these points will be moved accordingly.

    >>> aug = iaa.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=8, nb_cols=8)

    Same as the previous example, but uses a denser grid of ``8x8`` points
    (default is ``4x4``). This can be useful for large images.

    """

    def __init__(self, scale=0, nb_rows=4, nb_cols=4, order=1, cval=0,
                 mode="constant", absolute_scale=False, polygon_recoverer=None,
                 name=None, deterministic=False, random_state=None):
        super(PiecewiseAffine, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.scale = iap.handle_continuous_param(
            scale, "scale", value_range=(0, None), tuple_to_uniform=True,
            list_to_choice=True)
        self.jitter = iap.Normal(loc=0, scale=self.scale)
        self.nb_rows = iap.handle_discrete_param(
            nb_rows, "nb_rows", value_range=(2, None), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=False)
        self.nb_cols = iap.handle_discrete_param(
            nb_cols, "nb_cols", value_range=(2, None), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=False)

        self.order = _handle_order_arg(order, backend="skimage")
        self.cval = _handle_cval_arg(cval)
        self.mode = _handle_mode_arg(mode)

        self.absolute_scale = absolute_scale
        self.polygon_recoverer = polygon_recoverer
        if polygon_recoverer == "auto":
            self.polygon_recoverer = _ConcavePolygonRecoverer()

        # Special order, mode and cval parameters for heatmaps and
        # segmentation maps. These may either be None or a fixed value.
        # Stochastic parameters are currently *not* supported.
        # If set to None, the same values as for images will be used.
        # That is really not recommended for the cval parameter.
        self._order_heatmaps = 3
        self._order_segmentation_maps = 0
        self._mode_heatmaps = "constant"
        self._mode_segmentation_maps = "constant"
        self._cval_heatmaps = 0
        self._cval_segmentation_maps = 0

    def _augment_batch(self, batch, random_state, parents, hooks):
        samples = self._draw_samples(batch.nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images,
                                                           samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps, "arr_0to1", samples, self._cval_heatmaps,
                self._mode_heatmaps, self._order_heatmaps)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, "arr", samples,
                self._cval_segmentation_maps, self._mode_segmentation_maps,
                self._order_segmentation_maps)

        # TODO add test for recoverer
        if batch.polygons is not None:
            func = functools.partial(
                self._augment_keypoints_by_samples,
                samples=samples)
            batch.polygons = self._apply_to_polygons_as_keypoints(
                batch.polygons, func, recoverer=self.polygon_recoverer)

        for augm_name in ["keypoints", "bounding_boxes", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(
                    self._augment_keypoints_by_samples,
                    samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    def _augment_images_by_samples(self, images, samples):
        iadt.gate_dtypes(
            images,
            allowed=["bool",
                     "uint8", "uint16", "uint32",
                     "int8", "int16", "int32",
                     "float16", "float32", "float64"],
            disallowed=["uint64", "uint128", "uint256",
                        "int64", "int128", "int256",
                        "float96", "float128", "float256"],
            augmenter=self)

        result = images

        for i, image in enumerate(images):
            transformer = self._get_transformer(
                image.shape, image.shape, samples.nb_rows[i],
                samples.nb_cols[i], samples.jitter[i])

            if transformer is not None:
                input_dtype = image.dtype
                if image.dtype.kind == "b":
                    image = image.astype(np.float64)

                image_warped = tf.warp(
                    image,
                    transformer,
                    order=samples.order[i],
                    mode=samples.mode[i],
                    cval=samples.get_clipped_cval(i, image.dtype),
                    preserve_range=True,
                    output_shape=images[i].shape
                )

                if input_dtype.kind == "b":
                    image_warped = image_warped > 0.5
                else:
                    # warp seems to change everything to float64, including
                    # uint8, making this necessary
                    image_warped = iadt.restore_dtypes_(
                        image_warped, input_dtype)

                result[i] = image_warped

        return result

    def _augment_maps_by_samples(self, augmentables, arr_attr_name, samples,
                                 cval, mode, order):
        result = augmentables

        for i, augmentable in enumerate(augmentables):
            arr = getattr(augmentable, arr_attr_name)

            transformer = self._get_transformer(
                arr.shape, augmentable.shape, samples.nb_rows[i],
                samples.nb_cols[i], samples.jitter[i])

            if transformer is not None:
                arr_warped = tf.warp(
                    arr,
                    transformer,
                    order=order if order is not None else samples.order[i],
                    mode=mode if mode is not None else samples.mode[i],
                    cval=cval if cval is not None else samples.cval[i],
                    preserve_range=True,
                    output_shape=arr.shape
                )

                # skimage converts to float64
                arr_warped = arr_warped.astype(arr.dtype)

                # TODO not entirely clear whether this breaks the value
                #      range -- Affine does
                # TODO add test for this
                # order=3 matches cubic interpolation and can cause values
                # to go outside of the range [0.0, 1.0] not clear whether
                # 4+ also do that
                # We don't modify segmaps here, because they don't have a
                # clear value range of [0, 1]
                if order >= 3 and isinstance(augmentable, ia.HeatmapsOnImage):
                    arr_warped = np.clip(arr_warped, 0.0, 1.0, out=arr_warped)

                setattr(augmentable, arr_attr_name, arr_warped)

        return result

    def _augment_keypoints_by_samples(self, kpsois, samples):
        result = []

        for i, kpsoi in enumerate(kpsois):
            h, w = kpsoi.shape[0:2]
            transformer = self._get_transformer(
                kpsoi.shape, kpsoi.shape, samples.nb_rows[i],
                samples.nb_cols[i], samples.jitter[i])

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

                # TODO this could be done a little bit more efficient by
                #      removing first all KPs that are outside of the image
                #      plane so that no corresponding distance map has to
                #      be augmented
                # Image based augmentation routine. Draws the keypoints on
                # the image plane using distance maps (more accurate than
                # just marking the points),  then augments these images, then
                # searches for the new (visual) location of the keypoints.
                # Much slower than directly augmenting the coordinates, but
                # here the only method that reliably works.
                dist_maps = kpsoi.to_distance_maps(inverted=True)
                dist_maps_warped = tf.warp(
                    dist_maps,
                    transformer,
                    order=1,
                    preserve_range=True,
                    output_shape=(kpsoi.shape[0], kpsoi.shape[1],
                                  len(kpsoi.keypoints))
                )

                kps_aug = ia.KeypointsOnImage.from_distance_maps(
                    dist_maps_warped,
                    inverted=True,
                    threshold=0.01,
                    if_not_found_coords={"x": -1, "y": -1},
                    nb_channels=(
                        None if len(kpsoi.shape) < 3 else kpsoi.shape[2])
                )

                for kp, kp_aug in zip(kpsoi.keypoints, kps_aug.keypoints):
                    # Keypoints that were outside of the image plane before the
                    # augmentation were replaced with (-1, -1) by default (as
                    # they can't be drawn on the keypoint images).
                    within_image = (0 <= kp.x < w and 0 <= kp.y < h)
                    if within_image:
                        kp.x = kp_aug.x
                        kp.y = kp_aug.y

                result.append(kpsoi)

        return result

    def _draw_samples(self, nb_images, random_state):
        rss = random_state.duplicate(6)

        nb_rows_samples = self.nb_rows.draw_samples((nb_images,),
                                                    random_state=rss[-6])
        nb_cols_samples = self.nb_cols.draw_samples((nb_images,),
                                                    random_state=rss[-5])
        order_samples = self.order.draw_samples((nb_images,),
                                                random_state=rss[-4])
        cval_samples = self.cval.draw_samples((nb_images,),
                                              random_state=rss[-3])
        mode_samples = self.mode.draw_samples((nb_images,),
                                              random_state=rss[-2])

        nb_rows_samples = np.clip(nb_rows_samples, 2, None)
        nb_cols_samples = np.clip(nb_cols_samples, 2, None)
        nb_cells = nb_rows_samples * nb_cols_samples
        jitter = self.jitter.draw_samples((int(np.sum(nb_cells)), 2),
                                          random_state=rss[-1])

        jitter_by_image = []
        counter = 0
        for i, nb_cells_i in enumerate(nb_cells):
            jitter_img = jitter[counter:counter+nb_cells_i, :]
            jitter_by_image.append(jitter_img)
            counter += nb_cells_i

        return _PiecewiseAffineSamplingResult(
            nb_rows=nb_rows_samples, nb_cols=nb_cols_samples,
            jitter=jitter_by_image,
            order=order_samples, cval=cval_samples, mode=mode_samples)

    def _get_transformer(self, augmentable_shape, image_shape, nb_rows,
                         nb_cols, jitter_img):
        # get coords on y and x axis of points to move around
        # these coordinates are supposed to be at the centers of each cell
        # (otherwise the first coordinate would be at (0, 0) and could hardly
        # be moved around before leaving the image),
        # so we use here (half cell height/width to H/W minus half
        # height/width) instead of (0, H/W)
        y = np.linspace(0, augmentable_shape[0], nb_rows)
        x = np.linspace(0, augmentable_shape[1], nb_cols)

        # (H, W) and (H, W) for H=rows, W=cols
        xx_src, yy_src = np.meshgrid(x, y)

        # (1, HW, 2) => (HW, 2) for H=rows, W=cols
        points_src = np.dstack([yy_src.flat, xx_src.flat])[0]

        any_nonzero = np.any(jitter_img > 0)
        if not any_nonzero:
            return None
        else:
            # Without this, jitter gets changed between different augmentables.
            # TODO if left out, only one test failed -- should be more
            jitter_img = np.copy(jitter_img)

            if self.absolute_scale:
                if image_shape[0] > 0:
                    jitter_img[:, 0] = jitter_img[:, 0] / image_shape[0]
                else:
                    jitter_img[:, 0] = 0.0

                if image_shape[1] > 0:
                    jitter_img[:, 1] = jitter_img[:, 1] / image_shape[1]
                else:
                    jitter_img[:, 1] = 0.0

            jitter_img[:, 0] = jitter_img[:, 0] * augmentable_shape[0]
            jitter_img[:, 1] = jitter_img[:, 1] * augmentable_shape[1]

            points_dest = np.copy(points_src)
            points_dest[:, 0] = points_dest[:, 0] + jitter_img[:, 0]
            points_dest[:, 1] = points_dest[:, 1] + jitter_img[:, 1]

            # Restrict all destination points to be inside the image plane.
            # This is necessary, as otherwise keypoints could be augmented
            # outside of the image plane and these would be replaced by
            # (-1, -1), which would not conform with the behaviour of the
            # other augmenters.
            points_dest[:, 0] = np.clip(points_dest[:, 0],
                                        0, augmentable_shape[0]-1)
            points_dest[:, 1] = np.clip(points_dest[:, 1],
                                        0, augmentable_shape[1]-1)

            # tf.warp() results in qhull error if the points are identical,
            # which is mainly the case if any axis is 0
            has_low_axis = any([axis <= 1 for axis in augmentable_shape[0:2]])
            has_zero_channels = (
                (
                    augmentable_shape is not None
                    and len(augmentable_shape) == 3
                    and augmentable_shape[-1] == 0
                )
                or
                (
                    image_shape is not None
                    and len(image_shape) == 3
                    and image_shape[-1] == 0
                )
            )

            if has_low_axis or has_zero_channels:
                return None
            else:
                matrix = tf.PiecewiseAffineTransform()
                matrix.estimate(points_src[:, ::-1], points_dest[:, ::-1])
                return matrix

    def get_parameters(self):
        return [
            self.scale, self.nb_rows, self.nb_cols, self.order, self.cval,
            self.mode, self.absolute_scale]


class _PerspectiveTransformSamplingResult(object):
    def __init__(self, matrices, max_heights, max_widths, cvals, modes):
        self.matrices = matrices
        self.max_heights = max_heights
        self.max_widths = max_widths
        self.cvals = cvals
        self.modes = modes


# TODO add arg for image interpolation
class PerspectiveTransform(meta.Augmenter):
    """
    Apply random four point perspective transformations to images.

    Each of the four points is placed on the image using a random distance from
    its respective corner. The distance is sampled from a normal distribution.
    As a result, most transformations don't change the image very much, while
    some "focus" on polygons far inside the image.

    The results of this augmenter have some similarity with ``Crop``.

    Code partially from
    http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

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
            - (2) leads to opencv error: cv2.error: ``OpenCV(3.4.4)
                  (...)imgwarp.cpp:1805: error: (-215:Assertion failed)
                  ifunc != 0 in function 'remap'``.
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
        Standard deviation of the normal distributions. These are used to
        sample the random distances of the subimage's corners from the full
        image's corners. The sampled values reflect percentage values (with
        respect to image height/width). Recommended values are in the range
        ``0.0`` to ``0.1``.

            * If a single number, then that value will always be used as the
              scale.
            * If a tuple ``(a, b)`` of numbers, then a random value will be
              uniformly sampled per image from the interval ``(a, b)``.
            * If a list of values, a random value will be picked from the
              list per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    keep_size : bool, optional
        Whether to resize image's back to their original size after applying
        the perspective transform. If set to ``False``, the resulting images
        may end up having different shapes and will always be a list, never
        an array.

    cval : number or tuple of number or list of number or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant value used to fill up pixels in the result image that
        didn't exist in the input image (e.g. when translating to the left,
        some new pixels are created at the right). Such a fill-up with a
        constant value only happens, when `mode` is ``constant``.
        The expected value range is ``[0, 255]`` for ``uint8`` images.
        It may be a float value.

            * If this is a single int or float, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple ``(a, b)``, then a random value is uniformly sampled
              per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If ``imgaug.ALL``, then equivalent to tuple ``(0, 255)``.
            * If a ``StochasticParameter``, a new value will be sampled from
              the parameter per image.

    mode : int or str or list of str or list of int or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Parameter that defines the handling of newly created pixels.
        Same meaning as in OpenCV's border mode. Let ``abcdefgh`` be an image's
        content and ``|`` be an image boundary, then:

            * ``cv2.BORDER_REPLICATE``: ``aaaaaa|abcdefgh|hhhhhhh``
            * ``cv2.BORDER_CONSTANT``: ``iiiiii|abcdefgh|iiiiiii``, where
              ``i`` is the defined cval.
            * ``replicate``: Same as ``cv2.BORDER_REPLICATE``.
            * ``constant``: Same as ``cv2.BORDER_CONSTANT``.

        The datatype of the parameter may be:

            * If a single ``int``, then it must be one of ``cv2.BORDER_*``.
            * If a single string, then it must be one of: ``replicate``,
              ``reflect``, ``reflect_101``, ``wrap``, ``constant``.
            * If a list of ints/strings, then per image a random mode will be
              picked from that list.
            * If ``imgaug.ALL``, then a random mode from all possible modes
              will be picked per image.
            * If ``StochasticParameter``, then the mode will be sampled from
              that parameter per image, i.e. it must return only the above
              mentioned strings.

    fit_output : bool, optional
        If ``True``, the image plane size and position will be adjusted
        to still capture the whole image after perspective transformation.
        (Followed by image resizing if `keep_size` is set to ``True``.)
        Otherwise, parts of the transformed image may be outside of the image
        plane.
        This setting should not be set to ``True`` when using large `scale`
        values as it could lead to very large images.

    polygon_recoverer : 'auto' or None or imgaug.augmentables.polygons._ConcavePolygonRecoverer, optional
        The class to use to repair invalid polygons.
        If ``"auto"``, a new instance of
        :class`imgaug.augmentables.polygons._ConcavePolygonRecoverer`
        will be created.
        If ``None``, no polygon recoverer will be used.
        If an object, then that object will be used and must provide a
        ``recover_from()`` method, similar to
        :class:`imgaug.augmentables.polygons._ConcavePolygonRecoverer`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.PerspectiveTransform(scale=(0.01, 0.15))

    Apply perspective transformations using a random scale between ``0.01``
    and ``0.15`` per image, where the scale is roughly a measure of how far
    the perspective transformation's corner points may be distanced from the
    image's corner points. Higher scale values lead to stronger "zoom-in"
    effects (and thereby stronger distortions).

    >>> aug = iaa.PerspectiveTransform(scale=(0.01, 0.15), keep_size=False)

    Same as in the previous example, but images are not resized back to
    the input image size after augmentation. This will lead to smaller
    output images.

    """

    _BORDER_MODE_STR_TO_INT = {
        "replicate": cv2.BORDER_REPLICATE,
        "constant": cv2.BORDER_CONSTANT
    }

    def __init__(self, scale=0, cval=0, mode="constant", keep_size=True,
                 fit_output=False, polygon_recoverer="auto",
                 name=None, deterministic=False, random_state=None):
        super(PerspectiveTransform, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.scale = iap.handle_continuous_param(
            scale, "scale", value_range=(0, None), tuple_to_uniform=True,
            list_to_choice=True)
        self.jitter = iap.Normal(loc=0, scale=self.scale)

        # setting these to 1x1 caused problems for large scales and polygon
        # augmentation
        # TODO there is now a recoverer for polygons - are these minima still
        #      needed/sensible?
        self.min_width = 2
        self.min_height = 2

        self.cval = _handle_cval_arg(cval)
        self.mode = self._handle_mode_arg(mode)
        self.keep_size = keep_size
        self.fit_output = fit_output

        self.polygon_recoverer = polygon_recoverer
        if polygon_recoverer == "auto":
            self.polygon_recoverer = _ConcavePolygonRecoverer()

        # Special order, mode and cval parameters for heatmaps and
        # segmentation maps. These may either be None or a fixed value.
        # Stochastic parameters are currently *not* supported.
        # If set to None, the same values as for images will be used.
        # That is really not recommended for the cval parameter.
        self._order_heatmaps = cv2.INTER_LINEAR
        self._order_segmentation_maps = cv2.INTER_NEAREST
        self._mode_heatmaps = cv2.BORDER_CONSTANT
        self._mode_segmentation_maps = cv2.BORDER_CONSTANT
        self._cval_heatmaps = 0
        self._cval_segmentation_maps = 0

    # TODO unify this somehow with the global _handle_mode_arg() that is
    #      currently used for Affine and PiecewiseAffine
    @classmethod
    def _handle_mode_arg(cls, mode):
        available_modes = [cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT]
        available_modes_str = ["replicate", "constant"]
        if mode == ia.ALL:
            return iap.Choice(available_modes)
        elif ia.is_single_integer(mode):
            assert mode in available_modes, (
                "Expected mode to be in %s, got %d." % (
                    str(available_modes), mode))
            return iap.Deterministic(mode)
        elif ia.is_string(mode):
            assert mode in available_modes_str, (
                "Expected mode to be in %s, got %s." % (
                    str(available_modes_str), mode))
            return iap.Deterministic(mode)
        elif isinstance(mode, list):
            valid_types = all([ia.is_single_integer(val) or ia.is_string(val)
                               for val in mode])
            assert valid_types, (
                "Expected mode list to only contain integers/strings, got "
                "types %s." % (
                    ", ".join([str(type(val)) for val in mode]),))
            valid_modes = all([val in available_modes + available_modes_str
                               for val in mode])
            assert valid_modes, (
                "Expected all mode values to be in %s, got %s." % (
                    str(available_modes + available_modes_str), str(mode)))
            return iap.Choice(mode)
        elif isinstance(mode, iap.StochasticParameter):
            return mode
        else:
            raise Exception(
                "Expected mode to be imgaug.ALL, an int, a string, a list "
                "of int/strings or StochasticParameter, got %s." % (
                    type(mode),))

    def _augment_batch(self, batch, random_state, parents, hooks):
        samples_images = self._draw_samples(batch.get_rowwise_shapes(),
                                            random_state.copy())

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images,
                                                           samples_images)

        if batch.heatmaps is not None:
            samples = self._draw_samples(
                [augmentable.arr_0to1.shape
                 for augmentable in batch.heatmaps],
                random_state.copy())

            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps, "arr_0to1", samples, samples_images,
                self._cval_heatmaps, self._mode_heatmaps, self._order_heatmaps)

        if batch.segmentation_maps is not None:
            samples = self._draw_samples(
                [augmentable.arr.shape
                 for augmentable in batch.segmentation_maps],
                random_state.copy())

            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, "arr", samples, samples_images,
                self._cval_segmentation_maps, self._mode_segmentation_maps,
                self._order_segmentation_maps)

        # large scale values cause invalid polygons (unclear why that happens),
        # hence the recoverer
        # TODO add test for recoverer
        if batch.polygons is not None:
            func = functools.partial(
                self._augment_keypoints_by_samples,
                samples_images=samples_images)
            batch.polygons = self._apply_to_polygons_as_keypoints(
                batch.polygons, func, recoverer=self.polygon_recoverer)

        for augm_name in ["keypoints", "bounding_boxes", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(
                    self._augment_keypoints_by_samples,
                    samples_images=samples_images)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    def _augment_images_by_samples(self, images, samples):
        iadt.gate_dtypes(
            images,
            allowed=["bool",
                     "uint8", "uint16",
                     "int8", "int16",
                     "float16", "float32", "float64"],
            disallowed=["uint32", "uint64", "uint128", "uint256",
                        "int32", "int64", "int128", "int256",
                        "float96", "float128", "float256"],
            augmenter=self)

        result = images
        if not self.keep_size:
            result = list(result)

        gen = enumerate(zip(images, samples.matrices, samples.max_heights,
                            samples.max_widths, samples.cvals, samples.modes))

        for i, (image, matrix, max_height, max_width, cval, mode) in gen:
            input_dtype = image.dtype
            if input_dtype.name in ["int8"]:
                image = image.astype(np.int16)
            elif input_dtype.name in ["bool", "float16"]:
                image = image.astype(np.float32)

            # cv2.warpPerspective only supports <=4 channels and errors
            # on axes with size zero
            nb_channels = image.shape[2]
            has_zero_sized_axis = (image.size == 0)
            if has_zero_sized_axis:
                warped = image
            elif nb_channels <= 4:
                warped = cv2.warpPerspective(
                    image,
                    matrix,
                    (max_width, max_height),
                    borderValue=cval,
                    borderMode=mode)
                if warped.ndim == 2 and images[i].ndim == 3:
                    warped = np.expand_dims(warped, 2)
            else:
                # warp each channel on its own
                # note that cv2 removes the channel axis in case of (H,W,1)
                # inputs
                warped = [
                    cv2.warpPerspective(
                        image[..., c],
                        matrix,
                        (max_width, max_height),
                        borderValue=cval[min(c, len(cval)-1)],
                        borderMode=mode,
                        flags=cv2.INTER_LINEAR
                    )
                    for c in sm.xrange(nb_channels)
                ]
                warped = np.stack(warped, axis=-1)

            if self.keep_size and not has_zero_sized_axis:
                h, w = image.shape[0:2]
                warped = ia.imresize_single_image(warped, (h, w))

            if input_dtype.name == "bool":
                warped = warped > 0.5
            elif warped.dtype.name != input_dtype.name:
                warped = iadt.restore_dtypes_(warped, input_dtype)

            result[i] = warped

        return result

    def _augment_maps_by_samples(self, augmentables, arr_attr_name,
                                 samples, samples_images, cval, mode, flags):
        result = augmentables

        # estimate max_heights/max_widths for the underlying images
        # this is only necessary if keep_size is False as then the underlying
        # image sizes change and we need to update them here
        # TODO this was re-used from before _augment_batch() -- reoptimize
        if self.keep_size:
            max_heights_imgs = samples.max_heights
            max_widths_imgs = samples.max_widths
        else:
            max_heights_imgs = samples_images.max_heights
            max_widths_imgs = samples_images.max_widths

        gen = enumerate(zip(augmentables, samples.matrices, samples.max_heights,
                            samples.max_widths))

        for i, (augmentable_i, matrix, max_height, max_width) in gen:
            arr = getattr(augmentable_i, arr_attr_name)

            mode_i = mode
            if mode is None:
                mode_i = samples.modes[i]

            cval_i = cval
            if cval is None:
                cval_i = samples.cvals[i]

            nb_channels = arr.shape[2]
            image_has_zero_sized_axis = (0 in augmentable_i.shape)
            map_has_zero_sized_axis = (arr.size == 0)

            if not image_has_zero_sized_axis:
                if not map_has_zero_sized_axis:
                    warped = [
                        cv2.warpPerspective(
                            arr[..., c],
                            matrix,
                            (max_width, max_height),
                            borderValue=cval_i,
                            borderMode=mode_i,
                            flags=flags
                        )
                        for c in sm.xrange(nb_channels)
                    ]
                    warped = np.stack(warped, axis=-1)

                    setattr(augmentable_i, arr_attr_name, warped)

                if self.keep_size:
                    h, w = arr.shape[0:2]
                    augmentable_i = augmentable_i.resize((h, w))
                else:
                    new_shape = (
                        max_heights_imgs[i], max_widths_imgs[i]
                    ) + augmentable_i.shape[2:]
                    augmentable_i.shape = new_shape

                result[i] = augmentable_i

        return result

    def _augment_keypoints_by_samples(self, kpsois, samples_images):
        result = kpsois

        gen = enumerate(zip(kpsois,
                            samples_images.matrices,
                            samples_images.max_heights,
                            samples_images.max_widths))

        for i, (kpsoi, matrix, max_height, max_width) in gen:
            image_has_zero_sized_axis = (0 in kpsoi.shape)

            if not image_has_zero_sized_axis:
                shape_orig = kpsoi.shape
                shape_new = (max_height, max_width) + kpsoi.shape[2:]
                kpsoi.shape = shape_new
                if not kpsoi.empty:
                    kps_arr = kpsoi.to_xy_array()
                    warped = cv2.perspectiveTransform(
                        np.array([kps_arr], dtype=np.float32), matrix)
                    warped = warped[0]
                    for kp, coords in zip(kpsoi.keypoints, warped):
                        kp.x = coords[0]
                        kp.y = coords[1]
                if self.keep_size:
                    kpsoi = kpsoi.on(shape_orig)
                result[i] = kpsoi

        return result

    def _draw_samples(self, shapes, random_state):
        matrices = []
        max_heights = []
        max_widths = []
        nb_images = len(shapes)
        rngs = random_state.duplicate(3)

        cval_samples = self.cval.draw_samples((nb_images, 3),
                                              random_state=rngs[0])
        mode_samples = self.mode.draw_samples((nb_images,),
                                              random_state=rngs[1])
        jitter = self.jitter.draw_samples((nb_images, 4, 2),
                                          random_state=rngs[2])

        # cv2 perspectiveTransform doesn't accept numpy arrays as cval
        cval_samples_cv2 = cval_samples.tolist()

        # if border modes are represented by strings, convert them to cv2
        # border mode integers
        if mode_samples.dtype.kind not in ["i", "u"]:
            for mode, mapped_mode in self._BORDER_MODE_STR_TO_INT.items():
                mode_samples[mode_samples == mode] = mapped_mode

        # modify jitter to the four corner point coordinates
        # some x/y values have to be modified from `jitter` to `1-jtter`
        # for that
        # TODO remove the abs() here. it currently only allows to "zoom-in",
        #      not to "zoom-out"
        points = np.mod(np.abs(jitter), 1)

        # top left -- no changes needed, just use jitter
        # top right
        points[:, 1, 0] = 1.0 - points[:, 1, 0]  # w = 1.0 - jitter
        # bottom right
        points[:, 2, 0] = 1.0 - points[:, 2, 0]  # w = 1.0 - jitter
        points[:, 2, 1] = 1.0 - points[:, 2, 1]  # h = 1.0 - jitter
        # bottom left
        points[:, 3, 1] = 1.0 - points[:, 3, 1]  # h = 1.0 - jitter

        for shape, points_i in zip(shapes, points):
            h, w = shape[0:2]

            points_i[:, 0] *= w
            points_i[:, 1] *= h

            # Obtain a consistent order of the points and unpack them
            # individually.
            # Warning: don't just do (tl, tr, br, bl) = _order_points(...)
            # here, because the reordered points_i is used further below.
            points_i = self._order_points(points_i)
            (tl, tr, br, bl) = points_i

            # compute the width of the new image, which will be the
            # maximum distance between bottom-right and bottom-left
            # x-coordiates or the top-right and top-left x-coordinates
            min_width = None
            max_width = None
            while min_width is None or min_width < self.min_width:
                width_top = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
                width_bottom = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
                max_width = int(max(width_top, width_bottom))
                min_width = int(min(width_top, width_bottom))
                if min_width < self.min_width:
                    step_size = (self.min_width - min_width)/2
                    tl[0] -= step_size
                    tr[0] += step_size
                    bl[0] -= step_size
                    br[0] += step_size

            # compute the height of the new image, which will be the
            # maximum distance between the top-right and bottom-right
            # y-coordinates or the top-left and bottom-left y-coordinates
            min_height = None
            max_height = None
            while min_height is None or min_height < self.min_height:
                height_right = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
                height_left = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
                max_height = int(max(height_right, height_left))
                min_height = int(min(height_right, height_left))
                if min_height < self.min_height:
                    step_size = (self.min_height - min_height)/2
                    tl[1] -= step_size
                    tr[1] -= step_size
                    bl[1] += step_size
                    br[1] += step_size

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
            ], dtype=np.float32)

            # compute the perspective transform matrix and then apply it
            m = cv2.getPerspectiveTransform(points_i, dst)

            if self.fit_output:
                m, max_width, max_height = self._expand_transform(m, (h, w))

            matrices.append(m)
            max_heights.append(max_height)
            max_widths.append(max_width)

        mode_samples = mode_samples.astype(np.int32)
        return _PerspectiveTransformSamplingResult(
            matrices, max_heights, max_widths, cval_samples_cv2,
            mode_samples)

    @classmethod
    def _order_points(cls, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        pts_ordered = np.zeros((4, 2), dtype=np.float32)

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

    @classmethod
    def _expand_transform(cls, M, shape):
        height, width = shape
        rect = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype=np.float32)
        dst = cv2.perspectiveTransform(np.array([rect]), M)[0]

        # get min x, y over transformed 4 points
        # then modify target points by subtracting these minima
        # => shift to (0, 0)
        dst -= dst.min(axis=0, keepdims=True)
        dst = np.around(dst, decimals=0)

        M_expanded = cv2.getPerspectiveTransform(rect, dst)
        max_width, max_height = dst.max(axis=0) + 1
        return M_expanded, max_width, max_height

    def get_parameters(self):
        return [self.jitter, self.keep_size, self.cval, self.mode,
                self.fit_output]


class _ElasticTransformationSamplingResult(object):
    def __init__(self, random_states, alphas, sigmas, orders, cvals, modes):
        self.random_states = random_states
        self.alphas = alphas
        self.sigmas = sigmas
        self.orders = orders
        self.cvals = cvals
        self.modes = modes


# TODO add independent sigmas for x/y
# TODO add independent alphas for x/y
# TODO add backend arg
class ElasticTransformation(meta.Augmenter):
    """
    Transform images by moving pixels locally around using displacement fields.

    The augmenter has the parameters ``alpha`` and ``sigma``. ``alpha``
    controls the strength of the displacement: higher values mean that pixels
    are moved further. ``sigma`` controls the smoothness of the displacement:
    higher values lead to smoother patterns -- as if the image was below water
    -- while low values will cause indivdual pixels to be moved very
    differently from their neighbours, leading to noisy and pixelated images.

    A relation of 10:1 seems to be good for ``alpha`` and ``sigma``, e.g.
    ``alpha=10`` and ``sigma=1`` or ``alpha=50``, ``sigma=5``. For ``128x128``
    a setting of ``alpha=(0, 70.0)``, ``sigma=(4.0, 6.0)`` may be a good
    choice and will lead to a water-like effect.

    Code here was initially inspired by
    https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a

    For a detailed explanation, see ::

        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003

    .. note::

        For coordinate-based inputs (keypoints, bounding boxes, polygons,
        ...), this augmenter still has to perform an image-based augmentation,
        which will make it significantly slower for such inputs than other
        augmenters. See :ref:`performance`.

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

        - (1) Always handled by ``cv2``.
        - (2) Always handled by ``scipy``.
        - (3) Only supported for ``order != 0``. Will fail for ``order=0``.
        - (4) Mapped internally to ``float64`` when ``order=1``.
        - (5) Mapped internally to ``int16`` when ``order>=2``.
        - (6) Handled by ``cv2`` when ``order=0`` or ``order=1``, otherwise by
              ``scipy``.
        - (7) Mapped internally to ``float32``.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Strength of the distortion field. Higher values mean that pixels are
        moved further with respect to the distortion field's direction. Set
        this to around 10 times the value of `sigma` for visible effects.

            * If number, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter will be used to
              sample a value per image.

    sigma : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the gaussian kernel used to smooth the distortion
        fields. Higher values (for ``128x128`` images around 5.0) lead to more
        water-like effects, while lower values (for ``128x128`` images
        around ``1.0`` and lower) lead to more noisy, pixelated images. Set
        this to around 1/10th of `alpha` for visible effects.

            * If number, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``StochasticParameter``, then that parameter will be used to
              sample a value per image.

    order : int or list of int or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        Interpolation order to use. Same meaning as in
        :func:`scipy.ndimage.map_coordinates` and may take any integer value
        in the range ``0`` to ``5``, where orders close to ``0`` are faster.

            * If a single int, then that order will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``imgaug.ALL``, then equivalant to list
              ``[0, 1, 2, 3, 4, 5]``.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the order value to use.

    cval : number or tuple of number or list of number or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to ``constant``.
        For standard ``uint8`` images (value range ``0`` to ``255``), this
        value may also should also be in the range ``0`` to ``255``. It may
        be a ``float`` value, even for images with integer dtypes.

            * If this is a single number, then that value will be used
              (e.g. ``0`` results in black pixels).
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then a random value will be picked from that list per
              image.
            * If ``imgaug.ALL``, a value from the discrete range ``[0..255]``
              will be sampled per image.
            * If a ``StochasticParameter``, a new value will be sampled from
              the parameter per image.

    mode : str or list of str or imgaug.ALL or imgaug.parameters.StochasticParameter, optional
        Parameter that defines the handling of newly created pixels.
        May take the same values as in :func:`scipy.ndimage.map_coordinates`,
        i.e. ``constant``, ``nearest``, ``reflect`` or ``wrap``.

            * If a single string, then that mode will be used for all images.
            * If a list of strings, then per image a random mode will be picked
              from that list.
            * If ``imgaug.ALL``, then a random mode from all possible modes
              will be picked.
            * If ``StochasticParameter``, then the mode will be sampled from
              that parameter per image, i.e. it must return only the above
              mentioned strings.

    polygon_recoverer : 'auto' or None or imgaug.augmentables.polygons._ConcavePolygonRecoverer, optional
        The class to use to repair invalid polygons.
        If ``"auto"``, a new instance of
        :class`imgaug.augmentables.polygons._ConcavePolygonRecoverer`
        will be created.
        If ``None``, no polygon recoverer will be used.
        If an object, then that object will be used and must provide a
        ``recover_from()`` method, similar to
        :class:`imgaug.augmentables.polygons._ConcavePolygonRecoverer`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.ElasticTransformation(alpha=50.0, sigma=5.0)

    Apply elastic transformations with a strength/alpha of ``50.0`` and
    smoothness of ``5.0`` to all images.

    >>> aug = iaa.ElasticTransformation(alpha=(0.0, 70.0), sigma=5.0)

    Apply elastic transformations with a strength/alpha that comes
    from the interval ``[0.0, 70.0]`` (randomly picked per image) and
    with a smoothness of ``5.0``.

    """

    NB_NEIGHBOURING_KEYPOINTS = 3
    NEIGHBOURING_KEYPOINTS_DISTANCE = 1.0
    KEYPOINT_AUG_ALPHA_THRESH = 0.05
    # even at high alphas we don't augment keypoints if the sigma is too low,
    # because then the pixel movements are mostly gaussian noise anyways
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
                 polygon_recoverer="auto", name=None, deterministic=False,
                 random_state=None):
        super(ElasticTransformation, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.alpha = iap.handle_continuous_param(
            alpha, "alpha", value_range=(0, None), tuple_to_uniform=True,
            list_to_choice=True)
        self.sigma = iap.handle_continuous_param(
            sigma, "sigma", value_range=(0, None), tuple_to_uniform=True,
            list_to_choice=True)

        self.order = self._handle_order_arg(order)
        self.cval = _handle_cval_arg(cval)
        self.mode = self._handle_mode_arg(mode)

        self.polygon_recoverer = polygon_recoverer
        if polygon_recoverer == "auto":
            self.polygon_recoverer = _ConcavePolygonRecoverer()

        # Special order, mode and cval parameters for heatmaps and
        # segmentation maps. These may either be None or a fixed value.
        # Stochastic parameters are currently *not* supported.
        # If set to None, the same values as for images will be used.
        # That is really not recommended for the cval parameter.
        #
        self._order_heatmaps = 3
        self._order_segmentation_maps = 0
        self._mode_heatmaps = "constant"
        self._mode_segmentation_maps = "constant"
        self._cval_heatmaps = 0.0
        self._cval_segmentation_maps = 0

    @classmethod
    def _handle_order_arg(cls, order):
        if order == ia.ALL:
            return iap.Choice([0, 1, 2, 3, 4, 5])
        else:
            return iap.handle_discrete_param(
                order, "order", value_range=(0, 5), tuple_to_uniform=True,
                list_to_choice=True, allow_floats=False)

    @classmethod
    def _handle_mode_arg(cls, mode):
        if mode == ia.ALL:
            return iap.Choice(["constant", "nearest", "reflect", "wrap"])
        elif ia.is_string(mode):
            return iap.Deterministic(mode)
        elif ia.is_iterable(mode):
            assert all([ia.is_string(val) for val in mode]), (
                "Expected mode list to only contain strings, got "
                "types %s." % (
                    ", ".join([str(type(val)) for val in mode]),))
            return iap.Choice(mode)
        elif isinstance(mode, iap.StochasticParameter):
            return mode
        else:
            raise Exception(
                "Expected mode to be imgaug.ALL, a string, a list of strings "
                "or StochasticParameter, got %s." % (type(mode),))

    def _draw_samples(self, nb_images, random_state):
        rss = random_state.duplicate(nb_images+5)
        alphas = self.alpha.draw_samples((nb_images,), random_state=rss[-5])
        sigmas = self.sigma.draw_samples((nb_images,), random_state=rss[-4])
        orders = self.order.draw_samples((nb_images,), random_state=rss[-3])
        cvals = self.cval.draw_samples((nb_images,), random_state=rss[-2])
        modes = self.mode.draw_samples((nb_images,), random_state=rss[-1])
        return _ElasticTransformationSamplingResult(
            rss[0:-5], alphas, sigmas, orders, cvals, modes)

    def _augment_batch(self, batch, random_state, parents, hooks):
        if batch.images is not None:
            iadt.gate_dtypes(
                batch.images,
                allowed=["bool",
                         "uint8", "uint16", "uint32", "uint64",
                         "int8", "int16", "int32", "int64",
                         "float16", "float32", "float64"],
                disallowed=["uint128", "uint256",
                            "int128", "int256",
                            "float96", "float128", "float256"],
                augmenter=self)

        shapes = batch.get_rowwise_shapes()
        samples = self._draw_samples(len(shapes), random_state)

        for i, shape in enumerate(shapes):
            dx, dy = self._generate_shift_maps(
                shape[0:2],
                alpha=samples.alphas[i],
                sigma=samples.sigmas[i],
                random_state=samples.random_states[i])

            if batch.images is not None:
                batch.images[i] = self._augment_image_by_samples(
                    batch.images[i], i, samples, dx, dy)
            if batch.heatmaps is not None:
                batch.heatmaps[i] = self._augment_hm_or_sm_by_samples(
                    batch.heatmaps[i], i, samples, dx, dy, "arr_0to1",
                    self._cval_heatmaps, self._mode_heatmaps,
                    self._order_heatmaps)
            if batch.segmentation_maps is not None:
                batch.segmentation_maps[i] = self._augment_hm_or_sm_by_samples(
                    batch.segmentation_maps[i], i, samples, dx, dy, "arr",
                    self._cval_segmentation_maps, self._mode_segmentation_maps,
                    self._order_segmentation_maps)
            if batch.keypoints is not None:
                batch.keypoints[i] = self._augment_kpsoi_by_samples(
                    batch.keypoints[i], i, samples, dx, dy)
            if batch.bounding_boxes is not None:
                batch.bounding_boxes[i] = self._augment_bbsoi_by_samples(
                    batch.bounding_boxes[i], i, samples, dx, dy)
            if batch.polygons is not None:
                batch.polygons[i] = self._augment_psoi_by_samples(
                    batch.polygons[i], i, samples, dx, dy)
            if batch.line_strings is not None:
                batch.line_strings[i] = self._augment_lsoi_by_samples(
                    batch.line_strings[i], i, samples, dx, dy)

        return batch

    def _augment_image_by_samples(self, image, row_idx, samples, dx, dy):
        min_value, _center_value, max_value = \
            iadt.get_value_range_of_dtype(image.dtype)
        cval = max(min(samples.cvals[row_idx], max_value), min_value)

        input_dtype = image.dtype
        if image.dtype.name == "float16":
            image = image.astype(np.float32)

        image_aug = self._map_coordinates(
            image, dx, dy,
            order=samples.orders[row_idx],
            cval=cval,
            mode=samples.modes[row_idx])

        if image.dtype.name != input_dtype.name:
            image_aug = iadt.restore_dtypes_(image_aug, input_dtype)
        return image_aug

    def _augment_hm_or_sm_by_samples(self, augmentable, row_idx, samples,
                                     dx, dy, arr_attr_name, cval, mode, order):
        cval = cval if cval is not None else samples.cvals[row_idx]
        mode = mode if mode is not None else samples.modes[row_idx]
        order = order if order is not None else samples.orders[row_idx]

        # note that we do not have to check for zero-sized axes here,
        # because _generate_shift_maps(), _map_coordinates(), .resize()
        # and np.clip() are all known to handle arrays with zero-sized axes

        arr = getattr(augmentable, arr_attr_name)

        if arr.shape[0:2] == augmentable.shape[0:2]:
            arr_warped = self._map_coordinates(
                arr, dx, dy, order=order, cval=cval, mode=mode)

            # interpolation in map_coordinates() can cause some values to
            # be below/above 1.0, so we clip here
            if order >= 3 and isinstance(augmentable, ia.HeatmapsOnImage):
                arr_warped = np.clip(arr_warped, 0.0, 1.0, out=arr_warped)

            setattr(augmentable, arr_attr_name, arr_warped)
        else:
            # Heatmaps/Segmaps do not have the same size as augmented
            # images. This may result in indices of moved pixels being
            # different. To prevent this, we use the same image size as
            # for the base images, but that requires resizing the heatmaps
            # temporarily to the image sizes.
            height_orig, width_orig = arr.shape[0:2]
            augmentable = augmentable.resize(augmentable.shape[0:2])
            arr = getattr(augmentable, arr_attr_name)

            # TODO will it produce similar results to first downscale the
            #      shift maps and then remap? That would make the remap
            #      step take less operations and would also mean that the
            #      heatmaps wouldnt have to be scaled up anymore. It would
            #      also simplify the code as this branch could be merged
            #      with the one above.
            arr_warped = self._map_coordinates(
                arr, dx, dy, order=order, cval=cval, mode=mode)

            # interpolation in map_coordinates() can cause some values to
            # be below/above 1.0, so we clip here
            if order >= 3 and isinstance(augmentable, ia.HeatmapsOnImage):
                arr_warped = np.clip(arr_warped, 0.0, 1.0, out=arr_warped)

            setattr(augmentable, arr_attr_name, arr_warped)

            augmentable = augmentable.resize((height_orig, width_orig))

        return augmentable

    def _augment_kpsoi_by_samples(self, kpsoi, row_idx, samples, dx, dy):
        height, width = kpsoi.shape[0:2]
        alpha = samples.alphas[row_idx]
        sigma = samples.sigmas[row_idx]

        # TODO add test for keypoint alignment when keypoints are empty
        # Note: this block must be placed after _generate_shift_maps() to
        # keep samples aligned
        # Note: we should stop for zero-sized axes early here, event though
        # there is a height/width check for each keypoint, because the
        # channel number can also be zero
        image_has_zero_sized_axes = (0 in kpsoi.shape)
        params_below_thresh = (
            alpha <= self.KEYPOINT_AUG_ALPHA_THRESH
            or sigma <= self.KEYPOINT_AUG_SIGMA_THRESH)

        if kpsoi.empty or image_has_zero_sized_axes or params_below_thresh:
            # ElasticTransformation does not change the shape, hence we can
            # skip the below steps
            return kpsoi

        for kp in kpsoi.keypoints:
            within_image_plane = (0 <= kp.x < width and 0 <= kp.y < height)
            if within_image_plane:
                kp_neighborhood = kp.generate_similar_points_manhattan(
                    self.NB_NEIGHBOURING_KEYPOINTS,
                    self.NEIGHBOURING_KEYPOINTS_DISTANCE,
                    return_array=True
                )

                # We can clip here, because we made sure above that the
                # keypoint is inside the image plane. Keypoints at the
                # bottom row or right columns might be rounded outside
                # the image plane, which we prevent here. We reduce
                # neighbours to only those within the image plane as only
                # for such points we know where to move them.
                xx = np.round(kp_neighborhood[:, 0]).astype(np.int32)
                yy = np.round(kp_neighborhood[:, 1]).astype(np.int32)
                inside_image_mask = np.logical_and(
                    np.logical_and(0 <= xx, xx < width),
                    np.logical_and(0 <= yy, yy < height)
                )
                xx = xx[inside_image_mask]
                yy = yy[inside_image_mask]

                xxyy = np.concatenate(
                    [xx[:, np.newaxis], yy[:, np.newaxis]],
                    axis=1)

                xxyy_aug = np.copy(xxyy).astype(np.float32)
                xxyy_aug[:, 0] += dx[yy, xx]
                xxyy_aug[:, 1] += dy[yy, xx]

                med = ia.compute_geometric_median(xxyy_aug)
                # uncomment to use average instead of median
                # med = np.average(xxyy_aug, 0)
                kp.x = med[0]
                kp.y = med[1]

        return kpsoi

    def _augment_psoi_by_samples(self, psoi, row_idx, samples, dx, dy):
        func = functools.partial(self._augment_kpsoi_by_samples,
                                 row_idx=row_idx, samples=samples, dx=dx, dy=dy)
        return self._apply_to_polygons_as_keypoints(
            psoi, func, recoverer=self.polygon_recoverer)

    def _augment_lsoi_by_samples(self, lsoi, row_idx, samples, dx, dy):
        func = functools.partial(self._augment_kpsoi_by_samples,
                                 row_idx=row_idx, samples=samples, dx=dx, dy=dy)
        return self._apply_to_cbaois_as_keypoints(lsoi, func)

    def _augment_bbsoi_by_samples(self, bbsoi, row_idx, samples, dx, dy):
        func = functools.partial(self._augment_kpsoi_by_samples,
                                 row_idx=row_idx, samples=samples, dx=dx, dy=dy)
        return self._apply_to_cbaois_as_keypoints(bbsoi, func)

    def get_parameters(self):
        return [self.alpha, self.sigma, self.order, self.cval, self.mode]

    @classmethod
    def _generate_shift_maps(cls, shape, alpha, sigma, random_state):
        assert len(shape) == 2, ("Expected 2d shape, got %s." % (shape,))

        ksize = blur_lib._compute_gaussian_blur_ksize(sigma)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize

        padding = ksize
        h, w = shape[0:2]
        h_pad = h + 2*padding
        w_pad = w + 2*padding

        # The step of random number generation could be batched, so that
        # random numbers are sampled once for the whole batch. Would get rid
        # of creating many random_states.
        dxdy_unsmoothed = random_state.random((2 * h_pad, w_pad)) * 2 - 1

        dx_unsmoothed = dxdy_unsmoothed[0:h_pad, :]
        dy_unsmoothed = dxdy_unsmoothed[h_pad:, :]

        # TODO could this also work with an average blur? would probably be
        #      faster
        dx = blur_lib.blur_gaussian_(dx_unsmoothed, sigma) * alpha
        dy = blur_lib.blur_gaussian_(dy_unsmoothed, sigma) * alpha

        if padding > 0:
            dx = dx[padding:-padding, padding:-padding]
            dy = dy[padding:-padding, padding:-padding]

        return dx, dy

    @classmethod
    def _map_coordinates(cls, image, dx, dy, order=1, cval=0, mode="constant"):
        """Remap pixels in an image according to x/y shift maps.

        dtype support::

            if (backend="scipy" and order=0)::

                * ``uint8``: yes
                * ``uint16``: yes
                * ``uint32``: yes
                * ``uint64``: no (1)
                * ``int8``: yes
                * ``int16``: yes
                * ``int32``: yes
                * ``int64``: no (2)
                * ``float16``: yes
                * ``float32``: yes
                * ``float64``: yes
                * ``float128``: no (3)
                * ``bool``: yes

                - (1) produces array filled with only 0
                - (2) produces array filled with <min_value> when testing
                      with <max_value>
                - (3) causes: 'data type no supported'

            if (backend="scipy" and order>0)::

                * ``uint8``: yes (1)
                * ``uint16``: yes (1)
                * ``uint32``: yes (1)
                * ``uint64``: yes (1)
                * ``int8``: yes (1)
                * ``int16``: yes (1)
                * ``int32``: yes (1)
                * ``int64``: yes (1)
                * ``float16``: yes (1)
                * ``float32``: yes (1)
                * ``float64``: yes (1)
                * ``float128``: no (2)
                * ``bool``: yes

                - (1) rather loose test, to avoid having to re-compute the
                      interpolation
                - (2) causes: 'data type no supported'

            if (backend="cv2" and order=0)::

                * ``uint8``: yes
                * ``uint16``: yes
                * ``uint32``: no (1)
                * ``uint64``: no (2)
                * ``int8``: yes
                * ``int16``: yes
                * ``int32``: yes
                * ``int64``: no (2)
                * ``float16``: yes
                * ``float32``: yes
                * ``float64``: yes
                * ``float128``: no (3)
                * ``bool``: no (4)

                - (1) causes: src data type = 6 is not supported
                - (2) silently converts to int32
                - (3) causes: src data type = 13 is not supported
                - (4) causes: src data type = 0 is not supported

            if (backend="cv2" and order=1)::

                * ``uint8``: yes
                * ``uint16``: yes
                * ``uint32``: no (1)
                * ``uint64``: no (2)
                * ``int8``: no (2)
                * ``int16``: no (2)
                * ``int32``: no (2)
                * ``int64``: no (2)
                * ``float16``: yes
                * ``float32``: yes
                * ``float64``: yes
                * ``float128``: no (3)
                * ``bool``: no (4)

                - (1) causes: src data type = 6 is not supported
                - (2) causes: OpenCV(3.4.5) (...)/imgwarp.cpp:1805:
                      error: (-215:Assertion failed) ifunc != 0 in function
                      'remap'
                - (3) causes: src data type = 13 is not supported
                - (4) causes: src data type = 0 is not supported

            if (backend="cv2" and order>=2)::

                * ``uint8``: yes
                * ``uint16``: yes
                * ``uint32``: no (1)
                * ``uint64``: no (2)
                * ``int8``: no (2)
                * ``int16``: yes
                * ``int32``: no (2)
                * ``int64``: no (2)
                * ``float16``: yes
                * ``float32``: yes
                * ``float64``: yes
                * ``float128``: no (3)
                * ``bool``: no (4)

                - (1) causes: src data type = 6 is not supported
                - (2) causes: OpenCV(3.4.5) (...)/imgwarp.cpp:1805:
                      error: (-215:Assertion failed) ifunc != 0 in function
                      'remap'
                - (3) causes: src data type = 13 is not supported
                - (4) causes: src data type = 0 is not supported

        """
        if image.size == 0:
            return np.copy(image)

        if order == 0 and image.dtype.name in ["uint64", "int64"]:
            raise Exception(
                "dtypes uint64 and int64 are only supported in "
                "ElasticTransformation for order=0, got order=%d with "
                "dtype=%s." % (order, image.dtype.name))

        input_dtype = image.dtype
        if image.dtype.name == "bool":
            image = image.astype(np.float32)
        elif order == 1 and image.dtype.name in ["int8", "int16", "int32"]:
            image = image.astype(np.float64)
        elif order >= 2 and image.dtype.name == "int8":
            image = image.astype(np.int16)
        elif order >= 2 and image.dtype.name == "int32":
            image = image.astype(np.float64)

        shrt_max = 32767  # maximum of datatype `short`
        backend = "cv2"
        if order == 0:
            bad_dtype_cv2 = (
                image.dtype.name in [
                    "uint32", "uint64",
                    "int64",
                    "float128",
                    "bool"]
            )
        elif order == 1:
            bad_dtype_cv2 = (
                image.dtype.name in [
                    "uint32", "uint64",
                    "int8", "int16", "int32", "int64",
                    "float128",
                    "bool"]
            )
        else:
            bad_dtype_cv2 = (
                image.dtype.name in [
                    "uint32", "uint64",
                    "int8", "int32", "int64",
                    "float128",
                    "bool"]
            )

        bad_dx_shape_cv2 = (dx.shape[0] >= shrt_max or dx.shape[1] >= shrt_max)
        bad_dy_shape_cv2 = (dy.shape[0] >= shrt_max or dy.shape[1] >= shrt_max)
        if bad_dtype_cv2 or bad_dx_shape_cv2 or bad_dy_shape_cv2:
            backend = "scipy"

        assert image.ndim == 3, (
            "Expected 3-dimensional image, got %d dimensions." % (image.ndim,))
        result = np.copy(image)
        height, width = image.shape[0:2]
        if backend == "scipy":
            h, w = image.shape[0:2]
            y, x = np.meshgrid(
                np.arange(h).astype(np.float32),
                np.arange(w).astype(np.float32),
                indexing="ij")
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

            y, x = np.meshgrid(
                np.arange(h).astype(np.float32),
                np.arange(w).astype(np.float32),
                indexing="ij")
            x_shifted = x + (-1) * dx
            y_shifted = y + (-1) * dy

            if image.dtype.kind == "f":
                cval = float(cval)
            else:
                cval = int(cval)
            border_mode = cls._MAPPING_MODE_SCIPY_CV2[mode]
            interpolation = cls._MAPPING_ORDER_SCIPY_CV2[order]

            is_nearest_neighbour = (interpolation == cv2.INTER_NEAREST)
            map1, map2 = cv2.convertMaps(
                x_shifted, y_shifted, cv2.CV_16SC2,
                nninterpolation=is_nearest_neighbour)
            # remap only supports up to 4 channels
            if nb_channels <= 4:
                result = cv2.remap(
                    image, map1, map2, interpolation=interpolation,
                    borderMode=border_mode, borderValue=cval)
                if image.ndim == 3 and result.ndim == 2:
                    result = result[..., np.newaxis]
            else:
                current_chan_idx = 0
                result = []
                while current_chan_idx < nb_channels:
                    channels = image[..., current_chan_idx:current_chan_idx+4]
                    result_c = cv2.remap(
                        channels, map1, map2, interpolation=interpolation,
                        borderMode=border_mode, borderValue=cval)
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
    Rotate images clockwise by multiples of 90 degrees.

    This could also be achieved using ``Affine``, but ``Rot90`` is
    significantly more efficient.

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

            * If a single ``int``, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If ``imgaug.ALL``, then equivalant to list ``[0, 1, 2, 3]``.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the value to use.

    keep_size : bool, optional
        After rotation by an odd-valued `k` (e.g. 1 or 3), the resulting image
        may have a different height/width than the original image.
        If this parameter is set to ``True``, then the rotated
        image will be resized to the input image's size. Note that this might
        also cause the augmented image to look distorted.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.


    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Rot90(1)

    Rotate all images by 90 degrees.
    Resize these images afterwards to keep the size that they had before
    augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90([1, 3])

    Rotate all images by 90 or 270 degrees.
    Resize these images afterwards to keep the size that they had before
    augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90((1, 3))

    Rotate all images by 90, 180 or 270 degrees.
    Resize these images afterwards to keep the size that they had before
    augmentation.
    This may cause the images to look distorted.

    >>> aug = iaa.Rot90((1, 3), keep_size=False)

    Rotate all images by 90, 180 or 270 degrees.
    Does not resize to the original image size afterwards, i.e. each image's
    size may change.

    """

    def __init__(self, k, keep_size=True, name=None, deterministic=False,
                 random_state=None):
        super(Rot90, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        if k == ia.ALL:
            k = [0, 1, 2, 3]
        self.k = iap.handle_discrete_param(
            k, "k", value_range=None, tuple_to_uniform=True,
            list_to_choice=True, allow_floats=False)

        self.keep_size = keep_size

    def _draw_samples(self, nb_images, random_state):
        return self.k.draw_samples((nb_images,), random_state=random_state)

    def _augment_batch(self, batch, random_state, parents, hooks):
        ks = self._draw_samples(batch.nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._augment_arrays_by_samples(
                batch.images, ks, self.keep_size, ia.imresize_single_image)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps, "arr_0to1", ks)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, "arr", ks)

        for augm_name in ["keypoints", "bounding_boxes", "polygons",
                          "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(
                    self._augment_keypoints_by_samples,
                    ks=ks)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    @classmethod
    def _augment_arrays_by_samples(cls, arrs, ks, keep_size, resize_func):
        input_was_array = ia.is_np_array(arrs)
        input_dtype = arrs.dtype if input_was_array else None
        arrs_aug = []
        for arr, k_i in zip(arrs, ks):
            # adding axes here rotates clock-wise instead of ccw
            arr_aug = np.rot90(arr, k_i, axes=(1, 0))

            do_resize = (
                keep_size
                and arr.shape != arr_aug.shape
                and resize_func is not None)
            if do_resize:
                arr_aug = resize_func(arr_aug, arr.shape[0:2])
            arrs_aug.append(arr_aug)
        if keep_size and input_was_array:
            n_shapes = len(set([arr.shape for arr in arrs_aug]))
            if n_shapes == 1:
                arrs_aug = np.array(arrs_aug, dtype=input_dtype)
        return arrs_aug

    def _augment_maps_by_samples(self, augmentables, arr_attr_name, ks):
        arrs = [getattr(map_i, arr_attr_name) for map_i in augmentables]
        arrs_aug = self._augment_arrays_by_samples(
            arrs, ks, self.keep_size, None)

        maps_aug = []
        gen = zip(augmentables, arrs, arrs_aug, ks)
        for augmentable_i, arr, arr_aug, k_i in gen:
            shape_orig = arr.shape
            setattr(augmentable_i, arr_attr_name, arr_aug)
            if self.keep_size:
                augmentable_i = augmentable_i.resize(shape_orig[0:2])
            elif k_i % 2 == 1:
                h, w = augmentable_i.shape[0:2]
                augmentable_i.shape = tuple(
                    [w, h] + list(augmentable_i.shape[2:]))
            else:
                # keep_size was False, but rotated by a multiple of 2,
                # hence height and width do not change
                pass
            maps_aug.append(augmentable_i)
        return maps_aug

    def _augment_keypoints_by_samples(self, keypoints_on_images, ks):
        result = []
        for kpsoi_i, k_i in zip(keypoints_on_images, ks):
            shape_orig = kpsoi_i.shape

            if (k_i % 4) == 0:
                result.append(kpsoi_i)
            else:
                k_i = int(k_i) % 4  # this is also correct when k_i is negative
                h, w = kpsoi_i.shape[0:2]
                h_aug, w_aug = (h, w) if (k_i % 2) == 0 else (w, h)

                for kp in kpsoi_i.keypoints:
                    y, x = kp.y, kp.x
                    yr, xr = y, x
                    wr, hr = w, h
                    for _ in sm.xrange(k_i):
                        # for int coordinates this would instead be
                        #   xr, yr = (hr - 1) - yr, xr
                        # here we assume that coordinates are always
                        # subpixel-accurate
                        xr, yr = hr - yr, xr
                        wr, hr = hr, wr
                    kp.x = xr
                    kp.y = yr

                shape_aug = tuple([h_aug, w_aug] + list(kpsoi_i.shape[2:]))
                kpsoi_i.shape = shape_aug

                if self.keep_size and (h, w) != (h_aug, w_aug):
                    kpsoi_i = kpsoi_i.on(shape_orig)
                    kpsoi_i.shape = shape_orig

                result.append(kpsoi_i)
        return result

    def get_parameters(self):
        return [self.k, self.keep_size]
