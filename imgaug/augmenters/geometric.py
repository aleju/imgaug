"""
Augmenters that apply affine transformations or other similar augmentations.

Do not import directly from this file, as the categorization is not final.
Use instead
    `from imgaug import augmenters as iaa`
and then e.g. ::

    seq = iaa.Sequential([
        iaa.Affine(...),
        iaa.PerspectiveTransform(...)
    ])

List of augmenters:
    * Affine
    * PiecewiseAffine
    * PerspectiveTransform
    * ElasticTransformation
"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, Choice, DiscreteUniform, Normal, Uniform
import numpy as np
import math
from scipy import ndimage
from skimage import transform as tf
import cv2
import six.moves as sm

from .meta import Augmenter

class Affine(Augmenter):
    """
    Augmenter to apply affine transformations to images.

    This is mostly a wrapper around skimage's AffineTransform class and
    warp function.

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

    Parameters
    ----------
    scale : float or tuple of two floats or StochasticParameter or dict {"x": float/tuple/StochasticParameter, "y": float/tuple/StochasticParameter}, optional(default=1.0)
        Scaling factor to use, where 1.0 represents no change and 0.5 is
        zoomed out to 50 percent of the original size.
            * If a single float, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled from the range
              a <= x <= b per image. That value will be used identically for
              both x- and y-axis.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`scale`). Using a dictionary allows to
              set different values for the axis. If they are set to the same
              ranges, different values may still be sampled per axis.

    translate_percent : float or tuple of two floats or StochasticParameter or dict {"x": float/tuple/StochasticParameter, "y": float/tuple/StochasticParameter}, optional(default=1.0)
        Translation in percent relative to the image
        height/width (x-translation, y-translation) to use,
        where 0 represents no change and 0.5 is half of the image
        height/width.
            * If a single float, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled from the range
              a <= x <= b per image. That percent value will be used identically
              for both x- and y-axis.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`translate_percent`).
              Using a dictionary allows to set different values for the axis.
              If they are set to the same ranges, different values may still
              be sampled per axis.

    translate_px : int or tuple of two ints or StochasticParameter or dict {"x": int/tuple/StochasticParameter, "y": int/tuple/StochasticParameter}, optional(default=1.0)
        Translation in
        pixels.
            * If a single int, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled from the discrete
              range [a .. b] per image. That number will be used identically
              for both x- and y-axis.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`translate_px`).
              Using a dictionary allows to set different values for the axis.
              If they are set to the same ranges, different values may still
              be sampled per axis.

    rotate : float or int or tuple of two floats/ints or StochasticParameter, optional(default=0)
        Rotation in degrees (NOT radians), i.e. expected value range is
        0 to 360 for positive rotations (may also be negative).
            * If a float/int, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled per image from the
              range a <= x <= b and be used as the rotation value.
            * If a StochasticParameter, then this parameter will be used to
              sample the rotation value per image.

    shear : float or int or tuple of two floats/ints or StochasticParameter, optional(default=0)
        Shear in degrees (NOT radians), i.e. expected value range is
        0 to 360 for positive shear (may also be negative).
            * If a float/int, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled per image from the
              range a <= x <= b and be used as the rotation value.
            * If a StochasticParameter, then this parameter will be used to
              sample the shear value per image.

    order : int or iterable of int or ia.ALL or StochasticParameter, optional(default=1)
        Interpolation order to use. Same meaning as in
        skimage:
            * 0: Nearest-neighbor
            * 1: Bi-linear (default)
            * 2: Bi-quadratic (not recommended by skimage)
            * 3: Bi-cubic
            * 4: Bi-quartic
            * 5: Bi-quintic
        Method 0 and 1 are fast, 3 is a bit slower, 4 and 5 are very
        slow.
        If the backend is `cv2`, the mapping to opencv's interpolation modes
        is as follows:
            * 0 -> cv2.INTER_NEAREST
            * 1 -> cv2.INTER_LINEAR
            * 2 -> cv2.INTER_CUBIC
            * 3 -> cv2.INTER_CUBIC
            * 4 -> cv2.INTER_CUBIC
        As datatypes this parameter
        accepts:
            * If a single int, then that order will be used for all images.
            * If an iterable, then for each image a random value will be sampled
              from that iterable (i.e. list of allowed order values).
            * If ia.ALL, then equivalant to list [0, 1, 3, 4, 5].
            * If StochasticParameter, then that parameter is queried per image
              to sample the order value to use.

    cval : number or tuple of two number or ia.ALL or StochasticParameter, optional(default=0)
        The constant value used for skimage's transform function.
        This is the value used to fill up pixels in the result image that
        didn't exist in the input image (e.g. when translating to the left,
        some new pixels are created at the right). Such a fill-up with a
        constant value only happens, when `mode` is "constant".
        For standard uint8 images (value range 0-255), this value may also
        come from the range 0-255. It may be a float value, even for
        integer image dtypes.
            * If this is a single int or float, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple (a, b), then a random value from the range a <= x <= b
              is picked per image.
            * If ia.ALL, a value from the discrete range [0 .. 255] will be
              sampled per image.
            * If a StochasticParameter, a new value will be sampled from the
              parameter per image.

    mode : string or list of string or ia.ALL or StochasticParameter, optional(default="constant")
        Parameter that defines the handling of newly created pixels.
        Same meaning as in skimage (and numpy.pad):
            * "constant": Pads with a constant value
            * "edge": Pads with the edge values of array
            * "symmetric": Pads with the reflection of the vector mirrored
              along the edge of the array.
            * "reflect": Pads with the reflection of the vector mirrored on
              the first and last values of the vector along each axis.
            * "wrap": Pads with the wrap of the vector along the axis.
              The first values are used to pad the end and the end values
              are used to pad the beginning.
        If `cv2` is chosen as the backend the mapping is as
        follows:
            * "constant" -> cv2.BORDER_CONSTANT
            * "edge" -> cv2.BORDER_REPLICATE
            * "symmetric" -> cv2.BORDER_REFLECT
            * "reflect" -> cv2.BORDER_REFLECT_101
            * "wrap" -> cv2.BORDER_WRAP
        The datatype of the parameter may
        be:
            * If a single string, then that mode will be used for all images.
            * If a list of strings, then per image a random mode will be picked
              from that list.
            * If ia.ALL, then a random mode from all possible modes will be
              picked.
            * If StochasticParameter, then the mode will be sampled from that
              parameter per image, i.e. it must return only the above mentioned
              strings.

    backend : string, optional(default="auto")
        Framework to use as a backend. Valid values are `auto`, `skimage`
        (scikit-image's warp) and `cv2` (opencv's warp).
        If `auto` is used, the augmenter will automatically try
        to use cv2 where possible (order must be in [0, 1, 3] and
        image's dtype uint8, otherwise skimage is chosen). It will
        silently fall back to skimage if order/dtype is not supported by cv2.
        cv2 is generally faster than skimage. It also supports RGB cvals,
        while skimage will resort to intensity cvals (i.e. 3x the same value
        as RGB). If `cv2` is chosen and order is 2 or 4, it will automatically
        fall back to order 3.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

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

    def __init__(self, scale=1.0, translate_percent=None, translate_px=None,
                 rotate=0.0, shear=0.0, order=1, cval=0, mode="constant",
                 backend="auto",
                 name=None, deterministic=False, random_state=None):
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
                self.order = Choice([0, 1, 3])
            else:
                self.order = Choice([0, 1, 3, 4, 5]) # dont use order=2 (bi-quadratic) because that is apparently currently not recommended (and throws a warning)
        elif ia.is_single_integer(order):
            ia.do_assert(0 <= order <= 5, "Expected order's integer value to be in range 0 <= x <= 5, got %d." % (order,))
            if backend == "cv2":
                ia.do_assert(order in [0, 1, 3])
            self.order = Deterministic(order)
        elif isinstance(order, list):
            ia.do_assert(all([ia.is_single_integer(val) for val in order]), "Expected order list to only contain integers, got types %s." % (str([type(val) for val in order]),))
            ia.do_assert(all([0 <= val <= 5 for val in order]), "Expected all of order's integer values to be in range 0 <= x <= 5, got %s." % (str(order),))
            if backend == "cv2":
                ia.do_assert(all([val in [0, 1, 3] for val in order]))
            self.order = Choice(order)
        elif isinstance(order, StochasticParameter):
            self.order = order
        else:
            raise Exception("Expected order to be imgaug.ALL, int, list of int or StochasticParameter, got %s." % (type(order),))

        if cval == ia.ALL:
            self.cval = Uniform(0, 255) # skimage transform expects float
        elif ia.is_single_number(cval):
            self.cval = Deterministic(cval)
        elif ia.is_iterable(cval):
            ia.do_assert(len(cval) == 2)
            ia.do_assert(0 <= cval[0] <= 255)
            ia.do_assert(0 <= cval[1] <= 255)
            self.cval = Uniform(cval[0], cval[1]) # skimage transform expects float
        elif isinstance(cval, StochasticParameter):
            self.cval = cval
        else:
            raise Exception("Expected cval to be imgaug.ALL, int, float or StochasticParameter, got %s." % (type(cval),))

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
            self.mode = Choice(["constant", "edge", "symmetric", "reflect", "wrap"])
        elif ia.is_string(mode):
            self.mode = Deterministic(mode)
        elif isinstance(mode, list):
            ia.do_assert(all([ia.is_string(val) for val in mode]))
            self.mode = Choice(mode)
        elif isinstance(mode, StochasticParameter):
            self.mode = mode
        else:
            raise Exception("Expected mode to be imgaug.ALL, a string, a list of strings or StochasticParameter, got %s." % (type(mode),))

        # scale
        # float | (float, float) | [float, float] | StochasticParameter
        def scale_handle_param(param, allow_dict):
            if isinstance(param, StochasticParameter):
                return param
            elif ia.is_single_number(param):
                ia.do_assert(param > 0.0, "Expected scale to have range (0, inf), got value %.4f. Note: The value to _not_ change the scale of images is 1.0, not 0.0." % (param,))
                return Deterministic(param)
            elif ia.is_iterable(param) and not isinstance(param, dict):
                ia.do_assert(len(param) == 2, "Expected scale tuple/list with 2 entries, got %d entries." % (len(param),))
                ia.do_assert(param[0] > 0.0 and param[1] > 0.0, "Expected scale tuple/list to have values in range (0, inf), got values %.4f and %.4f. Note: The value to _not_ change the scale of images is 1.0, not 0.0." % (param[0], param[1]))
                return Uniform(param[0], param[1])
            elif allow_dict and isinstance(param, dict):
                ia.do_assert("x" in param or "y" in param)
                x = param.get("x")
                y = param.get("y")

                x = x if x is not None else 1.0
                y = y if y is not None else 1.0

                return (scale_handle_param(x, False), scale_handle_param(y, False))
            else:
                raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(param),))
        self.scale = scale_handle_param(scale, True)

        # translate
        if translate_percent is None and translate_px is None:
            translate_px = 0

        ia.do_assert(translate_percent is None or translate_px is None)

        if translate_percent is not None:
            # translate by percent
            def translate_handle_param(param, allow_dict):
                if ia.is_single_number(param):
                    return Deterministic(float(param))
                elif ia.is_iterable(param) and not isinstance(param, dict):
                    ia.do_assert(len(param) == 2, "Expected translate_percent tuple/list with 2 entries, got %d entries." % (len(param),))
                    all_numbers = all([ia.is_single_number(p) for p in param])
                    ia.do_assert(all_numbers, "Expected translate_percent tuple/list to contain only numbers, got types %s." % (str([type(p) for p in param]),))
                    #ia.do_assert(param[0] > 0.0 and param[1] > 0.0, "Expected translate_percent tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (param[0], param[1]))
                    return Uniform(param[0], param[1])
                elif allow_dict and isinstance(param, dict):
                    ia.do_assert("x" in param or "y" in param)
                    x = param.get("x")
                    y = param.get("y")

                    x = x if x is not None else 0
                    y = y if y is not None else 0

                    return (translate_handle_param(x, False), translate_handle_param(y, False))
                elif isinstance(param, StochasticParameter):
                    return param
                else:
                    raise Exception("Expected float, int or tuple/list with 2 entries of both floats or ints or StochasticParameter. Got %s." % (type(param),))
            self.translate = translate_handle_param(translate_percent, True)
        else:
            # translate by pixels
            def translate_handle_param(param, allow_dict):
                if ia.is_single_integer(param):
                    return Deterministic(param)
                elif ia.is_iterable(param) and not isinstance(param, dict):
                    ia.do_assert(len(param) == 2, "Expected translate_px tuple/list with 2 entries, got %d entries." % (len(param),))
                    all_integer = all([ia.is_single_integer(p) for p in param])
                    ia.do_assert(all_integer, "Expected translate_px tuple/list to contain only integers, got types %s." % (str([type(p) for p in param]),))
                    return DiscreteUniform(param[0], param[1])
                elif allow_dict and isinstance(param, dict):
                    ia.do_assert("x" in param or "y" in param)
                    x = param.get("x")
                    y = param.get("y")

                    x = x if x is not None else 0
                    y = y if y is not None else 0

                    return (translate_handle_param(x, False), translate_handle_param(y, False))
                elif isinstance(param, StochasticParameter):
                    return param
                else:
                    raise Exception("Expected int or tuple/list with 2 ints or StochasticParameter. Got %s." % (type(param),))
            self.translate = translate_handle_param(translate_px, True)

        # rotate
        # StochasticParameter | float | int | (float or int, float or int) | [float or int, float or int]
        if isinstance(rotate, StochasticParameter):
            self.rotate = rotate
        elif ia.is_single_number(rotate):
            self.rotate = Deterministic(rotate)
        elif ia.is_iterable(rotate):
            ia.do_assert(len(rotate) == 2, "Expected rotate tuple/list with 2 entries, got %d entries." % (len(rotate),))
            ia.do_assert(all([ia.is_single_number(val) for val in rotate]), "Expected floats/ints in rotate tuple/list")
            self.rotate = Uniform(rotate[0], rotate[1])
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(rotate),))

        # shear
        # StochasticParameter | float | int | (float or int, float or int) | [float or int, float or int]
        if isinstance(shear, StochasticParameter):
            self.shear = shear
        elif ia.is_single_number(shear):
            self.shear = Deterministic(shear)
        elif ia.is_iterable(shear):
            ia.do_assert(len(shear) == 2, "Expected rotate tuple/list with 2 entries, got %d entries." % (len(shear),))
            ia.do_assert(all([ia.is_single_number(val) for val in shear]), "Expected floats/ints in shear tuple/list.")
            self.shear = Uniform(shear[0], shear[1])
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(shear),))

    def _augment_images(self, images, random_state, parents, hooks):
        #images = images if isinstance(images, list) else [images]
        nb_images = len(images)
        #result = [None] * nb_images
        result = images

        scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, mode_samples, order_samples = self._draw_samples(nb_images, random_state)

        for i in sm.xrange(nb_images):
            image = images[i]
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x, translate_y = translate_samples[0][i], translate_samples[1][i]
            #ia.do_assert(isinstance(translate_x, (float, int)))
            #ia.do_assert(isinstance(translate_y, (float, int)))
            if ia.is_single_float(translate_y):
                translate_y_px = int(round(translate_y * images[i].shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(round(translate_x * images[i].shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            cval = cval_samples[i]
            mode = mode_samples[i]
            order = order_samples[i]
            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 or shear != 0:
                cv2_bad_order = order not in [0, 1, 3]
                cv2_bad_dtype = image.dtype not in [np.uint8, np.float32, np.float64]
                cv2_bad_shape = image.shape[2] > 4
                cv2_impossible = cv2_bad_order or cv2_bad_dtype or cv2_bad_shape
                if self.backend == "skimage" or (self.backend == "auto" and cv2_impossible):
                    # cval contains 3 values as cv2 can handle 3, but skimage only 1
                    cval = cval[0]
                    # skimage does not clip automatically
                    if image.dtype == np.uint8:
                        cval = np.clip(cval, 0, 255)
                    image_warped = self._warp_skimage(
                        image,
                        scale_x, scale_y,
                        translate_x_px, translate_y_px,
                        rotate, shear,
                        cval,
                        mode, order
                    )
                else:
                    ia.do_assert(not cv2_bad_dtype, "cv2 backend can only handle images of dtype uint8, float32 and float64, got %s." % (image.dtype,))
                    # opencv seems to support arrays of three cvals (ie RGB)
                    # in python2, but for some reason not in python3, so
                    # we chose one cval here
                    #cval = cval[0]
                    #print(cval, type(cval), int(cval), type(int(cval)))
                    image_warped = self._warp_cv2(
                        image,
                        scale_x, scale_y,
                        translate_x_px, translate_y_px,
                        rotate, shear,
                        tuple([int(v) for v in cval]),
                        self.mode_map_skimage_cv2[mode],
                        self.order_map_skimage_cv2[order]
                    )

                result[i] = image_warped
            else:
                result[i] = images[i]

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        scale_samples, translate_samples, rotate_samples, shear_samples, _cval_samples, _mode_samples, _order_samples = self._draw_samples(nb_images, random_state)

        for i, keypoints_on_image in enumerate(keypoints_on_images):
            height, width = keypoints_on_image.height, keypoints_on_image.width
            shift_x = width / 2.0 - 0.5
            shift_y = height / 2.0 - 0.5
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x, translate_y = translate_samples[0][i], translate_samples[1][i]
            #ia.do_assert(isinstance(translate_x, (float, int)))
            #ia.do_assert(isinstance(translate_y, (float, int)))
            if ia.is_single_float(translate_y):
                translate_y_px = int(round(translate_y * keypoints_on_image.shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(round(translate_x * keypoints_on_image.shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            #cval = cval_samples[i]
            #mode = mode_samples[i]
            #order = order_samples[i]
            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 or shear != 0:
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
                #print("coords", coords)
                #print("matrix", matrix.params)
                coords_aug = tf.matrix_transform(coords, matrix.params)
                #print("coords before", coords)
                #print("coordsa ftre", coords_aug, np.around(coords_aug).astype(np.int32))
                result.append(ia.KeypointsOnImage.from_coords_array(coords_aug, shape=keypoints_on_image.shape))
            else:
                result.append(keypoints_on_image)
        return result

    def get_parameters(self):
        return [self.scale, self.translate, self.rotate, self.shear]

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

        return scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, mode_samples, order_samples

    def _warp_skimage(self, image, scale_x, scale_y, translate_x_px, translate_y_px, rotate, shear, cval, mode, order):
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
        image_warped = tf.warp(
            image,
            matrix.inverse,
            order=order,
            mode=mode,
            cval=cval,
            preserve_range=True
        )
        # warp changes uint8 to float64, making this necessary
        if image_warped.dtype != image.dtype:
            image_warped = image_warped.astype(image.dtype, copy=False)
        return image_warped

    def _warp_cv2(self, image, scale_x, scale_y, translate_x_px, translate_y_px, rotate, shear, cval, mode, order):
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

        image_warped = cv2.warpAffine(
            image,
            matrix.params[:2],
            #np.zeros((2, 3)),
            dsize=(width, height),
            flags=order,
            borderMode=mode,
            borderValue=cval
        )

        # cv2 warp drops last axis if shape is (H, W, 1)
        if image_warped.ndim == 2:
            image_warped = image_warped[..., np.newaxis]

        return image_warped

class AffineCv2(Augmenter):
    """
    Augmenter to apply affine transformations to images using cv2 (i.e. opencv)
    backend.

    This is mostly a wrapper around skimage's AffineTransform class and
    warp function.

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

    Parameters
    ----------
    scale : float or tuple of two floats or StochasticParameter or dict {"x": float/tuple/StochasticParameter, "y": float/tuple/StochasticParameter}, optional(default=1.0)
        Scaling factor to use, where 1.0 represents no change and 0.5 is
        zoomed out to 50 percent of the original size.
            * If a single float, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled from the range
              a <= x <= b per image. That value will be used identically for
              both x- and y-axis.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`scale`). Using a dictionary allows to
              set different values for the axis. If they are set to the same
              ranges, different values may still be sampled per axis.

    translate_percent : float or tuple of two floats or StochasticParameter or dict {"x": float/tuple/StochasticParameter, "y": float/tuple/StochasticParameter}, optional(default=1.0)
        Translation in percent relative to the image
        height/width (x-translation, y-translation) to use,
        where 0 represents no change and 0.5 is half of the image
        height/width.
            * If a single float, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled from the range
              a <= x <= b per image. That percent value will be used identically
              for both x- and y-axis.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`translate_percent`).
              Using a dictionary allows to set different values for the axis.
              If they are set to the same ranges, different values may still
              be sampled per axis.

    translate_px : int or tuple of two ints or StochasticParameter or dict {"x": int/tuple/StochasticParameter, "y": int/tuple/StochasticParameter}, optional(default=1.0)
        Translation in
        pixels.
            * If a single int, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled from the discrete
              range [a .. b] per image. That number will be used identically
              for both x- and y-axis.
            * If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`translate_px`).
              Using a dictionary allows to set different values for the axis.
              If they are set to the same ranges, different values may still
              be sampled per axis.

    rotate : float or int or tuple of two floats/ints or StochasticParameter, optional(default=0)
        Rotation in degrees (NOT radians), i.e. expected value range is
        0 to 360 for positive rotations (may also be negative).
            * If a float/int, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled per image from the
              range a <= x <= b and be used as the rotation value.
            * If a StochasticParameter, then this parameter will be used to
              sample the rotation value per image.

    shear : float or int or tuple of two floats/ints or StochasticParameter, optional(default=0)
        Shear in degrees (NOT radians), i.e. expected value range is
        0 to 360 for positive shear (may also be negative).
            * If a float/int, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled per image from the
              range a <= x <= b and be used as the rotation value.
            * If a StochasticParameter, then this parameter will be used to
              sample the shear value per image.

    order : int or iterable of int or string or iterable of string or ia.ALL or StochasticParameter, optional(default=1)
        Interpolation order to use. Allowed are:
            * cv2.INTER_NEAREST - a nearest-neighbor interpolation
            * cv2.INTER_LINEAR - a bilinear interpolation (used by default)
            * cv2.INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
            * cv2.INTER_LANCZOS4
            * "nearest"
            * "linear"
            * "cubic",
            * "lanczos4"
        The first four are OpenCV constants, the other four are strings that
        are automatically replaced by the OpenCV constants.
        INTER_NEAREST (nearest neighbour interpolation) and INTER_NEAREST
        (linear interpolation) are the fastest.
            * If a single int, then that order will be used for all images.
            * If a string, then it must be one of: "nearest", "linear", "cubic",
              "lanczos4".
            * If an iterable of int/string, then for each image a random value
              will be sampled from that iterable (i.e. list of allowed order
              values).
            * If ia.ALL, then equivalant to list [cv2.INTER_NEAREST,
              cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4].
            * If StochasticParameter, then that parameter is queried per image
              to sample the order value to use.

    cval : number or tuple of two number or ia.ALL or StochasticParameter, optional(default=0)
        The constant value used for skimage's transform function.
        This is the value used to fill up pixels in the result image that
        didn't exist in the input image (e.g. when translating to the left,
        some new pixels are created at the right). Such a fill-up with a
        constant value only happens, when `mode` is "constant".
        For standard uint8 images (value range 0-255), this value may also
        come from the range 0-255. It may be a float value, even for
        integer image dtypes.
            * If this is a single int or float, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple (a, b), then a random value from the range a <= x <= b
              is picked per image.
            * If ia.ALL, a value from the discrete range [0 .. 255] will be
              sampled per image.
            * If a StochasticParameter, a new value will be sampled from the
              parameter per image.

    mode : int or string or list of string or list of ints or ia.ALL or StochasticParameter, optional(default="constant")
        Parameter that defines the handling of newly created pixels.
        Same meaning as in opencv's border mode. Let `abcdefgh` be an image
        content and `|` be an image boundary, then:
            * `cv2.BORDER_REPLICATE`: `aaaaaa|abcdefgh|hhhhhhh`
            * `cv2.BORDER_REFLECT`: `fedcba|abcdefgh|hgfedcb`
            * `cv2.BORDER_REFLECT_101`: `gfedcb|abcdefgh|gfedcba`
            * `cv2.BORDER_WRAP`: `cdefgh|abcdefgh|abcdefg`
            * `cv2.BORDER_CONSTANT`: `iiiiii|abcdefgh|iiiiiii`, where `i` is
              the defined cval.
            * "replicate": Same as cv2.BORDER_REPLICATE.
            * "reflect": Same as cv2.BORDER_REFLECT.
            * "reflect_101": Same as cv2.BORDER_REFLECT_101.
            * "wrap": Same as cv2.BORDER_WRAP.
            * "constant": Same as cv2.BORDER_CONSTANT.
        The datatype of the parameter may
        be:
            * If a single int, then it must be one of `cv2.BORDER_*`.
            * If a single string, then it must be one of: "replicate",
              "reflect", "reflect_101", "wrap", "constant".
            * If a list of ints/strings, then per image a random mode will be
              picked from that list.
            * If ia.ALL, then a random mode from all possible modes will be
              picked.
            * If StochasticParameter, then the mode will be sampled from that
              parameter per image, i.e. it must return only the above mentioned
              strings.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

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
            self.order = Choice(available_orders)
        elif ia.is_single_integer(order):
            ia.do_assert(order in available_orders, "Expected order's integer value to be in %s, got %d." % (str(available_orders), order))
            self.order = Deterministic(order)
        elif ia.is_string(order):
            ia.do_assert(order in available_orders_str, "Expected order to be in %s, got %s." % (str(available_orders_str), order))
            self.order = Deterministic(order)
        elif isinstance(order, list):
            ia.do_assert(all([ia.is_single_integer(val) or ia.is_string(val) for val in order]), "Expected order list to only contain integers/strings, got types %s." % (str([type(val) for val in order]),))
            ia.do_assert(all([val in available_orders + available_orders_str for val in order]), "Expected all order values to be in %s, got %s." % (available_orders + available_orders_str, str(order),))
            self.order = Choice(order)
        elif isinstance(order, StochasticParameter):
            self.order = order
        else:
            raise Exception("Expected order to be imgaug.ALL, int, string, a list of int/string or StochasticParameter, got %s." % (type(order),))

        if cval == ia.ALL:
            self.cval = DiscreteUniform(0, 255)
        elif ia.is_single_number(cval):
            self.cval = Deterministic(cval)
        elif ia.is_iterable(cval):
            ia.do_assert(len(cval) == 2)
            ia.do_assert(0 <= cval[0] <= 255)
            ia.do_assert(0 <= cval[1] <= 255)
            self.cval = DiscreteUniform(cval[0], cval[1])
        elif isinstance(cval, StochasticParameter):
            self.cval = cval
        else:
            raise Exception("Expected cval to be imgaug.ALL, int, float or StochasticParameter, got %s." % (type(cval),))

        available_modes = [cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101, cv2.BORDER_WRAP, cv2.BORDER_CONSTANT]
        available_modes_str = ["replicate", "reflect", "reflect_101", "wrap", "constant"]
        if mode == ia.ALL:
            self.mode = Choice(available_modes)
        elif ia.is_single_integer(mode):
            ia.do_assert(mode in available_modes, "Expected mode to be in %s, got %d." % (str(available_modes), mode))
            self.mode = Deterministic(mode)
        elif ia.is_string(mode):
            ia.do_assert(mode in available_modes_str, "Expected mode to be in %s, got %s." % (str(available_modes_str), mode))
            self.mode = Deterministic(mode)
        elif isinstance(mode, list):
            ia.do_assert(all([ia.is_single_integer(val) or ia.is_string(val) for val in mode]), "Expected mode list to only contain integers/strings, got types %s." % (str([type(val) for val in mode]),))
            ia.do_assert(all([val in available_modes + available_modes_str for val in mode]), "Expected all mode values to be in %s, got %s." % (str(available_modes + available_modes_str), str(mode)))
            self.mode = Choice(mode)
        elif isinstance(mode, StochasticParameter):
            self.mode = mode
        else:
            raise Exception("Expected mode to be imgaug.ALL, an int, a string, a list of int/strings or StochasticParameter, got %s." % (type(mode),))

        # scale
        # float | (float, float) | [float, float] | StochasticParameter
        def scale_handle_param(param, allow_dict):
            if isinstance(param, StochasticParameter):
                return param
            elif ia.is_single_number(param):
                ia.do_assert(param > 0.0, "Expected scale to have range (0, inf), got value %.4f. Note: The value to _not_ change the scale of images is 1.0, not 0.0." % (param,))
                return Deterministic(param)
            elif ia.is_iterable(param) and not isinstance(param, dict):
                ia.do_assert(len(param) == 2, "Expected scale tuple/list with 2 entries, got %d entries." % (len(param),))
                ia.do_assert(param[0] > 0.0 and param[1] > 0.0, "Expected scale tuple/list to have values in range (0, inf), got values %.4f and %.4f. Note: The value to _not_ change the scale of images is 1.0, not 0.0." % (param[0], param[1]))
                return Uniform(param[0], param[1])
            elif allow_dict and isinstance(param, dict):
                ia.do_assert("x" in param or "y" in param)
                x = param.get("x")
                y = param.get("y")

                x = x if x is not None else 1.0
                y = y if y is not None else 1.0

                return (scale_handle_param(x, False), scale_handle_param(y, False))
            else:
                raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(param),))
        self.scale = scale_handle_param(scale, True)

        # translate
        if translate_percent is None and translate_px is None:
            translate_px = 0

        ia.do_assert(translate_percent is None or translate_px is None)

        if translate_percent is not None:
            # translate by percent
            def translate_handle_param(param, allow_dict):
                if ia.is_single_number(param):
                    return Deterministic(float(param))
                elif ia.is_iterable(param) and not isinstance(param, dict):
                    ia.do_assert(len(param) == 2, "Expected translate_percent tuple/list with 2 entries, got %d entries." % (len(param),))
                    all_numbers = all([ia.is_single_number(p) for p in param])
                    ia.do_assert(all_numbers, "Expected translate_percent tuple/list to contain only numbers, got types %s." % (str([type(p) for p in param]),))
                    #ia.do_assert(param[0] > 0.0 and param[1] > 0.0, "Expected translate_percent tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (param[0], param[1]))
                    return Uniform(param[0], param[1])
                elif allow_dict and isinstance(param, dict):
                    ia.do_assert("x" in param or "y" in param)
                    x = param.get("x")
                    y = param.get("y")

                    x = x if x is not None else 0
                    y = y if y is not None else 0

                    return (translate_handle_param(x, False), translate_handle_param(y, False))
                elif isinstance(param, StochasticParameter):
                    return param
                else:
                    raise Exception("Expected float, int or tuple/list with 2 entries of both floats or ints or StochasticParameter. Got %s." % (type(param),))
            self.translate = translate_handle_param(translate_percent, True)
        else:
            # translate by pixels
            def translate_handle_param(param, allow_dict):
                if ia.is_single_integer(param):
                    return Deterministic(param)
                elif ia.is_iterable(param) and not isinstance(param, dict):
                    ia.do_assert(len(param) == 2, "Expected translate_px tuple/list with 2 entries, got %d entries." % (len(param),))
                    all_integer = all([ia.is_single_integer(p) for p in param])
                    ia.do_assert(all_integer, "Expected translate_px tuple/list to contain only integers, got types %s." % (str([type(p) for p in param]),))
                    return DiscreteUniform(param[0], param[1])
                elif allow_dict and isinstance(param, dict):
                    ia.do_assert("x" in param or "y" in param)
                    x = param.get("x")
                    y = param.get("y")

                    x = x if x is not None else 0
                    y = y if y is not None else 0

                    return (translate_handle_param(x, False), translate_handle_param(y, False))
                elif isinstance(param, StochasticParameter):
                    return param
                else:
                    raise Exception("Expected int or tuple/list with 2 ints or StochasticParameter. Got %s." % (type(param),))
            self.translate = translate_handle_param(translate_px, True)

        # rotate
        # StochasticParameter | float | int | (float or int, float or int) | [float or int, float or int]
        if isinstance(rotate, StochasticParameter):
            self.rotate = rotate
        elif ia.is_single_number(rotate):
            self.rotate = Deterministic(rotate)
        elif ia.is_iterable(rotate):
            ia.do_assert(len(rotate) == 2, "Expected rotate tuple/list with 2 entries, got %d entries." % (len(rotate),))
            ia.do_assert(all([ia.is_single_number(val) for val in rotate]), "Expected floats/ints in rotate tuple/list")
            self.rotate = Uniform(rotate[0], rotate[1])
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(rotate),))

        # shear
        # StochasticParameter | float | int | (float or int, float or int) | [float or int, float or int]
        if isinstance(shear, StochasticParameter):
            self.shear = shear
        elif ia.is_single_number(shear):
            self.shear = Deterministic(shear)
        elif ia.is_iterable(shear):
            ia.do_assert(len(shear) == 2, "Expected rotate tuple/list with 2 entries, got %d entries." % (len(shear),))
            ia.do_assert(all([ia.is_single_number(val) for val in shear]), "Expected floats/ints in shear tuple/list.")
            self.shear = Uniform(shear[0], shear[1])
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(shear),))

    def _augment_images(self, images, random_state, parents, hooks):
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

        #images = images if isinstance(images, list) else [images]
        nb_images = len(images)
        #result = [None] * nb_images
        result = images

        scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, mode_samples, order_samples = self._draw_samples(nb_images, random_state)

        for i in sm.xrange(nb_images):
            height, width = images[i].shape[0], images[i].shape[1]
            shift_x = width / 2.0 - 0.5
            shift_y = height / 2.0 - 0.5
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x, translate_y = translate_samples[0][i], translate_samples[1][i]
            #ia.do_assert(isinstance(translate_x, (float, int)))
            #ia.do_assert(isinstance(translate_y, (float, int)))
            if ia.is_single_float(translate_y):
                translate_y_px = int(round(translate_y * images[i].shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(round(translate_x * images[i].shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            cval = cval_samples[i]
            #if ia.is_single_number(cval) or (ia.is_np_array(cval) and cval.shape == (1,)):
            #    cval = [cval, cval, cval]
            mode = mode_samples[i]
            order = order_samples[i]

            mode = mode if ia.is_single_integer(mode) else mode_str_to_int[mode]
            order = order if ia.is_single_integer(order) else order_str_to_int[order]

            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 or shear != 0:
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
                    #np.zeros((2, 3)),
                    dsize=(width, height),
                    flags=order,
                    borderMode=mode,
                    borderValue=cval
                )

                # warp changes uint8 to float64, making this necessary
                #if image_warped.dtype != images[i].dtype:
                #    image_warped = image_warped.astype(images[i].dtype, copy=False)
                result[i] = image_warped
            else:
                result[i] = images[i]

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        scale_samples, translate_samples, rotate_samples, shear_samples, _cval_samples, _mode_samples, _order_samples = self._draw_samples(nb_images, random_state)

        for i, keypoints_on_image in enumerate(keypoints_on_images):
            height, width = keypoints_on_image.height, keypoints_on_image.width
            shift_x = width / 2.0 - 0.5
            shift_y = height / 2.0 - 0.5
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x, translate_y = translate_samples[0][i], translate_samples[1][i]
            #ia.do_assert(isinstance(translate_x, (float, int)))
            #ia.do_assert(isinstance(translate_y, (float, int)))
            if ia.is_single_float(translate_y):
                translate_y_px = int(round(translate_y * keypoints_on_image.shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(round(translate_x * keypoints_on_image.shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            #cval = cval_samples[i]
            #mode = mode_samples[i]
            #order = order_samples[i]
            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 or shear != 0:
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
                #print("coords", coords)
                #print("matrix", matrix.params)
                coords_aug = tf.matrix_transform(coords, matrix.params)
                #print("coords before", coords)
                #print("coordsa ftre", coords_aug, np.around(coords_aug).astype(np.int32))
                result.append(ia.KeypointsOnImage.from_coords_array(coords_aug, shape=keypoints_on_image.shape))
            else:
                result.append(keypoints_on_image)
        return result

    def get_parameters(self):
        return [self.scale, self.translate, self.rotate, self.shear]

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

        return scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, mode_samples, order_samples

class PiecewiseAffine(Augmenter):
    """
    Augmenter that places a regular grid of points on an image and randomly
    moves the neighbourhood of these point around via affine transformations.
    This leads to local distortions.

    This is mostly a wrapper around scikit-image's PiecewiseAffine.
    See also the Affine augmenter for a similar technique.

    Parameters
    ----------
    scale : float or tuple of two floats or StochasticParameter, optional(default=0)
        Each point on the regular grid is moved around via a normal
        distribution. This scale factor is equivalent to the normal
        distribution's sigma. Note that the jitter (how far each point is
        moved in which direction) is multiplied by the height/width of the
        image, so this scale can be the same for different sized images.
        Recommended values are in the range 0.01 to 0.05 (weak to strong
        augmentations).
            * If a single float, then that value will always be used as the
              scale.
            * If a tuple (a, b) of floats, then a random value will be picked
              from the interval (a, b) (per image).
            * If a StochasticParameter, then that parameter will be queried to
              draw one value per image.

    nb_rows : int or tuple of ints or StochasticParameter, optional(default=4)
        Number of rows of points that the regular grid should have.
        Must be at least 2. For large images, you might want to pick a
        higher value than 4. You might have to then adjust scale to lower
        values.
            * If a single int, then that value will always be used as the
              number of rows.
            * If a tuple (a, b), then a value from the discrete interval [a..b]
              will be sampled per image.
            * If a StochasticParameter, then that parameter will be queried to
              draw one value per image.

    nb_cols : int or tuple of ints or StochasticParameter, optional(default=4)
        Number of columns. See `nb_rows`.

    order : int or iterable of int or ia.ALL or StochasticParameter, optional(default=1)
        See Affine.__init__().

    cval : int or float or tuple of two floats or ia.ALL or StochasticParameter, optional(default=0)
        See Affine.__init__().

    mode : string or list of string or ia.ALL or StochasticParameter, optional(default="constant")
        See Affine.__init__().

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))

    Puts a grid of points on each image and then randomly moves each point
    around by 1 to 5 percent (with respect to the image height/width). Pixels
    between these points will be moved accordingly.

    >>> aug = iaa.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=8, nb_cols=8)

    Same as the previous example, but uses a denser grid of 8x8 points (default
    is 4x4). This can be useful for large images.

    """

    def __init__(self, scale=0, nb_rows=4, nb_cols=4, order=1, cval=0, mode="constant",
                 name=None, deterministic=False, random_state=None):
        super(PiecewiseAffine, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(scale):
            self.scale = Deterministic(scale)
        elif ia.is_iterable(scale):
            ia.do_assert(len(scale) == 2, "Expected tuple/list with 2 entries for argument 'scale', got %d entries." % (len(scale),))
            self.scale = Uniform(scale[0], scale[1])
        elif isinstance(scale, StochasticParameter):
            self.scale = scale
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter for argument 'scale'. Got %s." % (type(scale),))

        self.jitter = Normal(loc=0, scale=self.scale)

        if ia.is_single_number(nb_rows):
            ia.do_assert(nb_rows >= 2)
            self.nb_rows = Deterministic(int(nb_rows))
        elif ia.is_iterable(nb_rows):
            ia.do_assert(len(nb_rows) == 2, "Expected tuple/list with 2 entries for argument 'nb_rows', got %d entries." % (len(nb_rows),))
            ia.do_assert(nb_rows[0] >= 2)
            ia.do_assert(nb_rows[1] >= 2)
            self.nb_rows = DiscreteUniform(nb_rows[0], nb_rows[1])
        elif isinstance(nb_rows, StochasticParameter):
            self.nb_rows = nb_rows
        else:
            raise Exception("Expected int, tuple of two ints or StochasticParameter as nb_rows, got %s." % (type(nb_rows),))

        if ia.is_single_number(nb_cols):
            ia.do_assert(nb_cols >= 2)
            self.nb_cols = Deterministic(int(nb_cols))
        elif ia.is_iterable(nb_cols):
            ia.do_assert(len(nb_cols) == 2, "Expected tuple/list with 2 entries for argument 'nb_cols', got %d entries." % (len(nb_cols),))
            ia.do_assert(nb_cols[0] >= 2)
            ia.do_assert(nb_cols[1] >= 2)
            self.nb_cols = DiscreteUniform(nb_cols[0], nb_cols[1])
        elif isinstance(nb_cols, StochasticParameter):
            self.nb_cols = nb_cols
        else:
            raise Exception("Expected int, tuple of two ints or StochasticParameter as nb_cols, got %s." % (type(nb_cols),))

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
            # self.order = DiscreteUniform(0, 5)
            self.order = Choice([0, 1, 3, 4, 5]) # dont use order=2 (bi-quadratic) because that is apparently currently not recommended (and throws a warning)
        elif ia.is_single_integer(order):
            ia.do_assert(0 <= order <= 5, "Expected order's integer value to be in range 0 <= x <= 5, got %d." % (order,))
            self.order = Deterministic(order)
        elif isinstance(order, list):
            ia.do_assert(all([ia.is_single_integer(val) for val in order]), "Expected order list to only contain integers, got types %s." % (str([type(val) for val in order]),))
            ia.do_assert(all([0 <= val <= 5 for val in order]), "Expected all of order's integer values to be in range 0 <= x <= 5, got %s." % (str(order),))
            self.order = Choice(order)
        elif isinstance(order, StochasticParameter):
            self.order = order
        else:
            raise Exception("Expected order to be imgaug.ALL, int or StochasticParameter, got %s." % (type(order),))

        if cval == ia.ALL:
            self.cval = DiscreteUniform(0, 255)
        elif ia.is_single_number(cval):
            self.cval = Deterministic(cval)
        elif ia.is_iterable(cval):
            ia.do_assert(len(cval) == 2)
            ia.do_assert(0 <= cval[0] <= 255)
            ia.do_assert(0 <= cval[1] <= 255)
            self.cval = Uniform(cval[0], cval[1])
        elif isinstance(cval, StochasticParameter):
            self.cval = cval
        else:
            raise Exception("Expected cval to be imgaug.ALL, int, float or StochasticParameter, got %s." % (type(cval),))

        # constant, edge, symmetric, reflect, wrap
        if mode == ia.ALL:
            self.mode = Choice(["constant", "edge", "symmetric", "reflect", "wrap"])
        elif ia.is_string(mode):
            self.mode = Deterministic(mode)
        elif isinstance(mode, list):
            ia.do_assert(all([ia.is_string(val) for val in mode]))
            self.mode = Choice(mode)
        elif isinstance(mode, StochasticParameter):
            self.mode = mode
        else:
            raise Exception("Expected mode to be imgaug.ALL, a string, a list of strings or StochasticParameter, got %s." % (type(mode),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)

        seeds = ia.copy_random_state(random_state).randint(0, 10**6, (nb_images+1,))

        seed = seeds[-1]
        nb_rows_samples = self.nb_rows.draw_samples((nb_images,), random_state=ia.new_random_state(seed + 1))
        nb_cols_samples = self.nb_cols.draw_samples((nb_images,), random_state=ia.new_random_state(seed + 2))
        cval_samples = self.cval.draw_samples((nb_images,), random_state=ia.new_random_state(seed + 3))
        mode_samples = self.mode.draw_samples((nb_images,), random_state=ia.new_random_state(seed + 4))
        order_samples = self.order.draw_samples((nb_images,), random_state=ia.new_random_state(seed + 5))

        for i in sm.xrange(nb_images):
            rs_image = ia.new_random_state(seeds[i])
            h, w = images[i].shape[0:2]
            transformer = self._get_transformer(h, w, nb_rows_samples[i], nb_cols_samples[i], rs_image)

            if transformer is not None:
                #print("transformer vertices img", transformer._tesselation.vertices)
                image_warped = tf.warp(
                    images[i],
                    transformer,
                    order=order_samples[i],
                    mode=mode_samples[i],
                    cval=cval_samples[i],
                    preserve_range=True,
                    output_shape=images[i].shape
                )

                # warp changes uint8 to float64, making this necessary
                if image_warped.dtype != images[i].dtype:
                    image_warped = image_warped.astype(images[i].dtype, copy=False)

                result[i] = image_warped

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)

        seeds = ia.copy_random_state(random_state).randint(0, 10**6, (nb_images+1,))
        seed = seeds[-1]
        nb_rows_samples = self.nb_rows.draw_samples((nb_images,), random_state=ia.new_random_state(seed + 1))
        nb_cols_samples = self.nb_cols.draw_samples((nb_images,), random_state=ia.new_random_state(seed + 2))

        for i in sm.xrange(nb_images):
            rs_image = ia.new_random_state(seeds[i])
            kpsoi = keypoints_on_images[i]
            h, w = kpsoi.shape[0:2]
            transformer = self._get_transformer(h, w, nb_rows_samples[i], nb_cols_samples[i], rs_image)

            if transformer is None or len(kpsoi.keypoints) == 0:
                result.append(kpsoi)
            else:
                #print("transformer vertices kp", transformer._tesselation.vertices)

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


                # Image based augmentation routine. Draws the keypoints on
                # the image plane (black and white, only keypoint marked),
                # then augments these images, then searches for the new
                # (visual) location of the keypoints.
                # Much slower than directly augmenting the coordinates, but
                # here the only method that reliably works.
                kp_image = kpsoi.to_keypoint_image(size=3) # size=1 sometimes leads to dropped/lost keypoints
                kp_image_warped = tf.warp(
                    kp_image,
                    transformer,
                    order=1,
                    preserve_range=True,
                    output_shape=(kpsoi.shape[0], kpsoi.shape[1], len(kpsoi.keypoints))
                )

                kps_aug = ia.KeypointsOnImage.from_keypoint_image(
                    kp_image_warped,
                    if_not_found_coords={"x": -1, "y": -1},
                    nb_channels=None if len(kpsoi.shape) < 3 else kpsoi.shape[2]
                )
                # TODO is this still necessary after nb_channels was added to
                # from_keypoint_image() ?
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
        #cell_height = h / self.rows
        #cell_width = w / self.cols
        #cell_height_h = cell_height / 2
        #cell_width_h = cell_width / 2

        # get coords on y and x axis of points to move around
        # these coordinates are supposed to be at the centers of each cell
        # (otherwise the first coordinate would be at (0, 0) and could hardly
        # be moved around before leaving the image),
        # so we use here (half cell height/width to H/W minus half height/width)
        # instead of (0, H/W)
        #y = np.linspace(cell_height_h, h - cell_height_h, self.rows)
        #x = np.linspace(cell_width_h, w - cell_width_h, self.cols)

        nb_rows = max(nb_rows, 2)
        nb_cols = max(nb_cols, 2)

        y = np.linspace(0, h, nb_rows)
        x = np.linspace(0, w, nb_cols)

        xx_src, yy_src = np.meshgrid(x, y) # (H, W) and (H, W) for H=rows, W=cols
        points_src = np.dstack([yy_src.flat, xx_src.flat])[0] # (1, HW, 2) => (HW, 2) for H=rows, W=cols
        #print("nb_rows", nb_rows, "nb_cols", nb_cols, "x", x, "y", y, "xx_src", xx_src.shape, "yy_src", yy_src.shape, "points_src", np.dstack([yy_src.flat, xx_src.flat]).shape)

        jitter_img = self.jitter.draw_samples(points_src.shape, random_state=random_state)

        nb_nonzero = len(jitter_img.flatten().nonzero()[0])
        if nb_nonzero == 0:
            return None
        else:
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
            #print("points_src", points_src, "points_dest", points_dest)

            matrix = tf.PiecewiseAffineTransform()
            matrix.estimate(points_src[:, ::-1], points_dest[:, ::-1])
            return matrix

    def get_parameters(self):
        return [self.scale]

class PerspectiveTransform(Augmenter):
    """
    Augmenter that performs a random four point perspective transform.

    Each of the four points is placed on the image using a random distance from
    its respective corner. The distance is sampled from a normal distribution.
    As a result, most transformations don't change very much, while some
    "focus" on polygons far inside the image.

    The results of this augmenter have some similarity with Crop.

    Code partially from http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ .

    Parameters
    ----------
    scale : float or tuple of two floats or StochasticParameter, optional(default=0)
        Standard deviation of the normal distributions. These are used to sample
        the random distances of the subimage's corners from the full image's
        corners. The sampled values reflect percentage values (with respect
        to image height/width). Recommended values are in the range 0.0 to 0.1.
            * If a single float, then that value will always be used as the
              scale.
            * If a tuple (a, b) of floats, then a random value will be picked
              from the interval (a, b) (per image).
            * If a StochasticParameter, then that parameter will be queried to
              draw one value per image.

    keep_size : bool, optional(default=True)
        Whether to resize image's back to their original size after applying
        the perspective transform. If set to False, the resulting images
        may end up having different shapes and will always be a list, never
        an array.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

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

        if ia.is_single_number(scale):
            self.scale = Deterministic(scale)
        elif ia.is_iterable(scale):
            ia.do_assert(len(scale) == 2, "Expected tuple/list with 2 entries for argument 'scale', got %d entries." % (len(scale),))
            self.scale = Uniform(scale[0], scale[1])
        elif isinstance(scale, StochasticParameter):
            self.scale = scale
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter for argument 'scale'. Got %s." % (type(scale),))

        self.jitter = Normal(loc=0, scale=self.scale)

        self.keep_size = keep_size

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        if not self.keep_size:
            result = list(result)

        matrices, max_heights, max_widths = self._create_matrices(
            [image.shape for image in images],
            random_state
        )

        for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
            # cv2.warpPerspective only supports <=4 channels
            #ia.do_assert(images[i].shape[2] <= 4, "PerspectiveTransform is currently limited to images with 4 or less channels.")
            nb_channels = images[i].shape[2]
            if nb_channels <= 4:
                warped = cv2.warpPerspective(images[i], M, (max_width, max_height))
                if warped.ndim == 2 and images[i].ndim == 3:
                    warped = np.expand_dims(warped, 2)
            else:
                # warp each channel on its own, re-add channel axis, then stack
                # the result from a list of [H, W, 1] to (H, W, C).
                warped = [cv2.warpPerspective(images[i][..., c], M, (max_width, max_height)) for c in sm.xrange(nb_channels)]
                warped = [warped_i[..., np.newaxis] for warped_i in warped]
                warped = np.dstack(warped)
            #print(np.min(warped), np.max(warped), warped.dtype)
            if self.keep_size:
                h, w = images[i].shape[0:2]
                warped = ia.imresize_single_image(warped, (h, w), interpolation="cubic")
            result[i] = warped

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = keypoints_on_images
        matrices, max_heights, max_widths = self._create_matrices(
            [kps.shape for kps in keypoints_on_images],
            random_state
        )

        for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
            keypoints_on_image = keypoints_on_images[i]
            kps_arr = keypoints_on_image.get_coords_array()
            #nb_channels = keypoints_on_image.shape[2] if len(keypoints_on_image.shape) >= 3 else None

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
            points[0, 1] = 1.0 - points[0, 1] # h = 1.0 - jitter

            # top right
            points[1, 0] = 1.0 - points[1, 0] # w = 1.0 - jitter
            points[1, 1] = 1.0 - points[1, 1] # h = 1.0 - jitter

            # bottom right
            points[2, 0] = 1.0 - points[2, 0] # h = 1.0 - jitter

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
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            # compute the height of the new image, which will be the
            # maximum distance between the top-right and bottom-right
            # y-coordinates or the top-left and bottom-left y-coordinates
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            # now that we have the dimensions of the new image, construct
            # the set of destination points to obtain a "birds eye view",
            # (i.e. top-down view) of the image, again specifying points
            # in the top-left, top-right, bottom-right, and bottom-left
            # order
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            # compute the perspective transform matrix and then apply it
            M = cv2.getPerspectiveTransform(points, dst)
            matrices.append(M)
            max_heights.append(maxHeight)
            max_widths.append(maxWidth)

        return matrices, max_heights, max_widths

    def _order_points(self, pts):
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
        return [self.scale]

# code partially from
# https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
class ElasticTransformation(Augmenter):
    """
    Augmenter to transform images by moving pixels locally around using
    displacement fields.

    See
        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003
    for a detailed explanation.

    Parameters
    ----------
    alpha : float or tuple of two floats or StochasticParameter, optional(default=0)
        Strength of the distortion field. Higher values mean more "movement" of
        pixels.
            * If float, then that value will be used for all images.
            * If tuple (a, b), then a random value from range a <= x <= b will be
              sampled per image.
            * If StochasticParameter, then that parameter will be used to sample
              a value per image.

    sigma : float or tuple of two floats or StochasticParameter, optional(default=0)
        Standard deviation of the gaussian kernel used to smooth the distortion
        fields.
            * If float, then that value will be used for all images.
            * If tuple (a, b), then a random value from range a <= x <= b will be
              sampled per image.
            * If StochasticParameter, then that parameter will be used to sample
              a value per image.

    order : int or iterable of int or ia.ALL or StochasticParameter, optional(default=1)
        Interpolation order to use. Same meaning as in
        `scipy.ndimage.map_coordinates` and may take any integer value
        in the range 0 to 5, where orders close to 0 are faster.
            * If a single int, then that order will be used for all images.
            * If an iterable, then for each image a random value will be sampled
              from that iterable (i.e. list of allowed order values).
            * If ia.ALL, then equivalant to list [0, 1, 2, 3, 4, 5].
            * If StochasticParameter, then that parameter is queried per image
              to sample the order value to use.

    cval : number or tuple of two number or ia.ALL or StochasticParameter, optional(default=0)
        The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to "constant".
        For standard uint8 images (value range 0-255), this value may also
        come from the range 0-255. It may be a float value, even for
        integer image dtypes.
            * If this is a single int or float, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple (a, b), then a random value from the range a <= x <= b
              is picked per image.
            * If ia.ALL, a value from the discrete range [0 .. 255] will be
              sampled per image.
            * If a StochasticParameter, a new value will be sampled from the
              parameter per image.

    mode : string or list of string or ia.ALL or StochasticParameter, optional(default="constant")
        Parameter that defines the handling of newly created pixels.
        May take the same values as in `scipy.ndimage.map_coordinates`,
        i.e. "constant", "nearest", "reflect" or "wrap".
        The datatype of the parameter may
        be:
            * If a single string, then that mode will be used for all images.
            * If a list of strings, then per image a random mode will be picked
              from that list.
            * If ia.ALL, then a random mode from all possible modes will be
              picked.
            * If StochasticParameter, then the mode will be sampled from that
              parameter per image, i.e. it must return only the above mentioned
              strings.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)

    apply elastic transformations with a strength/alpha of 0.5 and
    smoothness of 0.25 to all images.


    >>> aug = iaa.ElasticTransformation(alpha=(0.25, 3.0), sigma=0.25)

    apply elastic transformations with a strength/alpha that comes
    from the range 0.25 <= x <= 3.0 (randomly picked per image) and
    smoothness of 0.25.

    """

    def __init__(self, alpha=0, sigma=0, order=3, cval=0, mode="constant",
                 name=None, deterministic=False, random_state=None):
        super(ElasticTransformation, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(alpha):
            ia.do_assert(alpha >= 0.0, "Expected alpha to have range [0, inf), got value %.4f." % (alpha,))
            self.alpha = Deterministic(alpha)
        elif ia.is_iterable(alpha):
            ia.do_assert(len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(alpha),))
            self.alpha = Uniform(alpha[0], alpha[1])
        elif isinstance(alpha, StochasticParameter):
            self.alpha = alpha
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

        if ia.is_single_number(sigma):
            ia.do_assert(sigma >= 0.0, "Expected sigma to have range [0, inf), got value %.4f." % (sigma,))
            self.sigma = Deterministic(sigma)
        elif ia.is_iterable(sigma):
            ia.do_assert(len(sigma) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(sigma),))
            self.sigma = Uniform(sigma[0], sigma[1])
        elif isinstance(sigma, StochasticParameter):
            self.sigma = sigma
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(sigma),))

        if order == ia.ALL:
            self.order = Choice([0, 1, 2, 3, 4, 5])
        elif ia.is_single_integer(order):
            ia.do_assert(0 <= order <= 5, "Expected order's integer value to be in range 0 <= x <= 5, got %d." % (order,))
            self.order = Deterministic(order)
        elif isinstance(order, list):
            ia.do_assert(all([ia.is_single_integer(val) for val in order]), "Expected order list to only contain integers, got types %s." % (str([type(val) for val in order]),))
            ia.do_assert(all([0 <= val <= 5 for val in order]), "Expected all of order's integer values to be in range 0 <= x <= 5, got %s." % (str(order),))
            self.order = Choice(order)
        elif isinstance(order, StochasticParameter):
            self.order = order
        else:
            raise Exception("Expected order to be imgaug.ALL, int, list of int or StochasticParameter, got %s." % (type(order),))

        if cval == ia.ALL:
            self.cval = DiscreteUniform(0, 255)
        elif ia.is_single_number(cval):
            self.cval = Deterministic(cval)
        elif ia.is_iterable(cval):
            ia.do_assert(len(cval) == 2)
            ia.do_assert(0 <= cval[0] <= 255)
            ia.do_assert(0 <= cval[1] <= 255)
            self.cval = Uniform(cval[0], cval[1])
        elif isinstance(cval, StochasticParameter):
            self.cval = cval
        else:
            raise Exception("Expected cval to be imgaug.ALL, int, float or StochasticParameter, got %s." % (type(cval),))

        if mode == ia.ALL:
            self.mode = Choice(["constant", "nearest", "reflect", "wrap"])
        elif ia.is_string(mode):
            self.mode = Deterministic(mode)
        elif isinstance(mode, list):
            ia.do_assert(all([ia.is_string(val) for val in mode]))
            self.mode = Choice(mode)
        elif isinstance(mode, StochasticParameter):
            self.mode = mode
        else:
            raise Exception("Expected mode to be imgaug.ALL, a string, a list of strings or StochasticParameter, got %s." % (type(mode),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = ia.copy_random_state(random_state).randint(0, 10**6, (nb_images+1,))
        alphas = self.alpha.draw_samples((nb_images,), random_state=ia.new_random_state(seeds[-1]+10000))
        sigmas = self.sigma.draw_samples((nb_images,), random_state=ia.new_random_state(seeds[-1]+10100))
        orders = self.order.draw_samples((nb_images,), random_state=ia.new_random_state(seeds[-1]+10200))
        cvals = self.cval.draw_samples((nb_images,), random_state=ia.new_random_state(seeds[-1]+10300))
        modes = self.mode.draw_samples((nb_images,), random_state=ia.new_random_state(seeds[-1]+10400))
        for i in sm.xrange(nb_images):
            image = images[i]
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = ElasticTransformation.generate_indices(image_first_channel.shape, alpha=alphas[i], sigma=sigmas[i], random_state=ia.new_random_state(seeds[i]))
            result[i] = ElasticTransformation.map_coordinates(
                images[i],
                indices_x,
                indices_y,
                order=orders[i],
                cval=cvals[i],
                mode=modes[i]
            )
        return result

    """
    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        # TODO do keypoints even have to be augmented for elastic transformations?
        # TODO this transforms keypoints to images, augments the images, then transforms
        # back to keypoints - inefficient and keypoints that get outside of the images
        # cannot be recovered
        result = []
        nb_images = len(keypoints_on_images)
        seeds = ia.copy_random_state(random_state).randint(0, 10**6, (nb_images,))
        alphas = self.alpha.draw_samples((nb_images,), random_state=ia.copy_random_state(random_state))
        sigmas = self.sigma.draw_samples((nb_images,), random_state=ia.copy_random_state(random_state))
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            indices_x, indices_y = ElasticTransformation.generate_indices(keypoints_on_image.shape[0:2], alpha=alphas[i], sigma=sigmas[i], random_state=ia.new_random_state(seeds[i]))
            keypoint_image = keypoints_on_image.to_keypoint_image()
            keypoint_image_aug = ElasticTransformation.map_coordinates(keypoint_image, indices_x, indices_y)
            keypoints_aug = ia.KeypointsOnImage.from_keypoint_image(keypoint_image_aug)
            result.append(keypoints_aug)
        return result
    """

    # no transformation of keypoints for this currently,
    # it seems like this is the more appropriate choice overall for this augmentation
    # technique
    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.alpha, self.sigma, self.orders, self.cvals, self.modes]

    @staticmethod
    def generate_indices(shape, alpha, sigma, random_state):
        ia.do_assert(len(shape) == 2)

        dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    @staticmethod
    def map_coordinates(image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        ia.do_assert(len(image.shape) == 3)
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in sm.xrange(image.shape[2]):
            remapped_flat = ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result
