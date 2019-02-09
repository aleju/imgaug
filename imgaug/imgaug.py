from __future__ import print_function, division, absolute_import

import math
import copy
import numbers
import sys
import os
import json
import types
import warnings

import numpy as np
import cv2
import imageio
import scipy.spatial.distance
import six
import six.moves as sm
import skimage.draw
import skimage.measure
import collections
from PIL import Image as PIL_Image, ImageDraw as PIL_ImageDraw, ImageFont as PIL_ImageFont

ALL = "ALL"

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# filepath to the quokka image, its annotations and depth map
QUOKKA_FP = os.path.join(FILE_DIR, "quokka.jpg")
QUOKKA_ANNOTATIONS_FP = os.path.join(FILE_DIR, "quokka_annotations.json")
QUOKKA_DEPTH_MAP_HALFRES_FP = os.path.join(FILE_DIR, "quokka_depth_map_halfres.png")

DEFAULT_FONT_FP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "DejaVuSans.ttf"
)

# We instantiate a current/global random state here once.
# One can also call np.random, but that is (in contrast to np.random.RandomState)
# a module and hence cannot be copied via deepcopy. That's why we use RandomState
# here (and in all augmenters) instead of np.random.
CURRENT_RANDOM_STATE = np.random.RandomState(42)
SEED_MIN_VALUE = 0
SEED_MAX_VALUE = 2**31-1  # use 2**31 instead of 2**32 here because 2**31 errored on some systems


# to check if a dtype instance is among these dtypes, use e.g. `dtype.type in NP_FLOAT_TYPES`
# do not just use `dtype in NP_FLOAT_TYPES` as that would fail
NP_FLOAT_TYPES = set(np.sctypes["float"])
NP_INT_TYPES = set(np.sctypes["int"])
NP_UINT_TYPES = set(np.sctypes["uint"])

IMSHOW_BACKEND_DEFAULT = "matplotlib"

IMRESIZE_VALID_INTERPOLATIONS = ["nearest", "linear", "area", "cubic",
                                 cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]


def is_np_array(val):
    """
    Checks whether a variable is a numpy array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    out : bool
        True if the variable is a numpy array. Otherwise False.

    """
    # using np.generic here via isinstance(val, (np.ndarray, np.generic)) seems to also fire for scalar numpy values
    # even though those are not arrays
    return isinstance(val, np.ndarray)


def is_single_integer(val):
    """
    Checks whether a variable is an integer.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is an integer. Otherwise False.

    """
    return isinstance(val, numbers.Integral) and not isinstance(val, bool)


def is_single_float(val):
    """
    Checks whether a variable is a float.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a float. Otherwise False.

    """
    return isinstance(val, numbers.Real) and not is_single_integer(val) and not isinstance(val, bool)


def is_single_number(val):
    """
    Checks whether a variable is a number, i.e. an integer or float.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a number. Otherwise False.

    """
    return is_single_integer(val) or is_single_float(val)


def is_iterable(val):
    """
    Checks whether a variable is iterable.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is an iterable. Otherwise False.

    """
    return isinstance(val, collections.Iterable)


# TODO convert to is_single_string() or rename is_single_integer/float/number()
def is_string(val):
    """
    Checks whether a variable is a string.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a string. Otherwise False.

    """
    return isinstance(val, six.string_types)


def is_single_bool(val):
    """
    Checks whether a variable is a boolean.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a boolean. Otherwise False.

    """
    return type(val) == type(True)


def is_integer_array(val):
    """
    Checks whether a variable is a numpy integer array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a numpy integer array. Otherwise False.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.integer)


def is_float_array(val):
    """
    Checks whether a variable is a numpy float array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a numpy float array. Otherwise False.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.floating)


def is_callable(val):
    """
    Checks whether a variable is a callable, e.g. a function.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a callable. Otherwise False.

    """
    # python 3.x with x <= 2 does not support callable(), apparently
    if sys.version_info[0] == 3 and sys.version_info[1] <= 2:
        return hasattr(val, '__call__')
    else:
        return callable(val)


def is_generator(val):
    """
    Checks whether a variable is a generator.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True is the variable is a generator. Otherwise False.

    """
    return isinstance(val, types.GeneratorType)


def caller_name():
    """
    Returns the name of the caller, e.g. a function.

    Returns
    -------
    str
        The name of the caller as a string

    """
    return sys._getframe(1).f_code.co_name


def seed(seedval):
    """
    Set the seed used by the global random state and thereby all randomness
    in the library.

    This random state is by default by all augmenters. Under special
    circumstances (e.g. when an augmenter is switched to deterministic mode),
    the global random state is replaced by another -- local -- one.
    The replacement is dependent on the global random state.

    Parameters
    ----------
    seedval : int
        The seed to use.

    """
    CURRENT_RANDOM_STATE.seed(seedval)


def current_random_state():
    """
    Returns the current/global random state of the library.

    Returns
    ----------
    numpy.random.RandomState
        The current/global random state.

    """
    return CURRENT_RANDOM_STATE


def new_random_state(seed=None, fully_random=False):
    """
    Returns a new random state.

    Parameters
    ----------
    seed : None or int, optional
        Optional seed value to use.
        The same datatypes are allowed as for ``numpy.random.RandomState(seed)``.

    fully_random : bool, optional
        Whether to use numpy's random initialization for the
        RandomState (used if set to True). If False, a seed is sampled from
        the global random state, which is a bit faster and hence the default.

    Returns
    -------
    numpy.random.RandomState
        The new random state.

    """
    if seed is None:
        if not fully_random:
            # sample manually a seed instead of just RandomState(),
            # because the latter one
            # is way slower.
            seed = CURRENT_RANDOM_STATE.randint(SEED_MIN_VALUE, SEED_MAX_VALUE, 1)[0]
    return np.random.RandomState(seed)


def dummy_random_state():
    """
    Returns a dummy random state that is always based on a seed of 1.

    Returns
    -------
    numpy.random.RandomState
        The new random state.

    """
    return np.random.RandomState(1)


def copy_random_state(random_state, force_copy=False):
    """
    Creates a copy of a random state.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        The random state to copy.

    force_copy : bool, optional
        If True, this function will always create a copy of every random
        state. If False, it will not copy numpy's default random state,
        but all other random states.

    Returns
    -------
    rs_copy : numpy.random.RandomState
        The copied random state.

    """
    if random_state == np.random and not force_copy:
        return random_state
    else:
        rs_copy = dummy_random_state()
        orig_state = random_state.get_state()
        rs_copy.set_state(orig_state)
        return rs_copy


def derive_random_state(random_state):
    """
    Create a new random states based on an existing random state or seed.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        Random state or seed from which to derive the new random state.

    Returns
    -------
    numpy.random.RandomState
        Derived random state.

    """
    return derive_random_states(random_state, n=1)[0]


# TODO use this everywhere instead of manual seed + create
def derive_random_states(random_state, n=1):
    """
    Create N new random states based on an existing random state or seed.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        Random state or seed from which to derive new random states.

    n : int, optional
        Number of random states to derive.

    Returns
    -------
    list of numpy.random.RandomState
        Derived random states.

    """
    seed_ = random_state.randint(SEED_MIN_VALUE, SEED_MAX_VALUE, 1)[0]
    return [new_random_state(seed_+i) for i in sm.xrange(n)]


def forward_random_state(random_state):
    """
    Forward the internal state of a random state.

    This makes sure that future calls to the random_state will produce new random values.

    Parameters
    ----------
    random_state : numpy.random.RandomState
        Random state to forward.

    """
    random_state.uniform()


def _quokka_normalize_extract(extract):
    """
    Generate a normalized rectangle to be extract from the standard quokka image.

    Parameters
    ----------
    extract : 'square' or tuple of number or imgaug.BoundingBox or imgaug.BoundingBoxesOnImage
        Unnormalized representation of the image subarea to be extracted.

            * If string ``square``, then a squared area ``(x: 0 to max 643, y: 0 to max 643)``
              will be extracted from the image.
            * If a tuple, then expected to contain four numbers denoting ``x1``, ``y1``, ``x2``
              and ``y2``.
            * If a BoundingBox, then that bounding box's area will be extracted from the image.
            * If a BoundingBoxesOnImage, then expected to contain exactly one bounding box
              and a shape matching the full image dimensions (i.e. (643, 960, *)). Then the
              one bounding box will be used similar to BoundingBox.

    Returns
    -------
    bb : imgaug.BoundingBox
        Normalized representation of the area to extract from the standard quokka image.

    """
    if extract == "square":
        bb = BoundingBox(x1=0, y1=0, x2=643, y2=643)
    elif isinstance(extract, tuple) and len(extract) == 4:
        bb = BoundingBox(x1=extract[0], y1=extract[1], x2=extract[2], y2=extract[3])
    elif isinstance(extract, BoundingBox):
        bb = extract
    elif isinstance(extract, BoundingBoxesOnImage):
        do_assert(len(extract.bounding_boxes) == 1)
        do_assert(extract.shape[0:2] == (643, 960))
        bb = extract.bounding_boxes[0]
    else:
        raise Exception(
            "Expected 'square' or tuple of four entries or BoundingBox or BoundingBoxesOnImage "
            + "for parameter 'extract', got %s." % (type(extract),)
        )
    return bb


def _compute_resized_shape(from_shape, to_shape):
    """
    Computes the intended new shape of an image-like array after resizing.

    Parameters
    ----------
    from_shape : tuple or ndarray
        Old shape of the array. Usually expected to be a tuple of form ``(H, W)`` or ``(H, W, C)`` or
        alternatively an array with two or three dimensions.

    to_shape : None or tuple of ints or tuple of floats or int or float or ndarray
        New shape of the array.

            * If None, then `from_shape` will be used as the new shape.
            * If an int ``V``, then the new shape will be ``(V, V, [C])``, where ``C`` will be added if it
              is part of `from_shape`.
            * If a float ``V``, then the new shape will be ``(H*V, W*V, [C])``, where ``H`` and ``W`` are the old
              height/width.
            * If a tuple ``(H', W', [C'])`` of ints, then ``H'`` and ``W'`` will be used as the new height
              and width.
            * If a tuple ``(H', W', [C'])`` of floats (except ``C``), then ``H'`` and ``W'`` will
              be used as the new height and width.
            * If a numpy array, then the array's shape will be used.

    Returns
    -------
    to_shape_computed : tuple of int
        New shape.

    """
    if is_np_array(from_shape):
        from_shape = from_shape.shape
    if is_np_array(to_shape):
        to_shape = to_shape.shape

    to_shape_computed = list(from_shape)

    if to_shape is None:
        pass
    elif isinstance(to_shape, tuple):
        do_assert(len(from_shape) in [2, 3])
        do_assert(len(to_shape) in [2, 3])

        if len(from_shape) == 3 and len(to_shape) == 3:
            do_assert(from_shape[2] == to_shape[2])
        elif len(to_shape) == 3:
            to_shape_computed.append(to_shape[2])

        do_assert(all([v is None or is_single_number(v) for v in to_shape[0:2]]),
                  "Expected the first two entries in to_shape to be None or numbers, "
                  + "got types %s." % (str([type(v) for v in to_shape[0:2]]),))

        for i, from_shape_i in enumerate(from_shape[0:2]):
            if to_shape[i] is None:
                to_shape_computed[i] = from_shape_i
            elif is_single_integer(to_shape[i]):
                to_shape_computed[i] = to_shape[i]
            else:  # float
                to_shape_computed[i] = int(np.round(from_shape_i * to_shape[i]))
    elif is_single_integer(to_shape) or is_single_float(to_shape):
        to_shape_computed = _compute_resized_shape(from_shape, (to_shape, to_shape))
    else:
        raise Exception("Expected to_shape to be None or ndarray or tuple of floats or tuple of ints or single int "
                        + "or single float, got %s." % (type(to_shape),))

    return tuple(to_shape_computed)


def quokka(size=None, extract=None):
    """
    Returns an image of a quokka as a numpy array.

    Parameters
    ----------
    size : None or float or tuple of int, optional
        Size of the output image. Input into :func:`imgaug.imgaug.imresize_single_image`.
        Usually expected to be a tuple ``(H, W)``, where ``H`` is the desired height
        and ``W`` is the width. If None, then the image will not be resized.

    extract : None or 'square' or tuple of number or imgaug.BoundingBox or imgaug.BoundingBoxesOnImage
        Subarea of the quokka image to extract:

            * If None, then the whole image will be used.
            * If string ``square``, then a squared area ``(x: 0 to max 643, y: 0 to max 643)`` will
              be extracted from the image.
            * If a tuple, then expected to contain four numbers denoting ``x1``, ``y1``, ``x2``
              and ``y2``.
            * If a BoundingBox, then that bounding box's area will be extracted from the image.
            * If a BoundingBoxesOnImage, then expected to contain exactly one bounding box
              and a shape matching the full image dimensions (i.e. ``(643, 960, *)``). Then the
              one bounding box will be used similar to BoundingBox.

    Returns
    -------
    img : (H,W,3) ndarray
        The image array of dtype uint8.

    """
    img = imageio.imread(QUOKKA_FP, pilmode="RGB")
    if extract is not None:
        bb = _quokka_normalize_extract(extract)
        img = bb.extract_from_image(img)
    if size is not None:
        shape_resized = _compute_resized_shape(img.shape, size)
        img = imresize_single_image(img, shape_resized[0:2])
    return img


def quokka_square(size=None):
    """
    Returns an (square) image of a quokka as a numpy array.

    Parameters
    ----------
    size : None or float or tuple of int, optional
        Size of the output image. Input into :func:`imgaug.imgaug.imresize_single_image`.
        Usually expected to be a tuple ``(H, W)``, where ``H`` is the desired height
        and ``W`` is the width. If None, then the image will not be resized.

    Returns
    -------
    img : (H,W,3) ndarray
        The image array of dtype uint8.

    """
    return quokka(size=size, extract="square")


def quokka_heatmap(size=None, extract=None):
    """
    Returns a heatmap (here: depth map) for the standard example quokka image.

    Parameters
    ----------
    size : None or float or tuple of int, optional
        See :func:`imgaug.quokka`.

    extract : None or 'square' or tuple of number or imgaug.BoundingBox or imgaug.BoundingBoxesOnImage
        See :func:`imgaug.quokka`.

    Returns
    -------
    result : imgaug.HeatmapsOnImage
        Depth map as an heatmap object. Values close to 0.0 denote objects that are close to
        the camera. Values close to 1.0 denote objects that are furthest away (among all shown
        objects).

    """
    img = imageio.imread(QUOKKA_DEPTH_MAP_HALFRES_FP, pilmode="RGB")
    img = imresize_single_image(img, (643, 960), interpolation="cubic")

    if extract is not None:
        bb = _quokka_normalize_extract(extract)
        img = bb.extract_from_image(img)
    if size is None:
        size = img.shape[0:2]

    shape_resized = _compute_resized_shape(img.shape, size)
    img = imresize_single_image(img, shape_resized[0:2])
    img_0to1 = img[..., 0]  # depth map was saved as 3-channel RGB
    img_0to1 = img_0to1.astype(np.float32) / 255.0
    img_0to1 = 1 - img_0to1  # depth map was saved as 0 being furthest away

    return HeatmapsOnImage(img_0to1, shape=img_0to1.shape[0:2] + (3,))


def quokka_segmentation_map(size=None, extract=None):
    """
    Returns a segmentation map for the standard example quokka image.

    Parameters
    ----------
    size : None or float or tuple of int, optional
        See :func:`imgaug.quokka`.

    extract : None or 'square' or tuple of number or imgaug.BoundingBox or imgaug.BoundingBoxesOnImage
        See :func:`imgaug.quokka`.

    Returns
    -------
    result : imgaug.SegmentationMapOnImage
        Segmentation map object.

    """
    with open(QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)

    xx = []
    yy = []
    for kp_dict in json_dict["polygons"][0]["keypoints"]:
        x = kp_dict["x"]
        y = kp_dict["y"]
        xx.append(x)
        yy.append(y)

    img_seg = np.zeros((643, 960, 1), dtype=np.float32)
    rr, cc = skimage.draw.polygon(np.array(yy), np.array(xx), shape=img_seg.shape)
    img_seg[rr, cc] = 1.0

    if extract is not None:
        bb = _quokka_normalize_extract(extract)
        img_seg = bb.extract_from_image(img_seg)

    segmap = SegmentationMapOnImage(img_seg, shape=img_seg.shape[0:2] + (3,))

    if size is not None:
        shape_resized = _compute_resized_shape(img_seg.shape, size)
        segmap = segmap.resize(shape_resized[0:2])
        segmap.shape = tuple(shape_resized[0:2]) + (3,)

    return segmap


def quokka_keypoints(size=None, extract=None):
    """
    Returns example keypoints on the standard example quokke image.

    The keypoints cover the eyes, ears, nose and paws.

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the keypoints are placed. If None, then the keypoints
        are not projected to any new size (positions on the original image are used).
        Floats lead to relative size changes, ints to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.BoundingBox or imgaug.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`imgaug.quokka`.

    Returns
    -------
    kpsoi : imgaug.KeypointsOnImage
        Example keypoints on the quokka image.

    """
    left, top = 0, 0
    if extract is not None:
        bb_extract = _quokka_normalize_extract(extract)
        left = bb_extract.x1
        top = bb_extract.y1
    with open(QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    keypoints = []
    for kp_dict in json_dict["keypoints"]:
        keypoints.append(Keypoint(x=kp_dict["x"] - left, y=kp_dict["y"] - top))
    if extract is not None:
        shape = (bb_extract.height, bb_extract.width, 3)
    else:
        shape = (643, 960, 3)
    kpsoi = KeypointsOnImage(keypoints, shape=shape)
    if size is not None:
        shape_resized = _compute_resized_shape(shape, size)
        kpsoi = kpsoi.on(shape_resized)
    return kpsoi


def quokka_bounding_boxes(size=None, extract=None):
    """
    Returns example bounding boxes on the standard example quokke image.

    Currently only a single bounding box is returned that covers the quokka.

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the BBs are placed. If None, then the BBs
        are not projected to any new size (positions on the original image are used).
        Floats lead to relative size changes, ints to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.BoundingBox or imgaug.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`imgaug.quokka`.

    Returns
    -------
    bbsoi : imgaug.BoundingBoxesOnImage
        Example BBs on the quokka image.

    """
    left, top = 0, 0
    if extract is not None:
        bb_extract = _quokka_normalize_extract(extract)
        left = bb_extract.x1
        top = bb_extract.y1
    with open(QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    bbs = []
    for bb_dict in json_dict["bounding_boxes"]:
        bbs.append(
            BoundingBox(
                x1=bb_dict["x1"] - left,
                y1=bb_dict["y1"] - top,
                x2=bb_dict["x2"] - left,
                y2=bb_dict["y2"] - top
            )
        )
    if extract is not None:
        shape = (bb_extract.height, bb_extract.width, 3)
    else:
        shape = (643, 960, 3)
    bbsoi = BoundingBoxesOnImage(bbs, shape=shape)
    if size is not None:
        shape_resized = _compute_resized_shape(shape, size)
        bbsoi = bbsoi.on(shape_resized)
    return bbsoi


def angle_between_vectors(v1, v2):
    """
    Returns the angle in radians between vectors `v1` and `v2`.

    From http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

    Parameters
    ----------
    v1 : (N,) ndarray
        First vector.

    v2 : (N,) ndarray
        Second vector.

    Returns
    -------
    out : float
        Angle in radians.

    Examples
    --------
    >>> angle_between_vectors(np.float32([1, 0, 0]), np.float32([0, 1, 0]))
    1.570796...

    >>> angle_between_vectors(np.float32([1, 0, 0]), np.float32([1, 0, 0]))
    0.0

    >>> angle_between_vectors(np.float32([1, 0, 0]), np.float32([-1, 0, 0]))
    3.141592...

    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# TODO is this used anywhere?
def compute_line_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Compute the intersection point of two lines.

    Taken from https://stackoverflow.com/a/20679579 .

    Parameters
    ----------
    x1 : number
        x coordinate of the first point on line 1. (The lines extends beyond this point.)

    y1 : number:
        y coordinate of the first point on line 1. (The lines extends beyond this point.)

    x2 : number
        x coordinate of the second point on line 1. (The lines extends beyond this point.)

    y2 : number:
        y coordinate of the second point on line 1. (The lines extends beyond this point.)

    x3 : number
        x coordinate of the first point on line 2. (The lines extends beyond this point.)

    y3 : number:
        y coordinate of the first point on line 2. (The lines extends beyond this point.)

    x4 : number
        x coordinate of the second point on line 2. (The lines extends beyond this point.)

    y4 : number:
        y coordinate of the second point on line 2. (The lines extends beyond this point.)

    Returns
    -------
    tuple of number or bool
        The coordinate of the intersection point as a tuple ``(x, y)``.
        If the lines are parallel (no intersection point or an infinite number of them), the result is False.

    """
    def _make_line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    L1 = _make_line((x1, y1), (x2, y2))
    L2 = _make_line((x3, y3), (x4, y4))

    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


# TODO replace by cv2.putText()?
def draw_text(img, y, x, text, color=(0, 255, 0), size=25):
    """
    Draw text on an image.

    This uses by default DejaVuSans as its font, which is included in this library.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: yes; not tested
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

        TODO check if other dtypes could be enabled

    Parameters
    ----------
    img : (H,W,3) ndarray
        The image array to draw text on.
        Expected to be of dtype uint8 or float32 (value range 0.0 to 255.0).

    y : int
        x-coordinate of the top left corner of the text.

    x : int
        y- coordinate of the top left corner of the text.

    text : str
        The text to draw.

    color : iterable of int, optional
        Color of the text to draw. For RGB-images this is expected to be an RGB color.

    size : int, optional
        Font size of the text to draw.

    Returns
    -------
    img_np : (H,W,3) ndarray
        Input image with text drawn on it.

    """
    do_assert(img.dtype in [np.uint8, np.float32])

    input_dtype = img.dtype
    if img.dtype == np.float32:
        img = img.astype(np.uint8)

    img = PIL_Image.fromarray(img)
    font = PIL_ImageFont.truetype(DEFAULT_FONT_FP, size)
    context = PIL_ImageDraw.Draw(img)
    context.text((x, y), text, fill=tuple(color), font=font)
    img_np = np.asarray(img)
    img_np.setflags(write=True)  # PIL/asarray returns read only array

    if img_np.dtype != input_dtype:
        img_np = img_np.astype(input_dtype)

    return img_np


# TODO rename sizes to size?
def imresize_many_images(images, sizes=None, interpolation=None):
    """
    Resize many images to a specified size.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (3)
        * ``int16``: yes; tested
        * ``int32``: limited; tested (4)
        * ``int64``: no (2)
        * ``float16``: yes; tested (5)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (6)

        - (1) rejected by ``cv2.imresize``
        - (2) results too inaccurate
        - (3) mapped internally to ``int16`` when interpolation!="nearest"
        - (4) only supported for interpolation="nearest", other interpolations lead to cv2 error
        - (5) mapped internally to ``float32``
        - (6) mapped internally to ``uint8``

    Parameters
    ----------
    images : (N,H,W,[C]) ndarray or list of (H,W,[C]) ndarray
        Array of the images to resize.
        Usually recommended to be of dtype uint8.

    sizes : float or iterable of int or iterable of float
        The new size of the images, given either as a fraction (a single float) or as
        a ``(height, width)`` tuple of two integers or as a ``(height fraction, width fraction)``
        tuple of two floats.

    interpolation : None or str or int, optional
        The interpolation to use during resize.
        If int, then expected to be one of:

            * ``cv2.INTER_NEAREST`` (nearest neighbour interpolation)
            * ``cv2.INTER_LINEAR`` (linear interpolation)
            * ``cv2.INTER_AREA`` (area interpolation)
            * ``cv2.INTER_CUBIC`` (cubic interpolation)

        If string, then expected to be one of:

            * ``nearest`` (identical to ``cv2.INTER_NEAREST``)
            * ``linear`` (identical to ``cv2.INTER_LINEAR``)
            * ``area`` (identical to ``cv2.INTER_AREA``)
            * ``cubic`` (identical to ``cv2.INTER_CUBIC``)

        If None, the interpolation will be chosen automatically. For size
        increases, area interpolation will be picked and for size decreases,
        linear interpolation will be picked.

    Returns
    -------
    result : (N,H',W',[C]) ndarray
        Array of the resized images.

    Examples
    --------
    >>> imresize_many_images(np.zeros((2, 16, 16, 3), dtype=np.uint8), 2.0)

    Converts 2 RGB images of height and width 16 to images of height and width 16*2 = 32.

    >>> imresize_many_images(np.zeros((2, 16, 16, 3), dtype=np.uint8), (16, 32))

    Converts 2 RGB images of height and width 16 to images of height 16 and width 32.

    >>> imresize_many_images(np.zeros((2, 16, 16, 3), dtype=np.uint8), (2.0, 4.0))

    Converts 2 RGB images of height and width 16 to images of height 32 and width 64.

    """
    # we just do nothing if the input contains zero images
    # one could also argue that an exception would be appropriate here
    if len(images) == 0:
        return images

    # verify that all input images have height/width > 0
    do_assert(
        all([image.shape[0] > 0 and image.shape[1] > 0 for image in images]),
        ("Cannot resize images, because at least one image has a height and/or width of zero. "
         + "Observed shapes were: %s.") % (str([image.shape for image in images]),)
    )

    # verify that sizes contains only values >0
    if is_single_number(sizes) and sizes <= 0:
        raise Exception(
            "Cannot resize to the target size %.8f, because the value is zero or lower than zero." % (sizes,))
    elif isinstance(sizes, tuple) and (sizes[0] <= 0 or sizes[1] <= 0):
        sizes_str = [
            "int %d" % (sizes[0],) if is_single_integer(sizes[0]) else "float %.8f" % (sizes[0],),
            "int %d" % (sizes[1],) if is_single_integer(sizes[1]) else "float %.8f" % (sizes[1],),
        ]
        sizes_str = "(%s, %s)" % (sizes_str[0], sizes_str[1])
        raise Exception(
            "Cannot resize to the target sizes %s. At least one value is zero or lower than zero." % (sizes_str,))

    # change after the validation to make the above error messages match the original input
    if is_single_number(sizes):
        sizes = (sizes, sizes)
    else:
        do_assert(len(sizes) == 2, "Expected tuple with exactly two entries, got %d entries." % (len(sizes),))
        do_assert(all([is_single_number(val) for val in sizes]),
                  "Expected tuple with two ints or floats, got types %s." % (str([type(val) for val in sizes]),))

    # if input is a list, call this function N times for N images
    # but check beforehand if all images have the same shape, then just convert to a single array and de-convert
    # afterwards
    if isinstance(images, list):
        nb_shapes = len(set([image.shape for image in images]))
        if nb_shapes == 1:
            return list(imresize_many_images(np.array(images), sizes=sizes, interpolation=interpolation))
        else:
            return [imresize_many_images(image[np.newaxis, ...], sizes=sizes, interpolation=interpolation)[0, ...]
                    for image in images]

    shape = images.shape
    do_assert(images.ndim in [3, 4], "Expected array of shape (N, H, W, [C]), got shape %s" % (str(shape),))
    nb_images = shape[0]
    im_height, im_width = shape[1], shape[2]
    nb_channels = shape[3] if images.ndim > 3 else None

    height, width = sizes[0], sizes[1]
    height = int(np.round(im_height * height)) if is_single_float(height) else height
    width = int(np.round(im_width * width)) if is_single_float(width) else width

    if height == im_height and width == im_width:
        return np.copy(images)

    ip = interpolation
    do_assert(ip is None or ip in IMRESIZE_VALID_INTERPOLATIONS)
    if ip is None:
        if height > im_height or width > im_width:
            ip = cv2.INTER_AREA
        else:
            ip = cv2.INTER_LINEAR
    elif ip in ["nearest", cv2.INTER_NEAREST]:
        ip = cv2.INTER_NEAREST
    elif ip in ["linear", cv2.INTER_LINEAR]:
        ip = cv2.INTER_LINEAR
    elif ip in ["area", cv2.INTER_AREA]:
        ip = cv2.INTER_AREA
    else:  # if ip in ["cubic", cv2.INTER_CUBIC]:
        ip = cv2.INTER_CUBIC

    # TODO find more beautiful way to avoid circular imports
    from . import dtypes as iadt
    if ip == cv2.INTER_NEAREST:
        iadt.gate_dtypes(images,
                         allowed=["bool", "uint8", "uint16", "int8", "int16", "int32", "float16", "float32", "float64"],
                         disallowed=["uint32", "uint64", "uint128", "uint256", "int64", "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=None)
    else:
        iadt.gate_dtypes(images,
                         allowed=["bool", "uint8", "uint16", "int8", "int16", "float16", "float32", "float64"],
                         disallowed=["uint32", "uint64", "uint128", "uint256", "int32", "int64", "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=None)

    result_shape = (nb_images, height, width)
    if nb_channels is not None:
        result_shape = result_shape + (nb_channels,)
    result = np.zeros(result_shape, dtype=images.dtype)
    for i, image in enumerate(images):
        input_dtype = image.dtype
        if image.dtype.type == np.bool_:
            image = image.astype(np.uint8) * 255
        elif image.dtype.type == np.int8 and ip != cv2.INTER_NEAREST:
            image = image.astype(np.int16)
        elif image.dtype.type == np.float16:
            image = image.astype(np.float32)

        result_img = cv2.resize(image, (width, height), interpolation=ip)
        assert result_img.dtype == image.dtype

        # cv2 removes the channel axis if input was (H, W, 1)
        # we re-add it (but only if input was not (H, W))
        if len(result_img.shape) == 2 and nb_channels is not None and nb_channels == 1:
            result_img = result_img[:, :, np.newaxis]

        if input_dtype.type == np.bool_:
            result_img = result_img > 127
        elif input_dtype.type == np.int8 and ip != cv2.INTER_NEAREST:
            # TODO somehow better avoid circular imports here
            from . import dtypes as iadt
            result_img = iadt.restore_dtypes_(result_img, np.int8)
        elif input_dtype.type == np.float16:
            # TODO see above
            from . import dtypes as iadt
            result_img = iadt.restore_dtypes_(result_img, np.float16)
        result[i] = result_img
    return result


def imresize_single_image(image, sizes, interpolation=None):
    """
    Resizes a single image.


    dtype support::

        See :func:`imgaug.imgaug.imresize_many_images`.

    Parameters
    ----------
    image : (H,W,C) ndarray or (H,W) ndarray
        Array of the image to resize.
        Usually recommended to be of dtype uint8.

    sizes : float or iterable of int or iterable of float
        See :func:`imgaug.imgaug.imresize_many_images`.

    interpolation : None or str or int, optional
        See :func:`imgaug.imgaug.imresize_many_images`.

    Returns
    -------
    out : (H',W',C) ndarray or (H',W') ndarray
        The resized image.

    """
    grayscale = False
    if image.ndim == 2:
        grayscale = True
        image = image[:, :, np.newaxis]
    do_assert(len(image.shape) == 3, image.shape)
    rs = imresize_many_images(image[np.newaxis, :, :, :], sizes, interpolation=interpolation)
    if grayscale:
        return np.squeeze(rs[0, :, :, 0])
    else:
        return rs[0, ...]


# TODO add crop() function too
def pad(arr, top=0, right=0, bottom=0, left=0, mode="constant", cval=0):
    """
    Pad an image-like array on its top/right/bottom/left side.

    This function is a wrapper around :func:`numpy.pad`.

    dtype support::

        * ``uint8``: yes; fully tested (1)
        * ``uint16``: yes; fully tested (1)
        * ``uint32``: yes; fully tested (2) (3)
        * ``uint64``: yes; fully tested (2) (3)
        * ``int8``: yes; fully tested (1)
        * ``int16``: yes; fully tested (1)
        * ``int32``: yes; fully tested (1)
        * ``int64``: yes; fully tested (2) (3)
        * ``float16``: yes; fully tested (2) (3)
        * ``float32``: yes; fully tested (1)
        * ``float64``: yes; fully tested (1)
        * ``float128``: yes; fully tested (2) (3)
        * ``bool``: yes; tested (2) (3)

        - (1) Uses ``cv2`` if `mode` is one of: ``"constant"``, ``"edge"``, ``"reflect"``, ``"symmetric"``.
              Otherwise uses ``numpy``.
        - (2) Uses ``numpy``.
        - (3) Rejected by ``cv2``.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pad.

    top : int, optional
        Amount of pixels to add at the top side of the image. Must be 0 or greater.

    right : int, optional
        Amount of pixels to add at the right side of the image. Must be 0 or greater.

    bottom : int, optional
        Amount of pixels to add at the bottom side of the image. Must be 0 or greater.

    left : int, optional
        Amount of pixels to add at the left side of the image. Must be 0 or greater.

    mode : str, optional
        Padding mode to use. See :func:`numpy.pad` for details.
        In case of mode ``constant``, the parameter `cval` will be used as the ``constant_values``
        parameter to :func:`numpy.pad`.
        In case of mode ``linear_ramp``, the parameter `cval` will be used as the ``end_values``
        parameter to :func:`numpy.pad`.

    cval : number, optional
        Value to use for padding if `mode` is ``constant``. See :func:`numpy.pad` for details.
        The cval is expected to match the input array's dtype and value range.

    Returns
    -------
    arr_pad : (H',W') ndarray or (H',W',C) ndarray
        Padded array with height ``H'=H+top+bottom`` and width ``W'=W+left+right``.

    """
    do_assert(arr.ndim in [2, 3])
    do_assert(top >= 0)
    do_assert(right >= 0)
    do_assert(bottom >= 0)
    do_assert(left >= 0)
    if top > 0 or right > 0 or bottom > 0 or left > 0:
        mapping_mode_np_to_cv2 = {
            "constant": cv2.BORDER_CONSTANT,
            "edge": cv2.BORDER_REPLICATE,
            "linear_ramp": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "minimum": None,
            "reflect": cv2.BORDER_REFLECT_101,
            "symmetric": cv2.BORDER_REFLECT,
            "wrap": None,
            cv2.BORDER_CONSTANT: cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE: cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT_101: cv2.BORDER_REFLECT_101,
            cv2.BORDER_REFLECT: cv2.BORDER_REFLECT
        }
        bad_mode_cv2 = mapping_mode_np_to_cv2.get(mode, None) is None

        # these datatypes all simply generate a "TypeError: src data type = X is not supported" error
        bad_datatype_cv2 = arr.dtype.name in ["uint32", "uint64", "int64", "float16", "float128", "bool"]

        if not bad_datatype_cv2 and not bad_mode_cv2:
            cval = float(cval) if arr.dtype.kind == "f" else int(cval)  # results in TypeError otherwise for np inputs

            if arr.ndim == 2 or arr.shape[2] <= 4:
                # without this, only the first channel is padded with the cval, all following channels with 0
                if arr.ndim == 3:
                    cval = tuple([cval] * arr.shape[2])

                arr_pad = cv2.copyMakeBorder(arr, top=top, bottom=bottom, left=left, right=right,
                                             borderType=mapping_mode_np_to_cv2[mode], value=cval)
                if arr.ndim == 3 and arr_pad.ndim == 2:
                    arr_pad = arr_pad[..., np.newaxis]
            else:
                result = []
                channel_start_idx = 0
                while channel_start_idx < arr.shape[2]:
                    arr_c = arr[..., channel_start_idx:channel_start_idx+4]
                    cval_c = tuple([cval] * arr_c.shape[2])
                    arr_pad_c = cv2.copyMakeBorder(arr_c, top=top, bottom=bottom, left=left, right=right,
                                                   borderType=mapping_mode_np_to_cv2[mode], value=cval_c)
                    arr_pad_c = np.atleast_3d(arr_pad_c)
                    result.append(arr_pad_c)
                    channel_start_idx += 4
                arr_pad = np.concatenate(result, axis=2)
        else:
            paddings_np = [(top, bottom), (left, right)]  # paddings for 2d case
            if arr.ndim == 3:
                paddings_np.append((0, 0))  # add paddings for 3d case

            if mode == "constant":
                arr_pad = np.pad(arr, paddings_np, mode=mode, constant_values=cval)
            elif mode == "linear_ramp":
                arr_pad = np.pad(arr, paddings_np, mode=mode, end_values=cval)
            else:
                arr_pad = np.pad(arr, paddings_np, mode=mode)

        return arr_pad
    return np.copy(arr)


# TODO allow shape as input instead of array
def compute_paddings_for_aspect_ratio(arr, aspect_ratio):
    """
    Compute the amount of pixels by which an array has to be padded to fulfill an aspect ratio.

    The aspect ratio is given as width/height.
    Depending on which dimension is smaller (height or width), only the corresponding
    sides (left/right or top/bottom) will be padded. In each case, both of the sides will
    be padded equally.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array for which to compute pad amounts.

    aspect_ratio : float
        Target aspect ratio, given as width/height. E.g. 2.0 denotes the image having twice
        as much width as height.

    Returns
    -------
    result : tuple of int
        Required paddign amounts to reach the target aspect ratio, given as a tuple
        of the form ``(top, right, bottom, left)``.

    """
    do_assert(arr.ndim in [2, 3])
    do_assert(aspect_ratio > 0)
    height, width = arr.shape[0:2]
    do_assert(height > 0)
    aspect_ratio_current = width / height

    pad_top = 0
    pad_right = 0
    pad_bottom = 0
    pad_left = 0

    if aspect_ratio_current < aspect_ratio:
        # vertical image, height > width
        diff = (aspect_ratio * height) - width
        pad_right = int(np.ceil(diff / 2))
        pad_left = int(np.floor(diff / 2))
    elif aspect_ratio_current > aspect_ratio:
        # horizontal image, width > height
        diff = ((1/aspect_ratio) * width) - height
        pad_top = int(np.floor(diff / 2))
        pad_bottom = int(np.ceil(diff / 2))

    return pad_top, pad_right, pad_bottom, pad_left


def pad_to_aspect_ratio(arr, aspect_ratio, mode="constant", cval=0, return_pad_amounts=False):
    """
    Pad an image-like array on its sides so that it matches a target aspect ratio.

    Depending on which dimension is smaller (height or width), only the corresponding
    sides (left/right or top/bottom) will be padded. In each case, both of the sides will
    be padded equally.

    dtype support::

        See :func:`imgaug.imgaug.pad`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pad.

    aspect_ratio : float
        Target aspect ratio, given as width/height. E.g. 2.0 denotes the image having twice
        as much width as height.

    mode : str, optional
        Padding mode to use. See :func:`numpy.pad` for details.

    cval : number, optional
        Value to use for padding if `mode` is ``constant``. See :func:`numpy.pad` for details.

    return_pad_amounts : bool, optional
        If False, then only the padded image will be returned. If True, a tuple with two
        entries will be returned, where the first entry is the padded image and the second
        entry are the amounts by which each image side was padded. These amounts are again a
        tuple of the form (top, right, bottom, left), with each value being an integer.

    Returns
    -------
    arr_padded : (H',W') ndarray or (H',W',C) ndarray
        Padded image as (H',W') or (H',W',C) ndarray, fulfulling the given aspect_ratio.

    tuple of int
        Amounts by which the image was padded on each side, given as a tuple ``(top, right, bottom, left)``.
        This tuple is only returned if `return_pad_amounts` was set to True.
        Otherwise only ``arr_padded`` is returned.

    """
    pad_top, pad_right, pad_bottom, pad_left = compute_paddings_for_aspect_ratio(arr, aspect_ratio)
    arr_padded = pad(
        arr,
        top=pad_top,
        right=pad_right,
        bottom=pad_bottom,
        left=pad_left,
        mode=mode,
        cval=cval
    )

    if return_pad_amounts:
        return arr_padded, (pad_top, pad_right, pad_bottom, pad_left)
    else:
        return arr_padded


def pool(arr, block_size, func, cval=0, preserve_dtype=True):
    """
    Resize an array by pooling values within blocks.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested (2)
        * ``uint64``: no (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested (2)
        * ``int64``: no (1)
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested (2)
        * ``bool``: yes; tested

        - (1) results too inaccurate (at least when using np.average as func)
        - (2) Note that scikit-image documentation says that the wrapped pooling function converts
              inputs to float64. Actual tests showed no indication of that happening (at least when
              using preserve_dtype=True).

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool. Ideally of datatype ``numpy.float64``.

    block_size : int or tuple of int
        Spatial size of each group of values to pool, aka kernel size.
        If a single integer, then a symmetric block of that size along height and width will be used.
        If a tuple of two values, it is assumed to be the block size along height and width of the image-like,
        with pooling happening per channel.
        If a tuple of three values, it is assumed to be the block size along height, width and channels.

    func : callable
        Function to apply to a given block in order to convert it to a single number,
        e.g. :func:`numpy.average`, :func:`numpy.min`, :func:`numpy.max`.

    cval : number, optional
        Value to use in order to pad the array along its border if the array cannot be divided
        by `block_size` without remainder.

    preserve_dtype : bool, optional
        Whether to convert the array back to the input datatype if it is changed away from
        that in the pooling process.

    Returns
    -------
    arr_reduced : (H',W') ndarray or (H',W',C') ndarray
        Array after pooling.

    """
    # TODO find better way to avoid circular import
    from . import dtypes as iadt
    iadt.gate_dtypes(arr,
                     allowed=["bool", "uint8", "uint16", "uint32", "int8", "int16", "int32",
                              "float16", "float32", "float64", "float128"],
                     disallowed=["uint64", "uint128", "uint256", "int64", "int128", "int256",
                                 "float256"],
                     augmenter=None)

    do_assert(arr.ndim in [2, 3])
    is_valid_int = is_single_integer(block_size) and block_size >= 1
    is_valid_tuple = is_iterable(block_size) and len(block_size) in [2, 3] \
        and [is_single_integer(val) and val >= 1 for val in block_size]
    do_assert(is_valid_int or is_valid_tuple)

    if is_single_integer(block_size):
        block_size = [block_size, block_size]
    if len(block_size) < arr.ndim:
        block_size = list(block_size) + [1]

    input_dtype = arr.dtype
    arr_reduced = skimage.measure.block_reduce(arr, tuple(block_size), func, cval=cval)
    if preserve_dtype and arr_reduced.dtype.type != input_dtype:
        arr_reduced = arr_reduced.astype(input_dtype)
    return arr_reduced


def avg_pool(arr, block_size, cval=0, preserve_dtype=True):
    """
    Resize an array using average pooling.

    dtype support::

        See :func:`imgaug.imgaug.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool. See :func:`imgaug.pool` for details.

    block_size : int or tuple of int or tuple of int
        Size of each block of values to pool. See :func:`imgaug.pool` for details.

    cval : number, optional
        Padding value. See :func:`imgaug.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype. See :func:`imgaug.pool` for details.

    Returns
    -------
    arr_reduced : (H',W') ndarray or (H',W',C') ndarray
        Array after average pooling.

    """
    return pool(arr, block_size, np.average, cval=cval, preserve_dtype=preserve_dtype)


def max_pool(arr, block_size, cval=0, preserve_dtype=True):
    """
    Resize an array using max-pooling.

    dtype support::

        See :func:`imgaug.imgaug.pool`.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Image-like array to pool. See :func:`imgaug.pool` for details.

    block_size : int or tuple of int or tuple of int
        Size of each block of values to pool. See `imgaug.pool` for details.

    cval : number, optional
        Padding value. See :func:`imgaug.pool` for details.

    preserve_dtype : bool, optional
        Whether to preserve the input array dtype. See :func:`imgaug.pool` for details.

    Returns
    -------
    arr_reduced : (H',W') ndarray or (H',W',C') ndarray
        Array after max-pooling.

    """
    return pool(arr, block_size, np.max, cval=cval, preserve_dtype=preserve_dtype)


def draw_grid(images, rows=None, cols=None):
    """
    Converts multiple input images into a single image showing them in a grid.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested
        * ``float128``: yes; fully tested
        * ``bool``: yes; fully tested

    Parameters
    ----------
    images : (N,H,W,3) ndarray or iterable of (H,W,3) array
        The input images to convert to a grid.

    rows : None or int, optional
        The number of rows to show in the grid.
        If None, it will be automatically derived.

    cols : None or int, optional
        The number of cols to show in the grid.
        If None, it will be automatically derived.

    Returns
    -------
    grid : (H',W',3) ndarray
        Image of the generated grid.

    """
    nb_images = len(images)
    do_assert(nb_images > 0)

    if is_np_array(images):
        do_assert(images.ndim == 4)
    else:
        do_assert(is_iterable(images) and is_np_array(images[0]) and images[0].ndim == 3)
        dts = [image.dtype.name for image in images]
        nb_dtypes = len(set(dts))
        do_assert(nb_dtypes == 1, ("All images provided to draw_grid() must have the same dtype, "
                                   + "found %d dtypes (%s)") % (nb_dtypes, ", ".join(dts)))

    cell_height = max([image.shape[0] for image in images])
    cell_width = max([image.shape[1] for image in images])
    channels = set([image.shape[2] for image in images])
    do_assert(
        len(channels) == 1,
        "All images are expected to have the same number of channels, "
        + "but got channel set %s with length %d instead." % (str(channels), len(channels))
    )
    nb_channels = list(channels)[0]
    if rows is None and cols is None:
        rows = cols = int(math.ceil(math.sqrt(nb_images)))
    elif rows is not None:
        cols = int(math.ceil(nb_images / rows))
    elif cols is not None:
        rows = int(math.ceil(nb_images / cols))
    do_assert(rows * cols >= nb_images)

    width = cell_width * cols
    height = cell_height * rows
    dt = images.dtype if is_np_array(images) else images[0].dtype
    grid = np.zeros((height, width, nb_channels), dtype=dt)
    cell_idx = 0
    for row_idx in sm.xrange(rows):
        for col_idx in sm.xrange(cols):
            if cell_idx < nb_images:
                image = images[cell_idx]
                cell_y1 = cell_height * row_idx
                cell_y2 = cell_y1 + image.shape[0]
                cell_x1 = cell_width * col_idx
                cell_x2 = cell_x1 + image.shape[1]
                grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image
            cell_idx += 1

    return grid


def show_grid(images, rows=None, cols=None):
    """
    Converts the input images to a grid image and shows it in a new window.

    dtype support::

        minimum of (
            :func:`imgaug.imgaug.draw_grid`,
            :func:`imgaug.imgaug.imshow`
        )

    Parameters
    ----------
    images : (N,H,W,3) ndarray or iterable of (H,W,3) array
        See :func:`imgaug.draw_grid`.

    rows : None or int, optional
        See :func:`imgaug.draw_grid`.

    cols : None or int, optional
        See :func:`imgaug.draw_grid`.

    """
    grid = draw_grid(images, rows=rows, cols=cols)
    imshow(grid)


def imshow(image, backend=IMSHOW_BACKEND_DEFAULT):
    """
    Shows an image in a window.

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
    image : (H,W,3) ndarray
        Image to show.

    backend : {'matplotlib', 'cv2'}, optional
        Library to use to show the image. May be either matplotlib or OpenCV ('cv2').
        OpenCV tends to be faster, but apparently causes more technical issues.

    """
    do_assert(backend in ["matplotlib", "cv2"], "Expected backend 'matplotlib' or 'cv2', got %s." % (backend,))

    if backend == "cv2":
        image_bgr = image
        if image.ndim == 3 and image.shape[2] in [3, 4]:
            image_bgr = image[..., 0:3][..., ::-1]

        win_name = "imgaug-default-window"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, image_bgr)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)
    else:
        # import only when necessary (faster startup; optional dependency; less fragile -- see issue #225)
        import matplotlib.pyplot as plt

        dpi = 96
        h, w = image.shape[0] / dpi, image.shape[1] / dpi
        w = max(w, 6)  # if the figure is too narrow, the footer may appear and make the fig suddenly wider (ugly)
        fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
        fig.canvas.set_window_title("imgaug.imshow(%s)" % (image.shape,))
        ax.imshow(image, cmap="gray")  # cmap is only activate for grayscale images
        plt.show()


def do_assert(condition, message="Assertion failed."):
    """
    Function that behaves equally to an `assert` statement, but raises an
    Exception.

    This is added because `assert` statements are removed in optimized code.
    It replaces `assert` statements throughout the library that should be
    kept even in optimized code.

    Parameters
    ----------
    condition : bool
        If False, an exception is raised.

    message : str, optional
        Error message.

    """
    if not condition:
        raise AssertionError(str(message))


class HooksImages(object):
    """
    Class to intervene with image augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    Parameters
    ----------
    activator : None or callable, optional
        A function that gives permission to execute an augmenter.
        The expected interface is ``f(images, augmenter, parents, default)``,
        where ``images`` are the input images to augment, ``augmenter`` is the
        instance of the augmenter to execute, ``parents`` are previously
        executed augmenters and ``default`` is an expected default value to be
        returned if the activator function does not plan to make a decision
        for the given inputs.

    propagator : None or callable, optional
        A function that gives permission to propagate the augmentation further
        to the children of an augmenter. This happens after the activator.
        In theory, an augmenter may augment images itself (if allowed by the
        activator) and then execute child augmenters afterwards (if allowed by
        the propagator). If the activator returned False, the propagation step
        will never be executed.
        The expected interface is ``f(images, augmenter, parents, default)``,
        with all arguments having identical meaning to the activator.

    preprocessor : None or callable, optional
        A function to call before an augmenter performed any augmentations.
        The interface is ``f(images, augmenter, parents)``,
        with all arguments having identical meaning to the activator.
        It is expected to return the input images, optionally modified.

    postprocessor : None or callable, optional
        A function to call after an augmenter performed augmentations.
        The interface is the same as for the preprocessor.

    Examples
    --------
    >>> seq = iaa.Sequential([
    >>>     iaa.GaussianBlur(3.0, name="blur"),
    >>>     iaa.Dropout(0.05, name="dropout"),
    >>>     iaa.Affine(translate_px=-5, name="affine")
    >>> ])
    >>> images = [np.zeros((10, 10), dtype=np.uint8)]
    >>>
    >>> def activator(images, augmenter, parents, default):
    >>>     return False if augmenter.name in ["blur", "dropout"] else default
    >>>
    >>> seq_det = seq.to_deterministic()
    >>> images_aug = seq_det.augment_images(images)
    >>> heatmaps = [np.random.rand(*(3, 10, 10))]
    >>> heatmaps_aug = seq_det.augment_images(
    >>>     heatmaps,
    >>>     hooks=ia.HooksImages(activator=activator)
    >>> )

    This augments images and their respective heatmaps in the same way.
    The heatmaps however are only modified by Affine, not by GaussianBlur or
    Dropout.

    """

    def __init__(self, activator=None, propagator=None, preprocessor=None, postprocessor=None):
        self.activator = activator
        self.propagator = propagator
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def is_activated(self, images, augmenter, parents, default):
        """
        Returns whether an augmenter may be executed.

        Returns
        -------
        bool
            If True, the augmenter may be executed. If False, it may not be executed.

        """
        if self.activator is None:
            return default
        else:
            return self.activator(images, augmenter, parents, default)

    def is_propagating(self, images, augmenter, parents, default):
        """
        Returns whether an augmenter may call its children to augment an
        image. This is independent of the augmenter itself possible changing
        the image, without calling its children. (Most (all?) augmenters with
        children currently dont perform any changes themselves.)

        Returns
        -------
        bool
            If True, the augmenter may be propagate to its children. If False, it may not.

        """
        if self.propagator is None:
            return default
        else:
            return self.propagator(images, augmenter, parents, default)

    def preprocess(self, images, augmenter, parents):
        """
        A function to be called before the augmentation of images starts (per augmenter).

        Returns
        -------
        (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            The input images, optionally modified.

        """
        if self.preprocessor is None:
            return images
        else:
            return self.preprocessor(images, augmenter, parents)

    def postprocess(self, images, augmenter, parents):
        """
        A function to be called after the augmentation of images was
        performed.

        Returns
        -------
        (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            The input images, optionally modified.

        """
        if self.postprocessor is None:
            return images
        else:
            return self.postprocessor(images, augmenter, parents)


class HooksHeatmaps(HooksImages):
    """
    Class to intervene with heatmap augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    This class is currently the same as the one for images. This may or may
    not change in the future.

    """
    pass


class HooksKeypoints(HooksImages):
    """
    Class to intervene with keypoint augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    This class is currently the same as the one for images. This may or may
    not change in the future.

    """
    pass


def compute_geometric_median(X, eps=1e-5):
    """
    Estimate the geometric median of points in 2D.

    Code from https://stackoverflow.com/a/30305181

    Parameters
    ----------
    X : (N,2) ndarray
        Points in 2D. Second axis must be given in xy-form.

    eps : float, optional
        Distance threshold when to return the median.

    Returns
    -------
    (2,) ndarray
        Geometric median as xy-coordinate.

    """
    y = np.mean(X, 0)

    while True:
        D = scipy.spatial.distance.cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if scipy.spatial.distance.euclidean(y, y1) < eps:
            return y1

        y = y1


class Keypoint(object):
    """
    A single keypoint (aka landmark) on an image.

    Parameters
    ----------
    x : number
        Coordinate of the keypoint on the x axis.

    y : number
        Coordinate of the keypoint on the y axis.

    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def x_int(self):
        """
        Return the keypoint's x-coordinate, rounded to the closest integer.

        Returns
        -------
        result : int
            Keypoint's x-coordinate, rounded to the closest integer.

        """
        return int(np.round(self.x))

    @property
    def y_int(self):
        """
        Return the keypoint's y-coordinate, rounded to the closest integer.

        Returns
        -------
        result : int
            Keypoint's y-coordinate, rounded to the closest integer.

        """
        return int(np.round(self.y))

    def project(self, from_shape, to_shape):
        """
        Project the keypoint onto a new position on a new image.

        E.g. if the keypoint is on its original image at x=(10 of 100 pixels)
        and y=(20 of 100 pixels) and is projected onto a new image with
        size (width=200, height=200), its new position will be (20, 40).

        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).

        Parameters
        ----------
        from_shape : tuple of int
            Shape of the original image. (Before resize.)

        to_shape : tuple of int
            Shape of the new image. (After resize.)

        Returns
        -------
        imgaug.Keypoint
            Keypoint object with new coordinates.

        """
        if from_shape[0:2] == to_shape[0:2]:
            return Keypoint(x=self.x, y=self.y)
        else:
            from_height, from_width = from_shape[0:2]
            to_height, to_width = to_shape[0:2]
            x = (self.x / from_width) * to_width
            y = (self.y / from_height) * to_height
            return Keypoint(x=x, y=y)

    def shift(self, x=0, y=0):
        """
        Move the keypoint around on an image.

        Parameters
        ----------
        x : number, optional
            Move by this value on the x axis.

        y : number, optional
            Move by this value on the y axis.

        Returns
        -------
        imgaug.Keypoint
            Keypoint object with new coordinates.

        """
        return Keypoint(self.x + x, self.y + y)

    def generate_similar_points_manhattan(self, nb_steps, step_size, return_array=False):
        """
        Generate nearby points to this keypoint based on manhattan distance.

        To generate the first neighbouring points, a distance of S (step size) is moved from the
        center point (this keypoint) to the top, right, bottom and left, resulting in four new
        points. From these new points, the pattern is repeated. Overlapping points are ignored.

        The resulting points have a shape similar to a square rotated by 45 degrees.

        Parameters
        ----------
        nb_steps : int
            The number of steps to move from the center point. nb_steps=1 results in a total of
            5 output points (1 center point + 4 neighbours).

        step_size : number
            The step size to move from every point to its neighbours.

        return_array : bool, optional
            Whether to return the generated points as a list of keypoints or an array
            of shape ``(N,2)``, where ``N`` is the number of generated points and the second axis contains
            the x- (first value) and y- (second value) coordinates.

        Returns
        -------
        points : list of imgaug.Keypoint or (N,2) ndarray
            If return_array was False, then a list of Keypoint.
            Otherwise a numpy array of shape ``(N,2)``, where ``N`` is the number of generated points and
            the second axis contains the x- (first value) and y- (second value) coordinates.
            The center keypoint (the one on which this function was called) is always included.

        """
        # TODO add test
        # Points generates in manhattan style with S steps have a shape similar to a 45deg rotated
        # square. The center line with the origin point has S+1+S = 1+2*S points (S to the left,
        # S to the right). The lines above contain (S+1+S)-2 + (S+1+S)-2-2 + ... + 1 points. E.g.
        # for S=2 it would be 3+1=4 and for S=3 it would be 5+3+1=9. Same for the lines below the
        # center. Hence the total number of points is S+1+S + 2*(S^2).
        points = np.zeros((nb_steps + 1 + nb_steps + 2*(nb_steps**2), 2), dtype=np.float32)

        # we start at the bottom-most line and move towards the top-most line
        yy = np.linspace(self.y - nb_steps * step_size, self.y + nb_steps * step_size, nb_steps + 1 + nb_steps)

        # bottom-most line contains only one point
        width = 1

        nth_point = 0
        for i_y, y in enumerate(yy):
            if width == 1:
                xx = [self.x]
            else:
                xx = np.linspace(self.x - (width-1)//2 * step_size, self.x + (width-1)//2 * step_size, width)
            for x in xx:
                points[nth_point] = [x, y]
                nth_point += 1
            if i_y < nb_steps:
                width += 2
            else:
                width -= 2

        if return_array:
            return points
        return [Keypoint(x=points[i, 0], y=points[i, 1]) for i in sm.xrange(points.shape[0])]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Keypoint(x=%.8f, y=%.8f)" % (self.x, self.y)


class KeypointsOnImage(object):
    """
    Object that represents all keypoints on a single image.

    Parameters
    ----------
    keypoints : list of imgaug.Keypoint
        List of keypoints on the image.

    shape : tuple of int
        The shape of the image on which the keypoints are placed.

    Examples
    --------
    >>> image = np.zeros((70, 70))
    >>> kps = [Keypoint(x=10, y=20), Keypoint(x=34, y=60)]
    >>> kps_oi = KeypointsOnImage(kps, shape=image.shape)

    """
    def __init__(self, keypoints, shape):
        self.keypoints = keypoints
        if is_np_array(shape):
            self.shape = shape.shape
        else:
            do_assert(isinstance(shape, (tuple, list)))
            self.shape = tuple(shape)

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    @property
    def empty(self):
        """
        Returns whether this object contains zero keypoints.

        Returns
        -------
        result : bool
            True if this object contains zero keypoints.

        """
        return len(self.keypoints) == 0

    def on(self, image):
        """
        Project keypoints from one image to a new one.

        Parameters
        ----------
        image : ndarray or tuple of int
            New image onto which the keypoints are to be projected.
            May also simply be that new image's shape tuple.

        Returns
        -------
        keypoints : imgaug.KeypointsOnImage
            Object containing all projected keypoints.

        """
        if is_np_array(image):
            shape = image.shape
        else:
            shape = image

        if shape[0:2] == self.shape[0:2]:
            return self.deepcopy()
        else:
            keypoints = [kp.project(self.shape, shape) for kp in self.keypoints]
            return KeypointsOnImage(keypoints, shape)

    def draw_on_image(self, image, color=(0, 255, 0), size=3, copy=True, raise_if_out_of_image=False):
        """
        Draw all keypoints onto a given image. Each keypoint is marked by a square of a chosen color and size.

        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the keypoints.
            This image should usually have the same shape as
            set in KeypointsOnImage.shape.

        color : int or list of int or tuple of int or (3,) ndarray, optional
            The RGB color of all keypoints. If a single int ``C``, then that is
            equivalent to ``(C,C,C)``.

        size : int, optional
            The size of each point. If set to ``C``, each square will have size ``C x C``.

        copy : bool, optional
            Whether to copy the image before drawing the points.

        raise_if_out_of_image : bool, optional
            Whether to raise an exception if any keypoint is outside of the image.

        Returns
        -------
        image : (H,W,3) ndarray
            Image with drawn keypoints.

        """
        if copy:
            image = np.copy(image)

        height, width = image.shape[0:2]

        for keypoint in self.keypoints:
            y, x = keypoint.y_int, keypoint.x_int
            if 0 <= y < height and 0 <= x < width:
                x1 = max(x - size//2, 0)
                x2 = min(x + 1 + size//2, width)
                y1 = max(y - size//2, 0)
                y2 = min(y + 1 + size//2, height)
                image[y1:y2, x1:x2] = color
            else:
                if raise_if_out_of_image:
                    raise Exception("Cannot draw keypoint x=%.8f, y=%.8f on image with shape %s." % (y, x, image.shape))

        return image

    def shift(self, x=0, y=0):
        """
        Move the keypoints around on an image.

        Parameters
        ----------
        x : number, optional
            Move each keypoint by this value on the x axis.

        y : number, optional
            Move each keypoint by this value on the y axis.

        Returns
        -------
        out : KeypointsOnImage
            Keypoints after moving them.

        """
        keypoints = [keypoint.shift(x=x, y=y) for keypoint in self.keypoints]
        return KeypointsOnImage(keypoints, self.shape)

    def get_coords_array(self):
        """
        Convert the coordinates of all keypoints in this object to an array of shape (N,2).

        Returns
        -------
        result : (N, 2) ndarray
            Where N is the number of keypoints. Each first value is the
            x coordinate, each second value is the y coordinate.

        """
        result = np.zeros((len(self.keypoints), 2), np.float32)
        for i, keypoint in enumerate(self.keypoints):
            result[i, 0] = keypoint.x
            result[i, 1] = keypoint.y
        return result

    @staticmethod
    def from_coords_array(coords, shape):
        """
        Convert an array (N,2) with a given image shape to a KeypointsOnImage object.

        Parameters
        ----------
        coords : (N, 2) ndarray
            Coordinates of ``N`` keypoints on the original image.
            Each first entry ``coords[i, 0]`` is expected to be the x coordinate.
            Each second entry ``coords[i, 1]`` is expected to be the y coordinate.

        shape : tuple
            Shape tuple of the image on which the keypoints are placed.

        Returns
        -------
        out : KeypointsOnImage
            KeypointsOnImage object that contains all keypoints from the array.

        """
        keypoints = [Keypoint(x=coords[i, 0], y=coords[i, 1]) for i in sm.xrange(coords.shape[0])]
        return KeypointsOnImage(keypoints, shape)

    # TODO add to_gaussian_heatmaps(), from_gaussian_heatmaps()
    def to_keypoint_image(self, size=1):
        """
        Draws a new black image of shape ``(H,W,N)`` in which all keypoint coordinates are set to 255.
        (H=shape height, W=shape width, N=number of keypoints)

        This function can be used as a helper when augmenting keypoints with a method that only supports the
        augmentation of images.

        Parameters
        -------
        size : int
            Size of each (squared) point.

        Returns
        -------
        image : (H,W,N) ndarray
            Image in which the keypoints are marked. H is the height,
            defined in KeypointsOnImage.shape[0] (analogous W). N is the
            number of keypoints.

        """
        do_assert(len(self.keypoints) > 0)
        height, width = self.shape[0:2]
        image = np.zeros((height, width, len(self.keypoints)), dtype=np.uint8)
        do_assert(size % 2 != 0)
        sizeh = max(0, (size-1)//2)
        for i, keypoint in enumerate(self.keypoints):
            # TODO for float values spread activation over several cells
            # here and do voting at the end
            y = keypoint.y_int
            x = keypoint.x_int

            x1 = np.clip(x - sizeh, 0, width-1)
            x2 = np.clip(x + sizeh + 1, 0, width)
            y1 = np.clip(y - sizeh, 0, height-1)
            y2 = np.clip(y + sizeh + 1, 0, height)

            if x1 < x2 and y1 < y2:
                image[y1:y2, x1:x2, i] = 128
            if 0 <= y < height and 0 <= x < width:
                image[y, x, i] = 255
        return image

    @staticmethod
    def from_keypoint_image(image, if_not_found_coords={"x": -1, "y": -1}, threshold=1, nb_channels=None): # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
        """
        Converts an image generated by ``to_keypoint_image()`` back to a KeypointsOnImage object.

        Parameters
        ----------
        image : (H,W,N) ndarray
            The keypoints image. N is the number of keypoints.

        if_not_found_coords : tuple or list or dict or None, optional
            Coordinates to use for keypoints that cannot be found in `image`.
            If this is a list/tuple, it must have two integer values.
            If it is a dictionary, it must have the keys ``x`` and ``y`` with
            each containing one integer value.
            If this is None, then the keypoint will not be added to the final
            KeypointsOnImage object.

        threshold : int, optional
            The search for keypoints works by searching for the argmax in
            each channel. This parameters contains the minimum value that
            the max must have in order to be viewed as a keypoint.

        nb_channels : None or int, optional
            Number of channels of the image on which the keypoints are placed.
            Some keypoint augmenters require that information.
            If set to None, the keypoint's shape will be set
            to ``(height, width)``, otherwise ``(height, width, nb_channels)``.

        Returns
        -------
        out : KeypointsOnImage
            The extracted keypoints.

        """
        do_assert(len(image.shape) == 3)
        height, width, nb_keypoints = image.shape

        drop_if_not_found = False
        if if_not_found_coords is None:
            drop_if_not_found = True
            if_not_found_x = -1
            if_not_found_y = -1
        elif isinstance(if_not_found_coords, (tuple, list)):
            do_assert(len(if_not_found_coords) == 2)
            if_not_found_x = if_not_found_coords[0]
            if_not_found_y = if_not_found_coords[1]
        elif isinstance(if_not_found_coords, dict):
            if_not_found_x = if_not_found_coords["x"]
            if_not_found_y = if_not_found_coords["y"]
        else:
            raise Exception("Expected if_not_found_coords to be None or tuple or list or dict, got %s." % (
                type(if_not_found_coords),))

        keypoints = []
        for i in sm.xrange(nb_keypoints):
            maxidx_flat = np.argmax(image[..., i])
            maxidx_ndim = np.unravel_index(maxidx_flat, (height, width))
            found = (image[maxidx_ndim[0], maxidx_ndim[1], i] >= threshold)
            if found:
                keypoints.append(Keypoint(x=maxidx_ndim[1], y=maxidx_ndim[0]))
            else:
                if drop_if_not_found:
                    pass  # dont add the keypoint to the result list, i.e. drop it
                else:
                    keypoints.append(Keypoint(x=if_not_found_x, y=if_not_found_y))

        out_shape = (height, width)
        if nb_channels is not None:
            out_shape += (nb_channels,)
        return KeypointsOnImage(keypoints, shape=out_shape)

    def to_distance_maps(self, inverted=False):
        """
        Generates a ``(H,W,K)`` output containing ``K`` distance maps for ``K`` keypoints.

        The k-th distance map contains at every location ``(y, x)`` the euclidean distance to the k-th keypoint.

        This function can be used as a helper when augmenting keypoints with a method that only supports
        the augmentation of images.

        Parameters
        -------
        inverted : bool, optional
            If True, inverted distance maps are returned where each distance value d is replaced
            by ``d/(d+1)``, i.e. the distance maps have values in the range ``(0.0, 1.0]`` with 1.0
            denoting exactly the position of the respective keypoint.

        Returns
        -------
        distance_maps : (H,W,K) ndarray
            A ``float32`` array containing ``K`` distance maps for ``K`` keypoints. Each location
            ``(y, x, k)`` in the array denotes the euclidean distance at ``(y, x)`` to the ``k``-th keypoint.
            In inverted mode the distance ``d`` is replaced by ``d/(d+1)``. The height and width
            of the array match the height and width in ``KeypointsOnImage.shape``.

        """
        do_assert(len(self.keypoints) > 0)
        height, width = self.shape[0:2]
        distance_maps = np.zeros((height, width, len(self.keypoints)), dtype=np.float32)

        yy = np.arange(0, height)
        xx = np.arange(0, width)
        grid_xx, grid_yy = np.meshgrid(xx, yy)

        for i, keypoint in enumerate(self.keypoints):
            y, x = keypoint.y, keypoint.x
            distance_maps[:, :, i] = (grid_xx - x) ** 2 + (grid_yy - y) ** 2
        distance_maps = np.sqrt(distance_maps)
        if inverted:
            return 1/(distance_maps+1)
        return distance_maps

    # TODO add option to if_not_found_coords to reuse old keypoint coords
    @staticmethod
    def from_distance_maps(distance_maps, inverted=False, if_not_found_coords={"x": -1, "y": -1}, threshold=None, # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
                           nb_channels=None):
        """
        Converts maps generated by ``to_distance_maps()`` back to a KeypointsOnImage object.

        Parameters
        ----------
        distance_maps : (H,W,N) ndarray
            The distance maps. N is the number of keypoints.

        inverted : bool, optional
            Whether the given distance maps were generated in inverted or normal mode.

        if_not_found_coords : tuple or list or dict or None, optional
            Coordinates to use for keypoints that cannot be found in ``distance_maps``.
            If this is a list/tuple, it must have two integer values.
            If it is a dictionary, it must have the keys ``x`` and ``y``, with each
            containing one integer value.
            If this is None, then the keypoint will not be added to the final
            KeypointsOnImage object.

        threshold : float, optional
            The search for keypoints works by searching for the argmin (non-inverted) or
            argmax (inverted) in each channel. This parameters contains the maximum (non-inverted)
            or minimum (inverted) value to accept in order to view a hit as a keypoint.
            Use None to use no min/max.

        nb_channels : None or int, optional
            Number of channels of the image on which the keypoints are placed.
            Some keypoint augmenters require that information.
            If set to None, the keypoint's shape will be set
            to ``(height, width)``, otherwise ``(height, width, nb_channels)``.

        Returns
        -------
        imgaug.KeypointsOnImage
            The extracted keypoints.

        """
        do_assert(len(distance_maps.shape) == 3)
        height, width, nb_keypoints = distance_maps.shape

        drop_if_not_found = False
        if if_not_found_coords is None:
            drop_if_not_found = True
            if_not_found_x = -1
            if_not_found_y = -1
        elif isinstance(if_not_found_coords, (tuple, list)):
            do_assert(len(if_not_found_coords) == 2)
            if_not_found_x = if_not_found_coords[0]
            if_not_found_y = if_not_found_coords[1]
        elif isinstance(if_not_found_coords, dict):
            if_not_found_x = if_not_found_coords["x"]
            if_not_found_y = if_not_found_coords["y"]
        else:
            raise Exception("Expected if_not_found_coords to be None or tuple or list or dict, got %s." % (
                type(if_not_found_coords),))

        keypoints = []
        for i in sm.xrange(nb_keypoints):
            # TODO introduce voting here among all distance values that have min/max values
            if inverted:
                hitidx_flat = np.argmax(distance_maps[..., i])
            else:
                hitidx_flat = np.argmin(distance_maps[..., i])
            hitidx_ndim = np.unravel_index(hitidx_flat, (height, width))
            if not inverted and threshold is not None:
                found = (distance_maps[hitidx_ndim[0], hitidx_ndim[1], i] < threshold)
            elif inverted and threshold is not None:
                found = (distance_maps[hitidx_ndim[0], hitidx_ndim[1], i] >= threshold)
            else:
                found = True
            if found:
                keypoints.append(Keypoint(x=hitidx_ndim[1], y=hitidx_ndim[0]))
            else:
                if drop_if_not_found:
                    pass  # dont add the keypoint to the result list, i.e. drop it
                else:
                    keypoints.append(Keypoint(x=if_not_found_x, y=if_not_found_y))

        out_shape = (height, width)
        if nb_channels is not None:
            out_shape += (nb_channels,)
        return KeypointsOnImage(keypoints, shape=out_shape)

    def copy(self):
        """
        Create a shallow copy of the KeypointsOnImage object.

        Returns
        -------
        imgaug.KeypointsOnImage
            Shallow copy.

        """
        return copy.copy(self)

    def deepcopy(self):
        """
        Create a deep copy of the KeypointsOnImage object.

        Returns
        -------
        imgaug.KeypointsOnImage
            Deep copy.

        """
        # for some reason deepcopy is way slower here than manual copy
        kps = [Keypoint(x=kp.x, y=kp.y) for kp in self.keypoints]
        return KeypointsOnImage(kps, tuple(self.shape))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "KeypointsOnImage(%s, shape=%s)" % (str(self.keypoints), self.shape)


# TODO functions: square(), to_aspect_ratio(), contains_point()
class BoundingBox(object):
    """
    Class representing bounding boxes.

    Each bounding box is parameterized by its top left and bottom right corners. Both are given
    as x and y-coordinates. The corners are intended to lie inside the bounding box area.
    As a result, a bounding box that lies completely inside the image but has maximum extensions
    would have coordinates ``(0.0, 0.0)`` and ``(W - epsilon, H - epsilon)``. Note that coordinates
    are saved internally as floats.

    Parameters
    ----------
    x1 : number
        X-coordinate of the top left of the bounding box.

    y1 : number
        Y-coordinate of the top left of the bounding box.

    x2 : number
        X-coordinate of the bottom right of the bounding box.

    y2 : number
        Y-coordinate of the bottom right of the bounding box.

    label : None or str, optional
        Label of the bounding box, e.g. a string representing the class.

    """

    def __init__(self, x1, y1, x2, y2, label=None):
        """Create a new BoundingBox instance."""
        if x1 > x2:
            x2, x1 = x1, x2
        do_assert(x2 >= x1)
        if y1 > y2:
            y2, y1 = y1, y2
        do_assert(y2 >= y1)

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label

    @property
    def x1_int(self):
        """
        Return the x-coordinate of the top left corner as an integer.

        Returns
        -------
        int
            X-coordinate of the top left corner, rounded to the closest integer.

        """
        return int(np.round(self.x1))  # use numpy's round to have consistent behaviour between python versions

    @property
    def y1_int(self):
        """
        Return the y-coordinate of the top left corner as an integer.

        Returns
        -------
        int
            Y-coordinate of the top left corner, rounded to the closest integer.

        """
        return int(np.round(self.y1))  # use numpy's round to have consistent behaviour between python versions

    @property
    def x2_int(self):
        """
        Return the x-coordinate of the bottom left corner as an integer.

        Returns
        -------
        int
            X-coordinate of the bottom left corner, rounded to the closest integer.

        """
        return int(np.round(self.x2))  # use numpy's round to have consistent behaviour between python versions

    @property
    def y2_int(self):
        """
        Return the y-coordinate of the bottom left corner as an integer.

        Returns
        -------
        int
            Y-coordinate of the bottom left corner, rounded to the closest integer.

        """
        return int(np.round(self.y2))  # use numpy's round to have consistent behaviour between python versions

    @property
    def height(self):
        """
        Estimate the height of the bounding box.

        Returns
        -------
        number
            Height of the bounding box.

        """
        return self.y2 - self.y1

    @property
    def width(self):
        """
        Estimate the width of the bounding box.

        Returns
        -------
        number
            Width of the bounding box.

        """
        return self.x2 - self.x1

    @property
    def center_x(self):
        """
        Estimate the x-coordinate of the center point of the bounding box.

        Returns
        -------
        number
            X-coordinate of the center point of the bounding box.

        """
        return self.x1 + self.width/2

    @property
    def center_y(self):
        """
        Estimate the y-coordinate of the center point of the bounding box.

        Returns
        -------
        number
            Y-coordinate of the center point of the bounding box.

        """
        return self.y1 + self.height/2

    @property
    def area(self):
        """
        Estimate the area of the bounding box.

        Returns
        -------
        number
            Area of the bounding box, i.e. `height * width`.

        """
        return self.height * self.width

    def contains(self, other):
        """
        Estimate whether the bounding box contains a point.

        Parameters
        ----------
        other : imgaug.Keypoint
            Point to check for.

        Returns
        -------
        bool
            True if the point is contained in the bounding box, False otherwise.

        """
        x, y = other.x, other.y
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def project(self, from_shape, to_shape):
        """
        Project the bounding box onto a new position on a new image.

        E.g. if the bounding box is on its original image at
        x1=(10 of 100 pixels) and y1=(20 of 100 pixels) and is projected onto
        a new image with size (width=200, height=200), its new position will
        be (x1=20, y1=40). (Analogous for x2/y2.)

        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).

        Parameters
        ----------
        from_shape : tuple of int
            Shape of the original image. (Before resize.)

        to_shape : tuple of int
            Shape of the new image. (After resize.)

        Returns
        -------
        out : imgaug.BoundingBox
            BoundingBox object with new coordinates.

        """
        if from_shape[0:2] == to_shape[0:2]:
            return self.copy()
        else:
            from_height, from_width = from_shape[0:2]
            to_height, to_width = to_shape[0:2]
            do_assert(from_height > 0)
            do_assert(from_width > 0)
            do_assert(to_height > 0)
            do_assert(to_width > 0)
            x1 = (self.x1 / from_width) * to_width
            y1 = (self.y1 / from_height) * to_height
            x2 = (self.x2 / from_width) * to_width
            y2 = (self.y2 / from_height) * to_height
            return self.copy(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=self.label
            )

    def extend(self, all_sides=0, top=0, right=0, bottom=0, left=0):
        """
        Extend the size of the bounding box along its sides.

        Parameters
        ----------
        all_sides : number, optional
            Value by which to extend the bounding box size along all sides.

        top : number, optional
            Value by which to extend the bounding box size along its top side.

        right : number, optional
            Value by which to extend the bounding box size along its right side.

        bottom : number, optional
            Value by which to extend the bounding box size along its bottom side.

        left : number, optional
            Value by which to extend the bounding box size along its left side.

        Returns
        -------
        imgaug.BoundingBox
            Extended bounding box.

        """
        return BoundingBox(
            x1=self.x1 - all_sides - left,
            x2=self.x2 + all_sides + right,
            y1=self.y1 - all_sides - top,
            y2=self.y2 + all_sides + bottom
        )

    def intersection(self, other, default=None):
        """
        Compute the intersection bounding box of this bounding box and another one.

        Note that in extreme cases, the intersection can be a single point, meaning that the intersection bounding box
        will exist, but then also has a height and width of zero.

        Parameters
        ----------
        other : imgaug.BoundingBox
            Other bounding box with which to generate the intersection.

        default : any, optional
            Default value to return if there is no intersection.

        Returns
        -------
        imgaug.BoundingBox or any
            Intersection bounding box of the two bounding boxes if there is an intersection.
            If there is no intersection, the default value will be returned, which can by anything.

        """
        x1_i = max(self.x1, other.x1)
        y1_i = max(self.y1, other.y1)
        x2_i = min(self.x2, other.x2)
        y2_i = min(self.y2, other.y2)
        if x1_i > x2_i or y1_i > y2_i:
            return default
        else:
            return BoundingBox(x1=x1_i, y1=y1_i, x2=x2_i, y2=y2_i)

    def union(self, other):
        """
        Compute the union bounding box of this bounding box and another one.

        This is equivalent to drawing a bounding box around all corners points of both
        bounding boxes.

        Parameters
        ----------
        other : imgaug.BoundingBox
            Other bounding box with which to generate the union.

        Returns
        -------
        imgaug.BoundingBox
            Union bounding box of the two bounding boxes.

        """
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def iou(self, other):
        """
        Compute the IoU of this bounding box with another one.

        IoU is the intersection over union, defined as::

            ``area(intersection(A, B)) / area(union(A, B))``
            ``= area(intersection(A, B)) / (area(A) + area(B) - area(intersection(A, B)))``

        Parameters
        ----------
        other : imgaug.BoundingBox
            Other bounding box with which to compare.

        Returns
        -------
        float
            IoU between the two bounding boxes.

        """
        inters = self.intersection(other)
        if inters is None:
            return 0.0
        else:
            area_union = self.area + other.area - inters.area
            return inters.area / area_union if area_union > 0 else 0.0

    def is_fully_within_image(self, image):
        """
        Estimate whether the bounding box is fully inside the image area.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape
            and must contain at least two integers.

        Returns
        -------
        bool
            True if the bounding box is fully inside the image area. False otherwise.

        """
        if isinstance(image, tuple):
            shape = image
        else:
            shape = image.shape
        height, width = shape[0:2]
        return self.x1 >= 0 and self.x2 < width and self.y1 >= 0 and self.y2 < height

    def is_partly_within_image(self, image):
        """
        Estimate whether the bounding box is at least partially inside the image area.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape
            and must contain at least two integers.

        Returns
        -------
        bool
            True if the bounding box is at least partially inside the image area. False otherwise.

        """
        if isinstance(image, tuple):
            shape = image
        else:
            shape = image.shape
        height, width = shape[0:2]
        eps = np.finfo(np.float32).eps
        img_bb = BoundingBox(x1=0, x2=width-eps, y1=0, y2=height-eps)
        return self.intersection(img_bb) is not None

    def is_out_of_image(self, image, fully=True, partly=False):
        """
        Estimate whether the bounding box is partially or fully outside of the image area.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use. If an ndarray, its shape will be used. If a tuple, it is
            assumed to represent the image shape and must contain at least two integers.

        fully : bool, optional
            Whether to return True if the bounding box is fully outside fo the image area.

        partly : bool, optional
            Whether to return True if the bounding box is at least partially outside fo the
            image area.

        Returns
        -------
        bool
            True if the bounding box is partially/fully outside of the image area, depending
            on defined parameters. False otherwise.

        """
        if self.is_fully_within_image(image):
            return False
        elif self.is_partly_within_image(image):
            return partly
        else:
            return fully

    def cut_out_of_image(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("BoundingBox.cut_out_of_image() is deprecated. Use "
                                         "BoundingBox.clip_out_of_image() instead. It has the "
                                         "exactly same interface (simple renaming)."))
        return self.clip_out_of_image(*args, **kwargs)

    def clip_out_of_image(self, image):
        """
        Clip off all parts of the bounding box that are outside of the image.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use for the clipping of the bounding box.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape and must contain at least two integers.

        Returns
        -------
        result : imgaug.BoundingBox
            Bounding box, clipped to fall within the image dimensions.

        """
        if isinstance(image, tuple):
            shape = image
        else:
            shape = image.shape

        height, width = shape[0:2]
        do_assert(height > 0)
        do_assert(width > 0)

        eps = np.finfo(np.float32).eps
        x1 = np.clip(self.x1, 0, width - eps)
        x2 = np.clip(self.x2, 0, width - eps)
        y1 = np.clip(self.y1, 0, height - eps)
        y2 = np.clip(self.y2, 0, height - eps)

        return self.copy(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label=self.label
        )

    # TODO convert this to x/y params?
    def shift(self, top=None, right=None, bottom=None, left=None):
        """
        Shift the bounding box from one or more image sides, i.e. move it on the x/y-axis.

        Parameters
        ----------
        top : None or int, optional
            Amount of pixels by which to shift the bounding box from the top.

        right : None or int, optional
            Amount of pixels by which to shift the bounding box from the right.

        bottom : None or int, optional
            Amount of pixels by which to shift the bounding box from the bottom.

        left : None or int, optional
            Amount of pixels by which to shift the bounding box from the left.

        Returns
        -------
        result : imgaug.BoundingBox
            Shifted bounding box.

        """
        top = top if top is not None else 0
        right = right if right is not None else 0
        bottom = bottom if bottom is not None else 0
        left = left if left is not None else 0
        return self.copy(
            x1=self.x1+left-right,
            x2=self.x2+left-right,
            y1=self.y1+top-bottom,
            y2=self.y2+top-bottom
        )

    # TODO add explicit test for zero-sized BBs (worked when tested by hand)
    def draw_on_image(self, image, color=(0, 255, 0), alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False):
        """
        Draw the bounding box on an image.

        Parameters
        ----------
        image : (H,W,C) ndarray(uint8)
            The image onto which to draw the bounding box.

        color : iterable of int, optional
            The color to use, corresponding to the channel layout of the image. Usually RGB.

        alpha : float, optional
            The transparency of the drawn bounding box, where 1.0 denotes no transparency and
            0.0 is invisible.

        thickness : int, optional
            The thickness of the bounding box in pixels. If the value is larger than 1, then
            additional pixels will be added around the bounding box (i.e. extension towards the
            outside).

        copy : bool, optional
            Whether to copy the input image or change it in-place.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the bounding box is partially/fully outside of the
            image. If set to False, no error will be raised and only the parts inside the image
            will be drawn.

        Returns
        -------
        result : (H,W,C) ndarray(uint8)
            Image with bounding box drawn on it.

        """
        if raise_if_out_of_image and self.is_out_of_image(image):
            raise Exception("Cannot draw bounding box x1=%.8f, y1=%.8f, x2=%.8f, y2=%.8f on image with shape %s." % (
                self.x1, self.y1, self.x2, self.y2, image.shape))

        result = np.copy(image) if copy else image

        if isinstance(color, (tuple, list)):
            color = np.uint8(color)

        for i in range(thickness):
            y1, y2, x1, x2 = self.y1_int, self.y2_int, self.x1_int, self.x2_int

            # When y values get into the range (H-0.5, H), the *_int functions round them to H.
            # That is technically sensible, but in the case of drawing means that the border lies
            # just barely outside of the image, making the border disappear, even though the BB
            # is fully inside the image. Here we correct for that because of beauty reasons.
            # Same is the case for x coordinates.
            if self.is_fully_within_image(image):
                y1 = np.clip(y1, 0, image.shape[0]-1)
                y2 = np.clip(y2, 0, image.shape[0]-1)
                x1 = np.clip(x1, 0, image.shape[1]-1)
                x2 = np.clip(x2, 0, image.shape[1]-1)

            y = [y1-i, y1-i, y2+i, y2+i]
            x = [x1-i, x2+i, x2+i, x1-i]
            rr, cc = skimage.draw.polygon_perimeter(y, x, shape=result.shape)
            if alpha >= 0.99:
                result[rr, cc, :] = color
            else:
                if is_float_array(result):
                    result[rr, cc, :] = (1 - alpha) * result[rr, cc, :] + alpha * color
                    result = np.clip(result, 0, 255)
                else:
                    input_dtype = result.dtype
                    result = result.astype(np.float32)
                    result[rr, cc, :] = (1 - alpha) * result[rr, cc, :] + alpha * color
                    result = np.clip(result, 0, 255).astype(input_dtype)

        return result

    def extract_from_image(self, image, prevent_zero_size=True):
        """
        Extract the image pixels within the bounding box.

        This function will zero-pad the image if the bounding box is partially/fully outside of
        the image.

        Parameters
        ----------
        image : (H,W) ndarray or (H,W,C) ndarray
            The image from which to extract the pixels within the bounding box.

        prevent_zero_size : bool, optional
            Whether to prevent height or width of the extracted image from becoming zero.
            If this is set to True and height or width of the bounding box is below 1, the height/width will
            be increased to 1. This can be useful to prevent problems, e.g. with image saving or plotting.
            If it is set to False, images will be returned as ``(H', W')`` or ``(H', W', 3)`` with ``H`` or
            ``W`` potentially being 0.

        Returns
        -------
        image : (H',W') ndarray or (H',W',C) ndarray
            Pixels within the bounding box. Zero-padded if the bounding box is partially/fully
            outside of the image. If prevent_zero_size is activated, it is guarantueed that ``H'>0``
            and ``W'>0``, otherwise only ``H'>=0`` and ``W'>=0``.

        """
        pad_top = 0
        pad_right = 0
        pad_bottom = 0
        pad_left = 0

        height, width = image.shape[0], image.shape[1]
        x1, x2, y1, y2 = self.x1_int, self.x2_int, self.y1_int, self.y2_int

        # When y values get into the range (H-0.5, H), the *_int functions round them to H.
        # That is technically sensible, but in the case of extraction leads to a black border,
        # which is both ugly and unexpected after calling cut_out_of_image(). Here we correct for
        # that because of beauty reasons.
        # Same is the case for x coordinates.
        if self.is_fully_within_image(image):
            y1 = np.clip(y1, 0, image.shape[0]-1)
            y2 = np.clip(y2, 0, image.shape[0]-1)
            x1 = np.clip(x1, 0, image.shape[1]-1)
            x2 = np.clip(x2, 0, image.shape[1]-1)

        # TODO add test
        if prevent_zero_size:
            if abs(x2 - x1) < 1:
                x2 = x1 + 1
            if abs(y2 - y1) < 1:
                y2 = y1 + 1

        # if the bb is outside of the image area, the following pads the image
        # first with black pixels until the bb is inside the image
        # and only then extracts the image area
        # TODO probably more efficient to initialize an array of zeros
        # and copy only the portions of the bb into that array that are
        # natively inside the image area
        if x1 < 0:
            pad_left = abs(x1)
            x2 = x2 + abs(x1)
            x1 = 0
        if y1 < 0:
            pad_top = abs(y1)
            y2 = y2 + abs(y1)
            y1 = 0
        if x2 >= width:
            pad_right = x2 - (width - 1)
        if y2 >= height:
            pad_bottom = y2 - (height - 1)

        if any([val > 0 for val in [pad_top, pad_right, pad_bottom, pad_left]]):
            if len(image.shape) == 2:
                image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            else:
                image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")

        return image[y1:y2, x1:x2]

    # TODO also add to_heatmap
    # TODO add this to BoundingBoxesOnImage
    def to_keypoints(self):
        """
        Convert the corners of the bounding box to keypoints (clockwise, starting at top left).

        Returns
        -------
        list of imgaug.Keypoint
            Corners of the bounding box as keypoints.

        """
        return [
            Keypoint(x=self.x1, y=self.y1),
            Keypoint(x=self.x2, y=self.y1),
            Keypoint(x=self.x2, y=self.y2),
            Keypoint(x=self.x1, y=self.y2)
        ]

    def copy(self, x1=None, y1=None, x2=None, y2=None, label=None):
        """
        Create a shallow copy of the BoundingBox object.

        Parameters
        ----------
        x1 : None or number
            If not None, then the x1 coordinate of the copied object will be set to this value.

        y1 : None or number
            If not None, then the y1 coordinate of the copied object will be set to this value.

        x2 : None or number
            If not None, then the x2 coordinate of the copied object will be set to this value.

        y2 : None or number
            If not None, then the y2 coordinate of the copied object will be set to this value.

        label : None or string
            If not None, then the label of the copied object will be set to this value.

        Returns
        -------
        imgaug.BoundingBox
            Shallow copy.

        """
        return BoundingBox(
            x1=self.x1 if x1 is None else x1,
            x2=self.x2 if x2 is None else x2,
            y1=self.y1 if y1 is None else y1,
            y2=self.y2 if y2 is None else y2,
            label=self.label if label is None else label
        )

    def deepcopy(self, x1=None, y1=None, x2=None, y2=None, label=None):
        """
        Create a deep copy of the BoundingBox object.

        Parameters
        ----------
        x1 : None or number
            If not None, then the x1 coordinate of the copied object will be set to this value.

        y1 : None or number
            If not None, then the y1 coordinate of the copied object will be set to this value.

        x2 : None or number
            If not None, then the x2 coordinate of the copied object will be set to this value.

        y2 : None or number
            If not None, then the y2 coordinate of the copied object will be set to this value.

        label : None or string
            If not None, then the label of the copied object will be set to this value.

        Returns
        -------
        imgaug.BoundingBox
            Deep copy.

        """
        return self.copy(x1=x1, y1=y1, x2=x2, y2=y2, label=label)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox(x1=%.4f, y1=%.4f, x2=%.4f, y2=%.4f, label=%s)" % (
            self.x1, self.y1, self.x2, self.y2, self.label)


class BoundingBoxesOnImage(object):
    """
    Object that represents all bounding boxes on a single image.

    Parameters
    ----------
    bounding_boxes : list of imgaug.BoundingBox
        List of bounding boxes on the image.

    shape : tuple of int
        The shape of the image on which the bounding boxes are placed.

    Examples
    --------
    >>> image = np.zeros((100, 100))
    >>> bbs = [
    >>>     BoundingBox(x1=10, y1=20, x2=20, y2=30),
    >>>     BoundingBox(x1=25, y1=50, x2=30, y2=70)
    >>> ]
    >>> bbs_oi = BoundingBoxesOnImage(bbs, shape=image.shape)

    """
    def __init__(self, bounding_boxes, shape):
        self.bounding_boxes = bounding_boxes
        if is_np_array(shape):
            self.shape = shape.shape
        else:
            do_assert(isinstance(shape, (tuple, list)))
            self.shape = tuple(shape)

    # TODO remove this? here it is image height at BoundingBox it is bounding box height
    @property
    def height(self):
        """
        Get the height of the image on which the bounding boxes fall.

        Returns
        -------
        int
            Image height.

        """
        return self.shape[0]

    # TODO remove this? here it is image width at BoundingBox it is bounding box width
    @property
    def width(self):
        """
        Get the width of the image on which the bounding boxes fall.

        Returns
        -------
        int
            Image width.

        """
        return self.shape[1]

    @property
    def empty(self):
        """
        Returns whether this object contains zero bounding boxes.

        Returns
        -------
        bool
            True if this object contains zero bounding boxes.

        """
        return len(self.bounding_boxes) == 0

    def on(self, image):
        """
        Project bounding boxes from one image to a new one.

        Parameters
        ----------
        image : ndarray or tuple of int
            New image onto which the bounding boxes are to be projected.
            May also simply be that new image's shape tuple.

        Returns
        -------
        bounding_boxes : imgaug.BoundingBoxesOnImage
            Object containing all projected bounding boxes.

        """
        if is_np_array(image):
            shape = image.shape
        else:
            shape = image

        if shape[0:2] == self.shape[0:2]:
            return self.deepcopy()
        else:
            bounding_boxes = [bb.project(self.shape, shape) for bb in self.bounding_boxes]
            return BoundingBoxesOnImage(bounding_boxes, shape)

    @classmethod
    def from_xyxy_array(cls, xyxy, shape):
        """
        Convert an (N,4) ndarray to a BoundingBoxesOnImage object.

        This is the inverse of :func:`imgaug.BoundingBoxesOnImage.to_xyxy_array`.

        Parameters
        ----------
        xyxy : (N,4) ndarray
            Array containing the corner coordinates (top-left, bottom-right) of ``N`` bounding boxes
            in the form ``(x1, y1, x2, y2)``. Should usually be of dtype ``float32``.

        shape : tuple of int
            Shape of the image on which the bounding boxes are placed.
            Should usually be ``(H, W, C)`` or ``(H, W)``.

        Returns
        -------
        imgaug.BoundingBoxesOnImage
            Object containing a list of BoundingBox objects following the provided corner coordinates.

        """
        do_assert(xyxy.shape[1] == 4, "Expected input array of shape (N, 4), got shape %s." % (xyxy.shape,))

        boxes = [BoundingBox(*row) for row in xyxy]

        return cls(boxes, shape)

    def to_xyxy_array(self, dtype=np.float32):
        """
        Convert the BoundingBoxesOnImage object to an (N,4) ndarray.

        This is the inverse of :func:`imgaug.BoundingBoxesOnImage.from_xyxy_array`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            Desired output datatype of the ndarray.

        Returns
        -------
        ndarray
            (N,4) ndarray array, where ``N`` denotes the number of bounding boxes and ``4`` denotes the
            top-left and bottom-right bounding box corner coordinates in form ``(x1, y1, x2, y2)``.

        """
        xyxy_array = np.zeros((len(self.bounding_boxes), 4), dtype=np.float32)

        for i, box in enumerate(self.bounding_boxes):
            xyxy_array[i] = [box.x1, box.y1, box.x2, box.y2]

        return xyxy_array.astype(dtype)

    def draw_on_image(self, image, color=(0, 255, 0), alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False):
        """
        Draw all bounding boxes onto a given image.

        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the bounding boxes.
            This image should usually have the same shape as
            set in BoundingBoxesOnImage.shape.

        color : int or list of int or tuple of int or (3,) ndarray, optional
            The RGB color of all bounding boxes. If a single int ``C``, then that is
            equivalent to ``(C,C,C)``.

        alpha : float, optional
            Alpha/transparency of the bounding box.

        thickness : int, optional
            Thickness in pixels.

        copy : bool, optional
            Whether to copy the image before drawing the points.

        raise_if_out_of_image : bool, optional
            Whether to raise an exception if any bounding box is outside of the image.

        Returns
        -------
        image : (H,W,3) ndarray
            Image with drawn bounding boxes.

        """
        # TODO improve efficiency here by copying only once
        for bb in self.bounding_boxes:
            image = bb.draw_on_image(
                image,
                color=color,
                alpha=alpha,
                thickness=thickness,
                copy=copy,
                raise_if_out_of_image=raise_if_out_of_image
            )

        return image

    def remove_out_of_image(self, fully=True, partly=False):
        """
        Remove all bounding boxes that are fully or partially outside of the image.

        Parameters
        ----------
        fully : bool, optional
            Whether to remove bounding boxes that are fully outside of the image.

        partly : bool, optional
            Whether to remove bounding boxes that are partially outside of the image.

        Returns
        -------
        imgaug.BoundingBoxesOnImage
            Reduced set of bounding boxes, with those that were fully/partially outside of
            the image removed.

        """
        bbs_clean = [bb for bb in self.bounding_boxes
                     if not bb.is_out_of_image(self.shape, fully=fully, partly=partly)]
        return BoundingBoxesOnImage(bbs_clean, shape=self.shape)

    def cut_out_of_image(self):
        warnings.warn(DeprecationWarning("BoundingBoxesOnImage.cut_out_of_image() is deprecated."
                                         "Use BoundingBoxesOnImage.clip_out_of_image() instead. It "
                                         "has the exactly same interface (simple renaming)."))
        return self.clip_out_of_image()

    def clip_out_of_image(self):
        """
        Clip off all parts from all bounding boxes that are outside of the image.

        Returns
        -------
        imgaug.BoundingBoxesOnImage
            Bounding boxes, clipped to fall within the image dimensions.

        """
        bbs_cut = [bb.clip_out_of_image(self.shape)
                   for bb in self.bounding_boxes if bb.is_partly_within_image(self.shape)]
        return BoundingBoxesOnImage(bbs_cut, shape=self.shape)

    def shift(self, top=None, right=None, bottom=None, left=None):
        """
        Shift all bounding boxes from one or more image sides, i.e. move them on the x/y-axis.

        Parameters
        ----------
        top : None or int, optional
            Amount of pixels by which to shift all bounding boxes from the top.

        right : None or int, optional
            Amount of pixels by which to shift all bounding boxes from the right.

        bottom : None or int, optional
            Amount of pixels by which to shift all bounding boxes from the bottom.

        left : None or int, optional
            Amount of pixels by which to shift all bounding boxes from the left.

        Returns
        -------
        imgaug.BoundingBoxesOnImage
            Shifted bounding boxes.

        """
        bbs_new = [bb.shift(top=top, right=right, bottom=bottom, left=left) for bb in self.bounding_boxes]
        return BoundingBoxesOnImage(bbs_new, shape=self.shape)

    def copy(self):
        """
        Create a shallow copy of the BoundingBoxesOnImage object.

        Returns
        -------
        imgaug.BoundingBoxesOnImage
            Shallow copy.

        """
        return copy.copy(self)

    def deepcopy(self):
        """
        Create a deep copy of the BoundingBoxesOnImage object.

        Returns
        -------
        imgaug.BoundingBoxesOnImage
            Deep copy.

        """
        # Manual copy is far faster than deepcopy for KeypointsOnImage,
        # so use manual copy here too
        bbs = [bb.deepcopy() for bb in self.bounding_boxes]
        return BoundingBoxesOnImage(bbs, tuple(self.shape))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBoxesOnImage(%s, shape=%s)" % (str(self.bounding_boxes), self.shape)


# TODO somehow merge with BoundingBox
# TODO add functions: simplify() (eg via shapely.ops.simplify()),
# extend(all_sides=0, top=0, right=0, bottom=0, left=0),
# intersection(other, default=None), union(other), iou(other), to_heatmap, to_mask
class Polygon(object):
    """
    Class representing polygons.

    Each polygon is parameterized by its corner points, given as absolute x- and y-coordinates
    with sub-pixel accuracy.

    Parameters
    ----------
    exterior : list of imgaug.Keypoint or list of tuple of float or (N,2) ndarray
        List of points defining the polygon. May be either a list of Keypoint objects or a list of tuples in xy-form
        or a numpy array of shape (N,2) for N points in xy-form.
        All coordinates are expected to be the absolute coordinates in the image, given as floats, e.g. x=10.7
        and y=3.4 for a point at coordinates (10.7, 3.4). Their order is expected to be clock-wise. They are expected
        to not be closed (i.e. first and last coordinate differ).

    label : None or str, optional
        Label of the polygon, e.g. a string representing the class.

    """

    def __init__(self, exterior, label=None):
        """Create a new Polygon instance."""
        if isinstance(exterior, list):
            if not exterior:
                # for empty lists, make sure that the shape is (0, 2) and not (0,) as that is also expected when the
                # input is a numpy array
                self.exterior = np.zeros((0, 2), dtype=np.float32)
            elif isinstance(exterior[0], Keypoint):
                # list of Keypoint
                self.exterior = np.float32([[point.x, point.y] for point in exterior])
            else:
                # list of tuples (x, y)
                self.exterior = np.float32([[point[0], point[1]] for point in exterior])
        else:
            do_assert(is_np_array(exterior))
            do_assert(exterior.ndim == 2)
            do_assert(exterior.shape[1] == 2)
            self.exterior = np.float32(exterior)

        # Remove last point if it is essentially the same as the first point (polygons are always assumed to be
        # closed anyways). This also prevents problems with shapely, which seems to add the last point automatically.
        if len(self.exterior) >= 2 and np.allclose(self.exterior[0, :], self.exterior[-1, :]):
            self.exterior = self.exterior[:-1]

        self.label = label

    @property
    def xx(self):
        """
        Return the x-coordinates of all points in the exterior.

        Returns
        -------
        (N,2) ndarray
            X-coordinates of all points in the exterior as a float32 ndarray.

        """
        return self.exterior[:, 0]

    @property
    def yy(self):
        """
        Return the y-coordinates of all points in the exterior.

        Returns
        -------
        (N,2) ndarray
            Y-coordinates of all points in the exterior as a float32 ndarray.

        """
        return self.exterior[:, 1]

    @property
    def xx_int(self):
        """
        Return the x-coordinates of all points in the exterior, rounded to the closest integer value.

        Returns
        -------
        (N,2) ndarray
            X-coordinates of all points in the exterior, rounded to the closest integer value.
            Result dtype is int32.

        """
        return np.int32(np.round(self.xx))

    @property
    def yy_int(self):
        """
        Return the y-coordinates of all points in the exterior, rounded to the closest integer value.

        Returns
        -------
        (N,2) ndarray
            Y-coordinates of all points in the exterior, rounded to the closest integer value.
            Result dtype is int32.

        """
        return np.int32(np.round(self.yy))

    @property
    def is_valid(self):
        """
        Estimate whether the polygon has a valid shape.

        To to be considered valid, the polygons must be made up of at least 3 points and have concave shape.
        Multiple consecutive points are allowed to have the same coordinates.

        Returns
        -------
        bool
            True if polygon has at least 3 points and is concave, otherwise False.

        """
        if len(self.exterior) < 3:
            return False
        return self.to_shapely_polygon().is_valid

    @property
    def area(self):
        """
        Estimate the area of the polygon.

        Returns
        -------
        number
            Area of the polygon.

        """
        if len(self.exterior) < 3:
            raise Exception("Cannot compute the polygon's area because it contains less than three points.")
        poly = self.to_shapely_polygon()
        return poly.area

    def project(self, from_shape, to_shape):
        """
        Project the polygon onto an image with different shape.

        The relative coordinates of all points remain the same.
        E.g. a point at (x=20, y=20) on an image (width=100, height=200) will be
        projected on a new image (width=200, height=100) to (x=40, y=10).

        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).

        Parameters
        ----------
        from_shape : tuple of int
            Shape of the original image. (Before resize.)

        to_shape : tuple of int
            Shape of the new image. (After resize.)

        Returns
        -------
        imgaug.Polygon
            Polygon object with new coordinates.

        """
        if from_shape[0:2] == to_shape[0:2]:
            return self.copy()
        exterior = [Keypoint(x=x, y=y).project(from_shape, to_shape) for x, y in self.exterior]
        return self.copy(exterior=exterior)

    def find_closest_point_index(self, x, y, return_distance=False):
        """
        Find the index of the point within the exterior that is closest to the given coordinates.

        "Closeness" is here defined based on euclidean distance.
        This method will raise an AssertionError if the exterior contains no points.

        Parameters
        ----------
        x : number
            X-coordinate around which to search for close points.

        y : number
            Y-coordinate around which to search for close points.

        return_distance : bool, optional
            Whether to also return the distance of the closest point.

        Returns
        -------
        int
            Index of the closest point.

        number
            Euclidean distance to the closest point.
            This value is only returned if `return_distance` was set to True.

        """
        do_assert(len(self.exterior) > 0)
        distances = []
        for x2, y2 in self.exterior:
            d = (x2 - x) ** 2 + (y2 - y) ** 2
            distances.append(d)
        distances = np.sqrt(distances)
        closest_idx = np.argmin(distances)
        if return_distance:
            return closest_idx, distances[closest_idx]
        return closest_idx

    def _compute_inside_image_point_mask(self, image):
        if isinstance(image, tuple):
            shape = image
        else:
            shape = image.shape
        h, w = shape[0:2]
        return np.logical_and(
            np.logical_and(0 <= self.exterior[:, 0], self.exterior[:, 0] < w),
            np.logical_and(0 <= self.exterior[:, 1], self.exterior[:, 1] < h)
        )

    # TODO keep this method? it is almost an alias for is_out_of_image()
    def is_fully_within_image(self, image):
        """
        Estimate whether the polygon is fully inside the image area.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape and must contain at least two integers.

        Returns
        -------
        bool
            True if the polygon is fully inside the image area.
            False otherwise.

        """
        return not self.is_out_of_image(image, fully=True, partly=True)

    # TODO keep this method? it is almost an alias for is_out_of_image()
    def is_partly_within_image(self, image):
        """
        Estimate whether the polygon is at least partially inside the image area.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape and must contain at least two integers.

        Returns
        -------
        bool
            True if the polygon is at least partially inside the image area.
            False otherwise.

        """
        return not self.is_out_of_image(image, fully=True, partly=False)

    def is_out_of_image(self, image, fully=True, partly=False):
        """
        Estimate whether the polygon is partially or fully outside of the image area.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape and must contain at least two integers.

        fully : bool, optional
            Whether to return True if the polygon is fully outside fo the image area.

        partly : bool, optional
            Whether to return True if the polygon is at least partially outside fo the image area.

        Returns
        -------
        bool
            True if the polygon is partially/fully outside of the image area, depending
            on defined parameters. False otherwise.

        """
        if len(self.exterior) == 0:
            raise Exception("Cannot determine whether the polygon is inside the image, because it contains no points.")
        inside = self._compute_inside_image_point_mask(image)
        nb_inside = sum(inside)
        if nb_inside == len(inside):
            return False
        elif nb_inside > 0:
            return partly
        else:
            return fully

    def cut_out_of_image(self, image):
        warnings.warn(DeprecationWarning("Polygon.cut_out_of_image() is deprecated. Use "
                                         "Polygon.clip_out_of_image() instead. It has the exactly "
                                         "same interface (simple renaming)."))
        return self.clip_out_of_image(image)

    def clip_out_of_image(self, image):
        """
        Cut off all parts of the polygon that are outside of the image.

        This operation may lead to new points being created.
        As a single polygon may be split into multiple new polygons, the result is a MultiPolygon.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use for the clipping of the polygon.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape and must contain at least two integers.

        Returns
        -------
        imgaug.MultiPolygon
            Polygon, clipped to fall within the image dimensions.
            Returned as MultiPolygon, because the clipping can split the polygon into multiple parts.

        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        # if fully out of image, clip everything away, nothing remaining
        if self.is_out_of_image(image, fully=True, partly=False):
            return MultiPolygon([])

        h, w = image.shape[0:2]
        poly_shapely = self.to_shapely_polygon()
        poly_image = shapely.geometry.Polygon([(0, 0), (w, 0), (w, h), (0, h)])
        multipoly_inter_shapely = poly_shapely.intersection(poly_image)
        if not isinstance(multipoly_inter_shapely, shapely.geometry.MultiPolygon):
            do_assert(isinstance(multipoly_inter_shapely, shapely.geometry.Polygon))
            multipoly_inter_shapely = shapely.geometry.MultiPolygon([multipoly_inter_shapely])

        polygons = []
        for poly_inter_shapely in multipoly_inter_shapely.geoms:
            polygons.append(Polygon.from_shapely(poly_inter_shapely, label=self.label))

        # shapely changes the order of points, we try here to preserve it as good as possible
        polygons_reordered = []
        for polygon in polygons:
            found = False
            for x, y in self.exterior:
                closest_idx, dist = polygon.find_closest_point_index(x=x, y=y, return_distance=True)
                if dist < 1e-6:
                    polygon_reordered = polygon.change_first_point_by_index(closest_idx)
                    polygons_reordered.append(polygon_reordered)
                    found = True
                    break
            do_assert(found)  # could only not find closest points if new polys are empty

        return MultiPolygon(polygons_reordered)

    def shift(self, top=None, right=None, bottom=None, left=None):
        """
        Shift the polygon from one or more image sides, i.e. move it on the x/y-axis.

        Parameters
        ----------
        top : None or int, optional
            Amount of pixels by which to shift the polygon from the top.

        right : None or int, optional
            Amount of pixels by which to shift the polygon from the right.

        bottom : None or int, optional
            Amount of pixels by which to shift the polygon from the bottom.

        left : None or int, optional
            Amount of pixels by which to shift the polygon from the left.

        Returns
        -------
        imgaug.Polygon
            Shifted polygon.

        """
        top = top if top is not None else 0
        right = right if right is not None else 0
        bottom = bottom if bottom is not None else 0
        left = left if left is not None else 0
        exterior = np.copy(self.exterior)
        exterior[:, 0] += (left - right)
        exterior[:, 1] += (top - bottom)
        return self.deepcopy(exterior=exterior)

    # TODO add boundary thickness
    def draw_on_image(self,
                      image,
                      color=(0, 255, 0), color_perimeter=(0, 128, 0),
                      alpha=0.5, alpha_perimeter=1.0,
                      raise_if_out_of_image=False):
        """
        Draw the polygon on an image.

        Parameters
        ----------
        image : (H,W,C) ndarray
            The image onto which to draw the polygon. Usually expected to be of dtype uint8, though other dtypes
            are also handled.

        color : iterable of int, optional
            The color to use for the polygon (excluding perimeter). Must correspond to the channel layout of the
            image. Usually RGB.

        color_perimeter : iterable of int, optional
            The color to use for the perimeter/border of the polygon. Must correspond to the channel layout of the
            image. Usually RGB.

        alpha : float, optional
            The transparency of the polygon (excluding the perimeter), where 1.0 denotes no transparency and 0.0 is
            invisible.

        alpha_perimeter : float, optional
            The transparency of the polygon's perimeter/border, where 1.0 denotes no transparency and 0.0 is
            invisible.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the polygon is partially/fully outside of the
            image. If set to False, no error will be raised and only the parts inside the image
            will be drawn.

        Returns
        -------
        result : (H,W,C) ndarray
            Image with polygon drawn on it. Result dtype is the same as the input dtype.

        """
        # TODO separate this into draw_face_on_image() and draw_border_on_image()

        if raise_if_out_of_image and self.is_out_of_image(image):
            raise Exception("Cannot draw polygon %s on image with shape %s." % (
                str(self), image.shape
            ))

        xx = self.xx_int
        yy = self.yy_int

        # TODO np.clip to image plane if is_fully_within_image(), similar to how it is done for bounding boxes

        # TODO improve efficiency by only drawing in rectangle that covers poly instead of drawing in the whole image
        # TODO for a rectangular polygon, the face coordinates include the top/left boundary but not the right/bottom
        # boundary. This may be unintuitive when not drawing the boundary. Maybe somehow remove the boundary
        # coordinates from the face coordinates after generating both?
        rr, cc = skimage.draw.polygon(yy, xx, shape=image.shape)
        rr_perimeter, cc_perimeter = skimage.draw.polygon_perimeter(yy, xx, shape=image.shape)

        params = (rr, cc, color, alpha)
        params_perimeter = (rr_perimeter, cc_perimeter, color_perimeter, alpha_perimeter)

        input_dtype = image.dtype
        result = image.astype(np.float32)

        for rr, cc, color, alpha in [params, params_perimeter]:
            color = np.float32(color)

            if alpha >= 0.99:
                result[rr, cc, :] = color
            elif alpha < 1e-4:
                pass  # invisible, do nothing
            else:
                result[rr, cc, :] = (1 - alpha) * result[rr, cc, :] + alpha * color

        if input_dtype.type == np.uint8:
            result = np.clip(result, 0, 255).astype(input_dtype)  # TODO make clipping more flexible
        else:
            result = result.astype(input_dtype)

        return result

    def extract_from_image(self, image):
        """
        Extract the image pixels within the polygon.

        This function will zero-pad the image if the polygon is partially/fully outside of
        the image.

        Parameters
        ----------
        image : (H,W) ndarray or (H,W,C) ndarray
            The image from which to extract the pixels within the polygon.

        Returns
        -------
        result : (H',W') ndarray or (H',W',C) ndarray
            Pixels within the polygon. Zero-padded if the polygon is partially/fully
            outside of the image.

        """
        do_assert(image.ndim in [2, 3])
        if len(self.exterior) <= 2:
            raise Exception("Polygon must be made up of at least 3 points to extract its area from an image.")

        bb = self.to_bounding_box()
        bb_area = bb.extract_from_image(image)
        if self.is_out_of_image(image, fully=True, partly=False):
            return bb_area

        xx = self.xx_int
        yy = self.yy_int
        xx_mask = xx - np.min(xx)
        yy_mask = yy - np.min(yy)
        height_mask = np.max(yy_mask)
        width_mask = np.max(xx_mask)

        rr_face, cc_face = skimage.draw.polygon(yy_mask, xx_mask, shape=(height_mask, width_mask))

        mask = np.zeros((height_mask, width_mask), dtype=np.bool)
        mask[rr_face, cc_face] = True

        if image.ndim == 3:
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, image.shape[2]))

        return bb_area * mask

    def change_first_point_by_coords(self, x, y, max_distance=1e-4):
        """
        Set the first point of the exterior to the given point based on its coordinates.

        If multiple points are found, the closest one will be picked.
        If no matching points are found, an exception is raised.

        Note: This method does *not* work in-place.

        Parameters
        ----------
        x : number
            X-coordinate of the point.

        y : number
            Y-coordinate of the point.

        max_distance : number
            Maximum distance past which possible matches are ignored.

        Returns
        -------
        imgaug.Polygon
            Copy of this polygon with the new point order.

        """
        if len(self.exterior) == 0:
            raise Exception("Cannot reorder polygon points, because it contains no points.")

        closest_idx, closest_dist = self.find_closest_point_index(x=x, y=y, return_distance=True)
        if max_distance is not None and closest_dist > max_distance:
            closest_point = self.exterior[closest_idx, :]
            raise Exception(
                "Closest found point (%.9f, %.9f) exceeds max_distance of %.9f exceeded" % (
                    closest_point[0], closest_point[1], closest_dist)
            )
        return self.change_first_point_by_index(closest_idx)

    def change_first_point_by_index(self, point_idx):
        """
        Set the first point of the exterior to the given point based on its index.

        Note: This method does *not* work in-place.

        Parameters
        ----------
        point_idx : int
            Index of the desired starting point.

        Returns
        -------
        imgaug.Polygon
            Copy of this polygon with the new point order.

        """
        do_assert(0 <= point_idx < len(self.exterior))
        if point_idx == 0:
            return self.deepcopy()
        exterior = np.concatenate(
            (self.exterior[point_idx:, :], self.exterior[:point_idx, :]),
            axis=0
        )
        return self.deepcopy(exterior=exterior)

    def to_shapely_polygon(self):
        """
        Convert this polygon to a Shapely polygon.

        Returns
        -------
        shapely.geometry.Polygon
            The Shapely polygon matching this polygon's exterior.

        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        return shapely.geometry.Polygon([(point[0], point[1]) for point in self.exterior])

    def to_shapely_line_string(self, closed=False, interpolate=0):
        """
        Convert this polygon to a Shapely LineString object.

        Parameters
        ----------
        closed : bool, optional
            Whether to return the line string with the last point being identical to the first point.

        interpolate : int, optional
            Number of points to interpolate between any pair of two consecutive points. These points are added
            to the final line string.

        Returns
        -------
        shapely.geometry.LineString
            The Shapely LineString matching the polygon's exterior.

        """
        return _convert_points_to_shapely_line_string(self.exterior, closed=closed, interpolate=interpolate)

    def to_bounding_box(self):
        """
        Convert this polygon to a bounding box tightly containing the whole polygon.

        Returns
        -------
        imgaug.BoundingBox
            The bounding box tightly containing the polygon.

        """
        xx = self.xx
        yy = self.yy
        return BoundingBox(x1=min(xx), x2=max(xx), y1=min(yy), y2=max(yy), label=self.label)

    @staticmethod
    def from_shapely(polygon_shapely, label=None):
        """
        Create a polygon from a Shapely polygon.

        Note: This will remove any holes in the Shapely polygon.

        Parameters
        ----------
        polygon_shapely : shapely.geometry.Polygon
             The shapely polygon.

        label : None or str, optional
            The label of the new polygon.

        Returns
        -------
        imgaug.Polygon
            A polygon with the same exterior as the Shapely polygon.

        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        do_assert(isinstance(polygon_shapely, shapely.geometry.Polygon))
        # polygon_shapely.exterior can be None if the polygon was instantiated without points
        if polygon_shapely.exterior is None or len(polygon_shapely.exterior.coords) == 0:
            return Polygon([], label=label)
        exterior = np.float32([[x, y] for (x, y) in polygon_shapely.exterior.coords])
        return Polygon(exterior, label=label)

    def exterior_almost_equals(self, other_polygon, max_distance=1e-6, interpolate=8):
        """
        Estimate whether the geometry of the exterior of this polygon and another polygon are comparable.

        The two exteriors can have different numbers of points, but any point randomly sampled on the exterior
        of one polygon should be close to the closest point on the exterior of the other polygon.

        Note that this method works approximately. One can come up with polygons with fairly different shapes that
        will still be estimated as equal by this method. In practice however this should be unlikely to be the case.
        The probability for something like that goes down as the interpolation parameter is increased.

        Parameters
        ----------
        other_polygon : imgaug.Polygon or (N,2) ndarray
            The other polygon with which to compare the exterior.
            If this is an ndarray, it is assumed to represent an exterior.
            It must then have dtype float32 and shape (N,2) with the second dimension denoting xy-coordinates.

        max_distance : number
            The maximum euclidean distance between a point on one polygon and the closest point on the other polygon.
            If the distance is exceeded for any such pair, the two exteriors are not viewed as equal.
            The points are other the points contained in the polygon's exterior ndarray or interpolated points
            between these.

        interpolate : int
            How many points to interpolate between the points of the polygon's exteriors.
            If this is set to zero, then only the points given by the polygon's exterior ndarrays will be used.
            Higher values make it less likely that unequal polygons are evaluated as equal.

        Returns
        -------
        bool
            Whether the two polygon's exteriors can be viewed as equal (approximate test).

        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        atol = max_distance

        ext_a = self.exterior
        ext_b = other_polygon.exterior if not is_np_array(other_polygon) else other_polygon
        len_a = len(ext_a)
        len_b = len(ext_b)

        if len_a == 0 and len_b == 0:
            return True
        elif len_a == 0 and len_b > 0:
            return False
        elif len_a > 0 and len_b == 0:
            return False

        # neither A nor B is zero-sized at this point

        # if A or B only contain points identical to the first point, merge them to one point
        if len_a > 1:
            if all([np.allclose(ext_a[0, :], ext_a[1 + i, :], rtol=0, atol=atol) for i in sm.xrange(len_a - 1)]):
                ext_a = ext_a[0:1, :]
                len_a = 1
        if len_b > 1:
            if all([np.allclose(ext_b[0, :], ext_b[1 + i, :], rtol=0, atol=atol) for i in sm.xrange(len_b - 1)]):
                ext_b = ext_b[0:1, :]
                len_b = 1

        # handle polygons that contain a single point
        if len_a == 1 and len_b == 1:
            return np.allclose(ext_a[0, :], ext_b[0, :], rtol=0, atol=atol)
        elif len_a == 1:
            return all([np.allclose(ext_a[0, :], ext_b[i, :], rtol=0, atol=atol) for i in sm.xrange(len_b)])
        elif len_b == 1:
            return all([np.allclose(ext_b[0, :], ext_a[i, :], rtol=0, atol=atol) for i in sm.xrange(len_a)])

        # After this point, both polygons have at least 2 points, i.e. LineStrings can be used.
        # We can also safely go back to the original exteriors (before close points were merged).
        ls_a = self.to_shapely_line_string(closed=True, interpolate=interpolate)
        ls_b = other_polygon.to_shapely_line_string(closed=True, interpolate=interpolate) \
            if not is_np_array(other_polygon) \
            else _convert_points_to_shapely_line_string(other_polygon, closed=True, interpolate=interpolate)

        # Measure the distance from each point in A to LineString B and vice versa.
        # Make sure that no point violates the tolerance.
        # Note that we can't just use LineString.almost_equals(LineString) -- that seems to expect the same number
        # and order of points in both LineStrings (failed with duplicated points).
        for x, y in ls_a.coords:
            point = shapely.geometry.Point(x, y)
            if not ls_b.distance(point) <= max_distance:
                return False

        for x, y in ls_b.coords:
            point = shapely.geometry.Point(x, y)
            if not ls_a.distance(point) <= max_distance:
                return False

        return True

    def almost_equals(self, other, max_distance=1e-6, interpolate=8):
        """
        Compare this polygon with another one and estimate whether they can be viewed as equal.

        This is the same as :func:`imgaug.Polygon.exterior_almost_equals` but additionally compares the labels.

        Parameters
        ----------
        other
            The object to compare against. If not a Polygon, then False will be returned.

        max_distance : float
            See :func:`imgaug.Polygon.exterior_almost_equals`.

        interpolate : int
            See :func:`imgaug.Polygon.exterior_almost_equals`.

        Returns
        -------
        bool
            Whether the two polygons can be viewed as equal. In the case of the exteriors this is an approximate test.

        """
        if not isinstance(other, Polygon):
            return False
        if self.label is not None or other.label is not None:
            if self.label is None:
                return False
            if other.label is None:
                return False
            if self.label != other.label:
                return False
        return self.exterior_almost_equals(other, max_distance=max_distance, interpolate=interpolate)

    def copy(self, exterior=None, label=None):
        """
        Create a shallow copy of the Polygon object.

        Parameters
        ----------
        exterior : list of imgaug.Keypoint or list of tuple or (N,2) ndarray, optional
            List of points defining the polygon. See :func:`imgaug.Polygon.__init__` for details.

        label : None or str, optional
            If not None, then the label of the copied object will be set to this value.

        Returns
        -------
        imgaug.Polygon
            Shallow copy.

        """
        return self.deepcopy(exterior=exterior, label=label)

    def deepcopy(self, exterior=None, label=None):
        """
        Create a deep copy of the Polygon object.

        Parameters
        ----------
        exterior : list of Keypoint or list of tuple or (N,2) ndarray, optional
            List of points defining the polygon. See `imgaug.Polygon.__init__` for details.

        label : None or str
            If not None, then the label of the copied object will be set to this value.

        Returns
        -------
        imgaug.Polygon
            Deep copy.

        """
        return Polygon(
            exterior=np.copy(self.exterior) if exterior is None else exterior,
            label=self.label if label is None else label
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        points_str = ", ".join(["(x=%.3f, y=%.3f)" % (point[0], point[1]) for point in self.exterior])
        return "Polygon([%s] (%d points), label=%s)" % (points_str, len(self.exterior), self.label)


def _convert_points_to_shapely_line_string(points, closed=False, interpolate=0):
    # load shapely lazily, which makes the dependency more optional
    import shapely.geometry

    if len(points) <= 1:
        raise Exception(
            ("Conversion to shapely line string requires at least two points, but points input contains "
             "only %d points.") % (len(points),)
        )

    points_tuples = [(point[0], point[1]) for point in points]

    # interpolate points between each consecutive pair of points
    if interpolate > 0:
        points_tuples = _interpolate_points(points_tuples, interpolate)

    # close if requested and not yet closed
    if closed and len(points) > 1:  # here intentionally used points instead of points_tuples
        points_tuples.append(points_tuples[0])

    return shapely.geometry.LineString(points_tuples)


def _interpolate_point_pair(point_a, point_b, nb_steps):
    if nb_steps < 1:
        return []
    x1, y1 = point_a
    x2, y2 = point_b
    vec = np.float32([x2 - x1, y2 - y1])
    step_size = vec / (1 + nb_steps)
    return [(x1 + (i + 1) * step_size[0], y1 + (i + 1) * step_size[1]) for i in sm.xrange(nb_steps)]


def _interpolate_points(points, nb_steps, closed=True):
    if len(points) <= 1:
        return points
    if closed:
        points = list(points) + [points[0]]
    points_interp = []
    for point_a, point_b in zip(points[:-1], points[1:]):
        points_interp.extend([point_a] + _interpolate_point_pair(point_a, point_b, nb_steps))
    if not closed:
        points_interp.append(points[-1])
    # close does not have to be reverted here, as last point is not included in the extend()
    return points_interp


def _interpolate_points_by_max_distance(points, max_distance, closed=True):
    do_assert(max_distance > 0, "max_distance must have value greater than 0, got %.8f" % (max_distance,))
    if len(points) <= 1:
        return points
    if closed:
        points = list(points) + [points[0]]
    points_interp = []
    for point_a, point_b in zip(points[:-1], points[1:]):
        dist = np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
        nb_steps = int((dist / max_distance) - 1)
        points_interp.extend([point_a] + _interpolate_point_pair(point_a, point_b, nb_steps))
    if not closed:
        points_interp.append(points[-1])
    return points_interp


class MultiPolygon(object):
    """
    Class that represents several polygons.

    Parameters
    ----------
    geoms : list of imgaug.Polygon
        List of the polygons.

    """
    def __init__(self, geoms):
        """Create a new MultiPolygon instance."""
        do_assert(len(geoms) == 0 or all([isinstance(el, Polygon) for el in geoms]))
        self.geoms = geoms

    @staticmethod
    def from_shapely(geometry, label=None):
        """
        Create a MultiPolygon from a Shapely MultiPolygon, a Shapely Polygon or a Shapely GeometryCollection.

        This also creates all necessary Polygons contained by this MultiPolygon.

        Parameters
        ----------
        geometry : shapely.geometry.MultiPolygon or shapely.geometry.Polygon\
                   or shapely.geometry.collection.GeometryCollection
            The object to convert to a MultiPolygon.

        label : None or str, optional
            A label assigned to all Polygons within the MultiPolygon.

        Returns
        -------
        imgaug.MultiPolygon
            The derived MultiPolygon.

        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        if isinstance(geometry, shapely.geometry.MultiPolygon):
            return MultiPolygon([Polygon.from_shapely(poly, label=label) for poly in geometry.geoms])
        elif isinstance(geometry, shapely.geometry.Polygon):
            return MultiPolygon([Polygon.from_shapely(geometry, label=label)])
        elif isinstance(geometry, shapely.geometry.collection.GeometryCollection):
            do_assert(all([isinstance(poly, shapely.geometry.Polygon) for poly in geometry.geoms]))
            return MultiPolygon([Polygon.from_shapely(poly, label=label) for poly in geometry.geoms])
        else:
            raise Exception("Unknown datatype '%s'. Expected shapely.geometry.Polygon or "
                            "shapely.geometry.MultiPolygon or "
                            "shapely.geometry.collections.GeometryCollection." % (type(geometry),))


class HeatmapsOnImage(object):
    """
    Object representing heatmaps on images.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Array representing the heatmap(s).
        Must be of dtype float32.
        If multiple heatmaps are provided, then ``C`` is expected to denote their number.

    shape : tuple of int
        Shape of the image on which the heatmap(s) is/are placed. NOT the shape of the
        heatmap(s) array, unless it is identical to the image shape (note the likely
        difference between the arrays in the number of channels).
        If there is not a corresponding image, use the shape of the heatmaps array.

    min_value : float, optional
        Minimum value for the heatmaps that `arr` represents. This will usually be ``0.0``.

    max_value : float, optional
        Maximum value for the heatmaps that `arr` represents. This will usually be ``1.0``.

    """

    def __init__(self, arr, shape, min_value=0.0, max_value=1.0):
        """Construct a new HeatmapsOnImage object."""
        do_assert(is_np_array(arr), "Expected numpy array as heatmap input array, got type %s" % (type(arr),))
        # TODO maybe allow 0-sized heatmaps? in that case the min() and max() must be adjusted
        do_assert(arr.shape[0] > 0 and arr.shape[1] > 0,
                  "Expected numpy array as heatmap with height and width greater than 0, got shape %s." % (arr.shape,))
        do_assert(arr.dtype.type in [np.float32],
                  "Heatmap input array expected to be of dtype float32, got dtype %s." % (arr.dtype,))
        do_assert(arr.ndim in [2, 3], "Heatmap input array must be 2d or 3d, got shape %s." % (arr.shape,))
        do_assert(len(shape) in [2, 3],
                  "Argument 'shape' in HeatmapsOnImage expected to be 2d or 3d, got shape %s." % (shape,))
        do_assert(min_value < max_value)
        do_assert(np.min(arr.flat[0:50]) >= min_value - np.finfo(arr.dtype).eps,
                  ("Value range of heatmap was chosen to be (%.8f, %.8f), but found value below minimum in first "
                   + "50 heatmap array values.") % (min_value, max_value))
        do_assert(np.max(arr.flat[0:50]) <= max_value + np.finfo(arr.dtype).eps,
                  ("Value range of heatmap was chosen to be (%.8f, %.8f), but found value above maximum in first "
                   + "50 heatmap array values.") % (min_value, max_value))

        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
            self.arr_was_2d = True
        else:
            self.arr_was_2d = False

        eps = np.finfo(np.float32).eps
        min_is_zero = 0.0 - eps < min_value < 0.0 + eps
        max_is_one = 1.0 - eps < max_value < 1.0 + eps
        if min_is_zero and max_is_one:
            self.arr_0to1 = arr
        else:
            self.arr_0to1 = (arr - min_value) / (max_value - min_value)
        self.shape = shape
        self.min_value = min_value
        self.max_value = max_value

    def get_arr(self):
        """
        Get the heatmap's array within the value range originally provided in ``__init__()``.

        The HeatmapsOnImage object saves heatmaps internally in the value range ``(min=0.0, max=1.0)``.
        This function converts the internal representation to ``(min=min_value, max=max_value)``,
        where ``min_value`` and ``max_value`` are provided upon instantiation of the object.

        Returns
        -------
        result : (H,W) ndarray or (H,W,C) ndarray
            Heatmap array. Dtype is float32.

        """
        if self.arr_was_2d and self.arr_0to1.shape[2] == 1:
            arr = self.arr_0to1[:, :, 0]
        else:
            arr = self.arr_0to1

        eps = np.finfo(np.float32).eps
        min_is_zero = 0.0 - eps < self.min_value < 0.0 + eps
        max_is_one = 1.0 - eps < self.max_value < 1.0 + eps
        if min_is_zero and max_is_one:
            return np.copy(arr)
        else:
            diff = self.max_value - self.min_value
            return self.min_value + diff * arr

    # TODO
    # def find_global_maxima(self):
    #    raise NotImplementedError()

    def draw(self, size=None, cmap="jet"):
        """
        Render the heatmaps as RGB images.

        Parameters
        ----------
        size : None or float or iterable of int or iterable of float, optional
            Size of the rendered RGB image as ``(height, width)``.
            See :func:`imgaug.imgaug.imresize_single_image` for details.
            If set to None, no resizing is performed and the size of the heatmaps array is used.

        cmap : str or None, optional
            Color map of ``matplotlib`` to use in order to convert the heatmaps to RGB images.
            If set to None, no color map will be used and the heatmaps will be converted
            to simple intensity maps.

        Returns
        -------
        heatmaps_drawn : list of (H,W,3) ndarray
            Rendered heatmaps. One per heatmap array channel. Dtype is uint8.

        """
        heatmaps_uint8 = self.to_uint8()
        heatmaps_drawn = []

        for c in sm.xrange(heatmaps_uint8.shape[2]):
            # c:c+1 here, because the additional axis is needed by imresize_single_image
            heatmap_c = heatmaps_uint8[..., c:c+1]

            if size is not None:
                heatmap_c_rs = imresize_single_image(heatmap_c, size, interpolation="nearest")
            else:
                heatmap_c_rs = heatmap_c
            heatmap_c_rs = np.squeeze(heatmap_c_rs).astype(np.float32) / 255.0

            if cmap is not None:
                # import only when necessary (faster startup; optional dependency; less fragile -- see issue #225)
                import matplotlib.pyplot as plt

                cmap_func = plt.get_cmap(cmap)
                heatmap_cmapped = cmap_func(heatmap_c_rs)
                heatmap_cmapped = np.delete(heatmap_cmapped, 3, 2)
            else:
                heatmap_cmapped = np.tile(heatmap_c_rs[..., np.newaxis], (1, 1, 3))

            heatmap_cmapped = np.clip(heatmap_cmapped * 255, 0, 255).astype(np.uint8)

            heatmaps_drawn.append(heatmap_cmapped)
        return heatmaps_drawn

    def draw_on_image(self, image, alpha=0.75, cmap="jet", resize="heatmaps"):
        """
        Draw the heatmaps as overlays over an image.

        Parameters
        ----------
        image : (H,W,3) ndarray
            Image onto which to draw the heatmaps. Expected to be of dtype uint8.

        alpha : float, optional
            Alpha/opacity value to use for the mixing of image and heatmaps.
            Higher values mean that the heatmaps will be more visible and the image less visible.

        cmap : str or None, optional
            Color map to use. See :func:`imgaug.HeatmapsOnImage.draw` for details.

        resize : {'heatmaps', 'image'}, optional
            In case of size differences between the image and heatmaps, either the image or
            the heatmaps can be resized. This parameter controls which of the two will be resized
            to the other's size.

        Returns
        -------
        mix : list of (H,W,3) ndarray
            Rendered overlays. One per heatmap array channel. Dtype is uint8.

        """
        # assert RGB image
        do_assert(image.ndim == 3)
        do_assert(image.shape[2] == 3)
        do_assert(image.dtype.type == np.uint8)

        do_assert(0 - 1e-8 <= alpha <= 1.0 + 1e-8)
        do_assert(resize in ["heatmaps", "image"])

        if resize == "image":
            image = imresize_single_image(image, self.arr_0to1.shape[0:2], interpolation="cubic")

        heatmaps_drawn = self.draw(
            size=image.shape[0:2] if resize == "heatmaps" else None,
            cmap=cmap
        )

        mix = [
            np.clip((1-alpha) * image + alpha * heatmap_i, 0, 255).astype(np.uint8)
            for heatmap_i
            in heatmaps_drawn
        ]

        return mix

    def invert(self):
        """
        Inverts each value in the heatmap, shifting low towards high values and vice versa.

        This changes each value to::

            v' = max - (v - min)

        where ``v`` is the value at some spatial location, ``min`` is the minimum value in the heatmap
        and ``max`` is the maximum value.
        As the heatmap uses internally a 0.0 to 1.0 representation, this simply becomes ``v' = 1.0 - v``.

        Note that the attributes ``min_value`` and ``max_value`` are not switched. They both keep their values.

        This function can be useful e.g. when working with depth maps, where algorithms might have
        an easier time representing the furthest away points with zeros, requiring an inverted
        depth map.

        Returns
        -------
        arr_inv : imgaug.HeatmapsOnImage
            Inverted heatmap.

        """
        arr_inv = HeatmapsOnImage.from_0to1(1 - self.arr_0to1, shape=self.shape, min_value=self.min_value,
                                            max_value=self.max_value)
        arr_inv.arr_was_2d = self.arr_was_2d
        return arr_inv

    def pad(self, top=0, right=0, bottom=0, left=0, mode="constant", cval=0.0):
        """
        Pad the heatmaps on their top/right/bottom/left side.

        Parameters
        ----------
        top : int, optional
            Amount of pixels to add at the top side of the heatmaps. Must be 0 or greater.

        right : int, optional
            Amount of pixels to add at the right side of the heatmaps. Must be 0 or greater.

        bottom : int, optional
            Amount of pixels to add at the bottom side of the heatmaps. Must be 0 or greater.

        left : int, optional
            Amount of pixels to add at the left side of the heatmaps. Must be 0 or greater.

        mode : string, optional
            Padding mode to use. See :func:`numpy.pad` for details.

        cval : number, optional
            Value to use for padding if `mode` is ``constant``. See :func:`numpy.pad` for details.

        Returns
        -------
        imgaug.HeatmapsOnImage
            Padded heatmaps of height ``H'=H+top+bottom`` and width ``W'=W+left+right``.

        """
        arr_0to1_padded = pad(self.arr_0to1, top=top, right=right, bottom=bottom, left=left, mode=mode, cval=cval)
        return HeatmapsOnImage.from_0to1(arr_0to1_padded, shape=self.shape, min_value=self.min_value,
                                         max_value=self.max_value)

    def pad_to_aspect_ratio(self, aspect_ratio, mode="constant", cval=0.0, return_pad_amounts=False):
        """
        Pad the heatmaps on their sides so that they match a target aspect ratio.

        Depending on which dimension is smaller (height or width), only the corresponding
        sides (left/right or top/bottom) will be padded. In each case, both of the sides will
        be padded equally.

        Parameters
        ----------
        aspect_ratio : float
            Target aspect ratio, given as width/height. E.g. 2.0 denotes the image having twice
            as much width as height.

        mode : str, optional
            Padding mode to use. See :func:`numpy.pad` for details.

        cval : number, optional
            Value to use for padding if `mode` is ``constant``. See :func:`numpy.pad` for details.

        return_pad_amounts : bool, optional
            If False, then only the padded image will be returned. If True, a tuple with two
            entries will be returned, where the first entry is the padded image and the second
            entry are the amounts by which each image side was padded. These amounts are again a
            tuple of the form (top, right, bottom, left), with each value being an integer.

        Returns
        -------
        heatmaps : imgaug.HeatmapsOnImage
            Padded heatmaps as HeatmapsOnImage object.

        pad_amounts : tuple of int
            Amounts by which the heatmaps were padded on each side, given as a tuple ``(top, right, bottom, left)``.
            This tuple is only returned if `return_pad_amounts` was set to True.

        """
        arr_0to1_padded, pad_amounts = pad_to_aspect_ratio(self.arr_0to1, aspect_ratio=aspect_ratio, mode=mode,
                                                           cval=cval, return_pad_amounts=True)
        heatmaps = HeatmapsOnImage.from_0to1(arr_0to1_padded, shape=self.shape, min_value=self.min_value,
                                             max_value=self.max_value)
        if return_pad_amounts:
            return heatmaps, pad_amounts
        else:
            return heatmaps

    def avg_pool(self, block_size):
        """
        Resize the heatmap(s) array using average pooling of a given block/kernel size.

        Parameters
        ----------
        block_size : int or tuple of int
            Size of each block of values to pool, aka kernel size. See :func:`imgaug.pool` for details.

        Returns
        -------
        imgaug.HeatmapsOnImage
            Heatmaps after average pooling.

        """
        arr_0to1_reduced = avg_pool(self.arr_0to1, block_size, cval=0.0)
        return HeatmapsOnImage.from_0to1(arr_0to1_reduced, shape=self.shape, min_value=self.min_value,
                                         max_value=self.max_value)

    def max_pool(self, block_size):
        """
        Resize the heatmap(s) array using max-pooling of a given block/kernel size.

        Parameters
        ----------
        block_size : int or tuple of int
            Size of each block of values to pool, aka kernel size. See :func:`imgaug.pool` for details.

        Returns
        -------
        imgaug.HeatmapsOnImage
            Heatmaps after max-pooling.

        """
        arr_0to1_reduced = max_pool(self.arr_0to1, block_size)
        return HeatmapsOnImage.from_0to1(arr_0to1_reduced, shape=self.shape, min_value=self.min_value,
                                         max_value=self.max_value)

    def scale(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("HeatmapsOnImage.scale() is deprecated. "
                                         "Use HeatmapsOnImage.resize() instead. "
                                         "It has the exactly same interface "
                                         "(simple renaming)."))
        return self.resize(*args, **kwargs)

    def resize(self, sizes, interpolation="cubic"):
        """
        Resize the heatmap(s) array to the provided size given the provided interpolation.

        Parameters
        ----------
        sizes : float or iterable of int or iterable of float
            New size of the array in ``(height, width)``.
            See :func:`imgaug.imgaug.imresize_single_image` for details.

        interpolation : None or str or int, optional
            The interpolation to use during resize.
            See :func:`imgaug.imgaug.imresize_single_image` for details.

        Returns
        -------
        imgaug.HeatmapsOnImage
            Resized heatmaps object.

        """
        arr_0to1_resized = imresize_single_image(self.arr_0to1, sizes, interpolation=interpolation)

        # cubic interpolation can lead to values outside of [0.0, 1.0],
        # see https://github.com/opencv/opencv/issues/7195
        # TODO area interpolation too?
        arr_0to1_resized = np.clip(arr_0to1_resized, 0.0, 1.0)

        return HeatmapsOnImage.from_0to1(arr_0to1_resized, shape=self.shape, min_value=self.min_value,
                                         max_value=self.max_value)

    def to_uint8(self):
        """
        Convert this heatmaps object to a 0-to-255 array.

        Returns
        -------
        arr_uint8 : (H,W,C) ndarray
            Heatmap as a 0-to-255 array (dtype is uint8).

        """
        # TODO this always returns (H,W,C), even if input ndarray was originall (H,W)
        # does it make sense here to also return (H,W) if self.arr_was_2d?
        arr_0to255 = np.clip(np.round(self.arr_0to1 * 255), 0, 255)
        arr_uint8 = arr_0to255.astype(np.uint8)
        return arr_uint8

    @staticmethod
    def from_uint8(arr_uint8, shape, min_value=0.0, max_value=1.0):
        """
        Create a heatmaps object from an heatmap array containing values ranging from 0 to 255.

        Parameters
        ----------
        arr_uint8 : (H,W) ndarray or (H,W,C) ndarray
            Heatmap(s) array, where ``H`` is height, ``W`` is width and ``C`` is the number of heatmap channels.
            Expected dtype is uint8.

        shape : tuple of int
            Shape of the image on which the heatmap(s) is/are placed. NOT the shape of the
            heatmap(s) array, unless it is identical to the image shape (note the likely
            difference between the arrays in the number of channels).
            If there is not a corresponding image, use the shape of the heatmaps array.

        min_value : float, optional
            Minimum value for the heatmaps that the 0-to-255 array represents. This will usually
            be 0.0. It is used when calling :func:`imgaug.HeatmapsOnImage.get_arr`, which converts the
            underlying ``(0, 255)`` array to value range ``(min_value, max_value)``.

        max_value : float, optional
            Maximum value for the heatmaps that 0-to-255 array represents.
            See parameter `min_value` for details.

        Returns
        -------
        imgaug.HeatmapsOnImage
            Heatmaps object.

        """
        arr_0to1 = arr_uint8.astype(np.float32) / 255.0
        return HeatmapsOnImage.from_0to1(arr_0to1, shape, min_value=min_value, max_value=max_value)

    @staticmethod
    def from_0to1(arr_0to1, shape, min_value=0.0, max_value=1.0):
        """
        Create a heatmaps object from an heatmap array containing values ranging from 0.0 to 1.0.

        Parameters
        ----------
        arr_0to1 : (H,W) or (H,W,C) ndarray
            Heatmap(s) array, where ``H`` is height, ``W`` is width and ``C`` is the number of heatmap channels.
            Expected dtype is float32.

        shape : tuple of ints
            Shape of the image on which the heatmap(s) is/are placed. NOT the shape of the
            heatmap(s) array, unless it is identical to the image shape (note the likely
            difference between the arrays in the number of channels).
            If there is not a corresponding image, use the shape of the heatmaps array.

        min_value : float, optional
            Minimum value for the heatmaps that the 0-to-1 array represents. This will usually
            be 0.0. It is used when calling :func:`imgaug.HeatmapsOnImage.get_arr`, which converts the
            underlying ``(0.0, 1.0)`` array to value range ``(min_value, max_value)``.
            E.g. if you started with heatmaps in the range ``(-1.0, 1.0)`` and projected these
            to (0.0, 1.0), you should call this function with ``min_value=-1.0``, ``max_value=1.0``
            so that :func:`imgaug.HeatmapsOnImage.get_arr` returns heatmap arrays having value
            range (-1.0, 1.0).

        max_value : float, optional
            Maximum value for the heatmaps that to 0-to-255 array represents.
            See parameter min_value for details.

        Returns
        -------
        heatmaps : imgaug.HeatmapsOnImage
            Heatmaps object.

        """
        heatmaps = HeatmapsOnImage(arr_0to1, shape, min_value=0.0, max_value=1.0)
        heatmaps.min_value = min_value
        heatmaps.max_value = max_value
        return heatmaps

    @classmethod
    def change_normalization(cls, arr, source, target):
        """
        Change the value range of a heatmap from one min-max to another min-max.

        E.g. the value range may be changed from min=0.0, max=1.0 to min=-1.0, max=1.0.

        Parameters
        ----------
        arr : ndarray
            Heatmap array to modify.

        source : tuple of float
            Current value range of the input array, given as (min, max), where both are float values.

        target : tuple of float
            Desired output value range of the array, given as (min, max), where both are float values.

        Returns
        -------
        arr_target : ndarray
            Input array, with value range projected to the desired target value range.

        """
        do_assert(is_np_array(arr))

        if isinstance(source, HeatmapsOnImage):
            source = (source.min_value, source.max_value)
        else:
            do_assert(isinstance(source, tuple))
            do_assert(len(source) == 2)
            do_assert(source[0] < source[1])

        if isinstance(target, HeatmapsOnImage):
            target = (target.min_value, target.max_value)
        else:
            do_assert(isinstance(target, tuple))
            do_assert(len(target) == 2)
            do_assert(target[0] < target[1])

        # Check if source and target are the same (with a tiny bit of tolerance)
        # if so, evade compuation and just copy the array instead.
        # This is reasonable, as source and target will often both be (0.0, 1.0).
        eps = np.finfo(arr.dtype).eps
        mins_same = source[0] - 10*eps < target[0] < source[0] + 10*eps
        maxs_same = source[1] - 10*eps < target[1] < source[1] + 10*eps
        if mins_same and maxs_same:
            return np.copy(arr)

        min_source, max_source = source
        min_target, max_target = target

        diff_source = max_source - min_source
        diff_target = max_target - min_target

        arr_0to1 = (arr - min_source) / diff_source
        arr_target = min_target + arr_0to1 * diff_target

        return arr_target

    def copy(self):
        """
        Create a shallow copy of the Heatmaps object.

        Returns
        -------
        imgaug.HeatmapsOnImage
            Shallow copy.

        """
        return self.deepcopy()

    def deepcopy(self):
        """
        Create a deep copy of the Heatmaps object.

        Returns
        -------
        imgaug.HeatmapsOnImage
            Deep copy.

        """
        return HeatmapsOnImage(self.get_arr(), shape=self.shape, min_value=self.min_value, max_value=self.max_value)


class SegmentationMapOnImage(object):
    """
    Object representing a segmentation map associated with an image.

    Attributes
    ----------
    DEFAULT_SEGMENT_COLORS : list of tuple of int
        Standard RGB colors to use during drawing, ordered by class index.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,1) ndarray or (H,W,C) ndarray
        Array representing the segmentation map. May have datatypes bool, integer or float.

            * If bool: Assumed to be of shape (H,W), (H,W,1) or (H,W,C). If (H,W) or (H,W,1) it
              is assumed to be for the case of having a single class (where any False denotes
              background). Otherwise there are assumed to be C channels, one for each class,
              with each of them containing a mask for that class. The masks may overlap.
            * If integer: Assumed to be of shape (H,W) or (H,W,1). Each pixel is assumed to
              contain an integer denoting the class index. Classes are assumed to be
              non-overlapping. The number of classes cannot be guessed from this input, hence
              nb_classes must be set.
            * If float: Assumed to b eof shape (H,W), (H,W,1) or (H,W,C) with meanings being
              similar to the case of `bool`. Values are expected to fall always in the range
              0.0 to 1.0 and are usually expected to be either 0.0 or 1.0 upon instantiation
              of a new segmentation map. Classes may overlap.

    shape : iterable of int
        Shape of the corresponding image (NOT the segmentation map array). This is expected
        to be ``(H, W)`` or ``(H, W, C)`` with ``C`` usually being 3. If there is no corresponding image,
        then use the segmentation map's shape instead.

    nb_classes : int or None
        Total number of unique classes that may appear in an segmentation map, i.e. the max
        class index. This may be None if the input array is of type bool or float. The number
        of classes however must be provided if the input array is of type int, as then the
        number of classes cannot be guessed.

    """

    DEFAULT_SEGMENT_COLORS = [
        (0, 0, 0),  # black
        (230, 25, 75),  # red
        (60, 180, 75),  # green
        (255, 225, 25),  # yellow
        (0, 130, 200),  # blue
        (245, 130, 48),  # orange
        (145, 30, 180),  # purple
        (70, 240, 240),  # cyan
        (240, 50, 230),  # magenta
        (210, 245, 60),  # lime
        (250, 190, 190),  # pink
        (0, 128, 128),  # teal
        (230, 190, 255),  # lavender
        (170, 110, 40),  # brown
        (255, 250, 200),  # beige
        (128, 0, 0),  # maroon
        (170, 255, 195),  # mint
        (128, 128, 0),  # olive
        (255, 215, 180),  # coral
        (0, 0, 128),  # navy
        (128, 128, 128),  # grey
        (255, 255, 255),  # white
        # --
        (115, 12, 37),  # dark red
        (30, 90, 37),  # dark green
        (127, 112, 12),  # dark yellow
        (0, 65, 100),  # dark blue
        (122, 65, 24),  # dark orange
        (72, 15, 90),  # dark purple
        (35, 120, 120),  # dark cyan
        (120, 25, 115),  # dark magenta
        (105, 122, 30),  # dark lime
        (125, 95, 95),  # dark pink
        (0, 64, 64),  # dark teal
        (115, 95, 127),  # dark lavender
        (85, 55, 20),  # dark brown
        (127, 125, 100),  # dark beige
        (64, 0, 0),  # dark maroon
        (85, 127, 97),  # dark mint
        (64, 64, 0),  # dark olive
        (127, 107, 90),  # dark coral
        (0, 0, 64),  # dark navy
        (64, 64, 64),  # dark grey
    ]

    def __init__(self, arr, shape, nb_classes=None):
        do_assert(is_np_array(arr), "Expected to get numpy array, got %s." % (type(arr),))

        if arr.dtype.name == "bool":
            do_assert(arr.ndim in [2, 3])
            self.input_was = ("bool", arr.ndim)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            arr = arr.astype(np.float32)
        elif arr.dtype.kind in ["i", "u"]:
            do_assert(arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1))
            do_assert(nb_classes is not None)
            do_assert(nb_classes > 0)
            do_assert(np.min(arr.flat[0:100]) >= 0)
            do_assert(np.max(arr.flat[0:100]) <= nb_classes)
            self.input_was = ("int", arr.dtype.type, arr.ndim)
            if arr.ndim == 3:
                arr = arr[..., 0]
            # TODO improve efficiency here by building only sub-heatmaps for classes actually
            # present in the image. This would also get rid of nb_classes.
            arr = np.eye(nb_classes)[arr]  # from class indices to one hot
            arr = arr.astype(np.float32)
        elif arr.dtype.kind == "f":
            do_assert(arr.ndim == 3)
            self.input_was = ("float", arr.dtype.type, arr.ndim)
            arr = arr.astype(np.float32)
        else:
            raise Exception(("Input was expected to be an ndarray any bool, int, uint or float dtype. "
                             + "Got dtype %s.") % (arr.dtype.name,))
        do_assert(arr.ndim == 3)
        do_assert(arr.dtype.name == "float32")
        self.arr = arr
        self.shape = shape
        self.nb_classes = nb_classes if nb_classes is not None else arr.shape[2]

    def get_arr_int(self, background_threshold=0.01, background_class_id=None):
        """
        Get the segmentation map array as an integer array of shape (H, W).

        Each pixel in that array contains an integer value representing the pixel's class.
        If multiple classes overlap, the one with the highest local float value is picked.
        If that highest local value is below `background_threshold`, the method instead uses
        the background class id as the pixel's class value.
        By default, class id 0 is the background class. This may only be changed if the original
        input to the segmentation map object was an integer map.

        Parameters
        ----------
        background_threshold : float, optional
            At each pixel, each class-heatmap has a value between 0.0 and 1.0. If none of the
            class-heatmaps has a value above this threshold, the method uses the background class
            id instead.

        background_class_id : None or int, optional
            Class id to fall back to if no class-heatmap passes the threshold at a spatial
            location. May only be provided if the original input was an integer mask and in these
            cases defaults to 0. If the input were float or boolean masks, the background class id
            may not be set as it is assumed that the background is implicitly defined
            as 'any spatial location that has zero-like values in all masks'.

        Returns
        -------
        result : (H,W) ndarray
            Segmentation map array (int32).
            If the original input consisted of boolean or float masks, then the highest possible
            class id is ``1+C``, where ``C`` is the number of provided float/boolean masks. The value
            ``0`` in the integer mask then denotes the background class.

        """
        if self.input_was[0] in ["bool", "float"]:
            do_assert(background_class_id is None,
                      "The background class id may only be changed if the original input to SegmentationMapOnImage "
                      + "was an *integer* based segmentation map.")

        if background_class_id is None:
            background_class_id = 0

        channelwise_max_idx = np.argmax(self.arr, axis=2)
        # for bool and float input masks, we assume that the background is implicitly given,
        # i.e. anything where all masks/channels have zero-like values
        # for int, we assume that the background class is explicitly given and has the index 0
        if self.input_was[0] in ["bool", "float"]:
            result = 1 + channelwise_max_idx
        else: # integer mask was provided
            result = channelwise_max_idx
        if background_threshold is not None and background_threshold > 0:
            probs = np.amax(self.arr, axis=2)
            result[probs < background_threshold] = background_class_id

        return result.astype(np.int32)

    # TODO
    # def get_arr_bool(self, allow_overlapping=False, threshold=0.5, background_threshold=0.01, background_class_id=0):
    #    raise NotImplementedError()

    def draw(self, size=None, background_threshold=0.01, background_class_id=None, colors=None,
             return_foreground_mask=False):
        """
        Render the segmentation map as an RGB image.

        Parameters
        ----------
        size : None or float or iterable of int or iterable of float, optional
            Size of the rendered RGB image as ``(height, width)``.
            See :func:`imgaug.imgaug.imresize_single_image` for details.
            If set to None, no resizing is performed and the size of the segmentation map array is used.

        background_threshold : float, optional
            See :func:`imgaug.SegmentationMapOnImage.get_arr_int`.

        background_class_id : None or int, optional
            See :func:`imgaug.SegmentationMapOnImage.get_arr_int`.

        colors : None or list of tuple of int, optional
            Colors to use. One for each class to draw. If None, then default colors will be used.

        return_foreground_mask : bool, optional
            Whether to return a mask of the same size as the drawn segmentation map, containing
            True at any spatial location that is not the background class and False everywhere else.

        Returns
        -------
        segmap_drawn : (H,W,3) ndarray
            Rendered segmentation map (dtype is uint8).

        foreground_mask : (H,W) ndarray
            Mask indicating the locations of foreground classes (dtype is bool).
            This value is only returned if `return_foreground_mask` is True.

        """
        arr = self.get_arr_int(background_threshold=background_threshold, background_class_id=background_class_id)
        nb_classes = 1 + np.max(arr)
        segmap_drawn = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        if colors is None:
            colors = SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS
        do_assert(nb_classes <= len(colors),
                  "Can't draw all %d classes as it would exceed the maximum number of %d available colors." % (
                      nb_classes, len(colors),))

        ids_in_map = np.unique(arr)
        for c, color in zip(sm.xrange(nb_classes), colors):
            if c in ids_in_map:
                class_mask = (arr == c)
                segmap_drawn[class_mask] = color

        if return_foreground_mask:
            background_class_id = 0 if background_class_id is None else background_class_id
            foreground_mask = (arr != background_class_id)
        else:
            foreground_mask = None

        if size is not None:
            segmap_drawn = imresize_single_image(segmap_drawn, size, interpolation="nearest")
            if foreground_mask is not None:
                foreground_mask = imresize_single_image(
                    foreground_mask.astype(np.uint8), size, interpolation="nearest") > 0

        if foreground_mask is not None:
            return segmap_drawn, foreground_mask
        return segmap_drawn

    def draw_on_image(self, image, alpha=0.75, resize="segmentation_map", background_threshold=0.01,
                      background_class_id=None, colors=None, draw_background=False):
        """
        Draw the segmentation map as an overlay over an image.

        Parameters
        ----------
        image : (H,W,3) ndarray
            Image onto which to draw the segmentation map. Dtype is expected to be uint8.

        alpha : float, optional
            Alpha/opacity value to use for the mixing of image and segmentation map.
            Higher values mean that the segmentation map will be more visible and the image less visible.

        resize : {'segmentation_map', 'image'}, optional
            In case of size differences between the image and segmentation map, either the image or
            the segmentation map can be resized. This parameter controls which of the two will be
            resized to the other's size.

        background_threshold : float, optional
            See :func:`imgaug.SegmentationMapOnImage.get_arr_int`.

        background_class_id : None or int, optional
            See :func:`imgaug.SegmentationMapOnImage.get_arr_int`.

        colors : None or list of tuple of int, optional
            Colors to use. One for each class to draw. If None, then default colors will be used.

        draw_background : bool, optional
            If True, the background will be drawn like any other class.
            If False, the background will not be drawn, i.e. the respective background pixels
            will be identical with the image's RGB color at the corresponding spatial location
            and no color overlay will be applied.

        Returns
        -------
        mix : (H,W,3) ndarray
            Rendered overlays (dtype is uint8).

        """
        # assert RGB image
        do_assert(image.ndim == 3)
        do_assert(image.shape[2] == 3)
        do_assert(image.dtype.type == np.uint8)

        do_assert(0 - 1e-8 <= alpha <= 1.0 + 1e-8)
        do_assert(resize in ["segmentation_map", "image"])

        if resize == "image":
            image = imresize_single_image(image, self.arr.shape[0:2], interpolation="cubic")

        segmap_drawn, foreground_mask = self.draw(
            background_threshold=background_threshold,
            background_class_id=background_class_id,
            size=image.shape[0:2] if resize == "segmentation_map" else None,
            colors=colors,
            return_foreground_mask=True
        )

        if draw_background:
            mix = np.clip(
                (1-alpha) * image + alpha * segmap_drawn,
                0,
                255
            ).astype(np.uint8)
        else:
            foreground_mask = foreground_mask[..., np.newaxis]
            mix = np.zeros_like(image)
            mix += (~foreground_mask).astype(np.uint8) * image
            mix += foreground_mask.astype(np.uint8) * np.clip(
                (1-alpha) * image + alpha * segmap_drawn,
                0,
                255
            ).astype(np.uint8)
        return mix

    def pad(self, top=0, right=0, bottom=0, left=0, mode="constant", cval=0.0):
        """
        Pad the segmentation map on its top/right/bottom/left side.

        Parameters
        ----------
        top : int, optional
            Amount of pixels to add at the top side of the segmentation map. Must be 0 or greater.

        right : int, optional
            Amount of pixels to add at the right side of the segmentation map. Must be 0 or greater.

        bottom : int, optional
            Amount of pixels to add at the bottom side of the segmentation map. Must be 0 or greater.

        left : int, optional
            Amount of pixels to add at the left side of the segmentation map. Must be 0 or greater.

        mode : str, optional
            Padding mode to use. See :func:`numpy.pad` for details.

        cval : number, optional
            Value to use for padding if `mode` is ``constant``. See :func:`numpy.pad` for details.

        Returns
        -------
        segmap : imgaug.SegmentationMapOnImage
            Padded segmentation map of height ``H'=H+top+bottom`` and width ``W'=W+left+right``.

        """
        arr_padded = pad(self.arr, top=top, right=right, bottom=bottom, left=left, mode=mode, cval=cval)
        segmap = SegmentationMapOnImage(arr_padded, shape=self.shape)
        segmap.input_was = self.input_was
        return segmap

    def pad_to_aspect_ratio(self, aspect_ratio, mode="constant", cval=0.0, return_pad_amounts=False):
        """
        Pad the segmentation map on its sides so that its matches a target aspect ratio.

        Depending on which dimension is smaller (height or width), only the corresponding
        sides (left/right or top/bottom) will be padded. In each case, both of the sides will
        be padded equally.

        Parameters
        ----------
        aspect_ratio : float
            Target aspect ratio, given as width/height. E.g. 2.0 denotes the image having twice
            as much width as height.

        mode : str, optional
            Padding mode to use. See :func:`numpy.pad` for details.

        cval : number, optional
            Value to use for padding if `mode` is ``constant``. See :func:`numpy.pad` for details.

        return_pad_amounts : bool, optional
            If False, then only the padded image will be returned. If True, a tuple with two
            entries will be returned, where the first entry is the padded image and the second
            entry are the amounts by which each image side was padded. These amounts are again a
            tuple of the form (top, right, bottom, left), with each value being an integer.

        Returns
        -------
        segmap : imgaug.SegmentationMapOnImage
            Padded segmentation map as SegmentationMapOnImage object.

        pad_amounts : tuple of int
            Amounts by which the segmentation map was padded on each side, given as a
            tuple ``(top, right, bottom, left)``.
            This tuple is only returned if `return_pad_amounts` was set to True.

        """
        arr_padded, pad_amounts = pad_to_aspect_ratio(self.arr, aspect_ratio=aspect_ratio, mode=mode, cval=cval,
                                                      return_pad_amounts=True)
        segmap = SegmentationMapOnImage(arr_padded, shape=self.shape)
        segmap.input_was = self.input_was
        if return_pad_amounts:
            return segmap, pad_amounts
        else:
            return segmap

    def scale(self, *args, **kwargs):
        warnings.warn(DeprecationWarning("SegmentationMapOnImage.scale() is deprecated. "
                                         "Use SegmentationMapOnImage.resize() instead. "
                                         "It has the exactly same interface (simple renaming)."))
        return self.resize(*args, **kwargs)

    def resize(self, sizes, interpolation="cubic"):
        """
        Resize the segmentation map array to the provided size given the provided interpolation.

        Parameters
        ----------
        sizes : float or iterable of int or iterable of float
            New size of the array in ``(height, width)``.
            See :func:`imgaug.imgaug.imresize_single_image` for details.

        interpolation : None or str or int, optional
            The interpolation to use during resize.
            See :func:`imgaug.imgaug.imresize_single_image` for details.
            Note: The segmentation map is internally stored as multiple float-based heatmaps,
            making smooth interpolations potentially more reasonable than nearest neighbour
            interpolation.

        Returns
        -------
        segmap : imgaug.SegmentationMapOnImage
            Resized segmentation map object.

        """
        arr_resized = imresize_single_image(self.arr, sizes, interpolation=interpolation)

        # cubic interpolation can lead to values outside of [0.0, 1.0],
        # see https://github.com/opencv/opencv/issues/7195
        # TODO area interpolation too?
        arr_resized = np.clip(arr_resized, 0.0, 1.0)
        segmap = SegmentationMapOnImage(arr_resized, shape=self.shape)
        segmap.input_was = self.input_was
        return segmap

    def to_heatmaps(self, only_nonempty=False, not_none_if_no_nonempty=False):
        """
        Convert segmentation map to heatmaps object.

        Each segmentation map class will be represented as a single heatmap channel.

        Parameters
        ----------
        only_nonempty : bool, optional
            If True, then only heatmaps for classes that appear in the segmentation map will be
            generated. Additionally, a list of these class ids will be returned.

        not_none_if_no_nonempty : bool, optional
            If `only_nonempty` is True and for a segmentation map no channel was non-empty,
            this function usually returns None as the heatmaps object. If however this parameter
            is set to True, a heatmaps object with one channel (representing class 0)
            will be returned as a fallback in these cases.

        Returns
        -------
        imgaug.HeatmapsOnImage or None
            Segmentation map as a heatmaps object.
            If `only_nonempty` was set to True and no class appeared in the segmentation map,
            then this is None.

        class_indices : list of int
            Class ids (0 to C-1) of the classes that were actually added to the heatmaps.
            Only returned if `only_nonempty` was set to True.

        """
        if not only_nonempty:
            return HeatmapsOnImage.from_0to1(self.arr, self.shape, min_value=0.0, max_value=1.0)
        else:
            nonempty_mask = np.sum(self.arr, axis=(0, 1)) > 0 + 1e-4
            if np.sum(nonempty_mask) == 0:
                if not_none_if_no_nonempty:
                    nonempty_mask[0] = True
                else:
                    return None, []

            class_indices = np.arange(self.arr.shape[2])[nonempty_mask]
            channels = self.arr[..., class_indices]
            return HeatmapsOnImage(channels, self.shape, min_value=0.0, max_value=1.0), class_indices

    @staticmethod
    def from_heatmaps(heatmaps, class_indices=None, nb_classes=None):
        """
        Convert heatmaps to segmentation map.

        Assumes that each class is represented as a single heatmap channel.

        Parameters
        ----------
        heatmaps : imgaug.HeatmapsOnImage
            Heatmaps to convert.

        class_indices : None or list of int, optional
            List of class indices represented by each heatmap channel. See also the
            secondary output of :func:`imgaug.SegmentationMapOnImage.to_heatmap`.
            If this is provided, it must have the same length as the number of heatmap channels.

        nb_classes : None or int, optional
            Number of classes. Must be provided if class_indices is set.

        Returns
        -------
        imgaug.SegmentationMapOnImage
            Segmentation map derived from heatmaps.

        """
        if class_indices is None:
            return SegmentationMapOnImage(heatmaps.arr_0to1, shape=heatmaps.shape)
        else:
            do_assert(nb_classes is not None)
            do_assert(min(class_indices) >= 0)
            do_assert(max(class_indices) < nb_classes)
            do_assert(len(class_indices) == heatmaps.arr_0to1.shape[2])
            arr_0to1 = heatmaps.arr_0to1
            arr_0to1_full = np.zeros((arr_0to1.shape[0], arr_0to1.shape[1], nb_classes), dtype=np.float32)
            class_indices_set = set(class_indices)
            heatmap_channel = 0
            for c in sm.xrange(nb_classes):
                if c in class_indices_set:
                    arr_0to1_full[:, :, c] = arr_0to1[:, :, heatmap_channel]
                    heatmap_channel += 1
            return SegmentationMapOnImage(arr_0to1_full, shape=heatmaps.shape)

    def copy(self):
        """
        Create a shallow copy of the segmentation map object.

        Returns
        -------
        imgaug.SegmentationMapOnImage
            Shallow copy.

        """
        return self.deepcopy()

    def deepcopy(self):
        """
        Create a deep copy of the segmentation map object.

        Returns
        -------
        imgaug.SegmentationMapOnImage
            Deep copy.

        """
        segmap = SegmentationMapOnImage(self.arr, shape=self.shape, nb_classes=self.nb_classes)
        segmap.input_was = self.input_was
        return segmap


class Batch(object):
    """
    Class encapsulating a batch before and after augmentation.

    Parameters
    ----------
    images : None or (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
        The images to augment.

    heatmaps : None or list of imgaug.HeatmapsOnImage
        The heatmaps to augment.

    segmentation_maps : None or list of SegmentationMapOnImage
        The segmentation maps to augment.

    keypoints : None or list of KeypointOnImage
        The keypoints to augment.

    bounding_boxes : None or list of BoundingBoxesOnImage
        The bounding boxes to augment.

    data
        Additional data that is saved in the batch and may be read out
        after augmentation. This could e.g. contain filepaths to each image
        in `images`. As this object is usually used for background
        augmentation with multiple processes, the augmented Batch objects might
        not be returned in the original order, making this information useful.

    """
    def __init__(self, images=None, heatmaps=None, segmentation_maps=None, keypoints=None, bounding_boxes=None,
                 data=None):
        self.images_unaug = images
        self.images_aug = None
        self.heatmaps_unaug = heatmaps
        self.heatmaps_aug = None
        self.segmentation_maps_unaug = segmentation_maps
        self.segmentation_maps_aug = None
        self.keypoints_unaug = keypoints
        self.keypoints_aug = None
        self.bounding_boxes_unaug = bounding_boxes
        self.bounding_boxes_aug = None
        self.data = data

    @property
    def images(self):
        warnings.warn(DeprecationWarning(
            "Accessing imgaug.Batch.images is deprecated. Access instead "
            "imgaug.Batch.images_unaug or imgaug.Batch.images_aug."))
        return self.images_unaug

    @property
    def heatmaps(self):
        warnings.warn(DeprecationWarning(
            "Accessing imgaug.Batch.heatmaps is deprecated. Access instead "
            "imgaug.Batch.heatmaps_unaug or imgaug.Batch.heatmaps_aug."))
        return self.heatmaps_unaug

    @property
    def segmentation_maps(self):
        warnings.warn(DeprecationWarning(
            "Accessing imgaug.Batch.segmentation_maps is deprecated. Access "
            "instead imgaug.Batch.segmentation_maps_unaug or "
            "imgaug.Batch.segmentation_maps_aug."))
        return self.segmentation_maps_unaug

    @property
    def keypoints(self):
        warnings.warn(DeprecationWarning(
            "Accessing imgaug.Batch.keypoints is deprecated. Access "
            "instead imgaug.Batch.keypoints_unaug or "
            "imgaug.Batch.keypoints_aug."))
        return self.keypoints_unaug

    @property
    def bounding_boxes(self):
        warnings.warn(DeprecationWarning(
            "Accessing imgaug.Batch.bounding_boxes is deprecated. Access "
            "instead imgaug.Batch.bounding_boxes_unaug or "
            "imgaug.Batch.bounding_boxes_aug."))
        return self.bounding_boxes_unaug

    def deepcopy(self):
        def _copy_images(images):
            if images is None:
                images_copy = None
            elif is_np_array(images):
                images_copy = np.copy(images)
            else:
                do_assert(is_iterable(images))
                do_assert(all([is_np_array(image) for image in images]))
                images_copy = list([np.copy(image) for image in images])
            return images_copy

        def _copy_augmentable_objects(augmentables, clazz):
            if augmentables is None:
                augmentables_copy = None
            else:
                do_assert(is_iterable(augmentables))
                do_assert(all([isinstance(augmentable, clazz) for augmentable in augmentables]))
                augmentables_copy = [augmentable.deepcopy() for augmentable in augmentables]
            return augmentables_copy

        batch = Batch(
            images=_copy_images(self.images_unaug),
            heatmaps=_copy_augmentable_objects(self.heatmaps_unaug, HeatmapsOnImage),
            segmentation_maps=_copy_augmentable_objects(self.segmentation_maps_unaug, SegmentationMapOnImage),
            keypoints=_copy_augmentable_objects(self.keypoints_unaug, KeypointsOnImage),
            bounding_boxes=_copy_augmentable_objects(self.bounding_boxes_unaug, BoundingBoxesOnImage),
            data=copy.deepcopy(self.data)
        )
        batch.images_aug = _copy_images(self.images_aug)
        batch.heatmaps_aug = _copy_augmentable_objects(self.heatmaps_aug, HeatmapsOnImage)
        batch.segmentation_maps_aug = _copy_augmentable_objects(self.segmentation_maps_aug, SegmentationMapOnImage)
        batch.keypoints_aug = _copy_augmentable_objects(self.keypoints_aug, KeypointsOnImage)
        batch.bounding_boxes_aug = _copy_augmentable_objects(self.bounding_boxes_aug, BoundingBoxesOnImage)

        return batch


def BatchLoader(*args, **kwargs):
    warnings.warn(DeprecationWarning("Using imgaug.imgaug.BatchLoader is depcrecated. "
                                     "Use imgaug.multicore.BatchLoader instead."))

    from . import multicore
    return multicore.BatchLoader(*args, **kwargs)


def BackgroundAugmenter(*args, **kwargs):
    warnings.warn(DeprecationWarning("Using imgaug.imgaug.BackgroundAugmenter is depcrecated. "
                                     "Use imgaug.multicore.BackgroundAugmenter instead."))

    from . import multicore
    return multicore.BackgroundAugmenter(*args, **kwargs)
