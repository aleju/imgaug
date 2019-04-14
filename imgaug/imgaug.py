from __future__ import print_function, division, absolute_import

import math
import numbers
import sys
import os
import json
import types
import functools

import numpy as np
import cv2
import imageio
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


def flatten(nested_iterable):
    """
    Flattens arbitrarily nested lists/tuples.

    Code partially taken from https://stackoverflow.com/a/10824420.

    Parameters
    ----------
    nested_iterable
        A list or tuple of arbitrarily nested values.

    Yields
    ------
    any
        Non-list and non-tuple values in `nested_iterable`.

    """
    # don't just check if something is iterable here, because then strings
    # and arrays will be split into their characters and components
    if not isinstance(nested_iterable, (list, tuple)):
        yield nested_iterable
    else:
        for i in nested_iterable:
            if isinstance(i, (list, tuple)):
                for j in flatten(i):
                    yield j
            else:
                yield i


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
    # TODO get rid of this deferred import
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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
    # TODO get rid of this deferred import
    from imgaug.augmentables.heatmaps import HeatmapsOnImage

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
    # TODO get rid of this deferred import
    from imgaug.augmentables.segmaps import SegmentationMapOnImage

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
    # TODO get rid of this deferred import
    from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

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
    # TODO get rid of this deferred import
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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


def quokka_polygons(size=None, extract=None):
    """
    Returns example polygons on the standard example quokke image.

    The result contains one polygon, covering the quokka's outline.

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the polygons are placed. If None,
        then the polygons are not projected to any new size (positions on the
        original image are used). Floats lead to relative size changes, ints
        to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.BoundingBox or \
              imgaug.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`imgaug.quokka`.

    Returns
    -------
    psoi : imgaug.PolygonsOnImage
        Example polygons on the quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.polys import Polygon, PolygonsOnImage

    left, top = 0, 0
    if extract is not None:
        bb_extract = _quokka_normalize_extract(extract)
        left = bb_extract.x1
        top = bb_extract.y1
    with open(QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    polygons = []
    for poly_json in json_dict["polygons"]:
        polygons.append(
            Polygon([(point["x"] - left, point["y"] - top)
                    for point in poly_json["keypoints"]])
        )
    if extract is not None:
        shape = (bb_extract.height, bb_extract.width, 3)
    else:
        shape = (643, 960, 3)
    psoi = PolygonsOnImage(polygons, shape=shape)
    if size is not None:
        shape_resized = _compute_resized_shape(shape, size)
        psoi = psoi.on(shape_resized)
    return psoi


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
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    v1_u = (v1 / l1) if l1 > 0 else np.float32(v1) * 0
    v2_u = (v2 / l2) if l2 > 0 else np.float32(v2) * 0
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

    y1 : number
        y coordinate of the first point on line 1. (The lines extends beyond this point.)

    x2 : number
        x coordinate of the second point on line 1. (The lines extends beyond this point.)

    y2 : number
        y coordinate of the second point on line 1. (The lines extends beyond this point.)

    x3 : number
        x coordinate of the first point on line 2. (The lines extends beyond this point.)

    y3 : number
        y coordinate of the first point on line 2. (The lines extends beyond this point.)

    x4 : number
        x coordinate of the second point on line 2. (The lines extends beyond this point.)

    y4 : number
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

    # PIL/asarray returns read only array
    if not img_np.flags["WRITEABLE"]:
        try:
            # this seems to no longer work with np 1.16 (or was pillow updated?)
            img_np.setflags(write=True)
        except ValueError as ex:
            if "cannot set WRITEABLE flag to True of this array" in str(ex):
                img_np = np.copy(img_np)

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


#####################################################################
# Helpers for deprecation
#####################################################################

class DeprecationWarning(Warning):  # pylint: disable=redefined-builtin
    """Warning for deprecated calls.

    Since python 2.7 DeprecatedWarning is silent by default. So we define
    our own DeprecatedWarning here so that it is not silent by default.

    """
    pass


def warn_deprecated(msg, stacklevel=2):
    """Generate a non-silent deprecation warning with stacktrace.

    The used warning is ``imgaug.imgaug.DeprecationWarning``.

    Parameters
    ----------
    msg : str
        The message of the warning.

    stacklevel : int, optional
        How many steps above this function to "jump" in the stacktrace for
        the displayed file and line number of the error message.
        Usually 2.

    """
    import warnings
    warnings.warn(msg,
                  category=DeprecationWarning,
                  stacklevel=stacklevel)


class deprecated(object):
    """Decorator to mark deprecated functions with warning.

    Adapted from
    <https://github.com/scikit-image/scikit-image/blob/master/skimage/_shared/utils.py>.

    Parameters
    ----------
    alt_func : None or str, optional
        If given, tell user what function to use instead.

    behavior : {'warn', 'raise'}, optional
        Behavior during call to deprecated function: 'warn' = warn user that
        function is deprecated; 'raise' = raise error.

    removed_version : None or str, optional
        The package version in which the deprecated function will be removed.

    comment : None or str, optional
        An optional comment that will be appended to the warning message.

    """

    def __init__(self, alt_func=None, behavior="warn", removed_version=None,
                 comment=None):
        self.alt_func = alt_func
        self.behavior = behavior
        self.removed_version = removed_version
        self.comment = comment

    def __call__(self, func):
        alt_msg = None
        if self.alt_func is not None:
            alt_msg = "Use ``%s`` instead." % (self.alt_func,)

        rmv_msg = None
        if self.removed_version is not None:
            rmv_msg = "It will be removed in version %s." % (
                self.removed_version,)

        comment_msg = None
        if self.comment is not None and len(self.comment) > 0:
            comment_msg = "%s." % (self.comment.rstrip(". "),)

        addendum = " ".join([submsg
                             for submsg
                             in [alt_msg, rmv_msg, comment_msg]
                             if submsg is not None])

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # TODO add class name if class method
            import inspect
            # arg_names = func.__code__.co_varnames

            # getargspec() was deprecated in py3, but doesn't exist in py2
            if hasattr(inspect, "getfullargspec"):
                arg_names = inspect.getfullargspec(func)[0]
            else:
                arg_names = inspect.getargspec(func)[0]

            if "self" in arg_names or "cls" in arg_names:
                main_msg = "Method ``%s.%s()`` is deprecated." % (
                    args[0].__class__.__name__, func.__name__)
            else:
                main_msg = "Function ``%s()`` is deprecated." % (
                    func.__name__,)

            msg = (main_msg + " " + addendum).rstrip(" ").replace("``", "`")

            if self.behavior == "warn":
                warn_deprecated(msg, stacklevel=3)
            elif self.behavior == "raise":
                raise DeprecationWarning(msg)
            return func(*args, **kwargs)

        # modify doc string to display deprecation warning
        doc = "**Deprecated**. " + addendum
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + "\n\n    " + wrapped.__doc__

        return wrapped


#####################################################################
# Create classes/functions that were moved to other files and create
# DeprecatedWarnings when they are called.
#####################################################################

def _mark_moved_class_or_function(class_name_old, module_name_new, class_name_new):
    class_name_new = class_name_new if class_name_new is not None else class_name_old

    def _func(*args, **kwargs):
        import importlib
        warn_deprecated(
            "Using imgaug.imgaug.%s is deprecated. Use %s.%s instead." % (
                class_name_old, module_name_new, class_name_new
            ))
        module = importlib.import_module(module_name_new)
        return getattr(module, class_name_new)(*args, **kwargs)

    return _func


MOVED = [
    ("Keypoint", "imgaug.augmentables.kps", None),
    ("KeypointsOnImage", "imgaug.augmentables.kps", None),
    ("BoundingBox", "imgaug.augmentables.bbs", None),
    ("BoundingBoxesOnImage", "imgaug.augmentables.bbs", None),
    ("Polygon", "imgaug.augmentables.polys", None),
    ("PolygonsOnImage", "imgaug.augmentables.polys", None),
    ("MultiPolygon", "imgaug.augmentables.polys", None),
    ("_ConcavePolygonRecoverer", "imgaug.augmentables.polys", None),
    ("HeatmapsOnImage", "imgaug.augmentables.heatmaps", None),
    ("SegmentationMapOnImage", "imgaug.augmentables.segmaps", None),
    ("Batch", "imgaug.augmentables.batches", None),
    ("BatchLoader", "imgaug.multicore", None),
    ("BackgroundAugmenter", "imgaug.multicore", None),
    ("compute_geometric_median", "imgaug.augmentables.kps", None),
    ("_convert_points_to_shapely_line_string", "imgaug.augmentables.polys", None),
    ("_interpolate_point_pair", "imgaug.augmentables.polys", None),
    ("_interpolate_points", "imgaug.augmentables.polys", None),
    ("_interpolate_points_by_max_distance", "imgaug.augmentables.polys", None)
]

for class_name_old, module_name_new, class_name_new in MOVED:
    locals()[class_name_old] = _mark_moved_class_or_function(
        class_name_old, module_name_new, class_name_new)
