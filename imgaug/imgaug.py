from __future__ import print_function, division, absolute_import
import random
import numpy as np
import copy
import numbers
import cv2
import math
from scipy import misc, ndimage
import multiprocessing
import threading
import traceback
import sys
import six
import six.moves as sm
import os
from skimage import draw
import collections
import time

if sys.version_info[0] == 2:
    import cPickle as pickle
    from Queue import Empty as QueueEmpty, Full as QueueFull
elif sys.version_info[0] == 3:
    import pickle
    from queue import Empty as QueueEmpty, Full as QueueFull
    xrange = range

ALL = "ALL"

# filepath to the quokka image
QUOKKA_FP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "quokka.jpg"
)

DEFAULT_FONT_FP = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "DejaVuSans.ttf"
)

# We instantiate a current/global random state here once.
# One can also call np.random, but that is (in contrast to np.random.RandomState)
# a module and hence cannot be copied via deepcopy. That's why we use RandomState
# here (and in all augmenters) instead of np.random.
CURRENT_RANDOM_STATE = np.random.RandomState(42)

def is_np_array(val):
    """
    Checks whether a variable is a numpy array.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a numpy array. Otherwise False.

    """
    # using np.generic here seems to also fire for scalar numpy values even
    # though those are not arrays
    #return isinstance(val, (np.ndarray, np.generic))
    return isinstance(val, np.ndarray)

def is_single_integer(val):
    """
    Checks whether a variable is an integer.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is an integer. Otherwise False.

    """
    return isinstance(val, numbers.Integral) and not isinstance(val, bool)

def is_single_float(val):
    """
    Checks whether a variable is a float.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a float. Otherwise False.

    """
    return isinstance(val, numbers.Real) and not is_single_integer(val) and not isinstance(val, bool)

def is_single_number(val):
    """
    Checks whether a variable is a number, i.e. an integer or float.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a number. Otherwise False.

    """
    return is_single_integer(val) or is_single_float(val)

def is_iterable(val):
    """
    Checks whether a variable is iterable.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is an iterable. Otherwise False.

    """
    return isinstance(val, collections.Iterable)

# TODO convert to is_single_string() or rename is_single_integer/float/number()
def is_string(val):
    """
    Checks whether a variable is a string.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a string. Otherwise False.

    """
    return isinstance(val, six.string_types)

def is_integer_array(val):
    """
    Checks whether a variable is a numpy integer array.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a numpy integer array. Otherwise False.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.integer)

def is_float_array(val):
    """
    Checks whether a variable is a numpy float array.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a numpy float array. Otherwise False.

    """
    return is_np_array(val) and issubclass(val.dtype.type, np.floating)

def is_callable(val):
    """
    Checks whether a variable is a callable, e.g. a function.

    Parameters
    ----------
    val : anything
        The variable to
        check.

    Returns
    -------
    out : bool
        True if the variable is a callable. Otherwise False.

    """
    # python 3.x with x <= 2 does not support callable(), apparently
    if sys.version_info[0] == 3 and sys.version_info[1] <= 2:
        return hasattr(val, '__call__')
    else:
        return callable(val)

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
        The seed to
        use.
    """
    CURRENT_RANDOM_STATE.seed(seedval)

def current_random_state():
    """
    Returns the current/global random state of the library.

    Returns
    ----------
    out : np.random.RandomState
        The current/global random state.

    """
    return CURRENT_RANDOM_STATE

def new_random_state(seed=None, fully_random=False):
    """
    Returns a new random state.

    Parameters
    ----------
    seed : None or int, optional(default=None)
        Optional seed value to use.
        The same datatypes are allowed as for np.random.RandomState(seed).

    fully_random : bool, optional(default=False)
        Whether to use numpy's random initialization for the
        RandomState (used if set to True). If False, a seed is sampled from
        the global random state, which is a bit faster and hence the default.

    Returns
    -------
    out : np.random.RandomState
        The new random state.

    """
    if seed is None:
        if not fully_random:
            # sample manually a seed instead of just RandomState(),
            # because the latter one
            # is way slower.
            seed = CURRENT_RANDOM_STATE.randint(0, 10**6, 1)[0]
    return np.random.RandomState(seed)

def dummy_random_state():
    """
    Returns a dummy random state that is always based on a seed of 1.

    Returns
    -------
    out : np.random.RandomState
        The new random state.

    """
    return np.random.RandomState(1)

def copy_random_state(random_state, force_copy=False):
    """
    Creates a copy of a random state.

    Parameters
    ----------
    random_state : np.random.RandomState
        The random state to
        copy.

    force_copy : bool, optional(default=False)
        If True, this function will always create a copy of every random
        state. If False, it will not copy numpy's default random state,
        but all other random states.

    Returns
    -------
    rs_copy : np.random.RandomState
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
    return derive_random_states(random_state, n=1)[0]

# TODO use this everywhere instead of manual seed + create
def derive_random_states(random_state, n=1):
    seed = random_state.randint(0, 10**6, 1)[0]
    return [new_random_state(seed+i) for i in sm.xrange(n)]

def forward_random_state(random_state):
    random_state.uniform()

# TODO
# def from_json(json_str):
#    pass

def quokka(size=None):
    """
    Returns an image of a quokka as a numpy array.

    Parameters
    ----------
    size : None or float or tuple of two ints, optional(default=None)
        Size of the output image. Input into scipy.misc.imresize.
        Usually expected to be a tuple (H, W), where H is the desired height
        and W is the width. If None, then the image will not be resized.

    Returns
    -------
    img : (H,W,3) ndarray
        The image array of dtype uint8.

    """
    img = ndimage.imread(QUOKKA_FP, mode="RGB")
    if size is not None:
        img = misc.imresize(img, size)
    return img

def quokka_square(size=None):
    """
    Returns an (square) image of a quokka as a numpy array.

    Parameters
    ----------
    size : None or float or tuple of two ints, optional(default=None)
        Size of the output image. Input into scipy.misc.imresize.
        Usually expected to be a tuple (H, W), where H is the desired height
        and W is the width. If None, then the image will not be resized.

    Returns
    -------
    img : (H,W,3) ndarray
        The image array of dtype uint8.

    """
    img = ndimage.imread(QUOKKA_FP, mode="RGB")
    img = img[0:643, 0:643]
    if size is not None:
        img = misc.imresize(img, size)
    return img

def angle_between_vectors(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'.

    From http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python

    Parameters
    ----------
    {v1, v2} : (N,) ndarray
        Input
        vectors.

    Returns
    -------
    out : float
        Angle in radians.

    Examples
    --------
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966

    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0

    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793

    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def draw_text(img, y, x, text, color=[0, 255, 0], size=25): # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
    """
    Draw text on an image.

    This uses by default DejaVuSans as its font, which is included in the
    library.

    Parameters
    ----------
    img : (H,W,3) ndarray
        The image array to draw text on.
        Expected to be of dtype uint8 or float32 (value range 0.0 to 255.0).

    {y, x} : int
        x- and y- coordinate of the top left corner of the
        text.

    color : iterable of 3 ints, optional(default=[0, 255, 0])
        Color of the text to draw. For RGB-images this is expected to be
        an RGB color.

    size : int, optional(default=25)
        Font size of the text to
        draw.

    Returns
    -------
    img_np : (H,W,3) ndarray
        Input image with text drawn on it.

    """
    # keeping PIL here so that it is not a dependency of the library right now
    from PIL import Image, ImageDraw, ImageFont

    do_assert(img.dtype in [np.uint8, np.float32])

    input_dtype = img.dtype
    if img.dtype == np.float32:
        img = img.astype(np.uint8)

    for i in range(len(color)):
        val = color[i]
        if isinstance(val, float):
            val = int(val * 255)
            val = np.clip(val, 0, 255)
            color[i] = val

    img = Image.fromarray(img)
    font = ImageFont.truetype(DEFAULT_FONT_FP, size)
    context = ImageDraw.Draw(img)
    context.text((x, y), text, fill=tuple(color), font=font)
    img_np = np.asarray(img)
    img_np.setflags(write=True)  # PIL/asarray returns read only array

    if img_np.dtype != input_dtype:
        img_np = img_np.astype(input_dtype)

    return img_np

def imresize_many_images(images, sizes=None, interpolation=None):
    """
    Resize many images to a specified size.

    Parameters
    ----------
    images : (N,H,W,C) ndarray
        Array of the images to resize.
        Expected to usually be of dtype uint8.

    sizes : iterable of two ints
        The new size in (height, width)
        format.

    interpolation : None or string or int, optional(default=None)
        The interpolation to use during resize.
        If int, then expected to be one of:
            * cv2.INTER_NEAREST (nearest neighbour interpolation)
            * cv2.INTER_LINEAR (linear interpolation)
            * cv2.INTER_AREA (area interpolation)
            * cv2.INTER_CUBIC (cubic interpolation)
        If string, then expected to be one of:
            * "nearest" (identical to cv2.INTER_NEAREST)
            * "linear" (identical to cv2.INTER_LINEAR)
            * "area" (identical to cv2.INTER_AREA)
            * "cubic" (identical to cv2.INTER_CUBIC)
        If None, the interpolation will be chosen automatically. For size
        increases, area interpolation will be picked and for size decreases,
        linear interpolation will be picked.

    Returns
    -------
    result : (N,H',W',C) ndarray
        Array of the resized images.

    """
    s = images.shape
    do_assert(len(s) == 4, s)
    nb_images = s[0]
    im_height, im_width = s[1], s[2]
    nb_channels = s[3]
    height, width = sizes[0], sizes[1]

    if height == im_height and width == im_width:
        return np.copy(images)

    ip = interpolation
    do_assert(ip is None or ip in ["nearest", "linear", "area", "cubic", cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC])
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

    result = np.zeros((nb_images, height, width, nb_channels), dtype=images.dtype)
    for img_idx in sm.xrange(nb_images):
        # TODO fallback to scipy here if image isn't uint8
        result_img = cv2.resize(images[img_idx], (width, height), interpolation=ip)
        if len(result_img.shape) == 2:
            result_img = result_img[:, :, np.newaxis]
        result[img_idx] = result_img.astype(images.dtype)
    return result


def imresize_single_image(image, sizes, interpolation=None):
    """
    Resizes a single image.

    Parameters
    ----------
    image : (H,W,C) ndarray or (H,W) ndarray
        Array of the image to resize.
        Expected to usually be of dtype uint8.

    sizes : iterable of two ints
        See `imresize_many_images()`.

    interpolation : None or string or int, optional(default=None)
        See `imresize_many_images()`.

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


def draw_grid(images, rows=None, cols=None):
    """
    Converts multiple input images into a single image showing them in a grid.

    Parameters
    ----------
    images : (N,H,W,3) ndarray or iterable of (H,W,3) array
        The input images to convert to a grid.
        Expected to be RGB and have dtype uint8.

    rows : None or int, optional(default=None)
        The number of rows to show in the grid.
        If None, it will be automatically derived.

    cols : None or int, optional(default=None)
        The number of cols to show in the grid.
        If None, it will be automatically derived.

    Returns
    -------
    grid : (H',W',3) ndarray
        Image of the generated grid.

    """
    if is_np_array(images):
        do_assert(images.ndim == 4)
    else:
        do_assert(is_iterable(images) and is_np_array(images[0]) and images[0].ndim == 3)

    nb_images = len(images)
    do_assert(nb_images > 0)
    cell_height = max([image.shape[0] for image in images])
    cell_width = max([image.shape[1] for image in images])
    channels = set([image.shape[2] for image in images])
    do_assert(len(channels) == 1, "All images are expected to have the same number of channels, but got channel set %s with length %d instead." % (str(channels), len(channels)))
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
    grid = np.zeros((height, width, nb_channels), dtype=np.uint8)
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

    This function wraps around scipy.misc.imshow(), which requires the
    `see <image>` command to work. On Windows systems, this tends to not be
    the case.

    Parameters
    ----------
    images : (N,H,W,3) ndarray or iterable of (H,W,3) array
        See `draw_grid()`.

    rows : None or int, optional(default=None)
        See `draw_grid()`.

    cols : None or int, optional(default=None)
        See `draw_grid()`.

    """
    grid = draw_grid(images, rows=rows, cols=cols)
    misc.imshow(grid)

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

    message : string, optional(default="Assertion failed.")
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
    activator : None or callable, optional(default=None)
        A function that gives permission to execute an augmenter.
        The expected interface is
            `f(images, augmenter, parents, default)`,
        where `images` are the input images to augment, `augmenter` is the
        instance of the augmenter to execute, `parents` are previously
        executed augmenters and `default` is an expected default value to be
        returned if the activator function does not plan to make a decision
        for the given inputs.

    propagator : None or callable, optional(default=None)
        A function that gives permission to propagate the augmentation further
        to the children of an augmenter. This happens after the activator.
        In theory, an augmenter may augment images itself (if allowed by the
        activator) and then execute child augmenters afterwards (if allowed by
        the propagator). If the activator returned False, the propagation step
        will never be executed.
        The expected interface is
            `f(images, augmenter, parents, default)`,
        with all arguments having identical meaning to the activator.

    preprocessor : None or callable, optional(default=None)
        A function to call before an augmenter performed any augmentations.
        The interface is
            `f(images, augmenter, parents)`,
        with all arguments having identical meaning to the activator.
        It is expected to return the input images, optionally modified.

    postprocessor : None or callable, optional(default=None)
        A function to call after an augmenter performed augmentations.
        The interface is the same as for the preprocessor.

    Examples
    --------
    >>> seq = iaa.Sequential([
    >>>     iaa.GaussianBlur(3.0, name="blur"),
    >>>     iaa.Dropout(0.05, name="dropout"),
    >>>     iaa.Affine(translate_px=-5, name="affine")
    >>> ])
    >>>
    >>> def activator(images, augmenter, parents, default):
    >>>     return False if augmenter.name in ["blur", "dropout"] else default
    >>>
    >>> seq_det = seq.to_deterministic()
    >>> images_aug = seq_det.augment_images(images)
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
        out : bool
            If True, the augmenter may be executed. If False, it may
            not be executed.

        """
        if self.activator is None:
            return default
        else:
            return self.activator(images, augmenter, parents, default)

    # TODO is a propagating hook necessary? seems to be covered by activated
    # hook already
    def is_propagating(self, images, augmenter, parents, default):
        """
        Returns whether an augmenter may call its children to augment an
        image. This is independent of the augmenter itself possible changing
        the image, without calling its children. (Most (all?) augmenters with
        children currently dont perform any changes themselves.)

        Returns
        -------
        out : bool
            If True, the augmenter may be propagate to its children.
            If False, it may not.

        """
        if self.propagator is None:
            return default
        else:
            return self.propagator(images, augmenter, parents, default)

    def preprocess(self, images, augmenter, parents):
        """
        A function to be called before the augmentation of images starts (per
        augmenter).

        Returns
        -------
        out : (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
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
        out : (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            The input images, optionally modified.

        """
        if self.postprocessor is None:
            return images
        else:
            return self.postprocessor(images, augmenter, parents)


class HooksKeypoints(HooksImages):
    """
    Class to intervene with keypoint augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    This class is currently the same as the one for images. This may or may
    not change in the future.

    """
    pass


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
        # these checks are currently removed because they are very slow for some
        # reason
        #assert is_single_integer(x), type(x)
        #assert is_single_integer(y), type(y)
        self.x = x
        self.y = y

    @property
    def x_int(self):
        return int(round(self.x))

    @property
    def y_int(self):
        return int(round(self.y))

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
        from_shape : tuple
            Shape of the original image. (Before resize.)

        to_shape : tuple
            Shape of the new image. (After resize.)

        Returns
        -------
        out : Keypoint
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
        x : number, optional(default=0)
            Move by this value on the x axis.

        y : number, optional(default=0)
            Move by this value on the y axis.

        Returns
        -------
        out : Keypoint
            Keypoint object with new coordinates.

        """
        return Keypoint(self.x + x, self.y + y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Keypoint(x=%.8f, y=%.8f)" % (self.x, self.y)


class KeypointsOnImage(object):
    """
    Object that represents all keypoints on a single image.

    Parameters
    ----------
    keypoints : list of Keypoint
        List of keypoints on the image.

    shape : tuple of int
        The shape of the image on which the keypoints are placed.

    Examples
    --------
    >>> kps = [Keypoint(x=10, y=20), Keypoint(x=34, y=60)]
    >>> kps_oi = KeypointsOnImage(kps, shape=image.shape)

    """
    def __init__(self, keypoints, shape):
        #assert len(shape) == 3, "KeypointsOnImage requires shape tuples of form (H, W, C) but got %s. Use C=1 for 2-dimensional images." % (str(shape),)
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

    def on(self, image):
        """
        Project keypoints from one image to a new one.

        Parameters
        ----------
        image : ndarray or tuple
            New image onto which the keypoints are to be projected.
            May also simply be that new image's shape tuple.

        Returns
        -------
        keypoints : KeypointsOnImage
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

    def draw_on_image(self, image, color=[0, 255, 0], size=3, copy=True, raise_if_out_of_image=False): # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
        """
        Draw all keypoints onto a given image. Each keypoint is marked by a
        square of a chosen color and size.

        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the keypoints.
            This image should usually have the same shape as
            set in KeypointsOnImage.shape.

        color : int or list of ints or tuple of ints or (3,) ndarray, optional(default=[0, 255, 0])
            The RGB color of all keypoints. If a single int `C`, then that is
            equivalent to (C,C,C).

        size : int, optional(default=3)
            The size of each point. If set to C, each square will have
            size CxC.

        copy : bool, optional(default=True)
            Whether to copy the image before drawing the points.

        raise_if_out_of_image : bool, optional(default=False)
            Whether to raise an exception if any keypoint is outside of the
            image.

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
        x : number, optional(default=0)
            Move each keypoint by this value on the x axis.

        y : number, optional(default=0)
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
        Convert the coordinates of all keypoints in this object to
        an array of shape (N,2).

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
        Convert an array (N,2) with a given image shape to a KeypointsOnImage
        object.

        Parameters
        ----------
        coords : (N, 2) ndarray
            Coordinates of N keypoints on the original image.
            Each first entry (i, 0) is expected to be the x coordinate.
            Each second entry (i, 1) is expected to be the y coordinate.

        shape : tuple
            Shape tuple of the image on which the keypoints are placed.

        Returns
        -------
        out : KeypointsOnImage
            KeypointsOnImage object that contains all keypoints from the array.

        """
        keypoints = [Keypoint(x=coords[i, 0], y=coords[i, 1]) for i in sm.xrange(coords.shape[0])]
        return KeypointsOnImage(keypoints, shape)

    def to_keypoint_image(self, size=1):
        """
        Draws a new black image of shape (H,W,N) in which all keypoint coordinates
        are set to 255.
        (H=shape height, W=shape width, N=number of keypoints)

        This function can be used as a helper when augmenting keypoints with
        a method that only supports the augmentation of images.

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

            #if 0 <= y < height and 0 <= x < width:
            #    image[y, x, i] = 255
            if x1 < x2 and y1 < y2:
                image[y1:y2, x1:x2, i] = 128
            if 0 <= y < height and 0 <= x < width:
                image[y, x, i] = 255
        return image

    @staticmethod
    def from_keypoint_image(image, if_not_found_coords={"x": -1, "y": -1}, threshold=1, nb_channels=None): # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
        """
        Converts an image generated by `to_keypoint_image()` back to
        an KeypointsOnImage object.

        Parameters
        ----------
        image : (H,W,N) ndarray
            The keypoints image. N is the number of
            keypoints.

        if_not_found_coords : tuple or list or dict or None
            Coordinates to use for keypoints that cannot be found in `image`.
            If this is a list/tuple, it must have two integer values. If it
            is a dictionary, it must have the keys "x" and "y". If this
            is None, then the keypoint will not be added to the final
            KeypointsOnImage object.

        threshold : int
            The search for keypoints works by searching for the argmax in
            each channel. This parameters contains the minimum value that
            the max must have in order to be viewed as a keypoint.

        nb_channels : None or int
            Number of channels of the image on which the keypoints are placed.
            Some keypoint augmenters require that information.
            If set to None, the keypoint's shape will be set
            to `(height, width)`, otherwise `(height, width, nb_channels)`.

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
            raise Exception("Expected if_not_found_coords to be None or tuple or list or dict, got %s." % (type(if_not_found_coords),))

        keypoints = []
        for i in sm.xrange(nb_keypoints):
            maxidx_flat = np.argmax(image[..., i])
            maxidx_ndim = np.unravel_index(maxidx_flat, (height, width))
            found = (image[maxidx_ndim[0], maxidx_ndim[1], i] >= threshold)
            if found:
                keypoints.append(Keypoint(x=maxidx_ndim[1], y=maxidx_ndim[0]))
            else:
                if drop_if_not_found:
                    pass # dont add the keypoint to the result list, i.e. drop it
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
        out : KeypointsOnImage
            Shallow copy.

        """
        return copy.copy(self)

    def deepcopy(self):
        """
        Create a deep copy of the KeypointsOnImage object.

        Returns
        -------
        out : KeypointsOnImage
            Deep copy.

        """
        # for some reason deepcopy is way slower here than manual copy
        #return copy.deepcopy(self)
        kps = [Keypoint(x=kp.x, y=kp.y) for kp in self.keypoints]
        return KeypointsOnImage(kps, tuple(self.shape))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "KeypointsOnImage(%s, shape=%s)" % (str(self.keypoints), self.shape)

# TODO functions: square(), to_aspect_ratio(), extend()/add_border(), contains_point()
class BoundingBox(object):
    def __init__(self, x1, y1, x2, y2, label=None):
        if x1 > x2:
            x2, x1 = x1, x2
        do_assert(x2 > x1)
        if y1 > y2:
            y2, y1 = y1, y2
        do_assert(y2 > y1)

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label

    @property
    def x1_int(self):
        return int(round(self.x1))

    @property
    def y1_int(self):
        return int(round(self.y1))

    @property
    def x2_int(self):
        return int(round(self.x2))

    @property
    def y2_int(self):
        return int(round(self.y2))

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def center_x(self):
        return self.x1 + self.width/2

    @property
    def center_y(self):
        return self.y1 + self.height/2

    @property
    def area(self):
        return self.height * self.width

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
        from_shape : tuple
            Shape of the original image. (Before resize.)

        to_shape : tuple
            Shape of the new image. (After resize.)

        Returns
        -------
        out : BoundingBox
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
        return BoundingBox(
            x1=self.x1 - all_sides - left,
            x2=self.x2 + all_sides + right,
            y1=self.y1 - all_sides - top,
            y2=self.y2 + all_sides + bottom
        )

    def intersection(self, other, default=None):
        x1_i = max(self.x1, other.x1)
        y1_i = max(self.y1, other.y1)
        x2_i = min(self.x2, other.x2)
        y2_i = min(self.y2, other.y2)
        if x1_i >= x2_i or y1_i >= y2_i:
            return default
        else:
            return BoundingBox(x1=x1_i, y1=y1_i, x2=x2_i, y2=y2_i)

    def union(self, other):
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def iou(self, other):
        inters = self.intersection(other)
        if inters is None:
            return 0
        else:
            return inters.area / self.union(other).area

    def is_fully_within_image(self, image):
        if isinstance(image, tuple):
            shape = image
        else:
            shape = image.shape
        height, width = shape[0:2]
        return self.x1 >= 0 and self.x2 <= width and self.y1 >= 0 and self.y2 <= height

    def is_partly_within_image(self, image):
        if isinstance(image, tuple):
            shape = image
        else:
            shape = image.shape
        height, width = shape[0:2]
        img_bb = BoundingBox(x1=0, x2=width, y1=0, y2=height)
        return self.intersection(img_bb) is not None

    def is_out_of_image(self, image, fully=True, partly=False):
        if self.is_fully_within_image(image):
            return False
        elif self.is_partly_within_image(image):
            return partly
        else:
            return fully

    def cut_out_of_image(self, image):
        if isinstance(image, tuple):
            shape = image
        else:
            shape = image.shape

        height, width = shape[0:2]
        do_assert(height > 0)
        do_assert(width > 0)

        x1 = np.clip(self.x1, 0, width)
        x2 = np.clip(self.x2, 0, width)
        y1 = np.clip(self.y1, 0, height)
        y2 = np.clip(self.y2, 0, height)

        return self.copy(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            label=self.label
        )

    def shift(self, top=None, right=None, bottom=None, left=None):
        top = top if top is not None else 0
        right = right if right is not None else 0
        bottom = bottom if bottom is not None else 0
        left = left if left is not None else 0
        return self.copy(
            x1=self.x1+left-right,
            x2=self.x2+left-right,
            y1=self.y1+top-bottom,
            y2=self.y2+top-bottom,
            label=self.label
        )

    def draw_on_image(self, image, color=[0, 255, 0], alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False): # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
        if raise_if_out_of_image and self.is_out_of_image(image):
            raise Exception("Cannot draw bounding box x1=%.8f, y1=%.8f, x2=%.8f, y2=%.8f on image with shape %s." % (self.x1, self.y1, self.x2, self.y2, image.shape))

        result = np.copy(image) if copy else image

        if isinstance(color, (tuple, list)):
            color = np.uint8(color)

        for i in range(thickness):
            y = [self.y1_int-i, self.y1_int-i, self.y2_int+i, self.y2_int+i]
            x = [self.x1_int-i, self.x2_int+i, self.x2_int+i, self.x1_int-i]
            rr, cc = draw.polygon_perimeter(y, x, shape=result.shape)
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

    def extract_from_image(self, image):
        pad_top = 0
        pad_right = 0
        pad_bottom = 0
        pad_left = 0

        height, width = image.shape[0], image.shape[1]
        x1, x2, y1, y2 = self.x1_int, self.x2_int, self.y1_int, self.y2_int

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

    def to_keypoints(self):
        return [
            Keypoint(x=self.x1, y=self.y1),
            Keypoint(x=self.x2, y=self.y1),
            Keypoint(x=self.x2, y=self.y2),
            Keypoint(x=self.x1, y=self.y2)
        ]

    def copy(self, x1=None, y1=None, x2=None, y2=None, label=None):
        return BoundingBox(
            x1=self.x1 if x1 is None else x1,
            x2=self.x2 if x2 is None else x2,
            y1=self.y1 if y1 is None else y1,
            y2=self.y2 if y2 is None else y2,
            label=self.label if label is None else label
        )

    def deepcopy(self, x1=None, y1=None, x2=None, y2=None, label=None):
        return self.copy(x1=x1, y1=y1, x2=x2, y2=y2, label=label)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox(x1=%.4f, y1=%.4f, x2=%.4f, y2=%.4f, label=%s)" % (self.x1, self.y1, self.x2, self.y2, self.label)

class BoundingBoxesOnImage(object):
    """
    Object that represents all bounding boxes on a single image.

    Parameters
    ----------
    bounding_boxes : list of BoundingBox
        List of bounding boxes on the image.

    shape : tuple of int
        The shape of the image on which the bounding boxes are placed.

    Examples
    --------
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

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    def on(self, image):
        """
        Project bounding boxes from one image to a new one.

        Parameters
        ----------
        image : ndarray or tuple
            New image onto which the bounding boxes are to be projected.
            May also simply be that new image's shape tuple.

        Returns
        -------
        keypoints : BoundingBoxesOnImage
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

    def draw_on_image(self, image, color=[0, 255, 0], alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False):
        """
        Draw all bounding boxes onto a given image.

        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the bounding boxes.
            This image should usually have the same shape as
            set in BoundingBoxesOnImage.shape.

        color : int or list of ints or tuple of ints or (3,) ndarray, optional(default=[0, 255, 0])
            The RGB color of all bounding boxes. If a single int `C`, then that is
            equivalent to (C,C,C).

        size : float, optional(default=1.0)
            Alpha/transparency of the bounding box.

        thickness : int, optional(default=1)
            Thickness in pixels.

        copy : bool, optional(default=True)
            Whether to copy the image before drawing the points.

        raise_if_out_of_image : bool, optional(default=False)
            Whether to raise an exception if any bounding box is outside of the
            image.

        Returns
        -------
        image : (H,W,3) ndarray
            Image with drawn bounding boxes.

        """
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
        bbs_clean = [bb for bb in self.bounding_boxes if not bb.is_out_of_image(self.shape, fully=fully, partly=partly)]
        return BoundingBoxesOnImage(bbs_clean, shape=self.shape)

    def cut_out_of_image(self):
        bbs_cut = [bb.cut_out_of_image(self.shape) for bb in self.bounding_boxes if bb.is_partly_within_image(self.shape)]
        return BoundingBoxesOnImage(bbs_cut, shape=self.shape)

    def shift(self, top=None, right=None, bottom=None, left=None):
        bbs_new = [bb.shift(top=top, right=right, bottom=bottom, left=left) for bb in self.bounding_boxes]
        return BoundingBoxesOnImage(bbs_new, shape=self.shape)

    def copy(self):
        """
        Create a shallow copy of the BoundingBoxesOnImage object.

        Returns
        -------
        out : BoundingBoxesOnImage
            Shallow copy.

        """
        return copy.copy(self)

    def deepcopy(self):
        """
        Create a deep copy of the BoundingBoxesOnImage object.

        Returns
        -------
        out : KeypointsOnImage
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

############################
# Background augmentation
############################

class Batch(object):
    """
    Class encapsulating a batch before and after augmentation.

    Parameters
    ----------
    images : None or (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
        The images to
        augment.

    keypoints : None or list of KeypointOnImage
        The keypoints to
        augment.

    data : anything
        Additional data that is saved in the batch and may be read out
        after augmentation. This could e.g. contain filepaths to each image
        in `images`. As this object is usually used for background
        augmentation with multiple processes, the augmented Batch objects might
        not be returned in the original order, making this information useful.

    """
    def __init__(self, images=None, keypoints=None, data=None):
        self.images = images
        self.images_aug = None
        self.keypoints = keypoints
        self.keypoints_aug = None
        self.data = data

class BatchLoader(object):
    """
    Class to load batches in the background.

    Loaded batches can be accesses using `BatchLoader.queue`.

    Parameters
    ----------
    load_batch_func : callable
        Function that yields Batch objects (i.e. expected to be a generator).
        Background loading automatically stops when the last batch was yielded.

    queue_size : int, optional(default=50)
        Maximum number of batches to store in the queue. May be set higher
        for small images and/or small batches.

    nb_workers : int, optional(default=1)
        Number of workers to run in the background.

    threaded : bool, optional(default=True)
        Whether to run the background processes using threads (true) or
        full processes (false).

    """

    def __init__(self, load_batch_func, queue_size=50, nb_workers=1, threaded=True):
        do_assert(queue_size > 0)
        do_assert(nb_workers >= 1)
        self.queue = multiprocessing.Queue(queue_size)
        self.join_signal = multiprocessing.Event()
        self.finished_signals = []
        self.workers = []
        self.threaded = threaded
        seeds = current_random_state().randint(0, 10**6, size=(nb_workers,))
        for i in range(nb_workers):
            finished_signal = multiprocessing.Event()
            self.finished_signals.append(finished_signal)
            if threaded:
                worker = threading.Thread(target=self._load_batches, args=(load_batch_func, self.queue, finished_signal, self.join_signal, None))
            else:
                worker = multiprocessing.Process(target=self._load_batches, args=(load_batch_func, self.queue, finished_signal, self.join_signal, seeds[i]))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def all_finished(self):
        """
        Determine whether the workers have finished the loading process.

        Returns
        -------
        out : bool
            True if all workers have finished. Else False.

        """
        return all([event.is_set() for event in self.finished_signals])

    def _load_batches(self, load_batch_func, queue, finished_signal, join_signal, seedval):
        if seedval is not None:
            random.seed(seedval)
            np.random.seed(seedval)
            seed(seedval)

        try:
            for batch in load_batch_func():
                do_assert(isinstance(batch, Batch), "Expected batch returned by lambda function to be of class imgaug.Batch, got %s." % (type(batch),))
                batch_pickled = pickle.dumps(batch, protocol=-1)
                while not join_signal.is_set():
                    try:
                        queue.put(batch_pickled, timeout=0.001)
                        break
                    except QueueFull:
                        pass
                if join_signal.is_set():
                    break
        except Exception as exc:
            traceback.print_exc()
        finally:
            finished_signal.set()

    def terminate(self):
        """
        Stop all workers.

        """
        self.join_signal.set()
        # give minimal time to put generated batches in queue and gracefully shut down
        time.sleep(0.002)

        # clean the queue, this reportedly prevents hanging threads
        while True:
            try:
                self.queue.get(timeout=0.005)
            except QueueEmpty:
                break

        if self.threaded:
            for worker in self.workers:
                worker.join()
            # we don't have to set the finished_signals here, because threads always finish
            # gracefully
        else:
            for worker in self.workers:
                worker.terminate()
                worker.join()

            # wait here a tiny bit to really make sure that everything is killed before setting
            # the finished_signals. calling set() and is_set() (via a subprocess) on them at the
            # same time apparently results in a deadlock (at least in python 2).
            #time.sleep(0.02)
            for finished_signal in self.finished_signals:
                finished_signal.set()

        self.queue.close()

class BackgroundAugmenter(object):
    """
    Class to augment batches in the background (while training on the GPU).

    This is a wrapper around the multiprocessing module.

    Parameters
    ----------
    batch_loader : BatchLoader
        BatchLoader object to load data in the
        background.

    augseq : Augmenter
        An augmenter to apply to all loaded images.
        This may be e.g. a Sequential to apply multiple augmenters.

    queue_size : int
        Size of the queue that is used to temporarily save the augmentation
        results. Larger values offer the background processes more room
        to save results when the main process doesn't load much, i.e. they
        can lead to smoother and faster training. For large images, high
        values can block a lot of RAM though.

    nb_workers : "auto" or int
        Number of background workers to spawn. If auto, it will be set
        to C-1, where C is the number of CPU cores.

    """
    def __init__(self, batch_loader, augseq, queue_size=50, nb_workers="auto"):
        do_assert(queue_size > 0)
        self.augseq = augseq
        self.source_finished_signals = batch_loader.finished_signals
        self.queue_source = batch_loader.queue
        self.queue_result = multiprocessing.Queue(queue_size)

        if nb_workers == "auto":
            try:
                nb_workers = multiprocessing.cpu_count()
            except (ImportError, NotImplementedError):
                nb_workers = 1
            # try to reserve at least one core for the main process
            nb_workers = max(1, nb_workers - 1)
        else:
            do_assert(nb_workers >= 1)
        #print("Starting %d background processes" % (nb_workers,))

        self.nb_workers = nb_workers
        self.workers = []
        self.nb_workers_finished = 0

        self.augment_images = True
        self.augment_keypoints = True

        seeds = current_random_state().randint(0, 10**6, size=(nb_workers,))
        for i in range(nb_workers):
            worker = multiprocessing.Process(target=self._augment_images_worker, args=(augseq, self.queue_source, self.queue_result, self.source_finished_signals, seeds[i]))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def get_batch(self):
        """
        Returns a batch from the queue of augmented batches.

        If workers are still running and there are no batches in the queue,
        it will automatically wait for the next batch.

        Returns
        -------
        out : None or ia.Batch
            One batch or None if all workers have finished.

        """
        batch_str = self.queue_result.get()
        batch = pickle.loads(batch_str)
        if batch is not None:
            return batch
        else:
            self.nb_workers_finished += 1
            if self.nb_workers_finished == self.nb_workers:
                return None
            else:
                return self.get_batch()

    def _augment_images_worker(self, augseq, queue_source, queue_result, source_finished_signals, seedval):
        """
        Worker function that endlessly queries the source queue (input
        batches), augments batches in it and sends the result to the output
        queue.

        """
        np.random.seed(seedval)
        random.seed(seedval)
        augseq.reseed(seedval)
        seed(seedval)

        while True:
            # wait for a new batch in the source queue and load it
            try:
                batch_str = queue_source.get(timeout=0.1)
                batch = pickle.loads(batch_str)
                # augment the batch
                batch_augment_images = batch.images is not None and self.augment_images
                batch_augment_keypoints = batch.keypoints is not None and self.augment_keypoints

                if batch_augment_images and batch_augment_keypoints:
                    augseq_det = augseq.to_deterministic() if not augseq.deterministic else augseq
                    batch.images_aug = augseq_det.augment_images(batch.images)
                    batch.keypoints_aug = augseq_det.augment_keypoints(batch.keypoints)
                elif batch_augment_images:
                    batch.images_aug = augseq.augment_images(batch.images)
                elif batch_augment_keypoints:
                    batch.keypoints_aug = augseq.augment_keypoints(batch.keypoints)

                # send augmented batch to output queue
                batch_str = pickle.dumps(batch, protocol=-1)
                queue_result.put(batch_str)
            except QueueEmpty:
                if all([signal.is_set() for signal in source_finished_signals]):
                    queue_result.put(pickle.dumps(None, protocol=-1))
                    return

    def terminate(self):
        """
        Terminates all background processes immediately.
        This will also free their RAM.

        """
        for worker in self.workers:
            worker.terminate()

        self.queue_result.close()
