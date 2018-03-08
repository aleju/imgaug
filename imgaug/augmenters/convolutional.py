"""
Augmenters that apply convolutions to images.

Do not import directly from this file, as the categorization is not final.
Use instead
    `from imgaug import augmenters as iaa`
and then e.g. ::

    seq = iaa.Sequential([
        iaa.Sharpen((0.0, 1.0)),
        iaa.Emboss((0.0, 1.0))
    ])

List of augmenters:
    * Convolve
    * Sharpen
    * Emboss
    * EdgeDetect
    * DirectedEdgeDetect
"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, Uniform
import numpy as np
import cv2
import six.moves as sm
import types

from . import meta
from .meta import Augmenter

# TODO tests
class Convolve(Augmenter):
    """
    Apply a Convolution to input images.

    Parameters
    ----------
    matrix : None or (H, W) ndarray or StochasticParameter or callable, optional(default=None)
        The weight matrix of the convolution kernel to
        apply.
            * If None, the input images will not be changed.
            * If a numpy array, that array will be used for all images and
              channels as the kernel.
            * If a stochastic parameter, C new matrices will be generated
              via param.draw_samples(C) for each image, where C is the number
              of channels.
            * If a callable, the parameter will be called for each image
              via param(image, C, random_state). The function must return C
              matrices, one per channel. It may return None, then that channel
              will not be changed.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> matrix = np.array([[0, -1, 0],
    >>>                    [-1, 4, -1],
    >>>                    [0, -1, 0]])
    >>> aug = iaa.Convolve(matrix=matrix)

    convolves all input images with the kernel shown in the `matrix`
    variable.

    >>> def gen_matrix(image, nb_channels, random_state):
    >>>     matrix_A = np.array([[0, -1, 0],
    >>>                          [-1, 4, -1],
    >>>                          [0, -1, 0]])
    >>>     matrix_B = np.array([[0, 1, 0],
    >>>                          [1, -4, 1],
    >>>                          [0, 1, 0]])
    >>>     if image.shape[0] % 2 == 0:
    >>>         return [matrix_A] * nb_channels
    >>>     else:
    >>>         return [matrix_B] * nb_channels
    >>> aug = iaa.Convolve(matrix=gen_matrix)

    convolves images that have an even height with matrix A and images
    with an odd height with matrix B.

    """

    def __init__(self, matrix=None, name=None, deterministic=False, random_state=None):
        super(Convolve, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if matrix is None:
            self.matrix = None #np.array([[1]], dtype=np.float32)
            self.matrix_type = "None"
        elif ia.is_np_array(matrix):
            ia.do_assert(len(matrix.shape) == 2, "Expected convolution matrix to have 2 axis, got %d (shape %s)." % (len(matrix.shape), matrix.shape))
            self.matrix = matrix
            self.matrix_type = "constant"
        elif isinstance(matrix, StochasticParameter):
            self.matrix = matrix
            self.matrix_type = "stochastic"
        elif isinstance(matrix, types.FunctionType):
            self.matrix = matrix
            self.matrix_type = "function"
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(matrix),))

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images)

        result = images
        nb_images = len(images)
        for i in sm.xrange(nb_images):
            _height, _width, nb_channels = images[i].shape
            if self.matrix_type == "None":
                matrices = [None] * nb_channels
            elif self.matrix_type == "constant":
                matrices = [self.matrix] * nb_channels
            elif self.matrix_type == "stochastic":
                matrices = self.matrix.draw_samples((nb_channels), random_state=random_state)
            elif self.matrix_type == "function":
                matrices = self.matrix(images[i], nb_channels, random_state)
                ia.do_assert(
                    (isinstance(matrices, list) and len(matrices) == nb_channels)
                    or (ia.is_np_array(matrices) and matrices.ndim == 3)
                )
            else:
                raise Exception("Invalid matrix type")

            for channel in sm.xrange(nb_channels):
                if matrices[channel] is not None:
                    # ndimage.convolve caused problems here
                    result[i][..., channel] = cv2.filter2D(result[i][..., channel], -1, matrices[channel])

        # TODO move this into the loop to avoid overflows
        # TODO make value range more flexible
        result = meta.clip_augmented_images_(result, 0, 255)
        result = meta.restore_augmented_images_dtypes_(result, input_dtypes)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        # TODO this can fail for some matrices, e.g. [[0, 0, 1]]
        return keypoints_on_images

    def get_parameters(self):
        return [self.matrix, self.matrix_type]

# TODO tests
def Sharpen(alpha=0, lightness=1, name=None, deterministic=False, random_state=None):
    """
    Augmenter that sharpens images and overlays the result with the original
    image.

    Parameters
    ----------
    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=0)
        Visibility of the sharpened image. At 0, only the original image is
        visible, at 1.0 only its sharpened version is visible.
            * If an int or float, exactly that value will be used.
            * If a tuple (a, b), a random value from the range a <= x <= b will
              be sampled per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    lightness : int or float or tuple of two ints/floats or StochasticParameter, optional(default=1)
        Parameter that controls the lightness/brightness of the sharped image.
        Sane values are somewhere in the range (0.5, 2).
        The value 0 results in an edge map. Values higher than 1 create bright
        images. Default value is 1.
            * If an int or float, exactly that value will be used.
            * If a tuple (a, b), a random value from the range a <= x <= b will
              be sampled per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = Sharpen(alpha=(0.0, 1.0))

    sharpens input images and overlays the sharpened image by a variable
    amount over the old image.

    >>> aug = Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))

    sharpens input images with a variable lightness in the range
    0.75 <= x <= 2.0 and with a variable alpha.

    """
    if ia.is_single_number(alpha):
        alpha_param = Deterministic(alpha)
    elif ia.is_iterable(alpha):
        ia.do_assert(len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(alpha),))
        alpha_param = Uniform(alpha[0], alpha[1])
    elif isinstance(alpha, StochasticParameter):
        alpha_param = alpha
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

    if ia.is_single_number(lightness):
        lightness_param = Deterministic(lightness)
    elif ia.is_iterable(lightness):
        ia.do_assert(len(lightness) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(lightness),))
        lightness_param = Uniform(lightness[0], lightness[1])
    elif isinstance(lightness, StochasticParameter):
        lightness_param = lightness
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(lightness),))

    def create_matrices(image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        ia.do_assert(0 <= alpha_sample <= 1.0)
        lightness_sample = lightness_param.draw_sample(random_state=random_state_func)
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [-1, -1, -1],
            [-1, 8+lightness_sample, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        matrix = (1-alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return [matrix] * nb_channels

    return Convolve(create_matrices, name=name, deterministic=deterministic, random_state=random_state)

# TODO tests
def Emboss(alpha=0, strength=1, name=None, deterministic=False, random_state=None):
    """
    Augmenter that embosses images and overlays the result with the original
    image.

    The embossed version pronounces highlights and shadows,
    letting the image look as if it was recreated on a metal plate ("embossed").

    Parameters
    ----------
    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=0)
        Visibility of the sharpened image. At 0, only the original image is
        visible, at 1.0 only its sharpened version is visible.
            * If an int or float, exactly that value will be used.
            * If a tuple (a, b), a random value from the range a <= x <= b will
              be sampled per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    strength : int or float or tuple of two ints/floats or StochasticParameter, optional(default=1)
        Parameter that controls the strength of the embossing.
        Sane values are somewhere in the range (0, 2) with 1 being the standard
        embossing effect. Default value is 1.
            * If an int or float, exactly that value will be used.
            * If a tuple (a, b), a random value from the range a <= x <= b will
              be sampled per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))

    embosses an image with a variable strength in the range 0.5 <= x <= 1.5
    and overlays the result with a variable alpha in the range 0.0 <= a <= 1.0
    over the old image.

    """

    if ia.is_single_number(alpha):
        alpha_param = Deterministic(alpha)
    elif ia.is_iterable(alpha):
        ia.do_assert(len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(alpha),))
        alpha_param = Uniform(alpha[0], alpha[1])
    elif isinstance(alpha, StochasticParameter):
        alpha_param = alpha
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

    if ia.is_single_number(strength):
        strength_param = Deterministic(strength)
    elif ia.is_iterable(strength):
        ia.do_assert(len(strength) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(strength),))
        strength_param = Uniform(strength[0], strength[1])
    elif isinstance(strength, StochasticParameter):
        strength_param = strength
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(strength),))

    def create_matrices(image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        ia.do_assert(0 <= alpha_sample <= 1.0)
        strength_sample = strength_param.draw_sample(random_state=random_state_func)
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [-1-strength_sample, 0-strength_sample, 0],
            [0-strength_sample, 1, 0+strength_sample],
            [0, 0+strength_sample, 1+strength_sample]
        ], dtype=np.float32)
        matrix = (1-alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return [matrix] * nb_channels

    return Convolve(create_matrices, name=name, deterministic=deterministic, random_state=random_state)

# TODO tests
def EdgeDetect(alpha=0, name=None, deterministic=False, random_state=None):
    """
    Augmenter that detects all edges in images, marks them in
    a black and white image and then overlays the result with the original
    image.

    Parameters
    ----------
    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=0)
        Visibility of the sharpened image. At 0, only the original image is
        visible, at 1.0 only its sharpened version is visible.
            * If an int or float, exactly that value will be used.
            * If a tuple (a, b), a random value from the range a <= x <= b will
              be sampled per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = EdgeDetect(alpha=(0.0, 1.0))

    detects edges in an image  and overlays the result with a variable alpha
    in the range 0.0 <= a <= 1.0 over the old image.

    """
    if ia.is_single_number(alpha):
        alpha_param = Deterministic(alpha)
    elif ia.is_iterable(alpha):
        ia.do_assert(len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(alpha),))
        alpha_param = Uniform(alpha[0], alpha[1])
    elif isinstance(alpha, StochasticParameter):
        alpha_param = alpha
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

    def create_matrices(image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        ia.do_assert(0 <= alpha_sample <= 1.0)
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        matrix = (1-alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return [matrix] * nb_channels

    return Convolve(create_matrices, name=name, deterministic=deterministic, random_state=random_state)

# TODO tests
# TODO merge EdgeDetect and DirectedEdgeDetect?
def DirectedEdgeDetect(alpha=0, direction=(0.0, 1.0), name=None, deterministic=False, random_state=None):
    """
    Augmenter that detects edges that have certain directions and marks them
    in a black and white image and then overlays the result with the original
    image.

    Parameters
    ----------
    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=0)
        Visibility of the sharpened image. At 0, only the original image is
        visible, at 1.0 only its sharpened version is visible.
            * If an int or float, exactly that value will be used.
            * If a tuple (a, b), a random value from the range a <= x <= b will
              be sampled per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    direction : int or float or tuple of two ints/floats or StochasticParameter, optional(default=(0.0, 1.0))
        Angle of edges to pronounce, where 0 represents 0 degrees and 1.0
        represents 360 degrees (both clockwise, starting at the top).
        Default value is (0.0, 1.0), i.e. pick a random angle per image.
            * If an int or float, exactly that value will be used.
            * If a tuple (a, b), a random value from the range a <= x <= b will
              be sampled per image.
            * If a StochasticParameter, a value will be sampled from the
              parameter per image.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = DirectedEdgeDetect(alpha=1.0, direction=0)

    turns input images into edge images in which edges are detected from
    top side of the image (i.e. the top sides of horizontal edges are
    added to the output).

    >>> aug = DirectedEdgeDetect(alpha=1.0, direction=90/360)

    same as before, but detecting edges from the right (right side of each
    vertical edge).

    >>> aug = DirectedEdgeDetect(alpha=1.0, direction=(0.0, 1.0))

    same as before, but detecting edges from a variable direction (anything
    between 0 and 1.0, i.e. 0 degrees and 360 degrees, starting from the
    top and moving clockwise).

    >>> aug = DirectedEdgeDetect(alpha=(0.0, 0.3), direction=0)

    generates edge images (edges detected from the top) and overlays them
    with the input images by a variable amount between 0 and 30 percent
    (e.g. for 0.3 then `0.7*old_image + 0.3*edge_image`).

    """
    if ia.is_single_number(alpha):
        alpha_param = Deterministic(alpha)
    elif ia.is_iterable(alpha):
        ia.do_assert(len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(alpha),))
        alpha_param = Uniform(alpha[0], alpha[1])
    elif isinstance(alpha, StochasticParameter):
        alpha_param = alpha
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

    if ia.is_single_number(direction):
        direction_param = Deterministic(direction)
    elif ia.is_iterable(direction):
        ia.do_assert(len(direction) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(direction),))
        direction_param = Uniform(direction[0], direction[1])
    elif isinstance(direction, StochasticParameter):
        direction_param = direction
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(direction),))

    def create_matrices(image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        ia.do_assert(0 <= alpha_sample <= 1.0)
        direction_sample = direction_param.draw_sample(random_state=random_state_func)

        deg = int(direction_sample * 360) % 360
        rad = np.deg2rad(deg)
        x = np.cos(rad - 0.5*np.pi)
        y = np.sin(rad - 0.5*np.pi)
        #x = (deg % 90) / 90 if 0 <= deg <= 180 else -(deg % 90) / 90
        #y = (-1) + (deg % 90) / 90 if 90 < deg < 270 else 1 - (deg % 90) / 90
        direction_vector = np.array([x, y])

        #print("direction_vector", direction_vector)

        #vertical_vector = np.array([0, 1])

        matrix_effect = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (x, y) != (0, 0):
                    cell_vector = np.array([x, y])
                    #deg_cell = angle_between_vectors(vertical_vector, vec_cell)
                    distance_deg = np.rad2deg(ia.angle_between_vectors(cell_vector, direction_vector))
                    distance = distance_deg / 180
                    similarity = (1 - distance)**4
                    matrix_effect[y+1, x+1] = similarity
                    #print("cell", y, x, "distance_deg", distance_deg, "distance", distance, "similarity", similarity)
        matrix_effect = matrix_effect / np.sum(matrix_effect)
        matrix_effect = matrix_effect * (-1)
        matrix_effect[1, 1] = 1
        #for y in [0, 1, 2]:
        #    vals = []
        #    for x in [0, 1, 2]:
        #        vals.append("%.2f" % (matrix_effect[y, x],))
        #    print(" ".join(vals))
        #print("matrix_effect", matrix_effect)

        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)

        matrix = (1-alpha_sample) * matrix_nochange + alpha_sample * matrix_effect

        return [matrix] * nb_channels

    return Convolve(create_matrices, name=name, deterministic=deterministic, random_state=random_state)
