from __future__ import print_function, division, absolute_import

import copy

import numpy as np
import scipy.spatial.distance
import six.moves as sm

from .. import imgaug as ia
from .utils import normalize_shape, project_coords


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
        xy_proj = project_coords([(self.x, self.y)], from_shape, to_shape)
        return self.deepcopy(x=xy_proj[0][0], y=xy_proj[0][1])

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
        return self.deepcopy(self.x + x, self.y + y)

    def draw_on_image(self, image, color=(0, 255, 0), alpha=1.0, size=3,
                      copy=True, raise_if_out_of_image=False):
        """
        Draw the keypoint onto a given image.

        The keypoint is drawn as a square.

        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the keypoint.

        color : int or list of int or tuple of int or (3,) ndarray, optional
            The RGB color of the keypoint. If a single int ``C``, then that is
            equivalent to ``(C,C,C)``.

        alpha : float, optional
            The opacity of the drawn keypoint, where ``1.0`` denotes a fully
            visible keypoint and ``0.0`` an invisible one.

        size : int, optional
            The size of the keypoint. If set to ``S``, each square will have
            size ``S x S``.

        copy : bool, optional
            Whether to copy the image before drawing the keypoint.

        raise_if_out_of_image : bool, optional
            Whether to raise an exception if the keypoint is outside of the
            image.

        Returns
        -------
        image : (H,W,3) ndarray
            Image with drawn keypoint.

        """
        if copy:
            image = np.copy(image)

        if image.ndim == 2:
            assert ia.is_single_number(color), (
                "Got a 2D image. Expected then 'color' to be a single number, "
                "but got %s." % (str(color),))
        elif image.ndim == 3 and ia.is_single_number(color):
            color = [color] * image.shape[-1]

        input_dtype = image.dtype
        alpha_color = color
        if alpha < 0.01:
            # keypoint invisible, nothing to do
            return image
        elif alpha > 0.99:
            alpha = 1
        else:
            image = image.astype(np.float32, copy=False)
            alpha_color = alpha * np.array(color)

        height, width = image.shape[0:2]

        y, x = self.y_int, self.x_int

        x1 = max(x - size//2, 0)
        x2 = min(x + 1 + size//2, width)
        y1 = max(y - size//2, 0)
        y2 = min(y + 1 + size//2, height)

        x1_clipped, x2_clipped = np.clip([x1, x2], 0, width)
        y1_clipped, y2_clipped = np.clip([y1, y2], 0, height)

        x1_clipped_ooi = (x1_clipped < 0 or x1_clipped >= width)
        x2_clipped_ooi = (x2_clipped < 0 or x2_clipped >= width+1)
        y1_clipped_ooi = (y1_clipped < 0 or y1_clipped >= height)
        y2_clipped_ooi = (y2_clipped < 0 or y2_clipped >= height+1)
        x_ooi = (x1_clipped_ooi and x2_clipped_ooi)
        y_ooi = (y1_clipped_ooi and y2_clipped_ooi)
        x_zero_size = (x2_clipped - x1_clipped) < 1  # min size is 1px
        y_zero_size = (y2_clipped - y1_clipped) < 1
        if not x_ooi and not y_ooi and not x_zero_size and not y_zero_size:
            if alpha == 1:
                image[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = color
            else:
                image[y1_clipped:y2_clipped, x1_clipped:x2_clipped] = (
                        (1 - alpha)
                        * image[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
                        + alpha_color
                )
        else:
            if raise_if_out_of_image:
                raise Exception(
                    "Cannot draw keypoint x=%.8f, y=%.8f on image with "
                    "shape %s." % (y, x, image.shape))

        if image.dtype.name != input_dtype.name:
            if input_dtype.name == "uint8":
                image = np.clip(image, 0, 255, out=image)
            image = image.astype(input_dtype, copy=False)
        return image

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
        return [self.deepcopy(x=points[i, 0], y=points[i, 1]) for i in sm.xrange(points.shape[0])]

    def copy(self, x=None, y=None):
        """
        Create a shallow copy of the Keypoint object.

        Parameters
        ----------
        x : None or number, optional
            Coordinate of the keypoint on the x axis.
            If ``None``, the instance's value will be copied.

        y : None or number, optional
            Coordinate of the keypoint on the y axis.
            If ``None``, the instance's value will be copied.

        Returns
        -------
        imgaug.Keypoint
            Shallow copy.

        """
        return self.deepcopy(x=x, y=y)

    def deepcopy(self, x=None, y=None):
        """
        Create a deep copy of the Keypoint object.

        Parameters
        ----------
        x : None or number, optional
            Coordinate of the keypoint on the x axis.
            If ``None``, the instance's value will be copied.

        y : None or number, optional
            Coordinate of the keypoint on the y axis.
            If ``None``, the instance's value will be copied.

        Returns
        -------
        imgaug.Keypoint
            Deep copy.

        """
        x = self.x if x is None else x
        y = self.y if y is None else y
        return Keypoint(x=x, y=y)

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
        self.shape = normalize_shape(shape)

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
        shape = normalize_shape(image)
        if shape[0:2] == self.shape[0:2]:
            return self.deepcopy()
        else:
            keypoints = [kp.project(self.shape, shape) for kp in self.keypoints]
            return self.deepcopy(keypoints, shape)

    def draw_on_image(self, image, color=(0, 255, 0), alpha=1.0, size=3,
                      copy=True, raise_if_out_of_image=False):
        """
        Draw all keypoints onto a given image.

        Each keypoint is marked by a square of a chosen color and size.

        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the keypoints.
            This image should usually have the same shape as
            set in KeypointsOnImage.shape.

        color : int or list of int or tuple of int or (3,) ndarray, optional
            The RGB color of all keypoints. If a single int ``C``, then that is
            equivalent to ``(C,C,C)``.

        alpha : float, optional
            The opacity of the drawn keypoint, where ``1.0`` denotes a fully
            visible keypoint and ``0.0`` an invisible one.

        size : int, optional
            The size of each point. If set to ``C``, each square will have
            size ``C x C``.

        copy : bool, optional
            Whether to copy the image before drawing the points.

        raise_if_out_of_image : bool, optional
            Whether to raise an exception if any keypoint is outside of the image.

        Returns
        -------
        image : (H,W,3) ndarray
            Image with drawn keypoints.

        """
        image = np.copy(image) if copy else image
        for keypoint in self.keypoints:
            image = keypoint.draw_on_image(
                image, color=color, alpha=alpha, size=size, copy=False,
                raise_if_out_of_image=raise_if_out_of_image)
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
        return self.deepcopy(keypoints)

    @ia.deprecated(alt_func="KeypointsOnImage.to_xy_array()")
    def get_coords_array(self):
        """
        Convert the coordinates of all keypoints in this object to an array of shape (N,2).

        Returns
        -------
        result : (N, 2) ndarray
            Where N is the number of keypoints. Each first value is the
            x coordinate, each second value is the y coordinate.

        """
        return self.to_xy_array()

    def to_xy_array(self):
        """
        Convert keypoint coordinates to ``(N,2)`` array.

        Returns
        -------
        (N, 2) ndarray
            Array containing the coordinates of all keypoints.
            Shape is ``(N,2)`` with coordinates in xy-form.

        """
        result = np.zeros((len(self.keypoints), 2), dtype=np.float32)
        for i, keypoint in enumerate(self.keypoints):
            result[i, 0] = keypoint.x
            result[i, 1] = keypoint.y
        return result

    @staticmethod
    @ia.deprecated(alt_func="KeypointsOnImage.from_xy_array()")
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
        KeypointsOnImage
            KeypointsOnImage object that contains all keypoints from the array.

        """
        return KeypointsOnImage.from_xy_array(coords, shape)

    @classmethod
    def from_xy_array(cls, xy, shape):
        """
        Convert an array (N,2) with a given image shape to a KeypointsOnImage object.

        Parameters
        ----------
        xy : (N, 2) ndarray
            Coordinates of ``N`` keypoints on the original image, given
            as ``(N,2)`` array of xy-coordinates.

        shape : tuple of int or ndarray
            Shape tuple of the image on which the keypoints are placed.

        Returns
        -------
        KeypointsOnImage
            KeypointsOnImage object that contains all keypoints from the array.

        """
        keypoints = [Keypoint(x=coord[0], y=coord[1]) for coord in xy]
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
        ia.do_assert(len(self.keypoints) > 0)
        height, width = self.shape[0:2]
        image = np.zeros((height, width, len(self.keypoints)), dtype=np.uint8)
        ia.do_assert(size % 2 != 0)
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
        ia.do_assert(len(image.shape) == 3)
        height, width, nb_keypoints = image.shape

        drop_if_not_found = False
        if if_not_found_coords is None:
            drop_if_not_found = True
            if_not_found_x = -1
            if_not_found_y = -1
        elif isinstance(if_not_found_coords, (tuple, list)):
            ia.do_assert(len(if_not_found_coords) == 2)
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
        ia.do_assert(len(self.keypoints) > 0)
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
        ia.do_assert(len(distance_maps.shape) == 3)
        height, width, nb_keypoints = distance_maps.shape

        drop_if_not_found = False
        if if_not_found_coords is None:
            drop_if_not_found = True
            if_not_found_x = -1
            if_not_found_y = -1
        elif isinstance(if_not_found_coords, (tuple, list)):
            ia.do_assert(len(if_not_found_coords) == 2)
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

    def copy(self, keypoints=None, shape=None):
        """
        Create a shallow copy of the KeypointsOnImage object.

        Parameters
        ----------
        keypoints : None or list of imgaug.Keypoint, optional
            List of keypoints on the image. If ``None``, the instance's
            keypoints will be copied.

        shape : tuple of int, optional
            The shape of the image on which the keypoints are placed.
            If ``None``, the instance's shape will be copied.

        Returns
        -------
        imgaug.KeypointsOnImage
            Shallow copy.

        """
        result = copy.copy(self)
        if keypoints is not None:
            result.keypoints = keypoints
        if shape is not None:
            result.shape = shape
        return result

    def deepcopy(self, keypoints=None, shape=None):
        """
        Create a deep copy of the KeypointsOnImage object.

        Parameters
        ----------
        keypoints : None or list of imgaug.Keypoint, optional
            List of keypoints on the image. If ``None``, the instance's
            keypoints will be copied.

        shape : tuple of int, optional
            The shape of the image on which the keypoints are placed.
            If ``None``, the instance's shape will be copied.

        Returns
        -------
        imgaug.KeypointsOnImage
            Deep copy.

        """
        # for some reason deepcopy is way slower here than manual copy
        if keypoints is None:
            keypoints = [kp.deepcopy() for kp in self.keypoints]
        if shape is None:
            shape = tuple(self.shape)
        return KeypointsOnImage(keypoints, shape)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "KeypointsOnImage(%s, shape=%s)" % (str(self.keypoints), self.shape)
