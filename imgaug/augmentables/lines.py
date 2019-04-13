from __future__ import print_function, division, absolute_import

import copy as copylib

import numpy as np
import skimage.draw
import skimage.measure
import cv2

from .. import imgaug as ia
from .utils import normalize_shape, project_coords, interpolate_points


# TODO Add Line class and make LineString a list of Line elements
# TODO add to_distance_maps(), compute_hausdorff_distance(), intersects(),
#      find_self_intersections(), is_self_intersecting(),
#      remove_self_intersections()
class LineString(object):
    """
    Class representing line strings.

    A line string is a collection of connected line segments, each
    having a start and end point. Each point is given as its ``(x, y)``
    absolute (sub-)pixel coordinates. The end point of each segment is
    also the start point of the next segment.

    The line string is not closed, i.e. start and end point are expected to
    differ and will not be connected in drawings.

    Parameters
    ----------
    coords : iterable of tuple of number or ndarray
        The points of the line string.

    label : None or str, optional
        The label of the line string.

    """

    def __init__(self, coords, label=None):
        """Create a new LineString instance."""
        # use the conditions here to avoid unnecessary copies of ndarray inputs
        if ia.is_np_array(coords):
            if coords.dtype.name != "float32":
                coords = coords.astype(np.float32)
        elif len(coords) == 0:
            coords = np.zeros((0, 2), dtype=np.float32)
        else:
            assert ia.is_iterable(coords), (
                "Expected 'coords' to be an iterable, "
                "got type %s." % (type(coords),))
            assert all([len(coords_i) == 2 for coords_i in coords]), (
                "Expected 'coords' to contain (x,y) tuples, "
                "got %s." % (str(coords),))
            coords = np.float32(coords)

        assert coords.ndim == 2 and coords.shape[-1] == 2, (
            "Expected 'coords' to have shape (N, 2), got shape %s." % (
                coords.shape,))

        self.coords = coords
        self.label = label

    @property
    def length(self):
        """
        Get the total euclidean length of the line string.

        Returns
        -------
        float
            The length based on euclidean distance.

        """
        if len(self.coords) == 0:
            return 0
        return np.sum(self.compute_neighbour_distances())

    @property
    def xx(self):
        """Get an array of x-coordinates of all points of the line string."""
        return self.coords[:, 0]

    @property
    def yy(self):
        """Get an array of y-coordinates of all points of the line string."""
        return self.coords[:, 1]

    @property
    def xx_int(self):
        """Get an array of discrete x-coordinates of all points."""
        return np.round(self.xx).astype(np.int32)

    @property
    def yy_int(self):
        """Get an array of discrete y-coordinates of all points."""
        return np.round(self.yy).astype(np.int32)

    @property
    def height(self):
        """Get the height of a bounding box encapsulating the line."""
        if len(self.coords) <= 1:
            return 0
        return np.max(self.yy) - np.min(self.yy)

    @property
    def width(self):
        """Get the width of a bounding box encapsulating the line."""
        if len(self.coords) <= 1:
            return 0
        return np.max(self.xx) - np.min(self.xx)

    def get_pointwise_inside_image_mask(self, image):
        """
        Get for each point whether it is inside of the given image plane.

        Parameters
        ----------
        image : ndarray or tuple of int
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.

        Returns
        -------
        ndarray
            Boolean array with one value per point indicating whether it is
            inside of the provided image plane (``True``) or not (``False``).

        """
        if len(self.coords) == 0:
            return np.zeros((0,), dtype=bool)
        shape = normalize_shape(image)
        height, width = shape[0:2]
        x_within = np.logical_and(0 <= self.xx, self.xx < width)
        y_within = np.logical_and(0 <= self.yy, self.yy < height)
        return np.logical_and(x_within, y_within)

    # TODO add closed=False/True?
    def compute_neighbour_distances(self):
        """
        Get the euclidean distance between each two consecutive points.

        Returns
        -------
        ndarray
            Euclidean distances between point pairs.
            Same order as in `coords`. For ``N`` points, ``N-1`` distances
            are returned.

        """
        if len(self.coords) <= 1:
            return np.zeros((0,), dtype=np.float32)
        return np.sqrt(
            np.sum(
                (self.coords[:-1, :] - self.coords[1:, :]) ** 2,
                axis=1
            )
        )

    def compute_pointwise_distances(self, other, default=None):
        """
        Compute the minimal distance between each point on self and other.

        Parameters
        ----------
        other : tuple of number \
                or imgaug.augmentables.kps.Keypoint \
                or imgaug.augmentables.LineString
            Other object to which to compute the distances.

        default
            Value to return if `other` contains no points.

        Returns
        -------
        list of float
            Distances to `other` or `default` if not distance could be computed.

        """
        import shapely.geometry
        from .kps import Keypoint

        if isinstance(other, Keypoint):
            other = shapely.geometry.Point((other.x, other.y))
        elif isinstance(other, LineString):
            if len(other.coords) == 0:
                return default
            elif len(other.coords) == 1:
                other = shapely.geometry.Point(other.coords[0, :])
            else:
                other = shapely.geometry.LineString(other.coords)
        elif isinstance(other, tuple):
            assert len(other) == 2
            other = shapely.geometry.Point(other)
        else:
            raise ValueError(
                ("Expected Keypoint or LineString or tuple (x,y), "
                 + "got type %s.") % (type(other),))

        return [shapely.geometry.Point(point).distance(other)
                for point in self.coords]

    def compute_distance(self, other, default=None):
        """
        Compute the minimal distance between the line string and `other`.

        Parameters
        ----------
        other : tuple of number \
                or imgaug.augmentables.kps.Keypoint \
                or imgaug.augmentables.LineString
            Other object to which to compute the distance.

        default
            Value to return if this line string or `other` contain no points.

        Returns
        -------
        float
            Distance to `other` or `default` if not distance could be computed.

        """
        # FIXME this computes distance pointwise, does not have to be identical
        #       with the actual min distance (e.g. edge center to other's point)
        distances = self.compute_pointwise_distances(other, default=[])
        if len(distances) == 0:
            return default
        return min(distances)

    # TODO update BB's contains(), which can only accept Keypoint currently
    def contains(self, other, max_distance=1e-4):
        """
        Estimate whether the bounding box contains a point.

        Parameters
        ----------
        other : tuple of number or imgaug.augmentables.kps.Keypoint
            Point to check for.

        max_distance : float
            Maximum allowed euclidean distance between the point and the
            closest point on the line. If the threshold is exceeded, the point
            is not considered to be contained in the line.

        Returns
        -------
        bool
            True if the point is contained in the line string, False otherwise.
            It is contained if its distance to the line or any of its points
            is below a threshold.

        """
        return self.compute_distance(other, default=np.inf) < max_distance

    def project(self, from_shape, to_shape):
        """
        Project the line string onto a differently shaped image.

        E.g. if a point of the line string is on its original image at
        ``x=(10 of 100 pixels)`` and ``y=(20 of 100 pixels)`` and is projected
        onto a new image with size ``(width=200, height=200)``, its new
        position will be ``(x=20, y=40)``.

        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).

        Parameters
        ----------
        from_shape : tuple of int or ndarray
            Shape of the original image. (Before resize.)

        to_shape : tuple of int or ndarray
            Shape of the new image. (After resize.)

        Returns
        -------
        out : imgaug.augmentables.lines.LineString
            Line string with new coordinates.

        """
        coords_proj = project_coords(self.coords, from_shape, to_shape)
        return self.copy(coords=coords_proj)

    def is_fully_within_image(self, image, default=False):
        """
        Estimate whether the line string is fully inside the image area.

        Parameters
        ----------
        image : ndarray or tuple of int
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.

        default
            Default value to return if the line string contains no points.

        Returns
        -------
        bool
            True if the line string is fully inside the image area.
            False otherwise.

        """
        if len(self.coords) == 0:
            return default
        return np.all(self.get_pointwise_inside_image_mask(image))

    def is_partly_within_image(self, image, default=False):
        """
        Estimate whether the line string is at least partially inside the image.

        Parameters
        ----------
        image : ndarray or tuple of int
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.

        default
            Default value to return if the line string contains no points.

        Returns
        -------
        bool
            True if the line string is at least partially inside the image area.
            False otherwise.

        """
        if len(self.coords) == 0:
            return default
        # check mask first to avoid costly computation of intersection points
        # whenever possible
        mask = self.get_pointwise_inside_image_mask(image)
        if np.any(mask):
            return True
        return len(self.clip_out_of_image(image)) > 0

    def is_out_of_image(self, image, fully=True, partly=False, default=True):
        """
        Estimate whether the line is partially/fully outside of the image area.

        Parameters
        ----------
        image : ndarray or tuple of int
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.

        fully : bool, optional
            Whether to return True if the bounding box is fully outside fo the
            image area.

        partly : bool, optional
            Whether to return True if the bounding box is at least partially
            outside fo the image area.

        default
            Default value to return if the line string contains no points.

        Returns
        -------
        bool
            `default` if the line string has no points.
            True if the line string is partially/fully outside of the image
            area, depending on defined parameters.
            False otherwise.

        """
        if len(self.coords) == 0:
            return default

        if self.is_fully_within_image(image):
            return False
        elif self.is_partly_within_image(image):
            return partly
        else:
            return fully

    def clip_out_of_image(self, image):
        """
        Clip off all parts of the line_string that are outside of the image.

        Parameters
        ----------
        image : ndarray or tuple of int
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.

        Returns
        -------
        list of imgaug.augmentables.lines.LineString
            Line strings, clipped to the image shape.
            The result may contain any number of line strins, including zero.

        """
        if len(self.coords) == 0:
            return []

        inside_image_mask = self.get_pointwise_inside_image_mask(image)
        ooi_mask = ~inside_image_mask

        if len(self.coords) == 1:
            if not np.any(inside_image_mask):
                return []
            return [self.copy()]

        if np.all(inside_image_mask):
            return [self.copy()]

        # top, right, bottom, left image edges
        # we subtract eps here, because intersection() works inclusively,
        # i.e. not subtracting eps would be equivalent to 0<=x<=C for C being
        # height or width
        # don't set the eps too low, otherwise points at height/width seem
        # to get rounded to height/width by shapely, which can cause problems
        # when first clipping and then calling is_fully_within_image()
        # returning false
        height, width = normalize_shape(image)[0:2]
        eps = 1e-3
        edges = [
            LineString([(0.0, 0.0), (width - eps, 0.0)]),
            LineString([(width - eps, 0.0), (width - eps, height - eps)]),
            LineString([(width - eps, height - eps), (0.0, height - eps)]),
            LineString([(0.0, height - eps), (0.0, 0.0)])
        ]
        intersections = self.find_intersections_with(edges)

        points = []
        gen = enumerate(zip(self.coords[:-1], self.coords[1:],
                            ooi_mask[:-1], ooi_mask[1:],
                            intersections))
        for i, (line_start, line_end, ooi_start, ooi_end, inter_line) in gen:
            points.append((line_start, False, ooi_start))
            for p_inter in inter_line:
                points.append((p_inter, True, False))

            is_last = (i == len(self.coords) - 2)
            if is_last and not ooi_end:
                points.append((line_end, False, ooi_end))

        lines = []
        line = []
        for i, (coord, was_added, ooi) in enumerate(points):
            # remove any point that is outside of the image,
            # also start a new line once such a point is detected
            if ooi:
                if len(line) > 0:
                    lines.append(line)
                    line = []
                continue

            if not was_added:
                # add all points that were part of the original line string
                # AND that are inside the image plane
                line.append(coord)
            else:
                is_last_point = (i == len(points)-1)
                # ooi is a numpy.bool_, hence the bool(.)
                is_next_ooi = (not is_last_point
                               and bool(points[i+1][2]) is True)

                # Add all points that were new (i.e. intersections), so
                # long that they aren't essentially identical to other point.
                # This prevents adding overlapping intersections multiple times.
                # (E.g. when a line intersects with a corner of the image plane
                # and also with one of its edges.)
                p_prev = line[-1] if len(line) > 0 else None
                # ignore next point if end reached or next point is out of image
                p_next = None
                if not is_last_point and not is_next_ooi:
                    p_next = points[i+1][0]
                dist_prev = None
                dist_next = None
                if p_prev is not None:
                    dist_prev = np.linalg.norm(
                        np.float32(coord) - np.float32(p_prev))
                if p_next is not None:
                    dist_next = np.linalg.norm(
                        np.float32(coord) - np.float32(p_next))

                dist_prev_ok = (dist_prev is None or dist_prev > 1e-2)
                dist_next_ok = (dist_next is None or dist_next > 1e-2)
                if dist_prev_ok and dist_next_ok:
                    line.append(coord)

        if len(line) > 0:
            lines.append(line)

        lines = [line for line in lines if len(line) > 0]
        return [self.deepcopy(coords=line) for line in lines]

    # TODO add tests for this
    def find_intersections_with(self, other):
        """
        Find all intersection points between the line string and `other`.

        Parameters
        ----------
        other : tuple of number or list of tuple of number or \
                list of LineString or LineString
            The other geometry to use during intersection tests.

        Returns
        -------
        list of list of tuple of number
            All intersection points. One list per pair of consecutive start
            and end point, i.e. `N-1` lists of `N` points. Each list may
            be empty or may contain multiple points.

        """
        import shapely.geometry

        geom = _convert_var_to_shapely_geometry(other)

        result = []
        for p_start, p_end in zip(self.coords[:-1], self.coords[1:]):
            ls = shapely.geometry.LineString([p_start, p_end])
            intersections = ls.intersection(geom)
            intersections = list(_flatten_shapely_collection(intersections))

            intersections_points = []
            for inter in intersections:
                if isinstance(inter, shapely.geometry.linestring.LineString):
                    inter_start = (inter.coords[0][0], inter.coords[0][1])
                    inter_end = (inter.coords[-1][0], inter.coords[-1][1])
                    intersections_points.extend([inter_start, inter_end])
                else:
                    assert isinstance(inter, shapely.geometry.point.Point), (
                        "Expected to find shapely.geometry.point.Point or "
                        "shapely.geometry.linestring.LineString intersection, "
                        "actually found %s." % (type(inter),))
                    intersections_points.append((inter.x, inter.y))

            # sort by distance to start point, this makes it later on easier
            # to remove duplicate points
            inter_sorted = sorted(
                intersections_points,
                key=lambda p: np.linalg.norm(np.float32(p) - p_start)
            )

            result.append(inter_sorted)
        return result

    # TODO convert this to x/y params?
    def shift(self, top=None, right=None, bottom=None, left=None):
        """
        Shift/move the line string from one or more image sides.

        Parameters
        ----------
        top : None or int, optional
            Amount of pixels by which to shift the bounding box from the
            top.

        right : None or int, optional
            Amount of pixels by which to shift the bounding box from the
            right.

        bottom : None or int, optional
            Amount of pixels by which to shift the bounding box from the
            bottom.

        left : None or int, optional
            Amount of pixels by which to shift the bounding box from the
            left.

        Returns
        -------
        result : imgaug.augmentables.lines.LineString
            Shifted line string.

        """
        top = top if top is not None else 0
        right = right if right is not None else 0
        bottom = bottom if bottom is not None else 0
        left = left if left is not None else 0
        coords = np.copy(self.coords)
        coords[:, 0] += left - right
        coords[:, 1] += top - bottom
        return self.copy(coords=coords)

    def draw_mask(self, image_shape, size_lines=1, size_points=0,
                  raise_if_out_of_image=False):
        """
        Draw this line segment as a binary image mask.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        size_lines : int, optional
            Thickness of the line segments.

        size_points : int, optional
            Size of the points in pixels.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            Boolean line mask of shape `image_shape` (no channel axis).

        """
        heatmap = self.draw_heatmap_array(
            image_shape,
            alpha_lines=1.0, alpha_points=1.0,
            size_lines=size_lines, size_points=size_points,
            antialiased=False,
            raise_if_out_of_image=raise_if_out_of_image)
        return heatmap > 0.5

    def draw_lines_heatmap_array(self, image_shape, alpha=1.0,
                                 size=1, antialiased=True,
                                 raise_if_out_of_image=False):
        """
        Draw the line segments of the line string as a heatmap array.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        alpha : float, optional
            Opacity of the line string. Higher values denote a more visible
            line string.

        size : int, optional
            Thickness of the line segments.

        antialiased : bool, optional
            Whether to draw the line with anti-aliasing activated.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            Float array of shape `image_shape` (no channel axis) with drawn
            line string. All values are in the interval ``[0.0, 1.0]``.

        """
        assert len(image_shape) == 2 or (
            len(image_shape) == 3 and image_shape[-1] == 1), (
            "Expected (H,W) or (H,W,1) as image_shape, got %s." % (
                image_shape,))

        arr = self.draw_lines_on_image(
            np.zeros(image_shape, dtype=np.uint8),
            color=255, alpha=alpha, size=size,
            antialiased=antialiased,
            raise_if_out_of_image=raise_if_out_of_image
        )
        return arr.astype(np.float32) / 255.0

    def draw_points_heatmap_array(self, image_shape, alpha=1.0,
                                  size=1, raise_if_out_of_image=False):
        """
        Draw the points of the line string as a heatmap array.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the point mask.

        alpha : float, optional
            Opacity of the line string points. Higher values denote a more
            visible points.

        size : int, optional
            Size of the points in pixels.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            Float array of shape `image_shape` (no channel axis) with drawn
            line string points. All values are in the interval ``[0.0, 1.0]``.

        """
        assert len(image_shape) == 2 or (
            len(image_shape) == 3 and image_shape[-1] == 1), (
            "Expected (H,W) or (H,W,1) as image_shape, got %s." % (
                image_shape,))

        arr = self.draw_points_on_image(
            np.zeros(image_shape, dtype=np.uint8),
            color=255, alpha=alpha, size=size,
            raise_if_out_of_image=raise_if_out_of_image
        )
        return arr.astype(np.float32) / 255.0

    def draw_heatmap_array(self, image_shape, alpha_lines=1.0, alpha_points=1.0,
                           size_lines=1, size_points=0, antialiased=True,
                           raise_if_out_of_image=False):
        """
        Draw the line segments and points of the line string as a heatmap array.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        alpha_lines : float, optional
            Opacity of the line string. Higher values denote a more visible
            line string.

        alpha_points : float, optional
            Opacity of the line string points. Higher values denote a more
            visible points.

        size_lines : int, optional
            Thickness of the line segments.

        size_points : int, optional
            Size of the points in pixels.

        antialiased : bool, optional
            Whether to draw the line with anti-aliasing activated.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            Float array of shape `image_shape` (no channel axis) with drawn
            line segments and points. All values are in the
            interval ``[0.0, 1.0]``.

        """
        heatmap_lines = self.draw_lines_heatmap_array(
            image_shape,
            alpha=alpha_lines,
            size=size_lines,
            antialiased=antialiased,
            raise_if_out_of_image=raise_if_out_of_image)
        if size_points <= 0:
            return heatmap_lines

        heatmap_points = self.draw_points_heatmap_array(
            image_shape,
            alpha=alpha_points,
            size=size_points,
            raise_if_out_of_image=raise_if_out_of_image)

        heatmap = np.dstack([heatmap_lines, heatmap_points])
        return np.max(heatmap, axis=2)

    # TODO only draw line on image of size BB around line, then paste into full
    #      sized image
    def draw_lines_on_image(self, image, color=(0, 255, 0),
                            alpha=1.0, size=3,
                            antialiased=True,
                            raise_if_out_of_image=False):
        """
        Draw the line segments of the line string on a given image.

        Parameters
        ----------
        image : ndarray or tuple of int
            The image onto which to draw.
            Expected to be ``uint8`` and of shape ``(H, W, C)`` with ``C``
            usually being ``3`` (other values are not tested).
            If a tuple, expected to be ``(H, W, C)`` and will lead to a new
            ``uint8`` array of zeros being created.

        color : int or iterable of int
            Color to use as RGB, i.e. three values.

        alpha : float, optional
            Opacity of the line string. Higher values denote a more visible
            line string.

        size : int, optional
            Thickness of the line segments.

        antialiased : bool, optional
            Whether to draw the line with anti-aliasing activated.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            `image` with line drawn on it.

        """
        from .. import dtypes as iadt
        from ..augmenters import blend as blendlib

        image_was_empty = False
        if isinstance(image, tuple):
            image_was_empty = True
            image = np.zeros(image, dtype=np.uint8)
        assert image.ndim in [2, 3], (
            ("Expected image or shape of form (H,W) or (H,W,C), "
             + "got shape %s.") % (image.shape,))

        if len(self.coords) <= 1 or alpha < 0 + 1e-4 or size < 1:
            return np.copy(image)

        if raise_if_out_of_image \
                and self.is_out_of_image(image, partly=False, fully=True):
            raise Exception(
                "Cannot draw line string '%s' on image with shape %s, because "
                "it would be out of bounds." % (
                    self.__str__(), image.shape))

        if image.ndim == 2:
            assert ia.is_single_number(color), (
                "Got a 2D image. Expected then 'color' to be a single number, "
                "but got %s." % (str(color),))
            color = [color]
        elif image.ndim == 3 and ia.is_single_number(color):
            color = [color] * image.shape[-1]

        image = image.astype(np.float32)
        height, width = image.shape[0:2]

        # We can't trivially exclude lines outside of the image here, because
        # even if start and end point are outside, there can still be parts of
        # the line inside the image.
        # TODO Do this with edge-wise intersection tests
        lines = []
        for line_start, line_end in zip(self.coords[:-1], self.coords[1:]):
            # note that line() expects order (y1, x1, y2, x2), hence ([1], [0])
            lines.append((line_start[1], line_start[0],
                          line_end[1], line_end[0]))

        # skimage.draw.line can only handle integers
        lines = np.round(np.float32(lines)).astype(np.int32)

        # size == 0 is already covered above
        # Note here that we have to be careful not to draw lines two times
        # at their intersection points, e.g. for (p0, p1), (p1, 2) we could
        # end up drawing at p1 twice, leading to higher values if alpha is used.
        color = np.float32(color)
        heatmap = np.zeros(image.shape[0:2], dtype=np.float32)
        for line in lines:
            if antialiased:
                rr, cc, val = skimage.draw.line_aa(*line)
            else:
                rr, cc = skimage.draw.line(*line)
                val = 1.0

            # mask check here, because line() can generate coordinates
            # outside of the image plane
            rr_mask = np.logical_and(0 <= rr, rr < height)
            cc_mask = np.logical_and(0 <= cc, cc < width)
            mask = np.logical_and(rr_mask, cc_mask)

            if np.any(mask):
                rr = rr[mask]
                cc = cc[mask]
                val = val[mask] if not ia.is_single_number(val) else val
                heatmap[rr, cc] = val * alpha

        if size > 1:
            kernel = np.ones((size, size), dtype=np.uint8)
            heatmap = cv2.dilate(heatmap, kernel)

        if image_was_empty:
            image_blend = image + heatmap * color
        else:
            image_color_shape = image.shape[0:2]
            if image.ndim == 3:
                image_color_shape = image_color_shape + (1,)
            image_color = np.tile(color, image_color_shape)
            image_blend = blendlib.blend_alpha(image_color, image, heatmap)

        image_blend = iadt.restore_dtypes_(image_blend, np.uint8)
        return image_blend

    def draw_points_on_image(self, image, color=(0, 128, 0),
                             alpha=1.0, size=3,
                             copy=True, raise_if_out_of_image=False):
        """
        Draw the points of the line string on a given image.

        Parameters
        ----------
        image : ndarray or tuple of int
            The image onto which to draw.
            Expected to be ``uint8`` and of shape ``(H, W, C)`` with ``C``
            usually being ``3`` (other values are not tested).
            If a tuple, expected to be ``(H, W, C)`` and will lead to a new
            ``uint8`` array of zeros being created.

        color : iterable of int
            Color to use as RGB, i.e. three values.

        alpha : float, optional
            Opacity of the line string points. Higher values denote a more
            visible points.

        size : int, optional
            Size of the points in pixels.

        copy : bool, optional
            Whether it is allowed to draw directly in the input
            array (``False``) or it has to be copied (``True``).
            The routine may still have to copy, even if ``copy=False`` was
            used. Always use the return value.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            Float array of shape `image_shape` (no channel axis) with drawn
            line string points. All values are in the interval ``[0.0, 1.0]``.

        """
        from .kps import KeypointsOnImage
        kpsoi = KeypointsOnImage.from_xy_array(self.coords, shape=image.shape)
        image = kpsoi.draw_on_image(
            image, color=color, alpha=alpha,
            size=size, copy=copy,
            raise_if_out_of_image=raise_if_out_of_image)

        return image

    def draw_on_image(self, image,
                      color=(0, 255, 0), color_lines=None, color_points=None,
                      alpha=1.0, alpha_lines=None, alpha_points=None,
                      size=1, size_lines=None, size_points=None,
                      antialiased=True,
                      raise_if_out_of_image=False):
        """
        Draw the line string on an image.

        Parameters
        ----------
        image : ndarray
            The `(H,W,C)` `uint8` image onto which to draw the line string.

        color : iterable of int, optional
            Color to use as RGB, i.e. three values.
            The color of the line and points are derived from this value,
            unless they are set.

        color_lines : None or iterable of int
            Color to use for the line segments as RGB, i.e. three values.
            If ``None``, this value is derived from `color`.

        color_points : None or iterable of int
            Color to use for the points as RGB, i.e. three values.
            If ``None``, this value is derived from ``0.5 * color``.

        alpha : float, optional
            Opacity of the line string. Higher values denote more visible
            points.
            The alphas of the line and points are derived from this value,
            unless they are set.

        alpha_lines : None or float, optional
            Opacity of the line string. Higher values denote more visible
            line string.
            If ``None``, this value is derived from `alpha`.

        alpha_points : None or float, optional
            Opacity of the line string points. Higher values denote more
            visible points.
            If ``None``, this value is derived from `alpha`.

        size : int, optional
            Size of the line string.
            The sizes of the line and points are derived from this value,
            unless they are set.

        size_lines : None or int, optional
            Thickness of the line segments.
            If ``None``, this value is derived from `size`.

        size_points : None or int, optional
            Size of the points in pixels.
            If ``None``, this value is derived from ``3 * size``.

        antialiased : bool, optional
            Whether to draw the line with anti-aliasing activated.
            This does currently not affect the point drawing.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            Image with line string drawn on it.

        """
        assert color is not None
        assert alpha is not None
        assert size is not None

        color_lines = color_lines if color_lines is not None \
            else np.float32(color)
        color_points = color_points if color_points is not None \
            else np.float32(color) * 0.5

        alpha_lines = alpha_lines if alpha_lines is not None \
            else np.float32(alpha)
        alpha_points = alpha_points if alpha_points is not None \
            else np.float32(alpha)

        size_lines = size_lines if size_lines is not None else size
        size_points = size_points if size_points is not None else size * 3

        image = self.draw_lines_on_image(
            image, color=np.array(color_lines).astype(np.uint8),
            alpha=alpha_lines, size=size_lines,
            antialiased=antialiased,
            raise_if_out_of_image=raise_if_out_of_image)

        image = self.draw_points_on_image(
            image, color=np.array(color_points).astype(np.uint8),
            alpha=alpha_points, size=size_points,
            copy=False,
            raise_if_out_of_image=raise_if_out_of_image)

        return image

    def extract_from_image(self, image, size=1, pad=True, pad_max=None,
                           antialiased=True, prevent_zero_size=True):
        """
        Extract the image pixels covered by the line string.

        It will only extract pixels overlapped by the line string.

        This function will by default zero-pad the image if the line string is
        partially/fully outside of the image. This is for consistency with
        the same implementations for bounding boxes and polygons.

        Parameters
        ----------
        image : ndarray
            The image of shape `(H,W,[C])` from which to extract the pixels
            within the line string.

        size : int, optional
            Thickness of the line.

        pad : bool, optional
            Whether to zero-pad the image if the object is partially/fully
            outside of it.

        pad_max : None or int, optional
            The maximum number of pixels that may be zero-paded on any side,
            i.e. if this has value ``N`` the total maximum of added pixels
            is ``4*N``.
            This option exists to prevent extremely large images as a result of
            single points being moved very far away during augmentation.

        antialiased : bool, optional
            Whether to apply anti-aliasing to the line string.

        prevent_zero_size : bool, optional
            Whether to prevent height or width of the extracted image from
            becoming zero. If this is set to True and height or width of the
            line string is below 1, the height/width will be increased to 1.
            This can be useful to prevent problems, e.g. with image saving or
            plotting. If it is set to False, images will be returned as
            ``(H', W')`` or ``(H', W', 3)`` with ``H`` or ``W`` potentially
            being 0.

        Returns
        -------
        image : (H',W') ndarray or (H',W',C) ndarray
            Pixels overlapping with the line string. Zero-padded if the
            line string is partially/fully outside of the image and
            ``pad=True``. If `prevent_zero_size` is activated, it is
            guarantueed that ``H'>0`` and ``W'>0``, otherwise only
            ``H'>=0`` and ``W'>=0``.

        """
        from .bbs import BoundingBox

        assert image.ndim in [2, 3], (
            "Expected image of shape (H,W,[C]), "
            "got shape %s." % (image.shape,))

        if len(self.coords) == 0 or size <= 0:
            if prevent_zero_size:
                return np.zeros((1, 1) + image.shape[2:], dtype=image.dtype)
            return np.zeros((0, 0) + image.shape[2:], dtype=image.dtype)

        xx = self.xx_int
        yy = self.yy_int

        # this would probably work if drawing was subpixel-accurate
        # x1 = np.min(self.coords[:, 0]) - (size / 2)
        # y1 = np.min(self.coords[:, 1]) - (size / 2)
        # x2 = np.max(self.coords[:, 0]) + (size / 2)
        # y2 = np.max(self.coords[:, 1]) + (size / 2)

        # this works currently with non-subpixel-accurate drawing
        sizeh = (size - 1) / 2
        x1 = np.min(xx) - sizeh
        y1 = np.min(yy) - sizeh
        x2 = np.max(xx) + 1 + sizeh
        y2 = np.max(yy) + 1 + sizeh
        bb = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

        if len(self.coords) == 1:
            return bb.extract_from_image(image, pad=pad, pad_max=pad_max,
                                         prevent_zero_size=prevent_zero_size)

        heatmap = self.draw_lines_heatmap_array(
            image.shape[0:2], alpha=1.0, size=size, antialiased=antialiased)
        if image.ndim == 3:
            heatmap = np.atleast_3d(heatmap)
        image_masked = image.astype(np.float32) * heatmap
        extract = bb.extract_from_image(image_masked, pad=pad, pad_max=pad_max,
                                        prevent_zero_size=prevent_zero_size)
        return np.clip(np.round(extract), 0, 255).astype(np.uint8)

    def concatenate(self, other):
        """
        Concatenate this line string with another one.

        This will add a line segment between the end point of this line string
        and the start point of `other`.

        Parameters
        ----------
        other : imgaug.augmentables.lines.LineString or ndarray \
                or iterable of tuple of number
            The points to add to this line string.

        Returns
        -------
        imgaug.augmentables.lines.LineString
            New line string with concatenated points.
            The `label` of this line string will be kept.

        """
        if not isinstance(other, LineString):
            other = LineString(other)
        return self.deepcopy(
            coords=np.concatenate([self.coords, other.coords], axis=0))

    # TODO add tests
    def subdivide(self, points_per_edge):
        """
        Adds ``N`` interpolated points with uniform spacing to each edge.

        For each edge between points ``A`` and ``B`` this adds points
        at ``A + (i/(1+N)) * (B - A)``, where ``i`` is the index of the added
        point and ``N`` is the number of points to add per edge.

        Calling this method two times will split each edge at its center
        and then again split each newly created edge at their center.
        It is equivalent to calling `subdivide(3)`.

        Parameters
        ----------
        points_per_edge : int
            Number of points to interpolate on each edge.

        Returns
        -------
        LineString
            Line string with subdivided edges.

        """
        if len(self.coords) <= 1 or points_per_edge < 1:
            return self.deepcopy()
        coords = interpolate_points(self.coords, nb_steps=points_per_edge,
                                    closed=False)
        return self.deepcopy(coords=coords)

    def to_keypoints(self):
        """
        Convert the line string points to keypoints.

        Returns
        -------
        list of imgaug.augmentables.kps.Keypoint
            Points of the line string as keypoints.

        """
        # TODO get rid of this deferred import
        from imgaug.augmentables.kps import Keypoint
        return [Keypoint(x=x, y=y) for (x, y) in self.coords]

    def to_bounding_box(self):
        """
        Generate a bounding box encapsulating the line string.

        Returns
        -------
        None or imgaug.augmentables.bbs.BoundingBox
            Bounding box encapsulating the line string.
            ``None`` if the line string contained no points.

        """
        from .bbs import BoundingBox
        # we don't have to mind the case of len(.) == 1 here, because
        # zero-sized BBs are considered valid
        if len(self.coords) == 0:
            return None
        return BoundingBox(x1=np.min(self.xx), y1=np.min(self.yy),
                           x2=np.max(self.xx), y2=np.max(self.yy),
                           label=self.label)

    def to_polygon(self):
        """
        Generate a polygon from the line string points.

        Returns
        -------
        imgaug.augmentables.polys.Polygon
            Polygon with the same corner points as the line string.
            Note that the polygon might be invalid, e.g. contain less than 3
            points or have self-intersections.

        """
        from .polys import Polygon
        return Polygon(self.coords, label=self.label)

    def to_heatmap(self, image_shape, size_lines=1, size_points=0,
                   antialiased=True, raise_if_out_of_image=False):
        """
        Generate a heatmap object from the line string.

        This is similar to
        :func:`imgaug.augmentables.lines.LineString.draw_lines_heatmap_array`
        executed with ``alpha=1.0``. The result is wrapped in a
        ``HeatmapsOnImage`` object instead of just an array.
        No points are drawn.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        size_lines : int, optional
            Thickness of the line.

        size_points : int, optional
            Size of the points in pixels.

        antialiased : bool, optional
            Whether to draw the line with anti-aliasing activated.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapOnImage
            Heatmap object containing drawn line string.

        """
        from .heatmaps import HeatmapsOnImage
        return HeatmapsOnImage(
            self.draw_heatmap_array(
                image_shape, size_lines=size_lines, size_points=size_points,
                antialiased=antialiased,
                raise_if_out_of_image=raise_if_out_of_image),
            shape=image_shape
        )

    def to_segmentation_map(self, image_shape, size_lines=1, size_points=0,
                            raise_if_out_of_image=False):
        """
        Generate a segmentation map object from the line string.

        This is similar to
        :func:`imgaug.augmentables.lines.LineString.draw_mask`.
        The result is wrapped in a ``SegmentationMapOnImage`` object
        instead of just an array.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        size_lines : int, optional
            Thickness of the line.

        size_points : int, optional
            Size of the points in pixels.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the line string is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        imgaug.augmentables.segmaps.SegmentationMapOnImage
            Segmentation map object containing drawn line string.

        """
        from .segmaps import SegmentationMapOnImage
        return SegmentationMapOnImage(
            self.draw_mask(
                image_shape, size_lines=size_lines, size_points=size_points,
                raise_if_out_of_image=raise_if_out_of_image),
            shape=image_shape
        )

    # TODO make this non-approximate
    def coords_almost_equals(self, other, max_distance=1e-6, points_per_edge=8):
        """
        Compare this and another LineString's coordinates.

        This is an approximate method based on pointwise distances and can
        in rare corner cases produce wrong outputs.

        Parameters
        ----------
        other : imgaug.augmentables.lines.LineString \
                or tuple of number \
                or ndarray \
                or list of ndarray \
                or list of tuple of number
            The other line string or its coordinates.

        max_distance : float
            Max distance of any point from the other line string before
            the two line strings are evaluated to be unequal.

        points_per_edge : int, optional
            How many points to interpolate on each edge.

        Returns
        -------
        bool
            Whether the two LineString's coordinates are almost identical,
            i.e. the max distance is below the threshold.
            If both have no coordinates, ``True`` is returned.
            If only one has no coordinates, ``False`` is returned.
            Beyond that, the number of points is not evaluated.

        """
        if isinstance(other, LineString):
            pass
        elif isinstance(other, tuple):
            other = LineString([other])
        else:
            other = LineString(other)

        if len(self.coords) == 0 and len(other.coords) == 0:
            return True
        elif 0 in [len(self.coords), len(other.coords)]:
            # only one of the two line strings has no coords
            return False

        self_subd = self.subdivide(points_per_edge)
        other_subd = other.subdivide(points_per_edge)

        dist_self2other = self_subd.compute_pointwise_distances(other_subd)
        dist_other2self = other_subd.compute_pointwise_distances(self_subd)
        dist = max(np.max(dist_self2other), np.max(dist_other2self))
        return  dist < max_distance

    def almost_equals(self, other, max_distance=1e-4, points_per_edge=8):
        """
        Compare this and another LineString.

        Parameters
        ----------
        other: imgaug.augmentables.lines.LineString
            The other line string. Must be a LineString instance, not just
            its coordinates.

        max_distance : float, optional
            See :func:`imgaug.augmentables.lines.LineString.coords_almost_equals`.

        points_per_edge : int, optional
            See :func:`imgaug.augmentables.lines.LineString.coords_almost_equals`.

        Returns
        -------
        bool
            ``True`` if the coordinates are almost equal according to
            :func:`imgaug.augmentables.lines.LineString.coords_almost_equals`
            and additionally the labels are identical. Otherwise ``False``.

        """
        if self.label != other.label:
            return False
        return self.coords_almost_equals(
            other, max_distance=max_distance, points_per_edge=points_per_edge)

    def copy(self, coords=None, label=None):
        """
        Create a shallow copy of the LineString object.

        Parameters
        ----------
        coords : None or iterable of tuple of number or ndarray
            If not ``None``, then the coords of the copied object will be set
            to this value.

        label : None or str
            If not ``None``, then the label of the copied object will be set to
            this value.

        Returns
        -------
        imgaug.augmentables.lines.LineString
            Shallow copy.

        """
        return LineString(coords=self.coords if coords is None else coords,
                          label=self.label if label is None else label)

    def deepcopy(self, coords=None, label=None):
        """
        Create a deep copy of the BoundingBox object.

        Parameters
        ----------
        coords : None or iterable of tuple of number or ndarray
            If not ``None``, then the coords of the copied object will be set
            to this value.

        label : None or str
            If not ``None``, then the label of the copied object will be set to
            this value.

        Returns
        -------
        imgaug.augmentables.lines.LineString
            Deep copy.

        """
        return LineString(
            coords=np.copy(self.coords) if coords is None else coords,
            label=copylib.deepcopy(self.label) if label is None else label)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        points_str = ", ".join(
            ["(%.2f, %.2f)" % (x, y) for x, y in self.coords])
        return "LineString([%s], label=%s)" % (points_str, self.label)


# TODO
# distance
# hausdorff_distance
# is_fully_within_image()
# is_partly_within_image()
# is_out_of_image()
# draw()
# draw_mask()
# extract_from_image()
# to_keypoints()
# intersects(other)
# concat(other)
# is_self_intersecting()
# remove_self_intersections()
class LineStringsOnImage(object):
    """
    Object that represents all line strings on a single image.

    Parameters
    ----------
    line_strings : list of imgaug.augmentables.lines.LineString
        List of line strings on the image.

    shape : tuple of int or ndarray
        The shape of the image on which the objects are placed.
        Either an image with shape ``(H,W,[C])`` or a tuple denoting
        such an image shape.

    Examples
    --------
    >>> import imgaug.augmentables.lines as lines
    >>> image = np.zeros((100, 100))
    >>> lss = [
    >>>     lines.LineString([(0, 0), (10, 0)]),
    >>>     lines.LineString([(10, 20), (30, 30), (50, 70)])
    >>> ]
    >>> lsoi = lines.LineStringsOnImage(lss, shape=image.shape)

    """

    def __init__(self, line_strings, shape):
        assert ia.is_iterable(line_strings), (
            "Expected 'line_strings' to be an iterable, got type '%s'." % (
                type(line_strings),))
        assert all([isinstance(v, LineString) for v in line_strings]), (
            "Expected iterable of LineString, got types: %s." % (
                ", ".join([str(type(v)) for v in line_strings])
            ))
        self.line_strings = line_strings
        self.shape = normalize_shape(shape)

    @property
    def empty(self):
        """
        Returns whether this object contains zero line strings.

        Returns
        -------
        bool
            True if this object contains zero line strings.

        """
        return len(self.line_strings) == 0

    def on(self, image):
        """
        Project bounding boxes from one image to a new one.

        Parameters
        ----------
        image : ndarray or tuple of int
            The new image onto which to project.
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.

        Returns
        -------
        line_strings : imgaug.augmentables.lines.LineStrings
            Object containing all projected line strings.

        """
        shape = normalize_shape(image)
        if shape[0:2] == self.shape[0:2]:
            return self.deepcopy()
        line_strings = [ls.project(self.shape, shape)
                        for ls in self.line_strings]
        return self.deepcopy(line_strings=line_strings, shape=shape)

    @classmethod
    def from_xy_arrays(cls, xy, shape):
        """
        Convert an `(N,M,2)` ndarray to a LineStringsOnImage object.

        This is the inverse of
        :func:`imgaug.augmentables.lines.LineStringsOnImage.to_xy_array`.

        Parameters
        ----------
        xy : (N,M,2) ndarray or iterable of (M,2) ndarray
            Array containing the point coordinates ``N`` line strings
            with each ``M`` points given as ``(x,y)`` coordinates.
            ``M`` may differ if an iterable of arrays is used.
            Each array should usually be of dtype ``float32``.

        shape : tuple of int
            ``(H,W,[C])`` shape of the image on which the line strings are
            placed.

        Returns
        -------
        imgaug.augmentables.lines.LineStringsOnImage
            Object containing a list of ``LineString`` objects following the
            provided point coordinates.

        """
        lss = []
        for xy_ls in xy:
            lss.append(LineString(xy_ls))
        return cls(lss, shape)

    def to_xy_arrays(self, dtype=np.float32):
        """
        Convert this object to an iterable of ``(M,2)`` arrays of points.

        This is the inverse of
        :func:`imgaug.augmentables.lines.LineStringsOnImage.from_xy_array`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            Desired output datatype of the ndarray.

        Returns
        -------
        list of ndarray
            The arrays of point coordinates, each given as ``(M,2)``.

        """
        from .. import dtypes as iadt
        return [iadt.restore_dtypes_(np.copy(ls.coords), dtype)
                for ls in self.line_strings]

    def draw_on_image(self, image,
                      color=(0, 255, 0), color_lines=None, color_points=None,
                      alpha=1.0, alpha_lines=None, alpha_points=None,
                      size=1, size_lines=None, size_points=None,
                      antialiased=True,
                      raise_if_out_of_image=False):
        """
        Draw all line strings onto a given image.

        Parameters
        ----------
        image : ndarray
            The `(H,W,C)` `uint8` image onto which to draw the line strings.

        color : iterable of int, optional
            Color to use as RGB, i.e. three values.
            The color of the lines and points are derived from this value,
            unless they are set.

        color_lines : None or iterable of int
            Color to use for the line segments as RGB, i.e. three values.
            If ``None``, this value is derived from `color`.

        color_points : None or iterable of int
            Color to use for the points as RGB, i.e. three values.
            If ``None``, this value is derived from ``0.5 * color``.

        alpha : float, optional
            Opacity of the line strings. Higher values denote more visible
            points.
            The alphas of the line and points are derived from this value,
            unless they are set.

        alpha_lines : None or float, optional
            Opacity of the line strings. Higher values denote more visible
            line string.
            If ``None``, this value is derived from `alpha`.

        alpha_points : None or float, optional
            Opacity of the line string points. Higher values denote more
            visible points.
            If ``None``, this value is derived from `alpha`.

        size : int, optional
            Size of the line strings.
            The sizes of the line and points are derived from this value,
            unless they are set.

        size_lines : None or int, optional
            Thickness of the line segments.
            If ``None``, this value is derived from `size`.

        size_points : None or int, optional
            Size of the points in pixels.
            If ``None``, this value is derived from ``3 * size``.

        antialiased : bool, optional
            Whether to draw the lines with anti-aliasing activated.
            This does currently not affect the point drawing.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if a line string is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        ndarray
            Image with line strings drawn on it.

        """
        # TODO improve efficiency here by copying only once
        for ls in self.line_strings:
            image = ls.draw_on_image(
                image,
                color=color, color_lines=color_lines, color_points=color_points,
                alpha=alpha, alpha_lines=alpha_lines, alpha_points=alpha_points,
                size=size, size_lines=size_lines, size_points=size_points,
                antialiased=antialiased,
                raise_if_out_of_image=raise_if_out_of_image
            )

        return image

    def remove_out_of_image(self, fully=True, partly=False):
        """
        Remove all line strings that are fully/partially outside of the image.

        Parameters
        ----------
        fully : bool, optional
            Whether to remove line strings that are fully outside of the image.

        partly : bool, optional
            Whether to remove line strings that are partially outside of the
            image.

        Returns
        -------
        imgaug.augmentables.lines.LineStringsOnImage
            Reduced set of line strings, with those that were fully/partially
            outside of the image removed.

        """
        lss_clean = [ls for ls in self.line_strings
                     if not ls.is_out_of_image(
                         self.shape, fully=fully, partly=partly)]
        return LineStringsOnImage(lss_clean, shape=self.shape)

    def clip_out_of_image(self):
        """
        Clip off all parts of the line strings that are outside of the image.

        Returns
        -------
        imgaug.augmentables.lines.LineStringsOnImage
            Line strings, clipped to fall within the image dimensions.

        """
        lss_cut = [ls_clipped
                   for ls in self.line_strings
                   for ls_clipped in ls.clip_out_of_image(self.shape)]
        return LineStringsOnImage(lss_cut, shape=self.shape)

    def shift(self, top=None, right=None, bottom=None, left=None):
        """
        Shift/move the line strings from one or more image sides.

        Parameters
        ----------
        top : None or int, optional
            Amount of pixels by which to shift all bounding boxes from the
            top.

        right : None or int, optional
            Amount of pixels by which to shift all bounding boxes from the
            right.

        bottom : None or int, optional
            Amount of pixels by which to shift all bounding boxes from the
            bottom.

        left : None or int, optional
            Amount of pixels by which to shift all bounding boxes from the
            left.

        Returns
        -------
        imgaug.augmentables.lines.LineStringsOnImage
            Shifted line strings.

        """
        lss_new = [ls.shift(top=top, right=right, bottom=bottom, left=left)
                   for ls in self.line_strings]
        return LineStringsOnImage(lss_new, shape=self.shape)

    def copy(self, line_strings=None, shape=None):
        """
        Create a shallow copy of the LineStringsOnImage object.

        Parameters
        ----------
        line_strings : None \
                       or list of imgaug.augmentables.lines.LineString, optional
            List of line strings on the image.
            If not ``None``, then the ``line_strings`` attribute of the copied
            object will be set to this value.

        shape : None or tuple of int or ndarray, optional
            The shape of the image on which the objects are placed.
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.
            If not ``None``, then the ``shape`` attribute of the copied object
            will be set to this value.

        Returns
        -------
        imgaug.augmentables.lines.LineStringsOnImage
            Shallow copy.

        """
        lss = self.line_strings if line_strings is None else line_strings
        shape = self.shape if shape is None else shape
        return LineStringsOnImage(line_strings=lss, shape=shape)

    def deepcopy(self, line_strings=None, shape=None):
        """
        Create a deep copy of the LineStringsOnImage object.

        Parameters
        ----------
        line_strings : None \
                       or list of imgaug.augmentables.lines.LineString, optional
            List of line strings on the image.
            If not ``None``, then the ``line_strings`` attribute of the copied
            object will be set to this value.

        shape : None or tuple of int or ndarray, optional
            The shape of the image on which the objects are placed.
            Either an image with shape ``(H,W,[C])`` or a tuple denoting
            such an image shape.
            If not ``None``, then the ``shape`` attribute of the copied object
            will be set to this value.

        Returns
        -------
        imgaug.augmentables.lines.LineStringsOnImage
            Deep copy.

        """
        lss = self.line_strings if line_strings is None else line_strings
        shape = self.shape if shape is None else shape
        return LineStringsOnImage(
            line_strings=[ls.deepcopy() for ls in lss],
            shape=tuple(shape))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "LineStringsOnImage(%s, shape=%s)" % (
            str(self.line_strings), self.shape)


def _is_point_on_line(line_start, line_end, point, eps=1e-4):
    dist_s2e = np.linalg.norm(np.float32(line_start) - np.float32(line_end))
    dist_s2p2e = (
        np.linalg.norm(np.float32(line_start) - np.float32(point))
        + np.linalg.norm(np.float32(point) - np.float32(line_end))
    )
    return -eps < (dist_s2p2e - dist_s2e) < eps


def _flatten_shapely_collection(collection):
    import shapely.geometry
    if not isinstance(collection, list):
        collection = [collection]
    for el in collection:
        if hasattr(el, "geoms"):
            for subel in _flatten_shapely_collection(el.geoms):
                # MultiPoint.geoms actually returns a GeometrySequence
                if isinstance(subel, shapely.geometry.base.GeometrySequence):
                    for subsubel in subel:
                        yield subsubel
                else:
                    yield _flatten_shapely_collection(subel)
        else:
            yield el


def _convert_var_to_shapely_geometry(var):
    import shapely.geometry
    if isinstance(var, tuple):
        geom = shapely.geometry.Point(var[0], var[1])
    elif isinstance(var, list):
        assert len(var) > 0
        if isinstance(var[0], tuple):
            geom = shapely.geometry.LineString(var)
        elif all([isinstance(v, LineString) for v in var]):
            geom = shapely.geometry.MultiLineString([
                shapely.geometry.LineString(ls.coords) for ls in var
            ])
        else:
            raise ValueError(
                "Could not convert list-input to shapely geometry. Invalid "
                "datatype. List elements had datatypes: %s." % (
                    ", ".join([str(type(v)) for v in var]),))
    elif isinstance(var, LineString):
        geom = shapely.geometry.LineString(var.coords)
    else:
        raise ValueError(
            "Could not convert input to shapely geometry. Invalid datatype. "
            "Got: %s" % (type(var),))
    return geom
