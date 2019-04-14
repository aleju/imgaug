from __future__ import print_function, division, absolute_import

import copy
import warnings

import numpy as np
import scipy.spatial.distance
import six.moves as sm
import skimage.draw
import skimage.measure
import collections

from .. import imgaug as ia
from .utils import normalize_shape, interpolate_points


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
        # TODO get rid of this deferred import
        from imgaug.augmentables.kps import Keypoint

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
            ia.do_assert(ia.is_np_array(exterior),
                         ("Expected exterior to be a list of tuples (x, y) or "
                          + "an (N, 2) array, got type %s") % (exterior,))
            ia.do_assert(exterior.ndim == 2 and exterior.shape[1] == 2,
                         ("Expected exterior to be a list of tuples (x, y) or "
                          + "an (N, 2) array, got an array of shape %s") % (
                             exterior.shape,))
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

    @property
    def height(self):
        """
        Estimate the height of the polygon.

        Returns
        -------
        number
            Height of the polygon.

        """
        yy = self.yy
        return max(yy) - min(yy)

    @property
    def width(self):
        """
        Estimate the width of the polygon.

        Returns
        -------
        number
            Width of the polygon.

        """
        xx = self.xx
        return max(xx) - min(xx)

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
        ls_proj = self.to_line_string(closed=False).project(
            from_shape, to_shape)
        return self.copy(exterior=ls_proj.coords)

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
        ia.do_assert(len(self.exterior) > 0)
        distances = []
        for x2, y2 in self.exterior:
            d = (x2 - x) ** 2 + (y2 - y) ** 2
            distances.append(d)
        distances = np.sqrt(distances)
        closest_idx = np.argmin(distances)
        if return_distance:
            return closest_idx, distances[closest_idx]
        return closest_idx

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
            Whether to return True if the polygon is fully outside of the image area.

        partly : bool, optional
            Whether to return True if the polygon is at least partially outside fo the image area.

        Returns
        -------
        bool
            True if the polygon is partially/fully outside of the image area, depending
            on defined parameters. False otherwise.

        """
        # TODO this is inconsistent with line strings, which return a default
        #      value in these cases
        if len(self.exterior) == 0:
            raise Exception("Cannot determine whether the polygon is inside the image, because it contains no points.")
        ls = self.to_line_string()
        return ls.is_out_of_image(image, fully=fully, partly=partly)

    @ia.deprecated(alt_func="Polygon.clip_out_of_image()",
                   comment="clip_out_of_image() has the exactly same "
                           "interface.")
    def cut_out_of_image(self, image):
        return self.clip_out_of_image(image)

    # TODO this currently can mess up the order of points - change somehow to
    #      keep the order
    def clip_out_of_image(self, image):
        """
        Cut off all parts of the polygon that are outside of the image.

        This operation may lead to new points being created.
        As a single polygon may be split into multiple new polygons, the result
        is always a list, which may contain more than one output polygon.

        This operation will return an empty list if the polygon is completely
        outside of the image plane.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use for the clipping of the polygon.
            If an ndarray, its shape will be used.
            If a tuple, it is assumed to represent the image shape and must
            contain at least two integers.

        Returns
        -------
        list of imgaug.Polygon
            Polygon, clipped to fall within the image dimensions.
            Returned as a list, because the clipping can split the polygon into
            multiple parts. The list may also be empty, if the polygon was
            fully outside of the image plane.

        """
        # load shapely lazily, which makes the dependency more optional
        import shapely.geometry

        # if fully out of image, clip everything away, nothing remaining
        if self.is_out_of_image(image, fully=True, partly=False):
            return []

        h, w = image.shape[0:2] if ia.is_np_array(image) else image[0:2]
        poly_shapely = self.to_shapely_polygon()
        poly_image = shapely.geometry.Polygon([(0, 0), (w, 0), (w, h), (0, h)])
        multipoly_inter_shapely = poly_shapely.intersection(poly_image)
        if not isinstance(multipoly_inter_shapely, shapely.geometry.MultiPolygon):
            ia.do_assert(isinstance(multipoly_inter_shapely, shapely.geometry.Polygon))
            multipoly_inter_shapely = shapely.geometry.MultiPolygon([multipoly_inter_shapely])

        polygons = []
        for poly_inter_shapely in multipoly_inter_shapely.geoms:
            polygons.append(Polygon.from_shapely(poly_inter_shapely, label=self.label))

        # shapely changes the order of points, we try here to preserve it as
        # much as possible
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
            ia.do_assert(found)  # could only not find closest points if new polys are empty

        return polygons_reordered

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
        ls_shifted = self.to_line_string(closed=False).shift(
            top=top, right=right, bottom=bottom, left=left)
        return self.copy(exterior=ls_shifted.coords)

    # TODO separate this into draw_face_on_image() and draw_border_on_image()
    # TODO add tests for line thickness
    def draw_on_image(self,
                      image,
                      color=(0, 255, 0), color_face=None,
                      color_lines=None, color_points=None,
                      alpha=1.0, alpha_face=None,
                      alpha_lines=None, alpha_points=None,
                      size=1, size_lines=None, size_points=None,
                      raise_if_out_of_image=False):
        """
        Draw the polygon on an image.

        Parameters
        ----------
        image : (H,W,C) ndarray
            The image onto which to draw the polygon. Usually expected to be
            of dtype ``uint8``, though other dtypes are also handled.

        color : iterable of int, optional
            The color to use for the whole polygon.
            Must correspond to the channel layout of the image. Usually RGB.
            The values for `color_face`, `color_lines` and `color_points`
            will be derived from this color if they are set to ``None``.
            This argument has no effect if `color_face`, `color_lines`
            and `color_points` are all set anything other than ``None``.

        color_face : None or iterable of int, optional
            The color to use for the inner polygon area (excluding perimeter).
            Must correspond to the channel layout of the image. Usually RGB.
            If this is ``None``, it will be derived from ``color * 1.0``.

        color_lines : None or iterable of int, optional
            The color to use for the line (aka perimeter/border) of the polygon.
            Must correspond to the channel layout of the image. Usually RGB.
            If this is ``None``, it will be derived from ``color * 0.5``.

        color_points : None or iterable of int, optional
            The color to use for the corner points of the polygon.
            Must correspond to the channel layout of the image. Usually RGB.
            If this is ``None``, it will be derived from ``color * 0.5``.

        alpha : float, optional
            The opacity of the whole polygon, where ``1.0`` denotes a completely
            visible polygon and ``0.0`` an invisible one.
            The values for `alpha_face`, `alpha_lines` and `alpha_points`
            will be derived from this alpha value if they are set to ``None``.
            This argument has no effect if `alpha_face`, `alpha_lines`
            and `alpha_points` are all set anything other than ``None``.

        alpha_face : None or number, optional
            The opacity of the polygon's inner area (excluding the perimeter),
            where ``1.0`` denotes a completely visible inner area and ``0.0``
            an invisible one.
            If this is ``None``, it will be derived from ``alpha * 0.5``.

        alpha_lines : None or number, optional
            The opacity of the polygon's line (aka perimeter/border),
            where ``1.0`` denotes a completely visible line and ``0.0`` an
            invisible one.
            If this is ``None``, it will be derived from ``alpha * 1.0``.

        alpha_points : None or number, optional
            The opacity of the polygon's corner points, where ``1.0`` denotes
            completely visible corners and ``0.0`` invisible ones.
            If this is ``None``, it will be derived from ``alpha * 1.0``.

        size : int, optional
            Size of the polygon.
            The sizes of the line and points are derived from this value,
            unless they are set.

        size_lines : None or int, optional
            Thickness of the polygon's line (aka perimeter/border).
            If ``None``, this value is derived from `size`.

        size_points : int, optional
            Size of the points in pixels.
            If ``None``, this value is derived from ``3 * size``.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the polygon is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        result : (H,W,C) ndarray
            Image with polygon drawn on it. Result dtype is the same as the input dtype.

        """
        assert color is not None
        assert alpha is not None
        assert size is not None

        color_face = color_face if color_face is not None else np.array(color)
        color_lines = color_lines if color_lines is not None else np.array(color) * 0.5
        color_points = color_points if color_points is not None else np.array(color) * 0.5

        alpha_face = alpha_face if alpha_face is not None else alpha * 0.5
        alpha_lines = alpha_lines if alpha_lines is not None else alpha
        alpha_points = alpha_points if alpha_points is not None else alpha

        size_lines = size_lines if size_lines is not None else size
        size_points = size_points if size_points is not None else size * 3

        if image.ndim == 2:
            assert ia.is_single_number(color_face), (
                "Got a 2D image. Expected then 'color_face' to be a single "
                "number, but got %s." % (str(color_face),))
            color_face = [color_face]
        elif image.ndim == 3 and ia.is_single_number(color_face):
            color_face = [color_face] * image.shape[-1]

        if alpha_face < 0.01:
            alpha_face = 0
        elif alpha_face > 0.99:
            alpha_face = 1

        if raise_if_out_of_image and self.is_out_of_image(image):
            raise Exception("Cannot draw polygon %s on image with shape %s." % (
                str(self), image.shape
            ))

        # TODO np.clip to image plane if is_fully_within_image(), similar to how it is done for bounding boxes

        # TODO improve efficiency by only drawing in rectangle that covers poly instead of drawing in the whole image
        # TODO for a rectangular polygon, the face coordinates include the top/left boundary but not the right/bottom
        # boundary. This may be unintuitive when not drawing the boundary. Maybe somehow remove the boundary
        # coordinates from the face coordinates after generating both?
        input_dtype = image.dtype
        result = image.astype(np.float32)
        rr, cc = skimage.draw.polygon(self.yy_int, self.xx_int, shape=image.shape)
        if len(rr) > 0:
            if alpha_face == 1:
                result[rr, cc] = np.float32(color_face)
            elif alpha_face == 0:
                pass
            else:
                result[rr, cc] = (
                        (1 - alpha_face) * result[rr, cc, :]
                        + alpha_face * np.float32(color_face)
                )

        ls_open = self.to_line_string(closed=False)
        ls_closed = self.to_line_string(closed=True)
        result = ls_closed.draw_lines_on_image(
            result, color=color_lines, alpha=alpha_lines,
            size=size_lines, raise_if_out_of_image=raise_if_out_of_image)
        result = ls_open.draw_points_on_image(
            result, color=color_points, alpha=alpha_points,
            size=size_points, raise_if_out_of_image=raise_if_out_of_image)

        if input_dtype.type == np.uint8:
            result = np.clip(np.round(result), 0, 255).astype(input_dtype)  # TODO make clipping more flexible
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
        ia.do_assert(image.ndim in [2, 3])
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

    def change_first_point_by_coords(self, x, y, max_distance=1e-4,
                                     raise_if_too_far_away=True):
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

        max_distance : None or number, optional
            Maximum distance past which possible matches are ignored.
            If ``None`` the distance limit is deactivated.

        raise_if_too_far_away : bool, optional
            Whether to raise an exception if the closest found point is too
            far away (``True``) or simply return an unchanged copy if this
            object (``False``).

        Returns
        -------
        imgaug.Polygon
            Copy of this polygon with the new point order.

        """
        if len(self.exterior) == 0:
            raise Exception("Cannot reorder polygon points, because it contains no points.")

        closest_idx, closest_dist = self.find_closest_point_index(x=x, y=y, return_distance=True)
        if max_distance is not None and closest_dist > max_distance:
            if not raise_if_too_far_away:
                return self.deepcopy()

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
        ia.do_assert(0 <= point_idx < len(self.exterior))
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
            Tight bounding box around the polygon.

        """
        # TODO get rid of this deferred import
        from imgaug.augmentables.bbs import BoundingBox

        xx = self.xx
        yy = self.yy
        return BoundingBox(x1=min(xx), x2=max(xx), y1=min(yy), y2=max(yy), label=self.label)

    def to_keypoints(self):
        """
        Convert this polygon's `exterior` to ``Keypoint`` instances.

        Returns
        -------
        list of imgaug.Keypoint
            Exterior vertices as ``Keypoint`` instances.

        """
        # TODO get rid of this deferred import
        from imgaug.augmentables.kps import Keypoint

        return [Keypoint(x=point[0], y=point[1]) for point in self.exterior]

    def to_line_string(self, closed=True):
        """
        Convert this polygon's `exterior` to a ``LineString`` instance.

        Parameters
        ----------
        closed : bool, optional
            Whether to close the line string, i.e. to add the first point of
            the `exterior` also as the last point at the end of the line string.
            This has no effect if the polygon has a single point or zero
            points.

        Returns
        -------
        imgaug.augmentables.lines.LineString
            Exterior of the polygon as a line string.

        """
        from imgaug.augmentables.lines import LineString
        if not closed or len(self.exterior) <= 1:
            return LineString(self.exterior, label=self.label)
        return LineString(
            np.concatenate([self.exterior, self.exterior[0:1, :]], axis=0),
            label=self.label)

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

        ia.do_assert(isinstance(polygon_shapely, shapely.geometry.Polygon))
        # polygon_shapely.exterior can be None if the polygon was instantiated without points
        if polygon_shapely.exterior is None or len(polygon_shapely.exterior.coords) == 0:
            return Polygon([], label=label)
        exterior = np.float32([[x, y] for (x, y) in polygon_shapely.exterior.coords])
        return Polygon(exterior, label=label)

    def exterior_almost_equals(self, other, max_distance=1e-6, points_per_edge=8):
        """
        Estimate if this and other polygon's exterior are almost identical.

        The two exteriors can have different numbers of points, but any point
        randomly sampled on the exterior of one polygon should be close to the
        closest point on the exterior of the other polygon.

        Note that this method works approximately. One can come up with
        polygons with fairly different shapes that will still be estimated as
        equal by this method. In practice however this should be unlikely to be
        the case. The probability for something like that goes down as the
        interpolation parameter is increased.

        Parameters
        ----------
        other : imgaug.Polygon or (N,2) ndarray or list of tuple
            The other polygon with which to compare the exterior.
            If this is an ndarray, it is assumed to represent an exterior.
            It must then have dtype ``float32`` and shape ``(N,2)`` with the
            second dimension denoting xy-coordinates.
            If this is a list of tuples, it is assumed to represent an exterior.
            Each tuple then must contain exactly two numbers, denoting
            xy-coordinates.

        max_distance : number, optional
            The maximum euclidean distance between a point on one polygon and
            the closest point on the other polygon. If the distance is exceeded
            for any such pair, the two exteriors are not viewed as equal. The
            points are other the points contained in the polygon's exterior
            ndarray or interpolated points between these.

        points_per_edge : int, optional
            How many points to interpolate on each edge.

        Returns
        -------
        bool
            Whether the two polygon's exteriors can be viewed as equal
            (approximate test).

        """
        if isinstance(other, list):
            other = Polygon(np.float32(other))
        elif ia.is_np_array(other):
            other = Polygon(other)
        else:
            assert isinstance(other, Polygon)
            other = other

        return self.to_line_string(closed=True).coords_almost_equals(
            other.to_line_string(closed=True),
            max_distance=max_distance,
            points_per_edge=points_per_edge
        )

    def almost_equals(self, other, max_distance=1e-6, points_per_edge=8):
        """
        Estimate if this polygon's and another's geometry/labels are similar.

        This is the same as :func:`imgaug.Polygon.exterior_almost_equals` but
        additionally compares the labels.

        Parameters
        ----------
        other
            The object to compare against. If not a Polygon, then False will
            be returned.

        max_distance : float, optional
            See :func:`imgaug.augmentables.polys.Polygon.exterior_almost_equals`.

        points_per_edge : int, optional
            See :func:`imgaug.augmentables.polys.Polygon.exterior_almost_equals`.

        Returns
        -------
        bool
            Whether the two polygons can be viewed as equal. In the case of
            the exteriors this is an approximate test.

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
        return self.exterior_almost_equals(
            other, max_distance=max_distance, points_per_edge=points_per_edge)

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


# TODO add tests for this
class PolygonsOnImage(object):
    """
    Object that represents all polygons on a single image.

    Parameters
    ----------
    polygons : list of imgaug.Polygon
        List of polygons on the image.

    shape : tuple of int
        The shape of the image on which the polygons are placed.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug as ia
    >>> image = np.zeros((100, 100))
    >>> polys = [
    >>>     ia.Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
    >>>     ia.Polygon([(50, 0), (100, 50), (50, 100), (0, 50)])
    >>> ]
    >>> polys_oi = ia.PolygonsOnImage(polys, shape=image.shape)

    """

    def __init__(self, polygons, shape):
        self.polygons = polygons
        self.shape = normalize_shape(shape)

    @property
    def empty(self):
        """
        Returns whether this object contains zero polygons.

        Returns
        -------
        bool
            True if this object contains zero polygons.

        """
        return len(self.polygons) == 0

    def on(self, image):
        """
        Project polygons from one image to a new one.

        Parameters
        ----------
        image : ndarray or tuple of int
            New image onto which the polygons are to be projected.
            May also simply be that new image's shape tuple.

        Returns
        -------
        imgaug.PolygonsOnImage
            Object containing all projected polygons.

        """
        shape = normalize_shape(image)
        if shape[0:2] == self.shape[0:2]:
            return self.deepcopy()
        polygons = [poly.project(self.shape, shape) for poly in self.polygons]
        # TODO use deepcopy() here
        return PolygonsOnImage(polygons, shape)

    def draw_on_image(self,
                      image,
                      color=(0, 255, 0), color_face=None,
                      color_lines=None, color_points=None,
                      alpha=1.0, alpha_face=None,
                      alpha_lines=None, alpha_points=None,
                      size=1, size_lines=None, size_points=None,
                      raise_if_out_of_image=False):
        """
        Draw all polygons onto a given image.

        Parameters
        ----------
        image : (H,W,C) ndarray
            The image onto which to draw the bounding boxes.
            This image should usually have the same shape as set in
            ``PolygonsOnImage.shape``.

        color : iterable of int, optional
            The color to use for the whole polygons.
            Must correspond to the channel layout of the image. Usually RGB.
            The values for `color_face`, `color_lines` and `color_points`
            will be derived from this color if they are set to ``None``.
            This argument has no effect if `color_face`, `color_lines`
            and `color_points` are all set anything other than ``None``.

        color_face : None or iterable of int, optional
            The color to use for the inner polygon areas (excluding perimeters).
            Must correspond to the channel layout of the image. Usually RGB.
            If this is ``None``, it will be derived from ``color * 1.0``.

        color_lines : None or iterable of int, optional
            The color to use for the lines (aka perimeters/borders) of the
            polygons. Must correspond to the channel layout of the image.
            Usually RGB. If this is ``None``, it will be derived
            from ``color * 0.5``.

        color_points : None or iterable of int, optional
            The color to use for the corner points of the polygons.
            Must correspond to the channel layout of the image. Usually RGB.
            If this is ``None``, it will be derived from ``color * 0.5``.

        alpha : float, optional
            The opacity of the whole polygons, where ``1.0`` denotes
            completely visible polygons and ``0.0`` invisible ones.
            The values for `alpha_face`, `alpha_lines` and `alpha_points`
            will be derived from this alpha value if they are set to ``None``.
            This argument has no effect if `alpha_face`, `alpha_lines`
            and `alpha_points` are all set anything other than ``None``.

        alpha_face : None or number, optional
            The opacity of the polygon's inner areas (excluding the perimeters),
            where ``1.0`` denotes completely visible inner areas and ``0.0``
            invisible ones.
            If this is ``None``, it will be derived from ``alpha * 0.5``.

        alpha_lines : None or number, optional
            The opacity of the polygon's lines (aka perimeters/borders),
            where ``1.0`` denotes completely visible perimeters and ``0.0``
            invisible ones.
            If this is ``None``, it will be derived from ``alpha * 1.0``.

        alpha_points : None or number, optional
            The opacity of the polygon's corner points, where ``1.0`` denotes
            completely visible corners and ``0.0`` invisible ones.
            Currently this is an on/off choice, i.e. only ``0.0`` or ``1.0``
            are allowed.
            If this is ``None``, it will be derived from ``alpha * 1.0``.

        size : int, optional
            Size of the polygons.
            The sizes of the line and points are derived from this value,
            unless they are set.

        size_lines : None or int, optional
            Thickness of the polygon lines (aka perimeter/border).
            If ``None``, this value is derived from `size`.

        size_points : int, optional
            The size of all corner points. If set to ``C``, each corner point
            will be drawn as a square of size ``C x C``.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if any polygon is fully
            outside of the image. If set to False, no error will be raised and
            only the parts inside the image will be drawn.

        Returns
        -------
        image : (H,W,C) ndarray
            Image with drawn polygons.

        """
        for poly in self.polygons:
            image = poly.draw_on_image(
                image,
                color=color,
                color_face=color_face,
                color_lines=color_lines,
                color_points=color_points,
                alpha=alpha,
                alpha_face=alpha_face,
                alpha_lines=alpha_lines,
                alpha_points=alpha_points,
                size=size,
                size_lines=size_lines,
                size_points=size_points,
                raise_if_out_of_image=raise_if_out_of_image
            )
        return image

    def remove_out_of_image(self, fully=True, partly=False):
        """
        Remove all polygons that are fully or partially outside of the image.

        Parameters
        ----------
        fully : bool, optional
            Whether to remove polygons that are fully outside of the image.

        partly : bool, optional
            Whether to remove polygons that are partially outside of the image.

        Returns
        -------
        imgaug.PolygonsOnImage
            Reduced set of polygons, with those that were fully/partially
            outside of the image removed.

        """
        polys_clean = [
            poly for poly in self.polygons
            if not poly.is_out_of_image(self.shape, fully=fully, partly=partly)
        ]
        # TODO use deepcopy() here
        return PolygonsOnImage(polys_clean, shape=self.shape)

    def clip_out_of_image(self):
        """
        Clip off all parts from all polygons that are outside of the image.

        NOTE: The result can contain less polygons than the input did. That
        happens when a polygon is fully outside of the image plane.

        NOTE: The result can also contain *more* polygons than the input
        did. That happens when distinct parts of a polygon are only
        connected by areas that are outside of the image plane and hence will
        be clipped off, resulting in two or more unconnected polygon parts that
        are left in the image plane.

        Returns
        -------
        imgaug.PolygonsOnImage
            Polygons, clipped to fall within the image dimensions. Count of
            output polygons may differ from the input count.

        """
        polys_cut = [
            poly.clip_out_of_image(self.shape)
            for poly
            in self.polygons
            if poly.is_partly_within_image(self.shape)
        ]
        polys_cut_flat = [poly for poly_lst in polys_cut for poly in poly_lst]
        # TODO use deepcopy() here
        return PolygonsOnImage(polys_cut_flat, shape=self.shape)

    def shift(self, top=None, right=None, bottom=None, left=None):
        """
        Shift all polygons from one or more image sides, i.e. move them on the x/y-axis.

        Parameters
        ----------
        top : None or int, optional
            Amount of pixels by which to shift all polygons from the top.

        right : None or int, optional
            Amount of pixels by which to shift all polygons from the right.

        bottom : None or int, optional
            Amount of pixels by which to shift all polygons from the bottom.

        left : None or int, optional
            Amount of pixels by which to shift all polygons from the left.

        Returns
        -------
        imgaug.PolygonsOnImage
            Shifted polygons.

        """
        polys_new = [
            poly.shift(top=top, right=right, bottom=bottom, left=left)
            for poly
            in self.polygons
        ]
        return PolygonsOnImage(polys_new, shape=self.shape)

    def copy(self):
        """
        Create a shallow copy of the PolygonsOnImage object.

        Returns
        -------
        imgaug.PolygonsOnImage
            Shallow copy.

        """
        return copy.copy(self)

    def deepcopy(self):
        """
        Create a deep copy of the PolygonsOnImage object.

        Returns
        -------
        imgaug.PolygonsOnImage
            Deep copy.

        """
        # Manual copy is far faster than deepcopy for PolygonsOnImage,
        # so use manual copy here too
        polys = [poly.deepcopy() for poly in self.polygons]
        return PolygonsOnImage(polys, tuple(self.shape))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "PolygonsOnImage(%s, shape=%s)" % (str(self.polygons), self.shape)


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
        points_tuples = interpolate_points(points_tuples, interpolate)

    # close if requested and not yet closed
    if closed and len(points) > 1:  # here intentionally used points instead of points_tuples
        points_tuples.append(points_tuples[0])

    return shapely.geometry.LineString(points_tuples)


class _ConcavePolygonRecoverer(object):
    def __init__(self, threshold_duplicate_points=1e-4, noise_strength=1e-4,
                 oversampling=0.01, max_segment_difference=1e-4):
        self.threshold_duplicate_points = threshold_duplicate_points
        self.noise_strength = noise_strength
        self.oversampling = oversampling
        self.max_segment_difference = max_segment_difference

        # this limits the maximum amount of points after oversampling, i.e.
        # if N points are input into oversampling, then M oversampled points are
        # generated such that N+M <= this value
        self.oversample_up_to_n_points_max = 75

        # ----
        # parameters for _fit_best_valid_polygon()
        # ----
        # how many changes may be done max to the initial (convex hull) polygon
        # before simply returning the result
        self.fit_n_changes_max = 100
        # for how many iterations the optimization loop may run max
        # before simply returning the result
        self.fit_n_iters_max = 3
        # how far (wrt. to their position in the input list) two points may be
        # apart max to consider adding an edge between them (in the first loop
        # iteration and the ones after that)
        self.fit_max_dist_first_iter = 1
        self.fit_max_dist_other_iters = 2
        # The fit loop first generates candidate edges and then modifies the
        # polygon based on these candidates. This limits the maximum amount
        # of considered candidates. If the number is less than the possible
        # number of candidates, they are randomly subsampled. Values beyond
        # 100 significantly increase runtime (for polygons that reach that
        # number).
        self.fit_n_candidates_before_sort_max = 100

    def recover_from(self, new_exterior, old_polygon, random_state=0):
        assert isinstance(new_exterior, list) or (
                ia.is_np_array(new_exterior)
                and new_exterior.ndim == 2
                and new_exterior.shape[1] == 2)
        assert len(new_exterior) >= 3, \
            "Cannot recover a concave polygon from less than three points."

        # create Polygon instance, if it is already valid then just return
        # immediately
        polygon = old_polygon.deepcopy(exterior=new_exterior)
        if polygon.is_valid:
            return polygon

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        rss = ia.derive_random_states(random_state, 3)

        # remove consecutive duplicate points
        new_exterior = self._remove_consecutive_duplicate_points(new_exterior)

        # check that points are not all identical or on a line
        new_exterior = self._fix_polygon_is_line(new_exterior, rss[0])

        # jitter duplicate points
        new_exterior = self._jitter_duplicate_points(new_exterior, rss[1])

        # generate intersection points
        segment_add_points = self._generate_intersection_points(new_exterior)

        # oversample points around intersections
        if self.oversampling is not None and self.oversampling > 0:
            segment_add_points = self._oversample_intersection_points(
                new_exterior, segment_add_points)

        # integrate new points into exterior
        new_exterior_inter = self._insert_intersection_points(
            new_exterior, segment_add_points)

        # find best fit polygon, starting from convext polygon
        new_exterior_concave_ids = self._fit_best_valid_polygon(new_exterior_inter, rss[2])
        new_exterior_concave = [new_exterior_inter[idx] for idx in new_exterior_concave_ids]

        # TODO return new_exterior_concave here instead of polygon, leave it to
        #      caller to decide what to do with it
        return old_polygon.deepcopy(exterior=new_exterior_concave)

    def _remove_consecutive_duplicate_points(self, points):
        result = []
        for point in points:
            if result:
                dist = np.linalg.norm(np.float32(point) - np.float32(result[-1]))
                is_same = (dist < self.threshold_duplicate_points)
                if not is_same:
                    result.append(point)
            else:
                result.append(point)
        if len(result) >= 2:
            dist = np.linalg.norm(np.float32(result[0]) - np.float32(result[-1]))
            is_same = (dist < self.threshold_duplicate_points)
            result = result[0:-1] if is_same else result
        return result

    # fix polygons for which all points are on a line
    def _fix_polygon_is_line(self, exterior, random_state):
        assert len(exterior) >= 3
        noise_strength = self.noise_strength
        while self._is_polygon_line(exterior):
            noise = random_state.uniform(
                -noise_strength, noise_strength, size=(len(exterior), 2)
            ).astype(np.float32)
            exterior = [(point[0] + noise_i[0], point[1] + noise_i[1])
                        for point, noise_i in zip(exterior, noise)]
            noise_strength = noise_strength * 10
            assert noise_strength > 0
        return exterior

    @classmethod
    def _is_polygon_line(cls, exterior):
        vec_down = np.float32([0, 1])
        p1 = exterior[0]
        angles = set()
        for p2 in exterior[1:]:
            vec = np.float32(p2) - np.float32(p1)
            angle = ia.angle_between_vectors(vec_down, vec)
            angles.add(int(angle * 1000))
        return len(angles) <= 1

    def _jitter_duplicate_points(self, exterior, random_state):
        def _find_duplicates(exterior_with_duplicates):
            points_map = collections.defaultdict(list)

            for i, point in enumerate(exterior_with_duplicates):
                # we use 10/x here to be a bit more lenient, the precise
                # distance test is further below
                x = int(np.round(point[0] * ((1/10) / self.threshold_duplicate_points)))
                y = int(np.round(point[1] * ((1/10) / self.threshold_duplicate_points)))
                for d0 in [-1, 0, 1]:
                    for d1 in [-1, 0, 1]:
                        points_map[(x+d0, y+d1)].append(i)

            duplicates = [False] * len(exterior_with_duplicates)
            for key in points_map:
                candidates = points_map[key]
                for i, p0_idx in enumerate(candidates):
                    p0_idx = candidates[i]
                    p0 = exterior_with_duplicates[p0_idx]
                    if duplicates[p0_idx]:
                        continue

                    for j in range(i+1, len(candidates)):
                        p1_idx = candidates[j]
                        p1 = exterior_with_duplicates[p1_idx]
                        if duplicates[p1_idx]:
                            continue

                        dist = np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
                        if dist < self.threshold_duplicate_points:
                            duplicates[p1_idx] = True

            return duplicates

        noise_strength = self.noise_strength
        assert noise_strength > 0
        exterior = exterior[:]
        converged = False
        while not converged:
            duplicates = _find_duplicates(exterior)
            if any(duplicates):
                noise = random_state.uniform(
                    -self.noise_strength, self.noise_strength, size=(len(exterior), 2)
                ).astype(np.float32)
                for i, is_duplicate in enumerate(duplicates):
                    if is_duplicate:
                        exterior[i] = (exterior[i][0] + noise[i][0], exterior[i][1] + noise[i][1])

                noise_strength *= 10
            else:
                converged = True

        return exterior

    # TODO remove?
    @classmethod
    def _calculate_circumference(cls, points):
        assert len(points) >= 3
        points = np.array(points, dtype=np.float32)
        points_matrix = np.zeros((len(points), 4), dtype=np.float32)
        points_matrix[:, 0:2] = points
        points_matrix[0:-1, 2:4] = points_matrix[1:, 0:2]
        points_matrix[-1, 2:4] = points_matrix[0, 0:2]
        distances = np.linalg.norm(
            points_matrix[:, 0:2] - points_matrix[:, 2:4], axis=1)
        return np.sum(distances)

    def _generate_intersection_points(self, exterior, one_point_per_intersection=True):
        assert isinstance(exterior, list)
        assert all([len(point) == 2 for point in exterior])
        if len(exterior) <= 0:
            return []

        # use (*[i][0], *[i][1]) formulation here imnstead of just *[i],
        # because this way we convert numpy arrays to tuples of floats, which
        # is required by isect_segments_include_segments
        segments = [
            (
                (exterior[i][0], exterior[i][1]),
                (exterior[(i + 1) % len(exterior)][0], exterior[(i + 1) % len(exterior)][1])
            )
            for i in range(len(exterior))
        ]

        # returns [(point, [(segment_p0, segment_p1), ..]), ...]
        from imgaug.external.poly_point_isect_py2py3 import isect_segments_include_segments
        intersections = isect_segments_include_segments(segments)

        # estimate to which segment the found intersection points belong
        segments_add_points = [[] for _ in range(len(segments))]
        for point, associated_segments in intersections:
            # the intersection point may be associated with multiple segments,
            # but we only want to add it once, so pick the first segment
            if one_point_per_intersection:
                associated_segments = [associated_segments[0]]

            for seg_inter_p0, seg_inter_p1 in associated_segments:
                diffs = []
                dists = []
                for seg_p0, seg_p1 in segments:
                    dist_p0p0 = np.linalg.norm(seg_p0 - np.array(seg_inter_p0))
                    dist_p1p1 = np.linalg.norm(seg_p1 - np.array(seg_inter_p1))
                    dist_p0p1 = np.linalg.norm(seg_p0 - np.array(seg_inter_p1))
                    dist_p1p0 = np.linalg.norm(seg_p1 - np.array(seg_inter_p0))
                    diff = min(dist_p0p0 + dist_p1p1, dist_p0p1 + dist_p1p0)
                    diffs.append(diff)
                    dists.append(np.linalg.norm(
                        (seg_p0[0] - point[0], seg_p0[1] - point[1])
                    ))

                min_diff = np.min(diffs)
                if min_diff < self.max_segment_difference:
                    idx = int(np.argmin(diffs))
                    segments_add_points[idx].append((point, dists[idx]))
                else:
                    warnings.warn(
                        "Couldn't find fitting segment in "
                        "_generate_intersection_points(). Ignoring intersection "
                        "point.")

        # sort intersection points by their distance to point 0 in each segment
        # (clockwise ordering, this does something only for segments with
        # >=2 intersection points)
        segment_add_points_sorted = []
        for idx in range(len(segments_add_points)):
            points = [t[0] for t in segments_add_points[idx]]
            dists = [t[1] for t in segments_add_points[idx]]
            if len(points) < 2:
                segment_add_points_sorted.append(points)
            else:
                both = sorted(zip(points, dists), key=lambda t: t[1])
                # keep points, drop distances
                segment_add_points_sorted.append([a for a, _b in both])
        return segment_add_points_sorted

    def _oversample_intersection_points(self, exterior, segment_add_points):
        # segment_add_points must be sorted

        if self.oversampling is None or self.oversampling <= 0:
            return segment_add_points

        segment_add_points_sorted_overs = [[] for _ in range(len(segment_add_points))]

        n_points = len(exterior)
        for i, last in enumerate(exterior):
            for j, p_inter in enumerate(segment_add_points[i]):
                direction = (p_inter[0] - last[0], p_inter[1] - last[1])

                if j == 0:
                    # previous point was non-intersection, place 1 new point
                    oversample = [1.0 - self.oversampling]
                else:
                    # previous point was intersection, place 2 new points
                    oversample = [self.oversampling, 1.0 - self.oversampling]

                for dist in oversample:
                    point_over = (last[0] + dist * direction[0],
                                  last[1] + dist * direction[1])
                    segment_add_points_sorted_overs[i].append(point_over)
                segment_add_points_sorted_overs[i].append(p_inter)
                last = p_inter

                is_last_in_group = (j == len(segment_add_points[i]) - 1)
                if is_last_in_group:
                    # previous point was oversampled, next point is
                    # non-intersection, place 1 new point between the two
                    exterior_point = exterior[(i + 1) % len(exterior)]
                    direction = (exterior_point[0] - last[0],
                                 exterior_point[1] - last[1])
                    segment_add_points_sorted_overs[i].append(
                        (last[0] + self.oversampling * direction[0],
                         last[1] + self.oversampling * direction[1])
                    )
                    last = segment_add_points_sorted_overs[i][-1]

                n_points += len(segment_add_points_sorted_overs[i])
                if n_points > self.oversample_up_to_n_points_max:
                    return segment_add_points_sorted_overs

        return segment_add_points_sorted_overs

    @classmethod
    def _insert_intersection_points(cls, exterior, segment_add_points):
        # segment_add_points must be sorted

        assert len(exterior) == len(segment_add_points)
        exterior_interp = []
        for i, p0 in enumerate(exterior):
            p0 = exterior[i]
            exterior_interp.append(p0)
            for p_inter in segment_add_points[i]:
                exterior_interp.append(p_inter)
        return exterior_interp

    def _fit_best_valid_polygon(self, points, random_state):
        if len(points) < 2:
            return None

        def _compute_distance_point_to_line(point, line_start, line_end):
            x_diff = line_end[0] - line_start[0]
            y_diff = line_end[1] - line_start[1]
            num = abs(
                y_diff*point[0] - x_diff*point[1]
                + line_end[0]*line_start[1] - line_end[1]*line_start[0]
            )
            den = np.sqrt(y_diff**2 + x_diff**2)
            if den == 0:
                return np.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)
            return num / den

        poly = Polygon(points)
        if poly.is_valid:
            return sm.xrange(len(points))

        hull = scipy.spatial.ConvexHull(points)
        points_kept = list(hull.vertices)
        points_left = [i for i in range(len(points)) if i not in hull.vertices]

        iteration = 0
        n_changes = 0
        converged = False
        while not converged:
            candidates = []

            # estimate distance metrics for points-segment pairs:
            #  (1) distance (in vertices) between point and segment-start-point
            #      in original input point chain
            #  (2) euclidean distance between point and segment/line
            # TODO this can be done more efficiently by caching the values and
            #      only computing distances to segments that have changed in
            #      the last iteration
            # TODO these distances are not really the best metrics here. Something
            #      like IoU between new and old (invalid) polygon would be
            #      better, but can probably only be computed for pairs of valid
            #      polygons. Maybe something based on pointwise distances,
            #      where the points are sampled on the edges (not edge vertices
            #      themselves). Maybe something based on drawing the perimeter
            #      on images or based on distance maps.
            point_kept_idx_to_pos = {point_idx: i for i, point_idx in enumerate(points_kept)}

            # generate all possible combinations from <points_kept> and <points_left>
            combos = np.transpose([np.tile(np.int32(points_left), len(np.int32(points_kept))),
                                   np.repeat(np.int32(points_kept), len(np.int32(points_left)))])
            combos = np.concatenate(
                (combos, np.zeros((combos.shape[0], 3), dtype=np.int32)),
                axis=1)

            # copy columns 0, 1 into 2, 3 so that 2 is always the lower value
            mask = combos[:, 0] < combos[:, 1]
            combos[:, 2:4] = combos[:, 0:2]
            combos[mask, 2] = combos[mask, 1]
            combos[mask, 3] = combos[mask, 0]

            # distance (in indices) between each pair of <point_kept> and <point_left>
            combos[:, 4] = np.minimum(
                combos[:, 3] - combos[:, 2],
                len(points) - combos[:, 3] + combos[:, 2]
            )

            # limit candidates
            max_dist = self.fit_max_dist_other_iters
            if iteration > 0:
                max_dist = self.fit_max_dist_first_iter
            candidate_rows = combos[combos[:, 4] <= max_dist]
            if self.fit_n_candidates_before_sort_max is not None \
                    and len(candidate_rows) > self.fit_n_candidates_before_sort_max:
                random_state.shuffle(candidate_rows)
                candidate_rows = candidate_rows[0:self.fit_n_candidates_before_sort_max]

            for row in candidate_rows:
                point_left_idx = row[0]
                point_kept_idx = row[1]
                in_points_kept_pos = point_kept_idx_to_pos[point_kept_idx]
                segment_start_idx = point_kept_idx
                segment_end_idx = points_kept[(in_points_kept_pos+1) % len(points_kept)]
                segment_start = points[segment_start_idx]
                segment_end = points[segment_end_idx]
                if iteration == 0:
                    dist_eucl = 0
                else:
                    dist_eucl = _compute_distance_point_to_line(
                        points[point_left_idx], segment_start, segment_end)
                candidates.append((point_left_idx, point_kept_idx, row[4], dist_eucl))

            # Sort computed distances first by minimal vertex-distance (see
            # above, metric 1) (ASC), then by euclidean distance
            # (metric 2) (ASC).
            candidate_ids = np.arange(len(candidates))
            candidate_ids = sorted(candidate_ids, key=lambda idx: (candidates[idx][2], candidates[idx][3]))
            if self.fit_n_changes_max is not None:
                candidate_ids = candidate_ids[:self.fit_n_changes_max]

            # Iterate over point-segment pairs in sorted order. For each such
            # candidate: Add the point to the already collected points,
            # create a polygon from that and check if the polygon is valid.
            # If it is, add the point to the output list and recalculate
            # distance metrics. If it isn't valid, proceed with the next
            # candidate until no more candidates are left.
            #
            # small change: this now no longer breaks upon the first found point
            # that leads to a valid polygon, but checks all candidates instead
            is_valid = False
            done = set()
            for candidate_idx in candidate_ids:
                point_left_idx = candidates[candidate_idx][0]
                point_kept_idx = candidates[candidate_idx][1]
                if (point_left_idx, point_kept_idx) not in done:
                    in_points_kept_idx = [i for i, point_idx in enumerate(points_kept) if point_idx == point_kept_idx][0]
                    points_kept_hypothesis = points_kept[:]
                    points_kept_hypothesis.insert(in_points_kept_idx+1, point_left_idx)
                    poly_hypothesis = Polygon([points[idx] for idx in points_kept_hypothesis])
                    if poly_hypothesis.is_valid:
                        is_valid = True
                        points_kept = points_kept_hypothesis
                        points_left = [point_idx for point_idx in points_left if point_idx != point_left_idx]
                        n_changes += 1
                        if n_changes >= self.fit_n_changes_max:
                            return points_kept
                    done.add((point_left_idx, point_kept_idx))
                    done.add((point_kept_idx, point_left_idx))

            # none of the left points could be used to create a valid polygon?
            # (this automatically covers the case of no points being left)
            if not is_valid and iteration > 0:
                converged = True

            iteration += 1
            if self.fit_n_iters_max is not None and iteration > self.fit_n_iters_max:
                break

        return points_kept


# TODO remove this? was previously only used by Polygon.clip_*(), but that
#      doesn't use it anymore
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
        ia.do_assert(len(geoms) == 0 or all([isinstance(el, Polygon) for el in geoms]))
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
            ia.do_assert(all([isinstance(poly, shapely.geometry.Polygon) for poly in geometry.geoms]))
            return MultiPolygon([Polygon.from_shapely(poly, label=label) for poly in geometry.geoms])
        else:
            raise Exception("Unknown datatype '%s'. Expected shapely.geometry.Polygon or "
                            "shapely.geometry.MultiPolygon or "
                            "shapely.geometry.collections.GeometryCollection." % (type(geometry),))
