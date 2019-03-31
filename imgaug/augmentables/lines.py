from __future__ import print_function, division, absolute_import

import copy as copylib

import numpy as np
import skimage.draw
import skimage.measure
import cv2

from .. import imgaug as ia

"""
TODO
def compute_distance(self, other):
        # TODO
        pass

def compute_hausdorff_distance(self, other):
    # TODO
    pass

# TODO
def intersects(self, other):
    pass

# TODO
def find_self_intersections(self):
    pass

# TODO
def is_self_intersecting(self):
    pass

# TODO
def remove_self_intersections(self):
    pass

"""


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
        return np.sum(self.get_pointwise_distances())

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

    def get_pointwise_distances(self):
        """
        Get the euclidean length of each point to the next.

        Returns
        -------
        ndarray
            Euclidean distances between point pairs.
            Same order as in `coords`. For ``N`` points, ``N-1`` distances
            are returned.

        """
        if len(self.coords) <= 1:
            return 0
        return np.sqrt(
            np.sum(
                (self.coords[:-1, :] - self.coords[1:, :]) ** 2,
                axis=1
            )
        )

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
        ndarray(bool):
            Boolean array with one value per point indicating whether it is
            inside of the provided image plane (``True``) or not (``False``).

        """
        shape = _parse_shape(image)
        height, width = shape[0:2]
        x_within = np.logical_and(0 <= self.xx, self.xx < width)
        y_within = np.logical_and(0 <= self.yy, self.yy < height)
        return np.logical_and(x_within, y_within)

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
        import shapely.geometry
        from .kps import Keypoint

        if isinstance(other, Keypoint):
            other = shapely.geometry.Point((other.x, other.y))
        elif isinstance(other, LineString):
            if len(other.coords) == 0:
                return default
            other = shapely.geometry.LineString(other.coords)
        elif isinstance(other, tuple):
            assert len(other) == 2
            other = shapely.geometry.Point(other)
        else:
            raise ValueError(
                ("Expected Keypoint or LineString or tuple (x,y), "
                 + "got type %s.") % (type(other),))

        if len(self.coords) == 0:
            return default
        elif len(self.coords) == 1:
            return shapely.geometry.Point(self.coords[0]).distance(other)

        return shapely.geometry.LineString(self.coords).distance(other)

    # TODO update BB's contains(), which can only accept Keypoint currently
    def contains(self, other, distance_threshold=1e-4):
        """
        Estimate whether the bounding box contains a point.

        Parameters
        ----------
        other : tuple of number or imgaug.augmentables.kps.Keypoint
            Point to check for.

        distance_threshold : float
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
        return self.compute_distance(other, default=np.inf) < distance_threshold

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
        coords_proj = _project_coords(self.coords, from_shape, to_shape)
        return self.copy(coords=coords_proj, label=self.label)

    # TODO
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

    # TODO
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
        return len(self.clip_out_of_image(image).coords) > 0

    def is_out_of_image(self, image, fully=True, partly=False):
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

        Returns
        -------
        bool
            True if the line string is partially/fully outside of the image
            area, depending on defined parameters.
            False otherwise.

        """
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
        result : imgaug.augmentables.lines.LineString
            Line string, clipped to fall within the image dimensions.

        """
        if len(self.coords) == 0:
            return self
        elif len(self.coords) == 1:
            if not np.any(self.get_pointwise_inside_image_mask(image)):
                return self.copy(np.zeros((0, 2), dtype=np.float32))
            return self.copy()

        ooi_mask = ~self.get_pointwise_inside_image_mask(image)

        # all points inside the image?
        if not np.any(ooi_mask):
            return self.copy()

        # top, right, bottom, left image edges
        import shapely.geometry
        height, width = image.shape[0:2]
        edges = [
            shapely.geometry.LineString([(0.0, 0.0), (width, 0.0)]),
            shapely.geometry.LineString([(width, 0.0), (width, height)]),
            shapely.geometry.LineString([(width, height), (0.0, height)]),
            shapely.geometry.LineString([(0.0, height), (0.0, 0.0)])
        ]

        coords = []
        gen = enumerate(zip(self.coords[:-1], self.coords[1:],
                            ooi_mask[:-1], ooi_mask[1:]))
        for i, (line_start, line_end, ooi_start, ooi_end) in gen:
            # note that we can't skip the line segment if both points are
            # outside of the image, because the line segment may still
            # intersect with the image edges
            if not ooi_start and not ooi_end:
                # line segment fully inside the image
                coords.append(line_start)
                continue
            elif not ooi_start:
                # the start point is inside the image
                coords.append(line_start)

            line_segment = shapely.geometry.LineString([line_start, line_end])
            intersections = [line_segment.intersection(edge) for edge in edges]

            if intersections:
                # We might get line segments instead of points as intersections
                # if the line segment overlaps with an edge.
                inter_clean = []
                for p in intersections:
                    if isinstance(p, shapely.geometry.Point):
                        inter_clean.append(p)
                    elif isinstance(p, shapely.geometry.LineString):
                        inter_start = p.coords[0]
                        inter_end = p.coords[1]
                        if ooi_start:
                            inter_clean.append(inter_start)
                        if ooi_end:
                            inter_clean.append(inter_end)
                    else:
                        raise Exception(
                            "Got unexpected type from shapely."
                            "Input was (%s, %s), got %s (type %s)." % (
                                line_start, line_end, p, type(p)))

                # There can be 0, 1 or 2 intersection points.
                # They are ordered as: top-side, right-side, bottom-side,
                # left-side. That may not match the direction of the line
                # segment, so we sort here by distance to the line segment
                # start point.
                intersections = sorted(
                    intersections,
                    key=lambda p: np.linalg.norm(np.float32(p) - line_start)
                )
                self.coords.extend(intersections)

            is_last = (i == len(self.coords) - 1)
            if is_last and not ooi_end:
                coords.append(line_end)

        return self.copy(coords)

    # TODO convert this to x/y params?
    def shift(self, top=None, right=None, bottom=None, left=None):
        """
        Shift/move the line string from one or more image sides.

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

    def draw_mask(self, image_shape, size_line=1, size_points=0,
                  raise_if_out_of_image=False):
        """
        Draw this line segment as a binary image mask.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        size_line : int, optional
            Thickness of the line.

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
            alpha_line=1.0, alpha_points=1.0,
            size_line=size_line, size_points=size_points,
            antialiased=False,
            raise_if_out_of_image=raise_if_out_of_image)
        return heatmap > 0.5

    def draw_line_heatmap_array(self, image_shape, alpha=1.0,
                                size=1, antialiased=False,
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
            Thickness of the line.

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

        # TODO remove?
        """
        if raise_if_out_of_image and self.is_out_of_image(image_shape):
            raise Exception(
                "Cannot draw line string '%s' on image shape %s, because "
                "it would be out of bounds." % (
                    self.__str__(), image_shape))

        image_shape = _parse_shape(image_shape)[0:2]
        if len(self.coords) <= 1 or alpha < 0 + 1e-4:
            return np.zeros(image_shape, dtype=np.float32)

        lines = []
        for line_start, line_end in zip(self.coords):
            lines.append((line_start[0], line_start[1],
                          line_end[0], line_end[1]))

        heatmap = np.zeros(image_shape, dtype=np.float32)
        for line in lines:
            if antialiased:
                rr, cc, val = skimage.draw.line_aa(*line)
            else:
                rr, cc = skimage.draw.line(*line)
                val = 1.0
            heatmap[rr, cc] = val * alpha

        if size > 1:
            kernel = np.ones((size, size), dtype=np.uint8)
            heatmap = cv2.dilate(heatmap, kernel)

        return heatmap
        """
        image = self.draw_line_on_image(
            np.zeros(image_shape, dtype=np.uint8),
            color=(255, 255, 255), alpha=alpha, size=size,
            antialiased=antialiased,
            raise_if_out_of_image=raise_if_out_of_image
        )
        return image[:, :, 0].astype(np.float32) / 255.0

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

        # TODO remove?
        """
        points_ooi = TODO
        if raise_if_out_of_image and all(points_ooi):
            raise Exception(
                "Cannot draw points of line string '%s' on image shape %s, "
                "because they would be out of bounds." % (
                    self.__str__(), image_shape))

        image_shape = _parse_shape(image_shape)[0:2]
        if len(self.coords) <= 1 or alpha < 0 + 1e-4:
            return np.zeros(image_shape, dtype=np.float32)

        # keypoint drawing currently only works with uint8 RGB images,
        # so we add (3,) to the shape and set dtype to uint8. Later on and
        # we will remove the axis and convert to [0.0, 1.0].
        heatmap = np.zeros(image_shape[0:2] + (3,), dtype=np.uint8)

        if alpha > 0:
            from .kps import KeypointsOnImage

            kpsoi = KeypointsOnImage.from_coords_array(self.coords,
                                                       shape=image_shape)
            heatmap = kpsoi.draw_on_image(
                heatmap, color=(255, 255, 255), alpha=alpha,
                size=size, copy=False,
                raise_if_out_of_image=raise_if_out_of_image)

        heatmap = heatmap.astype(np.float32) / 255.0
        heatmap = heatmap[:, :, 0]
        return heatmap
        """
        image = self.draw_points_on_image(
            np.zeros(image_shape, dtype=np.uint8),
            color=(255, 255, 255), alpha=alpha, size=size,
            raise_if_out_of_image=raise_if_out_of_image
        )
        return image[:, :, 0].astype(np.float32) / 255.0

    def draw_heatmap_array(self, image_shape, alpha_line=1.0, alpha_points=1.0,
                           size_line=1, size_points=0, antialiased=False,
                           raise_if_out_of_image=False):
        """
        Draw the line segments and points of the line string as a heatmap array.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        alpha_line : float, optional
            Opacity of the line string. Higher values denote a more visible
            line string.

        alpha_points : float, optional
            Opacity of the line string points. Higher values denote a more
            visible points.

        size_line : int, optional
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
        ndarray
            Float array of shape `image_shape` (no channel axis) with drawn
            line segments and points. All values are in the
            interval ``[0.0, 1.0]``.

        """
        heatmap_line = self.draw_line_heatmap_array(
            image_shape,
            alpha=alpha_line,
            size=size_line,
            antialiased=antialiased,
            raise_if_out_of_image=raise_if_out_of_image)
        if size_points <= 0:
            return heatmap_line

        heatmap_points = self.draw_points_heatmap_array(
            image_shape,
            alpha=alpha_points,
            size=size_points,
            raise_if_out_of_image=raise_if_out_of_image)

        heatmap = np.dstack([heatmap_line, heatmap_points])
        return np.max(heatmap, axis=2)

    # TODO only draw line on image of size BB around line, then paste into full
    #      sized image
    def draw_line_on_image(self, image, color=(0, 255, 0),
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

        color : iterable of int
            Color to use as RGB, i.e. three values.

        alpha : float, optional
            Opacity of the line string. Higher values denote a more visible
            line string.

        size : int, optional
            Thickness of the line.

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
        assert len(image) == 3, (
            ("Expected image or shape of form (H, W, C), "
             + "got shape %s.") % (image.shape,))

        if len(self.coords) <= 1 or alpha < 0 + 1e-4 or size < 1:
            return np.copy(image)

        if raise_if_out_of_image and self.is_out_of_image(image):
            raise Exception(
                "Cannot draw line string '%s' on image with shape %s, because "
                "it would be out of bounds." % (
                    self.__str__(), image.shape))

        lines = []
        for line_start, line_end in zip(self.coords):
            lines.append((line_start[0], line_start[1],
                          line_end[0], line_end[1]))

        if size == 1:
            color = np.float32(color).reshape((1, 1, -1))
            image = image.astype(np.float32)
            for line in lines:
                if antialiased:
                    rr, cc, val = skimage.draw.line_aa(*line)
                else:
                    rr, cc = skimage.draw.line(*line)
                    val = 1.0
                pixels = image[rr, cc]
                val = val * alpha
                if image_was_empty:
                    image[rr, cc] = val * color
                else:
                    image[rr, cc] = (1 - val) * pixels + val * color
            return iadt.restore_dtypes_(image, np.uint8)
        else:
            color = np.uint8(color).reshape((1, 1, -1))
            heatmap = np.zeros(image.shape[0:2], dtype=np.float32)
            for line in lines:
                if antialiased:
                    rr, cc, val = skimage.draw.line_aa(*line)
                else:
                    rr, cc = skimage.draw.line(*line)
                    val = 1.0
                heatmap[rr, cc] = val * alpha

            kernel = np.ones((size, size), dtype=np.uint8)
            heatmap = cv2.dilate(heatmap, kernel)

            if image_was_empty:
                image_blend = image + heatmap * color
            else:
                mask = heatmap * alpha
                image_color = np.tile(
                    color, image.shape[0:2] + (color.shape[2],))
                image_blend = blendlib.blend_alpha(image_color, image, mask)

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
        if len(self.coords) <= 0 or alpha < 0 + 1e-4 or size < 1:
            return np.copy(image)

        points_inside = self.get_pointwise_inside_image_mask(image)
        if raise_if_out_of_image and not np.any(points_inside):
            raise Exception(
                "Cannot draw points of line string '%s' on image with "
                "shape %s, because they would be out of bounds." % (
                    self.__str__(), image.shape))

        from .kps import KeypointsOnImage
        kpsoi = KeypointsOnImage.from_coords_array(self.coords,
                                                   shape=image.shape)
        image = kpsoi.draw_on_image(
            image, color=color, alpha=alpha,
            size=size, copy=copy,
            raise_if_out_of_image=raise_if_out_of_image)

        return image

    def draw_on_image(self, image,
                      color=(0, 255, 0), color_line=None, color_points=None,
                      alpha=1.0, alpha_line=None, alpha_points=None,
                      size=1, size_line=None, size_points=None,
                      antialiased=True,
                      raise_if_out_of_image=False):
        """
        Draw the line string on an image.

        Parameters
        ----------
        image : (H,W,C) ndarray(uint8)
            The image onto which to draw the bounding box.

        color : iterable of int, optional
            Color to use as RGB, i.e. three values.
            The color of the line and points are derived from this value,
            unless they are set.

        color_line : None or iterable of int
            Color to use for the line segments as RGB, i.e. three values.
            If ``None``, this value is derived from `color`.

        color_points : None or iterable of int
            Color to use for the points as RGB, i.e. three values.
            If ``None``, this value is derived from ``0.5 * color``.

        alpha : float, optional
            Opacity of the line string. Higher values denote a more visible
            points.
            The alphas of the line and points are derived from this value,
            unless they are set.

        alpha_line : None or float, optional
            Opacity of the line string. Higher values denote a more visible
            line string.
            If ``None``, this value is derived from `alpha`.

        alpha_points : None or float, optional
            Opacity of the line string points. Higher values denote a more
            visible points.
            If ``None``, this value is derived from `alpha`.

        size : int, optional
            Thickness of the line string.
            The sizes of the line and points are derived from this value,
            unless they are set.

        size_line : None or int, optional
            Thickness of the line.
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

        color_line = color_line if color_line is not None else np.float32(color)
        color_points = color_points if color_points is not None else np.float32(color) * 0.5

        alpha_line = alpha_line if alpha_line is not None else np.float32(alpha)
        alpha_points = alpha_points if alpha_points is not None else np.float32(alpha)

        size_line = size_line if size_line is not None else size
        size_points = size_points if size_points is not None else size * 3

        image = self.draw_line_on_image(
            image, color=color_line.astype(np.uint8),
            alpha=alpha_line, size=size_line,
            antialiased=antialiased,
            raise_if_out_of_image=raise_if_out_of_image)

        image = self.draw_points_on_image(
            image, color=color_points.astype(np.uint8),
            alpha=alpha_points, size=size_points,
            copy=False,
            raise_if_out_of_image=raise_if_out_of_image)

        return image

    # TODO
    def extract_from_image(self, image, size=3, pad=True, pad_max=10*1000,
                           antialiased=True, prevent_zero_size=True):
        """
        Extract the image pixels covered by the line string.

        It will only extract pixels overlapped by the line string.

        This function will by default zero-pad the image if the line string is
        partially/fully outside of the image. This is for consistency with
        the same implementations for bounding boxes and polygons.

        Parameters
        ----------
        image : (H,W) ndarray or (H,W,C) ndarray
            The image from which to extract the pixels within the bounding box.

        size : int, optional
            Thickness of the line.

        pad : bool, optional
            Whether to zero-pad the image if the line string is
            partially/fully outside of it.

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

        if len(self.coords) == 0:
            if prevent_zero_size:
                return np.zeros((1, 1) + image.shape[2:], dtype=image.dtype)
            return np.zeros((0, 0) + image.shape[2:], dtype=image.dtype)
        elif len(self.coords) == 1:
            x1 = self.coords[0, 0] - (size / 2)
            y1 = self.coords[0, 1] - (size / 2)
            x2 = self.coords[1, 0] + (size / 2)
            y2 = self.coords[1, 1] + (size / 2)
            bb = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            return bb.extract_from_image(image, pad=pad, pad_max=pad_max,
                                         prevent_zero_size=prevent_zero_size)

        heatmap = self.draw_line_heatmap_array(
            image.shape[0:2], alpha=1.0, size=size, antialiased=antialiased)

        image = image.astype(np.float32) * heatmap
        bb = self.to_bounding_box()
        return bb.extract_from_image(image, pad=pad, pad_max=pad_max,
                                     prevent_zero_size=prevent_zero_size)

    def concat(self, other):
        """
        Concat this line string with another one.

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

    # TODO
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

    # TODO
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

    # TODO
    def to_heatmap(self, image_shape, size_line=1, size_points=0,
                   antialiased=True, raise_if_out_of_image=False):
        """
        Generate a heatmap object from the line string.

        This is similar to
        :func:`imgaug.augmentables.lines.LineString.draw_line_heatmap_array`
        executed with ``alpha=1.0``. The result is wrapped in a
        ``HeatmapsOnImage`` object instead of just an array.
        No points are drawn.

        Parameters
        ----------
        image_shape : tuple of int
            The shape of the image onto which to draw the line mask.

        size_line : int, optional
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
                image_shape, size_line=size_line, size_points=size_points,
                antialiased=antialiased,
                raise_if_out_of_image=raise_if_out_of_image),
            shape=image_shape
        )

    # TODO
    def to_segmentation_map(self, image_shape, size_line=1, size_points=0,
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

        size_line : int, optional
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
                image_shape, size_line=size_line, size_points=size_points,
                raise_if_out_of_image=raise_if_out_of_image),
            shape=image_shape
        )

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


def _is_point_on_line(line_start, line_end, point, eps=1e-4):
    dist_s2e = np.linalg.norm(np.float32(line_start) - np.float32(line_end))
    dist_s2p2e = (
        np.linalg.norm(np.float32(line_start) - np.float32(point))
        + np.linalg.norm(np.float32(point) - np.float32(line_end))
    )
    return -eps < (dist_s2p2e - dist_s2e) < eps


# TODO merge with implementation in bbs.py
def _parse_shape(shape):
    if isinstance(shape, tuple):
        return shape
    assert ia.is_np_array(shape), (
        "Expected tuple of ints or array, got %s." % (type(shape),))
    return shape.shape


# TODO merge with line BBs
# TODO intergrate into polygons
# TODO integrate into keypoints
def _project_coords(coords, from_shape, to_shape):
    from_shape = _parse_shape(from_shape)
    to_shape = _parse_shape(to_shape)
    if from_shape[0:2] == to_shape[0:2]:
        return coords

    from_height, from_width = from_shape[0:2]
    to_height, to_width = to_shape[0:2]
    assert all([v > 0 for v in [from_height, from_width, to_height, to_width]])

    coords_proj = np.float32(coords)
    coords_proj[:, 0] = (coords_proj[:, 0] / from_width) * to_width
    coords_proj[:, 1] = (coords_proj[:, 1] / from_height) * to_height
    return coords
