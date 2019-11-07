from __future__ import print_function, division, absolute_import

import copy

import numpy as np
import skimage.draw
import skimage.measure

from .. import imgaug as ia
from .utils import normalize_shape, project_coords


# TODO functions: square(), to_aspect_ratio(), contains_point()
class BoundingBox(object):
    """Class representing bounding boxes.

    Each bounding box is parameterized by its top left and bottom right
    corners. Both are given as x and y-coordinates. The corners are intended
    to lie inside the bounding box area. As a result, a bounding box that lies
    completely inside the image but has maximum extensions would have
    coordinates ``(0.0, 0.0)`` and ``(W - epsilon, H - epsilon)``. Note that
    coordinates are saved internally as floats.

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
        if y1 > y2:
            y2, y1 = y1, y2

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label

    @property
    def coords(self):
        """Get the top-left and bottom-right coordinates as one array.

        Returns
        -------
        ndarray
            A ``(N, 2)`` numpy array with ``N=2`` containing the top-left
            and bottom-right coordinates.

        """
        arr = np.empty((2, 2), dtype=np.float32)
        arr[0, :] = (self.x1, self.y1)
        arr[1, :] = (self.x2, self.y2)
        return arr

    @property
    def x1_int(self):
        """Get the x-coordinate of the top left corner as an integer.

        Returns
        -------
        int
            X-coordinate of the top left corner, rounded to the closest
            integer.

        """
        # use numpy's round to have consistent behaviour between python
        # versions
        return int(np.round(self.x1))

    @property
    def y1_int(self):
        """Get the y-coordinate of the top left corner as an integer.

        Returns
        -------
        int
            Y-coordinate of the top left corner, rounded to the closest
            integer.

        """
        # use numpy's round to have consistent behaviour between python
        # versions
        return int(np.round(self.y1))

    @property
    def x2_int(self):
        """Get the x-coordinate of the bottom left corner as an integer.

        Returns
        -------
        int
            X-coordinate of the bottom left corner, rounded to the closest
            integer.

        """
        # use numpy's round to have consistent behaviour between python
        # versions
        return int(np.round(self.x2))

    @property
    def y2_int(self):
        """Get the y-coordinate of the bottom left corner as an integer.

        Returns
        -------
        int
            Y-coordinate of the bottom left corner, rounded to the closest
            integer.

        """
        # use numpy's round to have consistent behaviour between python
        # versions
        return int(np.round(self.y2))

    @property
    def height(self):
        """Estimate the height of the bounding box.

        Returns
        -------
        number
            Height of the bounding box.

        """
        return self.y2 - self.y1

    @property
    def width(self):
        """Estimate the width of the bounding box.

        Returns
        -------
        number
            Width of the bounding box.

        """
        return self.x2 - self.x1

    @property
    def center_x(self):
        """Estimate the x-coordinate of the center point of the bounding box.

        Returns
        -------
        number
            X-coordinate of the center point of the bounding box.

        """
        return self.x1 + self.width/2

    @property
    def center_y(self):
        """Estimate the y-coordinate of the center point of the bounding box.

        Returns
        -------
        number
            Y-coordinate of the center point of the bounding box.

        """
        return self.y1 + self.height/2

    @property
    def area(self):
        """Estimate the area of the bounding box.

        Returns
        -------
        number
            Area of the bounding box, i.e. ``height * width``.

        """
        return self.height * self.width

    # TODO add test for tuple of number
    def contains(self, other):
        """Estimate whether the bounding box contains a given point.

        Parameters
        ----------
        other : tuple of number or imgaug.augmentables.kps.Keypoint
            Point to check for.

        Returns
        -------
        bool
            ``True`` if the point is contained in the bounding box,
            ``False`` otherwise.

        """
        if isinstance(other, tuple):
            x, y = other
        else:
            x, y = other.x, other.y
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    # TODO add tests for ndarray inputs
    def project(self, from_shape, to_shape):
        """Project the bounding box onto a differently shaped image.

        E.g. if the bounding box is on its original image at
        ``x1=(10 of 100 pixels)`` and ``y1=(20 of 100 pixels)`` and is
        projected onto a new image with size ``(width=200, height=200)``,
        its new position will be ``(x1=20, y1=40)``.
        (Analogous for ``x2``/``y2``.)

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
        imgaug.augmentables.bbs.BoundingBox
            ``BoundingBox`` instance with new coordinates.

        """
        coords_proj = project_coords([(self.x1, self.y1), (self.x2, self.y2)],
                                     from_shape, to_shape)
        return self.copy(
            x1=coords_proj[0][0],
            y1=coords_proj[0][1],
            x2=coords_proj[1][0],
            y2=coords_proj[1][1],
            label=self.label)

    def extend(self, all_sides=0, top=0, right=0, bottom=0, left=0):
        """Extend the size of the bounding box along its sides.

        Parameters
        ----------
        all_sides : number, optional
            Value by which to extend the bounding box size along all
            sides.

        top : number, optional
            Value by which to extend the bounding box size along its top
            side.

        right : number, optional
            Value by which to extend the bounding box size along its right
            side.

        bottom : number, optional
            Value by which to extend the bounding box size along its bottom
            side.

        left : number, optional
            Value by which to extend the bounding box size along its left
            side.

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
        """Compute the intersection BB between this BB and another BB.

        Note that in extreme cases, the intersection can be a single point.
        In that case the intersection bounding box exists and it will be
        returned, but it will have a height and width of zero.

        Parameters
        ----------
        other : imgaug.augmentables.bbs.BoundingBox
            Other bounding box with which to generate the intersection.

        default : any, optional
            Default value to return if there is no intersection.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBox or any
            Intersection bounding box of the two bounding boxes if there is
            an intersection.
            If there is no intersection, the default value will be returned,
            which can by anything.

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
        """Compute the union BB between this BB and another BB.

        This is equivalent to drawing a bounding box around all corner points
        of both bounding boxes.

        Parameters
        ----------
        other : imgaug.augmentables.bbs.BoundingBox
            Other bounding box with which to generate the union.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBox
            Union bounding box of the two bounding boxes.

        """
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def iou(self, other):
        """Compute the IoU between this bounding box and another one.

        IoU is the intersection over union, defined as::

            ``area(intersection(A, B)) / area(union(A, B))``
            ``= area(intersection(A, B))
                / (area(A) + area(B) - area(intersection(A, B)))``

        Parameters
        ----------
        other : imgaug.augmentables.bbs.BoundingBox
            Other bounding box with which to compare.

        Returns
        -------
        float
            IoU between the two bounding boxes.

        """
        inters = self.intersection(other)
        if inters is None:
            return 0.0
        area_union = self.area + other.area - inters.area
        return inters.area / area_union if area_union > 0 else 0.0

    def compute_out_of_image_area(self, image):
        """Compute the area of the BB that is outside of the image plane.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.

        Returns
        -------
        float
            Total area of the bounding box that is outside of the image plane.
            Can be ``0.0``.

        """
        shape = normalize_shape(image)
        height, width = shape[0:2]
        bb_image = BoundingBox(x1=0, y1=0, x2=width, y2=height)
        inter = self.intersection(bb_image, default=None)
        area = self.area
        return area if inter is None else area - inter.area

    def compute_out_of_image_fraction(self, image):
        """Compute fraction of BB area outside of the image plane.

        This estimates ``f = A_ooi / A``, where ``A_ooi`` is the area of the
        bounding box that is outside of the image plane, while ``A`` is the
        total area of the bounding box.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.

        Returns
        -------
        float
            Fraction of the bounding box area that is outside of the image
            plane. Returns ``0.0`` if the bounding box is fully inside of
            the image plane. If the bounding box has an area of zero, the
            result is ``1.0`` if its coordinates are outside of the image
            plane, otherwise ``0.0``.

        """
        area = self.area
        if area == 0:
            shape = normalize_shape(image)
            height, width = shape[0:2]
            y1_outside = self.y1 < 0 or self.y1 >= height
            x1_outside = self.x1 < 0 or self.x1 >= width
            is_outside = (y1_outside or x1_outside)
            return 1.0 if is_outside else 0.0
        return self.compute_out_of_image_area(image) / area

    def is_fully_within_image(self, image):
        """Estimate whether the bounding box is fully inside the image area.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.

        Returns
        -------
        bool
            ``True`` if the bounding box is fully inside the image area.
            ``False`` otherwise.

        """
        shape = normalize_shape(image)
        height, width = shape[0:2]
        return (
            self.x1 >= 0
            and self.x2 < width
            and self.y1 >= 0
            and self.y2 < height)

    def is_partly_within_image(self, image):
        """Estimate whether the BB is at least partially inside the image area.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.

        Returns
        -------
        bool
            ``True`` if the bounding box is at least partially inside the
            image area.
            ``False`` otherwise.

        """
        shape = normalize_shape(image)
        height, width = shape[0:2]
        eps = np.finfo(np.float32).eps
        img_bb = BoundingBox(x1=0, x2=width-eps, y1=0, y2=height-eps)
        return self.intersection(img_bb) is not None

    def is_out_of_image(self, image, fully=True, partly=False):
        """Estimate whether the BB is partially/fully outside of the image area.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape and
            must contain at least two integers.

        fully : bool, optional
            Whether to return ``True`` if the bounding box is fully outside
            of the image area.

        partly : bool, optional
            Whether to return ``True`` if the bounding box is at least
            partially outside fo the image area.

        Returns
        -------
        bool
            ``True`` if the bounding box is partially/fully outside of the
            image area, depending on defined parameters.
            ``False`` otherwise.

        """
        if self.is_fully_within_image(image):
            return False
        elif self.is_partly_within_image(image):
            return partly
        return fully

    @ia.deprecated(alt_func="BoundingBox.clip_out_of_image()",
                   comment="clip_out_of_image() has the exactly same "
                           "interface.")
    def cut_out_of_image(self, *args, **kwargs):
        return self.clip_out_of_image(*args, **kwargs)

    def clip_out_of_image(self, image):
        """Clip off all parts of the BB box that are outside of the image.

        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use for the clipping of the bounding box.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape and
            must contain at least two integers.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBox
            Bounding box, clipped to fall within the image dimensions.

        """
        shape = normalize_shape(image)

        height, width = shape[0:2]
        assert height > 0, (
            "Expected image with height>0, got shape %s." % (image.shape,))
        assert width > 0, (
            "Expected image with width>0, got shape %s." % (image.shape,))

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
        """Move this bounding box along the x/y-axis.

        Parameters
        ----------
        top : None or int, optional
            Amount of pixels by which to shift this object *from* the
            top (towards the bottom).

        right : None or int, optional
            Amount of pixels by which to shift this object *from* the
            right (towards the left).

        bottom : None or int, optional
            Amount of pixels by which to shift this object *from* the
            bottom (towards the top).

        left : None or int, optional
            Amount of pixels by which to shift this object *from* the
            left (towards the right).

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBox
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
    def draw_on_image(self, image, color=(0, 255, 0), alpha=1.0, size=1,
                      copy=True, raise_if_out_of_image=False, thickness=None):
        """Draw the bounding box on an image.

        Parameters
        ----------
        image : (H,W,C) ndarray
            The image onto which to draw the bounding box.
            Currently expected to be ``uint8``.

        color : iterable of int, optional
            The color to use, corresponding to the channel layout of the
            image. Usually RGB.

        alpha : float, optional
            The transparency of the drawn bounding box, where ``1.0`` denotes
            no transparency and ``0.0`` is invisible.

        size : int, optional
            The thickness of the bounding box in pixels. If the value is
            larger than ``1``, then additional pixels will be added around
            the bounding box (i.e. extension towards the outside).

        copy : bool, optional
            Whether to copy the input image or change it in-place.

        raise_if_out_of_image : bool, optional
            Whether to raise an error if the bounding box is fully outside of
            the image. If set to ``False``, no error will be raised and only
            the parts inside the image will be drawn.

        thickness : None or int, optional
            Deprecated.

        Returns
        -------
        (H,W,C) ndarray(uint8)
            Image with bounding box drawn on it.

        """
        if thickness is not None:
            ia.warn_deprecated(
                "Usage of argument 'thickness' in BoundingBox.draw_on_image() "
                "is deprecated. The argument was renamed to 'size'.")
            size = thickness

        if raise_if_out_of_image and self.is_out_of_image(image):
            raise Exception(
                "Cannot draw bounding box x1=%.8f, y1=%.8f, x2=%.8f, y2=%.8f "
                "on image with shape %s." % (
                    self.x1, self.y1, self.x2, self.y2, image.shape))

        result = np.copy(image) if copy else image

        if isinstance(color, (tuple, list)):
            color = np.uint8(color)

        for i in range(size):
            y1, y2, x1, x2 = self.y1_int, self.y2_int, self.x1_int, self.x2_int

            # When y values get into the range (H-0.5, H), the *_int functions
            # round them to H. That is technically sensible, but in the case
            # of drawing means that the border lies just barely outside of
            # the image, making the border disappear, even though the BB is
            # fully inside the image. Here we correct for that because of
            # beauty reasons. Same is the case for x coordinates.
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
                if ia.is_float_array(result):
                    # TODO use blend_alpha here
                    result[rr, cc, :] = (
                        (1 - alpha) * result[rr, cc, :]
                        + alpha * color)
                    result = np.clip(result, 0, 255)
                else:
                    input_dtype = result.dtype
                    result = result.astype(np.float32)
                    result[rr, cc, :] = (
                        (1 - alpha) * result[rr, cc, :]
                        + alpha * color)
                    result = np.clip(result, 0, 255).astype(input_dtype)

        return result

    # TODO add tests for pad and pad_max
    def extract_from_image(self, image, pad=True, pad_max=None,
                           prevent_zero_size=True):
        """Extract the image pixels within the bounding box.

        This function will zero-pad the image if the bounding box is
        partially/fully outside of the image.

        Parameters
        ----------
        image : (H,W) ndarray or (H,W,C) ndarray
            The image from which to extract the pixels within the bounding box.

        pad : bool, optional
            Whether to zero-pad the image if the object is partially/fully
            outside of it.

        pad_max : None or int, optional
            The maximum number of pixels that may be zero-paded on any side,
            i.e. if this has value ``N`` the total maximum of added pixels
            is ``4*N``.
            This option exists to prevent extremely large images as a result of
            single points being moved very far away during augmentation.

        prevent_zero_size : bool, optional
            Whether to prevent the height or width of the extracted image from
            becoming zero.
            If this is set to ``True`` and the height or width of the bounding
            box is below ``1``, the height/width will be increased to ``1``.
            This can be useful to prevent problems, e.g. with image saving or
            plotting.
            If it is set to ``False``, images will be returned as ``(H', W')``
            or ``(H', W', 3)`` with ``H`` or ``W`` potentially being 0.

        Returns
        -------
        (H',W') ndarray or (H',W',C) ndarray
            Pixels within the bounding box. Zero-padded if the bounding box
            is partially/fully outside of the image.
            If `prevent_zero_size` is activated, it is guarantueed that
            ``H'>0`` and ``W'>0``, otherwise only ``H'>=0`` and ``W'>=0``.

        """
        height, width = image.shape[0], image.shape[1]
        x1, x2, y1, y2 = self.x1_int, self.x2_int, self.y1_int, self.y2_int

        # When y values get into the range (H-0.5, H), the *_int functions
        # round them to H. That is technically sensible, but in the case of
        # extraction leads to a black border, which is both ugly and
        # unexpected after calling cut_out_of_image(). Here we correct for
        # that because of beauty reasons. Same is the case for x coordinates.
        fully_within = self.is_fully_within_image(image)
        if fully_within:
            y1, y2 = np.clip([y1, y2], 0, height-1)
            x1, x2 = np.clip([x1, x2], 0, width-1)

        # TODO add test
        if prevent_zero_size:
            if abs(x2 - x1) < 1:
                x2 = x1 + 1
            if abs(y2 - y1) < 1:
                y2 = y1 + 1

        if pad:
            # if the bb is outside of the image area, the following pads the
            # image first with black pixels until the bb is inside the image
            # and only then extracts the image area
            # TODO probably more efficient to initialize an array of zeros
            #      and copy only the portions of the bb into that array that
            #      are natively inside the image area
            from ..augmenters import size as iasize

            pad_top = 0
            pad_right = 0
            pad_bottom = 0
            pad_left = 0

            if x1 < 0:
                pad_left = abs(x1)
                x2 = x2 + pad_left
                width = width + pad_left
                x1 = 0
            if y1 < 0:
                pad_top = abs(y1)
                y2 = y2 + pad_top
                height = height + pad_top
                y1 = 0
            if x2 >= width:
                pad_right = x2 - width
            if y2 >= height:
                pad_bottom = y2 - height

            paddings = [pad_top, pad_right, pad_bottom, pad_left]
            any_padded = any([val > 0 for val in paddings])
            if any_padded:
                if pad_max is None:
                    pad_max = max(paddings)

                image = iasize.pad(
                    image,
                    top=min(pad_top, pad_max),
                    right=min(pad_right, pad_max),
                    bottom=min(pad_bottom, pad_max),
                    left=min(pad_left, pad_max)
                )
            return image[y1:y2, x1:x2]
        else:
            within_image = (
                (0, 0, 0, 0)
                <= (x1, y1, x2, y2)
                < (width, height, width, height)
            )
            out_height, out_width = (y2 - y1), (x2 - x1)
            nonzero_height = (out_height > 0)
            nonzero_width = (out_width > 0)
            if within_image and nonzero_height and nonzero_width:
                return image[y1:y2, x1:x2]
            if prevent_zero_size:
                out_height = 1
                out_width = 1
            else:
                out_height = 0
                out_width = 0
            if image.ndim == 2:
                return np.zeros((out_height, out_width), dtype=image.dtype)
            return np.zeros((out_height, out_width, image.shape[-1]),
                            dtype=image.dtype)

    # TODO also add to_heatmap
    # TODO add this to BoundingBoxesOnImage
    def to_keypoints(self):
        """Convert the BB's corners to keypoints (clockwise, from top left).

        Returns
        -------
        list of imgaug.augmentables.kps.Keypoint
            Corners of the bounding box as keypoints.

        """
        # TODO get rid of this deferred import
        from imgaug.augmentables.kps import Keypoint

        return [
            Keypoint(x=self.x1, y=self.y1),
            Keypoint(x=self.x2, y=self.y1),
            Keypoint(x=self.x2, y=self.y2),
            Keypoint(x=self.x1, y=self.y2)
        ]

    def coords_almost_equals(self, other, max_distance=1e-4):
        """Estimate if this and another BB have almost identical coordinates.

        Parameters
        ----------
        other : imgaug.augmentables.bbs.BoundingBox or iterable
            The other bounding box with which to compare this one.
            If this is an ``iterable``, it is assumed to represent the top-left
            and bottom-right coordinates of that bounding box, given as e.g.
            an ``(2,2)`` ndarray or an ``(4,)`` ndarray or as a similar list.

        max_distance : number, optional
            The maximum euclidean distance between a corner on one bounding
            box and the closest corner on the other bounding box. If the
            distance is exceeded for any such pair, the two BBs are not
            viewed as equal.

        Returns
        -------
        bool
            Whether the two bounding boxes have almost identical corner
            coordinates.

        """
        if ia.is_np_array(other):
            # we use flat here in case other is (N,2) instead of (4,)
            coords_b = other.flat
        elif ia.is_iterable(other):
            coords_b = list(ia.flatten(other))
        else:
            assert isinstance(other, BoundingBox), (
                "Expected 'other' to be an iterable containing two "
                "(x,y)-coordinate pairs or a BoundingBox. "
                "Got type %s." % (type(other),))
            coords_b = other.coords.flat

        coords_a = self.coords

        return np.allclose(coords_a.flat, coords_b, atol=max_distance, rtol=0)

    def almost_equals(self, other, max_distance=1e-4):
        """Compare this and another BB's label and coordinates.

        This is the same as
        :func:`imgaug.augmentables.bbs.BoundingBox.coords_almost_equals` but
        additionally compares the labels.

        Parameters
        ----------
        other : imgaug.augmentables.bbs.BoundingBox or iterable
            The other object to compare against. Expected to be a
            ``BoundingBox``.

        max_distance : number, optional
            See
            :func:`imgaug.augmentables.bbs.BoundingBox.coords_almost_equals`.

        Returns
        -------
        bool
            ``True`` if the coordinates are almost equal and additionally
            the labels are equal. Otherwise ``False``.

        """
        if self.label != other.label:
            return False
        return self.coords_almost_equals(other, max_distance=max_distance)

    @classmethod
    def from_point_soup(cls, xy):
        """Convert a ``(2P,) or (P,2) ndarray`` to a BB instance.

        This is the inverse of
        :func:`imgaug.BoundingBoxesOnImage.to_xyxy_array`.

        Parameters
        ----------
        xy : (2P,) ndarray or (P, 2) array or iterable of number or iterable of iterable of number
            Array containing ``P`` points in xy-form denoting a soup of
            points around which to place a bounding box.
            The array should usually be of dtype ``float32``.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBox
            Bounding box around the points.

        """
        xy = np.array(xy, dtype=np.float32)

        assert len(xy) > 0, (
            "Expected to get at least one point to place a bounding box "
            "around, got shape %s." % (xy.shape,))

        assert xy.ndim == 1 or (xy.ndim == 2 and xy.shape[-1] == 2), (
            "Expected input array of shape (P,) or (P, 2), "
            "got shape %s." % (xy.shape,))

        if xy.ndim == 1:
            xy = xy.reshape((-1, 2))

        x1, y1 = np.min(xy, axis=0)
        x2, y2 = np.max(xy, axis=0)

        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    def copy(self, x1=None, y1=None, x2=None, y2=None, label=None):
        """Create a shallow copy of this BoundingBox instance.

        Parameters
        ----------
        x1 : None or number
            If not ``None``, then the ``x1`` coordinate of the copied object
            will be set to this value.

        y1 : None or number
            If not ``None``, then the ``y1`` coordinate of the copied object
            will be set to this value.

        x2 : None or number
            If not ``None``, then the ``x2`` coordinate of the copied object
            will be set to this value.

        y2 : None or number
            If not ``None``, then the ``y2`` coordinate of the copied object
            will be set to this value.

        label : None or string
            If not ``None``, then the ``label`` of the copied object
            will be set to this value.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBox
            Shallow copy.

        """
        return BoundingBox(
            x1=self.x1 if x1 is None else x1,
            x2=self.x2 if x2 is None else x2,
            y1=self.y1 if y1 is None else y1,
            y2=self.y2 if y2 is None else y2,
            label=copy.deepcopy(self.label) if label is None else label
        )

    def deepcopy(self, x1=None, y1=None, x2=None, y2=None, label=None):
        """
        Create a deep copy of the BoundingBox object.

        Parameters
        ----------
        x1 : None or number
            If not ``None``, then the ``x1`` coordinate of the copied object
            will be set to this value.

        y1 : None or number
            If not ``None``, then the ``y1`` coordinate of the copied object
            will be set to this value.

        x2 : None or number
            If not ``None``, then the ``x2`` coordinate of the copied object
            will be set to this value.

        y2 : None or number
            If not ``None``, then the ``y2`` coordinate of the copied object
            will be set to this value.

        label : None or string
            If not ``None``, then the ``label`` of the copied object
            will be set to this value.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBox
            Deep copy.

        """
        # TODO write specific copy routine with deepcopy for label and remove
        #      the deepcopy from copy()
        return self.copy(x1=x1, y1=y1, x2=x2, y2=y2, label=label)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "BoundingBox(x1=%.4f, y1=%.4f, x2=%.4f, y2=%.4f, label=%s)" % (
            self.x1, self.y1, self.x2, self.y2, self.label)


class BoundingBoxesOnImage(object):
    """Container for the list of all bounding boxes on a single image.

    Parameters
    ----------
    bounding_boxes : list of imgaug.augmentables.bbs.BoundingBox
        List of bounding boxes on the image.

    shape : tuple of int or ndarray
        The shape of the image on which the objects are placed.
        Either an image with shape ``(H,W,[C])`` or a ``tuple`` denoting
        such an image shape.

    Examples
    --------
    >>> import numpy as np
    >>> from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
    >>>
    >>> image = np.zeros((100, 100))
    >>> bbs = [
    >>>     BoundingBox(x1=10, y1=20, x2=20, y2=30),
    >>>     BoundingBox(x1=25, y1=50, x2=30, y2=70)
    >>> ]
    >>> bbs_oi = BoundingBoxesOnImage(bbs, shape=image.shape)

    """
    def __init__(self, bounding_boxes, shape):
        self.bounding_boxes = bounding_boxes
        self.shape = normalize_shape(shape)

    @property
    def items(self):
        """Get the bounding boxes in this container.

        Returns
        -------
        list of BoundingBox
            Bounding boxes within this container.

        """
        return self.bounding_boxes

    # TODO remove this? here it is image height, but in BoundingBox it is
    #      bounding box height
    @property
    def height(self):
        """Get the height of the image on which the bounding boxes fall.

        Returns
        -------
        int
            Image height.

        """
        return self.shape[0]

    # TODO remove this? here it is image width, but in BoundingBox it is
    #      bounding box width
    @property
    def width(self):
        """Get the width of the image on which the bounding boxes fall.

        Returns
        -------
        int
            Image width.

        """
        return self.shape[1]

    @property
    def empty(self):
        """Determine whether this instance contains zero bounding boxes.

        Returns
        -------
        bool
            True if this object contains zero bounding boxes.

        """
        return len(self.bounding_boxes) == 0

    def on(self, image):
        """Project bounding boxes from one image (shape) to a another one.

        Parameters
        ----------
        image : ndarray or tuple of int
            New image onto which the bounding boxes are to be projected.
            May also simply be that new image's shape tuple.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBoxesOnImage
            Object containing the same bounding boxes after projection to
            the new image shape.

        """
        shape = normalize_shape(image)
        if shape[0:2] == self.shape[0:2]:
            return self.deepcopy()
        bounding_boxes = [bb.project(self.shape, shape)
                          for bb in self.bounding_boxes]
        return BoundingBoxesOnImage(bounding_boxes, shape)

    @classmethod
    def from_xyxy_array(cls, xyxy, shape):
        """Convert an ``(N, 4) or (N, 2, 2) ndarray`` to a BBsOI instance.

        This is the inverse of
        :func:`imgaug.BoundingBoxesOnImage.to_xyxy_array`.

        Parameters
        ----------
        xyxy : (N, 4) ndarray or (N, 2, 2) array
            Array containing the corner coordinates of ``N`` bounding boxes.
            Each bounding box is represented by its top-left and bottom-right
            coordinates.
            The array should usually be of dtype ``float32``.

        shape : tuple of int
            Shape of the image on which the bounding boxes are placed.
            Should usually be ``(H, W, C)`` or ``(H, W)``.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBoxesOnImage
            Object containing a list of :class:`BoundingBox` instances
            derived from the provided corner coordinates.

        """
        xyxy = np.array(xyxy, dtype=np.float32)

        # note that np.array([]) is (0,), not (0, 2)
        if xyxy.shape[0] == 0:
            return BoundingBoxesOnImage([], shape)

        assert (
            (xyxy.ndim == 2 and xyxy.shape[-1] == 4)
            or (xyxy.ndim == 3 and xyxy.shape[1:3] == (2, 2))), (
            "Expected input array of shape (N, 4) or (N, 2, 2), "
            "got shape %s." % (xyxy.shape,))

        xyxy = xyxy.reshape((-1, 2, 2))
        boxes = [BoundingBox.from_point_soup(row) for row in xyxy]

        return cls(boxes, shape)

    @classmethod
    def from_point_soups(cls, xy, shape):
        """Convert an ``(N, 2P) or (N, P, 2) ndarray`` to a BBsOI instance.

        Parameters
        ----------
        xy : (N, 2P) ndarray or (N, P, 2) array or iterable of iterable of number or iterable of iterable of iterable of number
            Array containing the corner coordinates of ``N`` bounding boxes.
            Each bounding box is represented by a soup of ``P`` points.
            If ``(N, P)`` then the second axis is expected to be in
            xy-form (e.g. ``x1``, ``y1``, ``x2``, ``y2``, ...).
            The final bounding box coordinates will be derived using ``min``
            and ``max`` operations on the xy-values.
            The array should usually be of dtype ``float32``.

        shape : tuple of int
            Shape of the image on which the bounding boxes are placed.
            Should usually be ``(H, W, C)`` or ``(H, W)``.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBoxesOnImage
            Object containing a list of :class:`BoundingBox` instances
            derived from the provided point soups.

        """
        xy = np.array(xy, dtype=np.float32)

        # from_xy_array() already checks the ndim/shape, so we don't have to
        # do it here
        boxes = [BoundingBox.from_point_soup(row) for row in xy]

        return cls(boxes, shape)

    def to_xyxy_array(self, dtype=np.float32):
        """Convert the ``BoundingBoxesOnImage`` object to an ``(N,4) ndarray``.

        This is the inverse of
        :func:`imgaug.BoundingBoxesOnImage.from_xyxy_array`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            Desired output datatype of the ndarray.

        Returns
        -------
        ndarray
            ``(N,4) ndarray``, where ``N`` denotes the number of bounding
            boxes and ``4`` denotes the top-left and bottom-right bounding
            box corner coordinates in form ``(x1, y1, x2, y2)``.

        """
        xyxy_array = np.zeros((len(self.bounding_boxes), 4), dtype=np.float32)

        for i, box in enumerate(self.bounding_boxes):
            xyxy_array[i] = [box.x1, box.y1, box.x2, box.y2]

        return xyxy_array.astype(dtype)

    def to_xy_array(self):
        """Convert the ``BoundingBoxesOnImage`` object to an ``(N,2) ndarray``.

        Returns
        -------
        ndarray
            ``(2*B,2) ndarray`` of xy-coordinates, where ``B`` denotes the
            number of bounding boxes.

        """
        return self.to_xyxy_array().reshape((-1, 2))

    def fill_from_xyxy_array_(self, xyxy):
        """Modify the BB coordinates of this instance in-place.

        .. note ::

            This currently expects exactly one entry in `xyxy` per bounding
            in this instance. (I.e. two corner coordinates per instance.)
            Otherwise, an ``AssertionError`` will be raised.

        .. note ::

            This method will automatically flip x-coordinates if ``x1>x2``
            for a bounding box. (Analogous for y-coordinates.)

        Parameters
        ----------
        xyxy : (N, 4) ndarray or iterable of iterable of number
            Coordinates of ``N`` bounding boxes on an image, given as
            a ``(N,4)`` array of two corner xy-coordinates per bounding box.
            ``N`` must match the number of bounding boxes in this instance.

        Returns
        -------
        BoundingBoxesOnImage
            This instance itself, with updated bounding box coordinates.
            Note that the instance was modified in-place.

        """
        xyxy = np.array(xyxy, dtype=np.float32)

        # note that np.array([]) is (0,), not (0, 4)
        assert xyxy.shape[0] == 0 or (xyxy.ndim == 2 and xyxy.shape[-1] == 4), (
            "Expected input array to have shape (N,4), "
            "got shape %s." % (xyxy.shape,))

        assert len(xyxy) == len(self.bounding_boxes), (
            "Expected to receive an array with as many rows there are "
            "bounding boxes in this instance. Got %d rows, expected %d." % (
                len(xyxy), len(self.bounding_boxes)))

        for bb, (x1, y1, x2, y2) in zip(self.bounding_boxes, xyxy):
            bb.x1 = min([x1, x2])
            bb.y1 = min([y1, y2])
            bb.x2 = max([x1, x2])
            bb.y2 = max([y1, y2])

        return self

    def fill_from_xy_array_(self, xy):
        """Modify the BB coordinates of this instance in-place.

        See
        :func:`imgaug.augmentables.bbs.BoundingBoxesOnImage.fill_from_xyxy_array_`.

        Parameters
        ----------
        xy : (2*B, 2) ndarray or iterable of iterable of number
            Coordinates of ``B`` bounding boxes on an image, given as
            a ``(2*B,2)`` array of two corner xy-coordinates per bounding box.
            ``B`` must match the number of bounding boxes in this instance.

        Returns
        -------
        BoundingBoxesOnImage
            This instance itself, with updated bounding box coordinates.
            Note that the instance was modified in-place.

        """
        xy = np.array(xy, dtype=np.float32)
        return self.fill_from_xyxy_array_(xy.reshape((-1, 4)))

    def draw_on_image(self, image, color=(0, 255, 0), alpha=1.0, size=1,
                      copy=True, raise_if_out_of_image=False, thickness=None):
        """Draw all bounding boxes onto a given image.

        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the bounding boxes.
            This image should usually have the same shape as set in
            ``BoundingBoxesOnImage.shape``.

        color : int or list of int or tuple of int or (3,) ndarray, optional
            The RGB color of all bounding boxes.
            If a single ``int`` ``C``, then that is equivalent to ``(C,C,C)``.

        alpha : float, optional
            Alpha/transparency of the bounding box.

        size : int, optional
            Thickness in pixels.

        copy : bool, optional
            Whether to copy the image before drawing the bounding boxes.

        raise_if_out_of_image : bool, optional
            Whether to raise an exception if any bounding box is outside of the
            image.

        thickness : None or int, optional
            Deprecated.

        Returns
        -------
        (H,W,3) ndarray
            Image with drawn bounding boxes.

        """
        image = np.copy(image) if copy else image

        for bb in self.bounding_boxes:
            image = bb.draw_on_image(
                image,
                color=color,
                alpha=alpha,
                size=size,
                copy=False,
                raise_if_out_of_image=raise_if_out_of_image,
                thickness=thickness
            )

        return image

    def remove_out_of_image(self, fully=True, partly=False):
        """Remove all BBs that are fully/partially outside of the image.

        Parameters
        ----------
        fully : bool, optional
            Whether to remove bounding boxes that are fully outside of the
            image.

        partly : bool, optional
            Whether to remove bounding boxes that are partially outside of
            the image.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBoxesOnImage
            Reduced set of bounding boxes, with those that were
            fully/partially outside of the image being removed.

        """
        bbs_clean = [
            bb
            for bb
            in self.bounding_boxes
            if not bb.is_out_of_image(self.shape, fully=fully, partly=partly)]
        return BoundingBoxesOnImage(bbs_clean, shape=self.shape)

    @ia.deprecated(alt_func="BoundingBoxesOnImage.clip_out_of_image()",
                   comment="clip_out_of_image() has the exactly same "
                           "interface.")
    def cut_out_of_image(self):
        return self.clip_out_of_image()

    def clip_out_of_image(self):
        """Clip off all parts from all BBs that are outside of the image.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBoxesOnImage
            Bounding boxes, clipped to fall within the image dimensions.

        """
        bbs_cut = [
            bb.clip_out_of_image(self.shape)
            for bb
            in self.bounding_boxes
            if bb.is_partly_within_image(self.shape)]
        return BoundingBoxesOnImage(bbs_cut, shape=self.shape)

    def shift(self, top=None, right=None, bottom=None, left=None):
        """Move all all BBs along the x/y-axis.

        Parameters
        ----------
        top : None or int, optional
            Amount of pixels by which to shift all objects *from* the
            top (towards the bottom).

        right : None or int, optional
            Amount of pixels by which to shift all objects *from* the
            right (towads the left).

        bottom : None or int, optional
            Amount of pixels by which to shift all objects *from* the
            bottom (towards the top).

        left : None or int, optional
            Amount of pixels by which to shift all objects *from* the
            left (towards the right).

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBoxesOnImage
            Shifted bounding boxes.

        """
        bbs_new = [
            bb.shift(top=top, right=right, bottom=bottom, left=left)
            for bb
            in self.bounding_boxes]
        return BoundingBoxesOnImage(bbs_new, shape=self.shape)

    def to_keypoints_on_image(self):
        """Convert the bounding boxes to one ``KeypointsOnImage`` instance.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage
            A keypoints instance containing ``N*4`` coordinates for ``N``
            bounding boxes. Order matches the order in ``bounding_boxes``.

        """
        from .kps import KeypointsOnImage

        # This currently uses 4 points instead of 2 points as the method
        # is primarily used during augmentation and 4 points are overall
        # the better choice there.
        arr = np.zeros((len(self.bounding_boxes), 2*4), dtype=np.float32)

        for i, box in enumerate(self.bounding_boxes):
            arr[i] = [
                box.x1, box.y1,
                box.x2, box.y1,
                box.x2, box.y2,
                box.x1, box.y2
            ]

        return KeypointsOnImage.from_xy_array(
            arr.reshape((-1, 2)),
            shape=self.shape
        )

    def invert_to_keypoints_on_image_(self, kpsoi):
        """Invert the output of ``to_keypoints_on_image()`` in-place.

        This function writes in-place into this ``BoundingBoxesOnImage``
        instance.

        Parameters
        ----------
        kpsoi : imgaug.augmentables.kps.KeypointsOnImages
            Keypoints to convert back to bounding boxes, i.e. the outputs
            of ``to_keypoints_on_image()``.

        Returns
        -------
        BoundingBoxesOnImage
            Bounding boxes container with updated coordinates.
            Note that the instance is also updated in-place.

        """
        assert len(kpsoi.keypoints) == len(self.bounding_boxes) * 4, (
            "Expected %d coordinates, got %d." % (
                len(self.bounding_boxes) * 2, len(kpsoi.keypoints)))
        for i, bb in enumerate(self.bounding_boxes):
            xx = [kpsoi.keypoints[4*i+0].x, kpsoi.keypoints[4*i+1].x,
                  kpsoi.keypoints[4*i+2].x, kpsoi.keypoints[4*i+3].x]
            yy = [kpsoi.keypoints[4*i+0].y, kpsoi.keypoints[4*i+1].y,
                  kpsoi.keypoints[4*i+2].y, kpsoi.keypoints[4*i+3].y]
            bb.x1 = min(xx)
            bb.y1 = min(yy)
            bb.x2 = max(xx)
            bb.y2 = max(yy)
        self.shape = kpsoi.shape
        return self

    def copy(self):
        """Create a shallow copy of the ``BoundingBoxesOnImage`` instance.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBoxesOnImage
            Shallow copy.

        """
        return copy.copy(self)

    def deepcopy(self):
        """Create a deep copy of the ``BoundingBoxesOnImage`` object.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBoxesOnImage
            Deep copy.

        """
        # Manual copy is far faster than deepcopy for BoundingBoxesOnImage,
        # so use manual copy here too
        bbs = [bb.deepcopy() for bb in self.bounding_boxes]
        return BoundingBoxesOnImage(bbs, tuple(self.shape))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            "BoundingBoxesOnImage(%s, shape=%s)"
            % (str(self.bounding_boxes), self.shape))
