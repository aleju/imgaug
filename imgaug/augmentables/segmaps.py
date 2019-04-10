from __future__ import print_function, division, absolute_import

import warnings

import numpy as np
import six.moves as sm

from .. import imgaug as ia


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
        class index plus 1. This may be None if the input array is of type bool or float. The number
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
        ia.do_assert(ia.is_np_array(arr), "Expected to get numpy array, got %s." % (type(arr),))

        if arr.dtype.name == "bool":
            ia.do_assert(arr.ndim in [2, 3])
            self.input_was = ("bool", arr.ndim)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            arr = arr.astype(np.float32)
        elif arr.dtype.kind in ["i", "u"]:
            ia.do_assert(arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1))
            ia.do_assert(nb_classes is not None)
            ia.do_assert(nb_classes > 0)
            ia.do_assert(np.min(arr.flat[0:100]) >= 0)
            ia.do_assert(np.max(arr.flat[0:100]) < nb_classes)
            self.input_was = ("int", arr.dtype.type, arr.ndim)
            if arr.ndim == 3:
                arr = arr[..., 0]
            # TODO improve efficiency here by building only sub-heatmaps for classes actually
            # present in the image. This would also get rid of nb_classes.
            arr = np.eye(nb_classes)[arr]  # from class indices to one hot
            arr = arr.astype(np.float32)
        elif arr.dtype.kind == "f":
            ia.do_assert(arr.ndim == 3)
            self.input_was = ("float", arr.dtype.type, arr.ndim)
            arr = arr.astype(np.float32)
        else:
            raise Exception(("Input was expected to be an ndarray any bool, int, uint or float dtype. "
                             + "Got dtype %s.") % (arr.dtype.name,))
        ia.do_assert(arr.ndim == 3)
        ia.do_assert(arr.dtype.name == "float32")
        self.arr = arr

        # don't allow arrays here as an alternative to tuples as input
        # as allowing arrays introduces risk to mix up 'arr' and 'shape' args
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
            ia.do_assert(background_class_id is None,
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
        else:  # integer mask was provided
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
        ia.do_assert(nb_classes <= len(colors),
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
            segmap_drawn = ia.imresize_single_image(segmap_drawn, size, interpolation="nearest")
            if foreground_mask is not None:
                foreground_mask = ia.imresize_single_image(
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
        ia.do_assert(image.ndim == 3)
        ia.do_assert(image.shape[2] == 3)
        ia.do_assert(image.dtype.type == np.uint8)

        ia.do_assert(0 - 1e-8 <= alpha <= 1.0 + 1e-8)
        ia.do_assert(resize in ["segmentation_map", "image"])

        if resize == "image":
            image = ia.imresize_single_image(image, self.arr.shape[0:2], interpolation="cubic")

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
        arr_padded = ia.pad(self.arr, top=top, right=right, bottom=bottom, left=left, mode=mode, cval=cval)
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
        arr_padded, pad_amounts = ia.pad_to_aspect_ratio(self.arr, aspect_ratio=aspect_ratio, mode=mode, cval=cval,
                                                         return_pad_amounts=True)
        segmap = SegmentationMapOnImage(arr_padded, shape=self.shape)
        segmap.input_was = self.input_was
        if return_pad_amounts:
            return segmap, pad_amounts
        else:
            return segmap

    @ia.deprecated(alt_func="SegmentationMapOnImage.resize()",
                   comment="resize() has the exactly same interface.")
    def scale(self, *args, **kwargs):
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
        arr_resized = ia.imresize_single_image(self.arr, sizes, interpolation=interpolation)

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
        # TODO get rid of this deferred import
        from imgaug.augmentables.heatmaps import HeatmapsOnImage

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
            ia.do_assert(nb_classes is not None)
            ia.do_assert(min(class_indices) >= 0)
            ia.do_assert(max(class_indices) < nb_classes)
            ia.do_assert(len(class_indices) == heatmaps.arr_0to1.shape[2])
            arr_0to1 = heatmaps.arr_0to1
            arr_0to1_full = np.zeros((arr_0to1.shape[0], arr_0to1.shape[1], nb_classes), dtype=np.float32)
            for heatmap_channel, mapped_channel in enumerate(class_indices):
                arr_0to1_full[:, :, mapped_channel] = arr_0to1[:, :, heatmap_channel]
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
