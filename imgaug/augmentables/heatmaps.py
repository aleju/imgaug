from __future__ import print_function, division, absolute_import

import numpy as np
import six.moves as sm

from .. import imgaug as ia


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
        ia.do_assert(ia.is_np_array(arr), "Expected numpy array as heatmap input array, got type %s" % (type(arr),))
        # TODO maybe allow 0-sized heatmaps? in that case the min() and max() must be adjusted
        ia.do_assert(arr.shape[0] > 0 and arr.shape[1] > 0,
                     "Expected numpy array as heatmap with height and width greater than 0, got shape %s." % (
                         arr.shape,))
        ia.do_assert(arr.dtype.type in [np.float32],
                     "Heatmap input array expected to be of dtype float32, got dtype %s." % (arr.dtype,))
        ia.do_assert(arr.ndim in [2, 3], "Heatmap input array must be 2d or 3d, got shape %s." % (arr.shape,))
        ia.do_assert(len(shape) in [2, 3],
                     "Argument 'shape' in HeatmapsOnImage expected to be 2d or 3d, got shape %s." % (shape,))
        ia.do_assert(min_value < max_value)
        if np.min(arr.flat[0:50]) < min_value - np.finfo(arr.dtype).eps \
                or np.max(arr.flat[0:50]) > max_value + np.finfo(arr.dtype).eps:
            import warnings
            warnings.warn(
                ("Value range of heatmap was chosen to be (%.8f, %.8f), but "
                 "found actual min/max of (%.8f, %.8f). Array will be "
                 "clipped to chosen value range.") % (
                    min_value, max_value, np.min(arr), np.max(arr)))
            arr = np.clip(arr, min_value, max_value)

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

        # don't allow arrays here as an alternative to tuples as input
        # as allowing arrays introduces risk to mix up 'arr' and 'shape' args
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
                heatmap_c_rs = ia.imresize_single_image(heatmap_c, size, interpolation="nearest")
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
        ia.do_assert(image.ndim == 3)
        ia.do_assert(image.shape[2] == 3)
        ia.do_assert(image.dtype.type == np.uint8)

        ia.do_assert(0 - 1e-8 <= alpha <= 1.0 + 1e-8)
        ia.do_assert(resize in ["heatmaps", "image"])

        if resize == "image":
            image = ia.imresize_single_image(image, self.arr_0to1.shape[0:2], interpolation="cubic")

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
        arr_0to1_padded = ia.pad(self.arr_0to1, top=top, right=right, bottom=bottom, left=left, mode=mode, cval=cval)
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
        arr_0to1_padded, pad_amounts = ia.pad_to_aspect_ratio(self.arr_0to1, aspect_ratio=aspect_ratio, mode=mode,
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
        arr_0to1_reduced = ia.avg_pool(self.arr_0to1, block_size, cval=0.0)
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
        arr_0to1_reduced = ia.max_pool(self.arr_0to1, block_size)
        return HeatmapsOnImage.from_0to1(arr_0to1_reduced, shape=self.shape, min_value=self.min_value,
                                         max_value=self.max_value)

    @ia.deprecated(alt_func="HeatmapsOnImage.resize()",
                   comment="resize() has the exactly same interface.")
    def scale(self, *args, **kwargs):
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
        arr_0to1_resized = ia.imresize_single_image(self.arr_0to1, sizes, interpolation=interpolation)

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
        ia.do_assert(ia.is_np_array(arr))

        if isinstance(source, HeatmapsOnImage):
            source = (source.min_value, source.max_value)
        else:
            ia.do_assert(isinstance(source, tuple))
            ia.do_assert(len(source) == 2)
            ia.do_assert(source[0] < source[1])

        if isinstance(target, HeatmapsOnImage):
            target = (target.min_value, target.max_value)
        else:
            ia.do_assert(isinstance(target, tuple))
            ia.do_assert(len(target) == 2)
            ia.do_assert(target[0] < target[1])

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
