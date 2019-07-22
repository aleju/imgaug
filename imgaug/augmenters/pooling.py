"""
Augmenters that apply pooling operations to images.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([
        iaa.AveragePooling((1, 3))
    ])

List of augmenters:

    * AveragePooling
    * MaxPooling
    * MinPooling
    * MedianPooling

"""
from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod

import numpy as np
import six

from . import meta
import imgaug as ia
from .. import parameters as iap


@six.add_metaclass(ABCMeta)
class _AbstractPoolingBase(meta.Augmenter):
    # TODO add floats as ksize denoting fractions of image sizes
    #      (note possible overlap with fractional kernel sizes here)
    def __init__(self, kernel_size, keep_size=True,
                 name=None, deterministic=False, random_state=None):
        super(_AbstractPoolingBase, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        self.kernel_size = iap.handle_discrete_kernel_size_param(
            kernel_size,
            "kernel_size",
            value_range=(0, None),
            allow_floats=False)
        self.keep_size = keep_size

    @abstractmethod
    def _pool_image(self, image, kernel_size_h, kernel_size_w):
        """Apply pooling method with given kernel height/width to an image."""

    def _draw_samples(self, augmentables, random_state):
        nb_images = len(augmentables)
        rss = ia.derive_random_states(random_state, 2)
        mode = "single" if self.kernel_size[1] is None else "two"
        kernel_sizes_h = self.kernel_size[0].draw_samples(
            (nb_images,),
            random_state=rss[0])
        if mode == "single":
            kernel_sizes_w = kernel_sizes_h
        else:
            kernel_sizes_w = self.kernel_size[1].draw_samples(
                (nb_images,), random_state=rss[1])
        return kernel_sizes_h, kernel_sizes_w

    def _augment_images(self, images, random_state, parents, hooks):
        if not self.keep_size:
            images = list(images)

        kernel_sizes_h, kernel_sizes_w = self._draw_samples(
            images, random_state)

        gen = enumerate(zip(images, kernel_sizes_h, kernel_sizes_w))
        for i, (image, ksize_h, ksize_w) in gen:
            if ksize_h >= 2 or ksize_w >= 2:
                image_pooled = self._pool_image(
                    image,
                    max(ksize_h, 1), max(ksize_w, 1)
                )
                if self.keep_size:
                    image_pooled = ia.imresize_single_image(
                        image_pooled, image.shape[0:2])
                images[i] = image_pooled

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        # For some reason pylint raises a warning here, which it doesn't seem
        # to do for other classes that also implement this method with self use.
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        # For some reason pylint raises a warning here, which it doesn't seem
        # to do for other classes that also implement this method with self use.
        return keypoints_on_images

    def get_parameters(self):
        return [self.kernel_size, self.keep_size]


# TODO rename kernel size parameters in all augmenters to kernel_size
# TODO add per_channel
# TODO add upscaling interpolation mode?
# TODO add dtype support
class AveragePooling(_AbstractPoolingBase):
    """
    Apply average pooling to images.

    This pools images with kernel sizes ``H x W`` by averaging the pixel
    values within these windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    This augmenter does not affect heatmaps, segmentation maps or
    coordinates-based augmentables (e.g. keypoints, bounding boxes, ...).

    Note that this augmenter is very similar to ``AverageBlur``.
    ``AverageBlur`` applies averaging within windows of given kernel size
    *without* striding, while ``AveragePooling`` applies striding corresponding
    to the kernel size, with optional upscaling afterwards. The upscaling
    is configured to create "pixelated"/"blocky" images by default.

    dtype support::

        See :func:`imgaug.imgaug.avg_pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int \
                  or imgaug.parameters.StochasticParameter \
                  or tuple of tuple of int or tuple of list of int \
                  or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = AveragePooling(2)

    Creates an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> import imgaug.augmenters as iaa
    >>> aug = AveragePooling(2, keep_size=False)

    Creates an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> import imgaug.augmenters as iaa
    >>> aug = AveragePooling([2, 8])

    Creates an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> import imgaug.augmenters as iaa
    >>> aug = AveragePooling((1, 7))

    Creates an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> import imgaug.augmenters as iaa
    >>> aug = AveragePooling(((1, 7), (1, 7)))

    Creates an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    """

    # TODO add floats as ksize denoting fractions of image sizes
    #      (note possible overlap with fractional kernel sizes here)
    def __init__(self, kernel_size, keep_size=True,
                 name=None, deterministic=False, random_state=None):
        super(AveragePooling, self).__init__(
            kernel_size=kernel_size, keep_size=keep_size,
            name=name, deterministic=deterministic, random_state=random_state)

    def _pool_image(self, image, kernel_size_h, kernel_size_w):
        return ia.avg_pool(
            image,
            (max(kernel_size_h, 1), max(kernel_size_w, 1))
        )


class MaxPooling(_AbstractPoolingBase):
    """
    Apply max pooling to images.

    This pools images with kernel sizes ``H x W`` by taking the maximum
    pixel value over windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    The maximum within each pixel window is always taken channelwise.

    This augmenter does not affect heatmaps, segmentation maps or
    coordinates-based augmentables (e.g. keypoints, bounding boxes, ...).

    dtype support::

        See :func:`imgaug.imgaug.max_pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int \
                  or imgaug.parameters.StochasticParameter \
                  or tuple of tuple of int or tuple of list of int \
                  or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = MaxPooling(2)

    Creates an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> import imgaug.augmenters as iaa
    >>> aug = MaxPooling(2, keep_size=False)

    Creates an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> import imgaug.augmenters as iaa
    >>> aug = MaxPooling([2, 8])

    Creates an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> import imgaug.augmenters as iaa
    >>> aug = MaxPooling((1, 7))

    Creates an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> import imgaug.augmenters as iaa
    >>> aug = MaxPooling(((1, 7), (1, 7)))

    Creates an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    """

    # TODO add floats as ksize denoting fractions of image sizes
    #      (note possible overlap with fractional kernel sizes here)
    def __init__(self, kernel_size, keep_size=True,
                 name=None, deterministic=False, random_state=None):
        super(MaxPooling, self).__init__(
            kernel_size=kernel_size, keep_size=keep_size,
            name=name, deterministic=deterministic, random_state=random_state)

    def _pool_image(self, image, kernel_size_h, kernel_size_w):
        # TODO extend max_pool to support pad_mode and set it here
        #      to reflection padding
        return ia.max_pool(
            image,
            (max(kernel_size_h, 1), max(kernel_size_w, 1))
        )


class MinPooling(_AbstractPoolingBase):
    """
    Apply minimum pooling to images.

    This pools images with kernel sizes ``H x W`` by taking the minimum
    pixel value over windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    The minimum within each pixel window is always taken channelwise.

    This augmenter does not affect heatmaps, segmentation maps or
    coordinates-based augmentables (e.g. keypoints, bounding boxes, ...).

    dtype support::

        See :func:`imgaug.imgaug.pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int \
                  or imgaug.parameters.StochasticParameter \
                  or tuple of tuple of int or tuple of list of int \
                  or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = MinPooling(2)

    Creates an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> import imgaug.augmenters as iaa
    >>> aug = MinPooling(2, keep_size=False)

    Creates an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> import imgaug.augmenters as iaa
    >>> aug = MinPooling([2, 8])

    Creates an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> import imgaug.augmenters as iaa
    >>> aug = MinPooling((1, 7))

    Creates an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> import imgaug.augmenters as iaa
    >>> aug = MinPooling(((1, 7), (1, 7)))

    Creates an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    """

    # TODO add floats as ksize denoting fractions of image sizes
    #      (note possible overlap with fractional kernel sizes here)
    def __init__(self, kernel_size, keep_size=True,
                 name=None, deterministic=False, random_state=None):
        super(MinPooling, self).__init__(
            kernel_size=kernel_size, keep_size=keep_size,
            name=name, deterministic=deterministic, random_state=random_state)

    def _pool_image(self, image, kernel_size_h, kernel_size_w):
        # TODO extend pool to support pad_mode and set it here
        #      to reflection padding
        return ia.min_pool(
            image,
            (max(kernel_size_h, 1), max(kernel_size_w, 1))
        )


class MedianPooling(_AbstractPoolingBase):
    """
    Apply median pooling to images.

    This pools images with kernel sizes ``H x W`` by taking the median
    pixel value over windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    The median within each pixel window is always taken channelwise.

    This augmenter does not affect heatmaps, segmentation maps or
    coordinates-based augmentables (e.g. keypoints, bounding boxes, ...).

    dtype support::

        See :func:`imgaug.imgaug.pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int \
                  or imgaug.parameters.StochasticParameter \
                  or tuple of tuple of int or tuple of list of int \
                  or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = MedianPooling(2)

    Creates an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> import imgaug.augmenters as iaa
    >>> aug = MedianPooling(2, keep_size=False)

    Creates an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> import imgaug.augmenters as iaa
    >>> aug = MedianPooling([2, 8])

    Creates an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> import imgaug.augmenters as iaa
    >>> aug = MedianPooling((1, 7))

    Creates an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> import imgaug.augmenters as iaa
    >>> aug = MedianPooling(((1, 7), (1, 7)))

    Creates an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    """

    # TODO add floats as ksize denoting fractions of image sizes
    #      (note possible overlap with fractional kernel sizes here)
    def __init__(self, kernel_size, keep_size=True,
                 name=None, deterministic=False, random_state=None):
        super(MedianPooling, self).__init__(
            kernel_size=kernel_size, keep_size=keep_size,
            name=name, deterministic=deterministic, random_state=random_state)

    def _pool_image(self, image, kernel_size_h, kernel_size_w):
        # TODO extend pool to support pad_mode and set it here
        #      to reflection padding
        return ia.median_pool(
            image,
            (max(kernel_size_h, 1), max(kernel_size_w, 1))
        )
