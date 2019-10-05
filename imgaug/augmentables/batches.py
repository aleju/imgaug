from __future__ import print_function, division, absolute_import

import copy

import numpy as np

from .. import imgaug as ia
from . import normalization as nlib
from . import utils as utils

DEFAULT = "DEFAULT"

_AUGMENTABLE_NAMES = [
    "images", "heatmaps", "segmentation_maps", "keypoints",
    "bounding_boxes", "polygons", "line_strings"]


def _get_augmentables_names(batch, postfix):
    return [name
            for name, value, attr_name
            in _get_augmentables(batch, postfix)]


def _get_augmentables(batch, postfix):
    result = []
    for name in _AUGMENTABLE_NAMES:
        attr_name = name + postfix
        value = getattr(batch, name + postfix)
        # Every data item is either an array or a list. If there are no
        # items in the array/list, there are also no shapes to change
        # as shape-changes are imagewise. Hence, we can afford to check
        # len() here.
        if value is not None and len(value) > 0:
            result.append((name, value, attr_name))
    return result


# TODO also support (H,W,C) for heatmaps of len(images) == 1
# TODO also support (H,W) for segmaps of len(images) == 1
class UnnormalizedBatch(object):
    """
    Class for batches of unnormalized data before and after augmentation.

    Parameters
    ----------
    images : None or (N,H,W,C) ndarray or (N,H,W) ndarray or iterable of (H,W,C) ndarray or iterable of (H,W) ndarray
        The images to augment.

    heatmaps : None or (N,H,W,C) ndarray or imgaug.augmentables.heatmaps.HeatmapsOnImage or iterable of (H,W,C) ndarray or iterable of imgaug.augmentables.heatmaps.HeatmapsOnImage
        The heatmaps to augment.
        If anything else than ``HeatmapsOnImage``, then the number of heatmaps
        must match the number of images provided via parameter `images`.
        The number is contained either in ``N`` or the first iterable's size.

    segmentation_maps : None or (N,H,W) ndarray or imgaug.augmentables.segmaps.SegmentationMapsOnImage or iterable of (H,W) ndarray or iterable of imgaug.augmentables.segmaps.SegmentationMapsOnImage
        The segmentation maps to augment.
        If anything else than ``SegmentationMapsOnImage``, then the number of
        segmaps must match the number of images provided via parameter
        `images`. The number is contained either in ``N`` or the first
        iterable's size.

    keypoints : None or list of (N,K,2) ndarray or tuple of number or imgaug.augmentables.kps.Keypoint or iterable of (K,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.kps.KeypointOnImage or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint
        The keypoints to augment.
        If a tuple (or iterable(s) of tuple), then iterpreted as (x,y)
        coordinates and must hence contain two numbers.
        A single tuple represents a single coordinate on one image, an
        iterable of tuples the coordinates on one image and an iterable of
        iterable of tuples the coordinates on several images. Analogous if
        ``Keypoint`` objects are used instead of tuples.
        If an ndarray, then ``N`` denotes the number of images and ``K`` the
        number of keypoints on each image.
        If anything else than ``KeypointsOnImage`` is provided, then the
        number of keypoint groups must match the number of images provided
        via parameter `images`. The number is contained e.g. in ``N`` or
        in case of "iterable of iterable of tuples" in the first iterable's
        size.

    bounding_boxes : None or (N,B,4) ndarray or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage or iterable of (B,4) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.bbs.BoundingBox or iterable of imgaug.augmentables.bbs.BoundingBoxesOnImage or iterable of iterable of tuple of number or iterable of iterable imgaug.augmentables.bbs.BoundingBox
        The bounding boxes to augment.
        This is analogous to the `keypoints` parameter. However, each
        tuple -- and also the last index in case of arrays -- has size 4,
        denoting the bounding box coordinates ``x1``, ``y1``, ``x2`` and ``y2``.

    polygons : None  or (N,#polys,#points,2) ndarray or imgaug.augmentables.polys.Polygon or imgaug.augmentables.polys.PolygonsOnImage or iterable of (#polys,#points,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.polys.Polygon or iterable of imgaug.augmentables.polys.PolygonsOnImage or iterable of iterable of (#points,2) ndarray or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint or iterable of iterable of imgaug.augmentables.polys.Polygon or iterable of iterable of iterable of tuple of number or iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint
        The polygons to augment.
        This is similar to the `keypoints` parameter. However, each polygon
        may be made up of several ``(x,y)`` coordinates (three or more are
        required for valid polygons).
        The following datatypes will be interpreted as a single polygon on a
        single image:

          * ``imgaug.augmentables.polys.Polygon``
          * ``iterable of tuple of number``
          * ``iterable of imgaug.augmentables.kps.Keypoint``

        The following datatypes will be interpreted as multiple polygons on a
        single image:

          * ``imgaug.augmentables.polys.PolygonsOnImage``
          * ``iterable of imgaug.augmentables.polys.Polygon``
          * ``iterable of iterable of tuple of number``
          * ``iterable of iterable of imgaug.augmentables.kps.Keypoint``
          * ``iterable of iterable of imgaug.augmentables.polys.Polygon``

        The following datatypes will be interpreted as multiple polygons on
        multiple images:

          * ``(N,#polys,#points,2) ndarray``
          * ``iterable of (#polys,#points,2) ndarray``
          * ``iterable of iterable of (#points,2) ndarray``
          * ``iterable of iterable of iterable of tuple of number``
          * ``iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint``

    line_strings : None or (N,#lines,#points,2) ndarray or imgaug.augmentables.lines.LineString or imgaug.augmentables.lines.LineStringOnImage or iterable of (#lines,#points,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.lines.LineString or iterable of imgaug.augmentables.lines.LineStringOnImage or iterable of iterable of (#points,2) ndarray or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint or iterable of iterable of imgaug.augmentables.polys.LineString or iterable of iterable of iterable of tuple of number or iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint
        The line strings to augment.
        See `polygons` for more details as polygons follow a similar
        structure to line strings.

    data
        Additional data that is saved in the batch and may be read out
        after augmentation. This could e.g. contain filepaths to each image
        in `images`. As this object is usually used for background
        augmentation with multiple processes, the augmented Batch objects might
        not be returned in the original order, making this information useful.

    """
    def __init__(self, images=None, heatmaps=None, segmentation_maps=None,
                 keypoints=None, bounding_boxes=None, polygons=None,
                 line_strings=None, data=None):
        self.images_unaug = images
        self.images_aug = None
        self.heatmaps_unaug = heatmaps
        self.heatmaps_aug = None
        self.segmentation_maps_unaug = segmentation_maps
        self.segmentation_maps_aug = None
        self.keypoints_unaug = keypoints
        self.keypoints_aug = None
        self.bounding_boxes_unaug = bounding_boxes
        self.bounding_boxes_aug = None
        self.polygons_unaug = polygons
        self.polygons_aug = None
        self.line_strings_unaug = line_strings
        self.line_strings_aug = None
        self.data = data

    def get_augmentables_names(self):
        """Get the names of types of augmentables that contain data.

        This method is intended for situations where one wants to know which
        data is contained in the batch that has to be augmented, visualized
        or something similar.

        Returns
        -------
        list of str
            Names of types of augmentables. E.g. ``["images", "polygons"]``.

        """
        return _get_augmentables_names(self, "_unaug")

    def to_normalized_batch(self):
        """Convert this unnormalized batch to an instance of Batch.

        As this method is intended to be called before augmentation, it
        assumes that none of the ``*_aug`` attributes is yet set.
        It will produce an AssertionError otherwise.

        The newly created Batch's ``*_unaug`` attributes will match the ones
        in this batch, just in normalized form.

        Returns
        -------
        imgaug.augmentables.batches.Batch
            The batch, with ``*_unaug`` attributes being normalized.

        """
        contains_no_augmented_data_yet = all([
            attr is None
            for attr_name, attr
            in self.__dict__.items()
            if attr_name.endswith("_aug")])
        assert contains_no_augmented_data_yet, (
            "Expected UnnormalizedBatch to not contain any augmented data "
            "before normalization, but at least one '*_aug' attribute was "
            "already set.")

        images_unaug = nlib.normalize_images(self.images_unaug)
        shapes = None
        if images_unaug is not None:
            shapes = [image.shape for image in images_unaug]

        return Batch(
            images=images_unaug,
            heatmaps=nlib.normalize_heatmaps(
                self.heatmaps_unaug, shapes),
            segmentation_maps=nlib.normalize_segmentation_maps(
                self.segmentation_maps_unaug, shapes),
            keypoints=nlib.normalize_keypoints(
                self.keypoints_unaug, shapes),
            bounding_boxes=nlib.normalize_bounding_boxes(
                self.bounding_boxes_unaug, shapes),
            polygons=nlib.normalize_polygons(
                self.polygons_unaug, shapes),
            line_strings=nlib.normalize_line_strings(
                self.line_strings_unaug, shapes),
            data=self.data
        )

    def to_batch_in_augmentation(self):
        """Convert this batch to a :class:`BatchInAugmentation` instance.

        Returns
        -------
        imgaug.augmentables.batches.BatchInAugmentation
            The converted batch.

        """
        return self.to_normalized_batch().to_batch_in_augmentation()

    def fill_from_batch_in_augmentation_(self, batch_in_augmentation):
        self.images_aug = batch_in_augmentation.images
        self.heatmaps_aug = batch_in_augmentation.heatmaps
        self.segmentation_maps_aug = batch_in_augmentation.segmentation_maps
        self.keypoints_aug = batch_in_augmentation.keypoints
        self.bounding_boxes_aug = batch_in_augmentation.bounding_boxes
        self.polygons_aug = batch_in_augmentation.polygons
        self.line_strings_aug = batch_in_augmentation.line_strings

    def fill_from_augmented_normalized_batch(self, batch_aug_norm):
        """
        Fill this batch with (normalized) augmentation results.

        This method receives a (normalized) Batch instance, takes all
        ``*_aug`` attributes out if it and assigns them to this
        batch *in unnormalized form*. Hence, the datatypes of all ``*_aug``
        attributes will match the datatypes of the ``*_unaug`` attributes.

        Parameters
        ----------
        batch_aug_norm: imgaug.augmentables.batches.Batch
            Batch after normalization and augmentation.

        Returns
        -------
        imgaug.augmentables.batches.UnnormalizedBatch
            New UnnormalizedBatch instance. All ``*_unaug`` attributes are
            taken from the old UnnormalizedBatch (without deepcopying them)
            and all ``*_aug`` attributes are taken from `batch_normalized`
            converted to unnormalized form.

        """
        # we take here the .data from the normalized batch instead of from
        # self for the rare case where one has decided to somehow change it
        # during augmentation
        batch = UnnormalizedBatch(
            images=self.images_unaug,
            heatmaps=self.heatmaps_unaug,
            segmentation_maps=self.segmentation_maps_unaug,
            keypoints=self.keypoints_unaug,
            bounding_boxes=self.bounding_boxes_unaug,
            polygons=self.polygons_unaug,
            line_strings=self.line_strings_unaug,
            data=batch_aug_norm.data
        )

        batch.images_aug = nlib.invert_normalize_images(
            batch_aug_norm.images_aug, self.images_unaug)
        batch.heatmaps_aug = nlib.invert_normalize_heatmaps(
            batch_aug_norm.heatmaps_aug, self.heatmaps_unaug)
        batch.segmentation_maps_aug = nlib.invert_normalize_segmentation_maps(
            batch_aug_norm.segmentation_maps_aug, self.segmentation_maps_unaug)
        batch.keypoints_aug = nlib.invert_normalize_keypoints(
            batch_aug_norm.keypoints_aug, self.keypoints_unaug)
        batch.bounding_boxes_aug = nlib.invert_normalize_bounding_boxes(
            batch_aug_norm.bounding_boxes_aug, self.bounding_boxes_unaug)
        batch.polygons_aug = nlib.invert_normalize_polygons(
            batch_aug_norm.polygons_aug, self.polygons_unaug)
        batch.line_strings_aug = nlib.invert_normalize_line_strings(
            batch_aug_norm.line_strings_aug, self.line_strings_unaug)

        return batch


class Batch(object):
    """
    Class encapsulating a batch before and after augmentation.

    Parameters
    ----------
    images : None or (N,H,W,C) ndarray or list of (H,W,C) ndarray
        The images to augment.

    heatmaps : None or list of imgaug.augmentables.heatmaps.HeatmapsOnImage
        The heatmaps to augment.

    segmentation_maps : None or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
        The segmentation maps to augment.

    keypoints : None or list of imgaug.augmentables.kps.KeypointOnImage
        The keypoints to augment.

    bounding_boxes : None or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
        The bounding boxes to augment.

    polygons : None or list of imgaug.augmentables.polys.PolygonsOnImage
        The polygons to augment.

    line_strings : None or list of imgaug.augmentables.lines.LineStringsOnImage
        The line strings to augment.

    data
        Additional data that is saved in the batch and may be read out
        after augmentation. This could e.g. contain filepaths to each image
        in `images`. As this object is usually used for background
        augmentation with multiple processes, the augmented Batch objects might
        not be returned in the original order, making this information useful.

    """
    def __init__(self, images=None, heatmaps=None, segmentation_maps=None,
                 keypoints=None, bounding_boxes=None, polygons=None,
                 line_strings=None, data=None):
        self.images_unaug = images
        self.images_aug = None
        self.heatmaps_unaug = heatmaps
        self.heatmaps_aug = None
        self.segmentation_maps_unaug = segmentation_maps
        self.segmentation_maps_aug = None
        self.keypoints_unaug = keypoints
        self.keypoints_aug = None
        self.bounding_boxes_unaug = bounding_boxes
        self.bounding_boxes_aug = None
        self.polygons_unaug = polygons
        self.polygons_aug = None
        self.line_strings_unaug = line_strings
        self.line_strings_aug = None
        self.data = data

    @property
    @ia.deprecated("Batch.images_unaug")
    def images(self):
        return self.images_unaug

    @property
    @ia.deprecated("Batch.heatmaps_unaug")
    def heatmaps(self):
        return self.heatmaps_unaug

    @property
    @ia.deprecated("Batch.segmentation_maps_unaug")
    def segmentation_maps(self):
        return self.segmentation_maps_unaug

    @property
    @ia.deprecated("Batch.keypoints_unaug")
    def keypoints(self):
        return self.keypoints_unaug

    @property
    @ia.deprecated("Batch.bounding_boxes_unaug")
    def bounding_boxes(self):
        return self.bounding_boxes_unaug

    def get_augmentables_names(self):
        """Get the names of types of augmentables that contain data.

        This method is intended for situations where one wants to know which
        data is contained in the batch that has to be augmented, visualized
        or something similar.

        Returns
        -------
        list of str
            Names of types of augmentables. E.g. ``["images", "polygons"]``.

        """
        return _get_augmentables_names(self, "_unaug")

    def to_normalized_batch(self):
        """Return this batch.

        This method does nothing and only exists to simplify interfaces
        that accept both :class:`UnnormalizedBatch` and :class:`Batch`.

        Returns
        -------
        imgaug.augmentables.batches.Batch
            This batch (not copied).

        """
        return self

    def to_batch_in_augmentation(self):
        """Convert this batch to a :class:`BatchInAugmentation` instance.

        Returns
        -------
        imgaug.augmentables.batches.BatchInAugmentation
            The converted batch.

        """
        def _copy(var):
            if var is not None:
                return utils.copy_augmentables(var)
            return var

        return BatchInAugmentation(
            images=_copy(self.images_unaug),
            heatmaps=_copy(self.heatmaps_unaug),
            segmentation_maps=_copy(self.segmentation_maps_unaug),
            keypoints=_copy(self.keypoints_unaug),
            bounding_boxes=_copy(self.bounding_boxes_unaug),
            polygons=_copy(self.polygons_unaug),
            line_strings=_copy(self.line_strings_unaug)
        )

    def fill_from_batch_in_augmentation_(self, batch_in_augmentation):
        self.images_aug = batch_in_augmentation.images
        self.heatmaps_aug = batch_in_augmentation.heatmaps
        self.segmentation_maps_aug = batch_in_augmentation.segmentation_maps
        self.keypoints_aug = batch_in_augmentation.keypoints
        self.bounding_boxes_aug = batch_in_augmentation.bounding_boxes
        self.polygons_aug = batch_in_augmentation.polygons
        self.line_strings_aug = batch_in_augmentation.line_strings

    @classmethod
    def _deepcopy_obj(cls, obj):
        if obj is None:
            return None
        elif ia.is_single_number(obj) or ia.is_string(obj):
            return obj
        elif isinstance(obj, list):
            return [cls._deepcopy_obj(el) for el in obj]
        elif isinstance(obj, tuple):
            return tuple([cls._deepcopy_obj(el) for el in obj])
        elif ia.is_np_array(obj):
            return np.copy(obj)
        elif hasattr(obj, "deepcopy"):
            return obj.deepcopy()
        else:
            return copy.deepcopy(obj)

    def deepcopy(self,
                 images_unaug=DEFAULT,
                 images_aug=DEFAULT,
                 heatmaps_unaug=DEFAULT,
                 heatmaps_aug=DEFAULT,
                 segmentation_maps_unaug=DEFAULT,
                 segmentation_maps_aug=DEFAULT,
                 keypoints_unaug=DEFAULT,
                 keypoints_aug=DEFAULT,
                 bounding_boxes_unaug=DEFAULT,
                 bounding_boxes_aug=DEFAULT,
                 polygons_unaug=DEFAULT,
                 polygons_aug=DEFAULT,
                 line_strings_unaug=DEFAULT,
                 line_strings_aug=DEFAULT):
        def _copy_optional(self_attr, arg):
            return self._deepcopy_obj(arg if arg is not DEFAULT else self_attr)

        batch = Batch(
            images=_copy_optional(self.images_unaug, images_unaug),
            heatmaps=_copy_optional(self.heatmaps_unaug, heatmaps_unaug),
            segmentation_maps=_copy_optional(self.segmentation_maps_unaug,
                                             segmentation_maps_unaug),
            keypoints=_copy_optional(self.keypoints_unaug, keypoints_unaug),
            bounding_boxes=_copy_optional(self.bounding_boxes_unaug,
                                          bounding_boxes_unaug),
            polygons=_copy_optional(self.polygons_unaug, polygons_unaug),
            line_strings=_copy_optional(self.line_strings_unaug,
                                        line_strings_unaug),
            data=copy.deepcopy(self.data)
        )
        batch.images_aug = _copy_optional(self.images_aug, images_aug)
        batch.heatmaps_aug = _copy_optional(self.heatmaps_aug, heatmaps_aug)
        batch.segmentation_maps_aug = _copy_optional(self.segmentation_maps_aug,
                                                     segmentation_maps_aug)
        batch.keypoints_aug = _copy_optional(self.keypoints_aug, keypoints_aug)
        batch.bounding_boxes_aug = _copy_optional(self.bounding_boxes_aug,
                                                  bounding_boxes_aug)
        batch.polygons_aug = _copy_optional(self.polygons_aug, polygons_aug)
        batch.line_strings_aug = _copy_optional(self.line_strings_aug,
                                                line_strings_aug)

        return batch


class _BatchInAugmentationPropagationContext(object):
    def __init__(self, batch, augmenter, hooks, parents):
        self.batch = batch
        self.augmenter = augmenter
        self.hooks = hooks
        self.parents = parents
        self.noned_info = None

    def __enter__(self):
        if self.hooks is not None:
            self.noned_info = self.batch.apply_propagation_hooks_(
                self.augmenter, self.hooks, self.parents)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.noned_info is not None:
            self.batch.unapply_propagation_hooks_(self.noned_info)


class BatchInAugmentation(object):
    """
    Class encapsulating a batch during the augmentation process.

    Data within the batch is already verified and normalized, similar to
    :class:`Batch`. Data within the batch may be changed in-place. No initial
    copy is needed.

    Parameters
    ----------
    images : None or (N,H,W,C) ndarray or list of (H,W,C) ndarray
        The images to augment.

    heatmaps : None or list of imgaug.augmentables.heatmaps.HeatmapsOnImage
        The heatmaps to augment.

    segmentation_maps : None or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
        The segmentation maps to augment.

    keypoints : None or list of imgaug.augmentables.kps.KeypointOnImage
        The keypoints to augment.

    bounding_boxes : None or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
        The bounding boxes to augment.

    polygons : None or list of imgaug.augmentables.polys.PolygonsOnImage
        The polygons to augment.

    line_strings : None or list of imgaug.augmentables.lines.LineStringsOnImage
        The line strings to augment.

    """
    def __init__(self, images=None, heatmaps=None, segmentation_maps=None,
                 keypoints=None, bounding_boxes=None, polygons=None,
                 line_strings=None, data=None):
        self.images = images
        self.heatmaps = heatmaps
        self.segmentation_maps = segmentation_maps
        self.keypoints = keypoints
        self.bounding_boxes = bounding_boxes
        self.polygons = polygons
        self.line_strings = line_strings
        self.data = data

    @property
    def empty(self):
        return self.nb_items == 0

    @property
    def nb_items(self):
        for augm_name in _AUGMENTABLE_NAMES:
            value = getattr(self, augm_name)
            if value is not None:
                return len(value)
        return 0

    def get_augmentables_names(self):
        """Get the names of types of augmentables that contain data.

        This method is intended for situations where one wants to know which
        data is contained in the batch that has to be augmented, visualized
        or something similar.

        Returns
        -------
        list of str
            Names of types of augmentables. E.g. ``["images", "polygons"]``.

        """
        return _get_augmentables_names(self, "")

    def get_augmentables(self):
        return _get_augmentables(self, "")

    def get_rowwise_shapes(self):
        nb_items = self.nb_items
        augmentables = self.get_augmentables()
        shapes = [None] * nb_items
        found = np.zeros((nb_items,), dtype=bool)
        for augm_name, augm_value, _augm_attr_name in augmentables:
            if augm_value is not None:
                if augm_name == "images" and ia.is_np_array(augm_value):
                    shapes = [augm_value.shape[1:]] * nb_items
                else:
                    for i, item in enumerate(augm_value):
                        if item is not None:
                            shapes[i] = item.shape
                            found[i] = True
                if np.all(found):
                    return shapes
        return shapes

    def subselect_items_by_indices(self, indices):
        kwargs = {"data": self.data}
        for augm_name in _AUGMENTABLE_NAMES:
            items = getattr(self, augm_name)
            if items is not None:
                if augm_name == "images" and ia.is_np_array(items):
                    items = items[indices]
                else:
                    items = [items[index] for index in indices]
            kwargs[augm_name] = items

        return BatchInAugmentation(**kwargs)

    def invert_subselect_items_by_indices_(self, indices, batch_subselected):
        for augm_name in _AUGMENTABLE_NAMES:
            items = getattr(self, augm_name)
            if items is not None:
                items_sub = getattr(batch_subselected, augm_name)
                if augm_name == "images" and ia.is_np_array(items):
                    # An array does not have to stay an array after
                    # augmentation. The shapes and/or dtypes of items may
                    # change, turning the array into a list.
                    if ia.is_np_array(items_sub):
                        shapes = {items.shape[1:], items_sub.shape[1:]}
                        dtypes = {items.dtype.name, items_sub.dtype.name}
                    else:
                        shapes = set(
                            [items.shape[1:]]
                            + [image.shape for image in items_sub])
                        dtypes = set(
                            [items.dtype.name]
                            + [image.dtype.name for image in items_sub])

                    if len(shapes) == 1 and len(dtypes) == 1:
                        items[indices] = items_sub
                    else:
                        self.images = list(items)
                        for ith_index, index in enumerate(indices):
                            self.images[index] = items_sub[ith_index]
                else:
                    for ith_index, index in enumerate(indices):
                        items[index] = items_sub[ith_index]

    def propagation_hooks_ctx(self, augmenter, hooks, parents):
        return _BatchInAugmentationPropagationContext(
            self, augmenter=augmenter, hooks=hooks, parents=parents)

    def apply_propagation_hooks_(self, augmenter, hooks, parents):
        if hooks is None:
            return None

        noned_info = []
        for _augm_name, value, attr_name in self.get_augmentables():
            is_prop = hooks.is_propagating(
                value, augmenter=augmenter, parents=parents, default=True)
            if not is_prop:
                setattr(self, attr_name, None)
                noned_info.append((attr_name, value))
        return noned_info

    def unapply_propagation_hooks_(self, noned_info):
        for attr_name, value in noned_info:
            setattr(self, attr_name, value)

    def to_batch_in_augmentation(self):
        """Convert this batch to a :class:`BatchInAugmentation` instance.

        This method simply returns the batch itself. It exists for consistency
        with the other batch classes.

        Returns
        -------
        imgaug.augmentables.batches.BatchInAugmentation
            The batch itself. (Not copied.)

        """
        return self

    def fill_from_batch_in_augmentation_(self, batch_in_augmentation):
        if batch_in_augmentation is self:
            return

        self.images = batch_in_augmentation.images
        self.heatmaps = batch_in_augmentation.heatmaps
        self.segmentation_maps = batch_in_augmentation.segmentation_maps
        self.keypoints = batch_in_augmentation.keypoints
        self.bounding_boxes = batch_in_augmentation.bounding_boxes
        self.polygons = batch_in_augmentation.polygons
        self.line_strings = batch_in_augmentation.line_strings

    def to_batch(self, batch_before_aug):
        """Convert this batch into a :class:`Batch` instance.

        Parameters
        ----------
        batch_before_aug : imgaug.augmentables.batches.Batch
            The batch before augmentation. It is required to set the input
            data of the :class:`Batch` instance, e.g. ``images_unaug``
            or ``data``.

        Returns
        -------
        imgaug.augmentables.batches.Batch
            Batch, with original unaugmented inputs from `batch_before_aug`
            and augmented outputs from this :class:`BatchInAugmentation`
            instance.

        """
        batch = Batch(
            images=batch_before_aug.images_unaug,
            heatmaps=batch_before_aug.heatmaps_unaug,
            segmentation_maps=batch_before_aug.segmentation_maps_unaug,
            keypoints=batch_before_aug.keypoints_unaug,
            bounding_boxes=batch_before_aug.bounding_boxes_unaug,
            polygons=batch_before_aug.polygons_unaug,
            line_strings=batch_before_aug.line_strings_unaug,
            data=batch_before_aug.data
        )
        batch.images_aug = self.images
        batch.heatmaps_aug = self.heatmaps
        batch.segmentation_maps_aug = self.segmentation_maps
        batch.keypoints_aug = self.keypoints
        batch.bounding_boxes_aug = self.bounding_boxes
        batch.polygons_aug = self.polygons
        batch.line_strings_aug = self.line_strings
        return batch
