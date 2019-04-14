from __future__ import print_function, division, absolute_import

import copy
import warnings

import numpy as np

from .. import imgaug as ia
from . import normalization as nlib

DEFAULT = "DEFAULT"


# TODO also support (H,W,C) for heatmaps of len(images) == 1
# TODO also support (H,W) for segmaps of len(images) == 1
class UnnormalizedBatch(object):
    """
    Class for batches of unnormalized data before and after augmentation.

    Parameters
    ----------
    images : None \
             or (N,H,W,C) ndarray \
             or (N,H,W) ndarray \
             or iterable of (H,W,C) ndarray \
             or iterable of (H,W) ndarray
        The images to augment.

    heatmaps : None \
               or (N,H,W,C) ndarray \
               or imgaug.augmentables.heatmaps.HeatmapsOnImage \
               or iterable of (H,W,C) ndarray \
               or iterable of imgaug.augmentables.heatmaps.HeatmapsOnImage
        The heatmaps to augment.
        If anything else than ``HeatmapsOnImage``, then the number of heatmaps
        must match the number of images provided via parameter `images`.
        The number is contained either in ``N`` or the first iterable's size.

    segmentation_maps : None \
            or (N,H,W) ndarray \
            or imgaug.augmentables.segmaps.SegmentationMapOnImage \
            or iterable of (H,W) ndarray \
            or iterable of imgaug.augmentables.segmaps.SegmentationMapOnImage
        The segmentation maps to augment.
        If anything else than ``SegmentationMapOnImage``, then the number of
        segmaps must match the number of images provided via parameter
        `images`. The number is contained either in ``N`` or the first
        iterable's size.

    keypoints : None \
                or list of (N,K,2) ndarray \
                or tuple of number \
                or imgaug.augmentables.kps.Keypoint \
                or iterable of (K,2) ndarray \
                or iterable of tuple of number \
                or iterable of imgaug.augmentables.kps.Keypoint \
                or iterable of imgaug.augmentables.kps.KeypointOnImage \
                or iterable of iterable of tuple of number \
                or iterable of iterable of imgaug.augmentables.kps.Keypoint
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

    bounding_boxes : None \
                or (N,B,4) ndarray \
                or tuple of number \
                or imgaug.augmentables.bbs.BoundingBox \
                or imgaug.augmentables.bbs.BoundingBoxesOnImage \
                or iterable of (B,4) ndarray \
                or iterable of tuple of number \
                or iterable of imgaug.augmentables.bbs.BoundingBox \
                or iterable of imgaug.augmentables.bbs.BoundingBoxesOnImage \
                or iterable of iterable of tuple of number \
                or iterable of iterable imgaug.augmentables.bbs.BoundingBox
        The bounding boxes to augment.
        This is analogous to the `keypoints` parameter. However, each
        tuple -- and also the last index in case of arrays -- has size 4,
        denoting the bounding box coordinates ``x1``, ``y1``, ``x2`` and ``y2``.

    polygons : None  \
               or (N,#polys,#points,2) ndarray \
               or imgaug.augmentables.polys.Polygon \
               or imgaug.augmentables.polys.PolygonsOnImage \
               or iterable of (#polys,#points,2) ndarray \
               or iterable of tuple of number \
               or iterable of imgaug.augmentables.kps.Keypoint \
               or iterable of imgaug.augmentables.polys.Polygon \
               or iterable of imgaug.augmentables.polys.PolygonsOnImage \
               or iterable of iterable of (#points,2) ndarray \
               or iterable of iterable of tuple of number \
               or iterable of iterable of imgaug.augmentables.kps.Keypoint \
               or iterable of iterable of imgaug.augmentables.polys.Polygon \
               or iterable of iterable of iterable of tuple of number \
               or iterable of iterable of iterable of tuple of \
               imgaug.augmentables.kps.Keypoint
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

    line_strings : None  \
               or (N,#lines,#points,2) ndarray \
               or imgaug.augmentables.lines.LineString \
               or imgaug.augmentables.lines.LineStringOnImage \
               or iterable of (#lines,#points,2) ndarray \
               or iterable of tuple of number \
               or iterable of imgaug.augmentables.kps.Keypoint \
               or iterable of imgaug.augmentables.lines.LineString \
               or iterable of imgaug.augmentables.lines.LineStringOnImage \
               or iterable of iterable of (#points,2) ndarray \
               or iterable of iterable of tuple of number \
               or iterable of iterable of imgaug.augmentables.kps.Keypoint \
               or iterable of iterable of imgaug.augmentables.polys.LineString \
               or iterable of iterable of iterable of tuple of number \
               or iterable of iterable of iterable of tuple of \
               imgaug.augmentables.kps.Keypoint
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
        assert all([
            attr is None for attr_name, attr in self.__dict__.items()
            if attr_name.endswith("_aug")]), \
            "Expected UnnormalizedBatch to not contain any augmented data " \
            "before normalization, but at least one '*_aug' attribute was " \
            "already set."

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

    segmentation_maps : None or list of \
                        imgaug.augmentables.segmaps.SegmentationMapOnImage
        The segmentation maps to augment.

    keypoints : None or list of imgaug.augmentables.kps.KeypointOnImage
        The keypoints to augment.

    bounding_boxes : None \
                     or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
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
