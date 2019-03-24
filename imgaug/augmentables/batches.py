from __future__ import print_function, division, absolute_import

import copy
import warnings

import numpy as np

import imgaug.imgaug as ia
import imgaug.augmentables.normalization as nlib

DEFAULT = "DEFAULT"


class Batch(object):
    """
    Class encapsulating a batch before and after augmentation.

    Parameters
    ----------
    images : None or (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
        The images to augment.

    heatmaps : None or list of imgaug.HeatmapsOnImage
        The heatmaps to augment.

    segmentation_maps : None or list of SegmentationMapOnImage
        The segmentation maps to augment.

    keypoints : None or list of KeypointOnImage
        The keypoints to augment.

    bounding_boxes : None or list of BoundingBoxesOnImage
        The bounding boxes to augment.

    polygons : None or list of PolygonsOnImage
        The polygons to augment.

    data
        Additional data that is saved in the batch and may be read out
        after augmentation. This could e.g. contain filepaths to each image
        in `images`. As this object is usually used for background
        augmentation with multiple processes, the augmented Batch objects might
        not be returned in the original order, making this information useful.

    """
    def __init__(self, images=None, heatmaps=None, segmentation_maps=None,
                 keypoints=None, bounding_boxes=None, polygons=None,
                 data=None):
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
        self.data = data

    def to_normalized_batch(self):
        images_unaug = nlib.normalize_images(self.images_unaug)
        images_aug = nlib.normalize_images(self.images_aug)
        heatmaps_unaug = nlib.normalize_heatmaps(self.heatmaps_unaug, images_unaug)
        heatmaps_aug = nlib.normalize_heatmaps(self.heatmaps_aug, images_aug)
        segmaps_unaug = nlib.normalize_segmentation_maps(self.segmentation_maps_unaug, images_unaug)
        segmaps_aug = nlib.normalize_segmentation_maps(self.segmentation_maps_aug, images_aug)
        keypoints_unaug = nlib.normalize_keypoints(self.keypoints_unaug, images_unaug)
        keypoints_aug = nlib.normalize_keypoints(self.keypoints_aug, images_aug)
        bounding_boxes_unaug = nlib.normalize_bounding_boxes(self.bounding_boxes_unaug, images_unaug)
        bounding_boxes_aug = nlib.normalize_bounding_boxes(self.bounding_boxes_aug, images_aug)
        polygons_unaug = nlib.normalize_polygons(self.polygons_unaug, images_unaug)
        polygons_aug = nlib.normalize_keypoints(self.polygons_aug, images_aug)

        return self.deepcopy(
            images_unaug=images_unaug,
            images_aug=images_aug,
            heatmaps_unaug=heatmaps_unaug,
            heatmaps_aug=heatmaps_aug,
            segmentation_maps_unaug=segmaps_unaug,
            segmentation_maps_aug=segmaps_aug,
            keypoints_unaug=keypoints_unaug,
            keypoints_aug=keypoints_aug,
            bounding_boxes_unaug=bounding_boxes_unaug,
            bounding_boxes_aug=bounding_boxes_aug,
            polygons_unaug=polygons_unaug,
            polygons_aug=polygons_aug
        )

    @classmethod
    def from_normalized_batch(cls, batch_normalized, batch_old):
        bnorm = batch_normalized
        bold = batch_old

        images_unaug = nlib.invert_normalize_images(bnorm.images_unaug, bold.images_unaug)
        images_aug = nlib.invert_normalize_images(bnorm.images_aug, bold.images_aug)
        heatmaps_unaug = nlib.invert_normalize_heatmaps(bnorm.heatmaps_unaug, bold.heatmaps_unaug)
        heatmaps_aug = nlib.invert_normalize_heatmaps(bnorm.heatmaps_aug, bold.heatmaps_aug)
        segmaps_unaug = nlib.invert_normalize_segmentation_maps(bnorm.segmentation_maps_unaug,
                                                                bold.segmentation_maps_unaug)
        segmaps_aug = nlib.invert_normalize_segmentation_maps(bnorm.segmentation_maps_aug,
                                                              bold.segmentation_maps_aug)
        keypoints_unaug = nlib.invert_normalize_keypoints(bnorm.keypoints_unaug, bold.keypoints_unaug)
        keypoints_aug = nlib.invert_normalize_keypoints(bnorm.keypoints_aug, bold.keypoints_aug)
        bbs_unaug = nlib.invert_normalize_bounding_boxes(bnorm.bounding_boxes_unaug, bold.bounding_boxes_unaug)
        bbs_aug = nlib.invert_normalize_bounding_boxes(bnorm.bounding_boxes_aug, bold.bounding_boxes_aug)
        polygons_unaug = nlib.invert_normalize_polygons(bnorm.polygons_unaug, bold.polygons_unaug)
        polygons_aug = nlib.invert_normalize_keypoints(bnorm.polygons_aug, bold.polygons_aug)
        
        return batch_old.deepcopy(
            images_unaug=images_unaug,
            images_aug=images_aug,
            heatmaps_unaug=heatmaps_unaug,
            heatmaps_aug=heatmaps_aug,
            segmentation_maps_unaug=segmaps_unaug,
            segmentation_maps_aug=segmaps_aug,
            keypoints_unaug=keypoints_unaug,
            keypoints_aug=keypoints_aug,
            bounding_boxes_unaug=bbs_unaug,
            bounding_boxes_aug=bbs_aug,
            polygons_unaug=polygons_unaug,
            polygons_aug=polygons_aug
        )

    def set_images_aug_normalized(self, images):
        self.images_aug = nlib.invert_normalize_images(
            images, self.images_unaug)

    def set_heatmaps_aug_normalized(self, heatmaps):
        self.heatmaps_aug = nlib.invert_normalize_heatmaps(
            heatmaps, self.heatmaps_unaug)

    def set_segmentation_maps_aug_normalized(self, segmentation_maps):
        self.segmentation_maps_aug = nlib.invert_normalize_segmentation_maps(
            segmentation_maps, self.segmentation_maps_unaug)

    def set_keypoints_aug_normalized(self, keypoints):
        self.keypoints_aug = nlib.invert_normalize_keypoints(
            keypoints, self.keypoints_unaug)

    def set_bounding_boxes_aug_normalized(self, bounding_boxes):
        self.bounding_boxes_aug = nlib.invert_normalize_bounding_boxes(
            bounding_boxes, self.bounding_boxes_unaug)

    def set_polygons_aug_normalized(self, polygons):
        self.polygons_aug = nlib.invert_normalize_polygons(
            polygons, self.polygons_unaug)

    def get_images_unaug_normalized(self):
        return nlib.normalize_images(self.images_unaug)

    def get_heatmaps_unaug_normalized(self):
        return nlib.normalize_heatmaps(
            self.heatmaps_unaug, self.get_images_unaug_normalized())

    def get_segmentation_maps_unaug_normalized(self):
        return nlib.normalize_segmentation_maps(
            self.segmentation_maps_unaug, self.get_images_unaug_normalized())

    def get_keypoints_unaug_normalized(self):
        return nlib.normalize_keypoints(
            self.keypoints_unaug, self.get_images_unaug_normalized())

    def get_bounding_boxes_unaug_normalized(self):
        return nlib.normalize_bounding_boxes(
            self.bounding_boxes_unaug, self.get_images_unaug_normalized())

    def get_polygons_unaug_normalized(self):
        return nlib.normalize_polygons(
            self.polygons_unaug, self.get_images_unaug_normalized())

    @property
    def images(self):
        warnings.warn(DeprecationWarning(
            "Accessing imgaug.Batch.images is deprecated. Access instead "
            "imgaug.Batch.images_unaug or imgaug.Batch.images_aug."))
        return self.images_unaug

    @property
    def heatmaps(self):
        warnings.warn(DeprecationWarning(
            "Accessing imgaug.Batch.heatmaps is deprecated. Access instead "
            "imgaug.Batch.heatmaps_unaug or imgaug.Batch.heatmaps_aug."))
        return self.heatmaps_unaug

    @property
    def segmentation_maps(self):
        warnings.warn(DeprecationWarning(
            "Accessing imgaug.Batch.segmentation_maps is deprecated. Access "
            "instead imgaug.Batch.segmentation_maps_unaug or "
            "imgaug.Batch.segmentation_maps_aug."))
        return self.segmentation_maps_unaug

    @property
    def keypoints(self):
        warnings.warn(DeprecationWarning(
            "Accessing imgaug.Batch.keypoints is deprecated. Access "
            "instead imgaug.Batch.keypoints_unaug or "
            "imgaug.Batch.keypoints_aug."))
        return self.keypoints_unaug

    @property
    def bounding_boxes(self):
        warnings.warn(DeprecationWarning(
            "Accessing imgaug.Batch.bounding_boxes is deprecated. Access "
            "instead imgaug.Batch.bounding_boxes_unaug or "
            "imgaug.Batch.bounding_boxes_aug."))
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
                 polygons_aug=DEFAULT):
        def _copy_optional(var, arg):
            return self._deepcopy_obj(var) if arg is not DEFAULT else arg

        batch = Batch(
            images=_copy_optional(self.images_unaug, images_unaug),
            heatmaps=_copy_optional(self.heatmaps_unaug, heatmaps_unaug),
            segmentation_maps=_copy_optional(self.segmentation_maps_unaug, segmentation_maps_unaug),
            keypoints=_copy_optional(self.keypoints_unaug, keypoints_unaug),
            bounding_boxes=_copy_optional(self.bounding_boxes_unaug, bounding_boxes_unaug),
            polygons=_copy_optional(self.polygons_unaug, polygons_unaug),
            data=copy.deepcopy(self.data)
        )
        batch.images_aug = _copy_optional(self.images_aug, images_aug)
        batch.heatmaps_aug = _copy_optional(self.heatmaps_aug, heatmaps_aug)
        batch.segmentation_maps_aug = _copy_optional(self.segmentation_maps_aug, segmentation_maps_aug)
        batch.keypoints_aug = _copy_optional(self.keypoints_aug, keypoints_aug)
        batch.bounding_boxes_aug = _copy_optional(self.bounding_boxes_aug, bounding_boxes_aug)
        batch.polygons_aug = _copy_optional(self.polygons_aug, polygons_aug)

        return batch
