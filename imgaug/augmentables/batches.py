from __future__ import print_function, division, absolute_import

import copy
import warnings

import numpy as np

import imgaug.imgaug as ia
import imgaug.augmentables.normalization as normalization


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

    def set_images_aug_normalized(self, images):
        self.images_aug = normalization.invert_normalize_images(
            images, self.images_unaug)

    def set_heatmaps_aug_normalized(self, heatmaps):
        self.heatmaps_aug = normalization.invert_normalize_heatmaps(
            heatmaps, self.heatmaps_unaug)

    def set_segmentation_maps_aug_normalized(self, segmentation_maps):
        self.segmentation_maps_aug = normalization.invert_normalize_segmentation_maps(
            segmentation_maps, self.segmentation_maps_unaug)

    def set_keypoints_aug_normalized(self, keypoints):
        self.keypoints_aug = normalization.invert_normalize_keypoints(
            keypoints, self.keypoints_unaug)

    def set_bounding_boxes_aug_normalized(self, bounding_boxes):
        self.bounding_boxes_aug = normalization.invert_normalize_bounding_boxes(
            bounding_boxes, self.bounding_boxes_unaug)

    def set_polygons_aug_normalized(self, polygons):
        self.polygons_aug = normalization.invert_normalize_polygons(
            polygons, self.polygons_unaug)

    def get_images_unaug_normalized(self):
        return normalization.normalize_images(self.images_unaug)

    def get_heatmaps_unaug_normalized(self):
        return normalization.normalize_heatmaps(
            self.heatmaps_unaug, self.get_images_unaug_normalized())

    def get_segmentation_maps_unaug_normalized(self):
        return normalization.normalize_segmentation_maps(
            self.segmentation_maps_unaug, self.get_images_unaug_normalized())

    def get_keypoints_unaug_normalized(self):
        return normalization.normalize_keypoints(
            self.keypoints_unaug, self.get_images_unaug_normalized())

    def get_bounding_boxes_unaug_normalized(self):
        return normalization.normalize_bounding_boxes(
            self.bounding_boxes_unaug, self.get_images_unaug_normalized())

    def get_polygons_unaug_normalized(self):
        return normalization.normalize_polygons(
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

    def deepcopy(self):
        batch = Batch(
            images=self._deepcopy_obj(self.images_unaug),
            heatmaps=self._deepcopy_obj(self.heatmaps_unaug),
            segmentation_maps=self._deepcopy_obj(self.segmentation_maps_unaug),
            keypoints=self._deepcopy_obj(self.keypoints_unaug),
            bounding_boxes=self._deepcopy_obj(self.bounding_boxes_unaug),
            polygons=self._deepcopy_obj(self.polygons_unaug),
            data=copy.deepcopy(self.data)
        )
        batch.images_aug = self._deepcopy_obj(self.images_aug)
        batch.heatmaps_aug = self._deepcopy_obj(self.heatmaps_aug)
        batch.segmentation_maps_aug = self._deepcopy_obj(self.segmentation_maps_aug)
        batch.keypoints_aug = self._deepcopy_obj(self.keypoints_aug)
        batch.bounding_boxes_aug = self._deepcopy_obj(self.bounding_boxes_aug)
        batch.polygons_aug = self._deepcopy_obj(self.polygons_aug)

        return batch
