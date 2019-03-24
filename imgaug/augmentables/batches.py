from __future__ import print_function, division, absolute_import

import copy
import warnings

import numpy as np

import imgaug.imgaug as ia


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

    # TODO replace partially with dtypes.restore_dtypes_()
    @classmethod
    def _restore_dtype_and_merge(cls, arr, input_dtype):
        if isinstance(arr, list):
            arr = [cls._restore_dtype_and_merge(arr_i, input_dtype)
                   for arr_i in arr]
            shapes = [arr_i.shape for arr_i in arr]
            if len(set(shapes)) == 1:
                arr = np.array(arr)

        if ia.is_np_array(arr):
            if input_dtype.kind == "i":
                arr = np.round(arr).astype(input_dtype)
            elif input_dtype.kind == "u":
                arr = np.round(arr)
                arr = np.clip(arr, 0, np.iinfo(input_dtype).max)
                arr = arr.astype(input_dtype)
        return arr

    def set_images_aug_normalized(self, images):
        attr = self.images_unaug
        if attr is None:
            assert images is None
            self.images_aug = None
        elif ia.is_np_array(attr):
            if attr.ndim == 2:
                self.images_aug = images[0, ..., 0]
            elif attr.ndim == 3:
                self.images_aug = images[..., 0]
            else:
                self.images_aug = images
        elif ia.is_iterable(attr):
            if isinstance(attr, tuple):
                self.images_aug = tuple(images)
            else:
                self.images_aug = list(images)
        raise ValueError(
            ("Expected argument 'images' for Batch to be any of the following: "
             + "None or array or iterable of array. Got type: %s.") % (
                type(self.images_unaug),)
        )

    def set_heatmaps_aug_normalized(self, heatmaps):
        ntype = self._get_heatmaps_unaug_normalization_type()
        if ntype == "None":
            assert heatmaps is None
            self.heatmaps_aug = heatmaps
        elif ntype == "array[float]":
            assert len(heatmaps) == 1
            self.heatmaps_aug = heatmaps[0].arr_0to1
        elif ntype == "HeatmapsOnImage":
            assert len(heatmaps) == 1
            self.heatmaps_aug = heatmaps[0]
        elif ntype == "iterable[empty]":
            assert heatmaps is None
            self.heatmaps_aug = []
        elif ntype == "iterable-array[float]":
            self.heatmaps_aug = [hm_i.arr_0to1 for hm_i in heatmaps]
        else:
            assert ntype == "iterable-HeatmapsOnImage"
            self.heatmaps_aug = heatmaps

    def set_segmentation_maps_aug_normalized(self, segmentation_maps):
        ntype = self._get_segmentation_maps_unaug_normalization_type()
        if ntype == "None":
            assert segmentation_maps is None
            self.segmentation_maps_aug = segmentation_maps
        elif ntype in ["array[int]", "array[uint]", "array[bool]"]:
            assert len(segmentation_maps) == 1
            self.segmentation_maps_aug = segmentation_maps[0].arr
        elif ntype == "SegmentationMapOnImage":
            assert len(segmentation_maps) == 1
            self.segmentation_maps_aug = segmentation_maps[0]
        elif ntype == "iterable[empty]":
            assert segmentation_maps is None
            self.segmentation_maps_aug = []
        elif ntype in ["iterable-array[int]", "iterable-array[uint]", "iterable-array[bool]"]:
            self.segmentation_maps_aug = [
                segmap_i.arr for segmap_i in segmentation_maps]
        else:
            assert ntype == "iterable-SegmentationMapOnImage"
            self.segmentation_maps_aug = segmentation_maps

    def set_keypoints_aug_normalized(self, keypoints):
        ntype = self._get_keypoints_unaug_normalization_type()
        if ntype == "None":
            assert keypoints is None
            self.keypoints_aug = keypoints
        elif ntype in ["array[float]", "array[int]", "array[uint]"]:
            assert len(keypoints) == 1
            input_dtype = self.keypoints_unaug.dtype
            self.keypoints_aug = self._restore_dtype_and_merge(
                [kpsoi.get_coords_array() for kpsoi in keypoints],
                input_dtype)
        elif ntype == "(x,y)":
            assert len(keypoints) == 1
            assert len(keypoints[0].keypoints) == 1
            self.keypoints_aug = (keypoints[0].keypoints[0].x,
                                  keypoints[0].keypoints[0].y)
        elif ntype == "Keypoint":
            assert len(keypoints) == 1
            assert len(keypoints[0].keypoints) == 1
            self.keypoints_aug = keypoints[0].keypoints[0]
        elif ntype == "KeypointsOnImage":
            assert len(keypoints) == 1
            self.keypoints_aug = keypoints[0]
        elif ntype == "iterable[empty]":
            assert keypoints is None
            self.keypoints_aug = []
        elif ntype in ["iterable-array[float]", "iterable-array[int]", "iterable-array[uint]"]:
            nonempty, _, _ = self._find_first_nonempty(self.keypoints_unaug)
            input_dtype = nonempty.dtype
            self.keypoints_aug = [
                self._restore_dtype_and_merge(kps_i.get_coords_array(),
                                              input_dtype)
                for kps_i in keypoints]
        elif ntype == "iterable-(x,y)":
            assert len(keypoints) == 1
            self.keypoints_aug = [
                (kp.x, kp.y) for kp in keypoints[0].keypoints]
        elif ntype == "iterable-KeypointsOnImage":
            self.keypoints_aug = keypoints
        elif ntype == "iterable-iterable[empty]":
            assert keypoints is None
            self.keypoints_aug = self.keypoints_unaug[:]
        elif ntype == "iterable-iterable-(x,y)":
            self.keypoints_aug = [
                [(kp.x, kp.y) for kp in kpsoi.keypoints]
                for kpsoi in keypoints]
        else:
            assert ntype == "iterable-iterable-Keypoint"
            self.keypoints_aug = [
                [kp for kp in kpsoi.keypoints]
                for kpsoi in keypoints]

    def set_bounding_boxes_aug_normalized(self, bounding_boxes):
        ntype = self._get_bounding_boxes_unaug_normalization_type()
        if ntype == "None":
            assert bounding_boxes is None
            self.bounding_boxes_aug = bounding_boxes
        elif ntype in ["array[float]", "array[int]", "array[uint]"]:
            assert len(bounding_boxes) == 1
            input_dtype = self.bounding_boxes_unaug.dtype
            self.bounding_boxes_aug = self._restore_dtype_and_merge([
                bbsoi.to_xyxy_array() for bbsoi in bounding_boxes
            ], input_dtype)
        elif ntype == "(x1,y1,x2,y2)":
            assert len(bounding_boxes) == 1
            assert len(bounding_boxes[0].bounding_boxes) == 1
            bb = bounding_boxes[0].bounding_boxes[0]
            self.bounding_boxes_aug = (bb.x1, bb.y1, bb.x2, bb.y2)
        elif ntype == "BoundingBox":
            assert len(bounding_boxes) == 1
            assert len(bounding_boxes[0].bounding_boxes) == 1
            self.bounding_boxes_aug = bounding_boxes[0].bounding_boxes[0]
        elif ntype == "BoundingBoxesOnImage":
            assert len(bounding_boxes) == 1
            self.bounding_boxes_aug = bounding_boxes[0]
        elif ntype == "iterable[empty]":
            assert bounding_boxes is None
            self.bounding_boxes_aug = []
        elif ntype in ["iterable-array[float]", "iterable-array[int]", "iterable-array[uint]"]:
            nonempty, _, _ = self._find_first_nonempty(self.bounding_boxes_unaug)
            input_dtype = nonempty.dtype
            self.bounding_boxes_aug = [
                self._restore_dtype_and_merge(bbsoi.to_xyxy_array(), input_dtype)
                for bbsoi in bounding_boxes]
        elif ntype == "iterable-(x1,y1,x2,y2)":
            assert len(bounding_boxes) == 1
            self.bounding_boxes_aug = [
                (bb.x1, bb.y1, bb.x2, bb.y2)
                for bb in bounding_boxes[0].bounding_boxes]
        elif ntype == "iterable-BoundingBoxesOnImage":
            self.bounding_boxes_aug = bounding_boxes
        elif ntype == "iterable-iterable[empty]":
            assert bounding_boxes is None
            self.bounding_boxes_aug = self.bounding_boxes_unaug[:]
        elif ntype == "iterable-iterable-(x1,y1,x2,y2)":
            self.bounding_boxes_aug = [
                [(bb.x1, bb.y1, bb.x2, bb.y2) for bb in bbsoi.bounding_boxes]
                for bbsoi in bounding_boxes]
        else:
            assert ntype == "iterable-iterable-BoundingBox"
            self.bounding_boxes_aug = [
                [bb for bb in bbsoi.bounding_boxes]
                for bbsoi in bounding_boxes]

    def set_polygons_aug_normalized(self, polygons):
        # TODO get rid of this deferred import
        from imgaug.augmentables.kps import Keypoint

        ntype = self._get_polygons_unaug_normalization_type()
        if ntype == "None":
            assert polygons is None
            self.polygons_aug = polygons
        elif ntype in ["array[float]", "array[int]", "array[uint]"]:
            input_dtype = self.polygons_unaug.dtype
            self.polygons_aug = self._restore_dtype_and_merge([
                [poly.exterior for poly in psoi.polygons]
                for psoi in polygons
            ], input_dtype)
        elif ntype == "Polygon":
            assert len(polygons) == 1
            assert len(polygons[0].polygons) == 1
            self.polygons_aug = polygons[0].polygons[0]
        elif ntype == "PolygonsOnImage":
            assert len(polygons) == 1
            self.polygons_aug = polygons[0]
        elif ntype == "iterable[empty]":
            assert polygons is None
            self.polygons_aug = []
        elif ntype in ["iterable-array[float]", "iterable-array[int]", "iterable-array[uint]"]:
            nonempty, _, _ = self._find_first_nonempty(self.polygons_unaug)
            input_dtype = nonempty.dtype
            self.polygons_aug = [
                self._restore_dtype_and_merge(
                    [poly.exterior for poly in psoi.poylgons],
                    input_dtype)
                for psoi in polygons
            ]
        elif ntype == "iterable-(x,y)":
            assert len(polygons) == 1
            assert len(polygons[0].polygons) == 1
            self.polygons_aug = [(point.x, point.y)
                                 for point in polygons[0].polygons[0].exterior]
        elif ntype == "iterable-Keypoint":
            assert len(polygons) == 1
            assert len(polygons[0].polygons) == 1
            self.polygons_aug = [Keypoint(x=point.x, y=point.y)
                                 for point in polygons[0].polygons[0].exterior]
        elif ntype == "iterable-Polygon":
            assert len(polygons) == 1
            assert len(polygons[0].polygons) == len(self.polygons_unaug[0].polygons)
            self.polygons_aug = polygons[0].polygons
        elif ntype == "iterable-PolygonsOnImage":
            self.polygons_aug = polygons
        elif ntype == "iterable-iterable[empty]":
            assert polygons is None
            self.polygons_aug = self.polygons_unaug[:]
        elif ntype in ["iterable-iterable-array[float]", "iterable-iterable-array[int]", "iterable-iterable-array[uint]"]:
            nonempty, _, _ = self._find_first_nonempty(self.polygons_unaug)
            input_dtype = nonempty.dtype
            self.polygons_aug = [
                [self._restore_dtype_and_merge(poly.exterior, input_dtype)
                 for poly in psoi.polygons]
                for psoi in polygons
            ]
        elif ntype == "iterable-iterable-(x,y)":
            assert len(polygons) == 1
            self.polygons_aug = [
                [(point[0], point[1]) for point in polygon.exterior]
                for polygon in polygons[0].polygons]
        elif ntype == "iterable-iterable-Keypoint":
            assert len(polygons) == 1
            self.polygons_aug = [
                [Keypoint(x=point[0], y=point[1]) for point in polygon.exterior]
                for polygon in polygons[0].polygons]
        elif ntype == "iterable-iterable-Polygon":
            assert len(polygons) == 1
            self.polygons_aug = polygons[0].polygons
        elif ntype == "iterable-iterable-iterable[empty]":
            self.polygons_aug = self.polygons_unaug[:]
        elif ntype == "iterable-iterable-iterable-(x,y)":
            self.polygons_aug = [
                [
                    [
                        (point[0], point[1])
                        for point in polygon.exterior
                    ]
                    for polygon in psoi.polygons
                ]
                for psoi in polygons]
        else:
            assert ntype == "iterable-iterable-iterable-Keypoint"
            self.polygons_aug = [
                [
                    [
                        Keypoint(x=point[0], y=point[1])
                        for point in polygon.exterior
                    ]
                    for polygon in psoi.polygons
                ]
                for psoi in polygons]

    def get_images_unaug_normalized(self):
        attr = self.images_unaug
        if attr is None:
            return None
        elif ia.is_np_array(attr):
            if attr.ndim == 2:
                return attr[np.newaxis, ..., np.newaxis]
            elif attr.ndim == 3:
                return attr[..., np.newaxis]
            else:
                return attr
        elif ia.is_iterable(attr):
            return list(attr)
        raise ValueError(
            ("Expected argument 'images' for Batch to be any of the following: "
             + "None or array or iterable of array. Got type: %s.") % (
                type(self.images_unaug),)
        )

    def get_heatmaps_unaug_normalized(self):
        # TODO get rid of this deferred import
        from imgaug.augmentables.heatmaps import HeatmapsOnImage

        attr = self.heatmaps_unaug
        ntype = self._get_heatmaps_unaug_normalization_type()
        images = self.get_images_unaug_normalized()
        if ntype == "None":
            return None
        elif ntype == "array[float]":
            assert images is not None
            assert attr.ndim == 4  # always (N,H,W,C)
            assert len(attr) == len(images)
            return [HeatmapsOnImage(attr_i, shape=image_i.shape)
                    for attr_i, image_i in zip(attr, images)]
        elif ntype == "HeatmapsOnImage":
            return [attr]
        elif ntype == "iterable[empty]":
            return None
        elif ntype == "iterable-array[float]":
            assert images is not None
            assert len(attr) == len(images)
            assert all([attr_i.ndim == 3 for attr_i in attr])  # all (H,W,C)
            return [HeatmapsOnImage(attr_i, shape=image_i.shape)
                    for attr_i, image_i in zip(attr, images)]
        else:
            assert ntype == "iterable-HeatmapsOnImage"
            return attr  # len allowed to differ from len of images

    def get_segmentation_maps_unaug_normalized(self):
        # TODO get rid of this deferred import
        from imgaug.augmentables.segmaps import SegmentationMapOnImage

        attr = self.segmentation_maps_unaug
        ntype = self._get_segmentation_maps_unaug_normalization_type()
        images = self.get_images_unaug_normalized()

        if ntype == "None":
            return None
        elif ntype in ["array[int]", "array[uint]", "array[bool]"]:
            assert images is not None
            assert attr.ndim == 4  # always (N,H,W,C)
            assert len(attr) == len(images)
            if ntype == "array[bool]":
                return [SegmentationMapOnImage(attr_i, shape=image_i.shape)
                        for attr_i, image_i in zip(attr, images)]
            return [SegmentationMapOnImage(
                        attr_i, shape=image_i.shape, nb_classes=1+np.max(attr_i))
                    for attr_i, image_i in zip(attr, images)]
        elif ntype == "SegmentationMapOnImage":
            return [attr]
        elif ntype == "iterable[empty]":
            return None
        elif ntype in ["iterable-array[int]", "iterable-array[uint]", "iterable-array[bool]"]:
            assert images is not None
            assert len(attr) == len(images)
            assert all([attr_i.ndim == 3 for attr_i in attr])  # all (H,W,C)
            if ntype == "iterable-array[bool]":
                return [SegmentationMapOnImage(attr_i, shape=image_i.shape)
                        for attr_i, image_i in zip(attr, images)]
            return [SegmentationMapOnImage(
                        attr_i, shape=image_i.shape, nb_classes=1+np.max(attr_i))
                    for attr_i, image_i in zip(attr, images)]
        else:
            assert ntype == "iterable-SegmentationMapOnImage"
            return attr  # len allowed to differ from len of images

    def get_keypoints_unaug_normalized(self):
        # TODO get rid of this deferred import
        from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

        attr = self.keypoints_unaug
        ntype = self._get_keypoints_unaug_normalization_type()
        images = self.get_images_unaug_normalized()

        if ntype == "None":
            return attr
        elif ntype in ["array[float]", "array[int]", "array[uint]"]:
            assert images is not None
            assert attr.ndim == 3  # (N,K,2)
            assert attr.shape[2] == 2
            assert len(attr) == len(images)
            return [
                KeypointsOnImage.from_coords_array(attr_i, shape=image_i.shape)
                for attr_i, image_i
                in zip(attr, images)
            ]
        elif ntype == "(x,y)":
            assert images is not None
            assert len(images) == 1
            return [KeypointsOnImage([Keypoint(x=attr[0], y=attr[1])],
                                     shape=images[0].shape)]
        elif ntype == "Keypoint":
            assert images is not None
            assert len(images) == 1
            return [KeypointsOnImage([attr], shape=images[0].shape)]
        elif ntype == "KeypointsOnImage":
            return [attr]
        elif ntype == "iterable[empty]":
            return None
        elif ntype in ["iterable-array[float]", "iterable-array[int]", "iterable-array[uint]"]:
            assert images is not None
            assert all([attr_i.ndim == 2 for attr_i in attr])  # (K,2)
            assert all([attr_i.shape[1] == 2 for attr_i in attr])
            assert len(attr) == len(images)
            return [
                KeypointsOnImage.from_coords_array(attr_i, shape=image_i.shape)
                for attr_i, image_i
                in zip(attr, images)
            ]
        elif ntype == "iterable-(x,y)":
            assert images is not None
            assert len(images) == 1
            return [KeypointsOnImage([Keypoint(x=x, y=y) for x, y in attr],
                                     shape=images[0].shape)]
        elif ntype == "iterable-Keypoint":
            assert images is not None
            assert len(images) == 1
            return [KeypointsOnImage(attr, shape=images[0].shape)]
        elif ntype == "iterable-KeypointsOnImage":
            return attr
        elif ntype == "iterable-iterable[empty]":
            return None
        elif ntype == "iterable-iterable-(x,y)":
            assert images is not None
            assert len(attr) == len(images)
            return [
                KeypointsOnImage.from_coords_array(
                    np.array(attr_i, dtype=np.float32),
                    shape=image_i.shape)
                for attr_i, image_i
                in zip(attr, images)
            ]
        else:
            assert ntype == "iterable-iterable-Keypoint"
            assert images is not None
            assert len(attr) == len(images)
            return [KeypointsOnImage(attr_i, shape=image_i.shape)
                    for attr_i, image_i
                    in zip(attr, images)]

    def get_bounding_boxes_unaug_normalized(self):
        # TODO get rid of this deferred import
        from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

        attr = self.bounding_boxes_unaug
        ntype = self._get_bounding_boxes_unaug_normalization_type()
        images = self.get_images_unaug_normalized()

        if ntype == "None":
            return None
        elif ntype in ["array[float]", "array[int]", "array[uint]"]:
            assert images is not None
            assert attr.ndim == 3  # (N,B,4)
            assert attr.shape[2] == 4
            assert len(attr) == len(images)
            return [
                BoundingBoxesOnImage.from_xyxy_array(attr_i, shape=image_i.shape)
                for attr_i, image_i
                in zip(attr, images)
            ]
        elif ntype == "(x1,y1,x2,y2)":
            assert images is not None
            assert len(images) == 1
            return [
                BoundingBoxesOnImage(
                    [BoundingBox(
                        x1=attr[0], y1=attr[1], x2=attr[2], y2=attr[3])],
                    shape=images[0].shape)
            ]
        elif ntype == "BoundingBox":
            assert images is not None
            assert len(images) == 1
            return [BoundingBoxesOnImage([attr], shape=images[0].shape)]
        elif ntype == "BoundingBoxesOnImage":
            return [attr]
        elif ntype == "iterable[empty]":
            return None
        elif ntype in ["iterable-array[float]", "iterable-array[int]", "iterable-array[uint]"]:
            assert images is not None
            assert all([attr_i.ndim == 2 for attr_i in attr])  # (B,4)
            assert all([attr_i.shape[1] == 4 for attr_i in attr])
            assert len(attr) == len(images)
            return [
                BoundingBoxesOnImage.from_xyxy_array(attr_i, shape=image_i.shape)
                for attr_i, image_i
                in zip(attr, images)
            ]
        elif ntype == "iterable-(x1,y1,x2,y2)":
            assert images is not None
            assert len(images) == 1
            return [
                BoundingBoxesOnImage(
                    [BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2) for x1, y1, x2, y2 in attr],
                    shape=images[0].shape)
            ]
        elif ntype == "iterable-BoundingBox":
            assert images is not None
            assert len(images) == 1
            return [BoundingBoxesOnImage(attr, shape=images[0].shape)]
        elif ntype == "iterable-BoundingBoxesOnImage":
            return attr
        elif ntype == "iterable-iterable[empty]":
            return None
        elif ntype == "iterable-iterable-(x1,y1,x2,y2)":
            assert images is not None
            assert len(attr) == len(images)
            return [
                BoundingBoxesOnImage.from_xyxy_array(
                    np.array(attr_i, dtype=np.float32),
                    shape=image_i.shape)
                for attr_i, image_i
                in zip(attr, images)
            ]
        else:
            assert ntype == "iterable-iterable-BoundingBox"
            assert images is not None
            assert len(attr) == len(images)
            return [BoundingBoxesOnImage(attr_i, shape=image_i.shape)
                    for attr_i, image_i
                    in zip(attr, images)]

    def get_polygons_unaug_normalized(self):
        # TODO get rid of this deferred import
        from imgaug.augmentables.polys import Polygon, PolygonsOnImage

        attr = self.polygons_unaug
        ntype = self._get_polygons_unaug_normalization_type()
        images = self.get_images_unaug_normalized()

        if ntype == "None":
            return None
        elif ntype in ["array[float]", "array[int]", "array[uint]"]:
            assert images is not None
            assert attr.ndim == 4  # (N,#polys,#points,2)
            assert attr.shape[-1] == 2
            assert len(attr) == len(images)
            return [
                PolygonsOnImage(
                    [Polygon(poly_points) for poly_points in attr_i],
                    shape=image_i.shape)
                for attr_i, image_i
                in zip(attr, images)
            ]
        elif ntype == "Polygon":
            assert images is not None
            assert len(images) == 1
            return [PolygonsOnImage([attr], shape=images[0].shape)]
        elif ntype == "PolygonsOnImage":
            return [attr]
        elif ntype == "iterable[empty]":
            return None
        elif ntype in ["iterable-array[float]", "iterable-array[int]", "iterable-array[uint]"]:
            assert images is not None
            assert all([attr_i.ndim == 3 for attr_i in attr])  # (#polys,#points,2)
            assert all([attr_i.shape[-1] == 2 for attr_i in attr])
            assert len(attr) == len(images)
            return [
                PolygonsOnImage([Polygon(poly_points) for poly_points in attr_i],
                                shape=image_i.shape)
                for attr_i, image_i
                in zip(attr, images)
            ]
        elif ntype == "iterable-(x,y)":
            assert images is not None
            assert len(images) == 1
            return [PolygonsOnImage([Polygon(attr)], shape=images[0].shape)]
        elif ntype == "iterable-Keypoint":
            assert images is not None
            assert len(images) == 1
            return [PolygonsOnImage([Polygon(attr)], shape=images[0].shape)]
        elif ntype == "iterable-Polygon":
            assert images is not None
            assert len(images) == 1
            return [PolygonsOnImage(attr, shape=images[0].shape)]
        elif ntype == "iterable-PolygonsOnImage":
            return attr
        elif ntype == "iterable-iterable[empty]":
            return None
        elif ntype in ["iterable-iterable-array[float]", "iterable-iterable-array[int]", "iterable-iterable-array[uint]"]:
            assert images is not None
            assert len(attr) == len(images)
            assert all([poly_points.ndim == 2 and poly_points.shape[-1] == 2
                        for attr_i in attr
                        for poly_points in attr_i])
            return [
                PolygonsOnImage(
                    [Polygon(poly_points) for poly_points in attr_i],
                    shape=image_i.shape)
                for attr_i, image_i in zip(attr, images)
            ]
        elif ntype == "iterable-iterable-(x,y)":
            assert images is not None
            assert len(images) == 1
            return [
                PolygonsOnImage([Polygon(attr_i) for attr_i in attr],
                                shape=images[0].shape)
            ]
        elif ntype == "iterable-iterable-Keypoint":
            assert images is not None
            assert len(images) == 1
            return [
                PolygonsOnImage([Polygon(attr_i) for attr_i in attr],
                                shape=images[0].shape)
            ]
        elif ntype == "iterable-iterable-Polygon":
            assert images is not None
            assert len(attr) == len(images)
            return [
                PolygonsOnImage(attr_i, shape=images[0].shape)
                for attr_i, image_i in zip(attr, images)
            ]
        elif ntype == "iterable-iterable-iterable[empty]":
            return None
        else:
            assert ntype in ["iterable-iterable-iterable-(x,y)",
                             "iterable-iterable-iterable-Keypoint"]
            assert images is not None
            assert len(attr) == len(images)
            return [
                PolygonsOnImage(
                    [Polygon(poly_points) for poly_points in attr_i],
                    shape=image_i.shape)
                for attr_i, image_i in zip(attr, images)
            ]

    def _get_heatmaps_unaug_normalization_type(self):
        nonempty, success, parents = self._find_first_nonempty(self.heatmaps_unaug)
        type_str = self._nonempty_info_to_type_str(nonempty, success, parents)
        valid_type_strs = [
            "None",
            "array[float]",
            "HeatmapsOnImage",
            "iterable[empty]",
            "iterable-array[float]",
            "iterable-HeatmapsOnImage"
        ]
        assert type_str in valid_type_strs, (
            "Got an unknown datatype for argument 'heatmaps' in Batch. "
            "Expected datatypes were: %s. Got: %s." % (
                ", ".join(valid_type_strs), type_str))

        return type_str

    def _get_segmentation_maps_unaug_normalization_type(self):
        nonempty, success, parents = self._find_first_nonempty(self.segmentation_maps_unaug)
        type_str = self._nonempty_info_to_type_str(nonempty, success, parents)
        valid_type_strs = [
            "None",
            "array[int]",
            "array[uint]",
            "array[bool]",
            "SegmentationMapOnImage",
            "iterable[empty]",
            "iterable-array[int]",
            "iterable-array[uint]",
            "iterable-array[bool]",
            "iterable-SegmentationMapOnImage"
        ]
        assert type_str in valid_type_strs, (
            "Got an unknown datatype for argument 'segmentation_maps' in Batch. "
            "Expected datatypes were: %s. Got: %s." % (
                ", ".join(valid_type_strs), type_str))

        return type_str

    def _get_keypoints_unaug_normalization_type(self):
        nonempty, success, parents = self._find_first_nonempty(self.keypoints_unaug)
        type_str = self._nonempty_info_to_type_str(nonempty, success, parents)
        valid_type_strs = [
            "None",
            "array[float]",
            "array[int]",
            "array[uint]",
            "(x,y)",
            "Keypoint",
            "KeypointsOnImage",
            "iterable[empty]",
            "iterable-array[float]",
            "iterable-array[int]",
            "iterable-array[uint]",
            "iterable-(x,y)",
            "iterable-Keypoint",
            "iterable-KeypointsOnImage",
            "iterable-iterable[empty]",
            "iterable-iterable-(x,y)",
            "iterable-iterable-Keypoint"
        ]
        assert type_str in valid_type_strs, (
            "Got an unknown datatype for argument 'keypoints' in Batch. "
            "Expected datatypes were: %s. Got: %s." % (
                ", ".join(valid_type_strs), type_str))

        return type_str

    def _get_bounding_boxes_unaug_normalization_type(self):
        nonempty, success, parents = self._find_first_nonempty(self.bounding_boxes_unaug)
        type_str = self._nonempty_info_to_type_str(nonempty, success, parents, tuple_size=4)
        valid_type_strs = [
            "None",
            "array[float]",
            "array[int]",
            "array[uint]",
            "(x1,y1,x2,y2)",
            "BoundingBox",
            "BoundingBoxesOnImage",
            "iterable[empty]",
            "iterable-array[float]",
            "iterable-array[int]",
            "iterable-array[uint]",
            "iterable-(x1,y1,x2,y2)",
            "iterable-BoundingBox",
            "iterable-BoundingBoxesOnImage",
            "iterable-iterable[empty]",
            "iterable-iterable-(x1,y1,x2,y2)",
            "iterable-iterable-BoundingBox"
        ]
        assert type_str in valid_type_strs, (
            "Got an unknown datatype for argument 'bounding_boxes' in Batch. "
            "Expected datatypes were: %s. Got: %s." % (
                ", ".join(valid_type_strs), type_str))

        return type_str

    def _get_polygons_unaug_normalization_type(self):
        nonempty, success, parents = self._find_first_nonempty(self.polygons_unaug)
        type_str = self._nonempty_info_to_type_str(nonempty, success, parents)
        valid_type_strs = [
            "None",
            "array[float]",
            "array[int]",
            "array[uint]",
            "Polygon",
            "PolygonsOnImage",
            "iterable[empty]",
            "iterable-array[float]",
            "iterable-array[int]",
            "iterable-array[uint]",
            "iterable-(x,y)",
            "iterable-Keypoint",
            "iterable-Polygon",
            "iterable-PolygonsOnImage",
            "iterable-iterable[empty]",
            "iterable-iterable-array[float]",
            "iterable-iterable-array[int]",
            "iterable-iterable-array[uint]",
            "iterable-iterable-(x,y)",
            "iterable-iterable-Keypoint",
            "iterable-iterable-Polygon",
            "iterable-iterable-iterable[empty]",
            "iterable-iterable-iterable-(x,y)",
            "iterable-iterable-iterable-Keypoint"
        ]
        assert type_str in valid_type_strs, (
            "Got an unknown datatype for argument 'polygons' in Batch. "
            "Expected datatypes were: %s. Got: %s." % (
                ", ".join(valid_type_strs), type_str))

        return type_str

    @classmethod
    def _find_first_nonempty(cls, attr, parents=None):
        if parents is None:
            parents = []

        if attr is None or ia.is_np_array(attr):
            return attr, True, parents
        # we exclude strings here, as otherwise we would get the first
        # character, while we want to get the whole string
        elif ia.is_iterable(attr) and not ia.is_string(attr):
            if len(attr) == 0:
                return None, False, parents

            # this prevents the loop below from becoming infinite if the
            # element in the iterable is identical with the iterable,
            # as is the case for e.g. strings
            if attr[0] is attr:
                return attr, True, parents

            # Usually in case of empty lists, all lists should have similar
            # depth. We are a bit more tolerant here and pick the deepest one.
            # Only parents would really need to be tracked here, we could
            # ignore nonempty and success as they will always have the same
            # values (if only empty lists exist).
            nonempty_deepest = None
            success_deepest = False
            parents_deepest = parents
            for attr_i in attr:
                nonempty, success, parents_found = cls._find_first_nonempty(
                    attr_i, parents=parents+[attr])
                if success:
                    # on any nonempty hit we return immediately as we assume
                    # that the datatypes do not change between child branches
                    return nonempty, success, parents_found
                elif len(parents_found) > len(parents_deepest):
                    nonempty_deepest = nonempty
                    success_deepest = success
                    parents_deepest = parents_found

            return nonempty_deepest, success_deepest, parents_deepest

        return attr, True, parents

    @classmethod
    def _nonempty_info_to_type_str(cls, nonempty, success, parents, tuple_size=2):
        assert len(parents) <= 4
        parent_iters = ""
        if len(parents) > 0:
            parent_iters = "%s-" % ("-".join(["iterable"] * len(parents)),)

        if not success:
            return "%siterable[empty]" % (parent_iters,)

        # check if this is an (x, y) tuple
        # if tuple_size=4 (i.e. for BBs) check if it is (x1, y1, x2, y2)
        assert tuple_size in [2, 4]
        if len(parents) >= 1 and isinstance(parents[-1], tuple) \
                and len(parents[-1]) == tuple_size \
                and all([ia.is_single_number(val) for val in parents[-1]]):
            parent_iters = "-".join(["iterable"] * (len(parents)-1))
            if tuple_size == 4:
                return "-".join([parent_iters, "(x1,y1,x2,y2)"]).lstrip("-")
            return "-".join([parent_iters, "(x,y)"]).lstrip("-")

        if nonempty is None:
            return "None"
        elif ia.is_np_array(nonempty):
            kind = nonempty.dtype.kind
            kind_map = {"f": "float", "u": "uint", "i": "int", "b": "bool"}
            return "%sarray[%s]" % (parent_iters, kind_map[kind] if kind in kind_map else kind)

        # even int, str etc. are objects in python, so anything left should
        # offer a __class__ attribute
        assert isinstance(nonempty, object)
        return "%s%s" % (parent_iters, nonempty.__class__.__name__)

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
