from __future__ import print_function, division, absolute_import

import numpy as np

import imgaug.imgaug as ia


def normalize_images(images):
    if images is None:
        return None
    elif ia.is_np_array(images):
        if images.ndim == 2:
            return images[np.newaxis, ..., np.newaxis]
        elif images.ndim == 3:
            return images[..., np.newaxis]
        else:
            return images
    elif ia.is_iterable(images):
        result = []
        for image in images:
            if image.ndim == 2:
                result.append(image[..., np.newaxis])
            else:
                assert image.ndim == 3
                result.append(image)
        return result
    raise ValueError(
        ("Expected argument 'images' to be any of the following: "
         + "None or array or iterable of array. Got type: %s.") % (
            type(images),)
    )


def normalize_heatmaps(inputs, images=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.heatmaps import HeatmapsOnImage

    ntype = estimate_heatmaps_norm_type(inputs)
    if ntype == "None":
        return None
    elif ntype == "array[float]":
        assert images is not None
        assert inputs.ndim == 4  # always (N,H,W,C)
        assert len(inputs) == len(images)
        return [HeatmapsOnImage(attr_i, shape=image_i.shape)
                for attr_i, image_i in zip(inputs, images)]
    elif ntype == "HeatmapsOnImage":
        return [inputs]
    elif ntype == "iterable[empty]":
        return None
    elif ntype == "iterable-array[float]":
        assert images is not None
        assert len(inputs) == len(images)
        assert all([attr_i.ndim == 3 for attr_i in inputs])  # all (H,W,C)
        return [HeatmapsOnImage(attr_i, shape=image_i.shape)
                for attr_i, image_i in zip(inputs, images)]
    else:
        assert ntype == "iterable-HeatmapsOnImage"
        return inputs  # len allowed to differ from len of images


def normalize_segmentation_maps(inputs, images=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.segmaps import SegmentationMapOnImage

    ntype = estimate_segmaps_norm_type(inputs)
    if ntype == "None":
        return None
    elif ntype in ["array[int]", "array[uint]", "array[bool]"]:
        assert images is not None
        assert inputs.ndim == 4  # always (N,H,W,C)
        assert len(inputs) == len(images)
        if ntype == "array[bool]":
            return [SegmentationMapOnImage(attr_i, shape=image_i.shape)
                    for attr_i, image_i in zip(inputs, images)]
        return [SegmentationMapOnImage(
                    attr_i, shape=image_i.shape, nb_classes=1+np.max(attr_i))
                for attr_i, image_i in zip(inputs, images)]
    elif ntype == "SegmentationMapOnImage":
        return [inputs]
    elif ntype == "iterable[empty]":
        return None
    elif ntype in ["iterable-array[int]",
                   "iterable-array[uint]",
                   "iterable-array[bool]"]:
        assert images is not None
        assert len(inputs) == len(images)
        assert all([attr_i.ndim == 3 for attr_i in inputs])  # all (H,W,C)
        if ntype == "iterable-array[bool]":
            return [SegmentationMapOnImage(attr_i, shape=image_i.shape)
                    for attr_i, image_i in zip(inputs, images)]
        return [SegmentationMapOnImage(
                    attr_i, shape=image_i.shape, nb_classes=1+np.max(attr_i))
                for attr_i, image_i in zip(inputs, images)]
    else:
        assert ntype == "iterable-SegmentationMapOnImage"
        return inputs  # len allowed to differ from len of images


def normalize_keypoints(inputs, images=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

    ntype = estimate_keypoints_norm_type(inputs)
    if ntype == "None":
        return inputs
    elif ntype in ["array[float]", "array[int]", "array[uint]"]:
        assert images is not None
        assert inputs.ndim == 3  # (N,K,2)
        assert inputs.shape[2] == 2
        assert len(inputs) == len(images)
        return [
            KeypointsOnImage.from_coords_array(attr_i, shape=image_i.shape)
            for attr_i, image_i
            in zip(inputs, images)
        ]
    elif ntype == "tuple[number,size=2]":
        assert images is not None
        assert len(images) == 1
        return [KeypointsOnImage([Keypoint(x=inputs[0], y=inputs[1])],
                                 shape=images[0].shape)]
    elif ntype == "Keypoint":
        assert images is not None
        assert len(images) == 1
        return [KeypointsOnImage([inputs], shape=images[0].shape)]
    elif ntype == "KeypointsOnImage":
        return [inputs]
    elif ntype == "iterable[empty]":
        return None
    elif ntype in ["iterable-array[float]",
                   "iterable-array[int]",
                   "iterable-array[uint]"]:
        assert images is not None
        assert all([attr_i.ndim == 2 for attr_i in inputs])  # (K,2)
        assert all([attr_i.shape[1] == 2 for attr_i in inputs])
        assert len(inputs) == len(images)
        return [
            KeypointsOnImage.from_coords_array(attr_i, shape=image_i.shape)
            for attr_i, image_i
            in zip(inputs, images)
        ]
    elif ntype == "iterable-tuple[number,size=2]":
        assert images is not None
        assert len(images) == 1
        return [KeypointsOnImage([Keypoint(x=x, y=y) for x, y in inputs],
                                 shape=images[0].shape)]
    elif ntype == "iterable-Keypoint":
        assert images is not None
        assert len(images) == 1
        return [KeypointsOnImage(inputs, shape=images[0].shape)]
    elif ntype == "iterable-KeypointsOnImage":
        return inputs
    elif ntype == "iterable-iterable[empty]":
        return None
    elif ntype == "iterable-iterable-tuple[number,size=2]":
        assert images is not None
        assert len(inputs) == len(images)
        return [
            KeypointsOnImage.from_coords_array(
                np.array(attr_i, dtype=np.float32),
                shape=image_i.shape)
            for attr_i, image_i
            in zip(inputs, images)
        ]
    else:
        assert ntype == "iterable-iterable-Keypoint"
        assert images is not None
        assert len(inputs) == len(images)
        return [KeypointsOnImage(attr_i, shape=image_i.shape)
                for attr_i, image_i
                in zip(inputs, images)]


def normalize_bounding_boxes(inputs, images=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

    ntype = estimate_bounding_boxes_norm_type(inputs)
    if ntype == "None":
        return None
    elif ntype in ["array[float]", "array[int]", "array[uint]"]:
        assert images is not None
        assert inputs.ndim == 3  # (N,B,4)
        assert inputs.shape[2] == 4
        assert len(inputs) == len(images)
        return [
            BoundingBoxesOnImage.from_xyxy_array(attr_i, shape=image_i.shape)
            for attr_i, image_i
            in zip(inputs, images)
        ]
    elif ntype == "tuple[number,size=4]":
        assert images is not None
        assert len(images) == 1
        return [
            BoundingBoxesOnImage(
                [BoundingBox(
                    x1=inputs[0], y1=inputs[1],
                    x2=inputs[2], y2=inputs[3])],
                shape=images[0].shape)
        ]
    elif ntype == "BoundingBox":
        assert images is not None
        assert len(images) == 1
        return [BoundingBoxesOnImage([inputs], shape=images[0].shape)]
    elif ntype == "BoundingBoxesOnImage":
        return [inputs]
    elif ntype == "iterable[empty]":
        return None
    elif ntype in ["iterable-array[float]",
                   "iterable-array[int]",
                   "iterable-array[uint]"]:
        assert images is not None
        assert all([attr_i.ndim == 2 for attr_i in inputs])  # (B,4)
        assert all([attr_i.shape[1] == 4 for attr_i in inputs])
        assert len(inputs) == len(images)
        return [
            BoundingBoxesOnImage.from_xyxy_array(attr_i, shape=image_i.shape)
            for attr_i, image_i
            in zip(inputs, images)
        ]
    elif ntype == "iterable-tuple[number,size=4]":
        assert images is not None
        assert len(images) == 1
        return [
            BoundingBoxesOnImage(
                [BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                 for x1, y1, x2, y2 in inputs],
                shape=images[0].shape)
        ]
    elif ntype == "iterable-BoundingBox":
        assert images is not None
        assert len(images) == 1
        return [BoundingBoxesOnImage(inputs, shape=images[0].shape)]
    elif ntype == "iterable-BoundingBoxesOnImage":
        return inputs
    elif ntype == "iterable-iterable[empty]":
        return None
    elif ntype == "iterable-iterable-tuple[number,size=4]":
        assert images is not None
        assert len(inputs) == len(images)
        return [
            BoundingBoxesOnImage.from_xyxy_array(
                np.array(attr_i, dtype=np.float32),
                shape=image_i.shape)
            for attr_i, image_i
            in zip(inputs, images)
        ]
    else:
        assert ntype == "iterable-iterable-BoundingBox"
        assert images is not None
        assert len(inputs) == len(images)
        return [BoundingBoxesOnImage(attr_i, shape=image_i.shape)
                for attr_i, image_i
                in zip(inputs, images)]


def normalize_polygons(inputs, images=None):
    # TODO get rid of this deferred import
    from imgaug.augmentables.polys import Polygon, PolygonsOnImage

    ntype = estimate_polygons_norm_type(inputs)
    if ntype == "None":
        return None
    elif ntype in ["array[float]", "array[int]", "array[uint]"]:
        assert images is not None
        assert inputs.ndim == 4  # (N,#polys,#points,2)
        assert inputs.shape[-1] == 2
        assert len(inputs) == len(images)
        return [
            PolygonsOnImage(
                [Polygon(poly_points) for poly_points in attr_i],
                shape=image_i.shape)
            for attr_i, image_i
            in zip(inputs, images)
        ]
    elif ntype == "Polygon":
        assert images is not None
        assert len(images) == 1
        return [PolygonsOnImage([inputs], shape=images[0].shape)]
    elif ntype == "PolygonsOnImage":
        return [inputs]
    elif ntype == "iterable[empty]":
        return None
    elif ntype in ["iterable-array[float]",
                   "iterable-array[int]",
                   "iterable-array[uint]"]:
        assert images is not None
        assert all([attr_i.ndim == 3 for attr_i in inputs])  # (#polys,#points,2)
        assert all([attr_i.shape[-1] == 2 for attr_i in inputs])
        assert len(inputs) == len(images)
        return [
            PolygonsOnImage([Polygon(poly_points) for poly_points in attr_i],
                            shape=image_i.shape)
            for attr_i, image_i
            in zip(inputs, images)
        ]
    elif ntype == "iterable-tuple[number,size=2]":
        assert images is not None
        assert len(images) == 1
        return [PolygonsOnImage([Polygon(inputs)], shape=images[0].shape)]
    elif ntype == "iterable-Keypoint":
        assert images is not None
        assert len(images) == 1
        return [PolygonsOnImage([Polygon(inputs)], shape=images[0].shape)]
    elif ntype == "iterable-Polygon":
        assert images is not None
        assert len(images) == 1
        return [PolygonsOnImage(inputs, shape=images[0].shape)]
    elif ntype == "iterable-PolygonsOnImage":
        return inputs
    elif ntype == "iterable-iterable[empty]":
        return None
    elif ntype in ["iterable-iterable-array[float]",
                   "iterable-iterable-array[int]",
                   "iterable-iterable-array[uint]"]:
        assert images is not None
        assert len(inputs) == len(images)
        assert all([poly_points.ndim == 2 and poly_points.shape[-1] == 2
                    for attr_i in inputs
                    for poly_points in attr_i])
        return [
            PolygonsOnImage(
                [Polygon(poly_points) for poly_points in attr_i],
                shape=image_i.shape)
            for attr_i, image_i in zip(inputs, images)
        ]
    elif ntype == "iterable-iterable-tuple[number,size=2]":
        assert images is not None
        assert len(images) == 1
        return [
            PolygonsOnImage([Polygon(attr_i) for attr_i in inputs],
                            shape=images[0].shape)
        ]
    elif ntype == "iterable-iterable-Keypoint":
        assert images is not None
        assert len(images) == 1
        return [
            PolygonsOnImage([Polygon(attr_i) for attr_i in inputs],
                            shape=images[0].shape)
        ]
    elif ntype == "iterable-iterable-Polygon":
        assert images is not None
        assert len(inputs) == len(images)
        return [
            PolygonsOnImage(attr_i, shape=images[0].shape)
            for attr_i, image_i in zip(inputs, images)
        ]
    elif ntype == "iterable-iterable-iterable[empty]":
        return None
    else:
        assert ntype in ["iterable-iterable-iterable-tuple[number,size=2]",
                         "iterable-iterable-iterable-Keypoint"]
        assert images is not None
        assert len(inputs) == len(images)
        return [
            PolygonsOnImage(
                [Polygon(poly_points) for poly_points in attr_i],
                shape=image_i.shape)
            for attr_i, image_i in zip(inputs, images)
        ]


def invert_normalize_images(images, images_old):
    if images_old is None:
        assert images is None
        return None
    elif ia.is_np_array(images_old):
        if images_old.ndim == 2:
            assert images.shape[0] == 1
            assert images.shape[3] == 1
            return images[0, ..., 0]
        elif images_old.ndim == 3:
            assert images.shape[3] == 1
            return images[..., 0]
        else:
            return images
    elif ia.is_iterable(images_old):
        result = []
        for image, image_old in zip(images, images_old):
            if image_old.ndim == 2:
                assert image.shape[2] == 1
                result.append(image[:, :, 0])
            else:
                assert image.ndim == 3
                assert image_old.ndim == 3
                result.append(image)
        return result
    raise ValueError(
        ("Expected argument 'images_old' to be any of the following: "
         + "None or array or iterable of array. Got type: %s.") % (
            type(images_old),)
    )


def invert_normalize_heatmaps(heatmaps, heatmaps_old):
    ntype = estimate_heatmaps_norm_type(heatmaps_old)
    if ntype == "None":
        assert heatmaps is None
        return heatmaps
    elif ntype == "array[float]":
        assert len(heatmaps) == heatmaps_old.shape[0]
        input_dtype = heatmaps_old.dtype
        return restore_dtype_and_merge(
            [hm_i.arr_0to1 for hm_i in heatmaps],
            input_dtype)
    elif ntype == "HeatmapsOnImage":
        assert len(heatmaps) == 1
        return heatmaps[0]
    elif ntype == "iterable[empty]":
        assert heatmaps is None
        return []
    elif ntype == "iterable-array[float]":
        nonempty, _, _ = find_first_nonempty(heatmaps_old)
        input_dtype = nonempty.dtype
        return [restore_dtype_and_merge(hm_i.arr_0to1, input_dtype)
                for hm_i in heatmaps]
    else:
        assert ntype == "iterable-HeatmapsOnImage"
        return heatmaps


def invert_normalize_segmentation_maps(segmentation_maps,
                                       segmentation_maps_old):
    ntype = estimate_segmaps_norm_type(segmentation_maps_old)
    if ntype == "None":
        assert segmentation_maps is None
        return segmentation_maps
    elif ntype in ["array[int]", "array[uint]", "array[bool]"]:
        assert len(segmentation_maps) == 1
        return segmentation_maps[0].arr
    elif ntype == "SegmentationMapOnImage":
        assert len(segmentation_maps) == 1
        return segmentation_maps[0]
    elif ntype == "iterable[empty]":
        assert segmentation_maps is None
        return []
    elif ntype in ["iterable-array[int]",
                   "iterable-array[uint]",
                   "iterable-array[bool]"]:
        return [segmap_i.arr for segmap_i in segmentation_maps]
    else:
        assert ntype == "iterable-SegmentationMapOnImage"
        return segmentation_maps


def invert_normalize_keypoints(keypoints, keypoints_old):
    ntype = estimate_keypoints_norm_type(keypoints_old)
    if ntype == "None":
        assert keypoints is None
        return keypoints
    elif ntype in ["array[float]", "array[int]", "array[uint]"]:
        assert len(keypoints) == 1
        input_dtype = keypoints_old.dtype
        return restore_dtype_and_merge(
            [kpsoi.get_coords_array() for kpsoi in keypoints],
            input_dtype)
    elif ntype == "tuple[number,size=2]":
        assert len(keypoints) == 1
        assert len(keypoints[0].keypoints) == 1
        return (keypoints[0].keypoints[0].x,
                keypoints[0].keypoints[0].y)
    elif ntype == "Keypoint":
        assert len(keypoints) == 1
        assert len(keypoints[0].keypoints) == 1
        return keypoints[0].keypoints[0]
    elif ntype == "KeypointsOnImage":
        assert len(keypoints) == 1
        return keypoints[0]
    elif ntype == "iterable[empty]":
        assert keypoints is None
        return []
    elif ntype in ["iterable-array[float]",
                   "iterable-array[int]",
                   "iterable-array[uint]"]:
        nonempty, _, _ = find_first_nonempty(keypoints_old)
        input_dtype = nonempty.dtype
        return [
            restore_dtype_and_merge(kps_i.get_coords_array(),
                                    input_dtype)
            for kps_i in keypoints]
    elif ntype == "iterable-tuple[number,size=2]":
        assert len(keypoints) == 1
        return [
            (kp.x, kp.y) for kp in keypoints[0].keypoints]
    elif ntype == "iterable-KeypointsOnImage":
        return keypoints
    elif ntype == "iterable-iterable[empty]":
        assert keypoints is None
        return keypoints_old[:]
    elif ntype == "iterable-iterable-tuple[number,size=2]":
        return [
            [(kp.x, kp.y) for kp in kpsoi.keypoints]
            for kpsoi in keypoints]
    else:
        assert ntype == "iterable-iterable-Keypoint"
        return [
            [kp for kp in kpsoi.keypoints]
            for kpsoi in keypoints]


def invert_normalize_bounding_boxes(bounding_boxes, bounding_boxes_old):
    ntype = estimate_normalization_type(bounding_boxes_old)
    if ntype == "None":
        assert bounding_boxes is None
        return bounding_boxes
    elif ntype in ["array[float]", "array[int]", "array[uint]"]:
        assert len(bounding_boxes) == 1
        input_dtype = bounding_boxes_old.dtype
        return restore_dtype_and_merge([
            bbsoi.to_xyxy_array() for bbsoi in bounding_boxes
        ], input_dtype)
    elif ntype == "tuple[number,size=4]":
        assert len(bounding_boxes) == 1
        assert len(bounding_boxes[0].bounding_boxes) == 1
        bb = bounding_boxes[0].bounding_boxes[0]
        return bb.x1, bb.y1, bb.x2, bb.y2
    elif ntype == "BoundingBox":
        assert len(bounding_boxes) == 1
        assert len(bounding_boxes[0].bounding_boxes) == 1
        return bounding_boxes[0].bounding_boxes[0]
    elif ntype == "BoundingBoxesOnImage":
        assert len(bounding_boxes) == 1
        return bounding_boxes[0]
    elif ntype == "iterable[empty]":
        assert bounding_boxes is None
        return []
    elif ntype in ["iterable-array[float]",
                   "iterable-array[int]",
                   "iterable-array[uint]"]:
        nonempty, _, _ = find_first_nonempty(bounding_boxes_old)
        input_dtype = nonempty.dtype
        return [
            restore_dtype_and_merge(bbsoi.to_xyxy_array(), input_dtype)
            for bbsoi in bounding_boxes]
    elif ntype == "iterable-tuple[number,size=4]":
        assert len(bounding_boxes) == 1
        return [
            (bb.x1, bb.y1, bb.x2, bb.y2)
            for bb in bounding_boxes[0].bounding_boxes]
    elif ntype == "iterable-BoundingBoxesOnImage":
        return bounding_boxes
    elif ntype == "iterable-iterable[empty]":
        assert bounding_boxes is None
        return bounding_boxes_old[:]
    elif ntype == "iterable-iterable-tuple[number,size=4]":
        return [
            [(bb.x1, bb.y1, bb.x2, bb.y2) for bb in bbsoi.bounding_boxes]
            for bbsoi in bounding_boxes]
    else:
        assert ntype == "iterable-iterable-BoundingBox"
        return [
            [bb for bb in bbsoi.bounding_boxes]
            for bbsoi in bounding_boxes]


def invert_normalize_polygons(polygons, polygons_old):
    # TODO get rid of this deferred import
    from imgaug.augmentables.kps import Keypoint

    ntype = estimate_polygons_norm_type(polygons_old)
    if ntype == "None":
        assert polygons is None
        return polygons
    elif ntype in ["array[float]", "array[int]", "array[uint]"]:
        input_dtype = polygons_old.dtype
        return restore_dtype_and_merge([
            [poly.exterior for poly in psoi.polygons]
            for psoi in polygons
        ], input_dtype)
    elif ntype == "Polygon":
        assert len(polygons) == 1
        assert len(polygons[0].polygons) == 1
        return polygons[0].polygons[0]
    elif ntype == "PolygonsOnImage":
        assert len(polygons) == 1
        return polygons[0]
    elif ntype == "iterable[empty]":
        assert polygons is None
        return []
    elif ntype in ["iterable-array[float]",
                   "iterable-array[int]",
                   "iterable-array[uint]"]:
        nonempty, _, _ = find_first_nonempty(polygons_old)
        input_dtype = nonempty.dtype
        return [
            restore_dtype_and_merge(
                [poly.exterior for poly in psoi.poylgons],
                input_dtype)
            for psoi in polygons
        ]
    elif ntype == "iterable-tuple[number,size=2]":
        assert len(polygons) == 1
        assert len(polygons[0].polygons) == 1
        return [(point.x, point.y)
                for point in polygons[0].polygons[0].exterior]
    elif ntype == "iterable-Keypoint":
        assert len(polygons) == 1
        assert len(polygons[0].polygons) == 1
        return [Keypoint(x=point.x, y=point.y)
                for point in polygons[0].polygons[0].exterior]
    elif ntype == "iterable-Polygon":
        assert len(polygons) == 1
        assert len(polygons[0].polygons) == len(polygons_old[0].polygons)
        return polygons[0].polygons
    elif ntype == "iterable-PolygonsOnImage":
        return polygons
    elif ntype == "iterable-iterable[empty]":
        assert polygons is None
        return polygons[:]
    elif ntype in ["iterable-iterable-array[float]",
                   "iterable-iterable-array[int]",
                   "iterable-iterable-array[uint]"]:
        nonempty, _, _ = find_first_nonempty(polygons_old)
        input_dtype = nonempty.dtype
        return [
            [restore_dtype_and_merge(poly.exterior, input_dtype)
             for poly in psoi.polygons]
            for psoi in polygons
        ]
    elif ntype == "iterable-iterable-tuple[number,size=2]":
        assert len(polygons) == 1
        return [
            [(point[0], point[1]) for point in polygon.exterior]
            for polygon in polygons[0].polygons]
    elif ntype == "iterable-iterable-Keypoint":
        assert len(polygons) == 1
        return [
            [Keypoint(x=point[0], y=point[1]) for point in polygon.exterior]
            for polygon in polygons[0].polygons]
    elif ntype == "iterable-iterable-Polygon":
        assert len(polygons) == 1
        return polygons[0].polygons
    elif ntype == "iterable-iterable-iterable[empty]":
        return polygons_old[:]
    elif ntype == "iterable-iterable-iterable-tuple[number,size=2]":
        return [
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
        return [
            [
                [
                    Keypoint(x=point[0], y=point[1])
                    for point in polygon.exterior
                ]
                for polygon in psoi.polygons
            ]
            for psoi in polygons]


def _assert_is_of_norm_type(type_str, valid_type_strs, arg_name):
    assert type_str in valid_type_strs, (
        "Got an unknown datatype for argument '%s'. "
        "Expected datatypes were: %s. Got: %s." % (
            arg_name, ", ".join(valid_type_strs), type_str))


def estimate_heatmaps_norm_type(heatmaps):
    type_str = estimate_normalization_type(heatmaps)
    valid_type_strs = [
        "None",
        "array[float]",
        "HeatmapsOnImage",
        "iterable[empty]",
        "iterable-array[float]",
        "iterable-HeatmapsOnImage"
    ]
    _assert_is_of_norm_type(type_str, valid_type_strs, "heatmaps")
    return type_str


def estimate_segmaps_norm_type(segmentation_maps):
    type_str = estimate_normalization_type(segmentation_maps)
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
    _assert_is_of_norm_type(
        type_str, valid_type_strs, "segmentation_maps")
    return type_str


def estimate_keypoints_norm_type(keypoints):
    type_str = estimate_normalization_type(keypoints)
    valid_type_strs = [
        "None",
        "array[float]",
        "array[int]",
        "array[uint]",
        "tuple[number,size=2]",
        "Keypoint",
        "KeypointsOnImage",
        "iterable[empty]",
        "iterable-array[float]",
        "iterable-array[int]",
        "iterable-array[uint]",
        "iterable-tuple[number,size=2]",
        "iterable-Keypoint",
        "iterable-KeypointsOnImage",
        "iterable-iterable[empty]",
        "iterable-iterable-tuple[number,size=2]",
        "iterable-iterable-Keypoint"
    ]
    _assert_is_of_norm_type(type_str, valid_type_strs, "keypoints")
    return type_str


def estimate_bounding_boxes_norm_type(bounding_boxes):
    type_str = estimate_normalization_type(bounding_boxes)
    valid_type_strs = [
        "None",
        "array[float]",
        "array[int]",
        "array[uint]",
        "tuple[number,size=4]",
        "BoundingBox",
        "BoundingBoxesOnImage",
        "iterable[empty]",
        "iterable-array[float]",
        "iterable-array[int]",
        "iterable-array[uint]",
        "iterable-tuple[number,size=4]",
        "iterable-BoundingBox",
        "iterable-BoundingBoxesOnImage",
        "iterable-iterable[empty]",
        "iterable-iterable-tuple[number,size=4]",
        "iterable-iterable-BoundingBox"
    ]
    _assert_is_of_norm_type(
        type_str, valid_type_strs, "bounding_boxes")
    return type_str


def estimate_polygons_norm_type(polygons):
    type_str = estimate_normalization_type(polygons)
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
        "iterable-tuple[number,size=2]",
        "iterable-Keypoint",
        "iterable-Polygon",
        "iterable-PolygonsOnImage",
        "iterable-iterable[empty]",
        "iterable-iterable-array[float]",
        "iterable-iterable-array[int]",
        "iterable-iterable-array[uint]",
        "iterable-iterable-tuple[number,size=2]",
        "iterable-iterable-Keypoint",
        "iterable-iterable-Polygon",
        "iterable-iterable-iterable[empty]",
        "iterable-iterable-iterable-tuple[number,size=2]",
        "iterable-iterable-iterable-Keypoint"
    ]
    _assert_is_of_norm_type(type_str, valid_type_strs, "polygons")
    return type_str


def estimate_normalization_type(inputs):
    nonempty, success, parents = find_first_nonempty(inputs)
    type_str = _nonempty_info_to_type_str(nonempty, success, parents)
    return type_str


# TODO replace partially with dtypes.restore_dtypes_()
def restore_dtype_and_merge(arr, input_dtype):
    if isinstance(arr, list):
        arr = [restore_dtype_and_merge(arr_i, input_dtype)
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


def find_first_nonempty(attr, parents=None):
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
            nonempty, success, parents_found = find_first_nonempty(
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


def _nonempty_info_to_type_str(nonempty, success, parents):
    assert len(parents) <= 4
    parent_iters = ""
    if len(parents) > 0:
        parent_iters = "%s-" % ("-".join(["iterable"] * len(parents)),)

    if not success:
        return "%siterable[empty]" % (parent_iters,)

    # check if this is an (x, y) tuple
    # if tuple_size=4 (i.e. for BBs) check if it is (x1, y1, x2, y2)
    """assert tuple_size in [2, 4]
    if len(parents) >= 1 and isinstance(parents[-1], tuple) \
            and len(parents[-1]) == tuple_size \
            and all([ia.is_single_number(val) for val in parents[-1]]):
        parent_iters = "-".join(["iterable"] * (len(parents)-1))
        if tuple_size == 4:
            return "-".join([parent_iters, "(x1,y1,x2,y2)"]).lstrip("-")
        return "-".join([parent_iters, "(x,y)"]).lstrip("-")"""

    is_parent_tuple = (
        len(parents) >= 1
        and isinstance(parents[-1], tuple)
    )

    if is_parent_tuple:
        is_only_numbers_in_tuple = (
            len(parents[-1]) > 0
            and all([ia.is_single_number(val) for val in parents[-1]])
        )

        if is_only_numbers_in_tuple:
            parent_iters = "-".join(["iterable"] * (len(parents)-1))
            tpl_name = "tuple[number,size=%d]" % (len(parents[-1]),)
            return "-".join([parent_iters, tpl_name]).lstrip("-")

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
