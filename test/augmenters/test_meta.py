from __future__ import print_function, division, absolute_import

import os
import warnings
import sys
import itertools
import copy
from abc import ABCMeta, abstractmethod
# unittest only added in 3.4 self.subTest()
if sys.version_info[0] < 3 or sys.version_info[1] < 4:
    import unittest2 as unittest
else:
    import unittest
# unittest.mock is not available in 2.7 (though unittest2 might contain it?)
try:
    import unittest.mock as mock
except ImportError:
    import mock
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import six
import six.moves as sm
import cv2
import PIL.Image
import imageio

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug import random as iarandom
from imgaug.testutils import (create_random_images, create_random_keypoints,
                              array_equal_lists, keypoints_equal, reseed,
                              assert_cbaois_equal,
                              runtest_pickleable_uint8_img,
                              TemporaryDirectory, is_parameter_instance)
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.polys import _ConcavePolygonRecoverer
from imgaug.augmentables.batches import _BatchInAugmentation


IS_PY36_OR_HIGHER = (sys.version_info[0] == 3 and sys.version_info[1] >= 6)


class _InplaceDummyAugmenterImgsArray(iaa.meta.Augmenter):
    def __init__(self, addval):
        super(_InplaceDummyAugmenterImgsArray, self).__init__()
        self.addval = addval

    def _augment_batch_(self, batch, random_state, parents, hooks):
        batch.images += self.addval
        return batch

    def get_parameters(self):
        return []


class _InplaceDummyAugmenterImgsList(iaa.meta.Augmenter):
    def __init__(self, addval):
        super(_InplaceDummyAugmenterImgsList, self).__init__()
        self.addval = addval

    def _augment_batch_(self, batch, random_state, parents, hooks):
        assert len(batch.images) > 0
        for i in range(len(batch.images)):
            batch.images[i] += self.addval
        return batch

    def get_parameters(self):
        return []


class _InplaceDummyAugmenterSegMaps(iaa.meta.Augmenter):
    def __init__(self, addval):
        super(_InplaceDummyAugmenterSegMaps, self).__init__()
        self.addval = addval

    def _augment_batch_(self, batch, random_state, parents, hooks):
        assert len(batch.segmentation_maps) > 0
        for i in range(len(batch.segmentation_maps)):
            batch.segmentation_maps[i].arr += self.addval
        return batch

    def get_parameters(self):
        return []


class _InplaceDummyAugmenterKeypoints(iaa.meta.Augmenter):
    def __init__(self, x, y):
        super(_InplaceDummyAugmenterKeypoints, self).__init__()
        self.x = x
        self.y = y

    def _augment_batch_(self, batch, random_state, parents, hooks):
        assert len(batch.keypoints) > 0
        for i in range(len(batch.keypoints)):
            kpsoi = batch.keypoints[i]
            for j in range(len(kpsoi)):
                batch.keypoints[i].keypoints[j].x += self.x
                batch.keypoints[i].keypoints[j].y += self.y
        return batch

    def get_parameters(self):
        return []


class TestIdentity(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_images(self):
        aug = iaa.Identity()
        images = create_random_images((16, 70, 50, 3))

        observed = aug.augment_images(images)

        expected = images
        assert np.array_equal(observed, expected)

    def test_images_deterministic(self):
        aug_det = iaa.Identity().to_deterministic()
        images = create_random_images((16, 70, 50, 3))

        observed = aug_det.augment_images(images)

        expected = images
        assert np.array_equal(observed, expected)

    def test_heatmaps(self):
        aug = iaa.Identity()
        heatmaps_arr = np.linspace(0.0, 1.0, 2*2, dtype="float32")\
            .reshape((2, 2, 1))
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(2, 2, 3))

        observed = aug.augment_heatmaps(heatmaps)

        assert np.allclose(observed.arr_0to1, heatmaps.arr_0to1)

    def test_heatmaps_deterministic(self):
        aug_det = iaa.Identity().to_deterministic()
        heatmaps_arr = np.linspace(0.0, 1.0, 2*2, dtype="float32")\
            .reshape((2, 2, 1))
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(2, 2, 3))

        observed = aug_det.augment_heatmaps(heatmaps)

        assert np.allclose(observed.arr_0to1, heatmaps.arr_0to1)

    def test_segmentation_maps(self):
        aug = iaa.Identity()
        segmaps_arr = np.arange(2*2).reshape((2, 2, 1)).astype(np.int32)
        segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(2, 2, 3))

        observed = aug.augment_segmentation_maps(segmaps)

        assert np.array_equal(observed.arr, segmaps.arr)

    def test_segmentation_maps_deterministic(self):
        aug_det = iaa.Identity().to_deterministic()
        segmaps_arr = np.arange(2*2).reshape((2, 2, 1)).astype(np.int32)
        segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(2, 2, 3))

        observed = aug_det.augment_segmentation_maps(segmaps)

        assert np.array_equal(observed.arr, segmaps.arr)

    def test_keypoints(self):
        aug = iaa.Identity()
        keypoints = create_random_keypoints((16, 70, 50, 3), 4)

        observed = aug.augment_keypoints(keypoints)

        assert_cbaois_equal(observed, keypoints)

    def test_keypoints_deterministic(self):
        aug_det = iaa.Identity().to_deterministic()
        keypoints = create_random_keypoints((16, 70, 50, 3), 4)

        observed = aug_det.augment_keypoints(keypoints)

        assert_cbaois_equal(observed, keypoints)

    def test_polygons(self):
        aug = iaa.Identity()
        polygon = ia.Polygon([(10, 10), (30, 10), (30, 50), (10, 50)])
        psoi = ia.PolygonsOnImage([polygon], shape=(100, 75, 3))

        observed = aug.augment_polygons(psoi)

        assert_cbaois_equal(observed, psoi)

    def test_polygons_deterministic(self):
        aug_det = iaa.Identity().to_deterministic()
        polygon = ia.Polygon([(10, 10), (30, 10), (30, 50), (10, 50)])
        psoi = ia.PolygonsOnImage([polygon], shape=(100, 75, 3))

        observed = aug_det.augment_polygons(psoi)

        assert_cbaois_equal(observed, psoi)

    def test_line_strings(self):
        aug = iaa.Identity()
        ls = LineString([(10, 10), (30, 10), (30, 50), (10, 50)])
        lsoi = LineStringsOnImage([ls], shape=(100, 75, 3))

        observed = aug.augment_line_strings(lsoi)

        assert_cbaois_equal(observed, lsoi)

    def test_line_strings_deterministic(self):
        aug_det = iaa.Identity().to_deterministic()
        ls = LineString([(10, 10), (30, 10), (30, 50), (10, 50)])
        lsoi = LineStringsOnImage([ls], shape=(100, 75, 3))

        observed = aug_det.augment_line_strings(lsoi)

        assert_cbaois_equal(observed, lsoi)

    def test_bounding_boxes(self):
        aug = iaa.Identity()
        bbs = ia.BoundingBox(x1=10, y1=10, x2=30, y2=50)
        bbsoi = ia.BoundingBoxesOnImage([bbs], shape=(100, 75, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        assert_cbaois_equal(observed, bbsoi)

    def test_bounding_boxes_deterministic(self):
        aug_det = iaa.Identity().to_deterministic()
        bbs = ia.BoundingBox(x1=10, y1=10, x2=30, y2=50)
        bbsoi = ia.BoundingBoxesOnImage([bbs], shape=(100, 75, 3))

        observed = aug_det.augment_bounding_boxes(bbsoi)

        assert_cbaois_equal(observed, bbsoi)

    def test_keypoints_empty(self):
        aug = iaa.Identity()
        kpsoi = ia.KeypointsOnImage([], shape=(4, 5, 3))

        observed = aug.augment_keypoints(kpsoi)

        assert_cbaois_equal(observed, kpsoi)

    def test_polygons_empty(self):
        aug = iaa.Identity()
        psoi = ia.PolygonsOnImage([], shape=(4, 5, 3))

        observed = aug.augment_polygons(psoi)

        assert_cbaois_equal(observed, psoi)

    def test_line_strings_empty(self):
        aug = iaa.Identity()
        lsoi = ia.LineStringsOnImage([], shape=(4, 5, 3))

        observed = aug.augment_line_strings(lsoi)

        assert_cbaois_equal(observed, lsoi)

    def test_bounding_boxes_empty(self):
        aug = iaa.Identity()
        bbsoi = ia.BoundingBoxesOnImage([], shape=(4, 5, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        assert_cbaois_equal(observed, bbsoi)

    def test_get_parameters(self):
        assert iaa.Identity().get_parameters() == []

    def test_other_dtypes_bool(self):
        aug = iaa.Identity()
        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.type == image.dtype.type
        assert np.all(image_aug == image)

    def test_other_dtypes_uint_int(self):
        aug = iaa.Identity()

        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, image)

    def test_other_dtypes_float(self):
        aug = iaa.Identity()

        try:
            f128 = [np.dtype("float128").name]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128
        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]

        for dtype, value in zip(dtypes, values):
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.all(image_aug == image)

    def test_pickleable(self):
        aug = iaa.Noop()
        runtest_pickleable_uint8_img(aug, iterations=2)


class TestNoop(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        aug = iaa.Noop()
        assert isinstance(aug, iaa.Identity)

    def test_images(self):
        image = np.mod(np.arange(10*10*3), 255)
        image = image.astype(np.uint8).reshape((10, 10, 3))

        image_aug = iaa.Noop()(image=image)

        assert np.array_equal(image, image_aug)


# TODO add tests for line strings
class TestLambda(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def base_img(self):
        base_img = np.array([[0, 0, 1],
                             [0, 0, 1],
                             [0, 1, 1]], dtype=np.uint8)
        base_img = base_img[:, :, np.newaxis]
        return base_img

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 1.0, 1.0]])
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))
        return heatmaps

    @property
    def heatmaps_aug(self):
        heatmaps_arr_aug = np.float32([[0.5, 0.0, 1.0],
                                       [0.0, 0.0, 1.0],
                                       [0.0, 1.0, 1.0]])
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr_aug, shape=(3, 3, 3))
        return heatmaps

    @property
    def segmentation_maps(self):
        segmaps_arr = np.int32([[0, 0, 1],
                                [0, 0, 1],
                                [0, 1, 1]])
        segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))
        return segmaps

    @property
    def segmentation_maps_aug(self):
        segmaps_arr_aug = np.int32([[1, 1, 2],
                                    [1, 1, 2],
                                    [1, 2, 2]])
        segmaps = SegmentationMapsOnImage(segmaps_arr_aug, shape=(3, 3, 3))
        return segmaps

    @property
    def keypoints(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        kpsoi = [ia.KeypointsOnImage(kps, shape=(3, 3, 3))]
        return kpsoi

    @property
    def keypoints_aug(self):
        expected_kps = [ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=1),
                        ia.Keypoint(x=0, y=2)]
        expected = [ia.KeypointsOnImage(expected_kps, shape=(3, 3, 3))]
        return expected

    @property
    def polygons(self):
        poly = ia.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        psois = [ia.PolygonsOnImage([poly], shape=(3, 3, 3))]
        return psois

    @property
    def polygons_aug(self):
        expected_poly = ia.Polygon([(1, 2), (3, 2), (3, 4), (1, 4)])
        expected_psoi = [ia.PolygonsOnImage([expected_poly], shape=(3, 3, 3))]
        return expected_psoi

    @property
    def lsoi(self):
        ls = ia.LineString([(0, 0), (2, 0), (2, 2), (0, 2)])
        lsois = [ia.LineStringsOnImage([ls], shape=(3, 3, 3))]
        return lsois

    @property
    def lsoi_aug(self):
        ls = ia.LineString([(1, 2), (3, 2), (3, 4), (1, 4)])
        lsois = [ia.LineStringsOnImage([ls], shape=(3, 3, 3))]
        return lsois

    @property
    def bbsoi(self):
        bb = ia.BoundingBox(x1=0, y1=1, x2=3, y2=4)
        bbsois = [ia.BoundingBoxesOnImage([bb], shape=(3, 3, 3))]
        return bbsois

    @property
    def bbsoi_aug(self):
        bb = ia.BoundingBox(x1=0+1, y1=1+2, x2=3+1, y2=4+2)
        bbsois = [ia.BoundingBoxesOnImage([bb], shape=(3, 3, 3))]
        return bbsois

    @classmethod
    def func_images(cls, images, random_state, parents, hooks):
        if isinstance(images, list):
            images = [image + 1 for image in images]
        else:
            images = images + 1
        return images

    @classmethod
    def func_heatmaps(cls, heatmaps, random_state, parents, hooks):
        heatmaps[0].arr_0to1[0, 0] += 0.5
        return heatmaps

    @classmethod
    def func_segmaps(cls, segmaps, random_state, parents, hooks):
        segmaps[0].arr += 1
        return segmaps

    @classmethod
    def func_keypoints(cls, keypoints_on_images, random_state, parents, hooks):
        for keypoints_on_image in keypoints_on_images:
            for kp in keypoints_on_image.keypoints:
                kp.x = (kp.x + 1) % 3
        return keypoints_on_images

    @classmethod
    def func_polygons(cls, polygons_on_images, random_state, parents, hooks):
        if len(polygons_on_images[0].polygons) == 0:
            return [ia.PolygonsOnImage([], shape=polygons_on_images[0].shape)]
        new_exterior = np.copy(polygons_on_images[0].polygons[0].exterior)
        new_exterior[:, 0] += 1
        new_exterior[:, 1] += 2
        return [
            ia.PolygonsOnImage([ia.Polygon(new_exterior)],
                               shape=polygons_on_images[0].shape)
        ]

    @classmethod
    def func_line_strings(cls, line_strings_on_images, random_state, parents,
                          hooks):
        if line_strings_on_images[0].empty:
            return [ia.LineStringsOnImage(
                [], shape=line_strings_on_images[0].shape)]
        new_coords = np.copy(line_strings_on_images[0].items[0].coords)
        new_coords[:, 0] += 1
        new_coords[:, 1] += 2
        return [
            ia.LineStringsOnImage(
                [ia.LineString(new_coords)],
                shape=line_strings_on_images[0].shape)
        ]

    @classmethod
    def func_bbs(cls, bounding_boxes_on_images, random_state, parents, hooks):
        if bounding_boxes_on_images[0].empty:
            return [
                ia.BoundingBoxesOnImage(
                    [], shape=bounding_boxes_on_images[0].shape)
            ]
        new_coords = np.copy(bounding_boxes_on_images[0].items[0].coords)
        new_coords[:, 0] += 1
        new_coords[:, 1] += 2
        return [
            ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1=new_coords[0][0], y1=new_coords[0][1],
                                x2=new_coords[1][0], y2=new_coords[1][1])],
                shape=bounding_boxes_on_images[0].shape)
        ]

    def test_images(self):
        image = self.base_img
        expected = image + 1
        aug = iaa.Lambda(func_images=self.func_images)

        for _ in sm.xrange(3):
            observed = aug.augment_image(image)

            assert np.array_equal(observed, expected)

    def test_images_deterministic(self):
        image = self.base_img
        expected = image + 1
        aug_det = iaa.Lambda(func_images=self.func_images).to_deterministic()

        for _ in sm.xrange(3):
            observed = aug_det.augment_image(image)

            assert np.array_equal(observed, expected)

    def test_images_list(self):
        image = self.base_img
        expected = [image + 1]
        aug = iaa.Lambda(func_images=self.func_images)

        observed = aug.augment_images([image])

        assert array_equal_lists(observed, expected)

    def test_images_list_deterministic(self):
        image = self.base_img
        expected = [image + 1]
        aug_det = iaa.Lambda(func_images=self.func_images).to_deterministic()

        observed = aug_det.augment_images([image])

        assert array_equal_lists(observed, expected)

    def test_heatmaps(self):
        heatmaps = self.heatmaps
        heatmaps_arr_aug = self.heatmaps_aug.get_arr()
        aug = iaa.Lambda(func_heatmaps=self.func_heatmaps)

        for _ in sm.xrange(3):
            observed = aug.augment_heatmaps(heatmaps)

            assert observed.shape == (3, 3, 3)
            assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
            assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
            assert np.allclose(observed.get_arr(), heatmaps_arr_aug)

    def test_heatmaps_deterministic(self):
        heatmaps = self.heatmaps
        heatmaps_arr_aug = self.heatmaps_aug.get_arr()
        aug_det = iaa.Lambda(func_heatmaps=self.func_heatmaps)\
            .to_deterministic()

        for _ in sm.xrange(3):
            observed = aug_det.augment_heatmaps(heatmaps)

            assert observed.shape == (3, 3, 3)
            assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
            assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
            assert np.allclose(observed.get_arr(), heatmaps_arr_aug)

    def test_segmentation_maps(self):
        segmaps = self.segmentation_maps
        segmaps_arr_aug = self.segmentation_maps_aug.get_arr()
        aug = iaa.Lambda(func_segmentation_maps=self.func_segmaps)

        for _ in sm.xrange(3):
            observed = aug.augment_segmentation_maps(segmaps)

            assert observed.shape == (3, 3, 3)
            assert np.array_equal(observed.get_arr(), segmaps_arr_aug)

    def test_segmentation_maps_deterministic(self):
        segmaps = self.segmentation_maps
        segmaps_arr_aug = self.segmentation_maps_aug.get_arr()
        aug_det = iaa.Lambda(func_segmentation_maps=self.func_segmaps)\
            .to_deterministic()

        for _ in sm.xrange(3):
            observed = aug_det.augment_segmentation_maps(segmaps)

            assert observed.shape == (3, 3, 3)
            assert np.array_equal(observed.get_arr(), segmaps_arr_aug)

    def test_keypoints(self):
        kpsoi = self.keypoints
        aug = iaa.Lambda(func_keypoints=self.func_keypoints)

        for _ in sm.xrange(3):
            observed = aug.augment_keypoints(kpsoi)

            expected = self.keypoints_aug
            assert_cbaois_equal(observed, expected)

    def test_keypoints_deterministic(self):
        kpsoi = self.keypoints
        aug = iaa.Lambda(func_keypoints=self.func_keypoints)
        aug = aug.to_deterministic()

        for _ in sm.xrange(3):
            observed = aug.augment_keypoints(kpsoi)

            expected = self.keypoints_aug
            assert_cbaois_equal(observed, expected)

    def test_polygons(self):
        psois = self.polygons
        aug = iaa.Lambda(func_polygons=self.func_polygons)

        for _ in sm.xrange(3):
            observed = aug.augment_polygons(psois)

            expected_psoi = self.polygons_aug
            assert_cbaois_equal(observed, expected_psoi)

    def test_polygons_deterministic(self):
        psois = self.polygons

        aug = iaa.Lambda(func_polygons=self.func_polygons)
        aug = aug.to_deterministic()

        for _ in sm.xrange(3):
            observed = aug.augment_polygons(psois)

            expected_psoi = self.polygons_aug
            assert_cbaois_equal(observed, expected_psoi)

    def test_line_strings(self):
        lsois = self.lsoi
        aug = iaa.Lambda(func_line_strings=self.func_line_strings)

        for _ in sm.xrange(3):
            observed = aug.augment_line_strings(lsois)

            expected_lsoi = self.lsoi_aug
            assert_cbaois_equal(observed, expected_lsoi)

    def test_line_strings_deterministic(self):
        lsois = self.lsoi

        aug = iaa.Lambda(func_line_strings=self.func_line_strings)
        aug = aug.to_deterministic()

        for _ in sm.xrange(3):
            observed = aug.augment_line_strings(lsois)

            expected_lsoi = self.lsoi_aug
            assert_cbaois_equal(observed, expected_lsoi)

    def test_bounding_boxes(self):
        bbsoi = self.bbsoi
        aug = iaa.Lambda(func_bounding_boxes=self.func_bbs)

        for _ in sm.xrange(3):
            observed = aug.augment_bounding_boxes(bbsoi)

            expected = self.bbsoi_aug
            assert_cbaois_equal(observed, expected)

    def test_bounding_boxes_deterministic(self):
        bbsoi = self.bbsoi
        aug = iaa.Lambda(func_bounding_boxes=self.func_bbs)
        aug = aug.to_deterministic()

        for _ in sm.xrange(3):
            observed = aug.augment_bounding_boxes(bbsoi)

            expected = self.bbsoi_aug
            assert_cbaois_equal(observed, expected)

    def test_bounding_boxes_x1_x2_coords_can_get_flipped(self):
        # Verify that if any augmented BB ends up with x1 > x2 that the
        # x-coordinates will be flipped to ensure that x1 is always below x2
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)
        ], shape=(10, 10, 3))

        def _func_bbs(bounding_boxes_on_images, random_state, parents, hooks):
            bounding_boxes_on_images[0].bounding_boxes[0].x1 += 10
            return bounding_boxes_on_images

        aug = iaa.Lambda(func_bounding_boxes=_func_bbs)

        for _ in sm.xrange(3):
            observed = aug.augment_bounding_boxes(bbsoi)

            assert np.allclose(
                observed.bounding_boxes[0].coords,
                [(2, 1), (0+10, 3)]
            )

    def test_bounding_boxes_y1_y2_coords_can_get_flipped(self):
        # Verify that if any augmented BB ends up with y1 > y2 that the
        # x-coordinates will be flipped to ensure that y1 is always below y2
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)
        ], shape=(10, 10, 3))

        def _func_bbs(bounding_boxes_on_images, random_state, parents, hooks):
            bounding_boxes_on_images[0].bounding_boxes[0].y1 += 10
            return bounding_boxes_on_images

        aug = iaa.Lambda(func_bounding_boxes=_func_bbs)

        for _ in sm.xrange(3):
            observed = aug.augment_bounding_boxes(bbsoi)

            assert np.allclose(
                observed.bounding_boxes[0].coords,
                [(0, 3), (2, 1+10)]
            )

    def test_keypoints_empty(self):
        kpsoi = ia.KeypointsOnImage([], shape=(1, 2, 3))
        aug = iaa.Lambda(func_keypoints=self.func_keypoints)

        observed = aug.augment_keypoints(kpsoi)

        assert_cbaois_equal(observed, kpsoi)

    def test_polygons_empty(self):
        psoi = ia.PolygonsOnImage([], shape=(1, 2, 3))
        aug = iaa.Lambda(func_polygons=self.func_polygons)

        observed = aug.augment_polygons(psoi)

        assert_cbaois_equal(observed, psoi)

    def test_line_strings_empty(self):
        lsoi = ia.LineStringsOnImage([], shape=(1, 2, 3))
        aug = iaa.Lambda(func_line_strings=self.func_line_strings)

        observed = aug.augment_line_strings(lsoi)

        assert_cbaois_equal(observed, lsoi)

    def test_bounding_boxes_empty(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(1, 2, 3))
        aug = iaa.Lambda(func_bounding_boxes=self.func_bbs)

        observed = aug.augment_bounding_boxes(bbsoi)

        assert_cbaois_equal(observed, bbsoi)

    # TODO add tests when funcs are not set in Lambda

    def test_other_dtypes_bool(self):
        def func_images(images, random_state, parents, hooks):
            aug = iaa.Flipud(1.0)  # flipud is know to work with all dtypes
            return aug.augment_images(images)

        aug = iaa.Lambda(func_images=func_images)

        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True
        expected = np.zeros((3, 3), dtype=bool)
        expected[2, 0] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == "bool"
        assert np.all(image_aug == expected)

    def test_other_dtypes_uint_int(self):
        def func_images(images, random_state, parents, hooks):
            aug = iaa.Flipud(1.0)  # flipud is know to work with all dtypes
            return aug.augment_images(images)

        aug = iaa.Lambda(func_images=func_images)

        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value
                expected = np.zeros((3, 3), dtype=dtype)
                expected[2, 0] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, expected)

    def test_other_dtypes_float(self):
        def func_images(images, random_state, parents, hooks):
            aug = iaa.Flipud(1.0)  # flipud is know to work with all dtypes
            return aug.augment_images(images)

        aug = iaa.Lambda(func_images=func_images)

        try:
            f128 = [np.dtype("float128").name]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128
        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]

        for dtype, value in zip(dtypes, values):
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value
                expected = np.zeros((3, 3), dtype=dtype)
                expected[2, 0] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.all(image_aug == expected)

    def test_pickleable(self):
        aug = iaa.Lambda(
            func_images=_lambda_pickleable_callback_images,
            seed=1)
        runtest_pickleable_uint8_img(aug)


def _lambda_pickleable_callback_images(images, random_state, parents, hooks):
    aug = iaa.Flipud(0.5, seed=random_state)
    return aug.augment_images(images)


class TestAssertLambda(unittest.TestCase):
    DTYPES_UINT = ["uint8", "uint16", "uint32", "uint64"]
    DTYPES_INT = ["int8", "int32", "int64"]
    DTYPES_FLOAT = (
        ["float16", "float32", "float64"]
        + (
            ["float128"] if hasattr(np, "float128") else []
        )
    )

    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.array([[0, 0, 1],
                             [0, 0, 1],
                             [0, 1, 1]], dtype=np.uint8)
        return np.atleast_3d(base_img)

    @property
    def images(self):
        return np.array([self.image])

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 1.0, 1.0]])
        return ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps(self):
        segmaps_arr = np.int32([[0, 0, 1],
                                [0, 0, 1],
                                [0, 1, 1]])
        return SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        return ia.KeypointsOnImage(kps, shape=self.image.shape)

    @property
    def psoi(self):
        polygons = [ia.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
        return ia.PolygonsOnImage(polygons, shape=self.image.shape)

    @property
    def lsoi(self):
        lss = [ia.LineString([(0, 0), (2, 0), (2, 2), (0, 2)])]
        return ia.LineStringsOnImage(lss, shape=self.image.shape)

    @property
    def bbsoi(self):
        bb = ia.BoundingBox(x1=0, y1=0, x2=2, y2=2)
        return ia.BoundingBoxesOnImage([bb], shape=self.image.shape)

    @property
    def aug_succeeds(self):
        def _func_images_succeeds(images, random_state, parents, hooks):
            return images[0][0, 0] == 0 and images[0][2, 2] == 1

        def _func_heatmaps_succeeds(heatmaps, random_state, parents, hooks):
            return heatmaps[0].arr_0to1[0, 0] < 0 + 1e-6

        def _func_segmaps_succeeds(segmaps, random_state, parents, hooks):
            return segmaps[0].arr[0, 0] == 0

        def _func_keypoints_succeeds(keypoints_on_images, random_state, parents,
                                    hooks):
            return (
                keypoints_on_images[0].keypoints[0].x == 0
                and keypoints_on_images[0].keypoints[2].x == 2
            )

        def _func_bounding_boxes_succeeds(bounding_boxes_on_images,
                                          random_state, parents, hooks):
            return (bounding_boxes_on_images[0].items[0].x1 == 0
                    and bounding_boxes_on_images[0].items[0].x2 == 2)

        def _func_polygons_succeeds(polygons_on_images, random_state, parents,
                                   hooks):
            return (polygons_on_images[0].polygons[0].exterior[0][0] == 0
                    and polygons_on_images[0].polygons[0].exterior[2][1] == 2)

        def _func_line_strings_succeeds(line_strings_on_image, random_state,
                                        parents, hooks):
            return (line_strings_on_image[0].items[0].coords[0][0] == 0
                    and line_strings_on_image[0].items[0].coords[2][1] == 2)

        return iaa.AssertLambda(
            func_images=_func_images_succeeds,
            func_heatmaps=_func_heatmaps_succeeds,
            func_segmentation_maps=_func_segmaps_succeeds,
            func_keypoints=_func_keypoints_succeeds,
            func_bounding_boxes=_func_bounding_boxes_succeeds,
            func_polygons=_func_polygons_succeeds,
            func_line_strings=_func_line_strings_succeeds)

    @property
    def aug_fails(self):
        def _func_images_fails(images, random_state, parents, hooks):
            return images[0][0, 0] == 1

        def _func_heatmaps_fails(heatmaps, random_state, parents, hooks):
            return heatmaps[0].arr_0to1[0, 0] > 0 + 1e-6

        def _func_segmaps_fails(segmaps, random_state, parents, hooks):
            return segmaps[0].arr[0, 0] == 1

        def _func_keypoints_fails(keypoints_on_images, random_state, parents,
                                  hooks):
            return keypoints_on_images[0].keypoints[0].x == 2

        def _func_bounding_boxes_fails(bounding_boxes_on_images, random_state,
                                       parents, hooks):
            return bounding_boxes_on_images[0].items[0].x1 == 2

        def _func_polygons_fails(polygons_on_images, random_state, parents,
                                 hooks):
            return polygons_on_images[0].polygons[0].exterior[0][0] == 2

        def _func_line_strings_fails(line_strings_on_images, random_state,
                                     parents, hooks):
            return line_strings_on_images[0].items[0].coords[0][0] == 2

        return iaa.AssertLambda(
            func_images=_func_images_fails,
            func_heatmaps=_func_heatmaps_fails,
            func_segmentation_maps=_func_segmaps_fails,
            func_keypoints=_func_keypoints_fails,
            func_bounding_boxes=_func_bounding_boxes_fails,
            func_polygons=_func_polygons_fails,
            func_line_strings=_func_line_strings_fails)

    def test_images_as_array_with_assert_that_succeeds(self):
        observed = self.aug_succeeds.augment_images(self.images)
        expected = self.images
        assert np.array_equal(observed, expected)

    def test_images_as_array_with_assert_that_fails(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_images(self.images)

    def test_images_as_array_with_assert_that_succeeds__deterministic(self):
        aug_succeeds_det = self.aug_succeeds.to_deterministic()
        observed = aug_succeeds_det.augment_images(self.images)
        expected = self.images
        assert np.array_equal(observed, expected)

    def test_images_as_array_with_assert_that_fails__deterministic(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_images(self.images)

    def test_images_as_list_with_assert_that_succeeds(self):
        observed = self.aug_succeeds.augment_images([self.images[0]])
        expected = [self.images[0]]
        assert array_equal_lists(observed, expected)

    def test_images_as_list_with_assert_that_fails(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_images([self.images[0]])

    def test_images_as_list_with_assert_that_succeeds__deterministic(self):
        aug_succeeds_det = self.aug_succeeds.to_deterministic()
        observed = aug_succeeds_det.augment_images([self.images[0]])
        expected = [self.images[0]]
        assert array_equal_lists(observed, expected)

    def test_images_as_list_with_assert_that_fails__deterministic(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_images([self.images[0]])

    def test_heatmaps_with_assert_that_succeeds(self):
        observed = self.aug_succeeds.augment_heatmaps(self.heatmaps)
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps.get_arr())

    def test_heatmaps_with_assert_that_fails(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_heatmaps(self.heatmaps)

    def test_heatmaps_with_assert_that_succeeds__deterministic(self):
        aug_succeeds_det = self.aug_succeeds.to_deterministic()
        observed = aug_succeeds_det.augment_heatmaps(self.heatmaps)
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps.get_arr())

    def test_heatmaps_with_assert_that_fails__deterministic(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_heatmaps(self.heatmaps)

    def test_segmaps_with_assert_that_succeeds(self):
        observed = self.aug_succeeds.augment_segmentation_maps(self.segmaps)
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps.get_arr())

    def test_segmaps_with_assert_that_fails(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_segmentation_maps(self.segmaps)

    def test_segmaps_with_assert_that_succeeds__deterministic(self):
        aug_succeeds_det = self.aug_succeeds.to_deterministic()
        observed = aug_succeeds_det.augment_segmentation_maps(self.segmaps)
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps.get_arr())

    def test_segmaps_with_assert_that_fails__deterministic(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_segmentation_maps(self.segmaps)

    def test_keypoints_with_assert_that_succeeds(self):
        observed = self.aug_succeeds.augment_keypoints(self.kpsoi)
        assert_cbaois_equal(observed, self.kpsoi)

    def test_keypoints_with_assert_that_fails(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_keypoints(self.kpsoi)

    def test_keypoints_with_assert_that_succeeds__deterministic(self):
        aug_succeeds_det = self.aug_succeeds.to_deterministic()
        observed = aug_succeeds_det.augment_keypoints(self.kpsoi)
        assert_cbaois_equal(observed, self.kpsoi)

    def test_keypoints_with_assert_that_fails__deterministic(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_keypoints(self.kpsoi)

    def test_polygons_with_assert_that_succeeds(self):
        observed = self.aug_succeeds.augment_polygons(self.psoi)
        assert_cbaois_equal(observed, self.psoi)

    def test_polygons_with_assert_that_fails(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_polygons(self.psoi)

    def test_polygons_with_assert_that_succeeds__deterministic(self):
        aug_succeeds_det = self.aug_succeeds.to_deterministic()
        observed = aug_succeeds_det.augment_polygons(self.psoi)
        assert_cbaois_equal(observed, self.psoi)

    def test_polygons_with_assert_that_fails__deterministic(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_polygons(self.psoi)

    def test_line_strings_with_assert_that_succeeds(self):
        observed = self.aug_succeeds.augment_line_strings(self.lsoi)
        assert_cbaois_equal(observed, self.lsoi)

    def test_line_strings_with_assert_that_fails(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_line_strings(self.lsoi)

    def test_line_strings_with_assert_that_succeeds__deterministic(self):
        aug_succeeds_det = self.aug_succeeds.to_deterministic()
        observed = aug_succeeds_det.augment_line_strings(self.lsoi)
        assert_cbaois_equal(observed, self.lsoi)

    def test_line_strings_with_assert_that_fails__deterministic(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_line_strings(self.lsoi)

    def test_bounding_boxes_with_assert_that_succeeds(self):
        observed = self.aug_succeeds.augment_bounding_boxes(self.bbsoi)
        assert_cbaois_equal(observed, self.bbsoi)

    def test_bounding_boxes_with_assert_that_fails(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_bounding_boxes(self.bbsoi)

    def test_bounding_boxes_with_assert_that_succeeds__deterministic(self):
        aug_succeeds_det = self.aug_succeeds.to_deterministic()
        observed = aug_succeeds_det.augment_bounding_boxes(self.bbsoi)
        assert_cbaois_equal(observed, self.bbsoi)

    def test_bounding_boxes_with_assert_that_fails__deterministic(self):
        with self.assertRaises(AssertionError):
            _ = self.aug_fails.augment_bounding_boxes(self.bbsoi)

    def test_other_dtypes_bool__with_assert_that_succeeds(self):
        def func_images_succeeds(images, random_state, parents, hooks):
            return np.allclose(images[0][0, 0], 1, rtol=0, atol=1e-6)

        aug = iaa.AssertLambda(func_images=func_images_succeeds)

        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.name == image.dtype.name
        assert np.all(image_aug == image)

    def test_other_dtypes_uint_int__with_assert_that_succeeds(self):
        def func_images_succeeds(images, random_state, parents, hooks):
            return np.allclose(images[0][0, 0], 1, rtol=0, atol=1e-6)

        aug = iaa.AssertLambda(func_images=func_images_succeeds)

        dtypes = self.DTYPES_UINT + self.DTYPES_INT
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = 1
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, image)

    def test_other_dtypes_float__with_assert_that_succeeds(self):
        def func_images_succeeds(images, random_state, parents, hooks):
            return np.allclose(images[0][0, 0], 1, rtol=0, atol=1e-6)

        aug = iaa.AssertLambda(func_images=func_images_succeeds)

        dtypes = self.DTYPES_FLOAT
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = 1
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.name == dtype
                assert np.all(image_aug == image)

    def test_other_dtypes_bool__with_assert_that_fails(self):
        def func_images_fails(images, random_state, parents, hooks):
            return np.allclose(images[0][0, 1], 1, rtol=0, atol=1e-6)

        aug = iaa.AssertLambda(func_images=func_images_fails)

        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True
        with self.assertRaises(AssertionError):
            _ = aug.augment_image(image)

    def test_other_dtypes_uint_int__with_assert_that_fails(self):
        def func_images_fails(images, random_state, parents, hooks):
            return np.allclose(images[0][0, 1], 1, rtol=0, atol=1e-6)

        aug = iaa.AssertLambda(func_images=func_images_fails)

        dtypes = self.DTYPES_UINT + self.DTYPES_INT
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = 1
                with self.assertRaises(AssertionError):
                    _ = aug.augment_image(image)

    def test_other_dtypes_float__with_assert_that_fails(self):
        def func_images_fails(images, random_state, parents, hooks):
            return np.allclose(images[0][0, 1], 1, rtol=0, atol=1e-6)

        aug = iaa.AssertLambda(func_images=func_images_fails)

        dtypes = self.DTYPES_FLOAT
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = 1
                with self.assertRaises(AssertionError):
                    _ = aug.augment_image(image)

    def test_pickleable(self):
        aug = iaa.AssertLambda(
            func_images=_assertlambda_pickleable_callback_images,
            seed=1)
        runtest_pickleable_uint8_img(aug, iterations=2)


# in py3+, this could be a classmethod of TestAssertLambda,
# but in py2.7 such classmethods are not pickle-able and would cause an error
def _assertlambda_pickleable_callback_images(images, random_state,
                                             parents, hooks):
    return np.any(images[0] > 0)


class TestAssertShape(unittest.TestCase):
    DTYPES_UINT = ["uint8", "uint16", "uint32", "uint64"]
    DTYPES_INT = ["int8", "int32", "int64"]
    DTYPES_FLOAT = (
        ["float16", "float32", "float64"]
        + (
            ["float128"] if hasattr(np, "float128") else []
        )
    )

    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.array([[0, 0, 1, 0],
                             [0, 0, 1, 0],
                             [0, 1, 1, 0]], dtype=np.uint8)
        return np.atleast_3d(base_img)

    @property
    def images(self):
        return np.array([self.image])

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([[0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 1.0, 1.0, 0.0]])
        return ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 4, 3))

    @property
    def segmaps(self):
        segmaps_arr = np.int32([[0, 0, 1, 0],
                                [0, 0, 1, 0],
                                [0, 1, 1, 0]])
        return SegmentationMapsOnImage(segmaps_arr, shape=(3, 4, 3))

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        return ia.KeypointsOnImage(kps, shape=self.image.shape)

    @property
    def psoi(self):
        polygons = [ia.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
        return ia.PolygonsOnImage(polygons, shape=self.image.shape)

    @property
    def lsoi(self):
        lss = [ia.LineString([(0, 0), (2, 0), (2, 2), (0, 2)])]
        return ia.LineStringsOnImage(lss, shape=self.image.shape)

    @property
    def bbsoi(self):
        bb = ia.BoundingBox(x1=0, y1=0, x2=2, y2=2)
        return ia.BoundingBoxesOnImage([bb], shape=self.image.shape)

    @property
    def image_h4(self):
        base_img_h4 = np.array([[0, 0, 1, 0],
                                [0, 0, 1, 0],
                                [0, 1, 1, 0],
                                [1, 0, 1, 0]], dtype=np.uint8)
        return np.atleast_3d(base_img_h4)

    @property
    def images_h4(self):
        return np.array([self.image_h4])

    @property
    def heatmaps_h4(self):
        heatmaps_arr_h4 = np.float32([[0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [0.0, 1.0, 1.0, 0.0],
                                      [1.0, 0.0, 1.0, 0.0]])
        return ia.HeatmapsOnImage(heatmaps_arr_h4, shape=(4, 4, 3))

    @property
    def segmaps_h4(self):
        segmaps_arr_h4 = np.int32([[0, 0, 1, 0],
                                   [0, 0, 1, 0],
                                   [0, 1, 1, 0],
                                   [1, 0, 1, 0]])
        return SegmentationMapsOnImage(segmaps_arr_h4, shape=(4, 4, 3))

    @property
    def kpsoi_h4(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        return ia.KeypointsOnImage(kps, shape=self.image_h4.shape)

    @property
    def psoi_h4(self):
        polygons = [ia.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
        return ia.PolygonsOnImage(polygons, shape=self.image_h4.shape)

    @property
    def lsoi_h4(self):
        lss = [ia.LineString([(0, 0), (2, 0), (2, 2), (0, 2)])]
        return ia.LineStringsOnImage(lss, shape=self.image_h4.shape)

    @property
    def bbsoi_h4(self):
        bb = ia.BoundingBox(x1=0, y1=0, x2=2, y2=2)
        return ia.BoundingBoxesOnImage([bb], shape=self.image_h4.shape)

    @property
    def aug_exact_shape(self):
        return iaa.AssertShape((1, 3, 4, 1))

    @property
    def aug_none_in_shape(self):
        return iaa.AssertShape((None, 3, 4, 1))

    @property
    def aug_list_in_shape(self):
        return iaa.AssertShape((1, [1, 3, 5], 4, 1))

    @property
    def aug_tuple_in_shape(self):
        return iaa.AssertShape((1, (1, 4), 4, 1))

    def test_images_with_exact_shape__succeeds(self):
        aug = self.aug_exact_shape
        observed = aug.augment_images(self.images)
        expected = self.images
        assert np.array_equal(observed, expected)

    def test_images_with_exact_shape__succeeds__deterministic(self):
        aug_det = self.aug_exact_shape.to_deterministic()
        observed = aug_det.augment_images(self.images)
        expected = self.images
        assert np.array_equal(observed, expected)

    def test_images_with_exact_shape__succeeds__list(self):
        aug = self.aug_exact_shape
        observed = aug.augment_images([self.images[0]])
        expected = [self.images[0]]
        assert array_equal_lists(observed, expected)

    def test_images_with_exact_shape__succeeds__deterministic__list(self):
        aug_det = self.aug_exact_shape.to_deterministic()
        observed = aug_det.augment_images([self.images[0]])
        expected = [self.images[0]]
        assert array_equal_lists(observed, expected)

    def test_heatmaps_with_exact_shape__succeeds(self):
        aug = self.aug_exact_shape
        observed = aug.augment_heatmaps(self.heatmaps)
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps.get_arr())

    def test_heatmaps_with_exact_shape__succeeds__deterministic(self):
        aug_det = self.aug_exact_shape.to_deterministic()
        observed = aug_det.augment_heatmaps(self.heatmaps)
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps.get_arr())

    def test_segmaps_with_exact_shape__succeeds(self):
        aug = self.aug_exact_shape
        observed = aug.augment_segmentation_maps(self.segmaps)
        assert observed.shape == (3, 4, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps.get_arr())

    def test_segmaps_with_exact_shape__succeeds__deterministic(self):
        aug_det = self.aug_exact_shape.to_deterministic()
        observed = aug_det.augment_segmentation_maps(self.segmaps)
        assert observed.shape == (3, 4, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps.get_arr())

    def test_keypoints_with_exact_shape__succeeds(self):
        aug = self.aug_exact_shape
        observed = aug.augment_keypoints(self.kpsoi)
        assert_cbaois_equal(observed, self.kpsoi)

    def test_keypoints_with_exact_shape__succeeds__deterministic(self):
        aug_det = self.aug_exact_shape.to_deterministic()
        observed = aug_det.augment_keypoints(self.kpsoi)
        assert_cbaois_equal(observed, self.kpsoi)

    def test_polygons_with_exact_shape__succeeds(self):
        aug = self.aug_exact_shape
        observed = aug.augment_polygons(self.psoi)
        assert_cbaois_equal(observed, self.psoi)

    def test_polygons_with_exact_shape__succeeds__deterministic(self):
        aug_det = self.aug_exact_shape.to_deterministic()
        observed = aug_det.augment_polygons(self.psoi)
        assert_cbaois_equal(observed, self.psoi)

    def test_line_strings_with_exact_shape__succeeds(self):
        aug = self.aug_exact_shape
        observed = aug.augment_line_strings(self.lsoi)
        assert_cbaois_equal(observed, self.lsoi)

    def test_line_strings_with_exact_shape__succeeds__deterministic(self):
        aug_det = self.aug_exact_shape.to_deterministic()
        observed = aug_det.augment_line_strings(self.lsoi)
        assert_cbaois_equal(observed, self.lsoi)

    def test_bounding_boxes_with_exact_shape__succeeds(self):
        aug = self.aug_exact_shape
        observed = aug.augment_bounding_boxes(self.bbsoi)
        assert_cbaois_equal(observed, self.bbsoi)

    def test_bounding_boxes_with_exact_shape__succeeds__deterministic(self):
        aug_det = self.aug_exact_shape.to_deterministic()
        observed = aug_det.augment_bounding_boxes(self.bbsoi)
        assert_cbaois_equal(observed, self.bbsoi)

    def test_images_with_exact_shape__fails(self):
        aug = self.aug_exact_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_images(self.images_h4)

    def test_heatmaps_with_exact_shape__fails(self):
        aug = self.aug_exact_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_heatmaps(self.heatmaps_h4)

    def test_keypoints_with_exact_shape__fails(self):
        aug = self.aug_exact_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_keypoints(self.kpsoi_h4)

    def test_polygons_with_exact_shape__fails(self):
        aug = self.aug_exact_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_polygons(self.psoi_h4)

    def test_line_strings_with_exact_shape__fails(self):
        aug = self.aug_exact_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_line_strings(self.lsoi_h4)

    def test_bounding_boxes_with_exact_shape__fails(self):
        aug = self.aug_exact_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_bounding_boxes(self.bbsoi_h4)

    def test_images_with_none_in_shape__succeeds(self):
        aug = self.aug_none_in_shape
        observed = aug.augment_images(self.images)
        expected = self.images
        assert np.array_equal(observed, expected)

    def test_images_with_none_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_none_in_shape.to_deterministic()
        observed = aug_det.augment_images(self.images)
        expected = self.images
        assert np.array_equal(observed, expected)

    def test_heatmaps_with_none_in_shape__succeeds(self):
        aug = self.aug_none_in_shape
        observed = aug.augment_heatmaps(self.heatmaps)
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps.get_arr())

    def test_heatmaps_with_none_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_none_in_shape.to_deterministic()
        observed = aug_det.augment_heatmaps(self.heatmaps)
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps.get_arr())

    def test_segmaps_with_none_in_shape__succeeds(self):
        aug = self.aug_none_in_shape
        observed = aug.augment_segmentation_maps(self.segmaps)
        assert observed.shape == (3, 4, 3)
        assert np.array_equal(observed.get_arr(), self.heatmaps.get_arr())

    def test_segmaps_with_none_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_none_in_shape.to_deterministic()
        observed = aug_det.augment_segmentation_maps(self.segmaps)
        assert observed.shape == (3, 4, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps.get_arr())

    def test_keypoints_with_none_in_shape__succeeds(self):
        aug = self.aug_none_in_shape
        observed = aug.augment_keypoints(self.kpsoi)
        assert_cbaois_equal(observed, self.kpsoi)

    def test_keypoints_with_none_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_none_in_shape.to_deterministic()
        observed = aug_det.augment_keypoints(self.kpsoi)
        assert_cbaois_equal(observed, self.kpsoi)

    def test_polygons_with_none_in_shape__succeeds(self):
        aug = self.aug_none_in_shape
        observed = aug.augment_polygons(self.psoi)
        assert_cbaois_equal(observed, self.psoi)

    def test_polygons_with_none_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_none_in_shape.to_deterministic()
        observed = aug_det.augment_polygons(self.psoi)
        assert_cbaois_equal(observed, self.psoi)

    def test_line_strings_with_none_in_shape__succeeds(self):
        aug = self.aug_none_in_shape
        observed = aug.augment_line_strings(self.lsoi)
        assert_cbaois_equal(observed, self.lsoi)

    def test_line_strings_with_none_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_none_in_shape.to_deterministic()
        observed = aug_det.augment_line_strings(self.lsoi)
        assert_cbaois_equal(observed, self.lsoi)

    def test_bounding_boxes_with_none_in_shape__succeeds(self):
        aug = self.aug_none_in_shape
        observed = aug.augment_bounding_boxes(self.bbsoi)
        assert_cbaois_equal(observed, self.bbsoi)

    def test_bounding_boxes_with_none_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_none_in_shape.to_deterministic()
        observed = aug_det.augment_bounding_boxes(self.bbsoi)
        assert_cbaois_equal(observed, self.bbsoi)

    def test_images_with_none_in_shape__fails(self):
        aug = self.aug_none_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_images(self.images_h4)

    def test_heatmaps_with_none_in_shape__fails(self):
        aug = self.aug_none_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_heatmaps(self.heatmaps_h4)

    def test_keypoints_with_none_in_shape__fails(self):
        aug = self.aug_none_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_keypoints(self.kpsoi_h4)

    def test_polygons_with_none_in_shape__fails(self):
        aug = self.aug_none_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_polygons(self.psoi_h4)

    def test_line_strings_with_none_in_shape__fails(self):
        aug = self.aug_none_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_line_strings(self.lsoi_h4)

    def test_bounding_boxes_with_none_in_shape__fails(self):
        aug = self.aug_none_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_bounding_boxes(self.bbsoi_h4)

    def test_images_with_list_in_shape__succeeds(self):
        aug = self.aug_list_in_shape
        observed = aug.augment_images(self.images)
        expected = self.images
        assert np.array_equal(observed, expected)

    def test_images_with_list_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_list_in_shape.to_deterministic()
        observed = aug_det.augment_images(self.images)
        expected = self.images
        assert np.array_equal(observed, expected)

    def test_heatmaps_with_list_in_shape__succeeds(self):
        aug = self.aug_list_in_shape
        observed = aug.augment_heatmaps(self.heatmaps)
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps.get_arr())

    def test_heatmaps_with_list_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_list_in_shape.to_deterministic()
        observed = aug_det.augment_heatmaps(self.heatmaps)
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps.get_arr())

    def test_segmaps_with_list_in_shape__succeeds(self):
        aug = self.aug_list_in_shape
        observed = aug.augment_segmentation_maps(self.segmaps)
        assert observed.shape == (3, 4, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps.get_arr())

    def test_segmaps_with_list_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_list_in_shape.to_deterministic()
        observed = aug_det.augment_segmentation_maps(self.segmaps)
        assert observed.shape == (3, 4, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps.get_arr())

    def test_keypoints_with_list_in_shape__succeeds(self):
        aug = self.aug_list_in_shape
        observed = aug.augment_keypoints(self.kpsoi)
        assert_cbaois_equal(observed, self.kpsoi)

    def test_keypoints_with_list_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_list_in_shape.to_deterministic()
        observed = aug_det.augment_keypoints(self.kpsoi)
        assert_cbaois_equal(observed, self.kpsoi)

    def test_polygons_with_list_in_shape__succeeds(self):
        aug = self.aug_list_in_shape
        observed = aug.augment_polygons(self.psoi)
        assert_cbaois_equal(observed, self.psoi)

    def test_polygons_with_list_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_list_in_shape.to_deterministic()
        observed = aug_det.augment_polygons(self.psoi)
        assert_cbaois_equal(observed, self.psoi)

    def test_line_strings_with_list_in_shape__succeeds(self):
        aug = self.aug_list_in_shape
        observed = aug.augment_line_strings(self.lsoi)
        assert_cbaois_equal(observed, self.lsoi)

    def test_line_strings_with_list_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_list_in_shape.to_deterministic()
        observed = aug_det.augment_line_strings(self.lsoi)
        assert_cbaois_equal(observed, self.lsoi)

    def test_bounding_boxes_with_list_in_shape__succeeds(self):
        aug = self.aug_list_in_shape
        observed = aug.augment_bounding_boxes(self.bbsoi)
        assert_cbaois_equal(observed, self.bbsoi)

    def test_bounding_boxes_with_list_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_list_in_shape.to_deterministic()
        observed = aug_det.augment_bounding_boxes(self.bbsoi)
        assert_cbaois_equal(observed, self.bbsoi)

    def test_images_with_list_in_shape__fails(self):
        aug = self.aug_list_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_images(self.images_h4)

    def test_heatmaps_with_list_in_shape__fails(self):
        aug = self.aug_list_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_heatmaps(self.heatmaps_h4)

    def test_segmaps_with_list_in_shape__fails(self):
        aug = self.aug_list_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_segmentation_maps(self.segmaps_h4)

    def test_keypoints_with_list_in_shape__fails(self):
        aug = self.aug_list_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_keypoints(self.kpsoi_h4)

    def test_polygons_with_list_in_shape__fails(self):
        aug = self.aug_list_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_polygons(self.psoi_h4)

    def test_line_strings_with_list_in_shape__fails(self):
        aug = self.aug_list_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_line_strings(self.lsoi_h4)

    def test_bounding_boxes_with_list_in_shape__fails(self):
        aug = self.aug_list_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_bounding_boxes(self.bbsoi_h4)

    def test_images_with_tuple_in_shape__succeeds(self):
        aug = self.aug_tuple_in_shape
        observed = aug.augment_images(self.images)
        expected = self.images
        assert np.array_equal(observed, expected)

    def test_images_with_tuple_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_tuple_in_shape.to_deterministic()
        observed = aug_det.augment_images(self.images)
        expected = self.images
        assert np.array_equal(observed, expected)

    def test_heatmaps_with_tuple_in_shape__succeeds(self):
        aug = self.aug_tuple_in_shape
        observed = aug.augment_heatmaps(self.heatmaps)
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps.get_arr())

    def test_heatmaps_with_tuple_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_tuple_in_shape.to_deterministic()
        observed = aug_det.augment_heatmaps(self.heatmaps)
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps.get_arr())

    def test_segmaps_with_tuple_in_shape__succeeds(self):
        aug = self.aug_tuple_in_shape
        observed = aug.augment_segmentation_maps(self.segmaps)
        assert observed.shape == (3, 4, 3)
        assert np.array_equal(observed.get_arr(), self.heatmaps.get_arr())

    def test_segmaps_with_tuple_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_tuple_in_shape.to_deterministic()
        observed = aug_det.augment_segmentation_maps(self.segmaps)
        assert observed.shape == (3, 4, 3)
        assert np.array_equal(observed.get_arr(), self.heatmaps.get_arr())

    def test_keypoints_with_tuple_in_shape__succeeds(self):
        aug = self.aug_tuple_in_shape
        observed = aug.augment_keypoints(self.kpsoi)
        assert_cbaois_equal(observed, self.kpsoi)

    def test_keypoints_with_tuple_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_tuple_in_shape.to_deterministic()
        observed = aug_det.augment_keypoints(self.kpsoi)
        assert_cbaois_equal(observed, self.kpsoi)

    def test_polygons_with_tuple_in_shape__succeeds(self):
        aug = self.aug_tuple_in_shape
        observed = aug.augment_polygons(self.psoi)
        assert_cbaois_equal(observed, self.psoi)

    def test_polygons_with_tuple_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_tuple_in_shape.to_deterministic()
        observed = aug_det.augment_polygons(self.psoi)
        assert_cbaois_equal(observed, self.psoi)

    def test_line_strings_with_tuple_in_shape__succeeds(self):
        aug = self.aug_tuple_in_shape
        observed = aug.augment_line_strings(self.lsoi)
        assert_cbaois_equal(observed, self.lsoi)

    def test_line_strings_with_tuple_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_tuple_in_shape.to_deterministic()
        observed = aug_det.augment_line_strings(self.lsoi)
        assert_cbaois_equal(observed, self.lsoi)

    def test_bounding_boxes_with_tuple_in_shape__succeeds(self):
        aug = self.aug_tuple_in_shape
        observed = aug.augment_bounding_boxes(self.bbsoi)
        assert_cbaois_equal(observed, self.bbsoi)

    def test_bounding_boxes_with_tuple_in_shape__succeeds__deterministic(self):
        aug_det = self.aug_tuple_in_shape.to_deterministic()
        observed = aug_det.augment_bounding_boxes(self.bbsoi)
        assert_cbaois_equal(observed, self.bbsoi)

    def test_images_with_tuple_in_shape__fails(self):
        aug = self.aug_tuple_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_images(self.images_h4)

    def test_heatmaps_with_tuple_in_shape__fails(self):
        aug = self.aug_tuple_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_heatmaps(self.heatmaps_h4)

    def test_segmaps_with_tuple_in_shape__fails(self):
        aug = self.aug_tuple_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_segmentation_maps(self.segmaps_h4)

    def test_keypoints_with_tuple_in_shape__fails(self):
        aug = self.aug_tuple_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_keypoints(self.kpsoi_h4)

    def test_polygons_with_tuple_in_shape__fails(self):
        aug = self.aug_tuple_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_polygons(self.psoi_h4)

    def test_line_strings_with_tuple_in_shape__fails(self):
        aug = self.aug_tuple_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_line_strings(self.lsoi_h4)

    def test_bounding_boxes_with_tuple_in_shape__fails(self):
        aug = self.aug_tuple_in_shape
        with self.assertRaises(AssertionError):
            _ = aug.augment_bounding_boxes(self.bbsoi_h4)

    def test_fails_if_shape_contains_invalid_datatype(self):
        got_exception = False
        try:
            aug = iaa.AssertShape((1, False, 4, 1))
            _ = aug.augment_images(np.zeros((1, 2, 2, 1), dtype=np.uint8))
        except Exception as exc:
            assert "Invalid datatype " in str(exc)
            got_exception = True
        assert got_exception

    def test_other_dtypes_bool__succeeds(self):
        aug = iaa.AssertShape((None, 3, 3, 1))
        image = np.zeros((3, 3, 1), dtype=bool)
        image[0, 0, 0] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.type == image.dtype.type
        assert np.all(image_aug == image)

    def test_other_dtypes_uint_int__succeeds(self):
        aug = iaa.AssertShape((None, 3, 3, 1))
        for dtype in self.DTYPES_UINT + self.DTYPES_INT:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3, 1), dtype=dtype)
                image[0, 0, 0] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, image)

    def test_other_dtypes_float__succeeds(self):
        aug = iaa.AssertShape((None, 3, 3, 1))
        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]

        for dtype, value in zip(self.DTYPES_FLOAT, values):
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3, 1), dtype=dtype)
                image[0, 0, 0] = 1
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.name == dtype
                assert np.all(image_aug == image)

    def test_other_dtypes_bool__fails(self):
        aug = iaa.AssertShape((None, 3, 4, 1))
        image = np.zeros((3, 3, 1), dtype=bool)
        image[0, 0, 0] = True

        with self.assertRaises(AssertionError):
            _ = aug.augment_image(image)

    def test_other_dtypes_uint_int__fails(self):
        aug = iaa.AssertShape((None, 3, 4, 1))

        for dtype in self.DTYPES_UINT + self.DTYPES_INT:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)
            value = max_value
            image = np.zeros((3, 3, 1), dtype=dtype)
            image[0, 0, 0] = value
            with self.assertRaises(AssertionError):
                _ = aug.augment_image(image)

    def test_other_dtypes_float__fails(self):
        aug = iaa.AssertShape((None, 3, 4, 1))
        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]

        for dtype, value in zip(self.DTYPES_FLOAT, values):
            image = np.zeros((3, 3, 1), dtype=dtype)
            image[0, 0, 0] = value
            with self.assertRaises(AssertionError):
                _ = aug.augment_image(image)

    def test_pickleable(self):
        aug = iaa.AssertShape(
            shape=(None, 15, 15, None), check_images=True,
            seed=1)
        runtest_pickleable_uint8_img(aug, iterations=2, shape=(15, 15, 1))


def test_clip_augmented_image_():
    warnings.resetwarnings()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        image = np.zeros((1, 3), dtype=np.uint8)
        image[0, 0] = 10
        image[0, 1] = 20
        image[0, 2] = 30
        image_clipped = iaa.clip_augmented_image_(image,
                                                  min_value=15, max_value=25)
        assert image_clipped[0, 0] == 15
        assert image_clipped[0, 1] == 20
        assert image_clipped[0, 2] == 25

    assert len(caught_warnings) >= 1
    assert "deprecated" in str(caught_warnings[-1].message)


def test_clip_augmented_image():
    warnings.resetwarnings()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        image = np.zeros((1, 3), dtype=np.uint8)
        image[0, 0] = 10
        image[0, 1] = 20
        image[0, 2] = 30
        image_clipped = iaa.clip_augmented_image(image,
                                                 min_value=15, max_value=25)
        assert image_clipped[0, 0] == 15
        assert image_clipped[0, 1] == 20
        assert image_clipped[0, 2] == 25

    assert len(caught_warnings) >= 1
    assert "deprecated" in str(caught_warnings[-1].message)


def test_clip_augmented_images_():
    warnings.resetwarnings()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        images = np.zeros((2, 1, 3), dtype=np.uint8)
        images[:, 0, 0] = 10
        images[:, 0, 1] = 20
        images[:, 0, 2] = 30
        imgs_clipped = iaa.clip_augmented_images_(images,
                                                  min_value=15, max_value=25)
        assert np.all(imgs_clipped[:, 0, 0] == 15)
        assert np.all(imgs_clipped[:, 0, 1] == 20)
        assert np.all(imgs_clipped[:, 0, 2] == 25)

        images = [np.zeros((1, 3), dtype=np.uint8) for _ in sm.xrange(2)]
        for i in sm.xrange(len(images)):
            images[i][0, 0] = 10
            images[i][0, 1] = 20
            images[i][0, 2] = 30
        imgs_clipped = iaa.clip_augmented_images_(images,
                                                  min_value=15, max_value=25)
        assert isinstance(imgs_clipped, list)
        assert np.all([imgs_clipped[i][0, 0] == 15
                       for i in sm.xrange(len(images))])
        assert np.all([imgs_clipped[i][0, 1] == 20
                       for i in sm.xrange(len(images))])
        assert np.all([imgs_clipped[i][0, 2] == 25
                       for i in sm.xrange(len(images))])

    assert len(caught_warnings) >= 1
    assert "deprecated" in str(caught_warnings[-1].message)


def test_clip_augmented_images():
    warnings.resetwarnings()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        images = np.zeros((2, 1, 3), dtype=np.uint8)
        images[:, 0, 0] = 10
        images[:, 0, 1] = 20
        images[:, 0, 2] = 30
        imgs_clipped = iaa.clip_augmented_images(images,
                                                 min_value=15, max_value=25)
        assert np.all(imgs_clipped[:, 0, 0] == 15)
        assert np.all(imgs_clipped[:, 0, 1] == 20)
        assert np.all(imgs_clipped[:, 0, 2] == 25)

        images = [np.zeros((1, 3), dtype=np.uint8) for _ in sm.xrange(2)]
        for i in sm.xrange(len(images)):
            images[i][0, 0] = 10
            images[i][0, 1] = 20
            images[i][0, 2] = 30
        imgs_clipped = iaa.clip_augmented_images(images,
                                                 min_value=15, max_value=25)
        assert isinstance(imgs_clipped, list)
        assert np.all([imgs_clipped[i][0, 0] == 15
                       for i in sm.xrange(len(images))])
        assert np.all([imgs_clipped[i][0, 1] == 20
                       for i in sm.xrange(len(images))])
        assert np.all([imgs_clipped[i][0, 2] ==
                       25 for i in sm.xrange(len(images))])

    assert len(caught_warnings) >= 1
    assert "deprecated" in str(caught_warnings[-1].message)


def test_reduce_to_nonempty():
    kpsois = [
        ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(4, 4, 3)),
        ia.KeypointsOnImage([ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=0)],
                            shape=(4, 4, 3)),
        ia.KeypointsOnImage([], shape=(4, 4, 3)),
        ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(4, 4, 3)),
        ia.KeypointsOnImage([], shape=(4, 4, 3))
    ]

    kpsois_reduced, ids = iaa.reduce_to_nonempty(kpsois)
    assert kpsois_reduced == [kpsois[0], kpsois[1], kpsois[3]]
    assert ids == [0, 1, 3]

    kpsois = [
        ia.KeypointsOnImage([], shape=(4, 4, 3)),
        ia.KeypointsOnImage([], shape=(4, 4, 3))
    ]

    kpsois_reduced, ids = iaa.reduce_to_nonempty(kpsois)
    assert kpsois_reduced == []
    assert ids == []

    kpsois = [
        ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(4, 4, 3))
    ]

    kpsois_reduced, ids = iaa.reduce_to_nonempty(kpsois)
    assert kpsois_reduced == [kpsois[0]]
    assert ids == [0]

    kpsois = []

    kpsois_reduced, ids = iaa.reduce_to_nonempty(kpsois)
    assert kpsois_reduced == []
    assert ids == []


def test_invert_reduce_to_nonempty():
    kpsois = [
        ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(4, 4, 3)),
        ia.KeypointsOnImage([ia.Keypoint(x=0, y=1),
                             ia.Keypoint(x=1, y=0)], shape=(4, 4, 3)),
        ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(4, 4, 3)),
    ]

    kpsois_recovered = iaa.invert_reduce_to_nonempty(
        kpsois, [0, 1, 2], ["foo1", "foo2", "foo3"])
    assert kpsois_recovered == ["foo1", "foo2", "foo3"]

    kpsois_recovered = iaa.invert_reduce_to_nonempty(kpsois, [1], ["foo1"])
    assert np.all([
        isinstance(kpsoi, ia.KeypointsOnImage)
        for kpsoi
        in kpsois])  # assert original list not changed
    assert kpsois_recovered == [kpsois[0], "foo1", kpsois[2]]

    kpsois_recovered = iaa.invert_reduce_to_nonempty(kpsois, [], [])
    assert kpsois_recovered == [kpsois[0], kpsois[1], kpsois[2]]

    kpsois_recovered = iaa.invert_reduce_to_nonempty([], [], [])
    assert kpsois_recovered == []


class _DummyAugmenter(iaa.Augmenter):
    def _augment_images(self, images, random_state, parents, hooks):
        return images

    def get_parameters(self):
        return []


class _DummyAugmenterBBs(iaa.Augmenter):
    def _augment_images(self, images, random_state, parents, hooks):
        return images

    def _augment_bounding_boxes(self, bounding_boxes_on_images, random_state,
                                parents, hooks):
        return [bbsoi.shift(x=1)
                for bbsoi
                in bounding_boxes_on_images]

    def get_parameters(self):
        return []


# TODO remove _augment_heatmaps() and _augment_keypoints() here once they are
#      no longer abstract methods but default to noop
class _DummyAugmenterCallsParent(iaa.Augmenter):
    def _augment_images(self, images, random_state, parents, hooks):
        return super(_DummyAugmenterCallsParent, self)\
            ._augment_images(images, random_state, parents, hooks)

    def get_parameters(self):
        return super(_DummyAugmenterCallsParent, self)\
            .get_parameters()


def _same_rs(rs1, rs2):
    return rs1.equals(rs2)


# TODO the test in here do not check everything, but instead only the cases
#      that were not yet indirectly tested via other tests
class TestAugmenter(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___global_rng(self):
        aug = _DummyAugmenter()
        assert not aug.deterministic
        assert aug.random_state.is_global_rng()

    def test___init___deterministic(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            aug = _DummyAugmenter(deterministic=True)
            assert aug.deterministic
            assert not aug.random_state.is_global_rng()

            assert len(caught_warnings) == 1
            assert (
                "is deprecated"
                in str(caught_warnings[-1].message))

    # old name for parameter `seed`
    def test___init___random_state_is_rng(self):
        rs = iarandom.RNG(123)
        aug = _DummyAugmenter(seed=rs)
        assert aug.random_state.generator is rs.generator

    # old name for parameter `seed`
    def test___init___random_state_is_seed(self):
        aug = _DummyAugmenter(seed=123)
        assert aug.random_state.equals(iarandom.RNG(123))

    def test___init___seed_is_random_state(self):
        rs = iarandom.RNG(123)
        aug = _DummyAugmenter(seed=rs)
        assert aug.random_state.generator is rs.generator

    def test___init___seed_is_seed(self):
        aug = _DummyAugmenter(seed=123)
        assert aug.random_state.equals(iarandom.RNG(123))

    def test_augment_images_called_probably_with_single_image(self):
        aug = _DummyAugmenter()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            _ = aug.augment_images(np.zeros((16, 32, 3), dtype=np.uint8))

        assert len(caught_warnings) == 1
        assert (
            "indicates that you provided a single image with shape (H, W, C)"
            in str(caught_warnings[-1].message)
        )

    def test_augment_images_array_in_list_out(self):
        self._test_augment_images_array_in_list_out_varying_channels(
            [3] * 20)

    def test_augment_images_array_in_list_out_single_channel(self):
        self._test_augment_images_array_in_list_out_varying_channels(
            [1] * 20)

    def test_augment_images_array_in_list_out_no_channels(self):
        self._test_augment_images_array_in_list_out_varying_channels(
            [None] * 20)

    def test_augment_images_array_in_list_out_varying_channels(self):
        self._test_augment_images_array_in_list_out_varying_channels(
            ["random"] * 20)

    @classmethod
    def _test_augment_images_array_in_list_out_varying_channels(cls,
                                                                nb_channels):
        assert len(nb_channels) == 20

        aug = iaa.Crop(((1, 8), (1, 8), (1, 8), (1, 8)), keep_size=False)
        seen = [0, 0]

        for nb_channels_i in nb_channels:
            if nb_channels_i == "random":
                channels = np.random.choice([None, 1, 3, 4, 9], size=(16,))

            elif nb_channels_i is None:
                channels = np.random.choice([None], size=(16,))
            else:
                channels = np.random.choice([nb_channels_i], size=(16,))

            images = [np.zeros((64, 64), dtype=np.uint8)
                      if c is None
                      else np.zeros((64, 64, c), dtype=np.uint8)
                      for c in channels]

            if nb_channels_i != "random":
                images = np.array(images)

            observed = aug.augment_images(images)

            if ia.is_np_array(observed):
                seen[0] += 1
            else:
                seen[1] += 1

            for image, c in zip(observed, channels):
                if c is None:
                    assert image.ndim == 2
                else:
                    assert image.ndim == 3
                    assert image.shape[2] == c
                assert 48 <= image.shape[0] <= 62
                assert 48 <= image.shape[1] <= 62

        assert seen[0] <= 3
        assert seen[1] >= 17

    def test_augment_images_with_2d_inputs(self):
        base_img1 = np.array([[0, 0, 1, 1],
                              [0, 0, 1, 1],
                              [0, 1, 1, 1]], dtype=np.uint8)
        base_img2 = np.array([[0, 0, 1, 1],
                              [0, 1, 1, 1],
                              [0, 1, 0, 0]], dtype=np.uint8)

        base_img1_flipped = np.array([[1, 1, 0, 0],
                                      [1, 1, 0, 0],
                                      [1, 1, 1, 0]], dtype=np.uint8)
        base_img2_flipped = np.array([[1, 1, 0, 0],
                                      [1, 1, 1, 0],
                                      [0, 0, 1, 0]], dtype=np.uint8)

        images = np.array([base_img1, base_img2])
        images_flipped = np.array([base_img1_flipped, base_img2_flipped])
        images_list = [base_img1, base_img2]
        images_flipped_list = [base_img1_flipped, base_img2_flipped]
        images_list2d3d = [base_img1, base_img2[:, :, np.newaxis]]
        images_flipped_list2d3d = [
            base_img1_flipped,
            base_img2_flipped[:, :, np.newaxis]]

        aug = iaa.Fliplr(1.0)
        noaug = iaa.Fliplr(0.0)

        # one numpy array as input
        observed = aug.augment_images(images)
        assert np.array_equal(observed, images_flipped)

        observed = noaug.augment_images(images)
        assert np.array_equal(observed, images)

        # list of 2d images
        observed = aug.augment_images(images_list)
        assert array_equal_lists(observed, images_flipped_list)

        observed = noaug.augment_images(images_list)
        assert array_equal_lists(observed, images_list)

        # list of images, one 2d and one 3d
        observed = aug.augment_images(images_list2d3d)
        assert array_equal_lists(observed, images_flipped_list2d3d)

        observed = noaug.augment_images(images_list2d3d)
        assert array_equal_lists(observed, images_list2d3d)

    def test_augment_keypoints_single_instance(self):
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(10, 10)], shape=(32, 32, 3))
        aug = iaa.Affine(translate_px={"x": 1})

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert len(kpsoi_aug.keypoints) == 1
        assert kpsoi_aug.keypoints[0].x == 11

    def test_augment_keypoints_single_instance_rot90(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=2, y=5),
               ia.Keypoint(x=3, y=3)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(5, 10, 3))
        aug = iaa.Rot90(1, keep_size=False)

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        # set offset to -1 if Rot90 uses int-based coordinate transformation
        kp_offset = 0
        assert np.allclose(kpsoi_aug.keypoints[0].x, 5 - 2 + kp_offset)
        assert np.allclose(kpsoi_aug.keypoints[0].y, 1)
        assert np.allclose(kpsoi_aug.keypoints[1].x, 5 - 5 + kp_offset)
        assert np.allclose(kpsoi_aug.keypoints[1].y, 2)
        assert np.allclose(kpsoi_aug.keypoints[2].x, 5 - 3 + kp_offset)
        assert np.allclose(kpsoi_aug.keypoints[2].y, 3)

    def test_augment_keypoints_many_instances_rot90(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=2, y=5),
               ia.Keypoint(x=3, y=3)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(5, 10, 3))
        aug = iaa.Rot90(1, keep_size=False)

        kpsoi_aug = aug.augment_keypoints([kpsoi, kpsoi, kpsoi])

        # set offset to -1 if Rot90 uses int-based coordinate transformation
        kp_offset = 0
        for i in range(3):
            assert np.allclose(kpsoi_aug[i].keypoints[0].x, 5 - 2 + kp_offset)
            assert np.allclose(kpsoi_aug[i].keypoints[0].y, 1)
            assert np.allclose(kpsoi_aug[i].keypoints[1].x, 5 - 5 + kp_offset)
            assert np.allclose(kpsoi_aug[i].keypoints[1].y, 2)
            assert np.allclose(kpsoi_aug[i].keypoints[2].x, 5 - 3 + kp_offset)
            assert np.allclose(kpsoi_aug[i].keypoints[2].y, 3)

    def test_augment_keypoints_empty_instance(self):
        # test empty KeypointsOnImage objects
        kpsoi = ia.KeypointsOnImage([], shape=(32, 32, 3))
        aug = iaa.Affine(translate_px={"x": 1})

        kpsoi_aug = aug.augment_keypoints([kpsoi])

        assert len(kpsoi_aug) == 1
        assert len(kpsoi_aug[0].keypoints) == 0

    def test_augment_keypoints_mixed_filled_and_empty_instances(self):
        kpsoi1 = ia.KeypointsOnImage([], shape=(32, 32, 3))
        kpsoi2 = ia.KeypointsOnImage([ia.Keypoint(10, 10)], shape=(32, 32, 3))
        aug = iaa.Affine(translate_px={"x": 1})

        kpsoi_aug = aug.augment_keypoints([kpsoi1, kpsoi2])

        assert len(kpsoi_aug) == 2
        assert len(kpsoi_aug[0].keypoints) == 0
        assert len(kpsoi_aug[1].keypoints) == 1
        assert kpsoi_aug[1].keypoints[0].x == 11

    def test_augment_keypoints_aligned_despite_empty_instance(self):
        # Test if augmenting lists of KeypointsOnImage is still aligned with
        # image augmentation when one KeypointsOnImage instance is empty
        # (no keypoints)
        kpsoi_lst = [
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(1, 10)),
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(1, 10)),
            ia.KeypointsOnImage([ia.Keypoint(x=1, y=0)], shape=(1, 10)),
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(1, 10)),
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(1, 10)),
            ia.KeypointsOnImage([], shape=(1, 8)),
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(1, 10)),
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(1, 10)),
            ia.KeypointsOnImage([ia.Keypoint(x=1, y=0)], shape=(1, 10)),
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(1, 10)),
            ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(1, 10))
        ]
        image = np.zeros((1, 10), dtype=np.uint8)
        image[0, 0] = 255
        images = np.tile(image[np.newaxis, :, :], (len(kpsoi_lst), 1, 1))

        aug = iaa.Affine(translate_px={"x": (0, 8)}, order=0, mode="constant",
                         cval=0)

        for i in sm.xrange(10):
            for is_list in [False, True]:
                with self.subTest(i=i, is_list=is_list):
                    aug_det = aug.to_deterministic()
                    if is_list:
                        images_aug = aug_det.augment_images(list(images))
                    else:
                        images_aug = aug_det.augment_images(images)
                    kpsoi_lst_aug = aug_det.augment_keypoints(kpsoi_lst)

                    if is_list:
                        images_aug = np.array(images_aug, dtype=np.uint8)
                    translations_imgs = np.argmax(images_aug[:, 0, :], axis=1)
                    translations_kps = [
                        kpsoi.keypoints[0].x
                        if len(kpsoi.keypoints) > 0
                        else None
                        for kpsoi
                        in kpsoi_lst_aug]

                    assert len([kpresult
                                for kpresult
                                in translations_kps
                                if kpresult is None]) == 1
                    assert translations_kps[5] is None
                    translations_imgs = np.concatenate(
                        [translations_imgs[0:5], translations_imgs[6:]])
                    translations_kps = np.array(
                        translations_kps[0:5] + translations_kps[6:],
                        dtype=translations_imgs.dtype)
                    translations_kps[2] -= 1
                    translations_kps[8-1] -= 1
                    assert np.array_equal(translations_imgs, translations_kps)

    def test_augment_keypoints_aligned_despite_nongeometric_image_ops(self):
        # Verify for keypoints that adding augmentations that only
        # affect images doesn't lead to misalignments between image
        # and keypoint transformations
        augs = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.AdditiveGaussianNoise(scale=(0.01, 0.1)),
            iaa.Affine(translate_px={"x": (-10, 10), "y": (-10, 10)},
                       order=0, mode="constant", cval=0),
            iaa.AddElementwise((0, 1)),
            iaa.Flipud(0.5)
        ], random_order=True)

        kps = [ia.Keypoint(x=15.5, y=12.5), ia.Keypoint(x=23.5, y=20.5),
               ia.Keypoint(x=61.5, y=36.5), ia.Keypoint(x=47.5, y=32.5)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(50, 80, 4))
        image = kpsoi.to_keypoint_image(size=1)
        images = np.tile(image[np.newaxis, ...], (20, 1, 1, 1))

        for _ in sm.xrange(50):
            images_aug, kpsois_aug = augs(images=images,
                                          keypoints=[kpsoi]*len(images))

            for image_aug, kpsoi_aug in zip(images_aug, kpsois_aug):
                kpsoi_recovered = ia.KeypointsOnImage.from_keypoint_image(
                    image_aug, nb_channels=4, threshold=100
                )

                for kp, kp_image in zip(kpsoi_aug.keypoints,
                                        kpsoi_recovered.keypoints):
                    distance = np.sqrt((kp.x - kp_image.x)**2
                                       + (kp.y - kp_image.y)**2)
                    assert distance <= 1

    def test_augment_bounding_boxes(self):
        aug = _DummyAugmenterBBs()
        bb = ia.BoundingBox(x1=1, y1=4, x2=2, y2=5)
        bbs = [bb]
        bbsois = [ia.BoundingBoxesOnImage(bbs, shape=(10, 10, 3))]
        bbsois_aug = aug.augment_bounding_boxes(bbsois)

        bb_aug = bbsois_aug[0].bounding_boxes[0]

        assert bb_aug.x1 == 1+1
        assert bb_aug.y1 == 4
        assert bb_aug.x2 == 2+1
        assert bb_aug.y2 == 5

    def test_augment_bounding_boxes_empty_bboi(self):
        aug = _DummyAugmenterBBs()
        bbsois = [ia.BoundingBoxesOnImage([], shape=(10, 10, 3))]

        bbsois_aug = aug.augment_bounding_boxes(bbsois)

        assert len(bbsois_aug) == 1
        assert bbsois_aug[0].bounding_boxes == []

    def test_augment_bounding_boxes_empty_list(self):
        aug = _DummyAugmenterBBs()

        bbsois_aug = aug.augment_bounding_boxes([])

        assert bbsois_aug == []

    def test_augment_bounding_boxes_single_instance(self):
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=1, x2=3, y1=4, y2=5),
            ia.BoundingBox(x1=2.5, x2=3, y1=0, y2=2)
        ], shape=(5, 10, 3))
        aug = iaa.Identity()

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        for bb_aug, bb in zip(bbsoi_aug.bounding_boxes, bbsoi.bounding_boxes):
            assert np.allclose(bb_aug.x1, bb.x1)
            assert np.allclose(bb_aug.x2, bb.x2)
            assert np.allclose(bb_aug.y1, bb.y1)
            assert np.allclose(bb_aug.y2, bb.y2)

    def test_augment_bounding_boxes_single_instance_rot90(self):
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=1, x2=3, y1=4, y2=5),
            ia.BoundingBox(x1=2.5, x2=3, y1=0, y2=2)
        ], shape=(5, 10, 3))
        aug = iaa.Rot90(1, keep_size=False)

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        # set offset to -1 if Rot90 uses int-based coordinate transformation
        kp_offset = 0
        # Note here that the new coordinates are minima/maxima of the BB, so
        # not as straight forward to compute the new coords as for keypoint
        # augmentation
        bb0 = bbsoi_aug.bounding_boxes[0]
        bb1 = bbsoi_aug.bounding_boxes[1]
        assert np.allclose(bb0.x1, 5 - 5 + kp_offset)
        assert np.allclose(bb0.x2, 5 - 4 + kp_offset)
        assert np.allclose(bb0.y1, 1)
        assert np.allclose(bb0.y2, 3)
        assert np.allclose(bb1.x1, 5 - 2 + kp_offset)
        assert np.allclose(bb1.x2, 5 - 0 + kp_offset)
        assert np.allclose(bb1.y1, 2.5)
        assert np.allclose(bb1.y2, 3)

    def test_augment_bounding_box_list_of_many_instances(self):
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=1, x2=3, y1=4, y2=5),
            ia.BoundingBox(x1=2.5, x2=3, y1=0, y2=2)
        ], shape=(5, 10, 3))
        aug = iaa.Rot90(1, keep_size=False)

        bbsoi_aug = aug.augment_bounding_boxes([bbsoi, bbsoi, bbsoi])

        # set offset to -1 if Rot90 uses int-based coordinate transformation
        kp_offset = 0
        for i in range(3):
            bb0 = bbsoi_aug[i].bounding_boxes[0]
            bb1 = bbsoi_aug[i].bounding_boxes[1]
            assert np.allclose(bb0.x1, 5 - 5 + kp_offset)
            assert np.allclose(bb0.x2, 5 - 4 + kp_offset)
            assert np.allclose(bb0.y1, 1)
            assert np.allclose(bb0.y2, 3)
            assert np.allclose(bb1.x1, 5 - 2 + kp_offset)
            assert np.allclose(bb1.x2, 5 - 0 + kp_offset)
            assert np.allclose(bb1.y1, 2.5)
            assert np.allclose(bb1.y2, 3)

    def test_augment_heatmaps_noop_single_heatmap(self):
        heatmap_arr = np.linspace(0.0, 1.0, num=4*4).reshape((4, 4, 1))
        heatmap = ia.HeatmapsOnImage(heatmap_arr.astype(np.float32),
                                     shape=(4, 4, 3))

        aug = iaa.Identity()
        heatmap_aug = aug.augment_heatmaps(heatmap)
        assert np.allclose(heatmap_aug.arr_0to1, heatmap.arr_0to1)

    def test_augment_heatmaps_rot90_single_heatmap(self):
        heatmap_arr = np.linspace(0.0, 1.0, num=4*4).reshape((4, 4, 1))
        heatmap = ia.HeatmapsOnImage(heatmap_arr.astype(np.float32),
                                     shape=(4, 4, 3))
        aug = iaa.Rot90(1, keep_size=False)

        heatmap_aug = aug.augment_heatmaps(heatmap)

        assert np.allclose(heatmap_aug.arr_0to1, np.rot90(heatmap.arr_0to1, -1))

    def test_augment_heatmaps_rot90_list_of_many_heatmaps(self):
        heatmap_arr = np.linspace(0.0, 1.0, num=4*4).reshape((4, 4, 1))
        heatmap = ia.HeatmapsOnImage(heatmap_arr.astype(np.float32),
                                     shape=(4, 4, 3))
        aug = iaa.Rot90(1, keep_size=False)

        heatmaps_aug = aug.augment_heatmaps([heatmap] * 3)

        for hm in heatmaps_aug:
            assert np.allclose(hm.arr_0to1, np.rot90(heatmap.arr_0to1, -1))

    def test_legacy_fallback_to_kp_aug_for_cbaois(self):
        class _LegacyAugmenter(iaa.Augmenter):
            def _augment_keypoints(self, keypoints_on_images, random_state,
                                   parents, hooks):
                return [kpsoi.shift(x=1) for kpsoi in keypoints_on_images]

            def get_parameters(self):
                return []

        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)
        ], shape=(4, 5, 3))
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (1, 0), (1, 1)])
        ], shape=(4, 5, 3))
        lsoi = ia.LineStringsOnImage([
            ia.LineString([(0, 0), (1, 0), (1, 1)])
        ], shape=(4, 5, 3))

        aug = _LegacyAugmenter()
        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)
        psoi_aug = aug.augment_polygons(psoi)
        lsoi_aug = aug.augment_line_strings(lsoi)

        assert bbsoi_aug[0].coords_almost_equals(bbsoi[0].shift(x=1))
        assert psoi_aug[0].coords_almost_equals(psoi[0].shift(x=1))
        assert lsoi_aug[0].coords_almost_equals(lsoi[0].shift(x=1))

    def test_localize_random_state(self):
        aug = _DummyAugmenter()

        aug_localized = aug.localize_random_state()

        assert aug_localized is not aug
        assert aug.random_state.is_global_rng()
        assert not aug_localized.random_state.is_global_rng()

    def test_seed_(self):
        aug1 = _DummyAugmenter()
        aug2 = _DummyAugmenter().to_deterministic()
        aug0 = iaa.Sequential([aug1, aug2])

        aug0_copy = aug0.deepcopy()
        assert _same_rs(aug0.random_state, aug0_copy.random_state)
        assert _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
        assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)

        aug0_copy.seed_()

        assert not _same_rs(aug0.random_state, aug0_copy.random_state)
        assert not _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
        assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)

    def test_seed__deterministic_too(self):
        aug1 = _DummyAugmenter()
        aug2 = _DummyAugmenter().to_deterministic()
        aug0 = iaa.Sequential([aug1, aug2])

        aug0_copy = aug0.deepcopy()
        assert _same_rs(aug0.random_state, aug0_copy.random_state)
        assert _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
        assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)

        aug0_copy.seed_(deterministic_too=True)

        assert not _same_rs(aug0.random_state, aug0_copy.random_state)
        assert not _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
        assert not _same_rs(aug0[1].random_state, aug0_copy[1].random_state)

    def test_seed__with_integer(self):
        aug1 = _DummyAugmenter()
        aug2 = _DummyAugmenter().to_deterministic()
        aug0 = iaa.Sequential([aug1, aug2])

        aug0_copy = aug0.deepcopy()
        assert _same_rs(aug0.random_state, aug0_copy.random_state)
        assert _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
        assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)

        aug0_copy.seed_(123)

        assert not _same_rs(aug0.random_state, aug0_copy.random_state)
        assert not _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
        assert _same_rs(aug0_copy.random_state, iarandom.RNG(123))
        expected = iarandom.RNG(123).derive_rng_()
        assert _same_rs(aug0_copy[0].random_state, expected)

    def test_seed__with_rng(self):
        aug1 = _DummyAugmenter()
        aug2 = _DummyAugmenter().to_deterministic()
        aug0 = iaa.Sequential([aug1, aug2])

        aug0_copy = aug0.deepcopy()
        assert _same_rs(aug0.random_state, aug0_copy.random_state)
        assert _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
        assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)

        aug0_copy.seed_(iarandom.RNG(123))

        assert not _same_rs(aug0.random_state, aug0_copy.random_state)
        assert not _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
        assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)
        assert _same_rs(aug0_copy.random_state,
                        iarandom.RNG(123))
        expected = iarandom.RNG(123).derive_rng_()
        assert _same_rs(aug0_copy[0].random_state, expected)

    def test_get_parameters(self):
        # test for "raise NotImplementedError"
        aug = _DummyAugmenterCallsParent()
        with self.assertRaises(NotImplementedError):
            aug.get_parameters()

    def test_get_all_children_flat(self):
        aug1 = _DummyAugmenter()
        aug21 = _DummyAugmenter()
        aug2 = iaa.Sequential([aug21])
        aug0 = iaa.Sequential([aug1, aug2])

        children = aug0.get_all_children(flat=True)

        assert isinstance(children, list)
        assert children[0] == aug1
        assert children[1] == aug2
        assert children[2] == aug21

    def test_get_all_children_not_flat(self):
        aug1 = _DummyAugmenter()
        aug21 = _DummyAugmenter()
        aug2 = iaa.Sequential([aug21])
        aug0 = iaa.Sequential([aug1, aug2])

        children = aug0.get_all_children(flat=False)

        assert isinstance(children, list)
        assert children[0] == aug1
        assert children[1] == aug2
        assert isinstance(children[2], list)
        assert children[2][0] == aug21

    def test___repr___and___str__(self):
        class DummyAugmenterRepr(iaa.Augmenter):
            def _augment_images(self, images, random_state, parents, hooks):
                return images

            def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
                return heatmaps

            def _augment_keypoints(self, keypoints_on_images, random_state,
                                   parents, hooks):
                return keypoints_on_images

            def get_parameters(self):
                return ["A", "B", "C"]

        aug1 = DummyAugmenterRepr(name="Example")
        aug2 = DummyAugmenterRepr(name="Example").to_deterministic()

        expected1 = (
            "DummyAugmenterRepr("
            "name=Example, parameters=[A, B, C], deterministic=False"
            ")")
        expected2 = (
            "DummyAugmenterRepr("
            "name=Example, parameters=[A, B, C], deterministic=True"
            ")")

        assert aug1.__repr__() == aug1.__str__() == expected1
        assert aug2.__repr__() == aug2.__str__() == expected2


# -----------
# lambda functions used in Test TestAugmenter_augment_batches
# in test method test_augment_batches_with_many_different_augmenters().
# They are here instead of in the test method, because otherwise there were
# issues with spawn mode not being able to pickle functions,
# see issue #414.

def _augment_batches__lambda_func_images(
        images, random_state, parents, hooks):
    return images


def _augment_batches__lambda_func_keypoints(
        keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


def _augment_batches__assertlambda_func_images(
        images, random_state, parents, hooks):
    return True


def _augment_batches__assertlambda_func_keypoints(
        keypoints_on_images, random_state, parents, hooks):
    return True
# -----------


class TestAugmenter_augment_batches(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_augment_batches_list_of_empty_list_deprecated(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            aug = _DummyAugmenter()

            batches_aug = list(aug.augment_batches([[]]))

            assert isinstance(batches_aug, list)
            assert len(batches_aug) == 1
            assert isinstance(batches_aug[0], list)

        assert len(caught_warnings) == 1
        assert "deprecated" in str(caught_warnings[-1].message)

    def test_augment_batches_list_of_arrays_deprecated(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            aug = _DummyAugmenter()
            image_batches = [np.zeros((1, 2, 2, 3), dtype=np.uint8)]

            batches_aug = list(aug.augment_batches(image_batches))

            assert isinstance(batches_aug, list)
            assert len(batches_aug) == 1
            assert array_equal_lists(batches_aug, image_batches)

        assert len(caught_warnings) == 1
        assert "deprecated" in str(caught_warnings[-1].message)

    def test_augment_batches_list_of_list_of_arrays_deprecated(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            aug = _DummyAugmenter()
            image_batches = [[np.zeros((2, 2, 3), dtype=np.uint8),
                              np.zeros((2, 3, 3))]]

            batches_aug = list(aug.augment_batches(image_batches))

            assert isinstance(batches_aug, list)
            assert len(batches_aug) == 1
            assert array_equal_lists(batches_aug[0], image_batches[0])

        assert len(caught_warnings) == 1
        assert "deprecated" in str(caught_warnings[-1].message)

    def test_augment_batches_invalid_datatype(self):
        aug = _DummyAugmenter()
        with self.assertRaises(Exception):
            _ = list(aug.augment_batches(None))

    def test_augment_batches_list_of_invalid_datatype(self):
        aug = _DummyAugmenter()
        got_exception = False
        try:
            _ = list(aug.augment_batches([None]))
        except Exception as exc:
            got_exception = True
            assert "Unknown datatype of batch" in str(exc)
        assert got_exception

    def test_augment_batches_list_of_list_of_invalid_datatype(self):
        aug = _DummyAugmenter()
        got_exception = False
        try:
            _ = list(aug.augment_batches([[None]]))
        except Exception as exc:
            got_exception = True
            assert "Unknown datatype in batch[0]" in str(exc)
        assert got_exception

    def test_augment_batches_batch_with_list_of_images(self):
        image = np.array([[0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 1, 1, 1, 1]], dtype=np.uint8)
        image_flipped = np.fliplr(image)
        keypoint = ia.Keypoint(x=2, y=1)
        keypoints = [ia.KeypointsOnImage([keypoint], shape=image.shape + (1,))]
        kp_flipped = ia.Keypoint(
            x=image.shape[1]-keypoint.x,
            y=keypoint.y
        )

        # basic functionality test (images as list)
        for bg in [True, False]:
            seq = iaa.Fliplr(1.0)
            batches = [ia.Batch(images=[np.copy(image)], keypoints=keypoints)]
            batches_aug = list(seq.augment_batches(batches, background=bg))
            baug0 = batches_aug[0]
            assert np.array_equal(baug0.images_aug[0], image_flipped)
            assert baug0.keypoints_aug[0].keypoints[0].x == kp_flipped.x
            assert baug0.keypoints_aug[0].keypoints[0].y == kp_flipped.y
            assert np.array_equal(baug0.images_unaug[0], image)
            assert baug0.keypoints_unaug[0].keypoints[0].x == keypoint.x
            assert baug0.keypoints_unaug[0].keypoints[0].y == keypoint.y

    def test_augment_batches_batch_with_array_of_images(self):
        image = np.array([[0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 1, 1, 1, 1]], dtype=np.uint8)
        image_flipped = np.fliplr(image)
        keypoint = ia.Keypoint(x=2, y=1)
        keypoints = [ia.KeypointsOnImage([keypoint], shape=image.shape + (1,))]
        kp_flipped = ia.Keypoint(
            x=image.shape[1]-keypoint.x,
            y=keypoint.y
        )

        # basic functionality test (images as array)
        for bg in [True, False]:
            seq = iaa.Fliplr(1.0)
            batches = [ia.Batch(images=np.uint8([np.copy(image)]),
                                keypoints=keypoints)]
            batches_aug = list(seq.augment_batches(batches, background=bg))
            baug0 = batches_aug[0]
            assert np.array_equal(baug0.images_aug, np.uint8([image_flipped]))
            assert baug0.keypoints_aug[0].keypoints[0].x == kp_flipped.x
            assert baug0.keypoints_aug[0].keypoints[0].y == kp_flipped.y
            assert np.array_equal(baug0.images_unaug, np.uint8([image]))
            assert baug0.keypoints_unaug[0].keypoints[0].x == keypoint.x
            assert baug0.keypoints_unaug[0].keypoints[0].y == keypoint.y

    def test_augment_batches_background(self):
        image = np.array([[0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 1, 1, 1, 1]], dtype=np.uint8)
        image_flipped = np.fliplr(image)
        kps = ia.Keypoint(x=2, y=1)
        kpsoi = ia.KeypointsOnImage([kps], shape=image.shape + (1,))
        kp_flipped = ia.Keypoint(
            x=image.shape[1]-kps.x,
            y=kps.y
        )

        seq = iaa.Fliplr(0.5)

        for bg, as_array in itertools.product([False, True], [False, True]):
            # with images as list
            nb_flipped_images = 0
            nb_flipped_keypoints = 0
            nb_iterations = 1000
            images = (
                np.uint8([np.copy(image)])
                if as_array
                else [np.copy(image)])
            batches = [
                ia.Batch(images=images,
                         keypoints=[kpsoi.deepcopy()])
                for _ in sm.xrange(nb_iterations)
            ]

            batches_aug = list(seq.augment_batches(batches, background=bg))

            for batch_aug in batches_aug:
                image_aug = batch_aug.images_aug[0]
                keypoint_aug = batch_aug.keypoints_aug[0].keypoints[0]

                img_matches_unflipped = np.array_equal(image_aug, image)
                img_matches_flipped = np.array_equal(image_aug, image_flipped)
                assert img_matches_unflipped or img_matches_flipped
                if img_matches_flipped:
                    nb_flipped_images += 1

                kp_matches_unflipped = (
                    np.isclose(keypoint_aug.x, kps.x)
                    and np.isclose(keypoint_aug.y, kps.y))
                kp_matches_flipped = (
                    np.isclose(keypoint_aug.x, kp_flipped.x)
                    and np.isclose(keypoint_aug.y, kp_flipped.y))
                assert kp_matches_flipped or kp_matches_unflipped
                if kp_matches_flipped:
                    nb_flipped_keypoints += 1
            assert 0.4*nb_iterations <= nb_flipped_images <= 0.6*nb_iterations
            assert nb_flipped_images == nb_flipped_keypoints

    def test_augment_batches_with_many_different_augmenters(self):
        image = np.array([[0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 1, 1, 1, 1]], dtype=np.uint8)
        keypoint = ia.Keypoint(x=2, y=1)
        keypoints = [ia.KeypointsOnImage([keypoint], shape=image.shape + (1,))]

        augs = [
            iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
            iaa.SomeOf(1, [iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
            iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
            iaa.Sometimes(1.0, iaa.Fliplr(1)),
            iaa.WithColorspace("HSV", children=iaa.Add((-50, 50))),
            iaa.WithChannels([0], iaa.Add((-50, 50))),
            iaa.Identity(name="Identity-nochange"),
            iaa.Lambda(
                func_images=_augment_batches__lambda_func_images,
                func_keypoints=_augment_batches__lambda_func_keypoints,
                name="Lambda-nochange"
            ),
            iaa.AssertLambda(
                func_images=_augment_batches__assertlambda_func_images,
                func_keypoints=_augment_batches__assertlambda_func_keypoints,
                name="AssertLambda-nochange"
            ),
            iaa.AssertShape(
                (None, 64, 64, 3),
                check_keypoints=False,
                name="AssertShape-nochange"
            ),
            iaa.Resize((0.5, 0.9)),
            iaa.CropAndPad(px=(-50, 50)),
            iaa.Pad(px=(1, 50)),
            iaa.Crop(px=(1, 50)),
            iaa.Fliplr(1.0),
            iaa.Flipud(1.0),
            iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128)),
            iaa.ChangeColorspace(to_colorspace="GRAY"),
            iaa.Grayscale(alpha=(0.1, 1.0)),
            iaa.GaussianBlur(1.0),
            iaa.AverageBlur(5),
            iaa.MedianBlur(5),
            iaa.Convolve(np.array([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]])),
            iaa.Sharpen(alpha=(0.1, 1.0), lightness=(0.8, 1.2)),
            iaa.Emboss(alpha=(0.1, 1.0), strength=(0.8, 1.2)),
            iaa.EdgeDetect(alpha=(0.1, 1.0)),
            iaa.DirectedEdgeDetect(alpha=(0.1, 1.0), direction=(0.0, 1.0)),
            iaa.Add((-50, 50)),
            iaa.AddElementwise((-50, 50)),
            iaa.AdditiveGaussianNoise(scale=(0.1, 1.0)),
            iaa.Multiply((0.6, 1.4)),
            iaa.MultiplyElementwise((0.6, 1.4)),
            iaa.Dropout((0.3, 0.5)),
            iaa.CoarseDropout((0.3, 0.5), size_percent=(0.05, 0.2)),
            iaa.Invert(0.5),
            iaa.Affine(
                scale=(0.7, 1.3),
                translate_percent=(-0.1, 0.1),
                rotate=(-20, 20),
                shear=(-20, 20),
                order=ia.ALL,
                mode=ia.ALL,
                cval=(0, 255)),
            iaa.PiecewiseAffine(scale=(0.1, 0.3)),
            iaa.ElasticTransformation(alpha=2.0)
        ]

        nb_iterations = 100
        image = ia.data.quokka(size=(64, 64))
        batches = [ia.Batch(images=[np.copy(image)],
                            keypoints=[keypoints[0].deepcopy()])
                   for _ in sm.xrange(nb_iterations)]
        for aug in augs:
            nb_changed = 0
            batches_aug = list(aug.augment_batches(batches, background=True))
            for batch_aug in batches_aug:
                image_aug = batch_aug.images_aug[0]
                if (image.shape != image_aug.shape
                        or not np.array_equal(image, image_aug)):
                    nb_changed += 1
                    if nb_changed > 10:
                        break
            if "-nochange" not in aug.name:
                assert nb_changed > 0
            else:
                assert nb_changed == 0


class TestAugmenter_augment_batch(unittest.TestCase):
    def test_deprecation(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            aug = _InplaceDummyAugmenterImgsArray(1)

            batch = ia.UnnormalizedBatch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8))
            _batch_aug = aug.augment_batch(batch)

            assert len(caught_warnings) == 1
            assert "is deprecated" in str(caught_warnings[0].message)

    def test_augments_correctly_images(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            image = np.arange(10*20).astype(np.uint8).reshape((10, 20, 1))
            image = np.tile(image, (1, 1, 3))
            image[:, :, 0] += 0
            image[:, :, 1] += 1
            image[:, :, 2] += 2
            images = image[np.newaxis, :, :, :]
            image_cp = np.copy(image)

            aug = _InplaceDummyAugmenterImgsArray(1)

            batch = ia.UnnormalizedBatch(images=images)
            batch_aug = aug.augment_batch(batch)

            image_unaug = batch_aug.images_unaug[0, :, :, :]
            image_aug = batch_aug.images_aug[0, :, :, :]

            assert batch_aug is batch
            assert batch_aug.images_aug is not batch.images_unaug
            assert batch_aug.images_aug is not batch_aug.images_unaug

            assert np.array_equal(image, image_cp)
            assert np.array_equal(image_unaug, image_cp)
            assert np.array_equal(image_aug, image_cp + 1)


class TestAugmenter_augment_batch_(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_verify_inplace_aug__imgs__unnormalized_batch(self):
        image = np.arange(10*20).astype(np.uint8).reshape((10, 20, 1))
        image = np.tile(image, (1, 1, 3))
        image[:, :, 0] += 0
        image[:, :, 1] += 1
        image[:, :, 2] += 2
        images = image[np.newaxis, :, :, :]
        image_cp = np.copy(image)

        aug = _InplaceDummyAugmenterImgsArray(1)

        batch = ia.UnnormalizedBatch(images=images)
        batch_aug = aug.augment_batch_(batch)
        image_unaug = batch_aug.images_unaug[0, :, :, :]
        image_aug = batch_aug.images_aug[0, :, :, :]

        assert batch_aug is batch
        assert batch_aug.images_aug is not batch.images_unaug
        assert batch_aug.images_aug is not batch_aug.images_unaug

        assert np.array_equal(image, image_cp)
        assert np.array_equal(image_unaug, image_cp)
        assert np.array_equal(image_aug, image_cp + 1)

    def test_verify_inplace_aug__imgs__normalized_batch(self):
        image = np.arange(10*20).astype(np.uint8).reshape((10, 20, 1))
        image = np.tile(image, (1, 1, 3))
        image[:, :, 0] += 0
        image[:, :, 1] += 1
        image[:, :, 2] += 2
        images = image[np.newaxis, :, :, :]
        image_cp = np.copy(image)

        aug = _InplaceDummyAugmenterImgsArray(1)

        batch = ia.Batch(images=images)
        batch_aug = aug.augment_batch_(batch)
        image_unaug = batch_aug.images_unaug[0, :, :, :]
        image_aug = batch_aug.images_aug[0, :, :, :]

        assert batch_aug is batch
        assert batch_aug.images_aug is not batch.images_unaug
        assert batch_aug.images_aug is not batch_aug.images_unaug

        assert np.array_equal(image, image_cp)
        assert np.array_equal(image_unaug, image_cp)
        assert np.array_equal(image_aug, image_cp + 1)

    def test_verify_inplace_aug__imgs__batchinaug(self):
        image = np.arange(10*20).astype(np.uint8).reshape((10, 20, 1))
        image = np.tile(image, (1, 1, 3))
        image[:, :, 0] += 0
        image[:, :, 1] += 1
        image[:, :, 2] += 2
        images = image[np.newaxis, :, :, :]
        image_cp = np.copy(image)

        aug = _InplaceDummyAugmenterImgsArray(1)

        batch = _BatchInAugmentation(images=images)
        batch_aug = aug.augment_batch_(batch)
        image_aug = batch_aug.images[0, :, :, :]

        assert batch_aug is batch
        assert batch_aug.images is batch.images

        assert not np.array_equal(image, image_cp)
        assert np.array_equal(image_aug, image_cp + 1)

    def test_verify_inplace_aug__segmaps__normalized_batch(self):
        segmap_arr = np.zeros((10, 20, 3), dtype=np.int32)
        segmap_arr[3:6, 3:9] = 1
        segmap = ia.SegmentationMapsOnImage(segmap_arr, shape=(10, 20, 3))
        segmap_cp = ia.SegmentationMapsOnImage(np.copy(segmap_arr),
                                               shape=(10, 20, 3))

        aug = _InplaceDummyAugmenterSegMaps(1)

        batch = ia.Batch(segmentation_maps=[segmap])
        batch_aug = aug.augment_batch_(batch)
        segmap_unaug = batch_aug.segmentation_maps_unaug[0]
        segmap_aug = batch_aug.segmentation_maps_aug[0]

        assert batch_aug is batch
        assert (batch_aug.segmentation_maps_aug
                is not batch.segmentation_maps_unaug)
        assert (batch_aug.segmentation_maps_aug
                is not batch_aug.segmentation_maps_unaug)

        assert np.array_equal(segmap.get_arr(), segmap_cp.get_arr())
        assert np.array_equal(segmap_unaug.get_arr(), segmap_cp.get_arr())
        assert np.array_equal(segmap_aug.get_arr(), segmap_cp.get_arr() + 1)

    def test_verify_inplace_aug__keypoints_normalized_batch(self):
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)],
                                    shape=(10, 20, 3))
        kpsoi_cp = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)],
                                       shape=(10, 20, 3))

        aug = _InplaceDummyAugmenterKeypoints(x=1, y=3)

        batch = ia.Batch(keypoints=[kpsoi])
        batch_aug = aug.augment_batch_(batch)
        kpsoi_unaug = batch_aug.keypoints_unaug[0]
        kpsoi_aug = batch_aug.keypoints_aug[0]

        assert batch_aug is batch
        assert (batch_aug.keypoints_aug
                is not batch.keypoints_unaug)
        assert (batch_aug.keypoints_aug
                is not batch_aug.keypoints_unaug)

        assert np.allclose(kpsoi.to_xy_array(), kpsoi_cp.to_xy_array())
        assert np.allclose(kpsoi_unaug.to_xy_array(), kpsoi_cp.to_xy_array())
        assert np.allclose(kpsoi_aug.to_xy_array()[:, 0],
                           kpsoi_cp.to_xy_array()[:, 0] + 1)
        assert np.allclose(kpsoi_aug.to_xy_array()[:, 1],
                           kpsoi_cp.to_xy_array()[:, 1] + 3)

    def test_call_changes_global_rng_state(self):
        state_before = copy.deepcopy(iarandom.get_global_rng().state)
        aug = iaa.Rot90(k=(0, 3))
        image = np.arange(4*4).astype(np.uint8).reshape((4, 4))
        batch = ia.UnnormalizedBatch(images=[image])

        _batch_aug = aug.augment_batch_(batch)

        state_after = iarandom.get_global_rng().state
        assert repr(state_before) != repr(state_after)

    def test_multiple_calls_produce_not_the_same_results(self):
        aug = iaa.Rot90(k=(0, 3))
        image = np.arange(4*4).astype(np.uint8).reshape((4, 4))
        nb_images = 1000
        batch1 = ia.UnnormalizedBatch(images=[image] * nb_images)
        batch2 = ia.UnnormalizedBatch(images=[image] * nb_images)
        batch3 = ia.UnnormalizedBatch(images=[image] * nb_images)

        batch_aug1 = aug.augment_batch_(batch1)
        batch_aug2 = aug.augment_batch_(batch2)
        batch_aug3 = aug.augment_batch_(batch3)

        assert batch_aug1 is not batch_aug2
        assert batch_aug1 is not batch_aug2
        assert batch_aug2 is not batch_aug3

        nb_equal = [0, 0, 0]
        for image_aug1, image_aug2, image_aug3 in zip(batch_aug1.images_aug,
                                                      batch_aug2.images_aug,
                                                      batch_aug3.images_aug):
            nb_equal[0] += int(np.array_equal(image_aug1, image_aug2))
            nb_equal[1] += int(np.array_equal(image_aug1, image_aug3))
            nb_equal[2] += int(np.array_equal(image_aug2, image_aug3))

        assert nb_equal[0] < (0.25 + 0.1) * nb_images
        assert nb_equal[1] < (0.25 + 0.1) * nb_images
        assert nb_equal[2] < (0.25 + 0.1) * nb_images

    def test_calls_affect_other_augmenters_with_global_rng(self):
        # with calling aug1
        iarandom.seed(1)
        aug1 = iaa.Rot90(k=(0, 3))
        aug2 = iaa.Add((0, 255))
        image = np.arange(4*4).astype(np.uint8).reshape((4, 4))
        nb_images = 50
        batch1 = ia.UnnormalizedBatch(images=[image] * 1)
        batch2 = ia.UnnormalizedBatch(images=[image] * nb_images)

        batch_aug11 = aug1.augment_batch_(batch1)
        batch_aug12 = aug2.augment_batch_(batch2)

        # with calling aug1, repetition (to see that seed() works)
        iarandom.seed(1)
        aug1 = iaa.Rot90(k=(0, 3))
        aug2 = iaa.Add((0, 255))
        image = np.arange(4*4).astype(np.uint8).reshape((4, 4))
        nb_images = 50
        batch1 = ia.UnnormalizedBatch(images=[image] * 1)
        batch2 = ia.UnnormalizedBatch(images=[image] * nb_images)

        batch_aug21 = aug1.augment_batch_(batch1)
        batch_aug22 = aug2.augment_batch_(batch2)

        # without calling aug1
        iarandom.seed(1)
        aug2 = iaa.Add((0, 255))
        image = np.arange(4*4).astype(np.uint8).reshape((4, 4))
        nb_images = 50
        batch2 = ia.UnnormalizedBatch(images=[image] * nb_images)

        batch_aug32 = aug2.augment_batch_(batch2)

        # comparison
        assert np.array_equal(
            np.array(batch_aug12.images_aug, dtype=np.uint8),
            np.array(batch_aug22.images_aug, dtype=np.uint8)
        )
        assert not np.array_equal(
            np.array(batch_aug12.images_aug, dtype=np.uint8),
            np.array(batch_aug32.images_aug, dtype=np.uint8)
        )


class TestAugmenter_augment_segmentation_maps(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_augment_segmentation_maps_single_instance(self):
        arr = np.int32([
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]
        ])
        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))
        aug = iaa.Identity()

        segmap_aug = aug.augment_segmentation_maps(segmap)

        assert np.array_equal(segmap_aug.arr, segmap.arr)

    def test_augment_segmentation_maps_list_of_single_instance(self):
        arr = np.int32([
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]
        ])
        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))
        aug = iaa.Identity()

        segmap_aug = aug.augment_segmentation_maps([segmap])[0]

        assert np.array_equal(segmap_aug.arr, segmap.arr)

    def test_augment_segmentation_maps_affine(self):
        arr = np.int32([
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]
        ])
        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))
        aug = iaa.Affine(translate_px={"x": 1})

        segmap_aug = aug.augment_segmentation_maps(segmap)

        expected = np.int32([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ])
        expected = expected[:, :, np.newaxis]
        assert np.array_equal(segmap_aug.arr, expected)

    def test_augment_segmentation_maps_pad(self):
        arr = np.int32([
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]
        ])
        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))
        aug = iaa.Pad(px=(1, 0, 0, 0), keep_size=False)

        segmap_aug = aug.augment_segmentation_maps(segmap)

        expected = np.int32([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]
        ])
        expected = expected[:, :, np.newaxis]
        assert np.array_equal(segmap_aug.arr, expected)

    def test_augment_segmentation_maps_pad_some_classes_not_provided(self):
        # only classes 0 and 3
        arr = np.int32([
            [0, 3, 3],
            [0, 3, 3],
            [0, 3, 3]
        ])
        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))
        aug = iaa.Pad(px=(1, 0, 0, 0), keep_size=False)

        segmap_aug = aug.augment_segmentation_maps(segmap)

        expected = np.int32([
            [0, 0, 0],
            [0, 3, 3],
            [0, 3, 3],
            [0, 3, 3]
        ])
        expected = expected[:, :, np.newaxis]
        assert np.array_equal(segmap_aug.arr, expected)

    def test_augment_segmentation_maps_pad_only_background_class(self):
        # only class 0
        arr = np.int32([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))
        aug = iaa.Pad(px=(1, 0, 0, 0), keep_size=False)

        segmap_aug = aug.augment_segmentation_maps(segmap)

        expected = np.int32([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        expected = expected[:, :, np.newaxis]
        assert np.array_equal(segmap_aug.arr, expected)

    def test_augment_segmentation_maps_multichannel_rot90(self):
        segmap = ia.SegmentationMapsOnImage(
            np.arange(0, 4*4).reshape((4, 4, 1)).astype(np.int32),
            shape=(4, 4, 3)
        )
        aug = iaa.Rot90(1, keep_size=False)

        segmaps_aug = aug.augment_segmentation_maps([segmap, segmap, segmap])

        for i in range(3):
            assert np.allclose(segmaps_aug[i].arr, np.rot90(segmap.arr, -1))


class TestAugmenter_draw_grid(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_draw_grid_list_of_3d_arrays(self):
        # list, shape (3, 3, 3)
        aug = _DummyAugmenter()
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        image[0, 0, :] = 10
        image[0, 1, :] = 50
        image[1, 1, :] = 255

        grid = aug.draw_grid([image], rows=2, cols=2)

        grid_expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert np.array_equal(grid, grid_expected)

    def test_draw_grid_list_of_2d_arrays(self):
        # list, shape (3, 3)
        aug = _DummyAugmenter()
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        image[0, 0, :] = 10
        image[0, 1, :] = 50
        image[1, 1, :] = 255

        grid = aug.draw_grid([image[..., 0]], rows=2, cols=2)

        grid_expected = np.vstack([
            np.hstack([image[..., 0:1], image[..., 0:1]]),
            np.hstack([image[..., 0:1], image[..., 0:1]])
        ])
        grid_expected = np.tile(grid_expected, (1, 1, 3))
        assert np.array_equal(grid, grid_expected)

    def test_draw_grid_list_of_1d_arrays_fails(self):
        # list, shape (2,)
        aug = _DummyAugmenter()

        with self.assertRaises(Exception):
            _ = aug.draw_grid([np.zeros((2,), dtype=np.uint8)], rows=2, cols=2)

    def test_draw_grid_4d_array(self):
        # array, shape (1, 3, 3, 3)
        aug = _DummyAugmenter()
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        image[0, 0, :] = 10
        image[0, 1, :] = 50
        image[1, 1, :] = 255

        grid = aug.draw_grid(np.uint8([image]), rows=2, cols=2)

        grid_expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert np.array_equal(grid, grid_expected)

    def test_draw_grid_3d_array(self):
        # array, shape (3, 3, 3)
        aug = _DummyAugmenter()
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        image[0, 0, :] = 10
        image[0, 1, :] = 50
        image[1, 1, :] = 255

        grid = aug.draw_grid(image, rows=2, cols=2)

        grid_expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert np.array_equal(grid, grid_expected)

    def test_draw_grid_2d_array(self):
        # array, shape (3, 3)
        aug = _DummyAugmenter()
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        image[0, 0, :] = 10
        image[0, 1, :] = 50
        image[1, 1, :] = 255

        grid = aug.draw_grid(image[..., 0], rows=2, cols=2)

        grid_expected = np.vstack([
            np.hstack([image[..., 0:1], image[..., 0:1]]),
            np.hstack([image[..., 0:1], image[..., 0:1]])
        ])
        grid_expected = np.tile(grid_expected, (1, 1, 3))
        assert np.array_equal(grid, grid_expected)

    def test_draw_grid_1d_array(self):
        # array, shape (2,)
        aug = _DummyAugmenter()

        with self.assertRaises(Exception):
            _ = aug.draw_grid(np.zeros((2,), dtype=np.uint8), rows=2, cols=2)


@six.add_metaclass(ABCMeta)
class _TestAugmenter_augment_cbaois(object):
    """Class that is used to test augment_polygons() and augment_line_strings().

    Originally this was only used for polygons and then made more flexible.
    This is why some descriptions are still geared towards polygons.

    Abbreviations:
        cba = coordinate based augmentable, e.g. Polygon
        cbaoi = coordinate based augmentable on image, e.g. PolygonsOnImage

    """

    def setUp(self):
        reseed()

    @abstractmethod
    def _augfunc(self, augmenter, *args, **kwargs):
        """Return augmenter.augment_*(...)."""

    @property
    @abstractmethod
    def _ObjClass(self):
        """Return Polygon, LineString or similar class."""

    @property
    @abstractmethod
    def _ObjOnImageClass(self):
        """Return PolygonsOnImage, LineStringsOnImage or similar class."""

    def _Obj(self, *args, **kwargs):
        return self._ObjClass(*args, **kwargs)

    def _ObjOnImage(self, *args, **kwargs):
        return self._ObjOnImageClass(*args, **kwargs)

    def _compare_coords_of_cba(self, observed, expected, atol=1e-4, rtol=0):
        return np.allclose(observed, expected, atol=atol, rtol=rtol)

    def test_single_empty_instance(self):
        # single instance of PolygonsOnImage with 0 polygons
        aug = iaa.Rot90(1, keep_size=False)
        cbaoi = self._ObjOnImage([], shape=(10, 11, 3))

        cbaoi_aug = self._augfunc(aug, cbaoi)

        assert isinstance(cbaoi_aug, self._ObjOnImageClass)
        assert cbaoi_aug.empty
        assert cbaoi_aug.shape == (11, 10, 3)

    def test_list_of_single_empty_instance(self):
        # list of PolygonsOnImage with 0 polygons
        aug = iaa.Rot90(1, keep_size=False)
        cbaoi = self._ObjOnImage([], shape=(10, 11, 3))

        cbaois_aug = self._augfunc(aug, [cbaoi])

        assert isinstance(cbaois_aug, list)
        assert isinstance(cbaois_aug[0], self._ObjOnImageClass)
        assert cbaois_aug[0].empty
        assert cbaois_aug[0].shape == (11, 10, 3)

    def test_two_cbaois_each_two_cbas(self):
        # 2 PolygonsOnImage, each 2 polygons
        aug = iaa.Rot90(1, keep_size=False)
        cbaois = [
            self._ObjOnImage(
                [self._Obj([(0, 0), (5, 0), (5, 5)]),
                 self._Obj([(1, 1), (6, 1), (6, 6)])],
                shape=(10, 10, 3)),
            self._ObjOnImage(
                [self._Obj([(2, 2), (7, 2), (7, 7)]),
                 self._Obj([(3, 3), (8, 3), (8, 8)])],
                shape=(10, 10, 3)),
        ]

        cbaois_aug = self._augfunc(aug, cbaois)

        assert isinstance(cbaois_aug, list)
        assert isinstance(cbaois_aug[0], self._ObjOnImageClass)
        assert isinstance(cbaois_aug[0], self._ObjOnImageClass)
        assert len(cbaois_aug[0].items) == 2
        assert len(cbaois_aug[1].items) == 2
        kp_offset = 0
        assert self._compare_coords_of_cba(
            cbaois_aug[0].items[0].coords,
            [(10-0+kp_offset, 0), (10-0+kp_offset, 5), (10-5+kp_offset, 5)],
            atol=1e-4, rtol=0
        )
        assert self._compare_coords_of_cba(
            cbaois_aug[0].items[1].coords,
            [(10-1+kp_offset, 1), (10-1+kp_offset, 6), (10-6+kp_offset, 6)],
            atol=1e-4, rtol=0
        )
        assert self._compare_coords_of_cba(
            cbaois_aug[1].items[0].coords,
            [(10-2+kp_offset, 2), (10-2+kp_offset, 7), (10-7+kp_offset, 7)],
            atol=1e-4, rtol=0
        )
        assert self._compare_coords_of_cba(
            cbaois_aug[1].items[1].coords,
            [(10-3+kp_offset, 3), (10-3+kp_offset, 8), (10-8+kp_offset, 8)],
            atol=1e-4, rtol=0
        )
        assert cbaois_aug[0].shape == (10, 10, 3)
        assert cbaois_aug[1].shape == (10, 10, 3)

    def test_randomness_between_and_within_batches(self):
        # test whether there is randomness within each batch and between
        # batches
        aug = iaa.Rot90((0, 3), keep_size=False)
        cba = self._Obj([(0, 0), (5, 0), (5, 5)])
        cbaoi = self._ObjOnImage(
            [cba.deepcopy() for _ in sm.xrange(1)],
            shape=(10, 11, 3)
        )
        cbaois = [cbaoi.deepcopy() for _ in sm.xrange(100)]

        cbaois_aug1 = self._augfunc(aug, cbaois)
        cbaois_aug2 = self._augfunc(aug, cbaois)

        # --> different between runs
        cbas1 = [cba
                 for cbaoi in cbaois_aug1
                 for cba in cbaoi.items]
        cbas2 = [cba
                 for cbaoi in cbaois_aug2
                 for cba in cbaoi.items]
        assert len(cbas1) == len(cbas2)
        same = []
        for cba1, cba2 in zip(cbas1, cbas2):
            points1 = np.float32(cba1.coords)
            points2 = np.float32(cba2.coords)
            same.append(self._compare_coords_of_cba(points1, points2,
                                                    atol=1e-2, rtol=0))
        assert not np.all(same)

        # --> different between PolygonOnImages
        same = []
        points1 = np.float32([cba.coords
                              for cba
                              in cbaois_aug1[0].items])
        for cba in cbaois_aug1[1:]:
            points2 = np.float32([cba.coords
                                  for cba
                                  in cba.items])
            same.append(self._compare_coords_of_cba(points1, points2,
                                                    atol=1e-2, rtol=0))
        assert not np.all(same)

        # --> different between polygons
        points1 = set()
        for cba in cbaois_aug1[0].items:
            for point in cba.coords:
                points1.add(tuple(
                    [int(point[0]*10), int(point[1]*10)]
                ))
        assert len(points1) > 1

    def test_determinism(self):
        aug = iaa.Rot90((0, 3), keep_size=False)
        aug_det = aug.to_deterministic()
        cba = self._Obj([(0, 0), (5, 0), (5, 5)])
        cbaoi = self._ObjOnImage(
            [cba.deepcopy() for _ in sm.xrange(1)],
            shape=(10, 11, 3)
        )
        cbaois = [cbaoi.deepcopy() for _ in sm.xrange(100)]

        cbaois_aug1 = self._augfunc(aug_det, cbaois)
        cbaois_aug2 = self._augfunc(aug_det, cbaois)

        # --> different between PolygonsOnImages
        same = []
        points1 = np.float32([cba.coords
                              for cba
                              in cbaois_aug1[0].items])
        for cbaoi in cbaois_aug1[1:]:
            points2 = np.float32([cba.coords
                                  for cba
                                  in cbaoi.items])
            same.append(self._compare_coords_of_cba(points1, points2,
                                                    atol=1e-2, rtol=0))
        assert not np.all(same)

        # --> similar between augmentation runs
        cbas1 = [cba
                 for cbaoi in cbaois_aug1
                 for cba in cbaoi.items]
        cbas2 = [cba
                 for cbaoi in cbaois_aug2
                 for cba in cbaoi.items]
        assert len(cbas1) == len(cbas2)
        for cba1, cba2 in zip(cbas1, cbas2):
            points1 = np.float32(cba1.coords)
            points2 = np.float32(cba2.coords)
            assert self._compare_coords_of_cba(points1, points2,
                                               atol=1e-2, rtol=0)

    def test_aligned_with_images(self):
        aug = iaa.Rot90((0, 3), keep_size=False)
        aug_det = aug.to_deterministic()
        image = np.zeros((10, 20), dtype=np.uint8)
        image[5, :] = 255
        image[2:5, 10] = 255
        image_rots = [iaa.Rot90(k, keep_size=False).augment_image(image)
                      for k in [0, 1, 2, 3]]
        cba = self._Obj([(0, 0), (10, 0), (10, 20)])
        kp_offs = 0  # offset
        cbas_rots = [
            [(0, 0), (10, 0), (10, 20)],
            [(10-0+kp_offs, 0), (10-0+kp_offs, 10), (10-20+kp_offs, 10)],
            [(20-0+kp_offs, 10), (20-10+kp_offs, 10), (20-10+kp_offs, -10)],
            [(10-10+kp_offs, 20), (10-10+kp_offs, 10), (10-(-10)+kp_offs, 10)]
        ]
        cbaois = [self._ObjOnImage([cba], shape=image.shape)
                  for _ in sm.xrange(50)]

        images_aug = aug_det.augment_images([image] * 50)
        cbaois_aug = self._augfunc(aug_det, cbaois)

        seen = set()
        for image_aug, cbaoi_aug in zip(images_aug, cbaois_aug):
            found_image = False
            for img_rot_idx, img_rot in enumerate(image_rots):
                if (image_aug.shape == img_rot.shape
                        and np.allclose(image_aug, img_rot)):
                    found_image = True
                    break

            found_cba = False
            for poly_rot_idx, cba_rot in enumerate(cbas_rots):
                coords_observed = cbaoi_aug.items[0].coords
                if self._compare_coords_of_cba(coords_observed, cba_rot):
                    found_cba = True
                    break

            assert found_image
            assert found_cba
            assert img_rot_idx == poly_rot_idx
            seen.add((img_rot_idx, poly_rot_idx))
        assert 2 <= len(seen) <= 4  # assert not always the same rot

    def test_aligned_with_images_despite_empty_instances(self):
        # Test if augmenting lists of e.g. PolygonsOnImage is still aligned
        # with image augmentation when one e.g. PolygonsOnImage instance is
        # empty (e.g. contains no polygons)
        cba = self._Obj([(0, 0), (5, 0), (5, 5), (0, 5)])
        cbaoi_lst = [
            self._ObjOnImage([cba.deepcopy()], shape=(10, 20)),
            self._ObjOnImage([cba.deepcopy()], shape=(10, 20)),
            self._ObjOnImage([cba.shift(x=1)], shape=(10, 20)),
            self._ObjOnImage([cba.deepcopy()], shape=(10, 20)),
            self._ObjOnImage([cba.deepcopy()], shape=(10, 20)),
            self._ObjOnImage([], shape=(1, 8)),
            self._ObjOnImage([cba.deepcopy()], shape=(10, 20)),
            self._ObjOnImage([cba.deepcopy()], shape=(10, 20)),
            self._ObjOnImage([cba.shift(x=1)], shape=(10, 20)),
            self._ObjOnImage([cba.deepcopy()], shape=(10, 20)),
            self._ObjOnImage([cba.deepcopy()], shape=(10, 20))
        ]
        image = np.zeros((10, 20), dtype=np.uint8)
        image[0, 0] = 255
        image[0, 5] = 255
        image[5, 5] = 255
        image[5, 0] = 255
        images = np.tile(image[np.newaxis, :, :], (len(cbaoi_lst), 1, 1))

        aug = iaa.Affine(translate_px={"x": (0, 8)}, order=0, mode="constant",
                         cval=0)

        for _ in sm.xrange(10):
            for is_list in [False, True]:
                aug_det = aug.to_deterministic()
                inputs = images
                if is_list:
                    inputs = list(inputs)

                images_aug = aug_det.augment_images(inputs)
                cbaoi_aug_lst = self._augfunc(aug_det, cbaoi_lst)

                if is_list:
                    images_aug = np.array(images_aug, dtype=np.uint8)
                translations_imgs = np.argmax(images_aug[:, 0, :], axis=1)

                translations_points = [
                    (cbaoi.items[0].coords[0][0] if not cbaoi.empty else None)
                    for cbaoi
                    in cbaoi_aug_lst]

                assert len([
                    pointresult for
                    pointresult
                    in translations_points
                    if pointresult is None
                ]) == 1
                assert translations_points[5] is None
                translations_imgs = np.concatenate(
                    [translations_imgs[0:5], translations_imgs[6:]])
                translations_points = np.array(
                    translations_points[0:5] + translations_points[6:],
                    dtype=translations_imgs.dtype)
                translations_points[2] -= 1
                translations_points[8-1] -= 1
                assert np.array_equal(translations_imgs, translations_points)


# This is the same as _ConcavePolygonRecoverer, but we make sure that we
# always sample random values. This is to advance the state of random_state
# and ensure that this breaks not alignment.
class _DummyRecoverer(_ConcavePolygonRecoverer):
    def recover_from(self, new_exterior, old_polygon, random_state=0):
        # sample lots of values to ensure that the RNG is advanced
        _ = random_state.integers(0, 2**30, 100)
        return super(_DummyRecoverer, self).recover_from(
            new_exterior, old_polygon, random_state=random_state)


class _DummyAugmenterWithRecoverer(iaa.Augmenter):
    def __init__(self, use_recoverer=True):
        super(_DummyAugmenterWithRecoverer, self).__init__()
        self.random_samples_images = []
        self.random_samples_kps = []

        if use_recoverer:
            self.recoverer = _DummyRecoverer()
        else:
            self.recoverer = None

    def _augment_images(self, images, random_state, parents, hooks):
        sample = random_state.integers(0, 2**30)
        self.random_samples_images.append(sample)
        return images

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        return self._augment_polygons_as_keypoints(
            polygons_on_images, random_state, parents, hooks,
            recoverer=self.recoverer)

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        sample = random_state.integers(0, 2**30)
        self.random_samples_kps.append(sample)

        assert len(keypoints_on_images) in [1, 2]
        assert len(keypoints_on_images[0].keypoints) == 7

        result = []

        for _ in keypoints_on_images:
            # every second call of _augment_polygons()...
            if len(self.random_samples_kps) % 2 == 1:
                # not concave
                kpsoi = ia.KeypointsOnImage([
                    ia.Keypoint(x=0, y=0),
                    ia.Keypoint(x=10, y=0),
                    ia.Keypoint(x=10, y=4),
                    ia.Keypoint(x=-1, y=5),
                    ia.Keypoint(x=10, y=6),
                    ia.Keypoint(x=10, y=10),
                    ia.Keypoint(x=0, y=10)
                ], shape=(10, 10, 3))
            else:
                # concave
                kpsoi = ia.KeypointsOnImage([
                    ia.Keypoint(x=0, y=0),
                    ia.Keypoint(x=10, y=0),
                    ia.Keypoint(x=10, y=4),
                    ia.Keypoint(x=10, y=5),
                    ia.Keypoint(x=10, y=6),
                    ia.Keypoint(x=10, y=10),
                    ia.Keypoint(x=0, y=10)
                ], shape=(10, 10, 3))
            result.append(kpsoi)
        return result

    def get_parameters(self):
        return []


class TestAugmenter_augment_polygons(_TestAugmenter_augment_cbaois,
                                     unittest.TestCase):
    def _augfunc(self, augmenter, *args, **kwargs):
        return augmenter.augment_polygons(*args, **kwargs)

    @property
    def _ObjClass(self):
        return ia.Polygon

    @property
    def _ObjOnImageClass(self):
        return ia.PolygonsOnImage

    def _coords(self, obj):
        return obj.exterior
    
    def _entities(self, obj_on_image):
        return obj_on_image.polygons

    def test_polygon_recoverer(self):
        # This is mostly a dummy polygon. The augmenter always returns the
        # same non-concave polygon.
        poly = ia.Polygon([(0, 0), (10, 0),
                           (10, 4), (10, 5), (10, 6),
                           (10, 10), (0, 10)])
        psoi = ia.PolygonsOnImage([poly], shape=(10, 10, 3))
        aug = _DummyAugmenterWithRecoverer()

        psoi_aug = aug.augment_polygons(psoi)
        poly_aug = psoi_aug.polygons[0]

        bb = ia.BoundingBox(x1=0, y1=0, x2=10, y2=10)
        bb_aug = ia.BoundingBox(
            x1=np.min(poly_aug.exterior[:, 0]),
            y1=np.min(poly_aug.exterior[:, 1]),
            x2=np.max(poly_aug.exterior[:, 0]),
            y2=np.max(poly_aug.exterior[:, 1])
        )
        assert bb.iou(bb_aug) > 0.9
        assert psoi_aug.polygons[0].is_valid

    def test_polygon_aligned_without_recoverer(self):
        # This is mostly a dummy polygon. The augmenter always returns the
        # same non-concave polygon.
        poly = ia.Polygon([(0, 0), (10, 0),
                           (10, 4), (10, 5), (10, 6),
                           (10, 10), (0, 10)])
        psoi = ia.PolygonsOnImage([poly], shape=(10, 10, 3))
        image = np.zeros((10, 10, 3))
        aug = _DummyAugmenterWithRecoverer(use_recoverer=False)

        images_aug1, psois_aug1 = aug(images=[image, image],
                                      polygons=[psoi, psoi])

        images_aug2, psois_aug2 = aug(images=[image, image],
                                      polygons=[psoi, psoi])

        images_aug3, psois_aug3 = aug(images=[image, image],
                                      polygons=[psoi, psoi])

        images_aug4, psois_aug4 = aug(images=[image, image],
                                      polygons=[psoi, psoi])

        assert not psois_aug1[0].polygons[0].is_valid
        assert not psois_aug1[1].polygons[0].is_valid
        assert psois_aug2[0].polygons[0].is_valid
        assert psois_aug2[1].polygons[0].is_valid
        assert not psois_aug3[0].polygons[0].is_valid
        assert not psois_aug3[1].polygons[0].is_valid
        assert psois_aug4[0].polygons[0].is_valid
        assert psois_aug4[1].polygons[0].is_valid

        assert aug.random_samples_images == aug.random_samples_kps

    def test_polygon_aligned_with_recoverer(self):
        # This is mostly a dummy polygon. The augmenter always returns the
        # same non-concave polygon.
        poly = ia.Polygon([(0, 0), (10, 0),
                           (10, 4), (10, 5), (10, 6),
                           (10, 10), (0, 10)])
        psoi = ia.PolygonsOnImage([poly], shape=(10, 10, 3))
        image = np.zeros((10, 10, 3))
        aug = _DummyAugmenterWithRecoverer(use_recoverer=True)

        images_aug1, psois_aug1 = aug(images=[image, image],
                                      polygons=[psoi, psoi])

        images_aug2, psois_aug2 = aug(images=[image, image],
                                      polygons=[psoi, psoi])

        images_aug3, psois_aug3 = aug(images=[image, image],
                                      polygons=[psoi, psoi])

        images_aug4, psois_aug4 = aug(images=[image, image],
                                      polygons=[psoi, psoi])

        assert psois_aug1[0].polygons[0].is_valid
        assert psois_aug1[1].polygons[0].is_valid
        assert psois_aug2[0].polygons[0].is_valid
        assert psois_aug2[1].polygons[0].is_valid
        assert psois_aug3[0].polygons[0].is_valid
        assert psois_aug3[1].polygons[0].is_valid
        assert psois_aug4[0].polygons[0].is_valid
        assert psois_aug4[1].polygons[0].is_valid

        assert aug.random_samples_images == aug.random_samples_kps


class TestAugmenter_augment_line_strings(_TestAugmenter_augment_cbaois,
                                         unittest.TestCase):
    def _augfunc(self, augmenter, *args, **kwargs):
        return augmenter.augment_line_strings(*args, **kwargs)

    @property
    def _ObjClass(self):
        return ia.LineString

    @property
    def _ObjOnImageClass(self):
        return ia.LineStringsOnImage


class TestAugmenter_augment_bounding_boxes(_TestAugmenter_augment_cbaois,
                                           unittest.TestCase):
    def _augfunc(self, augmenter, *args, **kwargs):
        return augmenter.augment_bounding_boxes(*args, **kwargs)

    @property
    def _ObjClass(self):
        return ia.BoundingBox

    @property
    def _ObjOnImageClass(self):
        return ia.BoundingBoxesOnImage

    def _Obj(self, *args, **kwargs):
        assert len(args) == 1
        coords = np.float32(args[0]).reshape((-1, 2))
        x1 = np.min(coords[:, 0])
        y1 = np.min(coords[:, 1])
        x2 = np.max(coords[:, 0])
        y2 = np.max(coords[:, 1])
        return self._ObjClass(x1=x1, y1=y1, x2=x2, y2=y2, **kwargs)

    def _compare_coords_of_cba(self, observed, expected, atol=1e-4, rtol=0):
        observed = np.float32(observed).reshape((-1, 2))
        expected = np.float32(expected).reshape((-1, 2))
        assert observed.shape[0] == 2
        assert expected.shape[1] == 2

        obs_x1 = np.min(observed[:, 0])
        obs_y1 = np.min(observed[:, 1])
        obs_x2 = np.max(observed[:, 0])
        obs_y2 = np.max(observed[:, 1])

        exp_x1 = np.min(expected[:, 0])
        exp_y1 = np.min(expected[:, 1])
        exp_x2 = np.max(expected[:, 0])
        exp_y2 = np.max(expected[:, 1])

        return np.allclose(
            [obs_x1, obs_y1, obs_x2, obs_y2],
            [exp_x1, exp_y1, exp_x2, exp_y2],
            atol=atol, rtol=rtol)


# the method is mostly tested indirectly, so very few tests here
class TestAugmenter_augment_bounding_boxes_by_keypoints(unittest.TestCase):
    def test_x_min_max(self):
        # ensure that min() and max() are applied to augmented x-coordinates
        # when they are converted back to BBs

        class _ShiftingXCoordAugmenter(iaa.Augmenter):
            def _augment_images(self, images, random_state, parents, hooks):
                return images

            def _augment_bounding_boxes(self, bounding_boxes_on_images,
                                        random_state, parents, hooks):
                return self._augment_bounding_boxes_as_keypoints(
                    bounding_boxes_on_images, random_state, parents, hooks)

            def _augment_keypoints(self, keypoints_on_images, random_state,
                                   parents, hooks):
                keypoints_on_images[0].keypoints[0].x += 10
                keypoints_on_images[0].keypoints[1].x -= 10
                return keypoints_on_images

            def get_parameters(self):
                return []

        aug = _ShiftingXCoordAugmenter()
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)], shape=(10, 10, 3))
        observed = aug(bounding_boxes=bbsoi)
        assert np.allclose(
            observed.bounding_boxes[0].coords,
            [(2-10, 1), (0+10, 3)]
        )

    def test_y_min_max(self):
        # ensure that min() and max() are applied to augmented y-coordinates
        # when they are converted back to BBs

        class _ShiftingYCoordAugmenter(iaa.Augmenter):
            def _augment_images(self, images, random_state, parents, hooks):
                return images

            def _augment_bounding_boxes(self, bounding_boxes_on_images,
                                        random_state, parents, hooks):
                return self._augment_bounding_boxes_as_keypoints(
                    bounding_boxes_on_images, random_state, parents, hooks)

            def _augment_keypoints(self, keypoints_on_images, random_state,
                                   parents, hooks):
                keypoints_on_images[0].keypoints[0].y += 10
                keypoints_on_images[0].keypoints[1].y -= 10
                return keypoints_on_images

            def get_parameters(self):
                return []

        aug = _ShiftingYCoordAugmenter()
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)], shape=(10, 10, 3))
        observed = aug(bounding_boxes=bbsoi)
        assert np.allclose(
            observed.bounding_boxes[0].coords,
            [(0, 1-10), (2, 1+10)]
        )

    def test_x1_x2_can_get_flipped(self):
        # ensure that augmented x-coordinates where x1>x2 are flipped
        # before creating BBs from them

        class _FlippingX1X2Augmenter(iaa.Augmenter):
            def _augment_images(self, images, random_state, parents, hooks):
                return images

            def _augment_bounding_boxes(self, bounding_boxes_on_images,
                                        random_state, parents, hooks):
                return self._augment_bounding_boxes_as_keypoints(
                    bounding_boxes_on_images, random_state, parents, hooks)

            def _augment_keypoints(self, keypoints_on_images, random_state,
                                   parents, hooks):
                keypoints_on_images[0].keypoints[0].x += 10  # top left
                keypoints_on_images[0].keypoints[3].x += 10  # bottom left
                return keypoints_on_images

            def get_parameters(self):
                return []

        aug = _FlippingX1X2Augmenter()
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)], shape=(10, 10, 3))
        observed = aug(bounding_boxes=bbsoi)
        assert np.allclose(
            observed.bounding_boxes[0].coords,
            [(2, 1), (0+10, 3)]
        )

    def test_y1_y2_can_get_flipped(self):
        # ensure that augmented y-coordinates where y1>y2 are flipped
        # before creating BBs from them

        class _FlippingY1Y2Augmenter(iaa.Augmenter):
            def _augment_images(self, images, random_state, parents, hooks):
                return images

            def _augment_bounding_boxes(self, bounding_boxes_on_images,
                                        random_state, parents, hooks):
                return self._augment_bounding_boxes_as_keypoints(
                    bounding_boxes_on_images, random_state, parents, hooks)

            def _augment_keypoints(self, keypoints_on_images, random_state,
                                   parents, hooks):
                keypoints_on_images[0].keypoints[0].y += 10  # top left
                keypoints_on_images[0].keypoints[1].y += 10  # top right
                return keypoints_on_images

            def get_parameters(self):
                return []

        aug = _FlippingY1Y2Augmenter()
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)], shape=(10, 10, 3))
        observed = aug(bounding_boxes=bbsoi)
        assert np.allclose(
            observed.bounding_boxes[0].coords,
            [(0, 3), (2, 1+10)]
        )


class TestAugmenter_augment(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_image(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")

        image_aug = aug.augment(image=image)

        assert image_aug.shape == image.shape
        assert np.array_equal(image_aug, image)

    def test_images_list(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")

        images_aug = aug.augment(images=[image])

        assert images_aug[0].shape == image.shape
        assert np.array_equal(images_aug[0], image)

    def test_images_and_heatmaps(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        heatmaps = ia.data.quokka_heatmap((128, 128), extract="square")

        images_aug, heatmaps_aug = aug.augment(images=[image],
                                               heatmaps=[heatmaps])

        assert np.array_equal(images_aug[0], image)
        assert np.allclose(heatmaps_aug[0].arr_0to1, heatmaps.arr_0to1)

    def test_images_and_segmentation_maps(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")

        images_aug, segmaps_aug = aug.augment(images=[image],
                                              segmentation_maps=[segmaps])
        assert np.array_equal(images_aug[0], image)
        assert np.allclose(segmaps_aug[0].arr, segmaps.arr)

    def test_images_and_keypoints(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")

        images_aug, keypoints_aug = aug.augment(images=[image],
                                                keypoints=[keypoints])

        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(keypoints_aug[0], keypoints)

    def test_images_and_polygons(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")

        images_aug, polygons_aug = aug.augment(images=[image],
                                               polygons=[polygons])

        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(polygons_aug[0], polygons)

    def test_images_and_line_strings(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        psoi = ia.data.quokka_polygons((128, 128), extract="square")
        lsoi = ia.LineStringsOnImage([
            psoi.polygons[0].to_line_string(closed=False)
        ], shape=psoi.shape)

        images_aug, lsoi_aug = aug.augment(images=[image],
                                           line_strings=[lsoi])

        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(lsoi_aug[0], lsoi)

    def test_images_and_bounding_boxes(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        bbs = ia.data.quokka_bounding_boxes((128, 128), extract="square")

        images_aug, bbs_aug = aug.augment(images=[image], bounding_boxes=[bbs])

        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(bbs_aug[0], bbs)

    def test_image_return_batch(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")

        batch = aug.augment(image=image, return_batch=True)

        image_aug = batch.images_aug[0]
        assert np.array_equal(image, image_aug)

    def test_images_and_heatmaps_return_batch(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        heatmaps = ia.data.quokka_heatmap((128, 128), extract="square")

        batch = aug.augment(images=[image], heatmaps=[heatmaps],
                            return_batch=True)

        images_aug = batch.images_aug
        heatmaps_aug = batch.heatmaps_aug
        assert np.array_equal(images_aug[0], image)
        assert np.allclose(heatmaps_aug[0].arr_0to1, heatmaps.arr_0to1)

    def test_images_and_segmentation_maps_return_batch(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")

        batch = aug.augment(images=[image], segmentation_maps=[segmaps],
                            return_batch=True)

        images_aug = batch.images_aug
        segmaps_aug = batch.segmentation_maps_aug
        assert np.array_equal(images_aug[0], image)
        assert np.allclose(segmaps_aug[0].arr, segmaps.arr)

    def test_images_and_keypoints_return_batch(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")

        batch = aug.augment(images=[image], keypoints=[keypoints],
                            return_batch=True)

        images_aug = batch.images_aug
        keypoints_aug = batch.keypoints_aug
        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(keypoints_aug[0], keypoints)

    def test_images_and_polygons_return_batch(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")

        batch = aug.augment(images=[image], polygons=[polygons],
                            return_batch=True)

        images_aug = batch.images_aug
        polygons_aug = batch.polygons_aug
        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(polygons_aug[0], polygons)

    def test_images_and_line_strings_return_batch(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        psoi = ia.data.quokka_polygons((128, 128), extract="square")
        lsoi = ia.LineStringsOnImage([
            psoi.polygons[0].to_line_string(closed=False)
        ], shape=psoi.shape)

        batch = aug.augment(images=[image], line_strings=[lsoi],
                            return_batch=True)

        images_aug = batch.images_aug
        lsoi_aug = batch.line_strings_aug
        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(lsoi_aug[0], lsoi)

    def test_images_and_bounding_boxes_return_batch(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        bbs = ia.data.quokka_bounding_boxes((128, 128), extract="square")

        batch = aug.augment(images=[image], bounding_boxes=[bbs],
                            return_batch=True)

        images_aug = batch.images_aug
        bbs_aug = batch.bounding_boxes_aug
        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(bbs_aug[0], bbs)

    def test_non_image_data(self):
        aug = iaa.Identity()
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")

        batch = aug.augment(segmentation_maps=[segmaps], keypoints=[keypoints],
                            polygons=[polygons], return_batch=True)

        segmaps_aug = batch.segmentation_maps_aug
        keypoints_aug = batch.keypoints_aug
        polygons_aug = batch.polygons_aug
        assert np.allclose(segmaps_aug[0].arr, segmaps.arr)
        assert_cbaois_equal(keypoints_aug[0], keypoints)
        assert_cbaois_equal(polygons_aug[0], polygons)

    def test_non_image_data_unexpected_args_order(self):
        aug = iaa.Identity()
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")

        batch = aug.augment(polygons=[polygons], segmentation_maps=[segmaps],
                            keypoints=[keypoints], return_batch=True)

        segmaps_aug = batch.segmentation_maps_aug
        keypoints_aug = batch.keypoints_aug
        polygons_aug = batch.polygons_aug
        assert np.allclose(segmaps_aug[0].arr, segmaps.arr)
        assert np.allclose(keypoints_aug[0].to_xy_array(),
                           keypoints.to_xy_array())
        for polygon_aug, polygon in zip(polygons_aug[0].polygons,
                                        polygons.polygons):
            assert polygon_aug.exterior_almost_equals(polygon)

    def test_with_affine(self):
        # make sure that augment actually does something
        aug = iaa.Affine(translate_px={"x": 1}, order=0, mode="constant",
                         cval=0)
        image = np.zeros((4, 4, 1), dtype=np.uint8) + 255
        heatmaps = np.ones((1, 4, 4, 1), dtype=np.float32)
        segmaps = np.ones((1, 4, 4, 1), dtype=np.int32)
        kps = [(0, 0), (1, 2)]
        bbs = [(0, 0, 1, 1), (1, 2, 2, 3)]
        polygons = [(0, 0), (1, 0), (1, 1)]
        ls = [(0, 0), (1, 0), (1, 1)]

        image_aug = aug.augment(image=image)
        _, heatmaps_aug = aug.augment(image=image, heatmaps=heatmaps)
        _, segmaps_aug = aug.augment(image=image, segmentation_maps=segmaps)
        _, kps_aug = aug.augment(image=image, keypoints=kps)
        _, bbs_aug = aug.augment(image=image, bounding_boxes=bbs)
        _, polygons_aug = aug.augment(image=image, polygons=polygons)
        _, ls_aug = aug.augment(image=image, line_strings=ls)

        # all augmentables must have been moved to the right by 1px
        assert np.all(image_aug[:, 0] == 0)
        assert np.all(image_aug[:, 1:] == 255)
        assert np.allclose(heatmaps_aug[0][:, 0], 0.0)
        assert np.allclose(heatmaps_aug[0][:, 1:], 1.0)
        assert np.all(segmaps_aug[0][:, 0] == 0)
        assert np.all(segmaps_aug[0][:, 1:] == 1)
        assert np.allclose(kps_aug, [(1, 0), (2, 2)])
        assert np.allclose(bbs_aug, [(1, 0, 2, 1), (2, 2, 3, 3)])
        assert np.allclose(polygons_aug, [(1, 0), (2, 0), (2, 1)])
        assert np.allclose(ls_aug, [(1, 0), (2, 0), (2, 1)])

    def test_alignment(self):
        # make sure that changes from augment() are aligned and vary between
        # call
        aug = iaa.Affine(translate_px={"x": (0, 100)}, order=0, mode="constant",
                         cval=0)
        image = np.zeros((1, 100, 1), dtype=np.uint8) + 255
        heatmaps = np.ones((1, 1, 100, 1), dtype=np.float32)
        segmaps = np.ones((1, 1, 100, 1), dtype=np.int32)
        kps = [(0, 0)]
        bbs = [(0, 0, 1, 1)]
        polygons = [(0, 0), (1, 0), (1, 1)]
        ls = [(0, 0), (1, 0), (1, 1)]

        seen = []
        for _ in sm.xrange(10):
            batch_aug = aug.augment(image=image, heatmaps=heatmaps,
                                    segmentation_maps=segmaps, keypoints=kps,
                                    bounding_boxes=bbs, polygons=polygons,
                                    line_strings=ls, return_batch=True)

            shift_image = np.sum(batch_aug.images_aug[0][0, :] == 0)
            shift_heatmaps = np.sum(
                np.isclose(batch_aug.heatmaps_aug[0][0, :, 0], 0.0))
            shift_segmaps = np.sum(
                batch_aug.segmentation_maps_aug[0][0, :, 0] == 0)
            shift_kps = batch_aug.keypoints_aug[0][0]
            shift_bbs = batch_aug.bounding_boxes_aug[0][0]
            shift_polygons = batch_aug.polygons_aug[0][0]
            shift_ls = batch_aug.line_strings_aug[0][0]

            assert len({shift_image, shift_heatmaps, shift_segmaps,
                        shift_kps, shift_bbs, shift_polygons,
                        shift_ls}) == 1
            seen.append(shift_image)
        assert len(set(seen)) > 7

    def test_alignment_and_same_outputs_in_deterministic_mode(self):
        # make sure that changes from augment() are aligned
        # and do NOT vary if the augmenter was already in deterministic mode
        aug = iaa.Affine(translate_px={"x": (0, 100)}, order=0, mode="constant",
                         cval=0)
        aug = aug.to_deterministic()

        image = np.zeros((1, 100, 1), dtype=np.uint8) + 255
        heatmaps = np.ones((1, 1, 100, 1), dtype=np.float32)
        segmaps = np.ones((1, 1, 100, 1), dtype=np.int32)
        kps = [(0, 0)]
        bbs = [(0, 0, 1, 1)]
        polygons = [(0, 0), (1, 0), (1, 1)]
        ls = [(0, 0), (1, 0), (1, 1)]

        seen = []
        for _ in sm.xrange(10):
            batch_aug = aug.augment(image=image, heatmaps=heatmaps,
                                    segmentation_maps=segmaps, keypoints=kps,
                                    bounding_boxes=bbs, polygons=polygons,
                                    line_strings=ls,
                                    return_batch=True)

            shift_image = np.sum(batch_aug.images_aug[0][0, :] == 0)
            shift_heatmaps = np.sum(
                np.isclose(batch_aug.heatmaps_aug[0][0, :, 0], 0.0))
            shift_segmaps = np.sum(
                batch_aug.segmentation_maps_aug[0][0, :, 0] == 0)
            shift_kps = batch_aug.keypoints_aug[0][0]
            shift_bbs = batch_aug.bounding_boxes_aug[0][0]
            shift_polygons = batch_aug.polygons_aug[0][0]
            shift_ls = batch_aug.line_strings_aug[0][0]

            assert len({shift_image, shift_heatmaps, shift_segmaps,
                        shift_kps, shift_bbs, shift_polygons,
                        shift_ls}) == 1
            seen.append(shift_image)
        assert len(set(seen)) == 1

    def test_arrays_become_lists_if_augmenter_changes_shapes(self):
        # make sure that arrays (of images, heatmaps, segmaps) get split to
        # lists of arrays if the augmenter changes shapes in non-uniform
        # (between images) ways
        # we augment 100 images here with rotation of either 0deg or 90deg
        # and do not resize back to the original image size afterwards, so
        # shapes change
        aug = iaa.Rot90([0, 1], keep_size=False)

        # base_arr is (100, 1, 2) array, each containing [[0, 1]]
        base_arr = np.tile(np.arange(1*2).reshape((1, 2))[np.newaxis, :, :],
                           (100, 1, 1))
        images = np.copy(base_arr)[:, :, :, np.newaxis].astype(np.uint8)
        heatmaps = (
            np.copy(base_arr)[:, :, :, np.newaxis].astype(np.float32)
            / np.max(base_arr)
        )
        segmaps = np.copy(base_arr)[:, :, :, np.newaxis].astype(np.int32)

        batch_aug = aug.augment(images=images, heatmaps=heatmaps,
                                segmentation_maps=segmaps,
                                return_batch=True)
        assert isinstance(batch_aug.images_aug, list)
        assert isinstance(batch_aug.heatmaps_aug, list)
        assert isinstance(batch_aug.segmentation_maps_aug, list)
        shapes_images = [arr.shape for arr in batch_aug.images_aug]
        shapes_heatmaps = [arr.shape for arr in batch_aug.heatmaps_aug]
        shapes_segmaps = [arr.shape for arr in batch_aug.segmentation_maps_aug]
        assert (
            [shape[0:2] for shape in shapes_images]
            == [shape[0:2] for shape in shapes_heatmaps]
            == [shape[0:2] for shape in shapes_segmaps]
        )
        assert len(set(shapes_images)) == 2

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_two_outputs_none_of_them_images(self):
        aug = iaa.Identity()
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")

        keypoints_aug, polygons_aug = aug.augment(keypoints=[keypoints],
                                                  polygons=[polygons])

        assert_cbaois_equal(keypoints_aug[0], keypoints)
        assert_cbaois_equal(polygons_aug[0], polygons)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_two_outputs_none_of_them_images_inverted(self):
        aug = iaa.Identity()
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")

        polygons_aug, keypoints_aug = aug.augment(polygons=[polygons],
                                                  keypoints=[keypoints])

        assert_cbaois_equal(keypoints_aug[0], keypoints)
        assert_cbaois_equal(polygons_aug[0], polygons)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_two_outputs_inverted_order_heatmaps(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        heatmaps = ia.data.quokka_heatmap((128, 128), extract="square")

        heatmaps_aug, images_aug = aug.augment(heatmaps=[heatmaps],
                                               images=[image])
        assert np.array_equal(images_aug[0], image)
        assert np.allclose(heatmaps_aug[0].arr_0to1, heatmaps.arr_0to1)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_two_outputs_inverted_order_segmaps(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")

        segmaps_aug, images_aug = aug.augment(segmentation_maps=[segmaps],
                                              images=[image])

        assert np.array_equal(images_aug[0], image)
        assert np.array_equal(segmaps_aug[0].arr, segmaps.arr)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_two_outputs_inverted_order_keypoints(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")

        keypoints_aug, images_aug = aug.augment(keypoints=[keypoints],
                                                images=[image])

        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(keypoints_aug[0], keypoints)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_two_outputs_inverted_order_bbs(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        bbs = ia.data.quokka_bounding_boxes((128, 128), extract="square")

        bbs_aug, images_aug = aug.augment(bounding_boxes=[bbs],
                                          images=[image])

        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(bbs_aug[0], bbs)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_two_outputs_inverted_order_polygons(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")

        polygons_aug, images_aug = aug.augment(polygons=[polygons],
                                               images=[image])

        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(polygons_aug[0], polygons)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_two_outputs_inverted_order_line_strings(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        psoi = ia.data.quokka_polygons((128, 128), extract="square")
        lsoi = ia.LineStringsOnImage([
            psoi.polygons[0].to_line_string(closed=False)
        ], shape=psoi.shape)

        lsoi_aug, images_aug = aug.augment(line_strings=[lsoi],
                                           images=[image])

        assert np.array_equal(images_aug[0], image)
        assert_cbaois_equal(lsoi_aug[0], lsoi)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_three_inputs_expected_order(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        heatmaps = ia.data.quokka_heatmap((128, 128), extract="square")
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")

        images_aug, heatmaps_aug, segmaps_aug = aug.augment(
            images=[image],
            heatmaps=[heatmaps],
            segmentation_maps=[segmaps])

        assert np.array_equal(images_aug[0], image)
        assert np.allclose(heatmaps_aug[0].arr_0to1, heatmaps.arr_0to1)
        assert np.array_equal(segmaps_aug[0].arr, segmaps.arr)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_three_inputs_expected_order2(self):
        aug = iaa.Identity()
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")

        segmaps_aug, keypoints_aug, polygons_aug = aug.augment(
            segmentation_maps=[segmaps],
            keypoints=[keypoints],
            polygons=[polygons])

        assert np.array_equal(segmaps_aug[0].arr, segmaps.arr)
        assert_cbaois_equal(keypoints_aug[0], keypoints)
        assert_cbaois_equal(polygons_aug[0], polygons)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_three_inputs_inverted_order(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        heatmaps = ia.data.quokka_heatmap((128, 128), extract="square")
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")

        segmaps_aug, heatmaps_aug, images_aug = aug.augment(
            segmentation_maps=[segmaps],
            heatmaps=[heatmaps],
            images=[image])

        assert np.array_equal(images_aug[0], image)
        assert np.allclose(heatmaps_aug[0].arr_0to1, heatmaps.arr_0to1)
        assert np.array_equal(segmaps_aug[0].arr, segmaps.arr)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_three_inputs_inverted_order2(self):
        aug = iaa.Identity()
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")

        polygons_aug, keypoints_aug, segmaps_aug = aug.augment(
            polygons=[polygons],
            keypoints=[keypoints],
            segmentation_maps=[segmaps])

        assert np.array_equal(segmaps_aug[0].arr, segmaps.arr)
        assert_cbaois_equal(keypoints_aug[0], keypoints)
        assert_cbaois_equal(polygons_aug[0], polygons)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_all_inputs_expected_order(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        heatmaps = ia.data.quokka_heatmap((128, 128), extract="square")
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")
        bbs = ia.data.quokka_bounding_boxes((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")
        lsoi = ia.LineStringsOnImage([
            polygons.polygons[0].to_line_string(closed=False)
        ], shape=polygons.shape)

        images_aug, heatmaps_aug, segmaps_aug, keypoints_aug, bbs_aug, \
            polygons_aug, lsoi_aug = aug.augment(
                images=[image],
                heatmaps=[heatmaps],
                segmentation_maps=[segmaps],
                keypoints=[keypoints],
                bounding_boxes=[bbs],
                polygons=[polygons],
                line_strings=[lsoi])

        assert np.array_equal(images_aug[0], image)
        assert np.allclose(heatmaps_aug[0].arr_0to1, heatmaps.arr_0to1)
        assert np.array_equal(segmaps_aug[0].arr, segmaps.arr)
        assert_cbaois_equal(keypoints_aug[0], keypoints)
        assert_cbaois_equal(bbs_aug[0], bbs)
        assert_cbaois_equal(polygons_aug[0], polygons)
        assert_cbaois_equal(lsoi_aug[0], lsoi)

    @unittest.skipIf(not IS_PY36_OR_HIGHER,
                     "Behaviour is only supported in python 3.6+")
    def test_py_gte_36_all_inputs_inverted_order(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        heatmaps = ia.data.quokka_heatmap((128, 128), extract="square")
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")
        bbs = ia.data.quokka_bounding_boxes((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")
        lsoi = ia.LineStringsOnImage([
            polygons.polygons[0].to_line_string(closed=False)
        ], shape=polygons.shape)

        lsoi_aug, polygons_aug, bbs_aug, keypoints_aug, segmaps_aug, \
            heatmaps_aug, images_aug = aug.augment(
                line_strings=[lsoi],
                polygons=[polygons],
                bounding_boxes=[bbs],
                keypoints=[keypoints],
                segmentation_maps=[segmaps],
                heatmaps=[heatmaps],
                images=[image])

        assert np.array_equal(images_aug[0], image)
        assert np.allclose(heatmaps_aug[0].arr_0to1, heatmaps.arr_0to1)
        assert np.array_equal(segmaps_aug[0].arr, segmaps.arr)
        assert_cbaois_equal(keypoints_aug[0], keypoints)
        assert_cbaois_equal(bbs_aug[0], bbs)
        assert_cbaois_equal(polygons_aug[0], polygons)
        assert_cbaois_equal(lsoi_aug[0], lsoi)

    @unittest.skipIf(IS_PY36_OR_HIGHER,
                     "Test checks behaviour for python <=3.5")
    def test_py_lte_35_calls_without_images_fail(self):
        aug = iaa.Identity()
        keypoints = ia.data.quokka_keypoints((128, 128), extract="square")
        polygons = ia.data.quokka_polygons((128, 128), extract="square")

        got_exception = False
        try:
            _ = aug.augment(keypoints=[keypoints], polygons=[polygons])
        except Exception as exc:
            msg = "Requested two outputs from augment() that were not 'images'"
            assert msg in str(exc)
            got_exception = True
        assert got_exception

    @unittest.skipIf(IS_PY36_OR_HIGHER,
                     "Test checks behaviour for python <=3.5")
    def test_py_lte_35_calls_with_more_than_three_args_fail(self):
        aug = iaa.Identity()
        image = ia.data.quokka((128, 128), extract="square")
        heatmaps = ia.data.quokka_heatmap((128, 128), extract="square")
        segmaps = ia.data.quokka_segmentation_map((128, 128), extract="square")

        got_exception = False
        try:
            _ = aug.augment(images=[image], heatmaps=[heatmaps],
                            segmentation_maps=[segmaps])
        except Exception as exc:
            assert "Requested more than two outputs" in str(exc)
            got_exception = True
        assert got_exception


class TestAugmenter___call__(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_with_two_augmentables(self):
        image = ia.data.quokka(size=(128, 128), extract="square")
        heatmaps = ia.data.quokka_heatmap(size=(128, 128), extract="square")

        images_aug, heatmaps_aug = iaa.Identity()(images=[image],
                                                  heatmaps=[heatmaps])

        assert np.array_equal(images_aug[0], image)
        assert np.allclose(heatmaps_aug[0].arr_0to1, heatmaps.arr_0to1)


class TestAugmenter_pool(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_pool(self):
        augseq = iaa.Identity()

        mock_Pool = mock.MagicMock()
        mock_Pool.return_value = mock_Pool
        mock_Pool.__enter__.return_value = None
        mock_Pool.__exit__.return_value = None
        with mock.patch("imgaug.multicore.Pool", mock_Pool):
            with augseq.pool(processes=2, maxtasksperchild=10, seed=17):
                pass

        assert mock_Pool.call_count == 1
        assert mock_Pool.__enter__.call_count == 1
        assert mock_Pool.__exit__.call_count == 1
        assert mock_Pool.call_args[0][0] == augseq
        assert mock_Pool.call_args[1]["processes"] == 2
        assert mock_Pool.call_args[1]["maxtasksperchild"] == 10
        assert mock_Pool.call_args[1]["seed"] == 17


class TestAugmenter_find_augmenters_by_name(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def seq(self):
        noop1 = iaa.Identity(name="Identity")
        fliplr = iaa.Fliplr(name="Fliplr")
        flipud = iaa.Flipud(name="Flipud")
        noop2 = iaa.Identity(name="Identity2")
        seq2 = iaa.Sequential([flipud, noop2], name="Seq2")
        seq1 = iaa.Sequential([noop1, fliplr, seq2], name="Seq")
        return seq1, seq2

    def test_find_top_element(self):
        seq1, seq2 = self.seq

        augs = seq1.find_augmenters_by_name("Seq")

        assert len(augs) == 1
        assert augs[0] == seq1

    def test_find_nested_element(self):
        seq1, seq2 = self.seq

        augs = seq1.find_augmenters_by_name("Seq2")

        assert len(augs) == 1
        assert augs[0] == seq2

    def test_find_list_of_names(self):
        seq1, seq2 = self.seq

        augs = seq1.find_augmenters_by_names(["Seq", "Seq2"])

        assert len(augs) == 2
        assert augs[0] == seq1
        assert augs[1] == seq2

    def test_find_by_regex(self):
        seq1, seq2 = self.seq

        augs = seq1.find_augmenters_by_name(r"Seq.*", regex=True)

        assert len(augs) == 2
        assert augs[0] == seq1
        assert augs[1] == seq2


class TestAugmenter_find_augmenters(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def seq(self):
        noop1 = iaa.Identity(name="Identity")
        fliplr = iaa.Fliplr(name="Fliplr")
        flipud = iaa.Flipud(name="Flipud")
        noop2 = iaa.Identity(name="Identity2")
        seq2 = iaa.Sequential([flipud, noop2], name="Seq2")
        seq1 = iaa.Sequential([noop1, fliplr, seq2], name="Seq")
        return seq1, seq2

    def test_find_by_list_of_names(self):
        def _func(aug, parents):
            return aug.name in ["Seq", "Seq2"]

        seq1, seq2 = self.seq

        augs = seq1.find_augmenters(_func)

        assert len(augs) == 2
        assert augs[0] == seq1
        assert augs[1] == seq2

    def test_use_parents_arg(self):
        def _func(aug, parents):
            return (
                aug.name in ["Seq", "Seq2"]
                and len(parents) > 0
            )
        seq1, seq2 = self.seq

        augs = seq1.find_augmenters(_func)

        assert len(augs) == 1
        assert augs[0] == seq2

    def test_find_by_list_of_names_flat_false(self):
        def _func(aug, parents):
            return aug.name in ["Seq", "Seq2"]

        seq1, seq2 = self.seq

        augs = seq1.find_augmenters(_func, flat=False)

        assert len(augs) == 2
        assert augs[0] == seq1
        assert augs[1] == [seq2]


class TestAugmenter_remove(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def seq(self):
        noop1 = iaa.Identity(name="Identity")
        fliplr = iaa.Fliplr(name="Fliplr")
        flipud = iaa.Flipud(name="Flipud")
        noop2 = iaa.Identity(name="Identity2")
        seq2 = iaa.Sequential([flipud, noop2], name="Seq2")
        seq1 = iaa.Sequential([noop1, fliplr, seq2], name="Seq")
        return seq1

    def test_remove_by_name(self):
        def _func(aug, parents):
            return aug.name == "Seq2"

        augs = self.seq

        augs = augs.remove_augmenters(_func)

        seqs = augs.find_augmenters_by_name(r"Seq.*", regex=True)
        assert len(seqs) == 1
        assert seqs[0].name == "Seq"

    def test_remove_by_name_and_parents_arg(self):
        def _func(aug, parents):
            return aug.name == "Seq2" and len(parents) == 0

        augs = self.seq

        augs = augs.remove_augmenters(_func)

        seqs = augs.find_augmenters_by_name(r"Seq.*", regex=True)
        assert len(seqs) == 2
        assert seqs[0].name == "Seq"
        assert seqs[1].name == "Seq2"

    def test_remove_all_without_inplace_removal(self):
        def _func(aug, parents):
            return True

        augs = self.seq

        augs = augs.remove_augmenters(_func)

        assert augs is not None
        assert isinstance(augs, iaa.Identity)

    def test_remove_all_with_inplace_removal(self):
        def _func(aug, parents):
            return aug.name == "Seq"

        augs = self.seq
        got_exception = False
        try:
            _ = augs.remove_augmenters(_func, copy=False)
        except Exception as exc:
            got_exception = True
            expected = (
                "Inplace removal of topmost augmenter requested, "
                "which is currently not possible")
            assert expected in str(exc)
        assert got_exception

    def test_remove_all_without_inplace_removal_and_no_identity(self):
        def _func(aug, parents):
            return True

        augs = self.seq

        augs = augs.remove_augmenters(_func, identity_if_topmost=False)

        assert augs is None

    def test_remove_all_without_inplace_removal_and_no_noop(self):
        def _func(aug, parents):
            return True

        augs = self.seq

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            augs = augs.remove_augmenters(_func, noop_if_topmost=False)
        assert len(caught_warnings) == 1
        assert "deprecated" in str(caught_warnings[-1].message)

        assert augs is None


class TestAugmenter_copy_random_state(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        return ia.data.quokka_square(size=(128, 128))

    @property
    def images(self):
        return np.array([self.image] * 64, dtype=np.uint8)

    @property
    def source(self):
        source = iaa.Sequential([
            iaa.Fliplr(0.5, name="hflip"),
            iaa.Dropout(0.05, name="dropout"),
            iaa.Affine(translate_px=(-10, 10), name="translate",
                       seed=3),
            iaa.GaussianBlur(1.0, name="blur", seed=4)
        ], seed=5)
        return source

    @property
    def target(self):
        target = iaa.Sequential([
            iaa.Fliplr(0.5, name="hflip"),
            iaa.Dropout(0.05, name="dropout"),
            iaa.Affine(translate_px=(-10, 10), name="translate")
        ])
        return target

    def test_matching_position(self):
        def _func(aug, parents):
            return aug.name == "blur"

        images = self.images
        source = self.source
        target = self.target
        source.localize_random_state_()

        target_cprs = target.copy_random_state(source, matching="position")

        source_alt = source.remove_augmenters(_func)
        images_aug_source = source_alt.augment_images(images)
        images_aug_target = target_cprs.augment_images(images)

        assert target_cprs.random_state.equals(source_alt.random_state)
        for i in sm.xrange(3):
            assert target_cprs[i].random_state.equals(
                source_alt[i].random_state)
        assert np.array_equal(images_aug_source, images_aug_target)

    def test_matching_position_copy_determinism(self):
        def _func(aug, parents):
            return aug.name == "blur"

        images = self.images
        source = self.source
        target = self.target
        source.localize_random_state_()
        source[0].deterministic = True

        target_cprs = target.copy_random_state(
            source, matching="position", copy_determinism=True)

        source_alt = source.remove_augmenters(_func)
        images_aug_source = source_alt.augment_images(images)
        images_aug_target = target_cprs.augment_images(images)

        assert target_cprs[0].deterministic is True
        assert np.array_equal(images_aug_source, images_aug_target)

    def test_matching_name(self):
        def _func(aug, parents):
            return aug.name == "blur"

        images = self.images
        source = self.source
        target = self.target
        source.localize_random_state_()

        target_cprs = target.copy_random_state(source, matching="name")

        source_alt = source.remove_augmenters(_func)
        images_aug_source = source_alt.augment_images(images)
        images_aug_target = target_cprs.augment_images(images)

        assert np.array_equal(images_aug_source, images_aug_target)

    def test_matching_name_copy_determinism(self):
        def _func(aug, parents):
            return aug.name == "blur"

        images = self.images
        source = self.source
        target = self.target
        source.localize_random_state_()

        source_alt = source.remove_augmenters(_func)
        source_det = source_alt.to_deterministic()

        target_cprs_det = target.copy_random_state(
            source_det, matching="name", copy_determinism=True)

        images_aug_source1 = source_det.augment_images(images)
        images_aug_target1 = target_cprs_det.augment_images(images)
        images_aug_source2 = source_det.augment_images(images)
        images_aug_target2 = target_cprs_det.augment_images(images)
        assert np.array_equal(images_aug_source1, images_aug_source2)
        assert np.array_equal(images_aug_target1, images_aug_target2)
        assert np.array_equal(images_aug_source1, images_aug_target1)
        assert np.array_equal(images_aug_source2, images_aug_target2)

    def test_copy_fails_when_source_rngs_are_not_localized__name(self):
        source = iaa.Fliplr(0.5, name="hflip")
        target = iaa.Fliplr(0.5, name="hflip")
        got_exception = False
        try:
            _ = target.copy_random_state(source, matching="name")
        except Exception as exc:
            got_exception = True
            assert "localize_random_state" in str(exc)
        assert got_exception

    def test_copy_fails_when_source_rngs_are_not_localized__position(self):
        source = iaa.Fliplr(0.5, name="hflip")
        target = iaa.Fliplr(0.5, name="hflip")
        got_exception = False
        try:
            _ = target.copy_random_state(source, matching="position")
        except Exception as exc:
            got_exception = True
            assert "localize_random_state" in str(exc)
        assert got_exception

    def test_copy_fails_when_names_not_match_and_matching_not_tolerant(self):
        source = iaa.Fliplr(0.5, name="hflip-other-name")
        target = iaa.Fliplr(0.5, name="hflip")
        source.localize_random_state_()
        got_exception = False
        try:
            _ = target.copy_random_state(
                source, matching="name", matching_tolerant=False)
        except Exception as exc:
            got_exception = True
            assert "not found among source augmenters" in str(exc)
        assert got_exception

    def test_copy_fails_for_not_tolerant_position_matching(self):
        source = iaa.Sequential([iaa.Fliplr(0.5, name="hflip"),
                                 iaa.Fliplr(0.5, name="hflip2")])
        target = iaa.Sequential([iaa.Fliplr(0.5, name="hflip")])
        source.localize_random_state_()
        got_exception = False
        try:
            _ = target.copy_random_state(
                source, matching="position", matching_tolerant=False)
        except Exception as exc:
            got_exception = True
            assert "different lengths" in str(exc)
        assert got_exception

    def test_copy_fails_for_unknown_matching_method(self):
        source = iaa.Sequential([iaa.Fliplr(0.5, name="hflip"),
                                 iaa.Fliplr(0.5, name="hflip2")])
        target = iaa.Sequential([iaa.Fliplr(0.5, name="hflip")])
        source.localize_random_state_()
        got_exception = False
        try:
            _ = target.copy_random_state(source, matching="test")
        except Exception as exc:
            got_exception = True
            assert "Unknown matching method" in str(exc)
        assert got_exception

    def test_warn_if_multiple_augmenters_with_same_name(self):
        source = iaa.Sequential([iaa.Fliplr(0.5, name="hflip"),
                                 iaa.Fliplr(0.5, name="hflip")])
        target = iaa.Sequential([iaa.Fliplr(0.5, name="hflip")])
        source.localize_random_state_()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            _ = target.copy_random_state(source, matching="name")

        assert len(caught_warnings) == 1
        assert (
            "contains multiple augmenters with the same name"
            in str(caught_warnings[-1].message)
        )


# TODO these tests change the input type from list to array. Might be
#      reasonable to change and test that scenario separetely
class TestAugmenterHooks(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        image = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [0, 1, 1]], dtype=np.uint8)
        return np.atleast_3d(image)

    @property
    def image_lr(self):
        image_lr = np.array([[1, 0, 0],
                             [1, 0, 0],
                             [1, 1, 0]], dtype=np.uint8)
        return np.atleast_3d(image_lr)

    @property
    def image_lrud(self):
        image_lrud = np.array([[1, 1, 0],
                               [1, 0, 0],
                               [1, 0, 0]], dtype=np.uint8)
        return np.atleast_3d(image_lrud)

    def test_preprocessor(self):
        def preprocessor(images, augmenter, parents):
            img = np.copy(images)
            img[0][1, 1, 0] += 1
            return img

        hooks = ia.HooksImages(preprocessor=preprocessor)
        seq = iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)])

        images_aug = seq.augment_images([self.image], hooks=hooks)

        expected = np.copy(self.image_lrud)
        expected[1, 1, 0] = 3
        assert np.array_equal(images_aug[0], expected)

    def test_postprocessor(self):
        def postprocessor(images, augmenter, parents):
            img = np.copy(images)
            img[0][1, 1, 0] += 1
            return img

        hooks = ia.HooksImages(postprocessor=postprocessor)
        seq = iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)])

        images_aug = seq.augment_images([self.image], hooks=hooks)

        expected = np.copy(self.image_lrud)
        expected[1, 1, 0] = 3
        assert np.array_equal(images_aug[0], expected)

    def test_propagator(self):
        def propagator(images, augmenter, parents, default):
            if "Seq" in augmenter.name:
                return False
            else:
                return default

        hooks = ia.HooksImages(propagator=propagator)
        seq = iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)])

        images_aug = seq.augment_images([self.image], hooks=hooks)

        assert np.array_equal(images_aug[0], self.image)

    def test_activator(self):
        def activator(images, augmenter, parents, default):
            if "Flipud" in augmenter.name:
                return False
            else:
                return default

        hooks = ia.HooksImages(activator=activator)
        seq = iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)])

        images_aug = seq.augment_images([self.image], hooks=hooks)

        assert np.array_equal(images_aug[0], self.image_lr)

    def test_activator_keypoints(self):
        def activator(keypoints_on_images, augmenter, parents, default):
            return False

        hooks = ia.HooksKeypoints(activator=activator)
        kps = [ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=0),
               ia.Keypoint(x=2, y=1)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(5, 10, 3))
        aug = iaa.Affine(translate_px=1)

        keypoints_aug = aug.augment_keypoints(kpsoi, hooks=hooks)

        assert keypoints_equal([keypoints_aug], [kpsoi])


class TestAugmenterWithLoadedImages(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_with_cv2(self):
        image = np.arange(10*20).astype(np.uint8).reshape((10, 20, 1))
        image = np.tile(image, (1, 1, 3))
        image[:, :, 0] += 0
        image[:, :, 1] += 1
        image[:, :, 2] += 2
        images = image[np.newaxis, :, :, :]
        image_cp = np.copy(image)
        images_cp = np.copy(images)

        aug_arrs = _InplaceDummyAugmenterImgsArray(1)
        aug_lists = _InplaceDummyAugmenterImgsList(1)

        with TemporaryDirectory() as dirpath:
            imgpath = os.path.join(dirpath, "temp_cv2.png")
            imageio.imwrite(imgpath, image)
            image_reloaded = cv2.imread(imgpath)[:, :, ::-1]
            images_reloaded = image_reloaded[np.newaxis, :, :, :]

            image_aug = aug_lists(image=image_reloaded)
            assert image_aug is not image_reloaded
            assert np.array_equal(image_reloaded, image_cp)
            assert np.array_equal(image_aug, image_cp + 1)

            image_aug = aug_lists.augment_image(image=image_reloaded)
            assert image_aug is not image_reloaded
            assert np.array_equal(image_reloaded, image_cp)
            assert np.array_equal(image_aug, image_cp + 1)

            images_aug = aug_arrs(images=images_reloaded)
            assert images_aug is not images_reloaded
            assert np.array_equal(images_reloaded, images_cp)
            assert np.array_equal(images_aug, images_cp + 1)

            images_aug = aug_arrs.augment_images(images=images_reloaded)
            assert images_aug is not images_reloaded
            assert np.array_equal(images_reloaded, images_cp)
            assert np.array_equal(images_aug, images_cp + 1)

    def test_with_imageio(self):
        image = np.arange(10*20).astype(np.uint8).reshape((10, 20, 1))
        image = np.tile(image, (1, 1, 3))
        image[:, :, 0] += 0
        image[:, :, 1] += 1
        image[:, :, 2] += 2
        images = image[np.newaxis, :, :, :]
        image_cp = np.copy(image)
        images_cp = np.copy(images)

        aug_arrs = _InplaceDummyAugmenterImgsArray(1)
        aug_lists = _InplaceDummyAugmenterImgsList(1)

        with TemporaryDirectory() as dirpath:
            imgpath = os.path.join(dirpath, "temp_imageio.png")
            imageio.imwrite(imgpath, image)
            image_reloaded = imageio.imread(imgpath)
            images_reloaded = image_reloaded[np.newaxis, :, :, :]

            image_aug = aug_lists(image=image_reloaded)
            assert image_aug is not image_reloaded
            assert np.array_equal(image_reloaded, image_cp)
            assert np.array_equal(image_aug, image_cp + 1)

            image_aug = aug_lists.augment_image(image=image_reloaded)
            assert image_aug is not image_reloaded
            assert np.array_equal(image_reloaded, image_cp)
            assert np.array_equal(image_aug, image_cp + 1)

            images_aug = aug_arrs(images=images_reloaded)
            assert images_aug is not images_reloaded
            assert np.array_equal(images_reloaded, images_cp)
            assert np.array_equal(images_aug, images_cp + 1)

            images_aug = aug_arrs.augment_images(images=images_reloaded)
            assert images_aug is not images_reloaded
            assert np.array_equal(images_reloaded, images_cp)
            assert np.array_equal(images_aug, images_cp + 1)

    def test_with_pil(self):
        fnames = ["asarray", "array"]
        for fname in fnames:
            with self.subTest(fname=fname):
                image = np.arange(10*20).astype(np.uint8).reshape((10, 20, 1))
                image = np.tile(image, (1, 1, 3))
                image[:, :, 0] += 0
                image[:, :, 1] += 1
                image[:, :, 2] += 2
                images = image[np.newaxis, :, :, :]
                image_cp = np.copy(image)
                images_cp = np.copy(images)

                aug_arrs = _InplaceDummyAugmenterImgsArray(1)
                aug_lists = _InplaceDummyAugmenterImgsList(1)

                with TemporaryDirectory() as dirpath:
                    imgpath = os.path.join(dirpath,
                                           "temp_pil_%s.png" % (fname,))
                    imageio.imwrite(imgpath, image)
                    image_reloaded = getattr(np, fname)(PIL.Image.open(imgpath))
                    images_reloaded = image_reloaded[np.newaxis, :, :, :]

                    image_aug = aug_lists(image=image_reloaded)
                    assert image_aug is not image_reloaded
                    assert np.array_equal(image_reloaded, image_cp)
                    assert np.array_equal(image_aug, image_cp + 1)

                    image_aug = aug_lists.augment_image(image=image_reloaded)
                    assert image_aug is not image_reloaded
                    assert np.array_equal(image_reloaded, image_cp)
                    assert np.array_equal(image_aug, image_cp + 1)

                    images_aug = aug_arrs(images=images_reloaded)
                    assert images_aug is not images_reloaded
                    assert np.array_equal(images_reloaded, images_cp)
                    assert np.array_equal(images_aug, images_cp + 1)

                    images_aug = aug_arrs.augment_images(images=images_reloaded)
                    assert images_aug is not images_reloaded
                    assert np.array_equal(images_reloaded, images_cp)
                    assert np.array_equal(images_aug, images_cp + 1)


class TestSequential(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        image = np.array([[0, 1, 1],
                          [0, 0, 1],
                          [0, 0, 1]], dtype=np.uint8) * 255
        return np.atleast_3d(image)

    @property
    def images(self):
        return np.array([self.image], dtype=np.uint8)

    @property
    def image_lr(self):
        image_lr = np.array([[1, 1, 0],
                             [1, 0, 0],
                             [1, 0, 0]], dtype=np.uint8) * 255
        return np.atleast_3d(image_lr)

    @property
    def images_lr(self):
        return np.array([self.image_lr], dtype=np.uint8)

    @property
    def image_ud(self):
        image_ud = np.array([[0, 0, 1],
                             [0, 0, 1],
                             [0, 1, 1]], dtype=np.uint8) * 255
        return np.atleast_3d(image_ud)

    @property
    def images_ud(self):
        return np.array([self.image_ud], dtype=np.uint8)

    @property
    def image_lr_ud(self):
        image_lr_ud = np.array([[1, 0, 0],
                                [1, 0, 0],
                                [1, 1, 0]], dtype=np.uint8) * 255
        return np.atleast_3d(image_lr_ud)

    @property
    def images_lr_ud(self):
        return np.array([self.image_lr_ud])

    @property
    def keypoints(self):
        kps = [ia.Keypoint(x=1, y=0),
               ia.Keypoint(x=2, y=0),
               ia.Keypoint(x=2, y=1)]
        return ia.KeypointsOnImage(kps, shape=self.image.shape)

    @property
    def keypoints_aug(self):
        kps = [ia.Keypoint(x=3-1, y=3-0),
               ia.Keypoint(x=3-2, y=3-0),
               ia.Keypoint(x=3-2, y=3-1)]
        return ia.KeypointsOnImage(kps, shape=self.image.shape)

    @property
    def polygons(self):
        polygon = ia.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        return ia.PolygonsOnImage([polygon], shape=self.image.shape)

    @property
    def polygons_aug(self):
        polygon = ia.Polygon([(3-0, 3-0), (3-2, 3-0), (3-2, 3-2), (3-0, 3-2)])
        return ia.PolygonsOnImage([polygon], shape=self.image.shape)

    @property
    def lsoi(self):
        ls = ia.LineString([(0, 0), (2, 0), (2, 2), (0, 2)])
        return ia.LineStringsOnImage([ls], shape=self.image.shape)

    @property
    def lsoi_aug(self):
        ls = ia.LineString([(3-0, 3-0), (3-2, 3-0), (3-2, 3-2), (3-0, 3-2)])
        return ia.LineStringsOnImage([ls], shape=self.image.shape)

    @property
    def bbsoi(self):
        bb = ia.BoundingBox(x1=0, y1=0, x2=2, y2=2)
        return ia.BoundingBoxesOnImage([bb], shape=self.image.shape)

    @property
    def bbsoi_aug(self):
        x1 = 3-0
        x2 = 3-2
        y1 = 3-0
        y2 = 3-2
        bb = ia.BoundingBox(x1=min(x1, x2), y1=min(y1, y2),
                            x2=max(x1, x2), y2=max(y1, y2))
        return ia.BoundingBoxesOnImage([bb], shape=self.image.shape)

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([[0, 0, 1.0],
                                   [0, 0, 1.0],
                                   [0, 1.0, 1.0]])
        return ia.HeatmapsOnImage(heatmaps_arr, shape=self.image.shape)

    @property
    def heatmaps_aug(self):
        heatmaps_arr_expected = np.float32([[1.0, 1.0, 0.0],
                                            [1.0, 0, 0],
                                            [1.0, 0, 0]])
        return ia.HeatmapsOnImage(heatmaps_arr_expected, shape=self.image.shape)

    @property
    def segmaps(self):
        segmaps_arr = np.int32([[0, 0, 1],
                                [0, 0, 1],
                                [0, 1, 1]])
        return ia.SegmentationMapsOnImage(segmaps_arr, shape=self.image.shape)

    @property
    def segmaps_aug(self):
        segmaps_arr_expected = np.int32([[1, 1, 0],
                                         [1, 0, 0],
                                         [1, 0, 0]])
        return ia.SegmentationMapsOnImage(segmaps_arr_expected,
                                          shape=self.image.shape)

    @property
    def seq_two_flips(self):
        return iaa.Sequential([
            iaa.Fliplr(1.0),
            iaa.Flipud(1.0)
        ])

    def test_images__two_flips(self):
        aug = self.seq_two_flips
        observed = aug.augment_images(self.images)
        assert np.array_equal(observed, self.images_lr_ud)

    def test_images__two_flips__deterministic(self):
        aug = self.seq_two_flips
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        assert np.array_equal(observed, self.images_lr_ud)

    def test_images_as_list__two_flips(self):
        aug = self.seq_two_flips

        observed = aug.augment_images([self.image])

        assert array_equal_lists(observed, [self.image_lr_ud])

    def test_images_as_list__two_flips__deterministic(self):
        aug = self.seq_two_flips
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.image])

        assert array_equal_lists(observed, [self.image_lr_ud])

    def test_keypoints__two_flips(self):
        aug = self.seq_two_flips

        observed = aug.augment_keypoints([self.keypoints])

        assert_cbaois_equal(observed, [self.keypoints_aug])

    def test_keypoints__two_flips__deterministic(self):
        aug = self.seq_two_flips
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_keypoints([self.keypoints])

        assert_cbaois_equal(observed, [self.keypoints_aug])

    def test_polygons__two_flips(self):
        aug = self.seq_two_flips

        observed = aug.augment_polygons(self.polygons)

        assert_cbaois_equal(observed, self.polygons_aug)

    def test_polygons__two_flips__deterministic(self):
        aug = self.seq_two_flips
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_polygons(self.polygons)

        assert_cbaois_equal(observed, self.polygons_aug)

    def test_line_strings__two_flips(self):
        aug = self.seq_two_flips

        observed = aug.augment_line_strings(self.lsoi)

        assert_cbaois_equal(observed, self.lsoi_aug)

    def test_line_strings__two_flips__deterministic(self):
        aug = self.seq_two_flips
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_line_strings(self.lsoi)

        assert_cbaois_equal(observed, self.lsoi_aug)

    def test_bounding_boxes__two_flips(self):
        aug = self.seq_two_flips

        observed = aug.augment_bounding_boxes(self.bbsoi)

        assert_cbaois_equal(observed, self.bbsoi_aug)

    def test_bounding_boxes__two_flips__deterministic(self):
        aug = self.seq_two_flips
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_bounding_boxes(self.bbsoi)

        assert_cbaois_equal(observed, self.bbsoi_aug)

    def test_heatmaps__two_flips(self):
        aug = self.seq_two_flips
        heatmaps = self.heatmaps

        observed = aug.augment_heatmaps([heatmaps])[0]

        assert observed.shape == (3, 3, 1)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1.0 - 1e-6 < observed.max_value < 1.0 + 1e-6
        assert np.allclose(observed.get_arr(),
                           self.heatmaps_aug.get_arr())

    def test_segmentation_maps__two_flips(self):
        aug = self.seq_two_flips
        segmaps = self.segmaps

        observed = aug.augment_segmentation_maps([segmaps])[0]

        assert observed.shape == (3, 3, 1)
        assert np.array_equal(observed.get_arr(),
                              self.segmaps_aug.get_arr())

    def test_children_not_provided(self):
        aug = iaa.Sequential()
        image = np.arange(4*4).reshape((4, 4)).astype(np.uint8)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, image)

    def test_children_are_none(self):
        aug = iaa.Sequential(children=None)
        image = np.arange(4*4).reshape((4, 4)).astype(np.uint8)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, image)

    def test_children_is_single_augmenter_without_list(self):
        aug = iaa.Sequential(iaa.Fliplr(1.0))
        image = np.arange(4*4).reshape((4, 4)).astype(np.uint8)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, np.fliplr(image))

    def test_children_is_a_sequential(self):
        aug = iaa.Sequential(iaa.Sequential(iaa.Fliplr(1.0)))
        image = np.arange(4*4).reshape((4, 4)).astype(np.uint8)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, np.fliplr(image))

    def test_children_is_list_of_sequentials(self):
        aug = iaa.Sequential([
            iaa.Sequential(iaa.Flipud(1.0)),
            iaa.Sequential(iaa.Fliplr(1.0))
        ])
        image = np.arange(4*4).reshape((4, 4)).astype(np.uint8)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, np.fliplr(np.flipud(image)))

    def test_randomness__two_flips(self):
        # 50% horizontal flip, 50% vertical flip
        aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ])

        frac_same = self._test_randomness__two_flips__compute_fraction_same(
            aug, 200)
        assert np.isclose(frac_same, 0.25, rtol=0, atol=0.1)

    def test_randomness__two_flips__deterministic(self):
        # 50% horizontal flip, 50% vertical flip
        aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ])
        aug_det = aug.to_deterministic()

        frac_same = self._test_randomness__two_flips__compute_fraction_same(
            aug_det, 200)
        assert (
            np.isclose(frac_same, 0.0, rtol=0, atol=1e-5)
            or np.isclose(frac_same, 1.0, rtol=0, atol=1e-5)
        )

    def _test_randomness__two_flips__compute_fraction_same(self, aug,
                                                           nb_iterations):
        expected = [self.images, self.images_lr, self.images_ud,
                    self.images_lr_ud]

        last_aug = None
        nb_changed_aug = 0

        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(self.images)
            if i == 0:
                last_aug = observed_aug
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                last_aug = observed_aug

            assert np.any([np.array_equal(observed_aug, expected_i)
                           for expected_i in expected])

        # should be the same in roughly 25% of all cases
        frac_changed = nb_changed_aug / nb_iterations
        return 1 - frac_changed

    def test_random_order_true_images(self):
        aug = iaa.Sequential([
            iaa.Affine(translate_px={"x": 1}, mode="constant", cval=0, order=0),
            iaa.Fliplr(1.0)
        ], random_order=True)

        frac_12 = self._test_random_order_images_frac_12(aug, 200)

        assert np.isclose(frac_12, 0.5, 0.075)

    def test_random_order_false_images(self):
        aug = iaa.Sequential([
            iaa.Affine(translate_px={"x": 1}, mode="constant", cval=0, order=0),
            iaa.Fliplr(1.0)
        ], random_order=False)

        frac_12 = self._test_random_order_images_frac_12(aug, 25)

        assert frac_12 >= 1.0 - 1e-4

    def test_random_order_true_deterministic_images(self):
        aug = iaa.Sequential([
            iaa.Affine(translate_px={"x": 1}, mode="constant", cval=0, order=0),
            iaa.Fliplr(1.0)
        ], random_order=True)
        aug = aug.to_deterministic()

        frac_12 = self._test_random_order_images_frac_12(aug, 25)

        assert (frac_12 >= 1.0-1e-4 or frac_12 <= 0.0+1e-4)

    @classmethod
    def _test_random_order_images_frac_12(cls, aug, nb_iterations):
        image = np.uint8([[0, 1],
                           [2, 3]])
        image_12 = np.uint8([[0, 0],
                              [2, 0]])
        image_21 = np.uint8([[0, 1],
                              [0, 3]])

        seen = [False, False]
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_images([image])[0]
            if np.array_equal(observed, image_12):
                seen[0] = True
            elif np.array_equal(observed, image_21):
                seen[1] = True
            else:
                assert False

        frac_12 = seen[0] / np.sum(seen)
        return frac_12

    # TODO add random_order=False
    def test_random_order_heatmaps(self):
        aug = iaa.Sequential([
            iaa.Affine(translate_px={"x": 1}),
            iaa.Fliplr(1.0)
        ], random_order=True)
        heatmaps_arr = np.float32([[0, 0, 1.0],
                                   [0, 0, 1.0],
                                   [0, 1.0, 1.0]])
        heatmaps_arr_expected1 = np.float32([[0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0],
                                             [1.0, 0.0, 0.0]])
        heatmaps_arr_expected2 = np.float32([[0.0, 1.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 1.0, 1.0]])
        seen = [False, False]
        for _ in sm.xrange(100):
            observed = aug.augment_heatmaps([
                ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))])[0]
            if np.allclose(observed.get_arr(), heatmaps_arr_expected1):
                seen[0] = True
            elif np.allclose(observed.get_arr(), heatmaps_arr_expected2):
                seen[1] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    # TODO add random_order=False
    def test_random_order_segmentation_maps(self):
        aug = iaa.Sequential([
            iaa.Affine(translate_px={"x": 1}),
            iaa.Fliplr(1.0)
        ], random_order=True)
        segmaps_arr = np.int32([[0, 0, 1],
                                [0, 0, 1],
                                [0, 1, 1]])
        segmaps_arr_expected1 = np.int32([[0, 0, 0],
                                          [0, 0, 0],
                                          [1, 0, 0]])
        segmaps_arr_expected2 = np.int32([[0, 1, 0],
                                          [0, 1, 0],
                                          [0, 1, 1]])
        seen = [False, False]
        for _ in sm.xrange(100):
            observed = aug.augment_segmentation_maps([
                SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))])[0]
            if np.array_equal(observed.get_arr(), segmaps_arr_expected1):
                seen[0] = True
            elif np.array_equal(observed.get_arr(), segmaps_arr_expected2):
                seen[1] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    # TODO add random_order=False
    def test_random_order_keypoints(self):
        KP = ia.Keypoint
        kps = [KP(0, 0), KP(2, 0), KP(2, 2)]
        kps_12 = [KP((0+1)*2, 0), KP((2+1)*2, 0), KP((2+1)*2, 2)]
        kps_21 = [KP((0*2)+1, 0), KP((2*2)+1, 0), KP((2*2)+1, 2)]

        kpsoi = ia.KeypointsOnImage(kps, shape=(3, 3))
        kpsoi_12 = ia.KeypointsOnImage(kps_12, shape=(3, 3))
        kpsoi_21 = ia.KeypointsOnImage(kps_21, shape=(3, 3))

        def func1(keypoints_on_images, random_state, parents, hooks):
            for kpsoi in keypoints_on_images:
                for kp in kpsoi.keypoints:
                    kp.x += 1
            return keypoints_on_images

        def func2(keypoints_on_images, random_state, parents, hooks):
            for kpsoi in keypoints_on_images:
                for kp in kpsoi.keypoints:
                    kp.x *= 2
            return keypoints_on_images

        aug_1 = iaa.Lambda(func_keypoints=func1)
        aug_2 = iaa.Lambda(func_keypoints=func2)
        seq = iaa.Sequential([aug_1, aug_2], random_order=True)

        seen = [False, False]
        for _ in sm.xrange(100):
            observed = seq.augment_keypoints(kpsoi)
            if np.allclose(observed.to_xy_array(), kpsoi_12.to_xy_array()):
                seen[0] = True
            elif np.allclose(observed.to_xy_array(), kpsoi_21.to_xy_array()):
                seen[1] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    # TODO add random_order=False
    def test_random_order_polygons(self):
        cba = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        cba_12 = ia.Polygon([(0, 0), (1, 0), ((1+1)*2, 1)])
        cba_21 = ia.Polygon([(0, 0), (1, 0), ((1*2)+1, 1)])

        cbaoi = ia.PolygonsOnImage([cba], shape=(3, 3))

        def func1(polygons_on_images, random_state, parents, hooks):
            for cbaoi_ in polygons_on_images:
                for cba_ in cbaoi_.items:
                    cba_.exterior[-1, 0] += 1
            return polygons_on_images

        def func2(polygons_on_images, random_state, parents, hooks):
            for cbaoi_ in polygons_on_images:
                for cba_ in cbaoi_.items:
                    cba_.exterior[-1, 0] *= 2
            return polygons_on_images

        aug_1 = iaa.Lambda(func_polygons=func1)
        aug_2 = iaa.Lambda(func_polygons=func2)
        seq = iaa.Sequential([aug_1, aug_2], random_order=True)

        seen = [False, False]
        for _ in sm.xrange(100):
            observed = seq.augment_polygons(cbaoi)
            if np.allclose(observed.items[0].coords, cba_12.coords):
                seen[0] = True
            elif np.allclose(observed.items[0].coords, cba_21.coords):
                seen[1] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    # TODO add random_order=False
    def test_random_order_line_strings(self):
        cba = ia.LineString([(0, 0), (1, 0), (1, 1)])
        cba_12 = ia.LineString([(0, 0), (1, 0), ((1+1)*2, 1)])
        cba_21 = ia.LineString([(0, 0), (1, 0), ((1*2)+1, 1)])

        cbaoi = ia.LineStringsOnImage([cba], shape=(3, 3))

        def func1(line_strings_on_images, random_state, parents, hooks):
            for cbaoi_ in line_strings_on_images:
                for cba_ in cbaoi_.items:
                    cba_.coords[-1, 0] += 1
            return line_strings_on_images

        def func2(line_strings_on_images, random_state, parents, hooks):
            for cbaoi_ in line_strings_on_images:
                for cba_ in cbaoi_.items:
                    cba_.coords[-1, 0] *= 2
            return line_strings_on_images

        aug_1 = iaa.Lambda(func_line_strings=func1)
        aug_2 = iaa.Lambda(func_line_strings=func2)
        seq = iaa.Sequential([aug_1, aug_2], random_order=True)

        seen = [False, False]
        for _ in sm.xrange(100):
            observed = seq.augment_line_strings(cbaoi)
            if np.allclose(observed.items[0].coords, cba_12.coords):
                seen[0] = True
            elif np.allclose(observed.items[0].coords, cba_21.coords):
                seen[1] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    # TODO add random_order=False
    def test_random_order_bounding_boxes(self):
        bbs = [ia.BoundingBox(x1=1, y1=2, x2=30, y2=40)]
        bbs_12 = [ia.BoundingBox(x1=(1+1)*2, y1=2, x2=30, y2=40)]
        bbs_21 = [ia.BoundingBox(x1=(1*2)+1, y1=2, x2=30, y2=40)]

        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(3, 3))
        bbsoi_12 = ia.BoundingBoxesOnImage(bbs_12, shape=(3, 3))
        bbsoi_21 = ia.BoundingBoxesOnImage(bbs_21, shape=(3, 3))

        def func1(bounding_boxes_on_images, random_state, parents, hooks):
            for bbsoi in bounding_boxes_on_images:
                for bb in bbsoi.bounding_boxes:
                    bb.x1 += 1
            return bounding_boxes_on_images

        def func2(bounding_boxes_on_images, random_state, parents, hooks):
            for bbsoi in bounding_boxes_on_images:
                for bb in bbsoi.bounding_boxes:
                    bb.x1 *= 2
            return bounding_boxes_on_images

        aug_1 = iaa.Lambda(func_bounding_boxes=func1)
        aug_2 = iaa.Lambda(func_bounding_boxes=func2)
        seq = iaa.Sequential([aug_1, aug_2], random_order=True)

        seen = [False, False]
        for _ in sm.xrange(100):
            observed = seq.augment_bounding_boxes(bbsoi)
            if np.allclose(observed.to_xyxy_array(),
                           bbsoi_12.to_xyxy_array()):
                seen[0] = True
            elif np.allclose(observed.to_xyxy_array(),
                             bbsoi_21.to_xyxy_array()):
                seen[1] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            for random_order in [False, True]:
                with self.subTest(shape=shape):
                    image = np.zeros(shape, dtype=np.uint8)
                    aug = iaa.Sequential([iaa.Identity()],
                                         random_order=random_order)

                    image_aug = aug(image=image)

                    assert image_aug.dtype.name == "uint8"
                    assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            for random_order in [False, True]:
                with self.subTest(shape=shape):
                    image = np.zeros(shape, dtype=np.uint8)
                    aug = iaa.Sequential([iaa.Identity()],
                                         random_order=random_order)

                    image_aug = aug(image=image)

                    assert np.all(image_aug == 0)
                    assert image_aug.dtype.name == "uint8"
                    assert image_aug.shape == shape

    def test_add_to_empty_sequential(self):
        aug = iaa.Sequential()
        aug.add(iaa.Fliplr(1.0))
        image = np.arange(4*4).reshape((4, 4)).astype(np.uint8)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, np.fliplr(image))

    def test_add_to_sequential_with_child(self):
        aug = iaa.Sequential(iaa.Fliplr(1.0))
        aug.add(iaa.Flipud(1.0))
        image = np.arange(4*4).reshape((4, 4)).astype(np.uint8)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, np.fliplr(np.flipud(image)))

    def test_get_parameters(self):
        aug1 = iaa.Sequential(iaa.Fliplr(1.0), random_order=False)
        aug2 = iaa.Sequential(iaa.Fliplr(1.0), random_order=True)
        assert aug1.get_parameters() == [False]
        assert aug2.get_parameters() == [True]

    def test_get_children_lists(self):
        flip = iaa.Fliplr(1.0)
        aug = iaa.Sequential(flip)
        assert aug.get_children_lists() == [aug]

    def test_to_deterministic(self):
        child = iaa.Identity()
        aug = iaa.Sequential([child])

        aug_det = aug.to_deterministic()

        assert aug_det.random_state is not aug.random_state
        assert aug_det.deterministic
        assert aug_det[0].deterministic

    def test___str___and___repr__(self):
        flip = iaa.Fliplr(1.0)
        aug = iaa.Sequential(flip, random_order=True)
        expected = (
            "Sequential("
            "name=%s, random_order=%s, children=[%s], deterministic=%s"
            ")" % (aug.name, "True", str(flip), "False")
        )
        assert aug.__str__() == aug.__repr__() == expected

    def test_other_dtypes_noop__bool(self):
        for random_order in [False, True]:
            aug = iaa.Sequential([
                iaa.Identity(),
                iaa.Identity()
            ], random_order=random_order)

            image = np.zeros((3, 3), dtype=bool)
            image[0, 0] = True
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.name == "bool"
            assert np.all(image_aug == image)

    def test_other_dtypes__noop__uint_int(self):
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]

        for dtype, random_order in itertools.product(dtypes, [False, True]):
            with self.subTest(dtype=dtype, random_order=random_order):
                aug = iaa.Sequential([
                    iaa.Identity(),
                    iaa.Identity()
                ], random_order=random_order)

                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, image)

    def test_other_dtypes_noop__float(self):
        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128
        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]

        for random_order in [False, True]:
            for dtype, value in zip(dtypes, values):
                with self.subTest(dtype=dtype, random_order=random_order):
                    aug = iaa.Sequential([
                        iaa.Identity(),
                        iaa.Identity()
                    ], random_order=random_order)

                    image = np.zeros((3, 3), dtype=dtype)
                    image[0, 0] = value
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.name == dtype
                    assert np.all(image_aug == image)

    def test_other_dtypes_flips__bool(self):
        for random_order in [False, True]:
            # note that we use 100% probabilities with square images here,
            # so random_order does not influence the output
            aug = iaa.Sequential([
                iaa.Fliplr(1.0),
                iaa.Flipud(1.0)
            ], random_order=random_order)

            image = np.zeros((3, 3), dtype=bool)
            image[0, 0] = True
            expected = np.zeros((3, 3), dtype=bool)
            expected[2, 2] = True
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.name == "bool"
            assert np.all(image_aug == expected)

    def test_other_dtypes__flips__uint_int(self):
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]

        for dtype, random_order in itertools.product(dtypes, [False, True]):
            with self.subTest(dtype=dtype, random_order=random_order):
                # note that we use 100% probabilities with square images here,
                # so random_order does not influence the output
                aug = iaa.Sequential([
                    iaa.Fliplr(1.0),
                    iaa.Flipud(1.0)
                ], random_order=random_order)

                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value
                expected = np.zeros((3, 3), dtype=dtype)
                expected[2, 2] = value
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, expected)

    def test_other_dtypes_flips__float(self):
        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128
        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]

        for random_order in [False, True]:
            for dtype, value in zip(dtypes, values):
                with self.subTest(dtype=dtype, random_order=random_order):
                    # note that we use 100% probabilities with square images
                    # here, so random_order does not influence the output
                    aug = iaa.Sequential([
                        iaa.Fliplr(1.0),
                        iaa.Flipud(1.0)
                    ], random_order=random_order)

                    image = np.zeros((3, 3), dtype=dtype)
                    image[0, 0] = value
                    expected = np.zeros((3, 3), dtype=dtype)
                    expected[2, 2] = value
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.name == dtype
                    assert np.all(image_aug == expected)

    def test_pickleable(self):
        aug = iaa.Sequential(
            [iaa.Add(1, seed=1),
             iaa.Multiply(3, seed=2)],
            random_order=True,
            seed=3)
        runtest_pickleable_uint8_img(aug, iterations=5)


class TestSomeOf(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_children_are_empty_list(self):
        zeros = np.zeros((3, 3, 1), dtype=np.uint8)
        aug = iaa.SomeOf(n=0, children=[])
        observed = aug.augment_image(zeros)
        assert np.array_equal(observed, zeros)

    def test_children_are_not_provided(self):
        zeros = np.zeros((3, 3, 1), dtype=np.uint8)
        aug = iaa.SomeOf(n=0)
        observed = aug.augment_image(zeros)
        assert np.array_equal(observed, zeros)

    def test_several_children_and_various_fixed_n(self):
        zeros = np.zeros((3, 3, 1), dtype=np.uint8)
        children = [iaa.Add(1), iaa.Add(2), iaa.Add(3)]

        ns = [0, 1, 2, 3, 4, None, (2, None), (2, 2),
              iap.Deterministic(3)]
        expecteds = [[0],  # 0
                     [9*1, 9*2, 9*3],  # 1
                     [9*1+9*2, 9*1+9*3, 9*2+9*3],  # 2
                     [9*1+9*2+9*3],  # 3
                     [9*1+9*2+9*3],  # 4
                     [9*1+9*2+9*3],  # None
                     [9*1+9*2, 9*1+9*3, 9*2+9*3, 9*1+9*2+9*3],  # (2, None)
                     [9*1+9*2, 9*1+9*3, 9*2+9*3],  # (2, 2)
                     [9*1+9*2+9*3]]  # Deterministic(3)

        for n, expected in zip(ns, expecteds):
            with self.subTest(n=n):
                aug = iaa.SomeOf(n=n, children=children)
                observed = aug.augment_image(zeros)
                assert np.sum(observed) in expected

    def test_several_children_and_n_as_tuple(self):
        zeros = np.zeros((1, 1, 1), dtype=np.uint8)
        augs = [iaa.Add(2**0), iaa.Add(2**1), iaa.Add(2**2)]
        aug = iaa.SomeOf(n=(0, 3), children=augs)

        nb_iterations = 1000
        nb_observed = [0, 0, 0, 0]
        for i in sm.xrange(nb_iterations):
            observed = aug.augment_image(zeros)
            s = observed[0, 0, 0]
            if s == 0:
                nb_observed[0] += 1
            else:
                if s & 2**0 > 0:
                    nb_observed[1] += 1
                if s & 2**1 > 0:
                    nb_observed[2] += 1
                if s & 2**2 > 0:
                    nb_observed[3] += 1
        p_observed = [n/nb_iterations for n in nb_observed]
        assert np.isclose(p_observed[0], 0.25, rtol=0, atol=0.1)
        assert np.isclose(p_observed[1], 0.5, rtol=0, atol=0.1)
        assert np.isclose(p_observed[2], 0.5, rtol=0, atol=0.1)
        assert np.isclose(p_observed[3], 0.5, rtol=0, atol=0.1)

    def test_several_children_and_various_fixed_n__heatmaps(self):
        augs = [iaa.Affine(translate_px={"x": 1}),
                iaa.Affine(translate_px={"x": 1}),
                iaa.Affine(translate_px={"x": 1})]

        heatmaps_arr = np.float32([[1.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]])
        heatmaps_arr0 = np.float32([[1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]])
        heatmaps_arr1 = np.float32([[0.0, 1.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 0.0]])
        heatmaps_arr2 = np.float32([[0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0]])
        heatmaps_arr3 = np.float32([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0]])

        ns = [0, 1, 2, 3, None]
        expecteds = [[heatmaps_arr0],
                     [heatmaps_arr1],
                     [heatmaps_arr2],
                     [heatmaps_arr3],
                     [heatmaps_arr3]]

        for n, expected in zip(ns, expecteds):
            with self.subTest(n=n):
                heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))
                aug = iaa.SomeOf(n=n, children=augs)
                observed = aug.augment_heatmaps(heatmaps)
                assert observed.shape == (3, 3, 3)
                assert np.isclose(observed.min_value, 0.0)
                assert np.isclose(observed.max_value, 1.0)
                matches = [
                    np.allclose(observed.get_arr(), expected_i)
                    for expected_i in expected]
                assert np.any(matches)

    def test_several_children_and_various_fixed_n__segmaps(self):
        augs = [iaa.Affine(translate_px={"x": 1}),
                iaa.Affine(translate_px={"x": 1}),
                iaa.Affine(translate_px={"x": 1})]
        segmaps_arr = np.int32([[1, 0, 0],
                                [1, 0, 0],
                                [1, 0, 0]])
        segmaps_arr0 = np.int32([[1, 0, 0],
                                 [1, 0, 0],
                                 [1, 0, 0]])
        segmaps_arr1 = np.int32([[0, 1, 0],
                                 [0, 1, 0],
                                 [0, 1, 0]])
        segmaps_arr2 = np.int32([[0, 0, 1],
                                 [0, 0, 1],
                                 [0, 0, 1]])
        segmaps_arr3 = np.int32([[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]])

        ns = [0, 1, 2, 3, None]
        expecteds = [[segmaps_arr0],
                     [segmaps_arr1],
                     [segmaps_arr2],
                     [segmaps_arr3],
                     [segmaps_arr3]]

        for n, expected in zip(ns, expecteds):
            with self.subTest(n=n):
                segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))
                aug = iaa.SomeOf(n=n, children=augs)
                observed = aug.augment_segmentation_maps(segmaps)
                assert observed.shape == (3, 3, 3)
                matches = [
                    np.array_equal(observed.get_arr(), expected_i)
                    for expected_i in expected]
                assert np.any(matches)

    def _test_several_children_and_various_fixed_n__cbaois(
            self, cbaoi, augf_name):
        augs = [iaa.Affine(translate_px={"x": 1}),
                iaa.Affine(translate_px={"y": 1})]

        cbaoi_x = cbaoi.shift(x=1)
        cbaoi_y = cbaoi.shift(y=1)
        cbaoi_xy = cbaoi.shift(x=1, y=1)

        ns = [0, 1, 2, None]
        expecteds = [[cbaoi],
                     [cbaoi_x, cbaoi_y],
                     [cbaoi_xy],
                     [cbaoi_xy]]

        for n, expected in zip(ns, expecteds):
            with self.subTest(n=n):
                aug = iaa.SomeOf(n=n, children=augs)
                cbaoi_aug = getattr(aug, augf_name)(cbaoi)
                cba = cbaoi_aug.items[0]
                assert len(cbaoi_aug.items) == len(cbaoi.items)
                assert cbaoi_aug.shape == (5, 6, 3)
                if hasattr(cba, "is_valid"):
                    assert cba.is_valid
                matches = [
                    cba.coords_almost_equals(cbaoi_i.items[0])
                    for cbaoi_i in expected
                ]
                assert np.any(matches)

    def test_several_children_and_various_fixed_n__keypoints(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(5, 6, 3))
        self._test_several_children_and_various_fixed_n__cbaois(
            kpsoi, "augment_keypoints")

    def test_several_children_and_various_fixed_n__polygons(self):
        ps = [ia.Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
        psoi = ia.PolygonsOnImage(ps, shape=(5, 6, 3))
        self._test_several_children_and_various_fixed_n__cbaois(
            psoi, "augment_polygons")

    def test_several_children_and_various_fixed_n__line_strings(self):
        ls = [ia.LineString([(0, 0), (3, 0), (3, 3), (0, 3)])]
        lsoi = ia.LineStringsOnImage(ls, shape=(5, 6, 3))
        self._test_several_children_and_various_fixed_n__cbaois(
            lsoi, "augment_line_strings")

    def test_several_children_and_various_fixed_n__bounding_boxes(self):
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=3, y2=3)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(5, 6, 3))
        self._test_several_children_and_various_fixed_n__cbaois(
            bbsoi, "augment_bounding_boxes")

    @classmethod
    def _test_empty_cbaoi(cls, cbaoi, augf_name):
        augs = [iaa.Affine(translate_px={"x": 1}),
                iaa.Affine(translate_px={"y": 1})]
        aug = iaa.SomeOf(n=2, children=augs)

        cbaoi_aug = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(cbaoi_aug, cbaoi)

    def test_empty_keypoints_on_image_instance(self):
        kpsoi = ia.KeypointsOnImage([], shape=(5, 6, 3))
        self._test_empty_cbaoi(kpsoi, "augment_keypoints")

    def test_empty_polygons_on_image_instance(self):
        psoi = ia.PolygonsOnImage([], shape=(5, 6, 3))
        self._test_empty_cbaoi(psoi, "augment_polygons")

    def test_empty_line_strings_on_image_instance(self):
        lsoi = ia.LineStringsOnImage([], shape=(5, 6, 3))
        self._test_empty_cbaoi(lsoi, "augment_line_strings")

    def test_empty_bounding_boxes_on_image_instance(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(5, 6, 3))
        self._test_empty_cbaoi(bbsoi, "augment_bounding_boxes")

    def test_random_order_false__images(self):
        augs = [iaa.Multiply(2.0), iaa.Add(100)]
        aug = iaa.SomeOf(n=2, children=augs, random_order=False)
        p_observed = self._test_random_order(aug, 10)
        assert np.isclose(p_observed[0], 1.0, rtol=0, atol=1e-8)
        assert np.isclose(p_observed[1], 0.0, rtol=0, atol=1e-8)

    def test_random_order_true__images(self):
        augs = [iaa.Multiply(2.0), iaa.Add(100)]
        aug = iaa.SomeOf(n=2, children=augs, random_order=True)
        p_observed = self._test_random_order(aug, 300)
        assert np.isclose(p_observed[0], 0.5, rtol=0, atol=0.15)
        assert np.isclose(p_observed[1], 0.5, rtol=0, atol=0.15)

    @classmethod
    def _test_random_order(cls, aug, nb_iterations):
        zeros = np.ones((1, 1, 1), dtype=np.uint8)

        nb_observed = [0, 0]
        for i in sm.xrange(nb_iterations):
            observed = aug.augment_image(zeros)
            s = np.sum(observed)
            if s == (1*2)+100:
                nb_observed[0] += 1
            elif s == (1+100)*2:
                nb_observed[1] += 1
            else:
                raise Exception("Unexpected sum: %.8f (@2)" % (s,))

        p_observed = [n/nb_iterations for n in nb_observed]
        return p_observed

    @classmethod
    def _test_images_and_cbaoi_aligned(cls, cbaoi, augf_name):
        img = np.zeros((3, 3), dtype=np.uint8)
        img_x = np.copy(img)
        img_y = np.copy(img)
        img_xy = np.copy(img)
        img[1, 1] = 255
        img_x[1, 2] = 255
        img_y[2, 1] = 255
        img_xy[2, 2] = 255

        augs = [
            iaa.Affine(translate_px={"x": 1}, order=0),
            iaa.Affine(translate_px={"y": 1}, order=0)
        ]
        cbaoi_x = cbaoi.shift(x=1)
        cbaoi_y = cbaoi.shift(y=1)
        cbaoi_xy = cbaoi.shift(x=1, y=1)

        aug = iaa.SomeOf((0, 2), children=augs)
        seen = [False, False, False, False]
        for _ in sm.xrange(100):
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(img)
            cbaoi_aug = getattr(aug_det, augf_name)(cbaoi)
            if np.array_equal(img_aug, img):
                assert_cbaois_equal(cbaoi_aug, cbaoi)
                seen[0] = True
            elif np.array_equal(img_aug, img_x):
                assert_cbaois_equal(cbaoi_aug, cbaoi_x)
                seen[1] = True
            elif np.array_equal(img_aug, img_y):
                assert_cbaois_equal(cbaoi_aug, cbaoi_y)
                seen[2] = True
            elif np.array_equal(img_aug, img_xy):
                assert_cbaois_equal(cbaoi_aug, cbaoi_xy)
                seen[3] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    def test_images_and_keypoints_aligned(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(5, 6, 3))
        self._test_images_and_cbaoi_aligned(kpsoi, "augment_keypoints")

    def test_images_and_polygons_aligned(self):
        ps = [ia.Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
        psoi = ia.PolygonsOnImage(ps, shape=(5, 6, 3))
        self._test_images_and_cbaoi_aligned(psoi, "augment_polygons")

    def test_images_and_line_strings_aligned(self):
        ls = [ia.LineString([(0, 0), (3, 0), (3, 3), (0, 3)])]
        lsoi = ia.LineStringsOnImage(ls, shape=(5, 6, 3))
        self._test_images_and_cbaoi_aligned(lsoi, "augment_line_strings")

    def test_images_and_bounding_boxes_aligned(self):
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=3, y2=3)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(5, 6, 3))
        self._test_images_and_cbaoi_aligned(bbsoi, "augment_bounding_boxes")

    def test_invalid_argument_as_children(self):
        got_exception = False
        try:
            _ = iaa.SomeOf(1, children=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_invalid_datatype_as_n(self):
        got_exception = False
        try:
            _ = iaa.SomeOf(False, children=iaa.Fliplr(1.0))
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_invalid_tuple_as_n(self):
        got_exception = False
        try:
            _ = iaa.SomeOf((2, "test"), children=iaa.Fliplr(1.0))
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_invalid_none_none_tuple_as_n(self):
        got_exception = False
        try:
            _ = iaa.SomeOf((None, None), children=iaa.Fliplr(1.0))
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_with_children_that_change_shapes_keep_size_false(self):
        # test for https://github.com/aleju/imgaug/issues/143
        # (shapes change in child augmenters, leading to problems if input
        # arrays are assumed to stay input arrays)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        aug = iaa.SomeOf(1, [
            iaa.Crop((2, 0, 2, 0), keep_size=False),
            iaa.Crop((1, 0, 1, 0), keep_size=False)
        ])
        expected_shapes = [(4, 8, 3), (6, 8, 3)]

        for _ in sm.xrange(10):
            observed = aug.augment_images(np.uint8([image] * 4))
            assert isinstance(observed, list)
            assert np.all([img.shape in expected_shapes for img in observed])

            observed = aug.augment_images([image] * 4)
            assert isinstance(observed, list)
            assert np.all([img.shape in expected_shapes for img in observed])

            observed = aug.augment_images(np.uint8([image]))
            assert isinstance(observed, list)
            assert np.all([img.shape in expected_shapes for img in observed])

            observed = aug.augment_images([image])
            assert isinstance(observed, list)
            assert np.all([img.shape in expected_shapes for img in observed])

            observed = aug.augment_image(image)
            assert ia.is_np_array(image)
            assert observed.shape in expected_shapes

    def test_with_children_that_change_shapes_keep_size_true(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        aug = iaa.SomeOf(1, [
            iaa.Crop((2, 0, 2, 0), keep_size=True),
            iaa.Crop((1, 0, 1, 0), keep_size=True)
        ])
        expected_shapes = [(8, 8, 3)]

        for _ in sm.xrange(10):
            observed = aug.augment_images(np.uint8([image] * 4))
            assert ia.is_np_array(observed)
            assert np.all([img.shape in expected_shapes for img in observed])

            observed = aug.augment_images([image] * 4)
            assert isinstance(observed, list)
            assert np.all([img.shape in expected_shapes for img in observed])

            observed = aug.augment_images(np.uint8([image]))
            assert ia.is_np_array(observed)
            assert np.all([img.shape in expected_shapes for img in observed])

            observed = aug.augment_images([image])
            assert isinstance(observed, list)
            assert np.all([img.shape in expected_shapes for img in observed])

            observed = aug.augment_image(image)
            assert ia.is_np_array(observed)
            assert observed.shape in expected_shapes

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            for random_order in [False, True]:
                with self.subTest(shape=shape):
                    image = np.zeros(shape, dtype=np.uint8)
                    aug = iaa.SomeOf(
                        1, [iaa.Identity()], random_order=random_order)

                    image_aug = aug(image=image)

                    assert image_aug.dtype.name == "uint8"
                    assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            for random_order in [False, True]:
                with self.subTest(shape=shape):
                    image = np.zeros(shape, dtype=np.uint8)
                    aug = iaa.SomeOf(
                        1, [iaa.Identity()], random_order=random_order)

                    image_aug = aug(image=image)

                    assert np.all(image_aug == 0)
                    assert image_aug.dtype.name == "uint8"
                    assert image_aug.shape == shape

    def test_other_dtypes_via_noop__bool(self):
        for random_order in [False, True]:
            with self.subTest(random_order=random_order):
                aug = iaa.SomeOf(2, [
                    iaa.Identity(),
                    iaa.Identity(),
                    iaa.Identity()
                ], random_order=random_order)

                image = np.zeros((3, 3), dtype=bool)
                image[0, 0] = True
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.name == image.dtype.name
                assert np.all(image_aug == image)

    def test_other_dtypes_via_noop__uint_int(self):
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]
        random_orders = [False, True]

        for dtype, random_order in itertools.product(dtypes, random_orders):
            with self.subTest(dtype=dtype, random_order=random_order):
                aug = iaa.SomeOf(2, [
                    iaa.Identity(),
                    iaa.Identity(),
                    iaa.Identity()
                ], random_order=random_order)

                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, image)

    def test_other_dtypes_via_noop__float(self):
        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128
        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]
        random_orders = [False, True]

        for random_order in random_orders:
            for dtype, value in zip(dtypes, values):
                with self.subTest(dtype=dtype, random_order=random_order):
                    aug = iaa.SomeOf(2, [
                        iaa.Identity(),
                        iaa.Identity(),
                        iaa.Identity()
                    ], random_order=random_order)
                    image = np.zeros((3, 3), dtype=dtype)
                    image[0, 0] = value
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.name == dtype
                    assert np.all(image_aug == image)

    def test_other_dtypes_via_flip__bool(self):
        for random_order in [False, True]:
            with self.subTest(random_order=random_order):
                aug = iaa.SomeOf(2, [
                    iaa.Fliplr(1.0),
                    iaa.Flipud(1.0),
                    iaa.Identity()
                ], random_order=random_order)

                image = np.zeros((3, 3), dtype=bool)
                image[0, 0] = True
                expected = [np.zeros((3, 3), dtype=bool)
                            for _ in sm.xrange(3)]
                expected[0][0, 2] = True
                expected[1][2, 0] = True
                expected[2][2, 2] = True

                for _ in sm.xrange(10):
                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == image.dtype.name
                    assert any([np.all(image_aug == expected_i)
                                for expected_i in expected])

    def test_other_dtypes_via_flip__uint_int(self):
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]
        random_orders = [False, True]

        for dtype, random_order in itertools.product(dtypes, random_orders):
            with self.subTest(dtype=dtype, random_order=random_order):
                aug = iaa.SomeOf(2, [
                    iaa.Fliplr(1.0),
                    iaa.Flipud(1.0),
                    iaa.Identity()
                ], random_order=random_order)

                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value
                expected = [np.zeros((3, 3), dtype=dtype)
                            for _ in sm.xrange(3)]
                expected[0][0, 2] = value
                expected[1][2, 0] = value
                expected[2][2, 2] = value

                for _ in sm.xrange(10):
                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert any([np.all(image_aug == expected_i)
                                for expected_i in expected])

    def test_other_dtypes_via_flip__float(self):
        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128
        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]
        random_orders = [False, True]

        for random_order in random_orders:
            for dtype, value in zip(dtypes, values):
                with self.subTest(dtype=dtype, random_order=random_order):
                    aug = iaa.SomeOf(2, [
                        iaa.Fliplr(1.0),
                        iaa.Flipud(1.0),
                        iaa.Identity()
                    ], random_order=random_order)

                    image = np.zeros((3, 3), dtype=dtype)
                    image[0, 0] = value
                    expected = [np.zeros((3, 3), dtype=dtype)
                                for _ in sm.xrange(3)]
                    expected[0][0, 2] = value
                    expected[1][2, 0] = value
                    expected[2][2, 2] = value

                    for _ in sm.xrange(10):
                        image_aug = aug.augment_image(image)

                        assert image_aug.dtype.name == dtype
                        assert any([np.all(image_aug == expected_i)
                                    for expected_i in expected])

    def test_pickleable(self):
        aug = iaa.SomeOf((0, 3),
            [iaa.Add(1, seed=1),
             iaa.Add(2, seed=2),
             iaa.Multiply(1.5, seed=3),
             iaa.Multiply(2.0, seed=4)],
            random_order=True,
            seed=5)
        runtest_pickleable_uint8_img(aug, iterations=5)

    def test_get_children_lists(self):
        child = iaa.Identity()
        aug = iaa.SomeOf(1, [child])
        children_lsts = aug.get_children_lists()
        assert len(children_lsts) == 1
        assert len(children_lsts[0]) == 1
        assert children_lsts[0][0] is child

    def test_to_deterministic(self):
        child = iaa.Identity()
        aug = iaa.SomeOf(1, [child])

        aug_det = aug.to_deterministic()

        assert aug_det.random_state is not aug.random_state
        assert aug_det.deterministic
        assert aug_det[0].deterministic


class TestOneOf(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_returns_someof(self):
        child = iaa.Identity()
        aug = iaa.OneOf(children=child)
        assert isinstance(aug, iaa.SomeOf)
        assert aug.n == 1
        assert aug[0] is child

    def test_single_child_that_is_augmenter(self):
        zeros = np.zeros((3, 3, 1), dtype=np.uint8)
        aug = iaa.OneOf(children=iaa.Add(1))
        observed = aug.augment_image(zeros)
        assert np.array_equal(observed, zeros + 1)

    def test_single_child_that_is_sequential(self):
        zeros = np.zeros((3, 3, 1), dtype=np.uint8)
        aug = iaa.OneOf(children=iaa.Sequential([iaa.Add(1)]))
        observed = aug.augment_image(zeros)
        assert np.array_equal(observed, zeros + 1)

    def test_single_child_that_is_list(self):
        zeros = np.zeros((3, 3, 1), dtype=np.uint8)
        aug = iaa.OneOf(children=[iaa.Add(1)])
        observed = aug.augment_image(zeros)
        assert np.array_equal(observed, zeros + 1)

    def test_three_children(self):
        zeros = np.zeros((1, 1, 1), dtype=np.uint8)
        augs = [iaa.Add(1), iaa.Add(2), iaa.Add(3)]
        aug = iaa.OneOf(augs)

        results = {1: 0, 2: 0, 3: 0}
        nb_iterations = 1000
        for _ in sm.xrange(nb_iterations):
            result = aug.augment_image(zeros)
            s = int(np.sum(result))
            results[s] += 1

        expected = int(nb_iterations / len(augs))
        tolerance = int(nb_iterations * 0.05)
        for key, val in results.items():
            assert np.isclose(val, expected, rtol=0, atol=tolerance)
        assert len(list(results.keys())) == 3

    def test_pickleable(self):
        aug = iaa.OneOf(
            [iaa.Add(1, seed=1),
             iaa.Add(10, seed=2),
             iaa.Multiply(2.0, seed=3)],
            seed=4)
        runtest_pickleable_uint8_img(aug, iterations=5)

    def test_get_children_lists(self):
        child = iaa.Identity()
        aug = iaa.OneOf([child])
        children_lsts = aug.get_children_lists()
        assert len(children_lsts) == 1
        assert len(children_lsts[0]) == 1
        assert children_lsts[0][0] is child

    def test_to_deterministic(self):
        child = iaa.Identity()
        aug = iaa.OneOf([child])

        aug_det = aug.to_deterministic()

        assert aug_det.random_state is not aug.random_state
        assert aug_det.deterministic
        assert aug_det[0].deterministic


class TestSometimes(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        image = np.array([[0, 1, 1],
                          [0, 0, 1],
                          [0, 0, 1]], dtype=np.uint8) * 255
        return np.atleast_3d(image)

    @property
    def images(self):
        return np.uint8([self.image])

    @property
    def image_lr(self):
        image_lr = np.array([[1, 1, 0],
                             [1, 0, 0],
                             [1, 0, 0]], dtype=np.uint8) * 255
        return np.atleast_3d(image_lr)

    @property
    def images_lr(self):
        return np.uint8([self.image_lr])

    @property
    def image_ud(self):
        image_ud = np.array([[0, 0, 1],
                             [0, 0, 1],
                             [0, 1, 1]], dtype=np.uint8) * 255
        return np.atleast_3d(image_ud)

    @property
    def images_ud(self):
        return np.uint8([self.image_ud])

    @property
    def keypoints(self):
        keypoints = [ia.Keypoint(x=1, y=0),
                     ia.Keypoint(x=2, y=0),
                     ia.Keypoint(x=2, y=1)]
        return ia.KeypointsOnImage(keypoints, shape=self.image.shape)

    @property
    def keypoints_lr(self):
        keypoints = [ia.Keypoint(x=3-1, y=0),
                     ia.Keypoint(x=3-2, y=0),
                     ia.Keypoint(x=3-2, y=1)]
        return ia.KeypointsOnImage(keypoints, shape=self.image.shape)

    @property
    def keypoints_ud(self):
        keypoints = [ia.Keypoint(x=1, y=3-0),
                     ia.Keypoint(x=2, y=3-0),
                     ia.Keypoint(x=2, y=3-1)]
        return ia.KeypointsOnImage(keypoints, shape=self.image.shape)

    @property
    def polygons(self):
        polygons = [ia.Polygon([(0, 0), (2, 0), (2, 2)])]
        return ia.PolygonsOnImage(polygons, shape=self.image.shape)

    @property
    def polygons_lr(self):
        polygons = [ia.Polygon([(3-0, 0), (3-2, 0), (3-2, 2)])]
        return ia.PolygonsOnImage(polygons, shape=self.image.shape)

    @property
    def polygons_ud(self):
        polygons = [ia.Polygon([(0, 3-0), (2, 3-0), (2, 3-2)])]
        return ia.PolygonsOnImage(polygons, shape=self.image.shape)

    @property
    def lsoi(self):
        lss = [ia.LineString([(0, 0), (2, 0), (2, 2)])]
        return ia.LineStringsOnImage(lss, shape=self.image.shape)

    @property
    def lsoi_lr(self):
        lss = [ia.LineString([(3-0, 0), (3-2, 0), (3-2, 2)])]
        return ia.LineStringsOnImage(lss, shape=self.image.shape)

    @property
    def lsoi_ud(self):
        lss = [ia.LineString([(0, 3-0), (2, 3-0), (2, 3-2)])]
        return ia.LineStringsOnImage(lss, shape=self.image.shape)

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=1.5, y2=1.0)]
        return ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)

    @property
    def bbsoi_lr(self):
        x1 = 3-0
        y1 = 0
        x2 = 3-1.5
        y2 = 1.0
        bbs = [ia.BoundingBox(x1=min([x1, x2]), y1=min([y1, y2]),
                              x2=max([x1, x2]), y2=max([y1, y2]))]
        return ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)

    @property
    def bbsoi_ud(self):
        x1 = 0
        y1 = 3-0
        x2 = 1.5
        y2 = 3-1.0
        bbs = [ia.BoundingBox(x1=min([x1, x2]), y1=min([y1, y2]),
                              x2=max([x1, x2]), y2=max([y1, y2]))]
        return ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 1.0, 1.0]])
        return ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def heatmaps_lr(self):
        heatmaps_arr = np.float32([[1.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [1.0, 1.0, 0.0]])
        return ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def heatmaps_ud(self):
        heatmaps_arr = np.float32([[0.0, 1.0, 1.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0]])
        return ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps(self):
        segmaps_arr = np.int32([[0, 0, 1],
                                [0, 0, 1],
                                [0, 1, 1]])
        return ia.SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps_lr(self):
        segmaps_arr = np.int32([[1, 0, 0],
                                [1, 0, 0],
                                [1, 1, 0]])
        return ia.SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps_ud(self):
        segmaps_arr = np.int32([[0, 1, 1],
                                [0, 0, 1],
                                [0, 0, 1]])
        return ia.SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    def test_two_branches_always_first__images(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_images(self.images)

        assert np.array_equal(observed, self.images_lr)

    def test_two_branches_always_first__images__deterministic(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()
        observed = aug_det.augment_images(self.images)
        assert np.array_equal(observed, self.images_lr)

    def test_two_branches_always_first__images__list(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        observed = aug.augment_images([self.images[0]])
        assert array_equal_lists(observed, [self.images_lr[0]])

    def test_two_branches_always_first__images__deterministic__list(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()
        observed = aug_det.augment_images([self.images[0]])
        assert array_equal_lists(observed, [self.images_lr[0]])

    def test_two_branches_always_first__keypoints(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        observed = aug.augment_keypoints(self.keypoints)
        assert keypoints_equal(observed, self.keypoints_lr)

    def test_two_branches_always_first__keypoints__deterministic(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_keypoints(self.keypoints)

        assert_cbaois_equal(observed, self.keypoints_lr)

    def test_two_branches_always_first__polygons(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_polygons([self.polygons])

        assert_cbaois_equal(observed, [self.polygons_lr])

    def test_two_branches_always_first__polygons__deterministic(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_polygons([self.polygons])

        assert_cbaois_equal(observed, [self.polygons_lr])

    def test_two_branches_always_first__line_strings(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_line_strings([self.lsoi])

        assert_cbaois_equal(observed, [self.lsoi_lr])

    def test_two_branches_always_first__line_strings__deterministic(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_line_strings([self.lsoi])

        assert_cbaois_equal(observed, [self.lsoi_lr])

    def test_two_branches_always_first__bounding_boxes(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_bounding_boxes([self.bbsoi])

        assert_cbaois_equal(observed, [self.bbsoi_lr])

    def test_two_branches_always_first__bounding_boxes__deterministic(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_bounding_boxes([self.bbsoi])

        assert_cbaois_equal(observed, [self.bbsoi_lr])

    def test_two_branches_always_first__heatmaps(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_heatmaps([self.heatmaps])[0]

        assert observed.shape == self.heatmaps.shape
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.array_equal(observed.get_arr(), self.heatmaps_lr.get_arr())

    def test_two_branches_always_first__segmaps(self):
        aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_segmentation_maps(self.segmaps)

        assert observed.shape == self.segmaps.shape
        assert np.array_equal(observed.get_arr(), self.segmaps_lr.get_arr())

    def test_two_branches_always_second__images(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        observed = aug.augment_images(self.images)
        assert np.array_equal(observed, self.images_ud)

    def test_two_branches_always_second__images__deterministic(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()
        observed = aug_det.augment_images(self.images)
        assert np.array_equal(observed, self.images_ud)

    def test_two_branches_always_second__images__list(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        observed = aug.augment_images([self.images[0]])
        assert array_equal_lists(observed, [self.images_ud[0]])

    def test_two_branches_always_second__images__list__deterministic(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()
        observed = aug_det.augment_images([self.images[0]])
        assert array_equal_lists(observed, [self.images_ud[0]])

    def test_two_branches_always_second__keypoints(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_keypoints([self.keypoints])

        assert_cbaois_equal(observed[0], self.keypoints_ud)

    def test_two_branches_always_second__keypoints__deterministic(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_keypoints([self.keypoints])

        assert_cbaois_equal(observed[0], self.keypoints_ud)

    def test_two_branches_always_second__polygons(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_polygons(self.polygons)

        assert_cbaois_equal(observed, self.polygons_ud)

    def test_two_branches_always_second__polygons__deterministic(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_polygons(self.polygons)

        assert_cbaois_equal(observed, self.polygons_ud)

    def test_two_branches_always_second__line_strings(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_line_strings(self.lsoi)

        assert_cbaois_equal(observed, self.lsoi_ud)

    def test_two_branches_always_second__line_strings__deterministic(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_line_strings(self.lsoi)

        assert_cbaois_equal(observed, self.lsoi_ud)

    def test_two_branches_always_second__bounding_boxes(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_bounding_boxes(self.bbsoi)

        assert_cbaois_equal(observed, self.bbsoi_ud)

    def test_two_branches_always_second__bounding_boxes__deterministic(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_bounding_boxes(self.bbsoi)

        assert_cbaois_equal(observed, self.bbsoi_ud)

    def test_two_branches_always_second__heatmaps(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_heatmaps(self.heatmaps)

        assert observed.shape == self.heatmaps.shape
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.array_equal(observed.get_arr(), self.heatmaps_ud.get_arr())

    def test_two_branches_always_second__segmaps(self):
        aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])

        observed = aug.augment_segmentation_maps(self.segmaps)

        assert observed.shape == self.segmaps.shape
        assert np.array_equal(observed.get_arr(), self.segmaps_ud.get_arr())

    def test_two_branches_both_50_percent__images(self):
        aug = iaa.Sometimes(0.5, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        last_aug = None
        nb_changed_aug = 0
        nb_iterations = 500
        nb_images_if_branch = 0
        nb_images_else_branch = 0
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(self.images)
            if i == 0:
                last_aug = observed_aug
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                last_aug = observed_aug

            if np.array_equal(observed_aug, self.images_lr):
                nb_images_if_branch += 1
            elif np.array_equal(observed_aug, self.images_ud):
                nb_images_else_branch += 1
            else:
                raise Exception(
                    "Received output doesnt match any expected output.")

        p_if_branch = nb_images_if_branch / nb_iterations
        p_else_branch = nb_images_else_branch / nb_iterations
        p_changed = 1 - (nb_changed_aug / nb_iterations)

        assert np.isclose(p_if_branch, 0.5, rtol=0, atol=0.1)
        assert np.isclose(p_else_branch, 0.5, rtol=0, atol=0.1)
        # should be the same in roughly 50% of all cases
        assert np.isclose(p_changed, 0.5, rtol=0, atol=0.1)

    def test_two_branches_both_50_percent__images__deterministic(self):
        aug = iaa.Sometimes(0.5, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        aug_det = aug.to_deterministic()
        last_aug_det = None
        nb_changed_aug_det = 0
        nb_iterations = 20
        for i in sm.xrange(nb_iterations):
            observed_aug_det = aug_det.augment_images(self.images)
            if i == 0:
                last_aug_det = observed_aug_det
            else:
                if not np.array_equal(observed_aug_det, last_aug_det):
                    nb_changed_aug_det += 1
                last_aug_det = observed_aug_det

        assert nb_changed_aug_det == 0

    @classmethod
    def _test_two_branches_both_50_percent__cbaois(
            cls, cbaoi, cbaoi_lr, cbaoi_ud, augf_name):
        def _same_coords(cbaoi1, cbaoi2):
            assert len(cbaoi1.items) == len(cbaoi2.items)
            for i1, i2 in zip(cbaoi1.items, cbaoi2.items):
                if not np.allclose(i1.coords, i2.coords, atol=1e-4, rtol=0):
                    return False
            return True

        aug = iaa.Sometimes(0.5, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
        nb_iterations = 250
        nb_if_branch = 0
        nb_else_branch = 0
        for i in sm.xrange(nb_iterations):
            cbaoi_aug = getattr(aug, augf_name)(cbaoi)

            # use allclose() instead of coords_almost_equals() for efficiency
            if _same_coords(cbaoi_aug, cbaoi_lr):
                nb_if_branch += 1
            elif _same_coords(cbaoi_aug, cbaoi_ud):
                nb_else_branch += 1
            else:
                raise Exception(
                    "Received output doesnt match any expected output.")

        p_if_branch = nb_if_branch / nb_iterations
        p_else_branch = nb_else_branch / nb_iterations

        assert np.isclose(p_if_branch, 0.5, rtol=0, atol=0.15)
        assert np.isclose(p_else_branch, 0.5, rtol=0, atol=0.15)

    def test_two_branches_both_50_percent__keypoints(self):
        self._test_two_branches_both_50_percent__cbaois(
            self.keypoints, self.keypoints_lr, self.keypoints_ud,
            "augment_keypoints")

    def test_two_branches_both_50_percent__polygons(self):
        self._test_two_branches_both_50_percent__cbaois(
            self.polygons, self.polygons_lr, self.polygons_ud,
            "augment_polygons")

    def test_two_branches_both_50_percent__line_strings(self):
        self._test_two_branches_both_50_percent__cbaois(
            self.lsoi, self.lsoi_lr, self.lsoi_ud,
            "augment_line_strings")

    def test_two_branches_both_50_percent__bounding_boxes(self):
        self._test_two_branches_both_50_percent__cbaois(
            self.bbsoi, self.bbsoi_lr, self.bbsoi_ud,
            "augment_bounding_boxes")

    def test_one_branch_50_percent__images(self):
        aug = iaa.Sometimes(0.5, iaa.Fliplr(1.0))
        last_aug = None
        nb_changed_aug = 0
        nb_iterations = 500
        nb_images_if_branch = 0
        nb_images_else_branch = 0
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(self.images)
            if i == 0:
                last_aug = observed_aug
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                last_aug = observed_aug

            if np.array_equal(observed_aug, self.images_lr):
                nb_images_if_branch += 1
            elif np.array_equal(observed_aug, self.images):
                nb_images_else_branch += 1
            else:
                raise Exception(
                    "Received output doesnt match any expected output.")

        p_if_branch = nb_images_if_branch / nb_iterations
        p_else_branch = nb_images_else_branch / nb_iterations
        p_changed = 1 - (nb_changed_aug / nb_iterations)

        assert np.isclose(p_if_branch, 0.5, rtol=0, atol=0.1)
        assert np.isclose(p_else_branch, 0.5, rtol=0, atol=0.1)
        # should be the same in roughly 50% of all cases
        assert np.isclose(p_changed, 0.5, rtol=0, atol=0.1)

    def test_one_branch_50_percent__images__deterministic(self):
        aug = iaa.Sometimes(0.5, iaa.Fliplr(1.0))
        aug_det = aug.to_deterministic()
        last_aug_det = None
        nb_changed_aug_det = 0
        nb_iterations = 10
        for i in sm.xrange(nb_iterations):
            observed_aug_det = aug_det.augment_images(self.images)
            if i == 0:
                last_aug_det = observed_aug_det
            else:
                if not np.array_equal(observed_aug_det, last_aug_det):
                    nb_changed_aug_det += 1
                last_aug_det = observed_aug_det

        assert nb_changed_aug_det == 0

    @classmethod
    def _test_one_branch_50_percent__cbaois(
            cls, cbaoi, cbaoi_lr, augf_name):
        def _same_coords(cbaoi1, cbaoi2):
            assert len(cbaoi1.items) == len(cbaoi2.items)
            for i1, i2 in zip(cbaoi1.items, cbaoi2.items):
                if not np.allclose(i1.coords, i2.coords, atol=1e-4, rtol=0):
                    return False
            return True

        aug = iaa.Sometimes(0.5, iaa.Fliplr(1.0))
        nb_iterations = 250
        nb_if_branch = 0
        nb_else_branch = 0
        for i in sm.xrange(nb_iterations):
            cbaoi_aug = getattr(aug, augf_name)(cbaoi)

            # use allclose() instead of coords_almost_equals() for efficiency
            if _same_coords(cbaoi_aug, cbaoi_lr):
                nb_if_branch += 1
            elif _same_coords(cbaoi_aug, cbaoi):
                nb_else_branch += 1
            else:
                raise Exception(
                    "Received output doesnt match any expected output.")

        p_if_branch = nb_if_branch / nb_iterations
        p_else_branch = nb_else_branch / nb_iterations

        assert np.isclose(p_if_branch, 0.5, rtol=0, atol=0.15)
        assert np.isclose(p_else_branch, 0.5, rtol=0, atol=0.15)

    def test_one_branch_50_percent__keypoints(self):
        self._test_one_branch_50_percent__cbaois(
            self.keypoints, self.keypoints_lr, "augment_keypoints")

    def test_one_branch_50_percent__polygons(self):
        self._test_one_branch_50_percent__cbaois(
            self.polygons, self.polygons_lr, "augment_polygons")

    def test_one_branch_50_percent__bounding_boxes(self):
        self._test_one_branch_50_percent__cbaois(
            self.bbsoi, self.bbsoi_lr, "augment_bounding_boxes")

    @classmethod
    def _test_empty_cbaoi(cls, cbaoi, augf_name):
        aug = iaa.Sometimes(0.5, iaa.Identity())

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi)

    def test_empty_keypoints(self):
        kpsoi = ia.KeypointsOnImage([], shape=(1, 2, 3))
        self._test_empty_cbaoi(kpsoi, "augment_keypoints")

    def test_empty_polygons(self):
        psoi = ia.PolygonsOnImage([], shape=(1, 2, 3))
        self._test_empty_cbaoi(psoi, "augment_polygons")

    def test_empty_line_strings(self):
        lsoi = ia.LineStringsOnImage([], shape=(1, 2, 3))
        self._test_empty_cbaoi(lsoi, "augment_line_strings")

    def test_empty_bounding_boxes(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(1, 2, 3))
        self._test_empty_cbaoi(bbsoi, "augment_bounding_boxes")

    def test_p_is_stochastic_parameter(self):
        image = np.zeros((1, 1), dtype=np.uint8) + 100
        images = [image] * 10
        aug = iaa.Sometimes(
            p=iap.Binomial(iap.Choice([0.0, 1.0])),
            then_list=iaa.Add(10))

        seen = [0, 0]
        for _ in sm.xrange(100):
            observed = aug.augment_images(images)
            uq = np.unique(np.uint8(observed))
            assert len(uq) == 1
            if uq[0] == 100:
                seen[0] += 1
            elif uq[0] == 110:
                seen[1] += 1
            else:
                assert False
        assert seen[0] > 20
        assert seen[1] > 20

    def test_bad_datatype_for_p_fails(self):
        got_exception = False
        try:
            _ = iaa.Sometimes(p="foo")
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_bad_datatype_for_then_list_fails(self):
        got_exception = False
        try:
            _ = iaa.Sometimes(p=0.2, then_list=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_bad_datatype_for_else_list_fails(self):
        got_exception = False
        try:
            _ = iaa.Sometimes(p=0.2, then_list=None, else_list=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_two_branches_both_none(self):
        aug = iaa.Sometimes(0.2, then_list=None, else_list=None)
        image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, image)

    def test_using_hooks_to_deactivate_propagation(self):
        image = np.random.randint(0, 255-10, size=(16, 16), dtype=np.uint8)
        aug = iaa.Sometimes(1.0, iaa.Add(10))

        def _propagator(images, augmenter, parents, default):
            return False if augmenter == aug else default

        hooks = ia.HooksImages(propagator=_propagator)

        observed1 = aug.augment_image(image)
        observed2 = aug.augment_image(image, hooks=hooks)
        assert np.array_equal(observed1, image + 10)
        assert np.array_equal(observed2, image)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Sometimes(1.0, iaa.Identity())

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Sometimes(1.0, iaa.Identity())

                image_aug = aug(image=image)

                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_get_parameters(self):
        aug = iaa.Sometimes(0.75)
        params = aug.get_parameters()
        assert is_parameter_instance(params[0], iap.Binomial)
        assert is_parameter_instance(params[0].p, iap.Deterministic)
        assert 0.75 - 1e-8 < params[0].p.value < 0.75 + 1e-8

    def test___str___and___repr__(self):
        then_list = iaa.Add(1)
        else_list = iaa.Add(2)
        aug = iaa.Sometimes(
            0.5,
            then_list=then_list,
            else_list=else_list,
            name="SometimesTest")

        expected_then_list = (
            "Sequential("
            "name=SometimesTest-then, "
            "random_order=False, "
            "children=[%s], "
            "deterministic=False"
            ")" % (str(then_list),))
        expected_else_list = (
            "Sequential("
            "name=SometimesTest-else, "
            "random_order=False, "
            "children=[%s], "
            "deterministic=False"
            ")" % (str(else_list),))
        expected = (
            "Sometimes("
            "p=%s, name=%s, then_list=%s, else_list=%s, deterministic=%s"
            ")" % (
                str(aug.p),
                "SometimesTest",
                expected_then_list,
                expected_else_list,
                "False"))

        observed_str = aug.__str__()
        observed_repr = aug.__repr__()

        assert observed_str == expected
        assert observed_repr == expected

    def test___str___and___repr___with_nones_as_children(self):
        aug = iaa.Sometimes(
            0.5,
            then_list=None,
            else_list=None,
            name="SometimesTest")

        expected = (
            "Sometimes("
            "p=%s, "
            "name=%s, "
            "then_list=%s, "
            "else_list=%s, "
            "deterministic=%s"
            ")" % (
                str(aug.p),
                "SometimesTest",
                "None",
                "None",
                "False"))

        observed_str = aug.__str__()
        observed_repr = aug.__repr__()

        assert observed_str == expected
        assert observed_repr == expected

    def test_shapes_changed_by_children__no_keep_size_non_stochastic(self):
        # Test for https://github.com/aleju/imgaug/issues/143
        # (shapes change in child augmenters, leading to problems if input
        # arrays are assumed to stay input arrays)
        def _assert_all_valid_shapes(images):
            expected_shapes = [(4, 8, 3), (6, 8, 3)]
            assert np.all([img.shape in expected_shapes for img in images])

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        aug = iaa.Sometimes(
            0.5,
            iaa.Crop((2, 0, 2, 0), keep_size=False),
            iaa.Crop((1, 0, 1, 0), keep_size=False)
        )

        for _ in sm.xrange(10):
            observed = aug.augment_images(
                np.uint8([image, image, image, image]))
            assert isinstance(observed, list) or ia.is_np_array(observed)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images([image, image, image, image])
            assert isinstance(observed, list)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images(np.uint8([image]))
            assert isinstance(observed, list) or ia.is_np_array(observed)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images([image])
            assert isinstance(observed, list)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_image(image)
            assert ia.is_np_array(image)
            _assert_all_valid_shapes([observed])

    def test_shapes_changed_by_children__no_keep_size_stochastic(self):
        def _assert_all_valid_shapes(images):
            assert np.all([
                16 <= img.shape[0] <= 30
                and img.shape[1:] == (32, 3) for img in images
            ])

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        aug = iaa.Sometimes(
            0.5,
            iaa.Crop(((1, 4), 0, (1, 4), 0), keep_size=False),
            iaa.Crop(((4, 8), 0, (4, 8), 0), keep_size=False)
        )

        for _ in sm.xrange(10):
            observed = aug.augment_images(
                np.uint8([image, image, image, image]))
            assert isinstance(observed, list) or ia.is_np_array(observed)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images([image, image, image, image])
            assert isinstance(observed, list)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images(np.uint8([image]))
            assert isinstance(observed, list)  or ia.is_np_array(observed)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images([image])
            assert isinstance(observed, list)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_image(image)
            assert ia.is_np_array(image)
            _assert_all_valid_shapes([observed])

    def test_shapes_changed_by_children__keep_size_non_stochastic(self):
        def _assert_all_valid_shapes(images):
            expected_shapes = [(8, 8, 3)]
            assert np.all([img.shape in expected_shapes for img in images])

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        aug = iaa.Sometimes(
            0.5,
            iaa.Crop((2, 0, 2, 0), keep_size=True),
            iaa.Crop((1, 0, 1, 0), keep_size=True)
        )

        for _ in sm.xrange(10):
            observed = aug.augment_images(
                np.uint8([image, image, image, image]))
            assert ia.is_np_array(observed)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images([image, image, image, image])
            assert isinstance(observed, list)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images(np.uint8([image]))
            assert ia.is_np_array(observed)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images([image])
            assert isinstance(observed, list)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_image(image)
            assert ia.is_np_array(observed)
            _assert_all_valid_shapes([observed])

    def test_shapes_changed_by_children__keep_size_stochastic(self):
        def _assert_all_valid_shapes(images):
            # only one shape expected here despite stochastic crop ranges
            # due to keep_size=True
            expected_shapes = [(8, 8, 3)]
            assert np.all([img.shape in expected_shapes for img in images])

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        aug = iaa.Sometimes(
            0.5,
            iaa.Crop(((1, 4), 0, (1, 4), 0), keep_size=True),
            iaa.Crop(((4, 8), 0, (4, 8), 0), keep_size=True)
        )

        for _ in sm.xrange(10):
            observed = aug.augment_images(
                np.uint8([image, image, image, image]))
            assert ia.is_np_array(observed)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images([image, image, image, image])
            assert isinstance(observed, list)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images(np.uint8([image]))
            assert ia.is_np_array(observed)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_images([image])
            assert isinstance(observed, list)
            _assert_all_valid_shapes(observed)

            observed = aug.augment_image(image)
            assert ia.is_np_array(observed)
            _assert_all_valid_shapes([observed])

    def test_other_dtypes_via_noop__bool(self):
        aug = iaa.Sometimes(1.0, iaa.Identity())
        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == image.dtype.name
        assert np.all(image_aug == image)

    def test_other_dtypes_via_noop__uint_int(self):
        aug = iaa.Sometimes(1.0, iaa.Identity())
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, _center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, image)

    def test_other_dtypes_via_noop__float(self):
        aug = iaa.Sometimes(1.0, iaa.Identity())

        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128
        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]
        for dtype, value in zip(dtypes, values):
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.all(image_aug == image)

    def test_other_dtypes_via_flip__bool(self):
        aug = iaa.Sometimes(0.5, iaa.Fliplr(1.0), iaa.Flipud(1.0))
        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True
        expected = [np.zeros((3, 3), dtype=bool) for _ in sm.xrange(2)]
        expected[0][0, 2] = True
        expected[1][2, 0] = True
        seen = [False, False]
        for _ in sm.xrange(100):
            image_aug = aug.augment_image(image)

            assert image_aug.dtype.name == image.dtype.name
            if np.all(image_aug == expected[0]):
                seen[0] = True
            elif np.all(image_aug == expected[1]):
                seen[1] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    def test_other_dtypes_via_flip__uint_int(self):
        aug = iaa.Sometimes(0.5, iaa.Fliplr(1.0), iaa.Flipud(1.0))
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, _center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value
                expected = [np.zeros((3, 3), dtype=dtype) for _ in sm.xrange(2)]
                expected[0][0, 2] = value
                expected[1][2, 0] = value
                seen = [False, False]
                for _ in sm.xrange(100):
                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    if np.all(image_aug == expected[0]):
                        seen[0] = True
                    elif np.all(image_aug == expected[1]):
                        seen[1] = True
                    else:
                        assert False
                    if np.all(seen):
                        break
                assert np.all(seen)

    def test_other_dtypes_via_flip__float(self):
        aug = iaa.Sometimes(0.5, iaa.Fliplr(1.0), iaa.Flipud(1.0))

        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128

        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]
        for dtype, value in zip(dtypes, values):
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = value
                expected = [np.zeros((3, 3), dtype=dtype) for _ in sm.xrange(2)]
                expected[0][0, 2] = value
                expected[1][2, 0] = value
                seen = [False, False]
                for _ in sm.xrange(100):
                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    if np.all(image_aug == expected[0]):
                        seen[0] = True
                    elif np.all(image_aug == expected[1]):
                        seen[1] = True
                    else:
                        assert False
                    if np.all(seen):
                        break
                assert np.all(seen)

    def test_pickleable(self):
        aug = iaa.Sometimes(0.5, iaa.Add(10), [iaa.Add(1), iaa.Multiply(2.0)],
                            seed=1)
        runtest_pickleable_uint8_img(aug, iterations=5)

    def test_get_children_lists(self):
        child = iaa.Identity()
        aug = iaa.Sometimes(0.5, [child])
        children_lsts = aug.get_children_lists()
        assert len(children_lsts) == 1
        assert len(children_lsts[0]) == 1
        assert children_lsts[0][0] is child

    def test_get_children_lists_both_lists(self):
        child = iaa.Identity()
        child2 = iaa.Identity()
        aug = iaa.Sometimes(0.5, [child], [child2])
        children_lsts = aug.get_children_lists()
        assert len(children_lsts) == 2
        assert len(children_lsts[0]) == 1
        assert len(children_lsts[1]) == 1
        assert children_lsts[0][0] is child
        assert children_lsts[1][0] is child2

    def test_to_deterministic(self):
        child = iaa.Identity()
        child2 = iaa.Identity()
        aug = iaa.Sometimes(0.5, [child], [child2])

        aug_det = aug.to_deterministic()

        assert aug_det.deterministic
        assert aug_det.random_state is not aug.random_state
        assert aug_det.then_list[0].deterministic
        assert aug_det.else_list[0].deterministic


class TestWithChannels(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.zeros((3, 3, 2), dtype=np.uint8)
        base_img[..., 0] += 100
        base_img[..., 1] += 200
        return base_img

    def test_augment_only_channel_0(self):
        aug = iaa.WithChannels(0, iaa.Add(10))
        observed = aug.augment_image(self.image)
        expected = self.image
        expected[..., 0] += 10
        assert np.allclose(observed, expected)

    def test_augment_only_channel_1(self):
        aug = iaa.WithChannels(1, iaa.Add(10))
        observed = aug.augment_image(self.image)
        expected = self.image
        expected[..., 1] += 10
        assert np.allclose(observed, expected)

    def test_augment_all_channels_via_none(self):
        aug = iaa.WithChannels(None, iaa.Add(10))
        observed = aug.augment_image(self.image)
        expected = self.image + 10
        assert np.allclose(observed, expected)

    def test_augment_channels_0_and_1_via_list(self):
        aug = iaa.WithChannels([0, 1], iaa.Add(10))
        observed = aug.augment_image(self.image)
        expected = self.image + 10
        assert np.allclose(observed, expected)

    def test_apply_multiple_augmenters(self):
        image = np.zeros((3, 3, 2), dtype=np.uint8)
        image[..., 0] += 5
        image[..., 1] += 10
        aug = iaa.WithChannels(1, [iaa.Add(10), iaa.Multiply(2.0)])

        observed = aug.augment_image(image)

        expected = np.copy(image)
        expected[..., 1] += 10
        expected[..., 1] *= 2
        assert np.allclose(observed, expected)

    def test_multiple_images_given_as_array(self):
        images = np.concatenate([
            self.image[np.newaxis, ...],
            self.image[np.newaxis, ...]],
            axis=0)
        aug = iaa.WithChannels(1, iaa.Add(10))

        observed = aug.augment_images(images)

        expected = np.copy(images)
        expected[..., 1] += 10
        assert np.allclose(observed, expected)

    def test_multiple_images_given_as_list_of_arrays(self):
        images = [self.image, self.image]
        aug = iaa.WithChannels(1, iaa.Add(10))

        observed = aug.augment_images(images)

        expected = self.image
        expected[..., 1] += 10
        expected = [expected, expected]
        assert array_equal_lists(observed, expected)

    def test_children_list_is_none(self):
        aug = iaa.WithChannels(1, children=None)
        observed = aug.augment_image(self.image)
        expected = self.image
        assert np.array_equal(observed, expected)

    def test_channels_is_empty_list(self):
        aug = iaa.WithChannels([], iaa.Add(10))
        observed = aug.augment_image(self.image)
        expected = self.image
        assert np.array_equal(observed, expected)

    def test_heatmap_augmentation_single_channel(self):
        heatmap_arr = np.float32([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        heatmap = HeatmapsOnImage(heatmap_arr, shape=(3, 3, 3))
        affine = iaa.Affine(translate_px={"x": 1})
        aug = iaa.WithChannels(1, children=[affine])

        heatmap_aug = aug.augment_heatmaps(heatmap)

        assert heatmap_aug.shape == (3, 3, 3)
        assert np.allclose(heatmap_aug.get_arr(), heatmap_arr)

    def test_heatmap_augmentation_multiple_channels(self):
        heatmap_arr = np.float32([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        heatmap_arr_shifted = np.float32([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0]
        ])
        heatmap = HeatmapsOnImage(heatmap_arr, shape=(3, 3, 3))
        affine = iaa.Affine(translate_px={"x": 1})
        aug = iaa.WithChannels([0, 1, 2], children=[affine])

        heatmap_aug = aug.augment_heatmaps(heatmap)

        assert heatmap_aug.shape == (3, 3, 3)
        assert np.allclose(heatmap_aug.get_arr(), heatmap_arr_shifted)

    def test_segmentation_map_augmentation_single_channel(self):
        segmap_arr = np.int32([
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ])
        segmap = SegmentationMapsOnImage(segmap_arr, shape=(3, 3, 3))

        aug = iaa.WithChannels(1, children=[iaa.Affine(translate_px={"x": 1})])
        segmap_aug = aug.augment_segmentation_maps(segmap)
        assert segmap_aug.shape == (3, 3, 3)
        assert np.array_equal(segmap_aug.get_arr(), segmap_arr)

    def test_segmentation_map_augmentation_multiple_channels(self):
        segmap_arr = np.int32([
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ])
        segmap_arr_shifted = np.int32([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1]
        ])
        segmap = SegmentationMapsOnImage(segmap_arr, shape=(3, 3, 3))
        affine = iaa.Affine(translate_px={"x": 1})
        aug = iaa.WithChannels([0, 1, 2], children=[affine])

        segmap_aug = aug.augment_segmentation_maps(segmap)

        assert segmap_aug.shape == (3, 3, 3)
        assert np.array_equal(segmap_aug.get_arr(), segmap_arr_shifted)

    @classmethod
    def _test_cbaoi_augmentation_single_channel(cls, cbaoi, augf_name):
        affine = iaa.Affine(translate_px={"x": 1})
        aug = iaa.WithChannels(1, children=[affine])

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi)

    @classmethod
    def _test_cbaoi_augmentation_all_channels_via_list(cls, cbaoi, cbaoi_x,
                                                       augf_name):
        affine = iaa.Affine(translate_px={"x": 1})
        aug = iaa.WithChannels([0, 1, 2], children=[affine])

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi_x)

    @classmethod
    def _test_cbaoi_augmentation_subset_of_channels(cls, cbaoi, cbaoi_x,
                                                    augf_name):
        affine = iaa.Affine(translate_px={"x": 1})
        aug = iaa.WithChannels([0, 1], children=[affine])

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi_x)

    @classmethod
    def _test_cbaoi_augmentation_with_empty_cbaoi(cls, cbaoi, augf_name):
        affine = iaa.Affine(translate_px={"x": 1})
        aug = iaa.WithChannels([0, 1], children=[affine])

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi)

    def test_keypoint_augmentation_single_channel(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=2)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(5, 6, 3))
        self._test_cbaoi_augmentation_single_channel(kpsoi, "augment_keypoints")

    def test_keypoint_augmentation_all_channels_via_list(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=2)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(5, 6, 3))
        kpsoi_x = kpsoi.shift(x=1)
        self._test_cbaoi_augmentation_all_channels_via_list(
            kpsoi, kpsoi_x, "augment_keypoints")

    def test_keypoint_augmentation_subset_of_channels(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=2)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(5, 6, 3))
        kpsoi_x = kpsoi.shift(x=1)
        self._test_cbaoi_augmentation_subset_of_channels(
            kpsoi, kpsoi_x, "augment_keypoints")

    def test_keypoint_augmentation_with_empty_keypoints_instance(self):
        kpsoi = ia.KeypointsOnImage([], shape=(5, 6, 3))
        self._test_cbaoi_augmentation_with_empty_cbaoi(
            kpsoi, "augment_keypoints")

    def test_polygon_augmentation(self):
        polygons = [ia.Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
        psoi = ia.PolygonsOnImage(polygons, shape=(5, 6, 3))
        self._test_cbaoi_augmentation_single_channel(psoi, "augment_polygons")

    def test_polygon_augmentation_all_channels_via_list(self):
        polygons = [ia.Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
        psoi = ia.PolygonsOnImage(polygons, shape=(5, 6, 3))
        psoi_x = psoi.shift(x=1)
        self._test_cbaoi_augmentation_all_channels_via_list(
            psoi, psoi_x, "augment_polygons")

    def test_polygon_augmentation_subset_of_channels(self):
        polygons = [ia.Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])]
        psoi = ia.PolygonsOnImage(polygons, shape=(5, 6, 3))
        psoi_x = psoi.shift(x=1)
        self._test_cbaoi_augmentation_subset_of_channels(
            psoi, psoi_x, "augment_polygons")

    def test_polygon_augmentation_with_empty_polygons_instance(self):
        psoi = ia.PolygonsOnImage([], shape=(5, 6, 3))
        self._test_cbaoi_augmentation_with_empty_cbaoi(
            psoi, "augment_polygons")

    def test_line_string_augmentation(self):
        lss = [ia.LineString([(0, 0), (3, 0), (3, 3), (0, 3)])]
        lsoi = ia.LineStringsOnImage(lss, shape=(5, 6, 3))
        self._test_cbaoi_augmentation_single_channel(
            lsoi, "augment_line_strings")

    def test_line_string_augmentation_all_channels_via_list(self):
        lss = [ia.LineString([(0, 0), (3, 0), (3, 3), (0, 3)])]
        lsoi = ia.LineStringsOnImage(lss, shape=(5, 6, 3))
        lsoi_x = lsoi.shift(x=1)
        self._test_cbaoi_augmentation_all_channels_via_list(
            lsoi, lsoi_x, "augment_line_strings")

    def test_line_string_augmentation_subset_of_channels(self):
        lss = [ia.LineString([(0, 0), (3, 0), (3, 3), (0, 3)])]
        lsoi = ia.LineStringsOnImage(lss, shape=(5, 6, 3))
        lsoi_x = lsoi.shift(x=1)
        self._test_cbaoi_augmentation_subset_of_channels(
            lsoi, lsoi_x, "augment_line_strings")

    def test_line_string_augmentation_with_empty_polygons_instance(self):
        lsoi = ia.LineStringsOnImage([], shape=(5, 6, 3))
        self._test_cbaoi_augmentation_with_empty_cbaoi(
            lsoi, "augment_line_strings")

    def test_bounding_boxes_augmentation(self):
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=1.0, y2=1.5)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(5, 6, 3))
        self._test_cbaoi_augmentation_single_channel(
            bbsoi, "augment_bounding_boxes")

    def test_bounding_boxes_augmentation_all_channels_via_list(self):
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=1.0, y2=1.5)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(5, 6, 3))
        bbsoi_x = bbsoi.shift(x=1)
        self._test_cbaoi_augmentation_all_channels_via_list(
            bbsoi, bbsoi_x, "augment_bounding_boxes")

    def test_bounding_boxes_augmentation_subset_of_channels(self):
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=1.0, y2=1.5)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(5, 6, 3))
        bbsoi_x = bbsoi.shift(x=1)
        self._test_cbaoi_augmentation_subset_of_channels(
            bbsoi, bbsoi_x, "augment_bounding_boxes")

    def test_bounding_boxes_augmentation_with_empty_bb_instance(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(5, 6, 3))
        self._test_cbaoi_augmentation_with_empty_cbaoi(
            bbsoi, "augment_bounding_boxes")

    def test_invalid_datatype_for_channels_fails(self):
        got_exception = False
        try:
            _ = iaa.WithChannels(False, iaa.Add(10))
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_invalid_datatype_for_children_fails(self):
        got_exception = False
        try:
            _ = iaa.WithChannels(1, False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.WithChannels([0], iaa.Add(1))

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.WithChannels([0], iaa.Add(1))

                image_aug = aug(image=image)

                assert np.all(image_aug[..., 0] == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_get_parameters(self):
        aug = iaa.WithChannels([1], iaa.Add(10))
        params = aug.get_parameters()
        assert len(params) == 1
        assert params[0] == [1]

    def test_get_children_lists(self):
        children = iaa.Sequential([iaa.Add(10)])
        aug = iaa.WithChannels(1, children)
        assert aug.get_children_lists() == [children]

    def test_to_deterministic(self):
        child = iaa.Identity()
        aug = iaa.WithChannels(1, [child])

        aug_det = aug.to_deterministic()

        assert aug_det.deterministic
        assert aug_det.random_state is not aug.random_state
        assert aug_det.children[0].deterministic

    def test___repr___and___str__(self):
        children = iaa.Sequential([iaa.Identity()])
        aug = iaa.WithChannels(1, children, name="WithChannelsTest")
        expected = (
            "WithChannels("
            "channels=[1], "
            "name=WithChannelsTest, "
            "children=%s, "
            "deterministic=False"
            ")" % (str(children),))

        assert aug.__repr__() == expected
        assert aug.__str__() == expected

    def test_other_dtypes_via_noop__bool(self):
        aug = iaa.WithChannels([0], iaa.Identity())

        image = np.zeros((3, 3, 2), dtype=bool)
        image[0, 0, :] = True
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.name == image.dtype.name
        assert np.all(image_aug == image)

    def test_other_dtypes_via_noop__uint_int(self):
        aug = iaa.WithChannels([0], iaa.Identity())
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3, 2), dtype=dtype)
                image[0, 0, :] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, image)

    def test_other_dtypes_via_noop__float(self):
        aug = iaa.WithChannels([0], iaa.Identity())

        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128

        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]
        for dtype, value in zip(dtypes, values):
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3, 2), dtype=dtype)
                image[0, 0, :] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.all(image_aug == image)

    def test_other_dtypes_via_flips__bool(self):
        aug = iaa.WithChannels([0], iaa.Fliplr(1.0))

        image = np.zeros((3, 3, 2), dtype=bool)
        image[0, 0, :] = True
        expected = np.zeros((3, 3, 2), dtype=bool)
        expected[0, 2, 0] = True
        expected[0, 0, 1] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == image.dtype.name
        assert np.all(image_aug == expected)

    def test_other_dtypes_via_flips__uint_int(self):
        aug = iaa.WithChannels([0], iaa.Fliplr(1.0))
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3, 2), dtype=dtype)
                image[0, 0, :] = value
                expected = np.zeros((3, 3, 2), dtype=dtype)
                expected[0, 2, 0] = value
                expected[0, 0, 1] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, expected)

    def test_other_dtypes_via_flips__float(self):
        aug = iaa.WithChannels([0], iaa.Fliplr(1.0))

        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128

        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]
        for dtype, value in zip(dtypes, values):
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3, 2), dtype=dtype)
                image[0, 0, :] = value
                expected = np.zeros((3, 3, 2), dtype=dtype)
                expected[0, 2, 0] = value
                expected[0, 0, 1] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.all(image_aug == expected)

    def test_pickleable(self):
        aug = iaa.WithChannels([0], iaa.Add((1, 10), seed=2),
                               seed=1)
        runtest_pickleable_uint8_img(aug, iterations=5)


class TestChannelShuffle(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        aug = iaa.ChannelShuffle(p=0.9, channels=[0, 2])
        assert is_parameter_instance(aug.p, iap.Binomial)
        assert is_parameter_instance(aug.p.p, iap.Deterministic)
        assert np.allclose(aug.p.p.value, 0.9)
        assert aug.channels == [0, 2]

    def test_p_is_1(self):
        aug = iaa.ChannelShuffle(p=1.0)
        img = np.uint8([0, 1]).reshape((1, 1, 2))
        expected = [
            np.uint8([0, 1]).reshape((1, 1, 2)),
            np.uint8([1, 0]).reshape((1, 1, 2))
        ]
        seen = [False, False]
        for _ in sm.xrange(100):
            img_aug = aug.augment_image(img)
            if np.array_equal(img_aug, expected[0]):
                seen[0] = True
            elif np.array_equal(img_aug, expected[1]):
                seen[1] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    def test_p_is_0(self):
        aug = iaa.ChannelShuffle(p=0)
        img = np.uint8([0, 1]).reshape((1, 1, 2))
        for _ in sm.xrange(20):
            img_aug = aug.augment_image(img)
            assert np.array_equal(img_aug, img)

    def test_p_is_1_and_channels_is_limited_subset(self):
        aug = iaa.ChannelShuffle(p=1.0, channels=[0, 2])
        img = np.uint8([0, 1, 2]).reshape((1, 1, 3))
        expected = [
            np.uint8([0, 1, 2]).reshape((1, 1, 3)),
            np.uint8([2, 1, 0]).reshape((1, 1, 3))
        ]
        seen = [False, False]
        for _ in sm.xrange(100):
            img_aug = aug.augment_image(img)
            if np.array_equal(img_aug, expected[0]):
                seen[0] = True
            elif np.array_equal(img_aug, expected[1]):
                seen[1] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    def test_get_parameters(self):
        aug = iaa.ChannelShuffle(p=1.0, channels=[0, 2])
        assert aug.get_parameters()[0] == aug.p
        assert aug.get_parameters()[1] == aug.channels

    def test_heatmaps_must_not_change(self):
        aug = iaa.ChannelShuffle(p=1.0)
        hm = ia.HeatmapsOnImage(np.float32([[0, 0.5, 1.0]]), shape=(4, 4, 3))
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert hm_aug.shape == (4, 4, 3)
        assert hm_aug.arr_0to1.shape == (1, 3, 1)
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_segmentation_maps_must_not_change(self):
        aug = iaa.ChannelShuffle(p=1.0)
        segmap = SegmentationMapsOnImage(np.int32([[0, 1, 2]]), shape=(4, 4, 3))
        segmap_aug = aug.augment_segmentation_maps([segmap])[0]
        assert segmap_aug.shape == (4, 4, 3)
        assert segmap_aug.arr.shape == (1, 3, 1)
        assert np.array_equal(segmap.arr, segmap_aug.arr)

    def test_keypoints_must_not_change(self):
        aug = iaa.ChannelShuffle(p=1.0)
        kpsoi = ia.KeypointsOnImage([
            ia.Keypoint(x=3, y=1), ia.Keypoint(x=2, y=4)
        ], shape=(10, 10, 3))

        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]

        assert_cbaois_equal(kpsoi_aug, kpsoi)

    def test_polygons_must_not_change(self):
        aug = iaa.ChannelShuffle(p=1.0)
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (5, 0), (5, 5)])
        ], shape=(10, 10, 3))

        psoi_aug = aug.augment_polygons(psoi)

        assert_cbaois_equal(psoi_aug, psoi)

    def test_line_strings_must_not_change(self):
        aug = iaa.ChannelShuffle(p=1.0)
        lsoi = ia.LineStringsOnImage([
            ia.LineString([(0, 0), (5, 0), (5, 5)])
        ], shape=(10, 10, 3))

        lsoi_aug = aug.augment_line_strings(lsoi)

        assert_cbaois_equal(lsoi_aug, lsoi)

    def test_bounding_boxes_must_not_change(self):
        aug = iaa.ChannelShuffle(p=1.0)
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=0, x2=1.0, y2=1.5)
        ], shape=(10, 10, 3))

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        assert_cbaois_equal(bbsoi_aug, bbsoi)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.ChannelShuffle(1.0)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.ChannelShuffle(1.0)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_other_dtypes_bool(self):
        aug = iaa.ChannelShuffle(p=0.5)

        image = np.zeros((3, 3, 2), dtype=bool)
        image[0, 0, 0] = True
        expected = [np.zeros((3, 3, 2), dtype=bool) for _ in sm.xrange(2)]
        expected[0][0, 0, 0] = True
        expected[1][0, 0, 1] = True
        seen = [False, False]
        for _ in sm.xrange(100):
            image_aug = aug.augment_image(image)

            assert image_aug.dtype.name == image.dtype.name
            if np.all(image_aug == expected[0]):
                seen[0] = True
            elif np.all(image_aug == expected[1]):
                seen[1] = True
            else:
                assert False
            if np.all(seen):
                break
        assert np.all(seen)

    def test_other_dtypes_uint_int(self):
        aug = iaa.ChannelShuffle(p=0.5)
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = np.zeros((3, 3, 2), dtype=dtype)
                image[0, 0, 0] = value
                expected = [np.zeros((3, 3, 2), dtype=dtype)
                            for _
                            in sm.xrange(2)]
                expected[0][0, 0, 0] = value
                expected[1][0, 0, 1] = value
                seen = [False, False]
                for _ in sm.xrange(100):
                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    if np.all(image_aug == expected[0]):
                        seen[0] = True
                    elif np.all(image_aug == expected[1]):
                        seen[1] = True
                    else:
                        assert False
                    if np.all(seen):
                        break
                assert np.all(seen)

    def test_other_dtypes_float(self):
        aug = iaa.ChannelShuffle(p=0.5)

        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = ["float16", "float32", "float64"] + f128

        values = [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]
        for dtype, value in zip(dtypes, values):
            with self.subTest(dtype=dtype):
                image = np.zeros((3, 3, 2), dtype=dtype)
                image[0, 0, 0] = value
                expected = [np.zeros((3, 3, 2), dtype=dtype)
                            for _
                            in sm.xrange(2)]
                expected[0][0, 0, 0] = value
                expected[1][0, 0, 1] = value
                seen = [False, False]
                for _ in sm.xrange(100):
                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    if np.all(image_aug == expected[0]):
                        seen[0] = True
                    elif np.all(image_aug == expected[1]):
                        seen[1] = True
                    else:
                        assert False
                    if np.all(seen):
                        break
                assert np.all(seen)

    def test_pickleable(self):
        aug = iaa.ChannelShuffle(0.5, seed=1)
        runtest_pickleable_uint8_img(aug, iterations=5, shape=(2, 2, 10))


class TestRemoveCBAsByOutOfImageFraction(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        aug = iaa.RemoveCBAsByOutOfImageFraction(0.51)
        assert np.isclose(aug.fraction, 0.51)

    def test_no_cbas_in_batch(self):
        aug = iaa.RemoveCBAsByOutOfImageFraction(0.51)

        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = [
            "uint8", "uint16", "uint32", "uint64",
            "int8", "int16", "int32", "int64",
            "float16", "float32", "float64",
            "bool"
        ] + f128

        for dt in dtypes:
            arr = np.ones((5, 10, 3), dtype=dt)

            image_aug = aug(image=arr)

            assert image_aug.dtype.name == dt
            assert image_aug.shape == (5, 10, 3)
            if arr.dtype.kind == "f":
                assert np.allclose(image_aug, 1.0)
            else:
                assert np.all(image_aug == 1)

    def test_keypoints(self):
        aug = iaa.RemoveCBAsByOutOfImageFraction(0.51)
        item1 = ia.Keypoint(x=5, y=1)
        item2 = ia.Keypoint(x=15, y=1)
        cbaoi = ia.KeypointsOnImage([item1, item2], shape=(10, 10, 3))

        cbaoi_aug = aug(keypoints=cbaoi)

        assert len(cbaoi_aug.items) == 1
        for item_obs, item_exp in zip(cbaoi_aug.items, [item1]):
            assert item_obs.coords_almost_equals(item_exp)

    def test_bounding_boxes(self):
        aug = iaa.RemoveCBAsByOutOfImageFraction(0.51)
        item1 = ia.BoundingBox(y1=1, x1=5, y2=6, x2=9)
        item2 = ia.BoundingBox(y1=1, x1=5, y2=6, x2=15)
        item3 = ia.BoundingBox(y1=1, x1=15, y2=6, x2=25)
        cbaoi = ia.BoundingBoxesOnImage([item1, item2, item3],
                                        shape=(10, 10, 3))

        cbaoi_aug = aug(bounding_boxes=cbaoi)

        assert len(cbaoi_aug.items) == 2
        for item_obs, item_exp in zip(cbaoi_aug.items, [item1, item2]):
            assert item_obs.coords_almost_equals(item_exp)

    def test_polygons(self):
        aug = iaa.RemoveCBAsByOutOfImageFraction(0.51)
        item1 = ia.Polygon([(5, 1), (9, 1), (9, 2), (5, 2)])
        item2 = ia.Polygon([(5, 1), (15, 1), (15, 2), (5, 2)])
        item3 = ia.Polygon([(15, 1), (25, 1), (25, 2), (15, 2)])
        cbaoi = ia.PolygonsOnImage([item1, item2, item3],
                                   shape=(10, 10, 3))

        cbaoi_aug = aug(polygons=cbaoi)

        assert len(cbaoi_aug.items) == 2
        for item_obs, item_exp in zip(cbaoi_aug.items, [item1, item2]):
            assert item_obs.coords_almost_equals(item_exp)

    def test_line_strings(self):
        aug = iaa.RemoveCBAsByOutOfImageFraction(0.51)
        item1 = ia.LineString([(5, 1), (9, 1)])
        item2 = ia.LineString([(5, 1), (15, 1)])
        item3 = ia.LineString([(15, 1), (25, 1)])
        cbaoi = ia.LineStringsOnImage([item1, item2, item3],
                                      shape=(10, 10, 3))

        cbaoi_aug = aug(line_strings=cbaoi)

        assert len(cbaoi_aug.items) == 2
        for item_obs, item_exp in zip(cbaoi_aug.items, [item1, item2]):
            assert item_obs.coords_almost_equals(item_exp)

    def test_get_parameters(self):
        aug = iaa.RemoveCBAsByOutOfImageFraction(0.51)
        params = aug.get_parameters()
        assert len(params) == 1
        assert np.isclose(params[0], 0.51)

    def test_pickleable(self):
        item1 = ia.Keypoint(x=5, y=1)
        item2 = ia.Keypoint(x=15, y=1)
        cbaoi = ia.KeypointsOnImage([item1, item2], shape=(10, 10, 3))

        augmenter = iaa.RemoveCBAsByOutOfImageFraction(0.51)
        augmenter_pkl = pickle.loads(pickle.dumps(augmenter, protocol=-1))

        for _ in np.arange(3):
            cbaoi_aug = augmenter(keypoints=cbaoi)
            cbaoi_aug_pkl = augmenter_pkl(keypoints=cbaoi)
            assert np.allclose(cbaoi_aug.to_xy_array(), cbaoi_aug_pkl.to_xy_array())


class TestClipCBAsToImagePlanes(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_no_cbas_in_batch(self):
        aug = iaa.RemoveCBAsByOutOfImageFraction(0.51)

        try:
            f128 = [np.dtype("float128")]
        except TypeError:
            f128 = []  # float128 not known by user system

        dtypes = [
            "uint8", "uint16", "uint32", "uint64",
            "int8", "int16", "int32", "int64",
            "float16", "float32", "float64",
            "bool"
        ] + f128

        for dt in dtypes:
            arr = np.ones((5, 10, 3), dtype=dt)

            image_aug = aug(image=arr)

            assert image_aug.dtype.name == dt
            assert image_aug.shape == (5, 10, 3)
            if arr.dtype.kind == "f":
                assert np.allclose(image_aug, 1.0)
            else:
                assert np.all(image_aug == 1)

    def test_keypoints(self):
        aug = iaa.RemoveCBAsByOutOfImageFraction(0.51)
        item1 = ia.Keypoint(x=5, y=1)
        item2 = ia.Keypoint(x=15, y=1)
        cbaoi = ia.KeypointsOnImage([item1, item2], shape=(10, 10, 3))

        cbaoi_aug = aug(keypoints=cbaoi)

        assert len(cbaoi_aug.items) == 1
        for item_obs, item_exp in zip(cbaoi_aug.items, [item1]):
            assert item_obs.coords_almost_equals(item_exp)

    def test_bounding_boxes(self):
        aug = iaa.ClipCBAsToImagePlanes()
        item1 = ia.BoundingBox(y1=1, x1=5, y2=6, x2=9)
        item2 = ia.BoundingBox(y1=1, x1=5, y2=6, x2=15)
        item3 = ia.BoundingBox(y1=1, x1=15, y2=6, x2=25)
        cbaoi = ia.BoundingBoxesOnImage([item1, item2, item3],
                                        shape=(10, 10, 3))

        cbaoi_aug = aug(bounding_boxes=cbaoi)

        expected = [
            ia.BoundingBox(y1=1, x1=5, y2=6, x2=9),
            ia.BoundingBox(y1=1, x1=5, y2=6, x2=10)
        ]
        assert len(cbaoi_aug.items) == len(expected)
        for item_obs, item_exp in zip(cbaoi_aug.items, expected):
            assert item_obs.coords_almost_equals(item_exp)

    def test_polygons(self):
        aug = iaa.ClipCBAsToImagePlanes()
        item1 = ia.Polygon([(5, 1), (9, 1), (9, 2), (5, 2)])
        item2 = ia.Polygon([(5, 1), (15, 1), (15, 2), (5, 2)])
        item3 = ia.Polygon([(15, 1), (25, 1), (25, 2), (15, 2)])
        cbaoi = ia.PolygonsOnImage([item1, item2, item3],
                                   shape=(10, 10, 3))

        cbaoi_aug = aug(polygons=cbaoi)

        expected = [
            ia.Polygon([(5, 1), (9, 1), (9, 2), (5, 2)]),
            ia.Polygon([(5, 1), (10, 1), (10, 2), (5, 2)])
        ]
        assert len(cbaoi_aug.items) == len(expected)
        for item_obs, item_exp in zip(cbaoi_aug.items, expected):
            assert item_obs.coords_almost_equals(item_exp)

    def test_line_strings(self):
        aug = iaa.ClipCBAsToImagePlanes()
        item1 = ia.LineString([(5, 1), (9, 1)])
        item2 = ia.LineString([(5, 1), (15, 1)])
        item3 = ia.LineString([(15, 1), (25, 1)])
        cbaoi = ia.LineStringsOnImage([item1, item2, item3],
                                      shape=(10, 10, 3))

        cbaoi_aug = aug(line_strings=cbaoi)

        expected = [
            ia.LineString([(5, 1), (9, 1)]),
            ia.LineString([(5, 1), (10, 1)])
        ]
        assert len(cbaoi_aug.items) == len(expected)
        for item_obs, item_exp in zip(cbaoi_aug.items, expected):
            assert item_obs.coords_almost_equals(item_exp, max_distance=1e-2)

    def test_get_parameters(self):
        aug = iaa.ClipCBAsToImagePlanes()
        params = aug.get_parameters()
        assert len(params) == 0

    def test_pickleable(self):
        item1 = ia.Keypoint(x=5, y=1)
        item2 = ia.Keypoint(x=15, y=1)
        cbaoi = ia.KeypointsOnImage([item1, item2], shape=(10, 10, 3))

        augmenter = iaa.ClipCBAsToImagePlanes()
        augmenter_pkl = pickle.loads(pickle.dumps(augmenter, protocol=-1))

        for _ in np.arange(3):
            cbaoi_aug = augmenter(keypoints=cbaoi)
            cbaoi_aug_pkl = augmenter_pkl(keypoints=cbaoi)
            assert np.allclose(cbaoi_aug.to_xy_array(), cbaoi_aug_pkl.to_xy_array())
