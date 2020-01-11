from __future__ import print_function, division, absolute_import

import itertools
import warnings
import sys
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

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm
import skimage.morphology
import cv2

import imgaug as ia
from imgaug import random as iarandom
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.testutils import (
    array_equal_lists, keypoints_equal, reseed, assert_cbaois_equal,
    runtest_pickleable_uint8_img)
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.augmenters.geometric as geometriclib


def _assert_same_min_max(observed, actual):
    assert np.isclose(observed.min_value, actual.min_value, rtol=0, atol=1e-6)
    assert np.isclose(observed.max_value, actual.max_value, rtol=0, atol=1e-6)


def _assert_same_shape(observed, actual):
    assert observed.shape == actual.shape

# TODO add more tests for Affine .mode
# TODO add more tests for Affine shear


class TestAffine(unittest.TestCase):
    def test_get_parameters(self):
        aug = iaa.Affine(scale=1, translate_px=2, rotate=3, shear=4,
                         order=1, cval=0, mode="constant", backend="cv2",
                         fit_output=True)

        params = aug.get_parameters()

        assert isinstance(params[0], iap.Deterministic)  # scale
        assert isinstance(params[1], tuple)  # translate
        assert isinstance(params[2], iap.Deterministic)  # rotate
        assert isinstance(params[3], iap.Deterministic)  # shear
        assert params[0].value == 1  # scale
        assert params[1][0].value == 2  # translate
        assert params[2].value == 3  # rotate
        assert params[3].value == 4  # shear
        assert params[4].value == 1  # order
        assert params[5].value == 0  # cval
        assert params[6].value == "constant"  # mode
        assert params[7] == "cv2"  # backend
        assert params[8] is True  # fit_output


class TestAffine___init__(unittest.TestCase):
    def test___init___scale_is_stochastic_parameter(self):
        aug = iaa.Affine(scale=iap.Uniform(0.7, 0.9))

        assert isinstance(aug.scale, iap.Uniform)
        assert isinstance(aug.scale.a, iap.Deterministic)
        assert isinstance(aug.scale.b, iap.Deterministic)
        assert 0.7 - 1e-8 < aug.scale.a.value < 0.7 + 1e-8
        assert 0.9 - 1e-8 < aug.scale.b.value < 0.9 + 1e-8

    def test___init___translate_percent_is_stochastic_parameter(self):
        aug = iaa.Affine(translate_percent=iap.Uniform(0.7, 0.9))

        assert isinstance(aug.translate, tuple)
        assert isinstance(aug.translate[0], iap.Uniform)
        assert isinstance(aug.translate[0].a, iap.Deterministic)
        assert isinstance(aug.translate[0].b, iap.Deterministic)
        assert 0.7 - 1e-8 < aug.translate[0].a.value < 0.7 + 1e-8
        assert 0.9 - 1e-8 < aug.translate[0].b.value < 0.9 + 1e-8
        assert aug.translate[1] is None
        assert aug.translate[2] == "percent"

    def test___init___translate_px_is_stochastic_parameter(self):
        aug = iaa.Affine(translate_px=iap.DiscreteUniform(1, 10))

        assert isinstance(aug.translate, tuple)
        assert isinstance(aug.translate[0], iap.DiscreteUniform)
        assert isinstance(aug.translate[0].a, iap.Deterministic)
        assert isinstance(aug.translate[0].b, iap.Deterministic)
        assert aug.translate[0].a.value == 1
        assert aug.translate[0].b.value == 10
        assert aug.translate[1] is None
        assert aug.translate[2] == "px"

    def test___init___rotate_is_stochastic_parameter(self):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=iap.Uniform(10, 20),
                         shear=0)

        assert isinstance(aug.rotate, iap.Uniform)
        assert isinstance(aug.rotate.a, iap.Deterministic)
        assert aug.rotate.a.value == 10
        assert isinstance(aug.rotate.b, iap.Deterministic)
        assert aug.rotate.b.value == 20

    def test___init___shear_is_stochastic_parameter(self):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0,
                         shear=iap.Uniform(10, 20))

        assert isinstance(aug.shear, iap.Uniform)
        assert isinstance(aug.shear.a, iap.Deterministic)
        assert aug.shear.a.value == 10
        assert isinstance(aug.shear.b, iap.Deterministic)
        assert aug.shear.b.value == 20

    def test___init___cval_is_all(self):
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=ia.ALL)

        assert isinstance(aug.cval, iap.Uniform)
        assert isinstance(aug.cval.a, iap.Deterministic)
        assert isinstance(aug.cval.b, iap.Deterministic)
        assert aug.cval.a.value == 0
        assert aug.cval.b.value == 255

    def test___init___cval_is_stochastic_parameter(self):
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=iap.DiscreteUniform(1, 5))

        assert isinstance(aug.cval, iap.DiscreteUniform)
        assert isinstance(aug.cval.a, iap.Deterministic)
        assert isinstance(aug.cval.b, iap.Deterministic)
        assert aug.cval.a.value == 1
        assert aug.cval.b.value == 5

    def test___init___mode_is_all(self):
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=0, mode=ia.ALL)
        assert isinstance(aug.mode, iap.Choice)

    def test___init___mode_is_string(self):
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=0, mode="edge")
        assert isinstance(aug.mode, iap.Deterministic)
        assert aug.mode.value == "edge"

    def test___init___mode_is_list(self):
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=0, mode=["constant", "edge"])
        assert isinstance(aug.mode, iap.Choice)
        assert (
            len(aug.mode.a) == 2
            and "constant" in aug.mode.a
            and "edge" in aug.mode.a)

    def test___init___mode_is_stochastic_parameter(self):
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=0, mode=iap.Choice(["constant", "edge"]))
        assert isinstance(aug.mode, iap.Choice)
        assert (
            len(aug.mode.a) == 2
            and "constant" in aug.mode.a
            and "edge" in aug.mode.a)

    def test___init___fit_output_is_true(self):
        aug = iaa.Affine(fit_output=True)
        assert aug.fit_output is True

    # ------------
    # exceptions for bad inputs
    # ------------
    def test___init___bad_datatype_for_scale_fails(self):
        with self.assertRaises(Exception):
            _ = iaa.Affine(scale=False)

    def test___init___bad_datatype_for_translate_px_fails(self):
        with self.assertRaises(Exception):
            _ = iaa.Affine(translate_px=False)

    def test___init___bad_datatype_for_translate_percent_fails(self):
        with self.assertRaises(Exception):
            _ = iaa.Affine(translate_percent=False)

    def test___init___bad_datatype_for_rotate_fails(self):
        with self.assertRaises(Exception):
            _ = iaa.Affine(scale=1.0, translate_px=0, rotate=False, shear=0,
                           cval=0)

    def test___init___bad_datatype_for_shear_fails(self):
        with self.assertRaises(Exception):
            _ = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=False,
                           cval=0)

    def test___init___bad_datatype_for_cval_fails(self):
        with self.assertRaises(Exception):
            _ = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                           cval=None)

    def test___init___bad_datatype_for_mode_fails(self):
        with self.assertRaises(Exception):
            _ = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                           cval=0, mode=False)

    def test___init___bad_datatype_for_order_fails(self):
        # bad order datatype in case of backend=cv2
        with self.assertRaises(Exception):
            _ = iaa.Affine(backend="cv2", order="test")

    def test___init___nonexistent_order_for_cv2_fails(self):
        # non-existent order in case of backend=cv2
        with self.assertRaises(AssertionError):
            _ = iaa.Affine(backend="cv2", order=-1)


# TODO add test with multiple images
class TestAffine_noop(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def base_img(self):
        base_img = np.array([[0, 0, 0],
                             [0, 255, 0],
                             [0, 0, 0]], dtype=np.uint8)
        return base_img[:, :, np.newaxis]

    @property
    def images(self):
        return np.array([self.base_img])

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        return [ia.KeypointsOnImage(kps, shape=self.base_img.shape)]

    @property
    def psoi(self):
        polygons = [ia.Polygon([(0, 0), (2, 0), (2, 2)])]
        return [ia.PolygonsOnImage(polygons, shape=self.base_img.shape)]

    @property
    def lsoi(self):
        ls = [ia.LineString([(0, 0), (2, 0), (2, 2)])]
        return [ia.LineStringsOnImage(ls, shape=self.base_img.shape)]

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)]
        return [ia.BoundingBoxesOnImage(bbs, shape=self.base_img.shape)]

    def test_image_noop(self):
        # no translation/scale/rotate/shear, shouldnt change nothing
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=0)

        observed = aug.augment_images(self.images)

        expected = self.images
        assert np.array_equal(observed, expected)

    def test_image_noop__deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        expected = self.images
        assert np.array_equal(observed, expected)

    def test_image_noop__list(self):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=0)

        observed = aug.augment_images([self.base_img])

        expected = [self.base_img]
        assert array_equal_lists(observed, expected)

    def test_image_noop__list_and_deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.base_img])

        expected = [self.base_img]
        assert array_equal_lists(observed, expected)

    def test_keypoints_noop(self):
        self._test_cba_noop("augment_keypoints", self.kpsoi, False)

    def test_keypoints_noop__deterministic(self):
        self._test_cba_noop("augment_keypoints", self.kpsoi, True)

    def test_polygons_noop(self):
        self._test_cba_noop("augment_polygons", self.psoi, False)

    def test_polygons_noop__deterministic(self):
        self._test_cba_noop("augment_polygons", self.psoi, True)

    def test_line_strings_noop(self):
        self._test_cba_noop("augment_line_strings", self.lsoi, False)

    def test_line_strings_noop__deterministic(self):
        self._test_cba_noop("augment_line_strings", self.lsoi, True)

    def test_bounding_boxes_noop(self):
        self._test_cba_noop("augment_bounding_boxes", self.bbsoi, False)

    def test_bounding_boxes_noop__deterministic(self):
        self._test_cba_noop("augment_bounding_boxes", self.bbsoi, True)

    @classmethod
    def _test_cba_noop(cls, augf_name, cbaoi, deterministic):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=0)
        if deterministic:
            aug = aug.to_deterministic()

        observed = getattr(aug, augf_name)(cbaoi)

        expected = cbaoi
        assert_cbaois_equal(observed, expected)


# TODO add test with multiple images
class TestAffine_scale(unittest.TestCase):
    def setUp(self):
        reseed()

    # ---------------------
    # scale: zoom in
    # ---------------------

    @property
    def base_img(self):
        base_img = np.array([[0, 0, 0],
                             [0, 255, 0],
                             [0, 0, 0]], dtype=np.uint8)
        return base_img[:, :, np.newaxis]

    @property
    def images(self):
        return np.array([self.base_img])

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        return [ia.KeypointsOnImage(kps, shape=self.base_img.shape)]

    def kpsoi_scaled(self, scale_y, scale_x):
        coords = np.array([
            [0, 0],
            [1, 1],
            [2, 2]
        ], dtype=np.float32)
        coords_scaled = self._scale_coordinates(coords, scale_y, scale_x)
        return [ia.KeypointsOnImage.from_xy_array(
                    coords_scaled,
                    shape=self.base_img.shape)]

    @property
    def psoi(self):
        polys = [ia.Polygon([(0, 0), (0, 2), (2, 2)])]
        return [ia.PolygonsOnImage(polys, shape=self.base_img.shape)]

    def psoi_scaled(self, scale_y, scale_x):
        coords = np.array([
            [0, 0],
            [0, 2],
            [2, 2]
        ], dtype=np.float32)
        coords_scaled = self._scale_coordinates(coords, scale_y, scale_x)
        return [ia.PolygonsOnImage(
                    [ia.Polygon(coords_scaled)],
                    shape=self.base_img.shape)]

    @property
    def lsoi(self):
        ls = [ia.LineString([(0, 0), (0, 2), (2, 2)])]
        return [ia.LineStringsOnImage(ls, shape=self.base_img.shape)]

    def lsoi_scaled(self, scale_y, scale_x):
        coords = np.array([
            [0, 0],
            [0, 2],
            [2, 2]
        ], dtype=np.float32)
        coords_scaled = self._scale_coordinates(coords, scale_y, scale_x)
        return [ia.LineStringsOnImage(
                    [ia.LineString(coords_scaled)],
                    shape=self.base_img.shape)]

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)]
        return [ia.BoundingBoxesOnImage(bbs, shape=self.base_img.shape)]

    def bbsoi_scaled(self, scale_y, scale_x):
        coords = np.array([
            [0, 1],
            [2, 3]
        ], dtype=np.float32)
        coords_scaled = self._scale_coordinates(coords, scale_y, scale_x)
        return [ia.BoundingBoxesOnImage.from_xyxy_array(
                    coords_scaled.reshape((1, 4)),
                    shape=self.base_img.shape)]

    def _scale_coordinates(self, coords, scale_y, scale_x):
        height, width = self.base_img.shape[0:2]
        coords_scaled = []
        for x, y in coords:
            # the additional +0.5 and -0.5 here makes up for the shift factor
            # used in the affine matrix generation
            offset = 0.0
            x_centered = x - width/2 + offset
            y_centered = y - height/2 + offset
            x_new = x_centered * scale_x + width/2 - offset
            y_new = y_centered * scale_y + height/2 - offset
            coords_scaled.append((x_new, y_new))
        return np.float32(coords_scaled)

    @property
    def scale_zoom_in_outer_pixels(self):
        base_img = self.base_img
        outer_pixels = ([], [])
        for i in sm.xrange(base_img.shape[0]):
            for j in sm.xrange(base_img.shape[1]):
                if i != j:
                    outer_pixels[0].append(i)
                    outer_pixels[1].append(j)
        return outer_pixels

    def test_image_scale_zoom_in(self):
        aug = iaa.Affine(scale=1.75, translate_px=0, rotate=0, shear=0)

        observed = aug.augment_images(self.images)

        outer_pixels = self.scale_zoom_in_outer_pixels
        assert observed[0][1, 1] > 250
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    def test_image_scale_zoom_in__deterministic(self):
        aug = iaa.Affine(scale=1.75, translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        outer_pixels = self.scale_zoom_in_outer_pixels
        assert observed[0][1, 1] > 250
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    def test_image_scale_zoom_in__list(self):
        aug = iaa.Affine(scale=1.75, translate_px=0, rotate=0, shear=0)

        observed = aug.augment_images([self.base_img])

        outer_pixels = self.scale_zoom_in_outer_pixels
        assert observed[0][1, 1] > 250
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    def test_image_scale_zoom_in__list_and_deterministic(self):
        aug = iaa.Affine(scale=1.75, translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.base_img])

        outer_pixels = self.scale_zoom_in_outer_pixels
        assert observed[0][1, 1] > 250
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    def test_keypoints_scale_zoom_in(self):
        self._test_cba_scale(
            "augment_keypoints", 1.75,
            self.kpsoi, self.kpsoi_scaled(1.75, 1.75), False)

    def test_keypoints_scale_zoom_in__deterministic(self):
        self._test_cba_scale(
            "augment_keypoints", 1.75,
            self.kpsoi, self.kpsoi_scaled(1.75, 1.75), True)

    def test_polygons_scale_zoom_in(self):
        self._test_cba_scale(
            "augment_polygons", 1.75,
            self.psoi, self.psoi_scaled(1.75, 1.75), False)

    def test_polygons_scale_zoom_in__deterministic(self):
        self._test_cba_scale(
            "augment_polygons", 1.75,
            self.psoi, self.psoi_scaled(1.75, 1.75), True)

    def test_line_strings_scale_zoom_in(self):
        self._test_cba_scale(
            "augment_line_strings", 1.75,
            self.lsoi, self.lsoi_scaled(1.75, 1.75), False)

    def test_line_strings_scale_zoom_in__deterministic(self):
        self._test_cba_scale(
            "augment_line_strings", 1.75,
            self.lsoi, self.lsoi_scaled(1.75, 1.75), True)

    def test_bounding_boxes_scale_zoom_in(self):
        self._test_cba_scale(
            "augment_bounding_boxes", 1.75,
            self.bbsoi, self.bbsoi_scaled(1.75, 1.75), False)

    def test_bounding_boxes_scale_zoom_in__deterministic(self):
        self._test_cba_scale(
            "augment_bounding_boxes", 1.75,
            self.bbsoi, self.bbsoi_scaled(1.75, 1.75), True)

    @classmethod
    def _test_cba_scale(cls, augf_name, scale, cbaoi, cbaoi_scaled,
                        deterministic):
        aug = iaa.Affine(scale=scale, translate_px=0, rotate=0, shear=0)
        if deterministic:
            aug = aug.to_deterministic()

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi_scaled)

    # ---------------------
    # scale: zoom in only on x axis
    # ---------------------
    def test_image_scale_zoom_in_only_x_axis(self):
        aug = iaa.Affine(scale={"x": 1.75, "y": 1.0},
                         translate_px=0, rotate=0, shear=0)

        observed = aug.augment_images(self.images)

        assert observed[0][1, 1] > 250
        assert (observed[0][[1, 1], [0, 2]] > 20).all()
        assert (observed[0][[1, 1], [0, 2]] < 150).all()
        assert (observed[0][0, :] < 5).all()
        assert (observed[0][2, :] < 5).all()

    def test_image_scale_zoom_in_only_x_axis__deterministic(self):
        aug = iaa.Affine(scale={"x": 1.75, "y": 1.0},
                         translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        assert observed[0][1, 1] > 250
        assert (observed[0][[1, 1], [0, 2]] > 20).all()
        assert (observed[0][[1, 1], [0, 2]] < 150).all()
        assert (observed[0][0, :] < 5).all()
        assert (observed[0][2, :] < 5).all()

    def test_image_scale_zoom_in_only_x_axis__list(self):
        aug = iaa.Affine(scale={"x": 1.75, "y": 1.0},
                         translate_px=0, rotate=0, shear=0)

        observed = aug.augment_images([self.base_img])

        assert observed[0][1, 1] > 250
        assert (observed[0][[1, 1], [0, 2]] > 20).all()
        assert (observed[0][[1, 1], [0, 2]] < 150).all()
        assert (observed[0][0, :] < 5).all()
        assert (observed[0][2, :] < 5).all()

    def test_image_scale_zoom_in_only_x_axis__deterministic_and_list(self):
        aug = iaa.Affine(scale={"x": 1.75, "y": 1.0},
                         translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.base_img])

        assert observed[0][1, 1] > 250
        assert (observed[0][[1, 1], [0, 2]] > 20).all()
        assert (observed[0][[1, 1], [0, 2]] < 150).all()
        assert (observed[0][0, :] < 5).all()
        assert (observed[0][2, :] < 5).all()

    def test_keypoints_scale_zoom_in_only_x_axis(self):
        self._test_cba_scale(
            "augment_keypoints", {"y": 1.0, "x": 1.75}, self.kpsoi,
            self.kpsoi_scaled(1.0, 1.75), False)

    def test_keypoints_scale_zoom_in_only_x_axis__deterministic(self):
        self._test_cba_scale(
            "augment_keypoints", {"y": 1.0, "x": 1.75}, self.kpsoi,
            self.kpsoi_scaled(1.0, 1.75), True)

    def test_polygons_scale_zoom_in_only_x_axis(self):
        self._test_cba_scale(
            "augment_polygons", {"y": 1.0, "x": 1.75}, self.psoi,
            self.psoi_scaled(1.0, 1.75), False)

    def test_polygons_scale_zoom_in_only_x_axis__deterministic(self):
        self._test_cba_scale(
            "augment_polygons", {"y": 1.0, "x": 1.75}, self.psoi,
            self.psoi_scaled(1.0, 1.75), True)

    def test_line_strings_scale_zoom_in_only_x_axis(self):
        self._test_cba_scale(
            "augment_line_strings", {"y": 1.0, "x": 1.75}, self.lsoi,
            self.lsoi_scaled(1.0, 1.75), False)

    def test_line_strings_scale_zoom_in_only_x_axis__deterministic(self):
        self._test_cba_scale(
            "augment_line_strings", {"y": 1.0, "x": 1.75}, self.lsoi,
            self.lsoi_scaled(1.0, 1.75), True)

    def test_bounding_boxes_scale_zoom_in_only_x_axis(self):
        self._test_cba_scale(
            "augment_bounding_boxes", {"y": 1.0, "x": 1.75}, self.bbsoi,
            self.bbsoi_scaled(1.0, 1.75), False)

    def test_bounding_boxes_scale_zoom_in_only_x_axis__deterministic(self):
        self._test_cba_scale(
            "augment_bounding_boxes", {"y": 1.0, "x": 1.75}, self.bbsoi,
            self.bbsoi_scaled(1.0, 1.75), True)

    # ---------------------
    # scale: zoom in only on y axis
    # ---------------------
    def test_image_scale_zoom_in_only_y_axis(self):
        aug = iaa.Affine(scale={"x": 1.0, "y": 1.75},
                         translate_px=0, rotate=0, shear=0)

        observed = aug.augment_images(self.images)

        assert observed[0][1, 1] > 250
        assert (observed[0][[0, 2], [1, 1]] > 20).all()
        assert (observed[0][[0, 2], [1, 1]] < 150).all()
        assert (observed[0][:, 0] < 5).all()
        assert (observed[0][:, 2] < 5).all()

    def test_image_scale_zoom_in_only_y_axis__deterministic(self):
        aug = iaa.Affine(scale={"x": 1.0, "y": 1.75},
                         translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        assert observed[0][1, 1] > 250
        assert (observed[0][[0, 2], [1, 1]] > 20).all()
        assert (observed[0][[0, 2], [1, 1]] < 150).all()
        assert (observed[0][:, 0] < 5).all()
        assert (observed[0][:, 2] < 5).all()

    def test_image_scale_zoom_in_only_y_axis__list(self):
        aug = iaa.Affine(scale={"x": 1.0, "y": 1.75},
                         translate_px=0, rotate=0, shear=0)

        observed = aug.augment_images([self.base_img])

        assert observed[0][1, 1] > 250
        assert (observed[0][[0, 2], [1, 1]] > 20).all()
        assert (observed[0][[0, 2], [1, 1]] < 150).all()
        assert (observed[0][:, 0] < 5).all()
        assert (observed[0][:, 2] < 5).all()

    def test_image_scale_zoom_in_only_y_axis__deterministic_and_list(self):
        aug = iaa.Affine(scale={"x": 1.0, "y": 1.75},
                         translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.base_img])

        assert observed[0][1, 1] > 250
        assert (observed[0][[0, 2], [1, 1]] > 20).all()
        assert (observed[0][[0, 2], [1, 1]] < 150).all()
        assert (observed[0][:, 0] < 5).all()
        assert (observed[0][:, 2] < 5).all()

    def test_keypoints_scale_zoom_in_only_y_axis(self):
        self._test_cba_scale(
            "augment_keypoints", {"y": 1.75, "x": 1.0}, self.kpsoi,
            self.kpsoi_scaled(1.75, 1.0), False)

    def test_keypoints_scale_zoom_in_only_y_axis__deterministic(self):
        self._test_cba_scale(
            "augment_keypoints", {"y": 1.75, "x": 1.0}, self.kpsoi,
            self.kpsoi_scaled(1.75, 1.0), True)

    def test_polygons_scale_zoom_in_only_y_axis(self):
        self._test_cba_scale(
            "augment_polygons", {"y": 1.75, "x": 1.0}, self.psoi,
            self.psoi_scaled(1.75, 1.0), False)

    def test_polygons_scale_zoom_in_only_y_axis__deterministic(self):
        self._test_cba_scale(
            "augment_polygons", {"y": 1.75, "x": 1.0}, self.psoi,
            self.psoi_scaled(1.75, 1.0), True)

    def test_line_strings_scale_zoom_in_only_y_axis(self):
        self._test_cba_scale(
            "augment_polygons", {"y": 1.75, "x": 1.0}, self.psoi,
            self.psoi_scaled(1.75, 1.0), False)

    def test_line_strings_scale_zoom_in_only_y_axis__deterministic(self):
        self._test_cba_scale(
            "augment_line_strings", {"y": 1.75, "x": 1.0}, self.lsoi,
            self.lsoi_scaled(1.75, 1.0), True)

    def test_bounding_boxes_scale_zoom_in_only_y_axis(self):
        self._test_cba_scale(
            "augment_bounding_boxes", {"y": 1.75, "x": 1.0}, self.bbsoi,
            self.bbsoi_scaled(1.75, 1.0), False)

    def test_bounding_boxes_scale_zoom_in_only_y_axis__deterministic(self):
        self._test_cba_scale(
            "augment_bounding_boxes", {"y": 1.75, "x": 1.0}, self.bbsoi,
            self.bbsoi_scaled(1.75, 1.0), True)

    # ---------------------
    # scale: zoom out
    # ---------------------
    # these tests use a 4x4 area of all 255, which is zoomed out to a 4x4 area
    # in which the center 2x2 area is 255
    # zoom in should probably be adapted to this style
    # no separate tests here for x/y axis, should work fine if zoom in works
    # with that

    @property
    def scale_zoom_out_base_img(self):
        return np.ones((4, 4, 1), dtype=np.uint8) * 255

    @property
    def scale_zoom_out_images(self):
        return np.array([self.scale_zoom_out_base_img])

    @property
    def scale_zoom_out_outer_pixels(self):
        outer_pixels = ([], [])
        for y in sm.xrange(4):
            xs = sm.xrange(4) if y in [0, 3] else [0, 3]
            for x in xs:
                outer_pixels[0].append(y)
                outer_pixels[1].append(x)
        return outer_pixels

    @property
    def scale_zoom_out_inner_pixels(self):
        return [1, 1, 2, 2], [1, 2, 1, 2]

    @property
    def scale_zoom_out_kpsoi(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=3, y=0),
               ia.Keypoint(x=0, y=3), ia.Keypoint(x=3, y=3)]
        return [ia.KeypointsOnImage(kps,
                                    shape=self.scale_zoom_out_base_img.shape)]

    @property
    def scale_zoom_out_kpsoi_aug(self):
        kps_aug = [ia.Keypoint(x=0.765, y=0.765),
                   ia.Keypoint(x=2.235, y=0.765),
                   ia.Keypoint(x=0.765, y=2.235),
                   ia.Keypoint(x=2.235, y=2.235)]
        return [ia.KeypointsOnImage(kps_aug,
                                    shape=self.scale_zoom_out_base_img.shape)]

    def test_image_scale_zoom_out(self):
        aug = iaa.Affine(scale=0.49, translate_px=0, rotate=0, shear=0)

        observed = aug.augment_images(self.scale_zoom_out_images)

        outer_pixels = self.scale_zoom_out_outer_pixels
        inner_pixels = self.scale_zoom_out_inner_pixels
        assert (observed[0][outer_pixels] < 25).all()
        assert (observed[0][inner_pixels] > 200).all()

    def test_image_scale_zoom_out__deterministic(self):
        aug = iaa.Affine(scale=0.49, translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.scale_zoom_out_images)

        outer_pixels = self.scale_zoom_out_outer_pixels
        inner_pixels = self.scale_zoom_out_inner_pixels
        assert (observed[0][outer_pixels] < 25).all()
        assert (observed[0][inner_pixels] > 200).all()

    def test_image_scale_zoom_out__list(self):
        aug = iaa.Affine(scale=0.49, translate_px=0, rotate=0, shear=0)

        observed = aug.augment_images([self.scale_zoom_out_base_img])

        outer_pixels = self.scale_zoom_out_outer_pixels
        inner_pixels = self.scale_zoom_out_inner_pixels
        assert (observed[0][outer_pixels] < 25).all()
        assert (observed[0][inner_pixels] > 200).all()

    def test_image_scale_zoom_out__list_and_deterministic(self):
        aug = iaa.Affine(scale=0.49, translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.scale_zoom_out_base_img])

        outer_pixels = self.scale_zoom_out_outer_pixels
        inner_pixels = self.scale_zoom_out_inner_pixels
        assert (observed[0][outer_pixels] < 25).all()
        assert (observed[0][inner_pixels] > 200).all()

    def test_keypoints_scale_zoom_out(self):
        self._test_cba_scale(
            "augment_keypoints", 0.49, self.kpsoi,
            self.kpsoi_scaled(0.49, 0.49), False)

    def test_keypoints_scale_zoom_out__deterministic(self):
        self._test_cba_scale(
            "augment_keypoints", 0.49, self.kpsoi,
            self.kpsoi_scaled(0.49, 0.49), True)

    def test_polygons_scale_zoom_out(self):
        self._test_cba_scale(
            "augment_polygons", 0.49, self.psoi,
            self.psoi_scaled(0.49, 0.49), False)

    def test_polygons_scale_zoom_out__deterministic(self):
        self._test_cba_scale(
            "augment_polygons", 0.49, self.psoi,
            self.psoi_scaled(0.49, 0.49), True)

    def test_line_strings_scale_zoom_out(self):
        self._test_cba_scale(
            "augment_line_strings", 0.49, self.lsoi,
            self.lsoi_scaled(0.49, 0.49), False)

    def test_line_strings_scale_zoom_out__deterministic(self):
        self._test_cba_scale(
            "augment_line_strings", 0.49, self.lsoi,
            self.lsoi_scaled(0.49, 0.49), True)

    def test_bounding_boxes_scale_zoom_out(self):
        self._test_cba_scale(
            "augment_bounding_boxes", 0.49, self.bbsoi,
            self.bbsoi_scaled(0.49, 0.49), False)

    def test_bounding_boxes_scale_zoom_out__deterministic(self):
        self._test_cba_scale(
            "augment_bounding_boxes", 0.49, self.bbsoi,
            self.bbsoi_scaled(0.49, 0.49), True)

    # ---------------------
    # scale: x and y axis are both tuples
    # ---------------------
    def test_image_x_and_y_axis_are_tuples(self):
        aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                         translate_px=0, rotate=0, shear=0)

        image = np.array([[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 2, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]], dtype=np.uint8) * 100
        image = image[:, :, np.newaxis]
        images = np.array([image])

        last_aug = None
        nb_changed_aug = 0
        nb_iterations = 1000
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(images)
            if i == 0:
                last_aug = observed_aug
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                last_aug = observed_aug
        assert nb_changed_aug >= int(nb_iterations * 0.8)

    def test_image_x_and_y_axis_are_tuples__deterministic(self):
        aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                         translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        image = np.array([[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 2, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]], dtype=np.uint8) * 100
        image = image[:, :, np.newaxis]
        images = np.array([image])

        last_aug_det = None
        nb_changed_aug_det = 0
        nb_iterations = 10
        for i in sm.xrange(nb_iterations):
            observed_aug_det = aug_det.augment_images(images)
            if i == 0:
                last_aug_det = observed_aug_det
            else:
                if not np.array_equal(observed_aug_det, last_aug_det):
                    nb_changed_aug_det += 1
                last_aug_det = observed_aug_det
        assert nb_changed_aug_det == 0

    # ------------
    # alignment
    # TODO add alignment tests for: BBs, Polys, LS
    # ------------
    def test_keypoint_alignment(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40-1:40+2, 40-1:40+2] = 255
        image[40-1:40+2, 60-1:60+2] = 255

        kps = [ia.Keypoint(x=40, y=40), ia.Keypoint(x=60, y=40)]
        kpsoi = ia.KeypointsOnImage(kps, shape=image.shape)

        images = [image, image, image]
        kpsois = [kpsoi.deepcopy(),
                  ia.KeypointsOnImage([], shape=image.shape),
                  kpsoi.deepcopy()]

        aug = iaa.Affine(scale=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                1.6, 1.7],
                         order=0)

        for iter in sm.xrange(40):
            images_aug, kpsois_aug = aug(images=images, keypoints=kpsois)

            assert kpsois_aug[1].empty

            for i in [0, 2]:
                image_aug = images_aug[i]
                kpsoi_aug = kpsois_aug[i]

                for kp in kpsoi_aug.keypoints:
                    value = image_aug[int(kp.y), int(kp.x)]
                    assert value > 200

    # ------------
    # make sure that polygons stay valid upon extreme scaling
    # ------------
    def test_polygons_stay_valid_when_using_extreme_scalings(self):
        scales = [1e-4, 1e-2, 1e2, 1e4]
        backends = ["auto", "cv2", "skimage"]
        orders = [0, 1, 3]

        gen = itertools.product(scales, backends, orders)
        for scale, backend, order in gen:
            with self.subTest(scale=scale, backend=backend, order=order):
                aug = iaa.Affine(scale=scale, order=order)
                psoi = ia.PolygonsOnImage([
                    ia.Polygon([(0, 0), (10, 0), (5, 5)])],
                    shape=(10, 10))

                psoi_aug = aug.augment_polygons(psoi)

                poly = psoi_aug.polygons[0]
                ext = poly.exterior
                assert poly.is_valid
                assert ext[0][0] < ext[2][0] < ext[1][0]
                assert ext[0][1] < ext[2][1]
                assert np.allclose(ext[0][1], ext[1][1])


class TestAffine_translate(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        return np.uint8([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])[:, :, np.newaxis]

    @property
    def image_1px_right(self):
        return np.uint8([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])[:, :, np.newaxis]

    @property
    def image_1px_bottom(self):
        return np.uint8([
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])[:, :, np.newaxis]

    @property
    def images(self):
        return np.array([self.image])

    @property
    def images_1px_right(self):
        return np.array([self.image_1px_right])

    @property
    def images_1px_bottom(self):
        return np.array([self.image_1px_bottom])

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=1, y=1)]
        return [ia.KeypointsOnImage(kps, shape=self.image.shape)]

    @property
    def kpsoi_1px_right(self):
        kps = [ia.Keypoint(x=2, y=1)]
        return [ia.KeypointsOnImage(kps, shape=self.image.shape)]

    @property
    def kpsoi_1px_bottom(self):
        kps = [ia.Keypoint(x=1, y=2)]
        return [ia.KeypointsOnImage(kps, shape=self.image.shape)]

    @property
    def psoi(self):
        polys = [ia.Polygon([(0, 0), (2, 0), (2, 2)])]
        return [ia.PolygonsOnImage(polys, shape=self.image.shape)]

    @property
    def psoi_1px_right(self):
        polys = [ia.Polygon([(0+1, 0), (2+1, 0), (2+1, 2)])]
        return [ia.PolygonsOnImage(polys, shape=self.image.shape)]

    @property
    def psoi_1px_bottom(self):
        polys = [ia.Polygon([(0, 0+1), (2, 0+1), (2, 2+1)])]
        return [ia.PolygonsOnImage(polys, shape=self.image.shape)]

    @property
    def lsoi(self):
        ls = [ia.LineString([(0, 0), (2, 0), (2, 2)])]
        return [ia.LineStringsOnImage(ls, shape=self.image.shape)]

    @property
    def lsoi_1px_right(self):
        ls = [ia.LineString([(0+1, 0), (2+1, 0), (2+1, 2)])]
        return [ia.LineStringsOnImage(ls, shape=self.image.shape)]

    @property
    def lsoi_1px_bottom(self):
        ls = [ia.LineString([(0, 0+1), (2, 0+1), (2, 2+1)])]
        return [ia.LineStringsOnImage(ls, shape=self.image.shape)]

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)]
        return [ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)]

    @property
    def bbsoi_1px_right(self):
        bbs = [ia.BoundingBox(x1=0+1, y1=1, x2=2+1, y2=3)]
        return [ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)]

    @property
    def bbsoi_1px_bottom(self):
        bbs = [ia.BoundingBox(x1=0, y1=1+1, x2=2, y2=3+1)]
        return [ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)]

    # ---------------------
    # translate: move one pixel to the right
    # ---------------------
    def test_image_translate_1px_right(self):
        # move one pixel to the right
        aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0,
                         shear=0)

        observed = aug.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_right)

    def test_image_translate_1px_right__deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0,
                         shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_right)

    def test_image_translate_1px_right__list(self):
        aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0,
                         shear=0)

        observed = aug.augment_images([self.image])

        assert array_equal_lists(observed, [self.image_1px_right])

    def test_image_translate_1px_right__list_and_deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0,
                         shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.image])

        assert array_equal_lists(observed, [self.image_1px_right])

    def test_keypoints_translate_1px_right(self):
        self._test_cba_translate_px(
            "augment_keypoints", {"x": 1, "y": 0},
            self.kpsoi, self.kpsoi_1px_right, False)

    def test_keypoints_translate_1px_right__deterministic(self):
        self._test_cba_translate_px(
            "augment_keypoints", {"x": 1, "y": 0},
            self.kpsoi, self.kpsoi_1px_right, True)

    def test_polygons_translate_1px_right(self):
        self._test_cba_translate_px(
            "augment_polygons", {"x": 1, "y": 0},
            self.psoi, self.psoi_1px_right, False)

    def test_polygons_translate_1px_right__deterministic(self):
        self._test_cba_translate_px(
            "augment_polygons", {"x": 1, "y": 0},
            self.psoi, self.psoi_1px_right, True)

    def test_line_strings_translate_1px_right(self):
        self._test_cba_translate_px(
            "augment_line_strings", {"x": 1, "y": 0},
            self.lsoi, self.lsoi_1px_right, False)

    def test_line_strings_translate_1px_right__deterministic(self):
        self._test_cba_translate_px(
            "augment_line_strings", {"x": 1, "y": 0},
            self.lsoi, self.lsoi_1px_right, True)

    def test_bounding_boxes_translate_1px_right(self):
        self._test_cba_translate_px(
            "augment_bounding_boxes", {"x": 1, "y": 0},
            self.bbsoi, self.bbsoi_1px_right, False)

    def test_bounding_boxes_translate_1px_right__deterministic(self):
        self._test_cba_translate_px(
            "augment_bounding_boxes", {"x": 1, "y": 0},
            self.bbsoi, self.bbsoi_1px_right, True)

    @classmethod
    def _test_cba_translate_px(cls, augf_name, px, cbaoi, cbaoi_translated,
                               deterministic):
        aug = iaa.Affine(scale=1.0, translate_px=px, rotate=0, shear=0)
        if deterministic:
            aug = aug.to_deterministic()

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi_translated)

    def test_image_translate_1px_right_skimage(self):
        # move one pixel to the right
        # with backend = skimage
        aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0,
                         shear=0, backend="skimage")

        observed = aug.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_right)

    def test_image_translate_1px_right_skimage_order_all(self):
        # move one pixel to the right
        # with backend = skimage, order=ALL
        aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0,
                         shear=0, backend="skimage", order=ia.ALL)

        observed = aug.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_right)

    def test_image_translate_1px_right_skimage_order_is_list(self):
        # move one pixel to the right
        # with backend = skimage, order=list
        aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0,
                         shear=0, backend="skimage", order=[0, 1, 3])

        observed = aug.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_right)

    def test_image_translate_1px_right_cv2_order_is_list(self):
        # move one pixel to the right
        # with backend = cv2, order=list
        aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0,
                         shear=0, backend="cv2", order=[0, 1, 3])

        observed = aug.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_right)

    def test_image_translate_1px_right_cv2_order_is_stoch_param(self):
        # move one pixel to the right
        # with backend = cv2, order=StochasticParameter
        aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0,
                         shear=0, backend="cv2", order=iap.Choice([0, 1, 3]))

        observed = aug.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_right)

    # ---------------------
    # translate: move one pixel to the bottom
    # ---------------------
    def test_image_translate_1px_bottom(self):
        aug = iaa.Affine(scale=1.0, translate_px={"x": 0, "y": 1}, rotate=0,
                         shear=0)

        observed = aug.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_bottom)

    def test_image_translate_1px_bottom__deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px={"x": 0, "y": 1}, rotate=0,
                         shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_bottom)

    def test_image_translate_1px_bottom__list(self):
        aug = iaa.Affine(scale=1.0, translate_px={"x": 0, "y": 1}, rotate=0,
                         shear=0)

        observed = aug.augment_images([self.image])

        assert array_equal_lists(observed, [self.image_1px_bottom])

    def test_image_translate_1px_bottom__list_and_deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px={"x": 0, "y": 1}, rotate=0,
                         shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.image])

        assert array_equal_lists(observed, [self.image_1px_bottom])

    def test_keypoints_translate_1px_bottom(self):
        self._test_cba_translate_px(
            "augment_keypoints", {"x": 0, "y": 1},
            self.kpsoi, self.kpsoi_1px_bottom, False)

    def test_keypoints_translate_1px_bottom__deterministic(self):
        self._test_cba_translate_px(
            "augment_keypoints", {"x": 0, "y": 1},
            self.kpsoi, self.kpsoi_1px_bottom, True)

    def test_polygons_translate_1px_bottom(self):
        self._test_cba_translate_px(
            "augment_polygons", {"x": 0, "y": 1},
            self.psoi, self.psoi_1px_bottom, False)

    def test_polygons_translate_1px_bottom__deterministic(self):
        self._test_cba_translate_px(
            "augment_polygons", {"x": 0, "y": 1},
            self.psoi, self.psoi_1px_bottom, True)

    def test_line_strings_translate_1px_bottom(self):
        self._test_cba_translate_px(
            "augment_line_strings", {"x": 0, "y": 1},
            self.lsoi, self.lsoi_1px_bottom, False)

    def test_line_strings_translate_1px_bottom__deterministic(self):
        self._test_cba_translate_px(
            "augment_line_strings", {"x": 0, "y": 1},
            self.lsoi, self.lsoi_1px_bottom, True)

    def test_bounding_boxes_translate_1px_bottom(self):
        self._test_cba_translate_px(
            "augment_bounding_boxes", {"x": 0, "y": 1},
            self.bbsoi, self.bbsoi_1px_bottom, False)

    def test_bounding_boxes_translate_1px_bottom__deterministic(self):
        self._test_cba_translate_px(
            "augment_bounding_boxes", {"x": 0, "y": 1},
            self.bbsoi, self.bbsoi_1px_bottom, True)

    # ---------------------
    # translate: fraction of the image size (towards the right)
    # ---------------------
    def test_image_translate_33percent_right(self):
        aug = iaa.Affine(scale=1.0, translate_percent={"x": 0.3333, "y": 0},
                         rotate=0, shear=0)

        observed = aug.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_right)

    def test_image_translate_33percent_right__deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_percent={"x": 0.3333, "y": 0},
                         rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_right)

    def test_image_translate_33percent_right__list(self):
        aug = iaa.Affine(scale=1.0, translate_percent={"x": 0.3333, "y": 0},
                         rotate=0, shear=0)

        observed = aug.augment_images([self.image])

        assert array_equal_lists(observed, [self.image_1px_right])

    def test_image_translate_33percent_right__list_and_deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_percent={"x": 0.3333, "y": 0},
                         rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.image])

        assert array_equal_lists(observed, [self.image_1px_right])

    def test_keypoints_translate_33percent_right(self):
        self._test_cba_translate_percent(
            "augment_keypoints", {"x": 0.3333, "y": 0},
            self.kpsoi, self.kpsoi_1px_right, False)

    def test_keypoints_translate_33percent_right__deterministic(self):
        self._test_cba_translate_percent(
            "augment_keypoints", {"x": 0.3333, "y": 0},
            self.kpsoi, self.kpsoi_1px_right, True)

    def test_polygons_translate_33percent_right(self):
        self._test_cba_translate_percent(
            "augment_polygons", {"x": 0.3333, "y": 0},
            self.psoi, self.psoi_1px_right, False)

    def test_polygons_translate_33percent_right__deterministic(self):
        self._test_cba_translate_percent(
            "augment_polygons", {"x": 0.3333, "y": 0},
            self.psoi, self.psoi_1px_right, True)

    def test_line_strings_translate_33percent_right(self):
        self._test_cba_translate_percent(
            "augment_line_strings", {"x": 0.3333, "y": 0},
            self.lsoi, self.lsoi_1px_right, False)

    def test_line_strings_translate_33percent_right__deterministic(self):
        self._test_cba_translate_percent(
            "augment_line_strings", {"x": 0.3333, "y": 0},
            self.lsoi, self.lsoi_1px_right, True)

    def test_bounding_boxes_translate_33percent_right(self):
        self._test_cba_translate_percent(
            "augment_bounding_boxes", {"x": 0.3333, "y": 0},
            self.bbsoi, self.bbsoi_1px_right, False)

    def test_bounding_boxes_translate_33percent_right__deterministic(self):
        self._test_cba_translate_percent(
            "augment_bounding_boxes", {"x": 0.3333, "y": 0},
            self.bbsoi, self.bbsoi_1px_right, True)

    def test_keypoints_with_continuous_param_results_in_absolute_shift(self):
        # This test ensures that t ~ uniform(a, b) results in a translation
        # by t pixels and not t%
        # see issue #505
        # use iap.Uniform() here to ensure that is really a float value that
        # is sampled and not accidentally DisceteUniform
        aug = iaa.Affine(translate_px=iap.Uniform(10, 20))
        kps = [ia.Keypoint(x=10, y=10)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(1000, 1000))

        for _ in np.arange(5):
            kpsoi_aug = aug.augment_keypoints(kpsoi)

            kp_aug = kpsoi_aug.keypoints[0]
            assert 10+10 <= kp_aug.x <= 10+20
            assert 10+10 <= kp_aug.y <= 10+20

    @classmethod
    def _test_cba_translate_percent(cls, augf_name, percent, cbaoi,
                                    cbaoi_translated, deterministic):
        aug = iaa.Affine(scale=1.0, translate_percent=percent, rotate=0,
                         shear=0)
        if deterministic:
            aug = aug.to_deterministic()

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi_translated, max_distance=1e-3)

    # ---------------------
    # translate: fraction of the image size (towards the bottom)
    # ---------------------
    def test_image_translate_33percent_bottom(self):
        # move 33% (one pixel) to the bottom
        aug = iaa.Affine(scale=1.0, translate_percent={"x": 0, "y": 0.3333},
                         rotate=0, shear=0)

        observed = aug.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_bottom)

    def test_image_translate_33percent_bottom__deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_percent={"x": 0, "y": 0.3333},
                         rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        assert np.array_equal(observed, self.images_1px_bottom)

    def test_image_translate_33percent_bottom__list(self):
        aug = iaa.Affine(scale=1.0, translate_percent={"x": 0, "y": 0.3333},
                         rotate=0, shear=0)

        observed = aug.augment_images([self.image])

        assert array_equal_lists(observed, [self.image_1px_bottom])

    def test_image_translate_33percent_bottom__list_and_deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_percent={"x": 0, "y": 0.3333},
                         rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.image])

        assert array_equal_lists(observed, [self.image_1px_bottom])

    def test_keypoints_translate_33percent_bottom(self):
        self._test_cba_translate_percent(
            "augment_keypoints", {"x": 0, "y": 0.3333},
            self.kpsoi, self.kpsoi_1px_bottom, False)

    def test_keypoints_translate_33percent_bottom__deterministic(self):
        self._test_cba_translate_percent(
            "augment_keypoints", {"x": 0, "y": 0.3333},
            self.kpsoi, self.kpsoi_1px_bottom, True)

    def test_polygons_translate_33percent_bottom(self):
        self._test_cba_translate_percent(
            "augment_polygons", {"x": 0, "y": 0.3333},
            self.psoi, self.psoi_1px_bottom, False)

    def test_polygons_translate_33percent_bottom__deterministic(self):
        self._test_cba_translate_percent(
            "augment_polygons", {"x": 0, "y": 0.3333},
            self.psoi, self.psoi_1px_bottom, True)

    def test_line_strings_translate_33percent_bottom(self):
        self._test_cba_translate_percent(
            "augment_line_strings", {"x": 0, "y": 0.3333},
            self.lsoi, self.lsoi_1px_bottom, False)

    def test_line_strings_translate_33percent_bottom__deterministic(self):
        self._test_cba_translate_percent(
            "augment_line_strings", {"x": 0, "y": 0.3333},
            self.lsoi, self.lsoi_1px_bottom, True)

    def test_bounding_boxes_translate_33percent_bottom(self):
        self._test_cba_translate_percent(
            "augment_bounding_boxes", {"x": 0, "y": 0.3333},
            self.bbsoi, self.bbsoi_1px_bottom, False)

    def test_bounding_boxes_translate_33percent_bottom__deterministic(self):
        self._test_cba_translate_percent(
            "augment_bounding_boxes", {"x": 0, "y": 0.3333},
            self.bbsoi, self.bbsoi_1px_bottom, True)

    # ---------------------
    # translate: axiswise uniform distributions
    # ---------------------
    def test_image_translate_by_axiswise_uniform_distributions(self):
        # 0-1px to left/right and 0-1px to top/bottom
        aug = iaa.Affine(scale=1.0, translate_px={"x": (-1, 1), "y": (-1, 1)},
                         rotate=0, shear=0)
        last_aug = None
        nb_changed_aug = 0
        nb_iterations = 1000
        centers_aug = self.image.astype(np.int32) * 0
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(self.images)
            if i == 0:
                last_aug = observed_aug
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                last_aug = observed_aug

            assert len(observed_aug[0].nonzero()[0]) == 1
            centers_aug += (observed_aug[0] > 0)

        assert nb_changed_aug >= int(nb_iterations * 0.7)
        assert (centers_aug > int(nb_iterations * (1/9 * 0.6))).all()
        assert (centers_aug < int(nb_iterations * (1/9 * 1.4))).all()

    def test_image_translate_by_axiswise_uniform_distributions__det(self):
        # 0-1px to left/right and 0-1px to top/bottom
        aug = iaa.Affine(scale=1.0, translate_px={"x": (-1, 1), "y": (-1, 1)},
                         rotate=0, shear=0)
        aug_det = aug.to_deterministic()
        last_aug_det = None
        nb_changed_aug_det = 0
        nb_iterations = 10
        centers_aug_det = self.image.astype(np.int32) * 0
        for i in sm.xrange(nb_iterations):
            observed_aug_det = aug_det.augment_images(self.images)
            if i == 0:
                last_aug_det = observed_aug_det
            else:
                if not np.array_equal(observed_aug_det, last_aug_det):
                    nb_changed_aug_det += 1
                last_aug_det = observed_aug_det

            assert len(observed_aug_det[0].nonzero()[0]) == 1
            centers_aug_det += (observed_aug_det[0] > 0)

        assert nb_changed_aug_det == 0

    # ---------------------
    # translate heatmaps
    # ---------------------
    @property
    def heatmaps(self):
        return ia.HeatmapsOnImage(
            np.float32([
                [0.0, 0.5, 0.75],
                [0.0, 0.5, 0.75],
                [0.75, 0.75, 0.75],
            ]),
            shape=(3, 3, 3)
        )

    @property
    def heatmaps_1px_right(self):
        return ia.HeatmapsOnImage(
            np.float32([
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5],
                [0.0, 0.75, 0.75],
            ]),
            shape=(3, 3, 3)
        )

    def test_heatmaps_translate_1px_right(self):
        aug = iaa.Affine(translate_px={"x": 1})

        observed = aug.augment_heatmaps([self.heatmaps])[0]

        _assert_same_shape(observed, self.heatmaps)
        _assert_same_min_max(observed, self.heatmaps)
        assert np.array_equal(observed.get_arr(),
                              self.heatmaps_1px_right.get_arr())

    def test_heatmaps_translate_1px_right_should_ignore_cval(self):
        # should still use mode=constant cval=0 even when other settings chosen
        aug = iaa.Affine(translate_px={"x": 1}, cval=255)

        observed = aug.augment_heatmaps([self.heatmaps])[0]

        _assert_same_shape(observed, self.heatmaps)
        _assert_same_min_max(observed, self.heatmaps)
        assert np.array_equal(observed.get_arr(),
                              self.heatmaps_1px_right.get_arr())

    def test_heatmaps_translate_1px_right_should_ignore_mode(self):
        aug = iaa.Affine(translate_px={"x": 1}, mode="edge", cval=255)

        observed = aug.augment_heatmaps([self.heatmaps])[0]

        _assert_same_shape(observed, self.heatmaps)
        _assert_same_min_max(observed, self.heatmaps)
        assert np.array_equal(observed.get_arr(),
                              self.heatmaps_1px_right.get_arr())

    # ---------------------
    # translate segmaps
    # ---------------------
    @property
    def segmaps(self):
        return SegmentationMapsOnImage(
            np.int32([
                [0, 1, 2],
                [0, 1, 2],
                [2, 2, 2],
            ]),
            shape=(3, 3, 3)
        )

    @property
    def segmaps_1px_right(self):
        return SegmentationMapsOnImage(
            np.int32([
                [0, 0, 1],
                [0, 0, 1],
                [0, 2, 2],
            ]),
            shape=(3, 3, 3)
        )

    def test_segmaps_translate_1px_right(self):
        aug = iaa.Affine(translate_px={"x": 1})

        observed = aug.augment_segmentation_maps([self.segmaps])[0]

        _assert_same_shape(observed, self.segmaps)
        assert np.array_equal(observed.get_arr(),
                              self.segmaps_1px_right.get_arr())

    def test_segmaps_translate_1px_right_should_ignore_cval(self):
        # should still use mode=constant cval=0 even when other settings chosen
        aug = iaa.Affine(translate_px={"x": 1}, cval=255)

        observed = aug.augment_segmentation_maps([self.segmaps])[0]

        _assert_same_shape(observed, self.segmaps)
        assert np.array_equal(observed.get_arr(),
                              self.segmaps_1px_right.get_arr())

    def test_segmaps_translate_1px_right_should_ignore_mode(self):
        aug = iaa.Affine(translate_px={"x": 1}, mode="edge", cval=255)

        observed = aug.augment_segmentation_maps([self.segmaps])[0]

        _assert_same_shape(observed, self.segmaps)
        assert np.array_equal(observed.get_arr(),
                              self.segmaps_1px_right.get_arr())


class TestAffine_rotate(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        return np.uint8([
            [0, 0, 0],
            [255, 255, 255],
            [0, 0, 0]
        ])[:, :, np.newaxis]

    @property
    def image_rot90(self):
        return np.uint8([
            [0, 255, 0],
            [0, 255, 0],
            [0, 255, 0]
        ])[:, :, np.newaxis]

    @property
    def images(self):
        return np.array([self.image])

    @property
    def images_rot90(self):
        return np.array([self.image_rot90])

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=1)]
        return [ia.KeypointsOnImage(kps, shape=self.image.shape)]

    @property
    def kpsoi_rot90(self):
        kps = [ia.Keypoint(x=3-1, y=0), ia.Keypoint(x=3-1, y=1),
               ia.Keypoint(x=3-1, y=2)]
        return [ia.KeypointsOnImage(kps, shape=self.image_rot90.shape)]

    @property
    def psoi(self):
        polys = [ia.Polygon([(0, 0), (3, 0), (3, 3)])]
        return [ia.PolygonsOnImage(polys, shape=self.image.shape)]

    @property
    def psoi_rot90(self):
        polys = [ia.Polygon([(3-0, 0), (3-0, 3), (3-3, 3)])]
        return [ia.PolygonsOnImage(polys, shape=self.image_rot90.shape)]

    @property
    def lsoi(self):
        ls = [ia.LineString([(0, 0), (3, 0), (3, 3)])]
        return [ia.LineStringsOnImage(ls, shape=self.image.shape)]

    @property
    def lsoi_rot90(self):
        ls = [ia.LineString([(3-0, 0), (3-0, 3), (3-3, 3)])]
        return [ia.LineStringsOnImage(ls, shape=self.image_rot90.shape)]

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)]
        return [ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)]

    @property
    def bbsoi_rot90(self):
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=2, y2=2)]
        return [ia.BoundingBoxesOnImage(bbs, shape=self.image_rot90.shape)]

    def test_image_rot90(self):
        # rotate by 90 degrees
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=90, shear=0)

        observed = aug.augment_images(self.images)

        observed[observed >= 100] = 255
        observed[observed < 100] = 0
        assert np.array_equal(observed, self.images_rot90)

    def test_image_rot90__deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=90, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        observed[observed >= 100] = 255
        observed[observed < 100] = 0
        assert np.array_equal(observed, self.images_rot90)

    def test_image_rot90__list(self):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=90, shear=0)

        observed = aug.augment_images([self.image])

        observed[0][observed[0] >= 100] = 255
        observed[0][observed[0] < 100] = 0
        assert array_equal_lists(observed, [self.image_rot90])

    def test_image_rot90__list_and_deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=90, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.image])

        observed[0][observed[0] >= 100] = 255
        observed[0][observed[0] < 100] = 0
        assert array_equal_lists(observed, [self.image_rot90])

    def test_keypoints_rot90(self):
        self._test_cba_rotate(
            "augment_keypoints", 90, self.kpsoi, self.kpsoi_rot90, False)

    def test_keypoints_rot90__deterministic(self):
        self._test_cba_rotate(
            "augment_keypoints", 90, self.kpsoi, self.kpsoi_rot90, True)

    def test_polygons_rot90(self):
        self._test_cba_rotate(
            "augment_polygons", 90, self.psoi, self.psoi_rot90, False)

    def test_polygons_rot90__deterministic(self):
        self._test_cba_rotate(
            "augment_polygons", 90, self.psoi, self.psoi_rot90, True)

    def test_line_strings_rot90(self):
        self._test_cba_rotate(
            "augment_line_strings", 90, self.lsoi, self.lsoi_rot90, False)

    def test_line_strings_rot90__deterministic(self):
        self._test_cba_rotate(
            "augment_line_strings", 90, self.lsoi, self.lsoi_rot90, True)

    def test_bounding_boxes_rot90(self):
        self._test_cba_rotate(
            "augment_bounding_boxes", 90, self.bbsoi, self.bbsoi_rot90, False)

    def test_bounding_boxes_rot90__deterministic(self):
        self._test_cba_rotate(
            "augment_bounding_boxes", 90, self.bbsoi, self.bbsoi_rot90, True)

    @classmethod
    def _test_cba_rotate(cls, augf_name, rotate, cbaoi,
                         cbaoi_rotated, deterministic):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=rotate,
                         shear=0)
        if deterministic:
            aug = aug.to_deterministic()

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi_rotated)

    def test_image_rotate_is_tuple_0_to_364_deg(self):
        # random rotation 0-364 degrees
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=(0, 364), shear=0)
        last_aug = None
        nb_changed_aug = 0
        nb_iterations = 1000
        pixels_sums_aug = self.image.astype(np.int32) * 0
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(self.images)
            if i == 0:
                last_aug = observed_aug
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                last_aug = observed_aug

            pixels_sums_aug += (observed_aug[0] > 100)

        assert nb_changed_aug >= int(nb_iterations * 0.9)
        # center pixel, should always be white when rotating line around center
        assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
        assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)

        # outer pixels, should sometimes be white
        # the values here had to be set quite tolerant, the middle pixels at
        # top/left/bottom/right get more activation than expected
        outer_pixels = ([0, 0, 0, 1, 1, 2, 2, 2],
                        [0, 1, 2, 0, 2, 0, 1, 2])
        assert (
            pixels_sums_aug[outer_pixels] > int(nb_iterations * (2/8 * 0.4))
        ).all()
        assert (
            pixels_sums_aug[outer_pixels] < int(nb_iterations * (2/8 * 2.0))
        ).all()

    def test_image_rotate_is_tuple_0_to_364_deg__deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=(0, 364), shear=0)
        aug_det = aug.to_deterministic()
        last_aug_det = None
        nb_changed_aug_det = 0
        nb_iterations = 10
        pixels_sums_aug_det = self.image.astype(np.int32) * 0
        for i in sm.xrange(nb_iterations):
            observed_aug_det = aug_det.augment_images(self.images)
            if i == 0:
                last_aug_det = observed_aug_det
            else:
                if not np.array_equal(observed_aug_det, last_aug_det):
                    nb_changed_aug_det += 1
                last_aug_det = observed_aug_det

            pixels_sums_aug_det += (observed_aug_det[0] > 100)

        assert nb_changed_aug_det == 0
        # center pixel, should always be white when rotating line around center
        assert pixels_sums_aug_det[1, 1] > (nb_iterations * 0.98)
        assert pixels_sums_aug_det[1, 1] < (nb_iterations * 1.02)

    def test_alignment_between_images_and_heatmaps_for_fixed_rot(self):
        # measure alignment between images and heatmaps when rotating
        for backend in ["auto", "cv2", "skimage"]:
            aug = iaa.Affine(rotate=45, backend=backend)
            image = np.zeros((7, 6), dtype=np.uint8)
            image[:, 2:3+1] = 255
            hm = ia.HeatmapsOnImage(image.astype(np.float32)/255, shape=(7, 6))

            img_aug = aug.augment_image(image)
            hm_aug = aug.augment_heatmaps([hm])[0]

            img_aug_mask = img_aug > 255*0.1
            hm_aug_mask = hm_aug.arr_0to1 > 0.1
            same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
            assert hm_aug.shape == (7, 6)
            assert hm_aug.arr_0to1.shape == (7, 6, 1)
            assert (same / img_aug_mask.size) >= 0.95

    def test_alignment_between_images_and_smaller_heatmaps_for_fixed_rot(self):
        # measure alignment between images and heatmaps when rotating
        # here with smaller heatmaps
        for backend in ["auto", "cv2", "skimage"]:
            aug = iaa.Affine(rotate=45, backend=backend)

            image = np.zeros((56, 48), dtype=np.uint8)
            image[:, 16:24+1] = 255
            hm = ia.HeatmapsOnImage(
                ia.imresize_single_image(
                    image, (28, 24), interpolation="cubic"
                ).astype(np.float32)/255,
                shape=(56, 48)
            )

            img_aug = aug.augment_image(image)
            hm_aug = aug.augment_heatmaps([hm])[0]

            img_aug_mask = img_aug > 255*0.1
            hm_aug_mask = ia.imresize_single_image(
                hm_aug.arr_0to1, img_aug.shape[0:2], interpolation="cubic"
            ) > 0.1
            same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
            assert hm_aug.shape == (56, 48)
            assert hm_aug.arr_0to1.shape == (28, 24, 1)
            assert (same / img_aug_mask.size) >= 0.9

    def test_bounding_boxes_have_expected_shape_after_augmentation(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        image[20:80, 20:80] = 255
        bb = ia.BoundingBox(x1=20, y1=20, x2=80, y2=80)
        bbsoi = ia.BoundingBoxesOnImage([bb], shape=image.shape)
        for rotate in [10, 20, 40, 80, 120]:
            with self.subTest(rotate=rotate):
                aug = iaa.Affine(rotate=rotate, order=0)

                image_aug, bbsoi_aug = aug(image=image, bounding_boxes=bbsoi)

                xx = np.nonzero(np.max(image_aug > 100, axis=0))[0]
                yy = np.nonzero(np.max(image_aug > 100, axis=1))[0]
                bb_exp_x1 = xx[0]
                bb_exp_x2 = xx[-1]
                bb_exp_y1 = yy[0]
                bb_exp_y2 = yy[-1]
                bb_expected = ia.BoundingBox(x1=bb_exp_x1, y1=bb_exp_y1,
                                             x2=bb_exp_x2, y2=bb_exp_y2)
                assert bbsoi_aug.bounding_boxes[0].iou(bb_expected) > 0.95


class TestAffine_cval(unittest.TestCase):
    @property
    def image(self):
        return np.ones((3, 3, 1), dtype=np.uint8) * 255

    @property
    def images(self):
        return np.array([self.image])

    def test_image_fixed_cval(self):
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=128)

        observed = aug.augment_images(self.images)

        assert (observed[0] > 128 - 30).all()
        assert (observed[0] < 128 + 30).all()

    def test_image_fixed_cval__deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=128)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images(self.images)

        assert (observed[0] > 128 - 30).all()
        assert (observed[0] < 128 + 30).all()

    def test_image_fixed_cval__list(self):
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=128)

        observed = aug.augment_images([self.image])

        assert (observed[0] > 128 - 30).all()
        assert (observed[0] < 128 + 30).all()

    def test_image_fixed_cval__list_and_deterministic(self):
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=128)
        aug_det = aug.to_deterministic()

        observed = aug_det.augment_images([self.image])

        assert (observed[0] > 128 - 30).all()
        assert (observed[0] < 128 + 30).all()

    def test_image_cval_is_tuple(self):
        # random cvals
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=(0, 255))
        last_aug = None
        nb_changed_aug = 0
        nb_iterations = 1000
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(self.images)

            if i == 0:
                last_aug = observed_aug
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                last_aug = observed_aug

        assert nb_changed_aug >= int(nb_iterations * 0.9)

    def test_image_cval_is_tuple__deterministic(self):
        # random cvals
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0,
                         cval=(0, 255))
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


class TestAffine_fit_output(unittest.TestCase):
    @property
    def image(self):
        return np.ones((3, 3, 1), dtype=np.uint8) * 255

    @property
    def images(self):
        return np.array([self.image])

    @property
    def heatmaps(self):
        return ia.HeatmapsOnImage(
            np.float32([
                [0.0, 0.5, 0.75],
                [0.0, 0.5, 0.75],
                [0.75, 0.75, 0.75],
            ]),
            shape=(3, 3, 3)
        )

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=1)]
        return [ia.KeypointsOnImage(kps, shape=self.image.shape)]

    def test_image_translate(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(translate_px=100, fit_output=True,
                                 backend=backend)

                observed = aug.augment_images(self.images)

                expected = self.images
                assert np.array_equal(observed, expected)

    def test_keypoints_translate(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(translate_px=100, fit_output=True,
                                 backend=backend)

                observed = aug.augment_keypoints(self.kpsoi)

                expected = self.kpsoi
                assert keypoints_equal(observed, expected)

    def test_heatmaps_translate(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(translate_px=100, fit_output=True,
                                 backend=backend)

                observed = aug.augment_heatmaps([self.heatmaps])[0]

                expected = self.heatmaps
                assert np.allclose(observed.arr_0to1, expected.arr_0to1)

    def test_image_rot45(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(rotate=45, fit_output=True,
                                 backend=backend)
                img = np.zeros((10, 10), dtype=np.uint8)
                img[0:2, 0:2] = 255
                img[-2:, 0:2] = 255
                img[0:2, -2:] = 255
                img[-2:, -2:] = 255

                img_aug = aug.augment_image(img)

                _labels, nb_labels = skimage.morphology.label(
                    img_aug > 240, return_num=True, connectivity=2)
                assert nb_labels == 4

    def test_heatmaps_rot45(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(rotate=45, fit_output=True,
                                 backend=backend)
                img = np.zeros((10, 10), dtype=np.uint8)
                img[0:2, 0:2] = 255
                img[-2:, 0:2] = 255
                img[0:2, -2:] = 255
                img[-2:, -2:] = 255
                hm = ia.HeatmapsOnImage(img.astype(np.float32)/255,
                                        shape=(10, 10))

                hm_aug = aug.augment_heatmaps([hm])[0]

                _labels, nb_labels = skimage.morphology.label(
                    hm_aug.arr_0to1 > 240/255, return_num=True, connectivity=2)
                assert nb_labels == 4

    def test_heatmaps_rot45__heatmaps_smaller_than_image(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(rotate=45, fit_output=True,
                                 backend=backend)
                img = np.zeros((80, 80), dtype=np.uint8)
                img[0:5, 0:5] = 255
                img[-5:, 0:5] = 255
                img[0:5, -5:] = 255
                img[-5:, -5:] = 255
                hm = HeatmapsOnImage(
                    ia.imresize_single_image(
                        img, (40, 40), interpolation="cubic"
                    ).astype(np.float32)/255,
                    shape=(80, 80)
                )

                hm_aug = aug.augment_heatmaps([hm])[0]

                # these asserts are deactivated because the image size can
                # change under fit_output=True
                # assert hm_aug.shape == (80, 80)
                # assert hm_aug.arr_0to1.shape == (40, 40, 1)
                _labels, nb_labels = skimage.morphology.label(
                    hm_aug.arr_0to1 > 200/255, return_num=True, connectivity=2)
                assert nb_labels == 4

    def test_image_heatmap_alignment_random_rots(self):
        nb_iterations = 50
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                for _ in sm.xrange(nb_iterations):
                    aug = iaa.Affine(rotate=(0, 364), fit_output=True,
                                     backend=backend)
                    img = np.zeros((80, 80), dtype=np.uint8)
                    img[0:5, 0:5] = 255
                    img[-5:, 0:5] = 255
                    img[0:5, -5:] = 255
                    img[-5:, -5:] = 255
                    hm = HeatmapsOnImage(
                        img.astype(np.float32)/255,
                        shape=(80, 80)
                    )

                    img_aug = aug.augment_image(img)
                    hm_aug = aug.augment_heatmaps([hm])[0]

                    img_aug_mask = img_aug > 255*0.1
                    hm_aug_mask = ia.imresize_single_image(
                        hm_aug.arr_0to1, img_aug.shape[0:2],
                        interpolation="cubic"
                    ) > 0.1
                    same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
                    assert (same / img_aug_mask.size) >= 0.95

    def test_image_heatmap_alignment_random_rots__hms_smaller_than_img(self):
        nb_iterations = 50
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                for _ in sm.xrange(nb_iterations):
                    aug = iaa.Affine(rotate=(0, 364), fit_output=True,
                                     backend=backend)
                    img = np.zeros((80, 80), dtype=np.uint8)
                    img[0:5, 0:5] = 255
                    img[-5:, 0:5] = 255
                    img[0:5, -5:] = 255
                    img[-5:, -5:] = 255
                    hm = HeatmapsOnImage(
                        ia.imresize_single_image(
                            img, (40, 40), interpolation="cubic"
                        ).astype(np.float32)/255,
                        shape=(80, 80)
                    )

                    img_aug = aug.augment_image(img)
                    hm_aug = aug.augment_heatmaps([hm])[0]

                    img_aug_mask = img_aug > 255*0.1
                    hm_aug_mask = ia.imresize_single_image(
                        hm_aug.arr_0to1, img_aug.shape[0:2],
                        interpolation="cubic"
                    ) > 0.1
                    same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
                    assert (same / img_aug_mask.size) >= 0.95

    def test_segmaps_rot45(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(rotate=45, fit_output=True,
                                 backend=backend)
                img = np.zeros((80, 80), dtype=np.uint8)
                img[0:5, 0:5] = 255
                img[-5:, 0:5] = 255
                img[0:5, -5:] = 255
                img[-5:, -5:] = 255
                segmap = SegmentationMapsOnImage(
                    (img > 100).astype(np.int32),
                    shape=(80, 80)
                )

                segmap_aug = aug.augment_segmentation_maps([segmap])[0]

                # these asserts are deactivated because the image size can
                # change under fit_output=True
                # assert segmap_aug.shape == (80, 80)
                # assert segmap_aug.arr_0to1.shape == (40, 40, 1)
                _labels, nb_labels = skimage.morphology.label(
                    segmap_aug.arr > 0, return_num=True, connectivity=2)
                assert nb_labels == 4

    def test_segmaps_rot45__segmaps_smaller_than_img(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(rotate=45, fit_output=True,
                                 backend=backend)
                img = np.zeros((80, 80), dtype=np.uint8)
                img[0:5, 0:5] = 255
                img[-5:, 0:5] = 255
                img[0:5, -5:] = 255
                img[-5:, -5:] = 255
                segmap = SegmentationMapsOnImage(
                    (
                        ia.imresize_single_image(
                            img, (40, 40), interpolation="cubic"
                        ) > 100
                     ).astype(np.int32),
                    shape=(80, 80)
                )

                segmap_aug = aug.augment_segmentation_maps([segmap])[0]

                # these asserts are deactivated because the image size can
                # change under fit_output=True
                # assert segmap_aug.shape == (80, 80)
                # assert segmap_aug.arr_0to1.shape == (40, 40, 1)
                _labels, nb_labels = skimage.morphology.label(
                    segmap_aug.arr > 0, return_num=True, connectivity=2)
                assert nb_labels == 4

    def test_image_segmap_alignment_random_rots(self):
        nb_iterations = 50
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                for _ in sm.xrange(nb_iterations):
                    aug = iaa.Affine(rotate=(0, 364), fit_output=True,
                                     backend=backend)
                    img = np.zeros((80, 80), dtype=np.uint8)
                    img[0:5, 0:5] = 255
                    img[-5:, 0:5] = 255
                    img[0:5, -5:] = 255
                    img[-5:, -5:] = 255
                    segmap = SegmentationMapsOnImage(
                        (img > 100).astype(np.int32),
                        shape=(80, 80)
                    )

                    img_aug = aug.augment_image(img)
                    segmap_aug = aug.augment_segmentation_maps([segmap])[0]

                    img_aug_mask = img_aug > 100
                    segmap_aug_mask = ia.imresize_single_image(
                        segmap_aug.arr,
                        img_aug.shape[0:2],
                        interpolation="nearest"
                    ) > 0
                    same = np.sum(img_aug_mask == segmap_aug_mask[:, :, 0])
                    assert (same / img_aug_mask.size) >= 0.95

    def test_image_segmap_alignment_random_rots__sms_smaller_than_img(self):
        nb_iterations = 50
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                for _ in sm.xrange(nb_iterations):
                    aug = iaa.Affine(rotate=(0, 364), fit_output=True,
                                     backend=backend)
                    img = np.zeros((80, 80), dtype=np.uint8)
                    img[0:5, 0:5] = 255
                    img[-5:, 0:5] = 255
                    img[0:5, -5:] = 255
                    img[-5:, -5:] = 255
                    segmap = SegmentationMapsOnImage(
                        (
                            ia.imresize_single_image(
                                img, (40, 40), interpolation="cubic"
                            ) > 100
                         ).astype(np.int32),
                        shape=(80, 80)
                    )

                    img_aug = aug.augment_image(img)
                    segmap_aug = aug.augment_segmentation_maps([segmap])[0]

                    img_aug_mask = img_aug > 100
                    segmap_aug_mask = ia.imresize_single_image(
                        segmap_aug.arr,
                        img_aug.shape[0:2],
                        interpolation="nearest"
                    ) > 0
                    same = np.sum(img_aug_mask == segmap_aug_mask[:, :, 0])
                    assert (same / img_aug_mask.size) >= 0.95

    def test_keypoints_rot90_without_fit_output(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(rotate=90, backend=backend)
                kps = ia.KeypointsOnImage([ia.Keypoint(10, 10)],
                                          shape=(100, 200, 3))
                kps_aug = aug.augment_keypoints(kps)
                assert kps_aug.shape == (100, 200, 3)
                assert not np.allclose(
                    [kps_aug.keypoints[0].x, kps_aug.keypoints[0].y],
                    [kps.keypoints[0].x, kps.keypoints[0].y],
                    atol=1e-2, rtol=0)

    def test_keypoints_rot90(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(rotate=90, fit_output=True, backend=backend)
                kps = ia.KeypointsOnImage([ia.Keypoint(10, 10)],
                                          shape=(100, 200, 3))

                kps_aug = aug.augment_keypoints(kps)

                assert kps_aug.shape == (200, 100, 3)
                assert not np.allclose(
                    [kps_aug.keypoints[0].x, kps_aug.keypoints[0].y],
                    [kps.keypoints[0].x, kps.keypoints[0].y],
                    atol=1e-2, rtol=0)

    def test_empty_keypoints_rot90(self):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(rotate=90, fit_output=True, backend=backend)
                kps = ia.KeypointsOnImage([], shape=(100, 200, 3))

                kps_aug = aug.augment_keypoints(kps)

                assert kps_aug.shape == (200, 100, 3)
                assert len(kps_aug.keypoints) == 0

    def _test_cbaoi_rot90_without_fit_output(self, cbaoi, augf_name):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                # verify that shape in PolygonsOnImages changes
                aug = iaa.Affine(rotate=90, backend=backend)

                cbaoi_aug = getattr(aug, augf_name)([cbaoi, cbaoi])

                assert len(cbaoi_aug) == 2
                for cbaoi_aug_i in cbaoi_aug:
                    if isinstance(cbaoi, (ia.PolygonsOnImage,
                                          ia.LineStringsOnImage)):
                        assert cbaoi_aug_i.shape == cbaoi.shape
                        assert not cbaoi_aug_i.items[0].coords_almost_equals(
                            cbaoi.items[0].coords, max_distance=1e-2)
                    else:
                        assert_cbaois_equal(cbaoi_aug_i, cbaoi)

    def test_polygons_rot90_without_fit_output(self):
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(10, 10), (20, 10), (20, 20)])
        ], shape=(100, 200, 3))

        self._test_cbaoi_rot90_without_fit_output(psoi, "augment_polygons")

    def test_line_strings_rot90_without_fit_output(self):
        lsoi = ia.LineStringsOnImage([
            ia.LineString([(10, 10), (20, 10), (20, 20), (10, 10)])
        ], shape=(100, 200, 3))

        self._test_cbaoi_rot90_without_fit_output(lsoi, "augment_line_strings")

    def _test_cbaoi_rot90(self, cbaoi, expected, augf_name):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(rotate=90, fit_output=True, backend=backend)

                cbaoi_aug = getattr(aug, augf_name)([cbaoi, cbaoi])

                assert len(cbaoi_aug) == 2
                for cbaoi_aug_i in cbaoi_aug:
                    assert_cbaois_equal(cbaoi_aug_i, expected)

    def test_polygons_rot90(self):
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(10, 10), (20, 10), (20, 20)])
        ], shape=(100, 200, 3))
        expected = ia.PolygonsOnImage([
            ia.Polygon([(100-10-1, 10), (100-10-1, 20), (100-20-1, 20)])
        ], shape=(200, 100, 3))
        self._test_cbaoi_rot90(psoi, expected, "augment_polygons")

    def test_line_strings_rot90(self):
        lsoi = ia.LineStringsOnImage([
            ia.LineString([(10, 10), (20, 10), (20, 20), (10, 10)])
        ], shape=(100, 200, 3))
        expected = ia.LineStringsOnImage([
            ia.LineString([(100-10-1, 10), (100-10-1, 20), (100-20-1, 20),
                           (100-10-1, 10)])
        ], shape=(200, 100, 3))
        self._test_cbaoi_rot90(lsoi, expected, "augment_line_strings")

    def test_bounding_boxes_rot90(self):
        lsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=10, y1=10, x2=20, y2=20)
        ], shape=(100, 200, 3))
        expected = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=100-20-1, y1=10, x2=100-10-1, y2=20)
        ], shape=(200, 100, 3))
        self._test_cbaoi_rot90(lsoi, expected, "augment_bounding_boxes")

    def _test_empty_cbaoi_rot90(self, cbaoi, expected, augf_name):
        for backend in ["auto", "cv2", "skimage"]:
            with self.subTest(backend=backend):
                aug = iaa.Affine(rotate=90, fit_output=True, backend=backend)

                cbaoi_aug = getattr(aug, augf_name)(cbaoi)

                assert_cbaois_equal(cbaoi_aug, expected)

    def test_empty_polygons_rot90(self):
        psoi = ia.PolygonsOnImage([], shape=(100, 200, 3))
        expected = ia.PolygonsOnImage([], shape=(200, 100, 3))
        self._test_empty_cbaoi_rot90(psoi, expected, "augment_polygons")

    def test_empty_line_strings_rot90(self):
        lsoi = ia.LineStringsOnImage([], shape=(100, 200, 3))
        expected = ia.LineStringsOnImage([], shape=(200, 100, 3))
        self._test_empty_cbaoi_rot90(lsoi, expected, "augment_line_strings")

    def test_empty_bounding_boxes_rot90(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(100, 200, 3))
        expected = ia.BoundingBoxesOnImage([], shape=(200, 100, 3))
        self._test_empty_cbaoi_rot90(bbsoi, expected, "augment_bounding_boxes")


# TODO merge these into TestAffine_rotate since they are rotations?
#      or extend to contain other affine params too?
class TestAffine_alignment(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_image_segmap_alignment_with_translate_px(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[40-10:40+10, 50-10:50+10, :] = 255
        hm = np.zeros((40, 50, 1), dtype=np.float32)
        hm[20-5:20+5, 25-5:25+5, 0] = 1.0
        hm = ia.HeatmapsOnImage(hm, shape=image.shape)

        # note that if x is an odd value (e.g. 1), the projection is a bit
        # less accurate as x=1 projected to a half-sized segmap is x=0.5,
        # leading to interpolation effects
        xvals = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, [0, 10, 20]]

        for xvals_i in xvals:
            with self.subTest(x=xvals_i):
                aug = iaa.Affine(translate_px={"x": xvals_i})
                iterations = 2 if ia.is_single_number(xvals_i) else 20

                for _ in np.arange(iterations):
                    image_aug, hm_aug = aug(image=image, heatmaps=hm)

                    hm_aug_arr_rs = ia.imresize_single_image(
                        hm_aug.get_arr(), (80, 100), interpolation="nearest")
                    overlap_true = np.sum(
                        np.logical_and(
                            (image_aug[..., 0] > 220),
                            (hm_aug_arr_rs[..., 0] > 0.9)
                        )
                    )
                    p_same_on_zero_cells = np.average(
                        (image_aug[..., 0] > 220)
                        == (hm_aug_arr_rs[..., 0] > 0.9))
                    assert overlap_true > 19*19
                    assert p_same_on_zero_cells > 0.98

    def test_image_segmap_alignment_with_translate_percent(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[40-10:40+10, 50-10:50+10, :] = 255
        hm = np.zeros((40, 50, 1), dtype=np.float32)
        hm[20-5:20+5, 25-5:25+5, 0] = 1.0
        hm = ia.HeatmapsOnImage(hm, shape=image.shape)

        # note that if x is an odd value (e.g. 1), the projection is a bit
        # less accurate as x=1 projected to a half-sized segmap is x=0.5,
        # leading to interpolation effects
        width = image.shape[1]
        xvals = [0/width, 2/width, 4/width, 6/width, 8/width, 10/width,
                 12/width, 14/width, 16/width, 18/width, 20/width,
                 [0/width, 10/width, 20/width]]

        for xvals_i in xvals:
            with self.subTest(x=xvals_i):
                aug = iaa.Affine(translate_percent={"x": xvals_i})
                iterations = 2 if ia.is_single_number(xvals_i) else 20

                for _ in np.arange(iterations):
                    image_aug, hm_aug = aug(image=image, heatmaps=hm)

                    hm_aug_arr_rs = ia.imresize_single_image(
                        hm_aug.get_arr(), (80, 100), interpolation="nearest")
                    overlap_true = np.sum(
                        np.logical_and(
                            (image_aug[..., 0] > 220),
                            (hm_aug_arr_rs[..., 0] > 0.9)
                        )
                    )
                    p_same_on_zero_cells = np.average(
                        (image_aug[..., 0] > 220)
                        == (hm_aug_arr_rs[..., 0] > 0.9))
                    assert overlap_true > 19*19
                    assert p_same_on_zero_cells > 0.98

    def test_image_keypoint_alignment(self):
        aug = iaa.Affine(rotate=[0, 180], order=0)
        img = np.zeros((10, 10), dtype=np.uint8)
        img[0:5, 5] = 255
        img[2, 4:6] = 255
        img_rot = [np.copy(img), np.copy(np.flipud(np.fliplr(img)))]
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=5, y=2)], shape=img.shape)
        kpsoi_rot = [(5, 2), (5, 10-2)]
        img_aug_indices = []
        kpsois_aug_indices = []
        for _ in sm.xrange(40):
            aug_det = aug.to_deterministic()
            imgs_aug = aug_det.augment_images([img, img])
            kpsois_aug = aug_det.augment_keypoints([kpsoi, kpsoi])

            assert kpsois_aug[0].shape == img.shape
            assert kpsois_aug[1].shape == img.shape

            for img_aug in imgs_aug:
                if np.array_equal(img_aug, img_rot[0]):
                    img_aug_indices.append(0)
                elif np.array_equal(img_aug, img_rot[1]):
                    img_aug_indices.append(1)
                else:
                    assert False
            for kpsoi_aug in kpsois_aug:
                similar_to_rot_0 = np.allclose(
                    [kpsoi_aug.keypoints[0].x, kpsoi_aug.keypoints[0].y],
                    kpsoi_rot[0])
                similar_to_rot_180 = np.allclose(
                    [kpsoi_aug.keypoints[0].x, kpsoi_aug.keypoints[0].y],
                    kpsoi_rot[1])
                if similar_to_rot_0:
                    kpsois_aug_indices.append(0)
                elif similar_to_rot_180:
                    kpsois_aug_indices.append(1)
                else:
                    assert False
        assert np.array_equal(img_aug_indices, kpsois_aug_indices)
        assert len(set(img_aug_indices)) == 2
        assert len(set(kpsois_aug_indices)) == 2

    @classmethod
    def _test_image_cbaoi_alignment(cls, cbaoi, cbaoi_rot, augf_name):
        aug = iaa.Affine(rotate=[0, 180], order=0)
        img = np.zeros((10, 10), dtype=np.uint8)
        img[0:5, 5] = 255
        img[2, 4:6] = 255
        img_rot = [np.copy(img), np.copy(np.flipud(np.fliplr(img)))]

        img_aug_indices = []
        cbaois_aug_indices = []
        for _ in sm.xrange(40):
            aug_det = aug.to_deterministic()
            imgs_aug = aug_det.augment_images([img, img])
            cbaois_aug = getattr(aug_det, augf_name)([cbaoi, cbaoi])

            assert cbaois_aug[0].shape == img.shape
            assert cbaois_aug[1].shape == img.shape
            if hasattr(cbaois_aug[0].items[0], "is_valid"):
                assert cbaois_aug[0].items[0].is_valid
                assert cbaois_aug[1].items[0].is_valid

            for img_aug in imgs_aug:
                if np.array_equal(img_aug, img_rot[0]):
                    img_aug_indices.append(0)
                elif np.array_equal(img_aug, img_rot[1]):
                    img_aug_indices.append(1)
                else:
                    assert False
            for cbaoi_aug in cbaois_aug:
                if cbaoi_aug.items[0].coords_almost_equals(cbaoi_rot[0]):
                    cbaois_aug_indices.append(0)
                elif cbaoi_aug.items[0].coords_almost_equals(cbaoi_rot[1]):
                    cbaois_aug_indices.append(1)
                else:
                    assert False
        assert np.array_equal(img_aug_indices, cbaois_aug_indices)
        assert len(set(img_aug_indices)) == 2
        assert len(set(cbaois_aug_indices)) == 2

    def test_image_polygon_alignment(self):
        psoi = ia.PolygonsOnImage([ia.Polygon([(1, 1), (9, 1), (5, 5)])],
                                  shape=(10, 10))
        psoi_rot = [
            psoi.polygons[0].deepcopy(),
            ia.Polygon([(10-1, 10-1), (10-9, 10-1), (10-5, 10-5)])
        ]
        self._test_image_cbaoi_alignment(psoi, psoi_rot,
                                         "augment_polygons")

    def test_image_line_string_alignment(self):
        lsoi = ia.LineStringsOnImage([ia.LineString([(1, 1), (9, 1), (5, 5)])],
                                     shape=(10, 10))
        lsoi_rot = [
            lsoi.items[0].deepcopy(),
            ia.LineString([(10-1, 10-1), (10-9, 10-1), (10-5, 10-5)])
        ]
        self._test_image_cbaoi_alignment(lsoi, lsoi_rot,
                                         "augment_line_strings")

    def test_image_bounding_box_alignment(self):
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=1, y1=1, x2=9, y2=5)], shape=(10, 10))
        bbsoi_rot = [
            bbsoi.items[0].deepcopy(),
            ia.BoundingBox(x1=10-9, y1=10-5, x2=10-1, y2=10-1)]
        self._test_image_cbaoi_alignment(bbsoi, bbsoi_rot,
                                         "augment_bounding_boxes")


class TestAffine_other_dtypes(unittest.TestCase):
    @property
    def translate_mask(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 2] = True
        return mask

    @property
    def image(self):
        image = np.zeros((17, 17), dtype=bool)
        image[2:15, 5:13] = True
        return image

    @property
    def rot_mask_inner(self):
        img_flipped = iaa.Fliplr(1.0)(image=self.image)
        return img_flipped == 1

    @property
    def rot_mask_outer(self):
        img_flipped = iaa.Fliplr(1.0)(image=self.image)
        return img_flipped == 0

    @property
    def rot_thresh_inner(self):
        return 0.9

    @property
    def rot_thresh_outer(self):
        return 0.9

    def rot_thresh_inner_float(self, order):
        return 0.85 if order == 1 else 0.7

    def rot_thresh_outer_float(self, order):
        return 0.85 if order == 1 else 0.4

    def test_translate_skimage_order_0_bool(self):
        aug = iaa.Affine(translate_px={"x": 1}, order=0, mode="constant",
                         backend="skimage")
        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == image.dtype.name
        assert np.all(image_aug[~self.translate_mask] == 0)
        assert np.all(image_aug[self.translate_mask] == 1)

    def test_translate_skimage_order_0_uint_int(self):
        dtypes = ["uint8", "uint16", "uint32", "int8", "int16", "int32"]
        for dtype in dtypes:
            aug = iaa.Affine(translate_px={"x": 1}, order=0, mode="constant",
                             backend="skimage")

            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            if np.dtype(dtype).kind == "i":
                values = [1, 5, 10, 100, int(0.1 * max_value),
                          int(0.2 * max_value), int(0.5 * max_value),
                          max_value - 100, max_value]
                values = values + [(-1) * value for value in values]
            else:
                values = [1, 5, 10, 100, int(center_value),
                          int(0.1 * max_value), int(0.2 * max_value),
                          int(0.5 * max_value), max_value - 100, max_value]

            for value in values:
                image = np.zeros((3, 3), dtype=dtype)
                image[1, 1] = value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.all(image_aug[~self.translate_mask] == 0)
                assert np.all(image_aug[self.translate_mask] == value)

    def test_translate_skimage_order_0_float(self):
        # float
        dtypes = ["float16", "float32", "float64"]
        for dtype in dtypes:
            aug = iaa.Affine(translate_px={"x": 1}, order=0, mode="constant",
                             backend="skimage")

            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            def _isclose(a, b):
                atol = 1e-4 if dtype == "float16" else 1e-8
                return np.isclose(a, b, atol=atol, rtol=0)

            isize = np.dtype(dtype).itemsize
            values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1),
                      1000 ** (isize - 1)]
            values = values + [(-1) * value for value in values]
            values = values + [min_value, max_value]
            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((3, 3), dtype=dtype)
                    image[1, 1] = value

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert np.all(_isclose(image_aug[~self.translate_mask], 0))
                    assert np.all(_isclose(image_aug[self.translate_mask],
                                           np.float128(value)))

    def test_rotate_skimage_order_not_0_bool(self):
        # skimage, order!=0 and rotate=180
        for order in [1, 3, 4, 5]:
            aug = iaa.Affine(rotate=180, order=order, mode="constant",
                             backend="skimage")
            aug_flip = iaa.Sequential([iaa.Flipud(1.0), iaa.Fliplr(1.0)])

            image = np.zeros((17, 17), dtype=bool)
            image[2:15, 5:13] = True

            image_aug = aug.augment_image(image)
            image_exp = aug_flip.augment_image(image)

            assert image_aug.dtype.name == image.dtype.name
            assert (
                np.sum(image_aug == image_exp)/image.size
            ) > self.rot_thresh_inner

    def test_rotate_skimage_order_not_0_uint_int(self):
        def _compute_matching(image_aug, image_exp, mask):
            return np.sum(
                np.isclose(image_aug[mask], image_exp[mask], rtol=0,
                           atol=1.001)
            ) / np.sum(mask)

        dtypes = ["uint8", "uint16", "uint32", "int8", "int16", "int32"]
        for dtype in dtypes:
            for order in [1, 3, 4, 5]:
                aug = iaa.Affine(rotate=180, order=order, mode="constant",
                                 backend="skimage")
                aug_flip = iaa.Sequential([iaa.Flipud(1.0), iaa.Fliplr(1.0)])

                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                if np.dtype(dtype).kind == "i":
                    values = [1, 5, 10, 100, int(0.1 * max_value),
                              int(0.2 * max_value), int(0.5 * max_value),
                              max_value - 100, max_value]
                    values = values + [(-1) * value for value in values]
                else:
                    values = [1, 5, 10, 100, int(center_value),
                              int(0.1 * max_value), int(0.2 * max_value),
                              int(0.5 * max_value), max_value - 100, max_value]

                for value in values:
                    with self.subTest(dtype=dtype, order=order, value=value):
                        image = np.zeros((17, 17), dtype=dtype)
                        image[2:15, 5:13] = value

                        image_aug = aug.augment_image(image)
                        image_exp = aug_flip.augment_image(image)

                        assert image_aug.dtype.name == dtype
                        assert _compute_matching(
                            image_aug, image_exp, self.rot_mask_inner
                        ) > self.rot_thresh_inner
                        assert _compute_matching(
                            image_aug, image_exp, self.rot_mask_outer
                        ) > self.rot_thresh_outer

    def test_rotate_skimage_order_not_0_float(self):
        def _compute_matching(image_aug, image_exp, mask):
            return np.sum(
                _isclose(image_aug[mask], image_exp[mask])
            ) / np.sum(mask)

        for order in [1, 3, 4, 5]:
            dtypes = ["float16", "float32", "float64"]
            if order == 5:
                # float64 caused too many interpolation inaccuracies for
                # order=5, not wrong but harder to test
                dtypes = ["float16", "float32"]
            for dtype in dtypes:
                aug = iaa.Affine(rotate=180, order=order, mode="constant",
                                 backend="skimage")
                aug_flip = iaa.Sequential([iaa.Flipud(1.0), iaa.Fliplr(1.0)])

                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                def _isclose(a, b):
                    atol = 1e-4 if dtype == "float16" else 1e-8
                    if order not in [0, 1]:
                        atol = 1e-2
                    return np.isclose(a, b, atol=atol, rtol=0)

                isize = np.dtype(dtype).itemsize
                values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1),
                          1000 ** (isize - 1)]
                values = values + [(-1) * value for value in values]
                if order not in [3, 4]:  # results in NaNs otherwise
                    values = values + [min_value, max_value]
                for value in values:
                    with self.subTest(order=order, dtype=dtype, value=value):
                        image = np.zeros((17, 17), dtype=dtype)
                        image[2:15, 5:13] = value

                        image_aug = aug.augment_image(image)
                        image_exp = aug_flip.augment_image(image)

                        assert image_aug.dtype.name == dtype
                        assert _compute_matching(
                            image_aug, image_exp, self.rot_mask_inner
                        ) > self.rot_thresh_inner_float(order)
                        assert _compute_matching(
                            image_aug, image_exp, self.rot_mask_outer
                        ) > self.rot_thresh_outer_float(order)

    def test_translate_cv2_order_0_bool(self):
        aug = iaa.Affine(translate_px={"x": 1}, order=0, mode="constant",
                         backend="cv2")

        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == image.dtype.name
        assert np.all(image_aug[~self.translate_mask] == 0)
        assert np.all(image_aug[self.translate_mask] == 1)

    def test_translate_cv2_order_0_uint_int(self):
        aug = iaa.Affine(translate_px={"x": 1}, order=0, mode="constant",
                         backend="cv2")

        dtypes = ["uint8", "uint16", "int8", "int16", "int32"]
        for dtype in dtypes:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            if np.dtype(dtype).kind == "i":
                values = [1, 5, 10, 100, int(0.1 * max_value),
                          int(0.2 * max_value), int(0.5 * max_value),
                          max_value - 100, max_value]
                values = values + [(-1) * value for value in values]
            else:
                values = [1, 5, 10, 100, int(center_value),
                          int(0.1 * max_value), int(0.2 * max_value),
                          int(0.5 * max_value), max_value - 100, max_value]

            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((3, 3), dtype=dtype)
                    image[1, 1] = value

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert np.all(image_aug[~self.translate_mask] == 0)
                    assert np.all(image_aug[self.translate_mask] == value)

    def test_translate_cv2_order_0_float(self):
        aug = iaa.Affine(translate_px={"x": 1}, order=0, mode="constant",
                         backend="cv2")

        dtypes = ["float16", "float32", "float64"]
        for dtype in dtypes:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            def _isclose(a, b):
                atol = 1e-4 if dtype == "float16" else 1e-8
                return np.isclose(a, b, atol=atol, rtol=0)

            isize = np.dtype(dtype).itemsize
            values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1),
                      1000 ** (isize - 1)]
            values = values + [(-1) * value for value in values]
            values = values + [min_value, max_value]
            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((3, 3), dtype=dtype)
                    image[1, 1] = value

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert np.all(_isclose(image_aug[~self.translate_mask], 0))
                    assert np.all(_isclose(image_aug[self.translate_mask],
                                           np.float128(value)))

    def test_rotate_cv2_order_1_and_3_bool(self):
        # cv2, order=1 and rotate=180
        for order in [1, 3]:
            aug = iaa.Affine(rotate=180, order=order, mode="constant",
                             backend="cv2")
            aug_flip = iaa.Sequential([iaa.Flipud(1.0), iaa.Fliplr(1.0)])

            image = np.zeros((17, 17), dtype=bool)
            image[2:15, 5:13] = True

            image_aug = aug.augment_image(image)
            image_exp = aug_flip.augment_image(image)

            assert image_aug.dtype.name == image.dtype.name
            assert (np.sum(image_aug == image_exp) / image.size) > 0.9

    def test_rotate_cv2_order_1_and_3_uint_int(self):
        # cv2, order=1 and rotate=180
        for order in [1, 3]:
            aug = iaa.Affine(rotate=180, order=order, mode="constant",
                             backend="cv2")
            aug_flip = iaa.Sequential([iaa.Flipud(1.0), iaa.Fliplr(1.0)])

            dtypes = ["uint8", "uint16", "int8", "int16"]
            for dtype in dtypes:
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                if np.dtype(dtype).kind == "i":
                    values = [1, 5, 10, 100, int(0.1 * max_value),
                              int(0.2 * max_value), int(0.5 * max_value),
                              max_value - 100, max_value]
                    values = values + [(-1) * value for value in values]
                else:
                    values = [1, 5, 10, 100, int(center_value),
                              int(0.1 * max_value), int(0.2 * max_value),
                              int(0.5 * max_value), max_value - 100, max_value]

                for value in values:
                    with self.subTest(order=order, dtype=dtype, value=value):
                        image = np.zeros((17, 17), dtype=dtype)
                        image[2:15, 5:13] = value

                        image_aug = aug.augment_image(image)
                        image_exp = aug_flip.augment_image(image)

                        assert image_aug.dtype.name == dtype
                        assert (
                            np.sum(image_aug == image_exp) / image.size
                        ) > 0.9

    def test_rotate_cv2_order_1_and_3_float(self):
        # cv2, order=1 and rotate=180
        for order in [1, 3]:
            aug = iaa.Affine(rotate=180, order=order, mode="constant",
                             backend="cv2")
            aug_flip = iaa.Sequential([iaa.Flipud(1.0), iaa.Fliplr(1.0)])

            dtypes = ["float16", "float32", "float64"]
            for dtype in dtypes:
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                def _isclose(a, b):
                    atol = 1e-4 if dtype == "float16" else 1e-8
                    return np.isclose(a, b, atol=atol, rtol=0)

                isize = np.dtype(dtype).itemsize
                values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1),
                          1000 ** (isize - 1)]
                values = values + [(-1) * value for value in values]
                values = values + [min_value, max_value]
                for value in values:
                    with self.subTest(order=order, dtype=dtype, value=value):
                        image = np.zeros((17, 17), dtype=dtype)
                        image[2:15, 5:13] = value

                        image_aug = aug.augment_image(image)
                        image_exp = aug_flip.augment_image(image)

                        assert image_aug.dtype.name == dtype
                        assert (
                            np.sum(_isclose(image_aug, image_exp)) / image.size
                        ) > 0.9


class TestAffine_other(unittest.TestCase):
    def test_unusual_channel_numbers(self):
        nb_channels_lst = [4, 5, 512, 513]
        orders = [0, 1, 3]
        backends = ["auto", "skimage", "cv2"]
        for nb_channels, order, backend in itertools.product(nb_channels_lst,
                                                             orders, backends):
            with self.subTest(nb_channels=nb_channels, order=order,
                              backend=backend):
                aug = iaa.Affine(translate_px={"x": -1}, mode="constant",
                                 cval=255, order=order, backend=backend)

                image = np.full((3, 3, nb_channels), 128, dtype=np.uint8)
                heatmap_arr = np.full((3, 3, nb_channels), 0.5,
                                      dtype=np.float32)
                heatmap = ia.HeatmapsOnImage(heatmap_arr, shape=image.shape)

                image_aug, heatmap_aug = aug(image=image, heatmaps=heatmap)
                hm_aug_arr = heatmap_aug.arr_0to1

                assert image_aug.shape == (3, 3, nb_channels)
                assert heatmap_aug.arr_0to1.shape == (3, 3, nb_channels)
                assert heatmap_aug.shape == image.shape
                assert np.allclose(image_aug[:, 0:2, :], 128, rtol=0, atol=2)
                assert np.allclose(image_aug[:, 2:3, 0:3], 255, rtol=0, atol=2)
                assert np.allclose(image_aug[:, 2:3, 3:], 255, rtol=0, atol=2)
                assert np.allclose(hm_aug_arr[:, 0:2, :], 0.5, rtol=0,
                                   atol=0.025)
                assert np.allclose(hm_aug_arr[:, 2:3, :], 0.0, rtol=0,
                                   atol=0.025)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for fit_output in [False, True]:
            for shape in shapes:
                with self.subTest(shape=shape, fit_output=fit_output):
                    image = np.zeros(shape, dtype=np.uint8)
                    aug = iaa.Affine(rotate=45, fit_output=fit_output)

                    image_aug = aug(image=image)

                    assert image_aug.dtype.name == "uint8"
                    assert image_aug.shape == shape

    def test_pickleable(self):
        aug = iaa.Affine(scale=(0.9, 1.1), translate_px=(-4, 4),
                         rotate=(-10, 10), shear=(-10, 10), order=[0, 1])
        runtest_pickleable_uint8_img(aug, iterations=20)


class TestScaleX(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        aug = iaa.ScaleX(1.5)
        assert isinstance(aug, iaa.Affine)
        assert np.isclose(aug.scale[0].value, 1.5)
        assert aug.order.value == 1
        assert aug.cval.value == 0
        assert aug.mode.value == "constant"
        assert aug.fit_output is False

    def test_integrationtest(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[5, 5] = 255
        aug = iaa.ScaleX(4.0, order=0)

        image_aug = aug(image=image)

        xx = np.nonzero(np.max(image_aug, axis=0) > 200)[0]
        yy = np.nonzero(np.max(image_aug, axis=1) > 200)[0]
        x1, x2 = xx[0], xx[-1]
        y1, y2 = yy[0], yy[-1]
        # not >=3, because if e.g. index 1 is spread to 0 to 3 after scaling,
        # it covers four cells (0, 1, 2, 3), but 3-0 is 3
        assert x2 - x1 >= 3
        assert y2 - y1 < 1


class TestScaleY(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        aug = iaa.ScaleY(1.5)
        assert isinstance(aug, iaa.Affine)
        assert np.isclose(aug.scale[1].value, 1.5)
        assert aug.order.value == 1
        assert aug.cval.value == 0
        assert aug.mode.value == "constant"
        assert aug.fit_output is False

    def test_integrationtest(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[5, 5] = 255
        aug = iaa.ScaleY(4.0, order=0)

        image_aug = aug(image=image)

        xx = np.nonzero(np.max(image_aug, axis=0) > 200)[0]
        yy = np.nonzero(np.max(image_aug, axis=1) > 200)[0]
        x1, x2 = xx[0], xx[-1]
        y1, y2 = yy[0], yy[-1]
        # not >=3, because if e.g. index 1 is spread to 0 to 3 after scaling,
        # it covers four cells (0, 1, 2, 3), but 3-0 is 3
        assert y2 - y1 >= 3
        assert x2 - x1 < 1


class TestTranslateX(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___translate_percent(self):
        aug = iaa.TranslateX(percent=0.5)
        assert isinstance(aug, iaa.Affine)
        assert np.isclose(aug.translate[0].value, 0.5)
        assert aug.order.value == 1
        assert aug.cval.value == 0
        assert aug.mode.value == "constant"
        assert aug.fit_output is False

    def test___init___translate_px(self):
        aug = iaa.TranslateX(px=2)
        assert isinstance(aug, iaa.Affine)
        assert np.isclose(aug.translate[0].value, 2)
        assert aug.order.value == 1
        assert aug.cval.value == 0
        assert aug.mode.value == "constant"
        assert aug.fit_output is False

    def test___init___both_none(self):
        with self.assertRaises(AssertionError) as ctx:
            _aug = iaa.TranslateX()
        assert "but both were None" in str(ctx.exception)

    def test_integrationtest_translate_percent(self):
        image = np.full((50, 50), 255, dtype=np.uint8)
        aug = iaa.TranslateX(percent=0.5, order=1, cval=0)

        image_aug = aug(image=image)

        expected = np.copy(image)
        expected[:, 0:25] = 0
        overlap = np.average(np.isclose(image_aug, expected, atol=1.01))
        assert overlap > (1.0 - (1/50) - 1e-4)

    def test_integrationtest_translate_px(self):
        image = np.full((50, 50), 255, dtype=np.uint8)
        aug = iaa.TranslateX(px=25, order=1, cval=0)

        image_aug = aug(image=image)

        expected = np.copy(image)
        expected[:, 0:25] = 0
        overlap = np.average(np.isclose(image_aug, expected, atol=1.01))
        assert overlap > (1.0 - (1/50) - 1e-4)


class TestTranslateY(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___translate_percent(self):
        aug = iaa.TranslateY(percent=0.5)
        assert isinstance(aug, iaa.Affine)
        assert np.isclose(aug.translate[1].value, 0.5)
        assert aug.order.value == 1
        assert aug.cval.value == 0
        assert aug.mode.value == "constant"
        assert aug.fit_output is False

    def test___init___translate_px(self):
        aug = iaa.TranslateY(px=2)
        assert isinstance(aug, iaa.Affine)
        assert np.isclose(aug.translate[1].value, 2)
        assert aug.order.value == 1
        assert aug.cval.value == 0
        assert aug.mode.value == "constant"
        assert aug.fit_output is False

    def test___init___both_none(self):
        with self.assertRaises(AssertionError) as ctx:
            _aug = iaa.TranslateY()
        assert "but both were None" in str(ctx.exception)

    def test_integrationtest_translate_percent(self):
        image = np.full((50, 50), 255, dtype=np.uint8)
        aug = iaa.TranslateY(percent=0.5, order=1, cval=0)

        image_aug = aug(image=image)

        expected = np.copy(image)
        expected[0:25, :] = 0
        overlap = np.average(np.isclose(image_aug, expected, atol=1.01))
        assert overlap > (1.0 - (1/50) - 1e-4)

    def test_integrationtest_translate_px(self):
        image = np.full((50, 50), 255, dtype=np.uint8)
        aug = iaa.TranslateY(px=25, order=1, cval=0)

        image_aug = aug(image=image)

        expected = np.copy(image)
        expected[0:25, :] = 0
        overlap = np.average(np.isclose(image_aug, expected, atol=1.01))
        assert overlap > (1.0 - (1/50) - 1e-4)


class TestRotate(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___(self):
        aug = iaa.Rotate(rotate=45)
        assert isinstance(aug, iaa.Affine)
        assert np.isclose(aug.rotate.value, 45)
        assert aug.order.value == 1
        assert aug.cval.value == 0
        assert aug.mode.value == "constant"
        assert aug.fit_output is False

    def test_integrationtest(self):
        image = np.zeros((40, 20), dtype=np.uint8)
        image[:, 10:10+1] = 255
        aug = iaa.Rotate(90, order=0)

        image_aug = aug(image=image)

        assert image_aug.shape == (40, 20)
        assert np.isclose(np.sum(image_aug[20-1:20+2, :]), 255*20, atol=1)


class TestShearX(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        aug = iaa.ShearX(40)
        assert isinstance(aug, iaa.Affine)
        assert aug.shear[0].value == 40
        assert aug.order.value == 1
        assert aug.cval.value == 0
        assert aug.mode.value == "constant"
        assert aug.fit_output is False

    def test_integrationtest(self):
        def _find_coords(arr):
            xx = np.nonzero(np.max(arr, axis=0) > 200)[0]
            yy = np.nonzero(np.max(arr, axis=1) > 200)[0]
            x1 = xx[0]
            x2 = xx[-1]
            y1 = yy[0]
            y2 = yy[-1]
            return x1+(x2-x1)/2, y1+(y2-y1)/2

        image = np.zeros((50, 50, 4), dtype=np.uint8)
        image[10:10+1, 20:20+1, 0] = 255
        image[10:10+1, 30:30+1, 1] = 255
        image[40:40+1, 30:30+1, 2] = 255
        image[40:40+1, 20:20+1, 3] = 255
        aug = iaa.ShearX(30, order=0)

        image_aug = aug(image=image)

        x1, y1 = _find_coords(image_aug[..., 0])
        x2, y2 = _find_coords(image_aug[..., 1])
        x3, y3 = _find_coords(image_aug[..., 2])
        x4, y4 = _find_coords(image_aug[..., 3])
        assert x1 > 20
        assert y1 > 10
        assert y2 > 10
        assert np.isclose(y1, y2)
        assert x3 < 30
        assert y3 < 40
        assert y4 < 40
        assert np.isclose(y3, y4)
        assert not np.isclose(x1, x4)
        assert not np.isclose(x2, x3)


class TestShearY(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        aug = iaa.ShearY(40)
        assert isinstance(aug, iaa.Affine)
        assert aug.shear[1].value == 40
        assert aug.order.value == 1
        assert aug.cval.value == 0
        assert aug.mode.value == "constant"
        assert aug.fit_output is False

    def test_integrationtest(self):
        def _find_coords(arr):
            xx = np.nonzero(np.max(arr, axis=0) > 200)[0]
            yy = np.nonzero(np.max(arr, axis=1) > 200)[0]
            x1 = xx[0]
            x2 = xx[-1]
            y1 = yy[0]
            y2 = yy[-1]
            return x1+(x2-x1)/2, y1+(y2-y1)/2

        image = np.zeros((50, 50, 4), dtype=np.uint8)
        image[20:20+1, 10:10+1, 0] = 255
        image[20:20+1, 40:40+1, 1] = 255
        image[30:30+1, 40:40+1, 2] = 255
        image[30:30+1, 10:10+1, 3] = 255
        aug = iaa.ShearY(30, order=0)

        image_aug = aug(image=image)

        x1, y1 = _find_coords(image_aug[..., 0])
        x2, y2 = _find_coords(image_aug[..., 1])
        x3, y3 = _find_coords(image_aug[..., 2])
        x4, y4 = _find_coords(image_aug[..., 3])
        assert y1 < 20
        assert x1 > 10
        assert x4 > 10
        assert np.isclose(x1, x4)
        assert y2 > 20
        assert x2 < 40
        assert x3 < 40
        assert np.isclose(x2, x3)
        assert not np.isclose(y1, y2)
        assert not np.isclose(y3, y4)


# TODO migrate to unittest and split up tests or remove AffineCv2
def test_AffineCv2():
    reseed()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        _ = iaa.AffineCv2()

    assert "is deprecated" in str(caught_warnings[0].message)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ia.DeprecationWarning)

        base_img = np.array([[0, 0, 0],
                             [0, 255, 0],
                             [0, 0, 0]], dtype=np.uint8)
        base_img = base_img[:, :, np.newaxis]

        images = np.array([base_img])
        images_list = [base_img]
        outer_pixels = ([], [])
        for i in sm.xrange(base_img.shape[0]):
            for j in sm.xrange(base_img.shape[1]):
                if i != j:
                    outer_pixels[0].append(i)
                    outer_pixels[1].append(j)

        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        keypoints = [ia.KeypointsOnImage(kps, shape=base_img.shape)]

        # no translation/scale/rotate/shear, shouldnt change nothing
        aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        # ---------------------
        # scale
        # ---------------------
        # zoom in
        aug = iaa.AffineCv2(scale=1.75, translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        assert observed[0][1, 1] > 250
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

        observed = aug_det.augment_images(images)
        assert observed[0][1, 1] > 250
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

        observed = aug.augment_images(images_list)
        assert observed[0][1, 1] > 250
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

        observed = aug_det.augment_images(images_list)
        assert observed[0][1, 1] > 250
        assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
        assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

        observed = aug.augment_keypoints(keypoints)
        assert observed[0].keypoints[0].x < 0
        assert observed[0].keypoints[0].y < 0
        assert observed[0].keypoints[1].x == 1
        assert observed[0].keypoints[1].y == 1
        assert observed[0].keypoints[2].x > 2
        assert observed[0].keypoints[2].y > 2

        observed = aug_det.augment_keypoints(keypoints)
        assert observed[0].keypoints[0].x < 0
        assert observed[0].keypoints[0].y < 0
        assert observed[0].keypoints[1].x == 1
        assert observed[0].keypoints[1].y == 1
        assert observed[0].keypoints[2].x > 2
        assert observed[0].keypoints[2].y > 2

        # zoom in only on x axis
        aug = iaa.AffineCv2(scale={"x": 1.75, "y": 1.0}, translate_px=0,
                            rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        assert observed[0][1, 1] > 250
        assert (observed[0][[1, 1], [0, 2]] > 20).all()
        assert (observed[0][[1, 1], [0, 2]] < 150).all()
        assert (observed[0][0, :] < 5).all()
        assert (observed[0][2, :] < 5).all()

        observed = aug_det.augment_images(images)
        assert observed[0][1, 1] > 250
        assert (observed[0][[1, 1], [0, 2]] > 20).all()
        assert (observed[0][[1, 1], [0, 2]] < 150).all()
        assert (observed[0][0, :] < 5).all()
        assert (observed[0][2, :] < 5).all()

        observed = aug.augment_images(images_list)
        assert observed[0][1, 1] > 250
        assert (observed[0][[1, 1], [0, 2]] > 20).all()
        assert (observed[0][[1, 1], [0, 2]] < 150).all()
        assert (observed[0][0, :] < 5).all()
        assert (observed[0][2, :] < 5).all()

        observed = aug_det.augment_images(images_list)
        assert observed[0][1, 1] > 250
        assert (observed[0][[1, 1], [0, 2]] > 20).all()
        assert (observed[0][[1, 1], [0, 2]] < 150).all()
        assert (observed[0][0, :] < 5).all()
        assert (observed[0][2, :] < 5).all()

        observed = aug.augment_keypoints(keypoints)
        assert observed[0].keypoints[0].x < 0
        assert observed[0].keypoints[0].y == 0
        assert observed[0].keypoints[1].x == 1
        assert observed[0].keypoints[1].y == 1
        assert observed[0].keypoints[2].x > 2
        assert observed[0].keypoints[2].y == 2

        observed = aug_det.augment_keypoints(keypoints)
        assert observed[0].keypoints[0].x < 0
        assert observed[0].keypoints[0].y == 0
        assert observed[0].keypoints[1].x == 1
        assert observed[0].keypoints[1].y == 1
        assert observed[0].keypoints[2].x > 2
        assert observed[0].keypoints[2].y == 2

        # zoom in only on y axis
        aug = iaa.AffineCv2(scale={"x": 1.0, "y": 1.75}, translate_px=0,
                            rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        assert observed[0][1, 1] > 250
        assert (observed[0][[0, 2], [1, 1]] > 20).all()
        assert (observed[0][[0, 2], [1, 1]] < 150).all()
        assert (observed[0][:, 0] < 5).all()
        assert (observed[0][:, 2] < 5).all()

        observed = aug_det.augment_images(images)
        assert observed[0][1, 1] > 250
        assert (observed[0][[0, 2], [1, 1]] > 20).all()
        assert (observed[0][[0, 2], [1, 1]] < 150).all()
        assert (observed[0][:, 0] < 5).all()
        assert (observed[0][:, 2] < 5).all()

        observed = aug.augment_images(images_list)
        assert observed[0][1, 1] > 250
        assert (observed[0][[0, 2], [1, 1]] > 20).all()
        assert (observed[0][[0, 2], [1, 1]] < 150).all()
        assert (observed[0][:, 0] < 5).all()
        assert (observed[0][:, 2] < 5).all()

        observed = aug_det.augment_images(images_list)
        assert observed[0][1, 1] > 250
        assert (observed[0][[0, 2], [1, 1]] > 20).all()
        assert (observed[0][[0, 2], [1, 1]] < 150).all()
        assert (observed[0][:, 0] < 5).all()
        assert (observed[0][:, 2] < 5).all()

        observed = aug.augment_keypoints(keypoints)
        assert observed[0].keypoints[0].x == 0
        assert observed[0].keypoints[0].y < 0
        assert observed[0].keypoints[1].x == 1
        assert observed[0].keypoints[1].y == 1
        assert observed[0].keypoints[2].x == 2
        assert observed[0].keypoints[2].y > 2

        observed = aug_det.augment_keypoints(keypoints)
        assert observed[0].keypoints[0].x == 0
        assert observed[0].keypoints[0].y < 0
        assert observed[0].keypoints[1].x == 1
        assert observed[0].keypoints[1].y == 1
        assert observed[0].keypoints[2].x == 2
        assert observed[0].keypoints[2].y > 2

        # zoom out
        # this one uses a 4x4 area of all 255, which is zoomed out to a 4x4
        # area in which the center 2x2 area is 255
        # zoom in should probably be adapted to this style
        # no separate tests here for x/y axis, should work fine if zoom in
        # works with that
        aug = iaa.AffineCv2(scale=0.49, translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        image = np.ones((4, 4, 1), dtype=np.uint8) * 255
        images = np.array([image])
        images_list = [image]
        outer_pixels = ([], [])
        for y in sm.xrange(4):
            xs = sm.xrange(4) if y in [0, 3] else [0, 3]
            for x in xs:
                outer_pixels[0].append(y)
                outer_pixels[1].append(x)
        inner_pixels = ([1, 1, 2, 2], [1, 2, 1, 2])
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=3, y=0),
               ia.Keypoint(x=0, y=3), ia.Keypoint(x=3, y=3)]
        keypoints = [ia.KeypointsOnImage(kps, shape=image.shape)]
        kps_aug = [ia.Keypoint(x=0.765, y=0.765),
                   ia.Keypoint(x=2.235, y=0.765),
                   ia.Keypoint(x=0.765, y=2.235),
                   ia.Keypoint(x=2.235, y=2.235)]
        keypoints_aug = [ia.KeypointsOnImage(kps_aug, shape=image.shape)]

        observed = aug.augment_images(images)
        assert (observed[0][outer_pixels] < 25).all()
        assert (observed[0][inner_pixels] > 200).all()

        observed = aug_det.augment_images(images)
        assert (observed[0][outer_pixels] < 25).all()
        assert (observed[0][inner_pixels] > 200).all()

        observed = aug.augment_images(images_list)
        assert (observed[0][outer_pixels] < 25).all()
        assert (observed[0][inner_pixels] > 200).all()

        observed = aug_det.augment_images(images_list)
        assert (observed[0][outer_pixels] < 25).all()
        assert (observed[0][inner_pixels] > 200).all()

        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        observed = aug_det.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        # varying scales
        aug = iaa.AffineCv2(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                            translate_px=0, rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        image = np.array([[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 2, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0]], dtype=np.uint8) * 100
        image = image[:, :, np.newaxis]
        images = np.array([image])

        last_aug = None
        last_aug_det = None
        nb_changed_aug = 0
        nb_changed_aug_det = 0
        nb_iterations = 1000
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(images)
            observed_aug_det = aug_det.augment_images(images)
            if i == 0:
                last_aug = observed_aug
                last_aug_det = observed_aug_det
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                if not np.array_equal(observed_aug_det, last_aug_det):
                    nb_changed_aug_det += 1
                last_aug = observed_aug
                last_aug_det = observed_aug_det
        assert nb_changed_aug >= int(nb_iterations * 0.8)
        assert nb_changed_aug_det == 0

        aug = iaa.AffineCv2(scale=iap.Uniform(0.7, 0.9))
        assert isinstance(aug.scale, iap.Uniform)
        assert isinstance(aug.scale.a, iap.Deterministic)
        assert isinstance(aug.scale.b, iap.Deterministic)
        assert 0.7 - 1e-8 < aug.scale.a.value < 0.7 + 1e-8
        assert 0.9 - 1e-8 < aug.scale.b.value < 0.9 + 1e-8

        # ---------------------
        # translate
        # ---------------------
        # move one pixel to the right
        aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0},
                            rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        image = np.zeros((3, 3, 1), dtype=np.uint8)
        image_aug = np.copy(image)
        image[1, 1] = 255
        image_aug[1, 2] = 255
        images = np.array([image])
        images_aug = np.array([image_aug])
        images_list = [image]
        images_aug_list = [image_aug]
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)],
                                         shape=base_img.shape)]
        keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)],
                                             shape=base_img.shape)]

        observed = aug.augment_images(images)
        assert np.array_equal(observed, images_aug)

        observed = aug_det.augment_images(images)
        assert np.array_equal(observed, images_aug)

        observed = aug.augment_images(images_list)
        assert array_equal_lists(observed, images_aug_list)

        observed = aug_det.augment_images(images_list)
        assert array_equal_lists(observed, images_aug_list)

        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        observed = aug_det.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        # move one pixel to the right
        aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0},
                            rotate=0, shear=0)
        observed = aug.augment_images(images)
        assert np.array_equal(observed, images_aug)

        # move one pixel to the right
        aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0},
                            rotate=0, shear=0)
        observed = aug.augment_images(images)
        assert np.array_equal(observed, images_aug)

        # move one pixel to the right
        # with order=ALL
        aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0},
                            rotate=0, shear=0, order=ia.ALL)
        observed = aug.augment_images(images)
        assert np.array_equal(observed, images_aug)

        # move one pixel to the right
        # with order=list
        aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0},
                            rotate=0, shear=0, order=[0, 1, 2])
        observed = aug.augment_images(images)
        assert np.array_equal(observed, images_aug)

        # move one pixel to the right
        # with order=StochasticParameter
        aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0},
                            rotate=0, shear=0, order=iap.Choice([0, 1, 2]))
        observed = aug.augment_images(images)
        assert np.array_equal(observed, images_aug)

        # move one pixel to the bottom
        aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 0, "y": 1},
                            rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        image = np.zeros((3, 3, 1), dtype=np.uint8)
        image_aug = np.copy(image)
        image[1, 1] = 255
        image_aug[2, 1] = 255
        images = np.array([image])
        images_aug = np.array([image_aug])
        images_list = [image]
        images_aug_list = [image_aug]
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)],
                                         shape=base_img.shape)]
        keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)],
                                             shape=base_img.shape)]

        observed = aug.augment_images(images)
        assert np.array_equal(observed, images_aug)

        observed = aug_det.augment_images(images)
        assert np.array_equal(observed, images_aug)

        observed = aug.augment_images(images_list)
        assert array_equal_lists(observed, images_aug_list)

        observed = aug_det.augment_images(images_list)
        assert array_equal_lists(observed, images_aug_list)

        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        observed = aug_det.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        # move 33% (one pixel) to the right
        aug = iaa.AffineCv2(scale=1.0, translate_percent={"x": 0.3333, "y": 0},
                            rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        image = np.zeros((3, 3, 1), dtype=np.uint8)
        image_aug = np.copy(image)
        image[1, 1] = 255
        image_aug[1, 2] = 255
        images = np.array([image])
        images_aug = np.array([image_aug])
        images_list = [image]
        images_aug_list = [image_aug]
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)],
                                         shape=base_img.shape)]
        keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)],
                                             shape=base_img.shape)]

        observed = aug.augment_images(images)
        assert np.array_equal(observed, images_aug)

        observed = aug_det.augment_images(images)
        assert np.array_equal(observed, images_aug)

        observed = aug.augment_images(images_list)
        assert array_equal_lists(observed, images_aug_list)

        observed = aug_det.augment_images(images_list)
        assert array_equal_lists(observed, images_aug_list)

        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        observed = aug_det.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        # move 33% (one pixel) to the bottom
        aug = iaa.AffineCv2(scale=1.0, translate_percent={"x": 0, "y": 0.3333},
                            rotate=0, shear=0)
        aug_det = aug.to_deterministic()

        image = np.zeros((3, 3, 1), dtype=np.uint8)
        image_aug = np.copy(image)
        image[1, 1] = 255
        image_aug[2, 1] = 255
        images = np.array([image])
        images_aug = np.array([image_aug])
        images_list = [image]
        images_aug_list = [image_aug]
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)],
                                         shape=base_img.shape)]
        keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)],
                                             shape=base_img.shape)]

        observed = aug.augment_images(images)
        assert np.array_equal(observed, images_aug)

        observed = aug_det.augment_images(images)
        assert np.array_equal(observed, images_aug)

        observed = aug.augment_images(images_list)
        assert array_equal_lists(observed, images_aug_list)

        observed = aug_det.augment_images(images_list)
        assert array_equal_lists(observed, images_aug_list)

        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        observed = aug_det.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        # 0-1px to left/right and 0-1px to top/bottom
        aug = iaa.AffineCv2(scale=1.0,
                            translate_px={"x": (-1, 1), "y": (-1, 1)},
                            rotate=0, shear=0)
        aug_det = aug.to_deterministic()
        last_aug = None
        last_aug_det = None
        nb_changed_aug = 0
        nb_changed_aug_det = 0
        nb_iterations = 1000
        centers_aug = np.copy(image).astype(np.int32) * 0
        centers_aug_det = np.copy(image).astype(np.int32) * 0
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(images)
            observed_aug_det = aug_det.augment_images(images)
            if i == 0:
                last_aug = observed_aug
                last_aug_det = observed_aug_det
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                if not np.array_equal(observed_aug_det, last_aug_det):
                    nb_changed_aug_det += 1
                last_aug = observed_aug
                last_aug_det = observed_aug_det

            assert len(observed_aug[0].nonzero()[0]) == 1
            assert len(observed_aug_det[0].nonzero()[0]) == 1
            centers_aug += (observed_aug[0] > 0)
            centers_aug_det += (observed_aug_det[0] > 0)

        assert nb_changed_aug >= int(nb_iterations * 0.7)
        assert nb_changed_aug_det == 0
        assert (centers_aug > int(nb_iterations * (1/9 * 0.6))).all()
        assert (centers_aug < int(nb_iterations * (1/9 * 1.4))).all()

        aug = iaa.AffineCv2(translate_percent=iap.Uniform(0.7, 0.9))
        assert isinstance(aug.translate, iap.Uniform)
        assert isinstance(aug.translate.a, iap.Deterministic)
        assert isinstance(aug.translate.b, iap.Deterministic)
        assert 0.7 - 1e-8 < aug.translate.a.value < 0.7 + 1e-8
        assert 0.9 - 1e-8 < aug.translate.b.value < 0.9 + 1e-8

        aug = iaa.AffineCv2(translate_px=iap.DiscreteUniform(1, 10))
        assert isinstance(aug.translate, iap.DiscreteUniform)
        assert isinstance(aug.translate.a, iap.Deterministic)
        assert isinstance(aug.translate.b, iap.Deterministic)
        assert aug.translate.a.value == 1
        assert aug.translate.b.value == 10

        # ---------------------
        # translate heatmaps
        # ---------------------
        heatmaps = HeatmapsOnImage(
            np.float32([
                [0.0, 0.5, 0.75],
                [0.0, 0.5, 0.75],
                [0.75, 0.75, 0.75],
            ]),
            shape=(3, 3, 3)
        )
        arr_expected_1px_right = np.float32([
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.75, 0.75],
        ])
        aug = iaa.AffineCv2(translate_px={"x": 1})
        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == heatmaps.shape
        assert np.isclose(observed.min_value, heatmaps.min_value,
                          rtol=0, atol=1e-6)
        assert np.isclose(observed.max_value, heatmaps.max_value,
                          rtol=0, atol=1e-6)
        assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

        # should still use mode=constant cval=0 even when other settings chosen
        aug = iaa.AffineCv2(translate_px={"x": 1}, cval=255)
        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == heatmaps.shape
        assert np.isclose(observed.min_value, heatmaps.min_value,
                          rtol=0, atol=1e-6)
        assert np.isclose(observed.max_value, heatmaps.max_value,
                          rtol=0, atol=1e-6)
        assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

        aug = iaa.AffineCv2(translate_px={"x": 1}, mode="replicate", cval=255)
        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == heatmaps.shape
        assert np.isclose(observed.min_value, heatmaps.min_value,
                          rtol=0, atol=1e-6)
        assert np.isclose(observed.max_value, heatmaps.max_value,
                          rtol=0, atol=1e-6)
        assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

        # ---------------------
        # translate segmaps
        # ---------------------
        segmaps = SegmentationMapsOnImage(
            np.int32([
                [0, 1, 2],
                [0, 1, 2],
                [2, 2, 2],
            ]),
            shape=(3, 3, 3)
        )
        arr_expected_1px_right = np.int32([
            [0, 0, 1],
            [0, 0, 1],
            [0, 2, 2],
        ])
        aug = iaa.AffineCv2(translate_px={"x": 1})
        observed = aug.augment_segmentation_maps([segmaps])[0]
        assert observed.shape == segmaps.shape
        assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

        # should still use mode=constant cval=0 even when other settings chosen
        aug = iaa.AffineCv2(translate_px={"x": 1}, cval=255)
        observed = aug.augment_segmentation_maps([segmaps])[0]
        assert observed.shape == segmaps.shape
        assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

        aug = iaa.AffineCv2(translate_px={"x": 1}, mode="replicate", cval=255)
        observed = aug.augment_segmentation_maps([segmaps])[0]
        assert observed.shape == segmaps.shape
        assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

        # ---------------------
        # rotate
        # ---------------------
        # rotate by 45 degrees
        aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=90, shear=0)
        aug_det = aug.to_deterministic()

        image = np.zeros((3, 3, 1), dtype=np.uint8)
        image_aug = np.copy(image)
        image[1, :] = 255
        image_aug[0, 1] = 255
        image_aug[1, 1] = 255
        image_aug[2, 1] = 255
        images = np.array([image])
        images_aug = np.array([image_aug])
        images_list = [image]
        images_aug_list = [image_aug]
        kps = [ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=1)]
        keypoints = [ia.KeypointsOnImage(kps, shape=base_img.shape)]
        kps_aug = [ia.Keypoint(x=1, y=0), ia.Keypoint(x=1, y=1),
                   ia.Keypoint(x=1, y=2)]
        keypoints_aug = [ia.KeypointsOnImage(kps_aug, shape=base_img.shape)]

        observed = aug.augment_images(images)
        observed[observed >= 100] = 255
        observed[observed < 100] = 0
        assert np.array_equal(observed, images_aug)

        observed = aug_det.augment_images(images)
        observed[observed >= 100] = 255
        observed[observed < 100] = 0
        assert np.array_equal(observed, images_aug)

        observed = aug.augment_images(images_list)
        observed[0][observed[0] >= 100] = 255
        observed[0][observed[0] < 100] = 0
        assert array_equal_lists(observed, images_aug_list)

        observed = aug_det.augment_images(images_list)
        observed[0][observed[0] >= 100] = 255
        observed[0][observed[0] < 100] = 0
        assert array_equal_lists(observed, images_aug_list)

        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        observed = aug_det.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_aug)

        # rotate by StochasticParameter
        aug = iaa.AffineCv2(scale=1.0, translate_px=0,
                            rotate=iap.Uniform(10, 20), shear=0)
        assert isinstance(aug.rotate, iap.Uniform)
        assert isinstance(aug.rotate.a, iap.Deterministic)
        assert aug.rotate.a.value == 10
        assert isinstance(aug.rotate.b, iap.Deterministic)
        assert aug.rotate.b.value == 20

        # random rotation 0-364 degrees
        aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=(0, 364),
                            shear=0)
        aug_det = aug.to_deterministic()
        last_aug = None
        last_aug_det = None
        nb_changed_aug = 0
        nb_changed_aug_det = 0
        nb_iterations = 1000
        pixels_sums_aug = np.copy(image).astype(np.int32) * 0
        pixels_sums_aug_det = np.copy(image).astype(np.int32) * 0
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(images)
            observed_aug_det = aug_det.augment_images(images)
            if i == 0:
                last_aug = observed_aug
                last_aug_det = observed_aug_det
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                if not np.array_equal(observed_aug_det, last_aug_det):
                    nb_changed_aug_det += 1
                last_aug = observed_aug
                last_aug_det = observed_aug_det

            pixels_sums_aug += (observed_aug[0] > 100)
            pixels_sums_aug_det += (observed_aug_det[0] > 100)

        assert nb_changed_aug >= int(nb_iterations * 0.9)
        assert nb_changed_aug_det == 0
        # center pixel, should always be white when rotating line around center
        assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
        assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)

        # outer pixels, should sometimes be white
        # the values here had to be set quite tolerant, the middle pixels at
        # top/left/bottom/right get more activation than expected
        outer_pixels = ([0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2])
        assert (
            pixels_sums_aug[outer_pixels] > int(nb_iterations * (2/8 * 0.4))
        ).all()
        assert (
            pixels_sums_aug[outer_pixels] < int(nb_iterations * (2/8 * 2.0))
        ).all()

        # ---------------------
        # shear
        # ---------------------
        # TODO

        # shear by StochasticParameter
        aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=0,
                            shear=iap.Uniform(10, 20))
        assert isinstance(aug.shear, iap.Uniform)
        assert isinstance(aug.shear.a, iap.Deterministic)
        assert aug.shear.a.value == 10
        assert isinstance(aug.shear.b, iap.Deterministic)
        assert aug.shear.b.value == 20

        # ---------------------
        # cval
        # ---------------------
        aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0,
                            cval=128)
        aug_det = aug.to_deterministic()

        image = np.ones((3, 3, 1), dtype=np.uint8) * 255
        image_aug = np.copy(image)
        images = np.array([image])
        images_list = [image]

        observed = aug.augment_images(images)
        assert (observed[0] > 128 - 30).all()
        assert (observed[0] < 128 + 30).all()

        observed = aug_det.augment_images(images)
        assert (observed[0] > 128 - 30).all()
        assert (observed[0] < 128 + 30).all()

        observed = aug.augment_images(images_list)
        assert (observed[0] > 128 - 30).all()
        assert (observed[0] < 128 + 30).all()

        observed = aug_det.augment_images(images_list)
        assert (observed[0] > 128 - 30).all()
        assert (observed[0] < 128 + 30).all()

        # random cvals
        aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0,
                            cval=(0, 255))
        aug_det = aug.to_deterministic()
        last_aug = None
        last_aug_det = None
        nb_changed_aug = 0
        nb_changed_aug_det = 0
        nb_iterations = 1000
        averages = []
        for i in sm.xrange(nb_iterations):
            observed_aug = aug.augment_images(images)
            observed_aug_det = aug_det.augment_images(images)
            if i == 0:
                last_aug = observed_aug
                last_aug_det = observed_aug_det
            else:
                if not np.array_equal(observed_aug, last_aug):
                    nb_changed_aug += 1
                if not np.array_equal(observed_aug_det, last_aug_det):
                    nb_changed_aug_det += 1
                last_aug = observed_aug
                last_aug_det = observed_aug_det

            averages.append(int(np.average(observed_aug)))

        assert nb_changed_aug >= int(nb_iterations * 0.9)
        assert nb_changed_aug_det == 0
        # center pixel, should always be white when rotating line around center
        assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
        assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)
        assert len(set(averages)) > 200

        aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0,
                            cval=ia.ALL)
        assert isinstance(aug.cval, iap.DiscreteUniform)
        assert isinstance(aug.cval.a, iap.Deterministic)
        assert isinstance(aug.cval.b, iap.Deterministic)
        assert aug.cval.a.value == 0
        assert aug.cval.b.value == 255

        aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0,
                            cval=iap.DiscreteUniform(1, 5))
        assert isinstance(aug.cval, iap.DiscreteUniform)
        assert isinstance(aug.cval.a, iap.Deterministic)
        assert isinstance(aug.cval.b, iap.Deterministic)
        assert aug.cval.a.value == 1
        assert aug.cval.b.value == 5

        # ------------
        # mode
        # ------------
        aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0,
                            cval=0, mode=ia.ALL)
        assert isinstance(aug.mode, iap.Choice)
        aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0,
                            cval=0, mode="replicate")
        assert isinstance(aug.mode, iap.Deterministic)
        assert aug.mode.value == "replicate"
        aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0,
                            cval=0, mode=["replicate", "reflect"])
        assert isinstance(aug.mode, iap.Choice)
        assert (
            len(aug.mode.a) == 2
            and "replicate" in aug.mode.a
            and "reflect" in aug.mode.a)
        aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0,
                            cval=0,
                            mode=iap.Choice(["replicate", "reflect"]))
        assert isinstance(aug.mode, iap.Choice)
        assert (
            len(aug.mode.a) == 2
            and "replicate" in aug.mode.a
            and "reflect" in aug.mode.a)

        # ------------
        # exceptions for bad inputs
        # ------------
        # scale
        got_exception = False
        try:
            _ = iaa.AffineCv2(scale=False)
        except Exception:
            got_exception = True
        assert got_exception

        # translate_px
        got_exception = False
        try:
            _ = iaa.AffineCv2(translate_px=False)
        except Exception:
            got_exception = True
        assert got_exception

        # translate_percent
        got_exception = False
        try:
            _ = iaa.AffineCv2(translate_percent=False)
        except Exception:
            got_exception = True
        assert got_exception

        # rotate
        got_exception = False
        try:
            _ = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=False,
                              shear=0, cval=0)
        except Exception:
            got_exception = True
        assert got_exception

        # shear
        got_exception = False
        try:
            _ = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=0,
                              shear=False, cval=0)
        except Exception:
            got_exception = True
        assert got_exception

        # cval
        got_exception = False
        try:
            _ = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0,
                              shear=0, cval=None)
        except Exception:
            got_exception = True
        assert got_exception

        # mode
        got_exception = False
        try:
            _ = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0,
                              shear=0, cval=0, mode=False)
        except Exception:
            got_exception = True
        assert got_exception

        # non-existent order
        got_exception = False
        try:
            _ = iaa.AffineCv2(order=-1)
        except Exception:
            got_exception = True
        assert got_exception

        # bad order datatype
        got_exception = False
        try:
            _ = iaa.AffineCv2(order="test")
        except Exception:
            got_exception = True
        assert got_exception

        # ----------
        # get_parameters
        # ----------
        aug = iaa.AffineCv2(scale=1, translate_px=2, rotate=3, shear=4,
                            order=1, cval=0, mode="constant")
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)  # scale
        assert isinstance(params[1], iap.Deterministic)  # translate
        assert isinstance(params[2], iap.Deterministic)  # rotate
        assert isinstance(params[3], iap.Deterministic)  # shear
        assert params[0].value == 1  # scale
        assert params[1].value == 2  # translate
        assert params[2].value == 3  # rotate
        assert params[3].value == 4  # shear
        assert params[4].value == 1  # order
        assert params[5].value == 0  # cval
        assert params[6].value == "constant"  # mode


class TestPiecewiseAffine(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        img = np.zeros((60, 80), dtype=np.uint8)
        img[:, 9:11+1] = 255
        img[:, 69:71+1] = 255
        return img

    @property
    def mask(self):
        return self.image > 0

    @property
    def heatmaps(self):
        return HeatmapsOnImage((self.image / 255.0).astype(np.float32),
                               shape=(60, 80, 3))

    @property
    def segmaps(self):
        return SegmentationMapsOnImage(self.mask.astype(np.int32),
                                       shape=(60, 80, 3))

    # -----
    # __init__
    # -----
    def test___init___scale_is_list(self):
        # scale as list
        aug = iaa.PiecewiseAffine(scale=[0.01, 0.10], nb_rows=12, nb_cols=4)
        assert isinstance(aug.scale, iap.Choice)
        assert 0.01 - 1e-8 < aug.scale.a[0] < 0.01 + 1e-8
        assert 0.10 - 1e-8 < aug.scale.a[1] < 0.10 + 1e-8

    def test___init___scale_is_tuple(self):
        # scale as tuple
        aug = iaa.PiecewiseAffine(scale=(0.01, 0.10), nb_rows=12, nb_cols=4)
        assert isinstance(aug.jitter.scale, iap.Uniform)
        assert isinstance(aug.jitter.scale.a, iap.Deterministic)
        assert isinstance(aug.jitter.scale.b, iap.Deterministic)
        assert 0.01 - 1e-8 < aug.jitter.scale.a.value < 0.01 + 1e-8
        assert 0.10 - 1e-8 < aug.jitter.scale.b.value < 0.10 + 1e-8

    def test___init___scale_is_stochastic_parameter(self):
        # scale as StochasticParameter
        aug = iaa.PiecewiseAffine(scale=iap.Uniform(0.01, 0.10), nb_rows=12,
                                  nb_cols=4)
        assert isinstance(aug.jitter.scale, iap.Uniform)
        assert isinstance(aug.jitter.scale.a, iap.Deterministic)
        assert isinstance(aug.jitter.scale.b, iap.Deterministic)
        assert 0.01 - 1e-8 < aug.jitter.scale.a.value < 0.01 + 1e-8
        assert 0.10 - 1e-8 < aug.jitter.scale.b.value < 0.10 + 1e-8

    def test___init___bad_datatype_for_scale_leads_to_failure(self):
        # bad datatype for scale
        got_exception = False
        try:
            _ = iaa.PiecewiseAffine(scale=False, nb_rows=12, nb_cols=4)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test___init___nb_rows_is_list(self):
        # rows as list
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=[4, 20], nb_cols=4)
        assert isinstance(aug.nb_rows, iap.Choice)
        assert aug.nb_rows.a[0] == 4
        assert aug.nb_rows.a[1] == 20

    def test___init___nb_rows_is_tuple(self):
        # rows as tuple
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=(4, 20), nb_cols=4)
        assert isinstance(aug.nb_rows, iap.DiscreteUniform)
        assert isinstance(aug.nb_rows.a, iap.Deterministic)
        assert isinstance(aug.nb_rows.b, iap.Deterministic)
        assert aug.nb_rows.a.value == 4
        assert aug.nb_rows.b.value == 20

    def test___init___nb_rows_is_stochastic_parameter(self):
        # rows as StochasticParameter
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=iap.DiscreteUniform(4, 20),
                                  nb_cols=4)
        assert isinstance(aug.nb_rows, iap.DiscreteUniform)
        assert isinstance(aug.nb_rows.a, iap.Deterministic)
        assert isinstance(aug.nb_rows.b, iap.Deterministic)
        assert aug.nb_rows.a.value == 4
        assert aug.nb_rows.b.value == 20

    def test___init___bad_datatype_for_nb_rows_leads_to_failure(self):
        # bad datatype for rows
        got_exception = False
        try:
            _ = iaa.PiecewiseAffine(scale=0.05, nb_rows=False, nb_cols=4)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test___init___nb_cols_is_list(self):
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=[4, 20])
        assert isinstance(aug.nb_cols, iap.Choice)
        assert aug.nb_cols.a[0] == 4
        assert aug.nb_cols.a[1] == 20

    def test___init___nb_cols_is_tuple(self):
        # cols as tuple
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=(4, 20))
        assert isinstance(aug.nb_cols, iap.DiscreteUniform)
        assert isinstance(aug.nb_cols.a, iap.Deterministic)
        assert isinstance(aug.nb_cols.b, iap.Deterministic)
        assert aug.nb_cols.a.value == 4
        assert aug.nb_cols.b.value == 20

    def test___init___nb_cols_is_stochastic_parameter(self):
        # cols as StochasticParameter
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4,
                                  nb_cols=iap.DiscreteUniform(4, 20))
        assert isinstance(aug.nb_cols, iap.DiscreteUniform)
        assert isinstance(aug.nb_cols.a, iap.Deterministic)
        assert isinstance(aug.nb_cols.b, iap.Deterministic)
        assert aug.nb_cols.a.value == 4
        assert aug.nb_cols.b.value == 20

    def test___init___bad_datatype_for_nb_cols_leads_to_failure(self):
        # bad datatype for cols
        got_exception = False
        try:
            _aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test___init___order_is_int(self):
        # single int for order
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=0)
        assert isinstance(aug.order, iap.Deterministic)
        assert aug.order.value == 0

    def test___init___order_is_list(self):
        # list for order
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8,
                                  order=[0, 1, 3])
        assert isinstance(aug.order, iap.Choice)
        assert all([v in aug.order.a for v in [0, 1, 3]])

    def test___init___order_is_stochastic_parameter(self):
        # StochasticParameter for order
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8,
                                  order=iap.Choice([0, 1, 3]))
        assert isinstance(aug.order, iap.Choice)
        assert all([v in aug.order.a for v in [0, 1, 3]])

    def test___init___order_is_all(self):
        # ALL for order
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8,
                                  order=ia.ALL)
        assert isinstance(aug.order, iap.Choice)
        assert all([v in aug.order.a for v in [0, 1, 3, 4, 5]])

    def test___init___bad_datatype_for_order_leads_to_failure(self):
        # bad datatype for order
        got_exception = False
        try:
            _ = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8,
                                    order=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test___init___cval_is_list(self):
        # cval as list
        aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=5, nb_cols=5,
                                  mode="constant", cval=[0, 10])
        assert isinstance(aug.cval, iap.Choice)
        assert aug.cval.a[0] == 0
        assert aug.cval.a[1] == 10

    def test___init___cval_is_tuple(self):
        # cval as tuple
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8,
                                  mode="constant", cval=(0, 10))
        assert isinstance(aug.cval, iap.Uniform)
        assert isinstance(aug.cval.a, iap.Deterministic)
        assert isinstance(aug.cval.b, iap.Deterministic)
        assert aug.cval.a.value == 0
        assert aug.cval.b.value == 10

    def test___init___cval_is_stochastic_parameter(self):
        # cval as StochasticParameter
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8,
                                  mode="constant",
                                  cval=iap.DiscreteUniform(0, 10))
        assert isinstance(aug.cval, iap.DiscreteUniform)
        assert isinstance(aug.cval.a, iap.Deterministic)
        assert isinstance(aug.cval.b, iap.Deterministic)
        assert aug.cval.a.value == 0
        assert aug.cval.b.value == 10

    def test___init___cval_is_all(self):
        # ALL as cval
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8,
                                  mode="constant", cval=ia.ALL)
        assert isinstance(aug.cval, iap.Uniform)
        assert isinstance(aug.cval.a, iap.Deterministic)
        assert isinstance(aug.cval.b, iap.Deterministic)
        assert aug.cval.a.value == 0
        assert aug.cval.b.value == 255

    def test___init___bad_datatype_for_cval_leads_to_failure(self):
        # bas datatype for cval
        got_exception = False
        try:
            _ = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, cval=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test___init___mode_is_string(self):
        # single string for mode
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8,
                                  mode="nearest")
        assert isinstance(aug.mode, iap.Deterministic)
        assert aug.mode.value == "nearest"

    def test___init___mode_is_list(self):
        # list for mode
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8,
                                  mode=["nearest", "edge", "symmetric"])
        assert isinstance(aug.mode, iap.Choice)
        assert all([
            v in aug.mode.a for v in ["nearest", "edge", "symmetric"]
        ])

    def test___init___mode_is_stochastic_parameter(self):
        # StochasticParameter for mode
        aug = iaa.PiecewiseAffine(
            scale=0.1, nb_rows=8, nb_cols=8,
            mode=iap.Choice(["nearest", "edge", "symmetric"]))
        assert isinstance(aug.mode, iap.Choice)
        assert all([
            v in aug.mode.a for v in ["nearest", "edge", "symmetric"]
        ])

    def test___init___mode_is_all(self):
        # ALL for mode
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode=ia.ALL)
        assert isinstance(aug.mode, iap.Choice)
        assert all([
            v in aug.mode.a
            for v
            in ["constant", "edge", "symmetric", "reflect", "wrap"]
        ])

    def test___init___bad_datatype_for_mode_leads_to_failure(self):
        # bad datatype for mode
        got_exception = False
        try:
            _ = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8,
                                    mode=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    # -----
    # scale
    # -----
    def test_scale_is_small_image(self):
        # basic test
        aug = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)

        observed = aug.augment_image(self.image)

        assert (
            100.0
            < np.average(observed[self.mask])
            < np.average(self.image[self.mask])
        )
        assert (
            100.0-75.0
            > np.average(observed[~self.mask])
            > np.average(self.image[~self.mask])
        )

    def test_scale_is_small_image_absolute_scale(self):
        aug = iaa.PiecewiseAffine(scale=1, nb_rows=12, nb_cols=4,
                                  absolute_scale=True)

        observed = aug.augment_image(self.image)

        assert (
            100.0
            < np.average(observed[self.mask])
            < np.average(self.image[self.mask])
        )
        assert (
            100.0-75.0
            > np.average(observed[~self.mask])
            > np.average(self.image[~self.mask])
        )

    def test_scale_is_small_heatmaps(self):
        # basic test, heatmaps
        aug = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)

        observed = aug.augment_heatmaps([self.heatmaps])[0]

        observed_arr = observed.get_arr()
        assert observed.shape == self.heatmaps.shape
        _assert_same_min_max(observed, self.heatmaps)
        assert (
            100.0/255.0
            < np.average(observed_arr[self.mask])
            < np.average(self.heatmaps.get_arr()[self.mask]))
        assert (
            (100.0-75.0)/255.0
            > np.average(observed_arr[~self.mask])
            > np.average(self.heatmaps.get_arr()[~self.mask]))

    def test_scale_is_small_segmaps(self):
        # basic test, segmaps
        aug = iaa.PiecewiseAffine(scale=0.001, nb_rows=12, nb_cols=4)

        observed = aug.augment_segmentation_maps([self.segmaps])[0]

        observed_arr = observed.get_arr()
        # left column starts at 9-11 and right one at 69-71
        # result is 9-11 (curvy, i.e. like 50% filled) and 70-71 (straight,
        # i.e. 100% filled). Reason for that is unclear, maybe a scikit-image
        # problem.
        observed_arr_left_col = observed_arr[:, 9:11+1]
        observed_arr_right_col = observed_arr[:, 69:71+1]
        assert observed.shape == self.segmaps.shape
        assert np.average(observed_arr_left_col == 1) > 0.5
        assert np.average(observed_arr_right_col == 1) > 0.5
        assert np.average(observed_arr[~self.mask] == 0) > 0.9

    def test_scale_is_zero_image(self):
        # scale 0
        aug = iaa.PiecewiseAffine(scale=0, nb_rows=12, nb_cols=4)

        observed = aug.augment_image(self.image)

        assert np.array_equal(observed, self.image)

    def test_scale_is_zero_image_absolute_scale(self):
        aug = iaa.PiecewiseAffine(scale=0, nb_rows=12, nb_cols=4,
                                  absolute_scale=True)

        observed = aug.augment_image(self.image)

        assert np.array_equal(observed, self.image)

    def test_scale_is_zero_heatmaps(self):
        # scale 0, heatmaps
        aug = iaa.PiecewiseAffine(scale=0, nb_rows=12, nb_cols=4)

        observed = aug.augment_heatmaps([self.heatmaps])[0]

        observed_arr = observed.get_arr()
        assert observed.shape == self.heatmaps.shape
        _assert_same_min_max(observed, self.heatmaps)
        assert np.array_equal(observed_arr, self.heatmaps.get_arr())

    def test_scale_is_zero_segmaps(self):
        # scale 0, segmaps
        aug = iaa.PiecewiseAffine(scale=0, nb_rows=12, nb_cols=4)

        observed = aug.augment_segmentation_maps([self.segmaps])[0]

        observed_arr = observed.get_arr()
        assert observed.shape == self.segmaps.shape
        assert np.array_equal(observed_arr, self.segmaps.get_arr())

    def test_scale_is_zero_keypoints(self):
        # scale 0, keypoints
        aug = iaa.PiecewiseAffine(scale=0, nb_rows=12, nb_cols=4)
        kps = [ia.Keypoint(x=5, y=3), ia.Keypoint(x=3, y=8)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(14, 14, 3))

        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]

        assert_cbaois_equal(kpsoi_aug, kpsoi)

    @classmethod
    def _test_scale_is_zero_cbaoi(cls, cbaoi, augf_name):
        aug = iaa.PiecewiseAffine(scale=0, nb_rows=10, nb_cols=10)

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi)

    def test_scale_is_zero_polygons(self):
        exterior = [(10, 10),
                    (70, 10), (70, 20), (70, 30), (70, 40),
                    (70, 50), (70, 60), (70, 70), (70, 80),
                    (70, 90),
                    (10, 90),
                    (10, 80), (10, 70), (10, 60), (10, 50),
                    (10, 40), (10, 30), (10, 20), (10, 10)]
        poly = ia.Polygon(exterior)
        psoi = ia.PolygonsOnImage([poly, poly.shift(x=1, y=1)],
                                  shape=(100, 80))

        self._test_scale_is_zero_cbaoi(psoi, "augment_polygons")

    def test_scale_is_zero_line_strings(self):
        coords = [(10, 10),
                  (70, 10), (70, 20), (70, 30), (70, 40),
                  (70, 50), (70, 60), (70, 70), (70, 80),
                  (70, 90),
                  (10, 90),
                  (10, 80), (10, 70), (10, 60), (10, 50),
                  (10, 40), (10, 30), (10, 20), (10, 10)]
        ls = ia.LineString(coords)
        lsoi = ia.LineStringsOnImage([ls, ls.shift(x=1, y=1)],
                                     shape=(100, 80))

        self._test_scale_is_zero_cbaoi(lsoi, "augment_line_strings")

    def test_scale_is_zero_bounding_boxes(self):
        bb = ia.BoundingBox(x1=10, y1=10, x2=70, y2=20)
        bbsoi = ia.BoundingBoxesOnImage([bb, bb.shift(x=1, y=1)],
                                        shape=(100, 80))

        self._test_scale_is_zero_cbaoi(bbsoi, "augment_bounding_boxes")

    def test_scale_stronger_values_should_increase_changes_images(self):
        # stronger scale should lead to stronger changes
        aug1 = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
        aug2 = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)

        observed1 = aug1.augment_image(self.image)
        observed2 = aug2.augment_image(self.image)

        assert (
            np.average(observed1[~self.mask])
            < np.average(observed2[~self.mask])
        )

    def test_scale_stronger_values_should_increase_changes_images_abs(self):
        aug1 = iaa.PiecewiseAffine(scale=1, nb_rows=12, nb_cols=4,
                                   absolute_scale=True)
        aug2 = iaa.PiecewiseAffine(scale=10, nb_rows=12, nb_cols=4,
                                   absolute_scale=True)

        observed1 = aug1.augment_image(self.image)
        observed2 = aug2.augment_image(self.image)

        assert (
            np.average(observed1[~self.mask])
            < np.average(observed2[~self.mask])
        )

    def test_scale_stronger_values_should_increase_changes_heatmaps(self):
        # stronger scale should lead to stronger changes, heatmaps
        aug1 = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
        aug2 = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
        
        observed1 = aug1.augment_heatmaps([self.heatmaps])[0]
        observed2 = aug2.augment_heatmaps([self.heatmaps])[0]
        
        observed1_arr = observed1.get_arr()
        observed2_arr = observed2.get_arr()
        assert observed1.shape == self.heatmaps.shape
        assert observed2.shape == self.heatmaps.shape
        _assert_same_min_max(observed1, self.heatmaps)
        _assert_same_min_max(observed2, self.heatmaps)
        assert (
            np.average(observed1_arr[~self.mask])
            < np.average(observed2_arr[~self.mask])
        )

    def test_scale_stronger_values_should_increase_changes_heatmaps_abs(self):
        aug1 = iaa.PiecewiseAffine(scale=1, nb_rows=12, nb_cols=4,
                                   absolute_scale=True)
        aug2 = iaa.PiecewiseAffine(scale=10, nb_rows=12, nb_cols=4,
                                   absolute_scale=True)

        observed1 = aug1.augment_heatmaps([self.heatmaps])[0]
        observed2 = aug2.augment_heatmaps([self.heatmaps])[0]

        observed1_arr = observed1.get_arr()
        observed2_arr = observed2.get_arr()
        assert observed1.shape == self.heatmaps.shape
        assert observed2.shape == self.heatmaps.shape
        _assert_same_min_max(observed1, self.heatmaps)
        _assert_same_min_max(observed2, self.heatmaps)
        assert (
            np.average(observed1_arr[~self.mask])
            < np.average(observed2_arr[~self.mask])
        )

    def test_scale_stronger_values_should_increase_changes_segmaps(self):
        # stronger scale should lead to stronger changes, segmaps
        aug1 = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
        aug2 = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)

        observed1 = aug1.augment_segmentation_maps([self.segmaps])[0]
        observed2 = aug2.augment_segmentation_maps([self.segmaps])[0]

        observed1_arr = observed1.get_arr()
        observed2_arr = observed2.get_arr()
        assert observed1.shape == self.segmaps.shape
        assert observed2.shape == self.segmaps.shape
        assert (
            np.average(observed1_arr[~self.mask] == 0)
            > np.average(observed2_arr[~self.mask] == 0)
        )

    def test_scale_alignment_between_images_and_heatmaps(self):
        # strong scale, measure alignment between images and heatmaps
        aug = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
        aug_det = aug.to_deterministic()

        img_aug = aug_det.augment_image(self.image)
        hm_aug = aug_det.augment_heatmaps([self.heatmaps])[0]

        img_aug_mask = img_aug > 255*0.1
        hm_aug_mask = hm_aug.arr_0to1 > 0.1
        same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
        assert hm_aug.shape == (60, 80, 3)
        _assert_same_min_max(hm_aug, self.heatmaps)
        assert (same / img_aug_mask.size) >= 0.98

    def test_scale_alignment_between_images_and_segmaps(self):
        # strong scale, measure alignment between images and segmaps
        aug = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
        aug_det = aug.to_deterministic()

        img_aug = aug_det.augment_image(self.image)
        segmap_aug = aug_det.augment_segmentation_maps([self.segmaps])[0]

        img_aug_mask = (img_aug > 255*0.1)
        segmap_aug_mask = (segmap_aug.arr == 1)
        same = np.sum(img_aug_mask == segmap_aug_mask[:, :, 0])
        assert segmap_aug.shape == (60, 80, 3)
        assert (same / img_aug_mask.size) >= 0.9

    def test_scale_alignment_between_images_and_smaller_heatmaps(self):
        # strong scale, measure alignment between images and heatmaps
        # heatmaps here smaller than image
        aug = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
        aug_det = aug.to_deterministic()

        heatmaps_small = ia.HeatmapsOnImage(
            (
                ia.imresize_single_image(
                    self.image, (30, 40+10), interpolation="cubic"
                ) / 255.0
            ).astype(np.float32),
            shape=(60, 80, 3)
        )

        img_aug = aug_det.augment_image(self.image)
        hm_aug = aug_det.augment_heatmaps([heatmaps_small])[0]

        img_aug_mask = img_aug > 255*0.1
        hm_aug_mask = ia.imresize_single_image(
            hm_aug.arr_0to1, (60, 80), interpolation="cubic"
        ) > 0.1
        same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
        assert hm_aug.shape == (60, 80, 3)
        assert hm_aug.arr_0to1.shape == (30, 40+10, 1)
        assert (same / img_aug_mask.size) >= 0.9  # seems to be 0.948 actually

    def test_scale_alignment_between_images_and_smaller_heatmaps_abs(self):
        # image is 60x80, so a scale of 8 is about 0.1*max(60,80)
        aug = iaa.PiecewiseAffine(scale=8, nb_rows=12, nb_cols=4,
                                  absolute_scale=True)
        aug_det = aug.to_deterministic()

        heatmaps_small = ia.HeatmapsOnImage(
            (
                ia.imresize_single_image(
                    self.image, (30, 40+10), interpolation="cubic"
                ) / 255.0
            ).astype(np.float32),
            shape=(60, 80, 3)
        )

        img_aug = aug_det.augment_image(self.image)
        hm_aug = aug_det.augment_heatmaps([heatmaps_small])[0]

        img_aug_mask = img_aug > 255*0.1
        hm_aug_mask = ia.imresize_single_image(
            hm_aug.arr_0to1, (60, 80), interpolation="cubic"
        ) > 0.1
        same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
        assert hm_aug.shape == (60, 80, 3)
        assert hm_aug.arr_0to1.shape == (30, 40+10, 1)
        assert (same / img_aug_mask.size) >= 0.9  # seems to be 0.930 actually

    def test_scale_alignment_between_images_and_smaller_segmaps(self):
        # strong scale, measure alignment between images and segmaps
        # segmaps here smaller than image
        aug = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
        aug_det = aug.to_deterministic()
        segmaps_small = SegmentationMapsOnImage(
            (
                ia.imresize_single_image(
                    self.image, (30, 40+10), interpolation="cubic"
                ) > 100
            ).astype(np.int32),
            shape=(60, 80, 3)
        )

        img_aug = aug_det.augment_image(self.image)
        segmaps_aug = aug_det.augment_segmentation_maps([segmaps_small])[0]

        img_aug_mask = img_aug > 255*0.1
        segmaps_aug_mask = (
            ia.imresize_single_image(
                segmaps_aug.arr, (60, 80),
                interpolation="nearest"
            ) == 1
        )
        same = np.sum(img_aug_mask == segmaps_aug_mask[:, :, 0])
        assert segmaps_aug.shape == (60, 80, 3)
        assert segmaps_aug.arr.shape == (30, 40+10, 1)
        assert (same / img_aug_mask.size) >= 0.9

    def test_scale_alignment_between_images_and_keypoints(self):
        # strong scale, measure alignment between images and keypoints
        aug = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
        aug_det = aug.to_deterministic()
        kps = [ia.Keypoint(x=5, y=15), ia.Keypoint(x=17, y=12)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(24, 30, 3))
        img_kps = np.zeros((24, 30, 3), dtype=np.uint8)
        img_kps = kpsoi.draw_on_image(img_kps, color=[255, 255, 255])

        img_kps_aug = aug_det.augment_image(img_kps)
        kpsoi_aug = aug_det.augment_keypoints([kpsoi])[0]

        assert kpsoi_aug.shape == (24, 30, 3)
        bb1 = ia.BoundingBox(
            x1=kpsoi_aug.keypoints[0].x-1, y1=kpsoi_aug.keypoints[0].y-1,
            x2=kpsoi_aug.keypoints[0].x+1, y2=kpsoi_aug.keypoints[0].y+1)
        bb2 = ia.BoundingBox(
            x1=kpsoi_aug.keypoints[1].x-1, y1=kpsoi_aug.keypoints[1].y-1,
            x2=kpsoi_aug.keypoints[1].x+1, y2=kpsoi_aug.keypoints[1].y+1)
        patch1 = bb1.extract_from_image(img_kps_aug)
        patch2 = bb2.extract_from_image(img_kps_aug)
        assert np.max(patch1) > 150
        assert np.max(patch2) > 150
        assert np.average(img_kps_aug) < 40

    # this test was apparently added later on (?) without noticing that
    # a similar test already existed
    def test_scale_alignment_between_images_and_keypoints2(self):
        img = np.zeros((100, 80), dtype=np.uint8)
        img[:, 9:11+1] = 255
        img[:, 69:71+1] = 255
        kps = [ia.Keypoint(x=10, y=20), ia.Keypoint(x=10, y=40),
               ia.Keypoint(x=70, y=20), ia.Keypoint(x=70, y=40)]
        kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)

        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=10, nb_cols=10)
        aug_det = aug.to_deterministic()

        observed_img = aug_det.augment_image(img)
        observed_kpsoi = aug_det.augment_keypoints([kpsoi])

        assert not keypoints_equal([kpsoi], observed_kpsoi)
        for kp in observed_kpsoi[0].keypoints:
            assert observed_img[int(kp.y), int(kp.x)] > 0

    @classmethod
    def _test_scale_alignment_between_images_and_poly_or_line_strings(
            cls, cba_class, cbaoi_class, augf_name):
        img = np.zeros((100, 80), dtype=np.uint8)
        img[:, 10-5:10+5] = 255
        img[:, 70-5:70+5] = 255
        coords = [(10, 10),
                  (70, 10), (70, 20), (70, 30), (70, 40),
                  (70, 50), (70, 60), (70, 70), (70, 80),
                  (70, 90),
                  (10, 90),
                  (10, 80), (10, 70), (10, 60), (10, 50),
                  (10, 40), (10, 30), (10, 20), (10, 10)]
        cba = cba_class(coords)
        cbaoi = cbaoi_class([cba, cba.shift(x=1, y=1)],
                            shape=img.shape)

        aug = iaa.PiecewiseAffine(scale=0.03, nb_rows=10, nb_cols=10)
        aug_det = aug.to_deterministic()

        observed_imgs = aug_det.augment_images([img, img])
        observed_cbaois = getattr(aug_det, augf_name)([cbaoi, cbaoi])

        for observed_img, observed_cbaoi in zip(observed_imgs, observed_cbaois):
            assert observed_cbaoi.shape == img.shape
            for cba_aug in observed_cbaoi.items:
                if hasattr(cba_aug, "is_valid"):
                    assert cba_aug.is_valid
                for point_aug in cba_aug.coords:
                    x = int(np.round(point_aug[0]))
                    y = int(np.round(point_aug[1]))
                    assert observed_img[y, x] > 0

    def test_scale_alignment_between_images_and_polygons(self):
        self._test_scale_alignment_between_images_and_poly_or_line_strings(
            ia.Polygon, ia.PolygonsOnImage, "augment_polygons")

    def test_scale_alignment_between_images_and_line_strings(self):
        self._test_scale_alignment_between_images_and_poly_or_line_strings(
            ia.LineString, ia.LineStringsOnImage, "augment_line_strings")

    def test_scale_alignment_between_images_and_bounding_boxes(self):
        img = np.zeros((100, 80), dtype=np.uint8)
        s = 0
        img[10-s:10+s+1, 20-s:20+s+1] = 255
        img[60-s:60+s+1, 70-s:70+s+1] = 255
        bb = ia.BoundingBox(y1=10, x1=20, y2=60, x2=70)
        bbsoi = ia.BoundingBoxesOnImage([bb], shape=img.shape)

        aug = iaa.PiecewiseAffine(scale=0.03, nb_rows=10, nb_cols=10)

        observed_imgs, observed_bbsois = aug(
            images=[img], bounding_boxes=[bbsoi])

        for observed_img, observed_bbsoi in zip(observed_imgs, observed_bbsois):
            assert observed_bbsoi.shape == img.shape

            observed_img_x = np.max(observed_img, axis=0)
            observed_img_y = np.max(observed_img, axis=1)

            nonz_x = np.nonzero(observed_img_x)[0]
            nonz_y = np.nonzero(observed_img_y)[0]

            img_x1 = min(nonz_x)
            img_x2 = max(nonz_x)
            img_y1 = min(nonz_y)
            img_y2 = max(nonz_y)
            expected = ia.BoundingBox(x1=img_x1, y1=img_y1,
                                      x2=img_x2, y2=img_y2)

            for bb_aug in observed_bbsoi.bounding_boxes:
                # we don't expect perfect IoU here, because the actual
                # underlying KP aug used distance maps
                # most IoUs seem to end up in the range 0.9-0.95
                assert bb_aug.iou(expected) > 0.8

    def test_scale_is_list(self):
        aug1 = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
        aug2 = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
        aug = iaa.PiecewiseAffine(scale=[0.01, 0.10], nb_rows=12, nb_cols=4)

        avg1 = np.average([
            np.average(
                aug1.augment_image(self.image)
                * (~self.mask).astype(np.float32)
            )
            for _ in sm.xrange(3)
        ])
        avg2 = np.average([
            np.average(
                aug2.augment_image(self.image)
                * (~self.mask).astype(np.float32)
            )
            for _ in sm.xrange(3)
        ])
        seen = [0, 0]
        for _ in sm.xrange(15):
            observed = aug.augment_image(self.image)

            avg = np.average(observed * (~self.mask).astype(np.float32))
            diff1 = abs(avg - avg1)
            diff2 = abs(avg - avg2)
            if diff1 < diff2:
                seen[0] += 1
            else:
                seen[1] += 1
        assert seen[0] > 0
        assert seen[1] > 0

    # -----
    # rows and cols
    # -----
    @classmethod
    def _compute_observed_std_ygrad_in_mask(cls, observed, mask):
        grad_vert = (
                observed[1:, :].astype(np.float32)
                - observed[:-1, :].astype(np.float32)
            )
        grad_vert = grad_vert * (~mask[1:, :]).astype(np.float32)
        return np.std(grad_vert)

    def _compute_std_ygrad_in_mask(self, aug, image, mask, nb_iterations):
        stds = []
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_image(image)

            stds.append(
                self._compute_observed_std_ygrad_in_mask(observed, mask)
            )
        return np.average(stds)

    def test_nb_rows_affects_images(self):
        # verify effects of rows
        aug1 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=4)
        aug2 = iaa.PiecewiseAffine(scale=0.05, nb_rows=30, nb_cols=4)

        std1 = self._compute_std_ygrad_in_mask(aug1, self.image, self.mask, 3)
        std2 = self._compute_std_ygrad_in_mask(aug2, self.image, self.mask, 3)

        assert std1 < std2

    def test_nb_rows_is_list_affects_images(self):
        # rows as list
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=[4, 20], nb_cols=4)
        aug1 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=4)
        aug2 = iaa.PiecewiseAffine(scale=0.05, nb_rows=30, nb_cols=4)

        std1 = self._compute_std_ygrad_in_mask(aug1, self.image, self.mask, 3)
        std2 = self._compute_std_ygrad_in_mask(aug2, self.image, self.mask, 3)

        seen = [0, 0]
        for _ in sm.xrange(20):
            observed = aug.augment_image(self.image)

            std = self._compute_observed_std_ygrad_in_mask(observed, self.mask)
            diff1 = abs(std - std1)
            diff2 = abs(std - std2)
            if diff1 < diff2:
                seen[0] += 1
            else:
                seen[1] += 1
        assert seen[0] > 0
        assert seen[1] > 0

    def test_nb_cols_affects_images(self):
        # verify effects of cols
        image = self.image.T
        mask = self.mask.T

        aug1 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=4)
        aug2 = iaa.PiecewiseAffine(scale=0.05, nb_rows=20, nb_cols=4)

        std1 = self._compute_std_ygrad_in_mask(aug1, image, mask, 3)
        std2 = self._compute_std_ygrad_in_mask(aug2, image, mask, 3)

        assert std1 < std2

    def test_nb_cols_is_list_affects_images(self):
        # cols as list
        image = self.image.T
        mask = self.mask.T

        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=[4, 20])
        aug1 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=4)
        aug2 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=30)

        std1 = self._compute_std_ygrad_in_mask(aug1, image, mask, 3)
        std2 = self._compute_std_ygrad_in_mask(aug2, image, mask, 3)

        seen = [0, 0]
        for _ in sm.xrange(20):
            observed = aug.augment_image(image)

            std = self._compute_observed_std_ygrad_in_mask(observed, mask)
            diff1 = abs(std - std1)
            diff2 = abs(std - std2)
            if diff1 < diff2:
                seen[0] += 1
            else:
                seen[1] += 1
        assert seen[0] > 0
        assert seen[1] > 0

    # -----
    # order
    # -----
    # TODO

    # -----
    # cval
    # -----
    def test_cval_is_zero(self):
        # since scikit-image 0.16.2 and scipy 1.4.0(!), this test requires
        # several iterations to find one image that required filling with cval
        found = False
        for _ in np.arange(50):
            img = np.zeros((16, 16, 3), dtype=np.uint8) + 255
            aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=10, nb_cols=10,
                                      mode="constant", cval=0)
            observed = aug.augment_image(img)
            if np.sum([observed[:, :] == [0, 0, 0]]) > 0:
                found = True
                break
        assert found

    def test_cval_should_be_ignored_by_heatmaps(self):
        # cval as deterministic, heatmaps should always use cval=0
        heatmaps = HeatmapsOnImage(
            np.zeros((50, 50, 1), dtype=np.float32), shape=(50, 50, 3))
        aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=10, nb_cols=10,
                                  mode="constant", cval=255)
        observed = aug.augment_heatmaps([heatmaps])[0]
        assert np.sum([observed.get_arr()[:, :] >= 0.01]) == 0

    def test_cval_should_be_ignored_by_segmaps(self):
        # cval as deterministic, segmaps should always use cval=0
        segmaps = SegmentationMapsOnImage(
            np.zeros((50, 50, 1), dtype=np.int32), shape=(50, 50, 3))
        aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=10, nb_cols=10,
                                  mode="constant", cval=255)
        observed = aug.augment_segmentation_maps([segmaps])[0]
        assert np.sum([observed.get_arr()[:, :] > 0]) == 0

    def test_cval_is_list(self):
        # cval as list
        img = np.zeros((20, 20), dtype=np.uint8) + 255
        aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=5, nb_cols=5,
                                  mode="constant", cval=[0, 10])

        seen = [0, 0, 0]
        for _ in sm.xrange(30):
            observed = aug.augment_image(img)
            nb_0 = np.sum([observed[:, :] == 0])
            nb_10 = np.sum([observed[:, :] == 10])
            if nb_0 > 0:
                seen[0] += 1
            elif nb_10 > 0:
                seen[1] += 1
            else:
                seen[2] += 1
        assert seen[0] > 5
        assert seen[1] > 5
        assert seen[2] <= 4

    # -----
    # mode
    # -----
    # TODO

    # ---------
    # remaining keypoints tests
    # ---------
    def test_keypoints_outside_of_image(self):
        # keypoints outside of image
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=10, nb_cols=10)
        kps = [ia.Keypoint(x=-10, y=-20)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(10, 10, 3))

        observed = aug.augment_keypoints(kpsoi)

        assert_cbaois_equal(observed, kpsoi)

    def test_keypoints_empty(self):
        # empty keypoints
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=10, nb_cols=10)
        kpsoi = ia.KeypointsOnImage([], shape=(10, 10, 3))

        observed = aug.augment_keypoints(kpsoi)

        assert_cbaois_equal(observed, kpsoi)

    # ---------
    # remaining polygons tests
    # ---------
    def test_polygons_outside_of_image(self):
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=10, nb_cols=10)
        exterior = [(-10, -10), (110, -10), (110, 90), (-10, 90)]
        poly = ia.Polygon(exterior)
        psoi = ia.PolygonsOnImage([poly], shape=(10, 10, 3))

        observed = aug.augment_polygons(psoi)

        assert_cbaois_equal(observed, psoi)

    def test_empty_polygons(self):
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=10, nb_cols=10)
        psoi = ia.PolygonsOnImage([], shape=(10, 10, 3))

        observed = aug.augment_polygons(psoi)

        assert_cbaois_equal(observed, psoi)

    # ---------
    # remaining line string tests
    # ---------
    def test_line_strings_outside_of_image(self):
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=10, nb_cols=10)
        coords = [(-10, -10), (110, -10), (110, 90), (-10, 90)]
        ls = ia.LineString(coords)
        lsoi = ia.LineStringsOnImage([ls], shape=(10, 10, 3))

        observed = aug.augment_line_strings(lsoi)

        assert_cbaois_equal(observed, lsoi)

    def test_empty_line_strings(self):
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=10, nb_cols=10)
        lsoi = ia.LineStringsOnImage([], shape=(10, 10, 3))

        observed = aug.augment_line_strings(lsoi)

        assert_cbaois_equal(observed, lsoi)

    # ---------
    # remaining bounding box tests
    # ---------
    def test_bounding_boxes_outside_of_image(self):
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=10, nb_cols=10)
        bbs = ia.BoundingBox(x1=-10, y1=-10, x2=15, y2=15)
        bbsoi = ia.BoundingBoxesOnImage([bbs], shape=(10, 10, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        assert_cbaois_equal(observed, bbsoi)

    def test_empty_bounding_boxes(self):
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=10, nb_cols=10)
        bbsoi = ia.BoundingBoxesOnImage([], shape=(10, 10, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        assert_cbaois_equal(observed, bbsoi)

    # ---------
    # zero-sized axes
    # ---------
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
                aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=2, nb_cols=2)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_zero_sized_axes_absolute_scale(self):
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
                aug = iaa.PiecewiseAffine(scale=5, nb_rows=2, nb_cols=2,
                                          absolute_scale=True)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    # ---------
    # other methods
    # ---------
    def test_get_parameters(self):
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=10, order=1,
                                  cval=2, mode="constant",
                                  absolute_scale=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert isinstance(params[2], iap.Deterministic)
        assert isinstance(params[3], iap.Deterministic)
        assert isinstance(params[4], iap.Deterministic)
        assert isinstance(params[5], iap.Deterministic)
        assert params[6] is False
        assert 0.1 - 1e-8 < params[0].value < 0.1 + 1e-8
        assert params[1].value == 8
        assert params[2].value == 10
        assert params[3].value == 1
        assert params[4].value == 2
        assert params[5].value == "constant"

    # ---------
    # other dtypes
    # ---------
    @property
    def other_dtypes_mask(self):
        mask = np.zeros((21, 21), dtype=bool)
        mask[:, 7:13] = True
        return mask

    def test_other_dtypes_bool(self):
        aug = iaa.PiecewiseAffine(scale=0.2, nb_rows=8, nb_cols=4, order=0,
                                  mode="constant")

        image = np.zeros((21, 21), dtype=bool)
        image[self.other_dtypes_mask] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == image.dtype.name
        assert not np.all(image_aug == 1)
        assert np.any(image_aug[~self.other_dtypes_mask] == 1)

    def test_other_dtypes_uint_int(self):
        aug = iaa.PiecewiseAffine(scale=0.2, nb_rows=8, nb_cols=4, order=0,
                                  mode="constant")

        dtypes = ["uint8", "uint16", "uint32", "int8", "int16", "int32"]
        for dtype in dtypes:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            if np.dtype(dtype).kind == "i":
                values = [1, 5, 10, 100, int(0.1 * max_value),
                          int(0.2 * max_value), int(0.5 * max_value),
                          max_value-100, max_value]
                values = values + [(-1)*value for value in values]
            else:
                values = [1, 5, 10, 100, int(center_value),
                          int(0.1 * max_value), int(0.2 * max_value),
                          int(0.5 * max_value), max_value-100, max_value]

            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((21, 21), dtype=dtype)
                    image[:, 7:13] = value

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert not np.all(image_aug == value)
                    assert np.any(image_aug[~self.other_dtypes_mask] == value)

    def test_other_dtypes_float(self):
        aug = iaa.PiecewiseAffine(scale=0.2, nb_rows=8, nb_cols=4, order=0,
                                  mode="constant")

        dtypes = ["float16", "float32", "float64"]
        for dtype in dtypes:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            def _isclose(a, b):
                atol = 1e-4 if dtype == "float16" else 1e-8
                return np.isclose(a, b, atol=atol, rtol=0)

            isize = np.dtype(dtype).itemsize
            values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1),
                      1000 ** (isize - 1)]
            values = values + [(-1) * value for value in values]
            values = values + [min_value, max_value]
            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((21, 21), dtype=dtype)
                    image[:, 7:13] = value

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    # TODO switch all other tests from float(...) to
                    #      np.float128(...) pattern, seems to be more accurate
                    #      for 128bit floats
                    assert not np.all(_isclose(image_aug, np.float128(value)))
                    assert np.any(_isclose(image_aug[~self.other_dtypes_mask],
                                           np.float128(value)))

    def test_pickleable(self):
        aug = iaa.PiecewiseAffine(scale=0.2, nb_rows=4, nb_cols=4,
                                  random_state=1)
        runtest_pickleable_uint8_img(aug, iterations=3, shape=(25, 25, 1))


class TestPerspectiveTransform(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        img = np.zeros((30, 30), dtype=np.uint8)
        img[10:20, 10:20] = 255
        return img

    @property
    def heatmaps(self):
        return HeatmapsOnImage((self.image / 255.0).astype(np.float32),
                               shape=self.image.shape)

    @property
    def segmaps(self):
        return SegmentationMapsOnImage((self.image > 0).astype(np.int32),
                                       shape=self.image.shape)

    # --------
    # __init__
    # --------
    def test___init___scale_is_tuple(self):
        # tuple for scale
        aug = iaa.PerspectiveTransform(scale=(0.1, 0.2))
        assert isinstance(aug.jitter.scale, iap.Uniform)
        assert isinstance(aug.jitter.scale.a, iap.Deterministic)
        assert isinstance(aug.jitter.scale.b, iap.Deterministic)
        assert 0.1 - 1e-8 < aug.jitter.scale.a.value < 0.1 + 1e-8
        assert 0.2 - 1e-8 < aug.jitter.scale.b.value < 0.2 + 1e-8

    def test___init___scale_is_list(self):
        # list for scale
        aug = iaa.PerspectiveTransform(scale=[0.1, 0.2, 0.3])
        assert isinstance(aug.jitter.scale, iap.Choice)
        assert len(aug.jitter.scale.a) == 3
        assert 0.1 - 1e-8 < aug.jitter.scale.a[0] < 0.1 + 1e-8
        assert 0.2 - 1e-8 < aug.jitter.scale.a[1] < 0.2 + 1e-8
        assert 0.3 - 1e-8 < aug.jitter.scale.a[2] < 0.3 + 1e-8

    def test___init___scale_is_stochastic_parameter(self):
        # StochasticParameter for scale
        aug = iaa.PerspectiveTransform(scale=iap.Choice([0.1, 0.2, 0.3]))
        assert isinstance(aug.jitter.scale, iap.Choice)
        assert len(aug.jitter.scale.a) == 3
        assert 0.1 - 1e-8 < aug.jitter.scale.a[0] < 0.1 + 1e-8
        assert 0.2 - 1e-8 < aug.jitter.scale.a[1] < 0.2 + 1e-8
        assert 0.3 - 1e-8 < aug.jitter.scale.a[2] < 0.3 + 1e-8

    def test___init___bad_datatype_for_scale_leads_to_failure(self):
        # bad datatype for scale
        got_exception = False
        try:
            _ = iaa.PerspectiveTransform(scale=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test___init___mode_is_all(self):
        aug = iaa.PerspectiveTransform(cval=0, mode=ia.ALL)
        assert isinstance(aug.mode, iap.Choice)

    def test___init___mode_is_string(self):
        aug = iaa.PerspectiveTransform(cval=0, mode="replicate")
        assert isinstance(aug.mode, iap.Deterministic)
        assert aug.mode.value == "replicate"

    def test___init___mode_is_list(self):
        aug = iaa.PerspectiveTransform(cval=0, mode=["replicate", "constant"])
        assert isinstance(aug.mode, iap.Choice)
        assert (
            len(aug.mode.a) == 2
            and "replicate" in aug.mode.a
            and "constant" in aug.mode.a)

    def test___init___mode_is_stochastic_parameter(self):
        aug = iaa.PerspectiveTransform(
            cval=0, mode=iap.Choice(["replicate", "constant"]))
        assert isinstance(aug.mode, iap.Choice)
        assert (
            len(aug.mode.a) == 2
            and "replicate" in aug.mode.a
            and "constant" in aug.mode.a)

    # --------
    # image, heatmaps, segmaps
    # --------
    def test_image_without_keep_size(self):
        # without keep_size
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)

        observed = aug.augment_image(self.image)

        y1 = int(30*0.2)
        y2 = int(30*0.8)
        x1 = int(30*0.2)
        x2 = int(30*0.8)

        expected = self.image[y1:y2, x1:x2]
        assert all([
            abs(s1-s2) <= 1 for s1, s2 in zip(observed.shape, expected.shape)
        ])
        if observed.shape != expected.shape:
            observed = ia.imresize_single_image(
                observed, expected.shape[0:2], interpolation="cubic")
        # differences seem to mainly appear around the border of the inner
        # rectangle, possibly due to interpolation
        assert np.average(
            np.abs(observed.astype(np.int32) - expected.astype(np.int32))
        ) < 30.0

    def test_image_heatmaps_alignment_without_keep_size(self):
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)
        hm = HeatmapsOnImage(
            self.image.astype(np.float32)/255.0,
            shape=(30, 30)
        )

        observed = aug.augment_image(self.image)
        hm_aug = aug.augment_heatmaps([hm])[0]

        y1 = int(30*0.2)
        y2 = int(30*0.8)
        x1 = int(30*0.2)
        x2 = int(30*0.8)

        expected = (y2 - y1, x2 - x1)
        assert all([
            abs(s1-s2) <= 1
            for s1, s2
            in zip(hm_aug.shape, expected)
        ])
        assert all([
            abs(s1-s2) <= 1
            for s1, s2
            in zip(hm_aug.arr_0to1.shape, expected + (1,))
        ])
        img_aug_mask = observed > 255*0.1
        hm_aug_mask = hm_aug.arr_0to1 > 0.1
        same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
        assert (same / img_aug_mask.size) >= 0.99

    def test_image_segmaps_alignment_without_keep_size(self):
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)
        segmaps = SegmentationMapsOnImage(
            (self.image > 100).astype(np.int32),
            shape=(30, 30)
        )

        observed = aug.augment_image(self.image)
        segmaps_aug = aug.augment_segmentation_maps([segmaps])[0]

        y1 = int(30*0.2)
        y2 = int(30*0.8)
        x1 = int(30*0.2)
        x2 = int(30*0.8)

        expected = (y2 - y1, x2 - x1)
        assert all([
            abs(s1-s2) <= 1
            for s1, s2
            in zip(segmaps_aug.shape, expected)
        ])
        assert all([
            abs(s1-s2) <= 1
            for s1, s2
            in zip(segmaps_aug.arr.shape, expected + (1,))
        ])
        img_aug_mask = observed > 255*0.5
        segmaps_aug_mask = segmaps_aug.arr > 0
        same = np.sum(img_aug_mask == segmaps_aug_mask[:, :, 0])
        assert (same / img_aug_mask.size) >= 0.99

    def test_heatmaps_smaller_than_image_without_keep_size(self):
        # without keep_size, different heatmap size
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)

        y1 = int(30*0.2)
        y2 = int(30*0.8)
        x1 = int(30*0.2)
        x2 = int(30*0.8)
        x1_small = int(25*0.2)
        x2_small = int(25*0.8)
        y1_small = int(20*0.2)
        y2_small = int(20*0.8)

        img_small = ia.imresize_single_image(
            self.image,
            (20, 25),
            interpolation="cubic")
        hm = ia.HeatmapsOnImage(
            img_small.astype(np.float32)/255.0,
            shape=(30, 30))

        img_aug = aug.augment_image(self.image)
        hm_aug = aug.augment_heatmaps([hm])[0]

        expected = (y2 - y1, x2 - x1)
        expected_small = (y2_small - y1_small, x2_small - x1_small, 1)
        assert all([
            abs(s1-s2) <= 1
            for s1, s2
            in zip(hm_aug.shape, expected)
        ])
        assert all([
            abs(s1-s2) <= 1
            for s1, s2
            in zip(hm_aug.arr_0to1.shape, expected_small)
        ])
        img_aug_mask = img_aug > 255*0.1
        hm_aug_mask = ia.imresize_single_image(
            hm_aug.arr_0to1, img_aug.shape[0:2], interpolation="cubic"
        ) > 0.1
        same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
        assert (same / img_aug_mask.size) >= 0.96

    def test_segmaps_smaller_than_image_without_keep_size(self):
        # without keep_size, different segmap size
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)

        y1 = int(30*0.2)
        y2 = int(30*0.8)
        x1 = int(30*0.2)
        x2 = int(30*0.8)
        x1_small = int(25*0.2)
        x2_small = int(25*0.8)
        y1_small = int(20*0.2)
        y2_small = int(20*0.8)

        img_small = ia.imresize_single_image(
            self.image,
            (20, 25),
            interpolation="cubic")
        seg = SegmentationMapsOnImage(
            (img_small > 100).astype(np.int32),
            shape=(30, 30))

        img_aug = aug.augment_image(self.image)
        seg_aug = aug.augment_segmentation_maps([seg])[0]

        expected = (y2 - y1, x2 - x1)
        expected_small = (y2_small - y1_small, x2_small - x1_small, 1)
        assert all([
            abs(s1-s2) <= 1
            for s1, s2
            in zip(seg_aug.shape, expected)
        ])
        assert all([
            abs(s1-s2) <= 1
            for s1, s2
            in zip(seg_aug.arr.shape, expected_small)
        ])
        img_aug_mask = img_aug > 255*0.5
        seg_aug_mask = ia.imresize_single_image(
            seg_aug.arr, img_aug.shape[0:2], interpolation="nearest") > 0
        same = np.sum(img_aug_mask == seg_aug_mask[:, :, 0])
        assert (same / img_aug_mask.size) >= 0.92

    def test_image_with_keep_size(self):
        # with keep_size
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
        aug.jitter = iap.Deterministic(0.2)

        observed = aug.augment_image(self.image)

        expected = self.image[int(30*0.2):int(30*0.8),
                              int(30*0.2):int(30*0.8)]
        expected = ia.imresize_single_image(
            expected,
            self.image.shape[0:2],
            interpolation="cubic")
        assert observed.shape == self.image.shape
        # differences seem to mainly appear around the border of the inner
        # rectangle, possibly due to interpolation
        assert np.average(
            np.abs(observed.astype(np.int32) - expected.astype(np.int32))
        ) < 30.0

    def test_heatmaps_with_keep_size(self):
        # with keep_size, heatmaps
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
        aug.jitter = iap.Deterministic(0.2)

        observed = aug.augment_heatmaps([self.heatmaps])[0]

        heatmaps_arr = self.heatmaps.get_arr()
        expected = heatmaps_arr[int(30*0.2):int(30*0.8),
                                int(30*0.2):int(30*0.8)]
        expected = ia.imresize_single_image(
            (expected*255).astype(np.uint8),
            self.image.shape[0:2],
            interpolation="cubic")
        expected = (expected / 255.0).astype(np.float32)
        assert observed.shape == self.heatmaps.shape
        _assert_same_min_max(observed, self.heatmaps)
        # differences seem to mainly appear around the border of the inner
        # rectangle, possibly due to interpolation
        assert np.average(np.abs(observed.get_arr() - expected)) < 30.0

    def test_segmaps_with_keep_size(self):
        # with keep_size, segmaps
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
        aug.jitter = iap.Deterministic(0.2)

        observed = aug.augment_segmentation_maps([self.segmaps])[0]

        segmaps_arr = self.segmaps.get_arr()
        expected = segmaps_arr[int(30*0.2):int(30*0.8),
                               int(30*0.2):int(30*0.8)]
        expected = ia.imresize_single_image(
            (expected*255).astype(np.uint8),
            self.image.shape[0:2],
            interpolation="cubic")
        expected = (expected > 255*0.5).astype(np.int32)
        assert observed.shape == self.segmaps.shape
        assert np.average(observed.get_arr() != expected) < 0.05

    def test_image_rgb_with_keep_size(self):
        # with keep_size, RGB images
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
        aug.jitter = iap.Deterministic(0.2)
        imgs = np.tile(self.image[np.newaxis, :, :, np.newaxis], (2, 1, 1, 3))

        observed = aug.augment_images(imgs)

        for img_idx in sm.xrange(2):
            for c in sm.xrange(3):
                observed_i = observed[img_idx, :, :, c]
                expected = imgs[img_idx,
                                int(30*0.2):int(30*0.8),
                                int(30*0.2):int(30*0.8),
                                c]
                expected = ia.imresize_single_image(
                    expected, imgs.shape[1:3], interpolation="cubic")
                assert observed_i.shape == imgs.shape[1:3]
                # differences seem to mainly appear around the border of the
                # inner rectangle, possibly due to interpolation
                assert np.average(
                    np.abs(
                        observed_i.astype(np.int32) - expected.astype(np.int32)
                    )
                ) < 30.0

    # --------
    # keypoints
    # --------
    def test_keypoints_without_keep_size(self):
        # keypoint augmentation without keep_size
        # TODO deviations of around 0.4-0.7 in this and the next test (between
        #      expected and observed coordinates) -- why?
        kps = [ia.Keypoint(x=10, y=10), ia.Keypoint(x=14, y=11)]
        kpsoi = ia.KeypointsOnImage(kps, shape=self.image.shape)
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)

        observed = aug.augment_keypoints([kpsoi])

        kps_expected = [
            ia.Keypoint(x=10-0.2*30, y=10-0.2*30),
            ia.Keypoint(x=14-0.2*30, y=11-0.2*30)
        ]
        gen = zip(observed[0].keypoints, kps_expected)
        # TODO deviations of around 0.5 here from expected values, why?
        for kp_observed, kp_expected in gen:
            assert kp_observed.coords_almost_equals(
                kp_expected, max_distance=1.5)

    def test_keypoints_with_keep_size(self):
        # keypoint augmentation with keep_size
        kps = [ia.Keypoint(x=10, y=10), ia.Keypoint(x=14, y=11)]
        kpsoi = ia.KeypointsOnImage(kps, shape=self.image.shape)
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
        aug.jitter = iap.Deterministic(0.2)

        observed = aug.augment_keypoints([kpsoi])

        kps_expected = [
            ia.Keypoint(x=((10-0.2*30)/(30*0.6))*30,
                        y=((10-0.2*30)/(30*0.6))*30),
            ia.Keypoint(x=((14-0.2*30)/(30*0.6))*30,
                        y=((11-0.2*30)/(30*0.6))*30)
        ]
        gen = zip(observed[0].keypoints, kps_expected)
        # TODO deviations of around 0.5 here from expected values, why?
        for kp_observed, kp_expected in gen:
            assert kp_observed.coords_almost_equals(
                kp_expected, max_distance=1.5)

    def test_image_keypoint_alignment(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25-3:25+3, 25-3:25+3] = 255
        img[50-3:50+3, 25-3:25+3] = 255
        img[75-3:75+3, 25-3:25+3] = 255
        img[25-3:25+3, 75-3:75+3] = 255
        img[50-3:50+3, 75-3:75+3] = 255
        img[75-3:75+3, 75-3:75+3] = 255
        img[50-3:75+3, 50-3:75+3] = 255
        kps = [
            ia.Keypoint(y=25, x=25), ia.Keypoint(y=50, x=25),
            ia.Keypoint(y=75, x=25), ia.Keypoint(y=25, x=75),
            ia.Keypoint(y=50, x=75), ia.Keypoint(y=75, x=75),
            ia.Keypoint(y=50, x=50)
        ]
        kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)
        aug = iaa.PerspectiveTransform(scale=(0.05, 0.15), keep_size=True)

        for _ in sm.xrange(10):
            aug_det = aug.to_deterministic()
            imgs_aug = aug_det.augment_images([img, img])
            kpsois_aug = aug_det.augment_keypoints([kpsoi, kpsoi])

            for img_aug, kpsoi_aug in zip(imgs_aug, kpsois_aug):
                assert kpsoi_aug.shape == img.shape
                for kp_aug in kpsoi_aug.keypoints:
                    x, y = int(np.round(kp_aug.x)), int(np.round(kp_aug.y))
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        assert img_aug[y, x] > 10

    def test_empty_keypoints(self):
        # test empty keypoints
        kpsoi = ia.KeypointsOnImage([], shape=(20, 10, 3))
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)

        observed = aug.augment_keypoints(kpsoi)

        assert_cbaois_equal(observed, kpsoi)

    # --------
    # abstract test methods for polygons and line strings
    # --------
    @classmethod
    def _test_cbaois_without_keep_size(cls, cba_class, cbaoi_class, augf_name):
        points = np.float32([
            [10, 10],
            [25, 10],
            [25, 25],
            [10, 25]
        ])
        cbaoi = cbaoi_class([cba_class(points)], shape=(30, 30, 3))
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)

        observed = getattr(aug, augf_name)(cbaoi)

        assert observed.shape == (30 - 12, 30 - 12, 3)
        assert len(observed.items) == 1
        if hasattr(observed.items[0], "is_valid"):
            assert observed.items[0].is_valid

        points_expected = np.copy(points)
        points_expected[:, 0] -= 0.2 * 30
        points_expected[:, 1] -= 0.2 * 30
        # TODO deviations of around 0.5 here from expected values, why?
        assert observed.items[0].coords_almost_equals(
            points_expected, max_distance=1.5)

    @classmethod
    def _test_cbaois_with_keep_size(cls, cba_class, cbaoi_class, augf_name):
        # polygon augmentation with keep_size
        points = np.float32([
            [10, 10],
            [25, 10],
            [25, 25],
            [10, 25]
        ])
        cbaoi = cbaoi_class([cba_class(points)], shape=(30, 30, 3))
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
        aug.jitter = iap.Deterministic(0.2)

        observed = getattr(aug, augf_name)(cbaoi)

        assert observed.shape == (30, 30, 3)
        assert len(observed.items) == 1
        if hasattr(observed.items[0], "is_valid"):
            assert observed.items[0].is_valid

        points_expected = np.copy(points)
        points_expected[:, 0] = (
            (points_expected[:, 0] - 0.2 * 30) / (30 * 0.6)
        ) * 30
        points_expected[:, 1] = (
            (points_expected[:, 1] - 0.2 * 30) / (30 * 0.6)
        ) * 30
        # TODO deviations of around 0.5 here from expected values, why?
        assert observed.items[0].coords_almost_equals(
            points_expected, max_distance=2.5)

    @classmethod
    def _test_image_cba_alignment(cls, cba_class, cbaoi_class, augf_name):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25-3:25+3, 25-3:25+3] = 255
        img[50-3:50+3, 25-3:25+3] = 255
        img[75-3:75+3, 25-3:25+3] = 255
        img[25-3:25+3, 75-3:75+3] = 255
        img[50-3:50+3, 75-3:75+3] = 255
        img[75-3:75+3, 75-3:75+3] = 255
        points = [
            [25, 25],
            [75, 25],
            [75, 50],
            [75, 75],
            [25, 75],
            [25, 50]
        ]

        cbaoi = cbaoi_class([cba_class(points)], shape=img.shape)
        aug = iaa.PerspectiveTransform(scale=0.1, keep_size=True)
        for _ in sm.xrange(10):
            aug_det = aug.to_deterministic()
            imgs_aug = aug_det.augment_images([img] * 4)
            cbaois_aug = getattr(aug_det, augf_name)([cbaoi] * 4)

            for img_aug, cbaoi_aug in zip(imgs_aug, cbaois_aug):
                assert cbaoi_aug.shape == img.shape
                for cba_aug in cbaoi_aug.items:
                    if hasattr(cba_aug, "is_valid"):
                        assert cba_aug.is_valid
                    for x, y in cba_aug.coords:
                        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                            bb = ia.BoundingBox(x1=x-2, x2=x+2, y1=y-2, y2=y+2)
                            img_ex = bb.extract_from_image(img_aug)
                            assert np.any(img_ex > 10)

    @classmethod
    def _test_empty_cba(cls, cbaoi, augf_name):
        # test empty polygons
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)

        observed = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(observed, cbaoi)

    # --------
    # polygons
    # --------
    def test_polygons_without_keep_size(self):
        self._test_cbaois_without_keep_size(ia.Polygon, ia.PolygonsOnImage,
                                            "augment_polygons")

    def test_polygons_with_keep_size(self):
        self._test_cbaois_with_keep_size(ia.Polygon, ia.PolygonsOnImage,
                                         "augment_polygons")

    def test_image_polygon_alignment(self):
        self._test_image_cba_alignment(ia.Polygon, ia.PolygonsOnImage,
                                       "augment_polygons")

    def test_empty_polygons(self):
        psoi = ia.PolygonsOnImage([], shape=(20, 10, 3))
        self._test_empty_cba(psoi, "augment_polygons")

    def test_polygons_under_extreme_scale_values(self):
        # test extreme scales
        # TODO when setting .min_height and .min_width in PerspectiveTransform
        #      to 1x1, at least one of the output polygons was invalid and had
        #      only 3 instead of the expected 4 points - why?
        for scale in [0.1, 0.2, 0.3, 0.4]:
            with self.subTest(scale=scale):
                exterior = np.float32([
                    [10, 10],
                    [25, 10],
                    [25, 25],
                    [10, 25]
                ])
                psoi = ia.PolygonsOnImage([ia.Polygon(exterior)],
                                          shape=(30, 30, 3))
                aug = iaa.PerspectiveTransform(scale=scale, keep_size=True)
                aug.jitter = iap.Deterministic(scale)

                observed = aug.augment_polygons(psoi)

                assert observed.shape == (30, 30, 3)
                assert len(observed.polygons) == 1
                assert observed.polygons[0].is_valid

                # FIXME this part is currently deactivated due to too large
                #       deviations from expectations. As the alignment check
                #       works, this is probably some error on the test side
                """
                exterior_expected = np.copy(exterior)
                exterior_expected[:, 0] = (
                    (exterior_expected[:, 0] - scale * 30) / (30*(1-2*scale))
                ) * 30
                exterior_expected[:, 1] = (
                    (exterior_expected[:, 1] - scale * 30) / (30*(1-2*scale))
                ) * 30
                poly0 = observed.polygons[0]
                # TODO deviations of around 0.5 here from expected values, why?
                assert poly0.exterior_almost_equals(
                    exterior_expected, max_distance=2.0)
                """

    # --------
    # line strings
    # --------
    def test_line_strings_without_keep_size(self):
        self._test_cbaois_without_keep_size(ia.LineString, ia.LineStringsOnImage,
                                            "augment_line_strings")

    def test_line_strings_with_keep_size(self):
        self._test_cbaois_with_keep_size(ia.LineString, ia.LineStringsOnImage,
                                         "augment_line_strings")

    def test_image_line_string_alignment(self):
        self._test_image_cba_alignment(ia.LineString, ia.LineStringsOnImage,
                                       "augment_line_strings")

    def test_empty_line_strings(self):
        lsoi = ia.LineStringsOnImage([], shape=(20, 10, 3))
        self._test_empty_cba(lsoi, "augment_line_strings")

    # --------
    # bounding boxes
    # --------
    def test_bounding_boxes_without_keep_size(self):
        # BB augmentation without keep_size
        # TODO deviations of around 0.4-0.7 in this and the next test (between
        #      expected and observed coordinates) -- why?
        bbs = [ia.BoundingBox(x1=0, y1=10, x2=20, y2=20)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)

        observed = aug.augment_bounding_boxes([bbsoi])

        bbs_expected = [
            ia.BoundingBox(x1=0-0.2*30, y1=10-0.2*30,
                           x2=20-0.2*30, y2=20-0.2*30)
        ]
        gen = zip(observed[0].bounding_boxes, bbs_expected)
        # TODO deviations of around 0.5 here from expected values, why?
        for bb_observed, bb_expected in gen:
            assert bb_observed.coords_almost_equals(
                bb_expected, max_distance=1.5)

    def test_bounding_boxes_with_keep_size(self):
        # BB augmentation with keep_size
        bbs = [ia.BoundingBox(x1=0, y1=10, x2=20, y2=20)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
        aug.jitter = iap.Deterministic(0.2)

        observed = aug.augment_bounding_boxes([bbsoi])

        bbs_expected = [
            ia.BoundingBox(
                x1=((0-0.2*30)/(30*0.6))*30,
                y1=((10-0.2*30)/(30*0.6))*30,
                x2=((20-0.2*30)/(30*0.6))*30,
                y2=((20-0.2*30)/(30*0.6))*30
            )
        ]
        gen = zip(observed[0].bounding_boxes, bbs_expected)
        # TODO deviations of around 0.5 here from expected values, why?
        for bb_observed, bb_expected in gen:
            assert bb_observed.coords_almost_equals(
                bb_expected, max_distance=1.5)

    def test_image_bounding_box_alignment(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[35:35+1, 35:65+1] = 255
        img[65:65+1, 35:65+1] = 255
        img[35:65+1, 35:35+1] = 255
        img[35:65+1, 65:65+1] = 255
        bbs = [
            ia.BoundingBox(y1=35.5, x1=35.5, y2=65.5, x2=65.5),
        ]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=img.shape)
        aug = iaa.PerspectiveTransform(scale=(0.05, 0.2), keep_size=True)

        for _ in sm.xrange(10):
            imgs_aug, bbsois_aug = aug(
                images=[img, img, img, img],
                bounding_boxes=[bbsoi, bbsoi, bbsoi, bbsoi])

            nb_skipped = 0
            for img_aug, bbsoi_aug in zip(imgs_aug, bbsois_aug):
                assert bbsoi_aug.shape == img_aug.shape
                for bb_aug in bbsoi_aug.bounding_boxes:
                    if bb_aug.is_fully_within_image(img_aug):
                        # top, bottom, left, right
                        x1 = bb_aug.x1_int
                        x2 = bb_aug.x2_int
                        y1 = bb_aug.y1_int
                        y2 = bb_aug.y2_int
                        top_row = img_aug[y1-1:y1+1, x1-1:x2+1]
                        btm_row = img_aug[y2-1:y2+1, x1-1:x2+1]
                        lft_row = img_aug[y1-1:y2+1, x1-1:x1+1]
                        rgt_row = img_aug[y1-1:y2+1, x2-1:x2+1]
                        assert np.max(top_row) > 10
                        assert np.max(btm_row) > 10
                        assert np.max(lft_row) > 10
                        assert np.max(rgt_row) > 10
                    else:
                        nb_skipped += 1
            assert nb_skipped <= 2

    def test_bounding_boxes_cover_extreme_points(self):
        # Test that for BBs, the augmented BB x coord is really the minimum
        # of the BB corner x-coords after augmentation and e.g. not just always
        # the augmented top-left corner's coordinate.
        h = w = 200  # height, width
        s = 5  # block size
        j_r = 0.1  # relative amount of jitter
        j = int(h * j_r)  # absolute amount of jitter

        # Note that PerspectiveTransform currently places four points on the
        # image and back-projects to the image size (roughly).
        # That's why e.g. TopWiderThanBottom has coordinates that seem like
        # the top is thinner than the bottom (after projecting back to the
        # image rectangle, the top becomes wider).
        class _JitterTopWiderThanBottom(object):
            def draw_samples(self, size, random_state):
                return np.float32([
                    [
                        [j_r, 0.0],  # top-left
                        [j_r, 0.0],  # top-right
                        [0.0, 0.0],  # bottom-right
                        [0.0, 0.0],  # bottom-left
                    ]
                ])

        class _JitterTopThinnerThanBottom(object):
            def draw_samples(self, size, random_state):
                return np.float32([
                    [
                        [0.0, 0.0],  # top-left
                        [0.0, 0.0],  # top-right
                        [j_r, 0.0],  # bottom-right
                        [j_r, 0.0],  # bottom-left
                    ]
                ])

        class _JitterLeftWiderThanRight(object):
            def draw_samples(self, size, random_state):
                return np.float32([
                    [
                        [0.0, j_r],  # top-left
                        [0.0, 0.0],  # top-right
                        [0.0, 0.0],  # bottom-right
                        [0.0, j_r],  # bottom-left
                    ]
                ])

        class _JitterLeftThinnerThanRight(object):
            def draw_samples(self, size, random_state):
                return np.float32([
                    [
                        [0.0, 0.0],  # top-left
                        [0.0, j_r],  # top-right
                        [0.0, j_r],  # bottom-right
                        [0.0, 0.0],  # bottom-left
                    ]
                ])

        jitters = [
            _JitterTopWiderThanBottom(),
            _JitterTopThinnerThanBottom(),
            _JitterLeftWiderThanRight(),
            _JitterLeftThinnerThanRight(),
        ]

        # expected coordinates after applying the above jitter
        # coordinates here are given as
        #   (ystart, yend), (xstart, xend)
        coords = [
            # top wider than bottom
            [
                [(0+j, s+j+1), (0, s+1)],  # top left
                [(0+j, s+j+1), (w-s, w+1)],  # top right
                [(h-s-j, h-j+1), (w-s-j, w-j+1)],  # bottom right
                [(h-s-j, h-j+1), (0+j, s+j+1)]  # bottom left
            ],
            # top thinner than bottom
            [
                [(0+j, s+j+1), (0+j, s+j+1)],
                [(0+j, s+j+1), (w-s-j, w-j+1)],
                [(h-s-j, h-j+1), (w-s, w+1)],
                [(h-s-j, h-j+1), (0, s+1)]
            ],
            # left wider than right
            [
                [(0, s+1), (0+j, s+j+1)],
                [(0+j, s+j+1), (w-s-j, w-j+1)],
                [(h-s-j, h-j+1), (w-s-j, w-j+1)],
                [(h-s, h+1), (0+j, s+j+1)]
            ],
            # left thinner than right
            [
                [(0+j, s+j+1), (0+j, s+j+1)],
                [(0, s+1), (w-s-j, w-j+1)],
                [(h-s, h+1), (w-s-j, w-j+1)],
                [(h-s-j, h-j+1), (0+j, s+j+1)]
            ],
        ]

        image = np.zeros((h-1, w-1, 4), dtype=np.uint8)
        image = iaa.pad(image, top=1, right=1, bottom=1, left=1, cval=50)
        image[0+j:s+j+1, 0+j:s+j+1, 0] = 255
        image[0+j:s+j+1, w-s-j:w-j+1, 1] = 255
        image[h-s-j:h-j+1, w-s-j:w-j+1, 2] = 255
        image[h-s-j:h-j+1, 0+j:s+j+1, 3] = 255

        bb = ia.BoundingBox(x1=0.0+j,
                            y1=0.0+j,
                            x2=w-j,
                            y2=h-j)
        bbsoi = ia.BoundingBoxesOnImage([bb], shape=image.shape)

        i = 0
        for jitter, coords_i in zip(jitters, coords):
            with self.subTest(jitter=jitter.__class__.__name__):
                aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
                aug.jitter = jitter

                image_aug, bbsoi_aug = aug(image=image, bounding_boxes=bbsoi)
                assert image_aug.shape == image.shape

                (tl_y1, tl_y2), (tl_x1, tl_x2) = coords_i[0]
                (tr_y1, tr_y2), (tr_x1, tr_x2) = coords_i[1]
                (br_y1, br_y2), (br_x1, br_x2) = coords_i[2]
                (bl_y1, bl_y2), (bl_x1, bl_x2) = coords_i[3]

                # We have to be rather tolerant here (>100 instead of e.g.
                # >200), because the transformation seems to be not that
                # accurate and the blobs may be a few pixels off the expected
                # coorindates.
                assert np.max(image_aug[tl_y1:tl_y2, tl_x1:tl_x2, 0]) > 100
                assert np.max(image_aug[tr_y1:tr_y2, tr_x1:tr_x2, 1]) > 100
                assert np.max(image_aug[br_y1:br_y2, br_x1:br_x2, 2]) > 100
                assert np.max(image_aug[bl_y1:bl_y2, bl_x1:bl_x2, 3]) > 100

                # We have rather strong tolerances of 7.5 here, partially
                # because the blobs are wide and the true coordinates are in
                # the center of the blobs; partially, because of above
                # mentioned inaccuracy of PerspectiveTransform.
                bb_aug = bbsoi_aug.bounding_boxes[0]
                exp_x1 = min([tl_x1, tr_x1, br_x1, bl_x1])
                exp_x2 = max([tl_x2, tr_x2, br_x2, bl_x2])
                exp_y1 = min([tl_y1, tr_y1, br_y1, bl_y1])
                exp_y2 = max([tl_y2, tr_y2, br_y2, bl_y2])
                assert np.isclose(bb_aug.x1, exp_x1, atol=7.5)
                assert np.isclose(bb_aug.y1, exp_y1, atol=7.5)
                assert np.isclose(bb_aug.x2, exp_x2, atol=7.5)
                assert np.isclose(bb_aug.y2, exp_y2, atol=7.5)

    def test_empty_bounding_boxes(self):
        # test empty bounding boxes
        bbsoi = ia.BoundingBoxesOnImage([], shape=(20, 10, 3))
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)

        observed = aug.augment_bounding_boxes(bbsoi)

        assert_cbaois_equal(observed, bbsoi)

    # ------------
    # mode
    # ------------
    def test_draw_samples_with_mode_being_int(self):
        aug = iaa.PerspectiveTransform(scale=0.001, mode=cv2.BORDER_REPLICATE)

        samples = aug._draw_samples([(10, 10, 3)], iarandom.RNG(0))

        assert samples.modes.shape == (1,)
        assert samples.modes[0] == cv2.BORDER_REPLICATE

    def test_draw_samples_with_mode_being_string(self):
        aug = iaa.PerspectiveTransform(scale=0.001, mode="replicate")

        samples = aug._draw_samples([(10, 10, 3)], iarandom.RNG(0))

        assert samples.modes.shape == (1,)
        assert samples.modes[0] == cv2.BORDER_REPLICATE

    def test_mode_replicate_copies_values(self):
        aug = iaa.PerspectiveTransform(
            scale=0.001, mode="replicate", cval=0, random_state=31)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255

        img_aug = aug.augment_image(img)

        assert (img_aug == 255).all()

    def test_mode_constant_uses_cval(self):
        aug255 = iaa.PerspectiveTransform(
            scale=0.001, mode="constant", cval=255, random_state=31)
        aug0 = iaa.PerspectiveTransform(
            scale=0.001, mode="constant", cval=0, random_state=31)
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255

        img_aug255 = aug255.augment_image(img)
        img_aug0 = aug0.augment_image(img)

        assert (img_aug255 == 255).all()
        assert not (img_aug0 == 255).all()

    # ---------
    # fit_output
    # ---------
    def test_fit_output_with_fixed_jitter(self):
        aug = iaa.PerspectiveTransform(scale=0.2, fit_output=True,
                                       keep_size=False)
        aug.jitter = iap.Deterministic(0.2)

        image = np.zeros((40, 40, 3), dtype=np.uint8)
        image[0:3, 0:3, 0] = 255
        image[0:3, 40-3:, 1] = 255
        image[40-3:, 40-3:, 2] = 255

        image_aug = aug(image=image)

        h, w = image_aug.shape[0:2]
        y0 = np.argmax(image_aug[:, 0, 0])
        x0 = np.argmax(image_aug[0, :, 0])
        y1 = np.argmax(image_aug[:, w-1, 1])
        x1 = np.argmax(image_aug[0, :, 1])
        y2 = np.argmax(image_aug[:, w-1, 2])
        x2 = np.argmax(image_aug[h-1, :, 2])

        # different shape
        assert image_aug.shape != image.shape

        # corners roughly still at top-left, top-right, bottom-right
        assert 0 <= y0 <= 3
        assert 0 <= x0 <= 3
        assert 0 <= y1 <= 3
        assert image_aug.shape[1]-3 <= x1 <= image_aug.shape[1]
        assert image_aug.shape[1]-3 <= y2 <= image_aug.shape[1]
        assert image_aug.shape[1]-3 <= x2 <= image_aug.shape[1]

        # no corner pixels now in the center
        assert np.max(image_aug[8:h-8, 8:w-8, :]) == 0

    def test_fit_output_with_random_jitter(self):
        aug = iaa.PerspectiveTransform(scale=0.1, fit_output=True,
                                       keep_size=False)

        image = np.zeros((50, 50, 4), dtype=np.uint8)
        image[0:5, 0:5, 0] = 255
        image[0:5, 50-5:, 1] = 255
        image[50-5:, 50-5:, 2] = 255
        image[50-5:, 0:5, 3] = 255

        for _ in sm.xrange(10):
            image_aug = aug(image=image)

            h, w = image_aug.shape[0:2]
            arr_nochan = np.max(image_aug, axis=2)
            y_idx = np.where(np.max(arr_nochan, axis=1))[0]
            x_idx = np.where(np.max(arr_nochan, axis=0))[0]
            y_min = np.min(y_idx)
            y_max = np.max(y_idx)
            x_min = np.min(x_idx)
            x_max = np.max(x_idx)

            tol = 0
            assert 0 <= y_min <= 5+tol
            assert 0 <= x_min <= 5+tol
            assert h-5-tol <= y_max <= h-1
            assert w-5-tol <= x_max <= w-1

    def test_fit_output_with_random_jitter__segmentation_maps(self):
        aug = iaa.PerspectiveTransform(scale=0.1, fit_output=True,
                                       keep_size=False)

        arr = np.zeros((50, 50, 4), dtype=np.uint8)
        arr[0:5, 0:5, 0] = 1
        arr[0:5, 50-5:, 1] = 1
        arr[50-5:, 50-5:, 2] = 1
        arr[50-5:, 0:5, 3] = 1
        segmap = ia.SegmentationMapsOnImage(arr, shape=(50, 50, 3))

        image = np.zeros((49, 49, 3), dtype=np.uint8)
        image = iaa.pad(image, top=1, right=1, bottom=1, left=1, cval=128)

        for _ in sm.xrange(10):
            image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)

            h, w = segmap_aug.arr.shape[0:2]
            arr_nochan = np.max(segmap_aug.arr, axis=2)
            y_idx = np.where(np.max(arr_nochan, axis=1))[0]
            x_idx = np.where(np.max(arr_nochan, axis=0))[0]
            y_min = np.min(y_idx)
            y_max = np.max(y_idx)
            x_min = np.min(x_idx)
            x_max = np.max(x_idx)

            tol = 0
            assert 0 <= y_min <= 5+tol
            assert 0 <= x_min <= 5+tol
            assert h-5-tol <= y_max <= h-1
            assert w-5-tol <= x_max <= w-1

    def test_fit_output_with_fixed_jitter__keypoints(self):
        aug = iaa.PerspectiveTransform(scale=0.1, fit_output=True,
                                       keep_size=False)

        kpsoi = ia.KeypointsOnImage.from_xy_array([
            (0, 0),
            (50, 0),
            (50, 50),
            (0, 50)
        ], shape=(50, 50, 3))

        for _ in sm.xrange(10):
            kpsoi_aug = aug(keypoints=kpsoi)

            h, w = kpsoi_aug.shape[0:2]
            y0, x0 = kpsoi_aug.keypoints[0].y, kpsoi_aug.keypoints[0].x
            y1, x1 = kpsoi_aug.keypoints[1].y, kpsoi_aug.keypoints[1].x
            y2, x2 = kpsoi_aug.keypoints[2].y, kpsoi_aug.keypoints[2].x
            y3, x3 = kpsoi_aug.keypoints[3].y, kpsoi_aug.keypoints[3].x

            y_min = min([y0, y1, y2, y3])
            y_max = max([y0, y1, y2, y3])
            x_min = min([x0, x1, x2, x3])
            x_max = max([x0, x1, x2, x3])
            tol = 0.5
            assert 0-tol <= y_min <= tol
            assert 0-tol <= x_min <= tol
            assert h-tol <= y_max <= h+tol
            assert w-tol <= x_max <= w+tol

    # ---------
    # unusual channel numbers
    # ---------
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
                aug = iaa.PerspectiveTransform(scale=0.01)

                image_aug = aug(image=image)

                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    # ---------
    # zero-sized axes
    # ---------
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
            for keep_size in [False, True]:
                with self.subTest(shape=shape, keep_size=keep_size):
                    for _ in sm.xrange(3):
                        image = np.zeros(shape, dtype=np.uint8)
                        aug = iaa.PerspectiveTransform(scale=0.01)

                        image_aug = aug(image=image)

                        assert image_aug.dtype.name == "uint8"
                        assert image_aug.shape == shape

    # --------
    # get_parameters
    # --------
    def test_get_parameters(self):
        aug = iaa.PerspectiveTransform(scale=0.1, keep_size=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Normal)
        assert isinstance(params[0].scale, iap.Deterministic)
        assert 0.1 - 1e-8 < params[0].scale.value < 0.1 + 1e-8
        assert params[1] is False
        assert params[2].value == 0
        assert params[3].value == "constant"
        assert params[4] is False

    # --------
    # other dtypes
    # --------
    def test_other_dtypes_bool(self):
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)

        y1 = int(30 * 0.2)
        y2 = int(30 * 0.8)
        x1 = int(30 * 0.2)
        x2 = int(30 * 0.8)

        image = np.zeros((30, 30), dtype=bool)
        image[12:18, :] = True
        image[:, 12:18] = True
        expected = image[y1:y2, x1:x2]
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.name == image.dtype.name
        assert image_aug.shape == expected.shape
        assert (np.sum(image_aug == expected) / expected.size) > 0.9

    def test_other_dtypes_uint_int(self):
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)

        y1 = int(30 * 0.2)
        y2 = int(30 * 0.8)
        x1 = int(30 * 0.2)
        x2 = int(30 * 0.8)

        dtypes = ["uint8", "uint16", "int8", "int16"]
        for dtype in dtypes:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            if np.dtype(dtype).kind == "i":
                values = [0, 1, 5, 10, 100, int(0.1 * max_value),
                          int(0.2 * max_value), int(0.5 * max_value),
                          max_value-100, max_value]
                values = values + [(-1)*value for value in values]
            else:
                values = [0, 1, 5, 10, 100, int(center_value),
                          int(0.1 * max_value), int(0.2 * max_value),
                          int(0.5 * max_value), max_value-100, max_value]

            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((30, 30), dtype=dtype)
                    image[12:18, :] = value
                    image[:, 12:18] = value
                    expected = image[y1:y2, x1:x2]

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert image_aug.shape == expected.shape
                    # rather high tolerance of 0.7 here because of
                    # interpolation
                    assert (
                        np.sum(image_aug == expected) / expected.size
                    ) > 0.7

    def test_other_dtypes_float(self):
        aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
        aug.jitter = iap.Deterministic(0.2)

        y1 = int(30 * 0.2)
        y2 = int(30 * 0.8)
        x1 = int(30 * 0.2)
        x2 = int(30 * 0.8)

        dtypes = ["float16", "float32", "float64"]
        for dtype in dtypes:
            def _isclose(a, b):
                atol = 1e-4 if dtype == "float16" else 1e-8
                return np.isclose(a, b, atol=atol, rtol=0)

            isize = np.dtype(dtype).itemsize
            values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1),
                      1000 ** (isize - 1)]
            values = values + [(-1) * value for value in values]
            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((30, 30), dtype=dtype)
                    image[12:18, :] = value
                    image[:, 12:18] = value
                    expected = image[y1:y2, x1:x2]

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert image_aug.shape == expected.shape
                    # rather high tolerance of 0.7 here because of
                    # interpolation
                    assert (
                        np.sum(_isclose(image_aug, expected)) / expected.size
                    ) > 0.7

    def test_pickleable(self):
        aug = iaa.PerspectiveTransform(0.2, random_state=1)
        runtest_pickleable_uint8_img(aug, iterations=4, shape=(25, 25, 1))


class _elastic_trans_temp_thresholds(object):
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma
        self.old_alpha = None
        self.old_sigma = None

    def __enter__(self):
        self.old_alpha = iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH
        self.old_sigma = iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH
        iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = self.alpha
        iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = self.sigma

    def __exit__(self, exc_type, exc_val, exc_tb):
        iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = self.old_alpha
        iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = self.old_sigma


# TODO add tests for order
# TODO improve tests for cval
# TODO add tests for mode
class TestElasticTransformation(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        img = np.zeros((50, 50), dtype=np.uint8) + 255
        img = np.pad(img, ((100, 100), (100, 100)), mode="constant",
                     constant_values=0)
        return img

    @property
    def mask(self):
        img = self.image
        mask = img > 0
        return mask

    @property
    def heatmaps(self):
        img = self.image
        return HeatmapsOnImage(img.astype(np.float32) / 255.0,
                               shape=img.shape)

    @property
    def segmaps(self):
        img = self.image
        return SegmentationMapsOnImage((img > 0).astype(np.int32),
                                       shape=img.shape)

    # -----------
    # __init__
    # -----------
    def test___init___bad_datatype_for_alpha_leads_to_failure(self):
        # test alpha having bad datatype
        got_exception = False
        try:
            _ = iaa.ElasticTransformation(alpha=False, sigma=0.25)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test___init___alpha_is_tuple(self):
        # test alpha being tuple
        aug = iaa.ElasticTransformation(alpha=(1.0, 2.0), sigma=0.25)
        assert isinstance(aug.alpha, iap.Uniform)
        assert isinstance(aug.alpha.a, iap.Deterministic)
        assert isinstance(aug.alpha.b, iap.Deterministic)
        assert 1.0 - 1e-8 < aug.alpha.a.value < 1.0 + 1e-8
        assert 2.0 - 1e-8 < aug.alpha.b.value < 2.0 + 1e-8

    def test___init___sigma_is_tuple(self):
        # test sigma being tuple
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=(1.0, 2.0))
        assert isinstance(aug.sigma, iap.Uniform)
        assert isinstance(aug.sigma.a, iap.Deterministic)
        assert isinstance(aug.sigma.b, iap.Deterministic)
        assert 1.0 - 1e-8 < aug.sigma.a.value < 1.0 + 1e-8
        assert 2.0 - 1e-8 < aug.sigma.b.value < 2.0 + 1e-8

    def test___init___bad_datatype_for_sigma_leads_to_failure(self):
        # test sigma having bad datatype
        got_exception = False
        try:
            _ = iaa.ElasticTransformation(alpha=0.25, sigma=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test___init___order_is_all(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=ia.ALL)
        assert isinstance(aug.order, iap.Choice)
        assert all([order in aug.order.a for order in [0, 1, 2, 3, 4, 5]])

    def test___init___order_is_int(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=1)
        assert isinstance(aug.order, iap.Deterministic)
        assert aug.order.value == 1

    def test___init___order_is_list(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=[0, 1, 2])
        assert isinstance(aug.order, iap.Choice)
        assert all([order in aug.order.a for order in [0, 1, 2]])

    def test___init___order_is_stochastic_parameter(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0,
                                        order=iap.Choice([0, 1, 2, 3]))
        assert isinstance(aug.order, iap.Choice)
        assert all([order in aug.order.a for order in [0, 1, 2, 3]])

    def test___init___bad_datatype_for_order_leads_to_failure(self):
        got_exception = False
        try:
            _ = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test___init___cval_is_all(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=ia.ALL)
        assert isinstance(aug.cval, iap.Uniform)
        assert isinstance(aug.cval.a, iap.Deterministic)
        assert isinstance(aug.cval.b, iap.Deterministic)
        assert aug.cval.a.value == 0
        assert aug.cval.b.value == 255

    def test___init___cval_is_int(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=128)
        assert isinstance(aug.cval, iap.Deterministic)
        assert aug.cval.value == 128

    def test___init___cval_is_list(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0,
                                        cval=[16, 32, 64])
        assert isinstance(aug.cval, iap.Choice)
        assert all([cval in aug.cval.a for cval in [16, 32, 64]])

    def test___init___cval_is_stochastic_parameter(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0,
                                        cval=iap.Choice([16, 32, 64]))
        assert isinstance(aug.cval, iap.Choice)
        assert all([cval in aug.cval.a for cval in [16, 32, 64]])

    def test___init___cval_is_tuple(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=(128, 255))
        assert isinstance(aug.cval, iap.Uniform)
        assert isinstance(aug.cval.a, iap.Deterministic)
        assert isinstance(aug.cval.b, iap.Deterministic)
        assert aug.cval.a.value == 128
        assert aug.cval.b.value == 255

    def test___init___bad_datatype_for_cval_leads_to_failure(self):
        got_exception = False
        try:
            _ = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test___init___mode_is_all(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode=ia.ALL)
        assert isinstance(aug.mode, iap.Choice)
        assert all([
            mode in aug.mode.a
            for mode
            in ["constant", "nearest", "reflect", "wrap"]])

    def test___init___mode_is_string(self):
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode="nearest")
        assert isinstance(aug.mode, iap.Deterministic)
        assert aug.mode.value == "nearest"

    def test___init___mode_is_list(self):
        aug = iaa.ElasticTransformation(
            alpha=0.25, sigma=1.0, mode=["constant", "nearest"])
        assert isinstance(aug.mode, iap.Choice)
        assert all([mode in aug.mode.a for mode in ["constant", "nearest"]])

    def test___init___mode_is_stochastic_parameter(self):
        aug = iaa.ElasticTransformation(
            alpha=0.25, sigma=1.0, mode=iap.Choice(["constant", "nearest"]))
        assert isinstance(aug.mode, iap.Choice)
        assert all([mode in aug.mode.a for mode in ["constant", "nearest"]])

    def test___init___bad_datatype_for_mode_leads_to_failure(self):
        got_exception = False
        try:
            _ = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    # -----------
    # alpha, sigma
    # -----------
    def test_images(self):
        # test basic funtionality
        aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)

        observed = aug.augment_image(self.image)

        mask = self.mask
        # assume that some white/255 pixels have been moved away from the
        # center and replaced by black/0 pixels
        assert np.sum(observed[mask]) < np.sum(self.image[mask])
        # assume that some black/0 pixels have been moved away from the outer
        # area and replaced by white/255 pixels
        assert np.sum(observed[~mask]) > np.sum(self.image[~mask])

    def test_images_nonsquare(self):
        # test basic funtionality with non-square images
        aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)
        img_nonsquare = np.zeros((50, 100), dtype=np.uint8) + 255
        img_nonsquare = np.pad(img_nonsquare, ((100, 100), (100, 100)),
                               mode="constant", constant_values=0)
        mask_nonsquare = (img_nonsquare > 0)

        observed = aug.augment_image(img_nonsquare)

        assert (
            np.sum(observed[mask_nonsquare])
            < np.sum(img_nonsquare[mask_nonsquare]))
        assert (
            np.sum(observed[~mask_nonsquare])
            > np.sum(img_nonsquare[~mask_nonsquare]))

    def test_images_unusual_channel_numbers(self):
        # test unusual channels numbers
        aug = iaa.ElasticTransformation(alpha=5, sigma=0.5)
        for nb_channels in [1, 2, 4, 5, 7, 10, 11]:
            img_c = np.tile(self.image[..., np.newaxis], (1, 1, nb_channels))
            assert img_c.shape == (250, 250, nb_channels)

            observed = aug.augment_image(img_c)

            assert observed.shape == (250, 250, nb_channels)
            for c in sm.xrange(1, nb_channels):
                assert np.array_equal(observed[..., c], observed[..., 0])

    def test_heatmaps(self):
        # test basic funtionality, heatmaps
        aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)
        observed = aug.augment_heatmaps([self.heatmaps])[0]

        mask = self.mask
        assert observed.shape == self.heatmaps.shape
        _assert_same_min_max(observed, self.heatmaps)
        assert (
            np.sum(observed.get_arr()[mask])
            < np.sum(self.heatmaps.get_arr()[mask]))
        assert (
            np.sum(observed.get_arr()[~mask])
            > np.sum(self.heatmaps.get_arr()[~mask]))

    def test_segmaps(self):
        # test basic funtionality, segmaps
        # alpha=1.5 instead of 0.5 as above here, because otherwise nothing
        # is moved
        aug = iaa.ElasticTransformation(alpha=1.5, sigma=0.25)

        observed = aug.augment_segmentation_maps([self.segmaps])[0]

        mask = self.mask
        assert observed.shape == self.segmaps.shape
        assert (
            np.sum(observed.get_arr()[mask])
            < np.sum(self.segmaps.get_arr()[mask]))
        assert (
            np.sum(observed.get_arr()[~mask])
            > np.sum(self.segmaps.get_arr()[~mask]))

    def test_images_weak_vs_strong_alpha(self):
        # test effects of increased alpha strength
        aug1 = iaa.ElasticTransformation(alpha=0.1, sigma=0.25)
        aug2 = iaa.ElasticTransformation(alpha=5.0, sigma=0.25)

        observed1 = aug1.augment_image(self.image)
        observed2 = aug2.augment_image(self.image)

        mask = self.mask
        # assume that the inner area has become more black-ish when using high
        # alphas (more white pixels were moved out of the inner area)
        assert np.sum(observed1[mask]) > np.sum(observed2[mask])
        # assume that the outer area has become more white-ish when using high
        # alphas (more black pixels were moved into the inner area)
        assert np.sum(observed1[~mask]) < np.sum(observed2[~mask])

    def test_heatmaps_weak_vs_strong_alpha(self):
        # test effects of increased alpha strength, heatmaps
        aug1 = iaa.ElasticTransformation(alpha=0.1, sigma=0.25)
        aug2 = iaa.ElasticTransformation(alpha=5.0, sigma=0.25)

        observed1 = aug1.augment_heatmaps([self.heatmaps])[0]
        observed2 = aug2.augment_heatmaps([self.heatmaps])[0]

        mask = self.mask
        assert observed1.shape == self.heatmaps.shape
        assert observed2.shape == self.heatmaps.shape
        _assert_same_min_max(observed1, self.heatmaps)
        _assert_same_min_max(observed2, self.heatmaps)
        assert (
            np.sum(observed1.get_arr()[mask])
            > np.sum(observed2.get_arr()[mask]))
        assert (
            np.sum(observed1.get_arr()[~mask])
            < np.sum(observed2.get_arr()[~mask]))

    def test_segmaps_weak_vs_strong_alpha(self):
        # test effects of increased alpha strength, segmaps
        aug1 = iaa.ElasticTransformation(alpha=0.1, sigma=0.25)
        aug2 = iaa.ElasticTransformation(alpha=5.0, sigma=0.25)

        observed1 = aug1.augment_segmentation_maps([self.segmaps])[0]
        observed2 = aug2.augment_segmentation_maps([self.segmaps])[0]

        mask = self.mask
        assert observed1.shape == self.segmaps.shape
        assert observed2.shape == self.segmaps.shape
        assert (
            np.sum(observed1.get_arr()[mask])
            > np.sum(observed2.get_arr()[mask]))
        assert (
            np.sum(observed1.get_arr()[~mask])
            < np.sum(observed2.get_arr()[~mask]))

    def test_images_low_vs_high_sigma(self):
        # test effects of increased sigmas
        aug1 = iaa.ElasticTransformation(alpha=3.0, sigma=0.1)
        aug2 = iaa.ElasticTransformation(alpha=3.0, sigma=3.0)

        observed1 = aug1.augment_image(self.image)
        observed2 = aug2.augment_image(self.image)

        observed1_std_hori = np.std(
            observed1.astype(np.float32)[:, 1:]
            - observed1.astype(np.float32)[:, :-1])
        observed2_std_hori = np.std(
            observed2.astype(np.float32)[:, 1:]
            - observed2.astype(np.float32)[:, :-1])
        observed1_std_vert = np.std(
            observed1.astype(np.float32)[1:, :]
            - observed1.astype(np.float32)[:-1, :])
        observed2_std_vert = np.std(
            observed2.astype(np.float32)[1:, :]
            - observed2.astype(np.float32)[:-1, :])
        observed1_std = (observed1_std_hori + observed1_std_vert) / 2
        observed2_std = (observed2_std_hori + observed2_std_vert) / 2
        assert observed1_std > observed2_std

    def test_images_alpha_is_stochastic_parameter(self):
        # test alpha being iap.Choice
        aug = iaa.ElasticTransformation(alpha=iap.Choice([0.001, 5.0]),
                                        sigma=0.25)
        seen = [0, 0]
        for _ in sm.xrange(100):
            observed = aug.augment_image(self.image)
            diff = np.average(
                np.abs(
                    self.image.astype(np.float32)
                    - observed.astype(np.float32)
                )
            )
            if diff < 1.0:
                seen[0] += 1
            else:
                seen[1] += 1
        assert seen[0] > 10
        assert seen[1] > 10

    def test_sigma_is_stochastic_parameter(self):
        # test sigma being iap.Choice
        aug = iaa.ElasticTransformation(alpha=3.0,
                                        sigma=iap.Choice([0.01, 5.0]))
        seen = [0, 0]
        for _ in sm.xrange(100):
            observed = aug.augment_image(self.image)

            observed_std_hori = np.std(
                observed.astype(np.float32)[:, 1:]
                - observed.astype(np.float32)[:, :-1])
            observed_std_vert = np.std(
                observed.astype(np.float32)[1:, :]
                - observed.astype(np.float32)[:-1, :])
            observed_std = (observed_std_hori + observed_std_vert) / 2

            if observed_std > 10.0:
                seen[0] += 1
            else:
                seen[1] += 1
        assert seen[0] > 10
        assert seen[1] > 10

    # -----------
    # cval
    # -----------
    def test_images_cval_is_int_and_order_is_0(self):
        aug = iaa.ElasticTransformation(alpha=30.0, sigma=3.0, mode="constant",
                                        cval=255, order=0)
        img = np.zeros((100, 100), dtype=np.uint8)

        observed = aug.augment_image(img)

        assert np.sum(observed == 255) > 0
        assert np.sum(np.logical_and(0 < observed, observed < 255)) == 0

    def test_images_cval_is_int_and_order_is_0_weak_alpha(self):
        aug = iaa.ElasticTransformation(alpha=3.0, sigma=3.0, mode="constant",
                                        cval=0, order=0)
        img = np.zeros((100, 100), dtype=np.uint8)

        observed = aug.augment_image(img)

        assert np.sum(observed == 255) == 0

    def test_images_cval_is_int_and_order_is_2(self):
        aug = iaa.ElasticTransformation(alpha=3.0, sigma=3.0, mode="constant",
                                        cval=255, order=2)
        img = np.zeros((100, 100), dtype=np.uint8)

        observed = aug.augment_image(img)

        assert np.sum(np.logical_and(0 < observed, observed < 255)) > 0

    def test_heatmaps_ignore_cval(self):
        # cval with heatmaps
        heatmaps = HeatmapsOnImage(
            np.zeros((32, 32, 1), dtype=np.float32), shape=(32, 32, 3))
        aug = iaa.ElasticTransformation(alpha=3.0, sigma=3.0,
                                        mode="constant", cval=255)

        observed = aug.augment_heatmaps([heatmaps])[0]

        assert observed.shape == heatmaps.shape
        _assert_same_min_max(observed, heatmaps)
        assert np.sum(observed.get_arr() > 0.01) == 0

    def test_segmaps_ignore_cval(self):
        # cval with segmaps
        segmaps = SegmentationMapsOnImage(
            np.zeros((32, 32, 1), dtype=np.int32), shape=(32, 32, 3))
        aug = iaa.ElasticTransformation(alpha=3.0, sigma=3.0, mode="constant",
                                        cval=255)

        observed = aug.augment_segmentation_maps([segmaps])[0]

        assert observed.shape == segmaps.shape
        assert np.sum(observed.get_arr() > 0) == 0

    # -----------
    # keypoints
    # -----------
    def test_keypoints_no_movement_if_alpha_below_threshold(self):
        # for small alpha, should not move if below threshold
        with _elastic_trans_temp_thresholds(alpha=1.0, sigma=0.0):
            kps = [
                ia.Keypoint(x=1, y=1), ia.Keypoint(x=15, y=25),
                ia.Keypoint(x=5, y=5), ia.Keypoint(x=7, y=4),
                ia.Keypoint(x=48, y=5), ia.Keypoint(x=21, y=37),
                ia.Keypoint(x=32, y=39), ia.Keypoint(x=6, y=8),
                ia.Keypoint(x=12, y=21), ia.Keypoint(x=3, y=45),
                ia.Keypoint(x=45, y=3), ia.Keypoint(x=7, y=48)]
            kpsoi = ia.KeypointsOnImage(kps, shape=(50, 50))
            aug = iaa.ElasticTransformation(alpha=0.001, sigma=1.0)
    
            observed = aug.augment_keypoints([kpsoi])[0]
    
            d = kpsoi.to_xy_array() - observed.to_xy_array()
            d[:, 0] = d[:, 0] ** 2
            d[:, 1] = d[:, 1] ** 2
            d = np.sum(d, axis=1)
            d = np.average(d, axis=0)
            assert d < 1e-8

    def test_keypoints_no_movement_if_sigma_below_threshold(self):
        # for small sigma, should not move if below threshold
        with _elastic_trans_temp_thresholds(alpha=0.0, sigma=1.0):
            kps = [
                ia.Keypoint(x=1, y=1), ia.Keypoint(x=15, y=25),
                ia.Keypoint(x=5, y=5), ia.Keypoint(x=7, y=4),
                ia.Keypoint(x=48, y=5), ia.Keypoint(x=21, y=37),
                ia.Keypoint(x=32, y=39), ia.Keypoint(x=6, y=8),
                ia.Keypoint(x=12, y=21), ia.Keypoint(x=3, y=45),
                ia.Keypoint(x=45, y=3), ia.Keypoint(x=7, y=48)]
            kpsoi = ia.KeypointsOnImage(kps, shape=(50, 50))
            aug = iaa.ElasticTransformation(alpha=1.0, sigma=0.001)

            observed = aug.augment_keypoints([kpsoi])[0]

            d = kpsoi.to_xy_array() - observed.to_xy_array()
            d[:, 0] = d[:, 0] ** 2
            d[:, 1] = d[:, 1] ** 2
            d = np.sum(d, axis=1)
            d = np.average(d, axis=0)
            assert d < 1e-8

    def test_keypoints_small_movement_for_weak_alpha_if_threshold_zero(self):
        # for small alpha (at sigma 1.0), should barely move
        # if thresholds set to zero
        with _elastic_trans_temp_thresholds(alpha=0.0, sigma=0.0):
            kps = [
                ia.Keypoint(x=1, y=1), ia.Keypoint(x=15, y=25),
                ia.Keypoint(x=5, y=5), ia.Keypoint(x=7, y=4),
                ia.Keypoint(x=48, y=5), ia.Keypoint(x=21, y=37),
                ia.Keypoint(x=32, y=39), ia.Keypoint(x=6, y=8),
                ia.Keypoint(x=12, y=21), ia.Keypoint(x=3, y=45),
                ia.Keypoint(x=45, y=3), ia.Keypoint(x=7, y=48)]
            kpsoi = ia.KeypointsOnImage(kps, shape=(50, 50))
            aug = iaa.ElasticTransformation(alpha=0.001, sigma=1.0)

            observed = aug.augment_keypoints([kpsoi])[0]

            d = kpsoi.to_xy_array() - observed.to_xy_array()
            d[:, 0] = d[:, 0] ** 2
            d[:, 1] = d[:, 1] ** 2
            d = np.sum(d, axis=1)
            d = np.average(d, axis=0)
            assert d < 0.5

    def test_image_keypoint_alignment(self):
        # test alignment between between images and keypoints
        image = np.zeros((120, 70), dtype=np.uint8)
        s = 3
        image[:, 35-s:35+s+1] = 255
        kps = [ia.Keypoint(x=35, y=20),
               ia.Keypoint(x=35, y=40),
               ia.Keypoint(x=35, y=60),
               ia.Keypoint(x=35, y=80),
               ia.Keypoint(x=35, y=100)]
        kpsoi = ia.KeypointsOnImage(kps, shape=image.shape)
        aug = iaa.ElasticTransformation(alpha=70, sigma=5)
        aug_det = aug.to_deterministic()

        images_aug = aug_det.augment_images([image, image])
        kpsois_aug = aug_det.augment_keypoints([kpsoi, kpsoi])

        count_bad = 0
        for image_aug, kpsoi_aug in zip(images_aug, kpsois_aug):
            assert kpsoi_aug.shape == (120, 70)
            assert len(kpsoi_aug.keypoints) == 5
            for kp_aug in kpsoi_aug.keypoints:
                x, y = int(np.round(kp_aug.x)), int(np.round(kp_aug.y))
                bb = ia.BoundingBox(x1=x-2, x2=x+2+1, y1=y-2, y2=y+2+1)
                img_ex = bb.extract_from_image(image_aug)
                if np.any(img_ex > 10):
                    pass  # close to expected location
                else:
                    count_bad += 1
        assert count_bad <= 1

    def test_empty_keypoints(self):
        aug = iaa.ElasticTransformation(alpha=10, sigma=10)
        kpsoi = ia.KeypointsOnImage([], shape=(10, 10, 3))

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert len(kpsoi_aug.keypoints) == 0
        assert kpsoi_aug.shape == (10, 10, 3)

    # -----------
    # abstract methods for polygons and line strings
    # -----------
    @classmethod
    def _test_cbaois_no_movement_if_alpha_below_threshold(
            cls, cba_class, cbaoi_class, augf_name):
        # for small alpha, should not move if below threshold
        with _elastic_trans_temp_thresholds(alpha=1.0, sigma=0.0):
            cba = cba_class([(10, 15), (40, 15), (40, 35), (10, 35)])
            cbaoi = cbaoi_class([cba], shape=(50, 50))
            aug = iaa.ElasticTransformation(alpha=0.001, sigma=1.0)

            observed = getattr(aug, augf_name)(cbaoi)

            assert observed.shape == (50, 50)
            assert len(observed.items) == 1
            assert observed.items[0].coords_almost_equals(cba)
            if hasattr(observed.items[0], "is_valid"):
                assert observed.items[0].is_valid

    @classmethod
    def _test_cbaois_no_movement_if_sigma_below_threshold(
            cls, cba_class, cbaoi_class, augf_name):
        # for small sigma, should not move if below threshold
        with _elastic_trans_temp_thresholds(alpha=0.0, sigma=1.0):
            cba = cba_class([(10, 15), (40, 15), (40, 35), (10, 35)])
            cbaoi = cbaoi_class([cba], shape=(50, 50))
            aug = iaa.ElasticTransformation(alpha=1.0, sigma=0.001)

            observed = getattr(aug, augf_name)(cbaoi)

            assert observed.shape == (50, 50)
            assert len(observed.items) == 1
            assert observed.items[0].coords_almost_equals(cba)
            if hasattr(observed.items[0], "is_valid"):
                assert observed.items[0].is_valid

    @classmethod
    def _test_cbaois_small_movement_for_weak_alpha_if_threshold_zero(
            cls, cba_class, cbaoi_class, augf_name):
        # for small alpha (at sigma 1.0), should barely move
        # if thresholds set to zero
        with _elastic_trans_temp_thresholds(alpha=0.0, sigma=0.0):
            cba = cba_class([(10, 15), (40, 15), (40, 35), (10, 35)])
            cbaoi = cbaoi_class([cba], shape=(50, 50))
            aug = iaa.ElasticTransformation(alpha=0.001, sigma=1.0)

            observed = getattr(aug, augf_name)(cbaoi)

            assert observed.shape == (50, 50)
            assert len(observed.items) == 1
            assert observed.items[0].coords_almost_equals(
                cba, max_distance=0.5)
            if hasattr(observed.items[0], "is_valid"):
                assert observed.items[0].is_valid

    @classmethod
    def _test_image_cbaoi_alignment(cls, cba_class, cbaoi_class, augf_name):
        # test alignment between between images and polygons
        height_step_size = 50
        width_step_size = 30
        height_steps = 2  # don't set >2, otherwise polygon will be broken
        width_steps = 10
        height = (2+height_steps) * height_step_size
        width = (2+width_steps) * width_step_size
        s = 3

        image = np.zeros((height, width), dtype=np.uint8)

        points = []
        for w in sm.xrange(0, 2+width_steps):
            if w not in [0, width_steps+2-1]:
                x = width_step_size * w
                y = height_step_size
                points.append((x, y))
                image[y-s:y+s+1, x-s:x+s+1] = 255
        for w in sm.xrange(2+width_steps-1, 0, -1):
            if w not in [0, width_steps+2-1]:
                x = width_step_size * w
                y = height_step_size*2
                points.append((x, y))
                image[y-s:y+s+1, x-s:x+s+1] = 255

        cba = cba_class(points)
        cbaoi = cbaoi_class([cba], shape=image.shape)
        aug = iaa.ElasticTransformation(alpha=100, sigma=7)
        aug_det = aug.to_deterministic()

        images_aug = aug_det.augment_images([image, image])
        cbaois_aug = getattr(aug_det, augf_name)([cbaoi, cbaoi])

        count_bad = 0
        for image_aug, cbaoi_aug in zip(images_aug, cbaois_aug):
            assert cbaoi_aug.shape == image.shape
            assert len(cbaoi_aug.items) == 1
            for cba_aug in cbaoi_aug.items:
                if hasattr(cba_aug, "is_valid"):
                    assert cba_aug.is_valid
                for point_aug in cba_aug.coords:
                    x, y = point_aug[0], point_aug[1]
                    bb = ia.BoundingBox(x1=x-2, x2=x+2, y1=y-2, y2=y+2)
                    img_ex = bb.extract_from_image(image_aug)
                    if np.any(img_ex > 10):
                        pass  # close to expected location
                    else:
                        count_bad += 1
        assert count_bad <= 3

    @classmethod
    def _test_empty_cbaois(cls, cbaoi, augf_name):
        aug = iaa.ElasticTransformation(alpha=10, sigma=10)

        cbaoi_aug = getattr(aug, augf_name)(cbaoi)

        assert_cbaois_equal(cbaoi_aug, cbaoi)

    # -----------
    # polygons
    # -----------
    def test_polygons_no_movement_if_alpha_below_threshold(self):
        self._test_cbaois_no_movement_if_alpha_below_threshold(
            ia.Polygon, ia.PolygonsOnImage, "augment_polygons")

    def test_polygons_no_movement_if_sigma_below_threshold(self):
        self._test_cbaois_no_movement_if_sigma_below_threshold(
            ia.Polygon, ia.PolygonsOnImage, "augment_polygons")

    def test_polygons_small_movement_for_weak_alpha_if_threshold_zero(self):
        self._test_cbaois_small_movement_for_weak_alpha_if_threshold_zero(
            ia.Polygon, ia.PolygonsOnImage, "augment_polygons")

    def test_image_polygon_alignment(self):
        self._test_image_cbaoi_alignment(
            ia.Polygon, ia.PolygonsOnImage, "augment_polygons")

    def test_empty_polygons(self):
        cbaoi = ia.PolygonsOnImage([], shape=(10, 10, 3))
        self._test_empty_cbaois(cbaoi, "augment_polygons")

    # -----------
    # line strings
    # -----------
    def test_line_strings_no_movement_if_alpha_below_threshold(self):
        self._test_cbaois_no_movement_if_alpha_below_threshold(
            ia.LineString, ia.LineStringsOnImage, "augment_line_strings")

    def test_line_strings_no_movement_if_sigma_below_threshold(self):
        self._test_cbaois_no_movement_if_sigma_below_threshold(
            ia.LineString, ia.LineStringsOnImage, "augment_line_strings")

    def test_line_strings_small_movement_for_weak_alpha_if_threshold_zero(self):
        self._test_cbaois_small_movement_for_weak_alpha_if_threshold_zero(
            ia.LineString, ia.LineStringsOnImage, "augment_line_strings")

    def test_image_line_string_alignment(self):
        self._test_image_cbaoi_alignment(
            ia.LineString, ia.LineStringsOnImage, "augment_line_strings")

    def test_empty_line_strings(self):
        cbaoi = ia.LineStringsOnImage([], shape=(10, 10, 3))
        self._test_empty_cbaois(cbaoi, "augment_line_strings")

    # -----------
    # bounding boxes
    # -----------
    def test_bounding_boxes_no_movement_if_alpha_below_threshold(self):
        # for small alpha, should not move if below threshold
        with _elastic_trans_temp_thresholds(alpha=1.0, sigma=0.0):
            bbs = [
                ia.BoundingBox(x1=10, y1=12, x2=20, y2=22),
                ia.BoundingBox(x1=20, y1=32, x2=40, y2=42)
            ]
            bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(50, 50))
            aug = iaa.ElasticTransformation(alpha=0.001, sigma=1.0)

            observed = aug.augment_bounding_boxes([bbsoi])[0]

            d = bbsoi.to_xyxy_array() - observed.to_xyxy_array()
            d = d.reshape((2*2, 2))
            d[:, 0] = d[:, 0] ** 2
            d[:, 1] = d[:, 1] ** 2
            d = np.sum(d, axis=1)
            d = np.average(d, axis=0)
            assert d < 1e-8

    def test_bounding_boxes_no_movement_if_sigma_below_threshold(self):
        # for small sigma, should not move if below threshold
        with _elastic_trans_temp_thresholds(alpha=0.0, sigma=1.0):
            bbs = [
                ia.BoundingBox(x1=10, y1=12, x2=20, y2=22),
                ia.BoundingBox(x1=20, y1=32, x2=40, y2=42)
            ]
            bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(50, 50))
            aug = iaa.ElasticTransformation(alpha=1.0, sigma=0.001)

            observed = aug.augment_bounding_boxes([bbsoi])[0]

            d = bbsoi.to_xyxy_array() - observed.to_xyxy_array()
            d = d.reshape((2*2, 2))
            d[:, 0] = d[:, 0] ** 2
            d[:, 1] = d[:, 1] ** 2
            d = np.sum(d, axis=1)
            d = np.average(d, axis=0)
            assert d < 1e-8

    def test_bounding_boxes_small_movement_for_weak_alpha_if_threshold_zero(
            self):
        # for small alpha (at sigma 1.0), should barely move
        # if thresholds set to zero
        with _elastic_trans_temp_thresholds(alpha=0.0, sigma=0.0):
            bbs = [
                ia.BoundingBox(x1=10, y1=12, x2=20, y2=22),
                ia.BoundingBox(x1=20, y1=32, x2=40, y2=42)
            ]
            bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(50, 50))
            aug = iaa.ElasticTransformation(alpha=0.001, sigma=1.0)

            observed = aug.augment_bounding_boxes([bbsoi])[0]

            d = bbsoi.to_xyxy_array() - observed.to_xyxy_array()
            d = d.reshape((2*2, 2))
            d[:, 0] = d[:, 0] ** 2
            d[:, 1] = d[:, 1] ** 2
            d = np.sum(d, axis=1)
            d = np.average(d, axis=0)
            assert d < 0.5

    def test_image_bounding_box_alignment(self):
        # test alignment between between images and bounding boxes
        image = np.zeros((100, 100), dtype=np.uint8)
        image[35:35+1, 35:65+1] = 255
        image[65:65+1, 35:65+1] = 255
        image[35:65+1, 35:35+1] = 255
        image[35:65+1, 65:65+1] = 255
        bbs = [
            ia.BoundingBox(x1=35.5, y1=35.5, x2=65.5, y2=65.5)
        ]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=image.shape)
        aug = iaa.ElasticTransformation(alpha=70, sigma=5)

        images_aug, bbsois_aug = aug(images=[image, image],
                                     bounding_boxes=[bbsoi, bbsoi])

        count_bad = 0
        for image_aug, bbsoi_aug in zip(images_aug, bbsois_aug):
            assert bbsoi_aug.shape == (100, 100)
            assert len(bbsoi_aug.bounding_boxes) == 1
            for bb_aug in bbsoi_aug.bounding_boxes:
                if bb_aug.is_fully_within_image(image_aug):
                    # top, bottom, left, right
                    x1 = bb_aug.x1_int
                    x2 = bb_aug.x2_int
                    y1 = bb_aug.y1_int
                    y2 = bb_aug.y2_int
                    top_row = image_aug[y1-2:y1+2, x1-2:x2+2]
                    btm_row = image_aug[y2-2:y2+2, x1-2:x2+2]
                    lft_row = image_aug[y1-2:y2+2, x1-2:x1+2]
                    rgt_row = image_aug[y1-2:y2+2, x2-2:x2+2]
                    assert np.max(top_row) > 10
                    assert np.max(btm_row) > 10
                    assert np.max(lft_row) > 10
                    assert np.max(rgt_row) > 10
                else:
                    count_bad += 1
        assert count_bad <= 1

    def test_empty_bounding_boxes(self):
        aug = iaa.ElasticTransformation(alpha=10, sigma=10)
        bbsoi = ia.BoundingBoxesOnImage([], shape=(10, 10, 3))

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        assert len(bbsoi_aug.bounding_boxes) == 0
        assert bbsoi_aug.shape == (10, 10, 3)

    # -----------
    # heatmaps alignment
    # -----------
    def test_image_heatmaps_alignment(self):
        # test alignment between images and heatmaps
        img = np.zeros((80, 80), dtype=np.uint8)
        img[:, 30:50] = 255
        img[30:50, :] = 255
        hm = HeatmapsOnImage(img.astype(np.float32)/255.0, shape=(80, 80))
        aug = iaa.ElasticTransformation(alpha=60.0, sigma=4.0, mode="constant",
                                        cval=0)
        aug_det = aug.to_deterministic()

        img_aug = aug_det.augment_image(img)
        hm_aug = aug_det.augment_heatmaps([hm])[0]

        img_aug_mask = img_aug > 255*0.1
        hm_aug_mask = hm_aug.arr_0to1 > 0.1
        same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
        assert hm_aug.shape == (80, 80)
        assert hm_aug.arr_0to1.shape == (80, 80, 1)
        assert (same / img_aug_mask.size) >= 0.99

    def test_image_heatmaps_alignment_if_heatmaps_smaller_than_image(self):
        # test alignment between images and heatmaps
        # here with heatmaps that are smaller than the image
        img = np.zeros((80, 80), dtype=np.uint8)
        img[:, 30:50] = 255
        img[30:50, :] = 255
        img_small = ia.imresize_single_image(
            img, (40, 40), interpolation="nearest")
        hm = HeatmapsOnImage(
            img_small.astype(np.float32)/255.0,
            shape=(80, 80))
        aug = iaa.ElasticTransformation(
            alpha=60.0, sigma=4.0, mode="constant", cval=0)
        aug_det = aug.to_deterministic()

        img_aug = aug_det.augment_image(img)
        hm_aug = aug_det.augment_heatmaps([hm])[0]

        img_aug_mask = img_aug > 255*0.1
        hm_aug_mask = ia.imresize_single_image(
            hm_aug.arr_0to1, (80, 80), interpolation="nearest"
        ) > 0.1
        same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
        assert hm_aug.shape == (80, 80)
        assert hm_aug.arr_0to1.shape == (40, 40, 1)
        assert (same / img_aug_mask.size) >= 0.94

    # -----------
    # segmaps alignment
    # -----------
    def test_image_segmaps_alignment(self):
        # test alignment between images and segmaps
        img = np.zeros((80, 80), dtype=np.uint8)
        img[:, 30:50] = 255
        img[30:50, :] = 255
        segmaps = SegmentationMapsOnImage(
            (img > 0).astype(np.int32),
            shape=(80, 80))
        aug = iaa.ElasticTransformation(
            alpha=60.0, sigma=4.0, mode="constant", cval=0, order=0)
        aug_det = aug.to_deterministic()

        img_aug = aug_det.augment_image(img)
        segmaps_aug = aug_det.augment_segmentation_maps([segmaps])[0]

        img_aug_mask = img_aug > 255*0.1
        segmaps_aug_mask = segmaps_aug.arr > 0
        same = np.sum(img_aug_mask == segmaps_aug_mask[:, :, 0])
        assert segmaps_aug.shape == (80, 80)
        assert segmaps_aug.arr.shape == (80, 80, 1)
        assert (same / img_aug_mask.size) >= 0.99

    def test_image_segmaps_alignment_if_heatmaps_smaller_than_image(self):
        # test alignment between images and segmaps
        # here with segmaps that are smaller than the image
        img = np.zeros((80, 80), dtype=np.uint8)
        img[:, 30:50] = 255
        img[30:50, :] = 255
        img_small = ia.imresize_single_image(
            img, (40, 40), interpolation="nearest")
        segmaps = SegmentationMapsOnImage(
            (img_small > 0).astype(np.int32), shape=(80, 80))
        aug = iaa.ElasticTransformation(
            alpha=60.0, sigma=4.0, mode="constant", cval=0, order=0)
        aug_det = aug.to_deterministic()

        img_aug = aug_det.augment_image(img)
        segmaps_aug = aug_det.augment_segmentation_maps([segmaps])[0]

        img_aug_mask = img_aug > 255*0.1
        segmaps_aug_mask = ia.imresize_single_image(
            segmaps_aug.arr, (80, 80), interpolation="nearest") > 0
        same = np.sum(img_aug_mask == segmaps_aug_mask[:, :, 0])
        assert segmaps_aug.shape == (80, 80)
        assert segmaps_aug.arr.shape == (40, 40, 1)
        assert (same / img_aug_mask.size) >= 0.94

    # ---------
    # unusual channel numbers
    # ---------
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
                aug = iaa.ElasticTransformation(alpha=2.0, sigma=2.0)

                image_aug = aug(image=image)

                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    # ---------
    # zero-sized axes
    # ---------
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
            for keep_size in [False, True]:
                with self.subTest(shape=shape, keep_size=keep_size):
                    for _ in sm.xrange(3):
                        image = np.zeros(shape, dtype=np.uint8)
                        aug = iaa.ElasticTransformation(alpha=2.0, sigma=2.0)

                        image_aug = aug(image=image)

                        assert image_aug.dtype.name == "uint8"
                        assert image_aug.shape == shape

    # -----------
    # get_parameters
    # -----------
    def test_get_parameters(self):
        aug = iaa.ElasticTransformation(
            alpha=0.25, sigma=1.0, order=2, cval=10, mode="constant")
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert isinstance(params[2], iap.Deterministic)
        assert isinstance(params[3], iap.Deterministic)
        assert isinstance(params[4], iap.Deterministic)
        assert 0.25 - 1e-8 < params[0].value < 0.25 + 1e-8
        assert 1.0 - 1e-8 < params[1].value < 1.0 + 1e-8
        assert params[2].value == 2
        assert params[3].value == 10
        assert params[4].value == "constant"

    # -----------
    # other dtypes
    # -----------
    def test_other_dtypes_bool(self):
        aug = iaa.ElasticTransformation(sigma=0.5, alpha=5, order=0)
        mask = np.zeros((21, 21), dtype=bool)
        mask[7:13, 7:13] = True

        image = np.zeros((21, 21), dtype=bool)
        image[mask] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == image.dtype.name
        assert not np.all(image_aug == 1)
        assert np.any(image_aug[~mask] == 1)

    def test_other_dtypes_uint_int(self):
        aug = iaa.ElasticTransformation(sigma=0.5, alpha=5, order=0)
        mask = np.zeros((21, 21), dtype=bool)
        mask[7:13, 7:13] = True

        dtypes = ["uint8", "uint16", "uint32", "int8", "int16", "int32"]
        for dtype in dtypes:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            image = np.zeros((21, 21), dtype=dtype)
            image[7:13, 7:13] = max_value

            image_aug = aug.augment_image(image)

            assert image_aug.dtype.name == dtype
            assert not np.all(image_aug == max_value)
            assert np.any(image_aug[~mask] == max_value)

    def test_other_dtypes_float(self):
        aug = iaa.ElasticTransformation(sigma=0.5, alpha=5, order=0)
        mask = np.zeros((21, 21), dtype=bool)
        mask[7:13, 7:13] = True

        for dtype in ["float16", "float32", "float64"]:
            def _isclose(a, b):
                atol = 1e-4 if dtype == "float16" else 1e-8
                return np.isclose(a, b, atol=atol, rtol=0)

            isize = np.dtype(dtype).itemsize
            values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1),
                      1000 ** (isize - 1)]
            values = values + [(-1) * value for value in values]
            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((21, 21), dtype=dtype)
                    image[7:13, 7:13] = value

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert not np.all(_isclose(image_aug, np.float128(value)))
                    assert np.any(_isclose(image_aug[~mask],
                                           np.float128(value)))

    def test_other_dtypes_bool_all_orders(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 20:30] = True
        mask[20:30, 10:40] = True

        for order in [0, 1, 2, 3, 4, 5]:
            aug = iaa.ElasticTransformation(sigma=1.0, alpha=50, order=order)

            image = np.zeros((50, 50), dtype=bool)
            image[mask] = True

            image_aug = aug.augment_image(image)

            assert image_aug.dtype.name == image.dtype.name
            assert not np.all(image_aug == 1)
            assert np.any(image_aug[~mask] == 1)

    def test_other_dtypes_uint_int_all_orders(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 20:30] = True
        mask[20:30, 10:40] = True

        for order in [0, 1, 2, 3, 4, 5]:
            aug = iaa.ElasticTransformation(sigma=1.0, alpha=50, order=order)

            dtypes = ["uint8", "uint16", "uint32", "uint64",
                      "int8", "int16", "int32", "int64"]
            if order == 0:
                dtypes = ["uint8", "uint16", "uint32",
                          "int8", "int16", "int32"]
            for dtype in dtypes:
                with self.subTest(dtype=dtype):
                    min_value, center_value, max_value = \
                        iadt.get_value_range_of_dtype(dtype)
                    dynamic_range = max_value - min_value

                    image = np.zeros((50, 50), dtype=dtype)
                    image[mask] = max_value
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.name == dtype
                    if order == 0:
                        assert not np.all(image_aug == max_value)
                        assert np.any(image_aug[~mask] == max_value)
                    else:
                        atol = 0.1 * dynamic_range
                        assert not np.all(
                            np.isclose(image_aug,
                                       max_value,
                                       rtol=0, atol=atol)
                        )
                        assert np.any(
                            np.isclose(image_aug[~mask],
                                       max_value,
                                       rtol=0, atol=atol))

    def test_other_dtypes_float_all_orders(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 20:30] = True
        mask[20:30, 10:40] = True

        for order in [0, 1, 2, 3, 4, 5]:
            aug = iaa.ElasticTransformation(sigma=1.0, alpha=50, order=order)

            dtypes = ["float16", "float32", "float64"]
            for dtype in dtypes:
                with self.subTest(dtype=dtype):
                    min_value, center_value, max_value = \
                        iadt.get_value_range_of_dtype(dtype)

                    def _isclose(a, b):
                        atol = 1e-4 if dtype == "float16" else 1e-8
                        return np.isclose(a, b, atol=atol, rtol=0)

                    value = (
                        0.1 * max_value
                        if dtype != "float64"
                        else 0.0001 * max_value)
                    image = np.zeros((50, 50), dtype=dtype)
                    image[mask] = value
                    image_aug = aug.augment_image(image)
                    if order == 0:
                        assert image_aug.dtype.name == dtype
                        assert not np.all(
                            _isclose(image_aug, np.float128(value))
                        )
                        assert np.any(
                            _isclose(image_aug[~mask], np.float128(value))
                        )
                    else:
                        atol = (
                            10
                            if dtype == "float16"
                            else 0.00001 * max_value)
                        assert not np.all(
                            np.isclose(
                                image_aug,
                                np.float128(value),
                                rtol=0, atol=atol
                            ))
                        assert np.any(
                            np.isclose(
                                image_aug[~mask],
                                np.float128(value),
                                rtol=0, atol=atol
                            ))

    def test_pickleable(self):
        aug = iaa.ElasticTransformation(alpha=(0.2, 1.5), sigma=(1.0, 10.0),
                                        random_state=1)
        runtest_pickleable_uint8_img(aug, iterations=4, shape=(25, 25, 1))


class _TwoValueParam(iap.StochasticParameter):
    def __init__(self, v1, v2):
        super(_TwoValueParam, self).__init__()
        self.v1 = v1
        self.v2 = v2

    def _draw_samples(self, size, random_state):
        arr = np.full(size, self.v1, dtype=np.int32)
        arr[1::2] = self.v2
        return arr


class TestRot90(unittest.TestCase):
    @property
    def kp_offset(self):
        # set this to -1 when using integer-based KP rotation instead of
        # subpixel/float-based rotation
        return 0

    @property
    def image(self):
        return np.arange(4*4*3).reshape((4, 4, 3)).astype(np.uint8)

    @property
    def heatmaps(self):
        return HeatmapsOnImage(self.image[..., 0:1].astype(np.float32) / 255,
                               shape=(4, 4, 3))

    @property
    def heatmaps_smaller(self):
        return HeatmapsOnImage(
            np.float32([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), shape=(4, 8, 3))

    @property
    def segmaps(self):
        return SegmentationMapsOnImage(
            self.image[..., 0:1].astype(np.int32), shape=(4, 4, 3))

    @property
    def segmaps_smaller(self):
        return SegmentationMapsOnImage(
            np.int32([[0, 1, 2], [3, 4, 5]]), shape=(4, 8, 3))

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=2, y=3)]
        return ia.KeypointsOnImage(kps, shape=(4, 8, 3))

    @property
    def psoi(self):
        return ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])],
            shape=(4, 8, 3)
        )

    @property
    def lsoi(self):
        return ia.LineStringsOnImage(
            [ia.LineString([(1, 1), (3, 1), (3, 3), (1, 3)])],
            shape=(4, 8, 3)
        )

    @property
    def bbsoi(self):
        return ia.BoundingBoxesOnImage(
            [ia.BoundingBox(x1=1, y1=1, x2=3, y2=3)],
            shape=(4, 8, 3)
        )

    @property
    def kpsoi_k1(self):
        # without keep size
        kp_offset = self.kp_offset
        expected_k1_kps = [(4-2+kp_offset, 1),
                           (4-3+kp_offset, 2)]
        kps = [ia.Keypoint(x, y) for x, y in expected_k1_kps]
        return ia.KeypointsOnImage(kps, shape=(8, 4, 3))

    @property
    def kpsoi_k2(self):
        # without keep size
        kp_offset = self.kp_offset
        expected_k1_kps = self.kpsoi_k1.to_xy_array()
        expected_k2_kps = [
            (8-expected_k1_kps[0][1]+kp_offset, expected_k1_kps[0][0]),
            (8-expected_k1_kps[1][1]+kp_offset, expected_k1_kps[1][0])]
        kps = [ia.Keypoint(x, y) for x, y in expected_k2_kps]
        return ia.KeypointsOnImage(kps, shape=(4, 8, 3))

    @property
    def kpsoi_k3(self):
        # without keep size
        kp_offset = self.kp_offset
        expected_k2_kps = self.kpsoi_k2.to_xy_array()
        expected_k3_kps = [
            (4-expected_k2_kps[0][1]+kp_offset, expected_k2_kps[0][0]),
            (4-expected_k2_kps[1][1]+kp_offset, expected_k2_kps[1][0])]
        kps = [ia.Keypoint(x, y) for x, y in expected_k3_kps]
        return ia.KeypointsOnImage(kps, shape=(8, 4, 3))

    @property
    def psoi_k1(self):
        # without keep size
        kp_offset = self.kp_offset
        expected_k1_polys = [(4-1+kp_offset, 1),
                             (4-1+kp_offset, 3),
                             (4-3+kp_offset, 3),
                             (4-3+kp_offset, 1)]
        return ia.PolygonsOnImage([ia.Polygon(expected_k1_polys)],
                                  shape=(8, 4, 3))

    @property
    def psoi_k2(self):
        # without keep size
        kp_offset = self.kp_offset
        expected_k1_polys = self.psoi_k1.polygons[0].exterior
        expected_k2_polys = [
            (8-expected_k1_polys[0][1]+kp_offset, expected_k1_polys[0][0]),
            (8-expected_k1_polys[1][1]+kp_offset, expected_k1_polys[1][0]),
            (8-expected_k1_polys[2][1]+kp_offset, expected_k1_polys[2][0]),
            (8-expected_k1_polys[3][1]+kp_offset, expected_k1_polys[3][0])]
        return ia.PolygonsOnImage([ia.Polygon(expected_k2_polys)],
                                  shape=(4, 8, 3))

    @property
    def psoi_k3(self):
        # without keep size
        kp_offset = self.kp_offset
        expected_k2_polys = self.psoi_k2.polygons[0].exterior
        expected_k3_polys = [
            (4-expected_k2_polys[0][1]+kp_offset, expected_k2_polys[0][0]),
            (4-expected_k2_polys[1][1]+kp_offset, expected_k2_polys[1][0]),
            (4-expected_k2_polys[2][1]+kp_offset, expected_k2_polys[2][0]),
            (4-expected_k2_polys[3][1]+kp_offset, expected_k2_polys[3][0])]
        return ia.PolygonsOnImage([ia.Polygon(expected_k3_polys)],
                                  shape=(8, 4, 3))

    @property
    def lsoi_k1(self):
        # without keep size
        kp_offset = self.kp_offset
        expected_k1_ls = [(4-1+kp_offset, 1),
                          (4-1+kp_offset, 3),
                          (4-3+kp_offset, 3),
                          (4-3+kp_offset, 1)]
        return ia.LineStringsOnImage([ia.LineString(expected_k1_ls)],
                                     shape=(8, 4, 3))

    @property
    def lsoi_k2(self):
        # without keep size
        kp_offset = self.kp_offset
        expected_k1_ls = self.psoi_k1.items[0].coords
        expected_k2_ls = [
            (8-expected_k1_ls[0][1]+kp_offset, expected_k1_ls[0][0]),
            (8-expected_k1_ls[1][1]+kp_offset, expected_k1_ls[1][0]),
            (8-expected_k1_ls[2][1]+kp_offset, expected_k1_ls[2][0]),
            (8-expected_k1_ls[3][1]+kp_offset, expected_k1_ls[3][0])]
        return ia.LineStringsOnImage([ia.LineString(expected_k2_ls)],
                                     shape=(4, 8, 3))

    @property
    def lsoi_k3(self):
        # without keep size
        kp_offset = self.kp_offset
        expected_k2_ls = self.lsoi_k2.items[0].coords
        expected_k3_ls = [
            (4-expected_k2_ls[0][1]+kp_offset, expected_k2_ls[0][0]),
            (4-expected_k2_ls[1][1]+kp_offset, expected_k2_ls[1][0]),
            (4-expected_k2_ls[2][1]+kp_offset, expected_k2_ls[2][0]),
            (4-expected_k2_ls[3][1]+kp_offset, expected_k2_ls[3][0])]
        return ia.LineStringsOnImage([ia.LineString(expected_k3_ls)],
                                     shape=(8, 4, 3))

    @property
    def bbsoi_k1(self):
        # without keep size
        kp_offset = self.kp_offset
        expected_k1_coords = [
            (4-1+kp_offset, 1),
            (4-3+kp_offset, 3)]
        return ia.BoundingBoxesOnImage([
            ia.BoundingBox(
                x1=min(expected_k1_coords[0][0], expected_k1_coords[1][0]),
                y1=min(expected_k1_coords[0][1], expected_k1_coords[1][1]),
                x2=max(expected_k1_coords[1][0], expected_k1_coords[0][0]),
                y2=max(expected_k1_coords[1][1], expected_k1_coords[0][1])
            )], shape=(8, 4, 3))

    @property
    def bbsoi_k2(self):
        # without keep size
        kp_offset = self.kp_offset
        coords = self.bbsoi_k1.bounding_boxes[0].coords
        expected_k2_coords = [
            (8-coords[0][1]+kp_offset, coords[0][0]),
            (8-coords[1][1]+kp_offset, coords[1][0])]
        return ia.BoundingBoxesOnImage([
            ia.BoundingBox(
                x1=min(expected_k2_coords[0][0], expected_k2_coords[1][0]),
                y1=min(expected_k2_coords[0][1], expected_k2_coords[1][1]),
                x2=max(expected_k2_coords[1][0], expected_k2_coords[0][0]),
                y2=max(expected_k2_coords[1][1], expected_k2_coords[0][1])
            )],
            shape=(4, 8, 3))

    @property
    def bbsoi_k3(self):
        # without keep size
        kp_offset = self.kp_offset
        coords = self.bbsoi_k2.bounding_boxes[0].coords
        expected_k3_coords = [
            (4-coords[0][1]+kp_offset, coords[0][0]),
            (4-coords[1][1]+kp_offset, coords[1][0])]
        return ia.BoundingBoxesOnImage([
            ia.BoundingBox(
                x1=min(expected_k3_coords[0][0], expected_k3_coords[1][0]),
                y1=min(expected_k3_coords[0][1], expected_k3_coords[1][1]),
                x2=max(expected_k3_coords[1][0], expected_k3_coords[0][0]),
                y2=max(expected_k3_coords[1][1], expected_k3_coords[0][1])
            )],
            shape=(8, 4, 3))

    def test___init___k_is_list(self):
        aug = iaa.Rot90([1, 3])
        assert isinstance(aug.k, iap.Choice)
        assert len(aug.k.a) == 2
        assert aug.k.a[0] == 1
        assert aug.k.a[1] == 3

    def test___init___k_is_all(self):
        aug = iaa.Rot90(ia.ALL)
        assert isinstance(aug.k, iap.Choice)
        assert len(aug.k.a) == 4
        assert aug.k.a == [0, 1, 2, 3]

    def test_images_k_is_0_and_4(self):
        for k in [0, 4]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                img_aug = aug.augment_image(self.image)

                assert img_aug.dtype.name == "uint8"
                assert np.array_equal(img_aug, self.image)

    def test_heatmaps_k_is_0_and_4(self):
        for k in [0, 4]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                hms_aug = aug.augment_heatmaps([self.heatmaps])[0]

                assert (hms_aug.arr_0to1.dtype.name
                        == self.heatmaps.arr_0to1.dtype.name)
                assert np.allclose(hms_aug.arr_0to1, self.heatmaps.arr_0to1)
                assert hms_aug.shape == self.heatmaps.shape

    def test_segmaps_k_is_0_and_4(self):
        for k in [0, 4]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                segmaps_aug = aug.augment_segmentation_maps(
                    [self.segmaps]
                )[0]

                assert (
                    segmaps_aug.arr.dtype.name
                    == self.segmaps.arr.dtype.name)
                assert np.allclose(segmaps_aug.arr, self.segmaps.arr)
                assert segmaps_aug.shape == self.segmaps.shape

    def test_keypoints_k_is_0_and_4(self):
        for k in [0, 4]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                kpsoi_aug = aug.augment_keypoints([self.kpsoi])[0]

                assert_cbaois_equal(kpsoi_aug, self.kpsoi)

    def test_polygons_k_is_0_and_4(self):
        for k in [0, 4]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                psoi_aug = aug.augment_polygons(self.psoi)

                assert_cbaois_equal(psoi_aug, self.psoi)

    def test_line_strings_k_is_0_and_4(self):
        for k in [0, 4]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                lsoi_aug = aug.augment_line_strings(self.lsoi)

                assert_cbaois_equal(lsoi_aug, self.lsoi)

    def test_bounding_boxes_k_is_0_and_4(self):
        for k in [0, 4]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                bbsoi_aug = aug.augment_bounding_boxes(self.bbsoi)

                assert_cbaois_equal(bbsoi_aug, self.bbsoi)

    def test_images_k_is_1_and_5(self):
        for k in [1, 5]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                img_aug = aug.augment_image(self.image)

                assert img_aug.dtype.name == "uint8"
                assert np.array_equal(img_aug,
                                      np.rot90(self.image, 1, axes=(1, 0)))

    def test_heatmaps_k_is_1_and_5(self):
        for k in [1, 5]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                hms_aug = aug.augment_heatmaps([self.heatmaps])[0]

                assert (hms_aug.arr_0to1.dtype.name
                        == self.heatmaps.arr_0to1.dtype.name)
                assert np.allclose(
                    hms_aug.arr_0to1,
                    np.rot90(self.heatmaps.arr_0to1, 1, axes=(1, 0)))
                assert hms_aug.shape == (4, 4, 3)

    def test_heatmaps_smaller_than_image_k_is_1_and_5(self):
        for k in [1, 5]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                hms_smaller_aug = aug.augment_heatmaps(
                    [self.heatmaps_smaller]
                )[0]

                assert (
                    hms_smaller_aug.arr_0to1.dtype.name
                    == self.heatmaps_smaller.arr_0to1.dtype.name)
                assert np.allclose(
                    hms_smaller_aug.arr_0to1,
                    np.rot90(self.heatmaps_smaller.arr_0to1, 1, axes=(1, 0)))
                assert hms_smaller_aug.shape == (8, 4, 3)

    def test_segmaps_k_is_1_and_5(self):
        for k in [1, 5]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                segmaps_aug = aug.augment_segmentation_maps(
                    [self.segmaps]
                )[0]

                assert (
                    segmaps_aug.arr.dtype.name
                    == self.segmaps.arr.dtype.name)
                assert np.allclose(
                    segmaps_aug.arr,
                    np.rot90(self.segmaps.arr, 1, axes=(1, 0)))
                assert segmaps_aug.shape == (4, 4, 3)

    def test_segmaps_smaller_than_image_k_is_1_and_5(self):
        for k in [1, 5]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                segmaps_smaller_aug = aug.augment_segmentation_maps(
                    self.segmaps_smaller)

                assert (
                    segmaps_smaller_aug.arr.dtype.name
                    == self.segmaps_smaller.arr.dtype.name)
                assert np.allclose(
                    segmaps_smaller_aug.arr,
                    np.rot90(self.segmaps_smaller.arr, 1, axes=(1, 0)))
                assert segmaps_smaller_aug.shape == (8, 4, 3)

    def test_keypoints_k_is_1_and_5(self):
        for k in [1, 5]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                kpsoi_aug = aug.augment_keypoints([self.kpsoi])[0]

                assert_cbaois_equal(kpsoi_aug, self.kpsoi_k1)

    def test_polygons_k_is_1_and_5(self):
        for k in [1, 5]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                psoi_aug = aug.augment_polygons(self.psoi)

                assert_cbaois_equal(psoi_aug, self.psoi_k1)

    def test_line_strings_k_is_1_and_5(self):
        for k in [1, 5]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                lsoi_aug = aug.augment_line_strings(self.lsoi)

                assert_cbaois_equal(lsoi_aug, self.lsoi_k1)

    def test_bounding_boxes_k_is_1_and_5(self):
        for k in [1, 5]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                bbsoi_aug = aug.augment_bounding_boxes(self.bbsoi)

                assert_cbaois_equal(bbsoi_aug, self.bbsoi_k1)

    def test_images_k_is_2(self):
        aug = iaa.Rot90(2, keep_size=False)
        img = self.image

        img_aug = aug.augment_image(img)

        assert img_aug.dtype.name == "uint8"
        assert np.array_equal(img_aug, np.rot90(img, 2, axes=(1, 0)))

    def test_heatmaps_k_is_2(self):
        aug = iaa.Rot90(2, keep_size=False)
        hms = self.heatmaps

        hms_aug = aug.augment_heatmaps([hms])[0]

        assert hms_aug.arr_0to1.dtype.name == hms.arr_0to1.dtype.name
        assert np.allclose(
            hms_aug.arr_0to1,
            np.rot90(hms.arr_0to1, 2, axes=(1, 0)))
        assert hms_aug.shape == (4, 4, 3)

    def test_heatmaps_smaller_than_image_k_is_2(self):
        aug = iaa.Rot90(2, keep_size=False)
        hms_smaller = self.heatmaps_smaller

        hms_smaller_aug = aug.augment_heatmaps([hms_smaller])[0]

        assert (hms_smaller_aug.arr_0to1.dtype.name
                == hms_smaller.arr_0to1.dtype.name)
        assert np.allclose(
            hms_smaller_aug.arr_0to1,
            np.rot90(hms_smaller.arr_0to1, 2, axes=(1, 0)))
        assert hms_smaller_aug.shape == (4, 8, 3)

    def test_segmaps_k_is_2(self):
        aug = iaa.Rot90(2, keep_size=False)
        segmaps = self.segmaps

        segmaps_aug = aug.augment_segmentation_maps([segmaps])[0]

        assert segmaps_aug.arr.dtype.name == segmaps.arr.dtype.name
        assert np.allclose(
            segmaps_aug.arr,
            np.rot90(segmaps.arr, 2, axes=(1, 0)))
        assert segmaps_aug.shape == (4, 4, 3)

    def test_segmaps_smaller_than_image_k_is_2(self):
        aug = iaa.Rot90(2, keep_size=False)
        segmaps_smaller = self.segmaps_smaller

        segmaps_smaller_aug = aug.augment_segmentation_maps(segmaps_smaller)

        assert (segmaps_smaller_aug.arr.dtype.name
                == segmaps_smaller.arr.dtype.name)
        assert np.allclose(
            segmaps_smaller_aug.arr,
            np.rot90(segmaps_smaller.arr, 2, axes=(1, 0)))
        assert segmaps_smaller_aug.shape == (4, 8, 3)

    def test_keypoints_k_is_2(self):
        aug = iaa.Rot90(2, keep_size=False)

        kpsoi_aug = aug.augment_keypoints([self.kpsoi])[0]

        assert_cbaois_equal(kpsoi_aug, self.kpsoi_k2)

    def test_polygons_k_is_2(self):
        aug = iaa.Rot90(2, keep_size=False)

        psoi_aug = aug.augment_polygons(self.psoi)

        assert_cbaois_equal(psoi_aug, self.psoi_k2)

    def test_line_strings_k_is_2(self):
        aug = iaa.Rot90(2, keep_size=False)

        lsoi_aug = aug.augment_line_strings(self.lsoi)

        assert_cbaois_equal(lsoi_aug, self.lsoi_k2)

    def test_bounding_boxes_k_is_2(self):
        aug = iaa.Rot90(2, keep_size=False)

        bbsoi_aug = aug.augment_bounding_boxes(self.bbsoi)

        assert_cbaois_equal(bbsoi_aug, self.bbsoi_k2)

    def test_images_k_is_3_and_minus1(self):
        img = self.image
        for k in [3, -1]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                img_aug = aug.augment_image(img)

                assert img_aug.dtype.name == "uint8"
                assert np.array_equal(img_aug, np.rot90(img, 3, axes=(1, 0)))

    def test_heatmaps_k_is_3_and_minus1(self):
        hms = self.heatmaps
        for k in [3, -1]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                hms_aug = aug.augment_heatmaps([hms])[0]

                assert (hms_aug.arr_0to1.dtype.name
                        == hms.arr_0to1.dtype.name)
                assert np.allclose(
                    hms_aug.arr_0to1,
                    np.rot90(hms.arr_0to1, 3, axes=(1, 0)))
                assert hms_aug.shape == (4, 4, 3)

    def test_heatmaps_smaller_than_image_k_is_3_and_minus1(self):
        hms_smaller = self.heatmaps_smaller
        for k in [3, -1]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                hms_smaller_aug = aug.augment_heatmaps([hms_smaller])[0]

                assert (hms_smaller_aug.arr_0to1.dtype.name
                        == hms_smaller.arr_0to1.dtype.name)
                assert np.allclose(
                    hms_smaller_aug.arr_0to1,
                    np.rot90(hms_smaller.arr_0to1, 3, axes=(1, 0)))
                assert hms_smaller_aug.shape == (8, 4, 3)

    def test_segmaps_k_is_3_and_minus1(self):
        segmaps = self.segmaps
        for k in [3, -1]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                segmaps_aug = aug.augment_segmentation_maps([segmaps])[0]

                assert (segmaps_aug.arr.dtype.name
                        == segmaps.arr.dtype.name)
                assert np.allclose(
                    segmaps_aug.arr,
                    np.rot90(segmaps.arr, 3, axes=(1, 0)))
                assert segmaps_aug.shape == (4, 4, 3)

    def test_segmaps_smaller_than_image_k_is_3_and_minus1(self):
        segmaps_smaller = self.segmaps_smaller
        for k in [3, -1]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                segmaps_smaller_aug = aug.augment_segmentation_maps(
                    segmaps_smaller)

                assert (segmaps_smaller_aug.arr.dtype.name
                        == segmaps_smaller.arr.dtype.name)
                assert np.allclose(
                    segmaps_smaller_aug.arr,
                    np.rot90(segmaps_smaller.arr, 3, axes=(1, 0)))
                assert segmaps_smaller_aug.shape == (8, 4, 3)

    def test_keypoints_k_is_3_and_minus1(self):
        for k in [3, -1]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                kpsoi_aug = aug.augment_keypoints([self.kpsoi])[0]

                assert_cbaois_equal(kpsoi_aug, self.kpsoi_k3)

    def test_polygons_k_is_3_and_minus1(self):
        for k in [3, -1]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                psoi_aug = aug.augment_polygons(self.psoi)

                assert_cbaois_equal(psoi_aug, self.psoi_k3)

    def test_line_strings_k_is_3_and_minus1(self):
        for k in [3, -1]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                lsoi_aug = aug.augment_line_strings(self.lsoi)

                assert_cbaois_equal(lsoi_aug, self.lsoi_k3)

    def test_bounding_boxes_k_is_3_and_minus1(self):
        for k in [3, -1]:
            with self.subTest(k=k):
                aug = iaa.Rot90(k, keep_size=False)

                bbsoi_aug = aug.augment_bounding_boxes(self.bbsoi)

                assert_cbaois_equal(bbsoi_aug, self.bbsoi_k3)

    def test_images_k_is_1_verify_without_using_numpy_rot90(self):
        # verify once without np.rot90
        aug = iaa.Rot90(k=1, keep_size=False)
        image = np.uint8([[1, 0, 0],
                          [0, 2, 0]])

        img_aug = aug.augment_image(image)

        expected = np.uint8([[0, 1], [2, 0], [0, 0]])
        assert np.array_equal(img_aug, expected)

    def test_images_k_is_1_keep_size_is_true(self):
        # keep_size=True, k=1
        aug = iaa.Rot90(1, keep_size=True)
        img_nonsquare = np.arange(5*4*3).reshape((5, 4, 3)).astype(np.uint8)

        img_aug = aug.augment_image(img_nonsquare)

        assert img_aug.dtype.name == "uint8"
        assert np.array_equal(
            img_aug,
            ia.imresize_single_image(
                np.rot90(img_nonsquare, 1, axes=(1, 0)),
                (5, 4)
            )
        )

    def test_heatmaps_k_is_1_keep_size_is_true(self):
        aug = iaa.Rot90(1, keep_size=True)
        hms = self.heatmaps

        hms_aug = aug.augment_heatmaps([hms])[0]

        assert hms_aug.arr_0to1.dtype.name == hms.arr_0to1.dtype.name
        assert np.allclose(
            hms_aug.arr_0to1,
            np.rot90(hms.arr_0to1, 1, axes=(1, 0)))
        assert hms_aug.shape == (4, 4, 3)

    def test_heatmaps_smaller_than_image_k_is_1_keep_size_is_true(self):
        aug = iaa.Rot90(1, keep_size=True)
        hms_smaller = self.heatmaps_smaller

        hms_smaller_aug = aug.augment_heatmaps([hms_smaller])[0]

        hms_smaller_rot = np.rot90(hms_smaller.arr_0to1, 1, axes=(1, 0))
        hms_smaller_rot = np.clip(
            ia.imresize_single_image(
                hms_smaller_rot, (2, 3), interpolation="cubic"
            ),
            0.0, 1.0)
        assert (hms_smaller_aug.arr_0to1.dtype.name
                == hms_smaller.arr_0to1.dtype.name)
        assert np.allclose(hms_smaller_aug.arr_0to1, hms_smaller_rot)
        assert hms_smaller_aug.shape == (4, 8, 3)

    def test_segmaps_k_is_1_keep_size_is_true(self):
        aug = iaa.Rot90(1, keep_size=True)
        segmaps = self.segmaps

        segmaps_aug = aug.augment_segmentation_maps([segmaps])[0]

        assert (segmaps_aug.arr.dtype.name
                == segmaps.arr.dtype.name)
        assert np.allclose(segmaps_aug.arr,
                           np.rot90(segmaps.arr, 1, axes=(1, 0)))
        assert segmaps_aug.shape == (4, 4, 3)

    def test_segmaps_smaller_than_image_k_is_1_keep_size_is_true(self):
        aug = iaa.Rot90(1, keep_size=True)
        segmaps_smaller = self.segmaps_smaller

        segmaps_smaller_aug = aug.augment_segmentation_maps(segmaps_smaller)

        segmaps_smaller_rot = np.rot90(segmaps_smaller.arr, 1, axes=(1, 0))
        segmaps_smaller_rot = ia.imresize_single_image(
            segmaps_smaller_rot, (2, 3), interpolation="nearest")
        assert (segmaps_smaller_aug.arr.dtype.name
                == segmaps_smaller.arr.dtype.name)
        assert np.allclose(segmaps_smaller_aug.arr, segmaps_smaller_rot)
        assert segmaps_smaller_aug.shape == (4, 8, 3)

    def test_keypoints_k_is_1_keep_size_is_true(self):
        aug = iaa.Rot90(1, keep_size=True)
        kp_offset = self.kp_offset
        kpsoi = self.kpsoi

        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]

        expected = [(4-2+kp_offset, 1), (4-3+kp_offset, 2)]
        expected = [(8*x/4, 4*y/8) for x, y in expected]
        assert kpsoi_aug.shape == (4, 8, 3)
        for kp_aug, kp in zip(kpsoi_aug.keypoints, expected):
            assert np.allclose([kp_aug.x, kp_aug.y], [kp[0], kp[1]])

    def test_polygons_k_is_1_keep_size_is_true(self):
        aug = iaa.Rot90(1, keep_size=True)
        psoi = self.psoi
        kp_offset = self.kp_offset

        psoi_aug = aug.augment_polygons(psoi)

        expected = [(4-1+kp_offset, 1), (4-1+kp_offset, 3),
                    (4-3+kp_offset, 3), (4-3+kp_offset, 1)]
        expected = [(8*x/4, 4*y/8) for x, y in expected]
        assert psoi_aug.shape == (4, 8, 3)
        assert len(psoi_aug.polygons) == 1
        assert psoi_aug.polygons[0].is_valid
        assert psoi_aug.polygons[0].exterior_almost_equals(expected)

    def test_line_strings_k_is_1_keep_size_is_true(self):
        aug = iaa.Rot90(1, keep_size=True)
        lsoi = self.lsoi
        kp_offset = self.kp_offset

        lsoi_aug = aug.augment_line_strings(lsoi)

        expected = [(4-1+kp_offset, 1), (4-1+kp_offset, 3),
                    (4-3+kp_offset, 3), (4-3+kp_offset, 1)]
        expected = [(8*x/4, 4*y/8) for x, y in expected]
        assert lsoi_aug.shape == (4, 8, 3)
        assert len(lsoi_aug.items) == 1
        assert lsoi_aug.items[0].coords_almost_equals(expected)

    def test_bounding_boxes_k_is_1_keep_size_is_true(self):
        aug = iaa.Rot90(1, keep_size=True)
        bbsoi = self.bbsoi
        kp_offset = self.kp_offset

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        expected = [(4-1+kp_offset, 1),
                    (4-3+kp_offset, 3)]
        expected = [(8*x/4, 4*y/8) for x, y in expected]
        expected = np.float32([
            [min(expected[0][0], expected[1][0]),
             min(expected[0][1], expected[1][1])],
            [max(expected[0][0], expected[1][0]),
             max(expected[0][1], expected[1][1])]
        ])
        assert bbsoi_aug.shape == (4, 8, 3)
        assert len(bbsoi_aug.bounding_boxes) == 1
        assert bbsoi_aug.bounding_boxes[0].coords_almost_equals(expected)

    def test_images_k_is_list(self):
        aug = iaa.Rot90(_TwoValueParam(1, 2), keep_size=False)
        img = self.image

        imgs_aug = aug.augment_images([img] * 4)

        assert np.array_equal(imgs_aug[0], np.rot90(img, 1, axes=(1, 0)))
        assert np.array_equal(imgs_aug[1], np.rot90(img, 2, axes=(1, 0)))
        assert np.array_equal(imgs_aug[2], np.rot90(img, 1, axes=(1, 0)))
        assert np.array_equal(imgs_aug[3], np.rot90(img, 2, axes=(1, 0)))

    def test_heatmaps_smaller_than_image_k_is_list(self):
        def _rot_hm(hm, k):
            return np.rot90(hm.arr_0to1, k, axes=(1, 0))

        aug = iaa.Rot90(_TwoValueParam(1, 2), keep_size=False)
        hms_smaller = self.heatmaps_smaller

        hms_aug = aug.augment_heatmaps([hms_smaller] * 4)

        assert hms_aug[0].shape == (8, 4, 3)
        assert hms_aug[1].shape == (4, 8, 3)
        assert hms_aug[2].shape == (8, 4, 3)
        assert hms_aug[3].shape == (4, 8, 3)
        assert np.allclose(hms_aug[0].arr_0to1, _rot_hm(hms_smaller, 1))
        assert np.allclose(hms_aug[1].arr_0to1, _rot_hm(hms_smaller, 2))
        assert np.allclose(hms_aug[2].arr_0to1, _rot_hm(hms_smaller, 1))
        assert np.allclose(hms_aug[3].arr_0to1, _rot_hm(hms_smaller, 2))

    def test_segmaps_smaller_than_image_k_is_list(self):
        def _rot_sm(segmap, k):
            return np.rot90(segmap.arr, k, axes=(1, 0))

        aug = iaa.Rot90(_TwoValueParam(1, 2), keep_size=False)
        segmaps_smaller = self.segmaps_smaller

        segmaps_aug = aug.augment_segmentation_maps([segmaps_smaller] * 4)

        assert segmaps_aug[0].shape == (8, 4, 3)
        assert segmaps_aug[1].shape == (4, 8, 3)
        assert segmaps_aug[2].shape == (8, 4, 3)
        assert segmaps_aug[3].shape == (4, 8, 3)
        assert np.allclose(segmaps_aug[0].arr, _rot_sm(segmaps_smaller, 1))
        assert np.allclose(segmaps_aug[1].arr, _rot_sm(segmaps_smaller, 2))
        assert np.allclose(segmaps_aug[2].arr, _rot_sm(segmaps_smaller, 1))
        assert np.allclose(segmaps_aug[3].arr, _rot_sm(segmaps_smaller, 2))

    def test_keypoints_k_is_list(self):
        aug = iaa.Rot90(_TwoValueParam(1, 2), keep_size=False)
        kpsoi = self.kpsoi

        kpsoi_aug = aug.augment_keypoints([kpsoi] * 4)

        assert_cbaois_equal(kpsoi_aug[0], self.kpsoi_k1)
        assert_cbaois_equal(kpsoi_aug[1], self.kpsoi_k2)
        assert_cbaois_equal(kpsoi_aug[2], self.kpsoi_k1)
        assert_cbaois_equal(kpsoi_aug[3], self.kpsoi_k2)

    def test_polygons_k_is_list(self):
        aug = iaa.Rot90(_TwoValueParam(1, 2), keep_size=False)
        psoi = self.psoi

        psoi_aug = aug.augment_polygons([psoi] * 4)

        assert_cbaois_equal(psoi_aug[0], self.psoi_k1)
        assert_cbaois_equal(psoi_aug[1], self.psoi_k2)
        assert_cbaois_equal(psoi_aug[2], self.psoi_k1)
        assert_cbaois_equal(psoi_aug[3], self.psoi_k2)

    def test_line_strings_k_is_list(self):
        aug = iaa.Rot90(_TwoValueParam(1, 2), keep_size=False)
        lsoi = self.lsoi

        lsoi_aug = aug.augment_line_strings([lsoi] * 4)

        assert_cbaois_equal(lsoi_aug[0], self.lsoi_k1)
        assert_cbaois_equal(lsoi_aug[1], self.lsoi_k2)
        assert_cbaois_equal(lsoi_aug[2], self.lsoi_k1)
        assert_cbaois_equal(lsoi_aug[3], self.lsoi_k2)

    def test_bounding_boxes_k_is_list(self):
        aug = iaa.Rot90(_TwoValueParam(1, 2), keep_size=False)
        bbsoi = self.bbsoi

        bbsoi_aug = aug.augment_bounding_boxes([bbsoi] * 4)

        assert_cbaois_equal(bbsoi_aug[0], self.bbsoi_k1)
        assert_cbaois_equal(bbsoi_aug[1], self.bbsoi_k2)
        assert_cbaois_equal(bbsoi_aug[2], self.bbsoi_k1)
        assert_cbaois_equal(bbsoi_aug[3], self.bbsoi_k2)

    def test_empty_keypoints(self):
        aug = iaa.Rot90(k=1, keep_size=False)
        kpsoi = ia.KeypointsOnImage([], shape=(4, 8, 3))

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        expected = self.kpsoi_k1
        expected.keypoints = []
        assert_cbaois_equal(kpsoi_aug, expected)

    def test_empty_polygons(self):
        aug = iaa.Rot90(k=1, keep_size=False)
        psoi = ia.PolygonsOnImage([], shape=(4, 8, 3))

        psoi_aug = aug.augment_polygons(psoi)

        expected = self.psoi_k1
        expected.polygons = []
        assert_cbaois_equal(psoi_aug, expected)

    def test_empty_line_strings(self):
        aug = iaa.Rot90(k=1, keep_size=False)
        lsoi = ia.LineStringsOnImage([], shape=(4, 8, 3))

        lsoi_aug = aug.augment_line_strings(lsoi)

        expected = self.lsoi_k1
        expected.line_strings = []
        assert_cbaois_equal(lsoi_aug, expected)

    def test_empty_bounding_boxes(self):
        aug = iaa.Rot90(k=1, keep_size=False)
        bbsoi = ia.BoundingBoxesOnImage([], shape=(4, 8, 3))

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        expected = self.bbsoi_k1
        expected.bounding_boxes = []
        assert_cbaois_equal(bbsoi_aug, expected)

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
                aug = iaa.Rot90(k=1)

                image_aug = aug(image=image)

                shape_expected = tuple([shape[1], shape[0]] + list(shape[2:]))
                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape_expected

    def test_zero_sized_axes_k_0_or_2(self):
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
            for keep_size in [False, True]:
                with self.subTest(shape=shape, keep_size=keep_size):
                    for _ in sm.xrange(10):
                        image = np.zeros(shape, dtype=np.uint8)
                        aug = iaa.Rot90([0, 2], keep_size=keep_size)

                        image_aug = aug(image=image)

                        assert image_aug.shape == shape

    def test_zero_sized_axes_k_1_or_3_no_keep_size(self):
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
                for _ in sm.xrange(10):
                    image = np.zeros(shape, dtype=np.uint8)
                    aug = iaa.Rot90([1, 3], keep_size=False)

                    image_aug = aug(image=image)

                    shape_expected = tuple([shape[1], shape[0]]
                                           + list(shape[2:]))
                    assert image_aug.shape == shape_expected

    def test_zero_sized_axes_k_1_or_3_keep_size(self):
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
                for _ in sm.xrange(10):
                    image = np.zeros(shape, dtype=np.uint8)
                    aug = iaa.Rot90([1, 3], keep_size=True)

                    image_aug = aug(image=image)

                    assert image_aug.shape == image.shape

    def test_get_parameters(self):
        aug = iaa.Rot90([1, 3], keep_size=False)
        assert aug.get_parameters()[0] == aug.k
        assert aug.get_parameters()[1] is False

    def test_other_dtypes_bool(self):
        aug = iaa.Rot90(2)

        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == image.dtype.name
        assert np.all(image_aug[0, 0] == 0)
        assert np.all(image_aug[2, 2] == 1)

    def test_other_dtypes_uint_int(self):
        aug = iaa.Rot90(2)

        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                image = np.zeros((3, 3), dtype=dtype)
                image[0, 0] = max_value

                image_aug = aug.augment_image(image)

                assert image_aug.dtype.name == dtype
                assert np.all(image_aug[0, 0] == 0)
                assert np.all(image_aug[2, 2] == max_value)

    def test_other_dtypes_float(self):
        aug = iaa.Rot90(2)

        dtypes = ["float16", "float32", "float64", "float128"]
        for dtype in dtypes:
            def _allclose(a, b):
                atol = 1e-4 if dtype == "float16" else 1e-8
                return np.allclose(a, b, atol=atol, rtol=0)

            isize = np.dtype(dtype).itemsize
            values = [0, 1.0, 10.0, 100.0, 500 ** (isize-1), 1000 ** (isize-1)]
            values = values + [(-1) * value for value in values]
            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((3, 3), dtype=dtype)
                    image[0, 0] = value

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert _allclose(image_aug[0, 0], 0)
                    assert _allclose(image_aug[2, 2], np.float128(value))

    def test_pickleable(self):
        aug = iaa.Rot90([0, 1, 2, 3], random_state=1)
        runtest_pickleable_uint8_img(aug, iterations=5)


class TestWithPolarWarping(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___single_augmenter_as_child(self):
        aug = iaa.WithPolarWarping(iaa.Noop())
        assert isinstance(aug.children, iaa.Sequential)
        assert isinstance(aug.children[0], iaa.Noop)

    def test___init___list_of_augmenters_as_child(self):
        aug = iaa.WithPolarWarping([iaa.Noop(), iaa.Noop()])
        assert isinstance(aug.children, iaa.Sequential)
        assert isinstance(aug.children[0], iaa.Noop)
        assert isinstance(aug.children[1], iaa.Noop)

    def test_images_no_change(self):
        image = np.mod(np.arange(10*20*3), 255).astype(np.uint8)
        image = image.reshape((10, 20, 3))
        aug = iaa.WithPolarWarping(iaa.Noop())

        image_aug = aug(image=image)

        avg_dist = np.average(
            np.abs(
                image_aug.astype(np.int32)[2:-2, 2:-2]
                - image.astype(np.int32)[2:-2, 2:-2]
            )
        )
        assert image_aug.shape == (10, 20, 3)
        assert avg_dist < 7.0

    def test_heatmaps_no_change(self):
        hm = np.linspace(0, 1.0, 10*20, dtype=np.float32).reshape((10, 20, 1))
        hm = ia.HeatmapsOnImage(hm, shape=(10, 20, 3))
        aug = iaa.WithPolarWarping(iaa.Noop())

        hm_aug = aug(heatmaps=hm)

        avg_dist = np.average(
            np.abs(
                hm_aug.get_arr()[2:-2, 2:-2]
                - hm.get_arr()[2:-2, 2:-2]
            )
        )
        assert hm_aug.shape == (10, 20, 3)
        assert avg_dist < 0.0125

    def test_segmentation_maps_no_change(self):
        sm = np.zeros((10, 20, 1), dtype=np.int32)
        sm[1, 0:5] = 1
        sm[3:3, 3:3] = 2
        sm[7:9, :] = 3
        sm = ia.SegmentationMapsOnImage(sm, shape=(10, 20, 3))
        aug = iaa.WithPolarWarping(iaa.Noop())

        sm_aug = aug(segmentation_maps=sm)

        p_same = np.average(
            sm_aug.get_arr()[2:-2, 2:-2]
            == sm.get_arr()[2:-2, 2:-2]
        )
        assert sm_aug.shape == (10, 20, 3)
        assert p_same > 0.95

    def test_keypoints_no_change(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=5, y=5),
               ia.Keypoint(x=5, y=9)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(10, 20, 3))
        aug = iaa.WithPolarWarping(iaa.Noop())

        kpsoi_aug = aug(keypoints=kpsoi)

        assert kpsoi_aug.shape == (10, 20, 3)
        assert np.allclose(kpsoi_aug.to_xy_array(), kpsoi.to_xy_array(),
                           atol=0.01)

    def test_bounding_boxes_no_change(self):
        bbs = [
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4, label="foo"),
            ia.BoundingBox(x1=3, y1=5, x2=7, y2=10),
        ]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(10, 20, 3))
        aug = iaa.WithPolarWarping(iaa.Noop())

        bbsoi_aug = aug(bounding_boxes=bbsoi)

        assert bbsoi_aug.items[0].label == "foo"
        assert bbsoi_aug.items[1].label is None
        assert bbsoi_aug.shape == (10, 20, 3)
        assert np.allclose(bbsoi_aug.to_xy_array(), bbsoi.to_xy_array(),
                           atol=0.01)

    def test_polygons_no_change(self):
        ps = [
            ia.Polygon([(0, 2), (4, 2), (4, 4)], label="foo"),
            ia.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        ]
        psoi = ia.PolygonsOnImage(ps, shape=(10, 20, 3))
        aug = iaa.WithPolarWarping(iaa.Noop())

        psoi_aug = aug(polygons=psoi)

        assert psoi_aug.items[0].label == "foo"
        assert psoi_aug.items[1].label is None
        assert psoi_aug.shape == (10, 20, 3)
        assert np.allclose(psoi_aug.to_xy_array(), psoi.to_xy_array(),
                           atol=0.01)

    def test_line_strings_no_change(self):
        ls = [
            ia.LineString([(0, 2), (4, 2), (4, 4)]),
            ia.LineString([(0, 0), (5, 0), (5, 5), (0, 5)])
        ]
        lsoi = ia.LineStringsOnImage(ls, shape=(10, 20, 3))
        aug = iaa.WithPolarWarping(iaa.Noop())

        lsoi_aug = aug(line_strings=lsoi)

        assert lsoi_aug.shape == (10, 20, 3)
        assert np.allclose(lsoi_aug.to_xy_array(), lsoi.to_xy_array(),
                           atol=0.01)

    def test_bounding_boxes_and_polygons_provided_no_change(self):
        bbs = [
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4, label="foo"),
            ia.BoundingBox(x1=3, y1=5, x2=7, y2=10),
        ]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(10, 20, 3))
        ps = [
            ia.Polygon([(0, 2), (4, 2), (4, 4)], label="foo"),
            ia.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        ]
        psoi = ia.PolygonsOnImage(ps, shape=(10, 20, 3))

        aug = iaa.WithPolarWarping(iaa.Noop())

        aug = aug.to_deterministic()
        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)
        psoi_aug = aug.augment_polygons(psoi)

        assert bbsoi_aug.items[0].label == "foo"
        assert bbsoi_aug.items[1].label is None
        assert bbsoi_aug.shape == (10, 20, 3)
        assert np.allclose(bbsoi_aug.to_xy_array(), bbsoi.to_xy_array(),
                           atol=0.01)

        assert psoi_aug.items[0].label == "foo"
        assert psoi_aug.items[1].label is None
        assert psoi_aug.shape == (10, 20, 3)
        assert np.allclose(psoi_aug.to_xy_array(), psoi.to_xy_array(),
                           atol=0.01)

    def test_images_translation_x(self):
        image = np.zeros((50, 70, 3), dtype=np.uint8)
        image[20-1:20+1, 30-1:30+1, 0] = 255
        image[30-1:30+1, 40-1:40+1, 1] = 255
        aug = iaa.WithPolarWarping(iaa.Affine(translate_px={"x": 15}))

        image_aug = aug(image=image)

        x1 = np.argmax(np.max(image_aug[..., 0], axis=0))
        y1 = np.argmax(np.max(image_aug[..., 0], axis=1))
        x2 = np.argmax(np.max(image_aug[..., 1], axis=0))
        y2 = np.argmax(np.max(image_aug[..., 1], axis=1))

        # translation on x axis in polar representation should move all points
        # a bit away from the center
        min_diff = 4
        assert image_aug.shape == (50, 70, 3)
        assert x1 < 30 - min_diff
        assert y1 < 20 - min_diff
        assert x2 > 40 + min_diff
        assert y2 > 30 + min_diff

    def test_heatmaps_translation_x(self):
        hm = np.zeros((50, 70, 2), dtype=np.float32)
        hm[20-1:20+1, 30-1:30+1, 0] = 1.0
        hm[30-1:30+1, 40-1:40+1, 1] = 1.0
        hm = ia.HeatmapsOnImage(hm, shape=(50, 70, 3))
        aug = iaa.WithPolarWarping(iaa.Affine(translate_px={"x": 15}))

        hm_aug = aug(heatmaps=hm)

        hm_aug_arr = hm_aug.get_arr()
        x1 = np.argmax(np.max(hm_aug_arr[..., 0], axis=0))
        y1 = np.argmax(np.max(hm_aug_arr[..., 0], axis=1))
        x2 = np.argmax(np.max(hm_aug_arr[..., 1], axis=0))
        y2 = np.argmax(np.max(hm_aug_arr[..., 1], axis=1))

        # translation on x axis in polar representation should move all points
        # a bit away from the center
        min_diff = 4
        assert hm_aug_arr.shape == (50, 70, 2)
        assert hm_aug.shape == (50, 70, 3)
        assert x1 < 30 - min_diff
        assert y1 < 20 - min_diff
        assert x2 > 40 + min_diff
        assert y2 > 30 + min_diff

    def test_segmentation_maps_translation_x(self):
        sm = np.zeros((50, 70, 2), dtype=np.int32)
        sm[20-1:20+1, 30-1:30+1, 0] = 1
        sm[30-1:30+1, 40-1:40+1, 1] = 2
        sm = ia.SegmentationMapsOnImage(sm, shape=(50, 70, 3))
        aug = iaa.WithPolarWarping(iaa.Affine(translate_px={"x": 15}))

        sm_aug = aug(segmentation_maps=sm)

        sm_aug_arr = sm_aug.get_arr()
        x1 = np.argmax(np.max(sm_aug_arr[..., 0], axis=0))
        y1 = np.argmax(np.max(sm_aug_arr[..., 0], axis=1))
        x2 = np.argmax(np.max(sm_aug_arr[..., 1], axis=0))
        y2 = np.argmax(np.max(sm_aug_arr[..., 1], axis=1))

        # translation on x axis in polar representation should move all points
        # a bit away from the center
        min_diff = 4
        assert sm_aug_arr.shape == (50, 70, 2)
        assert sm_aug.shape == (50, 70, 3)
        assert x1 < 30 - min_diff
        assert y1 < 20 - min_diff
        assert x2 > 40 + min_diff
        assert y2 > 30 + min_diff

    def test_keypoints_translation_x(self):
        cbas = [ia.Keypoint(y=20, x=30), ia.Keypoint(y=30, x=40)]
        cbaoi = ia.KeypointsOnImage(cbas, shape=(50, 70, 3))
        aug = iaa.WithPolarWarping(iaa.Affine(translate_px={"x": 15}))

        cbaoi_aug = aug(keypoints=cbaoi)

        x1 = cbaoi_aug.items[0].x
        y1 = cbaoi_aug.items[0].y
        x2 = cbaoi_aug.items[1].x
        y2 = cbaoi_aug.items[1].y

        # translation on x axis in polar representation should move all points
        # a bit away from the center
        min_diff = 4
        assert cbaoi_aug.shape == (50, 70, 3)
        assert x1 < 30 - min_diff
        assert y1 < 20 - min_diff
        assert x2 > 40 + min_diff
        assert y2 > 30 + min_diff

    def test_bounding_boxes_translation_x(self):
        cbas = [ia.BoundingBox(y1=20, x1=30, y2=20+2, x2=30+2),
                ia.BoundingBox(y1=30, x1=40, y2=30+2, x2=40+2)]
        cbaoi = ia.BoundingBoxesOnImage(cbas, shape=(50, 70, 3))
        aug = iaa.WithPolarWarping(iaa.Affine(translate_px={"x": 15}))

        cbaoi_aug = aug(bounding_boxes=cbaoi)

        x1 = cbaoi_aug.items[0].x1
        y1 = cbaoi_aug.items[0].y1
        x2 = cbaoi_aug.items[1].x2
        y2 = cbaoi_aug.items[1].y2

        # translation on x axis in polar representation should move all points
        # a bit away from the center
        min_diff = 4
        assert cbaoi_aug.shape == (50, 70, 3)
        assert x1 < 30 - min_diff
        assert y1 < 20 - min_diff
        assert x2 > 40 + min_diff
        assert y2 > 30 + min_diff

    def test_polygons_translation_x(self):
        cbas = [ia.Polygon([(30, 20), (30+2, 20), (30+2, 20+2)]),
                ia.Polygon([(40, 30), (40+2, 30), (40+2, 30+2)])]
        cbaoi = ia.PolygonsOnImage(cbas, shape=(50, 70, 3))
        aug = iaa.WithPolarWarping(iaa.Affine(translate_px={"x": 15}))

        cbaoi_aug = aug(polygons=cbaoi)

        x1 = cbaoi_aug.items[0].coords[0][0]
        y1 = cbaoi_aug.items[0].coords[0][1]
        x2 = cbaoi_aug.items[1].coords[2][0]
        y2 = cbaoi_aug.items[1].coords[2][1]

        # translation on x axis in polar representation should move all points
        # a bit away from the center
        min_diff = 4
        assert cbaoi_aug.shape == (50, 70, 3)
        assert x1 < 30 - min_diff
        assert y1 < 20 - min_diff
        assert x2 > 40 + min_diff
        assert y2 > 30 + min_diff

    def test_line_strings_translation_x(self):
        cbas = [ia.LineString([(30, 20), (30+2, 20), (30+2, 20+2)]),
                ia.LineString([(40, 30), (40+2, 30), (40+2, 30+2)])]
        cbaoi = ia.LineStringsOnImage(cbas, shape=(50, 70, 3))
        aug = iaa.WithPolarWarping(iaa.Affine(translate_px={"x": 15}))

        cbaoi_aug = aug(line_strings=cbaoi)

        x1 = cbaoi_aug.items[0].coords[0][0]
        y1 = cbaoi_aug.items[0].coords[0][1]
        x2 = cbaoi_aug.items[1].coords[2][0]
        y2 = cbaoi_aug.items[1].coords[2][1]

        # translation on x axis in polar representation should move all points
        # a bit away from the center
        min_diff = 4
        assert cbaoi_aug.shape == (50, 70, 3)
        assert x1 < 30 - min_diff
        assert y1 < 20 - min_diff
        assert x2 > 40 + min_diff
        assert y2 > 30 + min_diff

    def test_image_heatmap_alignment(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[40-10:40+10, 50-10:50+10, :] = 255
        hm = np.zeros((40, 50, 1), dtype=np.float32)
        hm[20-5:20+5, 25-5:25+5, :] = 1.0
        hm = ia.HeatmapsOnImage(hm, shape=image.shape)
        aug = iaa.WithPolarWarping(iaa.Affine(translate_px={"x": 10}))

        image_aug, hm_aug = aug(image=image, heatmaps=hm)

        hm_aug_arr = hm_aug.get_arr()
        hm_aug_arr_rs = ia.imresize_single_image(hm_aug_arr, (80, 100),
                                                 interpolation="nearest")
        overlap = np.average(
            (image_aug[..., 0] > 200)
            == (hm_aug_arr_rs[..., 0] > 0.9)
        )
        assert image_aug.shape == (80, 100, 3)
        assert hm_aug.shape == (80, 100, 3)
        assert hm_aug_arr.shape == (40, 50, 1)
        assert overlap > 0.96

    def test_image_segmentation_map_alignment(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[40-10:40+10, 50-10:50+10, :] = 255
        sm = np.zeros((40, 50, 1), dtype=np.int32)
        sm[20-5:20+5, 25-5:25+5, :] = 1
        sm = ia.SegmentationMapsOnImage(sm, shape=image.shape)
        aug = iaa.WithPolarWarping(iaa.Affine(translate_px={"x": 10}))

        image_aug, sm_aug = aug(image=image, segmentation_maps=sm)

        sm_aug_arr = sm_aug.get_arr()
        sm_aug_arr_rs = ia.imresize_single_image(sm_aug_arr, (80, 100),
                                                 interpolation="nearest")
        overlap = np.average(
            (image_aug[..., 0] > 200)
            == (sm_aug_arr_rs[..., 0] == 1)
        )
        assert image_aug.shape == (80, 100, 3)
        assert sm_aug.shape == (80, 100, 3)
        assert sm_aug_arr.shape == (40, 50, 1)
        assert overlap > 0.96

    def test_image_keypoint_alignment(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[40-10:40-10+3, 50-10:50-10+3, :] = 255
        image[40+10:40+10+3, 50+10:50+10+3, :] = 255

        kps = [ia.Keypoint(y=40-10+1.5, x=50-10+1.5),
               ia.Keypoint(y=40+10+1.5, x=50+10+1.5)]
        kpsoi = ia.KeypointsOnImage(kps, shape=image.shape)
        aug = iaa.WithPolarWarping(iaa.Affine(translate_px={"x": 10}))

        image_aug, kpsoi_aug = aug(image=image, keypoints=kpsoi)

        kp1 = kpsoi_aug.items[0]
        kp2 = kpsoi_aug.items[1]
        kp1_intensity = image_aug[int(kp1.y), int(kp1.x), 0]
        kp2_intensity = image_aug[int(kp2.y), int(kp2.x), 0]
        assert image_aug.shape == (80, 100, 3)
        assert kpsoi_aug.shape == (80, 100, 3)
        assert kp1_intensity > 200
        assert kp2_intensity > 200

    def test_image_is_noncontiguous(self):
        image = np.mod(np.arange(10*20*3), 255).astype(np.uint8)
        image = image.reshape((10, 20, 3))
        image_cp = np.fliplr(np.copy(image))
        image = np.fliplr(image)
        assert image.flags["C_CONTIGUOUS"] is False
        aug = iaa.WithPolarWarping(iaa.Noop())

        image_aug = aug(image=image)

        avg_dist = np.average(
            np.abs(
                image_aug.astype(np.int32)[2:-2, 2:-2]
                - image_cp.astype(np.int32)[2:-2, 2:-2]
            )
        )
        assert image_aug.shape == (10, 20, 3)
        assert avg_dist < 7.0

    def test_image_is_view(self):
        image = np.mod(np.arange(10*20*3), 255).astype(np.uint8)
        image = image.reshape((10, 20, 3))
        image_cp = np.copy(image)[2:, 2:, :]
        image = image[2:, 2:, :]
        assert image.flags["OWNDATA"] is False
        aug = iaa.WithPolarWarping(iaa.Noop())

        image_aug = aug(image=image)

        avg_dist = np.average(
            np.abs(
                image_aug.astype(np.int32)[2:-2, 2:-2]
                - image_cp.astype(np.int32)[2:-2, 2:-2]
            )
        )
        assert image_aug.shape == (8, 18, 3)
        assert avg_dist < 7.0

    def test_propagation_hooks(self):
        image = np.mod(np.arange(30*30), 255).astype(np.uint8)
        image = image.reshape((30, 30))
        aug = iaa.WithPolarWarping(iaa.Add(50))

        def _propagator(images, augmenter, parents, default):
            return False if augmenter is aug else default

        hooks = ia.HooksImages(propagator=_propagator)

        observed1 = aug.augment_image(image)
        observed2 = aug.augment_image(image, hooks=hooks)

        image_plus50 = np.clip(image.astype(np.int32)+50, 0, 255)
        diff1 = np.abs(observed1[2:-2].astype(np.int32)
                       - image_plus50[2:-2].astype(np.int32))
        diff2 = np.abs(observed2[2:-2].astype(np.int32)
                       - image_plus50[2:-2].astype(np.int32))
        overlap_1_add = np.average(diff1 <= 1)
        overlap_2_add = np.average(diff2 <= 2)
        assert overlap_1_add >= 0.9
        assert overlap_2_add < 0.01

    def test_unusual_channel_numbers(self):
        shapes = [
            (5, 5, 4),
            (5, 5, 5),
            (5, 5, 512),
            (5, 5, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.WithPolarWarping(iaa.Noop())

                image_aug = aug(image=image)

                shape_expected = tuple([shape[1], shape[0]] + list(shape[2:]))
                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape_expected

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
                kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)],
                                            shape=image.shape)
                sm_arr = np.zeros((3, 3), dtype=np.int32)
                sm_arr[1, 1] = 1
                sm = ia.SegmentationMapsOnImage(sm_arr, shape=image.shape)
                aug = iaa.WithPolarWarping(iaa.Noop())

                aug_det = aug.to_deterministic()
                image_aug = aug(image=image)
                kpsoi_aug = aug(keypoints=kpsoi)
                sm_aug = aug(segmentation_maps=sm)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape
                assert np.allclose(kpsoi_aug.to_xy_array(),
                                   kpsoi.to_xy_array())
                assert kpsoi_aug.shape == shape
                assert np.array_equal(sm_aug.get_arr(), sm_arr)
                assert sm_aug.shape == shape

    def test_other_dtypes_bool(self):
        aug = iaa.WithPolarWarping(iaa.Noop())
        arr = np.zeros((20, 20), dtype=bool)
        arr[10-3:10+3, 10-3:10+3] = True

        arr_aug = aug(image=arr)

        overlap = np.average(arr_aug == arr)
        assert arr_aug.shape == (20, 20)
        assert arr_aug.dtype.name == "bool"
        assert overlap > 0.95

    def test_other_dtypes_uint_int(self):
        aug = iaa.WithPolarWarping(iaa.Noop())

        dtypes = ["uint8", "uint16",
                  "int8", "int16", "int32",]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                center_value = int(center_value)

                image = np.zeros((30, 10), dtype=dtype)
                image[0:10, :] = min_value
                image[10:20, :] = center_value
                image[20:30, :] = max_value
                image = iaa.pad(image, top=2, right=2, bottom=2, left=2,
                                cval=0)

                image_aug = aug.augment_image(image)
                image_aug = image_aug[2:-2, 2:-2]

                overlap_min = np.average(image_aug[0:10] == min_value)
                overlap_cv = np.average(image_aug[10:20] == center_value)
                overlap_max = np.average(image_aug[20:30] == max_value)
                assert image_aug.dtype.name == dtype
                assert overlap_min > 0.9
                assert overlap_cv > 0.9
                assert overlap_max > 0.9

    def test_other_dtypes_float(self):
        def _avg_close(arr_aug, expected_val):
            atol = 1e-8
            return np.average(np.isclose(arr_aug, expected_val,
                                         rtol=0, atol=atol))

        aug = iaa.WithPolarWarping(iaa.Noop())

        dtypes = ["float16", "float32", "float64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                center_value = center_value

                image = np.zeros((70, 10), dtype=dtype)
                image[0:10, :] = min_value
                image[10:20, :] = center_value
                image[20:30, :] = max_value
                image[30:40, :] = -1.0
                image[40:50, :] = 1.0
                image[50:60, :] = -100.0
                image[60:70, :] = 100.0
                image = iaa.pad(image, top=2, right=2, bottom=2, left=2,
                                cval=0)

                image_aug = aug.augment_image(image)
                image_aug = image_aug[2:-2, 2:-2]

                overlap1 = _avg_close(image_aug[0:10], min_value)
                overlap2 = _avg_close(image_aug[10:20], center_value)
                overlap3 = _avg_close(image_aug[20:30], max_value)
                overlap4 = _avg_close(image_aug[30:40], -1.0)
                overlap5 = _avg_close(image_aug[40:50], 1.0)
                overlap6 = _avg_close(image_aug[50:60], -100.0)
                overlap7 = _avg_close(image_aug[60:70], 100.0)
                assert image_aug.dtype.name == dtype
                assert overlap1 > 0.9
                assert overlap2 > 0.9
                assert overlap3 > 0.9
                assert overlap4 > 0.9
                assert overlap5 > 0.9
                assert overlap6 > 0.9
                assert overlap7 > 0.9

    def test_get_parameters(self):
        aug = iaa.WithPolarWarping(iaa.Noop())
        params = aug.get_parameters()
        assert len(params) == 0

    def test_get_children_lists(self):
        children = iaa.Sequential([iaa.Noop()])
        aug = iaa.WithPolarWarping(children)
        assert aug.get_children_lists() == [children]

    def test_to_deterministic(self):
        child = iaa.Identity()
        aug = iaa.WithPolarWarping([child])

        aug_det = aug.to_deterministic()

        assert aug_det.deterministic
        assert aug_det.random_state is not aug.random_state
        assert aug_det.children.deterministic
        assert aug_det.children[0].deterministic

    def test___repr___and___str__(self):
        children = iaa.Sequential([iaa.Noop()])
        aug = iaa.WithPolarWarping(children, name="WithPolarWarpingTest")
        expected = (
            "WithPolarWarping("
            "name=WithPolarWarpingTest, "
            "children=%s, "
            "deterministic=False"
            ")" % (str(children),))

        assert aug.__repr__() == expected
        assert aug.__str__() == expected

    def test_pickleable(self):
        aug = iaa.WithPolarWarping(
            iaa.Affine(translate_px=(0, 10), random_state=1),
            random_state=2)
        runtest_pickleable_uint8_img(aug, iterations=5, shape=(25, 25, 1))


class Test_apply_jigsaw(unittest.TestCase):
    def test_no_movement(self):
        dtypes = ["bool",
                  "uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64",
                  "float16", "float32", "float64", "float128"]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                arr = np.arange(20*20*1).reshape((20, 20, 1))
                if dtype == "bool":
                    mask = np.logical_or(
                        arr % 4 == 0,
                        arr % 7 == 0)
                    arr[mask] = 1
                    arr[~mask] = 0
                arr = arr.astype(dtype)
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                arr[0, 0] = min_value
                arr[0, 1] = max_value

                destinations = np.arange(5*5).reshape((5, 5))

                observed = iaa.apply_jigsaw(arr, destinations)

                if arr.dtype.kind != "f":
                    assert np.array_equal(observed, arr)
                else:
                    atol = 1e-4 if dtype == "float16" else 1e-8
                    assert np.allclose(observed, arr, rtol=0, atol=atol)

    def test_no_movement_zero_sized_axes(self):
        sizes = [
            (0, 1),
            (1, 0),
            (0, 0)
        ]

        dtype = "uint8"
        for size in sizes:
            with self.subTest(size=size):
                arr = np.zeros(size, dtype=dtype)
                destinations = np.arange(1*1).reshape((1, 1))

                observed = iaa.apply_jigsaw(arr, destinations)

                assert np.array_equal(observed, arr)

    def _test_two_cells_moved__n_channels(self, nb_channels):
        dtypes = ["bool",
                  "uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64",
                  "float16", "float32", "float64", "float128"]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                c = 1 if nb_channels is None else nb_channels
                arr = np.arange(20*20*c)
                if dtype == "bool":
                    mask = np.logical_or(
                        arr % 4 == 0,
                        arr % 7 == 0)
                    arr[mask] = 1
                    arr[~mask] = 0
                if nb_channels is not None:
                    arr = arr.reshape((20, 20, c))
                else:
                    arr = arr.reshape((20, 20))
                arr = arr.astype(dtype)
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                arr[0, 0] = min_value
                arr[0, 1] = max_value

                destinations = np.arange(5*5).reshape((5, 5))
                destinations[0, 0] = 4  # cell 0 will be filled with 4
                destinations[0, 4] = 0  # cell 4 will be filled with 0
                destinations[0, 1] = 6  # cell 1 will be filled with 6
                destinations[1, 1] = 1  # cell 6 will be filled with 1

                observed = iaa.apply_jigsaw(arr, destinations)

                cell_0_obs = observed[0:4, 0:4]
                cell_0_exp = arr[0:4, 16:20]
                cell_4_obs = observed[0:4, 16:20]
                cell_4_exp = arr[0:4, 0:4]
                cell_1_obs = observed[0:4, 4:8]
                cell_1_exp = arr[4:8, 4:8]
                cell_6_obs = observed[4:8, 4:8]
                cell_6_exp = arr[0:4, 4:8]
                cell_2_obs = observed[0:4, 8:12]
                cell_2_exp = arr[0:4, 8:12]
                if arr.dtype.kind != "f":
                    assert np.array_equal(cell_0_obs, cell_0_exp)
                    assert np.array_equal(cell_4_obs, cell_4_exp)
                    assert np.array_equal(cell_1_obs, cell_1_exp)
                    assert np.array_equal(cell_6_obs, cell_6_exp)
                    assert np.array_equal(cell_2_obs, cell_2_exp)
                else:
                    atol = 1e-4 if dtype == "float16" else 1e-8
                    kwargs = {"rtol": 0, "atol": atol}
                    assert np.allclose(cell_0_obs, cell_0_exp, **kwargs)
                    assert np.allclose(cell_4_obs, cell_4_exp, **kwargs)
                    assert np.allclose(cell_1_obs, cell_1_exp, **kwargs)
                    assert np.allclose(cell_6_obs, cell_6_exp, **kwargs)
                    assert np.allclose(cell_2_obs, cell_2_exp, **kwargs)

                assert observed.shape == arr.shape
                assert observed.dtype.name == dtype

    def test_two_cells_moved__no_channels(self):
        self._test_two_cells_moved__n_channels(None)

    def test_two_cells_moved__1_channel(self):
        self._test_two_cells_moved__n_channels(1)

    def test_two_cells_moved__3_channels(self):
        self._test_two_cells_moved__n_channels(3)


class Test_apply_jigsaw_to_coords(unittest.TestCase):
    def test_no_movement(self):
        arr = np.float32([
            (0.0, 0.0),
            (5.0, 5.0),
            (25.0, 50.5),
            (10.01, 21.0)
        ])
        destinations = np.arange(10*10).reshape((10, 10))

        observed = iaa.apply_jigsaw_to_coords(arr, destinations, (50, 100))

        assert np.allclose(observed, arr)

    def test_with_movement(self):
        arr = np.float32([
            (0.0, 0.0),  # in cell (0,0) = idx 0
            (5.0, 5.0),  # in cell (0,0) = idx 0
            (25.0, 50.5),  # in cell (5,2) = idx 52
            (10.01, 21.0)  # in cell (2,1) = idx 12
        ])
        destinations = np.arange(10*10).reshape((10, 10))
        destinations[0, 0] = 1
        destinations[0, 1] = 0
        destinations[5, 2] = 7
        destinations[0, 7] = 52

        observed = iaa.apply_jigsaw_to_coords(arr, destinations, (100, 100))

        expected = np.float32([
            (10.0, 0.0),
            (15.0, 5.0),
            (75.0, 0.5),
            (10.01, 21.0)
        ])
        assert np.allclose(observed, expected)

    def test_with_movement_non_square_image(self):
        arr = np.float32([
            (0.5, 0.6),  # in cell (0,0) = idx 0
            (180.7, 90.8),  # in cell (9,9) = idx 99
        ])
        destinations = np.arange(10*10).reshape((10, 10))
        destinations[0, 0] = 99
        destinations[9, 9] = 0

        observed = iaa.apply_jigsaw_to_coords(arr, destinations, (100, 200))

        expected = np.float32([
            (180+0.5, 90+0.6),
            (0+0.7, 0+0.8)
        ])
        assert np.allclose(observed, expected)

    def test_empty_coords(self):
        arr = np.zeros((0, 2), dtype=np.float32)
        destinations = np.arange(10*10).reshape((10, 10))

        observed = iaa.apply_jigsaw_to_coords(arr, destinations, (100, 100))

        assert np.allclose(observed, arr)


class Test_generate_jigsaw_destinations(unittest.TestCase):
    def test_max_steps_0(self):
        rng = iarandom.RNG(0)
        max_steps = 0
        rows = 10
        cols = 20

        observed = iaa.generate_jigsaw_destinations(rows, cols, max_steps, rng,
                                                    connectivity=8)

        assert np.array_equal(
            observed,
            np.arange(rows*cols).reshape((rows, cols)))

    def test_max_steps_1(self):
        rng = iarandom.RNG(0)
        max_steps = 1
        rows = 10
        cols = 20

        observed = iaa.generate_jigsaw_destinations(rows, cols, max_steps, rng,
                                                    connectivity=8)

        yy = (observed // cols).reshape((rows, cols))
        xx = np.mod(observed, cols).reshape((rows, cols))
        yy_expected = np.tile(np.arange(rows).reshape((rows, 1)), (1, cols))
        xx_expected = np.tile(np.arange(cols).reshape((1, cols)), (rows, 1))

        yy_diff = yy_expected - yy
        xx_diff = xx_expected - xx
        dist = np.sqrt(yy_diff ** 2 + xx_diff ** 2)

        assert np.min(dist) <= 0.01
        assert np.any(dist >= np.sqrt(2) - 1e-4)
        assert np.max(dist) <= np.sqrt(2) + 1e-4

    def test_max_steps_1_connectivity_4(self):
        rng = iarandom.RNG(0)
        max_steps = 1
        rows = 10
        cols = 20

        observed = iaa.generate_jigsaw_destinations(rows, cols, max_steps, rng,
                                                    connectivity=4)

        yy = (observed // cols).reshape((rows, cols))
        xx = np.mod(observed, cols).reshape((rows, cols))
        yy_expected = np.tile(np.arange(rows).reshape((rows, 1)), (1, cols))
        xx_expected = np.tile(np.arange(cols).reshape((1, cols)), (rows, 1))

        yy_diff = yy_expected - yy
        xx_diff = xx_expected - xx
        dist = np.sqrt(yy_diff ** 2 + xx_diff ** 2)

        assert np.min(dist) <= 0.01
        assert np.any(dist >= 0.99)
        assert np.max(dist) <= 1.01


class TestJigsaw(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___defaults(self):
        aug = iaa.Jigsaw(nb_rows=1, nb_cols=2)
        assert aug.nb_rows.value == 1
        assert aug.nb_cols.value == 2
        assert aug.max_steps.value == 2
        assert aug.allow_pad is True

    def test___init___custom(self):
        aug = iaa.Jigsaw(nb_rows=1, nb_cols=2, max_steps=3, allow_pad=False)
        assert aug.nb_rows.value == 1
        assert aug.nb_cols.value == 2
        assert aug.max_steps.value == 3
        assert aug.allow_pad is False

    def test__draw_samples(self):
        aug = iaa.Jigsaw(nb_rows=(1, 5), nb_cols=(1, 6), max_steps=(1, 3))
        batch = mock.Mock()
        batch.nb_rows = 100

        samples = aug._draw_samples(batch, iarandom.RNG(0))

        assert len(np.unique(samples.nb_rows)) > 1
        assert len(np.unique(samples.nb_cols)) > 1
        assert len(np.unique(samples.max_steps)) > 1
        assert np.all(samples.nb_rows >= 1)
        assert np.all(samples.nb_rows <= 5)
        assert np.all(samples.nb_cols >= 1)
        assert np.all(samples.nb_cols <= 6)
        assert np.all(samples.max_steps >= 1)
        assert np.all(samples.max_steps <= 3)

        all_same = True
        first = samples.destinations[0]
        for dest in samples.destinations:
            this_same = (dest.shape == first.shape
                         and np.array_equal(dest, first))
            all_same = all_same and this_same
        assert not all_same

    def test_images_without_shifts(self):
        aug = iaa.Jigsaw(nb_rows=2, nb_cols=2, max_steps=0)
        image = np.mod(np.arange(20*20*3), 255).astype(np.uint8)
        image = image.reshape((20, 20, 3))

        image_aug = aug(image=image)

        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == (20, 20, 3)
        assert np.array_equal(image_aug, image)

    def test_heatmaps_without_shifts(self):
        aug = iaa.Jigsaw(nb_rows=2, nb_cols=2, max_steps=0)
        arr = np.linspace(0, 1.0, 20*20*1).astype(np.float32)
        arr = arr.reshape((20, 20, 1))
        heatmap = ia.HeatmapsOnImage(arr, shape=(20, 20, 3))

        heatmap_aug = aug(heatmaps=heatmap)

        assert heatmap_aug.shape == (20, 20, 3)
        assert np.allclose(heatmap_aug.arr_0to1, heatmap.arr_0to1)

    def test_segmaps_without_shifts(self):
        aug = iaa.Jigsaw(nb_rows=2, nb_cols=2, max_steps=0)
        arr = np.zeros((20, 20, 1), dtype=np.int32)
        arr[0:10, :] = 1
        arr[10:20, 10:20] = 2
        arr = arr.reshape((20, 20, 1))
        segmap = ia.SegmentationMapsOnImage(arr, shape=(20, 20, 3))

        segmap_aug = aug(segmentation_maps=segmap)

        assert segmap_aug.shape == (20, 20, 3)
        assert np.array_equal(segmap_aug.arr, segmap.arr)

    def test_keypoints_without_shifts(self):
        aug = iaa.Jigsaw(nb_rows=2, nb_cols=2, max_steps=0)
        kpsoi = ia.KeypointsOnImage.from_xy_array([
            (0, 0),
            (5.5, 3.5),
            (12.1, 23.5)
        ], shape=(20, 20, 3))

        kpsoi_aug = aug(keypoints=kpsoi)

        assert kpsoi_aug.shape == (20, 20, 3)
        assert np.allclose(kpsoi_aug.to_xy_array(), kpsoi.to_xy_array())

    def test_images_with_shifts(self):
        # these rows/cols/max_steps parameters are mostly ignored due to the
        # mocked _draw_samples method below
        aug = iaa.Jigsaw(nb_rows=2, nb_cols=2, max_steps=1)
        image = np.mod(np.arange(19*19*3), 255).astype(np.uint8)
        image = image.reshape((19, 19, 3))
        destinations = np.array([
            [3, 1],
            [2, 0]
        ], dtype=np.int32)

        old_func = aug._draw_samples

        def _mocked_draw_samples(batch, random_state):
            samples = old_func(batch, random_state)
            return geometriclib._JigsawSamples(
                nb_rows=samples.nb_rows,
                nb_cols=samples.nb_cols,
                max_steps=samples.max_steps,
                destinations=[destinations])

        aug._draw_samples = _mocked_draw_samples

        image_aug = aug(image=image)

        expected = iaa.pad(image, bottom=1, right=1, cval=0)
        expected = iaa.apply_jigsaw(expected, destinations)
        assert np.array_equal(image_aug, expected)

    def test_heatmaps_with_shifts(self):
        # these rows/cols/max_steps parameters are mostly ignored due to the
        # mocked _draw_samples method below
        aug = iaa.Jigsaw(nb_rows=2, nb_cols=2, max_steps=1)
        arr = np.linspace(0, 1.0, 18*18*1).astype(np.float32)
        arr = arr.reshape((18, 18, 1))
        heatmap = ia.HeatmapsOnImage(arr, shape=(19, 19, 3))
        destinations = np.array([
            [3, 1],
            [2, 0]
        ], dtype=np.int32)

        old_func = aug._draw_samples

        def _mocked_draw_samples(batch, random_state):
            samples = old_func(batch, random_state)
            return geometriclib._JigsawSamples(
                nb_rows=samples.nb_rows,
                nb_cols=samples.nb_cols,
                max_steps=samples.max_steps,
                destinations=[destinations])

        aug._draw_samples = _mocked_draw_samples

        heatmap_aug = aug(heatmaps=heatmap)

        expected = ia.imresize_single_image(arr, (19, 19),
                                            interpolation="cubic")
        expected = np.clip(expected, 0, 1.0)
        expected = iaa.pad(expected, bottom=1, right=1, cval=0.0)
        expected = iaa.apply_jigsaw(expected, destinations)
        expected = ia.imresize_single_image(expected, (18, 18),
                                            interpolation="cubic")
        expected = np.clip(expected, 0, 1.0)
        assert np.allclose(heatmap_aug.arr_0to1, expected)

    def test_segmaps_with_shifts(self):
        # these rows/cols/max_steps parameters are mostly ignored due to the
        # mocked _draw_samples method below
        aug = iaa.Jigsaw(nb_rows=2, nb_cols=2, max_steps=1)
        arr = np.zeros((18, 18, 1), dtype=np.int32)
        arr[0:10, :] = 1
        arr[10:18, 10:18] = 2
        arr = arr.reshape((18, 18, 1))
        segmap = ia.SegmentationMapsOnImage(arr, shape=(19, 19, 3))
        destinations = np.array([
            [3, 1],
            [2, 0]
        ], dtype=np.int32)

        old_func = aug._draw_samples

        def _mocked_draw_samples(batch, random_state):
            samples = old_func(batch, random_state)
            return geometriclib._JigsawSamples(
                nb_rows=samples.nb_rows,
                nb_cols=samples.nb_cols,
                max_steps=samples.max_steps,
                destinations=[destinations])

        aug._draw_samples = _mocked_draw_samples

        segmap_aug = aug(segmentation_maps=segmap)

        expected = ia.imresize_single_image(arr, (19, 19),
                                            interpolation="nearest")
        expected = iaa.pad(expected, bottom=1, right=1, cval=0)
        expected = iaa.apply_jigsaw(expected, destinations)
        expected = ia.imresize_single_image(expected, (18, 18),
                                            interpolation="nearest")
        assert np.array_equal(segmap_aug.arr, expected)

    def test_keypoints_with_shifts(self):
        # these rows/cols/max_steps parameters are mostly ignored due to the
        # mocked _draw_samples method below
        aug = iaa.Jigsaw(nb_rows=5, nb_cols=5, max_steps=1)
        kpsoi = ia.KeypointsOnImage.from_xy_array([
            (0, 0),
            (5.5, 3.5),
            (4.0, 12.5),
            (11.1, 11.2),
            (12.1, 23.5)
        ], shape=(18, 18, 3))
        destinations = np.array([
            [3, 1],
            [2, 0]
        ], dtype=np.int32)

        old_func = aug._draw_samples

        def _mocked_draw_samples(batch, random_state):
            samples = old_func(batch, random_state)
            return geometriclib._JigsawSamples(
                nb_rows=samples.nb_rows,
                nb_cols=samples.nb_cols,
                max_steps=samples.max_steps,
                destinations=[destinations])

        aug._draw_samples = _mocked_draw_samples

        kpsoi_aug = aug(keypoints=kpsoi)

        expected = kpsoi.deepcopy()
        expected.shape = (20, 20, 3)
        # (0.0, 0.0) to cell at bottom-right, 1px pad at top and left
        expected.keypoints[0].x = 10.0 + (0.0 - 0.0) + 1.0
        expected.keypoints[0].y = 10.0 + (0.0 - 0.0) + 1.0
        # (5.5, 3.5) to cell at bottom-right, 1px pad at top and left
        expected.keypoints[1].x = 10.0 + (5.5 - 0.0) + 1.0
        expected.keypoints[1].y = 10.0 + (3.5 - 0.0) + 1.0
        # (4.0, 12.5) not moved to other cell, but 1px pad at top and left
        expected.keypoints[2].x = 4.0 + 1.0
        expected.keypoints[2].y = 12.5 + 1.0
        # (11.0, 11.0) to cell at top-left, 1px pad at top and left
        expected.keypoints[3].x = 0.0 + (11.1 - 10.0) + 1.0
        expected.keypoints[3].y = 0.0 + (11.2 - 10.0) + 1.0
        # (12.1, 23.5) not moved to other cell, but 1px pad at top and left
        expected.keypoints[4].x = 12.1 + 1.0
        expected.keypoints[4].y = 23.5 + 1.0
        expected.shape = (20, 20, 3)
        assert kpsoi_aug.shape == (20, 20, 3)
        assert np.allclose(kpsoi_aug.to_xy_array(), expected.to_xy_array())

    def test_images_and_heatmaps_aligned(self):
        nb_changed = 0
        rs = iarandom.RNG(0)
        for _ in np.arange(10):
            aug = iaa.Jigsaw(nb_rows=(2, 5), nb_cols=(2, 5), max_steps=(0, 3))
            image_small = rs.integers(0, 10, size=(10, 15)).astype(np.float32)
            image_small = image_small / 10.0
            image = ia.imresize_single_image(image_small, (20, 30),
                                             interpolation="cubic")
            image = np.clip(image, 0, 1.0)
            hm = ia.HeatmapsOnImage(image_small, shape=(20, 30))

            images_aug, hms_aug = aug(images=[image, image, image],
                                      heatmaps=[hm, hm, hm])

            for image_aug, hm_aug in zip(images_aug, hms_aug):
                # TODO added squeeze here because get_arr() falsely returns
                #      (H,W,1) for 2D inputs
                arr = np.squeeze(hm_aug.get_arr())
                image_aug_rs = ia.imresize_single_image(
                    image_aug.astype(np.float32),
                    arr.shape[0:2],
                    interpolation="cubic")
                image_aug_rs = np.clip(image_aug_rs, 0, 1.0)
                overlap = np.average(np.isclose(image_aug_rs, arr))

                assert overlap > 0.99
                if not np.array_equal(arr, hm.get_arr()):
                    nb_changed += 1
        assert nb_changed > 5

    def test_images_and_segmaps_aligned(self):
        nb_changed = 0
        rs = iarandom.RNG(0)
        for _ in np.arange(10):
            aug = iaa.Jigsaw(nb_rows=(2, 5), nb_cols=(2, 5), max_steps=(0, 3))
            image_small = rs.integers(0, 10, size=(10, 15))
            image = ia.imresize_single_image(image_small, (20, 30),
                                             interpolation="nearest")
            image = image.astype(np.uint8)
            segm = ia.SegmentationMapsOnImage(image_small, shape=(20, 30))

            images_aug, sms_aug = aug(images=[image, image, image],
                                      segmentation_maps=[segm, segm, segm])

            for image_aug, sm_aug in zip(images_aug, sms_aug):
                arr = sm_aug.get_arr()
                image_aug_rs = ia.imresize_single_image(
                    image_aug, arr.shape[0:2], interpolation="nearest")
                overlap = np.average(image_aug_rs == arr)

                assert overlap > 0.99
                if not np.array_equal(arr, segm.arr):
                    nb_changed += 1
        assert nb_changed > 5

    def test_images_and_keypoints_aligned(self):
        for i in np.arange(20):
            aug = iaa.Jigsaw(nb_rows=(1, 3), nb_cols=(1, 3), max_steps=(2, 5),
                             random_state=i)
            # make sure that these coords are not exactly at a grid cell
            # border with any possibly sampled height/width in grid cells
            y = 17.5
            x = 25.5
            kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=x, y=y)],
                                        shape=(20, 30))
            image = np.zeros((20, 30), dtype=np.uint8)
            image[int(y), int(x)] = 255

            images_aug, kpsois_aug = aug(images=[image, image, image],
                                         keypoints=[kpsoi, kpsoi, kpsoi])

            for image_aug, kpsoi_aug in zip(images_aug, kpsois_aug):
                x_aug = kpsoi_aug.keypoints[0].x
                y_aug = kpsoi_aug.keypoints[0].y
                idx = np.argmax(image_aug)
                y_aug_img, x_aug_img = np.unravel_index(idx,
                                                        image_aug.shape)
                dist = np.sqrt((x_aug - x_aug_img)**2 + (y_aug - y_aug_img)**2)
                # best possible distance is about 0.7 as KP coords are in cell
                # center and sampled coords are at cell top left
                assert dist < 0.8

    def test_no_error_for_1x1_grids(self):
        aug = iaa.Jigsaw(nb_rows=1, nb_cols=1, max_steps=2)
        image = np.mod(np.arange(19*19*3), 255).astype(np.uint8)
        image = image.reshape((19, 19, 3))
        kpsoi = ia.KeypointsOnImage.from_xy_array([
            (0, 0),
            (5.5, 3.5),
            (4.0, 12.5),
            (11.1, 11.2),
            (12.1, 23.5)
        ], shape=(19, 19, 3))

        image_aug, kpsoi_aug = aug(image=image, keypoints=kpsoi)

        assert np.array_equal(image_aug, image)
        assert np.allclose(kpsoi_aug.to_xy_array(), kpsoi.to_xy_array())

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
                for _ in sm.xrange(3):
                    image = np.zeros(shape, dtype=np.uint8)
                    aug = iaa.Jigsaw(nb_rows=2, nb_cols=2, max_steps=2)

                    image_aug = aug(image=image)

                    # (2, 2, [C]) here, because rows/cols are padded to be
                    # multiple of nb_rows and nb_cols
                    shape_exp = tuple([2, 2] + list(shape[2:]))
                    assert image_aug.dtype.name == "uint8"
                    assert np.array_equal(image_aug,
                                          np.zeros(shape_exp, dtype=np.uint8))

    def test_get_parameters(self):
        aug = iaa.Jigsaw(nb_rows=1, nb_cols=2)
        params = aug.get_parameters()
        assert params[0] is aug.nb_rows
        assert params[1] is aug.nb_cols
        assert params[2] is aug.max_steps
        assert params[3] is True
