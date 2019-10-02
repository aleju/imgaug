from __future__ import print_function, division, absolute_import

import sys
import warnings
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
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug import random as iarandom
import imgaug.augmenters.size as iaa_size
from imgaug.testutils import (array_equal_lists, keypoints_equal, reseed,
                              assert_cbaois_equal, shift_cbaoi)
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class Test__handle_position_parameter(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_string_uniform(self):
        observed = iaa_size._handle_position_parameter("uniform")
        assert isinstance(observed, tuple)
        assert len(observed) == 2
        for i in range(2):
            assert isinstance(observed[i], iap.Uniform)
            assert isinstance(observed[i].a, iap.Deterministic)
            assert isinstance(observed[i].b, iap.Deterministic)
            assert 0.0 - 1e-4 < observed[i].a.value < 0.0 + 1e-4
            assert 1.0 - 1e-4 < observed[i].b.value < 1.0 + 1e-4

    def test_string_center(self):
        observed = iaa_size._handle_position_parameter("center")
        assert isinstance(observed, tuple)
        assert len(observed) == 2
        for i in range(2):
            assert isinstance(observed[i], iap.Deterministic)
            assert 0.5 - 1e-4 < observed[i].value < 0.5 + 1e-4

    def test_string_normal(self):
        observed = iaa_size._handle_position_parameter("normal")
        assert isinstance(observed, tuple)
        assert len(observed) == 2
        for i in range(2):
            assert isinstance(observed[i], iap.Clip)
            assert isinstance(observed[i].other_param, iap.Normal)
            assert isinstance(observed[i].other_param.loc, iap.Deterministic)
            assert isinstance(observed[i].other_param.scale, iap.Deterministic)
            assert 0.5 - 1e-4 < observed[i].other_param.loc.value < 0.5 + 1e-4
            assert 0.35/2 - 1e-4 < observed[i].other_param.scale.value < 0.35/2 + 1e-4

    def test_xy_axis_combined_strings(self):
        pos_x = [
            ("left", 0.0),
            ("center", 0.5),
            ("right", 1.0)
        ]
        pos_y = [
            ("top", 0.0),
            ("center", 0.5),
            ("bottom", 1.0)
        ]
        for x_str, x_val in pos_x:
            for y_str, y_val in pos_y:
                position = "%s-%s" % (x_str, y_str)
                with self.subTest(position=position):
                    observed = iaa_size._handle_position_parameter(position)
                    assert isinstance(observed[0], iap.Deterministic)
                    assert x_val - 1e-4 < observed[0].value < x_val + 1e-4
                    assert isinstance(observed[1], iap.Deterministic)
                    assert y_val - 1e-4 < observed[1].value < y_val + 1e-4

    def test_stochastic_parameter(self):
        observed = iaa_size._handle_position_parameter(iap.Poisson(2))
        assert isinstance(observed, iap.Poisson)

    def test_tuple_of_floats(self):
        observed = iaa_size._handle_position_parameter((0.4, 0.6))
        assert isinstance(observed, tuple)
        assert len(observed) == 2
        assert isinstance(observed[0], iap.Deterministic)
        assert 0.4 - 1e-4 < observed[0].value < 0.4 + 1e-4
        assert isinstance(observed[1], iap.Deterministic)
        assert 0.6 - 1e-4 < observed[1].value < 0.6 + 1e-4

    def test_tuple_of_floats_outside_value_range_leads_to_failure(self):
        got_exception = False
        try:
            _ = iaa_size._handle_position_parameter((1.2, 0.6))
        except Exception as e:
            assert "must be within the value range" in str(e)
            got_exception = True
        assert got_exception

    def test_tuple_of_stochastic_parameters(self):
        observed = iaa_size._handle_position_parameter(
            (iap.Poisson(2), iap.Poisson(3)))
        assert isinstance(observed[0], iap.Poisson)
        assert isinstance(observed[0].lam, iap.Deterministic)
        assert 2 - 1e-4 < observed[0].lam.value < 2 + 1e-4
        assert isinstance(observed[1], iap.Poisson)
        assert isinstance(observed[1].lam, iap.Deterministic)
        assert 3 - 1e-4 < observed[1].lam.value < 3 + 1e-4

    def test_tuple_of_float_and_stochastic_paramter(self):
        observed = iaa_size._handle_position_parameter((0.4, iap.Poisson(3)))
        assert isinstance(observed, tuple)
        assert len(observed) == 2
        assert isinstance(observed[0], iap.Deterministic)
        assert 0.4 - 1e-4 < observed[0].value < 0.4 + 1e-4
        assert isinstance(observed[1], iap.Poisson)
        assert isinstance(observed[1].lam, iap.Deterministic)
        assert 3 - 1e-4 < observed[1].lam.value < 3 + 1e-4

    def test_bad_datatype_leads_to_failure(self):
        got_exception = False
        try:
            _ = iaa_size._handle_position_parameter(False)
        except Exception as e:
            assert "Expected one of the following as position parameter" in str(e)
            got_exception = True
        assert got_exception


class TestResize(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image2d(self):
        # 4x8
        base_img2d = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 255, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 255, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
        base_img2d = np.array(base_img2d, dtype=np.uint8)
        return base_img2d

    @property
    def image3d(self):
        base_img3d = np.tile(self.image2d[..., np.newaxis], (1, 1, 3))
        return base_img3d

    @property
    def kpsoi2d(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=4, y=1)]
        return ia.KeypointsOnImage(kps, shape=self.image2d.shape)

    @property
    def kpsoi3d(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=4, y=1)]
        return ia.KeypointsOnImage(kps, shape=self.image3d.shape)

    @property
    def psoi2d(self):
        polygons = [
            ia.Polygon([(0, 0), (8, 0), (8, 4)]),
            ia.Polygon([(1, 1), (7, 1), (7, 3), (1, 3)]),
        ]
        return ia.PolygonsOnImage(polygons, shape=self.image2d.shape)

    @property
    def psoi3d(self):
        polygons = [
            ia.Polygon([(0, 0), (8, 0), (8, 4)]),
            ia.Polygon([(1, 1), (7, 1), (7, 3), (1, 3)]),
        ]
        return ia.PolygonsOnImage(polygons, shape=self.image3d.shape)

    @property
    def lsoi2d(self):
        lss = [
            ia.LineString([(0, 0), (8, 0), (8, 4)]),
            ia.LineString([(1, 1), (7, 1), (7, 3), (1, 3)]),
        ]
        return ia.LineStringsOnImage(lss, shape=self.image2d.shape)

    @property
    def lsoi3d(self):
        lss = [
            ia.LineString([(0, 0), (8, 0), (8, 4)]),
            ia.LineString([(1, 1), (7, 1), (7, 3), (1, 3)]),
        ]
        return ia.LineStringsOnImage(lss, shape=self.image3d.shape)

    @property
    def bbsoi2d(self):
        bbs = [
            ia.BoundingBox(x1=0, y1=0, x2=8, y2=4),
            ia.BoundingBox(x1=1, y1=2, x2=6, y2=3),
        ]
        return ia.BoundingBoxesOnImage(bbs, shape=self.image2d.shape)

    @property
    def bbsoi3d(self):
        bbs = [
            ia.BoundingBox(x1=0, y1=0, x2=8, y2=4),
            ia.BoundingBox(x1=1, y1=2, x2=6, y2=3),
        ]
        return ia.BoundingBoxesOnImage(bbs, shape=self.image3d.shape)

    @classmethod
    def _aspect_ratio(cls, image):
        return image.shape[1] / image.shape[0]

    def test_resize_to_fixed_int(self):
        aug = iaa.Resize(12)
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        assert observed2d.shape == (12, 12)
        assert observed3d.shape == (12, 12, 3)
        assert 50 < np.average(observed2d) < 205
        assert 50 < np.average(observed3d) < 205

    def test_resize_to_fixed_float(self):
        aug = iaa.Resize(0.5)
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        assert observed2d.shape == (2, 4)
        assert observed3d.shape == (2, 4, 3)
        assert 50 < np.average(observed2d) < 205
        assert 50 < np.average(observed3d) < 205

    def test_heatmaps_with_width_int_and_height_int(self):
        aug = iaa.Resize({"height": 8, "width": 12})
        heatmaps_arr = (self.image2d / 255.0).astype(np.float32)
        heatmaps_aug = aug.augment_heatmaps([
            HeatmapsOnImage(heatmaps_arr, shape=self.image3d.shape)])[0]
        assert heatmaps_aug.shape == (8, 12, 3)
        assert 0 - 1e-6 < heatmaps_aug.min_value < 0 + 1e-6
        assert 1 - 1e-6 < heatmaps_aug.max_value < 1 + 1e-6
        assert np.average(heatmaps_aug.get_arr()[0, :]) < 0.05
        assert np.average(heatmaps_aug.get_arr()[-1, :]) < 0.05
        assert np.average(heatmaps_aug.get_arr()[:, 0]) < 0.05
        assert 0.8 < np.average(heatmaps_aug.get_arr()[2:6, 2:10]) < 1 + 1e-6

    def test_segmaps_with_width_int_and_height_int(self):
        for nb_channels in [None, 1, 10]:
            aug = iaa.Resize({"height": 8, "width": 12})
            segmaps_arr = (self.image2d > 0).astype(np.int32)
            if nb_channels is not None:
                segmaps_arr = np.tile(
                    segmaps_arr[..., np.newaxis], (1, 1, nb_channels))
            segmaps_aug = aug.augment_segmentation_maps([
                SegmentationMapsOnImage(segmaps_arr, shape=self.image3d.shape)])[0]
            assert segmaps_aug.shape == (8, 12, 3)
            assert segmaps_aug.arr.shape == (8, 12, nb_channels if nb_channels is not None else 1)
            assert np.all(segmaps_aug.arr[0, 1:-1, :] == 0)
            assert np.all(segmaps_aug.arr[-1, 1:-1, :] == 0)
            assert np.all(segmaps_aug.arr[1:-1, 0, :] == 0)
            assert np.all(segmaps_aug.arr[1:-1, -1, :] == 0)
            assert np.all(segmaps_aug.arr[2:-2, 2:-2, :] == 1)

    def test_heatmaps_with_diff_size_than_img_and_width_float_height_int(self):
        aug = iaa.Resize({"width": 2.0, "height": 16})
        heatmaps_arr = (self.image2d / 255.0).astype(np.float32)
        heatmaps = HeatmapsOnImage(
            heatmaps_arr,
            shape=(2*self.image3d.shape[0], 2*self.image3d.shape[1], 3))
        heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
        assert heatmaps_aug.shape == (16, int(self.image3d.shape[1]*2*2), 3)
        assert heatmaps_aug.arr_0to1.shape == (8, 16, 1)
        assert 0 - 1e-6 < heatmaps_aug.min_value < 0 + 1e-6
        assert 1 - 1e-6 < heatmaps_aug.max_value < 1 + 1e-6
        assert np.average(heatmaps_aug.get_arr()[0, :]) < 0.05
        assert np.average(heatmaps_aug.get_arr()[-1:, :]) < 0.05
        assert np.average(heatmaps_aug.get_arr()[:, 0]) < 0.05
        assert 0.8 < np.average(heatmaps_aug.get_arr()[2:6, 2:10]) < 1 + 1e-6

    def test_segmaps_with_diff_size_than_img_and_width_float_height_int(self):
        aug = iaa.Resize({"width": 2.0, "height": 16})
        segmaps_arr = (self.image2d > 0).astype(np.int32)
        segmaps = SegmentationMapsOnImage(
            segmaps_arr,
            shape=(2*self.image3d.shape[0], 2*self.image3d.shape[1], 3))
        segmaps_aug = aug.augment_segmentation_maps([segmaps])[0]
        assert segmaps_aug.shape == (16, int(self.image3d.shape[1]*2*2), 3)
        assert segmaps_aug.arr.shape == (8, 16, 1)
        assert np.all(segmaps_aug.arr[0, 1:-1, :] == 0)
        assert np.all(segmaps_aug.arr[-1, 1:-1, :] == 0)
        assert np.all(segmaps_aug.arr[1:-1, 0, :] == 0)
        assert np.all(segmaps_aug.arr[1:-1, -1, :] == 0)
        assert np.all(segmaps_aug.arr[2:-2, 2:-2, :] == 1)

    def test_keypoints_on_3d_img_and_with_width_int_and_height_int(self):
        aug = iaa.Resize({"height": 8, "width": 12})
        kpsoi_aug = aug.augment_keypoints([self.kpsoi3d])[0]
        assert len(kpsoi_aug.keypoints) == 2
        assert kpsoi_aug.shape == (8, 12, 3)
        assert np.allclose(kpsoi_aug.keypoints[0].x, 1.5)
        assert np.allclose(kpsoi_aug.keypoints[0].y, 4)
        assert np.allclose(kpsoi_aug.keypoints[1].x, 6)
        assert np.allclose(kpsoi_aug.keypoints[1].y, 2)

    def test_polygons_on_3d_img_and_with_width_int_and_height_int(self):
        aug = iaa.Resize({"width": 12, "height": 8})
        cbaoi_aug = aug.augment_polygons(self.psoi3d)
        assert len(cbaoi_aug.items) == 2
        assert cbaoi_aug.shape == (8, 12, 3)
        assert cbaoi_aug.items[0].coords_almost_equals(
            [(0, 0), (12, 0), (12, 8)]
        )
        assert cbaoi_aug.items[1].coords_almost_equals(
            [(1.5, 2), (10.5, 2), (10.5, 6), (1.5, 6)]
        )

    def test_line_strings_on_3d_img_and_with_width_int_and_height_int(self):
        aug = iaa.Resize({"width": 12, "height": 8})
        cbaoi_aug = aug.augment_line_strings(self.lsoi3d)
        assert len(cbaoi_aug.items) == 2
        assert cbaoi_aug.shape == (8, 12, 3)
        assert cbaoi_aug.items[0].coords_almost_equals(
            [(0, 0), (12, 0), (12, 8)]
        )
        assert cbaoi_aug.items[1].coords_almost_equals(
            [(1.5, 2), (10.5, 2), (10.5, 6), (1.5, 6)]
        )

    def test_bounding_boxes_on_3d_img_and_with_width_int_and_height_int(self):
        aug = iaa.Resize({"width": 12, "height": 8})
        bbsoi_aug = aug.augment_bounding_boxes(self.bbsoi3d)
        assert len(bbsoi_aug.bounding_boxes) == 2
        assert bbsoi_aug.shape == (8, 12, 3)
        assert bbsoi_aug.bounding_boxes[0].coords_almost_equals(
            [(0, 0), (12, 8)]
        )
        assert bbsoi_aug.bounding_boxes[1].coords_almost_equals(
            [((1/8)*12, (2/4)*8), ((6/8)*12, (3/4)*8)]
        )

    def test_keypoints_on_2d_img_and_with_width_float_and_height_int(self):
        aug = iaa.Resize({"width": 3.0, "height": 8})
        kpsoi_aug = aug.augment_keypoints([self.kpsoi2d])[0]
        assert len(kpsoi_aug.keypoints) == 2
        assert kpsoi_aug.shape == (8, 24)
        assert np.allclose(kpsoi_aug.keypoints[0].x, 3)
        assert np.allclose(kpsoi_aug.keypoints[0].y, 4)
        assert np.allclose(kpsoi_aug.keypoints[1].x, 12)
        assert np.allclose(kpsoi_aug.keypoints[1].y, 2)

    def test_polygons_on_2d_img_and_with_width_float_and_height_int(self):
        aug = iaa.Resize({"width": 3.0, "height": 8})
        cbaoi_aug = aug.augment_polygons(self.psoi2d)
        assert len(cbaoi_aug.items) == 2
        assert cbaoi_aug.shape == (8, 24)
        assert cbaoi_aug.items[0].coords_almost_equals(
            [(3*0, 0), (3*8, 0), (3*8, 8)]
        )
        assert cbaoi_aug.items[1].coords_almost_equals(
            [(3*1, 2), (3*7, 2), (3*7, 6), (3*1, 6)]
        )

    def test_line_strings_on_2d_img_and_with_width_float_and_height_int(self):
        aug = iaa.Resize({"width": 3.0, "height": 8})
        cbaoi_aug = aug.augment_line_strings(self.lsoi2d)
        assert len(cbaoi_aug.items) == 2
        assert cbaoi_aug.shape == (8, 24)
        assert cbaoi_aug.items[0].coords_almost_equals(
            [(3*0, 0), (3*8, 0), (3*8, 8)]
        )
        assert cbaoi_aug.items[1].coords_almost_equals(
            [(3*1, 2), (3*7, 2), (3*7, 6), (3*1, 6)]
        )

    def test_bounding_boxes_on_2d_img_and_with_width_float_and_height_int(self):
        aug = iaa.Resize({"width": 3.0, "height": 8})
        bbsoi_aug = aug.augment_bounding_boxes(self.bbsoi2d)
        assert len(bbsoi_aug.bounding_boxes) == 2
        assert bbsoi_aug.shape == (8, 24)
        assert bbsoi_aug.bounding_boxes[0].coords_almost_equals(
            [(3*0, 0), (3*8, 8)]
        )
        assert bbsoi_aug.bounding_boxes[1].coords_almost_equals(
            [(3*1, (2/4)*8), (3*6, (3/4)*8)]
        )

    def test_empty_keypoints(self):
        aug = iaa.Resize({"height": 8, "width": 12})
        kpsoi = ia.KeypointsOnImage([], shape=(4, 8, 3))
        kpsoi_aug = aug.augment_keypoints(kpsoi)
        assert len(kpsoi_aug.keypoints) == 0
        assert kpsoi_aug.shape == (8, 12, 3)

    def test_empty_polygons(self):
        aug = iaa.Resize({"height": 8, "width": 12})
        psoi = ia.PolygonsOnImage([], shape=(4, 8, 3))
        psoi_aug = aug.augment_polygons(psoi)
        assert len(psoi_aug.polygons) == 0
        assert psoi_aug.shape == (8, 12, 3)

    def test_empty_line_strings(self):
        aug = iaa.Resize({"height": 8, "width": 12})
        lsoi = ia.LineStringsOnImage([], shape=(4, 8, 3))
        lsoi_aug = aug.augment_line_strings(lsoi)
        assert len(lsoi_aug.items) == 0
        assert lsoi_aug.shape == (8, 12, 3)

    def test_empty_bounding_boxes(self):
        aug = iaa.Resize({"height": 8, "width": 12})
        bbsoi = ia.BoundingBoxesOnImage([], shape=(4, 8, 3))
        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)
        assert len(bbsoi_aug.bounding_boxes) == 0
        assert bbsoi_aug.shape == (8, 12, 3)

    def test_size_is_list_of_ints(self):
        aug = iaa.Resize([12, 14])
        seen2d = [False, False]
        seen3d = [False, False]
        for _ in sm.xrange(100):
            observed2d = aug.augment_image(self.image2d)
            observed3d = aug.augment_image(self.image3d)
            assert observed2d.shape in [(12, 12), (14, 14)]
            assert observed3d.shape in [(12, 12, 3), (14, 14, 3)]
            if observed2d.shape == (12, 12):
                seen2d[0] = True
            else:
                seen2d[1] = True
            if observed3d.shape == (12, 12, 3):
                seen3d[0] = True
            else:
                seen3d[1] = True
            if all(seen2d) and all(seen3d):
                break
        assert np.all(seen2d)
        assert np.all(seen3d)

    def test_size_is_tuple_of_ints(self):
        aug = iaa.Resize((12, 14))
        seen2d = [False, False, False]
        seen3d = [False, False, False]
        for _ in sm.xrange(100):
            observed2d = aug.augment_image(self.image2d)
            observed3d = aug.augment_image(self.image3d)
            assert observed2d.shape in [(12, 12), (13, 13), (14, 14)]
            assert observed3d.shape in [(12, 12, 3), (13, 13, 3), (14, 14, 3)]
            if observed2d.shape == (12, 12):
                seen2d[0] = True
            elif observed2d.shape == (13, 13):
                seen2d[1] = True
            else:
                seen2d[2] = True
            if observed3d.shape == (12, 12, 3):
                seen3d[0] = True
            elif observed3d.shape == (13, 13, 3):
                seen3d[1] = True
            else:
                seen3d[2] = True
            if all(seen2d) and all(seen3d):
                break
        assert np.all(seen2d)
        assert np.all(seen3d)

    def test_size_is_string_keep(self):
        aug = iaa.Resize("keep")
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        assert observed2d.shape == self.image2d.shape
        assert observed3d.shape == self.image3d.shape

    # TODO shouldn't this be more an error?
    def test_size_is_empty_list(self):
        aug = iaa.Resize([])
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        assert observed2d.shape == self.image2d.shape
        assert observed3d.shape == self.image3d.shape

    def test_size_is_empty_dict(self):
        aug = iaa.Resize({})
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        assert observed2d.shape == self.image2d.shape
        assert observed3d.shape == self.image3d.shape

    def test_change_height_to_fixed_int(self):
        aug = iaa.Resize({"height": 11})
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        assert observed2d.shape == (11, self.image2d.shape[1])
        assert observed3d.shape == (11, self.image3d.shape[1], 3)

    def test_change_width_to_fixed_int(self):
        aug = iaa.Resize({"width": 13})
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        assert observed2d.shape == (self.image2d.shape[0], 13)
        assert observed3d.shape == (self.image3d.shape[0], 13, 3)

    def test_change_height_and_width_to_fixed_ints(self):
        aug = iaa.Resize({"height": 12, "width": 13})
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        assert observed2d.shape == (12, 13)
        assert observed3d.shape == (12, 13, 3)

    def test_change_height_to_fixed_int_but_dont_change_width(self):
        aug = iaa.Resize({"height": 12, "width": "keep"})
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        assert observed2d.shape == (12, self.image2d.shape[1])
        assert observed3d.shape == (12, self.image3d.shape[1], 3)

    def test_dont_change_height_but_width_to_fixed_int(self):
        aug = iaa.Resize({"height": "keep", "width": 12})
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        assert observed2d.shape == (self.image2d.shape[0], 12)
        assert observed3d.shape == (self.image3d.shape[0], 12, 3)

    def test_change_height_to_fixed_int_width_keeps_aspect_ratio(self):
        aug = iaa.Resize({"height": 12, "width": "keep-aspect-ratio"})
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        aspect_ratio2d = self._aspect_ratio(self.image2d)
        aspect_ratio3d = self._aspect_ratio(self.image3d)
        assert observed2d.shape == (12, int(12 * aspect_ratio2d))
        assert observed3d.shape == (12, int(12 * aspect_ratio3d), 3)

    def test_height_keeps_aspect_ratio_width_changed_to_fixed_int(self):
        aug = iaa.Resize({"height": "keep-aspect-ratio", "width": 12})
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        aspect_ratio2d = self._aspect_ratio(self.image2d)
        aspect_ratio3d = self._aspect_ratio(self.image3d)
        assert observed2d.shape == (int(12 * (1/aspect_ratio2d)), 12)
        assert observed3d.shape == (int(12 * (1/aspect_ratio3d)), 12, 3)

    # TODO add test for shorter side being tuple, list, stochastic parameter
    def test_change_shorter_side_by_fixed_int_longer_keeps_aspect_ratio(self):
        aug = iaa.Resize({"shorter-side": 6,
                          "longer-side": "keep-aspect-ratio"})
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        aspect_ratio2d = self._aspect_ratio(self.image2d)
        aspect_ratio3d = self._aspect_ratio(self.image3d)
        assert observed2d.shape == (6, int(6 * aspect_ratio2d))
        assert observed3d.shape == (6, int(6 * aspect_ratio3d), 3)

    # TODO add test for longer side being tuple, list, stochastic parameter
    def test_change_longer_side_by_fixed_int_shorter_keeps_aspect_ratio(self):
        aug = iaa.Resize({"longer-side": 6,
                          "shorter-side": "keep-aspect-ratio"})
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)
        aspect_ratio2d = self._aspect_ratio(self.image2d)
        aspect_ratio3d = self._aspect_ratio(self.image3d)
        assert observed2d.shape == (int(6 * (1/aspect_ratio2d)), 6)
        assert observed3d.shape == (int(6 * (1/aspect_ratio3d)), 6, 3)

    def test_change_height_by_list_of_ints_width_by_fixed_int(self):
        aug = iaa.Resize({"height": [12, 14], "width": 12})
        seen2d = [False, False]
        seen3d = [False, False]
        for _ in sm.xrange(100):
            observed2d = aug.augment_image(self.image2d)
            observed3d = aug.augment_image(self.image3d)
            assert observed2d.shape in [(12, 12), (14, 12)]
            assert observed3d.shape in [(12, 12, 3), (14, 12, 3)]
            if observed2d.shape == (12, 12):
                seen2d[0] = True
            else:
                seen2d[1] = True
            if observed3d.shape == (12, 12, 3):
                seen3d[0] = True
            else:
                seen3d[1] = True
            if np.all(seen2d) and np.all(seen3d):
                break
        assert np.all(seen2d)
        assert np.all(seen3d)

    def test_change_height_by_fixed_int_width_by_list_of_ints(self):
        aug = iaa.Resize({"height": 12, "width": [12, 14]})
        seen2d = [False, False]
        seen3d = [False, False]
        for _ in sm.xrange(100):
            observed2d = aug.augment_image(self.image2d)
            observed3d = aug.augment_image(self.image3d)
            assert observed2d.shape in [(12, 12), (12, 14)]
            assert observed3d.shape in [(12, 12, 3), (12, 14, 3)]
            if observed2d.shape == (12, 12):
                seen2d[0] = True
            else:
                seen2d[1] = True
            if observed3d.shape == (12, 12, 3):
                seen3d[0] = True
            else:
                seen3d[1] = True
            if np.all(seen2d) and np.all(seen3d):
                break
        assert np.all(seen2d)
        assert np.all(seen3d)

    def test_change_height_by_fixed_int_width_by_stochastic_parameter(self):
        aug = iaa.Resize({"height": 12, "width": iap.Choice([12, 14])})
        seen2d = [False, False]
        seen3d = [False, False]
        for _ in sm.xrange(100):
            observed2d = aug.augment_image(self.image2d)
            observed3d = aug.augment_image(self.image3d)
            assert observed2d.shape in [(12, 12), (12, 14)]
            assert observed3d.shape in [(12, 12, 3), (12, 14, 3)]
            if observed2d.shape == (12, 12):
                seen2d[0] = True
            else:
                seen2d[1] = True
            if observed3d.shape == (12, 12, 3):
                seen3d[0] = True
            else:
                seen3d[1] = True
            if np.all(seen2d) and np.all(seen3d):
                break
        assert np.all(seen2d)
        assert np.all(seen3d)

    def test_change_height_by_tuple_of_ints_width_by_fixed_int(self):
        aug = iaa.Resize({"height": (12, 14), "width": 12})
        seen2d = [False, False, False]
        seen3d = [False, False, False]
        for _ in sm.xrange(100):
            observed2d = aug.augment_image(self.image2d)
            observed3d = aug.augment_image(self.image3d)
            assert observed2d.shape in [(12, 12), (13, 12), (14, 12)]
            assert observed3d.shape in [(12, 12, 3), (13, 12, 3), (14, 12, 3)]
            if observed2d.shape == (12, 12):
                seen2d[0] = True
            elif observed2d.shape == (13, 12):
                seen2d[1] = True
            else:
                seen2d[2] = True
            if observed3d.shape == (12, 12, 3):
                seen3d[0] = True
            elif observed3d.shape == (13, 12, 3):
                seen3d[1] = True
            else:
                seen3d[2] = True
            if np.all(seen2d) and np.all(seen3d):
                break
        assert np.all(seen2d)
        assert np.all(seen3d)

    def test_size_is_float(self):
        aug = iaa.Resize(2.0)
        observed2d = aug.augment_image(self.image2d)
        observed3d = aug.augment_image(self.image3d)

        intensity_avg = np.average(self.image2d)
        intensity_low = intensity_avg - 0.2 * np.abs(intensity_avg - 128)
        intensity_high = intensity_avg + 0.2 * np.abs(intensity_avg - 128)
        assert observed2d.shape == (self.image2d.shape[0]*2,
                                    self.image2d.shape[1]*2)
        assert observed3d.shape == (self.image3d.shape[0]*2,
                                    self.image3d.shape[1]*2,
                                    3)
        assert intensity_low < np.average(observed2d) < intensity_high
        assert intensity_low < np.average(observed3d) < intensity_high

    def test_size_is_list(self):
        aug = iaa.Resize([2.0, 4.0])
        seen2d = [False, False]
        seen3d = [False, False]
        expected_shapes_2d = [
            (self.image2d.shape[0]*2, self.image2d.shape[1]*2),
            (self.image2d.shape[0]*4, self.image2d.shape[1]*4)]
        expected_shapes_3d = [
            (self.image3d.shape[0]*2, self.image3d.shape[1]*2, 3),
            (self.image3d.shape[0]*4, self.image3d.shape[1]*4, 3)]
        for _ in sm.xrange(100):
            observed2d = aug.augment_image(self.image2d)
            observed3d = aug.augment_image(self.image3d)
            assert observed2d.shape in expected_shapes_2d
            assert observed3d.shape in expected_shapes_3d
            if observed2d.shape == expected_shapes_2d[0]:
                seen2d[0] = True
            else:
                seen2d[1] = True
            if observed3d.shape == expected_shapes_3d[0]:
                seen3d[0] = True
            else:
                seen3d[1] = True
            if np.all(seen2d) and np.all(seen3d):
                break
        assert np.all(seen2d)
        assert np.all(seen3d)

    def test_size_is_stochastic_parameter(self):
        aug = iaa.Resize(iap.Choice([2.0, 4.0]))
        seen2d = [False, False]
        seen3d = [False, False]
        expected_shapes_2d = [
            (self.image2d.shape[0]*2, self.image2d.shape[1]*2),
            (self.image2d.shape[0]*4, self.image2d.shape[1]*4)]
        expected_shapes_3d = [
            (self.image3d.shape[0]*2, self.image3d.shape[1]*2, 3),
            (self.image3d.shape[0]*4, self.image3d.shape[1]*4, 3)]
        for _ in sm.xrange(100):
            observed2d = aug.augment_image(self.image2d)
            observed3d = aug.augment_image(self.image3d)
            assert observed2d.shape in expected_shapes_2d
            assert observed3d.shape in expected_shapes_3d
            if observed2d.shape == expected_shapes_2d[0]:
                seen2d[0] = True
            else:
                seen2d[1] = True
            if observed3d.shape == expected_shapes_3d[0]:
                seen3d[0] = True
            else:
                seen3d[1] = True
            if all(seen2d) and all(seen3d):
                break
        assert np.all(seen2d)
        assert np.all(seen3d)

    def test_decrease_size_by_tuple_of_floats__one_for_both_sides(self):
        image2d = self.image2d[0:4, 0:4]
        image3d = self.image3d[0:4, 0:4, :]
        aug = iaa.Resize((0.76, 1.0))
        not_seen2d = set()
        not_seen3d = set()
        for size in sm.xrange(3, 4+1):
            not_seen2d.add((size, size))
        for size in sm.xrange(3, 4+1):
            not_seen3d.add((size, size, 3))
        possible2d = set(list(not_seen2d))
        possible3d = set(list(not_seen3d))
        for _ in sm.xrange(100):
            observed2d = aug.augment_image(image2d)
            observed3d = aug.augment_image(image3d)
            assert observed2d.shape in possible2d
            assert observed3d.shape in possible3d
            if observed2d.shape in not_seen2d:
                not_seen2d.remove(observed2d.shape)
            if observed3d.shape in not_seen3d:
                not_seen3d.remove(observed3d.shape)
            if not not_seen2d and not not_seen3d:
                break
        assert not not_seen2d
        assert not not_seen3d

    def test_decrease_size_by_tuples_of_floats__one_per_side(self):
        image2d = self.image2d[0:4, 0:4]
        image3d = self.image3d[0:4, 0:4, :]
        aug = iaa.Resize({"height": (0.76, 1.0), "width": (0.76, 1.0)})
        not_seen2d = set()
        not_seen3d = set()
        for hsize in sm.xrange(3, 4+1):
            for wsize in sm.xrange(3, 4+1):
                not_seen2d.add((hsize, wsize))
        for hsize in sm.xrange(3, 4+1):
            for wsize in sm.xrange(3, 4+1):
                not_seen3d.add((hsize, wsize, 3))
        possible2d = set(list(not_seen2d))
        possible3d = set(list(not_seen3d))
        for _ in sm.xrange(100):
            observed2d = aug.augment_image(image2d)
            observed3d = aug.augment_image(image3d)
            assert observed2d.shape in possible2d
            assert observed3d.shape in possible3d
            if observed2d.shape in not_seen2d:
                not_seen2d.remove(observed2d.shape)
            if observed3d.shape in not_seen3d:
                not_seen3d.remove(observed3d.shape)
            if not not_seen2d and not not_seen3d:
                break
        assert not not_seen2d
        assert not not_seen3d

    def test_bad_datatype_for_size_leads_to_failure(self):
        got_exception = False
        try:
            aug = iaa.Resize("foo")
            _ = aug.augment_image(self.image2d)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_get_parameters(self):
        aug = iaa.Resize(size=1, interpolation="nearest")
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert params[0].value == 1
        assert params[1].value == "nearest"

    def test_dtypes_roughly(self):
        # most of the dtype testing is done for imresize_many_images()
        # so we focus here on a rough test that merely checks if the dtype
        # does not change

        # these dtypes should be kept in sync with imresize_many_images()
        dtypes = [
            "uint8",
            "uint16",
            "int8",
            "int16",
            "float16",
            "float32",
            "float64",
            "bool"
        ]

        for dt in dtypes:
            for ip in ["nearest", "cubic"]:
                aug = iaa.Resize({"height": 10, "width": 20}, interpolation=ip)
                for is_list in [False, True]:
                    with self.subTest(dtype=dt, interpolation=ip,
                                      is_list=is_list):
                        image = np.full((9, 19, 3), 1, dtype=dt)
                        images = [image, image]
                        if not is_list:
                            images = np.array(images, dtype=dt)

                        images_aug = aug(images=images)

                        if is_list:
                            assert isinstance(images_aug, list)
                        else:
                            assert ia.is_np_array(images_aug)

                        assert len(images_aug) == 2
                        for image_aug in images_aug:
                            assert image_aug.dtype.name == dt
                            assert image_aug.shape == (10, 20, 3)
                            assert np.all(image_aug >= 1 - 1e-4)


class TestPad(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]], dtype=np.uint8)
        return base_img[:, :, np.newaxis]

    @property
    def images(self):
        return np.array([self.image])

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        return ia.KeypointsOnImage(kps, shape=self.image.shape)

    @property
    def psoi(self):
        polys = [ia.Polygon([(1, 1), (2, 1), (2, 2)])]
        return ia.PolygonsOnImage(polys, shape=self.image.shape)

    @property
    def lsoi(self):
        ls = [ia.LineString([(1, 1), (2, 1), (2, 2)])]
        return ia.LineStringsOnImage(ls, shape=self.image.shape)

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)]
        return ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)

    @property
    def heatmap(self):
        heatmaps_arr = np.float32([[0, 0, 0],
                                   [0, 1.0, 0],
                                   [0, 0, 0]])
        return ia.HeatmapsOnImage(heatmaps_arr, shape=self.image.shape)

    @property
    def segmap(self):
        segmaps_arr = np.int32([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])
        return ia.SegmentationMapsOnImage(segmaps_arr, shape=self.image.shape)

    def test___init___pad_mode_is_all(self):
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode=ia.ALL,
                      pad_cval=0,
                      keep_size=False)
        expected = ["constant", "edge", "linear_ramp", "maximum", "mean",
                    "median", "minimum", "reflect", "symmetric", "wrap"]
        assert isinstance(aug.pad_mode, iap.Choice)
        assert len(aug.pad_mode.a) == len(expected)
        assert np.all([v in aug.pad_mode.a for v in expected])

    def test___init___pad_mode_is_list(self):
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode=["constant", "maximum"],
                      pad_cval=0,
                      keep_size=False)
        expected = ["constant", "maximum"]
        assert isinstance(aug.pad_mode, iap.Choice)
        assert len(aug.pad_mode.a) == len(expected)
        assert np.all([v in aug.pad_mode.a for v in expected])

    def test___init___pad_cval_is_list(self):
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode="constant",
                      pad_cval=[50, 100],
                      keep_size=False)
        expected = [50, 100]
        assert isinstance(aug.pad_cval, iap.Choice)
        assert len(aug.pad_cval.a) == len(expected)
        assert np.all([v in aug.pad_cval.a for v in expected])

    def test_pad_images_by_1px_each_side_on_its_own(self):
        # test pad by 1 pixel on each side
        pads = [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ]
        for pad in pads:
            with self.subTest(px=pad):
                aug = iaa.Pad(px=pad, keep_size=False)

                top, right, bottom, left = pad

                base_img_padded = np.pad(
                    self.image,
                    ((top, bottom), (left, right), (0, 0)),
                    mode="constant",
                    constant_values=0)
                observed = aug.augment_images(self.images)
                assert np.array_equal(observed, np.array([base_img_padded]))

                observed = aug.augment_images([self.image])
                assert array_equal_lists(observed, [base_img_padded])

    def _test_pad_cbaoi_by_1px_each_side_on_its_own(self, cbaoi, augf_name):
        pads = [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ]
        for pad in pads:
            with self.subTest(px=pad):
                aug = iaa.Pad(px=pad, keep_size=False)

                top, right, bottom, left = pad

                image_padded_shape = list(self.image.shape)
                image_padded_shape[0] += top + bottom
                image_padded_shape[1] += left + right

                observed = getattr(aug, augf_name)(cbaoi)

                expected = shift_cbaoi(cbaoi, left=left, top=top)
                expected.shape = tuple(image_padded_shape)
                assert_cbaois_equal(observed, expected)

    def test_pad_keypoints_by_1px_each_side_on_its_own(self):
        self._test_pad_cbaoi_by_1px_each_side_on_its_own(
            self.kpsoi, "augment_keypoints")

    def test_pad_polygons_by_1px_each_side_on_its_own(self):
        self._test_pad_cbaoi_by_1px_each_side_on_its_own(
            self.psoi, "augment_polygons")

    def test_pad_line_strings_by_1px_each_side_on_its_own(self):
        self._test_pad_cbaoi_by_1px_each_side_on_its_own(
            self.lsoi, "augment_line_strings")

    def test_pad_bounding_boxes_by_1px_each_side_on_its_own(self):
        self._test_pad_cbaoi_by_1px_each_side_on_its_own(
            self.bbsoi, "augment_bounding_boxes")

    def test_pad_heatmaps_by_1px_each_side_on_its_own(self):
        pads = [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ]
        for pad in pads:
            with self.subTest(px=pad):
                aug = iaa.Pad(px=pad, keep_size=False)

                top, right, bottom, left = pad

                heatmaps_arr = self.heatmap.get_arr()
                heatmaps_arr_padded = np.pad(
                    heatmaps_arr,
                    ((top, bottom), (left, right)),
                    mode="constant",
                    constant_values=0)
                heatmaps = [ia.HeatmapsOnImage(
                    heatmaps_arr, shape=self.image.shape)]
                image_padded_shape = list(self.image.shape)
                image_padded_shape[0] += top + bottom
                image_padded_shape[1] += left + right

                observed = aug.augment_heatmaps(heatmaps)[0]

                assert observed.shape == tuple(image_padded_shape)
                assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
                assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
                assert np.array_equal(observed.get_arr(), heatmaps_arr_padded)

    def test_pad_segmaps_by_1px_each_side_on_its_own(self):
        pads = [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ]
        for pad in pads:
            with self.subTest(px=pad):
                aug = iaa.Pad(px=pad, keep_size=False)

                top, right, bottom, left = pad

                segmaps_arr = self.segmap.get_arr()
                segmaps_arr_padded = np.pad(
                    segmaps_arr,
                    ((top, bottom), (left, right)),
                    mode="constant",
                    constant_values=0)
                segmaps = [SegmentationMapsOnImage(
                    segmaps_arr, shape=self.image.shape)]
                image_padded_shape = list(self.image.shape)
                image_padded_shape[0] += top + bottom
                image_padded_shape[1] += left + right

                observed = aug.augment_segmentation_maps(segmaps)[0]

                assert observed.shape == tuple(image_padded_shape)
                assert np.array_equal(observed.get_arr(), segmaps_arr_padded)

    # TODO split up, add similar tests for polygons/LS/BBs
    def test_pad_each_side_on_its_own_by_tuple_of_ints(self):
        def _to_range_tuple(val):
            return val if isinstance(val, tuple) else (val, val)

        pads = [
            ((0, 2), 0, 0, 0),
            (0, (0, 2), 0, 0),
            (0, 0, (0, 2), 0),
            (0, 0, 0, (0, 2)),
        ]
        for pad in pads:
            with self.subTest(px=pad):
                aug = iaa.Pad(px=pad, keep_size=False)
                aug_det = aug.to_deterministic()

                top, right, bottom, left = pad

                images_padded = []
                keypoints_padded = []
                top_range = _to_range_tuple(top)
                right_range = _to_range_tuple(right)
                bottom_range = _to_range_tuple(bottom)
                left_range = _to_range_tuple(left)

                top_values = sm.xrange(top_range[0], top_range[1]+1)
                right_values = sm.xrange(right_range[0], right_range[1]+1)
                bottom_values = sm.xrange(bottom_range[0], bottom_range[1]+1)
                left_values = sm.xrange(left_range[0], left_range[1]+1)

                for top_val in top_values:
                    for right_val in right_values:
                        for bottom_val in bottom_values:
                            for left_val in left_values:
                                images_padded.append(
                                    np.pad(
                                        self.image,
                                        ((top_val, bottom_val),
                                         (left_val, right_val),
                                         (0, 0)),
                                        mode="constant",
                                        constant_values=0
                                    )
                                )
                                keypoints_padded.append(
                                    self.kpsoi.shift(x=left_val, y=top_val))

                movements = []
                movements_det = []
                for i in sm.xrange(100):
                    observed = aug.augment_images(self.images)

                    matches = [
                        (1 if np.array_equal(observed,
                                             np.array([base_img_padded]))
                         else 0)
                        for base_img_padded
                        in images_padded
                    ]
                    movements.append(np.argmax(np.array(matches)))
                    assert any([val == 1 for val in matches])

                    observed = aug_det.augment_images(self.images)
                    matches = [
                        (1 if np.array_equal(observed,
                                             np.array([base_img_padded]))
                         else 0)
                        for base_img_padded
                        in images_padded
                    ]
                    movements_det.append(np.argmax(np.array(matches)))
                    assert any([val == 1 for val in matches])

                    observed = aug.augment_images([self.image])
                    assert any([
                        array_equal_lists(observed, [base_img_padded])
                        for base_img_padded
                        in images_padded])

                    observed = aug.augment_keypoints(self.kpsoi)
                    assert any([
                        keypoints_equal(observed, kp)
                        for kp
                        in keypoints_padded])

                assert len(set(movements)) == 3
                assert len(set(movements_det)) == 1

    # TODO split up, add similar tests for polygons/LS/BBs
    def test_pad_each_side_on_its_own_by_list_of_ints(self):
        # test pad by list of exact pixel values
        pads = [
            ([0, 2], 0, 0, 0),
            (0, [0, 2], 0, 0),
            (0, 0, [0, 2], 0),
            (0, 0, 0, [0, 2]),
        ]
        for pad in pads:
            top, right, bottom, left = pad
            aug = iaa.Pad(px=pad, keep_size=False)
            aug_det = aug.to_deterministic()

            images_padded = []
            keypoints_padded = []
            top_range = top if isinstance(top, list) else [top]
            right_range = right if isinstance(right, list) else [right]
            bottom_range = bottom if isinstance(bottom, list) else [bottom]
            left_range = left if isinstance(left, list) else [left]

            for top_val in top_range:
                for right_val in right_range:
                    for bottom_val in bottom_range:
                        for left_val in left_range:
                            images_padded.append(
                                np.pad(
                                    self.image,
                                    ((top_val, bottom_val),
                                     (left_val, right_val),
                                     (0, 0)),
                                    mode="constant",
                                    constant_values=0
                                )
                            )
                            keypoints_padded.append(
                                self.kpsoi.shift(x=left_val, y=top_val))

            movements = []
            movements_det = []
            for i in sm.xrange(100):
                observed = aug.augment_images(self.images)
                matches = [
                    (1 if np.array_equal(observed,
                                         np.array([base_img_padded]))
                     else 0)
                    for base_img_padded
                    in images_padded]
                movements.append(np.argmax(np.array(matches)))
                assert any([val == 1 for val in matches])

                observed = aug_det.augment_images(self.images)
                matches = [
                    (1 if np.array_equal(observed,
                                         np.array([base_img_padded]))
                     else 0)
                    for base_img_padded
                    in images_padded]
                movements_det.append(np.argmax(np.array(matches)))
                assert any([val == 1 for val in matches])

                observed = aug.augment_images([self.image])
                assert any([
                    array_equal_lists(observed, [base_img_padded])
                    for base_img_padded
                    in images_padded])

                observed = aug.augment_keypoints(self.kpsoi)
                assert any([
                    keypoints_equal(observed, kp)
                    for kp
                    in keypoints_padded])

            assert len(set(movements)) == 2
            assert len(set(movements_det)) == 1

    def test_pad_heatmaps_smaller_than_img_by_tuple_of_ints_without_ks(self):
        # pad smaller heatmaps
        # heatmap is (6, 4), image is (6, 16)
        # image is padded by (2, 4, 2, 4)
        # expected image size: (10, 24)
        # expected heatmap size: (10, 6)
        aug = iaa.Pad(px=(2, 4, 2, 4), keep_size=False)
        heatmaps_arr_small = np.float32([
            [0, 0, 0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 0, 0, 0]
        ])
        top, bottom, left, right = 2, 2, 1, 1
        heatmaps_arr_small_padded = np.pad(
            heatmaps_arr_small,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0)
        heatmaps = [ia.HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))]
        observed = aug.augment_heatmaps(heatmaps)[0]

        assert observed.shape == (10, 24)
        assert observed.arr_0to1.shape == (10, 6, 1)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.arr_0to1[..., 0], heatmaps_arr_small_padded)

    def test_pad_segmaps_smaller_than_img_by_tuple_of_ints_without_ks(self):
        # pad smaller segmaps
        # same sizes and paddings as above
        aug = iaa.Pad(px=(2, 4, 2, 4), keep_size=False)
        segmaps_arr_small = np.int32([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        segmaps = [SegmentationMapsOnImage(segmaps_arr_small, shape=(6, 16))]
        top, bottom, left, right = 2, 2, 1, 1
        segmaps_arr_small_padded = np.pad(
            segmaps_arr_small,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0)

        observed = aug.augment_segmentation_maps(segmaps)[0]

        assert observed.shape == (10, 24)
        assert observed.arr.shape == (10, 6, 1)
        assert np.array_equal(observed.arr[..., 0], segmaps_arr_small_padded)

    def test_pad_heatmaps_smaller_than_img_by_tuple_of_ints_with_ks(self):
        # pad smaller heatmaps, with keep_size=True
        # heatmap is (6, 4), image is (6, 16)
        # image is padded by (2, 4, 2, 4)
        # expected image size: (10, 24) -> (6, 16) after resize
        # expected heatmap size: (10, 6) -> (6, 4) after resize
        aug = iaa.Pad(px=(2, 4, 2, 4), keep_size=True)
        heatmaps_arr_small = np.float32([
            [0, 0, 0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 0, 0, 0]
        ])
        top, bottom, left, right = 2, 2, 1, 1
        heatmaps_arr_small_padded = np.pad(
            heatmaps_arr_small,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0)
        heatmaps = [ia.HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))]

        observed = aug.augment_heatmaps(heatmaps)[0]

        assert observed.shape == (6, 16)
        assert observed.arr_0to1.shape == (6, 4, 1)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(
            observed.arr_0to1[..., 0],
            np.clip(
                ia.imresize_single_image(
                    heatmaps_arr_small_padded,
                    (6, 4),
                    interpolation="cubic"),
                0, 1.0
            )
        )

    def test_pad_segmaps_smaller_than_img_by_tuple_of_ints_with_keep_size(self):
        # pad smaller segmaps, with keep_size=True
        # same sizes and paddings as above
        aug = iaa.Pad(px=(2, 4, 2, 4), keep_size=True)
        segmaps_arr_small = np.int32([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        top, bottom, left, right = 2, 2, 1, 1
        segmaps_arr_small_padded = np.pad(
            segmaps_arr_small,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0)
        segmaps = [SegmentationMapsOnImage(segmaps_arr_small, shape=(6, 16))]

        observed = aug.augment_segmentation_maps(segmaps)[0]

        assert observed.shape == (6, 16)
        assert observed.arr.shape == (6, 4, 1)
        assert np.array_equal(
            observed.arr[..., 0],
            ia.imresize_single_image(
                segmaps_arr_small_padded,
                (6, 4),
                interpolation="nearest"
            ),
        )

    def test_pad_keypoints_by_tuple_of_fixed_ints_without_keep_size(self):
        aug = iaa.Pad((2, 0, 4, 4), keep_size=False)
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=0)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(4, 4, 3))
        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
        assert kpsoi_aug.shape == (10, 8, 3)
        assert len(kpsoi_aug.keypoints) == 2
        assert np.allclose(kpsoi_aug.keypoints[0].x, 4+1)
        assert np.allclose(kpsoi_aug.keypoints[0].y, 2+2)
        assert np.allclose(kpsoi_aug.keypoints[1].x, 4+3)
        assert np.allclose(kpsoi_aug.keypoints[1].y, 2+0)

    def test_pad_keypoints_by_tuple_of_fixed_ints_with_keep_size(self):
        aug = iaa.Pad((2, 0, 4, 4), keep_size=True)
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=0)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(4, 4, 3))
        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
        assert kpsoi_aug.shape == (4, 4, 3)
        assert len(kpsoi_aug.keypoints) == 2
        assert np.allclose(kpsoi_aug.keypoints[0].x, ((4+1)/8)*4)
        assert np.allclose(kpsoi_aug.keypoints[0].y, ((2+2)/10)*4)
        assert np.allclose(kpsoi_aug.keypoints[1].x, ((4+3)/8)*4)
        assert np.allclose(kpsoi_aug.keypoints[1].y, ((2+0)/10)*4)

    def test_pad_polygons_by_tuple_of_fixed_ints_without_keep_size(self):
        aug = iaa.Pad((2, 0, 4, 4), keep_size=False)
        polygons = [ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
                    ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])]
        psoi = ia.PolygonsOnImage(polygons, shape=(4, 4, 3))
        psoi_aug = aug.augment_polygons([psoi, psoi])
        assert len(psoi_aug) == 2
        for psoi_aug_i in psoi_aug:
            assert psoi_aug_i.shape == (10, 8, 3)
            assert len(psoi_aug_i.items) == 2
            assert psoi_aug_i.items[0].coords_almost_equals(
                [(4, 2), (8, 2), (8, 6), (4, 6)])
            assert psoi_aug_i.items[1].coords_almost_equals(
                [(5, 3), (9, 3), (9, 7), (5, 7)])

    def test_pad_polygons_by_tuple_of_fixed_ints_with_keep_size(self):
        aug = iaa.Pad((2, 0, 4, 4), keep_size=True)
        polygons = [ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
                    ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])]
        psoi = ia.PolygonsOnImage(polygons, shape=(4, 4, 3))
        psoi_aug = aug.augment_polygons([psoi, psoi])
        assert len(psoi_aug) == 2
        for psoi_aug_i in psoi_aug:
            assert psoi_aug_i.shape == (4, 4, 3)
            assert len(psoi_aug_i.items) == 2
            assert psoi_aug_i.items[0].coords_almost_equals(
                [(4*(4/8), 4*(2/10)),
                 (4*(8/8), 4*(2/10)),
                 (4*(8/8), 4*(6/10)),
                 (4*(4/8), 4*(6/10))]
            )
            assert psoi_aug_i.items[1].coords_almost_equals(
                [(4*(5/8), 4*(3/10)),
                 (4*(9/8), 4*(3/10)),
                 (4*(9/8), 4*(7/10)),
                 (4*(5/8), 4*(7/10))]
            )

    def test_pad_line_strings_by_tuple_of_fixed_ints_without_keep_size(self):
        aug = iaa.Pad((2, 0, 4, 4), keep_size=False)
        lss = [ia.LineString([(0, 0), (4, 0), (4, 4), (0, 4)]),
               ia.LineString([(1, 1), (5, 1), (5, 5), (1, 5)])]
        cbaoi = ia.LineStringsOnImage(lss, shape=(4, 4, 3))
        cbaoi_aug = aug.augment_line_strings([cbaoi, cbaoi])
        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (10, 8, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(4, 2), (8, 2), (8, 6), (4, 6)])
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(5, 3), (9, 3), (9, 7), (5, 7)])

    def test_pad_line_strings_by_tuple_of_fixed_ints_with_keep_size(self):
        aug = iaa.Pad((2, 0, 4, 4), keep_size=True)
        lss = [ia.LineString([(0, 0), (4, 0), (4, 4), (0, 4)]),
               ia.LineString([(1, 1), (5, 1), (5, 5), (1, 5)])]
        cbaoi = ia.LineStringsOnImage(lss, shape=(4, 4, 3))
        cbaoi_aug = aug.augment_line_strings([cbaoi, cbaoi])
        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (4, 4, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(4*(4/8), 4*(2/10)),
                 (4*(8/8), 4*(2/10)),
                 (4*(8/8), 4*(6/10)),
                 (4*(4/8), 4*(6/10))]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(4*(5/8), 4*(3/10)),
                 (4*(9/8), 4*(3/10)),
                 (4*(9/8), 4*(7/10)),
                 (4*(5/8), 4*(7/10))]
            )

    def test_pad_bounding_boxes_by_tuple_of_fixed_ints_without_keep_size(self):
        aug = iaa.Pad((2, 0, 4, 4), keep_size=False)
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=4, y2=4),
               ia.BoundingBox(x1=1, y1=1, x2=3, y2=4)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(4, 4, 3))
        bbsoi_aug = aug.augment_bounding_boxes([bbsoi, bbsoi])
        assert len(bbsoi_aug) == 2
        for bbsoi_aug_i in bbsoi_aug:
            assert bbsoi_aug_i.shape == (10, 8, 3)
            assert len(bbsoi_aug_i.bounding_boxes) == 2
            assert bbsoi_aug_i.bounding_boxes[0].coords_almost_equals(
                [(4+0, 2+0), (4+4, 2+4)]
            )
            assert bbsoi_aug_i.bounding_boxes[1].coords_almost_equals(
                [(4+1, 2+1), (4+3, 2+4)]
            )

    def test_pad_bounding_boxes_by_tuple_of_fixed_ints_with_keep_size(self):
        aug = iaa.Pad((2, 0, 4, 4), keep_size=True)
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=4, y2=4),
               ia.BoundingBox(x1=1, y1=1, x2=3, y2=4)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(4, 4, 3))
        bbsoi_aug = aug.augment_bounding_boxes([bbsoi, bbsoi])
        assert len(bbsoi_aug) == 2
        for bbsoi_aug_i in bbsoi_aug:
            assert bbsoi_aug_i.shape == (4, 4, 3)
            assert len(bbsoi_aug_i.bounding_boxes) == 2
            assert bbsoi_aug_i.bounding_boxes[0].coords_almost_equals(
                [(4*((4+0)/8), 4*((2+0)/10)), (4*((4+4)/8), 4*((2+4)/10))]
            )
            assert bbsoi_aug_i.bounding_boxes[1].coords_almost_equals(
                [(4*((4+1)/8), 4*((2+1)/10)), (4*((4+3)/8), 4*((2+4)/10))]
            )

    def test_pad_mode_is_stochastic_parameter(self):
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode=iap.Choice(["constant", "maximum", "edge"]),
                      pad_cval=0,
                      keep_size=False)

        image = np.zeros((1, 2), dtype=np.uint8)
        image[0, 0] = 100
        image[0, 1] = 50

        seen = [0, 0, 0]
        for _ in sm.xrange(300):
            observed = aug.augment_image(image)
            if observed[0, 2] == 0:
                seen[0] += 1
            elif observed[0, 2] == 100:
                seen[1] += 1
            elif observed[0, 2] == 50:
                seen[2] += 1
            else:
                assert False
        assert np.all([100 - 50 < v < 100 + 50 for v in seen])

    def test_bad_datatype_for_pad_mode_causes_failure(self):
        got_exception = False
        try:
            _aug = iaa.Pad(px=(0, 1, 0, 0),
                           pad_mode=False,
                           pad_cval=0,
                           keep_size=False)
        except Exception as exc:
            assert "Expected pad_mode to be " in str(exc)
            got_exception = True
        assert got_exception

    def test_pad_heatmaps_with_pad_mode_set(self):
        # pad modes, heatmaps (always uses constant padding)
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode="edge",
                      pad_cval=0,
                      keep_size=False)
        heatmaps_arr = np.ones((3, 3, 1), dtype=np.float32)
        heatmaps = HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

        observed = aug.augment_heatmaps([heatmaps])[0]

        assert np.sum(observed.get_arr() <= 1e-4) == 3

    def test_pad_segmaps_with_pad_mode_set(self):
        # pad modes, segmaps (always uses constant padding)
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode="edge",
                      pad_cval=0,
                      keep_size=False)
        segmaps_arr = np.ones((3, 3, 1), dtype=np.int32)
        segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

        observed = aug.augment_segmentation_maps([segmaps])[0]

        assert np.sum(observed.get_arr() == 0) == 3

    def test_pad_cval_is_int(self):
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode="constant",
                      pad_cval=100,
                      keep_size=False)
        image = np.zeros((1, 1), dtype=np.uint8)
        observed = aug.augment_image(image)
        assert observed[0, 0] == 0
        assert observed[0, 1] == 100

    def test_pad_cval_is_stochastic_parameter(self):
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode="constant",
                      pad_cval=iap.Choice([50, 100]),
                      keep_size=False)
        image = np.zeros((1, 1), dtype=np.uint8)
        seen = [0, 0]
        for _ in sm.xrange(200):
            observed = aug.augment_image(image)
            if observed[0, 1] == 50:
                seen[0] += 1
            elif observed[0, 1] == 100:
                seen[1] += 1
            else:
                assert False
        assert np.all([100 - 50 < v < 100 + 50 for v in seen])

    def test_pad_cval_is_tuple(self):
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode="constant",
                      pad_cval=(50, 52),
                      keep_size=False)
        image = np.zeros((1, 1), dtype=np.uint8)

        seen = [0, 0, 0]
        for _ in sm.xrange(300):
            observed = aug.augment_image(image)

            if observed[0, 1] == 50:
                seen[0] += 1
            elif observed[0, 1] == 51:
                seen[1] += 1
            elif observed[0, 1] == 52:
                seen[2] += 1
            else:
                assert False
        assert np.all([100 - 50 < v < 100 + 50 for v in seen])

    def test_invalid_pad_cval_datatype_leads_to_failure(self):
        got_exception = False
        try:
            _aug = iaa.Pad(px=(0, 1, 0, 0),
                           pad_mode="constant",
                           pad_cval="test",
                           keep_size=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_pad_heatmaps_with_cval_set(self):
        # pad cvals, heatmaps (should always use cval 0)
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode="constant",
                      pad_cval=255,
                      keep_size=False)
        heatmaps_arr = np.zeros((3, 3, 1), dtype=np.float32)
        heatmaps = HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

        observed = aug.augment_heatmaps([heatmaps])[0]

        assert np.sum(observed.get_arr() > 1e-4) == 0

    def test_pad_segmaps_with_cval_set(self):
        # pad cvals, segmaps (should always use cval 0)
        aug = iaa.Pad(px=(0, 1, 0, 0),
                      pad_mode="constant",
                      pad_cval=255,
                      keep_size=False)
        segmaps_arr = np.zeros((3, 3, 1), dtype=np.int32)
        segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

        observed = aug.augment_segmentation_maps([segmaps])[0]

        assert np.sum(observed.get_arr() > 0) == 0

    def test_pad_all_sides_by_100_percent_without_keep_size(self):
        aug = iaa.Pad(percent=1.0, keep_size=False)
        image = np.zeros((4, 4), dtype=np.uint8) + 1

        observed = aug.augment_image(image)

        assert observed.shape == (4+4+4, 4+4+4)
        assert np.sum(observed[4:-4, 4:-4]) == 4*4
        assert np.sum(observed) == 4*4

    def test_pad_all_sides_by_stochastic_param_without_keep_size(self):
        aug = iaa.Pad(percent=iap.Deterministic(1.0), keep_size=False)
        image = np.zeros((4, 4), dtype=np.uint8) + 1

        observed = aug.augment_image(image)

        assert observed.shape == (4+4+4, 4+4+4)
        assert np.sum(observed[4:-4, 4:-4]) == 4*4
        assert np.sum(observed) == 4*4

    def test_pad_by_tuple_of_two_floats_dont_sample_independently_noks(self):
        aug = iaa.Pad(percent=(1.0, 2.0),
                      sample_independently=False,
                      keep_size=False)
        image = np.zeros((4, 4), dtype=np.uint8) + 1

        observed = aug.augment_image(image)

        assert np.sum(observed) == 4*4
        assert (observed.shape[0] - 4) % 2 == 0
        assert (observed.shape[1] - 4) % 2 == 0

    def test_bad_datatype_for_percent_leads_to_failure_without_keep_size(self):
        got_exception = False
        try:
            _ = iaa.Pad(percent="test", keep_size=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_pad_each_side_by_100_percent_without_keep_size(self):
        image = np.zeros((4, 4), dtype=np.uint8)
        image[0, 0] = 255
        image[3, 0] = 255
        image[0, 3] = 255
        image[3, 3] = 255
        height, width = image.shape[0:2]
        pads = [
            (1.0, 0, 0, 0),
            (0, 1.0, 0, 0),
            (0, 0, 1.0, 0),
            (0, 0, 0, 1.0),
        ]
        for pad in pads:
            with self.subTest(pad=pad):
                top, right, bottom, left = pad
                top_px = int(top * height)
                right_px = int(right * width)
                bottom_px = int(bottom * height)
                left_px = int(left * width)
                aug = iaa.Pad(percent=pad, keep_size=False)
                image_padded = np.pad(
                    image,
                    ((top_px, bottom_px), (left_px, right_px)),
                    mode="constant",
                    constant_values=0)

                observed = aug.augment_image(image)

                assert np.array_equal(observed, image_padded)

    def _test_pad_cba_each_side_by_100_percent_without_keep_size(
            self, augf_name, cbaoi):
        height, width = cbaoi.shape[0:2]
        pads = [
            (1.0, 0, 0, 0),
            (0, 1.0, 0, 0),
            (0, 0, 1.0, 0),
            (0, 0, 0, 1.0),
        ]
        for pad in pads:
            with self.subTest(pad=pad):
                top, right, bottom, left = pad
                top_px = int(top * height)
                left_px = int(left * width)
                aug = iaa.Pad(percent=pad, keep_size=False)
                cbaoi_moved = shift_cbaoi(cbaoi, left=left_px, top=top_px)
                cbaoi_moved.shape = (
                    int(height+top*height+bottom*height),
                    int(width+left*width+right*width)
                )

                observed = getattr(aug, augf_name)(cbaoi)

                assert_cbaois_equal(observed, cbaoi_moved)

    def test_pad_keypoints_each_side_by_100_percent_without_keep_size(self):
        height, width = (4, 4)
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=3, y=3),
               ia.Keypoint(x=3, y=3)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(height, width))
        self._test_pad_cba_each_side_by_100_percent_without_keep_size(
            "augment_keypoints", kpsoi)

    def test_pad_polygons_each_side_by_100_percent_without_keep_size(self):
        height, width = (4, 4)
        polys = [ia.Polygon([(0, 0), (4, 0), (4, 4)]),
                 ia.Polygon([(1, 2), (2, 3), (0, 4)])]
        psoi = ia.PolygonsOnImage(polys, shape=(height, width))
        self._test_pad_cba_each_side_by_100_percent_without_keep_size(
            "augment_polygons", psoi)

    def test_pad_line_strings_each_side_by_100_percent_without_keep_size(self):
        height, width = (4, 4)
        lss = [ia.LineString([(0, 0), (4, 0), (4, 4)]),
               ia.LineString([(1, 2), (2, 3), (0, 4)])]
        lsoi = ia.LineStringsOnImage(lss, shape=(height, width))
        self._test_pad_cba_each_side_by_100_percent_without_keep_size(
            "augment_line_strings", lsoi)

    def test_pad_bbs_each_side_by_100_percent_without_keep_size(self):
        height, width = (4, 4)
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=4, y2=4),
               ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(height, width))
        self._test_pad_cba_each_side_by_100_percent_without_keep_size(
            "augment_bounding_boxes", bbsoi)

    def test_pad_heatmaps_smaller_than_img_by_floats_without_keep_size(self):
        # pad smaller heatmaps
        # heatmap is (6, 4), image is (6, 16)
        # image is padded by (0.5, 0.25, 0.5, 0.25)
        # expected image size: (12, 24)
        # expected heatmap size: (12, 6)
        aug = iaa.Pad(percent=(0.5, 0.25, 0.5, 0.25), keep_size=False)
        heatmaps_arr_small = np.float32([
            [0, 0, 0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 0, 0, 0]
        ])
        top, bottom, left, right = 3, 3, 1, 1
        heatmaps_arr_small_padded = np.pad(
            heatmaps_arr_small,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0)
        heatmaps = [ia.HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))]

        observed = aug.augment_heatmaps(heatmaps)[0]

        assert observed.shape == (12, 24)
        assert observed.arr_0to1.shape == (12, 6, 1)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.arr_0to1[..., 0], heatmaps_arr_small_padded)

    def test_pad_segmaps_smaller_than_img_by_floats_without_keep_size(self):
        aug = iaa.Pad(percent=(0.5, 0.25, 0.5, 0.25), keep_size=False)
        segmaps_arr_small = np.int32([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        top, bottom, left, right = 3, 3, 1, 1
        segmaps_arr_small_padded = np.pad(
            segmaps_arr_small,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0)
        segmaps = [SegmentationMapsOnImage(segmaps_arr_small, shape=(6, 16))]

        observed = aug.augment_segmentation_maps(segmaps)[0]

        assert observed.shape == (12, 24)
        assert observed.arr.shape == (12, 6, 1)
        assert np.array_equal(observed.arr[..., 0], segmaps_arr_small_padded)

    def test_pad_heatmaps_smaller_than_img_by_floats_with_keep_size(self):
        # pad smaller heatmaps, with keep_size=True
        # heatmap is (6, 4), image is (6, 16)
        # image is padded by (0.5, 0.25, 0.5, 0.25)
        # expected image size: (12, 24) -> (6, 16) after resize
        # expected heatmap size: (12, 6) -> (6, 4) after resize
        aug = iaa.Pad(percent=(0.5, 0.25, 0.5, 0.25), keep_size=True)
        heatmaps_arr_small = np.float32([
            [0, 0, 0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 1.0, 1.0, 0],
            [0, 0, 0, 0]
        ])
        top, bottom, left, right = 3, 3, 1, 1
        heatmaps_arr_small_padded = np.pad(
            heatmaps_arr_small,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0)
        heatmaps = [ia.HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))]

        observed = aug.augment_heatmaps(heatmaps)[0]
        assert observed.shape == (6, 16)
        assert observed.arr_0to1.shape == (6, 4, 1)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(
            observed.arr_0to1[..., 0],
            np.clip(
                ia.imresize_single_image(
                    heatmaps_arr_small_padded, (6, 4), interpolation="cubic"),
                0, 1.0
            )
        )

    def test_pad_segmaps_smaller_than_img_by_floats_with_keep_size(self):
        aug = iaa.Pad(percent=(0.5, 0.25, 0.5, 0.25), keep_size=True)
        segmaps_arr_small = np.int32([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ])
        top, bottom, left, right = 3, 3, 1, 1
        segmaps_arr_small_padded = np.pad(
            segmaps_arr_small,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0)
        segmaps = [SegmentationMapsOnImage(segmaps_arr_small, shape=(6, 16))]

        observed = aug.augment_segmentation_maps(segmaps)[0]

        assert observed.shape == (6, 16)
        assert observed.arr.shape == (6, 4, 1)
        assert np.array_equal(
            observed.arr[..., 0],
            ia.imresize_single_image(
                segmaps_arr_small_padded, (6, 4), interpolation="nearest")
        )

    def test_pad_keypoints_by_floats_without_keep_size(self):
        aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=False)
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=0)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(4, 4, 3))
        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
        assert kpsoi_aug.shape == (10, 8, 3)
        assert len(kpsoi_aug.keypoints) == 2
        assert np.allclose(kpsoi_aug.keypoints[0].x, 4+1)
        assert np.allclose(kpsoi_aug.keypoints[0].y, 2+2)
        assert np.allclose(kpsoi_aug.keypoints[1].x, 4+3)
        assert np.allclose(kpsoi_aug.keypoints[1].y, 2+0)

    def test_pad_keypoints_by_floats_with_keep_size(self):
        aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=True)
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=0)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(4, 4, 3))
        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
        assert kpsoi_aug.shape == (4, 4, 3)
        assert len(kpsoi_aug.keypoints) == 2
        assert np.allclose(kpsoi_aug.keypoints[0].x, ((4+1)/8)*4)
        assert np.allclose(kpsoi_aug.keypoints[0].y, ((2+2)/10)*4)
        assert np.allclose(kpsoi_aug.keypoints[1].x, ((4+3)/8)*4)
        assert np.allclose(kpsoi_aug.keypoints[1].y, ((2+0)/10)*4)

    def test_pad_polygons_by_floats_without_keep_size(self):
        aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=False)
        cbaoi = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
            ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
        ], shape=(4, 4, 3))
        cbaoi_aug = aug.augment_polygons([cbaoi, cbaoi])
        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (10, 8, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(4, 2), (8, 2), (8, 6), (4, 6)]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(5, 3), (9, 3), (9, 7), (5, 7)]
            )

    def test_pad_polygons_by_floats_with_keep_size(self):
        # polygons, with keep_size=True
        aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=True)
        cbaoi = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
            ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
        ], shape=(4, 4, 3))
        cbaoi_aug = aug.augment_polygons([cbaoi, cbaoi])
        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (4, 4, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(4*(4/8), 4*(2/10)),
                 (4*(8/8), 4*(2/10)),
                 (4*(8/8), 4*(6/10)),
                 (4*(4/8), 4*(6/10))]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(4*(5/8), 4*(3/10)),
                 (4*(9/8), 4*(3/10)),
                 (4*(9/8), 4*(7/10)),
                 (4*(5/8), 4*(7/10))]
            )

    def test_pad_line_strings_by_floats_without_keep_size(self):
        aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=False)
        cbaoi = ia.LineStringsOnImage([
            ia.LineString([(0, 0), (4, 0), (4, 4), (0, 4)]),
            ia.LineString([(1, 1), (5, 1), (5, 5), (1, 5)])
        ], shape=(4, 4, 3))
        cbaoi_aug = aug.augment_line_strings([cbaoi, cbaoi])
        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (10, 8, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(4, 2), (8, 2), (8, 6), (4, 6)]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(5, 3), (9, 3), (9, 7), (5, 7)]
            )

    def test_pad_line_strings_by_floats_with_keep_size(self):
        # polygons, with keep_size=True
        aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=True)
        cbaoi = ia.LineStringsOnImage([
            ia.LineString([(0, 0), (4, 0), (4, 4), (0, 4)]),
            ia.LineString([(1, 1), (5, 1), (5, 5), (1, 5)])
        ], shape=(4, 4, 3))
        cbaoi_aug = aug.augment_line_strings([cbaoi, cbaoi])
        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (4, 4, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(4*(4/8), 4*(2/10)),
                 (4*(8/8), 4*(2/10)),
                 (4*(8/8), 4*(6/10)),
                 (4*(4/8), 4*(6/10))]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(4*(5/8), 4*(3/10)),
                 (4*(9/8), 4*(3/10)),
                 (4*(9/8), 4*(7/10)),
                 (4*(5/8), 4*(7/10))]
            )

    def test_pad_bounding_boxes_by_floats_without_keep_size(self):
        aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=False)
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=4, y2=4),
               ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(4, 4, 3))
        bbsoi_aug = aug.augment_bounding_boxes([bbsoi, bbsoi])
        assert len(bbsoi_aug) == 2
        for bbsoi_aug_i in bbsoi_aug:
            assert bbsoi_aug_i.shape == (10, 8, 3)
            assert len(bbsoi_aug_i.bounding_boxes) == 2
            assert bbsoi_aug_i.bounding_boxes[0].coords_almost_equals(
                [(int(1.0*4+0), int(0.5*4+0)),
                 (int(1.0*4+4), int(0.5*4+4))]
            )
            assert bbsoi_aug_i.bounding_boxes[1].coords_almost_equals(
                [(int(1.0*4+1), int(0.5*4+2)),
                 (int(1.0*4+3), int(0.5*4+4))]
            )

    def test_pad_bounding_boxes_by_floats_with_keep_size(self):
        # BBs, with keep_size=True
        aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=True)
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=4, y2=4),
               ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(4, 4, 3))
        bbsoi_aug = aug.augment_bounding_boxes([bbsoi, bbsoi])
        assert len(bbsoi_aug) == 2
        for bbsoi_aug_i in bbsoi_aug:
            assert bbsoi_aug_i.shape == (4, 4, 3)
            assert len(bbsoi_aug_i.bounding_boxes) == 2
            assert bbsoi_aug_i.bounding_boxes[0].coords_almost_equals(
                [(4*(4/8), 4*(2/10)),
                 (4*(8/8), 4*(6/10))]
            )
            assert bbsoi_aug_i.bounding_boxes[1].coords_almost_equals(
                [(4*(5/8), 4*(4/10)),
                 (4*(7/8), 4*(6/10))]
            )

    def test_pad_by_tuple_of_floats_at_top_side_without_keep_size(self):
        # test pad by range of percentages
        aug = iaa.Pad(percent=((0, 1.0), 0, 0, 0), keep_size=False)
        seen = [0, 0, 0, 0, 0]
        for _ in sm.xrange(500):
            observed = aug.augment_image(
                np.zeros((4, 4), dtype=np.uint8) + 255)
            n_padded = 0
            while np.all(observed[0, :] == 0):
                n_padded += 1
                observed = observed[1:, :]
            seen[n_padded] += 1
        # note that we cant just check for 100-50 < x < 100+50 here. The
        # first and last value (0px and 4px) padding have half the
        # probability of occuring compared to the other values. E.g. 0px is
        # padded if sampled p falls in range [0, 0.125). 1px is padded if
        # sampled p falls in range [0.125, 0.375].
        assert np.all([v > 30 for v in seen])

    def test_pad_by_tuple_of_floats_at_right_side_without_keep_size(self):
        aug = iaa.Pad(percent=(0, (0, 1.0), 0, 0), keep_size=False)
        seen = [0, 0, 0, 0, 0]
        for _ in sm.xrange(500):
            observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 255)
            n_padded = 0
            while np.all(observed[:, -1] == 0):
                n_padded += 1
                observed = observed[:, 0:-1]
            seen[n_padded] += 1
        assert np.all([v > 30 for v in seen])

    def test_pad_by_list_of_floats_at_top_side_without_keep_size(self):
        aug = iaa.Pad(percent=([0.0, 1.0], 0, 0, 0), keep_size=False)
        seen = [0, 0, 0, 0, 0]
        for _ in sm.xrange(500):
            observed = aug.augment_image(
                np.zeros((4, 4), dtype=np.uint8) + 255)
            n_padded = 0
            while np.all(observed[0, :] == 0):
                n_padded += 1
                observed = observed[1:, :]
            seen[n_padded] += 1
        assert 250 - 50 < seen[0] < 250 + 50
        assert seen[1] == 0
        assert seen[2] == 0
        assert seen[3] == 0
        assert 250 - 50 < seen[4] < 250 + 50

    def test_pad_by_list_of_floats_at_right_side_without_keep_size(self):
        aug = iaa.Pad(percent=(0, [0.0, 1.0], 0, 0), keep_size=False)
        seen = [0, 0, 0, 0, 0]
        for _ in sm.xrange(500):
            observed = aug.augment_image(
                np.zeros((4, 4), dtype=np.uint8) + 255)
            n_padded = 0
            while np.all(observed[:, -1] == 0):
                n_padded += 1
                observed = observed[:, 0:-1]
            seen[n_padded] += 1
        assert 250 - 50 < seen[0] < 250 + 50
        assert seen[1] == 0
        assert seen[2] == 0
        assert seen[3] == 0
        assert 250 - 50 < seen[4] < 250 + 50

    @classmethod
    def _test_pad_empty_cba(cls, augf_name, cbaoi):
        aug = iaa.Pad(px=(1, 2, 3, 4), keep_size=False)

        cbaoi_aug = getattr(aug, augf_name)(cbaoi)

        expected = cbaoi.deepcopy()
        expected.shape = tuple(
            [1+expected.shape[0]+3, 4+expected.shape[1]+2]
            + list(expected.shape[2:]))
        assert_cbaois_equal(cbaoi_aug, expected)

    def test_pad_empty_keypoints(self):
        cbaoi = ia.KeypointsOnImage([], shape=(2, 4, 3))
        self._test_pad_empty_cba("augment_keypoints", cbaoi)

    def test_pad_empty_polygons(self):
        cbaoi = ia.PolygonsOnImage([], shape=(2, 4, 3))
        self._test_pad_empty_cba("augment_polygons", cbaoi)

    def test_pad_empty_line_strings(self):
        cbaoi = ia.LineStringsOnImage([], shape=(2, 4, 3))
        self._test_pad_empty_cba("augment_line_strings", cbaoi)

    def test_pad_empty_bounding_boxes(self):
        cbaoi = ia.BoundingBoxesOnImage([], shape=(2, 4, 3))
        self._test_pad_empty_cba("augment_bounding_boxes", cbaoi)

    def test_zero_sized_axes_no_keep_size(self):
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
                aug = iaa.Pad(px=1, keep_size=False)

                image_aug = aug(image=image)

                expected_height = shape[0] + 2
                expected_width = shape[1] + 2
                expected_shape = tuple([expected_height, expected_width]
                                       + list(shape[2:]))
                assert image_aug.shape == expected_shape

    def test_zero_sized_axes_keep_size(self):
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
                aug = iaa.Pad(px=1, keep_size=True)

                image_aug = aug(image=image)

                assert image_aug.shape == image.shape

    def test_pad_other_dtypes_bool_by_int_without_keep_size(self):
        aug = iaa.Pad(px=(1, 0, 0, 0), keep_size=False)
        mask = np.zeros((4, 3), dtype=bool)
        mask[2, 1] = True
        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == image.dtype.name
        assert image_aug.shape == (4, 3)
        assert np.all(image_aug[~mask] == 0)
        assert np.all(image_aug[mask] == 1)

    def test_pad_other_dtypes_uint_int_by_int_without_keep_size(self):
        aug = iaa.Pad(px=(1, 0, 0, 0), keep_size=False)

        mask = np.zeros((4, 3), dtype=bool)
        mask[2, 1] = True

        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64"]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                if np.dtype(dtype).kind == "i":
                    values = [
                        1, 5, 10, 100, int(0.1 * max_value),
                        int(0.2 * max_value), int(0.5 * max_value),
                        max_value - 100, max_value]
                    values = values + [(-1) * value for value in values]
                else:
                    values = [
                        1, 5, 10, 100, int(center_value),
                        int(0.1 * max_value), int(0.2 * max_value),
                        int(0.5 * max_value), max_value - 100, max_value]

                for value in values:
                    image = np.zeros((3, 3), dtype=dtype)
                    image[1, 1] = value
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.name == dtype
                    assert image_aug.shape == (4, 3)
                    assert np.all(image_aug[~mask] == 0)
                    assert np.all(image_aug[mask] == value)

    def test_pad_other_dtypes_float_by_int_without_keep_size(self):
        aug = iaa.Pad(px=(1, 0, 0, 0), keep_size=False)

        mask = np.zeros((4, 3), dtype=bool)
        mask[2, 1] = True

        dtypes = ["float16", "float32", "float64", "float128"]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                def _isclose(a, b):
                    atol = 1e-4 if dtype == np.float16 else 1e-8
                    return np.isclose(a, b, atol=atol, rtol=0)

                isize = np.dtype(dtype).itemsize
                values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1),
                          1000 ** (isize - 1)]
                values = values + [(-1) * value for value in values]
                values = values + [min_value, max_value]
                for value in values:
                    image = np.zeros((3, 3), dtype=dtype)
                    image[1, 1] = value
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype == np.dtype(dtype)
                    assert image_aug.shape == (4, 3)
                    assert np.all(_isclose(image_aug[~mask], 0))
                    assert np.all(_isclose(image_aug[mask],
                                           np.float128(value)))


class TestCrop(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]], dtype=np.uint8)
        base_img = base_img[:, :, np.newaxis]
        return base_img

    @property
    def images(self):
        return np.array([self.image])

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        kpsoi = ia.KeypointsOnImage(kps, shape=self.image.shape)
        return kpsoi

    @property
    def psoi(self):
        ps = [ia.Polygon([(1, 1), (2, 1), (2, 2)])]
        psoi = ia.PolygonsOnImage(ps, shape=self.image.shape)
        return psoi

    @property
    def lsoi(self):
        ls = [ia.LineString([(1, 1), (2, 1), (2, 2)])]
        lsoi = ia.LineStringsOnImage(ls, shape=self.image.shape)
        return lsoi

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=self.image.shape)
        return bbsoi

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([[0, 0, 0],
                                   [0, 1.0, 0],
                                   [0, 0, 0]])
        return [ia.HeatmapsOnImage(heatmaps_arr, shape=self.image.shape)]

    @property
    def segmaps(self):
        segmaps_arr = np.int32([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])
        return [ia.SegmentationMapsOnImage(segmaps_arr, shape=self.image.shape)]

    # TODO split up and add polys/LS/BBs
    def test_crop_by_fixed_int_on_each_side_on_its_own(self):
        # test crop by 1 pixel on each side
        crops = [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ]
        for crop in crops:
            with self.subTest(px=crop):
                aug = iaa.Crop(px=crop, keep_size=False)

                top, right, bottom, left = crop
                height, width = self.image.shape[0:2]

                base_img_cropped = self.image[top:height-bottom,
                                              left:width-right,
                                              :]

                observed = aug.augment_images(self.images)
                assert np.array_equal(observed, np.array([base_img_cropped]))

                observed = aug.augment_images([self.image])
                assert array_equal_lists(observed, [base_img_cropped])

                keypoints_moved = self.kpsoi.shift(x=-left, y=-top)
                observed = aug.augment_keypoints(self.kpsoi)
                assert keypoints_equal(observed, keypoints_moved)

                heatmaps_arr = self.heatmaps[0].get_arr()
                height, width = heatmaps_arr.shape[0:2]
                heatmaps_arr_cropped = heatmaps_arr[top:height-bottom,
                                                    left:width-right]
                observed = aug.augment_heatmaps(self.heatmaps)[0]
                assert observed.shape == base_img_cropped.shape
                assert np.array_equal(observed.get_arr(), heatmaps_arr_cropped)

                segmaps_arr = self.segmaps[0].get_arr()
                height, width = segmaps_arr.shape[0:2]
                segmaps_arr_cropped = segmaps_arr[top:height-bottom,
                                                  left:width-right]
                observed = aug.augment_segmentation_maps(self.segmaps)[0]
                assert observed.shape == base_img_cropped.shape
                assert np.array_equal(observed.get_arr(), segmaps_arr_cropped)

    # TODO split up and add polys/LS/BBs
    def test_crop_by_tuple_of_ints_on_each_side_on_its_own(self):
        def _to_range_tuple(val):
            return val if isinstance(val, tuple) else (val, val)

        crops = [
            ((0, 2), 0, 0, 0),
            (0, (0, 2), 0, 0),
            (0, 0, (0, 2), 0),
            (0, 0, 0, (0, 2)),
        ]
        for crop in crops:
            with self.subTest(px=crop):
                aug = iaa.Crop(px=crop, keep_size=False)
                aug_det = aug.to_deterministic()

                top, right, bottom, left = crop
                height, width = self.image.shape[0:2]

                top_range = _to_range_tuple(top)
                right_range = _to_range_tuple(right)
                bottom_range = _to_range_tuple(bottom)
                left_range = _to_range_tuple(left)

                top_values = sm.xrange(top_range[0], top_range[1]+1)
                right_values = sm.xrange(right_range[0], right_range[1]+1)
                bottom_values = sm.xrange(bottom_range[0], bottom_range[1]+1)
                left_values = sm.xrange(left_range[0], left_range[1]+1)

                images_cropped = []
                keypoints_cropped = []
                for top_val in top_values:
                    for right_val in right_values:
                        for bottom_val in bottom_values:
                            for left_val in left_values:
                                images_cropped.append(
                                    self.image[top_val:height-bottom_val,
                                               left_val:width-right_val,
                                               :]
                                )
                                keypoints_cropped.append(
                                    self.kpsoi.shift(
                                        x=-left_val, y=-top_val)
                                )

                movements = []
                movements_det = []
                for i in sm.xrange(100):
                    observed = aug.augment_images(self.images)

                    matches = [
                        (1
                         if np.array_equal(observed,
                                           np.array([base_img_cropped]))
                         else 0)
                        for base_img_cropped
                        in images_cropped]
                    movements.append(np.argmax(np.array(matches)))
                    assert any([val == 1 for val in matches])

                    observed = aug_det.augment_images(self.images)
                    matches = [
                        (1
                         if np.array_equal(observed,
                                           np.array([base_img_cropped]))
                         else 0)
                        for base_img_cropped
                        in images_cropped]
                    movements_det.append(np.argmax(np.array(matches)))
                    assert any([val == 1 for val in matches])

                    observed = aug.augment_images([self.image])
                    assert any([array_equal_lists(observed, [base_img_cropped])
                                for base_img_cropped
                                in images_cropped])

                    observed = aug.augment_keypoints(self.kpsoi)
                    assert any([keypoints_equal(observed, kp)
                                for kp
                                in keypoints_cropped])

                assert len(set(movements)) == 3
                assert len(set(movements_det)) == 1

    # TODO split up and add polys/LS/BBs
    def test_crop_by_list_of_ints_on_each_side_on_its_own(self):
        # test crop by list of exact pixel values
        crops = [
            ([0, 2], 0, 0, 0),
            (0, [0, 2], 0, 0),
            (0, 0, [0, 2], 0),
            (0, 0, 0, [0, 2]),
        ]
        for crop in crops:
            with self.subTest(px=crop):
                aug = iaa.Crop(px=crop, keep_size=False)
                aug_det = aug.to_deterministic()

                top, right, bottom, left = crop
                height, width = self.image.shape[0:2]

                top_range = top if isinstance(top, list) else [top]
                right_range = right if isinstance(right, list) else [right]
                bottom_range = bottom if isinstance(bottom, list) else [bottom]
                left_range = left if isinstance(left, list) else [left]

                images_cropped = []
                keypoints_cropped = []
                for top_val in top_range:
                    for right_val in right_range:
                        for bottom_val in bottom_range:
                            for left_val in left_range:
                                images_cropped.append(
                                    self.image[top_val:height-bottom_val,
                                               left_val:width-right_val,
                                               :]
                                )
                                keypoints_cropped.append(
                                    self.kpsoi.shift(
                                        x=-left_val, y=-top_val)
                                )

                movements = []
                movements_det = []
                for i in sm.xrange(100):
                    observed = aug.augment_images(self.images)
                    matches = [
                        (1
                         if np.array_equal(observed,
                                           np.array([base_img_cropped]))
                         else 0)
                        for base_img_cropped
                        in images_cropped]
                    movements.append(np.argmax(np.array(matches)))
                    assert any([val == 1 for val in matches])

                    observed = aug_det.augment_images(self.images)
                    matches = [
                        (1
                         if np.array_equal(observed,
                                           np.array([base_img_cropped]))
                         else 0)
                        for base_img_cropped in images_cropped]
                    movements_det.append(np.argmax(np.array(matches)))
                    assert any([val == 1 for val in matches])

                    observed = aug.augment_images([self.image])
                    assert any([array_equal_lists(observed, [base_img_cropped])
                                for base_img_cropped
                                in images_cropped])

                    observed = aug.augment_keypoints(self.kpsoi)
                    assert any([keypoints_equal(observed, kp)
                                for kp
                                in keypoints_cropped])

                assert len(set(movements)) == 2
                assert len(set(movements_det)) == 1

    def test_crop_heatmaps_smaller_than_img_by_fixed_ints_without_ks(self):
        # crop smaller heatmaps
        # heatmap is (6, 8), image is (6, 16)
        # image is cropped by (1, 4, 1, 4)
        # expected image size: (4, 8)
        # expected heatmap size: (4, 4)
        aug = iaa.Crop(px=(1, 4, 1, 4), keep_size=False)
        heatmaps_arr_small = np.zeros((6, 8), dtype=np.float32)
        heatmaps_arr_small[1:-1, 1:-1] = 1.0
        heatmaps = HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))
        top, bottom, left, right = 1, 1, 2, 2
        heatmaps_arr_small_cropped = \
            heatmaps_arr_small[top:-bottom, left:-right]

        observed = aug.augment_heatmaps([heatmaps])[0]

        assert observed.shape == (4, 8)
        assert observed.arr_0to1.shape == (4, 4, 1)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.arr_0to1[..., 0],
                           heatmaps_arr_small_cropped)

    def test_crop_segmaps_smaller_than_img_by_fixed_ints_without_ks(self):
        aug = iaa.Crop(px=(1, 4, 1, 4), keep_size=False)
        segmaps_arr_small = np.zeros((6, 8), dtype=np.int32)
        segmaps_arr_small[1:-1, 1:-1] = 1
        segmaps = SegmentationMapsOnImage(segmaps_arr_small, shape=(6, 16))
        top, bottom, left, right = 1, 1, 2, 2
        segmaps_arr_small_cropped = segmaps_arr_small[top:-bottom, left:-right]

        observed = aug.augment_segmentation_maps([segmaps])[0]

        assert observed.shape == (4, 8)
        assert observed.arr.shape == (4, 4, 1)
        assert np.array_equal(observed.arr[..., 0], segmaps_arr_small_cropped)

    def test_crop_heatmaps_smaller_than_img_by_fixed_ints_with_ks(self):
        # crop smaller heatmaps, with keep_size=True
        # heatmap is (6, 8), image is (6, 16)
        # image is cropped by (1, 4, 1, 4)
        # expected image size: (4, 8) -> (6, 16) after resize
        # expected heatmap size: (4, 4) -> (6, 4) after resize
        aug = iaa.Crop(px=(1, 4, 1, 4), keep_size=True)
        heatmaps_arr_small = np.zeros((6, 8), dtype=np.float32)
        heatmaps_arr_small[1:-1, 1:-1] = 1.0
        heatmaps = HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))
        top, bottom, left, right = 1, 1, 2, 2
        heatmaps_arr_small_cropped = \
            heatmaps_arr_small[top:-bottom, left:-right]

        observed = aug.augment_heatmaps([heatmaps])[0]

        assert observed.shape == (6, 16)
        assert observed.arr_0to1.shape == (6, 8, 1)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(
            observed.arr_0to1[..., 0],
            np.clip(
                ia.imresize_single_image(
                    heatmaps_arr_small_cropped,
                    (6, 8),
                    interpolation="cubic"),
                0,
                1.0
            )
        )

    def test_crop_segmaps_smaller_than_img_by_fixed_ints_with_ks(self):
        aug = iaa.Crop(px=(1, 4, 1, 4), keep_size=True)
        segmaps_arr_small = np.zeros((6, 8), dtype=np.int32)
        segmaps_arr_small[1:-1, 1:-1] = 1
        segmaps = SegmentationMapsOnImage(segmaps_arr_small, shape=(6, 16))
        top, bottom, left, right = 1, 1, 2, 2
        segmaps_arr_small_cropped = segmaps_arr_small[top:-bottom, left:-right]

        observed = aug.augment_segmentation_maps([segmaps])[0]

        assert observed.shape == (6, 16)
        assert observed.arr.shape == (6, 8, 1)
        assert np.array_equal(
            observed.arr[..., 0],
            ia.imresize_single_image(
                segmaps_arr_small_cropped,
                (6, 8),
                interpolation="nearest"),
        )

    def test_crop_keypoints_by_fixed_ints_without_keep_size(self):
        aug = iaa.Crop((1, 0, 4, 4), keep_size=False)
        kps = [ia.Keypoint(x=3, y=6), ia.Keypoint(x=8, y=5)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(14, 14, 3))

        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]

        assert kpsoi_aug.shape == (9, 10, 3)
        assert len(kpsoi_aug.keypoints) == 2
        assert np.allclose(kpsoi_aug.keypoints[0].x, 3-4)
        assert np.allclose(kpsoi_aug.keypoints[0].y, 6-1)
        assert np.allclose(kpsoi_aug.keypoints[1].x, 8-4)
        assert np.allclose(kpsoi_aug.keypoints[1].y, 5-1)

    def test_crop_keypoints_by_fixed_ints_with_keep_size(self):
        aug = iaa.Crop((1, 0, 4, 4), keep_size=True)
        kps = [ia.Keypoint(x=3, y=6), ia.Keypoint(x=8, y=5)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(14, 14, 3))

        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]

        assert kpsoi_aug.shape == (14, 14, 3)
        assert len(kpsoi_aug.keypoints) == 2
        assert np.allclose(kpsoi_aug.keypoints[0].x, ((3-4)/10)*14)
        assert np.allclose(kpsoi_aug.keypoints[0].y, ((6-1)/9)*14)
        assert np.allclose(kpsoi_aug.keypoints[1].x, ((8-4)/10)*14)
        assert np.allclose(kpsoi_aug.keypoints[1].y, ((5-1)/9)*14)

    def test_crop_polygons_by_fixed_ints_without_keep_size(self):
        aug = iaa.Crop((1, 0, 4, 4), keep_size=False)
        polygons = [ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
                    ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])]
        cbaoi = ia.PolygonsOnImage(polygons, shape=(10, 10, 3))

        cbaoi_aug = aug.augment_polygons([cbaoi, cbaoi])

        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (5, 6, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(0-4, 0-1), (4-4, 0-1), (4-4, 4-1), (0-4, 4-1)]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(1-4, 1-1), (5-4, 1-1), (5-4, 5-1), (1-4, 5-1)]
            )

    def test_crop_polygons_by_fixed_ints_with_keep_size(self):
        aug = iaa.Crop((1, 0, 4, 4), keep_size=True)
        polygons = [ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
                    ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])]
        cbaoi = ia.PolygonsOnImage(polygons, shape=(10, 10, 3))

        cbaoi_aug = aug.augment_polygons([cbaoi, cbaoi])

        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (10, 10, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(10*(-4/6), 10*(-1/5)),
                 (10*(0/6), 10*(-1/5)),
                 (10*(0/6), 10*(3/5)),
                 (10*(-4/6), 10*(3/5))]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(10*(-3/6), 10*(0/5)),
                 (10*(1/6), 10*(0/5)),
                 (10*(1/6), 10*(4/5)),
                 (10*(-3/6), 10*(4/5))]
            )

    def test_crop_line_strings_by_fixed_ints_without_keep_size(self):
        aug = iaa.Crop((1, 0, 4, 4), keep_size=False)
        lss = [ia.LineString([(0, 0), (4, 0), (4, 4), (0, 4)]),
               ia.LineString([(1, 1), (5, 1), (5, 5), (1, 5)])]
        cbaoi = ia.LineStringsOnImage(lss, shape=(10, 10, 3))

        cbaoi_aug = aug.augment_line_strings([cbaoi, cbaoi])

        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (5, 6, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(0-4, 0-1), (4-4, 0-1), (4-4, 4-1), (0-4, 4-1)]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(1-4, 1-1), (5-4, 1-1), (5-4, 5-1), (1-4, 5-1)]
            )

    def test_crop_line_strings_by_fixed_ints_with_keep_size(self):
        aug = iaa.Crop((1, 0, 4, 4), keep_size=True)
        lss = [ia.LineString([(0, 0), (4, 0), (4, 4), (0, 4)]),
               ia.LineString([(1, 1), (5, 1), (5, 5), (1, 5)])]
        cbaoi = ia.LineStringsOnImage(lss, shape=(10, 10, 3))

        cbaoi_aug = aug.augment_line_strings([cbaoi, cbaoi])

        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (10, 10, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(10*(-4/6), 10*(-1/5)),
                 (10*(0/6), 10*(-1/5)),
                 (10*(0/6), 10*(3/5)),
                 (10*(-4/6), 10*(3/5))]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(10*(-3/6), 10*(0/5)),
                 (10*(1/6), 10*(0/5)),
                 (10*(1/6), 10*(4/5)),
                 (10*(-3/6), 10*(4/5))]
            )

    def test_crop_bounding_boxes_by_fixed_ints_without_keep_size(self):
        aug = iaa.Crop((1, 0, 4, 4), keep_size=False)
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=10, y2=10),
               ia.BoundingBox(x1=1, y1=2, x2=9, y2=10)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(10, 10, 3))

        bbsoi_aug = aug.augment_bounding_boxes([bbsoi, bbsoi])

        assert len(bbsoi_aug) == 2
        for bbsoi_aug_i in bbsoi_aug:
            assert bbsoi_aug_i.shape == (5, 6, 3)
            assert len(bbsoi_aug_i.bounding_boxes) == 2
            assert bbsoi_aug_i.bounding_boxes[0].coords_almost_equals(
                [(0-4, 0-1), (10-4, 10-1)]
            )
            assert bbsoi_aug_i.bounding_boxes[1].coords_almost_equals(
                [(1-4, 2-1), (9-4, 10-1)]
            )

    def test_crop_bounding_boxes_by_fixed_ints_with_keep_size(self):
        aug = iaa.Crop((1, 0, 4, 4), keep_size=True)
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=10, y2=10),
               ia.BoundingBox(x1=1, y1=2, x2=9, y2=10)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(10, 10, 3))

        bbsoi_aug = aug.augment_bounding_boxes([bbsoi, bbsoi])

        assert len(bbsoi_aug) == 2
        for bbsoi_aug_i in bbsoi_aug:
            assert bbsoi_aug_i.shape == (10, 10, 3)
            assert len(bbsoi_aug_i.bounding_boxes) == 2
            assert bbsoi_aug_i.bounding_boxes[0].coords_almost_equals(
                [(10*(-4/6), 10*(-1/5)),
                 (10*(6/6), 10*(9/5))]
            )
            assert bbsoi_aug_i.bounding_boxes[1].coords_almost_equals(
                [(10*(-3/6), 10*(1/5)),
                 (10*(5/6), 10*(9/5))]
            )

    def test_crop_by_one_fixed_float_without_keep_size(self):
        aug = iaa.Crop(percent=0.1, keep_size=False)
        image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)

        observed = aug.augment_image(image)

        assert observed.shape == (40, 40)
        assert np.all(observed == image[5:-5, 5:-5])

    def test_crop_by_stochastic_parameter_without_keep_size(self):
        aug = iaa.Crop(percent=iap.Deterministic(0.1), keep_size=False)
        image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)

        observed = aug.augment_image(image)

        assert observed.shape == (40, 40)
        assert np.all(observed == image[5:-5, 5:-5])

    def test_crop_by_tuple_of_two_floats_without_keep_size(self):
        aug = iaa.Crop(percent=(0.1, 0.2), keep_size=False)
        image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)

        observed = aug.augment_image(image)

        assert 30 <= observed.shape[0] <= 40
        assert 30 <= observed.shape[1] <= 40

    def test_invalid_datatype_for_percent_parameter_fails(self):
        got_exception = False
        try:
            _ = iaa.Crop(percent="test", keep_size=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_crop_by_fixed_float_on_each_side_on_its_own(self):
        image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
        height, width = image.shape[0:2]
        crops = [
            (0.1, 0, 0, 0),
            (0, 0.1, 0, 0),
            (0, 0, 0.1, 0),
            (0, 0, 0, 0.1),
        ]
        for crop in crops:
            with self.subTest(percent=crop):
                aug = iaa.Crop(percent=crop, keep_size=False)

                top, right, bottom, left = crop
                top_px = int(round(top * height))
                right_px = int(round(right * width))
                bottom_px = int(round(bottom * height))
                left_px = int(round(left * width))

                # dont use :-bottom_px and ;-right_px here, because these
                # values can be 0
                image_cropped = image[top_px:50-bottom_px, left_px:50-right_px]
                observed = aug.augment_image(image)
                assert np.array_equal(observed, image_cropped)

    def _test_crop_cba_by_fixed_float_on_each_side_on_its_own(
            self, augf_name, cbaoi):
        height, width = cbaoi.shape[0:2]
        crops = [
            (0.1, 0, 0, 0),
            (0, 0.1, 0, 0),
            (0, 0, 0.1, 0),
            (0, 0, 0, 0.1),
        ]
        for crop in crops:
            with self.subTest(augf_name=augf_name, percent=crop):
                aug = iaa.Crop(percent=crop, keep_size=False)

                top, right, bottom, left = crop
                top_px = int(round(top * height))
                right_px = int(round(right * width))
                left_px = int(round(left * width))
                bottom_px = int(round(bottom * height))

                observed = getattr(aug, augf_name)(cbaoi)

                expected = shift_cbaoi(cbaoi, left=-left_px, top=-top_px)
                expected.shape = tuple(
                    [expected.shape[0] - top_px - bottom_px,
                     expected.shape[1] - left_px - right_px]
                    + list(expected.shape[2:])
                )
                assert_cbaois_equal(observed, expected)

    def test_crop_keypoints_by_fixed_float_on_each_side_on_its_own(self):
        height, width = (50, 50)
        kps = [ia.Keypoint(x=10, y=11), ia.Keypoint(x=20, y=21),
               ia.Keypoint(x=30, y=31)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(height, width))
        self._test_crop_cba_by_fixed_float_on_each_side_on_its_own(
            "augment_keypoints", kpsoi)

    def test_crop_polygons_by_fixed_float_on_each_side_on_its_own(self):
        height, width = (50, 50)
        polygons = [ia.Polygon([(0, 0), (40, 0), (40, 40), (0, 40)]),
                    ia.Polygon([(10, 10), (50, 10), (50, 50), (10, 50)])]
        psoi = ia.PolygonsOnImage(polygons, shape=(height, width, 3))
        self._test_crop_cba_by_fixed_float_on_each_side_on_its_own(
            "augment_polygons", psoi)

    def test_crop_line_strings_by_fixed_float_on_each_side_on_its_own(self):
        height, width = (50, 50)
        lss = [ia.LineString([(0, 0), (40, 0), (40, 40), (0, 40)]),
               ia.LineString([(10, 10), (50, 10), (50, 50), (10, 50)])]
        lsoi = ia.LineStringsOnImage(lss, shape=(height, width, 3))
        self._test_crop_cba_by_fixed_float_on_each_side_on_its_own(
            "augment_line_strings", lsoi)

    def test_crop_bounding_boxes_by_fixed_float_on_each_side_on_its_own(self):
        height, width = (50, 50)
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=40, y2=40),
               ia.BoundingBox(x1=10, y1=10, x2=30, y2=40)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(height, width, 3))
        self._test_crop_cba_by_fixed_float_on_each_side_on_its_own(
            "augment_bounding_boxes", bbsoi)

    def test_crop_heatmaps_smaller_than_img_by_fixed_floats_without_ks(self):
        # crop smaller heatmaps
        # heatmap is (8, 12), image is (16, 32)
        # image is cropped by (0.25, 0.25, 0.25, 0.25)
        # expected image size: (8, 16)
        # expected heatmap size: (4, 6)
        aug = iaa.Crop(percent=(0.25, 0.25, 0.25, 0.25), keep_size=False)
        heatmaps_arr_small = np.zeros((8, 12), dtype=np.float32)
        heatmaps_arr_small[2:-2, 4:-4] = 1.0
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr_small, shape=(16, 32))
        top, bottom, left, right = 2, 2, 3, 3
        heatmaps_arr_small_cropped = \
            heatmaps_arr_small[top:-bottom, left:-right]

        observed = aug.augment_heatmaps([heatmaps])[0]

        assert observed.shape == (8, 16)
        assert observed.arr_0to1.shape == (4, 6, 1)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.arr_0to1[..., 0], heatmaps_arr_small_cropped)

    def test_crop_segmaps_smaller_than_img_by_fixed_floats_without_ks(self):
        aug = iaa.Crop(percent=(0.25, 0.25, 0.25, 0.25), keep_size=False)
        segmaps_arr_small = np.zeros((8, 12), dtype=np.int32)
        segmaps_arr_small[2:-2, 4:-4] = 1
        segmaps = SegmentationMapsOnImage(segmaps_arr_small, shape=(16, 32))
        top, bottom, left, right = 2, 2, 3, 3
        segmaps_arr_small_cropped = segmaps_arr_small[top:-bottom, left:-right]

        observed = aug.augment_segmentation_maps([segmaps])[0]

        assert observed.shape == (8, 16)
        assert observed.arr.shape == (4, 6, 1)
        assert np.array_equal(observed.arr[..., 0], segmaps_arr_small_cropped)

    def test_crop_heatmaps_smaller_than_img_by_fixed_floats_with_ks(self):
        # crop smaller heatmaps, with keep_size=True
        # heatmap is (8, 12), image is (16, 32)
        # image is cropped by (0.25, 0.25, 0.25, 0.25)
        # expected image size: (8, 16) -> (16, 32) after resize
        # expected heatmap size: (4, 6) -> (8, 12) after resize
        aug = iaa.Crop(percent=(0.25, 0.25, 0.25, 0.25), keep_size=True)
        heatmaps_arr_small = np.zeros((8, 12), dtype=np.float32)
        heatmaps_arr_small[2:-2, 4:-4] = 1.0
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr_small, shape=(16, 32))
        top, bottom, left, right = 2, 2, 3, 3
        heatmaps_arr_small_cropped = \
            heatmaps_arr_small[top:-bottom, left:-right]

        observed = aug.augment_heatmaps([heatmaps])[0]

        assert observed.shape == (16, 32)
        assert observed.arr_0to1.shape == (8, 12, 1)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(
            observed.arr_0to1[..., 0],
            np.clip(
                ia.imresize_single_image(
                    heatmaps_arr_small_cropped,
                    (8, 12),
                    interpolation="cubic"),
                0,
                1.0
            )
        )

    def test_crop_segmaps_smaller_than_img_by_fixed_floats_with_ks(self):
        aug = iaa.Crop(percent=(0.25, 0.25, 0.25, 0.25), keep_size=True)
        segmaps_arr_small = np.zeros((8, 12), dtype=np.int32)
        segmaps_arr_small[2:-2, 4:-4] = 1
        segmaps = SegmentationMapsOnImage(segmaps_arr_small, shape=(16, 32))
        top, bottom, left, right = 2, 2, 3, 3
        segmaps_arr_small_cropped = segmaps_arr_small[top:-bottom, left:-right]

        observed = aug.augment_segmentation_maps([segmaps])[0]

        assert observed.shape == (16, 32)
        assert observed.arr.shape == (8, 12, 1)
        assert np.allclose(
            observed.arr[..., 0],
            ia.imresize_single_image(
                segmaps_arr_small_cropped,
                (8, 12),
                interpolation="nearest")
        )

    def test_crop_keypoints_by_fixed_floats_without_keep_size(self):
        aug = iaa.Crop(percent=(0.25, 0, 0.5, 0.1), keep_size=False)
        kps = [ia.Keypoint(x=12, y=10), ia.Keypoint(x=8, y=12)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(16, 20, 3))

        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]

        assert kpsoi_aug.shape == (4, 18, 3)
        assert len(kpsoi_aug.keypoints) == 2
        assert np.allclose(kpsoi_aug.keypoints[0].x, 12-2)
        assert np.allclose(kpsoi_aug.keypoints[0].y, 10-4)
        assert np.allclose(kpsoi_aug.keypoints[1].x, 8-2)
        assert np.allclose(kpsoi_aug.keypoints[1].y, 12-4)

    def test_crop_keypoints_by_fixed_floats_with_keep_size(self):
        aug = iaa.Crop(percent=(0.25, 0, 0.5, 0.1), keep_size=True)
        kps = [ia.Keypoint(x=12, y=10), ia.Keypoint(x=8, y=12)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(16, 20, 3))

        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]

        assert kpsoi_aug.shape == (16, 20, 3)
        assert len(kpsoi_aug.keypoints) == 2
        assert np.allclose(kpsoi_aug.keypoints[0].x, ((12-2)/18)*20)
        assert np.allclose(kpsoi_aug.keypoints[0].y, ((10-4)/4)*16)
        assert np.allclose(kpsoi_aug.keypoints[1].x, ((8-2)/18)*20)
        assert np.allclose(kpsoi_aug.keypoints[1].y, ((12-4)/4)*16)

    def test_crop_polygons_by_fixed_floats_without_keep_size(self):
        aug = iaa.Crop(percent=(0.2, 0, 0.5, 0.1), keep_size=False)
        polygons = [ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
                    ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])]
        cbaoi = ia.PolygonsOnImage(polygons, shape=(10, 10, 3))

        cbaoi_aug = aug.augment_polygons([cbaoi, cbaoi])

        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (3, 9, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(0-1, 0-2), (4-1, 0-2), (4-1, 4-2), (0-1, 4-2)]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(1-1, 1-2), (5-1, 1-2), (5-1, 5-2), (1-1, 5-2)]
            )

    def test_crop_polygons_by_fixed_floats_with_keep_size(self):
        aug = iaa.Crop(percent=(0.2, 0, 0.5, 0.1), keep_size=True)
        polygons = [ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
                    ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])]
        cbaoi = ia.PolygonsOnImage(polygons, shape=(10, 10, 3))

        cbaoi_aug = aug.augment_polygons([cbaoi, cbaoi])

        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (10, 10, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(10*(-1/9), 10*(-2/3)),
                 (10*(3/9), 10*(-2/3)),
                 (10*(3/9), 10*(2/3)),
                 (10*(-1/9), 10*(2/3))]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(10*(0/9), 10*(-1/3)),
                 (10*(4/9), 10*(-1/3)),
                 (10*(4/9), 10*(3/3)),
                 (10*(0/9), 10*(3/3))]
            )

    def test_crop_line_strings_by_fixed_floats_without_keep_size(self):
        aug = iaa.Crop(percent=(0.2, 0, 0.5, 0.1), keep_size=False)
        lss = [ia.LineString([(0, 0), (4, 0), (4, 4), (0, 4)]),
               ia.LineString([(1, 1), (5, 1), (5, 5), (1, 5)])]
        cbaoi = ia.LineStringsOnImage(lss, shape=(10, 10, 3))

        cbaoi_aug = aug.augment_line_strings([cbaoi, cbaoi])

        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (3, 9, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(0-1, 0-2), (4-1, 0-2), (4-1, 4-2), (0-1, 4-2)]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(1-1, 1-2), (5-1, 1-2), (5-1, 5-2), (1-1, 5-2)]
            )

    def test_crop_line_strings_by_fixed_floats_with_keep_size(self):
        aug = iaa.Crop(percent=(0.2, 0, 0.5, 0.1), keep_size=True)
        lss = [ia.LineString([(0, 0), (4, 0), (4, 4), (0, 4)]),
               ia.LineString([(1, 1), (5, 1), (5, 5), (1, 5)])]
        cbaoi = ia.LineStringsOnImage(lss, shape=(10, 10, 3))

        cbaoi_aug = aug.augment_line_strings([cbaoi, cbaoi])

        assert len(cbaoi_aug) == 2
        for cbaoi_aug_i in cbaoi_aug:
            assert cbaoi_aug_i.shape == (10, 10, 3)
            assert len(cbaoi_aug_i.items) == 2
            assert cbaoi_aug_i.items[0].coords_almost_equals(
                [(10*(-1/9), 10*(-2/3)),
                 (10*(3/9), 10*(-2/3)),
                 (10*(3/9), 10*(2/3)),
                 (10*(-1/9), 10*(2/3))]
            )
            assert cbaoi_aug_i.items[1].coords_almost_equals(
                [(10*(0/9), 10*(-1/3)),
                 (10*(4/9), 10*(-1/3)),
                 (10*(4/9), 10*(3/3)),
                 (10*(0/9), 10*(3/3))]
            )

    def test_crop_bounding_boxes_by_fixed_floats_without_keep_size(self):
        aug = iaa.Crop(percent=(0.2, 0, 0.5, 0.1), keep_size=False)
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=4, y2=4),
               ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(10, 10, 3))

        bbsoi_aug = aug.augment_bounding_boxes([bbsoi, bbsoi])

        assert len(bbsoi_aug) == 2
        for bbsoi_aug_i in bbsoi_aug:
            assert bbsoi_aug_i.shape == (3, 9, 3)
            assert len(bbsoi_aug_i.bounding_boxes) == 2
            assert bbsoi_aug_i.bounding_boxes[0].coords_almost_equals(
                [(0-1, 0-2), (4-1, 4-2)]
            )
            assert bbsoi_aug_i.bounding_boxes[1].coords_almost_equals(
                [(1-1, 2-2), (3-1, 4-2)]
            )

    def test_crop_bounding_boxes_by_fixed_floats_with_keep_size(self):
        aug = iaa.Crop(percent=(0.2, 0, 0.5, 0.1), keep_size=True)
        bbs = [ia.BoundingBox(x1=0, y1=0, x2=4, y2=4),
               ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]
        bbsoi = ia.BoundingBoxesOnImage(bbs, shape=(10, 10, 3))

        bbsoi_aug = aug.augment_bounding_boxes([bbsoi, bbsoi])

        assert len(bbsoi_aug) == 2
        for bbsoi_aug_i in bbsoi_aug:
            assert bbsoi_aug_i.shape == (10, 10, 3)
            assert len(bbsoi_aug_i.bounding_boxes) == 2
            assert bbsoi_aug_i.bounding_boxes[0].coords_almost_equals(
                [(10*((0-1)/9), 10*((0-2)/3)),
                 (10*((4-1)/9), 10*((4-2)/3))]
            )
            assert bbsoi_aug_i.bounding_boxes[1].coords_almost_equals(
                [(10*((1-1)/9), 10*((2-2)/3)),
                 (10*((3-1)/9), 10*((4-2)/3))]
            )

    def test_crop_by_tuple_of_floats_on_top_side_without_ks(self):
        aug = iaa.Crop(percent=((0, 0.1), 0, 0, 0), keep_size=False)
        image = np.zeros((40, 40), dtype=np.uint8)
        seen = [0, 0, 0, 0, 0]
        for _ in sm.xrange(500):
            observed = aug.augment_image(image)
            n_cropped = 40 - observed.shape[0]
            seen[n_cropped] += 1
        # note that we cant just check for 100-50 < x < 100+50 here. The first
        # and last value (0px and 4px) have half the probability of occuring
        # compared to the other values. E.g. 0px is cropped if sampled p
        # falls in range [0, 0.125). 1px is cropped if sampled p falls in
        # range [0.125, 0.375].
        assert np.all([v > 30 for v in seen])

    def test_crop_by_tuple_of_floats_on_right_side_without_ks(self):
        aug = iaa.Crop(percent=(0, (0, 0.1), 0, 0), keep_size=False)
        image = np.zeros((40, 40), dtype=np.uint8) + 255
        seen = [0, 0, 0, 0, 0]
        for _ in sm.xrange(500):
            observed = aug.augment_image(image)
            n_cropped = 40 - observed.shape[1]
            seen[n_cropped] += 1
        assert np.all([v > 30 for v in seen])

    def test_crop_by_list_of_floats_on_top_side_without_ks(self):
        aug = iaa.Crop(percent=([0.0, 0.1], 0, 0, 0), keep_size=False)
        image = np.zeros((40, 40), dtype=np.uint8) + 255
        seen = [0, 0, 0, 0, 0]
        for _ in sm.xrange(500):
            observed = aug.augment_image(image)
            n_cropped = 40 - observed.shape[0]
            seen[n_cropped] += 1
        assert 250 - 50 < seen[0] < 250 + 50
        assert seen[1] == 0
        assert seen[2] == 0
        assert seen[3] == 0
        assert 250 - 50 < seen[4] < 250 + 50

    def test_crop_by_list_of_floats_on_right_side_without_ks(self):
        aug = iaa.Crop(percent=(0, [0.0, 0.1], 0, 0), keep_size=False)
        image = np.zeros((40, 40), dtype=np.uint8) + 255
        seen = [0, 0, 0, 0, 0]
        for _ in sm.xrange(500):
            observed = aug.augment_image(image)
            n_cropped = 40 - observed.shape[1]
            seen[n_cropped] += 1
        assert 250 - 50 < seen[0] < 250 + 50
        assert seen[1] == 0
        assert seen[2] == 0
        assert seen[3] == 0
        assert 250 - 50 < seen[4] < 250 + 50

    @classmethod
    def _test_crop_empty_cba(cls, augf_name, cbaoi):
        aug = iaa.Crop(px=(1, 2, 3, 4), keep_size=False)

        cbaoi_aug = getattr(aug, augf_name)(cbaoi)

        expected = cbaoi.deepcopy()
        expected.shape = tuple(
            [expected.shape[0]-1-3, expected.shape[1]-2-4]
            + list(expected.shape[2:]))
        assert_cbaois_equal(cbaoi_aug, expected)

    def test_pad_empty_keypoints(self):
        cbaoi = ia.KeypointsOnImage([], shape=(12, 14, 3))
        self._test_crop_empty_cba("augment_keypoints", cbaoi)

    def test_pad_empty_polygons(self):
        cbaoi = ia.PolygonsOnImage([], shape=(12, 14, 3))
        self._test_crop_empty_cba("augment_polygons", cbaoi)

    def test_pad_empty_line_strings(self):
        cbaoi = ia.LineStringsOnImage([], shape=(12, 14, 3))
        self._test_crop_empty_cba("augment_line_strings", cbaoi)

    def test_pad_empty_bounding_boxes(self):
        cbaoi = ia.BoundingBoxesOnImage([], shape=(12, 14, 3))
        self._test_crop_empty_cba("augment_bounding_boxes", cbaoi)

    def test_zero_sized_axes_no_keep_size(self):
        # we also use height/width 2 here, because a height/width of 1 is
        # actually not changed due to prevent_zero_size
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (0, 2),
            (2, 0),
            (0, 2, 0),
            (2, 0, 0),
            (0, 2, 1),
            (2, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Crop(px=1, keep_size=False)

                with warnings.catch_warnings(record=True) as caught_warnings:
                    image_aug = aug(image=image)

                # we don't check the number of warnings here as it varies by
                # shape
                for warning in caught_warnings:
                    assert (
                        "crop amounts in CropAndPad"
                        in str(warning.message)
                    )

                expected_height = 0 if shape[0] == 0 else 1
                expected_width = 0 if shape[1] == 0 else 1
                expected_shape = tuple([expected_height, expected_width]
                                       + list(shape[2:]))
                assert image_aug.shape == expected_shape

    def test_zero_sized_axes_keep_size(self):
        # we also use height/width 2 here, because a height/width of 1 is
        # actually not changed due to prevent_zero_size
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (0, 2),
            (2, 0),
            (0, 2, 0),
            (2, 0, 0),
            (0, 2, 1),
            (2, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Crop(px=1, keep_size=True)

                with warnings.catch_warnings(record=True) as caught_warnings:
                    image_aug = aug(image=image)

                # we don't check the number of warnings here as it varies by
                # shape
                for warning in caught_warnings:
                    assert (
                        "crop amounts in CropAndPad"
                        in str(warning.message)
                    )

                assert image_aug.shape == image.shape

    def test_other_dtypes_bool(self):
        aug = iaa.Crop(px=(1, 0, 0, 0), keep_size=False)
        mask = np.zeros((2, 3), dtype=bool)
        mask[0, 1] = True

        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.name == image.dtype.name
        assert image_aug.shape == (2, 3)
        assert np.all(image_aug[~mask] == 0)
        assert np.all(image_aug[mask] == 1)

    def test_other_dtypes_uint_int(self):
        aug = iaa.Crop(px=(1, 0, 0, 0), keep_size=False)
        mask = np.zeros((2, 3), dtype=bool)
        mask[0, 1] = True

        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64"]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                if np.dtype(dtype).kind == "i":
                    values = [
                        1, 5, 10, 100, int(0.1 * max_value),
                        int(0.2 * max_value), int(0.5 * max_value),
                        max_value - 100, max_value]
                    values = values + [(-1) * value for value in values]
                else:
                    values = [
                        1, 5, 10, 100, int(center_value), int(0.1 * max_value),
                        int(0.2 * max_value), int(0.5 * max_value),
                        max_value - 100, max_value]

                for value in values:
                    image = np.zeros((3, 3), dtype=dtype)
                    image[1, 1] = value
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype.name == dtype
                    assert image_aug.shape == (2, 3)
                    assert np.all(image_aug[~mask] == 0)
                    assert np.all(image_aug[mask] == value)

    def test_other_dtypes_float(self):
        aug = iaa.Crop(px=(1, 0, 0, 0), keep_size=False)
        mask = np.zeros((2, 3), dtype=bool)
        mask[0, 1] = True

        dtypes = ["float16", "float32", "float64", "float128"]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                def _isclose(a, b):
                    atol = 1e-4 if dtype == np.float16 else 1e-8
                    return np.isclose(a, b, atol=atol, rtol=0)

                isize = np.dtype(dtype).itemsize
                values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1),
                          1000 ** (isize - 1)]
                values = values + [(-1) * value for value in values]
                values = values + [min_value, max_value]
                for value in values:
                    image = np.zeros((3, 3), dtype=dtype)
                    image[1, 1] = value
                    image_aug = aug.augment_image(image)
                    assert image_aug.dtype == np.dtype(dtype)
                    assert image_aug.shape == (2, 3)
                    assert np.all(_isclose(image_aug[~mask], 0))
                    assert np.all(_isclose(image_aug[mask],
                                           np.float128(value)))


class TestPadToFixedSize(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_image2d_that_needs_to_be_padded_on_both_sides(self):
        aug = iaa.PadToFixedSize(height=5, width=5)
        image = np.uint8([[255]])

        observed = aug.augment_image(image)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (5, 5)

    def test_image3d_that_needs_to_be_padded_on_both_sides(self):
        aug = iaa.PadToFixedSize(height=5, width=5)
        image3d = np.atleast_3d(np.uint8([[255]]))

        observed = aug.augment_image(image3d)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (5, 5, 1)

    def test_image3d_rgb_that_needs_to_be_padded_on_both_sides(self):
        aug = iaa.PadToFixedSize(height=5, width=5)
        image3d_rgb = np.tile(
            np.atleast_3d(np.uint8([[255]])),
            (1, 1, 3)
        )

        observed = aug.augment_image(image3d_rgb)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (5, 5, 3)

    # why does this exist when there is already a test for other float dtypes?
    def test_image2d_with_other_dtypes(self):
        aug = iaa.PadToFixedSize(height=5, width=5)
        image = np.uint8([[255]])

        for dtype in ["float32", "float64", "int32"]:
            with self.subTest(dtype=dtype):
                observed = aug.augment_image(image.astype(dtype))

                assert observed.dtype.name == dtype
                assert observed.shape == (5, 5)

    def test_image_with_height_being_too_small(self):
        aug = iaa.PadToFixedSize(height=5, width=5)
        image = np.zeros((1, 5, 3), dtype=np.uint8)

        observed = aug.augment_image(image)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (5, 5, 3)

    def test_image_with_width_being_too_small(self):
        aug = iaa.PadToFixedSize(height=5, width=5)
        image = np.zeros((5, 1, 3), dtype=np.uint8)

        observed = aug.augment_image(image)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (5, 5, 3)

    def test_image_fullfills_exactly_min_shape(self):
        # change no side when all sides have exactly desired size
        aug = iaa.PadToFixedSize(height=5, width=5)
        img5x5 = np.zeros((5, 5, 3), dtype=np.uint8)
        img5x5[2, 2, :] = 255

        observed = aug.augment_image(img5x5)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (5, 5, 3)
        assert np.array_equal(observed, img5x5)

    def test_image_that_is_larger_than_min_shape(self):
        # change no side when all sides have larger than desired size
        aug = iaa.PadToFixedSize(height=5, width=5)
        img6x6 = np.zeros((6, 6, 3), dtype=np.uint8)
        img6x6[3, 3, :] = 255

        observed = aug.augment_image(img6x6)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (6, 6, 3)
        assert np.array_equal(observed, img6x6)

    def test_image_pad_mode(self):
        # make sure that pad mode is recognized
        aug = iaa.PadToFixedSize(height=4, width=4, pad_mode="edge")
        aug.position = (iap.Deterministic(0.5), iap.Deterministic(0.5))
        img2x2 = np.uint8([
            [50, 100],
            [150, 200]
        ])

        observed = aug.augment_image(img2x2)

        expected = np.uint8([
            [50, 50, 100, 100],
            [50, 50, 100, 100],
            [150, 150, 200, 200],
            [150, 150, 200, 200]
        ])
        assert observed.dtype.name == "uint8"
        assert observed.shape == (4, 4)
        assert np.array_equal(observed, expected)

    def test_image_pad_at_left_top(self):
        # explicit non-center position test
        aug = iaa.PadToFixedSize(
            height=3, width=3, pad_mode="constant", pad_cval=128,
            position="left-top")
        img1x1 = np.uint8([[255]])
        observed = aug.augment_image(img1x1)
        expected = np.uint8([
            [128, 128, 128],
            [128, 128, 128],
            [128, 128, 255]
        ])
        assert observed.dtype.name == "uint8"
        assert observed.shape == (3, 3)
        assert np.array_equal(observed, expected)

    def test_image_pad_at_right_bottom(self):
        aug = iaa.PadToFixedSize(
            height=3, width=3, pad_mode="constant", pad_cval=128,
            position="right-bottom")
        img1x1 = np.uint8([[255]])

        observed = aug.augment_image(img1x1)

        expected = np.uint8([
            [255, 128, 128],
            [128, 128, 128],
            [128, 128, 128]
        ])
        assert observed.dtype.name == "uint8"
        assert observed.shape == (3, 3)
        assert np.array_equal(observed, expected)

    def test_image_pad_at_bottom_center_given_as_tuple_of_floats(self):
        aug = iaa.PadToFixedSize(
            height=3, width=3, pad_mode="constant", pad_cval=128,
            position=(0.5, 1.0))
        img1x1 = np.uint8([[255]])

        observed = aug.augment_image(img1x1)

        expected = np.uint8([
            [128, 255, 128],
            [128, 128, 128],
            [128, 128, 128]
        ])
        assert observed.dtype.name == "uint8"
        assert observed.shape == (3, 3)
        assert np.array_equal(observed, expected)

    def test_keypoints__image_already_fullfills_min_shape(self):
        # keypoint test with shape not being changed
        aug = iaa.PadToFixedSize(
            height=3, width=3, pad_mode="edge", position="center")
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))

        observed = aug.augment_keypoints(kpsoi)

        expected = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_keypoints_pad_at_center(self):
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="center")
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))

        observed = aug.augment_keypoints(kpsoi)

        expected = ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_keypoints_pad_at_left_top(self):
        # keypoint test with explicit non-center position
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="left-top")
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))

        observed = aug.augment_keypoints(kpsoi)

        expected = ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_keypoints_pad_at_right_bottom(self):
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="right-bottom")
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))

        observed = aug.augment_keypoints(kpsoi)

        expected = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_keypoints_empty(self):
        aug = iaa.PadToFixedSize(height=5, width=6)
        kpsoi = ia.KeypointsOnImage([], shape=(3, 3))

        observed = aug.augment_keypoints(kpsoi)

        expected = ia.KeypointsOnImage([], shape=(5, 6))
        assert_cbaois_equal(observed, expected)

    def test_polygons__image_already_fullfills_min_shape(self):
        # polygons test with shape not being changed
        aug = iaa.PadToFixedSize(
            height=3, width=3, pad_mode="edge", position="center")
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (3, 0), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_polygons(psoi)

        expected = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (3, 0), (3, 3)])
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_polygons_pad_at_center(self):
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="center")
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (3, 0), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_polygons(psoi)

        expected = ia.PolygonsOnImage([
            ia.Polygon([(1+0, 1+0), (1+3, 1+0), (1+3, 1+3)])
        ], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_polygons_pad_at_left_top(self):
        # polygon test with explicit non-center position
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="left-top")
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (3, 0), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_polygons(psoi)

        expected = ia.PolygonsOnImage([
            ia.Polygon([(1+0, 1+0), (1+3, 1+0), (1+3, 1+3)])
        ], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_polygons_pad_at_right_bottom(self):
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="right-bottom")
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (3, 0), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_polygons(psoi)

        expected = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (3, 0), (3, 3)])
        ], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_polygons_empty(self):
        aug = iaa.PadToFixedSize(height=5, width=6)
        psoi = ia.PolygonsOnImage([], shape=(3, 3))

        observed = aug.augment_polygons(psoi)

        expected = ia.PolygonsOnImage([], shape=(5, 6))
        assert_cbaois_equal(observed, expected)

    def test_line_strings__image_already_fullfills_min_shape(self):
        # line string test with shape not being changed
        aug = iaa.PadToFixedSize(
            height=3, width=3, pad_mode="edge", position="center")
        cbaoi = ia.LineStringsOnImage([
            ia.LineString([(0, 0), (3, 0), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_line_strings(cbaoi)

        expected = ia.LineStringsOnImage([
            ia.LineString([(0, 0), (3, 0), (3, 3)])
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_line_strings_pad_at_center(self):
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="center")
        cbaoi = ia.LineStringsOnImage([
            ia.LineString([(0, 0), (3, 0), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_line_strings(cbaoi)

        expected = ia.LineStringsOnImage([
            ia.LineString([(1+0, 1+0), (1+3, 1+0), (1+3, 1+3)])
        ], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_line_strings_pad_at_left_top(self):
        # line string test with explicit non-center position
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="left-top")
        cbaoi = ia.LineStringsOnImage([
            ia.LineString([(0, 0), (3, 0), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_line_strings(cbaoi)

        expected = ia.LineStringsOnImage([
            ia.LineString([(1+0, 1+0), (1+3, 1+0), (1+3, 1+3)])
        ], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_line_strings_pad_at_right_bottom(self):
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="right-bottom")
        cbaoi = ia.LineStringsOnImage([
            ia.LineString([(0, 0), (3, 0), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_line_strings(cbaoi)

        expected = ia.LineStringsOnImage([
            ia.LineString([(0, 0), (3, 0), (3, 3)])
        ], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_line_strings_empty(self):
        aug = iaa.PadToFixedSize(height=5, width=6)
        cbaoi = ia.LineStringsOnImage([], shape=(3, 3))

        observed = aug.augment_line_strings(cbaoi)

        expected = ia.LineStringsOnImage([], shape=(5, 6))
        assert_cbaois_equal(observed, expected)

    def test_bounding_boxes__image_already_fullfills_min_shape(self):
        # bounding boxes test with shape not being changed
        aug = iaa.PadToFixedSize(
            height=3, width=3, pad_mode="edge", position="center")
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3),
        ], shape=(3, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        expected = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3),
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_bounding_boxes_pad_at_center(self):
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="center")
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3),
        ], shape=(3, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        expected = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=1+0, y1=1+1, x2=1+2, y2=1+3),
        ], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_bounding_boxes_pad_at_left_top(self):
        # bounding boxes test with explicit non-center position
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="left-top")
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3),
        ], shape=(3, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        expected = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=1+0, y1=1+1, x2=1+2, y2=1+3),
        ], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_bounding_boxes_pad_at_right_bottom(self):
        aug = iaa.PadToFixedSize(
            height=4, width=4, pad_mode="edge", position="right-bottom")
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3),
        ], shape=(3, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        expected = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3),
        ], shape=(4, 4))
        assert_cbaois_equal(observed, expected)

    def test_bounding_boxes_empty(self):
        aug = iaa.PadToFixedSize(height=5, width=6)
        bbsoi = ia.BoundingBoxesOnImage([], shape=(3, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        expected = ia.BoundingBoxesOnImage([], shape=(5, 6))
        assert_cbaois_equal(observed, expected)

    def test_heatmaps__pad_mode_should_be_ignored(self):
        # basic heatmaps test
        # pad_mode should be ignored for heatmaps
        aug = iaa.PadToFixedSize(
            height=3, width=3, pad_mode="edge", position="center")
        heatmaps_arr = np.zeros((1, 1, 1), dtype=np.float32) + 1.0
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(1, 1, 3))

        observed = aug.augment_heatmaps([heatmaps])[0]

        expected = np.float32([
            [0, 0, 0],
            [0, 1.0, 0],
            [0, 0, 0]
        ])
        expected = expected[..., np.newaxis]
        assert observed.shape == (3, 3, 3)
        assert np.allclose(observed.arr_0to1, expected)

    def test_heatmaps_smaller_than_image__pad_mode_should_be_ignored(self):
        # heatmaps with size unequal to image
        # pad_mode should be ignored for heatmaps
        aug = iaa.PadToFixedSize(
            height=32, width=32, pad_mode="edge", position="left-top")
        heatmaps_arr = np.zeros((15, 15, 1), dtype=np.float32) + 1.0
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(30, 30, 3))

        observed = aug.augment_heatmaps([heatmaps])[0]

        expected = np.zeros((16, 16, 1), dtype=np.float32) + 1.0
        expected[:, 0, 0] = 0.0
        expected[0, :, 0] = 0.0
        assert observed.shape == (32, 32, 3)
        assert np.allclose(observed.arr_0to1, expected)

    def test_segmaps__pad_mode_should_be_ignored(self):
        # basic segmaps test
        # pad_mode should be ignored for segmaps
        aug = iaa.PadToFixedSize(
            height=3, width=3, pad_mode="edge", position="center")
        segmaps_arr = np.ones((1, 1, 1), dtype=np.int32)
        segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(1, 1, 3))

        observed = aug.augment_segmentation_maps([segmaps])[0]

        expected = np.int32([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        expected = expected[..., np.newaxis]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.arr, expected)

    def test_segmaps_smaller_than_image__pad_mode_should_be_ignored(self):
        # segmaps with size unequal to image
        # pad_mode should be ignored for segmaps
        aug = iaa.PadToFixedSize(
            height=32, width=32, pad_mode="edge", position="left-top")
        segmaps_arr = np.ones((15, 15, 1), dtype=np.int32)
        segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(30, 30, 3))

        observed = aug.augment_segmentation_maps([segmaps])[0]

        expected = np.ones((16, 16, 1), dtype=np.int32)
        expected[:, 0, 0] = 0
        expected[0, :, 0] = 0
        assert observed.shape == (32, 32, 3)
        assert np.array_equal(observed.arr, expected)

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
                aug = iaa.PadToFixedSize(height=1, width=1)

                image_aug = aug(image=image)

                expected_height = 1
                expected_width = 1
                expected_shape = tuple([expected_height, expected_width]
                                       + list(shape[2:]))
                assert image_aug.shape == expected_shape

    def test_other_dtypes_bool(self):
        aug = iaa.PadToFixedSize(height=4, width=3, position="center-top")
        mask = np.zeros((4, 3), dtype=bool)
        mask[2, 1] = True
        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == image.dtype.name
        assert image_aug.shape == (4, 3)
        assert np.all(image_aug[~mask] == 0)
        assert np.all(image_aug[mask] == 1)

    def test_other_dtypes_uint_int(self):
        aug = iaa.PadToFixedSize(height=4, width=3, position="center-top")
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64"]

        mask = np.zeros((4, 3), dtype=bool)
        mask[2, 1] = True

        for dtype in dtypes:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            if np.dtype(dtype).kind == "i":
                values = [
                    1, 5, 10, 100, int(0.1 * max_value),
                    int(0.2 * max_value), int(0.5 * max_value),
                    max_value - 100, max_value]
                values = values + [(-1) * value for value in values]
            else:
                values = [
                    1, 5, 10, 100, int(center_value), int(0.1 * max_value),
                    int(0.2 * max_value), int(0.5 * max_value),
                    max_value - 100, max_value]

            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((3, 3), dtype=dtype)
                    image[1, 1] = value

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert image_aug.shape == (4, 3)
                    assert np.all(image_aug[~mask] == 0)
                    assert np.all(image_aug[mask] == value)

    def test_other_dtypes_float(self):
        aug = iaa.PadToFixedSize(height=4, width=3, position="center-top")
        dtypes = ["float16", "float32", "float64", "float128"]

        mask = np.zeros((4, 3), dtype=bool)
        mask[2, 1] = True

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
                    assert image_aug.shape == (4, 3)
                    assert np.all(_isclose(image_aug[~mask], 0))
                    assert np.all(_isclose(image_aug[mask],
                                           np.float128(value)))


class TestCropToFixedSize(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_image2d_that_needs_to_be_cropped_on_both_sides(self):
        aug = iaa.CropToFixedSize(height=1, width=1)
        image = np.uint8([
            [128, 129, 130],
            [131, 132, 133],
            [134, 135, 136]
        ])

        observed = aug.augment_image(image)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (1, 1)

    def test_image3d_that_needs_to_be_cropped_on_both_sides(self):
        aug = iaa.CropToFixedSize(height=1, width=1)
        image = np.uint8([
            [128, 129, 130],
            [131, 132, 133],
            [134, 135, 136]
        ])
        image3d = np.atleast_3d(image)

        observed = aug.augment_image(image3d)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (1, 1, 1)

    def test_image3d_rgb_that_needs_to_be_cropped_on_both_sides(self):
        aug = iaa.CropToFixedSize(height=1, width=1)
        image = np.uint8([
            [128, 129, 130],
            [131, 132, 133],
            [134, 135, 136]
        ])
        image3d_rgb = np.tile(
            np.atleast_3d(image),
            (1, 1, 3)
        )

        observed = aug.augment_image(image3d_rgb)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (1, 1, 3)

    def test_image2d_with_other_dtypes(self):
        aug = iaa.CropToFixedSize(height=1, width=1)
        image = np.uint8([
            [128, 129, 130],
            [131, 132, 133],
            [134, 135, 136]
        ])

        for dtype in ["float32", "float64", "int32"]:
            with self.subTest(dtype=dtype):
                observed = aug.augment_image(image.astype(dtype))

                assert observed.dtype.name == dtype
                assert observed.shape == (1, 1)

    def test_image_with_height_being_too_large(self):
        # change only one side when other side has already desired size
        aug = iaa.CropToFixedSize(height=1, width=5)
        image = np.zeros((3, 5, 3), dtype=np.uint8)

        observed = aug.augment_image(image)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (1, 5, 3)

    def test_image_with_width_being_too_large(self):
        aug = iaa.CropToFixedSize(height=5, width=1)
        image = np.zeros((5, 3, 3), dtype=np.uint8)

        observed = aug.augment_image(image)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (5, 1, 3)

    def test_image_fullfills_exactly_max_shape(self):
        # change no side when all sides have exactly desired size
        aug = iaa.CropToFixedSize(height=5, width=5)
        img5x5 = np.zeros((5, 5, 3), dtype=np.uint8)
        img5x5[2, 2, :] = 255

        observed = aug.augment_image(img5x5)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (5, 5, 3)
        assert np.array_equal(observed, img5x5)

    def test_image_that_is_smaller_than_max_shape(self):
        # change no side when all sides have smaller than desired size
        aug = iaa.CropToFixedSize(height=5, width=5)
        img4x4 = np.zeros((4, 4, 3), dtype=np.uint8)
        img4x4[2, 2, :] = 255

        observed = aug.augment_image(img4x4)

        assert observed.dtype.name == "uint8"
        assert observed.shape == (4, 4, 3)
        assert np.array_equal(observed, img4x4)

    def test_image_crop_at_left_top(self):
        # explicit non-center position test
        aug = iaa.CropToFixedSize(height=3, width=3, position="left-top")
        img5x5 = np.arange(25, dtype=np.uint8).reshape((5, 5))

        observed = aug.augment_image(img5x5)

        expected = img5x5[2:, 2:]
        assert observed.dtype.name == "uint8"
        assert observed.shape == (3, 3)
        assert np.array_equal(observed, expected)

    def test_image_crop_at_right_bottom(self):
        aug = iaa.CropToFixedSize(height=3, width=3, position="right-bottom")
        img5x5 = np.arange(25, dtype=np.uint8).reshape((5, 5))

        observed = aug.augment_image(img5x5)

        expected = img5x5[:3, :3]
        assert observed.dtype.name == "uint8"
        assert observed.shape == (3, 3)
        assert np.array_equal(observed, expected)

    def test_image_crop_at_bottom_center_given_as_tuple_of_floats(self):
        aug = iaa.CropToFixedSize(height=3, width=3, position=(0.5, 1.0))
        img5x5 = np.arange(25, dtype=np.uint8).reshape((5, 5))

        observed = aug.augment_image(img5x5)

        expected = img5x5[:3, 1:4]
        assert observed.dtype.name == "uint8"
        assert observed.shape == (3, 3)
        assert np.array_equal(observed, expected)

    def test_keypoints__image_already_fullfills_max_shape(self):
        # keypoint test with shape not being changed
        aug = iaa.CropToFixedSize(height=3, width=3, position="center")
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))

        observed = aug.augment_keypoints(kpsoi)

        expected = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_keypoints_crop_at_center(self):
        # basic keypoint test
        aug = iaa.CropToFixedSize(height=1, width=1, position="center")
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))

        observed = aug.augment_keypoints(kpsoi)

        expected = ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(1, 1))
        assert_cbaois_equal(observed, expected)

    def test_keypoints_crop_at_left_top(self):
        # keypoint test with explicit non-center position
        aug = iaa.CropToFixedSize(height=3, width=3, position="left-top")
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(5, 5))

        observed = aug.augment_keypoints(kpsoi)

        expected = ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_keypoints_crop_at_right_bottom(self):
        aug = iaa.CropToFixedSize(height=3, width=3, position="right-bottom")
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(5, 5))

        observed = aug.augment_keypoints(kpsoi)

        expected = ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_keypoints_empty(self):
        aug = iaa.CropToFixedSize(height=3, width=3, position="center")
        kpsoi = ia.KeypointsOnImage([], shape=(5, 4))

        observed = aug.augment_keypoints(kpsoi)

        expected = ia.KeypointsOnImage([], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_polygons__image_already_fullfills_max_shape(self):
        # polygons test with shape not being changed
        aug = iaa.CropToFixedSize(height=3, width=3, position="center")
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(1, 1), (3, 1), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_polygons(psoi)

        expected = ia.PolygonsOnImage([
            ia.Polygon([(1, 1), (3, 1), (3, 3)])
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_polygons_crop_at_center(self):
        # basic polygons test
        aug = iaa.CropToFixedSize(height=1, width=1, position="center")
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(1, 1), (3, 1), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_polygons(psoi)

        expected = ia.PolygonsOnImage([
            ia.Polygon([(1-1, 1-1), (3-1, 1-1), (3-1, 3-1)])
        ], shape=(1, 1))
        assert_cbaois_equal(observed, expected)

    def test_polygons_crop_at_left_top(self):
        # polygons test with explicit non-center position
        aug = iaa.CropToFixedSize(height=3, width=3, position="left-top")
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(1, 1), (3, 1), (3, 3)])
        ], shape=(5, 5))

        observed = aug.augment_polygons(psoi)

        expected = ia.PolygonsOnImage([
            ia.Polygon([(1-2, 1-2), (3-2, 1-2), (3-2, 3-2)])
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_polygons_crop_at_right_bottom(self):
        aug = iaa.CropToFixedSize(height=3, width=3, position="right-bottom")
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(1, 1), (3, 1), (3, 3)])
        ], shape=(5, 5))

        observed = aug.augment_polygons(psoi)

        expected = ia.PolygonsOnImage([
            ia.Polygon([(1, 1), (3, 1), (3, 3)])
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_polygons_empty(self):
        aug = iaa.CropToFixedSize(height=3, width=3, position="center")
        psoi = ia.PolygonsOnImage([], shape=(5, 4))

        observed = aug.augment_polygons(psoi)

        expected = ia.PolygonsOnImage([], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_line_strings__image_already_fullfills_max_shape(self):
        # line strings test with shape not being changed
        aug = iaa.CropToFixedSize(height=3, width=3, position="center")
        cbaoi = ia.LineStringsOnImage([
            ia.LineString([(1, 1), (3, 1), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_line_strings(cbaoi)

        expected = ia.LineStringsOnImage([
            ia.LineString([(1, 1), (3, 1), (3, 3)])
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_line_strings_crop_at_center(self):
        # basic line strings test
        aug = iaa.CropToFixedSize(height=1, width=1, position="center")
        cbaoi = ia.LineStringsOnImage([
            ia.LineString([(1, 1), (3, 1), (3, 3)])
        ], shape=(3, 3))

        observed = aug.augment_line_strings(cbaoi)

        expected = ia.LineStringsOnImage([
            ia.LineString([(1-1, 1-1), (3-1, 1-1), (3-1, 3-1)])
        ], shape=(1, 1))
        assert_cbaois_equal(observed, expected)

    def test_line_strings_crop_at_left_top(self):
        # polygons test with explicit non-center position
        aug = iaa.CropToFixedSize(height=3, width=3, position="left-top")
        cbaoi = ia.LineStringsOnImage([
            ia.LineString([(1, 1), (3, 1), (3, 3)])
        ], shape=(5, 5))

        observed = aug.augment_line_strings(cbaoi)

        expected = ia.LineStringsOnImage([
            ia.LineString([(1-2, 1-2), (3-2, 1-2), (3-2, 3-2)])
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_line_strings_crop_at_right_bottom(self):
        aug = iaa.CropToFixedSize(height=3, width=3, position="right-bottom")
        cbaoi = ia.LineStringsOnImage([
            ia.LineString([(1, 1), (3, 1), (3, 3)])
        ], shape=(5, 5))

        observed = aug.augment_line_strings(cbaoi)

        expected = ia.LineStringsOnImage([
            ia.LineString([(1, 1), (3, 1), (3, 3)])
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_line_strings_empty(self):
        aug = iaa.CropToFixedSize(height=3, width=3, position="center")
        cbaoi = ia.LineStringsOnImage([], shape=(5, 4))

        observed = aug.augment_line_strings(cbaoi)

        expected = ia.LineStringsOnImage([], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_bounding_boxes__image_already_fullfills_max_shape(self):
        # bounding boxes test with shape not being changed
        aug = iaa.CropToFixedSize(height=3, width=3, position="center")
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)
        ], shape=(3, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        expected = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_bounding_boxes_crop_at_center(self):
        # basic bounding boxes test
        aug = iaa.CropToFixedSize(height=1, width=1, position="center")
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)
        ], shape=(3, 3))

        observed = aug.augment_bounding_boxes(bbsoi)

        expected = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0-1, y1=1-1, x2=2-1, y2=3-1)
        ], shape=(1, 1))
        assert_cbaois_equal(observed, expected)

    def test_bounding_boxes_crop_at_left_top(self):
        # bounding boxes test with explicit non-center position
        aug = iaa.CropToFixedSize(height=3, width=3, position="left-top")
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)
        ], shape=(5, 5))

        observed = aug.augment_bounding_boxes(bbsoi)

        expected = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0-2, y1=1-2, x2=2-2, y2=3-2)
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_bounding_boxes_crop_at_right_bottom(self):
        aug = iaa.CropToFixedSize(height=3, width=3, position="right-bottom")
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)
        ], shape=(5, 5))

        observed = aug.augment_bounding_boxes(bbsoi)

        expected = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=2, y2=3)
        ], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_bounding_boxes_empty(self):
        aug = iaa.CropToFixedSize(height=3, width=3, position="center")
        bbsoi = ia.BoundingBoxesOnImage([], shape=(5, 4))

        observed = aug.augment_bounding_boxes(bbsoi)

        expected = ia.BoundingBoxesOnImage([], shape=(3, 3))
        assert_cbaois_equal(observed, expected)

    def test_heatmaps(self):
        # basic heatmaps test
        aug = iaa.CropToFixedSize(height=3, width=3, position="center")
        heatmaps_arr = np.zeros((5, 5, 1), dtype=np.float32) + 1.0
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(5, 5, 3))

        observed = aug.augment_heatmaps([heatmaps])[0]

        expected = np.zeros((3, 3, 1), dtype=np.float32) + 1.0
        assert observed.shape == (3, 3, 3)
        assert np.allclose(observed.arr_0to1, expected)

    def test_heatmaps_crop_at_left_top(self):
        # heatmaps, crop at non-center position
        aug = iaa.CropToFixedSize(height=3, width=3, position="left-top")
        heatmaps_arr = np.linspace(
            0.0, 1.0, 5 * 5 * 1).reshape((5, 5, 1)).astype(np.float32)
        heatmaps_oi = ia.HeatmapsOnImage(heatmaps_arr, shape=(5, 5, 3))

        observed = aug.augment_heatmaps([heatmaps_oi])[0]

        expected = heatmaps_arr[2:, 2:, :]
        assert observed.shape == (3, 3, 3)
        assert np.allclose(observed.arr_0to1, expected)

    def test_heatmaps_crop_at_right_bottom(self):
        # heatmaps, crop at non-center position
        aug = iaa.CropToFixedSize(height=3, width=3, position="right-bottom")
        heatmaps_arr = np.linspace(
            0.0, 1.0, 5 * 5 * 1).reshape((5, 5, 1)).astype(np.float32)
        heatmaps_oi = ia.HeatmapsOnImage(heatmaps_arr, shape=(5, 5, 3))

        observed = aug.augment_heatmaps([heatmaps_oi])[0]

        expected = heatmaps_arr[:3, :3, :]
        assert observed.shape == (3, 3, 3)
        assert np.allclose(observed.arr_0to1, expected)

    def test_heatmaps_smaller_than_image(self):
        # heatmaps with size unequal to image
        aug = iaa.CropToFixedSize(height=32, width=32, position="left-top")
        heatmaps_arr = np.zeros((17, 17, 1), dtype=np.float32) + 1.0
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(34, 34, 3))

        observed = aug.augment_heatmaps([heatmaps])[0]

        expected = np.zeros((16, 16, 1), dtype=np.float32) + 1.0
        assert observed.shape == (32, 32, 3)
        assert np.allclose(observed.arr_0to1, expected)

    def test_segmaps_crop_at_center(self):
        # basic segmaps test
        aug = iaa.CropToFixedSize(height=3, width=3, position="center")
        segmaps_arr = np.ones((5, 5, 1), dtype=np.int32)
        segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(5, 5, 3))

        observed = aug.augment_segmentation_maps([segmaps])[0]

        expected = np.ones((3, 3, 1), dtype=np.int32)
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.arr, expected)

    def test_segmaps_crop_at_left_top(self):
        # segmaps, crop at non-center position
        aug = iaa.CropToFixedSize(height=3, width=3, position="left-top")
        segmaps_arr = np.arange(5*5).reshape((5, 5, 1)).astype(np.int32)
        segmaps_oi = SegmentationMapsOnImage(segmaps_arr, shape=(5, 5, 3))

        observed = aug.augment_segmentation_maps([segmaps_oi])[0]

        expected = segmaps_arr[2:, 2:, :]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.arr, expected)

    def test_segmaps_crop_at_right_bottom(self):
        # segmaps, crop at non-center position
        aug = iaa.CropToFixedSize(height=3, width=3, position="right-bottom")
        segmaps_arr = np.arange(5*5).reshape((5, 5, 1)).astype(np.int32)
        segmaps_oi = SegmentationMapsOnImage(segmaps_arr, shape=(5, 5, 3))

        observed = aug.augment_segmentation_maps([segmaps_oi])[0]

        expected = segmaps_arr[:3, :3, :]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.arr, expected)

    def test_segmaps_smaller_than_image(self):
        # segmaps with size unequal to image
        aug = iaa.CropToFixedSize(height=32, width=32, position="left-top")
        segmaps_arr = np.ones((17, 17, 1), dtype=np.int32)
        segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(34, 34, 3))

        observed = aug.augment_segmentation_maps([segmaps])[0]

        expected = np.ones((16, 16, 1), dtype=np.int32)
        assert observed.shape == (32, 32, 3)
        assert np.array_equal(observed.arr, expected)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (0, 2),
            (2, 0),
            (0, 2, 0),
            (2, 0, 0),
            (0, 2, 1),
            (2, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.CropToFixedSize(height=1, width=1)

                image_aug = aug(image=image)

                expected_height = 0 if shape[0] == 0 else 1
                expected_width = 0 if shape[1] == 0 else 1
                expected_shape = tuple([expected_height, expected_width]
                                       + list(shape[2:]))
                assert image_aug.shape == expected_shape

    def test_other_dtypes_bool(self):
        aug = iaa.CropToFixedSize(height=2, width=3, position="center-top")
        mask = np.zeros((2, 3), dtype=bool)
        mask[0, 1] = True
        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == image.dtype.name
        assert image_aug.shape == (2, 3)
        assert np.all(image_aug[~mask] == 0)
        assert np.all(image_aug[mask] == 1)

    def test_other_dtypes_uint_int(self):
        aug = iaa.CropToFixedSize(height=2, width=3, position="center-top")
        mask = np.zeros((2, 3), dtype=bool)
        mask[0, 1] = True

        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64"]

        for dtype in dtypes:
            min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            if np.dtype(dtype).kind == "i":
                values = [
                    1, 5, 10, 100, int(0.1 * max_value), int(0.2 * max_value),
                    int(0.5 * max_value), max_value - 100, max_value]
                values = values + [(-1) * value for value in values]
            else:
                values = [
                    1, 5, 10, 100, int(center_value), int(0.1 * max_value),
                    int(0.2 * max_value), int(0.5 * max_value),
                    max_value - 100, max_value]

            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    image = np.zeros((3, 3), dtype=dtype)
                    image[1, 1] = value

                    image_aug = aug.augment_image(image)

                    assert image_aug.dtype.name == dtype
                    assert image_aug.shape == (2, 3)
                    assert np.all(image_aug[~mask] == 0)
                    assert np.all(image_aug[mask] == value)

    def test_other_dtypes_float(self):
        aug = iaa.CropToFixedSize(height=2, width=3, position="center-top")
        mask = np.zeros((2, 3), dtype=bool)
        mask[0, 1] = True

        dtypes = ["float16", "float32", "float64", "float128"]

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
                    assert image_aug.shape == (2, 3)
                    assert np.all(_isclose(image_aug[~mask], 0))
                    assert np.all(_isclose(image_aug[mask],
                                           np.float128(value)))


class TestKeepSizeByResize(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def children(self):
        return iaa.Crop((1, 0, 0, 0), keep_size=False)

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=3)]
        return ia.KeypointsOnImage(kps, shape=(4, 4, 3))

    @property
    def heatmaps(self):
        heatmaps_arr = np.linspace(
            0.0, 1.0, 4*4*1).reshape((4, 4, 1)).astype(np.float32)
        return HeatmapsOnImage(heatmaps_arr, shape=(4, 4, 1))

    @property
    def heatmaps_cubic(self):
        heatmaps_arr = self.heatmaps.get_arr()
        heatmaps_oi_cubic = HeatmapsOnImage(
            heatmaps_arr[1:, :, :], shape=(3, 4, 3)
        ).resize((4, 4), interpolation="cubic")
        heatmaps_oi_cubic.shape = (4, 4, 3)
        return heatmaps_oi_cubic

    @property
    def heatmaps_nearest(self):
        heatmaps_arr = self.heatmaps.get_arr()
        heatmaps_oi_nearest = HeatmapsOnImage(
            heatmaps_arr[1:, :, :], shape=(3, 4, 1)
        ).resize((4, 4), interpolation="nearest")
        heatmaps_oi_nearest.shape = (4, 4, 3)
        return heatmaps_oi_nearest

    @property
    def segmaps(self):
        segmaps_arr = np.arange(4*4*1).reshape((4, 4, 1)).astype(np.int32)
        return SegmentationMapsOnImage(segmaps_arr, shape=(4, 4, 1))

    @property
    def segmaps_nearest(self):
        segmaps_arr = self.segmaps.get_arr()
        segmaps_oi_nearest = SegmentationMapsOnImage(
            segmaps_arr[1:, :, :], shape=(3, 4, 1)
        ).resize((4, 4), interpolation="nearest")
        segmaps_oi_nearest.shape = (4, 4, 3)
        return segmaps_oi_nearest

    def test__draw_samples_each_one_interpolation(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation="nearest",
            interpolation_heatmaps="linear",
            interpolation_segmaps="cubic")

        samples, samples_heatmaps, samples_segmaps = aug._draw_samples(
            1000, iarandom.RNG(1))

        assert "nearest" in samples
        assert len(set(samples)) == 1
        assert "linear" in samples_heatmaps
        assert len(set(samples_heatmaps)) == 1
        assert "cubic" in samples_segmaps
        assert len(set(samples_segmaps)) == 1

    def test__draw_samples_each_one_interpolation_via_cv2_constants(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation=cv2.INTER_LINEAR,
            interpolation_heatmaps=cv2.INTER_NEAREST,
            interpolation_segmaps=cv2.INTER_CUBIC)

        samples, samples_heatmaps, samples_segmaps = aug._draw_samples(
            1000, iarandom.RNG(1))

        assert cv2.INTER_LINEAR in samples
        assert len(set(samples)) == 1
        assert cv2.INTER_NEAREST in samples_heatmaps
        assert len(set(samples_heatmaps)) == 1
        assert cv2.INTER_CUBIC in samples_segmaps
        assert len(set(samples_segmaps)) == 1

    def test__draw_samples_with_images_no_resize_and_others_same_as_imgs(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation=iaa.KeepSizeByResize.NO_RESIZE,
            interpolation_heatmaps=iaa.KeepSizeByResize.SAME_AS_IMAGES,
            interpolation_segmaps=iaa.KeepSizeByResize.SAME_AS_IMAGES)

        samples, samples_heatmaps, samples_segmaps = aug._draw_samples(
            1000, iarandom.RNG(1))

        assert iaa.KeepSizeByResize.NO_RESIZE in samples
        assert len(set(samples)) == 1
        assert iaa.KeepSizeByResize.NO_RESIZE in samples_heatmaps
        assert len(set(samples_heatmaps)) == 1
        assert iaa.KeepSizeByResize.NO_RESIZE in samples_segmaps
        assert len(set(samples_segmaps)) == 1

    def test__draw_samples_list_of_interpolations_incl_same_as_images(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation=["cubic", "nearest"],
            interpolation_heatmaps=[
                "linear", iaa.KeepSizeByResize.SAME_AS_IMAGES],
            interpolation_segmaps=[
                "linear", iaa.KeepSizeByResize.SAME_AS_IMAGES])

        samples, samples_heatmaps, samples_segmaps = aug._draw_samples(
            5000, iarandom.RNG(1))

        assert "cubic" in samples
        assert "nearest" in samples
        assert len(set(samples)) == 2
        assert "linear" in samples_heatmaps
        assert "nearest" in samples_heatmaps
        assert len(set(samples_heatmaps)) == 3
        assert np.isclose(
            np.sum(samples == samples_heatmaps) / samples_heatmaps.size,
            0.5,
            rtol=0, atol=0.1)
        assert "linear" in samples_segmaps
        assert "nearest" in samples_segmaps
        assert len(set(samples_segmaps)) == 3
        assert np.isclose(
            np.sum(samples == samples_segmaps) / samples_segmaps.size,
            0.5,
            rtol=0, atol=0.1)

    def test__draw_samples_list_of_each_two_interpolations(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation=iap.Choice(["cubic", "linear"]),
            interpolation_heatmaps=iap.Choice(["linear", "nearest"]),
            interpolation_segmaps=iap.Choice(["linear", "nearest"]))

        samples, samples_heatmaps, samples_segmaps = aug._draw_samples(
            10000, iarandom.RNG(1))

        assert "cubic" in samples
        assert "linear" in samples
        assert len(set(samples)) == 2
        assert "linear" in samples_heatmaps
        assert "nearest" in samples_heatmaps
        assert len(set(samples_heatmaps)) == 2
        assert "linear" in samples_segmaps
        assert "nearest" in samples_segmaps
        assert len(set(samples_segmaps)) == 2

    def test_image_interpolation_is_cubic(self):
        aug = iaa.KeepSizeByResize(self.children, interpolation="cubic")
        img = np.arange(0, 4*4*3, 1).reshape((4, 4, 3)).astype(np.uint8)

        observed = aug.augment_image(img)

        assert observed.shape == (4, 4, 3)
        assert observed.dtype.type == np.uint8
        expected = ia.imresize_single_image(
            img[1:, :, :], img.shape[0:2], interpolation="cubic")
        assert np.allclose(observed, expected)

    def test_image_interpolation_is_no_resize(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation=iaa.KeepSizeByResize.NO_RESIZE)
        img = np.arange(0, 4*4*3, 1).reshape((4, 4, 3)).astype(np.uint8)

        observed = aug.augment_image(img)

        expected = img[1:, :, :]
        assert observed.shape == (3, 4, 3)
        assert observed.dtype.type == np.uint8
        assert np.allclose(observed, expected)

    def test_keypoints_interpolation_is_cubic(self):
        aug = iaa.KeepSizeByResize(self.children, interpolation="cubic")
        kpsoi = self.kpsoi

        kpoi_aug = aug.augment_keypoints([kpsoi])[0]
        
        assert kpoi_aug.shape == (4, 4, 3)
        assert np.isclose(kpoi_aug.keypoints[0].x, 0, rtol=0, atol=1e-4)
        assert np.isclose(kpoi_aug.keypoints[0].y,
                          ((1-1)/3)*4,
                          rtol=0, atol=1e-4)
        assert np.isclose(kpoi_aug.keypoints[1].x, 1, rtol=0, atol=1e-4)
        assert np.isclose(kpoi_aug.keypoints[1].y,
                          ((1-1)/3)*4,
                          rtol=0, atol=1e-4)
        assert np.isclose(kpoi_aug.keypoints[2].x, 2, rtol=0, atol=1e-4)
        assert np.isclose(kpoi_aug.keypoints[2].y,
                          ((3-1)/3)*4,
                          rtol=0, atol=1e-4)

    def test_keypoints_interpolation_is_no_resize(self):
        aug = iaa.KeepSizeByResize(
            self.children, interpolation=iaa.KeepSizeByResize.NO_RESIZE)
        kpsoi = self.kpsoi

        kpoi_aug = aug.augment_keypoints([kpsoi])[0]

        assert kpoi_aug.shape == (3, 4, 3)
        assert np.isclose(kpoi_aug.keypoints[0].x, 0, rtol=0, atol=1e-4)
        assert np.isclose(kpoi_aug.keypoints[0].y, 0, rtol=0, atol=1e-4)
        assert np.isclose(kpoi_aug.keypoints[1].x, 1, rtol=0, atol=1e-4)
        assert np.isclose(kpoi_aug.keypoints[1].y, 0, rtol=0, atol=1e-4)
        assert np.isclose(kpoi_aug.keypoints[2].x, 2, rtol=0, atol=1e-4)
        assert np.isclose(kpoi_aug.keypoints[2].y, 2, rtol=0, atol=1e-4)


    def test_polygons_interpolation_is_cubic(self):
        aug = iaa.KeepSizeByResize(self.children, interpolation="cubic")
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (3, 0), (3, 3)])
        ], shape=(4, 4, 3))

        psoi_aug = aug.augment_polygons(psoi)

        assert psoi_aug.shape == (4, 4, 3)
        assert np.allclose(
            psoi_aug.polygons[0].coords,
            [(0, ((0-1)/3)*4),
             (3, ((0-1)/3)*4),
             (3, ((3-1)/3)*4)]
        )

    def test_polygons_interpolation_is_no_resize(self):
        aug = iaa.KeepSizeByResize(
            self.children, interpolation=iaa.KeepSizeByResize.NO_RESIZE)
        psoi = ia.PolygonsOnImage([
            ia.Polygon([(0, 0), (3, 0), (3, 3)])
        ], shape=(4, 4, 3))

        psoi_aug = aug.augment_polygons(psoi)

        assert psoi_aug.shape == (3, 4, 3)
        assert np.allclose(
            psoi_aug.polygons[0].coords,
            [(0, 0-1),
             (3, 0-1),
             (3, 3-1)]
        )

    def test_bounding_boxes_interpolation_is_cubic(self):
        aug = iaa.KeepSizeByResize(self.children, interpolation="cubic")
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=3, y2=4)
        ], shape=(4, 4, 3))

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        assert bbsoi_aug.shape == (4, 4, 3)
        assert np.allclose(
            bbsoi_aug.bounding_boxes[0].coords,
            [(0, ((1-1)/3)*4),
             (3, ((4-1)/3)*4)]
        )

    def test_bounding_boxes_interpolation_is_no_resize(self):
        aug = iaa.KeepSizeByResize(
            self.children, interpolation=iaa.KeepSizeByResize.NO_RESIZE)
        bbsoi = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=0, y1=1, x2=3, y2=4)
        ], shape=(4, 4, 3))

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        assert bbsoi_aug.shape == (3, 4, 3)
        assert np.allclose(
            bbsoi_aug.bounding_boxes[0].coords,
            [(0, 1-1),
             (3, 4-1)]
        )

    def test_heatmaps_specific_interpolation_set_to_no_nearest(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation="cubic",
            interpolation_heatmaps="nearest")

        heatmaps_oi = self.heatmaps
        heatmaps_oi_nearest = self.heatmaps_nearest

        heatmaps_oi_aug = aug.augment_heatmaps([heatmaps_oi])[0]

        assert heatmaps_oi_aug.arr_0to1.shape == (4, 4, 1)
        assert np.allclose(heatmaps_oi_aug.arr_0to1,
                           heatmaps_oi_nearest.arr_0to1)

    def test_heatmaps_specific_interpolation_set_to_list_of_two(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation="cubic",
            interpolation_heatmaps=["nearest", "cubic"])

        heatmaps_oi = self.heatmaps
        heatmaps_oi_cubic = self.heatmaps_cubic
        heatmaps_oi_nearest = self.heatmaps_nearest

        hmoi_aug = aug.augment_heatmaps([heatmaps_oi])[0]

        assert hmoi_aug.arr_0to1.shape == (4, 4, 1)
        assert (
            np.allclose(hmoi_aug.arr_0to1, heatmaps_oi_nearest.arr_0to1)
            or np.allclose(hmoi_aug.arr_0to1, heatmaps_oi_cubic.arr_0to1)
        )

    def test_heatmaps_specific_interpolation_set_to_no_resize(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation="cubic",
            interpolation_heatmaps=iaa.KeepSizeByResize.NO_RESIZE)

        heatmaps_oi = self.heatmaps

        heatmaps_oi_aug = aug.augment_heatmaps([heatmaps_oi])[0]

        assert heatmaps_oi_aug.arr_0to1.shape == (3, 4, 1)
        assert np.allclose(
            heatmaps_oi_aug.arr_0to1, heatmaps_oi.arr_0to1[1:, :, :])

    def test_heatmaps_specific_interpolation_set_to_same_as_images(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation="cubic",
            interpolation_heatmaps=iaa.KeepSizeByResize.SAME_AS_IMAGES)

        heatmaps_oi = self.heatmaps
        heatmaps_oi_cubic = self.heatmaps_cubic

        heatmaps_oi_aug = aug.augment_heatmaps([heatmaps_oi])[0]

        assert heatmaps_oi_aug.arr_0to1.shape == (4, 4, 1)
        assert np.allclose(
            heatmaps_oi_aug.arr_0to1, heatmaps_oi_cubic.arr_0to1)

    def test_segmaps_general_interpolation_set_to_cubic(self):
        aug = iaa.KeepSizeByResize(self.children, interpolation="cubic")
        segmaps_oi = self.segmaps
        segmaps_oi_nearest = self.segmaps_nearest

        segmaps_oi_aug = aug.augment_segmentation_maps([segmaps_oi])[0]

        assert segmaps_oi_aug.arr.shape == (4, 4, 1)
        assert np.array_equal(segmaps_oi_aug.arr, segmaps_oi_nearest.arr)

    def test_segmaps_specific_interpolation_set_to_nearest(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation="cubic",
            interpolation_segmaps="nearest")
        segmaps_oi = self.segmaps
        segmaps_oi_nearest = self.segmaps_nearest

        segmaps_oi_aug = aug.augment_segmentation_maps([segmaps_oi])[0]

        assert segmaps_oi_aug.arr.shape == (4, 4, 1)
        assert np.array_equal(segmaps_oi_aug.arr, segmaps_oi_nearest.arr)

    def test_segmaps_specific_interpolation_set_to_no_resize(self):
        aug = iaa.KeepSizeByResize(
            self.children,
            interpolation="cubic",
            interpolation_segmaps=iaa.KeepSizeByResize.NO_RESIZE)
        segmaps_oi = self.segmaps

        segmaps_oi_aug = aug.augment_segmentation_maps([segmaps_oi])[0]

        assert segmaps_oi_aug.arr.shape == (3, 4, 1)
        assert np.array_equal(segmaps_oi_aug.arr, segmaps_oi.arr[1:, :, :])

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (0, 2),
            (2, 0),
            (0, 2, 0),
            (2, 0, 0),
            (0, 2, 1),
            (2, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.KeepSizeByResize(
                    iaa.CropToFixedSize(height=1, width=1)
                )

                image_aug = aug(image=image)

                assert image_aug.shape == image.shape
