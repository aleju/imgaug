from __future__ import print_function, division, absolute_import

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

import numpy as np

import imgaug as ia
import imgaug.random as iarandom
from imgaug.augmentables.bbs import _LabelOnImageDrawer
from imgaug.testutils import wrap_shift_deprecation, assertWarns


class TestBoundingBox_project_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, cba, *args, **kwargs):
        return cba.project_(*args, **kwargs)

    def test_project_same_shape(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, (10, 10), (10, 10))

        assert np.isclose(bb2.y1, 10)
        assert np.isclose(bb2.x1, 20)
        assert np.isclose(bb2.y2, 30)
        assert np.isclose(bb2.x2, 40)

    def test_project_upscale_by_2(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, (10, 10), (20, 20))

        assert np.isclose(bb2.y1, 10*2)
        assert np.isclose(bb2.x1, 20*2)
        assert np.isclose(bb2.y2, 30*2)
        assert np.isclose(bb2.x2, 40*2)

    def test_project_downscale_by_2(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, (10, 10), (5, 5))

        assert np.isclose(bb2.y1, 10*0.5)
        assert np.isclose(bb2.x1, 20*0.5)
        assert np.isclose(bb2.y2, 30*0.5)
        assert np.isclose(bb2.x2, 40*0.5)

    def test_project_onto_wider_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, (10, 10), (10, 20))

        assert np.isclose(bb2.y1, 10*1)
        assert np.isclose(bb2.x1, 20*2)
        assert np.isclose(bb2.y2, 30*1)
        assert np.isclose(bb2.x2, 40*2)

    def test_project_onto_higher_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, (10, 10), (20, 10))

        assert np.isclose(bb2.y1, 10*2)
        assert np.isclose(bb2.x1, 20*1)
        assert np.isclose(bb2.y2, 30*2)
        assert np.isclose(bb2.x2, 40*1)

    def test_inplaceness(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, (10, 10), (10, 10))

        if self._is_inplace:
            assert bb2 is bb
        else:
            assert bb2 is not bb


class TestBoundingBox_project(TestBoundingBox_project_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, cba, *args, **kwargs):
        return cba.project(*args, **kwargs)


class TestBoundingBox_extend_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, cba, *args, **kwargs):
        return cba.extend_(*args, **kwargs)

    def test_extend_all_sides_by_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, all_sides=1)

        assert bb2.y1 == 10-1
        assert bb2.y2 == 30+1
        assert bb2.x1 == 20-1
        assert bb2.x2 == 40+1

    def test_extend_all_sides_by_minus_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, all_sides=-1)

        assert bb2.y1 == 10-(-1)
        assert bb2.y2 == 30+(-1)
        assert bb2.x1 == 20-(-1)
        assert bb2.x2 == 40+(-1)

    def test_extend_top_by_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, top=1)

        assert bb2.y1 == 10-1
        assert bb2.y2 == 30+0
        assert bb2.x1 == 20-0
        assert bb2.x2 == 40+0

    def test_extend_right_by_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, right=1)

        assert bb2.y1 == 10-0
        assert bb2.y2 == 30+0
        assert bb2.x1 == 20-0
        assert bb2.x2 == 40+1

    def test_extend_bottom_by_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, bottom=1)

        assert bb2.y1 == 10-0
        assert bb2.y2 == 30+1
        assert bb2.x1 == 20-0
        assert bb2.x2 == 40+0

    def test_extend_left_by_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, left=1)

        assert bb2.y1 == 10-0
        assert bb2.y2 == 30+0
        assert bb2.x1 == 20-1
        assert bb2.x2 == 40+0

    def test_inplaceness(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, all_sides=1)

        if self._is_inplace:
            assert bb2 is bb
        else:
            assert bb2 is not bb


class TestBoundingBox_extend(TestBoundingBox_extend_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, cba, *args, **kwargs):
        return cba.extend(*args, **kwargs)


class TestBoundingBox_clip_out_of_image_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, cba, *args, **kwargs):
        return cba.clip_out_of_image_(*args, **kwargs)

    def test_clip_out_of_image_with_bb_fully_inside_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb_cut = self._func(bb, (100, 100, 3))

        assert bb_cut.y1 == 10
        assert bb_cut.x1 == 20
        assert bb_cut.y2 == 30
        assert bb_cut.x2 == 40

    def test_clip_out_of_image_with_array_as_shape(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        bb_cut = bb.clip_out_of_image(image)

        assert bb_cut.y1 == 10
        assert bb_cut.x1 == 20
        assert bb_cut.y2 == 30
        assert bb_cut.x2 == 40

    def test_clip_out_of_image_with_bb_too_high(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb_cut = self._func(bb, (20, 100, 3))

        assert bb_cut.y1 == 10
        assert bb_cut.x1 == 20
        assert np.isclose(bb_cut.y2, 20)
        assert bb_cut.x2 == 40

    def test_clip_out_of_image_with_bb_too_wide(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb_cut = self._func(bb, (100, 30, 3))

        assert bb_cut.y1 == 10
        assert bb_cut.x1 == 20
        assert bb_cut.y2 == 30
        assert np.isclose(bb_cut.x2, 30)

    def test_inplaceness(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = self._func(bb, (100, 100, 3))

        if self._is_inplace:
            assert bb2 is bb
        else:
            assert bb2 is not bb


class TestBoundingBox_clip_out_of_image(TestBoundingBox_clip_out_of_image_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, cba, *args, **kwargs):
        return cba.clip_out_of_image(*args, **kwargs)


class TestBoundingBox_shift_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, cba, *args, **kwargs):
        def _func_impl():
            return cba.shift_(*args, **kwargs)

        return wrap_shift_deprecation(_func_impl, *args, **kwargs)

    def test_shift_by_x(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_top = self._func(bb, x=1)
        assert bb_top.y1 == 10
        assert bb_top.x1 == 20 + 1
        assert bb_top.y2 == 30
        assert bb_top.x2 == 40 + 1

    def test_shift_by_y(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_top = self._func(bb, y=1)
        assert bb_top.y1 == 10 + 1
        assert bb_top.x1 == 20
        assert bb_top.y2 == 30 + 1
        assert bb_top.x2 == 40

    def test_inplaceness(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = self._func(bb, y=0)

        if self._is_inplace:
            assert bb2 is bb
        else:
            assert bb2 is not bb


class TestBoundingBox_shift(TestBoundingBox_shift_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, cba, *args, **kwargs):
        def _func_impl():
            return cba.shift(*args, **kwargs)

        return wrap_shift_deprecation(_func_impl, *args, **kwargs)

    def test_shift_top_by_zero(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_top = self._func(bb, top=0)
        assert bb_top.y1 == 10
        assert bb_top.x1 == 20
        assert bb_top.y2 == 30
        assert bb_top.x2 == 40

    def test_shift_right_by_zero(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_right = self._func(bb, right=0)
        assert bb_right.y1 == 10
        assert bb_right.x1 == 20
        assert bb_right.y2 == 30
        assert bb_right.x2 == 40

    def test_shift_bottom_by_zero(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_bottom = self._func(bb, bottom=0)
        assert bb_bottom.y1 == 10
        assert bb_bottom.x1 == 20
        assert bb_bottom.y2 == 30
        assert bb_bottom.x2 == 40

    def test_shift_left_by_zero(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_left = self._func(bb, left=0)
        assert bb_left.y1 == 10
        assert bb_left.x1 == 20
        assert bb_left.y2 == 30
        assert bb_left.x2 == 40

    def test_shift_top_by_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_top = self._func(bb, top=1)
        assert bb_top.y1 == 10+1
        assert bb_top.x1 == 20
        assert bb_top.y2 == 30+1
        assert bb_top.x2 == 40

    def test_shift_right_by_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_right = self._func(bb, right=1)
        assert bb_right.y1 == 10
        assert bb_right.x1 == 20-1
        assert bb_right.y2 == 30
        assert bb_right.x2 == 40-1

    def test_shift_bottom_by_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_bottom = self._func(bb, bottom=1)
        assert bb_bottom.y1 == 10-1
        assert bb_bottom.x1 == 20
        assert bb_bottom.y2 == 30-1
        assert bb_bottom.x2 == 40

    def test_shift_left_by_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_left = self._func(bb, left=1)
        assert bb_left.y1 == 10
        assert bb_left.x1 == 20+1
        assert bb_left.y2 == 30
        assert bb_left.x2 == 40+1

    def test_shift_top_by_minus_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_top = self._func(bb, top=-1)
        assert bb_top.y1 == 10-1
        assert bb_top.x1 == 20
        assert bb_top.y2 == 30-1
        assert bb_top.x2 == 40

    def test_shift_right_by_minus_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_right = self._func(bb, right=-1)
        assert bb_right.y1 == 10
        assert bb_right.x1 == 20+1
        assert bb_right.y2 == 30
        assert bb_right.x2 == 40+1

    def test_shift_bottom_by_minus_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_bottom = self._func(bb, bottom=-1)
        assert bb_bottom.y1 == 10+1
        assert bb_bottom.x1 == 20
        assert bb_bottom.y2 == 30+1
        assert bb_bottom.x2 == 40

    def test_shift_left_by_minus_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_left = self._func(bb, left=-1)
        assert bb_left.y1 == 10
        assert bb_left.x1 == 20-1
        assert bb_left.y2 == 30
        assert bb_left.x2 == 40-1

    def test_shift_all_sides_by_individual_amounts(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_mix = self._func(bb, top=1, bottom=2, left=3, right=4)
        assert bb_mix.y1 == 10+1-2
        assert bb_mix.x1 == 20+3-4
        assert bb_mix.y2 == 30+3-4
        assert bb_mix.x2 == 40+1-2


class TestBoundingBox(unittest.TestCase):
    def test___init__(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        assert bb.y1 == 10
        assert bb.x1 == 20
        assert bb.y2 == 30
        assert bb.x2 == 40
        assert bb.label is None

    def test___init___floats(self):
        bb = ia.BoundingBox(y1=10.1, x1=20.2, y2=30.3, x2=40.4)
        assert np.isclose(bb.y1, 10.1)
        assert np.isclose(bb.x1, 20.2)
        assert np.isclose(bb.y2, 30.3)
        assert np.isclose(bb.x2, 40.4)
        assert bb.label is None

    def test___init___label(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label="foo")
        assert bb.y1 == 10
        assert bb.x1 == 20
        assert bb.y2 == 30
        assert bb.x2 == 40
        assert bb.label == "foo"

    def test___init___wrong_x1_x2_order(self):
        bb = ia.BoundingBox(y1=10, x1=40, y2=30, x2=20)
        assert bb.y1 == 10
        assert bb.x1 == 20
        assert bb.y2 == 30
        assert bb.x2 == 40

    def test___init___wrong_y1_y2_order(self):
        bb = ia.BoundingBox(y1=30, x1=20, y2=10, x2=40)
        assert bb.y1 == 10
        assert bb.x1 == 20
        assert bb.y2 == 30
        assert bb.x2 == 40

    def test_coords_property_ints(self):
        bb = ia.BoundingBox(x1=10, y1=20, x2=30, y2=40)
        coords = bb.coords
        assert np.allclose(coords, [[10, 20], [30, 40]],
                           atol=1e-4, rtol=0)

    def test_coords_property_floats(self):
        bb = ia.BoundingBox(x1=10.1, y1=20.2, x2=30.3, y2=40.4)
        coords = bb.coords
        assert np.allclose(coords, [[10.1, 20.2], [30.3, 40.4]],
                           atol=1e-4, rtol=0)

    def test_xy_int_properties(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        assert bb.y1_int == 10
        assert bb.x1_int == 20
        assert bb.y2_int == 30
        assert bb.x2_int == 40

    def test_xy_int_properties_floats(self):
        bb = ia.BoundingBox(y1=10.1, x1=20.2, y2=30.6, x2=40.7)
        assert bb.y1_int == 10
        assert bb.x1_int == 20
        assert bb.y2_int == 31
        assert bb.x2_int == 41

    def test_width(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        assert bb.width == 40 - 20

    def test_width_floats(self):
        bb = ia.BoundingBox(y1=10.1, x1=20.2, y2=30.3, x2=40.4)
        assert np.isclose(bb.width, 40.4 - 20.2)

    def test_height(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        assert bb.height == 30 - 10

    def test_height_floats(self):
        bb = ia.BoundingBox(y1=10.1, x1=20.2, y2=30.3, x2=40.4)
        assert np.isclose(bb.height, 30.3 - 10.1)

    def test_center_x(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        expected = 20 + (40 - 20)/2
        assert np.isclose(bb.center_x, expected)

    def test_center_x_floats(self):
        bb = ia.BoundingBox(y1=10.1, x1=20.2, y2=30.3, x2=40.4)
        expected = 20.2 + (40.4 - 20.2)/2
        assert np.isclose(bb.center_x, expected)

    def test_center_y(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        expected = 10 + (30 - 10)/2
        assert np.isclose(bb.center_y, expected)

    def test_center_y_floats(self):
        bb = ia.BoundingBox(y1=10.1, x1=20.2, y2=30.3, x2=40.4)
        expected = 10.1 + (30.3 - 10.1)/2
        assert np.isclose(bb.center_y, expected)

    def test_area(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        assert bb.area == (30-10) * (40-20)

    def test_area_floats(self):
        bb = ia.BoundingBox(y1=10.1, x1=20.2, y2=30.3, x2=40.4)
        assert np.isclose(bb.area, (30.3-10.1) * (40.4-20.2))

    def test_contains(self):
        bb = ia.BoundingBox(y1=1, x1=2, y2=1+4, x2=2+5, label=None)
        assert bb.contains(ia.Keypoint(x=2.5, y=1.5)) is True
        assert bb.contains(ia.Keypoint(x=2, y=1)) is True
        assert bb.contains(ia.Keypoint(x=0, y=0)) is False

    def test_intersection(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=10, x1=39, y2=30, x2=59)

        bb_inter = bb1.intersection(bb2)

        assert bb_inter.x1 == 39
        assert bb_inter.x2 == 40
        assert bb_inter.y1 == 10
        assert bb_inter.y2 == 30

    def test_intersection_of_non_overlapping_bbs(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=10, x1=41, y2=30, x2=61)

        bb_inter = bb1.intersection(bb2, default=False)

        assert bb_inter is False

    def test_union(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=10, x1=39, y2=30, x2=59)

        bb_union = bb1.union(bb2)

        assert bb_union.x1 == 20
        assert bb_union.x2 == 59
        assert bb_union.y1 == 10
        assert bb_union.y2 == 30

    def test_iou_of_identical_bbs(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        iou = bb1.iou(bb2)

        assert np.isclose(iou, 1.0)

    def test_iou_of_non_overlapping_bbs(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=10, x1=41, y2=30, x2=61)

        iou = bb1.iou(bb2)

        assert np.isclose(iou, 0.0)

    def test_iou_of_partially_overlapping_bbs(self):
        bb1 = ia.BoundingBox(y1=10, x1=10, y2=20, x2=20)
        bb2 = ia.BoundingBox(y1=15, x1=15, y2=25, x2=25)

        iou = bb1.iou(bb2)

        area_union = 10 * 10 + 10 * 10 - 5 * 5
        area_intersection = 5 * 5
        iou_expected = area_intersection / area_union
        assert np.isclose(iou, iou_expected)

    def test_compute_out_of_image_area__fully_inside(self):
        bb = ia.BoundingBox(y1=10.1, x1=20.2, y2=30.3, x2=40.4)
        image_shape = (100, 200, 3)
        area_ooi = bb.compute_out_of_image_area(image_shape)
        assert np.isclose(area_ooi, 0.0)

    def test_compute_out_of_image_area__partially_ooi(self):
        bb = ia.BoundingBox(y1=10, x1=-20, y2=30, x2=40)
        image_shape = (100, 200, 3)
        area_ooi = bb.compute_out_of_image_area(image_shape)
        assert np.isclose(area_ooi, (0-(-20))*(30-10))

    def test_compute_out_of_image_area__fully_ooi(self):
        bb = ia.BoundingBox(y1=10, x1=-20, y2=30, x2=-10)
        image_shape = (100, 200, 3)
        area_ooi = bb.compute_out_of_image_area(image_shape)
        assert np.isclose(area_ooi, 20*10)

    def test_compute_out_of_image_area__zero_sized_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        image_shape = (0, 0, 3)
        area_ooi = bb.compute_out_of_image_area(image_shape)
        assert np.isclose(area_ooi, bb.area)

    def test_compute_out_of_image_area__bb_has_zero_sized_area(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=10, x2=20)
        image_shape = (100, 200, 3)
        area_ooi = bb.compute_out_of_image_area(image_shape)
        assert np.isclose(area_ooi, 0.0)

    def test_compute_out_of_image_fraction__inside_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        image_shape = (100, 200, 3)

        factor = bb.compute_out_of_image_fraction(image_shape)

        assert np.isclose(factor, 0.0)

    def test_compute_out_of_image_fraction__partially_ooi(self):
        bb = ia.BoundingBox(y1=10, x1=-20, y2=30, x2=40)
        image_shape = (100, 200, 3)

        factor = bb.compute_out_of_image_fraction(image_shape)

        expected = (20 * 20) / (20 * 60)
        assert np.isclose(factor, expected)

    def test_compute_out_of_image_fraction__fully_ooi(self):
        bb = ia.BoundingBox(y1=10, x1=-20, y2=30, x2=0)
        image_shape = (100, 200, 3)

        factor = bb.compute_out_of_image_fraction(image_shape)

        assert np.isclose(factor, 1.0)

    def test_compute_out_of_image_fraction__zero_area_inside_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=10, x2=20)
        image_shape = (100, 200, 3)

        factor = bb.compute_out_of_image_fraction(image_shape)

        assert np.isclose(factor, 0.0)

    def test_compute_out_of_image_fraction__zero_area_ooi(self):
        bb = ia.BoundingBox(y1=-10, x1=20, y2=-10, x2=20)
        image_shape = (100, 200, 3)

        factor = bb.compute_out_of_image_fraction(image_shape)

        assert np.isclose(factor, 1.0)

    def test_is_fully_within_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
        assert bb.is_fully_within_image((100, 100, 3)) is True
        assert bb.is_fully_within_image((20, 100, 3)) is False
        assert bb.is_fully_within_image((100, 30, 3)) is False
        assert bb.is_fully_within_image((1, 1, 3)) is False

    def test_is_partly_within_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
        assert bb.is_partly_within_image((100, 100, 3)) is True
        assert bb.is_partly_within_image((20, 100, 3)) is True
        assert bb.is_partly_within_image((100, 30, 3)) is True
        assert bb.is_partly_within_image((1, 1, 3)) is False

    def test_is_out_of_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)

        subtests = [
            ((100, 100, 3), True, True, False),
            ((100, 100, 3), False, True, False),
            ((100, 100, 3), True, False, False),
            ((20, 100, 3), True, True, True),
            ((20, 100, 3), False, True, False),
            ((20, 100, 3), True, False, True),
            ((100, 30, 3), True, True, True),
            ((100, 30, 3), False, True, False),
            ((100, 30, 3), True, False, True),
            ((1, 1, 3), True, True, True),
            ((1, 1, 3), False, True, True),
            ((1, 1, 3), True, False, False)
        ]

        for shape, partly, fully, expected in subtests:
            with self.subTest(shape=shape, partly=partly, fully=fully):
                observed = bb.is_out_of_image(shape,
                                              partly=partly, fully=fully)
                assert observed is expected

    @mock.patch("imgaug.augmentables.bbs._LabelOnImageDrawer")
    def test_draw_label_on_image_mocked(self, mock_drawer):
        mock_drawer.return_value = mock_drawer
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        bb = ia.BoundingBox(y1=1, x1=1, y2=3, x2=3)

        result = bb.draw_label_on_image(image)

        kwargs = mock_drawer.call_args_list[0][1]
        assert kwargs["color"] == (0, 255, 0)
        assert kwargs["color_text"] is None
        assert kwargs["color_bg"] is None
        assert np.isclose(kwargs["alpha"], 1.0)
        assert kwargs["size"] == 1
        assert kwargs["size_text"] == 20
        assert kwargs["height"] == 30
        assert kwargs["raise_if_out_of_image"] is False

        assert mock_drawer.draw_on_image.call_count == 1

    @mock.patch("imgaug.augmentables.bbs._LabelOnImageDrawer")
    def test_draw_label_on_image_mocked_inplace(self, mock_drawer):
        mock_drawer.return_value = mock_drawer
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        bb = ia.BoundingBox(y1=1, x1=1, y2=3, x2=3)

        result = bb.draw_label_on_image(image, copy=False)

        kwargs = mock_drawer.call_args_list[0][1]
        assert kwargs["color"] == (0, 255, 0)
        assert kwargs["color_text"] is None
        assert kwargs["color_bg"] is None
        assert np.isclose(kwargs["alpha"], 1.0)
        assert kwargs["size"] == 1
        assert kwargs["size_text"] == 20
        assert kwargs["height"] == 30
        assert kwargs["raise_if_out_of_image"] is False

        assert mock_drawer.draw_on_image_.call_count == 1

    def test_draw_label_on_image(self):
        image = np.zeros((100, 70, 3), dtype=np.uint8)
        bb = ia.BoundingBox(y1=40, x1=10, y2=50, x2=40)

        result = bb.draw_label_on_image(image,
                                        color_bg=(123, 123, 123),
                                        color_text=(222, 222, 222))

        color_bg = np.uint8([123, 123, 123]).reshape((1, 1, -1))
        color_text = np.uint8([222, 222, 222]).reshape((1, 1, -1))
        matches_bg = np.min(result == color_bg, axis=-1)
        matches_text = np.min(result == color_text, axis=-1)
        assert np.any(matches_bg > 0)
        assert np.any(matches_text > 0)

    @classmethod
    def _get_standard_draw_box_on_image_vars(cls):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        bb = ia.BoundingBox(y1=1, x1=1, y2=3, x2=3)
        bb_mask = np.zeros(image.shape[0:2], dtype=bool)
        bb_mask[1:3+1, 1] = True
        bb_mask[1:3+1, 3] = True
        bb_mask[1, 1:3+1] = True
        bb_mask[3, 1:3+1] = True
        return image, bb, bb_mask

    def test_draw_box_on_image(self):
        image, bb, bb_mask = self._get_standard_draw_box_on_image_vars()

        image_bb = bb.draw_box_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 255, 255])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])
        assert np.all(image == 0)

    def test_draw_box_on_image_red_color(self):
        image, bb, bb_mask = self._get_standard_draw_box_on_image_vars()

        image_bb = bb.draw_box_on_image(
            image, color=[255, 0, 0], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 0, 0])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    def test_draw_box_on_image_single_int_as_color(self):
        image, bb, bb_mask = self._get_standard_draw_box_on_image_vars()

        image_bb = bb.draw_box_on_image(
            image, color=128, alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [128, 128, 128])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    def test_draw_box_on_image_alpha_at_50_percent(self):
        image, bb, bb_mask = self._get_standard_draw_box_on_image_vars()

        image_bb = bb.draw_box_on_image(
            image + 100, color=[200, 200, 200], alpha=0.5, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [150, 150, 150])
        assert np.all(image_bb[~bb_mask] == [100, 100, 100])

    def test_draw_box_on_image_alpha_at_50_percent_and_float32_image(self):
        image, bb, bb_mask = self._get_standard_draw_box_on_image_vars()

        image_bb = bb.draw_box_on_image(
            (image+100).astype(np.float32),
            color=[200, 200, 200], alpha=0.5, size=1,
            copy=True, raise_if_out_of_image=False)

        assert np.sum(np.abs((image_bb - [150, 150, 150])[bb_mask])) < 0.1
        assert np.sum(np.abs((image_bb - [100, 100, 100])[~bb_mask])) < 0.1

    def test_draw_box_on_image_no_copy(self):
        image, bb, bb_mask = self._get_standard_draw_box_on_image_vars()

        image_bb = bb.draw_box_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=1, copy=False,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 255, 255])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])
        assert np.all(image[bb_mask] == [255, 255, 255])
        assert np.all(image[~bb_mask] == [0, 0, 0])

    def test_draw_box_on_image_bb_outside_of_image(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        bb = ia.BoundingBox(y1=-1, x1=-1, y2=2, x2=2)
        bb_mask = np.zeros(image.shape[0:2], dtype=bool)
        bb_mask[2, 0:3] = True
        bb_mask[0:3, 2] = True

        image_bb = bb.draw_box_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 255, 255])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    def test_draw_box_on_image_bb_outside_of_image_and_very_small(self):
        image, bb, bb_mask = self._get_standard_draw_box_on_image_vars()
        bb = ia.BoundingBox(y1=-1, x1=-1, y2=1, x2=1)
        bb_mask = np.zeros(image.shape[0:2], dtype=bool)
        bb_mask[0:1+1, 1] = True
        bb_mask[1, 0:1+1] = True

        image_bb = bb.draw_box_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 255, 255])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    def test_draw_box_on_image_size_2(self):
        image, bb, _ = self._get_standard_draw_box_on_image_vars()
        bb_mask = np.zeros(image.shape[0:2], dtype=bool)
        bb_mask[0:5, 0:5] = True
        bb_mask[2, 2] = False

        image_bb = bb.draw_box_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=2, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 255, 255])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    def test_draw_box_on_image_raise_true_but_bb_partially_inside_image(self):
        image, bb, bb_mask = self._get_standard_draw_box_on_image_vars()
        bb = ia.BoundingBox(y1=-1, x1=-1, y2=1, x2=1)

        _ = bb.draw_box_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=True)

    def test_draw_box_on_image_raise_true_and_bb_fully_outside_image(self):
        image, bb, bb_mask = self._get_standard_draw_box_on_image_vars()
        bb = ia.BoundingBox(y1=-5, x1=-5, y2=-1, x2=-1)

        with self.assertRaises(Exception) as context:
            _ = bb.draw_box_on_image(
                image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
                raise_if_out_of_image=True)

        assert "Cannot draw bounding box" in str(context.exception)

    def test_draw_on_image_label_is_none(self):
        # if label is None, no label box should be drawn, only the rectangle
        # box below the label
        image = np.zeros((100, 70, 3), dtype=np.uint8)
        bb = ia.BoundingBox(y1=40, x1=10, y2=50, x2=40, label=None)

        image_drawn = bb.draw_on_image(image)

        expected = bb.draw_box_on_image(image)
        assert np.array_equal(image_drawn, expected)

    def test_draw_on_image_label_is_str(self):
        # if label is None, no label box should be drawn, only the rectangle
        # box below the label
        image = np.zeros((100, 70, 3), dtype=np.uint8)
        bb = ia.BoundingBox(y1=40, x1=10, y2=50, x2=40, label="Foo")

        image_drawn = bb.draw_on_image(image)

        expected = bb.draw_box_on_image(image)
        expected = bb.draw_label_on_image(expected)
        assert np.array_equal(image_drawn, expected)

    def test_extract_from_image(self):
        image = iarandom.RNG(1234).integers(0, 255, size=(10, 10, 3))
        bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3)
        image_sub = bb.extract_from_image(image)
        assert np.array_equal(image_sub, image[1:3, 1:3, :])

    def test_extract_from_image_no_channels(self):
        image = iarandom.RNG(1234).integers(0, 255, size=(10, 10))
        bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3)
        image_sub = bb.extract_from_image(image)
        assert np.array_equal(image_sub, image[1:3, 1:3])

    def test_extract_from_image_bb_partially_out_of_image(self):
        image = iarandom.RNG(1234).integers(0, 255, size=(10, 10, 3))

        bb = ia.BoundingBox(y1=8, y2=11, x1=8, x2=11)
        image_sub = bb.extract_from_image(image)

        image_pad = np.pad(
            image,
            ((0, 1), (0, 1), (0, 0)),
            mode="constant",
            constant_values=0)  # pad at bottom and right each 1px (black)
        assert np.array_equal(image_sub, image_pad[8:11, 8:11, :])

    def test_extract_from_image_bb_partially_out_of_image_no_channels(self):
        image = iarandom.RNG(1234).integers(0, 255, size=(10, 10))

        bb = ia.BoundingBox(y1=8, y2=11, x1=8, x2=11)
        image_sub = bb.extract_from_image(image)

        image_pad = np.pad(
            image,
            ((0, 1), (0, 1)),
            mode="constant",
            constant_values=0)  # pad at bottom and right each 1px (black)
        assert np.array_equal(image_sub, image_pad[8:11, 8:11])

    def test_extract_from_image_bb_partially_out_of_image_top_left(self):
        image = iarandom.RNG(1234).integers(0, 255, size=(10, 10, 3))

        bb = ia.BoundingBox(y1=-1, y2=3, x1=-1, x2=4)
        image_sub = bb.extract_from_image(image)

        image_pad = np.pad(
            image,
            ((1, 0), (1, 0), (0, 0)),
            mode="constant",
            constant_values=0)  # pad at top and left each 1px (black)
        assert np.array_equal(image_sub, image_pad[0:4, 0:5, :])

    def test_extract_from_image_float_coords(self):
        image = iarandom.RNG(1234).integers(0, 255, size=(10, 10, 3))

        bb = ia.BoundingBox(y1=1, y2=1.99999, x1=1, x2=1.99999)
        image_sub = bb.extract_from_image(image)

        assert np.array_equal(image_sub, image[1:1+1, 1:1+1, :])

    def test_extract_from_image_bb_height_is_zero(self):
        image = iarandom.RNG(1234).integers(0, 255, size=(10, 10, 3))

        bb = ia.BoundingBox(y1=1, y2=1, x1=2, x2=4)
        image_sub = bb.extract_from_image(image)

        assert np.array_equal(image_sub, image[1:1+1, 2:4, :])

    def test_extract_from_image_bb_width_is_zero(self):
        image = iarandom.RNG(1234).integers(0, 255, size=(10, 10, 3))

        bb = ia.BoundingBox(y1=1, y2=1, x1=2, x2=2)
        image_sub = bb.extract_from_image(image)

        assert np.array_equal(image_sub, image[1:1+1, 2:2+1, :])

    def test_to_keypoints(self):
        bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3)

        kps = bb.to_keypoints()

        assert len(kps) == 4
        assert kps[0].y == 1
        assert kps[0].x == 1
        assert kps[1].y == 1
        assert kps[1].x == 3
        assert kps[2].y == 3
        assert kps[2].x == 3
        assert kps[3].y == 3
        assert kps[3].x == 1

    def test_to_polygon(self):
        bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3)

        poly = bb.to_polygon()

        assert poly.coords_almost_equals([
            (1, 1),
            (3, 1),
            (3, 3,),
            (1, 3)
        ])

    def test_coords_almost_equals(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)
        other = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)

        equal = bb.coords_almost_equals(other)

        assert equal

    def test_coords_almost_equals__unequal(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)
        other = ia.BoundingBox(x1=1+1, y1=3+1, x2=1+1, y2=3+1)

        equal = bb.coords_almost_equals(other)

        assert not equal

    def test_coords_almost_equals__dist_below_max_distance(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)
        other = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3+1e-5)

        equal = bb.coords_almost_equals(other, max_distance=1e-4)

        assert equal

    def test_coords_almost_equals__dist_above_max_distance(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)
        other = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3+1e-3)

        equal = bb.coords_almost_equals(other, max_distance=1e-4)

        assert not equal

    def test_coords_almost_equals__input_is_array(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)
        other = np.float32([[1, 3], [1, 3]])

        equal = bb.coords_almost_equals(other)

        assert equal

    def test_coords_almost_equals__input_is_array_not_equal(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)
        other = np.float32([[1, 3], [1, 3+0.5]])

        equal = bb.coords_almost_equals(other)

        assert not equal

    def test_coords_almost_equals__input_is_list(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)
        other = [[1, 3], [1, 3]]

        equal = bb.coords_almost_equals(other)

        assert equal

    def test_coords_almost_equals__input_is_list_not_equal(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)
        other = [[1, 3], [1, 3+0.5]]

        equal = bb.coords_almost_equals(other)

        assert not equal

    def test_coords_almost_equals__bad_datatype(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)

        with self.assertRaises(ValueError) as cm:
            _ = bb.coords_almost_equals(False)

        assert "Expected 'other'" in str(cm.exception)

    @mock.patch("imgaug.augmentables.bbs.BoundingBox.coords_almost_equals")
    def test_almost_equals(self, mock_cae):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)
        other = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)

        equal = bb.almost_equals(other, max_distance=1)

        assert equal
        mock_cae.assert_called_once_with(other, max_distance=1)

    def test_almost_equals__labels_none_vs_string(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3, label="foo")
        other = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3)

        equal = bb.almost_equals(other)

        assert not equal

    def test_almost_equals__labels_different_strings(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3, label="foo")
        other = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3, label="bar")

        equal = bb.almost_equals(other)

        assert not equal

    def test_almost_equals__same_string(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3, label="foo")
        other = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3, label="foo")

        equal = bb.almost_equals(other)

        assert equal

    def test_almost_equals__distance_above_threshold(self):
        bb = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3, label="foo")
        other = ia.BoundingBox(x1=1, y1=3, x2=1, y2=3+1e-1, label="foo")

        equal = bb.almost_equals(other, max_distance=1e-2)

        assert not equal

    def test_from_point_soup__empty_list(self):
        with self.assertRaises(AssertionError) as ctx:
            _ = ia.BoundingBox.from_point_soup([])
        assert "Expected to get at least one point" in str(ctx.exception)

    def test_from_point_soup__empty_array(self):
        with self.assertRaises(AssertionError) as ctx:
            _ = ia.BoundingBox.from_point_soup(np.zeros((0, 2)))
        assert "Expected to get at least one point" in str(ctx.exception)

    def test_from_point_soup__list_with_single_point(self):
        points = [(1, 2)]
        bb = ia.BoundingBox.from_point_soup(points)
        assert bb.x1 == 1
        assert bb.y1 == 2
        assert bb.x2 == 1
        assert bb.y2 == 2

    def test_from_point_soup__list_with_single_point__single_level(self):
        points = [1, 2]
        bb = ia.BoundingBox.from_point_soup(points)
        assert bb.x1 == 1
        assert bb.y1 == 2
        assert bb.x2 == 1
        assert bb.y2 == 2

    def test_from_point_soup__list_with_two_points(self):
        points = [(1, 2), (3, 4)]
        bb = ia.BoundingBox.from_point_soup(points)
        assert bb.x1 == 1
        assert bb.y1 == 2
        assert bb.x2 == 3
        assert bb.y2 == 4

    def test_from_point_soup__list_with_three_points(self):
        points = [(1, 4), (3, 2), (15, 16)]
        bb = ia.BoundingBox.from_point_soup(points)
        assert bb.x1 == 1
        assert bb.y1 == 2
        assert bb.x2 == 15
        assert bb.y2 == 16

    def test_from_point_soup__array_with_single_point(self):
        points = np.float32([(1, 2)])
        bb = ia.BoundingBox.from_point_soup(points)
        assert bb.x1 == 1
        assert bb.y1 == 2
        assert bb.x2 == 1
        assert bb.y2 == 2

    def test_from_point_soup__array_with_single_point__single_level(self):
        points = np.float32([1, 2])
        bb = ia.BoundingBox.from_point_soup(points)
        assert bb.x1 == 1
        assert bb.y1 == 2
        assert bb.x2 == 1
        assert bb.y2 == 2

    def test_from_point_soup__array_with_two_points__single_level(self):
        points = np.float32([1, 2, 3, 4])
        bb = ia.BoundingBox.from_point_soup(points)
        assert bb.x1 == 1
        assert bb.y1 == 2
        assert bb.x2 == 3
        assert bb.y2 == 4

    def test_from_point_soup__array_with_two_points(self):
        points = np.float32([(1, 2), (3, 4)])
        bb = ia.BoundingBox.from_point_soup(points)
        assert bb.x1 == 1
        assert bb.y1 == 2
        assert bb.x2 == 3
        assert bb.y2 == 4

    def test_from_point_soup__array_with_three_points(self):
        points = np.float32([(1, 4), (3, 2), (15, 16)])
        bb = ia.BoundingBox.from_point_soup(points)
        assert bb.x1 == 1
        assert bb.y1 == 2
        assert bb.x2 == 15
        assert bb.y2 == 16

    def test_copy(self):
        bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label="test")

        bb2 = bb.copy()

        assert bb2.y1 == 1
        assert bb2.y2 == 3
        assert bb2.x1 == 1
        assert bb2.x2 == 3
        assert bb2.label == "test"

    def test_copy_and_replace_attributes(self):
        bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label="test")

        bb2 = bb.copy(y1=10, x1=20, y2=30, x2=40, label="test2")

        assert bb2.y1 == 10
        assert bb2.x1 == 20
        assert bb2.y2 == 30
        assert bb2.x2 == 40
        assert bb2.label == "test2"

    def test_deepcopy(self):
        bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=["test"])

        bb2 = bb.deepcopy()
        bb2.label[0] = "foo"

        assert bb2.y1 == 1
        assert bb2.y2 == 3
        assert bb2.x1 == 1
        assert bb2.x2 == 3
        assert bb2.label[0] == "foo"
        assert bb.label[0] == "test"

    def test_deepcopy_and_replace_attributes(self):
        bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label="test")

        bb2 = bb.deepcopy(y1=10, y2=30, x1=15, x2=35, label="asd")

        assert bb2.y1 == 10
        assert bb2.y2 == 30
        assert bb2.x1 == 15
        assert bb2.x2 == 35
        assert bb2.label == "asd"
        assert bb.label == "test"

    def test___getitem__(self):
        cba = ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)
        assert np.allclose(cba[0], (1, 2))
        assert np.allclose(cba[1], (3, 4))

    def test___iter__(self):
        cba = ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)
        for i, xy in enumerate(cba):
            assert i in [0, 1]
            if i == 0:
                assert np.allclose(xy, (1, 2))
            elif i == 1:
                assert np.allclose(xy, (3, 4))
        assert i == 1

    def test_string_conversion(self):
        bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3)
        assert (
            bb.__str__()
            == bb.__repr__()
            == "BoundingBox("
               "x1=1.0000, y1=1.0000, x2=3.0000, y2=3.0000, "
               "label=None)"
        )

    def test_string_conversion_with_label(self):
        bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label="foo")
        assert (
            bb.__str__()
            == bb.__repr__()
            == "BoundingBox("
               "x1=1.0000, y1=1.0000, x2=3.0000, y2=3.0000, "
               "label=foo)"
        )


class TestBoundingBoxesOnImage_items_setter(unittest.TestCase):
    def test_with_list_of_bounding_boxes(self):
        bbs = [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
               ia.BoundingBox(x1=3, y1=4, x2=5, y2=6)]
        bbsoi = ia.BoundingBoxesOnImage([], shape=(10, 20, 3))
        bbsoi.items = bbs
        assert np.all([
            (bb_i.x1 == bb_j.x1
             and bb_i.y1 == bb_j.y1
             and bb_i.x2 == bb_j.x2
             and bb_i.y2 == bb_j.y2)
            for bb_i, bb_j
            in zip(bbsoi.bounding_boxes, bbs)
        ])


class TestBoundingBoxesOnImage_on_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, cbaoi, *args, **kwargs):
        return cbaoi.on_(*args, **kwargs)

    def test_on_same_height_width(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_projected = self._func(bbsoi, (40, 50))

        assert bbsoi_projected.bounding_boxes[0].y1 == 10
        assert bbsoi_projected.bounding_boxes[0].x1 == 20
        assert bbsoi_projected.bounding_boxes[0].y2 == 30
        assert bbsoi_projected.bounding_boxes[0].x2 == 40
        assert bbsoi_projected.bounding_boxes[1].y1 == 15
        assert bbsoi_projected.bounding_boxes[1].x1 == 25
        assert bbsoi_projected.bounding_boxes[1].y2 == 35
        assert bbsoi_projected.bounding_boxes[1].x2 == 45
        assert bbsoi_projected.shape == (40, 50)

    def test_on_upscaled_by_2(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_projected = self._func(bbsoi, (40*2, 50*2, 3))

        assert bbsoi_projected.bounding_boxes[0].y1 == 10*2
        assert bbsoi_projected.bounding_boxes[0].x1 == 20*2
        assert bbsoi_projected.bounding_boxes[0].y2 == 30*2
        assert bbsoi_projected.bounding_boxes[0].x2 == 40*2
        assert bbsoi_projected.bounding_boxes[1].y1 == 15*2
        assert bbsoi_projected.bounding_boxes[1].x1 == 25*2
        assert bbsoi_projected.bounding_boxes[1].y2 == 35*2
        assert bbsoi_projected.bounding_boxes[1].x2 == 45*2
        assert bbsoi_projected.shape == (40*2, 50*2, 3)

    def test_on_upscaled_by_2_with_shape_given_as_array(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_projected = self._func(bbsoi, np.zeros((40*2, 50*2, 3), dtype=np.uint8))

        assert bbsoi_projected.bounding_boxes[0].y1 == 10*2
        assert bbsoi_projected.bounding_boxes[0].x1 == 20*2
        assert bbsoi_projected.bounding_boxes[0].y2 == 30*2
        assert bbsoi_projected.bounding_boxes[0].x2 == 40*2
        assert bbsoi_projected.bounding_boxes[1].y1 == 15*2
        assert bbsoi_projected.bounding_boxes[1].x1 == 25*2
        assert bbsoi_projected.bounding_boxes[1].y2 == 35*2
        assert bbsoi_projected.bounding_boxes[1].x2 == 45*2
        assert bbsoi_projected.shape == (40*2, 50*2, 3)

    def test_inplaceness(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi2 = self._func(bbsoi, (40, 50))

        if self._is_inplace:
            assert bbsoi2 is bbsoi
        else:
            assert bbsoi2 is not bbsoi


class TestBoundingBoxesOnImage_on(TestBoundingBoxesOnImage_on_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, cbaoi, *args, **kwargs):
        return cbaoi.on(*args, **kwargs)


class TestBoundingBoxesOnImage_clip_out_of_image_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, cbaoi, *args, **kwargs):
        return cbaoi.clip_out_of_image_()

    def test_clip_out_of_image(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_clip = self._func(bbsoi)

        assert len(bbsoi_clip.bounding_boxes) == 2
        assert bbsoi_clip.bounding_boxes[0].y1 == 10
        assert bbsoi_clip.bounding_boxes[0].x1 == 20
        assert bbsoi_clip.bounding_boxes[0].y2 == 30
        assert bbsoi_clip.bounding_boxes[0].x2 == 40
        assert bbsoi_clip.bounding_boxes[1].y1 == 15
        assert bbsoi_clip.bounding_boxes[1].x1 == 25
        assert bbsoi_clip.bounding_boxes[1].y2 == 35
        assert np.isclose(bbsoi_clip.bounding_boxes[1].x2, 50)

    def test_inplaceness(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi2 = self._func(bbsoi, (40, 50))

        if self._is_inplace:
            assert bbsoi2 is bbsoi
        else:
            assert bbsoi2 is not bbsoi


class TestBoundingBoxesOnImage_clip_out_of_image(TestBoundingBoxesOnImage_clip_out_of_image_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, cbaoi, *args, **kwargs):
        return cbaoi.clip_out_of_image()


class TestBoundingBoxesOnImage(unittest.TestCase):
    def test___init__(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
        assert bbsoi.bounding_boxes == [bb1, bb2]
        assert bbsoi.shape == (40, 50, 3)

    def test___init___array_as_shape(self):
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        with assertWarns(self, ia.DeprecationWarning):
            bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=image)
        assert bbsoi.bounding_boxes == [bb1, bb2]
        assert bbsoi.shape == (40, 50, 3)

    def test_items(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        items = bbsoi.items

        assert items == [bb1, bb2]

    def test_items_empty(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(40, 50, 3))

        items = bbsoi.items

        assert items == []

    def test_height(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
        assert bbsoi.height == 40

    def test_width(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
        assert bbsoi.width == 50

    def test_empty_when_bbs_not_empty(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bbsoi = ia.BoundingBoxesOnImage([bb], shape=(40, 50, 3))
        assert not bbsoi.empty

    def test_empty_when_bbs_actually_empty(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(40, 50, 3))
        assert bbsoi.empty

    def test_from_xyxy_array_float(self):
        xyxy = np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0]
        ])

        bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy, shape=(40, 50, 3))

        assert len(bbsoi.bounding_boxes) == 2
        assert np.allclose(bbsoi.bounding_boxes[0].x1, 0.0)
        assert np.allclose(bbsoi.bounding_boxes[0].y1, 0.0)
        assert np.allclose(bbsoi.bounding_boxes[0].x2, 1.0)
        assert np.allclose(bbsoi.bounding_boxes[0].y2, 1.0)
        assert np.allclose(bbsoi.bounding_boxes[1].x1, 1.0)
        assert np.allclose(bbsoi.bounding_boxes[1].y1, 2.0)
        assert np.allclose(bbsoi.bounding_boxes[1].x2, 3.0)
        assert np.allclose(bbsoi.bounding_boxes[1].y2, 4.0)
        assert bbsoi.shape == (40, 50, 3)

    def test_from_xyxy_array_float_3d(self):
        xyxy = np.float32([
            [
                [0.0, 0.0],
                [1.0, 1.0]
            ],
            [
                [1.0, 2.0],
                [3.0, 4.0]
            ]
        ])

        bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy, shape=(40, 50, 3))

        assert len(bbsoi.bounding_boxes) == 2
        assert np.allclose(bbsoi.bounding_boxes[0].x1, 0.0)
        assert np.allclose(bbsoi.bounding_boxes[0].y1, 0.0)
        assert np.allclose(bbsoi.bounding_boxes[0].x2, 1.0)
        assert np.allclose(bbsoi.bounding_boxes[0].y2, 1.0)
        assert np.allclose(bbsoi.bounding_boxes[1].x1, 1.0)
        assert np.allclose(bbsoi.bounding_boxes[1].y1, 2.0)
        assert np.allclose(bbsoi.bounding_boxes[1].x2, 3.0)
        assert np.allclose(bbsoi.bounding_boxes[1].y2, 4.0)
        assert bbsoi.shape == (40, 50, 3)

    def test_from_xyxy_array_int32(self):
        xyxy = np.int32([
            [0, 0, 1, 1],
            [1, 2, 3, 4]
        ])

        bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy, shape=(40, 50, 3))

        assert len(bbsoi.bounding_boxes) == 2
        assert np.allclose(bbsoi.bounding_boxes[0].x1, 0.0)
        assert np.allclose(bbsoi.bounding_boxes[0].y1, 0.0)
        assert np.allclose(bbsoi.bounding_boxes[0].x2, 1.0)
        assert np.allclose(bbsoi.bounding_boxes[0].y2, 1.0)
        assert np.allclose(bbsoi.bounding_boxes[1].x1, 1.0)
        assert np.allclose(bbsoi.bounding_boxes[1].y1, 2.0)
        assert np.allclose(bbsoi.bounding_boxes[1].x2, 3.0)
        assert np.allclose(bbsoi.bounding_boxes[1].y2, 4.0)
        assert bbsoi.shape == (40, 50, 3)

    def test_from_xyxy_array_empty_array(self):
        xyxy = np.zeros((0, 4), dtype=np.float32)

        bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy, shape=(40, 50, 3))

        assert len(bbsoi.bounding_boxes) == 0
        assert bbsoi.shape == (40, 50, 3)

    def test_from_point_soups__2d_array(self):
        xy = np.float32([
            [7, 3,
             11, 5,
             1, 7,
             12, 19]
        ])

        bbsoi = ia.BoundingBoxesOnImage.from_point_soups(
            xy, shape=(40, 50, 3))

        assert len(bbsoi.bounding_boxes) == 1
        assert bbsoi.bounding_boxes[0].x1 == 1
        assert bbsoi.bounding_boxes[0].y1 == 3
        assert bbsoi.bounding_boxes[0].x2 == 12
        assert bbsoi.bounding_boxes[0].y2 == 19
        assert bbsoi.shape == (40, 50, 3)

    def test_from_point_soups__3d_array(self):
        xy = np.float32([
            [
                [7, 3],
                [11, 5],
                [1, 7],
                [12, 19]
            ]
        ])

        bbsoi = ia.BoundingBoxesOnImage.from_point_soups(
            xy, shape=(40, 50, 3))

        assert len(bbsoi.bounding_boxes) == 1
        assert bbsoi.bounding_boxes[0].x1 == 1
        assert bbsoi.bounding_boxes[0].y1 == 3
        assert bbsoi.bounding_boxes[0].x2 == 12
        assert bbsoi.bounding_boxes[0].y2 == 19
        assert bbsoi.shape == (40, 50, 3)

    def test_from_point_soups__2d_list(self):
        xy = [
            [7, 3,
             11, 5,
             1, 7,
             12, 19]
        ]

        bbsoi = ia.BoundingBoxesOnImage.from_point_soups(
            xy, shape=(40, 50, 3))

        assert len(bbsoi.bounding_boxes) == 1
        assert bbsoi.bounding_boxes[0].x1 == 1
        assert bbsoi.bounding_boxes[0].y1 == 3
        assert bbsoi.bounding_boxes[0].x2 == 12
        assert bbsoi.bounding_boxes[0].y2 == 19
        assert bbsoi.shape == (40, 50, 3)

    def test_from_point_soups__empty_array(self):
        xy = np.zeros((0, 4), dtype=np.float32)

        bbsoi = ia.BoundingBoxesOnImage.from_point_soups(
            xy, shape=(40, 50, 3))

        assert len(bbsoi.bounding_boxes) == 0
        assert bbsoi.shape == (40, 50, 3)

    def test_from_point_soups__empty_list(self):
        xy = []

        bbsoi = ia.BoundingBoxesOnImage.from_point_soups(
            xy, shape=(40, 50, 3))

        assert len(bbsoi.bounding_boxes) == 0
        assert bbsoi.shape == (40, 50, 3)

    def test_to_xyxy_array(self):
        xyxy = np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0]
        ])
        bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy, shape=(40, 50, 3))

        xyxy_out = bbsoi.to_xyxy_array()

        assert np.allclose(xyxy, xyxy_out)
        assert xyxy_out.dtype.name == "float32"

    def test_to_xyxy_array_convert_to_int32(self):
        xyxy = np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0]
        ])
        bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy, shape=(40, 50, 3))

        xyxy_out = bbsoi.to_xyxy_array(dtype=np.int32)

        assert np.allclose(xyxy.astype(np.int32), xyxy_out)
        assert xyxy_out.dtype.name == "int32"

    def test_to_xyxy_array_no_bbs_to_convert(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(40, 50, 3))

        xyxy_out = bbsoi.to_xyxy_array(dtype=np.int32)

        assert xyxy_out.shape == (0, 4)

    def test_to_xy_array(self):
        xyxy = np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0]
        ])
        bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy, shape=(40, 50, 3))

        xy_out = bbsoi.to_xy_array()

        expected = np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        assert xy_out.shape == (4, 2)
        assert np.allclose(xy_out, expected)
        assert xy_out.dtype.name == "float32"

    def test_to_xy_array__empty_instance(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(1, 2, 3))

        xy_out = bbsoi.to_xy_array()

        assert xy_out.shape == (0, 2)
        assert xy_out.dtype.name == "float32"

    def test_fill_from_xyxy_array___empty_array(self):
        xyxy = np.zeros((0, 4), dtype=np.float32)
        bbsoi = ia.BoundingBoxesOnImage([], shape=(2, 2, 3))

        bbsoi = bbsoi.fill_from_xyxy_array_(xyxy)

        assert len(bbsoi.bounding_boxes) == 0

    def test_fill_from_xyxy_array___empty_list(self):
        xyxy = []
        bbsoi = ia.BoundingBoxesOnImage([], shape=(2, 2, 3))

        bbsoi = bbsoi.fill_from_xyxy_array_(xyxy)

        assert len(bbsoi.bounding_boxes) == 0

    def test_fill_from_xyxy_array___array_with_two_coords(self):
        xyxy = np.array(
            [(100, 101, 102, 103),
             (200, 201, 202, 203)], dtype=np.float32)
        bbsoi = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(1, 2, 3, 4),
             ia.BoundingBox(10, 20, 30, 40)],
            shape=(2, 2, 3))

        bbsoi = bbsoi.fill_from_xyxy_array_(xyxy)

        assert len(bbsoi.bounding_boxes) == 2
        assert bbsoi.bounding_boxes[0].x1 == 100
        assert bbsoi.bounding_boxes[0].y1 == 101
        assert bbsoi.bounding_boxes[0].x2 == 102
        assert bbsoi.bounding_boxes[0].y2 == 103
        assert bbsoi.bounding_boxes[1].x1 == 200
        assert bbsoi.bounding_boxes[1].y1 == 201
        assert bbsoi.bounding_boxes[1].x2 == 202
        assert bbsoi.bounding_boxes[1].y2 == 203

    def test_fill_from_xyxy_array___list_with_two_coords(self):
        xyxy = [(100, 101, 102, 103),
                (200, 201, 202, 203)]
        bbsoi = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(1, 2, 3, 4),
             ia.BoundingBox(10, 20, 30, 40)],
            shape=(2, 2, 3))

        bbsoi = bbsoi.fill_from_xyxy_array_(xyxy)

        assert len(bbsoi.bounding_boxes) == 2
        assert bbsoi.bounding_boxes[0].x1 == 100
        assert bbsoi.bounding_boxes[0].y1 == 101
        assert bbsoi.bounding_boxes[0].x2 == 102
        assert bbsoi.bounding_boxes[0].y2 == 103
        assert bbsoi.bounding_boxes[1].x1 == 200
        assert bbsoi.bounding_boxes[1].y1 == 201
        assert bbsoi.bounding_boxes[1].x2 == 202
        assert bbsoi.bounding_boxes[1].y2 == 203

    def test_fill_from_xy_array___empty_array(self):
        xy = np.zeros((0, 2), dtype=np.float32)
        bbsoi = ia.BoundingBoxesOnImage([], shape=(2, 2, 3))

        bbsoi = bbsoi.fill_from_xy_array_(xy)

        assert len(bbsoi.bounding_boxes) == 0

    def test_fill_from_xy_array___empty_list(self):
        xy = []
        bbsoi = ia.BoundingBoxesOnImage([], shape=(2, 2, 3))

        bbsoi = bbsoi.fill_from_xy_array_(xy)

        assert len(bbsoi.bounding_boxes) == 0

    def test_fill_from_xy_array___array_with_two_coords(self):
        xy = np.array(
            [(100, 101),
             (102, 103),
             (200, 201),
             (202, 203)], dtype=np.float32)
        bbsoi = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(1, 2, 3, 4),
             ia.BoundingBox(10, 20, 30, 40)],
            shape=(2, 2, 3))

        bbsoi = bbsoi.fill_from_xy_array_(xy)

        assert len(bbsoi.bounding_boxes) == 2
        assert bbsoi.bounding_boxes[0].x1 == 100
        assert bbsoi.bounding_boxes[0].y1 == 101
        assert bbsoi.bounding_boxes[0].x2 == 102
        assert bbsoi.bounding_boxes[0].y2 == 103
        assert bbsoi.bounding_boxes[1].x1 == 200
        assert bbsoi.bounding_boxes[1].y1 == 201
        assert bbsoi.bounding_boxes[1].x2 == 202
        assert bbsoi.bounding_boxes[1].y2 == 203

    def test_fill_from_xy_array___list_with_two_coords(self):
        xy = [(100, 101),
              (102, 103),
              (200, 201),
              (202, 203)]
        bbsoi = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(1, 2, 3, 4),
             ia.BoundingBox(10, 20, 30, 40)],
            shape=(2, 2, 3))

        bbsoi = bbsoi.fill_from_xy_array_(xy)

        assert len(bbsoi.bounding_boxes) == 2
        assert bbsoi.bounding_boxes[0].x1 == 100
        assert bbsoi.bounding_boxes[0].y1 == 101
        assert bbsoi.bounding_boxes[0].x2 == 102
        assert bbsoi.bounding_boxes[0].y2 == 103
        assert bbsoi.bounding_boxes[1].x1 == 200
        assert bbsoi.bounding_boxes[1].y1 == 201
        assert bbsoi.bounding_boxes[1].x2 == 202
        assert bbsoi.bounding_boxes[1].y2 == 203

    def test_draw_on_image(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
        image = np.zeros(bbsoi.shape, dtype=np.uint8)

        image_drawn = bbsoi.draw_on_image(
            image,
            color=[0, 255, 0], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_drawn[10-1, 20-1, :] == [0, 0, 0])
        assert np.all(image_drawn[10-1, 20-0, :] == [0, 0, 0])
        assert np.all(image_drawn[10-0, 20-1, :] == [0, 0, 0])
        assert np.all(image_drawn[10-0, 20-0, :] == [0, 255, 0])
        assert np.all(image_drawn[10+1, 20+1, :] == [0, 0, 0])

        assert np.all(image_drawn[30-1, 40-1, :] == [0, 0, 0])
        assert np.all(image_drawn[30+1, 40-0, :] == [0, 0, 0])
        assert np.all(image_drawn[30+0, 40+1, :] == [0, 0, 0])
        assert np.all(image_drawn[30+0, 40+0, :] == [0, 255, 0])
        assert np.all(image_drawn[30+1, 40+1, :] == [0, 0, 0])

        assert np.all(image_drawn[15-1, 25-1, :] == [0, 0, 0])
        assert np.all(image_drawn[15-1, 25-0, :] == [0, 0, 0])
        assert np.all(image_drawn[15-0, 25-1, :] == [0, 0, 0])
        assert np.all(image_drawn[15-0, 25-0, :] == [0, 255, 0])
        assert np.all(image_drawn[15+1, 25+1, :] == [0, 0, 0])

        assert np.all(image_drawn[35-1, 45-1, :] == [0, 0, 0])
        assert np.all(image_drawn[35+1, 45+0, :] == [0, 0, 0])
        assert np.all(image_drawn[35+0, 45+1, :] == [0, 0, 0])
        assert np.all(image_drawn[35+0, 45+0, :] == [0, 255, 0])
        assert np.all(image_drawn[35+1, 45+1, :] == [0, 0, 0])

    def test_remove_out_of_image_(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_removed = bbsoi.remove_out_of_image_(fully=True, partly=True)

        assert len(bbsoi_removed.bounding_boxes) == 1
        assert bbsoi_removed.bounding_boxes[0] == bb1
        assert bbsoi_removed is bbsoi

    def test_remove_out_of_image(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_removed = bbsoi.remove_out_of_image(fully=True, partly=True)

        assert len(bbsoi_removed.bounding_boxes) == 1
        assert bbsoi_removed.bounding_boxes[0] == bb1
        assert bbsoi_removed is not bbsoi

    def test_remove_out_of_image_fraction_(self):
        item1 = ia.BoundingBox(y1=1, x1=5, y2=6, x2=9)
        item2 = ia.BoundingBox(y1=1, x1=5, y2=6, x2=15)
        item3 = ia.BoundingBox(y1=1, x1=15, y2=6, x2=25)
        cbaoi = ia.BoundingBoxesOnImage([item1, item2, item3],
                                        shape=(10, 10, 3))

        cbaoi_reduced = cbaoi.remove_out_of_image_fraction_(0.6)

        assert len(cbaoi_reduced.items) == 2
        assert cbaoi_reduced.items == [item1, item2]
        assert cbaoi_reduced is cbaoi

    def test_remove_out_of_image_fraction(self):
        item1 = ia.BoundingBox(y1=1, x1=5, y2=6, x2=9)
        item2 = ia.BoundingBox(y1=1, x1=5, y2=6, x2=15)
        item3 = ia.BoundingBox(y1=1, x1=15, y2=6, x2=25)
        cbaoi = ia.BoundingBoxesOnImage([item1, item2, item3],
                                        shape=(10, 10, 3))

        cbaoi_reduced = cbaoi.remove_out_of_image_fraction(0.6)

        assert len(cbaoi_reduced.items) == 2
        assert cbaoi_reduced.items == [item1, item2]
        assert cbaoi_reduced is not cbaoi

    def test_shift_(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_shifted = bbsoi.shift_(y=2)

        assert len(bbsoi_shifted.bounding_boxes) == 2
        assert bbsoi_shifted.bounding_boxes[0].y1 == 10 + 2
        assert bbsoi_shifted.bounding_boxes[0].x1 == 20
        assert bbsoi_shifted.bounding_boxes[0].y2 == 30 + 2
        assert bbsoi_shifted.bounding_boxes[0].x2 == 40
        assert bbsoi_shifted.bounding_boxes[1].y1 == 15 + 2
        assert bbsoi_shifted.bounding_boxes[1].x1 == 25
        assert bbsoi_shifted.bounding_boxes[1].y2 == 35 + 2
        assert bbsoi_shifted.bounding_boxes[1].x2 == 51
        assert bbsoi_shifted is bbsoi

    def test_shift(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_shifted = bbsoi.shift(y=2)

        assert len(bbsoi_shifted.bounding_boxes) == 2
        assert bbsoi_shifted.bounding_boxes[0].y1 == 10 + 2
        assert bbsoi_shifted.bounding_boxes[0].x1 == 20
        assert bbsoi_shifted.bounding_boxes[0].y2 == 30 + 2
        assert bbsoi_shifted.bounding_boxes[0].x2 == 40
        assert bbsoi_shifted.bounding_boxes[1].y1 == 15 + 2
        assert bbsoi_shifted.bounding_boxes[1].x1 == 25
        assert bbsoi_shifted.bounding_boxes[1].y2 == 35 + 2
        assert bbsoi_shifted.bounding_boxes[1].x2 == 51
        assert bbsoi_shifted is not bbsoi

    def test_shift__deprecated_args(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            bbsoi_shifted = bbsoi.shift(right=1)

            assert len(bbsoi_shifted.bounding_boxes) == 2
            assert bbsoi_shifted.bounding_boxes[0].y1 == 10
            assert bbsoi_shifted.bounding_boxes[0].x1 == 20 - 1
            assert bbsoi_shifted.bounding_boxes[0].y2 == 30
            assert bbsoi_shifted.bounding_boxes[0].x2 == 40 - 1
            assert bbsoi_shifted.bounding_boxes[1].y1 == 15
            assert bbsoi_shifted.bounding_boxes[1].x1 == 25 - 1
            assert bbsoi_shifted.bounding_boxes[1].y2 == 35
            assert bbsoi_shifted.bounding_boxes[1].x2 == 51 - 1
            assert bbsoi_shifted is not bbsoi

            assert (
                "These are deprecated. Use `x` and `y` instead."
                in str(caught_warnings[-1].message)
            )

    def test_to_keypoints_on_image(self):
        bbsoi = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(0, 1, 2, 3),
             ia.BoundingBox(10, 20, 30, 40)],
            shape=(1, 2, 3))

        kpsoi = bbsoi.to_keypoints_on_image()

        assert len(kpsoi.keypoints) == 2*4

        assert kpsoi.keypoints[0].x == 0
        assert kpsoi.keypoints[0].y == 1
        assert kpsoi.keypoints[1].x == 2
        assert kpsoi.keypoints[1].y == 1
        assert kpsoi.keypoints[2].x == 2
        assert kpsoi.keypoints[2].y == 3
        assert kpsoi.keypoints[3].x == 0
        assert kpsoi.keypoints[3].y == 3

        assert kpsoi.keypoints[4].x == 10
        assert kpsoi.keypoints[4].y == 20
        assert kpsoi.keypoints[5].x == 30
        assert kpsoi.keypoints[5].y == 20
        assert kpsoi.keypoints[6].x == 30
        assert kpsoi.keypoints[6].y == 40
        assert kpsoi.keypoints[7].x == 10
        assert kpsoi.keypoints[7].y == 40

    def test_to_keypoints_on_image__empty_instance(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(1, 2, 3))

        kpsoi = bbsoi.to_keypoints_on_image()

        assert len(kpsoi.keypoints) == 0

    def test_invert_to_keypoints_on_image_(self):
        bbsoi = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(0, 1, 2, 3),
             ia.BoundingBox(10, 20, 30, 40)],
            shape=(1, 2, 3))
        kpsoi = ia.KeypointsOnImage(
            [ia.Keypoint(100, 101), ia.Keypoint(102, 103),
             ia.Keypoint(104, 105), ia.Keypoint(106, 107),
             ia.Keypoint(110, 120), ia.Keypoint(130, 140),
             ia.Keypoint(150, 160), ia.Keypoint(170, 180)],
            shape=(10, 20, 30))

        bbsoi_inv = bbsoi.invert_to_keypoints_on_image_(kpsoi)

        assert len(bbsoi_inv.bounding_boxes) == 2
        assert bbsoi_inv.shape == (10, 20, 30)
        assert bbsoi_inv.bounding_boxes[0].x1 == 100
        assert bbsoi_inv.bounding_boxes[0].y1 == 101
        assert bbsoi_inv.bounding_boxes[0].x2 == 106
        assert bbsoi_inv.bounding_boxes[0].y2 == 107
        assert bbsoi_inv.bounding_boxes[1].x1 == 110
        assert bbsoi_inv.bounding_boxes[1].y1 == 120
        assert bbsoi_inv.bounding_boxes[1].x2 == 170
        assert bbsoi_inv.bounding_boxes[1].y2 == 180

    def test_invert_to_keypoints_on_image___empty_instance(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(1, 2, 3))
        kpsoi = ia.KeypointsOnImage([], shape=(10, 20, 30))

        bbsoi_inv = bbsoi.invert_to_keypoints_on_image_(kpsoi)

        assert len(bbsoi_inv.bounding_boxes) == 0
        assert bbsoi_inv.shape == (10, 20, 30)

    def test_to_polygons_on_image(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        psoi = bbsoi.to_polygons_on_image()

        assert psoi.shape == (40, 50, 3)
        assert len(psoi.items) == 2
        assert psoi.items[0].coords_almost_equals([
            (20, 10),
            (40, 10),
            (40, 30),
            (20, 30)
        ])
        assert psoi.items[1].coords_almost_equals([
            (25, 15),
            (51, 15),
            (51, 35),
            (25, 35)
        ])

    def test_to_polygons_on_image__empty_instance(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(40, 50, 3))

        psoi = bbsoi.to_polygons_on_image()

        assert psoi.shape == (40, 50, 3)
        assert len(psoi.items) == 0

    def test_copy(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_copy = bbsoi.copy()

        assert len(bbsoi.bounding_boxes) == 2
        assert bbsoi_copy.bounding_boxes[0].y1 == 10
        assert bbsoi_copy.bounding_boxes[0].x1 == 20
        assert bbsoi_copy.bounding_boxes[0].y2 == 30
        assert bbsoi_copy.bounding_boxes[0].x2 == 40
        assert bbsoi_copy.bounding_boxes[1].y1 == 15
        assert bbsoi_copy.bounding_boxes[1].x1 == 25
        assert bbsoi_copy.bounding_boxes[1].y2 == 35
        assert bbsoi_copy.bounding_boxes[1].x2 == 51

        bbsoi_copy.bounding_boxes[0].y1 = 0
        assert bbsoi.bounding_boxes[0].y1 == 0
        assert bbsoi_copy.bounding_boxes[0].y1 == 0

    def test_copy_bounding_boxes_set(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bb3 = ia.BoundingBox(y1=15+1, x1=25+1, y2=35+1, x2=51+1)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_copy = bbsoi.copy(bounding_boxes=[bb3])

        assert bbsoi_copy is not bbsoi
        assert bbsoi_copy.shape == (40, 50, 3)
        assert bbsoi_copy.bounding_boxes == [bb3]

    def test_copy_shape_set(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_copy = bbsoi.copy(shape=(40+1, 50+1, 3))

        assert bbsoi_copy is not bbsoi
        assert bbsoi_copy.shape == (40+1, 50+1, 3)
        assert bbsoi_copy.bounding_boxes == [bb1, bb2]

    def test_deepcopy(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_copy = bbsoi.deepcopy()

        assert len(bbsoi.bounding_boxes) == 2
        assert bbsoi_copy.bounding_boxes[0].y1 == 10
        assert bbsoi_copy.bounding_boxes[0].x1 == 20
        assert bbsoi_copy.bounding_boxes[0].y2 == 30
        assert bbsoi_copy.bounding_boxes[0].x2 == 40
        assert bbsoi_copy.bounding_boxes[1].y1 == 15
        assert bbsoi_copy.bounding_boxes[1].x1 == 25
        assert bbsoi_copy.bounding_boxes[1].y2 == 35
        assert bbsoi_copy.bounding_boxes[1].x2 == 51

        bbsoi_copy.bounding_boxes[0].y1 = 0
        assert bbsoi.bounding_boxes[0].y1 == 10
        assert bbsoi_copy.bounding_boxes[0].y1 == 0

    def test_deepcopy_bounding_boxes_set(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bb3 = ia.BoundingBox(y1=15+1, x1=25+1, y2=35+1, x2=51+1)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_copy = bbsoi.deepcopy(bounding_boxes=[bb3])

        assert bbsoi_copy is not bbsoi
        assert bbsoi_copy.shape == (40, 50, 3)
        assert bbsoi_copy.bounding_boxes == [bb3]

    def test_deepcopy_shape_set(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_copy = bbsoi.deepcopy(shape=(40+1, 50+1, 3))

        assert bbsoi_copy is not bbsoi
        assert bbsoi_copy.shape == (40+1, 50+1, 3)
        assert len(bbsoi_copy.bounding_boxes) == 2
        assert bbsoi_copy.bounding_boxes[0].coords_almost_equals(bb1)
        assert bbsoi_copy.bounding_boxes[1].coords_almost_equals(bb2)

    def test___getitem__(self):
        cbas = [
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
            ia.BoundingBox(x1=2, y1=3, x2=4, y2=5)
        ]
        cbasoi = ia.BoundingBoxesOnImage(cbas, shape=(3, 4, 3))

        assert cbasoi[0] is cbas[0]
        assert cbasoi[1] is cbas[1]
        assert cbasoi[0:2] == cbas

    def test___iter__(self):
        cbas = [ia.BoundingBox(x1=0, y1=0, x2=2, y2=2),
                ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]
        cbasoi = ia.BoundingBoxesOnImage(cbas, shape=(40, 50, 3))

        for i, cba in enumerate(cbasoi):
            assert cba is cbas[i]

    def test___iter___empty(self):
        cbasoi = ia.BoundingBoxesOnImage([], shape=(40, 50, 3))
        i = 0
        for _cba in cbasoi:
            i += 1
        assert i == 0

    def test___len__(self):
        cbas = [ia.BoundingBox(x1=0, y1=0, x2=2, y2=2),
                ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]
        cbasoi = ia.BoundingBoxesOnImage(cbas, shape=(40, 50, 3))
        assert len(cbasoi) == 2

    def test_string_conversion(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bb1_expected = "BoundingBox(x1=20.0000, y1=10.0000, " \
                       "x2=40.0000, y2=30.0000, label=None)"
        bb2_expected = "BoundingBox(x1=25.0000, y1=15.0000, " \
                       "x2=51.0000, y2=35.0000, label=None)"
        expected = "BoundingBoxesOnImage([%s, %s], shape=(40, 50, 3))" % (
            bb1_expected, bb2_expected)
        assert (
            bbsoi.__repr__()
            == bbsoi.__str__()
            == expected
        )

    def test_string_conversion_labels_are_not_none(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label="foo")
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label="bar")
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bb1_expected = "BoundingBox(x1=20.0000, y1=10.0000, " \
                       "x2=40.0000, y2=30.0000, label=foo)"
        bb2_expected = "BoundingBox(x1=25.0000, y1=15.0000, " \
                       "x2=51.0000, y2=35.0000, label=bar)"
        expected = "BoundingBoxesOnImage([%s, %s], shape=(40, 50, 3))" % (
            bb1_expected, bb2_expected)
        assert (
            bbsoi.__repr__()
            == bbsoi.__str__()
            == expected
        )


class Test_LabelOnImageDrawer(unittest.TestCase):
    def test_draw_on_image_(self):
        height = 30
        image = np.full((100, 50, 3), 100, dtype=np.uint8)
        bb = ia.BoundingBox(x1=5, x2=20, y1=50, y2=60)
        drawer = _LabelOnImageDrawer(color_text=(255, 255, 255),
                                     color_bg=(0, 0, 0),
                                     height=height)

        image_drawn = drawer.draw_on_image_(np.copy(image), bb)

        frac_colors_as_expected = np.average(
            np.logical_or(image_drawn[50-1-height:50-1, 5-1:20+1, :] == 0,
                          image_drawn[50-1-height:50-1, 5-1:20+1, :] == 255)
        )
        assert np.all(image_drawn[:50-1-height, :, :] == 100)
        assert np.all(image_drawn[50-1:, :, :] == 100)
        assert np.all(image_drawn[:, :5-1, :] == 100)
        assert np.all(image_drawn[:, 20+1:, :] == 100)
        assert frac_colors_as_expected > 0.75

    def test_draw_on_image(self):
        image = np.full((20, 30, 3), 100, dtype=np.uint8)
        bb = ia.BoundingBox(x1=1, x2=6, y1=2, y2=10)
        drawer = _LabelOnImageDrawer(color_text=(255, 255, 255),
                                     color_bg=(0, 0, 0))

        image_drawn_inplace = drawer.draw_on_image_(np.copy(image), bb)
        image_drawn = drawer.draw_on_image_(image, bb)

        assert np.array_equal(image_drawn, image_drawn_inplace)

    def test__do_raise_if_out_of_image__bb_is_fully_inside(self):
        drawer = _LabelOnImageDrawer(raise_if_out_of_image=True)
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        bb = ia.BoundingBox(x1=1, x2=6, y1=2, y2=10)

        # assert no exception
        drawer._do_raise_if_out_of_image(image, bb)

    def test__do_raise_if_out_of_image__bb_is_partially_outside(self):
        drawer = _LabelOnImageDrawer(raise_if_out_of_image=True)
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        bb = ia.BoundingBox(x1=30-5, x2=30+1, y1=2, y2=10)

        # assert no exception
        drawer._do_raise_if_out_of_image(image, bb)

    def test__do_raise_if_out_of_image__bb_is_fully_outside(self):
        drawer = _LabelOnImageDrawer(raise_if_out_of_image=True)
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        bb = ia.BoundingBox(x1=30+1, x2=30+6, y1=2, y2=10)

        with self.assertRaises(Exception):
            drawer._do_raise_if_out_of_image(image, bb)

    def test__preprocess_colors__only_main_color_set(self):
        drawer = _LabelOnImageDrawer(color=(0, 255, 0))
        color_text, color_bg = drawer._preprocess_colors()
        assert np.array_equal(color_text, [0, 0, 0])
        assert np.array_equal(color_bg, [0, 255, 0])

    def test__preprocess_colors__subcolors_set(self):
        drawer = _LabelOnImageDrawer(color_text=(128, 129, 130),
                                     color_bg=(131, 132, 133))
        color_text, color_bg = drawer._preprocess_colors()
        assert np.array_equal(color_text, [128, 129, 130])
        assert np.array_equal(color_bg, [131, 132, 133])

    def test__preprocess_colors__text_not_set_must_be_black(self):
        drawer = _LabelOnImageDrawer(color=(255, 255, 255),
                                     color_bg=(255, 255, 255))
        color_text, color_bg = drawer._preprocess_colors()
        assert np.array_equal(color_text, [0, 0, 0])
        assert np.array_equal(color_bg, [255, 255, 255])

    def test__compute_bg_corner_coords__standard_bb(self):
        height = 30
        for size in [1, 2]:
            with self.subTest(size=size):
                drawer = _LabelOnImageDrawer(size=size, height=height)
                bb = ia.BoundingBox(x1=10, x2=30, y1=60, y2=90)
                image = np.zeros((100, 200, 3), dtype=np.uint8)
                x1, y1, x2, y2 = drawer._compute_bg_corner_coords(image, bb)
                assert np.isclose(x1, max(bb.x1 - size + 1, 0))
                assert np.isclose(y1, max(bb.y1 - 1 - height, 0))
                assert np.isclose(x2, min(bb.x2 + size, image.shape[1]-1))
                assert np.isclose(y2, min(bb.y1 - 1, image.shape[0]-1))

    def test__compute_bg_corner_coords__zero_sized_bb(self):
        height = 30
        size = 1
        drawer = _LabelOnImageDrawer(size=1, height=height)
        bb = ia.BoundingBox(x1=10, x2=10, y1=60, y2=90)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        x1, y1, x2, y2 = drawer._compute_bg_corner_coords(image, bb)
        assert np.isclose(x1, bb.x1 - size + 1)
        assert np.isclose(y1, bb.y1 - 1 - height)
        assert np.isclose(x2, bb.x2 + size)
        assert np.isclose(y2, bb.y1 - 1)

    def test__draw_label_arr__label_is_none(self):
        drawer = _LabelOnImageDrawer()
        height = 50
        width = 100
        nb_channels = 3
        color_text = np.uint8([0, 255, 0])
        color_bg = np.uint8([255, 0, 0])
        size_text = 20

        label_arr = drawer._draw_label_arr(None, height, width, nb_channels,
                                           np.uint8,
                                           color_text, color_bg, size_text)

        frac_textcolor = np.average(
            np.min(label_arr == color_text.reshape((1, 1, -1)), axis=-1)
        )
        frac_bgcolor = np.average(
            np.min(label_arr == color_bg.reshape((1, 1, -1)), axis=-1)
        )
        assert label_arr.dtype.name == "uint8"
        assert label_arr.shape == (height, width, nb_channels)
        assert frac_textcolor > 0.02
        assert frac_bgcolor > 0.8
        # not all pixels of the text might be drawn with exactly the text
        # color
        assert frac_textcolor + frac_bgcolor > 0.75

    def test__draw_label_arr__label_is_str(self):
        drawer = _LabelOnImageDrawer()
        height = 50
        width = 100
        nb_channels = 3
        color_text = np.uint8([0, 255, 0])
        color_bg = np.uint8([255, 0, 0])
        size_text = 20

        label_arr = drawer._draw_label_arr("Fooo", height, width, nb_channels,
                                           np.uint8,
                                           color_text, color_bg, size_text)

        frac_textcolor = np.average(
            np.min(label_arr == color_text.reshape((1, 1, -1)), axis=-1)
        )
        frac_bgcolor = np.average(
            np.min(label_arr == color_bg.reshape((1, 1, -1)), axis=-1)
        )
        assert label_arr.dtype.name == "uint8"
        assert label_arr.shape == (height, width, nb_channels)
        assert frac_textcolor > 0.02
        assert frac_bgcolor > 0.8
        # not all pixels of the text might be drawn with exactly the text
        # color
        assert frac_textcolor + frac_bgcolor > 0.75

    def test__blend_label_arr__alpha_is_1(self):
        drawer = _LabelOnImageDrawer(alpha=1)
        image = np.full((50, 60, 3), 100, dtype=np.uint8)
        label_arr = np.full((10, 20, 3), 200, dtype=np.uint8)
        x1 = 15
        x2 = 15 + 20
        y1 = 10
        y2 = 10 + 10

        image_blend = drawer._blend_label_arr_with_image_(image, label_arr,
                                                          x1, y1, x2, y2)

        assert np.all(image_blend[:, :15, :] == 100)
        assert np.all(image_blend[:, 15+20:, :] == 100)
        assert np.all(image_blend[:10, :, :] == 100)
        assert np.all(image_blend[10+10:, :, :] == 100)
        assert np.all(image_blend[10:10+10, 15:15+20, :] == 200)

    def test__blend_label_arr__alpha_is_075(self):
        drawer = _LabelOnImageDrawer(alpha=0.75)
        image = np.full((50, 60, 3), 100, dtype=np.uint8)
        label_arr = np.full((10, 20, 3), 200, dtype=np.uint8)
        x1 = 15
        x2 = 15 + 20
        y1 = 10
        y2 = 10 + 10

        image_blend = drawer._blend_label_arr_with_image_(image, label_arr,
                                                          x1, y1, x2, y2)

        assert np.all(image_blend[:, :15, :] == 100)
        assert np.all(image_blend[:, 15+20:, :] == 100)
        assert np.all(image_blend[:10, :, :] == 100)
        assert np.all(image_blend[10+10:, :, :] == 100)
        assert np.all(image_blend[10:10+10, 15:15+20, :] == 100+75)
