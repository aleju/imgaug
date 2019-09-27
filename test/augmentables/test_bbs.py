from __future__ import print_function, division, absolute_import

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

import imgaug as ia
import imgaug.random as iarandom


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

    def test_project_same_shape(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.project((10, 10), (10, 10))

        assert np.isclose(bb2.y1, 10)
        assert np.isclose(bb2.x1, 20)
        assert np.isclose(bb2.y2, 30)
        assert np.isclose(bb2.x2, 40)

    def test_project_upscale_by_2(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.project((10, 10), (20, 20))

        assert np.isclose(bb2.y1, 10*2)
        assert np.isclose(bb2.x1, 20*2)
        assert np.isclose(bb2.y2, 30*2)
        assert np.isclose(bb2.x2, 40*2)

    def test_project_downscale_by_2(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.project((10, 10), (5, 5))

        assert np.isclose(bb2.y1, 10*0.5)
        assert np.isclose(bb2.x1, 20*0.5)
        assert np.isclose(bb2.y2, 30*0.5)
        assert np.isclose(bb2.x2, 40*0.5)

    def test_project_onto_wider_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.project((10, 10), (10, 20))

        assert np.isclose(bb2.y1, 10*1)
        assert np.isclose(bb2.x1, 20*2)
        assert np.isclose(bb2.y2, 30*1)
        assert np.isclose(bb2.x2, 40*2)

    def test_project_onto_higher_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.project((10, 10), (20, 10))

        assert np.isclose(bb2.y1, 10*2)
        assert np.isclose(bb2.x1, 20*1)
        assert np.isclose(bb2.y2, 30*2)
        assert np.isclose(bb2.x2, 40*1)

    def test_extend_all_sides_by_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.extend(all_sides=1)

        assert bb2.y1 == 10-1
        assert bb2.y2 == 30+1
        assert bb2.x1 == 20-1
        assert bb2.x2 == 40+1

    def test_extend_all_sides_by_minus_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.extend(all_sides=-1)

        assert bb2.y1 == 10-(-1)
        assert bb2.y2 == 30+(-1)
        assert bb2.x1 == 20-(-1)
        assert bb2.x2 == 40+(-1)

    def test_extend_top_by_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.extend(top=1)

        assert bb2.y1 == 10-1
        assert bb2.y2 == 30+0
        assert bb2.x1 == 20-0
        assert bb2.x2 == 40+0

    def test_extend_right_by_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.extend(right=1)

        assert bb2.y1 == 10-0
        assert bb2.y2 == 30+0
        assert bb2.x1 == 20-0
        assert bb2.x2 == 40+1

    def test_extend_bottom_by_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.extend(bottom=1)

        assert bb2.y1 == 10-0
        assert bb2.y2 == 30+1
        assert bb2.x1 == 20-0
        assert bb2.x2 == 40+0

    def test_extend_left_by_1(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb2 = bb.extend(left=1)

        assert bb2.y1 == 10-0
        assert bb2.y2 == 30+0
        assert bb2.x1 == 20-1
        assert bb2.x2 == 40+0

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

    def test_clip_out_of_image_with_bb_fully_inside_image(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb_cut = bb.clip_out_of_image((100, 100, 3))

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

        bb_cut = bb.clip_out_of_image((20, 100, 3))

        assert bb_cut.y1 == 10
        assert bb_cut.x1 == 20
        assert np.isclose(bb_cut.y2, 20)
        assert bb_cut.x2 == 40

    def test_clip_out_of_image_with_bb_too_wide(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)

        bb_cut = bb.clip_out_of_image((100, 30, 3))

        assert bb_cut.y1 == 10
        assert bb_cut.x1 == 20
        assert bb_cut.y2 == 30
        assert np.isclose(bb_cut.x2, 30)

    def test_shift_top_by_zero(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_top = bb.shift(top=0)
        assert bb_top.y1 == 10
        assert bb_top.x1 == 20
        assert bb_top.y2 == 30
        assert bb_top.x2 == 40

    def test_shift_right_by_zero(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_right = bb.shift(right=0)
        assert bb_right.y1 == 10
        assert bb_right.x1 == 20
        assert bb_right.y2 == 30
        assert bb_right.x2 == 40

    def test_shift_bottom_by_zero(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_bottom = bb.shift(bottom=0)
        assert bb_bottom.y1 == 10
        assert bb_bottom.x1 == 20
        assert bb_bottom.y2 == 30
        assert bb_bottom.x2 == 40

    def test_shift_left_by_zero(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_left = bb.shift(left=0)
        assert bb_left.y1 == 10
        assert bb_left.x1 == 20
        assert bb_left.y2 == 30
        assert bb_left.x2 == 40

    def test_shift_top_by_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_top = bb.shift(top=1)
        assert bb_top.y1 == 10+1
        assert bb_top.x1 == 20
        assert bb_top.y2 == 30+1
        assert bb_top.x2 == 40

    def test_shift_right_by_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_right = bb.shift(right=1)
        assert bb_right.y1 == 10
        assert bb_right.x1 == 20-1
        assert bb_right.y2 == 30
        assert bb_right.x2 == 40-1

    def test_shift_bottom_by_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_bottom = bb.shift(bottom=1)
        assert bb_bottom.y1 == 10-1
        assert bb_bottom.x1 == 20
        assert bb_bottom.y2 == 30-1
        assert bb_bottom.x2 == 40

    def test_shift_left_by_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_left = bb.shift(left=1)
        assert bb_left.y1 == 10
        assert bb_left.x1 == 20+1
        assert bb_left.y2 == 30
        assert bb_left.x2 == 40+1

    def test_shift_top_by_minus_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_top = bb.shift(top=-1)
        assert bb_top.y1 == 10-1
        assert bb_top.x1 == 20
        assert bb_top.y2 == 30-1
        assert bb_top.x2 == 40

    def test_shift_right_by_minus_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_right = bb.shift(right=-1)
        assert bb_right.y1 == 10
        assert bb_right.x1 == 20+1
        assert bb_right.y2 == 30
        assert bb_right.x2 == 40+1

    def test_shift_bottom_by_minus_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_bottom = bb.shift(bottom=-1)
        assert bb_bottom.y1 == 10+1
        assert bb_bottom.x1 == 20
        assert bb_bottom.y2 == 30+1
        assert bb_bottom.x2 == 40

    def test_shift_left_by_minus_one(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_left = bb.shift(left=-1)
        assert bb_left.y1 == 10
        assert bb_left.x1 == 20-1
        assert bb_left.y2 == 30
        assert bb_left.x2 == 40-1

    def test_shift_all_sides_by_individual_amounts(self):
        bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb_mix = bb.shift(top=1, bottom=2, left=3, right=4)
        assert bb_mix.y1 == 10+1-2
        assert bb_mix.x1 == 20+3-4
        assert bb_mix.y2 == 30+3-4
        assert bb_mix.x2 == 40+1-2

    @classmethod
    def _get_standard_draw_on_image_vars(cls):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        bb = ia.BoundingBox(y1=1, x1=1, y2=3, x2=3)
        bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        bb_mask[1:3+1, 1] = True
        bb_mask[1:3+1, 3] = True
        bb_mask[1, 1:3+1] = True
        bb_mask[3, 1:3+1] = True
        return image, bb, bb_mask

    def test_draw_on_image(self):
        image, bb, bb_mask = self._get_standard_draw_on_image_vars()

        image_bb = bb.draw_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 255, 255])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])
        assert np.all(image == 0)

    def test_draw_on_image_red_color(self):
        image, bb, bb_mask = self._get_standard_draw_on_image_vars()

        image_bb = bb.draw_on_image(
            image, color=[255, 0, 0], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 0, 0])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    def test_draw_on_image_single_int_as_color(self):
        image, bb, bb_mask = self._get_standard_draw_on_image_vars()

        image_bb = bb.draw_on_image(
            image, color=128, alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [128, 128, 128])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    def test_draw_on_image_alpha_at_50_percent(self):
        image, bb, bb_mask = self._get_standard_draw_on_image_vars()

        image_bb = bb.draw_on_image(
            image + 100, color=[200, 200, 200], alpha=0.5, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [150, 150, 150])
        assert np.all(image_bb[~bb_mask] == [100, 100, 100])

    def test_draw_on_image_alpha_at_50_percent_and_float32_image(self):
        image, bb, bb_mask = self._get_standard_draw_on_image_vars()

        image_bb = bb.draw_on_image(
            (image+100).astype(np.float32),
            color=[200, 200, 200], alpha=0.5, size=1,
            copy=True, raise_if_out_of_image=False)

        assert np.sum(np.abs((image_bb - [150, 150, 150])[bb_mask])) < 0.1
        assert np.sum(np.abs((image_bb - [100, 100, 100])[~bb_mask])) < 0.1

    def test_draw_on_image_no_copy(self):
        image, bb, bb_mask = self._get_standard_draw_on_image_vars()

        image_bb = bb.draw_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=1, copy=False,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 255, 255])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])
        assert np.all(image[bb_mask] == [255, 255, 255])
        assert np.all(image[~bb_mask] == [0, 0, 0])

    def test_draw_on_image_bb_outside_of_image(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        bb = ia.BoundingBox(y1=-1, x1=-1, y2=2, x2=2)
        bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        bb_mask[2, 0:3] = True
        bb_mask[0:3, 2] = True

        image_bb = bb.draw_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 255, 255])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    def test_draw_on_image_bb_outside_of_image_and_very_small(self):
        image, bb, bb_mask = self._get_standard_draw_on_image_vars()
        bb = ia.BoundingBox(y1=-1, x1=-1, y2=1, x2=1)
        bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        bb_mask[0:1+1, 1] = True
        bb_mask[1, 0:1+1] = True

        image_bb = bb.draw_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 255, 255])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    def test_draw_on_image_size_2(self):
        image, bb, _ = self._get_standard_draw_on_image_vars()
        bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        bb_mask[0:5, 0:5] = True
        bb_mask[2, 2] = False

        image_bb = bb.draw_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=2, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_bb[bb_mask] == [255, 255, 255])
        assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    def test_draw_on_image_raise_true_but_bb_partially_inside_image(self):
        image, bb, bb_mask = self._get_standard_draw_on_image_vars()
        bb = ia.BoundingBox(y1=-1, x1=-1, y2=1, x2=1)

        _ = bb.draw_on_image(
            image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
            raise_if_out_of_image=True)

    def test_draw_on_image_raise_true_and_bb_fully_outside_image(self):
        image, bb, bb_mask = self._get_standard_draw_on_image_vars()
        bb = ia.BoundingBox(y1=-5, x1=-5, y2=-1, x2=-1)

        with self.assertRaises(Exception) as context:
            _ = bb.draw_on_image(
                image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
                raise_if_out_of_image=True)

        assert "Cannot draw bounding box" in str(context.exception)

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

        with self.assertRaises(AssertionError) as cm:
            _ = bb.coords_almost_equals(False)

        assert "Expected 'other'" in str(cm.exception)

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
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=image)
        assert bbsoi.bounding_boxes == [bb1, bb2]
        assert bbsoi.shape == (40, 50, 3)

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

    def test_on_same_height_width(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_projected = bbsoi.on((40, 50))

        assert bbsoi_projected.bounding_boxes[0].y1 == 10
        assert bbsoi_projected.bounding_boxes[0].x1 == 20
        assert bbsoi_projected.bounding_boxes[0].y2 == 30
        assert bbsoi_projected.bounding_boxes[0].x2 == 40
        assert bbsoi_projected.bounding_boxes[1].y1 == 15
        assert bbsoi_projected.bounding_boxes[1].x1 == 25
        assert bbsoi_projected.bounding_boxes[1].y2 == 35
        assert bbsoi_projected.bounding_boxes[1].x2 == 45

    def test_on_upscaled_by_2(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_projected = bbsoi.on((40*2, 50*2, 3))

        assert bbsoi_projected.bounding_boxes[0].y1 == 10*2
        assert bbsoi_projected.bounding_boxes[0].x1 == 20*2
        assert bbsoi_projected.bounding_boxes[0].y2 == 30*2
        assert bbsoi_projected.bounding_boxes[0].x2 == 40*2
        assert bbsoi_projected.bounding_boxes[1].y1 == 15*2
        assert bbsoi_projected.bounding_boxes[1].x1 == 25*2
        assert bbsoi_projected.bounding_boxes[1].y2 == 35*2
        assert bbsoi_projected.bounding_boxes[1].x2 == 45*2

    def test_on_upscaled_by_2_with_shape_given_as_array(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_projected = bbsoi.on(np.zeros((40*2, 50*2, 3), dtype=np.uint8))

        assert bbsoi_projected.bounding_boxes[0].y1 == 10*2
        assert bbsoi_projected.bounding_boxes[0].x1 == 20*2
        assert bbsoi_projected.bounding_boxes[0].y2 == 30*2
        assert bbsoi_projected.bounding_boxes[0].x2 == 40*2
        assert bbsoi_projected.bounding_boxes[1].y1 == 15*2
        assert bbsoi_projected.bounding_boxes[1].x1 == 25*2
        assert bbsoi_projected.bounding_boxes[1].y2 == 35*2
        assert bbsoi_projected.bounding_boxes[1].x2 == 45*2

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

    def test_to_xyxy_array(self):
        xyxy = np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0]
        ])
        bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy, shape=(40, 50, 3))

        xyxy_out = bbsoi.to_xyxy_array()

        assert np.allclose(xyxy, xyxy_out)
        assert xyxy_out.dtype == np.float32

    def test_to_xyxy_array_convert_to_int32(self):
        xyxy = np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0]
        ])
        bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy, shape=(40, 50, 3))

        xyxy_out = bbsoi.to_xyxy_array(dtype=np.int32)

        assert np.allclose(xyxy.astype(np.int32), xyxy_out)
        assert xyxy_out.dtype == np.int32

    def test_to_xyxy_array_no_bbs_to_convert(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(40, 50, 3))

        xyxy_out = bbsoi.to_xyxy_array(dtype=np.int32)

        assert xyxy_out.shape == (0, 4)

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

    def test_remove_out_of_image(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_removed = bbsoi.remove_out_of_image(fully=True, partly=True)

        assert len(bbsoi_removed.bounding_boxes) == 1
        assert bbsoi_removed.bounding_boxes[0] == bb1

    def test_clip_out_of_image(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

        bbsoi_clip = bbsoi.clip_out_of_image()

        assert len(bbsoi_clip.bounding_boxes) == 2
        assert bbsoi_clip.bounding_boxes[0].y1 == 10
        assert bbsoi_clip.bounding_boxes[0].x1 == 20
        assert bbsoi_clip.bounding_boxes[0].y2 == 30
        assert bbsoi_clip.bounding_boxes[0].x2 == 40
        assert bbsoi_clip.bounding_boxes[1].y1 == 15
        assert bbsoi_clip.bounding_boxes[1].x1 == 25
        assert bbsoi_clip.bounding_boxes[1].y2 == 35
        assert np.isclose(bbsoi_clip.bounding_boxes[1].x2, 50)

    def test_shift(self):
        bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40)
        bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51)
        bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))

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
