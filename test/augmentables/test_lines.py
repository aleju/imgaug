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
from imgaug.testutils import reseed
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.kps import Keypoint
from imgaug.augmentables.heatmaps import HeatmapsOnImage


class TestLineString_project_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, ls, from_shape, to_shape):
        return ls.project_(from_shape, to_shape)

    def test_project_to_2x_image_size(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        ls_proj = self._func(ls, (10, 10), (20, 20))
        assert np.allclose(ls_proj.coords, [(0, 0), (2, 0), (4, 2)])

    def test_project_to_2x_image_width(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        ls_proj = self._func(ls, (10, 10), (10, 20))
        assert np.allclose(ls_proj.coords, [(0, 0), (2, 0), (4, 1)])

    def test_project_to_2x_image_height(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        ls_proj = self._func(ls, (10, 10), (20, 10))
        assert np.allclose(ls_proj.coords, [(0, 0), (1, 0), (2, 2)])

    def test_inplaceness(self):
        ls = ia.LineString([(0, 0), (1, 0)])
        ls2 = self._func(ls, (10, 10), (10, 10))
        if self._is_inplace:
            assert ls is ls2
        else:
            assert ls is not ls2


class TestLineString_project(TestLineString_project_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, ls, from_shape, to_shape):
        return ls.project(from_shape, to_shape)


class TestLineString_shift_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, ls, top=0, right=0, bottom=0, left=0):
        return ls.shift_(top=top, right=right, bottom=bottom, left=left)

    def test_shift_by_positive_args(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert self._func(ls.deepcopy(), top=1).coords_almost_equals(
            [(0, 1), (1, 1), (2, 2)])
        assert self._func(ls.deepcopy(), right=1).coords_almost_equals(
            [(-1, 0), (0, 0), (1, 1)])
        assert self._func(ls.deepcopy(), bottom=1).coords_almost_equals(
            [(0, -1), (1, -1), (2, 0)])
        assert self._func(ls.deepcopy(), left=1).coords_almost_equals(
            [(1, 0), (2, 0), (3, 1)])

    def test_shift_by_negative_values(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert self._func(ls.deepcopy(), top=-1).coords_almost_equals(
            [(0, -1), (1, -1), (2, 0)])
        assert self._func(ls.deepcopy(), right=-1).coords_almost_equals(
            [(1, 0), (2, 0), (3, 1)])
        assert self._func(ls.deepcopy(), bottom=-1).coords_almost_equals(
            [(0, 1), (1, 1), (2, 2)])
        assert self._func(ls.deepcopy(), left=-1).coords_almost_equals(
            [(-1, 0), (0, 0), (1, 1)])

    def test_shift_by_positive_values_all_arguments_provided(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert self._func(
            ls.deepcopy(), top=1, right=2, bottom=3, left=4
        ).coords_almost_equals(
            [(0-2+4, 0+1-3), (1-2+4, 0+1-3), (2-2+4, 1+1-3)])

    def test_shift_of_empty_line_string(self):
        ls = LineString([])
        assert self._func(
            ls.deepcopy(), top=1, right=2, bottom=3, left=4
        ).coords_almost_equals([])

    def test_inplaceness(self):
        ls = ia.LineString([(0, 0), (1, 0)])
        ls2 = self._func(ls, top=0, right=0, bottom=0, left=0)
        if self._is_inplace:
            assert ls is ls2
        else:
            assert ls is not ls2


class TestLineString_shift(TestLineString_shift_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, ls, top=0, right=0, bottom=0, left=0):
        return ls.shift(top=top, right=right, bottom=bottom, left=left)


class TestLineString(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___float32_array_as_coords(self):
        ls = LineString(np.float32([[0, 0], [1, 2]]))
        assert np.allclose(ls.coords, np.float32([[0, 0], [1, 2]]))
        assert ls.label is None

    def test___init___list_of_tuples_as_coords(self):
        ls = LineString([(0, 0), (1, 2)])
        assert np.allclose(ls.coords, np.float32([[0, 0], [1, 2]]))
        assert ls.label is None

    def test___init___empty_list_as_coords(self):
        ls = LineString([])
        assert ls.coords.shape == (0, 2)
        assert ls.label is None

    def test___init___label_set(self):
        ls = LineString([], label="test")
        assert ls.coords.shape == (0, 2)
        assert ls.label == "test"

    def test_length_with_triangle(self):
        ls = LineString(np.float32([[0, 0], [1, 0], [1, 1]]))
        assert np.isclose(ls.length, 2.0)

    def test_length_with_realworld_line_string(self):
        ls = LineString(np.float32([[0, 0], [1, 2], [4, 5]]))
        assert np.isclose(ls.length,
                          np.sqrt(1**2 + 2**2) + np.sqrt(3**2 + 3**2))

    def test_length_with_single_point(self):
        ls = LineString([(0, 0)])
        assert np.isclose(ls.length, 0.0)

    def test_length_with_zero_points(self):
        ls = LineString([])
        assert np.isclose(ls.length, 0.0)

    def test_xx_three_points(self):
        ls = LineString(np.float32([[0, 0], [1, 0], [2, 1]]))
        assert np.allclose(ls.xx, np.float32([0, 1, 2]))

    def test_xx_no_points(self):
        ls = LineString([])
        assert np.allclose(ls.xx, np.zeros((0,), dtype=np.float32))

    def test_yy_three_points(self):
        ls = LineString(np.float32([[0, 0], [0, 1], [0, 2]]))
        assert np.allclose(ls.yy, np.float32([0, 1, 2]))

    def test_yy_no_points(self):
        ls = LineString([])
        assert np.allclose(ls.yy, np.zeros((0,), dtype=np.float32))

    def test_xx_int_three_points(self):
        ls = LineString(np.float32([[0, 0], [1.4, 0], [2.6, 1]]))
        assert ls.xx_int.dtype.name == "int32"
        assert np.array_equal(ls.xx_int, np.int32([0, 1, 3]))

    def test_xx_int_no_points(self):
        ls = LineString([])
        assert ls.xx_int.dtype.name == "int32"
        assert np.array_equal(ls.xx_int, np.zeros((0,), dtype=np.int32))

    def test_yy_int_three_points(self):
        ls = LineString(np.float32([[0, 0], [0, 1.4], [1, 2.6]]))
        assert ls.yy_int.dtype.name == "int32"
        assert np.array_equal(ls.yy_int, np.int32([0, 1, 3]))

    def test_yy_int_no_points(self):
        ls = LineString([])
        assert ls.yy_int.dtype.name == "int32"
        assert np.array_equal(ls.yy_int, np.zeros((0,), dtype=np.int32))

    def test_height_three_points(self):
        ls = LineString(np.float32([[0, 0], [0, 1.4], [1, 2.6]]))
        assert np.isclose(ls.height, 2.6)

    def test_height_no_points(self):
        ls = LineString([])
        assert np.isclose(ls.height, 0.0)

    def test_width_three_points(self):
        ls = LineString(np.float32([[0, 0], [1.4, 0], [2.6, 1]]))
        assert np.isclose(ls.width, 2.6)

    def test_width_no_points(self):
        ls = LineString([])
        assert np.isclose(ls.width, 0.0)

    def test_get_pointwise_inside_image_mask_with_single_point_tuple(self):
        ls = LineString([(0, 0), (1.4, 0), (2.6, 1)])
        mask = ls.get_pointwise_inside_image_mask((2, 2))
        assert np.array_equal(mask, [True, True, False])

    def test_get_pointwise_inside_image_mask_with_array(self):
        ls = LineString([(0, 0), (1.4, 0), (2.6, 1)])
        mask = ls.get_pointwise_inside_image_mask(
            np.zeros((2, 2), dtype=np.uint8))
        assert np.array_equal(mask, [True, True, False])

    def test_get_pointwise_inside_image_mask_with_single_point_ls(self):
        ls = LineString([(0, 0)])
        mask = ls.get_pointwise_inside_image_mask((2, 2))
        assert np.array_equal(mask, [True])

    def test_get_pointwise_inside_image_mask_with_zero_points_ls(self):
        ls = LineString([])
        mask = ls.get_pointwise_inside_image_mask((2, 2))
        assert mask.shape == (0,)

    def test_compute_neighbour_distances_three_points(self):
        ls = LineString([(0, 0), (1.4, 0), (2.6, 1)])
        dists = ls.compute_neighbour_distances()
        assert np.allclose(dists, [np.sqrt(1.4**2), np.sqrt(1.2**2+1**2)])

    def test_compute_neighbour_distances_two_points(self):
        ls = LineString([(0, 0), (1.4, 0)])
        dists = ls.compute_neighbour_distances()
        assert np.allclose(dists, [np.sqrt(1.4**2)])

    def test_compute_neighbour_distances_single_point(self):
        ls = LineString([(0, 0)])
        dists = ls.compute_neighbour_distances()
        assert dists.shape == (0,)

    def test_compute_neighbour_distances_zero_points(self):
        ls = LineString([])
        dists = ls.compute_neighbour_distances()
        assert dists.shape == (0,)

    def test_compute_pointwise_distances_to_point_at_origin(self):
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        dists = ls.compute_pointwise_distances((0, 0))
        assert np.allclose(dists, [0,
                                   5,
                                   np.sqrt(5**2 + 5**2)])

    def test_compute_pointwise_distances_to_point_at_x1_y1(self):
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        dists = ls.compute_pointwise_distances((1, 1))
        assert np.allclose(dists, [np.sqrt(1**2 + 1**2),
                                   np.sqrt(4**2 + 1**2),
                                   np.sqrt(4**2 + 4**2)])

    def test_compute_pointwise_distances_from_empty_line_string(self):
        ls = LineString([])
        dists = ls.compute_pointwise_distances((1, 1))
        assert dists == []

    def test_compute_pointwise_distances_to_keypoint_at_origin(self):
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        dists = ls.compute_pointwise_distances(Keypoint(x=0, y=0))
        assert np.allclose(dists, [0, 5, np.sqrt(5**2 + 5**2)])

    def test_compute_pointwise_distances_to_keypoint_at_x1_y1(self):
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        dists = ls.compute_pointwise_distances(Keypoint(x=1, y=1))
        assert np.allclose(dists, [np.sqrt(1**2 + 1**2),
                                   np.sqrt(4**2 + 1**2),
                                   np.sqrt(4**2 + 4**2)])

    def test_compute_pointwise_distances_to_other_line_string_at_origin(self):
        # line string
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        other = LineString([(0, 0)])
        dists = ls.compute_pointwise_distances(other)
        assert np.allclose(dists, [0,
                                   5,
                                   np.sqrt(5**2 + 5**2)])

    def test_compute_pointwise_distances_to_other_line_string_at_x1_y1(self):
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        other = LineString([(1, 1)])
        dists = ls.compute_pointwise_distances(other)
        assert np.allclose(dists, [np.sqrt(1**2 + 1**2),
                                   np.sqrt(4**2 + 1**2),
                                   np.sqrt(4**2 + 4**2)])

    def test_compute_pointwise_distances_to_other_line_string_two_points(self):
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        other = LineString([(0, -1), (5, -1)])
        dists = ls.compute_pointwise_distances(other)
        assert np.allclose(dists, [np.sqrt(0**2 + 1**2),
                                   np.sqrt(0**2 + 1**2),
                                   np.sqrt(0**2 + 6**2)])

    def test_compute_pointwise_distances_to_other_empty_line_string(self):
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        other = LineString([])
        dists = ls.compute_pointwise_distances(other, default=False)
        assert dists is False

    def test_compute_distance_from_three_point_line_string(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        points = [(0, 0), (1, 0), (0, 1), (-0.5, -0.6)]
        expecteds = [0, 0, 1, np.sqrt(0.5**2 + 0.6**2)]

        for point, expected in zip(points, expecteds):
            with self.subTest(point=point):
                assert np.isclose(ls.compute_distance(point), expected)

    def test_compute_distance_from_single_point_line_string(self):
        ls = LineString([(0, 0)])
        points = [(0, 0), (-0.5, -0.6)]
        expecteds = [0, np.sqrt(0.5**2 + 0.6**2)]

        for point, expected in zip(points, expecteds):
            with self.subTest(point=point):
                assert np.isclose(ls.compute_distance(point), expected)

    def test_compute_distance_from_empty_line_string_no_default(self):
        ls = LineString([])
        assert ls.compute_distance((0, 0)) is None

    def test_compute_distance_from_empty_line_string_with_default(self):
        ls = LineString([])
        assert ls.compute_distance((0, 0), default=-1) == -1

    def test_compute_distance_to_keypoint(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert np.isclose(ls.compute_distance(ia.Keypoint(x=0, y=1)), 1)

    def test_compute_distance_to_line_string(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        points = [
            [(0, 0)],
            [(0, 1)],
            [(0, 0), (0, 1)],
            [(-1, -1), (-1, 1)]
        ]
        expecteds = [0, 1, 0, 1]

        for point, expected in zip(points, expecteds):
            with self.subTest(point=point):
                assert np.isclose(ls.compute_distance(LineString(point)),
                                  expected)

    def test_compute_distance_to_invalid_datatype(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        with self.assertRaises(ValueError):
            assert ls.compute_distance("foo")

    def test_contains_tuple(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        points = [(100, 0), (0, 0), (1, 0), (2, 1), (0+1e-8, 0), (0-1, 0)]
        expecteds = [False, True, True, True, True, False]

        for point, expected in zip(points, expecteds):
            with self.subTest(point=point):
                assert ls.contains(point) is expected

    def test_contains_tuple_max_distance(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        points = [(0+1e-8, 0), (0-1, 0)]
        max_distances = [0, 2]
        expecteds = [False, True]

        for point, max_distance, expected in zip(points, max_distances,
                                                 expecteds):
            with self.subTest(point=point, max_distance=max_distance):
                assert (
                    ls.contains(point, max_distance=max_distance)
                    is expected
                )

    def test_contains_keypoint(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        points = [(100, 0), (0, 0), (1, 0), (2, 1), (0+1e-8, 0), (0-1, 0)]
        expecteds = [False, True, True, True, True, False]

        for point, expected in zip(points, expecteds):
            with self.subTest(point=point):
                assert (
                    ls.contains(Keypoint(x=point[0], y=point[1]))
                    is expected
                )

    def test_contains_with_single_point_line_string(self):
        ls = LineString([(0, 0)])
        assert ls.contains((0, 0))
        assert not ls.contains((1, 0))

    def test_contains_with_empty_line_string(self):
        ls = LineString([])
        assert not ls.contains((0, 0))
        assert not ls.contains((1, 0))

    def test_project_empty_line_string_to_2x_image_size(self):
        ls = LineString([])
        ls_proj = ls.project((10, 10), (20, 20))
        assert ls_proj.coords.shape == (0, 2)

    def test_compute_out_of_image_fraction__no_points(self):
        ls = LineString([])
        image_shape = (100, 200, 3)
        factor = ls.compute_out_of_image_fraction(image_shape)
        assert np.isclose(factor, 0.0)

    def test_compute_out_of_image_fraction__one_point(self):
        ls = LineString([(1.0, 2.0)])
        image_shape = (100, 200, 3)
        factor = ls.compute_out_of_image_fraction(image_shape)
        assert np.isclose(factor, 0.0)

    def test_compute_out_of_image_fraction__one_point__ooi(self):
        ls = LineString([(-10.0, -20.0)])
        image_shape = (100, 200, 3)
        factor = ls.compute_out_of_image_fraction(image_shape)
        assert np.isclose(factor, 1.0)

    def test_compute_out_of_image_fraction__two_points(self):
        ls = LineString([(1.0, 2.0), (10.0, 20.0)])
        image_shape = (100, 200, 3)
        factor = ls.compute_out_of_image_fraction(image_shape)
        assert np.isclose(factor, 0.0)

    def test_compute_out_of_image_fraction__three_points_at_same_pos(self):
        ls = LineString([(10.0, 20.0), (10.0, 20.0), (10.0, 20.0)])
        image_shape = (100, 200, 3)
        factor = ls.compute_out_of_image_fraction(image_shape)
        assert len(ls.coords) == 3
        assert np.isclose(factor, 0.0)

    def test_compute_out_of_image_fraction__partially_ooi(self):
        ls = LineString([(9.0, 1.0), (11.0, 1.0)])
        image_shape = (10, 10, 3)
        factor = ls.compute_out_of_image_fraction(image_shape)
        assert np.isclose(factor, 0.5, atol=1e-3)

    def test_compute_out_of_image_fraction__leaves_image_multiple_times(self):
        ls = LineString([(9.0, 1.0), (11.0, 1.0), (11.0, 3.0),
                         (9.0, 3.0), (9.0, 5.0), (11.0, 5.0)])
        image_shape = (10, 10, 3)
        factor = ls.compute_out_of_image_fraction(image_shape)
        assert np.isclose(factor, 0.5, atol=1e-3)

    def test_compute_out_of_image_fraction__fully_ooi(self):
        ls = LineString([(15.0, 15.0), (20.0, 15.0)])
        image_shape = (10, 10, 3)
        factor = ls.compute_out_of_image_fraction(image_shape)
        assert np.isclose(factor, 1.0)

    def test_is_fully_within_image_with_simple_line_string(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert ls.is_fully_within_image((10, 10))
        assert ls.is_fully_within_image((2, 3))
        assert not ls.is_fully_within_image((2, 2))
        assert not ls.is_fully_within_image((1, 1))

    def test_is_fully_within_image_with_negative_coords_line_string(self):
        ls = LineString([(-1, 0), (1, 0), (2, 1)])
        assert not ls.is_fully_within_image((10, 10))
        assert not ls.is_fully_within_image((2, 3))
        assert not ls.is_fully_within_image((2, 2))
        assert not ls.is_fully_within_image((1, 1))

    def test_is_fully_within_image_with_single_point_line_string(self):
        ls = LineString([(0, 0)])
        assert ls.is_fully_within_image((10, 10))
        assert ls.is_fully_within_image((2, 3))
        assert ls.is_fully_within_image((2, 2))
        assert ls.is_fully_within_image((1, 1))

    def test_is_fully_within_image_with_empty_line_string(self):
        ls = LineString([])
        assert not ls.is_fully_within_image((10, 10))
        assert not ls.is_fully_within_image((2, 3))
        assert not ls.is_fully_within_image((2, 2))
        assert not ls.is_fully_within_image((1, 1))

    def test_is_fully_within_image_with_empty_line_string_default_set(self):
        ls = LineString([])
        assert ls.is_fully_within_image((10, 10), default=True)
        assert ls.is_fully_within_image((2, 3), default=True)
        assert ls.is_fully_within_image((2, 2), default=True)
        assert ls.is_fully_within_image((1, 1), default=True)
        assert ls.is_fully_within_image((10, 10), default=None) is None

    def test_is_partly_within_image_with_simple_line_string(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert ls.is_partly_within_image((10, 10))
        assert ls.is_partly_within_image((2, 3))
        assert ls.is_partly_within_image((2, 2))
        assert ls.is_partly_within_image((1, 1))

    def test_is_partly_within_image_with_simple_line_string2(self):
        ls = LineString([(1, 0), (2, 0), (3, 1)])
        assert ls.is_partly_within_image((10, 10))
        assert ls.is_partly_within_image((2, 3))
        assert ls.is_partly_within_image((2, 2))
        assert not ls.is_partly_within_image((1, 1))

    def test_is_partly_within_image_with_ls_cutting_through_image(self):
        # line string that cuts through the middle of the image,
        # with both points outside of a BB (0, 0), (10, 10)
        ls = LineString([(-1, 5), (11, 5)])
        assert ls.is_partly_within_image((100, 100))
        assert ls.is_partly_within_image((10, 12))
        assert ls.is_partly_within_image((10, 10))
        assert ls.is_partly_within_image((10, 1))
        assert not ls.is_partly_within_image((1, 1))

    def test_is_partly_within_image_with_line_string_around_rectangle(self):
        # line string around inner rectangle of (-1, -1), (11, 11)
        ls = LineString([(-1, -1), (11, -1), (11, 11), (-1, 11)])
        assert ls.is_partly_within_image((100, 100))
        assert ls.is_partly_within_image((12, 12))
        assert not ls.is_partly_within_image((10, 10))

    def test_is_partly_within_image_with_single_point_line_string(self):
        ls = LineString([(11, 11)])
        assert ls.is_partly_within_image((100, 100))
        assert ls.is_partly_within_image((12, 12))
        assert not ls.is_partly_within_image((10, 10))

    def test_is_partly_within_image_with_empty_line_string(self):
        ls = LineString([])
        assert not ls.is_partly_within_image((100, 100))
        assert not ls.is_partly_within_image((10, 10))
        assert ls.is_partly_within_image((100, 100), default=True)
        assert ls.is_partly_within_image((10, 10), default=True)
        assert ls.is_partly_within_image((100, 100), default=None) is None
        assert ls.is_partly_within_image((10, 10), default=None) is None

    def test_is_out_of_image_with_simple_line_string(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert not ls.is_out_of_image((10, 10))
        assert ls.is_out_of_image((1, 1), fully=False, partly=True)
        assert not ls.is_out_of_image((1, 1), fully=True, partly=False)
        assert ls.is_out_of_image((1, 1), fully=True, partly=True)
        assert not ls.is_out_of_image((1, 1), fully=False, partly=False)

    def test_is_out_of_image_with_empty_line_string(self):
        ls = LineString([])
        assert ls.is_out_of_image((10, 10))
        assert not ls.is_out_of_image((10, 10), default=False)
        assert ls.is_out_of_image((10, 10), default=None) is None

    def test_clip_out_of_image_with_simple_line_string(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((2, 2))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((2, 1))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], [(0, 0), (1, 0)])

    def test_clip_out_of_image_with_shifted_simple_line_string(self):
        # same as above, all coords at x+5, y+5
        ls = LineString([(5, 5), (6, 5), (7, 6)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((4, 4))
        assert len(lss_clipped) == 0

    def test_clip_out_of_image_with_ls_partially_outside_image(self):
        # line that leaves image plane and comes back
        ls = LineString([(0, 0), (1, 0), (3, 0),
                         (3, 2), (1, 2), (0, 2)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((10, 2))
        assert len(lss_clipped) == 2
        assert _coords_eq(lss_clipped[0], [(0, 0), (1, 0), (2, 0)])
        assert _coords_eq(lss_clipped[1], [(2, 2), (1, 2), (0, 2)])

        lss_clipped = ls.clip_out_of_image((10, 1))
        assert len(lss_clipped) == 2
        assert _coords_eq(lss_clipped[0], [(0, 0), (1, 0)])
        assert _coords_eq(lss_clipped[1], [(1, 2), (0, 2)])

    def test_clip_out_of_image_with_ls_partially_ooi_less_points(self):
        # same as above, but removing first and last point
        # so that only one point before and after out of image part remain
        ls = LineString([(1, 0), (3, 0),
                         (3, 2), (1, 2)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((10, 2))
        assert len(lss_clipped) == 2
        assert _coords_eq(lss_clipped[0], [(1, 0), (2, 0)])
        assert _coords_eq(lss_clipped[1], [(2, 2), (1, 2)])

        lss_clipped = ls.clip_out_of_image((10, 1))
        assert len(lss_clipped) == 0

    def test_clip_out_of_image_when_only_one_point_remains(self):
        # same as above, but only one point out of image remains
        ls = LineString([(1, 0), (3, 0),
                         (1, 2)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((10, 2))
        assert len(lss_clipped) == 2
        assert _coords_eq(lss_clipped[0], [(1, 0), (2, 0)])
        assert _coords_eq(lss_clipped[1], [(2, 1), (1, 2)])

        lss_clipped = ls.clip_out_of_image((10, 1))
        assert len(lss_clipped) == 0

    def test_clip_out_of_image_with_ls_leaving_image_multiple_times(self):
        # line string that leaves image, comes back, then leaves again, then
        # comes back again
        ls = LineString([(1, 0), (3, 0),  # leaves
                         (3, 1), (1, 1),  # comes back
                         (1, 2), (3, 2),  # leaves
                         (3, 3), (1, 3)])  # comes back

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((10, 2))
        assert len(lss_clipped) == 3  # from above: 1s line, 2nd+3rd, 4th
        assert _coords_eq(lss_clipped[0], [(1, 0), (2, 0)])
        assert _coords_eq(lss_clipped[1], [(2, 1), (1, 1), (1, 2), (2, 2)])
        assert _coords_eq(lss_clipped[2], [(2, 3), (1, 3)])

    def test_clip_out_of_image_with_ls_that_enters_image_from_outside(self):
        # line string that starts out of image and ends within the image plane
        for y in [1, 0]:
            with self.subTest(y=y):
                # one point inside image
                ls = LineString([(-10, y), (3, y)])

                lss_clipped = ls.clip_out_of_image((10, 10))
                assert len(lss_clipped) == 1
                assert _coords_eq(lss_clipped[0], [(0, y), (3, y)])

                lss_clipped = ls.clip_out_of_image((2, 1))
                assert len(lss_clipped) == 1
                assert _coords_eq(lss_clipped[0], [(0, y), (1, y)])

                lss_clipped = ls.clip_out_of_image((1, 1))
                if y == 1:
                    assert len(lss_clipped) == 0
                else:
                    assert len(lss_clipped) == 1
                    assert _coords_eq(lss_clipped[0], [(0, y), (1, y)])

                # two points inside image
                ls = LineString([(-10, y), (3, y), (5, y)])

                lss_clipped = ls.clip_out_of_image((10, 10))
                assert len(lss_clipped) == 1
                assert _coords_eq(lss_clipped[0], [(0, y), (3, y), (5, y)])

                lss_clipped = ls.clip_out_of_image((10, 4))
                assert len(lss_clipped) == 1
                assert _coords_eq(lss_clipped[0], [(0, y), (3, y), (4, y)])

                lss_clipped = ls.clip_out_of_image((2, 1))
                assert len(lss_clipped) == 1
                assert _coords_eq(lss_clipped[0], [(0, y), (1, y)])

                lss_clipped = ls.clip_out_of_image((1, 1))
                if y == 1:
                    assert len(lss_clipped) == 0
                else:
                    assert len(lss_clipped) == 1
                    assert _coords_eq(lss_clipped[0], [(0, y), (1, y)])

    def test_clip_out_of_image_with_ls_that_leaves_image_from_inside(self):
        # line string that starts within the image plane and ends outside
        for y in [1, 0]:
            with self.subTest(y=y):
                # one point inside image
                ls = LineString([(2, y), (5, y)])

                lss_clipped = ls.clip_out_of_image((10, 10))
                assert len(lss_clipped) == 1
                assert _coords_eq(lss_clipped[0], [(2, y), (5, y)])

                lss_clipped = ls.clip_out_of_image((10, 4))
                assert _coords_eq(lss_clipped[0], [(2, y), (4, y)])

                # two points inside image
                ls = LineString([(1, y), (2, y), (5, y)])

                lss_clipped = ls.clip_out_of_image((10, 10))
                assert len(lss_clipped) == 1
                assert _coords_eq(lss_clipped[0], [(1, y), (2, y), (5, y)])

                lss_clipped = ls.clip_out_of_image((10, 4))
                assert len(lss_clipped) == 1
                assert _coords_eq(lss_clipped[0], [(1, y), (2, y), (4, y)])

                lss_clipped = ls.clip_out_of_image((2, 1))
                assert len(lss_clipped) == 0

                # two points outside image
                ls = LineString([(2, y), (5, y), (6, y)])

                lss_clipped = ls.clip_out_of_image((10, 10))
                assert len(lss_clipped) == 1
                assert _coords_eq(lss_clipped[0], [(2, y), (5, y), (6, y)])

                lss_clipped = ls.clip_out_of_image((10, 4))
                assert len(lss_clipped) == 1
                assert _coords_eq(lss_clipped[0], [(2, y), (4, y)])

                lss_clipped = ls.clip_out_of_image((2, 1))
                assert len(lss_clipped) == 0

    def test_clip_out_of_image_with_ls_that_cuts_through_image(self):
        # line string that cuts through the image plane in the center
        for y in [1, 0]:
            ls = LineString([(-5, y), (5, y)])

            lss_clipped = ls.clip_out_of_image((10, 10))
            assert len(lss_clipped) == 1
            assert _coords_eq(lss_clipped[0], [(0, y), (5, y)])

            lss_clipped = ls.clip_out_of_image((4, 4))
            assert len(lss_clipped) == 1
            assert _coords_eq(lss_clipped[0], [(0, y), (4, y)])

    def test_clip_out_of_image_with_ls_that_runs_through_diagonal_corners(self):
        # line string that cuts through the image plane from the bottom left
        # corner to the top right corner
        ls = LineString([(-5, -5), (5, 5)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], [(0, 0), (5, 5)])

        lss_clipped = ls.clip_out_of_image((4, 4))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], [(0, 0), (4, 4)])

    def test_clip_out_of_image_with_ls_that_overlaps_with_image_edge(self):
        # line string that overlaps with the bottom edge
        ls = LineString([(1, 0), (4, 0)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((3, 3))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], [(1, 0), (3, 0)])

    def test_clip_out_of_image_with_ls_that_overlaps_with_image_edge2(self):
        # same as above, multiple points on line
        ls = LineString([(1, 0), (4, 0), (5, 0), (6, 0), (7, 0)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((5, 5))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], [(1, 0), (4, 0), (5, 0)])

        lss_clipped = ls.clip_out_of_image((5, 4))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], [(1, 0), (4, 0)])

        lss_clipped = ls.clip_out_of_image((5, 2))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], [(1, 0), (2, 0)])

    def test_clip_out_of_image_with_ls_that_overlaps_with_image_edge3(self):
        # line string that starts outside the image, intersects with the
        # bottom left corner and overlaps with the bottom border
        ls = LineString([(-5, 0), (5, 0)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], [(0, 0), (5, 0)])

        lss_clipped = ls.clip_out_of_image((10, 5))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], [(0, 0), (5, 0)])

        lss_clipped = ls.clip_out_of_image((10, 4))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], [(0, 0), (4, 0)])

    def test_clip_out_of_image_with_ls_that_contains_a_single_point(self):
        # line string that contains a single point
        ls = LineString([(2, 2)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _coords_eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((1, 1))
        assert len(lss_clipped) == 0

    def test_clip_out_of_image_with_ls_that_is_empty(self):
        # line string that is empty
        ls = LineString([])
        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 0

    def test_clip_out_of_image_and_is_fully_within_image(self):
        # combine clip + is_fully_within_image
        sizes = [(200, 400), (400, 800), (800, 1600), (1600, 3200),
                 (3200, 6400)]
        sizes = sizes + [(w, h) for h, w in sizes]
        for h, w in sizes:
            ls = LineString([(0, 10), (w, 10), (w, h), (w-10, h-10)])
            lss_clipped = ls.clip_out_of_image((h, w))
            assert len(lss_clipped) == 2
            assert lss_clipped[0].is_fully_within_image((h, w))
            assert lss_clipped[1].is_fully_within_image((h, w))

            ls = LineString([(0, 10), (w+10, 10), (w+10, h-10), (w-10, h-10)])
            lss_clipped = ls.clip_out_of_image((h, w))
            assert len(lss_clipped) == 2
            assert lss_clipped[0].is_fully_within_image((h, w))
            assert lss_clipped[1].is_fully_within_image((h, w))

            ls = LineString([(-10, 10), (w+10, 10), (w-10, h-10)])
            lss_clipped = ls.clip_out_of_image((h, w))
            assert len(lss_clipped) == 2
            assert lss_clipped[0].is_fully_within_image((h, w))
            assert lss_clipped[1].is_fully_within_image((h, w))

    def test_draw_mask(self):
        ls = LineString([(0, 1), (5, 1), (5, 5)])

        arr = ls.draw_mask(
            (10, 10), size_lines=1, size_points=0, raise_if_out_of_image=False)

        assert np.all(arr[1, 0:5])
        assert np.all(arr[1:5, 5])
        assert not np.any(arr[0, :])
        assert not np.any(arr[2:, 0:5])

    def test_draw_mask_of_empty_line_string(self):
        ls = LineString([])

        arr = ls.draw_mask(
            (10, 10), size_lines=1, raise_if_out_of_image=False)

        assert not np.any(arr)

    def test_draw_line_heatmap_array(self):
        ls = LineString([(0, 1), (5, 1), (5, 5)])

        arr = ls.draw_lines_heatmap_array(
            (10, 10), alpha=0.5, size=1, raise_if_out_of_image=False)

        assert _drawing_allclose(arr[1, 0:5], 0.5)
        assert _drawing_allclose(arr[1:5, 5], 0.5)
        assert _drawing_allclose(arr[0, :], 0.0)
        assert _drawing_allclose(arr[2:, 0:5], 0.0)

    def test_draw_line_heatmap_array_with_empty_line_string(self):
        ls = LineString([])

        arr = ls.draw_lines_heatmap_array(
            (10, 10), alpha=0.5, size=1, raise_if_out_of_image=False)

        assert _drawing_allclose(arr, 0.0)

    def test_draw_points_heatmap_array(self):
        ls = LineString([(0, 1), (5, 1), (5, 5)])

        arr = ls.draw_points_heatmap_array(
            (10, 10), alpha=0.5, size=1, raise_if_out_of_image=False)

        assert _drawing_allclose(arr[1, 0], 0.5)
        assert _drawing_allclose(arr[1, 5], 0.5)
        assert _drawing_allclose(arr[5, 5], 0.5)
        assert _drawing_allclose(arr[0, :], 0.0)
        assert _drawing_allclose(arr[2:, 0:5], 0.0)

    def test_draw_points_heatmap_array_with_empty_line_string(self):
        ls = LineString([])

        arr = ls.draw_points_heatmap_array(
            (10, 10), alpha=0.5, size=1, raise_if_out_of_image=False)

        assert _drawing_allclose(arr, 0.0)

    def test_draw_heatmap_array_calls_other_drawing_functions(self):
        ls = LineString([(0, 1), (9, 1)])

        module_name = "imgaug.augmentables.lines."
        line_fname = "%sLineString.draw_lines_heatmap_array" % (module_name,)
        points_fname = "%sLineString.draw_points_heatmap_array" % (module_name,)
        with mock.patch(line_fname, return_value=1) as mock_line, \
                mock.patch(points_fname, return_value=2) as mock_points:
            _arr = ls.draw_heatmap_array(
                (10, 10),
                alpha_lines=0.9, alpha_points=0.8,
                size_lines=3, size_points=5,
                antialiased=True,
                raise_if_out_of_image=True)

            assert mock_line.call_count == 1
            assert mock_points.call_count == 1

            assert mock_line.call_args_list[0][0][0] == (10, 10)
            assert np.isclose(mock_line.call_args_list[0][1]["alpha"], 0.9)
            assert mock_line.call_args_list[0][1]["size"] == 3
            assert mock_line.call_args_list[0][1]["antialiased"] is True
            assert mock_line.call_args_list[0][1]["raise_if_out_of_image"] \
                is True

            assert mock_points.call_args_list[0][0][0] == (10, 10)
            assert np.isclose(mock_points.call_args_list[0][1]["alpha"], 0.8)
            assert mock_points.call_args_list[0][1]["size"] == 5
            assert mock_points.call_args_list[0][1]["raise_if_out_of_image"] \
                is True

    def test_draw_heatmap_array_without_mocking(self):
        ls = LineString([(0, 1), (5, 1), (5, 5)])

        arr = ls.draw_heatmap_array((10, 10),
                                    alpha_lines=0.9, alpha_points=0.5,
                                    size_lines=1, size_points=3,
                                    antialiased=False,
                                    raise_if_out_of_image=False)

        assert _drawing_allclose(arr[1, 0:5], 0.9)
        assert _drawing_allclose(arr[1, 0:5], 0.9)
        assert _drawing_allclose(arr[1, 0:5], 0.9)
        assert _drawing_allclose(arr[2:5, 5], 0.9)
        assert _drawing_allclose(arr[2:5, 5], 0.9)
        assert _drawing_allclose(arr[2:5, 5], 0.9)

        assert _drawing_allclose(arr[0, 0:2], 0.5)
        assert _drawing_allclose(arr[2, 0:2], 0.5)

        assert _drawing_allclose(arr[0, 4:6 + 1], 0.5)
        assert _drawing_allclose(arr[2, 4], 0.5)
        assert _drawing_allclose(arr[2, 6], 0.5)

        assert _drawing_allclose(arr[4, 4], 0.5)
        assert _drawing_allclose(arr[4, 6], 0.5)
        assert _drawing_allclose(arr[6, 4:6 + 1], 0.5)

        assert _drawing_allclose(arr[0, 3], 0.0)
        assert _drawing_allclose(arr[7:, :], 0.0)

    def test_draw_heatmap_array_with_empty_line_string(self):
        ls = LineString([])

        arr = ls.draw_heatmap_array((10, 10))

        assert arr.shape == (10, 10)
        assert np.sum(arr) == 0

    def test_draw_line_on_image_with_image_of_zeros(self):
        # image of 0s
        ls = LineString([(0, 1), (9, 1)])

        img = ls.draw_lines_on_image(
            np.zeros((3, 10, 3), dtype=np.uint8),
            color=(10, 200, 20),
            alpha=1.0, size=1,
            antialiased=False,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img[0, :, :] == 0)
        assert np.all(img[1, :, 0] == 10)
        assert np.all(img[1, :, 1] == 200)
        assert np.all(img[1, :, 2] == 20)
        assert np.all(img[2, :, :] == 0)

    def test_draw_line_on_image_with_2d_image_of_zeros(self):
        # image of 0s, 2D input image
        ls = LineString([(0, 1), (9, 1)])

        img = ls.draw_lines_on_image(
            np.zeros((3, 10), dtype=np.uint8),
            color=200,
            alpha=1.0, size=1,
            antialiased=False,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img[0, :] == 0)
        assert np.all(img[1, :] == 200)
        assert np.all(img[2, :] == 0)

    def test_draw_line_on_image_of_ones(self):
        # image of 1s
        ls = LineString([(0, 1), (9, 1)])

        img = ls.draw_lines_on_image(
            np.ones((3, 10, 3), dtype=np.uint8),
            color=(10, 200, 20),
            alpha=1.0, size=1,
            antialiased=False,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img[0, :, :] == 1)
        assert np.all(img[1, :, 0] == 10)
        assert np.all(img[1, :, 1] == 200)
        assert np.all(img[1, :, 2] == 20)
        assert np.all(img[2, :, :] == 1)

    def test_draw_line_on_image_alpha_at_50_percent(self):
        # alpha=0.5
        ls = LineString([(0, 1), (9, 1)])

        img = ls.draw_lines_on_image(
            np.zeros((3, 10, 3), dtype=np.uint8),
            color=(10, 200, 20),
            alpha=0.5, size=1,
            antialiased=False,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img[0, :, :] == 0)
        assert np.all(img[1, :, 0] == 5)
        assert np.all(img[1, :, 1] == 100)
        assert np.all(img[1, :, 2] == 10)
        assert np.all(img[2, :, :] == 0)

    def test_draw_line_on_image_alpha_at_50_percent_with_background(self):
        # alpha=0.5 with background
        ls = LineString([(0, 1), (9, 1)])

        img = ls.draw_lines_on_image(
            10 + np.zeros((3, 10, 3), dtype=np.uint8),
            color=(10, 200, 20),
            alpha=0.5, size=1,
            antialiased=False,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img[0, :, :] == 10)
        assert np.all(img[1, :, 0] == 5+5)
        assert np.all(img[1, :, 1] == 5+100)
        assert np.all(img[1, :, 2] == 5+10)
        assert np.all(img[2, :, :] == 10)

    def test_draw_line_on_image_with_size_3(self):
        # size=3
        ls = LineString([(0, 5), (9, 5)])

        img = ls.draw_lines_on_image(
            np.zeros((10, 10, 3), dtype=np.uint8),
            color=(10, 200, 20),
            alpha=1.0, size=3,
            antialiased=False,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img[5-1:5+1+1, :, 0] == 10)
        assert np.all(img[5-1:5+1+1, :, 1] == 200)
        assert np.all(img[5-1:5+1+1, :, 2] == 20)
        assert np.all(img[:5-1, :, :] == 0)
        assert np.all(img[5+1+1:, :, :] == 0)

    def test_draw_line_on_image_with_2d_image_and_size_3(self):
        # size=3, 2D input image
        ls = LineString([(0, 5), (9, 5)])

        img = ls.draw_lines_on_image(
            np.zeros((10, 10), dtype=np.uint8),
            color=200,
            alpha=1.0, size=3,
            antialiased=False,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img[5-1:5+1+1, :] == 200)
        assert np.all(img[:5-1, :] == 0)
        assert np.all(img[5+1+1:, :] == 0)

    def test_draw_line_on_image_with_size_3_and_antialiasing(self):
        # size=3, antialiasing
        ls = LineString([(0, 0), (9, 9)])

        img = ls.draw_lines_on_image(
            np.zeros((10, 10, 3), dtype=np.uint8),
            color=(100, 100, 100),
            alpha=1.0, size=3,
            antialiased=False,
            raise_if_out_of_image=False
        )
        img_aa = ls.draw_lines_on_image(
            np.zeros((10, 10, 3), dtype=np.uint8),
            color=(100, 100, 100),
            alpha=1.0, size=3,
            antialiased=True,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert img_aa.dtype.name == "uint8"
        assert np.sum(img) > 5 * 3 * 100
        assert np.sum(img_aa) > 5 * 3 * 100
        assert not np.array_equal(img, img_aa)
        assert np.all(img[:3, -3:, :] == 0)
        assert np.all(img_aa[:3, -3:, :] == 0)
        assert np.all(img[-3:, :3, :] == 0)
        assert np.all(img_aa[-3:, :3, :] == 0)

    def test_draw_line_on_image_with_line_partially_outside_image(self):
        # line partially outside if image
        ls = LineString([(-1, 1), (9, 1)])

        img = ls.draw_lines_on_image(
            np.zeros((3, 10, 3), dtype=np.uint8),
            color=(10, 200, 20),
            alpha=1.0, size=1,
            antialiased=False,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img[0, :, :] == 0)
        assert np.all(img[1, :, 0] == 10)
        assert np.all(img[1, :, 1] == 200)
        assert np.all(img[1, :, 2] == 20)
        assert np.all(img[2, :, :] == 0)

    def test_draw_line_on_image_with_line_fully_outside_image(self):
        # line fully outside if image
        ls = LineString([(-10, 1), (-9, 1)])

        img = ls.draw_lines_on_image(
            np.zeros((3, 10, 3), dtype=np.uint8),
            color=(10, 200, 20),
            alpha=1.0, size=1,
            antialiased=False,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img == 0)

    def test_draw_line_on_image_with_line_fully_ooi_and_raise_true(self):
        # raise_if_out_of_image=True
        ls = LineString([(0-5, 5), (-1, 5)])

        with self.assertRaises(Exception) as context:
            _img = ls.draw_lines_on_image(
                np.zeros((10, 10, 3), dtype=np.uint8),
                color=(100, 100, 100),
                alpha=1.0, size=3,
                antialiased=False,
                raise_if_out_of_image=True
            )

        assert "Cannot draw line string " in str(context.exception)

    def test_draw_line_on_image_with_line_part_inside_img_and_raise_true(self):
        # raise_if_out_of_image=True BUT line is partially inside image
        # (no point is inside image though)
        ls = LineString([(-1, 5), (11, 5)])

        _img = ls.draw_lines_on_image(
            np.zeros((10, 10, 3), dtype=np.uint8),
            color=(100, 100, 100),
            alpha=1.0, size=3,
            antialiased=False,
            raise_if_out_of_image=True
        )

    def test_draw_points_on_image_with_image_of_zeros(self):
        # iamge of 0s
        ls = LineString([(0, 1), (9, 1)])

        img = ls.draw_points_on_image(
            np.zeros((3, 10, 3), dtype=np.uint8),
            color=(10, 255, 20),
            alpha=1.0, size=3,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img[0:2, 0:2, 0] == 10)
        assert np.all(img[0:2, 0:2, 1] == 255)
        assert np.all(img[0:2, 0:2, 2] == 20)
        assert np.all(img[0:2, -2:, 0] == 10)
        assert np.all(img[0:2, -2:, 1] == 255)
        assert np.all(img[0:2, -2:, 2] == 20)
        assert np.all(img[:, 2:-2, :] == 0)

    def test_draw_points_on_image_with_image_of_ones(self):
        # image of 1s
        ls = LineString([(0, 1), (9, 1)])

        img = ls.draw_points_on_image(
            np.ones((3, 10, 3), dtype=np.uint8),
            color=(10, 200, 20),
            alpha=1.0, size=3,
            raise_if_out_of_image=False
        )

        assert img.dtype.name == "uint8"
        assert np.all(img[0:2, 0:2, 0] == 10)
        assert np.all(img[0:2, 0:2, 1] == 200)
        assert np.all(img[0:2, 0:2, 2] == 20)
        assert np.all(img[0:2, -2:, 0] == 10)
        assert np.all(img[0:2, -2:, 1] == 200)
        assert np.all(img[0:2, -2:, 2] == 20)
        assert np.all(img[:, 2:-2, :] == 1)

    def test_draw_points_on_image_with_alpha_50_percent(self):
        # alpha=0.5
        ls = LineString([(0, 1), (9, 1)])

        img = ls.draw_points_on_image(
            np.zeros((3, 10, 3), dtype=np.uint8),
            color=(10, 200, 20),
            alpha=0.5, size=3,
            raise_if_out_of_image=False
        )

        assert np.all(img[0:2, 0:2, 0] == 5)
        assert np.all(img[0:2, 0:2, 1] == 100)
        assert np.all(img[0:2, 0:2, 2] == 10)
        assert np.all(img[0:2, -2:, 0] == 5)
        assert np.all(img[0:2, -2:, 1] == 100)
        assert np.all(img[0:2, -2:, 2] == 10)
        assert np.all(img[:, 2:-2, :] == 0)

    def test_draw_points_on_image_with_size_one(self):
        # size=1
        ls = LineString([(0, 1), (9, 1)])

        img = ls.draw_points_on_image(
            np.zeros((3, 10, 3), dtype=np.uint8),
            color=(10, 200, 20),
            alpha=1.0, size=1,
            raise_if_out_of_image=False
        )

        assert np.all(img[0, :, :] == 0)
        assert np.all(img[2, :, :] == 0)

        assert np.all(img[1, 0, 0] == 10)
        assert np.all(img[1, 0, 1] == 200)
        assert np.all(img[1, 0, 2] == 20)

        assert np.all(img[1, -1, 0] == 10)
        assert np.all(img[1, -1, 1] == 200)
        assert np.all(img[1, -1, 2] == 20)

    def test_draw_points_on_image_with_ls_ooi_and_raise_true(self):
        with self.assertRaises(Exception) as context:
            ls = LineString([(0-5, 1), (9+5, 1)])
            _img = ls.draw_points_on_image(
                np.zeros((3, 10, 3), dtype=np.uint8),
                color=(10, 200, 20),
                alpha=0.5, size=1,
                raise_if_out_of_image=True
            )

        assert "Cannot draw keypoint " in str(context.exception)

    def test_draw_on_image_with_mocking(self):
        ls = LineString([(0, 1), (9, 1)])

        module_name = "imgaug.augmentables.lines."
        line_fname = "%sLineString.draw_lines_on_image" % (module_name,)
        points_fname = "%sLineString.draw_points_on_image" % (module_name,)
        with mock.patch(line_fname, return_value=1) as mock_line, \
                mock.patch(points_fname, return_value=2) as mock_points:
            _image = ls.draw_on_image(
                np.zeros((10, 10, 3), dtype=np.uint8),
                color=(1, 2, 3), color_lines=(4, 5, 6), color_points=(7, 8, 9),
                alpha=1.0, alpha_lines=0.9, alpha_points=0.8,
                size=1, size_lines=3, size_points=5,
                antialiased=False,
                raise_if_out_of_image=True)

        assert mock_line.call_count == 1
        assert mock_points.call_count == 1

        assert mock_line.call_args_list[0][0][0].shape == (10, 10, 3)
        assert mock_line.call_args_list[0][1]["color"][0] == 4
        assert mock_line.call_args_list[0][1]["color"][1] == 5
        assert mock_line.call_args_list[0][1]["color"][2] == 6
        assert np.isclose(mock_line.call_args_list[0][1]["alpha"], 0.9)
        assert mock_line.call_args_list[0][1]["size"] == 3
        assert mock_line.call_args_list[0][1]["antialiased"] is False
        assert mock_line.call_args_list[0][1]["raise_if_out_of_image"] \
            is True

        assert mock_points.call_args_list[0][0][0] == 1  # mock_line result
        assert mock_points.call_args_list[0][1]["color"][0] == 7
        assert mock_points.call_args_list[0][1]["color"][1] == 8
        assert mock_points.call_args_list[0][1]["color"][2] == 9
        assert np.isclose(mock_points.call_args_list[0][1]["alpha"], 0.8)
        assert mock_points.call_args_list[0][1]["size"] == 5
        assert mock_points.call_args_list[0][1]["raise_if_out_of_image"] \
            is True

    def test_draw_on_image_without_mocking(self):
        ls = LineString([(0, 1), (5, 1), (5, 5)])

        img = ls.draw_on_image(np.zeros((10, 10, 3), dtype=np.uint8),
                               color=(200, 120, 40), alpha=0.5, size=1)

        assert np.all(img[1, 0:5, 0] == 100)
        assert np.all(img[1, 0:5, 1] == 60)
        assert np.all(img[1, 0:5, 2] == 20)
        assert np.all(img[1:5, 5, 0] == 100)
        assert np.all(img[1:5, 5, 1] == 60)
        assert np.all(img[1:5, 5, 2] == 20)
        assert np.all(img[0:2+1, 0:2, 0] >= 50)  # color_points is 0.5*color
        assert np.all(img[0:2+1, 0:2, 1] >= 30)
        assert np.all(img[0:2+1, 0:2, 2] >= 10)
        assert np.all(img[0:2+1, 4:6+1, 0] >= 50)
        assert np.all(img[0:2+1, 4:6+1, 1] >= 30)
        assert np.all(img[0:2+1, 4:6+1, 2] >= 10)
        assert np.all(img[4:6+1, 4:6+1, 0] >= 50)
        assert np.all(img[4:6+1, 4:6+1, 1] >= 30)
        assert np.all(img[4:6+1, 4:6+1, 2] >= 10)
        assert np.all(img[0, 3, :] == 0)
        assert np.all(img[7:, :, :] == 0)

    def test_draw_on_image_with_empty_line_string(self):
        ls = LineString([])

        img = ls.draw_on_image(np.zeros((10, 10, 3), dtype=np.uint8))

        assert img.shape == (10, 10, 3)
        assert np.sum(img) == 0

    def test_extract_from_image_size_1_single_channel(self):
        ls = LineString([(0, 5), (9, 5)])
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)

        extract = ls.extract_from_image(img, antialiased=False)

        assert extract.shape == (1, 10, 1)
        assert np.array_equal(extract, img[5:6, 0:10, :])

    def test_extract_from_image_size_3_single_channel(self):
        ls = LineString([(1, 5), (8, 5)])
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)

        extract = ls.extract_from_image(img, size=3, antialiased=False)

        assert extract.shape == (3, 10, 1)
        assert np.array_equal(extract, img[4:6+1, 0:10, :])

    def test_extract_from_image_size_3_rgb(self):
        # size=3, RGB image
        ls = LineString([(1, 5), (8, 5)])
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)
        img_rgb = np.tile(img, (1, 1, 3))
        img_rgb[..., 1] += 10
        img_rgb[..., 2] += 20

        extract = ls.extract_from_image(img_rgb, size=3, antialiased=False)

        assert extract.shape == (3, 10, 3)
        assert np.array_equal(extract, img_rgb[4:6+1, 0:10, :])

    def test_extract_from_image_antialiased_true(self):
        # weak antialiased=True test
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)
        ls = LineString([(1, 1), (9, 9)])

        extract_aa = ls.extract_from_image(img, size=3, antialiased=True)
        extract = ls.extract_from_image(img, size=3, antialiased=False)

        assert extract_aa.shape == extract.shape
        assert np.sum(extract_aa) > np.sum(extract)

    def test_extract_from_image_pad_false(self):
        # pad=False
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)
        ls = LineString([(-5, 5), (-3, 5)])

        extract = ls.extract_from_image(img, size=1, antialiased=False,
                                        pad=False, prevent_zero_size=True)

        assert extract.shape == (1, 1, 1)
        assert np.sum(extract) == 0

    def test_extract_from_image_pad_false_and_prevent_zero_size_false(self):
        # pad=False, prevent_zero_size=False
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)
        ls = LineString([(-5, 5), (-3, 5)])

        extract = ls.extract_from_image(img, size=1, antialiased=False,
                                        pad=False, prevent_zero_size=False)

        assert extract.shape == (0, 0, 1)

    def test_extract_from_image_pad_max(self):
        # pad_max=1
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)
        ls = LineString([(-5, 5), (9, 5)])

        extract = ls.extract_from_image(img, antialiased=False, pad=True,
                                        pad_max=1)

        assert extract.shape == (1, 11, 1)
        assert np.array_equal(extract[:, 1:], img[5:6, 0:10, :])
        assert np.all(extract[0, 0, :] == 0)

    def test_extract_from_image_with_single_point_line_string(self):
        # 1 coord
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)
        ls = LineString([(1, 1)])

        extract = ls.extract_from_image(img)

        assert extract.shape == (1, 1, 1)
        assert np.sum(extract) == img[1:2, 1:2, :]

    def test_extract_from_image_with_single_point_ls_negative_coords(self):
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)
        ls = LineString([(-10, -10)])

        extract = ls.extract_from_image(img)

        assert extract.shape == (1, 1, 1)
        assert np.sum(extract) == 0

    def test_extract_from_image_with_1_point_neg_coords_prevent_zero_size(self):
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)
        ls = LineString([(-10, -10)])

        extract = ls.extract_from_image(img, prevent_zero_size=True)

        assert extract.shape == (1, 1, 1)
        assert np.sum(extract) == 0

    def test_extract_from_image_with_empty_line_string(self):
        # 0 coords
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)
        ls = LineString([])

        extract = ls.extract_from_image(img)

        assert extract.shape == (1, 1, 1)
        assert np.sum(extract) == 0

    def test_extract_from_image_with_empty_line_string_prevent_zero_size(self):
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)
        ls = LineString([])

        extract = ls.extract_from_image(img, prevent_zero_size=False)

        assert extract.shape == (0, 0, 1)
        assert np.sum(extract) == 0

    def test_concatenate_line_string_with_itself(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        ls_concat = ls.concatenate(ls)
        assert ls_concat.coords_almost_equals([
            (0, 0), (1, 0), (2, 1), (0, 0), (1, 0), (2, 1)
        ])

    def test_concatenate_empty_line_string_with_itself(self):
        ls = LineString([])
        ls_concat = ls.concatenate(ls)
        assert ls_concat.coords_almost_equals([])

    def test_concatenate_empty_line_string_with_single_point_line_string(self):
        ls = LineString([])
        ls_concat = ls.concatenate(LineString([(0, 0)]))
        assert ls_concat.coords_almost_equals([(0, 0)])

    def test_concatenate_single_point_line_string_with_empty_line_string(self):
        ls = LineString([(0, 0)])
        ls_concat = ls.concatenate(LineString([]))
        assert ls_concat.coords_almost_equals([(0, 0)])

    def test_concatenate_empty_line_string_with_list_of_tuples(self):
        ls = LineString([])
        ls_concat = ls.concatenate([(0, 0)])
        assert ls_concat.coords_almost_equals([(0, 0)])

    def test_to_keypoints(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        observed = ls.to_keypoints()
        assert all([isinstance(kp, ia.Keypoint) for kp in observed])
        assert np.isclose(observed[0].x, 0)
        assert np.isclose(observed[0].y, 0)
        assert np.isclose(observed[1].x, 1)
        assert np.isclose(observed[1].y, 0)
        assert np.isclose(observed[2].x, 2)
        assert np.isclose(observed[2].y, 1)

    def test_to_keypoints_with_empty_line_string(self):
        ls = LineString([])
        assert len(ls.to_keypoints()) == 0

    def test_to_bounding_box(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        observed = ls.to_bounding_box()
        assert isinstance(observed, ia.BoundingBox)
        assert np.isclose(observed.x1, 0)
        assert np.isclose(observed.y1, 0)
        assert np.isclose(observed.x2, 1)
        assert np.isclose(observed.y2, 1)

    def test_to_bounding_box_with_single_point_line_string(self):
        ls = LineString([(0, 0)])
        observed = ls.to_bounding_box()
        assert isinstance(observed, ia.BoundingBox)
        assert np.isclose(observed.x1, 0)
        assert np.isclose(observed.y1, 0)
        assert np.isclose(observed.x2, 0)
        assert np.isclose(observed.y2, 0)

    def test_to_bounding_box_with_empty_line_string(self):
        ls = LineString([])
        observed = ls.to_bounding_box()
        assert observed is None

    def test_to_polygon(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        observed = ls.to_polygon()
        assert isinstance(observed, ia.Polygon)
        assert np.allclose(observed.exterior, [(0, 0), (1, 0), (1, 1)])

    def test_to_polygon_with_single_point_line_string(self):
        ls = LineString([(0, 0)])
        observed = ls.to_polygon()
        assert isinstance(observed, ia.Polygon)
        assert np.allclose(observed.exterior, [(0, 0)])

    def test_to_polygon_with_empty_line_string(self):
        ls = LineString([])
        observed = ls.to_polygon()
        assert isinstance(observed, ia.Polygon)
        assert len(observed.exterior) == 0

    # TODO add antialiased=True test
    def test_to_heatmap(self):
        ls = LineString([(0, 5), (5, 5)])
        observed = ls.to_heatmap((10, 10), antialiased=False)
        assert isinstance(observed, HeatmapsOnImage)
        assert observed.shape == (10, 10)
        assert observed.arr_0to1.shape == (10, 10, 1)
        assert np.allclose(observed.arr_0to1[0:5, :, :], 0.0)
        assert np.allclose(observed.arr_0to1[5, 0:5, :], 1.0)
        assert np.allclose(observed.arr_0to1[6:, :, :], 0.0)

    def test_to_heatmap_with_empty_line_string(self):
        ls = LineString([])
        observed = ls.to_heatmap((5, 5), antialiased=False)
        assert observed.shape == (5, 5)
        assert observed.arr_0to1.shape == (5, 5, 1)
        assert np.allclose(observed.arr_0to1, 0.0)

    # TODO change this after the segmap PR was merged

    def test_segmentation_map(self):
        from imgaug.augmentables.segmaps import SegmentationMapsOnImage
        ls = LineString([(0, 5), (5, 5)])
        observed = ls.to_segmentation_map((10, 10))
        assert isinstance(observed, SegmentationMapsOnImage)
        assert observed.shape == (10, 10)
        assert observed.arr.shape == (10, 10, 1)
        assert np.all(observed.arr[0:5, :, :] == 0)
        assert np.all(observed.arr[5, 0:5, :] == 1)
        assert np.all(observed.arr[6:, :, :] == 0)

        ls = LineString([])
        observed = ls.to_segmentation_map((5, 5))
        assert observed.shape == (5, 5)
        assert observed.arr.shape == (5, 5, 1)
        assert np.all(observed.arr == 0)

    def test_coords_almost_equals_with_3_point_ls_90deg_angle(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        assert ls.coords_almost_equals(ls)
        assert ls.coords_almost_equals([(0, 0), (1, 0), (1, 1)])
        assert not ls.shift(top=1).coords_almost_equals(ls)
        assert ls.shift(top=1).coords_almost_equals(ls, max_distance=1.01)
        assert ls.coords_almost_equals([(0, 0), (0.5, 0), (1, 0), (1, 1)])

    def test_coords_almost_equals_with_4_point_ls_90deg_angle(self):
        ls = LineString([(0, 0), (0.5, 0), (1, 0), (1, 1)])
        assert ls.coords_almost_equals([(0, 0), (1, 0), (1, 1)])

    def test_coords_almost_equals_with_1_point_ls(self):
        ls = LineString([(0, 0)])
        assert ls.coords_almost_equals([(0, 0)])
        assert not ls.coords_almost_equals([(0+1, 0)])
        assert ls.coords_almost_equals([(0+1, 0)], max_distance=1.01)
        assert not ls.coords_almost_equals([])

    def test_coords_almost_equals_with_empty_line_string(self):
        ls = LineString([])
        assert ls.coords_almost_equals([])
        assert not ls.coords_almost_equals([(0, 0)])

    def test_coords_almost_equals_with_two_rectangular_line_strings(self):
        # two rectangles around height=10, width=10 image,
        # both LS closed, second one has more points around the left edge
        ls_a = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        ls_b = LineString([(0, 0), (10, 0), (10, 10), (0, 10),
                           (0, 5.01), (0, 5.0), (0, 4.99), (0, 0)])
        assert ls_a.coords_almost_equals(ls_b)

    def test_coords_almost_equals_different_ls_but_overlapping_points(self):
        # almost the same as in above test
        # two rectangles around height=10, width=10 image,
        # both LS closed, second one has more points around the left edge
        # AND around left edge the second line string suddenly has a line
        # up to the right edge, which immediately returns back to the left
        # edge
        # All points overlap between the lines, i.e. this test fails if naively
        # checking for overlapping points.
        ls_a = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        ls_b = LineString([(0, 0), (10, 0), (10, 10), (0, 10),
                           (0, 5.01), (10, 5.0), (0, 4.99), (0, 0)])
        assert not ls_a.coords_almost_equals(ls_b)

    def test_almost_equals_three_point_ls_without_label(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        ls_shifted = ls.shift(top=1)
        assert ls.almost_equals(ls)
        assert not ls.almost_equals(ls_shifted)
        assert ls.almost_equals(LineString([(0, 0), (1, 0), (1, 1)],
                                           label=None))
        assert not ls.almost_equals(LineString([(0, 0), (1, 0), (1, 1)],
                                               label="foo"))

    def test_almost_equals_three_point_ls_with_label(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)], label="foo")
        assert not ls.almost_equals(LineString([(0, 0), (1, 0), (1, 1)],
                                               label=None))
        assert ls.almost_equals(LineString([(0, 0), (1, 0), (1, 1)],
                                           label="foo"))

    def test_almost_equals_empty_line_string(self):
        ls = LineString([])
        assert ls.almost_equals(LineString([]))
        assert not ls.almost_equals(LineString([], label="foo"))

    def test_almost_equals_empty_line_string_with_label(self):
        ls = LineString([], label="foo")
        assert not ls.almost_equals(LineString([]))
        assert ls.almost_equals(LineString([], label="foo"))

    def test_copy_with_various_line_strings(self):
        coords = [
            [(0, 0), (1, 0), (1, 1)],
            [(0, 0), (1.5, 0), (1.6, 1)],
            [(0, 0)],
            [],
            [(0, 0), (1, 0), (1, 1)]
        ]
        labels = [None, None, None, None, "foo"]

        for coords_i, label in zip(coords, labels):
            with self.subTest(coords=coords_i, label=label):
                ls = LineString(coords_i, label=label)
                observed = ls.copy()
                assert observed is not ls
                assert observed.coords is ls.coords
                assert observed.label is ls.label

    def test_copy_with_coords_arg_set(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        observed = ls.copy(coords=[(0, 0)])
        assert observed.coords_almost_equals([(0, 0)])
        assert observed.label is None

    def test_copy_with_label_arg_set(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        observed = ls.copy(label="bar")
        assert observed.coords is ls.coords
        assert observed.label == "bar"

    def test_deepcopy_with_various_line_strings(self):
        coords = [
            [(0, 0), (1, 0), (1, 1)],
            [(0, 0), (1.5, 0), (1.6, 1)],
            [(0, 0)],
            [],
            [(0, 0), (1, 0), (1, 1)]
        ]
        labels = [None, None, None, None, "foo"]

        for coords_i, label in zip(coords, labels):
            with self.subTest(coords=coords_i, label=label):
                ls = LineString(coords_i, label=label)
                observed = ls.deepcopy()
                assert observed is not ls
                assert observed.coords is not ls.coords
                assert observed.label == ls.label

    def test_deepcopy_with_coords_arg_set(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        observed = ls.deepcopy(coords=[(0, 0)])
        assert observed.coords_almost_equals([(0, 0)])
        assert observed.label is None

    def test_deepcopy_with_label_arg_set(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        observed = ls.deepcopy(label="bar")
        assert observed.coords is not ls.coords
        assert observed.label == "bar"

    def test_string_conversion(self):
        coords = [
            [(0, 0), (1, 0), (1, 1)],
            [(0, 0), (1.5, 0), (1.6, 1)],
            [(0, 0)],
            [],
            [(0, 0), (1, 0), (1, 1)]
        ]
        labels = [None, None, None, None, "foo"]

        for coords_i, label in zip(coords, labels):
            with self.subTest(coords=coords_i, label=label):
                ls = LineString(coords_i, label=label)
                # __str__() is tested more thoroughly in other tests
                assert ls.__repr__() == ls.__str__()

    def test___getitem__(self):
        cba = ia.LineString([(0, 1), (2, 3)])
        assert np.allclose(cba[0], (0, 1))
        assert np.allclose(cba[1], (2, 3))

    def test___getitem___slice(self):
        cba = ia.LineString([(0, 1), (2, 3), (4, 5)])
        assert np.allclose(cba[1:], [(2, 3), (4, 5)])

    def test___iter___two_points(self):
        cba = LineString([(1, 2), (3, 4)])
        for i, xy in enumerate(cba):
            assert i in [0, 1]
            if i == 0:
                assert np.allclose(xy, (1, 2))
            elif i == 1:
                assert np.allclose(xy, (3, 4))
        assert i == 1

    def test___iter___zero_points(self):
        cba = LineString([])
        i = 0
        for xy in cba:
            i += 1
        assert i == 0

    def test___str__(self):
        coords = [
            [(0, 0), (1, 0), (1, 1)],
            [(0, 0), (1.5, 0), (1.6, 1)],
            [(0, 0)],
            [],
            [(0, 0), (1, 0), (1, 1)]
        ]
        labels = [None, None, None, None, "foo"]
        expecteds = [
            "LineString([(0.00, 0.00), (1.00, 0.00), (1.00, 1.00)], "
            "label=None)",
            "LineString([(0.00, 0.00), (1.50, 0.00), (1.60, 1.00)], "
            "label=None)",
            "LineString([(0.00, 0.00)], label=None)",
            "LineString([], label=None)",
            "LineString([(0.00, 0.00), (1.00, 0.00), (1.00, 1.00)], "
            "label=foo)"
        ]

        for coords_i, label, expected in zip(coords, labels, expecteds):
            with self.subTest(coords=coords_i, label=label):
                ls = LineString(coords_i, label=label)
                observed = ls.__str__()
                assert observed == expected


class TestLineStringsOnImage_items_setter(unittest.TestCase):
    def test_with_list_of_line_strings(self):
        ls = [ia.LineString([(0, 0), (1, 0)]),
              ia.LineString([(1, 1), (10, 0)])]
        lsoi = ia.LineStringsOnImage([], shape=(10, 20, 3))
        lsoi.items = ls
        assert np.all([
            (np.allclose(ls_i.coords, ls_j.coords))
            for ls_i, ls_j
            in zip(lsoi.line_strings, ls)
        ])


class TestLineStringsOnImage_on_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, lsoi, to_shape):
        return lsoi.on_(to_shape)

    def test_on_image_with_same_shape(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        lsoi_proj = self._func(lsoi, (100, 100, 3))

        assert np.all([ls_a.coords_almost_equals(ls_b)
                      for ls_a, ls_b
                      in zip(lsoi.line_strings, lsoi_proj.line_strings)])
        assert lsoi_proj.shape == (100, 100, 3)

    def test_on_image_with_2x_size(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        lsoi_proj = self._func(lsoi, (200, 200, 3))

        assert lsoi_proj.line_strings[0].coords_almost_equals(
            [(0, 0), (1*2, 0), (2*2, 1*2)]
        )
        assert lsoi_proj.line_strings[1].coords_almost_equals(
            [(10*2, 10*2)]
        )
        assert lsoi_proj.shape == (200, 200, 3)

    def test_on_image_with_2x_size_and_empty_list_of_line_strings(self):
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        lsoi_proj = self._func(lsoi, (200, 200, 3))

        assert len(lsoi_proj.line_strings) == 0
        assert lsoi_proj.shape == (200, 200, 3)

    def test_inplaceness(self):
        ls = ia.LineString([(0, 0), (1, 0)])
        lsoi = LineStringsOnImage([ls], shape=(10, 10, 3))
        lsoi2 = self._func(lsoi, (10, 10))
        if self._is_inplace:
            assert lsoi is lsoi2
        else:
            assert lsoi is not lsoi2


class TestLineStringsOnImage_on(TestLineStringsOnImage_on_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, lsoi, to_shape):
        return lsoi.on(to_shape)


class TestLineStringsOnImage_remove_out_of_image_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, lsoi, fully=True, partly=False):
        return lsoi.remove_out_of_image_(fully=fully, partly=partly)

    def test_remove_out_of_image_with_two_line_strings(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        observed = self._func(lsoi)

        assert len(observed.line_strings) == 2
        assert observed.line_strings[0] is ls1
        assert observed.line_strings[1] is ls2

    def test_remove_out_of_image_with_empty_list_of_line_strings(self):
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = self._func(lsoi)

        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

    def test_remove_out_of_image_with_one_empty_line_string(self):
        ls1 = LineString([])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))

        observed = self._func(lsoi)

        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

    def test_remove_out_of_image_with_one_line_string(self):
        ls1 = LineString([(-10, -10), (5, 5)])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))

        observed = self._func(lsoi)

        assert len(observed.line_strings) == 1
        assert observed.line_strings[0] is ls1
        assert observed.shape == (100, 100, 3)

    def test_remove_out_of_image_remove_even_partial_oois(self):
        ls1 = LineString([(-10, -10), (5, 5)])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))

        observed = self._func(lsoi, partly=True, fully=True)

        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

    def test_remove_out_of_image_with_one_single_point_line_strings(self):
        ls1 = LineString([(-10, -10)])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))

        observed = self._func(lsoi)

        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

    def test_remove_out_of_image_partly_false_and_fully_false(self):
        ls1 = LineString([(-10, -10)])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))

        observed = self._func(lsoi, partly=False, fully=False)

        assert len(observed.line_strings) == 1
        assert observed.line_strings[0] is ls1
        assert observed.shape == (100, 100, 3)

    def test_inplaceness(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        lsoi2 = self._func(lsoi)

        if self._is_inplace:
            assert lsoi is lsoi2
        else:
            assert lsoi is not lsoi2


class TestLineStringsOnImage_remove_out_of_image(
        TestLineStringsOnImage_remove_out_of_image_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, lsoi, fully=True, partly=False):
        return lsoi.remove_out_of_image(fully=fully, partly=partly)


class TestLineStringsOnImage_clip_out_of_image_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, lsoi):
        return lsoi.clip_out_of_image_()

    def test_clip_out_of_image_with_two_simple_line_strings(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        observed = self._func(lsoi)

        expected = []
        expected.extend(ls1.clip_out_of_image((100, 100, 3)))
        expected.extend(ls2.clip_out_of_image((100, 100, 3)))
        assert len(lsoi.line_strings) == len(expected)
        for ls_obs, ls_exp in zip(observed.line_strings, expected):
            assert ls_obs.coords_almost_equals(ls_exp)
        assert observed.shape == (100, 100, 3)

    def test_clip_out_of_image_with_empty_list_of_line_strings(self):
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = self._func(lsoi)

        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

    def test_clip_out_of_image_with_one_empty_line_string(self):
        ls1 = LineString([])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))

        observed = self._func(lsoi)

        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

    def test_clip_out_of_image_with_single_point_ls_and_negative_coords(self):
        ls1 = LineString([(-10, -10)])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))

        observed = self._func(lsoi)

        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

    def test_inplaceness(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        lsoi2 = self._func(lsoi)

        if self._is_inplace:
            assert lsoi is lsoi2
        else:
            assert lsoi is not lsoi2


class TestLineStringsOnImage_clip_out_of_image(
        TestLineStringsOnImage_clip_out_of_image_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, lsoi):
        return lsoi.clip_out_of_image()


class TestLineStringsOnImage_shift_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, lsoi, top=0, right=0, bottom=0, left=0):
        return lsoi.shift_(top=top, right=right, bottom=bottom, left=left)

    def test_shift_with_two_simple_line_strings(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        observed = self._func(lsoi.deepcopy(), top=1, right=2, bottom=3, left=4)

        assert observed.line_strings[0].coords_almost_equals(
            ls1.shift(top=1, right=2, bottom=3, left=4)
        )
        assert observed.line_strings[1].coords_almost_equals(
            ls2.shift(top=1, right=2, bottom=3, left=4)
        )
        assert observed.shape == (100, 100, 3)

    def test_shift_with_empty_list_of_line_strings(self):
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = self._func(lsoi, top=1, right=2, bottom=3, left=4)

        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

    def test_inplaceness(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        lsoi2 = self._func(lsoi, top=1, right=2, bottom=3, left=4)

        if self._is_inplace:
            assert lsoi is lsoi2
        else:
            assert lsoi is not lsoi2


class TestLineStringsOnImage_shift(TestLineStringsOnImage_shift_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, lsoi, top=0, right=0, bottom=0, left=0):
        return lsoi.shift(top=top, right=right, bottom=bottom, left=left)


# TODO test to_keypoints_on_image()
#      test invert_to_keypoints_on_image()
#      test to_xy_array()
#      test fill_from_xy_array_()
class TestLineStringsOnImage(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        lsoi = LineStringsOnImage([
            LineString([]),
            LineString([(0, 0), (5, 0)])
        ], shape=(10, 10, 3))
        assert len(lsoi.line_strings) == 2
        assert lsoi.shape == (10, 10, 3)

    def test___init___with_empty_list(self):
        lsoi = LineStringsOnImage([], shape=(10, 10))
        assert lsoi.line_strings == []
        assert lsoi.shape == (10, 10)

    def test_items(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        items = lsoi.items

        assert items == [ls1, ls2]

    def test_items_empty(self):
        kpsoi = LineStringsOnImage([], shape=(40, 50, 3))

        items = kpsoi.items

        assert items == []

    def test_empty_with_empty_list(self):
        lsoi = LineStringsOnImage([], shape=(10, 10, 3))
        assert lsoi.empty

    def test_empty_with_list_of_empty_line_string(self):
        lsoi = LineStringsOnImage([LineString([])], shape=(10, 10, 3))
        assert not lsoi.empty

    def test_empty_with_list_of_single_point_line_string(self):
        lsoi = LineStringsOnImage([LineString([(0, 0)])], shape=(10, 10, 3))
        assert not lsoi.empty

    def test_from_xy_arrays_single_input_array_with_two_line_strings(self):
        arrs = np.float32([
            [(0, 0), (10, 10), (5, 10)],
            [(5, 5), (15, 15), (10, 15)]
        ])

        lsoi = LineStringsOnImage.from_xy_arrays(arrs, shape=(100, 100, 3))

        assert len(lsoi.line_strings) == 2
        assert lsoi.line_strings[0].coords_almost_equals(arrs[0])
        assert lsoi.line_strings[1].coords_almost_equals(arrs[1])

    def test_from_xy_arrays_with_list_of_two_line_string_arrays(self):
        arrs = [
            np.float32([(0, 0), (10, 10), (5, 10)]),
            np.float32([(5, 5), (15, 15), (10, 15), (25, 25)])
        ]

        lsoi = LineStringsOnImage.from_xy_arrays(arrs, shape=(100, 100, 3))

        assert len(lsoi.line_strings) == 2
        assert lsoi.line_strings[0].coords_almost_equals(arrs[0])
        assert lsoi.line_strings[1].coords_almost_equals(arrs[1])

    def test_from_xy_arrays_with_empty_3d_array_and_0_points_per_ls(self):
        arrs = np.zeros((0, 0, 2), dtype=np.float32)

        lsoi = LineStringsOnImage.from_xy_arrays(arrs, shape=(100, 100, 3))

        assert len(lsoi.line_strings) == 0

    def test_from_xy_arrays_with_empty_3d_array_and_5_points_per_ls(self):
        arrs = np.zeros((0, 5, 2), dtype=np.float32)

        lsoi = LineStringsOnImage.from_xy_arrays(arrs, shape=(100, 100, 3))

        assert len(lsoi.line_strings) == 0

    def test_to_xy_arrays_with_two_line_strings_of_same_length(self):
        ls1 = LineString([(0, 0), (10, 10), (5, 10)])
        ls2 = LineString([(5, 5), (15, 15), (10, 15)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        arrs = lsoi.to_xy_arrays()

        assert isinstance(arrs, list)
        assert len(arrs) == 2
        assert arrs[0].dtype.name == "float32"
        assert arrs[1].dtype.name == "float32"
        assert np.allclose(arrs, [
            [(0, 0), (10, 10), (5, 10)],
            [(5, 5), (15, 15), (10, 15)]
        ])

    def test_to_xy_arrays_with_two_line_strings_of_different_lengths(self):
        ls1 = LineString([(0, 0), (10, 10), (5, 10)])
        ls2 = LineString([(5, 5), (15, 15), (10, 15), (25, 25)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        arrs = lsoi.to_xy_arrays()

        assert isinstance(arrs, list)
        assert len(arrs) == 2
        assert arrs[0].dtype.name == "float32"
        assert arrs[1].dtype.name == "float32"
        assert np.allclose(arrs[0], [(0, 0), (10, 10), (5, 10)])
        assert np.allclose(arrs[1], [(5, 5), (15, 15), (10, 15), (25, 25)])

    def test_to_xy_arrays_with_two_line_strings_one_of_them_empty(self):
        ls1 = LineString([])
        ls2 = LineString([(5, 5), (15, 15), (10, 15), (25, 25)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        arrs = lsoi.to_xy_arrays()

        assert isinstance(arrs, list)
        assert len(arrs) == 2
        assert arrs[0].dtype.name == "float32"
        assert arrs[1].dtype.name == "float32"
        assert arrs[0].shape == (0, 2)
        assert np.allclose(arrs[1], [(5, 5), (15, 15), (10, 15), (25, 25)])

    def test_to_xy_arrays_with_two_empty_list_of_line_strings(self):
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        arrs = lsoi.to_xy_arrays()

        assert isinstance(arrs, list)
        assert len(arrs) == 0

    def test_draw_on_image_with_default_settings(self):
        ls1 = LineString([(0, 0), (10, 10), (5, 10)])
        ls2 = LineString([(5, 5), (15, 15), (10, 15), (25, 25)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        img = np.zeros((100, 100, 3), dtype=np.uint8) + 1

        observed = lsoi.draw_on_image(img)

        expected = np.copy(img)
        for ls in [ls1, ls2]:
            expected = ls.draw_on_image(expected)
        assert np.array_equal(observed, expected)

    def test_draw_on_image_with_custom_settings(self):
        ls1 = LineString([(0, 0), (10, 10), (5, 10)])
        ls2 = LineString([(5, 5), (15, 15), (10, 15), (25, 25)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        img = np.zeros((100, 100, 3), dtype=np.uint8) + 1

        observed = lsoi.draw_on_image(img,
                                      color_lines=(0, 0, 255),
                                      color_points=(255, 0, 0),
                                      alpha_lines=0.5,
                                      alpha_points=0.6,
                                      antialiased=False)

        expected = np.copy(img)
        for ls in [ls1, ls2]:
            expected = ls.draw_on_image(expected,
                                        color_lines=(0, 0, 255),
                                        color_points=(255, 0, 0),
                                        alpha_lines=0.5,
                                        alpha_points=0.6,
                                        antialiased=False)
        assert np.array_equal(observed, expected)

    def test_draw_on_image_with_default_settings_one_ls_empty(self):
        ls1 = LineString([])
        ls2 = LineString([(5, 5), (15, 15), (10, 15), (25, 25)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        img = np.zeros((100, 100, 3), dtype=np.uint8) + 1

        observed = lsoi.draw_on_image(img)

        expected = np.copy(img)
        for ls in [ls1, ls2]:
            expected = ls.draw_on_image(expected)
        assert np.array_equal(observed, expected)

    def test_draw_on_image_with_default_settings_and_empty_list_of_ls(self):
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        img = np.zeros((100, 100, 3), dtype=np.uint8) + 1

        observed = lsoi.draw_on_image(img)

        expected = np.copy(img)
        assert np.array_equal(observed, expected)

    def test_remove_out_of_image_fraction_(self):
        item1 = ia.LineString([(5, 1), (9, 1)])
        item2 = ia.LineString([(5, 1), (15, 1)])
        item3 = ia.LineString([(15, 1), (25, 1)])
        cbaoi = ia.LineStringsOnImage([item1, item2, item3],
                                      shape=(10, 10, 3))

        cbaoi_reduced = cbaoi.remove_out_of_image_fraction_(0.6)

        assert len(cbaoi_reduced.items) == 2
        assert cbaoi_reduced.items == [item1, item2]
        assert cbaoi_reduced is cbaoi

    def test_remove_out_of_image_fraction(self):
        item1 = ia.LineString([(5, 1), (9, 1)])
        item2 = ia.LineString([(5, 1), (15, 1)])
        item3 = ia.LineString([(15, 1), (25, 1)])
        cbaoi = ia.LineStringsOnImage([item1, item2, item3],
                                      shape=(10, 10, 3))

        cbaoi_reduced = cbaoi.remove_out_of_image_fraction(0.6)

        assert len(cbaoi_reduced.items) == 2
        assert cbaoi_reduced.items == [item1, item2]
        assert cbaoi_reduced is not cbaoi

    def test_to_xy_array(self):
        lsoi = ia.LineStringsOnImage(
            [ia.LineString([(0, 0), (1, 2)]),
             ia.LineString([(10, 20), (30, 40)])],
            shape=(1, 2, 3))

        xy_out = lsoi.to_xy_array()

        expected = np.float32([
            [0.0, 0.0],
            [1.0, 2.0],
            [10.0, 20.0],
            [30.0, 40.0]
        ])
        assert xy_out.shape == (4, 2)
        assert np.allclose(xy_out, expected)
        assert xy_out.dtype.name == "float32"

    def test_to_xy_array__empty_object(self):
        lsoi = ia.LineStringsOnImage(
            [],
            shape=(1, 2, 3))

        xy_out = lsoi.to_xy_array()

        assert xy_out.shape == (0, 2)
        assert xy_out.dtype.name == "float32"

    def test_fill_from_xy_array___empty_array(self):
        xy = np.zeros((0, 2), dtype=np.float32)
        lsoi = ia.LineStringsOnImage([], shape=(2, 2, 3))

        lsoi = lsoi.fill_from_xy_array_(xy)

        assert len(lsoi.line_strings) == 0

    def test_fill_from_xy_array___empty_list(self):
        xy = []
        lsoi = ia.LineStringsOnImage([], shape=(2, 2, 3))

        lsoi = lsoi.fill_from_xy_array_(xy)

        assert len(lsoi.line_strings) == 0

    def test_fill_from_xy_array___array_with_two_coords(self):
        xy = np.array(
            [(100, 101),
             (102, 103),
             (200, 201),
             (202, 203)], dtype=np.float32)
        lsoi = ia.LineStringsOnImage(
            [ia.LineString([(0, 0), (1, 2)]),
             ia.LineString([(10, 20), (30, 40)])],
            shape=(2, 2, 3))

        lsoi = lsoi.fill_from_xy_array_(xy)

        assert len(lsoi.line_strings) == 2
        assert np.allclose(
            lsoi.line_strings[0].coords,
            [(100, 101), (102, 103)])
        assert np.allclose(
            lsoi.line_strings[1].coords,
            [(200, 201), (202, 203)])

    def test_fill_from_xy_array___list_with_two_coords(self):
        xy = [(100, 101),
              (102, 103),
              (200, 201),
              (202, 203)]
        lsoi = ia.LineStringsOnImage(
            [ia.LineString([(0, 0), (1, 2)]),
             ia.LineString([(10, 20), (30, 40)])],
            shape=(2, 2, 3))

        lsoi = lsoi.fill_from_xy_array_(xy)

        assert len(lsoi.line_strings) == 2
        assert np.allclose(
            lsoi.line_strings[0].coords,
            [(100, 101), (102, 103)])
        assert np.allclose(
            lsoi.line_strings[1].coords,
            [(200, 201), (202, 203)])

    def test_to_keypoints_on_image(self):
        lsoi = ia.LineStringsOnImage(
            [ia.LineString([(0, 0), (1, 2)]),
             ia.LineString([(10, 20), (30, 40)])],
            shape=(1, 2, 3))

        kpsoi = lsoi.to_keypoints_on_image()

        assert len(kpsoi.keypoints) == 2*2
        assert kpsoi.keypoints[0].x == 0
        assert kpsoi.keypoints[0].y == 0
        assert kpsoi.keypoints[1].x == 1
        assert kpsoi.keypoints[1].y == 2
        assert kpsoi.keypoints[2].x == 10
        assert kpsoi.keypoints[2].y == 20
        assert kpsoi.keypoints[3].x == 30
        assert kpsoi.keypoints[3].y == 40

    def test_to_keypoints_on_image__empty_instance(self):
        lsoi = ia.LineStringsOnImage([], shape=(1, 2, 3))

        kpsoi = lsoi.to_keypoints_on_image()

        assert len(kpsoi.keypoints) == 0

    def test_invert_to_keypoints_on_image_(self):
        lsoi = ia.LineStringsOnImage(
            [ia.LineString([(0, 0), (1, 2)]),
             ia.LineString([(10, 20), (30, 40)])],
            shape=(1, 2, 3))
        kpsoi = ia.KeypointsOnImage(
            [ia.Keypoint(100, 101), ia.Keypoint(102, 103),
             ia.Keypoint(110, 120), ia.Keypoint(130, 140)],
            shape=(10, 20, 30))

        lsoi_inv = lsoi.invert_to_keypoints_on_image_(kpsoi)

        assert len(lsoi_inv.line_strings) == 2
        assert lsoi_inv.shape == (10, 20, 30)
        assert np.allclose(
            lsoi.line_strings[0].coords,
            [(100, 101), (102, 103)])
        assert np.allclose(
            lsoi.line_strings[1].coords,
            [(110, 120), (130, 140)])

    def test_invert_to_keypoints_on_image___empty_instance(self):
        lsoi = ia.LineStringsOnImage([], shape=(1, 2, 3))
        kpsoi = ia.KeypointsOnImage([], shape=(10, 20, 30))

        lsoi_inv = lsoi.invert_to_keypoints_on_image_(kpsoi)

        assert len(lsoi_inv.line_strings) == 0
        assert lsoi_inv.shape == (10, 20, 30)

    def test_copy_with_two_line_strings(self):
        # basic test, without labels
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        observed = lsoi.copy()

        assert observed.line_strings[0] is ls1
        assert observed.line_strings[1] is ls2
        assert observed.shape == (100, 100, 3)

    def test_copy_with_two_line_strings_and_labels(self):
        # basic test, with labels
        ls1 = LineString([(0, 0), (1, 0), (2, 1)], label="foo")
        ls2 = LineString([(10, 10)], label="bar")
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        observed = lsoi.copy()

        assert observed.line_strings[0] is ls1
        assert observed.line_strings[1] is ls2
        assert observed.line_strings[0].label == "foo"
        assert observed.line_strings[1].label == "bar"
        assert observed.shape == (100, 100, 3)

    def test_copy_with_empty_list_of_line_strings(self):
        # LSOI is empty
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = lsoi.copy()

        assert observed.line_strings == []
        assert observed.shape == (100, 100, 3)

    def test_copy_and_replace_line_strings_attribute(self):
        # provide line_strings
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = lsoi.copy(line_strings=[ls1, ls2], shape=(200, 201, 3))

        assert observed.line_strings[0] is ls1
        assert observed.line_strings[1] is ls2
        assert observed.shape == (200, 201, 3)

    def test_copy_and_replace_line_strings_attribute_with_labeled_ls(self):
        # provide line_strings, with labels
        ls1 = LineString([(0, 0), (1, 0), (2, 1)], label="foo")
        ls2 = LineString([(10, 10)], label="bar")
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = lsoi.copy(line_strings=[ls1, ls2], shape=(200, 201, 3))

        assert observed.line_strings[0] is ls1
        assert observed.line_strings[1] is ls2
        assert observed.line_strings[0].label == "foo"
        assert observed.line_strings[1].label == "bar"
        assert observed.shape == (200, 201, 3)

    def test_copy_and_replace_line_strings_attribute_with_empty_list(self):
        # provide empty list of line_strings
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = lsoi.copy(line_strings=[], shape=(200, 201, 3))

        assert observed.line_strings == []
        assert observed.shape == (200, 201, 3)

    def test_deepcopy_with_two_line_strings(self):
        # basic test, without labels
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        observed = lsoi.deepcopy()

        assert observed.line_strings[0] is not ls1
        assert observed.line_strings[1] is not ls2
        assert observed.line_strings[0].coords_almost_equals(ls1)
        assert observed.line_strings[1].coords_almost_equals(ls2)
        assert observed.shape == (100, 100, 3)

    def test_deepcopy_with_two_line_strings_and_labels(self):
        # basic test, with labels
        ls1 = LineString([(0, 0), (1, 0), (2, 1)], label="foo")
        ls2 = LineString([(10, 10)], label="bar")
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        observed = lsoi.deepcopy()

        assert observed.line_strings[0] is not ls1
        assert observed.line_strings[1] is not ls2
        assert observed.line_strings[0].coords_almost_equals(ls1)
        assert observed.line_strings[1].coords_almost_equals(ls2)
        assert observed.line_strings[0].label == "foo"
        assert observed.line_strings[1].label == "bar"
        assert observed.shape == (100, 100, 3)

    def test_deepcopy_with_empty_list_of_line_strings(self):
        # LSOI is empty
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = lsoi.deepcopy()

        assert observed.line_strings == []
        assert observed.shape == (100, 100, 3)

    def test_deepcopy_and_replace_line_strings_attribute(self):
        # provide line_strings
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = lsoi.deepcopy(line_strings=[ls1, ls2], shape=(200, 201, 3))

        # line strings provided via line_strings are also deepcopied
        assert observed.line_strings[0].coords_almost_equals(ls1)
        assert observed.line_strings[1].coords_almost_equals(ls2)
        assert observed.shape == (200, 201, 3)

    def test_deepcopy_and_replace_line_strings_attribute_with_labeled_ls(self):
        # provide line_strings, with labels
        ls1 = LineString([(0, 0), (1, 0), (2, 1)], label="foo")
        ls2 = LineString([(10, 10)], label="bar")
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = lsoi.deepcopy(line_strings=[ls1, ls2], shape=(200, 201, 3))

        # line strings provided via line_strings are also deepcopied
        assert observed.line_strings[0].coords_almost_equals(ls1)
        assert observed.line_strings[1].coords_almost_equals(ls2)
        assert observed.line_strings[0].label == "foo"
        assert observed.line_strings[1].label == "bar"
        assert observed.shape == (200, 201, 3)

    def test_deepcopy_and_replace_line_strings_attribute_with_empty_list(self):
        # provide empty list of line_strings
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = lsoi.deepcopy(line_strings=[], shape=(200, 201, 3))

        assert observed.line_strings == []
        assert observed.shape == (200, 201, 3)

    def test___getitem__(self):
        cbas = [
            ia.LineString([(0, 0), (1, 0), (1, 1)]),
            ia.LineString([(1, 1), (2, 1), (2, 2)])
        ]
        cbasoi = ia.LineStringsOnImage(cbas, shape=(3, 4, 3))

        assert cbasoi[0] is cbas[0]
        assert cbasoi[1] is cbas[1]
        assert cbasoi[0:2] == cbas

    def test___iter__(self):
        cbas = [ia.LineString([(0, 0), (1, 1)]),
                ia.LineString([(1, 2), (3, 4)])]
        cbasoi = ia.LineStringsOnImage(cbas, shape=(40, 50, 3))

        for i, cba in enumerate(cbasoi):
            assert cba is cbas[i]

    def test___iter___empty(self):
        cbasoi = ia.LineStringsOnImage([], shape=(40, 50, 3))
        i = 0
        for _cba in cbasoi:
            i += 1
        assert i == 0

    def test___len__(self):
        cbas = [ia.LineString([(0, 0), (1, 1)]),
                ia.LineString([(1, 2), (3, 4)])]
        cbasoi = ia.LineStringsOnImage(cbas, shape=(40, 50, 3))
        assert len(cbasoi) == 2

    def test___repr__(self):
        def _func(obj):
            return obj.__repr__()

        self._test_str_repr(_func)

    def test___str__(self):
        def _func(obj):
            return obj.__str__()

        self._test_str_repr(_func)

    def test___repr___empty_list_of_line_strings(self):
        def _func(obj):
            return obj.__repr__()

        self._test_str_repr_empty_list_of_line_strings(_func)

    def test___str___empty_list_of_line_strings(self):
        def _func(obj):
            return obj.__str__()

        self._test_str_repr_empty_list_of_line_strings(_func)

    @classmethod
    def _test_str_repr(cls, func):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))

        observed = func(lsoi)

        expected = "LineStringsOnImage([%s, %s], shape=(100, 100, 3))" % (
            func(ls1), func(ls2)
        )
        assert observed == expected

    @classmethod
    def _test_str_repr_empty_list_of_line_strings(cls, func):
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))

        observed = func(lsoi)

        expected = "LineStringsOnImage([], shape=(100, 100, 3))"
        assert observed == expected


def _coords_eq(ls, other):
    return ls.coords_almost_equals(other, max_distance=1e-2)


def _drawing_allclose(arr, v):
    # draw_points_heatmaps_array() and draw_line_heatmap_array() are
    # currently limited to 1/255 accuracy due to drawing in uint8
    return np.allclose(arr, v, atol=(1.01/255), rtol=0)
