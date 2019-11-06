from __future__ import print_function, division, absolute_import

import time
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
import shapely
import shapely.geometry

import imgaug as ia
import imgaug.random as iarandom
from imgaug.testutils import reseed
from imgaug.augmentables.polys import _ConcavePolygonRecoverer


class TestPolygon___init__(unittest.TestCase):
    def test_exterior_is_list_of_keypoints(self):
        kps = [ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=0.5, y=2.5)]

        poly = ia.Polygon(kps)

        assert poly.exterior.dtype.name == "float32"
        assert np.allclose(
            poly.exterior,
            np.float32([
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, 2.5]
            ])
        )

    def test_exterior_is_list_of_tuples_of_floats(self):
        poly = ia.Polygon([(0.0, 0.0), (1.0, 1.0), (0.5, 2.5)])

        assert poly.exterior.dtype.name == "float32"
        assert np.allclose(
            poly.exterior,
            np.float32([
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, 2.5]
            ])
        )

    def test_exterior_is_list_of_tuples_of_ints(self):
        poly = ia.Polygon([(0, 0), (1, 1), (1, 3)])

        assert poly.exterior.dtype.name == "float32"
        assert np.allclose(
            poly.exterior,
            np.float32([
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 3.0]
            ])
        )

    def test_exterior_is_float32_array(self):
        poly = ia.Polygon(
            np.float32([
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, 2.5]
            ])
        )

        assert poly.exterior.dtype.name == "float32"
        assert np.allclose(
            poly.exterior,
            np.float32([
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, 2.5]
            ])
        )

    def test_exterior_is_float64_array(self):
        poly = ia.Polygon(
            np.float64([
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, 2.5]
            ])
        )

        assert poly.exterior.dtype.name == "float32"
        assert np.allclose(
            poly.exterior,
            np.float32([
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, 2.5]
            ])
        )

    def test_exterior_is_empty_list(self):
        poly = ia.Polygon([])

        assert poly.exterior.dtype.name == "float32"
        assert poly.exterior.shape == (0, 2)

    def test_exterior_is_empty_array(self):
        poly = ia.Polygon(np.zeros((0, 2), dtype=np.float32))

        assert poly.exterior.dtype.name == "float32"
        assert poly.exterior.shape == (0, 2)

    def test_fails_if_exterior_is_array_with_wrong_shape(self):
        with self.assertRaises(AssertionError):
            _ = ia.Polygon(np.zeros((8,), dtype=np.float32))

    def test_label_is_none(self):
        poly = ia.Polygon([(0, 0)])

        assert poly.label is None

    def test_label_is_string(self):
        poly = ia.Polygon([(0, 0)], label="test")

        assert poly.label == "test"


class TestPolygon_coords(unittest.TestCase):
    def test_with_three_points(self):
        poly = ia.Polygon([(0, 0), (1, 0.5), (1.5, 2.0)])
        assert poly.coords is poly.exterior


class TestPolygon_xx(unittest.TestCase):
    def test_filled_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1.5, 0), (4.1, 1), (2.9, 2.0)])

        assert poly.xx.dtype.name == "float32"
        assert np.allclose(poly.xx, np.float32([0.0, 1.0, 1.5, 4.1, 2.9]))

    def test_empty_polygon(self):
        poly = ia.Polygon([])

        assert poly.xx.dtype.name == "float32"
        assert poly.xx.shape == (0,)


class TestPolygon_yy(unittest.TestCase):
    def test_filled_polygon(self):
        poly = ia.Polygon([(0, 0), (0, 1), (0, 1.5), (1, 4.1), (2.0, 2.9)])

        assert poly.yy.dtype.name == "float32"
        assert np.allclose(poly.yy, np.float32([0.0, 1.0, 1.5, 4.1, 2.9]))

    def test_empty_polygon(self):
        poly = ia.Polygon([])

        assert poly.yy.dtype.name == "float32"
        assert poly.yy.shape == (0,)


class TestPolygon_xx_int(unittest.TestCase):
    def test_filled_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1.5, 0), (4.1, 1), (2.9, 2.0)])

        assert poly.xx_int.dtype.name == "int32"
        assert np.allclose(poly.xx_int, np.int32([0, 1, 2, 4, 3]))

    def test_empty_polygon(self):
        poly = ia.Polygon([])

        assert poly.xx_int.dtype.name == "int32"
        assert poly.xx_int.shape == (0,)


class TestPolygon_yy_int(unittest.TestCase):
    def test_filled_polygon(self):
        poly = ia.Polygon([(0, 0), (0, 1), (0, 1.5), (1, 4.1), (2.0, 2.9)])

        assert poly.yy_int.dtype.name == "int32"
        assert np.allclose(poly.yy_int, np.int32([0, 1, 2, 4, 3]))

    def test_empty_polygon(self):
        poly = ia.Polygon([])

        assert poly.yy_int.dtype.name == "int32"
        assert poly.yy_int.shape == (0,)


class TestPolygon_is_valid(unittest.TestCase):
    def test_filled_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert poly.is_valid

    def test_empty_polygon(self):
        poly = ia.Polygon([])
        assert not poly.is_valid

    def test_polygon_with_one_point(self):
        poly = ia.Polygon([(0, 0)])
        assert not poly.is_valid

    def test_polygon_with_two_points(self):
        poly = ia.Polygon([(0, 0), (1, 0)])
        assert not poly.is_valid

    def test_polygon_with_self_intersection(self):
        # self intersection around the line segment from (0, 1) to (0, 0)
        poly = ia.Polygon([(0, 0), (1, 0), (-1, 0.5), (1, 1), (0, 1)])
        assert not poly.is_valid

    def test_polygon_with_consecutive_identical_points(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 0), (1, 1), (0, 1)])
        assert poly.is_valid


class TestPolygon_area(unittest.TestCase):
    def test_square_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert 1.0 - 1e-8 < poly.area < 1.0 + 1e-8

    def test_rectangular_polygon(self):
        poly = ia.Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
        assert 2.0 - 1e-8 < poly.area < 2.0 + 1e-8

    def test_triangular_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 1), (0, 1)])
        assert 1/2 - 1e-8 < poly.area < 1/2 + 1e-8

    def test_polygon_with_two_points(self):
        poly = ia.Polygon([(0, 0), (1, 1)])
        assert np.isclose(poly.area, 0.0)

    def test_polygon_with_one_point(self):
        poly = ia.Polygon([(0, 0)])
        assert np.isclose(poly.area, 0.0)

    def test_polygon_with_zero_points(self):
        poly = ia.Polygon([])
        assert np.isclose(poly.area, 0.0)


class TestPolygon_height(unittest.TestCase):
    def test_square_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert np.allclose(poly.height, 1.0, atol=1e-8, rtol=0)

    def test_rectangular_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
        assert np.allclose(poly.height, 2.0, atol=1e-8, rtol=0)

    def test_triangular_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 1), (0, 1)])
        assert np.allclose(poly.height, 1.0, atol=1e-8, rtol=0)

    def test_polygon_with_two_points(self):
        poly = ia.Polygon([(0, 0), (1, 1)])
        assert np.allclose(poly.height, 1.0, atol=1e-8, rtol=0)

    def test_polygon_with_one_point(self):
        poly = ia.Polygon([(0, 0)])
        assert np.allclose(poly.height, 0.0, atol=1e-8, rtol=0)


class TestPolygon_width(unittest.TestCase):
    def test_square_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert np.allclose(poly.width, 1.0, atol=1e-8, rtol=0)

    def test_rectangular_polygon(self):
        poly = ia.Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
        assert np.allclose(poly.width, 2.0, atol=1e-8, rtol=0)

    def test_triangular_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 1), (0, 1)])
        assert np.allclose(poly.width, 1.0, atol=1e-8, rtol=0)

    def test_polygon_with_two_points(self):
        poly = ia.Polygon([(0, 0), (1, 1)])
        assert np.allclose(poly.width, 1.0, atol=1e-8, rtol=0)

    def test_polygon_with_one_point(self):
        poly = ia.Polygon([(0, 0)])
        assert np.allclose(poly.width, 0.0, atol=1e-8, rtol=0)


class TestPolygon_compute_area_out_of_image(unittest.TestCase):
    def test_fully_inside_image_plane(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        image_shape = (10, 20, 3)
        area_ooi = poly.compute_area_out_of_image(image_shape)
        assert np.isclose(area_ooi, 0.0)

    def test_partially_outside_of_image_plane(self):
        poly = ia.Polygon([(-1, 0), (1, 0), (1, 2), (-1, 2)])
        image_shape = (10, 20, 3)
        area_ooi = poly.compute_area_out_of_image(image_shape)
        assert np.isclose(area_ooi, 2.0)

    def test_fully_outside_of_image_plane(self):
        poly = ia.Polygon([(-1, 0), (0, 0), (0, 1), (-1, 1)])
        image_shape = (10, 20, 3)
        area_ooi = poly.compute_area_out_of_image(image_shape)
        assert np.isclose(area_ooi, 1.0)

    def test_multiple_polygons_after_clip(self):
        # two polygons inside the image area remain after clipping
        # result is (area - poly1 - poly2) or here the part of the polygon
        # that is left of the y-axis (x=0.0)
        poly = ia.Polygon([(-10, 0), (5, 0), (5, 5), (-5, 5),
                           (-5, 10), (5, 10),
                           (5, 15), (-10, 15)])
        image_shape = (15, 10, 3)

        area_ooi = poly.compute_area_out_of_image(image_shape)

        # the part left of the y-axis is not exactly square, but has a hole
        # on its right (vertically centered), hence we have to subtract 5*5
        assert np.isclose(area_ooi, 10*15 - 5*5)


class TestPolygon_project(unittest.TestCase):
    def test_project_square_to_image_of_identical_shape(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        poly_proj = poly.project((1, 1), (1, 1))

        assert poly_proj.exterior.dtype.name == "float32"
        assert poly_proj.exterior.shape == (4, 2)
        assert np.allclose(
            poly_proj.exterior,
            np.float32([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1]
            ])
        )

    def test_project_square_to_image_with_twice_the_height_and_width(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        poly_proj = poly.project((1, 1), (2, 2))

        assert poly_proj.exterior.dtype.name == "float32"
        assert poly_proj.exterior.shape == (4, 2)
        assert np.allclose(
            poly_proj.exterior,
            np.float32([
                [0, 0],
                [2, 0],
                [2, 2],
                [0, 2]
            ])
        )

    def test_project_square_to_image_with_twice_the_height_but_same_width(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        poly_proj = poly.project((1, 1), (2, 1))

        assert poly_proj.exterior.dtype.name == "float32"
        assert poly_proj.exterior.shape == (4, 2)
        assert np.allclose(
            poly_proj.exterior,
            np.float32([
                [0, 0],
                [1, 0],
                [1, 2],
                [0, 2]
            ])
        )

    def test_project_empty_exterior(self):
        poly = ia.Polygon([])
        poly_proj = poly.project((1, 1), (2, 2))
        assert poly_proj.exterior.dtype.name == "float32"
        assert poly_proj.exterior.shape == (0, 2)


class TestPolygon_find_closest_point_idx(unittest.TestCase):
    def test_without_return_distance(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        coords = [(0, 0), (1, 0), (1.0001, -0.001), (0.2, 0.2)]
        expected_indices = [0, 1, 1, 0]

        for (x, y), expected_index in zip(coords, expected_indices):
            with self.subTest(x=x, y=0):
                closest_idx = poly.find_closest_point_index(x=x, y=y)
                assert closest_idx == expected_index

    def test_with_return_distance(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        coords = [(0, 0), (0.1, 0.15), (0.9, 0.15)]
        expected_indices = [0, 0, 1]
        expected_distances = [
            0.0,
            np.sqrt((0.1**2) + (0.15**2)),
            np.sqrt(((1.0-0.9)**2) + (0.15**2))
        ]

        gen = zip(coords, expected_indices, expected_distances)
        for (x, y), expected_index, expected_dist in gen:
            with self.subTest(x=x, y=y):
                closest_idx, distance = poly.find_closest_point_index(
                    x=x, y=y, return_distance=True)
                assert closest_idx == expected_index
                assert np.allclose(distance, expected_dist)

    def test_fails_for_empty_exterior(self):
        poly = ia.Polygon([])
        with self.assertRaises(AssertionError):
            _ = poly.find_closest_point_index(x=0, y=0)


class TestPolygon_is_fully_within_image(unittest.TestCase):
    def test_barely_within_image__shape_as_3d_tuple(self):
        poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
        assert poly.is_fully_within_image((1, 1, 3))

    def test_barely_within_image__shape_as_2d_tuple(self):
        poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
        assert poly.is_fully_within_image((1, 1))

    def test_barely_within_image__shape_as_ndarray(self):
        poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
        assert poly.is_fully_within_image(
            np.zeros((1, 1, 3), dtype=np.uint8)
        )

    def test_right_and_bottom_sides_overlap__shape_as_3d_tuple(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert not poly.is_fully_within_image((1, 1, 3))

    def test_right_and_bottom_sides_overlap__shape_as_2d_tuple(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert not poly.is_fully_within_image((1, 1))

    def test_right_and_bottom_sides_overlap__shape_as_ndarray(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert not poly.is_fully_within_image(
            np.zeros((1, 1, 3), dtype=np.uint8)
        )

    def test_far_outside_of_image__shape_as_3d_tuple(self):
        poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
        assert not poly.is_fully_within_image((1, 1, 3))

    def test_exterior_empty_fails(self):
        poly = ia.Polygon([])
        with self.assertRaises(Exception):
            _ = poly.is_fully_within_image((1, 1, 3))


class TestPolygon_is_partly_within_image(unittest.TestCase):
    def test_barely_within_image__shape_as_3d_tuple(self):
        poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
        assert poly.is_partly_within_image((1, 1, 3))

    def test_barely_within_image__shape_as_2d_tuple(self):
        poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
        assert poly.is_partly_within_image((1, 1))

    def test_barely_within_image__shape_as_ndarray(self):
        poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
        assert poly.is_partly_within_image(
            np.zeros((1, 1, 3), dtype=np.uint8)
        )

    def test_right_and_bottom_sides_overlap__shape_as_3d_tuple(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert poly.is_partly_within_image((1, 1, 3))

    def test_right_and_bottom_sides_overlap__shape_as_2d_tuple(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert poly.is_partly_within_image((1, 1))

    def test_right_and_bottom_sides_overlap__shape_as_ndarray(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert poly.is_partly_within_image(np.zeros((1, 1, 3), dtype=np.uint8))

    def test_far_outside_of_image__shape_as_3d_tuple(self):
        poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
        assert not poly.is_partly_within_image((1, 1, 3))

    def test_far_outside_of_image__shape_as_2d_tuple(self):
        poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
        assert not poly.is_partly_within_image((1, 1))

    def test_far_outside_of_image__shape_as_ndarray(self):
        poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
        assert not poly.is_partly_within_image(
            np.zeros((1, 1, 3), dtype=np.uint8)
        )

    def test_exterior_empty_fails(self):
        poly = ia.Polygon([])
        with self.assertRaises(Exception):
            _ = poly.is_partly_within_image((1, 1, 3))


class TestPolygon_is_out_of_image(unittest.TestCase):
    def test_barely_within_image(self):
        shapes = [(1, 1, 3), (1, 1), np.zeros((1, 1, 3), dtype=np.uint8)]
        for shape in shapes:
            shape_str = (
                str(shape) if isinstance(shape, tuple) else str(shape.shape)
            )
            with self.subTest(shape=shape_str):
                poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
                is_ooi = poly.is_out_of_image
                assert not is_ooi(shape, partly=False, fully=False)
                assert not is_ooi(shape, partly=True, fully=False)
                assert not is_ooi(shape, partly=False, fully=True)
                assert not is_ooi(shape, partly=True, fully=True)

    def test_right_and_bottom_sides_overlap__shape_as_ndarray(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        shape = np.zeros((1, 1, 3), dtype=np.uint8)

        assert not poly.is_out_of_image(shape, partly=False, fully=False)
        assert poly.is_out_of_image(shape, partly=True, fully=False)
        assert not poly.is_out_of_image(shape, partly=False, fully=True)
        assert poly.is_out_of_image(shape, partly=True, fully=True)

    def test_far_outside_of_image__shape_as_3d_tuple(self):
        poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
        shape = (1, 1, 3)

        assert not poly.is_out_of_image(shape, partly=False, fully=False)
        assert not poly.is_out_of_image(shape, partly=True, fully=False)
        assert poly.is_out_of_image(shape, partly=False, fully=True)
        assert poly.is_out_of_image(shape, partly=True, fully=True)

    def test_triangle_partially_outside_of_image(self):
        poly = ia.Polygon([(8, 11), (11, 8), (11, 11)])
        assert not poly.is_out_of_image((100, 100, 3), fully=True, partly=True)
        assert not poly.is_out_of_image((10, 10, 3), fully=True, partly=False)
        assert poly.is_out_of_image((10, 10, 3), fully=False, partly=True)

    def test_rectangle_with_all_corners_outside_of_the_image(self):
        poly = ia.Polygon([(-1.0, -1.0), (2.0, -1.0), (2.0, 2.0), (-1.0, 2.0)])
        assert not poly.is_out_of_image((100, 100, 3), fully=True, partly=False)
        assert poly.is_out_of_image((100, 100, 3), fully=False, partly=True)
        assert not poly.is_out_of_image((1, 1, 3), fully=True, partly=False)
        assert poly.is_out_of_image((1, 1, 3), fully=False, partly=True)
        assert poly.is_out_of_image((1, 1, 3), fully=True, partly=True)

    def test_polygon_with_two_points(self):
        poly = ia.Polygon([(2.0, 2.0), (10.0, 2.0)])
        assert not poly.is_out_of_image((100, 100, 3), fully=True, partly=False)
        assert not poly.is_out_of_image((100, 100, 3), fully=False, partly=True)
        assert not poly.is_out_of_image((3, 3, 3), fully=True, partly=False)
        assert poly.is_out_of_image((3, 3, 3), fully=False, partly=True)
        assert poly.is_out_of_image((1, 1, 3), fully=True, partly=False)
        assert not poly.is_out_of_image((1, 1, 3), fully=False, partly=True)

    def test_polygon_with_one_point(self):
        poly = ia.Polygon([(2.0, 2.0)])
        assert not poly.is_out_of_image((100, 100, 3), fully=True, partly=False)
        assert not poly.is_out_of_image((100, 100, 3), fully=False, partly=True)
        assert poly.is_out_of_image((1, 1, 3), fully=True, partly=False)
        assert not poly.is_out_of_image((1, 1, 3), fully=False, partly=True)

    def test_polygon_with_zero_points_fails(self):
        poly = ia.Polygon([])
        got_exception = False
        try:
            poly.is_out_of_image((1, 1, 3))
        except Exception as exc:
            assert (
                "Cannot determine whether the polygon is inside the "
                "image" in str(exc))
            got_exception = True
        assert got_exception


class TestPolygon_cut_out_of_image(unittest.TestCase):
    @mock.patch("imgaug.augmentables.polys.Polygon.clip_out_of_image")
    def test_warns_of_deprecation(self, mock_clip):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            polygon = ia.Polygon([(0, 0), (1, 0), (1, 1)])
            shape = (1, 1)
            _ = polygon.cut_out_of_image(shape)

        mock_clip.assert_called_once_with(shape)
        assert "is deprecated" in str(caught_warnings[0].message)


class TestPolygon_clip_out_of_image(unittest.TestCase):
    def test_polygon_inside_of_image(self):
        # poly inside image
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label=None)
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        multipoly_clipped = poly.clip_out_of_image(image)
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 1
        assert multipoly_clipped[0].exterior_almost_equals(poly.exterior)
        assert multipoly_clipped[0].label is None

    def test_polygon_half_outside_of_image(self):
        # square poly shifted by x=0.5, y=0.5 => half out of image
        poly = ia.Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)], label="test")
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        multipoly_clipped = poly.clip_out_of_image(image)
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 1
        assert multipoly_clipped[0].exterior_almost_equals(np.float32([
            [0.5, 0.5],
            [1.0, 0.5],
            [1.0, 1.0],
            [0.5, 1.0]
        ]))
        assert multipoly_clipped[0].label == "test"

    def test_single_edge_intersecting_with_image_edge(self):
        # square poly with a single edge intersecting the image (issue #310)
        poly = ia.Polygon([(-1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (-1.0, 1.0)])
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        multipoly_clipped = poly.clip_out_of_image(image)
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 0

    def test_tiny_area_around_image_edge_intersecting(self):
        # square poly with a tiny area on the left image edge intersecting with
        # the image
        offset = 1e-4
        poly = ia.Polygon([(-1.0, 0.0), (0.0+offset, 0.0),
                           (0.0+offset, 1.0), (-1.0, 1.0)])
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        multipoly_clipped = poly.clip_out_of_image(image)
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 1
        assert multipoly_clipped[0].exterior_almost_equals(np.float32([
            [0.0, 0.0],
            [0.0+offset, 0.0],
            [0.0+offset, 1.0],
            [0.0, 1.0]
        ]))

    def test_single_point_intersecting_with_image(self):
        # square poly with a single point intersecting the image (issue #310)
        poly = ia.Polygon([(-1.0, -1.0), (0.0, -1.0), (0.0, 0.0), (-1.0, 0.0)])
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        multipoly_clipped = poly.clip_out_of_image(image)
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 0

    def test_tiny_area_around_image_corner_point_intersecting_with_image(self):
        # square poly with a tiny area around the top left image corner
        # intersecting with the the image
        offset = 1e-4
        poly = ia.Polygon([(-1.0, -1.0), (0.0, -1.0),
                           (0.0+offset, 0.0+offset), (-1.0, 0.0)])
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        multipoly_clipped = poly.clip_out_of_image(image)
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 1
        assert multipoly_clipped[0].exterior_almost_equals(np.float32([
            [0.0, 0.0],
            [0.0+offset, 0.0],
            [0.0+offset, 0.0+offset],
            [0.0, 0.0+offset]
        ]))

    def test_polygon_clipped_to_two_separate_polygons(self):
        # non-square poly, with one rectangle on the left side of the image
        # and one on the right side, both sides are connected by a thin strip
        # below the image after clipping it should become two rectangles
        poly = ia.Polygon([(-0.1, 0.0), (0.4, 0.0), (0.4, 1.1), (0.6, 1.1),
                           (0.6, 0.0), (1.1, 0.0), (1.1, 1.2), (-0.1, 1.2)],
                          label="test")
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        multipoly_clipped = poly.clip_out_of_image(image)
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 2
        assert multipoly_clipped[0].exterior_almost_equals(np.float32([
            [0.0, 0.0],
            [0.4, 0.0],
            [0.4, 1.0],
            [0.0, 1.0]
        ]))
        assert multipoly_clipped[0].label == "test"
        assert multipoly_clipped[1].exterior_almost_equals(np.float32([
            [0.6, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.6, 1.0]
        ]))
        assert multipoly_clipped[0].label == "test"

    def test_polygon_fully_outside_of_the_image(self):
        # poly outside of image
        poly = ia.Polygon([(10.0, 10.0), (11,.0, 10.0), (11.0, 11.0)])
        multipoly_clipped = poly.clip_out_of_image((5, 5, 3))
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 0

    def test_small_intersection_with_image_one_poly_point_inside_image(self):
        # poly area partially inside image
        # and one point is inside the image
        poly = ia.Polygon([(50, 50), (110, 50), (110, 110), (50, 110)])
        multipoly_clipped = poly.clip_out_of_image((100, 100, 3))
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 1
        assert multipoly_clipped[0].exterior_almost_equals(np.float32([
            [50, 50],
            [100, 50],
            [100, 100],
            [50, 100]
        ]))

    def test_small_intersection_with_image_no_poly_point_inside_image(self):
        # poly area partially inside image,
        # but not a single point is inside the image
        poly = ia.Polygon([(100+0.5*100, 0),
                           (100+0.5*100, 100+0.5*100),
                           (0, 100+0.5*100)])
        multipoly_clipped = poly.clip_out_of_image((100, 100, 3))
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 1
        assert multipoly_clipped[0].exterior_almost_equals(np.float32([
            [100, 0.5*100],
            [100, 100],
            [0.5*100, 100]
        ]))

    def test_polygon_with_two_points_that_is_not_clipped(self):
        # polygon with two points
        poly = ia.Polygon([(2.0, 2.0), (10.0, 2.0)])
        multipoly_clipped = poly.clip_out_of_image((100, 100, 3))
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 1
        assert multipoly_clipped[0].exterior_almost_equals(np.float32([
            [2.0, 2.0],
            [10.0, 2.0]
        ]))

    def test_polygon_with_two_points_that_is_clipped(self):
        poly = ia.Polygon([(2.0, 2.0), (10.0, 2.0)])
        multipoly_clipped = poly.clip_out_of_image((3, 3, 3))
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 1
        assert multipoly_clipped[0].exterior_almost_equals(np.float32([
            [2.0, 2.0],
            [3.0, 2.0]
        ]), max_distance=1e-3)

    def test_polygon_with_one_point_that_is_not_clipped(self):
        # polygon with a single point
        poly = ia.Polygon([(2.0, 2.0)])
        multipoly_clipped = poly.clip_out_of_image((3, 3, 3))
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 1
        assert multipoly_clipped[0].exterior_almost_equals(np.float32([
            [2.0, 2.0]
        ]))

    def test_polygon_with_one_point_that_is_clipped(self):
        poly = ia.Polygon([(2.0, 2.0)])
        multipoly_clipped = poly.clip_out_of_image((1, 1, 3))
        assert isinstance(multipoly_clipped, list)
        assert len(multipoly_clipped) == 0


class TestPolygon_shift(unittest.TestCase):
    @property
    def poly(self):
        return ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")

    def test_shift_does_not_work_inplace(self):
        # make sure that shift does not change poly inplace
        poly = self.poly
        poly_shifted = poly.shift(top=1)
        assert np.allclose(poly.exterior, np.float32([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ]))
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0, 1],
            [1, 1],
            [1, 2],
            [0, 2]
        ]))

    def test_shift_from_top(self):
        for v in [1, 0, -1, 0.5]:
            with self.subTest(top=v):
                poly_shifted = self.poly.shift(top=v)
                assert np.allclose(poly_shifted.exterior, np.float32([
                    [0, 0 + v],
                    [1, 0 + v],
                    [1, 1 + v],
                    [0, 1 + v]
                ]))
                assert poly_shifted.label == "test"

    def test_shift_from_bottom(self):
        for v in [1, 0, -1, 0.5]:
            with self.subTest(bottom=v):
                poly_shifted = self.poly.shift(bottom=v)
                assert np.allclose(poly_shifted.exterior, np.float32([
                    [0, 0 - v],
                    [1, 0 - v],
                    [1, 1 - v],
                    [0, 1 - v]
                ]))
                assert poly_shifted.label == "test"

    def test_shift_from_top_and_bottom(self):
        for v in [1, 0, -1, 0.5]:
            with self.subTest(top=v, bottom=-v):
                poly_shifted = self.poly.shift(top=v, bottom=-v)
                assert np.allclose(poly_shifted.exterior, np.float32([
                    [0, 0 + 2*v],
                    [1, 0 + 2*v],
                    [1, 1 + 2*v],
                    [0, 1 + 2*v]
                ]))
                assert poly_shifted.label == "test"

    def test_shift_from_left(self):
        for v in [1, 0, -1, 0.5]:
            with self.subTest(left=v):
                poly_shifted = self.poly.shift(left=v)
                assert np.allclose(poly_shifted.exterior, np.float32([
                    [0 + v, 0],
                    [1 + v, 0],
                    [1 + v, 1],
                    [0 + v, 1]
                ]))
                assert poly_shifted.label == "test"

    def test_shift_from_right(self):
        for v in [1, 0, -1, 0.5]:
            with self.subTest(right=v):
                poly_shifted = self.poly.shift(right=v)
                assert np.allclose(poly_shifted.exterior, np.float32([
                    [0 - v, 0],
                    [1 - v, 0],
                    [1 - v, 1],
                    [0 - v, 1]
                ]))
                assert poly_shifted.label == "test"

    def test_shift_from_left_and_right(self):
        for v in [1, 0, -1, 0.5]:
            with self.subTest(left=v, right=-v):
                poly_shifted = self.poly.shift(left=v, right=-v)
                assert np.allclose(poly_shifted.exterior, np.float32([
                    [0 + 2 * v, 0],
                    [1 + 2 * v, 0],
                    [1 + 2 * v, 1],
                    [0 + 2 * v, 1]
                ]))
                assert poly_shifted.label == "test"


class TestPolygon_draw_on_image(unittest.TestCase):
    @property
    def image(self):
        return np.tile(
            np.arange(100).reshape((10, 10, 1)),
            (1, 1, 3)
        ).astype(np.uint8)

    def test_square_polygon(self):
        # simple drawing of square
        poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        image = self.image
        image_poly = poly.draw_on_image(image,
                                        color=[32, 128, 32],
                                        color_face=[32, 128, 32],
                                        color_lines=[0, 255, 0],
                                        color_points=[0, 255, 0],
                                        alpha=1.0,
                                        alpha_face=1.0,
                                        alpha_lines=1.0,
                                        alpha_points=0.0,
                                        raise_if_out_of_image=False)
        assert image_poly.dtype.type == np.uint8
        assert image_poly.shape == (10, 10, 3)
        # draw did not change original image (copy=True)
        assert np.sum(image) == 3 * np.sum(np.arange(100))

        for c_idx, value in enumerate([0, 255, 0]):
            # left boundary
            assert np.all(image_poly[2:9, 2:3, c_idx]
                          == np.zeros((7, 1), dtype=np.uint8) + value)
            # right boundary
            assert np.all(image_poly[2:9, 8:9, c_idx]
                          == np.zeros((7, 1), dtype=np.uint8) + value)
            # top boundary
            assert np.all(image_poly[2:3, 2:9, c_idx]
                          == np.zeros((1, 7), dtype=np.uint8) + value)
            # bottom boundary
            assert np.all(image_poly[8:9, 2:9, c_idx]
                          == np.zeros((1, 7), dtype=np.uint8) + value)
        expected = np.tile(
            np.uint8([32, 128, 32]).reshape((1, 1, 3)),
            (5, 5, 1)
        )
        assert np.all(image_poly[3:8, 3:8, :] == expected)

    def test_square_polygon_use_no_color_subargs(self):
        # simple drawing of square, use only "color" arg
        poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        image = self.image
        image_poly = poly.draw_on_image(image,
                                        color=[0, 255, 0],
                                        alpha=1.0,
                                        alpha_face=1.0,
                                        alpha_lines=1.0,
                                        alpha_points=0.0,
                                        raise_if_out_of_image=False)
        assert image_poly.dtype.type == np.uint8
        assert image_poly.shape == (10, 10, 3)
        # draw did not change original image (copy=True)
        assert np.sum(image) == 3 * np.sum(np.arange(100))

        for c_idx, value in enumerate([0, 0.5*255, 0]):
            value = int(np.round(value))
            # left boundary
            assert np.all(image_poly[2:9, 2:3, c_idx]
                          == np.zeros((7, 1), dtype=np.uint8) + value)
            # right boundary
            assert np.all(image_poly[2:9, 8:9, c_idx]
                          == np.zeros((7, 1), dtype=np.uint8) + value)
            # top boundary
            assert np.all(image_poly[2:3, 2:9, c_idx]
                          == np.zeros((1, 7), dtype=np.uint8) + value)
            # bottom boundary
            assert np.all(image_poly[8:9, 2:9, c_idx]
                          == np.zeros((1, 7), dtype=np.uint8) + value)
        expected = np.tile(
            np.uint8([0, 255, 0]).reshape((1, 1, 3)),
            (5, 5, 1)
        )
        assert np.all(image_poly[3:8, 3:8, :] == expected)

    def test_square_polygon_on_float32_image(self):
        # simple drawing of square with float32 input
        poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        image = self.image
        image_poly = poly.draw_on_image(image.astype(np.float32),
                                        color=[32, 128, 32],
                                        color_face=[32, 128, 32],
                                        color_lines=[0, 255, 0],
                                        color_points=[0, 255, 0],
                                        alpha=1.0,
                                        alpha_face=1.0,
                                        alpha_lines=1.0,
                                        alpha_points=0.0,
                                        raise_if_out_of_image=False)
        assert image_poly.dtype.type == np.float32
        assert image_poly.shape == (10, 10, 3)
        for c_idx, value in enumerate([0, 255, 0]):
            # left boundary
            assert np.allclose(image_poly[2:9, 2:3, c_idx],
                               np.zeros((7, 1), dtype=np.float32) + value)
            # right boundary
            assert np.allclose(image_poly[2:9, 8:9, c_idx],
                               np.zeros((7, 1), dtype=np.float32) + value)
            # top boundary
            assert np.allclose(image_poly[2:3, 2:9, c_idx],
                               np.zeros((1, 7), dtype=np.float32) + value)
            # bottom boundary
            assert np.allclose(image_poly[8:9, 2:9, c_idx],
                               np.zeros((1, 7), dtype=np.float32) + value)
        expected = np.tile(
            np.float32([32, 128, 32]).reshape((1, 1, 3)),
            (5, 5, 1)
        )
        assert np.allclose(image_poly[3:8, 3:8, :], expected)

    def test_square_polygon_half_outside_of_image(self):
        # drawing of poly that is half out of image
        poly = ia.Polygon([(2, 2+5), (8, 2+5), (8, 8+5), (2, 8+5)])
        image = self.image
        image_poly = poly.draw_on_image(image,
                                        color=[32, 128, 32],
                                        color_face=[32, 128, 32],
                                        color_lines=[0, 255, 0],
                                        color_points=[0, 255, 0],
                                        alpha=1.0,
                                        alpha_face=1.0,
                                        alpha_lines=1.0,
                                        alpha_points=0.0,
                                        raise_if_out_of_image=False)
        assert image_poly.dtype.type == np.uint8
        assert image_poly.shape == (10, 10, 3)
        # draw did not change original image (copy=True)
        assert np.sum(image) == 3 * np.sum(np.arange(100))

        for c_idx, value in enumerate([0, 255, 0]):
            # left boundary
            assert np.all(image_poly[2+5:, 2:3, c_idx]
                          == np.zeros((3, 1), dtype=np.uint8) + value)
            # right boundary
            assert np.all(image_poly[2+5:, 8:9, c_idx]
                          == np.zeros((3, 1), dtype=np.uint8) + value)
            # top boundary
            assert np.all(image_poly[2+5:3+5, 2:9, c_idx]
                          == np.zeros((1, 7), dtype=np.uint8) + value)
        expected = np.tile(
            np.uint8([32, 128, 32]).reshape((1, 1, 3)),
            (2, 5, 1)
        )
        assert np.all(image_poly[3+5:, 3:8, :] == expected)

    def test_square_polygon_half_outside_of_image_with_raise_if_ooi(self):
        # drawing of poly that is half out of image, with
        # raise_if_out_of_image=True
        poly = ia.Polygon([(2, 2+5), (8, 2+5), (8, 8+5), (0, 8+5)])
        image = self.image
        got_exception = False
        try:
            _ = poly.draw_on_image(image,
                                   color=[32, 128, 32],
                                   color_face=[32, 128, 32],
                                   color_lines=[0, 255, 0],
                                   color_points=[0, 255, 0],
                                   alpha=1.0,
                                   alpha_face=1.0,
                                   alpha_lines=1.0,
                                   alpha_points=0.0,
                                   raise_if_out_of_image=True)
        except Exception as exc:
            assert "Cannot draw polygon" in str(exc)
            got_exception = True
        # only polygons fully outside of the image plane lead to exceptions
        assert not got_exception

    def test_polygon_fully_outside_of_image(self):
        # drawing of poly that is fully out of image
        poly = ia.Polygon([(100, 100), (100+10, 100), (100+10, 100+10),
                           (100, 100+10)])
        image = self.image
        image_poly = poly.draw_on_image(image,
                                        color=[32, 128, 32],
                                        color_face=[32, 128, 32],
                                        color_lines=[0, 255, 0],
                                        color_points=[0, 255, 0],
                                        alpha=1.0,
                                        alpha_face=1.0,
                                        alpha_lines=1.0,
                                        alpha_points=0.0,
                                        raise_if_out_of_image=False)
        assert np.array_equal(image_poly, image)

    def test_polygon_fully_outside_of_image_with_raise_if_ooi(self):
        # drawing of poly that is fully out of image,
        # with raise_if_out_of_image=True
        poly = ia.Polygon([(100, 100), (100+10, 100), (100+10, 100+10),
                           (100, 100+10)])
        image = self.image
        got_exception = False
        try:
            _ = poly.draw_on_image(image,
                                   color=[32, 128, 32],
                                   color_face=[32, 128, 32],
                                   color_lines=[0, 255, 0],
                                   color_points=[0, 255, 0],
                                   alpha=1.0,
                                   alpha_face=1.0,
                                   alpha_lines=1.0,
                                   alpha_points=0.0,
                                   raise_if_out_of_image=True)
        except Exception as exc:
            assert "Cannot draw polygon" in str(exc)
            got_exception = True
        assert got_exception

    def test_only_lines_visible(self):
        # face+points invisible via alpha
        poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        image = self.image
        image_poly = poly.draw_on_image(image,
                                        color=[32, 128, 32],
                                        color_face=[32, 128, 32],
                                        color_lines=[0, 255, 0],
                                        color_points=[0, 255, 0],
                                        alpha=1.0,
                                        alpha_face=0.0,
                                        alpha_lines=1.0,
                                        alpha_points=0.0,
                                        raise_if_out_of_image=False)
        assert image_poly.dtype.type == np.uint8
        assert image_poly.shape == (10, 10, 3)
        # draw did not change original image (copy=True)
        assert np.sum(image) == 3 * np.sum(np.arange(100))
        for c_idx, value in enumerate([0, 255, 0]):
            # left boundary
            assert np.all(image_poly[2:9, 2:3, c_idx]
                          == np.zeros((7, 1), dtype=np.uint8) + value)
        assert np.all(image_poly[3:8, 3:8, :] == image[3:8, 3:8, :])

    def test_only_face_visible(self):
        # boundary+points invisible via alpha
        poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        image = self.image
        image_poly = poly.draw_on_image(image,
                                        color=[32, 128, 32],
                                        color_face=[32, 128, 32],
                                        color_lines=[0, 255, 0],
                                        color_points=[0, 255, 0],
                                        alpha=1.0,
                                        alpha_face=1.0,
                                        alpha_lines=0.0,
                                        alpha_points=0.0,
                                        raise_if_out_of_image=False)
        assert image_poly.dtype.type == np.uint8
        assert image_poly.shape == (10, 10, 3)
        # draw did not change original image (copy=True)
        assert np.sum(image) == 3 * np.sum(np.arange(100))
        expected = np.tile(
            np.uint8([32, 128, 32]).reshape((1, 1, 3)), (6, 6, 1)
        )
        assert np.all(image_poly[2:8, 2:8, :] == expected)

    def test_alpha_is_080(self):
        # alpha=0.8
        poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        image = self.image
        image_poly = poly.draw_on_image(image,
                                        color=[32, 128, 32],
                                        color_face=[32, 128, 32],
                                        color_lines=[0, 255, 0],
                                        color_points=[0, 255, 0],
                                        alpha=0.8,
                                        alpha_points=0.0,
                                        raise_if_out_of_image=False)
        assert image_poly.dtype.type == np.uint8
        assert image_poly.shape == (10, 10, 3)
        for c_idx, value in enumerate([0, 255, 0]):
            expected = np.round(
                (1-0.8)*image[2:9, 8:9, c_idx]
                + np.full((7, 1), 0.8*value, dtype=np.float32)
            ).astype(np.uint8)

            # right boundary
            assert np.all(image_poly[2:9, 8:9, c_idx] == expected)
        expected = (0.8 * 0.5) * np.tile(
            np.uint8([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1)
        ) + (1 - (0.8 * 0.5)) * image[3:8, 3:8, :]
        assert np.all(image_poly[3:8, 3:8, :]
                      == np.round(expected).astype(np.uint8))

    def test_face_and_lines_at_half_visibility(self):
        # alpha of fill and perimeter 0.5
        poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        image = self.image
        image_poly = poly.draw_on_image(image,
                                        color=[32, 128, 32],
                                        color_face=[32, 128, 32],
                                        color_lines=[0, 255, 0],
                                        color_points=[0, 255, 0],
                                        alpha=1.0,
                                        alpha_face=0.5,
                                        alpha_lines=0.5,
                                        alpha_points=0.0,
                                        raise_if_out_of_image=False)
        assert image_poly.dtype.type == np.uint8
        assert image_poly.shape == (10, 10, 3)
        for c_idx, value in enumerate([0, 255, 0]):
            expected = np.round(
                0.5*image[2:9, 8:9, c_idx]
                + np.full((7, 1), 0.5*value, dtype=np.float32)
            ).astype(np.uint8)

            # right boundary
            assert np.all(image_poly[2:9, 8:9, c_idx] == expected)

        expected = 0.5 * np.tile(
            np.uint8([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1)
        ) + 0.5 * image[3:8, 3:8, :]
        assert np.all(image_poly[3:8, 3:8, :]
                      == np.round(expected).astype(np.uint8))

        # copy=False
        # test deactivated as the function currently does not offer a copy
        # argument
        """
        image_cp = np.copy(image)
        poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
        image_poly = poly.draw_on_image(image_cp,
                                        color_face=[32, 128, 32],
                                        color_boundary=[0, 255, 0],
                                        alpha_face=1.0,
                                        alpha_boundary=1.0,
                                        raise_if_out_of_image=False)
        assert image_poly.dtype.type == np.uint8
        assert image_poly.shape == (10, 10, 3)
        assert np.all(image_cp == image_poly)
        assert not np.all(image_cp == image)
        for c_idx, value in enumerate([0, 255, 0]):
            # left boundary
            assert np.all(image_poly[2:9, 2:3, c_idx]
                          == np.zeros((6, 1, 3), dtype=np.uint8) + value)
            # left boundary
            assert np.all(image_cp[2:9, 2:3, c_idx]
                          == np.zeros((6, 1, 3), dtype=np.uint8) + value)
        expected = np.tile(
            np.uint8([32, 128, 32]).reshape((1, 1, 3)),
            (5, 5, 1)
        )
        assert np.all(image_poly[3:8, 3:8, :] == expected)
        assert np.all(image_cp[3:8, 3:8, :] == expected)
        """


class TestPolygon_extract_from_image(unittest.TestCase):
    @property
    def image(self):
        return np.arange(20*20*2).reshape((20, 20, 2)).astype(np.int32)

    def test_polygon_is_identical_with_image_shape(self):
        # inside image and completely covers it
        poly = ia.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        subimage = poly.extract_from_image(self.image)
        assert np.array_equal(subimage, self.image[0:10, 0:10, :])

    def test_polygon_is_subpart_of_image(self):
        # inside image, subpart of it (not all of the image has to be
        # extracted)
        poly = ia.Polygon([(1, 1), (9, 1), (9, 9), (1, 9)])
        subimage = poly.extract_from_image(self.image)
        assert np.array_equal(subimage, self.image[1:9, 1:9, :])

    def test_polygon_fully_inside_image__no_rectangular_shape(self):
        # inside image, two image areas that don't belong to the polygon but
        # have to be extracted
        poly = ia.Polygon([(0, 0), (10, 0), (10, 5), (20, 5),
                           (20, 20), (10, 20), (10, 5), (0, 5)])
        subimage = poly.extract_from_image(self.image)
        expected = np.copy(self.image)
        expected[:5, 10:, :] = 0  # top right block
        expected[5:, :10, :] = 0  # left bottom block
        assert np.array_equal(subimage, expected)

    def test_polygon_is_partially_outside_of_image(self):
        # partially out of image
        poly = ia.Polygon([(-5, 0), (5, 0), (5, 10), (-5, 10)])
        subimage = poly.extract_from_image(self.image)
        expected = np.zeros((10, 10, 2), dtype=np.int32)
        expected[0:10, 5:10, :] = self.image[0:10, 0:5, :]
        assert np.array_equal(subimage, expected)

    def test_polygon_is_fully_outside_of_image(self):
        # fully out of image
        poly = ia.Polygon([(30, 0), (40, 0), (40, 10), (30, 10)])
        subimage = poly.extract_from_image(self.image)
        expected = np.zeros((10, 10, 2), dtype=np.int32)
        assert np.array_equal(subimage, expected)

    def test_polygon_coords_after_rounding_are_identical_with_img_shape(self):
        # inside image, subpart of it
        # float coordinates, rounded so that the whole image will be extracted
        poly = ia.Polygon([(0.4, 0.4), (9.6, 0.4), (9.6, 9.6), (0.4, 9.6)])
        subimage = poly.extract_from_image(self.image)
        assert np.array_equal(subimage, self.image[0:10, 0:10, :])

    def test_polygon_coords_after_rounding_are_subpart_of_image(self):
        # inside image, subpart of it
        # float coordinates, rounded so that x/y 0<=i<9 will be extracted
        # (instead of 0<=i<10)
        poly = ia.Polygon([(0.5, 0.5), (9.4, 0.5), (9.4, 9.4), (0.5, 9.4)])
        subimage = poly.extract_from_image(self.image)
        assert np.array_equal(subimage, self.image[0:9, 0:9, :])

    def test_polygon_coords_after_rounding_are_subpart_of_image2(self):
        # inside image, subpart of it
        # float coordinates, rounded so that x/y 1<=i<9 will be extracted
        # (instead of 0<=i<10)
        poly = ia.Polygon([(0.51, 0.51), (9.4, 0.51), (9.4, 9.4), (0.51, 9.4)])
        subimage = poly.extract_from_image(self.image)
        assert np.array_equal(subimage, self.image[1:9, 1:9, :])

    def test_polygon_without_area_fails(self):
        # error for invalid polygons
        got_exception = False
        poly = ia.Polygon([(0.51, 0.51), (9.4, 0.51)])
        try:
            _ = poly.extract_from_image(self.image)
        except Exception as exc:
            assert "Polygon must be made up" in str(exc)
            got_exception = True
        assert got_exception


class TestPolygon_change_first_point_by_coords(unittest.TestCase):
    def test_change_to_first_point_in_exterior(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        poly_reordered = poly.change_first_point_by_coords(x=0, y=0)
        assert np.allclose(poly.exterior, poly_reordered.exterior)

    def test_change_to_first_second_in_exterior(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        poly_reordered = poly.change_first_point_by_coords(x=1, y=0)
        # make sure that it does not reorder inplace
        assert np.allclose(poly.exterior, np.float32([[0, 0], [1, 0], [1, 1]]))
        assert np.allclose(poly_reordered.exterior,
                           np.float32([[1, 0], [1, 1], [0, 0]]))

    def test_change_to_third_point_in_exterior(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        poly_reordered = poly.change_first_point_by_coords(x=1, y=1)
        assert np.allclose(poly_reordered.exterior,
                           np.float32([[1, 1], [0, 0], [1, 0]]))

    def test_coords_slightly_off_from_target_point_limited_max_distance(self):
        # inaccurate point, but close enough
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        poly_reordered = poly.change_first_point_by_coords(x=1.0, y=0.01,
                                                           max_distance=0.1)
        assert np.allclose(poly_reordered.exterior,
                           np.float32([[1, 0], [1, 1], [0, 0]]))

    def test_coords_slightly_off_from_target_point_infinite_max_distance(self):
        # inaccurate point, but close enough (infinite max distance)
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        poly_reordered = poly.change_first_point_by_coords(x=1.0, y=0.01,
                                                           max_distance=None)
        assert np.allclose(poly_reordered.exterior,
                           np.float32([[1, 0], [1, 1], [0, 0]]))

    def test_closest_point_to_coords_exceeds_max_distance(self):
        # point too far away
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        got_exception = False
        try:
            _ = poly.change_first_point_by_coords(x=1.0, y=0.01, max_distance=0.001)
        except Exception as exc:
            assert "Closest found point " in str(exc)
            got_exception = True
        assert got_exception

    def test_polygon_with_two_points(self):
        # reorder with two points
        poly = ia.Polygon([(0, 0), (1, 0)])
        poly_reordered = poly.change_first_point_by_coords(x=1, y=0)
        assert np.allclose(poly_reordered.exterior,
                           np.float32([[1, 0], [0, 0]]))

    def test_polygon_with_one_point(self):
        # reorder with one point
        poly = ia.Polygon([(0, 0)])
        poly_reordered = poly.change_first_point_by_coords(x=0, y=0)
        assert np.allclose(poly_reordered.exterior, np.float32([[0, 0]]))

    def test_polygon_with_zero_points_fails(self):
        # invalid polygon
        got_exception = False
        poly = ia.Polygon([])
        try:
            _ = poly.change_first_point_by_coords(x=0, y=0)
        except Exception as exc:
            assert "Cannot reorder polygon points" in str(exc)
            got_exception = True
        assert got_exception


class TestPolygon_change_first_point_by_index(unittest.TestCase):
    def test_change_to_point_with_index_0(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        poly_reordered = poly.change_first_point_by_index(0)
        assert np.allclose(poly.exterior, poly_reordered.exterior)

    def test_change_to_point_with_index_1(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        poly_reordered = poly.change_first_point_by_index(1)
        # make sure that it does not reorder inplace
        assert np.allclose(poly.exterior,
                           np.float32([[0, 0], [1, 0], [1, 1]]))
        assert np.allclose(poly_reordered.exterior,
                           np.float32([[1, 0], [1, 1], [0, 0]]))

    def test_change_to_point_with_index_2(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        poly_reordered = poly.change_first_point_by_index(2)
        assert np.allclose(poly_reordered.exterior,
                           np.float32([[1, 1], [0, 0], [1, 0]]))

    def test_polygon_with_two_points(self):
        # reorder with two points
        poly = ia.Polygon([(0, 0), (1, 0)])
        poly_reordered = poly.change_first_point_by_index(1)
        assert np.allclose(poly_reordered.exterior,
                           np.float32([[1, 0], [0, 0]]))

    def test_polygon_with_one_point(self):
        # reorder with one point
        poly = ia.Polygon([(0, 0)])
        poly_reordered = poly.change_first_point_by_index(0)
        assert np.allclose(poly_reordered.exterior, np.float32([[0, 0]]))

    def test_polygon_with_zero_points_fails(self):
        poly = ia.Polygon([])
        got_exception = False
        try:
            _ = poly.change_first_point_by_index(0)
        except AssertionError:
            got_exception = True
        assert got_exception

    def test_index_beyond_max_index_fails(self):
        # idx out of bounds
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        got_exception = False
        try:
            _ = poly.change_first_point_by_index(3)
        except AssertionError:
            got_exception = True
        assert got_exception

    def test_index_beyond_max_index_fails__single_point_polygon(self):
        poly = ia.Polygon([(0, 0)])
        got_exception = False
        try:
            _ = poly.change_first_point_by_index(1)
        except AssertionError:
            got_exception = True
        assert got_exception

    def test_index_below_zero_fails(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        got_exception = False
        try:
            _ = poly.change_first_point_by_index(-1)
        except AssertionError:
            got_exception = True
        assert got_exception


class TestPolygon_to_shapely_line_string(unittest.TestCase):
    def test_three_point_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        ls = poly.to_shapely_line_string()
        assert np.allclose(ls.coords, np.float32([[0, 0], [1, 0], [1, 1]]))

    def test_two_point_polygon(self):
        # two point polygon
        poly = ia.Polygon([(0, 0), (1, 0)])
        ls = poly.to_shapely_line_string()
        assert np.allclose(ls.coords, np.float32([[0, 0], [1, 0]]))

    def test_one_point_polygon_fails(self):
        # one point polygon
        poly = ia.Polygon([(0, 0)])
        got_exception = False
        try:
            _ = poly.to_shapely_line_string()
        except Exception as exc:
            assert (
                "Conversion to shapely line string requires at least two "
                "points" in str(exc))
            got_exception = True
        assert got_exception

    def test_zero_point_polygon_fails(self):
        # zero point polygon
        poly = ia.Polygon([])
        got_exception = False
        try:
            _ = poly.to_shapely_line_string()
        except Exception as exc:
            assert (
                "Conversion to shapely line string requires at least two "
                "points" in str(exc))
            got_exception = True
        assert got_exception

    def test_closed_is_true(self):
        # closed line string
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        ls = poly.to_shapely_line_string(closed=True)
        assert np.allclose(ls.coords,
                           np.float32([[0, 0], [1, 0], [1, 1], [0, 0]]))

    def test_interpolate_is_1(self):
        # interpolation
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        ls = poly.to_shapely_line_string(interpolate=1)
        assert np.allclose(
            ls.coords,
            np.float32([[0, 0], [0.5, 0], [1, 0], [1, 0.5],
                        [1, 1], [0.5, 0.5]]))

    def test_interpolate_is_2(self):
        # interpolation with 2 steps
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        ls = poly.to_shapely_line_string(interpolate=2)
        assert np.allclose(ls.coords, np.float32([
            [0, 0], [1/3, 0], [2/3, 0],
            [1, 0], [1, 1/3], [1, 2/3],
            [1, 1], [2/3, 2/3], [1/3, 1/3]
        ]))

    def test_closed_is_true_and_interpolate_is_1(self):
        # interpolation with closed=True
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        ls = poly.to_shapely_line_string(closed=True, interpolate=1)
        assert np.allclose(
            ls.coords,
            np.float32([[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1],
                        [0.5, 0.5], [0, 0]]))


class TestPolygon_to_shapely_polygon(unittest.TestCase):
    def test_square_polygon(self):
        exterior = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly = ia.Polygon(exterior)
        poly_shapely = poly.to_shapely_polygon()
        gen = zip(exterior, poly_shapely.exterior.coords)
        for (x_exp, y_exp), (x_obs, y_obs) in gen:
            assert np.isclose(x_obs, x_exp, rtol=0, atol=1e-8)
            assert np.isclose(y_obs, y_exp, rtol=0, atol=1e-8)


class TestPolygon_to_bounding_box(unittest.TestCase):
    def test_square_polygon(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        bb = poly.to_bounding_box()
        assert 0 - 1e-8 < bb.x1 < 0 + 1e-8
        assert 0 - 1e-8 < bb.y1 < 0 + 1e-8
        assert 1 - 1e-8 < bb.x2 < 1 + 1e-8
        assert 1 - 1e-8 < bb.y2 < 1 + 1e-8

    def test_triangular_polygon(self):
        poly = ia.Polygon([(0.5, 0), (1, 1), (0, 1)])
        bb = poly.to_bounding_box()
        assert 0 - 1e-8 < bb.x1 < 0 + 1e-8
        assert 0 - 1e-8 < bb.y1 < 0 + 1e-8
        assert 1 - 1e-8 < bb.x2 < 1 + 1e-8
        assert 1 - 1e-8 < bb.y2 < 1 + 1e-8

    def test_triangular_polygon2(self):
        poly = ia.Polygon([(0.5, 0.5), (2, 0.1), (1, 1)])
        bb = poly.to_bounding_box()
        assert 0.5 - 1e-8 < bb.x1 < 0.5 + 1e-8
        assert 0.1 - 1e-8 < bb.y1 < 0.1 + 1e-8
        assert 2.0 - 1e-8 < bb.x2 < 2.0 + 1e-8
        assert 1.0 - 1e-8 < bb.y2 < 1.0 + 1e-8


class TestPolygon_to_line_string(unittest.TestCase):
    def test_polygon_with_zero_points(self):
        poly = ia.Polygon([])
        ls = poly.to_line_string(closed=False)
        assert len(ls.coords) == 0
        assert ls.label is None

    def test_polygon_with_zero_points__closed_is_true(self):
        poly = ia.Polygon([])
        ls = poly.to_line_string(closed=True)
        assert len(ls.coords) == 0
        assert ls.label is None

    def test_polygon_with_zero_points__label_set(self):
        poly = ia.Polygon([], label="foo")
        ls = poly.to_line_string(closed=False)
        assert len(ls.coords) == 0
        assert ls.label == "foo"

    def test_polygon_with_one_point(self):
        poly = ia.Polygon([(0, 0)])
        ls = poly.to_line_string(closed=False)
        assert len(ls.coords) == 1
        assert ls.label is None

    def test_polygon_with_one_point__closed_is_true(self):
        poly = ia.Polygon([(0, 0)])
        ls = poly.to_line_string(closed=True)
        assert len(ls.coords) == 1
        assert ls.coords_almost_equals([(0, 0)])
        assert ls.label is None

    def test_polygon_with_two_points(self):
        poly = ia.Polygon([(0, 0), (1, 1)])
        ls = poly.to_line_string(closed=False)
        assert len(ls.coords) == 2
        assert ls.coords_almost_equals([(0, 0), (1, 1)])
        assert ls.label is None

    def test_polygon_with_two_point__closed_is_true(self):
        poly = ia.Polygon([(0, 0), (1, 1)])
        ls = poly.to_line_string(closed=True)
        assert len(ls.coords) == 3
        assert ls.coords_almost_equals([(0, 0), (1, 1), (0, 0)])
        assert ls.label is None

    def test_polygon_with_two_points__label_is_set(self):
        poly = ia.Polygon([(0, 0), (1, 1)], label="foo")
        ls = poly.to_line_string()
        assert len(ls.coords) == 3
        assert ls.coords_almost_equals([(0, 0), (1, 1), (0, 0)])
        assert ls.label == "foo"

    def test_polygon_with_two_point__closed_is_true_label_is_set(self):
        poly = ia.Polygon([(0, 0), (1, 1)], label="foo")
        ls = poly.to_line_string(closed=True)
        assert len(ls.coords) == 3
        assert ls.coords_almost_equals([(0, 0), (1, 1), (0, 0)])
        assert ls.label == "foo"


class TestPolygon_from_shapely(unittest.TestCase):
    def test_square_polygon(self):
        exterior = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly_shapely = shapely.geometry.Polygon(exterior)
        poly = ia.Polygon.from_shapely(poly_shapely)

        # shapely messes up the point ordering, so we try to correct it here
        start_idx = 0
        for i, (x, y) in enumerate(poly.exterior):
            dist = np.sqrt((exterior[0][0] - x) ** 2
                           + (exterior[0][1] - x) ** 2)
            if dist < 1e-4:
                start_idx = i
                break
        poly = poly.change_first_point_by_index(start_idx)

        for (x_exp, y_exp), (x_obs, y_obs) in zip(exterior, poly.exterior):
            assert x_exp - 1e-8 < x_obs < x_exp + 1e-8
            assert y_exp - 1e-8 < y_obs < y_exp + 1e-8

    def test_polygon_with_zero_points(self):
        poly_shapely = shapely.geometry.Polygon([])
        poly = ia.Polygon.from_shapely(poly_shapely)
        assert len(poly.exterior) == 0


class TestPolygon_copy(unittest.TestCase):
    def test_square_polygon_with_label(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
        poly_cp = poly.copy()
        assert poly.exterior.dtype.type == poly_cp.exterior.dtype.type
        assert poly.exterior.shape == poly_cp.exterior.shape
        assert np.allclose(poly.exterior, poly_cp.exterior)
        assert poly.label == poly_cp.label


class TestPolygon_deepcopy(unittest.TestCase):
    def test_square_polygon_with_label(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
        poly_cp = poly.deepcopy()
        assert poly.exterior.dtype.type == poly_cp.exterior.dtype.type
        assert poly.exterior.shape == poly_cp.exterior.shape
        assert np.allclose(poly.exterior, poly_cp.exterior)
        assert poly.label == poly_cp.label

    def test_copy_is_not_shallow(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
        poly_cp = poly.deepcopy()
        poly_cp.exterior[0, 0] = 100.0
        poly_cp.label = "test2"
        assert poly.exterior.dtype.type == poly_cp.exterior.dtype.type
        assert poly.exterior.shape == poly_cp.exterior.shape
        assert not np.allclose(poly.exterior, poly_cp.exterior)
        assert not poly.label == poly_cp.label


class TestPolygon___repr___and___str__(unittest.TestCase):
    def test_with_int_coordinates_provided_to_init(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
        expected = (
            "Polygon(["
            "(x=0.000, y=0.000), "
            "(x=1.000, y=0.000), "
            "(x=1.000, y=1.000), "
            "(x=0.000, y=1.000)"
            "] (4 points), label=test)"
        )
        assert poly.__repr__() == expected
        assert poly.__str__() == expected

    def test_with_float_coordinates_provided_to_init(self):
        poly = ia.Polygon([(0, 0.5), (1.5, 0), (1, 1), (0, 1)], label="test")
        expected = (
            "Polygon(["
            "(x=0.000, y=0.500), "
            "(x=1.500, y=0.000), "
            "(x=1.000, y=1.000), "
            "(x=0.000, y=1.000)"
            "] (4 points), label=test)"
        )
        assert poly.__repr__() == expected
        assert poly.__str__() == expected

    def test_label_is_none(self):
        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label=None)
        expected = (
            "Polygon(["
            "(x=0.000, y=0.000), "
            "(x=1.000, y=0.000), "
            "(x=1.000, y=1.000), "
            "(x=0.000, y=1.000)"
            "] (4 points), label=None)"
        )
        assert poly.__repr__() == expected
        assert poly.__str__() == expected

    def test_polygon_has_zero_points(self):
        poly = ia.Polygon([], label="test")
        expected = "Polygon([] (0 points), label=test)"
        assert poly.__repr__() == expected
        assert poly.__str__() == expected


class TestPolygon_coords_almost_equals(unittest.TestCase):
    @mock.patch("imgaug.augmentables.polys.Polygon.exterior_almost_equals")
    def test_calls_exterior_almost_equals(self, mock_eae):
        mock_eae.return_value = "foo"
        poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        result = poly_a.coords_almost_equals(poly_b)

        assert result == "foo"
        mock_eae.assert_called_once_with(poly_b, max_distance=1e-4,
                                         points_per_edge=8)

    @mock.patch("imgaug.augmentables.polys.Polygon.exterior_almost_equals")
    def test_calls_exterior_almost_equals__no_defaults(self, mock_eae):
        mock_eae.return_value = "foo"
        poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        result = poly_a.coords_almost_equals(poly_b, max_distance=1,
                                             points_per_edge=2)

        assert result == "foo"
        mock_eae.assert_called_once_with(poly_b, max_distance=1,
                                         points_per_edge=2)


class TestPolygon_exterior_almost_equals(unittest.TestCase):
    def test_exactly_same_exterior(self):
        # exactly same exterior
        poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_one_point_duplicated(self):
        # one point duplicated
        poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = ia.Polygon([(0, 0), (1, 0), (1, 1), (1, 1), (0, 1)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_several_points_added_without_changing_basic_shape(self):
        # several points added without changing geometry
        poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = ia.Polygon([(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1), (0, 0.5)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_different_order(self):
        # different order
        poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = ia.Polygon([(0, 1), (1, 1), (1, 0), (0, 0)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_tiny_shift_below_max_distance(self):
        # tiny shift below tolerance
        poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = ia.Polygon([(0+1e-6, 0), (1+1e-6, 0), (1+1e-6, 1), (0+1e-6, 1)])
        assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-3)

    def test_tiny_shift_above_max_distance(self):
        # tiny shift above tolerance
        poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = ia.Polygon([(0+1e-6, 0), (1+1e-6, 0), (1+1e-6, 1), (0+1e-6, 1)])
        assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-9)

    def test_polygons_with_half_intersection(self):
        # shifted polygon towards half overlap
        poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = ia.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_polygons_without_any_intersection(self):
        # shifted polygon towards no overlap at all
        poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly_b = ia.Polygon([(100, 0), (101, 0), (101, 1), (100, 1)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_both_polygons_with_zero_points(self):
        # both polygons without points
        poly_a = ia.Polygon([])
        poly_b = ia.Polygon([])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_both_polygons_with_one_point__both_identical(self):
        # both polygons with one point
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_both_polygons_with_zero_points__both_different(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(100, 100)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_both_polygons_with_zero_points__difference_below_max_dist(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0+1e-6, 0)])
        assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    def test_both_polygons_with_zero_points_difference_above_max_dist(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0+1, 0)])
        assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    def test_both_polygons_with_two_points__identical(self):
        # both polygons with two points
        poly_a = ia.Polygon([(0, 0), (1, 0)])
        poly_b = ia.Polygon([(0, 0), (1, 0)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_both_polygons_with_two_points__identical_and_zero_area(self):
        poly_a = ia.Polygon([(0, 0), (0, 0)])
        poly_b = ia.Polygon([(0, 0), (0, 0)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_both_polygons_with_two_points__second_point_different(self):
        poly_a = ia.Polygon([(0, 0), (1, 0)])
        poly_b = ia.Polygon([(0, 0), (2, 0)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_both_polygons_with_two_points__difference_below_max_dist(self):
        poly_a = ia.Polygon([(0, 0), (1, 0)])
        poly_b = ia.Polygon([(0+1e-6, 0), (1+1e-6, 0)])
        assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    def test_both_polygons_with_three_points__identical(self):
        # both polygons with three points
        poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
        poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_both_polygons_with_three_points__one_point_differs(self):
        poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
        poly_b = ia.Polygon([(0, 0), (1, -1), (0.5, 1)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_both_polygons_with_three_points__difference_below_max_dist(self):
        poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
        poly_b = ia.Polygon([(0, 0), (1+1e-6, 0), (0.5, 1)])
        assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    def test_one_polygon_zero_points_other_one_point(self):
        # one polygon with zero points, other with one
        poly_a = ia.Polygon([])
        poly_b = ia.Polygon([(0, 0)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_one_point_other_zero_points(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_one_point_other_two_points(self):
        # one polygon with one point, other with two
        poly_a = ia.Polygon([(-10, -20)])
        poly_b = ia.Polygon([(0, 0), (1, 0)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_one_point_other_two_points_2(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (1, 0)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_two_points_other_one_point(self):
        poly_a = ia.Polygon([(0, 0), (1, 0)])
        poly_b = ia.Polygon([(0, 0)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_two_points_other_one_point__all_identical(self):
        poly_a = ia.Polygon([(0, 0), (0, 0)])
        poly_b = ia.Polygon([(0, 0)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_one_point_other_two_points__all_identical(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (0, 0)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_two_points_other_one_point__diff_below_max_dist(self):
        poly_a = ia.Polygon([(0, 0), (0+1e-6, 0)])
        poly_b = ia.Polygon([(0, 0)])
        assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    def test_one_polygon_two_points_other_one_point__diff_above_max_dist(self):
        poly_a = ia.Polygon([(0, 0), (0+1e-4, 0)])
        poly_b = ia.Polygon([(0, 0)])
        assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-9)

    def test_one_polygon_one_point_other_three_points(self):
        # one polygon with one point, other with three
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_three_points_other_one_point(self):
        poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
        poly_b = ia.Polygon([(0, 0)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_one_point_other_three_points__all_identical(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (0, 0), (0, 0)])
        assert poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_one_point_other_three_points__one_point_differs(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (0, 0), (1, 0)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_one_point_other_three_points__one_point_differs2(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (1, 0), (0, 0)])
        assert not poly_a.exterior_almost_equals(poly_b)

    def test_one_polygon_one_point_other_three_points__dist_below_max(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (0+1e-6, 0), (0, 0+1e-6)])
        assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    def test_one_polygon_one_point_other_three_points__dist_above_max(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (0+1e-4, 0), (0, 0+1e-4)])
        assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-9)


class TestPolygon_almost_equals(unittest.TestCase):
    def test_both_polygons_empty(self):
        poly_a = ia.Polygon([])
        poly_b = ia.Polygon([])
        assert poly_a.almost_equals(poly_b)

    def test_both_polygons_one_point(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0)])
        assert poly_a.almost_equals(poly_b)

    def test_one_polygon_one_point_other_two_points__all_identical(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (0, 0)])
        assert poly_a.almost_equals(poly_b)

    def test_one_polygon_one_point_other_three_points__all_identical(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (0, 0), (0, 0)])
        assert poly_a.almost_equals(poly_b)

    def test_one_polygon_one_point_other_two_points__diff_below_max_dist(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (0+1e-10, 0)])
        assert poly_a.almost_equals(poly_b)

    def test_both_polygons_one_point__first_with_label(self):
        poly_a = ia.Polygon([(0, 0)], label="test")
        poly_b = ia.Polygon([(0, 0)])
        assert not poly_a.almost_equals(poly_b)

    def test_both_polygons_one_point__second_with_label(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0)], label="test")
        assert not poly_a.almost_equals(poly_b)

    def test_both_polygons_one_point__both_with_label(self):
        poly_a = ia.Polygon([(0, 0)], label="test")
        poly_b = ia.Polygon([(0, 0)], label="test")
        assert poly_a.almost_equals(poly_b)

    def test_both_polygons_one_point__both_with_label_but_point_differs(self):
        poly_a = ia.Polygon([(0, 0)], label="test")
        poly_b = ia.Polygon([(1, 0)], label="test")
        assert not poly_a.almost_equals(poly_b)

    def test_both_polygons_one_point__same_point_but_labels_differ(self):
        poly_a = ia.Polygon([(0, 0)], label="testA")
        poly_b = ia.Polygon([(0, 0)], label="testB")
        assert not poly_a.almost_equals(poly_b)

    def test_both_polygons_three_points(self):
        poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
        poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
        assert poly_a.almost_equals(poly_b)

    def test_one_polygon_one_point_other_three_points(self):
        poly_a = ia.Polygon([(0, 0)])
        poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
        assert not poly_a.almost_equals(poly_b)


# TODO add test for _convert_points_to_shapely_line_string


class TestPolygonsOnImage___init__(unittest.TestCase):
    def test_with_one_polygon(self):
        # standard case with one polygon
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            shape=(10, 10, 3)
        )
        assert len(poly_oi.polygons) == 1
        assert np.allclose(
            poly_oi.polygons[0].exterior,
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0, atol=1e-4)
        assert poly_oi.shape == (10, 10, 3)

    def test_with_multiple_polygons(self):
        # standard case with multiple polygons
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             ia.Polygon([(0, 0), (1, 0), (1, 1)]),
             ia.Polygon([(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)])],
            shape=(10, 10, 3)
        )
        assert len(poly_oi.polygons) == 3
        assert np.allclose(
            poly_oi.polygons[0].exterior,
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0, atol=1e-4)
        assert np.allclose(
            poly_oi.polygons[1].exterior,
            [(0, 0), (1, 0), (1, 1)],
            rtol=0, atol=1e-4)
        assert np.allclose(
            poly_oi.polygons[2].exterior,
            [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)],
            rtol=0, atol=1e-4)
        assert poly_oi.shape == (10, 10, 3)

    def test_with_zero_polygons(self):
        # list of polygons is empty
        poly_oi = ia.PolygonsOnImage(
            [],
            shape=(10, 10, 3)
        )
        assert len(poly_oi.polygons) == 0

    def test_with_invalid_polygon(self):
        # invalid polygon
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (0.5, 0), (0.5, 1.5), (0, 1), (1, 1), (0, 1)])],
            shape=(10, 10, 3)
        )
        assert len(poly_oi.polygons) == 1
        assert np.allclose(
            poly_oi.polygons[0].exterior,
            [(0, 0), (0.5, 0), (0.5, 1.5), (0, 1), (1, 1), (0, 1)],
            rtol=0, atol=1e-4)

    def test_with_zero_polygons_and_shape_given_as_array(self):
        # shape given as numpy array
        poly_oi = ia.PolygonsOnImage(
            [],
            shape=np.zeros((10, 10, 3), dtype=np.uint8)
        )
        assert poly_oi.shape == (10, 10, 3)

    def test_with_zero_polygons_and_shape_given_as_2d_tuple(self):
        # 2D shape
        poly_oi = ia.PolygonsOnImage(
            [],
            shape=(10, 11)
        )
        assert poly_oi.shape == (10, 11)


class TestPolygonsOnImage_items(unittest.TestCase):
    def test_with_two_polygons(self):
        poly1 = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        psoi = ia.PolygonsOnImage(
            [poly1, poly2],
            shape=(10, 10, 3)
        )

        items = psoi.items

        assert items == [poly1, poly2]

    def test_items_empty(self):
        psoi = ia.PolygonsOnImage([], shape=(40, 50, 3))

        items = psoi.items

        assert items == []


class TestPolygonsOnImage_empty(unittest.TestCase):
    def test_with_multiple_polygons(self):
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             ia.Polygon([(0, 0), (1, 0), (1, 1)]),
             ia.Polygon([(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)])],
            shape=(10, 10, 3)
        )
        assert poly_oi.empty is False

    def test_with_zero_polygons(self):
        # list of polygons is empty
        poly_oi = ia.PolygonsOnImage([], shape=(10, 10, 3))
        assert poly_oi.empty is True


class TestPolygonsOnImage_on(unittest.TestCase):
    def test_new_shape_is_identical_to_old_shape(self):
        # size unchanged
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             ia.Polygon([(0, 0), (1, 0), (1, 1)]),
             ia.Polygon([(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)])],
            shape=(1, 1, 3)
        )
        poly_oi_proj = poly_oi.on((1, 1, 3))
        assert np.allclose(
            poly_oi_proj.polygons[0].exterior,
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0, atol=1e-4)
        assert np.allclose(
            poly_oi_proj.polygons[1].exterior,
            [(0, 0), (1, 0), (1, 1)],
            rtol=0, atol=1e-4)
        assert np.allclose(
            poly_oi_proj.polygons[2].exterior,
            [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)],
            rtol=0, atol=1e-4)
        assert poly_oi_proj.shape == (1, 1, 3)

    def test_new_shape_is_10x_smaller_than_old_shape(self):
        # 10x decrease in size
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
             ia.Polygon([(0, 0), (10, 0), (10, 10)]),
             ia.Polygon([(5, 0), (10, 5), (5, 10), (0, 5)])],
            shape=(10, 10, 3)
        )
        poly_oi_proj = poly_oi.on((1, 1, 3))
        assert np.allclose(
            poly_oi_proj.polygons[0].exterior,
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0, atol=1e-4)
        assert np.allclose(
            poly_oi_proj.polygons[1].exterior,
            [(0, 0), (1, 0), (1, 1)],
            rtol=0, atol=1e-4)
        assert np.allclose(
            poly_oi_proj.polygons[2].exterior,
            [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)],
            rtol=0, atol=1e-4)
        assert poly_oi_proj.shape == (1, 1, 3)

    def test_new_shape_is_2x_width_and_10x_height_of_old_shape(self):
        # 2x increase in width, 10x decrease in height
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])],
            shape=(100, 100, 3)
        )
        poly_oi_proj = poly_oi.on((10, 200, 3))
        assert np.allclose(
            poly_oi_proj.polygons[0].exterior,
            [(0, 0), (100, 0), (100, 10), (0, 10)],
            rtol=0, atol=1e-4)
        assert poly_oi_proj.shape == (10, 200, 3)


class TestPolygonsOnImage_draw_on_image(unittest.TestCase):
    def test_with_zero_polygons(self):
        # no polygons, nothing changed
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        poly_oi = ia.PolygonsOnImage([], shape=image.shape)
        image_drawn = poly_oi.draw_on_image(image)
        assert np.sum(image) == 0
        assert np.sum(image_drawn) == 0

    def test_with_two_polygons(self):
        # draw two polygons
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (9, 1), (9, 9), (1, 9)]),
             ia.Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])],
            shape=image.shape)
        image_expected = np.copy(image)
        image_expected = poly_oi.polygons[0].draw_on_image(image_expected)
        image_expected = poly_oi.polygons[1].draw_on_image(image_expected)
        image_drawn = poly_oi.draw_on_image(image)

        assert np.sum(image) == 0
        assert np.sum(image_drawn) > 0
        assert np.sum(image_expected) > 0
        assert np.allclose(image_drawn, image_expected)


class TestPolygonsOnImage_remove_out_of_image(unittest.TestCase):
    def test_with_zero_polygons(self):
        # no polygons, nothing to remove
        poly_oi = ia.PolygonsOnImage([], shape=(10, 11, 3))
        for fully, partly in [(False, False), (False, True),
                              (True, False), (True, True)]:
            poly_oi_rm = poly_oi.remove_out_of_image(fully=fully, partly=partly)
            assert len(poly_oi_rm.polygons) == 0
            assert poly_oi_rm.shape == (10, 11, 3)

    def test_one_polygon_fully_inside_image(self):
        # one polygon, fully inside the image
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (9, 1), (9, 9), (1, 9)])],
            shape=(10, 11, 3))
        for fully, partly in [(False, False), (False, True),
                              (True, False), (True, True)]:
            poly_oi_rm = poly_oi.remove_out_of_image(fully=fully, partly=partly)
            assert len(poly_oi_rm.polygons) == 1
            assert np.allclose(poly_oi_rm.polygons[0].exterior,
                               [(1, 1), (9, 1), (9, 9), (1, 9)],
                               rtol=0, atol=1e-4)
            assert poly_oi_rm.shape == (10, 11, 3)

    def test_one_poly_partially_ooi_one_fully_ooi(self):
        # two polygons, one partly outside, one fully outside
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (11, 1), (11, 9), (1, 9)]),
             ia.Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])],
            shape=(10, 10, 3))

        poly_oi_rm = poly_oi.remove_out_of_image(fully=False, partly=False)
        assert len(poly_oi.polygons) == 2
        assert len(poly_oi_rm.polygons) == 2
        assert np.allclose(poly_oi_rm.polygons[0].exterior,
                           [(1, 1), (11, 1), (11, 9), (1, 9)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_rm.polygons[1].exterior,
                           [(100, 100), (200, 100), (200, 200), (100, 200)],
                           rtol=0, atol=1e-4)
        assert poly_oi_rm.shape == (10, 10, 3)

        poly_oi_rm = poly_oi.remove_out_of_image(fully=True, partly=False)
        assert len(poly_oi.polygons) == 2
        assert len(poly_oi_rm.polygons) == 1
        assert np.allclose(poly_oi_rm.polygons[0].exterior,
                           [(1, 1), (11, 1), (11, 9), (1, 9)],
                           rtol=0, atol=1e-4)
        assert poly_oi_rm.shape == (10, 10, 3)

        poly_oi_rm = poly_oi.remove_out_of_image(fully=False, partly=True)
        assert len(poly_oi.polygons) == 2
        assert len(poly_oi_rm.polygons) == 1
        assert np.allclose(poly_oi_rm.polygons[0].exterior,
                           [(100, 100), (200, 100), (200, 200), (100, 200)],
                           rtol=0, atol=1e-4)
        assert poly_oi_rm.shape == (10, 10, 3)

        poly_oi_rm = poly_oi.remove_out_of_image(fully=True, partly=True)
        assert len(poly_oi.polygons) == 2
        assert len(poly_oi_rm.polygons) == 0
        assert poly_oi_rm.shape == (10, 10, 3)


class TestPolygonsOnImage_clip_out_of_image(unittest.TestCase):
    # NOTE: clip_out_of_image() can change the order of points,
    # hence we check here for each expected point whether it appears
    # somewhere in the list of points

    @classmethod
    def _any_point_close(cls, points, point_search):
        found = False
        for point in points:
            if np.allclose(point, point_search, atol=1e-4, rtol=0):
                found = True
        return found

    def test_with_zero_polygons(self):
        # no polygons
        poly_oi = ia.PolygonsOnImage([], shape=(10, 11, 3))
        poly_oi_clip = poly_oi.clip_out_of_image()
        assert len(poly_oi_clip.polygons) == 0
        assert poly_oi_clip.shape == (10, 11, 3)

    def test_with_one_polygon_fully_inside(self):
        # one polygon, fully inside
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (8, 1), (8, 9), (1, 9)])],
            shape=(10, 11, 3))
        poly_oi_clip = poly_oi.clip_out_of_image()
        assert len(poly_oi_clip.polygons) == 1
        for point_search in [(1, 1), (8, 1), (8, 9), (1, 9)]:
            assert self._any_point_close(poly_oi_clip.polygons[0].exterior,
                                         point_search)
        assert poly_oi_clip.shape == (10, 11, 3)

    def test_with_one_polygon_partially_ooi(self):
        # one polygon, partially outside
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (15, 1), (15, 9), (1, 9)])],
            shape=(10, 11, 3))
        poly_oi_clip = poly_oi.clip_out_of_image()
        assert len(poly_oi_clip.polygons) == 1
        for point_search in [(1, 1), (11, 1), (11, 9), (1, 9)]:
            assert self._any_point_close(poly_oi_clip.polygons[0].exterior,
                                         point_search)
        assert poly_oi_clip.shape == (10, 11, 3)

    def test_with_one_polygon_fully_ooi(self):
        # one polygon, fully outside
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])],
            shape=(10, 11, 3))
        poly_oi_clip = poly_oi.clip_out_of_image()
        assert len(poly_oi_clip.polygons) == 0
        assert poly_oi_clip.shape == (10, 11, 3)

    def test_with_three_pols_one_not_ooi_one_partially_ooi_one_fully_ooi(self):
        # three polygons, one fully inside, one partially outside,
        # one fully outside
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (8, 1), (8, 9), (1, 9)]),
             ia.Polygon([(1, 1), (15, 1), (15, 9), (1, 9)]),
             ia.Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])],
            shape=(10, 11, 3))
        poly_oi_clip = poly_oi.clip_out_of_image()
        assert len(poly_oi_clip.polygons) == 2
        for point_search in [(1, 1), (8, 1), (8, 9), (1, 9)]:
            assert self._any_point_close(poly_oi_clip.polygons[0].exterior,
                                         point_search)
        for point_search in [(1, 1), (11, 1), (11, 9), (1, 9)]:
            assert self._any_point_close(poly_oi_clip.polygons[1].exterior,
                                         point_search)
        assert poly_oi_clip.shape == (10, 11, 3)


class TestPolygonsOnImage_shift(unittest.TestCase):
    def test_with_zero_polygons(self):
        # no polygons
        poly_oi = ia.PolygonsOnImage([], shape=(10, 11, 3))
        poly_oi_shifted = poly_oi.shift(top=3, right=0, bottom=1, left=-3)
        assert len(poly_oi_shifted.polygons) == 0
        assert poly_oi_shifted.shape == (10, 11, 3)

    def test_with_three_polygons(self):
        # three polygons
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (8, 1), (8, 9), (1, 9)]),
             ia.Polygon([(1, 1), (15, 1), (15, 9), (1, 9)]),
             ia.Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])],
            shape=(10, 11, 3))
        poly_oi_shifted = poly_oi.shift(top=3, right=0, bottom=1, left=-3)
        assert len(poly_oi_shifted.polygons) == 3
        assert np.allclose(poly_oi_shifted.polygons[0].exterior,
                           [(1-3, 1+2), (8-3, 1+2), (8-3, 9+2), (1-3, 9+2)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_shifted.polygons[1].exterior,
                           [(1-3, 1+2), (15-3, 1+2), (15-3, 9+2), (1-3, 9+2)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_shifted.polygons[2].exterior,
                           [(100-3, 100+2), (200-3, 100+2),
                            (200-3, 200+2), (100-3, 200+2)],
                           rtol=0, atol=1e-4)
        assert poly_oi_shifted.shape == (10, 11, 3)


class TestPolygonsOnImage_to_xy_array(unittest.TestCase):
    def test_filled_object(self):
        psoi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (1, 0), (1, 1)]),
             ia.Polygon([(10, 10), (20, 0), (20, 20)])],
            shape=(2, 2, 3))

        xy_out = psoi.to_xy_array()

        expected = np.float32([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [20.0, 0.0],
            [20.0, 20.0]
        ])
        assert xy_out.shape == (6, 2)
        assert np.allclose(xy_out, expected)
        assert xy_out.dtype.name == "float32"

    def test_empty_object(self):
        psoi = ia.PolygonsOnImage(
            [],
            shape=(1, 2, 3))

        xy_out = psoi.to_xy_array()

        assert xy_out.shape == (0, 2)
        assert xy_out.dtype.name == "float32"


class TestPolygonsOnImage_fill_from_xy_array_(unittest.TestCase):
    def test_empty_array(self):
        xy = np.zeros((0, 2), dtype=np.float32)
        psoi = ia.PolygonsOnImage([], shape=(2, 2, 3))

        psoi = psoi.fill_from_xy_array_(xy)

        assert len(psoi.polygons) == 0

    def test_empty_list(self):
        xy = []
        psoi = ia.PolygonsOnImage([], shape=(2, 2, 3))

        psoi = psoi.fill_from_xy_array_(xy)

        assert len(psoi.polygons) == 0

    def test_array_with_two_coords(self):
        xy = np.array(
            [(100, 100),
             (101, 100),
             (101, 101),
             (110, 110),
             (120, 100),
             (120, 120)], dtype=np.float32)
        psoi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (1, 0), (1, 1)]),
             ia.Polygon([(10, 10), (20, 0), (20, 20)])],
            shape=(2, 2, 3))

        psoi = psoi.fill_from_xy_array_(xy)

        assert len(psoi.polygons) == 2
        assert np.allclose(
            psoi.polygons[0].coords,
            [(100, 100), (101, 100), (101, 101)])
        assert np.allclose(
            psoi.polygons[1].coords,
            [(110, 110), (120, 100), (120, 120)])

    def test_list_with_two_coords(self):
        xy = [(100, 100),
              (101, 100),
              (101, 101),
              (110, 110),
              (120, 100),
              (120, 120)]
        psoi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (1, 0), (1, 1)]),
             ia.Polygon([(10, 10), (20, 0), (20, 20)])],
            shape=(2, 2, 3))

        psoi = psoi.fill_from_xy_array_(xy)

        assert len(psoi.polygons) == 2
        assert np.allclose(
            psoi.polygons[0].coords,
            [(100, 100), (101, 100), (101, 101)])
        assert np.allclose(
            psoi.polygons[1].coords,
            [(110, 110), (120, 100), (120, 120)])


class TestPolygonsOnImage_to_keypoints_on_image(unittest.TestCase):
    def test_filled_instance(self):
        psoi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (1, 0), (1, 1)]),
             ia.Polygon([(10, 10), (20, 0), (20, 20)])],
            shape=(1, 2, 3))

        kpsoi = psoi.to_keypoints_on_image()

        assert len(kpsoi.keypoints) == 2*3
        assert kpsoi.keypoints[0].x == 0
        assert kpsoi.keypoints[0].y == 0
        assert kpsoi.keypoints[1].x == 1
        assert kpsoi.keypoints[1].y == 0
        assert kpsoi.keypoints[2].x == 1
        assert kpsoi.keypoints[2].y == 1
        assert kpsoi.keypoints[3].x == 10
        assert kpsoi.keypoints[3].y == 10
        assert kpsoi.keypoints[4].x == 20
        assert kpsoi.keypoints[4].y == 0
        assert kpsoi.keypoints[5].x == 20
        assert kpsoi.keypoints[5].y == 20

    def test_empty_instance(self):
        psoi = ia.PolygonsOnImage([], shape=(1, 2, 3))

        kpsoi = psoi.to_keypoints_on_image()

        assert len(kpsoi.keypoints) == 0


class TestPolygonsOnImage_invert_to_keypoints_on_image(unittest.TestCase):
    def test_filled_instance(self):
        psoi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (1, 0), (1, 1)]),
             ia.Polygon([(10, 10), (20, 0), (20, 20)])],
            shape=(1, 2, 3))
        kpsoi = ia.KeypointsOnImage(
            [ia.Keypoint(100, 100), ia.Keypoint(101, 100),
             ia.Keypoint(101, 101),
             ia.Keypoint(110, 110), ia.Keypoint(120, 100),
             ia.Keypoint(120, 120)],
            shape=(10, 20, 30))

        psoi_inv = psoi.invert_to_keypoints_on_image_(kpsoi)

        assert len(psoi_inv.polygons) == 2
        assert psoi_inv.shape == (10, 20, 30)
        assert np.allclose(
            psoi.polygons[0].coords,
            [(100, 100), (101, 100), (101, 101)])
        assert np.allclose(
            psoi.polygons[1].coords,
            [(110, 110), (120, 100), (120, 120)])

    def test_empty_instance(self):
        psoi = ia.PolygonsOnImage([], shape=(1, 2, 3))
        kpsoi = ia.KeypointsOnImage([], shape=(10, 20, 30))

        psoi_inv = psoi.invert_to_keypoints_on_image_(kpsoi)

        assert len(psoi_inv.polygons) == 0
        assert psoi_inv.shape == (10, 20, 30)


class TestPolygonsOnImage_copy(unittest.TestCase):
    def test_with_two_polygons(self):
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (8, 1), (8, 9), (1, 9)]),
             ia.Polygon([(2, 2), (16, 2), (16, 10), (2, 10)])],
            shape=(10, 11, 3))
        poly_oi_copy = poly_oi.copy()
        assert len(poly_oi_copy.polygons) == 2
        assert np.allclose(poly_oi_copy.polygons[0].exterior,
                           [(1, 1), (8, 1), (8, 9), (1, 9)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_copy.polygons[1].exterior,
                           [(2, 2), (16, 2), (16, 10), (2, 10)],
                           rtol=0, atol=1e-4)

        poly_oi_copy.shape = (20, 30, 3)
        assert poly_oi.shape == (10, 11, 3)
        assert poly_oi_copy.shape == (20, 30, 3)

        # make sure that changing the polygons only affects the copy
        poly_oi_copy.polygons = [ia.Polygon([(0, 0), (1, 0), (1, 1)])]
        assert np.allclose(poly_oi.polygons[0].exterior,
                           [(1, 1), (8, 1), (8, 9), (1, 9)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_copy.polygons[0].exterior,
                           [(0, 0), (1, 0), (1, 1)],
                           rtol=0, atol=1e-4)


class TestPolygonsOnImage_deepcopy(unittest.TestCase):
    def test_with_two_polygons(self):
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (8, 1), (8, 9), (1, 9)]),
             ia.Polygon([(2, 2), (16, 2), (16, 10), (2, 10)])],
            shape=(10, 11, 3))
        poly_oi_copy = poly_oi.deepcopy()
        assert len(poly_oi_copy.polygons) == 2
        assert np.allclose(poly_oi_copy.polygons[0].exterior,
                           [(1, 1), (8, 1), (8, 9), (1, 9)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_copy.polygons[1].exterior,
                           [(2, 2), (16, 2), (16, 10), (2, 10)],
                           rtol=0, atol=1e-4)

        poly_oi_copy.shape = (20, 30, 3)
        assert poly_oi.shape == (10, 11, 3)
        assert poly_oi_copy.shape == (20, 30, 3)

        # make sure that changing the polygons only affects the copy
        poly_oi_copy.polygons[0] = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        assert np.allclose(poly_oi.polygons[0].exterior,
                           [(1, 1), (8, 1), (8, 9), (1, 9)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_copy.polygons[0].exterior,
                           [(0, 0), (1, 0), (1, 1)],
                           rtol=0, atol=1e-4)

        # make sure that the arrays were also copied
        poly_oi_copy.polygons[1].exterior[0][0] = 100
        assert np.allclose(poly_oi.polygons[1].exterior,
                           [(2, 2), (16, 2), (16, 10), (2, 10)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_copy.polygons[1].exterior,
                           [(100, 2), (16, 2), (16, 10), (2, 10)],
                           rtol=0, atol=1e-4)


class TestPolygonsOnImage___repr___and___str__(unittest.TestCase):
    def test_with_zero_polygons(self):
        poly_oi = ia.PolygonsOnImage([], shape=(10, 11, 3))
        expected = "PolygonsOnImage([], shape=(10, 11, 3))"
        assert poly_oi.__repr__() == expected
        assert poly_oi.__str__() == expected

    def test_with_two_polygons(self):
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (8, 1), (8, 9), (1, 9)]),
             ia.Polygon([(2, 2), (16, 2), (16, 10), (2, 10)])],
            shape=(10, 11, 3))
        expected = (
            "PolygonsOnImage(["
            "Polygon([(x=1.000, y=1.000), (x=8.000, y=1.000), "
            "(x=8.000, y=9.000), (x=1.000, y=9.000)] "
            "(4 points), label=None), "
            "Polygon([(x=2.000, y=2.000), (x=16.000, y=2.000), "
            "(x=16.000, y=10.000), (x=2.000, y=10.000)] "
            "(4 points), label=None)"
            "], shape=(10, 11, 3))"
        )
        assert poly_oi.__repr__() == expected
        assert poly_oi.__str__() == expected


class Test_ConcavePolygonRecoverer(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _assert_points_are_identical(cls, observed, expected, atol=1e-8,
                                     rtol=0):
        assert len(observed) == len(expected)
        for i, (ps_obs, ps_exp) in enumerate(zip(observed, expected)):
            assert len(ps_obs) == len(ps_exp), "Failed at point %d" % (i,)
            for p_obs, p_exp in zip(ps_obs, ps_exp):
                assert len(p_obs) == 2
                assert len(p_exp) == 2
                assert np.allclose(p_obs, p_exp, atol=atol, rtol=rtol), (
                    "Unexpected coords at %d" % (i,))

    # TODO split into multiple tests
    def test_recover_from_fails_for_less_than_three_points(self):
        old_polygon = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        cpr = _ConcavePolygonRecoverer()
        with self.assertRaises(AssertionError):
            _poly = cpr.recover_from([], old_polygon)

        with self.assertRaises(AssertionError):
            _poly = cpr.recover_from([(0, 0)], old_polygon)

        with self.assertRaises(AssertionError):
            _poly = cpr.recover_from([(0, 0), (1, 0)], old_polygon)

        _poly = cpr.recover_from([(0, 0), (1, 0), (1, 1)], old_polygon)

    def test_recover_from_concave_polygons(self):
        cpr = _ConcavePolygonRecoverer()

        polys = [
            [(0, 0), (1, 0), (1, 1)],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1.0), (0, 1)],
        ]

        for poly in polys:
            with self.subTest(poly=str(poly)):
                old_polygon = ia.Polygon(poly)
                poly_concave = cpr.recover_from(poly, old_polygon)
                assert poly_concave.is_valid
                found = [False] * len(poly)
                for i, point in enumerate(poly):
                    for point_ext in poly_concave.exterior:
                        dist = np.sqrt((point[0] - point_ext[0])**2
                                       + (point[1] - point_ext[1])**2)
                        if dist < 0.01:
                            found[i] = True
                assert np.all(found)

    def test_recover_from_line(self):
        cpr = _ConcavePolygonRecoverer()

        poly = [(0, 0), (1, 0), (2, 0)]
        old_polygon = ia.Polygon(poly)
        poly_concave = cpr.recover_from(poly, old_polygon)
        assert poly_concave.is_valid
        found = [False] * len(poly)
        for i, point in enumerate(poly):
            for point_ext in poly_concave.exterior:
                dist = np.sqrt((point[0] - point_ext[0])**2
                               + (point[1] - point_ext[1])**2)
                if dist < 0.025:
                    found[i] = True
        assert np.all(found)

    def test_recover_from_polygon_with_duplicate_points(self):
        cpr = _ConcavePolygonRecoverer()

        poly = [(0, 0), (1, 0), (1, 0), (1, 1)]
        old_polygon = ia.Polygon(poly)
        poly_concave = cpr.recover_from(poly, old_polygon)
        assert poly_concave.is_valid
        found = [False] * len(poly)
        for i, point in enumerate(poly):
            for point_ext in poly_concave.exterior:
                dist = np.sqrt((point[0] - point_ext[0])**2
                               + (point[1] - point_ext[1])**2)
                if dist < 0.01:
                    found[i] = True
        assert np.all(found)

    def test_recover_from_invalid_polygon(self):
        cpr = _ConcavePolygonRecoverer()
        poly = [(0, 0), (0.5, 0), (0.5, 1.2), (1, 0), (1, 1), (0, 1)]
        old_polygon = ia.Polygon(poly)
        poly_concave = cpr.recover_from(poly, old_polygon)
        assert poly_concave.is_valid
        found = [False] * len(poly)
        for i, point in enumerate(poly):
            for point_ext in poly_concave.exterior:
                dist = np.sqrt(
                    (point[0] - point_ext[0])**2
                    + (point[1] - point_ext[1])**2
                )
                if dist < 0.025:
                    found[i] = True
        assert np.all(found)

    def test_recover_from_random_polygons(self):
        cpr = _ConcavePolygonRecoverer()
        nb_iterations = 10
        height, width = 10, 20
        nb_points_matrix = np.random.randint(3, 30, size=(nb_iterations,))
        for nb_points in nb_points_matrix:
            points = np.random.random(size=(nb_points, 2))
            points[:, 0] *= width
            points[:, 1] *= height
            # currently mainly used to copy the label, so not a significant
            # issue that it is not concave
            old_polygon = ia.Polygon(points)
            poly_concave = cpr.recover_from(points, old_polygon)
            assert poly_concave.is_valid
            # test if all points are in BB around returned polygon
            # would be better to directly call a polygon.contains(point) method
            # but that does not yet exist
            xx = poly_concave.exterior[:, 0]
            yy = poly_concave.exterior[:, 1]
            bb_x1, bb_x2 = min(xx), max(xx)
            bb_y1, bb_y2 = min(yy), max(yy)
            bb = ia.BoundingBox(
                x1=bb_x1-1e-4, y1=bb_y1-1e-4,
                x2=bb_x2+1e-4, y2=bb_y2+1e-4)
            for point in points:
                assert bb.contains(ia.Keypoint(x=point[0], y=point[1]))

    def test__remove_consecutive_duplicate_points(self):
        recoverer = _ConcavePolygonRecoverer()

        points = [
            [(0, 0), (1, 1)],
            [(0.0, 0.5), (1.0, 1.0)],
            np.float32([(0.0, 0.5), (1.0, 1.0)]),
            [(0, 0), (0, 0)],
            [(0, 0), (0, 0), (1, 0)],
            [(0, 0), (1, 0), (1, 0)],
            [(0, 0), (1, 0), (1, 0), (2, 0), (0, 0)]
        ]
        expected = [
            [(0, 0), (1, 1)],
            [(0.0, 0.5), (1.0, 1.0)],
            [(0.0, 0.5), (1.0, 1.0)],
            [(0, 0)],
            [(0, 0), (1, 0)],
            [(0, 0), (1, 0)],
            [(0, 0), (1, 0), (2, 0)]
        ]

        for points_i, expected_i in zip(points, expected):
            with self.subTest(points=points_i):
                points_deduplicated = \
                    recoverer._remove_consecutive_duplicate_points(points_i)
                assert np.allclose(points_deduplicated, expected_i)

    # TODO split into multiple tests
    def test__jitter_duplicate_points(self):
        def _norm(a, b):
            return np.linalg.norm(np.float32(a) - np.float32(b))

        cpr = _ConcavePolygonRecoverer(threshold_duplicate_points=1e-4)
        rng = iarandom.RNG(0)

        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, rng.copy())
        assert np.allclose(points, points_jittered, rtol=0, atol=1e-4)

        points = [(0, 0), (1, 0), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, rng.copy())
        assert np.allclose(points, points_jittered, rtol=0, atol=1e-4)

        points = [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)]
        points_jittered = cpr._jitter_duplicate_points(points, rng.copy())
        assert np.allclose(points, points_jittered, rtol=0, atol=1e-4)

        points = [(0, 0), (1, 0), (1 + 1e-6, 0), (1, 1), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, rng.copy())
        assert np.allclose(
            [point
             for i, point
             in enumerate(points_jittered)
             if i in [0, 1, 3, 4]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert _norm([1, 0], points_jittered[2]) >= 1e-4

        points = [(0, 0), (1, 0), (1, 1), (1 + 1e-6, 0), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, rng.copy())
        assert np.allclose(
            [point
             for i, point
             in enumerate(points_jittered)
             if i in [0, 1, 2, 4]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert _norm([1, 0], points_jittered[3]) >= 1e-4

        points = [(0, 0), (1, 0), (1, 1), (0, 1), (1 + 1e-6, 0)]
        points_jittered = cpr._jitter_duplicate_points(points, rng.copy())
        assert np.allclose(
            [point
             for i, point
             in enumerate(points_jittered)
             if i in [0, 1, 2, 3]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert _norm([1, 0], points_jittered[4]) >= 1e-4

        points = [(0, 0), (1, 0), (1 + 1e-6, 0), (1, 1), (1 + 1e-6, 0), (0, 1),
                  (1 + 1e-6, 0), (1 + 1e-6, 0 + 1e-6), (1 + 1e-6, 0 + 2e-6)]
        points_jittered = cpr._jitter_duplicate_points(points, rng.copy())
        assert np.allclose(
            [point
             for i, point
             in enumerate(points_jittered)
             if i in [0, 1, 3, 5]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert _norm([1, 0], points_jittered[2]) >= 1e-4
        assert _norm([1, 0], points_jittered[4]) >= 1e-4
        assert _norm([1, 0], points_jittered[6]) >= 1e-4
        assert _norm([1, 0], points_jittered[7]) >= 1e-4
        assert _norm([1, 0], points_jittered[8]) >= 1e-4

        points = [(0, 0), (1, 0), (0 + 1e-6, 0 - 1e-6), (1 + 1e-6, 0), (1, 1),
                  (1 + 1e-6, 0), (0, 1), (1 + 1e-6, 0), (1 + 1e-6, 0 + 1e-6),
                  (1 + 1e-6, 0 + 2e-6)]
        points_jittered = cpr._jitter_duplicate_points(points, rng.copy())
        assert np.allclose(
            [point
             for i, point
             in enumerate(points_jittered)
             if i in [0, 1, 4, 6]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert _norm([0, 0], points_jittered[2]) >= 1e-4
        assert _norm([1, 0], points_jittered[3]) >= 1e-4
        assert _norm([1, 0], points_jittered[5]) >= 1e-4
        assert _norm([1, 0], points_jittered[7]) >= 1e-4
        assert _norm([1, 0], points_jittered[8]) >= 1e-4
        assert _norm([1, 0], points_jittered[9]) >= 1e-4

    # TODO split into multiple tests
    def test__calculate_circumference(self):
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        circ = _ConcavePolygonRecoverer._calculate_circumference(points)
        assert np.allclose(circ, 4)

        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        circ = _ConcavePolygonRecoverer._calculate_circumference(points)
        assert np.allclose(circ, 4)

        points = np.float32([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        circ = _ConcavePolygonRecoverer._calculate_circumference(points)
        assert np.allclose(circ, 4)

        points = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        circ = _ConcavePolygonRecoverer._calculate_circumference(points)
        assert np.allclose(circ, 6)

    # TODO split into multiple tests
    def test__fit_best_valid_polygon(self):
        def _assert_ids_match(observed, expected):
            assert len(observed) == len(expected), (
                "len mismatch: %d vs %d" % (len(observed), len(expected)))

            max_count = 0
            for i in range(len(observed)):
                counter = 0
                for j in range(i, i+len(expected)):
                    observed_item = observed[(i+j) % len(observed)]
                    expected_item = expected[j % len(expected)]
                    if observed_item == expected_item:
                        counter += 1
                    else:
                        break

                max_count = max(max_count, counter)

            assert max_count == len(expected), (
                "count mismatch: %d vs %d" % (max_count, len(expected)))

        cpr = _ConcavePolygonRecoverer()
        rng = iarandom.RNG(0)

        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(
            points, random_state=rng.copy())
        # doing this without the list(.) wrappers fails on python2.7
        assert list(points_fit) == list(sm.xrange(len(points)))

        # square-like, but top line has one point in its center which's
        # y-coordinate is below the bottom line
        points = [(0.0, 0.0), (0.45, 0.0), (0.5, 1.5), (0.55, 0.0), (1.0, 0.0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(
            points, random_state=rng.copy())
        _assert_ids_match(points_fit, [0, 1, 3, 4, 5, 2, 6])
        assert ia.Polygon([points[idx] for idx in points_fit]).is_valid

        # |--|  |--|
        # |  |  |  |
        # |  |  |  |
        # |--|--|--|
        #    |  |
        #    ----
        # the intersection points on the bottom line are not provided,
        # hence the result is expected to have triangles at the bottom left
        # and right
        points = [(0.0, 0), (0.25, 0), (0.25, 1.25),
                  (0.75, 1.25), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(
            points, random_state=rng.copy())
        _assert_ids_match(points_fit, [0, 1, 4, 5, 6, 3, 2, 7])
        poly_observed = ia.Polygon([points[idx] for idx in points_fit])
        assert poly_observed.is_valid

        # same as above, but intersection points at the bottom line are
        # provided without oversampling, i.e. incorporating these points
        # would lead to an invalid polygon
        points = [(0.0, 0), (0.25, 0), (0.25, 1.0), (0.25, 1.25),
                  (0.75, 1.25), (0.75, 1.0), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(
            points, random_state=rng.copy())
        assert len(points_fit) >= len(points) - 2  # TODO add IoU check here
        poly_observed = ia.Polygon([points[idx] for idx in points_fit])
        assert poly_observed.is_valid

    # TODO split into multiple tests
    def test__fix_polygon_is_line(self):
        cpr = _ConcavePolygonRecoverer()
        rng = iarandom.RNG(0)

        points = [(0, 0), (1, 0), (1, 1)]
        points_fixed = cpr._fix_polygon_is_line(points, rng.copy())
        assert np.allclose(points_fixed, points, atol=0, rtol=0)

        points = [(0, 0), (1, 0), (2, 0)]
        points_fixed = cpr._fix_polygon_is_line(points, rng.copy())
        assert not np.allclose(points_fixed, points, atol=0, rtol=0)
        assert not cpr._is_polygon_line(points_fixed)
        assert np.allclose(points_fixed, points, rtol=0, atol=1e-2)

        points = [(0, 0), (0, 1), (0, 2)]
        points_fixed = cpr._fix_polygon_is_line(points, rng.copy())
        assert not np.allclose(points_fixed, points, atol=0, rtol=0)
        assert not cpr._is_polygon_line(points_fixed)
        assert np.allclose(points_fixed, points, rtol=0, atol=1e-2)

        points = [(0, 0), (1, 1), (2, 2)]
        points_fixed = cpr._fix_polygon_is_line(points, rng.copy())
        assert not np.allclose(points_fixed, points, atol=0, rtol=0)
        assert not cpr._is_polygon_line(points_fixed)
        assert np.allclose(points_fixed, points, rtol=0, atol=1e-2)

    # TODO split into multiple tests
    def test__is_polygon_line(self):
        points = [(0, 0), (1, 0), (1, 1)]
        assert not _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert not _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert not _ConcavePolygonRecoverer._is_polygon_line(points)

        points = np.float32([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert not _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0)]
        assert _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (2, 0)]
        assert _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (1, 0)]
        assert _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (1, 0), (2, 0)]
        assert _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (1, 0), (2, 0), (0.5, 0)]
        assert _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (1, 0), (2, 0), (1, 1)]
        assert not _ConcavePolygonRecoverer._is_polygon_line(points)

    # TODO split into multiple tests
    def test__generate_intersection_points(self):
        cpr = _ConcavePolygonRecoverer()

        # triangle
        points = [(0.5, 0), (1, 1), (0, 1)]
        points_inter = cpr._generate_intersection_points(
            points, one_point_per_intersection=False)
        assert points_inter == [[], [], []]

        # rotated square
        points = [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)]
        points_inter = cpr._generate_intersection_points(
            points, one_point_per_intersection=False)
        assert points_inter == [[], [], [], []]

        # square
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        points_inter = cpr._generate_intersection_points(
            points, one_point_per_intersection=False)
        assert points_inter == [[], [], [], []]

        # |--|  |--|
        # |  |__|  |
        # |        |
        # |--------|
        points = [(0.0, 0), (0.25, 0), (0.25, 0.25),
                  (0.75, 0.25), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_inter = cpr._generate_intersection_points(
            points, one_point_per_intersection=False)
        assert points_inter == [[], [], [], [], [], [], [], []]

        # same as above, but middle part goes much further down,
        # crossing the bottom line
        points = [(0.0, 0), (0.25, 0), (0.25, 1.25),
                  (0.75, 1.25), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_inter = cpr._generate_intersection_points(
            points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [(0.25, 1.0)], [], [(0.75, 1.0)], [], [],
             [(0.75, 1.0), (0.25, 1.0)], []])

        # square-like structure with intersections in top right area
        points = [(0, 0), (0.5, 0), (1.01, 0.5), (1.0, 0), (1, 1), (0, 1),
                  (0, 0)]
        points_inter = cpr._generate_intersection_points(
            points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [(1.0, 0.4902)], [], [(1.0, 0.4902)], [], [], []],
            atol=1e-2)

        # same as above, but with a second intersection in bottom left
        points = [(0, 0), (0.5, 0), (1.01, 0.5), (1.0, 0), (1, 1), (-0.25, 1),
                  (0, 1.25)]
        points_inter = cpr._generate_intersection_points(
            points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [(1.0, 0.4902)], [], [(1.0, 0.4902)], [(0, 1.0)], [],
             [(0, 1.0)]],
            atol=1e-2)

        # double triangle with point in center that is shared by both triangles
        points = [(0, 0), (0.5, 0.5), (1.0, 0), (1.0, 1.0), (0.5, 0.5),
                  (0, 1.0)]
        points_inter = cpr._generate_intersection_points(
            points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [], [], [], [], []])

    # TODO split into multiple tests
    def test__oversample_intersection_points(self):
        cpr = _ConcavePolygonRecoverer()
        cpr.oversampling = 0.1

        points = [(0.0, 0.0), (1.0, 0.0)]
        segment_add_points_sorted = [[(0.5, 0.0)], []]
        points_oversampled = cpr._oversample_intersection_points(
            points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.55, 0.0)], []],
            atol=1e-4
        )

        points = [(0.0, 0.0), (2.0, 0.0)]
        segment_add_points_sorted = [[(0.5, 0.0)], []]
        points_oversampled = cpr._oversample_intersection_points(
            points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.65, 0.0)], []],
            atol=1e-4
        )

        points = [(0.0, 0.0), (1.0, 0.0)]
        segment_add_points_sorted = [[(0.5, 0.0), (0.6, 0.0)], []]
        points_oversampled = cpr._oversample_intersection_points(
            points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.51, 0.0), (0.59, 0.0), (0.6, 0.0),
              (0.64, 0.0)], []],
            atol=1e-4
        )

        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        segment_add_points_sorted = [[(0.5, 0.0)], [], [(0.8, 1.0)],
                                     [(0.0, 0.7)]]
        points_oversampled = cpr._oversample_intersection_points(
            points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.55, 0.0)],
             [],
             [(0.82, 1.0), (0.8, 1.0), (0.72, 1.0)],
             [(0.0, 0.73), (0.0, 0.7), (0.0, 0.63)]],
            atol=1e-4
        )

    # TODO split into multiple tests
    def test__insert_intersection_points(self):
        points = [(0, 0), (1, 0), (2, 0)]
        segments_add_point_sorted = [[], [], []]
        points_inserted = _ConcavePolygonRecoverer._insert_intersection_points(
            points, segments_add_point_sorted)
        assert points_inserted == points

        segments_add_point_sorted = [[(0.5, 0)], [], []]
        points_inserted = _ConcavePolygonRecoverer._insert_intersection_points(
            points, segments_add_point_sorted)
        assert points_inserted == [(0, 0), (0.5, 0), (1, 0), (2, 0)]

        segments_add_point_sorted = [[(0.5, 0), (0.75, 0)], [], []]
        points_inserted = _ConcavePolygonRecoverer._insert_intersection_points(
            points, segments_add_point_sorted)
        assert points_inserted == [(0, 0), (0.5, 0), (0.75, 0), (1, 0), (2, 0)]

        segments_add_point_sorted = [[(0.5, 0)], [(1.5, 0)], []]
        points_inserted = _ConcavePolygonRecoverer._insert_intersection_points(
            points, segments_add_point_sorted)
        assert points_inserted == [(0, 0), (0.5, 0), (1, 0), (1.5, 0), (2, 0)]

        segments_add_point_sorted = [[(0.5, 0)], [(1.5, 0)], [(2.5, 0)]]
        points_inserted = _ConcavePolygonRecoverer._insert_intersection_points(
            points, segments_add_point_sorted)
        assert points_inserted == [(0, 0), (0.5, 0), (1, 0), (1.5, 0), (2, 0),
                                   (2.5, 0)]
