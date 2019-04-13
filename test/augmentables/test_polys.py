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
from imgaug.testutils import reseed
from imgaug.augmentables.polys import (
    _convert_points_to_shapely_line_string, _ConcavePolygonRecoverer
)


def main():
    test_Polygon___init__()
    test_Polygon_xx()
    test_Polygon_yy()
    test_Polygon_xx_int()
    test_Polygon_yy_int()
    test_Polygon_is_valid()
    test_Polygon_area()
    test_Polygon_height()
    test_Polygon_width()
    test_Polygon_project()
    test_Polygon_find_closest_point_idx()
    test_Polygon_is_fully_within_image()
    test_Polygon_is_partly_within_image()
    test_Polygon_is_out_of_image()
    test_Polygon_cut_out_of_image()
    test_Polygon_clip_out_of_image()
    test_Polygon_shift()
    test_Polygon_draw_on_image()
    test_Polygon_extract_from_image()
    test_Polygon_to_shapely_polygon()
    test_Polygon_to_bounding_box()
    test_Polygon_from_shapely()
    test_Polygon_copy()
    test_Polygon_deepcopy()
    test_Polygon___repr__()
    test_Polygon___str__()
    test_Polygon_exterior_almost_equals()
    test_Polygon_almost_equals()
    test___convert_points_to_shapely_line_string()


def test_Polygon___init__():
    # exterior is list of Keypoint or
    poly = ia.Polygon([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=0.5, y=2.5)])
    assert poly.exterior.dtype.type == np.float32
    assert np.allclose(
        poly.exterior,
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )

    # exterior is list of tuple of floats
    poly = ia.Polygon([(0.0, 0.0), (1.0, 1.0), (0.5, 2.5)])
    assert poly.exterior.dtype.type == np.float32
    assert np.allclose(
        poly.exterior,
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )

    # exterior is list of tuple of integer
    poly = ia.Polygon([(0, 0), (1, 1), (1, 3)])
    assert poly.exterior.dtype.type == np.float32
    assert np.allclose(
        poly.exterior,
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 3.0]
        ])
    )

    # exterior is (N,2) ndarray
    poly = ia.Polygon(
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )
    assert poly.exterior.dtype.type == np.float32
    assert np.allclose(
        poly.exterior,
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )

    # exterior is (N,2) ndarray in float64
    poly = ia.Polygon(
        np.float64([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )
    assert poly.exterior.dtype.type == np.float32
    assert np.allclose(
        poly.exterior,
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )

    # arrays without points
    poly = ia.Polygon([])
    assert poly.exterior.dtype.type == np.float32
    assert poly.exterior.shape == (0, 2)

    poly = ia.Polygon(np.zeros((0, 2), dtype=np.float32))
    assert poly.exterior.dtype.type == np.float32
    assert poly.exterior.shape == (0, 2)

    # bad array shape
    got_exception = False
    try:
        _ = ia.Polygon(np.zeros((8,), dtype=np.float32))
    except:
        got_exception = True
    assert got_exception

    # label
    poly = ia.Polygon([(0, 0)])
    assert poly.label is None
    poly = ia.Polygon([(0, 0)], label="test")
    assert poly.label == "test"


def test_Polygon_xx():
    poly = ia.Polygon([(0, 0), (1, 0), (1.5, 0), (4.1, 1), (2.9, 2.0)])
    assert poly.xx.dtype.type == np.float32
    assert np.allclose(poly.xx, np.float32([0.0, 1.0, 1.5, 4.1, 2.9]))

    poly = ia.Polygon([])
    assert poly.xx.dtype.type == np.float32
    assert poly.xx.shape == (0,)


def test_Polygon_yy():
    poly = ia.Polygon([(0, 0), (0, 1), (0, 1.5), (1, 4.1), (2.0, 2.9)])
    assert poly.yy.dtype.type == np.float32
    assert np.allclose(poly.yy, np.float32([0.0, 1.0, 1.5, 4.1, 2.9]))

    poly = ia.Polygon([])
    assert poly.yy.dtype.type == np.float32
    assert poly.yy.shape == (0,)


def test_Polygon_xx_int():
    poly = ia.Polygon([(0, 0), (1, 0), (1.5, 0), (4.1, 1), (2.9, 2.0)])
    assert poly.xx_int.dtype.type == np.int32
    assert np.allclose(poly.xx_int, np.int32([0, 1, 2, 4, 3]))

    poly = ia.Polygon([])
    assert poly.xx_int.dtype.type == np.int32
    assert poly.xx_int.shape == (0,)


def test_Polygon_yy_int():
    poly = ia.Polygon([(0, 0), (0, 1), (0, 1.5), (1, 4.1), (2.0, 2.9)])
    assert poly.yy_int.dtype.type == np.int32
    assert np.allclose(poly.yy_int, np.int32([0, 1, 2, 4, 3]))

    poly = ia.Polygon([])
    assert poly.yy_int.dtype.type == np.int32
    assert poly.yy_int.shape == (0,)


def test_Polygon_is_valid():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_valid

    poly = ia.Polygon([])
    assert not poly.is_valid

    poly = ia.Polygon([(0, 0)])
    assert not poly.is_valid

    poly = ia.Polygon([(0, 0), (1, 0)])
    assert not poly.is_valid

    poly = ia.Polygon([(0, 0), (1, 0), (-1, 0.5), (1, 1), (0, 1)])
    assert not poly.is_valid

    poly = ia.Polygon([(0, 0), (1, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_valid


def test_Polygon_area():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert 1.0 - 1e-8 < poly.area < 1.0 + 1e-8

    poly = ia.Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    assert 2.0 - 1e-8 < poly.area < 2.0 + 1e-8

    poly = ia.Polygon([(0, 0), (1, 1), (0, 1)])
    assert 1/2 - 1e-8 < poly.area < 1/2 + 1e-8

    poly = ia.Polygon([(0, 0), (1, 1)])
    got_exception = False
    try:
        _ = poly.area
    except Exception as exc:
        assert "Cannot compute the polygon's area because" in str(exc)
        got_exception = True
    assert got_exception


def test_Polygon_height():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert np.allclose(poly.height, 1.0, atol=1e-8, rtol=0)

    poly = ia.Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
    assert np.allclose(poly.height, 2.0, atol=1e-8, rtol=0)

    poly = ia.Polygon([(0, 0), (1, 1), (0, 1)])
    assert np.allclose(poly.height, 1.0, atol=1e-8, rtol=0)

    poly = ia.Polygon([(0, 0), (1, 1)])
    assert np.allclose(poly.height, 1.0, atol=1e-8, rtol=0)

    poly = ia.Polygon([(0, 0)])
    assert np.allclose(poly.height, 0.0, atol=1e-8, rtol=0)


def test_Polygon_width():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert np.allclose(poly.width, 1.0, atol=1e-8, rtol=0)

    poly = ia.Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    assert np.allclose(poly.width, 2.0, atol=1e-8, rtol=0)

    poly = ia.Polygon([(0, 0), (1, 1), (0, 1)])
    assert np.allclose(poly.width, 1.0, atol=1e-8, rtol=0)

    poly = ia.Polygon([(0, 0), (1, 1)])
    assert np.allclose(poly.width, 1.0, atol=1e-8, rtol=0)

    poly = ia.Polygon([(0, 0)])
    assert np.allclose(poly.width, 0.0, atol=1e-8, rtol=0)


def test_Polygon_project():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_proj = poly.project((1, 1), (1, 1))
    assert poly_proj.exterior.dtype.type == np.float32
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

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_proj = poly.project((1, 1), (2, 2))
    assert poly_proj.exterior.dtype.type == np.float32
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

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_proj = poly.project((1, 1), (2, 1))
    assert poly_proj.exterior.dtype.type == np.float32
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

    poly = ia.Polygon([])
    poly_proj = poly.project((1, 1), (2, 2))
    assert poly_proj.exterior.dtype.type == np.float32
    assert poly_proj.exterior.shape == (0, 2)


def test_Polygon_find_closest_point_idx():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    closest_idx = poly.find_closest_point_index(x=0, y=0)
    assert closest_idx == 0
    closest_idx = poly.find_closest_point_index(x=1, y=0)
    assert closest_idx == 1
    closest_idx = poly.find_closest_point_index(x=1.0001, y=-0.001)
    assert closest_idx == 1
    closest_idx = poly.find_closest_point_index(x=0.2, y=0.2)
    assert closest_idx == 0

    closest_idx, distance = poly.find_closest_point_index(x=0, y=0, return_distance=True)
    assert closest_idx == 0
    assert np.allclose(distance, 0.0)
    closest_idx, distance = poly.find_closest_point_index(x=0.1, y=0.15, return_distance=True)
    assert closest_idx == 0
    assert np.allclose(distance, np.sqrt((0.1**2) + (0.15**2)))
    closest_idx, distance = poly.find_closest_point_index(x=0.9, y=0.15, return_distance=True)
    assert closest_idx == 1
    assert np.allclose(distance, np.sqrt(((1.0-0.9)**2) + (0.15**2)))


def test_Polygon_is_fully_within_image():
    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_fully_within_image((1, 1, 3))

    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_fully_within_image((1, 1))

    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_fully_within_image(np.zeros((1, 1, 3), dtype=np.uint8))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert not poly.is_fully_within_image((1, 1, 3))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert not poly.is_fully_within_image((1, 1))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert not poly.is_fully_within_image(np.zeros((1, 1, 3), dtype=np.uint8))

    poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    assert not poly.is_fully_within_image((1, 1, 3))


def test_Polygon_is_partly_within_image():
    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_partly_within_image((1, 1, 3))

    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_partly_within_image((1, 1))

    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_partly_within_image(np.zeros((1, 1, 3), dtype=np.uint8))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_partly_within_image((1, 1, 3))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_partly_within_image((1, 1))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_partly_within_image(np.zeros((1, 1, 3), dtype=np.uint8))

    poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    assert not poly.is_partly_within_image((1, 1, 3))

    poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    assert not poly.is_partly_within_image((1, 1))

    poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    assert not poly.is_partly_within_image(np.zeros((1, 1, 3), dtype=np.uint8))


def test_Polygon_is_out_of_image():
    for shape in [(1, 1, 3), (1, 1), np.zeros((1, 1, 3), dtype=np.uint8)]:
        poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
        assert not poly.is_out_of_image(shape, partly=False, fully=False)
        assert not poly.is_out_of_image(shape, partly=True, fully=False)
        assert not poly.is_out_of_image(shape, partly=False, fully=True)
        assert not poly.is_out_of_image(shape, partly=True, fully=True)

        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        shape = np.zeros((1, 1, 3), dtype=np.uint8)
        assert not poly.is_out_of_image(shape, partly=False, fully=False)
        assert poly.is_out_of_image(shape, partly=True, fully=False)
        assert not poly.is_out_of_image(shape, partly=False, fully=True)
        assert poly.is_out_of_image(shape, partly=True, fully=True)

        poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
        shape = (1, 1, 3)
        assert not poly.is_out_of_image(shape, partly=False, fully=False)
        assert not poly.is_out_of_image(shape, partly=True, fully=False)
        assert poly.is_out_of_image(shape, partly=False, fully=True)
        assert poly.is_out_of_image(shape, partly=True, fully=True)

    poly = ia.Polygon([(8, 11), (11, 8), (11, 11)])
    assert not poly.is_out_of_image((100, 100, 3), fully=True, partly=True)
    assert not poly.is_out_of_image((10, 10, 3), fully=True, partly=False)
    assert poly.is_out_of_image((10, 10, 3), fully=False, partly=True)

    poly = ia.Polygon([])
    got_exception = False
    try:
        poly.is_out_of_image((1, 1, 3))
    except Exception as exc:
        assert "Cannot determine whether the polygon is inside the image" in str(exc)
        got_exception = True
    assert got_exception


def test_Polygon_cut_out_of_image():
    with warnings.catch_warnings(record=True) as caught_warnings:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        _test_Polygon_cut_clip(lambda poly, image: poly.cut_out_of_image(image))
        # Verify
        # get multiple warnings here, one for each function call
        assert all([
            "is deprecated" in str(msg.message)
            for msg in caught_warnings])


def test_Polygon_clip_out_of_image():
    _test_Polygon_cut_clip(lambda poly, image: poly.clip_out_of_image(image))


def _test_Polygon_cut_clip(func):
    # poly inside image
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label=None)
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    multipoly_clipped = func(poly, image)
    assert isinstance(multipoly_clipped, list)
    assert len(multipoly_clipped) == 1
    assert multipoly_clipped[0].exterior_almost_equals(poly.exterior)
    assert multipoly_clipped[0].label is None

    # square poly shifted by x=0.5, y=0.5 => half out of image
    poly = ia.Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)], label="test")
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    multipoly_clipped = func(poly, image)
    assert isinstance(multipoly_clipped, list)
    assert len(multipoly_clipped) == 1
    assert multipoly_clipped[0].exterior_almost_equals(np.float32([
        [0.5, 0.5],
        [1.0, 0.5],
        [1.0, 1.0],
        [0.5, 1.0]
    ]))
    assert multipoly_clipped[0].label == "test"

    # non-square poly, with one rectangle on the left side of the image and one on the right side,
    # both sides are connected by a thin strip below the image
    # after clipping it should become two rectangles
    poly = ia.Polygon([(-0.1, 0.0), (0.4, 0.0), (0.4, 1.1), (0.6, 1.1), (0.6, 0.0), (1.1, 0.0),
                       (1.1, 1.2), (-0.1, 1.2)],
                      label="test")
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    multipoly_clipped = func(poly, image)
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

    # poly outside of image
    poly = ia.Polygon([(10.0, 10.0)])
    multipoly_clipped = func(poly, (5, 5, 3))
    assert isinstance(multipoly_clipped, list)
    assert len(multipoly_clipped) == 0


def test_Polygon_shift():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")

    # make sure that shift does not change poly inplace
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

    for v in [1, 0, -1, 0.5]:
        # top/bottom
        poly_shifted = poly.shift(top=v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0, 0 + v],
            [1, 0 + v],
            [1, 1 + v],
            [0, 1 + v]
        ]))
        assert poly_shifted.label == "test"

        poly_shifted = poly.shift(bottom=v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0, 0 - v],
            [1, 0 - v],
            [1, 1 - v],
            [0, 1 - v]
        ]))
        assert poly_shifted.label == "test"

        poly_shifted = poly.shift(top=v, bottom=-v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0, 0 + 2*v],
            [1, 0 + 2*v],
            [1, 1 + 2*v],
            [0, 1 + 2*v]
        ]))
        assert poly_shifted.label == "test"

        # left/right
        poly_shifted = poly.shift(left=v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0 + v, 0],
            [1 + v, 0],
            [1 + v, 1],
            [0 + v, 1]
        ]))
        assert poly_shifted.label == "test"

        poly_shifted = poly.shift(right=v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0 - v, 0],
            [1 - v, 0],
            [1 - v, 1],
            [0 - v, 1]
        ]))
        assert poly_shifted.label == "test"

        poly_shifted = poly.shift(left=v, right=-v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0 + 2 * v, 0],
            [1 + 2 * v, 0],
            [1 + 2 * v, 1],
            [0 + 2 * v, 1]
        ]))
        assert poly_shifted.label == "test"


def test_Polygon_draw_on_image():
    image = np.tile(np.arange(100).reshape(10, 10, 1), (1, 1, 3)).astype(np.uint8)

    # simple drawing of square
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
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
    assert np.sum(image) == 3 * np.sum(np.arange(100))  # draw did not change original image (copy=True)
    for c_idx, value in enumerate([0, 255, 0]):
        assert np.all(image_poly[2:9, 2:3, c_idx] == np.zeros((7, 1), dtype=np.uint8) + value)  # left boundary
        assert np.all(image_poly[2:9, 8:9, c_idx] == np.zeros((7, 1), dtype=np.uint8) + value)  # right boundary
        assert np.all(image_poly[2:3, 2:9, c_idx] == np.zeros((1, 7), dtype=np.uint8) + value)  # top boundary
        assert np.all(image_poly[8:9, 2:9, c_idx] == np.zeros((1, 7), dtype=np.uint8) + value)  # bottom boundary
    expected = np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1))
    assert np.all(image_poly[3:8, 3:8, :] == expected)

    # simple drawing of square, use only "color" arg
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    image_poly = poly.draw_on_image(image,
                                    color=[0, 255, 0],
                                    alpha=1.0,
                                    alpha_face=1.0,
                                    alpha_lines=1.0,
                                    alpha_points=0.0,
                                    raise_if_out_of_image=False)
    assert image_poly.dtype.type == np.uint8
    assert image_poly.shape == (10, 10, 3)
    assert np.sum(image) == 3 * np.sum(np.arange(100))  # draw did not change original image (copy=True)
    for c_idx, value in enumerate([0, 0.5*255, 0]):
        value = int(np.round(value))
        assert np.all(image_poly[2:9, 2:3, c_idx] == np.zeros((7, 1), dtype=np.uint8) + value)  # left boundary
        assert np.all(image_poly[2:9, 8:9, c_idx] == np.zeros((7, 1), dtype=np.uint8) + value)  # right boundary
        assert np.all(image_poly[2:3, 2:9, c_idx] == np.zeros((1, 7), dtype=np.uint8) + value)  # top boundary
        assert np.all(image_poly[8:9, 2:9, c_idx] == np.zeros((1, 7), dtype=np.uint8) + value)  # bottom boundary
    expected = np.tile(np.uint8([0, 255, 0]).reshape((1, 1, 3)), (5, 5, 1))
    assert np.all(image_poly[3:8, 3:8, :] == expected)

    # simple drawing of square with float32 input
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
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
        assert np.allclose(image_poly[2:9, 2:3, c_idx], np.zeros((7, 1), dtype=np.float32) + value)  # left boundary
        assert np.allclose(image_poly[2:9, 8:9, c_idx], np.zeros((7, 1), dtype=np.float32) + value)  # right boundary
        assert np.allclose(image_poly[2:3, 2:9, c_idx], np.zeros((1, 7), dtype=np.float32) + value)  # top boundary
        assert np.allclose(image_poly[8:9, 2:9, c_idx], np.zeros((1, 7), dtype=np.float32) + value)  # bottom boundary
    expected = np.tile(np.float32([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1))
    assert np.allclose(image_poly[3:8, 3:8, :], expected)

    # drawing of poly that is half out of image
    poly = ia.Polygon([(2, 2+5), (8, 2+5), (8, 8+5), (2, 8+5)])
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
    assert np.sum(image) == 3 * np.sum(np.arange(100))  # draw did not change original image (copy=True)
    for c_idx, value in enumerate([0, 255, 0]):
        assert np.all(image_poly[2+5:, 2:3, c_idx] == np.zeros((3, 1), dtype=np.uint8) + value)  # left boundary
        assert np.all(image_poly[2+5:, 8:9, c_idx] == np.zeros((3, 1), dtype=np.uint8) + value)  # right boundary
        assert np.all(image_poly[2+5:3+5, 2:9, c_idx] == np.zeros((1, 7), dtype=np.uint8) + value)  # top boundary
    expected = np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (2, 5, 1))
    assert np.all(image_poly[3+5:, 3:8, :] == expected)

    # drawing of poly that is half out of image, with raise_if_out_of_image=True
    poly = ia.Polygon([(2, 2+5), (8, 2+5), (8, 8+5), (0, 8+5)])
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
    assert not got_exception  # only polygons fully outside of the image plane lead to exceptions

    # drawing of poly that is fully out of image
    poly = ia.Polygon([(100, 100), (100+10, 100), (100+10, 100+10), (100, 100+10)])
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

    # drawing of poly that is fully out of image, with raise_if_out_of_image=True
    poly = ia.Polygon([(100, 100), (100+10, 100), (100+10, 100+10), (100, 100+10)])
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

    # face invisible via alpha
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
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
    assert np.sum(image) == 3 * np.sum(np.arange(100))  # draw did not change original image (copy=True)
    for c_idx, value in enumerate([0, 255, 0]):
        assert np.all(image_poly[2:9, 2:3, c_idx] == np.zeros((7, 1), dtype=np.uint8) + value)  # left boundary
    assert np.all(image_poly[3:8, 3:8, :] == image[3:8, 3:8, :])

    # boundary invisible via alpha
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
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
    assert np.sum(image) == 3 * np.sum(np.arange(100))  # draw did not change original image (copy=True)
    expected = np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (6, 6, 1))
    assert np.all(image_poly[2:8, 2:8, :] == expected)

    # alpha=0.8
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
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
        assert np.all(image_poly[2:9, 8:9, c_idx] == expected)  # right boundary
    expected = (0.8 * 0.5) * np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1)) \
        + (1 - (0.8 * 0.5)) * image[3:8, 3:8, :]
    assert np.all(image_poly[3:8, 3:8, :] == np.round(expected).astype(np.uint8))

    # alpha of fill and perimeter 0.5
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
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
        assert np.all(image_poly[2:9, 8:9, c_idx] == expected)  # right boundary
    expected = 0.5 * np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1)) \
        + 0.5 * image[3:8, 3:8, :]
    assert np.all(image_poly[3:8, 3:8, :] == np.round(expected).astype(np.uint8))

    # copy=False
    # test deactivated as the function currently does not offer a copy argument
    """
    image_cp = np.copy(image)
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    image_poly = poly.draw_on_image(image_cp,
                                    color_face=[32, 128, 32], color_boundary=[0, 255, 0],
                                    alpha_face=1.0, alpha_boundary=1.0,
                                    raise_if_out_of_image=False)
    assert image_poly.dtype.type == np.uint8
    assert image_poly.shape == (10, 10, 3)
    assert np.all(image_cp == image_poly)
    assert not np.all(image_cp == image)
    for c_idx, value in enumerate([0, 255, 0]):
        assert np.all(image_poly[2:9, 2:3, c_idx] == np.zeros((6, 1, 3), dtype=np.uint8) + value)  # left boundary
        assert np.all(image_cp[2:9, 2:3, c_idx] == np.zeros((6, 1, 3), dtype=np.uint8) + value)  # left boundary
    expected = np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1))
    assert np.all(image_poly[3:8, 3:8, :] == expected)
    assert np.all(image_cp[3:8, 3:8, :] == expected)
    """


def test_Polygon_extract_from_image():
    image = np.arange(20*20*2).reshape(20, 20, 2).astype(np.int32)

    # inside image and completely covers it
    poly = ia.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    subimage = poly.extract_from_image(image)
    assert np.array_equal(subimage, image[0:10, 0:10, :])

    # inside image, subpart of it (not all may be extracted)
    poly = ia.Polygon([(1, 1), (9, 1), (9, 9), (1, 9)])
    subimage = poly.extract_from_image(image)
    assert np.array_equal(subimage, image[1:9, 1:9, :])

    # inside image, two image areas that don't belong to the polygon but have to be extracted
    poly = ia.Polygon([(0, 0), (10, 0), (10, 5), (20, 5),
                       (20, 20), (10, 20), (10, 5), (0, 5)])
    subimage = poly.extract_from_image(image)
    expected = np.copy(image)
    expected[:5, 10:, :] = 0  # top right block
    expected[5:, :10, :] = 0  # left bottom block
    assert np.array_equal(subimage, expected)

    # partially out of image
    poly = ia.Polygon([(-5, 0), (5, 0), (5, 10), (-5, 10)])
    subimage = poly.extract_from_image(image)
    expected = np.zeros((10, 10, 2), dtype=np.int32)
    expected[0:10, 5:10, :] = image[0:10, 0:5, :]
    assert np.array_equal(subimage, expected)

    # fully out of image
    poly = ia.Polygon([(30, 0), (40, 0), (40, 10), (30, 10)])
    subimage = poly.extract_from_image(image)
    expected = np.zeros((10, 10, 2), dtype=np.int32)
    assert np.array_equal(subimage, expected)

    # inside image, subpart of it
    # float coordinates, rounded so that the whole image will be extracted
    poly = ia.Polygon([(0.4, 0.4), (9.6, 0.4), (9.6, 9.6), (0.4, 9.6)])
    subimage = poly.extract_from_image(image)
    assert np.array_equal(subimage, image[0:10, 0:10, :])

    # inside image, subpart of it
    # float coordinates, rounded so that x/y 0<=i<9 will be extracted (instead of 0<=i<10)
    poly = ia.Polygon([(0.5, 0.5), (9.4, 0.5), (9.4, 9.4), (0.5, 9.4)])
    subimage = poly.extract_from_image(image)
    assert np.array_equal(subimage, image[0:9, 0:9, :])

    # inside image, subpart of it
    # float coordinates, rounded so that x/y 1<=i<9 will be extracted (instead of 0<=i<10)
    poly = ia.Polygon([(0.51, 0.51), (9.4, 0.51), (9.4, 9.4), (0.51, 9.4)])
    subimage = poly.extract_from_image(image)
    assert np.array_equal(subimage, image[1:9, 1:9, :])

    # error for invalid polygons
    got_exception = False
    poly = ia.Polygon([(0.51, 0.51), (9.4, 0.51)])
    try:
        _ = poly.extract_from_image(image)
    except Exception as exc:
        assert "Polygon must be made up" in str(exc)
        got_exception = True
    assert got_exception


def test_Polygon_change_first_point_by_coords():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_coords(x=0, y=0)
    assert np.allclose(poly.exterior, poly_reordered.exterior)

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_coords(x=1, y=0)
    # make sure that it does not reorder inplace
    assert np.allclose(poly.exterior, np.float32([[0, 0], [1, 0], [1, 1]]))
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [1, 1], [0, 0]]))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_coords(x=1, y=1)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 1], [0, 0], [1, 0]]))

    # inaccurate point, but close enough
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_coords(x=1.0, y=0.01, max_distance=0.1)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [1, 1], [0, 0]]))

    # inaccurate point, but close enough (infinite max distance)
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_coords(x=1.0, y=0.01, max_distance=None)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [1, 1], [0, 0]]))

    # point too far away
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    got_exception = False
    try:
        _ = poly.change_first_point_by_coords(x=1.0, y=0.01, max_distance=0.001)
    except Exception as exc:
        assert "Closest found point " in str(exc)
        got_exception = True
    assert got_exception

    # reorder with two points
    poly = ia.Polygon([(0, 0), (1, 0)])
    poly_reordered = poly.change_first_point_by_coords(x=1, y=0)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [0, 0]]))

    # reorder with one point
    poly = ia.Polygon([(0, 0)])
    poly_reordered = poly.change_first_point_by_coords(x=0, y=0)
    assert np.allclose(poly_reordered.exterior, np.float32([[0, 0]]))

    # invalid polygon
    got_exception = False
    poly = ia.Polygon([])
    try:
        _ = poly.change_first_point_by_coords(x=0, y=0)
    except Exception as exc:
        assert "Cannot reorder polygon points" in str(exc)
        got_exception = True
    assert got_exception


def test_Polygon_change_first_point_by_index():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_index(0)
    assert np.allclose(poly.exterior, poly_reordered.exterior)

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_index(1)
    # make sure that it does not reorder inplace
    assert np.allclose(poly.exterior, np.float32([[0, 0], [1, 0], [1, 1]]))
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [1, 1], [0, 0]]))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_index(2)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 1], [0, 0], [1, 0]]))

    # reorder with two points
    poly = ia.Polygon([(0, 0), (1, 0)])
    poly_reordered = poly.change_first_point_by_index(1)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [0, 0]]))

    # reorder with one point
    poly = ia.Polygon([(0, 0)])
    poly_reordered = poly.change_first_point_by_index(0)
    assert np.allclose(poly_reordered.exterior, np.float32([[0, 0]]))

    # idx out of bounds
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    got_exception = False
    try:
        _ = poly.change_first_point_by_index(3)
    except AssertionError:
        got_exception = True
    assert got_exception

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    got_exception = False
    try:
        _ = poly.change_first_point_by_index(-1)
    except AssertionError:
        got_exception = True
    assert got_exception

    poly = ia.Polygon([(0, 0)])
    got_exception = False
    try:
        _ = poly.change_first_point_by_index(1)
    except AssertionError:
        got_exception = True
    assert got_exception

    poly = ia.Polygon([])
    got_exception = False
    try:
        _ = poly.change_first_point_by_index(0)
    except AssertionError:
        got_exception = True
    assert got_exception


def test_Polygon_to_shapely_line_string():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    ls = poly.to_shapely_line_string()
    assert np.allclose(ls.coords, np.float32([[0, 0], [1, 0], [1, 1]]))

    # two point polygon
    poly = ia.Polygon([(0, 0), (1, 0)])
    ls = poly.to_shapely_line_string()
    assert np.allclose(ls.coords, np.float32([[0, 0], [1, 0]]))

    # one point polygon
    poly = ia.Polygon([(0, 0)])
    got_exception = False
    try:
        _ = poly.to_shapely_line_string()
    except Exception as exc:
        assert "Conversion to shapely line string requires at least two points" in str(exc)
        got_exception = True
    assert got_exception

    # zero point polygon
    poly = ia.Polygon([])
    got_exception = False
    try:
        _ = poly.to_shapely_line_string()
    except Exception as exc:
        assert "Conversion to shapely line string requires at least two points" in str(exc)
        got_exception = True
    assert got_exception

    # closed line string
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    ls = poly.to_shapely_line_string(closed=True)
    assert np.allclose(ls.coords, np.float32([[0, 0], [1, 0], [1, 1], [0, 0]]))

    # interpolation
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    ls = poly.to_shapely_line_string(interpolate=1)
    assert np.allclose(ls.coords, np.float32([[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 0.5]]))

    # interpolation with 2 steps
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    ls = poly.to_shapely_line_string(interpolate=2)
    assert np.allclose(ls.coords, np.float32([
        [0, 0], [1/3, 0], [2/3, 0],
        [1, 0], [1, 1/3], [1, 2/3],
        [1, 1], [2/3, 2/3], [1/3, 1/3]
    ]))

    # interpolation with closed=True
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    ls = poly.to_shapely_line_string(closed=True, interpolate=1)
    assert np.allclose(ls.coords, np.float32([[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 0.5], [0, 0]]))


def test_Polygon_to_shapely_polygon():
    exterior = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly = ia.Polygon(exterior)
    poly_shapely = poly.to_shapely_polygon()
    for (x_exp, y_exp), (x_obs, y_obs) in zip(exterior, poly_shapely.exterior.coords):
        assert x_exp - 1e-8 < x_obs < x_exp + 1e-8
        assert y_exp - 1e-8 < y_obs < y_exp + 1e-8


def test_Polygon_to_bounding_box():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    bb = poly.to_bounding_box()
    assert 0 - 1e-8 < bb.x1 < 0 + 1e-8
    assert 0 - 1e-8 < bb.y1 < 0 + 1e-8
    assert 1 - 1e-8 < bb.x2 < 1 + 1e-8
    assert 1 - 1e-8 < bb.y2 < 1 + 1e-8

    poly = ia.Polygon([(0.5, 0), (1, 1), (0, 1)])
    bb = poly.to_bounding_box()
    assert 0 - 1e-8 < bb.x1 < 0 + 1e-8
    assert 0 - 1e-8 < bb.y1 < 0 + 1e-8
    assert 1 - 1e-8 < bb.x2 < 1 + 1e-8
    assert 1 - 1e-8 < bb.y2 < 1 + 1e-8

    poly = ia.Polygon([(0.5, 0.5), (2, 0.1), (1, 1)])
    bb = poly.to_bounding_box()
    assert 0.5 - 1e-8 < bb.x1 < 0.5 + 1e-8
    assert 0.1 - 1e-8 < bb.y1 < 0.1 + 1e-8
    assert 2.0 - 1e-8 < bb.x2 < 2.0 + 1e-8
    assert 1.0 - 1e-8 < bb.y2 < 1.0 + 1e-8


def test_Polygon_to_line_string():
    poly = ia.Polygon([])
    ls = poly.to_line_string(closed=False)
    assert len(ls.coords) == 0
    assert ls.label is None

    poly = ia.Polygon([])
    ls = poly.to_line_string(closed=True)
    assert len(ls.coords) == 0
    assert ls.label is None

    poly = ia.Polygon([], label="foo")
    ls = poly.to_line_string(closed=False)
    assert len(ls.coords) == 0
    assert ls.label == "foo"

    poly = ia.Polygon([(0, 0)])
    ls = poly.to_line_string(closed=False)
    assert len(ls.coords) == 1
    assert ls.label is None

    poly = ia.Polygon([(0, 0)])
    ls = poly.to_line_string(closed=True)
    assert len(ls.coords) == 1
    assert ls.coords_almost_equals([(0, 0)])
    assert ls.label is None

    poly = ia.Polygon([(0, 0), (1, 1)])
    ls = poly.to_line_string(closed=False)
    assert len(ls.coords) == 2
    assert ls.coords_almost_equals([(0, 0), (1, 1)])
    assert ls.label is None

    poly = ia.Polygon([(0, 0), (1, 1)])
    ls = poly.to_line_string(closed=True)
    assert len(ls.coords) == 3
    assert ls.coords_almost_equals([(0, 0), (1, 1), (0, 0)])
    assert ls.label is None

    poly = ia.Polygon([(0, 0), (1, 1)], label="foo")
    ls = poly.to_line_string(closed=True)
    assert len(ls.coords) == 3
    assert ls.coords_almost_equals([(0, 0), (1, 1), (0, 0)])
    assert ls.label == "foo"

    poly = ia.Polygon([(0, 0), (1, 1)], label="foo")
    ls = poly.to_line_string()
    assert len(ls.coords) == 3
    assert ls.coords_almost_equals([(0, 0), (1, 1), (0, 0)])
    assert ls.label == "foo"


def test_Polygon_from_shapely():
    exterior = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly_shapely = shapely.geometry.Polygon(exterior)
    poly = ia.Polygon.from_shapely(poly_shapely)

    # shapely messes up the point ordering, so we try to correct it here
    start_idx = 0
    for i, (x, y) in enumerate(poly.exterior):
        dist = np.sqrt((exterior[0][0] - x) ** 2 + (exterior[0][1] - x) ** 2)
        if dist < 1e-4:
            start_idx = i
            break
    poly = poly.change_first_point_by_index(start_idx)

    for (x_exp, y_exp), (x_obs, y_obs) in zip(exterior, poly.exterior):
        assert x_exp - 1e-8 < x_obs < x_exp + 1e-8
        assert y_exp - 1e-8 < y_obs < y_exp + 1e-8

    # empty polygon
    poly_shapely = shapely.geometry.Polygon([])
    poly = ia.Polygon.from_shapely(poly_shapely)
    assert len(poly.exterior) == 0


def test_Polygon_copy():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
    poly_cp = poly.copy()
    assert poly.exterior.dtype.type == poly_cp.exterior.dtype.type
    assert poly.exterior.shape == poly_cp.exterior.shape
    assert np.allclose(poly.exterior, poly_cp.exterior)
    assert poly.label == poly_cp.label


def test_Polygon_deepcopy():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
    poly_cp = poly.deepcopy()
    assert poly.exterior.dtype.type == poly_cp.exterior.dtype.type
    assert poly.exterior.shape == poly_cp.exterior.shape
    assert np.allclose(poly.exterior, poly_cp.exterior)
    assert poly.label == poly_cp.label

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
    poly_cp = poly.deepcopy()
    poly_cp.exterior[0, 0] = 100.0
    poly_cp.label = "test2"
    assert poly.exterior.dtype.type == poly_cp.exterior.dtype.type
    assert poly.exterior.shape == poly_cp.exterior.shape
    assert not np.allclose(poly.exterior, poly_cp.exterior)
    assert not poly.label == poly_cp.label


def test_Polygon___repr__():
    _test_Polygon_repr_str(lambda poly: poly.__repr__())


def test_Polygon___str__():
    _test_Polygon_repr_str(lambda poly: poly.__str__())


def _test_Polygon_repr_str(func):
    # ints
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
    s = func(poly)
    assert s == "Polygon([(x=0.000, y=0.000), (x=1.000, y=0.000), (x=1.000, y=1.000), (x=0.000, y=1.000)] " \
                + "(4 points), label=test)"

    # floats
    poly = ia.Polygon([(0, 0.5), (1.5, 0), (1, 1), (0, 1)], label="test")
    s = func(poly)
    assert s == "Polygon([(x=0.000, y=0.500), (x=1.500, y=0.000), (x=1.000, y=1.000), (x=0.000, y=1.000)] " \
                + "(4 points), label=test)"

    # label None
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label=None)
    s = func(poly)
    assert s == "Polygon([(x=0.000, y=0.000), (x=1.000, y=0.000), (x=1.000, y=1.000), (x=0.000, y=1.000)] " \
                + "(4 points), label=None)"

    # no points
    poly = ia.Polygon([], label="test")
    s = func(poly)
    assert s == "Polygon([] (0 points), label=test)"


def test_Polygon_exterior_almost_equals():
    # exactly same exterior
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly_a.exterior_almost_equals(poly_b)

    # one point duplicated
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (1, 1), (1, 1), (0, 1)])
    assert poly_a.exterior_almost_equals(poly_b)

    # several points added without changing geometry
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1), (0, 0.5)])
    assert poly_a.exterior_almost_equals(poly_b)

    # different order
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0, 1), (1, 1), (1, 0), (0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    # tiny shift below tolerance
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0+1e-6, 0), (1+1e-6, 0), (1+1e-6, 1), (0+1e-6, 1)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-3)

    # tiny shift above tolerance
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0+1e-6, 0), (1+1e-6, 0), (1+1e-6, 1), (0+1e-6, 1)])
    assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-9)

    # shifted polygon towards half overlap
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
    assert not poly_a.exterior_almost_equals(poly_b)

    # shifted polygon towards no overlap at all
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(100, 0), (101, 0), (101, 1), (100, 1)])
    assert not poly_a.exterior_almost_equals(poly_b)

    # both polygons without points
    poly_a = ia.Polygon([])
    poly_b = ia.Polygon([])
    assert poly_a.exterior_almost_equals(poly_b)

    # both polygons with one point
    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(100, 100)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0+1e-6, 0)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0+1, 0)])
    assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    # both polygons with two points
    poly_a = ia.Polygon([(0, 0), (1, 0)])
    poly_b = ia.Polygon([(0, 0), (1, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0)])
    poly_b = ia.Polygon([(0, 0), (2, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0)])
    poly_b = ia.Polygon([(0+1e-6, 0), (1+1e-6, 0)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    # both polygons with three points
    poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    poly_b = ia.Polygon([(0, 0), (1, -1), (0.5, 1)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    poly_b = ia.Polygon([(0, 0), (1+1e-6, 0), (0.5, 1)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    # one polygon with zero points, other with one
    poly_a = ia.Polygon([])
    poly_b = ia.Polygon([(0, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([])
    assert not poly_a.exterior_almost_equals(poly_b)

    # one polygon with one point, other with two
    poly_a = ia.Polygon([(-10, -20)])
    poly_b = ia.Polygon([(0, 0), (1, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (1, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (0, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (0+1e-6, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    poly_a = ia.Polygon([(0, 0), (0+1e-4, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-9)

    # one polygon with one point, other with three
    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    poly_b = ia.Polygon([(0, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0), (0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0), (1, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (0, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0+1e-6, 0), (0, 0+1e-6)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0+1e-4, 0), (0, 0+1e-4)])
    assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-9)


def test_Polygon_almost_equals():
    poly_a = ia.Polygon([])
    poly_b = ia.Polygon([])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0)])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0), (0, 0)])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0+1e-10, 0)])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)], label="test")
    poly_b = ia.Polygon([(0, 0)])
    assert not poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0)], label="test")
    assert not poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)], label="test")
    poly_b = ia.Polygon([(0, 0)], label="test")
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)], label="test")
    poly_b = ia.Polygon([(1, 0)], label="test")
    assert not poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)], label="testA")
    poly_b = ia.Polygon([(0, 0)], label="testB")
    assert not poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    assert not poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    assert not poly_a.almost_equals("foo")


def test___convert_points_to_shapely_line_string():
    # TODO this function seems to already be covered completely by other tests, so add a proper test later
    pass





class TestPolygonsOnImage(unittest.TestCase):
    def setUp(self):
        reseed()

    def test__init__(self):
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

        # list of polygons is empty
        poly_oi = ia.PolygonsOnImage(
            [],
            shape=(10, 10, 3)
        )
        assert len(poly_oi.polygons) == 0

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

        # shape given as numpy array
        poly_oi = ia.PolygonsOnImage(
            [],
            shape=np.zeros((10, 10, 3), dtype=np.uint8)
        )
        assert poly_oi.shape == (10, 10, 3)

        # 2D shape
        poly_oi = ia.PolygonsOnImage(
            [],
            shape=(10, 11)
        )
        assert poly_oi.shape == (10, 11)

    def test_empty(self):
        # standard case with multiple polygons
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             ia.Polygon([(0, 0), (1, 0), (1, 1)]),
             ia.Polygon([(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)])],
            shape=(10, 10, 3)
        )
        assert poly_oi.empty is False

        # list of polygons is empty
        poly_oi = ia.PolygonsOnImage([], shape=(10, 10, 3))
        assert poly_oi.empty is True

    def test_on(self):
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

    def test_draw_on_image(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        # no polygons, nothing changed
        poly_oi = ia.PolygonsOnImage([], shape=image.shape)
        image_drawn = poly_oi.draw_on_image(image)
        assert np.sum(image) == 0
        assert np.sum(image_drawn) == 0

        # draw two polygons
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

    def test_remove_out_of_image(self):
        # no polygons, nothing to remove
        poly_oi = ia.PolygonsOnImage([], shape=(10, 11, 3))
        for fully, partly in [(False, False), (False, True),
                              (True, False), (True, True)]:
            poly_oi_rm = poly_oi.remove_out_of_image(fully=fully, partly=partly)
            assert len(poly_oi_rm.polygons) == 0
            assert poly_oi_rm.shape == (10, 11, 3)

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

    def test_clip_out_of_image(self):
        # NOTE: clip_out_of_image() can change the order of points,
        # hence we check here for each expected point whether it appears
        # somewhere in the list of points

        def _any_point_close(points, point_search):
            found = False
            for point in points:
                if np.allclose(point, point_search, atol=1e-4, rtol=0):
                    found = True
            return found

        # no polygons
        poly_oi = ia.PolygonsOnImage([], shape=(10, 11, 3))
        poly_oi_clip = poly_oi.clip_out_of_image()
        assert len(poly_oi_clip.polygons) == 0
        assert poly_oi_clip.shape == (10, 11, 3)

        # one polygon, fully inside
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (8, 1), (8, 9), (1, 9)])],
            shape=(10, 11, 3))
        poly_oi_clip = poly_oi.clip_out_of_image()
        assert len(poly_oi_clip.polygons) == 1
        for point_search in [(1, 1), (8, 1), (8, 9), (1, 9)]:
            assert _any_point_close(poly_oi_clip.polygons[0].exterior,
                                    point_search)
        assert poly_oi_clip.shape == (10, 11, 3)

        # one polygon, partially outside
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (15, 1), (15, 9), (1, 9)])],
            shape=(10, 11, 3))
        poly_oi_clip = poly_oi.clip_out_of_image()
        assert len(poly_oi_clip.polygons) == 1
        for point_search in [(1, 1), (11, 1), (11, 9), (1, 9)]:
            assert _any_point_close(poly_oi_clip.polygons[0].exterior,
                                    point_search)
        assert poly_oi_clip.shape == (10, 11, 3)

        # one polygon, fully outside
        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])],
            shape=(10, 11, 3))
        poly_oi_clip = poly_oi.clip_out_of_image()
        assert len(poly_oi_clip.polygons) == 0
        assert poly_oi_clip.shape == (10, 11, 3)

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
            assert _any_point_close(poly_oi_clip.polygons[0].exterior,
                                    point_search)
        for point_search in [(1, 1), (11, 1), (11, 9), (1, 9)]:
            assert _any_point_close(poly_oi_clip.polygons[1].exterior,
                                    point_search)
        assert poly_oi_clip.shape == (10, 11, 3)

    def test_shift(self):
        # no polygons
        poly_oi = ia.PolygonsOnImage([], shape=(10, 11, 3))
        poly_oi_shifted = poly_oi.shift(top=3, right=0, bottom=1, left=-3)
        assert len(poly_oi_shifted.polygons) == 0
        assert poly_oi_shifted.shape == (10, 11, 3)

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

    def test_copy(self):
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

        poly_oi_copy.polygons = [ia.Polygon([(0, 0), (1, 0), (1, 1)])]
        assert np.allclose(poly_oi.polygons[0].exterior,
                           [(1, 1), (8, 1), (8, 9), (1, 9)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_copy.polygons[0].exterior,
                           [(0, 0), (1, 0), (1, 1)],
                           rtol=0, atol=1e-4)

        poly_oi_copy.shape = (20, 30, 3)
        assert poly_oi.shape == (10, 11, 3)
        assert poly_oi_copy.shape == (20, 30, 3)

    def test_deepcopy(self):
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

        poly_oi_copy.polygons[0] = ia.Polygon([(0, 0), (1, 0), (1, 1)])
        assert np.allclose(poly_oi.polygons[0].exterior,
                           [(1, 1), (8, 1), (8, 9), (1, 9)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_copy.polygons[0].exterior,
                           [(0, 0), (1, 0), (1, 1)],
                           rtol=0, atol=1e-4)

        poly_oi_copy.polygons[1].exterior[0][0] = 100
        assert np.allclose(poly_oi.polygons[1].exterior,
                           [(2, 2), (16, 2), (16, 10), (2, 10)],
                           rtol=0, atol=1e-4)
        assert np.allclose(poly_oi_copy.polygons[1].exterior,
                           [(100, 2), (16, 2), (16, 10), (2, 10)],
                           rtol=0, atol=1e-4)

        poly_oi_copy.shape = (20, 30, 3)
        assert poly_oi.shape == (10, 11, 3)
        assert poly_oi_copy.shape == (20, 30, 3)

    def test__repr__(self):
        poly_oi = ia.PolygonsOnImage([], shape=(10, 11, 3))
        assert poly_oi.__repr__() == "PolygonsOnImage([], shape=(10, 11, 3))"

        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (8, 1), (8, 9), (1, 9)]),
             ia.Polygon([(2, 2), (16, 2), (16, 10), (2, 10)])],
            shape=(10, 11, 3))
        assert poly_oi.__repr__() == (
            "PolygonsOnImage(["
            + "Polygon([(x=1.000, y=1.000), (x=8.000, y=1.000), "
            + "(x=8.000, y=9.000), (x=1.000, y=9.000)] "
            + "(4 points), label=None), "
            + "Polygon([(x=2.000, y=2.000), (x=16.000, y=2.000), "
            + "(x=16.000, y=10.000), (x=2.000, y=10.000)] "
            + "(4 points), label=None)"
            + "], shape=(10, 11, 3))")

    def test__str__(self):
        poly_oi = ia.PolygonsOnImage([], shape=(10, 11, 3))
        assert poly_oi.__repr__() == "PolygonsOnImage([], shape=(10, 11, 3))"

        poly_oi = ia.PolygonsOnImage(
            [ia.Polygon([(1, 1), (8, 1), (8, 9), (1, 9)]),
             ia.Polygon([(2, 2), (16, 2), (16, 10), (2, 10)])],
            shape=(10, 11, 3))
        assert poly_oi.__repr__() == (
            "PolygonsOnImage(["
            + "Polygon([(x=1.000, y=1.000), (x=8.000, y=1.000), "
            + "(x=8.000, y=9.000), (x=1.000, y=9.000)] "
            + "(4 points), label=None), "
            + "Polygon([(x=2.000, y=2.000), (x=16.000, y=2.000), "
            + "(x=16.000, y=10.000), (x=2.000, y=10.000)] "
            + "(4 points), label=None)"
            + "], shape=(10, 11, 3))")


class Test_ConcavePolygonRecoverer(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _assert_points_are_identical(cls, observed, expected, atol=1e-8, rtol=0):
        assert len(observed) == len(expected)
        for i, (ps_obs, ps_exp) in enumerate(zip(observed, expected)):
            assert len(ps_obs) == len(ps_exp), "Failed at point %d" % (i,)
            for p_obs, p_exp in zip(ps_obs, ps_exp):
                assert len(p_obs) == 2
                assert len(p_exp) == 2
                assert np.allclose(p_obs, p_exp, atol=atol, rtol=rtol), "Unexpected coords at %d" % (i,)

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

    def test_recover_from_predefined_polygons(self):
        cpr = _ConcavePolygonRecoverer()

        # concave input
        polys = [
            [(0, 0), (1, 0), (1, 1)],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            [(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1.0), (0, 1)],
        ]

        for poly in polys:
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
                    if dist < 0.01:
                        found[i] = True
            assert all(found)

        # line
        poly = [(0, 0), (1, 0), (2, 0)]
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
        assert all(found)

        # duplicate points
        poly = [(0, 0), (1, 0), (1, 0), (1, 1)]
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
                if dist < 0.01:
                    found[i] = True
        assert all(found)

        # other broken poly
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
        assert all(found)

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
            bb = ia.BoundingBox(x1=bb_x1-1e-4, y1=bb_y1-1e-4, x2=bb_x2+1e-4, y2=bb_y2+1e-4)
            for point in points:
                assert bb.contains(ia.Keypoint(x=point[0], y=point[1]))

    def test__remove_consecutive_duplicate_points(self):
        recoverer = _ConcavePolygonRecoverer()
        points = [(0, 0), (1, 1)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            points
        )

        points = [(0.0, 0.5), (1.0, 1.0)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            np.float32(points)
        )

        points = np.float32([(0.0, 0.5), (1.0, 1.0)])
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            np.float32(points)
        )

        points = [(0, 0), (0, 0)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            [(0, 0)],
            atol=1e-8, rtol=0
        )

        points = [(0, 0), (0, 0), (1, 0)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            [(0, 0), (1, 0)],
            atol=1e-8, rtol=0
        )

        points = [(0, 0), (1, 0), (1, 0)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            [(0, 0), (1, 0)],
            atol=1e-8, rtol=0
        )

        points = [(0, 0), (1, 0), (1, 0), (2, 0), (0, 0)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            [(0, 0), (1, 0), (2, 0)],
            atol=1e-8, rtol=0
        )

    def test__jitter_duplicate_points(self):
        cpr = _ConcavePolygonRecoverer(threshold_duplicate_points=1e-4)
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(points, points_jittered, rtol=0, atol=1e-4)

        points = [(0, 0), (1, 0), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(points, points_jittered, rtol=0, atol=1e-4)

        points = [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(points, points_jittered, rtol=0, atol=1e-4)

        points = [(0, 0), (1, 0), (1 + 1e-6, 0), (1, 1), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(
            [point for i, point in enumerate(points_jittered) if i in [0, 1, 3, 4]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[2])) >= 1e-4

        points = [(0, 0), (1, 0), (1, 1), (1 + 1e-6, 0), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(
            [point for i, point in enumerate(points_jittered) if i in [0, 1, 2, 4]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[3])) >= 1e-4

        points = [(0, 0), (1, 0), (1, 1), (0, 1), (1 + 1e-6, 0)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(
            [point for i, point in enumerate(points_jittered) if i in [0, 1, 2, 3]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[4])) >= 1e-4

        points = [(0, 0), (1, 0), (1 + 1e-6, 0), (1, 1), (1 + 1e-6, 0), (0, 1),
                  (1 + 1e-6, 0), (1 + 1e-6, 0 + 1e-6), (1 + 1e-6, 0 + 2e-6)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(
            [point for i, point in enumerate(points_jittered) if i in [0, 1, 3, 5]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[2])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[4])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[6])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[7])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[8])) >= 1e-4

        points = [(0, 0), (1, 0), (0 + 1e-6, 0 - 1e-6), (1 + 1e-6, 0), (1, 1),
                  (1 + 1e-6, 0), (0, 1), (1 + 1e-6, 0), (1 + 1e-6, 0 + 1e-6),
                  (1 + 1e-6, 0 + 2e-6)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(
            [point for i, point in enumerate(points_jittered) if i in [0, 1, 4, 6]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert np.linalg.norm(np.float32([0, 0]) - np.float32(points_jittered[2])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[3])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[5])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[7])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[8])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[9])) >= 1e-4

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

    def test__fit_best_valid_polygon(self):
        def _assert_ids_match(observed, expected):
            assert len(observed) == len(expected), "len mismatch: %d vs %d" % (len(observed), len(expected))

            max_count = 0
            for i in range(len(observed)):
                counter = 0
                for j in range(i, i+len(expected)):
                    if observed[(i+j) % len(observed)] == expected[j % len(expected)]:
                        counter += 1
                    else:
                        break

                max_count = max(max_count, counter)

            assert max_count == len(expected), "count mismatch: %d vs %d" % (max_count, len(expected))

        cpr = _ConcavePolygonRecoverer()
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(points, random_state=np.random.RandomState(0))
        # doing this without the list(.) wrappers fails on python2.7
        assert list(points_fit) == list(sm.xrange(len(points)))

        # square-like, but top line has one point in its center which's
        # y-coordinate is below the bottom line
        points = [(0.0, 0.0), (0.45, 0.0), (0.5, 1.5), (0.55, 0.0), (1.0, 0.0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(points, random_state=np.random.RandomState(0))
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
        points_fit = cpr._fit_best_valid_polygon(points, random_state=np.random.RandomState(0))
        _assert_ids_match(points_fit, [0, 1, 4, 5, 6, 3, 2, 7])
        poly_observed = ia.Polygon([points[idx] for idx in points_fit])
        assert poly_observed.is_valid

        # same as above, but intersection points at the bottom line are provided
        # without oversampling, i.e. incorporating these points would lead to an
        # invalid polygon
        points = [(0.0, 0), (0.25, 0), (0.25, 1.0), (0.25, 1.25),
                  (0.75, 1.25), (0.75, 1.0), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(points, random_state=np.random.RandomState(0))
        assert len(points_fit) >= len(points) - 2  # TODO add IoU check here
        poly_observed = ia.Polygon([points[idx] for idx in points_fit])
        assert poly_observed.is_valid

    def test__fix_polygon_is_line(self):
        cpr = _ConcavePolygonRecoverer()

        points = [(0, 0), (1, 0), (1, 1)]
        points_fixed = cpr._fix_polygon_is_line(points, np.random.RandomState(0))
        assert np.allclose(points_fixed, points, atol=0, rtol=0)

        points = [(0, 0), (1, 0), (2, 0)]
        points_fixed = cpr._fix_polygon_is_line(points, np.random.RandomState(0))
        assert not np.allclose(points_fixed, points, atol=0, rtol=0)
        assert not cpr._is_polygon_line(points_fixed)
        assert np.allclose(points_fixed, points, rtol=0, atol=1e-2)

        points = [(0, 0), (0, 1), (0, 2)]
        points_fixed = cpr._fix_polygon_is_line(points, np.random.RandomState(0))
        assert not np.allclose(points_fixed, points, atol=0, rtol=0)
        assert not cpr._is_polygon_line(points_fixed)
        assert np.allclose(points_fixed, points, rtol=0, atol=1e-2)

        points = [(0, 0), (1, 1), (2, 2)]
        points_fixed = cpr._fix_polygon_is_line(points, np.random.RandomState(0))
        assert not np.allclose(points_fixed, points, atol=0, rtol=0)
        assert not cpr._is_polygon_line(points_fixed)
        assert np.allclose(points_fixed, points, rtol=0, atol=1e-2)

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

    def test__generate_intersection_points(self):
        cpr = _ConcavePolygonRecoverer()

        # triangle
        points = [(0.5, 0), (1, 1), (0, 1)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        assert points_inter == [[], [], []]

        # rotated square
        points = [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        assert points_inter == [[], [], [], []]

        # square
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        assert points_inter == [[], [], [], []]

        # |--|  |--|
        # |  |__|  |
        # |        |
        # |--------|
        points = [(0.0, 0), (0.25, 0), (0.25, 0.25),
                  (0.75, 0.25), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        assert points_inter == [[], [], [], [], [], [], [], []]

        # same as above, but middle part goes much further down,
        # crossing the bottom line
        points = [(0.0, 0), (0.25, 0), (0.25, 1.25),
                  (0.75, 1.25), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [(0.25, 1.0)], [], [(0.75, 1.0)], [], [], [(0.75, 1.0), (0.25, 1.0)], []])

        # square-like structure with intersections in top right area
        points = [(0, 0), (0.5, 0), (1.01, 0.5), (1.0, 0), (1, 1), (0, 1), (0, 0)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [(1.0, 0.4902)], [], [(1.0, 0.4902)], [], [], []],
            atol=1e-2)

        # same as above, but with a second intersection in bottom left
        points = [(0, 0), (0.5, 0), (1.01, 0.5), (1.0, 0), (1, 1), (-0.25, 1),
                  (0, 1.25)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [(1.0, 0.4902)], [], [(1.0, 0.4902)], [(0, 1.0)], [], [(0, 1.0)]],
            atol=1e-2)

        # double triangle with point in center that is shared by both triangles
        points = [(0, 0), (0.5, 0.5), (1.0, 0), (1.0, 1.0), (0.5, 0.5), (0, 1.0)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [], [], [], [], []])

    def test__oversample_intersection_points(self):
        cpr = _ConcavePolygonRecoverer()
        cpr.oversampling = 0.1

        points = [(0.0, 0.0), (1.0, 0.0)]
        segment_add_points_sorted = [[(0.5, 0.0)], []]
        points_oversampled = cpr._oversample_intersection_points(points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.55, 0.0)], []],
            atol=1e-4
        )

        points = [(0.0, 0.0), (2.0, 0.0)]
        segment_add_points_sorted = [[(0.5, 0.0)], []]
        points_oversampled = cpr._oversample_intersection_points(points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.65, 0.0)], []],
            atol=1e-4
        )

        points = [(0.0, 0.0), (1.0, 0.0)]
        segment_add_points_sorted = [[(0.5, 0.0), (0.6, 0.0)], []]
        points_oversampled = cpr._oversample_intersection_points(points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.51, 0.0), (0.59, 0.0), (0.6, 0.0), (0.64, 0.0)], []],
            atol=1e-4
        )

        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        segment_add_points_sorted = [[(0.5, 0.0)], [], [(0.8, 1.0)], [(0.0, 0.7)]]
        points_oversampled = cpr._oversample_intersection_points(points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.55, 0.0)],
             [],
             [(0.82, 1.0), (0.8, 1.0), (0.72, 1.0)],
             [(0.0, 0.73), (0.0, 0.7), (0.0, 0.63)]],
            atol=1e-4
        )

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
