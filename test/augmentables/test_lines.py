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


class TestLineString(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        ls = LineString(np.float32([[0, 0], [1, 2]]))
        assert np.allclose(ls.coords, np.float32([[0, 0], [1, 2]]))
        assert ls.label is None

        ls = LineString([(0, 0), (1, 2)])
        assert np.allclose(ls.coords, np.float32([[0, 0], [1, 2]]))
        assert ls.label is None

        ls = LineString([])
        assert ls.coords.shape == (0, 2)
        assert ls.label is None

        ls = LineString([], label="test")
        assert ls.coords.shape == (0, 2)
        assert ls.label == "test"

    def test_length(self):
        ls = LineString(np.float32([[0, 0], [1, 0], [1, 1]]))
        assert np.isclose(ls.length, 2.0)

        ls = LineString(np.float32([[0, 0], [1, 2], [4, 5]]))
        assert np.isclose(ls.length,
                          np.sqrt(1**2, 2**2) + np.sqrt(3**2 + 3**2))

        ls = LineString([(0, 0)])
        assert np.isclose(ls.length, 0.0)

        ls = LineString([])
        assert np.isclose(ls.length, 0.0)

    def test_xx(self):
        ls = LineString(np.float32([[0, 0], [1, 0], [2, 1]]))
        assert np.allclose(ls.xx, np.float32([0, 1, 2]))

        ls = LineString([])
        assert np.allclose(ls.xx, np.zeros((0,), dtype=np.float32))

    def test_yy(self):
        ls = LineString(np.float32([[0, 0], [0, 1], [0, 2]]))
        assert np.allclose(ls.yy, np.float32([0, 1, 2]))

        ls = LineString([])
        assert np.allclose(ls.yy, np.zeros((0,), dtype=np.float32))

    def test_xx_int(self):
        ls = LineString(np.float32([[0, 0], [1.4, 0], [2.6, 1]]))
        assert ls.xx_int.dtype.name == "int32"
        assert np.array_equal(ls.xx_int, np.int32([0, 1, 3]))

        ls = LineString([])
        assert ls.xx_int.dtype.name == "int32"
        assert np.array_equal(ls.xx_int, np.zeros((0,), dtype=np.int32))

    def test_yy_int(self):
        ls = LineString(np.float32([[0, 0], [0, 1.4], [1, 2.6]]))
        assert ls.yy_int.dtype.name == "int32"
        assert np.array_equal(ls.yy_int, np.int32([0, 1, 3]))

        ls = LineString([])
        assert ls.yy_int.dtype.name == "int32"
        assert np.array_equal(ls.yy_int, np.zeros((0,), dtype=np.int32))

    def test_height(self):
        ls = LineString(np.float32([[0, 0], [0, 1.4], [1, 2.6]]))
        assert np.isclose(ls.height, 2.6)

        ls = LineString([])
        assert np.isclose(ls.height, 0.0)

    def test_width(self):
        ls = LineString(np.float32([[0, 0], [1.4, 0], [2.6, 1]]))
        assert np.isclose(ls.width, 2.6)

        ls = LineString([])
        assert np.isclose(ls.width, 0.0)

    def test_get_pointwise_distances(self):
        ls = LineString([(0, 0), (1.4, 0), (2.6, 1)])
        dists = ls.get_pointwise_distances()
        assert np.allclose(dists, [np.sqrt(1.4**2), np.sqrt(2.6**2+1**2)])

        ls = LineString([(0, 0), (1.4, 0)])
        dists = ls.get_pointwise_distances()
        assert np.allclose(dists, [np.sqrt(1.4**2)])

        ls = LineString([(0, 0)])
        dists = ls.get_pointwise_distances()
        assert dists.shape == (0,)

        ls = LineString([])
        dists = ls.get_pointwise_distances()
        assert dists.shape == (0,)

    def test_get_pointwise_inside_image_mask(self):
        ls = LineString([(0, 0), (1.4, 0), (2.6, 1)])
        mask = ls.get_pointwise_inside_image_mask((2, 2))
        assert np.array_equal(mask, [True, True, False])

        ls = LineString([(0, 0), (1.4, 0), (2.6, 1)])
        mask = ls.get_pointwise_inside_image_mask(
            np.zeros((2, 2), dtype=np.uint8))
        assert np.array_equal(mask, [True, True, False])

        ls = LineString([(0, 0)])
        mask = ls.get_pointwise_inside_image_mask((2, 2))
        assert np.array_equal(mask, [True])

        ls = LineString([])
        mask = ls.get_pointwise_inside_image_mask((2, 2))
        assert mask.shape == (0,)

    def test_compute_distance(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert np.isclose(ls.compute_distance((0, 0)), 0)
        assert np.isclose(ls.compute_distance((1, 0)), 0)
        assert np.isclose(ls.compute_distance((0, 1)), 1)
        assert np.isclose(ls.compute_distance((-0.5, -0.6)),
                          np.sqrt(0.5**2 + 0.6**2))

        ls = LineString([(0, 0)])
        assert np.isclose(ls.compute_distance((0, 0)), 0)
        assert np.isclose(ls.compute_distance((-0.5, -0.6)),
                          np.sqrt(0.5**2 + 0.6**2))

        ls = LineString([])
        assert ls.compute_distance((0, 0)) is None
        assert ls.compute_distance((0, 0), default=-1) == -1

        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert np.isclose(ls.compute_distance(ia.Keypoint(x=0, y=1)), 1)
        assert np.isclose(ls.compute_distance(LineString([(0, 0)]), 0))
        assert np.isclose(ls.compute_distance(LineString([(0, 1)]), 1))
        assert np.isclose(ls.compute_distance(LineString([(0, 0), (0, 1)]), 0))
        assert np.isclose(ls.compute_distance(LineString([(-1, 0), (0, 1)]), 1))

        with self.assertRaises(ValueError):
            assert ls.compute_distance("foo")

    def test_contains(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert ls.contains(ia.Keypoint(x=0, y=0))
        assert ls.contains(ia.Keypoint(x=1, y=0))
        assert not ls.contains(ia.Keypoint(x=100, y=0))
        assert ls.contains((0, 0))
        assert ls.contains((1, 0))
        assert ls.contains((2, 1))
        assert ls.contains((0+1e-8, 0))
        assert not ls.contains((0+1e-8, 0), distance_threshold=0)
        assert not ls.contains((0-1, 0))
        assert ls.contains((0-1, 0), distance_threshold=2)

        ls = LineString([(0, 0)])
        assert ls.contains((0, 0))
        assert not ls.contains((1, 0))

        ls = LineString([])
        assert not ls.contains((0, 0))
        assert not ls.contains((1, 0))

    def test_project(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        ls_proj = ls.project((10, 10), (20, 20))
        assert np.allclose(ls_proj.coords, [(0, 0), (2, 0), (4, 2)])

        ls = LineString([(0, 0), (1, 0), (2, 1)])
        ls_proj = ls.project((10, 10), (10, 20))
        assert np.allclose(ls_proj.coords, [(0, 0), (2, 0), (4, 1)])

        ls = LineString([(0, 0), (1, 0), (2, 1)])
        ls_proj = ls.project((10, 10), (10, 20))
        assert np.allclose(ls_proj.coords, [(0, 0), (1, 0), (2, 2)])

        ls = LineString([])
        ls_proj = ls.project((10, 10), (20, 20))
        assert ls_proj.coords.shape == (0, 2)

    def test_is_fully_within_image(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert ls.is_fully_within_image((10, 10))
        assert ls.is_fully_within_image((2, 3))
        assert not ls.is_fully_within_image((2, 2))
        assert not ls.is_fully_within_image((1, 1))

        ls = LineString([(-1, 0), (1, 0), (2, 1)])
        assert not ls.is_fully_within_image((10, 10))
        assert not ls.is_fully_within_image((2, 3))
        assert not ls.is_fully_within_image((2, 2))
        assert not ls.is_fully_within_image((1, 1))

        ls = LineString([(0, 0)])
        assert ls.is_fully_within_image((10, 10))
        assert ls.is_fully_within_image((2, 3))
        assert ls.is_fully_within_image((2, 2))
        assert ls.is_fully_within_image((1, 1))

        ls = LineString([])
        assert not ls.is_fully_within_image((10, 10))
        assert not ls.is_fully_within_image((2, 3))
        assert not ls.is_fully_within_image((2, 2))
        assert not ls.is_fully_within_image((1, 1))
        assert ls.is_fully_within_image((10, 10), default=True)
        assert ls.is_fully_within_image((2, 3), default=True)
        assert ls.is_fully_within_image((2, 2), default=True)
        assert ls.is_fully_within_image((1, 1), default=True)
        assert ls.is_fully_within_image((10, 10), default=None) is None

    def test_is_partly_within_image(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert ls.is_partly_within_image((10, 10))
        assert ls.is_partly_within_image((2, 3))
        assert ls.is_partly_within_image((2, 2))
        assert ls.is_partly_within_image((1, 1))

        ls = LineString([(1, 0), (2, 0), (3, 1)])
        assert ls.is_partly_within_image((10, 10))
        assert ls.is_partly_within_image((2, 3))
        assert ls.is_partly_within_image((2, 2))
        assert not ls.is_partly_within_image((1, 1))

        # line string that cuts through the middle of the image,
        # with both points outside of a BB (0, 0), (10, 10)
        ls = LineString([(-1, 5), (11, 5)])
        assert ls.is_partly_within_image((100, 100))
        assert ls.is_partly_within_image((10, 12))
        assert ls.is_partly_within_image((10, 10))
        assert ls.is_partly_within_image((1, 1))

        # line string around inner rectangle of (-1, -1), (11, 11)
        ls = LineString([(-1, -1), (11, -1), (11, 11), (-1, 11)])
        assert ls.is_partly_within_image((100, 100))
        assert ls.is_partly_within_image((12, 12))
        assert not ls.is_partly_within_image((10, 10))

        # just one point
        ls = LineString([(11, 11)])
        assert ls.is_partly_within_image((100, 100))
        assert ls.is_partly_within_image((12, 12))
        assert not ls.is_partly_within_image((10, 10))

        # no point
        ls = LineString([])
        assert not ls.is_partly_within_image((100, 100))
        assert not ls.is_partly_within_image((10, 10))
        assert ls.is_partly_within_image((100, 100), default=True)
        assert ls.is_partly_within_image((10, 10), default=True)
        assert ls.is_partly_within_image((100, 100), default=None) is None
        assert ls.is_partly_within_image((10, 10), default=None) is None

    def test_is_out_image(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert not ls.is_out_of_image((10, 10))
        assert ls.is_out_of_image((1, 1), fully=False, partly=True)
        assert not ls.is_out_of_image((1, 1), fully=True, partly=False)
        assert ls.is_out_of_image((1, 1), fully=True, partly=True)
        assert not ls.is_out_of_image((1, 1), fully=False, partly=False)

        ls = LineString([])
        assert ls.is_out_of_image((10, 10))
        assert not ls.is_out_of_image((10, 10), default=True)
        assert not ls.is_out_of_image((10, 10), default=None) is None

    def test_clip_out_image(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert ls.clip_out_of_image((10, 10)).coords_almost_equals(ls)
        assert ls.clip_out_of_image((2, 3)).coords_almost_equals(ls)
        assert ls.clip_out_of_image((2, 2)).coords_almost_equals(ls)
        assert ls.clip_out_of_image((1, 1)).coords_almost_equals([(0, 0), (1, 0)])

        ls = LineString([(1, 0), (2, 0), (3, 1)])
        assert ls.clip_out_of_image((10, 10)).coords_almost_equals(ls)
        assert ls.clip_out_of_image((2, 3)).coords_almost_equals(ls)
        assert ls.clip_out_of_image((2, 2)).coords_almost_equals([(1, 0), (2, 0)])
        assert ls.clip_out_of_image((1, 1)).coords_almost_equals([(1, 0)])

        # line string that cuts through the middle of the image,
        # with both points outside of a BB (0, 0), (10, 10)
        ls = LineString([(-1, 5), (11, 5)])
        assert ls.clip_out_of_image((100, 100)).coords_almost_equals([(0, 5), (11, 5)])
        assert ls.clip_out_of_image((10, 12)).coords_almost_equals([(0, 5), (11, 5)])
        assert ls.clip_out_of_image((10, 10)).coords_almost_equals([(0, 5), (10, 5)])
        assert ls.clip_out_of_image((1, 1)).coords_almost_equals([(0, 5), (1, 5)])

        # line string around inner rectangle of (-1, -1), (11, 11)
        ls = LineString([(-1, -1), (11, -1), (11, 11), (-1, 11)])
        assert ls.clip_out_of_image((100, 100)).coords_almost_equals([(11, 0), (11, 11)])
        assert ls.clip_out_of_image((12, 12)).coords_almost_equals([(11, 0), (11, 11)])
        assert ls.clip_out_of_image((10, 10)).coords_almost_equals([])

        # just one point
        ls = LineString([(11, 11)])
        assert ls.clip_out_of_image((100, 100)).coords_almost_equals([(11, 11)])
        assert ls.clip_out_of_image((12, 12)).coords_almost_equals([(11, 11)])
        assert not ls.clip_out_of_image((10, 10)).coords_almost_equals([])

        # no point
        ls = LineString([])
        assert ls.clip_out_of_image((100, 100)).coords_almost_equals([])
        assert ls.clip_out_of_image((10, 10)).coords_almost_equals([])

    def test_shift(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert ls.shift(top=1).coords_almost_equals([(0, 1), (1, 1), (2, 2)])
        assert ls.shift(right=1).coords_almost_equals([(-1, 0), (0, 0), (1, 1)])
        assert ls.shift(bottom=1).coords_almost_equals([(0, -1), (1, -1), (2, 0)])
        assert ls.shift(left=1).coords_almost_equals([(1, 0), (2, 0), (3, 1)])

        assert ls.shift(top=-1).coords_almost_equals([(0, -1), (1, -1), (2, 0)])
        assert ls.shift(right=-1).coords_almost_equals([(1, 0), (2, 0), (3, 1)])
        assert ls.shift(bottom=-1).coords_almost_equals([(0, 1), (1, 1), (2, 2)])
        assert ls.shift(left=-1).coords_almost_equals([(-1, 0), (0, 0), (1, 1)])

        assert ls.shift(top=1, right=2, bottom=3, left=4).coords_almost_equals(
            [(0-2+4, 0+1-3), (1-2+4, 0+1-3), (2-2+4, 1+1-3)])

        ls = LineString([])
        assert ls.shift(top=1, right=2, bottom=3, left=4).coords_almost_equals(
            [])

    def test_draw_mask(self):
        pass  # TODO

    def test_draw_line_heatmap_array(self):
        pass  # TODO

    def test_draw_points_heatmap_array(self):
        pass  # TODO

    def test_draw_heatmap_array(self):
        pass  # TODO

    def test_draw_line_on_image(self):
        pass  # TODO

    def test_draw_points_on_image(self):
        pass  # TODO

    def test_draw_on_image(self):
        pass  # TODO

    def test_extract_from_image(self):
        pass  # TODO

    def test_concat(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert ls.concat(ls).coords_almost_equals([
            (0, 0), (1, 0), (2, 1), (0, 0), (1, 0), (2, 1)
        ])

        ls = LineString([])
        assert ls.concat(ls).coords_almost_equals([])

        ls = LineString([])
        assert ls.concat(LineString([(0, 0)])).coords_almost_equals([(0, 0)])

        ls = LineString([(0, 0)])
        assert ls.concat(LineString([])).coords_almost_equals([(0, 0)])

        ls = LineString([])
        assert ls.concat([(0, 0)]).coords_almost_equals([(0, 0)])

    def test_to_keypoints(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        observed = ls.to_keypoints()
        assert all([isinstance(kp, ia.Keypoint) for kp in observed])
        assert np.isclose(observed.keypoints[0].x, 0)
        assert np.isclose(observed.keypoints[0].y, 0)
        assert np.isclose(observed.keypoints[1].x, 1)
        assert np.isclose(observed.keypoints[1].y, 0)
        assert np.isclose(observed.keypoints[2].x, 2)
        assert np.isclose(observed.keypoints[2].y, 1)

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

        ls = LineString([(0, 0)])
        observed = ls.to_bounding_box()
        assert isinstance(observed, ia.BoundingBox)
        assert np.isclose(observed.x1, 0)
        assert np.isclose(observed.y1, 0)
        assert np.isclose(observed.x2, 0)
        assert np.isclose(observed.y2, 0)

        ls = LineString([])
        observed = ls.to_bounding_box()
        assert observed is None

    def test_to_polygon(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        observed = ls.to_polygon()
        assert isinstance(observed, ia.Polygon)
        assert np.allclose(observed, [(0, 0), (1, 0), (1, 1)])

        ls = LineString([(0, 0)])
        observed = ls.to_polygon()
        assert isinstance(observed, ia.Polygon)
        assert np.allclose(observed, [(0, 0)])

        ls = LineString([])
        observed = ls.to_polygon()
        assert observed is None

    def test_to_heatmap(self):
        pass  # TODO

    def test_segmentation_map(self):
        pass  # TODO

    def test_coords_almost_equals(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        assert ls.coords_almost_equals(ls)
        assert ls.coords_almost_equals([(0, 0), (1, 0), (1, 1)])
        assert not ls.shift(top=1).coords_almost_equals(ls)
        assert ls.shift(top=1).coords_almost_equals(ls, distance_threshold=1.01)
        assert ls.coords_almost_equals([(0, 0), (0.5, 0), (1, 0), (1, 1)])

        ls = LineString([(0, 0), (0.5, 0), (1, 0), (1, 1)])
        assert ls.coords_almost_equals([(0, 0), (1, 0), (1, 1)])

        ls = LineString([(0, 0)])
        assert ls.coords_almost_equals([(0, 0)])
        assert not ls.coords_almost_equals([(0+1, 0)])
        assert ls.coords_almost_equals([(0+1, 0)], distance_threshold=1.01)

        ls = LineString([])
        assert ls.coords_almost_equals([])
        assert not ls.coords_almost_equals([(0, 0)])

        ls = LineString([(0, 0)])
        assert not ls.coords_almost_equals([])

    def test_almost_equals(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        assert ls.almost_equals(ls)
        assert not ls.shift(top=1).almost_equals(ls)
        assert ls.almost_equals(LineString([(0, 0), (1, 0), (1, 1)],
                                           label=None))
        assert not ls.almost_equals(LineString([(0, 0), (1, 0), (1, 1)],
                                               label="foo"))

        ls = LineString([(0, 0), (1, 0), (1, 1)], label="foo")
        assert not ls.almost_equals(LineString([(0, 0), (1, 0), (1, 1)],
                                               label=None))
        assert ls.almost_equals(LineString([(0, 0), (1, 0), (1, 1)],
                                           label="foo"))

        ls = LineString([])
        assert ls.almost_equals(LineString([]))
        assert not ls.almost_equals(LineString([], label="foo"))

        ls = LineString([], label="foo")
        assert not ls.almost_equals(LineString([]))
        assert ls.almost_equals(LineString([], label="foo"))

    def test_copy(self):
        lss = [
            LineString([(0, 0), (1, 0), (1, 1)]),
            LineString([(0, 0), (1.5, 0), (1.6, 1)]),
            LineString([(0, 0)]),
            LineString([]),
            LineString([(0, 0), (1, 0), (1, 1)], label="foo")
        ]
        for ls in lss:
            observed = ls.copy()
            assert observed is not ls
            assert observed.coords is ls.coords
            assert observed.label is ls.label

        ls = LineString([(0, 0), (1, 0), (1, 1)]),
        observed = ls.copy(coords=[(0, 0)])
        assert observed.coords_almost_equals([(0, 0)])
        assert observed.label is None

        ls = LineString([(0, 0), (1, 0), (1, 1)]),
        observed = ls.copy(label="bar")
        assert observed.coords is ls.coords
        assert observed.label == "bar"

    def test_deepcopy(self):
        lss = [
            LineString([(0, 0), (1, 0), (1, 1)]),
            LineString([(0, 0), (1.5, 0), (1.6, 1)]),
            LineString([(0, 0)]),
            LineString([]),
            LineString([(0, 0), (1, 0), (1, 1)], label="foo")
        ]
        for ls in lss:
            observed = ls.deepcopy()
            assert observed is not ls
            assert observed.coords is not ls.coords
            assert observed.label is None or observed.label is not ls.label

        ls = LineString([(0, 0), (1, 0), (1, 1)]),
        observed = ls.deepcopy(coords=[(0, 0)])
        assert observed.coords_almost_equals([(0, 0)])
        assert observed.label is None

        ls = LineString([(0, 0), (1, 0), (1, 1)]),
        observed = ls.deepcopy(label="bar")
        assert observed.coords is not ls.coords
        assert observed.label == "bar"

    def test___repr__(self):
        lss = [
            LineString([(0, 0), (1, 0), (1, 1)]),
            LineString([(0, 0), (1.5, 0), (1.6, 1)]),
            LineString([(0, 0)]),
            LineString([]),
            LineString([(0, 0), (1, 0), (1, 1)], label="foo")
        ]
        for ls in lss:
            assert ls.__repr__() == ls.__str__()

    def test___str__(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        observed = ls.__str__()
        expected = ("LineString([(0.00, 0.00), (1.00, 0.00), (1.00, 1.00)], "
                    "label=None)")
        assert observed == expected

        ls = LineString([(0, 0), (1.5, 0), (1.6, 1)])
        observed = ls.__str__()
        expected = ("LineString([(0.00, 0.00), (1.50, 0.00), (1.60, 1.00)], "
                    "label=None)")
        assert observed == expected

        ls = LineString([(0, 0)])
        observed = ls.__str__()
        expected = "LineString([(0.00, 0.00)], label=None)"
        assert observed == expected

        ls = LineString([])
        observed = ls.__str__()
        expected = "LineString([], label=None)"
        assert observed == expected

        ls = LineString([(0, 0), (1, 0), (1, 1)], label="foo")
        observed = ls.__str__()
        expected = ("LineString([(0.00, 0.00), (1.00, 0.00), (1.00, 1.00)], "
                    "label=foo)")
        assert observed == expected
