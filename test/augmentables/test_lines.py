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
                          np.sqrt(1**2 + 2**2) + np.sqrt(3**2 + 3**2))

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

    def test_compute_neighbour_distances(self):
        ls = LineString([(0, 0), (1.4, 0), (2.6, 1)])
        dists = ls.compute_neighbour_distances()
        assert np.allclose(dists, [np.sqrt(1.4**2), np.sqrt(1.2**2+1**2)])

        ls = LineString([(0, 0), (1.4, 0)])
        dists = ls.compute_neighbour_distances()
        assert np.allclose(dists, [np.sqrt(1.4**2)])

        ls = LineString([(0, 0)])
        dists = ls.compute_neighbour_distances()
        assert dists.shape == (0,)

        ls = LineString([])
        dists = ls.compute_neighbour_distances()
        assert dists.shape == (0,)

    def test_compute_pointwise_distances(self):
        from imgaug.augmentables.kps import Keypoint

        # (x, y) tuple
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        dists = ls.compute_pointwise_distances((0, 0))
        assert np.allclose(dists, [0,
                                   5,
                                   np.sqrt(5**2 + 5**2)])

        ls = LineString([(0, 0), (5, 0), (5, 5)])
        dists = ls.compute_pointwise_distances((1, 1))
        assert np.allclose(dists, [np.sqrt(1**2 + 1**2),
                                   np.sqrt(4**2 + 1**2),
                                   np.sqrt(4**2 + 4**2)])

        ls = LineString([])
        dists = ls.compute_pointwise_distances((1, 1))
        assert dists == []

        # keypoint
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        dists = ls.compute_pointwise_distances(Keypoint(x=0, y=0))
        assert np.allclose(dists, [0, 5, np.sqrt(5**2 + 5**2)])

        ls = LineString([(0, 0), (5, 0), (5, 5)])
        dists = ls.compute_pointwise_distances(Keypoint(x=1, y=1))
        assert np.allclose(dists, [np.sqrt(1**2 + 1**2),
                                   np.sqrt(4**2 + 1**2),
                                   np.sqrt(4**2 + 4**2)])

        # line string
        ls = LineString([(0, 0), (5, 0), (5, 5)])
        other = LineString([(0, 0)])
        dists = ls.compute_pointwise_distances(other)
        assert np.allclose(dists, [0,
                                   5,
                                   np.sqrt(5**2 + 5**2)])

        ls = LineString([(0, 0), (5, 0), (5, 5)])
        other = LineString([(1, 1)])
        dists = ls.compute_pointwise_distances(other)
        assert np.allclose(dists, [np.sqrt(1**2 + 1**2),
                                   np.sqrt(4**2 + 1**2),
                                   np.sqrt(4**2 + 4**2)])

        ls = LineString([(0, 0), (5, 0), (5, 5)])
        other = LineString([(0, -1), (5, -1)])
        dists = ls.compute_pointwise_distances(other)
        assert np.allclose(dists, [np.sqrt(0**2 + 1**2),
                                   np.sqrt(0**2 + 1**2),
                                   np.sqrt(0**2 + 6**2)])

        ls = LineString([(0, 0), (5, 0), (5, 5)])
        other = LineString([])
        dists = ls.compute_pointwise_distances(other, default=False)
        assert dists is False

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
        assert np.isclose(ls.compute_distance(LineString([(0, 0)])), 0)
        assert np.isclose(ls.compute_distance(LineString([(0, 1)])), 1)
        assert np.isclose(ls.compute_distance(LineString([(0, 0), (0, 1)])), 0)
        assert np.isclose(ls.compute_distance(LineString([(-1, -1), (-1, 1)])), 1)

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
        assert not ls.contains((0+1e-8, 0), max_distance=0)
        assert not ls.contains((0-1, 0))
        assert ls.contains((0-1, 0), max_distance=2)

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
        ls_proj = ls.project((10, 10), (20, 10))
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
        assert ls.is_partly_within_image((10, 1))
        assert not ls.is_partly_within_image((1, 1))

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

    def test_is_out_of_image(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert not ls.is_out_of_image((10, 10))
        assert ls.is_out_of_image((1, 1), fully=False, partly=True)
        assert not ls.is_out_of_image((1, 1), fully=True, partly=False)
        assert ls.is_out_of_image((1, 1), fully=True, partly=True)
        assert not ls.is_out_of_image((1, 1), fully=False, partly=False)

        ls = LineString([])
        assert ls.is_out_of_image((10, 10))
        assert not ls.is_out_of_image((10, 10), default=False)
        assert ls.is_out_of_image((10, 10), default=None) is None

    def test_clip_out_of_image(self):
        def _eq(ls, other):
            return ls.coords_almost_equals(other, max_distance=1e-2)

        ls = LineString([(0, 0), (1, 0), (2, 1)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((2, 2))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((2, 1))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], [(0, 0), (1, 0)])

        # same as above, all coords at x+5, y+5
        ls = LineString([(5, 5), (6, 5), (7, 6)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((4, 4))
        assert len(lss_clipped) == 0

        # line that leaves image plane and comes back
        ls = LineString([(0, 0), (1, 0), (3, 0),
                         (3, 2), (1, 2), (0, 2)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((10, 2))
        assert len(lss_clipped) == 2
        assert _eq(lss_clipped[0], [(0, 0), (1, 0), (2, 0)])
        assert _eq(lss_clipped[1], [(2, 2), (1, 2), (0, 2)])

        lss_clipped = ls.clip_out_of_image((10, 1))
        assert len(lss_clipped) == 2
        assert _eq(lss_clipped[0], [(0, 0), (1, 0)])
        assert _eq(lss_clipped[1], [(1, 2), (0, 2)])

        # same as above, but removing first and last point
        # so that only one point before and after out of image part remain
        ls = LineString([(1, 0), (3, 0),
                         (3, 2), (1, 2)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((10, 2))
        assert len(lss_clipped) == 2
        assert _eq(lss_clipped[0], [(1, 0), (2, 0)])
        assert _eq(lss_clipped[1], [(2, 2), (1, 2)])

        lss_clipped = ls.clip_out_of_image((10, 1))
        assert len(lss_clipped) == 0

        # same as above, but only one point out of image remains
        ls = LineString([(1, 0), (3, 0),
                         (1, 2)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((10, 2))
        assert len(lss_clipped) == 2
        assert _eq(lss_clipped[0], [(1, 0), (2, 0)])
        assert _eq(lss_clipped[1], [(2, 1), (1, 2)])

        lss_clipped = ls.clip_out_of_image((10, 1))
        assert len(lss_clipped) == 0

        # line string that leaves image, comes back, then leaves again, then
        # comes back again
        ls = LineString([(1, 0), (3, 0),  # leaves
                         (3, 1), (1, 1),  # comes back
                         (1, 2), (3, 2),  # leaves
                         (3, 3), (1, 3)])  # comes back

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((10, 2))
        assert len(lss_clipped) == 3  # from above: 1s line, 2nd+3rd, 4th
        assert _eq(lss_clipped[0], [(1, 0), (2, 0)])
        assert _eq(lss_clipped[1], [(2, 1), (1, 1), (1, 2), (2, 2)])
        assert _eq(lss_clipped[2], [(2, 3), (1, 3)])

        # line string that starts out of image and ends within the image plane
        for y in [1, 0]:
            # one point inside image
            ls = LineString([(-10, y), (3, y)])

            lss_clipped = ls.clip_out_of_image((10, 10))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(0, y), (3, y)])

            lss_clipped = ls.clip_out_of_image((2, 1))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(0, y), (1, y)])

            lss_clipped = ls.clip_out_of_image((1, 1))
            if y == 1:
                assert len(lss_clipped) == 0
            else:
                assert len(lss_clipped) == 1
                assert _eq(lss_clipped[0], [(0, y), (1, y)])

            # two points inside image
            ls = LineString([(-10, y), (3, y), (5, y)])

            lss_clipped = ls.clip_out_of_image((10, 10))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(0, y), (3, y), (5, y)])

            lss_clipped = ls.clip_out_of_image((10, 4))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(0, y), (3, y), (4, y)])

            lss_clipped = ls.clip_out_of_image((2, 1))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(0, y), (1, y)])

            lss_clipped = ls.clip_out_of_image((1, 1))
            if y == 1:
                assert len(lss_clipped) == 0
            else:
                assert len(lss_clipped) == 1
                assert _eq(lss_clipped[0], [(0, y), (1, y)])

        # line string that starts within the image plane and ends outside
        for y in [1, 0]:
            # one point inside image
            ls = LineString([(2, y), (5, y)])

            lss_clipped = ls.clip_out_of_image((10, 10))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(2, y), (5, y)])

            lss_clipped = ls.clip_out_of_image((10, 4))
            assert _eq(lss_clipped[0], [(2, y), (4, y)])

            # two points inside image
            ls = LineString([(1, y), (2, y), (5, y)])

            lss_clipped = ls.clip_out_of_image((10, 10))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(1, y), (2, y), (5, y)])

            lss_clipped = ls.clip_out_of_image((10, 4))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(1, y), (2, y), (4, y)])

            lss_clipped = ls.clip_out_of_image((2, 1))
            assert len(lss_clipped) == 0

            # two points outside image
            ls = LineString([(2, y), (5, y), (6, y)])

            lss_clipped = ls.clip_out_of_image((10, 10))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(2, y), (5, y), (6, y)])

            lss_clipped = ls.clip_out_of_image((10, 4))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(2, y), (4, y)])

            lss_clipped = ls.clip_out_of_image((2, 1))
            assert len(lss_clipped) == 0

        # line string that cuts through the image plane in the center
        for y in [1, 0]:
            ls = LineString([(-5, y), (5, y)])

            lss_clipped = ls.clip_out_of_image((10, 10))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(0, y), (5, y)])

            lss_clipped = ls.clip_out_of_image((4, 4))
            assert len(lss_clipped) == 1
            assert _eq(lss_clipped[0], [(0, y), (4, y)])

        # line string that cuts through the image plane from the bottom left
        # corner to the top right corner
        ls = LineString([(-5, -5), (5, 5)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], [(0, 0), (5, 5)])

        lss_clipped = ls.clip_out_of_image((4, 4))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], [(0, 0), (4, 4)])

        # line string that overlaps with the bottom edge
        ls = LineString([(1, 0), (4, 0)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((3, 3))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], [(1, 0), (3, 0)])

        # same as above, multiple points on line
        ls = LineString([(1, 0), (4, 0), (5, 0), (6, 0), (7, 0)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((5, 5))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], [(1, 0), (4, 0), (5, 0)])

        lss_clipped = ls.clip_out_of_image((5, 4))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], [(1, 0), (4, 0)])

        lss_clipped = ls.clip_out_of_image((5, 2))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], [(1, 0), (2, 0)])

        # line string that starts outside the image, intersects with the
        # bottom left corner and overlaps with the bottom border
        ls = LineString([(-5, 0), (5, 0)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], [(0, 0), (5, 0)])

        lss_clipped = ls.clip_out_of_image((10, 5))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], [(0, 0), (5, 0)])

        lss_clipped = ls.clip_out_of_image((10, 4))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], [(0, 0), (4, 0)])

        # line string that contains a single point
        ls = LineString([(2, 2)])

        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 1
        assert _eq(lss_clipped[0], ls)

        lss_clipped = ls.clip_out_of_image((1, 1))
        assert len(lss_clipped) == 0

        # line string that is empty
        ls = LineString([])
        lss_clipped = ls.clip_out_of_image((10, 10))
        assert len(lss_clipped) == 0

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
        ls = LineString([(0, 1), (5, 1), (5, 5)])
        arr = ls.draw_mask(
            (10, 10), size_lines=1, size_points=0, raise_if_out_of_image=False)
        assert np.all(arr[1, 0:5])
        assert np.all(arr[1:5, 5])
        assert not np.any(arr[0, :])
        assert not np.any(arr[2:, 0:5])

        ls = LineString([])
        arr = ls.draw_mask(
            (10, 10), size_lines=1, raise_if_out_of_image=False)
        assert not np.any(arr)

    def test_draw_line_heatmap_array(self):
        def _allclose(arr, v):
            # draw_line_heatmap_array() is currently limited to 1/255 accuracy
            # due to drawing in uint8
            return np.allclose(arr, v, atol=(1.01/255), rtol=0)

        ls = LineString([(0, 1), (5, 1), (5, 5)])
        arr = ls.draw_lines_heatmap_array(
            (10, 10), alpha=0.5, size=1, raise_if_out_of_image=False)
        assert _allclose(arr[1, 0:5], 0.5)
        assert _allclose(arr[1:5, 5], 0.5)
        assert _allclose(arr[0, :], 0.0)
        assert _allclose(arr[2:, 0:5], 0.0)

        ls = LineString([])
        arr = ls.draw_lines_heatmap_array(
            (10, 10), alpha=0.5, size=1, raise_if_out_of_image=False)
        assert _allclose(arr, 0.0)

    def test_draw_points_heatmap_array(self):
        def _allclose(arr, v):
            # draw_points_heatmap_array() is currently limited to 1/255 accuracy
            # due to drawing in uint8
            return np.allclose(arr, v, atol=(1.01/255), rtol=0)

        ls = LineString([(0, 1), (5, 1), (5, 5)])
        arr = ls.draw_points_heatmap_array(
            (10, 10), alpha=0.5, size=1, raise_if_out_of_image=False)
        assert _allclose(arr[1, 0], 0.5)
        assert _allclose(arr[1, 5], 0.5)
        assert _allclose(arr[5, 5], 0.5)
        assert _allclose(arr[0, :], 0.0)
        assert _allclose(arr[2:, 0:5], 0.0)

        ls = LineString([])
        arr = ls.draw_points_heatmap_array(
            (10, 10), alpha=0.5, size=1, raise_if_out_of_image=False)
        assert _allclose(arr, 0.0)

    def test_draw_heatmap_array(self):
        def _allclose(arr, v):
            # draw_line_heatmap_array() is currently limited to 1/255 accuracy
            # due to drawing in uint8
            return np.allclose(arr, v, atol=(1.01/255), rtol=0)

        module_name = "imgaug.augmentables.lines."
        line_fname = "%sLineString.draw_lines_heatmap_array" % (module_name,)
        points_fname = "%sLineString.draw_points_heatmap_array" % (module_name,)
        with mock.patch(line_fname, return_value=1) as mock_line, \
                mock.patch(points_fname, return_value=2) as mock_points:
            ls = LineString([(0, 1), (9, 1)])
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

        ls = LineString([(0, 1), (5, 1), (5, 5)])
        arr = ls.draw_heatmap_array((10, 10),
                                    alpha_lines=0.9, alpha_points=0.5,
                                    size_lines=1, size_points=3,
                                    antialiased=False,
                                    raise_if_out_of_image=False)
        assert _allclose(arr[1, 0:5], 0.9)
        assert _allclose(arr[1, 0:5], 0.9)
        assert _allclose(arr[1, 0:5], 0.9)
        assert _allclose(arr[2:5, 5], 0.9)
        assert _allclose(arr[2:5, 5], 0.9)
        assert _allclose(arr[2:5, 5], 0.9)

        assert _allclose(arr[0, 0:2], 0.5)
        assert _allclose(arr[2, 0:2], 0.5)

        assert _allclose(arr[0, 4:6+1], 0.5)
        assert _allclose(arr[2, 4], 0.5)
        assert _allclose(arr[2, 6], 0.5)

        assert _allclose(arr[4, 4], 0.5)
        assert _allclose(arr[4, 6], 0.5)
        assert _allclose(arr[6, 4:6+1], 0.5)

        assert _allclose(arr[0, 3], 0.0)
        assert _allclose(arr[7:, :], 0.0)

        ls = LineString([])
        arr = ls.draw_heatmap_array((10, 10))
        assert arr.shape == (10, 10)
        assert np.sum(arr) == 0

    def test_draw_line_on_image(self):
        ls = LineString([(0, 1), (9, 1)])

        # image of 0s
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

        # image of 0s, 2D input image
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

        # image of 1s
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

        # alpha=0.5
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

        # alpha=0.5 with background
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

        # raise_if_out_of_image=True
        got_exception = False
        try:
            ls = LineString([(0-5, 5), (-1, 5)])
            _img = ls.draw_lines_on_image(
                np.zeros((10, 10, 3), dtype=np.uint8),
                color=(100, 100, 100),
                alpha=1.0, size=3,
                antialiased=False,
                raise_if_out_of_image=True
            )
        except Exception as exc:
            assert "Cannot draw line string " in str(exc)
            got_exception = True
        assert got_exception

        # raise_if_out_of_image=True BUT line is partially inside image
        # (no point is inside image though)
        got_exception = False
        try:
            ls = LineString([(-1, 5), (11, 5)])
            _img = ls.draw_lines_on_image(
                np.zeros((10, 10, 3), dtype=np.uint8),
                color=(100, 100, 100),
                alpha=1.0, size=3,
                antialiased=False,
                raise_if_out_of_image=True
            )
        except Exception as exc:
            assert "Cannot draw line string " in str(exc)
            got_exception = True
        assert not got_exception

    def test_draw_points_on_image(self):
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

        # image of 1s
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

        # alpha=0.5
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

        # size=1
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

        # raise_if_out_of_image=True
        got_exception = False
        try:
            ls = LineString([(0-5, 1), (9+5, 1)])
            _img = ls.draw_points_on_image(
                np.zeros((3, 10, 3), dtype=np.uint8),
                color=(10, 200, 20),
                alpha=0.5, size=1,
                raise_if_out_of_image=True
            )
        except Exception as exc:
            assert "Cannot draw keypoint " in str(exc)
            got_exception = True
        assert got_exception

    def test_draw_on_image(self):
        module_name = "imgaug.augmentables.lines."
        line_fname = "%sLineString.draw_lines_on_image" % (module_name,)
        points_fname = "%sLineString.draw_points_on_image" % (module_name,)
        with mock.patch(line_fname, return_value=1) as mock_line, \
                mock.patch(points_fname, return_value=2) as mock_points:
            ls = LineString([(0, 1), (9, 1)])
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

        ls = LineString([])
        img = ls.draw_on_image(np.zeros((10, 10, 3), dtype=np.uint8))
        assert img.shape == (10, 10, 3)
        assert np.sum(img) == 0

    def test_extract_from_image(self):
        img = np.arange(10*10).reshape((10, 10, 1)).astype(np.uint8)

        # size=1
        ls = LineString([(0, 5), (9, 5)])
        extract = ls.extract_from_image(img, antialiased=False)
        assert extract.shape == (1, 10, 1)
        assert np.array_equal(extract, img[5:6, 0:10, :])

        # size=3
        ls = LineString([(1, 5), (8, 5)])
        extract = ls.extract_from_image(img, size=3, antialiased=False)
        assert extract.shape == (3, 10, 1)
        assert np.array_equal(extract, img[4:6+1, 0:10, :])

        # size=3, RGB image
        ls = LineString([(1, 5), (8, 5)])
        img_rgb = np.tile(img, (1, 1, 3))
        img_rgb[..., 1] += 10
        img_rgb[..., 2] += 20
        extract = ls.extract_from_image(img_rgb, size=3, antialiased=False)
        assert extract.shape == (3, 10, 3)
        assert np.array_equal(extract, img_rgb[4:6+1, 0:10, :])

        # weak antialiased=True test
        ls = LineString([(1, 1), (9, 9)])
        extract_aa = ls.extract_from_image(img, size=3, antialiased=True)
        extract = ls.extract_from_image(img, size=3, antialiased=False)
        assert extract_aa.shape == extract.shape
        assert np.sum(extract_aa) > np.sum(extract)

        # pad=False
        ls = LineString([(-5, 5), (-3, 5)])
        extract = ls.extract_from_image(img, size=1, antialiased=False,
                                        pad=False, prevent_zero_size=True)
        assert extract.shape == (1, 1, 1)
        assert np.sum(extract) == 0

        # pad=False, prevent_zero_size=False
        ls = LineString([(-5, 5), (-3, 5)])
        extract = ls.extract_from_image(img, size=1, antialiased=False,
                                        pad=False, prevent_zero_size=False)
        assert extract.shape == (0, 0, 1)

        # pad_max=1
        ls = LineString([(-5, 5), (9, 5)])
        extract = ls.extract_from_image(img, antialiased=False, pad=True,
                                        pad_max=1)
        assert extract.shape == (1, 11, 1)
        assert np.array_equal(extract[:, 1:], img[5:6, 0:10, :])
        assert np.all(extract[0, 0, :] == 0)

        # 1 coord
        ls = LineString([(1, 1)])
        extract = ls.extract_from_image(img)
        assert extract.shape == (1, 1, 1)
        assert np.sum(extract) == img[1:2, 1:2, :]

        ls = LineString([(-10, -10)])
        extract = ls.extract_from_image(img)
        assert extract.shape == (1, 1, 1)
        assert np.sum(extract) == 0

        ls = LineString([(-10, -10)])
        extract = ls.extract_from_image(img, prevent_zero_size=True)
        assert extract.shape == (1, 1, 1)
        assert np.sum(extract) == 0

        # 0 coords
        ls = LineString([])
        extract = ls.extract_from_image(img)
        assert extract.shape == (1, 1, 1)
        assert np.sum(extract) == 0

        ls = LineString([])
        extract = ls.extract_from_image(img, prevent_zero_size=False)
        assert extract.shape == (0, 0, 1)
        assert np.sum(extract) == 0

    def test_concatenate(self):
        ls = LineString([(0, 0), (1, 0), (2, 1)])
        assert ls.concatenate(ls).coords_almost_equals([
            (0, 0), (1, 0), (2, 1), (0, 0), (1, 0), (2, 1)
        ])

        ls = LineString([])
        assert ls.concatenate(ls).coords_almost_equals([])

        ls = LineString([])
        assert ls.concatenate(LineString([(0, 0)])).coords_almost_equals([(0, 0)])

        ls = LineString([(0, 0)])
        assert ls.concatenate(LineString([])).coords_almost_equals([(0, 0)])

        ls = LineString([])
        assert ls.concatenate([(0, 0)]).coords_almost_equals([(0, 0)])

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
        assert np.allclose(observed.exterior, [(0, 0), (1, 0), (1, 1)])

        ls = LineString([(0, 0)])
        observed = ls.to_polygon()
        assert isinstance(observed, ia.Polygon)
        assert np.allclose(observed.exterior, [(0, 0)])

        ls = LineString([])
        observed = ls.to_polygon()
        assert isinstance(observed, ia.Polygon)
        assert len(observed.exterior) == 0

    def test_to_heatmap(self):
        from imgaug.augmentables.heatmaps import HeatmapsOnImage
        ls = LineString([(0, 5), (5, 5)])
        observed = ls.to_heatmap((10, 10), antialiased=False)
        assert isinstance(observed, HeatmapsOnImage)
        assert observed.shape == (10, 10)
        assert observed.arr_0to1.shape == (10, 10, 1)
        assert np.allclose(observed.arr_0to1[0:5, :, :], 0.0)
        assert np.allclose(observed.arr_0to1[5, 0:5, :], 1.0)
        assert np.allclose(observed.arr_0to1[6:, :, :], 0.0)

        ls = LineString([])
        observed = ls.to_heatmap((5, 5), antialiased=False)
        assert observed.shape == (5, 5)
        assert observed.arr_0to1.shape == (5, 5, 1)
        assert np.allclose(observed.arr_0to1, 0.0)

    def test_segmentation_map(self):
        from imgaug.augmentables.segmaps import SegmentationMapOnImage
        ls = LineString([(0, 5), (5, 5)])
        observed = ls.to_segmentation_map((10, 10))
        assert isinstance(observed, SegmentationMapOnImage)
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

    def test_coords_almost_equals(self):
        ls = LineString([(0, 0), (1, 0), (1, 1)])
        assert ls.coords_almost_equals(ls)
        assert ls.coords_almost_equals([(0, 0), (1, 0), (1, 1)])
        assert not ls.shift(top=1).coords_almost_equals(ls)
        assert ls.shift(top=1).coords_almost_equals(ls, max_distance=1.01)
        assert ls.coords_almost_equals([(0, 0), (0.5, 0), (1, 0), (1, 1)])

        ls = LineString([(0, 0), (0.5, 0), (1, 0), (1, 1)])
        assert ls.coords_almost_equals([(0, 0), (1, 0), (1, 1)])

        ls = LineString([(0, 0)])
        assert ls.coords_almost_equals([(0, 0)])
        assert not ls.coords_almost_equals([(0+1, 0)])
        assert ls.coords_almost_equals([(0+1, 0)], max_distance=1.01)

        ls = LineString([])
        assert ls.coords_almost_equals([])
        assert not ls.coords_almost_equals([(0, 0)])

        ls = LineString([(0, 0)])
        assert not ls.coords_almost_equals([])

        ls_a = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        ls_b = LineString([(0, 0), (10, 0), (10, 10), (0, 10),
                           (0, 5.01), (0, 5.0), (0, 4.99), (0, 0)])
        assert ls_a.coords_almost_equals(ls_b)

        ls_a = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        ls_b = LineString([(0, 0), (10, 0), (10, 10), (0, 10),
                           (0, 5.01), (10, 5.0), (0, 4.99), (0, 0)])
        assert not ls_a.coords_almost_equals(ls_b)

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
            assert ls.coords is ls.coords
            assert observed.coords is ls.coords
            assert observed.label is ls.label

        ls = LineString([(0, 0), (1, 0), (1, 1)])
        observed = ls.copy(coords=[(0, 0)])
        assert observed.coords_almost_equals([(0, 0)])
        assert observed.label is None

        ls = LineString([(0, 0), (1, 0), (1, 1)])
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
            assert observed.label == ls.label

        ls = LineString([(0, 0), (1, 0), (1, 1)])
        observed = ls.deepcopy(coords=[(0, 0)])
        assert observed.coords_almost_equals([(0, 0)])
        assert observed.label is None

        ls = LineString([(0, 0), (1, 0), (1, 1)])
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


class TestLineStringsOnImage(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        lsoi = LineStringsOnImage([], shape=(10, 10))
        assert lsoi.line_strings == []
        assert lsoi.shape == (10, 10)

        lsoi = LineStringsOnImage([
            LineString([]),
            LineString([(0, 0), (5, 0)])
        ], shape=(10, 10, 3))
        assert len(lsoi.line_strings) == 2
        assert lsoi.shape == (10, 10, 3)

    def test_empty(self):
        lsoi = LineStringsOnImage([], shape=(10, 10, 3))
        assert lsoi.empty

        lsoi = LineStringsOnImage([LineString([])], shape=(10, 10, 3))
        assert not lsoi.empty

        lsoi = LineStringsOnImage([LineString([(0, 0)])], shape=(10, 10, 3))
        assert not lsoi.empty

    def test_on(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        lsoi_proj = lsoi.on((100, 100, 3))
        assert all([ls_a.coords_almost_equals(ls_b)
                    for ls_a, ls_b
                    in zip(lsoi.line_strings, lsoi_proj.line_strings)])
        assert lsoi_proj.shape == (100, 100, 3)

        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        lsoi_proj = lsoi.on((200, 200, 3))
        assert lsoi_proj.line_strings[0].coords_almost_equals(
            [(0, 0), (1*2, 0), (2*2, 1*2)]
        )
        assert lsoi_proj.line_strings[1].coords_almost_equals(
            [(10*2, 10*2)]
        )
        assert lsoi_proj.shape == (200, 200, 3)

        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        lsoi_proj = lsoi.on((200, 200, 3))
        assert len(lsoi_proj.line_strings) == 0
        assert lsoi_proj.shape == (200, 200, 3)

    def test_from_xy_arrays(self):
        arrs = np.float32([
            [(0, 0), (10, 10), (5, 10)],
            [(5, 5), (15, 15), (10, 15)]
        ])
        lsoi = LineStringsOnImage.from_xy_arrays(arrs, shape=(100, 100, 3))
        assert len(lsoi.line_strings) == 2
        assert lsoi.line_strings[0].coords_almost_equals(arrs[0])
        assert lsoi.line_strings[1].coords_almost_equals(arrs[1])

        arrs = [
            np.float32([(0, 0), (10, 10), (5, 10)]),
            np.float32([(5, 5), (15, 15), (10, 15), (25, 25)])
        ]
        lsoi = LineStringsOnImage.from_xy_arrays(arrs, shape=(100, 100, 3))
        assert len(lsoi.line_strings) == 2
        assert lsoi.line_strings[0].coords_almost_equals(arrs[0])
        assert lsoi.line_strings[1].coords_almost_equals(arrs[1])

        arrs = np.zeros((0, 0, 2), dtype=np.float32)
        lsoi = LineStringsOnImage.from_xy_arrays(arrs, shape=(100, 100, 3))
        assert len(lsoi.line_strings) == 0

        arrs = np.zeros((0, 5, 2), dtype=np.float32)
        lsoi = LineStringsOnImage.from_xy_arrays(arrs, shape=(100, 100, 3))
        assert len(lsoi.line_strings) == 0

    def test_to_xy_arrays(self):
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

        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        arrs = lsoi.to_xy_arrays()
        assert isinstance(arrs, list)
        assert len(arrs) == 0

    def test_draw_on_image(self):
        ls1 = LineString([(0, 0), (10, 10), (5, 10)])
        ls2 = LineString([(5, 5), (15, 15), (10, 15), (25, 25)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        img = np.zeros((100, 100, 3), dtype=np.uint8) + 1
        observed = lsoi.draw_on_image(img)
        expected = np.copy(img)
        for ls in [ls1, ls2]:
            expected = ls.draw_on_image(expected)
        assert np.array_equal(observed, expected)

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

        ls1 = LineString([])
        ls2 = LineString([(5, 5), (15, 15), (10, 15), (25, 25)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        img = np.zeros((100, 100, 3), dtype=np.uint8) + 1
        observed = lsoi.draw_on_image(img)
        expected = np.copy(img)
        for ls in [ls1, ls2]:
            expected = ls.draw_on_image(expected)
        assert np.array_equal(observed, expected)

        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        img = np.zeros((100, 100, 3), dtype=np.uint8) + 1
        observed = lsoi.draw_on_image(img)
        expected = np.copy(img)
        assert np.array_equal(observed, expected)

    def test_remove_out_of_image(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        observed = lsoi.remove_out_of_image()
        assert len(observed.line_strings) == 2
        assert observed.line_strings[0] is ls1
        assert observed.line_strings[1] is ls2

        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        observed = lsoi.remove_out_of_image()
        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

        ls1 = LineString([])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))
        observed = lsoi.remove_out_of_image()
        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

        ls1 = LineString([(-10, -10), (5, 5)])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))
        observed = lsoi.remove_out_of_image()
        assert len(observed.line_strings) == 1
        assert observed.line_strings[0] is ls1
        assert observed.shape == (100, 100, 3)

        ls1 = LineString([(-10, -10), (5, 5)])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))
        observed = lsoi.remove_out_of_image(partly=True, fully=True)
        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

        ls1 = LineString([(-10, -10)])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))
        observed = lsoi.remove_out_of_image()
        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

        ls1 = LineString([(-10, -10)])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))
        observed = lsoi.remove_out_of_image(partly=False, fully=False)
        assert len(observed.line_strings) == 1
        assert observed.line_strings[0] is ls1
        assert observed.shape == (100, 100, 3)

    def test_clip_out_of_image(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        observed = lsoi.clip_out_of_image()
        expected = []
        expected.extend(ls1.clip_out_of_image((100, 100, 3)))
        expected.extend(ls2.clip_out_of_image((100, 100, 3)))
        assert len(lsoi.line_strings) == len(expected)
        for ls_obs, ls_exp in zip(observed.line_strings, expected):
            assert ls_obs.coords_almost_equals(ls_exp)
        assert observed.shape == (100, 100, 3)

        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        observed = lsoi.clip_out_of_image()
        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

        ls1 = LineString([])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))
        observed = lsoi.clip_out_of_image()
        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

        ls1 = LineString([(-10, -10)])
        lsoi = LineStringsOnImage([ls1], shape=(100, 100, 3))
        observed = lsoi.clip_out_of_image()
        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

    def test_shift(self):
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        observed = lsoi.shift(top=1, right=2, bottom=3, left=4)
        assert observed.line_strings[0].coords_almost_equals(
            ls1.shift(top=1, right=2, bottom=3, left=4)
        )
        assert observed.line_strings[1].coords_almost_equals(
            ls2.shift(top=1, right=2, bottom=3, left=4)
        )
        assert observed.shape == (100, 100, 3)

        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        observed = lsoi.shift(top=1, right=2, bottom=3, left=4)
        assert len(observed.line_strings) == 0
        assert observed.shape == (100, 100, 3)

    def test_copy(self):
        # basic test, without labels
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([ls1, ls2], shape=(100, 100, 3))
        observed = lsoi.copy()
        assert observed.line_strings[0] is ls1
        assert observed.line_strings[1] is ls2
        assert observed.shape == (100, 100, 3)

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

        # LSOI is empty
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        observed = lsoi.copy()
        assert observed.line_strings == []
        assert observed.shape == (100, 100, 3)

        # provide line_strings
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        observed = lsoi.copy(line_strings=[ls1, ls2], shape=(200, 201, 3))
        assert observed.line_strings[0] is ls1
        assert observed.line_strings[1] is ls2
        assert observed.shape == (200, 201, 3)

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

        # provide empty list of line_strings
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        observed = lsoi.copy(line_strings=[], shape=(200, 201, 3))
        assert observed.line_strings == []
        assert observed.shape == (200, 201, 3)

    def test_deepcopy(self):
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

        # LSOI is empty
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        observed = lsoi.deepcopy()
        assert observed.line_strings == []
        assert observed.shape == (100, 100, 3)

        # provide line_strings
        ls1 = LineString([(0, 0), (1, 0), (2, 1)])
        ls2 = LineString([(10, 10)])
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        observed = lsoi.deepcopy(line_strings=[ls1, ls2], shape=(200, 201, 3))
        # line strings provided via line_strings are also deepcopied
        assert observed.line_strings[0].coords_almost_equals(ls1)
        assert observed.line_strings[1].coords_almost_equals(ls2)
        assert observed.shape == (200, 201, 3)

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

        # provide empty list of line_strings
        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        observed = lsoi.deepcopy(line_strings=[], shape=(200, 201, 3))
        assert observed.line_strings == []
        assert observed.shape == (200, 201, 3)

    def test___repr__(self):
        self._test_str_repr(lambda obj: obj.__repr__())

    def test___str__(self):
        self._test_str_repr(lambda obj: obj.__str__())

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

        lsoi = LineStringsOnImage([], shape=(100, 100, 3))
        observed = func(lsoi)
        expected = "LineStringsOnImage([], shape=(100, 100, 3))"
        assert observed == expected
