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

from imgaug.testutils import reseed
from imgaug.augmentables.utils import (
    interpolate_points, interpolate_point_pair,
    interpolate_points_by_max_distance
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_interpolate_point_pair(self):
        point_a = (0, 0)
        point_b = (1, 2)
        inter = interpolate_point_pair(point_a, point_b, 1)
        assert np.allclose(
            inter,
            np.float32([
                [0.5, 1.0]
            ])
        )

        inter = interpolate_point_pair(point_a, point_b, 2)
        assert np.allclose(
            inter,
            np.float32([
                [1*1/3, 1*2/3],
                [2*1/3, 2*2/3]
            ])
        )

        inter = interpolate_point_pair(point_a, point_b, 0)
        assert len(inter) == 0

    def test_interpolate_points(self):
        # 2 points
        points = [
            (0, 0),
            (1, 2)
        ]
        inter = interpolate_points(points, 0)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [1, 2]
            ])
        )

        inter = interpolate_points(points, 1)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [0.5, 1.0],
                [1, 2],
                [0.5, 1.0]
            ])
        )

        inter = interpolate_points(points, 1, closed=False)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [0.5, 1.0],
                [1, 2]
            ])
        )

        # 3 points
        points = [
            (0, 0),
            (1, 2),
            (0.5, 3)
        ]

        inter = interpolate_points(points, 0)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [1, 2],
                [0.5, 3]
            ])
        )

        inter = interpolate_points(points, 1)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [0.5, 1.0],
                [1, 2],
                [0.75, 2.5],
                [0.5, 3],
                [0.25, 1.5]
            ])
        )

        inter = interpolate_points(points, 1, closed=False)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [0.5, 1.0],
                [1, 2],
                [0.75, 2.5],
                [0.5, 3]
            ])
        )

        # 0 points
        points = []
        inter = interpolate_points(points, 1)
        assert len(inter) == 0

        # 1 point
        points = [(0, 0)]
        inter = interpolate_points(points, 0)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0]
            ])
        )
        inter = interpolate_points(points, 1)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0]
            ])
        )

    def test_interpolate_points_by_max_distance(self):
        # 2 points
        points = [
            (0, 0),
            (0, 2)
        ]
        inter = interpolate_points_by_max_distance(points, 10000)
        assert np.allclose(
            inter,
            points
        )

        inter = interpolate_points_by_max_distance(points, 1.0)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [0, 1.0],
                [0, 2],
                [0, 1.0]
            ])
        )

        inter = interpolate_points_by_max_distance(points, 1.0, closed=False)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [0, 1.0],
                [0, 2]
            ])
        )

        # 3 points
        points = [
            (0, 0),
            (0, 2),
            (2, 0)
        ]

        inter = interpolate_points_by_max_distance(points, 1.0)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [0, 1.0],
                [0, 2],
                [1.0, 1.0],
                [2, 0],
                [1.0, 0]
            ])
        )

        inter = interpolate_points_by_max_distance(points, 1.0, closed=False)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [0, 1.0],
                [0, 2],
                [1.0, 1.0],
                [2, 0]
            ])
        )

        # 0 points
        points = []
        inter = interpolate_points_by_max_distance(points, 1.0)
        assert len(inter) == 0

        # 1 points
        points = [(0, 0)]

        inter = interpolate_points_by_max_distance(points, 1.0)
        assert np.allclose(
            inter,
            np.float32([
                [0, 0]
            ])
        )
