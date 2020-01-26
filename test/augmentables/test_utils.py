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

import numpy as np

from imgaug.augmentables.utils import (
    interpolate_points, interpolate_point_pair,
    interpolate_points_by_max_distance
)


class Test_interpolate_point_pair(unittest.TestCase):
    def test_1_step(self):
        point_a = (0, 0)
        point_b = (1, 2)
        inter = interpolate_point_pair(point_a, point_b, 1)
        assert np.allclose(
            inter,
            np.float32([
                [0.5, 1.0]
            ])
        )

    def test_2_steps(self):
        point_a = (0, 0)
        point_b = (1, 2)
        inter = interpolate_point_pair(point_a, point_b, 2)
        assert np.allclose(
            inter,
            np.float32([
                [1*1/3, 1*2/3],
                [2*1/3, 2*2/3]
            ])
        )

    def test_0_steps(self):
        point_a = (0, 0)
        point_b = (1, 2)
        inter = interpolate_point_pair(point_a, point_b, 0)
        assert len(inter) == 0


class Test_interpolate_points(unittest.TestCase):
    def test_2_points_0_steps(self):
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

    def test_2_points_1_step(self):
        points = [
            (0, 0),
            (1, 2)
        ]

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

    def test_2_points_1_step_not_closed(self):
        points = [
            (0, 0),
            (1, 2)
        ]

        inter = interpolate_points(points, 1, closed=False)

        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [0.5, 1.0],
                [1, 2]
            ])
        )

    def test_3_points_0_steps(self):
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

    def test_3_points_1_step(self):
        points = [
            (0, 0),
            (1, 2),
            (0.5, 3)
        ]

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

    def test_3_points_1_step_not_closed(self):
        points = [
            (0, 0),
            (1, 2),
            (0.5, 3)
        ]

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

    def test_0_points_1_step(self):
        points = []

        inter = interpolate_points(points, 1)

        assert len(inter) == 0

    def test_1_point_0_steps(self):
        points = [(0, 0)]

        inter = interpolate_points(points, 0)

        assert np.allclose(
            inter,
            np.float32([
                [0, 0]
            ])
        )

    def test_1_point_1_step(self):
        points = [(0, 0)]

        inter = interpolate_points(points, 1)

        assert np.allclose(
            inter,
            np.float32([
                [0, 0]
            ])
        )


class Test_interpolate_points_by_max_distance(unittest.TestCase):
    def test_2_points_dist_10000(self):
        points = [
            (0, 0),
            (0, 2)
        ]

        inter = interpolate_points_by_max_distance(points, 10000)

        assert np.allclose(
            inter,
            points
        )

    def test_2_points_dist_1(self):
        points = [
            (0, 0),
            (0, 2)
        ]

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

    def test_2_points_dist_1_not_closed(self):
        points = [
            (0, 0),
            (0, 2)
        ]

        inter = interpolate_points_by_max_distance(points, 1.0, closed=False)

        assert np.allclose(
            inter,
            np.float32([
                [0, 0],
                [0, 1.0],
                [0, 2]
            ])
        )

    def test_3_points_dist_1(self):
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

    def test_3_points_dist_1_not_closed(self):
        points = [
            (0, 0),
            (0, 2),
            (2, 0)
        ]

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

    def test_0_points_dist_1(self):
        points = []

        inter = interpolate_points_by_max_distance(points, 1.0)

        assert len(inter) == 0

    def test_1_point_dist_1(self):
        points = [(0, 0)]

        inter = interpolate_points_by_max_distance(points, 1.0)

        assert np.allclose(
            inter,
            np.float32([
                [0, 0]
            ])
        )
