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
import imgaug as ia
from imgaug.testutils import assertWarns


class TestKeypoint_project_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, kp, from_shape, to_shape):
        return kp.project_(from_shape, to_shape)

    def test_project_same_image_size(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, (10, 10), (10, 10))
        assert kp2.y == 1
        assert kp2.x == 2

    def test_project_onto_higher_image(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, (10, 10), (20, 10))
        assert kp2.y == 2
        assert kp2.x == 2

    def test_project_onto_wider_image(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, (10, 10), (10, 20))
        assert kp2.y == 1
        assert kp2.x == 4

    def test_project_onto_higher_and_wider_image(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, (10, 10), (20, 20))
        assert kp2.y == 2
        assert kp2.x == 4

    def test_inplaceness(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, (10, 10), (10, 10))
        if self._is_inplace:
            assert kp is kp2
        else:
            assert kp is not kp2


class TestKeypoint_project(TestKeypoint_project_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, kp, from_shape, to_shape):
        return kp.project(from_shape, to_shape)


class TestKeypoint_shift_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, kp, *args, **kwargs):
        return kp.shift_(*args, **kwargs)

    def test_shift_on_y_axis(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, y=1)
        assert kp2.y == 2
        assert kp2.x == 2

    def test_shift_on_y_axis_by_negative_amount(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, y=-1)
        assert kp2.y == 0
        assert kp2.x == 2

    def test_shift_on_x_axis(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, x=1)
        assert kp2.y == 1
        assert kp2.x == 3

    def test_shift_on_x_axis_by_negative_amount(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, x=-1)
        assert kp2.y == 1
        assert kp2.x == 1

    def test_shift_on_both_axis(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, y=1, x=2)
        assert kp2.y == 2
        assert kp2.x == 4

    def test_inplaceness(self):
        kp = ia.Keypoint(y=1, x=2)
        kp2 = self._func(kp, x=1)
        if self._is_inplace:
            assert kp is kp2
        else:
            assert kp is not kp2


class TestKeypoint_shift(TestKeypoint_shift_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, kp, *args, **kwargs):
        return kp.shift(*args, **kwargs)


class TestKeypoint(unittest.TestCase):
    def test___init__(self):
        kp = ia.Keypoint(y=1, x=2)
        assert kp.y == 1
        assert kp.x == 2

    def test___init___negative_values(self):
        kp = ia.Keypoint(y=-1, x=-2)
        assert kp.y == -1
        assert kp.x == -2

    def test___init___floats(self):
        kp = ia.Keypoint(y=1.5, x=2.5)
        assert np.isclose(kp.y, 1.5)
        assert np.isclose(kp.x, 2.5)

    def test_coords(self):
        kp = ia.Keypoint(x=1, y=1.5)
        coords = kp.coords
        assert np.allclose(coords, [1, 1.5], atol=1e-8, rtol=0)

    def test_x_int(self):
        kp = ia.Keypoint(y=1, x=2)
        assert kp.x == 2
        assert kp.x_int == 2

    def test_x_int_for_float_inputs(self):
        kp = ia.Keypoint(y=1, x=2.7)
        assert np.isclose(kp.x, 2.7)
        assert kp.x_int == 3

    def test_y_int(self):
        kp = ia.Keypoint(y=1, x=2)
        assert kp.y == 1
        assert kp.y_int == 1

    def test_y_int_for_float_inputs(self):
        kp = ia.Keypoint(y=1.7, x=2)
        assert np.isclose(kp.y, 1.7)
        assert kp.y_int == 2

    def test_xy(self):
        kp = ia.Keypoint(x=2, y=1.7)
        assert np.allclose(kp.xy, (2, 1.7))

    def test_xy_int(self):
        kp = ia.Keypoint(x=1.3, y=1.6)
        xy = kp.xy_int
        assert np.allclose(xy, (1, 2))
        assert xy.dtype.name == "int32"

    def test_is_out_of_image(self):
        kp = ia.Keypoint(y=1, x=2)
        image_shape = (10, 20, 3)
        ooi = kp.is_out_of_image(image_shape)
        assert not ooi

    def test_is_out_of_image__ooi_y(self):
        kp = ia.Keypoint(y=11, x=2)
        image_shape = (10, 20, 3)
        ooi = kp.is_out_of_image(image_shape)
        assert ooi

    def test_is_out_of_image__ooi_x(self):
        kp = ia.Keypoint(y=1, x=21)
        image_shape = (10, 20, 3)
        ooi = kp.is_out_of_image(image_shape)
        assert ooi

    def test_compute_out_of_image_fraction(self):
        kp = ia.Keypoint(y=1, x=2)
        image_shape = (10, 20, 3)
        fraction = kp.compute_out_of_image_fraction(image_shape)
        assert np.isclose(fraction, 0.0)

    def test_compute_out_of_image_fraction_ooi_y(self):
        kp = ia.Keypoint(y=11, x=2)
        image_shape = (10, 20, 3)
        fraction = kp.compute_out_of_image_fraction(image_shape)
        assert np.isclose(fraction, 1.0)

    def test_compute_out_of_image_fraction_ooi_x(self):
        kp = ia.Keypoint(y=1, x=21)
        image_shape = (10, 20, 3)
        fraction = kp.compute_out_of_image_fraction(image_shape)
        assert np.isclose(fraction, 1.0)

    def test_draw_on_image(self):
        kp = ia.Keypoint(x=0, y=0)
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
        image_kp = kp.draw_on_image(
            image, color=(0, 255, 0), alpha=1, size=1, copy=True,
            raise_if_out_of_image=False)
        assert np.all(image_kp[0, 0, :] == [0, 255, 0])
        assert np.all(image_kp[1:, :, :] == 10)
        assert np.all(image_kp[:, 1:, :] == 10)

    def test_draw_on_image_kp_at_top_left_corner_size_1(self):
        kp = ia.Keypoint(x=4, y=4)
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
        image_kp = kp.draw_on_image(
            image, color=(0, 255, 0), alpha=1, size=1, copy=True,
            raise_if_out_of_image=False)
        assert np.all(image_kp[4, 4, :] == [0, 255, 0])
        assert np.all(image_kp[:4, :, :] == 10)
        assert np.all(image_kp[:, :4, :] == 10)

    def test_draw_on_image_kp_at_top_left_corner_size_5(self):
        kp = ia.Keypoint(x=0, y=0)
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
        image_kp = kp.draw_on_image(
            image, color=(0, 255, 0), alpha=1, size=5, copy=True,
            raise_if_out_of_image=False)
        assert np.all(image_kp[:3, :3, :] == [0, 255, 0])
        assert np.all(image_kp[3:, :, :] == 10)
        assert np.all(image_kp[:, 3:, :] == 10)

    def test_draw_on_image_kp_at_top_left_corner_custom_color_and_alpha(self):
        kp = ia.Keypoint(x=0, y=0)
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
        image_kp = kp.draw_on_image(
            image, color=(0, 200, 0), alpha=0.5, size=1, copy=True,
            raise_if_out_of_image=False)
        assert np.all(image_kp[0, 0, :] == [0 + 5, 100 + 5, 0 + 5])
        assert np.all(image_kp[1:, :, :] == 10)
        assert np.all(image_kp[:, 1:, :] == 10)

    def test_draw_on_image_kp_somewhere_inside_image_size_5(self):
        kp = ia.Keypoint(x=4, y=4)
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
        image_kp = kp.draw_on_image(
            image, color=(0, 255, 0), alpha=1, size=5, copy=True,
            raise_if_out_of_image=False)
        assert np.all(image_kp[2:, 2:, :] == [0, 255, 0])
        assert np.all(image_kp[:2, :, :] == 10)
        assert np.all(image_kp[:, :2, :] == 10)

    def test_draw_on_image_kp_at_bottom_right_corner_size_5(self):
        kp = ia.Keypoint(x=5, y=5)
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
        image_kp = kp.draw_on_image(
            image, color=(0, 255, 0), alpha=1, size=5, copy=True,
            raise_if_out_of_image=False)
        assert np.all(image_kp[3:, 3:, :] == [0, 255, 0])
        assert np.all(image_kp[:3, :, :] == 10)
        assert np.all(image_kp[:, :3, :] == 10)

    def test_draw_on_image_kp_outside_image(self):
        kp = ia.Keypoint(x=-1, y=-1)
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
        image_kp = kp.draw_on_image(
            image, color=(0, 255, 0), alpha=1, size=5, copy=True,
            raise_if_out_of_image=False)
        assert np.all(image_kp[:2, :2, :] == [0, 255, 0])
        assert np.all(image_kp[2:, :, :] == 10)
        assert np.all(image_kp[:, 2:, :] == 10)

    def test_generate_similar_points_manhattan_0_steps_list(self):
        kp = ia.Keypoint(y=4, x=5)
        kps_manhatten = kp.generate_similar_points_manhattan(
            0, 1.0, return_array=False)
        assert len(kps_manhatten) == 1
        assert kps_manhatten[0].y == 4
        assert kps_manhatten[0].x == 5

    def test_generate_similar_points_manhattan_1_step_list(self):
        kp = ia.Keypoint(y=4, x=5)
        kps_manhatten = kp.generate_similar_points_manhattan(
            1, 1.0, return_array=False)
        assert len(kps_manhatten) == 5
        expected = [(4, 5), (3, 5), (4, 6), (5, 5), (4, 4)]
        for y, x in expected:
            assert any([
                np.allclose(
                    [y, x],
                    [kp_manhatten.y, kp_manhatten.x]
                )
                for kp_manhatten
                in kps_manhatten
            ])

    def test_generate_similar_points_manhattan_1_step_array(self):
        kp = ia.Keypoint(y=4, x=5)
        kps_manhatten = kp.generate_similar_points_manhattan(
            1, 1.0, return_array=True)
        assert kps_manhatten.shape == (5, 2)
        expected = [(4, 5), (3, 5), (4, 6), (5, 5), (4, 4)]
        for y, x in expected:
            assert any([
                np.allclose(
                    [y, x],
                    [kp_manhatten_y, kp_manhatten_x]
                )
                for kp_manhatten_x, kp_manhatten_y
                in kps_manhatten
            ])

    def test_coords_almost_equals(self):
        kp1 = ia.Keypoint(x=1, y=1.5)
        kp2 = ia.Keypoint(x=1, y=1.5)

        equal = kp1.coords_almost_equals(kp2)

        assert equal

    def test_coords_almost_equals__unequal(self):
        kp1 = ia.Keypoint(x=1, y=1.5)
        kp2 = ia.Keypoint(x=1, y=1.5+10.0)

        equal = kp1.coords_almost_equals(kp2)

        assert not equal

    def test_coords_almost_equals__distance_below_threshold(self):
        kp1 = ia.Keypoint(x=1, y=1.5)
        kp2 = ia.Keypoint(x=1, y=1.5+1e-2)

        equal = kp1.coords_almost_equals(kp2, max_distance=1e-1)

        assert equal

    def test_coords_almost_equals__distance_exceeds_threshold(self):
        kp1 = ia.Keypoint(x=1, y=1.5)
        kp2 = ia.Keypoint(x=1, y=1.5+1e-2)

        equal = kp1.coords_almost_equals(kp2, max_distance=1e-3)

        assert not equal

    def test_coords_almost_equals__array(self):
        kp1 = ia.Keypoint(x=1, y=1.5)
        kp2 = np.float32([1, 1.5])

        equal = kp1.coords_almost_equals(kp2)

        assert equal

    def test_coords_almost_equals__array_unequal(self):
        kp1 = ia.Keypoint(x=1, y=1.5)
        kp2 = np.float32([1, 1.5+1.0])

        equal = kp1.coords_almost_equals(kp2)

        assert not equal

    def test_coords_almost_equals__tuple(self):
        kp1 = ia.Keypoint(x=1, y=1.5)
        kp2 = (1, 1.5)

        equal = kp1.coords_almost_equals(kp2)

        assert equal

    def test_coords_almost_equals__tuple_unequal(self):
        kp1 = ia.Keypoint(x=1, y=1.5)
        kp2 = (1, 1.5+1.0)

        equal = kp1.coords_almost_equals(kp2)

        assert not equal

    @mock.patch("imgaug.augmentables.kps.Keypoint.coords_almost_equals")
    def test_almost_equals(self, mock_cae):
        mock_cae.return_value = "foo"
        kp1 = ia.Keypoint(x=1, y=1.5)
        kp2 = ia.Keypoint(x=1, y=1.5)

        result = kp1.almost_equals(kp2, max_distance=2)

        assert result == "foo"
        mock_cae.assert_called_once_with(kp2, max_distance=2)

    def test_string_conversion_ints(self):
        kp = ia.Keypoint(y=1, x=2)
        assert (
            kp.__repr__()
            == kp.__str__()
            == "Keypoint(x=2.00000000, y=1.00000000)"
        )

    def test_string_conversion_floats(self):
        kp = ia.Keypoint(y=1.2, x=2.7)
        assert (
            kp.__repr__()
            == kp.__str__()
            == "Keypoint(x=2.70000000, y=1.20000000)"
        )


class TestKeypointsOnImage_items_setter(unittest.TestCase):
    def test_with_list_of_keypoints(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpsoi = ia.KeypointsOnImage(keypoints=[], shape=(10, 20, 3))
        kpsoi.items = kps
        assert np.all([
            kp_i.x == kp_j.x and kp_i.y == kp_j.y
            for kp_i, kp_j
            in zip(kpsoi.keypoints, kps)
        ])


class TestKeypointsOnImage_on_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, kpsoi, *args, **kwargs):
        return kpsoi.on_(*args, **kwargs)

    def test_same_image_size(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(10, 20, 3))

        kpi2 = self._func(kpi, (10, 20, 3))

        assert np.all([
            kp_i.x == kp_j.x and kp_i.y == kp_j.y
            for kp_i, kp_j
            in zip(kpi.keypoints, kpi2.keypoints)
        ])
        assert kpi2.shape == (10, 20, 3)

    def test_wider_image(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(10, 20, 3))

        kpi2 = self._func(kpi, (20, 40, 3))

        assert kpi2.keypoints[0].x == 2
        assert kpi2.keypoints[0].y == 4
        assert kpi2.keypoints[1].x == 6
        assert kpi2.keypoints[1].y == 8
        assert kpi2.shape == (20, 40, 3)

    def test_wider_image_shape_given_as_array(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(10, 20, 3))

        image = np.zeros((20, 40, 3), dtype=np.uint8)
        kpi2 = self._func(kpi, image)

        assert kpi2.keypoints[0].x == 2
        assert kpi2.keypoints[0].y == 4
        assert kpi2.keypoints[1].x == 6
        assert kpi2.keypoints[1].y == 8
        assert kpi2.shape == image.shape

    def test_inplaceness(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(10, 20, 3))

        kpi2 = self._func(kpi, (10, 20, 3))

        if self._is_inplace:
            assert kpi is kpi2
        else:
            assert kpi is not kpi2


class TestKeypointsOnImage_on(TestKeypointsOnImage_on_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, kpsoi, *args, **kwargs):
        return kpsoi.on(*args, **kwargs)


class TestKeypointsOnImage_shift_(unittest.TestCase):
    @property
    def _is_inplace(self):
        return True

    def _func(self, kpsoi, *args, **kwargs):
        return kpsoi.shift_(*args, **kwargs)

    def test_shift_by_zero_on_both_axis(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
        kpi2 = self._func(kpi, x=0, y=0)
        assert kpi2.keypoints[0].x == 1
        assert kpi2.keypoints[0].y == 2
        assert kpi2.keypoints[1].x == 3
        assert kpi2.keypoints[1].y == 4

    def test_shift_by_1_on_x_axis(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))

        kpi2 = self._func(kpi, x=1)

        assert kpi2.keypoints[0].x == 1 + 1
        assert kpi2.keypoints[0].y == 2
        assert kpi2.keypoints[1].x == 3 + 1
        assert kpi2.keypoints[1].y == 4

    def test_shift_by_negative_1_on_x_axis(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))

        kpi2 = self._func(kpi, x=-1)

        assert kpi2.keypoints[0].x == 1 - 1
        assert kpi2.keypoints[0].y == 2
        assert kpi2.keypoints[1].x == 3 - 1
        assert kpi2.keypoints[1].y == 4

    def test_shift_by_1_on_y_axis(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))

        kpi2 = self._func(kpi, y=1)

        assert kpi2.keypoints[0].x == 1
        assert kpi2.keypoints[0].y == 2 + 1
        assert kpi2.keypoints[1].x == 3
        assert kpi2.keypoints[1].y == 4 + 1

    def test_shift_by_negative_1_on_y_axis(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))

        kpi2 = self._func(kpi, y=-1)

        assert kpi2.keypoints[0].x == 1
        assert kpi2.keypoints[0].y == 2 - 1
        assert kpi2.keypoints[1].x == 3
        assert kpi2.keypoints[1].y == 4 - 1

    def test_shift_on_both_axis(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))

        kpi2 = self._func(kpi, x=1, y=2)

        assert kpi2.keypoints[0].x == 1 + 1
        assert kpi2.keypoints[0].y == 2 + 2
        assert kpi2.keypoints[1].x == 3 + 1
        assert kpi2.keypoints[1].y == 4 + 2

    def test_inplaceness(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
        kpi2 = self._func(kpi, x=0, y=0)

        if self._is_inplace:
            assert kpi is kpi2
        else:
            assert kpi is not kpi2


class TestKeypointsOnImage_shift(TestKeypointsOnImage_shift_):
    @property
    def _is_inplace(self):
        return False

    def _func(self, kpsoi, *args, **kwargs):
        return kpsoi.shift(*args, **kwargs)


class TestKeypointsOnImage(unittest.TestCase):
    def test_items(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpsoi = ia.KeypointsOnImage(kps, shape=(40, 50, 3))

        items = kpsoi.items

        assert items == kps

    def test_items_empty(self):
        kpsoi = ia.KeypointsOnImage([], shape=(40, 50, 3))

        items = kpsoi.items

        assert items == []

    def test_height(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(10, 20, 3))
        assert kpi.height == 10

    def test_width(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(10, 20, 3))
        assert kpi.width == 20

    def test_shape_is_array(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        with assertWarns(self, ia.DeprecationWarning):
            kpi = ia.KeypointsOnImage(
                keypoints=kps,
                shape=image
            )
        assert kpi.shape == (10, 20, 3)

    def test_draw_on_image(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10

        kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        kps_mask[2, 1] = 1
        kps_mask[4, 3] = 1

        image_kps = kpi.draw_on_image(
            image, color=[0, 255, 0], size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_kps[kps_mask] == [0, 255, 0])
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    def test_draw_on_image_alpha_is_50_percent(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10

        kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        kps_mask[2, 1] = 1
        kps_mask[4, 3] = 1

        image_kps = kpi.draw_on_image(
            image, color=[0, 255, 0], alpha=0.5, size=1, copy=True,
            raise_if_out_of_image=False)

        bg_plus_color_at_alpha = [int(0.5*10+0),
                                  int(0.5*10+0.5*255),
                                  int(10*0.5+0)]
        assert np.all(image_kps[kps_mask] == bg_plus_color_at_alpha)
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    def test_draw_on_image_size_3(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10

        kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        kps_mask[2, 1] = 1
        kps_mask[4, 3] = 1

        image_kps = kpi.draw_on_image(
            image, color=[0, 255, 0], size=3, copy=True,
            raise_if_out_of_image=False)
        kps_mask_size3 = np.copy(kps_mask)
        kps_mask_size3[2-1:2+1+1, 1-1:1+1+1] = 1
        kps_mask_size3[4-1:4+1+1, 3-1:3+1+1] = 1

        assert np.all(image_kps[kps_mask_size3] == [0, 255, 0])
        assert np.all(image_kps[~kps_mask_size3] == [10, 10, 10])

    def test_draw_on_image_blue_color(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10

        kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        kps_mask[2, 1] = 1
        kps_mask[4, 3] = 1

        image_kps = kpi.draw_on_image(
            image, color=[0, 0, 255], size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_kps[kps_mask] == [0, 0, 255])
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    def test_draw_on_image_single_int_as_color(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10

        kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        kps_mask[2, 1] = 1
        kps_mask[4, 3] = 1

        image_kps = kpi.draw_on_image(
            image, color=255, size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_kps[kps_mask] == [255, 255, 255])
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    def test_draw_on_image_copy_is_false(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10

        kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        kps_mask[2, 1] = 1
        kps_mask[4, 3] = 1

        image2 = np.copy(image)
        image_kps = kpi.draw_on_image(
            image2, color=[0, 255, 0], size=1, copy=False,
            raise_if_out_of_image=False)

        assert np.all(image2 == image_kps)
        assert np.all(image_kps[kps_mask] == [0, 255, 0])
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])
        assert np.all(image2[kps_mask] == [0, 255, 0])
        assert np.all(image2[~kps_mask] == [10, 10, 10])

    def test_draw_on_image_keypoint_is_outside_of_image(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(
            keypoints=kps + [ia.Keypoint(x=100, y=100)],
            shape=(5, 5, 3)
        )
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10

        kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        kps_mask[2, 1] = 1
        kps_mask[4, 3] = 1

        image_kps = kpi.draw_on_image(
            image, color=[0, 255, 0], size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_kps[kps_mask] == [0, 255, 0])
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    def test_draw_on_image_keypoint_is_outside_of_image_and_raise_true(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(
            keypoints=kps + [ia.Keypoint(x=100, y=100)],
            shape=(5, 5, 3)
        )
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10

        with self.assertRaises(Exception) as context:
            _ = kpi.draw_on_image(
                image, color=[0, 255, 0], size=1, copy=True,
                raise_if_out_of_image=True)

        assert "Cannot draw keypoint" in str(context.exception)

    def test_draw_on_image_one_kp_at_bottom_right_corner(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(
            keypoints=kps + [ia.Keypoint(x=5, y=5)],
            shape=(5, 5, 3))
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10

        kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        kps_mask[2, 1] = 1
        kps_mask[4, 3] = 1

        image_kps = kpi.draw_on_image(
            image, color=[0, 255, 0], size=1, copy=True,
            raise_if_out_of_image=False)

        assert np.all(image_kps[kps_mask] == [0, 255, 0])
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    def test_draw_on_image_one_kp_at_bottom_right_corner_and_raise_true(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(
            keypoints=kps + [ia.Keypoint(x=5, y=5)],
            shape=(5, 5, 3))
        image = np.zeros((5, 5, 3), dtype=np.uint8) + 10

        kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
        kps_mask[2, 1] = 1
        kps_mask[4, 3] = 1

        with self.assertRaises(Exception) as context:
            _ = kpi.draw_on_image(
                image, color=[0, 255, 0], size=1, copy=True,
                raise_if_out_of_image=True)

        assert "Cannot draw keypoint" in str(context.exception)

    @classmethod
    def _test_clip_remove_frac(cls, func, inplace):
        item1 = ia.Keypoint(x=5, y=1)
        item2 = ia.Keypoint(x=15, y=1)
        cbaoi = ia.KeypointsOnImage([item1, item2], shape=(10, 10, 3))

        cbaoi_reduced = func(cbaoi)

        assert len(cbaoi_reduced.items) == 1
        assert np.allclose(cbaoi_reduced.to_xy_array(), [item1.xy])
        if inplace:
            assert cbaoi_reduced is cbaoi
        else:
            assert cbaoi_reduced is not cbaoi
            assert len(cbaoi.items) == 2

    def test_remove_out_of_image_fraction_(self):
        def _func(cbaoi):
            return cbaoi.remove_out_of_image_fraction_(0.6)

        self._test_clip_remove_frac(_func, True)

    def test_remove_out_of_image_fraction(self):
        def _func(cbaoi):
            return cbaoi.remove_out_of_image_fraction(0.6)

        self._test_clip_remove_frac(_func, False)

    def test_clip_out_of_image_fraction_(self):
        def _func(cbaoi):
            return cbaoi.clip_out_of_image_()

        self._test_clip_remove_frac(_func, True)

    def test_clip_out_of_image_fraction(self):
        def _func(cbaoi):
            return cbaoi.clip_out_of_image()

        self._test_clip_remove_frac(_func, False)

    def test_to_xy_array(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))

        observed = kpi.to_xy_array()

        expected = np.float32([
            [1, 2],
            [3, 4]
        ])
        assert np.allclose(observed, expected)

    def test_from_xy_array(self):
        arr = np.float32([
            [1, 2],
            [3, 4]
        ])

        kpi = ia.KeypointsOnImage.from_xy_array(arr, shape=(5, 5, 3))

        assert np.isclose(kpi.keypoints[0].x, 1)
        assert np.isclose(kpi.keypoints[0].y, 2)
        assert np.isclose(kpi.keypoints[1].x, 3)
        assert np.isclose(kpi.keypoints[1].y, 4)

    def test_fill_from_xy_array___empty_array(self):
        xy = np.zeros((0, 2), dtype=np.float32)
        kps = ia.KeypointsOnImage([], shape=(2, 2, 3))

        kps = kps.fill_from_xy_array_(xy)

        assert len(kps.keypoints) == 0

    def test_fill_from_xy_array___empty_list(self):
        xy = []
        kps = ia.KeypointsOnImage([], shape=(2, 2, 3))

        kps = kps.fill_from_xy_array_(xy)

        assert len(kps.keypoints) == 0

    def test_fill_from_xy_array___array_with_two_coords(self):
        xy = np.array([(0, 0), (1, 2)], dtype=np.float32)
        kps = ia.KeypointsOnImage([ia.Keypoint(10, 20), ia.Keypoint(30, 40)],
                                  shape=(2, 2, 3))

        kps = kps.fill_from_xy_array_(xy)

        assert len(kps.keypoints) == 2
        assert kps.keypoints[0].x == 0
        assert kps.keypoints[0].y == 0
        assert kps.keypoints[1].x == 1
        assert kps.keypoints[1].y == 2

    def test_fill_from_xy_array___list_with_two_coords(self):
        xy = [(0, 0), (1, 2)]
        kps = ia.KeypointsOnImage([ia.Keypoint(10, 20), ia.Keypoint(30, 40)],
                                  shape=(2, 2, 3))

        kps = kps.fill_from_xy_array_(xy)

        assert len(kps.keypoints) == 2
        assert kps.keypoints[0].x == 0
        assert kps.keypoints[0].y == 0
        assert kps.keypoints[1].x == 1
        assert kps.keypoints[1].y == 2

    def test_to_keypoint_image_size_1(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))

        image = kpi.to_keypoint_image(size=1)

        kps_mask = np.zeros((5, 5, 2), dtype=np.bool)
        kps_mask[2, 1, 0] = 1
        kps_mask[4, 3, 1] = 1
        assert np.all(image[kps_mask] == 255)
        assert np.all(image[~kps_mask] == 0)

    def test_to_keypoint_image_size_3(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))

        image = kpi.to_keypoint_image(size=3)

        kps_mask = np.zeros((5, 5, 2), dtype=np.bool)
        kps_mask[2-1:2+1+1, 1-1:1+1+1, 0] = 1
        kps_mask[4-1:4+1+1, 3-1:3+1+1, 1] = 1
        assert np.all(image[kps_mask] >= 128)
        assert np.all(image[~kps_mask] == 0)

    def test_from_keypoint_image(self):
        kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
        kps_image[2, 1, 0] = 255
        kps_image[4, 3, 1] = 255

        kpi2 = ia.KeypointsOnImage.from_keypoint_image(
            kps_image, nb_channels=3)

        assert kpi2.shape == (5, 5, 3)
        assert len(kpi2.keypoints) == 2
        assert kpi2.keypoints[0].y == 2.5
        assert kpi2.keypoints[0].x == 1.5
        assert kpi2.keypoints[1].y == 4.5
        assert kpi2.keypoints[1].x == 3.5

    def test_from_keypoint_image_dict_as_if_not_found_thresh_20(self):
        kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
        kps_image[2, 1, 0] = 255
        kps_image[4, 3, 1] = 10

        kpi2 = ia.KeypointsOnImage.from_keypoint_image(
            kps_image,
            if_not_found_coords={"x": -1, "y": -2},
            threshold=20,
            nb_channels=3)

        assert kpi2.shape == (5, 5, 3)
        assert len(kpi2.keypoints) == 2
        assert kpi2.keypoints[0].y == 2.5
        assert kpi2.keypoints[0].x == 1.5
        assert kpi2.keypoints[1].y == -2
        assert kpi2.keypoints[1].x == -1

    def test_from_keypoint_image_tuple_as_if_not_found_thresh_20(self):
        kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
        kps_image[2, 1, 0] = 255
        kps_image[4, 3, 1] = 10

        kpi2 = ia.KeypointsOnImage.from_keypoint_image(
            kps_image,
            if_not_found_coords=(-1, -2),
            threshold=20,
            nb_channels=3)

        assert kpi2.shape == (5, 5, 3)
        assert len(kpi2.keypoints) == 2
        assert kpi2.keypoints[0].y == 2.5
        assert kpi2.keypoints[0].x == 1.5
        assert kpi2.keypoints[1].y == -2
        assert kpi2.keypoints[1].x == -1

    def test_from_keypoint_image_none_as_if_not_found_thresh_20(self):
        kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
        kps_image[2, 1, 0] = 255
        kps_image[4, 3, 1] = 10

        kpi2 = ia.KeypointsOnImage.from_keypoint_image(
            kps_image,
            if_not_found_coords=None,
            threshold=20,
            nb_channels=3)

        assert kpi2.shape == (5, 5, 3)
        assert len(kpi2.keypoints) == 1
        assert kpi2.keypoints[0].y == 2.5
        assert kpi2.keypoints[0].x == 1.5

    def test_from_keypoint_image_bad_datatype_as_if_not_found(self):
        kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
        kps_image[2, 1, 0] = 255
        kps_image[4, 3, 1] = 10

        with self.assertRaises(Exception) as context:
            _ = ia.KeypointsOnImage.from_keypoint_image(
                kps_image,
                if_not_found_coords="exception-please",
                threshold=20,
                nb_channels=3)

        assert "Expected if_not_found_coords to be" in str(context.exception)

    @classmethod
    def _get_single_keypoint_distance_map(cls):
        # distance map for one keypoint at (x=2, y=3) on (5, 5, 3) image
        distance_map_xx = np.float32([
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]
        ])
        distance_map_yy = np.float32([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]
        ])
        distance_map = np.sqrt(
            (distance_map_xx - 2)**2
            + (distance_map_yy - 3)**2)
        return distance_map[..., np.newaxis]

    def test_to_distance_maps(self):
        kpi = ia.KeypointsOnImage(
            keypoints=[ia.Keypoint(x=2, y=3)],
            shape=(5, 5, 3))

        distance_map = kpi.to_distance_maps()

        expected = self._get_single_keypoint_distance_map()
        assert distance_map.shape == (5, 5, 1)
        assert np.allclose(distance_map, expected)

    def test_to_distance_maps_inverted(self):
        kpi = ia.KeypointsOnImage(
            keypoints=[ia.Keypoint(x=2, y=3)],
            shape=(5, 5, 3))

        distance_map = kpi.to_distance_maps(inverted=True)

        expected = self._get_single_keypoint_distance_map()
        expected_inv = np.divide(np.ones_like(expected), expected+1)
        assert distance_map.shape == (5, 5, 1)
        assert np.allclose(distance_map, expected_inv)

    @classmethod
    def _get_two_points_keypoint_distance_map(cls):
        # distance map for two keypoints at (x=2, y=3) and (x=1, y=0) on
        # (4, 4, 3) image
        #
        # Visualization of positions on (4, 4) map (X=position, 1=KP 1 is
        # closest, 2=KP 2 is closest, B=close to both):
        #
        #     [1, X, 1, 1]
        #     [1, 1, 1, B]
        #     [B, 2, 2, 2]
        #     [2, 2, X, 2]
        #
        distance_map_x = np.float32([
            [(0-1)**2, (1-1)**2, (2-1)**2, (3-1)**2],
            [(0-1)**2, (1-1)**2, (2-1)**2, (3-1)**2],
            [(0-1)**2, (1-2)**2, (2-2)**2, (3-2)**2],
            [(0-2)**2, (1-2)**2, (2-2)**2, (3-2)**2],
        ])

        distance_map_y = np.float32([
            [(0-0)**2, (0-0)**2, (0-0)**2, (0-0)**2],
            [(1-0)**2, (1-0)**2, (1-0)**2, (1-0)**2],
            [(2-0)**2, (2-3)**2, (2-3)**2, (2-3)**2],
            [(3-3)**2, (3-3)**2, (3-3)**2, (3-3)**2],
        ])

        distance_map = np.sqrt(distance_map_x + distance_map_y)
        return distance_map

    def test_to_distance_maps_two_keypoints(self):
        # TODO this test could have been done a bit better by simply splitting
        #      the distance maps, one per keypoint, considering the function
        #      returns one distance map per keypoint
        kpi = ia.KeypointsOnImage(
            keypoints=[ia.Keypoint(x=2, y=3), ia.Keypoint(x=1, y=0)],
            shape=(4, 4, 3))

        distance_map = kpi.to_distance_maps()

        expected = self._get_two_points_keypoint_distance_map()
        assert np.allclose(np.min(distance_map, axis=2), expected)

    def test_to_distance_maps_two_keypoints_inverted(self):
        kpi = ia.KeypointsOnImage(
            keypoints=[ia.Keypoint(x=2, y=3), ia.Keypoint(x=1, y=0)],
            shape=(4, 4, 3))

        distance_map_inv = kpi.to_distance_maps(inverted=True)

        expected = self._get_two_points_keypoint_distance_map()
        expected_inv = np.divide(np.ones_like(expected), expected+1)
        assert np.allclose(np.max(distance_map_inv, axis=2), expected_inv)

    @classmethod
    def _get_distance_maps_for_from_dmap_tests(cls):
        distance_map1 = np.float32([
            [2, 2, 2, 2, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 0, 1, 2],
            [2, 1, 1, 1, 2]
        ])
        distance_map2 = np.float32([
            [4, 3, 2, 2, 2],
            [4, 3, 2, 1, 1],
            [4, 3, 2, 1, 0.1],
            [4, 3, 2, 1, 1]
        ])
        distance_maps = np.concatenate([
            distance_map1[..., np.newaxis],
            distance_map2[..., np.newaxis]
        ], axis=2)
        return distance_maps

    def test_from_distance_maps(self):
        distance_maps = self._get_distance_maps_for_from_dmap_tests()

        kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps)

        assert len(kpi.keypoints) == 2
        assert kpi.keypoints[0].x == 2
        assert kpi.keypoints[0].y == 2
        assert kpi.keypoints[1].x == 4
        assert kpi.keypoints[1].y == 2
        assert kpi.shape == (4, 5)

    def test_from_distance_maps_nb_channels_4(self):
        distance_maps = self._get_distance_maps_for_from_dmap_tests()

        kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps,
                                                     nb_channels=4)

        assert len(kpi.keypoints) == 2
        assert kpi.keypoints[0].x == 2
        assert kpi.keypoints[0].y == 2
        assert kpi.keypoints[1].x == 4
        assert kpi.keypoints[1].y == 2
        assert kpi.shape == (4, 5, 4)

    def test_from_distance_maps_inverted(self):
        distance_maps = self._get_distance_maps_for_from_dmap_tests()
        distance_maps_inv = np.divide(
            np.ones_like(distance_maps),
            distance_maps+1)

        kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps_inv,
                                                     inverted=True)

        assert len(kpi.keypoints) == 2
        assert kpi.keypoints[0].x == 2
        assert kpi.keypoints[0].y == 2
        assert kpi.keypoints[1].x == 4
        assert kpi.keypoints[1].y == 2
        assert kpi.shape == (4, 5)

    def test_from_distance_maps_if_not_found_is_tuple_thresh_009(self):
        distance_maps = self._get_distance_maps_for_from_dmap_tests()

        kpi = ia.KeypointsOnImage.from_distance_maps(
            distance_maps, if_not_found_coords=(1, 1), threshold=0.09)

        assert len(kpi.keypoints) == 2
        assert kpi.keypoints[0].x == 2
        assert kpi.keypoints[0].y == 2
        assert kpi.keypoints[1].x == 1
        assert kpi.keypoints[1].y == 1
        assert kpi.shape == (4, 5)

    def test_from_distance_maps_if_not_found_is_dict_thresh_009(self):
        distance_maps = self._get_distance_maps_for_from_dmap_tests()

        kpi = ia.KeypointsOnImage.from_distance_maps(
            distance_maps,
            if_not_found_coords={"x": 1, "y": 2},
            threshold=0.09)

        assert len(kpi.keypoints) == 2
        assert kpi.keypoints[0].x == 2
        assert kpi.keypoints[0].y == 2
        assert kpi.keypoints[1].x == 1
        assert kpi.keypoints[1].y == 2
        assert kpi.shape == (4, 5)

    def test_from_distance_maps_if_not_found_is_none_thresh_009(self):
        distance_maps = self._get_distance_maps_for_from_dmap_tests()

        kpi = ia.KeypointsOnImage.from_distance_maps(
            distance_maps,
            if_not_found_coords=None,
            threshold=0.09)

        assert len(kpi.keypoints) == 1
        assert kpi.keypoints[0].x == 2
        assert kpi.keypoints[0].y == 2
        assert kpi.shape == (4, 5)

    def test_from_distance_maps_bad_datatype_for_if_not_found(self):
        distance_maps = self._get_distance_maps_for_from_dmap_tests()

        with self.assertRaises(Exception) as context:
            _ = ia.KeypointsOnImage.from_distance_maps(
                distance_maps,
                if_not_found_coords=False,
                threshold=0.09)

        assert "Expected if_not_found_coords to be" in str(context.exception)

    def test_to_keypoints_on_image(self):
        kps = ia.KeypointsOnImage([ia.Keypoint(0, 0), ia.Keypoint(1, 2)],
                                  shape=(1, 2, 3))
        kps.deepcopy = mock.MagicMock()
        kps.deepcopy.return_value = "foo"

        kps_cp = kps.to_keypoints_on_image()

        assert kps.deepcopy.call_count == 1
        assert kps_cp == "foo"

    def test_invert_to_keypoints_on_image_(self):
        kps1 = ia.KeypointsOnImage([ia.Keypoint(0, 0), ia.Keypoint(1, 2)],
                                   shape=(2, 3, 4))
        kps2 = ia.KeypointsOnImage([ia.Keypoint(10, 10), ia.Keypoint(11, 12)],
                                   shape=(3, 4, 5))

        kps3 = kps1.invert_to_keypoints_on_image_(kps2)

        assert kps3 is not kps2
        assert kps3.shape == (3, 4, 5)
        assert kps3.keypoints[0].x == 10
        assert kps3.keypoints[0].y == 10
        assert kps3.keypoints[1].x == 11
        assert kps3.keypoints[1].y == 12

    def test_copy(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))

        kpi2 = kpi.copy()

        assert kpi2.keypoints[0].x == 1
        assert kpi2.keypoints[0].y == 2
        assert kpi2.keypoints[1].x == 3
        assert kpi2.keypoints[1].y == 4

        kps[0].x = 100

        assert kpi2.keypoints[0].x == 100
        assert kpi2.keypoints[0].y == 2
        assert kpi2.keypoints[1].x == 3
        assert kpi2.keypoints[1].y == 4

    def test_copy_keypoints_set(self):
        kp1 = ia.Keypoint(x=1, y=2)
        kp2 = ia.Keypoint(x=3, y=4)
        kp3 = ia.Keypoint(x=5, y=6)
        kpsoi = ia.KeypointsOnImage([kp1, kp2], shape=(40, 50, 3))

        kpsoi_copy = kpsoi.copy(keypoints=[kp3])

        assert kpsoi_copy is not kpsoi
        assert kpsoi_copy.shape == (40, 50, 3)
        assert kpsoi_copy.keypoints == [kp3]

    def test_copy_shape_set(self):
        kp1 = ia.Keypoint(x=1, y=2)
        kp2 = ia.Keypoint(x=3, y=4)
        kpsoi = ia.KeypointsOnImage([kp1, kp2], shape=(40, 50, 3))

        kpsoi_copy = kpsoi.copy(shape=(40+1, 50+1, 3))

        assert kpsoi_copy is not kpsoi
        assert kpsoi_copy.shape == (40+1, 50+1, 3)
        assert kpsoi_copy.keypoints == [kp1, kp2]

    def test_deepcopy(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))

        kpi2 = kpi.deepcopy()

        assert kpi2.keypoints[0].x == 1
        assert kpi2.keypoints[0].y == 2
        assert kpi2.keypoints[1].x == 3
        assert kpi2.keypoints[1].y == 4

        kps[0].x = 100

        assert kpi2.keypoints[0].x == 1
        assert kpi2.keypoints[0].y == 2
        assert kpi2.keypoints[1].x == 3
        assert kpi2.keypoints[1].y == 4

    def test_deepcopy_keypoints_set(self):
        kp1 = ia.Keypoint(x=1, y=2)
        kp2 = ia.Keypoint(x=3, y=4)
        kp3 = ia.Keypoint(x=5, y=6)
        kpsoi = ia.KeypointsOnImage([kp1, kp2], shape=(40, 50, 3))

        kpsoi_copy = kpsoi.deepcopy(keypoints=[kp3])

        assert kpsoi_copy is not kpsoi
        assert kpsoi_copy.shape == (40, 50, 3)
        assert kpsoi_copy.keypoints == [kp3]

    def test_deepcopy_shape_set(self):
        kp1 = ia.Keypoint(x=1, y=2)
        kp2 = ia.Keypoint(x=3, y=4)
        kpsoi = ia.KeypointsOnImage([kp1, kp2], shape=(40, 50, 3))

        kpsoi_copy = kpsoi.deepcopy(shape=(40+1, 50+1, 3))

        assert kpsoi_copy is not kpsoi
        assert kpsoi_copy.shape == (40+1, 50+1, 3)
        assert len(kpsoi_copy.keypoints) == 2
        assert kpsoi_copy.keypoints[0].coords_almost_equals(kp1)
        assert kpsoi_copy.keypoints[1].coords_almost_equals(kp2)

    def test___getitem__(self):
        cbas = [
            ia.Keypoint(x=1, y=2),
            ia.Keypoint(x=2, y=3)
        ]
        cbasoi = ia.KeypointsOnImage(cbas, shape=(3, 4, 3))

        assert cbasoi[0] is cbas[0]
        assert cbasoi[1] is cbas[1]
        assert cbasoi[0:2] == cbas

    def test___iter__(self):
        cbas = [ia.Keypoint(x=1, y=2),
                ia.Keypoint(x=3, y=4)]
        cbasoi = ia.KeypointsOnImage(cbas, shape=(40, 50, 3))

        for i, cba in enumerate(cbasoi):
            assert cba is cbas[i]

    def test___iter___empty(self):
        cbasoi = ia.KeypointsOnImage([], shape=(40, 50, 3))
        i = 0
        for _cba in cbasoi:
            i += 1
        assert i == 0

    def test___len__(self):
        cbas = [ia.Keypoint(x=1, y=2),
                ia.Keypoint(x=3, y=4)]
        cbasoi = ia.KeypointsOnImage(cbas, shape=(40, 50, 3))
        assert len(cbasoi) == 2

    def test_string_conversion(self):
        kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
        expected = (
            "KeypointsOnImage(["
            "Keypoint(x=1.00000000, y=2.00000000), "
            "Keypoint(x=3.00000000, y=4.00000000)"
            "], shape=(5, 5, 3)"
            ")"
        )
        assert (
            kpi.__repr__()
            == kpi.__str__()
            == expected
        )
