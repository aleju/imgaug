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
import six.moves as sm

import imgaug as ia


# TODO add tests for:
#      hooks is_activated
#      hooks is_propagating
#      hooks preprocess
#      hooks postprocess
#      HeatmapsOnImage.__init__()
#      HeatmapsOnImage.get_arr()
#      HeatmapsOnImage.to_uint8()
#      HeatmapsOnImage.from_0to1()
#      HeatmapsOnImage.copy()
#      HeatmapsOnImage.deepcopy()


class TestHeatmapsOnImage_draw(unittest.TestCase):
    def test_basic_functionality(self):
        heatmaps_arr = np.float32([
            [0.5, 0.0, 0.0, 0.5],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.5, 0.0, 0.0, 0.5],
        ])
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(4, 4, 3))

        heatmaps_drawn = heatmaps.draw()[0]
        assert heatmaps_drawn.shape == (4, 4, 3)
        v1 = heatmaps_drawn[0, 1]
        v2 = heatmaps_drawn[0, 0]
        v3 = heatmaps_drawn[1, 1]

        v1_coords = [(0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1),
                     (3, 2)]
        v2_coords = [(0, 0), (0, 3), (3, 0), (3, 3)]
        v3_coords = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for y, x in v1_coords:
            assert np.allclose(heatmaps_drawn[y, x], v1)

        for y, x in v2_coords:
            assert np.allclose(heatmaps_drawn[y, x], v2)

        for y, x in v3_coords:
            assert np.allclose(heatmaps_drawn[y, x], v3)

    def test_use_size_arg_with_different_shape_than_heatmap_arr_shape(self):
        # size differs from heatmap array size
        heatmaps_arr = np.float32([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(2, 2, 3))

        heatmaps_drawn = heatmaps.draw(size=(4, 4))[0]
        assert heatmaps_drawn.shape == (4, 4, 3)
        v1 = heatmaps_drawn[0, 0]
        v2 = heatmaps_drawn[0, -1]

        for y in sm.xrange(4):
            for x in sm.xrange(2):
                assert np.allclose(heatmaps_drawn[y, x], v1)

        for y in sm.xrange(4):
            for x in sm.xrange(2, 4):
                assert np.allclose(heatmaps_drawn[y, x], v2)


# TODO test other cmaps
class TestHeatmapsOnImage_draw_on_image(unittest.TestCase):
    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        return ia.HeatmapsOnImage(heatmaps_arr, shape=(2, 2, 3))

    def test_cmap_is_none(self):
        heatmaps = self.heatmaps

        image = np.uint8([
            [0, 0, 0, 255],
            [0, 0, 0, 255],
            [0, 0, 0, 255],
            [0, 0, 0, 255]
        ])
        image = np.tile(image[..., np.newaxis], (1, 1, 3))

        heatmaps_drawn = heatmaps.draw_on_image(image, alpha=0.5, cmap=None)[0]
        assert heatmaps_drawn.shape == (4, 4, 3)
        assert np.all(heatmaps_drawn[0:4, 0:2, :] == 0)
        assert (
            np.all(heatmaps_drawn[0:4, 2:3, :] == 128)
            or np.all(heatmaps_drawn[0:4, 2:3, :] == 127))
        assert (
            np.all(heatmaps_drawn[0:4, 3:4, :] == 255)
            or np.all(heatmaps_drawn[0:4, 3:4, :] == 254))

    def test_cmap_is_none_and_resize_is_image(self):
        heatmaps = self.heatmaps

        image = np.uint8([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        image = np.tile(image[..., np.newaxis], (1, 1, 3))

        heatmaps_drawn = heatmaps.draw_on_image(
            image, alpha=0.5, resize="image", cmap=None)[0]
        assert heatmaps_drawn.shape == (2, 2, 3)
        assert np.all(heatmaps_drawn[0:2, 0, :] == 0)
        assert (
            np.all(heatmaps_drawn[0:2, 1, :] == 128)
            or np.all(heatmaps_drawn[0:2, 1, :] == 127))


class TestHeatmapsOnImage_invert(unittest.TestCase):
    @property
    def heatmaps_arr(self):
        return np.float32([
            [0.0, 5.0, 10.0],
            [-1.0, -2.0, 7.5]
        ])

    @property
    def expected_arr(self):
        return np.float32([
            [8.0, 3.0, -2.0],
            [9.0, 10.0, 0.5]
        ])

    def test_with_2d_input_array(self):
        # (H, W)
        heatmaps_arr = self.heatmaps_arr
        expected = self.expected_arr
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr,
                                      shape=(2, 3),
                                      min_value=-2.0,
                                      max_value=10.0)
        assert np.allclose(heatmaps.get_arr(), heatmaps_arr)
        assert np.allclose(heatmaps.invert().get_arr(), expected)

    def test_with_3d_input_array(self):
        # (H, W, 1)
        heatmaps_arr = self.heatmaps_arr
        expected = self.expected_arr
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr[..., np.newaxis],
                                      shape=(2, 3),
                                      min_value=-2.0,
                                      max_value=10.0)
        assert np.allclose(heatmaps.get_arr(),
                           heatmaps_arr[..., np.newaxis])
        assert np.allclose(heatmaps.invert().get_arr(),
                           expected[..., np.newaxis])


class TestHeatmapsOnImage_pad(unittest.TestCase):
    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        return ia.HeatmapsOnImage(heatmaps_arr, shape=(2, 2, 3))

    def test_defaults(self):
        heatmaps = self.heatmaps
        heatmaps_padded = heatmaps.pad(top=1, right=2, bottom=3, left=4)
        assert heatmaps_padded.arr_0to1.shape == (2+(1+3), 2+(4+2), 1)
        assert np.allclose(
            heatmaps_padded.arr_0to1[:, :, 0],
            np.float32([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ])
        )

    def test_mode_constant_with_cval_050(self):
        heatmaps = self.heatmaps
        heatmaps_padded = heatmaps.pad(top=1, right=2, bottom=3, left=4,
                                       cval=0.5)
        assert heatmaps_padded.arr_0to1.shape == (2+(1+3), 2+(4+2), 1)
        assert np.allclose(
            heatmaps_padded.arr_0to1[:, :, 0],
            np.float32([
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.0, 1.0, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            ])
        )

    def test_mode_edge(self):
        heatmaps = self.heatmaps
        heatmaps_padded = heatmaps.pad(top=1, right=2, bottom=3, left=4,
                                       mode="edge")
        assert heatmaps_padded.arr_0to1.shape == (2+(1+3), 2+(4+2), 1)
        assert np.allclose(
            heatmaps_padded.arr_0to1[:, :, 0],
            np.float32([
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            ])
        )


class TestHeatmapsOnImage_pad_to_aspect_ratio(unittest.TestCase):
    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ])
        return ia.HeatmapsOnImage(heatmaps_arr, shape=(2, 2, 3))

    def test_square_ratio_with_default_mode_and_cval(self):
        heatmaps = self.heatmaps
        heatmaps_padded = heatmaps.pad_to_aspect_ratio(1.0)
        assert heatmaps_padded.arr_0to1.shape == (3, 3, 1)
        assert np.allclose(
            heatmaps_padded.arr_0to1[:, :, 0],
            np.float32([
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0]
            ])
        )

    def test_square_ratio_with_cval_050(self):
        heatmaps = self.heatmaps
        heatmaps_padded = heatmaps.pad_to_aspect_ratio(1.0, cval=0.5)
        assert heatmaps_padded.arr_0to1.shape == (3, 3, 1)
        assert np.allclose(
            heatmaps_padded.arr_0to1[:, :, 0],
            np.float32([
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.5]
            ])
        )

    def test_square_ratio_with_edge_mode(self):
        heatmaps = self.heatmaps
        heatmaps_padded = heatmaps.pad_to_aspect_ratio(1.0, mode="edge")
        assert heatmaps_padded.arr_0to1.shape == (3, 3, 1)
        assert np.allclose(
            heatmaps_padded.arr_0to1[:, :, 0],
            np.float32([
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0]
            ])
        )

    def test_wider_than_high_ratio_with_cval_010(self):
        heatmaps = self.heatmaps
        heatmaps_padded = heatmaps.pad_to_aspect_ratio(2.0, cval=0.1)
        assert heatmaps_padded.arr_0to1.shape == (2, 4, 1)
        assert np.allclose(
            heatmaps_padded.arr_0to1[:, :, 0],
            np.float32([
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 1.0, 0.1]
            ])
        )

    def test_higher_than_wide_ratio_with_cval_010(self):
        heatmaps = self.heatmaps
        heatmaps_padded = heatmaps.pad_to_aspect_ratio(0.25, cval=0.1)
        assert heatmaps_padded.arr_0to1.shape == (12, 3, 1)
        assert np.allclose(
            heatmaps_padded.arr_0to1[:, :, 0],
            np.float32([
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1]
            ])
        )


class TestHeatmapsOnImage_avg_pool(unittest.TestCase):
    def test_with_kernel_size_2(self):
        heatmaps_arr = np.float32([
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5, 1.0]
        ])
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(4, 4, 3))

        heatmaps_pooled = heatmaps.avg_pool(2)
        assert heatmaps_pooled.arr_0to1.shape == (2, 2, 1)
        assert np.allclose(
            heatmaps_pooled.arr_0to1[:, :, 0],
            np.float32([[0.0, 0.75],
                        [0.0, 0.75]])
        )

class TestHeatmapsOnImage_max_pool(unittest.TestCase):
    def test_with_kernel_size_2(self):
        heatmaps_arr = np.float32([
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5, 1.0]
        ])
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(4, 4, 3))

        heatmaps_pooled = heatmaps.max_pool(2)
        assert heatmaps_pooled.arr_0to1.shape == (2, 2, 1)
        assert np.allclose(
            heatmaps_pooled.arr_0to1[:, :, 0],
            np.float32([[0.0, 1.0],
                        [0.0, 1.0]])
        )


class TestHeatmapsOnImage_resize(unittest.TestCase):
    def test_resize_to_exact_shape(self):
        heatmaps_arr = np.float32([
            [0.0, 1.0]
        ])
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(4, 4, 3))

        heatmaps_scaled = heatmaps.resize((4, 4), interpolation="nearest")
        assert heatmaps_scaled.arr_0to1.shape == (4, 4, 1)
        assert heatmaps_scaled.arr_0to1.dtype.name == "float32"
        assert np.allclose(
            heatmaps_scaled.arr_0to1[:, :, 0],
            np.float32([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0]
            ])
        )

    def test_resize_to_twice_the_size(self):
        heatmaps_arr = np.float32([
            [0.0, 1.0]
        ])
        heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(4, 4, 3))

        heatmaps_scaled = heatmaps.resize(2.0, interpolation="nearest")
        assert heatmaps_scaled.arr_0to1.shape == (2, 4, 1)
        assert heatmaps_scaled.arr_0to1.dtype.name == "float32"
        assert np.allclose(
            heatmaps_scaled.arr_0to1[:, :, 0],
            np.float32([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0]
            ])
        )


class TestHeatmapsOnImage_from_uint8(unittest.TestCase):
    def test_3d_uint8_array(self):
        hm = ia.HeatmapsOnImage.from_uint8(
            np.uint8([
                [0, 128, 255],
                [255, 128, 0]
            ])[..., np.newaxis],
            (20, 30, 3)
        )
        assert hm.shape == (20, 30, 3)
        assert hm.arr_0to1.shape == (2, 3, 1)
        assert np.allclose(hm.arr_0to1[..., 0], np.float32([
            [0, 128/255, 1.0],
            [1.0, 128/255, 0]
        ]))

    def test_2d_uint8_array(self):
        hm = ia.HeatmapsOnImage.from_uint8(
            np.uint8([
                [0, 128, 255],
                [255, 128, 0]
            ]),
            (20, 30, 3)
        )
        assert hm.shape == (20, 30, 3)
        assert hm.arr_0to1.shape == (2, 3, 1)
        assert np.allclose(hm.arr_0to1[..., 0], np.float32([
            [0, 128/255, 1.0],
            [1.0, 128/255, 0]
        ]))

    def test_min_value_and_max_value(self):
        # min_value, max_value
        hm = ia.HeatmapsOnImage.from_uint8(
            np.uint8([
                [0, 128, 255],
                [255, 128, 0]
            ])[..., np.newaxis],
            (20, 30, 3),
            min_value=-1.0,
            max_value=2.0
        )
        assert hm.shape == (20, 30, 3)
        assert hm.arr_0to1.shape == (2, 3, 1)
        assert np.allclose(hm.arr_0to1[..., 0], np.float32([
            [0, 128/255, 1.0],
            [1.0, 128/255, 0]
        ]))
        assert np.allclose(hm.min_value, -1.0)
        assert np.allclose(hm.max_value, 2.0)


class TestHeatmapsOnImage_change_normalization(unittest.TestCase):
    def test_increase_max_value(self):
        # (0.0, 1.0) -> (0.0, 2.0)
        arr = np.float32([
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 0.0]
        ])

        observed = ia.HeatmapsOnImage.change_normalization(
            arr, (0.0, 1.0), (0.0, 2.0))

        expected = np.float32([
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0]
        ])
        assert np.allclose(observed, expected)

    def test_decrease_min_and_max_value(self):
        # (0.0, 1.0) -> (-1.0, 0.0)
        arr = np.float32([
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 0.0]
        ])

        observed = ia.HeatmapsOnImage.change_normalization(
            arr, (0.0, 1.0), (-1.0, 0.0))

        expected = np.float32([
            [-1.0, -0.5, 0.0],
            [0.0, -0.5, -1.0]
        ])
        assert np.allclose(observed, expected)

    def test_increase_min_and_max_value__non_standard_source(self):
        # (-1.0, 1.0) -> (1.0, 3.0)
        arr = np.float32([
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0]
        ])

        observed = ia.HeatmapsOnImage.change_normalization(
            arr, (-1.0, 1.0), (1.0, 3.0))

        expected = np.float32([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0]
        ])
        assert np.allclose(observed, expected)

    def test_value_ranges_given_as_heatmaps_on_image(self):
        # (-1.0, 1.0) -> (1.0, 3.0)
        # value ranges given as HeatmapsOnImage
        arr = np.float32([
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0]
        ])
        source = ia.HeatmapsOnImage(
            np.float32([[0.0]]), min_value=-1.0, max_value=1.0, shape=(1, 1, 3))
        target = ia.HeatmapsOnImage(
            np.float32([[1.0]]), min_value=1.0, max_value=3.0, shape=(1, 1, 3))

        observed = ia.HeatmapsOnImage.change_normalization(arr, source, target)

        expected = np.float32([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0]
        ])
        assert np.allclose(observed, expected)
