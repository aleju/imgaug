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
import six.moves as sm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.augmenters import blend
from imgaug.testutils import (
    keypoints_equal, reseed, assert_cbaois_equal, shift_cbaoi)
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class Test_blend_alpha(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_alpha_is_1(self):
        img_fg = np.full((3, 3, 1), 0, dtype=np.uint8)
        img_bg = np.full((3, 3, 1), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3, 1)
        assert np.all(img_blend == 0)

    def test_alpha_is_1_2d_arrays(self):
        img_fg = np.full((3, 3), 0, dtype=np.uint8)
        img_bg = np.full((3, 3), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3)
        assert np.all(img_blend == 0)

    def test_alpha_is_0(self):
        img_fg = np.full((3, 3, 1), 0, dtype=np.uint8)
        img_bg = np.full((3, 3, 1), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3, 1)
        assert np.all(img_blend == 255)

    def test_alpha_is_0_2d_arrays(self):
        img_fg = np.full((3, 3), 0, dtype=np.uint8)
        img_bg = np.full((3, 3), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3)
        assert np.all(img_blend == 255)

    def test_alpha_is_030(self):
        img_fg = np.full((3, 3, 1), 0, dtype=np.uint8)
        img_bg = np.full((3, 3, 1), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.3, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3, 1)
        assert np.allclose(img_blend, 0.7*255, atol=1.01, rtol=0)

    def test_alpha_is_030_2d_arrays(self):
        img_fg = np.full((3, 3), 0, dtype=np.uint8)
        img_bg = np.full((3, 3), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.3, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3)
        assert np.allclose(img_blend, 0.7*255, atol=1.01, rtol=0)

    def test_channelwise_alpha(self):
        img_fg = np.full((3, 3, 2), 0, dtype=np.uint8)
        img_bg = np.full((3, 3, 2), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, [1.0, 0.0], eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3, 2)
        assert np.all(img_blend[:, :, 0] == 0)
        assert np.all(img_blend[:, :, 1] == 255)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image_fg = np.full(shape, 0, dtype=np.uint8)
                image_bg = np.full(shape, 255, dtype=np.uint8)

                image_aug = blend.blend_alpha(image_fg, image_bg, 1.0)

                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image_fg = np.full(shape, 0, dtype=np.uint8)
                image_bg = np.full(shape, 255, dtype=np.uint8)

                image_aug = blend.blend_alpha(image_fg, image_bg, 1.0)

                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_other_dtypes_bool(self):
        img_fg = np.full((3, 3, 1), 0, dtype=bool)
        img_bg = np.full((3, 3, 1), 1, dtype=bool)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.3, eps=0)
        assert img_blend.dtype.name == "bool"
        assert img_blend.shape == (3, 3, 1)
        assert np.all(img_blend == 1)

    def test_other_dtypes_bool_2d_arrays(self):
        img_fg = np.full((3, 3), 0, dtype=bool)
        img_bg = np.full((3, 3), 1, dtype=bool)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.3, eps=0)
        assert img_blend.dtype.name == "bool"
        assert img_blend.shape == (3, 3)
        assert np.all(img_blend == 1)

    # TODO split this up into multiple tests
    def test_other_dtypes_uint_int(self):
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                dtype = np.dtype(dtype)

                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                values = [
                    (0, 0),
                    (0, 10),
                    (10, 20),
                    (min_value, min_value),
                    (max_value, max_value),
                    (min_value, max_value),
                    (min_value, int(center_value)),
                    (int(center_value), max_value),
                    (int(center_value + 0.20 * max_value), max_value),
                    (int(center_value + 0.27 * max_value), max_value),
                    (int(center_value + 0.40 * max_value), max_value),
                    (min_value, 0),
                    (0, max_value)
                ]
                values = values + [(v2, v1) for v1, v2 in values]

                for v1, v2 in values:
                    v1_scalar = np.full((), v1, dtype=dtype)
                    v2_scalar = np.full((), v2, dtype=dtype)
                    
                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert np.all(img_blend == v1_scalar)

                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 0.99, eps=0.1)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert np.all(img_blend == v1_scalar)

                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert np.all(img_blend == v2_scalar)

                    # TODO this test breaks for numpy <1.15 -- why?
                    for c in sm.xrange(3):
                        img_fg = np.full((3, 3, c), v1, dtype=dtype)
                        img_bg = np.full((3, 3, c), v2, dtype=dtype)
                        img_blend = blend.blend_alpha(
                            img_fg, img_bg, 0.75, eps=0)
                        assert img_blend.dtype.name == np.dtype(dtype)
                        assert img_blend.shape == (3, 3, c)
                        for ci in sm.xrange(c):
                            v_blend = min(
                                max(
                                    int(
                                        0.75*np.float128(v1)
                                        + 0.25*np.float128(v2)
                                    ),
                                    min_value
                                ),
                                max_value)
                            diff = (
                                v_blend - img_blend
                                if v_blend > img_blend[0, 0, ci]
                                else img_blend - v_blend)
                            assert np.all(diff < 1.01)

                    img_fg = np.full((3, 3, 2), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 2), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 0.75, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 2)
                    v_blend = min(
                        max(
                            int(
                                0.75 * np.float128(v1)
                                + 0.25 * np.float128(v2)
                            ),
                            min_value
                        ),
                        max_value)
                    diff = (
                        v_blend - img_blend
                        if v_blend > img_blend[0, 0, 0]
                        else img_blend - v_blend
                    )
                    assert np.all(diff < 1.01)

                    img_fg = np.full((3, 3, 2), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 2), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, [1.0, 0.0], eps=0.1)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 2)
                    assert np.all(img_blend[:, :, 0] == v1_scalar)
                    assert np.all(img_blend[:, :, 1] == v2_scalar)

                    # elementwise, alphas.shape = (1, 2)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2), dtype=np.float64)
                    alphas[:, :] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert np.all(img_blend[0, 0, :] == v1_scalar)
                    assert np.all(img_blend[0, 1, :] == v2_scalar)

                    # elementwise, alphas.shape = (1, 2, 1)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2, 1), dtype=np.float64)
                    alphas[:, :, 0] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert np.all(img_blend[0, 0, :] == v1_scalar)
                    assert np.all(img_blend[0, 1, :] == v2_scalar)

                    # elementwise, alphas.shape = (1, 2, 3)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2, 3), dtype=np.float64)
                    alphas[:, :, 0] = [1.0, 0.0]
                    alphas[:, :, 1] = [0.0, 1.0]
                    alphas[:, :, 2] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert np.all(img_blend[0, 0, [0, 2]] == v1_scalar)
                    assert np.all(img_blend[0, 1, [0, 2]] == v2_scalar)
                    assert np.all(img_blend[0, 0, 1] == v2_scalar)
                    assert np.all(img_blend[0, 1, 1] == v1_scalar)

    # TODO split this up into multiple tests
    def test_other_dtypes_float(self):
        dtypes = ["float16", "float32", "float64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                dtype = np.dtype(dtype)

                def _allclose(a, b):
                    atol = 1e-4 if dtype == np.float16 else 1e-8
                    return np.allclose(a, b, atol=atol, rtol=0)

                isize = np.dtype(dtype).itemsize
                max_value = 1000 ** (isize - 1)
                min_value = -max_value
                center_value = 0
                values = [
                    (0, 0),
                    (0, 10),
                    (10, 20),
                    (min_value, min_value),
                    (max_value, max_value),
                    (min_value, max_value),
                    (min_value, center_value),
                    (center_value, max_value),
                    (center_value + 0.20 * max_value, max_value),
                    (center_value + 0.27 * max_value, max_value),
                    (center_value + 0.40 * max_value, max_value),
                    (min_value, 0),
                    (0, max_value)
                ]
                values = values + [(v2, v1) for v1, v2 in values]

                max_float_dt = np.float128

                for v1, v2 in values:
                    v1_scalar = np.full((), v1, dtype=dtype)
                    v2_scalar = np.full((), v2, dtype=dtype)

                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert _allclose(img_blend, max_float_dt(v1))

                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, 0.99, eps=0.1)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert _allclose(img_blend, max_float_dt(v1))

                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert _allclose(img_blend, max_float_dt(v2))

                    for c in sm.xrange(3):
                        img_fg = np.full((3, 3, c), v1, dtype=dtype)
                        img_bg = np.full((3, 3, c), v2, dtype=dtype)
                        img_blend = blend.blend_alpha(
                            img_fg, img_bg, 0.75, eps=0)
                        assert img_blend.dtype.name == np.dtype(dtype)
                        assert img_blend.shape == (3, 3, c)
                        assert _allclose(
                            img_blend,
                            0.75*max_float_dt(v1) + 0.25*max_float_dt(v2)
                        )

                    img_fg = np.full((3, 3, 2), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 2), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, [1.0, 0.0], eps=0.1)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 2)
                    assert _allclose(img_blend[:, :, 0], max_float_dt(v1))
                    assert _allclose(img_blend[:, :, 1], max_float_dt(v2))

                    # elementwise, alphas.shape = (1, 2)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2), dtype=np.float64)
                    alphas[:, :] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert _allclose(img_blend[0, 0, :], v1_scalar)
                    assert _allclose(img_blend[0, 1, :], v2_scalar)

                    # elementwise, alphas.shape = (1, 2, 1)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2, 1), dtype=np.float64)
                    alphas[:, :, 0] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert _allclose(img_blend[0, 0, :], v1_scalar)
                    assert _allclose(img_blend[0, 1, :], v2_scalar)

                    # elementwise, alphas.shape = (1, 2, 3)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2, 3), dtype=np.float64)
                    alphas[:, :, 0] = [1.0, 0.0]
                    alphas[:, :, 1] = [0.0, 1.0]
                    alphas[:, :, 2] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert _allclose(img_blend[0, 0, [0, 2]], v1_scalar)
                    assert _allclose(img_blend[0, 1, [0, 2]], v2_scalar)
                    assert _allclose(img_blend[0, 0, 1], v2_scalar)
                    assert _allclose(img_blend[0, 1, 1], v1_scalar)


class TestAlpha(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.zeros((3, 3, 1), dtype=np.uint8)
        return base_img

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 1.0, 1.0]])
        return HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def heatmaps_r1(self):
        heatmaps_arr_r1 = np.float32([[0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0]])
        return HeatmapsOnImage(heatmaps_arr_r1, shape=(3, 3, 3))

    @property
    def heatmaps_l1(self):
        heatmaps_arr_l1 = np.float32([[0.0, 1.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [1.0, 1.0, 0.0]])
        return HeatmapsOnImage(heatmaps_arr_l1, shape=(3, 3, 3))

    @property
    def segmaps(self):
        segmaps_arr = np.int32([[0, 0, 1],
                                [0, 0, 1],
                                [0, 1, 1]])
        return SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps_r1(self):
        segmaps_arr_r1 = np.int32([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 1]])
        return SegmentationMapsOnImage(segmaps_arr_r1, shape=(3, 3, 3))

    @property
    def segmaps_l1(self):
        segmaps_arr_l1 = np.int32([[0, 1, 0],
                                   [0, 1, 0],
                                   [1, 1, 0]])
        return SegmentationMapsOnImage(segmaps_arr_l1, shape=(3, 3, 3))

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=5, y=10), ia.Keypoint(x=6, y=11)]
        return ia.KeypointsOnImage(kps, shape=(20, 20, 3))

    @property
    def psoi(self):
        ps = [ia.Polygon([(5, 5), (10, 5), (10, 10)])]
        return ia.PolygonsOnImage(ps, shape=(20, 20, 3))

    @property
    def lsoi(self):
        lss = [ia.LineString([(5, 5), (10, 5), (10, 10)])]
        return ia.LineStringsOnImage(lss, shape=(20, 20, 3))

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)]
        return ia.BoundingBoxesOnImage(bbs, shape=(20, 20, 3))

    def test_images_factor_is_1(self):
        aug = iaa.Alpha(1, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = np.round(self.image + 10).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_heatmaps_factor_is_1_with_affines_and_per_channel(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                aug = iaa.Alpha(
                    1,
                    iaa.Affine(translate_px={"x": 1}),
                    iaa.Affine(translate_px={"x": -1}),
                    per_channel=per_channel)
                observed = aug.augment_heatmaps([self.heatmaps])[0]
                assert observed.shape == self.heatmaps.shape
                assert 0 - 1e-6 < self.heatmaps.min_value < 0 + 1e-6
                assert 1 - 1e-6 < self.heatmaps.max_value < 1 + 1e-6
                assert np.allclose(observed.get_arr(),
                                   self.heatmaps_r1.get_arr())

    def test_segmaps_factor_is_1_with_affines_and_per_channel(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                aug = iaa.Alpha(
                    1,
                    iaa.Affine(translate_px={"x": 1}),
                    iaa.Affine(translate_px={"x": -1}),
                    per_channel=per_channel)
                observed = aug.augment_segmentation_maps([self.segmaps])[0]
                assert observed.shape == self.segmaps.shape
                assert np.array_equal(observed.get_arr(),
                                      self.segmaps_r1.get_arr())

    def test_images_factor_is_0(self):
        aug = iaa.Alpha(0, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = np.round(self.image + 20).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_heatmaps_factor_is_0_with_affines_and_per_channel(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                aug = iaa.Alpha(
                    0,
                    iaa.Affine(translate_px={"x": 1}),
                    iaa.Affine(translate_px={"x": -1}),
                    per_channel=per_channel)
                observed = aug.augment_heatmaps([self.heatmaps])[0]
                assert observed.shape == self.heatmaps.shape
                assert 0 - 1e-6 < self.heatmaps.min_value < 0 + 1e-6
                assert 1 - 1e-6 < self.heatmaps.max_value < 1 + 1e-6
                assert np.allclose(observed.get_arr(),
                                   self.heatmaps_l1.get_arr())

    def test_segmaps_factor_is_0_with_affines_and_per_channel(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                aug = iaa.Alpha(
                    0,
                    iaa.Affine(translate_px={"x": 1}),
                    iaa.Affine(translate_px={"x": -1}),
                    per_channel=per_channel)
                observed = aug.augment_segmentation_maps([self.segmaps])[0]
                assert observed.shape == self.segmaps.shape
                assert np.array_equal(observed.get_arr(),
                                      self.segmaps_l1.get_arr())

    def test_images_factor_is_075(self):
        aug = iaa.Alpha(0.75, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = np.round(
            self.image
            + 0.75 * 10
            + 0.25 * 20
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_075_first_branch_is_none(self):
        aug = iaa.Alpha(0.75, None, iaa.Add(20))
        observed = aug.augment_image(self.image + 10)
        expected = np.round(
            self.image
            + 0.75 * 10
            + 0.25 * (10 + 20)
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_075_second_branch_is_none(self):
        aug = iaa.Alpha(0.75, iaa.Add(10), None)
        observed = aug.augment_image(self.image + 10)
        expected = np.round(
            self.image
            + 0.75 * (10 + 10)
            + 0.25 * 10
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_tuple(self):
        image = np.zeros((1, 2, 1), dtype=np.uint8)
        nb_iterations = 1000
        aug = iaa.Alpha((0.0, 1.0), iaa.Add(10), iaa.Add(110))
        values = []
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_image(image)
            observed_val = np.round(np.average(observed)) - 10
            values.append(observed_val / 100)

        nb_bins = 5
        hist, _ = np.histogram(values, bins=nb_bins, range=(0.0, 1.0),
                               density=False)
        density_expected = 1.0/nb_bins
        density_tolerance = 0.05
        for nb_samples in hist:
            density = nb_samples / nb_iterations
            assert np.isclose(density, density_expected,
                              rtol=0, atol=density_tolerance)

    def test_bad_datatype_for_factor_fails(self):
        got_exception = False
        try:
            _ = iaa.Alpha(False, iaa.Add(10), None)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_images_with_per_channel_in_both_alpha_and_child(self):
        image = np.zeros((1, 1, 1000), dtype=np.uint8)
        aug = iaa.Alpha(
            1.0,
            iaa.Add((0, 100), per_channel=True),
            None,
            per_channel=True)
        observed = aug.augment_image(image)
        uq = np.unique(observed)
        assert len(uq) > 1
        assert np.max(observed) > 80
        assert np.min(observed) < 20

    def test_images_with_per_channel_in_alpha_and_tuple_as_factor(self):
        image = np.zeros((1, 1, 1000), dtype=np.uint8)
        aug = iaa.Alpha(
            (0.0, 1.0),
            iaa.Add(100),
            None,
            per_channel=True)
        observed = aug.augment_image(image)
        uq = np.unique(observed)
        assert len(uq) > 1
        assert np.max(observed) > 80
        assert np.min(observed) < 20

    def test_images_float_as_per_channel_tuple_as_factor_two_branches(self):
        aug = iaa.Alpha(
            (0.0, 1.0),
            iaa.Add(100),
            iaa.Add(0),
            per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(200):
            observed = aug.augment_image(np.zeros((1, 1, 100), dtype=np.uint8))
            uq = np.unique(observed)
            if len(uq) == 1:
                seen[0] += 1
            elif len(uq) > 1:
                seen[1] += 1
            else:
                assert False
        assert 100 - 50 < seen[0] < 100 + 50
        assert 100 - 50 < seen[1] < 100 + 50

    def test_bad_datatype_for_per_channel_fails(self):
        # bad datatype for per_channel
        got_exception = False
        try:
            _ = iaa.Alpha(0.5, iaa.Add(10), None, per_channel="test")
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_hooks_limiting_propagation(self):
        aug = iaa.Alpha(0.5, iaa.Add(100), iaa.Add(50), name="AlphaTest")

        def propagator(images, augmenter, parents, default):
            if "Alpha" in augmenter.name:
                return False
            else:
                return default

        hooks = ia.HooksImages(propagator=propagator)
        image = np.zeros((10, 10, 3), dtype=np.uint8) + 1
        observed = aug.augment_image(image, hooks=hooks)
        assert np.array_equal(observed, image)

    def test_keypoints_factor_is_1(self):
        self._test_cba_factor_is_1("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0(self):
        self._test_cba_factor_is_0("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_1_with_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0_with_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_choice_of_vals_close_to_050_per_channel(self):
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_keypoints", self.kpsoi)

    def test_keypoints_are_empty(self):
        self._test_empty_cba(
            "augment_keypoints", ia.KeypointsOnImage([], shape=(1, 2, 3)))

    def test_keypoints_hooks_limit_propagation(self):
        self._test_cba_hooks_limit_propagation(
            "augment_keypoints", self.kpsoi)

    def test_polygons_factor_is_1(self):
        self._test_cba_factor_is_1("augment_polygons", self.psoi)

    def test_polygons_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_polygons", self.psoi)

    def test_polygons_factor_is_0(self):
        self._test_cba_factor_is_0("augment_polygons", self.psoi)

    def test_polygons_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_polygons", self.psoi)

    def test_polygons_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_polygons", self.psoi)

    def test_polygons_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_polygons", self.psoi)

    def test_polygons_factor_is_choice_around_050_and_per_channel(self):
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_polygons", self.psoi
        )

    def test_empty_polygons(self):
        return self._test_empty_cba(
            "augment_polygons", ia.PolygonsOnImage([], shape=(1, 2, 3)))

    def test_polygons_hooks_limit_propagation(self):
        return self._test_cba_hooks_limit_propagation(
            "augment_polygons", self.psoi)

    def test_line_strings_factor_is_1(self):
        self._test_cba_factor_is_1("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0(self):
        self._test_cba_factor_is_0("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_choice_around_050_and_per_channel(self):
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_line_strings", self.lsoi
        )

    def test_empty_line_strings(self):
        return self._test_empty_cba(
            "augment_line_strings",
            ia.LineStringsOnImage([], shape=(1, 2, 3)))

    def test_bounding_boxes_hooks_limit_propagation(self):
        return self._test_cba_hooks_limit_propagation(
            "augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_1(self):
        self._test_cba_factor_is_1("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0(self):
        self._test_cba_factor_is_0("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_choice_around_050_and_per_channel(self):
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_bounding_boxes", self.bbsoi
        )

    def test_empty_bounding_boxes(self):
        return self._test_empty_cba(
            "augment_bounding_boxes",
            ia.BoundingBoxesOnImage([], shape=(1, 2, 3)))

    def test_bounding_boxes_hooks_limit_propagation(self):
        return self._test_cba_hooks_limit_propagation(
            "augment_bounding_boxes", self.bbsoi)

    # Tests for CBA (=coordinate based augmentable) below. This currently
    # covers keypoints, polygons and bounding boxes.

    @classmethod
    def _test_cba_factor_is_1(cls, augf_name, cbaoi):
        aug = iaa.Alpha(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0501(cls, augf_name, cbaoi):
        aug = iaa.Alpha(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0(cls, augf_name, cbaoi):
        aug = iaa.Alpha(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_0499(cls, augf_name, cbaoi):
        aug = iaa.Alpha(0.499, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_1_and_per_channel(cls, augf_name, cbaoi):
        aug = iaa.Alpha(
            1.0,
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0_and_per_channel(cls, augf_name, cbaoi):
        aug = iaa.Alpha(
            0.0,
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_choice_around_050_and_per_channel(
            cls, augf_name, cbaoi):
        aug = iaa.Alpha(
            iap.Choice([0.49, 0.51]),
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)
        expected_same = cbaoi.deepcopy()
        expected_shifted = shift_cbaoi(cbaoi, left=1)
        seen = [0, 0, 0]
        for _ in sm.xrange(200):
            observed = getattr(aug, augf_name)([cbaoi])[0]

            assert len(observed.items) == len(expected_same.items)
            assert len(observed.items) == len(expected_shifted.items)

            # We use here allclose() instead of coords_almost_equals()
            # as the latter one is much slower for polygons and we don't have
            # to deal with tricky geometry changes here, just naive shifting.
            if np.allclose(observed.items[0].coords,
                           expected_same.items[0].coords,
                           rtol=0, atol=0.1):
                seen[0] += 1
            elif np.allclose(observed.items[0].coords,
                             expected_shifted.items[0].coords,
                             rtol=0, atol=0.1):
                seen[1] += 1
            else:
                seen[2] += 1
        assert 100 - 50 < seen[0] < 100 + 50
        assert 100 - 50 < seen[1] < 100 + 50
        assert seen[2] == 0

    @classmethod
    def _test_empty_cba(cls, augf_name, cbaoi):
        # empty CBAs
        aug = iaa.Alpha(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)(cbaoi)

        assert len(observed.items) == 0
        assert observed.shape == cbaoi.shape

    @classmethod
    def _test_cba_hooks_limit_propagation(cls, augf_name, cbaoi):
        aug = iaa.Alpha(
            0.0,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"y": 1}),
            name="AlphaTest")

        def propagator(cbaoi_to_aug, augmenter, parents, default):
            if "Alpha" in augmenter.name:
                return False
            else:
                return default

        # no hooks for polygons yet, so we use HooksKeypoints
        hooks = ia.HooksKeypoints(propagator=propagator)
        observed = getattr(aug, augf_name)([cbaoi], hooks=hooks)[0]
        assert observed.items[0].coords_almost_equals(cbaoi.items[0])

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 0, dtype=np.uint8)
                aug = iaa.Alpha(1.0, iaa.Add(1), iaa.Add(100))

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 0, dtype=np.uint8)
                aug = iaa.Alpha(1.0, iaa.Add(1), iaa.Add(100))

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_get_parameters(self):
        first = iaa.Noop()
        second = iaa.Sequential([iaa.Add(1)])
        aug = iaa.Alpha(0.65, first, second, per_channel=1)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert 0.65 - 1e-6 < params[0].value < 0.65 + 1e-6
        assert params[1].value == 1

    def test_get_children_lists(self):
        first = iaa.Noop()
        second = iaa.Sequential([iaa.Add(1)])
        aug = iaa.Alpha(0.65, first, second, per_channel=1)
        children_lsts = aug.get_children_lists()
        assert len(children_lsts) == 2
        assert ia.is_iterable([lst for lst in children_lsts])
        assert first in children_lsts[0]
        assert second == children_lsts[1]


class _DummyMaskParameter(iap.StochasticParameter):
    def __init__(self, inverted=False):
        super(_DummyMaskParameter, self).__init__()
        self.nb_calls = 0
        self.inverted = inverted

    def _draw_samples(self, size, random_state):
        self.nb_calls += 1
        h, w = size
        ones = np.ones((h, w), dtype=np.float32)
        zeros = np.zeros((h, w), dtype=np.float32)
        if self.nb_calls == 1:
            return zeros if not self.inverted else ones
        elif self.nb_calls in [2, 3]:
            return ones if not self.inverted else zeros
        else:
            assert False


# TODO add tests for heatmaps and segmaps that differ from the image size
class TestAlphaElementwise(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.zeros((3, 3, 1), dtype=np.uint8)
        return base_img

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 1.0, 1.0]])
        return HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def heatmaps_r1(self):
        heatmaps_arr_r1 = np.float32([[0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0]])
        return HeatmapsOnImage(heatmaps_arr_r1, shape=(3, 3, 3))

    @property
    def heatmaps_l1(self):
        heatmaps_arr_l1 = np.float32([[0.0, 1.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [1.0, 1.0, 0.0]])

        return HeatmapsOnImage(heatmaps_arr_l1, shape=(3, 3, 3))

    @property
    def segmaps(self):
        segmaps_arr = np.int32([[0, 0, 1],
                                [0, 0, 1],
                                [0, 1, 1]])
        return SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps_r1(self):
        segmaps_arr_r1 = np.int32([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 1]])
        return SegmentationMapsOnImage(segmaps_arr_r1, shape=(3, 3, 3))

    @property
    def segmaps_l1(self):
        segmaps_arr_l1 = np.int32([[0, 1, 0],
                                   [0, 1, 0],
                                   [1, 1, 0]])
        return SegmentationMapsOnImage(segmaps_arr_l1, shape=(3, 3, 3))

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=5, y=10), ia.Keypoint(x=6, y=11)]
        return ia.KeypointsOnImage(kps, shape=(20, 20, 3))

    @property
    def psoi(self):
        ps = [ia.Polygon([(5, 5), (10, 5), (10, 10)])]
        return ia.PolygonsOnImage(ps, shape=(20, 20, 3))

    @property
    def lsoi(self):
        lss = [ia.LineString([(5, 5), (10, 5), (10, 10)])]
        return ia.LineStringsOnImage(lss, shape=(20, 20, 3))

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)]
        return ia.BoundingBoxesOnImage(bbs, shape=(20, 20, 3))

    def test_images_factor_is_1(self):
        aug = iaa.AlphaElementwise(1, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = self.image + 10
        assert np.allclose(observed, expected)

    def test_heatmaps_factor_is_1_with_affines(self):
        aug = iaa.AlphaElementwise(
            1,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}))
        observed = aug.augment_heatmaps([self.heatmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps_r1.get_arr())

    def test_segmaps_factor_is_1_with_affines(self):
        aug = iaa.AlphaElementwise(
            1,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}))
        observed = aug.augment_segmentation_maps([self.segmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps_r1.get_arr())

    def test_images_factor_is_0(self):
        aug = iaa.AlphaElementwise(0, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = self.image + 20
        assert np.allclose(observed, expected)

    def test_heatmaps_factor_is_0_with_affines(self):
        aug = iaa.AlphaElementwise(
            0,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}))
        observed = aug.augment_heatmaps([self.heatmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps_l1.get_arr())

    def test_segmaps_factor_is_0_with_affines(self):
        aug = iaa.AlphaElementwise(
            0,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}))
        observed = aug.augment_segmentation_maps([self.segmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps_l1.get_arr())

    def test_images_factor_is_075(self):
        aug = iaa.AlphaElementwise(0.75, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = np.round(
            self.image + 0.75 * 10 + 0.25 * 20
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_075_first_branch_is_none(self):
        aug = iaa.AlphaElementwise(0.75, None, iaa.Add(20))
        observed = aug.augment_image(self.image + 10)
        expected = np.round(
            self.image + 0.75 * 10 + 0.25 * (10 + 20)
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_075_second_branch_is_none(self):
        aug = iaa.AlphaElementwise(0.75, iaa.Add(10), None)
        observed = aug.augment_image(self.image + 10)
        expected = np.round(
            self.image + 0.75 * (10 + 10) + 0.25 * 10
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_tuple(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        aug = iaa.AlphaElementwise((0.0, 1.0), iaa.Add(10), iaa.Add(110))
        observed = (aug.augment_image(image) - 10) / 100
        nb_bins = 10
        hist, _ = np.histogram(
            observed.flatten(), bins=nb_bins, range=(0.0, 1.0), density=False)
        density_expected = 1.0/nb_bins
        density_tolerance = 0.05
        for nb_samples in hist:
            density = nb_samples / observed.size
            assert np.isclose(density, density_expected,
                              rtol=0, atol=density_tolerance)

    def test_bad_datatype_for_factor_fails(self):
        got_exception = False
        try:
            _ = iaa.AlphaElementwise(False, iaa.Add(10), None)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_images_with_per_channel_in_alpha_and_tuple_as_factor(self):
        image = np.zeros((1, 1, 100), dtype=np.uint8)
        aug = iaa.AlphaElementwise(
            (0.0, 1.0),
            iaa.Add(10),
            iaa.Add(110),
            per_channel=True)
        observed = aug.augment_image(image)
        assert len(set(observed.flatten())) > 1

    def test_bad_datatype_for_per_channel_fails(self):
        got_exception = False
        try:
            _ = iaa.AlphaElementwise(
                0.5,
                iaa.Add(10),
                None,
                per_channel="test")
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_hooks_limiting_propagation(self):
        aug = iaa.AlphaElementwise(
            0.5,
            iaa.Add(100),
            iaa.Add(50),
            name="AlphaElementwiseTest")

        def propagator(images, augmenter, parents, default):
            if "AlphaElementwise" in augmenter.name:
                return False
            else:
                return default

        hooks = ia.HooksImages(propagator=propagator)
        image = np.zeros((10, 10, 3), dtype=np.uint8) + 1
        observed = aug.augment_image(image, hooks=hooks)
        assert np.array_equal(observed, image)

    def test_heatmaps_and_per_channel_factor_is_zeros(self):
        aug = iaa.AlphaElementwise(
            _DummyMaskParameter(inverted=False),
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}),
            per_channel=True)
        observed = aug.augment_heatmaps([self.heatmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps_r1.get_arr())

    def test_heatmaps_and_per_channel_factor_is_ones(self):
        aug = iaa.AlphaElementwise(
            _DummyMaskParameter(inverted=True),
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}),
            per_channel=True)
        observed = aug.augment_heatmaps([self.heatmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps_l1.get_arr())

    def test_segmaps_and_per_channel_factor_is_zeros(self):
        aug = iaa.AlphaElementwise(
            _DummyMaskParameter(inverted=False),
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}),
            per_channel=True)
        observed = aug.augment_segmentation_maps([self.segmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps_r1.get_arr())

    def test_segmaps_and_per_channel_factor_is_ones(self):
        aug = iaa.AlphaElementwise(
            _DummyMaskParameter(inverted=True),
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}),
            per_channel=True)
        observed = aug.augment_segmentation_maps([self.segmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps_l1.get_arr())

    def test_keypoints_factor_is_1(self):
        self._test_cba_factor_is_1("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0(self):
        self._test_cba_factor_is_0("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_1_with_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0_with_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_choice_of_vals_close_050_per_channel(self):
        # TODO can this somehow be integrated into the CBA functions below?
        aug = iaa.Alpha(
            iap.Choice([0.49, 0.51]),
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)
        kpsoi = self.kpsoi

        expected_same = kpsoi.deepcopy()
        expected_both_shifted = kpsoi.shift(x=1)
        expected_first_shifted = ia.KeypointsOnImage(
            [kpsoi.keypoints[0].shift(x=1), kpsoi.keypoints[1]],
            shape=self.kpsoi.shape)
        expected_second_shifted = ia.KeypointsOnImage(
            [kpsoi.keypoints[0], kpsoi.keypoints[1].shift(x=1)],
            shape=self.kpsoi.shape)

        seen = [0, 0]
        for _ in sm.xrange(200):
            observed = aug.augment_keypoints([kpsoi])[0]
            if keypoints_equal([observed], [expected_same]):
                seen[0] += 1
            elif keypoints_equal([observed], [expected_both_shifted]):
                seen[1] += 1
            elif keypoints_equal([observed], [expected_first_shifted]):
                seen[2] += 1
            elif keypoints_equal([observed], [expected_second_shifted]):
                seen[3] += 1
            else:
                assert False
        assert 100 - 50 < seen[0] < 100 + 50
        assert 100 - 50 < seen[1] < 100 + 50

    def test_keypoints_are_empty(self):
        kpsoi = ia.KeypointsOnImage([], shape=(1, 2, 3))
        self._test_empty_cba("augment_keypoints", kpsoi)

    def test_keypoints_hooks_limit_propagation(self):
        self._test_cba_hooks_limit_propagation("augment_keypoints", self.kpsoi)

    def test_polygons_factor_is_1(self):
        self._test_cba_factor_is_1("augment_polygons", self.psoi)

    def test_polygons_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_polygons", self.psoi)

    def test_polygons_factor_is_0(self):
        self._test_cba_factor_is_0("augment_polygons", self.psoi)

    def test_polygons_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_polygons", self.psoi)

    def test_polygons_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_polygons", self.psoi)

    def test_polygons_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_polygons", self.psoi)

    def test_polygons_factor_is_choice_around_050_and_per_channel(self):
        # why were different polygons than self.psoi chosen here?
        ps = [ia.Polygon([(0, 0), (15, 0), (10, 0), (10, 5), (10, 10),
                          (5, 10), (5, 5), (0, 10), (0, 5), (0, 0)])]
        psoi = ia.PolygonsOnImage(ps, shape=(15, 15, 3))
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_polygons", psoi, pointwise=False
        )

    def test_empty_polygons(self):
        psoi = ia.PolygonsOnImage([], shape=(1, 2, 3))
        self._test_empty_cba("augment_polygons", psoi)

    def test_polygons_hooks_limit_propagation(self):
        self._test_cba_hooks_limit_propagation("augment_polygons", self.psoi)

    def test_line_strings_factor_is_1(self):
        self._test_cba_factor_is_1("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0(self):
        self._test_cba_factor_is_0("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_choice_around_050_and_per_channel(self):
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_line_strings", self.lsoi, pointwise=True
        )

    def test_empty_line_strings(self):
        lsoi = ia.LineStringsOnImage([], shape=(1, 2, 3))
        self._test_empty_cba("augment_line_strings", lsoi)

    def test_line_strings_hooks_limit_propagation(self):
        self._test_cba_hooks_limit_propagation(
            "augment_line_strings", self.lsoi)

    def test_bounding_boxes_factor_is_1(self):
        self._test_cba_factor_is_1("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0(self):
        self._test_cba_factor_is_0("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_choice_around_050_and_per_channel(self):
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_bounding_boxes", self.bbsoi, pointwise=True
        )

    def test_empty_bounding_boxes(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(1, 2, 3))
        self._test_empty_cba("augment_bounding_boxes", bbsoi)

    def test_bounding_boxes_hooks_limit_propagation(self):
        self._test_cba_hooks_limit_propagation(
            "augment_bounding_boxes", self.bbsoi)

    @classmethod
    def _test_cba_factor_is_1(cls, augf_name, cbaoi):
        aug = iaa.AlphaElementwise(
            1.0,
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0501(cls, augf_name, cbaoi):
        aug = iaa.AlphaElementwise(
            0.501,
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0(cls, augf_name, cbaoi):
        aug = iaa.AlphaElementwise(
            0.0,
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_0499(cls, augf_name, cbaoi):
        aug = iaa.AlphaElementwise(
            0.499,
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_1_and_per_channel(cls, augf_name, cbaoi):
        aug = iaa.AlphaElementwise(
            1.0,
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0_and_per_channel(cls, augf_name, cbaoi):
        aug = iaa.AlphaElementwise(
            0.0,
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_choice_around_050_and_per_channel(
            cls, augf_name, cbaoi, pointwise):
        aug = iaa.AlphaElementwise(
            iap.Choice([0.49, 0.51]),
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)

        expected_same = cbaoi.deepcopy()
        expected_shifted = shift_cbaoi(cbaoi, left=1)

        nb_iterations = 400
        seen = [0, 0, 0]
        for _ in sm.xrange(nb_iterations):
            observed = getattr(aug, augf_name)([cbaoi])[0]
            # We use here allclose() instead of coords_almost_equals()
            # as the latter one is much slower for polygons and we don't have
            # to deal with tricky geometry changes here, just naive shifting.
            if np.allclose(observed.items[0].coords,
                           expected_same.items[0].coords,
                           rtol=0, atol=0.1):
                seen[0] += 1
            elif np.allclose(observed.items[0].coords,
                             expected_shifted.items[0].coords,
                             rtol=0, atol=0.1):
                seen[1] += 1
            else:
                seen[2] += 1

        if pointwise:
            # This code can be used if the polygon augmentation mode is
            # AlphaElementwise._MODE_POINTWISE. Currently it is _MODE_EITHER_OR.
            nb_points = len(cbaoi.items[0].coords)
            p_all_same = 2 * ((1/2)**nb_points)  # all points moved in same way
            expected_iter = nb_iterations*p_all_same
            expected_iter_notsame = nb_iterations*(1-p_all_same)
            atol = nb_iterations * (5*p_all_same)

            assert np.isclose(seen[0], expected_iter, rtol=0, atol=atol)
            assert np.isclose(seen[1], expected_iter, rtol=0, atol=atol)
            assert np.isclose(seen[2], expected_iter_notsame, rtol=0, atol=atol)
        else:
            expected_iter = nb_iterations*0.5
            atol = nb_iterations*0.15
            assert np.isclose(seen[0], expected_iter, rtol=0, atol=atol)
            assert np.isclose(seen[1], expected_iter, rtol=0, atol=atol)
            assert seen[2] == 0

    @classmethod
    def _test_empty_cba(cls, augf_name, cbaoi):
        aug = iaa.AlphaElementwise(
            0.501,
            iaa.Noop(),
            iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)(cbaoi)

        assert len(observed.items) == 0
        assert observed.shape == (1, 2, 3)

    @classmethod
    def _test_cba_hooks_limit_propagation(cls, augf_name, cbaoi):
        aug = iaa.AlphaElementwise(
            0.0,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"y": 1}),
            name="AlphaTest")

        def propagator(cbaoi_to_aug, augmenter, parents, default):
            if "Alpha" in augmenter.name:
                return False
            else:
                return default

        # no hooks for polygons yet, so we use HooksKeypoints
        hooks = ia.HooksKeypoints(propagator=propagator)
        observed = getattr(aug, augf_name)([cbaoi], hooks=hooks)[0]
        assert observed.items[0].coords_almost_equals(cbaoi.items[0])

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 0, dtype=np.uint8)
                aug = iaa.Alpha(1.0, iaa.Add(1), iaa.Add(100))

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 0, dtype=np.uint8)
                aug = iaa.Alpha(1.0, iaa.Add(1), iaa.Add(100))

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape
