from __future__ import print_function, division, absolute_import

import itertools
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

import imgaug as ia
import imgaug.augmentables.segmaps as segmapslib


# old style segmentation maps (class name differs to new style by "Map"
# instead of "Maps")
class TestSegmentationMapOnImage(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            segmap = segmapslib.SegmentationMapOnImage(
                np.zeros((1, 1, 1), dtype=np.int32),
                shape=(1, 1, 3)
            )
            assert segmap.arr.dtype.name == "int32"
            assert segmap.arr.shape == (1, 1, 1)
            assert segmap.shape == (1, 1, 3)
        assert len(caught_warnings) == 1
        assert "is deprecated" in str(caught_warnings[0].message)


class TestSegmentationMapsOnImage___init__(unittest.TestCase):
    def test_uint_int_arrs(self):
        dtypes = ["int8", "int16", "int32", "uint8", "uint16"]
        ndims = [2, 3]
        img_shapes = [(3, 3), (3, 3, 3), (4, 5, 3)]

        gen = itertools.product(dtypes, ndims, img_shapes)
        for dtype, ndim, img_shape in gen:
            with self.subTest(dtype=dtype, ndim=ndim, shape=img_shape):
                dtype = np.dtype(dtype)
                shape = (3, 3) if ndim == 2 else (3, 3, 1)
                arr = np.array([
                    [0, 0, 1],
                    [0, 2, 1],
                    [1, 3, 1]
                ], dtype=dtype).reshape(shape)
                segmap = ia.SegmentationMapsOnImage(arr, shape=img_shape)
                assert segmap.shape == img_shape
                assert segmap.arr.dtype.name == "int32"
                assert segmap.arr.shape == (3, 3, 1)
                assert np.array_equal(segmap.arr,
                                      arr.reshape((3, 3, 1)).astype(np.int32))

                if ndim == 3:
                    arr = np.array([
                        [0, 0, 1],
                        [0, 2, 1],
                        [1, 3, 1]
                    ], dtype=dtype).reshape((3, 3, 1))
                    arr = np.tile(arr, (1, 1, 5))
                    segmap = ia.SegmentationMapsOnImage(arr, shape=img_shape)
                    assert segmap.shape == img_shape
                    assert segmap.arr.dtype.name == "int32"
                    assert segmap.arr.shape == (3, 3, 5)
                    assert np.array_equal(segmap.arr, arr.astype(np.int32))

    def test_bool_arr_2d(self):
        arr = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ], dtype=bool).reshape((3, 3))

        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))

        assert segmap.shape == (3, 3)
        assert segmap.arr.dtype.name == "int32"
        assert segmap.arr.shape == (3, 3, 1)
        assert np.array_equal(segmap.arr,
                              arr.reshape((3, 3, 1)).astype(np.int32))

    def test_bool_arr_3d(self):
        arr = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ], dtype=bool).reshape((3, 3, 1))
        arr = np.tile(arr, (1, 1, 5))

        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))

        assert segmap.shape == (3, 3)
        assert segmap.arr.dtype.name == "int32"
        assert segmap.arr.shape == (3, 3, 5)
        assert np.array_equal(segmap.arr, arr.astype(np.int32))

    # is this different from the test_bool_* tests?
    def test_boolean_masks(self):
        # Test for #189 (boolean mask inputs into SegmentationMapsOnImage not
        # working)
        for dt in [bool, np.bool]:
            arr = np.array([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ], dtype=dt)
            assert arr.dtype.kind == "b"
            segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))
            assert np.array_equal(
                segmap.arr,
                np.int32([
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]
                ])[:, :, np.newaxis]
            )
            assert segmap.get_arr().dtype.name == arr.dtype.name
            assert np.array_equal(segmap.get_arr(), arr)

    def test_uint32_fails(self):
        got_exception = False
        try:
            arr = np.array([
                [0, 0, 1],
                [0, 2, 1],
                [1, 3, 1]
            ], dtype=np.uint32)
            _segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3, 3))
        except Exception as exc:
            assert "only uint8 and uint16 " in str(exc)
            got_exception = True
        assert got_exception

    def test_uint64_fails(self):
        got_exception = False
        try:
            arr = np.array([
                [0, 0, 1],
                [0, 2, 1],
                [1, 3, 1]
            ], dtype=np.int64)
            _segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3, 3))
        except Exception as exc:
            assert "only int8, int16 and int32 " in str(exc)
            got_exception = True
        assert got_exception

    def test_legacy_support_for_float32_2d(self):
        arr = np.array([0.4, 0.6], dtype=np.float32).reshape((1, 2))
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            segmap = segmapslib.SegmentationMapsOnImage(arr, shape=(1, 1, 3))

            arr_expected = np.array([0, 1], dtype=np.int32).reshape((1, 2, 1))
            assert np.array_equal(segmap.arr, arr_expected)
            assert segmap.shape == (1, 1, 3)

        assert len(caught_warnings) == 1
        assert (
            "Got a float array as the segmentation map in"
            in str(caught_warnings[0].message)
        )

    def test_legacy_support_for_float32_3d(self):
        arr = np.array([
            [
                [0.4, 0.6],
                [0.2, 0.1]
            ]
        ], dtype=np.float32).reshape((1, 2, 2))
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            segmap = segmapslib.SegmentationMapsOnImage(arr, shape=(1, 1, 3))

            arr_expected = np.array([
                [1, 0]
            ], dtype=np.int32).reshape((1, 2, 1))
            assert np.array_equal(segmap.arr, arr_expected)
            assert segmap.shape == (1, 1, 3)

        assert len(caught_warnings) == 1
        assert (
            "Got a float array as the segmentation map in"
            in str(caught_warnings[0].message)
        )


class TestSegmentationMapsOnImage_get_arr(unittest.TestCase):
    def test_uint_int(self):
        dtypes = ["int8", "int16", "int32", "uint8", "uint16"]
        ndims = [2, 3]

        for dtype, ndim in itertools.product(dtypes, ndims):
            with self.subTest(dtype=dtype, ndim=ndim):
                dtype = np.dtype(dtype)
                shape = (3, 3) if ndim == 2 else (3, 3, 1)
                arr = np.array([
                    [0, 0, 1],
                    [0, 2, 1],
                    [1, 3, 1]
                ], dtype=dtype).reshape(shape)
                segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))

                observed = segmap.get_arr()

                assert segmap.arr.dtype.name == "int32"
                assert segmap.arr.ndim == 3
                assert np.array_equal(observed, arr)
                assert observed.dtype.name == dtype.name
                assert observed.ndim == ndim
                assert np.array_equal(observed, arr)

    def test_bool(self):
        ndims = [2, 3]
        for ndim in ndims:
            with self.subTest(ndim=ndim):
                shape = (3, 3) if ndim == 2 else (3, 3, 1)
                arr = np.array([
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1]
                ], dtype=bool).reshape(shape)
                segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))

                observed = segmap.get_arr()

                assert segmap.arr.dtype.name == "int32"
                assert segmap.arr.ndim == 3
                assert np.array_equal(observed, arr)
                assert observed.dtype.kind == "b"
                assert observed.ndim == ndim
                assert np.array_equal(observed, arr)


class TestSegmentationMapsOnImage_draw(unittest.TestCase):
    @property
    def segmap(self):
        arr = np.int32([
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]
        ])
        return ia.SegmentationMapsOnImage(arr, shape=(3, 3))

    @classmethod
    def col(cls, idx):
        return ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS[idx]

    def test_with_two_classes(self):
        # simple example with 2 classes
        col0 = self.col(0)
        col1 = self.col(1)
        expected = np.uint8([
            [col0, col1, col1],
            [col0, col1, col1],
            [col0, col1, col1]
        ])

        observed = self.segmap.draw()

        assert isinstance(observed, list)
        assert len(observed) == 1
        assert np.array_equal(observed[0], expected)

    def test_use_size_arg_to_resize_to_2x(self):
        # same example, with resizing to 2x the size
        double_size_args = [
            (6, 6),
            (2.0, 2.0),
            6,
            2.0
        ]

        col0 = self.col(0)
        col1 = self.col(1)
        expected = np.uint8([
            [col0, col1, col1],
            [col0, col1, col1],
            [col0, col1, col1]
        ])
        expected = ia.imresize_single_image(expected,
                                            (6, 6),
                                            interpolation="nearest")

        for double_size_arg in double_size_args:
            with self.subTest(size=double_size_arg):
                observed = self.segmap.draw(size=double_size_arg)

                assert isinstance(observed, list)
                assert len(observed) == 1
                assert np.array_equal(observed[0], expected)

    def test_use_size_arg_to_keep_at_same_size(self):
        # same example, keeps size at 3x3 via None and (int)3 or (float)1.0
        size_args = [
            None,
            (None, None),
            (3, None),
            (None, 3),
            (1.0, None),
            (None, 1.0)
        ]

        col0 = self.col(0)
        col1 = self.col(1)
        expected = np.uint8([
            [col0, col1, col1],
            [col0, col1, col1],
            [col0, col1, col1]
        ])
        expected = ia.imresize_single_image(expected,
                                            (3, 3),
                                            interpolation="nearest")

        for size_arg in size_args:
            with self.subTest(size=size_arg):
                observed = self.segmap.draw(size=size_arg)

                assert isinstance(observed, list)
                assert len(observed) == 1
                assert np.array_equal(observed[0], expected)

    def test_colors(self):
        # custom choice of colors
        col0 = (10, 10, 10)
        col1 = (50, 51, 52)
        expected = np.uint8([
            [col0, col1, col1],
            [col0, col1, col1],
            [col0, col1, col1]
        ])

        observed = self.segmap.draw(colors=[col0, col1])

        assert isinstance(observed, list)
        assert len(observed) == 1
        assert np.array_equal(observed[0], expected)

    def test_segmap_with_more_than_one_channel(self):
        # test segmentation maps with multiple channels
        arr_channel_1 = np.int32([
            [0, 1, 5],
            [0, 1, 1],
            [0, 4, 1]
        ])
        arr_channel_2 = np.int32([
            [1, 1, 0],
            [2, 2, 0],
            [1, 1, 0]
        ])
        arr_channel_3 = np.int32([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 3]
        ])
        arr_multi = np.stack(
            [arr_channel_1, arr_channel_2, arr_channel_3],
            axis=-1)

        col = ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS
        expected_channel_1 = np.uint8([
            [col[0], col[1], col[5]],
            [col[0], col[1], col[1]],
            [col[0], col[4], col[1]]
        ])
        expected_channel_2 = np.uint8([
            [col[1], col[1], col[0]],
            [col[2], col[2], col[0]],
            [col[1], col[1], col[0]]
        ])
        expected_channel_3 = np.uint8([
            [col[1], col[0], col[0]],
            [col[0], col[1], col[0]],
            [col[0], col[0], col[3]]
        ])

        segmap = ia.SegmentationMapsOnImage(arr_multi, shape=(3, 3, 3))

        observed = segmap.draw()

        assert isinstance(observed, list)
        assert len(observed) == 3
        assert np.array_equal(observed[0], expected_channel_1)
        assert np.array_equal(observed[1], expected_channel_2)
        assert np.array_equal(observed[2], expected_channel_3)


class TestSegmentationMapsOnImage_draw_on_image(unittest.TestCase):
    @property
    def segmap(self):
        arr = np.int32([
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]
        ])
        return ia.SegmentationMapsOnImage(arr, shape=(3, 3))

    @property
    def image(self):
        image = np.uint8([
            [0, 10, 20],
            [30, 40, 50],
            [60, 70, 80]
        ])
        return np.tile(image[:, :, np.newaxis], (1, 1, 3))

    @classmethod
    def col(cls, idx):
        return ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS[idx]

    def test_alpha_only_image_is_visible(self):
        # only image visible
        observed = self.segmap.draw_on_image(self.image, alpha=0)

        assert isinstance(observed, list)
        assert len(observed) == 1
        assert np.array_equal(observed[0], self.image)

    def test_alpha_only_segmap_is_visible(self):
        # only segmap visible
        observed = self.segmap.draw_on_image(self.image, alpha=1.0,
                                             draw_background=True)
        col0 = self.col(0)
        col1 = self.col(1)
        expected = np.uint8([
            [col0, col1, col1],
            [col0, col1, col1],
            [col0, col1, col1]
        ])
        assert isinstance(observed, list)
        assert len(observed) == 1
        assert np.array_equal(observed[0], expected)

    def test_alpha_with_draw_background(self):
        # only segmap visible - in foreground
        image = self.image

        observed = self.segmap.draw_on_image(image, alpha=1.0,
                                             draw_background=False)

        col1 = self.col(1)
        expected = np.uint8([
            [image[0, 0, :], col1, col1],
            [image[1, 0, :], col1, col1],
            [image[2, 0, :], col1, col1]
        ])
        assert isinstance(observed, list)
        assert len(observed) == 1
        assert np.array_equal(observed[0], expected)

    def test_alpha_with_draw_background_and_more_than_one_channel(self):
        # only segmap visible in foreground + multiple channels in segmap
        image = self.image

        arr_channel_1 = np.int32([
            [0, 1, 5],
            [0, 1, 1],
            [0, 4, 1]
        ])
        arr_channel_2 = np.int32([
            [1, 1, 0],
            [2, 2, 0],
            [1, 1, 0]
        ])
        arr_channel_3 = np.int32([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 3]
        ])
        arr_multi = np.stack(
            [arr_channel_1, arr_channel_2, arr_channel_3],
            axis=-1)

        col = ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS
        expected_channel_1 = np.uint8([
            [image[0, 0, :], col[1], col[5]],
            [image[1, 0, :], col[1], col[1]],
            [image[2, 0, :], col[4], col[1]]
        ])
        expected_channel_2 = np.uint8([
            [col[1], col[1], image[0, 2, :]],
            [col[2], col[2], image[1, 2, :]],
            [col[1], col[1], image[2, 2, :]]
        ])
        expected_channel_3 = np.uint8([
            [col[1], image[0, 1, :], image[0, 2, :]],
            [image[1, 0, :], col[1], image[1, 2, :]],
            [image[2, 0, :], image[2, 1, :], col[3]]
        ])

        segmap_multi = ia.SegmentationMapsOnImage(arr_multi, shape=(3, 3, 3))

        observed = segmap_multi.draw_on_image(
            image, alpha=1.0, draw_background=False)

        assert isinstance(observed, list)
        assert len(observed) == 3
        assert np.array_equal(observed[0], expected_channel_1)
        assert np.array_equal(observed[1], expected_channel_2)
        assert np.array_equal(observed[2], expected_channel_3)

    def test_non_binary_alpha_with_draw_background(self):
        # overlay without background drawn
        im = self.image
        segmap = self.segmap

        a1 = 0.7
        a0 = 1.0 - a1

        observed = segmap.draw_on_image(im, alpha=a1, draw_background=False)

        col1 = np.uint8(self.col(1))
        expected = np.float32([
            [im[0, 0, :], a0*im[0, 1, :] + a1*col1, a0*im[0, 2, :] + a1*col1],
            [im[1, 0, :], a0*im[1, 1, :] + a1*col1, a0*im[1, 2, :] + a1*col1],
            [im[2, 0, :], a0*im[2, 1, :] + a1*col1, a0*im[2, 2, :] + a1*col1]
        ])
        d_max = np.max(np.abs(observed[0].astype(np.float32) - expected))
        assert isinstance(observed, list)
        assert len(observed) == 1
        assert observed[0].shape == expected.shape
        assert d_max <= 1.0 + 1e-4

    def test_non_binary_alpha_with_draw_background_and_bg_class_id(self):
        # overlay without background drawn
        # different background class id
        image = self.image
        segmap = self.segmap

        a1 = 0.7
        a0 = 1.0 - a1

        observed = segmap.draw_on_image(image, alpha=a1, draw_background=False,
                                        background_class_id=1)

        col0 = np.uint8(self.col(0))
        expected = np.float32([
            [a0*image[0, 0, :] + a1*col0, image[0, 1, :], image[0, 2, :]],
            [a0*image[1, 0, :] + a1*col0, image[1, 1, :], image[1, 2, :]],
            [a0*image[2, 0, :] + a1*col0, image[2, 1, :], image[2, 2, :]]
        ])
        d_max = np.max(np.abs(observed[0].astype(np.float32) - expected))
        assert isinstance(observed, list)
        assert len(observed) == 1
        assert observed[0].shape == expected.shape
        assert d_max <= 1.0 + 1e-4

    def test_non_binary_alpha_with_draw_background_true(self):
        # overlay with background drawn
        segmap = self.segmap
        image = self.image

        a1 = 0.7
        a0 = 1.0 - a1

        observed = segmap.draw_on_image(image, alpha=a1, draw_background=True)

        col0 = self.col(0)
        col1 = self.col(1)
        expected = np.uint8([
            [col0, col1, col1],
            [col0, col1, col1],
            [col0, col1, col1]
        ])
        expected = a0 * image + a1 * expected
        d_max = np.max(
            np.abs(
                observed[0].astype(np.float32)
                - expected.astype(np.float32)
            )
        )
        assert isinstance(observed, list)
        assert len(observed) == 1
        assert observed[0].shape == expected.shape
        assert d_max <= 1.0 + 1e-4

    def test_resize_segmentation_map_to_image(self):
        # resizing of segmap to image
        arr = np.int32([
            [0, 1, 1]
        ])
        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))

        image = np.uint8([
            [0, 10, 20],
            [30, 40, 50],
            [60, 70, 80]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        a1 = 0.7
        a0 = 1.0 - a1

        observed = segmap.draw_on_image(image, alpha=a1, draw_background=True,
                                        resize="segmentation_map")

        col0 = self.col(0)
        col1 = self.col(1)
        expected = np.uint8([
            [col0, col1, col1],
            [col0, col1, col1],
            [col0, col1, col1]
        ])
        expected = a0 * image + a1 * expected
        d_max = np.max(
            np.abs(
                observed[0].astype(np.float32)
                - expected.astype(np.float32)
            )
        )
        assert isinstance(observed, list)
        assert len(observed) == 1
        assert observed[0].shape == expected.shape
        assert d_max <= 1.0 + 1e-4

    def test_resize_image_to_segmentation_map(self):
        # resizing of image to segmap
        arr = np.int32([
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1]
        ])
        segmap = ia.SegmentationMapsOnImage(arr, shape=(1, 3))

        image = np.uint8([[0, 10, 20]])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
        image_rs = ia.imresize_single_image(
            image, arr.shape[0:2], interpolation="cubic")

        a1 = 0.7
        a0 = 1.0 - a1

        observed = segmap.draw_on_image(image, alpha=a1, draw_background=True,
                                        resize="image")

        col0 = self.col(0)
        col1 = self.col(1)
        expected = np.uint8([
            [col0, col1, col1],
            [col0, col1, col1],
            [col0, col1, col1]
        ])
        expected = a0 * image_rs + a1 * expected
        d_max = np.max(
            np.abs(
                observed[0].astype(np.float32)
                - expected.astype(np.float32)
            )
        )
        assert isinstance(observed, list)
        assert len(observed) == 1
        assert observed[0].shape == expected.shape
        assert d_max <= 1.0 + 1e-4

    def test_background_threshold_leads_to_deprecation_warning(self):
        arr = np.zeros((1, 1, 1), dtype=np.int32)
        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))
        image = np.zeros((1, 1, 3), dtype=np.uint8)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = segmap.draw_on_image(image, background_threshold=0.01)

        assert len(caught_warnings) == 1
        assert (
            "The argument `background_threshold` is deprecated"
            in str(caught_warnings[0].message)
        )


class TestSegmentationMapsOnImage_pad(unittest.TestCase):
    @property
    def segmap(self):
        arr = np.int32([
            [0, 1, 1],
            [0, 2, 1],
            [0, 1, 3]
        ])
        return ia.SegmentationMapsOnImage(arr, shape=(3, 3))

    def test_default_pad_mode_and_cval(self):
        segmap_padded = self.segmap.pad(top=1, right=2, bottom=3, left=4)
        observed = segmap_padded.arr

        expected = np.pad(
            self.segmap.arr,
            ((1, 3), (4, 2), (0, 0)),
            mode="constant",
            constant_values=0)
        assert np.array_equal(observed, expected)

    def test_default_pad_mode(self):
        segmap_padded = self.segmap.pad(top=1, right=2, bottom=3, left=4,
                                        cval=1.0)
        observed = segmap_padded.arr

        expected = np.pad(
            self.segmap.arr,
            ((1, 3), (4, 2), (0, 0)),
            mode="constant",
            constant_values=1.0)
        assert np.array_equal(observed, expected)

    def test_default_cval(self):
        segmap_padded = self.segmap.pad(top=1, right=2, bottom=3, left=4,
                                        mode="edge")
        observed = segmap_padded.arr

        expected = np.pad(
            self.segmap.arr,
            ((1, 3), (4, 2), (0, 0)),
            mode="edge")
        assert np.array_equal(observed, expected)


class TestSegmentationMapsOnImage_pad_to_aspect_ratio(unittest.TestCase):
    @property
    def segmap(self):
        arr = np.int32([
            [0, 1, 1],
            [0, 2, 1]
        ])
        return ia.SegmentationMapsOnImage(arr, shape=(2, 3))

    def test_square_ratio_with_default_pad_mode_and_cval(self):
        segmap_padded = self.segmap.pad_to_aspect_ratio(1.0)
        observed = segmap_padded.arr

        expected = np.pad(
            self.segmap.arr,
            ((0, 1), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0)
        assert np.array_equal(observed, expected)

    def test_square_ratio_with_cval_set(self):
        segmap_padded = self.segmap.pad_to_aspect_ratio(1.0, cval=1.0)
        observed = segmap_padded.arr

        expected = np.pad(
            self.segmap.arr,
            ((0, 1), (0, 0), (0, 0)),
            mode="constant",
            constant_values=1.0)
        assert np.array_equal(observed, expected)

    def test_square_ratio_with_pad_mode_edge(self):
        segmap_padded = self.segmap.pad_to_aspect_ratio(1.0, mode="edge")
        observed = segmap_padded.arr

        expected = np.pad(
            self.segmap.arr,
            ((0, 1), (0, 0), (0, 0)),
            mode="edge")
        assert np.array_equal(observed, expected)

    def test_higher_than_wide_ratio_with_default_pad_mode_and_cval(self):
        segmap_padded = self.segmap.pad_to_aspect_ratio(0.5)
        observed = segmap_padded.arr

        expected = np.pad(
            self.segmap.arr,
            ((2, 2), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0)
        assert np.array_equal(observed, expected)

    def test_return_pad_amounts(self):
        segmap_padded, pad_amounts = self.segmap.pad_to_aspect_ratio(
            0.5, return_pad_amounts=True)
        observed = segmap_padded.arr

        expected = np.pad(
            self.segmap.arr,
            ((2, 2), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0)
        assert np.array_equal(observed, expected)
        assert pad_amounts == (2, 0, 2, 0)


class TestSegmentationMapsOnImage_resize(unittest.TestCase):
    @property
    def segmap(self):
        arr = np.int32([
            [0, 1],
            [0, 2]
        ])
        return ia.SegmentationMapsOnImage(arr, shape=(2, 2))

    def test_resize_to_twice_the_size(self):
        for sizes in [(4, 4), 2.0]:
            with self.subTest(sizes=sizes):
                # TODO also test other interpolation modes
                segmap_scaled = self.segmap.resize(sizes)
                observed = segmap_scaled.arr

                expected = np.int32([
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 2, 2],
                    [0, 0, 2, 2],
                ]).reshape((4, 4, 1))
                assert np.array_equal(observed, expected)


class TestSegmentationMapsOnImage_copy(unittest.TestCase):
    @property
    def segmap(self):
        arr = np.int32([
            [0, 1],
            [2, 3]
        ]).reshape((2, 2, 1))
        return ia.SegmentationMapsOnImage(arr, shape=(2, 2))

    def test_copy(self):
        segmap = self.segmap

        observed = segmap.copy()

        assert np.array_equal(observed.arr, segmap.arr)
        assert observed.shape == (2, 2)
        assert observed._input_was == segmap._input_was

        # ensure shallow copy
        observed.arr[0, 0, 0] = 10
        assert segmap.arr[0, 0, 0] == 10

    def test_set_new_arr(self):
        segmap = self.segmap

        observed = segmap.copy(np.int32([[10]]).reshape((1, 1, 1)))

        assert observed.arr.shape == (1, 1, 1)
        assert observed.arr[0, 0, 0] == 10
        assert observed._input_was == segmap._input_was

    def test_set_new_shape(self):
        segmap = self.segmap

        observed = segmap.copy(shape=(10, 11, 3))

        assert observed.shape == (10, 11, 3)
        assert segmap.shape != (10, 11, 3)
        assert observed._input_was == segmap._input_was


class TestSegmentationMapsOnImage_deepcopy(unittest.TestCase):
    @property
    def segmap(self):
        arr = np.int32([
            [0, 1],
            [2, 3]
        ]).reshape((2, 2, 1))
        return ia.SegmentationMapsOnImage(arr, shape=(2, 2))

    def test_deepcopy(self):
        segmap = self.segmap

        observed = segmap.deepcopy()

        assert np.array_equal(observed.arr, segmap.arr)
        assert observed.shape == (2, 2)
        assert observed._input_was == segmap._input_was

        observed.arr[0, 0, 0] = 10
        assert segmap.arr[0, 0, 0] != 10

    def test_set_new_arr(self):
        segmap = self.segmap

        observed = segmap.deepcopy(np.int32([[10]]).reshape((1, 1, 1)))

        assert observed.arr.shape == (1, 1, 1)
        assert observed.arr[0, 0, 0] == 10
        assert segmap.arr[0, 0, 0] != 10
        assert observed._input_was == segmap._input_was

    def test_set_new_shape(self):
        segmap = self.segmap

        observed = segmap.deepcopy(shape=(10, 11, 3))

        assert observed.shape == (10, 11, 3)
        assert segmap.shape != (10, 11, 3)
        assert observed._input_was == segmap._input_was
