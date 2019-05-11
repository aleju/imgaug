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


def main():
    time_start = time.time()

    test_SegmentationMapsOnImage___init__()
    test_SegmentationMapsOnImage_bool()
    test_SegmentationMapsOnImage_get_arr()
    test_SegmentationMapsOnImage_draw()
    test_SegmentationMapsOnImage_draw_on_image()
    test_SegmentationMapsOnImage_pad()
    test_SegmentationMapsOnImage_pad_to_aspect_ratio()
    test_SegmentationMapsOnImage_resize()
    test_SegmentationMapsOnImage_copy()
    test_SegmentationMapsOnImage_deepcopy()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_SegmentationMapsOnImage___init__():
    dtypes = ["int8", "int16", "int32", "uint8", "uint16"]
    ndims = [2, 3]
    img_shapes = [(3, 3), (3, 3, 3), (4, 5, 3)]

    # int and uint dtypes
    for dtype, ndim, img_shape in itertools.product(dtypes, ndims, img_shapes):
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

    # bool dtypes
    for ndim in ndims:
        shape = (3, 3) if ndim == 2 else (3, 3, 1)
        arr = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ], dtype=bool).reshape(shape)
        segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))
        assert segmap.shape == (3, 3)
        assert segmap.arr.dtype.name == "int32"
        assert segmap.arr.shape == (3, 3, 1)
        assert np.array_equal(segmap.arr,
                              arr.reshape((3, 3, 1)).astype(np.int32))

        if ndim == 3:
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

    # uint32 is not allowed
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


def test_SegmentationMapsOnImage_bool():
    # Test for #189 (boolean mask inputs into SegmentationMapsOnImage not working)
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


def test_SegmentationMapsOnImage_get_arr():
    dtypes = ["int8", "int16", "int32", "uint8", "uint16"]
    ndims = [2, 3]

    for dtype, ndim in itertools.product(dtypes, ndims):
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

    for ndim in ndims:
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


def test_SegmentationMapsOnImage_draw():
    arr = np.int32([
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1]
    ])
    segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))

    # simple example with 2 classes
    observed = segmap.draw()
    col0 = ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS[0]
    col1 = ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS[1]
    expected = np.uint8([
        [col0, col1, col1],
        [col0, col1, col1],
        [col0, col1, col1]
    ])
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert np.array_equal(observed[0], expected)

    # same example, with resizing to 2x the size
    observed = segmap.draw(size=(6, 6))
    expected = ia.imresize_single_image(expected, (6, 6), interpolation="nearest")
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert np.array_equal(observed[0], expected)

    # custom choice of colors
    col0 = (10, 10, 10)
    col1 = (50, 51, 52)
    observed = segmap.draw(colors=[col0, col1])
    expected = np.uint8([
        [col0, col1, col1],
        [col0, col1, col1],
        [col0, col1, col1]
    ])
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert np.array_equal(observed[0], expected)

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


def test_SegmentationMapsOnImage_draw_on_image():
    arr = np.int32([
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1]
    ])
    segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))

    image = np.uint8([
        [0, 10, 20],
        [30, 40, 50],
        [60, 70, 80]
    ])
    image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

    # only image visible
    observed = segmap.draw_on_image(image, alpha=0)
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert np.array_equal(observed[0], image)

    # only segmap visible
    observed = segmap.draw_on_image(image, alpha=1.0, draw_background=True)
    col0 = ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS[0]
    col1 = ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS[1]
    expected = np.uint8([
        [col0, col1, col1],
        [col0, col1, col1],
        [col0, col1, col1]
    ])
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert np.array_equal(observed[0], expected)

    # only segmap visible - in foreground
    observed = segmap.draw_on_image(image, alpha=1.0, draw_background=False)
    col1 = ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS[1]
    expected = np.uint8([
        [image[0, 0, :], col1, col1],
        [image[1, 0, :], col1, col1],
        [image[2, 0, :], col1, col1]
    ])
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert np.array_equal(observed[0], expected)

    # only segmap visible in foreground + multiple channels in segmap
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
    observed = segmap_multi.draw_on_image(image, alpha=1.0, draw_background=False)
    assert isinstance(observed, list)
    assert len(observed) == 3
    assert np.array_equal(observed[0], expected_channel_1)
    assert np.array_equal(observed[1], expected_channel_2)
    assert np.array_equal(observed[2], expected_channel_3)

    # overlay without background drawn
    a1 = 0.7
    a0 = 1.0 - a1
    observed = segmap.draw_on_image(image, alpha=a1, draw_background=False)
    col1 = np.uint8(ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS[1])
    expected = np.float32([
        [image[0, 0, :], a0*image[0, 1, :] + a1*col1, a0*image[0, 2, :] + a1*col1],
        [image[1, 0, :], a0*image[1, 1, :] + a1*col1, a0*image[1, 2, :] + a1*col1],
        [image[2, 0, :], a0*image[2, 1, :] + a1*col1, a0*image[2, 2, :] + a1*col1]
    ])
    d_max = np.max(np.abs(observed[0].astype(np.float32) - expected))
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert observed[0].shape == expected.shape
    assert d_max <= 1.0 + 1e-4

    # overlay with background drawn
    a1 = 0.7
    a0 = 1.0 - a1
    observed = segmap.draw_on_image(image, alpha=a1, draw_background=True)
    col0 = ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS[0]
    col1 = ia.SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS[1]
    expected = np.uint8([
        [col0, col1, col1],
        [col0, col1, col1],
        [col0, col1, col1]
    ])
    expected = a0 * image + a1 * expected
    d_max = np.max(np.abs(observed[0].astype(np.float32) - expected.astype(np.float32)))
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert observed[0].shape == expected.shape
    assert d_max <= 1.0 + 1e-4

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
    observed = segmap.draw_on_image(image, alpha=a1, draw_background=True, resize="segmentation_map")
    expected = np.uint8([
        [col0, col1, col1],
        [col0, col1, col1],
        [col0, col1, col1]
    ])
    expected = a0 * image + a1 * expected
    d_max = np.max(np.abs(observed[0].astype(np.float32) - expected.astype(np.float32)))
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert observed[0].shape == expected.shape
    assert d_max <= 1.0 + 1e-4

    # resizing of image to segmap
    arr = np.int32([
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1]
    ])
    segmap = ia.SegmentationMapsOnImage(arr, shape=(1, 3))

    image = np.uint8([
        [0, 10, 20]
    ])
    image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
    image_rs = ia.imresize_single_image(image, arr.shape[0:2], interpolation="cubic")

    a1 = 0.7
    a0 = 1.0 - a1
    observed = segmap.draw_on_image(image, alpha=a1, draw_background=True, resize="image")
    expected = np.uint8([
        [col0, col1, col1],
        [col0, col1, col1],
        [col0, col1, col1]
    ])
    expected = a0 * image_rs + a1 * expected
    d_max = np.max(np.abs(observed[0].astype(np.float32) - expected.astype(np.float32)))
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert observed[0].shape == expected.shape
    assert d_max <= 1.0 + 1e-4


def test_SegmentationMapsOnImage_pad():
    arr = np.int32([
        [0, 1, 1],
        [0, 2, 1],
        [0, 1, 3]
    ])
    segmap = ia.SegmentationMapsOnImage(arr, shape=(3, 3))

    segmap_padded = segmap.pad(top=1, right=2, bottom=3, left=4)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((1, 3), (4, 2), (0, 0)), mode="constant", constant_values=0)
    assert np.array_equal(observed, expected)

    segmap_padded = segmap.pad(top=1, right=2, bottom=3, left=4, cval=1.0)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((1, 3), (4, 2), (0, 0)), mode="constant", constant_values=1.0)
    assert np.array_equal(observed, expected)

    segmap_padded = segmap.pad(top=1, right=2, bottom=3, left=4, mode="edge")
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((1, 3), (4, 2), (0, 0)), mode="edge")
    assert np.array_equal(observed, expected)


def test_SegmentationMapsOnImage_pad_to_aspect_ratio():
    arr = np.int32([
        [0, 1, 1],
        [0, 2, 1]
    ])
    segmap = ia.SegmentationMapsOnImage(arr, shape=(2, 3))

    segmap_padded = segmap.pad_to_aspect_ratio(1.0)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=0)
    assert np.array_equal(observed, expected)

    segmap_padded = segmap.pad_to_aspect_ratio(1.0, cval=1.0)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=1.0)
    assert np.array_equal(observed, expected)

    segmap_padded = segmap.pad_to_aspect_ratio(1.0, mode="edge")
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((0, 1), (0, 0), (0, 0)), mode="edge")
    assert np.array_equal(observed, expected)

    segmap_padded = segmap.pad_to_aspect_ratio(0.5)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((2, 2), (0, 0), (0, 0)), mode="constant", constant_values=0)
    assert np.array_equal(observed, expected)

    segmap_padded, pad_amounts = segmap.pad_to_aspect_ratio(0.5, return_pad_amounts=True)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((2, 2), (0, 0), (0, 0)), mode="constant", constant_values=0)
    assert np.array_equal(observed, expected)
    assert pad_amounts == (2, 0, 2, 0)


def test_SegmentationMapsOnImage_resize():
    arr = np.int32([
        [0, 1],
        [0, 2]
    ])
    segmap = ia.SegmentationMapsOnImage(arr, shape=(2, 2))

    for factor in [(4, 4), 2.0]:
        # TODO also test other interpolation modes
        segmap_scaled = segmap.resize(factor)
        observed = segmap_scaled.arr
        expected = ia.imresize_single_image(segmap.arr, (4, 4), interpolation="nearest")
        assert np.array_equal(observed, expected)
        expected = np.int32([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ]).reshape((4, 4, 1))
        assert np.array_equal(segmap_scaled.arr, expected)


def test_SegmentationMapsOnImage_copy():
    arr = np.int32([
        [0, 1],
        [2, 3]
    ]).reshape((2, 2, 1))
    segmap = ia.SegmentationMapsOnImage(arr, shape=(2, 2))
    observed = segmap.copy()
    assert np.array_equal(observed.arr, segmap.arr)
    assert observed.shape == (2, 2)
    assert observed._input_was == segmap._input_was

    observed.arr[0, 0, 0] = 10
    assert segmap.arr[0, 0, 0] == 10
    assert arr[0, 0, 0] == 10

    observed = segmap.copy(np.int32([[10]]).reshape((1, 1, 1)))
    assert observed.arr.shape == (1, 1, 1)
    assert observed.arr[0, 0, 0] == 10
    assert observed._input_was == segmap._input_was

    observed = segmap.copy(shape=(10, 11, 3))
    assert observed.shape == (10, 11, 3)
    assert segmap.shape != (10, 11, 3)
    assert observed._input_was == segmap._input_was


def test_SegmentationMapsOnImage_deepcopy():
    arr = np.int32([
        [0, 1],
        [2, 3]
    ]).reshape((2, 2, 1))
    segmap = ia.SegmentationMapsOnImage(arr, shape=(2, 2))
    observed = segmap.deepcopy()
    assert np.array_equal(observed.arr, segmap.arr)
    assert observed.shape == (2, 2)
    assert observed._input_was == segmap._input_was

    observed.arr[0, 0, 0] = 10
    assert segmap.arr[0, 0, 0] != 10
    assert arr[0, 0, 0] != 10

    observed = segmap.copy(np.int32([[10]]).reshape((1, 1, 1)))
    assert observed.arr.shape == (1, 1, 1)
    assert observed.arr[0, 0, 0] == 10
    assert segmap.arr[0, 0, 0] != 10
    assert observed._input_was == segmap._input_was

    observed = segmap.copy(shape=(10, 11, 3))
    assert observed.shape == (10, 11, 3)
    assert segmap.shape != (10, 11, 3)
    assert observed._input_was == segmap._input_was
