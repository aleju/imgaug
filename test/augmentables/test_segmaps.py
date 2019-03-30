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

import imgaug as ia


def main():
    time_start = time.time()

    test_SegmentationMapOnImage_bool()
    test_SegmentationMapOnImage_get_arr_int()
    # test_SegmentationMapOnImage_get_arr_bool()
    test_SegmentationMapOnImage_draw()
    test_SegmentationMapOnImage_draw_on_image()
    test_SegmentationMapOnImage_pad()
    test_SegmentationMapOnImage_pad_to_aspect_ratio()
    test_SegmentationMapOnImage_scale()
    test_SegmentationMapOnImage_to_heatmaps()
    test_SegmentationMapOnImage_from_heatmaps()
    test_SegmentationMapOnImage_copy()
    test_SegmentationMapOnImage_deepcopy()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_SegmentationMapOnImage_bool():
    # Test for #189 (boolean mask inputs into SegmentationMapOnImage not working)
    arr = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=bool)
    assert arr.dtype.type == np.bool_
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3))
    observed = segmap.get_arr_int()
    assert observed.dtype.type == np.int32
    assert np.array_equal(arr, observed)

    arr = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.bool)
    assert arr.dtype.type == np.bool_
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3))
    observed = segmap.get_arr_int()
    assert observed.dtype.type == np.int32
    assert np.array_equal(arr, observed)


def test_SegmentationMapOnImage_get_arr_int():
    arr = np.int32([
        [0, 0, 1],
        [0, 2, 1],
        [1, 3, 1]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3), nb_classes=4)
    observed = segmap.get_arr_int()
    assert observed.dtype.type == np.int32
    assert np.array_equal(arr, observed)

    arr_c0 = np.float32([
        [0.1, 0.1, 0.1],
        [0.1, 0.9, 0.1],
        [0.0, 0.1, 0.0]
    ])
    arr_c1 = np.float32([
        [0.2, 1.0, 0.2],
        [0.2, 0.8, 0.2],
        [0.0, 0.0, 0.0]
    ])
    arr_c2 = np.float32([
        [0.0, 0.0, 0.0],
        [0.3, 0.7, 0.3],
        [0.1, 0.0, 0.0001]
    ])
    arr = np.concatenate([
        arr_c0[..., np.newaxis],
        arr_c1[..., np.newaxis],
        arr_c2[..., np.newaxis]
    ], axis=2)
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3))
    observed = segmap.get_arr_int()
    expected = np.int32([
        [2, 2, 2],
        [3, 1, 3],
        [3, 1, 0]
    ])
    assert observed.dtype.type == np.int32
    assert np.array_equal(observed, expected)

    got_exception = False
    try:
        _ = segmap.get_arr_int(background_class_id=2)
    except Exception as exc:
        assert "The background class id may only be changed if " in str(exc)
        got_exception = True
    assert got_exception

    observed = segmap.get_arr_int(background_threshold=0.21)
    expected = np.int32([
        [0, 2, 0],
        [3, 1, 3],
        [0, 0, 0]
    ])
    assert observed.dtype.type == np.int32
    assert np.array_equal(observed, expected)


def test_SegmentationMapOnImage_draw():
    arr = np.int32([
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3), nb_classes=2)

    # simple example with 2 classes
    observed = segmap.draw()
    col0 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[0]
    col1 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[1]
    expected = np.uint8([
        [col0, col1, col1],
        [col0, col1, col1],
        [col0, col1, col1]
    ])
    assert np.array_equal(observed, expected)

    # same example, with resizing to 2x the size
    observed = segmap.draw(size=(6, 6))
    expected = ia.imresize_single_image(expected, (6, 6), interpolation="nearest")
    assert np.array_equal(observed, expected)

    # custom choice of colors
    col0 = (10, 10, 10)
    col1 = (50, 51, 52)
    observed = segmap.draw(colors=[col0, col1])
    expected = np.uint8([
        [col0, col1, col1],
        [col0, col1, col1],
        [col0, col1, col1]
    ])
    assert np.array_equal(observed, expected)

    # background_threshold, background_class and foreground mask
    arr_c0 = np.float32([
        [0, 0, 0],
        [1.0, 0, 0],
        [0, 0, 0]
    ])
    arr_c1 = np.float32([
        [0, 1, 1],
        [0, 1, 1],
        [0.1, 1, 1]
    ])
    arr = np.concatenate([
        arr_c0[..., np.newaxis],
        arr_c1[..., np.newaxis]
    ], axis=2)
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3))

    observed, observed_fg = segmap.draw(background_threshold=0.01, return_foreground_mask=True)
    col0 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[0]
    col1 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[1]
    col2 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[2]
    expected = np.uint8([
        [col0, col2, col2],
        [col1, col2, col2],
        [col2, col2, col2]
    ])
    expected_fg = np.array([
        [False, True, True],
        [True, True, True],
        [True, True, True]
    ], dtype=np.bool)
    assert np.array_equal(observed, expected)
    assert np.array_equal(observed_fg, expected_fg)

    # background_threshold, background_class and foreground mask
    # here with higher threshold so that bottom left pixel switches to background
    observed, observed_fg = segmap.draw(background_threshold=0.11, return_foreground_mask=True)
    col0 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[0]
    col1 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[1]
    col2 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[2]
    expected = np.uint8([
        [col0, col2, col2],
        [col1, col2, col2],
        [col0, col2, col2]
    ])
    expected_fg = np.array([
        [False, True, True],
        [True, True, True],
        [False, True, True]
    ], dtype=np.bool)
    assert np.array_equal(observed, expected)
    assert np.array_equal(observed_fg, expected_fg)


def test_SegmentationMapOnImage_draw_on_image():
    arr = np.int32([
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3), nb_classes=2)

    image = np.uint8([
        [0, 10, 20],
        [30, 40, 50],
        [60, 70, 80]
    ])
    image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

    # only image visible
    observed = segmap.draw_on_image(image, alpha=0)
    assert np.array_equal(observed, image)

    # only segmap visible
    observed = segmap.draw_on_image(image, alpha=1.0, draw_background=True)
    col0 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[0]
    col1 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[1]
    expected = np.uint8([
        [col0, col1, col1],
        [col0, col1, col1],
        [col0, col1, col1]
    ])
    assert np.array_equal(observed, expected)

    # only segmap visible - in foreground
    observed = segmap.draw_on_image(image, alpha=1.0, draw_background=False)
    col1 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[1]
    expected = np.uint8([
        [image[0, 0, :], col1, col1],
        [image[1, 0, :], col1, col1],
        [image[2, 0, :], col1, col1]
    ])
    assert np.array_equal(observed, expected)

    # overlay without background drawn
    a1 = 0.7
    a0 = 1.0 - a1
    observed = segmap.draw_on_image(image, alpha=a1, draw_background=False)
    col1 = np.uint8(ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[1])
    expected = np.float32([
        [image[0, 0, :], a0*image[0, 1, :] + a1*col1, a0*image[0, 2, :] + a1*col1],
        [image[1, 0, :], a0*image[1, 1, :] + a1*col1, a0*image[1, 2, :] + a1*col1],
        [image[2, 0, :], a0*image[2, 1, :] + a1*col1, a0*image[2, 2, :] + a1*col1]
    ])
    d_max = np.max(np.abs(observed.astype(np.float32) - expected))
    assert observed.shape == expected.shape
    assert d_max <= 1.0 + 1e-4

    # overlay with background drawn
    a1 = 0.7
    a0 = 1.0 - a1
    observed = segmap.draw_on_image(image, alpha=a1, draw_background=True)
    col0 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[0]
    col1 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[1]
    expected = np.uint8([
        [col0, col1, col1],
        [col0, col1, col1],
        [col0, col1, col1]
    ])
    expected = a0 * image + a1 * expected
    d_max = np.max(np.abs(observed.astype(np.float32) - expected.astype(np.float32)))
    assert observed.shape == expected.shape
    assert d_max <= 1.0 + 1e-4

    # resizing of segmap to image
    arr = np.int32([
        [0, 1, 1]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3), nb_classes=2)

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
    d_max = np.max(np.abs(observed.astype(np.float32) - expected.astype(np.float32)))
    assert observed.shape == expected.shape
    assert d_max <= 1.0 + 1e-4

    # resizing of image to segmap
    arr = np.int32([
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(1, 3), nb_classes=2)

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
    d_max = np.max(np.abs(observed.astype(np.float32) - expected.astype(np.float32)))
    assert observed.shape == expected.shape
    assert d_max <= 1.0 + 1e-4


def test_SegmentationMapOnImage_pad():
    arr = np.int32([
        [0, 1, 1],
        [0, 2, 1],
        [0, 1, 3]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3), nb_classes=4)

    segmap_padded = segmap.pad(top=1, right=2, bottom=3, left=4)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((1, 3), (4, 2), (0, 0)), mode="constant", constant_values=0)
    assert np.allclose(observed, expected)

    segmap_padded = segmap.pad(top=1, right=2, bottom=3, left=4, cval=1.0)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((1, 3), (4, 2), (0, 0)), mode="constant", constant_values=1.0)
    assert np.allclose(observed, expected)

    segmap_padded = segmap.pad(top=1, right=2, bottom=3, left=4, mode="edge")
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((1, 3), (4, 2), (0, 0)), mode="edge")
    assert np.allclose(observed, expected)


def test_SegmentationMapOnImage_pad_to_aspect_ratio():
    arr = np.int32([
        [0, 1, 1],
        [0, 2, 1]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(2, 3), nb_classes=3)

    segmap_padded = segmap.pad_to_aspect_ratio(1.0)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=0)
    assert np.allclose(observed, expected)

    segmap_padded = segmap.pad_to_aspect_ratio(1.0, cval=1.0)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=1.0)
    assert np.allclose(observed, expected)

    segmap_padded = segmap.pad_to_aspect_ratio(1.0, mode="edge")
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((0, 1), (0, 0), (0, 0)), mode="edge")
    assert np.allclose(observed, expected)

    segmap_padded = segmap.pad_to_aspect_ratio(0.5)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((2, 2), (0, 0), (0, 0)), mode="constant", constant_values=0)
    assert np.allclose(observed, expected)

    segmap_padded, pad_amounts = segmap.pad_to_aspect_ratio(0.5, return_pad_amounts=True)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((2, 2), (0, 0), (0, 0)), mode="constant", constant_values=0)
    assert np.allclose(observed, expected)
    assert pad_amounts == (2, 0, 2, 0)


def test_SegmentationMapOnImage_scale():
    arr = np.int32([
        [0, 1],
        [0, 2]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(2, 2), nb_classes=3)

    segmap_scaled = segmap.resize((4, 4))
    observed = segmap_scaled.arr
    expected = np.clip(ia.imresize_single_image(segmap.arr, (4, 4), interpolation="cubic"), 0, 1.0)
    assert np.allclose(observed, expected)
    assert np.array_equal(segmap_scaled.get_arr_int(), np.int32([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ]))

    segmap_scaled = segmap.resize((4, 4), interpolation="nearest")
    observed = segmap_scaled.arr
    expected = ia.imresize_single_image(segmap.arr, (4, 4), interpolation="nearest")
    assert np.allclose(observed, expected)
    assert np.array_equal(segmap_scaled.get_arr_int(), np.int32([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ]))

    segmap_scaled = segmap.resize(2.0)
    observed = segmap_scaled.arr
    expected = np.clip(ia.imresize_single_image(segmap.arr, 2.0, interpolation="cubic"), 0, 1.0)
    assert np.allclose(observed, expected)
    assert np.array_equal(segmap_scaled.get_arr_int(), np.int32([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ]))


def test_SegmentationMapOnImage_to_heatmaps():
    arr = np.int32([
        [0, 1],
        [0, 2]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(2, 2), nb_classes=3)
    heatmaps = segmap.to_heatmaps()
    expected_c0 = np.float32([
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    expected_c1 = np.float32([
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    expected_c2 = np.float32([
        [0.0, 0.0],
        [0.0, 1.0]
    ])
    expected = np.concatenate([
        expected_c0[..., np.newaxis],
        expected_c1[..., np.newaxis],
        expected_c2[..., np.newaxis]
    ], axis=2)
    assert np.allclose(heatmaps.arr_0to1, expected)

    # only_nonempty when all are nonempty
    heatmaps, class_indices = segmap.to_heatmaps(only_nonempty=True)
    expected_c0 = np.float32([
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    expected_c1 = np.float32([
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    expected_c2 = np.float32([
        [0.0, 0.0],
        [0.0, 1.0]
    ])
    expected = np.concatenate([
        expected_c0[..., np.newaxis],
        expected_c1[..., np.newaxis],
        expected_c2[..., np.newaxis]
    ], axis=2)
    assert np.allclose(heatmaps.arr_0to1, expected)
    assert len(class_indices) == 3
    assert [idx in class_indices for idx in [0, 1, 2]]

    # only_nonempty when one is empty and two are nonempty
    arr = np.int32([
        [0, 2],
        [0, 2]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(2, 2), nb_classes=3)
    heatmaps, class_indices = segmap.to_heatmaps(only_nonempty=True)
    expected_c0 = np.float32([
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    expected_c2 = np.float32([
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    expected = np.concatenate([
        expected_c0[..., np.newaxis],
        expected_c2[..., np.newaxis]
    ], axis=2)
    assert np.allclose(heatmaps.arr_0to1, expected)
    assert len(class_indices) == 2
    assert [idx in class_indices for idx in [0, 2]]

    # only_nonempty when all are empty
    arr_c0 = np.float32([
        [0.0, 0.0],
        [0.0, 0.0]
    ])
    arr = arr_c0[..., np.newaxis]
    segmap = ia.SegmentationMapOnImage(arr, shape=(2, 2), nb_classes=3)
    heatmaps, class_indices = segmap.to_heatmaps(only_nonempty=True)
    assert heatmaps is None
    assert len(class_indices) == 0

    # only_nonempty when all are empty and not_none_if_no_nonempty is True
    arr_c0 = np.float32([
        [0.0, 0.0],
        [0.0, 0.0]
    ])
    arr = arr_c0[..., np.newaxis]
    segmap = ia.SegmentationMapOnImage(arr, shape=(2, 2), nb_classes=3)
    heatmaps, class_indices = segmap.to_heatmaps(only_nonempty=True, not_none_if_no_nonempty=True)
    assert np.allclose(heatmaps.arr_0to1, np.zeros((2, 2), dtype=np.float32))
    assert len(class_indices) == 1
    assert [idx in class_indices for idx in [0]]


def test_SegmentationMapOnImage_from_heatmaps():
    arr_c0 = np.float32([
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    arr_c1 = np.float32([
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    arr = np.concatenate([arr_c0[..., np.newaxis], arr_c1[..., np.newaxis]], axis=2)
    heatmaps = ia.HeatmapsOnImage.from_0to1(arr, shape=(2, 2))

    segmap = ia.SegmentationMapOnImage.from_heatmaps(heatmaps)
    assert np.allclose(segmap.arr, arr)

    # with class_indices
    arr_c0 = np.float32([
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    arr_c2 = np.float32([
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    arr = np.concatenate([arr_c0[..., np.newaxis], arr_c2[..., np.newaxis]], axis=2)
    heatmaps = ia.HeatmapsOnImage.from_0to1(arr, shape=(2, 2))

    segmap = ia.SegmentationMapOnImage.from_heatmaps(heatmaps, class_indices=[0, 2], nb_classes=4)
    expected_c0 = np.copy(arr_c0)
    expected_c1 = np.zeros(arr_c0.shape)
    expected_c2 = np.copy(arr_c2)
    expected_c3 = np.zeros(arr_c0.shape)
    expected = np.concatenate([
        expected_c0[..., np.newaxis],
        expected_c1[..., np.newaxis],
        expected_c2[..., np.newaxis],
        expected_c3[..., np.newaxis]
    ], axis=2)
    assert np.allclose(segmap.arr, expected)


def test_SegmentationMapOnImage_copy():
    arr_c0 = np.float32([
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    arr_c1 = np.float32([
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    arr = np.concatenate([arr_c0[..., np.newaxis], arr_c1[..., np.newaxis]], axis=2)
    segmap = ia.SegmentationMapOnImage(arr, shape=(2, 2))
    observed = segmap.copy()
    assert np.allclose(observed.arr, segmap.arr)
    assert observed.shape == (2, 2)
    assert observed.nb_classes == segmap.nb_classes
    assert observed.input_was == segmap.input_was

    arr = np.int32([
        [0, 1],
        [2, 3]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(2, 2), nb_classes=10)
    observed = segmap.copy()
    assert np.array_equal(observed.get_arr_int(), arr)
    assert observed.shape == (2, 2)
    assert observed.nb_classes == 10
    assert observed.input_was == segmap.input_was


def test_SegmentationMapOnImage_deepcopy():
    arr_c0 = np.float32([
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    arr_c1 = np.float32([
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    arr = np.concatenate([arr_c0[..., np.newaxis], arr_c1[..., np.newaxis]], axis=2)
    segmap = ia.SegmentationMapOnImage(arr, shape=(2, 2))
    observed = segmap.deepcopy()
    assert np.allclose(observed.arr, segmap.arr)
    assert observed.shape == (2, 2)
    assert observed.nb_classes == segmap.nb_classes
    assert observed.input_was == segmap.input_was
    segmap.arr[0, 0, 0] = 0.0
    assert not np.allclose(observed.arr, segmap.arr)

    arr = np.int32([
        [0, 1],
        [2, 3]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(2, 2), nb_classes=10)
    observed = segmap.deepcopy()
    assert np.array_equal(observed.get_arr_int(), segmap.get_arr_int())
    assert observed.shape == (2, 2)
    assert observed.nb_classes == 10
    assert observed.input_was == segmap.input_was
    segmap.arr[0, 0, 0] = 0.0
    segmap.arr[0, 0, 1] = 1.0
    assert not np.array_equal(observed.get_arr_int(), segmap.get_arr_int())
