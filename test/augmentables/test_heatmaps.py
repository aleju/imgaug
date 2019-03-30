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

    # test_HooksImages_is_activated()
    # test_HooksImages_is_propagating()
    # test_HooksImages_preprocess()
    # test_HooksImages_postprocess()
    # test_HeatmapsOnImage_get_arr()
    # test_HeatmapsOnImage_find_global_maxima()
    test_HeatmapsOnImage_draw()
    test_HeatmapsOnImage_draw_on_image()
    test_HeatmapsOnImage_invert()
    test_HeatmapsOnImage_pad()
    test_HeatmapsOnImage_pad_to_aspect_ratio()
    test_HeatmapsOnImage_avg_pool()
    test_HeatmapsOnImage_max_pool()
    test_HeatmapsOnImage_scale()
    # test_HeatmapsOnImage_to_uint8()
    test_HeatmapsOnImage_from_uint8()
    # test_HeatmapsOnImage_from_0to1()
    test_HeatmapsOnImage_change_normalization()
    # test_HeatmapsOnImage_copy()
    # test_HeatmapsOnImage_deepcopy()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_HeatmapsOnImage_draw():
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

    for y, x in [(0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2)]:
        assert np.allclose(heatmaps_drawn[y, x], v1)

    for y, x in [(0, 0), (0, 3), (3, 0), (3, 3)]:
        assert np.allclose(heatmaps_drawn[y, x], v2)

    for y, x in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        assert np.allclose(heatmaps_drawn[y, x], v3)

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

    for y in range(4):
        for x in range(2):
            assert np.allclose(heatmaps_drawn[y, x], v1)

    for y in range(4):
        for x in range(2, 4):
            assert np.allclose(heatmaps_drawn[y, x], v2)


def test_HeatmapsOnImage_draw_on_image():
    heatmaps_arr = np.float32([
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(2, 2, 3))

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
    assert np.all(heatmaps_drawn[0:4, 2:3, :] == 128) or np.all(heatmaps_drawn[0:4, 2:3, :] == 127)
    assert np.all(heatmaps_drawn[0:4, 3:4, :] == 255) or np.all(heatmaps_drawn[0:4, 3:4, :] == 254)

    image = np.uint8([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    image = np.tile(image[..., np.newaxis], (1, 1, 3))

    heatmaps_drawn = heatmaps.draw_on_image(image, alpha=0.5, resize="image", cmap=None)[0]
    assert heatmaps_drawn.shape == (2, 2, 3)
    assert np.all(heatmaps_drawn[0:2, 0, :] == 0)
    assert np.all(heatmaps_drawn[0:2, 1, :] == 128) or np.all(heatmaps_drawn[0:2, 1, :] == 127)


def test_HeatmapsOnImage_invert():
    heatmaps_arr = np.float32([
        [0.0, 5.0, 10.0],
        [-1.0, -2.0, 7.5]
    ])
    expected = np.float32([
        [8.0, 3.0, -2.0],
        [9.0, 10.0, 0.5]
    ])

    # (H, W)
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(2, 3), min_value=-2.0, max_value=10.0)
    assert np.allclose(heatmaps.get_arr(), heatmaps_arr)
    assert np.allclose(heatmaps.invert().get_arr(), expected)

    # (H, W, 1)
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr[..., np.newaxis], shape=(2, 3), min_value=-2.0, max_value=10.0)
    assert np.allclose(heatmaps.get_arr(), heatmaps_arr[..., np.newaxis])
    assert np.allclose(heatmaps.invert().get_arr(), expected[..., np.newaxis])


def test_HeatmapsOnImage_pad():
    heatmaps_arr = np.float32([
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(2, 2, 3))

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

    heatmaps_padded = heatmaps.pad(top=1, right=2, bottom=3, left=4, cval=0.5)
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

    heatmaps_padded = heatmaps.pad(top=1, right=2, bottom=3, left=4, mode="edge")
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


def test_HeatmapsOnImage_pad_to_aspect_ratio():
    heatmaps_arr = np.float32([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(2, 2, 3))

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

    # test aspect ratio != 1.0
    heatmaps_padded = heatmaps.pad_to_aspect_ratio(2.0, cval=0.1)
    assert heatmaps_padded.arr_0to1.shape == (2, 4, 1)
    assert np.allclose(
        heatmaps_padded.arr_0to1[:, :, 0],
        np.float32([
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 1.0, 0.1]
        ])
    )

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


def test_HeatmapsOnImage_avg_pool():
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


def test_HeatmapsOnImage_max_pool():
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


def test_HeatmapsOnImage_scale():
    heatmaps_arr = np.float32([
        [0.0, 1.0]
    ])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(4, 4, 3))

    heatmaps_scaled = heatmaps.resize((4, 4), interpolation="nearest")
    assert heatmaps_scaled.arr_0to1.shape == (4, 4, 1)
    assert heatmaps_scaled.arr_0to1.dtype.type == np.float32
    assert np.allclose(
        heatmaps_scaled.arr_0to1[:, :, 0],
        np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ])
    )

    heatmaps_arr = np.float32([
        [0.0, 1.0]
    ])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(4, 4, 3))

    heatmaps_scaled = heatmaps.resize(2.0, interpolation="nearest")
    assert heatmaps_scaled.arr_0to1.shape == (2, 4, 1)
    assert heatmaps_scaled.arr_0to1.dtype.type == np.float32
    assert np.allclose(
        heatmaps_scaled.arr_0to1[:, :, 0],
        np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ])
    )


def test_HeatmapsOnImage_from_uint8():
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

    # 2d uint8 arr
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


def test_HeatmapsOnImage_change_normalization():
    # (0.0, 1.0) -> (0.0, 2.0)
    arr = np.float32([
        [0.0, 0.5, 1.0],
        [1.0, 0.5, 0.0]
    ])
    observed = ia.HeatmapsOnImage.change_normalization(arr, (0.0, 1.0), (0.0, 2.0))
    expected = np.float32([
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0]
    ])
    assert np.allclose(observed, expected)

    # (0.0, 1.0) -> (-1.0, 0.0)
    observed = ia.HeatmapsOnImage.change_normalization(arr, (0.0, 1.0), (-1.0, 0.0))
    expected = np.float32([
        [-1.0, -0.5, 0.0],
        [0.0, -0.5, -1.0]
    ])
    assert np.allclose(observed, expected)

    # (-1.0, 1.0) -> (1.0, 3.0)
    arr = np.float32([
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0]
    ])
    observed = ia.HeatmapsOnImage.change_normalization(arr, (-1.0, 1.0), (1.0, 3.0))
    expected = np.float32([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0]
    ])
    assert np.allclose(observed, expected)

    # (-1.0, 1.0) -> (1.0, 3.0)
    # value ranges given as HeatmapsOnImage
    arr = np.float32([
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0]
    ])
    source = ia.HeatmapsOnImage(np.float32([[0.0]]), min_value=-1.0, max_value=1.0, shape=(1, 1, 3))
    target = ia.HeatmapsOnImage(np.float32([[1.0]]), min_value=1.0, max_value=3.0, shape=(1, 1, 3))
    observed = ia.HeatmapsOnImage.change_normalization(arr, source, target)
    expected = np.float32([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0]
    ])
    assert np.allclose(observed, expected)
