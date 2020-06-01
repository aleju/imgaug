from __future__ import print_function, division, absolute_import

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

import numpy as np

import imgaug as ia
import imgaug.data as iadata
from imgaug.data import _quokka_normalize_extract, _compute_resized_shape


def test__quokka_normalize_extract():
    observed = _quokka_normalize_extract("square")
    assert isinstance(observed, ia.BoundingBox)
    assert observed.x1 == 0
    assert observed.y1 == 0
    assert observed.x2 == 643
    assert observed.y2 == 643

    observed = _quokka_normalize_extract((1, 1, 644, 642))
    assert isinstance(observed, ia.BoundingBox)
    assert observed.x1 == 1
    assert observed.y1 == 1
    assert observed.x2 == 644
    assert observed.y2 == 642

    observed = _quokka_normalize_extract(ia.BoundingBox(x1=1, y1=1, x2=644, y2=642))
    assert isinstance(observed, ia.BoundingBox)
    assert observed.x1 == 1
    assert observed.y1 == 1
    assert observed.x2 == 644
    assert observed.y2 == 642

    observed = _quokka_normalize_extract(
        ia.BoundingBoxesOnImage([ia.BoundingBox(x1=1, y1=1, x2=644, y2=642)], shape=(643, 960, 3))
    )
    assert isinstance(observed, ia.BoundingBox)
    assert observed.x1 == 1
    assert observed.y1 == 1
    assert observed.x2 == 644
    assert observed.y2 == 642

    got_exception = False
    try:
        _ = _quokka_normalize_extract(False)
    except Exception as exc:
        assert "Expected 'square' or tuple" in str(exc)
        got_exception = True
    assert got_exception


def test__compute_resized_shape():
    # tuple of ints
    from_shape = (10, 15, 3)
    to_shape = (20, 30)
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == (20, 30, 3)

    from_shape = (10, 15, 3)
    to_shape = (20, 30, 3)
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == (20, 30, 3)

    # tuple of floats
    from_shape = (10, 15, 3)
    to_shape = (2.0, 3.0)
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == (20, 45, 3)

    # tuple of int and float
    from_shape = (10, 15, 3)
    to_shape = (2.0, 25)
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == (20, 25, 3)

    from_shape = (10, 17, 3)
    to_shape = (15, 2.0)
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == (15, 34, 3)

    # None
    from_shape = (10, 10, 3)
    to_shape = None
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == from_shape

    # tuple containing None
    from_shape = (10, 15, 3)
    to_shape = (2.0, None)
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == (20, 15, 3)

    from_shape = (10, 15, 3)
    to_shape = (None, 25)
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == (10, 25, 3)

    # single int
    from_shape = (10, 15, 3)
    to_shape = 20
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == (20, 20, 3)

    # single float
    from_shape = (10, 15, 3)
    to_shape = 2.0
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == (20, 30, 3)

    # from/to shape as arrays
    from_shape = (10, 10, 3)
    to_shape = (20, 30, 3)
    observed = _compute_resized_shape(np.zeros(from_shape), np.zeros(to_shape))
    assert observed == to_shape

    # from_shape is 2D
    from_shape = (10, 15)
    to_shape = (20, 30)
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == to_shape

    from_shape = (10, 15)
    to_shape = (20, 30, 3)
    observed = _compute_resized_shape(from_shape, to_shape)
    assert observed == (20, 30, 3)


def test_quokka():
    # we are intentionally a bit looser here with atol=0.1, because apparently
    # on some systems there are small differences in what exactly is loaded,
    # see issue #414

    img = iadata.quokka()
    assert img.shape == (643, 960, 3)
    assert np.allclose(
        np.average(img, axis=(0, 1)),
        [107.93576659, 118.18765066, 122.99378564],
        rtol=0, atol=0.1
    )

    img = iadata.quokka(extract="square")
    assert img.shape == (643, 643, 3)
    assert np.allclose(
        np.average(img, axis=(0, 1)),
        [111.25929196, 121.19431175, 125.71316898],
        rtol=0, atol=0.1
    )

    img = iadata.quokka(size=(642, 959))
    assert img.shape == (642, 959, 3)
    assert np.allclose(
        np.average(img, axis=(0, 1)),
        [107.84615822, 118.09832412, 122.90446467],
        rtol=0, atol=0.1
    )


def test_quokka_square():
    # we are intentionally a bit looser here with atol=0.1, because apparently
    # on some systems there are small differences in what exactly is loaded,
    # see issue #414

    img = iadata.quokka_square()
    assert img.shape == (643, 643, 3)
    assert np.allclose(
        np.average(img, axis=(0, 1)),
        [111.25929196, 121.19431175, 125.71316898],
        rtol=0, atol=0.1
    )


def test_quokka_heatmap():
    # we are intentionally a bit looser here with atol=0.1, because apparently
    # on some systems there are small differences in what exactly is loaded,
    # see issue #414

    hm = iadata.quokka_heatmap()
    assert hm.shape == (643, 960, 3)
    assert hm.arr_0to1.shape == (643, 960, 1)
    assert np.allclose(np.average(hm.arr_0to1), 0.57618505, rtol=0, atol=1e-3)

    hm = iadata.quokka_heatmap(extract="square")
    assert hm.shape == (643, 643, 3)
    assert hm.arr_0to1.shape == (643, 643, 1)
    # TODO this value is 0.48026073 in python 2.7, while 0.48026952 in 3.7 -- why?
    assert np.allclose(np.average(hm.arr_0to1), 0.48026952, rtol=0, atol=1e-3)

    hm = iadata.quokka_heatmap(size=(642, 959))
    assert hm.shape == (642, 959, 3)
    assert hm.arr_0to1.shape == (642, 959, 1)
    assert np.allclose(np.average(hm.arr_0to1), 0.5762454, rtol=0, atol=1e-3)


def test_quokka_segmentation_map():
    segmap = iadata.quokka_segmentation_map()
    assert segmap.shape == (643, 960, 3)
    assert segmap.arr.shape == (643, 960, 1)
    assert np.allclose(np.average(segmap.arr), 0.3016427, rtol=0, atol=1e-3)

    segmap = iadata.quokka_segmentation_map(extract="square")
    assert segmap.shape == (643, 643, 3)
    assert segmap.arr.shape == (643, 643, 1)
    assert np.allclose(np.average(segmap.arr), 0.450353, rtol=0, atol=1e-3)

    segmap = iadata.quokka_segmentation_map(size=(642, 959))
    assert segmap.shape == (642, 959, 3)
    assert segmap.arr.shape == (642, 959, 1)
    assert np.allclose(np.average(segmap.arr), 0.30160266, rtol=0, atol=1e-3)


def test_quokka_keypoints():
    kpsoi = iadata.quokka_keypoints()
    assert len(kpsoi.keypoints) > 0
    assert np.allclose(kpsoi.keypoints[0].x, 163.0)
    assert np.allclose(kpsoi.keypoints[0].y, 78.0)
    assert kpsoi.shape == (643, 960, 3)

    img = iadata.quokka()
    patches = []
    for kp in kpsoi.keypoints:
        bb = ia.BoundingBox(x1=kp.x-1, x2=kp.x+2, y1=kp.y-1, y2=kp.y+2)
        patches.append(bb.extract_from_image(img))

    img_square = iadata.quokka(extract="square")
    kpsoi_square = iadata.quokka_keypoints(extract="square")
    assert len(kpsoi.keypoints) == len(kpsoi_square.keypoints)
    assert kpsoi_square.shape == (643, 643, 3)

    for kp, patch in zip(kpsoi_square.keypoints, patches):
        bb = ia.BoundingBox(x1=kp.x-1, x2=kp.x+2, y1=kp.y-1, y2=kp.y+2)
        patch_square = bb.extract_from_image(img_square)
        assert np.average(np.abs(patch.astype(np.float32) - patch_square.astype(np.float32))) < 1.0

    kpsoi_resized = iadata.quokka_keypoints(size=(642, 959))
    assert kpsoi_resized.shape == (642, 959, 3)
    assert len(kpsoi.keypoints) == len(kpsoi_resized.keypoints)
    for kp, kp_resized in zip(kpsoi.keypoints, kpsoi_resized.keypoints):
        d = np.sqrt((kp.x - kp_resized.x) ** 2 + (kp.y - kp_resized.y) ** 2)
        assert d < 1.0


def test_quokka_bounding_boxes():
    bbsoi = iadata.quokka_bounding_boxes()
    assert len(bbsoi.bounding_boxes) > 0
    bb0 = bbsoi.bounding_boxes[0]
    assert np.allclose(bb0.x1, 148.0)
    assert np.allclose(bb0.y1, 50.0)
    assert np.allclose(bb0.x2, 550.0)
    assert np.allclose(bb0.y2, 642.0)
    assert bbsoi.shape == (643, 960, 3)

    img = iadata.quokka()
    patches = []
    for bb in bbsoi.bounding_boxes:
        patches.append(bb.extract_from_image(img))

    img_square = iadata.quokka(extract="square")
    bbsoi_square = iadata.quokka_bounding_boxes(extract="square")
    assert len(bbsoi.bounding_boxes) == len(bbsoi_square.bounding_boxes)
    assert bbsoi_square.shape == (643, 643, 3)

    for bb, patch in zip(bbsoi_square.bounding_boxes, patches):
        patch_square = bb.extract_from_image(img_square)
        assert np.average(np.abs(patch.astype(np.float32) - patch_square.astype(np.float32))) < 1.0

    bbsoi_resized = iadata.quokka_bounding_boxes(size=(642, 959))
    assert bbsoi_resized.shape == (642, 959, 3)
    assert len(bbsoi.bounding_boxes) == len(bbsoi_resized.bounding_boxes)
    for bb, bb_resized in zip(bbsoi.bounding_boxes, bbsoi_resized.bounding_boxes):
        d = np.sqrt((bb.center_x - bb_resized.center_x) ** 2 + (bb.center_y - bb_resized.center_y) ** 2)
        assert d < 1.0
