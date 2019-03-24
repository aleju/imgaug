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
import six.moves as sm
import cv2

import imgaug as ia
from imgaug.imgaug import _quokka_normalize_extract, _compute_resized_shape
from imgaug import dtypes as iadt
from imgaug.testutils import reseed


def main():
    time_start = time.time()

    test_is_np_array()
    test_is_single_integer()
    test_is_single_float()
    test_is_single_number()
    test_is_iterable()
    test_is_string()
    test_is_single_bool()
    test_is_integer_array()
    test_is_float_array()
    test_is_callable()
    test_caller_name()
    test_seed()
    test_current_random_state()
    test_new_random_state()
    test_dummy_random_state()
    test_copy_random_state()
    test_derive_random_state()
    test_derive_random_states()
    test_forward_random_state()
    test__quokka_normalize_extract()
    test__compute_resized_shape()
    test_quokka()
    test_quokka_square()
    test_quokka_heatmap()
    test_quokka_segmentation_map()
    test_quokka_keypoints()
    test_quokka_bounding_boxes()
    # test_angle_between_vectors()
    test_compute_line_intersection_point()
    test_draw_text()
    test_imresize_many_images()
    test_imresize_single_image()
    test_pad()
    test_compute_paddings_for_aspect_ratio()
    test_pad_to_aspect_ratio()
    test_pool()
    test_avg_pool()
    test_max_pool()
    test_draw_grid()
    # test_show_grid()
    # test_do_assert()
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
    # test_Batch()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_is_np_array():
    class _Dummy(object):
        pass
    values_true = [
        np.zeros((1, 2), dtype=np.uint8),
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((100,), dtype=np.float64)
    ]
    values_false = [
        "A", "BC", "1", True, False, (1.0, 2.0), [1.0, 2.0], _Dummy(),
        -100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2, 1e-4
    ]
    for value in values_true:
        assert ia.is_np_array(value) is True
    for value in values_false:
        assert ia.is_np_array(value) is False


def test_is_single_integer():
    assert ia.is_single_integer("A") is False
    assert ia.is_single_integer(None) is False
    assert ia.is_single_integer(1.2) is False
    assert ia.is_single_integer(1.0) is False
    assert ia.is_single_integer(np.ones((1,), dtype=np.float32)[0]) is False
    assert ia.is_single_integer(1) is True
    assert ia.is_single_integer(1234) is True
    assert ia.is_single_integer(np.ones((1,), dtype=np.uint8)[0]) is True
    assert ia.is_single_integer(np.ones((1,), dtype=np.int32)[0]) is True


def test_is_single_float():
    assert ia.is_single_float("A") is False
    assert ia.is_single_float(None) is False
    assert ia.is_single_float(1.2) is True
    assert ia.is_single_float(1.0) is True
    assert ia.is_single_float(np.ones((1,), dtype=np.float32)[0]) is True
    assert ia.is_single_float(1) is False
    assert ia.is_single_float(1234) is False
    assert ia.is_single_float(np.ones((1,), dtype=np.uint8)[0]) is False
    assert ia.is_single_float(np.ones((1,), dtype=np.int32)[0]) is False


def test_caller_name():
    assert ia.caller_name() == 'test_caller_name'


def test_is_single_number():
    class _Dummy(object):
        pass
    values_true = [-100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2, 1e-4]
    values_false = ["A", "BC", "1", True, False, (1.0, 2.0), [1.0, 2.0], _Dummy(), np.zeros((1, 2), dtype=np.uint8)]
    for value in values_true:
        assert ia.is_single_number(value) is True
    for value in values_false:
        assert ia.is_single_number(value) is False


def test_is_iterable():
    class _Dummy(object):
        pass
    values_true = [
        [0, 1, 2],
        ["A", "X"],
        [[123], [456, 789]],
        [],
        (1, 2, 3),
        (1,),
        tuple(),
        "A",
        "ABC",
        "",
        np.zeros((100,), dtype=np.uint8)
    ]
    values_false = [1, 100, 0, -100, -1, 1.2, -1.2, True, False, _Dummy()]
    for value in values_true:
        assert ia.is_iterable(value) is True, value
    for value in values_false:
        assert ia.is_iterable(value) is False


def test_is_string():
    class _Dummy(object):
        pass
    values_true = ["A", "BC", "1", ""]
    values_false = [-100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2, 1e-4, True, False, (1.0, 2.0), [1.0, 2.0],
                    _Dummy(), np.zeros((1, 2), dtype=np.uint8)]
    for value in values_true:
        assert ia.is_string(value) is True
    for value in values_false:
        assert ia.is_string(value) is False


def test_is_single_bool():
    class _Dummy(object):
        pass
    values_true = [False, True]
    values_false = [-100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2, 1e-4, (1.0, 2.0), [1.0, 2.0], _Dummy(),
                    np.zeros((1, 2), dtype=np.uint8), np.zeros((1,), dtype=bool)]
    for value in values_true:
        assert ia.is_single_bool(value) is True
    for value in values_false:
        assert ia.is_single_bool(value) is False


def test_is_integer_array():
    class _Dummy(object):
        pass
    values_true = [
        np.zeros((1, 2), dtype=np.uint8),
        np.zeros((100,), dtype=np.uint8),
        np.zeros((1, 2), dtype=np.uint16),
        np.zeros((1, 2), dtype=np.int32),
        np.zeros((1, 2), dtype=np.int64)
    ]
    values_false = [
        "A", "BC", "1", "", -100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2, 1e-4, True, False,
        (1.0, 2.0), [1.0, 2.0], _Dummy(),
        np.zeros((1, 2), dtype=np.float16),
        np.zeros((100,), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float64),
        np.zeros((1, 2), dtype=np.bool)
    ]
    for value in values_true:
        assert ia.is_integer_array(value) is True
    for value in values_false:
        assert ia.is_integer_array(value) is False


def test_is_float_array():
    class _Dummy(object):
        pass

    values_true = [
        np.zeros((1, 2), dtype=np.float16),
        np.zeros((100,), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float64)
    ]
    values_false = [
        "A", "BC", "1", "", -100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2, 1e-4, True, False,
        (1.0, 2.0), [1.0, 2.0], _Dummy(),
        np.zeros((1, 2), dtype=np.uint8),
        np.zeros((100,), dtype=np.uint8),
        np.zeros((1, 2), dtype=np.uint16),
        np.zeros((1, 2), dtype=np.int32),
        np.zeros((1, 2), dtype=np.int64),
        np.zeros((1, 2), dtype=np.bool)
    ]
    for value in values_true:
        assert ia.is_float_array(value) is True
    for value in values_false:
        assert ia.is_float_array(value) is False


def test_is_callable():
    def _dummy_func():
        pass

    _dummy_func2 = lambda x: x

    class _Dummy1(object):
        pass

    class _Dummy2(object):
        def __call__(self):
            pass

    values_true = [_dummy_func, _dummy_func2, _Dummy2()]
    values_false = ["A", "BC", "1", "", -100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2, 1e-4, True, False,
                    (1.0, 2.0), [1.0, 2.0], _Dummy1(), np.zeros((1, 2), dtype=np.uint8)]
    for value in values_true:
        assert ia.is_callable(value) == True
    for value in values_false:
        assert ia.is_callable(value) == False


def test_seed():
    ia.seed(10017)
    rs = np.random.RandomState(10017)
    assert ia.CURRENT_RANDOM_STATE.randint(0, 1000*1000) == rs.randint(0, 1000*1000)
    reseed()


def test_current_random_state():
    assert ia.current_random_state() == ia.CURRENT_RANDOM_STATE


def test_new_random_state():
    seed = 1000
    ia.seed(seed)

    rs_observed = ia.new_random_state(seed=None, fully_random=False)
    rs_expected = np.random.RandomState(
        np.random.RandomState(seed).randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE, 1)[0]
    )
    assert rs_observed.randint(0, 10**6) == rs_expected.randint(0, 10**6)
    rs_observed1 = ia.new_random_state(seed=None, fully_random=False)
    rs_observed2 = ia.new_random_state(seed=None, fully_random=False)
    assert rs_observed1.randint(0, 10**6) != rs_observed2.randint(0, 10**6)

    ia.seed(seed)
    np.random.seed(seed)
    rs_observed = ia.new_random_state(seed=None, fully_random=True)
    rs_not_expected = np.random.RandomState(
        np.random.RandomState(seed).randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE, 1)[0]
    )
    assert rs_observed.randint(0, 10**6) != rs_not_expected.randint(0, 10**6)

    rs_observed1 = ia.new_random_state(seed=None, fully_random=True)
    rs_observed2 = ia.new_random_state(seed=None, fully_random=True)
    assert rs_observed1.randint(0, 10**6) != rs_observed2.randint(0, 10**6)

    rs_observed1 = ia.new_random_state(seed=1234)
    rs_observed2 = ia.new_random_state(seed=1234)
    rs_expected = np.random.RandomState(1234)
    assert rs_observed1.randint(0, 10**6) == rs_observed2.randint(0, 10**6) == rs_expected.randint(0, 10**6)


def test_dummy_random_state():
    assert ia.dummy_random_state().randint(0, 10**6) == np.random.RandomState(1).randint(0, 10**6)


def test_copy_random_state():
    rs = np.random.RandomState(1017)
    rs_copy = ia.copy_random_state(rs)
    assert rs != rs_copy
    assert rs.randint(0, 10**6) == rs_copy.randint(0, 10**6)

    assert ia.copy_random_state(np.random) == np.random
    assert ia.copy_random_state(np.random, force_copy=True) != np.random


def test_derive_random_state():
    rs_observed = ia.derive_random_state(np.random.RandomState(1017))
    rs_expected = np.random.RandomState(np.random.RandomState(1017).randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE))
    assert rs_observed.randint(0, 10**6) == rs_expected.randint(0, 10**6)


def test_derive_random_states():
    rs_observed1, rs_observed2 = ia.derive_random_states(np.random.RandomState(1017), n=2)
    seed = np.random.RandomState(1017).randint(ia.SEED_MIN_VALUE, ia.SEED_MAX_VALUE)
    rs_expected1 = np.random.RandomState(seed+0)
    rs_expected2 = np.random.RandomState(seed+1)
    assert rs_observed1.randint(0, 10**6) == rs_expected1.randint(0, 10**6)
    assert rs_observed2.randint(0, 10**6) == rs_expected2.randint(0, 10**6)


def test_forward_random_state():
    rs1 = np.random.RandomState(1017)
    rs2 = np.random.RandomState(1017)
    ia.forward_random_state(rs1)
    rs2.uniform()
    assert rs1.randint(0, 10**6) == rs2.randint(0, 10**6)


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
    img = ia.quokka()
    assert img.shape == (643, 960, 3)
    assert np.allclose(
        np.average(img, axis=(0, 1)),
        [107.93576659, 118.18765066, 122.99378564]
    )

    img = ia.quokka(extract="square")
    assert img.shape == (643, 643, 3)
    assert np.allclose(
        np.average(img, axis=(0, 1)),
        [111.25929196, 121.19431175, 125.71316898]
    )

    img = ia.quokka(size=(642, 959))
    assert img.shape == (642, 959, 3)
    assert np.allclose(
        np.average(img, axis=(0, 1)),
        [107.84615822, 118.09832412, 122.90446467]
    )


def test_quokka_square():
    img = ia.quokka_square()
    assert img.shape == (643, 643, 3)
    assert np.allclose(
        np.average(img, axis=(0, 1)),
        [111.25929196, 121.19431175, 125.71316898]
    )


def test_quokka_heatmap():
    hm = ia.quokka_heatmap()
    assert hm.shape == (643, 960, 3)
    assert hm.arr_0to1.shape == (643, 960, 1)
    assert np.allclose(np.average(hm.arr_0to1), 0.57618505)

    hm = ia.quokka_heatmap(extract="square")
    assert hm.shape == (643, 643, 3)
    assert hm.arr_0to1.shape == (643, 643, 1)
    # TODO this value is 0.48026073 in python 2.7, while 0.48026952 in 3.7 -- why?
    assert np.allclose(np.average(hm.arr_0to1), 0.48026952, atol=1e-4)

    hm = ia.quokka_heatmap(size=(642, 959))
    assert hm.shape == (642, 959, 3)
    assert hm.arr_0to1.shape == (642, 959, 1)
    assert np.allclose(np.average(hm.arr_0to1), 0.5762454)


def test_quokka_segmentation_map():
    segmap = ia.quokka_segmentation_map()
    assert segmap.shape == (643, 960, 3)
    assert segmap.arr.shape == (643, 960, 1)
    assert np.allclose(np.average(segmap.arr), 0.3016427)

    segmap = ia.quokka_segmentation_map(extract="square")
    assert segmap.shape == (643, 643, 3)
    assert segmap.arr.shape == (643, 643, 1)
    assert np.allclose(np.average(segmap.arr), 0.450353)

    segmap = ia.quokka_segmentation_map(size=(642, 959))
    assert segmap.shape == (642, 959, 3)
    assert segmap.arr.shape == (642, 959, 1)
    assert np.allclose(np.average(segmap.arr), 0.30160266)


def test_quokka_keypoints():
    kpsoi = ia.quokka_keypoints()
    assert len(kpsoi.keypoints) > 0
    assert np.allclose(kpsoi.keypoints[0].x, 163.0)
    assert np.allclose(kpsoi.keypoints[0].y, 78.0)
    assert kpsoi.shape == (643, 960, 3)

    img = ia.quokka()
    patches = []
    for kp in kpsoi.keypoints:
        bb = ia.BoundingBox(x1=kp.x-1, x2=kp.x+2, y1=kp.y-1, y2=kp.y+2)
        patches.append(bb.extract_from_image(img))

    img_square = ia.quokka(extract="square")
    kpsoi_square = ia.quokka_keypoints(extract="square")
    assert len(kpsoi.keypoints) == len(kpsoi_square.keypoints)
    assert kpsoi_square.shape == (643, 643, 3)

    for kp, patch in zip(kpsoi_square.keypoints, patches):
        bb = ia.BoundingBox(x1=kp.x-1, x2=kp.x+2, y1=kp.y-1, y2=kp.y+2)
        patch_square = bb.extract_from_image(img_square)
        assert np.average(np.abs(patch.astype(np.float32) - patch_square.astype(np.float32))) < 1.0

    kpsoi_resized = ia.quokka_keypoints(size=(642, 959))
    assert kpsoi_resized.shape == (642, 959, 3)
    assert len(kpsoi.keypoints) == len(kpsoi_resized.keypoints)
    for kp, kp_resized in zip(kpsoi.keypoints, kpsoi_resized.keypoints):
        d = np.sqrt((kp.x - kp_resized.x) ** 2 + (kp.y - kp_resized.y) ** 2)
        assert d < 1.0


def test_quokka_bounding_boxes():
    bbsoi = ia.quokka_bounding_boxes()
    assert len(bbsoi.bounding_boxes) > 0
    bb0 = bbsoi.bounding_boxes[0]
    assert np.allclose(bb0.x1, 148.0)
    assert np.allclose(bb0.y1, 50.0)
    assert np.allclose(bb0.x2, 550.0)
    assert np.allclose(bb0.y2, 642.0)
    assert bbsoi.shape == (643, 960, 3)

    img = ia.quokka()
    patches = []
    for bb in bbsoi.bounding_boxes:
        patches.append(bb.extract_from_image(img))

    img_square = ia.quokka(extract="square")
    bbsoi_square = ia.quokka_bounding_boxes(extract="square")
    assert len(bbsoi.bounding_boxes) == len(bbsoi_square.bounding_boxes)
    assert bbsoi_square.shape == (643, 643, 3)

    for bb, patch in zip(bbsoi_square.bounding_boxes, patches):
        patch_square = bb.extract_from_image(img_square)
        assert np.average(np.abs(patch.astype(np.float32) - patch_square.astype(np.float32))) < 1.0

    bbsoi_resized = ia.quokka_bounding_boxes(size=(642, 959))
    assert bbsoi_resized.shape == (642, 959, 3)
    assert len(bbsoi.bounding_boxes) == len(bbsoi_resized.bounding_boxes)
    for bb, bb_resized in zip(bbsoi.bounding_boxes, bbsoi_resized.bounding_boxes):
        d = np.sqrt((bb.center_x - bb_resized.center_x) ** 2 + (bb.center_y - bb_resized.center_y) ** 2)
        assert d < 1.0


def test_compute_line_intersection_point():
    # intersecting lines
    line1 = (0, 0, 1, 0)
    line2 = (0.5, -1, 0.5, 1)
    point = ia.compute_line_intersection_point(
        line1[0], line1[1], line1[2], line1[3],
        line2[0], line2[1], line2[2], line2[3]
    )
    assert np.allclose(point[0], 0.5)
    assert np.allclose(point[1], 0)

    # intersection point outside of defined interval of one line, should not change anything
    line1 = (0, 0, 1, 0)
    line2 = (0.5, -1, 0.5, -0.5)
    point = ia.compute_line_intersection_point(
        line1[0], line1[1], line1[2], line1[3],
        line2[0], line2[1], line2[2], line2[3]
    )
    assert np.allclose(point[0], 0.5)
    assert np.allclose(point[1], 0)

    # touching lines
    line1 = (0, 0, 1, 0)
    line2 = (0.5, -1, 0.5, 0)
    point = ia.compute_line_intersection_point(
        line1[0], line1[1], line1[2], line1[3],
        line2[0], line2[1], line2[2], line2[3]
    )
    assert np.allclose(point[0], 0.5)
    assert np.allclose(point[1], 0)

    # parallel, not intersecting lines
    line1 = (0, 0, 1, 0)
    line2 = (0, -0.1, 1, -0.1)
    point = ia.compute_line_intersection_point(
        line1[0], line1[1], line1[2], line1[3],
        line2[0], line2[1], line2[2], line2[3]
    )
    assert point is False

    # parallel and overlapping lines (infinite intersection points)
    line1 = (0, 0, 1, 0)
    line2 = (0.1, 0, 1, 0)
    point = ia.compute_line_intersection_point(
        line1[0], line1[1], line1[2], line1[3],
        line2[0], line2[1], line2[2], line2[3]
    )
    assert point is False


def test_draw_text():
    # make roughly sure that shape of drawn text matches expected text
    img = np.zeros((20, 50, 3), dtype=np.uint8)
    img_text = ia.draw_text(img, y=5, x=5, text="---------", size=10, color=[255, 255, 255])
    assert np.max(img_text) == 255
    assert np.min(img_text) == 0
    assert np.sum(img_text == 255) / np.sum(img_text == 0)
    first_row = None
    last_row = None
    first_col = None
    last_col = None
    for i in range(img.shape[0]):
        if np.max(img_text[i, :, :]) == 255:
            first_row = i
            break
    for i in range(img.shape[0]-1, 0, -1):
        if np.max(img_text[i, :, :]) == 255:
            last_row = i
            break
    for i in range(img.shape[1]):
        if np.max(img_text[:, i, :]) == 255:
            first_col = i
            break
    for i in range(img.shape[1]-1, 0, -1):
        if np.max(img_text[:, i, :]) == 255:
            last_col = i
            break
    bb = ia.BoundingBox(x1=first_col, y1=first_row, x2=last_col, y2=last_row)
    assert bb.width > 4.0*bb.height

    # test x
    img = np.zeros((20, 100, 3), dtype=np.uint8)
    img_text1 = ia.draw_text(img, y=5, x=5, text="XXXXXXX", size=10, color=[255, 255, 255])
    img_text2 = ia.draw_text(img, y=5, x=50, text="XXXXXXX", size=10, color=[255, 255, 255])
    first_col1 = None
    first_col2 = None
    for i in range(img.shape[1]):
        if np.max(img_text1[:, i, :]) == 255:
            first_col1 = i
            break
    for i in range(img.shape[1]):
        if np.max(img_text2[:, i, :]) == 255:
            first_col2 = i
            break
    assert 0 < first_col1 < 10
    assert 45 < first_col2 < 55

    # test y
    img = np.zeros((100, 20, 3), dtype=np.uint8)
    img_text1 = ia.draw_text(img, y=5, x=5, text="XXXXXXX", size=10, color=[255, 255, 255])
    img_text2 = ia.draw_text(img, y=50, x=5, text="XXXXXXX", size=10, color=[255, 255, 255])
    first_row1 = None
    first_row2 = None
    for i in range(img.shape[0]):
        if np.max(img_text1[i, :, :]) == 255:
            first_row1 = i
            break
    for i in range(img.shape[0]):
        if np.max(img_text2[i, :, :]) == 255:
            first_row2 = i
            break
    assert 0 < first_row1 < 15
    assert 45 < first_row2 < 60

    # test size
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_text_small = ia.draw_text(img, y=5, x=5, text="X", size=10, color=[255, 255, 255])
    img_text_large = ia.draw_text(img, y=5, x=5, text="X", size=50, color=[255, 255, 255])
    nb_filled_small = np.sum(img_text_small > 10)
    nb_filled_large = np.sum(img_text_large > 10)
    assert nb_filled_large > 2*nb_filled_small

    # text color
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img_text = ia.draw_text(img, y=5, x=5, text="X", size=10, color=[128, 129, 130])
    maxcol = np.max(img_text, axis=(0, 1))
    assert maxcol[0] == 128
    assert maxcol[1] == 129
    assert maxcol[2] == 130


def test_imresize_many_images():
    interpolations = [None,
                      "nearest", "linear", "area", "cubic",
                      cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]

    for c in [1, 3]:
        image1 = np.zeros((16, 16, c), dtype=np.uint8) + 255
        image2 = np.zeros((16, 16, c), dtype=np.uint8)
        image3 = np.pad(
            np.zeros((8, 8, c), dtype=np.uint8) + 255,
            ((4, 4), (4, 4), (0, 0)),
            mode="constant",
            constant_values=0
        )

        image1_small = np.zeros((8, 8, c), dtype=np.uint8) + 255
        image2_small = np.zeros((8, 8, c), dtype=np.uint8)
        image3_small = np.pad(
            np.zeros((4, 4, c), dtype=np.uint8) + 255,
            ((2, 2), (2, 2), (0, 0)),
            mode="constant",
            constant_values=0
        )

        image1_large = np.zeros((32, 32, c), dtype=np.uint8) + 255
        image2_large = np.zeros((32, 32, c), dtype=np.uint8)
        image3_large = np.pad(
            np.zeros((16, 16, c), dtype=np.uint8) + 255,
            ((8, 8), (8, 8), (0, 0)),
            mode="constant",
            constant_values=0
        )

        images = np.uint8([image1, image2, image3])
        images_small = np.uint8([image1_small, image2_small, image3_small])
        images_large = np.uint8([image1_large, image2_large, image3_large])

        for images_this_iter in [images, list(images)]:  # test for ndarray and list(ndarray) input
            for interpolation in interpolations:
                images_same_observed = ia.imresize_many_images(images_this_iter, (16, 16), interpolation=interpolation)
                for image_expected, image_observed in zip(images_this_iter, images_same_observed):
                    diff = np.abs(image_expected.astype(np.int32) - image_observed.astype(np.int32))
                    assert np.sum(diff) == 0

            for interpolation in interpolations:
                images_small_observed = ia.imresize_many_images(images_this_iter, (8, 8), interpolation=interpolation)
                for image_expected, image_observed in zip(images_small, images_small_observed):
                    diff = np.abs(image_expected.astype(np.int32) - image_observed.astype(np.int32))
                    diff_fraction = np.sum(diff) / (image_observed.size * 255)
                    assert diff_fraction < 0.5

            for interpolation in interpolations:
                images_large_observed = ia.imresize_many_images(images_this_iter, (32, 32), interpolation=interpolation)
                for image_expected, image_observed in zip(images_large, images_large_observed):
                    diff = np.abs(image_expected.astype(np.int32) - image_observed.astype(np.int32))
                    diff_fraction = np.sum(diff) / (image_observed.size * 255)
                    assert diff_fraction < 0.5

    # test size given as single int
    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    observed = ia.imresize_many_images(images, 8)
    assert observed.shape == (1, 8, 8, 3)

    # test size given as single float
    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    observed = ia.imresize_many_images(images, 2.0)
    assert observed.shape == (1, 8, 8, 3)

    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    observed = ia.imresize_many_images(images, 0.5)
    assert observed.shape == (1, 2, 2, 3)

    # test size given as (float, float)
    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    observed = ia.imresize_many_images(images, (2.0, 2.0))
    assert observed.shape == (1, 8, 8, 3)

    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    observed = ia.imresize_many_images(images, (0.5, 0.5))
    assert observed.shape == (1, 2, 2, 3)

    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    observed = ia.imresize_many_images(images, (2.0, 0.5))
    assert observed.shape == (1, 8, 2, 3)

    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    observed = ia.imresize_many_images(images, (0.5, 2.0))
    assert observed.shape == (1, 2, 8, 3)

    # test size given as int+float or float+int
    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    observed = ia.imresize_many_images(images, (11, 2.0))
    assert observed.shape == (1, 11, 8, 3)

    images = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    observed = ia.imresize_many_images(images, (2.0, 11))
    assert observed.shape == (1, 8, 11, 3)

    # test no channels
    images = np.zeros((1, 4, 4), dtype=np.uint8)
    images_rs = ia.imresize_many_images(images, (2, 2))
    assert images_rs.shape == (1, 2, 2)

    images = [np.zeros((4, 4), dtype=np.uint8)]
    images_rs = ia.imresize_many_images(images, (2, 2))
    assert isinstance(images_rs, list)
    assert images_rs[0].shape == (2, 2)

    # test len 0 input
    observed = ia.imresize_many_images(np.zeros((0, 8, 8, 3), dtype=np.uint8), (4, 4))
    assert ia.is_np_array(observed)
    assert observed.dtype.type == np.uint8
    assert len(observed) == 0

    observed = ia.imresize_many_images([], (4, 4))
    assert isinstance(observed, list)
    assert len(observed) == 0

    # test images with zero height/width
    images = [np.zeros((0, 4, 3), dtype=np.uint8)]
    got_exception = False
    try:
        _ = ia.imresize_many_images(images, sizes=(2, 2))
    except Exception as exc:
        assert "Cannot resize images, because at least one image has a height and/or width of zero." in str(exc)
        got_exception = True
    assert got_exception

    images = [np.zeros((4, 0, 3), dtype=np.uint8)]
    got_exception = False
    try:
        _ = ia.imresize_many_images(images, sizes=(2, 2))
    except Exception as exc:
        assert "Cannot resize images, because at least one image has a height and/or width of zero." in str(exc)
        got_exception = True
    assert got_exception

    images = [np.zeros((0, 0, 3), dtype=np.uint8)]
    got_exception = False
    try:
        _ = ia.imresize_many_images(images, sizes=(2, 2))
    except Exception as exc:
        assert "Cannot resize images, because at least one image has a height and/or width of zero." in str(exc)
        got_exception = True
    assert got_exception

    # test invalid sizes
    sizes_all = [(-1, 2), (0, 2)]
    sizes_all = sizes_all\
        + [(float(a), b) for a, b in sizes_all]\
        + [(a, float(b)) for a, b in sizes_all]\
        + [(float(a), float(b)) for a, b in sizes_all]\
        + [(-a, -b) for a, b in sizes_all]\
        + [(-float(a), -b) for a, b in sizes_all]\
        + [(-a, -float(b)) for a, b in sizes_all]\
        + [(-float(a), -float(b)) for a, b in sizes_all]
    sizes_all = sizes_all\
        + [(b, a) for a, b in sizes_all]
    sizes_all = sizes_all\
        + [-1.0, 0.0, -1, 0]
    for sizes in sizes_all:
        images = [np.zeros((4, 4, 3), dtype=np.uint8)]
        got_exception = False
        try:
            _ = ia.imresize_many_images(images, sizes=sizes)
        except Exception as exc:
            assert "value is zero or lower than zero." in str(exc)
            got_exception = True
        assert got_exception

    # test list input but all with same shape
    images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    observed = ia.imresize_many_images(images, (4, 4))
    assert isinstance(observed, list)
    assert all([image.shape == (4, 4, 3) for image in observed])
    assert all([image.dtype.type == np.uint8 for image in observed])

    # test multiple shapes
    images = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)]
    observed = ia.imresize_many_images(images, (4, 4))
    assert observed[0].shape == (4, 4, 3)
    assert observed[1].shape == (4, 4)
    assert observed[0].dtype == np.uint8
    assert observed[1].dtype == np.uint8

    ###################
    # test other dtypes
    ###################
    # interpolation="nearest"
    image = np.zeros((4, 4), dtype=bool)
    image[1, :] = True
    image[2, :] = True
    expected = np.zeros((3, 3), dtype=bool)
    expected[1, :] = True
    expected[2, :] = True
    image_rs = ia.imresize_many_images([image], (3, 3), interpolation="nearest")[0]
    assert image_rs.dtype.type == image.dtype.type
    assert np.all(image_rs == expected)

    for dtype in [np.uint8, np.uint16, np.int8, np.int16, np.int32]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
        for value in [min_value, max_value]:
            image = np.zeros((4, 4), dtype=dtype)
            image[1, :] = value
            image[2, :] = value
            expected = np.zeros((3, 3), dtype=dtype)
            expected[1, :] = value
            expected[2, :] = value
            image_rs = ia.imresize_many_images([image], (3, 3), interpolation="nearest")[0]
            assert image_rs.dtype.type == dtype
            assert np.all(image_rs == expected)

    for dtype in [np.float16, np.float32, np.float64]:
        isize = np.dtype(dtype).itemsize
        for value in [0.5, -0.5, 1.0, -1.0, 10.0, -10.0, -1000 ** (isize-1), 1000 * (isize+1)]:
            image = np.zeros((4, 4), dtype=dtype)
            image[1, :] = value
            image[2, :] = value
            expected = np.zeros((3, 3), dtype=dtype)
            expected[1, :] = value
            expected[2, :] = value
            image_rs = ia.imresize_many_images([image], (3, 3), interpolation="nearest")[0]
            assert image_rs.dtype.type == dtype
            assert np.allclose(image_rs, expected, rtol=0, atol=1e-8)

    # other interpolations
    for ip in ["linear", "cubic", "area"]:
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, :] = 255
        mask[2, :] = 255
        mask = ia.imresize_many_images([mask], (3, 3), interpolation=ip)[0]
        mask = mask.astype(np.float128) / 255.0

        image = np.zeros((4, 4), dtype=bool)
        image[1, :] = True
        image[2, :] = True
        expected = mask > 0.5
        image_rs = ia.imresize_many_images([image], (3, 3), interpolation=ip)[0]
        assert image_rs.dtype.type == image.dtype.type
        assert np.all(image_rs == expected)

        for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
            dynamic_range = max_value - min_value
            for value in [min_value+1, max_value-1]:
                image = np.zeros((4, 4), dtype=dtype)
                image[1, :] = value
                image[2, :] = value
                expected = np.round(mask * value).astype(dtype)
                image_rs = ia.imresize_many_images([image], (3, 3), interpolation=ip)[0]
                assert image_rs.dtype.type == dtype
                diff = np.abs(image_rs.astype(np.int64) - expected.astype(np.int64))
                assert np.all(diff < 2 * (1/255) * dynamic_range)

        mask = np.zeros((4, 4), dtype=np.float64)
        mask[1, :] = 1.0
        mask[2, :] = 1.0
        mask = ia.imresize_many_images([mask], (3, 3), interpolation=ip)[0]
        mask = mask.astype(np.float128)

        for dtype in [np.float16, np.float32, np.float64]:
            isize = np.dtype(dtype).itemsize

            for value in [0.5, -0.5, 1.0, -1.0, 10.0, -10.0, -1000 ** (isize-1), 1000 * (isize+1)]:
                image = np.zeros((4, 4), dtype=dtype)
                image[1, :] = value
                image[2, :] = value
                expected = (mask * np.float128(value)).astype(dtype)
                image_rs = ia.imresize_many_images([image], (3, 3), interpolation=ip)[0]
                assert image_rs.dtype.type == dtype
                # Our basis for the expected image is derived from uint8 as that is most likely to work, so we will
                # have to accept here deviations of around 1/255.
                atol = np.float128(1 / 255) * np.abs(np.float128(value)) + 1e-8
                assert np.allclose(image_rs, expected, rtol=0, atol=atol)
                # Expect at least one cell to have a difference between observed and expected image of approx. 0,
                # currently we seem to be able to get away with this despite the above mentioned inaccuracy.
                assert np.any(np.isclose(image_rs, expected, rtol=0, atol=1e-4))


def test_imresize_single_image():
    for c in [-1, 1, 3]:
        image1 = np.zeros((16, 16, abs(c)), dtype=np.uint8) + 255
        image2 = np.zeros((16, 16, abs(c)), dtype=np.uint8)
        image3 = np.pad(
            np.zeros((8, 8, abs(c)), dtype=np.uint8) + 255,
            ((4, 4), (4, 4), (0, 0)),
            mode="constant",
            constant_values=0
        )

        image1_small = np.zeros((8, 8, abs(c)), dtype=np.uint8) + 255
        image2_small = np.zeros((8, 8, abs(c)), dtype=np.uint8)
        image3_small = np.pad(
            np.zeros((4, 4, abs(c)), dtype=np.uint8) + 255,
            ((2, 2), (2, 2), (0, 0)),
            mode="constant",
            constant_values=0
        )

        image1_large = np.zeros((32, 32, abs(c)), dtype=np.uint8) + 255
        image2_large = np.zeros((32, 32, abs(c)), dtype=np.uint8)
        image3_large = np.pad(
            np.zeros((16, 16, abs(c)), dtype=np.uint8) + 255,
            ((8, 8), (8, 8), (0, 0)),
            mode="constant",
            constant_values=0
        )

        images = np.uint8([image1, image2, image3])
        images_small = np.uint8([image1_small, image2_small, image3_small])
        images_large = np.uint8([image1_large, image2_large, image3_large])

        if c == -1:
            images = images[:, :, 0]
            images_small = images_small[:, :, 0]
            images_large = images_large[:, :, 0]

        interpolations = [None,
                          "nearest", "linear", "area", "cubic",
                          cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]

        for interpolation in interpolations:
            for image in images:
                image_observed = ia.imresize_single_image(image, (16, 16), interpolation=interpolation)
                diff = np.abs(image.astype(np.int32) - image_observed.astype(np.int32))
                assert np.sum(diff) == 0

        for interpolation in interpolations:
            for image, image_expected in zip(images, images_small):
                image_observed = ia.imresize_single_image(image, (8, 8), interpolation=interpolation)
                diff = np.abs(image_expected.astype(np.int32) - image_observed.astype(np.int32))
                diff_fraction = np.sum(diff) / (image_observed.size * 255)
                assert diff_fraction < 0.5

        for interpolation in interpolations:
            for image, image_expected in zip(images, images_large):
                image_observed = ia.imresize_single_image(image, (32, 32), interpolation=interpolation)
                diff = np.abs(image_expected.astype(np.int32) - image_observed.astype(np.int32))
                diff_fraction = np.sum(diff) / (image_observed.size * 255)
                assert diff_fraction < 0.5


def test_pad():
    # -------
    # uint, int
    # -------
    for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        arr = np.zeros((3, 3), dtype=dtype) + max_value

        arr_pad = ia.pad(arr)
        assert arr_pad.shape == (3, 3)
        # For some reason, arr_pad.dtype.type == dtype fails here for int64 but not for the other dtypes,
        # even though int64 is the dtype of arr_pad. Also checked .name and .str for them -- all same value.
        assert arr_pad.dtype == np.dtype(dtype)
        assert np.array_equal(arr_pad, arr)

        arr_pad = ia.pad(arr, top=1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert np.all(arr_pad[0, :] == 0)

        arr_pad = ia.pad(arr, right=1)
        assert arr_pad.shape == (3, 4)
        assert arr_pad.dtype == np.dtype(dtype)
        assert np.all(arr_pad[:, -1] == 0)

        arr_pad = ia.pad(arr, bottom=1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert np.all(arr_pad[-1, :] == 0)

        arr_pad = ia.pad(arr, left=1)
        assert arr_pad.shape == (3, 4)
        assert arr_pad.dtype == np.dtype(dtype)
        assert np.all(arr_pad[:, 0] == 0)

        arr_pad = ia.pad(arr, top=1, right=2, bottom=3, left=4)
        assert arr_pad.shape == (3+(1+3), 3+(2+4))
        assert arr_pad.dtype == np.dtype(dtype)
        assert np.all(arr_pad[0, :] == 0)
        assert np.all(arr_pad[:, -2:] == 0)
        assert np.all(arr_pad[-3:, :] == 0)
        assert np.all(arr_pad[:, :4] == 0)

        arr_pad = ia.pad(arr, top=1, cval=10)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert np.all(arr_pad[0, :] == 10)

        arr = np.zeros((3, 3, 3), dtype=dtype) + 127
        arr_pad = ia.pad(arr, top=1)
        assert arr_pad.shape == (4, 3, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert np.all(arr_pad[0, :, 0] == 0)
        assert np.all(arr_pad[0, :, 1] == 0)
        assert np.all(arr_pad[0, :, 2] == 0)

        v1 = int(center_value + 0.25 * max_value)
        v2 = int(center_value + 0.40 * max_value)
        arr = np.zeros((3, 3), dtype=dtype) + v1
        arr[1, 1] = v2
        arr_pad = ia.pad(arr, top=1, mode="maximum")
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert arr_pad[0, 0] == v1
        assert arr_pad[0, 1] == v2
        assert arr_pad[0, 2] == v1

        v1 = int(center_value + 0.25 * max_value)
        arr = np.zeros((3, 3), dtype=dtype)
        arr_pad = ia.pad(arr, top=1, mode="constant", cval=v1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert arr_pad[0, 0] == v1
        assert arr_pad[0, 1] == v1
        assert arr_pad[0, 2] == v1
        assert arr_pad[1, 0] == 0

        for nb_channels in [1, 2, 3, 4, 5]:
            v1 = int(center_value + 0.25 * max_value)
            arr = np.zeros((3, 3, nb_channels), dtype=dtype)
            arr_pad = ia.pad(arr, top=1, mode="constant", cval=v1)
            assert arr_pad.shape == (4, 3, nb_channels)
            assert arr_pad.dtype == np.dtype(dtype)
            assert np.all(arr_pad[0, 0, :] == v1)
            assert np.all(arr_pad[0, 1, :] == v1)
            assert np.all(arr_pad[0, 2, :] == v1)
            assert np.all(arr_pad[1, 0, :] == 0)

        arr = np.zeros((1, 1), dtype=dtype) + 0
        arr_pad = ia.pad(arr, top=4, mode="linear_ramp", cval=100)
        assert arr_pad.shape == (5, 1)
        assert arr_pad.dtype == np.dtype(dtype)
        assert arr_pad[0, 0] == 100
        assert arr_pad[1, 0] == 75
        assert arr_pad[2, 0] == 50
        assert arr_pad[3, 0] == 25
        assert arr_pad[4, 0] == 0

        # test other channel numbers
        value = int(center_value + 0.25 * max_value)
        for nb_channels in [None, 1, 2, 3, 4, 5, 7, 11]:
            arr = np.full((3, 3), value, dtype=dtype)
            if nb_channels is not None:
                arr = arr[..., np.newaxis]
                arr = np.tile(arr, (1, 1, nb_channels))
                for c in sm.xrange(nb_channels):
                    arr[..., c] += c
            arr_pad = ia.pad(arr, top=1, mode="constant", cval=0)
            assert arr_pad.dtype.name == np.dtype(dtype).name
            if nb_channels is None:
                assert arr_pad.shape == (4, 3)
                assert np.all(arr_pad[0, :] == 0)
                assert np.all(arr_pad[1:, :] == arr)
            else:
                assert arr_pad.shape == (4, 3, nb_channels)
                assert np.all(arr_pad[0, :, :] == 0)
                assert np.all(arr_pad[1:, :, :] == arr)

    # -------
    # float
    # -------
    for dtype in [np.float16, np.float32, np.float64, np.float128]:
        arr = np.zeros((3, 3), dtype=dtype) + 1.0

        def _allclose(a, b):
            atol = 1e-3 if dtype == np.float16 else 1e-7
            return np.allclose(a, b, atol=atol, rtol=0)

        arr_pad = ia.pad(arr)
        assert arr_pad.shape == (3, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad, arr)

        arr_pad = ia.pad(arr, top=1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[0, :], dtype([0, 0, 0]))

        arr_pad = ia.pad(arr, right=1)
        assert arr_pad.shape == (3, 4)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[:, -1], dtype([0, 0, 0]))

        arr_pad = ia.pad(arr, bottom=1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[-1, :], dtype([0, 0, 0]))

        arr_pad = ia.pad(arr, left=1)
        assert arr_pad.shape == (3, 4)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[:, 0], dtype([0, 0, 0]))

        arr_pad = ia.pad(arr, top=1, right=2, bottom=3, left=4)
        assert arr_pad.shape == (3+(1+3), 3+(2+4))
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(np.max(arr_pad[0, :]), 0)
        assert _allclose(np.max(arr_pad[:, -2:]), 0)
        assert _allclose(np.max(arr_pad[-3, :]), 0)
        assert _allclose(np.max(arr_pad[:, :4]), 0)

        arr_pad = ia.pad(arr, top=1, cval=0.2)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[0, :], dtype([0.2, 0.2, 0.2]))

        v1 = 1000 ** (np.dtype(dtype).itemsize - 1)
        arr_pad = ia.pad(arr, top=1, cval=v1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[0, :], dtype([v1, v1, v1]))

        v1 = (-1000) ** (np.dtype(dtype).itemsize - 1)
        arr_pad = ia.pad(arr, top=1, cval=v1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[0, :], dtype([v1, v1, v1]))

        arr = np.zeros((3, 3, 3), dtype=dtype) + 0.5
        arr_pad = ia.pad(arr, top=1)
        assert arr_pad.shape == (4, 3, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[0, :, 0], dtype([0, 0, 0]))
        assert _allclose(arr_pad[0, :, 1], dtype([0, 0, 0]))
        assert _allclose(arr_pad[0, :, 2], dtype([0, 0, 0]))

        arr = np.zeros((3, 3), dtype=dtype) + 0.5
        arr[1, 1] = 0.75
        arr_pad = ia.pad(arr, top=1, mode="maximum")
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[0, 0], 0.5)
        assert _allclose(arr_pad[0, 1], 0.75)
        assert _allclose(arr_pad[0, 2], 0.50)

        arr = np.zeros((3, 3), dtype=dtype)
        arr_pad = ia.pad(arr, top=1, mode="constant", cval=0.4)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[0, 0], 0.4)
        assert _allclose(arr_pad[0, 1], 0.4)
        assert _allclose(arr_pad[0, 2], 0.4)
        assert _allclose(arr_pad[1, 0], 0.0)

        for nb_channels in [1, 2, 3, 4, 5]:
            arr = np.zeros((3, 3, nb_channels), dtype=dtype)
            arr_pad = ia.pad(arr, top=1, mode="constant", cval=0.4)
            assert arr_pad.shape == (4, 3, nb_channels)
            assert arr_pad.dtype == np.dtype(dtype)
            assert _allclose(arr_pad[0, 0, :], 0.4)
            assert _allclose(arr_pad[0, 1, :], 0.4)
            assert _allclose(arr_pad[0, 2, :], 0.4)
            assert _allclose(arr_pad[1, 0, :], 0.0)

        arr = np.zeros((1, 1), dtype=dtype) + 0.6
        arr_pad = ia.pad(arr, top=4, mode="linear_ramp", cval=1.0)
        assert arr_pad.shape == (5, 1)
        assert arr_pad.dtype == np.dtype(dtype)
        assert _allclose(arr_pad[0, 0], 1.0)
        assert _allclose(arr_pad[1, 0], 0.9)
        assert _allclose(arr_pad[2, 0], 0.8)
        assert _allclose(arr_pad[3, 0], 0.7)
        assert _allclose(arr_pad[4, 0], 0.6)

        # test other channel numbers
        value = 1000 ** (np.dtype(dtype).itemsize - 1)
        for nb_channels in [None, 1, 2, 3, 4, 5, 7, 11]:
            arr = np.full((3, 3), value, dtype=dtype)
            if nb_channels is not None:
                arr = arr[..., np.newaxis]
                arr = np.tile(arr, (1, 1, nb_channels))
                for c in sm.xrange(nb_channels):
                    arr[..., c] += c
            arr_pad = ia.pad(arr, top=1, mode="constant", cval=0)
            assert arr_pad.dtype.name == np.dtype(dtype).name
            if nb_channels is None:
                assert arr_pad.shape == (4, 3)
                assert _allclose(arr_pad[0, :], 0)
                assert _allclose(arr_pad[1:, :], arr)
            else:
                assert arr_pad.shape == (4, 3, nb_channels)
                assert _allclose(arr_pad[0, :, :], 0)
                assert _allclose(arr_pad[1:, :, :], arr)

    # -------
    # bool
    # -------
    dtype = bool
    arr = np.zeros((3, 3), dtype=dtype)
    arr_pad = ia.pad(arr)
    assert arr_pad.shape == (3, 3)
    # For some reason, arr_pad.dtype.type == dtype fails here for int64 but not for the other dtypes,
    # even though int64 is the dtype of arr_pad. Also checked .name and .str for them -- all same value.
    assert arr_pad.dtype == np.dtype(dtype)
    assert np.all(arr_pad == arr)

    arr_pad = ia.pad(arr, top=1)
    assert arr_pad.shape == (4, 3)
    assert arr_pad.dtype == np.dtype(dtype)
    assert np.all(arr_pad[0, :] == 0)

    arr_pad = ia.pad(arr, top=1, cval=True)
    assert arr_pad.shape == (4, 3)
    assert arr_pad.dtype == np.dtype(dtype)
    assert np.all(arr_pad[0, :] == 1)


def test_compute_paddings_for_aspect_ratio():
    arr = np.zeros((4, 4), dtype=np.uint8)
    top, right, bottom, left = ia.compute_paddings_for_aspect_ratio(arr, 1.0)
    assert top == 0
    assert right == 0
    assert bottom == 0
    assert left == 0

    arr = np.zeros((1, 4), dtype=np.uint8)
    top, right, bottom, left = ia.compute_paddings_for_aspect_ratio(arr, 1.0)
    assert top == 1
    assert right == 0
    assert bottom == 2
    assert left == 0

    arr = np.zeros((4, 1), dtype=np.uint8)
    top, right, bottom, left = ia.compute_paddings_for_aspect_ratio(arr, 1.0)
    assert top == 0
    assert right == 2
    assert bottom == 0
    assert left == 1

    arr = np.zeros((2, 4), dtype=np.uint8)
    top, right, bottom, left = ia.compute_paddings_for_aspect_ratio(arr, 1.0)
    assert top == 1
    assert right == 0
    assert bottom == 1
    assert left == 0

    arr = np.zeros((4, 2), dtype=np.uint8)
    top, right, bottom, left = ia.compute_paddings_for_aspect_ratio(arr, 1.0)
    assert top == 0
    assert right == 1
    assert bottom == 0
    assert left == 1

    arr = np.zeros((4, 4), dtype=np.uint8)
    top, right, bottom, left = ia.compute_paddings_for_aspect_ratio(arr, 0.5)
    assert top == 2
    assert right == 0
    assert bottom == 2
    assert left == 0

    arr = np.zeros((4, 4), dtype=np.uint8)
    top, right, bottom, left = ia.compute_paddings_for_aspect_ratio(arr, 2.0)
    assert top == 0
    assert right == 2
    assert bottom == 0
    assert left == 2


def test_pad_to_aspect_ratio():
    for dtype in [np.uint8, np.int32, np.float32]:
        # aspect_ratio = 1.0
        arr = np.zeros((4, 4), dtype=dtype)
        arr_pad = ia.pad_to_aspect_ratio(arr, 1.0)
        assert arr_pad.dtype.type == dtype
        assert arr_pad.shape[0] == 4
        assert arr_pad.shape[1] == 4

        arr = np.zeros((1, 4), dtype=dtype)
        arr_pad = ia.pad_to_aspect_ratio(arr, 1.0)
        assert arr_pad.dtype.type == dtype
        assert arr_pad.shape[0] == 4
        assert arr_pad.shape[1] == 4

        arr = np.zeros((4, 1), dtype=dtype)
        arr_pad = ia.pad_to_aspect_ratio(arr, 1.0)
        assert arr_pad.dtype.type == dtype
        assert arr_pad.shape[0] == 4
        assert arr_pad.shape[1] == 4

        arr = np.zeros((2, 4), dtype=dtype)
        arr_pad = ia.pad_to_aspect_ratio(arr, 1.0)
        assert arr_pad.dtype.type == dtype
        assert arr_pad.shape[0] == 4
        assert arr_pad.shape[1] == 4

        arr = np.zeros((4, 2), dtype=dtype)
        arr_pad = ia.pad_to_aspect_ratio(arr, 1.0)
        assert arr_pad.dtype.type == dtype
        assert arr_pad.shape[0] == 4
        assert arr_pad.shape[1] == 4

        # aspect_ratio != 1.0
        arr = np.zeros((4, 4), dtype=dtype)
        arr_pad = ia.pad_to_aspect_ratio(arr, 2.0)
        assert arr_pad.dtype.type == dtype
        assert arr_pad.shape[0] == 4
        assert arr_pad.shape[1] == 8

        arr = np.zeros((4, 4), dtype=dtype)
        arr_pad = ia.pad_to_aspect_ratio(arr, 0.5)
        assert arr_pad.dtype.type == dtype
        assert arr_pad.shape[0] == 8
        assert arr_pad.shape[1] == 4

        # 3d arr
        arr = np.zeros((4, 2, 3), dtype=dtype)
        arr_pad = ia.pad_to_aspect_ratio(arr, 1.0)
        assert arr_pad.dtype.type == dtype
        assert arr_pad.shape[0] == 4
        assert arr_pad.shape[1] == 4
        assert arr_pad.shape[2] == 3

    # cval
    arr = np.zeros((4, 4), dtype=np.uint8) + 128
    arr_pad = ia.pad_to_aspect_ratio(arr, 2.0)
    assert arr_pad.shape[0] == 4
    assert arr_pad.shape[1] == 8
    assert np.max(arr_pad[:, 0:2]) == 0
    assert np.max(arr_pad[:, -2:]) == 0
    assert np.max(arr_pad[:, 2:-2]) == 128

    arr = np.zeros((4, 4), dtype=np.uint8) + 128
    arr_pad = ia.pad_to_aspect_ratio(arr, 2.0, cval=10)
    assert arr_pad.shape[0] == 4
    assert arr_pad.shape[1] == 8
    assert np.max(arr_pad[:, 0:2]) == 10
    assert np.max(arr_pad[:, -2:]) == 10
    assert np.max(arr_pad[:, 2:-2]) == 128

    arr = np.zeros((4, 4), dtype=np.float32) + 0.5
    arr_pad = ia.pad_to_aspect_ratio(arr, 2.0, cval=0.0)
    assert arr_pad.shape[0] == 4
    assert arr_pad.shape[1] == 8
    assert 0 - 1e-6 <= np.max(arr_pad[:, 0:2]) <= 0 + 1e-6
    assert 0 - 1e-6 <= np.max(arr_pad[:, -2:]) <= 0 + 1e-6
    assert 0.5 - 1e-6 <= np.max(arr_pad[:, 2:-2]) <= 0.5 + 1e-6

    arr = np.zeros((4, 4), dtype=np.float32) + 0.5
    arr_pad = ia.pad_to_aspect_ratio(arr, 2.0, cval=0.1)
    assert arr_pad.shape[0] == 4
    assert arr_pad.shape[1] == 8
    assert 0.1 - 1e-6 <= np.max(arr_pad[:, 0:2]) <= 0.1 + 1e-6
    assert 0.1 - 1e-6 <= np.max(arr_pad[:, -2:]) <= 0.1 + 1e-6
    assert 0.5 - 1e-6 <= np.max(arr_pad[:, 2:-2]) <= 0.5 + 1e-6

    # mode
    arr = np.zeros((4, 4), dtype=np.uint8) + 128
    arr[1:3, 1:3] = 200
    arr_pad = ia.pad_to_aspect_ratio(arr, 2.0, mode="maximum")
    assert arr_pad.shape[0] == 4
    assert arr_pad.shape[1] == 8
    assert np.max(arr_pad[0:1, 0:2]) == 128
    assert np.max(arr_pad[1:3, 0:2]) == 200
    assert np.max(arr_pad[3:, 0:2]) == 128
    assert np.max(arr_pad[0:1, -2:]) == 128
    assert np.max(arr_pad[1:3, -2:]) == 200
    assert np.max(arr_pad[3:, -2:]) == 128

    # TODO add tests for return_pad_values=True


def test_pool():
    # -----
    # uint, int
    # -----
    for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        for func in [np.min, np.average, np.max]:
            arr = np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]
            ], dtype=dtype)
            arr_pooled = ia.pool(arr, 2, func)
            assert arr_pooled.shape == (2, 2)
            assert arr_pooled.dtype == np.dtype(dtype)
            assert arr_pooled[0, 0] == int(func([0, 1, 4, 5]))
            assert arr_pooled[0, 1] == int(func([2, 3, 6, 7]))
            assert arr_pooled[1, 0] == int(func([8, 9, 12, 13]))
            assert arr_pooled[1, 1] == int(func([10, 11, 14, 15]))

            arr = np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]
            ], dtype=dtype)
            arr = np.tile(arr[:, :, np.newaxis], (1, 1, 3))
            arr[..., 1] += 1
            arr[..., 2] += 2
            arr_pooled = ia.pool(arr, 2, func)
            assert arr_pooled.shape == (2, 2, 3)
            assert arr_pooled.dtype == np.dtype(dtype)
            for c in sm.xrange(3):
                assert arr_pooled[0, 0, c] == int(func([0, 1, 4, 5])) + c
                assert arr_pooled[0, 1, c] == int(func([2, 3, 6, 7])) + c
                assert arr_pooled[1, 0, c] == int(func([8, 9, 12, 13])) + c
                assert arr_pooled[1, 1, c] == int(func([10, 11, 14, 15])) + c

            for value in [min_value, min_value+50, min_value+100, 0, 10, max_value,
                          int(center_value + 0.10*max_value),
                          int(center_value + 0.20*max_value),
                          int(center_value + 0.25*max_value),
                          int(center_value + 0.33*max_value)]:
                arr = np.full((4, 4), value, dtype=dtype)
                arr_pooled = ia.pool(arr, 2, func)
                assert arr_pooled.shape == (2, 2)
                assert arr_pooled.dtype == np.dtype(dtype)
                assert np.all(arr_pooled == value)

                arr = np.full((4, 4, 3), value, dtype=dtype)
                arr_pooled = ia.pool(arr, 2, func)
                assert arr_pooled.shape == (2, 2, 3)
                assert arr_pooled.dtype == np.dtype(dtype)
                assert np.all(arr_pooled == value)

    # -----
    # float
    # -----
    for dtype in [np.float16, np.float32, np.float64, np.float128]:
        def _allclose(a, b):
            atol = 1e-4 if dtype == np.float16 else 1e-8
            return np.allclose(a, b, atol=atol, rtol=0)

        for func in [np.min, np.average, np.max]:
            arr = np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]
            ], dtype=dtype)
            arr_pooled = ia.pool(arr, 2, func)
            assert arr_pooled.shape == (2, 2)
            assert arr_pooled.dtype == np.dtype(dtype)
            assert arr_pooled[0, 0] == func([0, 1, 4, 5])
            assert arr_pooled[0, 1] == func([2, 3, 6, 7])
            assert arr_pooled[1, 0] == func([8, 9, 12, 13])
            assert arr_pooled[1, 1] == func([10, 11, 14, 15])

            arr = np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]
            ], dtype=dtype)
            arr = np.tile(arr[:, :, np.newaxis], (1, 1, 3))
            arr[..., 1] += 1
            arr[..., 2] += 2
            arr_pooled = ia.pool(arr, 2, func)
            assert arr_pooled.shape == (2, 2, 3)
            assert arr_pooled.dtype == np.dtype(dtype)
            for c in sm.xrange(3):
                assert arr_pooled[0, 0, c] == func([0, 1, 4, 5]) + c
                assert arr_pooled[0, 1, c] == func([2, 3, 6, 7]) + c
                assert arr_pooled[1, 0, c] == func([8, 9, 12, 13]) + c
                assert arr_pooled[1, 1, c] == func([10, 11, 14, 15]) + c

            isize = np.dtype(dtype).itemsize
            for value in [(-1) * (1000 ** (isize-1)), -50.0, 0.0, 50.0, 1000 ** (isize-1)]:
                arr = np.full((4, 4), value, dtype=dtype)
                arr_pooled = ia.pool(arr, 2, func)
                dt = np.result_type(arr_pooled, 1.)
                y = np.array(arr_pooled, dtype=dt, copy=False, subok=True)
                assert arr_pooled.shape == (2, 2)
                assert arr_pooled.dtype == np.dtype(dtype)
                assert _allclose(arr_pooled, float(value))

                arr = np.full((4, 4, 3), value, dtype=dtype)
                arr_pooled = ia.pool(arr, 2, func)
                assert arr_pooled.shape == (2, 2, 3)
                assert arr_pooled.dtype == np.dtype(dtype)
                assert _allclose(arr_pooled, float(value))

    # ----
    # bool
    # ----
    arr = np.zeros((4, 4), dtype=bool)
    arr[0, 0] = True
    arr[0, 1] = True
    arr[1, 0] = True
    arr_pooled = ia.pool(arr, 2, np.min)
    assert arr_pooled.dtype == arr.dtype
    assert np.all(arr_pooled == 0)

    arr_pooled = ia.pool(arr, 2, np.average)
    assert arr_pooled.dtype == arr.dtype
    assert np.all(arr_pooled[0, 0] == 1)
    assert np.all(arr_pooled[:, 1] == 0)
    assert np.all(arr_pooled[1, :] == 0)

    arr_pooled = ia.pool(arr, 2, np.max)
    assert arr_pooled.dtype == arr.dtype
    assert np.all(arr_pooled[0, 0] == 1)
    assert np.all(arr_pooled[:, 1] == 0)
    assert np.all(arr_pooled[1, :] == 0)

    # preserve_dtype off
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    arr_pooled = ia.pool(arr, 2, np.average, preserve_dtype=False)
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == np.float64
    assert np.allclose(arr_pooled[0, 0], np.average([0, 1, 4, 5]))
    assert np.allclose(arr_pooled[0, 1], np.average([2, 3, 6, 7]))
    assert np.allclose(arr_pooled[1, 0], np.average([8, 9, 12, 13]))
    assert np.allclose(arr_pooled[1, 1], np.average([10, 11, 14, 15]))

    # maximum function
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    arr_pooled = ia.pool(arr, 2, np.max)
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.max([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.max([2, 3, 6, 7]))
    assert arr_pooled[1, 0] == int(np.max([8, 9, 12, 13]))
    assert arr_pooled[1, 1] == int(np.max([10, 11, 14, 15]))

    # 3d array
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    arr = np.tile(arr[..., np.newaxis], (1, 1, 3))
    arr_pooled = ia.pool(arr, 2, np.average)
    assert arr_pooled.shape == (2, 2, 3)
    assert np.array_equal(arr_pooled[..., 0], arr_pooled[..., 1])
    assert np.array_equal(arr_pooled[..., 1], arr_pooled[..., 2])
    arr_pooled = arr_pooled[..., 0]
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.average([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.average([2, 3, 6, 7]))
    assert arr_pooled[1, 0] == int(np.average([8, 9, 12, 13]))
    assert arr_pooled[1, 1] == int(np.average([10, 11, 14, 15]))

    # block_size per axis
    arr = np.float32([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    arr_pooled = ia.pool(arr, (2, 1), np.average)
    assert arr_pooled.shape == (2, 4)
    assert arr_pooled.dtype == arr.dtype.type
    assert np.allclose(arr_pooled[0, 0], np.average([0, 4]))
    assert np.allclose(arr_pooled[0, 1], np.average([1, 5]))
    assert np.allclose(arr_pooled[0, 2], np.average([2, 6]))
    assert np.allclose(arr_pooled[0, 3], np.average([3, 7]))
    assert np.allclose(arr_pooled[1, 0], np.average([8, 12]))
    assert np.allclose(arr_pooled[1, 1], np.average([9, 13]))
    assert np.allclose(arr_pooled[1, 2], np.average([10, 14]))
    assert np.allclose(arr_pooled[1, 3], np.average([11, 15]))

    # cval
    arr = np.uint8([
        [0, 1, 2],
        [4, 5, 6],
        [8, 9, 10]
    ])
    arr_pooled = ia.pool(arr, 2, np.average)
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.average([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.average([2, 0, 6, 0]))
    assert arr_pooled[1, 0] == int(np.average([8, 9, 0, 0]))
    assert arr_pooled[1, 1] == int(np.average([10, 0, 0, 0]))

    arr = np.uint8([
        [0, 1],
        [4, 5]
    ])
    arr_pooled = ia.pool(arr, (4, 1), np.average)
    assert arr_pooled.shape == (1, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.average([0, 4, 0, 0]))
    assert arr_pooled[0, 1] == int(np.average([1, 5, 0, 0]))

    arr = np.uint8([
        [0, 1, 2],
        [4, 5, 6],
        [8, 9, 10]
    ])
    arr_pooled = ia.pool(arr, 2, np.average, cval=22)
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.average([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.average([2, 22, 6, 22]))
    assert arr_pooled[1, 0] == int(np.average([8, 9, 22, 22]))
    assert arr_pooled[1, 1] == int(np.average([10, 22, 22, 22]))


def test_avg_pool():
    # very basic test, as avg_pool() just calls pool(), which is tested in test_pool()
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    arr_pooled = ia.avg_pool(arr, 2)
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.average([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.average([2, 3, 6, 7]))
    assert arr_pooled[1, 0] == int(np.average([8, 9, 12, 13]))
    assert arr_pooled[1, 1] == int(np.average([10, 11, 14, 15]))


def test_max_pool():
    # very basic test, as avg_pool() just calls pool(), which is tested in test_pool()
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    arr_pooled = ia.max_pool(arr, 2)
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.max([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.max([2, 3, 6, 7]))
    assert arr_pooled[1, 0] == int(np.max([8, 9, 12, 13]))
    assert arr_pooled[1, 1] == int(np.max([10, 11, 14, 15]))


def test_draw_grid():
    # bool
    dtype = bool
    image = np.zeros((2, 2, 3), dtype=dtype)

    image[0, 0] = False
    image[0, 1] = True
    image[1, 0] = True
    image[1, 1] = False

    grid = ia.draw_grid([image], rows=1, cols=1)
    assert grid.dtype == np.dtype(dtype)
    assert np.array_equal(grid, image)

    grid = ia.draw_grid(np.array([image], dtype=dtype), rows=1, cols=1)
    assert grid.dtype == np.dtype(dtype)
    assert np.array_equal(grid, image)

    grid = ia.draw_grid([image, image, image, image], rows=2, cols=2)
    expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert grid.dtype == np.dtype(dtype)
    assert np.array_equal(grid, expected)

    grid = ia.draw_grid([image, image], rows=1, cols=2)
    expected = np.hstack([image, image])
    assert grid.dtype == np.dtype(dtype)
    assert np.array_equal(grid, expected)

    grid = ia.draw_grid([image, image, image, image], rows=2, cols=None)
    expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert grid.dtype == np.dtype(dtype)
    assert np.array_equal(grid, expected)

    grid = ia.draw_grid([image, image, image, image], rows=None, cols=2)
    expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert grid.dtype == np.dtype(dtype)
    assert np.array_equal(grid, expected)

    grid = ia.draw_grid([image, image, image, image], rows=None, cols=None)
    expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert grid.dtype == np.dtype(dtype)
    assert np.array_equal(grid, expected)

    # int, uint
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        image = np.zeros((2, 2, 3), dtype=dtype)

        image[0, 0] = min_value
        image[0, 1] = center_value
        image[1, 0] = center_value + int(0.3 * max_value)
        image[1, 1] = max_value

        grid = ia.draw_grid([image], rows=1, cols=1)
        assert grid.dtype == np.dtype(dtype)
        assert np.array_equal(grid, image)

        grid = ia.draw_grid(np.array([image], dtype=dtype), rows=1, cols=1)
        assert grid.dtype == np.dtype(dtype)
        assert np.array_equal(grid, image)

        grid = ia.draw_grid([image, image, image, image], rows=2, cols=2)
        expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert np.array_equal(grid, expected)

        grid = ia.draw_grid([image, image], rows=1, cols=2)
        expected = np.hstack([image, image])
        assert grid.dtype == np.dtype(dtype)
        assert np.array_equal(grid, expected)

        grid = ia.draw_grid([image, image, image, image], rows=2, cols=None)
        expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert grid.dtype == np.dtype(dtype)
        assert np.array_equal(grid, expected)

        grid = ia.draw_grid([image, image, image, image], rows=None, cols=2)
        expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert grid.dtype == np.dtype(dtype)
        assert np.array_equal(grid, expected)

        grid = ia.draw_grid([image, image, image, image], rows=None, cols=None)
        expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert grid.dtype == np.dtype(dtype)
        assert np.array_equal(grid, expected)

    # float
    for dtype in [np.float16, np.float32, np.float64, np.float128]:
        def _allclose(a, b):
            atol = 1e-4 if dtype == np.float16 else 1e-8
            return np.allclose(a, b, atol=atol, rtol=0)

        image = np.zeros((2, 2, 3), dtype=dtype)

        isize = np.dtype(dtype).itemsize
        image[0, 0] = (-1) * (1000 ** (isize-1))
        image[0, 1] = -10.0
        image[1, 0] = 10.0
        image[1, 1] = 1000 ** (isize-1)

        grid = ia.draw_grid([image], rows=1, cols=1)
        assert grid.dtype == np.dtype(dtype)
        assert _allclose(grid, image)

        grid = ia.draw_grid(np.array([image], dtype=dtype), rows=1, cols=1)
        assert grid.dtype == np.dtype(dtype)
        assert _allclose(grid, image)

        grid = ia.draw_grid([image, image, image, image], rows=2, cols=2)
        expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert grid.dtype == np.dtype(dtype)
        assert _allclose(grid, expected)

        grid = ia.draw_grid([image, image], rows=1, cols=2)
        expected = np.hstack([image, image])
        assert grid.dtype == np.dtype(dtype)
        assert _allclose(grid, expected)

        grid = ia.draw_grid([image, image, image, image], rows=2, cols=None)
        expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert grid.dtype == np.dtype(dtype)
        assert _allclose(grid, expected)

        grid = ia.draw_grid([image, image, image, image], rows=None, cols=2)
        expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert grid.dtype == np.dtype(dtype)
        assert _allclose(grid, expected)

        grid = ia.draw_grid([image, image, image, image], rows=None, cols=None)
        expected = np.vstack([
            np.hstack([image, image]),
            np.hstack([image, image])
        ])
        assert grid.dtype == np.dtype(dtype)
        assert _allclose(grid, expected)


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


class TestBatch(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_init(self):
        attr_names = ["images", "heatmaps", "segmentation_maps", "keypoints",
                      "bounding_boxes", "polygons"]
        batch = ia.Batch()
        for attr_name in attr_names:
            assert getattr(batch, "%s_unaug" % (attr_name,)) is None
            assert getattr(batch, "%s_aug" % (attr_name,)) is None
        assert batch.data is None

        # we exploit here that Batch() init does not verify its inputs
        batch = ia.Batch(
            images=0,
            heatmaps=1,
            segmentation_maps=2,
            keypoints=3,
            bounding_boxes=4,
            polygons=5,
            data=6
        )
        for i, attr_name in enumerate(attr_names):
            assert getattr(batch, "%s_unaug" % (attr_name,)) == i
            assert getattr(batch, "%s_aug" % (attr_name,)) is None
        assert batch.data == 6

    def get_images_unaug_normalized(self):
        batch = ia.Batch()
        assert batch.get_images_unaug_normalized() is None

        batch = ia.Batch(images=None)
        assert batch.get_images_unaug_normalized() is None

        arr = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        batch = ia.Batch(images=arr)
        observed = batch.get_images_unaug_normalized()
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4, 3)
        assert observed.dtype.name == "uint8"

        arr = np.zeros((1, 4, 4), dtype=np.uint8)
        batch = ia.Batch(images=arr)
        observed = batch.get_images_unaug_normalized()
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4, 3)
        assert observed.dtype.name == "uint8"

        arr = np.zeros((4, 4), dtype=np.uint8)
        batch = ia.Batch(images=arr)
        observed = batch.get_images_unaug_normalized()
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4, 1)
        assert observed.dtype.name == "uint8"

        batch = ia.Batch(images=[])
        assert isinstance(batch.get_images_unaug_normalized(), list)
        assert len(batch.get_images_unaug_normalized()) == 0

        arr1 = np.zeros((4, 4), dtype=np.uint8)
        arr2 = np.zeros((5, 5, 3), dtype=np.uint8)
        batch = ia.Batch(images=[arr1, arr2])
        observed = batch.get_images_unaug_normalized()
        assert isinstance(observed, list)
        assert len(observed) == 2
        assert ia.is_np_array(observed[0])
        assert ia.is_np_array(observed[1])
        assert observed[0].shape == (4, 4)
        assert observed[1].shape == (5, 5, 3)
        assert observed[0].dtype.name == "uint8"
        assert observed[1].dtype.name == "uint8"

        batch = ia.Batch(images=False)
        with self.assertRaises(ValueError):
            batch.get_images_unaug_normalized()

    def test_get_heatmaps_unaug_normalized(self):
        # ----
        # None
        # ----
        batch = ia.Batch(heatmaps=None)
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert heatmaps_norm is None

        # ----
        # array
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            heatmaps=np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1)
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        batch = ia.Batch(
            images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            heatmaps=np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1)
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        # --> heatmaps for too many images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                heatmaps=np.zeros((2, 1, 1, 1), dtype=np.float32) + 0.1)
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> too few heatmaps
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                heatmaps=np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1)
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> wrong channel number
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                heatmaps=np.zeros((1, 1, 1), dtype=np.float32) + 0.1)
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                heatmaps=np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1)
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # ----
        # single HeatmapsOnImage
        # ----
        batch = ia.Batch(
            images=None,
            heatmaps=ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                shape=(1, 1, 3)))
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        # ----
        # empty iterable
        # ----
        batch = ia.Batch(images=None, heatmaps=[])
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert heatmaps_norm is None

        # ----
        # iterable of arrays
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            heatmaps=[np.zeros((1, 1, 1), dtype=np.float32) + 0.1])
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        batch = ia.Batch(
            images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            heatmaps=[np.zeros((1, 1, 1), dtype=np.float32) + 0.1])
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        # --> heatmaps for too many images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                heatmaps=[
                    np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                    np.zeros((1, 1, 1), dtype=np.float32) + 0.1
                ])
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> too few heatmaps
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                heatmaps=[np.zeros((1, 1, 1), dtype=np.float32) + 0.1])
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                heatmaps=[np.zeros((1, 1, 1), dtype=np.float32) + 0.1])
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> wrong number of dimensions
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                heatmaps=[np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1])
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # ----
        # iterable of HeatmapsOnImage
        # ----
        batch = ia.Batch(
            images=None,
            heatmaps=[ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                shape=(1, 1, 3))])
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

    def test_get_segmentation_maps_unaug_normalized(self):
        # ----
        # None
        # ----
        batch = ia.Batch(segmentation_maps=None)
        segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
        assert segmaps_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("int32"), np.dtype("uint32"), np.dtype(bool)]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                segmentation_maps=np.zeros((1, 1, 1, 1), dtype=dt) + 1)
            segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                segmentation_maps=np.zeros((1, 1, 1, 1), dtype=dt) + 1)
            segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            # --> heatmaps for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    segmentation_maps=np.zeros((2, 1, 1, 1), dtype=dt) + 1)
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> too few heatmaps
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    segmentation_maps=np.zeros((1, 1, 1, 1), dtype=dt) + 1)
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> wrong channel number
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    segmentation_maps=np.zeros((1, 1, 1), dtype=dt) + 1)
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    segmentation_maps=np.zeros((1, 1, 1, 1), dtype=dt) + 1)
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

        # ----
        # single SegmentationMapOnImage
        # ----
        batch = ia.Batch(
            images=None,
            segmentation_maps=ia.SegmentationMapOnImage(
                np.zeros((1, 1, 1), dtype=np.int32) + 1,
                shape=(1, 1, 3),
                nb_classes=2))
        segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
        assert isinstance(segmaps_norm, list)
        assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
        assert np.allclose(segmaps_norm[0].arr[..., 1], 0 + 1)

        # ----
        # empty iterable
        # ----
        batch = ia.Batch(images=None, segmentation_maps=[])
        segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
        assert segmaps_norm is None

        # ----
        # iterable of arrays
        # ----
        for dt in [np.dtype("int32"), np.dtype("uint32"), np.dtype(bool)]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                segmentation_maps=[np.zeros((1, 1, 1), dtype=dt) + 1])
            segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                segmentation_maps=[np.zeros((1, 1, 1), dtype=dt) + 1])
            segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            # --> heatmaps for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    segmentation_maps=[
                        np.zeros((1, 1, 1), dtype=np.int32) + 1,
                        np.zeros((1, 1, 1), dtype=np.int32) + 1
                    ])
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> too few heatmaps
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    segmentation_maps=[
                        np.zeros((1, 1, 1), dtype=np.int32) + 1])
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    segmentation_maps=[
                        np.zeros((1, 1, 1), dtype=np.int32) + 1])
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> wrong number of dimensions
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    segmentation_maps=[
                        np.zeros((1, 1, 1, 1), dtype=np.int32) + 1])
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

        # ----
        # iterable of SegmentationMapOnImage
        # ----
        batch = ia.Batch(
            images=None,
            segmentation_maps=[ia.SegmentationMapOnImage(
                np.zeros((1, 1, 1), dtype=np.int32) + 1,
                shape=(1, 1, 3),
                nb_classes=2)])
        segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
        assert isinstance(segmaps_norm, list)
        assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
        assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

    def test_get_keypoints_unaug_normalized(self):
        def _assert_single_image_expected(inputs):
            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    keypoints=inputs)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    keypoints=inputs)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8),
                            np.zeros((1, 1, 3), dtype=np.uint8)],
                    keypoints=inputs)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # ----
        # None
        # ----
        batch = ia.Batch(keypoints=None)
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert keypoints_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=np.zeros((1, 1, 2), dtype=dt) + 1)
            keypoints_norm = batch.get_keypoints_unaug_normalized()
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 1
            assert np.allclose(keypoints_norm[0].get_coords_array(), 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                keypoints=np.zeros((1, 5, 2), dtype=dt) + 1)
            keypoints_norm = batch.get_keypoints_unaug_normalized()
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 5
            assert np.allclose(keypoints_norm[0].get_coords_array(), 1)

            # --> keypoints for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    keypoints=np.zeros((2, 1, 2), dtype=dt) + 1)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> too few keypoints
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    keypoints=np.zeros((1, 1, 2), dtype=dt) + 1)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> wrong keypoints shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    keypoints=np.zeros((1, 1, 100), dtype=dt) + 1)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            _assert_single_image_expected(np.zeros((1, 1, 2), dtype=dt) + 1)

        # ----
        # (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=(1, 2))
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        _assert_single_image_expected((1, 2))

        # ----
        # single Keypoint instance
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=ia.Keypoint(x=1, y=2))
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        _assert_single_image_expected(ia.Keypoint(x=1, y=2))

        # ----
        # single KeypointsOnImage instance
        # ----
        batch = ia.Batch(
            images=None,
            keypoints=ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3))
        )
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        # ----
        # empty iterable
        # ----
        batch = ia.Batch(
            images=None,
            keypoints=[]
        )
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert keypoints_norm is None

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=[np.zeros((1, 2), dtype=dt) + 1])
            keypoints_norm = batch.get_keypoints_unaug_normalized()
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 1
            assert np.allclose(keypoints_norm[0].get_coords_array(), 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                keypoints=[np.zeros((5, 2), dtype=dt) + 1])
            keypoints_norm = batch.get_keypoints_unaug_normalized()
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 5
            assert np.allclose(keypoints_norm[0].get_coords_array(), 1)

            # --> keypoints for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    keypoints=[
                        np.zeros((1, 2), dtype=dt) + 1,
                        np.zeros((1, 2), dtype=dt) + 1
                    ])
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> too few keypoints
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    keypoints=[
                        np.zeros((1, 2), dtype=dt) + 1])
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    keypoints=[
                        np.zeros((1, 2), dtype=dt) + 1])
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> wrong shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    keypoints=[
                        np.zeros((1, 100), dtype=dt) + 1])
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # ----
        # iterable of (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=[(1, 2), (3, 4)])
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 2
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2
        assert keypoints_norm[0].keypoints[1].x == 3
        assert keypoints_norm[0].keypoints[1].y == 4

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=[(1, 2)])
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # ----
        # iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=[ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)])
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 2
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2
        assert keypoints_norm[0].keypoints[1].x == 3
        assert keypoints_norm[0].keypoints[1].y == 4

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=[ia.Keypoint(x=1, y=2)])
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # ----
        # iterable of KeypointsOnImage
        # ----
        batch = ia.Batch(
            images=None,
            keypoints=[
                ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3)),
                ia.KeypointsOnImage([ia.Keypoint(x=3, y=4)], shape=(1, 1, 3)),
            ]
        )
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)

        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        assert isinstance(keypoints_norm[1], ia.KeypointsOnImage)
        assert len(keypoints_norm[1].keypoints) == 1
        assert keypoints_norm[1].keypoints[0].x == 3
        assert keypoints_norm[1].keypoints[0].y == 4

        # ----
        # iterable of empty interables
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=[[]])
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert keypoints_norm is None

        # ----
        # iterable of iterable of (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=[
                [(1, 2), (3, 4)],
                [(5, 6), (7, 8)]
            ]
        )
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 2
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2
        assert keypoints_norm[0].keypoints[1].x == 3
        assert keypoints_norm[0].keypoints[1].y == 4

        assert len(keypoints_norm[1].keypoints) == 2
        assert keypoints_norm[1].keypoints[0].x == 5
        assert keypoints_norm[1].keypoints[0].y == 6
        assert keypoints_norm[1].keypoints[1].x == 7
        assert keypoints_norm[1].keypoints[1].y == 8

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                keypoints=[
                    [(1, 2), (3, 4)],
                    [(5, 6), (7, 8)]
                ]
            )
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=[
                    [(1, 2), (3, 4)],
                    [(5, 6), (7, 8)]
                ]
            )
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # ----
        # iterable of iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=[
                [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
                [ia.Keypoint(x=5, y=6), ia.Keypoint(x=7, y=8)]
            ]
        )
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 2
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2
        assert keypoints_norm[0].keypoints[1].x == 3
        assert keypoints_norm[0].keypoints[1].y == 4

        assert len(keypoints_norm[1].keypoints) == 2
        assert keypoints_norm[1].keypoints[0].x == 5
        assert keypoints_norm[1].keypoints[0].y == 6
        assert keypoints_norm[1].keypoints[1].x == 7
        assert keypoints_norm[1].keypoints[1].y == 8

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                keypoints=[
                    [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
                    [ia.Keypoint(x=5, y=6), ia.Keypoint(x=7, y=8)]
                ]
            )
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=[
                    [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
                    [ia.Keypoint(x=5, y=6), ia.Keypoint(x=7, y=8)]
                ]
            )
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

    def test_get_bounding_boxes_unaug_normalized(self):
        def _assert_single_image_expected(inputs):
            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    bounding_boxes=inputs)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    bounding_boxes=inputs)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8),
                            np.zeros((1, 1, 3), dtype=np.uint8)],
                    bounding_boxes=inputs)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # ----
        # None
        # ----
        batch = ia.Batch(bounding_boxes=None)
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert bbs_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=np.zeros((1, 1, 4), dtype=dt) + 1)
            bbs_norm = batch.get_bounding_boxes_unaug_normalized()
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 1
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                bounding_boxes=np.zeros((1, 5, 4), dtype=dt) + 1)
            bbs_norm = batch.get_bounding_boxes_unaug_normalized()
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 5
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            # --> bounding boxes for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    bounding_boxes=np.zeros((2, 1, 4), dtype=dt) + 1)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> too few bounding boxes
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    bounding_boxes=np.zeros((1, 1, 4), dtype=dt) + 1)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> wrong keypoints shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    bounding_boxes=np.zeros((1, 1, 100), dtype=dt) + 1)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            _assert_single_image_expected(np.zeros((1, 1, 4), dtype=dt) + 1)

        # ----
        # (x1,y1,x2,y2)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=(1, 2, 3, 4))
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        _assert_single_image_expected((1, 4))

        # ----
        # single BoundingBox instance
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=ia.BoundingBox(x1=1, y1=2, x2=3, y2=4))
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        _assert_single_image_expected(ia.BoundingBox(x1=1, y1=2, x2=3, y2=4))

        # ----
        # single BoundingBoxesOnImage instance
        # ----
        batch = ia.Batch(
            images=None,
            bounding_boxes=ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                shape=(1, 1, 3))
        )
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        # ----
        # empty iterable
        # ----
        batch = ia.Batch(
            images=None,
            bounding_boxes=[]
        )
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert bbs_norm is None

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=[np.zeros((1, 4), dtype=dt) + 1])
            bbs_norm = batch.get_bounding_boxes_unaug_normalized()
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 1
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                bounding_boxes=[np.zeros((5, 4), dtype=dt) + 1])
            bbs_norm = batch.get_bounding_boxes_unaug_normalized()
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 5
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            # --> bounding boxes for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    bounding_boxes=[
                        np.zeros((1, 4), dtype=dt) + 1,
                        np.zeros((1, 4), dtype=dt) + 1
                    ])
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> too few bounding boxes
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    bounding_boxes=[
                        np.zeros((1, 4), dtype=dt) + 1])
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    bounding_boxes=[
                        np.zeros((1, 4), dtype=dt) + 1])
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> wrong shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    bounding_boxes=[
                        np.zeros((1, 100), dtype=dt) + 1])
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # ----
        # iterable of (x1,y1,x2,y2)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=[(1, 2, 3, 4), (5, 6, 7, 8)])
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 2
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4
        assert bbs_norm[0].bounding_boxes[1].x1 == 5
        assert bbs_norm[0].bounding_boxes[1].y1 == 6
        assert bbs_norm[0].bounding_boxes[1].x2 == 7
        assert bbs_norm[0].bounding_boxes[1].y2 == 8

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=[(1, 4)])
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # ----
        # iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=[
                ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)
            ])
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 2
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4
        assert bbs_norm[0].bounding_boxes[1].x1 == 5
        assert bbs_norm[0].bounding_boxes[1].y1 == 6
        assert bbs_norm[0].bounding_boxes[1].x2 == 7
        assert bbs_norm[0].bounding_boxes[1].y2 == 8

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=[ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)])
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # ----
        # iterable of BoundingBoxesOnImage
        # ----
        batch = ia.Batch(
            images=None,
            bounding_boxes=[
                ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                    shape=(1, 1, 3)),
                ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                    shape=(1, 1, 3))
            ]
        )
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)

        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        assert isinstance(bbs_norm[1], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[1].bounding_boxes) == 1
        assert bbs_norm[1].bounding_boxes[0].x1 == 5
        assert bbs_norm[1].bounding_boxes[0].y1 == 6
        assert bbs_norm[1].bounding_boxes[0].x2 == 7
        assert bbs_norm[1].bounding_boxes[0].y2 == 8

        # ----
        # iterable of empty interables
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=[[]])
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert bbs_norm is None

        # ----
        # iterable of iterable of (x1,y1,x2,y2)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=[
                [(1, 2, 3, 4)],
                [(5, 6, 7, 8), (9, 10, 11, 12)]
            ]
        )
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        assert len(bbs_norm[1].bounding_boxes) == 2
        assert bbs_norm[1].bounding_boxes[0].x1 == 5
        assert bbs_norm[1].bounding_boxes[0].y1 == 6
        assert bbs_norm[1].bounding_boxes[0].x2 == 7
        assert bbs_norm[1].bounding_boxes[0].y2 == 8

        assert bbs_norm[1].bounding_boxes[1].x1 == 9
        assert bbs_norm[1].bounding_boxes[1].y1 == 10
        assert bbs_norm[1].bounding_boxes[1].x2 == 11
        assert bbs_norm[1].bounding_boxes[1].y2 == 12

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                bounding_boxes=[
                    [(1, 4), (3, 4)],
                    [(5, 6), (7, 8)]
                ]
            )
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=[
                    [(1, 2, 3, 4)],
                    [(5, 6, 7, 8)]
                ]
            )
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # ----
        # iterable of iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=[
                [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                 ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                [ia.BoundingBox(x1=9, y1=10, x2=11, y2=12),
                 ia.BoundingBox(x1=13, y1=14, x2=15, y2=16)]
            ]
        )
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 2
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4
        assert bbs_norm[0].bounding_boxes[1].x1 == 5
        assert bbs_norm[0].bounding_boxes[1].y1 == 6
        assert bbs_norm[0].bounding_boxes[1].x2 == 7
        assert bbs_norm[0].bounding_boxes[1].y2 == 8

        assert len(bbs_norm[1].bounding_boxes) == 2
        assert bbs_norm[1].bounding_boxes[0].x1 == 9
        assert bbs_norm[1].bounding_boxes[0].y1 == 10
        assert bbs_norm[1].bounding_boxes[0].x2 == 11
        assert bbs_norm[1].bounding_boxes[0].y2 == 12
        assert bbs_norm[1].bounding_boxes[1].x1 == 13
        assert bbs_norm[1].bounding_boxes[1].y1 == 14
        assert bbs_norm[1].bounding_boxes[1].x2 == 15
        assert bbs_norm[1].bounding_boxes[1].y2 == 16

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                bounding_boxes=[
                    [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                     ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                    [ia.BoundingBox(x1=9, y1=10, x2=11, y2=12),
                     ia.BoundingBox(x1=13, y1=14, x2=15, y2=16)]
                ]
            )
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=[
                    [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                     ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                    [ia.BoundingBox(x1=9, y1=10, x2=11, y2=12),
                     ia.BoundingBox(x1=13, y1=14, x2=15, y2=16)]
                ]
            )
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

    def test_get_polygons_unaug_normalized(self):
        def _assert_single_image_expected(inputs):
            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    polygons=inputs)
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    polygons=inputs)
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8),
                            np.zeros((1, 1, 3), dtype=np.uint8)],
                    polygons=inputs)
                _polygons_norm = batch.get_polygons_unaug_normalized()

        coords1 = [(0, 0), (10, 0), (10, 10)]
        coords2 = [(5, 5), (15, 5), (15, 15)]
        coords3 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        coords4 = [(5, 5), (15, 5), (15, 15), (5, 15)]

        coords1_kps = [ia.Keypoint(x=x, y=y) for x, y in coords1]
        coords2_kps = [ia.Keypoint(x=x, y=y) for x, y in coords2]
        coords3_kps = [ia.Keypoint(x=x, y=y) for x, y in coords3]
        coords4_kps = [ia.Keypoint(x=x, y=y) for x, y in coords4]

        coords1_arr = np.float32(coords1)
        coords2_arr = np.float32(coords2)
        coords3_arr = np.float32(coords3)
        coords4_arr = np.float32(coords4)

        # ----
        # None
        # ----
        batch = ia.Batch(polygons=None)
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert polygons_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=coords1_arr[np.newaxis, np.newaxis, ...].astype(dt))
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 1
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                polygons=np.tile(
                    coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                    (1, 5, 1, 1)
                ))
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 5
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            # --> polygons for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    polygons=np.tile(
                        coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                        (2, 1, 1, 1)
                    ))
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> too few polygons
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    polygons=np.tile(
                        coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                        (1, 1, 1, 1)
                    ))
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> wrong polygons shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    polygons=np.tile(
                        coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                        (1, 1, 1, 10)
                    ))
                _polygons_norm = batch.get_polygons_unaug_normalized()

            _assert_single_image_expected(
                coords1_arr[np.newaxis, np.newaxis, ...].astype(dt))

        # ----
        # single Polygon instance
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=ia.Polygon(coords1))
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        _assert_single_image_expected(ia.Polygon(coords1))

        # ----
        # single PolygonsOnImage instance
        # ----
        batch = ia.Batch(
            images=None,
            polygons=ia.PolygonsOnImage([ia.Polygon(coords1)], shape=(1, 1, 3))
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        # ----
        # empty iterable
        # ----
        batch = ia.Batch(
            images=None,
            polygons=[]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert polygons_norm is None

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[coords1_arr[np.newaxis, ...].astype(dt)])
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 1
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                polygons=[np.tile(
                    coords1_arr[np.newaxis, ...].astype(dt),
                    (5, 1, 1)
                )])
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 5
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            # --> polygons for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    polygons=[coords1_arr[np.newaxis, ...].astype(dt),
                              coords2_arr[np.newaxis, ...].astype(dt)])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> too few polygons
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    polygons=[coords1_arr[np.newaxis, ...].astype(dt)])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> wrong polygons shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    polygons=[np.tile(
                        coords1_arr[np.newaxis, ...].astype(dt),
                        (1, 1, 10)
                    )])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            _assert_single_image_expected(
                [coords1_arr[np.newaxis, ...].astype(dt)]
            )

        # ----
        # iterable of (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=coords1)
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=coords1)
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=coords1_kps)
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=coords1_kps)
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of Polygon
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[ia.Polygon(coords1), ia.Polygon(coords2)])
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[ia.Polygon(coords1)])
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of PolygonsOnImage
        # ----
        batch = ia.Batch(
            images=None,
            polygons=[
                ia.PolygonsOnImage([ia.Polygon(coords1)], shape=(1, 1, 3)),
                ia.PolygonsOnImage([ia.Polygon(coords2)], shape=(1, 1, 3))
            ]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)

        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        assert isinstance(polygons_norm[1], ia.PolygonsOnImage)
        assert len(polygons_norm[1].polygons) == 1
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords2)

        # ----
        # iterable of empty interables
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[[]])
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert polygons_norm is None

        # ----
        # iterable of iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[[coords1_arr.astype(dt)]])
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 1
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                polygons=[[
                    np.copy(coords1_arr).astype(dt) for _ in sm.xrange(5)
                ]])
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 5
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            # --> polygons for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    polygons=[[coords1_arr.astype(dt)],
                              [coords2_arr.astype(dt)]])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> too few polygons
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    polygons=[[coords1_arr.astype(dt)]])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> wrong polygons shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    polygons=[[np.tile(
                        coords1_arr.astype(dt),
                        (1, 1, 10)
                    )]])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            _assert_single_image_expected(
                [[coords1_arr.astype(dt)]]
            )

        # ----
        # iterable of iterable of (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[coords1, coords2]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                polygons=[coords1, coords2]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[coords1, coords2]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[coords1_kps, coords2_kps]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                polygons=[coords1_kps, coords2_kps]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[coords1_kps, coords2_kps]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of iterable of Polygon
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[
                [ia.Polygon(coords1), ia.Polygon(coords2)],
                [ia.Polygon(coords3), ia.Polygon(coords4)]
            ]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert isinstance(polygons_norm[1], ia.PolygonsOnImage)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        assert len(polygons_norm[1].polygons) == 2
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords3)
        assert polygons_norm[1].polygons[1].exterior_almost_equals(coords4)

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                polygons=[
                    [ia.Polygon(coords1), ia.Polygon(coords2)],
                    [ia.Polygon(coords3), ia.Polygon(coords4)]
                ]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[
                    [ia.Polygon(coords1), ia.Polygon(coords2)],
                    [ia.Polygon(coords3), ia.Polygon(coords4)]
                ]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of iterable of empty iterable
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[[[]]])
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert polygons_norm is None

        # ----
        # iterable of iterable of iterable of (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[[coords1, coords2], [coords3, coords4]]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords3)
        assert polygons_norm[1].polygons[1].exterior_almost_equals(coords4)

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                polygons=[[coords1, coords2]]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[[coords1, coords2], [coords3]]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of iterable of iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[[coords1_kps, coords2_kps], [coords3_kps, coords4_kps]]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords3)
        assert polygons_norm[1].polygons[1].exterior_almost_equals(coords4)

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                polygons=[[coords1_kps, coords2_kps]]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[[coords1_kps, coords2_kps], [coords3_kps]]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

    def test__get_heatmaps_unaug_normalization_type(self):
        batch = ia.Batch(heatmaps=None)
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "None"

        batch = ia.Batch(heatmaps=np.zeros((1, 1, 1, 1), dtype=np.float32))
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "array[float]"

        batch = ia.Batch(heatmaps=ia.HeatmapsOnImage(
            np.zeros((1, 1, 1), dtype=np.float32),
            shape=(1, 1, 1)
        ))
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "HeatmapsOnImage"

        batch = ia.Batch(heatmaps=[])
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "iterable[empty]"

        batch = ia.Batch(heatmaps=[np.zeros((1, 1, 1), dtype=np.float32)])
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "iterable-array[float]"

        batch = ia.Batch(heatmaps=[
            ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                               shape=(1, 1, 1))
        ])
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "iterable-HeatmapsOnImage"

        # --
        # error cases
        # --
        batch = ia.Batch(heatmaps=1)
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        batch = ia.Batch(heatmaps="foo")
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        batch = ia.Batch(heatmaps=np.zeros((1, 1, 1), dtype=np.int32))
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        batch = ia.Batch(heatmaps=[1])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        # wrong class
        batch = ia.Batch(heatmaps=ia.KeypointsOnImage([], shape=(1, 1, 1)))
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        batch = ia.Batch(heatmaps=[[]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        # list of list of Heatmaps, only list of Heatmaps is max
        batch = ia.Batch(
            heatmaps=[
                [ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                                    shape=(1, 1, 1))]
            ]
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

    def test__get_segmentation_maps_unaug_normalization_type(self):
        batch = ia.Batch(segmentation_maps=None)
        ntype = batch._get_segmentation_maps_unaug_normalization_type()
        assert ntype == "None"

        for name, dt in zip(["int", "uint", "bool"],
                            [np.int32, np.uint16, bool]):
            batch = ia.Batch(segmentation_maps=np.zeros((1, 1, 1, 1), dtype=dt))
            ntype = batch._get_segmentation_maps_unaug_normalization_type()
            assert ntype == "array[%s]" % (name,)

        batch = ia.Batch(segmentation_maps=ia.SegmentationMapOnImage(
            np.zeros((1, 1, 1), dtype=np.int32),
            shape=(1, 1, 1),
            nb_classes=1
        ))
        ntype = batch._get_segmentation_maps_unaug_normalization_type()
        assert ntype == "SegmentationMapOnImage"

        batch = ia.Batch(segmentation_maps=[])
        ntype = batch._get_segmentation_maps_unaug_normalization_type()
        assert ntype == "iterable[empty]"

        batch = ia.Batch(
            segmentation_maps=[np.zeros((1, 1, 1), dtype=np.int32)]
        )
        ntype = batch._get_segmentation_maps_unaug_normalization_type()
        assert ntype == "iterable-array[int]"

        batch = ia.Batch(segmentation_maps=[
            ia.SegmentationMapOnImage(np.zeros((1, 1, 1), dtype=np.int32),
                                      shape=(1, 1, 1),
                                      nb_classes=1)
        ])
        ntype = batch._get_segmentation_maps_unaug_normalization_type()
        assert ntype == "iterable-SegmentationMapOnImage"

        # --
        # error cases
        # --
        batch = ia.Batch(segmentation_maps=1)
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

        batch = ia.Batch(segmentation_maps="foo")
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

        batch = ia.Batch(segmentation_maps=[1])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

        # wrong class
        batch = ia.Batch(
            segmentation_maps=ia.KeypointsOnImage([], shape=(1, 1, 1))
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

        batch = ia.Batch(segmentation_maps=[[]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

        # list of list of SegMap, only list of SegMap is max
        batch = ia.Batch(
            segmentation_maps=[
                [ia.SegmentationMapOnImage(
                    np.zeros((1, 1, 1), dtype=np.int32),
                    shape=(1, 1, 1),
                    nb_classes=1)]
            ]
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

    def test__get_keypoints_unaug_normalization_type(self):
        batch = ia.Batch(keypoints=None)
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "None"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(keypoints=np.zeros((1, 5, 2), dtype=dt))
            ntype = batch._get_keypoints_unaug_normalization_type()
            assert ntype == "array[%s]" % (name,)

        batch = ia.Batch(keypoints=(1, 2))
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "(x,y)"

        batch = ia.Batch(keypoints=ia.Keypoint(x=1, y=2))
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "Keypoint"

        batch = ia.Batch(keypoints=ia.KeypointsOnImage(
            [ia.Keypoint(x=1, y=2)], shape=(1, 1, 3)))
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "KeypointsOnImage"

        batch = ia.Batch(keypoints=[])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(keypoints=[np.zeros((5, 2), dtype=dt)])
            ntype = batch._get_keypoints_unaug_normalization_type()
            assert ntype == "iterable-array[%s]" % (name,)

        batch = ia.Batch(keypoints=[(1, 2)])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-(x,y)"

        batch = ia.Batch(keypoints=[ia.Keypoint(x=1, y=2)])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-Keypoint"

        batch = ia.Batch(keypoints=[
            ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3))])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-KeypointsOnImage"

        batch = ia.Batch(keypoints=[[]])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-iterable[empty]"

        batch = ia.Batch(keypoints=[[(1, 2)]])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-iterable-(x,y)"

        batch = ia.Batch(keypoints=[[ia.Keypoint(x=1, y=2)]])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-iterable-Keypoint"

        # --
        # error cases
        # --
        batch = ia.Batch(keypoints=1)
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

        batch = ia.Batch(keypoints="foo")
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

        batch = ia.Batch(keypoints=[1])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

        # wrong class
        batch = ia.Batch(
            keypoints=ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                                         shape=(1, 1, 1))
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

        batch = ia.Batch(keypoints=[[[]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

        # list of list of of list of keypoints,
        # only list of list of keypoints is max
        batch = ia.Batch(keypoints=[[[ia.Keypoint(x=1, y=2)]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

    def test__get_bounding_boxes_unaug_normalization_type(self):
        batch = ia.Batch(bounding_boxes=None)
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "None"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(bounding_boxes=np.zeros((1, 5, 4), dtype=dt))
            ntype = batch._get_bounding_boxes_unaug_normalization_type()
            assert ntype == "array[%s]" % (name,)

        batch = ia.Batch(bounding_boxes=(1, 2, 3, 4))
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "(x1,y1,x2,y2)"

        batch = ia.Batch(bounding_boxes=ia.BoundingBox(x1=1, y1=2, x2=3, y2=4))
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "BoundingBox"

        batch = ia.Batch(bounding_boxes=ia.BoundingBoxesOnImage(
            [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)], shape=(1, 1, 3)))
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "BoundingBoxesOnImage"

        batch = ia.Batch(bounding_boxes=[])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(bounding_boxes=[np.zeros((5, 4), dtype=dt)])
            ntype = batch._get_bounding_boxes_unaug_normalization_type()
            assert ntype == "iterable-array[%s]" % (name,)

        batch = ia.Batch(bounding_boxes=[(1, 2, 3, 4)])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-(x1,y1,x2,y2)"

        batch = ia.Batch(bounding_boxes=[
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-BoundingBox"

        batch = ia.Batch(bounding_boxes=[
            ia.BoundingBoxesOnImage([ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                                     shape=(1, 1, 3))])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-BoundingBoxesOnImage"

        batch = ia.Batch(bounding_boxes=[[]])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-iterable[empty]"

        batch = ia.Batch(bounding_boxes=[[(1, 2, 3, 4)]])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-iterable-(x1,y1,x2,y2)"

        batch = ia.Batch(bounding_boxes=[[ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-iterable-BoundingBox"

        # --
        # error cases
        # --
        batch = ia.Batch(bounding_boxes=1)
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

        batch = ia.Batch(bounding_boxes="foo")
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

        batch = ia.Batch(bounding_boxes=[1])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

        # wrong class
        batch = ia.Batch(
            bounding_boxes=ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32),
                shape=(1, 1, 1))
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

        batch = ia.Batch(bounding_boxes=[[[]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

        # list of list of of list of bounding boxes,
        # only list of list of bounding boxes is max
        batch = ia.Batch(bounding_boxes=[[[
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

    def test__get_polygons_unaug_normalization_type(self):
        points = [(0, 0), (10, 0), (10, 10)]

        batch = ia.Batch(polygons=None)
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "None"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(polygons=np.zeros((1, 2, 5, 2), dtype=dt))
            ntype = batch._get_polygons_unaug_normalization_type()
            assert ntype == "array[%s]" % (name,)

        batch = ia.Batch(polygons=ia.Polygon(points))
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "Polygon"

        batch = ia.Batch(polygons=ia.PolygonsOnImage(
            [ia.Polygon(points)], shape=(1, 1, 3)))
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "PolygonsOnImage"

        batch = ia.Batch(polygons=[])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(polygons=[np.zeros((5, 4), dtype=dt)])
            ntype = batch._get_polygons_unaug_normalization_type()
            assert ntype == "iterable-array[%s]" % (name,)

        batch = ia.Batch(polygons=points)
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-(x,y)"

        batch = ia.Batch(polygons=[ia.Keypoint(x=x, y=y) for x, y in points])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-Keypoint"

        batch = ia.Batch(polygons=[
            ia.Polygon(points)])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-Polygon"

        batch = ia.Batch(polygons=[
            ia.PolygonsOnImage([ia.Polygon(points)],
                               shape=(1, 1, 3))])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-PolygonsOnImage"

        batch = ia.Batch(polygons=[[]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(polygons=[[np.zeros((5, 4), dtype=dt)]])
            ntype = batch._get_polygons_unaug_normalization_type()
            assert ntype == "iterable-iterable-array[%s]" % (name,)

        batch = ia.Batch(polygons=[points])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-(x,y)"

        batch = ia.Batch(polygons=[[
            ia.Keypoint(x=x, y=y) for x, y in points
        ]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-Keypoint"

        batch = ia.Batch(polygons=[[ia.Polygon(points)]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-Polygon"

        batch = ia.Batch(polygons=[[[]]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-iterable[empty]"

        batch = ia.Batch(polygons=[[points]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-iterable-(x,y)"

        batch = ia.Batch(polygons=[[[
            ia.Keypoint(x=x, y=y) for x, y in points
        ]]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-iterable-Keypoint"

        # --
        # error cases
        # --
        batch = ia.Batch(polygons=1)
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

        batch = ia.Batch(polygons="foo")
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

        batch = ia.Batch(polygons=[1])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

        # wrong class
        batch = ia.Batch(
            polygons=ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32),
                shape=(1, 1, 1))
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

        batch = ia.Batch(polygons=[[[[]]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

        # list of list of of list of polygons,
        # only list of list of polygons is max
        batch = ia.Batch(polygons=[[[
            ia.Polygon(points)]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

    def test__find_first_nonempty(self):
        # None
        observed = ia.Batch._find_first_nonempty(None)
        assert observed[0] is None
        assert observed[1] is True
        assert len(observed[2]) == 0

        # None with parents
        observed = ia.Batch._find_first_nonempty(None, parents=["foo"])
        assert observed[0] is None
        assert observed[1] is True
        assert len(observed[2]) == 1
        assert observed[2][0] == "foo"

        # array
        observed = ia.Batch._find_first_nonempty(np.zeros((4, 4, 3)))
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 0

        # int
        observed = ia.Batch._find_first_nonempty(0)
        assert observed[0] == 0
        assert observed[1] is True
        assert len(observed[2]) == 0

        # str
        observed = ia.Batch._find_first_nonempty("foo")
        assert observed[0] == "foo"
        assert observed[1] is True
        assert len(observed[2]) == 0

        # empty list
        observed = ia.Batch._find_first_nonempty([])
        assert observed[0] is None
        assert observed[1] is False
        assert len(observed[2]) == 0

        # empty list of empty lists
        observed = ia.Batch._find_first_nonempty([[], [], []])
        assert observed[0] is None
        assert observed[1] is False
        assert len(observed[2]) == 1

        # empty list of empty lists of empty lists
        observed = ia.Batch._find_first_nonempty([[], [[]], []])
        assert observed[0] is None
        assert observed[1] is False
        assert len(observed[2]) == 2

        # list of None
        observed = ia.Batch._find_first_nonempty([None, None])
        assert observed[0] is None
        assert observed[1] is True
        assert len(observed[2]) == 1

        # list of array
        observed = ia.Batch._find_first_nonempty([
            np.zeros((4, 4, 3)), np.zeros((5, 5, 3))])
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 1

        # list of list of array
        observed = ia.Batch._find_first_nonempty(
            [[np.zeros((4, 4, 3))], [np.zeros((5, 5, 3))]]
        )
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 2

        # list of tuple of array
        observed = ia.Batch._find_first_nonempty(
            [
                (
                    np.zeros((4, 4, 3)), np.zeros((5, 5, 3))
                ), (
                    np.zeros((6, 6, 3)), np.zeros((7, 7, 3))
                )
            ]
        )
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 2

    def test__nonempty_info_to_type_str(self):
        ntype = ia.Batch._nonempty_info_to_type_str(None, True, [])
        assert ntype == "None"

        ntype = ia.Batch._nonempty_info_to_type_str(None, False, [])
        assert ntype == "iterable[empty]"

        ntype = ia.Batch._nonempty_info_to_type_str(None, False, [[]])
        assert ntype == "iterable-iterable[empty]"

        ntype = ia.Batch._nonempty_info_to_type_str(None, False, [[], []])
        assert ntype == "iterable-iterable-iterable[empty]"

        ntype = ia.Batch._nonempty_info_to_type_str(None, False, [tuple(), []])
        assert ntype == "iterable-iterable-iterable[empty]"

        ntype = ia.Batch._nonempty_info_to_type_str(1, True, [tuple([1, 2])],
                                                    tuple_size=2)
        assert ntype == "(x,y)"

        ntype = ia.Batch._nonempty_info_to_type_str(1, True, [[], tuple([1, 2])],
                                                    tuple_size=2)
        assert ntype == "iterable-(x,y)"

        ntype = ia.Batch._nonempty_info_to_type_str(1, True, [tuple([1, 2, 3, 4])],
                                                    tuple_size=4)
        assert ntype == "(x1,y1,x2,y2)"

        ntype = ia.Batch._nonempty_info_to_type_str(1, True, [[], tuple([1, 2, 3, 4])],
                                                    tuple_size=4)
        assert ntype == "iterable-(x1,y1,x2,y2)"

        with self.assertRaises(AssertionError):
            ntype = ia.Batch._nonempty_info_to_type_str(1, True, [tuple([1, 2, 3])],
                                                        tuple_size=2)
            assert ntype == "(x1,y1,x2,y2)"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.uint8), True, [])
        assert ntype == "array[uint]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.float32), True, [])
        assert ntype == "array[float]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.int32), True, [])
        assert ntype == "array[int]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=bool), True, [])
        assert ntype == "array[bool]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.dtype("complex")), True, [])
        assert ntype == "array[c]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.uint8), True, [[]])
        assert ntype == "iterable-array[uint]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.uint8), True, [[], []])
        assert ntype == "iterable-iterable-array[uint]"

        cls_names = ["Keypoint", "KeypointsOnImage",
                     "BoundingBox", "BoundingBoxesOnImage",
                     "Polygon", "PolygonsOnImage",
                     "HeatmapsOnImage", "SegmentationMapOnImage"]
        clss = [
            ia.Keypoint(x=1, y=1),
            ia.KeypointsOnImage([], shape=(1, 1, 3)),
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
            ia.BoundingBoxesOnImage([], shape=(1, 1, 3)),
            ia.Polygon([(1, 1), (1, 2), (2, 2)]),
            ia.PolygonsOnImage([], shape=(1,)),
            ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                               shape=(1, 1, 3)),
            ia.SegmentationMapOnImage(np.zeros((1, 1, 1), dtype=np.int32),
                                      shape=(1, 1, 3), nb_classes=1)
        ]
        for cls_name, cls in zip(cls_names, clss):
            ntype = ia.Batch._nonempty_info_to_type_str(
                cls, True, [])
            assert ntype == cls_name

            ntype = ia.Batch._nonempty_info_to_type_str(
                cls, True, [[]])
            assert ntype == "iterable-%s" % (cls_name,)

            ntype = ia.Batch._nonempty_info_to_type_str(
                cls, True, [[], tuple()])
            assert ntype == "iterable-iterable-%s" % (cls_name,)

    def test_property_warnings(self):
        batch = ia.Batch()
        # self.assertWarns does not exist in py2.7
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            _ = batch.images
            assert len(caught_warnings) == 1
            assert "is deprecated" in str(caught_warnings[-1].message)

            _ = batch.heatmaps
            assert len(caught_warnings) == 2
            assert "is deprecated" in str(caught_warnings[-1].message)

            _ = batch.segmentation_maps
            assert len(caught_warnings) == 3
            assert "is deprecated" in str(caught_warnings[-1].message)

            _ = batch.keypoints
            assert len(caught_warnings) == 4
            assert "is deprecated" in str(caught_warnings[-1].message)

            _ = batch.bounding_boxes
            assert len(caught_warnings) == 5
            assert "is deprecated" in str(caught_warnings[-1].message)

    def test_deepcopy(self):
        batch = ia.Batch()
        observed = batch.deepcopy()
        keys = list(observed.__dict__.keys())
        assert len(keys) >= 12
        for attr_name in keys:
            assert getattr(observed, attr_name) is None

        batch = ia.Batch(images=np.zeros((1, 1, 3), dtype=np.uint8))
        observed = batch.deepcopy()
        for attr_name in observed.__dict__.keys():
            if attr_name != "images_unaug":
                assert getattr(observed, attr_name) is None
        assert ia.is_np_array(observed.images_unaug)

        batch = ia.Batch(
            images=np.zeros((1, 1, 3), dtype=np.uint8),
            heatmaps=[
                ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                                   shape=(4, 4, 3))
            ],
            segmentation_maps=[
                ia.SegmentationMapOnImage(np.zeros((1, 1), dtype=np.int32),
                                          shape=(5, 5, 3),
                                          nb_classes=20)
            ],
            keypoints=[
                ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(6, 6, 3))
            ],
            bounding_boxes=[
                ia.BoundingBoxesOnImage([
                    ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                    shape=(7, 7, 3))
            ],
            polygons=[
                ia.PolygonsOnImage([
                    ia.Polygon([(0, 0), (10, 0), (10, 10)])
                ], shape=(100, 100, 3))
            ],
            data={"test": 123, "foo": "bar", "test2": [1, 2, 3]}
        )
        observed = batch.deepcopy()
        for attr_name in observed.__dict__.keys():
            if "_unaug" not in attr_name and attr_name != "data":
                assert getattr(observed, attr_name) is None

        assert ia.is_np_array(observed.images_unaug)
        assert observed.images_unaug.shape == (1, 1, 3)
        assert isinstance(observed.heatmaps_unaug[0], ia.HeatmapsOnImage)
        assert isinstance(observed.segmentation_maps_unaug[0],
                          ia.SegmentationMapOnImage)
        assert isinstance(observed.keypoints_unaug[0], ia.KeypointsOnImage)
        assert isinstance(observed.bounding_boxes_unaug[0],
                          ia.BoundingBoxesOnImage)
        assert isinstance(observed.polygons_unaug[0], ia.PolygonsOnImage)
        assert isinstance(observed.data, dict)

        assert observed.heatmaps_unaug[0].shape == (4, 4, 3)
        assert observed.segmentation_maps_unaug[0].shape == (5, 5, 3)
        assert observed.keypoints_unaug[0].shape == (6, 6, 3)
        assert observed.bounding_boxes_unaug[0].shape == (7, 7, 3)
        assert observed.polygons_unaug[0].shape == (100, 100, 3)

        assert observed.heatmaps_unaug[0].arr_0to1.shape == (1, 1, 1)
        assert observed.segmentation_maps_unaug[0].arr.shape == (1, 1, 20)
        assert observed.keypoints_unaug[0].keypoints[0].x == 1
        assert observed.keypoints_unaug[0].keypoints[0].y == 2
        assert observed.bounding_boxes_unaug[0].bounding_boxes[0].x1 == 1
        assert observed.bounding_boxes_unaug[0].bounding_boxes[0].y1 == 2
        assert observed.bounding_boxes_unaug[0].bounding_boxes[0].x2 == 3
        assert observed.bounding_boxes_unaug[0].bounding_boxes[0].y2 == 4
        assert observed.polygons_unaug[0].polygons[0].exterior[0, 0] == 0
        assert observed.polygons_unaug[0].polygons[0].exterior[0, 1] == 0
        assert observed.polygons_unaug[0].polygons[0].exterior[1, 0] == 10
        assert observed.polygons_unaug[0].polygons[0].exterior[1, 1] == 0
        assert observed.polygons_unaug[0].polygons[0].exterior[2, 0] == 10
        assert observed.polygons_unaug[0].polygons[0].exterior[2, 1] == 10

        assert observed.data["test"] == 123
        assert observed.data["foo"] == "bar"
        assert observed.data["test2"] == [1, 2, 3]


if __name__ == "__main__":
    main()
