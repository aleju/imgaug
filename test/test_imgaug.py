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
    test_classes_and_functions_marked_deprecated()

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


def test_classes_and_functions_marked_deprecated():
    import imgaug.imgaug as iia

    # class
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        _kp = iia.Keypoint(x=1, y=2)
        assert len(caught_warnings) == 1
        assert "is deprecated" in str(caught_warnings[-1].message)

    # function
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        _result = iia.compute_geometric_median(np.float32([[0, 0]]))
        assert len(caught_warnings) == 1
        assert "is deprecated" in str(caught_warnings[-1].message)

    # no deprecated warning for calls to imgaug.<name>
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        _kp = ia.Keypoint(x=1, y=2)
        assert len(caught_warnings) == 0


if __name__ == "__main__":
    main()
