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
import shapely
import shapely.geometry

import imgaug as ia
from imgaug.imgaug import (
    _quokka_normalize_extract, _compute_resized_shape,
    _convert_points_to_shapely_line_string, _interpolate_point_pair, _interpolate_point_pair,
    _interpolate_points, _interpolate_points_by_max_distance,
    _ConcavePolygonRecoverer
)
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
    test_Keypoint()
    test_KeypointsOnImage()
    test_BoundingBox()
    test_BoundingBoxesOnImage()
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
    test_Polygon___init__()
    test_Polygon_xx()
    test_Polygon_yy()
    test_Polygon_xx_int()
    test_Polygon_yy_int()
    test_Polygon_is_valid()
    test_Polygon_area()
    test_Polygon_project()
    test_Polygon_find_closest_point_idx()
    test_Polygon__compute_inside_image_point_mask()
    test_Polygon_is_fully_within_image()
    test_Polygon_is_partly_within_image()
    test_Polygon_is_out_of_image()
    test_Polygon_cut_out_of_image()
    test_Polygon_clip_out_of_image()
    test_Polygon_shift()
    test_Polygon_draw_on_image()
    test_Polygon_extract_from_image()
    test_Polygon_to_shapely_polygon()
    test_Polygon_to_bounding_box()
    test_Polygon_from_shapely()
    test_Polygon_copy()
    test_Polygon_deepcopy()
    test_Polygon___repr__()
    test_Polygon___str__()
    test___convert_points_to_shapely_line_string()
    test__interpolate_point_pair()
    test__interpolate_points()
    test__interpolate_points_by_max_distance()
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


def test_Keypoint():
    eps = 1e-8

    # x/y/x_int/y_int
    kp = ia.Keypoint(y=1, x=2)
    assert kp.y == 1
    assert kp.x == 2
    assert kp.y_int == 1
    assert kp.x_int == 2
    kp = ia.Keypoint(y=1.1, x=2.7)
    assert 1.1 - eps < kp.y < 1.1 + eps
    assert 2.7 - eps < kp.x < 2.7 + eps
    assert kp.y_int == 1
    assert kp.x_int == 3

    # project
    kp = ia.Keypoint(y=1, x=2)
    kp2 = kp.project((10, 10), (10, 10))
    assert kp2.y == 1
    assert kp2.x == 2
    kp2 = kp.project((10, 10), (20, 10))
    assert kp2.y == 2
    assert kp2.x == 2
    kp2 = kp.project((10, 10), (10, 20))
    assert kp2.y == 1
    assert kp2.x == 4
    kp2 = kp.project((10, 10), (20, 20))
    assert kp2.y == 2
    assert kp2.x == 4

    # shift
    kp = ia.Keypoint(y=1, x=2)
    kp2 = kp.shift(y=1)
    assert kp2.y == 2
    assert kp2.x == 2
    kp2 = kp.shift(y=-1)
    assert kp2.y == 0
    assert kp2.x == 2
    kp2 = kp.shift(x=1)
    assert kp2.y == 1
    assert kp2.x == 3
    kp2 = kp.shift(x=-1)
    assert kp2.y == 1
    assert kp2.x == 1
    kp2 = kp.shift(y=1, x=2)
    assert kp2.y == 2
    assert kp2.x == 4

    # generate_similar_points_manhattan
    kp = ia.Keypoint(y=4, x=5)
    kps_manhatten = kp.generate_similar_points_manhattan(0, 1.0, return_array=False)
    assert len(kps_manhatten) == 1
    assert kps_manhatten[0].y == 4
    assert kps_manhatten[0].x == 5

    kps_manhatten = kp.generate_similar_points_manhattan(1, 1.0, return_array=False)
    assert len(kps_manhatten) == 5
    expected = [(4, 5), (3, 5), (4, 6), (5, 5), (4, 4)]
    for y, x in expected:
        assert any([np.allclose([y, x], [kp_manhatten.y, kp_manhatten.x]) for kp_manhatten in kps_manhatten])

    kps_manhatten = kp.generate_similar_points_manhattan(1, 1.0, return_array=True)
    assert kps_manhatten.shape == (5, 2)
    expected = [(4, 5), (3, 5), (4, 6), (5, 5), (4, 4)]
    for y, x in expected:
        assert any([np.allclose([y, x], [kp_manhatten_y, kp_manhatten_x])
                    for kp_manhatten_x, kp_manhatten_y in kps_manhatten])

    # __repr__ / __str_
    kp = ia.Keypoint(y=1, x=2)
    assert kp.__repr__() == kp.__str__() == "Keypoint(x=2.00000000, y=1.00000000)"
    kp = ia.Keypoint(y=1.2, x=2.7)
    assert kp.__repr__() == kp.__str__() == "Keypoint(x=2.70000000, y=1.20000000)"


def test_KeypointsOnImage():
    eps = 1e-8

    kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]

    # height/width
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(10, 20, 3))
    assert kpi.height == 10
    assert kpi.width == 20

    # image instead of shape
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=np.zeros((10, 20, 3), dtype=np.uint8))
    assert kpi.shape == (10, 20, 3)

    # on()
    kpi2 = kpi.on((10, 20, 3))
    assert all([kp_i.x == kp_j.x and kp_i.y == kp_j.y for kp_i, kp_j in zip(kpi.keypoints, kpi2.keypoints)])

    kpi2 = kpi.on((20, 40, 3))
    assert kpi2.keypoints[0].x == 2
    assert kpi2.keypoints[0].y == 4
    assert kpi2.keypoints[1].x == 6
    assert kpi2.keypoints[1].y == 8

    kpi2 = kpi.on(np.zeros((20, 40, 3), dtype=np.uint8))
    assert kpi2.keypoints[0].x == 2
    assert kpi2.keypoints[0].y == 4
    assert kpi2.keypoints[1].x == 6
    assert kpi2.keypoints[1].y == 8

    # draw_on_image
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    kps_mask[2, 1] = 1
    kps_mask[4, 3] = 1
    image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [0, 255, 0])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=3, copy=True, raise_if_out_of_image=False)
    kps_mask_size3 = np.copy(kps_mask)
    kps_mask_size3[2-1:2+1+1, 1-1:1+1+1] = 1
    kps_mask_size3[4-1:4+1+1, 3-1:3+1+1] = 1
    assert np.all(image_kps[kps_mask_size3] == [0, 255, 0])
    assert np.all(image_kps[~kps_mask_size3] == [10, 10, 10])

    image_kps = kpi.draw_on_image(image, color=[0, 0, 255], size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [0, 0, 255])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    image_kps = kpi.draw_on_image(image, color=255, size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [255, 255, 255])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    image2 = np.copy(image)
    image_kps = kpi.draw_on_image(image2, color=[0, 255, 0], size=1, copy=False, raise_if_out_of_image=False)
    assert np.all(image2 == image_kps)
    assert np.all(image_kps[kps_mask] == [0, 255, 0])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])
    assert np.all(image2[kps_mask] == [0, 255, 0])
    assert np.all(image2[~kps_mask] == [10, 10, 10])

    kpi = ia.KeypointsOnImage(keypoints=kps + [ia.Keypoint(x=100, y=100)], shape=(5, 5, 3))
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    kps_mask[2, 1] = 1
    kps_mask[4, 3] = 1
    image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [0, 255, 0])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    kpi = ia.KeypointsOnImage(keypoints=kps + [ia.Keypoint(x=100, y=100)], shape=(5, 5, 3))
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    got_exception = False
    try:
        image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=1, copy=True, raise_if_out_of_image=True)
        assert np.all(image_kps[kps_mask] == [0, 255, 0])
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])
    except Exception:
        got_exception = True
    assert got_exception

    kpi = ia.KeypointsOnImage(keypoints=kps + [ia.Keypoint(x=5, y=5)], shape=(5, 5, 3))
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    kps_mask[2, 1] = 1
    kps_mask[4, 3] = 1
    image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [0, 255, 0])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    got_exception = False
    try:
        image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=1, copy=True, raise_if_out_of_image=True)
        assert np.all(image_kps[kps_mask] == [0, 255, 0])
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])
    except Exception:
        got_exception = True
    assert got_exception

    # shift
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    kpi2 = kpi.shift(x=0, y=0)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y

    kpi2 = kpi.shift(x=1)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x + 1
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x + 1
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y

    kpi2 = kpi.shift(x=-1)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x - 1
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x - 1
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y

    kpi2 = kpi.shift(y=1)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y + 1
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y + 1

    kpi2 = kpi.shift(y=-1)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y - 1
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y - 1

    kpi2 = kpi.shift(x=1, y=2)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x + 1
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y + 2
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x + 1
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y + 2

    # get_coords_array
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    observed = kpi.get_coords_array()
    expected = np.float32([
        [1, 2],
        [3, 4]
    ])
    assert np.allclose(observed, expected)

    # from_coords_array
    arr = np.float32([
        [1, 2],
        [3, 4]
    ])
    kpi = ia.KeypointsOnImage.from_coords_array(arr, shape=(5, 5, 3))
    assert 1 - eps < kpi.keypoints[0].x < 1 + eps
    assert 2 - eps < kpi.keypoints[0].y < 2 + eps
    assert 3 - eps < kpi.keypoints[1].x < 3 + eps
    assert 4 - eps < kpi.keypoints[1].y < 4 + eps

    # to_keypoint_image
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    image = kpi.to_keypoint_image(size=1)
    image_size3 = kpi.to_keypoint_image(size=3)
    kps_mask = np.zeros((5, 5, 2), dtype=np.bool)
    kps_mask[2, 1, 0] = 1
    kps_mask[4, 3, 1] = 1
    kps_mask_size3 = np.zeros_like(kps_mask)
    kps_mask_size3[2-1:2+1+1, 1-1:1+1+1, 0] = 1
    kps_mask_size3[4-1:4+1+1, 3-1:3+1+1, 1] = 1
    assert np.all(image[kps_mask] == 255)
    assert np.all(image[~kps_mask] == 0)
    assert np.all(image_size3[kps_mask] == 255)
    assert np.all(image_size3[kps_mask_size3] >= 128)
    assert np.all(image_size3[~kps_mask_size3] == 0)

    # from_keypoint_image()
    kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
    kps_image[2, 1, 0] = 255
    kps_image[4, 3, 1] = 255
    kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, nb_channels=3)
    assert kpi2.shape == (5, 5, 3)
    assert len(kpi2.keypoints) == 2
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[0].x == 1
    assert kpi2.keypoints[1].y == 4
    assert kpi2.keypoints[1].x == 3

    kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
    kps_image[2, 1, 0] = 255
    kps_image[4, 3, 1] = 10
    kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords={"x": -1, "y": -2}, threshold=20,
                                                   nb_channels=3)
    assert kpi2.shape == (5, 5, 3)
    assert len(kpi2.keypoints) == 2
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[0].x == 1
    assert kpi2.keypoints[1].y == -2
    assert kpi2.keypoints[1].x == -1

    kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
    kps_image[2, 1, 0] = 255
    kps_image[4, 3, 1] = 10
    kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords=(-1, -2), threshold=20,
                                                   nb_channels=3)
    assert kpi2.shape == (5, 5, 3)
    assert len(kpi2.keypoints) == 2
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[0].x == 1
    assert kpi2.keypoints[1].y == -2
    assert kpi2.keypoints[1].x == -1

    kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
    kps_image[2, 1, 0] = 255
    kps_image[4, 3, 1] = 10
    kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords=None, threshold=20, nb_channels=3)
    assert kpi2.shape == (5, 5, 3)
    assert len(kpi2.keypoints) == 1
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[0].x == 1

    got_exception = False
    try:
        kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
        kps_image[2, 1, 0] = 255
        kps_image[4, 3, 1] = 10
        _ = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords="exception-please", threshold=20,
                                                    nb_channels=3)
    except Exception as exc:
        assert "Expected if_not_found_coords to be" in str(exc)
        got_exception = True
    assert got_exception

    # to_distance_maps()
    kpi = ia.KeypointsOnImage(keypoints=[ia.Keypoint(x=2, y=3)], shape=(5, 5, 3))
    distance_map = kpi.to_distance_maps()
    expected_xx = np.float32([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]
    ])
    expected_yy = np.float32([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4]
    ])
    expected = np.sqrt((expected_xx - 2)**2 + (expected_yy - 3)**2)
    assert distance_map.shape == (5, 5, 1)
    assert np.allclose(distance_map, expected[..., np.newaxis])

    distance_map_inv = kpi.to_distance_maps(inverted=True)
    expected_inv = np.divide(np.ones_like(expected), expected+1)
    assert np.allclose(distance_map_inv, expected_inv[..., np.newaxis])

    # to_distance_maps() with two keypoints
    # positions on (4, 4) map (X=position, 1=KP 1 is closest, 2=KP 2 is closest, B=close to both)
    # [1, X, 1, 1]
    # [1, 1, 1, B]
    # [B, 2, 2, 2]
    # [2, 2, X, 2]
    # this test could have been done a bit better by simply splitting the distance maps, one per keypoint, considering
    # the function returns one distance map per keypoint
    kpi = ia.KeypointsOnImage(keypoints=[ia.Keypoint(x=2, y=3), ia.Keypoint(x=1, y=0)], shape=(4, 4, 3))
    expected = np.float32([
        [(0-1)**2 + (0-0)**2, (1-1)**2 + (0-0)**2, (2-1)**2 + (0-0)**2, (3-1)**2 + (0-0)**2],
        [(0-1)**2 + (1-0)**2, (1-1)**2 + (1-0)**2, (2-1)**2 + (1-0)**2, (3-1)**2 + (1-0)**2],
        [(0-1)**2 + (2-0)**2, (1-2)**2 + (2-3)**2, (2-2)**2 + (2-3)**2, (3-2)**2 + (2-3)**2],
        [(0-2)**2 + (3-3)**2, (1-2)**2 + (3-3)**2, (2-2)**2 + (3-3)**2, (3-2)**2 + (3-3)**2],
    ])
    distance_map = kpi.to_distance_maps()
    expected = np.sqrt(expected)
    assert np.allclose(np.min(distance_map, axis=2), expected)

    distance_map_inv = kpi.to_distance_maps(inverted=True)
    expected_inv = np.divide(np.ones_like(expected), expected+1)
    assert np.allclose(np.max(distance_map_inv, axis=2), expected_inv)

    # from_distance_maps()
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
    distance_maps = np.concatenate([distance_map1[..., np.newaxis], distance_map2[..., np.newaxis]], axis=2)
    kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps, nb_channels=4)
    assert len(kpi.keypoints) == 2
    assert kpi.keypoints[0].x == 2
    assert kpi.keypoints[0].y == 2
    assert kpi.keypoints[1].x == 4
    assert kpi.keypoints[1].y == 2
    assert kpi.shape == (4, 5, 4)

    kpi = ia.KeypointsOnImage.from_distance_maps(np.divide(np.ones_like(distance_maps), distance_maps+1),
                                                 inverted=True)
    assert len(kpi.keypoints) == 2
    assert kpi.keypoints[0].x == 2
    assert kpi.keypoints[0].y == 2
    assert kpi.keypoints[1].x == 4
    assert kpi.keypoints[1].y == 2
    assert kpi.shape == (4, 5)

    kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps, if_not_found_coords=(1, 1), threshold=0.09)
    assert len(kpi.keypoints) == 2
    assert kpi.keypoints[0].x == 2
    assert kpi.keypoints[0].y == 2
    assert kpi.keypoints[1].x == 1
    assert kpi.keypoints[1].y == 1
    assert kpi.shape == (4, 5)

    kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps, if_not_found_coords={"x": 1, "y": 2}, threshold=0.09)
    assert len(kpi.keypoints) == 2
    assert kpi.keypoints[0].x == 2
    assert kpi.keypoints[0].y == 2
    assert kpi.keypoints[1].x == 1
    assert kpi.keypoints[1].y == 2
    assert kpi.shape == (4, 5)

    kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps, if_not_found_coords=None, threshold=0.09)
    assert len(kpi.keypoints) == 1
    assert kpi.keypoints[0].x == 2
    assert kpi.keypoints[0].y == 2
    assert kpi.shape == (4, 5)

    got_exception = False
    try:
        _ = ia.KeypointsOnImage.from_distance_maps(distance_maps, if_not_found_coords=False, threshold=0.09)
    except Exception as exc:
        assert "Expected if_not_found_coords to be" in str(exc)
        got_exception = True
    assert got_exception

    # copy()
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

    # deepcopy()
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

    # repr/str
    kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    expected = "KeypointsOnImage([Keypoint(x=1.00000000, y=2.00000000), Keypoint(x=3.00000000, y=4.00000000)], " \
               + "shape=(5, 5, 3))"
    assert kpi.__repr__() == kpi.__str__() == expected


def test_BoundingBox():
    eps = 1e-8

    # properties with ints
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.y1_int == 10
    assert bb.x1_int == 20
    assert bb.y2_int == 30
    assert bb.x2_int == 40
    assert bb.width == 40 - 20
    assert bb.height == 30 - 10
    center_x = bb.x1 + (bb.x2 - bb.x1)/2
    center_y = bb.y1 + (bb.y2 - bb.y1)/2
    assert center_x - eps < bb.center_x < center_x + eps
    assert center_y - eps < bb.center_y < center_y + eps

    # wrong order of y1/y2, x1/x2
    bb = ia.BoundingBox(y1=30, x1=40, y2=10, x2=20, label=None)
    assert bb.y1_int == 10
    assert bb.x1_int == 20
    assert bb.y2_int == 30
    assert bb.x2_int == 40

    # properties with floats
    bb = ia.BoundingBox(y1=10.1, x1=20.1, y2=30.9, x2=40.9, label=None)
    assert bb.y1_int == 10
    assert bb.x1_int == 20
    assert bb.y2_int == 31
    assert bb.x2_int == 41
    assert bb.width == 40.9 - 20.1
    assert bb.height == 30.9 - 10.1
    center_x = bb.x1 + (bb.x2 - bb.x1)/2
    center_y = bb.y1 + (bb.y2 - bb.y1)/2
    assert center_x - eps < bb.center_x < center_x + eps
    assert center_y - eps < bb.center_y < center_y + eps

    # area
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.area == (30-10) * (40-20)

    # contains
    bb = ia.BoundingBox(y1=1, x1=2, y2=1+4, x2=2+5, label=None)
    assert bb.contains(ia.Keypoint(x=2.5, y=1.5)) is True
    assert bb.contains(ia.Keypoint(x=2, y=1)) is True
    assert bb.contains(ia.Keypoint(x=0, y=0)) is False

    # project
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = bb.project((10, 10), (10, 10))
    assert 10 - eps < bb2.y1 < 10 + eps
    assert 20 - eps < bb2.x1 < 20 + eps
    assert 30 - eps < bb2.y2 < 30 + eps
    assert 40 - eps < bb2.x2 < 40 + eps

    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = bb.project((10, 10), (20, 20))
    assert 10*2 - eps < bb2.y1 < 10*2 + eps
    assert 20*2 - eps < bb2.x1 < 20*2 + eps
    assert 30*2 - eps < bb2.y2 < 30*2 + eps
    assert 40*2 - eps < bb2.x2 < 40*2 + eps

    bb2 = bb.project((10, 10), (5, 5))
    assert 10*0.5 - eps < bb2.y1 < 10*0.5 + eps
    assert 20*0.5 - eps < bb2.x1 < 20*0.5 + eps
    assert 30*0.5 - eps < bb2.y2 < 30*0.5 + eps
    assert 40*0.5 - eps < bb2.x2 < 40*0.5 + eps

    bb2 = bb.project((10, 10), (10, 20))
    assert 10*1 - eps < bb2.y1 < 10*1 + eps
    assert 20*2 - eps < bb2.x1 < 20*2 + eps
    assert 30*1 - eps < bb2.y2 < 30*1 + eps
    assert 40*2 - eps < bb2.x2 < 40*2 + eps

    bb2 = bb.project((10, 10), (20, 10))
    assert 10*2 - eps < bb2.y1 < 10*2 + eps
    assert 20*1 - eps < bb2.x1 < 20*1 + eps
    assert 30*2 - eps < bb2.y2 < 30*2 + eps
    assert 40*1 - eps < bb2.x2 < 40*1 + eps

    # extend
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = bb.extend(all_sides=1)
    assert bb2.y1 == 10-1
    assert bb2.y2 == 30+1
    assert bb2.x1 == 20-1
    assert bb2.x2 == 40+1

    bb2 = bb.extend(all_sides=-1)
    assert bb2.y1 == 10-(-1)
    assert bb2.y2 == 30+(-1)
    assert bb2.x1 == 20-(-1)
    assert bb2.x2 == 40+(-1)

    bb2 = bb.extend(top=1)
    assert bb2.y1 == 10-1
    assert bb2.y2 == 30+0
    assert bb2.x1 == 20-0
    assert bb2.x2 == 40+0

    bb2 = bb.extend(right=1)
    assert bb2.y1 == 10-0
    assert bb2.y2 == 30+0
    assert bb2.x1 == 20-0
    assert bb2.x2 == 40+1

    bb2 = bb.extend(bottom=1)
    assert bb2.y1 == 10-0
    assert bb2.y2 == 30+1
    assert bb2.x1 == 20-0
    assert bb2.x2 == 40+0

    bb2 = bb.extend(left=1)
    assert bb2.y1 == 10-0
    assert bb2.y2 == 30+0
    assert bb2.x1 == 20-1
    assert bb2.x2 == 40+0

    # intersection
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=10, x1=39, y2=30, x2=59, label=None)
    bb_inter = bb1.intersection(bb2)
    assert bb_inter.x1 == 39
    assert bb_inter.x2 == 40
    assert bb_inter.y1 == 10
    assert bb_inter.y2 == 30

    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=10, x1=41, y2=30, x2=61, label=None)
    bb_inter = bb1.intersection(bb2, default=False)
    assert bb_inter is False

    # union
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=10, x1=39, y2=30, x2=59, label=None)
    bb_union = bb1.union(bb2)
    assert bb_union.x1 == 20
    assert bb_union.x2 == 59
    assert bb_union.y1 == 10
    assert bb_union.y2 == 30

    # iou
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    iou = bb1.iou(bb2)
    assert 1.0 - eps < iou < 1.0 + eps

    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=10, x1=41, y2=30, x2=61, label=None)
    iou = bb1.iou(bb2)
    assert 0.0 - eps < iou < 0.0 + eps

    bb1 = ia.BoundingBox(y1=10, x1=10, y2=20, x2=20, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=15, y2=25, x2=25, label=None)
    iou = bb1.iou(bb2)
    area_union = 10 * 10 + 10 * 10 - 5 * 5
    area_intersection = 5 * 5
    iou_expected = area_intersection / area_union
    assert iou_expected - eps < iou < iou_expected + eps

    # is_fully_within_image
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.is_fully_within_image((100, 100, 3)) is True
    assert bb.is_fully_within_image((20, 100, 3)) is False
    assert bb.is_fully_within_image((100, 30, 3)) is False
    assert bb.is_fully_within_image((1, 1, 3)) is False

    # is_partly_within_image
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.is_partly_within_image((100, 100, 3)) is True
    assert bb.is_partly_within_image((20, 100, 3)) is True
    assert bb.is_partly_within_image((100, 30, 3)) is True
    assert bb.is_partly_within_image((1, 1, 3)) is False

    # is_out_of_image()
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.is_out_of_image((100, 100, 3), partly=True, fully=True) is False
    assert bb.is_out_of_image((100, 100, 3), partly=False, fully=True) is False
    assert bb.is_out_of_image((100, 100, 3), partly=True, fully=False) is False
    assert bb.is_out_of_image((20, 100, 3), partly=True, fully=True) is True
    assert bb.is_out_of_image((20, 100, 3), partly=False, fully=True) is False
    assert bb.is_out_of_image((20, 100, 3), partly=True, fully=False) is True
    assert bb.is_out_of_image((100, 30, 3), partly=True, fully=True) is True
    assert bb.is_out_of_image((100, 30, 3), partly=False, fully=True) is False
    assert bb.is_out_of_image((100, 30, 3), partly=True, fully=False) is True
    assert bb.is_out_of_image((1, 1, 3), partly=True, fully=True) is True
    assert bb.is_out_of_image((1, 1, 3), partly=False, fully=True) is True
    assert bb.is_out_of_image((1, 1, 3), partly=True, fully=False) is False

    # clip_out_of_image
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb_cut = bb.clip_out_of_image((100, 100, 3))
    eps = np.finfo(np.float32).eps
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert bb_cut.y2 == 30
    assert bb_cut.x2 == 40
    bb_cut = bb.clip_out_of_image(np.zeros((100, 100, 3), dtype=np.uint8))
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert bb_cut.y2 == 30
    assert bb_cut.x2 == 40
    bb_cut = bb.clip_out_of_image((20, 100, 3))
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert 20 - 2*eps < bb_cut.y2 < 20
    assert bb_cut.x2 == 40
    bb_cut = bb.clip_out_of_image((100, 30, 3))
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert bb_cut.y2 == 30
    assert 30 - 2*eps < bb_cut.x2 < 30

    # shift
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb_top = bb.shift(top=0)
    bb_right = bb.shift(right=0)
    bb_bottom = bb.shift(bottom=0)
    bb_left = bb.shift(left=0)
    assert bb_top.y1 == 10
    assert bb_top.x1 == 20
    assert bb_top.y2 == 30
    assert bb_top.x2 == 40
    assert bb_right.y1 == 10
    assert bb_right.x1 == 20
    assert bb_right.y2 == 30
    assert bb_right.x2 == 40
    assert bb_bottom.y1 == 10
    assert bb_bottom.x1 == 20
    assert bb_bottom.y2 == 30
    assert bb_bottom.x2 == 40
    assert bb_left.y1 == 10
    assert bb_left.x1 == 20
    assert bb_left.y2 == 30
    assert bb_left.x2 == 40
    bb_top = bb.shift(top=1)
    bb_right = bb.shift(right=1)
    bb_bottom = bb.shift(bottom=1)
    bb_left = bb.shift(left=1)
    assert bb_top.y1 == 10+1
    assert bb_top.x1 == 20
    assert bb_top.y2 == 30+1
    assert bb_top.x2 == 40
    assert bb_right.y1 == 10
    assert bb_right.x1 == 20-1
    assert bb_right.y2 == 30
    assert bb_right.x2 == 40-1
    assert bb_bottom.y1 == 10-1
    assert bb_bottom.x1 == 20
    assert bb_bottom.y2 == 30-1
    assert bb_bottom.x2 == 40
    assert bb_left.y1 == 10
    assert bb_left.x1 == 20+1
    assert bb_left.y2 == 30
    assert bb_left.x2 == 40+1
    bb_top = bb.shift(top=-1)
    bb_right = bb.shift(right=-1)
    bb_bottom = bb.shift(bottom=-1)
    bb_left = bb.shift(left=-1)
    assert bb_top.y1 == 10-1
    assert bb_top.x1 == 20
    assert bb_top.y2 == 30-1
    assert bb_top.x2 == 40
    assert bb_right.y1 == 10
    assert bb_right.x1 == 20+1
    assert bb_right.y2 == 30
    assert bb_right.x2 == 40+1
    assert bb_bottom.y1 == 10+1
    assert bb_bottom.x1 == 20
    assert bb_bottom.y2 == 30+1
    assert bb_bottom.x2 == 40
    assert bb_left.y1 == 10
    assert bb_left.x1 == 20-1
    assert bb_left.y2 == 30
    assert bb_left.x2 == 40-1
    bb_mix = bb.shift(top=1, bottom=2, left=3, right=4)
    assert bb_mix.y1 == 10+1-2
    assert bb_mix.x1 == 20+3-4
    assert bb_mix.y2 == 30+3-4
    assert bb_mix.x2 == 40+1-2

    # draw_on_image()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    bb = ia.BoundingBox(y1=1, x1=1, y2=3, x2=3, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[1:3+1, 1] = True
    bb_mask[1:3+1, 3] = True
    bb_mask[1, 1:3+1] = True
    bb_mask[3, 1:3+1] = True
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])
    assert np.all(image == 0)

    image_bb = bb.draw_on_image(image, color=[255, 0, 0], alpha=1.0, thickness=1, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 0, 0])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    image_bb = bb.draw_on_image(image, color=128, alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [128, 128, 128])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    image_bb = bb.draw_on_image(image+100, color=[200, 200, 200], alpha=0.5, thickness=1, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [150, 150, 150])
    assert np.all(image_bb[~bb_mask] == [100, 100, 100])

    image_bb = bb.draw_on_image((image+100).astype(np.float32), color=[200, 200, 200], alpha=0.5, thickness=1,
                                copy=True, raise_if_out_of_image=False)
    assert np.sum(np.abs((image_bb - [150, 150, 150])[bb_mask])) < 0.1
    assert np.sum(np.abs((image_bb - [100, 100, 100])[~bb_mask])) < 0.1

    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=False,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])
    assert np.all(image[bb_mask] == [255, 255, 255])
    assert np.all(image[~bb_mask] == [0, 0, 0])

    image = np.zeros_like(image)
    bb = ia.BoundingBox(y1=-1, x1=-1, y2=2, x2=2, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[2, 0:3] = True
    bb_mask[0:3, 2] = True
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    bb = ia.BoundingBox(y1=1, x1=1, y2=3, x2=3, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[0:5, 0:5] = True
    bb_mask[2, 2] = False
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=2, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    bb = ia.BoundingBox(y1=-1, x1=-1, y2=1, x2=1, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[0:1+1, 1] = True
    bb_mask[1, 0:1+1] = True
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    bb = ia.BoundingBox(y1=-1, x1=-1, y2=1, x2=1, label=None)
    got_exception = False
    try:
        _ = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=True,
                             raise_if_out_of_image=True)
    except Exception:
        got_exception = True
    assert got_exception is False

    bb = ia.BoundingBox(y1=-5, x1=-5, y2=-1, x2=-1, label=None)
    got_exception = False
    try:
        _ = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=True,
                             raise_if_out_of_image=True)
    except Exception:
        got_exception = True
    assert got_exception is True

    # extract_from_image()
    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:3, 1:3, :])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10))
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:3, 1:3])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10))
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:3, 1:3])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    image_pad = np.pad(image, ((0, 1), (0, 1), (0, 0)), mode="constant", constant_values=0)
    bb = ia.BoundingBox(y1=8, y2=11, x1=8, x2=11, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image_pad[8:11, 8:11, :])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10))
    image_pad = np.pad(image, ((0, 1), (0, 1)), mode="constant", constant_values=0)
    bb = ia.BoundingBox(y1=8, y2=11, x1=8, x2=11, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image_pad[8:11, 8:11])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    image_pad = np.pad(image, ((1, 0), (1, 0), (0, 0)), mode="constant", constant_values=0)
    bb = ia.BoundingBox(y1=-1, y2=3, x1=-1, x2=4, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image_pad[0:4, 0:5, :])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    bb = ia.BoundingBox(y1=1, y2=1.99999, x1=1, x2=1.99999, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:1+1, 1:1+1, :])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    bb = ia.BoundingBox(y1=1, y2=1, x1=2, x2=4, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:1+1, 2:4, :])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    bb = ia.BoundingBox(y1=1, y2=1, x1=2, x2=2, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:1+1, 2:2+1, :])

    # to_keypoints()
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    kps = bb.to_keypoints()
    assert kps[0].y == 1
    assert kps[0].x == 1
    assert kps[1].y == 1
    assert kps[1].x == 3
    assert kps[2].y == 3
    assert kps[2].x == 3
    assert kps[3].y == 3
    assert kps[3].x == 1

    # copy()
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label="test")
    bb2 = bb.copy()
    assert bb2.y1 == 1
    assert bb2.y2 == 3
    assert bb2.x1 == 1
    assert bb2.x2 == 3
    assert bb2.label == "test"

    bb2 = bb.copy(y1=10, x1=20, y2=30, x2=40, label="test2")
    assert bb2.y1 == 10
    assert bb2.x1 == 20
    assert bb2.y2 == 30
    assert bb2.x2 == 40
    assert bb2.label == "test2"

    # deepcopy()
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=["test"])
    bb2 = bb.deepcopy()
    assert bb2.y1 == 1
    assert bb2.y2 == 3
    assert bb2.x1 == 1
    assert bb2.x2 == 3
    assert bb2.label[0] == "test"

    # BoundingBox_repr()
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    assert bb.__repr__() == "BoundingBox(x1=1.0000, y1=1.0000, x2=3.0000, y2=3.0000, label=None)"

    # test_BoundingBox_str()
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    assert bb.__str__() == "BoundingBox(x1=1.0000, y1=1.0000, x2=3.0000, y2=3.0000, label=None)"


def test_BoundingBoxesOnImage():
    reseed()

    # test height/width
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    assert bbsoi.height == 40
    assert bbsoi.width == 50

    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=np.zeros((40, 50, 3), dtype=np.uint8))
    assert bbsoi.height == 40
    assert bbsoi.width == 50

    # empty
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb], shape=(40, 50, 3))
    assert not bbsoi.empty

    bbsoi = ia.BoundingBoxesOnImage([], shape=(40, 50, 3))
    assert bbsoi.empty

    # on()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=np.zeros((40, 50, 3), dtype=np.uint8))

    bbsoi_projected = bbsoi.on((40, 50))
    assert bbsoi_projected.bounding_boxes[0].y1 == 10
    assert bbsoi_projected.bounding_boxes[0].x1 == 20
    assert bbsoi_projected.bounding_boxes[0].y2 == 30
    assert bbsoi_projected.bounding_boxes[0].x2 == 40
    assert bbsoi_projected.bounding_boxes[1].y1 == 15
    assert bbsoi_projected.bounding_boxes[1].x1 == 25
    assert bbsoi_projected.bounding_boxes[1].y2 == 35
    assert bbsoi_projected.bounding_boxes[1].x2 == 45

    bbsoi_projected = bbsoi.on((40*2, 50*2, 3))
    assert bbsoi_projected.bounding_boxes[0].y1 == 10*2
    assert bbsoi_projected.bounding_boxes[0].x1 == 20*2
    assert bbsoi_projected.bounding_boxes[0].y2 == 30*2
    assert bbsoi_projected.bounding_boxes[0].x2 == 40*2
    assert bbsoi_projected.bounding_boxes[1].y1 == 15*2
    assert bbsoi_projected.bounding_boxes[1].x1 == 25*2
    assert bbsoi_projected.bounding_boxes[1].y2 == 35*2
    assert bbsoi_projected.bounding_boxes[1].x2 == 45*2

    bbsoi_projected = bbsoi.on(np.zeros((40*2, 50*2, 3), dtype=np.uint8))
    assert bbsoi_projected.bounding_boxes[0].y1 == 10*2
    assert bbsoi_projected.bounding_boxes[0].x1 == 20*2
    assert bbsoi_projected.bounding_boxes[0].y2 == 30*2
    assert bbsoi_projected.bounding_boxes[0].x2 == 40*2
    assert bbsoi_projected.bounding_boxes[1].y1 == 15*2
    assert bbsoi_projected.bounding_boxes[1].x1 == 25*2
    assert bbsoi_projected.bounding_boxes[1].y2 == 35*2
    assert bbsoi_projected.bounding_boxes[1].x2 == 45*2

    # from_xyxy_array()
    bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(
        np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0]
        ]),
        shape=(40, 50, 3)
    )
    assert len(bbsoi.bounding_boxes) == 2
    assert np.allclose(bbsoi.bounding_boxes[0].x1, 0.0)
    assert np.allclose(bbsoi.bounding_boxes[0].y1, 0.0)
    assert np.allclose(bbsoi.bounding_boxes[0].x2, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[0].y2, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[1].x1, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[1].y1, 2.0)
    assert np.allclose(bbsoi.bounding_boxes[1].x2, 3.0)
    assert np.allclose(bbsoi.bounding_boxes[1].y2, 4.0)
    assert bbsoi.shape == (40, 50, 3)

    bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(
        np.int32([
            [0, 0, 1, 1],
            [1, 2, 3, 4]
        ]),
        shape=(40, 50, 3)
    )
    assert len(bbsoi.bounding_boxes) == 2
    assert np.allclose(bbsoi.bounding_boxes[0].x1, 0.0)
    assert np.allclose(bbsoi.bounding_boxes[0].y1, 0.0)
    assert np.allclose(bbsoi.bounding_boxes[0].x2, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[0].y2, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[1].x1, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[1].y1, 2.0)
    assert np.allclose(bbsoi.bounding_boxes[1].x2, 3.0)
    assert np.allclose(bbsoi.bounding_boxes[1].y2, 4.0)
    assert bbsoi.shape == (40, 50, 3)

    bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(
        np.zeros((0, 4), dtype=np.float32),
        shape=(40, 50, 3)
    )
    assert len(bbsoi.bounding_boxes) == 0
    assert bbsoi.shape == (40, 50, 3)

    # to_xyxy_array()
    xyxy_arr = np.float32([
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 2.0, 3.0, 4.0]
    ])
    bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy_arr, shape=(40, 50, 3))
    xyxy_arr_out = bbsoi.to_xyxy_array()
    assert np.allclose(xyxy_arr, xyxy_arr_out)
    assert xyxy_arr_out.dtype == np.float32

    xyxy_arr_out = bbsoi.to_xyxy_array(dtype=np.int32)
    assert np.allclose(xyxy_arr.astype(np.int32), xyxy_arr_out)
    assert xyxy_arr_out.dtype == np.int32

    xyxy_arr_out = ia.BoundingBoxesOnImage([], shape=(40, 50, 3)).to_xyxy_array(dtype=np.int32)
    assert xyxy_arr_out.shape == (0, 4)

    # draw_on_image()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    image = bbsoi.draw_on_image(np.zeros(bbsoi.shape, dtype=np.uint8), color=[0, 255, 0], alpha=1.0, thickness=1,
                                copy=True, raise_if_out_of_image=False)
    assert np.all(image[10-1, 20-1, :] == [0, 0, 0])
    assert np.all(image[10-1, 20-0, :] == [0, 0, 0])
    assert np.all(image[10-0, 20-1, :] == [0, 0, 0])
    assert np.all(image[10-0, 20-0, :] == [0, 255, 0])
    assert np.all(image[10+1, 20+1, :] == [0, 0, 0])

    assert np.all(image[30-1, 40-1, :] == [0, 0, 0])
    assert np.all(image[30+1, 40-0, :] == [0, 0, 0])
    assert np.all(image[30+0, 40+1, :] == [0, 0, 0])
    assert np.all(image[30+0, 40+0, :] == [0, 255, 0])
    assert np.all(image[30+1, 40+1, :] == [0, 0, 0])

    assert np.all(image[15-1, 25-1, :] == [0, 0, 0])
    assert np.all(image[15-1, 25-0, :] == [0, 0, 0])
    assert np.all(image[15-0, 25-1, :] == [0, 0, 0])
    assert np.all(image[15-0, 25-0, :] == [0, 255, 0])
    assert np.all(image[15+1, 25+1, :] == [0, 0, 0])

    assert np.all(image[35-1, 45-1, :] == [0, 0, 0])
    assert np.all(image[35+1, 45+0, :] == [0, 0, 0])
    assert np.all(image[35+0, 45+1, :] == [0, 0, 0])
    assert np.all(image[35+0, 45+0, :] == [0, 255, 0])
    assert np.all(image[35+1, 45+1, :] == [0, 0, 0])

    # remove_out_of_image()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bbsoi_slim = bbsoi.remove_out_of_image(fully=True, partly=True)
    assert len(bbsoi_slim.bounding_boxes) == 1
    assert bbsoi_slim.bounding_boxes[0] == bb1

    # clip_out_of_image()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    eps = np.finfo(np.float32).eps
    bbsoi_clip = bbsoi.clip_out_of_image()
    assert len(bbsoi_clip.bounding_boxes) == 2
    assert bbsoi_clip.bounding_boxes[0].y1 == 10
    assert bbsoi_clip.bounding_boxes[0].x1 == 20
    assert bbsoi_clip.bounding_boxes[0].y2 == 30
    assert bbsoi_clip.bounding_boxes[0].x2 == 40
    assert bbsoi_clip.bounding_boxes[1].y1 == 15
    assert bbsoi_clip.bounding_boxes[1].x1 == 25
    assert bbsoi_clip.bounding_boxes[1].y2 == 35
    assert 50 - 2*eps < bbsoi_clip.bounding_boxes[1].x2 < 50

    # shift()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bbsoi_shifted = bbsoi.shift(right=1)
    assert len(bbsoi_shifted.bounding_boxes) == 2
    assert bbsoi_shifted.bounding_boxes[0].y1 == 10
    assert bbsoi_shifted.bounding_boxes[0].x1 == 20 - 1
    assert bbsoi_shifted.bounding_boxes[0].y2 == 30
    assert bbsoi_shifted.bounding_boxes[0].x2 == 40 - 1
    assert bbsoi_shifted.bounding_boxes[1].y1 == 15
    assert bbsoi_shifted.bounding_boxes[1].x1 == 25 - 1
    assert bbsoi_shifted.bounding_boxes[1].y2 == 35
    assert bbsoi_shifted.bounding_boxes[1].x2 == 51 - 1

    # copy()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bbsoi_copy = bbsoi.copy()
    assert len(bbsoi.bounding_boxes) == 2
    assert bbsoi_copy.bounding_boxes[0].y1 == 10
    assert bbsoi_copy.bounding_boxes[0].x1 == 20
    assert bbsoi_copy.bounding_boxes[0].y2 == 30
    assert bbsoi_copy.bounding_boxes[0].x2 == 40
    assert bbsoi_copy.bounding_boxes[1].y1 == 15
    assert bbsoi_copy.bounding_boxes[1].x1 == 25
    assert bbsoi_copy.bounding_boxes[1].y2 == 35
    assert bbsoi_copy.bounding_boxes[1].x2 == 51

    bbsoi.bounding_boxes[0].y1 = 0
    assert bbsoi_copy.bounding_boxes[0].y1 == 0

    # deepcopy()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bbsoi_copy = bbsoi.deepcopy()
    assert len(bbsoi.bounding_boxes) == 2
    assert bbsoi_copy.bounding_boxes[0].y1 == 10
    assert bbsoi_copy.bounding_boxes[0].x1 == 20
    assert bbsoi_copy.bounding_boxes[0].y2 == 30
    assert bbsoi_copy.bounding_boxes[0].x2 == 40
    assert bbsoi_copy.bounding_boxes[1].y1 == 15
    assert bbsoi_copy.bounding_boxes[1].x1 == 25
    assert bbsoi_copy.bounding_boxes[1].y2 == 35
    assert bbsoi_copy.bounding_boxes[1].x2 == 51

    bbsoi.bounding_boxes[0].y1 = 0
    assert bbsoi_copy.bounding_boxes[0].y1 == 10

    # repr() / str()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bb1_expected = "BoundingBox(x1=20.0000, y1=10.0000, x2=40.0000, y2=30.0000, label=None)"
    bb2_expected = "BoundingBox(x1=25.0000, y1=15.0000, x2=51.0000, y2=35.0000, label=None)"
    expected = "BoundingBoxesOnImage([%s, %s], shape=(40, 50, 3))" % (bb1_expected, bb2_expected)
    assert bbsoi.__repr__() == bbsoi.__str__() == expected


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


def test_Polygon___init__():
    # exterior is list of Keypoint or
    poly = ia.Polygon([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=0.5, y=2.5)])
    assert poly.exterior.dtype.type == np.float32
    assert np.allclose(
        poly.exterior,
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )

    # exterior is list of tuple of floats
    poly = ia.Polygon([(0.0, 0.0), (1.0, 1.0), (0.5, 2.5)])
    assert poly.exterior.dtype.type == np.float32
    assert np.allclose(
        poly.exterior,
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )

    # exterior is list of tuple of integer
    poly = ia.Polygon([(0, 0), (1, 1), (1, 3)])
    assert poly.exterior.dtype.type == np.float32
    assert np.allclose(
        poly.exterior,
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 3.0]
        ])
    )

    # exterior is (N,2) ndarray
    poly = ia.Polygon(
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )
    assert poly.exterior.dtype.type == np.float32
    assert np.allclose(
        poly.exterior,
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )

    # exterior is (N,2) ndarray in float64
    poly = ia.Polygon(
        np.float64([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )
    assert poly.exterior.dtype.type == np.float32
    assert np.allclose(
        poly.exterior,
        np.float32([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 2.5]
        ])
    )

    # arrays without points
    poly = ia.Polygon([])
    assert poly.exterior.dtype.type == np.float32
    assert poly.exterior.shape == (0, 2)

    poly = ia.Polygon(np.zeros((0, 2), dtype=np.float32))
    assert poly.exterior.dtype.type == np.float32
    assert poly.exterior.shape == (0, 2)

    # bad array shape
    got_exception = False
    try:
        _ = ia.Polygon(np.zeros((8,), dtype=np.float32))
    except:
        got_exception = True
    assert got_exception

    # label
    poly = ia.Polygon([(0, 0)])
    assert poly.label is None
    poly = ia.Polygon([(0, 0)], label="test")
    assert poly.label == "test"


def test_Polygon_xx():
    poly = ia.Polygon([(0, 0), (1, 0), (1.5, 0), (4.1, 1), (2.9, 2.0)])
    assert poly.xx.dtype.type == np.float32
    assert np.allclose(poly.xx, np.float32([0.0, 1.0, 1.5, 4.1, 2.9]))

    poly = ia.Polygon([])
    assert poly.xx.dtype.type == np.float32
    assert poly.xx.shape == (0,)


def test_Polygon_yy():
    poly = ia.Polygon([(0, 0), (0, 1), (0, 1.5), (1, 4.1), (2.0, 2.9)])
    assert poly.yy.dtype.type == np.float32
    assert np.allclose(poly.yy, np.float32([0.0, 1.0, 1.5, 4.1, 2.9]))

    poly = ia.Polygon([])
    assert poly.yy.dtype.type == np.float32
    assert poly.yy.shape == (0,)


def test_Polygon_xx_int():
    poly = ia.Polygon([(0, 0), (1, 0), (1.5, 0), (4.1, 1), (2.9, 2.0)])
    assert poly.xx_int.dtype.type == np.int32
    assert np.allclose(poly.xx_int, np.int32([0, 1, 2, 4, 3]))

    poly = ia.Polygon([])
    assert poly.xx_int.dtype.type == np.int32
    assert poly.xx_int.shape == (0,)


def test_Polygon_yy_int():
    poly = ia.Polygon([(0, 0), (0, 1), (0, 1.5), (1, 4.1), (2.0, 2.9)])
    assert poly.yy_int.dtype.type == np.int32
    assert np.allclose(poly.yy_int, np.int32([0, 1, 2, 4, 3]))

    poly = ia.Polygon([])
    assert poly.yy_int.dtype.type == np.int32
    assert poly.yy_int.shape == (0,)


def test_Polygon_is_valid():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_valid

    poly = ia.Polygon([])
    assert not poly.is_valid

    poly = ia.Polygon([(0, 0)])
    assert not poly.is_valid

    poly = ia.Polygon([(0, 0), (1, 0)])
    assert not poly.is_valid

    poly = ia.Polygon([(0, 0), (1, 0), (-1, 0.5), (1, 1), (0, 1)])
    assert not poly.is_valid

    poly = ia.Polygon([(0, 0), (1, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_valid


def test_Polygon_area():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.area == 1
    assert 1.0 - 1e-8 < poly.area < 1.0 + 1e-8

    poly = ia.Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    assert poly.area == 2
    assert 2.0 - 1e-8 < poly.area < 2.0 + 1e-8

    poly = ia.Polygon([(0, 0), (1, 1), (0, 1)])
    assert 1/2 - 1e-8 < poly.area < 1/2 + 1e-8

    poly = ia.Polygon([(0, 0), (1, 1)])
    got_exception = False
    try:
        _ = poly.area
    except Exception as exc:
        assert "Cannot compute the polygon's area because" in str(exc)
        got_exception = True
    assert got_exception


def test_Polygon_project():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_proj = poly.project((1, 1), (1, 1))
    assert poly_proj.exterior.dtype.type == np.float32
    assert poly_proj.exterior.shape == (4, 2)
    assert np.allclose(
        poly_proj.exterior,
        np.float32([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
    )

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_proj = poly.project((1, 1), (2, 2))
    assert poly_proj.exterior.dtype.type == np.float32
    assert poly_proj.exterior.shape == (4, 2)
    assert np.allclose(
        poly_proj.exterior,
        np.float32([
            [0, 0],
            [2, 0],
            [2, 2],
            [0, 2]
        ])
    )

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_proj = poly.project((1, 1), (2, 1))
    assert poly_proj.exterior.dtype.type == np.float32
    assert poly_proj.exterior.shape == (4, 2)
    assert np.allclose(
        poly_proj.exterior,
        np.float32([
            [0, 0],
            [1, 0],
            [1, 2],
            [0, 2]
        ])
    )

    poly = ia.Polygon([])
    poly_proj = poly.project((1, 1), (2, 2))
    assert poly_proj.exterior.dtype.type == np.float32
    assert poly_proj.exterior.shape == (0, 2)


def test_Polygon_find_closest_point_idx():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    closest_idx = poly.find_closest_point_index(x=0, y=0)
    assert closest_idx == 0
    closest_idx = poly.find_closest_point_index(x=1, y=0)
    assert closest_idx == 1
    closest_idx = poly.find_closest_point_index(x=1.0001, y=-0.001)
    assert closest_idx == 1
    closest_idx = poly.find_closest_point_index(x=0.2, y=0.2)
    assert closest_idx == 0

    closest_idx, distance = poly.find_closest_point_index(x=0, y=0, return_distance=True)
    assert closest_idx == 0
    assert np.allclose(distance, 0.0)
    closest_idx, distance = poly.find_closest_point_index(x=0.1, y=0.15, return_distance=True)
    assert closest_idx == 0
    assert np.allclose(distance, np.sqrt((0.1**2) + (0.15**2)))
    closest_idx, distance = poly.find_closest_point_index(x=0.9, y=0.15, return_distance=True)
    assert closest_idx == 1
    assert np.allclose(distance, np.sqrt(((1.0-0.9)**2) + (0.15**2)))


def test_Polygon__compute_inside_image_point_mask():
    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    mask = poly._compute_inside_image_point_mask((1, 1, 3))
    assert np.array_equal(mask, np.array([True, True, True, True], dtype=bool))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    mask = poly._compute_inside_image_point_mask((1, 1, 3))
    assert np.array_equal(mask, np.array([True, False, False, False], dtype=bool))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    mask = poly._compute_inside_image_point_mask((1, 1))
    assert np.array_equal(mask, np.array([True, False, False, False], dtype=bool))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    mask = poly._compute_inside_image_point_mask(np.zeros((1, 1, 3), dtype=np.uint8))
    assert np.array_equal(mask, np.array([True, False, False, False], dtype=bool))


def test_Polygon_is_fully_within_image():
    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_fully_within_image((1, 1, 3))

    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_fully_within_image((1, 1))

    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_fully_within_image(np.zeros((1, 1, 3), dtype=np.uint8))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert not poly.is_fully_within_image((1, 1, 3))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert not poly.is_fully_within_image((1, 1))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert not poly.is_fully_within_image(np.zeros((1, 1, 3), dtype=np.uint8))

    poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    assert not poly.is_fully_within_image((1, 1, 3))


def test_Polygon_is_partly_within_image():
    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_partly_within_image((1, 1, 3))

    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_partly_within_image((1, 1))

    poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
    assert poly.is_partly_within_image(np.zeros((1, 1, 3), dtype=np.uint8))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_partly_within_image((1, 1, 3))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_partly_within_image((1, 1))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_partly_within_image(np.zeros((1, 1, 3), dtype=np.uint8))

    poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    assert not poly.is_partly_within_image((1, 1, 3))

    poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    assert not poly.is_partly_within_image((1, 1))

    poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    assert not poly.is_partly_within_image(np.zeros((1, 1, 3), dtype=np.uint8))


def test_Polygon_is_out_of_image():
    for shape in [(1, 1, 3), (1, 1), np.zeros((1, 1, 3), dtype=np.uint8)]:
        poly = ia.Polygon([(0, 0), (0.999, 0), (0.999, 0.999), (0, 0.999)])
        assert not poly.is_out_of_image(shape, partly=False, fully=False)
        assert not poly.is_out_of_image(shape, partly=True, fully=False)
        assert not poly.is_out_of_image(shape, partly=False, fully=True)
        assert not poly.is_out_of_image(shape, partly=True, fully=True)

        poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        shape = np.zeros((1, 1, 3), dtype=np.uint8)
        assert not poly.is_out_of_image(shape, partly=False, fully=False)
        assert poly.is_out_of_image(shape, partly=True, fully=False)
        assert not poly.is_out_of_image(shape, partly=False, fully=True)
        assert poly.is_out_of_image(shape, partly=True, fully=True)

        poly = ia.Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
        shape = (1, 1, 3)
        assert not poly.is_out_of_image(shape, partly=False, fully=False)
        assert not poly.is_out_of_image(shape, partly=True, fully=False)
        assert poly.is_out_of_image(shape, partly=False, fully=True)
        assert poly.is_out_of_image(shape, partly=True, fully=True)

    poly = ia.Polygon([])
    got_exception = False
    try:
        poly.is_out_of_image((1, 1, 3))
    except Exception as exc:
        assert "Cannot determine whether the polygon is inside the image" in str(exc)
        got_exception = True
    assert got_exception


def test_Polygon_cut_out_of_image():
    with warnings.catch_warnings(record=True) as caught_warnings:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        _test_Polygon_cut_clip(lambda poly, image: poly.cut_out_of_image(image))
        # Verify
        # get multiple warnings here, one for each function call
        assert all([
            "Use Polygon.clip_out_of_image() instead" in str(msg.message)
            for msg in caught_warnings])


def test_Polygon_clip_out_of_image():
    _test_Polygon_cut_clip(lambda poly, image: poly.clip_out_of_image(image))


def _test_Polygon_cut_clip(func):
    # poly inside image
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label=None)
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    multipoly_clipped = func(poly, image)
    assert isinstance(multipoly_clipped, ia.MultiPolygon)
    assert len(multipoly_clipped.geoms) == 1
    assert multipoly_clipped.geoms[0].exterior_almost_equals(poly.exterior)
    assert multipoly_clipped.geoms[0].label is None

    # square poly shifted by x=0.5, y=0.5 => half out of image
    poly = ia.Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)], label="test")
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    multipoly_clipped = func(poly, image)
    assert isinstance(multipoly_clipped, ia.MultiPolygon)
    assert len(multipoly_clipped.geoms) == 1
    assert multipoly_clipped.geoms[0].exterior_almost_equals(np.float32([
        [0.5, 0.5],
        [1.0, 0.5],
        [1.0, 1.0],
        [0.5, 1.0]
    ]))
    assert multipoly_clipped.geoms[0].label == "test"

    # non-square poly, with one rectangle on the left side of the image and one on the right side,
    # both sides are connected by a thin strip below the image
    # after clipping it should become two rectangles
    poly = ia.Polygon([(-0.1, 0.0), (0.4, 0.0), (0.4, 1.1), (0.6, 1.1), (0.6, 0.0), (1.1, 0.0),
                       (1.1, 1.2), (-0.1, 1.2)],
                      label="test")
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    multipoly_clipped = func(poly, image)
    assert isinstance(multipoly_clipped, ia.MultiPolygon)
    assert len(multipoly_clipped.geoms) == 2
    assert multipoly_clipped.geoms[0].exterior_almost_equals(np.float32([
        [0.0, 0.0],
        [0.4, 0.0],
        [0.4, 1.0],
        [0.0, 1.0]
    ]))
    assert multipoly_clipped.geoms[0].label == "test"
    assert multipoly_clipped.geoms[1].exterior_almost_equals(np.float32([
        [0.6, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.6, 1.0]
    ]))
    assert multipoly_clipped.geoms[0].label == "test"

    # poly outside of image
    poly = ia.Polygon([(10.0, 10.0)])
    multipoly_clipped = func(poly, (5, 5, 3))
    assert isinstance(multipoly_clipped, ia.MultiPolygon)
    assert len(multipoly_clipped.geoms) == 0


def test_Polygon_shift():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")

    # make sure that shift does not change poly inplace
    poly_shifted = poly.shift(top=1)
    assert np.allclose(poly.exterior, np.float32([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ]))
    assert np.allclose(poly_shifted.exterior, np.float32([
        [0, 1],
        [1, 1],
        [1, 2],
        [0, 2]
    ]))

    for v in [1, 0, -1, 0.5]:
        # top/bottom
        poly_shifted = poly.shift(top=v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0, 0 + v],
            [1, 0 + v],
            [1, 1 + v],
            [0, 1 + v]
        ]))
        assert poly_shifted.label == "test"

        poly_shifted = poly.shift(bottom=v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0, 0 - v],
            [1, 0 - v],
            [1, 1 - v],
            [0, 1 - v]
        ]))
        assert poly_shifted.label == "test"

        poly_shifted = poly.shift(top=v, bottom=-v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0, 0 + 2*v],
            [1, 0 + 2*v],
            [1, 1 + 2*v],
            [0, 1 + 2*v]
        ]))
        assert poly_shifted.label == "test"

        # left/right
        poly_shifted = poly.shift(left=v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0 + v, 0],
            [1 + v, 0],
            [1 + v, 1],
            [0 + v, 1]
        ]))
        assert poly_shifted.label == "test"

        poly_shifted = poly.shift(right=v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0 - v, 0],
            [1 - v, 0],
            [1 - v, 1],
            [0 - v, 1]
        ]))
        assert poly_shifted.label == "test"

        poly_shifted = poly.shift(left=v, right=-v)
        assert np.allclose(poly_shifted.exterior, np.float32([
            [0 + 2 * v, 0],
            [1 + 2 * v, 0],
            [1 + 2 * v, 1],
            [0 + 2 * v, 1]
        ]))
        assert poly_shifted.label == "test"


def test_Polygon_draw_on_image():
    image = np.tile(np.arange(100).reshape(10, 10, 1), (1, 1, 3)).astype(np.uint8)

    # simple drawing of square
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    image_poly = poly.draw_on_image(image,
                                    color=[32, 128, 32], color_perimeter=[0, 255, 0],
                                    alpha=1.0, alpha_perimeter=1.0,
                                    raise_if_out_of_image=False)
    assert image_poly.dtype.type == np.uint8
    assert image_poly.shape == (10, 10, 3)
    assert np.sum(image) == 3 * np.sum(np.arange(100))  # draw did not change original image (copy=True)
    for c_idx, value in enumerate([0, 255, 0]):
        assert np.all(image_poly[2:9, 2:3, c_idx] == np.zeros((7, 1), dtype=np.uint8) + value)  # left boundary
        assert np.all(image_poly[2:9, 8:9, c_idx] == np.zeros((7, 1), dtype=np.uint8) + value)  # right boundary
        assert np.all(image_poly[2:3, 2:9, c_idx] == np.zeros((1, 7), dtype=np.uint8) + value)  # top boundary
        assert np.all(image_poly[8:9, 2:9, c_idx] == np.zeros((1, 7), dtype=np.uint8) + value)  # bottom boundary
    expected = np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1))
    assert np.all(image_poly[3:8, 3:8, :] == expected)

    # simple drawing of square with float32 input
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    image_poly = poly.draw_on_image(image.astype(np.float32),
                                    color=[32, 128, 32], color_perimeter=[0, 255, 0],
                                    alpha=1.0, alpha_perimeter=1.0,
                                    raise_if_out_of_image=False)
    assert image_poly.dtype.type == np.float32
    assert image_poly.shape == (10, 10, 3)
    for c_idx, value in enumerate([0, 255, 0]):
        assert np.allclose(image_poly[2:9, 2:3, c_idx], np.zeros((7, 1), dtype=np.float32) + value)  # left boundary
        assert np.allclose(image_poly[2:9, 8:9, c_idx], np.zeros((7, 1), dtype=np.float32) + value)  # right boundary
        assert np.allclose(image_poly[2:3, 2:9, c_idx], np.zeros((1, 7), dtype=np.float32) + value)  # top boundary
        assert np.allclose(image_poly[8:9, 2:9, c_idx], np.zeros((1, 7), dtype=np.float32) + value)  # bottom boundary
    expected = np.tile(np.float32([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1))
    assert np.allclose(image_poly[3:8, 3:8, :], expected)

    # drawing of poly that is half out of image
    poly = ia.Polygon([(2, 2+5), (8, 2+5), (8, 8+5), (2, 8+5)])
    image_poly = poly.draw_on_image(image,
                                    color=[32, 128, 32], color_perimeter=[0, 255, 0],
                                    alpha=1.0, alpha_perimeter=1.0,
                                    raise_if_out_of_image=False)
    assert image_poly.dtype.type == np.uint8
    assert image_poly.shape == (10, 10, 3)
    assert np.sum(image) == 3 * np.sum(np.arange(100))  # draw did not change original image (copy=True)
    for c_idx, value in enumerate([0, 255, 0]):
        assert np.all(image_poly[2+5:, 2:3, c_idx] == np.zeros((3, 1), dtype=np.uint8) + value)  # left boundary
        assert np.all(image_poly[2+5:, 8:9, c_idx] == np.zeros((3, 1), dtype=np.uint8) + value)  # right boundary
        assert np.all(image_poly[2+5:3+5, 2:9, c_idx] == np.zeros((1, 7), dtype=np.uint8) + value)  # top boundary
    expected = np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (2, 5, 1))
    assert np.all(image_poly[3+5:, 3:8, :] == expected)

    # drawing of poly that is half out of image, with raise_if_out_of_image=True
    poly = ia.Polygon([(2, 2+5), (8, 2+5), (8, 8+5), (0, 8+5)])
    got_exception = False
    try:
        _ = poly.draw_on_image(image,
                               color=[32, 128, 32], color_perimeter=[0, 255, 0],
                               alpha=1.0, alpha_perimeter=1.0,
                               raise_if_out_of_image=True)
    except Exception as exc:
        assert "Cannot draw polygon" in str(exc)
        got_exception = True
    assert not got_exception  # only polygons fully outside of the image plane lead to exceptions

    # drawing of poly that is fully out of image
    poly = ia.Polygon([(100, 100), (100+10, 100), (100+10, 100+10), (100, 100+10)])
    image_poly = poly.draw_on_image(image,
                                    color=[32, 128, 32], color_perimeter=[0, 255, 0],
                                    alpha=1.0, alpha_perimeter=1.0,
                                    raise_if_out_of_image=False)
    assert np.array_equal(image_poly, image)

    # drawing of poly that is fully out of image, with raise_if_out_of_image=True
    poly = ia.Polygon([(100, 100), (100+10, 100), (100+10, 100+10), (100, 100+10)])
    got_exception = False
    try:
        _ = poly.draw_on_image(image,
                               color=[32, 128, 32], color_perimeter=[0, 255, 0],
                               alpha=1.0, alpha_perimeter=1.0,
                               raise_if_out_of_image=True)
    except Exception as exc:
        assert "Cannot draw polygon" in str(exc)
        got_exception = True
    assert got_exception

    # face invisible via alpha
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    image_poly = poly.draw_on_image(image,
                                    color=[32, 128, 32], color_perimeter=[0, 255, 0],
                                    alpha=0.0, alpha_perimeter=1.0,
                                    raise_if_out_of_image=False)
    assert image_poly.dtype.type == np.uint8
    assert image_poly.shape == (10, 10, 3)
    assert np.sum(image) == 3 * np.sum(np.arange(100))  # draw did not change original image (copy=True)
    for c_idx, value in enumerate([0, 255, 0]):
        assert np.all(image_poly[2:9, 2:3, c_idx] == np.zeros((7, 1), dtype=np.uint8) + value)  # left boundary
    assert np.all(image_poly[3:8, 3:8, :] == image[3:8, 3:8, :])

    # boundary invisible via alpha
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    image_poly = poly.draw_on_image(image,
                                    color=[32, 128, 32], color_perimeter=[0, 255, 0],
                                    alpha=1.0, alpha_perimeter=0.0,
                                    raise_if_out_of_image=False)
    assert image_poly.dtype.type == np.uint8
    assert image_poly.shape == (10, 10, 3)
    assert np.sum(image) == 3 * np.sum(np.arange(100))  # draw did not change original image (copy=True)
    expected = np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (6, 6, 1))
    assert np.all(image_poly[2:8, 2:8, :] == expected)

    # alpha=0.5
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    image_poly = poly.draw_on_image(image,
                                    color=[32, 128, 32], color_perimeter=[0, 255, 0],
                                    alpha=0.5, alpha_perimeter=0.5,
                                    raise_if_out_of_image=False)
    assert image_poly.dtype.type == np.uint8
    assert image_poly.shape == (10, 10, 3)
    for c_idx, value in enumerate([0, 255, 0]):
        assert np.all(
            image_poly[2:9, 8:9, c_idx] ==
            (
                0.5*image[2:9, 8:9, c_idx]
                + np.full((7, 1), 0.5*value, dtype=np.float32)
            ).astype(np.uint8)
        )  # right boundary
    expected = 0.5 * np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1)) \
        + 0.5 * image[3:8, 3:8, :]
    assert np.all(image_poly[3:8, 3:8, :] == expected.astype(np.uint8))

    # copy=False
    # test deactivated as the function currently does not offer a copy argument
    """
    image_cp = np.copy(image)
    poly = ia.Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    image_poly = poly.draw_on_image(image_cp,
                                    color_face=[32, 128, 32], color_boundary=[0, 255, 0],
                                    alpha_face=1.0, alpha_boundary=1.0,
                                    raise_if_out_of_image=False)
    assert image_poly.dtype.type == np.uint8
    assert image_poly.shape == (10, 10, 3)
    assert np.all(image_cp == image_poly)
    assert not np.all(image_cp == image)
    for c_idx, value in enumerate([0, 255, 0]):
        assert np.all(image_poly[2:9, 2:3, c_idx] == np.zeros((6, 1, 3), dtype=np.uint8) + value)  # left boundary
        assert np.all(image_cp[2:9, 2:3, c_idx] == np.zeros((6, 1, 3), dtype=np.uint8) + value)  # left boundary
    expected = np.tile(np.uint8([32, 128, 32]).reshape((1, 1, 3)), (5, 5, 1))
    assert np.all(image_poly[3:8, 3:8, :] == expected)
    assert np.all(image_cp[3:8, 3:8, :] == expected)
    """


def test_Polygon_extract_from_image():
    image = np.arange(20*20*2).reshape(20, 20, 2).astype(np.int32)

    # inside image and completely covers it
    poly = ia.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    subimage = poly.extract_from_image(image)
    assert np.array_equal(subimage, image[0:10, 0:10, :])

    # inside image, subpart of it (not all may be extracted)
    poly = ia.Polygon([(1, 1), (9, 1), (9, 9), (1, 9)])
    subimage = poly.extract_from_image(image)
    assert np.array_equal(subimage, image[1:9, 1:9, :])

    # inside image, two image areas that don't belong to the polygon but have to be extracted
    poly = ia.Polygon([(0, 0), (10, 0), (10, 5), (20, 5),
                       (20, 20), (10, 20), (10, 5), (0, 5)])
    subimage = poly.extract_from_image(image)
    expected = np.copy(image)
    expected[:5, 10:, :] = 0  # top right block
    expected[5:, :10, :] = 0  # left bottom block
    assert np.array_equal(subimage, expected)

    # partially out of image
    poly = ia.Polygon([(-5, 0), (5, 0), (5, 10), (-5, 10)])
    subimage = poly.extract_from_image(image)
    expected = np.zeros((10, 10, 2), dtype=np.int32)
    expected[0:10, 5:10, :] = image[0:10, 0:5, :]
    assert np.array_equal(subimage, expected)

    # fully out of image
    poly = ia.Polygon([(30, 0), (40, 0), (40, 10), (30, 10)])
    subimage = poly.extract_from_image(image)
    expected = np.zeros((10, 10, 2), dtype=np.int32)
    assert np.array_equal(subimage, expected)

    # inside image, subpart of it
    # float coordinates, rounded so that the whole image will be extracted
    poly = ia.Polygon([(0.4, 0.4), (9.6, 0.4), (9.6, 9.6), (0.4, 9.6)])
    subimage = poly.extract_from_image(image)
    assert np.array_equal(subimage, image[0:10, 0:10, :])

    # inside image, subpart of it
    # float coordinates, rounded so that x/y 0<=i<9 will be extracted (instead of 0<=i<10)
    poly = ia.Polygon([(0.5, 0.5), (9.4, 0.5), (9.4, 9.4), (0.5, 9.4)])
    subimage = poly.extract_from_image(image)
    assert np.array_equal(subimage, image[0:9, 0:9, :])

    # inside image, subpart of it
    # float coordinates, rounded so that x/y 1<=i<9 will be extracted (instead of 0<=i<10)
    poly = ia.Polygon([(0.51, 0.51), (9.4, 0.51), (9.4, 9.4), (0.51, 9.4)])
    subimage = poly.extract_from_image(image)
    assert np.array_equal(subimage, image[1:9, 1:9, :])

    # error for invalid polygons
    got_exception = False
    poly = ia.Polygon([(0.51, 0.51), (9.4, 0.51)])
    try:
        _ = poly.extract_from_image(image)
    except Exception as exc:
        assert "Polygon must be made up" in str(exc)
        got_exception = True
    assert got_exception


def test_Polygon_change_first_point_by_coords():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_coords(x=0, y=0)
    assert np.allclose(poly.exterior, poly_reordered.exterior)

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_coords(x=1, y=0)
    # make sure that it does not reorder inplace
    assert np.allclose(poly.exterior, np.float32([[0, 0], [1, 0], [1, 1]]))
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [1, 1], [0, 0]]))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_coords(x=1, y=1)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 1], [0, 0], [1, 0]]))

    # inaccurate point, but close enough
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_coords(x=1.0, y=0.01, max_distance=0.1)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [1, 1], [0, 0]]))

    # inaccurate point, but close enough (infinite max distance)
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_coords(x=1.0, y=0.01, max_distance=None)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [1, 1], [0, 0]]))

    # point too far away
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    got_exception = False
    try:
        _ = poly.change_first_point_by_coords(x=1.0, y=0.01, max_distance=0.001)
    except Exception as exc:
        assert "Closest found point " in str(exc)
        got_exception = True
    assert got_exception

    # reorder with two points
    poly = ia.Polygon([(0, 0), (1, 0)])
    poly_reordered = poly.change_first_point_by_coords(x=1, y=0)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [0, 0]]))

    # reorder with one point
    poly = ia.Polygon([(0, 0)])
    poly_reordered = poly.change_first_point_by_coords(x=0, y=0)
    assert np.allclose(poly_reordered.exterior, np.float32([[0, 0]]))

    # invalid polygon
    git_exception = False
    poly = ia.Polygon([])
    try:
        _ = poly.change_first_point_by_coords(x=0, y=0)
    except Exception as exc:
        assert "Cannot reorder polygon points" in str(exc)
        got_exception = True


def test_Polygon_change_first_point_by_index():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_index(0)
    assert np.allclose(poly.exterior, poly_reordered.exterior)

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_index(1)
    # make sure that it does not reorder inplace
    assert np.allclose(poly.exterior, np.float32([[0, 0], [1, 0], [1, 1]]))
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [1, 1], [0, 0]]))

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    poly_reordered = poly.change_first_point_by_index(2)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 1], [0, 0], [1, 0]]))

    # reorder with two points
    poly = ia.Polygon([(0, 0), (1, 0)])
    poly_reordered = poly.change_first_point_by_index(1)
    assert np.allclose(poly_reordered.exterior, np.float32([[1, 0], [0, 0]]))

    # reorder with one point
    poly = ia.Polygon([(0, 0)])
    poly_reordered = poly.change_first_point_by_index(0)
    assert np.allclose(poly_reordered.exterior, np.float32([[0, 0]]))

    # idx out of bounds
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    got_exception = False
    try:
        _ = poly.change_first_point_by_index(3)
    except AssertionError:
        got_exception = True
    assert got_exception

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    got_exception = False
    try:
        _ = poly.change_first_point_by_index(-1)
    except AssertionError:
        got_exception = True
    assert got_exception

    poly = ia.Polygon([(0, 0)])
    got_exception = False
    try:
        _ = poly.change_first_point_by_index(1)
    except AssertionError:
        got_exception = True
    assert got_exception

    poly = ia.Polygon([])
    got_exception = False
    try:
        _ = poly.change_first_point_by_index(0)
    except AssertionError:
        got_exception = True
    assert got_exception


def test_Polygon_to_shapely_line_string():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    ls = poly.to_shapely_line_string()
    assert np.allclose(ls.coords, np.float32([[0, 0], [1, 0], [1, 1]]))

    # two point polygon
    poly = ia.Polygon([(0, 0), (1, 0)])
    ls = poly.to_shapely_line_string()
    assert np.allclose(ls.coords, np.float32([[0, 0], [1, 0]]))

    # one point polygon
    poly = ia.Polygon([(0, 0)])
    got_exception = False
    try:
        _ = poly.to_shapely_line_string()
    except Exception as exc:
        assert "Conversion to shapely line string requires at least two points" in str(exc)
        got_exception = True
    assert got_exception

    # zero point polygon
    poly = ia.Polygon([])
    got_exception = False
    try:
        _ = poly.to_shapely_line_string()
    except Exception as exc:
        assert "Conversion to shapely line string requires at least two points" in str(exc)
        got_exception = True
    assert got_exception

    # closed line string
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    ls = poly.to_shapely_line_string(closed=True)
    assert np.allclose(ls.coords, np.float32([[0, 0], [1, 0], [1, 1], [0, 0]]))

    # interpolation
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    ls = poly.to_shapely_line_string(interpolate=1)
    assert np.allclose(ls.coords, np.float32([[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 0.5]]))

    # interpolation with 2 steps
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    ls = poly.to_shapely_line_string(interpolate=2)
    assert np.allclose(ls.coords, np.float32([
        [0, 0], [1/3, 0], [2/3, 0],
        [1, 0], [1, 1/3], [1, 2/3],
        [1, 1], [2/3, 2/3], [1/3, 1/3]
    ]))

    # interpolation with closed=True
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1)])
    ls = poly.to_shapely_line_string(closed=True, interpolate=1)
    assert np.allclose(ls.coords, np.float32([[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 0.5], [0, 0]]))


def test_Polygon_to_shapely_polygon():
    exterior = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly = ia.Polygon(exterior)
    poly_shapely = poly.to_shapely_polygon()
    for (x_exp, y_exp), (x_obs, y_obs) in zip(exterior, poly_shapely.exterior.coords):
        assert x_exp - 1e-8 < x_obs < x_exp + 1e-8
        assert y_exp - 1e-8 < y_obs < y_exp + 1e-8


def test_Polygon_to_bounding_box():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    bb = poly.to_bounding_box()
    assert 0 - 1e-8 < bb.x1 < 0 + 1e-8
    assert 0 - 1e-8 < bb.y1 < 0 + 1e-8
    assert 1 - 1e-8 < bb.x2 < 1 + 1e-8
    assert 1 - 1e-8 < bb.y2 < 1 + 1e-8

    poly = ia.Polygon([(0.5, 0), (1, 1), (0, 1)])
    bb = poly.to_bounding_box()
    assert 0 - 1e-8 < bb.x1 < 0 + 1e-8
    assert 0 - 1e-8 < bb.y1 < 0 + 1e-8
    assert 1 - 1e-8 < bb.x2 < 1 + 1e-8
    assert 1 - 1e-8 < bb.y2 < 1 + 1e-8

    poly = ia.Polygon([(0.5, 0.5), (2, 0.1), (1, 1)])
    bb = poly.to_bounding_box()
    assert 0.5 - 1e-8 < bb.x1 < 0.5 + 1e-8
    assert 0.1 - 1e-8 < bb.y1 < 0.1 + 1e-8
    assert 2.0 - 1e-8 < bb.x2 < 2.0 + 1e-8
    assert 1.0 - 1e-8 < bb.y2 < 1.0 + 1e-8


def test_Polygon_from_shapely():
    exterior = [(0, 0), (1, 0), (1, 1), (0, 1)]
    poly_shapely = shapely.geometry.Polygon(exterior)
    poly = ia.Polygon.from_shapely(poly_shapely)

    # shapely messes up the point ordering, so we try to correct it here
    start_idx = 0
    for i, (x, y) in enumerate(poly.exterior):
        dist = np.sqrt((exterior[0][0] - x) ** 2 + (exterior[0][1] - x) ** 2)
        if dist < 1e-4:
            start_idx = i
            break
    poly = poly.change_first_point_by_index(start_idx)

    for (x_exp, y_exp), (x_obs, y_obs) in zip(exterior, poly.exterior):
        assert x_exp - 1e-8 < x_obs < x_exp + 1e-8
        assert y_exp - 1e-8 < y_obs < y_exp + 1e-8

    # empty polygon
    poly_shapely = shapely.geometry.Polygon([])
    poly = ia.Polygon.from_shapely(poly_shapely)
    assert len(poly.exterior) == 0


def test_Polygon_copy():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
    poly_cp = poly.copy()
    assert poly.exterior.dtype.type == poly_cp.exterior.dtype.type
    assert poly.exterior.shape == poly_cp.exterior.shape
    assert np.allclose(poly.exterior, poly_cp.exterior)
    assert poly.label == poly_cp.label


def test_Polygon_deepcopy():
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
    poly_cp = poly.deepcopy()
    assert poly.exterior.dtype.type == poly_cp.exterior.dtype.type
    assert poly.exterior.shape == poly_cp.exterior.shape
    assert np.allclose(poly.exterior, poly_cp.exterior)
    assert poly.label == poly_cp.label

    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
    poly_cp = poly.deepcopy()
    poly_cp.exterior[0, 0] = 100.0
    poly_cp.label = "test2"
    assert poly.exterior.dtype.type == poly_cp.exterior.dtype.type
    assert poly.exterior.shape == poly_cp.exterior.shape
    assert not np.allclose(poly.exterior, poly_cp.exterior)
    assert not poly.label == poly_cp.label


def test_Polygon___repr__():
    _test_Polygon_repr_str(lambda poly: poly.__repr__())


def test_Polygon___str__():
    _test_Polygon_repr_str(lambda poly: poly.__str__())


def _test_Polygon_repr_str(func):
    # ints
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label="test")
    s = func(poly)
    assert s == "Polygon([(x=0.000, y=0.000), (x=1.000, y=0.000), (x=1.000, y=1.000), (x=0.000, y=1.000)] " \
                + "(4 points), label=test)"

    # floats
    poly = ia.Polygon([(0, 0.5), (1.5, 0), (1, 1), (0, 1)], label="test")
    s = func(poly)
    assert s == "Polygon([(x=0.000, y=0.500), (x=1.500, y=0.000), (x=1.000, y=1.000), (x=0.000, y=1.000)] " \
                + "(4 points), label=test)"

    # label None
    poly = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], label=None)
    s = func(poly)
    assert s == "Polygon([(x=0.000, y=0.000), (x=1.000, y=0.000), (x=1.000, y=1.000), (x=0.000, y=1.000)] " \
                + "(4 points), label=None)"

    # no points
    poly = ia.Polygon([], label="test")
    s = func(poly)
    assert s == "Polygon([] (0 points), label=test)"


def test_Polygon_exterior_almost_equals():
    # exactly same exterior
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly_a.exterior_almost_equals(poly_b)

    # one point duplicated
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (1, 1), (1, 1), (0, 1)])
    assert poly_a.exterior_almost_equals(poly_b)

    # several points added without changing geometry
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1), (0, 0.5)])
    assert poly_a.exterior_almost_equals(poly_b)

    # different order
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0, 1), (1, 1), (1, 0), (0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    # tiny shift below tolerance
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0+1e-6, 0), (1+1e-6, 0), (1+1e-6, 1), (0+1e-6, 1)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-3)

    # tiny shift above tolerance
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0+1e-6, 0), (1+1e-6, 0), (1+1e-6, 1), (0+1e-6, 1)])
    assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-9)

    # shifted polygon towards half overlap
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
    assert not poly_a.exterior_almost_equals(poly_b)

    # shifted polygon towards no overlap at all
    poly_a = ia.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(100, 0), (101, 0), (101, 1), (100, 1)])
    assert not poly_a.exterior_almost_equals(poly_b)

    # both polygons without points
    poly_a = ia.Polygon([])
    poly_b = ia.Polygon([])
    assert poly_a.exterior_almost_equals(poly_b)

    # both polygons with one point
    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(100, 100)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0+1e-6, 0)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0+1, 0)])
    assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    # both polygons with two points
    poly_a = ia.Polygon([(0, 0), (1, 0)])
    poly_b = ia.Polygon([(0, 0), (1, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0)])
    poly_b = ia.Polygon([(0, 0), (2, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0)])
    poly_b = ia.Polygon([(0+1e-6, 0), (1+1e-6, 0)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    # both polygons with three points
    poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    poly_b = ia.Polygon([(0, 0), (1, -1), (0.5, 1)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    poly_b = ia.Polygon([(0, 0), (1+1e-6, 0), (0.5, 1)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    # one polygon with zero points, other with one
    poly_a = ia.Polygon([])
    poly_b = ia.Polygon([(0, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([])
    assert not poly_a.exterior_almost_equals(poly_b)

    # one polygon with one point, other with two
    poly_a = ia.Polygon([(-10, -20)])
    poly_b = ia.Polygon([(0, 0), (1, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (1, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (0, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (0+1e-6, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    poly_a = ia.Polygon([(0, 0), (0+1e-4, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-9)

    # one polygon with one point, other with three
    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    poly_b = ia.Polygon([(0, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0), (0, 0)])
    assert poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0), (1, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (0, 0)])
    assert not poly_a.exterior_almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0+1e-6, 0), (0, 0+1e-6)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-2)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0+1e-4, 0), (0, 0+1e-4)])
    assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-9)

    # two polygons that are different, but with carefully placed points so that interpolation between polygon
    # points is necessary to spot the difference
    poly_a = ia.Polygon([(1, 0), (1, 1), (0, 1)])
    poly_b = ia.Polygon([(1, 0), (1, 1), (0, 1), (1-1e-6, 1-1e-6)])
    assert poly_a.exterior_almost_equals(poly_b, max_distance=1e-4, interpolate=0)
    assert not poly_a.exterior_almost_equals(poly_b, max_distance=1e-4, interpolate=1)


def test_Polygon_almost_equals():
    poly_a = ia.Polygon([])
    poly_b = ia.Polygon([])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0)])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0)])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0, 0), (0, 0)])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (0+1e-10, 0)])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)], label="test")
    poly_b = ia.Polygon([(0, 0)])
    assert not poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0)], label="test")
    assert not poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)], label="test")
    poly_b = ia.Polygon([(0, 0)], label="test")
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)], label="test")
    poly_b = ia.Polygon([(1, 0)], label="test")
    assert not poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)], label="testA")
    poly_b = ia.Polygon([(0, 0)], label="testB")
    assert not poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    assert poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    poly_b = ia.Polygon([(0, 0), (1, 0), (0.5, 1)])
    assert not poly_a.almost_equals(poly_b)

    poly_a = ia.Polygon([(0, 0)])
    assert not poly_a.almost_equals("foo")


def test___convert_points_to_shapely_line_string():
    # TODO this function seems to already be covered completely by other tests, so add a proper test later
    pass


def test__interpolate_point_pair():
    point_a = (0, 0)
    point_b = (1, 2)
    inter = _interpolate_point_pair(point_a, point_b, 1)
    assert np.allclose(
        inter,
        np.float32([
            [0.5, 1.0]
        ])
    )

    inter = _interpolate_point_pair(point_a, point_b, 2)
    assert np.allclose(
        inter,
        np.float32([
            [1*1/3, 1*2/3],
            [2*1/3, 2*2/3]
        ])
    )

    inter = _interpolate_point_pair(point_a, point_b, 0)
    assert len(inter) == 0


def test__interpolate_points():
    # 2 points
    points = [
        (0, 0),
        (1, 2)
    ]
    inter = _interpolate_points(points, 0)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0],
            [1, 2]
        ])
    )

    inter = _interpolate_points(points, 1)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0],
            [0.5, 1.0],
            [1, 2],
            [0.5, 1.0]
        ])
    )

    inter = _interpolate_points(points, 1, closed=False)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0],
            [0.5, 1.0],
            [1, 2]
        ])
    )

    # 3 points
    points = [
        (0, 0),
        (1, 2),
        (0.5, 3)
    ]

    inter = _interpolate_points(points, 0)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0],
            [1, 2],
            [0.5, 3]
        ])
    )

    inter = _interpolate_points(points, 1)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0],
            [0.5, 1.0],
            [1, 2],
            [0.75, 2.5],
            [0.5, 3],
            [0.25, 1.5]
        ])
    )

    inter = _interpolate_points(points, 1, closed=False)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0],
            [0.5, 1.0],
            [1, 2],
            [0.75, 2.5],
            [0.5, 3]
        ])
    )

    # 0 points
    points = []
    inter = _interpolate_points(points, 1)
    assert len(inter) == 0

    # 1 point
    points = [(0, 0)]
    inter = _interpolate_points(points, 0)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0]
        ])
    )
    inter = _interpolate_points(points, 1)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0]
        ])
    )


def test__interpolate_points_by_max_distance():
    # 2 points
    points = [
        (0, 0),
        (0, 2)
    ]
    inter = _interpolate_points_by_max_distance(points, 10000)
    assert np.allclose(
        inter,
        points
    )

    inter = _interpolate_points_by_max_distance(points, 1.0)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0],
            [0, 1.0],
            [0, 2],
            [0, 1.0]
        ])
    )

    inter = _interpolate_points_by_max_distance(points, 1.0, closed=False)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0],
            [0, 1.0],
            [0, 2]
        ])
    )

    # 3 points
    points = [
        (0, 0),
        (0, 2),
        (2, 0)
    ]

    inter = _interpolate_points_by_max_distance(points, 1.0)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0],
            [0, 1.0],
            [0, 2],
            [1.0, 1.0],
            [2, 0],
            [1.0, 0]
        ])
    )

    inter = _interpolate_points_by_max_distance(points, 1.0, closed=False)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0],
            [0, 1.0],
            [0, 2],
            [1.0, 1.0],
            [2, 0]
        ])
    )

    # 0 points
    points = []
    inter = _interpolate_points_by_max_distance(points, 1.0)
    assert len(inter) == 0

    # 1 points
    points = [(0, 0)]

    inter = _interpolate_points_by_max_distance(points, 1.0)
    assert np.allclose(
        inter,
        np.float32([
            [0, 0]
        ])
    )


class Test_ConcavePolygonRecoverer(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _assert_points_are_identical(cls, observed, expected, atol=1e-8, rtol=0):
        assert len(observed) == len(expected)
        for i, (ps_obs, ps_exp) in enumerate(zip(observed, expected)):
            assert len(ps_obs) == len(ps_exp), "Failed at point %d" % (i,)
            for p_obs, p_exp in zip(ps_obs, ps_exp):
                assert len(p_obs) == 2
                assert len(p_exp) == 2
                assert np.allclose(p_obs, p_exp, atol=atol, rtol=rtol), "Unexpected coords at %d" % (i,)

    def test_recover_from(self):
        # TODO
        assert False

    def test__remove_consecutive_duplicate_points(self):
        recoverer = _ConcavePolygonRecoverer()
        points = [(0, 0), (1, 1)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            points
        )

        points = [(0.0, 0.5), (1.0, 1.0)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            np.float32(points)
        )

        points = np.float32([(0.0, 0.5), (1.0, 1.0)])
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            np.float32(points)
        )

        points = [(0, 0), (0, 0)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            [(0, 0)],
            atol=1e-8, rtol=0
        )

        points = [(0, 0), (0, 0), (1, 0)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            [(0, 0), (1, 0)],
            atol=1e-8, rtol=0
        )

        points = [(0, 0), (1, 0), (1, 0)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            [(0, 0), (1, 0)],
            atol=1e-8, rtol=0
        )

        points = [(0, 0), (1, 0), (1, 0), (2, 0), (0, 0)]
        assert np.allclose(
            recoverer._remove_consecutive_duplicate_points(points),
            [(0, 0), (1, 0), (2, 0)],
            atol=1e-8, rtol=0
        )

    def test__jitter_duplicate_points(self):
        cpr = _ConcavePolygonRecoverer(threshold_duplicate_points=1e-4)
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(points, points_jittered, rtol=0, atol=1e-4)

        points = [(0, 0), (1, 0), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(points, points_jittered, rtol=0, atol=1e-4)

        points = [(0, 0), (0.01, 0), (0.01, 0.01), (0, 0.01)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(points, points_jittered, rtol=0, atol=1e-4)

        points = [(0, 0), (1, 0), (1 + 1e-6, 0), (1, 1), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(
            [point for i, point in enumerate(points_jittered) if i in [0, 1, 3, 4]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[2])) >= 1e-4

        points = [(0, 0), (1, 0), (1, 1), (1 + 1e-6, 0), (0, 1)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(
            [point for i, point in enumerate(points_jittered) if i in [0, 1, 2, 4]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[3])) >= 1e-4

        points = [(0, 0), (1, 0), (1, 1), (0, 1), (1 + 1e-6, 0)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(
            [point for i, point in enumerate(points_jittered) if i in [0, 1, 2, 3]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[4])) >= 1e-4

        points = [(0, 0), (1, 0), (1 + 1e-6, 0), (1, 1), (1 + 1e-6, 0), (0, 1),
                  (1 + 1e-6, 0), (1 + 1e-6, 0 + 1e-6), (1 + 1e-6, 0 + 2e-6)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(
            [point for i, point in enumerate(points_jittered) if i in [0, 1, 3, 5]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[2])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[4])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[6])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[7])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[8])) >= 1e-4

        points = [(0, 0), (1, 0), (0 + 1e-6, 0 - 1e-6), (1 + 1e-6, 0), (1, 1),
                  (1 + 1e-6, 0), (0, 1), (1 + 1e-6, 0), (1 + 1e-6, 0 + 1e-6),
                  (1 + 1e-6, 0 + 2e-6)]
        points_jittered = cpr._jitter_duplicate_points(points, np.random.RandomState(0))
        assert np.allclose(
            [point for i, point in enumerate(points_jittered) if i in [0, 1, 4, 6]],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
            rtol=0,
            atol=1e-5
        )
        assert np.linalg.norm(np.float32([0, 0]) - np.float32(points_jittered[2])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[3])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[5])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[7])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[8])) >= 1e-4
        assert np.linalg.norm(np.float32([1, 0]) - np.float32(points_jittered[9])) >= 1e-4

    def test__calculate_circumference(self):
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        circ = _ConcavePolygonRecoverer._calculate_circumference(points)
        assert np.allclose(circ, 4)

        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        circ = _ConcavePolygonRecoverer._calculate_circumference(points)
        assert np.allclose(circ, 4)

        points = np.float32([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        circ = _ConcavePolygonRecoverer._calculate_circumference(points)
        assert np.allclose(circ, 4)

        points = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        circ = _ConcavePolygonRecoverer._calculate_circumference(points)
        assert np.allclose(circ, 6)

    def test__fit_best_valid_polygon(self):
        def _assert_ids_match(observed, expected):
            assert len(observed) == len(expected), "len mismatch: %d vs %d" % (len(observed), len(expected))

            max_count = 0
            for i in range(len(observed)):
                counter = 0
                for j in range(i, i+len(expected)):
                    if observed[(i+j) % len(observed)] == expected[j % len(expected)]:
                        counter += 1
                    else:
                        break

                max_count = max(max_count, counter)

            assert max_count == len(expected), "count mismatch: %d vs %d" % (max_count, len(expected))

        cpr = _ConcavePolygonRecoverer()
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(points)
        assert np.allclose(points, points_fit)
        assert ia.Polygon(points_fit).is_valid

        # square-like, but top line has one point in its center which's
        # y-coordinate is below the bottom line
        points = [(0.0, 0.0), (0.45, 0.0), (0.5, 1.5), (0.55, 0.0), (1.0, 0.0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(points)
        _assert_ids_match(points_fit, [0, 1, 3, 4, 5, 2, 6])
        assert ia.Polygon([points[idx] for idx in points_fit]).is_valid

        # |--|  |--|
        # |  |  |  |
        # |  |  |  |
        # |--|--|--|
        #    |  |
        #    ----
        # the intersection points on the bottom line are not provided,
        # hence the result is expected to have triangles at the bottom left
        # and right
        points = [(0.0, 0), (0.25, 0), (0.25, 1.25),
                  (0.75, 1.25), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(points)
        _assert_ids_match(points_fit, [0, 1, 4, 5, 6, 3, 2, 7])
        poly_observed = ia.Polygon([points[idx] for idx in points_fit])
        assert poly_observed.is_valid

        # same as above, but intersection points at the bottom line are provided
        # without oversampling, i.e. incorporating these points would lead to an
        # invalid polygon
        points = [(0.0, 0), (0.25, 0), (0.25, 1.0), (0.25, 1.25),
                  (0.75, 1.25), (0.75, 1.0), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_fit = cpr._fit_best_valid_polygon(points)
        assert len(points_fit) >= len(points) - 2  # TODO add IoU check here
        poly_observed = ia.Polygon([points[idx] for idx in points_fit])
        assert poly_observed.is_valid

    def test__fix_polygon_is_line(self):
        cpr = _ConcavePolygonRecoverer()

        points = [(0, 0), (1, 0), (1, 1)]
        points_fixed = cpr._fix_polygon_is_line(points, np.random.RandomState(0))
        assert np.allclose(points_fixed, points, atol=0, rtol=0)

        points = [(0, 0), (1, 0), (2, 0)]
        points_fixed = cpr._fix_polygon_is_line(points, np.random.RandomState(0))
        assert not np.allclose(points_fixed, points, atol=0, rtol=0)
        assert not cpr._is_polygon_line(points_fixed)
        assert np.allclose(points_fixed, points, rtol=0, atol=1e-2)

        points = [(0, 0), (0, 1), (0, 2)]
        points_fixed = cpr._fix_polygon_is_line(points, np.random.RandomState(0))
        assert not np.allclose(points_fixed, points, atol=0, rtol=0)
        assert not cpr._is_polygon_line(points_fixed)
        assert np.allclose(points_fixed, points, rtol=0, atol=1e-2)

        points = [(0, 0), (1, 1), (2, 2)]
        points_fixed = cpr._fix_polygon_is_line(points, np.random.RandomState(0))
        assert not np.allclose(points_fixed, points, atol=0, rtol=0)
        assert not cpr._is_polygon_line(points_fixed)
        assert np.allclose(points_fixed, points, rtol=0, atol=1e-2)

    def test__is_polygon_line(self):
        points = [(0, 0), (1, 0), (1, 1)]
        assert not _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert not _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert not _ConcavePolygonRecoverer._is_polygon_line(points)

        points = np.float32([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert not _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0)]
        assert _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (2, 0)]
        assert _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (1, 0)]
        assert _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (1, 0), (2, 0)]
        assert _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (1, 0), (2, 0), (0.5, 0)]
        assert _ConcavePolygonRecoverer._is_polygon_line(points)

        points = [(0, 0), (1, 0), (1, 0), (2, 0), (1, 1)]
        assert not _ConcavePolygonRecoverer._is_polygon_line(points)

    def test__generate_intersection_points(self):
        cpr = _ConcavePolygonRecoverer()

        # triangle
        points = [(0.5, 0), (1, 1), (0, 1)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        assert points_inter == [[], [], []]

        # rotated square
        points = [(0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        assert points_inter == [[], [], [], []]

        # square
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        assert points_inter == [[], [], [], []]

        # |--|  |--|
        # |  |__|  |
        # |        |
        # |--------|
        points = [(0.0, 0), (0.25, 0), (0.25, 0.25),
                  (0.75, 0.25), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        assert points_inter == [[], [], [], [], [], [], [], []]

        # same as above, but middle part goes much further down,
        # crossing the bottom line
        points = [(0.0, 0), (0.25, 0), (0.25, 1.25),
                  (0.75, 1.25), (0.75, 0), (1.0, 0),
                  (1.0, 1.0), (0.0, 1.0)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [(0.25, 1.0)], [], [(0.75, 1.0)], [], [], [(0.75, 1.0), (0.25, 1.0)], []])

        # square-like structure with intersections in top right area
        points = [(0, 0), (0.5, 0), (1.01, 0.5), (1.0, 0), (1, 1), (0, 1), (0, 0)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [(1.0, 0.4902)], [], [(1.0, 0.4902)], [], [], []],
            atol=1e-2)

        # same as above, but with a second intersection in bottom left
        points = [(0, 0), (0.5, 0), (1.01, 0.5), (1.0, 0), (1, 1), (-0.25, 1),
                  (0, 1.25)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [(1.0, 0.4902)], [], [(1.0, 0.4902)], [(0, 1.0)], [], [(0, 1.0)]],
            atol=1e-2)

        # double triangle with point in center that is shared by both triangles
        points = [(0, 0), (0.5, 0.5), (1.0, 0), (1.0, 1.0), (0.5, 0.5), (0, 1.0)]
        points_inter = cpr._generate_intersection_points(points, one_point_per_intersection=False)
        self._assert_points_are_identical(
            points_inter,
            [[], [], [], [], [], []])

    def test__oversample_intersection_points(self):
        cpr = _ConcavePolygonRecoverer()
        cpr.oversampling = 0.1
        #cpr.oversample_startpoint = [0.9]
        #cpr.oversample_endpoint = [0.2]
        #cpr.oversample_both = [0.3, 0.6]

        points = [(0.0, 0.0), (1.0, 0.0)]
        segment_add_points_sorted = [[(0.5, 0.0)], []]
        points_oversampled = cpr._oversample_intersection_points(points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.55, 0.0)], []],
            atol=1e-4
        )

        points = [(0.0, 0.0), (2.0, 0.0)]
        segment_add_points_sorted = [[(0.5, 0.0)], []]
        points_oversampled = cpr._oversample_intersection_points(points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.65, 0.0)], []],
            atol=1e-4
        )

        points = [(0.0, 0.0), (1.0, 0.0)]
        segment_add_points_sorted = [[(0.5, 0.0), (0.6, 0.0)], []]
        points_oversampled = cpr._oversample_intersection_points(points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.51, 0.0), (0.59, 0.0), (0.6, 0.0), (0.64, 0.0)], []],
            atol=1e-4
        )

        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        segment_add_points_sorted = [[(0.5, 0.0)], [], [(0.8, 1.0)], [(0.0, 0.7)]]
        points_oversampled = cpr._oversample_intersection_points(points, segment_add_points_sorted)
        self._assert_points_are_identical(
            points_oversampled,
            [[(0.45, 0.0), (0.5, 0.0), (0.55, 0.0)],
             [],
             [(0.82, 1.0), (0.8, 1.0), (0.72, 1.0)],
             [(0.0, 0.73), (0.0, 0.7), (0.0, 0.63)]],
            atol=1e-4
        )

    def test__insert_intersection_points(self):
        points = [(0, 0), (1, 0), (2, 0)]
        segments_add_point_sorted = [[], [], []]
        points_inserted = _ConcavePolygonRecoverer._insert_intersection_points(
            points, segments_add_point_sorted)
        assert points_inserted == points

        segments_add_point_sorted = [[(0.5, 0)], [], []]
        points_inserted = _ConcavePolygonRecoverer._insert_intersection_points(
            points, segments_add_point_sorted)
        assert points_inserted == [(0, 0), (0.5, 0), (1, 0), (2, 0)]

        segments_add_point_sorted = [[(0.5, 0), (0.75, 0)], [], []]
        points_inserted = _ConcavePolygonRecoverer._insert_intersection_points(
            points, segments_add_point_sorted)
        assert points_inserted == [(0, 0), (0.5, 0), (0.75, 0), (1, 0), (2, 0)]

        segments_add_point_sorted = [[(0.5, 0)], [(1.5, 0)], []]
        points_inserted = _ConcavePolygonRecoverer._insert_intersection_points(
            points, segments_add_point_sorted)
        assert points_inserted == [(0, 0), (0.5, 0), (1, 0), (1.5, 0), (2, 0)]

        segments_add_point_sorted = [[(0.5, 0)], [(1.5, 0)], [(2.5, 0)]]
        points_inserted = _ConcavePolygonRecoverer._insert_intersection_points(
            points, segments_add_point_sorted)
        assert points_inserted == [(0, 0), (0.5, 0), (1, 0), (1.5, 0), (2, 0),
                                   (2.5, 0)]


if __name__ == "__main__":
    main()
