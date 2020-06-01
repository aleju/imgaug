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
from imgaug import dtypes as iadt
import imgaug.random as iarandom
from imgaug.testutils import assertWarns

# TODO clean up this file


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
    # test_seed()
    # test_current_random_state()
    # test_new_random_state()
    # test_dummy_random_state()
    # test_copy_random_state()
    # test_derive_random_state()
    # test_derive_random_states()
    # test_forward_random_state()
    # test_angle_between_vectors()
    test_compute_line_intersection_point()
    test_draw_text()
    test_imresize_many_images()
    test_imresize_single_image()
    test_pool()
    test_avg_pool()
    test_max_pool()
    test_min_pool()
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


class TestDeprecatedDataFunctions(unittest.TestCase):
    def test_quokka(self):
        with assertWarns(self, ia.DeprecationWarning):
            img = ia.quokka()
            assert ia.is_np_array(img)

    def test_quokka_square(self):
        with assertWarns(self, ia.DeprecationWarning):
            img = ia.quokka_square()
            assert ia.is_np_array(img)

    def test_quokka_heatmap(self):
        with assertWarns(self, ia.DeprecationWarning):
            result = ia.quokka_heatmap()
            assert isinstance(result, ia.HeatmapsOnImage)

    def test_quokka_segmentation_map(self):
        with assertWarns(self, ia.DeprecationWarning):
            result = ia.quokka_segmentation_map()
            assert isinstance(result, ia.SegmentationMapsOnImage)

    def test_quokka_keypoints(self):
        with assertWarns(self, ia.DeprecationWarning):
            result = ia.quokka_keypoints()
            assert isinstance(result, ia.KeypointsOnImage)

    def test_quokka_bounding_boxes(self):
        with assertWarns(self, ia.DeprecationWarning):
            result = ia.quokka_bounding_boxes()
            assert isinstance(result, ia.BoundingBoxesOnImage)


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

    class _Dummy3(object):
        def foo(self):
            pass

    class _Dummy4(object):
        @classmethod
        def foo(cls):
            pass

    class _Dummy5(object):
        @classmethod
        def foo(cls):
            pass

    values_true = [_dummy_func, _dummy_func2, _Dummy2(), _Dummy3().foo,
                   _Dummy4.foo, _Dummy5.foo]
    values_false = [
        "A", "BC", "1", "", -100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2,
        1e-4, True, False, (1.0, 2.0), [1.0, 2.0], _Dummy1(),
        np.zeros((1, 2), dtype=np.uint8)]
    for value in values_true:
        assert ia.is_callable(value) is True
    for value in values_false:
        assert ia.is_callable(value) is False


@mock.patch("imgaug.random.seed")
def test_seed(mock_seed):
    ia.seed(10017)
    mock_seed.assert_called_once_with(10017)


def test_current_random_state():
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        rng = ia.current_random_state()

    assert rng.is_global_rng()
    assert len(caught_warnings) == 1
    assert "is deprecated" in str(caught_warnings[-1].message)


@mock.patch("imgaug.random.RNG")
def test_new_random_state__induce_pseudo_random(mock_rng):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        _ = ia.new_random_state(seed=None, fully_random=False)

    assert mock_rng.create_pseudo_random_.call_count == 1
    assert len(caught_warnings) == 1
    assert "is deprecated" in str(caught_warnings[-1].message)


@mock.patch("imgaug.random.RNG")
def test_new_random_state__induce_fully_random(mock_rng):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        _ = ia.new_random_state(seed=None, fully_random=True)

    assert mock_rng.create_fully_random.call_count == 1
    assert len(caught_warnings) == 1
    assert "is deprecated" in str(caught_warnings[-1].message)


@mock.patch("imgaug.random.RNG")
def test_new_random_state__use_seed(mock_rng):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        _ = ia.new_random_state(seed=1)

    mock_rng.assert_called_once_with(1)
    assert len(caught_warnings) == 1
    assert "is deprecated" in str(caught_warnings[-1].message)


@mock.patch("imgaug.random.RNG")
def test_dummy_random_state(mock_rng):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        _ = ia.dummy_random_state()

    mock_rng.assert_called_once_with(1)
    assert len(caught_warnings) == 1
    assert "is deprecated" in str(caught_warnings[-1].message)


@mock.patch("imgaug.random.copy_generator")
@mock.patch("imgaug.random.copy_generator_unless_global_generator")
def test_copy_random_state__not_global(mock_copy_gen_glob, mock_copy_gen):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        gen = iarandom.convert_seed_to_generator(1)
        _ = ia.copy_random_state(gen, force_copy=False)

    assert mock_copy_gen.call_count == 0
    mock_copy_gen_glob.assert_called_once_with(gen)
    assert len(caught_warnings) == 1
    assert "is deprecated" in str(caught_warnings[-1].message)


@mock.patch("imgaug.random.copy_generator")
@mock.patch("imgaug.random.copy_generator_unless_global_generator")
def test_copy_random_state__also_global(mock_copy_gen_glob, mock_copy_gen):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        gen = iarandom.convert_seed_to_generator(1)
        _ = ia.copy_random_state(gen, force_copy=True)

    mock_copy_gen.assert_called_once_with(gen)
    assert mock_copy_gen_glob.call_count == 0
    assert len(caught_warnings) == 1
    assert "is deprecated" in str(caught_warnings[-1].message)


@mock.patch("imgaug.random.derive_generator_")
def test_derive_random_state(mock_derive):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        gen = iarandom.convert_seed_to_generator(1)
        _ = ia.derive_random_state(gen)

    mock_derive.assert_called_once_with(gen)
    assert len(caught_warnings) == 1
    assert "is deprecated" in str(caught_warnings[-1].message)


@mock.patch("imgaug.random.derive_generators_")
def test_derive_random_states(mock_derive):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        gen = iarandom.convert_seed_to_generator(1)
        _ = ia.derive_random_states(gen, n=2)

    mock_derive.assert_called_once_with(gen, n=2)
    assert len(caught_warnings) == 1
    assert "is deprecated" in str(caught_warnings[-1].message)


@mock.patch("imgaug.random.advance_generator_")
def test_forward_random_state(mock_advance):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        
        gen = iarandom.convert_seed_to_generator(1)
        _ = ia.forward_random_state(gen)

    mock_advance.assert_called_once_with(gen)
    assert len(caught_warnings) == 1
    assert "is deprecated" in str(caught_warnings[-1].message)


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
    shapes = [(0, 4, 3), (4, 0, 3), (0, 0, 3)]
    for shape in shapes:
        images = [np.zeros(shape, dtype=np.uint8)]
        got_exception = False
        try:
            _ = ia.imresize_many_images(images, sizes=(2, 2))
        except Exception as exc:
            assert (
                "Cannot resize images, because at least one image has a height "
                "and/or width and/or number of channels of zero."
                in str(exc)
            )
            got_exception = True
        assert got_exception

    # test invalid sizes
    sizes_all = [(-1, 2)]
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
        + [-1.0, -1]
    for sizes in sizes_all:
        images = [np.zeros((4, 4, 3), dtype=np.uint8)]
        got_exception = False
        try:
            _ = ia.imresize_many_images(images, sizes=sizes)
        except Exception as exc:
            assert ">= 0" in str(exc)
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
        mask = mask.astype(np.float64) / 255.0

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
        mask = mask.astype(np.float64)

        for dtype in [np.float16, np.float32, np.float64]:
            isize = np.dtype(dtype).itemsize

            for value in [0.5, -0.5, 1.0, -1.0, 10.0, -10.0, -1000 ** (isize-1), 1000 * (isize+1)]:
                image = np.zeros((4, 4), dtype=dtype)
                image[1, :] = value
                image[2, :] = value
                expected = (mask * np.float64(value)).astype(dtype)
                image_rs = ia.imresize_many_images([image], (3, 3), interpolation=ip)[0]
                assert image_rs.dtype.type == dtype
                # Our basis for the expected image is derived from uint8 as that is most likely to work, so we will
                # have to accept here deviations of around 1/255.
                atol = np.float64(1 / 255) * np.abs(np.float64(value)) + 1e-8
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
    try:
        high_res_dt = np.float128
        dtypes = ["float16", "float32", "float64", "float128"]
    except AttributeError:
        high_res_dt = np.float64
        dtypes = ["float16", "float32", "float64"]

    for dtype in dtypes:
        dtype = np.dtype(dtype)

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
                assert _allclose(arr_pooled, high_res_dt(value))

                arr = np.full((4, 4, 3), value, dtype=dtype)
                arr_pooled = ia.pool(arr, 2, func)
                assert arr_pooled.shape == (2, 2, 3)
                assert arr_pooled.dtype == np.dtype(dtype)
                assert _allclose(arr_pooled, high_res_dt(value))

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
    arr_pooled = ia.pool(arr, 2, np.average, pad_cval=22)
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.average([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.average([2, 22, 6, 22]))
    assert arr_pooled[1, 0] == int(np.average([8, 9, 22, 22]))
    assert arr_pooled[1, 1] == int(np.average([10, 22, 22, 22]))

    # padding mode
    arr = np.uint8([
        [0, 1, 2],
        [4, 5, 6],
        [8, 9, 10]
    ])
    arr_pooled = ia.pool(arr, 2, np.average, pad_mode="edge")
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.average([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.average([2, 2, 6, 6]))
    assert arr_pooled[1, 0] == int(np.average([8, 9, 8, 9]))
    assert arr_pooled[1, 1] == int(np.average([10, 10, 10, 10]))

    # same as above, but with float32 to make averages more accurate
    arr = np.float32([
        [0, 1, 2],
        [4, 5, 6],
        [8, 9, 10]
    ])
    arr_pooled = ia.pool(arr, 2, np.average, pad_mode="edge")
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert np.isclose(arr_pooled[0, 0], np.average([0, 1, 4, 5]))
    assert np.isclose(arr_pooled[0, 1], np.average([2, 2, 6, 6]))
    assert np.isclose(arr_pooled[1, 0], np.average([8, 9, 8, 9]))
    assert np.isclose(arr_pooled[1, 1], np.average([10, 10, 10, 10]))


# TODO add test that verifies the default padding mode
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
    # add 1e-4 here to force 0.5 to be rounded up, as that's how OpenCV
    # handles it
    assert arr_pooled[0, 0] == int(np.round(1e-4 + np.average([0, 1, 4, 5])))
    assert arr_pooled[0, 1] == int(np.round(1e-4 + np.average([2, 3, 6, 7])))
    assert arr_pooled[1, 0] == int(np.round(1e-4 + np.average([8, 9, 12, 13])))
    assert arr_pooled[1, 1] == int(np.round(1e-4 + np.average([10, 11, 14, 15])))


# TODO add test that verifies the default padding mode
def test_max_pool():
    # very basic test, as max_pool() just calls pool(), which is tested in
    # test_pool()
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


# TODO add test that verifies the default padding mode
def test_min_pool():
    # very basic test, as min_pool() just calls pool(), which is tested in
    # test_pool()
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])

    arr_pooled = ia.min_pool(arr, 2)

    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.min([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.min([2, 3, 6, 7]))
    assert arr_pooled[1, 0] == int(np.min([8, 9, 12, 13]))
    assert arr_pooled[1, 1] == int(np.min([10, 11, 14, 15]))


# TODO add test that verifies the default padding mode
def test_median_pool():
    # very basic test, as median_pool() just calls pool(), which is tested in
    # test_pool()
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])

    arr_pooled = ia.median_pool(arr, 2)

    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.median([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.median([2, 3, 6, 7]))
    assert arr_pooled[1, 0] == int(np.median([8, 9, 12, 13]))
    assert arr_pooled[1, 1] == int(np.median([10, 11, 14, 15]))


# TODO add test that verifies the default padding mode
def test_median_pool_ksize_1_3():
    # very basic test, as median_pool() just calls pool(), which is tested in
    # test_pool()
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])

    arr_pooled = ia.median_pool(arr, (1, 3))

    assert arr_pooled.shape == (4, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.median([0, 1, 2]))
    assert arr_pooled[0, 1] == int(np.median([3, 2, 1]))
    assert arr_pooled[1, 0] == int(np.median([4, 5, 6]))
    assert arr_pooled[1, 1] == int(np.median([7, 6, 5]))
    assert arr_pooled[2, 0] == int(np.median([8, 9, 10]))
    assert arr_pooled[2, 1] == int(np.median([11, 10, 9]))
    assert arr_pooled[3, 0] == int(np.median([12, 13, 14]))
    assert arr_pooled[3, 1] == int(np.median([15, 14, 13]))


def test_median_pool_ksize_3():
    # After padding:
    # [5, 4, 5, 6, 7, 6],
    # [1, 0, 1, 2, 3, 2],
    # [5, 4, 5, 6, 7, 6],
    # [9, 8, 9, 10, 11, 10],
    # [13, 12, 13, 14, 15, 14],
    # [9, 8, 9, 10, 11, 10]
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])

    arr_pooled = ia.median_pool(arr, 3)

    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.median([5, 4, 5, 1, 0, 1, 5, 4, 5]))
    assert arr_pooled[0, 1] == int(np.median([6, 7, 6, 2, 3, 2, 6, 7, 6]))
    assert arr_pooled[1, 0] == int(np.median([9, 8, 9, 13, 12, 13, 9, 8, 9]))
    assert arr_pooled[1, 1] == int(np.median([10, 11, 10, 14, 15, 13, 10, 11,
                                              10]))


def test_median_pool_ksize_3_view():
    # After padding:
    # [5, 4, 5, 6, 7, 6],
    # [1, 0, 1, 2, 3, 2],
    # [5, 4, 5, 6, 7, 6],
    # [9, 8, 9, 10, 11, 10],
    # [13, 12, 13, 14, 15, 14],
    # [9, 8, 9, 10, 11, 10]
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [0, 0, 0, 0]
    ])

    arr_in = arr[0:4, :]
    assert arr_in.flags["OWNDATA"] is False
    assert arr_in.flags["C_CONTIGUOUS"] is True
    arr_pooled = ia.median_pool(arr_in, 3)

    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.median([5, 4, 5, 1, 0, 1, 5, 4, 5]))
    assert arr_pooled[0, 1] == int(np.median([6, 7, 6, 2, 3, 2, 6, 7, 6]))
    assert arr_pooled[1, 0] == int(np.median([9, 8, 9, 13, 12, 13, 9, 8, 9]))
    assert arr_pooled[1, 1] == int(np.median([10, 11, 10, 14, 15, 13, 10, 11,
                                              10]))


def test_median_pool_ksize_3_non_contiguous():
    # After padding:
    # [5, 4, 5, 6, 7, 6],
    # [1, 0, 1, 2, 3, 2],
    # [5, 4, 5, 6, 7, 6],
    # [9, 8, 9, 10, 11, 10],
    # [13, 12, 13, 14, 15, 14],
    # [9, 8, 9, 10, 11, 10]
    arr = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ], dtype=np.uint8, order="F")

    assert arr.flags["OWNDATA"] is True
    assert arr.flags["C_CONTIGUOUS"] is False
    arr_pooled = ia.median_pool(arr, 3)

    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.median([5, 4, 5, 1, 0, 1, 5, 4, 5]))
    assert arr_pooled[0, 1] == int(np.median([6, 7, 6, 2, 3, 2, 6, 7, 6]))
    assert arr_pooled[1, 0] == int(np.median([9, 8, 9, 13, 12, 13, 9, 8, 9]))
    assert arr_pooled[1, 1] == int(np.median([10, 11, 10, 14, 15, 13, 10, 11,
                                              10]))


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
    try:
        _high_res_dt = np.float128
        dtypes = ["float16", "float32", "float64", "float128"]
    except AttributeError:
        _high_res_dt = np.float64
        dtypes = ["float16", "float32", "float64"]

    for dtype in dtypes:
        dtype = np.dtype(dtype)

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


class Test_apply_lut(unittest.TestCase):
    def test_2d_image(self):
        table = np.mod(np.arange(256) + 10, 256).astype(np.uint8)

        image = np.uint8([
            [0, 50, 100, 245, 254, 255],
            [1, 51, 101, 246, 255, 0]
        ])

        image_aug = ia.apply_lut(image, table)

        expected = np.uint8([
            [10, 60, 110, 255, 8, 9],
            [11, 61, 111, 0, 9, 10]
        ])
        assert np.array_equal(image_aug, expected)
        assert image_aug is not image
        assert image_aug.shape == (2, 6)
        assert image_aug.dtype.name == "uint8"


class Test_apply_lut_(unittest.TestCase):
    def test_2d_image(self):
        table = np.mod(np.arange(256) + 10, 256).astype(np.uint8)
        tables = [
            ("array-1d", table),
            ("array-2d", table[:, np.newaxis]),
            ("array-3d", table[np.newaxis, :, np.newaxis]),
            ("list", [table])
        ]

        for subtable_descr, subtable in tables:
            with self.subTest(table_type=subtable_descr):
                image = np.uint8([
                    [0, 50, 100, 245, 254, 255],
                    [1, 51, 101, 246, 255, 0]
                ])

                image_aug = ia.apply_lut_(image, subtable)

                expected = np.uint8([
                    [10, 60, 110, 255, 8, 9],
                    [11, 61, 111, 0, 9, 10]
                ])
                assert np.array_equal(image_aug, expected)
                assert image_aug is image
                assert image_aug.shape == (2, 6)
                assert image_aug.dtype.name == "uint8"

    def test_HW1_image(self):
        table = np.mod(np.arange(256) + 10, 256).astype(np.uint8)
        tables = [
            ("array-1d", table),
            ("array-2d", table[:, np.newaxis]),
            ("array-3d", table[np.newaxis, :, np.newaxis]),
            ("list", [table])
        ]

        for subtable_descr, subtable in tables:
            with self.subTest(table_type=subtable_descr):
                image = np.uint8([
                    [0, 50, 100, 245, 254, 255],
                    [1, 51, 101, 246, 255, 0]
                ])
                image = image[:, :, np.newaxis]

                image_aug = ia.apply_lut_(image, subtable)

                expected = np.uint8([
                    [10, 60, 110, 255, 8, 9],
                    [11, 61, 111, 0, 9, 10]
                ])
                expected = expected[:, :, np.newaxis]
                assert np.array_equal(image_aug, expected)
                # (H,W,1) images always lead to a copy
                assert image_aug is not image
                assert image_aug.shape == (2, 6, 1)
                assert image_aug.dtype.name == "uint8"

    def test_HWC_image(self):
        # Base table, mapping all values to value+10.
        # For channels C>0 we additionally add +C below.
        table_base = np.mod(np.arange(256) + 10, 256).astype(np.int32)
        nb_channels_lst = [2, 3, 4, 5, 511, 512, 513, 512*2-1, 512*2, 512*2+1]

        for nb_channels in nb_channels_lst:
            # Create channelwise LUT.
            tables = []
            for c in np.arange(nb_channels):
                tables.append(np.mod(table_base + c, 256).astype(np.uint8))

            tables_by_type = [
                ("array-1d", table_base.astype(np.uint8)),
                ("array-2d", np.stack(tables, axis=-1)),
                ("array-3d", np.stack(tables, axis=-1).reshape((1, 256, -1))),
                ("list", tables)
            ]

            for subtable_descr, subtable in tables_by_type:
                with self.subTest(nb_channels=nb_channels,
                                  table_type=subtable_descr):
                    # Create a normalized lut table, so that we can easily
                    # find the projected value via x,y,c coordinates.
                    # In case of array-1d, all channels are treated the same
                    # way.
                    if subtable_descr == "array-1d":
                        tables_3d = np.stack([table_base] * nb_channels,
                                             axis=-1)
                    else:
                        tables_3d = np.stack(tables, axis=-1).reshape(
                            (256, -1))

                    image = np.int32([
                        [0, 50, 100, 245, 254, 255],
                        [1, 51, 101, 246, 255, 0]
                    ])
                    image = image[:, :, np.newaxis]
                    image = np.tile(image, (1, 1, nb_channels))
                    for c in np.arange(nb_channels):
                        image[:, :, c] += c
                    image = np.mod(image, 256).astype(np.uint8)
                    image_orig = np.copy(image)

                    image_aug = ia.apply_lut_(image, subtable)

                    # Reproduce effect of a LUT mapping on the input
                    # image.
                    expected = np.zeros_like(image_orig)
                    for c in np.arange(nb_channels):
                        for x in np.arange(image.shape[1]):
                            for y in np.arange(image.shape[0]):
                                v = image_orig[y, x, c]
                                v_proj = tables_3d[v, c]
                                expected[y, x, c] = v_proj

                    assert np.array_equal(image_aug, expected)
                    if nb_channels < 512:
                        assert image_aug is image
                    assert image_aug.shape == (2, 6, nb_channels)
                    assert image_aug.dtype.name == "uint8"

    def test_image_is_noncontiguous(self):
        table = np.mod(np.arange(256) + 10, 256).astype(np.uint8)

        image = np.uint8([
            [0, 50, 100, 245, 254, 255],
            [1, 51, 101, 246, 255, 0]
        ])
        image = np.fliplr(image)
        assert image.flags["C_CONTIGUOUS"] is False

        image_aug = ia.apply_lut_(image, table)

        expected = np.uint8([
            [10, 60, 110, 255, 8, 9],
            [11, 61, 111, 0, 9, 10]
        ])
        assert np.array_equal(np.fliplr(image_aug), expected)
        assert image_aug is not image  # non-contiguous should lead to copy
        assert image_aug.shape == (2, 6)
        assert image_aug.dtype.name == "uint8"

    def test_image_is_view(self):
        table = np.mod(np.arange(256) + 10, 256).astype(np.uint8)

        image = np.uint8([
            [0, 50, 100, 245, 254, 255],
            [1, 51, 101, 246, 255, 0]
        ])
        image = image[:, 1:4]
        assert image.flags["OWNDATA"] is False

        image_aug = ia.apply_lut_(image, table)

        expected = np.uint8([
            [60, 110, 255],
            [61, 111, 0]
        ])
        assert np.array_equal(image_aug, expected)
        assert image_aug is not image  # non-owndata should lead to copy
        assert image_aug.shape == (2, 3)
        assert image_aug.dtype.name == "uint8"

    def test_zero_sized_axes(self):
        table = np.mod(np.arange(256) + 10, 256).astype(np.uint8)
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
                image = np.zeros(shape, dtype=np.uint8)
                image_aug = ia.apply_lut_(image, table)
                assert image_aug.shape == shape
                assert image_aug.dtype.name == "uint8"


if __name__ == "__main__":
    main()
