"""
Automatically run tests for this library.
Simply execute
    python test.py
or execute
    nosetests --verbose
from within tests/
or add @attr("now") in front of a test and then execute
    nosetests --verbose -a now
to only execute a specific test.
"""
from __future__ import print_function, division

# fix execution of tests involving matplotlib on travis
import matplotlib
matplotlib.use('Agg')

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
import random
import six
import six.moves as sm
import skimage
from skimage import data, color
import cv2
import time
import scipy
import copy
import warnings

#from nose.plugins.attrib import attr


def main():
    time_start = time.time()

    # ----------------------
    # imgaug
    # ----------------------
    is_np_array()
    test_is_single_integer()
    test_is_single_float()
    test_is_single_number()
    test_is_iterable()
    test_is_string()
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
    # test_quokka()
    # test_quokka_square()
    # test_angle_between_vectors()
    # test_draw_text()
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
    # test_HeatmapsOnImage_pad_to_aspect_ratio()
    test_HeatmapsOnImage_avg_pool()
    test_HeatmapsOnImage_max_pool()
    test_HeatmapsOnImage_scale()
    # test_HeatmapsOnImage_to_uint8()
    # test_HeatmapsOnImage_from_uint8()
    # test_HeatmapsOnImage_from_0to1()
    # test_HeatmapsOnImage_change_normalization()
    # test_HeatmapsOnImage_copy()
    # test_HeatmapsOnImage_deepcopy()
    test_SegmentationMapOnImage_bool()
    test_SegmentationMapOnImage_get_arr_int()
    #test_SegmentationMapOnImage_get_arr_bool()
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
    test_BatchLoader()
    # test_BackgroundAugmenter.get_batch()
    # test_BackgroundAugmenter._augment_images_worker()
    # test_BackgroundAugmenter.terminate()

    # ----------------------
    # augmenters
    # ----------------------
    # arithmetic
    test_Add()
    test_AddElementwise()
    test_AdditiveGaussianNoise()
    test_Multiply()
    test_MultiplyElementwise()
    test_Dropout()
    test_CoarseDropout()
    test_SaltAndPepper()
    test_CoarseSaltAndPepper()
    test_Salt()
    test_CoarseSalt()
    test_Pepper()
    test_CoarsePepper()
    test_ReplaceElementwise()
    test_Invert()
    test_ContrastNormalization()

    # blur
    test_GaussianBlur()
    test_AverageBlur()
    test_MedianBlur()
    # TODO BilateralBlur

    # color
    # TODO WithColorspace
    test_AddToHueAndSaturation()
    # TODO ChangeColorspace
    test_Grayscale()

    # convolutional
    test_Convolve()
    test_Sharpen()
    test_Emboss()
    # TODO EdgeDetect
    # TODO DirectedEdgeDetect

    # flip
    test_Fliplr()
    test_Flipud()

    # geometric
    test_Affine()
    test_AffineCv2()
    test_PiecewiseAffine()
    test_PerspectiveTransform()
    test_ElasticTransformation()

    # meta
    test_copy_dtypes_for_restore()
    test_restore_augmented_image_dtype_()
    test_restore_augmented_image_dtype()
    test_restore_augmented_images_dtypes_()
    test_restore_augmented_images_dtypes()
    test_clip_augmented_image_()
    test_clip_augmented_image()
    test_clip_augmented_images_()
    test_clip_augmented_images()
    test_reduce_to_nonempty()
    test_invert_reduce_to_nonempty()
    test_Augmenter()
    test_Augmenter_augment_keypoints()
    test_Augmenter_augment_segmentation_maps()
    test_Augmenter_find()
    test_Augmenter_remove()
    test_Augmenter_hooks()
    test_Augmenter_copy_random_state()
    test_Augmenter_augment_batches()
    test_Sequential()
    test_SomeOf()
    test_OneOf()
    test_Sometimes()
    test_WithChannels()
    test_Noop()
    test_Lambda()
    test_AssertLambda()
    test_AssertShape()

    # overlay
    test_Alpha()
    test_AlphaElementwise()
    # TODO SimplexNoiseAlpha
    # TODO FrequencyNoiseAlpha

    # segmentation
    test_Superpixels()

    # size
    test_Scale()
    # TODO test_CropAndPad()
    test_Pad()
    test_Crop()
    # TODO test_PadToFixedSize()
    # TODO test_CropToFixedSize()

    # these functions use various augmenters, so test them last
    test_2d_inputs()
    test_determinism()
    test_keypoint_augmentation()
    test_unusual_channel_numbers()
    test_dtype_preservation()

    # ----------------------
    # parameters
    # ----------------------
    test_parameters_handle_continuous_param()
    test_parameters_handle_discrete_param()
    test_parameters_handle_probability_param()
    test_parameters_force_np_float_dtype()
    test_parameters_both_np_float_if_one_is_float()
    test_parameters_draw_distribution_grid()
    test_parameters_draw_distribution_graph()
    test_parameters_Biomial()
    test_parameters_Choice()
    test_parameters_DiscreteUniform()
    test_parameters_Poisson()
    test_parameters_Normal()
    test_parameters_Laplace()
    test_parameters_ChiSquare()
    test_parameters_Weibull()
    test_parameters_Uniform()
    test_parameters_Beta()
    test_parameters_Deterministic()
    test_parameters_FromLowerResolution()
    test_parameters_Clip()
    test_parameters_Discretize()
    test_parameters_Multiply()
    test_parameters_Divide()
    test_parameters_Add()
    test_parameters_Subtract()
    test_parameters_Power()
    test_parameters_Absolute()
    test_parameters_RandomSign()
    test_parameters_ForceSign()
    test_parameters_Positive()
    test_parameters_Negative()
    test_parameters_IterativeNoiseAggregator()
    test_parameters_Sigmoid()
    #test_parameters_SimplexNoise()
    #test_parameters_FrequencyNoise()
    test_parameters_operators()
    test_parameters_copy()

    time_end = time.time()
    print("Finished without errors in %.4fs." % (time_end - time_start,))


def is_np_array():
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
        assert ia.is_np_array(value) == True
    for value in values_false:
        assert ia.is_np_array(value) == False


def test_is_single_integer():
    assert ia.is_single_integer("A") == False
    assert ia.is_single_integer(None) == False
    assert ia.is_single_integer(1.2) == False
    assert ia.is_single_integer(1.0) == False
    assert ia.is_single_integer(np.ones((1,), dtype=np.float32)[0]) == False
    assert ia.is_single_integer(1) == True
    assert ia.is_single_integer(1234) == True
    assert ia.is_single_integer(np.ones((1,), dtype=np.uint8)[0]) == True
    assert ia.is_single_integer(np.ones((1,), dtype=np.int32)[0]) == True


def test_is_single_float():
    assert ia.is_single_float("A") == False
    assert ia.is_single_float(None) == False
    assert ia.is_single_float(1.2) == True
    assert ia.is_single_float(1.0) == True
    assert ia.is_single_float(np.ones((1,), dtype=np.float32)[0]) == True
    assert ia.is_single_float(1) == False
    assert ia.is_single_float(1234) == False
    assert ia.is_single_float(np.ones((1,), dtype=np.uint8)[0]) == False
    assert ia.is_single_float(np.ones((1,), dtype=np.int32)[0]) == False


def test_caller_name():
    assert ia.caller_name() == 'test_caller_name'


def test_is_single_number():
    class _Dummy(object):
        pass
    values_true = [-100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2, 1e-4]
    values_false = ["A", "BC", "1", True, False, (1.0, 2.0), [1.0, 2.0], _Dummy(), np.zeros((1, 2), dtype=np.uint8)]
    for value in values_true:
        assert ia.is_single_number(value) == True
    for value in values_false:
        assert ia.is_single_number(value) == False


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
        assert ia.is_iterable(value) == True, value
    for value in values_false:
        assert ia.is_iterable(value) == False


def test_is_string():
    class _Dummy(object):
        pass
    values_true = ["A", "BC", "1", ""]
    values_false = [-100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2, 1e-4, True, False, (1.0, 2.0), [1.0, 2.0], _Dummy(), np.zeros((1, 2), dtype=np.uint8)]
    for value in values_true:
        assert ia.is_string(value) == True
    for value in values_false:
        assert ia.is_string(value) == False


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
        assert ia.is_integer_array(value) == True
    for value in values_false:
        assert ia.is_integer_array(value) == False


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
        assert ia.is_float_array(value) == True
    for value in values_false:
        assert ia.is_float_array(value) == False


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
    values_false = ["A", "BC", "1", "", -100, 1, 0, 1, 100, -1.2, -0.001, 0.0, 0.001, 1.2, 1e-4, True, False, (1.0, 2.0), [1.0, 2.0], _Dummy1(), np.zeros((1, 2), dtype=np.uint8)]
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
    rs_expected = np.random.RandomState(np.random.RandomState(seed).randint(0, 10**6, 1)[0])
    assert rs_observed.randint(0, 10**6) == rs_expected.randint(0, 10**6)
    rs_observed1 = ia.new_random_state(seed=None, fully_random=False)
    rs_observed2 = ia.new_random_state(seed=None, fully_random=False)
    assert rs_observed1.randint(0, 10**6) != rs_observed2.randint(0, 10**6)

    ia.seed(seed)
    np.random.seed(seed)
    rs_observed = ia.new_random_state(seed=None, fully_random=True)
    rs_not_expected = np.random.RandomState(np.random.RandomState(seed).randint(0, 10**6, 1)[0])
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
    rs = np.random.RandomState(1017)
    rs_observed = ia.derive_random_state(np.random.RandomState(1017))
    rs_expected = np.random.RandomState(np.random.RandomState(1017).randint(0, 10**6))
    assert rs_observed.randint(0, 10**6) == rs_expected.randint(0, 10**6)


def test_derive_random_states():
    rs = np.random.RandomState(1017)
    rs_observed1, rs_observed2 = ia.derive_random_states(np.random.RandomState(1017), n=2)
    seed = np.random.RandomState(1017).randint(0, 10**6)
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


def test_imresize_many_images():
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
        interpolations = [None,
                          "nearest", "linear", "area", "cubic",
                          cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]

        for interpolation in interpolations:
            images_same_observed = ia.imresize_many_images(images, (16, 16), interpolation=interpolation)
            for image_expected, image_observed in zip(images, images_same_observed):
                diff = np.abs(image_expected.astype(np.int32) - image_observed.astype(np.int32))
                assert np.sum(diff) == 0

        for interpolation in interpolations:
            images_small_observed = ia.imresize_many_images(images, (8, 8), interpolation=interpolation)
            for image_expected, image_observed in zip(images_small, images_small_observed):
                diff = np.abs(image_expected.astype(np.int32) - image_observed.astype(np.int32))
                diff_fraction = np.sum(diff) / (image_observed.size * 255)
                assert diff_fraction < 0.5

        for interpolation in interpolations:
            images_large_observed = ia.imresize_many_images(images, (32, 32), interpolation=interpolation)
            for image_expected, image_observed in zip(images_large, images_large_observed):
                diff = np.abs(image_expected.astype(np.int32) - image_observed.astype(np.int32))
                diff_fraction = np.sum(diff) / (image_observed.size * 255)
                assert diff_fraction < 0.5


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
    # uint8, int32
    # -------
    for dtype in [np.uint8, np.int32]:
        arr = np.zeros((3, 3), dtype=dtype) + 255

        arr_pad = ia.pad(arr)
        assert arr_pad.shape == (3, 3)
        assert arr_pad.dtype.type == dtype
        assert np.array_equal(arr_pad, arr)

        arr_pad = ia.pad(arr, top=1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype.type == dtype
        assert np.all(arr_pad[0, :] == 0)

        arr_pad = ia.pad(arr, right=1)
        assert arr_pad.shape == (3, 4)
        assert arr_pad.dtype.type == dtype
        assert np.all(arr_pad[:, -1] == 0)

        arr_pad = ia.pad(arr, bottom=1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype.type == dtype
        assert np.all(arr_pad[-1, :] == 0)

        arr_pad = ia.pad(arr, left=1)
        assert arr_pad.shape == (3, 4)
        assert arr_pad.dtype.type == dtype
        assert np.all(arr_pad[:, 0] == 0)

        arr_pad = ia.pad(arr, top=1, right=2, bottom=3, left=4)
        assert arr_pad.shape == (3+(1+3), 3+(2+4))
        assert arr_pad.dtype.type == dtype
        assert np.all(arr_pad[0, :] == 0)
        assert np.all(arr_pad[:, -2:] == 0)
        assert np.all(arr_pad[-3:, :] == 0)
        assert np.all(arr_pad[:, :4] == 0)

        arr_pad = ia.pad(arr, top=1, cval=10)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype.type == dtype
        assert np.all(arr_pad[0, :] == 10)

        arr = np.zeros((3, 3, 3), dtype=dtype) + 128
        arr_pad = ia.pad(arr, top=1)
        assert arr_pad.shape == (4, 3, 3)
        assert arr_pad.dtype.type == dtype
        assert np.all(arr_pad[0, :, 0] == 0)
        assert np.all(arr_pad[0, :, 1] == 0)
        assert np.all(arr_pad[0, :, 2] == 0)

        arr = np.zeros((3, 3), dtype=dtype) + 128
        arr[1, 1] = 200
        arr_pad = ia.pad(arr, top=1, mode="maximum")
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype.type == dtype
        assert arr_pad[0, 0] == 128
        assert arr_pad[0, 1] == 200
        assert arr_pad[0, 2] == 128

    # -------
    # float32, float64
    # -------
    for dtype in [np.float32, np.float64]:
        arr = np.zeros((3, 3), dtype=dtype) + 1.0

        arr_pad = ia.pad(arr)
        assert arr_pad.shape == (3, 3)
        assert arr_pad.dtype.type == dtype
        assert np.allclose(arr_pad, arr)

        arr_pad = ia.pad(arr, top=1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype.type == dtype
        assert np.allclose(arr_pad[0, :], dtype([0, 0, 0]))

        arr_pad = ia.pad(arr, right=1)
        assert arr_pad.shape == (3, 4)
        assert arr_pad.dtype.type == dtype
        assert np.allclose(arr_pad[:, -1], dtype([0, 0, 0]))

        arr_pad = ia.pad(arr, bottom=1)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype.type == dtype
        assert np.allclose(arr_pad[-1, :], dtype([0, 0, 0]))

        arr_pad = ia.pad(arr, left=1)
        assert arr_pad.shape == (3, 4)
        assert arr_pad.dtype.type == dtype
        assert np.allclose(arr_pad[:, 0], dtype([0, 0, 0]))

        arr_pad = ia.pad(arr, top=1, right=2, bottom=3, left=4)
        assert arr_pad.shape == (3+(1+3), 3+(2+4))
        assert arr_pad.dtype.type == dtype
        assert 0 - 1e-6 < np.max(arr_pad[0, :]) < 0 + 1e-6
        assert 0 - 1e-6 < np.max(arr_pad[:, -2:]) < 0 + 1e-6
        assert 0 - 1e-6 < np.max(arr_pad[-3, :]) < 0 + 1e-6
        assert 0 - 1e-6 < np.max(arr_pad[:, :4]) < 0 + 1e-6

        arr_pad = ia.pad(arr, top=1, cval=0.2)
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype.type == dtype
        assert np.allclose(arr_pad[0, :], dtype([0.2, 0.2, 0.2]))

        arr = np.zeros((3, 3, 3), dtype=dtype) + 0.5
        arr_pad = ia.pad(arr, top=1)
        assert arr_pad.shape == (4, 3, 3)
        assert arr_pad.dtype.type == dtype
        assert np.allclose(arr_pad[0, :, 0], dtype([0, 0, 0]))
        assert np.allclose(arr_pad[0, :, 1], dtype([0, 0, 0]))
        assert np.allclose(arr_pad[0, :, 2], dtype([0, 0, 0]))

        arr = np.zeros((3, 3), dtype=dtype) + 0.5
        arr[1, 1] = 0.75
        arr_pad = ia.pad(arr, top=1, mode="maximum")
        assert arr_pad.shape == (4, 3)
        assert arr_pad.dtype.type == dtype
        assert 0.50 - 1e-6 < arr_pad[0, 0] < 0.50 + 1e-6
        assert 0.75 - 1e-6 < arr_pad[0, 1] < 0.75 + 1e-6
        assert 0.50 - 1e-6 < arr_pad[0, 2] < 0.50 + 1e-6


def test_compute_paddings_for_aspect_ratio():
    arr = np.zeros((4, 4), dtype=np.uint8)
    top, right, bottom, left = ia.compute_paddings_for_aspect_ratio(arr, 1.0)
    assert top == 0
    assert right == 0
    assert bottom == 0
    assert left == 0

    arr = np.zeros((1, 4), dtype=np.uint8)
    top, right, bottom, left = ia.compute_paddings_for_aspect_ratio(arr, 1.0)
    assert top == 2
    assert right == 0
    assert bottom == 1
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


def test_pool():
    # basic functionality with uint8, int32, float32
    arr = np.uint8([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    arr_pooled = ia.pool(arr, 2, np.average)
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.average([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.average([2, 3, 6, 7]))
    assert arr_pooled[1, 0] == int(np.average([8, 9, 12, 13]))
    assert arr_pooled[1, 1] == int(np.average([10, 11, 14, 15]))

    arr = np.int32([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    arr_pooled = ia.pool(arr, 2, np.average)
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert arr_pooled[0, 0] == int(np.average([0, 1, 4, 5]))
    assert arr_pooled[0, 1] == int(np.average([2, 3, 6, 7]))
    assert arr_pooled[1, 0] == int(np.average([8, 9, 12, 13]))
    assert arr_pooled[1, 1] == int(np.average([10, 11, 14, 15]))

    arr = np.float32([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ])
    arr_pooled = ia.pool(arr, 2, np.average)
    assert arr_pooled.shape == (2, 2)
    assert arr_pooled.dtype == arr.dtype.type
    assert np.allclose(arr_pooled[0, 0], np.average([0, 1, 4, 5]))
    assert np.allclose(arr_pooled[0, 1], np.average([2, 3, 6, 7]))
    assert np.allclose(arr_pooled[1, 0], np.average([8, 9, 12, 13]))
    assert np.allclose(arr_pooled[1, 1], np.average([10, 11, 14, 15]))

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
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = 64
    image[0, 1] = 128
    image[1, 0] = 192
    image[1, 1] = 256

    grid = ia.draw_grid([image], rows=1, cols=1)
    assert np.array_equal(grid, image)

    grid = ia.draw_grid(np.uint8([image]), rows=1, cols=1)
    assert np.array_equal(grid, image)

    grid = ia.draw_grid([image, image, image, image], rows=2, cols=2)
    expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert np.array_equal(grid, expected)

    grid = ia.draw_grid([image, image], rows=1, cols=2)
    expected = np.hstack([image, image])
    assert np.array_equal(grid, expected)

    grid = ia.draw_grid([image, image, image, image], rows=2, cols=None)
    expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert np.array_equal(grid, expected)

    grid = ia.draw_grid([image, image, image, image], rows=None, cols=2)
    expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert np.array_equal(grid, expected)

    grid = ia.draw_grid([image, image, image, image], rows=None, cols=None)
    expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert np.array_equal(grid, expected)


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
    except Exception as e:
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
    except Exception as e:
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
    kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords={"x": -1, "y": -2}, threshold=20, nb_channels=3)
    assert kpi2.shape == (5, 5, 3)
    assert len(kpi2.keypoints) == 2
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[0].x == 1
    assert kpi2.keypoints[1].y == -2
    assert kpi2.keypoints[1].x == -1

    kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
    kps_image[2, 1, 0] = 255
    kps_image[4, 3, 1] = 10
    kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords=(-1, -2), threshold=20, nb_channels=3)
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
        kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords="exception-please", threshold=20, nb_channels=3)
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
    expected = "KeypointsOnImage([Keypoint(x=1.00000000, y=2.00000000), Keypoint(x=3.00000000, y=4.00000000)], shape=(5, 5, 3))"
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
    assert bb_inter == False

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
    assert bb.is_fully_within_image((100, 100, 3)) == True
    assert bb.is_fully_within_image((20, 100, 3)) == False
    assert bb.is_fully_within_image((100, 30, 3)) == False
    assert bb.is_fully_within_image((1, 1, 3)) == False

    # is_partly_within_image
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.is_partly_within_image((100, 100, 3)) == True
    assert bb.is_partly_within_image((20, 100, 3)) == True
    assert bb.is_partly_within_image((100, 30, 3)) == True
    assert bb.is_partly_within_image((1, 1, 3)) == False

    # is_out_of_image()
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.is_out_of_image((100, 100, 3), partly=True, fully=True) == False
    assert bb.is_out_of_image((100, 100, 3), partly=False, fully=True) == False
    assert bb.is_out_of_image((100, 100, 3), partly=True, fully=False) == False
    assert bb.is_out_of_image((20, 100, 3), partly=True, fully=True) == True
    assert bb.is_out_of_image((20, 100, 3), partly=False, fully=True) == False
    assert bb.is_out_of_image((20, 100, 3), partly=True, fully=False) == True
    assert bb.is_out_of_image((100, 30, 3), partly=True, fully=True) == True
    assert bb.is_out_of_image((100, 30, 3), partly=False, fully=True) == False
    assert bb.is_out_of_image((100, 30, 3), partly=True, fully=False) == True
    assert bb.is_out_of_image((1, 1, 3), partly=True, fully=True) == True
    assert bb.is_out_of_image((1, 1, 3), partly=False, fully=True) == True
    assert bb.is_out_of_image((1, 1, 3), partly=True, fully=False) == False

    # cut_out_of_image
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb_cut = bb.cut_out_of_image((100, 100, 3))
    eps = np.finfo(np.float32).eps
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert bb_cut.y2 == 30
    assert bb_cut.x2 == 40
    bb_cut = bb.cut_out_of_image(np.zeros((100, 100, 3), dtype=np.uint8))
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert bb_cut.y2 == 30
    assert bb_cut.x2 == 40
    bb_cut = bb.cut_out_of_image((20, 100, 3))
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert 20 - 2*eps < bb_cut.y2 < 20
    assert bb_cut.x2 == 40
    bb_cut = bb.cut_out_of_image((100, 30, 3))
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
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])
    assert np.all(image == 0)

    image_bb = bb.draw_on_image(image, color=[255, 0, 0], alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 0, 0])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    image_bb = bb.draw_on_image(image, color=128, alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [128, 128, 128])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    image_bb = bb.draw_on_image(image+100, color=[200, 200, 200], alpha=0.5, thickness=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [150, 150, 150])
    assert np.all(image_bb[~bb_mask] == [100, 100, 100])

    image_bb = bb.draw_on_image((image+100).astype(np.float32), color=[200, 200, 200], alpha=0.5, thickness=1, copy=True, raise_if_out_of_image=False)
    assert np.sum(np.abs((image_bb - [150, 150, 150])[bb_mask])) < 0.1
    assert np.sum(np.abs((image_bb - [100, 100, 100])[~bb_mask])) < 0.1

    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=False, raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])
    assert np.all(image[bb_mask] == [255, 255, 255])
    assert np.all(image[~bb_mask] == [0, 0, 0])

    image = np.zeros_like(image)
    bb = ia.BoundingBox(y1=-1, x1=-1, y2=2, x2=2, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[2, 0:3] = True
    bb_mask[0:3, 2] = True
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    bb = ia.BoundingBox(y1=1, x1=1, y2=3, x2=3, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[0:5, 0:5] = True
    bb_mask[2, 2] = False
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=2, copy=True, raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    bb = ia.BoundingBox(y1=-1, x1=-1, y2=1, x2=1, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[0:1+1, 1] = True
    bb_mask[1, 0:1+1] = True
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    bb = ia.BoundingBox(y1=-1, x1=-1, y2=1, x2=1, label=None)
    got_exception = False
    try:
        image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=True)
    except Exception as e:
        got_exception = True
    assert got_exception == False

    bb = ia.BoundingBox(y1=-5, x1=-5, y2=-1, x2=-1, label=None)
    got_exception = False
    try:
        image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=True)
    except Exception as e:
        got_exception = True
    assert got_exception == True

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

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    image_pad = np.pad(image, ((1, 0), (1, 0), (0, 0)), mode="constant", constant_values=0)
    bb = ia.BoundingBox(y1=-1, y2=3, x1=-1, x2=4, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image_pad[0:4, 0:5, :])

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

    # draw_on_image()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    image = bbsoi.draw_on_image(np.zeros(bbsoi.shape, dtype=np.uint8), color=[0, 255, 0], alpha=1.0, thickness=1, copy=True, raise_if_out_of_image=False)
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

    # cut_out_of_image()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    eps = np.finfo(np.float32).eps
    bbsoi_cut = bbsoi.cut_out_of_image()
    assert len(bbsoi_cut.bounding_boxes) == 2
    assert bbsoi_cut.bounding_boxes[0].y1 == 10
    assert bbsoi_cut.bounding_boxes[0].x1 == 20
    assert bbsoi_cut.bounding_boxes[0].y2 == 30
    assert bbsoi_cut.bounding_boxes[0].x2 == 40
    assert bbsoi_cut.bounding_boxes[1].y1 == 15
    assert bbsoi_cut.bounding_boxes[1].x1 == 25
    assert bbsoi_cut.bounding_boxes[1].y2 == 35
    assert 50 - 2*eps < bbsoi_cut.bounding_boxes[1].x2 < 50

    # shift()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bbsoi_shifted = bbsoi.shift(right=1)
    assert len(bbsoi_cut.bounding_boxes) == 2
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

    heatmaps_scaled = heatmaps.scale((4, 4), interpolation="nearest")
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

    heatmaps_scaled = heatmaps.scale(2.0, interpolation="nearest")
    assert heatmaps_scaled.arr_0to1.shape == (2, 4, 1)
    assert heatmaps_scaled.arr_0to1.dtype.type == np.float32
    assert np.allclose(
        heatmaps_scaled.arr_0to1[:, :, 0],
        np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ])
    )


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
        observed = segmap.get_arr_int(background_class_id=2)
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
    # draw_on_image(self, image, alpha=0.5, resize="segmentation_map", background_threshold=0.01, background_class_id=0, colors=None, draw_background=False):
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
    col0 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[0]
    col1 = ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[1]
    expected = np.uint8([
        [image[0,0,:], col1, col1],
        [image[1,0,:], col1, col1],
        [image[2,0,:], col1, col1]
    ])
    assert np.array_equal(observed, expected)

    # overlay without background drawn
    a1 = 0.7
    a0 = 1.0 - a1
    observed = segmap.draw_on_image(image, alpha=a1, draw_background=False)
    col0 = np.uint8(ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[0])
    col1 = np.uint8(ia.SegmentationMapOnImage.DEFAULT_SEGMENT_COLORS[1])
    expected = np.float32([
        [image[0,0,:], a0*image[0,1,:] + a1*col1, a0*image[0,2,:] + a1*col1],
        [image[1,0,:], a0*image[1,1,:] + a1*col1, a0*image[1,2,:] + a1*col1],
        [image[2,0,:], a0*image[2,1,:] + a1*col1, a0*image[2,2,:] + a1*col1]
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
    expected = np.pad(segmap.arr, ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=0)
    assert np.allclose(observed, expected)

    segmap_padded = segmap.pad_to_aspect_ratio(1.0, cval=1.0)
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=1.0)
    assert np.allclose(observed, expected)

    segmap_padded = segmap.pad_to_aspect_ratio(1.0, mode="edge")
    observed = segmap_padded.arr
    expected = np.pad(segmap.arr, ((1, 0), (0, 0), (0, 0)), mode="edge")
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

    segmap_scaled = segmap.scale((4, 4))
    observed = segmap_scaled.arr
    expected = np.clip(ia.imresize_single_image(segmap.arr, (4, 4), interpolation="cubic"), 0, 1.0)
    assert np.allclose(observed, expected)
    assert np.array_equal(segmap_scaled.get_arr_int(), np.int32([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ]))

    segmap_scaled = segmap.scale((4, 4), interpolation="nearest")
    observed = segmap_scaled.arr
    expected = ia.imresize_single_image(segmap.arr, (4, 4), interpolation="nearest")
    assert np.allclose(observed, expected)
    assert np.array_equal(segmap_scaled.get_arr_int(), np.int32([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ]))

    segmap_scaled = segmap.scale(2.0)
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


def test_BatchLoader():
    def _load_func():
        for _ in sm.xrange(20):
            yield ia.Batch(images=np.zeros((2, 4, 4, 3), dtype=np.uint8))

    # TODO these loops somehow require a `or len(loaded) < 20*nb_workers` on Travis, but not
    # locally. (On Travis, usually one batch is missing, i.e. probably still in the queue.)
    # That shouldn't be neccessary due to loader.all_finished(), but something breaks here.
    # queue.close() works on Tavis py2, but not py3 as it raises an `OSError: handle is closed`.
    for nb_workers in [1, 2]:
        # repeat these tests many times to catch rarer race conditions
        for _ in sm.xrange(50):
            loader = ia.BatchLoader(_load_func, queue_size=2, nb_workers=nb_workers, threaded=True)
            loaded = []
            counter = 0
            while (not loader.all_finished() or not loader.queue.empty() or len(loaded) < 20*nb_workers) and counter < 1000:
                try:
                    batch = loader.queue.get(timeout=0.001)
                    loaded.append(batch)
                except:
                    pass
                counter += 1
            #loader.queue.close()
            #while not loader.queue.empty():
            #    loaded.append(loader.queue.get())
            assert len(loaded) == 20*nb_workers, "Expected %d to be loaded by threads, got %d for %d workers at counter %d." % (20*nb_workers, len(loaded), nb_workers, counter)

            loader = ia.BatchLoader(_load_func, queue_size=200, nb_workers=nb_workers, threaded=True)
            loader.terminate()
            assert loader.all_finished

            loader = ia.BatchLoader(_load_func, queue_size=2, nb_workers=nb_workers, threaded=False)
            loaded = []
            counter = 0
            while (not loader.all_finished() or not loader.queue.empty() or len(loaded) < 20*nb_workers) and counter < 1000:
                try:
                    batch = loader.queue.get(timeout=0.001)
                    loaded.append(batch)
                except:
                    pass
                counter += 1
            #loader.queue.close()
            #while not loader.queue.empty():
            #    loaded.append(loader.queue.get())
            assert len(loaded) == 20*nb_workers, "Expected %d to be loaded by background processes, got %d for %d workers at counter %d." % (20*nb_workers, len(loaded), nb_workers, counter)

            loader = ia.BatchLoader(_load_func, queue_size=200, nb_workers=nb_workers, threaded=False)
            loader.terminate()
            assert loader.all_finished


def test_Noop():
    reseed()

    images = create_random_images((16, 70, 50, 3))
    keypoints = create_random_keypoints((16, 70, 50, 3), 4)
    aug = iaa.Noop()
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    assert iaa.Noop().get_parameters() == []


def test_Lambda():
    reseed()

    base_img = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]
    images = np.array([base_img])
    images_list = [base_img]

    images_aug = images + 1
    images_aug_list = [image + 1 for image in images_list]

    heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 1.0, 1.0]])
    heatmaps_arr_aug = np.float32([[0.5, 0.0, 1.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 1.0, 1.0]])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=1),
                                          ia.Keypoint(x=0, y=2)], shape=base_img.shape)]

    def func_images(images, random_state, parents, hooks):
        if isinstance(images, list):
            images = [image + 1 for image in images]
        else:
            images = images + 1
        return images

    def func_heatmaps(heatmaps, random_state, parents, hooks):
        heatmaps[0].arr_0to1[0, 0] += 0.5
        return heatmaps

    def func_keypoints(keypoints_on_images, random_state, parents, hooks):
        for keypoints_on_image in keypoints_on_images:
            for kp in keypoints_on_image.keypoints:
                kp.x = (kp.x + 1) % 3
        return keypoints_on_images

    aug = iaa.Lambda(func_images, func_heatmaps, func_keypoints)
    aug_det = aug.to_deterministic()

    # check once that the augmenter can handle lists correctly
    observed = aug.augment_images(images_list)
    expected = images_aug_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_aug_list
    assert array_equal_lists(observed, expected)

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images_aug
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images_aug
        assert np.array_equal(observed, expected)

        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps_arr_aug)

        observed = aug_det.augment_heatmaps([heatmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps_arr_aug)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints_aug
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints_aug
        assert keypoints_equal(observed, expected)


def test_AssertLambda():
    reseed()

    base_img = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]
    images = np.array([base_img])
    images_list = [base_img]

    heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 1.0, 1.0]])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    def func_images_succeeds(images, random_state, parents, hooks):
        return images[0][0, 0] == 0 and images[0][2, 2] == 1

    def func_images_fails(images, random_state, parents, hooks):
        return images[0][0, 0] == 1

    def func_heatmaps_succeeds(heatmaps, random_state, parents, hooks):
        return heatmaps[0].arr_0to1[0, 0] < 0 + 1e-6

    def func_heatmaps_fails(heatmaps, random_state, parents, hooks):
        return heatmaps[0].arr_0to1[0, 0] > 0 + 1e-6

    def func_keypoints_succeeds(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images[0].keypoints[0].x == 0 and keypoints_on_images[0].keypoints[2].x == 2

    def func_keypoints_fails(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images[0].keypoints[0].x == 2

    aug_succeeds = iaa.AssertLambda(func_images=func_images_succeeds,
                                    func_heatmaps=func_heatmaps_succeeds,
                                    func_keypoints=func_keypoints_succeeds)
    aug_succeeds_det = aug_succeeds.to_deterministic()
    aug_fails = iaa.AssertLambda(func_images=func_images_fails,
                                 func_heatmaps=func_heatmaps_fails,
                                 func_keypoints=func_keypoints_fails)
    aug_fails_det = aug_fails.to_deterministic()

    # images as numpy array
    observed = aug_succeeds.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    try:
        observed = aug_fails.augment_images(images)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    try:
        observed = aug_fails.augment_images(images)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    # Lists of images
    observed = aug_succeeds.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    try:
        observed = aug_fails.augment_images(images_list)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    try:
        observed = aug_fails.augment_images(images_list)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    # heatmaps
    observed = aug_succeeds.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps.get_arr())

    try:
        observed = aug_fails.augment_heatmaps([heatmaps])[0]
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps.get_arr())

    try:
        observed = aug_fails.augment_heatmaps([heatmaps])[0]
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    # keypoints
    observed = aug_succeeds.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    try:
        observed = aug_fails.augment_keypoints(keypoints)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    try:
        observed = aug_fails.augment_keypoints(keypoints)
        errored = False
    except AssertionError as e:
        errored = True
    assert errored


def test_AssertShape():
    reseed()

    base_img = np.array([[0, 0, 1, 0],
                         [0, 0, 1, 0],
                         [0, 1, 1, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]
    images = np.array([base_img])
    images_list = [base_img]
    heatmaps_arr = np.float32([[0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 1.0, 1.0, 0.0]])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 4, 3))
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    base_img_h4 = np.array([[0, 0, 1, 0],
                            [0, 0, 1, 0],
                            [0, 1, 1, 0],
                            [1, 0, 1, 0]], dtype=np.uint8)
    base_img_h4 = base_img_h4[:, :, np.newaxis]
    images_h4 = np.array([base_img_h4])
    heatmaps_arr_h4 = np.float32([[0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 1.0, 1.0, 0.0],
                                  [1.0, 0.0, 1.0, 0.0]])
    heatmaps_h4 = ia.HeatmapsOnImage(heatmaps_arr_h4, shape=(4, 4, 3))
    keypoints_h4 = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                         ia.Keypoint(x=2, y=2)], shape=base_img_h4.shape)]

    # image must have exactly shape (1, 3, 4, 1)
    aug = iaa.AssertShape((1, 3, 4, 1))
    aug_det = aug.to_deterministic()

    # check once that the augmenter can handle lists correctly
    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps.get_arr())

        observed = aug_det.augment_heatmaps([heatmaps])[0]
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps.get_arr())

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_heatmaps([heatmaps_h4])[0]
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

    # any value for number of images allowed (None)
    aug = iaa.AssertShape((None, 3, 4, 1))
    aug_det = aug.to_deterministic()
    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps.get_arr())

        observed = aug_det.augment_heatmaps([heatmaps])[0]
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps.get_arr())

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_heatmaps([heatmaps_h4])[0]
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

    # list of possible choices [1, 3, 5] for height
    aug = iaa.AssertShape((1, [1, 3, 5], 4, 1))
    aug_det = aug.to_deterministic()
    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps.get_arr())

        observed = aug_det.augment_heatmaps([heatmaps])[0]
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps.get_arr())

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_heatmaps([heatmaps_h4])[0]
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

    # range of 1-3 for height (tuple comparison is a <= x < b, so we use (1,4) here)
    aug = iaa.AssertShape((1, (1, 4), 4, 1))
    aug_det = aug.to_deterministic()
    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps.get_arr())

        observed = aug_det.augment_heatmaps([heatmaps])[0]
        assert observed.shape == (3, 4, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps.get_arr())

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_heatmaps([heatmaps_h4])[0]
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError as e:
            errored = True
        assert errored

    # bad datatype
    got_exception = False
    try:
        aug = iaa.AssertShape((1, False, 4, 1))
        observed = aug.augment_images(np.zeros((1, 2, 2, 1), dtype=np.uint8))
    except Exception as exc:
        assert "Invalid datatype " in str(exc)
        got_exception = True
    assert got_exception


def test_Alpha():
    reseed()

    base_img = np.zeros((3, 3, 1), dtype=np.uint8)
    heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 1.0, 1.0]])
    heatmaps_arr_r1 = np.float32([[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0]])
    heatmaps_arr_l1 = np.float32([[0.0, 1.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [1.0, 1.0, 0.0]])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    aug = iaa.Alpha(1, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = (base_img + 10).astype(np.uint8)
    assert np.allclose(observed, expected)

    for per_channel in [False, True]:
        aug = iaa.Alpha(1, iaa.Affine(translate_px={"x":1}), iaa.Affine(translate_px={"x":-1}), per_channel=per_channel)
        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == heatmaps.shape
        assert 0 - 1e-6 < heatmaps.min_value < 0 + 1e-6
        assert 1 - 1e-6 < heatmaps.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps_arr_r1)

    aug = iaa.Alpha(0, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = (base_img + 20).astype(np.uint8)
    assert np.allclose(observed, expected)

    for per_channel in [False, True]:
        aug = iaa.Alpha(0, iaa.Affine(translate_px={"x":1}), iaa.Affine(translate_px={"x":-1}), per_channel=per_channel)
        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == heatmaps.shape
        assert 0 - 1e-6 < heatmaps.min_value < 0 + 1e-6
        assert 1 - 1e-6 < heatmaps.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps_arr_l1)

    aug = iaa.Alpha(0.75, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = (base_img + 0.75 * 10 + 0.25 * 20).astype(np.uint8)
    assert np.allclose(observed, expected)

    aug = iaa.Alpha(0.75, None, iaa.Add(20))
    observed = aug.augment_image(base_img + 10)
    expected = (base_img + 0.75 * 10 + 0.25 * (10 + 20)).astype(np.uint8)
    assert np.allclose(observed, expected)

    aug = iaa.Alpha(0.75, iaa.Add(10), None)
    observed = aug.augment_image(base_img + 10)
    expected = (base_img + 0.75 * (10 + 10) + 0.25 * 10).astype(np.uint8)
    assert np.allclose(observed, expected)

    base_img = np.zeros((1, 2, 1), dtype=np.uint8)
    nb_iterations = 1000
    aug = iaa.Alpha((0.0, 1.0), iaa.Add(10), iaa.Add(110))
    values = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        observed_val = np.round(np.average(observed)) - 10
        values.append(observed_val / 100)

    nb_bins = 5
    hist, _ = np.histogram(values, bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / nb_iterations
        assert density_expected - density_tolerance < density < density_expected + density_tolerance

    # bad datatype for factor
    got_exception = False
    try:
        aug = iaa.Alpha(False, iaa.Add(10), None)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # per_channel
    aug = iaa.Alpha(1.0, iaa.Add((0, 100), per_channel=True), None, per_channel=True)
    observed = aug.augment_image(np.zeros((1, 1, 1000), dtype=np.uint8))
    uq = np.unique(observed)
    assert len(uq) > 1
    assert np.max(observed) > 80
    assert np.min(observed) < 20

    aug = iaa.Alpha((0.0, 1.0), iaa.Add(100), None, per_channel=True)
    observed = aug.augment_image(np.zeros((1, 1, 1000), dtype=np.uint8))
    uq = np.unique(observed)
    assert len(uq) > 1
    assert np.max(observed) > 80
    assert np.min(observed) < 20

    aug = iaa.Alpha((0.0, 1.0), iaa.Add(100), iaa.Add(0), per_channel=0.5)
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

    # bad datatype for per_channel
    got_exception = False
    try:
        aug = iaa.Alpha(0.5, iaa.Add(10), None, per_channel="test")
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # propagating
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

    # -----
    # keypoints
    # -----
    kps = [ia.Keypoint(x=5, y=10), ia.Keypoint(x=6, y=11)]
    kpsoi = ia.KeypointsOnImage(kps, shape=(20, 20, 3))

    aug = iaa.Alpha(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.Alpha(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.Alpha(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    aug = iaa.Alpha(0.499, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    # per_channel
    aug = iaa.Alpha(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.Alpha(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    aug = iaa.Alpha(iap.Choice([0.49, 0.51]), iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    expected_same = kpsoi.deepcopy()
    expected_shifted = kpsoi.shift(x=1)
    seen = [0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_keypoints([kpsoi])[0]
        if keypoints_equal([observed], [expected_same]):
            seen[0] += 1
        elif keypoints_equal([observed], [expected_shifted]):
            seen[1] += 1
        else:
            assert False
    assert 100 - 50 < seen[0] < 100 + 50
    assert 100 - 50 < seen[1] < 100 + 50

    # propagating
    aug = iaa.Alpha(0.0, iaa.Affine(translate_px={"x": 1}), iaa.Affine(translate_px={"y": 1}), name="AlphaTest")
    def propagator(kpsoi_to_aug, augmenter, parents, default):
        if "Alpha" in augmenter.name:
            return False
        else:
            return default
    hooks = ia.HooksKeypoints(propagator=propagator)
    observed = aug.augment_keypoints([kpsoi], hooks=hooks)[0]
    assert keypoints_equal([observed], [kpsoi])

    # -----
    # get_parameters()
    # -----
    first = iaa.Noop()
    second = iaa.Sequential([iaa.Add(1)])
    aug = iaa.Alpha(0.65, first, second, per_channel=1)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert 0.65 - 1e-6 < params[0].value < 0.65 + 1e-6
    assert params[1].value == 1

    # -----
    # get_children_lists()
    # -----
    first = iaa.Noop()
    second = iaa.Sequential([iaa.Add(1)])
    aug = iaa.Alpha(0.65, first, second, per_channel=1)
    children_lsts = aug.get_children_lists()
    assert len(children_lsts) == 2
    assert ia.is_iterable([lst for lst in children_lsts])
    assert first in children_lsts[0]
    assert second == children_lsts[1]


def test_AlphaElementwise():
    reseed()

    base_img = np.zeros((3, 3, 1), dtype=np.uint8)
    heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 1.0, 1.0]])
    heatmaps_arr_r1 = np.float32([[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0]])
    heatmaps_arr_l1 = np.float32([[0.0, 1.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [1.0, 1.0, 0.0]])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    aug = iaa.AlphaElementwise(1, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = base_img + 10
    assert np.allclose(observed, expected)

    aug = iaa.AlphaElementwise(1, iaa.Affine(translate_px={"x": 1}), iaa.Affine(translate_px={"x": -1}))
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps_arr_r1)

    aug = iaa.AlphaElementwise(0, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = base_img + 20
    assert np.allclose(observed, expected)

    aug = iaa.AlphaElementwise(0, iaa.Affine(translate_px={"x": 1}), iaa.Affine(translate_px={"x": -1}))
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps_arr_l1)

    aug = iaa.AlphaElementwise(0.75, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = (base_img + 0.75 * 10 + 0.25 * 20).astype(np.uint8)
    assert np.allclose(observed, expected)

    aug = iaa.AlphaElementwise(0.75, None, iaa.Add(20))
    observed = aug.augment_image(base_img + 10)
    expected = (base_img + 0.75 * 10 + 0.25 * (10 + 20)).astype(np.uint8)
    assert np.allclose(observed, expected)

    aug = iaa.AlphaElementwise(0.75, iaa.Add(10), None)
    observed = aug.augment_image(base_img + 10)
    expected = (base_img + 0.75 * (10 + 10) + 0.25 * 10).astype(np.uint8)
    assert np.allclose(observed, expected)

    base_img = np.zeros((100, 100), dtype=np.uint8)
    aug = iaa.AlphaElementwise((0.0, 1.0), iaa.Add(10), iaa.Add(110))
    observed = (aug.augment_image(base_img) - 10) / 100
    nb_bins = 10
    hist, _ = np.histogram(observed.flatten(),  bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / observed.size
        assert density_expected - density_tolerance < density < density_expected + density_tolerance

    base_img = np.zeros((1, 1, 100), dtype=np.uint8)
    aug = iaa.AlphaElementwise((0.0, 1.0), iaa.Add(10), iaa.Add(110), per_channel=True)
    observed = aug.augment_image(base_img)
    assert len(set(observed.flatten())) > 1

    # propagating
    aug = iaa.AlphaElementwise(0.5, iaa.Add(100), iaa.Add(50), name="AlphaElementwiseTest")
    def propagator(images, augmenter, parents, default):
        if "AlphaElementwise" in augmenter.name:
            return False
        else:
            return default
    hooks = ia.HooksImages(propagator=propagator)
    image = np.zeros((10, 10, 3), dtype=np.uint8) + 1
    observed = aug.augment_image(image, hooks=hooks)
    assert np.array_equal(observed, image)

    # -----
    # heatmaps and per_channel
    # -----
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

    aug = iaa.AlphaElementwise(
        _DummyMaskParameter(inverted=False),
        iaa.Affine(translate_px={"x": 1}),
        iaa.Affine(translate_px={"x": -1}),
        per_channel=True
    )
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps_arr_r1)

    aug = iaa.AlphaElementwise(
        _DummyMaskParameter(inverted=True),
        iaa.Affine(translate_px={"x": 1}),
        iaa.Affine(translate_px={"x": -1}),
        per_channel=True
    )
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps_arr_l1)

    # -----
    # keypoints
    # -----
    kps = [ia.Keypoint(x=5, y=10), ia.Keypoint(x=6, y=11)]
    kpsoi = ia.KeypointsOnImage(kps, shape=(20, 20, 3))

    aug = iaa.AlphaElementwise(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.AlphaElementwise(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.AlphaElementwise(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    aug = iaa.AlphaElementwise(0.499, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    # per_channel
    aug = iaa.AlphaElementwise(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.AlphaElementwise(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    """
    TODO this test currently doesn't work as AlphaElementwise augments keypoints without sampling
    overlay factors per (x, y) location. (i.e. similar behaviour to Alpha)

    aug = iaa.Alpha(iap.Choice([0.49, 0.51]), iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    expected_same = kpsoi.deepcopy()
    expected_both_shifted = kpsoi.shift(x=1)
    expected_first_shifted = KeypointsOnImage([kps[0].shift(x=1), kps[1]], shape=kpsoi.shape)
    expected_second_shifted = KeypointsOnImage([kps[0], kps[1].shift(x=1)], shape=kpsoi.shape)
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
    """

    # propagating
    aug = iaa.AlphaElementwise(0.0, iaa.Affine(translate_px={"x": 1}), iaa.Affine(translate_px={"y": 1}), name="AlphaElementwiseTest")
    def propagator(kpsoi_to_aug, augmenter, parents, default):
        if "AlphaElementwise" in augmenter.name:
            return False
        else:
            return default
    hooks = ia.HooksKeypoints(propagator=propagator)
    observed = aug.augment_keypoints([kpsoi], hooks=hooks)[0]
    assert keypoints_equal([observed], [kpsoi])


def test_Superpixels():
    reseed()

    def _array_equals_tolerant(a, b, tolerance):
        diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
        return np.all(diff <= tolerance)

    base_img = [
        [255, 255, 255, 0, 0, 0],
        [255, 235, 255, 0, 20, 0],
        [250, 250, 250, 5, 5, 5]
    ]
    base_img = np.tile(np.array(base_img, dtype=np.uint8)[..., np.newaxis], (1, 1, 3))

    base_img_superpixels = [
        [251, 251, 251, 4, 4, 4],
        [251, 251, 251, 4, 4, 4],
        [251, 251, 251, 4, 4, 4]
    ]
    base_img_superpixels = np.tile(np.array(base_img_superpixels, dtype=np.uint8)[..., np.newaxis], (1, 1, 3))

    base_img_superpixels_left = np.copy(base_img_superpixels)
    base_img_superpixels_left[:, 3:, :] = base_img[:, 3:, :]

    base_img_superpixels_right = np.copy(base_img_superpixels)
    base_img_superpixels_right[:, :3, :] = base_img[:, :3, :]

    aug = iaa.Superpixels(p_replace=0, n_segments=2)
    observed = aug.augment_image(base_img)
    expected = base_img
    assert np.allclose(observed, expected)

    aug = iaa.Superpixels(p_replace=1.0, n_segments=2)
    observed = aug.augment_image(base_img)
    expected = base_img_superpixels
    assert _array_equals_tolerant(observed, expected, 2)

    aug = iaa.Superpixels(p_replace=1.0, n_segments=iap.Deterministic(2))
    observed = aug.augment_image(base_img)
    expected = base_img_superpixels
    assert _array_equals_tolerant(observed, expected, 2)

    aug = iaa.Superpixels(p_replace=iap.Binomial(iap.Choice([0.0, 1.0])), n_segments=2)
    observed = aug.augment_image(base_img)
    assert np.allclose(observed, base_img) or _array_equals_tolerant(observed, base_img_superpixels, 2)

    aug = iaa.Superpixels(p_replace=0.5, n_segments=2)
    seen = {"none": False, "left": False, "right": False, "both": False}
    for _ in sm.xrange(100):
        observed = aug.augment_image(base_img)
        if _array_equals_tolerant(observed, base_img, 2):
            seen["none"] = True
        elif _array_equals_tolerant(observed, base_img_superpixels_left, 2):
            seen["left"] = True
        elif _array_equals_tolerant(observed, base_img_superpixels_right, 2):
            seen["right"] = True
        elif _array_equals_tolerant(observed, base_img_superpixels, 2):
            seen["both"] = True
        else:
            raise Exception("Generated superpixels image does not match any expected image.")
        if all(seen.values()):
            break
    assert all(seen.values())

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.Superpixels(p_replace="test", n_segments=100)
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.Superpixels(p_replace=1, n_segments="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.Superpixels(p_replace=0.5, n_segments=2, max_size=100, interpolation="nearest")
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Binomial)
    assert isinstance(params[0].p, iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert 0.5 - 1e-4 < params[0].p.value < 0.5 + 1e-4
    assert params[1].value == 2
    assert params[2] == 100
    assert params[3] == "nearest"


def test_Scale():
    reseed()

    base_img2d = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 255, 255, 255, 255, 255, 255, 0],
        [0, 255, 255, 255, 255, 255, 255, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    base_img2d = np.array(base_img2d, dtype=np.uint8)
    base_img3d = np.tile(base_img2d[..., np.newaxis], (1, 1, 3))

    intensity_avg = np.average(base_img2d)
    intensity_low = intensity_avg - 0.2 * np.abs(intensity_avg - 128)
    intensity_high = intensity_avg + 0.2 * np.abs(intensity_avg - 128)

    aspect_ratio2d = base_img2d.shape[1] / base_img2d.shape[0]
    aspect_ratio3d = base_img3d.shape[1] / base_img3d.shape[0]

    aug = iaa.Scale(12)
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (12, 12)
    assert observed3d.shape == (12, 12, 3)
    assert 50 < np.average(observed2d) < 205
    assert 50 < np.average(observed3d) < 205

    aug = iaa.Scale({"height": 8, "width": 12})
    heatmaps_arr = (base_img2d / 255.0).astype(np.float32)
    heatmaps_aug = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr, shape=base_img3d.shape)])[0]
    assert heatmaps_aug.shape == (8, 12, 3)
    assert 0 - 1e-6 < heatmaps_aug.min_value < 0 + 1e-6
    assert 1 - 1e-6 < heatmaps_aug.max_value < 1 + 1e-6
    assert np.average(heatmaps_aug.get_arr()[0, :]) < 0.05
    assert np.average(heatmaps_aug.get_arr()[-1, :]) < 0.05
    assert np.average(heatmaps_aug.get_arr()[:, 0]) < 0.05
    assert 0.8 < np.average(heatmaps_aug.get_arr()[2:6, 2:10]) < 1 + 1e-6

    aug = iaa.Scale([12, 14])
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (14, 14)]
        assert observed3d.shape in [(12, 12, 3), (14, 14, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    aug = iaa.Scale((12, 14))
    seen2d = [False, False, False]
    seen3d = [False, False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (13, 13), (14, 14)]
        assert observed3d.shape in [(12, 12, 3), (13, 13, 3), (14, 14, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        elif observed2d.shape == (13, 13):
            seen2d[1] = True
        else:
            seen2d[2] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        elif observed3d.shape == (13, 13, 3):
            seen3d[1] = True
        else:
            seen3d[2] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    aug = iaa.Scale("keep")
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == base_img2d.shape
    assert observed3d.shape == base_img3d.shape

    aug = iaa.Scale([])
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == base_img2d.shape
    assert observed3d.shape == base_img3d.shape

    aug = iaa.Scale({})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == base_img2d.shape
    assert observed3d.shape == base_img3d.shape

    aug = iaa.Scale({"height": 11})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (11, base_img2d.shape[1])
    assert observed3d.shape == (11, base_img3d.shape[1], 3)

    aug = iaa.Scale({"width": 13})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (base_img2d.shape[0], 13)
    assert observed3d.shape == (base_img3d.shape[0], 13, 3)

    aug = iaa.Scale({"height": 12, "width": 13})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (12, 13)
    assert observed3d.shape == (12, 13, 3)

    aug = iaa.Scale({"height": 12, "width": "keep"})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (12, base_img2d.shape[1])
    assert observed3d.shape == (12, base_img3d.shape[1], 3)

    aug = iaa.Scale({"height": "keep", "width": 12})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (base_img2d.shape[0], 12)
    assert observed3d.shape == (base_img3d.shape[0], 12, 3)

    aug = iaa.Scale({"height": 12, "width": "keep-aspect-ratio"})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (12, int(12 * aspect_ratio2d))
    assert observed3d.shape == (12, int(12 * aspect_ratio3d), 3)

    aug = iaa.Scale({"height": "keep-aspect-ratio", "width": 12})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (int(12 * (1/aspect_ratio2d)), 12)
    assert observed3d.shape == (int(12 * (1/aspect_ratio3d)), 12, 3)

    aug = iaa.Scale({"height": [12, 14], "width": 12})
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (14, 12)]
        assert observed3d.shape in [(12, 12, 3), (14, 12, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    aug = iaa.Scale({"height": 12, "width": [12, 14]})
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (12, 14)]
        assert observed3d.shape in [(12, 12, 3), (12, 14, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    aug = iaa.Scale({"height": 12, "width": iap.Choice([12, 14])})
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (12, 14)]
        assert observed3d.shape in [(12, 12, 3), (12, 14, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    aug = iaa.Scale({"height": (12, 14), "width": 12})
    seen2d = [False, False, False]
    seen3d = [False, False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (13, 12), (14, 12)]
        assert observed3d.shape in [(12, 12, 3), (13, 12, 3), (14, 12, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        elif observed2d.shape == (13, 12):
            seen2d[1] = True
        else:
            seen2d[2] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        elif observed3d.shape == (13, 12, 3):
            seen3d[1] = True
        else:
            seen3d[2] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    aug = iaa.Scale(2.0)
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (base_img2d.shape[0]*2, base_img2d.shape[1]*2)
    assert observed3d.shape == (base_img3d.shape[0]*2, base_img3d.shape[1]*2, 3)
    assert intensity_low < np.average(observed2d) < intensity_high
    assert intensity_low < np.average(observed3d) < intensity_high

    aug = iaa.Scale([2.0, 4.0])
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(base_img2d.shape[0]*2, base_img2d.shape[1]*2), (base_img2d.shape[0]*4, base_img2d.shape[1]*4)]
        assert observed3d.shape in [(base_img3d.shape[0]*2, base_img3d.shape[1]*2, 3), (base_img3d.shape[0]*4, base_img3d.shape[1]*4, 3)]
        if observed2d.shape == (base_img2d.shape[0]*2, base_img2d.shape[1]*2):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (base_img3d.shape[0]*2, base_img3d.shape[1]*2, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    aug = iaa.Scale(iap.Choice([2.0, 4.0]))
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(base_img2d.shape[0]*2, base_img2d.shape[1]*2), (base_img2d.shape[0]*4, base_img2d.shape[1]*4)]
        assert observed3d.shape in [(base_img3d.shape[0]*2, base_img3d.shape[1]*2, 3), (base_img3d.shape[0]*4, base_img3d.shape[1]*4, 3)]
        if observed2d.shape == (base_img2d.shape[0]*2, base_img2d.shape[1]*2):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (base_img3d.shape[0]*2, base_img3d.shape[1]*2, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    base_img2d = base_img2d[0:4, 0:4]
    base_img3d = base_img3d[0:4, 0:4, :]
    aug = iaa.Scale((0.76, 1.0))
    not_seen2d = set()
    not_seen3d = set()
    for size in sm.xrange(3, 4+1):
        not_seen2d.add((size, size))
    for size in sm.xrange(3, 4+1):
        not_seen3d.add((size, size, 3))
    possible2d = set(list(not_seen2d))
    possible3d = set(list(not_seen3d))
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in possible2d
        assert observed3d.shape in possible3d
        if observed2d.shape in not_seen2d:
            not_seen2d.remove(observed2d.shape)
        if observed3d.shape in not_seen3d:
            not_seen3d.remove(observed3d.shape)
        if not not_seen2d and not not_seen3d:
            break
    assert not not_seen2d
    assert not not_seen3d

    base_img2d = base_img2d[0:4, 0:4]
    base_img3d = base_img3d[0:4, 0:4, :]
    aug = iaa.Scale({"height": (0.76, 1.0), "width": (0.76, 1.0)})
    not_seen2d = set()
    not_seen3d = set()
    for hsize in sm.xrange(3, 4+1):
        for wsize in sm.xrange(3, 4+1):
            not_seen2d.add((hsize, wsize))
    #print(base_img3d.shape[0]//2, base_img3d.shape[1]+1)
    for hsize in sm.xrange(3, 4+1):
        for wsize in sm.xrange(3, 4+1):
            not_seen3d.add((hsize, wsize, 3))
    possible2d = set(list(not_seen2d))
    possible3d = set(list(not_seen3d))
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in possible2d
        assert observed3d.shape in possible3d
        if observed2d.shape in not_seen2d:
            not_seen2d.remove(observed2d.shape)
        if observed3d.shape in not_seen3d:
            not_seen3d.remove(observed3d.shape)
        if not not_seen2d and not not_seen3d:
            break
    assert not not_seen2d
    assert not not_seen3d

    got_exception = False
    try:
        aug = iaa.Scale("foo")
        observed2d = aug.augment_image(base_img2d)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception



    aug = iaa.Scale(size=1, interpolation="nearest")
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert params[0].value == 1
    assert params[1].value == "nearest"


def test_Pad():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    heatmaps_arr = np.float32([[0, 0, 0],
                               [0, 1.0, 0],
                               [0, 0, 0]])

    # test pad by 1 pixel on each side
    pads = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ]
    for pad in pads:
        top, right, bottom, left = pad
        height, width = base_img.shape[0:2]
        aug = iaa.Pad(px=pad, keep_size=False)
        base_img_padded = np.pad(base_img, ((top, bottom), (left, right), (0, 0)),
                                 mode="constant",
                                 constant_values=0)
        observed = aug.augment_images(images)
        assert np.array_equal(observed, np.array([base_img_padded]))

        observed = aug.augment_images(images_list)
        assert array_equal_lists(observed, [base_img_padded])

        keypoints_moved = [keypoints[0].shift(x=left, y=top)]
        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_moved)

        # heatmaps
        height, width = heatmaps_arr.shape[0:2]
        aug = iaa.Pad(px=pad, keep_size=False)
        heatmaps_arr_padded = np.pad(heatmaps_arr, ((top, bottom), (left, right)),
                                     mode="constant",
                                     constant_values=0)
        observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr, shape=base_img.shape)])[0]
        assert observed.shape == base_img_padded.shape
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.array_equal(observed.get_arr(), heatmaps_arr_padded)

    # test pad by range of pixels
    pads = [
        ((0, 2), 0, 0, 0),
        (0, (0, 2), 0, 0),
        (0, 0, (0, 2), 0),
        (0, 0, 0, (0, 2)),
    ]
    for pad in pads:
        top, right, bottom, left = pad
        height, width = base_img.shape[0:2]
        aug = iaa.Pad(px=pad, keep_size=False)
        aug_det = aug.to_deterministic()

        images_padded = []
        keypoints_padded = []
        top_range = top if isinstance(top, tuple) else (top, top)
        right_range = right if isinstance(right, tuple) else (right, right)
        bottom_range = bottom if isinstance(bottom, tuple) else (bottom, bottom)
        left_range = left if isinstance(left, tuple) else (left, left)
        for top_val in sm.xrange(top_range[0], top_range[1]+1):
            for right_val in sm.xrange(right_range[0], right_range[1]+1):
                for bottom_val in sm.xrange(bottom_range[0], bottom_range[1]+1):
                    for left_val in sm.xrange(left_range[0], left_range[1]+1):
                        images_padded.append(np.pad(base_img, ((top_val, bottom_val), (left_val, right_val), (0, 0)), mode="constant", constant_values=0))
                        keypoints_padded.append(keypoints[0].shift(x=left_val, y=top_val))

        movements = []
        movements_det = []
        for i in sm.xrange(100):
            observed = aug.augment_images(images)

            matches = [1 if np.array_equal(observed, np.array([base_img_padded])) else 0
                       for base_img_padded in images_padded]
            movements.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug_det.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_padded])) else 0
                       for base_img_padded in images_padded]
            movements_det.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug.augment_images(images_list)
            assert any([array_equal_lists(observed, [base_img_padded])
                        for base_img_padded in images_padded])

            observed = aug.augment_keypoints(keypoints)
            assert any([keypoints_equal(observed, [kp]) for kp in keypoints_padded])

        assert len(set(movements)) == 3
        assert len(set(movements_det)) == 1

    # test pad by list of exact pixel values
    pads = [
        ([0, 2], 0, 0, 0),
        (0, [0, 2], 0, 0),
        (0, 0, [0, 2], 0),
        (0, 0, 0, [0, 2]),
    ]
    for pad in pads:
        top, right, bottom, left = pad
        height, width = base_img.shape[0:2]
        aug = iaa.Pad(px=pad, keep_size=False)
        aug_det = aug.to_deterministic()

        images_padded = []
        keypoints_padded = []
        top_range = top if isinstance(top, list) else [top]
        right_range = right if isinstance(right, list) else [right]
        bottom_range = bottom if isinstance(bottom, list) else [bottom]
        left_range = left if isinstance(left, list) else [left]
        for top_val in top_range:
            for right_val in right_range:
                for bottom_val in bottom_range:
                    for left_val in left_range:
                        images_padded.append(np.pad(base_img, ((top_val, bottom_val), (left_val, right_val), (0, 0)), mode="constant", constant_values=0))
                        keypoints_padded.append(keypoints[0].shift(x=left_val, y=top_val))

        movements = []
        movements_det = []
        for i in sm.xrange(100):
            observed = aug.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_padded])) else 0
                       for base_img_padded in images_padded]
            movements.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug_det.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_padded])) else 0
                       for base_img_padded in images_padded]
            movements_det.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug.augment_images(images_list)
            assert any([array_equal_lists(observed, [base_img_padded])
                        for base_img_padded in images_padded])

            observed = aug.augment_keypoints(keypoints)
            assert any([keypoints_equal(observed, [kp]) for kp in keypoints_padded])

        assert len(set(movements)) == 2
        assert len(set(movements_det)) == 1

    # pad modes
    image = np.zeros((1, 2), dtype=np.uint8)
    image[0, 0] = 100
    image[0, 1] = 50
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode=iap.Choice(["constant", "maximum", "edge"]), pad_cval=0, keep_size=False)
    seen = [0, 0, 0]
    for _ in sm.xrange(300):
        observed = aug.augment_image(image)
        if observed[0, 2] == 0:
            seen[0] += 1
        elif observed[0, 2] == 100:
            seen[1] += 1
        elif observed[0, 2] == 50:
            seen[2] += 1
        else:
            assert False
    assert all([100 - 50 < v < 100 + 50 for v in seen])

    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode=ia.ALL, pad_cval=0, keep_size=False)
    expected = ["constant", "edge", "linear_ramp", "maximum", "median", "minimum", "reflect", "symmetric", "wrap"]
    assert isinstance(aug.pad_mode, iap.Choice)
    assert len(aug.pad_mode.a) == len(expected)
    assert all([v in aug.pad_mode.a for v in expected])

    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode=["constant", "maximum"], pad_cval=0, keep_size=False)
    expected = ["constant", "maximum"]
    assert isinstance(aug.pad_mode, iap.Choice)
    assert len(aug.pad_mode.a) == len(expected)
    assert all([v in aug.pad_mode.a for v in expected])

    got_exception = False
    try:
        aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode=False, pad_cval=0, keep_size=False)
    except Exception as exc:
        assert "Expected pad_mode to be " in str(exc)
        got_exception = True
    assert got_exception

    # pad modes, heatmaps
    heatmaps = ia.HeatmapsOnImage(np.ones((3, 3, 1), dtype=np.float32), shape=(3, 3, 3))
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="edge", pad_cval=0, keep_size=False)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert np.sum(observed.get_arr() <= 1e-4) == 3

    # pad cvals
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval=100, keep_size=False)
    observed = aug.augment_image(np.zeros((1, 1), dtype=np.uint8))
    assert observed[0, 0] == 0
    assert observed[0, 1] == 100

    image = np.zeros((1, 1), dtype=np.uint8)
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval=iap.Choice([50, 100]), keep_size=False)
    seen = [0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_image(image)
        if observed[0, 1] == 50:
            seen[0] += 1
        elif observed[0, 1] == 100:
            seen[1] += 1
        else:
            assert False
    assert all([100 - 50 < v < 100 + 50 for v in seen])

    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval=[50, 100], keep_size=False)
    expected = [50, 100]
    assert isinstance(aug.pad_cval, iap.Choice)
    assert len(aug.pad_cval.a) == len(expected)
    assert all([v in aug.pad_cval.a for v in expected])

    image = np.zeros((1, 1), dtype=np.uint8)
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval=(50, 52), keep_size=False)
    seen = [0, 0, 0]
    for _ in sm.xrange(300):
        observed = aug.augment_image(image)
        if observed[0, 1] == 50:
            seen[0] += 1
        elif observed[0, 1] == 51:
            seen[1] += 1
        elif observed[0, 1] == 52:
            seen[2] += 1
        else:
            assert False
    assert all([100 - 50 < v < 100 + 50 for v in seen])

    got_exception = False
    try:
        aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval="test", keep_size=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # pad cvals, heatmaps
    heatmaps = ia.HeatmapsOnImage(np.zeros((3, 3, 1), dtype=np.float32), shape=(3, 3, 3))
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval=255, keep_size=False)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert np.sum(observed.get_arr() > 1e-4) == 0

    # ------------------
    # pad by percentages
    # ------------------
    # pad all sides by 100%
    aug = iaa.Pad(percent=1.0, keep_size=False)
    observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 1)
    assert observed.shape == (4+4+4, 4+4+4)
    assert np.sum(observed[4:-4, 4:-4]) == 4*4
    assert np.sum(observed) == 4*4

    # pad all sides by StochasticParameter
    aug = iaa.Pad(percent=iap.Deterministic(1.0), keep_size=False)
    observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 1)
    assert observed.shape == (4+4+4, 4+4+4)
    assert np.sum(observed[4:-4, 4:-4]) == 4*4
    assert np.sum(observed) == 4*4

    # pad all sides by 100-200%
    aug = iaa.Pad(percent=(1.0, 2.0), sample_independently=False, keep_size=False)
    observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 1)
    assert np.sum(observed) == 4*4
    assert (observed.shape[0] - 4) % 2 == 0
    assert (observed.shape[1] - 4) % 2 == 0

    # pad by invalid value
    got_exception = False
    try:
        aug = iaa.Pad(percent="test", keep_size=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # test pad by 100% on each side
    image = np.zeros((4, 4), dtype=np.uint8)
    image[0, 0] = 255
    image[3, 0] = 255
    image[0, 3] = 255
    image[3, 3] = 255
    height, width = image.shape[0:2]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=3, y=3),
                                      ia.Keypoint(x=3, y=3)], shape=image.shape)]
    pads = [
        (1.0, 0, 0, 0),
        (0, 1.0, 0, 0),
        (0, 0, 1.0, 0),
        (0, 0, 0, 1.0),
    ]
    for pad in pads:
        top, right, bottom, left = pad
        top_px = int(top * height)
        right_px = int(right * width)
        bottom_px = int(bottom * height)
        left_px = int(left * width)
        aug = iaa.Pad(percent=pad, keep_size=False)
        image_padded = np.pad(image, ((top_px, bottom_px), (left_px, right_px)),
                              mode="constant",
                              constant_values=0)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, image_padded)

        keypoints_moved = [keypoints[0].shift(x=left_px, y=top_px)]
        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_moved)

    # test pad by range of percentages
    aug = iaa.Pad(percent=((0, 1.0), 0, 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 255)
        n_padded = 0
        while np.all(observed[0, :] == 0):
            n_padded += 1
            observed = observed[1:, :]
        seen[n_padded] += 1
    # note that we cant just check for 100-50 < x < 100+50 here. The first and last value (0px
    # and 4px) padding have half the probability of occuring compared to the other values.
    # E.g. 0px is padded if sampled p falls in range [0, 0.125). 1px is padded if sampled p
    # falls in range [0.125, 0.375].
    assert all([v > 30 for v in seen])

    aug = iaa.Pad(percent=(0, (0, 1.0), 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 255)
        n_padded = 0
        while np.all(observed[:, -1] == 0):
            n_padded += 1
            observed = observed[:, 0:-1]
        seen[n_padded] += 1
    assert all([v > 30 for v in seen])

    # test pad by list of percentages
    aug = iaa.Pad(percent=([0.0, 1.0], 0, 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 255)
        n_padded = 0
        while np.all(observed[0, :] == 0):
            n_padded += 1
            observed = observed[1:, :]
        seen[n_padded] += 1
    assert 250 - 50 < seen[0] < 250 + 50
    assert seen[1] == 0
    assert seen[2] == 0
    assert seen[3] == 0
    assert 250 - 50 < seen[4] < 250 + 50

    aug = iaa.Pad(percent=(0, [0.0, 1.0], 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 255)
        n_padded = 0
        while np.all(observed[:, -1] == 0):
            n_padded += 1
            observed = observed[:, 0:-1]
        seen[n_padded] += 1
    assert 250 - 50 < seen[0] < 250 + 50
    assert seen[1] == 0
    assert seen[2] == 0
    assert seen[3] == 0
    assert 250 - 50 < seen[4] < 250 + 50


def test_Crop():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    heatmaps_arr = np.float32([[0, 0, 0],
                               [0, 1.0, 0],
                               [0, 0, 0]])

    # test crop by 1 pixel on each side
    crops = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        height, width = base_img.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        base_img_cropped = base_img[top:height-bottom, left:width-right, :]
        observed = aug.augment_images(images)
        assert np.array_equal(observed, np.array([base_img_cropped]))

        observed = aug.augment_images(images_list)
        assert array_equal_lists(observed, [base_img_cropped])

        keypoints_moved = [keypoints[0].shift(x=-left, y=-top)]
        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_moved)

        height, width = heatmaps_arr.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        heatmaps_arr_cropped = heatmaps_arr[top:height-bottom, left:width-right]
        observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr, shape=base_img.shape)])[0]
        assert observed.shape == base_img_cropped.shape
        assert np.array_equal(observed.get_arr(), heatmaps_arr_cropped)

    # test crop by range of pixels
    crops = [
        ((0, 2), 0, 0, 0),
        (0, (0, 2), 0, 0),
        (0, 0, (0, 2), 0),
        (0, 0, 0, (0, 2)),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        height, width = base_img.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        aug_det = aug.to_deterministic()

        images_cropped = []
        keypoints_cropped = []
        top_range = top if isinstance(top, tuple) else (top, top)
        right_range = right if isinstance(right, tuple) else (right, right)
        bottom_range = bottom if isinstance(bottom, tuple) else (bottom, bottom)
        left_range = left if isinstance(left, tuple) else (left, left)
        for top_val in sm.xrange(top_range[0], top_range[1]+1):
            for right_val in sm.xrange(right_range[0], right_range[1]+1):
                for bottom_val in sm.xrange(bottom_range[0], bottom_range[1]+1):
                    for left_val in sm.xrange(left_range[0], left_range[1]+1):

                        images_cropped.append(base_img[top_val:height-bottom_val, left_val:width-right_val, :])
                        keypoints_cropped.append(keypoints[0].shift(x=-left_val, y=-top_val))

        movements = []
        movements_det = []
        for i in sm.xrange(100):
            observed = aug.augment_images(images)

            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0
                       for base_img_cropped in images_cropped]
            movements.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug_det.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0
                       for base_img_cropped in images_cropped]
            movements_det.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug.augment_images(images_list)
            assert any([array_equal_lists(observed, [base_img_cropped])
                        for base_img_cropped in images_cropped])

            observed = aug.augment_keypoints(keypoints)
            assert any([keypoints_equal(observed, [kp]) for kp in keypoints_cropped])

        assert len(set(movements)) == 3
        assert len(set(movements_det)) == 1

    # test crop by list of exact pixel values
    crops = [
        ([0, 2], 0, 0, 0),
        (0, [0, 2], 0, 0),
        (0, 0, [0, 2], 0),
        (0, 0, 0, [0, 2]),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        height, width = base_img.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        aug_det = aug.to_deterministic()

        images_cropped = []
        keypoints_cropped = []
        top_range = top if isinstance(top, list) else [top]
        right_range = right if isinstance(right, list) else [right]
        bottom_range = bottom if isinstance(bottom, list) else [bottom]
        left_range = left if isinstance(left, list) else [left]
        for top_val in top_range:
            for right_val in right_range:
                for bottom_val in bottom_range:
                    for left_val in left_range:
                        images_cropped.append(base_img[top_val:height-bottom_val, left_val:width-right_val, :])
                        keypoints_cropped.append(keypoints[0].shift(x=-left_val, y=-top_val))

        movements = []
        movements_det = []
        for i in sm.xrange(100):
            observed = aug.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0
                       for base_img_cropped in images_cropped]
            movements.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug_det.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0
                       for base_img_cropped in images_cropped]
            movements_det.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug.augment_images(images_list)
            assert any([array_equal_lists(observed, [base_img_cropped])
                        for base_img_cropped in images_cropped])

            observed = aug.augment_keypoints(keypoints)
            assert any([keypoints_equal(observed, [kp]) for kp in keypoints_cropped])

        assert len(set(movements)) == 2
        assert len(set(movements_det)) == 1

    # ------------------
    # crop by percentages
    # ------------------
    # crop all sides by 10%
    aug = iaa.Crop(percent=0.1, keep_size=False)
    image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert observed.shape == (40, 40)
    assert np.all(observed == image[5:-5, 5:-5])

    # crop all sides by StochasticParameter
    aug = iaa.Crop(percent=iap.Deterministic(0.1), keep_size=False)
    image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert observed.shape == (40, 40)
    assert np.all(observed == image[5:-5, 5:-5])

    # crop all sides by 10-20%
    image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    aug = iaa.Crop(percent=(0.1, 0.2), keep_size=False)
    observed = aug.augment_image(image)
    assert 30 <= observed.shape[0] <= 40
    assert 30 <= observed.shape[1] <= 40

    # crop by invalid value
    got_exception = False
    try:
        aug = iaa.Crop(percent="test", keep_size=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # test crop by 10% on each side
    image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    height, width = image.shape[0:2]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=10, y=11), ia.Keypoint(x=20, y=21),
                                      ia.Keypoint(x=30, y=31)], shape=image.shape)]
    crops = [
        (0.1, 0, 0, 0),
        (0, 0.1, 0, 0),
        (0, 0, 0.1, 0),
        (0, 0, 0, 0.1),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        top_px = int(round(top * height))
        right_px = int(round(right * width))
        bottom_px = int(round(bottom * height))
        left_px = int(round(left * width))
        aug = iaa.Crop(percent=crop, keep_size=False)
        image_cropped = image[top_px:50-bottom_px, left_px:50-right_px] # dont use :-bottom_px and ;-right_px here, because these values can be 0
        observed = aug.augment_image(image)
        assert np.array_equal(observed, image_cropped)

        keypoints_moved = [keypoints[0].shift(x=-left_px, y=-top_px)]
        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_moved)

    # test crop by range of percentages
    aug = iaa.Crop(percent=((0, 0.1), 0, 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((40, 40), dtype=np.uint8))
        n_cropped = 40 - observed.shape[0]
        seen[n_cropped] += 1
    # note that we cant just check for 100-50 < x < 100+50 here. The first and last value (0px
    # and 4px) have half the probability of occuring compared to the other values.
    # E.g. 0px is cropped if sampled p falls in range [0, 0.125). 1px is cropped if sampled p
    # falls in range [0.125, 0.375].
    assert all([v > 30 for v in seen])

    aug = iaa.Crop(percent=(0, (0, 0.1), 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((40, 40), dtype=np.uint8) + 255)
        n_cropped = 40 - observed.shape[1]
        seen[n_cropped] += 1
    assert all([v > 30 for v in seen])

    # test crop by list of percentages
    aug = iaa.Crop(percent=([0.0, 0.1], 0, 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((40, 40), dtype=np.uint8) + 255)
        n_cropped = 40 - observed.shape[0]
        seen[n_cropped] += 1
    assert 250 - 50 < seen[0] < 250 + 50
    assert seen[1] == 0
    assert seen[2] == 0
    assert seen[3] == 0
    assert 250 - 50 < seen[4] < 250 + 50

    aug = iaa.Crop(percent=(0, [0.0, 0.1], 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((40, 40), dtype=np.uint8) + 255)
        n_cropped = 40 - observed.shape[1]
        seen[n_cropped] += 1
    assert 250 - 50 < seen[0] < 250 + 50
    assert seen[1] == 0
    assert seen[2] == 0
    assert seen[3] == 0
    assert 250 - 50 < seen[4] < 250 + 50


def test_Fliplr():
    reseed()

    base_img = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    base_img_flipped = np.array([[1, 0, 0],
                                 [1, 0, 0],
                                 [1, 1, 0]], dtype=np.uint8)
    base_img_flipped = base_img_flipped[:, :, np.newaxis]

    images = np.array([base_img])
    images_flipped = np.array([base_img_flipped])

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]
    keypoints_flipped = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=0), ia.Keypoint(x=1, y=1),
                                              ia.Keypoint(x=0, y=2)], shape=base_img.shape)]

    # 0% chance of flip
    aug = iaa.Fliplr(0)
    aug_det = aug.to_deterministic()

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    # 0% chance of flip, heatmaps
    aug = iaa.Fliplr(0)
    heatmaps = ia.HeatmapsOnImage(
        np.float32([
            [0, 0.5, 0.75],
            [0, 0.5, 0.75],
            [0.75, 0.75, 0.75],
        ]),
        shape=(3, 3, 3)
    )
    observed = aug.augment_heatmaps([heatmaps])[0]
    expected = heatmaps.get_arr()
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), expected)

    # 100% chance of flip
    aug = iaa.Fliplr(1.0)
    aug_det = aug.to_deterministic()

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images_flipped
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images_flipped
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints_flipped
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints_flipped
        assert keypoints_equal(observed, expected)

    # 100% chance of flip, heatmaps
    aug = iaa.Fliplr(1.0)
    heatmaps = ia.HeatmapsOnImage(
        np.float32([
            [0, 0.5, 0.75],
            [0, 0.5, 0.75],
            [0.75, 0.75, 0.75],
        ]),
        shape=(3, 3, 3)
    )
    observed = aug.augment_heatmaps([heatmaps])[0]
    expected = np.fliplr(heatmaps.get_arr())
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), expected)

    # 50% chance of flip
    aug = iaa.Fliplr(0.5)
    aug_det = aug.to_deterministic()

    nb_iterations = 1000
    nb_images_flipped = 0
    nb_images_flipped_det = 0
    nb_keypoints_flipped = 0
    nb_keypoints_flipped_det = 0
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_images(images)
        if np.array_equal(observed, images_flipped):
            nb_images_flipped += 1

        observed = aug_det.augment_images(images)
        if np.array_equal(observed, images_flipped):
            nb_images_flipped_det += 1

        observed = aug.augment_keypoints(keypoints)
        if keypoints_equal(observed, keypoints_flipped):
            nb_keypoints_flipped += 1

        observed = aug_det.augment_keypoints(keypoints)
        if keypoints_equal(observed, keypoints_flipped):
            nb_keypoints_flipped_det += 1

    assert int(nb_iterations * 0.3) <= nb_images_flipped <= int(nb_iterations * 0.7)
    assert int(nb_iterations * 0.3) <= nb_keypoints_flipped <= int(nb_iterations * 0.7)
    assert nb_images_flipped_det in [0, nb_iterations]
    assert nb_keypoints_flipped_det in [0, nb_iterations]

    # 50% chance of flipped, multiple images, list as input
    images_multi = [base_img, base_img]
    aug = iaa.Fliplr(0.5)
    aug_det = aug.to_deterministic()
    nb_iterations = 1000
    nb_flipped_by_pos = [0] * len(images_multi)
    nb_flipped_by_pos_det = [0] * len(images_multi)
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_images(images_multi)
        for i in sm.xrange(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos[i] += 1

        observed = aug_det.augment_images(images_multi)
        for i in sm.xrange(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos_det[i] += 1

    for val in nb_flipped_by_pos:
        assert int(nb_iterations * 0.3) <= val <= int(nb_iterations * 0.7)

    for val in nb_flipped_by_pos_det:
        assert val in [0, nb_iterations]

    # test StochasticParameter as p
    aug = iaa.Fliplr(p=iap.Choice([0, 1], p=[0.7, 0.3]))
    seen = [0, 0]
    for _ in sm.xrange(1000):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, base_img):
            seen[0] += 1
        elif np.array_equal(observed, base_img_flipped):
            seen[1] += 1
        else:
            assert False
    assert 700 - 75 < seen[0] < 700 + 75
    assert 300 - 75 < seen[1] < 300 + 75

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.Fliplr(p="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.Fliplr(p=0.5)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Binomial)
    assert isinstance(params[0].p, iap.Deterministic)
    assert 0.5 - 1e-4 < params[0].p.value < 0.5 + 1e-4


def test_Flipud():
    reseed()

    base_img = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    base_img_flipped = np.array([[0, 1, 1],
                                 [0, 0, 1],
                                 [0, 0, 1]], dtype=np.uint8)
    base_img_flipped = base_img_flipped[:, :, np.newaxis]

    images = np.array([base_img])
    images_flipped = np.array([base_img_flipped])

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]
    keypoints_flipped = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=2), ia.Keypoint(x=1, y=1),
                                              ia.Keypoint(x=2, y=0)], shape=base_img.shape)]

    # 0% chance of flip
    aug = iaa.Flipud(0)
    aug_det = aug.to_deterministic()

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    # 0% chance of flip, heatmaps
    aug = iaa.Flipud(0)
    heatmaps = ia.HeatmapsOnImage(
        np.float32([
            [0, 0.5, 0.75],
            [0, 0.5, 0.75],
            [0.75, 0.75, 0.75],
        ]),
        shape=(3, 3, 3)
    )
    observed = aug.augment_heatmaps([heatmaps])[0]
    expected = heatmaps.get_arr()
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), expected)

    # 100% chance of flip
    aug = iaa.Flipud(1.0)
    aug_det = aug.to_deterministic()

    for _ in sm.xrange(10):
        observed = aug.augment_images(images)
        expected = images_flipped
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images_flipped
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints_flipped
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints_flipped
        assert keypoints_equal(observed, expected)

    # 100% chance of flip, heatmaps
    aug = iaa.Flipud(1.0)
    heatmaps = ia.HeatmapsOnImage(
        np.float32([
            [0, 0.5, 0.75],
            [0, 0.5, 0.75],
            [0.75, 0.75, 0.75],
        ]),
        shape=(3, 3, 3)
    )
    observed = aug.augment_heatmaps([heatmaps])[0]
    expected = np.flipud(heatmaps.get_arr())
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), expected)

    # 50% chance of flip
    aug = iaa.Flipud(0.5)
    aug_det = aug.to_deterministic()

    nb_iterations = 1000
    nb_images_flipped = 0
    nb_images_flipped_det = 0
    nb_keypoints_flipped = 0
    nb_keypoints_flipped_det = 0
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_images(images)
        if np.array_equal(observed, images_flipped):
            nb_images_flipped += 1

        observed = aug_det.augment_images(images)
        if np.array_equal(observed, images_flipped):
            nb_images_flipped_det += 1

        observed = aug.augment_keypoints(keypoints)
        if keypoints_equal(observed, keypoints_flipped):
            nb_keypoints_flipped += 1

        observed = aug_det.augment_keypoints(keypoints)
        if keypoints_equal(observed, keypoints_flipped):
            nb_keypoints_flipped_det += 1

    assert int(nb_iterations * 0.3) <= nb_images_flipped <= int(nb_iterations * 0.7)
    assert int(nb_iterations * 0.3) <= nb_keypoints_flipped <= int(nb_iterations * 0.7)
    assert nb_images_flipped_det in [0, nb_iterations]
    assert nb_keypoints_flipped_det in [0, nb_iterations]

    # 50% chance of flipped, multiple images, list as input
    images_multi = [base_img, base_img]
    aug = iaa.Flipud(0.5)
    aug_det = aug.to_deterministic()
    nb_iterations = 1000
    nb_flipped_by_pos = [0] * len(images_multi)
    nb_flipped_by_pos_det = [0] * len(images_multi)
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_images(images_multi)
        for i in sm.xrange(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos[i] += 1

        observed = aug_det.augment_images(images_multi)
        for i in sm.xrange(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos_det[i] += 1

    for val in nb_flipped_by_pos:
        assert int(nb_iterations * 0.3) <= val <= int(nb_iterations * 0.7)

    for val in nb_flipped_by_pos_det:
        assert val in [0, nb_iterations]

    # test StochasticParameter as p
    aug = iaa.Flipud(p=iap.Choice([0, 1], p=[0.7, 0.3]))
    seen = [0, 0]
    for _ in sm.xrange(1000):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, base_img):
            seen[0] += 1
        elif np.array_equal(observed, base_img_flipped):
            seen[1] += 1
        else:
            assert False
    assert 700 - 75 < seen[0] < 700 + 75
    assert 300 - 75 < seen[1] < 300 + 75

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.Flipud(p="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.Flipud(p=0.5)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Binomial)
    assert isinstance(params[0].p, iap.Deterministic)
    assert 0.5 - 1e-4 < params[0].p.value < 0.5 + 1e-4


def test_GaussianBlur():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 255, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]
    outer_pixels = ([], [])
    for i in sm.xrange(base_img.shape[0]):
        for j in sm.xrange(base_img.shape[1]):
            if i != j:
                outer_pixels[0].append(i)
                outer_pixels[1].append(j)

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no blur, shouldnt change anything
    aug = iaa.GaussianBlur(sigma=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    # weak blur of center pixel
    aug = iaa.GaussianBlur(sigma=0.5)
    aug_det = aug.to_deterministic()

    #np.set_printoptions(formatter={'float_kind': lambda x: "%.6f" % x})
    #from scipy import ndimage
    #images2 = np.copy(images).astype(np.float32)
    #images2[0, ...] = ndimage.gaussian_filter(images2[0, ...], 0.4)
    #print(images2)

    # images as numpy array
    observed = aug.augment_images(images)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    observed = aug_det.augment_images(images)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    # images as list
    observed = aug.augment_images(images_list)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    observed = aug_det.augment_images(images_list)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    # keypoints shouldnt be changed
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # varying blur sigmas
    aug = iaa.GaussianBlur(sigma=(0, 1))
    aug_det = aug.to_deterministic()

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.8)
    assert nb_changed_aug_det == 0


def test_AverageBlur():
    reseed()

    base_img = np.zeros((11, 11, 1), dtype=np.uint8)
    base_img[5, 5, 0] = 200
    base_img[4, 5, 0] = 100
    base_img[6, 5, 0] = 100
    base_img[5, 4, 0] = 100
    base_img[5, 6, 0] = 100

    blur3x3 = np.copy(base_img)
    blur3x3 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 11, 11, 11, 0, 0, 0, 0],
        [0, 0, 0, 11, 44, 56, 44, 11, 0, 0, 0],
        [0, 0, 0, 11, 56, 67, 56, 11, 0, 0, 0],
        [0, 0, 0, 11, 44, 56, 44, 11, 0, 0, 0],
        [0, 0, 0, 0, 11, 11, 11, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    blur3x3 = np.array(blur3x3, dtype=np.uint8)[..., np.newaxis]

    blur4x4 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0],
        [0, 0, 0, 6, 25, 31, 31, 25, 6, 0, 0],
        [0, 0, 0, 6, 31, 38, 38, 31, 6, 0, 0],
        [0, 0, 0, 6, 31, 38, 38, 31, 6, 0, 0],
        [0, 0, 0, 6, 25, 31, 31, 25, 6, 0, 0],
        [0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    blur4x4 = np.array(blur4x4, dtype=np.uint8)[..., np.newaxis]

    blur5x5 = np.copy(base_img)
    blur5x5 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0],
        [0, 0, 4, 16, 20, 20, 20, 16, 4, 0, 0],
        [0, 0, 4, 20, 24, 24, 24, 20, 4, 0, 0],
        [0, 0, 4, 20, 24, 24, 24, 20, 4, 0, 0],
        [0, 0, 4, 20, 24, 24, 24, 20, 4, 0, 0],
        [0, 0, 4, 16, 20, 20, 20, 16, 4, 0, 0],
        [0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    blur5x5 = np.array(blur5x5, dtype=np.uint8)[..., np.newaxis]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no blur, shouldnt change anything
    aug = iaa.AverageBlur(k=0)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, base_img)

    # k=3
    aug = iaa.AverageBlur(k=3)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, blur3x3)

    # k=5
    aug = iaa.AverageBlur(k=5)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, blur5x5)

    # k as (3, 4)
    aug = iaa.AverageBlur(k=(3, 4))
    nb_iterations = 100
    nb_seen = [0, 0]
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, blur3x3):
            nb_seen[0] += 1
        elif np.array_equal(observed, blur4x4):
            nb_seen[1] += 1
        else:
            raise Exception("Unexpected result in AverageBlur@1")
    p_seen = [v/nb_iterations for v in nb_seen]
    assert 0.4 <= p_seen[0] <= 0.6
    assert 0.4 <= p_seen[1] <= 0.6

    # k as (3, 5)
    aug = iaa.AverageBlur(k=(3, 5))
    nb_iterations = 100
    nb_seen = [0, 0, 0]
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, blur3x3):
            nb_seen[0] += 1
        elif np.array_equal(observed, blur4x4):
            nb_seen[1] += 1
        elif np.array_equal(observed, blur5x5):
            nb_seen[2] += 1
        else:
            raise Exception("Unexpected result in AverageBlur@2")
    p_seen = [v/nb_iterations for v in nb_seen]
    assert 0.23 <= p_seen[0] <= 0.43
    assert 0.23 <= p_seen[1] <= 0.43
    assert 0.23 <= p_seen[2] <= 0.43

    # k as stochastic parameter
    aug = iaa.AverageBlur(k=iap.Choice([3, 5]))
    nb_iterations = 100
    nb_seen = [0, 0]
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, blur3x3):
            nb_seen[0] += 1
        elif np.array_equal(observed, blur5x5):
            nb_seen[1] += 1
        else:
            raise Exception("Unexpected result in AverageBlur@3")
    p_seen = [v/nb_iterations for v in nb_seen]
    assert 0.4 <= p_seen[0] <= 0.6
    assert 0.4 <= p_seen[1] <= 0.6

    # k as ((3, 5), (3, 5))
    aug = iaa.AverageBlur(k=((3, 5), (3, 5)))

    possible = dict()
    for kh in [3, 4, 5]:
        for kw in [3, 4, 5]:
            key = (kh, kw)
            if kh == 0 or kw == 0:
                possible[key] = np.copy(base_img)
            else:
                possible[key] = cv2.blur(base_img, (kh, kw))[..., np.newaxis]

    nb_iterations = 250
    #nb_seen = [0] * len(possible.keys())
    nb_seen = dict([(key, 0) for key, val in possible.items()])
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        for key, img_aug in possible.items():
            if np.array_equal(observed, img_aug):
                nb_seen[key] += 1
    # dont check sum here, because 0xX and Xx0 are all the same, i.e. much
    # higher sum than nb_iterations
    assert all([v > 0 for v in nb_seen.values()])

    # keypoints shouldnt be changed
    aug = iaa.AverageBlur(k=3)
    aug_det = aug.to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)


def test_MedianBlur():
    reseed()

    base_img = np.zeros((11, 11, 1), dtype=np.uint8)
    base_img[3:8, 3:8, 0] = 1
    base_img[4:7, 4:7, 0] = 2
    base_img[5:6, 5:6, 0] = 3

    blur3x3 = np.zeros_like(base_img)
    blur3x3[3:8, 3:8, 0] = 1
    blur3x3[4:7, 4:7, 0] = 2
    blur3x3[4, 4, 0] = 1
    blur3x3[4, 6, 0] = 1
    blur3x3[6, 4, 0] = 1
    blur3x3[6, 6, 0] = 1
    blur3x3[3, 3, 0] = 0
    blur3x3[3, 7, 0] = 0
    blur3x3[7, 3, 0] = 0
    blur3x3[7, 7, 0] = 0

    blur5x5 = np.copy(blur3x3)
    blur5x5[4, 3, 0] = 0
    blur5x5[3, 4, 0] = 0
    blur5x5[6, 3, 0] = 0
    blur5x5[7, 4, 0] = 0
    blur5x5[4, 7, 0] = 0
    blur5x5[3, 6, 0] = 0
    blur5x5[6, 7, 0] = 0
    blur5x5[7, 6, 0] = 0
    blur5x5[blur5x5 > 1] = 1
    #blur5x5 = np.zeros_like(base_img)
    #blur5x5[2:9, 2:9, 0] = 1
    #blur5x5[3:8, 3:8, 0] = 1
    #blur5x5[4:7, 4:7, 0] = 1
    #blur5x5[5:6, 5:6, 0] = 1

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no blur, shouldnt change anything
    aug = iaa.MedianBlur(k=1)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, base_img)

    # k=3
    aug = iaa.MedianBlur(k=3)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, blur3x3)

    # k=5
    aug = iaa.MedianBlur(k=5)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, blur5x5)

    # k as (3, 5)
    aug = iaa.MedianBlur(k=(3, 5))
    seen = [False, False]
    for i in sm.xrange(100):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, blur3x3):
            seen[0] = True
        elif np.array_equal(observed, blur5x5):
            seen[1] = True
        else:
            raise Exception("Unexpected result in MedianBlur@1")
        if all(seen):
            break
    assert all(seen)

    # k as stochastic parameter
    aug = iaa.MedianBlur(k=iap.Choice([3, 5]))
    seen = [False, False]
    for i in sm.xrange(100):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, blur3x3):
            seen[0] += True
        elif np.array_equal(observed, blur5x5):
            seen[1] += True
        else:
            raise Exception("Unexpected result in MedianBlur@2")
        if all(seen):
            break
    assert all(seen)

    # keypoints shouldnt be changed
    aug = iaa.MedianBlur(k=3)
    aug_det = aug.to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)


def test_AddToHueAndSaturation():
    reseed()

    # interestingly, when using this RGB2HSV and HSV2RGB conversion from skimage, the results
    # differ quite a bit from the cv2 ones
    """
    def _add_hue_saturation(img, value):
        img_hsv = color.rgb2hsv(img / 255.0)
        img_hsv[..., 0:2] += (value / 255.0)
        return color.hsv2rgb(img_hsv) * 255
    """

    def _add_hue_saturation(img, value):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv[..., 0:2] += value
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    base_img = np.zeros((2, 2, 3), dtype=np.uint8)
    base_img[..., 0] += 20
    base_img[..., 1] += 40
    base_img[..., 2] += 60

    aug = iaa.AddToHueAndSaturation(0)
    observed = aug.augment_image(base_img)
    expected = base_img
    assert np.allclose(observed, expected)

    aug = iaa.AddToHueAndSaturation(30)
    observed = aug.augment_image(base_img)
    expected = _add_hue_saturation(base_img, 30)
    diff = np.abs(observed.astype(np.float32) - expected)
    assert np.all(diff <= 3)

    aug = iaa.AddToHueAndSaturation((0, 2))
    base_img = base_img[0:1, 0:1, :]
    expected_imgs = [
        iaa.AddToHueAndSaturation(0).augment_image(base_img),
        iaa.AddToHueAndSaturation(1).augment_image(base_img),
        iaa.AddToHueAndSaturation(2).augment_image(base_img)
    ]
    assert not np.array_equal(expected_imgs[0], expected_imgs[1])
    assert not np.array_equal(expected_imgs[1], expected_imgs[2])
    assert not np.array_equal(expected_imgs[0], expected_imgs[2])
    nb_iterations = 300
    seen = dict([(i, 0) for i, _ in enumerate(expected_imgs)])
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        for i, expected_img in enumerate(expected_imgs):
            if np.allclose(observed, expected_img):
                seen[i] += 1
    assert np.sum(list(seen.values())) == nb_iterations
    n_exp = nb_iterations / 3
    n_exp_tol = nb_iterations * 0.1
    assert all([n_exp - n_exp_tol < v < n_exp + n_exp_tol for v in seen.values()])


def test_Grayscale():
    reseed()

    def _compute_luminosity(r, g, b):
        return 0.21 * r + 0.72 * g + 0.07 * b

    base_img = np.zeros((4, 4, 3), dtype=np.uint8)
    base_img[..., 0] += 10
    base_img[..., 1] += 20
    base_img[..., 2] += 30

    aug = iaa.Grayscale(0.0)
    observed = aug.augment_image(base_img)
    expected = np.copy(base_img)
    assert np.allclose(observed, expected)

    aug = iaa.Grayscale(1.0)
    observed = aug.augment_image(base_img)
    luminosity = _compute_luminosity(10, 20, 30)
    expected = np.zeros_like(base_img) + luminosity
    assert np.allclose(observed, expected.astype(np.uint8))

    aug = iaa.Grayscale(0.5)
    observed = aug.augment_image(base_img)
    luminosity = _compute_luminosity(10, 20, 30)
    expected = 0.5 * base_img + 0.5 * luminosity
    assert np.allclose(observed, expected.astype(np.uint8))

    aug = iaa.Grayscale((0.0, 1.0))
    base_img = base_img[0:1, 0:1, :]
    base_img_gray = iaa.Grayscale(1.0).augment_image(base_img)
    distance_max = np.average(np.abs(base_img_gray.astype(np.int32) - base_img.astype(np.int32)))
    nb_iterations = 1000
    distances = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        distance = np.average(np.abs(observed.astype(np.int32) - base_img.astype(np.int32))) / distance_max
        distances.append(distance)

    assert 0 - 1e-4 < min(distances) < 0.1
    assert 0.4 < np.average(distances) < 0.6
    assert 0.9 < max(distances) < 1.0 + 1e-4

    nb_bins = 5
    hist, _ = np.histogram(distances, bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / nb_iterations
        assert density_expected - density_tolerance < density < density_expected + density_tolerance


def test_Convolve():
    reseed()

    img = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    img = np.uint8(img)

    # matrix is None
    aug = iaa.Convolve(matrix=None)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: [None])
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    # matrix is [[1]]
    aug = iaa.Convolve(matrix=np.float32([[1]]))
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: np.float32([[1]]))
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    # matrix is [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    m = np.float32([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    aug = iaa.Convolve(matrix=m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    # matrix is [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    m = np.float32([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0]
    ])
    aug = iaa.Convolve(matrix=m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, 2*img)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, 2*img)

    # matrix is [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    # with 3 channels
    m = np.float32([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0]
    ])
    img3 = np.tile(img[..., np.newaxis], (1, 1, 3))
    aug = iaa.Convolve(matrix=m)
    observed = aug.augment_image(img3)
    assert np.array_equal(observed, 2*img3)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
    observed = aug.augment_image(img3)
    assert np.array_equal(observed, 2*img3)

    # matrix is [[0, -1, 0], [0, 10, 0], [0, 0, 0]]
    m = np.float32([
        [0, -1, 0],
        [0, 10, 0],
        [0, 0, 0]
    ])
    expected = np.uint8([
        [10*1+(-1)*4, 10*2+(-1)*5, 10*3+(-1)*6],
        [10*4+(-1)*1, 10*5+(-1)*2, 10*6+(-1)*3],
        [10*7+(-1)*4, 10*8+(-1)*5, 10*9+(-1)*6]
    ])

    aug = iaa.Convolve(matrix=m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, expected)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, expected)

    # changing matrices when using callable
    expected = []
    for i in sm.xrange(5):
        expected.append(img * i)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: np.float32([[random_state.randint(0, 5)]]))
    seen = [False] * 5
    for _ in sm.xrange(200):
        observed = aug.augment_image(img)
        found = False
        for i, expected_i in enumerate(expected):
            if np.array_equal(observed, expected_i):
                seen[i] = True
                found = True
                break
        assert found
        if all(seen):
            break
    assert all(seen)

    # bad datatype for matrix
    got_exception = False
    try:
        aug = iaa.Convolve(matrix=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # get_parameters()
    matrix = np.int32([[1]])
    aug = iaa.Convolve(matrix=matrix)
    params = aug.get_parameters()
    assert np.array_equal(params[0], matrix)
    assert params[1] == "constant"

    # TODO add test for keypoints once their handling was improved in Convolve


def test_Sharpen():
    reseed()

    def _compute_sharpened_base_img(lightness, m):
        base_img_sharpened = np.zeros((3, 3), dtype=np.float32)
        k = 1
        # note that cv2 uses reflection padding by default
        base_img_sharpened[0, 0] = (m[1, 1] + lightness)/k * 10 + 4 * (m[0, 0]/k) * 10 + 4 * (m[2, 2]/k) * 20
        base_img_sharpened[0, 2] = base_img_sharpened[0, 0]
        base_img_sharpened[2, 0] = base_img_sharpened[0, 0]
        base_img_sharpened[2, 2] = base_img_sharpened[0, 0]
        base_img_sharpened[0, 1] = (m[1, 1] + lightness)/k * 10 + 6 * (m[0, 1]/k) * 10 + 2 * (m[2, 2]/k) * 20
        base_img_sharpened[1, 0] = base_img_sharpened[0, 1]
        base_img_sharpened[1, 2] = base_img_sharpened[0, 1]
        base_img_sharpened[2, 1] = base_img_sharpened[0, 1]
        base_img_sharpened[1, 1] = (m[1, 1] + lightness)/k * 20 + 8 * (m[0, 1]/k) * 10

        #print("A", base_img_sharpened, "Am", m)
        base_img_sharpened = np.clip(base_img_sharpened, 0, 255).astype(np.uint8)

        return base_img_sharpened

    base_img = [[10, 10, 10],
                [10, 20, 10],
                [10, 10, 10]]
    base_img = np.uint8(base_img)
    m = np.float32([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
    m_noop = np.float32([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])
    base_img_sharpened = _compute_sharpened_base_img(1, m)

    aug = iaa.Sharpen(alpha=0, lightness=1)
    observed = aug.augment_image(base_img)
    expected = base_img
    assert np.allclose(observed, expected)

    aug = iaa.Sharpen(alpha=1.0, lightness=1)
    observed = aug.augment_image(base_img)
    expected = base_img_sharpened
    assert np.allclose(observed, expected)

    aug = iaa.Sharpen(alpha=0.5, lightness=1)
    observed = aug.augment_image(base_img)
    expected = _compute_sharpened_base_img(0.5*1, 0.5 * m_noop + 0.5 * m)
    assert np.allclose(observed, expected.astype(np.uint8))

    aug = iaa.Sharpen(alpha=0.75, lightness=1)
    observed = aug.augment_image(base_img)
    expected = _compute_sharpened_base_img(0.75*1, 0.25 * m_noop + 0.75 * m)
    assert np.allclose(observed, expected)

    aug = iaa.Sharpen(alpha=iap.Choice([0.5, 1.0]), lightness=1)
    observed = aug.augment_image(base_img)
    expected1 = _compute_sharpened_base_img(0.5*1, m)
    expected2 = _compute_sharpened_base_img(1.0*1, m)
    assert np.allclose(observed, expected1) or np.allclose(observed, expected2)

    got_exception = False
    try:
        aug = iaa.Sharpen(alpha="test", lightness=1)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    aug = iaa.Sharpen(alpha=1.0, lightness=2)
    observed = aug.augment_image(base_img)
    expected = _compute_sharpened_base_img(1.0*2, m)
    assert np.allclose(observed, expected)

    aug = iaa.Sharpen(alpha=1.0, lightness=3)
    observed = aug.augment_image(base_img)
    expected = _compute_sharpened_base_img(1.0*3, m)
    assert np.allclose(observed, expected)

    aug = iaa.Sharpen(alpha=1.0, lightness=iap.Choice([1.0, 1.5]))
    observed = aug.augment_image(base_img)
    expected1 = _compute_sharpened_base_img(1.0*1.0, m)
    expected2 = _compute_sharpened_base_img(1.0*1.5, m)
    assert np.allclose(observed, expected1) or np.allclose(observed, expected2)

    got_exception = False
    try:
        aug = iaa.Sharpen(alpha=1.0, lightness="test")
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # this part doesnt really work so far due to nonlinearities resulting from clipping to uint8
    """
    # alpha range
    aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=1)
    base_img = np.copy(base_img)
    base_img_sharpened_min = _compute_sharpened_base_img(0.0*1, 1.0 * m_noop + 0.0 * m)
    base_img_sharpened_max = _compute_sharpened_base_img(1.0*1, 0.0 * m_noop + 1.0 * m)
    #distance_max = np.average(np.abs(base_img_sharpened.astype(np.float32) - base_img.astype(np.float32)))
    distance_max = np.average(np.abs(base_img_sharpened_max - base_img_sharpened_min))
    nb_iterations = 250
    distances = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        distance = np.average(np.abs(observed.astype(np.float32) - base_img_sharpened_max.astype(np.float32))) / distance_max
        distances.append(distance)

    print(distances)
    print(min(distances), np.average(distances), max(distances))
    assert 0 - 1e-4 < min(distances) < 0.1
    assert 0.4 < np.average(distances) < 0.6
    assert 0.9 < max(distances) < 1.0 + 1e-4

    nb_bins = 5
    hist, _ = np.histogram(distances, bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / nb_iterations
        assert density_expected - density_tolerance < density < density_expected + density_tolerance

    # lightness range
    aug = iaa.Sharpen(alpha=1.0, lightness=(0.5, 2.0))
    base_img = np.copy(base_img)
    base_img_sharpened = _compute_sharpened_base_img(1.0*2.0, m)
    distance_max = np.average(np.abs(base_img_sharpened.astype(np.int32) - base_img.astype(np.int32)))
    nb_iterations = 250
    distances = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        distance = np.average(np.abs(observed.astype(np.int32) - base_img.astype(np.int32))) / distance_max
        distances.append(distance)

    assert 0 - 1e-4 < min(distances) < 0.1
    assert 0.4 < np.average(distances) < 0.6
    assert 0.9 < max(distances) < 1.0 + 1e-4

    nb_bins = 5
    hist, _ = np.histogram(distances, bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / nb_iterations
        assert density_expected - density_tolerance < density < density_expected + density_tolerance
    """


def test_Emboss():
    reseed()

    base_img = [[10, 10, 10],
                [10, 20, 10],
                [10, 10, 15]]
    base_img = np.uint8(base_img)

    def _compute_embossed_base_img(img, alpha, strength):
        img = np.copy(img)
        base_img_embossed = np.zeros((3, 3), dtype=np.float32)

        m = np.float32([[-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        strength_matrix = strength * np.float32([
            [-1, -1, 0],
            [-1, 0, 1],
            [0, 1, 1]
        ])
        ms = m + strength_matrix

        #print(ms)
        for i in range(base_img_embossed.shape[0]):
            for j in range(base_img_embossed.shape[1]):
                for u in range(ms.shape[0]):
                    for v in range(ms.shape[1]):
                        weight = ms[u, v]
                        inputs_i = abs(i + (u - (ms.shape[0]-1)//2))
                        inputs_j = abs(j + (v - (ms.shape[1]-1)//2))
                        #print("in1", inputs_i, inputs_j)
                        #print("A", i, j, u, v, "|", inputs_i, inputs_j, "|", None, weight, "->", None)
                        if inputs_i >= img.shape[0]:
                            diff = inputs_i - (img.shape[0]-1)
                            inputs_i = img.shape[0] - 1 - diff
                        if inputs_j >= img.shape[1]:
                            diff = inputs_j - (img.shape[1]-1)
                            inputs_j = img.shape[1] - 1 - diff
                        #print("in2", inputs_i, inputs_j)
                        inputs = img[inputs_i, inputs_j]
                        #print("B", i, j, u, v, "|", inputs_i, inputs_j, "|", inputs, weight, "->", inputs * weight)
                        base_img_embossed[i, j] += inputs * weight
        #print(ms)
        #print(base_img_embossed)

        return np.clip((1-alpha) * img + alpha * base_img_embossed, 0, 255).astype(np.uint8)

    def _allclose(a, b):
        return np.max(a.astype(np.float32) - b.astype(np.float32)) <= 2.1

    aug = iaa.Emboss(alpha=0, strength=1)
    observed = aug.augment_image(base_img)
    expected = base_img
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=1.0, strength=1)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=1.0, strength=1)
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=0.5, strength=1)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=0.5, strength=1)
    assert _allclose(observed, expected.astype(np.uint8))

    aug = iaa.Emboss(alpha=0.75, strength=1)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=0.75, strength=1)
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=iap.Choice([0.5, 1.0]), strength=1)
    observed = aug.augment_image(base_img)
    expected1 = _compute_embossed_base_img(base_img, alpha=0.5, strength=1)
    expected2 = _compute_embossed_base_img(base_img, alpha=1.0, strength=1)
    assert _allclose(observed, expected1) or np.allclose(observed, expected2)

    got_exception = False
    try:
        aug = iaa.Emboss(alpha="test", strength=1)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    aug = iaa.Emboss(alpha=1.0, strength=2)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=1.0, strength=2)
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=1.0, strength=3)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=1.0, strength=3)
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=1.0, strength=6)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=1.0, strength=6)
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=1.0, strength=iap.Choice([1.0, 1.5]))
    observed = aug.augment_image(base_img)
    expected1 = _compute_embossed_base_img(base_img, alpha=1.0, strength=1.0)
    expected2 = _compute_embossed_base_img(base_img, alpha=1.0, strength=1.5)
    assert _allclose(observed, expected1) or np.allclose(observed, expected2)

    got_exception = False
    try:
        aug = iaa.Emboss(alpha=1.0, strength="test")
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception


def test_AdditiveGaussianNoise():
    reseed()

    #base_img = np.array([[128, 128, 128],
    #                     [128, 128, 128],
    #                     [128, 128, 128]], dtype=np.uint8)
    base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128
    #base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no noise, shouldnt change anything
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    # zero-centered noise
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.2 * 255)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert not np.array_equal(observed, images)

    observed = aug_det.augment_images(images)
    assert not np.array_equal(observed, images)

    observed = aug.augment_images(images_list)
    assert not array_equal_lists(observed, images_list)

    observed = aug_det.augment_images(images_list)
    assert not array_equal_lists(observed, images_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints)

    # std correct?
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.2 * 255)
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    nb_iterations = 1000
    values = []
    for i in sm.xrange(nb_iterations):
        images_aug = aug.augment_images(images)
        values.append(images_aug[0, 0, 0, 0])
    values = np.array(values)
    assert np.min(values) == 0
    assert 0.1 < np.std(values) / 255.0 < 0.4

    # non-zero loc
    aug = iaa.AdditiveGaussianNoise(loc=0.25 * 255, scale=0.01 * 255)
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    nb_iterations = 1000
    values = []
    for i in sm.xrange(nb_iterations):
        images_aug = aug.augment_images(images)
        values.append(images_aug[0, 0, 0, 0] - 128)
    values = np.array(values)
    assert 54 < np.average(values) < 74 # loc=0.25 should be around 255*0.25=64 average

    # varying locs
    aug = iaa.AdditiveGaussianNoise(loc=(0, 0.5 * 255), scale=0.0001 * 255)
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.95)
    assert nb_changed_aug_det == 0

    # varying locs by stochastic param
    aug = iaa.AdditiveGaussianNoise(loc=iap.Choice([-20, 20]), scale=0.0001 * 255)
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    seen = [0, 0]
    for i in sm.xrange(200):
        observed = aug.augment_images(images)
        mean = np.mean(observed)
        diff_m20 = abs(mean - (128-20))
        diff_p20 = abs(mean - (128+20))
        if diff_m20 <= 1:
            seen[0] += 1
        elif diff_p20 <= 1:
            seen[1] += 1
        else:
            assert False
    assert 75 < seen[0] < 125
    assert 75 < seen[1] < 125

    # varying stds
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=(0.01 * 255, 0.2 * 255))
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.95)
    assert nb_changed_aug_det == 0

    # varying stds by stochastic param
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=iap.Choice([1, 20]))
    images = np.ones((1, 20, 20, 1), dtype=np.uint8) * 128
    seen = [0, 0, 0]
    for i in sm.xrange(200):
        observed = aug.augment_images(images)
        std = np.std(observed.astype(np.int32) - 128)
        diff_1 = abs(std - 1)
        diff_20 = abs(std - 20)
        if diff_1 <= 2:
            seen[0] += 1
        elif diff_20 <= 5:
            seen[1] += 1
        else:
            seen[2] += 1
    assert seen[2] <= 5
    assert 75 < seen[0] < 125
    assert 75 < seen[1] < 125

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.AdditiveGaussianNoise(loc="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.AdditiveGaussianNoise(scale="test")
    except Exception:
        got_exception = True
    assert got_exception


#def test_MultiplicativeGaussianNoise():
#    pass

#def test_ReplacingGaussianNoise():
#    pass


def test_Dropout():
    reseed()

    base_img = np.ones((512, 512, 1), dtype=np.uint8) * 255

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no dropout, shouldnt change anything
    aug = iaa.Dropout(p=0)
    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    # 100% dropout, should drop everything
    aug = iaa.Dropout(p=1.0)
    observed = aug.augment_images(images)
    expected = np.zeros((1, 512, 512, 1), dtype=np.uint8)
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [np.zeros((512, 512, 1), dtype=np.uint8)]
    assert array_equal_lists(observed, expected)

    # 50% dropout
    aug = iaa.Dropout(p=0.5)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert not np.array_equal(observed, images)
    percent_nonzero = len(observed.flatten().nonzero()[0]) \
                      / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
    assert 0.35 <= (1 - percent_nonzero) <= 0.65

    observed = aug_det.augment_images(images)
    assert not np.array_equal(observed, images)
    percent_nonzero = len(observed.flatten().nonzero()[0]) \
                      / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
    assert 0.35 <= (1 - percent_nonzero) <= 0.65

    observed = aug.augment_images(images_list)
    assert not array_equal_lists(observed, images_list)
    percent_nonzero = len(observed[0].flatten().nonzero()[0]) \
                      / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
    assert 0.35 <= (1 - percent_nonzero) <= 0.65

    observed = aug_det.augment_images(images_list)
    assert not array_equal_lists(observed, images_list)
    percent_nonzero = len(observed[0].flatten().nonzero()[0]) \
                      / (base_img.shape[0] * base_img.shape[1] * base_img.shape[2])
    assert 0.35 <= (1 - percent_nonzero) <= 0.65

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints)

    # varying p
    aug = iaa.Dropout(p=(0.0, 1.0))
    aug_det = aug.to_deterministic()
    images = np.ones((1, 8, 8, 1), dtype=np.uint8) * 255
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.95)
    assert nb_changed_aug_det == 0

    # varying p by stochastic parameter
    aug = iaa.Dropout(p=iap.Binomial(1-iap.Choice([0.0, 0.5])))
    images = np.ones((1, 20, 20, 1), dtype=np.uint8) * 255
    seen = [0, 0, 0]
    for i in sm.xrange(400):
        observed = aug.augment_images(images)
        p = np.mean(observed == 0)
        if 0.4 < p < 0.6:
            seen[0] += 1
        elif p < 0.1:
            seen[1] += 1
        else:
            seen[2] += 1
    assert seen[2] <= 10
    assert 150 < seen[0] < 250
    assert 150 < seen[1] < 250

    # test exception for wrong parameter datatype
    got_exception = False
    try:
        aug = iaa.Dropout(p="test")
    except Exception:
        got_exception = True
    assert got_exception


def test_CoarseDropout():
    reseed()

    base_img = np.ones((16, 16, 1), dtype=np.uint8) * 100

    aug = iaa.CoarseDropout(p=0, size_px=4, size_percent=None, per_channel=False, min_size=4)
    observed = aug.augment_image(base_img)
    expected = base_img
    assert np.array_equal(observed, expected)

    aug = iaa.CoarseDropout(p=1.0, size_px=4, size_percent=None, per_channel=False, min_size=4)
    observed = aug.augment_image(base_img)
    expected = np.zeros_like(base_img)
    assert np.array_equal(observed, expected)

    aug = iaa.CoarseDropout(p=0.5, size_px=1, size_percent=None, per_channel=False, min_size=1)
    averages = []
    for _ in sm.xrange(50):
        observed = aug.augment_image(base_img)
        averages.append(np.average(observed))
    assert all([v in [0, 100] for v in averages])
    assert 50 - 20 < np.average(averages) < 50 + 20

    aug = iaa.CoarseDropout(p=0.5, size_px=None, size_percent=0.001, per_channel=False, min_size=1)
    averages = []
    for _ in sm.xrange(50):
        observed = aug.augment_image(base_img)
        averages.append(np.average(observed))
    assert all([v in [0, 100] for v in averages])
    assert 50 - 20 < np.average(averages) < 50 + 20

    aug = iaa.CoarseDropout(p=0.5, size_px=1, size_percent=None, per_channel=True, min_size=1)
    base_img = np.ones((4, 4, 3), dtype=np.uint8) * 100
    found = False
    for _ in sm.xrange(100):
        observed = aug.augment_image(base_img)
        avgs = np.average(observed, axis=(0, 1))
        if len(set(avgs)) >= 2:
            found = True
            break
    assert found

    # varying p by stochastic parameter
    aug = iaa.CoarseDropout(p=iap.Binomial(1-iap.Choice([0.0, 0.5])), size_px=50)
    images = np.ones((1, 100, 100, 1), dtype=np.uint8) * 255
    seen = [0, 0, 0]
    for i in sm.xrange(400):
        observed = aug.augment_images(images)
        p = np.mean(observed == 0)
        if 0.4 < p < 0.6:
            seen[0] += 1
        elif p < 0.1:
            seen[1] += 1
        else:
            seen[2] += 1
    assert seen[2] <= 10
    assert 150 < seen[0] < 250
    assert 150 < seen[1] < 250

    # test exception for bad parameters
    got_exception = False
    try:
        aug = iaa.CoarseDropout(p="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.CoarseDropout(p=0.5, size_px=None, size_percent=None)
    except Exception:
        got_exception = True
    assert got_exception


def test_Multiply():
    reseed()

    base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
    images = np.array([base_img])
    images_list = [base_img]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no multiply, shouldnt change anything
    aug = iaa.Multiply(mul=1.0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    # multiply >1.0
    aug = iaa.Multiply(mul=1.2)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
    assert array_equal_lists(observed, expected)

    # multiply <1.0
    aug = iaa.Multiply(mul=0.8)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
    assert array_equal_lists(observed, expected)

    # keypoints shouldnt be changed
    aug = iaa.Multiply(mul=1.2)
    aug_det = iaa.Multiply(mul=1.2).to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # varying multiply factors
    aug = iaa.Multiply(mul=(0, 2.0))
    aug_det = aug.to_deterministic()

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.95)
    assert nb_changed_aug_det == 0

    # test channelwise
    aug = iaa.Multiply(mul=iap.Choice([0, 2]), per_channel=True)
    observed = aug.augment_image(np.ones((1, 1, 100), dtype=np.uint8))
    uq = np.unique(observed)
    assert 0 in uq
    assert 2 in uq
    assert len(uq) == 2

    # test channelwise with probability
    aug = iaa.Multiply(mul=iap.Choice([0, 2]), per_channel=0.5)
    seen = [0, 0]
    for _ in sm.xrange(400):
        observed = aug.augment_image(np.ones((1, 1, 20), dtype=np.uint8))
        uq = np.unique(observed)
        per_channel = (len(uq) == 2)
        if per_channel:
            seen[0] += 1
        else:
            seen[1] += 1
    assert 150 < seen[0] < 250
    assert 150 < seen[1] < 250

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.Multiply(mul="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.Multiply(mul=1, per_channel="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.Multiply(mul=1, per_channel=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert params[0].value == 1
    assert params[1].value == 0


def test_MultiplyElementwise():
    reseed()

    base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
    images = np.array([base_img])
    images_list = [base_img]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no multiply, shouldnt change anything
    aug = iaa.MultiplyElementwise(mul=1.0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    # multiply >1.0
    aug = iaa.MultiplyElementwise(mul=1.2)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
    assert array_equal_lists(observed, expected)

    # multiply <1.0
    aug = iaa.MultiplyElementwise(mul=0.8)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
    assert array_equal_lists(observed, expected)

    # keypoints shouldnt be changed
    aug = iaa.MultiplyElementwise(mul=1.2)
    aug_det = iaa.Multiply(mul=1.2).to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # varying multiply factors
    aug = iaa.MultiplyElementwise(mul=(0, 2.0))
    aug_det = aug.to_deterministic()

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.95)
    assert nb_changed_aug_det == 0

    # values should change between pixels
    aug = iaa.MultiplyElementwise(mul=(0.5, 1.5))

    nb_same = 0
    nb_different = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_flat = observed_aug.flatten()
        last = None
        for j in sm.xrange(observed_aug_flat.size):
            if last is not None:
                v = observed_aug_flat[j]
                if v - 0.0001 <= last <= v + 0.0001:
                    nb_same += 1
                else:
                    nb_different += 1
            last = observed_aug_flat[j]
    assert nb_different > 0.95 * (nb_different + nb_same)

    # test channelwise
    aug = iaa.MultiplyElementwise(mul=iap.Choice([0, 1]), per_channel=True)
    observed = aug.augment_image(np.ones((100, 100, 3), dtype=np.uint8))
    sums = np.sum(observed, axis=2)
    values = np.unique(sums)
    assert all([(value in values) for value in [0, 1, 2, 3]])

    # test channelwise with probability
    aug = iaa.MultiplyElementwise(mul=iap.Choice([0, 1]), per_channel=0.5)
    seen = [0, 0]
    for _ in sm.xrange(400):
        observed = aug.augment_image(np.ones((20, 20, 3), dtype=np.uint8))
        sums = np.sum(observed, axis=2)
        values = np.unique(sums)
        all_values_found = all([(value in values) for value in [0, 1, 2, 3]])
        if all_values_found:
            seen[0] += 1
        else:
            seen[1] += 1
    assert 150 < seen[0] < 250
    assert 150 < seen[1] < 250

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.MultiplyElementwise(mul="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.MultiplyElementwise(mul=1, per_channel="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.MultiplyElementwise(mul=1, per_channel=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert params[0].value == 1
    assert params[1].value == 0

def test_ReplaceElementwise():
    reseed()

    base_img = np.ones((3, 3, 1), dtype=np.uint8) + 99
    images = np.array([base_img])
    images_list = [base_img]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no replace, shouldnt change anything
    aug = iaa.ReplaceElementwise(mask=0, replacement=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    # replace at 100 percent prob., should change everything
    aug = iaa.ReplaceElementwise(mask=1, replacement=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = np.zeros((1, 3, 3, 1), dtype=np.uint8)
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [np.zeros((3, 3, 1), dtype=np.uint8)]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = np.zeros((1, 3, 3, 1), dtype=np.uint8)
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [np.zeros((3, 3, 1), dtype=np.uint8)]
    assert array_equal_lists(observed, expected)

    # replace half
    aug = iaa.ReplaceElementwise(mask=iap.Binomial(p=0.5), replacement=0)
    img = np.ones((100, 100, 1), dtype=np.uint8)

    nb_iterations = 100
    nb_diff_all = 0
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(img)
        nb_diff = np.sum(img != observed)
        nb_diff_all += nb_diff
    p = nb_diff_all / (nb_iterations * 100 * 100)
    assert 0.45 <= p <= 0.55

    # mask is list
    aug = iaa.ReplaceElementwise(mask=[0.2, 0.7], replacement=1)
    img = np.zeros((20, 20, 1), dtype=np.uint8)

    seen = [0, 0, 0]
    for i in sm.xrange(400):
        observed = aug.augment_image(img)
        p = np.mean(observed)
        if 0.1 < p < 0.3:
            seen[0] += 1
        elif 0.6 < p < 0.8:
            seen[1] += 1
        else:
            seen[2] += 1
    assert seen[2] <= 10
    assert 150 < seen[0] < 250
    assert 150 < seen[1] < 250

    """
    observed = aug.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
    assert array_equal_lists(observed, expected)
    """

    # keypoints shouldnt be changed
    aug = iaa.ReplaceElementwise(mask=iap.Binomial(p=0.5), replacement=0)
    aug_det = iaa.ReplaceElementwise(mask=iap.Binomial(p=0.5), replacement=0).to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # different replacements
    aug = iaa.ReplaceElementwise(mask=1, replacement=iap.Choice([100, 200]))
    img = np.zeros((1000, 1000, 1), dtype=np.uint8)
    img100 = img + 100
    img200 = img + 200
    observed = aug.augment_image(img)
    nb_diff_100 = np.sum(img100 != observed)
    nb_diff_200 = np.sum(img200 != observed)
    p100 = nb_diff_100 / (1000 * 1000)
    p200 = nb_diff_200 / (1000 * 1000)
    assert 0.45 <= p100 <= 0.55
    assert 0.45 <= p200 <= 0.55
    # test channelwise
    aug = iaa.MultiplyElementwise(mul=iap.Choice([0, 1]), per_channel=True)
    observed = aug.augment_image(np.ones((100, 100, 3), dtype=np.uint8))
    sums = np.sum(observed, axis=2)
    values = np.unique(sums)
    assert all([(value in values) for value in [0, 1, 2, 3]])

    # test channelwise with probability
    aug = iaa.ReplaceElementwise(mask=iap.Choice([0, 1]), replacement=1, per_channel=0.5)
    seen = [0, 0]
    for _ in sm.xrange(400):
        observed = aug.augment_image(np.zeros((20, 20, 3), dtype=np.uint8))
        sums = np.sum(observed, axis=2)
        values = np.unique(sums)
        all_values_found = all([(value in values) for value in [0, 1, 2, 3]])
        if all_values_found:
            seen[0] += 1
        else:
            seen[1] += 1
    assert 150 < seen[0] < 250
    assert 150 < seen[1] < 250

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.ReplaceElementwise(mask="test", replacement=1)
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.ReplaceElementwise(mask=1, replacement=1, per_channel="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.ReplaceElementwise(mask=0.5, replacement=2, per_channel=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Binomial)
    assert isinstance(params[0].p, iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert isinstance(params[2], iap.Deterministic)
    assert 0.5 - 1e-6 < params[0].p.value < 0.5 + 1e-6
    assert params[1].value == 2
    assert params[2].value == 0


def test_SaltAndPepper():
    reseed()

    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    aug = iaa.SaltAndPepper(p=0.5)
    observed = aug.augment_image(base_img)
    p = np.mean(observed != 128)
    assert 0.4 < p < 0.6

    aug = iaa.SaltAndPepper(p=1.0)
    observed = aug.augment_image(base_img)
    nb_pepper = np.sum(observed < 40)
    nb_salt = np.sum(observed > 255 - 40)
    assert nb_pepper > 200
    assert nb_salt > 200

    # not more tests necessary here as SaltAndPepper is just a tiny wrapper around
    # ReplaceElementwise


def test_CoarseSaltAndPepper():
    reseed()

    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    aug = iaa.CoarseSaltAndPepper(p=0.5, size_px=100)
    observed = aug.augment_image(base_img)
    p = np.mean(observed != 128)
    assert 0.4 < p < 0.6

    aug1 = iaa.CoarseSaltAndPepper(p=0.5, size_px=100)
    aug2 = iaa.CoarseSaltAndPepper(p=0.5, size_px=10)
    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    ps1 = []
    ps2 = []
    for _ in sm.xrange(100):
        observed1 = aug1.augment_image(base_img)
        observed2 = aug2.augment_image(base_img)
        p1 = np.mean(observed1 != 128)
        p2 = np.mean(observed2 != 128)
        ps1.append(p1)
        ps2.append(p2)
    assert 0.4 < np.mean(ps2) < 0.6
    assert np.std(ps1)*1.5 < np.std(ps2)

    aug = iaa.CoarseSaltAndPepper(p=[0.2, 0.5], size_px=100)
    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    seen = [0, 0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        diff_020 = abs(0.2 - p)
        diff_050 = abs(0.5 - p)
        if diff_020 < 0.025:
            seen[0] += 1
        elif diff_050 < 0.025:
            seen[1] += 1
        else:
            seen[2] += 1
    assert seen[2] < 10
    assert 75 < seen[0] < 125
    assert 75 < seen[1] < 125

    aug = iaa.CoarseSaltAndPepper(p=(0.0, 1.0), size_px=50)
    base_img = np.zeros((50, 50, 1), dtype=np.uint8) + 128
    ps = []
    for _ in sm.xrange(200):
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        ps.append(p)

    nb_bins = 5
    hist, _ = np.histogram(ps, bins=nb_bins, range=(0.0, 1.0), density=False)
    tolerance = 0.05
    for nb_seen in hist:
        density = nb_seen / len(ps)
        assert density - tolerance < density < density + tolerance

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.CoarseSaltAndPepper(p="test", size_px=100)
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.CoarseSaltAndPepper(p=0.5, size_px=None, size_percent=None)
    except Exception:
        got_exception = True
    assert got_exception


def test_Salt():
    reseed()

    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    aug = iaa.Salt(p=0.5)
    observed = aug.augment_image(base_img)
    p = np.mean(observed != 128)
    assert 0.4 < p < 0.6
    assert np.all(observed >= 127)  # Salt() occasionally replaces with 127,
                                    # which probably should be the center-point here anyways

    aug = iaa.Salt(p=1.0)
    observed = aug.augment_image(base_img)
    nb_pepper = np.sum(observed < 40)
    nb_salt = np.sum(observed > 255 - 40)
    assert nb_pepper == 0
    assert nb_salt > 200

    # not more tests necessary here as Salt is just a tiny wrapper around
    # ReplaceElementwise


def test_CoarseSalt():
    reseed()

    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    aug = iaa.CoarseSalt(p=0.5, size_px=100)
    observed = aug.augment_image(base_img)
    p = np.mean(observed != 128)
    assert 0.4 < p < 0.6

    aug1 = iaa.CoarseSalt(p=0.5, size_px=100)
    aug2 = iaa.CoarseSalt(p=0.5, size_px=10)
    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    ps1 = []
    ps2 = []
    for _ in sm.xrange(100):
        observed1 = aug1.augment_image(base_img)
        observed2 = aug2.augment_image(base_img)
        p1 = np.mean(observed1 != 128)
        p2 = np.mean(observed2 != 128)
        ps1.append(p1)
        ps2.append(p2)
    assert 0.4 < np.mean(ps2) < 0.6
    assert np.std(ps1)*1.5 < np.std(ps2)

    aug = iaa.CoarseSalt(p=[0.2, 0.5], size_px=100)
    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    seen = [0, 0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        diff_020 = abs(0.2 - p)
        diff_050 = abs(0.5 - p)
        if diff_020 < 0.025:
            seen[0] += 1
        elif diff_050 < 0.025:
            seen[1] += 1
        else:
            seen[2] += 1
    assert seen[2] < 10
    assert 75 < seen[0] < 125
    assert 75 < seen[1] < 125

    aug = iaa.CoarseSalt(p=(0.0, 1.0), size_px=50)
    base_img = np.zeros((50, 50, 1), dtype=np.uint8) + 128
    ps = []
    for _ in sm.xrange(200):
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        ps.append(p)

    nb_bins = 5
    hist, _ = np.histogram(ps, bins=nb_bins, range=(0.0, 1.0), density=False)
    tolerance = 0.05
    for nb_seen in hist:
        density = nb_seen / len(ps)
        assert density - tolerance < density < density + tolerance

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.CoarseSalt(p="test", size_px=100)
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.CoarseSalt(p=0.5, size_px=None, size_percent=None)
    except Exception:
        got_exception = True
    assert got_exception


def test_Pepper():
    reseed()

    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    aug = iaa.Pepper(p=0.5)
    observed = aug.augment_image(base_img)
    p = np.mean(observed != 128)
    assert 0.4 < p < 0.6
    assert np.all(observed <= 128)

    aug = iaa.Pepper(p=1.0)
    observed = aug.augment_image(base_img)
    nb_pepper = np.sum(observed < 40)
    nb_salt = np.sum(observed > 255 - 40)
    assert nb_pepper > 200
    assert nb_salt == 0

    # not more tests necessary here as Salt is just a tiny wrapper around
    # ReplaceElementwise


def test_CoarsePepper():
    reseed()

    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    aug = iaa.CoarsePepper(p=0.5, size_px=100)
    observed = aug.augment_image(base_img)
    p = np.mean(observed != 128)
    assert 0.4 < p < 0.6

    aug1 = iaa.CoarsePepper(p=0.5, size_px=100)
    aug2 = iaa.CoarsePepper(p=0.5, size_px=10)
    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    ps1 = []
    ps2 = []
    for _ in sm.xrange(100):
        observed1 = aug1.augment_image(base_img)
        observed2 = aug2.augment_image(base_img)
        p1 = np.mean(observed1 != 128)
        p2 = np.mean(observed2 != 128)
        ps1.append(p1)
        ps2.append(p2)
    assert 0.4 < np.mean(ps2) < 0.6
    assert np.std(ps1)*1.5 < np.std(ps2)

    aug = iaa.CoarsePepper(p=[0.2, 0.5], size_px=100)
    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    seen = [0, 0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        diff_020 = abs(0.2 - p)
        diff_050 = abs(0.5 - p)
        if diff_020 < 0.025:
            seen[0] += 1
        elif diff_050 < 0.025:
            seen[1] += 1
        else:
            seen[2] += 1
    assert seen[2] < 10
    assert 75 < seen[0] < 125
    assert 75 < seen[1] < 125

    aug = iaa.CoarsePepper(p=(0.0, 1.0), size_px=50)
    base_img = np.zeros((50, 50, 1), dtype=np.uint8) + 128
    ps = []
    for _ in sm.xrange(200):
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        ps.append(p)

    nb_bins = 5
    hist, _ = np.histogram(ps, bins=nb_bins, range=(0.0, 1.0), density=False)
    tolerance = 0.05
    for nb_seen in hist:
        density = nb_seen / len(ps)
        assert density - tolerance < density < density + tolerance

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.CoarsePepper(p="test", size_px=100)
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.CoarsePepper(p=0.5, size_px=None, size_percent=None)
    except Exception:
        got_exception = True
    assert got_exception


def test_Add():
    reseed()

    base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
    images = np.array([base_img])
    images_list = [base_img]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no add, shouldnt change anything
    aug = iaa.Add(value=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    # add > 0
    aug = iaa.Add(value=1)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images + 1
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [images_list[0] + 1]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images + 1
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [images_list[0] + 1]
    assert array_equal_lists(observed, expected)

    # add < 0
    aug = iaa.Add(value=-1)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images - 1
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [images_list[0] - 1]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images - 1
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [images_list[0] - 1]
    assert array_equal_lists(observed, expected)

    # test other parameters
    aug = iaa.Add(value=iap.DiscreteUniform(1, 10))
    observed = aug.augment_images(images)
    assert 100 + 1 <= np.average(observed) <= 100 + 10

    aug = iaa.Add(value=iap.Uniform(1, 10))
    observed = aug.augment_images(images)
    assert 100 + 1 <= np.average(observed) <= 100 + 10

    aug = iaa.Add(value=iap.Clip(iap.Normal(1, 1), -3, 3))
    observed = aug.augment_images(images)
    assert 100 - 3 <= np.average(observed) <= 100 + 3

    aug = iaa.Add(value=iap.Discretize(iap.Clip(iap.Normal(1, 1), -3, 3)))
    observed = aug.augment_images(images)
    assert 100 - 3 <= np.average(observed) <= 100 + 3

    # keypoints shouldnt be changed
    aug = iaa.Add(value=1)
    aug_det = iaa.Add(value=1).to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # varying values
    aug = iaa.Add(value=(0, 10))
    aug_det = aug.to_deterministic()

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.7)
    assert nb_changed_aug_det == 0

    # test channelwise
    aug = iaa.Add(value=iap.Choice([0, 1]), per_channel=True)
    observed = aug.augment_image(np.zeros((1, 1, 100), dtype=np.uint8))
    uq = np.unique(observed)
    assert 0 in uq
    assert 1 in uq
    assert len(uq) == 2

    # test channelwise with probability
    aug = iaa.Add(value=iap.Choice([0, 1]), per_channel=0.5)
    seen = [0, 0]
    for _ in sm.xrange(400):
        observed = aug.augment_image(np.zeros((1, 1, 20), dtype=np.uint8))
        uq = np.unique(observed)
        per_channel = (len(uq) == 2)
        if per_channel:
            seen[0] += 1
        else:
            seen[1] += 1
    assert 150 < seen[0] < 250
    assert 150 < seen[1] < 250

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.Add(value="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.Add(value=1, per_channel="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.Add(value=1, per_channel=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert params[0].value == 1
    assert params[1].value == 0

def test_AddElementwise():
    reseed()

    base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
    images = np.array([base_img])
    images_list = [base_img]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no add, shouldnt change anything
    aug = iaa.AddElementwise(value=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    # add > 0
    aug = iaa.AddElementwise(value=1)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images + 1
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [images_list[0] + 1]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images + 1
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [images_list[0] + 1]
    assert array_equal_lists(observed, expected)

    # add < 0
    aug = iaa.AddElementwise(value=-1)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images - 1
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = [images_list[0] - 1]
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images - 1
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = [images_list[0] - 1]
    assert array_equal_lists(observed, expected)

    # test other parameters
    aug = iaa.AddElementwise(value=iap.DiscreteUniform(1, 10))
    observed = aug.augment_images(images)
    assert np.min(observed) >= 100 + 1
    assert np.max(observed) <= 100 + 10

    aug = iaa.AddElementwise(value=iap.Uniform(1, 10))
    observed = aug.augment_images(images)
    assert np.min(observed) >= 100 + 1
    assert np.max(observed) <= 100 + 10

    aug = iaa.AddElementwise(value=iap.Clip(iap.Normal(1, 1), -3, 3))
    observed = aug.augment_images(images)
    assert np.min(observed) >= 100 - 3
    assert np.max(observed) <= 100 + 3

    aug = iaa.AddElementwise(value=iap.Discretize(iap.Clip(iap.Normal(1, 1), -3, 3)))
    observed = aug.augment_images(images)
    assert np.min(observed) >= 100 - 3
    assert np.max(observed) <= 100 + 3

    # keypoints shouldnt be changed
    aug = iaa.AddElementwise(value=1)
    aug_det = iaa.AddElementwise(value=1).to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # varying values
    aug = iaa.AddElementwise(value=(0, 10))
    aug_det = aug.to_deterministic()

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.7)
    assert nb_changed_aug_det == 0

    # values should change between pixels
    aug = iaa.AddElementwise(value=(-50, 50))

    nb_same = 0
    nb_different = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_flat = observed_aug.flatten()
        last = None
        for j in sm.xrange(observed_aug_flat.size):
            if last is not None:
                v = observed_aug_flat[j]
                if v - 0.0001 <= last <= v + 0.0001:
                    nb_same += 1
                else:
                    nb_different += 1
            last = observed_aug_flat[j]
    assert nb_different > 0.9 * (nb_different + nb_same)

    # test channelwise
    aug = iaa.AddElementwise(value=iap.Choice([0, 1]), per_channel=True)
    observed = aug.augment_image(np.zeros((100, 100, 3), dtype=np.uint8))
    sums = np.sum(observed, axis=2)
    values = np.unique(sums)
    assert all([(value in values) for value in [0, 1, 2, 3]])

    # test channelwise with probability
    aug = iaa.AddElementwise(value=iap.Choice([0, 1]), per_channel=0.5)
    seen = [0, 0]
    for _ in sm.xrange(400):
        observed = aug.augment_image(np.zeros((20, 20, 3), dtype=np.uint8))
        sums = np.sum(observed, axis=2)
        values = np.unique(sums)
        all_values_found = all([(value in values) for value in [0, 1, 2, 3]])
        if all_values_found:
            seen[0] += 1
        else:
            seen[1] += 1
    assert 150 < seen[0] < 250
    assert 150 < seen[1] < 250

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.AddElementwise(value="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.AddElementwise(value=1, per_channel="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.AddElementwise(value=1, per_channel=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert params[0].value == 1
    assert params[1].value == 0


def test_Invert():
    reseed()

    zeros = np.zeros((4, 4, 3), dtype=np.uint8)
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=zeros.shape)]

    observed = iaa.Invert(p=1.0).augment_image(zeros + 255)
    expected = zeros
    assert np.array_equal(observed, expected)

    observed = iaa.Invert(p=0.0).augment_image(zeros + 255)
    expected = zeros + 255
    assert np.array_equal(observed, expected)

    observed = iaa.Invert(p=1.0, max_value=200).augment_image(zeros + 200)
    expected = zeros
    assert np.array_equal(observed, expected)

    observed = iaa.Invert(p=1.0, max_value=200, min_value=100).augment_image(zeros + 200)
    expected = zeros + 100
    assert np.array_equal(observed, expected)

    observed = iaa.Invert(p=1.0, max_value=200, min_value=100).augment_image(zeros + 100)
    expected = zeros + 200
    assert np.array_equal(observed, expected)

    nb_iterations = 1000
    nb_inverted = 0
    aug = iaa.Invert(p=0.8)
    img = np.zeros((1, 1, 1), dtype=np.uint8) + 256
    expected = np.zeros((1, 1, 1), dtype=np.uint8)
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(img)
        if np.array_equal(observed, expected):
            nb_inverted += 1
    pinv = nb_inverted / nb_iterations
    assert 0.75 <= pinv <= 0.85

    nb_iterations = 1000
    nb_inverted = 0
    aug = iaa.Invert(p=iap.Binomial(0.8))
    img = np.zeros((1, 1, 1), dtype=np.uint8) + 256
    expected = np.zeros((1, 1, 1), dtype=np.uint8)
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(img)
        if np.array_equal(observed, expected):
            nb_inverted += 1
    pinv = nb_inverted / nb_iterations
    assert 0.75 <= pinv <= 0.85

    nb_iterations = 1000
    nb_inverted = 0
    aug = iaa.Invert(p=0.5, per_channel=True)
    img = np.zeros((1, 1, 100), dtype=np.uint8) + 256
    observed = aug.augment_image(img)
    assert len(np.unique(observed)) == 2

    nb_iterations = 1000
    nb_inverted = 0
    aug = iaa.Invert(p=iap.Binomial(0.8), per_channel=0.7)
    img = np.zeros((1, 1, 20), dtype=np.uint8) + 256
    seen = [0, 0]
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(img)
        uq = np.unique(observed)
        if len(uq) == 1:
            seen[0] += 1
        elif len(uq) == 2:
            seen[1] += 1
        else:
            assert False
    assert 300 - 75 < seen[0] < 300 + 75
    assert 700 - 75 < seen[1] < 700 + 75

    # keypoints shouldnt be changed
    aug = iaa.Invert(p=1.0)
    aug_det = iaa.Invert(p=1.0).to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.Invert(p="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.Invert(p=0.5, per_channel="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.Invert(p=0.5, per_channel=False, min_value=10, max_value=20)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Binomial)
    assert isinstance(params[0].p, iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert 0.5 - 1e-4 < params[0].p.value < 0.5 + 1e-4
    assert params[1].value == 0
    assert params[2] == 10
    assert params[3] == 20


def test_ContrastNormalization():
    reseed()

    zeros = np.zeros((4, 4, 3), dtype=np.uint8)
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=zeros.shape)]

    # contrast stays the same
    observed = iaa.ContrastNormalization(alpha=1.0).augment_image(zeros + 50)
    expected = zeros + 50
    assert np.array_equal(observed, expected)

    # image with mean intensity (ie 128), contrast cannot be changed
    observed = iaa.ContrastNormalization(alpha=2.0).augment_image(zeros + 128)
    expected = zeros + 128
    assert np.array_equal(observed, expected)

    # increase contrast
    observed = iaa.ContrastNormalization(alpha=2.0).augment_image(zeros + 128 + 10)
    expected = zeros + 128 + 20
    assert np.array_equal(observed, expected)

    observed = iaa.ContrastNormalization(alpha=2.0).augment_image(zeros + 128 - 10)
    expected = zeros + 128 - 20
    assert np.array_equal(observed, expected)

    # decrease contrast
    observed = iaa.ContrastNormalization(alpha=0.5).augment_image(zeros + 128 + 10)
    expected = zeros + 128 + 5
    assert np.array_equal(observed, expected)

    observed = iaa.ContrastNormalization(alpha=0.5).augment_image(zeros + 128 - 10)
    expected = zeros + 128 - 5
    assert np.array_equal(observed, expected)

    # increase contrast by stochastic parameter
    observed = iaa.ContrastNormalization(alpha=iap.Choice([2.0, 3.0])).augment_image(zeros + 128 + 10)
    expected1 = zeros + 128 + 20
    expected2 = zeros + 128 + 30
    assert np.array_equal(observed, expected1) or np.array_equal(observed, expected2)

    # change contrast by tuple
    nb_iterations = 1000
    nb_changed = 0
    last = None
    for i in sm.xrange(nb_iterations):
        observed = iaa.ContrastNormalization(alpha=(0.5, 2.0)).augment_image(zeros + 128 + 40)
        if last is None:
            last = observed
        else:
            if not np.array_equal(observed, last):
                nb_changed += 1
    p_changed = nb_changed / (nb_iterations-1)
    assert p_changed > 0.5

    # per_channel=True
    aug = iaa.ContrastNormalization(alpha=(1.0, 6.0), per_channel=True)
    img = np.zeros((1, 1, 100), dtype=np.uint8) + 128 + 10
    observed = aug.augment_image(img)
    uq = np.unique(observed)
    assert len(uq) > 5

    # per_channel with probability
    aug = iaa.ContrastNormalization(alpha=(1.0, 4.0), per_channel=0.7)
    img = np.zeros((1, 1, 100), dtype=np.uint8) + 128 + 10
    seen = [0, 0]
    for _ in sm.xrange(1000):
        observed = aug.augment_image(img)
        uq = np.unique(observed)
        if len(uq) == 1:
            seen[0] += 1
        elif len(uq) >= 2:
            seen[1] += 1
    assert 300 - 75 < seen[0] < 300 + 75
    assert 700 - 75 < seen[1] < 700 + 75

    # keypoints shouldnt be changed
    aug = iaa.ContrastNormalization(alpha=2.0)
    aug_det = iaa.ContrastNormalization(alpha=2.0).to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        aug = iaa.ContrastNormalization(alpha="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        aug = iaa.ContrastNormalization(alpha=1.5, per_channel="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.ContrastNormalization(alpha=1, per_channel=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert params[0].value == 1
    assert params[1].value == 0


def test_Affine():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 255, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]
    outer_pixels = ([], [])
    for i in sm.xrange(base_img.shape[0]):
        for j in sm.xrange(base_img.shape[1]):
            if i != j:
                outer_pixels[0].append(i)
                outer_pixels[1].append(j)

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no translation/scale/rotate/shear, shouldnt change nothing
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # ---------------------
    # scale
    # ---------------------
    # zoom in
    aug = iaa.Affine(scale=1.75, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y > 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y > 2

    # zoom in only on x axis
    aug = iaa.Affine(scale={"x": 1.75, "y": 1.0}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y == 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y == 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y == 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y == 2

    # zoom in only on y axis
    aug = iaa.Affine(scale={"x": 1.0, "y": 1.75}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x == 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x == 2
    assert observed[0].keypoints[2].y > 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x == 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x == 2
    assert observed[0].keypoints[2].y > 2

    # zoom out
    # this one uses a 4x4 area of all 255, which is zoomed out to a 4x4 area
    # in which the center 2x2 area is 255
    # zoom in should probably be adapted to this style
    # no separate tests here for x/y axis, should work fine if zoom in works with that
    aug = iaa.Affine(scale=0.49, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.ones((4, 4, 1), dtype=np.uint8) * 255
    images = np.array([image])
    images_list = [image]
    outer_pixels = ([], [])
    for y in sm.xrange(4):
        xs = sm.xrange(4) if y in [0, 3] else [0, 3]
        for x in xs:
            outer_pixels[0].append(y)
            outer_pixels[1].append(x)
    inner_pixels = ([1, 1, 2, 2], [1, 2, 1, 2])
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=3, y=0),
                                      ia.Keypoint(x=0, y=3), ia.Keypoint(x=3, y=3)],
                                     shape=image.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=0.765, y=0.765), ia.Keypoint(x=2.235, y=0.765),
                                          ia.Keypoint(x=0.765, y=2.235), ia.Keypoint(x=2.235, y=2.235)],
                                         shape=image.shape)]

    observed = aug.augment_images(images)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug_det.augment_images(images)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug.augment_images(images_list)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug_det.augment_images(images_list)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # varying scales
    aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, translate_px=0,
                     rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.array([[0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 1, 2, 1, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0]], dtype=np.uint8) * 100
    image = image[:, :, np.newaxis]
    images_list = [image]
    images = np.array([image])

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.8)
    assert nb_changed_aug_det == 0

    aug = iaa.Affine(scale=iap.Uniform(0.7, 0.9))
    assert isinstance(aug.scale, iap.Uniform)
    assert isinstance(aug.scale.a, iap.Deterministic)
    assert isinstance(aug.scale.b, iap.Deterministic)
    assert 0.7 - 1e-8 < aug.scale.a.value < 0.7 + 1e-8
    assert 0.9 - 1e-8 < aug.scale.b.value < 0.9 + 1e-8

    # ---------------------
    # translate
    # ---------------------
    # move one pixel to the right
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[1, 2] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move one pixel to the right
    # with backend = skimage
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="skimage")
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with backend = skimage
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="skimage")
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with backend = skimage, order=ALL
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="skimage", order=ia.ALL)
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with backend = skimage, order=list
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="skimage", order=[0, 1, 3])
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with backend = cv2, order=list
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="cv2", order=[0, 1, 3])
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with backend = cv2, order=StochasticParameter
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="cv2", order=iap.Choice([0, 1, 3]))
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the bottom
    aug = iaa.Affine(scale=1.0, translate_px={"x": 0, "y": 1}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move 33% (one pixel) to the right
    aug = iaa.Affine(scale=1.0, translate_percent={"x": 0.3333, "y": 0}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[1, 2] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move 33% (one pixel) to the bottom
    aug = iaa.Affine(scale=1.0, translate_percent={"x": 0, "y": 0.3333}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # 0-1px to left/right and 0-1px to top/bottom
    aug = iaa.Affine(scale=1.0, translate_px={"x": (-1, 1), "y": (-1, 1)}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    centers_aug = np.copy(image).astype(np.int32) * 0
    centers_aug_det = np.copy(image).astype(np.int32) * 0
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        assert len(observed_aug[0].nonzero()[0]) == 1
        assert len(observed_aug_det[0].nonzero()[0]) == 1
        centers_aug += (observed_aug[0] > 0)
        centers_aug_det += (observed_aug_det[0] > 0)

    assert nb_changed_aug >= int(nb_iterations * 0.7)
    assert nb_changed_aug_det == 0
    assert (centers_aug > int(nb_iterations * (1/9 * 0.6))).all()
    assert (centers_aug < int(nb_iterations * (1/9 * 1.4))).all()

    aug = iaa.Affine(translate_percent=iap.Uniform(0.7, 0.9))
    assert isinstance(aug.translate, iap.Uniform)
    assert isinstance(aug.translate.a, iap.Deterministic)
    assert isinstance(aug.translate.b, iap.Deterministic)
    assert 0.7 - 1e-8 < aug.translate.a.value < 0.7 + 1e-8
    assert 0.9 - 1e-8 < aug.translate.b.value < 0.9 + 1e-8

    aug = iaa.Affine(translate_px=iap.DiscreteUniform(1, 10))
    assert isinstance(aug.translate, iap.DiscreteUniform)
    assert isinstance(aug.translate.a, iap.Deterministic)
    assert isinstance(aug.translate.b, iap.Deterministic)
    assert aug.translate.a.value == 1
    assert aug.translate.b.value == 10

    # ---------------------
    # translate heatmaps
    # ---------------------
    heatmaps = ia.HeatmapsOnImage(
        np.float32([
            [0.0, 0.5, 0.75],
            [0.0, 0.5, 0.75],
            [0.75, 0.75, 0.75],
        ]),
        shape=(3, 3, 3)
    )
    arr_expected_1px_right = np.float32([
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.75, 0.75],
    ])
    aug = iaa.Affine(translate_px={"x": 1})
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    # should still use mode=constant cval=0 even when other settings chosen
    aug = iaa.Affine(translate_px={"x": 1}, cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    aug = iaa.Affine(translate_px={"x": 1}, mode="edge", cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    # ---------------------
    # rotate
    # ---------------------
    # rotate by 45 degrees
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=90, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, :] = 255
    image_aug[0, 1] = 255
    image_aug[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    observed[observed >= 100] = 255
    observed[observed < 100] = 0
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    observed[observed >= 100] = 255
    observed[observed < 100] = 0
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    observed[0][observed[0] >= 100] = 255
    observed[0][observed[0] < 100] = 0
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    observed[0][observed[0] >= 100] = 255
    observed[0][observed[0] < 100] = 0
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # rotate by StochasticParameter
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=iap.Uniform(10, 20), shear=0)
    assert isinstance(aug.rotate, iap.Uniform)
    assert isinstance(aug.rotate.a, iap.Deterministic)
    assert aug.rotate.a.value == 10
    assert isinstance(aug.rotate.b, iap.Deterministic)
    assert aug.rotate.b.value == 20

    # random rotation 0-364 degrees
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=(0, 364), shear=0)
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    pixels_sums_aug = np.copy(image).astype(np.int32) * 0
    pixels_sums_aug_det = np.copy(image).astype(np.int32) * 0
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        #assert len(observed_aug[0].nonzero()[0]) == 1
        #assert len(observed_aug_det[0].nonzero()[0]) == 1
        pixels_sums_aug += (observed_aug[0] > 100)
        pixels_sums_aug_det += (observed_aug_det[0] > 100)

    assert nb_changed_aug >= int(nb_iterations * 0.9)
    assert nb_changed_aug_det == 0
    # center pixel, should always be white when rotating line around center
    assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
    assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)

    # outer pixels, should sometimes be white
    # the values here had to be set quite tolerant, the middle pixels at top/left/bottom/right get more activation than expected
    outer_pixels = ([0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2])
    assert (pixels_sums_aug[outer_pixels] > int(nb_iterations * (2/8 * 0.4))).all()
    assert (pixels_sums_aug[outer_pixels] < int(nb_iterations * (2/8 * 2.0))).all()

    # ---------------------
    # shear
    # ---------------------
    # TODO

    # shear by StochasticParameter
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=iap.Uniform(10, 20))
    assert isinstance(aug.shear, iap.Uniform)
    assert isinstance(aug.shear.a, iap.Deterministic)
    assert aug.shear.a.value == 10
    assert isinstance(aug.shear.b, iap.Deterministic)
    assert aug.shear.b.value == 20

    # ---------------------
    # cval
    # ---------------------
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=128)
    aug_det = aug.to_deterministic()

    image = np.ones((3, 3, 1), dtype=np.uint8) * 255
    image_aug = np.copy(image)
    images = np.array([image])
    images_list = [image]

    observed = aug.augment_images(images)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug_det.augment_images(images)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug.augment_images(images_list)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug_det.augment_images(images_list)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    # random cvals
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=(0, 255))
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    averages = []
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        averages.append(int(np.average(observed_aug)))

    assert nb_changed_aug >= int(nb_iterations * 0.9)
    assert nb_changed_aug_det == 0
    # center pixel, should always be white when rotating line around center
    assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
    assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)
    assert len(set(averages)) > 200

    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=ia.ALL)
    assert isinstance(aug.cval, iap.Uniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 255

    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=iap.DiscreteUniform(1, 5))
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 1
    assert aug.cval.b.value == 5

    # ------------
    # mode
    # ------------
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=ia.ALL)
    assert isinstance(aug.mode, iap.Choice)
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode="edge")
    assert isinstance(aug.mode, iap.Deterministic)
    assert aug.mode.value == "edge"
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=["constant", "edge"])
    assert isinstance(aug.mode, iap.Choice)
    assert len(aug.mode.a) == 2 and "constant" in aug.mode.a and "edge" in aug.mode.a
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=iap.Choice(["constant", "edge"]))
    assert isinstance(aug.mode, iap.Choice)
    assert len(aug.mode.a) == 2 and "constant" in aug.mode.a and "edge" in aug.mode.a

    # ------------
    # exceptions for bad inputs
    # ------------
    # scale
    got_exception = False
    try:
        aug = iaa.Affine(scale=False)
    except Exception:
        got_exception = True
    assert got_exception

    # translate_px
    got_exception = False
    try:
        aug = iaa.Affine(translate_px=False)
    except Exception:
        got_exception = True
    assert got_exception

    # translate_percent
    got_exception = False
    try:
        aug = iaa.Affine(translate_percent=False)
    except Exception:
        got_exception = True
    assert got_exception

    # rotate
    got_exception = False
    try:
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=False, shear=0, cval=0)
    except Exception:
        got_exception = True
    assert got_exception

    # shear
    got_exception = False
    try:
        aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=False, cval=0)
    except Exception:
        got_exception = True
    assert got_exception

    # cval
    got_exception = False
    try:
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=None)
    except Exception:
        got_exception = True
    assert got_exception

    # mode
    got_exception = False
    try:
        aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=False)
    except Exception:
        got_exception = True
    assert got_exception

    # non-existent order in case of backend=cv2
    got_exception = False
    try:
        aug = iaa.Affine(backend="cv2", order=-1)
    except Exception:
        got_exception = True
    assert got_exception

    # bad order datatype in case of backend=cv2
    got_exception = False
    try:
        aug = iaa.Affine(backend="cv2", order="test")
    except Exception:
        got_exception = True
    assert got_exception

    # ----------
    # get_parameters
    # ----------
    aug = iaa.Affine(scale=1, translate_px=2, rotate=3, shear=4, order=1, cval=0, mode="constant", backend="cv2")
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)  # scale
    assert isinstance(params[1], iap.Deterministic)  # translate
    assert isinstance(params[2], iap.Deterministic)  # rotate
    assert isinstance(params[3], iap.Deterministic)  # shear
    assert params[0].value == 1  # scale
    assert params[1].value == 2  # translate
    assert params[2].value == 3  # rotate
    assert params[3].value == 4  # shear
    assert params[4].value == 1  # order
    assert params[5].value == 0  # cval
    assert params[6].value == "constant"  # mode
    assert params[7] == "cv2"  # backend


def test_AffineCv2():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 255, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]
    outer_pixels = ([], [])
    for i in sm.xrange(base_img.shape[0]):
        for j in sm.xrange(base_img.shape[1]):
            if i != j:
                outer_pixels[0].append(i)
                outer_pixels[1].append(j)

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no translation/scale/rotate/shear, shouldnt change nothing
    aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # ---------------------
    # scale
    # ---------------------
    # zoom in
    aug = iaa.AffineCv2(scale=1.75, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y > 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y > 2

    # zoom in only on x axis
    aug = iaa.AffineCv2(scale={"x": 1.75, "y": 1.0}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y == 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y == 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y == 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y == 2

    # zoom in only on y axis
    aug = iaa.AffineCv2(scale={"x": 1.0, "y": 1.75}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x == 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x == 2
    assert observed[0].keypoints[2].y > 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x == 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x == 2
    assert observed[0].keypoints[2].y > 2

    # zoom out
    # this one uses a 4x4 area of all 255, which is zoomed out to a 4x4 area
    # in which the center 2x2 area is 255
    # zoom in should probably be adapted to this style
    # no separate tests here for x/y axis, should work fine if zoom in works with that
    aug = iaa.AffineCv2(scale=0.49, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.ones((4, 4, 1), dtype=np.uint8) * 255
    images = np.array([image])
    images_list = [image]
    outer_pixels = ([], [])
    for y in sm.xrange(4):
        xs = sm.xrange(4) if y in [0, 3] else [0, 3]
        for x in xs:
            outer_pixels[0].append(y)
            outer_pixels[1].append(x)
    inner_pixels = ([1, 1, 2, 2], [1, 2, 1, 2])
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=3, y=0),
                                      ia.Keypoint(x=0, y=3), ia.Keypoint(x=3, y=3)],
                                     shape=image.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=0.765, y=0.765), ia.Keypoint(x=2.235, y=0.765),
                                          ia.Keypoint(x=0.765, y=2.235), ia.Keypoint(x=2.235, y=2.235)],
                                         shape=image.shape)]

    observed = aug.augment_images(images)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug_det.augment_images(images)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug.augment_images(images_list)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug_det.augment_images(images_list)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # varying scales
    aug = iaa.AffineCv2(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, translate_px=0,
                        rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.array([[0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 1, 2, 1, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0]], dtype=np.uint8) * 100
    image = image[:, :, np.newaxis]
    images_list = [image]
    images = np.array([image])

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.8)
    assert nb_changed_aug_det == 0

    aug = iaa.AffineCv2(scale=iap.Uniform(0.7, 0.9))
    assert isinstance(aug.scale, iap.Uniform)
    assert isinstance(aug.scale.a, iap.Deterministic)
    assert isinstance(aug.scale.b, iap.Deterministic)
    assert 0.7 - 1e-8 < aug.scale.a.value < 0.7 + 1e-8
    assert 0.9 - 1e-8 < aug.scale.b.value < 0.9 + 1e-8

    # ---------------------
    # translate
    # ---------------------
    # move one pixel to the right
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[1, 2] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move one pixel to the right
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0)
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0)
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with order=ALL
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, order=ia.ALL)
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with order=list
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, order=[0, 1, 2])
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with order=StochasticParameter
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, order=iap.Choice([0, 1, 2]))
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the bottom
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 0, "y": 1}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move 33% (one pixel) to the right
    aug = iaa.AffineCv2(scale=1.0, translate_percent={"x": 0.3333, "y": 0}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[1, 2] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move 33% (one pixel) to the bottom
    aug = iaa.AffineCv2(scale=1.0, translate_percent={"x": 0, "y": 0.3333}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # 0-1px to left/right and 0-1px to top/bottom
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": (-1, 1), "y": (-1, 1)}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    centers_aug = np.copy(image).astype(np.int32) * 0
    centers_aug_det = np.copy(image).astype(np.int32) * 0
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        assert len(observed_aug[0].nonzero()[0]) == 1
        assert len(observed_aug_det[0].nonzero()[0]) == 1
        centers_aug += (observed_aug[0] > 0)
        centers_aug_det += (observed_aug_det[0] > 0)

    assert nb_changed_aug >= int(nb_iterations * 0.7)
    assert nb_changed_aug_det == 0
    assert (centers_aug > int(nb_iterations * (1/9 * 0.6))).all()
    assert (centers_aug < int(nb_iterations * (1/9 * 1.4))).all()

    aug = iaa.AffineCv2(translate_percent=iap.Uniform(0.7, 0.9))
    assert isinstance(aug.translate, iap.Uniform)
    assert isinstance(aug.translate.a, iap.Deterministic)
    assert isinstance(aug.translate.b, iap.Deterministic)
    assert 0.7 - 1e-8 < aug.translate.a.value < 0.7 + 1e-8
    assert 0.9 - 1e-8 < aug.translate.b.value < 0.9 + 1e-8

    aug = iaa.AffineCv2(translate_px=iap.DiscreteUniform(1, 10))
    assert isinstance(aug.translate, iap.DiscreteUniform)
    assert isinstance(aug.translate.a, iap.Deterministic)
    assert isinstance(aug.translate.b, iap.Deterministic)
    assert aug.translate.a.value == 1
    assert aug.translate.b.value == 10

    # ---------------------
    # translate heatmaps
    # ---------------------
    heatmaps = ia.HeatmapsOnImage(
        np.float32([
            [0.0, 0.5, 0.75],
            [0.0, 0.5, 0.75],
            [0.75, 0.75, 0.75],
        ]),
        shape=(3, 3, 3)
    )
    arr_expected_1px_right = np.float32([
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.75, 0.75],
    ])
    aug = iaa.AffineCv2(translate_px={"x": 1})
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    # should still use mode=constant cval=0 even when other settings chosen
    aug = iaa.AffineCv2(translate_px={"x": 1}, cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    aug = iaa.AffineCv2(translate_px={"x": 1}, mode="replicate", cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    # ---------------------
    # rotate
    # ---------------------
    # rotate by 45 degrees
    aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=90, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, :] = 255
    image_aug[0, 1] = 255
    image_aug[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    observed[observed >= 100] = 255
    observed[observed < 100] = 0
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    observed[observed >= 100] = 255
    observed[observed < 100] = 0
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    observed[0][observed[0] >= 100] = 255
    observed[0][observed[0] < 100] = 0
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    observed[0][observed[0] >= 100] = 255
    observed[0][observed[0] < 100] = 0
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # rotate by StochasticParameter
    aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=iap.Uniform(10, 20), shear=0)
    assert isinstance(aug.rotate, iap.Uniform)
    assert isinstance(aug.rotate.a, iap.Deterministic)
    assert aug.rotate.a.value == 10
    assert isinstance(aug.rotate.b, iap.Deterministic)
    assert aug.rotate.b.value == 20

    # random rotation 0-364 degrees
    aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=(0, 364), shear=0)
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    pixels_sums_aug = np.copy(image).astype(np.int32) * 0
    pixels_sums_aug_det = np.copy(image).astype(np.int32) * 0
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        #assert len(observed_aug[0].nonzero()[0]) == 1
        #assert len(observed_aug_det[0].nonzero()[0]) == 1
        pixels_sums_aug += (observed_aug[0] > 100)
        pixels_sums_aug_det += (observed_aug_det[0] > 100)

    assert nb_changed_aug >= int(nb_iterations * 0.9)
    assert nb_changed_aug_det == 0
    # center pixel, should always be white when rotating line around center
    assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
    assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)

    # outer pixels, should sometimes be white
    # the values here had to be set quite tolerant, the middle pixels at top/left/bottom/right get more activation than expected
    outer_pixels = ([0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2])
    assert (pixels_sums_aug[outer_pixels] > int(nb_iterations * (2/8 * 0.4))).all()
    assert (pixels_sums_aug[outer_pixels] < int(nb_iterations * (2/8 * 2.0))).all()

    # ---------------------
    # shear
    # ---------------------
    # TODO

    # shear by StochasticParameter
    aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=0, shear=iap.Uniform(10, 20))
    assert isinstance(aug.shear, iap.Uniform)
    assert isinstance(aug.shear.a, iap.Deterministic)
    assert aug.shear.a.value == 10
    assert isinstance(aug.shear.b, iap.Deterministic)
    assert aug.shear.b.value == 20

    # ---------------------
    # cval
    # ---------------------
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=128)
    aug_det = aug.to_deterministic()

    image = np.ones((3, 3, 1), dtype=np.uint8) * 255
    image_aug = np.copy(image)
    images = np.array([image])
    images_list = [image]

    observed = aug.augment_images(images)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug_det.augment_images(images)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug.augment_images(images_list)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug_det.augment_images(images_list)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    # random cvals
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=(0, 255))
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    averages = []
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        averages.append(int(np.average(observed_aug)))

    assert nb_changed_aug >= int(nb_iterations * 0.9)
    assert nb_changed_aug_det == 0
    # center pixel, should always be white when rotating line around center
    assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
    assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)
    assert len(set(averages)) > 200

    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=ia.ALL)
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 255

    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=iap.DiscreteUniform(1, 5))
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 1
    assert aug.cval.b.value == 5

    # ------------
    # mode
    # ------------
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=ia.ALL)
    assert isinstance(aug.mode, iap.Choice)
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode="replicate")
    assert isinstance(aug.mode, iap.Deterministic)
    assert aug.mode.value == "replicate"
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=["replicate", "reflect"])
    assert isinstance(aug.mode, iap.Choice)
    assert len(aug.mode.a) == 2 and "replicate" in aug.mode.a and "reflect" in aug.mode.a
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=iap.Choice(["replicate", "reflect"]))
    assert isinstance(aug.mode, iap.Choice)
    assert len(aug.mode.a) == 2 and "replicate" in aug.mode.a and "reflect" in aug.mode.a

    # ------------
    # exceptions for bad inputs
    # ------------
    # scale
    got_exception = False
    try:
        aug = iaa.AffineCv2(scale=False)
    except Exception:
        got_exception = True
    assert got_exception

    # translate_px
    got_exception = False
    try:
        aug = iaa.AffineCv2(translate_px=False)
    except Exception:
        got_exception = True
    assert got_exception

    # translate_percent
    got_exception = False
    try:
        aug = iaa.AffineCv2(translate_percent=False)
    except Exception:
        got_exception = True
    assert got_exception

    # rotate
    got_exception = False
    try:
        aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=False, shear=0, cval=0)
    except Exception:
        got_exception = True
    assert got_exception

    # shear
    got_exception = False
    try:
        aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=0, shear=False, cval=0)
    except Exception:
        got_exception = True
    assert got_exception

    # cval
    got_exception = False
    try:
        aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=None)
    except Exception:
        got_exception = True
    assert got_exception

    # mode
    got_exception = False
    try:
        aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=False)
    except Exception:
        got_exception = True
    assert got_exception

    # non-existent order
    got_exception = False
    try:
        aug = iaa.AffineCv2(order=-1)
    except Exception:
        got_exception = True
    assert got_exception

    # bad order datatype
    got_exception = False
    try:
        aug = iaa.AffineCv2(order="test")
    except Exception:
        got_exception = True
    assert got_exception

    # ----------
    # get_parameters
    # ----------
    aug = iaa.AffineCv2(scale=1, translate_px=2, rotate=3, shear=4, order=1, cval=0, mode="constant")
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)  # scale
    assert isinstance(params[1], iap.Deterministic)  # translate
    assert isinstance(params[2], iap.Deterministic)  # rotate
    assert isinstance(params[3], iap.Deterministic)  # shear
    assert params[0].value == 1  # scale
    assert params[1].value == 2  # translate
    assert params[2].value == 3  # rotate
    assert params[3].value == 4  # shear
    assert params[4].value == 1  # order
    assert params[5].value == 0  # cval
    assert params[6].value == "constant"  # mode


def test_PiecewiseAffine():
    reseed()

    img = np.zeros((60, 80), dtype=np.uint8)
    img[:, 9:11+1] = 255
    img[:, 69:71+1] = 255
    mask = img > 0
    heatmaps = ia.HeatmapsOnImage((img / 255.0).astype(np.float32), shape=(60, 80, 3))
    heatmaps_arr = heatmaps.get_arr()

    # -----
    # scale
    # -----
    # basic test
    aug = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
    observed = aug.augment_image(img)
    assert 100.0 < np.average(observed[mask]) < np.average(img[mask])
    assert 75.0 > np.average(observed[~mask]) > np.average(img[~mask])

    # basic test, heatmaps
    aug = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
    observed = aug.augment_heatmaps([heatmaps])[0]
    observed_arr = observed.get_arr()
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert 100.0/255.0 < np.average(observed_arr[mask]) < np.average(heatmaps_arr[mask])
    assert 75.0/255.0 > np.average(observed_arr[~mask]) > np.average(heatmaps_arr[~mask])

    # scale 0
    aug = iaa.PiecewiseAffine(scale=0, nb_rows=12, nb_cols=4)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    # scale 0, heatmaps
    aug = iaa.PiecewiseAffine(scale=0, nb_rows=12, nb_cols=4)
    observed = aug.augment_heatmaps([heatmaps])[0]
    observed_arr = observed.get_arr()
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed_arr, heatmaps_arr)

    # stronger scale should lead to stronger changes
    aug1 = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
    aug2 = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
    observed1 = aug1.augment_image(img)
    observed2 = aug2.augment_image(img)
    assert np.average(observed1[~mask]) < np.average(observed2[~mask])

    # stronger scale should lead to stronger changes, heatmaps
    aug1 = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
    aug2 = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
    observed1 = aug1.augment_heatmaps([heatmaps])[0]
    observed1_arr = observed1.get_arr()
    observed2 = aug2.augment_heatmaps([heatmaps])[0]
    observed2_arr = observed2.get_arr()
    assert observed1.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed1.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed1.max_value < heatmaps.max_value + 1e-6
    assert observed2.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed2.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed2.max_value < heatmaps.max_value + 1e-6
    assert np.average(observed1_arr[~mask]) < np.average(observed2_arr[~mask])

    # scale as list
    aug1 = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
    aug2 = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
    aug = iaa.PiecewiseAffine(scale=[0.01, 0.10], nb_rows=12, nb_cols=4)
    assert isinstance(aug.scale, iap.Choice)
    assert 0.01 - 1e-8 < aug.scale.a[0] < 0.01 + 1e-8
    assert 0.10 - 1e-8 < aug.scale.a[1] < 0.10 + 1e-8

    avg1 = np.average([np.average(aug1.augment_image(img) * (~mask).astype(np.float32)) for _ in sm.xrange(3)])
    avg2 = np.average([np.average(aug2.augment_image(img) * (~mask).astype(np.float32)) for _ in sm.xrange(3)])
    seen = [0, 0]
    for _ in sm.xrange(15):
        observed = aug.augment_image(img)
        avg = np.average(observed * (~mask).astype(np.float32))
        diff1 = abs(avg - avg1)
        diff2 = abs(avg - avg2)
        if diff1 < diff2:
            seen[0] += 1
        else:
            seen[1] += 1
    assert seen[0] > 0
    assert seen[1] > 0

    # scale as tuple
    aug = iaa.PiecewiseAffine(scale=(0.01, 0.10), nb_rows=12, nb_cols=4)
    assert isinstance(aug.jitter.scale, iap.Uniform)
    assert isinstance(aug.jitter.scale.a, iap.Deterministic)
    assert isinstance(aug.jitter.scale.b, iap.Deterministic)
    assert 0.01 - 1e-8 < aug.jitter.scale.a.value < 0.01 + 1e-8
    assert 0.10 - 1e-8 < aug.jitter.scale.b.value < 0.10 + 1e-8

    # scale as StochasticParameter
    aug = iaa.PiecewiseAffine(scale=iap.Uniform(0.01, 0.10), nb_rows=12, nb_cols=4)
    assert isinstance(aug.jitter.scale, iap.Uniform)
    assert isinstance(aug.jitter.scale.a, iap.Deterministic)
    assert isinstance(aug.jitter.scale.b, iap.Deterministic)
    assert 0.01 - 1e-8 < aug.jitter.scale.a.value < 0.01 + 1e-8
    assert 0.10 - 1e-8 < aug.jitter.scale.b.value < 0.10 + 1e-8

    # bad datatype for scale
    got_exception = False
    try:
        aug = iaa.PiecewiseAffine(scale=False, nb_rows=12, nb_cols=4)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # -----
    # rows and cols
    # -----
    # verify effects of rows/cols
    aug1 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=4)
    aug2 = iaa.PiecewiseAffine(scale=0.05, nb_rows=30, nb_cols=4)
    std1 = []
    std2 = []
    for _ in sm.xrange(3):
        observed1 = aug1.augment_image(img)
        observed2 = aug2.augment_image(img)
        grad_vert1 = observed1[1:, :].astype(np.float32) - observed1[:-1, :].astype(np.float32)
        grad_vert2 = observed2[1:, :].astype(np.float32) - observed2[:-1, :].astype(np.float32)
        grad_vert1 = grad_vert1 * (~mask[1:, :]).astype(np.float32)
        grad_vert2 = grad_vert2 * (~mask[1:, :]).astype(np.float32)
        std1.append(np.std(grad_vert1))
        std2.append(np.std(grad_vert2))
    std1 = np.average(std1)
    std2 = np.average(std2)
    assert std1 < std2

    # -----
    # rows
    # -----
    # rows as list
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=[4, 20], nb_cols=4)
    assert isinstance(aug.nb_rows, iap.Choice)
    assert aug.nb_rows.a[0] == 4
    assert aug.nb_rows.a[1] == 20

    seen = [0, 0]
    for _ in sm.xrange(20):
        observed = aug.augment_image(img)
        grad_vert = observed[1:, :].astype(np.float32) - observed[:-1, :].astype(np.float32)
        grad_vert = grad_vert * (~mask[1:, :]).astype(np.float32)
        std = np.std(grad_vert)
        diff1 = abs(std - std1)
        diff2 = abs(std - std2)
        if diff1 < diff2:
            seen[0] += 1
        else:
            seen[1] += 1
    assert seen[0] > 0
    assert seen[1] > 0

    # rows as tuple
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=(4, 20), nb_cols=4)
    assert isinstance(aug.nb_rows, iap.DiscreteUniform)
    assert isinstance(aug.nb_rows.a, iap.Deterministic)
    assert isinstance(aug.nb_rows.b, iap.Deterministic)
    assert aug.nb_rows.a.value == 4
    assert aug.nb_rows.b.value == 20

    # rows as StochasticParameter
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=iap.DiscreteUniform(4, 20), nb_cols=4)
    assert isinstance(aug.nb_rows, iap.DiscreteUniform)
    assert isinstance(aug.nb_rows.a, iap.Deterministic)
    assert isinstance(aug.nb_rows.b, iap.Deterministic)
    assert aug.nb_rows.a.value == 4
    assert aug.nb_rows.b.value == 20

    # bad datatype for rows
    got_exception = False
    try:
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=False, nb_cols=4)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # -----
    # nb_cols
    # -----

    # cols as list
    img_cols = img.T
    mask_cols = mask.T
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=[4, 20])
    assert isinstance(aug.nb_cols, iap.Choice)
    assert aug.nb_cols.a[0] == 4
    assert aug.nb_cols.a[1] == 20

    aug1 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=4)
    aug2 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=20)

    std1 = []
    std2 = []
    for _ in sm.xrange(3):
        observed1 = aug1.augment_image(img_cols)
        observed2 = aug2.augment_image(img_cols)
        grad_hori1 = observed1[:, 1:].astype(np.float32) - observed1[:, :-1].astype(np.float32)
        grad_hori2 = observed2[:, 1:].astype(np.float32) - observed2[:, :-1].astype(np.float32)
        grad_hori1 = grad_hori1 * (~mask_cols[:, 1:]).astype(np.float32)
        grad_hori2 = grad_hori2 * (~mask_cols[:, 1:]).astype(np.float32)
        std1.append(np.std(grad_hori1))
        std2.append(np.std(grad_hori2))
    std1 = np.average(std1)
    std2 = np.average(std2)

    seen = [0, 0]
    for _ in sm.xrange(15):
        observed = aug.augment_image(img_cols)

        grad_hori = observed[:, 1:].astype(np.float32) - observed[:, :-1].astype(np.float32)
        grad_hori = grad_hori * (~mask_cols[:, 1:]).astype(np.float32)
        std = np.std(grad_hori)
        diff1 = abs(std - std1)
        diff2 = abs(std - std2)

        if diff1 < diff2:
            seen[0] += 1
        else:
            seen[1] += 1
    assert seen[0] > 0
    assert seen[1] > 0

    # cols as tuple
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=(4, 20))
    assert isinstance(aug.nb_cols, iap.DiscreteUniform)
    assert isinstance(aug.nb_cols.a, iap.Deterministic)
    assert isinstance(aug.nb_cols.b, iap.Deterministic)
    assert aug.nb_cols.a.value == 4
    assert aug.nb_cols.b.value == 20

    # cols as StochasticParameter
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=iap.DiscreteUniform(4, 20))
    assert isinstance(aug.nb_cols, iap.DiscreteUniform)
    assert isinstance(aug.nb_cols.a, iap.Deterministic)
    assert isinstance(aug.nb_cols.b, iap.Deterministic)
    assert aug.nb_cols.a.value == 4
    assert aug.nb_cols.b.value == 20

    # bad datatype for cols
    got_exception = False
    try:
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # -----
    # order
    # -----
    # single int for order
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=0)
    assert isinstance(aug.order, iap.Deterministic)
    assert aug.order.value == 0

    # list for order
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=[0, 1, 3])
    assert isinstance(aug.order, iap.Choice)
    assert all([v in aug.order.a for v in [0, 1, 3]])

    # StochasticParameter for order
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=iap.Choice([0, 1, 3]))
    assert isinstance(aug.order, iap.Choice)
    assert all([v in aug.order.a for v in [0, 1, 3]])

    # ALL for order
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=ia.ALL)
    assert isinstance(aug.order, iap.Choice)
    assert all([v in aug.order.a for v in [0, 1, 3, 4, 5]])

    # bad datatype for order
    got_exception = False
    try:
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # -----
    # cval
    # -----
    # cval as deterministic
    img = np.zeros((50, 50, 3), dtype=np.uint8) + 255
    aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=10, nb_cols=10, mode="constant", cval=0)
    observed = aug.augment_image(img)
    assert np.sum([observed[:, :] == [0, 0, 0]]) > 0

    # cval as deterministic, heatmaps should always use cval=0
    heatmaps = ia.HeatmapsOnImage(np.zeros((50, 50, 1), dtype=np.float32), shape=(50, 50, 3))
    aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=10, nb_cols=10, mode="constant", cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert np.sum([observed.get_arr()[:, :] >= 0.01]) == 0

    # cval as list
    img = np.zeros((20, 20), dtype=np.uint8) + 255
    aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=5, nb_cols=5, mode="constant", cval=[0, 10])
    assert isinstance(aug.cval, iap.Choice)
    assert aug.cval.a[0] == 0
    assert aug.cval.a[1] == 10

    seen = [0, 0, 0]
    for _ in sm.xrange(30):
        observed = aug.augment_image(img)
        nb_0 = np.sum([observed[:, :] == 0])
        nb_10 = np.sum([observed[:, :] == 10])
        if nb_0 > 0:
            seen[0] += 1
        elif nb_10 > 0:
            seen[1] += 1
        else:
            seen[2] += 1
    assert seen[0] > 5
    assert seen[1] > 5
    assert seen[2] <= 4

    # cval as tuple
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode="constant", cval=(0, 10))
    assert isinstance(aug.cval, iap.Uniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 10

    # cval as StochasticParameter
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode="constant", cval=iap.DiscreteUniform(0, 10))
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 10

    # ALL as cval
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode="constant", cval=ia.ALL)
    assert isinstance(aug.cval, iap.Uniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 255

    # bas datatype for cval
    got_exception = False
    try:
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, cval=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # -----
    # mode
    # -----
    # single string for mode
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode="nearest")
    assert isinstance(aug.mode, iap.Deterministic)
    assert aug.mode.value == "nearest"

    # list for mode
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode=["nearest", "edge", "symmetric"])
    assert isinstance(aug.mode, iap.Choice)
    assert all([v in aug.mode.a for v in ["nearest", "edge", "symmetric"]])

    # StochasticParameter for mode
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode=iap.Choice(["nearest", "edge", "symmetric"]))
    assert isinstance(aug.mode, iap.Choice)
    assert all([v in aug.mode.a for v in ["nearest", "edge", "symmetric"]])

    # ALL for mode
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode=ia.ALL)
    assert isinstance(aug.mode, iap.Choice)
    assert all([v in aug.mode.a for v in ["constant", "edge", "symmetric", "reflect", "wrap"]])

    # bad datatype for mode
    got_exception = False
    try:
        aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # ---------
    # keypoints
    # ---------
    # basic test
    img = np.zeros((100, 80), dtype=np.uint8)
    img[:, 9:11+1] = 255
    img[:, 69:71+1] = 255
    mask = img > 0
    kps = [ia.Keypoint(x=10, y=20), ia.Keypoint(x=10, y=40),
           ia.Keypoint(x=70, y=20), ia.Keypoint(x=70, y=40)]
    kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)

    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=10, nb_cols=10)
    aug_det = aug.to_deterministic()
    observed_img = aug_det.augment_image(img)
    observed_kpsoi = aug_det.augment_keypoints([kpsoi])
    assert not keypoints_equal([kpsoi], observed_kpsoi)
    for kp in observed_kpsoi[0].keypoints:
        assert observed_img[int(kp.y), int(kp.x)] > 0

    # scale 0
    aug = iaa.PiecewiseAffine(scale=0, nb_rows=10, nb_cols=10)
    observed = aug.augment_keypoints([kpsoi])
    assert keypoints_equal([kpsoi], observed)

    # keypoints outside of image
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=10, nb_cols=10)
    kps = [ia.Keypoint(x=-10, y=-20)]
    kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)
    observed = aug.augment_keypoints([kpsoi])
    assert keypoints_equal([kpsoi], observed)

    # ---------
    # get_parameters
    # ---------
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=10, order=1, cval=2, mode="nearest", absolute_scale=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert isinstance(params[2], iap.Deterministic)
    assert isinstance(params[3], iap.Deterministic)
    assert isinstance(params[4], iap.Deterministic)
    assert isinstance(params[5], iap.Deterministic)
    assert params[6] == False
    assert 0.1 - 1e-8 < params[0].value < 0.1 + 1e-8
    assert params[1].value == 8
    assert params[2].value == 10
    assert params[3].value == 1
    assert params[4].value == 2
    assert params[5].value == "nearest"


def test_PerspectiveTransform():
    reseed()

    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:20, 10:20] = 255
    heatmaps = ia.HeatmapsOnImage((img / 255.0).astype(np.float32), shape=img.shape)

    # without keep_size
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
    aug.jitter = iap.Deterministic(0.2)
    observed = aug.augment_image(img)
    expected = img[int(30*0.2):int(30*0.8), int(30*0.2):int(30*0.8)]
    assert all([abs(s1-s2)<=1 for s1, s2 in zip(observed.shape, expected.shape)])
    if observed.shape != expected.shape:
        observed = ia.imresize_single_image(observed, expected.shape[0:2], interpolation="cubic")
    # differences seem to mainly appear around the border of the inner rectangle, possibly
    # due to interpolation
    """
    ia.imshow(
        np.hstack([
            observed,
            expected,
            np.abs(observed.astype(np.int32) - expected.astype(np.int32)).astype(np.uint8)
        ])
    )
    print(np.average(np.abs(observed.astype(np.int32) - expected.astype(np.int32))))
    """
    assert np.average(np.abs(observed.astype(np.int32) - expected.astype(np.int32))) < 30.0

    # with keep_size
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
    aug.jitter = iap.Deterministic(0.2)
    observed = aug.augment_image(img)
    expected = img[int(30*0.2):int(30*0.8), int(30*0.2):int(30*0.8)]
    expected = ia.imresize_single_image(expected, img.shape[0:2], interpolation="cubic")
    assert observed.shape == img.shape
    # differences seem to mainly appear around the border of the inner rectangle, possibly
    # due to interpolation
    assert np.average(np.abs(observed.astype(np.int32) - expected.astype(np.int32))) < 30.0
    #expected = ia.imresize_single_image(expected, (30, 30))

    # with keep_size, heatmaps
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
    aug.jitter = iap.Deterministic(0.2)
    observed = aug.augment_heatmaps([heatmaps])[0]
    expected = heatmaps.get_arr()[int(30*0.2):int(30*0.8), int(30*0.2):int(30*0.8)]
    expected = ia.imresize_single_image((expected*255).astype(np.uint8), img.shape[0:2], interpolation="cubic")
    expected = (expected / 255.0).astype(np.float32)
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    # differences seem to mainly appear around the border of the inner rectangle, possibly
    # due to interpolation
    assert np.average(np.abs(observed.get_arr() - expected)) < 30.0
    #expected = ia.imresize_single_image(expected, (30, 30))

    # with keep_size, RGB images
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
    aug.jitter = iap.Deterministic(0.2)
    imgs = np.tile(img[np.newaxis, :, :, np.newaxis], (2, 1, 1, 3))
    observed = aug.augment_images(imgs)
    for img_idx in sm.xrange(2):
        for c in sm.xrange(3):
            observed_i = observed[img_idx, :, :, c]
            expected = imgs[img_idx, int(30*0.2):int(30*0.8), int(30*0.2):int(30*0.8), c]
            expected = ia.imresize_single_image(expected, imgs.shape[1:3], interpolation="cubic")
            assert observed_i.shape == imgs.shape[1:3]
            # differences seem to mainly appear around the border of the inner rectangle, possibly
            # due to interpolation
            assert np.average(np.abs(observed_i.astype(np.int32) - expected.astype(np.int32))) < 30.0
            #expected = ia.imresize_single_image(expected, (30, 30))

    # tuple for scale
    aug = iaa.PerspectiveTransform(scale=(0.1, 0.2))
    assert isinstance(aug.jitter.scale, iap.Uniform)
    assert isinstance(aug.jitter.scale.a, iap.Deterministic)
    assert isinstance(aug.jitter.scale.b, iap.Deterministic)
    assert 0.1 - 1e-8 < aug.jitter.scale.a.value < 0.1 + 1e-8
    assert 0.2 - 1e-8 < aug.jitter.scale.b.value < 0.2 + 1e-8

    # list for scale
    aug = iaa.PerspectiveTransform(scale=[0.1, 0.2, 0.3])
    assert isinstance(aug.jitter.scale, iap.Choice)
    assert len(aug.jitter.scale.a) == 3
    assert 0.1 - 1e-8 < aug.jitter.scale.a[0] < 0.1 + 1e-8
    assert 0.2 - 1e-8 < aug.jitter.scale.a[1] < 0.2 + 1e-8
    assert 0.3 - 1e-8 < aug.jitter.scale.a[2] < 0.3 + 1e-8

    # StochasticParameter for scale
    aug = iaa.PerspectiveTransform(scale=iap.Choice([0.1, 0.2, 0.3]))
    assert isinstance(aug.jitter.scale, iap.Choice)
    assert len(aug.jitter.scale.a) == 3
    assert 0.1 - 1e-8 < aug.jitter.scale.a[0] < 0.1 + 1e-8
    assert 0.2 - 1e-8 < aug.jitter.scale.a[1] < 0.2 + 1e-8
    assert 0.3 - 1e-8 < aug.jitter.scale.a[2] < 0.3 + 1e-8

    # bad datatype for scale
    got_exception = False
    try:
        aug = iaa.PerspectiveTransform(scale=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # keypoint augmentation without keep_size
    # TODO deviations of around 0.4-0.7 in this and the next test (between expected and observed
    # coordinates) -- why?
    kps = [ia.Keypoint(x=10, y=10), ia.Keypoint(x=14, y=11)]
    kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
    aug.jitter = iap.Deterministic(0.2)
    observed = aug.augment_keypoints([kpsoi])
    kps_expected = [
        ia.Keypoint(x=10-0.2*30, y=10-0.2*30),
        ia.Keypoint(x=14-0.2*30, y=11-0.2*30)
    ]
    for kp_observed, kp_expected in zip(observed[0].keypoints, kps_expected):
        assert kp_expected.x - 1.5 < kp_observed.x < kp_expected.x + 1.5
        assert kp_expected.y - 1.5 < kp_observed.y < kp_expected.y + 1.5

    # keypoint augmentation with keep_size
    kps = [ia.Keypoint(x=10, y=10), ia.Keypoint(x=14, y=11)]
    kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
    aug.jitter = iap.Deterministic(0.2)
    observed = aug.augment_keypoints([kpsoi])
    kps_expected = [
        ia.Keypoint(x=((10-0.2*30)/(30*0.6))*30, y=((10-0.2*30)/(30*0.6))*30),
        ia.Keypoint(x=((14-0.2*30)/(30*0.6))*30, y=((11-0.2*30)/(30*0.6))*30)
    ]
    for kp_observed, kp_expected in zip(observed[0].keypoints, kps_expected):
        assert kp_expected.x - 1.5 < kp_observed.x < kp_expected.x + 1.5
        assert kp_expected.y - 1.5 < kp_observed.y < kp_expected.y + 1.5

    # get_parameters
    aug = iaa.PerspectiveTransform(scale=0.1, keep_size=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Normal)
    assert isinstance(params[0].scale, iap.Deterministic)
    assert 0.1 - 1e-8 < params[0].scale.value < 0.1 + 1e-8
    assert params[1] == False


def test_ElasticTransformation():
    reseed()

    img = np.zeros((50, 50), dtype=np.uint8) + 255
    img = np.pad(img, ((100, 100), (100, 100)), mode="constant", constant_values=0)
    mask = img > 0
    heatmaps = ia.HeatmapsOnImage((img / 255.0).astype(np.float32), shape=img.shape)

    img_nonsquare = np.zeros((50, 100), dtype=np.uint8) + 255
    img_nonsquare = np.pad(img_nonsquare, ((100, 100), (100, 100)), mode="constant", constant_values=0)
    mask_nonsquare = img_nonsquare > 0

    # test basic funtionality
    aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)
    observed = aug.augment_image(img)
    # assume that some white/255 pixels have been moved away from the center and replaced by black/0 pixels
    assert np.sum(observed[mask]) < np.sum(img[mask])
    # assume that some black/0 pixels have been moved away from the outer area and replaced by white/255 pixels
    assert np.sum(observed[~mask]) > np.sum(img[~mask])

    # test basic funtionality with non-square images
    aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)
    observed = aug.augment_image(img_nonsquare)
    assert np.sum(observed[mask_nonsquare]) < np.sum(img_nonsquare[mask_nonsquare])
    assert np.sum(observed[~mask_nonsquare]) > np.sum(img_nonsquare[~mask_nonsquare])

    # test basic funtionality, heatmaps
    aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.sum(observed.get_arr()[mask]) < np.sum(heatmaps.get_arr()[mask])
    assert np.sum(observed.get_arr()[~mask]) > np.sum(heatmaps.get_arr()[~mask])

    # test effects of increased alpha strength
    aug1 = iaa.ElasticTransformation(alpha=0.1, sigma=0.25)
    aug2 = iaa.ElasticTransformation(alpha=5.0, sigma=0.25)
    observed1 = aug1.augment_image(img)
    observed2 = aug2.augment_image(img)
    # assume that the inner area has become more black-ish when using high alphas (more white pixels were moved out of the inner area)
    assert np.sum(observed1[mask]) > np.sum(observed2[mask])
    # assume that the outer area has become more white-ish when using high alphas (more black pixels were moved into the inner area)
    assert np.sum(observed1[~mask]) < np.sum(observed2[~mask])

    # test effects of increased alpha strength, heatmaps
    aug1 = iaa.ElasticTransformation(alpha=0.1, sigma=0.25)
    aug2 = iaa.ElasticTransformation(alpha=5.0, sigma=0.25)
    observed1 = aug1.augment_heatmaps([heatmaps])[0]
    observed2 = aug2.augment_heatmaps([heatmaps])[0]
    assert observed1.shape == heatmaps.shape
    assert observed2.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed1.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed1.max_value < heatmaps.max_value + 1e-6
    assert heatmaps.min_value - 1e-6 < observed2.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed2.max_value < heatmaps.max_value + 1e-6
    assert np.sum(observed1.get_arr()[mask]) > np.sum(observed2.get_arr()[mask])
    assert np.sum(observed1.get_arr()[~mask]) < np.sum(observed2.get_arr()[~mask])

    # test effectsof increased sigmas
    aug1 = iaa.ElasticTransformation(alpha=3.0, sigma=0.1)
    aug2 = iaa.ElasticTransformation(alpha=3.0, sigma=3.0)
    observed1 = aug1.augment_image(img)
    observed2 = aug2.augment_image(img)
    observed1_std_hori = np.std(observed1.astype(np.float32)[:, 1:] - observed1.astype(np.float32)[:, :-1])
    observed2_std_hori = np.std(observed2.astype(np.float32)[:, 1:] - observed2.astype(np.float32)[:, :-1])
    observed1_std_vert = np.std(observed1.astype(np.float32)[1:, :] - observed1.astype(np.float32)[:-1, :])
    observed2_std_vert = np.std(observed2.astype(np.float32)[1:, :] - observed2.astype(np.float32)[:-1, :])
    observed1_std = (observed1_std_hori + observed1_std_vert) / 2
    observed2_std = (observed2_std_hori + observed2_std_vert) / 2
    assert observed1_std > observed2_std

    # test alpha being iap.Choice
    aug = iaa.ElasticTransformation(alpha=iap.Choice([0.001, 5.0]), sigma=0.25)
    seen = [0, 0]
    for _ in sm.xrange(100):
        observed = aug.augment_image(img)
        diff = np.average(np.abs(img.astype(np.float32) - observed.astype(np.float32)))
        if diff < 1.0:
            seen[0] += 1
        else:
            seen[1] += 1
    assert seen[0] > 10
    assert seen[1] > 10

    # test alpha being tuple
    aug = iaa.ElasticTransformation(alpha=(1.0, 2.0), sigma=0.25)
    assert isinstance(aug.alpha, iap.Uniform)
    assert isinstance(aug.alpha.a, iap.Deterministic)
    assert isinstance(aug.alpha.b, iap.Deterministic)
    assert 1.0 - 1e-8 < aug.alpha.a.value < 1.0 + 1e-8
    assert 2.0 - 1e-8 < aug.alpha.b.value < 2.0 + 1e-8

    # test alpha having bad datatype
    got_exception = False
    try:
        aug = iaa.ElasticTransformation(alpha=False, sigma=0.25)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # test sigma being iap.Choice
    aug = iaa.ElasticTransformation(alpha=3.0, sigma=iap.Choice([0.01, 5.0]))
    seen = [0, 0]
    for _ in sm.xrange(100):
        observed = aug.augment_image(img)

        observed_std_hori = np.std(observed.astype(np.float32)[:, 1:] - observed.astype(np.float32)[:, :-1])
        observed_std_vert = np.std(observed.astype(np.float32)[1:, :] - observed.astype(np.float32)[:-1, :])
        observed_std = (observed_std_hori + observed_std_vert) / 2

        if observed_std > 10.0:
            seen[0] += 1
        else:
            seen[1] += 1
    assert seen[0] > 10
    assert seen[1] > 10

    # test sigma being tuple
    aug = iaa.ElasticTransformation(alpha=0.25, sigma=(1.0, 2.0))
    assert isinstance(aug.sigma, iap.Uniform)
    assert isinstance(aug.sigma.a, iap.Deterministic)
    assert isinstance(aug.sigma.b, iap.Deterministic)
    assert 1.0 - 1e-8 < aug.sigma.a.value < 1.0 + 1e-8
    assert 2.0 - 1e-8 < aug.sigma.b.value < 2.0 + 1e-8

    # test sigma having bad datatype
    got_exception = False
    try:
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # order
    # no proper tests here, because unclear how to test
    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=ia.ALL)
    assert isinstance(aug.order, iap.Choice)
    assert all([order in aug.order.a for order in [0, 1, 2, 3, 4, 5]])

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=1)
    assert isinstance(aug.order, iap.Deterministic)
    assert aug.order.value == 1

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=[0, 1, 2])
    assert isinstance(aug.order, iap.Choice)
    assert all([order in aug.order.a for order in [0, 1, 2]])

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=iap.Choice([0, 1, 2, 3]))
    assert isinstance(aug.order, iap.Choice)
    assert all([order in aug.order.a for order in [0, 1, 2, 3]])

    got_exception = False
    try:
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # cval
    # few proper tests here, because unclear how to test
    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=ia.ALL)
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 255

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=128)
    assert isinstance(aug.cval, iap.Deterministic)
    assert aug.cval.value == 128

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=(128, 255))
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 128
    assert aug.cval.b.value == 255

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=[16, 32, 64])
    assert isinstance(aug.cval, iap.Choice)
    assert all([cval in aug.cval.a for cval in [16, 32, 64]])

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=iap.Choice([16, 32, 64]))
    assert isinstance(aug.cval, iap.Choice)
    assert all([cval in aug.cval.a for cval in [16, 32, 64]])

    aug = iaa.ElasticTransformation(alpha=3.0, sigma=3.0, mode="constant", cval=255)
    img = np.zeros((50, 50), dtype=np.uint8)
    observed = aug.augment_image(img)
    assert np.sum(observed == 255) > 0

    aug = iaa.ElasticTransformation(alpha=3.0, sigma=3.0, mode="constant", cval=0)
    img = np.zeros((50, 50), dtype=np.uint8)
    observed = aug.augment_image(img)
    assert np.sum(observed == 255) == 0

    got_exception = False
    try:
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # cval with heatmaps
    heatmaps = ia.HeatmapsOnImage(np.zeros((32, 32, 1), dtype=np.float32), shape=(32, 32, 3))
    aug = iaa.ElasticTransformation(alpha=3.0, sigma=3.0, mode="constant", cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.sum(observed.get_arr() > 0.01) == 0

    # mode
    # no proper tests here, because unclear how to test
    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode=ia.ALL)
    assert isinstance(aug.mode, iap.Choice)
    assert all([mode in aug.mode.a for mode in ["constant", "nearest", "reflect", "wrap"]])

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode="nearest")
    assert isinstance(aug.mode, iap.Deterministic)
    assert aug.mode.value == "nearest"

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode=["constant", "nearest"])
    assert isinstance(aug.mode, iap.Choice)
    assert all([mode in aug.mode.a for mode in ["constant", "nearest"]])

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode=iap.Choice(["constant", "nearest"]))
    assert isinstance(aug.mode, iap.Choice)
    assert all([mode in aug.mode.a for mode in ["constant", "nearest"]])

    got_exception = False
    try:
        aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # keypoints
    # for small alpha, should not move if below threshold
    alpha_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH
    sigma_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = 1.0
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = 0
    kps = [ia.Keypoint(x=1, y=1), ia.Keypoint(x=15, y=25), ia.Keypoint(x=5, y=5),
           ia.Keypoint(x=7, y=4), ia.Keypoint(x=48, y=5), ia.Keypoint(x=21, y=37),
           ia.Keypoint(x=32, y=39), ia.Keypoint(x=6, y=8), ia.Keypoint(x=12, y=21),
           ia.Keypoint(x=3, y=45), ia.Keypoint(x=45, y=3), ia.Keypoint(x=7, y=48)]
    kpsoi = ia.KeypointsOnImage(kps, shape=(50, 50))
    aug = iaa.ElasticTransformation(alpha=0.001, sigma=1.0)
    observed = aug.augment_keypoints([kpsoi])[0]
    d = kpsoi.get_coords_array() - observed.get_coords_array()
    d[:, 0] = d[:, 0] ** 2
    d[:, 1] = d[:, 1] ** 2
    d = np.sum(d, axis=1)
    d = np.average(d, axis=0)
    assert d < 1e-8
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = alpha_thresh_orig
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = sigma_thresh_orig

    # for small sigma, should not move if below threshold
    alpha_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH
    sigma_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = 0.0
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = 1.0
    kps = [ia.Keypoint(x=1, y=1), ia.Keypoint(x=15, y=25), ia.Keypoint(x=5, y=5),
           ia.Keypoint(x=7, y=4), ia.Keypoint(x=48, y=5), ia.Keypoint(x=21, y=37),
           ia.Keypoint(x=32, y=39), ia.Keypoint(x=6, y=8), ia.Keypoint(x=12, y=21),
           ia.Keypoint(x=3, y=45), ia.Keypoint(x=45, y=3), ia.Keypoint(x=7, y=48)]
    kpsoi = ia.KeypointsOnImage(kps, shape=(50, 50))
    aug = iaa.ElasticTransformation(alpha=1.0, sigma=0.001)
    observed = aug.augment_keypoints([kpsoi])[0]
    d = kpsoi.get_coords_array() - observed.get_coords_array()
    d[:, 0] = d[:, 0] ** 2
    d[:, 1] = d[:, 1] ** 2
    d = np.sum(d, axis=1)
    d = np.average(d, axis=0)
    assert d < 1e-8
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = alpha_thresh_orig
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = sigma_thresh_orig

    # for small alpha (at sigma 1.0), should barely move
    alpha_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH
    sigma_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = 0
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = 0
    kps = [ia.Keypoint(x=1, y=1), ia.Keypoint(x=15, y=25), ia.Keypoint(x=5, y=5),
           ia.Keypoint(x=7, y=4), ia.Keypoint(x=48, y=5), ia.Keypoint(x=21, y=37),
           ia.Keypoint(x=32, y=39), ia.Keypoint(x=6, y=8), ia.Keypoint(x=12, y=21),
           ia.Keypoint(x=3, y=45), ia.Keypoint(x=45, y=3), ia.Keypoint(x=7, y=48)]
    kpsoi = ia.KeypointsOnImage(kps, shape=(50, 50))
    aug = iaa.ElasticTransformation(alpha=0.001, sigma=1.0)
    observed = aug.augment_keypoints([kpsoi])[0]
    d = kpsoi.get_coords_array() - observed.get_coords_array()
    d[:, 0] = d[:, 0] ** 2
    d[:, 1] = d[:, 1] ** 2
    d = np.sum(d, axis=1)
    d = np.average(d, axis=0)
    assert d < 0.5
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = alpha_thresh_orig
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = sigma_thresh_orig

    # get_parameters()
    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=2, cval=10, mode="constant")
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert isinstance(params[2], iap.Deterministic)
    assert isinstance(params[3], iap.Deterministic)
    assert isinstance(params[4], iap.Deterministic)
    assert 0.25 - 1e-8 < params[0].value < 0.25 + 1e-8
    assert 1.0 - 1e-8 < params[1].value < 1.0 + 1e-8
    assert params[2].value == 2
    assert params[3].value == 10
    assert params[4].value == "constant"


def test_copy_dtypes_for_restore():
    # TODO using dtype=np.bool is causing this to fail as it ends up being <type bool> instead of
    # <type 'numpy.bool_'>. Any problems from that for the library?
    images = [
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.zeros((10, 16, 3), dtype=np.float32),
        np.zeros((20, 10, 6), dtype=np.int32)
    ]

    dtypes_copy = iaa.copy_dtypes_for_restore(images, force_list=False)
    assert all([dtype_i.type == dtype_j for dtype_i, dtype_j in zip(dtypes_copy, [np.uint8, np.float32, np.int32])])

    dts = [np.uint8, np.float32, np.int32]
    for dt in dts:
        images = np.zeros((10, 16, 32, 3), dtype=dt)
        dtypes_copy = iaa.copy_dtypes_for_restore(images)
        assert isinstance(dtypes_copy, np.dtype)
        assert dtypes_copy.type == dt

        dtypes_copy = iaa.copy_dtypes_for_restore(images, force_list=True)
        assert isinstance(dtypes_copy, list)
        assert all([dtype_i.type == dt for dtype_i in dtypes_copy])


def test_restore_augmented_image_dtype_():
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image_result = iaa.restore_augmented_image_dtype_(image, np.int32)
    assert image_result.dtype.type == np.int32


def test_restore_augmented_image_dtype():
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image_result = iaa.restore_augmented_image_dtype(image, np.int32)
    assert image_result.dtype.type == np.int32


def test_restore_augmented_images_dtypes_():
    images = np.zeros((10, 16, 32, 3), dtype=np.int32)
    dtypes = iaa.copy_dtypes_for_restore(images)
    images = images.astype(np.uint8)
    assert images.dtype.type == np.uint8
    images_result = iaa.restore_augmented_images_dtypes_(images, dtypes)
    assert images_result.dtype.type == np.int32

    images = [np.zeros((16, 32, 3), dtype=np.int32) for _ in sm.xrange(10)]
    dtypes = iaa.copy_dtypes_for_restore(images)
    images = [image.astype(np.uint8) for image in images]
    assert all([image.dtype.type == np.uint8 for image in images])
    images_result = iaa.restore_augmented_images_dtypes_(images, dtypes)
    assert all([image_result.dtype.type == np.int32 for image_result in images_result])


def test_restore_augmented_images_dtypes():
    images = np.zeros((10, 16, 32, 3), dtype=np.int32)
    dtypes = iaa.copy_dtypes_for_restore(images)
    images = images.astype(np.uint8)
    assert images.dtype.type == np.uint8
    images_restored = iaa.restore_augmented_images_dtypes(images, dtypes)
    assert images_restored.dtype.type == np.int32

    images = [np.zeros((16, 32, 3), dtype=np.int32) for _ in sm.xrange(10)]
    dtypes = iaa.copy_dtypes_for_restore(images)
    images = [image.astype(np.uint8) for image in images]
    assert all([image.dtype.type == np.uint8 for image in images])
    images_restored = iaa.restore_augmented_images_dtypes(images, dtypes)
    assert all([image_restored.dtype.type == np.int32 for image_restored in images_restored])


def test_clip_augmented_image_():
    image = np.zeros((1, 3), dtype=np.uint8)
    image[0, 0] = 10
    image[0, 1] = 20
    image[0, 2] = 30
    image_clipped = iaa.clip_augmented_image_(image, min_value=15, max_value=25)
    assert image_clipped[0, 0] == 15
    assert image_clipped[0, 1] == 20
    assert image_clipped[0, 2] == 25


def test_clip_augmented_image():
    image = np.zeros((1, 3), dtype=np.uint8)
    image[0, 0] = 10
    image[0, 1] = 20
    image[0, 2] = 30
    image_clipped = iaa.clip_augmented_image(image, min_value=15, max_value=25)
    assert image_clipped[0, 0] == 15
    assert image_clipped[0, 1] == 20
    assert image_clipped[0, 2] == 25


def test_clip_augmented_images_():
    images = np.zeros((2, 1, 3), dtype=np.uint8)
    images[:, 0, 0] = 10
    images[:, 0, 1] = 20
    images[:, 0, 2] = 30
    images_clipped = iaa.clip_augmented_images_(images, min_value=15, max_value=25)
    assert np.all(images_clipped[:, 0, 0] == 15)
    assert np.all(images_clipped[:, 0, 1] == 20)
    assert np.all(images_clipped[:, 0, 2] == 25)

    images = [np.zeros((1, 3), dtype=np.uint8) for _ in sm.xrange(2)]
    for i in sm.xrange(len(images)):
        images[i][0, 0] = 10
        images[i][0, 1] = 20
        images[i][0, 2] = 30
    images_clipped = iaa.clip_augmented_images_(images, min_value=15, max_value=25)
    assert isinstance(images_clipped, list)
    assert all([images_clipped[i][0, 0] == 15 for i in sm.xrange(len(images))])
    assert all([images_clipped[i][0, 1] == 20 for i in sm.xrange(len(images))])
    assert all([images_clipped[i][0, 2] == 25 for i in sm.xrange(len(images))])


def test_clip_augmented_images():
    images = np.zeros((2, 1, 3), dtype=np.uint8)
    images[:, 0, 0] = 10
    images[:, 0, 1] = 20
    images[:, 0, 2] = 30
    images_clipped = iaa.clip_augmented_images(images, min_value=15, max_value=25)
    assert np.all(images_clipped[:, 0, 0] == 15)
    assert np.all(images_clipped[:, 0, 1] == 20)
    assert np.all(images_clipped[:, 0, 2] == 25)

    images = [np.zeros((1, 3), dtype=np.uint8) for _ in sm.xrange(2)]
    for i in sm.xrange(len(images)):
        images[i][0, 0] = 10
        images[i][0, 1] = 20
        images[i][0, 2] = 30
    images_clipped = iaa.clip_augmented_images(images, min_value=15, max_value=25)
    assert isinstance(images_clipped, list)
    assert all([images_clipped[i][0, 0] == 15 for i in sm.xrange(len(images))])
    assert all([images_clipped[i][0, 1] == 20 for i in sm.xrange(len(images))])
    assert all([images_clipped[i][0, 2] == 25 for i in sm.xrange(len(images))])


def test_reduce_to_nonempty():
    kpsois = [
        ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(4, 4, 3)),
        ia.KeypointsOnImage([ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=0)], shape=(4, 4, 3)),
        ia.KeypointsOnImage([], shape=(4, 4, 3)),
        ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(4, 4, 3)),
        ia.KeypointsOnImage([], shape=(4, 4, 3))
    ]

    kpsois_reduced, ids = iaa.reduce_to_nonempty(kpsois)
    assert kpsois_reduced == [kpsois[0], kpsois[1], kpsois[3]]
    assert ids == [0, 1, 3]

    kpsois = [
        ia.KeypointsOnImage([], shape=(4, 4, 3)),
        ia.KeypointsOnImage([], shape=(4, 4, 3))
    ]

    kpsois_reduced, ids = iaa.reduce_to_nonempty(kpsois)
    assert kpsois_reduced == []
    assert ids == []

    kpsois = [
        ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(4, 4, 3))
    ]

    kpsois_reduced, ids = iaa.reduce_to_nonempty(kpsois)
    assert kpsois_reduced == [kpsois[0]]
    assert ids == [0]

    kpsois = []

    kpsois_reduced, ids = iaa.reduce_to_nonempty(kpsois)
    assert kpsois_reduced == []
    assert ids == []


def test_invert_reduce_to_nonempty():
    kpsois = [
        ia.KeypointsOnImage([ia.Keypoint(x=0, y=1)], shape=(4, 4, 3)),
        ia.KeypointsOnImage([ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=0)], shape=(4, 4, 3)),
        ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(4, 4, 3)),
    ]

    kpsois_recovered = iaa.invert_reduce_to_nonempty(kpsois, [0, 1, 2], ["foo1", "foo2", "foo3"])
    assert kpsois_recovered == ["foo1", "foo2", "foo3"]

    kpsois_recovered = iaa.invert_reduce_to_nonempty(kpsois, [1], ["foo1"])
    assert all([isinstance(kpsoi, ia.KeypointsOnImage) for kpsoi in kpsois]) # assert original list not changed
    assert kpsois_recovered == [kpsois[0], "foo1", kpsois[2]]

    kpsois_recovered = iaa.invert_reduce_to_nonempty(kpsois, [], [])
    assert kpsois_recovered == [kpsois[0], kpsois[1], kpsois[2]]

    kpsois_recovered = iaa.invert_reduce_to_nonempty([], [], [])
    assert kpsois_recovered == []


def test_Augmenter():
    reseed()

    class DummyAugmenter(iaa.Augmenter):
        def _augment_images(self, images, random_state, parents, hooks):
            return images
        def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
            return heatmaps
        def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
            return keypoints_on_images
        def get_parameters(self):
            return []

    # --------
    # __init__
    # --------
    # TODO incomplete tests, handle only cases that were missing in code coverage report
    aug = DummyAugmenter()
    assert aug.random_state == ia.CURRENT_RANDOM_STATE
    aug = DummyAugmenter(deterministic=True)
    assert aug.random_state != ia.CURRENT_RANDOM_STATE
    rs = np.random.RandomState(123)
    aug = DummyAugmenter(random_state=rs)
    assert aug.random_state == rs
    aug = DummyAugmenter(random_state=123)
    assert aug.random_state.randint(0, 10**6) == np.random.RandomState(123).randint(0, 10**6)

    # --------
    # augment_batches
    # --------
    # TODO incomplete tests, handle only cases that were missing in code coverage report
    aug = DummyAugmenter()
    batches_aug = list(aug.augment_batches([[]]))
    assert isinstance(batches_aug, list)
    assert len(batches_aug) == 1
    assert isinstance(batches_aug[0], list)

    aug = DummyAugmenter()
    image_batches = [np.zeros((1, 2, 2, 3), dtype=np.uint8)]
    batches_aug = list(aug.augment_batches(image_batches))
    assert isinstance(batches_aug, list)
    assert len(batches_aug) == 1
    assert array_equal_lists(batches_aug, image_batches)

    aug = DummyAugmenter()
    image_batches = [[np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 3, 3))]]
    batches_aug = list(aug.augment_batches(image_batches))
    assert isinstance(batches_aug, list)
    assert len(batches_aug) == 1
    assert array_equal_lists(batches_aug[0], image_batches[0])

    aug = DummyAugmenter()
    got_exception = False
    try:
        batches_aug = list(aug.augment_batches(None))
    except Exception:
        got_exception = True
    assert got_exception

    aug = DummyAugmenter()
    got_exception = False
    try:
        batches_aug = list(aug.augment_batches([None]))
    except Exception as exc:
        got_exception = True
        assert "Unknown datatype of batch" in str(exc)
    assert got_exception

    aug = DummyAugmenter()
    got_exception = False
    try:
        batches_aug = list(aug.augment_batches([[None]]))
    except Exception as exc:
        got_exception = True
        assert "Unknown datatype in batch[0]" in str(exc)
    assert got_exception

    # --------
    # augment_images
    # --------
    # TODO incomplete tests, handle only cases that were missing in code coverage report
    aug = DummyAugmenter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        images_aug = aug.augment_images(np.zeros((16, 32, 3), dtype=np.uint8))
        # Verify some things
        assert len(caught_warnings) == 1
        assert "indicates that you provided a single image with shape (H, W, C)" in str(caught_warnings[-1].message)

    aug = DummyAugmenter()
    got_exception = False
    try:
        images_aug = aug.augment_images(None)
    except Exception:
        got_exception = True
    assert got_exception

    # behaviour when getting arrays as input and lists as output of augmenter
    aug = iaa.Crop(((1, 8), (1, 8), (1, 8), (1, 8)), keep_size=False)
    images = np.zeros((16, 64, 64, 3), dtype=np.uint8)
    seen = [0, 0]
    for _ in sm.xrange(20):
        observed = aug.augment_images(images)
        if ia.is_np_array(observed):
            seen[0] += 1
        else:
            seen[1] += 1
        assert all([image.ndim == 3 and 48 <= image.shape[0] <= 62 and 48 <= image.shape[1] <= 62 and image.shape[2] == 3 for image in observed])
    assert seen[0] <= 3
    assert seen[1] >= 17

    # same as above but image's channel axis is now 1
    aug = iaa.Crop(((1, 8), (1, 8), (1, 8), (1, 8)), keep_size=False)
    images = np.zeros((16, 64, 64, 1), dtype=np.uint8)
    seen = [0, 0]
    for _ in sm.xrange(20):
        observed = aug.augment_images(images)
        if ia.is_np_array(observed):
            seen[0] += 1
        else:
            seen[1] += 1
        assert all([image.ndim == 3 and 48 <= image.shape[0] <= 62 and 48 <= image.shape[1] <= 62  and image.shape[2] == 1 for image in observed])
    assert seen[0] <= 3
    assert seen[1] >= 17

    # same as above but now with 2D images
    aug = iaa.Crop(((1, 8), (1, 8), (1, 8), (1, 8)), keep_size=False)
    images = np.zeros((16, 64, 64), dtype=np.uint8)
    seen = [0, 0]
    for _ in sm.xrange(20):
        observed = aug.augment_images(images)
        if ia.is_np_array(observed):
            seen[0] += 1
        else:
            seen[1] += 1
        assert all([image.ndim == 2 and 48 <= image.shape[0] <= 62 and 48 <= image.shape[1] <= 62 for image in observed])
    assert seen[0] <= 3
    assert seen[1] >= 17

    # same as above but image's channel axis now varies between [None, 1, 3, 4, 9]
    aug = iaa.Crop(((1, 8), (1, 8), (1, 8), (1, 8)), keep_size=False)
    seen = [0, 0]
    for _ in sm.xrange(20):
        channels = np.random.choice([None, 1, 3, 4, 9], size=(16,))
        images = [np.zeros((64, 64), dtype=np.uint8) if c is None else np.zeros((64, 64, c), dtype=np.uint8) for c in channels]

        observed = aug.augment_images(images)
        if ia.is_np_array(observed):
            seen[0] += 1
        else:
            seen[1] += 1

        for image, c in zip(observed, channels):
            if c is None:
                assert image.ndim == 2
            else:
                assert image.ndim == 3
                assert image.shape[2] == c
            assert 48 <= image.shape[0] <= 62
            assert 48 <= image.shape[1] <= 62
    assert seen[0] == 0
    assert seen[1] == 20

    # --------
    # _augment_images
    # --------
    # TODO incomplete tests, handle only cases that were missing in code coverage report
    class DummyAugmenterCallsParent(iaa.Augmenter):
        def _augment_images(self, images, random_state, parents, hooks):
            return super(DummyAugmenterCallsParent, self)._augment_images(images, random_state, parents, hooks)
        def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
            return super(DummyAugmenterCallsParent, self)._augment_heatmaps(heatmaps, random_state, parents, hooks)
        def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
            return super(DummyAugmenterCallsParent, self)._augment_keypoints(keypoints_on_images, random_state, parents, hooks)
        def get_parameters(self):
            return super(DummyAugmenterCallsParent, self).get_parameters()
    aug = DummyAugmenterCallsParent()
    got_exception = False
    try:
        images_aug = aug.augment_images(np.zeros((2, 4, 4, 3), dtype=np.uint8))
    except NotImplementedError:
        got_exception = True
    assert got_exception

    # --------
    # _augment_heatmaps
    # --------
    # TODO incomplete tests, handle only cases that were missing in code coverage report
    heatmaps = ia.HeatmapsOnImage(np.zeros((3, 3, 1), dtype=np.float32), shape=(3, 3, 3))
    got_exception = False
    try:
        heatmaps_aug = aug.augment_heatmaps([heatmaps])
    except NotImplementedError:
        got_exception = True
    assert got_exception

    # --------
    # _augment_keypoints
    # --------
    # TODO incomplete tests, handle only cases that were missing in code coverage report
    aug = DummyAugmenterCallsParent()
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=0),
                                      ia.Keypoint(x=2, y=1)], shape=(4, 4, 3))]
    got_exception = False
    try:
        keypoints_aug = aug.augment_keypoints(keypoints)
    except NotImplementedError:
        got_exception = True
    assert got_exception

    # --------
    # augment_bounding_boxes
    # --------
    class DummyAugmenterBBs(iaa.Augmenter):
        def _augment_images(self, images, random_state, parents, hooks):
            return images
        def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
            return heatmaps
        def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
            return [keypoints_on_images_i.shift(x=1) for keypoints_on_images_i in keypoints_on_images]
        def get_parameters(self):
            return []
    aug = DummyAugmenterBBs()
    bb = ia.BoundingBox(x1=1, y1=4, x2=2, y2=5)
    bbs = [bb]
    bbsois = [ia.BoundingBoxesOnImage(bbs, shape=(10, 10, 3))]
    bbsois_aug = aug.augment_bounding_boxes(bbsois)
    bb_aug = bbsois_aug[0].bounding_boxes[0]
    assert bb_aug.x1 == 1+1
    assert bb_aug.y1 == 4
    assert bb_aug.x2 == 2+1
    assert bb_aug.y2 == 5

    # empty list of BBs
    bbsois = [ia.BoundingBoxesOnImage([], shape=(10, 10, 3))]
    bbsois_aug = aug.augment_bounding_boxes(bbsois)
    assert len(bbsois_aug) == 1
    assert bbsois_aug[0].bounding_boxes == []

    bbsois_aug = aug.augment_bounding_boxes([])
    assert bbsois_aug == []

    # --------
    # draw_grid
    # --------
    aug = DummyAugmenter()
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[0, 0, :] = 10
    image[0, 1, :] = 50
    image[1, 1, :] = 255

    # list, shape (3, 3, 3)
    grid = aug.draw_grid([image], rows=2, cols=2)
    grid_expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert np.array_equal(grid, grid_expected)

    # list, shape (3, 3)
    grid = aug.draw_grid([image[..., 0]], rows=2, cols=2)
    grid_expected = np.vstack([
        np.hstack([image[..., 0:1], image[..., 0:1]]),
        np.hstack([image[..., 0:1], image[..., 0:1]])
    ])
    grid_expected = np.tile(grid_expected, (1, 1, 3))
    assert np.array_equal(grid, grid_expected)

    # list, shape (2,)
    got_exception = False
    try:
        grid = aug.draw_grid([np.zeros((2,), dtype=np.uint8)], rows=2, cols=2)
    except Exception:
        got_exception = True
    assert got_exception

    # array, shape (1, 3, 3, 3)
    grid = aug.draw_grid(np.uint8([image]), rows=2, cols=2)
    grid_expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert np.array_equal(grid, grid_expected)

    # array, shape (3, 3, 3)
    grid = aug.draw_grid(image, rows=2, cols=2)
    grid_expected = np.vstack([
        np.hstack([image, image]),
        np.hstack([image, image])
    ])
    assert np.array_equal(grid, grid_expected)

    # array, shape (3, 3)
    grid = aug.draw_grid(image[..., 0], rows=2, cols=2)
    grid_expected = np.vstack([
        np.hstack([image[..., 0:1], image[..., 0:1]]),
        np.hstack([image[..., 0:1], image[..., 0:1]])
    ])
    grid_expected = np.tile(grid_expected, (1, 1, 3))
    assert np.array_equal(grid, grid_expected)

    # array, shape (2,)
    got_exception = False
    try:
        grid = aug.draw_grid(np.zeros((2,), dtype=np.uint8), rows=2, cols=2)
    except Exception:
        got_exception = True
    assert got_exception

    # --------
    # localize_random_state
    # --------
    aug = DummyAugmenter()
    assert aug.random_state == ia.CURRENT_RANDOM_STATE
    aug_localized = aug.localize_random_state()
    assert aug_localized.random_state != ia.CURRENT_RANDOM_STATE

    # --------
    # reseed
    # --------
    def _same_rs(rs1, rs2):
        rs1_copy = copy.deepcopy(rs1)
        rs2_copy = copy.deepcopy(rs2)
        rnd1 = rs1_copy.randint(0, 10**6)
        rnd2 = rs2_copy.randint(0, 10**6)
        return rnd1 == rnd2

    aug1 = DummyAugmenter()
    aug2 = DummyAugmenter(deterministic=True)
    aug0 = iaa.Sequential([aug1, aug2])

    aug0_copy = aug0.deepcopy()
    assert _same_rs(aug0.random_state, aug0_copy.random_state)
    assert _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
    assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)
    aug0_copy.reseed()
    assert not _same_rs(aug0.random_state, aug0_copy.random_state)
    assert not _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
    assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)

    aug0_copy = aug0.deepcopy()
    assert _same_rs(aug0.random_state, aug0_copy.random_state)
    assert _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
    assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)
    aug0_copy.reseed(deterministic_too=True)
    assert not _same_rs(aug0.random_state, aug0_copy.random_state)
    assert not _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
    assert not _same_rs(aug0[1].random_state, aug0_copy[1].random_state)

    aug0_copy = aug0.deepcopy()
    assert _same_rs(aug0.random_state, aug0_copy.random_state)
    assert _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
    assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)
    aug0_copy.reseed(random_state=123)
    assert not _same_rs(aug0.random_state, aug0_copy.random_state)
    assert not _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
    assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)
    assert aug0_copy.random_state.randint(0, 10**6) == np.random.RandomState(np.random.RandomState(123).randint(0, 10**6)).randint(0, 10**6)

    aug0_copy = aug0.deepcopy()
    assert _same_rs(aug0.random_state, aug0_copy.random_state)
    assert _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
    assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)
    aug0_copy.reseed(random_state=np.random.RandomState(123))
    assert not _same_rs(aug0.random_state, aug0_copy.random_state)
    assert not _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
    assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)
    assert aug0_copy.random_state.randint(0, 10**6) == np.random.RandomState(np.random.RandomState(123).randint(0, 10**6)).randint(0, 10**6)

    # --------
    # get_parameters
    # --------
    aug = DummyAugmenterCallsParent()
    got_exception = False
    try:
        aug.get_parameters()
    except NotImplementedError:
        got_exception = True
    assert got_exception

    # --------
    # get_all_children
    # --------
    aug1 = DummyAugmenter()
    aug21 = DummyAugmenter()
    aug2 = iaa.Sequential([aug21])
    aug0 = iaa.Sequential([aug1, aug2])
    children = aug0.get_all_children(flat=True)
    assert isinstance(children, list)
    assert children[0] == aug1
    assert children[1] == aug2
    assert children[2] == aug21
    children = aug0.get_all_children(flat=False)
    assert isinstance(children, list)
    assert children[0] == aug1
    assert children[1] == aug2
    assert isinstance(children[2], list)
    assert children[2][0] == aug21

    # --------
    # __repr__, __str__
    # --------
    class DummyAugmenterRepr(iaa.Augmenter):
        def _augment_images(self, images, random_state, parents, hooks):
            return images
        def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
            return heatmaps
        def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
            return keypoints_on_images
        def get_parameters(self):
            return ["A", "B", "C"]
    aug = DummyAugmenterRepr(name="Example")
    assert aug.__repr__() == aug.__str__() == "DummyAugmenterRepr(name=Example, parameters=[A, B, C], deterministic=False)"
    aug = DummyAugmenterRepr(name="Example", deterministic=True)
    assert aug.__repr__() == aug.__str__() == "DummyAugmenterRepr(name=Example, parameters=[A, B, C], deterministic=True)"


def test_Augmenter_augment_keypoints():
    # most stuff was already tested in other tests, so not tested here again
    reseed()

    # test empty KeypointsOnImage objects
    kpsoi1 = ia.KeypointsOnImage([], shape=(32, 32, 3))
    kpsoi2 = ia.KeypointsOnImage([ia.Keypoint(10, 10)], shape=(32, 32, 3))

    aug = iaa.Affine(translate_px={"x": 1})
    kpsoi_aug = aug.augment_keypoints([kpsoi1, kpsoi2])
    assert len(kpsoi_aug) == 2
    assert len(kpsoi_aug[0].keypoints) == 0
    assert len(kpsoi_aug[1].keypoints) == 1
    assert kpsoi_aug[1].keypoints[0].x == 11

    kpsoi1 = ia.KeypointsOnImage([], shape=(32, 32, 3))
    kpsoi2 = ia.KeypointsOnImage([], shape=(32, 32, 3))

    aug = iaa.Affine(translate_px={"x": 1})
    kpsoi_aug = aug.augment_keypoints([kpsoi1, kpsoi2])
    assert len(kpsoi_aug) == 2
    assert len(kpsoi_aug[0].keypoints) == 0
    assert len(kpsoi_aug[1].keypoints) == 0


def test_Augmenter_augment_segmentation_maps():
    reseed()

    arr = np.int32([
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3), nb_classes=2)
    segmap_aug = iaa.Noop().augment_segmentation_maps([segmap])[0]
    assert np.allclose(segmap_aug.arr, segmap.arr)
    segmap_aug = iaa.Add(10).augment_segmentation_maps([segmap])[0]
    assert np.allclose(segmap_aug.arr, segmap.arr)

    segmap_aug = iaa.Affine(translate_px={"x":1}).augment_segmentation_maps([segmap])[0]
    expected_c0 = np.float32([
        [0, 1.0, 0],
        [0, 1.0, 0],
        [0, 1.0, 0]
    ])
    expected_c1 = np.float32([
        [0, 0, 1.0],
        [0, 0, 1.0],
        [0, 0, 1.0]
    ])
    expected = np.concatenate([
        expected_c0[..., np.newaxis],
        expected_c1[..., np.newaxis]
    ], axis=2)
    assert np.allclose(segmap_aug.arr, expected)

    segmap_aug = iaa.Pad(px=(1, 0, 0, 0), keep_size=False).augment_segmentation_maps([segmap])[0]
    expected_c0 = np.float32([
        [0.0, 0, 0],
        [1.0, 0, 0],
        [1.0, 0, 0],
        [1.0, 0, 0]
    ])
    expected_c1 = np.float32([
        [0, 0.0, 0.0],
        [0, 1.0, 1.0],
        [0, 1.0, 1.0],
        [0, 1.0, 1.0]
    ])
    expected = np.concatenate([
        expected_c0[..., np.newaxis],
        expected_c1[..., np.newaxis]
    ], axis=2)
    assert np.allclose(segmap_aug.arr, expected)

    # some heatmaps empty
    arr = np.int32([
        [0, 3, 3],
        [0, 3, 3],
        [0, 3, 3]
    ])
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3), nb_classes=4)
    segmap_aug = iaa.Noop().augment_segmentation_maps([segmap])[0]
    assert np.allclose(segmap_aug.arr, segmap.arr)
    segmap_aug = iaa.Add(10).augment_segmentation_maps([segmap])[0]
    assert np.allclose(segmap_aug.arr, segmap.arr)

    segmap_aug = iaa.Affine(translate_px={"x":1}).augment_segmentation_maps([segmap])[0]
    expected_c0 = np.float32([
        [0, 1.0, 0],
        [0, 1.0, 0],
        [0, 1.0, 0]
    ])
    expected_c1 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_c2 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_c3 = np.float32([
        [0, 0, 1.0],
        [0, 0, 1.0],
        [0, 0, 1.0]
    ])
    expected = np.concatenate([
        expected_c0[..., np.newaxis],
        expected_c1[..., np.newaxis],
        expected_c2[..., np.newaxis],
        expected_c3[..., np.newaxis]
    ], axis=2)
    assert np.allclose(segmap_aug.arr, expected)

    segmap_aug = iaa.Pad(px=(1, 0, 0, 0), keep_size=False).augment_segmentation_maps([segmap])[0]
    expected_c0 = np.float32([
        [0.0, 0, 0],
        [1.0, 0, 0],
        [1.0, 0, 0],
        [1.0, 0, 0]
    ])
    expected_c1 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_c2 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_c3 = np.float32([
        [0, 0.0, 0.0],
        [0, 1.0, 1.0],
        [0, 1.0, 1.0],
        [0, 1.0, 1.0]
    ])
    expected = np.concatenate([
        expected_c0[..., np.newaxis],
        expected_c1[..., np.newaxis],
        expected_c2[..., np.newaxis],
        expected_c3[..., np.newaxis]
    ], axis=2)
    assert np.allclose(segmap_aug.arr, expected)

    # all heatmaps empty
    arr = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    arr = np.tile(arr[..., np.newaxis], (1, 1, 4))
    segmap = ia.SegmentationMapOnImage(arr, shape=(3, 3))
    segmap_aug = iaa.Noop().augment_segmentation_maps([segmap])[0]
    assert np.allclose(segmap_aug.arr, segmap.arr)
    segmap_aug = iaa.Add(10).augment_segmentation_maps([segmap])[0]
    assert np.allclose(segmap_aug.arr, segmap.arr)

    segmap_aug = iaa.Affine(translate_px={"x":1}).augment_segmentation_maps([segmap])[0]
    expected_c0 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_c1 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_c2 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_c3 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected = np.concatenate([
        expected_c0[..., np.newaxis],
        expected_c1[..., np.newaxis],
        expected_c2[..., np.newaxis],
        expected_c3[..., np.newaxis]
    ], axis=2)
    assert np.allclose(segmap_aug.arr, expected)

    segmap_aug = iaa.Pad(px=(1, 0, 0, 0), keep_size=False).augment_segmentation_maps([segmap])[0]
    expected_c0 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_c1 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_c2 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected_c3 = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    expected = np.concatenate([
        expected_c0[..., np.newaxis],
        expected_c1[..., np.newaxis],
        expected_c2[..., np.newaxis],
        expected_c3[..., np.newaxis]
    ], axis=2)
    assert np.allclose(segmap_aug.arr, expected)


def test_Augmenter_find():
    reseed()

    noop1 = iaa.Noop(name="Noop")
    fliplr = iaa.Fliplr(name="Fliplr")
    flipud = iaa.Flipud(name="Flipud")
    noop2 = iaa.Noop(name="Noop2")
    seq2 = iaa.Sequential([flipud, noop2], name="Seq2")
    seq1 = iaa.Sequential([noop1, fliplr, seq2], name="Seq")

    augs = seq1.find_augmenters_by_name("Seq")
    assert len(augs) == 1
    assert augs[0] == seq1

    augs = seq1.find_augmenters_by_name("Seq2")
    assert len(augs) == 1
    assert augs[0] == seq2

    augs = seq1.find_augmenters_by_names(["Seq", "Seq2"])
    assert len(augs) == 2
    assert augs[0] == seq1
    assert augs[1] == seq2

    augs = seq1.find_augmenters_by_name(r"Seq.*", regex=True)
    assert len(augs) == 2
    assert augs[0] == seq1
    assert augs[1] == seq2

    augs = seq1.find_augmenters(lambda aug, parents: aug.name in ["Seq", "Seq2"])
    assert len(augs) == 2
    assert augs[0] == seq1
    assert augs[1] == seq2

    augs = seq1.find_augmenters(lambda aug, parents: aug.name in ["Seq", "Seq2"] and len(parents) > 0)
    assert len(augs) == 1
    assert augs[0] == seq2

    augs = seq1.find_augmenters(lambda aug, parents: aug.name in ["Seq", "Seq2"], flat=False)
    assert len(augs) == 2
    assert augs[0] == seq1
    assert augs[1] == [seq2]


def test_Augmenter_remove():
    reseed()

    def get_seq():
        noop1 = iaa.Noop(name="Noop")
        fliplr = iaa.Fliplr(name="Fliplr")
        flipud = iaa.Flipud(name="Flipud")
        noop2 = iaa.Noop(name="Noop2")
        seq2 = iaa.Sequential([flipud, noop2], name="Seq2")
        seq1 = iaa.Sequential([noop1, fliplr, seq2], name="Seq")
        return seq1

    augs = get_seq()
    augs = augs.remove_augmenters(lambda aug, parents: aug.name == "Seq2")
    seqs = augs.find_augmenters_by_name(r"Seq.*", regex=True)
    assert len(seqs) == 1
    assert seqs[0].name == "Seq"

    augs = get_seq()
    augs = augs.remove_augmenters(lambda aug, parents: aug.name == "Seq2" and len(parents) == 0)
    seqs = augs.find_augmenters_by_name(r"Seq.*", regex=True)
    assert len(seqs) == 2
    assert seqs[0].name == "Seq"
    assert seqs[1].name == "Seq2"

    augs = get_seq()
    augs = augs.remove_augmenters(lambda aug, parents: True)
    assert augs is not None
    assert isinstance(augs, iaa.Noop)

    augs = get_seq()
    got_exception = False
    try:
        augs = augs.remove_augmenters(lambda aug, parents: aug.name == "Seq", copy=False)
    except Exception as exc:
        got_exception = True
        assert "Inplace removal of topmost augmenter requested, which is currently not possible" in str(exc)
    assert got_exception

    augs = get_seq()
    augs = augs.remove_augmenters(lambda aug, parents: True, noop_if_topmost=False)
    assert augs is None


def test_Augmenter_hooks():
    # TODO these tests change the input type from list to array. Might be reasnoable to change
    # and test that scenario separetely
    reseed()

    image = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 1, 1]], dtype=np.uint8)
    image_lr = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0]], dtype=np.uint8)
    image_ud = np.array([[0, 1, 1],
                         [0, 0, 1],
                         [0, 0, 1]], dtype=np.uint8)
    image_lrud = np.array([[1, 1, 0],
                           [1, 0, 0],
                           [1, 0, 0]], dtype=np.uint8)
    image = image[:, :, np.newaxis]
    image_lr = image_lr[:, :, np.newaxis]
    image_ud = image_ud[:, :, np.newaxis]
    image_lrud = image_lrud[:, :, np.newaxis]

    seq = iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)])

    # preprocessing
    def preprocessor(images, augmenter, parents):
        img = np.copy(images)
        img[0][1, 1, 0] += 1
        return img
    hooks = ia.HooksImages(preprocessor=preprocessor)
    images_aug = seq.augment_images([image], hooks=hooks)
    expected = np.copy(image_lrud)
    expected[1, 1, 0] = 3
    assert np.array_equal(images_aug[0], expected)

    # postprocessing
    def postprocessor(images, augmenter, parents):
        img = np.copy(images)
        img[0][1, 1, 0] += 1
        return img
    hooks = ia.HooksImages(postprocessor=postprocessor)
    images_aug = seq.augment_images([image], hooks=hooks)
    expected = np.copy(image_lrud)
    expected[1, 1, 0] = 3
    assert np.array_equal(images_aug[0], expected)

    # propagating
    def propagator(images, augmenter, parents, default):
        if "Seq" in augmenter.name:
            return False
        else:
            return default
    hooks = ia.HooksImages(propagator=propagator)
    images_aug = seq.augment_images([image], hooks=hooks)
    assert np.array_equal(images_aug[0], image)

    # activation
    def activator(images, augmenter, parents, default):
        if "Flipud" in augmenter.name:
            return False
        else:
            return default
    hooks = ia.HooksImages(activator=activator)
    images_aug = seq.augment_images([image], hooks=hooks)
    assert np.array_equal(images_aug[0], image_lr)

    # keypoint aug deactivated
    aug = iaa.Affine(translate_px=1)
    def activator(keypoints_on_images, augmenter, parents, default):
        return False
    hooks = ia.HooksKeypoints(activator=activator)
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=0),
                                      ia.Keypoint(x=2, y=1)], shape=image.shape)]
    keypoints_aug = seq.augment_keypoints(keypoints, hooks=hooks)
    assert keypoints_equal(keypoints_aug, keypoints)


def test_Augmenter_copy_random_state():
    image = ia.quokka_square(size=(128, 128))
    images = np.array([image] * 64, dtype=np.uint8)

    source = iaa.Sequential([
        iaa.Fliplr(0.5, name="hflip"),
        iaa.Dropout(0.05, name="dropout"),
        iaa.Affine(translate_px=(-10, 10), name="translate", random_state=3),
        iaa.GaussianBlur(1.0, name="blur", random_state=4)
    ], random_state=5)
    target = iaa.Sequential([
        iaa.Fliplr(0.5, name="hflip"),
        iaa.Dropout(0.05, name="dropout"),
        iaa.Affine(translate_px=(-10, 10), name="translate")
    ])

    source.localize_random_state_()

    target_cprs = target.copy_random_state(source, matching="position")
    source_alt = source.remove_augmenters(lambda aug, parents: aug.name == "blur")
    images_aug_source = source_alt.augment_images(images)
    images_aug_target = target_cprs.augment_images(images)
    #ia.imshow(np.hstack([images_aug_source[0], images_aug_source[1], images_aug_target[0], images_aug_target[1]]))
    assert np.array_equal(images_aug_source, images_aug_target)

    source[0].deterministic = True
    target_cprs = target.copy_random_state(source, matching="position", copy_determinism=True)
    source_alt = source.remove_augmenters(lambda aug, parents: aug.name == "blur")
    images_aug_source = source_alt.augment_images(images)
    images_aug_target = target_cprs.augment_images(images)
    assert target_cprs[0].deterministic == True
    assert np.array_equal(images_aug_source, images_aug_target)

    source[0].deterministic = False
    target[0].deterministic = False

    target_cprs = target.copy_random_state(source, matching="name")
    source_alt = source.remove_augmenters(lambda aug, parents: aug.name == "blur")
    images_aug_source = source_alt.augment_images(images)
    images_aug_target = target_cprs.augment_images(images)
    assert np.array_equal(images_aug_source, images_aug_target)

    source_alt = source.remove_augmenters(lambda aug, parents: aug.name == "blur")
    source_det = source_alt.to_deterministic()
    target_cprs_det = target.copy_random_state(source_det, matching="name",
                                               copy_determinism=True)
    images_aug_source1 = source_det.augment_images(images)
    images_aug_target1 = target_cprs_det.augment_images(images)
    images_aug_source2 = source_det.augment_images(images)
    images_aug_target2 = target_cprs_det.augment_images(images)
    assert np.array_equal(images_aug_source1, images_aug_source2)
    assert np.array_equal(images_aug_target1, images_aug_target2)
    assert np.array_equal(images_aug_source1, images_aug_target1)
    assert np.array_equal(images_aug_source2, images_aug_target2)

    source = iaa.Fliplr(0.5, name="hflip")
    target = iaa.Fliplr(0.5, name="hflip")
    got_exception = False
    try:
        target_cprs = target.copy_random_state(source, matching="name")
    except Exception as exc:
        got_exception = True
        assert "localize_random_state" in str(exc)
    assert got_exception

    source = iaa.Fliplr(0.5, name="hflip-other-name")
    target = iaa.Fliplr(0.5, name="hflip")
    source.localize_random_state_()
    got_exception = False
    try:
        target_cprs = target.copy_random_state(source, matching="name", matching_tolerant=False)
    except Exception as exc:
        got_exception = True
        assert "not found among source augmenters" in str(exc)
    assert got_exception

    source = iaa.Fliplr(0.5, name="hflip")
    target = iaa.Fliplr(0.5, name="hflip")
    got_exception = False
    try:
        target_cprs = target.copy_random_state(source, matching="position")
    except Exception as exc:
        got_exception = True
        assert "localize_random_state" in str(exc)
    assert got_exception

    source = iaa.Sequential([iaa.Fliplr(0.5, name="hflip"), iaa.Fliplr(0.5, name="hflip2")])
    target = iaa.Sequential([iaa.Fliplr(0.5, name="hflip")])
    source.localize_random_state_()
    got_exception = False
    try:
        target_cprs = target.copy_random_state(source, matching="position", matching_tolerant=False)
    except Exception as exc:
        got_exception = True
        assert "different lengths" in str(exc)
    assert got_exception

    source = iaa.Sequential([iaa.Fliplr(0.5, name="hflip"), iaa.Fliplr(0.5, name="hflip2")])
    target = iaa.Sequential([iaa.Fliplr(0.5, name="hflip")])
    source.localize_random_state_()
    got_exception = False
    try:
        target_cprs = target.copy_random_state(source, matching="test")
    except Exception as exc:
        got_exception = True
        assert "Unknown matching method" in str(exc)
    assert got_exception

    source = iaa.Sequential([iaa.Fliplr(0.5, name="hflip"), iaa.Fliplr(0.5, name="hflip")])
    target = iaa.Sequential([iaa.Fliplr(0.5, name="hflip")])
    source.localize_random_state_()
    with warnings.catch_warnings(record=True) as caught_warnings:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        target_cprs = target.copy_random_state(source, matching="name")
        # Verify some things
        assert len(caught_warnings) == 1
        assert "contains multiple augmenters with the same name" in str(caught_warnings[-1].message)


def test_Sequential():
    reseed()

    image = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 1]], dtype=np.uint8) * 255
    image = image[:, :, np.newaxis]
    images_list = [image]
    images = np.array([image])

    image_lr = np.array([[1, 1, 0],
                         [1, 0, 0],
                         [1, 0, 0]], dtype=np.uint8) * 255
    image_lr = image_lr[:, :, np.newaxis]
    images_lr = np.array([image_lr])

    image_ud = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8) * 255
    image_ud = image_ud[:, :, np.newaxis]
    images_ud = np.array([image_ud])

    image_lr_ud = np.array([[1, 0, 0],
                            [1, 0, 0],
                            [1, 1, 0]], dtype=np.uint8) * 255
    image_lr_ud = image_lr_ud[:, :, np.newaxis]
    images_lr_ud_list = [image_lr_ud]
    images_lr_ud = np.array([image_lr_ud])

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=0),
                                      ia.Keypoint(x=2, y=1)], shape=image.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=0, y=2),
                                          ia.Keypoint(x=0, y=1)], shape=image.shape)]

    aug = iaa.Sequential([
        iaa.Fliplr(1.0),
        iaa.Flipud(1.0)
    ])
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_lr_ud)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_lr_ud)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_lr_ud_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_lr_ud_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # heatmaps
    heatmaps_arr = np.float32([[0, 0, 1.0],
                               [0, 0, 1.0],
                               [0, 1.0, 1.0]])
    heatmaps_arr_expected = np.float32([[1.0, 1.0, 0.0],
                                        [1.0, 0, 0],
                                        [1.0, 0, 0]])
    observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1.0 - 1e-6 < observed.max_value < 1.0 + 1e-6
    assert np.array_equal(observed.get_arr(), heatmaps_arr_expected)

    # 50% horizontal flip, 50% vertical flip
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 200
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        assert np.array_equal(observed_aug, images) \
               or np.array_equal(observed_aug, images_lr) \
               or np.array_equal(observed_aug, images_ud) \
               or np.array_equal(observed_aug, images_lr_ud)
        assert np.array_equal(observed_aug_det, images) \
               or np.array_equal(observed_aug_det, images_lr) \
               or np.array_equal(observed_aug_det, images_ud) \
               or np.array_equal(observed_aug_det, images_lr_ud)

    assert (0.25 - 0.10) <= (1 - (nb_changed_aug / nb_iterations)) <= (0.25 + 0.10) # should be the same in roughly 25% of all cases
    assert nb_changed_aug_det == 0

    # random order
    image = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 1]], dtype=np.uint8)
    image = image[:, :, np.newaxis]
    images = np.array([image])

    images_first_second = (images + 10) * 10
    images_second_first = (images * 10) + 10

    heatmaps_arr = np.float32([[0.0, 0.5, 0.5],
                               [0.0, 0.0, 0.5],
                               [0.0, 0.0, 0.5]])
    heatmaps_arr_first_second = (heatmaps_arr + 0.1) * 0.5
    heatmaps_arr_second_first = (heatmaps_arr * 0.5) + 0.1
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=image.shape)]
    keypoints_first_second = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=image.shape)]
    keypoints_second_first = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0)], shape=image.shape)]

    def images_first(images, random_state, parents, hooks):
        return images + 10

    def images_second(images, random_state, parents, hooks):
        return images * 10

    def heatmaps_first(heatmaps, random_state, parents, hooks):
        for heatmaps_i in heatmaps:
            heatmaps_i.arr_0to1 += 0.1
        return heatmaps

    def heatmaps_second(heatmaps, random_state, parents, hooks):
        for heatmaps_i in heatmaps:
            heatmaps_i.arr_0to1 *= 0.5
        return heatmaps

    def keypoints_first(keypoints_on_images, random_state, parents, hooks):
        for keypoints_on_image in keypoints_on_images:
            for keypoint in keypoints_on_image.keypoints:
                keypoint.x = keypoint.x + 1
        return keypoints_on_images

    def keypoints_second(keypoints_on_images, random_state, parents, hooks):
        for keypoints_on_image in keypoints_on_images:
            for keypoint in keypoints_on_image.keypoints:
                keypoint.y = keypoint.y + keypoint.x
        return keypoints_on_images

    aug_unrandom = iaa.Sequential([
        iaa.Lambda(images_first, heatmaps_first, keypoints_first),
        iaa.Lambda(images_second, heatmaps_second, keypoints_second)
    ], random_order=False)
    aug_unrandom_det = aug_unrandom.to_deterministic()
    aug_random = iaa.Sequential([
        iaa.Lambda(images_first, heatmaps_first, keypoints_first),
        iaa.Lambda(images_second, heatmaps_second, keypoints_second)
    ], random_order=True)
    aug_random_det = aug_random.to_deterministic()

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 200

    nb_images_first_second_unrandom = 0
    nb_images_second_first_unrandom = 0
    nb_images_first_second_random = 0
    nb_images_second_first_random = 0
    nb_heatmaps_first_second_unrandom = 0
    nb_heatmaps_second_first_unrandom = 0
    nb_heatmaps_first_second_random = 0
    nb_heatmaps_second_first_random = 0
    nb_keypoints_first_second_unrandom = 0
    nb_keypoints_second_first_unrandom = 0
    nb_keypoints_first_second_random = 0
    nb_keypoints_second_first_random = 0

    for i in sm.xrange(nb_iterations):
        observed_aug_unrandom = aug_unrandom.augment_images(images)
        observed_aug_unrandom_det = aug_unrandom_det.augment_images(images)
        observed_aug_random = aug_random.augment_images(images)
        observed_aug_random_det = aug_random_det.augment_images(images)

        heatmaps_aug_unrandom = aug_unrandom.augment_heatmaps([heatmaps])[0]
        heatmaps_aug_random = aug_random.augment_heatmaps([heatmaps])[0]

        keypoints_aug_unrandom = aug_unrandom.augment_keypoints(keypoints)
        keypoints_aug_random = aug_random.augment_keypoints(keypoints)

        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        if np.array_equal(observed_aug_unrandom, images_first_second):
            nb_images_first_second_unrandom += 1
        elif np.array_equal(observed_aug_unrandom, images_second_first):
            nb_images_second_first_unrandom += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if np.array_equal(observed_aug_random, images_first_second):
            nb_images_first_second_random += 1
        elif np.array_equal(observed_aug_random, images_second_first):
            nb_images_second_first_random += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if np.allclose(heatmaps_aug_unrandom.get_arr(), heatmaps_arr_first_second):
            nb_heatmaps_first_second_unrandom += 1
        elif np.allclose(heatmaps_aug_unrandom.get_arr(), heatmaps_arr_second_first):
            nb_heatmaps_second_first_unrandom += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if np.allclose(heatmaps_aug_random.get_arr(), heatmaps_arr_first_second):
            nb_heatmaps_first_second_random += 1
        elif np.allclose(heatmaps_aug_random.get_arr(), heatmaps_arr_second_first):
            nb_heatmaps_second_first_random += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if keypoints_equal(keypoints_aug_unrandom, keypoints_first_second):
            nb_keypoints_first_second_unrandom += 1
        elif keypoints_equal(keypoints_aug_unrandom, keypoints_second_first):
            nb_keypoints_second_first_unrandom += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if keypoints_equal(keypoints_aug_random, keypoints_first_second):
            nb_keypoints_first_second_random += 1
        elif keypoints_equal(keypoints_aug_random, keypoints_second_first):
            nb_keypoints_second_first_random += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

    assert nb_changed_aug == 0
    assert nb_changed_aug_det == 0
    assert nb_images_first_second_unrandom == nb_iterations
    assert nb_images_second_first_unrandom == 0
    assert nb_heatmaps_first_second_unrandom == nb_iterations
    assert nb_heatmaps_second_first_unrandom == 0
    assert nb_keypoints_first_second_unrandom == nb_iterations
    assert nb_keypoints_second_first_unrandom == 0
    assert (0.50 - 0.1) <= nb_images_first_second_random / nb_iterations <= (0.50 + 0.1)
    assert (0.50 - 0.1) <= nb_images_second_first_random / nb_iterations <= (0.50 + 0.1)
    assert (0.50 - 0.1) <= nb_keypoints_first_second_random / nb_iterations <= (0.50 + 0.1)
    assert (0.50 - 0.1) <= nb_keypoints_second_first_random / nb_iterations <= (0.50 + 0.1)

    # random order for heatmaps
    # TODO this is now already tested above via lamdba functions?
    aug = iaa.Sequential([
        iaa.Affine(translate_px={"x": 1}),
        iaa.Fliplr(1.0)
    ], random_order=True)
    heatmaps_arr = np.float32([[0, 0, 1.0],
                               [0, 0, 1.0],
                               [0, 1.0, 1.0]])
    heatmaps_arr_expected1 = np.float32([[0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [1.0, 0.0, 0.0]])
    heatmaps_arr_expected2 = np.float32([[0.0, 1.0, 0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 1.0, 1.0]])
    seen = [False, False]
    for _ in sm.xrange(100):
        observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))])[0]
        if np.allclose(observed.get_arr(), heatmaps_arr_expected1):
            seen[0] = True
        elif np.allclose(observed.get_arr(), heatmaps_arr_expected2):
            seen[1] = True
        else:
            assert False
        if all(seen):
            break
    assert all(seen)

    # None as children
    aug = iaa.Sequential(children=None)
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, image)

    aug = iaa.Sequential()
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, image)

    # Single child
    aug = iaa.Sequential(iaa.Fliplr(1.0))
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, np.fliplr(image))

    # Sequential of Sequential
    aug = iaa.Sequential(iaa.Sequential(iaa.Fliplr(1.0)))
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, np.fliplr(image))

    # Sequential of list of Sequentials
    aug = iaa.Sequential([iaa.Sequential(iaa.Flipud(1.0)), iaa.Sequential(iaa.Fliplr(1.0))])
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, np.fliplr(np.flipud(image)))

    # add
    aug = iaa.Sequential()
    aug.add(iaa.Fliplr(1.0))
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, np.fliplr(image))

    aug = iaa.Sequential(iaa.Fliplr(1.0))
    aug.add(iaa.Flipud(1.0))
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, np.fliplr(np.flipud(image)))

    # get_parameters
    aug = iaa.Sequential(iaa.Fliplr(1.0), random_order=False)
    assert aug.get_parameters() == [False]

    aug = iaa.Sequential(iaa.Fliplr(1.0), random_order=True)
    assert aug.get_parameters() == [True]

    # get_children_lists
    flip = iaa.Fliplr(1.0)
    aug = iaa.Sequential(flip)
    assert aug.get_children_lists() == [aug]

    # str/repr
    flip = iaa.Fliplr(1.0)
    aug = iaa.Sequential(flip, random_order=True)
    expected = "Sequential(name=%s, random_order=%s, children=[%s], deterministic=%s)" % (aug.name, "True", str(flip), "False")
    assert aug.__str__() == aug.__repr__() == expected


def test_SomeOf():
    reseed()

    zeros = np.zeros((3, 3, 1), dtype=np.uint8)

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=0),
                                      ia.Keypoint(x=2, y=1)], shape=zeros.shape)]

    # no child augmenters
    observed = iaa.SomeOf(n=0, children=[]).augment_image(zeros)
    assert np.array_equal(observed, zeros)

    observed = iaa.SomeOf(n=0).augment_image(zeros)
    assert np.array_equal(observed, zeros)

    # up to three child augmenters
    augs = [iaa.Add(1), iaa.Add(2), iaa.Add(3)]

    observed = iaa.SomeOf(n=0, children=augs).augment_image(zeros)
    assert np.array_equal(observed, zeros)

    observed = iaa.SomeOf(n=1, children=augs).augment_image(zeros)
    assert np.sum(observed) in [9*1, 9*2, 9*3]

    observed = iaa.SomeOf(n=2, children=augs).augment_image(zeros)
    assert np.sum(observed) in [9*1+9*2, 9*1+9*3, 9*2+9*3]

    observed = iaa.SomeOf(n=3, children=augs).augment_image(zeros)
    assert np.sum(observed) in [9*1+9*2+9*3]

    observed = iaa.SomeOf(n=4, children=augs).augment_image(zeros)
    assert np.sum(observed) in [9*1+9*2+9*3]

    # basic heatmaps test
    augs = [iaa.Affine(translate_px={"x":1}), iaa.Affine(translate_px={"x":1}), iaa.Affine(translate_px={"x":1})]
    heatmaps_arr = np.float32([[1.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0]])
    heatmaps_arr0 = np.float32([[1.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0]])
    heatmaps_arr1 = np.float32([[0.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0]])
    heatmaps_arr2 = np.float32([[0.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0]])
    heatmaps_arr3 = np.float32([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]])
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))
    observed0 = iaa.SomeOf(n=0, children=augs).augment_heatmaps([heatmaps])[0]
    observed1 = iaa.SomeOf(n=1, children=augs).augment_heatmaps([heatmaps])[0]
    observed2 = iaa.SomeOf(n=2, children=augs).augment_heatmaps([heatmaps])[0]
    observed3 = iaa.SomeOf(n=3, children=augs).augment_heatmaps([heatmaps])[0]
    assert all([obs.shape == (3, 3, 3) for obs in [observed0, observed1, observed2, observed3]])
    assert all([0 - 1e-6 < obs.min_value < 0 + 1e-6 for obs in [observed0, observed1, observed2, observed3]])
    assert all([1 - 1e-6 < obs.max_value < 1 + 1e-6 for obs in [observed0, observed1, observed2, observed3]])
    for obs, exp in zip([observed0, observed1, observed2, observed3], [heatmaps_arr0, heatmaps_arr1, heatmaps_arr2, heatmaps_arr3]):
        assert np.array_equal(obs.get_arr(), exp)

    # n as tuple
    augs = [iaa.Add(1), iaa.Add(2), iaa.Add(4)]
    nb_iterations = 1000
    nb_observed = [0, 0, 0, 0]
    for i in sm.xrange(nb_iterations):
        observed = iaa.SomeOf(n=(0, 3), children=augs).augment_image(zeros)
        s = observed[0, 0, 0]
        if s == 0:
            nb_observed[0] += 1
        if s & 1 > 0:
            nb_observed[1] += 1
        if s & 2 > 0:
            nb_observed[2] += 1
        if s & 4 > 0:
            nb_observed[3] += 1
    p_observed = [n/nb_iterations for n in nb_observed]
    assert 0.25-0.1 <= p_observed[0] <= 0.25+0.1
    assert 0.5-0.1 <= p_observed[1] <= 0.5+0.1
    assert 0.5-0.1 <= p_observed[2] <= 0.5+0.1
    assert 0.5-0.1 <= p_observed[3] <= 0.5+0.1

    # in-order vs random order
    augs = [iaa.Multiply(2.0), iaa.Add(100)]
    observed = iaa.SomeOf(n=2, children=augs, random_order=False).augment_image(zeros)
    assert np.sum(observed) == 9*100

    nb_iterations = 1000
    nb_observed = [0, 0]
    for i in sm.xrange(nb_iterations):
        augs = [iaa.Multiply(2.0), iaa.Add(100)]
        observed = iaa.SomeOf(n=2, children=augs, random_order=True).augment_image(zeros)
        s = np.sum(observed)
        if s == 9*100:
            nb_observed[0] += 1
        elif s == 9*200:
            nb_observed[1] += 1
        else:
            raise Exception("Unexpected sum: %.8f (@2)" % (s,))
    p_observed = [n/nb_iterations for n in nb_observed]
    assert 0.5-0.1 <= p_observed[0] <= 0.5+0.1
    assert 0.5-0.1 <= p_observed[1] <= 0.5+0.1

    # invalid argument for children
    got_exception = False
    try:
        aug = iaa.SomeOf(1, children=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # n is None
    aug = iaa.SomeOf(None, children=[iaa.Fliplr(1.0), iaa.Flipud(1.0)])
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, np.flipud(np.fliplr(image)))

    # n is (x, None)
    children = [iaa.Fliplr(1.0), iaa.Flipud(1.0), iaa.Add(5)]
    image = np.random.randint(0, 255-5, size=(16, 16), dtype=np.uint8)
    expected = [iaa.Sequential(children).augment_image(image)]
    for _, aug in enumerate(children):
        children_i = [child for child in children if child != aug]
        expected.append(iaa.Sequential(children_i).augment_image(image))
    aug = iaa.SomeOf((2, None), children)
    seen = [0, 0, 0, 0]
    for _ in sm.xrange(400):
        observed = aug.augment_image(image)
        found = 0
        for i, expected_i in enumerate(expected):
            if np.array_equal(observed, expected_i):
                seen[i] += 1
                found += 1
        assert found == 1
    assert 200 - 50 < seen[0] < 200 + 50
    assert 200 - 50 < seen[1] + seen[2] + seen[3] < 200 + 50

    # n is bad (int, "test")
    got_exception = False
    try:
        aug = iaa.SomeOf((2, "test"), children=iaa.Fliplr(1.0))
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # n is stochastic param
    aug = iaa.SomeOf(iap.Choice([0, 1]), children=iaa.Fliplr(1.0))
    image = np.random.randint(0, 255-5, size=(16, 16), dtype=np.uint8)
    seen = [0, 1]
    for _ in sm.xrange(100):
        observed = aug.augment_image(image)
        if np.array_equal(observed, image):
            seen[0] += 1
        elif np.array_equal(observed, np.fliplr(image)):
            seen[1] += 1
        else:
            assert False
    assert seen[0] > 10
    assert seen[1] > 10

    # bad datatype for n
    got_exception = False
    try:
        aug = iaa.SomeOf(False, children=iaa.Fliplr(1.0))
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # test for https://github.com/aleju/imgaug/issues/143
    # (shapes change in child augmenters, leading to problems if input arrays are assumed to
    # stay input arrays)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    aug = iaa.SomeOf(1, [
        iaa.Crop((2, 0, 2, 0), keep_size=False),
        iaa.Crop((1, 0, 1, 0), keep_size=False)
    ])
    for _ in sm.xrange(10):
        observed = aug.augment_images(np.uint8([image, image, image, image]))
        assert isinstance(observed, list)
        assert all([img.shape in [(4, 8, 3), (6, 8, 3)] for img in observed])

        observed = aug.augment_images([image, image, image, image])
        assert isinstance(observed, list)
        assert all([img.shape in [(4, 8, 3), (6, 8, 3)] for img in observed])

        observed = aug.augment_images(np.uint8([image]))
        assert isinstance(observed, list)
        assert all([img.shape in [(4, 8, 3), (6, 8, 3)] for img in observed])

        observed = aug.augment_images([image])
        assert isinstance(observed, list)
        assert all([img.shape in [(4, 8, 3), (6, 8, 3)] for img in observed])

        observed = aug.augment_image(image)
        assert ia.is_np_array(image)
        assert observed.shape in [(4, 8, 3), (6, 8, 3)]

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    aug = iaa.SomeOf(1, [
        iaa.Crop((2, 0, 2, 0), keep_size=True),
        iaa.Crop((1, 0, 1, 0), keep_size=True)
    ])

    for _ in sm.xrange(10):
        observed = aug.augment_images(np.uint8([image, image, image, image]))
        assert ia.is_np_array(observed)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_images([image, image, image, image])
        assert isinstance(observed, list)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_images(np.uint8([image]))
        assert ia.is_np_array(observed)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_images([image])
        assert isinstance(observed, list)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_image(image)
        assert ia.is_np_array(observed)
        assert observed.shape in [(8, 8, 3)]


def test_OneOf():
    reseed()

    zeros = np.zeros((3, 3, 1), dtype=np.uint8)

    # one child augmenter
    observed = iaa.OneOf(children=iaa.Add(1)).augment_image(zeros)
    assert np.array_equal(observed, zeros + 1)
    observed = iaa.OneOf(children=iaa.Sequential([iaa.Add(1)])).augment_image(zeros)
    assert np.array_equal(observed, zeros + 1)
    observed = iaa.OneOf(children=[iaa.Add(1)]).augment_image(zeros)
    assert np.array_equal(observed, zeros + 1)

    # up to three child augmenters
    augs = [iaa.Add(1), iaa.Add(2), iaa.Add(3)]
    aug = iaa.OneOf(augs)

    results = {9*1: 0, 9*2: 0, 9*3: 0}
    nb_iterations = 1000
    for _ in sm.xrange(nb_iterations):
        result = aug.augment_image(zeros)
        s = np.sum(result)
        results[s] += 1
    expected = int(nb_iterations / len(augs))
    expected_tolerance = int(nb_iterations * 0.05)
    for key, val in results.items():
        assert expected - expected_tolerance < val < expected + expected_tolerance

def test_Sometimes():
    reseed()

    image = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 1]], dtype=np.uint8) * 255
    image = image[:, :, np.newaxis]
    images_list = [image]
    images = np.array([image])

    image_lr = np.array([[1, 1, 0],
                         [1, 0, 0],
                         [1, 0, 0]], dtype=np.uint8) * 255
    image_lr = image_lr[:, :, np.newaxis]
    images_lr_list = [image_lr]
    images_lr = np.array([image_lr])

    image_ud = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8) * 255
    image_ud = image_ud[:, :, np.newaxis]
    images_ud_list = [image_ud]
    images_ud = np.array([image_ud])

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=0),
                                      ia.Keypoint(x=2, y=1)], shape=image.shape)]
    keypoints_lr = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=0, y=0),
                                         ia.Keypoint(x=0, y=1)], shape=image.shape)]
    keypoints_ud = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=2, y=2),
                                         ia.Keypoint(x=2, y=1)], shape=image.shape)]

    heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 1.0, 1.0]])
    heatmaps_arr_lr = np.fliplr(heatmaps_arr)
    heatmaps_arr_ud = np.flipud(heatmaps_arr)
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    # 100% chance of if-branch
    aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_lr)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_lr)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_lr_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_lr_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_lr)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_lr)

    # 100% chance of if-branch, heatmaps
    aug = iaa.Sometimes(1.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.array_equal(observed.get_arr(), heatmaps_arr_lr)

    # 100% chance of else-branch
    aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_ud)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_ud)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_ud_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_ud_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_ud)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_ud)

    # 100% chance of else-branch, heatmaps
    aug = iaa.Sometimes(0.0, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.array_equal(observed.get_arr(), heatmaps_arr_ud)

    # 50% if branch, 50% else branch
    aug = iaa.Sometimes(0.5, [iaa.Fliplr(1.0)], [iaa.Flipud(1.0)])
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    nb_images_if_branch = 0
    nb_images_else_branch = 0
    nb_keypoints_if_branch = 0
    nb_keypoints_else_branch = 0
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        keypoints_aug = aug.augment_keypoints(keypoints)
        keypoints_aug_det = aug.augment_keypoints(keypoints)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        if np.array_equal(observed_aug, images_lr):
            nb_images_if_branch += 1
        elif np.array_equal(observed_aug, images_ud):
            nb_images_else_branch += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if keypoints_equal(keypoints_aug, keypoints_lr):
            nb_keypoints_if_branch += 1
        elif keypoints_equal(keypoints_aug, keypoints_ud):
            nb_keypoints_else_branch += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

    assert (0.50 - 0.10) <= nb_images_if_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_images_else_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_keypoints_if_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_keypoints_else_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= (1 - (nb_changed_aug / nb_iterations)) <= (0.50 + 0.10) # should be the same in roughly 50% of all cases
    assert nb_changed_aug_det == 0

    # 50% if branch, otherwise no change
    aug = iaa.Sometimes(0.5, iaa.Fliplr(1.0))
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    nb_images_if_branch = 0
    nb_images_else_branch = 0
    nb_keypoints_if_branch = 0
    nb_keypoints_else_branch = 0
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        keypoints_aug = aug.augment_keypoints(keypoints)
        keypoints_aug_det = aug.augment_keypoints(keypoints)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det

        if np.array_equal(observed_aug, images_lr):
            nb_images_if_branch += 1
        elif np.array_equal(observed_aug, images):
            nb_images_else_branch += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

        if keypoints_equal(keypoints_aug, keypoints_lr):
            nb_keypoints_if_branch += 1
        elif keypoints_equal(keypoints_aug, keypoints):
            nb_keypoints_else_branch += 1
        else:
            raise Exception("Received output doesnt match any expected output.")

    assert (0.50 - 0.10) <= nb_images_if_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_images_else_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_keypoints_if_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= nb_keypoints_else_branch / nb_iterations <= (0.50 + 0.10)
    assert (0.50 - 0.10) <= (1 - (nb_changed_aug / nb_iterations)) <= (0.50 + 0.10) # should be the same in roughly 50% of all cases
    assert nb_changed_aug_det == 0

    # p as stochastic parameter
    image = np.zeros((1, 1), dtype=np.uint8) + 100
    images = [image] * 10
    aug = iaa.Sometimes(p=iap.Binomial(iap.Choice([0.0, 1.0])), then_list=iaa.Add(10))

    seen = [0, 0]
    for _ in sm.xrange(100):
        observed = aug.augment_images(images)
        uq = np.unique(np.uint8(observed))
        assert len(uq) == 1
        if uq[0] == 100:
            seen[0] += 1
        elif uq[0] == 110:
            seen[1] += 1
        else:
            assert False
    assert seen[0] > 20
    assert seen[1] > 20

    # bad datatype for p
    got_exception = False
    try:
        aug = iaa.Sometimes(p="foo")
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # both lists none
    aug = iaa.Sometimes(0.2, then_list=None, else_list=None)
    image = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert np.array_equal(observed, image)

    # then_list bad datatype
    got_exception = False
    try:
        aug = iaa.Sometimes(p=0.2, then_list=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # else_list bad datatype
    got_exception = False
    try:
        aug = iaa.Sometimes(p=0.2, then_list=None, else_list=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # deactivated propagation via hooks
    image = np.random.randint(0, 255-10, size=(16, 16), dtype=np.uint8)
    aug = iaa.Sometimes(1.0, iaa.Add(10))
    observed1 = aug.augment_image(image)
    observed2 = aug.augment_image(image, hooks=ia.HooksImages(propagator=lambda images, augmenter, parents, default: False if augmenter == aug else default))
    assert np.array_equal(observed1, image + 10)
    assert np.array_equal(observed2, image)

    # get_parameters
    aug = iaa.Sometimes(0.75)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Binomial)
    assert isinstance(params[0].p, iap.Deterministic)
    assert 0.75 - 1e-8 < params[0].p.value < 0.75 + 1e-8

    # str/repr
    then_list = iaa.Add(1)
    else_list = iaa.Add(2)
    aug = iaa.Sometimes(0.5, then_list=then_list, else_list=else_list, name="SometimesTest")
    expected = "Sometimes(p=%s, name=%s, then_list=%s, else_list=%s, deterministic=%s)" % (
        "Binomial(Deterministic(float 0.50000000))",
        "SometimesTest",
        "Sequential(name=SometimesTest-then, random_order=False, children=[%s], deterministic=False)" % (str(then_list),),
        "Sequential(name=SometimesTest-else, random_order=False, children=[%s], deterministic=False)" % (str(else_list),),
        "False"
    )
    assert aug.__repr__() == aug.__str__() == expected

    aug = iaa.Sometimes(0.5, then_list=None, else_list=None, name="SometimesTest")
    expected = "Sometimes(p=%s, name=%s, then_list=%s, else_list=%s, deterministic=%s)" % (
        "Binomial(Deterministic(float 0.50000000))",
        "SometimesTest",
        "Sequential(name=SometimesTest-then, random_order=False, children=[], deterministic=False)",
        "Sequential(name=SometimesTest-else, random_order=False, children=[], deterministic=False)",
        "False"
    )
    assert aug.__repr__() == aug.__str__() == expected

    # Test for https://github.com/aleju/imgaug/issues/143
    # (shapes change in child augmenters, leading to problems if input arrays are assumed to
    # stay input arrays)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    aug = iaa.Sometimes(
        0.5,
        iaa.Crop((2, 0, 2, 0), keep_size=False),
        iaa.Crop((1, 0, 1, 0), keep_size=False)
    )
    for _ in sm.xrange(10):
        observed = aug.augment_images(np.uint8([image, image, image, image]))
        assert isinstance(observed, list) or (ia.is_np_array(observed) and len(set([img.shape for img in observed])) == 1)
        assert all([img.shape in [(4, 8, 3), (6, 8, 3)] for img in observed])

        observed = aug.augment_images([image, image, image, image])
        assert isinstance(observed, list)
        assert all([img.shape in [(4, 8, 3), (6, 8, 3)] for img in observed])

        observed = aug.augment_images(np.uint8([image]))
        assert isinstance(observed, list) or (ia.is_np_array(observed) and len(set([img.shape for img in observed])) == 1)
        assert all([img.shape in [(4, 8, 3), (6, 8, 3)] for img in observed])

        observed = aug.augment_images([image])
        assert isinstance(observed, list)
        assert all([img.shape in [(4, 8, 3), (6, 8, 3)] for img in observed])

        observed = aug.augment_image(image)
        assert ia.is_np_array(image)
        assert observed.shape in [(4, 8, 3), (6, 8, 3)]

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    aug = iaa.Sometimes(
        0.5,
        iaa.Crop(((1, 4), 0, (1, 4), 0), keep_size=False),
        iaa.Crop(((4, 8), 0, (4, 8), 0), keep_size=False)
    )
    for _ in sm.xrange(10):
        observed = aug.augment_images(np.uint8([image, image, image, image]))
        assert isinstance(observed, list) or (ia.is_np_array(observed) and len(set([img.shape for img in observed])) == 1)
        assert all([16 <= img.shape[0] <= 30 and img.shape[1:] == (32, 3) for img in observed])

        observed = aug.augment_images([image, image, image, image])
        assert isinstance(observed, list)
        assert all([16 <= img.shape[0] <= 30 and img.shape[1:] == (32, 3) for img in observed])

        observed = aug.augment_images(np.uint8([image]))
        assert isinstance(observed, list) or (ia.is_np_array(observed) and len(set([img.shape for img in observed])) == 1)
        assert all([16 <= img.shape[0] <= 30 and img.shape[1:] == (32, 3) for img in observed])

        observed = aug.augment_images([image])
        assert isinstance(observed, list)
        assert all([16 <= img.shape[0] <= 30 and img.shape[1:] == (32, 3) for img in observed])

        observed = aug.augment_image(image)
        assert ia.is_np_array(image)
        assert 16 <= observed.shape[0] <= 30 and observed.shape[1:] == (32, 3)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    aug = iaa.Sometimes(
        0.5,
        iaa.Crop((2, 0, 2, 0), keep_size=True),
        iaa.Crop((1, 0, 1, 0), keep_size=True)
    )
    for _ in sm.xrange(10):
        observed = aug.augment_images(np.uint8([image, image, image, image]))
        assert ia.is_np_array(observed)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_images([image, image, image, image])
        assert isinstance(observed, list)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_images(np.uint8([image]))
        assert ia.is_np_array(observed)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_images([image])
        assert isinstance(observed, list)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_image(image)
        assert ia.is_np_array(observed)
        assert observed.shape in [(8, 8, 3)]

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    aug = iaa.Sometimes(
        0.5,
        iaa.Crop(((1, 4), 0, (1, 4), 0), keep_size=True),
        iaa.Crop(((4, 8), 0, (4, 8), 0), keep_size=True)
    )
    for _ in sm.xrange(10):
        observed = aug.augment_images(np.uint8([image, image, image, image]))
        assert ia.is_np_array(observed)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_images([image, image, image, image])
        assert isinstance(observed, list)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_images(np.uint8([image]))
        assert ia.is_np_array(observed)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_images([image])
        assert isinstance(observed, list)
        assert all([img.shape in [(8, 8, 3)] for img in observed])

        observed = aug.augment_image(image)
        assert ia.is_np_array(observed)
        assert observed.shape in [(8, 8, 3)]


def test_WithChannels():
    base_img = np.zeros((3, 3, 2), dtype=np.uint8)
    base_img[..., 0] += 100
    base_img[..., 1] += 200

    aug = iaa.WithChannels(None, iaa.Add(10))
    observed = aug.augment_image(base_img)
    expected = base_img + 10
    assert np.allclose(observed, expected)

    aug = iaa.WithChannels(0, iaa.Add(10))
    observed = aug.augment_image(base_img)
    expected = np.copy(base_img)
    expected[..., 0] += 10
    assert np.allclose(observed, expected)

    aug = iaa.WithChannels(1, iaa.Add(10))
    observed = aug.augment_image(base_img)
    expected = np.copy(base_img)
    expected[..., 1] += 10
    assert np.allclose(observed, expected)

    base_img = np.zeros((3, 3, 2), dtype=np.uint8)
    base_img[..., 0] += 5
    base_img[..., 1] += 10
    aug = iaa.WithChannels(1, [iaa.Add(10), iaa.Multiply(2.0)])
    observed = aug.augment_image(base_img)
    expected = np.copy(base_img)
    expected[..., 1] += 10
    expected[..., 1] *= 2
    assert np.allclose(observed, expected)

    # multiple images, given as array
    images = np.concatenate([base_img[np.newaxis, ...], base_img[np.newaxis, ...]], axis=0)
    aug = iaa.WithChannels(1, iaa.Add(10))
    observed = aug.augment_images(images)
    expected = np.copy(images)
    expected[..., 1] += 10
    assert np.allclose(observed, expected)

    # multiple images, given as list
    images = [base_img, base_img]
    aug = iaa.WithChannels(1, iaa.Add(10))
    observed = aug.augment_images(images)
    expected = np.copy(base_img)
    expected[..., 1] += 10
    expected = [expected, expected]
    assert array_equal_lists(observed, expected)

    # children list is empty
    aug = iaa.WithChannels(1, children=None)
    observed = aug.augment_image(base_img)
    expected = np.copy(base_img)
    assert np.array_equal(observed, expected)

    # channel list is empty
    aug = iaa.WithChannels([], iaa.Add(10))
    observed = aug.augment_image(base_img)
    expected = np.copy(base_img)
    assert np.array_equal(observed, expected)

    # invalid datatype for channels
    got_exception = False
    try:
        aug = iaa.WithChannels(False, iaa.Add(10))
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # invalid datatype for children
    got_exception = False
    try:
        aug = iaa.WithChannels(1, False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # get_parameters
    aug = iaa.WithChannels([1], iaa.Add(10))
    params = aug.get_parameters()
    assert len(params) == 1
    assert params[0] == [1]

    # get_children_lists
    children = iaa.Sequential([iaa.Add(10)])
    aug = iaa.WithChannels(1, children)
    assert aug.get_children_lists() == [children]

    # repr/str
    children = iaa.Sequential([iaa.Noop()])
    aug = iaa.WithChannels(1, children, name="WithChannelsTest")
    expected = "WithChannels(channels=[1], name=WithChannelsTest, children=%s, deterministic=False)" % (str(children),)
    assert aug.__repr__() == aug.__str__() == expected


def test_2d_inputs():
    """Test whether inputs of 2D-images (i.e. (H, W) instead of (H, W, C)) work.
    """
    reseed()

    base_img1 = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 1, 1, 1]], dtype=np.uint8)
    base_img2 = np.array([[0, 0, 1, 1],
                          [0, 1, 1, 1],
                          [0, 1, 0, 0]], dtype=np.uint8)

    base_img1_flipped = np.array([[1, 1, 0, 0],
                                  [1, 1, 0, 0],
                                  [1, 1, 1, 0]], dtype=np.uint8)
    base_img2_flipped = np.array([[1, 1, 0, 0],
                                  [1, 1, 1, 0],
                                  [0, 0, 1, 0]], dtype=np.uint8)

    images = np.array([base_img1, base_img2])
    images_flipped = np.array([base_img1_flipped, base_img2_flipped])
    images_list = [base_img1, base_img2]
    images_flipped_list = [base_img1_flipped, base_img2_flipped]
    images_list2d3d = [base_img1, base_img2[:, :, np.newaxis]]
    images_flipped_list2d3d = [base_img1_flipped, base_img2_flipped[:, :, np.newaxis]]

    aug = iaa.Fliplr(1.0)
    noaug = iaa.Fliplr(0.0)

    # one numpy array as input
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_flipped)

    observed = noaug.augment_images(images)
    assert np.array_equal(observed, images)

    # list of 2d images
    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_flipped_list)

    observed = noaug.augment_images(images_list)
    assert array_equal_lists(observed, images_list)

    # list of images, one 2d and one 3d
    observed = aug.augment_images(images_list2d3d)
    assert array_equal_lists(observed, images_flipped_list2d3d)

    observed = noaug.augment_images(images_list2d3d)
    assert array_equal_lists(observed, images_list2d3d)


def test_Augmenter_augment_batches():
    reseed()

    image = np.array([[0, 0, 1, 1],
                      [0, 0, 1, 1],
                      [0, 1, 1, 1]], dtype=np.uint8)
    image_flipped = np.fliplr(image)
    keypoint = ia.Keypoint(x=2, y=1)
    keypoints = [ia.KeypointsOnImage([keypoint], shape=image.shape + (1,))]
    kp_flipped = ia.Keypoint(
        x=image.shape[1]-1-keypoint.x,
        y=keypoint.y
    )

    # basic functionality test (images as list)
    for bg in [True, False]:
        seq = iaa.Fliplr(1.0)
        batches = [ia.Batch(images=[np.copy(image)], keypoints=keypoints)]
        batches_aug = list(seq.augment_batches(batches, background=bg))
        assert np.array_equal(batches_aug[0].images_aug[0], image_flipped)
        assert batches_aug[0].keypoints_aug[0].keypoints[0].x == kp_flipped.x
        assert batches_aug[0].keypoints_aug[0].keypoints[0].y == kp_flipped.y
        assert np.array_equal(batches_aug[0].images[0], image)
        assert batches_aug[0].keypoints[0].keypoints[0].x == keypoint.x
        assert batches_aug[0].keypoints[0].keypoints[0].y == keypoint.y

    # basic functionality test (images as array)
    for bg in [True, False]:
        seq = iaa.Fliplr(1.0)
        batches = [ia.Batch(images=np.uint8([np.copy(image)]), keypoints=keypoints)]
        batches_aug = list(seq.augment_batches(batches, background=bg))
        assert np.array_equal(batches_aug[0].images_aug, np.uint8([image_flipped]))
        assert batches_aug[0].keypoints_aug[0].keypoints[0].x == kp_flipped.x
        assert batches_aug[0].keypoints_aug[0].keypoints[0].y == kp_flipped.y
        assert np.array_equal(batches_aug[0].images, np.uint8([image]))
        assert batches_aug[0].keypoints[0].keypoints[0].x == keypoint.x
        assert batches_aug[0].keypoints[0].keypoints[0].y == keypoint.y


    """
    seq = iaa.Fliplr(0.5)
    # with images as list, background=False
    nb_flipped_images = 0
    nb_flipped_keypoints = 0
    nb_iterations = 1000
    batches = [ia.Batch(images=[np.copy(image)], keypoints=[keypoints[0].deepcopy()]) for _ in sm.xrange(nb_iterations)]
    batches_aug = list(seq.augment_batches(batches, background=False))
    for batch_aug in batches_aug:
        image_aug = batch_aug.images_aug[0]
        keypoint_aug = batch_aug.keypoints_aug[0].keypoints[0]
        assert np.array_equal(image_aug, image) or np.array_equal(image_aug, image_flipped)
        if np.array_equal(image_aug, image_flipped):
            nb_flipped_images += 1

        assert (keypoint_aug.x == keypoint.x and keypoint_aug.y == keypoint.y) \
               or (keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y)
        if keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y:
            nb_flipped_keypoints += 1
    assert 0.4*nb_iterations <= nb_flipped_images <= 0.6*nb_iterations
    assert nb_flipped_images == nb_flipped_keypoints
    """

    seq = iaa.Fliplr(0.5)
    for bg in [False, True]:
        # with images as list
        nb_flipped_images = 0
        nb_flipped_keypoints = 0
        nb_iterations = 1000
        batches = [ia.Batch(images=[np.copy(image)], keypoints=[keypoints[0].deepcopy()]) for _ in sm.xrange(nb_iterations)]
        batches_aug = list(seq.augment_batches(batches, background=bg))
        for batch_aug in batches_aug:
            image_aug = batch_aug.images_aug[0]
            keypoint_aug = batch_aug.keypoints_aug[0].keypoints[0]
            assert np.array_equal(image_aug, image) or np.array_equal(image_aug, image_flipped)
            if np.array_equal(image_aug, image_flipped):
                nb_flipped_images += 1

            assert (keypoint_aug.x == keypoint.x and keypoint_aug.y == keypoint.y) \
                   or (keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y)
            if keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y:
                nb_flipped_keypoints += 1
        assert 0.4*nb_iterations <= nb_flipped_images <= 0.6*nb_iterations
        assert nb_flipped_images == nb_flipped_keypoints

        # with images as array
        nb_flipped_images = 0
        nb_flipped_keypoints = 0
        nb_iterations = 1000
        batches = [ia.Batch(images=np.array([np.copy(image)], dtype=np.uint8), keypoints=None) for _ in sm.xrange(nb_iterations)]
        batches_aug = list(seq.augment_batches(batches, background=bg))
        for batch_aug in batches_aug:
            #batch = ia.Batch(images=np.array([image], dtype=np.uint8), keypoints=keypoints)
            #batches_aug = list(seq.augment_batches([batch], background=True))
            #batch_aug = batches_aug[0]
            image_aug = batch_aug.images_aug[0]
            assert np.array_equal(image_aug, image) or np.array_equal(image_aug, image_flipped)
            if np.array_equal(image_aug, image_flipped):
                nb_flipped_images += 1
        assert 0.4*nb_iterations <= nb_flipped_images <= 0.6*nb_iterations

        # array (N, H, W) as input
        nb_flipped_images = 0
        nb_iterations = 1000
        batches = [np.array([np.copy(image)], dtype=np.uint8) for _ in sm.xrange(nb_iterations)]
        batches_aug = list(seq.augment_batches(batches, background=bg))
        for batch_aug in batches_aug:
            #batch = np.array([image], dtype=np.uint8)
            #batches_aug = list(seq.augment_batches([batch], background=True))
            #image_aug = batches_aug[0][0]
            image_aug = batch_aug[0]
            assert np.array_equal(image_aug, image) or np.array_equal(image_aug, image_flipped)
            if np.array_equal(image_aug, image_flipped):
                nb_flipped_images += 1
        assert 0.4*nb_iterations <= nb_flipped_images <= 0.6*nb_iterations

        # list of list of KeypointsOnImage as input
        nb_flipped_keypoints = 0
        nb_iterations = 1000
        #batches = [ia.Batch(images=[np.copy(image)], keypoints=None) for _ in sm.xrange(nb_iterations)]
        batches = [[keypoints[0].deepcopy()] for _ in sm.xrange(nb_iterations)]
        batches_aug = list(seq.augment_batches(batches, background=bg))
        for batch_aug in batches_aug:
            #batch = [keypoints]
            #batches_aug = list(seq.augment_batches([batch], background=True))
            #batch_aug = batches_aug[0]
            #keypoint_aug = batches_aug[0].keypoints[0].keypoints[0]
            keypoint_aug = batch_aug[0].keypoints[0]

            assert (keypoint_aug.x == keypoint.x and keypoint_aug.y == keypoint.y) \
                   or (keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y)
            if keypoint_aug.x == kp_flipped.x and keypoint_aug.y == kp_flipped.y:
                nb_flipped_keypoints += 1
        assert 0.4*nb_iterations <= nb_flipped_keypoints <= 0.6*nb_iterations

    # test all augmenters
    # this test is currently skipped by default as it takes around 40s on its own,
    # probably because of having to start background processes
    """
    augs = [
        iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.SomeOf(1, [iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.Sometimes(1.0, iaa.Fliplr(1)),
        iaa.WithColorspace("HSV", children=iaa.Add((-50, 50))),
        iaa.WithChannels([0], iaa.Add((-50, 50))),
        iaa.Noop(name="Noop-nochange"),
        iaa.Lambda(
            func_images=lambda images, random_state, parents, hooks: images,
            func_keypoints=lambda keypoints_on_images, random_state, parents, hooks: keypoints_on_images,
            name="Lambda-nochange"
        ),
        iaa.AssertLambda(
            func_images=lambda images, random_state, parents, hooks: True,
            func_keypoints=lambda keypoints_on_images, random_state, parents, hooks: True,
            name="AssertLambda-nochange"
        ),
        iaa.AssertShape(
            (None, 64, 64, 3),
            check_keypoints=False,
            name="AssertShape-nochange"
        ),
        iaa.Scale((0.5, 0.9)),
        iaa.CropAndPad(px=(-50, 50)),
        iaa.Pad(px=(1, 50)),
        iaa.Crop(px=(1, 50)),
        iaa.Fliplr(1.0),
        iaa.Flipud(1.0),
        iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128)),
        iaa.ChangeColorspace(to_colorspace="GRAY"),
        iaa.Grayscale(alpha=(0.1, 1.0)),
        iaa.GaussianBlur(1.0),
        iaa.AverageBlur(5),
        iaa.MedianBlur(5),
        iaa.Convolve(np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])),
        iaa.Sharpen(alpha=(0.1, 1.0), lightness=(0.8, 1.2)),
        iaa.Emboss(alpha=(0.1, 1.0), strength=(0.8, 1.2)),
        iaa.EdgeDetect(alpha=(0.1, 1.0)),
        iaa.DirectedEdgeDetect(alpha=(0.1, 1.0), direction=(0.0, 1.0)),
        iaa.Add((-50, 50)),
        iaa.AddElementwise((-50, 50)),
        iaa.AdditiveGaussianNoise(scale=(0.1, 1.0)),
        iaa.Multiply((0.6, 1.4)),
        iaa.MultiplyElementwise((0.6, 1.4)),
        iaa.Dropout((0.3, 0.5)),
        iaa.CoarseDropout((0.3, 0.5), size_percent=(0.05, 0.2)),
        iaa.Invert(0.5),
        iaa.ContrastNormalization((0.6, 1.4)),
        iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), rotate=(-20, 20),
                   shear=(-20, 20), order=ia.ALL, mode=ia.ALL, cval=(0, 255)),
        iaa.PiecewiseAffine(scale=(0.1, 0.3)),
        iaa.ElasticTransformation(alpha=0.5)
    ]

    nb_iterations = 100
    image = ia.quokka(size=(64, 64))
    batch = ia.Batch(images=np.array([image]), keypoints=keypoints)
    batches = [ia.Batch(images=[np.copy(image)], keypoints=[keypoints[0].deepcopy()])
               for _ in sm.xrange(nb_iterations)]
    for aug in augs:
        nb_changed = 0
        batches_aug = list(aug.augment_batches(batches, background=True))
        for batch_aug in batches_aug:
            image_aug = batch_aug.images_aug[0]
            if image.shape != image_aug.shape or not np.array_equal(image, image_aug):
                nb_changed += 1
                if nb_changed > 10:
                    break
        if "-nochange" not in aug.name:
            assert nb_changed > 0
        else:
            assert nb_changed == 0
    """


def test_determinism():
    reseed()

    images = [
        ia.quokka(size=(128, 128)),
        ia.quokka(size=(64, 64)),
        ia.imresize_single_image(data.astronaut(), (128, 256))
    ]
    keypoints = [
        ia.KeypointsOnImage([
            ia.Keypoint(x=20, y=10), ia.Keypoint(x=5, y=5), ia.Keypoint(x=10, y=43)
            ], shape=(50, 60, 3))
    ]

    augs = [
        iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.SomeOf(1, [iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.OneOf([iaa.Fliplr(1.0), iaa.Flipud(1.0)]),
        iaa.Sometimes(1.0, iaa.Fliplr(1)),
        iaa.WithColorspace("HSV", children=iaa.Add((-50, 50))),
        iaa.WithChannels([0], iaa.Add((-50, 50))),
        iaa.Noop(name="Noop-nochange"),
        iaa.Lambda(
            func_images=lambda images, random_state, parents, hooks: images,
            func_heatmaps=lambda heatmaps, random_state, parents, hooks: heatmaps,
            func_keypoints=lambda keypoints_on_images, random_state, parents, hooks: keypoints_on_images,
            name="Lambda-nochange"
        ),
        iaa.AssertLambda(
            func_images=lambda images, random_state, parents, hooks: True,
            func_heatmaps=lambda heatmaps, random_state, parents, hooks: True,
            func_keypoints=lambda keypoints_on_images, random_state, parents, hooks: True,
            name="AssertLambda-nochange"
        ),
        iaa.AssertShape(
            (None, None, None, 3),
            check_keypoints=False,
            name="AssertShape-nochange"
        ),
        iaa.Scale((0.5, 0.9)),
        iaa.CropAndPad(px=(-50, 50)),
        iaa.Pad(px=(1, 50)),
        iaa.Crop(px=(1, 50)),
        iaa.Fliplr(1.0),
        iaa.Flipud(1.0),
        iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128)),
        iaa.ChangeColorspace(to_colorspace="GRAY"),
        iaa.Grayscale(alpha=(0.1, 1.0)),
        iaa.GaussianBlur(1.0),
        iaa.AverageBlur(5),
        iaa.MedianBlur(5),
        iaa.Convolve(np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])),
        iaa.Sharpen(alpha=(0.1, 1.0), lightness=(0.8, 1.2)),
        iaa.Emboss(alpha=(0.1, 1.0), strength=(0.8, 1.2)),
        iaa.EdgeDetect(alpha=(0.1, 1.0)),
        iaa.DirectedEdgeDetect(alpha=(0.1, 1.0), direction=(0.0, 1.0)),
        iaa.Add((-50, 50)),
        iaa.AddElementwise((-50, 50)),
        iaa.AdditiveGaussianNoise(scale=(0.1, 1.0)),
        iaa.Multiply((0.6, 1.4)),
        iaa.MultiplyElementwise((0.6, 1.4)),
        iaa.Dropout((0.3, 0.5)),
        iaa.CoarseDropout((0.3, 0.5), size_percent=(0.05, 0.2)),
        iaa.Invert(0.5),
        iaa.ContrastNormalization((0.6, 1.4)),
        iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1),
                   rotate=(-20, 20), shear=(-20, 20), order=ia.ALL,
                   mode=ia.ALL, cval=(0, 255)),
        iaa.PiecewiseAffine(scale=(0.1, 0.3)),
        iaa.ElasticTransformation(alpha=0.5)
    ]

    for aug in augs:
        aug_det = aug.to_deterministic()
        images_aug1 = aug_det.augment_images(images)
        images_aug2 = aug_det.augment_images(images)
        kps_aug1 = aug_det.augment_keypoints(keypoints)
        kps_aug2 = aug_det.augment_keypoints(keypoints)
        assert array_equal_lists(images_aug1, images_aug2), \
            "Images not identical for %s" % (aug.name,)
        assert keypoints_equal(kps_aug1, kps_aug2), \
            "Keypoints not identical for %s" % (aug.name,)


def test_keypoint_augmentation():
    ia.seed(1)

    keypoints = []
    for y in range(40//5):
        for x in range(60//5):
            keypoints.append(ia.Keypoint(y=y*5, x=x*5))

    keypoints_oi = ia.KeypointsOnImage(keypoints, shape=(40, 60, 3))
    keypoints_oi_empty = ia.KeypointsOnImage([], shape=(40, 60, 3))

    augs = [
        iaa.Add((-5, 5), name="Add"),
        iaa.AddElementwise((-5, 5), name="AddElementwise"),
        iaa.AdditiveGaussianNoise(0.01*255, name="AdditiveGaussianNoise"),
        iaa.Multiply((0.95, 1.05), name="Multiply"),
        iaa.Dropout(0.01, name="Dropout"),
        iaa.CoarseDropout(0.01, size_px=6, name="CoarseDropout"),
        iaa.Invert(0.01, per_channel=True, name="Invert"),
        iaa.ContrastNormalization((0.95, 1.05), name="ContrastNormalization"),
        iaa.GaussianBlur(sigma=(0.95, 1.05), name="GaussianBlur"),
        iaa.AverageBlur((3, 5), name="AverageBlur"),
        iaa.MedianBlur((3, 5), name="MedianBlur"),
        #iaa.BilateralBlur((3, 5), name="BilateralBlur"),
        # WithColorspace ?
        #iaa.AddToHueAndSaturation((-5, 5), name="AddToHueAndSaturation"),
        # ChangeColorspace ?
        # Grayscale cannot be tested, input not RGB
        # Convolve ?
        iaa.Sharpen((0.0, 0.1), lightness=(1.0, 1.2), name="Sharpen"),
        iaa.Emboss(alpha=(0.0, 0.1), strength=(0.5, 1.5), name="Emboss"),
        iaa.EdgeDetect(alpha=(0.0, 0.1), name="EdgeDetect"),
        iaa.DirectedEdgeDetect(alpha=(0.0, 0.1), direction=0, name="DirectedEdgeDetect"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.Affine(translate_px=(-5, 5), name="Affine-translate-px"),
        iaa.Affine(translate_percent=(-0.05, 0.05), name="Affine-translate-percent"),
        iaa.Affine(rotate=(-20, 20), name="Affine-rotate"),
        iaa.Affine(shear=(-20, 20), name="Affine-shear"),
        iaa.Affine(scale=(0.9, 1.1), name="Affine-scale"),
        iaa.PiecewiseAffine(scale=(0.001, 0.005), name="PiecewiseAffine"),
        #iaa.PerspectiveTransform(scale=(0.01, 0.10), name="PerspectiveTransform"),
        iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=(0.1, 0.2), name="ElasticTransformation"),
        # Sequential
        # SomeOf
        # OneOf
        # Sometimes
        # WithChannels
        # Noop
        # Lambda
        # AssertLambda
        # AssertShape
        iaa.Alpha((0.0, 0.1), iaa.Add(10), name="Alpha"),
        iaa.AlphaElementwise((0.0, 0.1), iaa.Add(10), name="AlphaElementwise"),
        iaa.SimplexNoiseAlpha(iaa.Add(10), name="SimplexNoiseAlpha"),
        iaa.FrequencyNoiseAlpha(exponent=(-2, 2), first=iaa.Add(10),
                                name="SimplexNoiseAlpha"),
        iaa.Superpixels(p_replace=0.01, n_segments=64),
        iaa.Scale(0.5, name="Scale"),
        iaa.CropAndPad(px=(-10, 10), name="CropAndPad"),
        iaa.Pad(px=(0, 10), name="Pad"),
        iaa.Crop(px=(0, 10), name="Crop")
    ]

    for aug in augs:
        #if aug.name != "PiecewiseAffine":
        #    continue
        dss = []
        for i in range(10):
            aug_det = aug.to_deterministic()

            kp_fully_empty_aug = aug_det.augment_keypoints([])
            assert kp_fully_empty_aug == []

            kp_first_empty_aug = aug_det.augment_keypoints([keypoints_oi_empty])[0]
            assert len(kp_first_empty_aug.keypoints) == 0

            kp_image = keypoints_oi.to_keypoint_image(size=5)
            kp_image_aug = aug_det.augment_image(kp_image)
            kp_image_aug_rev = ia.KeypointsOnImage.from_keypoint_image(
                kp_image_aug,
                if_not_found_coords={"x": -9999, "y": -9999},
                nb_channels=1
            )
            kp_aug = aug_det.augment_keypoints([keypoints_oi])[0]
            ds = []
            assert len(kp_image_aug_rev.keypoints) == len(kp_aug.keypoints),\
                "Lost keypoints for '%s' (%d vs expected %d)" \
                % (aug.name, len(kp_aug.keypoints), len(kp_image_aug_rev.keypoints))
            for kp_pred, kp_pred_img in zip(kp_aug.keypoints, kp_image_aug_rev.keypoints):
                kp_pred_lost = (kp_pred.x == -9999 and kp_pred.y == -9999)
                kp_pred_img_lost = (kp_pred_img.x == -9999 and kp_pred_img.y == -9999)
                #if kp_pred_lost and not kp_pred_img_lost:
                #    print("lost kp_pred", kp_pred_img)
                #elif not kp_pred_lost and kp_pred_img_lost:
                #    print("lost kp_pred_img", kp_pred)
                #elif kp_pred_lost and kp_pred_img_lost:
                #    print("lost both keypoints")

                if not kp_pred_lost and not kp_pred_img_lost:
                    d = np.sqrt((kp_pred.x - kp_pred_img.x) ** 2
                                + (kp_pred.y - kp_pred_img.y) ** 2)
                    ds.append(d)
            #print(aug.name, np.average(ds), ds)
            dss.extend(ds)
            if len(ds) == 0:
                print("[INFO] No valid keypoints found for '%s' "
                      "in test_keypoint_augmentation()" % (str(aug),))
        assert np.average(dss) < 5.0, \
            "Average distance too high (%.2f, with ds: %s)" \
            % (np.average(dss), str(dss))


def test_unusual_channel_numbers():
    ia.seed(1)

    images = [
        (0, create_random_images((4, 16, 16))),
        (1, create_random_images((4, 16, 16, 1))),
        (2, create_random_images((4, 16, 16, 2))),
        (4, create_random_images((4, 16, 16, 4))),
        (5, create_random_images((4, 16, 16, 5))),
        (10, create_random_images((4, 16, 16, 10))),
        (20, create_random_images((4, 16, 16, 20)))
    ]

    augs = [
        iaa.Add((-5, 5), name="Add"),
        iaa.AddElementwise((-5, 5), name="AddElementwise"),
        iaa.AdditiveGaussianNoise(0.01*255, name="AdditiveGaussianNoise"),
        iaa.Multiply((0.95, 1.05), name="Multiply"),
        iaa.Dropout(0.01, name="Dropout"),
        iaa.CoarseDropout(0.01, size_px=6, name="CoarseDropout"),
        iaa.Invert(0.01, per_channel=True, name="Invert"),
        iaa.ContrastNormalization((0.95, 1.05), name="ContrastNormalization"),
        iaa.GaussianBlur(sigma=(0.95, 1.05), name="GaussianBlur"),
        iaa.AverageBlur((3, 5), name="AverageBlur"),
        iaa.MedianBlur((3, 5), name="MedianBlur"),
        #iaa.BilateralBlur((3, 5), name="BilateralBlur"), # works only with 3/RGB channels
        # WithColorspace ?
        #iaa.AddToHueAndSaturation((-5, 5), name="AddToHueAndSaturation"), # works only with 3/RGB channels
        # ChangeColorspace ?
        #iaa.Grayscale((0.0, 0.1), name="Grayscale"), # works only with 3 channels
        # Convolve ?
        iaa.Sharpen((0.0, 0.1), lightness=(1.0, 1.2), name="Sharpen"),
        iaa.Emboss(alpha=(0.0, 0.1), strength=(0.5, 1.5), name="Emboss"),
        iaa.EdgeDetect(alpha=(0.0, 0.1), name="EdgeDetect"),
        iaa.DirectedEdgeDetect(alpha=(0.0, 0.1), direction=0,
                               name="DirectedEdgeDetect"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.Affine(translate_px=(-5, 5), name="Affine-translate-px"),
        iaa.Affine(translate_percent=(-0.05, 0.05), name="Affine-translate-percent"),
        iaa.Affine(rotate=(-20, 20), name="Affine-rotate"),
        iaa.Affine(shear=(-20, 20), name="Affine-shear"),
        iaa.Affine(scale=(0.9, 1.1), name="Affine-scale"),
        iaa.PiecewiseAffine(scale=(0.001, 0.005), name="PiecewiseAffine"),
        iaa.PerspectiveTransform(scale=(0.01, 0.10), name="PerspectiveTransform"),
        iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=(0.1, 0.2),
                                  name="ElasticTransformation"),
        iaa.Sequential([iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))]),
        iaa.SomeOf(1, [iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))]),
        iaa.OneOf(iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))),
        iaa.Sometimes(0.5, iaa.Add((-5, 5)), name="Sometimes"),
        # WithChannels
        iaa.Noop(name="Noop"),
        # Lambda
        # AssertLambda
        # AssertShape
        iaa.Alpha((0.0, 0.1), iaa.Add(10), name="Alpha"),
        iaa.AlphaElementwise((0.0, 0.1), iaa.Add(10), name="AlphaElementwise"),
        iaa.SimplexNoiseAlpha(iaa.Add(10), name="SimplexNoiseAlpha"),
        iaa.FrequencyNoiseAlpha(exponent=(-2, 2), first=iaa.Add(10),
                                name="SimplexNoiseAlpha"),
        iaa.Superpixels(p_replace=0.01, n_segments=64),
        iaa.Scale({"height": 4, "width": 4}, name="Scale"),
        iaa.CropAndPad(px=(-10, 10), name="CropAndPad"),
        iaa.Pad(px=(0, 10), name="Pad"),
        iaa.Crop(px=(0, 10), name="Crop")
    ]

    for aug in augs:
        for (nb_channels, images_c) in images:
            #print("shape", images_c.shape, aug.name)
            if aug.name != "Scale":
                images_aug = aug.augment_images(images_c)
                assert images_aug.shape == images_c.shape
                image_aug = aug.augment_image(images_c[0])
                assert image_aug.shape == images_c[0].shape
            else:
                images_aug = aug.augment_images(images_c)
                image_aug = aug.augment_image(images_c[0])
                if images_c.ndim == 3:
                    assert images_aug.shape == (4, 4, 4)
                    assert image_aug.shape == (4, 4)
                else:
                    assert images_aug.shape == (4, 4, 4, images_c.shape[3])
                    assert image_aug.shape == (4, 4, images_c.shape[3])


#@attr("now")
def test_dtype_preservation():
    ia.seed(1)

    size = (4, 16, 16, 3)
    images = [
        np.random.uniform(0, 255, size).astype(np.uint8),
        np.random.uniform(0, 65535, size).astype(np.uint16),
        np.random.uniform(0, 4294967295, size).astype(np.uint32), # not supported by cv2.blur in AverageBlur
        np.random.uniform(-128, 127, size).astype(np.int16),
        np.random.uniform(-32768, 32767, size).astype(np.int32),
        np.random.uniform(0.0, 1.0, size).astype(np.float32),
        np.random.uniform(-1000.0, 1000.0, size).astype(np.float16), # not supported by scipy.ndimage.filter in GaussianBlur
        np.random.uniform(-1000.0, 1000.0, size).astype(np.float32),
        np.random.uniform(-1000.0, 1000.0, size).astype(np.float64)
    ]

    default_dtypes = set([arr.dtype for arr in images])
    # Some dtypes are here removed per augmenter, because the respective
    # augmenter does not support them. This test currently only checks whether
    # dtypes are preserved from in- to output for all dtypes that are supported
    # per augmenter.
    # dtypes are here removed via list comprehension instead of
    # `default_dtypes - set([dtype])`, because the latter one simply never
    # removed the dtype(s) for some reason?!
    augs = [
        (iaa.Add((-5, 5), name="Add"), default_dtypes),
        (iaa.AddElementwise((-5, 5), name="AddElementwise"), default_dtypes),
        (iaa.AdditiveGaussianNoise(0.01*255, name="AdditiveGaussianNoise"), default_dtypes),
        (iaa.Multiply((0.95, 1.05), name="Multiply"), default_dtypes),
        (iaa.Dropout(0.01, name="Dropout"), default_dtypes),
        (iaa.CoarseDropout(0.01, size_px=6, name="CoarseDropout"), default_dtypes),
        (iaa.Invert(0.01, per_channel=True, name="Invert"), default_dtypes),
        (iaa.ContrastNormalization((0.95, 1.05), name="ContrastNormalization"), default_dtypes),
        (iaa.GaussianBlur(sigma=(0.95, 1.05), name="GaussianBlur"), [dt for dt in default_dtypes if dt not in [np.float16]]),
        (iaa.AverageBlur((3, 5), name="AverageBlur"), [dt for dt in default_dtypes if dt not in [np.uint32, np.float16]]),
        (iaa.MedianBlur((3, 5), name="MedianBlur"), [dt for dt in default_dtypes if dt not in [np.uint32, np.int32, np.float16, np.float64]]),
        (iaa.BilateralBlur((3, 5), name="BilateralBlur"), [dt for dt in default_dtypes if dt not in [np.uint16, np.uint32, np.int16, np.int32, np.float16, np.float64]]),
        # WithColorspace ?
        #iaa.AddToHueAndSaturation((-5, 5), name="AddToHueAndSaturation"), # works only with RGB/uint8
        # ChangeColorspace ?
        #iaa.Grayscale((0.0, 0.1), name="Grayscale"), # works only with RGB/uint8
        # Convolve ?
        (iaa.Sharpen((0.0, 0.1), lightness=(1.0, 1.2), name="Sharpen"), [dt for dt in default_dtypes if dt not in [np.uint32, np.int32, np.float16, np.uint32]]),
        (iaa.Emboss(alpha=(0.0, 0.1), strength=(0.5, 1.5), name="Emboss"), [dt for dt in default_dtypes if dt not in [np.uint32, np.int32, np.float16, np.uint32]]),
        (iaa.EdgeDetect(alpha=(0.0, 0.1), name="EdgeDetect"), [dt for dt in default_dtypes if dt not in [np.uint32, np.int32, np.float16, np.uint32]]),
        (iaa.DirectedEdgeDetect(alpha=(0.0, 0.1), direction=0, name="DirectedEdgeDetect"), [dt for dt in default_dtypes if dt not in [np.uint32, np.int32, np.float16, np.uint32]]),
        (iaa.Fliplr(0.5, name="Fliplr"), default_dtypes),
        (iaa.Flipud(0.5, name="Flipud"), default_dtypes),
        (iaa.Affine(translate_px=(-5, 5), name="Affine-translate-px"), default_dtypes),
        (iaa.Affine(translate_percent=(-0.05, 0.05), name="Affine-translate-percent"), default_dtypes),
        (iaa.Affine(rotate=(-20, 20), name="Affine-rotate"), default_dtypes),
        (iaa.Affine(shear=(-20, 20), name="Affine-shear"), default_dtypes),
        (iaa.Affine(scale=(0.9, 1.1), name="Affine-scale"), default_dtypes),
        (iaa.PiecewiseAffine(scale=(0.001, 0.005), name="PiecewiseAffine"), default_dtypes),
        #(iaa.PerspectiveTransform(scale=(0.01, 0.10), name="PerspectiveTransform"), [dt for dt in default_dtypes if dt not in [np.uint32]]),
        (iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=(0.1, 0.2), name="ElasticTransformation"), [dt for dt in default_dtypes if dt not in [np.float16]]),
        (iaa.Sequential([iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))]), default_dtypes),
        (iaa.SomeOf(1, [iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))]), default_dtypes),
        (iaa.OneOf(iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))), default_dtypes),
        (iaa.Sometimes(0.5, iaa.Add((-5, 5)), name="Sometimes"), default_dtypes),
        # WithChannels
        (iaa.Noop(name="Noop"), default_dtypes),
        # Lambda
        # AssertLambda
        # AssertShape
        (iaa.Alpha((0.0, 0.1), iaa.Add(10), name="Alpha"), default_dtypes),
        (iaa.AlphaElementwise((0.0, 0.1), iaa.Add(10), name="AlphaElementwise"), default_dtypes),
        (iaa.SimplexNoiseAlpha(iaa.Add(10), name="SimplexNoiseAlpha"), default_dtypes),
        (iaa.FrequencyNoiseAlpha(exponent=(-2, 2), first=iaa.Add(10), name="SimplexNoiseAlpha"), default_dtypes),
        (iaa.Superpixels(p_replace=0.01, n_segments=64), [dt for dt in default_dtypes if dt not in [np.float16, np.float32]]),
        (iaa.Scale({"height": 4, "width": 4}, name="Scale"), [dt for dt in default_dtypes if dt not in [np.uint16, np.uint32, np.int16, np.int32, np.float32, np.float16, np.float64]]),
        (iaa.CropAndPad(px=(-10, 10), name="CropAndPad"), [dt for dt in default_dtypes if dt not in [np.uint16, np.uint32, np.int16, np.int32, np.float32, np.float16, np.float64]]),
        (iaa.Pad(px=(0, 10), name="Pad"), [dt for dt in default_dtypes if dt not in [np.uint16, np.uint32, np.int16, np.int32, np.float32, np.float16, np.float64]]),
        (iaa.Crop(px=(0, 10), name="Crop"), [dt for dt in default_dtypes if dt not in [np.uint16, np.uint32, np.int16, np.int32, np.float32, np.float16, np.float64]])
    ]

    for (aug, allowed_dtypes) in augs:
        #print(aug.name, allowed_dtypes)
        for images_i in images:
            if images_i.dtype in allowed_dtypes:
                #print("shape", images_i.shape, images_i.dtype, aug.name)
                images_aug = aug.augment_images(images_i)
                #assert images_aug.shape == images_i.shape
                assert images_aug.dtype == images_i.dtype
            else:
                #print("Skipped dtype %s for augmenter %s" % (images_i.dtype, aug.name))
                pass


def test_parameters_handle_continuous_param():
    # value without value range
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test1]", value_range=None, tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test1]" in str(e)
    assert got_exception == False

    # value without value range as (None, None)
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test1b]", value_range=(None, None), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test1b]" in str(e)
    assert got_exception == False

    # stochastic parameter
    got_exception = False
    try:
        result = iap.handle_continuous_param(iap.Deterministic(1), "[test2]", value_range=None, tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test2]" in str(e)
    assert got_exception == False

    # value within value range
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test3]", value_range=(0, 10), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test3]" in str(e)
    assert got_exception == False

    # value outside of value range
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test4]", value_range=(2, 12), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test4]" in str(e)
    assert got_exception == True

    # value within value range (without lower bound)
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test5]", value_range=(None, 12), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test5]" in str(e)
    assert got_exception == False

    # value outside of value range (without lower bound)
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test6]", value_range=(None, 0), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test6]" in str(e)
    assert got_exception == True

    # value within value range (without upper bound)
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test7]", value_range=(-1, None), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test7]" in str(e)
    assert got_exception == False

    # value outside of value range (without upper bound)
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test8]", value_range=(2, None), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test8]" in str(e)
    assert got_exception == True

    # tuple as value, but no tuples allowed
    got_exception = False
    try:
        result = iap.handle_continuous_param((1, 2), "[test9]", value_range=None, tuple_to_uniform=False, list_to_choice=True)
        assert isinstance(result, iap.Uniform)
    except Exception as e:
        got_exception = True
        assert "[test9]" in str(e)
    assert got_exception == True

    # tuple as value and tuple allowed
    got_exception = False
    try:
        result = iap.handle_continuous_param((1, 2), "[test10]", value_range=None, tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Uniform)
    except Exception as e:
        got_exception = True
        assert "[test10]" in str(e)
    assert got_exception == False

    # tuple as value and tuple allowed and tuple within value range
    got_exception = False
    try:
        result = iap.handle_continuous_param((1, 2), "[test11]", value_range=(0, 10), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Uniform)
    except Exception as e:
        got_exception = True
        assert "[test11]" in str(e)
    assert got_exception == False

    # tuple as value and tuple allowed and tuple partially outside of value range
    got_exception = False
    try:
        result = iap.handle_continuous_param((1, 2), "[test12]", value_range=(1.5, 13), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Uniform)
    except Exception as e:
        got_exception = True
        assert "[test12]" in str(e)
    assert got_exception == True

    # tuple as value and tuple allowed and tuple fully outside of value range
    got_exception = False
    try:
        result = iap.handle_continuous_param((1, 2), "[test13]", value_range=(3, 13), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Uniform)
    except Exception as e:
        got_exception = True
        assert "[test13]" in str(e)
    assert got_exception == True

    # list as value, but no list allowed
    got_exception = False
    try:
        result = iap.handle_continuous_param([1, 2, 3], "[test14]", value_range=None, tuple_to_uniform=True, list_to_choice=False)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test14]" in str(e)
    assert got_exception == True

    # list as value and list allowed
    got_exception = False
    try:
        result = iap.handle_continuous_param([1, 2, 3], "[test15]", value_range=None, tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test15]" in str(e)
    assert got_exception == False

    # list as value and list allowed and list partially outside of value range
    got_exception = False
    try:
        result = iap.handle_continuous_param([1, 2], "[test16]", value_range=(1.5, 13), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test16]" in str(e)
    assert got_exception == True

    # list as value and list allowed and list fully outside of value range
    got_exception = False
    try:
        result = iap.handle_continuous_param([1, 2], "[test17]", value_range=(3, 13), tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test17]" in str(e)
    assert got_exception == True

    # single value within value range given as callable
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test18]", value_range=lambda x: -1 < x < 1, tuple_to_uniform=True, list_to_choice=True)
    except Exception as e:
        got_exception = True
        assert "[test18]" in str(e)
    assert got_exception == False

    # bad datatype for value range
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test19]", value_range=False, tuple_to_uniform=True, list_to_choice=True)
    except Exception as e:
        got_exception = True
        assert "Unexpected input for value_range" in str(e)
    assert got_exception == True


def test_parameters_handle_discrete_param():
    # float value without value range when no float value is allowed
    got_exception = False
    try:
        result = iap.handle_discrete_param(1.5, "[test0]", value_range=None, tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test0]" in str(e)
    assert got_exception == True

    # value without value range
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test1]", value_range=None, tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test1]" in str(e)
    assert got_exception == False

    # value without value range as (None, None)
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test1b]", value_range=(None, None), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test1b]" in str(e)
    assert got_exception == False

    # stochastic parameter
    got_exception = False
    try:
        result = iap.handle_discrete_param(iap.Deterministic(1), "[test2]", value_range=None, tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test2]" in str(e)
    assert got_exception == False

    # value within value range
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test3]", value_range=(0, 10), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test3]" in str(e)
    assert got_exception == False

    # value outside of value range
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test4]", value_range=(2, 12), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test4]" in str(e)
    assert got_exception == True

    # value within value range (without lower bound)
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test5]", value_range=(None, 12), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test5]" in str(e)
    assert got_exception == False

    # value outside of value range (without lower bound)
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test6]", value_range=(None, 0), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test6]" in str(e)
    assert got_exception == True

    # value within value range (without upper bound)
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test7]", value_range=(-1, None), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test7]" in str(e)
    assert got_exception == False

    # value outside of value range (without upper bound)
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test8]", value_range=(2, None), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test8]" in str(e)
    assert got_exception == True

    # tuple as value, but no tuples allowed
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 2), "[test9]", value_range=None, tuple_to_uniform=False, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test9]" in str(e)
    assert got_exception == True

    # tuple as value and tuple allowed
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 2), "[test10]", value_range=None, tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test10]" in str(e)
    assert got_exception == False

    # tuple as value and tuple allowed and tuple within value range
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 2), "[test11]", value_range=(0, 10), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test11]" in str(e)
    assert got_exception == False

    # tuple as value and tuple allowed and tuple within value range with allow_floats=False
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 2), "[test11b]", value_range=(0, 10), tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test11b]" in str(e)
    assert got_exception == False

    # tuple as value and tuple allowed and tuple partially outside of value range
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 3), "[test12]", value_range=(2, 13), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test12]" in str(e)
    assert got_exception == True

    # tuple as value and tuple allowed and tuple fully outside of value range
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 2), "[test13]", value_range=(3, 13), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test13]" in str(e)
    assert got_exception == True

    # list as value, but no list allowed
    got_exception = False
    try:
        result = iap.handle_discrete_param([1, 2, 3], "[test14]", value_range=None, tuple_to_uniform=True, list_to_choice=False, allow_floats=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test14]" in str(e)
    assert got_exception == True

    # list as value and list allowed
    got_exception = False
    try:
        result = iap.handle_discrete_param([1, 2, 3], "[test15]", value_range=None, tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test15]" in str(e)
    assert got_exception == False

    # list as value and list allowed and list partially outside of value range
    got_exception = False
    try:
        result = iap.handle_discrete_param([1, 3], "[test16]", value_range=(2, 13), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test16]" in str(e)
    assert got_exception == True

    # list as value and list allowed and list fully outside of value range
    got_exception = False
    try:
        result = iap.handle_discrete_param([1, 2], "[test17]", value_range=(3, 13), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test17]" in str(e)
    assert got_exception == True

    # single value within value range given as callable
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test18]", value_range=lambda x: -1 < x < 1, tuple_to_uniform=True, list_to_choice=True)
    except Exception as e:
        got_exception = True
        assert "[test18]" in str(e)
    assert got_exception == False

    # bad datatype for value range
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test19]", value_range=False, tuple_to_uniform=True, list_to_choice=True)
    except Exception as e:
        got_exception = True
        assert "Unexpected input for value_range" in str(e)
    assert got_exception == True


def test_parameters_handle_probability_param():
    for val in [True, False, 0, 1, 0.0, 1.0]:
        p = iap.handle_probability_param(val, "[test1]")
        assert isinstance(p, iap.Deterministic)
        assert p.value == int(val)

    for val in [0.0001, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999, 0.9999]:
        p = iap.handle_probability_param(val, "[test2]")
        assert isinstance(p, iap.Binomial)
        assert isinstance(p.p, iap.Deterministic)
        assert val-1e-8 < p.p.value < val+1e-8

    det = iap.Deterministic(1)
    p = iap.handle_probability_param(det, "[test3]")
    assert p == det

    got_exception = False
    try:
        p = iap.handle_probability_param("test", "[test4]")
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        p = iap.handle_probability_param(-0.01, "[test5]")
    except AssertionError:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        p = iap.handle_probability_param(1.01, "[test6]")
    except AssertionError:
        got_exception = True
    assert got_exception


def test_parameters_force_np_float_dtype():
    dtypes = [
        (np.float16, np.float16),
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.uint8, np.float64),
        (np.int32, np.float64)
    ]
    for i, (dtype_in, dtype_out) in enumerate(dtypes):
        assert iap.force_np_float_dtype(np.zeros((1,), dtype=dtype_in)).dtype == dtype_out,\
            "force_np_float_dtype() failed at %d" % (i,)


def test_parameters_both_np_float_if_one_is_float():
    a1 = np.zeros((1,), dtype=np.float16)
    b1 = np.zeros((1,), dtype=np.float32)
    a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
    assert a2.dtype.type == np.float16, a2.dtype.type
    assert b2.dtype.type == np.float32, b2.dtype.type

    a1 = np.zeros((1,), dtype=np.float16)
    b1 = np.zeros((1,), dtype=np.int32)
    a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
    assert a2.dtype.type == np.float16, a2.dtype.type
    assert b2.dtype.type == np.float64, b2.dtype.type

    a1 = np.zeros((1,), dtype=np.int32)
    b1 = np.zeros((1,), dtype=np.float16)
    a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
    assert a2.dtype.type == np.float64, a2.dtype.type
    assert b2.dtype.type == np.float16, b2.dtype.type

    a1 = np.zeros((1,), dtype=np.int32)
    b1 = np.zeros((1,), dtype=np.uint8)
    a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
    assert a2.dtype.type == np.float64, a2.dtype.type
    assert b2.dtype.type == np.float64, b2.dtype.type


def test_parameters_draw_distribution_grid():
    params = [iap.Deterministic(1), iap.Uniform(0, 1.0)]
    graph1 = params[0].draw_distribution_graph(size=(100000,))
    graph2 = params[1].draw_distribution_graph(size=(100000,))
    graph1_rs = ia.imresize_many_images(np.array([graph1]), sizes=(100, 100))[0]
    graph2_rs = ia.imresize_many_images(np.array([graph2]), sizes=(100, 100))[0]
    grid_expected = ia.draw_grid([graph1_rs, graph2_rs])

    grid_observed = iap.draw_distributions_grid(
        params,
        rows=None,
        cols=None,
        graph_sizes=(100, 100),
        sample_sizes=[(100000,), (100000,)],
        titles=None
    )

    diff = np.abs(grid_expected.astype(np.int32) - grid_observed.astype(np.int32))
    #ia.imshow(np.vstack([grid_expected, grid_observed, diff]))
    #print(diff.flatten()[0:100])
    assert np.average(diff) < 10


def test_parameters_draw_distribution_graph():
    # this test is very rough as we get a not-very-well-defined image out of the function
    param = iap.Uniform(0.0, 1.0)

    graph_img = param.draw_distribution_graph(title=None, size=(10000,), bins=100)
    assert graph_img.ndim == 3
    assert graph_img.shape[2] == 3

    # at least 10% of the image should be white-ish (background)
    nb_white = np.sum(graph_img[..., :] > [200, 200, 200])
    nb_all = np.prod(graph_img.shape)
    assert nb_white > 0.1 * nb_all

    graph_img_title = param.draw_distribution_graph(title="test", size=(10000,), bins=100)
    assert graph_img_title.ndim == 3
    assert graph_img_title.shape[2] == 3
    assert not np.array_equal(graph_img_title, graph_img)


def test_parameters_Biomial():
    reseed()
    eps = np.finfo(np.float32).eps

    param = iap.Binomial(0)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == 0
    assert np.all(samples == 0)
    assert param.__str__() == param.__repr__() == "Binomial(Deterministic(int 0))"

    param = iap.Binomial(1.0)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == 1
    assert np.all(samples == 1)
    assert param.__str__() == param.__repr__() == "Binomial(Deterministic(float 1.00000000))"

    param = iap.Binomial(0.5)
    sample = param.draw_sample()
    samples = param.draw_samples((10000))
    assert sample.shape == tuple()
    assert samples.shape == (10000,)
    assert sample in [0, 1]
    unique, counts = np.unique(samples, return_counts=True)
    assert len(unique) == 2
    for val, count in zip(unique, counts):
        if val == 0:
            assert 5000 - 500 < count < 5000 + 500
        elif val == 1:
            assert 5000 - 500 < count < 5000 + 500
        else:
            assert False

    param = iap.Binomial(iap.Choice([0.25, 0.75]))
    for _ in sm.xrange(10):
        samples = param.draw_samples((1000,))
        p = np.sum(samples) / samples.size
        assert (0.25 - 0.05 < p < 0.25 + 0.05) or (0.75 - 0.05 < p < 0.75 + 0.05)

    param = iap.Binomial((0.0, 1.0))
    last_p = 0.5
    diffs = []
    for _ in sm.xrange(30):
        samples = param.draw_samples((1000,))
        p = np.sum(samples).astype(np.float32) / samples.size
        diffs.append(abs(p - last_p))
        last_p = p
    nb_p_changed = sum([diff > 0.05 for diff in diffs])
    assert nb_p_changed > 15

    param = iap.Binomial(0.5)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)


def test_parameters_Choice():
    reseed()
    eps = np.finfo(np.float32).eps

    param = iap.Choice([0, 1, 2])
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [0, 1, 2]
    assert np.all(np.logical_or(np.logical_or(samples == 0, samples == 1), samples==2))
    assert param.__str__() == param.__repr__() == "Choice(a=[0, 1, 2], replace=True, p=None)"

    samples = param.draw_samples((10000,))
    expected = 10000/3
    expected_tolerance = expected * 0.05
    for v in [0, 1, 2]:
        count = np.sum(samples == v)
        assert expected - expected_tolerance < count < expected + expected_tolerance

    param = iap.Choice([-1, 1])
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [-1, 1]
    assert np.all(np.logical_or(samples == -1, samples == 1))

    param = iap.Choice([-1.2, 1.7])
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert -1.2 - eps < sample < -1.2 + eps or 1.7 - eps < sample < 1.7 + eps
    assert np.all(
        np.logical_or(
            np.logical_and(-1.2 - eps < samples, samples < -1.2 + eps),
            np.logical_and(1.7 - eps < samples, samples < 1.7 + eps)
        )
    )

    param = iap.Choice(["first", "second", "third"])
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in ["first", "second", "third"]
    assert np.all(
        np.logical_or(
            np.logical_or(
                samples == "first",
                samples == "second"
            ),
            samples == "third"
        )
    )

    param = iap.Choice([1+i for i in sm.xrange(100)], replace=False)
    samples = param.draw_samples((50,))
    seen = [0 for _ in sm.xrange(100)]
    for sample in samples:
        seen[sample-1] += 1
    assert all([count in [0, 1] for count in seen])

    param = iap.Choice([0, 1], p=[0.25, 0.75])
    samples = param.draw_samples((10000,))
    unique, counts = np.unique(samples, return_counts=True)
    assert len(unique) == 2
    for val, count in zip(unique, counts):
        if val == 0:
            assert 2500 - 500 < count < 2500 + 500
        elif val == 1:
            assert 7500 - 500 < count < 7500 + 500
        else:
            assert False

    param = iap.Choice([iap.Choice([0, 1]), 2])
    samples = param.draw_samples((10000,))
    unique, counts = np.unique(samples, return_counts=True)
    assert len(unique) == 3
    for val, count in zip(unique, counts):
        if val in [0, 1]:
            assert 2500 - 500 < count < 2500 + 500
        elif val == 2:
            assert 5000 - 500 < count < 5000 + 500
        else:
            assert False

    param = iap.Choice([-1, 0, 1, 2, 3])
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)

    got_exception = False
    try:
        param = iap.Choice(123)
    except Exception as exc:
        assert "Expected a to be an iterable" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        param = iap.Choice([1, 2], p=123)
    except Exception as exc:
        assert "Expected p to be" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        param = iap.Choice([1, 2], p=[1])
    except Exception as exc:
        assert "Expected lengths of" in str(exc)
        got_exception = True
    assert got_exception


def test_parameters_DiscreteUniform():
    reseed()
    eps = np.finfo(np.float32).eps

    param = iap.DiscreteUniform(0, 2)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [0, 1, 2]
    assert np.all(np.logical_or(np.logical_or(samples == 0, samples == 1), samples==2))
    assert param.__str__() == param.__repr__() == "DiscreteUniform(Deterministic(int 0), Deterministic(int 2))"

    samples = param.draw_samples((10000,))
    expected = 10000/3
    expected_tolerance = expected * 0.05
    for v in [0, 1, 2]:
        count = np.sum(samples == v)
        assert expected - expected_tolerance < count < expected + expected_tolerance

    param = iap.DiscreteUniform(-1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [-1, 0, 1]
    assert np.all(np.logical_or(np.logical_or(samples == -1, samples == 0), samples==1))

    param = iap.DiscreteUniform(-1.2, 1.2)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [-1, 0, 1]
    assert np.all(np.logical_or(np.logical_or(samples == -1, samples == 0), samples==1))

    param = iap.DiscreteUniform(1, -1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [-1, 0, 1]
    assert np.all(np.logical_or(np.logical_or(samples == -1, samples == 0), samples==1))

    param = iap.DiscreteUniform(1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((100,))
    assert sample == 1
    assert np.all(samples == 1)

    param = iap.Uniform(-1, 1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)


def test_parameters_Poisson():
    reseed()
    eps = np.finfo(np.float32).eps

    param = iap.Poisson(1)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).poisson(lam=1, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert 0 < sample
    assert param.__str__() == param.__repr__() == "Poisson(Deterministic(int 1))"

    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        count_direct = np.sum(samples_direct == i)
        count = np.sum(samples == i)
        tolerance = max(count_direct * 0.1, 250)
        assert count_direct - tolerance < count < count_direct + tolerance

    param = iap.Poisson(1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)


def test_parameters_Normal():
    reseed()

    param = iap.Normal(0, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).normal(loc=0, scale=1, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert param.__str__() == param.__repr__() == "Normal(loc=Deterministic(int 0), scale=Deterministic(int 1))"

    samples = np.clip(samples, -1, 1)
    samples_direct = np.clip(samples_direct, -1, 1)
    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(-1.0, 1.0), density=False)
    hist_direct, _ = np.histogram(samples_direct, bins=nb_bins, range=(-1.0, 1.0), density=False)
    tolerance = 0.05
    for nb_samples, nb_samples_direct in zip(hist, hist_direct):
        density = nb_samples / samples.size
        density_direct = nb_samples_direct / samples_direct.size
        assert density_direct - tolerance < density < density_direct + tolerance

    param = iap.Normal(iap.Choice([-100, 100]), 1)
    seen = [0, 0]
    for _ in sm.xrange(1000):
        samples = param.draw_samples((100,))
        exp = np.mean(samples)

        if -100 - 10 < exp < -100 + 10:
            seen[0] += 1
        elif 100 - 10 < exp < 100 + 10:
            seen[1] += 1
        else:
            assert False

    assert 500 - 100 < seen[0] < 500 + 100
    assert 500 - 100 < seen[1] < 500 + 100

    param1 = iap.Normal(0, 1)
    param2 = iap.Normal(0, 100)
    samples1 = param1.draw_samples((1000,))
    samples2 = param2.draw_samples((1000,))
    assert np.std(samples1) < np.std(samples2)
    assert 100 - 10 < np.std(samples2) < 100 + 10

    param = iap.Normal(0, 1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_Laplace():
    reseed()
    eps = np.finfo(np.float32).eps

    param = iap.Laplace(0, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).laplace(loc=0, scale=1, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert param.__str__() == param.__repr__() == "Laplace(loc=Deterministic(int 0), scale=Deterministic(int 1))"

    samples = np.clip(samples, -1, 1)
    samples_direct = np.clip(samples_direct, -1, 1)
    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(-1.0, 1.0), density=False)
    hist_direct, _ = np.histogram(samples_direct, bins=nb_bins, range=(-1.0, 1.0), density=False)
    tolerance = 0.05
    for nb_samples, nb_samples_direct in zip(hist, hist_direct):
        density = nb_samples / samples.size
        density_direct = nb_samples_direct / samples_direct.size
        assert density_direct - tolerance < density < density_direct + tolerance

    param = iap.Laplace(iap.Choice([-100, 100]), 1)
    seen = [0, 0]
    for _ in sm.xrange(1000):
        samples = param.draw_samples((100,))
        exp = np.mean(samples)

        if -100 - 10 < exp < -100 + 10:
            seen[0] += 1
        elif 100 - 10 < exp < 100 + 10:
            seen[1] += 1
        else:
            assert False

    assert 500 - 100 < seen[0] < 500 + 100
    assert 500 - 100 < seen[1] < 500 + 100

    param1 = iap.Laplace(0, 1)
    param2 = iap.Laplace(0, 100)
    samples1 = param1.draw_samples((1000,))
    samples2 = param2.draw_samples((1000,))
    assert np.var(samples1) < np.var(samples2)

    param1 = iap.Laplace(1, 0)
    samples = param1.draw_samples((100,))
    assert np.all(np.logical_and(
        samples > 1 - eps,
        samples < 1 + eps
    ))

    param = iap.Laplace(0, 1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_ChiSquare():
    reseed()

    param = iap.ChiSquare(1)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).chisquare(df=1, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert 0 <= sample
    assert np.all(0 <= samples)
    assert param.__str__() == param.__repr__() == "ChiSquare(df=Deterministic(int 1))"

    samples = np.clip(samples, 0, 3)
    samples_direct = np.clip(samples_direct, 0, 3)
    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(0, 3.0), density=False)
    hist_direct, _ = np.histogram(samples_direct, bins=nb_bins, range=(0, 3.0), density=False)
    tolerance = 0.05
    for nb_samples, nb_samples_direct in zip(hist, hist_direct):
        density = nb_samples / samples.size
        density_direct = nb_samples_direct / samples_direct.size
        assert density_direct - tolerance < density < density_direct + tolerance

    param = iap.ChiSquare(iap.Choice([1, 10]))
    seen = [0, 0]
    for _ in sm.xrange(1000):
        samples = param.draw_samples((100,))
        exp = np.mean(samples)

        if 1 - 1.0 < exp < 1 + 1.0:
            seen[0] += 1
        elif 10 - 4.0 < exp < 10 + 4.0:
            seen[1] += 1
        else:
            assert False

    assert 500 - 100 < seen[0] < 500 + 100
    assert 500 - 100 < seen[1] < 500 + 100

    param1 = iap.ChiSquare(1)
    param2 = iap.ChiSquare(10)
    samples1 = param1.draw_samples((1000,))
    samples2 = param2.draw_samples((1000,))
    assert np.var(samples1) < np.var(samples2)
    assert 2*1 - 1.0 < np.var(samples1) < 2*1 + 1.0
    assert 2*10 - 5.0 < np.var(samples2) < 2*10 + 5.0

    param = iap.ChiSquare(1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_Weibull():
    reseed()

    param = iap.Weibull(1)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).weibull(a=1, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert 0 <= sample
    assert np.all(0 <= samples)
    assert param.__str__() == param.__repr__() == "Weibull(a=Deterministic(int 1))"

    samples = np.clip(samples, 0, 2)
    samples_direct = np.clip(samples_direct, 0, 2)
    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(0, 2.0), density=False)
    hist_direct, _ = np.histogram(samples_direct, bins=nb_bins, range=(0, 2.0), density=False)
    tolerance = 0.05
    for nb_samples, nb_samples_direct in zip(hist, hist_direct):
        density = nb_samples / samples.size
        density_direct = nb_samples_direct / samples_direct.size
        assert density_direct - tolerance < density < density_direct + tolerance

    param = iap.Weibull(iap.Choice([1, 0.5]))
    expected_first = scipy.special.gamma(1 + 1/1)
    expected_second = scipy.special.gamma(1 + 1/0.5)
    seen = [0, 0]
    for _ in sm.xrange(100):
        samples = param.draw_samples((50000,))
        observed = np.mean(samples)

        if expected_first - 0.2 * expected_first < observed < expected_first + 0.2 * expected_first:
            seen[0] += 1
        elif expected_second - 0.2 * expected_second < observed < expected_second + 0.2 * expected_second:
            seen[1] += 1
        else:
            assert False

    assert 50 - 25 < seen[0] < 50 + 25
    assert 50 - 25 < seen[1] < 50 + 25

    param1 = iap.Weibull(1)
    param2 = iap.Weibull(0.5)
    samples1 = param1.draw_samples((10000,))
    samples2 = param2.draw_samples((10000,))
    assert np.var(samples1) < np.var(samples2)
    expected_first = scipy.special.gamma(1 + 2/1) - (scipy.special.gamma(1 + 1/1))**2
    expected_second = scipy.special.gamma(1 + 2/0.5) - (scipy.special.gamma(1 + 1/0.5))**2
    assert expected_first - 0.2 * expected_first < np.var(samples1) < expected_first + 0.2 * expected_first
    assert expected_second - 0.2 * expected_second < np.var(samples2) < expected_second + 0.2 * expected_second

    param = iap.Weibull(1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_Uniform():
    reseed()
    eps = np.finfo(np.float32).eps

    param = iap.Uniform(0, 1.0)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert 0 - eps < sample < 1.0 + eps
    assert np.all(np.logical_and(0 - eps < samples, samples < 1.0 + eps))
    assert param.__str__() == param.__repr__() == "Uniform(Deterministic(int 0), Deterministic(float 1.00000000))"

    samples = param.draw_samples((10000,))
    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / samples.size
        assert density_expected - density_tolerance < density < density_expected + density_tolerance

    param = iap.Uniform(-1.0, 1.0)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert -1.0 - eps < sample < 1.0 + eps
    assert np.all(np.logical_and(-1.0 - eps < samples, samples < 1.0 + eps))

    param = iap.Uniform(1.0, -1.0)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert -1.0 - eps < sample < 1.0 + eps
    assert np.all(np.logical_and(-1.0 - eps < samples, samples < 1.0 + eps))

    param = iap.Uniform(-1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert -1.0 - eps < sample < 1.0 + eps
    assert np.all(np.logical_and(-1.0 - eps < samples, samples < 1.0 + eps))

    param = iap.Uniform(1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert 1.0 - eps < sample < 1.0 + eps
    assert np.all(np.logical_and(1.0 - eps < samples, samples < 1.0 + eps))

    param = iap.Uniform(-1.0, 1.0)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_Beta():
    def _mean(alpha, beta):
        return alpha / (alpha + beta)

    def _var(alpha, beta):
        return (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

    reseed()
    eps = np.finfo(np.float32).eps

    param = iap.Beta(0.5, 0.5)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).beta(a=0.5, b=0.5, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert 0 - eps < sample < 1.0 + eps
    assert np.all(np.logical_and(0 - eps <= samples, samples <= 1.0 + eps))
    assert param.__str__() == param.__repr__() == "Beta(Deterministic(float 0.50000000), Deterministic(float 0.50000000))"

    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(0, 1.0), density=False)
    hist_direct, _ = np.histogram(samples_direct, bins=nb_bins, range=(0, 1.0), density=False)
    tolerance = 0.05
    for nb_samples, nb_samples_direct in zip(hist, hist_direct):
        density = nb_samples / samples.size
        density_direct = nb_samples_direct / samples_direct.size
        assert density_direct - tolerance < density < density_direct + tolerance

    param = iap.Beta(iap.Choice([0.5, 2]), 0.5)
    expected_first = _mean(0.5, 0.5)
    expected_second = _mean(2, 0.5)
    seen = [0, 0]
    for _ in sm.xrange(100):
        samples = param.draw_samples((10000,))
        observed = np.mean(samples)

        if expected_first - 0.05 < observed < expected_first + 0.05:
            seen[0] += 1
        elif expected_second - 0.05 < observed < expected_second + 0.05:
            seen[1] += 1
        else:
            assert False

    assert 50 - 25 < seen[0] < 50 + 25
    assert 50 - 25 < seen[1] < 50 + 25

    param1 = iap.Beta(2, 2)
    param2 = iap.Beta(0.5, 0.5)
    samples1 = param1.draw_samples((10000,))
    samples2 = param2.draw_samples((10000,))
    assert np.var(samples1) < np.var(samples2)
    expected_first = _var(2, 2)
    expected_second = _var(0.5, 0.5)
    assert expected_first - 0.1 * expected_first < np.var(samples1) < expected_first + 0.1 * expected_first
    assert expected_second - 0.1 * expected_second < np.var(samples2) < expected_second + 0.1 * expected_second

    param = iap.Beta(0.5, 0.5)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_Deterministic():
    reseed()
    eps = np.finfo(np.float32).eps

    values_int = [-100, -54, -1, 0, 1, 54, 100]
    values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

    for value in values_int:
        param = iap.Deterministic(value)

        sample1 = param.draw_sample()
        sample2 = param.draw_sample()
        assert sample1.shape == tuple()
        assert sample1 == sample2

        samples1 = param.draw_samples(10)
        samples2 = param.draw_samples(10)
        samples3 = param.draw_samples((5, 3))
        samples4 = param.draw_samples((5, 3))
        samples5 = param.draw_samples((4, 5, 3))
        samples6 = param.draw_samples((4, 5, 3))

        samples1_unique = np.unique(samples1)
        samples2_unique = np.unique(samples2)
        samples3_unique = np.unique(samples3)
        samples4_unique = np.unique(samples4)
        samples5_unique = np.unique(samples5)
        samples6_unique = np.unique(samples6)

        assert samples1.shape == (10,)
        assert samples2.shape == (10,)
        assert samples3.shape == (5, 3)
        assert samples4.shape == (5, 3)
        assert samples5.shape == (4, 5, 3)
        assert samples6.shape == (4, 5, 3)
        assert len(samples1_unique) == 1 and samples1_unique[0] == value
        assert len(samples2_unique) == 1 and samples2_unique[0] == value
        assert len(samples3_unique) == 1 and samples3_unique[0] == value
        assert len(samples4_unique) == 1 and samples4_unique[0] == value
        assert len(samples5_unique) == 1 and samples5_unique[0] == value
        assert len(samples6_unique) == 1 and samples6_unique[0] == value

        rs1 = np.random.RandomState(123456)
        rs2 = np.random.RandomState(123456)
        assert np.array_equal(
            param.draw_samples(20, random_state=rs1),
            param.draw_samples(20, random_state=rs2)
        )

    for value in values_float:
        param = iap.Deterministic(value)

        sample1 = param.draw_sample()
        sample2 = param.draw_sample()
        assert sample1.shape == tuple()
        assert sample1 - eps < sample2 < sample1 + eps

        samples1 = param.draw_samples(10)
        samples2 = param.draw_samples(10)
        samples3 = param.draw_samples((5, 3))
        samples4 = param.draw_samples((5, 3))
        samples5 = param.draw_samples((4, 5, 3))
        samples6 = param.draw_samples((4, 5, 3))

        samples1_sorted = np.sort(samples1)
        samples2_sorted = np.sort(samples2)
        samples3_sorted = np.sort(samples3.flatten())
        samples4_sorted = np.sort(samples4.flatten())
        samples5_sorted = np.sort(samples5.flatten())
        samples6_sorted = np.sort(samples6.flatten())

        assert samples1.shape == (10,)
        assert samples2.shape == (10,)
        assert samples3.shape == (5, 3)
        assert samples4.shape == (5, 3)
        assert samples5.shape == (4, 5, 3)
        assert samples6.shape == (4, 5, 3)
        assert samples1_sorted[0] - eps < samples1_sorted[-1] < samples1_sorted[0] + eps
        assert samples2_sorted[0] - eps < samples2_sorted[-1] < samples2_sorted[0] + eps
        assert samples3_sorted[0] - eps < samples3_sorted[-1] < samples3_sorted[0] + eps
        assert samples4_sorted[0] - eps < samples4_sorted[-1] < samples4_sorted[0] + eps
        assert samples5_sorted[0] - eps < samples5_sorted[-1] < samples5_sorted[0] + eps
        assert samples6_sorted[0] - eps < samples6_sorted[-1] < samples6_sorted[0] + eps

        rs1 = np.random.RandomState(123456)
        rs2 = np.random.RandomState(123456)
        assert np.allclose(
            param.draw_samples(20, random_state=rs1),
            param.draw_samples(20, random_state=rs2)
        )

    param = iap.Deterministic(0)
    assert param.__str__() == param.__repr__() == "Deterministic(int 0)"
    param = iap.Deterministic(1.0)
    assert param.__str__() == param.__repr__() == "Deterministic(float 1.00000000)"
    param = iap.Deterministic("test")
    assert param.__str__() == param.__repr__() == "Deterministic(test)"

    seen = [0, 0]
    for _ in sm.xrange(200):
        param = iap.Deterministic(iap.Choice([0, 1]))
        seen[param.value] += 1
    assert 100 - 50 < seen[0] < 100 + 50
    assert 100 - 50 < seen[1] < 100 + 50

    got_exception = False
    try:
        param = iap.Deterministic([1, 2, 3])
    except Exception as exc:
        assert "Expected StochasticParameter object or number or string" in str(exc)
        got_exception = True
    assert got_exception


def test_parameters_FromLowerResolution():
    reseed()
    eps = np.finfo(np.float32).eps

    # (H, W, C)
    param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=8)
    samples = param.draw_samples((8, 8, 1))
    assert samples.shape == (8, 8, 1)
    uq = np.unique(samples)
    assert len(uq) == 2 and (0 in uq or 1 in uq)

    # (N, H, W, C)
    samples_nhwc = param.draw_samples((1, 8, 8, 1))
    assert samples_nhwc.shape == (1, 8, 8, 1)
    uq = np.unique(samples_nhwc)
    assert len(uq) == 2 and (0 in uq or 1 in uq)

    # (N, H, W, C, something) causing error
    got_exception = False
    try:
        samples_nhwcx = param.draw_samples((1, 8, 8, 1, 1))
    except Exception as exc:
        assert "FromLowerResolution can only generate samples of shape" in str(exc)
        got_exception = True
    assert got_exception

    # C=3
    param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=8)
    samples = param.draw_samples((8, 8, 3))
    assert samples.shape == (8, 8, 3)
    uq = np.unique(samples)
    assert len(uq) == 2 and (0 in uq or 1 in uq)

    # different sizes in px
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=16)
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(100):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # different sizes in px, one given as tuple (a, b)
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=(2, 16))
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(400):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # different sizes in px, given as StochasticParameter
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=iap.Deterministic(1))
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=iap.Choice([8, 16]))
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(100):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # bad datatype for size_px
    got_exception = False
    try:
        param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # min_size
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=1, min_size=16)
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(100):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # different sizes in percent
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=0.01)
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=0.8)
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(100):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # different sizes in percent, given as StochasticParameter
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=iap.Deterministic(0.01))
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=iap.Choice([0.4, 0.8]))
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(100):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # bad datatype for size_percent
    got_exception = False
    try:
        param = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # method given as StochasticParameter
    param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=4, method=iap.Choice(["nearest", "linear"]))
    seen = [0, 0]
    for _ in sm.xrange(200):
        samples = param.draw_samples((16, 16, 1))
        nb_in_between = np.sum(np.logical_and(samples < 0.95, samples > 0.05))
        if nb_in_between == 0:
            seen[0] += 1
        else:
            seen[1] += 1
    assert 100 - 50 < seen[0] < 100 + 50
    assert 100 - 50 < seen[1] < 100 + 50

    # bad datatype for method
    got_exception = False
    try:
        param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=4, method=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # multiple calls with same random_state
    param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
    samples1 = param.draw_samples((10, 5, 1), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5, 1), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)

    # str / repr
    param = iap.FromLowerResolution(other_param=iap.Deterministic(0), size_percent=1, method="nearest")
    assert param.__str__() == param.__repr__() == "FromLowerResolution(size_percent=Deterministic(int 1), method=Deterministic(nearest), other_param=Deterministic(int 0))"
    param = iap.FromLowerResolution(other_param=iap.Deterministic(0), size_px=1, method="nearest")
    assert param.__str__() == param.__repr__() == "FromLowerResolution(size_px=Deterministic(int 1), method=Deterministic(nearest), other_param=Deterministic(int 0))"


def test_parameters_Clip():
    reseed()
    eps = np.finfo(np.float32).eps

    param = iap.Clip(iap.Deterministic(0), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == 0
    assert np.all(samples == 0)
    assert param.__str__() == param.__repr__() == "Clip(Deterministic(int 0), -1.000000, 1.000000)"

    param = iap.Clip(iap.Deterministic(1), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == 1
    assert np.all(samples == 1)

    param = iap.Clip(iap.Deterministic(-1), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == -1
    assert np.all(samples == -1)

    param = iap.Clip(iap.Deterministic(0.5), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert 0.5 - eps < sample < 0.5 + eps
    assert np.all(np.logical_and(0.5 - eps < samples, samples < 0.5 + eps))

    param = iap.Clip(iap.Deterministic(2), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == 1
    assert np.all(samples == 1)

    param = iap.Clip(iap.Deterministic(-2), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == -1
    assert np.all(samples == -1)

    param = iap.Clip(iap.Choice([0, 2]), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [0, 1]
    assert np.all(np.logical_or(samples == 0, samples == 1))

    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)

    param = iap.Clip(iap.Deterministic(0), None, 1)
    sample = param.draw_sample()
    assert sample == 0
    assert param.__str__() == param.__repr__() == "Clip(Deterministic(int 0), None, 1.000000)"

    param = iap.Clip(iap.Deterministic(0), 0, None)
    sample = param.draw_sample()
    assert sample == 0
    assert param.__str__() == param.__repr__() == "Clip(Deterministic(int 0), 0.000000, None)"

    param = iap.Clip(iap.Deterministic(0), None, None)
    sample = param.draw_sample()
    assert sample == 0
    assert param.__str__() == param.__repr__() == "Clip(Deterministic(int 0), None, None)"


def test_parameters_Discretize():
    reseed()
    eps = np.finfo(np.float32).eps

    values = [-100.2, -54.3, -1.0, -1, -0.7, -0.00043, 0, 0.00043, 0.7, 1.0, 1, 54.3, 100.2]
    for value in values:
        value_expected = np.round(np.float64([value])).astype(np.int32)[0]
        param = iap.Discretize(iap.Deterministic(value))
        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))
        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample == value_expected
        assert np.all(samples == value_expected)

    param_orig = iap.DiscreteUniform(0, 1)
    param = iap.Discretize(param_orig)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [0, 1]
    assert np.all(np.logical_or(samples == 0, samples == 1))

    param_orig = iap.DiscreteUniform(0, 2)
    param = iap.Discretize(param_orig)
    samples1 = param_orig.draw_samples((10000,))
    samples2 = param.draw_samples((10000,))
    assert np.all(np.abs(samples1 - samples2) < 0.2*(10000/3))

    param_orig = iap.DiscreteUniform(0, 2)
    param = iap.Discretize(param_orig)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)

    param = iap.Discretize(iap.Deterministic(0))
    assert param.__str__() == param.__repr__() == "Discretize(Deterministic(int 0))"


def test_parameters_Multiply():
    reseed()
    eps = np.finfo(np.float32).eps

    values_int = [-100, -54, -1, 0, 1, 54, 100]
    values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

    for v1 in values_int:
        for v2 in values_int:
            p = iap.Multiply(iap.Deterministic(v1), v2)
            assert p.draw_sample() == v1 * v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.int64
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 * v2)

            p = iap.Multiply(iap.Deterministic(v1), iap.Deterministic(v2))
            assert p.draw_sample() == v1 * v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.int64
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 * v2)

    for v1 in values_float:
        for v2 in values_float:
            p = iap.Multiply(iap.Deterministic(v1), v2)
            assert v1 * v2 - eps < p.draw_sample() < v1 * v2 + eps
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 * v2)

            p = iap.Multiply(iap.Deterministic(v1), iap.Deterministic(v2))
            assert v1 * v2 - eps < p.draw_sample() < v1 * v2 + eps
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 * v2)

    param = iap.Multiply(iap.Deterministic(1.0), (1.0, 2.0), elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 * 1.0 - eps)
    assert np.all(samples < 1.0 * 2.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps

    param = iap.Multiply(iap.Deterministic(1.0), (1.0, 2.0), elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 * 1.0 - eps)
    assert np.all(samples < 1.0 * 2.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Multiply(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 * 1.0 - eps)
    assert np.all(samples < 2.0 * 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Multiply(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 * 1.0 - eps)
    assert np.all(samples < 2.0 * 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Multiply(iap.Deterministic(0), 1, elementwise=False)
    assert param.__str__() == param.__repr__() == "Multiply(Deterministic(int 0), Deterministic(int 1), False)"


def test_parameters_Divide():
    reseed()
    eps = np.finfo(np.float32).eps

    values_int = [-100, -54, -1, 0, 1, 54, 100]
    values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

    for v1 in values_int:
        for v2 in values_int:
            if v2 == 0:
                v2 = 1

            p = iap.Divide(iap.Deterministic(v1), v2)
            assert p.draw_sample() == v1 / v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.float64) + v1 / v2)

            p = iap.Divide(iap.Deterministic(v1), iap.Deterministic(v2))
            assert p.draw_sample() == v1 / v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.float64) + v1 / v2)

    for v1 in values_float:
        for v2 in values_float:
            if v2 == 0:
                v2 = 1

            p = iap.Divide(iap.Deterministic(v1), v2)
            assert v1 / v2 - eps < p.draw_sample() < v1 / v2 + eps
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 / v2)

            p = iap.Divide(iap.Deterministic(v1), iap.Deterministic(v2))
            assert v1 / v2 - eps < p.draw_sample() < v1 / v2 + eps
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 / v2)

    param = iap.Divide(iap.Deterministic(1.0), (1.0, 2.0), elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 / 2.0 - eps)
    assert np.all(samples < 1.0 / 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps

    param = iap.Divide(iap.Deterministic(1.0), (1.0, 2.0), elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 / 2.0 - eps)
    assert np.all(samples < 1.0 / 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Divide(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 / 1.0 - eps)
    assert np.all(samples < 2.0 / 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Divide(iap.Deterministic(1), 0, elementwise=False)
    sample = param.draw_sample()
    assert sample == 1

    param = iap.Divide(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 / 1.0 - eps)
    assert np.all(samples < 2.0 / 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    # test division by zero automatically being converted to division by 1
    param = iap.Divide(2, iap.Choice([0, 2]), elementwise=True)
    samples = param.draw_samples((10, 20))
    samples_unique = np.sort(np.unique(samples.flatten()))
    assert samples_unique[0] == 1 and samples_unique[1] == 2

    param = iap.Divide(iap.Deterministic(0), 1, elementwise=False)
    assert param.__str__() == param.__repr__() == "Divide(Deterministic(int 0), Deterministic(int 1), False)"


def test_parameters_Add():
    reseed()
    eps = np.finfo(np.float32).eps

    values_int = [-100, -54, -1, 0, 1, 54, 100]
    values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

    for v1 in values_int:
        for v2 in values_int:
            p = iap.Add(iap.Deterministic(v1), v2)
            assert p.draw_sample() == v1 + v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.int64
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 + v2)

            p = iap.Add(iap.Deterministic(v1), iap.Deterministic(v2))
            assert p.draw_sample() == v1 + v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.int64
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 + v2)

    for v1 in values_float:
        for v2 in values_float:
            p = iap.Add(iap.Deterministic(v1), v2)
            assert v1 + v2 - eps < p.draw_sample() < v1 + v2 + eps
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 + v2)

            p = iap.Add(iap.Deterministic(v1), iap.Deterministic(v2))
            assert v1 + v2 - eps < p.draw_sample() < v1 + v2 + eps
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 + v2)

    param = iap.Add(iap.Deterministic(1.0), (1.0, 2.0), elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 + 1.0 - eps)
    assert np.all(samples < 1.0 + 2.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps

    param = iap.Add(iap.Deterministic(1.0), (1.0, 2.0), elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 + 1.0 - eps)
    assert np.all(samples < 1.0 + 2.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Add(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 + 1.0 - eps)
    assert np.all(samples < 2.0 + 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Add(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 + 1.0 - eps)
    assert np.all(samples < 2.0 + 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Add(iap.Deterministic(0), 1, elementwise=False)
    assert param.__str__() == param.__repr__() == "Add(Deterministic(int 0), Deterministic(int 1), False)"


def test_parameters_Subtract():
    reseed()
    eps = np.finfo(np.float32).eps

    values_int = [-100, -54, -1, 0, 1, 54, 100]
    values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

    for v1 in values_int:
        for v2 in values_int:
            p = iap.Subtract(iap.Deterministic(v1), v2)
            assert p.draw_sample() == v1 - v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.int64
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 - v2)

            p = iap.Subtract(iap.Deterministic(v1), iap.Deterministic(v2))
            assert p.draw_sample() == v1 - v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.int64
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 - v2)

    for v1 in values_float:
        for v2 in values_float:
            p = iap.Subtract(iap.Deterministic(v1), v2)
            assert v1 - v2 - eps < p.draw_sample() < v1 - v2 + eps
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 - v2)

            p = iap.Subtract(iap.Deterministic(v1), iap.Deterministic(v2))
            assert v1 - v2 - eps < p.draw_sample() < v1 - v2 + eps
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 - v2)

    param = iap.Subtract(iap.Deterministic(1.0), (1.0, 2.0), elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 - 2.0 - eps)
    assert np.all(samples < 1.0 - 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps

    param = iap.Subtract(iap.Deterministic(1.0), (1.0, 2.0), elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 - 2.0 - eps)
    assert np.all(samples < 1.0 - 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Subtract(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 - 1.0 - eps)
    assert np.all(samples < 2.0 - 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Subtract(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 - 1.0 - eps)
    assert np.all(samples < 2.0 - 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Subtract(iap.Deterministic(0), 1, elementwise=False)
    assert param.__str__() == param.__repr__() == "Subtract(Deterministic(int 0), Deterministic(int 1), False)"


def test_parameters_Power():
    reseed()
    eps = np.finfo(np.float32).eps

    values = [-100, -54, -1, 0, 1, 54, 100]
    values = values + [float(v) for v in values]
    exponents = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    for v1 in values:
        for v2 in exponents:
            if v1 < 0 and ia.is_single_float(v2):
                continue
            if v1 == 0 and v2 < 0:
                continue
            p = iap.Power(iap.Deterministic(v1), v2)
            assert v1 ** v2 - eps < p.draw_sample() < v1 ** v2 + eps
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 ** v2)

            p = iap.Power(iap.Deterministic(v1), iap.Deterministic(v2))
            assert v1 ** v2 - eps < p.draw_sample() < v1 ** v2 + eps
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.float64
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 ** v2)

    param = iap.Power(iap.Deterministic(1.5), (1.0, 2.0), elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.5 ** 1.0 - eps)
    assert np.all(samples < 1.5 ** 2.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps

    param = iap.Power(iap.Deterministic(1.5), (1.0, 2.0), elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.5 ** 1.0 - eps)
    assert np.all(samples < 1.5 ** 2.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Power(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 ** 1.0 - eps)
    assert np.all(samples < 2.0 ** 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Power(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 ** 1.0 - eps)
    assert np.all(samples < 2.0 ** 1.0 + eps)
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - eps < samples_sorted[-1] < samples_sorted[0] + eps)

    param = iap.Power(iap.Deterministic(0), 1, elementwise=False)
    assert param.__str__() == param.__repr__() == "Power(Deterministic(int 0), Deterministic(int 1), False)"


def test_parameters_Absolute():
    reseed()
    eps = np.finfo(np.float32).eps

    simple_values = [-1.5, -1, -1.0, -0.1, 0, 0.0, 0.1, 1, 1.0, 1.5]

    for value in simple_values:
        param = iap.Absolute(iap.Deterministic(value))
        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))
        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        if ia.is_single_float(value):
            assert abs(value) - eps < sample < abs(value) + eps
            assert np.all(abs(value) - eps < samples)
            assert np.all(samples < abs(value) + eps)
        else:
            assert sample == abs(value)
            assert np.all(samples == abs(value))

    param = iap.Absolute(iap.Choice([-3, -1, 1, 3]))
    sample = param.draw_sample()
    samples = param.draw_samples((10, 10))
    samples_uq = np.sort(np.unique(samples))
    assert sample.shape == tuple()
    assert sample in [3, 1]
    assert samples.shape == (10, 10)
    assert len(samples_uq) == 2
    assert samples_uq[0] == 1 and samples_uq[1] == 3

    param = iap.Absolute(iap.Deterministic(0))
    assert param.__str__() == param.__repr__() == "Absolute(Deterministic(int 0))"


def test_parameters_RandomSign():
    reseed()

    param = iap.RandomSign(iap.Deterministic(1))
    samples = param.draw_samples((1000,))
    n_positive = np.sum(samples == 1)
    n_negative = np.sum(samples == -1)
    assert samples.shape == (1000,)
    assert n_positive + n_negative == 1000
    assert 350 < n_positive < 750

    seen = [0, 0]
    for _ in sm.xrange(1000):
        sample = param.draw_sample()
        assert sample.shape == tuple()
        if sample == 1:
            seen[1] += 1
        else:
            seen[0] += 1
    n_negative, n_positive = seen
    assert n_positive + n_negative == 1000
    assert 350 < n_positive < 750

    param = iap.RandomSign(iap.Choice([1, 2]))
    samples = param.draw_samples((4000,))
    seen = [0, 0, 0, 0]
    seen[0] = np.sum(samples == -2)
    seen[1] = np.sum(samples == -1)
    seen[2] = np.sum(samples == 1)
    seen[3] = np.sum(samples == 2)
    assert np.sum(seen) == 4000
    assert all([700 < v < 1300 for v in seen])

    param = iap.RandomSign(iap.Choice([1, 2]))
    samples1 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    assert samples1.shape == (100, 10)
    assert samples2.shape == (100, 10)
    assert np.array_equal(samples1, samples2)
    assert np.sum(samples == -2) > 50
    assert np.sum(samples == -1) > 50
    assert np.sum(samples == 1) > 50
    assert np.sum(samples == 2) > 50

    param = iap.RandomSign(iap.Deterministic(0), 0.5)
    assert param.__str__() == param.__repr__() == "RandomSign(Deterministic(int 0), 0.50)"


def test_parameters_ForceSign():
    reseed()

    param = iap.ForceSign(iap.Deterministic(1), positive=True, mode="invert")
    sample = param.draw_sample()
    assert sample.shape == tuple()
    assert sample == 1

    param = iap.ForceSign(iap.Deterministic(1), positive=False, mode="invert")
    sample = param.draw_sample()
    assert sample.shape == tuple()
    assert sample == -1

    param = iap.ForceSign(iap.Deterministic(1), positive=True, mode="invert")
    samples = param.draw_samples(100)
    assert samples.shape == (100,)
    assert np.all(samples == 1)

    param = iap.ForceSign(iap.Deterministic(1), positive=False, mode="invert")
    samples = param.draw_samples(100)
    assert samples.shape == (100,)
    assert np.all(samples == -1)

    param = iap.ForceSign(iap.Deterministic(-1), positive=True, mode="invert")
    samples = param.draw_samples(100)
    assert samples.shape == (100,)
    assert np.all(samples == 1)

    param = iap.ForceSign(iap.Deterministic(-1), positive=False, mode="invert")
    samples = param.draw_samples(100)
    assert samples.shape == (100,)
    assert np.all(samples == -1)

    param = iap.ForceSign(iap.Choice([-2, 1]), positive=True, mode="invert")
    samples = param.draw_samples(1000)
    assert samples.shape == (1000,)
    n_twos = np.sum(samples == 2)
    n_ones = np.sum(samples == 1)
    assert n_twos + n_ones == 1000
    assert 200 < n_twos < 700
    assert 200 < n_ones < 700

    param = iap.ForceSign(iap.Choice([-2, 1]), positive=True, mode="reroll")
    samples = param.draw_samples(1000)
    assert samples.shape == (1000,)
    n_twos = np.sum(samples == 2)
    n_ones = np.sum(samples == 1)
    assert n_twos + n_ones == 1000
    assert n_twos > 0
    assert n_ones > 0

    param = iap.ForceSign(iap.Choice([-2, 1]), positive=True, mode="reroll", reroll_count_max=100)
    samples = param.draw_samples(100)
    assert samples.shape == (100,)
    n_twos = np.sum(samples == 2)
    n_ones = np.sum(samples == 1)
    assert n_twos + n_ones == 100
    assert n_twos < 5

    param = iap.ForceSign(iap.Choice([-2, 1]), positive=True, mode="invert")
    samples1 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    assert samples1.shape == (100, 10)
    assert samples2.shape == (100, 10)
    assert np.array_equal(samples1, samples2)

    param = iap.ForceSign(iap.Deterministic(0), True, "invert", 1)
    assert param.__str__() == param.__repr__() == "ForceSign(Deterministic(int 0), True, invert, 1)"


def test_parameters_Positive():
    reseed()

    param = iap.Positive(iap.Deterministic(-1), mode="reroll", reroll_count_max=1)
    samples = param.draw_samples((100,))
    assert samples.shape == (100,)
    assert np.all(samples == 1)


def test_parameters_Negative():
    reseed()

    param = iap.Negative(iap.Deterministic(1), mode="reroll", reroll_count_max=1)
    samples = param.draw_samples((100,))
    assert samples.shape == (100,)
    assert np.all(samples == -1)


def test_parameters_IterativeNoiseAggregator():
    reseed()
    eps = np.finfo(np.float32).eps

    param = iap.IterativeNoiseAggregator(iap.Deterministic(1), iterations=1, aggregation_method="max")
    sample = param.draw_sample()
    samples = param.draw_samples((2, 4))
    assert sample.shape == tuple()
    assert samples.shape == (2, 4)
    assert sample == 1
    assert np.all(samples == 1)

    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=200, aggregation_method="avg")
    sample = param.draw_sample()
    samples = param.draw_samples((2, 4))
    assert sample.shape == tuple()
    assert samples.shape == (2, 4)
    assert 25 - 10 < sample < 25 + 10
    assert np.all(np.logical_and(25 - 10 < samples, samples < 25 + 10))

    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=100, aggregation_method="max")
    sample = param.draw_sample()
    samples = param.draw_samples((2, 4))
    assert sample.shape == tuple()
    assert samples.shape == (2, 4)
    assert sample == 50
    assert np.all(samples == 50)

    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=100, aggregation_method="min")
    sample = param.draw_sample()
    samples = param.draw_samples((2, 4))
    assert sample.shape == tuple()
    assert samples.shape == (2, 4)
    assert sample == 0
    assert np.all(samples == 0)

    seen = [0, 0, 0]
    for _ in sm.xrange(100):
        param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=100, aggregation_method=["avg", "max"])
        samples = param.draw_samples((1, 1))
        diff_0 = abs(0 - samples[0, 0])
        diff_25 = abs(25 - samples[0, 0])
        diff_50 = abs(50 - samples[0, 0])
        if diff_25 < 10.0:
            seen[0] += 1
        elif diff_50 < eps:
            seen[1] += 1
        elif diff_0 < eps:
            seen[2] += 1
        else:
            assert False
    assert seen[2] < 5
    assert 50 - 20 < seen[0] < 50 + 20
    assert 50 - 20 < seen[1] < 50 + 20

    # iterations as tuple
    param = iap.IterativeNoiseAggregator(iap.Uniform(-1.0, 1.0), iterations=(1, 100), aggregation_method="avg")
    diffs = []
    for _ in sm.xrange(100):
        samples = param.draw_samples((1, 1))
        diff = abs(samples[0, 0] - 0.0)
        diffs.append(diff)

    nb_bins = 3
    nb_iterations = 100
    hist, _ = np.histogram(diffs, bins=nb_bins, range=(-1.0, 1.0), density=False)
    #density_expected = 1.0/nb_bins
    #density_tolerance = 0.1
    #for nb_samples in hist:
    #    density = nb_samples / nb_iterations
    #    print(hist, nb_samples, nb_iterations, density)
    #    assert density_expected - density_tolerance < density < density_expected + density_tolerance
    assert hist[1] > hist[0]
    assert hist[1] > hist[2]

    # iterations as list
    seen = [0, 0]
    for _ in sm.xrange(400):
        param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=[1, 100], aggregation_method=["max"])
        samples = param.draw_samples((1, 1))
        diff_0 = abs(0 - samples[0, 0])
        diff_50 = abs(50 - samples[0, 0])
        if diff_50 < eps:
            seen[0] += 1
        elif diff_0 < eps:
            seen[1] += 1
        else:
            assert False
    assert 300 - 50 < seen[0] < 300 + 50
    assert 100 - 50 < seen[1] < 100 + 50

    # test ia.ALL as aggregation_method
    # note that each method individually and list of methods are already tested, so no in depth
    # test is needed here
    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=100, aggregation_method=ia.ALL)
    assert isinstance(param.aggregation_method, iap.Choice)
    assert len(param.aggregation_method.a) == 3
    assert [v in param.aggregation_method.a for v in ["min", "avg", "max"]]

    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=2, aggregation_method="max")
    samples = param.draw_samples((2, 1000))
    nb_0 = np.sum(samples == 0)
    nb_50 = np.sum(samples == 50)
    assert nb_0 + nb_50 == 2 * 1000
    assert 0.25 - 0.05 < nb_0 / (2 * 1000) < 0.25 + 0.05

    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=5, aggregation_method="avg")
    samples1 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    assert samples1.shape == (100, 10)
    assert samples2.shape == (100, 10)
    assert np.allclose(samples1, samples2)

    # StochasticParameter as aggregation_method
    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=5, aggregation_method=iap.Deterministic("max"))
    assert isinstance(param.aggregation_method, iap.Deterministic)
    assert param.aggregation_method.value == "max"

    # bad datatype as aggregation_method
    got_exception = False
    try:
        param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=5, aggregation_method=False)
    except Exception as exc:
        assert "Expected aggregation_method to be" in str(exc)
        got_exception = True
    assert got_exception

    # bad datatype as for iterations
    got_exception = False
    try:
        param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=False, aggregation_method="max")
    except Exception as exc:
        assert "Expected iterations to be" in str(exc)
        got_exception = True
    assert got_exception

    param = iap.IterativeNoiseAggregator(iap.Deterministic(0), iterations=(1, 3), aggregation_method="max")
    assert param.__str__() == param.__repr__() == "IterativeNoiseAggregator(Deterministic(int 0), DiscreteUniform(Deterministic(int 1), Deterministic(int 3)), Deterministic(max))"


def test_parameters_Sigmoid():
    reseed()
    eps = np.finfo(np.float32).eps

    param = iap.Sigmoid(iap.Deterministic(5), add=0, mul=1, threshold=0.5, activated=True)
    expected = 1 / (1 + np.exp(-(5 * 1 + 0 - 0.5)))
    sample = param.draw_sample()
    samples = param.draw_samples((5, 10))
    assert sample.shape == tuple()
    assert samples.shape == (5, 10)
    assert expected - eps < sample < expected + eps
    assert np.all(np.logical_and(expected - eps < samples, samples < expected + eps))

    param = iap.Sigmoid(iap.Deterministic(5), add=0, mul=1, threshold=0.5, activated=False)
    expected = 5
    sample = param.draw_sample()
    samples = param.draw_samples((5, 10))
    assert sample.shape == tuple()
    assert samples.shape == (5, 10)
    assert expected - eps < sample < expected + eps
    assert np.all(np.logical_and(expected - eps < samples, samples < expected + eps))

    param = iap.Sigmoid(iap.Deterministic(5), add=0, mul=1, threshold=0.5, activated=0.5)
    expected_first = 5
    expected_second = 1 / (1 + np.exp(-(5 * 1 + 0 - 0.5)))
    seen = [0, 0]
    for _ in sm.xrange(1000):
        sample = param.draw_sample()
        diff_first = abs(sample - expected_first)
        diff_second = abs(sample - expected_second)
        if diff_first < eps:
            seen[0] += 1
        elif diff_second < eps:
            seen[1] += 1
        else:
            assert False
    assert 500 - 150 < seen[0] < 500 + 150
    assert 500 - 150 < seen[1] < 500 + 150

    param = iap.Sigmoid(iap.Choice([1, 10]), add=0, mul=1, threshold=0.5, activated=True)
    expected_first = 1 / (1 + np.exp(-(1 * 1 + 0 - 0.5)))
    expected_second = 1 / (1 + np.exp(-(10 * 1 + 0 - 0.5)))
    seen = [0, 0]
    for _ in sm.xrange(1000):
        sample = param.draw_sample()
        diff_first = abs(sample - expected_first)
        diff_second = abs(sample - expected_second)
        if diff_first < eps:
            seen[0] += 1
        elif diff_second < eps:
            seen[1] += 1
        else:
            assert False
    assert 500 - 150 < seen[0] < 500 + 150
    assert 500 - 150 < seen[1] < 500 + 150

    muls = [0.1, 1, 10.3]
    adds = [-5.7, -1, -0.0734, 0, 0.0734, 1, 5.7]
    vals = [-1, -0.7, 0, 0.7, 1]
    threshs = [-5.7, -1, -0.0734, 0, 0.0734, 1, 5.7]
    for mul in muls:
        for add in adds:
            for val in vals:
                for thresh in threshs:
                    param = iap.Sigmoid(iap.Deterministic(val), add=add, mul=mul, threshold=thresh)
                    sample = param.draw_sample()
                    samples = param.draw_samples((2, 3))
                    assert sample.shape == tuple()
                    assert samples.shape == (2, 3)
                    expected = 1 / (1 + np.exp(-(val * mul + add - thresh)))
                    assert expected - eps < sample < expected + eps
                    assert np.all(np.logical_and(expected - eps < samples, samples < expected + eps))

    param = iap.Sigmoid(iap.Choice([1, 10]), add=0, mul=1, threshold=0.5, activated=True)
    samples1 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    assert samples1.shape == (100, 10)
    assert samples2.shape == (100, 10)
    assert np.array_equal(samples1, samples2)

    param = iap.Sigmoid(iap.Deterministic(0), threshold=(-10, 10), activated=True, mul=1, add=0)
    assert param.__str__() == param.__repr__() == "Sigmoid(Deterministic(int 0), Uniform(Deterministic(int -10), Deterministic(int 10)), Deterministic(int 1), 1, 0)"


def test_parameters_operators():
    reseed()

    param1 = iap.Normal(0, 1)
    param2 = iap.Uniform(-1.0, 1.0)

    # Multiply
    param3 = param1 * param2
    assert isinstance(param3, iap.Multiply)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1 * 2
    assert isinstance(param3, iap.Multiply)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    param3 = 2 * param1
    assert isinstance(param3, iap.Multiply)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        param3 = "test" * param1
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        param3 = param1 * "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Divide (__truediv__)
    param3 = param1 / param2
    assert isinstance(param3, iap.Divide)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1 / 2
    assert isinstance(param3, iap.Divide)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    param3 = 2 / param1
    assert isinstance(param3, iap.Divide)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        param3 = "test" / param1
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        param3 = param1 / "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Divide (__div__)
    param3 = param1.__div__(param2)
    assert isinstance(param3, iap.Divide)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1.__div__(2)
    assert isinstance(param3, iap.Divide)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    got_exception = False
    try:
        param3 = param1.__div__("test")
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Divide (__rdiv__)
    param3 = param1.__rdiv__(2)
    assert isinstance(param3, iap.Divide)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        param3 = param1.__rdiv__("test")
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Divide (__floordiv__)
    param1_int = iap.DiscreteUniform(0, 10)
    param2_int = iap.Choice([1, 2])
    param3 = param1_int // param2_int
    assert isinstance(param3, iap.Discretize)
    assert isinstance(param3.other_param, iap.Divide)
    assert param3.other_param.other_param == param1_int
    assert param3.other_param.val == param2_int

    param3 = param1_int // 2
    assert isinstance(param3, iap.Discretize)
    assert isinstance(param3.other_param, iap.Divide)
    assert param3.other_param.other_param == param1_int
    assert isinstance(param3.other_param.val, iap.Deterministic)
    assert param3.other_param.val.value == 2

    param3 = 2 // param1_int
    assert isinstance(param3, iap.Discretize)
    assert isinstance(param3.other_param, iap.Divide)
    assert isinstance(param3.other_param.other_param, iap.Deterministic)
    assert param3.other_param.other_param.value == 2
    assert param3.other_param.val == param1_int

    got_exception = False
    try:
        param3 = "test" // param1_int
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        param3 = param1_int // "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Add
    param3 = param1 + param2
    assert isinstance(param3, iap.Add)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1 + 2
    assert isinstance(param3, iap.Add)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    param3 = 2 + param1
    assert isinstance(param3, iap.Add)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        param3 = "test" + param1
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        param3 = param1 + "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Subtract
    param3 = param1 - param2
    assert isinstance(param3, iap.Subtract)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1 - 2
    assert isinstance(param3, iap.Subtract)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    param3 = 2 - param1
    assert isinstance(param3, iap.Subtract)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        param3 = "test" - param1
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        param3 = param1 - "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Power
    param3 = param1 ** param2
    assert isinstance(param3, iap.Power)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1 ** 2
    assert isinstance(param3, iap.Power)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    param3 = 2 ** param1
    assert isinstance(param3, iap.Power)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        param3 = "test" ** param1
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        param3 = param1 ** "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception


def test_parameters_copy():
    reseed()
    other_param = iap.Uniform(1.0, 10.0)
    param = iap.Discretize(other_param)
    other_param.a = [1.0]
    param_copy = param.copy()
    assert isinstance(param_copy, iap.Discretize)
    assert isinstance(param_copy.other_param, iap.Uniform)
    param.other_param.a[0] += 1
    assert param_copy.other_param.a[0] == param.other_param.a[0]

    other_param = iap.Uniform(1.0, 10.0)
    param = iap.Discretize(other_param)
    other_param.a = [1.0]
    param_copy = param.deepcopy()
    assert isinstance(param_copy, iap.Discretize)
    assert isinstance(param_copy.other_param, iap.Uniform)
    param.other_param.a[0] += 1
    assert param_copy.other_param.a[0] != param.other_param.a[0]


def create_random_images(size):
    return np.random.uniform(0, 255, size).astype(np.uint8)


def create_random_keypoints(size_images, nb_keypoints_per_img):
    result = []
    for i in sm.xrange(size_images[0]):
        kps = []
        height, width = size_images[1], size_images[2]
        for i in sm.xrange(nb_keypoints_per_img):
            x = np.random.randint(0, width-1)
            y = np.random.randint(0, height-1)
            kps.append(ia.Keypoint(x=x, y=y))
        result.append(ia.KeypointsOnImage(kps, shape=size_images[1:]))
    return result


def array_equal_lists(list1, list2):
    assert isinstance(list1, list)
    assert isinstance(list2, list)

    if len(list1) != len(list2):
        return False

    for a, b in zip(list1, list2):
        if not np.array_equal(a, b):
            return False

    return True


def keypoints_equal(kps1, kps2, eps=0.001):
    if len(kps1) != len(kps2):
        return False

    for i in sm.xrange(len(kps1)):
        a = kps1[i].keypoints
        b = kps2[i].keypoints
        if len(a) != len(b):
            return False

        for j in sm.xrange(len(a)):
            x_equal = float(b[j].x) - eps <= float(a[j].x) <= float(b[j].x) + eps
            y_equal = float(b[j].y) - eps <= float(a[j].y) <= float(b[j].y) + eps
            if not x_equal or not y_equal:
                return False

    return True


def reseed(seed=0):
    ia.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    main()
