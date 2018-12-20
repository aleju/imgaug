from __future__ import print_function, division, absolute_import

import time
import copy
import warnings

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.augmenters import meta
from imgaug.testutils import create_random_images, create_random_keypoints, array_equal_lists, keypoints_equal, reseed


def main():
    time_start = time.time()

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
    test_ChannelShuffle()
    test_2d_inputs()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


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

    ###################
    # test other dtypes
    ###################
    aug = iaa.Noop()

    image = np.zeros((3, 3), dtype=bool)
    image[0, 0] = True
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == image.dtype.type
    assert np.all(image_aug == image)

    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int32, np.int64]:
        min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)
        value = max_value
        image = np.zeros((3, 3), dtype=dtype)
        image[0, 0] = value
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.array_equal(image_aug, image)

    for dtype, value in zip([np.float16, np.float32, np.float64, np.float128], [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]):
        image = np.zeros((3, 3), dtype=dtype)
        image[0, 0] = value
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == image)


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

    # images as numpy array
    observed = aug_succeeds.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    errored = False
    try:
        _ = aug_fails.augment_images(images)
    except AssertionError:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    errored = False
    try:
        _ = aug_fails.augment_images(images)
    except AssertionError:
        errored = True
    assert errored

    # Lists of images
    observed = aug_succeeds.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    errored = False
    try:
        _ = aug_fails.augment_images(images_list)
    except AssertionError:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    errored = False
    try:
        _ = aug_fails.augment_images(images_list)
    except AssertionError:
        errored = True
    assert errored

    # heatmaps
    observed = aug_succeeds.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps.get_arr())

    errored = False
    try:
        _ = aug_fails.augment_heatmaps([heatmaps])[0]
    except AssertionError:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps.get_arr())

    errored = False
    try:
        _ = aug_fails.augment_heatmaps([heatmaps])[0]
    except AssertionError as e:
        errored = True
    assert errored

    # keypoints
    observed = aug_succeeds.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    errored = False
    try:
        _ = aug_fails.augment_keypoints(keypoints)
    except AssertionError as e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    errored = False
    try:
        _ = aug_fails.augment_keypoints(keypoints)
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

        errored = False
        try:
            _ = aug.augment_images(images_h4)
        except AssertionError:
            errored = True
        assert errored

        errored = False
        try:
            _ = aug.augment_heatmaps([heatmaps_h4])[0]
        except AssertionError:
            errored = True
        assert errored

        errored = False
        try:
            _ = aug.augment_keypoints(keypoints_h4)
        except AssertionError:
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

        errored = False
        try:
            _ = aug.augment_images(images_h4)
        except AssertionError:
            errored = True
        assert errored

        errored = False
        try:
            _ = aug.augment_heatmaps([heatmaps_h4])[0]
        except AssertionError:
            errored = True
        assert errored

        errored = False
        try:
            _ = aug.augment_keypoints(keypoints_h4)
        except AssertionError:
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

        errored = False
        try:
            _ = aug.augment_images(images_h4)
        except AssertionError:
            errored = True
        assert errored

        errored = False
        try:
            _ = aug.augment_heatmaps([heatmaps_h4])[0]
        except AssertionError:
            errored = True
        assert errored

        errored = False
        try:
            _ = aug.augment_keypoints(keypoints_h4)
        except AssertionError:
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

        errored = False
        try:
            _ = aug.augment_images(images_h4)
        except AssertionError:
            errored = True
        assert errored

        errored = False
        try:
            _ = aug.augment_heatmaps([heatmaps_h4])[0]
        except AssertionError:
            errored = True
        assert errored

        errored = False
        try:
            _ = aug.augment_keypoints(keypoints_h4)
        except AssertionError:
            errored = True
        assert errored

    # bad datatype
    got_exception = False
    try:
        aug = iaa.AssertShape((1, False, 4, 1))
        _ = aug.augment_images(np.zeros((1, 2, 2, 1), dtype=np.uint8))
    except Exception as exc:
        assert "Invalid datatype " in str(exc)
        got_exception = True
    assert got_exception


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
        _ = list(aug.augment_batches(None))
    except Exception:
        got_exception = True
    assert got_exception

    aug = DummyAugmenter()
    got_exception = False
    try:
        _ = list(aug.augment_batches([None]))
    except Exception as exc:
        got_exception = True
        assert "Unknown datatype of batch" in str(exc)
    assert got_exception

    aug = DummyAugmenter()
    got_exception = False
    try:
        _ = list(aug.augment_batches([[None]]))
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
        _ = aug.augment_images(np.zeros((16, 32, 3), dtype=np.uint8))
        # Verify some things
        assert len(caught_warnings) == 1
        assert "indicates that you provided a single image with shape (H, W, C)" in str(caught_warnings[-1].message)

    aug = DummyAugmenter()
    got_exception = False
    try:
        _ = aug.augment_images(None)
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
            return super(DummyAugmenterCallsParent, self)\
                ._augment_keypoints(keypoints_on_images, random_state, parents, hooks)

        def get_parameters(self):
            return super(DummyAugmenterCallsParent, self).get_parameters()

    aug = DummyAugmenterCallsParent()
    got_exception = False
    try:
        _ = aug.augment_images(np.zeros((2, 4, 4, 3), dtype=np.uint8))
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
        _ = aug.augment_heatmaps([heatmaps])
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
        _ = aug.augment_keypoints(keypoints)
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
        _ = aug.draw_grid([np.zeros((2,), dtype=np.uint8)], rows=2, cols=2)
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
        _ = aug.draw_grid(np.zeros((2,), dtype=np.uint8), rows=2, cols=2)
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
    expected = np.random.RandomState(np.random.RandomState(123).randint(0, 10**6)).randint(0, 10**6)
    assert aug0_copy.random_state.randint(0, 10**6) == expected

    aug0_copy = aug0.deepcopy()
    assert _same_rs(aug0.random_state, aug0_copy.random_state)
    assert _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
    assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)
    aug0_copy.reseed(random_state=np.random.RandomState(123))
    assert not _same_rs(aug0.random_state, aug0_copy.random_state)
    assert not _same_rs(aug0[0].random_state, aug0_copy[0].random_state)
    assert _same_rs(aug0[1].random_state, aug0_copy[1].random_state)
    expected = np.random.RandomState(np.random.RandomState(123).randint(0, 10**6)).randint(0, 10**6)
    assert aug0_copy.random_state.randint(0, 10**6) == expected

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
    assert aug.__repr__() == aug.__str__() == \
        "DummyAugmenterRepr(name=Example, parameters=[A, B, C], deterministic=False)"
    aug = DummyAugmenterRepr(name="Example", deterministic=True)
    assert aug.__repr__() == aug.__str__() == \
        "DummyAugmenterRepr(name=Example, parameters=[A, B, C], deterministic=True)"


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
        _ = augs.remove_augmenters(lambda aug, parents: aug.name == "Seq", copy=False)
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
    image_lrud = np.array([[1, 1, 0],
                           [1, 0, 0],
                           [1, 0, 0]], dtype=np.uint8)
    image = image[:, :, np.newaxis]
    image_lr = image_lr[:, :, np.newaxis]
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
    keypoints_aug = aug.augment_keypoints(keypoints, hooks=hooks)
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
        _ = target.copy_random_state(source, matching="name")
    except Exception as exc:
        got_exception = True
        assert "localize_random_state" in str(exc)
    assert got_exception

    source = iaa.Fliplr(0.5, name="hflip-other-name")
    target = iaa.Fliplr(0.5, name="hflip")
    source.localize_random_state_()
    got_exception = False
    try:
        _ = target.copy_random_state(source, matching="name", matching_tolerant=False)
    except Exception as exc:
        got_exception = True
        assert "not found among source augmenters" in str(exc)
    assert got_exception

    source = iaa.Fliplr(0.5, name="hflip")
    target = iaa.Fliplr(0.5, name="hflip")
    got_exception = False
    try:
        _ = target.copy_random_state(source, matching="position")
    except Exception as exc:
        got_exception = True
        assert "localize_random_state" in str(exc)
    assert got_exception

    source = iaa.Sequential([iaa.Fliplr(0.5, name="hflip"), iaa.Fliplr(0.5, name="hflip2")])
    target = iaa.Sequential([iaa.Fliplr(0.5, name="hflip")])
    source.localize_random_state_()
    got_exception = False
    try:
        _ = target.copy_random_state(source, matching="position", matching_tolerant=False)
    except Exception as exc:
        got_exception = True
        assert "different lengths" in str(exc)
    assert got_exception

    source = iaa.Sequential([iaa.Fliplr(0.5, name="hflip"), iaa.Fliplr(0.5, name="hflip2")])
    target = iaa.Sequential([iaa.Fliplr(0.5, name="hflip")])
    source.localize_random_state_()
    got_exception = False
    try:
        _ = target.copy_random_state(source, matching="test")
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
        _ = target.copy_random_state(source, matching="name")
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

    # should be the same in roughly 25% of all cases
    assert (0.25 - 0.10) <= (1 - (nb_changed_aug / nb_iterations)) <= (0.25 + 0.10)
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
    last_aug_random = None
    last_aug_random_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_changed_aug_random = 0
    nb_changed_aug_random_det = 0
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
            last_aug = observed_aug_unrandom
            last_aug_det = observed_aug_unrandom_det
            last_aug_random = observed_aug_random
            last_aug_random_det = observed_aug_random_det
        else:
            if not np.array_equal(observed_aug_unrandom, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_unrandom_det, last_aug_det):
                nb_changed_aug_det += 1
            if not np.array_equal(observed_aug_random, last_aug_random):
                nb_changed_aug_random += 1
            if not np.array_equal(observed_aug_random_det, last_aug_random_det):
                nb_changed_aug_random_det += 1
            last_aug = observed_aug_unrandom
            last_aug_det = observed_aug_unrandom_det
            last_aug_random = observed_aug_random
            last_aug_random_det = observed_aug_random_det

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
    assert (0.5 - 0.1) * nb_iterations <= nb_changed_aug_random <= (0.5 + 0.1) * nb_iterations
    assert nb_changed_aug_random_det == 0
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
    expected = "Sequential(name=%s, random_order=%s, children=[%s], deterministic=%s)" % (
        aug.name, "True", str(flip), "False"
    )
    assert aug.__str__() == aug.__repr__() == expected

    ###################
    # test other dtypes
    ###################
    # no change via Noop (known to work with any datatype)
    for random_order in [False, True]:
        aug = iaa.Sequential([
            iaa.Noop(),
            iaa.Noop()
        ], random_order=random_order)

        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == image.dtype.type
        assert np.all(image_aug == image)

        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int32, np.int64]:
            min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)
            value = max_value
            image = np.zeros((3, 3), dtype=dtype)
            image[0, 0] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.array_equal(image_aug, image)

        for dtype, value in zip([np.float16, np.float32, np.float64, np.float128],
                                [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]):
            image = np.zeros((3, 3), dtype=dtype)
            image[0, 0] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == image)

    # flips (known to work with any datatype)
    for random_order in [False, True]:
        aug = iaa.Sequential([
            iaa.Fliplr(1.0),
            iaa.Flipud(1.0)
        ], random_order=random_order)

        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True
        expected = np.zeros((3, 3), dtype=bool)
        expected[2, 2] = True
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == image.dtype.type
        assert np.all(image_aug == expected)

        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int32, np.int64]:
            min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)
            value = max_value
            image = np.zeros((3, 3), dtype=dtype)
            image[0, 0] = value
            expected = np.zeros((3, 3), dtype=dtype)
            expected[2, 2] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.array_equal(image_aug, expected)

        for dtype, value in zip([np.float16, np.float32, np.float64, np.float128],
                                [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]):
            image = np.zeros((3, 3), dtype=dtype)
            image[0, 0] = value
            expected = np.zeros((3, 3), dtype=dtype)
            expected[2, 2] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == expected)


def test_SomeOf():
    reseed()

    zeros = np.zeros((3, 3, 1), dtype=np.uint8)

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
    obs_lst = [observed0, observed1, observed2, observed3]
    heatmaps_lst = [heatmaps_arr0, heatmaps_arr1, heatmaps_arr2, heatmaps_arr3]
    for obs, exp in zip(obs_lst, heatmaps_lst):
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
        _ = iaa.SomeOf(1, children=False)
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
        _ = iaa.SomeOf((2, "test"), children=iaa.Fliplr(1.0))
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
        _ = iaa.SomeOf(False, children=iaa.Fliplr(1.0))
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

    ###################
    # test other dtypes
    ###################
    # no change via Noop (known to work with any datatype)
    for random_order in [False, True]:
        aug = iaa.SomeOf(2, [
            iaa.Noop(),
            iaa.Noop(),
            iaa.Noop()
        ], random_order=random_order)

        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == image.dtype.type
        assert np.all(image_aug == image)

        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int32, np.int64]:
            min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)
            value = max_value
            image = np.zeros((3, 3), dtype=dtype)
            image[0, 0] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.array_equal(image_aug, image)

        for dtype, value in zip([np.float16, np.float32, np.float64, np.float128],
                                [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]):
            image = np.zeros((3, 3), dtype=dtype)
            image[0, 0] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == image)

    # flips (known to work with any datatype)
    for random_order in [False, True]:
        aug = iaa.SomeOf(2, [
            iaa.Fliplr(1.0),
            iaa.Flipud(1.0),
            iaa.Noop()
        ], random_order=random_order)

        image = np.zeros((3, 3), dtype=bool)
        image[0, 0] = True
        expected = [np.zeros((3, 3), dtype=bool) for _ in sm.xrange(3)]
        expected[0][0, 2] = True
        expected[1][2, 0] = True
        expected[2][2, 2] = True
        for _ in sm.xrange(10):
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == image.dtype.type
            assert any([np.all(image_aug == expected_i) for expected_i in expected])

        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int32, np.int64]:
            min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)
            value = max_value
            image = np.zeros((3, 3), dtype=dtype)
            image[0, 0] = value
            expected = [np.zeros((3, 3), dtype=dtype) for _ in sm.xrange(3)]
            expected[0][0, 2] = value
            expected[1][2, 0] = value
            expected[2][2, 2] = value
            for _ in sm.xrange(10):
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert any([np.all(image_aug == expected_i) for expected_i in expected])

        for dtype, value in zip([np.float16, np.float32, np.float64, np.float128],
                                [5000, 1000 ** 2, 1000 ** 3, 1000 ** 4]):
            image = np.zeros((3, 3), dtype=dtype)
            image[0, 0] = value
            expected = [np.zeros((3, 3), dtype=dtype) for _ in sm.xrange(3)]
            expected[0][0, 2] = value
            expected[1][2, 0] = value
            expected[2][2, 2] = value
            for _ in sm.xrange(10):
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert any([np.all(image_aug == expected_i) for expected_i in expected])


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
        s = int(np.sum(result))
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
    # should be the same in roughly 50% of all cases
    assert (0.50 - 0.10) <= (1 - (nb_changed_aug / nb_iterations)) <= (0.50 + 0.10)
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
        _ = iaa.Sometimes(p="foo")
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
        _ = iaa.Sometimes(p=0.2, then_list=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # else_list bad datatype
    got_exception = False
    try:
        _ = iaa.Sometimes(p=0.2, then_list=None, else_list=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # deactivated propagation via hooks
    image = np.random.randint(0, 255-10, size=(16, 16), dtype=np.uint8)
    aug = iaa.Sometimes(1.0, iaa.Add(10))
    observed1 = aug.augment_image(image)
    observed2 = aug.augment_image(
        image,
        hooks=ia.HooksImages(
            propagator=lambda images, augmenter, parents, default: False if augmenter == aug else default
        )
    )
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
        "Sequential(name=SometimesTest-then, random_order=False, children=[%s], deterministic=False)" % (
            str(then_list),),
        "Sequential(name=SometimesTest-else, random_order=False, children=[%s], deterministic=False)" % (
            str(else_list),),
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
        assert isinstance(observed, list) \
            or (ia.is_np_array(observed) and len(set([img.shape for img in observed])) == 1)
        assert all([img.shape in [(4, 8, 3), (6, 8, 3)] for img in observed])

        observed = aug.augment_images([image, image, image, image])
        assert isinstance(observed, list)
        assert all([img.shape in [(4, 8, 3), (6, 8, 3)] for img in observed])

        observed = aug.augment_images(np.uint8([image]))
        assert isinstance(observed, list) \
            or (ia.is_np_array(observed) and len(set([img.shape for img in observed])) == 1)
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
        assert isinstance(observed, list) \
            or (ia.is_np_array(observed) and len(set([img.shape for img in observed])) == 1)
        assert all([16 <= img.shape[0] <= 30 and img.shape[1:] == (32, 3) for img in observed])

        observed = aug.augment_images([image, image, image, image])
        assert isinstance(observed, list)
        assert all([16 <= img.shape[0] <= 30 and img.shape[1:] == (32, 3) for img in observed])

        observed = aug.augment_images(np.uint8([image]))
        assert isinstance(observed, list) \
            or (ia.is_np_array(observed) and len(set([img.shape for img in observed])) == 1)
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
        _ = iaa.WithChannels(False, iaa.Add(10))
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # invalid datatype for children
    got_exception = False
    try:
        _ = iaa.WithChannels(1, False)
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


def test_ChannelShuffle():
    reseed()

    # p=1.0
    aug = iaa.ChannelShuffle(p=1.0)
    img = np.uint8([0, 1]).reshape((1, 1, 2))
    expected = [
        np.uint8([0, 1]).reshape((1, 1, 2)),
        np.uint8([1, 0]).reshape((1, 1, 2))
    ]
    seen = [False, False]
    for _ in sm.xrange(100):
        img_aug = aug.augment_image(img)
        if np.array_equal(img_aug, expected[0]):
            seen[0] = True
        elif np.array_equal(img_aug, expected[1]):
            seen[1] = True
        else:
            assert False
        if all(seen):
            break
    assert all(seen)

    # p=0
    aug = iaa.ChannelShuffle(p=0)
    img = np.uint8([0, 1]).reshape((1, 1, 2))
    for _ in sm.xrange(20):
        img_aug = aug.augment_image(img)
        assert np.array_equal(img_aug, img)

    # channels=[0, 2]
    aug = iaa.ChannelShuffle(p=1.0, channels=[0, 2])
    img = np.uint8([0, 1, 2]).reshape((1, 1, 3))
    expected = [
        np.uint8([0, 1, 2]).reshape((1, 1, 3)),
        np.uint8([2, 1, 0]).reshape((1, 1, 3))
    ]
    seen = [False, False]
    for _ in sm.xrange(100):
        img_aug = aug.augment_image(img)
        if np.array_equal(img_aug, expected[0]):
            seen[0] = True
        elif np.array_equal(img_aug, expected[1]):
            seen[1] = True
        else:
            assert False
        if all(seen):
            break
    assert all(seen)

    # check p parsing
    aug = iaa.ChannelShuffle(p=0.9, channels=[0, 2])
    assert isinstance(aug.p, iap.Binomial)
    assert isinstance(aug.p.p, iap.Deterministic)
    assert np.allclose(aug.p.p.value, 0.9)
    assert aug.channels == [0, 2]

    # get_parameters()
    aug = iaa.ChannelShuffle(p=1.0, channels=[0, 2])
    assert aug.get_parameters()[0] == aug.p
    assert aug.get_parameters()[1] == aug.channels

    # heatmaps may not change
    aug = iaa.ChannelShuffle(p=1.0)
    hm = ia.HeatmapsOnImage(np.float32([[0, 0.5, 1.0]]), shape=(4, 4, 3))
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert hm_aug.shape == (4, 4, 3)
    assert hm_aug.arr_0to1.shape == (1, 3, 1)
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    # keypoints may not change
    aug = iaa.ChannelShuffle(p=1.0)
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=3, y=1), ia.Keypoint(x=2, y=4)], shape=(10, 10, 3))
    kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (10, 10, 3)
    assert np.allclose(kpsoi_aug.keypoints[0].x, 3)
    assert np.allclose(kpsoi_aug.keypoints[0].y, 1)
    assert np.allclose(kpsoi_aug.keypoints[1].x, 2)
    assert np.allclose(kpsoi_aug.keypoints[1].y, 4)


def test_2d_inputs():
    """Test whether inputs of 2D-images (i.e. (H, W) instead of (H, W, C)) work."""
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
        batches = [ia.Batch(images=[np.copy(image)], keypoints=[keypoints[0].deepcopy()])
                   for _ in sm.xrange(nb_iterations)]
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
        nb_iterations = 1000
        batches = [ia.Batch(images=np.array([np.copy(image)], dtype=np.uint8), keypoints=None) for _ in sm.xrange(nb_iterations)]
        batches_aug = list(seq.augment_batches(batches, background=bg))
        for batch_aug in batches_aug:
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
            image_aug = batch_aug[0]
            assert np.array_equal(image_aug, image) or np.array_equal(image_aug, image_flipped)
            if np.array_equal(image_aug, image_flipped):
                nb_flipped_images += 1
        assert 0.4*nb_iterations <= nb_flipped_images <= 0.6*nb_iterations

        # list of list of KeypointsOnImage as input
        nb_flipped_keypoints = 0
        nb_iterations = 1000
        batches = [[keypoints[0].deepcopy()] for _ in sm.xrange(nb_iterations)]
        batches_aug = list(seq.augment_batches(batches, background=bg))
        for batch_aug in batches_aug:
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


if __name__ == "__main__":
    main()
