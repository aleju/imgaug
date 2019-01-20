from __future__ import print_function, division, absolute_import

import time
import functools

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.testutils import array_equal_lists, keypoints_equal, reseed


def main():
    time_start = time.time()

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
    # test_ContrastNormalization()
    test_JpegCompression()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_AdditiveGaussianNoise():
    reseed()

    base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no noise, shouldnt change anything
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0)

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
        _ = iaa.AdditiveGaussianNoise(loc="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.AdditiveGaussianNoise(scale="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test heatmaps (not affected by augmenter)
    aug = iaa.AdditiveGaussianNoise(loc=0.5, scale=10)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


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

    # test heatmaps (not affected by augmenter)
    aug = iaa.Dropout(p=1.0)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


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
        _ = iaa.CoarseDropout(p="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.CoarseDropout(p=0.5, size_px=None, size_percent=None)
    except Exception:
        got_exception = True
    assert got_exception

    # test heatmaps (not affected by augmenter)
    aug = iaa.CoarseDropout(p=1.0, size_px=2)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


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
        _ = iaa.Multiply(mul="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.Multiply(mul=1, per_channel="test")
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

    # test heatmaps (not affected by augmenter)
    aug = iaa.Multiply(mul=2)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    ###################
    # test other dtypes
    ###################
    # bool
    image = np.zeros((3, 3), dtype=bool)
    aug = iaa.Multiply(1.0)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.Multiply(1.0)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.Multiply(2.0)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.Multiply(0.0)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.Multiply(-1.0)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        image = np.full((3, 3), 10, dtype=dtype)
        aug = iaa.Multiply(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 10)

        image = np.full((3, 3), 10, dtype=dtype)
        aug = iaa.Multiply(10)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 100)

        image = np.full((3, 3), 10, dtype=dtype)
        aug = iaa.Multiply(0.5)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 5)

        image = np.full((3, 3), 0, dtype=dtype)
        aug = iaa.Multiply(0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 0)

        if np.dtype(dtype).kind == "u":
            image = np.full((3, 3), 10, dtype=dtype)
            aug = iaa.Multiply(-1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 0)
        else:
            image = np.full((3, 3), 10, dtype=dtype)
            aug = iaa.Multiply(-1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == -10)

        image = np.full((3, 3), int(center_value), dtype=dtype)
        aug = iaa.Multiply(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == int(center_value))

        image = np.full((3, 3), int(center_value), dtype=dtype)
        aug = iaa.Multiply(1.2)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == int(1.2 * int(center_value)))

        if np.dtype(dtype).kind == "u":
            image = np.full((3, 3), int(center_value), dtype=dtype)
            aug = iaa.Multiply(100)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == max_value)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.Multiply(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.Multiply(10)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.Multiply(0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 0)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.Multiply(-2)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value)

        for _ in sm.xrange(10):
            image = np.full((1, 1, 3), 10, dtype=dtype)
            aug = iaa.Multiply(iap.Uniform(0.5, 1.5))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(5 <= image_aug, image_aug <= 15))
            assert len(np.unique(image_aug)) == 1

            image = np.full((1, 1, 100), 10, dtype=dtype)
            aug = iaa.Multiply(iap.Uniform(0.5, 1.5), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(5 <= image_aug, image_aug <= 15))
            assert len(np.unique(image_aug)) > 1

            image = np.full((1, 1, 3), 10, dtype=dtype)
            aug = iaa.Multiply(iap.DiscreteUniform(1, 3))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) == 1

            image = np.full((1, 1, 100), 10, dtype=dtype)
            aug = iaa.Multiply(iap.DiscreteUniform(1, 3), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) > 1

    # float
    for dtype in [np.float16, np.float32]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        if dtype == np.float16:
            atol = 1e-3 * max_value
        else:
            atol = 1e-9 * max_value
        _allclose = functools.partial(np.allclose, atol=atol, rtol=0)

        image = np.full((3, 3), 10.0, dtype=dtype)
        aug = iaa.Multiply(1.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, 10.0)

        image = np.full((3, 3), 10.0, dtype=dtype)
        aug = iaa.Multiply(2.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, 20.0)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.Multiply(-10)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.Multiply(0.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, 0.0)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.Multiply(0.5)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, 0.5*max_value)

        image = np.full((3, 3), min_value, dtype=dtype)
        aug = iaa.Multiply(-2.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, max_value)

        image = np.full((3, 3), min_value, dtype=dtype)
        aug = iaa.Multiply(0.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, 0.0)

        # using tolerances of -100 - 1e-2 and 100 + 1e-2 is not enough for float16, had to be increased to -/+ 1e-1
        for _ in sm.xrange(10):
            image = np.full((1, 1, 3), 10.0, dtype=dtype)
            aug = iaa.Multiply(iap.Uniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
            assert np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
            assert np.allclose(image_aug[..., 0], image_aug[..., 1])

            image = np.full((1, 1, 100), 10.0, dtype=dtype)
            aug = iaa.Multiply(iap.Uniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
            assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])

            image = np.full((1, 1, 3), 10.0, dtype=dtype)
            aug = iaa.Multiply(iap.DiscreteUniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
            assert np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
            assert np.allclose(image_aug[..., 0], image_aug[..., 1])

            image = np.full((1, 1, 100), 10.0, dtype=dtype)
            aug = iaa.Multiply(iap.DiscreteUniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
            assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])


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

    # test heatmaps (not affected by augmenter)
    aug = iaa.MultiplyElementwise(mul=2)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    ###################
    # test other dtypes
    ###################
    # bool
    image = np.zeros((3, 3), dtype=bool)
    aug = iaa.MultiplyElementwise(1.0)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.MultiplyElementwise(1.0)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.MultiplyElementwise(2.0)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.MultiplyElementwise(0.0)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.MultiplyElementwise(-1.0)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        image = np.full((3, 3), 10, dtype=dtype)
        aug = iaa.MultiplyElementwise(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 10)

        image = np.full((3, 3), 10, dtype=dtype)
        aug = iaa.MultiplyElementwise(10)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 100)

        image = np.full((3, 3), 10, dtype=dtype)
        aug = iaa.MultiplyElementwise(0.5)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 5)

        image = np.full((3, 3), 0, dtype=dtype)
        aug = iaa.MultiplyElementwise(0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 0)

        if np.dtype(dtype).kind == "u":
            image = np.full((3, 3), 10, dtype=dtype)
            aug = iaa.MultiplyElementwise(-1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 0)
        else:
            image = np.full((3, 3), 10, dtype=dtype)
            aug = iaa.MultiplyElementwise(-1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == -10)

        image = np.full((3, 3), int(center_value), dtype=dtype)
        aug = iaa.MultiplyElementwise(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == int(center_value))

        image = np.full((3, 3), int(center_value), dtype=dtype)
        aug = iaa.MultiplyElementwise(1.2)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == int(1.2 * int(center_value)))

        if np.dtype(dtype).kind == "u":
            image = np.full((3, 3), int(center_value), dtype=dtype)
            aug = iaa.MultiplyElementwise(100)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == max_value)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.MultiplyElementwise(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.MultiplyElementwise(10)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.MultiplyElementwise(0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 0)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.MultiplyElementwise(-2)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value)

        for _ in sm.xrange(10):
            image = np.full((5, 5, 3), 10, dtype=dtype)
            aug = iaa.MultiplyElementwise(iap.Uniform(0.5, 1.5))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(5 <= image_aug, image_aug <= 15))
            assert len(np.unique(image_aug)) > 1
            assert np.all(image_aug[..., 0] == image_aug[..., 1])

            image = np.full((1, 1, 100), 10, dtype=dtype)
            aug = iaa.MultiplyElementwise(iap.Uniform(0.5, 1.5), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(5 <= image_aug, image_aug <= 15))
            assert len(np.unique(image_aug)) > 1

            image = np.full((5, 5, 3), 10, dtype=dtype)
            aug = iaa.MultiplyElementwise(iap.DiscreteUniform(1, 3))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) > 1
            assert np.all(image_aug[..., 0] == image_aug[..., 1])

            image = np.full((1, 1, 100), 10, dtype=dtype)
            aug = iaa.MultiplyElementwise(iap.DiscreteUniform(1, 3), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) > 1

    # float
    for dtype in [np.float16, np.float32]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        if dtype == np.float16:
            atol = 1e-3 * max_value
        else:
            atol = 1e-9 * max_value
        _allclose = functools.partial(np.allclose, atol=atol, rtol=0)

        image = np.full((3, 3), 10.0, dtype=dtype)
        aug = iaa.MultiplyElementwise(1.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, 10.0)

        image = np.full((3, 3), 10.0, dtype=dtype)
        aug = iaa.MultiplyElementwise(2.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, 20.0)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.MultiplyElementwise(-10)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.MultiplyElementwise(0.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, 0.0)

        image = np.full((3, 3), max_value, dtype=dtype)
        aug = iaa.MultiplyElementwise(0.5)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, 0.5*max_value)

        image = np.full((3, 3), min_value, dtype=dtype)
        aug = iaa.MultiplyElementwise(-2.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, max_value)

        image = np.full((3, 3), min_value, dtype=dtype)
        aug = iaa.MultiplyElementwise(0.0)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, 0.0)

        # using tolerances of -100 - 1e-2 and 100 + 1e-2 is not enough for float16, had to be increased to -/+ 1e-1
        for _ in sm.xrange(10):
            image = np.full((50, 1, 3), 10.0, dtype=dtype)
            aug = iaa.MultiplyElementwise(iap.Uniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
            assert not np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
            assert np.allclose(image_aug[..., 0], image_aug[..., 1])

            image = np.full((1, 1, 100), 10.0, dtype=dtype)
            aug = iaa.MultiplyElementwise(iap.Uniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
            assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])

            image = np.full((50, 1, 3), 10.0, dtype=dtype)
            aug = iaa.MultiplyElementwise(iap.DiscreteUniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
            assert not np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
            assert np.allclose(image_aug[..., 0], image_aug[..., 1])

            image = np.full((1, 1, 100), 10, dtype=dtype)
            aug = iaa.MultiplyElementwise(iap.DiscreteUniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-100 - 1e-1 < image_aug, image_aug < 100 + 1e-1))
            assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])


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

    # test heatmaps (not affected by augmenter)
    aug = iaa.ReplaceElementwise(mask=1, replacement=0.5)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    ###################
    # test other dtypes
    ###################
    # bool
    aug = iaa.ReplaceElementwise(mask=1, replacement=0)
    image = np.full((3, 3), False, dtype=bool)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    aug = iaa.ReplaceElementwise(mask=1, replacement=1)
    image = np.full((3, 3), False, dtype=bool)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    aug = iaa.ReplaceElementwise(mask=1, replacement=0)
    image = np.full((3, 3), True, dtype=bool)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    aug = iaa.ReplaceElementwise(mask=1, replacement=1)
    image = np.full((3, 3), True, dtype=bool)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    aug = iaa.ReplaceElementwise(mask=1, replacement=0.7)
    image = np.full((3, 3), False, dtype=bool)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    aug = iaa.ReplaceElementwise(mask=1, replacement=0.2)
    image = np.full((3, 3), False, dtype=bool)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        aug = iaa.ReplaceElementwise(mask=1, replacement=1)
        image = np.full((3, 3), 0, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 1)

        aug = iaa.ReplaceElementwise(mask=1, replacement=2)
        image = np.full((3, 3), 1, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == 2)

        aug = iaa.ReplaceElementwise(mask=1, replacement=max_value)
        image = np.full((3, 3), min_value, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value)

        aug = iaa.ReplaceElementwise(mask=1, replacement=min_value)
        image = np.full((3, 3), max_value, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value)

        aug = iaa.ReplaceElementwise(mask=1, replacement=iap.Uniform(1.0, 10.0))
        image = np.full((100, 1), 0, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(np.logical_and(1 <= image_aug, image_aug <= 10))
        assert len(np.unique(image_aug)) > 1

        aug = iaa.ReplaceElementwise(mask=1, replacement=iap.DiscreteUniform(1, 10))
        image = np.full((100, 1), 0, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(np.logical_and(1 <= image_aug, image_aug <= 10))
        assert len(np.unique(image_aug)) > 1

        aug = iaa.ReplaceElementwise(mask=0.5, replacement=iap.DiscreteUniform(1, 10), per_channel=True)
        image = np.full((1, 1, 100), 0, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(np.logical_and(0 <= image_aug, image_aug <= 10))
        assert len(np.unique(image_aug)) > 2

    # float
    for dtype in [np.float16, np.float32, np.float64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        atol = 1e-3*max_value if dtype == np.float16 else 1e-9 * max_value
        _allclose = functools.partial(np.allclose, atol=atol, rtol=0)

        aug = iaa.ReplaceElementwise(mask=1, replacement=1.0)
        image = np.full((3, 3), 0.0, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.allclose(image_aug, 1.0)

        aug = iaa.ReplaceElementwise(mask=1, replacement=2.0)
        image = np.full((3, 3), 1.0, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.allclose(image_aug, 2.0)

        aug = iaa.ReplaceElementwise(mask=1, replacement=max_value)
        image = np.full((3, 3), min_value, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, max_value)

        aug = iaa.ReplaceElementwise(mask=1, replacement=min_value)
        image = np.full((3, 3), max_value, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value)

        aug = iaa.ReplaceElementwise(mask=1, replacement=iap.Uniform(1.0, 10.0))
        image = np.full((100, 1), 0, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(np.logical_and(1 <= image_aug, image_aug <= 10))
        assert not np.allclose(image_aug[1:, :], image_aug[:-1, :], atol=0.01)

        aug = iaa.ReplaceElementwise(mask=1, replacement=iap.DiscreteUniform(1, 10))
        image = np.full((100, 1), 0, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(np.logical_and(1 <= image_aug, image_aug <= 10))
        assert not np.allclose(image_aug[1:, :], image_aug[:-1, :], atol=0.01)

        aug = iaa.ReplaceElementwise(mask=0.5, replacement=iap.DiscreteUniform(1, 10), per_channel=True)
        image = np.full((1, 1, 100), 0, dtype=dtype)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(np.logical_and(0 <= image_aug, image_aug <= 10))
        assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1], atol=0.01)


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
        _ = iaa.CoarseSaltAndPepper(p="test", size_px=100)
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.CoarseSaltAndPepper(p=0.5, size_px=None, size_percent=None)
    except Exception:
        got_exception = True
    assert got_exception

    # test heatmaps (not affected by augmenter)
    aug = iaa.CoarseSaltAndPepper(p=1.0, size_px=2)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


def test_Salt():
    reseed()

    base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
    aug = iaa.Salt(p=0.5)
    observed = aug.augment_image(base_img)
    p = np.mean(observed != 128)
    assert 0.4 < p < 0.6
    # Salt() occasionally replaces with 127, which probably should be the center-point here anyways
    assert np.all(observed >= 127)

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
        _ = iaa.CoarseSalt(p="test", size_px=100)
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.CoarseSalt(p=0.5, size_px=None, size_percent=None)
    except Exception:
        got_exception = True
    assert got_exception

    # test heatmaps (not affected by augmenter)
    aug = iaa.CoarseSalt(p=1.0, size_px=2)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


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
        _ = iaa.CoarsePepper(p="test", size_px=100)
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.CoarsePepper(p=0.5, size_px=None, size_percent=None)
    except Exception:
        got_exception = True
    assert got_exception

    # test heatmaps (not affected by augmenter)
    aug = iaa.CoarsePepper(p=1.0, size_px=2)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


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

    # uint8, every possible addition for base value 127
    for value_type in [float, int]:
        for per_channel in [False, True]:
            for value in np.arange(-255, 255+1):
                aug = iaa.Add(value=value_type(value), per_channel=per_channel)
                expected = np.clip(127 + value_type(value), 0, 255)

                img = np.full((1, 1), 127, dtype=np.uint8)
                img_aug = aug.augment_image(img)
                assert img_aug.item(0) == expected

                img = np.full((1, 1, 3), 127, dtype=np.uint8)
                img_aug = aug.augment_image(img)
                assert np.all(img_aug == expected)

    # specific tests with floats
    aug = iaa.Add(value=0.75)
    img = np.full((1, 1), 1, dtype=np.uint8)
    img_aug = aug.augment_image(img)
    assert img_aug.item(0) == 2

    img = np.full((1, 1), 1, dtype=np.uint16)
    img_aug = aug.augment_image(img)
    assert img_aug.item(0) == 2

    aug = iaa.Add(value=0.45)
    img = np.full((1, 1), 1, dtype=np.uint8)
    img_aug = aug.augment_image(img)
    assert img_aug.item(0) == 1

    img = np.full((1, 1), 1, dtype=np.uint16)
    img_aug = aug.augment_image(img)
    assert img_aug.item(0) == 1

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
        _ = iaa.Add(value="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.Add(value=1, per_channel="test")
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

    # test heatmaps (not affected by augmenter)
    aug = iaa.Add(value=10)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    ###################
    # test other dtypes
    ###################
    # bool
    image = np.zeros((3, 3), dtype=bool)
    aug = iaa.Add(value=1)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.Add(value=1)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.Add(value=-1)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.Add(value=-2)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        image = np.full((3, 3), min_value, dtype=dtype)
        aug = iaa.Add(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value + 1)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.Add(11)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value + 21)

        image = np.full((3, 3), max_value - 2, dtype=dtype)
        aug = iaa.Add(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value - 1)

        image = np.full((3, 3), max_value - 1, dtype=dtype)
        aug = iaa.Add(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value)

        image = np.full((3, 3), max_value - 1, dtype=dtype)
        aug = iaa.Add(2)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.Add(-9)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value + 1)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.Add(-10)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.Add(-11)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value)

        for _ in sm.xrange(10):
            image = np.full((1, 1, 3), 20, dtype=dtype)
            aug = iaa.Add(iap.Uniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) == 1

            image = np.full((1, 1, 100), 20, dtype=dtype)
            aug = iaa.Add(iap.Uniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) > 1

            image = np.full((1, 1, 3), 20, dtype=dtype)
            aug = iaa.Add(iap.DiscreteUniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) == 1

            image = np.full((1, 1, 100), 20, dtype=dtype)
            aug = iaa.Add(iap.DiscreteUniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) > 1

    # float
    for dtype in [np.float16, np.float32]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        if dtype == np.float16:
            atol = 1e-3 * max_value
        else:
            atol = 1e-9 * max_value
        _allclose = functools.partial(np.allclose, atol=atol, rtol=0)

        image = np.full((3, 3), min_value, dtype=dtype)
        aug = iaa.Add(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value + 1)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.Add(11)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value + 21)

        image = np.full((3, 3), max_value - 2, dtype=dtype)
        aug = iaa.Add(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, max_value - 1)

        image = np.full((3, 3), max_value - 1, dtype=dtype)
        aug = iaa.Add(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, max_value)

        image = np.full((3, 3), max_value - 1, dtype=dtype)
        aug = iaa.Add(2)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, max_value)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.Add(-9)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value + 1)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.Add(-10)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.Add(-11)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value)

        for _ in sm.xrange(10):
            image = np.full((50, 1, 3), 0, dtype=dtype)
            aug = iaa.Add(iap.Uniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
            assert np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
            assert np.allclose(image_aug[..., 0], image_aug[..., 1])

            image = np.full((1, 1, 100), 0, dtype=dtype)
            aug = iaa.Add(iap.Uniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
            assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])

            image = np.full((50, 1, 3), 0, dtype=dtype)
            aug = iaa.Add(iap.DiscreteUniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
            assert np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
            assert np.allclose(image_aug[..., 0], image_aug[..., 1])

            image = np.full((1, 1, 100), 0, dtype=dtype)
            aug = iaa.Add(iap.DiscreteUniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
            assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])


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

    # uint8, every possible addition for base value 127
    for value_type in [int]:
        for per_channel in [False, True]:
            for value in np.arange(-255, 255+1):
                aug = iaa.AddElementwise(value=value_type(value), per_channel=per_channel)
                expected = np.clip(127 + value_type(value), 0, 255)

                img = np.full((1, 1), 127, dtype=np.uint8)
                img_aug = aug.augment_image(img)
                assert img_aug.item(0) == expected

                img = np.full((1, 1, 3), 127, dtype=np.uint8)
                img_aug = aug.augment_image(img)
                assert np.all(img_aug == expected)

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

    # test heatmaps (not affected by augmenter)
    aug = iaa.AddElementwise(value=10)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    ###################
    # test other dtypes
    ###################
    # bool
    image = np.zeros((3, 3), dtype=bool)
    aug = iaa.AddElementwise(value=1)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.AddElementwise(value=1)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 1)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.AddElementwise(value=-1)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    image = np.full((3, 3), True, dtype=bool)
    aug = iaa.AddElementwise(value=-2)
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == 0)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        image = np.full((3, 3), min_value, dtype=dtype)
        aug = iaa.AddElementwise(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value + 1)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.AddElementwise(11)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value + 21)

        image = np.full((3, 3), max_value - 2, dtype=dtype)
        aug = iaa.AddElementwise(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value - 1)

        image = np.full((3, 3), max_value - 1, dtype=dtype)
        aug = iaa.AddElementwise(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value)

        image = np.full((3, 3), max_value - 1, dtype=dtype)
        aug = iaa.AddElementwise(2)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == max_value)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.AddElementwise(-9)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value + 1)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.AddElementwise(-10)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.AddElementwise(-11)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == min_value)

        for _ in sm.xrange(10):
            image = np.full((5, 5, 3), 20, dtype=dtype)
            aug = iaa.AddElementwise(iap.Uniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) > 1
            assert np.all(image_aug[..., 0] == image_aug[..., 1])

            image = np.full((1, 1, 100), 20, dtype=dtype)
            aug = iaa.AddElementwise(iap.Uniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) > 1

            image = np.full((5, 5, 3), 20, dtype=dtype)
            aug = iaa.AddElementwise(iap.DiscreteUniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) > 1
            assert np.all(image_aug[..., 0] == image_aug[..., 1])

            image = np.full((1, 1, 100), 20, dtype=dtype)
            aug = iaa.AddElementwise(iap.DiscreteUniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(10 <= image_aug, image_aug <= 30))
            assert len(np.unique(image_aug)) > 1

    # float
    for dtype in [np.float16, np.float32]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        if dtype == np.float16:
            atol = 1e-3 * max_value
        else:
            atol = 1e-9 * max_value
        _allclose = functools.partial(np.allclose, atol=atol, rtol=0)

        image = np.full((3, 3), min_value, dtype=dtype)
        aug = iaa.AddElementwise(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value + 1)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.AddElementwise(11)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value + 21)

        image = np.full((3, 3), max_value - 2, dtype=dtype)
        aug = iaa.AddElementwise(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, max_value - 1)

        image = np.full((3, 3), max_value - 1, dtype=dtype)
        aug = iaa.AddElementwise(1)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, max_value)

        image = np.full((3, 3), max_value - 1, dtype=dtype)
        aug = iaa.AddElementwise(2)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, max_value)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.AddElementwise(-9)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value + 1)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.AddElementwise(-10)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value)

        image = np.full((3, 3), min_value + 10, dtype=dtype)
        aug = iaa.AddElementwise(-11)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert _allclose(image_aug, min_value)

        for _ in sm.xrange(10):
            image = np.full((50, 1, 3), 0, dtype=dtype)
            aug = iaa.AddElementwise(iap.Uniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
            assert not np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
            assert np.allclose(image_aug[..., 0], image_aug[..., 1])

            image = np.full((1, 1, 100), 0, dtype=dtype)
            aug = iaa.AddElementwise(iap.Uniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
            assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])

            image = np.full((50, 1, 3), 0, dtype=dtype)
            aug = iaa.AddElementwise(iap.DiscreteUniform(-10, 10))
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
            assert not np.allclose(image_aug[1:, :, 0], image_aug[:-1, :, 0])
            assert np.allclose(image_aug[..., 0], image_aug[..., 1])

            image = np.full((1, 1, 100), 0, dtype=dtype)
            aug = iaa.AddElementwise(iap.DiscreteUniform(-10, 10), per_channel=True)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(np.logical_and(-10 - 1e-2 < image_aug, image_aug < 10 + 1e-2))
            assert not np.allclose(image_aug[:, :, 1:], image_aug[:, :, :-1])


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
    img = np.zeros((1, 1, 1), dtype=np.uint8) + 255
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
    img = np.zeros((1, 1, 1), dtype=np.uint8) + 255
    expected = np.zeros((1, 1, 1), dtype=np.uint8)
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(img)
        if np.array_equal(observed, expected):
            nb_inverted += 1
    pinv = nb_inverted / nb_iterations
    assert 0.75 <= pinv <= 0.85

    aug = iaa.Invert(p=0.5, per_channel=True)
    img = np.zeros((1, 1, 100), dtype=np.uint8) + 255
    observed = aug.augment_image(img)
    assert len(np.unique(observed)) == 2

    nb_iterations = 1000
    aug = iaa.Invert(p=iap.Binomial(0.8), per_channel=0.7)
    img = np.zeros((1, 1, 20), dtype=np.uint8) + 255
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
        _ = iaa.Invert(p="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.Invert(p=0.5, per_channel="test")
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

    # test heatmaps (not affected by augmenter)
    aug = iaa.Invert(p=1.0)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    #############################
    # test other dtypes below
    #############################
    # with p=0.0
    aug = iaa.Invert(p=0.0)
    dtypes = [bool,
              np.uint8, np.uint16, np.uint32, np.uint64,
              np.int8, np.int16, np.int32, np.int64,
              np.float16, np.float32, np.float64, np.float128]
    for dtype in dtypes:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
        kind = np.dtype(dtype).kind
        image_min = np.full((3, 3), min_value, dtype=dtype)
        if dtype is not bool:
            image_center = np.full((3, 3), center_value if kind == "f" else int(center_value), dtype=dtype)
        image_max = np.full((3, 3), max_value, dtype=dtype)
        image_min_aug = aug.augment_image(image_min)
        image_center_aug = None
        if dtype is not bool:
            image_center_aug = aug.augment_image(image_center)
        image_max_aug = aug.augment_image(image_max)

        assert image_min_aug.dtype == np.dtype(dtype)
        if image_center_aug is not None:
            assert image_center_aug.dtype == np.dtype(dtype)
        assert image_max_aug.dtype == np.dtype(dtype)

        if dtype is bool:
            assert np.all(image_min_aug == image_min)
            assert np.all(image_max_aug == image_max)
        elif np.dtype(dtype).kind in ["i", "u"]:
            assert np.array_equal(image_min_aug, image_min)
            assert np.array_equal(image_center_aug, image_center)
            assert np.array_equal(image_max_aug, image_max)
        else:
            assert np.allclose(image_min_aug, image_min)
            assert np.allclose(image_center_aug, image_center)
            assert np.allclose(image_max_aug, image_max)

    # with p=1.0
    aug = iaa.Invert(p=1.0)
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64,
              np.int8, np.int16, np.int32, np.int64,
              np.float16, np.float32, np.float64, np.float128]
    for dtype in dtypes:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
        kind = np.dtype(dtype).kind
        image_min = np.full((3, 3), min_value, dtype=dtype)
        if dtype is not bool:
            image_center = np.full((3, 3), center_value if kind == "f" else int(center_value), dtype=dtype)
        image_max = np.full((3, 3), max_value, dtype=dtype)
        image_min_aug = aug.augment_image(image_min)
        image_center_aug = None
        if dtype is not bool:
            image_center_aug = aug.augment_image(image_center)
        image_max_aug = aug.augment_image(image_max)

        assert image_min_aug.dtype == np.dtype(dtype)
        if image_center_aug is not None:
            assert image_center_aug.dtype == np.dtype(dtype)
        assert image_max_aug.dtype == np.dtype(dtype)

        if dtype is bool:
            assert np.all(image_min_aug == image_max)
            assert np.all(image_max_aug == image_min)
        elif np.dtype(dtype).kind in ["i", "u"]:
            assert np.array_equal(image_min_aug, image_max)
            assert np.allclose(image_center_aug, image_center, atol=1.0+1e-4, rtol=0)
            assert np.array_equal(image_max_aug, image_min)
        else:
            assert np.allclose(image_min_aug, image_max)
            assert np.allclose(image_center_aug, image_center)
            assert np.allclose(image_max_aug, image_min)

    # with p=1.0 and min_value
    aug = iaa.Invert(p=1.0, min_value=1)
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64,
              np.int8, np.int16, np.int32,
              np.float16, np.float32]
    for dtype in dtypes:
        _min_value, _center_value, max_value = iadt.get_value_range_of_dtype(dtype)
        min_value = 1
        kind = np.dtype(dtype).kind
        center_value = min_value + 0.5 * (max_value - min_value)
        image_min = np.full((3, 3), min_value, dtype=dtype)
        if dtype is not bool:
            image_center = np.full((3, 3), center_value if kind == "f" else int(center_value), dtype=dtype)
        image_max = np.full((3, 3), max_value, dtype=dtype)
        image_min_aug = aug.augment_image(image_min)
        image_center_aug = None
        if dtype is not bool:
            image_center_aug = aug.augment_image(image_center)
        image_max_aug = aug.augment_image(image_max)

        assert image_min_aug.dtype == np.dtype(dtype)
        if image_center_aug is not None:
            assert image_center_aug.dtype == np.dtype(dtype)
        assert image_max_aug.dtype == np.dtype(dtype)

        if dtype is bool:
            assert np.all(image_min_aug == 1)
            assert np.all(image_max_aug == 1)
        elif np.dtype(dtype).kind in ["i", "u"]:
            assert np.array_equal(image_min_aug, image_max)
            assert np.allclose(image_center_aug, image_center, atol=1.0+1e-4, rtol=0)
            assert np.array_equal(image_max_aug, image_min)
        else:
            assert np.allclose(image_min_aug, image_max)
            assert np.allclose(image_center_aug, image_center)
            assert np.allclose(image_max_aug, image_min)

    # with p=1.0 and max_value
    aug = iaa.Invert(p=1.0, max_value=16)
    dtypes = [np.uint8, np.uint16, np.uint32, np.uint64,
              np.int8, np.int16, np.int32,
              np.float16, np.float32]
    for dtype in dtypes:
        min_value, _center_value, _max_value = iadt.get_value_range_of_dtype(dtype)
        max_value = 16
        kind = np.dtype(dtype).kind
        center_value = min_value + 0.5 * (max_value - min_value)
        image_min = np.full((3, 3), min_value, dtype=dtype)
        if dtype is not bool:
            image_center = np.full((3, 3), center_value if kind == "f" else int(center_value), dtype=dtype)
        image_max = np.full((3, 3), max_value, dtype=dtype)
        image_min_aug = aug.augment_image(image_min)
        image_center_aug = None
        if dtype is not bool:
            image_center_aug = aug.augment_image(image_center)
        image_max_aug = aug.augment_image(image_max)

        assert image_min_aug.dtype == np.dtype(dtype)
        if image_center_aug is not None:
            assert image_center_aug.dtype == np.dtype(dtype)
        assert image_max_aug.dtype == np.dtype(dtype)

        if dtype is bool:
            assert not np.any(image_min_aug == 1)
            assert not np.any(image_max_aug == 1)
        elif np.dtype(dtype).kind in ["i", "u"]:
            assert np.array_equal(image_min_aug, image_max)
            assert np.allclose(image_center_aug, image_center, atol=1.0+1e-4, rtol=0)
            assert np.array_equal(image_max_aug, image_min)
        else:
            assert np.allclose(image_min_aug, image_max)
            if dtype is np.float16:
                # for float16, this is off by about 10
                assert np.allclose(image_center_aug, image_center, atol=0.001*np.finfo(dtype).max)
            else:
                assert np.allclose(image_center_aug, image_center)
            assert np.allclose(image_max_aug, image_min)


def deactivated_test_ContrastNormalization():
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
        _ = iaa.ContrastNormalization(alpha="test")
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.ContrastNormalization(alpha=1.5, per_channel="test")
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

    # test heatmaps (not affected by augmenter)
    aug = iaa.ContrastNormalization(alpha=2)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


def test_JpegCompression():
    reseed()

    img = ia.quokka(extract="square", size=(64, 64))

    # basic test at 0 compression
    aug = iaa.JpegCompression(0)
    img_aug = aug.augment_image(img)
    diff = np.average(np.abs(img.astype(np.float32) - img_aug.astype(np.float32)))
    assert diff < 1.0

    # basic test at 90 compression
    aug = iaa.JpegCompression(90)
    img_aug = aug.augment_image(img)
    diff = np.average(np.abs(img.astype(np.float32) - img_aug.astype(np.float32)))
    assert 1.0 < diff < 50.0

    # test if stochastic parameters are used by initializer
    aug = iaa.JpegCompression([0, 100])
    assert isinstance(aug.compression, iap.Choice)
    assert len(aug.compression.a) == 2
    assert aug.compression.a[0] == 0
    assert aug.compression.a[1] == 100

    # test get_parameters()
    assert len(aug.get_parameters()) == 1
    assert aug.get_parameters()[0] == aug.compression

    # test if stochastic parameters are used by augmentation
    class _TwoValueParam(iap.StochasticParameter):
        def __init__(self, v1, v2):
            super(_TwoValueParam, self).__init__()
            self.v1 = v1
            self.v2 = v2

        def _draw_samples(self, size, random_state):
            arr = np.full(size, self.v1, dtype=np.float32)
            arr[1::2] = self.v2
            return arr

    param = _TwoValueParam(0, 100)
    aug = iaa.JpegCompression(param)
    img_aug_c0 = iaa.JpegCompression(0).augment_image(img)
    img_aug_c100 = iaa.JpegCompression(100).augment_image(img)
    imgs_aug = aug.augment_images([img] * 4)
    assert np.array_equal(imgs_aug[0], img_aug_c0)
    assert np.array_equal(imgs_aug[1], img_aug_c100)
    assert np.array_equal(imgs_aug[2], img_aug_c0)
    assert np.array_equal(imgs_aug[3], img_aug_c100)

    # test keypoints (not affected by augmenter)
    aug = iaa.JpegCompression(50)
    kps = ia.quokka_keypoints()
    kps_aug = aug.augment_keypoints([kps])[0]
    for kp, kp_aug in zip(kps.keypoints, kps_aug.keypoints):
        assert np.allclose([kp.x, kp.y], [kp_aug.x, kp_aug.y])

    # test heatmaps (not affected by augmenter)
    aug = iaa.JpegCompression(50)
    hm = ia.quokka_heatmap()
    hm_aug = aug.augment_heatmaps([hm])[0]
    assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


if __name__ == "__main__":
    main()
