from __future__ import print_function, division, absolute_import

import time

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.testutils import keypoints_equal, reseed


def main():
    time_start = time.time()

    test_GaussianBlur()
    test_AverageBlur()
    test_MedianBlur()
    # TODO BilateralBlur

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


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


def test_MotionBlur():
    reseed()

    # simple scenario
    aug = iaa.MotionBlur(k=3, angle=0, direction=0.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 1.0/3, 0],
        [0, 1.0/3, 0],
        [0, 1.0/3, 0]
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected)

    # 90deg angle
    aug = iaa.MotionBlur(k=3, angle=90, direction=0.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 0, 0],
        [1.0/3, 1.0/3, 1.0/3],
        [0, 0, 0]
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected)

    # 45deg angle
    aug = iaa.MotionBlur(k=3, angle=45, direction=0.0, order=0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 0, 1.0/3],
        [0, 1.0/3, 0],
        [1.0/3, 0, 0]
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected)

    # random angle
    aug = iaa.MotionBlur(k=3, angle=[0, 90], direction=0.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(50)]
    expected1 = np.float32([
        [0, 1.0/3, 0],
        [0, 1.0/3, 0],
        [0, 1.0/3, 0]
    ])
    expected2 = np.float32([
        [0, 0, 0],
        [1.0/3, 1.0/3, 1.0/3],
        [0, 0, 0],
    ])
    nb_seen = [0, 0]
    for matrices_image in matrices:
        assert np.allclose(matrices_image[0], matrices_image[1])
        assert np.allclose(matrices_image[1], matrices_image[2])
        for matrix_channel in matrices_image:
            if np.allclose(matrix_channel, expected1):
                nb_seen[0] += 1
            elif np.allclose(matrix_channel, expected2):
                nb_seen[1] += 1
    assert nb_seen[0] > 0
    assert nb_seen[1] > 0

    # 5x5
    aug = iaa.MotionBlur(k=5, angle=90, direction=0.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected)

    # random k
    aug = iaa.MotionBlur(k=[3, 5], angle=90, direction=0.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(50)]
    expected1 = np.float32([
        [0, 0, 0],
        [1.0/3, 1.0/3, 1.0/3],
        [0, 0, 0],
    ])
    expected2 = np.float32([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    nb_seen = [0, 0]
    for matrices_image in matrices:
        assert np.allclose(matrices_image[0], matrices_image[1])
        assert np.allclose(matrices_image[1], matrices_image[2])
        for matrix_channel in matrices_image:
            if matrix_channel.shape == expected1.shape and np.allclose(matrix_channel, expected1):
                nb_seen[0] += 1
            elif matrix_channel.shape == expected2.shape and np.allclose(matrix_channel, expected2):
                nb_seen[1] += 1
    assert nb_seen[0] > 0
    assert nb_seen[1] > 0

    # direction 1.0
    aug = iaa.MotionBlur(k=3, angle=0, direction=1.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 1.0/1.5, 0],
        [0, 0.5/1.5, 0],
        [0, 0.0/1.5, 0]
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected, rtol=0, atol=1e-2)

    # direction -1.0
    aug = iaa.MotionBlur(k=3, angle=0, direction=-1.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 0.0/1.5, 0],
        [0, 0.5/1.5, 0],
        [0, 1.0/1.5, 0]
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected, rtol=0, atol=1e-2)

    # random direction
    aug = iaa.MotionBlur(k=3, angle=[0, 90], direction=[-1.0, 1.0])
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(50)]
    expected1 = np.float32([
        [0, 1.0/1.5, 0],
        [0, 0.5/1.5, 0],
        [0, 0.0/1.5, 0]
    ])
    expected2 = np.float32([
        [0, 0.0/1.5, 0],
        [0, 0.5/1.5, 0],
        [0, 1.0/1.5, 0]
    ])
    nb_seen = [0, 0]
    for matrices_image in matrices:
        assert np.allclose(matrices_image[0], matrices_image[1])
        assert np.allclose(matrices_image[1], matrices_image[2])
        for matrix_channel in matrices_image:
            if np.allclose(matrix_channel, expected1, rtol=0, atol=1e-2):
                nb_seen[0] += 1
            elif np.allclose(matrix_channel, expected2, rtol=0, atol=1e-2):
                nb_seen[1] += 1
    assert nb_seen[0] > 0
    assert nb_seen[1] > 0

    # test of actual augmenter
    img = np.zeros((7, 7, 3), dtype=np.uint8)
    img[3-1:3+2, 3-1:3+2, :] = 255
    aug = iaa.MotionBlur(k=3, angle=90, direction=0.0)
    img_aug = aug.augment_image(img)
    v1 = (255*(1/3))
    v2 = (255*(1/3)) * 2
    v3 = (255*(1/3)) * 3
    expected = np.float32([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, v1, v2, v3, v2, v1, 0],
        [0, v1, v2, v3, v2, v1, 0],
        [0, v1, v2, v3, v2, v1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]).astype(np.uint8)
    expected = np.tile(expected[..., np.newaxis], (1, 1, 3))
    assert np.allclose(img_aug, expected)


if __name__ == "__main__":
    main()
