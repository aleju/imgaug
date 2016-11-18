from __future__ import print_function, division
import pyimgaug as ia
import augmenters2 as iaa
import parameters as iap
#from skimage import
import numpy as np
import time

def main():
    # Sequential

    # Sometimes

    # Noop
    test_Noop()

    # Lambda
    test_Lambda()

    # AssertLambda
    test_AssertLambda()

    # AssertShape
    test_AssertShape()

    # Fliplr
    test_Fliplr()

    # Flipud
    test_Flipud()

    # GaussianBlur
    test_GaussianBlur()

    # AdditiveGaussianNoise
    test_AdditiveGaussianNoise()

    # MultiplicativeGaussianNoise

    # ReplacingGaussianNoise

    # Dropout

    # Multiply

    # Affine

    # ElasticTransformation

    print("Finished.")

def test_Sequential():
    pass

def test_Sometimes():
    pass

def test_Noop():
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

def test_Lambda():
    base_img = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]
    images = np.array([base_img])
    images_list = [base_img]

    images_aug = images + 1
    images_aug_list = [image + 1 for image in images_list]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=2, y=1), ia.Keypoint(x=0, y=2)], shape=base_img.shape)]

    def func_images(images, random_state, parents, hooks):
        if isinstance(images, list):
            images = [image + 1 for image in images]
        else:
            images = images + 1
        return images

    def func_keypoints(keypoints_on_images, random_state, parents, hooks):
        for keypoints_on_image in keypoints_on_images:
            for kp in keypoints_on_image.keypoints:
                kp.x = (kp.x + 1) % 3
        return keypoints_on_images

    aug = iaa.Lambda(func_images, func_keypoints)
    aug_det = aug.to_deterministic()

    # check once that the augmenter can handle lists correctly
    observed = aug.augment_images(images_list)
    expected = images_aug_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_aug_list
    assert array_equal_lists(observed, expected)

    for _ in range(10):
        observed = aug.augment_images(images)
        expected = images_aug
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images_aug
        assert np.array_equal(observed, expected)

        observed = aug.augment_keypoints(keypoints)
        expected = keypoints_aug
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints_aug
        assert keypoints_equal(observed, expected)

def test_AssertLambda():
    base_img = np.array([[0, 0, 1],
                         [0, 0, 1],
                         [0, 1, 1]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]
    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    def func_images_succeeds(images, random_state, parents, hooks):
        return images[0][0, 0] == 0 and images[0][2, 2] == 1

    def func_images_fails(images, random_state, parents, hooks):
        return images[0][0, 0] == 1

    def func_keypoints_succeeds(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images[0].keypoints[0].x == 0 and keypoints_on_images[0].keypoints[2].x == 2

    def func_keypoints_fails(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images[0].keypoints[0].x == 2

    aug_succeeds = iaa.AssertLambda(func_images_succeeds, func_keypoints_succeeds)
    aug_succeeds_det = aug_succeeds.to_deterministic()
    aug_fails = iaa.AssertLambda(func_images_fails, func_keypoints_fails)
    aug_fails_det = aug_fails.to_deterministic()

    # images as numpy array
    observed = aug_succeeds.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    try:
        observed = aug_fails.augment_images(images)
        errored = False
    except AssertionError, e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    try:
        observed = aug_fails.augment_images(images)
        errored = False
    except AssertionError, e:
        errored = True
    assert errored

    # Lists of images
    observed = aug_succeeds.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    try:
        observed = aug_fails.augment_images(images_list)
        errored = False
    except AssertionError, e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    try:
        observed = aug_fails.augment_images(images_list)
        errored = False
    except AssertionError, e:
        errored = True
    assert errored

    # keypoints
    observed = aug_succeeds.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    try:
        observed = aug_fails.augment_keypoints(keypoints)
        errored = False
    except AssertionError, e:
        errored = True
    assert errored

    observed = aug_succeeds_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    try:
        observed = aug_fails.augment_keypoints(keypoints)
        errored = False
    except AssertionError, e:
        errored = True
    assert errored

def test_AssertShape():
    base_img = np.array([[0, 0, 1, 0],
                         [0, 0, 1, 0],
                         [0, 1, 1, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]
    images = np.array([base_img])
    images_list = [base_img]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    base_img_h4 = np.array([[0, 0, 1, 0],
                            [0, 0, 1, 0],
                            [0, 1, 1, 0],
                            [1, 0, 1, 0]], dtype=np.uint8)
    base_img_h4 = base_img_h4[:, :, np.newaxis]
    images_h4 = np.array([base_img_h4])
    keypoints_h4 = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img_h4.shape)]

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

    for _ in range(10):
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

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError, e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError, e:
            errored = True
        assert errored

    # any value for number of images allowed (None)
    aug = iaa.AssertShape((None, 3, 4, 1))
    aug_det = aug.to_deterministic()
    for _ in range(10):
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

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError, e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError, e:
            errored = True
        assert errored

    # list of possible choices [1, 3, 5] for height
    aug = iaa.AssertShape((1, [1, 3, 5], 4, 1))
    aug_det = aug.to_deterministic()
    for _ in range(10):
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

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError, e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError, e:
            errored = True
        assert errored

    # range of 1-3 for height (tuple comparison is a <= x < b, so we use (1,4) here)
    aug = iaa.AssertShape((1, (1, 4), 4, 1))
    aug_det = aug.to_deterministic()
    for _ in range(10):
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

        try:
            observed = aug.augment_images(images_h4)
            errored = False
        except AssertionError, e:
            errored = True
        assert errored

        try:
            observed = aug.augment_keypoints(keypoints_h4)
            errored = False
        except AssertionError, e:
            errored = True
        assert errored


def test_Fliplr():
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

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]
    keypoints_flipped = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=0, y=2)], shape=base_img.shape)]

    # 0% chance of flip
    aug = iaa.Fliplr(0)
    aug_det = aug.to_deterministic()

    for _ in range(10):
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

    # 100% chance of flip
    aug = iaa.Fliplr(1.0)
    aug_det = aug.to_deterministic()

    for _ in range(10):
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

    # 50% chance of flip
    aug = iaa.Fliplr(0.5)
    aug_det = aug.to_deterministic()

    nb_iterations = 1000
    nb_images_flipped = 0
    nb_images_flipped_det = 0
    nb_keypoints_flipped = 0
    nb_keypoints_flipped_det = 0
    for _ in range(nb_iterations):
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
    for _ in range(nb_iterations):
        observed = aug.augment_images(images_multi)
        for i in range(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos[i] += 1

        observed = aug_det.augment_images(images_multi)
        for i in range(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos_det[i] += 1

    for val in nb_flipped_by_pos:
        assert int(nb_iterations * 0.3) <= val <= int(nb_iterations * 0.7)

    for val in nb_flipped_by_pos_det:
        assert val in [0, nb_iterations]

def test_Flipud():
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

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]
    keypoints_flipped = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=2), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=0)], shape=base_img.shape)]

    # 0% chance of flip
    aug = iaa.Flipud(0)
    aug_det = aug.to_deterministic()

    for _ in range(10):
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

    # 100% chance of flip
    aug = iaa.Flipud(1.0)
    aug_det = aug.to_deterministic()

    for _ in range(10):
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

    # 50% chance of flip
    aug = iaa.Flipud(0.5)
    aug_det = aug.to_deterministic()

    nb_iterations = 1000
    nb_images_flipped = 0
    nb_images_flipped_det = 0
    nb_keypoints_flipped = 0
    nb_keypoints_flipped_det = 0
    for _ in range(nb_iterations):
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
    for _ in range(nb_iterations):
        observed = aug.augment_images(images_multi)
        for i in range(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos[i] += 1

        observed = aug_det.augment_images(images_multi)
        for i in range(len(images_multi)):
            if np.array_equal(observed[i], base_img_flipped):
                nb_flipped_by_pos_det[i] += 1

    for val in nb_flipped_by_pos:
        assert int(nb_iterations * 0.3) <= val <= int(nb_iterations * 0.7)

    for val in nb_flipped_by_pos_det:
        assert val in [0, nb_iterations]

def test_GaussianBlur():
    base_img = np.array([[0, 0, 0],
                         [0, 255, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]
    outer_pixels = ([], [])
    for i in range(base_img.shape[0]):
        for j in range(base_img.shape[1]):
            if i != j:
                outer_pixels[0].append(i)
                outer_pixels[1].append(j)

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

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
    for i in range(nb_iterations):
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

def test_AdditiveGaussianNoise():
    #base_img = np.array([[128, 128, 128],
    #                     [128, 128, 128],
    #                     [128, 128, 128]], dtype=np.uint8)
    base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128
    #base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no noise, shouldnt change anything
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    # zero-centered noise
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.2)
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
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.2)
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    nb_iterations = 10000
    values = []
    for i in range(nb_iterations):
        images_aug = aug.augment_images(images)
        values.append(images_aug[0, 0, 0, 0])
    values = np.array(values)
    assert np.min(values) == 0
    assert 0.1 < np.std(values) / 255.0 < 0.4

    # non-zero loc
    aug = iaa.AdditiveGaussianNoise(loc=0.25, scale=0.01)
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    nb_iterations = 10000
    values = []
    for i in range(nb_iterations):
        images_aug = aug.augment_images(images)
        values.append(images_aug[0, 0, 0, 0] - 128)
    values = np.array(values)
    assert 54 < np.average(values) < 74 # loc=0.25 should be around 255*0.25=64 average

    # varying locs
    aug = iaa.AdditiveGaussianNoise(loc=(0, 0.5), scale=0.0001)
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in range(nb_iterations):
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

    # varying stds
    aug = iaa.AdditiveGaussianNoise(loc=0, scale=(0.01, 0.2))
    aug_det = aug.to_deterministic()
    images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in range(nb_iterations):
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


def test_MultiplicativeGaussianNoise():
    pass

def test_ReplacingGaussianNoise():
    pass

def test_Dropout():
    pass

def test_Multiply():
    pass

def test_Affine():
    pass

def test_ElasticTransformation():
    pass

def create_random_images(size):
    return np.random.uniform(0, 255, size).astype(np.uint8)

def create_random_keypoints(size_images, nb_keypoints_per_img):
    result = []
    for i in range(size_images[0]):
        kps = []
        height, width = size_images[1], size_images[2]
        for i in range(nb_keypoints_per_img):
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

def keypoints_equal(kps1, kps2):
    if len(kps1) != len(kps2):
        return False

    for i in range(len(kps1)):
        a = kps1[i].keypoints
        b = kps2[i].keypoints
        if len(a) != len(b):
            return False

        for j in range(len(a)):
            if a[j].x != b[j].x or a[j].y != b[j].y:
                return False

    return True

if __name__ == "__main__":
    main()
