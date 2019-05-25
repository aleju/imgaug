from __future__ import print_function, division, absolute_import

import time
import itertools
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
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import imgaug.augmenters.meta as meta
from imgaug.testutils import reseed


def main():
    time_start = time.time()

    # TODO WithColorspace
    # TODO ChangeColorspace
    test_Grayscale()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


class TestWithHueAndSaturation(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        child = iaa.Noop()
        aug = iaa.WithHueAndSaturation(child, from_colorspace="BGR")
        assert isinstance(aug.children, list)
        assert len(aug.children) == 1
        assert aug.children[0] is child
        assert aug.from_colorspace == "BGR"

        aug = iaa.WithHueAndSaturation([child])
        assert isinstance(aug.children, list)
        assert len(aug.children) == 1
        assert aug.children[0] is child
        assert aug.from_colorspace == "RGB"

    def test_augment_images(self):
        def do_return_images(images, parents, hooks):
            assert images[0].dtype.name == "int16"
            return images

        aug_mock = mock.MagicMock(spec=meta.Augmenter)
        aug_mock.augment_images.side_effect = do_return_images
        aug = iaa.WithHueAndSaturation(aug_mock)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image_aug = aug.augment_images([image])[0]
        assert image_aug.dtype.name == "uint8"
        assert np.array_equal(image_aug, image)
        assert aug_mock.augment_images.call_count == 1

    def test_augment_images__hue(self):
        def augment_images(images, random_state, parents, hooks):
            assert images[0].dtype.name == "int16"
            images = np.copy(images)
            images[..., 0] += 10
            return images

        aug = iaa.WithHueAndSaturation(iaa.Lambda(func_images=augment_images))

        # example image
        image = np.arange(0, 255).reshape((1, 255, 1)).astype(np.uint8)
        image = np.tile(image, (1, 1, 3))
        image[..., 0] += 0
        image[..., 1] += 1
        image[..., 2] += 2

        # compute expected output
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv = image_hsv.astype(np.int16)
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/180)*255).astype(np.int16)
        image_hsv[..., 0] += 10
        image_hsv[..., 0] = np.mod(image_hsv[..., 0], 255)
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/255)*180).astype(np.int16)
        image_hsv = image_hsv.astype(np.uint8)
        image_expected = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        assert not np.array_equal(image_expected, image)

        # augment and verify
        images_aug = aug.augment_images(np.stack([image, image], axis=0))
        assert ia.is_np_array(images_aug)
        for image_aug in images_aug:
            assert image_aug.shape == (1, 255, 3)
            assert np.array_equal(image_aug, image_expected)

    def test_augment_images__saturation(self):
        def augment_images(images, random_state, parents, hooks):
            assert images[0].dtype.name == "int16"
            images = np.copy(images)
            images[..., 1] += 10
            return images

        aug = iaa.WithHueAndSaturation(iaa.Lambda(func_images=augment_images))

        # example image
        image = np.arange(0, 255).reshape((1, 255, 1)).astype(np.uint8)
        image = np.tile(image, (1, 1, 3))
        image[..., 0] += 0
        image[..., 1] += 1
        image[..., 2] += 2

        # compute expected output
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv = image_hsv.astype(np.int16)
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/180)*255).astype(np.int16)
        image_hsv[..., 1] += 10
        image_hsv[..., 1] = np.clip(image_hsv[..., 1], 0, 255)
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/255)*180).astype(np.int16)
        image_hsv = image_hsv.astype(np.uint8)
        image_expected = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        assert not np.array_equal(image_expected, image)

        # augment and verify
        images_aug = aug.augment_images(np.stack([image, image], axis=0))
        assert ia.is_np_array(images_aug)
        for image_aug in images_aug:
            assert image_aug.shape == (1, 255, 3)
            assert np.array_equal(image_aug, image_expected)

    def test_augment_heatmaps(self):
        from imgaug.augmentables.heatmaps import HeatmapsOnImage

        def do_return_augmentables(heatmaps, parents, hooks):
            return heatmaps

        aug_mock = mock.MagicMock(spec=meta.Augmenter)
        aug_mock.augment_heatmaps.side_effect = do_return_augmentables
        hm = np.ones((8, 12, 1), dtype=np.float32)
        hmoi = HeatmapsOnImage(hm, shape=(16, 24, 3))

        aug = iaa.WithHueAndSaturation(aug_mock)
        hmoi_aug = aug.augment_heatmaps(hmoi)
        assert hmoi_aug.shape == (16, 24, 3)
        assert hmoi_aug.arr_0to1.shape == (8, 12, 1)

        assert aug_mock.augment_heatmaps.call_count == 1

    def test_augment_keypoints(self):
        from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

        def do_return_augmentables(keypoints_on_images, parents, hooks):
            return keypoints_on_images

        aug_mock = mock.MagicMock(spec=meta.Augmenter)
        aug_mock.augment_keypoints.side_effect = do_return_augmentables
        kpsoi = KeypointsOnImage.from_xy_array(np.float32([
            [0, 0],
            [5, 1]
        ]), shape=(16, 24, 3))

        aug = iaa.WithHueAndSaturation(aug_mock)
        kpsoi_aug = aug.augment_keypoints(kpsoi)
        assert kpsoi_aug.shape == (16, 24, 3)
        assert kpsoi.keypoints[0].x == 0
        assert kpsoi.keypoints[0].y == 0
        assert kpsoi.keypoints[1].x == 5
        assert kpsoi.keypoints[1].y == 1

        assert aug_mock.augment_keypoints.call_count == 1

    def test__to_deterministic(self):
        aug = iaa.WithHueAndSaturation([iaa.Noop()], from_colorspace="BGR")
        aug_det = aug.to_deterministic()

        assert not aug.deterministic  # ensure copy
        assert not aug.children[0].deterministic

        assert aug_det.deterministic
        assert isinstance(aug_det.children[0], iaa.Noop)
        assert aug_det.children[0].deterministic

    def test_get_parameters(self):
        aug = iaa.WithHueAndSaturation([iaa.Noop()], from_colorspace="BGR")
        assert aug.get_parameters()[0] == "BGR"

    def test_get_children_lists(self):
        child = iaa.Noop()
        aug = iaa.WithHueAndSaturation(child)
        children_lists = aug.get_children_lists()
        assert len(children_lists) == 1
        assert len(children_lists[0]) == 1
        assert children_lists[0][0] is child

        child = iaa.Noop()
        aug = iaa.WithHueAndSaturation([child])
        children_lists = aug.get_children_lists()
        assert len(children_lists) == 1
        assert len(children_lists[0]) == 1
        assert children_lists[0][0] is child

    def test___str__(self):
        child = iaa.Sequential([iaa.Noop(name="foo")])
        aug = iaa.WithHueAndSaturation(child)
        observed = aug.__str__()
        expected = (
            "WithHueAndSaturation("
            "from_colorspace=RGB, "
            "name=UnnamedWithHueAndSaturation, "
            "children=[%s], "
            "deterministic=False"
            ")" % (child.__str__(),)
        )
        assert observed == expected


class TestMultiplyHueAndSaturation(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_returns_correct_objects__mul(self):
        aug = iaa.MultiplyHueAndSaturation(
            (0.9, 1.1), per_channel=True)
        assert isinstance(aug, iaa.WithHueAndSaturation)
        assert isinstance(aug.children, iaa.Sequential)
        assert len(aug.children) == 1
        assert isinstance(aug.children[0], iaa.Multiply)
        assert isinstance(aug.children[0].mul, iap.Uniform)
        assert np.isclose(aug.children[0].mul.a.value, 0.9)
        assert np.isclose(aug.children[0].mul.b.value, 1.1)
        assert isinstance(aug.children[0].per_channel, iap.Deterministic)
        assert aug.children[0].per_channel.value == 1

    def test_returns_correct_objects__mul_hue(self):
        aug = iaa.MultiplyHueAndSaturation(mul_hue=(0.9, 1.1))
        assert isinstance(aug, iaa.WithHueAndSaturation)
        assert isinstance(aug.children, iaa.Sequential)
        assert len(aug.children) == 1
        assert isinstance(aug.children[0], iaa.WithChannels)
        assert aug.children[0].channels == [0]
        assert len(aug.children[0].children) == 1
        assert isinstance(aug.children[0].children[0], iaa.Multiply)
        assert isinstance(aug.children[0].children[0].mul, iap.Uniform)
        assert np.isclose(aug.children[0].children[0].mul.a.value, 0.9)
        assert np.isclose(aug.children[0].children[0].mul.b.value, 1.1)

    def test_returns_correct_objects__mul_saturation(self):
        aug = iaa.MultiplyHueAndSaturation(mul_saturation=(0.9, 1.1))
        assert isinstance(aug, iaa.WithHueAndSaturation)
        assert isinstance(aug.children, iaa.Sequential)
        assert len(aug.children) == 1
        assert isinstance(aug.children[0], iaa.WithChannels)
        assert aug.children[0].channels == [1]
        assert len(aug.children[0].children) == 1
        assert isinstance(aug.children[0].children[0], iaa.Multiply)
        assert isinstance(aug.children[0].children[0].mul, iap.Uniform)
        assert np.isclose(aug.children[0].children[0].mul.a.value, 0.9)
        assert np.isclose(aug.children[0].children[0].mul.b.value, 1.1)

    def test_returns_correct_objects__mul_hue_and_mul_saturation(self):
        aug = iaa.MultiplyHueAndSaturation(mul_hue=(0.9, 1.1),
                                           mul_saturation=(0.8, 1.2))
        assert isinstance(aug, iaa.WithHueAndSaturation)
        assert isinstance(aug.children, iaa.Sequential)
        assert len(aug.children) == 2

        assert isinstance(aug.children[0], iaa.WithChannels)
        assert aug.children[0].channels == [0]
        assert len(aug.children[0].children) == 1
        assert isinstance(aug.children[0].children[0], iaa.Multiply)
        assert isinstance(aug.children[0].children[0].mul, iap.Uniform)
        assert np.isclose(aug.children[0].children[0].mul.a.value, 0.9)
        assert np.isclose(aug.children[0].children[0].mul.b.value, 1.1)

        assert isinstance(aug.children[1], iaa.WithChannels)
        assert aug.children[1].channels == [1]
        assert len(aug.children[0].children) == 1
        assert isinstance(aug.children[1].children[0], iaa.Multiply)
        assert isinstance(aug.children[1].children[0].mul, iap.Uniform)
        assert np.isclose(aug.children[1].children[0].mul.a.value, 0.8)
        assert np.isclose(aug.children[1].children[0].mul.b.value, 1.2)

    def test_augment_images__mul(self):
        aug = iaa.MultiplyHueAndSaturation(1.5)

        # example image
        image = np.arange(0, 255).reshape((1, 255, 1)).astype(np.uint8)
        image = np.tile(image, (1, 1, 3))
        image[..., 0] += 0
        image[..., 1] += 5
        image[..., 2] += 10

        # compute expected output
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv = image_hsv.astype(np.int16)  # simulate WithHueAndSaturation
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/180)*255).astype(np.int16)
        image_hsv = image_hsv.astype(np.float32)  # simulate Multiply

        image_hsv[..., 0] *= 1.5
        image_hsv[..., 1] *= 1.5
        image_hsv = np.round(image_hsv).astype(np.int16)

        image_hsv[..., 0] = np.mod(image_hsv[..., 0], 255)
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/255)*180).astype(np.int16)
        image_hsv[..., 1] = np.clip(image_hsv[..., 1], 0, 255)

        image_hsv = image_hsv.astype(np.uint8)
        image_expected = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        assert not np.array_equal(image_expected, image)

        # augment and verify
        images_aug = aug.augment_images(np.stack([image, image], axis=0))
        assert ia.is_np_array(images_aug)
        for image_aug in images_aug:
            assert image_aug.shape == (1, 255, 3)
            diff = np.abs(image_aug.astype(np.int16) - image_expected)
            assert np.all(diff <= 1)

    def test_augment_images__mul_hue(self):
        # this is almost identical to test_augment_images__mul
        # only
        #     aug = ...
        # and
        #     image_hsv[...] *= 1.2
        # have been changed

        aug = iaa.MultiplyHueAndSaturation(mul_hue=1.5)  # changed over __mul

        # example image
        image = np.arange(0, 255).reshape((1, 255, 1)).astype(np.uint8)
        image = np.tile(image, (1, 1, 3))
        image[..., 0] += 0
        image[..., 1] += 5
        image[..., 2] += 10

        # compute expected output
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv = image_hsv.astype(np.int16)  # simulate WithHueAndSaturation
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/180)*255).astype(np.int16)
        image_hsv = image_hsv.astype(np.float32)  # simulate Multiply

        image_hsv[..., 0] *= 1.5
        image_hsv[..., 1] *= 1.0  # changed over __mul
        image_hsv = np.round(image_hsv).astype(np.int16)

        image_hsv[..., 0] = np.mod(image_hsv[..., 0], 255)
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/255)*180).astype(np.int16)
        image_hsv[..., 1] = np.clip(image_hsv[..., 1], 0, 255)

        image_hsv = image_hsv.astype(np.uint8)
        image_expected = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        assert not np.array_equal(image_expected, image)

        # augment and verify
        images_aug = aug.augment_images(np.stack([image, image], axis=0))
        assert ia.is_np_array(images_aug)
        for image_aug in images_aug:
            assert image_aug.shape == (1, 255, 3)
            diff = np.abs(image_aug.astype(np.int16) - image_expected)
            assert np.all(diff <= 1)

    def test_augment_images__mul_saturation(self):
        # this is almost identical to test_augment_images__mul
        # only
        #     aug = ...
        # and
        #     image_hsv[...] *= 1.2
        # have been changed

        aug = iaa.MultiplyHueAndSaturation(mul_saturation=1.5)  # changed

        # example image
        image = np.arange(0, 255).reshape((1, 255, 1)).astype(np.uint8)
        image = np.tile(image, (1, 1, 3))
        image[..., 0] += 0
        image[..., 1] += 5
        image[..., 2] += 10

        # compute expected output
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv = image_hsv.astype(np.int16)  # simulate WithHueAndSaturation
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/180)*255).astype(np.int16)
        image_hsv = image_hsv.astype(np.float32)  # simulate Multiply

        image_hsv[..., 0] *= 1.0  # changed over __mul
        image_hsv[..., 1] *= 1.5
        image_hsv = np.round(image_hsv).astype(np.int16)

        image_hsv[..., 0] = np.mod(image_hsv[..., 0], 255)
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/255)*180).astype(np.int16)
        image_hsv[..., 1] = np.clip(image_hsv[..., 1], 0, 255)

        image_hsv = image_hsv.astype(np.uint8)
        image_expected = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        assert not np.array_equal(image_expected, image)

        # augment and verify
        images_aug = aug.augment_images(np.stack([image, image], axis=0))
        assert ia.is_np_array(images_aug)
        for image_aug in images_aug:
            assert image_aug.shape == (1, 255, 3)
            diff = np.abs(image_aug.astype(np.int16) - image_expected)
            assert np.all(diff <= 1)

    def test_augment_images__mul_hue_and_mul_saturation(self):
        # this is almost identical to test_augment_images__mul
        # only
        #     aug = ...
        # and
        #     image_hsv[...] *= 1.2
        # have been changed

        aug = iaa.MultiplyHueAndSaturation(mul_hue=1.5,
                                           mul_saturation=1.6)  # changed

        # example image
        image = np.arange(0, 255).reshape((1, 255, 1)).astype(np.uint8)
        image = np.tile(image, (1, 1, 3))
        image[..., 0] += 0
        image[..., 1] += 5
        image[..., 2] += 10

        # compute expected output
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv = image_hsv.astype(np.int16)  # simulate WithHueAndSaturation
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/180)*255).astype(np.int16)
        image_hsv = image_hsv.astype(np.float32)  # simulate Multiply

        image_hsv[..., 0] *= 1.5
        image_hsv[..., 1] *= 1.6  # changed over __mul
        image_hsv = np.round(image_hsv).astype(np.int16)

        image_hsv[..., 0] = np.mod(image_hsv[..., 0], 255)
        image_hsv[..., 0] = (
            (image_hsv[..., 0].astype(np.float32)/255)*180).astype(np.int16)
        image_hsv[..., 1] = np.clip(image_hsv[..., 1], 0, 255)

        image_hsv = image_hsv.astype(np.uint8)
        image_expected = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        assert not np.array_equal(image_expected, image)

        # augment and verify
        images_aug = aug.augment_images(np.stack([image, image], axis=0))
        assert ia.is_np_array(images_aug)
        for image_aug in images_aug:
            assert image_aug.shape == (1, 255, 3)
            diff = np.abs(image_aug.astype(np.int16) - image_expected)
            assert np.all(diff <= 1)

    def test_augment_images__deterministic(self):
        rs = np.random.RandomState(1)
        images = rs.randint(0, 255, size=(32, 4, 4, 3), dtype=np.uint8)

        for deterministic in [False, True]:
            aug = iaa.MultiplyHueAndSaturation(mul=(0.1, 5.0),
                                               deterministic=deterministic)
            images_aug1 = aug.augment_images(images)
            images_aug2 = aug.augment_images(images)
            equal = np.array_equal(images_aug1, images_aug2)
            if deterministic:
                assert equal
            else:
                assert not equal

            aug = iaa.MultiplyHueAndSaturation(mul_hue=(0.1, 5.0),
                                               mul_saturation=(0.1, 5.0),
                                               deterministic=deterministic)
            images_aug1 = aug.augment_images(images)
            images_aug2 = aug.augment_images(images)
            equal = np.array_equal(images_aug1, images_aug2)
            if deterministic:
                assert equal
            else:
                assert not equal


class TestMultiplyToHue(unittest.TestCase):
    def test_returns_correct_class(self):
        # this test is practically identical to
        # TestMultiplyToHueAndSaturation.test_returns_correct_objects__mul_hue
        aug = iaa.MultiplyHue((0.9, 1.1))
        assert isinstance(aug, iaa.WithHueAndSaturation)
        assert isinstance(aug.children, iaa.Sequential)
        assert len(aug.children) == 1
        assert isinstance(aug.children[0], iaa.WithChannels)
        assert aug.children[0].channels == [0]
        assert len(aug.children[0].children) == 1
        assert isinstance(aug.children[0].children[0], iaa.Multiply)
        assert isinstance(aug.children[0].children[0].mul, iap.Uniform)
        assert np.isclose(aug.children[0].children[0].mul.a.value, 0.9)
        assert np.isclose(aug.children[0].children[0].mul.b.value, 1.1)


class TestMultiplyToSaturation(unittest.TestCase):
    def test_returns_correct_class(self):
        # this test is practically identical to
        # TestMultiplyToHueAndSaturation
        #     .test_returns_correct_objects__mul_saturation
        aug = iaa.MultiplySaturation((0.9, 1.1))
        assert isinstance(aug, iaa.WithHueAndSaturation)
        assert isinstance(aug.children, iaa.Sequential)
        assert len(aug.children) == 1
        assert isinstance(aug.children[0], iaa.WithChannels)
        assert aug.children[0].channels == [1]
        assert len(aug.children[0].children) == 1
        assert isinstance(aug.children[0].children[0], iaa.Multiply)
        assert isinstance(aug.children[0].children[0].mul, iap.Uniform)
        assert np.isclose(aug.children[0].children[0].mul.a.value, 0.9)
        assert np.isclose(aug.children[0].children[0].mul.b.value, 1.1)


class TestAddToHueAndSaturation(unittest.TestCase):
    def setUp(self):
        reseed()

    # interestingly, when using this RGB2HSV and HSV2RGB conversion from
    # skimage, the results differ quite a bit from the cv2 ones
    """
    def _add_hue_saturation(img, value):
        img_hsv = color.rgb2hsv(img / 255.0)
        img_hsv[..., 0:2] += (value / 255.0)
        return color.hsv2rgb(img_hsv) * 255
    """

    @classmethod
    def _add_hue_saturation(cls, img, value=None, value_hue=None,
                            value_saturation=None):
        if value is not None:
            assert value_hue is None and value_saturation is None
        else:
            assert value_hue is not None or value_saturation is not None

        if value is not None:
            value_hue = value
            value_saturation = value
        else:
            value_hue = value_hue or 0
            value_saturation = value_saturation or 0

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv = img_hsv.astype(np.int32)
        img_hsv[..., 0] = np.mod(
            img_hsv[..., 0] + (value_hue/255.0) * (360/2), 180)
        img_hsv[..., 1] = np.clip(
            img_hsv[..., 1] + value_saturation, 0, 255)
        img_hsv = img_hsv.astype(np.uint8)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    def test___init__(self):
        aug = iaa.AddToHueAndSaturation((-20, 20))
        assert isinstance(aug.value, iap.DiscreteUniform)
        assert aug.value.a.value == -20
        assert aug.value.b.value == 20
        assert aug.value_hue is None
        assert aug.value_saturation is None
        assert isinstance(aug.per_channel, iap.Deterministic)
        assert aug.per_channel.value == 0

    def test___init___value_none(self):
        aug = iaa.AddToHueAndSaturation(value_hue=(-20, 20),
                                        value_saturation=[0, 5, 10])
        assert aug.value is None
        assert isinstance(aug.value_hue, iap.DiscreteUniform)
        assert isinstance(aug.value_saturation, iap.Choice)
        assert aug.value_hue.a.value == -20
        assert aug.value_hue.b.value == 20
        assert aug.value_saturation.a == [0, 5, 10]
        assert isinstance(aug.per_channel, iap.Deterministic)
        assert aug.per_channel.value == 0

    def test___init___per_channel(self):
        aug = iaa.AddToHueAndSaturation(per_channel=0.5)
        assert aug.value is None
        assert aug.value_hue is None
        assert aug.value_saturation is None
        assert isinstance(aug.per_channel, iap.Binomial)
        assert np.isclose(aug.per_channel.p.value, 0.5)

    def test_augment_images(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        gen = itertools.product([False, True], ["cv2", "numpy"])
        for per_channel, backend in gen:
            with self.subTest(per_channel=per_channel, backend=backend):
                aug = iaa.AddToHueAndSaturation(0, per_channel=per_channel)
                aug.backend = backend
                observed = aug.augment_image(base_img)
                expected = base_img
                assert np.allclose(observed, expected)

                aug = iaa.AddToHueAndSaturation(30, per_channel=per_channel)
                aug.backend = backend
                observed = aug.augment_image(base_img)
                expected = self._add_hue_saturation(base_img, 30)
                diff = np.abs(observed.astype(np.float32) - expected)
                assert np.all(diff <= 1)

                aug = iaa.AddToHueAndSaturation(255, per_channel=per_channel)
                aug.backend = backend
                observed = aug.augment_image(base_img)
                expected = self._add_hue_saturation(base_img, 255)
                diff = np.abs(observed.astype(np.float32) - expected)
                assert np.all(diff <= 1)

                aug = iaa.AddToHueAndSaturation(-255, per_channel=per_channel)
                aug.backend = backend
                observed = aug.augment_image(base_img)
                expected = self._add_hue_saturation(base_img, -255)
                diff = np.abs(observed.astype(np.float32) - expected)
                assert np.all(diff <= 1)

    def test_augment_images__different_hue_and_saturation__no_per_channel(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                arr = np.float32([10, 20])
                return np.tile(arr[np.newaxis, :], (size[0], 1))

        aug = iaa.AddToHueAndSaturation(value=_DummyParam(), per_channel=False)
        img_expected = self._add_hue_saturation(base_img, value=10)
        img_observed = aug.augment_image(base_img)

        assert np.array_equal(img_observed, img_expected)

    def test_augment_images__different_hue_and_saturation__per_channel(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                arr = np.float32([10, 20])
                return np.tile(arr[np.newaxis, :], (size[0], 1))

        aug = iaa.AddToHueAndSaturation(value=_DummyParam(), per_channel=True)
        img_expected = self._add_hue_saturation(
            base_img, value_hue=10, value_saturation=20)
        img_observed = aug.augment_image(base_img)

        assert np.array_equal(img_observed, img_expected)

    def test_augment_images__different_hue_and_saturation__mixed_perchan(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParamValue(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                arr = np.float32([10, 20])
                return np.tile(arr[np.newaxis, :], (size[0], 1))

        class _DummyParamPerChannel(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                assert size == (3,)
                return np.float32([1.0, 0.0, 1.0])

        aug = iaa.AddToHueAndSaturation(
            value=_DummyParamValue(), per_channel=_DummyParamPerChannel())

        img_expected1 = self._add_hue_saturation(
            base_img, value_hue=10, value_saturation=20)
        img_expected2 = self._add_hue_saturation(
            base_img, value_hue=10, value_saturation=10)
        img_expected3 = self._add_hue_saturation(
            base_img, value_hue=10, value_saturation=20)

        img_observed1, img_observed2, img_observed3, = \
            aug.augment_images([base_img] * 3)

        assert np.array_equal(img_observed1, img_expected1)
        assert np.array_equal(img_observed2, img_expected2)
        assert np.array_equal(img_observed3, img_expected3)

    def test_augment_images__list_as_value(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        aug = iaa.AddToHueAndSaturation([0, 10, 20])
        base_img = base_img[0:1, 0:1, :]
        expected_imgs = [
            iaa.AddToHueAndSaturation(0).augment_image(base_img),
            iaa.AddToHueAndSaturation(10).augment_image(base_img),
            iaa.AddToHueAndSaturation(20).augment_image(base_img)
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
        assert all([n_exp - n_exp_tol < v < n_exp + n_exp_tol
                    for v in seen.values()])

    def test_augment_images__value_hue(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                return np.float32([10, 20, 30])

        aug = iaa.AddToHueAndSaturation(value_hue=_DummyParam())

        img_expected1 = self._add_hue_saturation(base_img, value_hue=10)
        img_expected2 = self._add_hue_saturation(base_img, value_hue=20)
        img_expected3 = self._add_hue_saturation(base_img, value_hue=30)

        img_observed1, img_observed2, img_observed3 = \
            aug.augment_images([base_img] * 3)

        assert np.array_equal(img_observed1, img_expected1)
        assert np.array_equal(img_observed2, img_expected2)
        assert np.array_equal(img_observed3, img_expected3)

    def test_augment_images__value_saturation(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                return np.float32([10, 20, 30])

        aug = iaa.AddToHueAndSaturation(value_saturation=_DummyParam())

        img_expected1 = self._add_hue_saturation(base_img, value_saturation=10)
        img_expected2 = self._add_hue_saturation(base_img, value_saturation=20)
        img_expected3 = self._add_hue_saturation(base_img, value_saturation=30)

        img_observed1, img_observed2, img_observed3 = \
            aug.augment_images([base_img] * 3)

        assert np.array_equal(img_observed1, img_expected1)
        assert np.array_equal(img_observed2, img_expected2)
        assert np.array_equal(img_observed3, img_expected3)

    def test_augment_images__value_hue_and_value_saturation(self):
        base_img = np.zeros((2, 2, 3), dtype=np.uint8)
        base_img[..., 0] += 20
        base_img[..., 1] += 40
        base_img[..., 2] += 60

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                return np.float32([10, 20, 30])

        aug = iaa.AddToHueAndSaturation(value_hue=_DummyParam(),
                                        value_saturation=_DummyParam()+40)

        img_expected1 = self._add_hue_saturation(base_img, value_hue=10,
                                                 value_saturation=40+10)
        img_expected2 = self._add_hue_saturation(base_img, value_hue=20,
                                                 value_saturation=40+20)
        img_expected3 = self._add_hue_saturation(base_img, value_hue=30,
                                                 value_saturation=40+30)

        img_observed1, img_observed2, img_observed3 = \
            aug.augment_images([base_img] * 3)

        assert np.array_equal(img_observed1, img_expected1)
        assert np.array_equal(img_observed2, img_expected2)
        assert np.array_equal(img_observed3, img_expected3)

    def test_get_parameters(self):
        aug = iaa.AddToHueAndSaturation((-20, 20), per_channel=0.5)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.DiscreteUniform)
        assert params[0].a.value == -20
        assert params[0].b.value == 20
        assert params[1] is None
        assert params[2] is None
        assert isinstance(params[3], iap.Binomial)
        assert np.isclose(params[3].p.value, 0.5)

    def test_get_parameters_value_hue_and_value_saturation(self):
        aug = iaa.AddToHueAndSaturation(value_hue=(-20, 20),
                                        value_saturation=5)
        params = aug.get_parameters()
        assert params[0] is None
        assert isinstance(params[1], iap.DiscreteUniform)
        assert params[1].a.value == -20
        assert params[1].b.value == 20
        assert isinstance(params[2], iap.Deterministic)
        assert params[2].value == 5
        assert isinstance(params[3], iap.Deterministic)
        assert params[3].value == 0


class TestAddToHue(unittest.TestCase):
    def test_returns_correct_class(self):
        aug = iaa.AddToHue((-20, 20))
        assert isinstance(aug, iaa.AddToHueAndSaturation)
        assert isinstance(aug.value_hue, iap.DiscreteUniform)
        assert aug.value_hue.a.value == -20
        assert aug.value_hue.b.value == 20


class TestAddToSaturation(unittest.TestCase):
    def test_returns_correct_class(self):
        aug = iaa.AddToSaturation((-20, 20))
        assert isinstance(aug, iaa.AddToHueAndSaturation)
        assert isinstance(aug.value_saturation, iap.DiscreteUniform)
        assert aug.value_saturation.a.value == -20
        assert aug.value_saturation.b.value == 20


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
    base_img = np.uint8([255, 0, 0]).reshape((1, 1, 3))
    base_img_float = base_img.astype(np.float64) / 255.0
    base_img_gray = iaa.Grayscale(1.0).augment_image(base_img).astype(np.float64) / 255.0
    distance_max = np.linalg.norm(base_img_gray.flatten() - base_img_float.flatten())
    nb_iterations = 1000
    distances = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img).astype(np.float64) / 255.0
        distance = np.linalg.norm(observed.flatten() - base_img_float.flatten()) / distance_max
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


if __name__ == "__main__":
    main()
