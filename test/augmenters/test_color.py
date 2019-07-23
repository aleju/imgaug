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


# Note that TestUniformColorQuantization inherits from this class,
# which is why it contains the overwriteable @property functions
class TestKMeansColorQuantization(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def augmenter(self):
        return iaa.KMeansColorQuantization

    @property
    def quantization_func_name(self):
        return "imgaug.augmenters.color.quantize_colors_kmeans"

    def test___init___defaults(self):
        aug = self.augmenter()
        assert isinstance(aug.n_colors, iap.DiscreteUniform)
        assert aug.n_colors.a.value == 2
        assert aug.n_colors.b.value == 16
        assert aug.from_colorspace == iaa.ChangeColorspace.RGB
        assert isinstance(aug.to_colorspace, list)
        assert aug.to_colorspace == [iaa.ChangeColorspace.RGB,
                                     iaa.ChangeColorspace.Lab]
        assert aug.max_size == 128
        assert aug.interpolation == "linear"

    def test___init___custom_parameters(self):
        aug = self.augmenter(
            n_colors=(5, 8),
            from_colorspace=iaa.ChangeColorspace.BGR,
            to_colorspace=[iaa.ChangeColorspace.HSV, iaa.ChangeColorspace.Lab],
            max_size=None,
            interpolation="cubic"
        )
        assert isinstance(aug.n_colors, iap.DiscreteUniform)
        assert aug.n_colors.a.value == 5
        assert aug.n_colors.b.value == 8
        assert aug.from_colorspace == iaa.ChangeColorspace.BGR
        assert isinstance(aug.to_colorspace, list)
        assert aug.to_colorspace == [iaa.ChangeColorspace.HSV,
                                     iaa.ChangeColorspace.Lab]
        assert aug.max_size is None
        assert aug.interpolation == "cubic"

    def test_n_colors_deterministic(self):
        aug = self.augmenter(n_colors=5)
        mock_quantize_func = mock.MagicMock(
            return_value=np.zeros((4, 4, 3), dtype=np.uint8))

        fname = self.quantization_func_name
        with mock.patch(fname, mock_quantize_func):
            _ = aug.augment_image(np.zeros((4, 4, 3), dtype=np.uint8))

        # call 0, args, argument 1
        assert mock_quantize_func.call_args_list[0][0][1] == 5

    def test_n_colors_tuple(self):
        aug = self.augmenter(n_colors=(2, 1000))
        mock_quantize_func = mock.MagicMock(
            return_value=np.zeros((4, 4, 3), dtype=np.uint8))

        n_images = 10
        fname = self.quantization_func_name
        with mock.patch(fname, mock_quantize_func):
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            _ = aug.augment_images([image] * n_images)

        # call i, args, argument 1
        n_colors = [mock_quantize_func.call_args_list[i][0][1]
                    for i in sm.xrange(n_images)]
        assert all([2 <= n_colors_i <= 1000 for n_colors_i in n_colors])
        assert len(set(n_colors)) > 1

    def test_to_colorspace(self):
        image = np.arange(3*3*3, dtype=np.uint8).reshape((3, 3, 3))
        aug = self.augmenter(to_colorspace="HSV")
        mock_quantize_func = mock.MagicMock(
            return_value=np.zeros((4, 4, 3), dtype=np.uint8))

        fname = self.quantization_func_name
        with mock.patch(fname, mock_quantize_func):
            _ = aug.augment_image(image)

        # call 0, kwargs, argument 'to_colorspace'
        expected = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        assert np.array_equal(mock_quantize_func.call_args_list[0][0][0],
                              expected)

    def test_to_colorspace_is_none(self):
        image = np.arange(3*3*3, dtype=np.uint8).reshape((3, 3, 3))
        aug = self.augmenter(to_colorspace=None)
        mock_quantize_func = mock.MagicMock(
            return_value=np.zeros((4, 4, 3), dtype=np.uint8))

        fname = self.quantization_func_name
        with mock.patch(fname, mock_quantize_func):
            _ = aug.augment_image(image)

        # call 0, kwargs, argument 'to_colorspace'
        assert np.array_equal(mock_quantize_func.call_args_list[0][0][0],
                              image)

    def test_from_colorspace(self):
        aug = self.augmenter(from_colorspace="BGR")
        mock_change_colorspace = mock.MagicMock()
        mock_change_colorspace.return_value = mock_change_colorspace
        mock_change_colorspace.augment_image.side_effect = lambda img: img
        mock_change_colorspace._draw_samples.return_value = (None, ["foo"])

        fname = "imgaug.augmenters.color.ChangeColorspace"
        with mock.patch(fname, mock_change_colorspace):
            _ = aug.augment_image(np.zeros((4, 4, 3), dtype=np.uint8))

        # call 0, kwargs, argument 'from_colorspace'
        assert (
            mock_change_colorspace.call_args_list[0][1]["from_colorspace"]
            == "BGR")
        # call 1, kwargs, argument 'from_colorspace' (inverse transform)
        assert (
            mock_change_colorspace.call_args_list[1][1]["from_colorspace"]
            == "foo")

    def test_max_size_is_none(self):
        image = np.zeros((1000, 4, 3), dtype=np.uint8)
        aug = iaa.KMeansColorQuantization(max_size=None)
        mock_imresize = mock.MagicMock(return_value=image)

        fname = "imgaug.imresize_single_image"
        with mock.patch(fname, mock_imresize):
            image_aug = aug.augment_image(image)
            assert image_aug.shape == image.shape

        assert mock_imresize.call_count == 0

    def test_max_size_is_int_and_resize_necessary(self):
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        aug = self.augmenter(max_size=100)
        mock_imresize = mock.MagicMock(
            return_value=np.zeros((100, 50, 3), dtype=np.uint8))

        fname = "imgaug.imresize_single_image"
        with mock.patch(fname, mock_imresize):
            _ = aug.augment_image(image)

        # call 0, args, argument 1 (size)
        # call 1, args, argument 1 (size)
        assert mock_imresize.call_count == 2
        assert mock_imresize.call_args_list[0][0][1] == (100, 50)
        assert mock_imresize.call_args_list[1][0][1] == image.shape[0:2]

    def test_max_size_is_int_and_resize_not_necessary(self):
        image = np.zeros((99, 4, 3), dtype=np.uint8)
        aug = self.augmenter(max_size=100)
        mock_imresize = mock.MagicMock(return_value=image)

        fname = "imgaug.imresize_single_image"
        with mock.patch(fname, mock_imresize):
            image_aug = aug.augment_image(image)
            assert image_aug.shape == image.shape

        assert mock_imresize.call_count == 0

    def test_interpolation(self):
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        aug = self.augmenter(max_size=100, interpolation="cubic")
        mock_imresize = mock.MagicMock(
            return_value=np.zeros((100, 50, 3), dtype=np.uint8))

        fname = "imgaug.imresize_single_image"
        with mock.patch(fname, mock_imresize):
            _ = aug.augment_image(image)

        # downscaling
        # call 0, args, argument 1 (sizes)
        # call 0, kwargs, argument "interpolation"
        assert mock_imresize.call_args_list[0][0][1] == (100, 50)
        assert mock_imresize.call_args_list[0][1]["interpolation"] == "cubic"

        # upscaling
        # call 1, args, argument 1 (sizes)
        # call 1, kwargs, argument "interpolation"
        assert mock_imresize.call_args_list[1][0][1] == image.shape[0:2]
        assert mock_imresize.call_args_list[1][1]["interpolation"] == "cubic"

    def test_images_with_1_channel_integrationtest(self):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])
        expected = np.uint8([
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ])

        aug = self.augmenter(
            n_colors=2,
            from_colorspace="RGB",
            to_colorspace="RGB",
            max_size=None)

        observed = aug(image=image)

        assert np.array_equal(observed, expected)

    def test_images_with_3_channels_integrationtest(self):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])
        expected = np.uint8([
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ])

        image = np.tile(image[..., np.newaxis], (1, 1, 3))
        expected = np.tile(expected[..., np.newaxis], (1, 1, 3))

        aug = self.augmenter(
            n_colors=2,
            from_colorspace="RGB",
            to_colorspace="RGB",
            max_size=None)

        observed = aug(image=image)

        assert np.array_equal(observed, expected)

    def test_images_with_4_channels_integrationtest(self):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])
        expected = np.uint8([
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ])

        image = np.tile(image[..., np.newaxis], (1, 1, 4))
        expected = np.tile(expected[..., np.newaxis], (1, 1, 3))

        # alpha channel is expected to not be altered by quantization
        expected = np.concatenate([expected, image[:, :, 3:4]], axis=-1)

        aug = self.augmenter(
            n_colors=2,
            from_colorspace="RGB",
            to_colorspace="RGB",
            max_size=None)

        observed = aug(image=image)

        assert np.array_equal(observed, expected)

    def test_get_parameters(self):
        aug = self.augmenter(
            n_colors=(5, 8),
            from_colorspace=iaa.ChangeColorspace.BGR,
            to_colorspace=[iaa.ChangeColorspace.HSV, iaa.ChangeColorspace.Lab],
            max_size=None,
            interpolation="cubic"
        )
        params = aug.get_parameters()
        assert isinstance(params[0], iap.DiscreteUniform)
        assert params[0].a.value == 5
        assert params[0].b.value == 8
        assert params[1] == iaa.ChangeColorspace.BGR
        assert isinstance(params[2], list)
        assert params[2] == [iaa.ChangeColorspace.HSV,
                             iaa.ChangeColorspace.Lab]
        assert params[3] is None
        assert params[4] == "cubic"


class Test_quantize_colors_kmeans(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _test_images_with_n_channels(cls, nb_channels):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])
        expected = np.uint8([
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ])

        if nb_channels is not None:
            image = np.tile(image[..., np.newaxis], (1, 1, nb_channels))
            expected = np.tile(expected[..., np.newaxis], (1, 1, nb_channels))

        observed = iaa.quantize_colors_kmeans(image, 2)

        assert np.array_equal(observed, expected)

    def test_images_with_no_channels(self):
        self._test_images_with_n_channels(None)

    def test_images_with_1_channel(self):
        self._test_images_with_n_channels(1)

    def test_images_with_3_channels(self):
        self._test_images_with_n_channels(3)

    def test_more_colors_than_pixels(self):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])
        expected = np.copy(image)

        observed = iaa.quantize_colors_kmeans(image, 100)

        assert np.array_equal(observed, expected)

    def test_failure_if_n_colors_less_than_2(self):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])

        got_exception = False
        try:
            _ = iaa.quantize_colors_kmeans(image, 1)
        except AssertionError as exc:
            assert "[2..256]" in str(exc)
            got_exception = True
        assert got_exception

    def test_quantization_is_deterministic(self):
        rs = np.random.RandomState(1)
        image = rs.randint(0, 255, (100, 100, 3)).astype(np.uint8)

        # simulate multiple calls, each one of them should produce the
        # same quantization
        images_quantized = []
        for _ in sm.xrange(20):
            images_quantized.append(iaa.quantize_colors_kmeans(image, 20))

        for image_quantized in images_quantized[1:]:
            assert np.array_equal(image_quantized, images_quantized[0])


class UniformColorQuantization(TestKMeansColorQuantization):
    def setUp(self):
        reseed()

    @property
    def augmenter(self):
        return iaa.UniformColorQuantization

    @property
    def quantization_func_name(self):
        return "imgaug.augmenters.color.quantize_colors_uniform"

    def test___init___defaults(self):
        aug = self.augmenter()
        assert isinstance(aug.n_colors, iap.DiscreteUniform)
        assert aug.n_colors.a.value == 2
        assert aug.n_colors.b.value == 16
        assert aug.from_colorspace == iaa.ChangeColorspace.RGB
        assert aug.to_colorspace is None
        assert aug.max_size is None
        assert aug.interpolation == "linear"

    def test___init___custom_parameters(self):
        aug = self.augmenter(
            n_colors=(5, 8),
            from_colorspace=iaa.ChangeColorspace.BGR,
            to_colorspace=[iaa.ChangeColorspace.HSV, iaa.ChangeColorspace.Lab],
            max_size=128,
            interpolation="cubic"
        )
        assert isinstance(aug.n_colors, iap.DiscreteUniform)
        assert aug.n_colors.a.value == 5
        assert aug.n_colors.b.value == 8
        assert aug.from_colorspace == iaa.ChangeColorspace.BGR
        assert isinstance(aug.to_colorspace, list)
        assert aug.to_colorspace == [iaa.ChangeColorspace.HSV,
                                     iaa.ChangeColorspace.Lab]
        assert aug.max_size == 128
        assert aug.interpolation == "cubic"

    def test_images_with_1_channel_integrationtest(self):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])
        expected = np.uint8([
            [64, 64, 192, 192],
            [64, 64, 192, 192],
        ])

        aug = self.augmenter(
            n_colors=2,
            from_colorspace="RGB",
            to_colorspace="RGB",
            max_size=None)

        observed = aug(image=image)

        assert np.array_equal(observed, expected)

    def test_images_with_3_channels_integrationtest(self):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])
        expected = np.uint8([
            [64, 64, 192, 192],
            [64, 64, 192, 192],
        ])

        image = np.tile(image[..., np.newaxis], (1, 1, 3))
        expected = np.tile(expected[..., np.newaxis], (1, 1, 3))

        aug = self.augmenter(
            n_colors=2,
            from_colorspace="RGB",
            to_colorspace="RGB",
            max_size=None)

        observed = aug(image=image)

        assert np.array_equal(observed, expected)

    def test_images_with_4_channels_integrationtest(self):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])
        expected = np.uint8([
            [64, 64, 192, 192],
            [64, 64, 192, 192],
        ])

        image = np.tile(image[..., np.newaxis], (1, 1, 4))
        expected = np.tile(expected[..., np.newaxis], (1, 1, 3))

        # alpha channel is expected to not be altered by quantization
        expected = np.concatenate([expected, image[:, :, 3:4]], axis=-1)

        aug = self.augmenter(
            n_colors=2,
            from_colorspace="RGB",
            to_colorspace="RGB",
            max_size=None)

        observed = aug(image=image)

        assert np.array_equal(observed, expected)

    def test_from_colorspace(self):
        # Actual to_colorspace doesn't matter here as it is overwritten
        # via return_value. Important is just to set it to a non-None value
        # so that a colorspace conversion actually happens.
        aug = self.augmenter(from_colorspace="BGR",
                             to_colorspace="Lab")
        mock_change_colorspace = mock.MagicMock()
        mock_change_colorspace.return_value = mock_change_colorspace
        mock_change_colorspace.augment_image.side_effect = lambda img: img
        mock_change_colorspace._draw_samples.return_value = (None, ["foo"])

        fname = "imgaug.augmenters.color.ChangeColorspace"
        with mock.patch(fname, mock_change_colorspace):
            _ = aug.augment_image(np.zeros((4, 4, 3), dtype=np.uint8))

        # call 0, kwargs, argument 'from_colorspace'
        assert (
            mock_change_colorspace.call_args_list[0][1]["from_colorspace"]
            == "BGR")
        # call 1, kwargs, argument 'from_colorspace' (inverse transform)
        assert (
            mock_change_colorspace.call_args_list[1][1]["from_colorspace"]
            == "foo")


class Test_quantize_colors_uniform(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _test_images_with_n_channels_2_colors(cls, nb_channels):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])
        expected = np.uint8([
            [64, 64, 192, 192],
            [64, 64, 192, 192],
        ])

        if nb_channels is not None:
            image = np.tile(image[..., np.newaxis], (1, 1, nb_channels))
            expected = np.tile(expected[..., np.newaxis], (1, 1, nb_channels))

        observed = iaa.quantize_colors_uniform(image, 2)

        assert np.array_equal(observed, expected)

    @classmethod
    def _test_images_with_n_channels_4_colors(cls, nb_channels):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
            [127, 128, 220, 220]
        ])

        q = 256/4
        c1 = np.floor(0/q) * q + q/2  # 32
        c2 = np.floor(64/q) * q + q/2  # 96
        c3 = np.floor(128/q) * q + q/2  # 160
        c4 = np.floor(192/q) * q + q/2  # 224

        assert int(c1) == 32
        assert int(c2) == 96
        assert int(c3) == 160
        assert int(c4) == 224

        expected = np.uint8([
            [c1, c1, c4, c4],
            [c1, c1, c4, c4],
            [c2, c3, c4, c4]
        ])

        if nb_channels is not None:
            image = np.tile(image[..., np.newaxis], (1, 1, nb_channels))
            expected = np.tile(expected[..., np.newaxis], (1, 1, nb_channels))

        observed = iaa.quantize_colors_uniform(image, 4)

        assert np.array_equal(observed, expected)

    def test_images_with_no_channels_2_colors(self):
        self._test_images_with_n_channels_2_colors(None)

    def test_images_with_1_channel_2_colors(self):
        self._test_images_with_n_channels_2_colors(1)

    def test_images_with_3_channels_2_colors(self):
        self._test_images_with_n_channels_2_colors(3)

    def test_images_with_no_channels_4_colors(self):
        self._test_images_with_n_channels_4_colors(None)

    def test_images_with_1_channel_4_colors(self):
        self._test_images_with_n_channels_4_colors(1)

    def test_images_with_3_channels_4_colors(self):
        self._test_images_with_n_channels_4_colors(3)

    def test_failure_if_n_colors_less_than_2(self):
        image = np.uint8([
            [0, 0, 255, 255],
            [0, 1, 255, 255],
        ])

        got_exception = False
        try:
            _ = iaa.quantize_colors_uniform(image, 1)
        except AssertionError as exc:
            assert "[2..256]" in str(exc)
            got_exception = True
        assert got_exception


if __name__ == "__main__":
    main()
