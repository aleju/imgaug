from __future__ import print_function, division, absolute_import

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
import copy as copylib

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm
import cv2

import imgaug as ia
import imgaug.random as iarandom
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import imgaug.augmenters.meta as meta
from imgaug.testutils import reseed
import imgaug.augmenters.color as colorlib


class Test_change_colorspace_(unittest.TestCase):
    def test_non_uint8_fails(self):
        image = np.arange(4*5*3).astype(np.uint8).reshape((4, 5, 3))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        image_float = image.astype(np.float32) / 255.0
        with self.assertRaises(ValueError) as cm:
            _ = iaa.change_colorspace_(image_float, iaa.CSPACE_BGR)
        assert "which is a forbidden dtype" in str(cm.exception)

    def test_unknown_to_colorspace_fails(self):
        image = np.arange(4*5*3).astype(np.uint8).reshape((4, 5, 3))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        from_cspace = iaa.CSPACE_RGB
        to_cspace = "foo"
        with self.assertRaises(AssertionError) as cm:
            _ = iaa.change_colorspace_(
                image, to_colorspace=to_cspace, from_colorspace=from_cspace)
        assert "Expected `to_colorspace` to be one of" in str(cm.exception)

    def test_unknown_from_colorspace_fails(self):
        image = np.arange(4*5*3).astype(np.uint8).reshape((4, 5, 3))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        from_cspace = "foo"
        to_cspace = iaa.CSPACE_RGB
        with self.assertRaises(AssertionError) as cm:
            _ = iaa.change_colorspace_(
                image, to_colorspace=to_cspace, from_colorspace=from_cspace)
        assert "Expected `from_colorspace` to be one of" in str(cm.exception)

    def test_change_to_same_colorspace_does_nothing(self):
        image = np.arange(4*5*3).astype(np.uint8).reshape((4, 5, 3))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        from_cspace = iaa.CSPACE_RGB
        to_cspace = iaa.CSPACE_RGB
        image_out = iaa.change_colorspace_(
            np.copy(image),
            to_colorspace=to_cspace, from_colorspace=from_cspace)
        assert np.array_equal(image_out, image)

    def test_function_works_inplace(self):
        image = np.arange(4*5*3).astype(np.uint8).reshape((4, 5, 3))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        image_orig = np.copy(image)
        from_cspace = iaa.CSPACE_RGB
        to_cspace = iaa.CSPACE_BGR
        image_out = iaa.change_colorspace_(
            image,
            to_colorspace=to_cspace, from_colorspace=from_cspace)
        assert image_out is image
        assert np.array_equal(image_out, image)
        assert not np.array_equal(image_out, image_orig)

    def test_image_is_view(self):
        image = np.arange(4*5*4).astype(np.uint8).reshape((4, 5, 4))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        image_copy = np.copy(image)
        image_view = image[..., 0:3]
        assert image_view.flags["OWNDATA"] is False

        from_cspace = iaa.CSPACE_RGB
        to_cspace = iaa.CSPACE_BGR
        image_out = iaa.change_colorspace_(
            image_view,
            to_colorspace=to_cspace, from_colorspace=from_cspace)

        expected = self._generate_expected_image(
            np.ascontiguousarray(image_copy[..., 0:3]),
            from_cspace, to_cspace)
        assert np.array_equal(image_out, expected)

    def test_image_is_noncontiguous(self):
        image = np.arange(4*5*3).astype(np.uint8).reshape((4, 5, 3))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        image_copy = np.copy(np.ascontiguousarray(np.fliplr(image)))
        image_noncontiguous = np.fliplr(image)
        assert image_noncontiguous.flags["C_CONTIGUOUS"] is False

        from_cspace = iaa.CSPACE_RGB
        to_cspace = iaa.CSPACE_BGR
        image_out = iaa.change_colorspace_(
            image_noncontiguous,
            to_colorspace=to_cspace, from_colorspace=from_cspace)

        expected = self._generate_expected_image(image_copy, from_cspace,
                                                 to_cspace)
        assert np.array_equal(image_out, expected)

    def test_cannot_transform_from_grayscale_to_another_cspace(self):
        image = np.arange(4*5*3).astype(np.uint8).reshape((4, 5, 3))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        from_cspace = iaa.CSPACE_GRAY
        to_cspace = iaa.CSPACE_RGB
        with self.assertRaises(AssertionError) as cm:
            _ = iaa.change_colorspace_(
                np.copy(image),
                from_colorspace=from_cspace, to_colorspace=to_cspace)
        assert (
            "Cannot convert from grayscale to another colorspace"
            in str(cm.exception))

    def test_image_without_channels_fails(self):
        image = np.arange(4*5).astype(np.uint8).reshape((4, 5))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        from_cspace = iaa.CSPACE_RGB
        to_cspace = iaa.CSPACE_BGR
        with self.assertRaises(AssertionError) as cm:
            _ = iaa.change_colorspace_(
                np.copy(image),
                from_colorspace=from_cspace, to_colorspace=to_cspace)
        assert (
            "Expected image shape to be three-dimensional"
            in str(cm.exception))

    def test_image_with_four_channels_fails(self):
        image = np.arange(4*5*4).astype(np.uint8).reshape((4, 5, 4))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        from_cspace = iaa.CSPACE_RGB
        to_cspace = iaa.CSPACE_BGR
        with self.assertRaises(AssertionError) as cm:
            _ = iaa.change_colorspace_(
                np.copy(image),
                from_colorspace=from_cspace, to_colorspace=to_cspace)
        assert (
            "Expected number of channels to be three"
            in str(cm.exception))

    def test_colorspace_combinations(self):
        image = np.arange(4*5*3).astype(np.uint8).reshape((4, 5, 3))
        image = np.copy(image)  # reshape sets flag OWNDATA=False
        from_cspaces = iaa.CSPACE_ALL
        to_cspaces = iaa.CSPACE_ALL
        gen = itertools.product(from_cspaces, to_cspaces)
        for from_cspace, to_cspace in gen:
            if from_cspace == iaa.CSPACE_GRAY:
                continue

            with self.subTest(from_colorspace=from_cspace,
                              to_colorspace=to_cspace):
                image_out = iaa.change_colorspace_(np.copy(image), to_cspace,
                                                   from_cspace)

                if from_cspace == to_cspace:
                    expected = np.copy(image)
                else:
                    expected = self._generate_expected_image(image, from_cspace,
                                                             to_cspace)

                if to_cspace == iaa.CSPACE_GRAY:
                    expected = np.tile(expected[..., np.newaxis], (1, 1, 3))

                assert np.array_equal(image_out, expected)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0, 3),
            (0, 1, 3),
            (1, 0, 3)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.change_colorspace_(
                    np.copy(image), from_colorspace="RGB", to_colorspace="BGR")

                assert image_aug.shape == image.shape

    @classmethod
    def _generate_expected_image(cls, image, from_colorspace, to_colorspace):
        cv_vars = colorlib._CSPACE_OPENCV_CONV_VARS
        if from_colorspace == iaa.CSPACE_RGB:
            from2rgb = None
        else:
            from2rgb = cv_vars[(from_colorspace, iaa.CSPACE_RGB)]

        if to_colorspace == iaa.CSPACE_RGB:
            rgb2to = None
        else:
            rgb2to = cv_vars[(iaa.CSPACE_RGB, to_colorspace)]

        image_rgb = image
        if from2rgb is not None:
            image_rgb = cv2.cvtColor(image, from2rgb)

        image_out = image_rgb
        if rgb2to is not None:
            image_out = cv2.cvtColor(image_rgb, rgb2to)

        return image_out


class Test_change_color_temperatures_(unittest.TestCase):
    def test_single_image(self):
        image = np.full((1, 1, 3), 255, dtype=np.uint8)

        multipliers = [
            (1000, [255, 56, 0]),
            (1100, [255, 71, 0]),
            (1200, [255, 83, 0]),
            (1300, [255, 93, 0]),
            (4300, [255, 215, 177]),
            (4400, [255, 217, 182]),
            (4500, [255, 219, 186]),
            (4600, [255, 221, 190]),
            (11100, [196, 214, 255]),
            (11200, [195, 214, 255]),
            (11300, [195, 214, 255]),
            (11400, [194, 213, 255]),
            (17200, [173, 200, 255]),
            (17300, [173, 200, 255]),
            (17400, [173, 200, 255]),
            (21900, [166, 195, 255]),
            (31300, [158, 190, 255]),
            (39700, [155, 188, 255]),
            (39800, [155, 188, 255]),
            (39900, [155, 188, 255]),
            (40000, [155, 188, 255])
        ]

        for kelvin, multiplier in multipliers:
            with self.subTest(kelvin=kelvin):
                image_temp = iaa.change_color_temperatures_(
                    [np.copy(image)],
                    kelvins=kelvin)[0]

                expected = np.uint8(multiplier).reshape((1, 1, 3))
                assert np.array_equal(image_temp, expected)

    def test_several_images_as_list(self):
        image = np.full((1, 1, 3), 255, dtype=np.uint8)
        
        images_temp = iaa.change_color_temperatures_(
            [np.copy(image), np.copy(image), np.copy(image)],
            [11100, 11200, 11300]
        )

        expected = np.array([
            [196, 214, 255],
            [195, 214, 255],
            [195, 214, 255]
        ], dtype=np.uint8).reshape((3, 1, 1, 3))
        assert isinstance(images_temp, list)
        assert np.array_equal(images_temp[0], expected[0])
        assert np.array_equal(images_temp[1], expected[1])
        assert np.array_equal(images_temp[2], expected[2])

    def test_several_images_as_array(self):
        image = np.full((1, 1, 3), 255, dtype=np.uint8)

        images_temp = iaa.change_color_temperatures_(
            np.uint8([np.copy(image), np.copy(image), np.copy(image)]),
            np.float32([11100, 11200, 11300])
        )

        expected = np.array([
            [196, 214, 255],
            [195, 214, 255],
            [195, 214, 255]
        ], dtype=np.uint8).reshape((3, 1, 1, 3))
        assert ia.is_np_array(images_temp)
        assert np.array_equal(images_temp, expected)

    def test_interpolation_of_kelvins(self):
        # at 1000: [255, 56, 0]
        # at 1100: [255, 71, 0]
        at1050 = [255, 56 + (71-56)/2, 0]

        image = np.full((1, 1, 3), 255, dtype=np.uint8)

        image_temp = iaa.change_color_temperatures_(
            [np.copy(image)],
            kelvins=1050)[0]

        expected = np.uint8(at1050).reshape((1, 1, 3))
        diff = np.abs(image_temp.astype(np.int32) - expected.astype(np.int32))
        assert np.all(diff <= 1)

    def test_from_colorspace(self):
        image_bgr = np.uint8([100, 255, 0]).reshape((1, 1, 3))

        image_temp = iaa.change_color_temperatures_(
            [np.copy(image_bgr)],
            kelvins=1000,
            from_colorspaces=iaa.CSPACE_BGR
        )[0]

        multiplier_rgb = np.float32(
            [255/255.0, 56/255.0, 0/255.0]
        ).reshape((1, 1, 3))
        expected = (
            image_bgr[:, :, ::-1].astype(np.float32)
            * multiplier_rgb
        ).astype(np.uint8)[:, :, ::-1]
        diff = np.abs(image_temp.astype(np.int32) - expected.astype(np.int32))
        assert np.all(diff <= 1)


class Test_change_color_temperature_(unittest.TestCase):
    @mock.patch("imgaug.augmenters.color.change_color_temperatures_")
    def test_calls_batch_function(self, mock_ccts):
        image = np.full((1, 1, 3), 255, dtype=np.uint8)
        mock_ccts.return_value = ["example"]

        image_temp = iaa.change_color_temperature(
            image, 1000, from_colorspace="foo")

        assert image_temp == "example"
        assert np.array_equal(mock_ccts.call_args_list[0][0][0],
                              image[np.newaxis, ...])
        assert mock_ccts.call_args_list[0][0][1] == [1000]
        assert mock_ccts.call_args_list[0][1]["from_colorspaces"] == ["foo"]

    def test_single_image(self):
        image = np.full((1, 1, 3), 255, dtype=np.uint8)

        image_temp = iaa.change_color_temperature(np.copy(image), 1000)

        expected = np.uint8([255, 56, 0]).reshape((1, 1, 3))
        assert np.array_equal(image_temp, expected)


class _BatchCapturingDummyAugmenter(iaa.Augmenter):
    def __init__(self):
        super(_BatchCapturingDummyAugmenter, self).__init__()
        self.last_batch = None

    def _augment_batch(self, batch, random_state, parents, hooks):
        self.last_batch = copylib.deepcopy(batch.deepcopy())
        return batch

    def get_parameters(self):
        return []


class TestWithBrightnessChannels(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def valid_colorspaces(self):
        return iaa.WithBrightnessChannels._VALID_COLORSPACES

    def test___init___defaults(self):
        aug = iaa.WithBrightnessChannels()
        assert isinstance(aug.children, iaa.Augmenter)
        assert len(aug.to_colorspace.a) == len(self.valid_colorspaces)
        for cspace in self.valid_colorspaces:
            assert cspace in aug.to_colorspace.a
        assert aug.from_colorspace == iaa.CSPACE_RGB

    def test___init___to_colorspace_is_all(self):
        aug = iaa.WithBrightnessChannels(to_colorspace=ia.ALL)
        assert isinstance(aug.children, iaa.Augmenter)
        assert len(aug.to_colorspace.a) == len(self.valid_colorspaces)
        for cspace in self.valid_colorspaces:
            assert cspace in aug.to_colorspace.a
        assert aug.from_colorspace == iaa.CSPACE_RGB

    def test___init___to_colorspace_is_cspace(self):
        aug = iaa.WithBrightnessChannels(to_colorspace=iaa.CSPACE_YUV)
        assert isinstance(aug.children, iaa.Augmenter)
        assert aug.to_colorspace.value == iaa.CSPACE_YUV
        assert aug.from_colorspace == iaa.CSPACE_RGB

    def test___init___to_colorspace_is_stochastic_parameter(self):
        aug = iaa.WithBrightnessChannels(
            to_colorspace=iap.Deterministic(iaa.CSPACE_YUV))
        assert isinstance(aug.children, iaa.Augmenter)
        assert aug.to_colorspace.value == iaa.CSPACE_YUV
        assert aug.from_colorspace == iaa.CSPACE_RGB

    def test_every_colorspace(self):
        def _image_to_channel(image, cspace):
            if cspace == iaa.CSPACE_YCrCb:
                image_cvt = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
                return image_cvt[:, :, 0:0+1]
            elif cspace == iaa.CSPACE_HSV:
                image_cvt = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                return image_cvt[:, :, 2:2+1]
            elif cspace == iaa.CSPACE_HLS:
                image_cvt = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                return image_cvt[:, :, 1:1+1]
            elif cspace == iaa.CSPACE_Lab:
                if hasattr(cv2, "COLOR_RGB2Lab"):
                    image_cvt = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
                else:
                    image_cvt = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                return image_cvt[:, :, 0:0+1]
            elif cspace == iaa.CSPACE_Luv:
                if hasattr(cv2, "COLOR_RGB2Luv"):
                    image_cvt = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
                else:
                    image_cvt = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                return image_cvt[:, :, 0:0+1]
            else:
                assert cspace == iaa.CSPACE_YUV
                image_cvt = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                return image_cvt[:, :, 0:0+1]

        # Max differences between input image and image after augmentation
        # when no child augmenter is used (for the given example image below).
        # For some colorspaces the conversion to input colorspace isn't
        # perfect.
        # Values were manually checked.
        max_diff_expected = {
            iaa.CSPACE_YCrCb: 1,
            iaa.CSPACE_HSV: 0,
            iaa.CSPACE_HLS: 0,
            iaa.CSPACE_Lab: 2,
            iaa.CSPACE_Luv: 4,
            iaa.CSPACE_YUV: 1
        }

        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))

        for cspace in self.valid_colorspaces:
            with self.subTest(colorspace=cspace):
                child = _BatchCapturingDummyAugmenter()
                aug = iaa.WithBrightnessChannels(
                    children=child,
                    to_colorspace=cspace)

                image_aug = aug(image=image)

                expected = _image_to_channel(image, cspace)
                diff = np.abs(
                    image.astype(np.int32) - image_aug.astype(np.int32))
                assert np.all(diff <= max_diff_expected[cspace])
                assert np.array_equal(child.last_batch.images[0], expected)

    def test_random_colorspace(self):
        def _images_to_cspaces(images, choices):
            result = np.full((len(images),), -1, dtype=np.int32)
            for i, image_aug in enumerate(images):
                for j, choice in enumerate(choices):
                    if np.array_equal(image_aug, choice):
                        result[i] = j
                        break
            assert np.all(result != -1)
            return result

        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))
        expected_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 2:2+1]
        expected_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 1:1+1]

        child = _BatchCapturingDummyAugmenter()
        aug = iaa.WithBrightnessChannels(children=child,
                                 to_colorspace=[iaa.CSPACE_HSV, iaa.CSPACE_HLS])

        images = [np.copy(image) for _ in sm.xrange(100)]

        _ = aug(images=images)
        images_aug1 = child.last_batch.images

        _ = aug(images=images)
        images_aug2 = child.last_batch.images

        cspaces1 = _images_to_cspaces(images_aug1, [expected_hsv, expected_hls])
        cspaces2 = _images_to_cspaces(images_aug2, [expected_hsv, expected_hls])

        assert np.any(cspaces1 != cspaces2)
        assert len(np.unique(cspaces1)) > 1
        assert len(np.unique(cspaces2)) > 1

    def test_from_colorspace_is_not_rgb(self):
        child = _BatchCapturingDummyAugmenter()
        aug = iaa.WithBrightnessChannels(children=child,
                                 to_colorspace=iaa.CSPACE_HSV,
                                 from_colorspace=iaa.CSPACE_BGR)

        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))
        expected_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2:2+1]

        _ = aug(image=image)
        observed = child.last_batch.images

        assert np.array_equal(observed[0], expected_hsv)

    def test_changes_from_child_propagate(self):
        aug = iaa.WithBrightnessChannels(children=iaa.Add(100),
                                 to_colorspace=iaa.CSPACE_HSV)
        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))

        image_aug = aug(image=image)

        assert not np.array_equal(image_aug, image)

    def test_using_hooks_to_deactivate_propagation(self):
        def _propagator(images, augmenter, parents, default):
            return False if augmenter.name == "foo" else default

        aug = iaa.WithBrightnessChannels(children=iaa.Add(100),
                                 to_colorspace=iaa.CSPACE_HSV,
                                 name="foo")
        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))

        image_aug = aug(image=image,
                        hooks=ia.HooksImages(propagator=_propagator))

        assert np.array_equal(image_aug, image)

    def test_batch_without_images(self):
        aug = iaa.WithBrightnessChannels(children=iaa.Affine(translate_px={"x": 1}))

        kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=0, y=2)],
                                    shape=(1, 1, 3))
        kpsoi_aug = aug(keypoints=kpsoi)

        assert np.isclose(kpsoi_aug.keypoints[0].x, 1.0)
        assert np.isclose(kpsoi_aug.keypoints[0].y, 2.0)

    def test_get_parameters(self):
        aug = iaa.WithBrightnessChannels(to_colorspace=iaa.CSPACE_HSV)

        params = aug.get_parameters()

        assert params[0].value == iaa.CSPACE_HSV
        assert params[1] == iaa.CSPACE_RGB

    def test___str__(self):
        child = iaa.Noop()
        aug = iaa.WithBrightnessChannels(
            child,
            from_colorspace=iaa.CSPACE_RGB,
            to_colorspace=iaa.CSPACE_HSV,
            name="foo")

        aug_str = aug.__str__()

        expected_child = iaa.Sequential([child], name="foo-then")
        expected = (
            "WithBrightnessChannels("
            "to_colorspace=Deterministic(HSV), "
            "from_colorspace=RGB, "
            "name=foo, "
            "children=%s, "
            "deterministic=False)" % (str(expected_child),))
        assert aug_str == expected


# MultiplyBrightness re-used MultiplyAndAddToBrightness, so we don't have
# to test much here.
class TestMultiplyBrightness(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def valid_colorspaces(self):
        return iaa.WithBrightnessChannels._VALID_COLORSPACES

    def test___init___defaults(self):
        aug = iaa.MultiplyBrightness()
        assert isinstance(aug.children, iaa.Augmenter)
        assert isinstance(aug.children[0], iaa.Multiply)
        assert len(aug.to_colorspace.a) == len(self.valid_colorspaces)
        for cspace in self.valid_colorspaces:
            assert cspace in aug.to_colorspace.a
        assert aug.from_colorspace == iaa.CSPACE_RGB

    def test_single_image(self):
        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))
        aug = iaa.MultiplyBrightness(2.0)

        image_aug = aug(image=image)

        assert np.average(image_aug) > np.average(image)


class TestMultiplyAndAddToBrightness(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___defaults(self):
        aug = iaa.MultiplyAndAddToBrightness()
        assert aug.children.random_order is True
        assert isinstance(aug.children[0], iaa.Multiply)
        assert isinstance(aug.children[1], iaa.Add)
        assert iaa.CSPACE_HSV in aug.to_colorspace.a
        assert aug.from_colorspace == iaa.CSPACE_RGB

    def test___init___add_is_zero(self):
        aug = iaa.MultiplyAndAddToBrightness(add=0)
        assert aug.children.random_order is True
        assert isinstance(aug.children[0], iaa.Multiply)
        assert isinstance(aug.children[1], iaa.Noop)
        assert iaa.CSPACE_HSV in aug.to_colorspace.a
        assert aug.from_colorspace == iaa.CSPACE_RGB

    def test___init___mul_is_1(self):
        aug = iaa.MultiplyAndAddToBrightness(mul=1.0)
        assert aug.children.random_order is True
        assert isinstance(aug.children[0], iaa.Noop)
        assert isinstance(aug.children[1], iaa.Add)
        assert iaa.CSPACE_HSV in aug.to_colorspace.a
        assert aug.from_colorspace == iaa.CSPACE_RGB

    def test_add_to_example_image(self):
        aug = iaa.MultiplyAndAddToBrightness(mul=1.0, add=10,
                                             to_colorspace=iaa.CSPACE_HSV,
                                             random_order=False)
        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))

        image_aug = aug(image=image)

        expected = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        expected[:, :, 2] += 10
        expected = cv2.cvtColor(expected.astype(np.uint8), cv2.COLOR_HSV2RGB)
        assert np.array_equal(image_aug, expected)

    def test_multiply_example_image(self):
        aug = iaa.MultiplyAndAddToBrightness(mul=1.2, add=0,
                                             to_colorspace=iaa.CSPACE_HSV,
                                             random_order=False)
        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))

        image_aug = aug(image=image)

        expected = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        expected[:, :, 2] *= 1.2
        expected = cv2.cvtColor(expected.astype(np.uint8), cv2.COLOR_HSV2RGB)
        assert np.array_equal(image_aug, expected)

    def test_multiply_and_add_example_image(self):
        aug = iaa.MultiplyAndAddToBrightness(mul=1.2, add=10,
                                             to_colorspace=iaa.CSPACE_HSV,
                                             random_order=False)
        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))

        image_aug = aug(image=image)

        expected = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        expected[:, :, 2] *= 1.2
        expected[:, :, 2] += 10
        expected = cv2.cvtColor(expected.astype(np.uint8), cv2.COLOR_HSV2RGB)
        assert np.array_equal(image_aug, expected)

    def test___str__(self):
        params = [
            (1.01, 1, iaa.Multiply(1.01), iaa.Add(1)),
            (1.00, 1, iaa.Noop(), iaa.Add(1)),
            (1.01, 0, iaa.Multiply(1.01), iaa.Noop()),
            (1.00, 0, iaa.Noop(), iaa.Noop()),
        ]

        for mul, add, exp_mul, exp_add in params:
            with self.subTest(mul=mul, add=add):
                aug = iaa.MultiplyAndAddToBrightness(
                    mul=mul,
                    add=add,
                    from_colorspace=iaa.CSPACE_RGB,
                    to_colorspace=iaa.CSPACE_HSV,
                    name="foo")

                aug_str = aug.__str__()

                expected = (
                    "MultiplyAndAddToBrightness("
                    "mul=%s, "
                    "add=%s, "
                    "to_colorspace=Deterministic(HSV), "
                    "from_colorspace=RGB, "
                    "random_order=True, "
                    "name=foo, "
                    "deterministic=False)" % (str(exp_mul), str(exp_add),))
                assert aug_str == expected


# TODO add tests for prop hooks
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
        class _DummyAugmenter(meta.Augmenter):
            def __init__(self):
                super(_DummyAugmenter, self).__init__()
                self.call_count = 0

            def _augment_batch(self, batch, random_state, parents, hooks):
                assert batch.images[0].dtype.name == "int16"
                self.call_count += 1
                return batch

            def get_parameters(self):
                return []

        aug_dummy = _DummyAugmenter()
        aug = iaa.WithHueAndSaturation(aug_dummy)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image_aug = aug.augment_images([image])[0]
        assert image_aug.dtype.name == "uint8"
        assert np.array_equal(image_aug, image)
        assert aug_dummy.call_count == 1

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

        class _DummyAugmenter(meta.Augmenter):
            def __init__(self):
                super(_DummyAugmenter, self).__init__()
                self.call_count = 0

            def _augment_batch(self, batch, random_state, parents, hooks):
                self.call_count += 1
                return batch

            def get_parameters(self):
                return []

        aug_dummy = _DummyAugmenter()
        hm = np.ones((8, 12, 1), dtype=np.float32)
        hmoi = HeatmapsOnImage(hm, shape=(16, 24, 3))

        aug = iaa.WithHueAndSaturation(aug_dummy)
        hmoi_aug = aug.augment_heatmaps(hmoi)
        assert hmoi_aug.shape == (16, 24, 3)
        assert hmoi_aug.arr_0to1.shape == (8, 12, 1)

        assert aug_dummy.call_count == 1

    def test_augment_keypoints(self):
        from imgaug.augmentables.kps import KeypointsOnImage

        class _DummyAugmenter(meta.Augmenter):
            def __init__(self):
                super(_DummyAugmenter, self).__init__()
                self.call_count = 0

            def _augment_batch(self, batch, random_state, parents, hooks):
                self.call_count += 1
                return batch

            def get_parameters(self):
                return []

        aug_dummy = _DummyAugmenter()
        kpsoi = KeypointsOnImage.from_xy_array(np.float32([
            [0, 0],
            [5, 1]
        ]), shape=(16, 24, 3))

        aug = iaa.WithHueAndSaturation(aug_dummy)
        kpsoi_aug = aug.augment_keypoints(kpsoi)
        assert kpsoi_aug.shape == (16, 24, 3)
        assert kpsoi.keypoints[0].x == 0
        assert kpsoi.keypoints[0].y == 0
        assert kpsoi.keypoints[1].x == 5
        assert kpsoi.keypoints[1].y == 1

        assert aug_dummy.call_count == 1

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
        rs = iarandom.RNG(1)
        images = rs.integers(0, 255, size=(32, 4, 4, 3), dtype=np.uint8)

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

    @classmethod
    def create_base_image(cls):
        base_img = np.zeros((4, 2, 3), dtype=np.uint8)

        base_img[0, :, 0] += 0
        base_img[0, :, 1] += 1
        base_img[0, :, 2] += 2

        base_img[1, :, 0] += 20
        base_img[1, :, 1] += 40
        base_img[1, :, 2] += 60

        base_img[2, :, 0] += 255
        base_img[2, :, 1] += 128
        base_img[2, :, 2] += 0

        base_img[3, :, 0] += 255
        base_img[3, :, 1] += 255
        base_img[3, :, 2] += 255

        return base_img

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
            img_hsv[..., 0] + int((value_hue/255.0) * (360/2)), 180)
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

    def test__generate_lut_table(self):
        def _hue(v):
            return np.mod(v, 180)

        def _sat(v):
            return np.clip(v, 0, 255)

        tables = iaa.AddToHueAndSaturation._generate_lut_table()
        table_hue, table_saturation = tables

        intensity_values = [0, 1, 128, 254, 255]  # = pixel values
        for iv in intensity_values:
            with self.subTest(intensity=iv):
                assert table_hue[0, iv] == _hue(iv-255)  # add value: -255
                assert table_hue[1, iv] == _hue(iv-254)  # add value: -254
                assert table_hue[254, iv] == _hue(iv-1)  # add value: -1
                assert table_hue[255, iv] == _hue(iv-0)  # add value: 0
                assert table_hue[256, iv] == _hue(iv+1)  # add value: 1
                assert table_hue[509, iv] == _hue(iv+254)  # add value: 254
                assert table_hue[510, iv] == _hue(iv+255)  # add value: 255

                assert table_saturation[0, iv] == _sat(iv-255)  # input: -255
                assert table_saturation[1, iv] == _sat(iv-254)  # input: -254
                assert table_saturation[254, iv] == _sat(iv-1)  # input: -1
                assert table_saturation[255, iv] == _sat(iv+0)  # input: 0
                assert table_saturation[256, iv] == _sat(iv+1)  # input: 1
                assert table_saturation[509, iv] == _sat(iv+254)  # input: 254
                assert table_saturation[510, iv] == _sat(iv+255)  # input: 255

    def test_augment_images_compare_backends(self):
        base_img = self.create_base_image()
        gen = itertools.product([False, True], [-255, -100, -1, 0, 1, 100, 255])
        for per_channel, value in gen:
            with self.subTest(value=value, per_channel=per_channel):
                aug_cv2 = iaa.AddToHueAndSaturation(value,
                                                    per_channel=per_channel)
                aug_cv2.backend = "cv2"

                aug_numpy = iaa.AddToHueAndSaturation(value,
                                                      per_channel=per_channel)
                aug_numpy.backend = "numpy"

                img_observed1 = aug_cv2(image=base_img)
                img_observed2 = aug_numpy(image=base_img)

                assert np.array_equal(img_observed1, img_observed2)

    def test_augment_images(self):
        base_img = self.create_base_image()

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
        base_img = self.create_base_image()

        class _DummyParam(iap.StochasticParameter):
            def _draw_samples(self, size, random_state):
                arr = np.float32([10, 20])
                return np.tile(arr[np.newaxis, :], (size[0], 1))

        aug = iaa.AddToHueAndSaturation(value=_DummyParam(), per_channel=False)
        img_expected = self._add_hue_saturation(base_img, value=10)
        img_observed = aug.augment_image(base_img)

        assert np.array_equal(img_observed, img_expected)

    def test_augment_images__different_hue_and_saturation__per_channel(self):
        base_img = self.create_base_image()

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
        base_img = self.create_base_image()

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
        base_img = self.create_base_image()

        aug = iaa.AddToHueAndSaturation([0, 10, 20])
        base_img = base_img[1:2, 0:1, :]
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
        base_img = self.create_base_image()

        for value_hue in [-255, -254, -128, -64, -10, -1,
                          0, 1, 10, 64, 128, 254, 255]:
            with self.subTest(value_hue=value_hue):
                aug = iaa.AddToHueAndSaturation(value_hue=value_hue)
                img_expected = self._add_hue_saturation(
                    base_img, value_hue=value_hue)

                img_observed = aug(image=base_img)

                assert np.array_equal(img_observed, img_expected)

    def test_augment_images__value_hue__multi_image_sampling(self):
        base_img = self.create_base_image()

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
        base_img = self.create_base_image()

        for value_saturation in [-255, -254, -128, -64, -10,
                                 0, 10, 64, 128, 254, 255]:
            with self.subTest(value_hue=value_saturation):
                aug = iaa.AddToHueAndSaturation(
                    value_saturation=value_saturation)
                img_expected = self._add_hue_saturation(
                    base_img, value_saturation=value_saturation)

                img_observed = aug(image=base_img)

                assert np.array_equal(img_observed, img_expected)

    def test_augment_images__value_saturation__multi_image_sampling(self):
        base_img = self.create_base_image()

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
        base_img = self.create_base_image()

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


class TestGrayscale(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def base_img(self):
        base_img = np.zeros((4, 4, 3), dtype=np.uint8)
        base_img[..., 0] += 10
        base_img[..., 1] += 20
        base_img[..., 2] += 30
        return base_img

    @classmethod
    def _compute_luminosity(cls, r, g, b):
        return 0.21 * r + 0.72 * g + 0.07 * b

    def test_alpha_is_0(self):
        aug = iaa.Grayscale(0.0)
        observed = aug.augment_image(self.base_img)
        expected = np.copy(self.base_img)
        assert np.allclose(observed, expected)

    def test_alpha_is_1(self):
        aug = iaa.Grayscale(1.0)
        observed = aug.augment_image(self.base_img)
        luminosity = self._compute_luminosity(10, 20, 30)
        expected = np.zeros_like(self.base_img) + luminosity
        assert np.allclose(observed, expected.astype(np.uint8))

    def test_alpha_is_050(self):
        aug = iaa.Grayscale(0.5)
        observed = aug.augment_image(self.base_img)
        luminosity = self._compute_luminosity(10, 20, 30)
        expected = 0.5 * self.base_img + 0.5 * luminosity
        assert np.allclose(observed, expected.astype(np.uint8))

    def test_alpha_is_tuple(self):
        aug = iaa.Grayscale((0.0, 1.0))
        base_img = np.uint8([255, 0, 0]).reshape((1, 1, 3))
        base_img_float = base_img.astype(np.float64) / 255.0
        base_img_gray = iaa.Grayscale(1.0)\
            .augment_image(base_img)\
            .astype(np.float64) / 255.0
        distance_max = np.linalg.norm(base_img_gray.flatten()
                                      - base_img_float.flatten())
        nb_iterations = 1000
        distances = []
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_image(base_img).astype(np.float64) / 255.0
            distance = np.linalg.norm(
                observed.flatten() - base_img_float.flatten()) / distance_max
            distances.append(distance)

        assert 0 - 1e-4 < min(distances) < 0.1
        assert 0.4 < np.average(distances) < 0.6
        assert 0.9 < max(distances) < 1.0 + 1e-4

        nb_bins = 5
        hist, _ = np.histogram(
            distances, bins=nb_bins, range=(0.0, 1.0), density=False)
        density_expected = 1.0/nb_bins
        density_tolerance = 0.05
        for nb_samples in hist:
            density = nb_samples / nb_iterations
            assert np.isclose(density, density_expected,
                              rtol=0, atol=density_tolerance)


class TestChangeColorTemperature(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___defaults(self):
        aug = iaa.ChangeColorTemperature()
        assert isinstance(aug.kelvin, iap.Uniform)
        assert aug.kelvin.a.value == 1000
        assert aug.kelvin.b.value == 11000
        assert aug.from_colorspace == iaa.CSPACE_RGB

    def test___init___kelvin_is_deterministic(self):
        aug = iaa.ChangeColorTemperature(1000)
        assert aug.kelvin.value == 1000

    def test___init___kelvin_is_tuple(self):
        aug = iaa.ChangeColorTemperature((2000, 3000))
        assert isinstance(aug.kelvin, iap.Uniform)
        assert aug.kelvin.a.value == 2000
        assert aug.kelvin.b.value == 3000

    def test___init___kelvin_is_list(self):
        aug = iaa.ChangeColorTemperature([1000, 2000, 3000])
        assert isinstance(aug.kelvin, iap.Choice)
        assert aug.kelvin.a == [1000, 2000, 3000]

    def test___init___kelvin_is_stochastic_param(self):
        param = iap.Deterministic(5000)
        aug = iaa.ChangeColorTemperature(param)
        assert aug.kelvin is param

    @mock.patch("imgaug.augmenters.color.change_color_temperatures_")
    def test_mocked(self, mock_ccts):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        aug = iaa.ChangeColorTemperature((1000, 40000),
                                         from_colorspace=iaa.CSPACE_HLS)

        def _side_effect(images, kelvins, from_colorspaces):
            return images

        mock_ccts.side_effect = _side_effect

        _image_aug = aug(images=[image, image])

        assert mock_ccts.call_count == 1
        assert np.array_equal(mock_ccts.call_args_list[0][0][0],
                              [image, image])
        assert not np.isclose(
            mock_ccts.call_args_list[0][0][1][0],  # kelvin img 1
            mock_ccts.call_args_list[0][0][1][1],  # kelvin img 2
        )
        assert (mock_ccts.call_args_list[0][1]["from_colorspaces"]
                == iaa.CSPACE_HLS)

    def test_single_image(self):
        image = np.full((1, 1, 3), 255, dtype=np.uint8)
        aug = iaa.ChangeColorTemperature(1000)

        image_aug = aug(image=image)

        expected = np.uint8([255, 56, 0]).reshape((1, 1, 3))
        assert np.array_equal(image_aug, expected)

    def test_get_parameters(self):
        aug = iaa.ChangeColorTemperature(1111,
                                         from_colorspace=iaa.CSPACE_HLS)
        params = aug.get_parameters()
        assert params[0].value == 1111
        assert params[1] == iaa.CSPACE_HLS


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
        def _noop(img):
            return img

        aug = self.augmenter(from_colorspace="BGR")
        mock_change_colorspace = mock.MagicMock()
        mock_change_colorspace.return_value = mock_change_colorspace
        mock_change_colorspace.augment_image.side_effect = _noop
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

        class _ImresizeSideEffect(object):
            def __init__(self):
                self.nth_call = 0

            def __call__(self, *_args, **_kwargs):
                if self.nth_call == 0:
                    self.nth_call += 1
                    return np.zeros((100, 50, 3), dtype=np.uint8)
                else:
                    return np.zeros((200, 100, 3), dtype=np.uint8)

        mock_imresize = mock.Mock()
        mock_imresize.side_effect = _ImresizeSideEffect()

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

        class _ImresizeSideEffect(object):
            def __init__(self):
                self.nth_call = 0

            def __call__(self, *_args, **_kwargs):
                if self.nth_call == 0:
                    self.nth_call += 1
                    return np.zeros((100, 50, 3), dtype=np.uint8)
                else:
                    return np.zeros((200, 100, 3), dtype=np.uint8)

        mock_imresize = mock.Mock()
        mock_imresize.side_effect = _ImresizeSideEffect()

        fname = "imgaug.imresize_single_image"
        with mock.patch(fname, mock_imresize):
            _ = aug.augment_image(image)

        assert mock_imresize.call_count == 2
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
        rs = iarandom.RNG(1)
        image = rs.integers(0, 255, (100, 100, 3)).astype(np.uint8)

        # simulate multiple calls, each one of them should produce the
        # same quantization
        images_quantized = []
        for _ in sm.xrange(20):
            images_quantized.append(iaa.quantize_colors_kmeans(image, 20))

        for image_quantized in images_quantized[1:]:
            assert np.array_equal(image_quantized, images_quantized[0])

    def test_zero_sized_axes(self):
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

                image_aug = iaa.quantize_colors_kmeans(image, 2)

                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.quantize_colors_kmeans(image, 2)

                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape


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
        def _noop(img):
            return img

        # Actual to_colorspace doesn't matter here as it is overwritten
        # via return_value. Important is just to set it to a non-None value
        # so that a colorspace conversion actually happens.
        aug = self.augmenter(from_colorspace="BGR",
                             to_colorspace="Lab")
        mock_change_colorspace = mock.MagicMock()
        mock_change_colorspace.return_value = mock_change_colorspace
        mock_change_colorspace.augment_image.side_effect = _noop
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

    def test_zero_sized_axes(self):
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

                image_aug = iaa.quantize_colors_uniform(image, 2)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.quantize_colors_uniform(image, 2)

                assert np.any(image_aug > 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape
