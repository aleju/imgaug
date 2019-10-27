from __future__ import print_function, division, absolute_import

import functools
import sys
import warnings
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

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.testutils import array_equal_lists, keypoints_equal, reseed
import imgaug.augmenters.arithmetic as arithmetic_lib
import imgaug.augmenters.contrast as contrast_lib


class TestAdd(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___bad_datatypes(self):
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

    def test_add_zero(self):
        # no add, shouldnt change anything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Add(value=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

    def test_add_one(self):
        # add > 0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Add(value=1)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images + 1
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [images_list[0] + 1]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images + 1
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [images_list[0] + 1]
        assert array_equal_lists(observed, expected)

    def test_minus_one(self):
        # add < 0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

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

    def test_uint8_every_possible_value(self):
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

    def test_add_floats(self):
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

    def test_stochastic_parameters_as_value(self):
        # test other parameters
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

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

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.Add(value=1)
        aug_det = iaa.Add(value=1).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test_tuple_as_value(self):
        # varying values
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

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

    def test_per_channel(self):
        # test channelwise
        aug = iaa.Add(value=iap.Choice([0, 1]), per_channel=True)
        observed = aug.augment_image(np.zeros((1, 1, 100), dtype=np.uint8))
        uq = np.unique(observed)
        assert observed.shape == (1, 1, 100)
        assert 0 in uq
        assert 1 in uq
        assert len(uq) == 2

    def test_per_channel_with_probability(self):
        # test channelwise with probability
        aug = iaa.Add(value=iap.Choice([0, 1]), per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(400):
            observed = aug.augment_image(np.zeros((1, 1, 20), dtype=np.uint8))
            assert observed.shape == (1, 1, 20)

            uq = np.unique(observed)
            per_channel = (len(uq) == 2)
            if per_channel:
                seen[0] += 1
            else:
                seen[1] += 1
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

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
                aug = iaa.Add(1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

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
                aug = iaa.Add(1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.Add(value=1, per_channel=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert params[0].value == 1
        assert params[1].value == 0

    def test_heatmaps(self):
        # test heatmaps (not affected by augmenter)
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        aug = iaa.Add(value=10)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_bool(self):
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

    def test_other_dtypes_uint_int(self):
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

    def test_other_dtypes_float(self):
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


class TestAddElementwise(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _aug = iaa.AddElementwise(value="test")
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _aug = iaa.AddElementwise(value=1, per_channel="test")
        except Exception:
            got_exception = True
        assert got_exception

    def test_add_zero(self):
        # no add, shouldnt change anything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.AddElementwise(value=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

    def test_add_one(self):
        # add > 0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.AddElementwise(value=1)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images + 1
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [images_list[0] + 1]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images + 1
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [images_list[0] + 1]
        assert array_equal_lists(observed, expected)

    def test_add_minus_one(self):
        # add < 0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

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

    def test_uint8_every_possible_value(self):
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

    def test_stochastic_parameters_as_value(self):
        # test other parameters
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

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

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.AddElementwise(value=1)
        aug_det = iaa.AddElementwise(value=1).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test_tuple_as_value(self):
        # varying values
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

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

    def test_samples_change_by_spatial_location(self):
        # values should change between pixels
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

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

    def test_per_channel(self):
        # test channelwise
        aug = iaa.AddElementwise(value=iap.Choice([0, 1]), per_channel=True)
        observed = aug.augment_image(np.zeros((100, 100, 3), dtype=np.uint8))
        sums = np.sum(observed, axis=2)
        values = np.unique(sums)
        assert all([(value in values) for value in [0, 1, 2, 3]])

    def test_per_channel_with_probability(self):
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
                aug = iaa.AddElementwise(1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

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
                aug = iaa.AddElementwise(1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.AddElementwise(value=1, per_channel=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert params[0].value == 1
        assert params[1].value == 0

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.AddElementwise(value=10)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_bool(self):
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

    def test_other_dtypes_uint_int(self):
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

    def test_other_dtypes_float(self):
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


class AdditiveGaussianNoise(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_loc_zero_scale_zero(self):
        # no noise, shouldnt change anything
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128
        images = np.array([base_img])

        aug = iaa.AdditiveGaussianNoise(loc=0, scale=0)

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

    def test_loc_zero_scale_nonzero(self):
        # zero-centered noise
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128
        images = np.array([base_img])
        images_list = [base_img]
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

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

    def test_std_dev_of_added_noise_matches_scale(self):
        # std correct?
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

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

    def test_nonzero_loc(self):
        # non-zero loc
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

        aug = iaa.AdditiveGaussianNoise(loc=0.25 * 255, scale=0.01 * 255)
        images = np.ones((1, 1, 1, 1), dtype=np.uint8) * 128
        nb_iterations = 1000
        values = []
        for i in sm.xrange(nb_iterations):
            images_aug = aug.augment_images(images)
            values.append(images_aug[0, 0, 0, 0] - 128)
        values = np.array(values)
        assert 54 < np.average(values) < 74 # loc=0.25 should be around 255*0.25=64 average

    def test_tuple_as_loc(self):
        # varying locs
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

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

    def test_stochastic_parameter_as_loc(self):
        # varying locs by stochastic param
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

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

    def test_tuple_as_scale(self):
        # varying stds
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

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

    def test_stochastic_parameter_as_scale(self):
        # varying stds by stochastic param
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

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

    def test___init___bad_datatypes(self):
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

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 128

        aug = iaa.AdditiveGaussianNoise(loc=0.5, scale=10)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


class TestDropout(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_zero(self):
        # no dropout, shouldnt change anything
        base_img = np.ones((512, 512, 1), dtype=np.uint8) * 255
        images = np.array([base_img])
        images_list = [base_img]

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

    def test_p_is_50_percent(self):
        # 50% dropout
        base_img = np.ones((512, 512, 1), dtype=np.uint8) * 255
        images = np.array([base_img])
        images_list = [base_img]
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

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

    def test_tuple_as_p(self):
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

    def test_stochastic_parameter_as_p(self):
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

    def test___init___bad_datatypes(self):
        # test exception for wrong parameter datatype
        got_exception = False
        try:
            _aug = iaa.Dropout(p="test")
        except Exception:
            got_exception = True
        assert got_exception

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.Dropout(p=1.0)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


class TestCoarseDropout(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_zero(self):
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 100
        aug = iaa.CoarseDropout(p=0, size_px=4, size_percent=None, per_channel=False, min_size=4)
        observed = aug.augment_image(base_img)
        expected = base_img
        assert np.array_equal(observed, expected)

    def test_p_is_one(self):
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 100
        aug = iaa.CoarseDropout(p=1.0, size_px=4, size_percent=None, per_channel=False, min_size=4)
        observed = aug.augment_image(base_img)
        expected = np.zeros_like(base_img)
        assert np.array_equal(observed, expected)

    def test_p_is_50_percent(self):
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 100
        aug = iaa.CoarseDropout(p=0.5, size_px=1, size_percent=None, per_channel=False, min_size=1)
        averages = []
        for _ in sm.xrange(50):
            observed = aug.augment_image(base_img)
            averages.append(np.average(observed))
        assert all([v in [0, 100] for v in averages])
        assert 50 - 20 < np.average(averages) < 50 + 20

    def test_size_percent(self):
        base_img = np.ones((16, 16, 1), dtype=np.uint8) * 100
        aug = iaa.CoarseDropout(p=0.5, size_px=None, size_percent=0.001, per_channel=False, min_size=1)
        averages = []
        for _ in sm.xrange(50):
            observed = aug.augment_image(base_img)
            averages.append(np.average(observed))
        assert all([v in [0, 100] for v in averages])
        assert 50 - 20 < np.average(averages) < 50 + 20

    def test_per_channel(self):
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

    def test_stochastic_parameter_as_p(self):
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

    def test___init___bad_datatypes(self):
        # test exception for bad parameters
        got_exception = False
        try:
            _ = iaa.CoarseDropout(p="test")
        except Exception:
            got_exception = True
        assert got_exception

    def test___init___size_px_and_size_percent_both_none(self):
        got_exception = False
        try:
            _ = iaa.CoarseDropout(p=0.5, size_px=None, size_percent=None)
        except Exception:
            got_exception = True
        assert got_exception

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.CoarseDropout(p=1.0, size_px=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


class TestMultiply(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_mul_is_one(self):
        # no multiply, shouldnt change anything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Multiply(mul=1.0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

    def test_mul_is_above_one(self):
        # multiply >1.0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Multiply(mul=1.2)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
        assert array_equal_lists(observed, expected)

    def test_mul_is_below_one(self):
        # multiply <1.0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.Multiply(mul=0.8)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
        assert array_equal_lists(observed, expected)

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.Multiply(mul=1.2)
        aug_det = iaa.Multiply(mul=1.2).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test_tuple_as_mul(self):
        # varying multiply factors
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

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

    def test_per_channel(self):
        aug = iaa.Multiply(mul=iap.Choice([0, 2]), per_channel=True)
        observed = aug.augment_image(np.ones((1, 1, 100), dtype=np.uint8))
        uq = np.unique(observed)
        assert observed.shape == (1, 1, 100)
        assert 0 in uq
        assert 2 in uq
        assert len(uq) == 2

    def test_per_channel_with_probability(self):
        # test channelwise with probability
        aug = iaa.Multiply(mul=iap.Choice([0, 2]), per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(400):
            observed = aug.augment_image(np.ones((1, 1, 20), dtype=np.uint8))
            assert observed.shape == (1, 1, 20)

            uq = np.unique(observed)
            per_channel = (len(uq) == 2)
            if per_channel:
                seen[0] += 1
            else:
                seen[1] += 1
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

    def test___init___bad_datatypes(self):
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
                image = np.ones(shape, dtype=np.uint8)
                aug = iaa.Multiply(1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 2)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.ones(shape, dtype=np.uint8)
                aug = iaa.Multiply(2)

                image_aug = aug(image=image)

                assert np.all(image_aug == 2)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.Multiply(mul=1, per_channel=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert params[0].value == 1
        assert params[1].value == 0

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.Multiply(mul=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_bool(self):
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

    def test_other_dtypes_uint_int(self):
        # uint, int
        for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            dtype = np.dtype(dtype)
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

            # non-uint8 currently don't increase the itemsize
            if dtype.name == "uint8":
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

            # non-uint8 currently don't increase the itemsize
            if dtype.name == "uint8":
                image = np.full((3, 3), max_value, dtype=dtype)
                aug = iaa.Multiply(-2)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.type == dtype
                assert np.all(image_aug == min_value)

            # non-uint8 currently don't increase the itemsize
            if dtype.name == "uint8":
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

    def test_other_dtypes_float(self):
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

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), max_value, dtype=dtype)
            # aug = iaa.Multiply(-10)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert _allclose(image_aug, min_value)

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

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), min_value, dtype=dtype)
            # aug = iaa.Multiply(-2.0)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert _allclose(image_aug, max_value)

            image = np.full((3, 3), min_value, dtype=dtype)
            aug = iaa.Multiply(0.0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 0.0)

            # using tolerances of -100 - 1e-2 and 100 + 1e-2 is not enough for float16, had to be increased to -/+ 1e-1
            # deactivated, because itemsize increase was deactivated
            """
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
            """


class TestMultiplyElementwise(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_mul_is_one(self):
        # no multiply, shouldnt change anything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.MultiplyElementwise(mul=1.0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

    def test_mul_is_above_one(self):
        # multiply >1.0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.MultiplyElementwise(mul=1.2)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 120
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 120]
        assert array_equal_lists(observed, expected)

    def test_mul_is_below_one(self):
        # multiply <1.0
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.MultiplyElementwise(mul=0.8)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = np.ones((1, 3, 3, 1), dtype=np.uint8) * 80
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [np.ones((3, 3, 1), dtype=np.uint8) * 80]
        assert array_equal_lists(observed, expected)

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.MultiplyElementwise(mul=1.2)
        aug_det = iaa.Multiply(mul=1.2).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test_tuple_as_mul(self):
        # varying multiply factors
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

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

    def test_samples_change_by_spatial_location(self):
        # values should change between pixels
        base_img = np.ones((3, 3, 1), dtype=np.uint8) * 100
        images = np.array([base_img])

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

    def test_per_channel(self):
        # test channelwise
        aug = iaa.MultiplyElementwise(mul=iap.Choice([0, 1]), per_channel=True)
        observed = aug.augment_image(np.ones((100, 100, 3), dtype=np.uint8))
        sums = np.sum(observed, axis=2)
        values = np.unique(sums)
        assert all([(value in values) for value in [0, 1, 2, 3]])
        assert observed.shape == (100, 100, 3)

    def test_per_channel_with_probability(self):
        # test channelwise with probability
        aug = iaa.MultiplyElementwise(mul=iap.Choice([0, 1]), per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(400):
            observed = aug.augment_image(np.ones((20, 20, 3), dtype=np.uint8))
            assert observed.shape == (20, 20, 3)

            sums = np.sum(observed, axis=2)
            values = np.unique(sums)
            all_values_found = all([(value in values) for value in [0, 1, 2, 3]])
            if all_values_found:
                seen[0] += 1
            else:
                seen[1] += 1
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _aug = iaa.MultiplyElementwise(mul="test")
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _aug = iaa.MultiplyElementwise(mul=1, per_channel="test")
        except Exception:
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
                image = np.ones(shape, dtype=np.uint8)
                aug = iaa.MultiplyElementwise(2)

                image_aug = aug(image=image)

                assert np.all(image_aug == 2)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.ones(shape, dtype=np.uint8)
                aug = iaa.MultiplyElementwise(2)

                image_aug = aug(image=image)

                assert np.all(image_aug == 2)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.MultiplyElementwise(mul=1, per_channel=False)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert params[0].value == 1
        assert params[1].value == 0

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.MultiplyElementwise(mul=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_bool(self):
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

    def test_other_dtypes_uint_int(self):
        # uint, int
        for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            dtype = np.dtype(dtype)
            min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

            image = np.full((3, 3), 10, dtype=dtype)
            aug = iaa.MultiplyElementwise(1)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 10)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), 10, dtype=dtype)
            # aug = iaa.MultiplyElementwise(10)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert np.all(image_aug == 100)

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

            # partially deactivated, because itemsize increase was deactivated
            if dtype.name == "uint8":
                if dtype.kind == "u":
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

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), int(center_value), dtype=dtype)
            # aug = iaa.MultiplyElementwise(1.2)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert np.all(image_aug == int(1.2 * int(center_value)))

            # deactivated, because itemsize increase was deactivated
            if dtype.name == "uint8":
                if dtype.kind == "u":
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

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), max_value, dtype=dtype)
            # aug = iaa.MultiplyElementwise(10)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert np.all(image_aug == max_value)

            image = np.full((3, 3), max_value, dtype=dtype)
            aug = iaa.MultiplyElementwise(0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert np.all(image_aug == 0)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), max_value, dtype=dtype)
            # aug = iaa.MultiplyElementwise(-2)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert np.all(image_aug == min_value)

            # partially deactivated, because itemsize increase was deactivated
            if dtype.name == "uint8":
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

    def test_other_dtypes_float(self):
        # float
        for dtype in [np.float16, np.float32]:
            dtype = np.dtype(dtype)
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

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), 10.0, dtype=dtype)
            # aug = iaa.MultiplyElementwise(2.0)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert _allclose(image_aug, 20.0)

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), max_value, dtype=dtype)
            # aug = iaa.MultiplyElementwise(-10)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert _allclose(image_aug, min_value)

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

            # deactivated, because itemsize increase was deactivated
            # image = np.full((3, 3), min_value, dtype=dtype)
            # aug = iaa.MultiplyElementwise(-2.0)
            # image_aug = aug.augment_image(image)
            # assert image_aug.dtype.type == dtype
            # assert _allclose(image_aug, max_value)

            image = np.full((3, 3), min_value, dtype=dtype)
            aug = iaa.MultiplyElementwise(0.0)
            image_aug = aug.augment_image(image)
            assert image_aug.dtype.type == dtype
            assert _allclose(image_aug, 0.0)

            # using tolerances of -100 - 1e-2 and 100 + 1e-2 is not enough for float16, had to be increased to -/+ 1e-1
            # deactivated, because itemsize increase was deactivated
            """
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
            """


class TestReplaceElementwise(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_mask_is_always_zero(self):
        # no replace, shouldnt change anything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) + 99
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.ReplaceElementwise(mask=0, replacement=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = images_list
        assert array_equal_lists(observed, expected)

    def test_mask_is_always_one(self):
        # replace at 100 percent prob., should change everything
        base_img = np.ones((3, 3, 1), dtype=np.uint8) + 99
        images = np.array([base_img])
        images_list = [base_img]

        aug = iaa.ReplaceElementwise(mask=1, replacement=0)
        aug_det = aug.to_deterministic()

        observed = aug.augment_images(images)
        expected = np.zeros((1, 3, 3, 1), dtype=np.uint8)
        assert np.array_equal(observed, expected)
        assert observed.shape == (1, 3, 3, 1)

        observed = aug.augment_images(images_list)
        expected = [np.zeros((3, 3, 1), dtype=np.uint8)]
        assert array_equal_lists(observed, expected)

        observed = aug_det.augment_images(images)
        expected = np.zeros((1, 3, 3, 1), dtype=np.uint8)
        assert np.array_equal(observed, expected)

        observed = aug_det.augment_images(images_list)
        expected = [np.zeros((3, 3, 1), dtype=np.uint8)]
        assert array_equal_lists(observed, expected)

    def test_mask_is_stochastic_parameter(self):
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

    def test_mask_is_list(self):
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

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        base_img = np.ones((3, 3, 1), dtype=np.uint8) + 99
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

        aug = iaa.ReplaceElementwise(mask=iap.Binomial(p=0.5), replacement=0)
        aug_det = iaa.ReplaceElementwise(mask=iap.Binomial(p=0.5), replacement=0).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test_replacement_is_stochastic_parameter(self):
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

    def test_per_channel_with_probability(self):
        # test channelwise with probability
        aug = iaa.ReplaceElementwise(mask=iap.Choice([0, 1]), replacement=1, per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(400):
            observed = aug.augment_image(np.zeros((20, 20, 3), dtype=np.uint8))
            assert observed.shape == (20, 20, 3)

            sums = np.sum(observed, axis=2)
            values = np.unique(sums)
            all_values_found = all([(value in values) for value in [0, 1, 2, 3]])
            if all_values_found:
                seen[0] += 1
            else:
                seen[1] += 1
        assert 150 < seen[0] < 250
        assert 150 < seen[1] < 250

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _aug = iaa.ReplaceElementwise(mask="test", replacement=1)
        except Exception:
            got_exception = True
        assert got_exception

        got_exception = False
        try:
            _aug = iaa.ReplaceElementwise(mask=1, replacement=1, per_channel="test")
        except Exception:
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
                aug = iaa.ReplaceElementwise(1.0, 1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

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
                aug = iaa.ReplaceElementwise(1.0, 1)

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
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

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.ReplaceElementwise(mask=1, replacement=0.5)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_bool(self):
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

    def test_other_dtypes_uint_int(self):
        # uint, int
        for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
            dtype = np.dtype(dtype)
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

            # deterministic stochastic parameters are by default int32 for
            # any integer value and hence cannot cover the full uint32 value
            # range
            if dtype.name != "uint32":
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

    def test_other_dtypes_float(self):
        # float
        for dtype in [np.float16, np.float32, np.float64]:
            dtype = np.dtype(dtype)
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

            # deterministic stochastic parameters are by default float32 for
            # any float value and hence cannot cover the full float64 value
            # range
            if dtype.name != "float64":
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


# not more tests necessary here as SaltAndPepper is just a tiny wrapper around
# ReplaceElementwise
class TestSaltAndPepper(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.SaltAndPepper(p=0.5)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6

    def test_p_is_one(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.SaltAndPepper(p=1.0)
        observed = aug.augment_image(base_img)
        nb_pepper = np.sum(observed < 40)
        nb_salt = np.sum(observed > 255 - 40)
        assert nb_pepper > 200
        assert nb_salt > 200


class TestCoarseSaltAndPepper(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.CoarseSaltAndPepper(p=0.5, size_px=100)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6

    def test_size_px(self):
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

    def test_p_is_list(self):
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

    def test_p_is_tuple(self):
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

    def test___init___bad_datatypes(self):
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

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.CoarseSaltAndPepper(p=1.0, size_px=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


# not more tests necessary here as Salt is just a tiny wrapper around
# ReplaceElementwise
class TestSalt(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.Salt(p=0.5)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6
        # Salt() occasionally replaces with 127, which probably should be the center-point here anyways
        assert np.all(observed >= 127)

    def test_p_is_one(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.Salt(p=1.0)
        observed = aug.augment_image(base_img)
        nb_pepper = np.sum(observed < 40)
        nb_salt = np.sum(observed > 255 - 40)
        assert nb_pepper == 0
        assert nb_salt > 200


class TestCoarseSalt(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.CoarseSalt(p=0.5, size_px=100)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6

    def test_size_px(self):
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

    def test_p_is_list(self):
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

    def test_p_is_tuple(self):
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

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _ = iaa.CoarseSalt(p="test", size_px=100)
        except Exception:
            got_exception = True
        assert got_exception

    def test_size_px_or_size_percent_not_none(self):
        got_exception = False
        try:
            _ = iaa.CoarseSalt(p=0.5, size_px=None, size_percent=None)
        except Exception:
            got_exception = True
        assert got_exception

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.CoarseSalt(p=1.0, size_px=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


# not more tests necessary here as Salt is just a tiny wrapper around
# ReplaceElementwise
class TestPepper(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_probability_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.Pepper(p=0.5)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6
        assert np.all(observed <= 128)

    def test_probability_is_one(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.Pepper(p=1.0)
        observed = aug.augment_image(base_img)
        nb_pepper = np.sum(observed < 40)
        nb_salt = np.sum(observed > 255 - 40)
        assert nb_pepper > 200
        assert nb_salt == 0


class TestCoarsePepper(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_fifty_percent(self):
        base_img = np.zeros((100, 100, 1), dtype=np.uint8) + 128
        aug = iaa.CoarsePepper(p=0.5, size_px=100)
        observed = aug.augment_image(base_img)
        p = np.mean(observed != 128)
        assert 0.4 < p < 0.6

    def test_size_px(self):
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

    def test_p_is_list(self):
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

    def test_p_is_tuple(self):
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

    def test___init___bad_datatypes(self):
        # test exceptions for wrong parameter types
        got_exception = False
        try:
            _ = iaa.CoarsePepper(p="test", size_px=100)
        except Exception:
            got_exception = True
        assert got_exception

    def test_size_px_or_size_percent_not_none(self):
        got_exception = False
        try:
            _ = iaa.CoarsePepper(p=0.5, size_px=None, size_percent=None)
        except Exception:
            got_exception = True
        assert got_exception

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.CoarsePepper(p=1.0, size_px=2)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)


class Test_invert(unittest.TestCase):
    @mock.patch("imgaug.augmenters.arithmetic.invert_")
    def test_mocked_defaults(self, mock_invert):
        mock_invert.return_value = "foo"
        arr = np.zeros((1,), dtype=np.uint8)
        observed = iaa.invert(arr)

        assert observed == "foo"
        args = mock_invert.call_args_list[0]
        assert np.array_equal(mock_invert.call_args_list[0][0][0], arr)
        assert args[1]["min_value"] is None
        assert args[1]["max_value"] is None
        assert args[1]["threshold"] is None
        assert args[1]["invert_above_threshold"] is True

    @mock.patch("imgaug.augmenters.arithmetic.invert_")
    def test_mocked(self, mock_invert):
        mock_invert.return_value = "foo"
        arr = np.zeros((1,), dtype=np.uint8)
        observed = iaa.invert(arr, min_value=1, max_value=10, threshold=5,
                              invert_above_threshold=False)

        assert observed == "foo"
        args = mock_invert.call_args_list[0]
        assert np.array_equal(mock_invert.call_args_list[0][0][0], arr)
        assert args[1]["min_value"] == 1
        assert args[1]["max_value"] == 10
        assert args[1]["threshold"] == 5
        assert args[1]["invert_above_threshold"] is False

    def test_uint8(self):
        values = np.array([0, 20, 45, 60, 128, 255], dtype=np.uint8)
        expected = np.array([
            255,
            255-20,
            255-45,
            255-60,
            255-128,
            255-255
        ], dtype=np.uint8)

        observed = iaa.invert(values)

        assert np.array_equal(observed, expected)
        assert observed is not values


# most parts of this function are tested via Invert
class Test_invert_(unittest.TestCase):
    def test_arr_is_noncontiguous_uint8(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)
        max_vr_flipped = np.fliplr(np.copy(zeros + 255))

        observed = iaa.invert_(max_vr_flipped)
        expected = zeros
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_arr_is_view_uint8(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)
        max_vr_view = np.copy(zeros + 255)[:, :, [0, 2]]

        observed = iaa.invert_(max_vr_view)
        expected = zeros[:, :, [0, 2]]
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_uint(self):
        dtypes = ["uint8", "uint16", "uint32", "uint64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    max_value - 0,
                    max_value - 20,
                    max_value - 45,
                    max_value - 60,
                    max_value - center_value,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values))

                assert np.array_equal(observed, expected)

    def test_uint_with_threshold_50_inv_above(self):
        threshold = 50
        dtypes = ["uint8", "uint16", "uint32", "uint64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    0,
                    20,
                    45,
                    max_value - 60,
                    max_value - center_value,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_uint_with_threshold_0_inv_above(self):
        threshold = 0
        dtypes = ["uint8", "uint16", "uint32", "uint64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    max_value - 0,
                    max_value - 20,
                    max_value - 45,
                    max_value - 60,
                    max_value - center_value,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_uint8_with_threshold_255_inv_above(self):
        threshold = 255
        dtypes = ["uint8"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    0,
                    20,
                    45,
                    60,
                    center_value,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_uint8_with_threshold_256_inv_above(self):
        threshold = 256
        dtypes = ["uint8"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    0,
                    20,
                    45,
                    60,
                    center_value,
                    max_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_uint_with_threshold_50_inv_below(self):
        threshold = 50
        dtypes = ["uint8", "uint16", "uint32", "uint64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    max_value - 0,
                    max_value - 20,
                    max_value - 45,
                    60,
                    center_value,
                    max_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=False)

                assert np.array_equal(observed, expected)

    def test_uint_with_threshold_50_inv_above_with_min_max(self):
        threshold = 50
        # uint64 does not support custom min/max, hence removed it here
        dtypes = ["uint8", "uint16", "uint32"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([0, 20, 45, 60, center_value, max_value],
                                  dtype=dt)
                expected = np.array([
                    0,  # not clipped to 10 as only >thresh affected
                    20,
                    45,
                    100 - 50,
                    100 - 90,
                    100 - 90
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       min_value=10,
                                       max_value=100,
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_int_with_threshold_50_inv_above(self):
        threshold = 50
        dtypes = ["int8", "int16", "int32", "int64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([-45, -20, center_value, 20, 45, max_value],
                                  dtype=dt)
                expected = np.array([
                    -45,
                    -20,
                    center_value,
                    20,
                    45,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.array_equal(observed, expected)

    def test_int_with_threshold_50_inv_below(self):
        threshold = 50
        dtypes = ["int8", "int16", "int32", "int64"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = int(center_value)

                values = np.array([-45, -20, center_value, 20, 45, max_value],
                                  dtype=dt)
                expected = np.array([
                    (-1) * (-45) - 1,
                    (-1) * (-20) - 1,
                    (-1) * center_value - 1,
                    (-1) * 20 - 1,
                    (-1) * 45 - 1,
                    max_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=False)

                assert np.array_equal(observed, expected)

    def test_float_with_threshold_50_inv_above(self):
        threshold = 50
        dtypes = ["float16", "float32", "float64", "float128"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = center_value

                values = np.array([-45.5, -20.5, center_value, 20.5, 45.5,
                                   max_value],
                                  dtype=dt)
                expected = np.array([
                    -45.5,
                    -20.5,
                    center_value,
                    20.5,
                    45.5,
                    min_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=True)

                assert np.allclose(observed, expected, rtol=0, atol=1e-4)

    def test_float_with_threshold_50_inv_below(self):
        threshold = 50
        dtypes = ["float16", "float32", "float64", "float128"]
        for dt in dtypes:
            with self.subTest(dtype=dt):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dt)
                center_value = center_value

                values = np.array([-45.5, -20.5, center_value, 20.5, 45.5,
                                   max_value],
                                  dtype=dt)
                expected = np.array([
                    (-1) * (-45.5),
                    (-1) * (-20.5),
                    (-1) * center_value,
                    (-1) * 20.5,
                    (-1) * 45.5,
                    max_value
                ], dtype=dt)

                observed = iaa.invert_(np.copy(values),
                                       threshold=threshold,
                                       invert_above_threshold=False)

                assert np.allclose(observed, expected, rtol=0, atol=1e-4)


class Test_solarize(unittest.TestCase):
    @mock.patch("imgaug.augmenters.arithmetic.solarize_")
    def test_mocked_defaults(self, mock_sol):
        arr = np.zeros((1,), dtype=np.uint8)
        mock_sol.return_value = "foo"

        observed = iaa.solarize(arr)

        args = mock_sol.call_args_list[0][0]
        kwargs = mock_sol.call_args_list[0][1]
        assert args[0] is not arr
        assert np.array_equal(args[0], arr)
        assert kwargs["threshold"] == 128
        assert observed == "foo"

    @mock.patch("imgaug.augmenters.arithmetic.solarize_")
    def test_mocked(self, mock_sol):
        arr = np.zeros((1,), dtype=np.uint8)
        mock_sol.return_value = "foo"

        observed = iaa.solarize(arr, threshold=5)

        args = mock_sol.call_args_list[0][0]
        kwargs = mock_sol.call_args_list[0][1]
        assert args[0] is not arr
        assert np.array_equal(args[0], arr)
        assert kwargs["threshold"] == 5
        assert observed == "foo"

    def test_uint8(self):
        arr = np.array([0, 10, 50, 150, 200, 255], dtype=np.uint8)
        arr = arr.reshape((2, 3, 1))

        observed = iaa.solarize(arr)

        expected = np.array([0, 10, 50, 255-150, 255-200, 255-255],
                            dtype=np.uint8).reshape((2, 3, 1))
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)


class Test_solarize_(unittest.TestCase):
    @mock.patch("imgaug.augmenters.arithmetic.invert_")
    def test_mocked_defaults(self, mock_sol):
        arr = np.zeros((1,), dtype=np.uint8)
        mock_sol.return_value = "foo"

        observed = iaa.solarize_(arr)

        args = mock_sol.call_args_list[0][0]
        kwargs = mock_sol.call_args_list[0][1]
        assert args[0] is arr
        assert kwargs["threshold"] == 128
        assert observed == "foo"

    @mock.patch("imgaug.augmenters.arithmetic.invert_")
    def test_mocked(self, mock_sol):
        arr = np.zeros((1,), dtype=np.uint8)
        mock_sol.return_value = "foo"

        observed = iaa.solarize_(arr, threshold=5)

        args = mock_sol.call_args_list[0][0]
        kwargs = mock_sol.call_args_list[0][1]
        assert args[0] is arr
        assert kwargs["threshold"] == 5
        assert observed == "foo"


class TestInvert(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_p_is_one(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        observed = iaa.Invert(p=1.0).augment_image(zeros + 255)
        expected = zeros
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_p_is_zero(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        observed = iaa.Invert(p=0.0).augment_image(zeros + 255)
        expected = zeros + 255
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_max_value_set(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        observed = iaa.Invert(p=1.0, max_value=200).augment_image(zeros + 200)
        expected = zeros
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_min_value_and_max_value_set(self):
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        observed = iaa.Invert(p=1.0, max_value=200, min_value=100).augment_image(zeros + 200)
        expected = zeros + 100
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

        observed = iaa.Invert(p=1.0, max_value=200, min_value=100).augment_image(zeros + 100)
        expected = zeros + 200
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_min_value_and_max_value_set_with_float_image(self):
        # with min/max and float inputs
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        zeros_f32 = zeros.astype(np.float32)
        observed = iaa.Invert(p=1.0, max_value=200, min_value=100).augment_image(zeros_f32 + 200)
        expected = zeros_f32 + 100
        assert observed.dtype.name == "float32"
        assert np.array_equal(observed, expected)

        observed = iaa.Invert(p=1.0, max_value=200, min_value=100).augment_image(zeros_f32 + 100)
        expected = zeros_f32 + 200
        assert observed.dtype.name == "float32"
        assert np.array_equal(observed, expected)

    def test_p_is_80_percent(self):
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

    def test_per_channel(self):
        aug = iaa.Invert(p=0.5, per_channel=True)
        img = np.zeros((1, 1, 100), dtype=np.uint8) + 255
        observed = aug.augment_image(img)
        assert len(np.unique(observed)) == 2

    # TODO split into two tests
    def test_p_is_stochastic_parameter_per_channel_is_probability(self):
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

    def test_threshold(self):
        arr = np.array([0, 10, 50, 150, 200, 255], dtype=np.uint8)
        arr = arr.reshape((2, 3, 1))
        aug = iaa.Invert(p=1.0, threshold=128, invert_above_threshold=True)

        observed = aug.augment_image(arr)

        expected = np.array([0, 10, 50, 255-150, 255-200, 255-255],
                            dtype=np.uint8).reshape((2, 3, 1))
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_threshold_inv_below(self):
        arr = np.array([0, 10, 50, 150, 200, 255], dtype=np.uint8)
        arr = arr.reshape((2, 3, 1))
        aug = iaa.Invert(p=1.0, threshold=128, invert_above_threshold=False)

        observed = aug.augment_image(arr)

        expected = np.array([255-0, 255-10, 255-50, 150, 200, 255],
                            dtype=np.uint8).reshape((2, 3, 1))
        assert observed.dtype.name == "uint8"
        assert np.array_equal(observed, expected)

    def test_keypoints_dont_change(self):
        # keypoints shouldnt be changed
        zeros = np.zeros((4, 4, 3), dtype=np.uint8)

        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=2, y=2)], shape=zeros.shape)]

        aug = iaa.Invert(p=1.0)
        aug_det = iaa.Invert(p=1.0).to_deterministic()
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

        observed = aug_det.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)

    def test___init___bad_datatypes(self):
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
                aug = iaa.Invert(1.0)

                image_aug = aug(image=image)

                assert np.all(image_aug == 255)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

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
                aug = iaa.Invert(1.0)

                image_aug = aug(image=image)

                assert np.all(image_aug == 255)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        # test get_parameters()
        aug = iaa.Invert(p=0.5, per_channel=False, min_value=10, max_value=20)
        params = aug.get_parameters()
        assert params[0] is aug.p
        assert params[1] is aug.per_channel
        assert params[2] == 10
        assert params[3] == 20
        assert params[4] is aug.threshold
        assert params[5] is aug.invert_above_threshold

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.Invert(p=1.0)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_other_dtypes_p_is_zero(self):
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

    def test_other_dtypes_p_is_one(self):
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

    def test_other_dtypes_p_is_one_with_min_value(self):
        # with p=1.0 and min_value
        aug = iaa.Invert(p=1.0, min_value=1)
        dtypes = [np.uint8, np.uint16, np.uint32,
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

    def test_other_dtypes_p_is_one_with_max_value(self):
        # with p=1.0 and max_value
        aug = iaa.Invert(p=1.0, max_value=16)
        dtypes = [np.uint8, np.uint16, np.uint32,
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


class TestContrastNormalization(unittest.TestCase):
    @unittest.skipIf(sys.version_info[0] <= 2,
                     "Warning is not generated in 2.7 on travis, but locally "
                     "in 2.7 it is?!")
    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            aug = arithmetic_lib.ContrastNormalization((0.9, 1.1))
            assert isinstance(aug, contrast_lib._ContrastFuncWrapper)

        assert len(caught_warnings) == 1
        assert (
            "deprecated"
            in str(caught_warnings[-1].message)
        )


# TODO use this in test_contrast.py or remove it?
"""
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
"""


class TestJpegCompression(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_compression_is_zero(self):
        # basic test at 0 compression
        img = ia.quokka(extract="square", size=(64, 64))
        aug = iaa.JpegCompression(0)
        img_aug = aug.augment_image(img)
        diff = np.average(np.abs(img.astype(np.float32) - img_aug.astype(np.float32)))
        assert diff < 1.0

    def test_compression_is_90(self):
        # basic test at 90 compression
        img = ia.quokka(extract="square", size=(64, 64))
        aug = iaa.JpegCompression(90)
        img_aug = aug.augment_image(img)
        diff = np.average(np.abs(img.astype(np.float32) - img_aug.astype(np.float32)))
        assert 1.0 < diff < 50.0

    def test___init__(self):
        aug = iaa.JpegCompression([0, 100])
        assert isinstance(aug.compression, iap.Choice)
        assert len(aug.compression.a) == 2
        assert aug.compression.a[0] == 0
        assert aug.compression.a[1] == 100

    def test_get_parameters(self):
        aug = iaa.JpegCompression([0, 100])
        assert len(aug.get_parameters()) == 1
        assert aug.get_parameters()[0] == aug.compression

    def test_compression_is_stochastic_parameter(self):
        # test if stochastic parameters are used by augmentation
        img = ia.quokka(extract="square", size=(64, 64))

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

    def test_keypoints_dont_change(self):
        # test keypoints (not affected by augmenter)
        aug = iaa.JpegCompression(50)
        kps = ia.quokka_keypoints()
        kps_aug = aug.augment_keypoints([kps])[0]
        for kp, kp_aug in zip(kps.keypoints, kps_aug.keypoints):
            assert np.allclose([kp.x, kp.y], [kp_aug.x, kp_aug.y])

    def test_heatmaps_dont_change(self):
        # test heatmaps (not affected by augmenter)
        aug = iaa.JpegCompression(50)
        hm = ia.quokka_heatmap()
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert np.allclose(hm.arr_0to1, hm_aug.arr_0to1)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0, 3),
            (0, 1, 3),
            (1, 0, 3)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.JpegCompression(100)

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == image.shape
