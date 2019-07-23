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

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.testutils import reseed


class TestRandomColorsBinaryImageColorizer(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___default_settings(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer()
        assert isinstance(colorizer.color_true, iap.DiscreteUniform)
        assert isinstance(colorizer.color_false, iap.DiscreteUniform)
        assert colorizer.color_true.a.value == 0
        assert colorizer.color_true.b.value == 255
        assert colorizer.color_false.a.value == 0
        assert colorizer.color_false.b.value == 255

    def test___init___deterministic_settinga(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer(color_true=1, color_false=2)
        assert isinstance(colorizer.color_true, iap.Deterministic)
        assert isinstance(colorizer.color_false, iap.Deterministic)
        assert colorizer.color_true.value == 1
        assert colorizer.color_false.value == 2

    def test___init___tuple_and_list(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=(0, 100), color_false=[200, 201, 202])
        assert isinstance(colorizer.color_true, iap.DiscreteUniform)
        assert isinstance(colorizer.color_false, iap.Choice)
        assert colorizer.color_true.a.value == 0
        assert colorizer.color_true.b.value == 100
        assert colorizer.color_false.a[0] == 200
        assert colorizer.color_false.a[1] == 201
        assert colorizer.color_false.a[2] == 202

    def test___init___stochastic_parameters(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=iap.DiscreteUniform(0, 100),
            color_false=iap.Choice([200, 201, 202]))
        assert isinstance(colorizer.color_true, iap.DiscreteUniform)
        assert isinstance(colorizer.color_false, iap.Choice)
        assert colorizer.color_true.a.value == 0
        assert colorizer.color_true.b.value == 100
        assert colorizer.color_false.a[0] == 200
        assert colorizer.color_false.a[1] == 201
        assert colorizer.color_false.a[2] == 202

    def test__draw_samples(self):
        class _ListSampler(iap.StochasticParameter):
            def __init__(self, offset):
                super(_ListSampler, self).__init__()
                self.offset = offset
                self.last_random_state = None

            def _draw_samples(self, size, random_state=None):
                assert size == (3,)
                self.last_random_state = random_state
                return np.uint8([0, 1, 2]) + self.offset

        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=_ListSampler(0),
            color_false=_ListSampler(1))
        random_state = np.random.RandomState(42)
        color_true, color_false = colorizer._draw_samples(random_state)
        assert np.array_equal(color_true, [0, 1, 2])
        assert np.array_equal(color_false, [1, 2, 3])
        assert colorizer.color_true.last_random_state == random_state
        assert colorizer.color_false.last_random_state == random_state

    def test_colorize__one_channel(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=100,
            color_false=10)
        random_state = np.random.RandomState(42)

        # input image has shape (H,W,1)
        image = np.zeros((5, 5, 1), dtype=np.uint8)
        image[:, 0:3, :] = 255
        image_binary = np.zeros((5, 5), dtype=bool)
        image_binary[:, 0:3] = True

        image_color = colorizer.colorize(
            image_binary, image, nth_image=0, random_state=random_state)

        assert image_color.ndim == 3
        assert image_color.shape[-1] == 1
        assert np.all(image_color[image_binary] == 100)
        assert np.all(image_color[~image_binary] == 10)

    def test_colorize__three_channels(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=100,
            color_false=10)
        random_state = np.random.RandomState(42)

        # input image has shape (H,W,3)
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[:, 0:3, :] = 255
        image_binary = np.zeros((5, 5), dtype=bool)
        image_binary[:, 0:3] = True

        image_color = colorizer.colorize(
            image_binary, image, nth_image=0, random_state=random_state)

        assert image_color.ndim == 3
        assert image_color.shape[-1] == 3
        assert np.all(image_color[image_binary] == 100)
        assert np.all(image_color[~image_binary] == 10)

    def test_colorize__four_channels(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=100,
            color_false=10)
        random_state = np.random.RandomState(42)

        # input image has shape (H,W,4)
        image = np.zeros((5, 5, 4), dtype=np.uint8)
        image[:, 0:3, 0:3] = 255
        image[:, 1:4, 3] = 123  # set some content for alpha channel

        image_binary = np.zeros((5, 5), dtype=bool)
        image_binary[:, 0:3] = True

        image_color = colorizer.colorize(
            image_binary, image, nth_image=0, random_state=random_state)

        assert image_color.ndim == 3
        assert image_color.shape[-1] == 4
        assert np.all(image_color[image_binary, 0:3] == 100)
        assert np.all(image_color[~image_binary, 0:3] == 10)
        # alpha channel must have been kept untouched
        assert np.all(image_color[:, :, 3:4] == image[:, :, 3:4])


class TestCanny(unittest.TestCase):
    def test___init___default_settings(self):
        aug = iaa.Canny()
        assert isinstance(aug.alpha, iap.Uniform)
        assert isinstance(aug.hysteresis_thresholds, tuple)
        assert isinstance(aug.sobel_kernel_size, iap.DiscreteUniform)
        assert isinstance(aug.colorizer, iaa.RandomColorsBinaryImageColorizer)
        assert np.isclose(aug.alpha.a.value, 0.0)
        assert np.isclose(aug.alpha.b.value, 1.0)
        assert len(aug.hysteresis_thresholds) == 2
        assert isinstance(aug.hysteresis_thresholds[0], iap.DiscreteUniform)
        assert np.isclose(aug.hysteresis_thresholds[0].a.value, 100-40)
        assert np.isclose(aug.hysteresis_thresholds[0].b.value, 100+40)
        assert isinstance(aug.hysteresis_thresholds[1], iap.DiscreteUniform)
        assert np.isclose(aug.hysteresis_thresholds[1].a.value, 200-40)
        assert np.isclose(aug.hysteresis_thresholds[1].b.value, 200+40)
        assert aug.sobel_kernel_size.a.value == 3
        assert aug.sobel_kernel_size.b.value == 7
        assert isinstance(aug.colorizer.color_true, iap.DiscreteUniform)
        assert isinstance(aug.colorizer.color_false, iap.DiscreteUniform)
        assert aug.colorizer.color_true.a.value == 0
        assert aug.colorizer.color_true.b.value == 255
        assert aug.colorizer.color_false.a.value == 0
        assert aug.colorizer.color_false.b.value == 255

    def test___init___custom_settings(self):
        aug = iaa.Canny(
            alpha=0.2,
            hysteresis_thresholds=([0, 1, 2], iap.DiscreteUniform(1, 10)),
            sobel_kernel_size=[3, 5],
            colorizer=iaa.RandomColorsBinaryImageColorizer(
                color_true=10, color_false=20)
        )
        assert isinstance(aug.alpha, iap.Deterministic)
        assert isinstance(aug.hysteresis_thresholds, tuple)
        assert isinstance(aug.sobel_kernel_size, iap.Choice)
        assert isinstance(aug.colorizer, iaa.RandomColorsBinaryImageColorizer)
        assert np.isclose(aug.alpha.value, 0.2)
        assert len(aug.hysteresis_thresholds) == 2
        assert isinstance(aug.hysteresis_thresholds[0], iap.Choice)
        assert aug.hysteresis_thresholds[0].a == [0, 1, 2]
        assert isinstance(aug.hysteresis_thresholds[1], iap.DiscreteUniform)
        assert np.isclose(aug.hysteresis_thresholds[1].a.value, 1)
        assert np.isclose(aug.hysteresis_thresholds[1].b.value, 10)
        assert isinstance(aug.sobel_kernel_size, iap.Choice)
        assert aug.sobel_kernel_size.a == [3, 5]
        assert isinstance(aug.colorizer.color_true, iap.Deterministic)
        assert isinstance(aug.colorizer.color_false, iap.Deterministic)
        assert aug.colorizer.color_true.value == 10
        assert aug.colorizer.color_false.value == 20

    def test___init___single_value_hysteresis(self):
        aug = iaa.Canny(
            alpha=0.2,
            hysteresis_thresholds=[0, 1, 2],
            sobel_kernel_size=[3, 5],
            colorizer=iaa.RandomColorsBinaryImageColorizer(
                color_true=10, color_false=20)
        )
        assert isinstance(aug.alpha, iap.Deterministic)
        assert isinstance(aug.hysteresis_thresholds, iap.Choice)
        assert isinstance(aug.sobel_kernel_size, iap.Choice)
        assert isinstance(aug.colorizer, iaa.RandomColorsBinaryImageColorizer)
        assert np.isclose(aug.alpha.value, 0.2)
        assert aug.hysteresis_thresholds.a == [0, 1, 2]
        assert isinstance(aug.sobel_kernel_size, iap.Choice)
        assert aug.sobel_kernel_size.a == [3, 5]
        assert isinstance(aug.colorizer.color_true, iap.Deterministic)
        assert isinstance(aug.colorizer.color_false, iap.Deterministic)
        assert aug.colorizer.color_true.value == 10
        assert aug.colorizer.color_false.value == 20

    def test__draw_samples__single_value_hysteresis(self):
        seed = 1
        nb_images = 1000

        aug = iaa.Canny(
            alpha=0.2,
            hysteresis_thresholds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            sobel_kernel_size=[3, 5, 7],
            random_state=np.random.RandomState(seed))

        example_image = np.zeros((5, 5, 3), dtype=np.uint8)
        samples = aug._draw_samples([example_image] * nb_images,
                                    random_state=np.random.RandomState(seed))
        alpha_samples = samples[0]
        hthresh_samples = samples[1]
        sobel_samples = samples[2]

        rss = ia.derive_random_states(np.random.RandomState(seed), 4)
        alpha_expected = [0.2] * nb_images
        hthresh_expected = rss[1].choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                         size=(nb_images, 2))
        sobel_expected = rss[3].choice([3, 5, 7],
                                       size=(nb_images,))

        invalid = hthresh_expected[:, 0] > hthresh_expected[:, 1]
        assert np.any(invalid)
        hthresh_expected[invalid, :] = hthresh_expected[invalid, :][:, [1, 0]]
        assert hthresh_expected.shape == (nb_images, 2)
        assert not np.any(hthresh_expected[:, 0] > hthresh_expected[:, 1])

        assert np.allclose(alpha_samples, alpha_expected)
        assert np.allclose(hthresh_samples, hthresh_expected)
        assert np.allclose(sobel_samples, sobel_expected)

    def test__draw_samples__tuple_as_hysteresis(self):
        seed = 1
        nb_images = 10

        aug = iaa.Canny(
            alpha=0.2,
            hysteresis_thresholds=([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                   iap.DiscreteUniform(5, 100)),
            sobel_kernel_size=[3, 5, 7],
            random_state=np.random.RandomState(seed))

        example_image = np.zeros((5, 5, 3), dtype=np.uint8)
        samples = aug._draw_samples([example_image] * nb_images,
                                    random_state=np.random.RandomState(seed))
        alpha_samples = samples[0]
        hthresh_samples = samples[1]
        sobel_samples = samples[2]

        rss = ia.derive_random_states(np.random.RandomState(seed), 4)
        alpha_expected = [0.2] * nb_images
        hthresh_expected = (
            rss[1].choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          size=(nb_images,)),
            # TODO simplify this to rss[2].randint(5, 100+1)
            #      would currenlty be a bit more ugly, because DiscrUniform
            #      samples two values for a and b first from rss[2]
            iap.DiscreteUniform(5, 100).draw_samples((nb_images,), rss[2])
        )
        hthresh_expected = np.stack(hthresh_expected, axis=-1)

        sobel_expected = rss[3].choice([3, 5, 7],
                                       size=(nb_images,))

        invalid = hthresh_expected[:, 0] > hthresh_expected[:, 1]
        hthresh_expected[invalid, :] = hthresh_expected[invalid, :][:, [1, 0]]
        assert hthresh_expected.shape == (nb_images, 2)
        assert not np.any(hthresh_expected[:, 0] > hthresh_expected[:, 1])

        assert np.allclose(alpha_samples, alpha_expected)
        assert np.allclose(hthresh_samples, hthresh_expected)
        assert np.allclose(sobel_samples, sobel_expected)

    def test_augment_images__alpha_is_zero(self):
        aug = iaa.Canny(
            alpha=0.0,
            hysteresis_thresholds=(0, 10),
            sobel_kernel_size=[3, 5, 7],
            random_state=1)

        image = np.arange(5*5*3).astype(np.uint8).reshape((5, 5, 3))
        image_aug = aug.augment_image(image)
        assert np.array_equal(image_aug, image)

    def test_augment_images__alpha_is_one(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=254,
            color_false=1
        )

        aug = iaa.Canny(
            alpha=1.0,
            hysteresis_thresholds=100,
            sobel_kernel_size=3,
            colorizer=colorizer,
            random_state=1)

        image_single_chan = np.uint8([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0]
        ])
        image = np.tile(image_single_chan[:, :, np.newaxis] * 128, (1, 1, 3))

        # canny image, looks a bit unintuitive, but is what OpenCV returns
        # can be checked via something like
        # print("canny\n", cv2.Canny(image_single_chan*255, threshold1=100,
        #            threshold2=200,
        #            apertureSize=3,
        #            L2gradient=True))
        image_canny = np.array([
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0]
        ], dtype=bool)

        image_aug_expected = np.copy(image)
        image_aug_expected[image_canny] = 254
        image_aug_expected[~image_canny] = 1

        image_aug = aug.augment_image(image)
        assert np.array_equal(image_aug, image_aug_expected)

    def test_augment_images__single_channel(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=254,
            color_false=1
        )

        aug = iaa.Canny(
            alpha=1.0,
            hysteresis_thresholds=100,
            sobel_kernel_size=3,
            colorizer=colorizer,
            random_state=1)

        image_single_chan = np.uint8([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0]
        ])
        image = image_single_chan[:, :, np.newaxis] * 128

        # canny image, looks a bit unintuitive, but is what OpenCV returns
        # can be checked via something like
        # print("canny\n", cv2.Canny(image_single_chan*255, threshold1=100,
        #            threshold2=200,
        #            apertureSize=3,
        #            L2gradient=True))
        image_canny = np.array([
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0]
        ], dtype=bool)

        image_aug_expected = np.copy(image)
        image_aug_expected[image_canny] = int(0.299*254 + 0.587*254 + 0.114*254)
        image_aug_expected[~image_canny] = int(0.299*1 + 0.587*1 + 0.114*1)

        image_aug = aug.augment_image(image)
        assert np.array_equal(image_aug, image_aug_expected)

    def test_augment_images__four_channels(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=254,
            color_false=1
        )

        aug = iaa.Canny(
            alpha=1.0,
            hysteresis_thresholds=100,
            sobel_kernel_size=3,
            colorizer=colorizer,
            random_state=1)

        image_single_chan = np.uint8([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0]
        ])
        image_alpha_channel = np.uint8([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0]
        ]) * 255
        image = np.tile(image_single_chan[:, :, np.newaxis] * 128, (1, 1, 3))
        image = np.dstack([image, image_alpha_channel[:, :, np.newaxis]])
        assert image.ndim == 3
        assert image.shape[-1] == 4

        # canny image, looks a bit unintuitive, but is what OpenCV returns
        # can be checked via something like
        # print("canny\n", cv2.Canny(image_single_chan*255, threshold1=100,
        #            threshold2=200,
        #            apertureSize=3,
        #            L2gradient=True))
        image_canny = np.array([
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0]
        ], dtype=bool)

        image_aug_expected = np.copy(image)
        image_aug_expected[image_canny, 0:3] = 254
        image_aug_expected[~image_canny, 0:3] = 1

        image_aug = aug.augment_image(image)
        assert np.array_equal(image_aug, image_aug_expected)

    def test_augment_images__random_color(self):
        class _Color(iap.StochasticParameter):
            def __init__(self, values):
                super(_Color, self).__init__()
                self.values = values

            def _draw_samples(self, size, random_state):
                v = random_state.choice(self.values)
                return np.full(size, v, dtype=np.uint8)

        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=_Color([253, 254]),
            color_false=_Color([1, 2])
        )

        image_single_chan = np.uint8([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0]
        ])
        image = np.tile(image_single_chan[:, :, np.newaxis] * 128, (1, 1, 3))

        # canny image, looks a bit unintuitive, but is what OpenCV returns
        # can be checked via something like
        # print("canny\n", cv2.Canny(image_single_chan*255, threshold1=100,
        #            threshold2=200,
        #            apertureSize=3,
        #            L2gradient=True))
        image_canny = np.array([
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0]
        ], dtype=bool)

        seen = {
            (253, 1): False,
            (253, 2): False,
            (254, 1): False,
            (254, 2): False
        }
        for i in range(100):
            aug = iaa.Canny(
                alpha=1.0,
                hysteresis_thresholds=100,
                sobel_kernel_size=3,
                colorizer=colorizer,
                random_state=i)

            image_aug = aug.augment_image(image)
            color_true = np.unique(image_aug[image_canny])
            color_false = np.unique(image_aug[~image_canny])
            assert len(color_true) == 1
            assert len(color_false) == 1
            color_true = int(color_true[0])
            color_false = int(color_false[0])

            seen[(int(color_true), int(color_false))] = True
            assert len(seen.keys()) == 4
            if all(seen.values()):
                break
        assert np.all(seen.values())

    def test_augment_images__random_values(self):
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=255,
            color_false=0
        )

        image_single_chan = np.random.RandomState(1).randint(
            0, 255, size=(100, 100)).astype(np.uint8)
        image = np.tile(image_single_chan[:, :, np.newaxis], (1, 1, 3))

        images_canny_uint8 = {}
        for thresh1, thresh2, ksize in itertools.product([100],
                                                         [200],
                                                         [3, 5]):
            if thresh1 > thresh2:
                continue

            image_canny = cv2.Canny(
                image,
                threshold1=thresh1,
                threshold2=thresh2,
                apertureSize=ksize,
                L2gradient=True)
            image_canny_uint8 = np.tile(
                image_canny[:, :, np.newaxis], (1, 1, 3))

            similar = 0
            for key, image_expected in images_canny_uint8.items():
                if np.array_equal(image_canny_uint8, image_expected):
                    similar += 1
            assert similar == 0

            images_canny_uint8[(thresh1, thresh2, ksize)] = image_canny_uint8

        seen = {key: False for key in images_canny_uint8.keys()}

        for i in range(500):
            aug = iaa.Canny(
                alpha=1.0,
                hysteresis_thresholds=(iap.Deterministic(100),
                                       iap.Deterministic(200)),
                sobel_kernel_size=[3, 5],
                colorizer=colorizer,
                random_state=i)

            image_aug = aug.augment_image(image)
            match_index = None
            for key, image_expected in images_canny_uint8.items():
                if np.array_equal(image_aug, image_expected):
                    match_index = key
                    break
            assert match_index is not None
            seen[match_index] = True

            assert len(seen.keys()) == len(images_canny_uint8.keys())
            if all(seen.values()):
                break
        assert np.all(seen.values())

    def test_get_parameters(self):
        alpha = iap.Deterministic(0.2)
        hysteresis_thresholds = iap.Deterministic(10)
        sobel_kernel_size = iap.Deterministic(3)
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=10, color_false=20)
        aug = iaa.Canny(
            alpha=alpha,
            hysteresis_thresholds=hysteresis_thresholds,
            sobel_kernel_size=sobel_kernel_size,
            colorizer=colorizer
        )
        params = aug.get_parameters()
        assert params[0] is alpha
        assert params[1] is hysteresis_thresholds
        assert params[2] is sobel_kernel_size
        assert params[3] is colorizer

    def test___str___single_value_hysteresis(self):
        alpha = iap.Deterministic(0.2)
        hysteresis_thresholds = iap.Deterministic(10)
        sobel_kernel_size = iap.Deterministic(3)
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=10, color_false=20)
        aug = iaa.Canny(
            alpha=alpha,
            hysteresis_thresholds=hysteresis_thresholds,
            sobel_kernel_size=sobel_kernel_size,
            colorizer=colorizer
        )
        observed = aug.__str__()
        expected = ("Canny(alpha=%s, hysteresis_thresholds=%s, "
                    "sobel_kernel_size=%s, colorizer=%s, name=UnnamedCanny, "
                    "deterministic=False)") % (
                        alpha, hysteresis_thresholds, sobel_kernel_size,
                        colorizer)
        assert observed == expected

    def test___str___tuple_as_hysteresis(self):
        alpha = iap.Deterministic(0.2)
        hysteresis_thresholds = (
            iap.Deterministic(10),
            iap.Deterministic(11)
        )
        sobel_kernel_size = iap.Deterministic(3)
        colorizer = iaa.RandomColorsBinaryImageColorizer(
            color_true=10, color_false=20)
        aug = iaa.Canny(
            alpha=alpha,
            hysteresis_thresholds=hysteresis_thresholds,
            sobel_kernel_size=sobel_kernel_size,
            colorizer=colorizer
        )
        observed = aug.__str__()
        expected = ("Canny(alpha=%s, hysteresis_thresholds=(%s, %s), "
                    "sobel_kernel_size=%s, colorizer=%s, name=UnnamedCanny, "
                    "deterministic=False)") % (
                        alpha,
                        hysteresis_thresholds[0], hysteresis_thresholds[1],
                        sobel_kernel_size, colorizer)
        assert observed == expected
