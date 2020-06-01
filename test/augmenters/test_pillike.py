from __future__ import print_function, division, absolute_import

import functools
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

import numpy as np
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageFilter

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import random as iarandom
from imgaug.testutils import reseed, runtest_pickleable_uint8_img, assertWarns


def _test_shape_hw(func):
    img = np.arange(20*10).reshape((20, 10)).astype(np.uint8)

    observed = func(np.copy(img))

    expected = func(
        np.tile(np.copy(img)[:, :, np.newaxis], (1, 1, 3)),
    )[:, :, 0]
    assert observed.dtype.name == "uint8"
    assert observed.shape == (20, 10)
    assert np.array_equal(observed, expected)


def _test_shape_hw1(func):
    img = np.arange(20*10*1).reshape((20, 10, 1)).astype(np.uint8)

    observed = func(np.copy(img))

    expected = func(
        np.tile(np.copy(img), (1, 1, 3)),
    )[:, :, 0:1]
    assert observed.dtype.name == "uint8"
    assert observed.shape == (20, 10, 1)
    assert np.array_equal(observed, expected)


class Test_solarize_(unittest.TestCase):
    @mock.patch("imgaug.augmenters.arithmetic.invert_")
    def test_mocked_defaults(self, mock_sol):
        arr = np.zeros((1,), dtype=np.uint8)
        mock_sol.return_value = "foo"

        observed = iaa.pillike.solarize_(arr)

        args = mock_sol.call_args_list[0][0]
        kwargs = mock_sol.call_args_list[0][1]
        assert args[0] is arr
        assert kwargs["threshold"] == 128
        assert observed == "foo"

    @mock.patch("imgaug.augmenters.arithmetic.invert_")
    def test_mocked(self, mock_sol):
        arr = np.zeros((1,), dtype=np.uint8)
        mock_sol.return_value = "foo"

        observed = iaa.pillike.solarize_(arr, threshold=5)

        args = mock_sol.call_args_list[0][0]
        kwargs = mock_sol.call_args_list[0][1]
        assert args[0] is arr
        assert kwargs["threshold"] == 5
        assert observed == "foo"

    def test_image_shape_hw(self):
        func = functools.partial(iaa.pillike.solarize_, threshold=5)
        _test_shape_hw(func)

    def test_image_shape_hw1(self):
        func = functools.partial(iaa.pillike.solarize_, threshold=5)
        _test_shape_hw1(func)


class Test_solarize(unittest.TestCase):
    def test_compare_with_pil(self):
        def _solarize_pil(image, threshold):
            img = PIL.Image.fromarray(image)
            return np.asarray(PIL.ImageOps.solarize(img, threshold))

        images = [
            np.mod(np.arange(20*20*3), 255).astype(np.uint8)\
                .reshape((20, 20, 3)),
            iarandom.RNG(0).integers(0, 256, size=(1, 1, 3), dtype="uint8"),
            iarandom.RNG(1).integers(0, 256, size=(20, 20, 3), dtype="uint8"),
            iarandom.RNG(2).integers(0, 256, size=(40, 40, 3), dtype="uint8"),
            iarandom.RNG(0).integers(0, 256, size=(20, 20), dtype="uint8")
        ]

        for image_idx, image in enumerate(images):
            for threshold in np.arange(256):
                with self.subTest(image_idx=image_idx, threshold=threshold):
                    image_pil = _solarize_pil(image, threshold)
                    image_iaa = iaa.pillike.solarize(image, threshold)
                    assert np.array_equal(image_pil, image_iaa)

    def test_image_shape_hw(self):
        func = functools.partial(iaa.pillike.solarize, threshold=5)
        _test_shape_hw(func)

    def test_image_shape_hw1(self):
        func = functools.partial(iaa.pillike.solarize, threshold=5)
        _test_shape_hw1(func)


class Test_posterize(unittest.TestCase):
    def test_by_comparison_with_pil(self):
        image = np.arange(64*64*3).reshape((64, 64, 3))
        image = np.mod(image, 255).astype(np.uint8)
        for nb_bits in [1, 2, 3, 4, 5, 6, 7, 8]:
            image_iaa = iaa.pillike.posterize(np.copy(image), nb_bits)
            image_pil = np.asarray(
                PIL.ImageOps.posterize(
                    PIL.Image.fromarray(image),
                    nb_bits
                )
            )

            assert np.array_equal(image_iaa, image_pil)

    def test_image_shape_hw(self):
        func = functools.partial(iaa.pillike.posterize, bits=2)
        _test_shape_hw(func)

    def test_image_shape_hw1(self):
        func = functools.partial(iaa.pillike.posterize, bits=2)
        _test_shape_hw1(func)


class Test_equalize(unittest.TestCase):
    def test_by_comparison_with_pil(self):
        shapes = [
            (1, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (5, 5),
            (10, 5),
            (5, 10),
            (10, 10),
            (20, 20),
            (100, 100),
            (100, 200),
            (200, 100),
            (200, 200)
        ]
        shapes = shapes + [shape + (3,) for shape in shapes]

        rng = iarandom.RNG(0)
        images = [rng.integers(0, 255, size=shape).astype(np.uint8)
                  for shape in shapes]
        images = images + [
            np.full((10, 10), 0, dtype=np.uint8),
            np.full((10, 10), 128, dtype=np.uint8),
            np.full((10, 10), 255, dtype=np.uint8)
        ]

        for i, image in enumerate(images):
            mask_vals = [False, True] if image.size >= (100*100) else [False]
            for use_mask in mask_vals:
                with self.subTest(image_idx=i, shape=image.shape,
                                  use_mask=use_mask):
                    mask_np = None
                    mask_pil = None
                    if use_mask:
                        mask_np = np.zeros(image.shape[0:2], dtype=np.uint8)
                        mask_np[25:75, 25:75] = 1
                        mask_pil = PIL.Image.fromarray(mask_np).convert("L")

                    image_iaa = iaa.pillike.equalize(image, mask=mask_np)
                    image_pil = np.asarray(
                        PIL.ImageOps.equalize(
                            PIL.Image.fromarray(image),
                            mask=mask_pil
                        )
                    )

                    assert np.array_equal(image_iaa, image_pil)

    def test_unusual_channel_numbers(self):
        nb_channels_lst = [1, 2, 4, 5, 512, 513]
        for nb_channels in nb_channels_lst:
            for size in [20, 100]:
                with self.subTest(nb_channels=nb_channels,
                                  size=size):
                    shape = (size, size, nb_channels)
                    image = iarandom.RNG(0).integers(50, 150, size=shape)
                    image = image.astype(np.uint8)

                    image_aug = iaa.pillike.equalize(image)

                    if size > 1:
                        channelwise_sums = np.sum(image_aug, axis=(0, 1))
                        assert np.all(channelwise_sums > 0)
                    assert np.min(image_aug) < 50
                    assert np.max(image_aug) > 150

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

                image_aug = iaa.pillike.equalize(image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_image_shape_hw(self):
        func = functools.partial(iaa.pillike.equalize)
        _test_shape_hw(func)

    # already covered by unusal channel numbers test, but we run this one here
    # anyways for consistency with other tests and because it works a bit
    # different
    def test_image_shape_hw1(self):
        func = functools.partial(iaa.pillike.equalize)
        _test_shape_hw1(func)


class Test_autocontrast(unittest.TestCase):
    def test_by_comparison_with_pil(self):
        rng = iarandom.RNG(0)
        shapes = [
            (1, 1),
            (10, 10),
            (1, 1, 3),
            (1, 2, 3),
            (2, 1, 3),
            (2, 2, 3),
            (5, 3, 3),
            (10, 5, 3),
            (5, 10, 3),
            (10, 10, 3),
            (20, 10, 3),
            (20, 40, 3),
            (50, 60, 3),
            (100, 100, 3),
            (200, 100, 3)
        ]
        images = [
            rng.integers(0, 255, size=shape).astype(np.uint8)
            for shape in shapes
        ]
        images = (
            images
            + [
                np.full((1, 1, 3), 0, dtype=np.uint8),
                np.full((1, 1, 3), 255, dtype=np.uint8),
                np.full((20, 20, 3), 0, dtype=np.uint8),
                np.full((20, 20, 3), 255, dtype=np.uint8)
            ]
        )

        cutoffs = [0, 1, 2, 10, 50, 90, 99, 100]
        ignores = [None, 0, 1, 100, 255, [0, 1], [5, 10, 50], [99, 100]]

        for cutoff in cutoffs:
            for ignore in ignores:
                for i, image in enumerate(images):
                    with self.subTest(cutoff=cutoff, ignore=ignore,
                                      image_idx=i, image_shape=image.shape):
                        result_pil = np.asarray(
                            PIL.ImageOps.autocontrast(
                                PIL.Image.fromarray(image),
                                cutoff=cutoff,
                                ignore=ignore
                            )
                        )
                        result_iaa = iaa.pillike.autocontrast(image,
                                                              cutoff=cutoff,
                                                              ignore=ignore)
                        assert np.array_equal(result_pil, result_iaa)

    def test_unusual_channel_numbers(self):
        nb_channels_lst = [1, 2, 4, 5, 512, 513]
        for nb_channels in nb_channels_lst:
            for size in [20]:
                for cutoff in [0, 1, 10]:
                    with self.subTest(nb_channels=nb_channels,
                                      size=size,
                                      cutoff=cutoff):
                        shape = (size, size, nb_channels)
                        image = iarandom.RNG(0).integers(50, 150, size=shape)
                        image = image.astype(np.uint8)

                        image_aug = iaa.pillike.autocontrast(image,
                                                             cutoff=cutoff)

                        if size > 1:
                            channelwise_sums = np.sum(image_aug, axis=(0, 1))
                            assert np.all(channelwise_sums > 0)
                        assert np.min(image_aug) < 50
                        assert np.max(image_aug) > 150

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
            for cutoff in [0, 1, 10]:
                for ignore in [None, 0, 1, [0, 1, 10]]:
                    with self.subTest(shape=shape, cutoff=cutoff,
                                      ignore=ignore):
                        image = np.zeros(shape, dtype=np.uint8)

                        image_aug = iaa.pillike.autocontrast(image,
                                                             cutoff=cutoff,
                                                             ignore=ignore)

                        assert image_aug.dtype.name == "uint8"
                        assert image_aug.shape == shape

    def test_image_shape_hw(self):
        func = functools.partial(iaa.pillike.autocontrast)
        _test_shape_hw(func)

    # already covered by unusal channel numbers test, but we run this one here
    # anyways for consistency with other tests and because it works a bit
    # different
    def test_image_shape_hw1(self):
        func = functools.partial(iaa.pillike.autocontrast)
        _test_shape_hw1(func)


# TODO add test for unusual channel numbers
class _TestEnhanceFunc(unittest.TestCase):
    def _test_by_comparison_with_pil(
            self, func, cls,
            factors=(0.0, 0.01, 0.1, 0.5, 0.95, 0.99, 1.0, 1.05, 1.5, 2.0,
                     3.0)):
        shapes = [(224, 224, 3), (32, 32, 3), (16, 8, 3), (1, 1, 3),
                  (32, 32, 4)]
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for seed in seeds:
            for shape in shapes:
                for factor in factors:
                    with self.subTest(shape=shape, seed=seed, factor=factor):
                        image = iarandom.RNG(seed).integers(
                            0, 256, size=shape, dtype="uint8")

                        image_iaa = func(image, factor)
                        image_pil = np.asarray(
                            cls(
                                PIL.Image.fromarray(image)
                            ).enhance(factor)
                        )

                        assert np.array_equal(image_iaa, image_pil)

    def _test_zero_sized_axes(self, func,
                              factors=(0.0, 0.4, 1.0)):
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
            for factor in factors:
                with self.subTest(shape=shape, factor=factor):
                    image = np.zeros(shape, dtype=np.uint8)

                    image_aug = func(image, factor=factor)

                    assert image_aug.dtype.name == "uint8"
                    assert image_aug.shape == shape

    @classmethod
    def _test_image_shape_hw(self, func):
        func = functools.partial(func, factor=0.2)
        _test_shape_hw(func)

    @classmethod
    def _test_image_shape_hw1(self, func):
        func = functools.partial(func, factor=0.2)
        _test_shape_hw1(func)


class Test_enhance_color(_TestEnhanceFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.enhance_color,
                                          PIL.ImageEnhance.Color)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.enhance_color)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.enhance_color)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.enhance_color)


class Test_enhance_contrast(_TestEnhanceFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.enhance_contrast,
                                          PIL.ImageEnhance.Contrast)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.enhance_contrast)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.enhance_contrast)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.enhance_contrast)


class Test_enhance_brightness(_TestEnhanceFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.enhance_brightness,
                                          PIL.ImageEnhance.Brightness)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.enhance_brightness)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.enhance_brightness)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.enhance_brightness)


class Test_enhance_sharpness(_TestEnhanceFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.enhance_sharpness,
                                          PIL.ImageEnhance.Sharpness)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.enhance_brightness,
                                   factors=[0.0, 0.4, 1.0, 1.5, 2.0])

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.enhance_sharpness)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.enhance_sharpness)


class _TestFilterFunc(unittest.TestCase):
    def _test_by_comparison_with_pil(self, func, pil_kernel):
        shapes = [(224, 224, 3), (32, 32, 3), (16, 8, 3), (1, 1, 3),
                  (32, 32, 4)]
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for seed in seeds:
            for shape in shapes:
                with self.subTest(shape=shape, seed=seed):
                    image = iarandom.RNG(seed).integers(
                        0, 256, size=shape, dtype="uint8")

                    image_iaa = func(image)
                    image_pil = np.asarray(
                        PIL.Image.fromarray(image).filter(pil_kernel)
                    )

                    assert np.array_equal(image_iaa, image_pil)

    def _test_zero_sized_axes(self, func):
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

                image_aug = func(image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    @classmethod
    def _test_image_shape_hw(self, func):
        _test_shape_hw(func)

    @classmethod
    def _test_image_shape_hw1(self, func):
        _test_shape_hw1(func)


class Test_filter_blur(_TestFilterFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.filter_blur,
                                          PIL.ImageFilter.BLUR)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.filter_blur)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.filter_blur)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.filter_blur)


class Test_filter_smooth(_TestFilterFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.filter_smooth,
                                          PIL.ImageFilter.SMOOTH)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.filter_smooth)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.filter_smooth)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.filter_smooth)


class Test_filter_smooth_more(_TestFilterFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.filter_smooth_more,
                                          PIL.ImageFilter.SMOOTH_MORE)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.filter_smooth_more)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.filter_smooth_more)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.filter_smooth_more)


class Test_filter_edge_enhance(_TestFilterFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.filter_edge_enhance,
                                          PIL.ImageFilter.EDGE_ENHANCE)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.filter_edge_enhance)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.filter_edge_enhance)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.filter_edge_enhance)


class Test_filter_edge_enhance_more(_TestFilterFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.filter_edge_enhance_more,
                                          PIL.ImageFilter.EDGE_ENHANCE_MORE)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.filter_edge_enhance_more)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.filter_edge_enhance_more)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.filter_edge_enhance_more)


class Test_filter_find_edges(_TestFilterFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.filter_find_edges,
                                          PIL.ImageFilter.FIND_EDGES)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.filter_find_edges)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.filter_find_edges)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.filter_find_edges)


class Test_filter_contour(_TestFilterFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.filter_contour,
                                          PIL.ImageFilter.CONTOUR)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.filter_contour)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.filter_contour)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.filter_contour)


class Test_filter_emboss(_TestFilterFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.filter_emboss,
                                          PIL.ImageFilter.EMBOSS)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.filter_emboss)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.filter_emboss)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.filter_emboss)


class Test_filter_sharpen(_TestFilterFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.filter_sharpen,
                                          PIL.ImageFilter.SHARPEN)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.filter_sharpen)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.filter_sharpen)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.filter_sharpen)


class Test_filter_detail(_TestFilterFunc):
    def test_by_comparison_with_pil(self):
        self._test_by_comparison_with_pil(iaa.pillike.filter_detail,
                                          PIL.ImageFilter.DETAIL)

    def test_zero_sized_axes(self):
        self._test_zero_sized_axes(iaa.pillike.filter_detail)

    def test_image_shape_hw(self):
        self._test_image_shape_hw(iaa.pillike.filter_detail)

    def test_image_shape_hw1(self):
        self._test_image_shape_hw1(iaa.pillike.filter_detail)


class Test_warp_affine(unittest.TestCase):
    def _test_aff_by_comparison_with_pil(self, arg_name, arg_values,
                                         matrix_gen):
        shapes = [(64, 64, 3), (32, 32, 3), (16, 8, 3), (1, 1, 3),
                  (32, 32, 4)]
        seeds = [1, 2, 3]
        fillcolors = [None, 0, 128, (0, 255, 0)]
        for shape in shapes:
            for seed in seeds:
                for fillcolor in fillcolors:
                    for arg_value in arg_values:
                        with self.subTest(shape=shape, seed=seed,
                                          fillcolor=fillcolor,
                                          **{arg_name: arg_value}):
                            image = iarandom.RNG(seed).integers(
                                0, 256, size=shape, dtype="uint8")

                            matrix = matrix_gen(arg_value)

                            image_warped = iaa.pillike.warp_affine(
                                image,
                                fillcolor=fillcolor,
                                center=(0.0, 0.0),
                                **{arg_name: arg_value})

                            image_warped_exp = np.asarray(
                                PIL.Image.fromarray(
                                    image
                                ).transform(shape[0:2][::-1],
                                            PIL.Image.AFFINE,
                                            matrix[:2, :].flat,
                                            fillcolor=fillcolor)
                            )

                            assert np.array_equal(image_warped,
                                                  image_warped_exp)

    def test_scale_x_by_comparison_with_pil(self):
        def _matrix_gen(scale):
            return np.float32([
                [1/scale, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])

        self._test_aff_by_comparison_with_pil(
            "scale_x",
            [0.01, 0.1, 0.9, 1.0, 1.5, 3.0],
            _matrix_gen
        )

    def test_scale_y_by_comparison_with_pil(self):
        def _matrix_gen(scale):
            return np.float32([
                [1, 0, 0],
                [0, 1/scale, 0],
                [0, 0, 1]
            ])

        self._test_aff_by_comparison_with_pil(
            "scale_y",
            [0.01, 0.1, 0.9, 1.0, 1.5, 3.0],
            _matrix_gen
        )

    def test_translate_x_by_comparison_with_pil(self):
        def _matrix_gen(translate):
            return np.float32([
                [1, 0, -translate],
                [0, 1, 0],
                [0, 0, 1]
            ])

        self._test_aff_by_comparison_with_pil(
            "translate_x_px",
            [-50, -10, -1, 0, 1, 10, 50],
            _matrix_gen
        )

    def test_translate_y_by_comparison_with_pil(self):
        def _matrix_gen(translate):
            return np.float32([
                [1, 0, 0],
                [0, 1, -translate],
                [0, 0, 1]
            ])

        self._test_aff_by_comparison_with_pil(
            "translate_y_px",
            [-50, -10, -1, 0, 1, 10, 50],
            _matrix_gen
        )

    def test_rotate_by_comparison_with_pil(self):
        def _matrix_gen(rotate):
            r = np.deg2rad(rotate)
            return np.float32([
                [np.cos(r), np.sin(r), 0],
                [-np.sin(r), np.cos(r), 0],
                [0, 0, 1]
            ])

        self._test_aff_by_comparison_with_pil(
            "rotate_deg",
            [-50, -10, -1, 0, 1, 10, 50],
            _matrix_gen
        )

    def test_shear_x_by_comparison_with_pil(self):
        def _matrix_gen(shear):
            s = (-1) * np.deg2rad(shear)
            return np.float32([
                [1, np.tanh(s), 0],
                [0, 1, 0],
                [0, 0, 1]
            ])

        self._test_aff_by_comparison_with_pil(
            "shear_x_deg",
            [-50, -10, -1, 0, 1, 10, 50],
            _matrix_gen
        )

    def test_shear_y_by_comparison_with_pil(self):
        def _matrix_gen(shear):
            s = (-1) * np.deg2rad(shear)
            return np.float32([
                [1, 0, 0],
                [np.tanh(s), 1, 0],
                [0, 0, 1]
            ])

        self._test_aff_by_comparison_with_pil(
            "shear_y_deg",
            [-50, -10, -1, 0, 1, 10, 50],
            _matrix_gen
        )

    def test_scale_x(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[50, 60] = 255

        image_aug = iaa.pillike.warp_affine(image, scale_x=1.5)

        y, x = np.unravel_index(np.argmax(image_aug[..., 0]),
                                image_aug.shape[0:2])

        assert 50 - 1 <= y <= 50 + 1
        assert x > 60

    def test_scale_y(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[60, 50] = 255

        image_aug = iaa.pillike.warp_affine(image, scale_y=1.5)

        y, x = np.unravel_index(np.argmax(image_aug[..., 0]),
                                image_aug.shape[0:2])

        assert 50 - 1 <= x <= 50 + 1
        assert y > 60

    def test_translate_x_px(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[10, 15] = 255

        image_aug = iaa.pillike.warp_affine(image, translate_x_px=1)

        assert image_aug[10, 15, 0] == 0
        assert image_aug[10, 16, 0] == 255
        assert np.all(image_aug[0, :] == 0)

    def test_translate_y_px(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[15, 10] = 255

        image_aug = iaa.pillike.warp_affine(image, translate_y_px=1)

        assert image_aug[15, 10, 0] == 0
        assert image_aug[16, 10, 0] == 255
        assert np.all(image_aug[:, 0] == 0)

    def test_rotate(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[0, 10] = 255

        image_aug = iaa.pillike.warp_affine(image,
                                            rotate_deg=45,
                                            center=(0.0, 0.0))

        assert image_aug[7, 7, 0] == 255

    def test_shear_x(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[5, 10] = 255

        image_aug = iaa.pillike.warp_affine(image,
                                            shear_x_deg=20,
                                            center=(0.0, 0.0))

        y, x = np.unravel_index(np.argmax(image_aug[..., 0]),
                                image_aug.shape[0:2])

        assert y == 5
        assert x > 10

    def test_shear_y(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[10, 15] = 255

        image_aug = iaa.pillike.warp_affine(image,
                                            shear_y_deg=20,
                                            center=(0.0, 0.0))

        y, x = np.unravel_index(np.argmax(image_aug[..., 0]),
                                image_aug.shape[0:2])

        assert y > 10
        assert x == 15

    def test_fillcolor_is_none(self):
        image = np.ones((20, 20, 3), dtype=np.uint8)

        image_aug = iaa.pillike.warp_affine(image,
                                            translate_x_px=1,
                                            fillcolor=None)

        assert np.all(image_aug[:, :1, :] == 0)
        assert np.all(image_aug[:, 1:, :] == 1)

    def test_fillcolor_is_int(self):
        image = np.ones((20, 20, 3), dtype=np.uint8)

        image_aug = iaa.pillike.warp_affine(image,
                                            translate_x_px=1,
                                            fillcolor=128)

        assert np.all(image_aug[:, :1, 0] == 128)
        assert np.all(image_aug[:, :1, 1] == 0)
        assert np.all(image_aug[:, :1, 2] == 0)
        assert np.all(image_aug[:, 1:, :] == 1)

    def test_fillcolor_is_int_grayscale(self):
        image = np.ones((20, 20), dtype=np.uint8)

        image_aug = iaa.pillike.warp_affine(image,
                                            translate_x_px=1,
                                            fillcolor=128)

        assert np.all(image_aug[:, :1] == 128)
        assert np.all(image_aug[:, 1:] == 1)

    def test_fillcolor_is_tuple(self):
        image = np.ones((20, 20, 3), dtype=np.uint8)

        image_aug = iaa.pillike.warp_affine(image,
                                            translate_x_px=1,
                                            fillcolor=(2, 3, 4))

        assert np.all(image_aug[:, :1, 0] == 2)
        assert np.all(image_aug[:, :1, 1] == 3)
        assert np.all(image_aug[:, :1, 2] == 4)
        assert np.all(image_aug[:, 1:, :] == 1)

    def test_fillcolor_is_tuple_more_values_than_channels(self):
        image = np.ones((20, 20, 3), dtype=np.uint8)

        image_aug = iaa.pillike.warp_affine(image,
                                            translate_x_px=1,
                                            fillcolor=(2, 3, 4, 5))

        assert image_aug.shape == (20, 20, 3)
        assert np.all(image_aug[:, :1, 0] == 2)
        assert np.all(image_aug[:, :1, 1] == 3)
        assert np.all(image_aug[:, :1, 2] == 4)
        assert np.all(image_aug[:, 1:, :] == 1)

    def test_center(self):
        image = np.zeros((21, 21, 3), dtype=np.uint8)
        image[2, 10] = 255

        image_aug = iaa.pillike.warp_affine(image,
                                            rotate_deg=90,
                                            center=(0.5, 0.5))

        assert image_aug[10, 18, 0] == 255

    def test_image_shape_hw(self):
        func = functools.partial(iaa.pillike.warp_affine, rotate_deg=90)
        _test_shape_hw(func)

    def test_image_shape_hw1(self):
        func = functools.partial(iaa.pillike.warp_affine, rotate_deg=90)
        _test_shape_hw1(func)


class TestSolarize(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_returns_correct_instance(self):
        aug = iaa.pillike.Solarize()
        assert isinstance(aug, iaa.Invert)
        assert aug.per_channel.value == 0
        assert aug.min_value is None
        assert aug.max_value is None
        assert np.isclose(aug.threshold.value, 128)
        assert aug.invert_above_threshold.value == 1

    def test_pickleable(self):
        aug = iaa.pillike.Solarize()
        runtest_pickleable_uint8_img(aug)


class TestPosterize(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_returns_posterize(self):
        aug = iaa.pillike.Posterize()
        assert isinstance(aug, iaa.Posterize)

    def test_pickleable(self):
        aug = iaa.pillike.Posterize()
        runtest_pickleable_uint8_img(aug)


class TestEqualize(unittest.TestCase):
    def setUp(self):
        reseed()

    @mock.patch("imgaug.augmenters.pillike.equalize_")
    def test_mocked(self, mock_eq):
        image = np.arange(1*1*3).astype(np.uint8).reshape((1, 1, 3))
        mock_eq.return_value = np.copy(image)
        aug = iaa.pillike.Equalize()

        _image_aug = aug(image=image)

        assert mock_eq.call_count == 1
        assert np.array_equal(mock_eq.call_args_list[0][0][0], image)

    def test_integrationtest(self):
        rng = iarandom.RNG(0)
        for size in [20, 100]:
            shape = (size, size, 3)
            image = rng.integers(50, 150, size=shape)
            image = image.astype(np.uint8)
            aug = iaa.pillike.Equalize()

            image_aug = aug(image=image)

            if size > 1:
                channelwise_sums = np.sum(image_aug, axis=(0, 1))
                assert np.all(channelwise_sums > 0)
            assert np.min(image_aug) < 50
            assert np.max(image_aug) > 150

    def test_pickleable(self):
        aug = iaa.pillike.Equalize()
        runtest_pickleable_uint8_img(aug)


class TestAutocontrast(unittest.TestCase):
    def setUp(self):
        reseed()

    @mock.patch("imgaug.augmenters.pillike.autocontrast")
    def test_mocked(self, mock_auto):
        image = np.mod(np.arange(10*10*3), 255)
        image = image.reshape((10, 10, 3)).astype(np.uint8)
        mock_auto.return_value = image
        aug = iaa.pillike.Autocontrast(15)

        _image_aug = aug(image=image)

        assert np.array_equal(mock_auto.call_args_list[0][0][0], image)
        assert mock_auto.call_args_list[0][0][1] == 15

    @mock.patch("imgaug.augmenters.pillike.autocontrast")
    def test_per_channel(self, mock_auto):
        image = np.mod(np.arange(10*10*1), 255)
        image = image.reshape((10, 10, 1)).astype(np.uint8)
        image = np.tile(image, (1, 1, 100))
        mock_auto.return_value = image[..., 0]
        aug = iaa.pillike.Autocontrast((0, 30), per_channel=True)

        with assertWarns(self, iaa.SuspiciousSingleImageShapeWarning):
            _image_aug = aug(image=image)

        assert mock_auto.call_count == 100
        cutoffs = []
        for i in np.arange(100):
            assert np.array_equal(mock_auto.call_args_list[i][0][0],
                                  image[..., i])
            cutoffs.append(mock_auto.call_args_list[i][0][1])
        assert len(set(cutoffs)) > 10

    def test_integrationtest(self):
        image = iarandom.RNG(0).integers(50, 150, size=(100, 100, 3))
        image = image.astype(np.uint8)
        aug = iaa.pillike.Autocontrast(10)

        image_aug = aug(image=image)

        assert np.min(image_aug) < 50
        assert np.max(image_aug) > 150

    def test_integrationtest_per_channel(self):
        image = iarandom.RNG(0).integers(50, 150, size=(100, 100, 50))
        image = image.astype(np.uint8)
        aug = iaa.pillike.Autocontrast(10, per_channel=True)

        with assertWarns(self, iaa.SuspiciousSingleImageShapeWarning):
            image_aug = aug(image=image)

        assert np.min(image_aug) < 50
        assert np.max(image_aug) > 150

    def test_pickleable(self):
        aug = iaa.pillike.Autocontrast((0, 30), per_channel=0.5)
        runtest_pickleable_uint8_img(aug)


class TestEnhanceColor(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___defaults(self):
        aug = iaa.pillike.EnhanceColor()
        assert np.isclose(aug.factor.a.value, 0.0)
        assert np.isclose(aug.factor.b.value, 3.0)

    def test___init___custom(self):
        aug = iaa.pillike.EnhanceColor(0.75)
        assert np.isclose(aug.factor.value, 0.75)

    @mock.patch("imgaug.augmenters.pillike.enhance_color")
    def test_mocked(self, mock_pilcol):
        aug = iaa.pillike.EnhanceColor(0.75)
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        mock_pilcol.return_value = np.full((1, 1, 3), 128, dtype=np.uint8)

        image_aug = aug(image=image)

        assert mock_pilcol.call_count == 1
        assert ia.is_np_array(mock_pilcol.call_args_list[0][0][0])
        assert np.isclose(mock_pilcol.call_args_list[0][0][1], 0.75, rtol=0,
                          atol=1e-4)
        assert np.all(image_aug == 128)

    def test_simple_image(self):
        aug = iaa.pillike.EnhanceColor(0.0)
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        image[:, :, 1] = 255

        image_aug = aug(image=image)

        assert image_aug[:, :, 2] > 200
        assert np.all(image_aug[:, :, 0] == image_aug[:, :, 1])
        assert np.all(image_aug[:, :, 0] == image_aug[:, :, 2])

    def test_batch_contains_no_images(self):
        aug = iaa.pillike.EnhanceColor(0.75)
        hm_arr = np.ones((3, 3, 1), dtype=np.float32)
        hm = ia.HeatmapsOnImage(hm_arr, shape=(3, 3, 3))

        hm_aug = aug(heatmaps=hm)

        assert np.allclose(hm_aug.get_arr(), hm.get_arr())

    def test_get_parameters(self):
        aug = iaa.pillike.EnhanceColor(0.75)
        params = aug.get_parameters()
        assert params[0] is aug.factor

    def test_pickleable(self):
        aug = iaa.pillike.EnhanceColor((0.1, 2.0))
        runtest_pickleable_uint8_img(aug)


# we don't have to test very much here, because some functions of the base
# class are already tested via EnhanceColor
class TestEnhanceContrast(unittest.TestCase):
    def setUp(self):
        reseed()

    @mock.patch("imgaug.augmenters.pillike.enhance_contrast")
    def test_mocked(self, mock_pilco):
        aug = iaa.pillike.EnhanceContrast(0.75)
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        mock_pilco.return_value = np.full((1, 1, 3), 128, dtype=np.uint8)

        image_aug = aug(image=image)

        assert mock_pilco.call_count == 1
        assert ia.is_np_array(mock_pilco.call_args_list[0][0][0])
        assert np.isclose(mock_pilco.call_args_list[0][0][1], 0.75, rtol=0,
                          atol=1e-4)
        assert np.all(image_aug == 128)

    def test_simple_image(self):
        aug = iaa.pillike.EnhanceContrast(0.0)
        image = np.full((2, 2, 3), 128, dtype=np.uint8)
        image[0, :, :] = 200

        image_aug = aug(image=image)

        diff_before = np.average(np.abs(image.astype(np.int32)
                                        - np.average(image)))
        diff_after = np.average(np.abs(image_aug.astype(np.int32)
                                       - np.average(image_aug)))
        assert diff_after < diff_before

    def test_batch_contains_no_images(self):
        aug = iaa.pillike.EnhanceContrast(0.75)
        hm_arr = np.ones((3, 3, 1), dtype=np.float32)
        hm = ia.HeatmapsOnImage(hm_arr, shape=(3, 3, 3))

        hm_aug = aug(heatmaps=hm)

        assert np.allclose(hm_aug.get_arr(), hm.get_arr())

    def test_pickleable(self):
        aug = iaa.pillike.EnhanceContrast((0.1, 2.0))
        runtest_pickleable_uint8_img(aug)


# we don't have to test very much here, because some functions of the base
# class are already tested via EnhanceColor
class TestEnhanceBrightness(unittest.TestCase):
    def setUp(self):
        reseed()

    @mock.patch("imgaug.augmenters.pillike.enhance_brightness")
    def test_mocked(self, mock_pilbr):
        aug = iaa.pillike.EnhanceBrightness(0.75)
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        mock_pilbr.return_value = np.full((1, 1, 3), 128, dtype=np.uint8)

        image_aug = aug(image=image)

        assert mock_pilbr.call_count == 1
        assert ia.is_np_array(mock_pilbr.call_args_list[0][0][0])
        assert np.isclose(mock_pilbr.call_args_list[0][0][1], 0.75, rtol=0,
                          atol=1e-4)
        assert np.all(image_aug == 128)

    def test_simple_image(self):
        aug = iaa.pillike.EnhanceBrightness(0.0)
        image = np.full((2, 2, 3), 255, dtype=np.uint8)

        image_aug = aug(image=image)

        assert np.all(image_aug < 255)

    def test_batch_contains_no_images(self):
        aug = iaa.pillike.EnhanceBrightness(0.75)
        hm_arr = np.ones((3, 3, 1), dtype=np.float32)
        hm = ia.HeatmapsOnImage(hm_arr, shape=(3, 3, 3))

        hm_aug = aug(heatmaps=hm)

        assert np.allclose(hm_aug.get_arr(), hm.get_arr())

    def test_pickleable(self):
        aug = iaa.pillike.EnhanceBrightness((0.1, 2.0))
        runtest_pickleable_uint8_img(aug)


# we don't have to test very much here, because some functions of the base
# class are already tested via EnhanceColor
class TestEnhanceSharpness(unittest.TestCase):
    def setUp(self):
        reseed()

    @mock.patch("imgaug.augmenters.pillike.enhance_sharpness")
    def test_mocked(self, mock_pilsh):
        aug = iaa.pillike.EnhanceSharpness(0.75)
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        mock_pilsh.return_value = np.full((3, 3, 3), 128, dtype=np.uint8)

        image_aug = aug(image=image)

        assert mock_pilsh.call_count == 1
        assert ia.is_np_array(mock_pilsh.call_args_list[0][0][0])
        assert np.isclose(mock_pilsh.call_args_list[0][0][1], 0.75, rtol=0,
                          atol=1e-4)
        assert np.all(image_aug == 128)

    def test_simple_image(self):
        aug = iaa.pillike.EnhanceSharpness(2.0)
        image = np.full((3, 3, 3), 64, dtype=np.uint8)
        image[1, 1, :] = 128

        image_aug = aug(image=image)

        assert np.all(image_aug[1, 1, :] > 128)

    def test_batch_contains_no_images(self):
        aug = iaa.pillike.EnhanceSharpness(0.75)
        hm_arr = np.ones((3, 3, 1), dtype=np.float32)
        hm = ia.HeatmapsOnImage(hm_arr, shape=(3, 3, 3))

        hm_aug = aug(heatmaps=hm)

        assert np.allclose(hm_aug.get_arr(), hm.get_arr())

    def test_pickleable(self):
        aug = iaa.pillike.EnhanceSharpness((0.1, 2.0))
        runtest_pickleable_uint8_img(aug)


class _TestFilter(unittest.TestCase):
    def _test___init__(self, cls, func):
        aug = cls()
        assert aug.func is func

    def _test_image(self, cls, pil_kernel):
        image = ia.data.quokka(0.25)
        image_aug = cls()(image=image)
        image_aug_pil = PIL.Image.fromarray(image).filter(pil_kernel)
        assert np.array_equal(image_aug, image_aug_pil)

    def _test_pickleable(self, cls):
        aug = cls()
        runtest_pickleable_uint8_img(aug)


class FilterBlur(_TestFilter):
    def test___init__(self):
        self._test___init__(iaa.pillike.FilterBlur,
                            iaa.pillike.filter_blur)

    def test_image(self):
        self._test_image(iaa.pillike.FilterBlur,
                         PIL.ImageFilter.BLUR)

    def test_pickleable(self):
        self._test_pickleable(iaa.pillike.FilterBlur)


class FilterSmooth(_TestFilter):
    def test___init__(self):
        self._test___init__(iaa.pillike.FilterSmooth,
                            iaa.pillike.filter_smooth)

    def test_image(self):
        self._test_image(iaa.pillike.FilterSmooth,
                         PIL.ImageFilter.SMOOTH)

    def test_pickleable(self):
        self._test_pickleable(iaa.pillike.FilterSmooth)


class FilterSmoothMore(_TestFilter):
    def test___init__(self):
        self._test___init__(iaa.pillike.FilterSmoothMore,
                            iaa.pillike.filter_smooth_more)

    def test_image(self):
        self._test_image(iaa.pillike.FilterSmoothMore,
                         PIL.ImageFilter.SMOOTH_MORE)

    def test_pickleable(self):
        self._test_pickleable(iaa.pillike.FilterSmoothMore)


class FilterEdgeEnhance(_TestFilter):
    def test___init__(self):
        self._test___init__(iaa.pillike.FilterEdgeEnhance,
                            iaa.pillike.filter_edge_enhance)

    def test_image(self):
        self._test_image(iaa.pillike.FilterEdgeEnhance,
                         PIL.ImageFilter.EDGE_ENHANCE)

    def test_pickleable(self):
        self._test_pickleable(iaa.pillike.FilterEdgeEnhance)


class FilterEdgeEnhanceMore(_TestFilter):
    def test___init__(self):
        self._test___init__(iaa.pillike.FilterEdgeEnhanceMore,
                            iaa.pillike.filter_edge_enhance_more)

    def test_image(self):
        self._test_image(iaa.pillike.FilterEdgeEnhanceMore,
                         PIL.ImageFilter.EDGE_ENHANCE_MORE)

    def test_pickleable(self):
        self._test_pickleable(iaa.pillike.FilterEdgeEnhanceMore)


class FilterFindEdges(_TestFilter):
    def test___init__(self):
        self._test___init__(iaa.pillike.FilterFindEdges,
                            iaa.pillike.filter_find_edges)

    def test_image(self):
        self._test_image(iaa.pillike.FilterFindEdges,
                         PIL.ImageFilter.FIND_EDGES)

    def test_pickleable(self):
        self._test_pickleable(iaa.pillike.FilterFindEdges)


class FilterContour(_TestFilter):
    def test___init__(self):
        self._test___init__(iaa.pillike.FilterContour,
                            iaa.pillike.filter_contour)

    def test_image(self):
        self._test_image(iaa.pillike.FilterContour,
                         PIL.ImageFilter.CONTOUR)

    def test_pickleable(self):
        self._test_pickleable(iaa.pillike.FilterContour)


class FilterEmboss(_TestFilter):
    def test___init__(self):
        self._test___init__(iaa.pillike.FilterEmboss,
                            iaa.pillike.filter_emboss)

    def test_image(self):
        self._test_image(iaa.pillike.FilterEmboss,
                         PIL.ImageFilter.EMBOSS)

    def test_pickleable(self):
        self._test_pickleable(iaa.pillike.FilterEmboss)


class FilterSharpen(_TestFilter):
    def test___init__(self):
        self._test___init__(iaa.pillike.FilterSharpen,
                            iaa.pillike.filter_sharpen)

    def test_image(self):
        self._test_image(iaa.pillike.FilterSharpen,
                         PIL.ImageFilter.SHARPEN)

    def test_pickleable(self):
        self._test_pickleable(iaa.pillike.FilterSharpen)


class FilterDetail(_TestFilter):
    def test___init__(self):
        self._test___init__(iaa.pillike.FilterDetail,
                            iaa.pillike.filter_detail)

    def test_image(self):
        self._test_image(iaa.pillike.FilterDetail,
                         PIL.ImageFilter.DETAIL)

    def test_pickleable(self):
        self._test_pickleable(iaa.pillike.FilterDetail)


class TestAffine(unittest.TestCase):
    def setUp(self):
        reseed()

    @mock.patch("imgaug.augmenters.pillike.warp_affine")
    def test_mocked(self, mock_pilaff):
        aug = iaa.pillike.Affine(
            scale={"x": 1.25, "y": 1.5},
            translate_px={"x": 10, "y": 20},
            rotate=30,
            shear={"x": 40, "y": 50},
            fillcolor=100,
            center=(0.1, 0.2)
        )
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        mock_pilaff.return_value = np.full((3, 3, 3), 128, dtype=np.uint8)

        image_aug = aug(image=image)

        assert mock_pilaff.call_count == 1

        args = mock_pilaff.call_args_list[0][0]
        assert np.all(args[0] == 128)  # due to in-place change

        kwargs = mock_pilaff.call_args_list[0][1]
        assert np.isclose(kwargs["scale_x"], 1.25)
        assert np.isclose(kwargs["scale_y"], 1.5)
        assert np.isclose(kwargs["translate_x_px"], 10)
        assert np.isclose(kwargs["translate_y_px"], 20)
        assert np.isclose(kwargs["rotate_deg"], 30)
        assert np.isclose(kwargs["shear_x_deg"], 40)
        assert np.isclose(kwargs["shear_y_deg"], 50)
        assert np.isclose(kwargs["fillcolor"][0], 100)
        assert np.isclose(kwargs["fillcolor"][1], 100)
        assert np.isclose(kwargs["fillcolor"][2], 100)
        assert np.isclose(kwargs["center"][0], 0.1)
        assert np.isclose(kwargs["center"][1], 0.2)
        assert np.all(image_aug == 128)

    @mock.patch("imgaug.augmenters.pillike.warp_affine")
    def test_mocked_translate_percent(self, mock_pilaff):
        aug = iaa.pillike.Affine(
            translate_percent={"x": 1.2, "y": 1.5}
        )
        image = np.zeros((20, 50, 3), dtype=np.uint8)
        mock_pilaff.return_value = np.full((20, 50, 3), 128, dtype=np.uint8)

        image_aug = aug(image=image)

        assert mock_pilaff.call_count == 1

        args = mock_pilaff.call_args_list[0][0]
        assert np.all(args[0] == 128)  # due to in-place change

        kwargs = mock_pilaff.call_args_list[0][1]
        assert np.isclose(kwargs["scale_x"], 1.0)
        assert np.isclose(kwargs["scale_y"], 1.0)
        assert np.isclose(kwargs["translate_x_px"], 50*1.2)
        assert np.isclose(kwargs["translate_y_px"], 20*1.5)
        assert np.isclose(kwargs["rotate_deg"], 0)
        assert np.isclose(kwargs["shear_x_deg"], 0)
        assert np.isclose(kwargs["shear_y_deg"], 0)
        assert np.isclose(kwargs["fillcolor"][0], 0)
        assert np.isclose(kwargs["fillcolor"][1], 0)
        assert np.isclose(kwargs["fillcolor"][2], 0)
        assert np.isclose(kwargs["center"][0], 0.5)
        assert np.isclose(kwargs["center"][1], 0.5)
        assert np.all(image_aug == 128)

    def test_parameters_affect_images(self):
        params = [
            ("scale", {"x": 1.3}),
            ("scale", {"y": 1.5}),
            ("translate_px", {"x": 5}),
            ("translate_px", {"y": 10}),
            ("translate_percent", {"x": 0.3}),
            ("translate_percent", {"y": 0.4}),
            ("rotate", 10),
            ("shear", {"x": 20}),
            ("shear", {"y": 20})
        ]
        image = ia.data.quokka_square((64, 64))

        images_aug = []
        for param_name, param_val in params:
            kwargs = {param_name: param_val}
            aug = iaa.pillike.Affine(**kwargs)
            image_aug = aug(image=image)
            images_aug.append(image_aug)

        for i, image_aug in enumerate(images_aug):
            assert not np.array_equal(image_aug, image)
            for j, other_image_aug in enumerate(images_aug):
                if i != j:
                    assert not np.array_equal(image_aug, other_image_aug)

    def test_batch_contains_no_images(self):
        aug = iaa.pillike.Affine(translate_px={"x": 10})
        hm_arr = np.ones((3, 3, 1), dtype=np.float32)
        hm = ia.HeatmapsOnImage(hm_arr, shape=(3, 3, 3))

        with self.assertRaises(AssertionError):
            _hm_aug = aug(heatmaps=hm)

    def test_get_parameters(self):
        aug = iaa.pillike.Affine(
            scale={"x": 1.25, "y": 1.5},
            translate_px={"x": 10, "y": 20},
            rotate=30,
            shear={"x": 40, "y": 50},
            fillcolor=100,
            center=(0.1, 0.2)
        )
        params = aug.get_parameters()
        assert params[0] is aug.scale
        assert params[1] is aug.translate
        assert params[2] is aug.rotate
        assert params[3] is aug.shear
        assert params[4] is aug.cval
        assert params[5] is aug.center

    def test_pickleable(self):
        aug = iaa.pillike.Affine(
            scale={"x": 1.25, "y": 1.5},
            translate_px={"x": 10, "y": 20},
            rotate=30,
            shear={"x": 40, "y": 50},
            fillcolor=(100, 200),
            center="uniform"
        )
        runtest_pickleable_uint8_img(aug)
