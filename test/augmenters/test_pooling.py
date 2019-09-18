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
import six.moves as sm

import imgaug as ia
import imgaug.random as iarandom
import imgaug.augmenters.pooling as iapooling
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.testutils import reseed


class Test_compute_shape_after_pooling(unittest.TestCase):
    def test_random_shapes_and_kernel_sizes(self):
        shapes = [
            (6, 5),
            (5, 6),
            (6, 6),
            (11, 1),
            (1, 11),
            (0, 1),
            (1, 0),
            (0, 0)
        ]
        kernel_sizes = [1, 2, 3, 5]
        nb_channels_lst = [None, 1, 3, 4]

        # 8*(4*4)*4 = 512 subtests
        gen = itertools.product(shapes, nb_channels_lst)
        for shape_nochan, nb_channels in gen:
            shape = shape_nochan
            if nb_channels is not None:
                shape = tuple(list(shape) + [nb_channels])
            image = np.zeros(shape, dtype=np.uint8)

            for ksize_h, ksize_w in itertools.product(kernel_sizes,
                                                      kernel_sizes):
                with self.subTest(shape=shape, ksize_h=ksize_h,
                                  ksize_w=ksize_w):
                    image_pooled = ia.avg_pool(image, (ksize_h, ksize_w))
                    shape_expected = image_pooled.shape

                    shape_observed = iapooling._compute_shape_after_pooling(
                        shape, ksize_h, ksize_w
                    )

                    assert shape_observed == shape_expected


class _TestPoolingAugmentersBase(object):
    def setUp(self):
        reseed()

    @property
    def augmenter(self):
        raise NotImplementedError()

    @mock.patch("imgaug.augmenters.pooling._AbstractPoolingBase."
                "_augment_hms_and_segmaps")
    def test_augment_segmaps(self, mock_aug_segmaps):
        from imgaug.augmentables.segmaps import SegmentationMapsOnImage
        arr = np.int32([
            [1, 2, 3],
            [4, 5, 6]
        ])
        segmap = SegmentationMapsOnImage(arr, shape=(6, 6, 3))
        rng = iarandom.RNG(0)
        aug = self.augmenter(2, keep_size=False, random_state=rng)

        _ = aug.augment_segmentation_maps(segmap)

        assert mock_aug_segmaps.call_count == 1
        # call 0, args, arg 0, segmap 0 within segmaps list
        assert np.array_equal(
            mock_aug_segmaps.call_args_list[0][0][0][0].arr,
            segmap.arr)

    def _test_augment_keypoints__kernel_size_is_noop(self, kernel_size):
        from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
        kps = [Keypoint(x=1.5, y=5.5), Keypoint(x=5.5, y=1.5)]
        kpsoi = KeypointsOnImage(kps, shape=(6, 6, 3))
        aug = self.augmenter(kernel_size)

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert kpsoi_aug.shape == (6, 6, 3)
        assert np.allclose(kpsoi_aug.to_xy_array(),
                           [[1.5, 5.5],
                            [5.5, 1.5]])

    def test_augment_keypoints__kernel_size_is_zero(self):
        self._test_augment_keypoints__kernel_size_is_noop(0)

    def test_augment_keypoints__kernel_size_is_one(self):
        self._test_augment_keypoints__kernel_size_is_noop(1)

    def _test_augment_heatmaps__kernel_size_is_noop(self, kernel_size):
        from imgaug.augmentables.heatmaps import HeatmapsOnImage
        arr = np.float32([
            [0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6]
        ])
        heatmaps = HeatmapsOnImage(arr, shape=(6, 6, 3))
        aug = self.augmenter(kernel_size)

        heatmaps_aug = aug.augment_heatmaps(heatmaps)

        assert heatmaps_aug.shape == (6, 6, 3)
        assert np.allclose(heatmaps_aug.arr_0to1, arr[..., np.newaxis])

    def test_augment_heatmaps__kernel_size_is_zero(self):
        self._test_augment_heatmaps__kernel_size_is_noop(0)

    def test_augment_heatmaps__kernel_size_is_one(self):
        self._test_augment_heatmaps__kernel_size_is_noop(1)

    def test_augment_keypoints__kernel_size_is_two__keep_size(self):
        from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
        kps = [Keypoint(x=1.5, y=5.5), Keypoint(x=5.5, y=1.5)]
        kpsoi = KeypointsOnImage(kps, shape=(6, 6, 3))
        aug = self.augmenter(2, keep_size=True)

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert kpsoi_aug.shape == (6, 6, 3)
        assert np.allclose(kpsoi_aug.to_xy_array(),
                           [[1.5, 5.5],
                            [5.5, 1.5]])

    def test_augment_keypoints__kernel_size_is_two__no_keep_size(self):
        from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
        kps = [Keypoint(x=1.5, y=5.5), Keypoint(x=5.5, y=1.5)]
        kpsoi = KeypointsOnImage(kps, shape=(6, 6, 3))
        aug = self.augmenter(2, keep_size=False)

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert kpsoi_aug.shape == (3, 3, 3)
        assert np.allclose(kpsoi_aug.to_xy_array(),
                           [[1.5/2, 5.5/2],
                            [5.5/2, 1.5/2]])

    def test_augment_heatmaps__kernel_size_is_two__keep_size(self):
        from imgaug.augmentables.heatmaps import HeatmapsOnImage
        arr = np.float32([
            [0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6]
        ])
        heatmaps = HeatmapsOnImage(arr, shape=(6, 6, 3))
        aug = self.augmenter(2, keep_size=True)

        heatmaps_aug = aug.augment_heatmaps(heatmaps)

        assert heatmaps_aug.shape == (6, 6, 3)
        assert np.allclose(heatmaps_aug.arr_0to1, arr[..., np.newaxis])

    def test_augment_heatmaps__kernel_size_is_two__no_keep_size(self):
        from imgaug.augmentables.heatmaps import HeatmapsOnImage
        arr = np.float32([
            [0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6]
        ])
        heatmaps = HeatmapsOnImage(arr, shape=(6, 6, 3))
        aug = self.augmenter(2, keep_size=False)

        heatmaps_aug = aug.augment_heatmaps(heatmaps)

        # heatmap aug is only supposed to update the image shape as the library
        # can handle heatmaps of different size than the image, so heatmap
        # array stays the same
        assert heatmaps_aug.shape == (3, 3, 3)
        assert np.allclose(heatmaps_aug.arr_0to1, arr[..., np.newaxis])

    def test_augment_keypoints__kernel_size_differs(self):
        from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
        kps = [Keypoint(x=1.5, y=5.5), Keypoint(x=5.5, y=1.5)]
        kpsoi = KeypointsOnImage(kps, shape=(6, 6, 3))
        aug = self.augmenter(
            (iap.Deterministic(3), iap.Deterministic(2)),
            keep_size=False)

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert kpsoi_aug.shape == (2, 3, 3)
        assert np.allclose(kpsoi_aug.to_xy_array(),
                           [[(1.5/6)*3, (5.5/6)*2],
                            [(5.5/6)*3, (1.5/6)*2]])

    def test_augment_keypoints__kernel_size_differs__requires_padding(self):
        from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
        kps = [Keypoint(x=1.5, y=5.5), Keypoint(x=5.5, y=1.5)]
        kpsoi = KeypointsOnImage(kps, shape=(5, 6, 3))
        aug = self.augmenter(
            (iap.Deterministic(3), iap.Deterministic(2)),
            keep_size=False)

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert kpsoi_aug.shape == (2, 3, 3)
        assert np.allclose(kpsoi_aug.to_xy_array(),
                           [[(1.5/6)*3, (5.5/5)*2],
                            [(5.5/6)*3, (1.5/5)*2]])

    def test_augment_polygons__kernel_size_differs(self):
        from imgaug.augmentables.polys import Polygon, PolygonsOnImage
        polys = [Polygon([(1.5, 5.5), (5.5, 1.5), (5.5, 5.5)])]
        psoi = PolygonsOnImage(polys, shape=(6, 6, 3))
        aug = self.augmenter(
            (iap.Deterministic(3), iap.Deterministic(2)),
            keep_size=False)

        psoi_aug = aug.augment_polygons(psoi)

        assert psoi_aug.shape == (2, 3, 3)
        assert np.allclose(psoi_aug.polygons[0].exterior,
                           [[(1.5/6)*3, (5.5/6)*2],
                            [(5.5/6)*3, (1.5/6)*2],
                            [(5.5/6)*3, (5.5/6)*2]])

    def test_keypoint_alignment(self):
        from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
        aug = self.augmenter((1, 2), keep_size=False)
        image = np.zeros((40, 40, 1), dtype=np.uint8)

        kps = [Keypoint(x=10, y=10), Keypoint(x=30, y=30)]
        kpsoi = KeypointsOnImage(kps, shape=image.shape)
        kpsoi_empty = KeypointsOnImage([], shape=image.shape)

        images_batch = [image, image, image, image]
        kpsoi_batch = [kpsoi, kpsoi, kpsoi_empty, kpsoi]

        nb_iterations = 10
        for _ in sm.xrange(nb_iterations):
            images_aug, kpsois_aug = aug(images=images_batch,
                                         keypoints=kpsoi_batch)

            for index in [0, 1, 3]:
                image_aug = images_aug[index]
                kpsoi_aug = kpsois_aug[index]

                assert image_aug.shape == kpsoi_aug.shape

                if image_aug.shape == (20, 20, 1):
                    assert np.allclose(
                        kpsoi_aug.to_xy_array(),
                        [[5, 5], [15, 15]]
                    )
                else:
                    assert np.allclose(
                        kpsoi_aug.to_xy_array(),
                        [[10, 10], [30, 30]]
                    )

            for index in [2]:
                image_aug = images_aug[index]
                kpsoi_aug = kpsois_aug[index]

                assert kpsoi_aug.shape == image_aug.shape
                assert len(kpsoi_aug.keypoints) == 0

    def test_empty_keypoints(self):
        from imgaug.augmentables.kps import KeypointsOnImage
        kpsoi = KeypointsOnImage([], shape=(5, 6, 3))
        aug = self.augmenter(3, keep_size=False)

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert kpsoi_aug.shape == (2, 2, 3)
        assert len(kpsoi_aug.keypoints) == 0

    def test_empty_polygons(self):
        from imgaug.augmentables.polys import PolygonsOnImage
        psoi = PolygonsOnImage([], shape=(5, 6, 3))
        aug = self.augmenter(3, keep_size=False)

        psoi_aug = aug.augment_polygons(psoi)

        assert psoi_aug.shape == (2, 2, 3)
        assert len(psoi_aug.polygons) == 0

    def test_get_parameters(self):
        aug = self.augmenter(2)
        params = aug.get_parameters()
        assert len(params) == 2
        assert len(params[0]) == 2
        assert isinstance(params[0][0], iap.Deterministic)
        assert params[0][0].value == 2
        assert params[0][1] is None


# TODO add test that checks the padding behaviour
class TestAveragePooling(_TestPoolingAugmentersBase, unittest.TestCase):
    @property
    def augmenter(self):
        return iaa.AveragePooling

    def test___init___default_settings(self):
        aug = iaa.AveragePooling(2)
        assert len(aug.kernel_size) == 2
        assert isinstance(aug.kernel_size[0], iap.Deterministic)
        assert aug.kernel_size[0].value == 2
        assert aug.kernel_size[1] is None
        assert aug.keep_size is True

    def test___init___custom_settings(self):
        aug = iaa.AveragePooling(((2, 4), (5, 6)), keep_size=False)
        assert len(aug.kernel_size) == 2
        assert isinstance(aug.kernel_size[0], iap.DiscreteUniform)
        assert isinstance(aug.kernel_size[1], iap.DiscreteUniform)
        assert aug.kernel_size[0].a.value == 2
        assert aug.kernel_size[0].b.value == 4
        assert aug.kernel_size[1].a.value == 5
        assert aug.kernel_size[1].b.value == 6
        assert aug.keep_size is False

    def test_augment_images__kernel_size_is_zero(self):
        aug = iaa.AveragePooling(0)
        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))
        assert np.array_equal(aug.augment_image(image), image)

    def test_augment_images__kernel_size_is_one(self):
        aug = iaa.AveragePooling(1)
        image = np.arange(6*6*3).astype(np.uint8).reshape((6, 6, 3))
        assert np.array_equal(aug.augment_image(image), image)

    def test_augment_images__kernel_size_is_two__array_of_100s(self):
        aug = iaa.AveragePooling(2, keep_size=False)
        image = np.full((6, 6, 3), 100, dtype=np.uint8)
        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - 100)
        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == (3, 3, 3)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_is_two__custom_array(self):
        aug = iaa.AveragePooling(2, keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50, 120]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == (1, 2, 3)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_is_two__four_channels(self):
        aug = iaa.AveragePooling(2, keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 4))

        expected = np.uint8([
            [50, 120]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 4))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == (1, 2, 4)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_differs(self):
        aug = iaa.AveragePooling(
            (iap.Deterministic(3), iap.Deterministic(2)),
            keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+2, 120-1],
            [50-5, 50+5, 120-2, 120+1],
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50, 120]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == (1, 2, 3)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_differs__requires_padding(self):
        aug = iaa.AveragePooling(
            (iap.Deterministic(3), iap.Deterministic(1)),
            keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+2, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [(50-2 + 50+1 + 50-2)/3,
             (50-1 + 50+2 + 50-1)/3,
             (120-4 + 120+2 + 120-4)/3,
             (120+4 + 120-1 + 120+4)/3]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)

        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == (1, 4, 3)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_is_two__keep_size(self):
        aug = iaa.AveragePooling(2, keep_size=True)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50, 50, 120, 120],
            [50, 50, 120, 120]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)

        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == (2, 4, 3)
        assert np.all(diff <= 1)

    def test_augment_images__kernel_size_is_two__single_channel(self):
        aug = iaa.AveragePooling(2, keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = image[:, :, np.newaxis]

        expected = np.uint8([
            [50, 120]
        ])
        expected = expected[:, :, np.newaxis]

        image_aug = aug.augment_image(image)

        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == (1, 2, 1)
        assert np.all(diff <= 1)


# TODO add test that checks the padding behaviour
# We don't have many tests here, because MaxPooling and AveragePooling derive
# from the same base class, i.e. they share most of the methods, which are then
# tested via TestAveragePooling.
class TestMaxPooling(_TestPoolingAugmentersBase, unittest.TestCase):
    @property
    def augmenter(self):
        return iaa.MaxPooling

    def test_augment_images(self):
        aug = iaa.MaxPooling(2, keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50+2, 120+4]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.shape == (1, 2, 3)
        assert np.all(diff <= 1)

    def test_augment_images__different_channels(self):
        aug = iaa.MaxPooling((iap.Deterministic(1), iap.Deterministic(4)),
                             keep_size=False)

        c1 = np.arange(start=1, stop=8+1).reshape((1, 8, 1))
        c2 = (100 + np.arange(start=1, stop=8+1)).reshape((1, 8, 1))
        image = np.dstack([c1, c2]).astype(np.uint8)

        c1_expected = np.uint8([4, 8]).reshape((1, 2, 1))
        c2_expected = np.uint8([100+4, 100+8]).reshape((1, 2, 1))
        image_expected = np.dstack([c1_expected, c2_expected])

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - image_expected)
        assert image_aug.shape == (1, 2, 2)
        assert np.all(diff <= 1)


# TODO add test that checks the padding behaviour
# We don't have many tests here, because MinPooling and AveragePooling derive
# from the same base class, i.e. they share most of the methods, which are then
# tested via TestAveragePooling.
class TestMinPooling(_TestPoolingAugmentersBase, unittest.TestCase):
    @property
    def augmenter(self):
        return iaa.MinPooling

    def test_augment_images(self):
        aug = iaa.MinPooling(2, keep_size=False)

        image = np.uint8([
            [50-2, 50-1, 120-4, 120+4],
            [50+1, 50+2, 120+1, 120-1]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50-2, 120-4]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.shape == (1, 2, 3)
        assert np.all(diff <= 1)

    def test_augment_images__different_channels(self):
        aug = iaa.MinPooling((iap.Deterministic(1), iap.Deterministic(4)),
                             keep_size=False)

        c1 = np.arange(start=1, stop=8+1).reshape((1, 8, 1))
        c2 = (100 + np.arange(start=1, stop=8+1)).reshape((1, 8, 1))
        image = np.dstack([c1, c2]).astype(np.uint8)

        c1_expected = np.uint8([1, 5]).reshape((1, 2, 1))
        c2_expected = np.uint8([100+1, 100+4]).reshape((1, 2, 1))
        image_expected = np.dstack([c1_expected, c2_expected])

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - image_expected)
        assert image_aug.shape == (1, 2, 2)
        assert np.all(diff <= 1)


# TODO add test that checks the padding behaviour
# We don't have many tests here, because MedianPooling and AveragePooling
# derive from the same base class, i.e. they share most of the methods, which
# are then tested via TestAveragePooling.
class TestMedianPool(_TestPoolingAugmentersBase, unittest.TestCase):
    @property
    def augmenter(self):
        return iaa.MedianPooling

    def test_augment_images(self):
        aug = iaa.MedianPooling(3, keep_size=False)

        image = np.uint8([
            [50-9, 50-8, 50-7, 120-5, 120-5, 120-5],
            [50-5, 50+0, 50+3, 120-3, 120+0, 120+1],
            [50+8, 50+9, 50+9, 120+2, 120+3, 120+4]
        ])
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

        expected = np.uint8([
            [50, 120]
        ])
        expected = np.tile(expected[:, :, np.newaxis], (1, 1, 3))

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - expected)
        assert image_aug.shape == (1, 2, 3)
        assert np.all(diff <= 1)

    def test_augment_images__different_channels(self):
        aug = iaa.MinPooling((iap.Deterministic(1), iap.Deterministic(3)),
                             keep_size=False)

        c1 = np.arange(start=1, stop=9+1).reshape((1, 9, 1))
        c2 = (100 + np.arange(start=1, stop=9+1)).reshape((1, 9, 1))
        image = np.dstack([c1, c2]).astype(np.uint8)

        c1_expected = np.uint8([2, 5, 8]).reshape((1, 3, 1))
        c2_expected = np.uint8([100+2, 100+5, 100+8]).reshape((1, 3, 1))
        image_expected = np.dstack([c1_expected, c2_expected])

        image_aug = aug.augment_image(image)
        diff = np.abs(image_aug.astype(np.int32) - image_expected)
        assert image_aug.shape == (1, 3, 2)
        assert np.all(diff <= 1)
