from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractproperty, abstractmethod
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
import six
import six.moves as sm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.testutils import keypoints_equal, reseed
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.augmenters.flip as fliplib


class TestHorizontalFlip(unittest.TestCase):
    def test_returns_fliplr(self):
        aug = iaa.HorizontalFlip(0.5)
        assert isinstance(aug, iaa.Fliplr)
        assert np.allclose(aug.p.p.value, 0.5)


class TestVerticalFlip(unittest.TestCase):
    def test_returns_flipud(self):
        aug = iaa.VerticalFlip(0.5)
        assert isinstance(aug, iaa.Flipud)
        assert np.allclose(aug.p.p.value, 0.5)


@six.add_metaclass(ABCMeta)
class _TestFliplrAndFlipudBase(object):
    def setUp(self):
        reseed()

    @property
    @abstractproperty
    def image(self):
        pass

    @property
    @abstractproperty
    def image_flipped(self):
        pass

    @property
    def images(self):
        return np.array([self.image])

    @property
    def images_flipped(self):
        return np.array([self.image_flipped])

    @property
    @abstractproperty
    def heatmaps(self):
        pass

    @property
    @abstractproperty
    def heatmaps_flipped(self):
        pass

    @property
    @abstractproperty
    def segmaps(self):
        pass

    @property
    @abstractproperty
    def segmaps_flipped(self):
        pass

    @property
    @abstractproperty
    def kpsoi(self):
        pass

    @property
    @abstractproperty
    def kpsoi_flipped(self):
        pass

    @property
    @abstractproperty
    def psoi(self):
        pass

    @property
    @abstractproperty
    def psoi_flipped(self):
        pass

    @abstractmethod
    def create_aug(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_arr(self, value, dtype):
        pass

    @abstractmethod
    def create_arr_flipped(self, value, dtype):
        pass

    def test_images_p_is_0(self):
        aug = self.create_aug(0)

        for _ in sm.xrange(3):
            observed = aug.augment_images(self.images)
            expected = self.images
            assert np.array_equal(observed, expected)

    def test_images_p_is_0__deterministic(self):
        aug = self.create_aug(0).to_deterministic()

        for _ in sm.xrange(3):
            observed = aug.augment_images(self.images)
            expected = self.images
            assert np.array_equal(observed, expected)

    def test_keypoints_p_is_0(self):
        aug = self.create_aug(0)

        for _ in sm.xrange(3):
            observed = aug.augment_keypoints(self.kpsoi)
            expected = self.kpsoi
            assert keypoints_equal(observed, expected)

    def test_keypoints_p_is_0__deterministic(self):
        aug = self.create_aug(0).to_deterministic()

        for _ in sm.xrange(3):
            observed = aug.augment_keypoints(self.kpsoi)
            expected = self.kpsoi
            assert keypoints_equal(observed, expected)

    def test_polygons_p_is_0(self):
        aug = self.create_aug(0)

        for _ in sm.xrange(3):
            observed = aug.augment_polygons(self.psoi)
            assert len(observed) == 1
            assert len(observed[0].polygons) == 1
            assert observed[0].shape == self.psoi[0].shape
            assert observed[0].polygons[0].exterior_almost_equals(
                self.psoi[0].polygons[0])
            assert observed[0].polygons[0].is_valid

    def test_polygons_p_is_0__deterministic(self):
        aug = self.create_aug(0).to_deterministic()

        for _ in sm.xrange(3):
            observed = aug.augment_polygons(self.psoi)
            assert len(observed) == 1
            assert len(observed[0].polygons) == 1
            assert observed[0].shape == self.psoi[0].shape
            assert observed[0].polygons[0].exterior_almost_equals(
                self.psoi[0].polygons[0])
            assert observed[0].polygons[0].is_valid

    def test_heatmaps_p_is_0(self):
        aug = self.create_aug(0)
        heatmaps = self.heatmaps
        observed = aug.augment_heatmaps(heatmaps)
        assert observed.shape == heatmaps.shape
        assert np.isclose(observed.min_value, heatmaps.min_value,
                          rtol=0, atol=1e-6)
        assert np.isclose(observed.max_value, heatmaps.max_value,
                          rtol=0, atol=1e-6)
        assert np.array_equal(observed.get_arr(), heatmaps.get_arr())

    def test_segmaps_p_is_0(self):
        aug = self.create_aug(0)
        observed = aug.augment_segmentation_maps(self.segmaps)
        assert observed.shape == self.segmaps.shape
        assert np.array_equal(observed.get_arr(), self.segmaps.get_arr())

    def test_images_p_is_1(self):
        aug = self.create_aug(1.0)

        for _ in sm.xrange(3):
            observed = aug.augment_images(self.images)
            expected = self.images_flipped
            assert np.array_equal(observed, expected)

    def test_images_p_is_1__deterministic(self):
        aug = self.create_aug(1.0).to_deterministic()

        for _ in sm.xrange(3):
            observed = aug.augment_images(self.images)
            expected = self.images_flipped
            assert np.array_equal(observed, expected)

    def test_keypoints_p_is_1(self):
        aug = self.create_aug(1.0)

        for _ in sm.xrange(3):
            observed = aug.augment_keypoints(self.kpsoi)
            expected = self.kpsoi_flipped
            assert keypoints_equal(observed, expected)

    def test_keypoints_p_is_1__deterministic(self):
        aug = self.create_aug(1.0).to_deterministic()

        for _ in sm.xrange(3):
            observed = aug.augment_keypoints(self.kpsoi)
            expected = self.kpsoi_flipped
            assert keypoints_equal(observed, expected)

    def test_polygons_p_is_1(self):
        aug = self.create_aug(1.0)

        for _ in sm.xrange(3):
            observed = aug.augment_polygons(self.psoi)
            assert len(observed) == 1
            assert len(observed[0].polygons) == 1
            assert observed[0].shape == self.psoi[0].shape
            assert observed[0].polygons[0].exterior_almost_equals(
                self.psoi_flipped[0].polygons[0])
            assert observed[0].polygons[0].is_valid

    def test_polygons_p_is_1__deterministic(self):
        aug = self.create_aug(1.0).to_deterministic()

        for _ in sm.xrange(3):
            observed = aug.augment_polygons(self.psoi)
            assert len(observed) == 1
            assert len(observed[0].polygons) == 1
            assert observed[0].shape == self.psoi[0].shape
            assert observed[0].polygons[0].exterior_almost_equals(
                self.psoi_flipped[0].polygons[0])
            assert observed[0].polygons[0].is_valid

    def test_heatmaps_p_is_1(self):
        aug = self.create_aug(1.0)
        heatmaps = self.heatmaps
        observed = aug.augment_heatmaps(heatmaps)
        assert observed.shape == heatmaps.shape
        assert np.isclose(observed.min_value, heatmaps.min_value,
                          rtol=0, atol=1e-6)
        assert np.isclose(observed.max_value, heatmaps.max_value,
                          rtol=0, atol=1e-6)
        assert np.array_equal(observed.get_arr(),
                              self.heatmaps_flipped.get_arr())

    def test_segmaps_p_is_1(self):
        aug = self.create_aug(1.0)
        observed = aug.augment_segmentation_maps(self.segmaps)
        assert observed.shape == self.segmaps.shape
        assert np.array_equal(observed.get_arr(),
                              self.segmaps_flipped.get_arr())

    def test_images_p_is_050(self):
        aug = self.create_aug(0.5)

        nb_iterations = 1000
        nb_images_flipped = 0
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_images(self.images)
            if np.array_equal(observed, self.images_flipped):
                nb_images_flipped += 1

        assert np.isclose(nb_images_flipped/nb_iterations,
                          0.5, rtol=0, atol=0.1)

    def test_images_p_is_050__deterministic(self):
        aug = self.create_aug(0.5).to_deterministic()

        nb_iterations = 1000
        nb_images_flipped_det = 0
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_images(self.images)
            if np.array_equal(observed, self.images_flipped):
                nb_images_flipped_det += 1

        assert nb_images_flipped_det in [0, nb_iterations]

    def test_keypoints_p_is_050(self):
        aug = self.create_aug(0.5)

        nb_iterations = 1000
        nb_keypoints_flipped = 0
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_keypoints(self.kpsoi)
            if keypoints_equal(observed, self.kpsoi_flipped):
                nb_keypoints_flipped += 1

        assert np.isclose(nb_keypoints_flipped/nb_iterations,
                          0.5, rtol=0, atol=0.1)

    def test_keypoints_p_is_050__deterministic(self):
        aug = self.create_aug(0.5).to_deterministic()

        nb_iterations = 10
        nb_keypoints_flipped_det = 0
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_keypoints(self.kpsoi)
            if keypoints_equal(observed, self.kpsoi_flipped):
                nb_keypoints_flipped_det += 1

        assert nb_keypoints_flipped_det in [0, nb_iterations]

    def test_polygons_p_is_050(self):
        aug = self.create_aug(0.5)

        nb_iterations = 250
        nb_polygons_flipped = 0
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_polygons(self.psoi)
            if observed[0].polygons[0].exterior_almost_equals(
                    self.psoi_flipped[0].polygons[0]):
                nb_polygons_flipped += 1

        assert np.isclose(nb_polygons_flipped/nb_iterations,
                          0.5, rtol=0, atol=0.2)

    def test_polygons_p_is_050__deterministic(self):
        aug = self.create_aug(0.5).to_deterministic()

        nb_iterations = 10
        nb_polygons_flipped_det = 0
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_polygons(self.psoi)
            if observed[0].polygons[0].exterior_almost_equals(
                    self.psoi_flipped[0].polygons[0]):
                nb_polygons_flipped_det += 1

        assert nb_polygons_flipped_det in [0, nb_iterations]

    def test_list_of_images_p_is_050(self):
        images_multi = [self.image, self.image]
        aug = self.create_aug(0.5)
        nb_iterations = 1000
        nb_flipped_by_pos = [0] * len(images_multi)
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_images(images_multi)
            for i in sm.xrange(len(images_multi)):
                if np.array_equal(observed[i], self.image_flipped):
                    nb_flipped_by_pos[i] += 1

        assert np.allclose(nb_flipped_by_pos,
                           500, rtol=0, atol=100)

    def test_list_of_images_p_is_050__deterministic(self):
        images_multi = [self.image, self.image]
        aug = self.create_aug(0.5).to_deterministic()
        nb_iterations = 10
        nb_flipped_by_pos_det = [0] * len(images_multi)
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_images(images_multi)
            for i in sm.xrange(len(images_multi)):
                if np.array_equal(observed[i], self.image_flipped):
                    nb_flipped_by_pos_det[i] += 1

        for val in nb_flipped_by_pos_det:
            assert val in [0, nb_iterations]

    def test_images_p_is_stochastic_parameter(self):
        aug = self.create_aug(p=iap.Choice([0, 1], p=[0.7, 0.3]))
        seen = [0, 0]
        for _ in sm.xrange(1000):
            observed = aug.augment_image(self.image)
            if np.array_equal(observed, self.image):
                seen[0] += 1
            elif np.array_equal(observed, self.image_flipped):
                seen[1] += 1
            else:
                assert False
        assert np.allclose(seen, [700, 300], rtol=0, atol=75)

    def test_invalid_datatype_for_p_results_in_failure(self):
        with self.assertRaises(Exception):
            _ = self.create_aug(p="test")

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (0, 2),
            (2, 0),
            (0, 2, 0),
            (2, 0, 0),
            (0, 2, 1),
            (2, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = self.create_aug(1.0)

                image_aug = aug(image=image)

                assert image_aug.shape == image.shape

    def test_get_parameters(self):
        aug = self.create_aug(p=0.5)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Binomial)
        assert isinstance(params[0].p, iap.Deterministic)
        assert 0.5 - 1e-4 < params[0].p.value < 0.5 + 1e-4

    def test_other_dtypes_bool(self):
        aug = self.create_aug(1.0)

        image = self.create_arr(True, bool)
        expected = self.create_arr_flipped(True, bool)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == image.dtype.type
        assert np.all(image_aug == expected)

    def test_other_dtypes_uint_int(self):
        aug = self.create_aug(1.0)
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int32", "int64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)
                value = max_value
                image = self.create_arr(value, dtype)
                expected = self.create_arr_flipped(value, dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.name == dtype
                assert np.array_equal(image_aug, expected)

    def test_other_dtypes_float(self):
        aug = self.create_aug(1.0)
        dtypes = ["float16", "float32", "float64", "float128"]
        values = [5000, 1000**2, 1000**3, 1000**4]
        for dtype, value in zip(dtypes, values):
            with self.subTest(dtype=dtype):
                atol = (1e-9 * value
                        if dtype != "float16"
                        else 1e-3 * value)
                image = self.create_arr(value, dtype)
                expected = self.create_arr_flipped(value, dtype)
                image_aug = aug.augment_image(image)
                assert image_aug.dtype.name == dtype
                assert np.allclose(image_aug, expected, atol=atol)


class TestFliplr(_TestFliplrAndFlipudBase, unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.array([[0, 0, 1],
                             [0, 0, 1],
                             [0, 1, 1]], dtype=np.uint8)
        return base_img[:, :, np.newaxis]

    @property
    def image_flipped(self):
        base_img_flipped = np.array([[1, 0, 0],
                                     [1, 0, 0],
                                     [1, 1, 0]], dtype=np.uint8)
        return base_img_flipped[:, :, np.newaxis]

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([
            [0.00, 0.50, 0.75],
            [0.00, 0.50, 0.75],
            [0.75, 0.75, 0.75],
        ])
        return HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def heatmaps_flipped(self):
        heatmaps_arr = np.float32([
            [0.75, 0.50, 0.00],
            [0.75, 0.50, 0.00],
            [0.75, 0.75, 0.75],
        ])
        return HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps(self):
        segmaps_arr = np.int32([
            [0, 1, 2],
            [0, 1, 2],
            [2, 2, 2],
        ])
        return SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps_flipped(self):
        segmaps_arr = np.int32([
            [2, 1, 0],
            [2, 1, 0],
            [2, 2, 2],
        ])
        return SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=0),
               ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        return [ia.KeypointsOnImage(kps, shape=self.image.shape)]

    @property
    def kpsoi_flipped(self):
        kps = [ia.Keypoint(x=3-0, y=0),
               ia.Keypoint(x=3-1, y=1),
               ia.Keypoint(x=3-2, y=2)]
        return [ia.KeypointsOnImage(kps, shape=self.image.shape)]

    @property
    def psoi(self):
        polygons = [ia.Polygon([(0, 0), (2, 0), (2, 2)])]
        return [ia.PolygonsOnImage(polygons, shape=self.image.shape)]

    @property
    def psoi_flipped(self):
        polygons = [ia.Polygon([(3-0, 0), (3-2, 0), (3-2, 2)])]
        return [ia.PolygonsOnImage(polygons, shape=self.image.shape)]

    def create_aug(self, *args, **kwargs):
        return iaa.Fliplr(*args, **kwargs)

    def create_arr(self, value, dtype):
        arr = np.zeros((3, 3), dtype=dtype)
        arr[0, 0] = value
        return arr

    def create_arr_flipped(self, value, dtype):
        arr = np.zeros((3, 3), dtype=dtype)
        arr[0, 2] = value
        return arr


class TestFlipud(_TestFliplrAndFlipudBase, unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.array([[0, 0, 1],
                             [0, 0, 1],
                             [0, 1, 1]], dtype=np.uint8)
        return base_img[:, :, np.newaxis]

    @property
    def image_flipped(self):
        base_img_flipped = np.array([[0, 1, 1],
                                     [0, 0, 1],
                                     [0, 0, 1]], dtype=np.uint8)
        return base_img_flipped[:, :, np.newaxis]

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([
            [0.00, 0.50, 0.75],
            [0.00, 0.50, 0.75],
            [0.75, 0.75, 0.75],
        ])
        return HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def heatmaps_flipped(self):
        heatmaps_arr = np.float32([
            [0.75, 0.75, 0.75],
            [0.00, 0.50, 0.75],
            [0.00, 0.50, 0.75],
        ])
        return HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps(self):
        segmaps_arr = np.int32([
            [0, 1, 2],
            [0, 1, 2],
            [2, 2, 2],
        ])
        return SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps_flipped(self):
        segmaps_arr = np.int32([
            [2, 2, 2],
            [0, 1, 2],
            [0, 1, 2],
        ])
        return SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=0, y=0),
               ia.Keypoint(x=1, y=1),
               ia.Keypoint(x=2, y=2)]
        return [ia.KeypointsOnImage(kps, shape=self.image.shape)]

    @property
    def kpsoi_flipped(self):
        kps = [ia.Keypoint(x=0, y=3-0),
               ia.Keypoint(x=1, y=3-1),
               ia.Keypoint(x=2, y=3-2)]
        return [ia.KeypointsOnImage(kps, shape=self.image.shape)]

    @property
    def psoi(self):
        polygons = [ia.Polygon([(0, 0), (2, 0), (2, 2)])]
        return [ia.PolygonsOnImage(polygons, shape=self.image.shape)]

    @property
    def psoi_flipped(self):
        polygons = [ia.Polygon([(0, 3-0), (2, 3-0), (2, 3-2)])]
        return [ia.PolygonsOnImage(polygons, shape=self.image.shape)]

    def create_aug(self, *args, **kwargs):
        return iaa.Flipud(*args, **kwargs)

    def create_arr(self, value, dtype):
        arr = np.zeros((3, 3), dtype=dtype)
        arr[0, 0] = value
        return arr

    def create_arr_flipped(self, value, dtype):
        arr = np.zeros((3, 3), dtype=dtype)
        arr[2, 0] = value
        return arr


class Test_fliplr(unittest.TestCase):
    def setUp(self):
        reseed()

    @mock.patch("imgaug.augmenters.flip._fliplr_sliced")
    @mock.patch("imgaug.augmenters.flip._fliplr_cv2")
    def test__fliplr_cv2_called_mocked(self, mock_cv2, mock_sliced):
        for dtype in ["uint8", "uint16", "int8", "int16"]:
            mock_cv2.reset_mock()
            mock_sliced.reset_mock()
            arr = np.zeros((1, 1), dtype=dtype)

            _ = fliplib.fliplr(arr)

            mock_cv2.assert_called_once_with(arr)
            assert mock_sliced.call_count == 0

    @mock.patch("imgaug.augmenters.flip._fliplr_sliced")
    @mock.patch("imgaug.augmenters.flip._fliplr_cv2")
    def test__fliplr_sliced_called_mocked(self, mock_cv2, mock_sliced):
        for dtype in ["bool", "uint32", "uint64", "int32", "int64",
                      "float16", "float32", "float64", "float128"]:
            mock_cv2.reset_mock()
            mock_sliced.reset_mock()
            arr = np.zeros((1, 1), dtype=dtype)

            _ = fliplib.fliplr(arr)

            assert mock_cv2.call_count == 0
            mock_sliced.assert_called_once_with(arr)

    def test__fliplr_cv2_2d(self):
        self._test__fliplr_subfunc_n_channels(fliplib._fliplr_cv2, None)

    def test__fliplr_cv2_3d_single_channel(self):
        self._test__fliplr_subfunc_n_channels(fliplib._fliplr_cv2, 1)

    def test__fliplr_cv2_3d_three_channels(self):
        self._test__fliplr_subfunc_n_channels(fliplib._fliplr_cv2, 3)

    def test__fliplr_cv2_3d_four_channels(self):
        self._test__fliplr_subfunc_n_channels(fliplib._fliplr_cv2, 4)

    def test__fliplr_sliced_2d(self):
        self._test__fliplr_subfunc_n_channels(fliplib._fliplr_sliced, None)

    def test__fliplr_sliced_3d_single_channel(self):
        self._test__fliplr_subfunc_n_channels(fliplib._fliplr_sliced, 1)

    def test__fliplr_sliced_3d_three_channels(self):
        self._test__fliplr_subfunc_n_channels(fliplib._fliplr_sliced, 3)

    def test__fliplr_sliced_3d_four_channels(self):
        self._test__fliplr_subfunc_n_channels(fliplib._fliplr_sliced, 4)

    def test__fliplr_sliced_3d_513_channels(self):
        self._test__fliplr_subfunc_n_channels(fliplib._fliplr_sliced, 513)

    @classmethod
    def _test__fliplr_subfunc_n_channels(cls, func, nb_channels):
        arr = np.uint8([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [10, 11, 12, 13]
        ])
        if nb_channels is not None:
            arr = np.tile(arr[..., np.newaxis], (1, 1, nb_channels))
            for c in sm.xrange(nb_channels):
                arr[..., c] += c

        arr_flipped = func(arr)

        expected = np.uint8([
            [3, 2, 1, 0],
            [7, 6, 5, 4],
            [13, 12, 11, 10]
        ])
        if nb_channels is not None:
            expected = np.tile(expected[..., np.newaxis], (1, 1, nb_channels))
            for c in sm.xrange(nb_channels):
                expected[..., c] += c
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == arr.shape
        assert np.array_equal(arr_flipped, expected)

    def test_zero_height_arr_cv2(self):
        arr = np.zeros((0, 4, 1), dtype=np.uint8)
        arr_flipped = fliplib._fliplr_cv2(arr)
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (0, 4, 1)

    def test_zero_width_arr_cv2(self):
        arr = np.zeros((4, 0, 1), dtype=np.uint8)
        arr_flipped = fliplib._fliplr_cv2(arr)
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (4, 0, 1)

    def test_zero_channels_arr_cv2(self):
        arr = np.zeros((4, 1, 0), dtype=np.uint8)
        arr_flipped = fliplib._fliplr_cv2(arr)
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (4, 1, 0)

    def test_513_channels_arr_cv2(self):
        arr = np.zeros((1, 2, 513), dtype=np.uint8)
        arr[:, 0, :] = 0
        arr[:, 1, :] = 255
        arr[0, 0, 0] = 1
        arr[0, 1, 0] = 254
        arr[0, 0, 512] = 2
        arr[0, 1, 512] = 253

        arr_flipped = fliplib._fliplr_cv2(arr)

        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (1, 2, 513)
        assert arr_flipped[0, 1, 0] == 1
        assert arr_flipped[0, 0, 0] == 254
        assert arr_flipped[0, 1, 512] == 2
        assert arr_flipped[0, 0, 512] == 253
        assert np.all(arr_flipped[0, 0, 1:-2] == 255)
        assert np.all(arr_flipped[0, 1, 1:-2] == 0)

    def test_zero_height_arr_sliced(self):
        arr = np.zeros((0, 4, 1), dtype=np.uint8)
        arr_flipped = fliplib._fliplr_sliced(arr)
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (0, 4, 1)

    def test_zero_width_arr_sliced(self):
        arr = np.zeros((4, 0, 1), dtype=np.uint8)
        arr_flipped = fliplib._fliplr_sliced(arr)
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (4, 0, 1)

    def test_zero_channels_arr_sliced(self):
        arr = np.zeros((4, 1, 0), dtype=np.uint8)
        arr_flipped = fliplib._fliplr_sliced(arr)
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (4, 1, 0)

    def test_513_channels_arr_sliced(self):
        arr = np.zeros((1, 2, 513), dtype=np.uint8)
        arr[:, 0, :] = 0
        arr[:, 1, :] = 255
        arr[0, 0, 0] = 1
        arr[0, 1, 0] = 254
        arr[0, 0, 512] = 2
        arr[0, 1, 512] = 253

        arr_flipped = fliplib._fliplr_sliced(arr)

        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (1, 2, 513)
        assert arr_flipped[0, 1, 0] == 1
        assert arr_flipped[0, 0, 0] == 254
        assert arr_flipped[0, 1, 512] == 2
        assert arr_flipped[0, 0, 512] == 253
        assert np.all(arr_flipped[0, 0, 1:-2] == 255)
        assert np.all(arr_flipped[0, 1, 1:-2] == 0)

    def test_bool_faithful(self):
        arr = np.array([[False, False, True]], dtype=bool)
        arr_flipped = fliplib.fliplr(arr)
        expected = np.array([[True, False, False]], dtype=bool)
        assert arr_flipped.dtype.name == "bool"
        assert arr_flipped.shape == (1, 3)
        assert np.array_equal(arr_flipped, expected)

    def test_uint_int_faithful(self):
        dts = ["uint8", "uint16", "uint32", "uint64",
               "int8", "int16", "int32", "int64"]
        for dt in dts:
            with self.subTest(dtype=dt):
                dt = np.dtype(dt)
                minv, center, maxv = iadt.get_value_range_of_dtype(dt)
                center = int(center)
                arr = np.array([[minv, center, maxv]], dtype=dt)

                arr_flipped = fliplib.fliplr(arr)

                expected = np.array([[maxv, center, minv]], dtype=dt)
                assert arr_flipped.dtype.name == dt.name
                assert arr_flipped.shape == (1, 3)
                assert np.array_equal(arr_flipped, expected)

    def test_float_faithful_to_min_max(self):
        dts = ["float16", "float32", "float64", "float128"]
        for dt in dts:
            with self.subTest(dtype=dt):
                dt = np.dtype(dt)
                minv, center, maxv = iadt.get_value_range_of_dtype(dt)
                center = int(center)
                atol = 1e-4 if dt.name == "float16" else 1e-8
                arr = np.array([[minv, center, maxv]], dtype=dt)

                arr_flipped = fliplib.fliplr(arr)

                expected = np.array([[maxv, center, minv]], dtype=dt)
                assert arr_flipped.dtype.name == dt.name
                assert arr_flipped.shape == (1, 3)
                assert np.allclose(arr_flipped, expected, rtol=0, atol=atol)

    def test_float_faithful_to_large_values(self):
        dts = ["float16", "float32", "float64", "float128"]
        values = [
            [0.01, 0.1, 1.0, 10.0**1, 10.0**2],  # float16
            [0.01, 0.1, 1.0, 10.0**1, 10.0**2, 10.0**4, 10.0**6],  # float32
            [0.01, 0.1, 1.0, 10.0**1, 10.0**2, 10.0**6, 10.0**10],  # float64
            [0.01, 0.1, 1.0, 10.0**1, 10.0**2, 10.0**7, 10.0**11],  # float128
        ]
        for dt, values_i in zip(dts, values):
            for value in values_i:
                with self.subTest(dtype=dt, value=value):
                    dt = np.dtype(dt)
                    minv, center, maxv = -value, 0.0, value
                    atol = 1e-4 if dt.name == "float16" else 1e-8
                    arr = np.array([[minv, center, maxv]], dtype=dt)

                    arr_flipped = fliplib.fliplr(arr)

                    expected = np.array([[maxv, center, minv]], dtype=dt)
                    assert arr_flipped.dtype.name == dt.name
                    assert arr_flipped.shape == (1, 3)
                    assert np.allclose(arr_flipped, expected, rtol=0, atol=atol)


class Test_flipud(unittest.TestCase):
    def setUp(self):
        reseed()

    def test__flipud_2d(self):
        self._test__flipud_subfunc_n_channels(fliplib.flipud, None)

    def test__flipud_3d_single_channel(self):
        self._test__flipud_subfunc_n_channels(fliplib.flipud, 1)

    def test__flipud_3d_three_channels(self):
        self._test__flipud_subfunc_n_channels(fliplib.flipud, 3)

    def test__flipud_3d_four_channels(self):
        self._test__flipud_subfunc_n_channels(fliplib.flipud, 4)

    @classmethod
    def _test__flipud_subfunc_n_channels(cls, func, nb_channels):
        arr = np.uint8([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [10, 11, 12, 13]
        ])
        if nb_channels is not None:
            arr = np.tile(arr[..., np.newaxis], (1, 1, nb_channels))
            for c in sm.xrange(nb_channels):
                arr[..., c] += c

        arr_flipped = func(arr)

        expected = np.uint8([
            [10, 11, 12, 13],
            [4, 5, 6, 7],
            [0, 1, 2, 3]
        ])
        if nb_channels is not None:
            expected = np.tile(expected[..., np.newaxis], (1, 1, nb_channels))
            for c in sm.xrange(nb_channels):
                expected[..., c] += c
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == arr.shape
        assert np.array_equal(arr_flipped, expected)

    def test_zero_width_arr(self):
        arr = np.zeros((4, 0, 1), dtype=np.uint8)
        arr_flipped = fliplib.flipud(arr)
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (4, 0, 1)

    def test_zero_height_arr(self):
        arr = np.zeros((0, 4, 1), dtype=np.uint8)
        arr_flipped = fliplib.flipud(arr)
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (0, 4, 1)

    def test_zero_channels_arr(self):
        arr = np.zeros((4, 1, 0), dtype=np.uint8)
        arr_flipped = fliplib.flipud(arr)
        assert arr_flipped.dtype.name == "uint8"
        assert arr_flipped.shape == (4, 1, 0)

    def test_bool_faithful(self):
        arr = np.array([[False], [False], [True]], dtype=bool)
        arr_flipped = fliplib.flipud(arr)
        expected = np.array([[True], [False], [False]], dtype=bool)
        assert arr_flipped.dtype.name == "bool"
        assert arr_flipped.shape == (3, 1)
        assert np.array_equal(arr_flipped, expected)

    def test_uint_int_faithful(self):
        dts = ["uint8", "uint16", "uint32", "uint64",
               "int8", "int16", "int32", "int64"]
        for dt in dts:
            with self.subTest(dtype=dt):
                dt = np.dtype(dt)
                minv, center, maxv = iadt.get_value_range_of_dtype(dt)
                center = int(center)
                arr = np.array([[minv], [center], [maxv]], dtype=dt)

                arr_flipped = fliplib.flipud(arr)

                expected = np.array([[maxv], [center], [minv]], dtype=dt)
                assert arr_flipped.dtype.name == dt.name
                assert arr_flipped.shape == (3, 1)
                assert np.array_equal(arr_flipped, expected)

    def test_float_faithful_to_min_max(self):
        dts = ["float16", "float32", "float64", "float128"]
        for dt in dts:
            with self.subTest(dtype=dt):
                dt = np.dtype(dt)
                minv, center, maxv = iadt.get_value_range_of_dtype(dt)
                center = int(center)
                atol = 1e-4 if dt.name == "float16" else 1e-8
                arr = np.array([[minv], [center], [maxv]], dtype=dt)

                arr_flipped = fliplib.flipud(arr)

                expected = np.array([[maxv], [center], [minv]], dtype=dt)
                assert arr_flipped.dtype.name == dt.name
                assert arr_flipped.shape == (3, 1)
                assert np.allclose(arr_flipped, expected, rtol=0, atol=atol)

    def test_float_faithful_to_large_values(self):
        dts = ["float16", "float32", "float64", "float128"]
        values = [
            [0.01, 0.1, 1.0, 10.0**1, 10.0**2],  # float16
            [0.01, 0.1, 1.0, 10.0**1, 10.0**2, 10.0**4, 10.0**6],  # float32
            [0.01, 0.1, 1.0, 10.0**1, 10.0**2, 10.0**6, 10.0**10],  # float64
            [0.01, 0.1, 1.0, 10.0**1, 10.0**2, 10.0**7, 10.0**11],  # float128
        ]
        for dt, values_i in zip(dts, values):
            for value in values_i:
                with self.subTest(dtype=dt, value=value):
                    dt = np.dtype(dt)
                    minv, center, maxv = -value, 0.0, value
                    atol = 1e-4 if dt.name == "float16" else 1e-8
                    arr = np.array([[minv], [center], [maxv]], dtype=dt)

                    arr_flipped = fliplib.flipud(arr)

                    expected = np.array([[maxv], [center], [minv]], dtype=dt)
                    assert arr_flipped.dtype.name == dt.name
                    assert arr_flipped.shape == (3, 1)
                    assert np.allclose(arr_flipped, expected, rtol=0, atol=atol)
