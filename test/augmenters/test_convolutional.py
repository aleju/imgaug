from __future__ import print_function, division, absolute_import

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

from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.testutils import reseed


# TODO add tests for EdgeDetect
# TODO add tests for DirectedEdgeDetect


# TODO add test for keypoints once their handling was improved in Convolve
class TestConvolve(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def img(self):
        return np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.uint8)

    def test_matrix_is_none(self):
        aug = iaa.Convolve(matrix=None)
        observed = aug.augment_image(self.img)
        assert np.array_equal(observed, self.img)

    def test_matrix_is_lambda_none(self):
        aug = iaa.Convolve(
            matrix=lambda _img, nb_channels, random_state: [None])
        observed = aug.augment_image(self.img)
        assert np.array_equal(observed, self.img)

    def test_matrix_is_1x1_identity(self):
        # matrix is [[1]]
        aug = iaa.Convolve(matrix=np.float32([[1]]))
        observed = aug.augment_image(self.img)
        assert np.array_equal(observed, self.img)

    def test_matrix_is_lambda_1x1_identity(self):
        aug = iaa.Convolve(
            matrix=lambda _img, nb_channels, random_state: np.float32([[1]]))
        observed = aug.augment_image(self.img)
        assert np.array_equal(observed, self.img)

    def test_matrix_is_3x3_identity(self):
        m = np.float32([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        aug = iaa.Convolve(matrix=m)
        observed = aug.augment_image(self.img)
        assert np.array_equal(observed, self.img)

    def test_matrix_is_lambda_3x3_identity(self):
        m = np.float32([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
        observed = aug.augment_image(self.img)
        assert np.array_equal(observed, self.img)

    def test_matrix_is_3x3_two_in_center(self):
        m = np.float32([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0]
        ])
        aug = iaa.Convolve(matrix=m)
        observed = aug.augment_image(self.img)
        assert np.array_equal(observed, 2*self.img)

    def test_matrix_is_lambda_3x3_two_in_center(self):
        m = np.float32([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0]
        ])
        aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
        observed = aug.augment_image(self.img)
        assert np.array_equal(observed, 2*self.img)

    def test_matrix_is_3x3_two_in_center_3_channels(self):
        m = np.float32([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0]
        ])
        aug = iaa.Convolve(matrix=m)
        img3 = np.tile(self.img[..., np.newaxis], (1, 1, 3))  # 3 channels
        observed = aug.augment_image(img3)
        assert np.array_equal(observed, 2*img3)

    def test_matrix_is_lambda_3x3_two_in_center_3_channels(self):
        m = np.float32([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0]
        ])
        aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
        img3 = np.tile(self.img[..., np.newaxis], (1, 1, 3))  # 3 channels
        observed = aug.augment_image(img3)
        assert np.array_equal(observed, 2*img3)

    def test_matrix_is_3x3_with_multiple_nonzero_values(self):
        m = np.float32([
            [0, -1, 0],
            [0, 10, 0],
            [0, 0, 0]
        ])
        expected = np.uint8([
            [10*1+(-1)*4, 10*2+(-1)*5, 10*3+(-1)*6],
            [10*4+(-1)*1, 10*5+(-1)*2, 10*6+(-1)*3],
            [10*7+(-1)*4, 10*8+(-1)*5, 10*9+(-1)*6]
        ])

        aug = iaa.Convolve(matrix=m)
        observed = aug.augment_image(self.img)
        assert np.array_equal(observed, expected)

    def test_matrix_is_lambda_3x3_with_multiple_nonzero_values(self):
        m = np.float32([
            [0, -1, 0],
            [0, 10, 0],
            [0, 0, 0]
        ])
        expected = np.uint8([
            [10*1+(-1)*4, 10*2+(-1)*5, 10*3+(-1)*6],
            [10*4+(-1)*1, 10*5+(-1)*2, 10*6+(-1)*3],
            [10*7+(-1)*4, 10*8+(-1)*5, 10*9+(-1)*6]
        ])

        aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
        observed = aug.augment_image(self.img)
        assert np.array_equal(observed, expected)

    def test_lambda_with_changing_matrices(self):
        # changing matrices when using callable
        expected = []
        for i in sm.xrange(5):
            expected.append(self.img * i)

        aug = iaa.Convolve(
            matrix=lambda _img, nb_channels, random_state:
                np.float32([[random_state.randint(0, 5)]])
        )
        seen = [False] * 5
        for _ in sm.xrange(200):
            observed = aug.augment_image(self.img)
            found = False
            for i, expected_i in enumerate(expected):
                if np.array_equal(observed, expected_i):
                    seen[i] = True
                    found = True
                    break
            assert found
            if all(seen):
                break
        assert np.all(seen)

    def test_matrix_has_bad_datatype(self):
        # don't use assertRaisesRegex, because it doesnt exist in 2.7
        got_exception = False
        try:
            _aug = iaa.Convolve(matrix=False)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_get_parameters(self):
        matrix = np.int32([[1]])
        aug = iaa.Convolve(matrix=matrix)
        params = aug.get_parameters()
        assert np.array_equal(params[0], matrix)
        assert params[1] == "constant"

    def test_other_dtypes_bool_identity_matrix(self):
        identity_matrix = np.int64([[1]])
        aug = iaa.Convolve(matrix=identity_matrix)

        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True
        image_aug = aug.augment_image(image)
        assert image.dtype.type == np.bool_
        assert np.all(image_aug == image)

    def test_other_dtypes_uint_int_identity_matrix(self):
        identity_matrix = np.int64([[1]])
        aug = iaa.Convolve(matrix=identity_matrix)

        for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = 100
            image_aug = aug.augment_image(image)
            assert image.dtype.type == dtype
            assert np.all(image_aug == image)

    def test_other_dtypes_float_identity_matrix(self):
        identity_matrix = np.int64([[1]])
        aug = iaa.Convolve(matrix=identity_matrix)

        for dtype in [np.float16, np.float32, np.float64]:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = 100.0
            image_aug = aug.augment_image(image)
            assert image.dtype.type == dtype
            assert np.allclose(image_aug, image)

    def test_other_dtypes_bool_non_identity_matrix_with_small_values(self):
        matrix = np.float64([
            [0, 0.6, 0],
            [0, 0.4, 0],
            [0,   0, 0]
        ])
        aug = iaa.Convolve(matrix=matrix)

        image = np.zeros((3, 3), dtype=bool)
        image[1, 1] = True
        image[2, 1] = True
        expected = np.zeros((3, 3), dtype=bool)
        expected[0, 1] = True
        expected[2, 1] = True
        image_aug = aug.augment_image(image)
        assert image.dtype.type == np.bool_
        assert np.all(image_aug == expected)

    def test_other_dtypes_uint_int_non_identity_matrix_with_small_values(self):
        matrix = np.float64([
            [0, 0.5, 0],
            [0, 0.5, 0],
            [0,   0, 0]
        ])
        aug = iaa.Convolve(matrix=matrix)

        for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = 100
            image[2, 1] = 100
            image_aug = aug.augment_image(image)

            expected = np.zeros((3, 3), dtype=dtype)
            expected[0, 1] = int(np.round(100 * 0.5))
            expected[1, 1] = int(np.round(100 * 0.5))
            expected[2, 1] = int(np.round(100 * 0.5 + 100 * 0.5))

            diff = np.abs(image_aug.astype(np.int64) - expected.astype(np.int64))
            assert image_aug.dtype.type == dtype
            assert np.max(diff) <= 2

    def test_other_dtypes_float_non_identity_matrix_with_small_values(self):
        matrix = np.float64([
            [0, 0.5, 0],
            [0, 0.5, 0],
            [0,   0, 0]
        ])
        aug = iaa.Convolve(matrix=matrix)

        for dtype in [np.float16, np.float32, np.float64]:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = 100.0
            image[2, 1] = 100.0
            image_aug = aug.augment_image(image)

            expected = np.zeros((3, 3), dtype=dtype)
            expected[0, 1] = 100 * 0.5
            expected[1, 1] = 100 * 0.5
            expected[2, 1] = 100 * 0.5 + 100 * 0.5

            diff = np.abs(
                image_aug.astype(np.float128) - expected.astype(np.float128))
            assert image_aug.dtype.type == dtype
            assert np.max(diff) < 1.0

    def test_other_dtypes_uint_int_non_identity_matrix_with_large_values(self):
        matrix = np.float64([
            [0, 0.5, 0],
            [0, 0.5, 0],
            [0,   0, 0]
        ])
        aug = iaa.Convolve(matrix=matrix)

        for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            _min_value, center_value, max_value = \
                iadt.get_value_range_of_dtype(dtype)

            value = int(center_value + 0.4 * max_value)

            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image[2, 1] = value
            image_aug = aug.augment_image(image)

            expected = np.zeros((3, 3), dtype=dtype)
            expected[0, 1] = int(np.round(value * 0.5))
            expected[1, 1] = int(np.round(value * 0.5))
            expected[2, 1] = int(np.round(value * 0.5 + value * 0.5))

            diff = np.abs(image_aug.astype(np.int64) - expected.astype(np.int64))
            assert image_aug.dtype.type == dtype
            assert np.max(diff) <= 2

    def test_other_dtypes_float_non_identity_matrix_with_large_values(self):
        matrix = np.float64([
            [0, 0.5, 0],
            [0, 0.5, 0],
            [0,   0, 0]
        ])
        aug = iaa.Convolve(matrix=matrix)

        for dtype, value in zip([np.float16, np.float32, np.float64],
                                [5000, 1000*1000, 1000*1000*1000]):
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image[2, 1] = value
            image_aug = aug.augment_image(image)

            expected = np.zeros((3, 3), dtype=dtype)
            expected[0, 1] = value * 0.5
            expected[1, 1] = value * 0.5
            expected[2, 1] = value * 0.5 + value * 0.5

            diff = np.abs(
                image_aug.astype(np.float128) - expected.astype(np.float128))
            assert image_aug.dtype.type == dtype
            assert np.max(diff) < 1.0

    def test_failure_on_invalid_dtypes(self):
        # don't use assertRaisesRegex, because it doesnt exist in 2.7
        identity_matrix = np.int64([[1]])
        aug = iaa.Convolve(matrix=identity_matrix)
        for dt in [np.uint32, np.uint64, np.int32, np.int64]:
            got_exception = False
            try:
                _ = aug.augment_image(np.zeros((1, 1), dtype=dt))
            except Exception as exc:
                assert "forbidden dtype" in str(exc)
                got_exception = True
            assert got_exception


class TestSharpen(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _compute_sharpened_base_img(cls, lightness, m):
        img = np.zeros((3, 3), dtype=np.float32)
        k = 1
        # note that cv2 uses reflection padding by default
        img[0, 0] = (
            (m[1, 1] + lightness)/k * 10
            + 4 * (m[0, 0]/k) * 10
            + 4 * (m[2, 2]/k) * 20
        )
        img[0, 2] = img[0, 0]
        img[2, 0] = img[0, 0]
        img[2, 2] = img[0, 0]
        img[0, 1] = (
            (m[1, 1] + lightness)/k * 10
            + 6 * (m[0, 1]/k) * 10
            + 2 * (m[2, 2]/k) * 20
        )
        img[1, 0] = img[0, 1]
        img[1, 2] = img[0, 1]
        img[2, 1] = img[0, 1]
        img[1, 1] = (
            (m[1, 1] + lightness)/k * 20
            + 8 * (m[0, 1]/k) * 10
        )

        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    @property
    def base_img(self):
        base_img = [[10, 10, 10],
                    [10, 20, 10],
                    [10, 10, 10]]
        base_img = np.uint8(base_img)
        return base_img

    @property
    def base_img_sharpened(self):
        return self._compute_sharpened_base_img(1, self.m)

    @property
    def m(self):
        return np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]], dtype=np.float32)

    @property
    def m_noop(self):
        return np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.float32)

    def test_alpha_zero(self):
        aug = iaa.Sharpen(alpha=0, lightness=1)
        observed = aug.augment_image(self.base_img)
        expected = self.base_img
        assert np.allclose(observed, expected)

    def test_alpha_one(self):
        aug = iaa.Sharpen(alpha=1.0, lightness=1)
        observed = aug.augment_image(self.base_img)
        expected = self.base_img_sharpened
        assert np.allclose(observed, expected)

    def test_alpha_050(self):
        aug = iaa.Sharpen(alpha=0.5, lightness=1)
        observed = aug.augment_image(self.base_img)
        expected = self._compute_sharpened_base_img(
            0.5*1, 0.5 * self.m_noop + 0.5 * self.m)
        assert np.allclose(observed, expected.astype(np.uint8))

    def test_alpha_075(self):
        aug = iaa.Sharpen(alpha=0.75, lightness=1)
        observed = aug.augment_image(self.base_img)
        expected = self._compute_sharpened_base_img(
            0.75*1, 0.25 * self.m_noop + 0.75 * self.m)
        assert np.allclose(observed, expected)

    def test_alpha_is_stochastic_parameter(self):
        aug = iaa.Sharpen(alpha=iap.Choice([0.5, 1.0]), lightness=1)
        observed = aug.augment_image(self.base_img)
        expected1 = self._compute_sharpened_base_img(
            0.5*1, 0.5 * self.m_noop + 0.5 * self.m)
        expected2 = self._compute_sharpened_base_img(
            1.0*1, 0.0 * self.m_noop + 1.0 * self.m)
        assert (
            np.allclose(observed, expected1)
            or np.allclose(observed, expected2)
        )

    def test_failure_if_alpha_has_bad_datatype(self):
        # don't use assertRaisesRegex, because it doesnt exist in 2.7
        got_exception = False
        try:
            _ = iaa.Sharpen(alpha="test", lightness=1)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_alpha_1_lightness_2(self):
        aug = iaa.Sharpen(alpha=1.0, lightness=2)
        observed = aug.augment_image(self.base_img)
        expected = self._compute_sharpened_base_img(1.0*2, self.m)
        assert np.allclose(observed, expected)

    def test_alpha_1_lightness_3(self):
        aug = iaa.Sharpen(alpha=1.0, lightness=3)
        observed = aug.augment_image(self.base_img)
        expected = self._compute_sharpened_base_img(1.0*3, self.m)
        assert np.allclose(observed, expected)

    def test_alpha_1_lightness_is_stochastic_parameter(self):
        aug = iaa.Sharpen(alpha=1.0, lightness=iap.Choice([1.0, 1.5]))
        observed = aug.augment_image(self.base_img)
        expected1 = self._compute_sharpened_base_img(1.0*1.0, self.m)
        expected2 = self._compute_sharpened_base_img(1.0*1.5, self.m)
        assert (
            np.allclose(observed, expected1)
            or np.allclose(observed, expected2)
        )

    def test_failure_if_lightness_has_bad_datatype(self):
        # don't use assertRaisesRegex, because it doesnt exist in 2.7
        got_exception = False
        try:
            _ = iaa.Sharpen(alpha=1.0, lightness="test")
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    # this part doesnt really work so far due to nonlinearities resulting from clipping to uint8
    """
    # alpha range
    aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=1)
    base_img = np.copy(base_img)
    base_img_sharpened_min = _compute_sharpened_base_img(0.0*1, 1.0 * m_noop + 0.0 * m)
    base_img_sharpened_max = _compute_sharpened_base_img(1.0*1, 0.0 * m_noop + 1.0 * m)
    #distance_max = np.average(np.abs(base_img_sharpened.astype(np.float32) - base_img.astype(np.float32)))
    distance_max = np.average(np.abs(base_img_sharpened_max - base_img_sharpened_min))
    nb_iterations = 250
    distances = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        distance = np.average(np.abs(observed.astype(np.float32) - base_img_sharpened_max.astype(np.float32))) / distance_max
        distances.append(distance)

    print(distances)
    print(min(distances), np.average(distances), max(distances))
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

    # lightness range
    aug = iaa.Sharpen(alpha=1.0, lightness=(0.5, 2.0))
    base_img = np.copy(base_img)
    base_img_sharpened = _compute_sharpened_base_img(1.0*2.0, m)
    distance_max = np.average(np.abs(base_img_sharpened.astype(np.int32) - base_img.astype(np.int32)))
    nb_iterations = 250
    distances = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        distance = np.average(np.abs(observed.astype(np.int32) - base_img.astype(np.int32))) / distance_max
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
    """


class TestEmboss(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _compute_embossed_base_img(cls, img, alpha, strength):
        img = np.copy(img)
        base_img_embossed = np.zeros((3, 3), dtype=np.float32)

        m = np.float32([[-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        strength_matrix = strength * np.float32([
            [-1, -1, 0],
            [-1, 0, 1],
            [0, 1, 1]
        ])
        ms = m + strength_matrix

        for i in range(base_img_embossed.shape[0]):
            for j in range(base_img_embossed.shape[1]):
                for u in range(ms.shape[0]):
                    for v in range(ms.shape[1]):
                        weight = ms[u, v]
                        inputs_i = abs(i + (u - (ms.shape[0]-1)//2))
                        inputs_j = abs(j + (v - (ms.shape[1]-1)//2))
                        if inputs_i >= img.shape[0]:
                            diff = inputs_i - (img.shape[0]-1)
                            inputs_i = img.shape[0] - 1 - diff
                        if inputs_j >= img.shape[1]:
                            diff = inputs_j - (img.shape[1]-1)
                            inputs_j = img.shape[1] - 1 - diff
                        inputs = img[inputs_i, inputs_j]
                        base_img_embossed[i, j] += inputs * weight

        return np.clip(
            (1-alpha) * img
            + alpha * base_img_embossed,
            0,
            255
        ).astype(np.uint8)

    @classmethod
    def _allclose(cls, a, b):
        return np.max(
            a.astype(np.float32)
            - b.astype(np.float32)
        ) <= 2.1

    @property
    def base_img(self):
        return np.array([[10, 10, 10],
                         [10, 20, 10],
                         [10, 10, 15]], dtype=np.uint8)

    def test_alpha_0_strength_1(self):
        aug = iaa.Emboss(alpha=0, strength=1)
        observed = aug.augment_image(self.base_img)
        expected = self.base_img
        assert self._allclose(observed, expected)

    def test_alpha_1_strength_1(self):
        aug = iaa.Emboss(alpha=1.0, strength=1)
        observed = aug.augment_image(self.base_img)
        expected = self._compute_embossed_base_img(
            self.base_img, alpha=1.0, strength=1)
        assert self._allclose(observed, expected)

    def test_alpha_050_strength_1(self):
        aug = iaa.Emboss(alpha=0.5, strength=1)
        observed = aug.augment_image(self.base_img)
        expected = self._compute_embossed_base_img(
            self.base_img, alpha=0.5, strength=1)
        assert self._allclose(observed, expected.astype(np.uint8))

    def test_alpha_075_strength_1(self):
        aug = iaa.Emboss(alpha=0.75, strength=1)
        observed = aug.augment_image(self.base_img)
        expected = self._compute_embossed_base_img(
            self.base_img, alpha=0.75, strength=1)
        assert self._allclose(observed, expected)

    def test_alpha_stochastic_parameter_strength_1(self):
        aug = iaa.Emboss(alpha=iap.Choice([0.5, 1.0]), strength=1)
        observed = aug.augment_image(self.base_img)
        expected1 = self._compute_embossed_base_img(
            self.base_img, alpha=0.5, strength=1)
        expected2 = self._compute_embossed_base_img(
            self.base_img, alpha=1.0, strength=1)
        assert (
            self._allclose(observed, expected1)
            or self._allclose(observed, expected2)
        )

    def test_failure_on_invalid_datatype_for_alpha(self):
        # don't use assertRaisesRegex, because it doesnt exist in 2.7
        got_exception = False
        try:
            _ = iaa.Emboss(alpha="test", strength=1)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_alpha_1_strength_2(self):
        aug = iaa.Emboss(alpha=1.0, strength=2)
        observed = aug.augment_image(self.base_img)
        expected = self._compute_embossed_base_img(
            self.base_img, alpha=1.0, strength=2)
        assert self._allclose(observed, expected)

    def test_alpha_1_strength_3(self):
        aug = iaa.Emboss(alpha=1.0, strength=3)
        observed = aug.augment_image(self.base_img)
        expected = self._compute_embossed_base_img(
            self.base_img, alpha=1.0, strength=3)
        assert self._allclose(observed, expected)

    def test_alpha_1_strength_6(self):
        aug = iaa.Emboss(alpha=1.0, strength=6)
        observed = aug.augment_image(self.base_img)
        expected = self._compute_embossed_base_img(
            self.base_img, alpha=1.0, strength=6)
        assert self._allclose(observed, expected)

    def test_alpha_1_strength_stochastic_parameter(self):
        aug = iaa.Emboss(alpha=1.0, strength=iap.Choice([1.0, 2.5]))
        observed = aug.augment_image(self.base_img)
        expected1 = self._compute_embossed_base_img(
            self.base_img, alpha=1.0, strength=1.0)
        expected2 = self._compute_embossed_base_img(
            self.base_img, alpha=1.0, strength=2.5)
        assert (
            self._allclose(observed, expected1)
            or self._allclose(observed, expected2)
        )

    def test_failure_on_invalid_datatype_for_strength(self):
        # don't use assertRaisesRegex, because it doesnt exist in 2.7
        got_exception = False
        try:
            _ = iaa.Emboss(alpha=1.0, strength="test")
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception
