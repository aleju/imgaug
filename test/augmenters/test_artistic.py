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

import numpy as np
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import random as iarandom
from imgaug.testutils import reseed, runtest_pickleable_uint8_img
import imgaug.augmenters.color as colorlib
import imgaug.augmenters.artistic as artisticlib


class Test_stylize_cartoon(unittest.TestCase):
    @classmethod
    def _test_integrationtest(cls, size, validate_grads):
        image = ia.quokka_square((size, size))

        image_cartoon = iaa.stylize_cartoon(image, blur_ksize=5,
                                            segmentation_size=2.0)

        image_avg = np.average(image.astype(np.float32), axis=2)
        image_cartoon_avg = np.average(image_cartoon.astype(np.float32), axis=2)

        if validate_grads:
            gradx_image = image_avg[:, :-1] - image_avg[:, 1:]
            grady_image = image_avg[:-1, :] - image_avg[1:, :]
            gradx_cartoon = image_cartoon_avg[:, :-1] - image_cartoon_avg[:, 1:]
            grady_cartoon = image_cartoon_avg[:-1, :] - image_cartoon_avg[1:, :]

            assert (
                (
                    np.average(np.abs(gradx_cartoon))
                    + np.average(np.abs(grady_cartoon))
                )
                <
                (
                    np.average(np.abs(gradx_image))
                    + np.average(np.abs(grady_image))
                )
            )

        # average saturation of cartoon image should be increased
        image_hsv = colorlib.change_colorspace_(np.copy(image),
                                                to_colorspace=iaa.CSPACE_HSV)
        cartoon_hsv = colorlib.change_colorspace_(np.copy(image_cartoon),
                                                  to_colorspace=iaa.CSPACE_HSV)
        assert (
            np.average(cartoon_hsv[:, :, 1])
            > np.average(image_hsv[:, :, 1])
        )

        # as edges are all drawn in completely black, there should be more
        # completely black pixels in the cartoon image
        image_black = np.sum(image_avg <= 0.01)
        cartoon_black = np.sum(image_cartoon_avg <= 0.01)

        assert cartoon_black > image_black

    def test_integrationtest(self):
        self._test_integrationtest(128, True)

    def test_integrationtest_large_image(self):
        # TODO the validation of gradients currently doesn't work well
        #      for the laplacian edge method, but it should
        self._test_integrationtest(400, False)

    @mock.patch("cv2.medianBlur")
    def test_blur_ksize_is_1(self, mock_blur):
        def _side_effect(image, ksize):
            return image

        mock_blur.side_effect = _side_effect
        image = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))

        _ = iaa.stylize_cartoon(image, blur_ksize=1)

        # median blur is called another time in _find_edge_laplacian, but
        # that function is only called if the image is larger
        assert mock_blur.call_count == 0

    @mock.patch("cv2.medianBlur")
    def test_blur_ksize_gt_1(self, mock_blur):
        def _side_effect(image, ksize):
            return image

        mock_blur.side_effect = _side_effect
        image = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))

        _ = iaa.stylize_cartoon(image, blur_ksize=7)

        assert mock_blur.call_count == 1
        assert mock_blur.call_args_list[0][0][1] == 7

    @mock.patch("cv2.pyrMeanShiftFiltering")
    def test_segmentation_size_is_0(self, mock_msf):
        def _side_effect(image, sp, sr, dst):
            dst[...] = image

        mock_msf.side_effect = _side_effect
        image = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))

        _ = iaa.stylize_cartoon(image, segmentation_size=0.0)

        assert mock_msf.call_count == 0

    @mock.patch("cv2.pyrMeanShiftFiltering")
    def test_segmentation_size_gt_0(self, mock_msf):
        def _side_effect(image, sp, sr, dst):
            dst[...] = image

        mock_msf.side_effect = _side_effect
        image = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))

        _ = iaa.stylize_cartoon(image, segmentation_size=0.5)

        assert mock_msf.call_count == 1
        assert np.allclose(mock_msf.call_args_list[0][1]["sp"], 10*0.5)
        assert np.allclose(mock_msf.call_args_list[0][1]["sr"], 20*0.5)

    @mock.patch("imgaug.augmenters.artistic._suppress_edge_blobs")
    def test_suppress_edges_true(self, mock_seb):
        image = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))
        mock_seb.return_value = np.copy(image[..., 0])

        _ = iaa.stylize_cartoon(image, suppress_edges=True)

        assert mock_seb.call_count == 2

    @mock.patch("imgaug.augmenters.artistic._suppress_edge_blobs")
    def test_suppress_edges_false(self, mock_seb):
        image = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))

        _ = iaa.stylize_cartoon(image, suppress_edges=False)

        assert mock_seb.call_count == 0

    @mock.patch("imgaug.augmenters.artistic._find_edges_laplacian")
    def test_large_image(self, mock_fel):
        def _side_effect_fel(image, edge_multiplier, from_colorspace):
            return image[..., 0]

        mock_fel.side_effect = _side_effect_fel
        image = np.zeros((10, 401, 3), dtype=np.uint8)

        _ = iaa.stylize_cartoon(image, segmentation_size=0)

        assert mock_fel.call_count == 1


class Test__saturate(unittest.TestCase):
    def _get_avg_saturation(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return np.average(hsv[..., 1])

    def test_saturation_is_1(self):
        image = np.array([
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
            [100, 110, 120],
            [10, 10, 10],
            [100, 0, 0],
            [0, 100, 0],
            [0, 0, 100]
        ], dtype=np.uint8).reshape((1, 8, 3))

        observed_1 = artisticlib._saturate(image, 1.0, colorlib.CSPACE_RGB)
        observed_2 = artisticlib._saturate(image, 2.0, colorlib.CSPACE_RGB)

        sat_img = self._get_avg_saturation(image)
        sat_1 = self._get_avg_saturation(observed_1)
        sat_2 = self._get_avg_saturation(observed_2)
        assert sat_img < sat_2
        assert sat_1 < sat_2


class TestCartoon(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___defaults(self):
        aug = iaa.Cartoon()
        assert aug.blur_ksize.a.value == 1
        assert aug.blur_ksize.b.value == 5
        assert np.isclose(aug.segmentation_size.a.value, 0.8)
        assert np.isclose(aug.segmentation_size.b.value, 1.2)
        assert np.isclose(aug.edge_prevalence.a.value, 0.9)
        assert np.isclose(aug.edge_prevalence.b.value, 1.1)
        assert np.isclose(aug.saturation.a.value, 1.5)
        assert np.isclose(aug.saturation.b.value, 2.5)
        assert aug.from_colorspace == iaa.CSPACE_RGB

    def test_draw_samples(self):
        mock_batch = mock.Mock()
        mock_batch.nb_rows = 50
        aug = iaa.Cartoon()
        rs = iarandom.RNG(0)

        samples = aug._draw_samples(mock_batch, rs)

        assert len(np.unique(np.round(samples[0]*100, decimals=0))) > 1
        assert len(np.unique(np.round(samples[1]*100, decimals=0))) > 1
        assert len(np.unique(np.round(samples[2]*100, decimals=0))) > 1
        assert len(np.unique(np.round(samples[3]*100, decimals=0))) > 1

    @mock.patch("imgaug.augmenters.artistic.stylize_cartoon")
    def test_call_of_stylize_cartoon(self, mock_sc):
        image = np.arange(4*4*3).astype(np.uint8).reshape((4, 4, 3))
        aug = iaa.Cartoon()

        mock_sc.return_value = np.copy(image)

        _ = aug(images=[image, image])

        assert mock_sc.call_count == 2

    def test_get_parameters(self):
        aug = iaa.Cartoon()
        params = aug.get_parameters()
        assert params[0] is aug.blur_ksize
        assert params[1] is aug.segmentation_size
        assert params[2] is aug.saturation
        assert params[3] is aug.edge_prevalence
        assert params[4] == iaa.CSPACE_RGB

    def test_pickleable(self):
        aug = iaa.Cartoon(seed=1)
        runtest_pickleable_uint8_img(aug, iterations=6)
