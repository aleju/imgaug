from __future__ import print_function, division, absolute_import

import time
import warnings
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
from imgaug.testutils import reseed


def main():
    time_start = time.time()

    # test_Batch()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


class TestBatch(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_init(self):
        attr_names = ["images", "heatmaps", "segmentation_maps", "keypoints",
                      "bounding_boxes", "polygons"]
        batch = ia.Batch()
        for attr_name in attr_names:
            assert getattr(batch, "%s_unaug" % (attr_name,)) is None
            assert getattr(batch, "%s_aug" % (attr_name,)) is None
        assert batch.data is None

        # we exploit here that Batch() init does not verify its inputs
        batch = ia.Batch(
            images=0,
            heatmaps=1,
            segmentation_maps=2,
            keypoints=3,
            bounding_boxes=4,
            polygons=5,
            data=6
        )
        for i, attr_name in enumerate(attr_names):
            assert getattr(batch, "%s_unaug" % (attr_name,)) == i
            assert getattr(batch, "%s_aug" % (attr_name,)) is None
        assert batch.data == 6

    def get_images_unaug_normalized(self):
        batch = ia.Batch()
        assert batch.get_images_unaug_normalized() is None

        batch = ia.Batch(images=None)
        assert batch.get_images_unaug_normalized() is None

        arr = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        batch = ia.Batch(images=arr)
        observed = batch.get_images_unaug_normalized()
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4, 3)
        assert observed.dtype.name == "uint8"

        arr = np.zeros((1, 4, 4), dtype=np.uint8)
        batch = ia.Batch(images=arr)
        observed = batch.get_images_unaug_normalized()
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4, 3)
        assert observed.dtype.name == "uint8"

        arr = np.zeros((4, 4), dtype=np.uint8)
        batch = ia.Batch(images=arr)
        observed = batch.get_images_unaug_normalized()
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4, 1)
        assert observed.dtype.name == "uint8"

        batch = ia.Batch(images=[])
        assert isinstance(batch.get_images_unaug_normalized(), list)
        assert len(batch.get_images_unaug_normalized()) == 0

        arr1 = np.zeros((4, 4), dtype=np.uint8)
        arr2 = np.zeros((5, 5, 3), dtype=np.uint8)
        batch = ia.Batch(images=[arr1, arr2])
        observed = batch.get_images_unaug_normalized()
        assert isinstance(observed, list)
        assert len(observed) == 2
        assert ia.is_np_array(observed[0])
        assert ia.is_np_array(observed[1])
        assert observed[0].shape == (4, 4)
        assert observed[1].shape == (5, 5, 3)
        assert observed[0].dtype.name == "uint8"
        assert observed[1].dtype.name == "uint8"

        batch = ia.Batch(images=False)
        with self.assertRaises(ValueError):
            batch.get_images_unaug_normalized()

    def test_get_heatmaps_unaug_normalized(self):
        # ----
        # None
        # ----
        batch = ia.Batch(heatmaps=None)
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert heatmaps_norm is None

        # ----
        # array
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            heatmaps=np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1)
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        batch = ia.Batch(
            images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            heatmaps=np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1)
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        # --> heatmaps for too many images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                heatmaps=np.zeros((2, 1, 1, 1), dtype=np.float32) + 0.1)
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> too few heatmaps
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                heatmaps=np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1)
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> wrong channel number
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                heatmaps=np.zeros((1, 1, 1), dtype=np.float32) + 0.1)
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                heatmaps=np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1)
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # ----
        # single HeatmapsOnImage
        # ----
        batch = ia.Batch(
            images=None,
            heatmaps=ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                shape=(1, 1, 3)))
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        # ----
        # empty iterable
        # ----
        batch = ia.Batch(images=None, heatmaps=[])
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert heatmaps_norm is None

        # ----
        # iterable of arrays
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            heatmaps=[np.zeros((1, 1, 1), dtype=np.float32) + 0.1])
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        batch = ia.Batch(
            images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            heatmaps=[np.zeros((1, 1, 1), dtype=np.float32) + 0.1])
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        # --> heatmaps for too many images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                heatmaps=[
                    np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                    np.zeros((1, 1, 1), dtype=np.float32) + 0.1
                ])
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> too few heatmaps
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                heatmaps=[np.zeros((1, 1, 1), dtype=np.float32) + 0.1])
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                heatmaps=[np.zeros((1, 1, 1), dtype=np.float32) + 0.1])
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # --> wrong number of dimensions
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                heatmaps=[np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1])
            _heatmaps_norm = batch.get_heatmaps_unaug_normalized()

        # ----
        # iterable of HeatmapsOnImage
        # ----
        batch = ia.Batch(
            images=None,
            heatmaps=[ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                shape=(1, 1, 3))])
        heatmaps_norm = batch.get_heatmaps_unaug_normalized()
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

    def test_get_segmentation_maps_unaug_normalized(self):
        # ----
        # None
        # ----
        batch = ia.Batch(segmentation_maps=None)
        segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
        assert segmaps_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("int32"), np.dtype("uint32"), np.dtype(bool)]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                segmentation_maps=np.zeros((1, 1, 1, 1), dtype=dt) + 1)
            segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                segmentation_maps=np.zeros((1, 1, 1, 1), dtype=dt) + 1)
            segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            # --> heatmaps for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    segmentation_maps=np.zeros((2, 1, 1, 1), dtype=dt) + 1)
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> too few heatmaps
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    segmentation_maps=np.zeros((1, 1, 1, 1), dtype=dt) + 1)
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> wrong channel number
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    segmentation_maps=np.zeros((1, 1, 1), dtype=dt) + 1)
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    segmentation_maps=np.zeros((1, 1, 1, 1), dtype=dt) + 1)
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

        # ----
        # single SegmentationMapOnImage
        # ----
        batch = ia.Batch(
            images=None,
            segmentation_maps=ia.SegmentationMapOnImage(
                np.zeros((1, 1, 1), dtype=np.int32) + 1,
                shape=(1, 1, 3),
                nb_classes=2))
        segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
        assert isinstance(segmaps_norm, list)
        assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
        assert np.allclose(segmaps_norm[0].arr[..., 1], 0 + 1)

        # ----
        # empty iterable
        # ----
        batch = ia.Batch(images=None, segmentation_maps=[])
        segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
        assert segmaps_norm is None

        # ----
        # iterable of arrays
        # ----
        for dt in [np.dtype("int32"), np.dtype("uint32"), np.dtype(bool)]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                segmentation_maps=[np.zeros((1, 1, 1), dtype=dt) + 1])
            segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                segmentation_maps=[np.zeros((1, 1, 1), dtype=dt) + 1])
            segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            # --> heatmaps for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    segmentation_maps=[
                        np.zeros((1, 1, 1), dtype=np.int32) + 1,
                        np.zeros((1, 1, 1), dtype=np.int32) + 1
                    ])
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> too few heatmaps
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    segmentation_maps=[
                        np.zeros((1, 1, 1), dtype=np.int32) + 1])
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    segmentation_maps=[
                        np.zeros((1, 1, 1), dtype=np.int32) + 1])
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

            # --> wrong number of dimensions
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    segmentation_maps=[
                        np.zeros((1, 1, 1, 1), dtype=np.int32) + 1])
                _segmaps_norm = batch.get_segmentation_maps_unaug_normalized()

        # ----
        # iterable of SegmentationMapOnImage
        # ----
        batch = ia.Batch(
            images=None,
            segmentation_maps=[ia.SegmentationMapOnImage(
                np.zeros((1, 1, 1), dtype=np.int32) + 1,
                shape=(1, 1, 3),
                nb_classes=2)])
        segmaps_norm = batch.get_segmentation_maps_unaug_normalized()
        assert isinstance(segmaps_norm, list)
        assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
        assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

    def test_get_keypoints_unaug_normalized(self):
        def _assert_single_image_expected(inputs):
            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    keypoints=inputs)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    keypoints=inputs)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8),
                            np.zeros((1, 1, 3), dtype=np.uint8)],
                    keypoints=inputs)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # ----
        # None
        # ----
        batch = ia.Batch(keypoints=None)
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert keypoints_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=np.zeros((1, 1, 2), dtype=dt) + 1)
            keypoints_norm = batch.get_keypoints_unaug_normalized()
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 1
            assert np.allclose(keypoints_norm[0].get_coords_array(), 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                keypoints=np.zeros((1, 5, 2), dtype=dt) + 1)
            keypoints_norm = batch.get_keypoints_unaug_normalized()
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 5
            assert np.allclose(keypoints_norm[0].get_coords_array(), 1)

            # --> keypoints for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    keypoints=np.zeros((2, 1, 2), dtype=dt) + 1)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> too few keypoints
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    keypoints=np.zeros((1, 1, 2), dtype=dt) + 1)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> wrong keypoints shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    keypoints=np.zeros((1, 1, 100), dtype=dt) + 1)
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            _assert_single_image_expected(np.zeros((1, 1, 2), dtype=dt) + 1)

        # ----
        # (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=(1, 2))
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        _assert_single_image_expected((1, 2))

        # ----
        # single Keypoint instance
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=ia.Keypoint(x=1, y=2))
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        _assert_single_image_expected(ia.Keypoint(x=1, y=2))

        # ----
        # single KeypointsOnImage instance
        # ----
        batch = ia.Batch(
            images=None,
            keypoints=ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3))
        )
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        # ----
        # empty iterable
        # ----
        batch = ia.Batch(
            images=None,
            keypoints=[]
        )
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert keypoints_norm is None

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=[np.zeros((1, 2), dtype=dt) + 1])
            keypoints_norm = batch.get_keypoints_unaug_normalized()
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 1
            assert np.allclose(keypoints_norm[0].get_coords_array(), 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                keypoints=[np.zeros((5, 2), dtype=dt) + 1])
            keypoints_norm = batch.get_keypoints_unaug_normalized()
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 5
            assert np.allclose(keypoints_norm[0].get_coords_array(), 1)

            # --> keypoints for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    keypoints=[
                        np.zeros((1, 2), dtype=dt) + 1,
                        np.zeros((1, 2), dtype=dt) + 1
                    ])
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> too few keypoints
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    keypoints=[
                        np.zeros((1, 2), dtype=dt) + 1])
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    keypoints=[
                        np.zeros((1, 2), dtype=dt) + 1])
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

            # --> wrong shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    keypoints=[
                        np.zeros((1, 100), dtype=dt) + 1])
                _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # ----
        # iterable of (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=[(1, 2), (3, 4)])
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 2
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2
        assert keypoints_norm[0].keypoints[1].x == 3
        assert keypoints_norm[0].keypoints[1].y == 4

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=[(1, 2)])
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # ----
        # iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=[ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)])
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 2
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2
        assert keypoints_norm[0].keypoints[1].x == 3
        assert keypoints_norm[0].keypoints[1].y == 4

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=[ia.Keypoint(x=1, y=2)])
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # ----
        # iterable of KeypointsOnImage
        # ----
        batch = ia.Batch(
            images=None,
            keypoints=[
                ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3)),
                ia.KeypointsOnImage([ia.Keypoint(x=3, y=4)], shape=(1, 1, 3)),
            ]
        )
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)

        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        assert isinstance(keypoints_norm[1], ia.KeypointsOnImage)
        assert len(keypoints_norm[1].keypoints) == 1
        assert keypoints_norm[1].keypoints[0].x == 3
        assert keypoints_norm[1].keypoints[0].y == 4

        # ----
        # iterable of empty interables
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=[[]])
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert keypoints_norm is None

        # ----
        # iterable of iterable of (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=[
                [(1, 2), (3, 4)],
                [(5, 6), (7, 8)]
            ]
        )
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 2
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2
        assert keypoints_norm[0].keypoints[1].x == 3
        assert keypoints_norm[0].keypoints[1].y == 4

        assert len(keypoints_norm[1].keypoints) == 2
        assert keypoints_norm[1].keypoints[0].x == 5
        assert keypoints_norm[1].keypoints[0].y == 6
        assert keypoints_norm[1].keypoints[1].x == 7
        assert keypoints_norm[1].keypoints[1].y == 8

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                keypoints=[
                    [(1, 2), (3, 4)],
                    [(5, 6), (7, 8)]
                ]
            )
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=[
                    [(1, 2), (3, 4)],
                    [(5, 6), (7, 8)]
                ]
            )
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # ----
        # iterable of iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            keypoints=[
                [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
                [ia.Keypoint(x=5, y=6), ia.Keypoint(x=7, y=8)]
            ]
        )
        keypoints_norm = batch.get_keypoints_unaug_normalized()
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 2
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2
        assert keypoints_norm[0].keypoints[1].x == 3
        assert keypoints_norm[0].keypoints[1].y == 4

        assert len(keypoints_norm[1].keypoints) == 2
        assert keypoints_norm[1].keypoints[0].x == 5
        assert keypoints_norm[1].keypoints[0].y == 6
        assert keypoints_norm[1].keypoints[1].x == 7
        assert keypoints_norm[1].keypoints[1].y == 8

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                keypoints=[
                    [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
                    [ia.Keypoint(x=5, y=6), ia.Keypoint(x=7, y=8)]
                ]
            )
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                keypoints=[
                    [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
                    [ia.Keypoint(x=5, y=6), ia.Keypoint(x=7, y=8)]
                ]
            )
            _keypoints_norm = batch.get_keypoints_unaug_normalized()

    def test_get_bounding_boxes_unaug_normalized(self):
        def _assert_single_image_expected(inputs):
            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    bounding_boxes=inputs)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    bounding_boxes=inputs)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8),
                            np.zeros((1, 1, 3), dtype=np.uint8)],
                    bounding_boxes=inputs)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # ----
        # None
        # ----
        batch = ia.Batch(bounding_boxes=None)
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert bbs_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=np.zeros((1, 1, 4), dtype=dt) + 1)
            bbs_norm = batch.get_bounding_boxes_unaug_normalized()
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 1
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                bounding_boxes=np.zeros((1, 5, 4), dtype=dt) + 1)
            bbs_norm = batch.get_bounding_boxes_unaug_normalized()
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 5
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            # --> bounding boxes for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    bounding_boxes=np.zeros((2, 1, 4), dtype=dt) + 1)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> too few bounding boxes
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    bounding_boxes=np.zeros((1, 1, 4), dtype=dt) + 1)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> wrong keypoints shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    bounding_boxes=np.zeros((1, 1, 100), dtype=dt) + 1)
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            _assert_single_image_expected(np.zeros((1, 1, 4), dtype=dt) + 1)

        # ----
        # (x1,y1,x2,y2)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=(1, 2, 3, 4))
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        _assert_single_image_expected((1, 4))

        # ----
        # single BoundingBox instance
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=ia.BoundingBox(x1=1, y1=2, x2=3, y2=4))
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        _assert_single_image_expected(ia.BoundingBox(x1=1, y1=2, x2=3, y2=4))

        # ----
        # single BoundingBoxesOnImage instance
        # ----
        batch = ia.Batch(
            images=None,
            bounding_boxes=ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                shape=(1, 1, 3))
        )
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        # ----
        # empty iterable
        # ----
        batch = ia.Batch(
            images=None,
            bounding_boxes=[]
        )
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert bbs_norm is None

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=[np.zeros((1, 4), dtype=dt) + 1])
            bbs_norm = batch.get_bounding_boxes_unaug_normalized()
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 1
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                bounding_boxes=[np.zeros((5, 4), dtype=dt) + 1])
            bbs_norm = batch.get_bounding_boxes_unaug_normalized()
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 5
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            # --> bounding boxes for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    bounding_boxes=[
                        np.zeros((1, 4), dtype=dt) + 1,
                        np.zeros((1, 4), dtype=dt) + 1
                    ])
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> too few bounding boxes
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    bounding_boxes=[
                        np.zeros((1, 4), dtype=dt) + 1])
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    bounding_boxes=[
                        np.zeros((1, 4), dtype=dt) + 1])
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

            # --> wrong shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    bounding_boxes=[
                        np.zeros((1, 100), dtype=dt) + 1])
                _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # ----
        # iterable of (x1,y1,x2,y2)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=[(1, 2, 3, 4), (5, 6, 7, 8)])
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 2
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4
        assert bbs_norm[0].bounding_boxes[1].x1 == 5
        assert bbs_norm[0].bounding_boxes[1].y1 == 6
        assert bbs_norm[0].bounding_boxes[1].x2 == 7
        assert bbs_norm[0].bounding_boxes[1].y2 == 8

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=[(1, 4)])
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # ----
        # iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=[
                ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)
            ])
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 2
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4
        assert bbs_norm[0].bounding_boxes[1].x1 == 5
        assert bbs_norm[0].bounding_boxes[1].y1 == 6
        assert bbs_norm[0].bounding_boxes[1].x2 == 7
        assert bbs_norm[0].bounding_boxes[1].y2 == 8

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=[ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)])
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # ----
        # iterable of BoundingBoxesOnImage
        # ----
        batch = ia.Batch(
            images=None,
            bounding_boxes=[
                ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                    shape=(1, 1, 3)),
                ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                    shape=(1, 1, 3))
            ]
        )
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)

        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        assert isinstance(bbs_norm[1], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[1].bounding_boxes) == 1
        assert bbs_norm[1].bounding_boxes[0].x1 == 5
        assert bbs_norm[1].bounding_boxes[0].y1 == 6
        assert bbs_norm[1].bounding_boxes[0].x2 == 7
        assert bbs_norm[1].bounding_boxes[0].y2 == 8

        # ----
        # iterable of empty interables
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=[[]])
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert bbs_norm is None

        # ----
        # iterable of iterable of (x1,y1,x2,y2)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=[
                [(1, 2, 3, 4)],
                [(5, 6, 7, 8), (9, 10, 11, 12)]
            ]
        )
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        assert len(bbs_norm[1].bounding_boxes) == 2
        assert bbs_norm[1].bounding_boxes[0].x1 == 5
        assert bbs_norm[1].bounding_boxes[0].y1 == 6
        assert bbs_norm[1].bounding_boxes[0].x2 == 7
        assert bbs_norm[1].bounding_boxes[0].y2 == 8

        assert bbs_norm[1].bounding_boxes[1].x1 == 9
        assert bbs_norm[1].bounding_boxes[1].y1 == 10
        assert bbs_norm[1].bounding_boxes[1].x2 == 11
        assert bbs_norm[1].bounding_boxes[1].y2 == 12

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                bounding_boxes=[
                    [(1, 4), (3, 4)],
                    [(5, 6), (7, 8)]
                ]
            )
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=[
                    [(1, 2, 3, 4)],
                    [(5, 6, 7, 8)]
                ]
            )
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # ----
        # iterable of iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            bounding_boxes=[
                [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                 ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                [ia.BoundingBox(x1=9, y1=10, x2=11, y2=12),
                 ia.BoundingBox(x1=13, y1=14, x2=15, y2=16)]
            ]
        )
        bbs_norm = batch.get_bounding_boxes_unaug_normalized()
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 2
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4
        assert bbs_norm[0].bounding_boxes[1].x1 == 5
        assert bbs_norm[0].bounding_boxes[1].y1 == 6
        assert bbs_norm[0].bounding_boxes[1].x2 == 7
        assert bbs_norm[0].bounding_boxes[1].y2 == 8

        assert len(bbs_norm[1].bounding_boxes) == 2
        assert bbs_norm[1].bounding_boxes[0].x1 == 9
        assert bbs_norm[1].bounding_boxes[0].y1 == 10
        assert bbs_norm[1].bounding_boxes[0].x2 == 11
        assert bbs_norm[1].bounding_boxes[0].y2 == 12
        assert bbs_norm[1].bounding_boxes[1].x1 == 13
        assert bbs_norm[1].bounding_boxes[1].y1 == 14
        assert bbs_norm[1].bounding_boxes[1].x2 == 15
        assert bbs_norm[1].bounding_boxes[1].y2 == 16

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                bounding_boxes=[
                    [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                     ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                    [ia.BoundingBox(x1=9, y1=10, x2=11, y2=12),
                     ia.BoundingBox(x1=13, y1=14, x2=15, y2=16)]
                ]
            )
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                bounding_boxes=[
                    [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                     ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                    [ia.BoundingBox(x1=9, y1=10, x2=11, y2=12),
                     ia.BoundingBox(x1=13, y1=14, x2=15, y2=16)]
                ]
            )
            _bbs_norm = batch.get_bounding_boxes_unaug_normalized()

    def test_get_polygons_unaug_normalized(self):
        def _assert_single_image_expected(inputs):
            # --> images None
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=None,
                    polygons=inputs)
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    polygons=inputs)
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8),
                            np.zeros((1, 1, 3), dtype=np.uint8)],
                    polygons=inputs)
                _polygons_norm = batch.get_polygons_unaug_normalized()

        coords1 = [(0, 0), (10, 0), (10, 10)]
        coords2 = [(5, 5), (15, 5), (15, 15)]
        coords3 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        coords4 = [(5, 5), (15, 5), (15, 15), (5, 15)]

        coords1_kps = [ia.Keypoint(x=x, y=y) for x, y in coords1]
        coords2_kps = [ia.Keypoint(x=x, y=y) for x, y in coords2]
        coords3_kps = [ia.Keypoint(x=x, y=y) for x, y in coords3]
        coords4_kps = [ia.Keypoint(x=x, y=y) for x, y in coords4]

        coords1_arr = np.float32(coords1)
        coords2_arr = np.float32(coords2)
        coords3_arr = np.float32(coords3)
        coords4_arr = np.float32(coords4)

        # ----
        # None
        # ----
        batch = ia.Batch(polygons=None)
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert polygons_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=coords1_arr[np.newaxis, np.newaxis, ...].astype(dt))
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 1
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                polygons=np.tile(
                    coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                    (1, 5, 1, 1)
                ))
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 5
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            # --> polygons for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    polygons=np.tile(
                        coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                        (2, 1, 1, 1)
                    ))
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> too few polygons
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    polygons=np.tile(
                        coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                        (1, 1, 1, 1)
                    ))
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> wrong polygons shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    polygons=np.tile(
                        coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                        (1, 1, 1, 10)
                    ))
                _polygons_norm = batch.get_polygons_unaug_normalized()

            _assert_single_image_expected(
                coords1_arr[np.newaxis, np.newaxis, ...].astype(dt))

        # ----
        # single Polygon instance
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=ia.Polygon(coords1))
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        _assert_single_image_expected(ia.Polygon(coords1))

        # ----
        # single PolygonsOnImage instance
        # ----
        batch = ia.Batch(
            images=None,
            polygons=ia.PolygonsOnImage([ia.Polygon(coords1)], shape=(1, 1, 3))
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        # ----
        # empty iterable
        # ----
        batch = ia.Batch(
            images=None,
            polygons=[]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert polygons_norm is None

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[coords1_arr[np.newaxis, ...].astype(dt)])
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 1
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                polygons=[np.tile(
                    coords1_arr[np.newaxis, ...].astype(dt),
                    (5, 1, 1)
                )])
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 5
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            # --> polygons for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    polygons=[coords1_arr[np.newaxis, ...].astype(dt),
                              coords2_arr[np.newaxis, ...].astype(dt)])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> too few polygons
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    polygons=[coords1_arr[np.newaxis, ...].astype(dt)])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> wrong polygons shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    polygons=[np.tile(
                        coords1_arr[np.newaxis, ...].astype(dt),
                        (1, 1, 10)
                    )])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            _assert_single_image_expected(
                [coords1_arr[np.newaxis, ...].astype(dt)]
            )

        # ----
        # iterable of (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=coords1)
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=coords1)
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=coords1_kps)
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=coords1_kps)
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of Polygon
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[ia.Polygon(coords1), ia.Polygon(coords2)])
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        # may only be used for single images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[ia.Polygon(coords1)])
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of PolygonsOnImage
        # ----
        batch = ia.Batch(
            images=None,
            polygons=[
                ia.PolygonsOnImage([ia.Polygon(coords1)], shape=(1, 1, 3)),
                ia.PolygonsOnImage([ia.Polygon(coords2)], shape=(1, 1, 3))
            ]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)

        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        assert isinstance(polygons_norm[1], ia.PolygonsOnImage)
        assert len(polygons_norm[1].polygons) == 1
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords2)

        # ----
        # iterable of empty interables
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[[]])
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert polygons_norm is None

        # ----
        # iterable of iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[[coords1_arr.astype(dt)]])
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 1
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            batch = ia.Batch(
                images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                polygons=[[
                    np.copy(coords1_arr).astype(dt) for _ in sm.xrange(5)
                ]])
            polygons_norm = batch.get_polygons_unaug_normalized()
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 5
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            # --> polygons for too many images
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=[np.zeros((1, 1, 3), dtype=np.uint8)],
                    polygons=[[coords1_arr.astype(dt)],
                              [coords2_arr.astype(dt)]])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> too few polygons
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((2, 1, 1, 3), dtype=np.uint8),
                    polygons=[[coords1_arr.astype(dt)]])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            # --> wrong polygons shape
            with self.assertRaises(AssertionError):
                batch = ia.Batch(
                    images=np.zeros((1, 1, 1, 3), dtype=np.uint8),
                    polygons=[[np.tile(
                        coords1_arr.astype(dt),
                        (1, 1, 10)
                    )]])
                _polygons_norm = batch.get_polygons_unaug_normalized()

            _assert_single_image_expected(
                [[coords1_arr.astype(dt)]]
            )

        # ----
        # iterable of iterable of (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[coords1, coords2]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                polygons=[coords1, coords2]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[coords1, coords2]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[coords1_kps, coords2_kps]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                polygons=[coords1_kps, coords2_kps]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[coords1_kps, coords2_kps]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of iterable of Polygon
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[
                [ia.Polygon(coords1), ia.Polygon(coords2)],
                [ia.Polygon(coords3), ia.Polygon(coords4)]
            ]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert isinstance(polygons_norm[1], ia.PolygonsOnImage)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        assert len(polygons_norm[1].polygons) == 2
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords3)
        assert polygons_norm[1].polygons[1].exterior_almost_equals(coords4)

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                polygons=[
                    [ia.Polygon(coords1), ia.Polygon(coords2)],
                    [ia.Polygon(coords3), ia.Polygon(coords4)]
                ]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[
                    [ia.Polygon(coords1), ia.Polygon(coords2)],
                    [ia.Polygon(coords3), ia.Polygon(coords4)]
                ]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of iterable of empty iterable
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[[[]]])
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert polygons_norm is None

        # ----
        # iterable of iterable of iterable of (x,y)
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[[coords1, coords2], [coords3, coords4]]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords3)
        assert polygons_norm[1].polygons[1].exterior_almost_equals(coords4)

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                polygons=[[coords1, coords2]]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[[coords1, coords2], [coords3]]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # ----
        # iterable of iterable of iterable of Keypoint
        # ----
        batch = ia.Batch(
            images=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)],
            polygons=[[coords1_kps, coords2_kps], [coords3_kps, coords4_kps]]
        )
        polygons_norm = batch.get_polygons_unaug_normalized()
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords3)
        assert polygons_norm[1].polygons[1].exterior_almost_equals(coords4)

        # --> images None
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=None,
                polygons=[[coords1_kps, coords2_kps]]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

        # --> different number of images
        with self.assertRaises(AssertionError):
            batch = ia.Batch(
                images=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)],
                polygons=[[coords1_kps, coords2_kps], [coords3_kps]]
            )
            _polygons_norm = batch.get_polygons_unaug_normalized()

    def test__get_heatmaps_unaug_normalization_type(self):
        batch = ia.Batch(heatmaps=None)
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "None"

        batch = ia.Batch(heatmaps=np.zeros((1, 1, 1, 1), dtype=np.float32))
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "array[float]"

        batch = ia.Batch(heatmaps=ia.HeatmapsOnImage(
            np.zeros((1, 1, 1), dtype=np.float32),
            shape=(1, 1, 1)
        ))
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "HeatmapsOnImage"

        batch = ia.Batch(heatmaps=[])
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "iterable[empty]"

        batch = ia.Batch(heatmaps=[np.zeros((1, 1, 1), dtype=np.float32)])
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "iterable-array[float]"

        batch = ia.Batch(heatmaps=[
            ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                               shape=(1, 1, 1))
        ])
        ntype = batch._get_heatmaps_unaug_normalization_type()
        assert ntype == "iterable-HeatmapsOnImage"

        # --
        # error cases
        # --
        batch = ia.Batch(heatmaps=1)
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        batch = ia.Batch(heatmaps="foo")
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        batch = ia.Batch(heatmaps=np.zeros((1, 1, 1), dtype=np.int32))
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        batch = ia.Batch(heatmaps=[1])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        # wrong class
        batch = ia.Batch(heatmaps=ia.KeypointsOnImage([], shape=(1, 1, 1)))
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        batch = ia.Batch(heatmaps=[[]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

        # list of list of Heatmaps, only list of Heatmaps is max
        batch = ia.Batch(
            heatmaps=[
                [ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                                    shape=(1, 1, 1))]
            ]
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_heatmaps_unaug_normalization_type()

    def test__get_segmentation_maps_unaug_normalization_type(self):
        batch = ia.Batch(segmentation_maps=None)
        ntype = batch._get_segmentation_maps_unaug_normalization_type()
        assert ntype == "None"

        for name, dt in zip(["int", "uint", "bool"],
                            [np.int32, np.uint16, bool]):
            batch = ia.Batch(segmentation_maps=np.zeros((1, 1, 1, 1), dtype=dt))
            ntype = batch._get_segmentation_maps_unaug_normalization_type()
            assert ntype == "array[%s]" % (name,)

        batch = ia.Batch(segmentation_maps=ia.SegmentationMapOnImage(
            np.zeros((1, 1, 1), dtype=np.int32),
            shape=(1, 1, 1),
            nb_classes=1
        ))
        ntype = batch._get_segmentation_maps_unaug_normalization_type()
        assert ntype == "SegmentationMapOnImage"

        batch = ia.Batch(segmentation_maps=[])
        ntype = batch._get_segmentation_maps_unaug_normalization_type()
        assert ntype == "iterable[empty]"

        batch = ia.Batch(
            segmentation_maps=[np.zeros((1, 1, 1), dtype=np.int32)]
        )
        ntype = batch._get_segmentation_maps_unaug_normalization_type()
        assert ntype == "iterable-array[int]"

        batch = ia.Batch(segmentation_maps=[
            ia.SegmentationMapOnImage(np.zeros((1, 1, 1), dtype=np.int32),
                                      shape=(1, 1, 1),
                                      nb_classes=1)
        ])
        ntype = batch._get_segmentation_maps_unaug_normalization_type()
        assert ntype == "iterable-SegmentationMapOnImage"

        # --
        # error cases
        # --
        batch = ia.Batch(segmentation_maps=1)
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

        batch = ia.Batch(segmentation_maps="foo")
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

        batch = ia.Batch(segmentation_maps=[1])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

        # wrong class
        batch = ia.Batch(
            segmentation_maps=ia.KeypointsOnImage([], shape=(1, 1, 1))
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

        batch = ia.Batch(segmentation_maps=[[]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

        # list of list of SegMap, only list of SegMap is max
        batch = ia.Batch(
            segmentation_maps=[
                [ia.SegmentationMapOnImage(
                    np.zeros((1, 1, 1), dtype=np.int32),
                    shape=(1, 1, 1),
                    nb_classes=1)]
            ]
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_segmentation_maps_unaug_normalization_type()

    def test__get_keypoints_unaug_normalization_type(self):
        batch = ia.Batch(keypoints=None)
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "None"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(keypoints=np.zeros((1, 5, 2), dtype=dt))
            ntype = batch._get_keypoints_unaug_normalization_type()
            assert ntype == "array[%s]" % (name,)

        batch = ia.Batch(keypoints=(1, 2))
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "(x,y)"

        batch = ia.Batch(keypoints=ia.Keypoint(x=1, y=2))
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "Keypoint"

        batch = ia.Batch(keypoints=ia.KeypointsOnImage(
            [ia.Keypoint(x=1, y=2)], shape=(1, 1, 3)))
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "KeypointsOnImage"

        batch = ia.Batch(keypoints=[])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(keypoints=[np.zeros((5, 2), dtype=dt)])
            ntype = batch._get_keypoints_unaug_normalization_type()
            assert ntype == "iterable-array[%s]" % (name,)

        batch = ia.Batch(keypoints=[(1, 2)])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-(x,y)"

        batch = ia.Batch(keypoints=[ia.Keypoint(x=1, y=2)])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-Keypoint"

        batch = ia.Batch(keypoints=[
            ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3))])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-KeypointsOnImage"

        batch = ia.Batch(keypoints=[[]])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-iterable[empty]"

        batch = ia.Batch(keypoints=[[(1, 2)]])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-iterable-(x,y)"

        batch = ia.Batch(keypoints=[[ia.Keypoint(x=1, y=2)]])
        ntype = batch._get_keypoints_unaug_normalization_type()
        assert ntype == "iterable-iterable-Keypoint"

        # --
        # error cases
        # --
        batch = ia.Batch(keypoints=1)
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

        batch = ia.Batch(keypoints="foo")
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

        batch = ia.Batch(keypoints=[1])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

        # wrong class
        batch = ia.Batch(
            keypoints=ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                                         shape=(1, 1, 1))
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

        batch = ia.Batch(keypoints=[[[]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

        # list of list of of list of keypoints,
        # only list of list of keypoints is max
        batch = ia.Batch(keypoints=[[[ia.Keypoint(x=1, y=2)]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_keypoints_unaug_normalization_type()

    def test__get_bounding_boxes_unaug_normalization_type(self):
        batch = ia.Batch(bounding_boxes=None)
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "None"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(bounding_boxes=np.zeros((1, 5, 4), dtype=dt))
            ntype = batch._get_bounding_boxes_unaug_normalization_type()
            assert ntype == "array[%s]" % (name,)

        batch = ia.Batch(bounding_boxes=(1, 2, 3, 4))
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "(x1,y1,x2,y2)"

        batch = ia.Batch(bounding_boxes=ia.BoundingBox(x1=1, y1=2, x2=3, y2=4))
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "BoundingBox"

        batch = ia.Batch(bounding_boxes=ia.BoundingBoxesOnImage(
            [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)], shape=(1, 1, 3)))
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "BoundingBoxesOnImage"

        batch = ia.Batch(bounding_boxes=[])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(bounding_boxes=[np.zeros((5, 4), dtype=dt)])
            ntype = batch._get_bounding_boxes_unaug_normalization_type()
            assert ntype == "iterable-array[%s]" % (name,)

        batch = ia.Batch(bounding_boxes=[(1, 2, 3, 4)])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-(x1,y1,x2,y2)"

        batch = ia.Batch(bounding_boxes=[
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-BoundingBox"

        batch = ia.Batch(bounding_boxes=[
            ia.BoundingBoxesOnImage([ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                                     shape=(1, 1, 3))])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-BoundingBoxesOnImage"

        batch = ia.Batch(bounding_boxes=[[]])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-iterable[empty]"

        batch = ia.Batch(bounding_boxes=[[(1, 2, 3, 4)]])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-iterable-(x1,y1,x2,y2)"

        batch = ia.Batch(bounding_boxes=[[ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]])
        ntype = batch._get_bounding_boxes_unaug_normalization_type()
        assert ntype == "iterable-iterable-BoundingBox"

        # --
        # error cases
        # --
        batch = ia.Batch(bounding_boxes=1)
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

        batch = ia.Batch(bounding_boxes="foo")
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

        batch = ia.Batch(bounding_boxes=[1])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

        # wrong class
        batch = ia.Batch(
            bounding_boxes=ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32),
                shape=(1, 1, 1))
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

        batch = ia.Batch(bounding_boxes=[[[]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

        # list of list of of list of bounding boxes,
        # only list of list of bounding boxes is max
        batch = ia.Batch(bounding_boxes=[[[
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_bounding_boxes_unaug_normalization_type()

    def test__get_polygons_unaug_normalization_type(self):
        points = [(0, 0), (10, 0), (10, 10)]

        batch = ia.Batch(polygons=None)
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "None"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(polygons=np.zeros((1, 2, 5, 2), dtype=dt))
            ntype = batch._get_polygons_unaug_normalization_type()
            assert ntype == "array[%s]" % (name,)

        batch = ia.Batch(polygons=ia.Polygon(points))
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "Polygon"

        batch = ia.Batch(polygons=ia.PolygonsOnImage(
            [ia.Polygon(points)], shape=(1, 1, 3)))
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "PolygonsOnImage"

        batch = ia.Batch(polygons=[])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(polygons=[np.zeros((5, 4), dtype=dt)])
            ntype = batch._get_polygons_unaug_normalization_type()
            assert ntype == "iterable-array[%s]" % (name,)

        batch = ia.Batch(polygons=points)
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-(x,y)"

        batch = ia.Batch(polygons=[ia.Keypoint(x=x, y=y) for x, y in points])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-Keypoint"

        batch = ia.Batch(polygons=[
            ia.Polygon(points)])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-Polygon"

        batch = ia.Batch(polygons=[
            ia.PolygonsOnImage([ia.Polygon(points)],
                               shape=(1, 1, 3))])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-PolygonsOnImage"

        batch = ia.Batch(polygons=[[]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            batch = ia.Batch(polygons=[[np.zeros((5, 4), dtype=dt)]])
            ntype = batch._get_polygons_unaug_normalization_type()
            assert ntype == "iterable-iterable-array[%s]" % (name,)

        batch = ia.Batch(polygons=[points])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-(x,y)"

        batch = ia.Batch(polygons=[[
            ia.Keypoint(x=x, y=y) for x, y in points
        ]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-Keypoint"

        batch = ia.Batch(polygons=[[ia.Polygon(points)]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-Polygon"

        batch = ia.Batch(polygons=[[[]]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-iterable[empty]"

        batch = ia.Batch(polygons=[[points]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-iterable-(x,y)"

        batch = ia.Batch(polygons=[[[
            ia.Keypoint(x=x, y=y) for x, y in points
        ]]])
        ntype = batch._get_polygons_unaug_normalization_type()
        assert ntype == "iterable-iterable-iterable-Keypoint"

        # --
        # error cases
        # --
        batch = ia.Batch(polygons=1)
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

        batch = ia.Batch(polygons="foo")
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

        batch = ia.Batch(polygons=[1])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

        # wrong class
        batch = ia.Batch(
            polygons=ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32),
                shape=(1, 1, 1))
        )
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

        batch = ia.Batch(polygons=[[[[]]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

        # list of list of of list of polygons,
        # only list of list of polygons is max
        batch = ia.Batch(polygons=[[[
            ia.Polygon(points)]]])
        with self.assertRaises(AssertionError):
            _ntype = batch._get_polygons_unaug_normalization_type()

    def test__find_first_nonempty(self):
        # None
        observed = ia.Batch._find_first_nonempty(None)
        assert observed[0] is None
        assert observed[1] is True
        assert len(observed[2]) == 0

        # None with parents
        observed = ia.Batch._find_first_nonempty(None, parents=["foo"])
        assert observed[0] is None
        assert observed[1] is True
        assert len(observed[2]) == 1
        assert observed[2][0] == "foo"

        # array
        observed = ia.Batch._find_first_nonempty(np.zeros((4, 4, 3)))
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 0

        # int
        observed = ia.Batch._find_first_nonempty(0)
        assert observed[0] == 0
        assert observed[1] is True
        assert len(observed[2]) == 0

        # str
        observed = ia.Batch._find_first_nonempty("foo")
        assert observed[0] == "foo"
        assert observed[1] is True
        assert len(observed[2]) == 0

        # empty list
        observed = ia.Batch._find_first_nonempty([])
        assert observed[0] is None
        assert observed[1] is False
        assert len(observed[2]) == 0

        # empty list of empty lists
        observed = ia.Batch._find_first_nonempty([[], [], []])
        assert observed[0] is None
        assert observed[1] is False
        assert len(observed[2]) == 1

        # empty list of empty lists of empty lists
        observed = ia.Batch._find_first_nonempty([[], [[]], []])
        assert observed[0] is None
        assert observed[1] is False
        assert len(observed[2]) == 2

        # list of None
        observed = ia.Batch._find_first_nonempty([None, None])
        assert observed[0] is None
        assert observed[1] is True
        assert len(observed[2]) == 1

        # list of array
        observed = ia.Batch._find_first_nonempty([
            np.zeros((4, 4, 3)), np.zeros((5, 5, 3))])
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 1

        # list of list of array
        observed = ia.Batch._find_first_nonempty(
            [[np.zeros((4, 4, 3))], [np.zeros((5, 5, 3))]]
        )
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 2

        # list of tuple of array
        observed = ia.Batch._find_first_nonempty(
            [
                (
                    np.zeros((4, 4, 3)), np.zeros((5, 5, 3))
                ), (
                    np.zeros((6, 6, 3)), np.zeros((7, 7, 3))
                )
            ]
        )
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 2

    def test__nonempty_info_to_type_str(self):
        ntype = ia.Batch._nonempty_info_to_type_str(None, True, [])
        assert ntype == "None"

        ntype = ia.Batch._nonempty_info_to_type_str(None, False, [])
        assert ntype == "iterable[empty]"

        ntype = ia.Batch._nonempty_info_to_type_str(None, False, [[]])
        assert ntype == "iterable-iterable[empty]"

        ntype = ia.Batch._nonempty_info_to_type_str(None, False, [[], []])
        assert ntype == "iterable-iterable-iterable[empty]"

        ntype = ia.Batch._nonempty_info_to_type_str(None, False, [tuple(), []])
        assert ntype == "iterable-iterable-iterable[empty]"

        ntype = ia.Batch._nonempty_info_to_type_str(1, True, [tuple([1, 2])],
                                                    tuple_size=2)
        assert ntype == "(x,y)"

        ntype = ia.Batch._nonempty_info_to_type_str(1, True, [[], tuple([1, 2])],
                                                    tuple_size=2)
        assert ntype == "iterable-(x,y)"

        ntype = ia.Batch._nonempty_info_to_type_str(1, True, [tuple([1, 2, 3, 4])],
                                                    tuple_size=4)
        assert ntype == "(x1,y1,x2,y2)"

        ntype = ia.Batch._nonempty_info_to_type_str(1, True, [[], tuple([1, 2, 3, 4])],
                                                    tuple_size=4)
        assert ntype == "iterable-(x1,y1,x2,y2)"

        with self.assertRaises(AssertionError):
            ntype = ia.Batch._nonempty_info_to_type_str(1, True, [tuple([1, 2, 3])],
                                                        tuple_size=2)
            assert ntype == "(x1,y1,x2,y2)"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.uint8), True, [])
        assert ntype == "array[uint]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.float32), True, [])
        assert ntype == "array[float]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.int32), True, [])
        assert ntype == "array[int]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=bool), True, [])
        assert ntype == "array[bool]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.dtype("complex")), True, [])
        assert ntype == "array[c]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.uint8), True, [[]])
        assert ntype == "iterable-array[uint]"

        ntype = ia.Batch._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.uint8), True, [[], []])
        assert ntype == "iterable-iterable-array[uint]"

        cls_names = ["Keypoint", "KeypointsOnImage",
                     "BoundingBox", "BoundingBoxesOnImage",
                     "Polygon", "PolygonsOnImage",
                     "HeatmapsOnImage", "SegmentationMapOnImage"]
        clss = [
            ia.Keypoint(x=1, y=1),
            ia.KeypointsOnImage([], shape=(1, 1, 3)),
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
            ia.BoundingBoxesOnImage([], shape=(1, 1, 3)),
            ia.Polygon([(1, 1), (1, 2), (2, 2)]),
            ia.PolygonsOnImage([], shape=(1,)),
            ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                               shape=(1, 1, 3)),
            ia.SegmentationMapOnImage(np.zeros((1, 1, 1), dtype=np.int32),
                                      shape=(1, 1, 3), nb_classes=1)
        ]
        for cls_name, cls in zip(cls_names, clss):
            ntype = ia.Batch._nonempty_info_to_type_str(
                cls, True, [])
            assert ntype == cls_name

            ntype = ia.Batch._nonempty_info_to_type_str(
                cls, True, [[]])
            assert ntype == "iterable-%s" % (cls_name,)

            ntype = ia.Batch._nonempty_info_to_type_str(
                cls, True, [[], tuple()])
            assert ntype == "iterable-iterable-%s" % (cls_name,)

    def test_property_warnings(self):
        batch = ia.Batch()
        # self.assertWarns does not exist in py2.7
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            _ = batch.images
            assert len(caught_warnings) == 1
            assert "is deprecated" in str(caught_warnings[-1].message)

            _ = batch.heatmaps
            assert len(caught_warnings) == 2
            assert "is deprecated" in str(caught_warnings[-1].message)

            _ = batch.segmentation_maps
            assert len(caught_warnings) == 3
            assert "is deprecated" in str(caught_warnings[-1].message)

            _ = batch.keypoints
            assert len(caught_warnings) == 4
            assert "is deprecated" in str(caught_warnings[-1].message)

            _ = batch.bounding_boxes
            assert len(caught_warnings) == 5
            assert "is deprecated" in str(caught_warnings[-1].message)

    def test_deepcopy(self):
        batch = ia.Batch()
        observed = batch.deepcopy()
        keys = list(observed.__dict__.keys())
        assert len(keys) >= 12
        for attr_name in keys:
            assert getattr(observed, attr_name) is None

        batch = ia.Batch(images=np.zeros((1, 1, 3), dtype=np.uint8))
        observed = batch.deepcopy()
        for attr_name in observed.__dict__.keys():
            if attr_name != "images_unaug":
                assert getattr(observed, attr_name) is None
        assert ia.is_np_array(observed.images_unaug)

        batch = ia.Batch(
            images=np.zeros((1, 1, 3), dtype=np.uint8),
            heatmaps=[
                ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                                   shape=(4, 4, 3))
            ],
            segmentation_maps=[
                ia.SegmentationMapOnImage(np.zeros((1, 1), dtype=np.int32),
                                          shape=(5, 5, 3),
                                          nb_classes=20)
            ],
            keypoints=[
                ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(6, 6, 3))
            ],
            bounding_boxes=[
                ia.BoundingBoxesOnImage([
                    ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                    shape=(7, 7, 3))
            ],
            polygons=[
                ia.PolygonsOnImage([
                    ia.Polygon([(0, 0), (10, 0), (10, 10)])
                ], shape=(100, 100, 3))
            ],
            data={"test": 123, "foo": "bar", "test2": [1, 2, 3]}
        )
        observed = batch.deepcopy()
        for attr_name in observed.__dict__.keys():
            if "_unaug" not in attr_name and attr_name != "data":
                assert getattr(observed, attr_name) is None

        assert ia.is_np_array(observed.images_unaug)
        assert observed.images_unaug.shape == (1, 1, 3)
        assert isinstance(observed.heatmaps_unaug[0], ia.HeatmapsOnImage)
        assert isinstance(observed.segmentation_maps_unaug[0],
                          ia.SegmentationMapOnImage)
        assert isinstance(observed.keypoints_unaug[0], ia.KeypointsOnImage)
        assert isinstance(observed.bounding_boxes_unaug[0],
                          ia.BoundingBoxesOnImage)
        assert isinstance(observed.polygons_unaug[0], ia.PolygonsOnImage)
        assert isinstance(observed.data, dict)

        assert observed.heatmaps_unaug[0].shape == (4, 4, 3)
        assert observed.segmentation_maps_unaug[0].shape == (5, 5, 3)
        assert observed.keypoints_unaug[0].shape == (6, 6, 3)
        assert observed.bounding_boxes_unaug[0].shape == (7, 7, 3)
        assert observed.polygons_unaug[0].shape == (100, 100, 3)

        assert observed.heatmaps_unaug[0].arr_0to1.shape == (1, 1, 1)
        assert observed.segmentation_maps_unaug[0].arr.shape == (1, 1, 20)
        assert observed.keypoints_unaug[0].keypoints[0].x == 1
        assert observed.keypoints_unaug[0].keypoints[0].y == 2
        assert observed.bounding_boxes_unaug[0].bounding_boxes[0].x1 == 1
        assert observed.bounding_boxes_unaug[0].bounding_boxes[0].y1 == 2
        assert observed.bounding_boxes_unaug[0].bounding_boxes[0].x2 == 3
        assert observed.bounding_boxes_unaug[0].bounding_boxes[0].y2 == 4
        assert observed.polygons_unaug[0].polygons[0].exterior[0, 0] == 0
        assert observed.polygons_unaug[0].polygons[0].exterior[0, 1] == 0
        assert observed.polygons_unaug[0].polygons[0].exterior[1, 0] == 10
        assert observed.polygons_unaug[0].polygons[0].exterior[1, 1] == 0
        assert observed.polygons_unaug[0].polygons[0].exterior[2, 0] == 10
        assert observed.polygons_unaug[0].polygons[0].exterior[2, 1] == 10

        assert observed.data["test"] == 123
        assert observed.data["foo"] == "bar"
        assert observed.data["test2"] == [1, 2, 3]
