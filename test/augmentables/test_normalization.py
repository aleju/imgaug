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
import imgaug.augmentables.normalization as normalization
from imgaug.testutils import reseed


def main():
    time_start = time.time()

    # test_Batch()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


class TestNormalization(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_invert_normalize_images(self):
        assert normalization.invert_normalize_images(None, None) is None

        arr = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        arr_old = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        observed = normalization.invert_normalize_images(arr, arr_old)
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4, 3)
        assert observed.dtype.name == "uint8"

        arr = np.zeros((1, 4, 4, 1), dtype=np.uint8)
        arr_old = np.zeros((4, 4), dtype=np.uint8)
        observed = normalization.invert_normalize_images(arr, arr_old)
        assert ia.is_np_array(observed)
        assert observed.shape == (4, 4)
        assert observed.dtype.name == "uint8"

        arr = np.zeros((1, 4, 4, 1), dtype=np.uint8)
        arr_old = np.zeros((1, 4, 4), dtype=np.uint8)
        observed = normalization.invert_normalize_images(arr, arr_old)
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4)
        assert observed.dtype.name == "uint8"

        images = []
        images_old = []
        observed = normalization.invert_normalize_images(images, images_old)
        assert isinstance(observed, list)
        assert len(observed) == 0

        arr1 = np.zeros((4, 4, 1), dtype=np.uint8)
        arr2 = np.zeros((5, 5, 3), dtype=np.uint8)
        arr1_old = np.zeros((4, 4), dtype=np.uint8)
        arr2_old = np.zeros((5, 5, 3), dtype=np.uint8)
        observed = normalization.invert_normalize_images([arr1, arr2],
                                                         [arr1_old, arr2_old])
        assert isinstance(observed, list)
        assert len(observed) == 2
        assert ia.is_np_array(observed[0])
        assert ia.is_np_array(observed[1])
        assert observed[0].shape == (4, 4)
        assert observed[1].shape == (5, 5, 3)
        assert observed[0].dtype.name == "uint8"
        assert observed[1].dtype.name == "uint8"

        with self.assertRaises(ValueError):
            normalization.invert_normalize_images(False, False)

    def test_invert_normalize_heatmaps(self):
        def _norm_and_invert(heatmaps, images):
            return normalization.invert_normalize_heatmaps(
                normalization.normalize_heatmaps(heatmaps, shapes=images),
                heatmaps
            )

        # ----
        # None
        # ----
        observed = normalization.invert_normalize_heatmaps(None, None)
        assert observed is None

        # ----
        # array
        # ----
        for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                       np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
            before = np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1
            after = _norm_and_invert(before, images=images)
            assert ia.is_np_array(after)
            assert after.shape == (1, 1, 1, 1)
            assert after.dtype.name == "float32"
            assert np.allclose(after, before)

        # ----
        # single HeatmapsOnImage
        # ----
        before = ia.HeatmapsOnImage(
                    np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                    shape=(1, 1, 3))
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, ia.HeatmapsOnImage)
        assert after.shape == before.shape
        assert np.allclose(after.arr_0to1, before.arr_0to1)

        # ----
        # empty iterable
        # ----
        before = []
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, list)
        assert len(after) == 0

        # ----
        # iterable of arrays
        # ----
        for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                       np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
            before = [np.zeros((1, 1, 1), dtype=np.float32) + 0.1]
            after = _norm_and_invert(before, images=images)
            assert isinstance(after, list)
            assert len(after) == 1
            assert after[0].shape == (1, 1, 1)
            assert after[0].dtype.name == "float32"
            assert np.allclose(after[0], before[0])

        # ----
        # iterable of HeatmapsOnImage
        # ----
        before = [ia.HeatmapsOnImage(
                    np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                    shape=(1, 1, 3))]
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, list)
        assert isinstance(after[0], ia.HeatmapsOnImage)
        assert after[0].shape == before[0].shape
        assert np.allclose(after[0].arr_0to1, before[0].arr_0to1)

    def test_invert_normalize_segmentation_maps(self):
        def _norm_and_invert(segmaps, images):
            return normalization.invert_normalize_segmentation_maps(
                normalization.normalize_segmentation_maps(
                    segmaps, shapes=images),
                segmaps
            )

        # ----
        # None
        # ----
        observed = normalization.invert_normalize_segmentation_maps(None, None)
        assert observed is None

        # ----
        # array
        # ----
        for dt in [np.dtype("int32"), np.dtype("uint32"), np.dtype(bool)]:
            for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                           np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
                before = np.ones((1, 1, 1), dtype=dt)
                after = _norm_and_invert(before, images=images)
                assert ia.is_np_array(after)
                assert after.shape == (1, 1, 1)
                assert after.dtype.name == dt.name
                assert np.array_equal(after, before)

        # ----
        # single SegmentationMapOnImage
        # ----
        before = ia.SegmentationMapOnImage(
                     np.zeros((1, 1), dtype=np.int32) + 1,
                     shape=(1, 1, 3),
                     nb_classes=2)
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, ia.SegmentationMapOnImage)
        assert after.shape == before.shape
        assert np.array_equal(after.arr, before.arr)

        # ----
        # empty iterable
        # ----
        before = []
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, list)
        assert len(after) == 0

        # ----
        # iterable of arrays
        # ----
        for dt in [np.dtype("int32"), np.dtype("uint32"), np.dtype(bool)]:
            for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                           np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
                before = [np.ones((1, 1), dtype=dt)]
                after = _norm_and_invert(before, images=images)
                assert isinstance(after, list)
                assert len(after) == 1
                assert after[0].shape == (1, 1)
                assert after[0].dtype.name == dt.name
                assert np.array_equal(after[0], before[0])

        # ----
        # iterable of SegmentationMapOnImage
        # ----
        before = [ia.SegmentationMapOnImage(
                    np.zeros((1, 1), dtype=np.int32) + 1,
                    shape=(1, 1, 3),
                    nb_classes=2)]
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, list)
        assert isinstance(after[0], ia.SegmentationMapOnImage)
        assert after[0].shape == before[0].shape
        assert np.allclose(after[0].arr, before[0].arr)

    def test_invert_normalize_keypoints(self):
        def _norm_and_invert(kps, images):
            return normalization.invert_normalize_keypoints(
                normalization.normalize_keypoints(
                    kps, shapes=images),
                kps
            )

        # ----
        # None
        # ----
        observed = normalization.invert_normalize_keypoints(None, None)
        assert observed is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                           np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
                before = np.zeros((1, 1, 2), dtype=dt) + 1
                after = _norm_and_invert(before, images=images)
                assert ia.is_np_array(after)
                assert after.shape == (1, 1, 2)
                assert after.dtype.name == dt.name
                assert np.allclose(after, 1)

        # ----
        # (x,y)
        # ----
        before = (1, 2)
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, tuple)
        assert after == (1, 2)

        # ----
        # single Keypoint instance
        # ----
        before = ia.Keypoint(x=1, y=2)
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, ia.Keypoint)
        assert after.x == 1
        assert after.y == 2

        # ----
        # single KeypointsOnImage instance
        # ----
        before = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3))
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, ia.KeypointsOnImage)
        assert len(after.keypoints) == 1
        assert after.keypoints[0].x == 1
        assert after.keypoints[0].y == 2
        assert after.shape == (1, 1, 3)

        # ----
        # empty iterable
        # ----
        before = []
        after = _norm_and_invert(before, images=None)
        assert after == []

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                           np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
                before = np.zeros((1, 1, 2), dtype=dt) + 1
                after = _norm_and_invert(before, images=images)
                assert ia.is_np_array(after)
                assert after.shape == (1, 1, 2)
                assert after.dtype.name == dt.name
                assert np.allclose(after, 1)

        # ----
        # iterable of (x,y)
        # ----
        before = [(1, 2), (3, 4)]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert after == [(1, 2), (3, 4)]

        # ----
        # iterable of Keypoint
        # ----
        before = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == 2
        assert isinstance(after[0], ia.Keypoint)
        assert isinstance(after[1], ia.Keypoint)
        assert after[0].x == 1
        assert after[0].y == 2
        assert after[1].x == 3
        assert after[1].y == 4

        # ----
        # iterable of KeypointsOnImage
        # ----
        before = [
            ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3)),
            ia.KeypointsOnImage([ia.Keypoint(x=3, y=4)], shape=(1, 1, 3)),
        ]
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, list)
        assert len(after) == 2
        assert isinstance(after[0], ia.KeypointsOnImage)
        assert isinstance(after[1], ia.KeypointsOnImage)
        assert after[0].keypoints[0].x == 1
        assert after[0].keypoints[0].y == 2
        assert after[1].keypoints[0].x == 3
        assert after[1].keypoints[0].y == 4

        # ----
        # iterable of empty interables
        # ----
        before = [[]]
        after = _norm_and_invert(before, [np.zeros((1, 1, 3), dtype=np.uint8)])
        assert after == [[]]

        # ----
        # iterable of iterable of (x,y)
        # ----
        before = [
            [(1, 2), (3, 4)],
            [(5, 6), (7, 8)]
        ]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8),
                                         np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == 2
        assert isinstance(after[0], list)
        assert isinstance(after[1], list)
        assert after[0][0][0] == 1
        assert after[0][0][1] == 2
        assert after[0][1][0] == 3
        assert after[0][1][1] == 4
        assert after[1][0][0] == 5
        assert after[1][0][1] == 6
        assert after[1][1][0] == 7
        assert after[1][1][1] == 8

        # ----
        # iterable of iterable of Keypoint
        # ----
        before = [
            [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
            [ia.Keypoint(x=5, y=6), ia.Keypoint(x=7, y=8)]
        ]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8),
                                         np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == 2
        assert isinstance(after[0], list)
        assert isinstance(after[1], list)
        assert after[0][0].x == 1
        assert after[0][0].y == 2
        assert after[0][1].x == 3
        assert after[0][1].y == 4
        assert after[1][0].x == 5
        assert after[1][0].y == 6
        assert after[1][1].x == 7
        assert after[1][1].y == 8

    def test_invert_normalize_bounding_boxes(self):
        def _norm_and_invert(bbs, images):
            return normalization.invert_normalize_bounding_boxes(
                normalization.normalize_bounding_boxes(
                    bbs, shapes=images),
                bbs
            )

        # ----
        # None
        # ----
        observed = normalization.invert_normalize_bounding_boxes(None, None)
        assert observed is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                           np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
                before = np.zeros((1, 1, 4), dtype=dt) + 1
                after = _norm_and_invert(before, images=images)
                assert ia.is_np_array(after)
                assert after.shape == (1, 1, 4)
                assert after.dtype.name == dt.name
                assert np.allclose(after, 1)

        # ----
        # (x1,y1,x2,y2)
        # ----
        before = (1, 2, 3, 4)
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, tuple)
        assert after == (1, 2, 3, 4)

        # ----
        # single BoundingBox instance
        # ----
        before = ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, ia.BoundingBox)
        assert after.x1 == 1
        assert after.y1 == 2
        assert after.x2 == 3
        assert after.y2 == 4

        # ----
        # single BoundingBoxesOnImage instance
        # ----
        before = ia.BoundingBoxesOnImage(
                     [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                     shape=(1, 1, 3))
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, ia.BoundingBoxesOnImage)
        assert len(after.bounding_boxes) == 1
        assert after.bounding_boxes[0].x1 == 1
        assert after.bounding_boxes[0].y1 == 2
        assert after.bounding_boxes[0].x2 == 3
        assert after.bounding_boxes[0].y2 == 4
        assert after.shape == (1, 1, 3)

        # ----
        # empty iterable
        # ----
        before = []
        after = _norm_and_invert(before, images=None)
        assert after == []

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                           np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
                before = [np.zeros((1, 4), dtype=dt) + 1]
                after = _norm_and_invert(before, images=images)
                assert isinstance(after, list)
                assert len(after) == 1
                assert ia.is_np_array(after[0])
                assert after[0].shape == (1, 4)
                assert after[0].dtype.name == dt.name
                assert np.allclose(after[0], 1)

        # ----
        # iterable of (x1,y1,x2,y2)
        # ----
        before = [(1, 2, 3, 4), (5, 6, 7, 8)]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert after == [(1, 2, 3, 4), (5, 6, 7, 8)]

        # ----
        # iterable of BoundingBox
        # ----
        before = [
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
            ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)
        ]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == 2
        assert isinstance(after[0], ia.BoundingBox)
        assert isinstance(after[1], ia.BoundingBox)
        assert after[0].x1 == 1
        assert after[0].y1 == 2
        assert after[0].x2 == 3
        assert after[0].y2 == 4
        assert after[1].x1 == 5
        assert after[1].y1 == 6
        assert after[1].x2 == 7
        assert after[1].y2 == 8

        # ----
        # iterable of BoundingBoxesOnImage
        # ----
        before = [
            ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                shape=(1, 1, 3)),
            ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                shape=(1, 1, 3))
        ]
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, list)
        assert len(after) == 2
        assert isinstance(after[0], ia.BoundingBoxesOnImage)
        assert isinstance(after[1], ia.BoundingBoxesOnImage)
        assert isinstance(after[0].bounding_boxes[0], ia.BoundingBox)
        assert isinstance(after[1].bounding_boxes[0], ia.BoundingBox)
        assert after[0].bounding_boxes[0].x1 == 1
        assert after[0].bounding_boxes[0].y1 == 2
        assert after[0].bounding_boxes[0].x2 == 3
        assert after[0].bounding_boxes[0].y2 == 4
        assert after[1].bounding_boxes[0].x1 == 5
        assert after[1].bounding_boxes[0].y1 == 6
        assert after[1].bounding_boxes[0].x2 == 7
        assert after[1].bounding_boxes[0].y2 == 8
        assert after[0].shape == (1, 1, 3)
        assert after[1].shape == (1, 1, 3)

        # ----
        # iterable of empty interables
        # ----
        before = [[]]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert after == [[]]

        # ----
        # iterable of iterable of (x1,y1,x2,y2)
        # ----
        before = [
            [(1, 2, 3, 4)],
            [(5, 6, 7, 8), (9, 10, 11, 12)]
        ]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8),
                                         np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert after == [
            [(1, 2, 3, 4)],
            [(5, 6, 7, 8), (9, 10, 11, 12)]
        ]

        # ----
        # iterable of iterable of Keypoint
        # ----
        before = [
            [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
             ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
            [ia.BoundingBox(x1=9, y1=10, x2=11, y2=12),
             ia.BoundingBox(x1=13, y1=14, x2=15, y2=16)]
        ]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8),
                                         np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert isinstance(after[0], list)
        assert isinstance(after[1], list)
        assert len(after[0]) == 2
        assert len(after[1]) == 2
        assert after[0][0].x1 == 1
        assert after[0][0].y1 == 2
        assert after[0][0].x2 == 3
        assert after[0][0].y2 == 4
        assert after[0][1].x1 == 5
        assert after[0][1].y1 == 6
        assert after[0][1].x2 == 7
        assert after[0][1].y2 == 8
        assert after[1][0].x1 == 9
        assert after[1][0].y1 == 10
        assert after[1][0].x2 == 11
        assert after[1][0].y2 == 12
        assert after[1][1].x1 == 13
        assert after[1][1].y1 == 14
        assert after[1][1].x2 == 15
        assert after[1][1].y2 == 16

    def test_invert_normalize_polygons(self):
        def _norm_and_invert(polys, images):
            return normalization.invert_normalize_polygons(
                normalization.normalize_polygons(
                    polys, shapes=images),
                polys
            )

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
        observed = normalization.invert_normalize_polygons(None, None)
        assert observed is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                           np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
                before = coords1_arr[np.newaxis, np.newaxis, ...].astype(dt)
                after = _norm_and_invert(before, images=images)
                assert ia.is_np_array(after)
                assert after.shape == (1, 1, 3, 2)
                assert after.dtype.name == dt.name
                assert np.allclose(after,
                                   coords1_arr[np.newaxis, np.newaxis, ...])

                before = np.tile(
                    coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                    (1, 5, 1, 1)
                )
                after = _norm_and_invert(before, images=images)
                assert ia.is_np_array(after)
                assert after.shape == (1, 5, 3, 2)
                assert after.dtype.name == dt.name
                assert np.allclose(after[0],
                                   coords1_arr[np.newaxis, ...])

        # ----
        # single Polygon instance
        # ----
        before = ia.Polygon(coords1)
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, ia.Polygon)
        assert after.exterior_almost_equals(coords1)

        # ----
        # single PolygonsOnImage instance
        # ----
        before = ia.PolygonsOnImage([ia.Polygon(coords1)], shape=(1, 1, 3))
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, ia.PolygonsOnImage)
        assert len(after.polygons) == 1
        assert after.polygons[0].exterior_almost_equals(coords1)
        assert after.shape == (1, 1, 3)

        # ----
        # empty iterable
        # ----
        before = []
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, list)
        assert after == []

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                           np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
                before = [coords1_arr[np.newaxis, ...].astype(dt)]
                after = _norm_and_invert(before, images=images)
                assert isinstance(after, list)
                assert len(after) == 1
                assert ia.is_np_array(after[0])
                assert after[0].shape == (1, 3, 2)
                assert after[0].dtype.name == dt.name
                assert np.allclose(after[0], coords1_arr[np.newaxis, ...])

                before = [np.tile(
                    coords1_arr[np.newaxis, ...].astype(dt),
                    (5, 1, 1)
                )]
                after = _norm_and_invert(before, images=images)
                assert isinstance(after, list)
                assert len(after) == 1
                assert ia.is_np_array(after[0])
                assert after[0].shape == (5, 3, 2)
                assert after[0].dtype.name == dt.name
                assert np.allclose(after[0][0], coords1_arr)

        # ----
        # iterable of (x,y)
        # ----
        before = coords1
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert after == coords1

        # ----
        # iterable of Keypoint
        # ----
        before = coords1_kps
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == len(coords1_kps)
        assert all([kp_after.x == kp_before.x and kp_after.y == kp_before.y
                    for kp_after, kp_before in zip(after, coords1_kps)])

        # ----
        # iterable of Polygon
        # ----
        before = [ia.Polygon(coords1), ia.Polygon(coords2)]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == 2
        assert after[0].exterior_almost_equals(coords1)
        assert after[1].exterior_almost_equals(coords2)

        # ----
        # iterable of PolygonsOnImage
        # ----
        before = [
            ia.PolygonsOnImage([ia.Polygon(coords1)], shape=(1, 1, 3)),
            ia.PolygonsOnImage([ia.Polygon(coords2)], shape=(2, 1, 3))
        ]
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, list)
        assert len(after) == 2
        assert isinstance(after[0], ia.PolygonsOnImage)
        assert isinstance(after[1], ia.PolygonsOnImage)
        assert after[0].polygons[0].exterior_almost_equals(coords1)
        assert after[1].polygons[0].exterior_almost_equals(coords2)
        assert after[0].shape == (1, 1, 3)
        assert after[1].shape == (2, 1, 3)

        # ----
        # iterable of empty interables
        # ----
        before = [[]]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert after == [[]]

        # ----
        # iterable of iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                           np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
                before = [[coords1_arr.astype(dt)]]
                after = _norm_and_invert(before, images=images)
                assert isinstance(after, list)
                assert len(after) == 1
                assert isinstance(after[0], list)
                assert len(after[0]) == 1
                assert ia.is_np_array(after[0][0])
                assert after[0][0].shape == (3, 2)
                assert after[0][0].dtype.name == dt.name
                assert np.allclose(after[0][0], coords1_arr)

                before = [[coords1_arr.astype(dt) for _ in sm.xrange(5)]]
                after = _norm_and_invert(before, images=images)
                assert isinstance(after, list)
                assert len(after) == 1
                assert isinstance(after[0], list)
                assert len(after[0]) == 5
                assert ia.is_np_array(after[0][0])
                assert after[0][0].shape == (3, 2)
                assert after[0][0].dtype.name == dt.name
                assert np.allclose(after[0][0], coords1_arr)

        # ----
        # iterable of iterable of (x,y)
        # ----
        before = [coords1, coords2]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == 2
        assert after[0] == coords1
        assert after[1] == coords2

        # ----
        # iterable of iterable of Keypoint
        # ----
        before = [coords1_kps, coords2_kps]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == 2
        assert len(after[0]) == len(coords1_kps)
        assert len(after[1]) == len(coords2_kps)
        assert all([kp_after.x == kp_before.x and kp_after.y == kp_before.y
                    for kp_after, kp_before in zip(after[0], coords1_kps)])
        assert all([kp_after.x == kp_before.x and kp_after.y == kp_before.y
                    for kp_after, kp_before in zip(after[1], coords2_kps)])

        # ----
        # iterable of iterable of Polygon
        # ----
        before = [
            [ia.Polygon(coords1), ia.Polygon(coords2)],
            [ia.Polygon(coords3), ia.Polygon(coords4)]
        ]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8),
                                         np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert isinstance(after[0], list)
        assert isinstance(after[1], list)
        assert len(after[0]) == 2
        assert len(after[1]) == 2
        assert after[0][0].exterior_almost_equals(coords1)
        assert after[0][1].exterior_almost_equals(coords2)
        assert after[1][0].exterior_almost_equals(coords3)
        assert after[1][1].exterior_almost_equals(coords4)

        # ----
        # iterable of iterable of empty iterable
        # ----
        before = [[[]]]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert after == [[[]]]

        # ----
        # iterable of iterable of iterable of (x,y)
        # ----
        before = [[coords1, coords2], [coords3, coords4]]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8),
                                         np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == 2
        assert len(after[0]) == 2
        assert len(after[1]) == 2
        assert after[0][0] == coords1
        assert after[0][1] == coords2
        assert after[1][0] == coords3
        assert after[1][1] == coords4

        # ----
        # iterable of iterable of iterable of Keypoint
        # ----
        before = [[coords1_kps, coords2_kps], [coords3_kps, coords4_kps]]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8),
                                         np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == 2
        assert len(after[0]) == 2
        assert len(after[1]) == 2
        assert all([kp_after.x == kp_before.x and kp_after.y == kp_before.y
                    for kp_after, kp_before in zip(after[0][0], coords1_kps)])
        assert all([kp_after.x == kp_before.x and kp_after.y == kp_before.y
                    for kp_after, kp_before in zip(after[0][1], coords2_kps)])
        assert all([kp_after.x == kp_before.x and kp_after.y == kp_before.y
                    for kp_after, kp_before in zip(after[1][0], coords3_kps)])
        assert all([kp_after.x == kp_before.x and kp_after.y == kp_before.y
                    for kp_after, kp_before in zip(after[1][1], coords4_kps)])

    # The underlying normalization functions are mostly identical for
    # LineStrings and Polygons, hence we run only a few tests for LineStrings
    # here. Most of the code was already tested for Polygons.
    def test_invert_normalize_line_strings(self):
        def _norm_and_invert(line_strings, images):
            return normalization.invert_normalize_line_strings(
                normalization.normalize_line_strings(
                    line_strings, shapes=images),
                line_strings
            )

        coords1 = [(0, 0), (10, 0), (10, 10)]
        coords2 = [(5, 5), (15, 5), (15, 15)]
        coords3 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        coords4 = [(5, 5), (15, 5), (15, 15), (5, 15)]

        coords1_arr = np.float32(coords1)

        # ----
        # None
        # ----
        observed = normalization.invert_normalize_line_strings(None, None)
        assert observed is None

        # ----
        # single LineString instance
        # ----
        before = ia.LineString(coords1)
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, ia.LineString)
        assert np.allclose(after.coords, coords1)

        # ----
        # single LineStringsOnImage instance
        # ----
        before = ia.LineStringsOnImage([ia.LineString(coords1)], shape=(1, 1, 3))
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, ia.LineStringsOnImage)
        assert len(after.line_strings) == 1
        assert np.allclose(after.line_strings[0].coords, coords1)
        assert after.shape == (1, 1, 3)

        # ----
        # iterable of LineStringsOnImage
        # ----
        before = [
            ia.LineStringsOnImage([ia.LineString(coords1)], shape=(1, 1, 3)),
            ia.LineStringsOnImage([ia.LineString(coords2)], shape=(2, 1, 3))
        ]
        after = _norm_and_invert(before, images=None)
        assert isinstance(after, list)
        assert len(after) == 2
        assert isinstance(after[0], ia.LineStringsOnImage)
        assert isinstance(after[1], ia.LineStringsOnImage)
        assert np.allclose(after[0].line_strings[0].coords, coords1)
        assert np.allclose(after[1].line_strings[0].coords, coords2)
        assert after[0].shape == (1, 1, 3)
        assert after[1].shape == (2, 1, 3)

        # ----
        # iterable of iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            for images in [[np.zeros((1, 1, 3), dtype=np.uint8)],
                           np.zeros((1, 1, 1, 3), dtype=np.uint8)]:
                before = [[coords1_arr.astype(dt)]]
                after = _norm_and_invert(before, images=images)
                assert isinstance(after, list)
                assert len(after) == 1
                assert isinstance(after[0], list)
                assert len(after[0]) == 1
                assert ia.is_np_array(after[0][0])
                assert after[0][0].shape == (3, 2)
                assert after[0][0].dtype.name == dt.name
                assert np.allclose(after[0][0], coords1_arr)

                before = [[coords1_arr.astype(dt) for _ in sm.xrange(5)]]
                after = _norm_and_invert(before, images=images)
                assert isinstance(after, list)
                assert len(after) == 1
                assert isinstance(after[0], list)
                assert len(after[0]) == 5
                assert ia.is_np_array(after[0][0])
                assert after[0][0].shape == (3, 2)
                assert after[0][0].dtype.name == dt.name
                assert np.allclose(after[0][0], coords1_arr)

        # ----
        # iterable of iterable of LineString
        # ----
        before = [
            [ia.LineString(coords1), ia.LineString(coords2)],
            [ia.LineString(coords3), ia.LineString(coords4)]
        ]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8),
                                         np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert isinstance(after[0], list)
        assert isinstance(after[1], list)
        assert len(after[0]) == 2
        assert len(after[1]) == 2
        assert np.allclose(after[0][0].coords, coords1)
        assert np.allclose(after[0][1].coords, coords2)
        assert np.allclose(after[1][0].coords, coords3)
        assert np.allclose(after[1][1].coords, coords4)

        # ----
        # iterable of iterable of iterable of (x,y)
        # ----
        before = [[coords1, coords2], [coords3, coords4]]
        after = _norm_and_invert(before,
                                 images=[np.zeros((1, 1, 3), dtype=np.uint8),
                                         np.zeros((1, 1, 3), dtype=np.uint8)])
        assert isinstance(after, list)
        assert len(after) == 2
        assert len(after[0]) == 2
        assert len(after[1]) == 2
        assert after[0][0] == coords1
        assert after[0][1] == coords2
        assert after[1][0] == coords3
        assert after[1][1] == coords4

    def test_normalize_images(self):
        assert normalization.normalize_images(None) is None

        arr = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        observed = normalization.normalize_images(arr)
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4, 3)
        assert observed.dtype.name == "uint8"

        arr = np.zeros((1, 4, 4), dtype=np.uint8)
        observed = normalization.normalize_images(arr)
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4, 1)
        assert observed.dtype.name == "uint8"

        arr = np.zeros((4, 4), dtype=np.uint8)
        observed = normalization.normalize_images(arr)
        assert ia.is_np_array(observed)
        assert observed.shape == (1, 4, 4, 1)
        assert observed.dtype.name == "uint8"

        observed = normalization.normalize_images([])
        assert isinstance(observed, list)
        assert len(observed) == 0

        arr1 = np.zeros((4, 4), dtype=np.uint8)
        arr2 = np.zeros((5, 5, 3), dtype=np.uint8)
        observed = normalization.normalize_images([arr1, arr2])
        assert isinstance(observed, list)
        assert len(observed) == 2
        assert ia.is_np_array(observed[0])
        assert ia.is_np_array(observed[1])
        assert observed[0].shape == (4, 4, 1)
        assert observed[1].shape == (5, 5, 3)
        assert observed[0].dtype.name == "uint8"
        assert observed[1].dtype.name == "uint8"

        with self.assertRaises(ValueError):
            normalization.normalize_images(False)

    def test_normalize_heatmaps(self):
        # ----
        # None
        # ----
        heatmaps_norm = normalization.normalize_heatmaps(None)
        assert heatmaps_norm is None

        # ----
        # array
        # ----
        heatmaps_norm = normalization.normalize_heatmaps(
            np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1,
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        heatmaps_norm = normalization.normalize_heatmaps(
            np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1,
            shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
        )
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        # --> heatmaps for too many images
        with self.assertRaises(ValueError):
            _heatmaps_norm = normalization.normalize_heatmaps(
                np.zeros((2, 1, 1, 1), dtype=np.float32) + 0.1,
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # --> too few heatmaps
        with self.assertRaises(ValueError):
            _heatmaps_norm = normalization.normalize_heatmaps(
                np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1,
                np.zeros((2, 1, 1, 3), dtype=np.uint8)
            )

        # --> wrong channel number
        with self.assertRaises(ValueError):
            _heatmaps_norm = normalization.normalize_heatmaps(
                np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )

        # --> images None
        with self.assertRaises(ValueError):
            _heatmaps_norm = normalization.normalize_heatmaps(
                np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1,
                shapes=None
            )

        # ----
        # single HeatmapsOnImage
        # ----
        heatmaps_norm = normalization.normalize_heatmaps(
            ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                shape=(1, 1, 3)),
            shapes=None
        )
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        # ----
        # empty iterable
        # ----
        heatmaps_norm = normalization.normalize_heatmaps(
            [],
            shapes=None
        )
        assert heatmaps_norm is None

        # ----
        # iterable of arrays
        # ----
        heatmaps_norm = normalization.normalize_heatmaps(
            [np.zeros((1, 1, 1), dtype=np.float32) + 0.1],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        heatmaps_norm = normalization.normalize_heatmaps(
            [np.zeros((1, 1, 1), dtype=np.float32) + 0.1],
            shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
        )
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)

        # --> heatmaps for too many images
        with self.assertRaises(ValueError):
            _heatmaps_norm = normalization.normalize_heatmaps(
                [
                    np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                    np.zeros((1, 1, 1), dtype=np.float32) + 0.1
                ],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # --> too few heatmaps
        with self.assertRaises(ValueError):
            _heatmaps_norm = normalization.normalize_heatmaps(
                [np.zeros((1, 1, 1), dtype=np.float32) + 0.1],
                shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
            )

        # --> images None
        with self.assertRaises(ValueError):
            _heatmaps_norm = normalization.normalize_heatmaps(
                [np.zeros((1, 1, 1), dtype=np.float32) + 0.1],
                shapes=None,
            )

        # --> wrong number of dimensions
        with self.assertRaises(ValueError):
            _heatmaps_norm = normalization.normalize_heatmaps(
                [np.zeros((1, 1, 1, 1), dtype=np.float32) + 0.1],
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )

        # ----
        # iterable of HeatmapsOnImage
        # ----
        heatmaps_norm = normalization.normalize_heatmaps(
            [ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32) + 0.1,
                shape=(1, 1, 3))],
            shapes=None
        )
        assert isinstance(heatmaps_norm, list)
        assert isinstance(heatmaps_norm[0], ia.HeatmapsOnImage)
        assert np.allclose(heatmaps_norm[0].arr_0to1, 0 + 0.1)
    
    def test_normalize_segmentation_maps(self):
        # ----
        # None
        # ----
        segmaps_norm = normalization.normalize_segmentation_maps(None)
        assert segmaps_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("int32"), np.dtype("uint32"), np.dtype(bool)]:
            segmaps_norm = normalization.normalize_segmentation_maps(
                np.zeros((1, 1, 1), dtype=dt) + 1,
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            segmaps_norm = normalization.normalize_segmentation_maps(
                np.zeros((1, 1, 1), dtype=dt) + 1,
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            # --> segmaps for too many images
            with self.assertRaises(ValueError):
                _segmaps_norm = normalization.normalize_segmentation_maps(
                    np.zeros((2, 1, 1), dtype=dt) + 1,
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
                )

            # --> too few segmaps
            with self.assertRaises(ValueError):
                _segmaps_norm = normalization.normalize_segmentation_maps(
                    np.zeros((1, 1, 1), dtype=dt) + 1,
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> images None
            with self.assertRaises(ValueError):
                _segmaps_norm = normalization.normalize_segmentation_maps(
                    np.zeros((1, 1, 1), dtype=dt) + 1,
                    shapes=None
                )

        # ----
        # single SegmentationMapOnImage
        # ----
        segmaps_norm = normalization.normalize_segmentation_maps(
            ia.SegmentationMapOnImage(
                np.zeros((1, 1), dtype=np.int32) + 1,
                shape=(1, 1, 3),
                nb_classes=2),
            shapes=None
        )
        assert isinstance(segmaps_norm, list)
        assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
        assert np.allclose(segmaps_norm[0].arr[..., 1], 0 + 1)

        # ----
        # empty iterable
        # ----
        segmaps_norm = normalization.normalize_segmentation_maps(
            [], shapes=None
        )
        assert segmaps_norm is None

        # ----
        # iterable of arrays
        # ----
        for dt in [np.dtype("int32"), np.dtype("uint32"), np.dtype(bool)]:
            segmaps_norm = normalization.normalize_segmentation_maps(
                [np.zeros((1, 1), dtype=dt) + 1],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            segmaps_norm = normalization.normalize_segmentation_maps(
                [np.zeros((1, 1), dtype=dt) + 1],
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )
            assert isinstance(segmaps_norm, list)
            assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
            assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

            # --> heatmaps for too many images
            with self.assertRaises(ValueError):
                _segmaps_norm = normalization.normalize_segmentation_maps(
                    [
                        np.zeros((1, 1), dtype=np.int32) + 1,
                        np.zeros((1, 1), dtype=np.int32) + 1
                    ],
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
                )

            # --> too few segmaps
            with self.assertRaises(ValueError):
                _segmaps_norm = normalization.normalize_segmentation_maps(
                    [np.zeros((1, 1), dtype=np.int32) + 1],
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> images None
            with self.assertRaises(ValueError):
                _segmaps_norm = normalization.normalize_segmentation_maps(
                    [np.zeros((1, 1), dtype=np.int32) + 1],
                    shapes=None
                )

            # --> wrong number of dimensions
            with self.assertRaises(ValueError):
                _segmaps_norm = normalization.normalize_segmentation_maps(
                    [np.zeros((1, 1, 1), dtype=np.int32) + 1],
                    shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
                )

        # ----
        # iterable of SegmentationMapOnImage
        # ----
        segmaps_norm = normalization.normalize_segmentation_maps(
            [ia.SegmentationMapOnImage(
                np.zeros((1, 1), dtype=np.int32) + 1,
                shape=(1, 1, 3),
                nb_classes=2)],
            shapes=None
        )
        assert isinstance(segmaps_norm, list)
        assert isinstance(segmaps_norm[0], ia.SegmentationMapOnImage)
        assert np.allclose(segmaps_norm[0].arr[..., 1], 1)

    def test_normalize_keypoints(self):
        def _assert_single_image_expected(inputs):
            # --> images None
            with self.assertRaises(ValueError):
                _keypoints_norm = normalization.normalize_keypoints(
                    inputs, None)

            # --> too many images
            with self.assertRaises(ValueError):
                _keypoints_norm = normalization.normalize_keypoints(
                    inputs,
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> too many images
            with self.assertRaises(ValueError):
                _keypoints_norm = normalization.normalize_keypoints(
                    inputs,
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                            np.zeros((1, 1, 3), dtype=np.uint8)]
                )

        # ----
        # None
        # ----
        keypoints_norm = normalization.normalize_keypoints(None)
        assert keypoints_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            keypoints_norm = normalization.normalize_keypoints(
                np.zeros((1, 1, 2), dtype=dt) + 1,
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 1
            assert np.allclose(keypoints_norm[0].to_xy_array(), 1)

            keypoints_norm = normalization.normalize_keypoints(
                np.zeros((1, 5, 2), dtype=dt) + 1,
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 5
            assert np.allclose(keypoints_norm[0].to_xy_array(), 1)

            # --> keypoints for too many images
            with self.assertRaises(ValueError):
                _keypoints_norm = normalization.normalize_keypoints(
                    np.zeros((2, 1, 2), dtype=dt) + 1,
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
                )

            # --> too few keypoints
            with self.assertRaises(ValueError):
                _keypoints_norm = normalization.normalize_keypoints(
                    np.zeros((1, 1, 2), dtype=dt) + 1,
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> wrong keypoints shape
            with self.assertRaises(ValueError):
                _keypoints_norm = normalization.normalize_keypoints(
                    np.zeros((1, 1, 100), dtype=dt) + 1,
                    shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
                )

            _assert_single_image_expected(np.zeros((1, 1, 2), dtype=dt) + 1)

        # ----
        # (x,y)
        # ----
        keypoints_norm = normalization.normalize_keypoints(
            (1, 2),
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        _assert_single_image_expected((1, 2))

        # ----
        # single Keypoint instance
        # ----
        keypoints_norm = normalization.normalize_keypoints(
            ia.Keypoint(x=1, y=2),
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        _assert_single_image_expected(ia.Keypoint(x=1, y=2))

        # ----
        # single KeypointsOnImage instance
        # ----
        keypoints_norm = normalization.normalize_keypoints(
            ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3)),
            shapes=None
        )
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 1
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2

        # ----
        # empty iterable
        # ----
        keypoints_norm = normalization.normalize_keypoints(
            [], shapes=None
        )
        assert keypoints_norm is None

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            keypoints_norm = normalization.normalize_keypoints(
                [np.zeros((1, 2), dtype=dt) + 1],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 1
            assert np.allclose(keypoints_norm[0].to_xy_array(), 1)

            keypoints_norm = normalization.normalize_keypoints(
                [np.zeros((5, 2), dtype=dt) + 1],
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )
            assert isinstance(keypoints_norm, list)
            assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
            assert len(keypoints_norm[0].keypoints) == 5
            assert np.allclose(keypoints_norm[0].to_xy_array(), 1)

            # --> keypoints for too many images
            with self.assertRaises(ValueError):
                _keypoints_norm = normalization.normalize_keypoints(
                    [
                        np.zeros((1, 2), dtype=dt) + 1,
                        np.zeros((1, 2), dtype=dt) + 1
                    ],
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
                )

            # --> too few keypoints
            with self.assertRaises(ValueError):
                _keypoints_norm = normalization.normalize_keypoints(
                    [np.zeros((1, 2), dtype=dt) + 1],
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> images None
            with self.assertRaises(ValueError):
                _keypoints_norm = normalization.normalize_keypoints(
                    [np.zeros((1, 2), dtype=dt) + 1],
                    shapes=None
                )

            # --> wrong shape
            with self.assertRaises(ValueError):
                _keypoints_norm = normalization.normalize_keypoints(
                    [np.zeros((1, 100), dtype=dt) + 1],
                    shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
                )

        # ----
        # iterable of (x,y)
        # ----
        keypoints_norm = normalization.normalize_keypoints(
            [(1, 2), (3, 4)],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 2
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2
        assert keypoints_norm[0].keypoints[1].x == 3
        assert keypoints_norm[0].keypoints[1].y == 4

        # may only be used for single images
        with self.assertRaises(ValueError):
            _keypoints_norm = normalization.normalize_keypoints(
                [(1, 2)],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of Keypoint
        # ----
        keypoints_norm = normalization.normalize_keypoints(
            [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(keypoints_norm, list)
        assert isinstance(keypoints_norm[0], ia.KeypointsOnImage)
        assert len(keypoints_norm[0].keypoints) == 2
        assert keypoints_norm[0].keypoints[0].x == 1
        assert keypoints_norm[0].keypoints[0].y == 2
        assert keypoints_norm[0].keypoints[1].x == 3
        assert keypoints_norm[0].keypoints[1].y == 4

        # may only be used for single images
        with self.assertRaises(ValueError):
            _keypoints_norm = normalization.normalize_keypoints(
                [ia.Keypoint(x=1, y=2)],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of KeypointsOnImage
        # ----
        keypoints_norm = normalization.normalize_keypoints(
            [
                ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3)),
                ia.KeypointsOnImage([ia.Keypoint(x=3, y=4)], shape=(1, 1, 3)),
            ],
            shapes=None
        )
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
        keypoints_norm = normalization.normalize_keypoints(
            [[]],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert keypoints_norm is None

        # ----
        # iterable of iterable of (x,y)
        # ----
        keypoints_norm = normalization.normalize_keypoints(
            [
                [(1, 2), (3, 4)],
                [(5, 6), (7, 8)]
            ],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)]
        )
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
        with self.assertRaises(ValueError):
            _keypoints_norm = normalization.normalize_keypoints(
                [
                    [(1, 2), (3, 4)],
                    [(5, 6), (7, 8)]
                ],
                shapes=None
            )

        # --> different number of images
        with self.assertRaises(ValueError):
            _keypoints_norm = normalization.normalize_keypoints(
                [
                    [(1, 2), (3, 4)],
                    [(5, 6), (7, 8)]
                ],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of iterable of Keypoint
        # ----
        keypoints_norm = normalization.normalize_keypoints(
            [
                [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
                [ia.Keypoint(x=5, y=6), ia.Keypoint(x=7, y=8)]
            ],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)]
        )
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
        with self.assertRaises(ValueError):
            _keypoints_norm = normalization.normalize_keypoints(
                [
                    [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
                    [ia.Keypoint(x=5, y=6), ia.Keypoint(x=7, y=8)]
                ],
                shapes=None
            )

        # --> different number of images
        with self.assertRaises(ValueError):
            _keypoints_norm = normalization.normalize_keypoints(
                [
                    [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
                    [ia.Keypoint(x=5, y=6), ia.Keypoint(x=7, y=8)]
                ],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

    def test_normalize_bounding_boxes(self):
        def _assert_single_image_expected(inputs):
            # --> images None
            with self.assertRaises(ValueError):
                _bbs_norm = normalization.normalize_bounding_boxes(
                    inputs,
                    shapes=None
                )

            # --> too many images
            with self.assertRaises(ValueError):
                _bbs_norm = normalization.normalize_bounding_boxes(
                    inputs,
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> too many images
            with self.assertRaises(ValueError):
                _bbs_norm = normalization.normalize_bounding_boxes(
                    inputs,
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                            np.zeros((1, 1, 3), dtype=np.uint8)]
                )

        # ----
        # None
        # ----
        bbs_norm = normalization.normalize_bounding_boxes(None)
        assert bbs_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            bbs_norm = normalization.normalize_bounding_boxes(
                np.zeros((1, 1, 4), dtype=dt) + 1,
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 1
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            bbs_norm = normalization.normalize_bounding_boxes(
                np.zeros((1, 5, 4), dtype=dt) + 1,
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 5
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            # --> bounding boxes for too many images
            with self.assertRaises(ValueError):
                _bbs_norm = normalization.normalize_bounding_boxes(
                    np.zeros((2, 1, 4), dtype=dt) + 1,
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
                )

            # --> too few bounding boxes
            with self.assertRaises(ValueError):
                _bbs_norm = normalization.normalize_bounding_boxes(
                    np.zeros((1, 1, 4), dtype=dt) + 1,
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> wrong keypoints shape
            with self.assertRaises(ValueError):
                _bbs_norm = normalization.normalize_bounding_boxes(
                    np.zeros((1, 1, 100), dtype=dt) + 1,
                    shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
                )

            _assert_single_image_expected(np.zeros((1, 1, 4), dtype=dt) + 1)

        # ----
        # (x1,y1,x2,y2)
        # ----
        bbs_norm = normalization.normalize_bounding_boxes(
            (1, 2, 3, 4),
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(bbs_norm, list)
        assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
        assert len(bbs_norm[0].bounding_boxes) == 1
        assert bbs_norm[0].bounding_boxes[0].x1 == 1
        assert bbs_norm[0].bounding_boxes[0].y1 == 2
        assert bbs_norm[0].bounding_boxes[0].x2 == 3
        assert bbs_norm[0].bounding_boxes[0].y2 == 4

        _assert_single_image_expected((1, 2, 3, 4))

        # ----
        # single BoundingBox instance
        # ----
        bbs_norm = normalization.normalize_bounding_boxes(
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
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
        bbs_norm = normalization.normalize_bounding_boxes(
            ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                shape=(1, 1, 3)),
            shapes=None
        )
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
        bbs_norm = normalization.normalize_bounding_boxes([], shapes=None)
        assert bbs_norm is None

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            bbs_norm = normalization.normalize_bounding_boxes(
                [np.zeros((1, 4), dtype=dt) + 1],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 1
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            bbs_norm = normalization.normalize_bounding_boxes(
                [np.zeros((5, 4), dtype=dt) + 1],
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )
            assert isinstance(bbs_norm, list)
            assert isinstance(bbs_norm[0], ia.BoundingBoxesOnImage)
            assert len(bbs_norm[0].bounding_boxes) == 5
            assert np.allclose(bbs_norm[0].to_xyxy_array(), 1)

            # --> bounding boxes for too many images
            with self.assertRaises(ValueError):
                _bbs_norm = normalization.normalize_bounding_boxes(
                    [
                        np.zeros((1, 4), dtype=dt) + 1,
                        np.zeros((1, 4), dtype=dt) + 1
                    ],
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
                )

            # --> too few bounding boxes
            with self.assertRaises(ValueError):
                _bbs_norm = normalization.normalize_bounding_boxes(
                    [np.zeros((1, 4), dtype=dt) + 1],
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> images None
            with self.assertRaises(ValueError):
                _bbs_norm = normalization.normalize_bounding_boxes(
                    [np.zeros((1, 4), dtype=dt) + 1],
                    shapes=None
                )

            # --> wrong shape
            with self.assertRaises(ValueError):
                _bbs_norm = normalization.normalize_bounding_boxes(
                    [np.zeros((1, 100), dtype=dt) + 1],
                    shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
                )

        # ----
        # iterable of (x1,y1,x2,y2)
        # ----
        bbs_norm = normalization.normalize_bounding_boxes(
            [(1, 2, 3, 4), (5, 6, 7, 8)],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
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
        with self.assertRaises(ValueError):
            _bbs_norm = normalization.normalize_bounding_boxes(
                [(1, 2, 3, 4)],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of BoundingBox
        # ----
        bbs_norm = normalization.normalize_bounding_boxes(
            [
                ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)
            ],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
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
        with self.assertRaises(ValueError):
            _bbs_norm = normalization.normalize_bounding_boxes(
                [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of BoundingBoxesOnImage
        # ----
        bbs_norm = normalization.normalize_bounding_boxes(
            [
                ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                    shape=(1, 1, 3)),
                ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                    shape=(1, 1, 3))
            ],
            shapes=None
        )
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
        bbs_norm = normalization.normalize_bounding_boxes(
            [[]],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert bbs_norm is None

        # ----
        # iterable of iterable of (x1,y1,x2,y2)
        # ----
        bbs_norm = normalization.normalize_bounding_boxes(
            [
                [(1, 2, 3, 4)],
                [(5, 6, 7, 8), (9, 10, 11, 12)]
            ],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)]
        )
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
        with self.assertRaises(ValueError):
            _bbs_norm = normalization.normalize_bounding_boxes(
                [
                    [(1, 2, 3, 4), (3, 4, 5, 6)],
                    [(5, 6, 7, 8), (7, 8, 9, 10)]
                ],
                shapes=None
            )

        # --> different number of images
        with self.assertRaises(ValueError):
            _bbs_norm = normalization.normalize_bounding_boxes(
                [
                    [(1, 2, 3, 4)],
                    [(5, 6, 7, 8)]
                ],
                [np.zeros((1, 1, 3), dtype=np.uint8),
                 np.zeros((1, 1, 3), dtype=np.uint8),
                 np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of iterable of Keypoint
        # ----
        bbs_norm = normalization.normalize_bounding_boxes(
            [
                [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                 ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                [ia.BoundingBox(x1=9, y1=10, x2=11, y2=12),
                 ia.BoundingBox(x1=13, y1=14, x2=15, y2=16)]
            ],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)]
        )
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
        with self.assertRaises(ValueError):
            _bbs_norm = normalization.normalize_bounding_boxes(
                [
                    [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                     ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                    [ia.BoundingBox(x1=9, y1=10, x2=11, y2=12),
                     ia.BoundingBox(x1=13, y1=14, x2=15, y2=16)]
                ],
                shapes=None
            )

        # --> different number of images
        with self.assertRaises(ValueError):
            _bbs_norm = normalization.normalize_bounding_boxes(
                [
                    [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4),
                     ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)],
                    [ia.BoundingBox(x1=9, y1=10, x2=11, y2=12),
                     ia.BoundingBox(x1=13, y1=14, x2=15, y2=16)]
                ],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )
    
    def test_normalize_polygons(self):
        def _assert_single_image_expected(inputs):
            # --> images None
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    inputs, shapes=None)

            # --> too many images
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    inputs,
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8))

            # --> too many images
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    inputs,
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                            np.zeros((1, 1, 3), dtype=np.uint8)]
                )

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
        polygons_norm = normalization.normalize_polygons(None)
        assert polygons_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            polygons_norm = normalization.normalize_polygons(
                coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 1
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            polygons_norm = normalization.normalize_polygons(
                np.tile(
                    coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                    (1, 5, 1, 1)
                ),
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 5
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            # --> polygons for too many images
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    np.tile(
                        coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                        (2, 1, 1, 1)
                    ),
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
                )

            # --> too few polygons
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    np.tile(
                        coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                        (1, 1, 1, 1)
                    ),
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> wrong polygons shape
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    np.tile(
                        coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                        (1, 1, 1, 10)
                    ),
                    shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
                )

            _assert_single_image_expected(
                coords1_arr[np.newaxis, np.newaxis, ...].astype(dt))

        # ----
        # single Polygon instance
        # ----
        polygons_norm = normalization.normalize_polygons(
            ia.Polygon(coords1),
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        _assert_single_image_expected(ia.Polygon(coords1))

        # ----
        # single PolygonsOnImage instance
        # ----
        polygons_norm = normalization.normalize_polygons(
            ia.PolygonsOnImage([ia.Polygon(coords1)], shape=(1, 1, 3)),
            shapes=None
        )
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        # ----
        # empty iterable
        # ----
        polygons_norm = normalization.normalize_polygons(
            [], shapes=None
        )
        assert polygons_norm is None

        # ----
        # iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            polygons_norm = normalization.normalize_polygons(
                [coords1_arr[np.newaxis, ...].astype(dt)],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 1
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            polygons_norm = normalization.normalize_polygons(
                [np.tile(
                    coords1_arr[np.newaxis, ...].astype(dt),
                    (5, 1, 1)
                )],
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 5
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            # --> polygons for too many images
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    [coords1_arr[np.newaxis, ...].astype(dt),
                     coords2_arr[np.newaxis, ...].astype(dt)],
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
                )

            # --> too few polygons
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    [coords1_arr[np.newaxis, ...].astype(dt)],
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> wrong polygons shape
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    [np.tile(
                        coords1_arr[np.newaxis, ...].astype(dt),
                        (1, 1, 10)
                    )],
                    shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
                )

            _assert_single_image_expected(
                [coords1_arr[np.newaxis, ...].astype(dt)]
            )

        # ----
        # iterable of (x,y)
        # ----
        polygons_norm = normalization.normalize_polygons(
            coords1,
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        # may only be used for single images
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                coords1,
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of Keypoint
        # ----
        polygons_norm = normalization.normalize_polygons(
            coords1_kps,
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        # may only be used for single images
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                coords1_kps,
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of Polygon
        # ----
        polygons_norm = normalization.normalize_polygons(
            [ia.Polygon(coords1), ia.Polygon(coords2)],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        # may only be used for single images
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [ia.Polygon(coords1)],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of PolygonsOnImage
        # ----
        polygons_norm = normalization.normalize_polygons(
            [
                ia.PolygonsOnImage([ia.Polygon(coords1)], shape=(1, 1, 3)),
                ia.PolygonsOnImage([ia.Polygon(coords2)], shape=(1, 1, 3))
            ],
            shapes=None
        )
        assert isinstance(polygons_norm, list)

        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 1
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)

        assert isinstance(polygons_norm[1], ia.PolygonsOnImage)
        assert len(polygons_norm[1].polygons) == 1
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords2)

        # ----
        # iterable of empty iterables
        # ----
        polygons_norm = normalization.normalize_polygons(
            [[]],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert polygons_norm is None

        # ----
        # iterable of iterable of array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            polygons_norm = normalization.normalize_polygons(
                [[coords1_arr.astype(dt)]],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 1
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            polygons_norm = normalization.normalize_polygons(
                [[
                    np.copy(coords1_arr).astype(dt) for _ in sm.xrange(5)
                ]],
                shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
            )
            assert isinstance(polygons_norm, list)
            assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
            assert len(polygons_norm[0].polygons) == 5
            assert np.allclose(polygons_norm[0].polygons[0].exterior,
                               coords1_arr)

            # --> polygons for too many images
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    [[coords1_arr.astype(dt)],
                     [coords2_arr.astype(dt)]],
                    shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
                )

            # --> too few polygons
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    [[coords1_arr.astype(dt)]],
                    shapes=np.zeros((2, 1, 1, 3), dtype=np.uint8)
                )

            # --> wrong polygons shape
            with self.assertRaises(ValueError):
                _polygons_norm = normalization.normalize_polygons(
                    [[np.tile(
                        coords1_arr.astype(dt),
                        (1, 1, 10)
                    )]],
                    shapes=np.zeros((1, 1, 1, 3), dtype=np.uint8)
                )

            _assert_single_image_expected(
                [[coords1_arr.astype(dt)]]
            )

        # ----
        # iterable of iterable of (x,y)
        # ----
        polygons_norm = normalization.normalize_polygons(
            [coords1, coords2],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        # --> images None
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [coords1, coords2],
                shapes=None
            )

        # --> different number of images
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [coords1, coords2],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of iterable of Keypoint
        # ----
        polygons_norm = normalization.normalize_polygons(
            [coords1_kps, coords2_kps],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)
        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        # --> images None
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [coords1_kps, coords2_kps],
                shapes=None
            )

        # --> different number of images
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [coords1_kps, coords2_kps],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of iterable of Polygon
        # ----
        polygons_norm = normalization.normalize_polygons(
            [
                [ia.Polygon(coords1), ia.Polygon(coords2)],
                [ia.Polygon(coords3), ia.Polygon(coords4)]
            ],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)]
        )
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
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [
                    [ia.Polygon(coords1), ia.Polygon(coords2)],
                    [ia.Polygon(coords3), ia.Polygon(coords4)]
                ],
                shapes=None
            )

        # --> different number of images
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [
                    [ia.Polygon(coords1), ia.Polygon(coords2)],
                    [ia.Polygon(coords3), ia.Polygon(coords4)]
                ],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of iterable of empty iterable
        # ----
        polygons_norm = normalization.normalize_polygons(
            [[[]]],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert polygons_norm is None

        # ----
        # iterable of iterable of iterable of (x,y)
        # ----
        polygons_norm = normalization.normalize_polygons(
            [[coords1, coords2], [coords3, coords4]],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords3)
        assert polygons_norm[1].polygons[1].exterior_almost_equals(coords4)

        # --> images None
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [[coords1, coords2]],
                shapes=None
            )

        # --> different number of images
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [[coords1, coords2], [coords3]],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

        # ----
        # iterable of iterable of iterable of Keypoint
        # ----
        polygons_norm = normalization.normalize_polygons(
            [[coords1_kps, coords2_kps], [coords3_kps, coords4_kps]],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(polygons_norm, list)
        assert isinstance(polygons_norm[0], ia.PolygonsOnImage)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[0].polygons[0].exterior_almost_equals(coords1)
        assert polygons_norm[0].polygons[1].exterior_almost_equals(coords2)

        assert len(polygons_norm[0].polygons) == 2
        assert polygons_norm[1].polygons[0].exterior_almost_equals(coords3)
        assert polygons_norm[1].polygons[1].exterior_almost_equals(coords4)

        # --> images None
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [[coords1_kps, coords2_kps]],
                shapes=None
            )

        # --> different number of images
        with self.assertRaises(ValueError):
            _polygons_norm = normalization.normalize_polygons(
                [[coords1_kps, coords2_kps], [coords3_kps]],
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        np.zeros((1, 1, 3), dtype=np.uint8)]
            )

    # essentially already tested via polygons, as they are based on the
    # same methods, hence a short test here
    def test_normalize_line_strings(self):
        coords1 = [(0, 0), (10, 0), (10, 10)]
        coords2 = [(5, 5), (15, 5), (15, 15)]
        coords3 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        coords4 = [(5, 5), (15, 5), (15, 15), (5, 15)]

        coords1_arr = np.float32(coords1)

        # ----
        # None
        # ----
        lss_norm = normalization.normalize_line_strings(None)
        assert lss_norm is None

        # ----
        # array
        # ----
        for dt in [np.dtype("float32"), np.dtype("int16"), np.dtype("uint16")]:
            lss_norm = normalization.normalize_line_strings(
                coords1_arr[np.newaxis, np.newaxis, ...].astype(dt),
                shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
            )
            assert isinstance(lss_norm, list)
            assert isinstance(lss_norm[0], ia.LineStringsOnImage)
            assert len(lss_norm[0].line_strings) == 1
            assert np.allclose(lss_norm[0].line_strings[0].coords, coords1_arr)

        # ----
        # single LineString instance
        # ----
        lss_norm = normalization.normalize_line_strings(
            ia.LineString(coords1),
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(lss_norm, list)
        assert isinstance(lss_norm[0], ia.LineStringsOnImage)
        assert len(lss_norm[0].line_strings) == 1
        assert np.allclose(lss_norm[0].line_strings[0].coords, coords1)

        # ----
        # single LineStringOnImage instance
        # ----
        lss_norm = normalization.normalize_line_strings(
            ia.LineStringsOnImage([ia.LineString(coords1)], shape=(1, 1, 3)),
            shapes=None
        )
        assert isinstance(lss_norm, list)
        assert isinstance(lss_norm[0], ia.LineStringsOnImage)
        assert len(lss_norm[0].line_strings) == 1
        assert np.allclose(lss_norm[0].line_strings[0].coords, coords1)

        # ----
        # empty iterable
        # ----
        lss_norm = normalization.normalize_line_strings(
            [], shapes=None
        )
        assert lss_norm is None

        # ----
        # iterable of LineStringOnImage
        # ----
        lss_norm = normalization.normalize_line_strings(
            [
                ia.LineStringsOnImage(
                    [ia.LineString(coords1)], shape=(1, 1, 3)),
                ia.LineStringsOnImage(
                    [ia.LineString(coords2)], shape=(1, 1, 3))
            ],
            shapes=None
        )
        assert isinstance(lss_norm, list)

        assert isinstance(lss_norm[0], ia.LineStringsOnImage)
        assert len(lss_norm[0].line_strings) == 1
        assert np.allclose(lss_norm[0].line_strings[0].coords, coords1)

        assert isinstance(lss_norm[1], ia.LineStringsOnImage)
        assert len(lss_norm[1].line_strings) == 1
        assert np.allclose(lss_norm[1].line_strings[0].coords, coords2)

        # ----
        # iterable of iterable of LineString
        # ----
        lss_norm = normalization.normalize_line_strings(
            [
                [ia.LineString(coords1), ia.LineString(coords2)],
                [ia.LineString(coords3), ia.LineString(coords4)]
            ],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(lss_norm, list)
        assert isinstance(lss_norm[0], ia.LineStringsOnImage)
        assert isinstance(lss_norm[1], ia.LineStringsOnImage)

        assert len(lss_norm[0].line_strings) == 2
        assert np.allclose(lss_norm[0].line_strings[0].coords, coords1)
        assert np.allclose(lss_norm[0].line_strings[1].coords, coords2)

        assert len(lss_norm[1].line_strings) == 2
        assert np.allclose(lss_norm[1].line_strings[0].coords, coords3)
        assert np.allclose(lss_norm[1].line_strings[1].coords, coords4)

        # ----
        # iterable of iterable of iterable of (x,y)
        # ----
        lss_norm = normalization.normalize_line_strings(
            [[coords1, coords2], [coords3, coords4]],
            shapes=[np.zeros((1, 1, 3), dtype=np.uint8),
                    np.zeros((1, 1, 3), dtype=np.uint8)]
        )
        assert isinstance(lss_norm, list)
        assert isinstance(lss_norm[0], ia.LineStringsOnImage)

        assert len(lss_norm[0].line_strings) == 2
        assert np.allclose(lss_norm[0].line_strings[0].coords, coords1)
        assert np.allclose(lss_norm[0].line_strings[1].coords, coords2)

        assert len(lss_norm[0].line_strings) == 2
        assert np.allclose(lss_norm[1].line_strings[0].coords, coords3)
        assert np.allclose(lss_norm[1].line_strings[1].coords, coords4)

    def test__find_first_nonempty(self):
        # None
        observed = normalization.find_first_nonempty(None)
        assert observed[0] is None
        assert observed[1] is True
        assert len(observed[2]) == 0

        # None with parents
        observed = normalization.find_first_nonempty(None, parents=["foo"])
        assert observed[0] is None
        assert observed[1] is True
        assert len(observed[2]) == 1
        assert observed[2][0] == "foo"

        # array
        observed = normalization.find_first_nonempty(np.zeros((4, 4, 3)))
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 0

        # int
        observed = normalization.find_first_nonempty(0)
        assert observed[0] == 0
        assert observed[1] is True
        assert len(observed[2]) == 0

        # str
        observed = normalization.find_first_nonempty("foo")
        assert observed[0] == "foo"
        assert observed[1] is True
        assert len(observed[2]) == 0

        # empty list
        observed = normalization.find_first_nonempty([])
        assert observed[0] is None
        assert observed[1] is False
        assert len(observed[2]) == 0

        # empty list of empty lists
        observed = normalization.find_first_nonempty([[], [], []])
        assert observed[0] is None
        assert observed[1] is False
        assert len(observed[2]) == 1

        # empty list of empty lists of empty lists
        observed = normalization.find_first_nonempty([[], [[]], []])
        assert observed[0] is None
        assert observed[1] is False
        assert len(observed[2]) == 2

        # list of None
        observed = normalization.find_first_nonempty([None, None])
        assert observed[0] is None
        assert observed[1] is True
        assert len(observed[2]) == 1

        # list of array
        observed = normalization.find_first_nonempty([
            np.zeros((4, 4, 3)), np.zeros((5, 5, 3))])
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 1

        # list of list of array
        observed = normalization.find_first_nonempty(
            [[np.zeros((4, 4, 3))], [np.zeros((5, 5, 3))]]
        )
        assert ia.is_np_array(observed[0])
        assert observed[0].shape == (4, 4, 3)
        assert observed[1] is True
        assert len(observed[2]) == 2

        # list of tuple of array
        observed = normalization.find_first_nonempty(
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
        ntype = normalization._nonempty_info_to_type_str(
            None, True, [])
        assert ntype == "None"

        ntype = normalization._nonempty_info_to_type_str(
            None, False, [])
        assert ntype == "iterable[empty]"

        ntype = normalization._nonempty_info_to_type_str(
            None, False, [[]])
        assert ntype == "iterable-iterable[empty]"

        ntype = normalization._nonempty_info_to_type_str(
            None, False, [[], []])
        assert ntype == "iterable-iterable-iterable[empty]"

        ntype = normalization._nonempty_info_to_type_str(
            None, False, [tuple(), []])
        assert ntype == "iterable-iterable-iterable[empty]"

        ntype = normalization._nonempty_info_to_type_str(
            1, True, [tuple([1, 2])])
        assert ntype == "tuple[number,size=2]"

        ntype = normalization._nonempty_info_to_type_str(
            1, True, [[], tuple([1, 2])])
        assert ntype == "iterable-tuple[number,size=2]"

        ntype = normalization._nonempty_info_to_type_str(
            1, True, [tuple([1, 2, 3, 4])])
        assert ntype == "tuple[number,size=4]"

        ntype = normalization._nonempty_info_to_type_str(
            1, True, [[], tuple([1, 2, 3, 4])])
        assert ntype == "iterable-tuple[number,size=4]"

        with self.assertRaises(AssertionError):
            ntype = normalization._nonempty_info_to_type_str(
                1, True, [tuple([1, 2, 3])])
            assert ntype == "tuple[number,size=4]"

        ntype = normalization._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.uint8), True, [])
        assert ntype == "array[uint]"

        ntype = normalization._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.float32), True, [])
        assert ntype == "array[float]"

        ntype = normalization._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.int32), True, [])
        assert ntype == "array[int]"

        ntype = normalization._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=bool), True, [])
        assert ntype == "array[bool]"

        ntype = normalization._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.dtype("complex")), True, [])
        assert ntype == "array[c]"

        ntype = normalization._nonempty_info_to_type_str(
            np.zeros((4, 4, 3), dtype=np.uint8), True, [[]])
        assert ntype == "iterable-array[uint]"

        ntype = normalization._nonempty_info_to_type_str(
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
            ntype = normalization._nonempty_info_to_type_str(
                cls, True, [])
            assert ntype == cls_name

            ntype = normalization._nonempty_info_to_type_str(
                cls, True, [[]])
            assert ntype == "iterable-%s" % (cls_name,)

            ntype = normalization._nonempty_info_to_type_str(
                cls, True, [[], tuple()])
            assert ntype == "iterable-iterable-%s" % (cls_name,)

    def test_estimate_heatmaps_norm_type(self):
        ntype = normalization.estimate_heatmaps_norm_type(None)
        assert ntype == "None"

        ntype = normalization.estimate_heatmaps_norm_type(
            np.zeros((1, 1, 1, 1), dtype=np.float32))
        assert ntype == "array[float]"

        ntype = normalization.estimate_heatmaps_norm_type(
            ia.HeatmapsOnImage(
                np.zeros((1, 1, 1), dtype=np.float32),
                shape=(1, 1, 1)
            )
        )
        assert ntype == "HeatmapsOnImage"

        ntype = normalization.estimate_heatmaps_norm_type([])
        assert ntype == "iterable[empty]"

        ntype = normalization.estimate_heatmaps_norm_type(
            [np.zeros((1, 1, 1), dtype=np.float32)])
        assert ntype == "iterable-array[float]"

        ntype = normalization.estimate_heatmaps_norm_type([
            ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                               shape=(1, 1, 1))
        ])
        assert ntype == "iterable-HeatmapsOnImage"

        # --
        # error cases
        # --
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_heatmaps_norm_type(1)

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_heatmaps_norm_type("foo")

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_heatmaps_norm_type(
                np.zeros((1, 1, 1), dtype=np.int32))

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_heatmaps_norm_type([1])

        # wrong class
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_heatmaps_norm_type(
                ia.KeypointsOnImage([], shape=(1, 1, 1)))

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_heatmaps_norm_type([[]])

        # list of list of Heatmaps, only list of Heatmaps is max
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_heatmaps_norm_type([
                [ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                                    shape=(1, 1, 1))]
            ])

    def test_estimate_segmaps_norm_type(self):
        ntype = normalization.estimate_segmaps_norm_type(None)
        assert ntype == "None"

        for name, dt in zip(["int", "uint", "bool"],
                            [np.int32, np.uint16, bool]):
            ntype = normalization.estimate_segmaps_norm_type(
                np.zeros((1, 1, 1, 1), dtype=dt))
            assert ntype == "array[%s]" % (name,)

        ntype = normalization.estimate_segmaps_norm_type(
            ia.SegmentationMapOnImage(
                np.zeros((1, 1, 1), dtype=np.int32),
                shape=(1, 1, 1),
                nb_classes=1
            )
        )
        assert ntype == "SegmentationMapOnImage"

        ntype = normalization.estimate_segmaps_norm_type([])
        assert ntype == "iterable[empty]"

        ntype = normalization.estimate_segmaps_norm_type(
            [np.zeros((1, 1, 1), dtype=np.int32)])
        assert ntype == "iterable-array[int]"

        ntype = normalization.estimate_segmaps_norm_type([
            ia.SegmentationMapOnImage(np.zeros((1, 1, 1), dtype=np.int32),
                                      shape=(1, 1, 1),
                                      nb_classes=1)
        ])
        assert ntype == "iterable-SegmentationMapOnImage"

        # --
        # error cases
        # --
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_segmaps_norm_type(1)

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_segmaps_norm_type("foo")

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_segmaps_norm_type([1])

        # wrong class
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_segmaps_norm_type(
                ia.KeypointsOnImage([], shape=(1, 1, 1)))

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_segmaps_norm_type([[]])

        # list of list of SegMap, only list of SegMap is max
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_segmaps_norm_type([
                [ia.SegmentationMapOnImage(
                    np.zeros((1, 1, 1), dtype=np.int32),
                    shape=(1, 1, 1),
                    nb_classes=1)]
            ])

    def test_estimate_keypoints_norm_type(self):
        ntype = normalization.estimate_keypoints_norm_type(None)
        assert ntype == "None"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            ntype = normalization.estimate_keypoints_norm_type(
                np.zeros((1, 5, 2), dtype=dt))
            assert ntype == "array[%s]" % (name,)

        ntype = normalization.estimate_keypoints_norm_type((1, 2))
        assert ntype == "tuple[number,size=2]"

        ntype = normalization.estimate_keypoints_norm_type(
            ia.Keypoint(x=1, y=2))
        assert ntype == "Keypoint"

        ntype = normalization.estimate_keypoints_norm_type(
            ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3)))
        assert ntype == "KeypointsOnImage"

        ntype = normalization.estimate_keypoints_norm_type([])
        assert ntype == "iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            ntype = normalization.estimate_keypoints_norm_type(
                [np.zeros((5, 2), dtype=dt)])
            assert ntype == "iterable-array[%s]" % (name,)

        ntype = normalization.estimate_keypoints_norm_type([(1, 2)])
        assert ntype == "iterable-tuple[number,size=2]"

        ntype = normalization.estimate_keypoints_norm_type(
            [ia.Keypoint(x=1, y=2)])
        assert ntype == "iterable-Keypoint"

        ntype = normalization.estimate_keypoints_norm_type([
            ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=(1, 1, 3))])
        assert ntype == "iterable-KeypointsOnImage"

        ntype = normalization.estimate_keypoints_norm_type([[]])
        assert ntype == "iterable-iterable[empty]"

        ntype = normalization.estimate_keypoints_norm_type([[(1, 2)]])
        assert ntype == "iterable-iterable-tuple[number,size=2]"

        ntype = normalization.estimate_keypoints_norm_type(
            [[ia.Keypoint(x=1, y=2)]])
        assert ntype == "iterable-iterable-Keypoint"

        # --
        # error cases
        # --
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_keypoints_norm_type(1)

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_keypoints_norm_type("foo")

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_keypoints_norm_type([1])

        # wrong class
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_keypoints_norm_type(
                ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                                   shape=(1, 1, 1)))

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_keypoints_norm_type([[[]]])

        # list of list of list of keypoints,
        # only list of list of keypoints is max
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_keypoints_norm_type(
                [[[ia.Keypoint(x=1, y=2)]]])

    def test_estimate_bounding_boxes_norm_type(self):
        ntype = normalization.estimate_bounding_boxes_norm_type(None)
        assert ntype == "None"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            ntype = normalization.estimate_bounding_boxes_norm_type(
                np.zeros((1, 5, 4), dtype=dt))
            assert ntype == "array[%s]" % (name,)

        ntype = normalization.estimate_bounding_boxes_norm_type((1, 2, 3, 4))
        assert ntype == "tuple[number,size=4]"

        ntype = normalization.estimate_bounding_boxes_norm_type(
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4))
        assert ntype == "BoundingBox"

        ntype = normalization.estimate_bounding_boxes_norm_type(
            ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)], shape=(1, 1, 3)))
        assert ntype == "BoundingBoxesOnImage"

        ntype = normalization.estimate_bounding_boxes_norm_type([])
        assert ntype == "iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            ntype = normalization.estimate_bounding_boxes_norm_type(
                [np.zeros((5, 4), dtype=dt)])
            assert ntype == "iterable-array[%s]" % (name,)

        ntype = normalization.estimate_bounding_boxes_norm_type([(1, 2, 3, 4)])
        assert ntype == "iterable-tuple[number,size=4]"

        ntype = normalization.estimate_bounding_boxes_norm_type([
            ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)])
        assert ntype == "iterable-BoundingBox"

        ntype = normalization.estimate_bounding_boxes_norm_type([
            ia.BoundingBoxesOnImage([ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)],
                                    shape=(1, 1, 3))])
        assert ntype == "iterable-BoundingBoxesOnImage"

        ntype = normalization.estimate_bounding_boxes_norm_type([[]])
        assert ntype == "iterable-iterable[empty]"

        ntype = normalization.estimate_bounding_boxes_norm_type(
            [[(1, 2, 3, 4)]])
        assert ntype == "iterable-iterable-tuple[number,size=4]"

        ntype = normalization.estimate_bounding_boxes_norm_type(
            [[ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]])
        assert ntype == "iterable-iterable-BoundingBox"

        # --
        # error cases
        # --
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_bounding_boxes_norm_type(1)

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_bounding_boxes_norm_type("foo")

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_bounding_boxes_norm_type([1])

        # wrong class
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_bounding_boxes_norm_type(
                ia.HeatmapsOnImage(
                    np.zeros((1, 1, 1), dtype=np.float32),
                    shape=(1, 1, 1))
            )

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_bounding_boxes_norm_type([[[]]])

        # list of list of list of bounding boxes,
        # only list of list of bounding boxes is max
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_bounding_boxes_norm_type([[[
                ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)]]])

    def test_estimate_polygons_norm_type(self):
        points = [(0, 0), (10, 0), (10, 10)]

        ntype = normalization.estimate_polygons_norm_type(None)
        assert ntype == "None"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            ntype = normalization.estimate_polygons_norm_type(
                np.zeros((1, 2, 5, 2), dtype=dt)
            )
            assert ntype == "array[%s]" % (name,)

        ntype = normalization.estimate_polygons_norm_type(
            ia.Polygon(points)
        )
        assert ntype == "Polygon"

        ntype = normalization.estimate_polygons_norm_type(
            ia.PolygonsOnImage(
                [ia.Polygon(points)], shape=(1, 1, 3))
        )
        assert ntype == "PolygonsOnImage"

        ntype = normalization.estimate_polygons_norm_type([])
        assert ntype == "iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            ntype = normalization.estimate_polygons_norm_type(
                [np.zeros((5, 4), dtype=dt)]
            )
            assert ntype == "iterable-array[%s]" % (name,)

        ntype = normalization.estimate_polygons_norm_type(points)
        assert ntype == "iterable-tuple[number,size=2]"

        ntype = normalization.estimate_polygons_norm_type(
            [ia.Keypoint(x=x, y=y) for x, y in points]
        )
        assert ntype == "iterable-Keypoint"

        ntype = normalization.estimate_polygons_norm_type([ia.Polygon(points)])
        assert ntype == "iterable-Polygon"

        ntype = normalization.estimate_polygons_norm_type(
            [ia.PolygonsOnImage([ia.Polygon(points)],
                                shape=(1, 1, 3))]
        )
        assert ntype == "iterable-PolygonsOnImage"

        ntype = normalization.estimate_polygons_norm_type([[]])
        assert ntype == "iterable-iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            ntype = normalization.estimate_polygons_norm_type(
                [[np.zeros((5, 4), dtype=dt)]]
            )
            assert ntype == "iterable-iterable-array[%s]" % (name,)

        ntype = normalization.estimate_polygons_norm_type([points])
        assert ntype == "iterable-iterable-tuple[number,size=2]"

        ntype = normalization.estimate_polygons_norm_type([[
            ia.Keypoint(x=x, y=y) for x, y in points
        ]])
        assert ntype == "iterable-iterable-Keypoint"

        ntype = normalization.estimate_polygons_norm_type(
            [[ia.Polygon(points)]]
        )
        assert ntype == "iterable-iterable-Polygon"

        ntype = normalization.estimate_polygons_norm_type([[[]]])
        assert ntype == "iterable-iterable-iterable[empty]"

        ntype = normalization.estimate_polygons_norm_type([[points]])
        assert ntype == "iterable-iterable-iterable-tuple[number,size=2]"

        ntype = normalization.estimate_polygons_norm_type(
            [[[ia.Keypoint(x=x, y=y) for x, y in points]]]
        )
        assert ntype == "iterable-iterable-iterable-Keypoint"

        # --
        # error cases
        # --
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_polygons_norm_type(1)

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_polygons_norm_type("foo")

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_polygons_norm_type([1])

        # wrong class
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_polygons_norm_type(
                ia.HeatmapsOnImage(
                    np.zeros((1, 1, 1), dtype=np.float32),
                    shape=(1, 1, 1))
            )

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_polygons_norm_type([[[[]]]])

        # list of list of list of polygons,
        # only list of list of polygons is max
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_polygons_norm_type([[[
                ia.Polygon(points)]]]
            )
    
    def test_estimate_line_strings_norm_type(self):
        points = [(0, 0), (10, 0), (10, 10)]

        ntype = normalization.estimate_line_strings_norm_type(None)
        assert ntype == "None"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            ntype = normalization.estimate_line_strings_norm_type(
                np.zeros((1, 2, 5, 2), dtype=dt)
            )
            assert ntype == "array[%s]" % (name,)

        ntype = normalization.estimate_line_strings_norm_type(
            ia.LineString(points)
        )
        assert ntype == "LineString"

        ntype = normalization.estimate_line_strings_norm_type(
            ia.LineStringsOnImage(
                [ia.LineString(points)], shape=(1, 1, 3))
        )
        assert ntype == "LineStringsOnImage"

        ntype = normalization.estimate_line_strings_norm_type([])
        assert ntype == "iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            ntype = normalization.estimate_line_strings_norm_type(
                [np.zeros((5, 4), dtype=dt)]
            )
            assert ntype == "iterable-array[%s]" % (name,)

        ntype = normalization.estimate_line_strings_norm_type(points)
        assert ntype == "iterable-tuple[number,size=2]"

        ntype = normalization.estimate_line_strings_norm_type(
            [ia.Keypoint(x=x, y=y) for x, y in points]
        )
        assert ntype == "iterable-Keypoint"

        ntype = normalization.estimate_line_strings_norm_type(
            [ia.LineString(points)])
        assert ntype == "iterable-LineString"

        ntype = normalization.estimate_line_strings_norm_type(
            [ia.LineStringsOnImage([ia.LineString(points)],
                                   shape=(1, 1, 3))]
        )
        assert ntype == "iterable-LineStringsOnImage"

        ntype = normalization.estimate_line_strings_norm_type([[]])
        assert ntype == "iterable-iterable[empty]"

        for name, dt in zip(["float", "int", "uint"],
                            [np.float32, np.int32, np.uint16]):
            ntype = normalization.estimate_line_strings_norm_type(
                [[np.zeros((5, 4), dtype=dt)]]
            )
            assert ntype == "iterable-iterable-array[%s]" % (name,)

        ntype = normalization.estimate_line_strings_norm_type([points])
        assert ntype == "iterable-iterable-tuple[number,size=2]"

        ntype = normalization.estimate_line_strings_norm_type([[
            ia.Keypoint(x=x, y=y) for x, y in points
        ]])
        assert ntype == "iterable-iterable-Keypoint"

        ntype = normalization.estimate_line_strings_norm_type(
            [[ia.LineString(points)]]
        )
        assert ntype == "iterable-iterable-LineString"

        ntype = normalization.estimate_line_strings_norm_type([[[]]])
        assert ntype == "iterable-iterable-iterable[empty]"

        ntype = normalization.estimate_line_strings_norm_type([[points]])
        assert ntype == "iterable-iterable-iterable-tuple[number,size=2]"

        ntype = normalization.estimate_line_strings_norm_type(
            [[[ia.Keypoint(x=x, y=y) for x, y in points]]]
        )
        assert ntype == "iterable-iterable-iterable-Keypoint"

        # --
        # error cases
        # --
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_line_strings_norm_type(1)

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_line_strings_norm_type("foo")

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_line_strings_norm_type([1])

        # wrong class
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_line_strings_norm_type(
                ia.HeatmapsOnImage(
                    np.zeros((1, 1, 1), dtype=np.float32),
                    shape=(1, 1, 1))
            )

        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_line_strings_norm_type([[[[]]]])

        # list of list of list of LineStrings,
        # only list of list of LineStrings is max
        with self.assertRaises(AssertionError):
            _ntype = normalization.estimate_line_strings_norm_type([[[
                ia.LineString(points)]]]
            )
