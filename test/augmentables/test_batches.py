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
                      "bounding_boxes", "polygons", "line_strings"]
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
            line_strings=6,
            data=7
        )
        for i, attr_name in enumerate(attr_names):
            assert getattr(batch, "%s_unaug" % (attr_name,)) == i
            assert getattr(batch, "%s_aug" % (attr_name,)) is None
        assert batch.data == 7

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
        assert len(keys) >= 14
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
            line_strings=[
                ia.LineStringsOnImage([
                    ia.LineString([(1, 1), (11, 1), (11, 11)])
                ], shape=(101, 101, 3))
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
        assert isinstance(observed.line_strings_unaug[0], ia.LineStringsOnImage)
        assert isinstance(observed.data, dict)

        assert observed.heatmaps_unaug[0].shape == (4, 4, 3)
        assert observed.segmentation_maps_unaug[0].shape == (5, 5, 3)
        assert observed.keypoints_unaug[0].shape == (6, 6, 3)
        assert observed.bounding_boxes_unaug[0].shape == (7, 7, 3)
        assert observed.polygons_unaug[0].shape == (100, 100, 3)
        assert observed.line_strings_unaug[0].shape == (101, 101, 3)

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
        assert observed.line_strings_unaug[0].line_strings[0].coords[0, 0] == 1
        assert observed.line_strings_unaug[0].line_strings[0].coords[0, 1] == 1
        assert observed.line_strings_unaug[0].line_strings[0].coords[1, 0] == 11
        assert observed.line_strings_unaug[0].line_strings[0].coords[1, 1] == 1
        assert observed.line_strings_unaug[0].line_strings[0].coords[2, 0] == 11
        assert observed.line_strings_unaug[0].line_strings[0].coords[2, 1] == 11

        assert observed.data["test"] == 123
        assert observed.data["foo"] == "bar"
        assert observed.data["test2"] == [1, 2, 3]
