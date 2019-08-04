from __future__ import print_function, division, absolute_import

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


ATTR_NAMES = ["images", "heatmaps", "segmentation_maps", "keypoints",
              "bounding_boxes", "polygons", "line_strings"]


class TestBatch(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___no_arguments(self):
        batch = ia.Batch()
        for attr_name in ATTR_NAMES:
            assert getattr(batch, "%s_unaug" % (attr_name,)) is None
            assert getattr(batch, "%s_aug" % (attr_name,)) is None
        assert batch.data is None

    def test___init___all_arguments_provided(self):
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
        for i, attr_name in enumerate(ATTR_NAMES):
            assert getattr(batch, "%s_unaug" % (attr_name,)) == i
            assert getattr(batch, "%s_aug" % (attr_name,)) is None
        assert batch.data == 7

    def test_warnings_for_deprecated_properties(self):
        batch = ia.Batch()
        # self.assertWarns does not exist in py2.7
        deprecated_attr_names = ["images", "heatmaps", "segmentation_maps",
                                 "keypoints", "bounding_boxes"]
        for attr_name in deprecated_attr_names:
            with self.subTest(attr_name=attr_name),\
                    warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                _ = getattr(batch, attr_name)
                assert len(caught_warnings) == 1
                assert "is deprecated" in str(caught_warnings[-1].message)

    def test_deepcopy_no_arguments(self):
        batch = ia.Batch()
        observed = batch.deepcopy()
        keys = list(observed.__dict__.keys())
        assert len(keys) >= 14
        for attr_name in keys:
            assert getattr(observed, attr_name) is None

    def test_deepcopy_only_images_provided(self):
        images = np.zeros((1, 1, 3), dtype=np.uint8)
        batch = ia.Batch(images=images)
        observed = batch.deepcopy()
        for attr_name in observed.__dict__.keys():
            if attr_name != "images_unaug":
                assert getattr(observed, attr_name) is None
        assert ia.is_np_array(observed.images_unaug)

    def test_deepcopy_every_argument_provided(self):
        images = np.zeros((1, 1, 1, 3), dtype=np.uint8)
        heatmaps = [ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32),
                                       shape=(4, 4, 3))]
        segmentation_maps = [
            ia.SegmentationMapsOnImage(np.zeros((1, 1), dtype=np.int32),
                                       shape=(5, 5, 3))]
        keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)],
                                         shape=(6, 6, 3))]
        bounding_boxes = [
            ia.BoundingBoxesOnImage([
                ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)
            ], shape=(7, 7, 3))]
        polygons = [
            ia.PolygonsOnImage([
                ia.Polygon([(0, 0), (10, 0), (10, 10)])
            ], shape=(100, 100, 3))]
        line_strings = [
            ia.LineStringsOnImage([
                ia.LineString([(1, 1), (11, 1), (11, 11)])
            ], shape=(101, 101, 3))]
        data = {"test": 123, "foo": "bar", "test2": [1, 2, 3]}

        batch = ia.Batch(
            images=images,
            heatmaps=heatmaps,
            segmentation_maps=segmentation_maps,
            keypoints=keypoints,
            bounding_boxes=bounding_boxes,
            polygons=polygons,
            line_strings=line_strings,
            data=data
        )
        observed = batch.deepcopy()

        for attr_name in observed.__dict__.keys():
            if "_unaug" not in attr_name and attr_name != "data":
                assert getattr(observed, attr_name) is None

        # must not be identical
        assert observed.images_unaug is not images
        assert observed.heatmaps_unaug is not heatmaps
        assert observed.segmentation_maps_unaug is not segmentation_maps
        assert observed.keypoints_unaug is not keypoints
        assert observed.bounding_boxes_unaug is not bounding_boxes
        assert observed.polygons_unaug is not polygons
        assert observed.line_strings_unaug is not line_strings
        assert observed.data is not data

        # verify that lists were not shallow-copied
        assert observed.heatmaps_unaug[0] is not heatmaps[0]
        assert observed.segmentation_maps_unaug[0] is not segmentation_maps[0]
        assert observed.keypoints_unaug[0] is not keypoints[0]
        assert observed.bounding_boxes_unaug[0] is not bounding_boxes[0]
        assert observed.polygons_unaug[0] is not polygons[0]
        assert observed.line_strings_unaug[0] is not line_strings[0]
        assert observed.data["test2"] is not data["test2"]

        # but must be equal
        assert ia.is_np_array(observed.images_unaug)
        assert observed.images_unaug.shape == (1, 1, 1, 3)
        assert isinstance(observed.heatmaps_unaug[0], ia.HeatmapsOnImage)
        assert isinstance(observed.segmentation_maps_unaug[0],
                          ia.SegmentationMapsOnImage)
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
        assert observed.segmentation_maps_unaug[0].arr.shape == (1, 1, 1)
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
