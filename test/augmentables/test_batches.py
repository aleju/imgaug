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
import imgaug.augmenters as iaa
from imgaug.testutils import reseed


ATTR_NAMES = ["images", "heatmaps", "segmentation_maps", "keypoints",
              "bounding_boxes", "polygons", "line_strings"]


# TODO test __init__()
class TestUnnormalizedBatch(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_get_column_names__only_images(self):
        batch = ia.UnnormalizedBatch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8)
        )

        names = batch.get_column_names()

        assert names == ["images"]

    def test_get_column_names__all_columns(self):
        batch = ia.UnnormalizedBatch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            heatmaps=[np.zeros((2, 2, 1), dtype=np.float32)],
            segmentation_maps=[np.zeros((2, 2, 1), dtype=np.int32)],
            keypoints=[[(0, 0)]],
            bounding_boxes=[[ia.BoundingBox(0, 0, 1, 1)]],
            polygons=[[ia.Polygon([(0, 0), (1, 0), (1, 1)])]],
            line_strings=[[ia.LineString([(0, 0), (1, 0)])]]
        )

        names = batch.get_column_names()

        assert names == ["images", "heatmaps", "segmentation_maps",
                         "keypoints", "bounding_boxes", "polygons",
                         "line_strings"]

    def test_to_normalized_batch__only_images(self):
        batch = ia.UnnormalizedBatch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8)
        )

        batch_norm = batch.to_normalized_batch()

        assert isinstance(batch_norm, ia.Batch)
        assert ia.is_np_array(batch_norm.images)
        assert batch_norm.images_unaug.shape == (1, 2, 2, 3)
        assert batch_norm.get_column_names() == ["images"]

    def test_to_normalized_batch__all_columns(self):
        batch = ia.UnnormalizedBatch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            heatmaps=[np.zeros((2, 2, 1), dtype=np.float32)],
            segmentation_maps=[np.zeros((2, 2, 1), dtype=np.int32)],
            keypoints=[[(0, 0)]],
            bounding_boxes=[[ia.BoundingBox(0, 0, 1, 1)]],
            polygons=[[ia.Polygon([(0, 0), (1, 0), (1, 1)])]],
            line_strings=[[ia.LineString([(0, 0), (1, 0)])]]
        )

        batch_norm = batch.to_normalized_batch()

        assert isinstance(batch_norm, ia.Batch)
        assert ia.is_np_array(batch_norm.images)
        assert batch_norm.images_unaug.shape == (1, 2, 2, 3)
        assert isinstance(batch_norm.heatmaps_unaug[0], ia.HeatmapsOnImage)
        assert isinstance(batch_norm.segmentation_maps_unaug[0],
                          ia.SegmentationMapsOnImage)
        assert isinstance(batch_norm.keypoints_unaug[0], ia.KeypointsOnImage)
        assert isinstance(batch_norm.bounding_boxes_unaug[0],
                          ia.BoundingBoxesOnImage)
        assert isinstance(batch_norm.polygons_unaug[0], ia.PolygonsOnImage)
        assert isinstance(batch_norm.line_strings_unaug[0],
                          ia.LineStringsOnImage)
        assert batch_norm.get_column_names() == [
            "images", "heatmaps", "segmentation_maps", "keypoints",
            "bounding_boxes", "polygons", "line_strings"]

    def test_fill_from_augmented_normalized_batch(self):
        batch = ia.UnnormalizedBatch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            heatmaps=[np.zeros((2, 2, 1), dtype=np.float32)],
            segmentation_maps=[np.zeros((2, 2, 1), dtype=np.int32)],
            keypoints=[[(0, 0)]],
            bounding_boxes=[[ia.BoundingBox(0, 0, 1, 1)]],
            polygons=[[ia.Polygon([(0, 0), (1, 0), (1, 1)])]],
            line_strings=[[ia.LineString([(0, 0), (1, 0)])]]
        )
        batch_norm = ia.Batch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            heatmaps=[
                ia.HeatmapsOnImage(
                    np.zeros((2, 2, 1), dtype=np.float32),
                    shape=(2, 2, 3)
                )
            ],
            segmentation_maps=[
                ia.SegmentationMapsOnImage(
                    np.zeros((2, 2, 1), dtype=np.int32),
                    shape=(2, 2, 3)
                )
            ],
            keypoints=[
                ia.KeypointsOnImage(
                    [ia.Keypoint(0, 0)],
                    shape=(2, 2, 3)
                )
            ],
            bounding_boxes=[
                ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(0, 0, 1, 1)],
                    shape=(2, 2, 3)
                )
            ],
            polygons=[
                ia.PolygonsOnImage(
                    [ia.Polygon([(0, 0), (1, 0), (1, 1)])],
                    shape=(2, 2, 3)
                )
            ],
            line_strings=[
                ia.LineStringsOnImage(
                    [ia.LineString([(0, 0), (1, 0)])],
                    shape=(2, 2, 3)
                )
            ]
        )
        batch_norm.images_aug = batch_norm.images_unaug
        batch_norm.heatmaps_aug = batch_norm.heatmaps_unaug
        batch_norm.segmentation_maps_aug = batch_norm.segmentation_maps_unaug
        batch_norm.keypoints_aug = batch_norm.keypoints_unaug
        batch_norm.bounding_boxes_aug = batch_norm.bounding_boxes_unaug
        batch_norm.polygons_aug = batch_norm.polygons_unaug
        batch_norm.line_strings_aug = batch_norm.line_strings_unaug

        batch = batch.fill_from_augmented_normalized_batch(batch_norm)

        assert batch.images_aug.shape == (1, 2, 2, 3)
        assert ia.is_np_array(batch.heatmaps_aug[0])
        assert ia.is_np_array(batch.segmentation_maps_aug[0])
        assert batch.keypoints_aug[0][0] == (0, 0)
        assert batch.bounding_boxes_aug[0][0].x1 == 0
        assert batch.polygons_aug[0][0].exterior[0][0] == 0
        assert batch.line_strings_aug[0][0].coords[0][0] == 0


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

    def test_get_column_names__only_images(self):
        batch = ia.Batch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8)
        )

        names = batch.get_column_names()

        assert names == ["images"]

    def test_get_column_names__all_columns(self):
        batch = ia.Batch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            heatmaps=[np.zeros((2, 2, 1), dtype=np.float32)],
            segmentation_maps=[np.zeros((2, 2, 1), dtype=np.int32)],
            keypoints=[
                ia.KeypointsOnImage(
                    [ia.Keypoint(x=0, y=0)],
                    shape=(2, 2, 3)
                )
            ],
            bounding_boxes=[
                ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(0, 0, 1, 1)],
                    shape=(2, 2, 3)
                )
            ],
            polygons=[
                ia.PolygonsOnImage(
                    [ia.Polygon([(0, 0), (1, 0), (1, 1)])],
                    shape=(2, 2, 3)
                )
            ],
            line_strings=[
                ia.LineStringsOnImage(
                    [ia.LineString([(0, 0), (1, 0)])],
                    shape=(2, 2, 3)
                )
            ]
        )

        names = batch.get_column_names()

        assert names == ["images", "heatmaps", "segmentation_maps",
                         "keypoints", "bounding_boxes", "polygons",
                         "line_strings"]

    def test_to_normalized_batch(self):
        batch = ia.Batch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8)
        )

        batch_norm = batch.to_normalized_batch()

        assert batch_norm is batch

    def test_to_batch_in_augmentation__only_images(self):
        batch = ia.Batch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8)
        )

        batch_inaug = batch.to_batch_in_augmentation()

        assert isinstance(batch_inaug, ia.BatchInAugmentation)
        assert ia.is_np_array(batch_inaug.images)
        assert batch_inaug.images.shape == (1, 2, 2, 3)
        assert batch_inaug.get_column_names() == ["images"]

    def test_to_batch_in_augmentation__all_columns(self):
        batch = ia.Batch(
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            heatmaps=[
                ia.HeatmapsOnImage(
                    np.zeros((2, 2, 1), dtype=np.float32),
                    shape=(2, 2, 3)
                )
            ],
            segmentation_maps=[
                ia.SegmentationMapsOnImage(
                    np.zeros((2, 2, 1), dtype=np.int32),
                    shape=(2, 2, 3)
                )
            ],
            keypoints=[
                ia.KeypointsOnImage(
                    [ia.Keypoint(x=0, y=0)],
                    shape=(2, 2, 3)
                )
            ],
            bounding_boxes=[
                ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(0, 0, 1, 1)],
                    shape=(2, 2, 3)
                )
            ],
            polygons=[
                ia.PolygonsOnImage(
                    [ia.Polygon([(0, 0), (1, 0), (1, 1)])],
                    shape=(2, 2, 3)
                )
            ],
            line_strings=[
                ia.LineStringsOnImage(
                    [ia.LineString([(0, 0), (1, 0)])],
                    shape=(2, 2, 3)
                )
            ]
        )

        batch_inaug = batch.to_batch_in_augmentation()

        assert isinstance(batch_inaug, ia.BatchInAugmentation)
        assert ia.is_np_array(batch_inaug.images)
        assert batch_inaug.images.shape == (1, 2, 2, 3)
        assert isinstance(batch_inaug.heatmaps[0], ia.HeatmapsOnImage)
        assert isinstance(batch_inaug.segmentation_maps[0],
                          ia.SegmentationMapsOnImage)
        assert isinstance(batch_inaug.keypoints[0], ia.KeypointsOnImage)
        assert isinstance(batch_inaug.bounding_boxes[0],
                          ia.BoundingBoxesOnImage)
        assert isinstance(batch_inaug.polygons[0], ia.PolygonsOnImage)
        assert isinstance(batch_inaug.line_strings[0], ia.LineStringsOnImage)
        assert batch_inaug.get_column_names() == [
            "images", "heatmaps", "segmentation_maps", "keypoints",
            "bounding_boxes", "polygons", "line_strings"]

    def test_fill_from_batch_in_augmentation(self):
        batch = ia.Batch(images=1)
        batch_inaug = ia.BatchInAugmentation(
            images=2,
            heatmaps=3,
            segmentation_maps=4,
            keypoints=5,
            bounding_boxes=6,
            polygons=7,
            line_strings=8
        )

        batch = batch.fill_from_batch_in_augmentation_(batch_inaug)

        assert batch.images_aug == 2
        assert batch.heatmaps_aug == 3
        assert batch.segmentation_maps_aug == 4
        assert batch.keypoints_aug == 5
        assert batch.bounding_boxes_aug == 6
        assert batch.polygons_aug == 7
        assert batch.line_strings_aug == 8

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


# TODO test __init__
#      test apply_propagation_hooks_
#      test invert_apply_propagation_hooks_
class TestBatchInAugmentation(unittest.TestCase):
    def test_empty__all_columns_none(self):
        batch = ia.BatchInAugmentation()
        assert batch.empty

    def test_empty__with_columns_set(self):
        kwargs = [
            {"images": [2]},
            {"heatmaps": [3]},
            {"segmentation_maps": [4]},
            {"keypoints": [5]},
            {"bounding_boxes": [6]},
            {"polygons": [7]},
            {"line_strings": [8]}
        ]
        for kwargs_i in kwargs:
            batch = ia.BatchInAugmentation(**kwargs_i)
            assert not batch.empty

    def test_nb_rows__when_empty(self):
        batch = ia.BatchInAugmentation()
        assert batch.nb_rows == 0

    def test_nb_rows__with_empty_column(self):
        batch = ia.BatchInAugmentation(images=[])
        assert batch.nb_rows == 0

    def test_nb_rows__with_columns_set(self):
        kwargs = [
            {"images": [0]},
            {"heatmaps": [0]},
            {"segmentation_maps": [0]},
            {"keypoints": [0]},
            {"bounding_boxes": [0]},
            {"polygons": [0]},
            {"line_strings": [0]}
        ]
        for kwargs_i in kwargs:
            batch = ia.BatchInAugmentation(**kwargs_i)
            assert batch.nb_rows == 1

    def test_nb_rows__with_two_columns(self):
        batch = ia.BatchInAugmentation(images=[0, 0], keypoints=[0, 0])
        assert batch.nb_rows == 2

    def test_columns__when_empty(self):
        batch = ia.BatchInAugmentation()
        assert len(batch.columns) == 0

    def test_columns__with_empty_column(self):
        batch = ia.BatchInAugmentation(images=[])

        columns = batch.columns

        assert len(columns) == 0

    def test_columns__with_columns_set(self):
        kwargs = [
            {"images": [0]},
            {"heatmaps": [0]},
            {"segmentation_maps": [0]},
            {"keypoints": [0]},
            {"bounding_boxes": [0]},
            {"polygons": [0]},
            {"line_strings": [0]}
        ]
        for kwargs_i in kwargs:
            batch = ia.BatchInAugmentation(**kwargs_i)
            columns = batch.columns
            assert len(columns) == 1
            assert columns[0].name == list(kwargs_i.keys())[0]

    def test_columns__with_two_columns(self):
        batch = ia.BatchInAugmentation(images=[0, 0], keypoints=[1, 1])

        columns = batch.columns

        assert len(columns) == 2
        assert columns[0].name == "images"
        assert columns[1].name == "keypoints"
        assert columns[0].value == [0, 0]
        assert columns[1].value == [1, 1]

    def test_get_column_names__with_two_columns(self):
        batch = ia.BatchInAugmentation(images=[0, 0], keypoints=[1, 1])
        assert batch.get_column_names() == ["images", "keypoints"]

    def test_get_rowwise_shapes__images_is_single_array(self):
        batch = ia.BatchInAugmentation(images=np.zeros((2, 3, 4, 1)))
        shapes = batch.get_rowwise_shapes()
        assert shapes == [(3, 4, 1), (3, 4, 1)]

    def test_get_rowwise_shapes__images_is_multiple_arrays(self):
        batch = ia.BatchInAugmentation(
            images=[np.zeros((3, 4, 1)), np.zeros((4, 5, 1))]
        )
        shapes = batch.get_rowwise_shapes()
        assert shapes == [(3, 4, 1), (4, 5, 1)]

    def test_get_rowwise_shapes__nonimages(self):
        heatmaps = [
            ia.HeatmapsOnImage(
                np.zeros((1, 2, 1), dtype=np.float32),
                shape=(1, 2, 3))
        ]
        segmaps = [
            ia.SegmentationMapsOnImage(
                np.zeros((1, 2, 1), dtype=np.int32),
                shape=(1, 2, 3))
        ]
        keypoints = [
            ia.KeypointsOnImage(
                [ia.Keypoint(0, 0)],
                shape=(1, 2, 3))
        ]
        bounding_boxes = [
            ia.BoundingBoxesOnImage(
                [ia.BoundingBox(0, 1, 2, 3)],
                shape=(1, 2, 3)
            )
        ]
        polygons = [
            ia.PolygonsOnImage(
                [ia.Polygon([(0, 0), (1, 0), (1, 1)])],
                shape=(1, 2, 3)
            )
        ]
        line_strings = [
            ia.LineStringsOnImage(
                [ia.LineString([(0, 0), (1, 0)])],
                shape=(1, 2, 3)
            )
        ]

        kwargs = [
            {"heatmaps": heatmaps},
            {"segmentation_maps": segmaps},
            {"keypoints": keypoints},
            {"bounding_boxes": bounding_boxes},
            {"polygons": polygons},
            {"line_strings": line_strings}
        ]
        for kwargs_i in kwargs:
            batch = ia.BatchInAugmentation(**kwargs_i)
            shapes = batch.get_rowwise_shapes()
            assert shapes == [(1, 2, 3)]

    def test_subselect_rows_by_indices__none_selected(self):
        batch = ia.BatchInAugmentation(
            images=np.zeros((3, 3, 4, 1)),
            keypoints=[
                ia.KeypointsOnImage(
                    [ia.Keypoint(0, 0)],
                    shape=(3, 4, 1)
                ),
                ia.KeypointsOnImage(
                    [ia.Keypoint(1, 1)],
                    shape=(3, 4, 1)
                ),
                ia.KeypointsOnImage(
                    [ia.Keypoint(2, 2)],
                    shape=(3, 4, 1)
                )
            ]
        )

        batch_sub = batch.subselect_rows_by_indices([])

        assert batch_sub.images is None
        assert batch_sub.keypoints is None

    def test_subselect_rows_by_indices__two_of_three_selected(self):
        batch = ia.BatchInAugmentation(
            images=np.zeros((3, 3, 4, 1)),
            keypoints=[
                ia.KeypointsOnImage(
                    [ia.Keypoint(0, 0)],
                    shape=(3, 4, 1)
                ),
                ia.KeypointsOnImage(
                    [ia.Keypoint(1, 1)],
                    shape=(3, 4, 1)
                ),
                ia.KeypointsOnImage(
                    [ia.Keypoint(2, 2)],
                    shape=(3, 4, 1)
                )
            ]
        )

        batch_sub = batch.subselect_rows_by_indices([0, 2])

        assert batch_sub.images.shape == (2, 3, 4, 1)
        assert batch_sub.keypoints[0].keypoints[0].x == 0
        assert batch_sub.keypoints[0].keypoints[0].y == 0
        assert batch_sub.keypoints[1].keypoints[0].x == 2
        assert batch_sub.keypoints[1].keypoints[0].y == 2

    def test_invert_subselect_rows_by_indices__none_selected(self):
        images = np.zeros((3, 3, 4, 1), dtype=np.uint8)
        images[0, ...] = 0
        images[1, ...] = 1
        images[2, ...] = 2
        batch = ia.BatchInAugmentation(
            images=images,
            keypoints=[
                ia.KeypointsOnImage(
                    [ia.Keypoint(0, 0)],
                    shape=(3, 4, 1)
                ),
                ia.KeypointsOnImage(
                    [ia.Keypoint(1, 1)],
                    shape=(3, 4, 1)
                ),
                ia.KeypointsOnImage(
                    [ia.Keypoint(2, 2)],
                    shape=(3, 4, 1)
                )
            ]
        )

        batch_sub = batch.subselect_rows_by_indices([])
        batch_inv = batch.invert_subselect_rows_by_indices_([], batch_sub)

        assert batch_inv.images.shape == (3, 3, 4, 1)
        assert np.max(batch_inv.images[0]) == 0
        assert np.max(batch_inv.images[1]) == 1
        assert np.max(batch_inv.images[2]) == 2
        assert batch_inv.keypoints[0].keypoints[0].x == 0
        assert batch_inv.keypoints[0].keypoints[0].y == 0
        assert batch_inv.keypoints[1].keypoints[0].x == 1
        assert batch_inv.keypoints[1].keypoints[0].y == 1
        assert batch_inv.keypoints[2].keypoints[0].x == 2
        assert batch_inv.keypoints[2].keypoints[0].y == 2

    def test_invert_subselect_rows_by_indices__two_of_three_selected(self):
        images = np.zeros((3, 3, 4, 1), dtype=np.uint8)
        images[0, ...] = 0
        images[1, ...] = 1
        images[2, ...] = 2
        batch = ia.BatchInAugmentation(
            images=images,
            keypoints=[
                ia.KeypointsOnImage(
                    [ia.Keypoint(0, 0)],
                    shape=(3, 4, 1)
                ),
                ia.KeypointsOnImage(
                    [ia.Keypoint(1, 1)],
                    shape=(3, 4, 1)
                ),
                ia.KeypointsOnImage(
                    [ia.Keypoint(2, 2)],
                    shape=(3, 4, 1)
                )
            ]
        )

        batch_sub = batch.subselect_rows_by_indices([0, 2])
        batch_sub.images[0, ...] = 10
        batch_sub.images[1, ...] = 20
        batch_sub.keypoints[0].keypoints[0].x = 10
        batch_inv = batch.invert_subselect_rows_by_indices_([0, 2], batch_sub)

        assert batch_inv.images.shape == (3, 3, 4, 1)
        assert np.max(batch_inv.images[0]) == 10
        assert np.max(batch_inv.images[1]) == 1
        assert np.max(batch_inv.images[2]) == 20
        assert batch_inv.keypoints[0].keypoints[0].x == 10
        assert batch_inv.keypoints[0].keypoints[0].y == 0
        assert batch_inv.keypoints[1].keypoints[0].x == 1
        assert batch_inv.keypoints[1].keypoints[0].y == 1
        assert batch_inv.keypoints[2].keypoints[0].x == 2
        assert batch_inv.keypoints[2].keypoints[0].y == 2

    def test_propagation_hooks_ctx(self):
        def propagator(images, augmenter, parents, default):
            if ia.is_np_array(images):
                return False
            else:
                return True

        hooks = ia.HooksImages(propagator=propagator)

        batch = ia.BatchInAugmentation(
            images=np.zeros((3, 3, 4, 1), dtype=np.uint8),
            keypoints=[
                ia.KeypointsOnImage(
                    [ia.Keypoint(0, 0)],
                    shape=(3, 4, 1)
                ),
                ia.KeypointsOnImage(
                    [ia.Keypoint(1, 1)],
                    shape=(3, 4, 1)
                ),
                ia.KeypointsOnImage(
                    [ia.Keypoint(2, 2)],
                    shape=(3, 4, 1)
                )
            ]
        )

        with batch.propagation_hooks_ctx(iaa.Noop(), hooks, []) as batch_prop:
            assert batch_prop.images is None
            assert batch_prop.keypoints is not None
            assert len(batch_prop.keypoints) == 3

            batch_prop.keypoints[0].keypoints[0].x = 10

        assert batch.images is not None
        assert batch.keypoints is not None
        assert batch.keypoints[0].keypoints[0].x == 10

    def test_to_batch_in_augmentation(self):
        batch = ia.BatchInAugmentation(images=1)
        batch_inaug = batch.to_batch_in_augmentation()
        assert batch_inaug is batch

    def test_fill_from_batch_in_augmentation(self):
        batch = ia.BatchInAugmentation(images=1)
        batch_inaug = ia.BatchInAugmentation(
            images=2,
            heatmaps=3,
            segmentation_maps=4,
            keypoints=5,
            bounding_boxes=6,
            polygons=7,
            line_strings=8
        )

        batch = batch.fill_from_batch_in_augmentation_(batch_inaug)

        assert batch.images == 2
        assert batch.heatmaps == 3
        assert batch.segmentation_maps == 4
        assert batch.keypoints == 5
        assert batch.bounding_boxes == 6
        assert batch.polygons == 7
        assert batch.line_strings == 8

    def test_to_batch(self):
        batch_before_aug = ia.Batch()
        batch_before_aug.images_unaug = 0
        batch_before_aug.heatmaps_unaug = 1
        batch_before_aug.segmentation_maps_unaug = 2
        batch_before_aug.keypoints_unaug = 3
        batch_before_aug.bounding_boxes_unaug = 4
        batch_before_aug.polygons_unaug = 5
        batch_before_aug.line_strings_unaug = 6

        batch_inaug = ia.BatchInAugmentation(
            images=10,
            heatmaps=20,
            segmentation_maps=30,
            keypoints=40,
            bounding_boxes=50,
            polygons=60,
            line_strings=70
        )

        batch = batch_inaug.to_batch(batch_before_aug)

        assert batch.images_unaug == 0
        assert batch.heatmaps_unaug == 1
        assert batch.segmentation_maps_unaug == 2
        assert batch.keypoints_unaug == 3
        assert batch.bounding_boxes_unaug == 4
        assert batch.polygons_unaug == 5
        assert batch.line_strings_unaug == 6

        assert batch.images_aug == 10
        assert batch.heatmaps_aug == 20
        assert batch.segmentation_maps_aug == 30
        assert batch.keypoints_aug == 40
        assert batch.bounding_boxes_aug == 50
        assert batch.polygons_aug == 60
        assert batch.line_strings_aug == 70

    def test_deepcopy(self):
        batch = ia.BatchInAugmentation(
            images=np.full((1,), 0, dtype=np.uint8),
            heatmaps=np.full((1,), 1, dtype=np.uint8),
            segmentation_maps=np.full((1,), 2, dtype=np.uint8),
            keypoints=np.full((1,), 3, dtype=np.uint8),
            bounding_boxes=np.full((1,), 4, dtype=np.uint8),
            polygons=np.full((1,), 5, dtype=np.uint8),
            line_strings=np.full((1,), 6, dtype=np.uint8)
        )

        batch_copy = batch.deepcopy()

        assert np.max(batch_copy.images) == 0
        assert np.max(batch_copy.heatmaps) == 1
        assert np.max(batch_copy.segmentation_maps) == 2
        assert np.max(batch_copy.keypoints) == 3
        assert np.max(batch_copy.bounding_boxes) == 4
        assert np.max(batch_copy.polygons) == 5
        assert np.max(batch_copy.line_strings) == 6
