# Added in-place Methods for Coordinate-based Augmentables #532

* Added `Keypoint.project_()`.
* Added `Keypoint.shift_()`.    
* Added `KeypointsOnImage.on_()`.
* Added setter for `KeypontsOnImage.items`.
* Added setter for `BoundingBoxesOnImage.items`.
* Added setter for `LineStringsOnImage.items`.
* Added setter for `PolygonsOnImage.items`.
* Added `KeypointsOnImage.remove_out_of_image_fraction_()`.
* Added `KeypointsOnImage.clip_out_of_image_fraction_()`.
* Added `KeypointsOnImage.shift_()`.
* Added `BoundingBox.project_()`.
* Added `BoundingBox.extend_()`.
* Added `BoundingBox.clip_out_of_image_()`.
* Added `BoundingBox.shift_()`.
* Added `BoundingBoxesOnImage.on_()`.
* Added `BoundingBoxesOnImage.clip_out_of_image_()`.
* Added `BoundingBoxesOnImage.remove_out_of_image_()`.
* Added `BoundingBoxesOnImage.remove_out_of_image_fraction_()`.
* Added `BoundingBoxesOnImage.shift_()`.
* Added `imgaug.augmentables.utils.project_coords_()`.
* Added `LineString.project_()`.
* Added `LineString.shift_()`.
* Added `LineStringsOnImage.on_()`.
* Added `LineStringsOnImage.remove_out_of_image_()`.
* Added `LineStringsOnImage.remove_out_of_image_fraction_()`.
* Added `LineStringsOnImage.clip_out_of_image_()`.
* Added `LineStringsOnImage.shift_()`.
* Added `Polygon.project_()`.
* Added `Polygon.shift_()`.
* Added `Polygon.on_()`.
* Added `Polygon.subdivide_()`.
* Added `PolygonsOnImage.remove_out_of_image_()`.
* Added `PolygonsOnImage.remove_out_of_image_fraction_()`.
* Added `PolygonsOnImage.clip_out_of_image_()`.
* Added `PolygonsOnImage.shift_()`.
* Added `PolygonsOnImage.subdivide_()`.
* Switched `BoundingBoxesOnImage.copy()` to a custom copy operation (away
  from module `copy` module).
* Added parameters `bounding_boxes` and `shape` to
  BoundingBoxesOnImage.copy()`.
* Added parameters `bounding_boxes` and `shape` to
  BoundingBoxesOnImage.deepcopy()`.
* Switched `KeypointsOnImage.copy()` to a custom copy operation (away
  from module `copy` module).
* Switched `PolygonsOnImage.copy()` to a custom copy operation (away
  from module `copy` module).
* Added parameters `polygons` and `shape` to
  PolygonsOnImage.copy()`.
* Added parameters `polygons` and `shape` to
  PolygonsOnImage.deepcopy()`.
* Switched augmenters to use in-place functions for keypoints,
  bounding boxes, line strings and polygons.
