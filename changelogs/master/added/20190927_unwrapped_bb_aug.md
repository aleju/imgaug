# Unwrapped Bounding Box Augmentation

* Added property `coords` to `BoundingBox`. The property returns an `(N,2)`
  numpy array containing the coordinates of the top-left and bottom-right
  bounding box corners.
* Added method `BoundingBox.coords_almost_equals(other)`.
* Added method `BoundingBox.almost_equals(other)`.
* Changed method `Polygon.almost_equals(other)` to no longer verify the
  datatype. It is assumed now that the input is a Polygon.
* Added property `items` to `KeypointsOnImage`, `BoundingBoxesOnImage`,
  `PolygonsOnImage`, `LineStringsOnImage`. The property returns the
  keypoints/BBs/polygons/LineStrings contained by that instance.
* Added method `Polygon.coords_almost_equals(other)`. Alias for
  `Polygon.exterior_almost_equals(other)`.
* Added property `Polygon.coords`. Alias for `Polygon.exterior`.
