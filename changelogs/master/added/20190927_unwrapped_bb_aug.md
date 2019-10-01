# Unwrapped Bounding Box Augmentation #446

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
* Added property `Keypoint.coords`.
* Added method `Keypoint.coords_almost_equals(other)`.
* Added method `Keypoint.almost_equals(other)`.
* Added method `imgaug.testutils.assert_cbaois_equal()`.
* Added method `imgaug.testutils.shift_cbaoi()`.
* Added internal `_augment_bounding_boxes()` methods to various augmenters.
  This allows to individually control how bounding boxes are supposed to
  be augmented. Previously, the bounding box augmentation was a wrapper around
  keypoint augmentation that did not allow such control.
* [breaking] Added parameter `parents` to `Augmenter.augment_bounding_boxes()`.
  This breaks if `hooks` was used as a *positional* argument in connection with
  that method.
* [breaking] Added parameter `func_bounding_boxes` to `Lambda`. This
  breaks if one relied on the order of the augmenter's parameters instead of
  their names.
* [breaking] Added parameter `func_bounding_boxes` to `AssertLambda`. This
  breaks if one relied on the order of the augmenter's parameters instead of
  their names.
* [breaking] Added parameter `check_bounding_boxes` to `AssertShape`. This
  breaks if one relied on the order of the augmenter's parameters instead of
  their names.
