# Unwrapped Bounding Box Augmentation

* Added property `coords` to `BoundingBox`. The property returns an `(N,2)`
  numpy array containing the coordinates of the top-left and bottom-right
  bounding box corners.
* Added method `BoundingBox.coords_almost_equals(other)`.
