# Removal of Coordinate-Based Augmentables Outside of the Image Plane #487

* Added `Keypoint.is_out_of_image()`.

* Added `BoundingBox.compute_out_of_image_area()`.
* Added `Polygon.compute_out_of_image_area()`.

* Added `Keypoint.compute_out_of_image_fraction()`
* Added `BoundingBox.compute_out_of_image_fraction()`.
* Added `Polygon.compute_out_of_image_fraction()`.
* Added `LineString.compute_out_of_image_fraction()`.

* Added `KeypointsOnImage.remove_out_of_image_fraction()`.
* Added `BoundingBoxesOnImage.remove_out_of_image_fraction()`.
* Added `PolygonsOnImage.remove_out_of_image_fraction()`.
* Added `LineStringsOnImage.remove_out_of_image_fraction()`.

* Added `KeypointsOnImage.clip_out_of_image()`.

* Added `imgaug.augmenters.meta.RemoveCBAsByOutOfImageFraction`.
  Removes coordinate-based augmentables (e.g. BBs) that have at least a
  specified fraction of their area outside of the image plane.
* Added `imgaug.augmenters.meta.ClipCBAsToImagePlanes`.
  Clips off all parts from coordinate-based augmentables (e.g. BBs) that are
  outside of the corresponding image.

* Changed `Polygon.area` to return `0.0` if the polygon contains less than
  three points (previously: exception).
