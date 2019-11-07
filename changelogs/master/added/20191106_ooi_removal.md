# Removal of Coordinate-Based Augmentables Outside of the Image Plane

* Added `BoundingBox.compute_out_of_image_area()`.
* Added `BoundingBox.compute_out_of_image_fraction()`.
* Added `Polygon.compute_out_of_image_area()`.
* Added `Polygon.compute_out_of_image_fraction()`.
* Added `LineString.compute_out_of_image_fraction()`.
* Changed `Polygon.area` to return `0.0` if the polygon contains less than
  three points (previously: exception).
