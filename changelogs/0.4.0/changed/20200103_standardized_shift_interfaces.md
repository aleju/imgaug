# Standardized `shift()` Interfaces of Coordinate-Based Augmentables #548

The interfaces for shift operations of all coordinate-based
augmentables (Keypoints, BoundingBoxes, LineStrings, Polygons)
were standardized. All of these augmentables have now the same
interface for shift operations. Previously, Keypoints used
a different interface (using `x` and `y` arguments) than the
other augmentables (using `top`, `right`, `bottom`, `left`
arguments). All augmentables use now the interface of Keypoints
as that is simpler and less ambiguous. Old arguments are still
accepted, but will produce deprecation warnings. Change the
arguments to `x` and `y` following `x=left-right` and
`y=top-bottom`.

**[breaking]** This breaks if one relied on calling `shift()` functions of
`BoundingBox`, `LineString`, `Polygon`, `BoundingBoxesOnImage`,
`LineStringsOnImage` or `PolygonsOnImage` without named arguments.
E.g. `bb = BoundingBox(...); bb_shifted = bb.shift(1, 2, 3, 4);`
will produce unexpected outputs now (equivalent to
`shift(x=1, y=2, top=3, right=4, bottom=0, left=0)`),
while `bb_shifted = bb.shift(top=1, right=2, bottom=3, left=4)` will still
work as expected.

* Added arguments `x`, `y` to `BoundingBox.shift()`, `LineString.shift()`
  and `Polygon.shift()`.
* Added arguments `x`, `y` to `BoundingBoxesOnImage.shift()`,
  `LineStringsOnImage.shift()` and `PolygonsOnImage.shift()`.
* Marked arguments `top`, `right`, `bottom`, `left` in
  `BoundingBox.shift()`, `LineString.shift()` and `Polygon.shift()`
  as deprecated. This also affects the corresponding `*OnImage`
  classes.
* Added function `testutils.wrap_shift_deprecation()`.
