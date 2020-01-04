# Simplified Access to Coordinates and Items in Augmentables #495 #541

* Added module `imgaug.augmentables.base`.
* Added interface `imgaug.augmentables.base.IAugmentable`, implemented by
  `HeatmapsOnImage`, `SegmentationMapsOnImage`, `KeypointsOnImage`,
  `BoundingBoxesOnImage`, `PolygonsOnImage` and `LineStringsOnImage`.
* Added ability to iterate over coordinate-based `*OnImage` instances
  (keypoints, bounding boxes, polygons, line strings), e.g.
  `bbsoi = BoundingBoxesOnImage(bbs, shape=...); for bb in bbsoi: ...`.
  would iterate now over `bbs`.
* Added implementations of `__len__` methods to coordinate-based `*OnImage`
  instances, e.g.
  `bbsoi = BoundingBoxesOnImage(bbs, shape=...); print(len(bbsoi))`
  would now print the number of bounding boxes in `bbsoi`.
* Added ability to iterate over coordinates of `BoundingBox` (top-left,
  bottom-right), `Polygon` and `LineString` via `for xy in obj: ...`.
* Added ability to access coordinates of `BoundingBox`, `Polygon` and
  `LineString` using indices or slices, e.g. `line_string[1:]` to get an
  array of all coordinates except the first one.
* Added property `Keypoint.xy`.
* Added property `Keypoint.xy_int`.
