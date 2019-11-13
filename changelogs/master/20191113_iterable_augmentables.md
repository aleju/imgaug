# Simplified Access to Coordinates and Items in Augmentables #495

* Added module `imgaug.augmentables.base`.
* Added interface `imgaug.augmentables.base.IAugmentable`, implemented by
  `HeatmapsOnImage`, `SegmentationMapsOnImage`, `KeypointsOnImage`,
  `BoundingBoxesOnImage`, `PolygonsOnImage` and `LineStringsOnImage`.
* Added ability to iterate over coordinate-based `*OnImage` instances
  (keypoints, bounding boxes, polygons, line strings), e.g.
  `bbsoi = BoundingBoxesOnImage(bbs, shape=...); for bb in bbsoi: ...`.
  would iterate now over `bbs`.
* Added ability to iterate over coordinates of `BoundingBox` (top-left,
  bottom-right), `Polygon` and `LineString` via `for xy in obj: ...`.
