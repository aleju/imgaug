# Simplified Access to Coordinates and Items in Augmentables #495

* Added module `imgaug.augmentables.base`.
* Added interface `imgaug.augmentables.base.IAugmentable`, implemented by
  `HeatmapsOnImage`, `SegmentationMapsOnImage`, `KeypointsOnImage`,
  `BoundingBoxesOnImage`, `PolygonsOnImage` and `LineStringsOnImage`.
