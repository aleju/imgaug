# Index-based Access to Coordinate-based `*OnImage` Instances #547

Enabled index-based access to coordinate-based `*OnImage` instances, i.e. to
`KeypointsOnImage`, `BoundingBoxesOnImage`, `LineStringsOnImage` and
`PolygonsOnImage`. This allows to do things like
`bbsoi = BoundingBoxesOnImage(...); bbs = bbsoi[0:2];`.

* Added `imgaug.augmentables.kps.KeypointsOnImage.__getitem__()`.
* Added `imgaug.augmentables.bbs.BoundingBoxesOnImage.__getitem__()`.
* Added `imgaug.augmentables.lines.LineStringsOnImage.__getitem__()`.
* Added `imgaug.augmentables.polys.PolygonsOnImage.__getitem__()`.
