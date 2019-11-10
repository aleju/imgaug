# Added Polygon Subdivision

* Added method `imgaug.augmentables.polys.Polygon.subdivide(N)`.
  The method increases the polygon's corner point count by interpolating
  `N` points on each edge with regular distance.
