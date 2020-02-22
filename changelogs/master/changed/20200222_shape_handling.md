# Stricter Shape Handling in Augmentables #623

Various methods of augmentables have so far accepted tuples
of integers or numpy arrays for `shape` parameters. This was
the case for e.g. `BoundingBoxesOnImage.__init__(bbs, shape)`
or `Polygon.clip_out_of_image(image)`. This tolerant handling
of shapes conveys some risk that an input is actually a
numpy representation of a shape, i.e. the equivalent of
`numpy.array(shape_tuple)`.

To decrease the risk of such an input leading to bugs, arrays
are no longer recommended inputs for `shape` in
`KeypointsOnImage.__init__`, `BoundingBoxesOnImage.__init__`,
`LineStringsOnImage.__init__`, and `PolygonsOnImage.__init__`.
Their usage in these methods will now raise a deprecation warning.

In all other methods of augmentables that currently accept
image-like numpy arrays and shape tuples for parameters,
only arrays that are 2-dimensional or 3-dimensional are from
now on accepted. Other arrays (e.g. 1-dimensional ones)
will be rejected with an assertion error.

Add functions:
* `imgaug.augmentables.utils.normalize_imglike_shape()`.

List of deprecations:
* `numpy.ndarray` as value of parameter `shape` in
  `KeypointsOnImage.__init__`.
* `numpy.ndarray` as value of parameter `shape` in
  `BoundingBoxesOnImage.__init__`.
* `numpy.ndarray` as value of parameter `shape` in
  `LineStringsOnImage.__init__`.
* `numpy.ndarray` as value of parameter `shape` in
  `PolygonsOnImage.__init__`.
