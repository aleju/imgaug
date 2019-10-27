# Improve Invert

* Improved performance of `imgaug.augmenters.arithmetic.invert()` and
  `imgaug.augmenters.arithmetic.Invert` for `uint8` images.
* Added function `imgaug.augmenters.arithmetic.invert_()`, an in-place version
  of `imgaug.augmenters.arithmetic.invert()`.
* Added parameters `threshold` and `invert_above_threshold` to
  `imgaug.augmenters.arithmetic.invert()`
