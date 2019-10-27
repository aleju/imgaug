# Improve Invert

* Improved performance of `imgaug.augmenters.arithmetic.invert()` and
  `imgaug.augmenters.arithmetic.Invert` for `uint8` images.
* Added function `imgaug.augmenters.arithmetic.invert_()`, an in-place version
  of `imgaug.augmenters.arithmetic.invert()`.
* Added parameters `threshold` and `invert_above_threshold` to
  `imgaug.augmenters.arithmetic.invert()`
* Added parameters `threshold` and `invert_above_threshold` to
  `imgaug.augmenters.arithmetic.Invert`.
* Added function `imgaug.augmenters.arithmetic.solarize()`, a wrapper around
  `solarize_()`.
* Added function `imgaug.augmenters.arithmetic.solarize_()`, a wrapper around
  `invert_()`.
* Added augmenter `imgaug.augmenters.Solarize`, a wrapper around `Invert`.
