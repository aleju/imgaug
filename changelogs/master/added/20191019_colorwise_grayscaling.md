# Colorwise Grayscaling #461

* Added `GrayscaleColorwise`, which applies grayscaling to randomly
  picked colors in the image.
* Added `RemoveSaturation`, a shortcut for `MultiplySaturation((0.0, 1.0))`
  with outputs similar to `Grayscale((0.0, 1.0))`.
