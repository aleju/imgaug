# Improve Performance of `Add` #???

This patch improves the performance of
`imgaug.arithmetic.add_scalar()`
and the corresponding augmenter `Add` for `uint8` inputs.
The function and the augmenter should be roughly 2x as fast after
this update.

Add functions:
* `imgaug.augmenters.arithmetic.add_scalar_()`.
