# Improve Performance of `multiply_scalar()` #614

This patch improves the performance of
`imgaug.augmenters.arithmetic.multiply_scalar()` for
`uint8` inputs. The function is now between 1.2x and 7x
faster (more for smaller images and channelwise
multipliers). This change affects `Multiply`.

Add functions:
* `imgaug.augmenters.arithmetic.multiply_scalar_()`.
