# Improve Performance of `multiply_scalar()` #???

This patch improves the performance of
`imgaug.augmenters.arithmetic.multiply_scalar()`. The function
is now roughly up to 8x faster for small images (32x32x3),
marginally faster at about 1.05x for large images (224x224x3,
increase of ~1.05) and decently faster at about 1.35x for very
large images (512x512x3, increase of ~1.35x).
This change affects `Multiply`.

Add functions:
* `imgaug.augmenters.arithmetic.multiply_scalar_()`.
