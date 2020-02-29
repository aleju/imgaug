# Improved Performance of `invert_()` #631

This patch improves the performance of
`imgaug.augmenters.arithmetic.invert_()` for `uint8`
images. The update is expected to improve the
performance by a factor of 4.5x to 5.3x (more for
smaller images) if no threshold is provided and by
1.5x to 2.7x (more for smaller images) if a threshold
is provided.

In both cases these improvements are only realised
if either no custom minimum and maximum for the
value range is provided or only a custom maximum
is provided. (This is expected to be the case for most
users.)

These improvements also affect `Invert` and `Solarize`.
