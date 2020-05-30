# Improved Performance of Glass Blur #683

This patch improves the performance of
`imgaug.augmenters.imgcorruptplike.apply_glass_blur()`
and the corresponding augmenter. The improvement is
around 14x to 45x, depending on the image size
(larger images have more speedup).

Added dependencies:
* `numba`
