# Improve Performance of Elementwise Multiplication #615

This patch improves the performance of
`imgaug.augmenters.arithmetic.multiply_elementwise()`. The
performance improvement is roughly between 1.5x and 10x.
The effect is stronger for smaller images and denser
matrices of multipliers (i.e. `(H,W,C)` instead of `(H,W)`).
This change affects `MultiplyElementwise`.

Add functions:
* `imgaug.augmenters.arithmetic.multiply_elementwise_()`.
