# Improve Performance of Elementwise Addition #???

This patch improves the performance of
`imgaug.augmenters.arithmetic.add_elementwise()` for `uint8`
images. The performance increases is expected to be between
roughly 1.5x and 5x (more for very dense `value` matrices,
i.e. for channelwise addition). This change affects
`AddElementwise`, `AdditiveGaussianNoise`,
`AdditiveLaplaceNoise` and `AdditivePoissonNoise`.
