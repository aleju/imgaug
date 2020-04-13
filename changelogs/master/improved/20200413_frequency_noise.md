# Improve Performance of `FrequencyNoise` #???

This patch improves the performance of
`imgaug.parameters.FrequencyNoise`, which is used in some
weather augmenters. The parameter now samples `HxW` arrays
about 1.3x to 1.5x faster (more improvement for larger
images).
