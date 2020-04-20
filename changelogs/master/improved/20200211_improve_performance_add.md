# Improve Performance of `Add` #608

This patch improves the performance of
`imgaug.arithmetic.add_scalar()`
and the corresponding augmenter `Add` for `uint8` inputs.
The expected performance improvement is 1.5x to 6x.
(More for image arrays with higher widths/heights than
smaller ones. More for more channels. More for a single
scalar added as opposed to channelwise values.)

Add functions:
* `imgaug.augmenters.arithmetic.add_scalar_()`.
