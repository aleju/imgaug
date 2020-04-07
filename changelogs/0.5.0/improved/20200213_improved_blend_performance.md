# Improve Performance of Alpha-Blending #610

This patch reworks `imgaug.augmenters.blend.blend_alpha()` to
improve its performance. In the case of a scalar constant alpha
value and both image inputs (foreground, background) being
`uint8`, the improved method is roughly 10x faster. In the case
of one constant alpha value per channel, the expected speedup
is around 2x to 7x (more for larger images). In the case of
alpha maks, the expected speedup is around 1.3x to
2.0x (`(H,W)` masks are faster for larger images,
`(H,W,C)` the other way round).

Add functions:
* `imgaug.augmenters.blend.blend_alpha_()`
