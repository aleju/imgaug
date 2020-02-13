# Improve Performance of Alpha-Blending #???

This patch reworks `imgaug.augmenters.blend.blend_alpha()` to
improve its performance. In the case of a scalar constant alpha
value and both image inputs (foreground, background) being
`uint8`, the improved method is roughly 10x faster. For alpha
masks there should be a very minor performance improvement.
