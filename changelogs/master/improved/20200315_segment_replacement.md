# Improved Performance of Segment Replacement #640

This patch improves the performance of segment
replacement (by average colors within the segments),
used in `Superpixels` and `segment_voronoi()`.
The new method is up to around 7x faster, more for
smaller images and more segments. It can be slightly
slower in some cases for large images (512x512 and
larger).

This change seems to improve the overall performance
of `Superpixels` by a factor of around 1.1x to 1.4x
(more for smaller images).
It improves the overall performance of
`segment_voronoi()` by about 1.1x to 2.0x and can
reach much higher improvements in the case of very few
segments that have to be replaced.

Note that `segment_voronoi()` is used in `Voronoi`.

Added functions:
* `imgaug.augmenters.segmentation.replace_segments_`

Added classes:
* `imgaug.testutils.temporary_constants` (context)
