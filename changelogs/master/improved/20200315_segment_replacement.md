# Improved Performance of Segment Replacement #640 #684

This patch improves the performance of segment
replacement (by average colors within the segments),
used in `Superpixels` and `segment_voronoi()`.
The new method is in some cases (especially small
images) up to 100x faster now. For 224x224 images
the speed improvement is around 1.4x to 10x,
depending on how many segments have to be replaced.

This change is expected to have a moderate positive
impact on `Superpixels` and `segment_voronoi()` (i.e.
`Voronoi`).

Added functions:
* `imgaug.augmenters.segmentation.replace_segments_`

Added classes:
* `imgaug.testutils.temporary_constants` (context)
