# Fixed Affine Translation of Map-Data

* Fixed `Affine` producing unaligned augmentations between images and
  segmentation maps or heatmaps when using `translate_px` and the segmentation
  map or heatmap had a different height/width than corresponding image.
