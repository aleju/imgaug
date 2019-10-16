# Pooling Augmenters now affects Maps #457

Pooling augmenters were previously implemented so that they did not pool
the arrays of maps (i.e. heatmap arrays, segmentation map arrays). Only
the image shape saved within `HeatmapsOnImage.shape` and
`SegmentationMapsOnImage.shape` were updated. That was done because the library
can handle map arrays that are larger than the corresponding images and hence
no pooling was necessary for the augmentation to work correctly. This was now
changed and pooling augmenters will also pool map arrays
(if `keep_size=False`). The motiviation for this change is that the old
behaviour was unintuitive and inconsistent with other augmenters (e.g. `Crop`). 
