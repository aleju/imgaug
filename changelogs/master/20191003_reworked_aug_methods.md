# Reworked Augmentation Methods #451

* Added method `to_normalized_batch()` to `imgaug.augmentables.batches.Batch`
  to have the same interface in `Batch` and `UnnormalizedBatch`.
* Added method `get_augmentable()` to
  `imgaug.augmentables.batches.Batch` and
  `imgaug.augmentables.batches.UnnormalizedBatch`.
* Added method `get_augmentable_names()` to
  `imgaug.augmentables.batches.Batch` and
  `imgaug.augmentables.batches.UnnormalizedBatch`.
* Added method `to_batch_in_augmentation()` to
  `imgaug.augmentables.batches.Batch`.
* Added method `fill_from_batch_in_augmentation_()` to
  `imgaug.augmentables.batches.Batch`.
* Added class `imgaug.augmentables.batches.BatchInAugmentation`.
* Added method `_augment_batch()` in `imgaug.augmenters.meta.Augmenter`.
  This method is now called from `augment_batch()`.
* Changed `augment_images()` in `imgaug.augmenters.meta.Augmenter` to be
  a thin wrapper around `augment_batch()`.
* Changed `augment_heatmaps()` in `imgaug.augmenters.meta.Augmenter` to be
  a thin wrapper around `augment_batch()`.
* Changed `augment_segmentation_maps()` in `imgaug.augmenters.meta.Augmenter`
  to be a thin wrapper around `augment_batch()`.
* Changed `augment_keypoints()` in `imgaug.augmenters.meta.Augmenter` to be
  a thin wrapper around `augment_batch()`.
* Changed `augment_bounding_boxes()` in `imgaug.augmenters.meta.Augmenter` to be
  a thin wrapper around `augment_batch()`.
* Changed `augment_polygons()` in `imgaug.augmenters.meta.Augmenter` to be
  a thin wrapper around `augment_batch()`.
* Changed `augment_line_strings()` in `imgaug.augmenters.meta.Augmenter` to be
  a thin wrapper around `augment_batch()`.
* Changed `augment_image()`, `augment_images()`, `augment_heatmaps()`,
  `augment_segmentation_maps()`, `augment_keypoints()`,
  `augment_bounding_boxes()`, `augment_polygons()` and `augment_line_strings()`
  to return `None` inputs without change. Previously they resulted in an
  exception. This is more consistent with the behaviour in the other
  `augment_*` methods.
* [breaking] Added parameter `parents` to
  `imgaug.augmenters.meta.Augmenter.augment_batch()`. This breaks if one relied
  on the order of arguments for that methods.
* Changed `augment_images()` to no longer be abstract. It defaults
  to not changing the input images.
* Refactored `Sequential` to use single `_augment_batch()` method.
* Refactored `SomeOf` to use single `_augment_batch()` method.
* Refactored `Sometimes` to use single `_augment_batch()` method.
* Refactored `AveragePooling`, `MaxPooling`, `MinPooling`, `MedianPooling`
  to use single `_augment_batch()` method.
* Refactored `ElasticTransformation` to use single `_augment_batch()` method.
* Refactored `Alpha` to use single `_augment_batch()` method.
* Refactored `AlphaElementwise` to use single `_augment_batch()` method.
* Refactored `WithColorspace` to use single `_augment_batch()` method.
* Refactored `WithHueAndSaturation` to use single `_augment_batch()` method.
* Refactored `Fliplr` to use single `_augment_batch()` method.
* Refactored `Flipud` to use single `_augment_batch()` method.
* Refactored `Affine` to use single `_augment_batch()` method.
* Refactored `Rot90` to use single `_augment_batch()` method.
* Refactored `Resize` to use single `_augment_batch()` method.
* Refactored `CropAndPad` to use single `_augment_batch()` method.
* Refactored `PadToFixedSize` to use single `_augment_batch()` method.
* Refactored `CropToFixedSize` to use single `_augment_batch()` method.
* Refactored `KeepSizeByResize` to use single `_augment_batch()` method.
* Refactored `PiecewiseAffine` to use single `_augment_batch()` method.
* Refactored `PerspectiveTransform` to use single `_augment_batch()` method.
* Refactored `WithChannels` to use single `_augment_batch()` method.
* Refactored `Add` to use single `_augment_batch()` method.
* Refactored `AddElementwise` to use single `_augment_batch()` method.
* Refactored `Multiply` to use single `_augment_batch()` method.
* Refactored `MultiplyElementwise` to use single `_augment_batch()` method.
* Refactored `ReplaceElementwise` to use single `_augment_batch()` method.
* Refactored `Invert` to use single `_augment_batch()` method.
* Refactored `JpegCompression` to use single `_augment_batch()` method.
* Refactored `GaussianBlur` to use single `_augment_batch()` method.
* Refactored `AverageBlur` to use single `_augment_batch()` method.
* Refactored `MedianBlur` to use single `_augment_batch()` method.
* Refactored `BilateralBlur` to use single `_augment_batch()` method.
* Refactored `AddToHueAndSaturation` to use single `_augment_batch()` method.
* Refactored `ChangeColorspace` to use single `_augment_batch()` method.
* Refactored `_AbstractColorQuantization` to use single `_augment_batch()`
  method.
* Refactored `_ContrastFuncWrapper` to use single `_augment_batch()` method.
* Refactored `AllChannelsCLAHE` to use single `_augment_batch()` method.
* Refactored `CLAHE` to use single `_augment_batch()` method.
* Refactored `AllChannelsHistogramEqualization` to use single `_augment_batch()`
  method.
* Refactored `HistogramEqualization` to use single `_augment_batch()` method.
* Refactored `Convolve` to use single `_augment_batch()` method.
* Refactored `Canny` to use single `_augment_batch()` method.
* Refactored `ChannelShuffle` to use single `_augment_batch()` method.
* Refactored `Superpixels` to use single `_augment_batch()` method.
* Refactored `Voronoi` to use single `_augment_batch()` method.
* Refactored `FastSnowyLandscape` to use single `_augment_batch()` method.
* Refactored `CloudLayer` to use single `_augment_batch()` method.
* Refactored `SnowflakesLayer` to use single `_augment_batch()` method.
* Added validation of input arguments to `KeypointsOnImage.from_xy_array()`.
* Improved validation of input arguments to
  `BoundingBoxesOnImage.from_xyxy_array()`.
* Added method `BoundingBoxesOnImage.to_keypoints_on_image()`.
* Added method `PolygonsOnImage.to_keypoints_on_image()`.
* Added method `LineStringsOnImage.to_keypoints_on_image()`.
* Added method `KeypointsOnImage.to_keypoints_on_image()`.
* Added method `BoundingBoxesOnImage.invert_to_keypoints_on_image_()`.
* Added method `PolygonsOnImage.invert_to_keypoints_on_image_()`.
* Added method `LineStringsOnImage.invert_to_keypoints_on_image_()`.
* Added method `KeypointsOnImage.invert_to_keypoints_on_image_()`.
* Added method `imgaug.augmentables.polys.recover_psois_()`.
* Added method `imgaug.augmentables.utils.convert_cbaois_to_kpsois()`.
* Added method `imgaug.augmentables.utils.invert_convert_cbaois_to_kpsois_()`.
* Added method `imgaug.augmentables.utils.deepcopy_fast()`.
* Added method `imgaug.augmentables.kps.BoundingBoxesOnImage.to_xy_array()`.
* Added method `imgaug.augmentables.kps.PolygonsOnImage.to_xy_array()`.
* Added method `imgaug.augmentables.kps.LineStringsOnImage.to_xy_array()`.
* Added method `imgaug.augmentables.kps.KeypointsOnImage.fill_from_xy_array_()`.
* Added method `imgaug.augmentables.kps.BoundingBoxesOnImage.fill_from_xy_array_()`.
* Added method `imgaug.augmentables.kps.PolygonsOnImage.fill_from_xy_array_()`.
* Added method `imgaug.augmentables.kps.LineStringsOnImage.fill_from_xy_array_()`.
* Added method `imgaug.augmentables.bbs.BoundingBoxesOnImage.fill_from_xyxy_array_()`.
* Added method `imgaug.augmentables.bbs.BoundingBox.from_point_soup()`.
* Added method `imgaug.augmentables.bbs.BoundingBoxesOnImages.from_point_soups()`.
* Changed `imgaug.augmentables.BoundingBoxesOnImage.from_xyxy_array()` to also
  accept `(N, 2, 2)` arrays instead of only `(N, 4)`.
