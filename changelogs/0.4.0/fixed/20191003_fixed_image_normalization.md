* Fixed image normalization crashing when an input ndarray of multiple images
  was changed during augmentation to a list of multiple images with different
  shapes *and* the original input ndarray represented a single image or
  a collection of 2D `(H,W)` images. This problem affected `augment()`,
  `augment_batch()` and `augment_batches()`.
