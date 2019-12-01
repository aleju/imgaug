# Generate Debug Images #502

* Added module `imgaug.augmenters.debug`.
* Added function `imgaug.augmenters.debug.draw_debug_image()`. The function
  draws an image containing debugging information for a provided set of
  images and non-image data (e.g. segmentation maps, bounding boxes)
  corresponding to a single batch. The debug image visualizes these
  informations (e.g. bounding boxes drawn on images) and offers relevant
  information (e.g. actual value ranges of images, labels of bounding
  boxes and their counts, etc.).
* Added augmenter `imgaug.augmenters.debug.SaveDebugImageEveryNBatches`.
  Augmenter corresponding to `draw_debug_image()`. Saves an image at every
  n-th batch into a provided folder.
