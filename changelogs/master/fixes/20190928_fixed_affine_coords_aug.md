* Fixed `Affine` coordinate-based augmentation applying wrong offset
  when shifting images to/from top-left corner. This would lead to an error
  of around 0.5 to 1.0 pixels.
