# Affine Translation Precision

* Removed a rounding operation in `Affine` translation that would unnecessarily
  round floats to integers. This should make coordinate augmentation overall
  more accurate.
