# Unwrapped Line String Augmentation #450

* Added internal `_augment_line_strings()` methods to various augmenters.
  This allows to individually control how line strings are supposed to
  be augmented. Previously, the line string augmentation was a wrapper around
  keypoint augmentation that did not allow such control.
* [rarely breaking] Added parameter `func_line_strings` to `Lambda`.
  This breaks if one relied on the order of the augmenter's parameters instead
  of their names.
* [rarely breaking] Added parameter `func_line_strings` to `AssertLambda`.
  This breaks if one relied on the order of the augmenter's parameters instead
  of their names.
* [rarely breaking] Added parameter `check_line_strings` to `AssertShape`.
  This breaks if one relied on the order of the augmenter's parameters instead
  of their names.