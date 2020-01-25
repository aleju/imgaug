# Improved Warnings on Probably-Wrong Image Inputs #594

Improved the errors and warnings on image augmentation calls.
`augment_image()` will now produce a more self-explanatory error
message when calling it as in `augment_image(list of images)`.
Calls of single-image augmentation functions (e.g.
`augment(image=...)`) with inputs that look like multiple images
will now produce warnings. This is the case for `(H, W, C)`
inputs when `C>=32` (as that indicates that `(N, H, W)` was
actually provided).
Calls of multi-image augmentation functions (e.g.
`augment(images=...)`) with inputs that look like single images
will now produce warnings. This is the case for `(N, H, W)`
inputs when `W=1` or `W=3` (as that indicates that `(H, W, C)`
was actually provided.)

* Added an assert in `augment_image()` to verify that inputs are
  arrays.
* Added warnings for probably-wrong image inputs in
  `augment_image()`, `augment_images()`, `augment()` (and its
  alias `__call__()`).
* Added module `imgaug.augmenters.base`.
* Added warning
  `imgaug.augmenters.base.SuspiciousMultiImageShapeWarning`.
* Added warning
  `imgaug.augmenters.base.SuspiciousSingleImageShapeWarning`.
* Added `imgaug.testutils.assertWarns`, similar to `unittest`'s
  `assertWarns`, but available in python <3.2.