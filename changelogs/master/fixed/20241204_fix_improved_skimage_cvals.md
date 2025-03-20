# Changed behavior of `_augment_images_by_samples`

The cval test became obsolete for skimage > 0.19 since the piecewise
tested affine transform does not produce empty pixels anymore. The
test for cval == 0 only passed due to uint overflows.