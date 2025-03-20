# Replace default (and depreciated) cval with replacement

In `skimage.measure.block_reduce` in `imgaug.pool` the constant value is
set by depreciated `cval` variable althought replacement `pad_cval` is
available.