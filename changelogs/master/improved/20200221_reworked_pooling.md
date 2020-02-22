# Reworked Pooling #622

This patch improves the performance of pooling operations.

For `uint8` arrays, `max_pool()` and `min_pool()` are now
between 3x and 8x faster. The improvements are more
significant for larger images and smaller kernel sizes.
In-place versions of `max_pool()` and `min_pool()` are also
added. Both `MaxPooling` and `MinPooling` now use these
functions.

The performance of `avg_pool()` for `uint8` is improved by
roughly 4x to 15x. (More for larger images and smaller
kernel sizes.)

The performance of `median_pool()` for `uint8` images is
improved by roughly 1.7x to 7x, if the kernel size is 3
or 5 or if the kernel size is 7, 9, 11, 13 and the image
size is 32x32 or less. In both cases the kernel also has to be
symmetric.
In the case of a kernel size of 3, the performance improvement
is most significant for larger images. In the case of 5, it
is fairly even over all kernel sizes. In the case of 7 or higher,
it is more significant for smaller images.

Add functions:
* `imgaug.imgaug.min_pool_()`.
* `imgaug.imgaug.max_pool_()`.
