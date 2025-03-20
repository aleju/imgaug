# Mitigate different behavior for scikit-image's transform.warp

Prior to `scikit-image=0.19.0` `transform.warp` converted float16 images
implicitly to float64 prior to the calculation. This has changed such
that we need to convert the image to a supported floating point format
ourselves in `_warp_affine_arr_skimage` and `_augment_images_by_samples` 
in `imgaug/augmenters/geometric.py`.