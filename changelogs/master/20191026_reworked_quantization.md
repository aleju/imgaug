# Reworked Quantization

* Renamed `imgaug.augmenters.color.quantize_colors_uniform(image, n_colors)`
  to `imgaug.augmenters.color.quantize_uniform(arr, nb_bins)`. The old name
  is now deprecated.
* Renamed `imgaug.augmenters.color.quantize_colors_kmeans(image, n_colors)`
  to `imgaug.augmenters.color.quantize_kmeans(arr, nb_clusters)`. The old name
  is now deprecated.
* Improved performance of `quantize_uniform()` by roughly 10x (small images
  around 64x64) to 100x (large images around 1024x1024). This also affects
  `UniformColorQuantization`.
* Improved performance of `UniformColorQuantization` by using more in-place
  functions.
* Added argument `to_bin_centers=True` to `quantize_uniform()`, controling
  whether each bin `(a, b)` should be quantized to `a + (b-a)/2` or `a`.
* Added function `imgaug.augmenters.color.quantize_uniform_()`, the in-place
  version of `quantize_uniform()`.
* Added function `imgaug.augmenters.color.quantize_uniform_to_n_bits()`.
* Added function `imgaug.augmenters.color.posterize()`, an alias of
  `quantize_uniform_to_n_bits()` that produces the same outputs as
  `PIL.ImageOps.posterize()`.
* Added augmenter `UniformColorQuantizationToNBits`.
* Added augmenter `Posterize` (alias of `UniformColorQuantizationToNBits`).
* Fixed `quantize_uniform()` producing wrong outputs for non-contiguous arrays.
