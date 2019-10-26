# Reworked Quantization

* Renamed `imgaug.augmenters.color.quantize_colors_uniform(image, n_colors)`
  to `imgaug.augmenters.color.quantize_uniform(arr, nb_bins)`. The old name
  is now deprecated.
* Renamed `imgaug.augmenters.color.quantize_colors_kmeans(image, n_colors)`
  to `imgaug.augmenters.color.quantize_kmeans(arr, nb_clusters)`. The old name
  is now deprecated.
* Improved performance of `quantize_uniform()` by roughly 10x (small images
  around 64x64) to 100x (large images around 1024x1024).
