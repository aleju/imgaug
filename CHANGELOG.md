# master (will be 0.2.8)

This update focused on extending and documenting the library's dtype support and on improving the performance.


# dtype support

Previous versions of `imgaug` were primarily geared towards `uint8`.
In this version, all augmenters and helper functions were refactored to be more tolerant towards non-uint8 dtypes.
Additionally all augmenters were tested with non-uint8 dtypes and an overview of the expected support-level
is now listed in the documentation on page [dtype support](https://imgaug.readthedocs.io/en/latest/source/dtype_support.html).
Further details are listed in the docstrings of each individual augmenter or helper function.


## Performance Improvements

Below are some numbers for the achieved performance improvements compared to 0.2.7.
The measurements were taken using realistic 224x224x3 uint8 images and batch size 128.
The percentage values denote the increase in bandwidth (i.e. mbyte/sec) of the respective
augmenter given the described input. Improvements for smaller images, smaller batch sizes
and non-uint8 dtypes may differ. Augmenters with less than roughly 10% improvement are not
listed. While the numbers here are exact, there is some measurement error involved as they
were calculated based on a rather low number of 100 repetitions.

* Sequential (with 2x Noop as children) +184% to +276%
* SomeOf (with 3x Noop as children) +24% to +49%
* OneOf (with 3x Noop as children) +21%
* Sometimes (with Noop as child) +23%
* WithChannels +32%
* Add +216%
* AddElementwise +49%
* AdditiveGaussianNoise +26%
* AdditiveLaplaceNoise +20%
* AdditivePoissonNoise +18%
* Multiply +206%
* MultiplyElementwise +74%
* Dropout +154%
* CoarseDropout +246%
* ReplaceElementwise +119%
* ImpulseNoise +333%
* SaltAndPepper +184%
* CoarseSaltAndPepper +227%
* Salt +204%
* CoarseSalt +260%
* Pepper +208%
* CoarsePepper +276%
* Invert +1192%
* GaussianBlur +885%
* AddToHueAndSaturation +48%
* GammaContrast +2988%
* SigmoidContrast +519%
* LogContrast +1048%
* LinearContrast +448%
* Convolve +47%
* Sharpen +29%
* Emboss +18%
* EdgeDetect +41%
* DirectedEdgeDetect +53%
* Fliplr +75%
* Flipud +25%
* Affine +7% to +33%
* ElasticTransformation +650 to +680%
* CropAndPad +30% to +77% (from improved padding)
* Pad +40 to +140%
* PadToFixedSize +288%
* KeepSizeByResize (with CropToFixedSize as child) +58%
* Snowflakes +44%
* SnowflakesLayer +42%


## imgaug.imgaug

* Added constants that control the min/max values for seed generation
* Improved performance of `pad()`
    * this change also improves the performance of:
        * `imgaug.imgaug.pad_to_aspect_ratio()`,
        * `imgaug.imgaug.HeatmapsOnImage.pad()`,
        * `imgaug.imgaug.HeatmapsOnImage.pad_to_aspect_ratio()`,
        * `imgaug.imgaug.SegmentationMapOnImage.pad()`,
        * `imgaug.imgaug.SegmentationMapOnImage.pad_to_aspect_ratio()`,
        * `imgaug.augmenters.size.PadToFixedSize`,
        * `imgaug.augmenters.size.Pad`,
        * `imgaug.augmenters.size.CropAndPad`
* Changed `imshow()` to explicitly make the plot figure size dependent on the input image size.
* Refactored `SegmentationMapOnImage` to have simplified dtype handling in `__init__`
* Fixed an issue with `SEED_MAX_VALUE` exceeding the `int32` maximum on some systems, causing crashes related to
  RandomState.
* Moved BatchLoader to `multicore.py` and replaced the class with an alias pointing to `imgaug.multicore.BatchLoader`.
* Moved BackgroundAugmenter to `multicore.py` and replaced the class with an alias pointing to `imgaug.multicore.BatchLoader`.
* Renamed `HeatmapsOnImage.scale()` to `HeatmapsOnImage.resize()`.
* Marked `HeatmapsOnImage.scale()` as deprecated.
* Renamed `SegmentationMapOnImage.scale()` to `SegmentationMapOnImage.resize()`.
* Marked `SegmentationMapOnImage.scale()` as deprecated.
* Renamed `BoundingBox.cut_out_of_image()` to `BoundingBox.clip_out_of_image()`.
* Marked `BoundingBox.cut_out_of_image()` as deprecated.
* Renamed `BoundingBoxesOnImage.cut_out_of_image()` to `BoundingBoxesOnImage.clip_out_of_image()`.
* Marked `BoundingBoxesOnImage.cut_out_of_image()` as deprecated.
* Marked `Polygon.cut_out_of_image()` as deprecated. (The analogous clip function existed already.)


## imgaug.multicore

* Created this file.
* Moved `BatchLoader` here from `imgaug.py`.
* Moved `BackgroudAugmenter` here from `imgaug.py`.
* Marked `BatchLoader` as deprecated.
* Marked `BackgroundAugmenter` as deprecated.
* Added class `Pool`. This is the new recommended way for multicore augmentation. `BatchLoader`/`BackgroundAugmenter` should not be used anymore. Example:
  ```python
  import imgaug as ia
  from imgaug import augmenters as iaa
  from imgaug import multicore
  import numpy as np
  aug = iaa.Add(1)
  images = np.zeros((16, 128, 128, 3), dtype=np.uint8)
  batches = [ia.Batch(images=np.copy(images)) for _ in range(100)]
  with multicore.Pool(aug, processes=-1, seed=2) as pool:
      batches_aug = pool.map_batches(batches, chunksize=8)
  print(np.sum(batches_aug[0].images_aug[0]))
  ```
  The example starts a pool with N-1 workers (N=number of CPU cores) and augments 100 batches using these workers.
  Use `imap_batches()` to feed in and get out a generator.


## imgaug.parameters

* Added `TruncatedNormal`
* Added `handle_discrete_kernel_size_param()`
* Improved dtype-related interplay of `FromLowerResolution` and `imresize_many_images()`
* Improved performance for sampling from `Deterministic` by about 2x
* Improved performance for sampling from `Uniform` with `a == b`
* Improved performance for sampling from `DiscreteUniform` with `a == b`
* Improved performance for sampling from `Laplace` with `scale=0`
* Improved performance for sampling from `Normal` with `scale=0`
* Improved performance of `Clip` and improved code style
* Refactored `float check in force_np_float_dtype()`
* Refactored `RandomSign`
* Refactored various unittests to be more flexible with regards to returned dtypes
* Refactored `StochasticParameter.draw_distribution_graph()` to use internally tempfile-based drawing. Should result in higher-quality outputs.
* Refactored unittest for `draw_distributions_grid()` to improve performance
* Fixed in `draw_distributions_grid()` a possible error from arrays with unequal shapes being combined to one array
* Fixed a problem with `Sigmoid` not returning floats
* Fixed noise produced by `SimplexNoise` having values below 0.0 or above 1.0
* Fixed noise produced by `SimplexNoise` being more biased towards 0 than it should be


## imgaug.dtypes

* Added new file `imgaug/dtypes.py` and respective test file `test_dtypes.py`.
* Added `clip_to_dtype_value_range_()`
* Added `get_value_range_of_dtype()`
* Added `promote_array_dtypes_()`
* Added `get_minimal_dtype()`
* Added `get_minimal_dtypes_for_values()`
* Added `get_minimal_dtype_by_value_range()`
* Added `restore_dtypes_()`
* Added `gate_dtypes()`
* Added `increase_array_resolutions()`
* Added `copy_dtypes_for_restore()`


## imgaug.augmenters.meta

* Added `estimate_max_number_of_channels()`
* Added `copy_arrays()`
* Added an optional parameter `default` to `handle_children_lists()`
* Enabled `None` as arguments for `Lambda` and made all arguments optional
* Enabled `None` as arguments for `AssertLambda` and made all arguments optional
* Improved dtype support of `AssertShape`
* Improved dtype support of `AssertLambda`
* Improved dtype support of `Lambda`
* Improved dtype support of `ChannelShuffle`
* Improved dtype support of `WithChannels`
* Improved dtype support of `Sometimes`
* Improved dtype support of `SomeOf`
* Improved dtype support of `Sequential`
* Improved dtype support of `Noop`
* [breaking, mostly internal] Removed `restore_augmented_images_dtypes()`
* [breaking, mostly internal] Removed `restore_augmented_images_dtypes_()`
* [breaking, mostly internal] Removed `restore_augmented_images_dtype()`
* [breaking, mostly internal] Removed `restore_augmented_images_dtype_()`
* [breaking, mostly internal] Refactored `Augmenter.augment_images()` and `Augmenter._augment_images()` to default
  `hooks` to `None`
    * This will affect any custom implemented augmenters that try to access the hooks argument.
* [breaking, mostly internal] Refactored `Augmenter.augment_heatmaps()` and `Augmenter._augment_heatmaps()` to default
  `hooks` to `None`
    * Same as above for images.
* [breaking, mostly internal] Refactored `Augmenter.augment_keypoints()` and `Augmenter._augment_keypoints()` to default
  `hooks` to None
    * Same as above for images.
* [breaking, mostly internal] Improved performance of image augmentation for augmenters with children
    * For calls to `augment_image()`, the validation, normalization and copying steps are skipped if the call is
      a child call (e.g. a `Sequential` calling `augment_images()` on a child `Add`). Hence, such child calls augment
      now fully in-place (the top-most call still creates a copy though, so from the user perspective nothing
      changes). Custom implemented augmenters that rely on child calls to `augment_images()` creating copies will
      break from this change.
      For an example `Sequential` containing two `Noop` augmenters, this change improves the performance by roughly 2x
* [breaking, mostly internal] Improved performance of heatmap augmentation for augmenters with children
    * Same as above for images.
    * Speedup is around 2-3x for an exemplary `Sequential` containing two `Noop`s.
    * This will similarly affect segmentation map augmentation too.
* [breaking, mostly internal] Improved performance of keypoint augmentation for augmenters with children
    * Same as above for images.
    * Speedup is around 1.5-2x for an exemplary `Sequential` containing two Noops.
    * This will similarly affect bounding box augmentation.
* [critical] Fixed a bug in the augmentation of empty `KeypointsOnImage` instances that would lead image and keypoint
  augmentation to be un-aligned within a batch after the first empty `KeypointsOnImage` instance. (#231)
* Added `pool()` to `Augmenter`. This is a helper to start a `imgaug.multicore.Pool` via `with augmenter.pool() as pool: ...`.
* Changed `to_deterministic()` in `Augmenter` and various child classes to derive its new random state from the augmenter's local random state instead of the global random state.
* Enabled support for non-list `HeatmapsOnImage` inputs in `Augmenter.augment_heatmaps()`. (Before, only lists were supported.)
* Enabled support for non-list `SegmentationMapOnImage` inputs in `Augmenter.augment_segmentation_maps()`. (Before, only lists were supported.)
* Enabled support for non-list `KeypointsOnImage` inputs in `Augmenter.augment_keypoints()`. (Before, only lists were supported.)
* Enabled support for non-list `BoundingBoxesOnImage` inputs in `Augmenter.augment_bounding_boxes()`. (Before, only lists were supported.)


## imgaug.augmenters.arithmetic

* `ContrastNormalization` is now an alias for `LinearContrast`
* Restricted `JpegCompression` to uint8 inputs. Other dtypes will now produce errors early on.
* Changed in `Add` the parameter `value` to be continuous and removed its `value_range`


## imgaug.augmenters.blur

* Added function `blur_gaussian()`


## imgaug.augmenters.color

* Added `Lab2RGB` and `Lab2BGR` to `ChangeColorspace`
* Refactored the main loop in `AddToHueAndSaturation` to make it simpler and faster
* Fixed `ChangeColorspace` not being able to convert from RGB/BGR to Lab/Luv


## imgaug.augmenters.contrast

* Added `AllChannelsCLAHE`
* Added `CLAHE`
* Added `AllChannelsHistogramEqualization`
* Added `HistogramEqualization`
* Added `_IntensityChannelBasedApplier`
* Added function `adjust_contrast_gamma()`
* Added function `adjust_contrast_sigmoid()`
* Added function `adjust_contrast_log()`
* Refactored random state handling in `_ContrastFuncWrapper`
* [breaking, internal] Removed `_PreserveDtype`
* [breaking, internal] Renamed `_adjust_linear` to `adjust_contrast_linear()`


## imgaug.augmenters.convolutional

* Refactored `AverageBlur` to have improved random state handling
* Refactored `GaussianBlur` to only overwrite input images when that is necessary
* Refactored `GaussianBlur` to have a simplified main loop
* Refactored `AverageBlur` to have a simplified main loop
* Refactored `MedianBlur` to have a simplified main loop
* Refactored `BilateralBlur` to have a simplified main loop
* Improved dtype support of `GaussianBlur`
* Improved dtype support of `AverageBlur`
* Improved dtype support of `Convolve`


## imgaug.augmenters.flip

* Improved dtype support of `Fliplr`
* Improved dtype support of `Flipud`
* Refactored `Fliplr` main loop to be more elegant and tolerant
* Refactored `Flipud` main loop to be more elegant and tolerant
* Added alias `HorizontalFlip` for `Fliplr`.
* Added alias `VerticalFlip` for `Flipud`.


## imgaug.augmenters.geometric

* `ElasticTransformation`
    * [breaking, mostly internal] `generate_indices()` now returns only the pixelwise shift as a tuple of x and y
    * [breaking, mostly internal] `generate_indices()` has no longer a `reshape` argument
    * [breaking, mostly internal] `renamed generate_indices()` to `generate_shift_maps()`
    * [breaking, mostly internal] `map_coordinates()` now expects to get the pixelwise shift as its input, instead of
      the target coordinates


## imgaug.augmenters.overlay

* Added `blend_alpha()`
* Refactored `Alpha` to be simpler and use `blend_alpha()`
* Fixed `Alpha` not having its own `__str__` method
* Improved dtype support of `AlphaElementwise`


## imgaug.augmenters.segmentation

* Improved dtype support of `Superpixels`


## imgaug.augmenters.size

* Removed the restriction to `uint8` in `Scale`. The augmenter now supports the same dtypes as `imresize_many_images()`.
* Fixed missing pad mode `mean` in `Pad` and `CropAndPad`.
* Improved error messages related to pad mode.
* Improved and fixed docstrings of `CropAndPad`, `Crop`, `Pad`.
* Renamed `Scale` to `Resize`.
* Marked `Scale` as deprecated.


## other

* Improved descriptions of the library in `setup.py`
* `matplotlib` is now an optional dependency of the library and loaded lazily when needed
* `Shapely` is now an optional dependency of the library and loaded lazily when needed
* `opencv-python` is now a dependency of the library
* `setup.py` no longer enforces `cv2` to be installed (to allow installing libraries in random order)
* Minimum required `numpy` version is now 1.15
