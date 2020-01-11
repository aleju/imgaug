# Added Wrappers for `imagecorruptions` Package #530

Added wrappers around the functions from package
[bethgelab/imagecorruptions](https://github.com/bethgelab/imagecorruptions).
The functions in that package were used in some recent papers and are added
here for convenience.
The wrappers produce arrays containing values identical to the output
arrays from the corresponding `imagecorruptions` functions when called
via the `imagecorruptions.corrupt()` (verified via unittests).
The interfaces of the wrapper functions are identical to the
`imagecorruptions` functions, with the only difference of also supporting
`seed` parameters.

* Added module `imgaug.augmenters.imgcorruptlike`. The `like` signals that
  the augmentation functions do not *have* to wrap `imagecorruptions`
  internally. They merely have to produce the same outputs.
* Added the following functions to module `imgaug.augmenters.imgcorruptlike`:
    * `apply_gaussian_noise()`
    * `apply_shot_noise()`
    * `apply_impulse_noise()`
    * `apply_speckle_noise()`
    * `apply_gaussian_blur()`
    * `apply_glass_blur()` (improved performance over original function)
    * `apply_defocus_blur()`
    * `apply_motion_blur()`
    * `apply_zoom_blur()`
    * `apply_fog()`
    * `apply_snow()`
    * `apply_spatter()`
    * `apply_contrast()`
    * `apply_brightness()`
    * `apply_saturate()`
    * `apply_jpeg_compression()`
    * `apply_pixelate()`
    * `apply_elastic_transform()`
* Added function
  `imgaug.augmenters.imgcorruptlike.get_corruption_names(subset)`.
  Similar to `imagecorruptions.get_corruption_names(subset)`, but returns a
  tuple
  `(list of corruption method names, list of corruption method functions)`,
  instead of only the names.
* Added the following augmenters to module `imgaug.augmenters.imgcorruptlike`:
    * `GaussianNoise`
    * `ShotNoise`
    * `ImpulseNoise`
    * `SpeckleNoise`
    * `GaussianBlur`
    * `GlassBlur`
    * `DefocusBlur`
    * `MotionBlur`
    * `ZoomBlur`
    * `Fog`
    * `Frost`
    * `Snow`
    * `Spatter`
    * `Contrast`
    * `Brightness`
    * `Saturate`
    * `JpegCompression`
    * `Pixelate`
    * `ElasticTransform`
* Added context `imgaug.random.temporary_numpy_seed()`.
