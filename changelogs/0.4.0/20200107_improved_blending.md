# More Choices for Image Blending #462 #556

The available augmenters for alpha-blending of images were
significantly extended. There are now new blending
augmenters available to alpha-blend acoording to:
* Some randomly chosen colors. (`BlendAlphaSomeColors`)
* Linear gradients. (`BlendAlphaHorizontalLinearGradient`,
  `BlendAlphaVerticalLinearGradient`)
* Regular grids and checkerboard patterns. (`BlendAlphaRegularGrid`,
  `BlendAlphaCheckerboard`)
* Only at locations that overlap with specific segmentation class
  IDs (or the inverse of that). (`BlendAlphaSegMapClassIds`)
* Only within bounding boxes with specific labels (or the inverse
  of that). (`BlendAlphaBoundingBoxes`)

This allows to e.g. randomly remove some colors while leaving
other colors unchanged (`BlendAlphaSomeColors(Grayscale(1.0))`),
to change the color of some objects
(`BlendAlphaSegMapClassIds(AddToHue((-256, 256)))`), to add
cloud-patterns only to the top of images
(`BlendAlphaVerticalLinearGradient(Clouds())`) or to apply
augmenters in some coarse rectangular areas (e.g.
`BlendAlphaRegularGrid(Multiply(0.0))` to achieve a similar
effect to `CoarseDropout` or
`BlendAlphaRegularGrid(AveragePooling(8))` to pool in equally
coarse image sub-regions).

Other mask-based alpha blending techniques can be achieved by
subclassing `IBatchwiseMaskGenerator` and providing an
instance of such a class to `BlendAlphaMask`.

This patch also changes the naming of the blending augmenters
as follows:
* `Alpha` -> `BlendAlpha`
* `AlphaElementwise` -> `BlendAlphaElementwise`
* `SimplexNoiseAlpha` -> `BlendAlphaSimplexNoise`
* `FrequencyNoiseAlpha` -> `BlendAlphaFrequencyNoise`
The old names are now deprecated.
Furthermore, the parameters `first` and `second`, which were
used by all blending augmenters, have now the names `foreground`
and `background`.

List of changes:
* Added `imgaug.augmenters.blend.BlendAlphaMask`, which uses
  a mask generator instance to generate per batch alpha masks and
  then alpha-blends using these masks.
* Added `imgaug.augmenters.blend.BlendAlphaSomeColors`.
* Added `imgaug.augmenters.blend.BlendAlphaHorizontalLinearGradient`.
* Added `imgaug.augmenters.blend.BlendAlphaVerticalLinearGradient`.
* Added `imgaug.augmenters.blend.BlendAlphaRegularGrid`.
* Added `imgaug.augmenters.blend.BlendAlphaCheckerboard`.
* Added `imgaug.augmenters.blend.BlendAlphaSegMapClassIds`.
* Added `imgaug.augmenters.blend.BlendAlphaBoundingBoxes`.
* Added `imgaug.augmenters.blend.IBatchwiseMaskGenerator`,
  an interface for classes generating masks on a batch-by-batch
  basis.
* Added `imgaug.augmenters.blend.StochasticParameterMaskGen`,
  a helper to generate masks from `StochasticParameter` instances.
* Added `imgaug.augmenters.blend.SomeColorsMaskGen`, a generator
  that produces masks marking randomly chosen colors in images.
* Added `imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`,
  a linear gradient mask generator.
* Added `imgaug.augmenters.blend.VerticalLinearGradientMaskGen`,
  a linear gradient mask generator.
* Added `imgaug.augmenters.blend.RegularGridMaskGen`,
  a checkerboard-like mask generator where every grid cell has
  a random alpha value.
* Added `imgaug.augmenters.blend.CheckerboardMaskGen`,
  a checkerboard-like mask generator where every grid cell has
  the opposite alpha value of its 4-neighbours.
* Added `imgaug.augmenters.blend.SegMapClassIdsMaskGen`, a
  segmentation map-based mask generator.
* Added `imgaug.augmenters.blend.BoundingBoxesMaskGen`, a bounding
  box-based mask generator.
* Added `imgaug.augmenters.blend.InvertMaskGen`, an mask generator
  that inverts masks produces by child generators.
* Changed `imgaug.parameters.SimplexNoise` and
  `imgaug.parameters.FrequencyNoise` to also accept `(H, W, C)`
  sampling shapes, instead of only `(H, W)`.
* Refactored `AlphaElementwise` to be a wrapper around
  `BlendAlphaMask`.
* Renamed `Alpha` to `BlendAlpha`.
  `Alpha` is now deprecated.
* Renamed `AlphaElementwise` to `BlendAlphaElementwise`.
  `AlphaElementwise` is now deprecated.
* Renamed `SimplexNoiseAlpha` to `BlendAlphaSimplexNoise`.
  `SimplexNoiseAlpha` is now deprecated.
* Renamed `FrequencyNoiseAlpha` to `BlendAlphaFrequencyNoise`.
  `FrequencyNoiseAlpha` is now deprecated.
* Renamed arguments `first` and `second` to `foreground` and `background`
  in `BlendAlpha`, `BlendAlphaElementwise`, `BlendAlphaSimplexNoise` and
  `BlendAlphaFrequencyNoise`.
* Changed `imgaug.parameters.handle_categorical_string_param()` to allow
  parameter `valid_values` to be `None`.
* Fixed a wrong error message in
  `imgaug.augmenters.color.change_colorspace_()`.
