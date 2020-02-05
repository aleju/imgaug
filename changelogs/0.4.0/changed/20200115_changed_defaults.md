# Improved Default Values of Augmenters #582

**[breaking]** Most augmenters had previously default values that
made them equivalent to identity functions. Users had to explicitly
change the defaults to proper values in order to "activate"
augmentations. To simplify the usage of the library, the default
values of most augmenters were changed to medium-strength
augmentations. E.g.
`Sequential([Affine(), UniformVoronoi(), CoarseDropout()])`
should now produce decent augmentations.

A few augmenters were set to always-on, maximum-strength
augmentations. This is the case for:

* `Grayscale` (always fully grayscales images, use
  `Grayscale((0.0, 1.0))` for random strengths)
* `RemoveSaturation` (same as `Grayscale`)
* `Fliplr` (always flips images, use `Fliplr(0.5)` for 50%
  probability)
* `Flipud` (same as `Fliplr`)
* `TotalDropout` (always drops everything, use
  `TotalDropout(0.1)` to drop everything for 10% of all images)
* `Invert` (always inverts images, use `Invert(0.1)` to invert
  10% of all images)
* `Rot90` (always rotates exactly once clockwise by 90 degrees,
  use `Rot90((0, 3))` for any rotation)

These settings seemed to better match user-expectations.
Such maximum-strength settings however were not chosen for all
augmenters where one might expect them. The defaults are set to
varying strengths for, e.g.  `Superpixels` (replaces only some
superpixels with cellwise average colors), `UniformVoronoi` (also
only replaces some cells), `Sharpen` (alpha-blends with variable
strength, the same is the case for `Emboss`, `EdgeDetect` and
`DirectedEdgeDetect`) and `CLAHE` (variable clip limits).

*Note*: Some of the new default values will cause issues with
non-`uint8` inputs.

*Note*: The defaults for `per_channel` and `keep_size` were not
adjusted. It is currently still the default behaviour of all
augmenters to affect all channels in the same way and to resize
their outputs back to the input sizes.

The exact changes to default values are listed below.

**imgaug.arithmetic**
  
  * `Add`
    * `value`: `0` -> `(-20, 20)`
  * `AddElementwise`
    * `value`: `0` -> `(-20, 20)`
  * `AdditiveGaussianNoise`
    * `scale`: `0` -> `(0, 15)`
  * `AdditiveLaplaceNoise`
    * `scale`: `0` -> `(0, 15)`
  * `AdditivePoissonNoise`
    * `scale`: `0` -> `(0, 15)`
  * `Multiply`
    * `mul`: `1.0` -> `(0.8, 1.2)`
  * `MultiplyElementwise`:
    * `mul`: `1.0` -> `(0.8, 1.2)`
  * `Dropout`: 
    * `p`: `0.0` -> `(0.0, 0.05)`
  * `CoarseDropout`:
    * `p`: `0.0` -> `(0.02, 0.1)`
    * `size_px`: `None` -> `(3, 8)`
    * `min_size`: `4` -> `3`
    * Default for `size_px` is only used if neither `size_percent`
      nor `size_px` is provided by the user.
  * `CoarseSaltAndPepper`:
    * `p`: `0.0` -> `(0.02, 0.1)`
    * `size_px`: `None` -> `(3, 8)`
    * `min_size`: `4` -> `3`
    * Default for `size_px` is only used if neither `size_percent`
      nor `size_px` is provided by the user.
  * `CoarseSalt`:
    * `p`: `0.0` -> `(0.02, 0.1)`
    * `size_px`: `None` -> `(3, 8)`
    * `min_size`: `4` -> `3`
    * Default for `size_px` is only used if neither `size_percent`
      nor `size_px` is provided by the user.
  * `CoarsePepper`:
    * `p`: `0.0` -> `(0.02, 0.1)`
    * `size_px`: `None` -> `(3, 8)`
    * `min_size`: `4` -> `3`
    * Default for `size_px` is only used if neither `size_percent`
      nor `size_px` is provided by the user.
  * `SaltAndPepper`:
    * `p`: `0.0` -> `(0.0, 0.03)`
  * `Salt`:
    * `p`: `0.0` -> `(0.0, 0.03)`
  * `Pepper`:
    * `p`: `0.0` -> `(0.0, 0.05)`
  * `ImpulseNoise`:
    * `p`: `0.0` -> `(0.0, 0.03)`
  * `Invert`: 
    * `p`: `0` -> `1`
  * `JpegCompression`:
    * `compression`: `50` -> `(0, 100)`

**imgaug.blend**

  * `BlendAlpha`
    * `factor`: `0` -> `(0.0, 1.0)`
  * `BlendAlphaElementwise`
    * `factor`: `0` -> `(0.0, 1.0)`

**imgaug.blur**

  * `GaussianBlur`:
    * `sigma`: `0` -> `(0.0, 3.0)`
  * `AverageBlur`:
    * `k`: `1` -> `(1, 7)`
  * `MedianBlur`:
    * `k`: `1` -> `(1, 7)`
  * `BilateralBlur`:
    * `d`: `1` -> `(1, 9)`
  * `MotionBlur`:
    * `k`: `5` -> `(3, 7)`

**imgaug.color**

  * `MultiplyHueAndSaturation`:
    * `mul_hue`: `None` -> `(0.5, 1.5)`
    * `mul_saturation`: `None` -> `(0.0, 1.7)`
    * These defaults are only used if the user provided neither
      `mul` nor `mul_hue` nor `mul_saturation`.
  * `MultiplyHue`:
    * `mul`: `(-1.0, 1.0)` -> `(-3.0, 3.0)`
  * `AddToHueAndSaturation`:
    * `value_hue`: `None` -> `(-40, 40)`
    * `value_saturation`: `None` -> `(-40, 40)`
    * These defaults are only used if the user provided neither
      `value` nor `value_hue` nor `value_saturation`.
  * `Grayscale`:
    * `alpha`: `0` -> `1`

**imgaug.contrast**

  * `GammaContrast`:
    * `gamma`: `1` -> `(0.7, 1.7)` 
  * `SigmoidContrast`:
    * `gain`: `10` -> `(5, 6)`
    * `cutoff`: `0.5` -> `(0.3, 0.6)`
  * `LogContrast`:
    * `gain`: `1` -> `(0.4, 1.6)`
  * `LinearContrast`:
    * `alpha`: `1` -> `(0.6, 1.4)`
  * `AllChannelsCLAHE`:
    * `clip_limit`: `40` -> `(0.1, 8)`
    * `tile_grid_size_px`: `8` -> `(3, 12)`
  * `CLAHE`: 
    * `clip_limit`: `40` -> `(0.1, 8)`
    * `tile_grid_size_px`: `8` -> `(3, 12)`

**convolutional**

  * `Sharpen`:
    * `alpha`: `0` -> `(0.0, 0.2)`
    * `lightness`: `1` -> `(0.8, 1.2)`
  * `Emboss`:
    * `alpha`: `0` -> `(0.0, 1.0)`
    * `strength`: `1` -> `(0.25, 1.0)`
  * `EdgeDetect`:
    * `alpha`: `0` -> `(0.0, 0.75)`
  * `DirectedEdgeDetect`:
    * `alpha`: `0` -> `(0.0, 0.75)`

**imgaug.flip**

  * `Fliplr`:
    * `p`: `0` -> `1`
  * `Flipud`:
    * `p`: `0` -> `1`

**imgaug.geometric**

  * `Affine`:
    * `scale`: `1` -> `{"x": (0.9, 1.1), "y": (0.9, 1.1)}`
    * `translate_percent`: None -> `{"x": (-0.1, 0.1), "y": (-0.1, 0.1)}`
    * `rotate`: `0` -> `(-15, 15)`
    * `shear`: `0` -> `shear={"x": (-10, 10), "y": (-10, 10)}`
    * These defaults are only used if no affine transformation
      parameter was set by the user. Otherwise the not-set
      parameters default again towards the identity function.
  * `PiecewiseAffine`:
    * `scale`: `0` -> `(0.0, 0.04)`
    * `nb_rows`: `4` -> `(2, 4)`
    * `nb_cols`: `4` -> `(2, 4)`
  * `PerspectiveTransform`:
    * `scale`: `0` -> `(0.0, 0.06)`
  * `ElasticTransformation`:
    * `alpha`: `0` -> `(0.0, 40.0)`
    * `sigma`: `0` -> `(4.0, 8.0)`
  * `Rot90`:
    * `k`: `(no default)` -> `k=1`

**imgaug.pooling**

  * `AveragePooling`:
    * `k`: `(no default)` -> `(1, 5)`
  * `MaxPooling`:
    * `k`: `(no default)` -> `(1, 5)`
  * `MinPooling`:
    * `k`: `(no default)` -> `(1, 5)`
  * `MedianPooling`:
    * `k`: `(no default)` -> `(1, 5)`

**imgaug.segmentation**
 
  * `Superpixels`:
    * `p_replace`: `0.0` -> `(0.5, 1.0)`
    * `n_segments`: `100` -> `(50, 120)`
  * `UniformVoronoi`:
    * `n_points`: `(no default)` -> `(50, 500)`
    * `p_replace`: `1.0` -> `(0.5, 1.0)`.
  * `RegularGridVoronoi`:
    * `n_rows`: `(no default)` -> `(10, 30)`
    * `n_cols`: `(no default)` -> `(10, 30)`
    * `p_drop_points`: `0.4` -> `(0.0, 0.5)`
    * `p_replace`: `1.0` -> `(0.5, 1.0)`
  * `RelativeRegularGridVoronoi`: Changed defaults from
    * `n_rows_frac`: `(no default)` -> `(0.05, 0.15)`
    * `n_cols_frac`: `(no default)` -> `(0.05, 0.15)`
    * `p_drop_points`: `0.4` -> `(0.0, 0.5)`
    * `p_replace`: `1.0` -> `(0.5, 1.0)`

**imgaug.size**

  * `CropAndPad`:
    * `percent`: `None` -> `(-0.1, 0.1)`
    * This default is only used if the user has provided
      neither `px` nor `percent`.
  * `Pad`:
    * `percent`: `None` -> `(0.0, 0.1)`
    * This default is only used if the user has provided
      neither `px` nor `percent`.
  * `Crop`:
    * `percent`: `None` -> `(0.0, 0.1)`
    * This default is only used if the user has provided
      neither `px` nor `percent`.
