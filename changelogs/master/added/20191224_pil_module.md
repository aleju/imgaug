# Added Module `imgaug.augmenters.pillike` #479 #480 #538

* Added module `imgaug.augmenters.pillike`, which contains augmenters and
  functions corresponding to commonly used PIL functions. Their outputs
  are guaranteed to be identical to the PIL outputs.
* Added the following functions to the module:
  * `imgaug.augmenters.pillike.equalize`
  * `imgaug.augmenters.pillike.equalize_`
  * `imgaug.augmenters.pillike.autocontrast`
  * `imgaug.augmenters.pillike.autocontrast_`
  * `imgaug.augmenters.pillike.solarize`
  * `imgaug.augmenters.pillike.solarize_`
  * `imgaug.augmenters.pillike.posterize`
  * `imgaug.augmenters.pillike.posterize_`
  * `imgaug.augmenters.pillike.enhance_color`
  * `imgaug.augmenters.pillike.enhance_contrast`
  * `imgaug.augmenters.pillike.enhance_brightness`
  * `imgaug.augmenters.pillike.enhance_sharpness`
  * `imgaug.augmenters.pillike.filter_blur`
  * `imgaug.augmenters.pillike.filter_smooth`
  * `imgaug.augmenters.pillike.filter_smooth_more`
  * `imgaug.augmenters.pillike.filter_edge_enhance`
  * `imgaug.augmenters.pillike.filter_edge_enhance_more`
  * `imgaug.augmenters.pillike.filter_find_edges`
  * `imgaug.augmenters.pillike.filter_contour`
  * `imgaug.augmenters.pillike.filter_emboss`
  * `imgaug.augmenters.pillike.filter_sharpen`
  * `imgaug.augmenters.pillike.filter_detail`
  * `imgaug.augmenters.pillike.warp_affine`
* Added the following augmenters to the module:
  * `imgaug.augmenters.pillike.Solarize`
  * `imgaug.augmenters.pillike.Posterize`.
    (Currently alias for `imgaug.augmenters.color.Posterize`.)
  * `imgaug.augmenters.pillike.Equalize`
  * `imgaug.augmenters.pillike.Autocontrast`
  * `imgaug.augmenters.pillike.EnhanceColor`
  * `imgaug.augmenters.pillike.EnhanceContrast`
  * `imgaug.augmenters.pillike.EnhanceBrightness`
  * `imgaug.augmenters.pillike.EnhanceSharpness`
  * `imgaug.augmenters.pillike.FilterBlur`
  * `imgaug.augmenters.pillike.FilterSmooth`
  * `imgaug.augmenters.pillike.FilterSmoothMore`
  * `imgaug.augmenters.pillike.FilterEdgeEnhance`
  * `imgaug.augmenters.pillike.FilterEdgeEnhanceMore`
  * `imgaug.augmenters.pillike.FilterFindEdges`
  * `imgaug.augmenters.pillike.FilterContour`
  * `imgaug.augmenters.pillike.FilterEmboss`
  * `imgaug.augmenters.pillike.FilterSharpen`
  * `imgaug.augmenters.pillike.FilterDetail`
  * `imgaug.augmenters.pillike.Affine`
