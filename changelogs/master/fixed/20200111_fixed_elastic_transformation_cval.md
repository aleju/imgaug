# Fixed `cval` in `ElasticTransformation` #561 #562

* Fixed `cval` in `ElasticTransformation` resulting new pixels in RGB images
  being filled with `(cval, 0, 0)` instead of `(cval, cval, cval)`.
