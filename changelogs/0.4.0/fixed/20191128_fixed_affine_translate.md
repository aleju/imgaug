# `Affine` Translate Type Falsely dependent on float/int Samples #508

* Fixed `Affine` parameter `translate_px` behaving like `translate_percent`
  if a continuous stochastic parameter was provided.
  Analogously `translate_percent` would behave like `translate_px` if
  a discrete stochastic parameter was provided.