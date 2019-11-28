# `Affine.get_parameters()` and `translate_px`/`translate_percent` #508

* Changed `Affine.get_parameters()` to always return a tuple `(x, y, mode)`
  for translation, where `mode` is either `px` or `percent`,
  and `x` and `y` are stochastic parameters. `y` may be `None` if the same
  parameter (and hence samples) are used for both axes.