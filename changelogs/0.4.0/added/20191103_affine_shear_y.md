# Affine Shear on the Y-Axis #482

* [rarely breaking] Extended `Affine` to also support shearing on the
  y-axis (previously, only x-axis was possible). This feature can be used
  via e.g. ``Affine(shear={"x": (-30, 30), "y": (-10, 10)})``. If instead
  a single number is used (e.g. ``Affine(shear=15)``), shearing will be done
  only on the x-axis. If a single ``tuple``, ``list`` or
  ``StochasticParameter`` is used, the generated samples will be used
  identically for both the x-axis and y-axis (this is consistent with
  translation and scaling). To get independent random samples per axis use
  the dictionary form.
