# Polygon test was inconsistent

`test_alpha_is_080` in `test/augmentables/test_polys.py`
hard-coded inconsistent behavior of `scikit-image` which
was fixed with `scikit-image >=0.18.0`.

Until this point, the polygon left and right boundaries were handled
differently.