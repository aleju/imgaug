# Changes to PerspectiveTransform #452 #456

* [rarely breaking] PerspectiveTransform has now a `fit_output` parameter,
  similar to `Affine`. This change may break code that relied on the order of
  arguments to `__init__`.
* The sampling code of `PerspectiveTransform` was reworked and should now
  be faster.
