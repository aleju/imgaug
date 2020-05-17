# Improved Performance of dtype checks #663

This patch improves the performance of dtype checks
throughout the library. The new method verifies input
arrays around 10x to 100x faster than the previous one.

Add functions:
* `imgaug.dtypes.gate_dtypes_strs()`
* `imgaug.dtypes.allow_only_uint8()`

Add decorators:
* `imgaug.testutils.ensure_deprecation_warning`

Deprecate functions:
* `imgaug.dtypes.gate_dtypes()`
