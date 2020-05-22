# Limit dtype Support in Alpha Blending in Windows #678

This patch marks the dtypes uint64, int64 and float64
as 'only supported to a limited degree' in blend_alpha().
The dtypes require float128 for accurate output
computations, which is not supported in Windows.

Additionally, a better error message is provided if one
of these dtypes is used and float128 is not supported.
