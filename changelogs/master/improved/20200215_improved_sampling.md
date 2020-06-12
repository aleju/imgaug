# Improve Performance of Gaussian and Uniform Sampling #???

This patch improves the speed at which samples are drawn from
gaussian and uniform distributions.
In the case of gaussians, the improvement is activated for
matrices with more than 2500 components (roughly 32x32x3 and
above). For large matrices (224x224x3 and above) the expected
speedup is up to 3x.
In the case of uniform distributions, the improvement is
activated for matrices with more than 12500 components (roughly
64x64x3 and above). For large matrices (224x224x3 and above)
the expected speedup is up to 2x.

The dtype of the array returned by `imgaug.random.RNG.normal()`,
`imgaug.random.RNG.uniform()` and `imgaug.random.RNG.random_sample()`
is now always `float32` instead of `float64`.
