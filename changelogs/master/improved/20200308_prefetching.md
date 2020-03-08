# Added Automatic Prefetching of Random Number Samples #634

This patch adds automatic prefetching of random samples,
which performs a single large random sampling call instead
of many smaller ones. This seems to improve the
performance of most augmenters by 5% to 40% for longer
augmentation sessions (50+ consecutive batches of 128
examples each). A few augmenters seem to have gotten
slightly slower, though these might be measuring errors.

The prefetching is done by adding a new parameter,
`imgaug.parameters.AutoPrefetcher`, which prefetches
samples from a child parameter.

The change is expected to have for most augmenters a
slight negative performance impact if the augmenters
are used only once and not for multiple batches. For a
few augmenters there might be sizeable negative
peformance impact (due to prefetching falsely being
performed). The negative impact can be avoided in
these cases by wrapping the augmentation calls in
`with imgaug.parameters.no_prefetching(): ...`.

This patch also adds the property `prefetchable` to
`StochasticParameter`, which defaults to `False` and
determines whether the parameter's outputs may be
prefetched.

It further adds to
`handle_continuous_param()`, `handle_discrete_param()`.
`handle_categorical_string_param()`,
`handle_discrete_kernel_size_param()` and
`handle_probability_param()` in `imgaug.parameters` the
new argument `prefetch`. If set to `True` (the default),
these functions may now partially or fully wrap their
results in `AutoPrefetcher`.

Add functions:
* `imgaug.random.RNG.create_if_not_rng_()`
* `imgaug.parameters.toggle_prefetching()`
* `imgaug.testutils.is_parameter_instance()`
* `imgaug.testutils.remove_prefetching()`

Add properties:
* `imgaug.parameters.StochasticParameter.prefetchable`

Add classes:
* `imgaug.parameters.toggled_prefetching()` (context)
* `imgaug.parameters.no_prefetching()` (context)
* `imgaug.parameters.AutoPrefetcher`
