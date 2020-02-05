# Fixed `random.seed` not always seeding in-place #557

Fixed `imgaug.random.seed()` not seeding the global `RNG` in-place
in numpy 1.17+. The (unfixed) function instead created a new
global `RNG` with the given seed. This set the seed of augmenters
created *after* the `seed()` call, but not of augmenters created
*before* the `seed()` call as they would continue to use the old
global RNG.
