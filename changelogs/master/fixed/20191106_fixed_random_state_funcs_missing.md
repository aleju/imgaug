# Fixed Missing RandomState Methods #486

* Added aliases to `imgaug.random.RNG` for some outdated numpy random number
  sampling methods that existed in `numpy.random.RandomState` but not in
  numpy's new RNG system (1.17+). These old methods are not used in `imgaug`,
  but some custom augmenters and `Lambda` calls may require them when
  interacting with the provided `random_state` instances. 
