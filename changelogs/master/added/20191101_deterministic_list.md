# Added DeterministicList #475

* Added `imgaug.parameters.DeterministicList`. Upon a request to generate
  samples of shape `S`, this parameter will create a new array of shape `S`
  and fill it by cycling over its list of values repeatedly.