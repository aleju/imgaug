* Fixed `KeepSizeByResize` potentially crashing if a single numpy array
  was provided as the input for an iterable of images (as opposed to
  a list of numpy arrays). #590