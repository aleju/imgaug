* Fixed an inaccuracy in `PerspectiveTransform` that could lead to slightly
  misaligned transformations between images and coordinate-based
  augmentables (e.g. bounding boxes). The problem was more significant the
  smaller the images and larger the `scale` values were. It was also
  worsened by using `fit_output`. #585
