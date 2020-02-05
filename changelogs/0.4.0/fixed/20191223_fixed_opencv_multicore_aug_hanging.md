# Fixed OpenCV Hanging in Multicore Augmentation #535

* Fixed an issue that could lead to endlessly hanging programs on some OS
  when using multicore augmentation (e.g. via pool) and augmenters using
  OpenCV.
