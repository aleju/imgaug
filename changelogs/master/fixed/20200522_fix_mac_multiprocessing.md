- Fixed an error on MacOS in python 3.7 that could appear
  when using multicore augmentation. The library will now
  use `spawn` mode in that situation. The error can thus
  still appear when using a custom multiprocessing
  implementation. It is recommended to use python 3.8 on
  Mac. #673
