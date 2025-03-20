# OpenCV 3 does not support zero-sized arrays

NumPy 1.23 can create zero-sized arrays which
cannot be handled by OpenCV in `_invert_uint8_subtract_`
in `imgaug/augmenters/arithmetic.py`. If a zero-sized
array is detected directly return it without further 
processing.
