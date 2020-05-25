- Fixed `Affine` casting float cvals to int, even when
  the image had a float dtype, making it impossible to
  properly use cvals for images with value
  range `[0.0, 1.0]`. #669 #680
