* Fixed `Resize` always returning an `uint8` array during image augmentation
  if the input was a single numpy array and all augmented images had the
  same shape. #442 #443
