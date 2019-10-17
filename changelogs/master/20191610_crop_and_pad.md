# Changes to Crop and Pad augmenters

* Added augmenter `CenterCropToFixedSize`, a wrapper
  for `CropToFixedSize(..., position="center")`.
* Added augmenter `CropToMultiplesOf`.
* Added augmenter `PadToMultiplesOf`.
* Added augmenter `CropToExponentsOf`.
* Added augmenter `PadToExponentsOf`.
* Extended augmenter `CropToFixedSize` to support `height` and/or `width`
  parameters to be `None`, in which case the respective axis is not changed.
* Extended augmenter `PadToFixedSize` to support `height` and/or `width`
  parameters to be `None`, in which case the respective axis is not changed.
* [rarely breaking] Changed `CropToFixedSize.get_parameters()` to also
  return the `height` and `width` values.
* [rarely breaking] Changed `PadToFixedSize.get_parameters()` to also
  return the `height` and `width` values.
* Changed function `imgaug.imgaug.compute_paddings_for_aspect_ratio()`
  to also support shape tuples instead of only ndarrays.
* Changed function `imgaug.imgaug.compute_paddings_to_reach_multiples_of()`
  to also support shape tuples instead of only ndarrays.
* Added function `imgaug.imgaug.compute_croppings_to_reach_multiples_of()`.
* Added function `imgaug.imgaug.compute_croppings_to_reach_exponents_of()`.
* Added function `imgaug.imgaug.compute_paddings_to_reach_exponents_of()`.
* Fixed a formatting error in an error message of
  `compute_paddings_to_reach_multiples_of()`.
