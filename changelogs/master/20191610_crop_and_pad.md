# Changes to Crop and Pad augmenters

* Added augmenter `CenterCropToFixedSize`, a wrapper
  for `CropToFixedSize(..., position="center")`.
* Added augmenter `CropToMultiplesOf`.
* Added augmenter `PadToMultiplesOf`.
* Added augmenter `CropToExponentsOf`.
* Changed function `imgaug.imgaug.compute_paddings_for_aspect_ratio()`
  to also support shape tuples instead of ndarrays.
* Changed function `imgaug.imgaug.compute_paddings_to_reach_multiples_of()`
  to also support shape tuples instead of ndarrays.
* Added function `imgaug.imgaug.compute_croppings_to_reach_multiples_of()`.
* Added function `imgaug.imgaug.compute_croppings_to_reach_exponents_of()`.
* Fixed a formatting error in an error message of
  `compute_paddings_to_reach_multiples_of()`.
