# Changes to Crop and Pad augmenters

* Added augmenter `CenterCropToFixedSize`, a wrapper
  for `CropToFixedSize(..., position="center")`.
* Added augmenter `CropToMultiplesOf`.
* Changed function `imgaug.imgaug.compute_paddings_for_aspect_ratio()`
  to also support shape tuples instead of ndarrays.
* Changed function `imgaug.imgaug.compute_paddings_to_reach_multiples_of()`
  to also support shape tuples instead of ndarrays.
* Added function `imgaug.imgaug.compute_croppings_to_reach_multiples_of()`.
