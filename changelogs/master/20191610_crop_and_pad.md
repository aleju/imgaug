# Changes to Crop and Pad augmenters #459

The following functions were moved. Their old names are now deprecated.
* Moved `imgaug.imgaug.pad` to `imgaug.augmenters.size.pad`
* Moved `imgaug.imgaug.pad_to_aspect_ratio` to
  `imgaug.augmenters.size.pad_to_aspect_ratio`.
* Moved `imgaug.imgaug.pad_to_multiples_of` to
  `imgaug.augmenters.size.pad_to_multiples_of`.
* Moved `imgaug.imgaug.compute_paddings_for_aspect_ratio` to
  `imgaug.augmenters.size.compute_paddings_to_reach_aspect_ratio`.
* Moved `imgaug.imgaug.compute_paddings_to_reach_multiples_of`
  to `imgaug.augmenters.size.compute_paddings_to_reach_multiples_of`.
* Moved `imgaug.imgaug.compute_paddings_to_reach_exponents_of`
  to `imgaug.augmenters.size.compute_paddings_to_reach_exponents_of`.


The following augmenters were added:
* Added augmenter `CenterCropToFixedSize`.
* Added augmenter `CenterPadToFixedSize`.
* Added augmenter `CropToMultiplesOf`.
* Added augmenter `CenterCropToMultiplesOf`.
* Added augmenter `PadToMultiplesOf`.
* Added augmenter `CenterPadToMultiplesOf`.
* Added augmenter `CropToExponentsOf`.
* Added augmenter `CenterCropToExponentsOf`.
* Added augmenter `PadToExponentsOf`.
* Added augmenter `CenterPadToExponentsOf`.
* Added augmenter `CropToAspectRatio`.
* Added augmenter `CenterCropToAspectRatio`.
* Added augmenter `PadToAspectRatio`.
* Added augmenter `CenterPadToAspectRatio`.
* Added augmenter `PadToSquare`.
* Added augmenter `CenterPadToSquare`.
* Added augmenter `CropToSquare`.
* Added augmenter `CenterCropToSquare`.

All `Center<name>` augmenters are wrappers around `<name>` with parameter
`position="center"`.


Added functions:
* Added function
  `imgaug.augmenters.size.compute_croppings_to_reach_aspect_ratio()`.
* Added function
  `imgaug.augmenters.size.compute_croppings_to_reach_multiples_of()`.
* Added function
  `imgaug.augmenters.size.compute_croppings_to_reach_exponents_of()`.
* Added function
  `imgaug.augmenters.size.compute_paddings_to_reach_exponents_of()`.


Other changes:
* Extended augmenter `CropToFixedSize` to support `height` and/or `width`
  parameters to be `None`, in which case the respective axis is not changed.
* Extended augmenter `PadToFixedSize` to support `height` and/or `width`
  parameters to be `None`, in which case the respective axis is not changed.
* [rarely breaking] Changed `CropToFixedSize.get_parameters()` to also
  return the `height` and `width` values.
* [rarely breaking] Changed `PadToFixedSize.get_parameters()` to also
  return the `height` and `width` values.
* [rarely breaking] Changed the order of parameters returned by
  `PadToFixedSize.get_parameters()` to match the order in
  `PadToFixedSize.__init__()`
* Changed `PadToFixedSize` to prefer padding the right side over the left side
  and the bottom side over the top side. E.g. if using a center pad and
  `3` columns have to be padded, it will pad `1` on the left and `2` on the
  right. Previously it was the other way round. This was changed to establish
  more consistency with the various other pad and crop methods.
* Changed the projection of pad/crop values between images and non-images
  to make the behaviour slightly more accurate in fringe cases.
* Improved behaviour of function
  `imgaug.augmenters.size.compute_paddings_for_aspect_ratio()` for zero-sized
  axes.
* Changed function `imgaug.augmenters.size.compute_paddings_for_aspect_ratio()`
  to also support shape tuples instead of only ndarrays.
* Changed function
  `imgaug.augmenters.size.compute_paddings_to_reach_multiples_of()`
  to also support shape tuples instead of only ndarrays.


Fixes:
* Fixed a formatting error in an error message of
  `compute_paddings_to_reach_multiples_of()`.
