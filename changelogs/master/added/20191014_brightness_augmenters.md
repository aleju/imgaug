# New brightness augmenters

* Added augmenter `imgaug.augmenters.color.WithBrightnessChannels`.
* Added augmenter `imgaug.augmenters.color.MultiplyAndAddToBrightness`.
* Added method `imgaug.parameters.handle_categorical_string_param()`.
* Changed `change_colorspaces_()` to accept any iterable of `str` for
  argument `to_colorspaces`, not just `list`.
