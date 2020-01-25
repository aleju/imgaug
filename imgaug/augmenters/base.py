"""Base classes and functions used by all/most augmenters.

This module is planned to contain :class:`imgaug.augmenters.meta.Augmenter`
in the future.

"""


class SuspiciousMultiImageShapeWarning(UserWarning):
    """Warning multi-image inputs that look like a single image."""
