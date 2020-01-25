"""Base classes and functions used by all/most augmenters.

This module is planned to contain :class:`imgaug.augmenters.meta.Augmenter`
in the future.

"""
import imgaug as ia


class SuspiciousMultiImageShapeWarning(UserWarning):
    """Warning multi-image inputs that look like a single image."""


class SuspiciousSingleImageShapeWarning(UserWarning):
    """Warning for single-image inputs that look like multiple images."""


def _warn_on_suspicious_multi_image_shapes(images):
    if images is None:
        return

    # check if it looks like (H, W, C) instead of (N, H, W)
    if ia.is_np_array(images):
        if images.ndim == 3 and images.shape[-1] in [1, 3]:
            ia.warn(
                "You provided a numpy array of shape %s as a "
                "multi-image augmentation input, which was interpreted as "
                "(N, H, W). The last dimension however has value 1 or "
                "3, which indicates that you provided a single image "
                "with shape (H, W, C) instead. If that is the case, "
                "you should use e.g. augmenter(image=<your input>) or "
                "augment_image(<your input>) -- note the singular 'image' "
                "instead of 'imageS'. Otherwise your single input image "
                "will be interpreted as multiple images of shape (H, W) "
                "during augmentation." % (images.shape,),
                category=SuspiciousMultiImageShapeWarning)
