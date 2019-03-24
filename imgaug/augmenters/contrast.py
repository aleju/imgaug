"""
Augmenters that perform contrast changes.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([iaa.GammaContrast((0.5, 1.5))])

List of augmenters:

    * GammaContrast
    * SigmoidContrast
    * LogContrast
    * LinearContrast
    * AllChannelsHistogramEqualization
    * HistogramEqualization
    * AllChannelsCLAHE
    * CLAHE

"""
from __future__ import print_function, division, absolute_import

import numpy as np
import six.moves as sm
import skimage.exposure as ski_exposure
import cv2
import warnings

from . import meta
from . import color as color_lib
import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt


# TODO quite similar to the other adjust_contrast_*() functions, make DRY
def adjust_contrast_gamma(arr, gamma):
    """
    Adjust contrast by scaling each pixel value to ``255 * ((I_ij/255)**gamma)``.

    dtype support::

        * ``uint8``: yes; fully tested (1) (2) (3)
        * ``uint16``: yes; tested (2) (3)
        * ``uint32``: yes; tested (2) (3)
        * ``uint64``: yes; tested (2) (3) (4)
        * ``int8``: limited; tested (2) (3) (5)
        * ``int16``: limited; tested (2) (3) (5)
        * ``int32``: limited; tested (2) (3) (5)
        * ``int64``: limited; tested (2) (3) (4) (5)
        * ``float16``: limited; tested (5)
        * ``float32``: limited; tested (5)
        * ``float64``: limited; tested (5)
        * ``float128``: no (6)
        * ``bool``: no (7)

        - (1) Handled by ``cv2``. Other dtypes are handled by ``skimage``.
        - (2) Normalization is done as ``I_ij/max``, where ``max`` is the maximum value of the
              dtype, e.g. 255 for ``uint8``. The normalization is reversed afterwards,
              e.g. ``result*255`` for ``uint8``.
        - (3) Integer-like values are not rounded after applying the contrast adjustment equation
              (before inverting the normalization to 0.0-1.0 space), i.e. projection from continuous
              space to discrete happens according to floor function.
        - (4) Note that scikit-image doc says that integers are converted to ``float64`` values before
              applying the contrast normalization method. This might lead to inaccuracies for large
              64bit integer values. Tests showed no indication of that happening though.
        - (5) Must not contain negative values. Values >=0 are fully supported.
        - (6) Leads to error in scikit-image.
        - (7) Does not make sense for contrast adjustments.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    gamma : number
        Exponent for the contrast adjustment. Higher values darken the image.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT , but here it seemed
    # like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype.name == "uint8":
        min_value, _center_value, max_value = iadt.get_value_range_of_dtype(arr.dtype)
        dynamic_range = max_value - min_value

        value_range = np.linspace(0, 1.0, num=dynamic_range+1, dtype=np.float32)
        # 255 * ((I_ij/255)**gamma)
        # using np.float32(.) here still works when the input is a numpy array of size 1
        table = (min_value + (value_range ** np.float32(gamma)) * dynamic_range)
        arr_aug = cv2.LUT(arr, np.clip(table, min_value, max_value).astype(arr.dtype))
        if arr.ndim == 3 and arr_aug.ndim == 2:
            return arr_aug[..., np.newaxis]
        return arr_aug
    else:
        return ski_exposure.adjust_gamma(arr, gamma)


# TODO quite similar to the other adjust_contrast_*() functions, make DRY
def adjust_contrast_sigmoid(arr, gain, cutoff):
    """
    Adjust contrast by scaling each pixel value to ``255 * 1/(1 + exp(gain*(cutoff - I_ij/255)))``.

    dtype support::

        * ``uint8``: yes; fully tested (1) (2) (3)
        * ``uint16``: yes; tested (2) (3)
        * ``uint32``: yes; tested (2) (3)
        * ``uint64``: yes; tested (2) (3) (4)
        * ``int8``: limited; tested (2) (3) (5)
        * ``int16``: limited; tested (2) (3) (5)
        * ``int32``: limited; tested (2) (3) (5)
        * ``int64``: limited; tested (2) (3) (4) (5)
        * ``float16``: limited; tested (5)
        * ``float32``: limited; tested (5)
        * ``float64``: limited; tested (5)
        * ``float128``: no (6)
        * ``bool``: no (7)

        - (1) Handled by ``cv2``. Other dtypes are handled by ``skimage``.
        - (2) Normalization is done as ``I_ij/max``, where ``max`` is the maximum value of the
              dtype, e.g. 255 for ``uint8``. The normalization is reversed afterwards,
              e.g. ``result*255`` for ``uint8``.
        - (3) Integer-like values are not rounded after applying the contrast adjustment equation
              (before inverting the normalization to 0.0-1.0 space), i.e. projection from continuous
              space to discrete happens according to floor function.
        - (4) Note that scikit-image doc says that integers are converted to ``float64`` values before
              applying the contrast normalization method. This might lead to inaccuracies for large
              64bit integer values. Tests showed no indication of that happening though.
        - (5) Must not contain negative values. Values >=0 are fully supported.
        - (6) Leads to error in scikit-image.
        - (7) Does not make sense for contrast adjustments.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    gain : number
        Multiplier for the sigmoid function's output.
        Higher values lead to quicker changes from dark to light pixels.

    cutoff : number
        Cutoff that shifts the sigmoid function in horizontal direction.
        Higher values mean that the switch from dark to light pixels happens later, i.e.
        the pixels will remain darker.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT , but here it seemed
    # like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype.name == "uint8":
        min_value, _center_value, max_value = iadt.get_value_range_of_dtype(arr.dtype)
        dynamic_range = max_value - min_value

        value_range = np.linspace(0, 1.0, num=dynamic_range+1, dtype=np.float32)
        # 255 * 1/(1 + exp(gain*(cutoff - I_ij/255)))
        # using np.float32(.) here still works when the input is a numpy array of size 1
        gain = np.float32(gain)
        cutoff = np.float32(cutoff)
        table = min_value + dynamic_range * 1/(1 + np.exp(gain * (cutoff - value_range)))
        arr_aug = cv2.LUT(arr, np.clip(table, min_value, max_value).astype(arr.dtype))
        if arr.ndim == 3 and arr_aug.ndim == 2:
            return arr_aug[..., np.newaxis]
        return arr_aug
    else:
        return ski_exposure.adjust_sigmoid(arr, cutoff=cutoff, gain=gain)


# TODO quite similar to the other adjust_contrast_*() functions, make DRY
def adjust_contrast_log(arr, gain):
    """
    Adjust contrast by scaling each pixel value to ``255 * gain * log_2(1 + I_ij/255)``.

    dtype support::

        * ``uint8``: yes; fully tested (1) (2) (3)
        * ``uint16``: yes; tested (2) (3)
        * ``uint32``: yes; tested (2) (3)
        * ``uint64``: yes; tested (2) (3) (4)
        * ``int8``: limited; tested (2) (3) (5)
        * ``int16``: limited; tested (2) (3) (5)
        * ``int32``: limited; tested (2) (3) (5)
        * ``int64``: limited; tested (2) (3) (4) (5)
        * ``float16``: limited; tested (5)
        * ``float32``: limited; tested (5)
        * ``float64``: limited; tested (5)
        * ``float128``: no (6)
        * ``bool``: no (7)

        - (1) Handled by ``cv2``. Other dtypes are handled by ``skimage``.
        - (2) Normalization is done as ``I_ij/max``, where ``max`` is the maximum value of the
              dtype, e.g. 255 for ``uint8``. The normalization is reversed afterwards,
              e.g. ``result*255`` for ``uint8``.
        - (3) Integer-like values are not rounded after applying the contrast adjustment equation
              (before inverting the normalization to 0.0-1.0 space), i.e. projection from continuous
              space to discrete happens according to floor function.
        - (4) Note that scikit-image doc says that integers are converted to ``float64`` values before
              applying the contrast normalization method. This might lead to inaccuracies for large
              64bit integer values. Tests showed no indication of that happening though.
        - (5) Must not contain negative values. Values >=0 are fully supported.
        - (6) Leads to error in scikit-image.
        - (7) Does not make sense for contrast adjustments.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    gain : number
        Multiplier for the logarithm result. Values around 1.0 lead to a contrast-adjusted
        images. Values above 1.0 quickly lead to partially broken images due to exceeding the
        datatype's value range.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT , but here it seemed
    # like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype.name == "uint8":
        min_value, _center_value, max_value = iadt.get_value_range_of_dtype(arr.dtype)
        dynamic_range = max_value - min_value

        value_range = np.linspace(0, 1.0, num=dynamic_range+1, dtype=np.float32)
        # 255 * 1/(1 + exp(gain*(cutoff - I_ij/255)))
        # using np.float32(.) here still works when the input is a numpy array of size 1
        gain = np.float32(gain)
        table = min_value + dynamic_range * gain * np.log2(1 + value_range)
        arr_aug = cv2.LUT(arr, np.clip(table, min_value, max_value).astype(arr.dtype))
        if arr.ndim == 3 and arr_aug.ndim == 2:
            return arr_aug[..., np.newaxis]
        return arr_aug
    else:
        return ski_exposure.adjust_log(arr, gain=gain)


# TODO quite similar to the other adjust_contrast_*() functions, make DRY
def adjust_contrast_linear(arr, alpha):
    """Adjust contrast by scaling each pixel value to ``127 + alpha*(I_ij-127)``.

    dtype support::

        * ``uint8``: yes; fully tested (1) (2)
        * ``uint16``: yes; tested (2)
        * ``uint32``: yes; tested (2)
        * ``uint64``: no (3)
        * ``int8``: yes; tested (2)
        * ``int16``: yes; tested (2)
        * ``int32``: yes; tested (2)
        * ``int64``: no (2)
        * ``float16``: yes; tested (2)
        * ``float32``: yes; tested (2)
        * ``float64``: yes; tested (2)
        * ``float128``: no (2)
        * ``bool``: no (4)

        - (1) Handled by ``cv2``. Other dtypes are handled by raw ``numpy``.
        - (2) Only tested for reasonable alphas with up to a value of around 100.
        - (3) Conversion to ``float64`` is done during augmentation, hence ``uint64``, ``int64``,
              and ``float128`` support cannot be guaranteed.
        - (4) Does not make sense for contrast adjustments.

    Parameters
    ----------
    arr : numpy.ndarray
        Array for which to adjust the contrast. Dtype ``uint8`` is fastest.

    alpha : number
        Multiplier to linearly pronounce (>1.0), dampen (0.0 to 1.0) or invert (<0.0) the
        difference between each pixel value and the center value, e.g. ``127`` for ``uint8``.

    Returns
    -------
    numpy.ndarray
        Array with adjusted contrast.

    """
    # int8 is also possible according to docs
    # https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html#cv2.LUT , but here it seemed
    # like `d` was 0 for CV_8S, causing that to fail
    if arr.dtype.name == "uint8":
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(arr.dtype)

        value_range = np.arange(0, 256, dtype=np.float32)
        # 127 + alpha*(I_ij-127)
        # using np.float32(.) here still works when the input is a numpy array of size 1
        alpha = np.float32(alpha)
        table = center_value + alpha * (value_range - center_value)
        arr_aug = cv2.LUT(arr, np.clip(table, min_value, max_value).astype(arr.dtype))
        if arr.ndim == 3 and arr_aug.ndim == 2:
            return arr_aug[..., np.newaxis]
        return arr_aug
    else:
        input_dtype = arr.dtype
        _min_value, center_value, _max_value = iadt.get_value_range_of_dtype(input_dtype)
        if input_dtype.kind in ["u", "i"]:
            center_value = int(center_value)
        image_aug = center_value + alpha * (arr.astype(np.float64)-center_value)
        image_aug = iadt.restore_dtypes_(image_aug, input_dtype)
        return image_aug


def GammaContrast(gamma=1, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Adjust contrast by scaling each pixel value to ``255 * ((I_ij/255)**gamma)``.

    Values in the range ``gamma=(0.5, 2.0)`` seem to be sensible.

    dtype support::

        See :func:`imgaug.augmenters.contrast.adjust_contrast_gamma`.

    Parameters
    ----------
    gamma : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Exponent for the contrast adjustment. Higher values darken the image.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]`` will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    per_channel :  bool or float, optional
        Whether to use the same value for all channels (False) or to sample a new value for each
        channel (True). If this value is a float ``p``, then for ``p`` percent of all images `per_channel`
        will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Returns
    -------
    _ContrastFuncWrapper
        Augmenter to perform gamma contrast adjustment.

    """
    params1d = [iap.handle_continuous_param(gamma, "gamma", value_range=None, tuple_to_uniform=True,
                                            list_to_choice=True)]
    func = adjust_contrast_gamma
    return _ContrastFuncWrapper(
        func, params1d, per_channel,
        dtypes_allowed=["uint8", "uint16", "uint32", "uint64",
                        "int8", "int16", "int32", "int64",
                        "float16", "float32", "float64"],
        dtypes_disallowed=["float96", "float128", "float256", "bool"],
        name=name if name is not None else ia.caller_name(),
        deterministic=deterministic,
        random_state=random_state
    )


def SigmoidContrast(gain=10, cutoff=0.5, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Adjust contrast by scaling each pixel value to ``255 * 1/(1 + exp(gain*(cutoff - I_ij/255)))``.

    Values in the range ``gain=(5, 20)`` and ``cutoff=(0.25, 0.75)`` seem to be sensible.

    dtype support::

        See :func:`imgaug.augmenters.contrast.adjust_contrast_sigmoid`.

    Parameters
    ----------
    gain : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier for the sigmoid function's output.
        Higher values lead to quicker changes from dark to light pixels.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]`` will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    cutoff : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Cutoff that shifts the sigmoid function in horizontal direction.
        Higher values mean that the switch from dark to light pixels happens later, i.e.
        the pixels will remain darker.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]`` will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    per_channel :  bool or float, optional
        Whether to use the same value for all channels (False) or to sample a new value for each
        channel (True). If this value is a float ``p``, then for ``p`` percent of all images `per_channel`
        will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Returns
    -------
    _ContrastFuncWrapper
        Augmenter to perform sigmoid contrast adjustment.

    """
    # TODO add inv parameter?
    params1d = [
        iap.handle_continuous_param(gain, "gain", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True),
        iap.handle_continuous_param(cutoff, "cutoff", value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True)
    ]
    func = adjust_contrast_sigmoid
    return _ContrastFuncWrapper(
        func, params1d, per_channel,
        dtypes_allowed=["uint8", "uint16", "uint32", "uint64",
                        "int8", "int16", "int32", "int64",
                        "float16", "float32", "float64"],
        dtypes_disallowed=["float96", "float128", "float256", "bool"],
        name=name if name is not None else ia.caller_name(),
        deterministic=deterministic,
        random_state=random_state
    )


def LogContrast(gain=1, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Adjust contrast by scaling each pixel value to ``255 * gain * log_2(1 + I_ij/255)``.

    dtype support::

        See :func:`imgaug.augmenters.contrast.adjust_contrast_log`.

    Parameters
    ----------
    gain : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier for the logarithm result. Values around 1.0 lead to a contrast-adjusted
        images. Values above 1.0 quickly lead to partially broken images due to exceeding the
        datatype's value range.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]`` will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    per_channel :  bool or float, optional
        Whether to use the same value for all channels (False) or to sample a new value for each
        channel (True). If this value is a float ``p``, then for ``p`` percent of all images `per_channel`
        will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Returns
    -------
    _ContrastFuncWrapper
        Augmenter to perform logarithmic contrast adjustment.

    """
    # TODO add inv parameter?
    params1d = [iap.handle_continuous_param(gain, "gain", value_range=(0, None), tuple_to_uniform=True,
                                            list_to_choice=True)]
    func = adjust_contrast_log
    return _ContrastFuncWrapper(
        func, params1d, per_channel,
        dtypes_allowed=["uint8", "uint16", "uint32", "uint64",
                        "int8", "int16", "int32", "int64",
                        "float16", "float32", "float64"],
        dtypes_disallowed=["float96", "float128", "float256", "bool"],
        name=name if name is not None else ia.caller_name(),
        deterministic=deterministic,
        random_state=random_state
    )


def LinearContrast(alpha=1, per_channel=False, name=None, deterministic=False, random_state=None):
    """Adjust contrast by scaling each pixel value to ``127 + alpha*(I_ij-127)``.

    dtype support::

        See :func:`imgaug.augmenters.contrast.adjust_contrast_linear`.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier to linearly pronounce (>1.0), dampen (0.0 to 1.0) or invert (<0.0) the
        difference between each pixel value and the center value, e.g. ``127`` for ``uint8``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]`` will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    per_channel :  bool or float, optional
        Whether to use the same value for all channels (False) or to sample a new value for each
        channel (True). If this value is a float ``p``, then for ``p`` percent of all images `per_channel`
        will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Returns
    -------
    _ContrastFuncWrapper
        Augmenter to perform contrast adjustment by linearly scaling the distance to 128.

    """
    params1d = [
        iap.handle_continuous_param(alpha, "alpha", value_range=None, tuple_to_uniform=True, list_to_choice=True)
    ]
    func = adjust_contrast_linear
    return _ContrastFuncWrapper(
        func, params1d, per_channel,
        dtypes_allowed=["uint8", "uint16", "uint32",
                        "int8", "int16", "int32",
                        "float16", "float32", "float64"],
        dtypes_disallowed=["uint64", "int64", "float96", "float128", "float256", "bool"],
        name=name if name is not None else ia.caller_name(),
        deterministic=deterministic,
        random_state=random_state
    )


# TODO maybe offer the other contrast augmenters also wrapped in this, similar to CLAHE and HistogramEqualization?
# this is essentially tested by tests for CLAHE
class _IntensityChannelBasedApplier(object):
    RGB = color_lib.ChangeColorspace.RGB
    BGR = color_lib.ChangeColorspace.BGR
    HSV = color_lib.ChangeColorspace.HSV
    HLS = color_lib.ChangeColorspace.HLS
    Lab = color_lib.ChangeColorspace.Lab
    _CHANNEL_MAPPING = {
        HSV: 2,
        HLS: 1,
        Lab: 0
    }

    def __init__(self, from_colorspace, to_colorspace, name):
        super(_IntensityChannelBasedApplier, self).__init__()

        # TODO maybe add CIE, Luv?
        ia.do_assert(from_colorspace in [self.RGB,
                                         self.BGR,
                                         self.Lab,
                                         self.HLS,
                                         self.HSV])
        ia.do_assert(to_colorspace in [self.Lab,
                                       self.HLS,
                                       self.HSV])

        self.change_colorspace = color_lib.ChangeColorspace(
            to_colorspace=to_colorspace,
            from_colorspace=from_colorspace,
            name="%s_IntensityChannelBasedApplier_ChangeColorspace" % (name,))
        self.change_colorspace_inv = color_lib.ChangeColorspace(
            to_colorspace=from_colorspace,
            from_colorspace=to_colorspace,
            name="%s_IntensityChannelBasedApplier_ChangeColorspaceInverse" % (name,))

    def apply(self, images, random_state, parents, hooks, func):
        input_was_array = ia.is_np_array(images)
        rss = ia.derive_random_states(random_state, 3)

        # normalize images
        # (H, W, 1) will be used directly in AllChannelsCLAHE
        # (H, W, 3) will be converted to target colorspace in the next block
        # (H, W, 4) will be reduced to (H, W, 3) (remove 4th channel) and converted to target colorspace in next block
        # (H, W, <else>) will raise a warning and be treated channelwise by AllChannelsCLAHE
        images_normalized = []
        images_change_cs = []
        images_change_cs_indices = []
        for i, image in enumerate(images):
            nb_channels = image.shape[2]
            if nb_channels == 1:
                images_normalized.append(image)
            elif nb_channels == 3:
                images_normalized.append(None)
                images_change_cs.append(image)
                images_change_cs_indices.append(i)
            elif nb_channels == 4:
                # assume that 4th channel is an alpha channel, e.g. in RGBA
                images_normalized.append(None)
                images_change_cs.append(image[..., 0:3])
                images_change_cs_indices.append(i)
            else:
                warnings.warn("Got image with %d channels in _IntensityChannelBasedApplier (parents: %s), "
                              "expected 0, 1, 3 or 4 channels." % (
                                  nb_channels, ", ".join(parent.name for parent in parents)))
                images_normalized.append(image)

        # convert colorspaces of normalized 3-channel images
        images_after_color_conversion = [None] * len(images_normalized)
        if len(images_change_cs) > 0:
            images_new_cs = self.change_colorspace._augment_images(images_change_cs, rss[0], parents + [self], hooks)
            for image_new_cs, target_idx in zip(images_new_cs, images_change_cs_indices):
                chan_idx = self._CHANNEL_MAPPING[self.change_colorspace.to_colorspace.value]
                images_normalized[target_idx] = image_new_cs[..., chan_idx:chan_idx+1]
                images_after_color_conversion[target_idx] = image_new_cs

        # apply CLAHE channelwise
        # images_aug = self.all_channel_clahe._augment_images(images_normalized, rss[1], parents + [self], hooks)
        images_aug = func(images_normalized, rss[1])

        # denormalize
        result = []
        images_change_cs = []
        images_change_cs_indices = []
        for i, (image, image_conv, image_aug) in enumerate(zip(images, images_after_color_conversion, images_aug)):
            nb_channels = image.shape[2]
            if nb_channels in [3, 4]:
                chan_idx = self._CHANNEL_MAPPING[self.change_colorspace.to_colorspace.value]
                image_tmp = image_conv
                image_tmp[..., chan_idx:chan_idx+1] = image_aug

                result.append(None if nb_channels == 3 else image[..., 3:4])
                images_change_cs.append(image_tmp)
                images_change_cs_indices.append(i)
            else:
                result.append(image_aug)

        # invert colorspace conversion
        if len(images_change_cs) > 0:
            images_new_cs = self.change_colorspace_inv._augment_images(images_change_cs, rss[0], parents + [self],
                                                                       hooks)
            for image_new_cs, target_idx in zip(images_new_cs, images_change_cs_indices):
                if result[target_idx] is None:
                    result[target_idx] = image_new_cs
                else:  # input image had four channels, 4th channel is already in result
                    result[target_idx] = np.dstack((image_new_cs, result[target_idx]))

        # convert to array if necessary
        if input_was_array:
            result = np.array(result, dtype=result[0].dtype)

        return result


# TODO add parameter `tile_grid_size_percent`
class AllChannelsCLAHE(meta.Augmenter):
    """
    Contrast Limited Adaptive Histogram Equalization, applied to all channels of the input images.

    CLAHE performs histogram equilization within image patches, i.e. over local neighbourhoods.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: no (2)
        * ``int16``: no (2)
        * ``int32``: no (2)
        * ``int64``: no (2)
        * ``float16``: no (2)
        * ``float32``: no (2)
        * ``float64``: no (2)
        * ``float128``: no (1)
        * ``bool``: no (1)

        - (1) rejected by cv2
        - (2) results in error in cv2: ``cv2.error: OpenCV(3.4.2) (...)/clahe.cpp:351: error: (-215:Assertion failed)
              src.type() == (((0) & ((1 << 3) - 1)) + (((1)-1) << 3))
              || _src.type() == (((2) & ((1 << 3) - 1)) + (((1)-1) << 3)) in function 'apply'``

    Parameters
    ----------
    clip_limit : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See ``imgaug.augmenters.contrast.CLAHE``.

    tile_grid_size_px : int or tuple of int or list of int or imgaug.parameters.StochasticParameter \
                        or tuple of tuple of int or tuple of list of int \
                        or tuple of imgaug.parameters.StochasticParameter, optional
        See ``imgaug.augmenters.contrast.CLAHE``.

    tile_grid_size_px_min : int, optional
        See ``imgaug.augmenters.contrast.CLAHE``.

    per_channel : bool or float, optional
        Whether to use the same values for all channels (False)
        or to sample new values for each channel (True).
        If this parameter is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    """
    def __init__(self, clip_limit=40, tile_grid_size_px=8, tile_grid_size_px_min=3, per_channel=False, name=None,
                 deterministic=False, random_state=None):
        super(AllChannelsCLAHE, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.clip_limit = iap.handle_continuous_param(clip_limit, "clip_limit", value_range=(0+1e-4, None),
                                                      tuple_to_uniform=True, list_to_choice=True)
        self.tile_grid_size_px = iap.handle_discrete_kernel_size_param(tile_grid_size_px, "tile_grid_size_px",
                                                                       value_range=(0, None),
                                                                       allow_floats=False)
        self.tile_grid_size_px_min = tile_grid_size_px_min
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["uint8", "uint16"],
                         disallowed=["bool",
                                     "uint32", "uint64", "uint128", "uint256",
                                     "int8", "int16", "int32", "int64", "int128", "int256",
                                     "float16", "float32", "float64", "float96", "float128", "float256"],
                         augmenter=self)

        nb_images = len(images)
        nb_channels = meta.estimate_max_number_of_channels(images)

        mode = "single" if self.tile_grid_size_px[1] is None else "two"
        rss = ia.derive_random_states(random_state, 3 if mode == "single" else 4)
        per_channel = self.per_channel.draw_samples((nb_images,), random_state=rss[0])
        clip_limit = self.clip_limit.draw_samples((nb_images, nb_channels), random_state=rss[1])
        tile_grid_size_px_h = self.tile_grid_size_px[0].draw_samples((nb_images, nb_channels), random_state=rss[2])
        if mode == "single":
            tile_grid_size_px_w = tile_grid_size_px_h
        else:
            tile_grid_size_px_w = self.tile_grid_size_px[1].draw_samples((nb_images, nb_channels), random_state=rss[3])

        tile_grid_size_px_w = np.maximum(tile_grid_size_px_w, self.tile_grid_size_px_min)
        tile_grid_size_px_h = np.maximum(tile_grid_size_px_h, self.tile_grid_size_px_min)

        gen = enumerate(zip(images, clip_limit, tile_grid_size_px_h, tile_grid_size_px_w, per_channel))
        for i, (image, clip_limit_i, tgs_px_h_i, tgs_px_w_i, per_channel_i) in gen:
            nb_channels = image.shape[2]
            c_param = 0
            image_warped = []
            for c in sm.xrange(nb_channels):
                if tgs_px_w_i[c_param] > 1 or tgs_px_h_i[c_param] > 1:
                    clahe = cv2.createCLAHE(clipLimit=clip_limit_i[c_param],
                                            tileGridSize=(tgs_px_w_i[c_param], tgs_px_h_i[c_param]))
                    channel_warped = clahe.apply(image[..., c])
                    image_warped.append(channel_warped)
                else:
                    image_warped.append(image[..., c])
                if per_channel_i > 0.5:
                    c_param += 1

            # combine channels to one image
            image_warped = np.array(image_warped, dtype=image_warped[0].dtype)
            image_warped = image_warped.transpose((1, 2, 0))

            images[i] = image_warped
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.clip_limit, self.tile_grid_size_px, self.tile_grid_size_px_min, self.per_channel]


class CLAHE(meta.Augmenter):
    """
    Contrast Limited Adaptive Histogram Equalization.

    This augmenter applies CLAHE to images, a form of histogram equalization that normalizes within local image
    patches.
    The augmenter transforms input images to a target colorspace (e.g. ``Lab``), extracts an intensity-related channel
    from the converted images (e.g. ``L`` for ``Lab``), applies CLAHE to the channel and then converts the resulting
    image back to the original colorspace.

    Grayscale images (images without channel axis or with only one channel axis) are automatically handled,
    `from_colorspace` does not have to be adjusted for them. For images with four channels (e.g. ``RGBA``), the fourth
    channel is ignored in the colorspace conversion (e.g. from an ``RGBA`` image, only the ``RGB`` part is converted,
    normalized, converted back and concatenated with the input ``A`` channel).
    Images with unusual channel numbers (2, 5 or more than 5) are normalized channel-by-channel (same behaviour as
    ``AllChannelsCLAHE``, though a warning will be raised).

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: no (1)
        * ``uint32``: no (1)
        * ``uint64``: no (1)
        * ``int8``: no (1)
        * ``int16``: no (1)
        * ``int32``: no (1)
        * ``int64``: no (1)
        * ``float16``: no (1)
        * ``float32``: no (1)
        * ``float64``: no (1)
        * ``float128``: no (1)
        * ``bool``: no (1)

        - (1) This augmenter uses ChangeColorspace, which is currently limited to ``uint8``.

    Parameters
    ----------
    clip_limit : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Clipping limit. Higher values result in stronger contrast. OpenCV uses a default of ``40``, though
        values around ``5`` seem to already produce decent contrast.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the range ``[a, b]`` will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    tile_grid_size_px : int or tuple of int or list of int or imgaug.parameters.StochasticParameter \
                        or tuple of tuple of int or tuple of list of int \
                        or tuple of imgaug.parameters.StochasticParameter, optional
        Kernel size, i.e. size of each local neighbourhood in pixels.

            * If an int, then that value will be used for all images for both kernel height and width.
            * If a tuple ``(a, b)``, then a value from the discrete range ``[a..b]`` will be sampled per
              image.
            * If a list, then a random value will be sampled from that list per image and used for both
              kernel height and width.
            * If a StochasticParameter, then a value will be sampled per image from that parameter per
              image and used for both kernel height and width.
            * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two values will be sampled
              independently from the discrete ranges ``[a..b]`` and ``[c..d]`` per image and used as
              the kernel height and width.
            * If a tuple of lists of int, then two values will be sampled independently per image, one
              from the first list and one from the second, and used as the kernel height and width.
            * If a tuple of StochasticParameter, then two values will be sampled indepdently per image,
              one from the first parameter and one from the second, and used as the kernel height and
              width.

    tile_grid_size_px_min : int, optional
        Minimum kernel size in px, per axis. If the sampling results in a value lower than this minimum,
        it will be clipped to this value.

    from_colorspace : {"RGB", "BGR", "HSV", "HLS", "Lab"}, optional
        Colorspace of the input images.
        If any input image has only one or zero channels, this setting will be ignored and it will be assumed that
        the input is grayscale.
        If a fourth channel is present in an input image, it will be removed before the colorspace conversion and
        later re-added.
        See also ``imgaug.augmenters.color.ChangeColorspace`` for details.

    to_colorspace : {"Lab", "HLS", "HSV"}, optional
        Colorspace in which to perform CLAHE. For ``Lab``, CLAHE will only be applied to the first channel (``L``),
        for ``HLS`` to the second (``L``) and for ``HSV`` to the third (``V``).
        To apply CLAHE to all channels of an input image (without colorspace conversion),
        see ``imgaug.augmenters.contrast.AllChannelsCLAHE``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.CLAHE()

    Creates a standard CLAHE augmenter.

    >>> aug = iaa.CLAHE(clip_limit=(1, 50))

    Creates a CLAHE augmenter with a clip limit uniformly sampled from ``[1..50]``, where ``1`` is rather low contrast
    and ``50`` is rather high contrast.

    >>> aug = iaa.CLAHE(tile_grid_size_px=(3, 21))

    Creates a CLAHE augmenter with kernel sizes of ``SxS``, where ``S`` is uniformly sampled from ``[3..21]``.
    Sampling happens once per image.

    >>> aug = iaa.CLAHE(tile_grid_size_px=iap.Discretize(iap.Normal(loc=7, scale=2)), tile_grid_size_px_min=3)

    Creates a CLAHE augmenter with kernel sizes of ``SxS``, where ``S`` is sampled from ``N(7, 2)``, but does not go
    below ``3``.

    >>> aug = iaa.CLAHE(tile_grid_size_px=((3, 21), [3, 5, 7]))

    Creates a CLAHE augmenter with kernel sizes of ``HxW``, where ``H`` is uniformly sampled from ``[3..21]`` and
    ``W`` is randomly picked from the list ``[3, 5, 7]``.

    >>> aug = iaa.CLAHE(from_colorspace=iaa.CLAHE.BGR, to_colorspace=iaa.CLAHE.HSV)

    Creates a CLAHE augmenter that converts images from BGR colorspace to HSV colorspace and then applies the local
    histogram equalization to the ``V`` channel of the images (before converting back to ``BGR``). Alternatively,
    ``Lab`` (default) or ``HLS`` can be used as the target colorspace. Grayscale images (no channels / one channel)
    are never converted and are instead directly normalized (i.e. `from_colorspace` does not have to be changed for
    them).

    """
    RGB = _IntensityChannelBasedApplier.RGB
    BGR = _IntensityChannelBasedApplier.BGR
    HSV = _IntensityChannelBasedApplier.HSV
    HLS = _IntensityChannelBasedApplier.HLS
    Lab = _IntensityChannelBasedApplier.Lab

    def __init__(self, clip_limit=40, tile_grid_size_px=8, tile_grid_size_px_min=3,
                 from_colorspace=color_lib.ChangeColorspace.RGB, to_colorspace=color_lib.ChangeColorspace.Lab,
                 name=None, deterministic=False, random_state=None):
        super(CLAHE, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.all_channel_clahe = AllChannelsCLAHE(clip_limit=clip_limit,
                                                  tile_grid_size_px=tile_grid_size_px,
                                                  tile_grid_size_px_min=tile_grid_size_px_min,
                                                  name="%s_AllChannelsCLAHE" % (name,))

        self.intensity_channel_based_applier = _IntensityChannelBasedApplier(from_colorspace, to_colorspace, name=name)

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["uint8"],
                         disallowed=["bool",
                                     "uint16", "uint32", "uint64", "uint128", "uint256",
                                     "int8", "int16", "int32", "int64", "int128", "int256",
                                     "float16", "float32", "float64", "float96", "float128", "float256"],
                         augmenter=self)

        def _augment_all_channels_clahe(images_normalized, random_state_derived):
            return self.all_channel_clahe._augment_images(images_normalized, random_state_derived, parents + [self],
                                                          hooks)

        return self.intensity_channel_based_applier.apply(images, random_state, parents + [self], hooks,
                                                          _augment_all_channels_clahe)

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.all_channel_clahe.clip_limit,
                self.all_channel_clahe.tile_grid_size_px,
                self.all_channel_clahe.tile_grid_size_px_min,
                self.intensity_channel_based_applier.change_colorspace.from_colorspace,  # from_colorspace is always str
                self.intensity_channel_based_applier.change_colorspace.to_colorspace.value]


class AllChannelsHistogramEqualization(meta.Augmenter):
    """
    Augmenter to perform standard histogram equalization on images, applied to all channels of each input image.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: no (1)
        * ``uint32``: no (2)
        * ``uint64``: no (1)
        * ``int8``: no (1)
        * ``int16``: no (1)
        * ``int32``: no (1)
        * ``int64``: no (1)
        * ``float16``: no (2)
        * ``float32``: no (1)
        * ``float64``: no (1)
        * ``float128``: no (2)
        * ``bool``: no (1)

        - (1) causes cv2 error: ``cv2.error: OpenCV(3.4.5) (...)/histogram.cpp:3345: error: (-215:Assertion failed)
              src.type() == CV_8UC1 in function 'equalizeHist'``
        - (2) rejected by cv2

    Parameters
    ----------
    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    """
    def __init__(self, name=None, deterministic=False, random_state=None):
        super(AllChannelsHistogramEqualization, self).__init__(name=name, deterministic=deterministic,
                                                               random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["uint8"],
                         disallowed=["bool",
                                     "uint16", "uint32", "uint64", "uint128", "uint256",
                                     "int8", "int16", "int32", "int64", "int128", "int256",
                                     "float16", "float32", "float64", "float96", "float128", "float256"],
                         augmenter=self)

        for i, image in enumerate(images):
            image_warped = [cv2.equalizeHist(image[..., c]) for c in sm.xrange(image.shape[2])]
            image_warped = np.array(image_warped, dtype=image_warped[0].dtype)
            image_warped = image_warped.transpose((1, 2, 0))

            images[i] = image_warped
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []


class HistogramEqualization(meta.Augmenter):
    """
    Augmenter to apply standard histogram equalization to images.

    This augmenter is similar to ``imgaug.augmenters.contrast.CLAHE``.

    The augmenter transforms input images to a target colorspace (e.g. ``Lab``), extracts an intensity-related channel
    from the converted images (e.g. ``L`` for ``Lab``), applies Histogram Equalization to the channel and then
    converts the resulting image back to the original colorspace.

    Grayscale images (images without channel axis or with only one channel axis) are automatically handled,
    `from_colorspace` does not have to be adjusted for them. For images with four channels (e.g. RGBA), the fourth
    channel is ignored in the colorspace conversion (e.g. from an ``RGBA`` image, only the ``RGB`` part is converted,
    normalized, converted back and concatenated with the input ``A`` channel).
    Images with unusual channel numbers (2, 5 or more than 5) are normalized channel-by-channel (same behaviour as
    ``AllChannelsHistogramEqualization``, though a warning will be raised).

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: no (1)
        * ``uint32``: no (1)
        * ``uint64``: no (1)
        * ``int8``: no (1)
        * ``int16``: no (1)
        * ``int32``: no (1)
        * ``int64``: no (1)
        * ``float16``: no (1)
        * ``float32``: no (1)
        * ``float64``: no (1)
        * ``float128``: no (1)
        * ``bool``: no (1)

        - (1) This augmenter uses AllChannelsHistogramEqualization, which only supports ``uint8``.

    Parameters
    ----------
    from_colorspace : {"RGB", "BGR", "HSV", "HLS", "Lab"}, optional
        Colorspace of the input images.
        If any input image has only one or zero channels, this setting will be ignored and it will be assumed that
        the input is grayscale.
        If a fourth channel is present in an input image, it will be removed before the colorspace conversion and
        later re-added.
        See also ``imgaug.augmenters.color.ChangeColorspace`` for details.

    to_colorspace : {"Lab", "HLS", "HSV"}, optional
        Colorspace in which to perform Histogram Equalization. For ``Lab``, the equalization will only be applied to
        the first channel (``L``), for ``HLS`` to the second (``L``) and for ``HSV`` to the third (``V``).
        To apply histogram equalization to all channels of an input image (without colorspace conversion),
        see ``imgaug.augmenters.contrast.AllChannelsHistogramEqualization``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.HistogramEqualization()

    Creates a standard histogram equalization augmenter.

    >>> aug = iaa.HistogramEqualization(from_colorspace=iaa.HistogramEqualization.BGR,
    >>>                                 to_colorspace=iaa.HistogramEqualization.HSV)

    Creates a histogram equalization augmenter that converts images from BGR colorspace to HSV colorspace and then
    applies the local histogram equalization to the ``V`` channel of the images (before converting back to ``BGR``).
    Alternatively, ``Lab`` (default) or ``HLS`` can be used as the target colorspace. Grayscale images
    (no channels / one channel) are never converted and are instead directly normalized (i.e. `from_colorspace` does
    not have to be changed for them).

    """
    RGB = _IntensityChannelBasedApplier.RGB
    BGR = _IntensityChannelBasedApplier.BGR
    HSV = _IntensityChannelBasedApplier.HSV
    HLS = _IntensityChannelBasedApplier.HLS
    Lab = _IntensityChannelBasedApplier.Lab

    def __init__(self,  from_colorspace=color_lib.ChangeColorspace.RGB, to_colorspace=color_lib.ChangeColorspace.Lab,
                 name=None, deterministic=False, random_state=None):
        super(HistogramEqualization, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.all_channel_histogram_equalization = AllChannelsHistogramEqualization(
            name="%s_AllChannelsHistogramEqualization" % (name,))

        self.intensity_channel_based_applier = _IntensityChannelBasedApplier(from_colorspace, to_colorspace, name=name)

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["uint8"],
                         disallowed=["bool",
                                     "uint16", "uint32", "uint64", "uint128", "uint256",
                                     "int8", "int16", "int32", "int64", "int128", "int256",
                                     "float16", "float32", "float64", "float96", "float128", "float256"],
                         augmenter=self)

        def _augment_all_channels_histogram_equalization(images_normalized, random_state_derived):
            return self.all_channel_histogram_equalization._augment_images(images_normalized, random_state_derived,
                                                                           parents + [self], hooks)

        return self.intensity_channel_based_applier.apply(images, random_state, parents + [self], hooks,
                                                          _augment_all_channels_histogram_equalization)

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.intensity_channel_based_applier.change_colorspace.from_colorspace,  # from_colorspace is always str
                self.intensity_channel_based_applier.change_colorspace.to_colorspace.value]


class _ContrastFuncWrapper(meta.Augmenter):
    def __init__(self, func, params1d, per_channel, dtypes_allowed=None, dtypes_disallowed=None,
                 name=None, deterministic=False, random_state=None):
        super(_ContrastFuncWrapper, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.func = func
        self.params1d = params1d
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")
        self.dtypes_allowed = dtypes_allowed
        self.dtypes_disallowed = dtypes_disallowed

    def _augment_images(self, images, random_state, parents, hooks):
        if self.dtypes_allowed is not None:
            iadt.gate_dtypes(images,
                             allowed=self.dtypes_allowed,
                             disallowed=self.dtypes_disallowed,
                             augmenter=self)

        nb_images = len(images)
        rss = ia.derive_random_states(random_state, 1+nb_images)
        per_channel = self.per_channel.draw_samples((nb_images,), random_state=rss[0])

        result = images
        for i, (image, per_channel_i, rs) in enumerate(zip(images, per_channel, rss[1:])):
            nb_channels = 1 if per_channel_i <= 0.5 else image.shape[2]
            samples_i = [param.draw_samples((nb_channels,), random_state=rs) for param in self.params1d]
            if per_channel_i > 0.5:
                input_dtype = image.dtype
                image_aug = image.astype(np.float64)
                for c in sm.xrange(nb_channels):
                    samples_i_c = [sample_i[c] for sample_i in samples_i]
                    args = tuple([image[..., c]] + samples_i_c)
                    image_aug[..., c] = self.func(*args)
                image_aug = image_aug.astype(input_dtype)
            else:
                # don't use something like samples_i[...][0] here, because that returns python scalars and is slightly
                # less accurate than keeping the numpy values
                args = tuple([image] + samples_i)
                image_aug = self.func(*args)
            result[i] = image_aug
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return self.params1d


# TODO delete this or maybe move it somewhere else
"""
class _PreserveDtype(object):
    def __init__(self, func, adjust_value_range=False):
        self.func = func
        self.adjust_value_range = adjust_value_range

    def __call__(self, *args, **kwargs):
        image = args[0]
        input_dtype = image.dtype
        image_aug = self.func(image, *args[1:], **kwargs)
        if input_dtype.type == np.uint8:
            if self.adjust_value_range:
                image_aug = image_aug * 255
            image_aug = meta.clip_augmented_image_(image_aug, 0, 255)
        image_aug = meta.restore_augmented_image_dtype_(image_aug, input_dtype)

        return image_aug
"""
