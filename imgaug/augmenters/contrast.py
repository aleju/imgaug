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

"""
from __future__ import print_function, division, absolute_import

import numpy as np
import six.moves as sm
import skimage.exposure as ski_exposure

from . import meta
from .. import imgaug as ia
from .. import parameters as iap


def GammaContrast(gamma=1, per_channel=False, name=None, deterministic=False, random_state=None):
    """
    Adjust contrast by scaling each pixel value to ``255 * ((I_ij/255)**gamma)``.

    Values in the range ``gamma=(0.5, 2.0)`` seem to be sensible.

    dtype support::

        * ``uint8``: yes; fully tested (1) (2)
        * ``uint16``: yes; tested (1) (2)
        * ``uint32``: yes; tested (1) (2)
        * ``uint64``: yes; tested (1) (2) (3)
        * ``int8``: limited; tested (1) (2) (4)
        * ``int16``: limited; tested (1) (2) (4)
        * ``int32``: limited; tested (1) (2) (4)
        * ``int64``: limited; tested (1) (2) (3) (4)
        * ``float16``: limited; tested (4)
        * ``float32``: limited; tested (4)
        * ``float64``: limited; tested (4)
        * ``float128``: no (5)
        * ``bool``: no (6)

        - (1) Normalization is done as ``I_ij/max``, where ``max`` is the maximum value of the
              dtype, e.g. 255 for ``uint8``. The normalization is reversed afterwards,
              e.g. ``result*255`` for uint8.
        - (2) Integer-like values are not rounded after applying the contrast adjustment equation
              (before inverting the normalization to 0.0-1.0 space), i.e. projection from continous
              space to discrete happens according to floor function.
        - (3) Note that scikit-image doc says that integers are converted to float64 values before
              applying the contrast normalization method. This might lead to inaccuracies for large
              64bit integer values. Tests showed no indication of that happening though.
        - (4) Must not contain negative values. Values >=0 are fully supported.
        - (5) Leads to error in scikit-image.
        - (6) Does not make sense for contrast adjustments.

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
    func = _PreserveDtype(ski_exposure.adjust_gamma)

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

        * ``uint8``: yes; fully tested (1) (2)
        * ``uint16``: yes; tested (1) (2)
        * ``uint32``: yes; tested (1) (2)
        * ``uint64``: yes; tested (1) (2) (3)
        * ``int8``: limited; tested (1) (2) (4)
        * ``int16``: limited; tested (1) (2) (4)
        * ``int32``: limited; tested (1) (2) (4)
        * ``int64``: limited; tested (1) (2) (3) (4)
        * ``float16``: limited; tested (4)
        * ``float32``: limited; tested (4)
        * ``float64``: limited; tested (4)
        * ``float128``: no (5)
        * ``bool``: no (6)

        - (1) Normalization is done as ``I_ij/max``, where ``max`` is the maximum value of the
              dtype, e.g. 255 for ``uint8``. The normalization is reversed afterwards,
              e.g. ``result*255`` for uint8.
        - (2) Integer-like values are not rounded after applying the contrast adjustment equation
              (before inverting the normalization to 0.0-1.0 space), i.e. projection from continous
              space to discrete happens according to floor function.
        - (3) Note that scikit-image doc says that integers are converted to float64 values before
              applying the contrast normalization method. This might lead to inaccuracies for large
              64bit integer values. Tests showed no indication of that happening though.
        - (4) Must not contain negative values. Values >=0 are fully supported.
        - (5) Leads to error in scikit-image.
        - (6) Does not make sense for contrast adjustments.

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
        iap.handle_continuous_param(cutoff, "cutoff", value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True),
        iap.handle_continuous_param(gain, "gain", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True)
    ]
    func = _PreserveDtype(ski_exposure.adjust_sigmoid)
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

        * ``uint8``: yes; fully tested (1) (2)
        * ``uint16``: yes; tested (1) (2)
        * ``uint32``: yes; tested (1) (2)
        * ``uint64``: yes; tested (1) (2) (3)
        * ``int8``: limited; tested (1) (2) (4)
        * ``int16``: limited; tested (1) (2) (4)
        * ``int32``: limited; tested (1) (2) (4)
        * ``int64``: limited; tested (1) (2) (3) (4)
        * ``float16``: limited; tested (4)
        * ``float32``: limited; tested (4)
        * ``float64``: limited; tested (4)
        * ``float128``: no (5)
        * ``bool``: no (6)

        - (1) Normalization is done as ``I_ij/max``, where ``max`` is the maximum value of the
              dtype, e.g. 255 for ``uint8``. The normalization is reversed afterwards,
              e.g. ``result*255`` for uint8.
        - (2) Integer-like values are not rounded after applying the contrast adjustment equation
              (before inverting the normalization to 0.0-1.0 space), i.e. projection from continous
              space to discrete happens according to floor function.
        - (3) Note that scikit-image doc says that integers are converted to float64 values before
              applying the contrast normalization method. This might lead to inaccuracies for large
              64bit integer values. Tests showed no indication of that happening though.
        - (4) Must not contain negative values. Values >=0 are fully supported.
        - (5) Leads to error in scikit-image.
        - (6) Does not make sense for contrast adjustments.

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
    func = _PreserveDtype(ski_exposure.adjust_log)
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

        * ``uint8``: yes; fully tested (1)
        * ``uint16``: yes; tested (1)
        * ``uint32``: yes; tested (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (1)
        * ``int16``: yes; tested (1)
        * ``int32``: yes; tested (1)
        * ``int64``: no (1)
        * ``float16``: yes; tested (1)
        * ``float32``: yes; tested (1)
        * ``float64``: yes; tested (1)
        * ``float128``: no (1)
        * ``bool``: no (3)

        - (1) Only tested for reasonable alphas with up to a value of around 100.
        - (2) Conversion to ``float64`` is done during augmentation, hence ``uint64``, ``int64``,
              and ``float128`` support cannot be guaranteed.
        - (3) Does not make sense for contrast adjustments.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier to linearly pronounce (>1.0), dampen (0.0 to 1.0) or invert (<0.0) the
        difference between each pixel value and the center value, i.e. `128`.

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
    func = _adjust_linear
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
            meta.gate_dtypes(images,
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


def _adjust_linear(image, alpha):
    input_dtype = image.dtype
    _min_value, center_value, _max_value = meta.get_value_range_of_dtype(input_dtype)
    if input_dtype.kind in ["u", "i"]:
        center_value = int(center_value)
    image_aug = center_value + alpha * (image.astype(np.float64)-center_value)
    image_aug = meta.restore_dtypes_(image_aug, input_dtype)
    return image_aug
