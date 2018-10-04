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
from .. import imgaug as ia
from .. import parameters as iap
import numpy as np
import six.moves as sm
import skimage.exposure as ski_exposure

from . import meta
from .meta import Augmenter


def GammaContrast(gamma=1, per_channel=False, name=None, deterministic=False, random_state=None):
    """Adjust contrast by scaling each pixel value to `(I_ij/255.0)**gamma`.

    Values in the range gamma=(0.5, 2.0) seem to be sensible.

    Parameters
    ----------
    gamma : number or tuple of number or list of number or StochasticParameter, optional(default=1)
        Exponent for the contrast adjustment. Higher values darken the image.

            * If a number, then that value will be used for all images.
            * If a tuple (a, b), then a value from the range [a, b] will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    per_channel :  bool or float, optional(default=False)
        Whether to use the same value for all channels (False) or to sample a new value for each
        channel (True). If this value is a float p, then for p percent of all images `per_channel`
        will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Returns
    -------
    result : _ContrastFuncWrapper
        Augmenter to perform gamma contrast adjustment.

    """
    params1d = [iap.handle_continuous_param(gamma, "gamma", value_range=(0, None), tuple_to_uniform=True, list_to_choice=True)]
    func = _PreserveDtype(ski_exposure.adjust_gamma)
    return _ContrastFuncWrapper(
        func, params1d, per_channel,
        name=name if name is not None else ia.caller_name(),
        deterministic=deterministic,
        random_state=random_state
    )

def SigmoidContrast(gain=10, cutoff=0.5, per_channel=False, name=None, deterministic=False, random_state=None):
    """Adjust contrast by scaling each pixel value to `1/(1 + exp(gain*(cutoff - I_ij/255.0)))`.

    Values in the range gain=(5, 20) and cutoff=(0.25, 0.75) seem to be sensible.

    Parameters
    ----------
    gain : number or tuple of number or list of number or StochasticParameter, optional(default=1)
        Multiplier for the sigmoid function's output.
        Higher values lead to quicker changes from dark to light pixels.

            * If a number, then that value will be used for all images.
            * If a tuple (a, b), then a value from the range [a, b] will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    cutoff : number or tuple of number or list of number or StochasticParameter, optional(default=1)
        Cutoff that shifts the sigmoid function in horizontal direction.
        Higher values mean that the switch from dark to light pixels happens later, i.e.
        the pixels will remain darker.

            * If a number, then that value will be used for all images.
            * If a tuple (a, b), then a value from the range [a, b] will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    per_channel :  bool or float, optional(default=False)
        Whether to use the same value for all channels (False) or to sample a new value for each
        channel (True). If this value is a float p, then for p percent of all images `per_channel`
        will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Returns
    -------
    result : _ContrastFuncWrapper
        Augmenter to perform sigmoid contrast adjustment.

    """
    # TODO add inv parameter?
    params1d = [
        iap.handle_continuous_param(cutoff, "cutoff", value_range=(0, 255), tuple_to_uniform=True, list_to_choice=True),
        iap.handle_continuous_param(gain, "gain", value_range=(0+1e-6, None), tuple_to_uniform=True, list_to_choice=True)
    ]
    func = _PreserveDtype(ski_exposure.adjust_sigmoid)
    return _ContrastFuncWrapper(
        func, params1d, per_channel,
        name=name if name is not None else ia.caller_name(),
        deterministic=deterministic,
        random_state=random_state
    )

def LogContrast(gain=1, per_channel=False, name=None, deterministic=False, random_state=None):
    """Adjust contrast by scaling each pixel value to `gain * log(1 + I_ij)`.

    Parameters
    ----------
    gain : number or tuple of number or list of number or StochasticParameter, optional(default=1)
        Multiplier for the logarithm result. Values around 1.0 lead to a contrast-adjusted
        image. Values above 1.0 quickly lead to partially broken images due to exceeding the
        datatype's value range.

            * If a number, then that value will be used for all images.
            * If a tuple (a, b), then a value from the range [a, b] will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    per_channel :  bool or float, optional(default=False)
        Whether to use the same value for all channels (False) or to sample a new value for each
        channel (True). If this value is a float p, then for p percent of all images `per_channel`
        will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Returns
    -------
    result : _ContrastFuncWrapper
        Augmenter to perform logarithmic contrast adjustment.

    """
    # TODO add inv parameter?
    params1d = [iap.handle_continuous_param(gain, "gain", value_range=None, tuple_to_uniform=True, list_to_choice=True)]
    func = _PreserveDtype(ski_exposure.adjust_log)
    return _ContrastFuncWrapper(
        func, params1d, per_channel,
        name=name if name is not None else ia.caller_name(),
        deterministic=deterministic,
        random_state=random_state
    )

def LinearContrast(alpha=1, per_channel=False, name=None, deterministic=False, random_state=None):
    """Adjust contrast by scaling each pixel value to `128 + alpha*(I_ij-128)`.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or StochasticParameter, optional(default=1)
        Multiplier to linearly pronounce (>1.0), dampen (0.0 to 1.0) or invert (<0.0) the
        difference between each pixel value and the center value, i.e. `128`.

            * If a number, then that value will be used for all images.
            * If a tuple (a, b), then a value from the range [a, b] will be used per image.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    per_channel :  bool or float, optional(default=False)
        Whether to use the same value for all channels (False) or to sample a new value for each
        channel (True). If this value is a float p, then for p percent of all images `per_channel`
        will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Returns
    -------
    result : _ContrastFuncWrapper
        Augmenter to preform center distance contrast adjustment.

    """
    params1d = [
        iap.handle_continuous_param(alpha, "alpha", value_range=None, tuple_to_uniform=True, list_to_choice=True),
        iap.handle_probability_param(per_channel, "per_channel")
    ]
    func = _adjust_linear
    return _ContrastFuncWrapper(
        func, params1d, per_channel,
        name=name if name is not None else ia.caller_name(),
        deterministic=deterministic,
        random_state=random_state
    )

class _ContrastFuncWrapper(Augmenter):
    def __init__(self, func, params1d, per_channel, name=None, deterministic=False, random_state=None):
        super(_ContrastFuncWrapper, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.func = func
        self.params1d = params1d
        self.per_channel = iap.handle_probability_param(per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, size=(1+nb_images,))
        per_channel = self.per_channel.draw_samples((nb_images,), random_state=ia.new_random_state(seeds[0]))

        result = images
        for i, (image, per_channel_i, seed) in enumerate(zip(images, per_channel, seeds[1:])):
            rs = ia.new_random_state(seed)
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

def _adjust_linear(image, alpha, per_channel):
    input_dtype = image.dtype
    image_aug = 128 + alpha * (image.astype(np.float32)-128)
    if input_dtype.type == np.uint8:
        image_aug = meta.clip_augmented_image_(image_aug, 0, 255)
    if input_dtype.type != image_aug.dtype.type:
        image_aug = meta.restore_augmented_image_dtype_(image_aug, input_dtype)
    return image_aug
