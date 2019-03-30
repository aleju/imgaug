"""
Augmenters that create wheather effects.

Do not import directly from this file, as the categorization is not final.
Use instead::

    from imgaug import augmenters as iaa

and then e.g.::

    seq = iaa.Sequential([iaa.Snowflakes()])

List of augmenters:

    * FastSnowyLandscape
    * Clouds
    * Fog
    * CloudLayer
    * Snowflakes
    * SnowflakesLayer

"""
from __future__ import print_function, division, absolute_import

import numpy as np
import cv2

from . import meta, arithmetic, blur, contrast, color as augmenters_color
import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt


class FastSnowyLandscape(meta.Augmenter):
    """
    Augmenter to convert non-snowy landscapes to snowy ones.

    This expects to get an image that roughly shows a landscape.

    This is based on the method proposed by
    https://medium.freecodecamp.org/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f?gi=bca4a13e634c

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

        - (1) This augmenter is based on a colorspace conversion to HLS. Hence, only RGB uint8
              inputs are sensible.

    Parameters
    ----------
    lightness_threshold : number or tuple of number or list of number\
                          or imgaug.parameters.StochasticParameter, optional
        All pixels with lightness in HLS colorspace below this value will have their lightness increased by
        `lightness_multiplier`.

            * If an int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the discrete range ``[a .. b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    lightness_multiplier : number or tuple of number or list of number\
                           or imgaug.parameters.StochasticParameter, optional
        Multiplier for pixel's lightness value in HLS colorspace. Affects all pixels selected via `lightness_threshold`.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    from_colorspace : str, optional
        The source colorspace of the input images. See :func:`imgaug.augmenters.color.ChangeColorspace.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.FastSnowyLandscape(lightness_threshold=140, lightness_multiplier=2.5)

    Search for all pixels in the image with a lightness value in HLS colorspace of less than 140 and increase their
    lightness by a factor of 2.5. This is the configuration proposed in the original article (see link above).

    >>> aug = iaa.FastSnowyLandscape(lightness_threshold=[128, 200], lightness_multiplier=(1.5, 3.5))

    Search for all pixels in the image with a lightness value in HLS colorspace of less than 128 or less than 200
    (one of these values is picked per image) and multiply their lightness by a factor of ``x`` with ``x`` being
    sampled from ``uniform(1.5, 3.5)`` (once per image).

    >>> aug = iaa.FastSnowyLandscape(lightness_threshold=(100, 255), lightness_multiplier=(1.0, 4.0))

    Similar to above, but the lightness threshold is sampled from ``uniform(100, 255)`` (per image) and the multiplier
    from ``uniform(1.0, 4.0)`` (per image). This seems to produce good and varied results.

    """

    def __init__(self, lightness_threshold=(100, 255), lightness_multiplier=(1.0, 4.0), from_colorspace="RGB",
                 name=None, deterministic=False, random_state=None):
        super(FastSnowyLandscape, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.lightness_threshold = iap.handle_continuous_param(lightness_threshold, "lightness_threshold",
                                                               value_range=(0, 255),
                                                               tuple_to_uniform=True,
                                                               list_to_choice=True)
        self.lightness_multiplier = iap.handle_continuous_param(lightness_multiplier, "lightness_multiplier",
                                                                value_range=(0, None), tuple_to_uniform=True,
                                                                list_to_choice=True)
        self.from_colorspace = from_colorspace

    def _draw_samples(self, augmentables, random_state):
        nb_augmentables = len(augmentables)
        rss = ia.derive_random_states(random_state, 2)
        thresh_samples = self.lightness_threshold.draw_samples((nb_augmentables,), rss[1])
        lmul_samples = self.lightness_multiplier.draw_samples((nb_augmentables,), rss[0])
        return thresh_samples, lmul_samples

    def _augment_images(self, images, random_state, parents, hooks):
        thresh_samples, lmul_samples = self._draw_samples(images, random_state)
        result = images

        for i, (image, thresh, lmul) in enumerate(zip(images, thresh_samples, lmul_samples)):
            color_transform = augmenters_color.ChangeColorspace.CV_VARS["%s2HLS" % (self.from_colorspace,)]
            color_transform_inverse = augmenters_color.ChangeColorspace.CV_VARS["HLS2%s" % (self.from_colorspace,)]

            image_hls = cv2.cvtColor(image, color_transform)
            cvt_dtype = image_hls.dtype
            image_hls = image_hls.astype(np.float64)
            lightness = image_hls[..., 1]

            lightness[lightness < thresh] *= lmul

            image_hls = iadt.restore_dtypes_(image_hls, cvt_dtype)
            image_rgb = cv2.cvtColor(image_hls, color_transform_inverse)

            result[i] = image_rgb

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.lightness_threshold, self.lightness_multiplier]


# TODO add vertical gradient alpha to have clouds only at skylevel/groundlevel
# TODO add configurable parameters
def Clouds(name=None, deterministic=False, random_state=None):
    """
    Augmenter to draw clouds in images.

    This is a wrapper around ``CloudLayer``. It executes 1 to 2 layers per image, leading to varying densities
    and frequency patterns of clouds.

    This augmenter seems to be fairly robust w.r.t. the image size. Tested with ``96x128``, ``192x256``
    and ``960x1280``.

    dtype support::

        * ``uint8``: yes; tested
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

        - (1) Parameters of this augmenter are optimized for the value range of uint8.
              While other dtypes may be accepted, they will lead to images augmented in
              ways inappropriate for the respective dtype.

    Parameters
    ----------
    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Clouds()

    Creates an augmenter that adds clouds to images.

    """
    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return meta.SomeOf((1, 2), children=[
        CloudLayer(
            intensity_mean=(196, 255), intensity_freq_exponent=(-2.5, -2.0), intensity_coarse_scale=10,
            alpha_min=0, alpha_multiplier=(0.25, 0.75), alpha_size_px_max=(2, 8), alpha_freq_exponent=(-2.5, -2.0),
            sparsity=(0.8, 1.0), density_multiplier=(0.5, 1.0)
        ),
        CloudLayer(
            intensity_mean=(196, 255), intensity_freq_exponent=(-2.0, -1.0), intensity_coarse_scale=10,
            alpha_min=0, alpha_multiplier=(0.5, 1.0), alpha_size_px_max=(64, 128), alpha_freq_exponent=(-2.0, -1.0),
            sparsity=(1.0, 1.4), density_multiplier=(0.8, 1.5)
        )
    ], random_order=False, name=name, deterministic=deterministic, random_state=random_state)


# TODO add vertical gradient alpha to have fog only at skylevel/groundlevel
# TODO add configurable parameters
def Fog(name=None, deterministic=False, random_state=None):
    """
    Augmenter to draw fog in images.

    This is a wrapper around ``CloudLayer``. It executes a single layer per image with a configuration leading
    to fairly dense clouds with low-frequency patterns.

    This augmenter seems to be fairly robust w.r.t. the image size. Tested with ``96x128``, ``192x256``
    and ``960x1280``.

    dtype support::

        * ``uint8``: yes; tested
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

        - (1) Parameters of this augmenter are optimized for the value range of uint8.
              While other dtypes may be accepted, they will lead to images augmented in
              ways inappropriate for the respective dtype.

    Parameters
    ----------
    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Fog()

    Creates an augmenter that adds fog to images.

    """
    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return CloudLayer(
        intensity_mean=(220, 255), intensity_freq_exponent=(-2.0, -1.5), intensity_coarse_scale=2,
        alpha_min=(0.7, 0.9), alpha_multiplier=0.3, alpha_size_px_max=(2, 8), alpha_freq_exponent=(-4.0, -2.0),
        sparsity=0.9, density_multiplier=(0.4, 0.9),
        name=name, deterministic=deterministic, random_state=random_state
    )


# TODO add perspective transform to each cloud layer to make them look more distant?
# TODO alpha_mean and density overlap - remove one of them
class CloudLayer(meta.Augmenter):
    """
    Augmenter to add a single layer of clouds to an image.

    dtype support::

        * ``uint8``: yes; indirectly tested (1)
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: yes; not tested
        * ``float32``: yes; not tested
        * ``float64``: yes; not tested
        * ``float128``: yes; not tested (2)
        * ``bool``: no

        - (1) indirectly tested via tests for ``Clouds`` and ``Fog``
        - (2) Note that random values are usually sampled as ``int64`` or ``float64``, which
              ``float128`` images would exceed. Note also that random values might have to upscaled,
              which is done via :func:`imgaug.imgaug.imresize_many_images` and has its own limited
              dtype support (includes however floats up to ``64bit``).

    Parameters
    ----------
    intensity_mean : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Mean intensity of the clouds (i.e. mean color). Recommended to be around ``(190, 255)``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    intensity_freq_exponent : number or tuple of number or list of number\
                              or imgaug.parameters.StochasticParameter
        Exponent of the frequency noise used to add fine intensity to the mean intensity.
        Recommended to be somewhere around ``(-2.5, -1.5)``.
        See :func:`imgaug.parameters.FrequencyNoise.__init__` for details.

    intensity_coarse_scale : number or tuple of number or list of number\
                             or imgaug.parameters.StochasticParameter
        Standard deviation of the gaussian distribution used to add more localized intensity to the mean intensity.
        Sampled in low resolution space, i.e. affects final intensity on a coarse level. Recommended to be
        around ``(0, 10)``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    alpha_min : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Minimum alpha when blending cloud noise with the image. High values will lead to clouds being "everywhere".
        Recommended to usually be at around ``0.0`` for clouds and ``>0`` for fog.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    alpha_multiplier : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Multiplier for the sampled alpha values. High values will lead to denser clouds wherever they are visible.
        Recommended to be at around ``(0.3, 1.0)``. Note that this parameter currently overlaps with
        `density_multiplier`, which is applied a bit later to the alpha mask.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    alpha_size_px_max : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Controls the image size at which the alpha mask is sampled. Lower values will lead to coarser alpha masks
        and hence larger clouds (and empty areas).
        See :func:`imgaug.parameters.FrequencyNoise.__init__` for details.

    alpha_freq_exponent : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Exponent of the frequency noise used to sample the alpha mask. Similarly to `alpha_size_max_px`, lower values
        will lead to coarser alpha patterns. Recommended to be somewhere around ``(-4.0, -1.5)``.
        See :func:`imgaug.parameters.FrequencyNoise.__init__` for details.

    sparsity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Exponent applied late to the alpha mask. Lower values will lead to coarser cloud patterns, higher values
        to finer patterns. Recommended to be somewhere around ``1.0``. Do not deviate far from that values, otherwise
        the alpha mask might get weird patterns with sudden fall-offs to zero that look very unnatural.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    density_multiplier : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Late multiplier for the alpha mask, similar to `alpha_multiplier`. Set this higher to get "denser" clouds
        wherever they are visible. Recommended to be around ``(0.5, 1.5)``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    """
    def __init__(self, intensity_mean, intensity_freq_exponent, intensity_coarse_scale,
                 alpha_min, alpha_multiplier, alpha_size_px_max, alpha_freq_exponent,
                 sparsity, density_multiplier,
                 name=None, deterministic=False, random_state=None):
        super(CloudLayer, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.intensity_mean = iap.handle_continuous_param(intensity_mean, "intensity_mean")
        self.intensity_freq_exponent = intensity_freq_exponent
        self.intensity_coarse_scale = intensity_coarse_scale
        self.alpha_min = iap.handle_continuous_param(alpha_min, "alpha_min")
        self.alpha_multiplier = iap.handle_continuous_param(alpha_multiplier, "alpha_multiplier")
        self.alpha_size_px_max = alpha_size_px_max
        self.alpha_freq_exponent = alpha_freq_exponent
        self.sparsity = iap.handle_continuous_param(sparsity, "sparsity")
        self.density_multiplier = iap.handle_continuous_param(density_multiplier, "density_multiplier")

    def _augment_images(self, images, random_state, parents, hooks):
        rss = ia.derive_random_states(random_state, len(images))
        result = images
        for i, (image, rs) in enumerate(zip(images, rss)):
            result[i] = self.draw_on_image(image, rs)
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.intensity_mean, self.alpha_min, self.alpha_multiplier, self.alpha_size_px_max,
                self.alpha_freq_exponent, self.intensity_freq_exponent, self.sparsity, self.density_min,
                self.density_multiplier,
                self.intensity_coarse_scale]

    def draw_on_image(self, image, random_state):
        iadt.gate_dtypes(image,
                         allowed=["uint8",  "float16", "float32", "float64", "float96", "float128", "float256"],
                         disallowed=["bool",
                                     "uint16", "uint32", "uint64", "uint128", "uint256",
                                     "int8", "int16", "int32", "int64", "int128", "int256"])

        alpha, intensity = self.generate_maps(image, random_state)
        alpha = alpha[..., np.newaxis]
        intensity = intensity[..., np.newaxis]
        if image.dtype.kind == "f":
            intensity = intensity.astype(image.dtype)
            return (1 - alpha) * image + alpha * intensity,
        else:
            intensity = np.clip(intensity, 0, 255)
            # TODO use blend_alpha_() here
            return np.clip(
                (1 - alpha) * image.astype(alpha.dtype) + alpha * intensity.astype(alpha.dtype),
                0,
                255
            ).astype(np.uint8)

    def generate_maps(self, image, random_state):
        intensity_mean_sample = self.intensity_mean.draw_sample(random_state)
        alpha_min_sample = self.alpha_min.draw_sample(random_state)
        alpha_multiplier_sample = self.alpha_multiplier.draw_sample(random_state)
        alpha_size_px_max = self.alpha_size_px_max
        intensity_freq_exponent = self.intensity_freq_exponent
        alpha_freq_exponent = self.alpha_freq_exponent
        sparsity_sample = self.sparsity.draw_sample(random_state)
        density_multiplier_sample = self.density_multiplier.draw_sample(random_state)

        height, width = image.shape[0:2]
        rss_alpha, rss_intensity = ia.derive_random_states(random_state, 2)

        intensity_coarse = self._generate_intensity_map_coarse(
            height, width, intensity_mean_sample,
            iap.Normal(0, scale=self.intensity_coarse_scale),
            rss_intensity
        )
        intensity_fine = self._generate_intensity_map_fine(height, width, intensity_mean_sample,
                                                           intensity_freq_exponent, rss_intensity)
        intensity = intensity_coarse + intensity_fine

        alpha = self._generate_alpha_mask(height, width, alpha_min_sample, alpha_multiplier_sample,
                                          alpha_freq_exponent, alpha_size_px_max,
                                          sparsity_sample, density_multiplier_sample, rss_alpha)

        return alpha, intensity

    @classmethod
    def _generate_intensity_map_coarse(cls, height, width, intensity_mean, intensity_local_offset, random_state):
        height_intensity, width_intensity = (8, 8)  # TODO this might be too simplistic for some image sizes
        intensity = intensity_mean\
            + intensity_local_offset.draw_samples((height_intensity, width_intensity), random_state)
        intensity = ia.imresize_single_image(intensity, (height, width), interpolation="cubic")

        return intensity

    @classmethod
    def _generate_intensity_map_fine(cls, height, width, intensity_mean, exponent, random_state):
        intensity_details_generator = iap.FrequencyNoise(
            exponent=exponent,
            size_px_max=max(height, width),
            upscale_method="cubic"
        )
        intensity_details = intensity_details_generator.draw_samples((height, width), random_state)
        return intensity_mean * ((2*intensity_details - 1.0)/5.0)

    @classmethod
    def _generate_alpha_mask(cls, height, width, alpha_min, alpha_multiplier, exponent, alpha_size_px_max, sparsity,
                             density_multiplier, random_state):
        alpha_generator = iap.FrequencyNoise(
            exponent=exponent,
            size_px_max=alpha_size_px_max,
            upscale_method="cubic"
        )
        alpha_local = alpha_generator.draw_samples((height, width), random_state)
        alpha = alpha_min + (alpha_multiplier * alpha_local)
        alpha = (alpha ** sparsity) * density_multiplier
        alpha = np.clip(alpha, 0.0, 1.0)

        return alpha


def Snowflakes(density=(0.005, 0.075), density_uniformity=(0.3, 0.9), flake_size=(0.2, 0.7),
               flake_size_uniformity=(0.4, 0.8), angle=(-30, 30), speed=(0.007, 0.03),
               name=None, deterministic=False, random_state=None):
    """
    Augmenter to add falling snowflakes to images.

    This is a wrapper around ``SnowflakesLayer``. It executes 1 to 3 layers per image.

    dtype support::

        * ``uint8``: yes; tested
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

        - (1) Parameters of this augmenter are optimized for the value range of uint8.
              While other dtypes may be accepted, they will lead to images augmented in
              ways inappropriate for the respective dtype.

    Parameters
    ----------
    density : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Density of the snowflake layer, as a probability of each pixel in low resolution space to be a snowflake.
        Valid value range is ``(0.0, 1.0)``. Recommended to be around ``(0.01, 0.075)``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    density_uniformity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Size uniformity of the snowflakes. Higher values denote more similarly sized snowflakes.
        Valid value range is ``(0.0, 1.0)``. Recommended to be around ``0.5``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    flake_size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Size of the snowflakes. This parameter controls the resolution at which snowflakes are sampled.
        Higher values mean that the resolution is closer to the input image's resolution and hence each sampled
        snowflake will be smaller (because of the smaller pixel size).

        Valid value range is ``[0.0, 1.0)``. Recommended values:

            * On ``96x128`` a value of ``(0.1, 0.4)`` worked well.
            * On ``192x256`` a value of ``(0.2, 0.7)`` worked well.
            * On ``960x1280`` a value of ``(0.7, 0.95)`` worked well.

        Allowed datatypes:

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    flake_size_uniformity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Controls the size uniformity of the snowflakes. Higher values mean that the snowflakes are more similarly
        sized. Valid value range is ``(0.0, 1.0)``. Recommended to be around ``0.5``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    angle : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Angle in degrees of motion blur applied to the snowflakes, where ``0.0`` is motion blur that points straight
        upwards. Recommended to be around ``(-30, 30)``.
        See also :func:`imgaug.augmenters.blur.MotionBlur.__init__`.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    speed : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Perceived falling speed of the snowflakes. This parameter controls the motion blur's kernel size.
        It follows roughly the form ``kernel_size = image_size * speed``. Hence,
        Values around ``1.0`` denote that the motion blur should "stretch" each snowflake over the whole image.

        Valid value range is ``(0.0, 1.0)``. Recommended values:

            * On ``96x128`` a value of ``(0.01, 0.05)`` worked well.
            * On ``192x256`` a value of ``(0.007, 0.03)`` worked well.
            * On ``960x1280`` a value of ``(0.001, 0.03)`` worked well.


        Allowed datatypes:

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))

    Adds snowflakes to small images (around ``96x128``).

    >>> aug = iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03))

    Adds snowflakes to medium-sized images (around ``192x256``).

    >>> aug = iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))

    Adds snowflakes to large images (around ``960x1280``).

    """
    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    layer = SnowflakesLayer(
        density=density, density_uniformity=density_uniformity,
        flake_size=flake_size, flake_size_uniformity=flake_size_uniformity,
        angle=angle, speed=speed,
        blur_sigma_fraction=(0.0001, 0.001)
    )

    return meta.SomeOf(
        (1, 3), children=[layer.deepcopy() for _ in range(3)],
        random_order=False, name=name, deterministic=deterministic, random_state=random_state
    )


# TODO snowflakes are all almost 100% white, add some grayish tones and maybe color to them
class SnowflakesLayer(meta.Augmenter):
    """
    Augmenter to add a single layer of falling snowflakes to images.

    dtype support::

        * ``uint8``: yes; indirectly tested (1)
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

        - (1) indirectly tested via tests for ``Snowflakes``

    Parameters
    ----------
    density : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Density of the snowflake layer, as a probability of each pixel in low resolution space to be a snowflake.
        Valid value range is ``(0.0, 1.0)``. Recommended to be around ``(0.01, 0.075)``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    density_uniformity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Size uniformity of the snowflakes. Higher values denote more similarly sized snowflakes.
        Valid value range is ``(0.0, 1.0)``. Recommended to be around ``0.5``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    flake_size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Size of the snowflakes. This parameter controls the resolution at which snowflakes are sampled.
        Higher values mean that the resolution is closer to the input image's resolution and hence each sampled
        snowflake will be smaller (because of the smaller pixel size).

        Valid value range is ``[0.0, 1.0)``. Recommended values:

            * On 96x128 a value of ``(0.1, 0.4)`` worked well.
            * On 192x256 a value of ``(0.2, 0.7)`` worked well.
            * On 960x1280 a value of ``(0.7, 0.95)`` worked well.

        Allowed datatypes:

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    flake_size_uniformity : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Controls the size uniformity of the snowflakes. Higher values mean that the snowflakes are more similarly
        sized. Valid value range is ``(0.0, 1.0)``. Recommended to be around ``0.5``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    angle : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Angle in degrees of motion blur applied to the snowflakes, where ``0.0`` is motion blur that points straight
        upwards. Recommended to be around ``(-30, 30)``.
        See also :func:`imgaug.augmenters.blur.MotionBlur.__init__`.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    speed : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Perceived falling speed of the snowflakes. This parameter controls the motion blur's kernel size.
        It follows roughly the form ``kernel_size = image_size * speed``. Hence,
        Values around ``1.0`` denote that the motion blur should "stretch" each snowflake over the whole image.

        Valid value range is ``(0.0, 1.0)``. Recommended values:

            * On 96x128 a value of ``(0.01, 0.05)`` worked well.
            * On 192x256 a value of ``(0.007, 0.03)`` worked well.
            * On 960x1280 a value of ``(0.001, 0.03)`` worked well.


        Allowed datatypes:

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    blur_sigma_fraction : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        Standard deviation (as a fraction of the image size) of gaussian blur applied to the snowflakes.
        Valid value range is ``(0.0, 1.0)``. Recommended to be around ``(0.0001, 0.001)``. May still require tinkering
        based on image size.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value from the continuous range ``[a, b]`` will be used.
            * If a list, then a random value will be sampled from that list per image.
            * If a StochasticParameter, then a value will be sampled per image from that parameter.

    blur_sigma_limits : tuple of float, optional
        Controls allows min and max values of `blur_sigma_fraction` after(!) multiplication with the image size.
        First value is the minimum, second value is the maximum. Values outside of that range will be clipped to be
        within that range. This prevents extreme values for very small or large images.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    """
    def __init__(self, density, density_uniformity, flake_size, flake_size_uniformity, angle, speed, blur_sigma_fraction,
                 blur_sigma_limits=(0.5, 3.75), name=None, deterministic=False,
                 random_state=None):
        super(SnowflakesLayer, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.density = density
        self.density_uniformity = iap.handle_continuous_param(density_uniformity, "density_uniformity",
                                                              value_range=(0.0, 1.0))
        self.flake_size = iap.handle_continuous_param(flake_size, "flake_size", value_range=(0.0+1e-4, 1.0))
        self.flake_size_uniformity = iap.handle_continuous_param(flake_size_uniformity, "flake_size_uniformity",
                                                                 value_range=(0.0, 1.0))
        self.angle = iap.handle_continuous_param(angle, "angle")
        self.speed = iap.handle_continuous_param(speed, "speed", value_range=(0.0, 1.0))
        self.blur_sigma_fraction = iap.handle_continuous_param(blur_sigma_fraction, "blur_sigma_fraction",
                                                               value_range=(0.0, 1.0))
        self.blur_sigma_limits = blur_sigma_limits  # (min, max), same for all images
        self.gate_noise_size = (8, 8)  # (height, width), same for all images

    def _augment_images(self, images, random_state, parents, hooks):
        rss = ia.derive_random_states(random_state, len(images))
        result = images
        for i, (image, rs) in enumerate(zip(images, rss)):
            result[i] = self.draw_on_image(image, rs)
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.density, self.density_uniformity, self.flake_size, self.flake_size_uniformity, self.angle,
                self.speed, self.blur_sigma_fraction, self.blur_sigma_limits, self.gate_noise_size]

    def draw_on_image(self, image, random_state):
        flake_size_sample = self.flake_size.draw_sample(random_state)
        flake_size_uniformity_sample = self.flake_size_uniformity.draw_sample(random_state)
        angle_sample = self.angle.draw_sample(random_state)
        speed_sample = self.speed.draw_sample(random_state)
        blur_sigma_fraction_sample = self.blur_sigma_fraction.draw_sample(random_state)

        height, width = image.shape[0:2]
        downscale_factor = np.clip(1.0 - flake_size_sample, 0.001, 1.0)
        height_down, width_down = int(height*downscale_factor), int(width*downscale_factor),
        noise = self._generate_noise(
            height_down,
            width_down,
            self.density,
            ia.derive_random_state(random_state)
        )

        # gate the sampled noise via noise in range [0.0, 1.0]
        # this leads to less flakes in some areas of the image and more in other areas
        gate_noise = iap.Beta(1.0, 1.0 - self.density_uniformity)
        noise = self._gate(noise, gate_noise, self.gate_noise_size, ia.derive_random_state(random_state))
        noise = ia.imresize_single_image(noise, (height, width), interpolation="cubic")

        # apply a bit of gaussian blur and then motion blur according to angle and speed
        sigma = max(height, width) * blur_sigma_fraction_sample
        sigma = np.clip(sigma, self.blur_sigma_limits[0], self.blur_sigma_limits[1])
        noise_small_blur = self._blur(noise, sigma)
        noise_small_blur = self._motion_blur(noise_small_blur, angle=angle_sample, speed=speed_sample,
                                             random_state=random_state)

        # use contrast adjustment of noise to make the flake size a bit less uniform
        # then readjust the noise values to make them more visible again
        gain = 1.0 + 2*(1 - flake_size_uniformity_sample)
        gain_adj = 1.0 + 5*(1 - flake_size_uniformity_sample)
        noise_small_blur = contrast.GammaContrast(gain).augment_image(noise_small_blur)
        noise_small_blur = noise_small_blur.astype(np.float32) * gain_adj
        noise_small_blur_rgb = np.tile(noise_small_blur[..., np.newaxis], (1, 1, 3))

        # blend:
        # sum for a bit of glowy, hardly visible flakes
        # max for the main flakes
        image_f32 = image.astype(np.float32)
        image_f32 = self._blend_by_sum(image_f32, (0.1 + 20*speed_sample) * noise_small_blur_rgb)
        image_f32 = self._blend_by_max(image_f32, (1.0 + 20*speed_sample) * noise_small_blur_rgb)
        return image_f32

    @classmethod
    def _generate_noise(cls, height, width, density, random_state):
        noise = arithmetic.Salt(p=density, random_state=random_state)
        return noise.augment_image(np.zeros((height, width), dtype=np.uint8))

    @classmethod
    def _gate(cls, noise, gate_noise, gate_size, random_state):
        # the beta distribution here has most of its weight around 1.0 and will only rarely sample values around 0.0
        # the average of the sampled values seems to be at around 0.6-0.75
        gate_noise = gate_noise.draw_samples(gate_size, random_state)
        gate_noise_up = ia.imresize_single_image(gate_noise, noise.shape[0:2], interpolation="cubic")
        gate_noise_up = np.clip(gate_noise_up, 0.0, 1.0)
        return np.clip(noise.astype(np.float32) * gate_noise_up, 0, 255).astype(np.uint8)

    @classmethod
    def _blur(cls, noise, sigma):
        return blur.blur_gaussian_(noise, sigma=sigma)

    @classmethod
    def _motion_blur(cls, noise, angle, speed, random_state):
        size = max(noise.shape[0:2])
        k = int(speed * size)
        if k <= 1:
            return noise

        # we use max(k, 3) here because MotionBlur errors for anything less than 3
        blurer = blur.MotionBlur(k=max(k, 3), angle=angle, direction=1.0, random_state=random_state)
        return blurer.augment_image(noise)

    # TODO replace this by a function from module blend.py
    @classmethod
    def _blend_by_sum(cls, image_f32, noise_small_blur_rgb):
        image_f32 = image_f32 + noise_small_blur_rgb
        return np.clip(image_f32, 0, 255).astype(np.uint8)

    # TODO replace this by a function from module blend.py
    @classmethod
    def _blend_by_max(cls, image_f32, noise_small_blur_rgb):
        image_f32 = np.maximum(image_f32, noise_small_blur_rgb)
        return np.clip(image_f32, 0, 255).astype(np.uint8)
