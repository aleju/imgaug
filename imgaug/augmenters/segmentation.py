"""
Augmenters that apply changes to images based on forms of segmentation.

Do not import directly from this file, as the categorization is not final.
Use instead
    `from imgaug import augmenters as iaa`
and then e.g. ::

    seq = iaa.Sequential([
        iaa.Superpixels(...)
    ])

List of augmenters:
    * Superpixels
"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, Binomial, DiscreteUniform, Uniform
import numpy as np
from skimage import segmentation, measure
import six.moves as sm

from .meta import Augmenter

# TODO tests
class Superpixels(Augmenter):
    """
    Completely or partially transform images to their superpixel representation.

    This implementation uses skimage's version of the SLIC algorithm.

    Parameters
    ----------
    p_replace : int or float or tuple/list of ints/floats or StochasticParameter, optional(default=0)
        Defines the probability of any superpixel area being replaced by the
        superpixel, i.e. by the average pixel color within its area.
        A probability of 0 would mean, that no superpixel area is replaced by
        its average (image is not changed at all).
        A probability of 0.5 would mean, that half of all superpixels are
        replaced by their average color.
        A probability of 1.0 would mean, that all superpixels are replaced
        by their average color (resulting in a standard superpixel image).
        This parameter can be a tuple (a, b), e.g. (0.5, 1.0). In this case,
        a random probability p with a <= p <= b will be rolled per image.

    n_segments : int or tuple/list of ints or StochasticParameter, optional(default=100)
        Target number of superpixels to generate.
        Lower numbers are faster.

    max_size : int or None, optional(default=128)
        Maximum image size at which the superpixels are generated.
        If the width or height of an image exceeds this value, it will be
        downscaled so that the longest side matches `max_size`.
        Though, the final output (superpixel) image has the same size as the
        input image.
        This is done to speed up the superpixel algorithm.
        Use None to apply no downscaling.

    interpolation : int or string, optional(default="linear")
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        `ia.imresize_single_image()`.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Superpixels(p_replace=1.0, n_segments=64)

    generates ~64 superpixels per image and replaces all of them with
    their average color (standard superpixel image).

    >>> aug = iaa.Superpixels(p_replace=0.5, n_segments=64)

    generates always ~64 superpixels per image and replaces half of them
    with their average color, while the other half are left unchanged (i.e.
    they still show the input image's content).

    >>> aug = iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128))

    generates between ~16 and ~128 superpixels per image and replaces
    25 to 100 percent of them with their average color.

    """

    def __init__(self, p_replace=0, n_segments=100, max_size=128, interpolation="linear", name=None, deterministic=False, random_state=None):
        super(Superpixels, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p_replace):
            self.p_replace = Binomial(p_replace)
        elif ia.is_iterable(p_replace):
            ia.do_assert(len(p_replace) == 2)
            ia.do_assert(p_replace[0] < p_replace[1])
            ia.do_assert(0 <= p_replace[0] <= 1.0)
            ia.do_assert(0 <= p_replace[1] <= 1.0)
            self.p_replace = p_replace = Binomial(Uniform(p_replace[0], p_replace[1]))
        elif isinstance(p_replace, StochasticParameter):
            self.p_replace = p_replace
        else:
            raise Exception("Expected p_replace to be float, int, list/tuple of floats/ints or StochasticParameter, got %s." % (type(p_replace),))

        if ia.is_single_integer(n_segments):
            self.n_segments = Deterministic(n_segments)
        elif ia.is_iterable(n_segments):
            ia.do_assert(len(n_segments) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(n_segments),))
            self.n_segments = DiscreteUniform(n_segments[0], n_segments[1])
        elif isinstance(n_segments, StochasticParameter):
            self.n_segments = n_segments
        else:
            raise Exception("Expected int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(n_segments),))

        self.max_size = max_size
        self.interpolation = interpolation

    def _augment_images(self, images, random_state, parents, hooks):
        #import time
        nb_images = len(images)
        #p_replace_samples = self.p_replace.draw_samples((nb_images,), random_state=random_state)
        n_segments_samples = self.n_segments.draw_samples((nb_images,), random_state=random_state)
        seeds = random_state.randint(0, 10**6, size=(nb_images,))
        for i in sm.xrange(nb_images):
            #replace_samples = ia.new_random_state(seeds[i]).binomial(1, p_replace_samples[i], size=(n_segments_samples[i],))
            # TODO this results in an error when n_segments is 0
            replace_samples = self.p_replace.draw_samples((n_segments_samples[i],), random_state=ia.new_random_state(seeds[i]))
            #print("n_segments", n_segments_samples[i], "replace_samples.shape", replace_samples.shape)
            #print("p", p_replace_samples[i])
            #print("replace_samples", replace_samples)

            if np.max(replace_samples) == 0:
                # not a single superpixel would be replaced by its average color,
                # i.e. the image would not be changed, so just keep it
                pass
            else:
                image = images[i]

                orig_shape = image.shape
                if self.max_size is not None:
                    size = max(image.shape[0], image.shape[1])
                    if size > self.max_size:
                        resize_factor = self.max_size / size
                        new_height, new_width = int(image.shape[0] * resize_factor), int(image.shape[1] * resize_factor)
                        image = ia.imresize_single_image(image, (new_height, new_width), interpolation=self.interpolation)

                #image_sp = np.random.randint(0, 255, size=image.shape).astype(np.uint8)
                image_sp = np.copy(image)
                #time_start = time.time()
                segments = segmentation.slic(image, n_segments=n_segments_samples[i], compactness=10)
                #print("seg", np.min(segments), np.max(segments), n_segments_samples[i])
                #print("segmented in %.4fs" % (time.time() - time_start))
                #print(np.bincount(segments.flatten()))
                #time_start = time.time()
                nb_channels = image.shape[2]
                for c in sm.xrange(nb_channels):
                    # segments+1 here because otherwise regionprops always misses
                    # the last label
                    regions = measure.regionprops(segments+1, intensity_image=image[..., c])
                    for ridx, region in enumerate(regions):
                        # with mod here, because slic can sometimes create more superpixel
                        # than requested. replace_samples then does not have enough
                        # values, so we just start over with the first one again.
                        if replace_samples[ridx % len(replace_samples)] == 1:
                            #print("changing region %d of %d, channel %d, #indices %d" % (ridx, np.max(segments), c, len(np.where(segments == ridx)[0])))
                            mean_intensity = region.mean_intensity
                            image_sp_c = image_sp[..., c]
                            image_sp_c[segments == ridx] = mean_intensity
                #print("colored in %.4fs" % (time.time() - time_start))

                if orig_shape != image.shape:
                    image_sp = ia.imresize_single_image(image_sp, orig_shape[0:2], interpolation=self.interpolation)

                images[i] = image_sp
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.n_segments, self.max_size]
