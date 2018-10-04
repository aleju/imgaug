"""
Augmenters that somehow change the size of the images.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([
        iaa.Scale({"height": 32, "width": 64})
        iaa.Crop((0, 20))
    ])

List of augmenters:

    * Scale
    * CropAndPad
    * Crop
    * Pad
    * PadToFixedSize
    * CropToFixedSize

"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
from .. import parameters as iap
import numpy as np
import six.moves as sm

from . import meta
from .meta import Augmenter

# TODO rename to Resize to avoid confusion with Affine's scale
class Scale(Augmenter):
    """
    Augmenter that scales/resizes images to specified heights and widths.

    Parameters
    ----------
    size : string "keep" or int or float or tuple of two ints/floats or list of ints/floats or StochasticParameter or dictionary
        The new size of the
        images.

            * If this has the string value 'keep', the original height and
              width values will be kept (image is not scaled).
            * If this is an integer, this value will always be used as the new
              height and width of the images.
            * If this is a float v, then per image the image's height H and
              width W will be changed to H*v and W*v.
            * If this is a tuple, it is expected to have two entries (a, b).
              If at least one of these are floats, a value will be sampled from
              range [a, b] and used as the float value to resize the image
              (see above). If both are integers, a value will be sampled from
              the discrete range [a .. b] and used as the integer value
              to resize the image (see above).
            * If this is a list, a random value from the list will be picked
              to resize the image. All values in the list must be integers or
              floats (no mixture is possible).
            * If this is a StochasticParameter, then this parameter will first
              be queried once per image. The resulting value will be used
              for both height and width.
            * If this is a dictionary, it may contain the keys "height" and
              "width". Each key may have the same datatypes as above and
              describes the scaling on x and y-axis. Both axis are sampled
              independently. Additionally, one of the keys may have the value
              "keep-aspect-ratio", which means that the respective side of the
              image will be resized so that the original aspect ratio is kept.
              This is useful when only resizing one image size by a pixel
              value (e.g. resize images to a height of 64 pixels and resize
              the width so that the overall aspect ratio is maintained).

    interpolation : ia.ALL or int or string or list of ints/strings or StochasticParameter, optional(default="cubic")
        Interpolation to
        use.

            * If ia.ALL, then a random interpolation from `nearest`, `linear`,
              `area` or `cubic` will be picked (per image).
            * If int, then this interpolation will always be used.
              Expected to be any of the following:
              `cv2.INTER_NEAREST`, `cv2.INTER_LINEAR`, `cv2.INTER_AREA`,
              `cv2.INTER_CUBIC`
            * If string, then this interpolation will always be used.
              Expected to be any of the following:
              "nearest", "linear", "area", "cubic"
            * If list of ints/strings, then a random one of the values will be
              picked per image as the interpolation.
              If a StochasticParameter, then this parameter will be queried per
              image and is expected to return an integer or string.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Scale(32)

    scales all images to 32x32 pixels.

    >>> aug = iaa.Scale(0.5)

    scales all images to 50 percent of their original size.

    >>> aug = iaa.Scale((16, 22))

    scales all images to a random height and width within the
    discrete range 16<=x<=22.

    >>> aug = iaa.Scale((0.5, 0.75))

    scales all image's height and width to H*v and W*v, where v is randomly
    sampled from the range 0.5<=x<=0.75.

    >>> aug = iaa.Scale([16, 32, 64])

    scales all images either to 16x16, 32x32 or 64x64 pixels.

    >>> aug = iaa.Scale({"height": 32})

    scales all images to a height of 32 pixels and keeps the original
    width.

    >>> aug = iaa.Scale({"height": 32, "width": 48})

    scales all images to a height of 32 pixels and a width of 48.

    >>> aug = iaa.Scale({"height": 32, "width": "keep-aspect-ratio"})

    scales all images to a height of 32 pixels and resizes the x-axis
    (width) so that the aspect ratio is maintained.

    >>> aug = iaa.Scale({"height": (0.5, 0.75), "width": [16, 32, 64]})

    scales all images to a height of H*v, where H is the original height
    and v is a random value sampled from the range 0.5<=x<=0.75.
    The width/x-axis of each image is resized to either 16 or 32 or
    64 pixels.

    >>> aug = iaa.Scale(32, interpolation=["linear", "cubic"])

    scales all images to 32x32 pixels. Randomly uses either "linear"
    or "cubic" interpolation.

    """
    def __init__(self, size, interpolation="cubic", name=None, deterministic=False, random_state=None):
        super(Scale, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        def handle(val, allow_dict):
            if val == "keep":
                return iap.Deterministic("keep")
            elif ia.is_single_integer(val):
                ia.do_assert(val > 0)
                return iap.Deterministic(val)
            elif ia.is_single_float(val):
                ia.do_assert(val > 0)
                return iap.Deterministic(val)
            elif allow_dict and isinstance(val, dict):
                if len(val.keys()) == 0:
                    return iap.Deterministic("keep")
                else:
                    ia.do_assert(all([key in ["height", "width"] for key in val.keys()]))
                    if "height" in val and "width" in val:
                        ia.do_assert(val["height"] != "keep-aspect-ratio" or val["width"] != "keep-aspect-ratio")

                    size_tuple = []
                    for k in ["height", "width"]:
                        if k in val:
                            if val[k] == "keep-aspect-ratio" or val[k] == "keep":
                                entry = iap.Deterministic(val[k])
                            else:
                                entry = handle(val[k], False)
                        else:
                            entry = iap.Deterministic("keep")
                        size_tuple.append(entry)
                    return tuple(size_tuple)
            elif isinstance(val, tuple):
                ia.do_assert(len(val) == 2)
                ia.do_assert(val[0] > 0 and val[1] > 0)
                if ia.is_single_float(val[0]) or ia.is_single_float(val[1]):
                    return iap.Uniform(val[0], val[1])
                else:
                    return iap.DiscreteUniform(val[0], val[1])
            elif isinstance(val, list):
                if len(val) == 0:
                    return iap.Deterministic("keep")
                else:
                    all_int = all([ia.is_single_integer(v) for v in val])
                    all_float = all([ia.is_single_float(v) for v in val])
                    ia.do_assert(all_int or all_float)
                    ia.do_assert(all([v > 0 for v in val]))
                    return iap.Choice(val)
            elif isinstance(val, iap.StochasticParameter):
                return val
            else:
                raise Exception(
                    "Expected number, tuple of two numbers, list of numbers, dictionary of "
                    "form {'height': number/tuple/list/'keep-aspect-ratio'/'keep', "
                    "'width': <analogous>}, or StochasticParameter, got %s." % (type(val),)
                )

        self.size = handle(size, True)

        if interpolation == ia.ALL:
            self.interpolation = iap.Choice(["nearest", "linear", "area", "cubic"])
        elif ia.is_single_integer(interpolation):
            self.interpolation = iap.Deterministic(interpolation)
        elif ia.is_string(interpolation):
            self.interpolation = iap.Deterministic(interpolation)
        elif ia.is_iterable(interpolation):
            self.interpolation = iap.Choice(interpolation)
        elif isinstance(interpolation, iap.StochasticParameter):
            self.interpolation = interpolation
        else:
            raise Exception("Expected int or string or iterable or StochasticParameter, got %s." % (type(interpolation),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        samples_h, samples_w, samples_ip = self._draw_samples(nb_images, random_state, do_sample_ip=True)
        for i in sm.xrange(nb_images):
            image = images[i]
            ia.do_assert(image.dtype == np.uint8, "Scale() can currently only process images of dtype uint8 (got %s)" % (image.dtype,))
            sample_h, sample_w, sample_ip = samples_h[i], samples_w[i], samples_ip[i]
            h, w = self._compute_height_width(image.shape, sample_h, sample_w)
            image_rs = ia.imresize_single_image(image, (h, w), interpolation=sample_ip)
            result.append(image_rs)

        if not isinstance(images, list):
            all_same_size = (len(set([image.shape for image in result])) == 1)
            if all_same_size:
                result = np.array(result, dtype=np.uint8)

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        result = []
        nb_heatmaps = len(heatmaps)
        samples_h, samples_w, samples_ip = self._draw_samples(nb_heatmaps, random_state, do_sample_ip=True)
        for i in sm.xrange(nb_heatmaps):
            heatmaps_i = heatmaps[i]
            sample_h, sample_w, sample_ip = samples_h[i], samples_w[i], samples_ip[i]
            h, w = self._compute_height_width(heatmaps_i.arr_0to1.shape, sample_h, sample_w)
            heatmaps_i_scaled = heatmaps_i.scale((h, w), interpolation=sample_ip)
            heatmaps_i_scaled.shape = (sample_h, sample_w) + heatmaps_i.shape[2:]
            result.append(heatmaps_i_scaled)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        samples_h, samples_w, _samples_ip = self._draw_samples(nb_images, random_state, do_sample_ip=False)
        for i in sm.xrange(nb_images):
            keypoints_on_image = keypoints_on_images[i]
            sample_h, sample_w = samples_h[i], samples_w[i]
            h, w = self._compute_height_width(keypoints_on_image.shape, sample_h, sample_w)
            new_shape = (h, w) + keypoints_on_image.shape[2:]
            keypoints_on_image_rs = keypoints_on_image.on(new_shape)

            result.append(keypoints_on_image_rs)

        return result

    def _draw_samples(self, nb_images, random_state, do_sample_ip=True):
        seed = random_state.randint(0, 10**6, 1)[0]
        if isinstance(self.size, tuple):
            samples_h = self.size[0].draw_samples(nb_images, random_state=ia.new_random_state(seed + 0))
            samples_w = self.size[1].draw_samples(nb_images, random_state=ia.new_random_state(seed + 1))
        else:
            samples_h = self.size.draw_samples(nb_images, random_state=ia.new_random_state(seed + 0))
            samples_w = samples_h
        if do_sample_ip:
            samples_ip = self.interpolation.draw_samples(nb_images, random_state=ia.new_random_state(seed + 2))
        else:
            samples_ip = None
        return samples_h, samples_w, samples_ip

    def _compute_height_width(self, image_shape, sample_h, sample_w):
        imh, imw = image_shape[0:2]
        h, w = sample_h, sample_w

        if ia.is_single_float(h):
            ia.do_assert(0 < h)
            h = int(np.round(imh * h))
            h = h if h > 0 else 1
        elif h == "keep":
            h = imh
        if ia.is_single_float(w):
            ia.do_assert(0 < w)
            w = int(np.round(imw * w))
            w = w if w > 0 else 1
        elif w == "keep":
            w = imw

        # at least the checks for keep-aspect-ratio must come after
        # the float checks, as they are dependent on the results
        # this is also why these are not written as elifs
        if h == "keep-aspect-ratio":
            h_per_w_orig = imh / imw
            h = int(np.round(w * h_per_w_orig))
        if w == "keep-aspect-ratio":
            w_per_h_orig = imw / imh
            w = int(np.round(h * w_per_h_orig))

        return h, w

    def get_parameters(self):
        return [self.size, self.interpolation]

class CropAndPad(Augmenter):
    """
    Augmenter that crops/pads images by defined amounts in pixels or
    percent (relative to input image size).
    Cropping removes pixels at the sides (i.e. extracts a subimage from
    a given full image). Padding adds pixels to the sides (e.g. black pixels).

    Parameters
    ----------
    px : None or int or StochasticParameter or tuple, optional(default=None)
        The number of pixels to crop (negative values) or
        pad (positive values) on each side of the image.
        Either this or the parameter `percent` may be set, not both at the
        same time.

            * If None, then pixel-based cropping will not be used.
            * If int, then that exact number of pixels will always be cropped.
            * If StochasticParameter, then that parameter will be used for each
              image. Four samples will be drawn per image (top, right, bottom,
              left).
              If however `sample_independently` is set to False, only one value
              will be sampled per image and used for all sides.
            * If a tuple of two ints with values a and b, then each side will
              be cropped by a random amount in the range a <= x <= b.
              x is sampled per image side.
              If however `sample_independently` is set to False, only one value
              will be sampled per image and used for all sides.
            * If a tuple of four entries, then the entries represent top, right,
              bottom, left. Each entry may be a single integer (always crop by
              exactly that value), a tuple of two ints a and b (crop by an
              amount a <= x <= b), a list of ints (crop by a random value that
              is contained in the list) or a StochasticParameter (sample the
              amount to crop from that parameter).

    percent : None or int or float or StochasticParameter or tuple, optional(default=None)
        The number of pixels to crop (negative values) or
        pad (positive values) on each side of the image given *in percent*
        of the image height/width. E.g. if this is set to 0.1, the
        augmenter will always crop away 10 percent of the image's height at
        the top, 10 percent of the width on the right, 10 percent of the
        height at the bottom and 10 percent of the width on the left.
        Either this or the parameter `px` may be set, not both at the same
        time.

            * If None, then percent-based cropping will not be used.
            * If int, then expected to be 0 (no padding/cropping).
            * If float, then that percentage will always be cropped away.
            * If StochasticParameter, then that parameter will be used for each
              image. Four samples will be drawn per image (top, right, bottom,
              left).
              If however `sample_independently` is set to False, only one value
              will be sampled per image and used for all sides.
            * If a tuple of two floats with values a and b, then each side will
              be cropped by a random percentage in the range a <= x <= b.
              x is sampled per image side.
              If however `sample_independently` is set to False, only one value
              will be sampled per image and used for all sides.
            * If a tuple of four entries, then the entries represent top, right,
              bottom, left. Each entry may be a single float (always crop by
              exactly that percent value), a tuple of two floats a and b (crop
              by a percentage a <= x <= b), a list of floats (crop by a random
              value that is contained in the list) or a StochasticParameter
              (sample the percentage to crop from that parameter).

    pad_mode : ia.ALL or string or list of strings or StochasticParameter, optional(default="constant")
        Padding mode to use for numpy's pad function. The available modes
        are `constant`, `edge`, `linear_ramp`, `maximum`, `median`,
        `minimum`, `reflect`, `symmetric`, `wrap`. Each one of these is
        explained in the numpy documentation. The modes "constant" and
        `linear_ramp` use extra values, which are provided by `pad_cval`
        when necessary.

            * If ia.ALL, then a random mode from all available
              modes will be sampled per image.
            * If a string, it will be used as the pad mode for all
              images.
            * If a list of strings, a random one of these will be
              sampled per image and used as the mode.
            * If StochasticParameter, a random mode will be sampled from this
              parameter per image.

    pad_cval : number or tuple of number or list of number or StochasticParameter, optional(default=0)
        The constant value to use (for numpy's pad function) if the pad
        mode is "constant" or the end value to use if the mode
        is `linear_ramp`.

            * If number, then that value will be used.
            * If a tuple of two numbers (a, b), a random value will be sampled
              from the discrete range [a..b].
            * If a list of numbers, then a random value will be sampled
              from the list per image and used as the value.
            * If StochasticParameter, a random value will be sampled from that
              parameter per image.

    keep_size : bool, optional(default=True)
        After cropping, the result image has a different height/width than
        the input image. If this parameter is set to True, then the cropped
        image will be resized to the input image's size, i.e. the image size
        is then not changed by the augmenter.

    sample_independently : bool, optional(default=True)
        If false AND the values for px/percent result in exactly one
        probability distribution for the amount to crop/pad, only one
        single value will be sampled from that probability distribution
        and used for all sides. I.e. the crop/pad amount then is the same
        for all sides.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.CropAndPad(px=(-10, 0))

    crops each side by a random value from the range -10px to 0px (the value
    is sampled per side).

    >>> aug = iaa.CropAndPad(px=(0, 10))

    pads each side by a random value from the range 0px to 10px (the values
    are sampled per side). The padding happens by zero-padding (i.e. adds
    black pixels).

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode="edge")

    pads each side by a random value from the range 0px to 10px (the values
    are sampled per side). The padding uses the 'edge' mode from numpy's
    pad function.

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode=["constant", "edge"])

    pads each side by a random value from the range 0px to 10px (the values
    are sampled per side). The padding uses randomly either the 'constant'
    or 'edge' mode from numpy's pad function.

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode=ia.ALL, pad_cval=(0, 255))

    pads each side by a random value from the range 0px to 10px (the values
    are sampled per side). It uses a random mode for numpy's pad function.
    If the mode is `constant` or `linear_ramp`, it samples a random value
    v from the range [0, 255] and uses that as the constant
    value (`mode=constant`) or end value (`mode=linear_ramp`).

    >>> aug = iaa.CropAndPad(px=(0, 10), sample_independently=False)

    samples one value v from the discrete range [0..10] and pads all sides
    by v pixels.

    >>> aug = iaa.CropAndPad(px=(0, 10), keep_size=False)

    pads each side by a random value from the range 0px to 10px (the value
    is sampled per side). After padding, the images are NOT resized to
    their original size (i.e. the images may end up having different
    heights/widths).

    >>> aug = iaa.CropAndPad(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    pads the top and bottom by a random value from the range 0px to 10px
    and the left and right by a random value in the range 0px to 5px.

    >>> aug = iaa.CropAndPad(percent=(0, 0.1))

    pads each side by a random value from the range 0 percent to
    10 percent. (Percent with respect to the side's size, e.g. for the
    top side it uses the image's height.)

    >>> aug = iaa.CropAndPad(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    pads each side by either 5 percent or 10 percent.

    >>> aug = iaa.CropAndPad(px=(-10, 10))

    samples per side and image a value v from the discrete range [-10..10]
    and either crops (negative value) or pads (positive value) the side
    by v pixels.

    """

    def __init__(self, px=None, percent=None, pad_mode="constant", pad_cval=0, keep_size=True, sample_independently=True, name=None, deterministic=False, random_state=None):
        super(CropAndPad, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.all_sides = None
        self.top = None
        self.right = None
        self.bottom = None
        self.left = None
        if px is None and percent is None:
            self.mode = "noop"
        elif px is not None and percent is not None:
            raise Exception("Can only pad by pixels or percent, not both.")
        elif px is not None:
            self.mode = "px"
            if ia.is_single_integer(px):
                self.all_sides = iap.Deterministic(px)
            elif isinstance(px, tuple):
                ia.do_assert(len(px) in [2, 4])
                def handle_param(p):
                    if ia.is_single_integer(p):
                        return iap.Deterministic(p)
                    elif isinstance(p, tuple):
                        ia.do_assert(len(p) == 2)
                        ia.do_assert(ia.is_single_integer(p[0]))
                        ia.do_assert(ia.is_single_integer(p[1]))
                        return iap.DiscreteUniform(p[0], p[1])
                    elif isinstance(p, list):
                        ia.do_assert(len(p) > 0)
                        ia.do_assert(all([ia.is_single_integer(val) for val in p]))
                        return iap.Choice(p)
                    elif isinstance(p, iap.StochasticParameter):
                        return p
                    else:
                        raise Exception("Expected int, tuple of two ints, list of ints or StochasticParameter, got type %s." % (type(p),))

                if len(px) == 2:
                    #self.top = self.right = self.bottom = self.left = handle_param(px)
                    self.all_sides = handle_param(px)
                else: # len == 4
                    self.top = handle_param(px[0])
                    self.right = handle_param(px[1])
                    self.bottom = handle_param(px[2])
                    self.left = handle_param(px[3])
            elif isinstance(px, iap.StochasticParameter):
                self.top = self.right = self.bottom = self.left = px
            else:
                raise Exception("Expected int, tuple of 4 ints/tuples/lists/StochasticParameters or StochasticParameter, got type %s." % (type(px),))
        else: # = elif percent is not None:
            self.mode = "percent"
            if ia.is_single_number(percent):
                ia.do_assert(-1.0 < percent)
                #self.top = self.right = self.bottom = self.left = Deterministic(percent)
                self.all_sides = iap.Deterministic(percent)
            elif isinstance(percent, tuple):
                ia.do_assert(len(percent) in [2, 4])
                def handle_param(p):
                    if ia.is_single_number(p):
                        return iap.Deterministic(p)
                    elif isinstance(p, tuple):
                        ia.do_assert(len(p) == 2)
                        ia.do_assert(ia.is_single_number(p[0]))
                        ia.do_assert(ia.is_single_number(p[1]))
                        ia.do_assert(-1.0 < p[0])
                        ia.do_assert(-1.0 < p[1])
                        return iap.Uniform(p[0], p[1])
                    elif isinstance(p, list):
                        ia.do_assert(len(p) > 0)
                        ia.do_assert(all([ia.is_single_number(val) for val in p]))
                        ia.do_assert(all([-1.0 < val for val in p]))
                        return iap.Choice(p)
                    elif isinstance(p, iap.StochasticParameter):
                        return p
                    else:
                        raise Exception("Expected int, tuple of two ints, list of ints or StochasticParameter, got type %s." % (type(p),))

                if len(percent) == 2:
                    #self.top = self.right = self.bottom = self.left = handle_param(percent)
                    self.all_sides = handle_param(percent)
                else: # len == 4
                    self.top = handle_param(percent[0])
                    self.right = handle_param(percent[1])
                    self.bottom = handle_param(percent[2])
                    self.left = handle_param(percent[3])
            elif isinstance(percent, iap.StochasticParameter):
                self.top = self.right = self.bottom = self.left = percent
            else:
                raise Exception("Expected number, tuple of 4 numbers/tuples/lists/StochasticParameters or StochasticParameter, got type %s." % (type(percent),))

        pad_modes_available = set(["constant", "edge", "linear_ramp", "maximum", "median", "minimum", "reflect", "symmetric", "wrap"])
        if pad_mode == ia.ALL:
            self.pad_mode = iap.Choice(list(pad_modes_available))
        elif ia.is_string(pad_mode):
            ia.do_assert(pad_mode in pad_modes_available)
            self.pad_mode = iap.Deterministic(pad_mode)
        elif isinstance(pad_mode, list):
            ia.do_assert(all([v in pad_modes_available for v in pad_mode]))
            self.pad_mode = iap.Choice(pad_mode)
        elif isinstance(pad_mode, iap.StochasticParameter):
            self.pad_mode = pad_mode
        else:
            raise Exception("Expected pad_mode to be ia.ALL or string or list of strings or StochasticParameter, got %s." % (type(pad_mode),))

        self.pad_cval = iap.handle_discrete_param(pad_cval, "pad_cval", value_range=(0, 255), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)

        self.keep_size = keep_size
        self.sample_independently = sample_independently

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images)

        result = []
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            seed = seeds[i]
            height, width = images[i].shape[0:2]
            crop_top, crop_right, crop_bottom, crop_left, pad_top, pad_right, pad_bottom, pad_left, pad_mode, pad_cval = self._draw_samples_image(seed, height, width)

            image_cr = images[i][crop_top:height-crop_bottom, crop_left:width-crop_right, :]

            if any([pad_top > 0, pad_right > 0, pad_bottom > 0, pad_left > 0]):
                if image_cr.ndim == 2:
                    pad_vals = ((pad_top, pad_bottom), (pad_left, pad_right))
                else:
                    pad_vals = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

                if pad_mode == "constant":
                    image_cr_pa = np.pad(image_cr, pad_vals, mode=pad_mode, constant_values=pad_cval)
                elif pad_mode == "linear_ramp":
                    image_cr_pa = np.pad(image_cr, pad_vals, mode=pad_mode, end_values=pad_cval)
                else:
                    image_cr_pa = np.pad(image_cr, pad_vals, mode=pad_mode)
            else:
                image_cr_pa = image_cr

            if self.keep_size:
                image_cr_pa = ia.imresize_single_image(image_cr_pa, (height, width))

            result.append(image_cr_pa)

        if ia.is_np_array(images):
            if self.keep_size:
                # this converts the list to an array of original input dtype
                result = np.array(result) # without this, restore_augmented_images_dtypes_() expects input_dtypes to be a list
                meta.restore_augmented_images_dtypes_(result, input_dtypes)

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # TODO add test
        result = []
        nb_heatmaps = len(heatmaps)
        seeds = random_state.randint(0, 10**6, (nb_heatmaps,))
        for i in sm.xrange(nb_heatmaps):
            seed = seeds[i]
            height_image, width_image = heatmaps[i].shape[0:2]
            height_heatmaps, width_heatmaps = heatmaps[i].arr_0to1.shape[0:2]

            vals = self._draw_samples_image(seed, height_image, width_image)
            crop_image_top, crop_image_right, crop_image_bottom, crop_image_left, \
                pad_image_top, pad_image_right, pad_image_bottom, pad_image_left, \
                _pad_mode, _pad_cval = vals

            if (height_image, width_image) != (height_heatmaps, width_heatmaps):
                crop_top = int(round(height_heatmaps * (crop_image_top/height_image)))
                crop_right = int(round(width_heatmaps * (crop_image_right/width_image)))
                crop_bottom = int(round(height_heatmaps * (crop_image_bottom/height_image)))
                crop_left = int(round(width_heatmaps * (crop_image_left/width_image)))

                crop_top, crop_right, crop_bottom, crop_left = self._prevent_zero_size(height_heatmaps, width_heatmaps, crop_top, crop_right, crop_bottom, crop_left)

                pad_top = int(round(height_heatmaps * (pad_image_top/height_image)))
                pad_right = int(round(width_heatmaps * (pad_image_right/width_image)))
                pad_bottom = int(round(height_heatmaps * (pad_image_bottom/height_image)))
                pad_left = int(round(width_heatmaps * (pad_image_left/width_image)))
            else:
                crop_top = crop_image_top
                crop_right = crop_image_right
                crop_bottom = crop_image_bottom
                crop_left = crop_image_left

                pad_top = pad_image_top
                pad_right = pad_image_right
                pad_bottom = pad_image_bottom
                pad_left = pad_image_left

            arr_cr = heatmaps[i].arr_0to1[crop_top:height_heatmaps-crop_bottom, crop_left:width_heatmaps-crop_right, :]

            if any([pad_top > 0, pad_right > 0, pad_bottom > 0, pad_left > 0]):
                if arr_cr.ndim == 2:
                    pad_vals = ((pad_top, pad_bottom), (pad_left, pad_right))
                else:
                    pad_vals = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

                arr_cr_pa = np.pad(arr_cr, pad_vals, mode="constant", constant_values=0)
            else:
                arr_cr_pa = arr_cr

            heatmaps[i].arr_0to1 = arr_cr_pa

            if self.keep_size:
                heatmaps[i] = heatmaps[i].scale((height_heatmaps, width_heatmaps))
            else:
                heatmaps[i].shape = (
                    heatmaps[i].shape[0] - crop_top - crop_bottom + pad_top + pad_bottom,
                    heatmaps[i].shape[1] - crop_left - crop_right + pad_left + pad_right
                ) + heatmaps[i].shape[2:]

            result.append(heatmaps[i])

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            seed = seeds[i]
            height, width = keypoints_on_image.shape[0:2]
            #top, right, bottom, left = self._draw_samples_image(seed, height, width)
            crop_top, crop_right, crop_bottom, crop_left, pad_top, pad_right, pad_bottom, pad_left, _pad_mode, _pad_cval = self._draw_samples_image(seed, height, width)
            shifted = keypoints_on_image.shift(x=-crop_left+pad_left, y=-crop_top+pad_top)
            shifted.shape = (
                height - crop_top - crop_bottom + pad_top + pad_bottom,
                width - crop_left - crop_right + pad_left + pad_right
            ) + shifted.shape[2:]
            if self.keep_size:
                result.append(shifted.on(keypoints_on_image.shape))
            else:
                result.append(shifted)

        return result

    def _draw_samples_image(self, seed, height, width):
        random_state = ia.new_random_state(seed)

        if self.mode == "noop":
            top = right = bottom = left = 0
        else:
            if self.all_sides is not None:
                if self.sample_independently:
                    samples = self.all_sides.draw_samples((4,), random_state=random_state)
                    top, right, bottom, left = samples
                else:
                    sample = self.all_sides.draw_sample(random_state=random_state)
                    top = right = bottom = left = sample
            else:
                top = self.top.draw_sample(random_state=random_state)
                right = self.right.draw_sample(random_state=random_state)
                bottom = self.bottom.draw_sample(random_state=random_state)
                left = self.left.draw_sample(random_state=random_state)

            if self.mode == "px":
                # no change necessary for pixel values
                pass
            elif self.mode == "percent":
                # percentage values have to be transformed to pixel values
                top = int(round(height * top))
                right = int(round(width * right))
                bottom = int(round(height * bottom))
                left = int(round(width * left))
            else:
                raise Exception("Invalid mode")

        crop_top = (-1) * top if top < 0 else 0
        crop_right = (-1) * right if right < 0 else 0
        crop_bottom = (-1) * bottom if bottom < 0 else 0
        crop_left = (-1) * left if left < 0 else 0

        pad_top = top if top > 0 else 0
        pad_right = right if right > 0 else 0
        pad_bottom = bottom if bottom > 0 else 0
        pad_left = left if left > 0 else 0

        pad_mode = self.pad_mode.draw_sample(random_state=random_state)
        pad_cval = self.pad_cval.draw_sample(random_state=random_state)
        pad_cval = np.clip(np.round(pad_cval), 0, 255).astype(np.uint8)

        crop_top, crop_right, crop_bottom, crop_left = self._prevent_zero_size(height, width, crop_top, crop_right, crop_bottom, crop_left)

        ia.do_assert(crop_top >= 0 and crop_right >= 0 and crop_bottom >= 0 and crop_left >= 0)
        ia.do_assert(crop_top + crop_bottom < height)
        ia.do_assert(crop_right + crop_left < width)

        return crop_top, crop_right, crop_bottom, crop_left, pad_top, pad_right, pad_bottom, pad_left, pad_mode, pad_cval

    def _prevent_zero_size(self, height, width, crop_top, crop_right, crop_bottom, crop_left):
        remaining_height = height - (crop_top + crop_bottom)
        remaining_width = width - (crop_left + crop_right)
        if remaining_height < 1:
            regain = abs(remaining_height) + 1
            regain_top = regain // 2
            regain_bottom = regain // 2
            if regain_top + regain_bottom < regain:
                regain_top += 1

            if regain_top > crop_top:
                diff = regain_top - crop_top
                regain_top = crop_top
                regain_bottom += diff
            elif regain_bottom > crop_bottom:
                diff = regain_bottom - crop_bottom
                regain_bottom = crop_bottom
                regain_top += diff

            ia.do_assert(regain_top <= crop_top)
            ia.do_assert(regain_bottom <= crop_bottom)

            crop_top = crop_top - regain_top
            crop_bottom = crop_bottom - regain_bottom

        if remaining_width < 1:
            regain = abs(remaining_width) + 1
            regain_right = regain // 2
            regain_left = regain // 2
            if regain_right + regain_left < regain:
                regain_right += 1

            if regain_right > crop_right:
                diff = regain_right - crop_right
                regain_right = crop_right
                regain_left += diff
            elif regain_left > crop_left:
                diff = regain_left - crop_left
                regain_left = crop_left
                regain_right += diff

            ia.do_assert(regain_right <= crop_right)
            ia.do_assert(regain_left <= crop_left)

            crop_right = crop_right - regain_right
            crop_left = crop_left - regain_left

        return crop_top, crop_right, crop_bottom, crop_left

    def get_parameters(self):
        return [self.all_sides, self.top, self.right, self.bottom, self.left, self.pad_mode, self.pad_cval]


def Pad(px=None, percent=None, pad_mode="constant", pad_cval=0, keep_size=True, sample_independently=True, name=None, deterministic=False, random_state=None):
    """
    Augmenter that pads images, i.e. adds columns/rows to them.

    Parameters
    ----------
    px : None or int or StochasticParameter or tuple, optional(default=None)
        The number of pixels to crop away (cut off) on each side of the image.
        Either this or the parameter `percent` may be set, not both at the same
        time.

            * If None, then pixel-based cropping will not be used.
            * If int, then that exact number of pixels will always be cropped.
            * If StochasticParameter, then that parameter will be used for each
              image. Four samples will be drawn per image (top, right, bottom,
              left).
            * If a tuple of two ints with values a and b, then each side will
              be cropped by a random amount in the range a <= x <= b.
              x is sampled per image side.
            * If a tuple of four entries, then the entries represent top, right,
              bottom, left. Each entry may be a single integer (always crop by
              exactly that value), a tuple of two ints a and b (crop by an
              amount a <= x <= b), a list of ints (crop by a random value that
              is contained in the list) or a StochasticParameter (sample the
              amount to crop from that parameter).

    percent : None or int or float or StochasticParameter or tuple, optional(default=None)
        The number of pixels to crop away (cut off) on each side of the image
        given *in percent* of the image height/width.
        E.g. if this is set to 0.1, the augmenter will always crop away
        10 percent of the image's height at the top, 10 percent of the width
        on the right, 10 percent of the height at the bottom and 10 percent
        of the width on the left.
        Either this or the parameter `px` may be set, not both at the same
        time.

            * If None, then percent-based cropping will not be used.
            * If int, then expected to be 0 (no cropping).
            * If float, then that percentage will always be cropped away.
            * If StochasticParameter, then that parameter will be used for each
              image. Four samples will be drawn per image (top, right, bottom,
              left).
            * If a tuple of two floats with values a and b, then each side will
              be cropped by a random percentage in the range a <= x <= b.
              x is sampled per image side.
            * If a tuple of four entries, then the entries represent top, right,
              bottom, left. Each entry may be a single float (always crop by
              exactly that percent value), a tuple of two floats a and b (crop
              by a percentage a <= x <= b), a list of floats (crop by a random
              value that is contained in the list) or a StochasticParameter
              (sample the percentage to crop from that parameter).

    pad_mode : ia.ALL or string or list of strings or StochasticParameter, optional(default="constant")
        Padding mode to use for numpy's pad function. The available modes
        are `constant`, `edge`, `linear_ramp`, `maximum`, `median`,
        `minimum`, `reflect`, `symmetric`, `wrap`. Each one of these is
        explained in the numpy documentation. The modes `constant` and
        `linear_ramp` use extra values, which are provided by `pad_cval`
        when necessary.

            * If ia.ALL, then a random mode from all available
              modes will be sampled per image.
            * If a string, it will be used as the pad mode for all
              images.
            * If a list of strings, a random one of these will be
              sampled per image and used as the mode.
            * If StochasticParameter, a random mode will be sampled from this
              parameter per image.

    pad_cval : float or int or tuple of two ints/floats or list of ints/floats or StochasticParameter, optional(default=0)
        The constant value to use (for numpy's pad function) if the pad
        mode is "constant" or the end value to use if the mode
        is `linear_ramp`.

            * If float/int, then that value will be used.
            * If a tuple of two numbers and at least one of them is a float,
              then a random number will be sampled from the continuous range
              a<=x<=b and used as the value. If both numbers are integers,
              the range is discrete.
            * If a list of ints/floats, then a random value will be chosen from
              the elements of the list and used as the value.
            * If StochasticParameter, a random value will be sampled from that
              parameter per image.

    keep_size : bool, optional(default=True)
        After cropping, the result image has a different height/width than
        the input image. If this parameter is set to True, then the cropped
        image will be resized to the input image's size, i.e. the image size
        is then not changed by the augmenter.

    sample_independently : bool, optional(default=True)
        If false AND the values for px/percent result in exactly one
        probability distribution for the amount to crop/pad, only one
        single value will be sampled from that probability distribution
        and used for all sides. I.e. the crop/pad amount then is the same
        for all sides.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Pad(px=(0, 10))

    pads each side by a random value from the range 0px to 10px (the value
    is sampled per side). The added rows/columns are filled with black pixels.

    >>> aug = iaa.Pad(px=(0, 10), sample_independently=False)

    samples one value v from the discrete range [0..10] and pads all sides
    by v pixels.

    >>> aug = iaa.Pad(px=(0, 10), keep_size=False)

    pads each side by a random value from the range 0px to 10px (the value
    is sampled per side). After padding, the images are NOT resized to their
    original size (i.e. the images may end up having different heights/widths).

    >>> aug = iaa.Pad(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    pads the top and bottom by a random value from the range 0px to 10px
    and the left and right by a random value in the range 0px to 5px.

    >>> aug = iaa.Pad(percent=(0, 0.1))

    pads each side by a random value from the range 0 percent to
    10 percent. (Percent with respect to the side's size, e.g. for the
    top side it uses the image's height.)

    >>> aug = iaa.Pad(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    pads each side by either 5 percent or 10 percent.

    >>> aug = iaa.Pad(px=(0, 10), pad_mode="edge")

    pads each side by a random value from the range 0px to 10px (the values
    are sampled per side). The padding uses the 'edge' mode from numpy's
    pad function.

    >>> aug = iaa.Pad(px=(0, 10), pad_mode=["constant", "edge"])

    pads each side by a random value from the range 0px to 10px (the values
    are sampled per side). The padding uses randomly either the 'constant'
    or 'edge' mode from numpy's pad function.

    >>> aug = iaa.Pad(px=(0, 10), pad_mode=ia.ALL, pad_cval=(0, 255))

    pads each side by a random value from the range 0px to 10px (the values
    are sampled per side). It uses a random mode for numpy's pad function.
    If the mode is `constant` or `linear_ramp`, it samples a random value
    v from the range [0, 255] and uses that as the constant
    value (`mode=constant`) or end value (`mode=linear_ramp`).

    """

    def recursive_validate(v):
        if v is None:
            return v
        elif ia.is_single_number(v):
            ia.do_assert(v >= 0)
            return v
        elif isinstance(v, iap.StochasticParameter):
            return v
        elif isinstance(v, tuple):
            return tuple([recursive_validate(v_) for v_ in v])
        elif isinstance(v, list):
            return [recursive_validate(v_) for v_ in v]
        else:
            raise Exception("Expected None or int or float or StochasticParameter or list or tuple, got %s." % (type(v),))

    px = recursive_validate(px)
    percent = recursive_validate(percent)

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    aug = CropAndPad(
        px=px, percent=percent,
        pad_mode=pad_mode, pad_cval=pad_cval,
        keep_size=keep_size, sample_independently=sample_independently,
        name=name, deterministic=deterministic, random_state=random_state
    )
    return aug


def Crop(px=None, percent=None, keep_size=True, sample_independently=True, name=None, deterministic=False, random_state=None):
    """
    Augmenter that crops/cuts away pixels at the sides of the image.

    That allows to cut out subimages from given (full) input images.
    The number of pixels to cut off may be defined in absolute values or
    percent of the image sizes.

    Parameters
    ----------
    px : None or int or StochasticParameter or tuple, optional(default=None)
        The number of pixels to crop away (cut off) on each side of the image.
        Either this or the parameter `percent` may be set, not both at the same
        time.

            * If None, then pixel-based cropping will not be used.
            * If int, then that exact number of pixels will always be cropped.
            * If StochasticParameter, then that parameter will be used for each
              image. Four samples will be drawn per image (top, right, bottom,
              left).
            * If a tuple of two ints with values a and b, then each side will
              be cropped by a random amount in the range a <= x <= b.
              x is sampled per image side.
            * If a tuple of four entries, then the entries represent top, right,
              bottom, left. Each entry may be a single integer (always crop by
              exactly that value), a tuple of two ints a and b (crop by an
              amount a <= x <= b), a list of ints (crop by a random value that
              is contained in the list) or a StochasticParameter (sample the
              amount to crop from that parameter).

    percent : None or int or float or StochasticParameter or tuple, optional(default=None)
        The number of pixels to crop away (cut off) on each side of the image
        given *in percent* of the image height/width.
        E.g. if this is set to 0.1, the augmenter will always crop away
        10 percent of the image's height at the top, 10 percent of the width
        on the right, 10 percent of the height at the bottom and 10 percent
        of the width on the left.
        Either this or the parameter `px` may be set, not both at the same
        time.

            * If None, then percent-based cropping will not be used.
            * If int, then expected to be 0 (no cropping).
            * If float, then that percentage will always be cropped away.
            * If StochasticParameter, then that parameter will be used for each
              image. Four samples will be drawn per image (top, right, bottom,
              left).
            * If a tuple of two floats with values a and b, then each side will
              be cropped by a random percentage in the range a <= x <= b.
              x is sampled per image side.
            * If a tuple of four entries, then the entries represent top, right,
              bottom, left. Each entry may be a single float (always crop by
              exactly that percent value), a tuple of two floats a and b (crop
              by a percentage a <= x <= b), a list of floats (crop by a random
              value that is contained in the list) or a StochasticParameter
              (sample the percentage to crop from that parameter).

    keep_size : bool, optional(default=True)
        After cropping, the result image has a different height/width than
        the input image. If this parameter is set to True, then the cropped
        image will be resized to the input image's size, i.e. the image size
        is then not changed by the augmenter.

    sample_independently : bool, optional(default=True)
        If false AND the values for px/percent result in exactly one
        probability distribution for the amount to crop, only one
        single value will be sampled from that probability distribution
        and used for all sides. I.e. the crop amount then is the same
        for all sides.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Crop(px=(0, 10))

    crops each side by a random value from the range 0px to 10px (the value
    is sampled per side).

    >>> aug = iaa.Crop(px=(0, 10), sample_independently=False)

    samples one value v from the discrete range [0..10] and crops all sides
    by v pixels.

    >>> aug = iaa.Crop(px=(0, 10), keep_size=False)

    crops each side by a random value from the range 0px to 10px (the value
    is sampled per side). After cropping, the images are NOT resized to their
    original size (i.e. the images may end up having different heights/widths).

    >>> aug = iaa.Crop(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    crops the top and bottom by a random value from the range 0px to 10px
    and the left and right by a random value in the range 0px to 5px.

    >>> aug = iaa.Crop(percent=(0, 0.1))

    crops each side by a random value from the range 0 percent to
    10 percent. (Percent with respect to the side's size, e.g. for the
    top side it uses the image's height.)

    >>> aug = iaa.Crop(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    crops each side by either 5 percent or 10 percent.

    """

    def recursive_negate(v):
        if v is None:
            return v
        elif ia.is_single_number(v):
            ia.do_assert(v >= 0)
            return -v
        elif isinstance(v, iap.StochasticParameter):
            return iap.Multiply(v, -1)
        elif isinstance(v, tuple):
            return tuple([recursive_negate(v_) for v_ in v])
        elif isinstance(v, list):
            return [recursive_negate(v_) for v_ in v]
        else:
            raise Exception("Expected None or int or float or StochasticParameter or list or tuple, got %s." % (type(v),))

    px = recursive_negate(px)
    percent = recursive_negate(percent)

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    aug = CropAndPad(
        px=px, percent=percent,
        keep_size=keep_size, sample_independently=sample_independently,
        name=name, deterministic=deterministic, random_state=random_state
    )
    return aug


class PadToFixedSize(Augmenter):

    """
    Augmenter that pads the images to specified width/height only if specified width/height is greater than the width/height of input images.
    The offset varies randomly within valid region (i.e. each input image is fully included in its output image).

    Parameters
    ----------
    width : int
        Minimum width of new images.

    height : int
        Minimum height of new images.

    pad_mode : ia.ALL or string or list of strings or StochasticParameter, optional(default="constant")
        See `CropAndPad.__init__()`

    pad_cval : number or tuple of number or list of number or StochasticParameter, optional(default=0)
        See `CropAndPad.__init__()`

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.PadToFixedSize(width=100, height=100)

    for edges smaller than 100 pixels, pads to 100 pixels, does nothing for the other edges.

    >>> aug = iaa.Sequential([
            iaa.PadToFixedSize(width=100, height=100),
            iaa.CropToFixedSize(width=100, height=100)
        ])

    pads to 100x100 pixel for smaller images, and crops to 100x100 pixel for larger images.
    The output images have fixed size, 100x100 pixel.

    """

    def __init__(self, width, height, pad_mode="constant", pad_cval=0, name=None, deterministic=False, random_state=None):
        super(PadToFixedSize, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.size = width, height
        self.position = (iap.Uniform(0.0, 1.0), iap.Uniform(0.0, 1.0))

        pad_modes_available = set(["constant", "edge", "linear_ramp", "maximum", "median", "minimum", "reflect", "symmetric", "wrap"])
        if pad_mode == ia.ALL:
            self.pad_mode = iap.Choice(list(pad_modes_available))
        elif ia.is_string(pad_mode):
            ia.do_assert(pad_mode in pad_modes_available)
            self.pad_mode = iap.Deterministic(pad_mode)
        elif isinstance(pad_mode, list):
            ia.do_assert(all([v in pad_modes_available for v in pad_mode]))
            self.pad_mode = iap.Choice(pad_mode)
        elif isinstance(pad_mode, iap.StochasticParameter):
            self.pad_mode = pad_mode
        else:
            raise Exception("Expected pad_mode to be ia.ALL or string or list of strings or StochasticParameter, got %s." % (type(pad_mode),))

        self.pad_cval = iap.handle_discrete_param(pad_cval, "pad_cval", value_range=(0, 255), tuple_to_uniform=True, list_to_choice=True, allow_floats=True)

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        w, h = self.size
        pad_xs, pad_ys, pad_modes, pad_cvals = self._draw_samples(nb_images, random_state)
        for i in sm.xrange(nb_images):
            image = images[i]
            ia.do_assert(image.dtype == np.uint8, "PadToFixedSize() can currently only process images of dtype uint8 (got %s)" % (image.dtype,))
            ih, iw = image.shape[:2]

            if iw < w or ih < h:
                if iw < w:
                    pad_x0 = int(pad_xs[i]*(w-iw+1))
                    pad_x1 = w-iw-pad_x0
                else:
                    pad_x0 = 0
                    pad_x1 = 0

                if ih < h:
                    pad_y0 = int(pad_ys[i]*(h-ih+1))
                    pad_y1 = h-ih-pad_y0
                else:
                    pad_y0 = 0
                    pad_y1 = 0

                if image.ndim == 2:
                    pad_vals = ((pad_y0, pad_y1), (pad_x0, pad_x1))
                else:
                    pad_vals = ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))

                pad_mode = pad_modes[i]
                if pad_mode == "constant":
                    image = np.pad(image, pad_vals, mode=pad_mode, constant_values=pad_cvals[i])
                elif pad_mode == "linear_ramp":
                    image = np.pad(image, pad_vals, mode=pad_mode, end_values=pad_cvals[i])
                else:
                    image = np.pad(image, pad_vals, mode=pad_mode)

            result.append(image)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        w, h = self.size
        pad_xs, pad_ys, _, _ = self._draw_samples(nb_images, random_state)
        for i in sm.xrange(nb_images):
            keypoints_on_image = keypoints_on_images[i]
            ih, iw = keypoints_on_image.shape[:2]

            pad_x = int(pad_xs[i]*(w-iw+1)) if iw < w else 0
            pad_y = int(pad_ys[i]*(h-ih+1)) if ih < h else 0

            keypoints_padded = keypoints_on_image.shift(x=pad_x, y=pad_y)
            keypoints_padded.shape = (max(ih, h), max(iw, w)) + keypoints_padded.shape[2:]

            result.append(keypoints_padded)

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise NotImplementedError()

    def _draw_samples(self, nb_images, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]

        pad_xs = self.position[0].draw_samples(nb_images, random_state=ia.new_random_state(seed + 0))
        pad_ys = self.position[1].draw_samples(nb_images, random_state=ia.new_random_state(seed + 1))

        pad_modes = self.pad_mode.draw_samples(nb_images, random_state=ia.new_random_state(seed + 2))
        pad_cvals = self.pad_cval.draw_samples(nb_images, random_state=ia.new_random_state(seed + 3))
        pad_cvals = np.clip(np.round(pad_cvals), 0, 255).astype(np.uint8)

        return pad_xs, pad_ys, pad_modes, pad_cvals

    def get_parameters(self):
        return [self.position, self.pad_mode, self.pad_cval]


class CropToFixedSize(Augmenter):

    """
    Augmenter that crops the images to specified width/height only if specified width/height is less than the width/height of input images.
    The offset varies randomly within valid region (i.e. each output image is fully included in its input image).

    Parameters
    ----------
    width : int
        Fixed width of new images.

    height : int
        Fixed height of new images.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.CropToFixedSize(width=100, height=100)

    for edges larger than 100 pixels, crops to 100 pixels, does nothing for the other edges.

    >>> aug = iaa.Sequential([
            iaa.PadToFixedSize(width=100, height=100),
            iaa.CropToFixedSize(width=100, height=100)
        ])

    pads to 100x100 pixel for smaller images, and crops to 100x100 pixel for larger images.
    The output images have fixed size, 100x100 pixel.

    """

    def __init__(self, width, height, name=None, deterministic=False, random_state=None):
        super(CropToFixedSize, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.size = width, height
        self.position = (iap.Uniform(0.0, 1.0), iap.Uniform(0.0, 1.0))

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        w, h = self.size
        offset_xs, offset_ys = self._draw_samples(nb_images, random_state)
        for i in sm.xrange(nb_images):
            image = images[i]
            ia.do_assert(image.dtype == np.uint8, "CropToFixedSize() can currently only process images of dtype uint8 (got %s)" % (image.dtype,))
            ih, iw = image.shape[:2]

            if ih > h:
                offset_y = int(offset_ys[i]*(ih-h+1))
                image = image[offset_y:offset_y+h, :, :]

            if iw > w:
                offset_x = int(offset_xs[i]*(iw-w+1))
                image = image[:, offset_x:offset_x+w, :]

            result.append(image)

        result = np.array(result, dtype=np.uint8)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        w, h = self.size
        offset_xs, offset_ys = self._draw_samples(nb_images, random_state)
        for i in sm.xrange(nb_images):
            keypoints_on_image = keypoints_on_images[i]
            ih, iw = keypoints_on_image.shape[:2]

            offset_x = int(offset_xs[i]*(iw-w+1)) if iw > w else 0
            offset_y = int(offset_ys[i]*(ih-h+1)) if ih > h else 0

            keypoints_cropped = keypoints_on_image.shift(x=-offset_x, y=-offset_y)
            keypoints_cropped.shape = (min(ih, h), min(iw, w)) + keypoints_cropped.shape[2:]

            result.append(keypoints_cropped)

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        raise NotImplementedError()

    def _draw_samples(self, nb_images, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        offset_xs = self.position[0].draw_samples(nb_images, random_state=ia.new_random_state(seed + 0))
        offset_ys = self.position[1].draw_samples(nb_images, random_state=ia.new_random_state(seed + 1))

        return offset_xs, offset_ys

    def get_parameters(self):
        return [self.position]
