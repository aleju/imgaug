"""
Augmenters that somehow change the size of the images.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([
        iaa.Resize({"height": 32, "width": 64})
        iaa.Crop((0, 20))
    ])

List of augmenters:

    * Resize
    * CropAndPad
    * Crop
    * Pad
    * PadToFixedSize
    * CropToFixedSize
    * KeepSizeByResize

"""
from __future__ import print_function, division, absolute_import

import re
import functools

import numpy as np
import six.moves as sm

from . import meta
import imgaug as ia
from .. import parameters as iap


def _crop_trbl_to_xyxy(shape, top, right, bottom, left, prevent_zero_size=True):
    if prevent_zero_size:
        top, right, bottom, left = _crop_prevent_zero_size(
            shape[0], shape[1], top, right, bottom, left)

    height, width = shape[0:2]
    x1 = left
    x2 = width - right
    y1 = top
    y2 = height - bottom

    # these steps prevent negative sizes
    # if x2==x1 or y2==y1 then the output arr has size 0 for the respective axis
    # note that if height/width of arr is zero, then y2==y1 or x2==x1, which
    # is still valid, even if height/width is zero and results in a zero-sized
    # axis
    x2 = max(x2, x1)
    y2 = max(y2, y1)

    return x1, y1, x2, y2


def _crop_arr_(arr, top, right, bottom, left, prevent_zero_size=True):
    x1, y1, x2, y2 = _crop_trbl_to_xyxy(arr.shape, top, right, bottom, left,
                                        prevent_zero_size=prevent_zero_size)
    return arr[y1:y2, x1:x2, ...]


def _crop_and_pad_arr(arr, croppings, paddings, pad_mode="constant",
                      pad_cval=0, keep_size=False):
    height, width = arr.shape[0:2]

    image_cr = _crop_arr_(arr, *croppings)

    image_cr_pa = ia.pad(
        image_cr,
        top=paddings[0], right=paddings[1],
        bottom=paddings[2], left=paddings[3],
        mode=pad_mode, cval=pad_cval)

    if keep_size:
        image_cr_pa = ia.imresize_single_image(image_cr_pa, (height, width))

    return image_cr_pa


def _crop_and_pad_heatmap_(heatmap, croppings_img, paddings_img,
                           pad_mode="constant", pad_cval=0.0, keep_size=False):
    return _crop_and_pad_hms_or_segmaps_(heatmap, croppings_img,
                                         paddings_img, pad_mode, pad_cval,
                                         keep_size)


def _crop_and_pad_segmap_(segmap, croppings_img, paddings_img,
                          pad_mode="constant", pad_cval=0, keep_size=False):
    return _crop_and_pad_hms_or_segmaps_(segmap, croppings_img,
                                         paddings_img, pad_mode, pad_cval,
                                         keep_size)


def _crop_and_pad_hms_or_segmaps_(augmentable, croppings_img,
                                  paddings_img, pad_mode="constant",
                                  pad_cval=None, keep_size=False):
    if isinstance(augmentable, ia.HeatmapsOnImage):
        arr_attr_name = "arr_0to1"
        pad_cval = pad_cval if pad_cval is not None else 0.0
    else:
        assert isinstance(augmentable, ia.SegmentationMapsOnImage), (
            "Expected HeatmapsOnImage or SegmentationMapsOnImage, got %s." % (
                type(augmentable)))
        arr_attr_name = "arr"
        pad_cval = pad_cval if pad_cval is not None else 0

    arr = getattr(augmentable, arr_attr_name)
    arr_shape_orig = arr.shape
    augm_shape = augmentable.shape

    croppings_proj = _project_size_changes(croppings_img, augm_shape, arr.shape)
    paddings_proj = _project_size_changes(paddings_img, augm_shape, arr.shape)

    croppings_proj = _crop_prevent_zero_size(arr.shape[0], arr.shape[1],
                                             *croppings_proj)

    arr_cr = _crop_arr_(arr,
                        croppings_proj[0], croppings_proj[1],
                        croppings_proj[2], croppings_proj[3])
    arr_cr_pa = ia.pad(
        arr_cr,
        top=paddings_proj[0], right=paddings_proj[1],
        bottom=paddings_proj[2], left=paddings_proj[3],
        mode=pad_mode,
        cval=pad_cval)

    setattr(augmentable, arr_attr_name, arr_cr_pa)

    if keep_size:
        augmentable = augmentable.resize(arr_shape_orig[0:2])
    else:
        augmentable.shape = _compute_shape_after_crop_and_pad(
            augmentable.shape, croppings_img, paddings_img)
    return augmentable


def _crop_and_pad_kpsoi(kpsoi, croppings_img, paddings_img, keep_size):
    # using the trbl function instead of croppings_img has the advantage
    # of incorporating prevent_zero_size, dealing with zero-sized input image
    # axis and dealing the negative crop amounts
    x1, y1, _x2, _y2 = _crop_trbl_to_xyxy(kpsoi.shape, *croppings_img)
    crop_left = x1
    crop_top = y1

    shifted = kpsoi.shift(
        x=-crop_left+paddings_img[3],
        y=-crop_top+paddings_img[0])
    shifted.shape = _compute_shape_after_crop_and_pad(
            kpsoi.shape, croppings_img, paddings_img)
    if keep_size:
        shifted = shifted.on(kpsoi.shape)
    return shifted


def _compute_shape_after_crop_and_pad(old_shape, croppings, paddings):
    x1, y1, x2, y2 = _crop_trbl_to_xyxy(old_shape, *croppings)
    new_shape = list(old_shape)
    new_shape[0] = y2 - y1 + paddings[0] + paddings[2]
    new_shape[1] = x2 - x1 + paddings[1] + paddings[3]
    return tuple(new_shape)


def _crop_prevent_zero_size(height, width, crop_top, crop_right, crop_bottom,
                            crop_left):
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

        crop_right = crop_right - regain_right
        crop_left = crop_left - regain_left

    return (
        max(crop_top, 0), max(crop_right, 0), max(crop_bottom, 0),
        max(crop_left, 0))


def _project_size_changes(trbl, from_shape, to_shape):
    if from_shape[0:2] == to_shape[0:2]:
        return trbl

    height_to = to_shape[0]
    width_to = to_shape[1]
    height_from = from_shape[0]
    width_from = from_shape[1]

    top = trbl[0]
    right = trbl[1]
    bottom = trbl[2]
    left = trbl[3]

    # Adding/subtracting 1e-4 here helps for the case where a heatmap/segmap
    # is exactly half the size of an image and the size change on an axis is
    # an odd value. Then the projected value would end up being <something>.5
    # and the rounding would always round up to the next integer. If both
    # sides then have the same change, they are both rounded up, resulting
    # in more change than expected.
    # E.g. image height is 8, map height is 4, change is 3 at the top and 3 at
    # the bottom. The changes are projected to 4*(3/8) = 1.5 and both rounded
    # up to 2.0. Hence, the maps are changed by 4 (100% of the map height,
    # vs. 6 for images, which is 75% of the image height).
    top = _int_r(height_to * (top/height_from) - 1e-4)
    right = _int_r(width_to * (right/width_from) + 1e-4)
    bottom = _int_r(height_to * (bottom/height_from) + 1e-4)
    left = _int_r(width_to * (left/width_from) - 1e-4)

    return top, right, bottom, left


def _int_r(value):
    return int(np.round(value))


# TODO somehow integrate this with ia.pad()
def _handle_pad_mode_param(pad_mode):
    pad_modes_available = {
        "constant", "edge", "linear_ramp", "maximum", "mean", "median",
        "minimum", "reflect", "symmetric", "wrap"}
    if pad_mode == ia.ALL:
        return iap.Choice(list(pad_modes_available))
    elif ia.is_string(pad_mode):
        assert pad_mode in pad_modes_available, (
            "Value '%s' is not a valid pad mode. Valid pad modes are: %s." % (
                pad_mode, ", ".join(pad_modes_available)))
        return iap.Deterministic(pad_mode)
    elif isinstance(pad_mode, list):
        assert all([v in pad_modes_available for v in pad_mode]), (
            "At least one in list %s is not a valid pad mode. Valid pad "
            "modes are: %s." % (str(pad_mode), ", ".join(pad_modes_available)))
        return iap.Choice(pad_mode)
    elif isinstance(pad_mode, iap.StochasticParameter):
        return pad_mode
    raise Exception(
        "Expected pad_mode to be ia.ALL or string or list of strings or "
        "StochasticParameter, got %s." % (type(pad_mode),))


def _handle_position_parameter(position):
    if position == "uniform":
        return iap.Uniform(0.0, 1.0), iap.Uniform(0.0, 1.0)
    elif position == "normal":
        return (
            iap.Clip(iap.Normal(loc=0.5, scale=0.35 / 2),
                     minval=0.0, maxval=1.0),
            iap.Clip(iap.Normal(loc=0.5, scale=0.35 / 2),
                     minval=0.0, maxval=1.0)
        )
    elif position == "center":
        return iap.Deterministic(0.5), iap.Deterministic(0.5)
    elif (ia.is_string(position)
          and re.match(r"^(left|center|right)-(top|center|bottom)$", position)):
        mapping = {"top": 0.0, "center": 0.5, "bottom": 1.0, "left": 0.0,
                   "right": 1.0}
        return (
            iap.Deterministic(mapping[position.split("-")[0]]),
            iap.Deterministic(mapping[position.split("-")[1]])
        )
    elif isinstance(position, iap.StochasticParameter):
        return position
    elif isinstance(position, tuple):
        assert len(position) == 2, (
            "Expected tuple with two entries as position parameter. "
            "Got %d entries with types %s.." % (
                len(position), str([type(el) for el in position])))
        for el in position:
            if ia.is_single_number(el) and (el < 0 or el > 1.0):
                raise Exception(
                    "Both position values must be within the value range "
                    "[0.0, 1.0]. Got type %s with value %.8f." % (
                        type(el), el,))
        position = [iap.Deterministic(el)
                    if ia.is_single_number(el)
                    else el for el in position]

        only_sparams = all([isinstance(el, iap.StochasticParameter)
                            for el in position])
        assert only_sparams, (
            "Expected tuple with two entries that are both either "
            "StochasticParameter or float/int. Got types %s." % (
                str([type(el) for el in position])
            ))
        return tuple(position)
    else:
        raise Exception(
            "Expected one of the following as position parameter: string "
            "'uniform', string 'normal', string 'center', a string matching "
            "regex ^(left|center|right)-(top|center|bottom)$, a single "
            "StochasticParameter or a tuple of two entries, both being either "
            "StochasticParameter or floats or int. Got instead type %s with "
            "content '%s'." % (
                type(position),
                (str(position)
                 if len(str(position)) < 20
                 else str(position)[0:20] + "...")
            )
        )


@ia.deprecated(alt_func="Resize",
               comment="Resize has the exactly same interface as Scale.")
def Scale(*args, **kwargs):
    return Resize(*args, **kwargs)


class Resize(meta.Augmenter):
    """Augmenter that resizes images to specified heights and widths.

    dtype support::

        See :func:`imgaug.imgaug.imresize_many_images`.

    Parameters
    ----------
    size : 'keep' or int or float or tuple of int or tuple of float or list of int or list of float or imgaug.parameters.StochasticParameter or dict
        The new size of the images.

            * If this has the string value ``keep``, the original height and
              width values will be kept (image is not resized).
            * If this is an ``int``, this value will always be used as the new
              height and width of the images.
            * If this is a ``float`` ``v``, then per image the image's height
              ``H`` and width ``W`` will be changed to ``H*v`` and ``W*v``.
            * If this is a ``tuple``, it is expected to have two entries
              ``(a, b)``. If at least one of these are ``float`` s, a value
              will be sampled from range ``[a, b]`` and used as the ``float``
              value to resize the image (see above). If both are ``int`` s, a
              value will be sampled from the discrete range ``[a..b]`` and
              used as the integer value to resize the image (see above).
            * If this is a ``list``, a random value from the ``list`` will be
              picked to resize the image. All values in the ``list`` must be
              ``int`` s or ``float`` s (no mixture is possible).
            * If this is a ``StochasticParameter``, then this parameter will
              first be queried once per image. The resulting value will be used
              for both height and width.
            * If this is a ``dict``, it may contain the keys ``height`` and
              ``width`` or the keys ``shorter-side`` and ``longer-side``. Each
              key may have the same datatypes as above and describes the
              scaling on x and y-axis or the shorter and longer axis,
              respectively. Both axis are sampled independently. Additionally,
              one of the keys may have the value ``keep-aspect-ratio``, which
              means that the respective side of the image will be resized so
              that the original aspect ratio is kept. This is useful when only
              resizing one image size by a pixel value (e.g. resize images to
              a height of ``64`` pixels and resize the width so that the
              overall aspect ratio is maintained).

    interpolation : imgaug.ALL or int or str or list of int or list of str or imgaug.parameters.StochasticParameter, optional
        Interpolation to use.

            * If ``imgaug.ALL``, then a random interpolation from ``nearest``,
              ``linear``, ``area`` or ``cubic`` will be picked (per image).
            * If ``int``, then this interpolation will always be used.
              Expected to be any of the following:
              ``cv2.INTER_NEAREST``, ``cv2.INTER_LINEAR``, ``cv2.INTER_AREA``,
              ``cv2.INTER_CUBIC``
            * If string, then this interpolation will always be used.
              Expected to be any of the following:
              ``nearest``, ``linear``, ``area``, ``cubic``
            * If ``list`` of ``int`` / ``str``, then a random one of the values
              will be picked per image as the interpolation.
            * If a ``StochasticParameter``, then this parameter will be
              queried per image and is expected to return an ``int`` or
              ``str``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Resize(32)

    Resize all images to ``32x32`` pixels.

    >>> aug = iaa.Resize(0.5)

    Resize all images to ``50`` percent of their original size.

    >>> aug = iaa.Resize((16, 22))

    Resize all images to a random height and width within the discrete
    interval ``[16..22]`` (uniformly sampled per image).

    >>> aug = iaa.Resize((0.5, 0.75))

    Resize all any input image so that its height (``H``) and width (``W``)
    become ``H*v`` and ``W*v``, where ``v`` is uniformly sampled from the
    interval ``[0.5, 0.75]``.

    >>> aug = iaa.Resize([16, 32, 64])

    Resize all images either to ``16x16``, ``32x32`` or ``64x64`` pixels.

    >>> aug = iaa.Resize({"height": 32})

    Resize all images to a height of ``32`` pixels and keeps the original
    width.

    >>> aug = iaa.Resize({"height": 32, "width": 48})

    Resize all images to a height of ``32`` pixels and a width of ``48``.

    >>> aug = iaa.Resize({"height": 32, "width": "keep-aspect-ratio"})

    Resize all images to a height of ``32`` pixels and resizes the
    x-axis (width) so that the aspect ratio is maintained.

    >>> aug = iaa.Resize(
    >>>     {"shorter-side": 224, "longer-side": "keep-aspect-ratio"})

    Resize all images to a height/width of ``224`` pixels, depending on which
    axis is shorter and resize the other axis so that the aspect ratio is
    maintained.

    >>> aug = iaa.Resize({"height": (0.5, 0.75), "width": [16, 32, 64]})

    Resize all images to a height of ``H*v``, where ``H`` is the original
    height and ``v`` is a random value sampled from the interval
    ``[0.5, 0.75]``. The width/x-axis of each image is resized to either
    ``16`` or ``32`` or ``64`` pixels.

    >>> aug = iaa.Resize(32, interpolation=["linear", "cubic"])

    Resize all images to ``32x32`` pixels. Randomly use either ``linear``
    or ``cubic`` interpolation.

    """

    def __init__(self, size, interpolation="cubic",
                 name=None, deterministic=False, random_state=None):
        super(Resize, self).__init__(name=name, deterministic=deterministic,
                                     random_state=random_state)

        self.size, self.size_order = self._handle_size_arg(size, False)
        self.interpolation = self._handle_interpolation_arg(interpolation)

    @classmethod
    def _handle_size_arg(cls, size, subcall):
        def _dict_to_size_tuple(v1, v2):
            kaa = "keep-aspect-ratio"
            not_both_kaa = (v1 != kaa or v2 != kaa)
            assert not_both_kaa, (
                "Expected at least one value to not be \"keep-aspect-ratio\", "
                "but got it two times.")

            size_tuple = []
            for k in [v1, v2]:
                if k in ["keep-aspect-ratio", "keep"]:
                    entry = iap.Deterministic(k)
                else:
                    entry = cls._handle_size_arg(k, True)
                size_tuple.append(entry)
            return tuple(size_tuple)

        def _contains_any_key(dict_, keys):
            return any([key in dict_ for key in keys])

        # HW = height, width
        # SL = shorter, longer
        size_order = "HW"

        if size == "keep":
            result = iap.Deterministic("keep")
        elif ia.is_single_number(size):
            assert size > 0, "Expected only values > 0, got %s" % (size,)
            result = iap.Deterministic(size)
        elif not subcall and isinstance(size, dict):
            if len(size.keys()) == 0:
                result = iap.Deterministic("keep")
            elif _contains_any_key(size, ["height", "width"]):
                height = size.get("height", "keep")
                width = size.get("width", "keep")
                result = _dict_to_size_tuple(height, width)
            elif _contains_any_key(size, ["shorter-side", "longer-side"]):
                shorter = size.get("shorter-side", "keep")
                longer = size.get("longer-side", "keep")
                result = _dict_to_size_tuple(shorter, longer)
                size_order = "SL"
            else:
                raise ValueError(
                    "Expected dictionary containing no keys, "
                    "the keys \"height\" and/or \"width\", "
                    "or the keys \"shorter-side\" and/or \"longer-side\". "
                    "Got keys: %s." % (str(size.keys()),))
        elif isinstance(size, tuple):
            assert len(size) == 2, (
                "Expected size tuple to contain exactly 2 values, "
                "got %d." % (len(size),))
            assert size[0] > 0 and size[1] > 0, (
                "Expected size tuple to only contain values >0, "
                "got %d and %d." % (size[0], size[1]))
            if ia.is_single_float(size[0]) or ia.is_single_float(size[1]):
                result = iap.Uniform(size[0], size[1])
            else:
                result = iap.DiscreteUniform(size[0], size[1])
        elif isinstance(size, list):
            if len(size) == 0:
                result = iap.Deterministic("keep")
            else:
                all_int = all([ia.is_single_integer(v) for v in size])
                all_float = all([ia.is_single_float(v) for v in size])
                assert all_int or all_float, (
                    "Expected to get only integers or floats.")
                assert all([v > 0 for v in size]), (
                    "Expected all values to be >0.")
                result = iap.Choice(size)
        elif isinstance(size, iap.StochasticParameter):
            result = size
        else:
            raise ValueError(
                "Expected number, tuple of two numbers, list of numbers, "
                "dictionary of form "
                "{'height': number/tuple/list/'keep-aspect-ratio'/'keep', "
                "'width': <analogous>}, dictionary of form "
                "{'shorter-side': number/tuple/list/'keep-aspect-ratio'/"
                "'keep', 'longer-side': <analogous>} "
                "or StochasticParameter, got %s." % (type(size),)
            )

        if subcall:
            return result
        return result, size_order

    @classmethod
    def _handle_interpolation_arg(cls, interpolation):
        if interpolation == ia.ALL:
            interpolation = iap.Choice(
                ["nearest", "linear", "area", "cubic"])
        elif ia.is_single_integer(interpolation):
            interpolation = iap.Deterministic(interpolation)
        elif ia.is_string(interpolation):
            interpolation = iap.Deterministic(interpolation)
        elif ia.is_iterable(interpolation):
            interpolation = iap.Choice(interpolation)
        elif isinstance(interpolation, iap.StochasticParameter):
            interpolation = interpolation
        else:
            raise Exception(
                "Expected int or string or iterable or StochasticParameter, "
                "got %s." % (type(interpolation),))
        return interpolation

    def _augment_batch(self, batch, random_state, parents, hooks):
        nb_rows = batch.nb_rows
        samples = self._draw_samples(nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images,
                                                           samples)

        if batch.heatmaps is not None:
            # TODO this uses the same interpolation as for images for heatmaps
            #      while other augmenters resort to cubic
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps, "arr_0to1", samples)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, "arr",
                (samples[0], samples[1], [None] * nb_rows))

        for augm_name in ["keypoints", "bounding_boxes", "polygons",
                          "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(
                    self._augment_keypoints_by_samples,
                    samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    def _augment_images_by_samples(self, images, samples):
        input_was_array = False
        input_dtype = None
        if ia.is_np_array(images):
            input_was_array = True
            input_dtype = images.dtype

        samples_a, samples_b, samples_ip = samples
        result = []
        for i, image in enumerate(images):
            h, w = self._compute_height_width(image.shape, samples_a[i],
                                              samples_b[i], self.size_order)
            image_rs = ia.imresize_single_image(image, (h, w),
                                                interpolation=samples_ip[i])
            result.append(image_rs)

        if input_was_array:
            all_same_size = (len(set([image.shape for image in result])) == 1)
            if all_same_size:
                result = np.array(result, dtype=input_dtype)

        return result

    def _augment_maps_by_samples(self, augmentables, arr_attr_name, samples):
        result = []
        samples_h, samples_w, samples_ip = samples

        for i, augmentable in enumerate(augmentables):
            arr = getattr(augmentable, arr_attr_name)
            arr_shape = arr.shape
            img_shape = augmentable.shape
            h_img, w_img = self._compute_height_width(
                img_shape, samples_h[i], samples_w[i], self.size_order)
            h = int(np.round(h_img * (arr_shape[0] / img_shape[0])))
            w = int(np.round(w_img * (arr_shape[1] / img_shape[1])))
            h = max(h, 1)
            w = max(w, 1)
            if samples_ip[0] is not None:
                # TODO change this for heatmaps to always have cubic or
                #      automatic interpolation?
                augmentable_resize = augmentable.resize(
                    (h, w), interpolation=samples_ip[i])
            else:
                augmentable_resize = augmentable.resize((h, w))
            augmentable_resize.shape = (h_img, w_img) + img_shape[2:]
            result.append(augmentable_resize)

        return result

    def _augment_keypoints_by_samples(self, kpsois, samples):
        result = []
        samples_a, samples_b, _samples_ip = samples
        for i, kpsoi in enumerate(kpsois):
            h, w = self._compute_height_width(
                kpsoi.shape, samples_a[i], samples_b[i], self.size_order)
            new_shape = (h, w) + kpsoi.shape[2:]
            keypoints_on_image_rs = kpsoi.on(new_shape)

            result.append(keypoints_on_image_rs)

        return result

    def _draw_samples(self, nb_images, random_state):
        rngs = random_state.duplicate(3)
        if isinstance(self.size, tuple):
            samples_h = self.size[0].draw_samples(nb_images,
                                                  random_state=rngs[0])
            samples_w = self.size[1].draw_samples(nb_images,
                                                  random_state=rngs[1])
        else:
            samples_h = self.size.draw_samples(nb_images, random_state=rngs[0])
            samples_w = samples_h

        samples_ip = self.interpolation.draw_samples(nb_images,
                                                     random_state=rngs[2])
        return samples_h, samples_w, samples_ip

    @classmethod
    def _compute_height_width(cls, image_shape, sample_a, sample_b, size_order):
        imh, imw = image_shape[0:2]

        if size_order == 'SL':
            # size order: short, long
            if imh < imw:
                h, w = sample_a, sample_b
            else:
                w, h = sample_a, sample_b

        else:
            # size order: height, width
            h, w = sample_a, sample_b

        if ia.is_single_float(h):
            assert h > 0, "Expected 'h' to be >0, got %.4f" % (h,)
            h = int(np.round(imh * h))
            h = h if h > 0 else 1
        elif h == "keep":
            h = imh
        if ia.is_single_float(w):
            assert w > 0, "Expected 'w' to be >0, got %.4f" % (w,)
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
        return [self.size, self.interpolation, self.size_order]


class _CropAndPadSamplingResult(object):
    def __init__(self, crop_top, crop_right, crop_bottom, crop_left,
                 pad_top, pad_right, pad_bottom, pad_left, pad_mode, pad_cval):
        self.crop_top = crop_top
        self.crop_right = crop_right
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.pad_top = pad_top
        self.pad_right = pad_right
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_mode = pad_mode
        self.pad_cval = pad_cval

    @property
    def croppings(self):
        return self.crop_top, self.crop_right, self.crop_bottom, self.crop_left

    @property
    def paddings(self):
        return self.pad_top, self.pad_right, self.pad_bottom, self.pad_left


class CropAndPad(meta.Augmenter):
    """Crop/pad images by pixel amounts or fractions of image sizes.

    Cropping removes pixels at the sides (i.e. extracts a subimage from
    a given full image). Padding adds pixels to the sides (e.g. black pixels).

    This augmenter will never crop images below a height or width of ``1``.

    .. note ::

        This augmenter automatically resizes images back to their original size
        after it has augmented them. To deactivate this, add the
        parameter ``keep_size=False``.

    dtype support::

        if (keep_size=False)::

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested
            * ``uint64``: yes; tested
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested
            * ``int64``: yes; tested
            * ``float16``: yes; tested
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: yes; tested
            * ``bool``: yes; tested

        if (keep_size=True)::

            minimum of (
                ``imgaug.augmenters.size.CropAndPad(keep_size=False)``,
                :func:`imgaug.imgaug.imresize_many_images`
            )

    Parameters
    ----------
    px : None or int or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop (negative values) or pad (positive values)
        on each side of the image. Either this or the parameter `percent` may
        be set, not both at the same time.

            * If ``None``, then pixel-based cropping/padding will not be used.
            * If ``int``, then that exact number of pixels will always be
              cropped/padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left), unless `sample_independently` is set to ``False``,
              as then only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
              then each side will be cropped/padded by a random amount sampled
              uniformly per image and side from the inteval ``[a, b]``. If
              however `sample_independently` is set to ``False``, only one
              value will be sampled per image and used for all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``int`` (always
              crop/pad by exactly that value), a ``tuple`` of two ``int`` s
              ``a`` and ``b`` (crop/pad by an amount within ``[a, b]``), a
              ``list`` of ``int`` s (crop/pad by a random value that is
              contained in the ``list``) or a ``StochasticParameter`` (sample
              the amount to crop/pad from that parameter).

    percent : None or number or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop (negative values) or pad (positive values)
        on each side of the image given as a *fraction* of the image
        height/width. E.g. if this is set to ``-0.1``, the augmenter will
        always crop away ``10%`` of the image's height at both the top and the
        bottom (both ``10%`` each), as well as ``10%`` of the width at the
        right and left.
        Expected value range is ``(-1.0, inf)``.
        Either this or the parameter `px` may be set, not both
        at the same time.

            * If ``None``, then fraction-based cropping/padding will not be
              used.
            * If ``number``, then that fraction will always be cropped/padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left). If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``,
              then each side will be cropped/padded by a random fraction
              sampled uniformly per image and side from the interval
              ``[a, b]``. If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``float``
              (always crop/pad by exactly that percent value), a ``tuple`` of
              two ``float`` s ``a`` and ``b`` (crop/pad by a fraction from
              ``[a, b]``), a ``list`` of ``float`` s (crop/pad by a random
              value that is contained in the list) or a ``StochasticParameter``
              (sample the percentage to crop/pad from that parameter).

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        Padding mode to use. The available modes match the numpy padding modes,
        i.e. ``constant``, ``edge``, ``linear_ramp``, ``maximum``, ``median``,
        ``minimum``, ``reflect``, ``symmetric``, ``wrap``. The modes
        ``constant`` and ``linear_ramp`` use extra values, which are provided
        by ``pad_cval`` when necessary. See :func:`imgaug.imgaug.pad` for
        more details.

            * If ``imgaug.ALL``, then a random mode from all available modes
              will be sampled per image.
            * If a ``str``, it will be used as the pad mode for all images.
            * If a ``list`` of ``str``, a random one of these will be sampled
              per image and used as the mode.
            * If ``StochasticParameter``, a random mode will be sampled from
              this parameter per image.

    pad_cval : number or tuple of number list of number or imgaug.parameters.StochasticParameter, optional
        The constant value to use if the pad mode is ``constant`` or the end
        value to use if the mode is ``linear_ramp``.
        See :func:`imgaug.imgaug.pad` for more details.

            * If ``number``, then that value will be used.
            * If a ``tuple`` of two ``number`` s and at least one of them is
              a ``float``, then a random number will be uniformly sampled per
              image from the continuous interval ``[a, b]`` and used as the
              value. If both ``number`` s are ``int`` s, the interval is
              discrete.
            * If a ``list`` of ``number``, then a random value will be chosen
              from the elements of the ``list`` and used as the value.
            * If ``StochasticParameter``, a random value will be sampled from
              that parameter per image.

    keep_size : bool, optional
        After cropping and padding, the result image will usually have a
        different height/width compared to the original input image. If this
        parameter is set to ``True``, then the cropped/padded image will be
        resized to the input image's size, i.e. the augmenter's output shape
        is always identical to the input shape.

    sample_independently : bool, optional
        If ``False`` *and* the values for `px`/`percent` result in exactly
        *one* probability distribution for all image sides, only one single
        value will be sampled from that probability distribution and used for
        all sides. I.e. the crop/pad amount then is the same for all sides.
        If ``True``, four values will be sampled independently, one per side.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CropAndPad(px=(-10, 0))

    Crop each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[-10..0]``.

    >>> aug = iaa.CropAndPad(px=(0, 10))

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding happens by
    zero-padding, i.e. it adds black pixels (default setting).

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode="edge")

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding uses the
    ``edge`` mode from numpy's pad function, i.e. the pixel colors around
    the image sides are repeated.

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode=["constant", "edge"])

    Similar to the previous example, but uses zero-padding (``constant``) for
    half of the images and ``edge`` padding for the other half.

    >>> aug = iaa.CropAndPad(px=(0, 10), pad_mode=ia.ALL, pad_cval=(0, 255))

    Similar to the previous example, but uses any available padding mode.
    In case the padding mode ends up being ``constant`` or ``linear_ramp``,
    and random intensity is uniformly sampled (once per image) from the
    discrete interval ``[0..255]`` and used as the intensity of the new
    pixels.

    >>> aug = iaa.CropAndPad(px=(0, 10), sample_independently=False)

    Pad each side by a random pixel value sampled uniformly once per image
    from the discrete interval ``[0..10]``. Each sampled value is used
    for *all* sides of the corresponding image.

    >>> aug = iaa.CropAndPad(px=(0, 10), keep_size=False)

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. Afterwards, do **not**
    resize the padded image back to the input image's size. This will increase
    the image's height and width by a maximum of ``20`` pixels.

    >>> aug = iaa.CropAndPad(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    Pad the top and bottom by a random pixel value sampled uniformly from the
    discrete interval ``[0..10]``. Pad the left and right analogously by
    a random value sampled from ``[0..5]``. Each value is always sampled
    independently.

    >>> aug = iaa.CropAndPad(percent=(0, 0.1))

    Pad each side by a random fraction sampled uniformly from the continuous
    interval ``[0.0, 0.10]``. The fraction is sampled once per image and
    side. E.g. a sampled fraction of ``0.1`` for the top side would pad by
    ``0.1*H``, where ``H`` is the height of the input image.

    >>> aug = iaa.CropAndPad(
    >>>     percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    Pads each side by either ``5%`` or ``10%``. The values are sampled
    once per side and image.

    >>> aug = iaa.CropAndPad(px=(-10, 10))

    Sample uniformly per image and side a value ``v`` from the discrete range
    ``[-10..10]``. Then either crop (negative sample) or pad (positive sample)
    the side by ``v`` pixels.

    """

    def __init__(self, px=None, percent=None, pad_mode="constant", pad_cval=0,
                 keep_size=True, sample_independently=True,
                 name=None, deterministic=False, random_state=None):
        super(CropAndPad, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.mode, self.all_sides, self.top, self.right, self.bottom, \
            self.left = self._handle_px_and_percent_args(px, percent)

        self.pad_mode = _handle_pad_mode_param(pad_mode)
        # TODO enable ALL here, like in e.g. Affine
        self.pad_cval = iap.handle_discrete_param(
            pad_cval, "pad_cval", value_range=None, tuple_to_uniform=True,
            list_to_choice=True, allow_floats=True)

        self.keep_size = keep_size
        self.sample_independently = sample_independently

        # set these to None to use the same values as sampled for the
        # images (not tested)
        self._pad_mode_heatmaps = "constant"
        self._pad_mode_segmentation_maps = "constant"
        self._pad_cval_heatmaps = 0.0
        self._pad_cval_segmentation_maps = 0

    @classmethod
    def _handle_px_and_percent_args(cls, px, percent):
        all_sides = None
        top, right, bottom, left = None, None, None, None

        if px is None and percent is None:
            mode = "noop"
        elif px is not None and percent is not None:
            raise Exception("Can only pad by pixels or percent, not both.")
        elif px is not None:
            mode = "px"
            all_sides, top, right, bottom, left = cls._handle_px_arg(px)
        else:  # = elif percent is not None:
            mode = "percent"
            all_sides, top, right, bottom, left = cls._handle_percent_arg(
                percent)
        return mode, all_sides, top, right, bottom, left

    @classmethod
    def _handle_px_arg(cls, px):
        all_sides = None
        top, right, bottom, left = None, None, None, None

        if ia.is_single_integer(px):
            all_sides = iap.Deterministic(px)
        elif isinstance(px, tuple):
            assert len(px) in [2, 4], (
                "Expected 'px' given as a tuple to contain 2 or 4 "
                "entries, got %d." % (len(px),))

            def handle_param(p):
                if ia.is_single_integer(p):
                    return iap.Deterministic(p)
                elif isinstance(p, tuple):
                    assert len(p) == 2, (
                        "Expected tuple of 2 values, got %d." % (len(p)))
                    only_ints = (
                        ia.is_single_integer(p[0])
                        and ia.is_single_integer(p[1]))
                    assert only_ints, (
                        "Expected tuple of integers, got %s and %s." % (
                            type(p[0]), type(p[1])))
                    return iap.DiscreteUniform(p[0], p[1])
                elif isinstance(p, list):
                    assert len(p) > 0, (
                        "Expected non-empty list, but got empty one.")
                    assert all([ia.is_single_integer(val) for val in p]), (
                        "Expected list of ints, got types %s." % (
                            ", ".join([str(type(v)) for v in p])))
                    return iap.Choice(p)
                elif isinstance(p, iap.StochasticParameter):
                    return p
                else:
                    raise Exception(
                        "Expected int, tuple of two ints, list of ints or "
                        "StochasticParameter, got type %s." % (type(p),))

            if len(px) == 2:
                all_sides = handle_param(px)
            else:  # len == 4
                top = handle_param(px[0])
                right = handle_param(px[1])
                bottom = handle_param(px[2])
                left = handle_param(px[3])
        elif isinstance(px, iap.StochasticParameter):
            top = right = bottom = left = px
        else:
            raise Exception(
                "Expected int, tuple of 4 "
                "ints/tuples/lists/StochasticParameters or "
                "StochasticParameter, got type %s." % (type(px),))
        return all_sides, top, right, bottom, left

    @classmethod
    def _handle_percent_arg(cls, percent):
        all_sides = None
        top, right, bottom, left = None, None, None, None

        if ia.is_single_number(percent):
            assert percent > -1.0, (
                "Expected 'percent' to be >-1.0, got %.4f." % (percent,))
            all_sides = iap.Deterministic(percent)
        elif isinstance(percent, tuple):
            assert len(percent) in [2, 4], (
                "Expected 'percent' given as a tuple to contain 2 or 4 "
                "entries, got %d." % (len(percent),))

            def handle_param(p):
                if ia.is_single_number(p):
                    return iap.Deterministic(p)
                elif isinstance(p, tuple):
                    assert len(p) == 2, (
                        "Expected tuple of 2 values, got %d." % (len(p),))
                    only_numbers = (
                        ia.is_single_number(p[0])
                        and ia.is_single_number(p[1]))
                    assert only_numbers, (
                        "Expected tuple of numbers, got %s and %s." % (
                            type(p[0]), type(p[1])))
                    assert p[0] > -1.0 and p[1] > -1.0, (
                        "Expected tuple of values >-1.0, got %.4f and "
                        "%.4f." % (p[0], p[1]))
                    return iap.Uniform(p[0], p[1])
                elif isinstance(p, list):
                    assert len(p) > 0, (
                        "Expected non-empty list, but got empty one.")
                    assert all([ia.is_single_number(val) for val in p]), (
                        "Expected list of numbers, got types %s." % (
                            ", ".join([str(type(v)) for v in p])))
                    assert all([val > -1.0 for val in p]), (
                        "Expected list of values >-1.0, got values %s." % (
                            ", ".join(["%.4f" % (v,) for v in p])))
                    return iap.Choice(p)
                elif isinstance(p, iap.StochasticParameter):
                    return p
                else:
                    raise Exception(
                        "Expected int, tuple of two ints, list of ints or "
                        "StochasticParameter, got type %s." % (type(p),))

            if len(percent) == 2:
                all_sides = handle_param(percent)
            else:  # len == 4
                top = handle_param(percent[0])
                right = handle_param(percent[1])
                bottom = handle_param(percent[2])
                left = handle_param(percent[3])
        elif isinstance(percent, iap.StochasticParameter):
            top = right = bottom = left = percent
        else:
            raise Exception(
                "Expected number, tuple of 4 "
                "numbers/tuples/lists/StochasticParameters or "
                "StochasticParameter, got type %s." % (type(percent),))
        return all_sides, top, right, bottom, left

    def _augment_batch(self, batch, random_state, parents, hooks):
        shapes = batch.get_rowwise_shapes()
        samples = self._draw_samples(random_state, shapes)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images,
                                                           samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps,
                self._pad_mode_heatmaps, self._pad_cval_heatmaps,
                samples)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps,
                self._pad_mode_segmentation_maps,
                self._pad_cval_segmentation_maps, samples)

        for augm_name in ["keypoints", "bounding_boxes", "polygons",
                          "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(
                    self._augment_keypoints_by_samples,
                    samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    def _augment_images_by_samples(self, images, samples):
        result = []
        for i, image in enumerate(images):
            samples_i = samples[i]

            image_cr_pa = _crop_and_pad_arr(
                image, samples_i.croppings, samples_i.paddings,
                samples_i.pad_mode, samples_i.pad_cval, self.keep_size)

            result.append(image_cr_pa)

        if ia.is_np_array(images):
            if self.keep_size:
                result = np.array(result, dtype=images.dtype)
            else:
                nb_shapes = len(set([image.shape for image in result]))
                if nb_shapes == 1:
                    result = np.array(result, dtype=images.dtype)

        return result

    def _augment_maps_by_samples(self, augmentables, pad_mode, pad_cval,
                                 samples):
        result = []
        for i, augmentable in enumerate(augmentables):
            samples_img = samples[i]

            augmentable = _crop_and_pad_hms_or_segmaps_(
                augmentable,
                croppings_img=samples_img.croppings,
                paddings_img=samples_img.paddings,
                pad_mode=(pad_mode
                          if pad_mode is not None
                          else samples_img.pad_mode),
                pad_cval=(pad_cval
                          if pad_cval is not None
                          else samples_img.pad_cval),
                keep_size=self.keep_size
            )

            result.append(augmentable)

        return result

    def _augment_keypoints_by_samples(self, keypoints_on_images, samples):
        result = []
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            samples_i = samples[i]

            kpsoi_aug = _crop_and_pad_kpsoi(
                keypoints_on_image, croppings_img=samples_i.croppings,
                paddings_img=samples_i.paddings, keep_size=self.keep_size)
            result.append(kpsoi_aug)

        return result

    def _draw_samples(self, random_state, shapes):
        nb_rows = len(shapes)

        if self.mode == "noop":
            top = right = bottom = left = np.full((nb_rows,), 0,
                                                  dtype=np.int32)
        else:
            if self.all_sides is not None:
                if self.sample_independently:
                    samples = self.all_sides.draw_samples(
                        (nb_rows, 4), random_state=random_state)
                    top = samples[:, 0]
                    right = samples[:, 1]
                    bottom = samples[:, 2]
                    left = samples[:, 3]
                else:
                    sample = self.all_sides.draw_samples(
                        (nb_rows,), random_state=random_state)
                    top = right = bottom = left = sample
            else:
                top = self.top.draw_samples(
                    (nb_rows,), random_state=random_state)
                right = self.right.draw_samples(
                    (nb_rows,), random_state=random_state)
                bottom = self.bottom.draw_samples(
                    (nb_rows,), random_state=random_state)
                left = self.left.draw_samples(
                    (nb_rows,), random_state=random_state)

            if self.mode == "px":
                # no change necessary for pixel values
                pass
            elif self.mode == "percent":
                # percentage values have to be transformed to pixel values
                shapes_arr = np.array([shape[0:2] for shape in shapes],
                                      dtype=np.float32)
                heights = shapes_arr[:, 0]
                widths = shapes_arr[:, 1]
                top = np.round(heights * top).astype(np.int32)
                right = np.round(widths * right).astype(np.int32)
                bottom = np.round(heights * bottom).astype(np.int32)
                left = np.round(widths * left).astype(np.int32)
            else:
                raise Exception("Invalid mode")

        def _only_above_zero(arr):
            arr = np.copy(arr)
            mask = (arr < 0)
            arr[mask] = 0
            return arr

        crop_top = _only_above_zero((-1) * top)
        crop_right = _only_above_zero((-1) * right)
        crop_bottom = _only_above_zero((-1) * bottom)
        crop_left = _only_above_zero((-1) * left)

        pad_top = _only_above_zero(top)
        pad_right = _only_above_zero(right)
        pad_bottom = _only_above_zero(bottom)
        pad_left = _only_above_zero(left)

        pad_mode = self.pad_mode.draw_samples((nb_rows,),
                                              random_state=random_state)
        pad_cval = self.pad_cval.draw_samples((nb_rows,),
                                              random_state=random_state)

        # TODO vectorize this part -- especially return only one instance
        result = []
        for i, shape in enumerate(shapes):
            height, width = shape[0:2]
            crop_top_i, crop_right_i, crop_bottom_i, crop_left_i = \
                _crop_prevent_zero_size(
                    height, width,
                    crop_top[i], crop_right[i], crop_bottom[i], crop_left[i])

            # add here any_crop_y to not warn in case of zero height/width
            # images
            any_crop_y = (crop_top_i > 0 or crop_bottom_i > 0)
            if any_crop_y and crop_top_i + crop_bottom_i >= height:
                ia.warn(
                    "Expected generated crop amounts in CropAndPad for top and "
                    "bottom image side to be less than the image's height, but "
                    "got %d (top) and %d (bottom) vs. image height %d. This "
                    "will result in an image with output height=1 (if input "
                    "height was >=1) or output height=0 (if input height "
                    "was 0)." % (crop_top_i, crop_bottom_i, height))

            # add here any_crop_x to not warn in case of zero height/width
            # images
            any_crop_x = (crop_left_i > 0 or crop_right_i > 0)
            if any_crop_x and crop_left_i + crop_right_i >= width:
                ia.warn(
                    "Expected generated crop amounts in CropAndPad for left "
                    "and right image side to be less than the image's width, "
                    "but got %d (left) and %d (right) vs. image width %d. "
                    "This will result in an image with output width=1 (if "
                    "input width was >=1) or output width=0 (if input width "
                    "was 0)." % (crop_left_i, crop_right_i, width))

            result.append(
                _CropAndPadSamplingResult(
                    crop_top=crop_top_i,
                    crop_right=crop_right_i,
                    crop_bottom=crop_bottom_i,
                    crop_left=crop_left_i,
                    pad_top=pad_top[i],
                    pad_right=pad_right[i],
                    pad_bottom=pad_bottom[i],
                    pad_left=pad_left[i],
                    pad_mode=pad_mode[i],
                    pad_cval=pad_cval[i]))
        return result

    def get_parameters(self):
        return [self.all_sides, self.top, self.right, self.bottom, self.left,
                self.pad_mode, self.pad_cval]


class Pad(CropAndPad):
    """Pad images, i.e. adds columns/rows of pixels to them.

    dtype support::

        See ``imgaug.augmenters.size.CropAndPad``.

    Parameters
    ----------
    px : None or int or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to pad on each side of the image.
        Expected value range is ``[0, inf)``.
        Either this or the parameter `percent` may be set, not both at the same
        time.

            * If ``None``, then pixel-based padding will not be used.
            * If ``int``, then that exact number of pixels will always be
              padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left), unless `sample_independently` is set to ``False``,
              as then only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
              then each side will be padded by a random amount sampled
              uniformly per image and side from the inteval ``[a, b]``. If
              however `sample_independently` is set to ``False``, only one
              value will be sampled per image and used for all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``int`` (always
              pad by exactly that value), a ``tuple`` of two ``int`` s
              ``a`` and ``b`` (pad by an amount within ``[a, b]``), a
              ``list`` of ``int`` s (pad by a random value that is
              contained in the ``list``) or a ``StochasticParameter`` (sample
              the amount to pad from that parameter).

    percent : None or int or float or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to pad
        on each side of the image given as a *fraction* of the image
        height/width. E.g. if this is set to ``0.1``, the augmenter will
        always pad ``10%`` of the image's height at both the top and the
        bottom (both ``10%`` each), as well as ``10%`` of the width at the
        right and left.
        Expected value range is ``[0.0, inf)``.
        Either this or the parameter `px` may be set, not both
        at the same time.

            * If ``None``, then fraction-based padding will not be
              used.
            * If ``number``, then that fraction will always be padded.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left). If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``,
              then each side will be padded by a random fraction
              sampled uniformly per image and side from the interval
              ``[a, b]``. If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``float``
              (always pad by exactly that fraction), a ``tuple`` of
              two ``float`` s ``a`` and ``b`` (pad by a fraction from
              ``[a, b]``), a ``list`` of ``float`` s (pad by a random
              value that is contained in the list) or a ``StochasticParameter``
              (sample the percentage to pad from that parameter).

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        Padding mode to use. The available modes match the numpy padding modes,
        i.e. ``constant``, ``edge``, ``linear_ramp``, ``maximum``, ``median``,
        ``minimum``, ``reflect``, ``symmetric``, ``wrap``. The modes
        ``constant`` and ``linear_ramp`` use extra values, which are provided
        by ``pad_cval`` when necessary. See :func:`imgaug.imgaug.pad` for
        more details.

            * If ``imgaug.ALL``, then a random mode from all available modes
              will be sampled per image.
            * If a ``str``, it will be used as the pad mode for all images.
            * If a ``list`` of ``str``, a random one of these will be sampled
              per image and used as the mode.
            * If ``StochasticParameter``, a random mode will be sampled from
              this parameter per image.

    pad_cval : number or tuple of number list of number or imgaug.parameters.StochasticParameter, optional
        The constant value to use if the pad mode is ``constant`` or the end
        value to use if the mode is ``linear_ramp``.
        See :func:`imgaug.imgaug.pad` for more details.

            * If ``number``, then that value will be used.
            * If a ``tuple`` of two ``number`` s and at least one of them is
              a ``float``, then a random number will be uniformly sampled per
              image from the continuous interval ``[a, b]`` and used as the
              value. If both ``number`` s are ``int`` s, the interval is
              discrete.
            * If a ``list`` of ``number``, then a random value will be chosen
              from the elements of the ``list`` and used as the value.
            * If ``StochasticParameter``, a random value will be sampled from
              that parameter per image.

    keep_size : bool, optional
        After padding, the result image will usually have a
        different height/width compared to the original input image. If this
        parameter is set to ``True``, then the padded image will be
        resized to the input image's size, i.e. the augmenter's output shape
        is always identical to the input shape.

    sample_independently : bool, optional
        If ``False`` *and* the values for `px`/`percent` result in exactly
        *one* probability distribution for all image sides, only one single
        value will be sampled from that probability distribution and used for
        all sides. I.e. the pad amount then is the same for all sides.
        If ``True``, four values will be sampled independently, one per side.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Pad(px=(0, 10))

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding happens by
    zero-padding, i.e. it adds black pixels (default setting).

    >>> aug = iaa.Pad(px=(0, 10), pad_mode="edge")

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. The padding uses the
    ``edge`` mode from numpy's pad function, i.e. the pixel colors around
    the image sides are repeated.

    >>> aug = iaa.Pad(px=(0, 10), pad_mode=["constant", "edge"])

    Similar to the previous example, but uses zero-padding (``constant``) for
    half of the images and ``edge`` padding for the other half.

    >>> aug = iaa.Pad(px=(0, 10), pad_mode=ia.ALL, pad_cval=(0, 255))

    Similar to the previous example, but uses any available padding mode.
    In case the padding mode ends up being ``constant`` or ``linear_ramp``,
    and random intensity is uniformly sampled (once per image) from the
    discrete interval ``[0..255]`` and used as the intensity of the new
    pixels.

    >>> aug = iaa.Pad(px=(0, 10), sample_independently=False)

    Pad each side by a random pixel value sampled uniformly once per image
    from the discrete interval ``[0..10]``. Each sampled value is used
    for *all* sides of the corresponding image.

    >>> aug = iaa.Pad(px=(0, 10), keep_size=False)

    Pad each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. Afterwards, do **not**
    resize the padded image back to the input image's size. This will increase
    the image's height and width by a maximum of ``20`` pixels.

    >>> aug = iaa.Pad(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    Pad the top and bottom by a random pixel value sampled uniformly from the
    discrete interval ``[0..10]``. Pad the left and right analogously by
    a random value sampled from ``[0..5]``. Each value is always sampled
    independently.

    >>> aug = iaa.Pad(percent=(0, 0.1))

    Pad each side by a random fraction sampled uniformly from the continuous
    interval ``[0.0, 0.10]``. The fraction is sampled once per image and
    side. E.g. a sampled fraction of ``0.1`` for the top side would pad by
    ``0.1*H``, where ``H`` is the height of the input image.

    >>> aug = iaa.Pad(
    >>>     percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    Pads each side by either ``5%`` or ``10%``. The values are sampled
    once per side and image.

    """

    def __init__(self, px=None, percent=None, pad_mode="constant", pad_cval=0,
                 keep_size=True, sample_independently=True,
                 name=None, deterministic=False, random_state=None):
        def recursive_validate(v):
            if v is None:
                return v
            elif ia.is_single_number(v):
                assert v >= 0, "Expected value >0, got %.4f" % (v,)
                return v
            elif isinstance(v, iap.StochasticParameter):
                return v
            elif isinstance(v, tuple):
                return tuple([recursive_validate(v_) for v_ in v])
            elif isinstance(v, list):
                return [recursive_validate(v_) for v_ in v]
            else:
                raise Exception(
                    "Expected None or int or float or StochasticParameter or "
                    "list or tuple, got %s." % (type(v),))

        px = recursive_validate(px)
        percent = recursive_validate(percent)

        super(Pad, self).__init__(
            px=px,
            percent=percent,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            keep_size=keep_size,
            sample_independently=sample_independently,
            name=name,
            deterministic=deterministic,
            random_state=random_state
        )


class Crop(CropAndPad):
    """Crop images, i.e. remove columns/rows of pixels at the sides of images.

    This augmenter allows to extract smaller-sized subimages from given
    full-sized input images. The number of pixels to cut off may be defined
    in absolute values or as fractions of the image sizes.

    This augmenter will never crop images below a height or width of ``1``.

    dtype support::

        See ``imgaug.augmenters.size.CropAndPad``.

    Parameters
    ----------
    px : None or int or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop on each side of the image.
        Expected value range is ``[0, inf)``.
        Either this or the parameter `percent` may be set, not both at the same
        time.

            * If ``None``, then pixel-based cropping will not be used.
            * If ``int``, then that exact number of pixels will always be
              cropped.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left), unless `sample_independently` is set to ``False``,
              as then only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``int`` s with values ``a`` and ``b``,
              then each side will be cropped by a random amount sampled
              uniformly per image and side from the inteval ``[a, b]``. If
              however `sample_independently` is set to ``False``, only one
              value will be sampled per image and used for all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``int`` (always
              crop by exactly that value), a ``tuple`` of two ``int`` s
              ``a`` and ``b`` (crop by an amount within ``[a, b]``), a
              ``list`` of ``int`` s (crop by a random value that is
              contained in the ``list``) or a ``StochasticParameter`` (sample
              the amount to crop from that parameter).

    percent : None or int or float or imgaug.parameters.StochasticParameter or tuple, optional
        The number of pixels to crop
        on each side of the image given as a *fraction* of the image
        height/width. E.g. if this is set to ``0.1``, the augmenter will
        always crop ``10%`` of the image's height at both the top and the
        bottom (both ``10%`` each), as well as ``10%`` of the width at the
        right and left.
        Expected value range is ``[0.0, 1.0)``.
        Either this or the parameter `px` may be set, not both
        at the same time.

            * If ``None``, then fraction-based cropping will not be
              used.
            * If ``number``, then that fraction will always be cropped.
            * If ``StochasticParameter``, then that parameter will be used for
              each image. Four samples will be drawn per image (top, right,
              bottom, left). If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of two ``float`` s with values ``a`` and ``b``,
              then each side will be cropped by a random fraction
              sampled uniformly per image and side from the interval
              ``[a, b]``. If however `sample_independently` is set to
              ``False``, only one value will be sampled per image and used for
              all sides.
            * If a ``tuple`` of four entries, then the entries represent top,
              right, bottom, left. Each entry may be a single ``float``
              (always crop by exactly that fraction), a ``tuple`` of
              two ``float`` s ``a`` and ``b`` (crop by a fraction from
              ``[a, b]``), a ``list`` of ``float`` s (crop by a random
              value that is contained in the list) or a ``StochasticParameter``
              (sample the percentage to crop from that parameter).

    keep_size : bool, optional
        After cropping, the result image will usually have a
        different height/width compared to the original input image. If this
        parameter is set to ``True``, then the cropped image will be
        resized to the input image's size, i.e. the augmenter's output shape
        is always identical to the input shape.

    sample_independently : bool, optional
        If ``False`` *and* the values for `px`/`percent` result in exactly
        *one* probability distribution for all image sides, only one single
        value will be sampled from that probability distribution and used for
        all sides. I.e. the crop amount then is the same for all sides.
        If ``True``, four values will be sampled independently, one per side.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Crop(px=(0, 10))

    Crop each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``.

    >>> aug = iaa.Crop(px=(0, 10), sample_independently=False)

    Crop each side by a random pixel value sampled uniformly once per image
    from the discrete interval ``[0..10]``. Each sampled value is used
    for *all* sides of the corresponding image.

    >>> aug = iaa.Crop(px=(0, 10), keep_size=False)

    Crop each side by a random pixel value sampled uniformly per image and
    side from the discrete interval ``[0..10]``. Afterwards, do **not**
    resize the cropped image back to the input image's size. This will decrease
    the image's height and width by a maximum of ``20`` pixels.

    >>> aug = iaa.Crop(px=((0, 10), (0, 5), (0, 10), (0, 5)))

    Crop the top and bottom by a random pixel value sampled uniformly from the
    discrete interval ``[0..10]``. Crop the left and right analogously by
    a random value sampled from ``[0..5]``. Each value is always sampled
    independently.

    >>> aug = iaa.Crop(percent=(0, 0.1))

    Crop each side by a random fraction sampled uniformly from the continuous
    interval ``[0.0, 0.10]``. The fraction is sampled once per image and
    side. E.g. a sampled fraction of ``0.1`` for the top side would crop by
    ``0.1*H``, where ``H`` is the height of the input image.

    >>> aug = iaa.Crop(
    >>>     percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))

    Crops each side by either ``5%`` or ``10%``. The values are sampled
    once per side and image.

    """

    def __init__(self, px=None, percent=None, keep_size=True,
                 sample_independently=True,
                 name=None, deterministic=False, random_state=None):
        def recursive_negate(v):
            if v is None:
                return v
            elif ia.is_single_number(v):
                assert v >= 0, "Expected value >0, got %.4f." % (v,)
                return -v
            elif isinstance(v, iap.StochasticParameter):
                return iap.Multiply(v, -1)
            elif isinstance(v, tuple):
                return tuple([recursive_negate(v_) for v_ in v])
            elif isinstance(v, list):
                return [recursive_negate(v_) for v_ in v]
            else:
                raise Exception(
                    "Expected None or int or float or StochasticParameter or "
                    "list or tuple, got %s." % (type(v),))

        px = recursive_negate(px)
        percent = recursive_negate(percent)

        super(Crop, self).__init__(
            px=px,
            percent=percent,
            keep_size=keep_size,
            sample_independently=sample_independently,
            name=name,
            deterministic=deterministic,
            random_state=random_state
        )


# TODO maybe rename this to PadToMinimumSize?
# TODO this is very similar to CropAndPad, maybe add a way to generate crop
#      values imagewise via a callback in in CropAndPad?
# TODO why is padding mode and cval here called pad_mode, pad_cval but in other
#      cases mode/cval?
class PadToFixedSize(meta.Augmenter):
    """Pad images to a predefined minimum width and/or height.

    If images are already at the minimum width/height or are larger, they will
    not be padded. Note that this also means that images will not be cropped if
    they exceed the required width/height.

    The augmenter randomly decides per image how to distribute the required
    padding amounts over the image axis. E.g. if 2px have to be padded on the
    left or right to reach the required width, the augmenter will sometimes
    add 2px to the left and 0px to the right, sometimes add 2px to the right
    and 0px to the left and sometimes add 1px to both sides. Set `position`
    to ``center`` to prevent that.

    dtype support::

        See :func:`imgaug.imgaug.pad`.

    Parameters
    ----------
    width : int or None
        Pad images up to this minimum width.
        If ``None``, image widths will not be altered.

    height : int or None
        Pad images up to this minimum height.
        If ``None``, image heights will not be altered.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.CropAndPad.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.CropAndPad.__init__`.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        Sets the center point of the padding, which determines how the
        required padding amounts are distributed to each side. For a ``tuple``
        ``(a, b)``, both ``a`` and ``b`` are expected to be in range
        ``[0.0, 1.0]`` and describe the fraction of padding applied to the
        left/right (low/high values for ``a``) and the fraction of padding
        applied to the top/bottom (low/high values for ``b``). A padding
        position at ``(0.5, 0.5)`` would be the center of the image and
        distribute the padding equally to all sides. A padding position at
        ``(0.0, 1.0)`` would be the left-bottom and would apply 100% of the
        required padding to the bottom and left sides of the image so that
        the bottom left corner becomes more and more the new image
        center (depending on how much is padded).

            * If string ``uniform`` then the share of padding is randomly and
              uniformly distributed over each side.
              Equivalent to ``(Uniform(0.0, 1.0), Uniform(0.0, 1.0))``.
            * If string ``normal`` then the share of padding is distributed
              based on a normal distribution, leading to a focus on the
              center of the images.
              Equivalent to
              ``(Clip(Normal(0.5, 0.45/2), 0, 1),
              Clip(Normal(0.5, 0.45/2), 0, 1))``.
            * If string ``center`` then center point of the padding is
              identical to the image center.
              Equivalent to ``(0.5, 0.5)``.
            * If a string matching regex
              ``^(left|center|right)-(top|center|bottom)$``, e.g. ``left-top``
              or ``center-bottom`` then sets the center point of the padding
              to the X-Y position matching that description.
            * If a tuple of float, then expected to have exactly two entries
              between ``0.0`` and ``1.0``, which will always be used as the
              combination the position matching (x, y) form.
            * If a ``StochasticParameter``, then that parameter will be queried
              once per call to ``augment_*()`` to get ``Nx2`` center positions
              in ``(x, y)`` form (with ``N`` the number of images).
            * If a ``tuple`` of ``StochasticParameter``, then expected to have
              exactly two entries that will both be queried per call to
              ``augment_*()``, each for ``(N,)`` values, to get the center
              positions. First parameter is used for ``x`` coordinates,
              second for ``y`` coordinates.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.PadToFixedSize(width=100, height=100)

    For image sides smaller than ``100`` pixels, pad to ``100`` pixels. Do
    nothing for the other edges. The padding is randomly (uniformly)
    distributed over the sides, so that e.g. sometimes most of the required
    padding is applied to the left, sometimes to the right (analogous
    top/bottom).

    >>> aug = iaa.PadToFixedSize(width=100, height=100, position="center")

    For image sides smaller than ``100`` pixels, pad to ``100`` pixels. Do
    nothing for the other image sides. The padding is always equally
    distributed over the left/right and top/bottom sides.

    >>> aug = iaa.PadToFixedSize(width=100, height=100, pad_mode=ia.ALL)

    For image sides smaller than ``100`` pixels, pad to ``100`` pixels and
    use any possible padding mode for that. Do nothing for the other image
    sides. The padding is always equally distributed over the left/right and
    top/bottom sides.

    >>> aug = iaa.Sequential([
    >>>     iaa.PadToFixedSize(width=100, height=100),
    >>>     iaa.CropToFixedSize(width=100, height=100)
    >>> ])

    Pad images smaller than ``100x100`` until they reach ``100x100``.
    Analogously, crop images larger than ``100x100`` until they reach
    ``100x100``. The output images therefore have a fixed size of ``100x100``.

    """

    def __init__(self, width, height, pad_mode="constant", pad_cval=0,
                 position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(PadToFixedSize, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        self.size = (width, height)

        # Position of where to pad. The further to the top left this is, the
        # larger the share of pixels that will be added to the top and left
        # sides. I.e. set to (Deterministic(0.0), Deterministic(0.0)) to only
        # add at the top and left, (Deterministic(1.0), Deterministic(1.0))
        # to only add at the bottom right. Analogously (0.5, 0.5) pads equally
        # on both axis, (0.0, 1.0) pads left and bottom, (1.0, 0.0) pads right
        # and top.
        self.position = _handle_position_parameter(position)

        self.pad_mode = _handle_pad_mode_param(pad_mode)
        # TODO enable ALL here like in eg Affine
        self.pad_cval = iap.handle_discrete_param(
            pad_cval, "pad_cval", value_range=None, tuple_to_uniform=True,
            list_to_choice=True, allow_floats=True)

        # set these to None to use the same values as sampled for the
        # images (not tested)
        self._pad_mode_heatmaps = "constant"
        self._pad_mode_segmentation_maps = "constant"
        self._pad_cval_heatmaps = 0.0
        self._pad_cval_segmentation_maps = 0

    def _augment_batch(self, batch, random_state, parents, hooks):
        # Providing the whole batch to _draw_samples() would not be necessary
        # for this augmenter. The number of rows would be sufficient. This
        # formulation however enables derived augmenters to use rowwise shapes
        # without having to compute them here for this augmenter.
        samples = self._draw_samples(batch, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images,
                                                           samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps, samples, self._pad_mode_heatmaps,
                self._pad_cval_heatmaps)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, samples, self._pad_mode_heatmaps,
                self._pad_cval_heatmaps)

        for augm_name in ["keypoints", "bounding_boxes", "polygons",
                          "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(
                    self._augment_keypoints_by_samples,
                    samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    def _augment_images_by_samples(self, images, samples):
        result = []
        sizes, pad_xs, pad_ys, pad_modes, pad_cvals = samples
        for i, (image, size) in enumerate(zip(images, sizes)):
            width_min, height_min = size
            height_image, width_image = image.shape[:2]
            paddings = self._calculate_paddings(height_image, width_image,
                                                height_min, width_min,
                                                pad_xs[i], pad_ys[i])

            image = _crop_and_pad_arr(
                image, (0, 0, 0, 0), paddings, pad_modes[i], pad_cvals[i],
                keep_size=False)

            result.append(image)

        # TODO result is always a list. Should this be converted to an array
        #      if possible (not guaranteed that all images have same size,
        #      some might have been larger than desired height/width)
        return result

    def _augment_keypoints_by_samples(self, keypoints_on_images, samples):
        result = []
        sizes, pad_xs, pad_ys, _, _ = samples
        for i, (kpsoi, size) in enumerate(zip(keypoints_on_images, sizes)):
            width_min, height_min = size
            height_image, width_image = kpsoi.shape[:2]
            paddings_img = self._calculate_paddings(height_image, width_image,
                                                    height_min, width_min,
                                                    pad_xs[i], pad_ys[i])

            keypoints_padded = _crop_and_pad_kpsoi(
                kpsoi, (0, 0, 0, 0), paddings_img,
                keep_size=False)

            result.append(keypoints_padded)

        return result

    def _augment_maps_by_samples(self, augmentables, samples, pad_mode,
                                 pad_cval):
        sizes, pad_xs, pad_ys, pad_modes, pad_cvals = samples

        for i, (augmentable, size) in enumerate(zip(augmentables, sizes)):
            width_min, height_min = size
            height_img, width_img = augmentable.shape[:2]
            paddings_img = self._calculate_paddings(
                height_img, width_img, height_min, width_min,
                pad_xs[i], pad_ys[i])

            # TODO for the previous method (and likely the new/current one
            #      too):
            #      for 30x30 padded to 32x32 with 15x15 heatmaps this results
            #      in paddings of 1 on each side (assuming
            #      position=(0.5, 0.5)) giving 17x17 heatmaps when they should
            #      be 16x16. Error is due to each side getting projected 0.5
            #      padding which is rounded to 1. This doesn't seem right.
            augmentables[i] = _crop_and_pad_hms_or_segmaps_(
                augmentables[i],
                (0, 0, 0, 0),
                paddings_img,
                pad_mode=pad_mode if pad_mode is not None else pad_modes[i],
                pad_cval=pad_cval if pad_cval is not None else pad_cvals[i],
                keep_size=False)

        return augmentables

    def _draw_samples(self, batch, random_state):
        nb_images = batch.nb_rows
        rngs = random_state.duplicate(4)

        if isinstance(self.position, tuple):
            pad_xs = self.position[0].draw_samples(nb_images,
                                                   random_state=rngs[0])
            pad_ys = self.position[1].draw_samples(nb_images,
                                                   random_state=rngs[1])
        else:
            pads = self.position.draw_samples((nb_images, 2),
                                              random_state=rngs[0])
            pad_xs = pads[:, 0]
            pad_ys = pads[:, 1]

        pad_modes = self.pad_mode.draw_samples(nb_images,
                                               random_state=rngs[2])
        pad_cvals = self.pad_cval.draw_samples(nb_images,
                                               random_state=rngs[3])

        # We return here the sizes even though they are static as it allows
        # derived augmenters to define image-specific heights/widths.
        return [self.size] * nb_images, pad_xs, pad_ys, pad_modes, pad_cvals

    @classmethod
    def _calculate_paddings(cls, height_image, width_image,
                            height_min, width_min, pad_xs_i, pad_ys_i):
        pad_top = 0
        pad_right = 0
        pad_bottom = 0
        pad_left = 0

        if width_min is not None and width_image < width_min:
            pad_total_x = width_min - width_image
            pad_left = int((1-pad_xs_i) * pad_total_x)
            pad_right = pad_total_x - pad_left

        if height_min is not None and height_image < height_min:
            pad_total_y = height_min - height_image
            pad_top = int((1-pad_ys_i) * pad_total_y)
            pad_bottom = pad_total_y - pad_top

        return pad_top, pad_right, pad_bottom, pad_left

    def get_parameters(self):
        return [self.size[0], self.size[1], self.pad_mode, self.pad_cval,
                self.position]


class CenterPadToFixedSize(PadToFixedSize):
    """Pad images equally on all sides up to given minimum heights/widths.

    This is an alias for :class:`PadToFixedSize` with ``position="center"`.
    It spreads the pad amounts equally over all image sides, while
    :class:`PadToFixedSize` by defaults spreads them randomly.

    dtype support::

        See :class:`imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width : int or None
        See :func:`PadToFixedSize.__init__`.

    height : int or None
        See :func:`PadToFixedSize.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange(7*7*3).astype(np.uint8).reshape((7, 7, 3))
    >>> image_padded = iaa.CenterPadToFixedSize(height=8, width=9)(image=image)

    Pad an image up to ``8x9``, adding ``0`` rows at the top, ``1`` at the
    bottom, ``1`` at the right and ``1`` at the left.

    """

    def __init__(self, width, height, pad_mode="constant", pad_cval=0,
                 name=None, deterministic=False, random_state=None):
        super(CenterPadToFixedSize, self).__init__(
            width=width, height=height, pad_mode=pad_mode, pad_cval=pad_cval,
            position="center",
            name=name, deterministic=deterministic, random_state=random_state)


# TODO maybe rename this to CropToMaximumSize ?
# TODO this is very similar to CropAndPad, maybe add a way to generate crop
#      values imagewise via a callback in in CropAndPad?
# TODO add crop() function in imgaug, similar to pad
class CropToFixedSize(meta.Augmenter):
    """Crop images down to a predefined maximum width and/or height.

    If images are already at the maximum width/height or are smaller, they
    will not be cropped. Note that this also means that images will not be
    padded if they are below the required width/height.

    The augmenter randomly decides per image how to distribute the required
    cropping amounts over the image axis. E.g. if 2px have to be cropped on
    the left or right to reach the required width, the augmenter will
    sometimes remove 2px from the left and 0px from the right, sometimes
    remove 2px from the right and 0px from the left and sometimes remove 1px
    from both sides. Set `position` to ``center`` to prevent that.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    width : int or None
        Crop images down to this maximum width.
        If ``None``, image widths will not be altered.

    height : int or None
        Crop images down to this maximum height.
        If ``None``, image heights will not be altered.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
         Sets the center point of the cropping, which determines how the
         required cropping amounts are distributed to each side. For a
         ``tuple`` ``(a, b)``, both ``a`` and ``b`` are expected to be in
         range ``[0.0, 1.0]`` and describe the fraction of cropping applied
         to the left/right (low/high values for ``a``) and the fraction
         of cropping applied to the top/bottom (low/high values for ``b``).
         A cropping position at ``(0.5, 0.5)`` would be the center of the
         image and distribute the cropping equally over all sides. A cropping
         position at ``(1.0, 0.0)`` would be the right-top and would apply
         100% of the required cropping to the right and top sides of the image.

            * If string ``uniform`` then the share of cropping is randomly
              and uniformly distributed over each side.
              Equivalent to ``(Uniform(0.0, 1.0), Uniform(0.0, 1.0))``.
            * If string ``normal`` then the share of cropping is distributed
              based on a normal distribution, leading to a focus on the center
              of the images.
              Equivalent to
              ``(Clip(Normal(0.5, 0.45/2), 0, 1),
              Clip(Normal(0.5, 0.45/2), 0, 1))``.
            * If string ``center`` then center point of the cropping is
              identical to the image center.
              Equivalent to ``(0.5, 0.5)``.
            * If a string matching regex
              ``^(left|center|right)-(top|center|bottom)$``, e.g.
              ``left-top`` or ``center-bottom`` then sets the center point of
              the cropping to the X-Y position matching that description.
            * If a tuple of float, then expected to have exactly two entries
              between ``0.0`` and ``1.0``, which will always be used as the
              combination the position matching (x, y) form.
            * If a ``StochasticParameter``, then that parameter will be queried
              once per call to ``augment_*()`` to get ``Nx2`` center positions
              in ``(x, y)`` form (with ``N`` the number of images).
            * If a ``tuple`` of ``StochasticParameter``, then expected to have
              exactly two entries that will both be queried per call to
              ``augment_*()``, each for ``(N,)`` values, to get the center
              positions. First parameter is used for ``x`` coordinates,
              second for ``y`` coordinates.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CropToFixedSize(width=100, height=100)

    For image sides larger than ``100`` pixels, crop to ``100`` pixels. Do
    nothing for the other sides. The cropping amounts are randomly (and
    uniformly) distributed over the sides of the image.

    >>> aug = iaa.CropToFixedSize(width=100, height=100, position="center")

    For sides larger than ``100`` pixels, crop to ``100`` pixels. Do nothing
    for the other sides. The cropping amounts are always equally distributed
    over the left/right sides of the image (and analogously for top/bottom).

    >>> aug = iaa.Sequential([
    >>>     iaa.PadToFixedSize(width=100, height=100),
    >>>     iaa.CropToFixedSize(width=100, height=100)
    >>> ])

    Pad images smaller than ``100x100`` until they reach ``100x100``.
    Analogously, crop images larger than ``100x100`` until they reach
    ``100x100``. The output images therefore have a fixed size of ``100x100``.

    """

    def __init__(self, width, height, position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(CropToFixedSize, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        self.size = (width, height)

        # Position of where to crop. The further to the top left this is,
        # the larger the share of pixels that will be cropped from the top
        # and left sides. I.e. set to (Deterministic(0.0), Deterministic(0.0))
        # to only crop at the top and left,
        # (Deterministic(1.0), Deterministic(1.0)) to only crop at the bottom
        # right. Analogously (0.5, 0.5) crops equally on both axis,
        # (0.0, 1.0) crops left and bottom, (1.0, 0.0) crops right and top.
        self.position = _handle_position_parameter(position)

    def _augment_batch(self, batch, random_state, parents, hooks):
        # Providing the whole batch to _draw_samples() would not be necessary
        # for this augmenter. The number of rows would be sufficient. This
        # formulation however enables derived augmenters to use rowwise shapes
        # without having to compute them here for this augmenter.
        samples = self._draw_samples(batch, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images,
                                                           samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps, samples)

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps, samples)

        for augm_name in ["keypoints", "bounding_boxes", "polygons",
                          "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                func = functools.partial(
                    self._augment_keypoints_by_samples,
                    samples=samples)
                cbaois = self._apply_to_cbaois_as_keypoints(augm_value, func)
                setattr(batch, augm_name, cbaois)

        return batch

    def _augment_images_by_samples(self, images, samples):
        result = []
        sizes, offset_xs, offset_ys = samples
        for i, (image, size) in enumerate(zip(images, sizes)):
            w, h = size
            height_image, width_image = image.shape[0:2]

            croppings = self._calculate_crop_amounts(
                height_image, width_image, h, w, offset_ys[i], offset_xs[i])

            image_cropped = _crop_and_pad_arr(image, croppings, (0, 0, 0, 0),
                                              keep_size=False)

            result.append(image_cropped)

        return result

    def _augment_keypoints_by_samples(self, kpsois, samples):
        result = []
        sizes, offset_xs, offset_ys = samples
        for i, (kpsoi, size) in enumerate(zip(kpsois, sizes)):
            w, h = size
            height_image, width_image = kpsoi.shape[0:2]

            croppings_img = self._calculate_crop_amounts(
                height_image, width_image, h, w, offset_ys[i], offset_xs[i])

            kpsoi_cropped = _crop_and_pad_kpsoi(
                kpsoi, croppings_img, (0, 0, 0, 0), keep_size=False)

            result.append(kpsoi_cropped)

        return result

    def _augment_maps_by_samples(self, augmentables, samples):
        sizes, offset_xs, offset_ys = samples
        for i, (augmentable, size) in enumerate(zip(augmentables, sizes)):
            w, h = size
            height_image, width_image = augmentable.shape[0:2]

            croppings_img = self._calculate_crop_amounts(
                height_image, width_image, h, w, offset_ys[i], offset_xs[i])

            augmentables[i] = _crop_and_pad_hms_or_segmaps_(
                augmentable, croppings_img, (0, 0, 0, 0), keep_size=False)

        return augmentables

    @classmethod
    def _calculate_crop_amounts(cls, height_image, width_image,
                                height_max, width_max,
                                offset_y, offset_x):
        crop_top = 0
        crop_right = 0
        crop_bottom = 0
        crop_left = 0

        if height_max is not None and height_image > height_max:
            crop_top = int(offset_y * (height_image - height_max))
            crop_bottom = height_image - height_max - crop_top

        if width_max is not None and width_image > width_max:
            crop_left = int(offset_x * (width_image - width_max))
            crop_right = width_image - width_max - crop_left

        return crop_top, crop_right, crop_bottom, crop_left

    def _draw_samples(self, batch, random_state):
        nb_images = batch.nb_rows
        rngs = random_state.duplicate(2)

        if isinstance(self.position, tuple):
            offset_xs = self.position[0].draw_samples(nb_images,
                                                      random_state=rngs[0])
            offset_ys = self.position[1].draw_samples(nb_images,
                                                      random_state=rngs[1])
        else:
            offsets = self.position.draw_samples((nb_images, 2),
                                                 random_state=rngs[0])
            offset_xs = offsets[:, 0]
            offset_ys = offsets[:, 1]

        offset_xs = 1.0 - offset_xs
        offset_ys = 1.0 - offset_ys

        # We return here the sizes even though they are static as it allows
        # derived augmenters to define image-specific heights/widths.
        return [self.size] * nb_images, offset_xs, offset_ys

    def get_parameters(self):
        return [self.size[0], self.size[1], self.position]


class CenterCropToFixedSize(CropToFixedSize):
    """Take a crop from the center of each image.

    This is an alias for :class:`CropToFixedSize` with ``position="center"`.

    .. note ::

        If images already have a width and/or height below the provided
        width and/or height then this augmenter will do nothing for the
        respective axis. Hence, resulting images can be smaller than the
        provided axis sizes.

    dtype support::

        See :class:`imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width : int or None
        See :func:`CropToFixedSize.__init__`.

    height : int or None
        See :func:`CropToFixedSize.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange(7*7*3).astype(np.uint8).reshape((7, 7, 3))
    >>> crop = iaa.CenterCropToFixedSize(height=3, width=3)(image=image)

    Take a ``3x3`` sized crop from the center of an ``7x7`` sized image.

    """

    def __init__(self, width, height,
                 name=None, deterministic=False, random_state=None):
        super(CenterCropToFixedSize, self).__init__(
            width=width, height=height, position="center",
            name=name, deterministic=deterministic, random_state=random_state)


class CropToMultiplesOf(CropToFixedSize):
    """Crop images down until their height/width is a multiple of a value.

    .. note ::

        For a given axis size ``A`` and multiple ``M``, if ``A`` is in the
        interval ``[0 .. M]``, the axis will not be changed.
        As a result, this augmenter can still produce axis sizes that are
        not multiples of the given values.

    dtype support::

        See :class:`imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        Multiple for the width. Images will be cropped down until their
        width is a multiple of this value.
        If ``None``, image widths will not be altered.

    height_multiple : int or None
        Multiple for the height. Images will be cropped down until their
        height is a multiple of this value.
        If ``None``, image heights will not be altered.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`CropToFixedSize.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((13*12)).astype(np.uint8).reshape((13, 12))
    >>> aug = iaa.CropToMultiplesOf(height_multiple=10, width_multiple=6)
    >>> crop = aug(image=image)

    Crop ``image`` down to size ``10x12`` (multiples of ``10`` and ``6``).
    The rows to be cropped will be randomly spread over the top and bottom
    sides (analogous for the left/right sides).

    """

    def __init__(self, width_multiple, height_multiple, position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(CropToMultiplesOf, self).__init__(
            width=None, height=None, position=position,
            name=name, deterministic=deterministic, random_state=random_state)
        self.width_multiple = width_multiple
        self.height_multiple = height_multiple

    def _draw_samples(self, batch, random_state):
        _sizes, offset_xs, offset_ys = super(
            CropToMultiplesOf, self
        )._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]
            croppings = ia.compute_croppings_to_reach_multiples_of(
                shape,
                height_multiple=self.height_multiple,
                width_multiple=self.width_multiple)

            # TODO change that
            # note that these are not in the same order as shape tuples
            # in CropToFixedSize
            new_size = (
                width - croppings[1] - croppings[3],
                height - croppings[0] - croppings[2]
            )
            sizes.append(new_size)

        return sizes, offset_xs, offset_ys

    def get_parameters(self):
        return [self.width_multiple, self.height_multiple, self.position]


class CenterCropToMultiplesOf(CropToMultiplesOf):
    """Crop images equally on all sides until H/W are multiples of given values.

    This is the same as :class`CropToMultiplesOf`, but uses
    ``position="center"`` by default, which spreads the crop amounts equally
    over all image sides, while :class`CropToMultiplesOf` by default spreads
    them randomly.

    dtype support::

        See :class:`imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        See :func:`CropToMultiplesOf.__init__`.

    height_multiple : int or None
        See :func:`CropToMultiplesOf.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((13*12)).astype(np.uint8).reshape((13, 14))
    >>> aug = iaa.CenterCropToMultiplesOf(height_multiple=10, width_multiple=6)
    >>> crop = aug(image=image)

    Crop ``image`` down to size ``10x12`` (multiples of ``10`` and ``6``).
    The operation will remove ``2`` rows from the bottom, ``1`` from the top,
    ``1`` column from the left and ``1`` from the right.

    """

    def __init__(self, width_multiple, height_multiple,
                 name=None, deterministic=False, random_state=None):
        super(CenterCropToMultiplesOf, self).__init__(
            width_multiple=width_multiple,
            height_multiple=height_multiple,
            name=name, deterministic=deterministic, random_state=random_state)


class PadToMultiplesOf(PadToFixedSize):
    """Pad images until their height/width is a multiple of a value.

    dtype support::

        See :class:`imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        Multiple for the width. Images will be padded until their
        width is a multiple of this value.
        If ``None``, image widths will not be altered.

    height_multiple : int or None
        Multiple for the height. Images will be padded until their
        height is a multiple of this value.
        If ``None``, image heights will not be altered.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToFixedSize.__init__`.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((13*12)).astype(np.uint8).reshape((13, 12))
    >>> aug = iaa.PadToMultiplesOf(height_multiple=10, width_multiple=6)
    >>> image_padded = aug(image=image)

    Pad ``image`` to size ``20x12`` (multiples of ``10`` and ``6``).

    """

    def __init__(self, width_multiple, height_multiple,
                 pad_mode="constant", pad_cval=0,
                 position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(PadToMultiplesOf, self).__init__(
            width=None, height=None, pad_mode=pad_mode, pad_cval=pad_cval,
            position=position,
            name=name, deterministic=deterministic, random_state=random_state)
        self.width_multiple = width_multiple
        self.height_multiple = height_multiple

    def _draw_samples(self, batch, random_state):
        _sizes, pad_xs, pad_ys, pad_modes, pad_cvals = super(
            PadToMultiplesOf, self
        )._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]
            paddings = ia.compute_paddings_to_reach_multiples_of(
                shape,
                height_multiple=self.height_multiple,
                width_multiple=self.width_multiple)

            # TODO change that
            # note that these are not in the same order as shape tuples
            # in PadToFixedSize
            new_size = (
                width + paddings[1] + paddings[3],
                height + paddings[0] + paddings[2]
            )
            sizes.append(new_size)

        return sizes, pad_xs, pad_ys, pad_modes, pad_cvals

    def get_parameters(self):
        return [self.width_multiple, self.height_multiple,
                self.pad_mode, self.pad_cval,
                self.position]


class CenterPadToMultiplesOf(PadToMultiplesOf):
    """Pad images equally on all sides until H/W are multiples of given values.

    This is the same as :class`PadToMultiplesOf`, but uses
    ``position="center"`` by default, which spreads the pad amounts equally
    over all image sides, while :class`PadToMultiplesOf` by default spreads
    them randomly.

    dtype support::

        See :class:`imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_multiple : int or None
        See :func:`PadToMultiplesOf.__init__`.

    height_multiple : int or None
        See :func:`PadToMultiplesOf.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToMultiplesOf.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToMultiplesOf.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((13*14)).astype(np.uint8).reshape((13, 14))
    >>> aug = iaa.CenterPadToMultiplesOf(height_multiple=10, width_multiple=6)
    >>> image_padded = aug(image=image)

    Pad ``image`` to size ``20x12`` (multiples of ``10`` and ``6``).
    The operation will pad ``4`` rows to the bottom, ``3`` to the top,
    ``1`` column to the left and ``1`` to the right.

    """

    def __init__(self, width_multiple, height_multiple,
                 pad_mode="constant", pad_cval=0,
                 position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(CenterPadToMultiplesOf, self).__init__(
            width_multiple=width_multiple,
            height_multiple=height_multiple,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            position=position,
            name=name, deterministic=deterministic, random_state=random_state)


class CropToExponentsOf(CropToFixedSize):
    """Crop images until their height/width is an exponent of a base.

    This augmenter removes pixels from an axis with size ``S`` leading to the
    new size ``S'`` until ``S' = B^E`` is fullfilled, where ``B`` is a
    provided base (e.g. ``2``) and ``E`` is an exponent from the discrete
    interval ``[1 .. inf)``.

    .. note ::

        This augmenter does nothing for axes with size less than ``B^1 = B``.
        If you have images with ``S < B^1``, it is recommended
        to combine this augmenter with a padding augmenter that pads each
        axis up to ``B``.

    dtype support::

        See :class:`imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        Base for the width. Images will be cropped down until their
        width fullfills ``width' = width_base ^ E`` with ``E`` being any
        natural number.
        If ``None``, image widths will not be altered.

    height_base : int or None
        Base for the height. Images will be cropped down until their
        height fullfills ``height' = height_base ^ E`` with ``E`` being any
        natural number.
        If ``None``, image heights will not be altered.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`CropToFixedSize.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((14*12)).astype(np.uint8).reshape((14, 12))
    >>> aug = iaa.CropToExponentsOf(height_base=3, width_base=2)
    >>> crop = aug(image=image)

    Crop ``image`` down to size ``9x8`` (``3^2`` and ``2^3``).

    """

    def __init__(self, width_base, height_base, position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(CropToExponentsOf, self).__init__(
            width=None, height=None, position=position,
            name=name, deterministic=deterministic, random_state=random_state)
        self.width_base = width_base
        self.height_base = height_base

    def _draw_samples(self, batch, random_state):
        _sizes, offset_xs, offset_ys = super(
            CropToExponentsOf, self
        )._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]
            croppings = ia.compute_croppings_to_reach_exponents_of(
                shape,
                height_base=self.height_base,
                width_base=self.width_base)

            # TODO change that
            # note that these are not in the same order as shape tuples
            # in CropToFixedSize
            new_size = (
                width - croppings[1] - croppings[3],
                height - croppings[0] - croppings[2]
            )
            sizes.append(new_size)

        return sizes, offset_xs, offset_ys

    def get_parameters(self):
        return [self.width_base, self.height_base, self.position]


class CenterCropToExponentsOf(CropToExponentsOf):
    """Crop images equally on all sides until H/W is an exponent of a base.

    This is the same as :class`CropToExponentsOf`, but uses
    ``position="center"`` by default, which spreads the crop amounts equally
    over all image sides, while :class`CropToExponentsOf` by default spreads
    them randomly.

    dtype support::

        See :class:`imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        See :func:`CropToExponentsOf.__init__`.

    height_base : int or None
        See :func:`CropToExponentsOf.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((14*12)).astype(np.uint8).reshape((14, 12))
    >>> aug = iaa.CenterCropToExponentsOf(height_base=3, width_base=2)
    >>> crop = aug(image=image)

    Crop ``image`` down to size ``9x8`` (``3^2`` and ``2^3``).
    This operation will remove ``2`` rows from the top, ``3`` from the bottom,
    ``2`` columns from the left and ``2`` from the right.

    """

    def __init__(self, width_base, height_base,
                 name=None, deterministic=False, random_state=None):
        super(CenterCropToExponentsOf, self).__init__(
            width_base=width_base, height_base=height_base, position="center",
            name=name, deterministic=deterministic, random_state=random_state)


class PadToExponentsOf(PadToFixedSize):
    """Pad images until their height/width is an exponent of a base.

    This augmenter adds pixels to an axis with size ``S`` leading to the
    new size ``S'`` until ``S' = B^E`` is fullfilled, where ``B`` is a
    provided base (e.g. ``2``) and ``E`` is an exponent from the discrete
    interval ``[1 .. inf)``.

    dtype support::

        See :class:`imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        Base for the width. Images will be padded down until their
        width fullfills ``width' = width_base ^ E`` with ``E`` being any
        natural number.
        If ``None``, image widths will not be altered.

    height_base : int or None
        Base for the height. Images will be padded until their
        height fullfills ``height' = height_base ^ E`` with ``E`` being any
        natural number.
        If ``None``, image heights will not be altered.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToFixedSize.__init__`.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((14*12)).astype(np.uint8).reshape((14, 12))
    >>> aug = iaa.PadToExponentsOf(height_base=5, width_base=2)
    >>> image_padded = aug(image=image)

    Pad ``image`` to size ``25x16`` (``5^2`` and ``2^3``).

    """

    def __init__(self, width_base, height_base,
                 pad_mode="constant", pad_cval=0,
                 position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(PadToExponentsOf, self).__init__(
            width=None, height=None, pad_mode=pad_mode, pad_cval=pad_cval,
            position=position,
            name=name, deterministic=deterministic, random_state=random_state)
        self.width_base = width_base
        self.height_base = height_base

    def _draw_samples(self, batch, random_state):
        _sizes, pad_xs, pad_ys, pad_modes, pad_cvals = super(
            PadToExponentsOf, self
        )._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]
            paddings = ia.compute_paddings_to_reach_exponents_of(
                shape,
                height_base=self.height_base,
                width_base=self.width_base)

            # TODO change that
            # note that these are not in the same order as shape tuples
            # in PadToFixedSize
            new_size = (
                width + paddings[1] + paddings[3],
                height + paddings[0] + paddings[2]
            )
            sizes.append(new_size)

        return sizes, pad_xs, pad_ys, pad_modes, pad_cvals

    def get_parameters(self):
        return [self.width_base, self.height_base,
                self.pad_mode, self.pad_cval,
                self.position]


class CenterPadToExponentsOf(PadToExponentsOf):
    """Pad images equally on all sides until H/W is an exponent of a base.

    This is the same as :class`PadToExponentsOf`, but uses
    ``position="center"`` by default, which spreads the pad amounts equally
    over all image sides, while :class`PadToExponentsOf` by default spreads
    them randomly.

    dtype support::

        See :class:`imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    width_base : int or None
        See :func:`PadToExponentsOf.__init__`.

    height_base : int or None
        See :func:`PadToExponentsOf.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToExponentsOf.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToExponentsOf.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((14*12)).astype(np.uint8).reshape((14, 12))
    >>> aug = iaa.CenterPadToExponentsOf(height_base=5, width_base=2)
    >>> image_padded = aug(image=image)

    Pad ``image`` to size ``25x16`` (``5^2`` and ``2^3``).
    This operation will add ``5`` rows at the top, ``6`` at the bottom,
    ``2`` columns at the left and ``2`` at the right.

    """

    def __init__(self, width_base, height_base,
                 pad_mode="constant", pad_cval=0,
                 name=None, deterministic=False, random_state=None):
        super(CenterPadToExponentsOf, self).__init__(
            width_base=width_base, height_base=height_base,
            pad_mode=pad_mode, pad_cval=pad_cval,
            position="center",
            name=name, deterministic=deterministic, random_state=random_state)


class CropToAspectRatio(CropToFixedSize):
    """Crop images until their width/height matches an aspect ratio.

    This augmenter removes either rows or columns until the image reaches
    the desired aspect ratio given in ``width / height``. The cropping
    operation is stopped once the desired aspect ratio is reached or the image
    side to crop reaches a size of ``1``. If any side of the image starts
    with a size of ``0``, the image will not be changed.

    dtype support::

        See :class:`imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        The desired aspect ratio, given as ``width/height``. E.g. a ratio
        of ``2.0`` denotes an image that is twice as wide as it is high.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`CropToFixedSize.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((14*12)).astype(np.uint8).reshape((14, 12))
    >>> aug = iaa.CropToAspectRatio(2.0)
    >>> crop = aug(image=image)

    Crop ``image`` down to size ``6x12``.

    """

    def __init__(self, aspect_ratio, position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(CropToAspectRatio, self).__init__(
            width=None, height=None, position=position,
            name=name, deterministic=deterministic, random_state=random_state)
        self.aspect_ratio = aspect_ratio

    def _draw_samples(self, batch, random_state):
        _sizes, offset_xs, offset_ys = super(
            CropToAspectRatio, self
        )._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]

            if height == 0 or width == 0:
                croppings = (0, 0, 0, 0)
            else:
                croppings = ia.compute_croppings_for_aspect_ratio(
                    shape,
                    aspect_ratio=self.aspect_ratio)

            # TODO change that
            # note that these are not in the same order as shape tuples
            # in CropToFixedSize
            new_size = (
                width - croppings[1] - croppings[3],
                height - croppings[0] - croppings[2]
            )
            sizes.append(new_size)

        return sizes, offset_xs, offset_ys

    def get_parameters(self):
        return [self.aspect_ratio, self.position]


class CenterCropToAspectRatio(CropToAspectRatio):
    """Crop images equally on all sides until they reach an aspect ratio.

    This is the same as :class`CropToAspectRatio`, but uses
    ``position="center"`` by default, which spreads the crop amounts equally
    over all image sides, while :class`CropToAspectRatio` by default spreads
    them randomly.

    dtype support::

        See :class:`imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        See :func:`CropToAspectRatio.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((15*12)).astype(np.uint8).reshape((15, 12))
    >>> aug = iaa.CenterCropToAspectRatio(1.0)
    >>> crop = aug(image=image)

    Crop ``image`` down to size ``12x12``.
    This operation will remove ``1`` row from the top and ``2`` rows from the
    bottom.

    """

    def __init__(self, aspect_ratio,
                 name=None, deterministic=False, random_state=None):
        super(CenterCropToAspectRatio, self).__init__(
            aspect_ratio=aspect_ratio, position="center",
            name=name, deterministic=deterministic, random_state=random_state)


class PadToAspectRatio(PadToFixedSize):
    """Pad images until their width/height matches an aspect ratio.

    This augmenter adds either rows or columns until the image reaches
    the desired aspect ratio given in ``width / height``.

    dtype support::

        See :class:`imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        The desired aspect ratio, given as ``width/height``. E.g. a ratio
        of ``2.0`` denotes an image that is twice as wide as it is high.

    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToFixedSize.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((14*12)).astype(np.uint8).reshape((14, 12))
    >>> aug = iaa.PadToAspectRatio(2.0)
    >>> crop = aug(image=image)

    Pad ``image`` to size ``14x28``.

    """

    def __init__(self, aspect_ratio, pad_mode="constant", pad_cval=0,
                 position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(PadToAspectRatio, self).__init__(
            width=None, height=None, pad_mode=pad_mode, pad_cval=pad_cval,
            position=position,
            name=name, deterministic=deterministic, random_state=random_state)
        self.aspect_ratio = aspect_ratio

    def _draw_samples(self, batch, random_state):
        _sizes, pad_xs, pad_ys, pad_modes, pad_cvals = super(
            PadToAspectRatio, self
        )._draw_samples(batch, random_state)

        shapes = batch.get_rowwise_shapes()
        sizes = []
        for shape in shapes:
            height, width = shape[0:2]

            paddings = ia.compute_paddings_for_aspect_ratio(
                shape,
                aspect_ratio=self.aspect_ratio)

            # TODO change that
            # note that these are not in the same order as shape tuples
            # in PadToFixedSize
            new_size = (
                width + paddings[1] + paddings[3],
                height + paddings[0] + paddings[2]
            )
            sizes.append(new_size)

        return sizes, pad_xs, pad_ys, pad_modes, pad_cvals

    def get_parameters(self):
        return [self.aspect_ratio, self.pad_mode, self.pad_cval,
                self.position]


class CenterPadToAspectRatio(PadToAspectRatio):
    """Pad images equally on all sides until H/W matches an aspect ratio.

    This is the same as :class`PadToAspectRatio`, but uses
    ``position="center"`` by default, which spreads the pad amounts equally
    over all image sides, while :class`PadToAspectRatio` by default spreads
    them randomly.

    dtype support::

        See :class:`imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    aspect_ratio : number
        See :func:`PadToAspectRatio.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToAspectRatio.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToAspectRatio.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((15*12)).astype(np.uint8).reshape((15, 12))
    >>> aug = iaa.CenterPadToAspectRatio(1.0)
    >>> image_padded = aug(image=image)

    Pad ``image`` to size ``15x12`. This operation will add ``1`` row at the
    top and ``2`` rows at the bottom.

    """

    def __init__(self, aspect_ratio, pad_mode="constant", pad_cval=0,
                 name=None, deterministic=False, random_state=None):
        super(CenterPadToAspectRatio, self).__init__(
            aspect_ratio=aspect_ratio, position="center",
            pad_mode=pad_mode, pad_cval=pad_cval,
            name=name, deterministic=deterministic, random_state=random_state)


class CropToSquare(CropToAspectRatio):
    """Crop images until their width and height are identical.

    This is identical to :class:`CropToAspectRatio` with ``aspect_ratio=1.0``.

    Images with axis sizes of ``0`` will not be altered.

    dtype support::

        See :class:`imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`CropToFixedSize.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((14*12)).astype(np.uint8).reshape((14, 12))
    >>> aug = iaa.CropToSquare()
    >>> crop = aug(image=image)

    Crop ``image`` down to size ``12x12``.

    """

    def __init__(self, position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(CropToSquare, self).__init__(
            aspect_ratio=1.0, position=position,
            name=name, deterministic=deterministic, random_state=random_state)


class CenterCropToSquare(CropToSquare):
    """Crop images equally on all sides until their height/width are identical.

    In contrast to :class:`CropToSquare`, this augmenter always tries to
    spread the columns/rows to remove equally over both sides of the respective
    axis to be cropped. :class:`CropToAspectRatio` by default spreads the
    croppings randomly.

    This augmenter is identical to :class:`CropToSquare` with
    ``position="center"``, and thereby the same as :class:`CropToAspectRatio`
    with ``aspect_ratio=1.0, position="center"``.

    Images with axis sizes of ``0`` will not be altered.

    dtype support::

        See :class:`imgaug.augmenters.size.CropToFixedSize`.

    Parameters
    ----------
    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((15*12)).astype(np.uint8).reshape((15, 12))
    >>> aug = iaa.CenterCropToSquare()
    >>> crop = aug(image=image)

    Crop ``image`` down to size ``12x12`` by removing ``1`` row at the top
    and ``2`` rows at the bottom.

    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        super(CenterCropToSquare, self).__init__(
            position="center",
            name=name, deterministic=deterministic, random_state=random_state)


class PadToSquare(PadToAspectRatio):
    """Pad images until their height and width are identical.

    This augmenter is identical to :class:`PadToAspectRatio` with
    ``aspect_ratio=1.0``.

    dtype support::

        See :class:`imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    position : {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center', 'center-bottom', 'right-top', 'right-center', 'right-bottom'} or tuple of float or StochasticParameter or tuple of StochasticParameter, optional
        See :func:`PadToFixedSize.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToFixedSize.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToFixedSize.__init__`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((14*12)).astype(np.uint8).reshape((14, 12))
    >>> aug = iaa.PadToSquare()
    >>> crop = aug(image=image)

    Pad ``image`` to size ``14x14``.

    """

    def __init__(self, pad_mode="constant", pad_cval=0, position="uniform",
                 name=None, deterministic=False, random_state=None):
        super(PadToSquare, self).__init__(
            aspect_ratio=1.0, pad_mode=pad_mode, pad_cval=pad_cval,
            position=position,
            name=name, deterministic=deterministic, random_state=random_state)


class CenterPadToSquare(PadToSquare):
    """Pad images equally on all sides until their height & width are identical.

    This is the same as :class`PadToSquare`, but uses
    ``position="center"`` by default, which spreads the pad amounts equally
    over all image sides, while :class`PadToSquare` by default spreads
    them randomly. This augmenter is thus also identical to
    :class:`PadToAspectRatio` with ``aspect_ratio=1.0, position="center"``.

    dtype support::

        See :class:`imgaug.augmenters.size.PadToFixedSize`.

    Parameters
    ----------
    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    pad_mode : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToAspectRatio.__init__`.

    pad_cval : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :func:`imgaug.augmenters.size.PadToAspectRatio.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.arange((15*12)).astype(np.uint8).reshape((15, 12))
    >>> aug = iaa.CenterPadToSquare()
    >>> image_padded = aug(image=image)

    Pad ``image`` to size ``15x12`. This operation will add ``1`` row at the
    top and ``2`` rows at the bottom.

    """

    def __init__(self, pad_mode="constant", pad_cval=0,
                 name=None, deterministic=False, random_state=None):
        super(CenterPadToSquare, self).__init__(
            pad_mode=pad_mode, pad_cval=pad_cval, position="center",
            name=name, deterministic=deterministic, random_state=random_state)


class KeepSizeByResize(meta.Augmenter):
    """Resize images back to their input sizes after applying child augmenters.

    Combining this with e.g. a cropping augmenter as the child will lead to
    images being resized back to the input size after the crop operation was
    applied. Some augmenters have a ``keep_size`` argument that achieves the
    same goal (if set to ``True``), though this augmenter offers control over
    the interpolation mode and which augmentables to resize (images, heatmaps,
    segmentation maps).

    dtype support::

        See :func:`imgaug.imgaug.imresize_many_images`.

    Parameters
    ----------
    children : Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to images. These augmenters may change
        the image size.

    interpolation : KeepSizeByResize.NO_RESIZE or {'nearest', 'linear', 'area', 'cubic'} or {cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC} or list of str or list of int or StochasticParameter, optional
        The interpolation mode to use when resizing images.
        Can take any value that :func:`imgaug.imgaug.imresize_single_image`
        accepts, e.g. ``cubic``.

            * If this is ``KeepSizeByResize.NO_RESIZE`` then images will not
              be resized.
            * If this is a single ``str``, it is expected to have one of the
              following values: ``nearest``, ``linear``, ``area``, ``cubic``.
            * If this is a single integer, it is expected to have a value
              identical to one of: ``cv2.INTER_NEAREST``,
              ``cv2.INTER_LINEAR``, ``cv2.INTER_AREA``, ``cv2.INTER_CUBIC``.
            * If this is a ``list`` of ``str`` or ``int``, it is expected that
              each ``str``/``int`` is one of the above mentioned valid ones.
              A random one of these values will be sampled per image.
            * If this is a ``StochasticParameter``, it will be queried once per
              call to ``_augment_images()`` and must return ``N`` ``str`` s or
              ``int`` s (matching the above mentioned ones) for ``N`` images.

    interpolation_heatmaps : KeepSizeByResize.SAME_AS_IMAGES or KeepSizeByResize.NO_RESIZE or {'nearest', 'linear', 'area', 'cubic'} or {cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC} or list of str or list of int or StochasticParameter, optional
        The interpolation mode to use when resizing heatmaps.
        Meaning and valid values are similar to `interpolation`. This
        parameter may also take the value ``KeepSizeByResize.SAME_AS_IMAGES``,
        which will lead to copying the interpolation modes used for the
        corresponding images. The value may also be returned on a per-image
        basis if `interpolation_heatmaps` is provided as a
        ``StochasticParameter`` or may be one possible value if it is
        provided as a ``list`` of ``str``.

    interpolation_segmaps : KeepSizeByResize.SAME_AS_IMAGES or KeepSizeByResize.NO_RESIZE or {'nearest', 'linear', 'area', 'cubic'} or {cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC} or list of str or list of int or StochasticParameter, optional
        The interpolation mode to use when resizing segmentation maps.
        Similar to `interpolation_heatmaps`.
        **Note**: For segmentation maps, only ``NO_RESIZE`` or nearest
        neighbour interpolation (i.e. ``nearest``) make sense in the vast
        majority of all cases.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.KeepSizeByResize(
    >>>     iaa.Crop((20, 40), keep_size=False)
    >>> )

    Apply random cropping to input images, then resize them back to their
    original input sizes. The resizing is done using this augmenter instead
    of the corresponding internal resizing operation in ``Crop``.

    >>> aug = iaa.KeepSizeByResize(
    >>>     iaa.Crop((20, 40), keep_size=False),
    >>>     interpolation="nearest"
    >>> )

    Same as in the previous example, but images are now always resized using
    nearest neighbour interpolation.

    >>> aug = iaa.KeepSizeByResize(
    >>>     iaa.Crop((20, 40), keep_size=False),
    >>>     interpolation=["nearest", "cubic"],
    >>>     interpolation_heatmaps=iaa.KeepSizeByResize.SAME_AS_IMAGES,
    >>>     interpolation_segmaps=iaa.KeepSizeByResize.NO_RESIZE
    >>> )

    Similar to the previous example, but images are now sometimes resized
    using linear interpolation and sometimes using nearest neighbour
    interpolation. Heatmaps are resized using the same interpolation as was
    used for the corresponding image. Segmentation maps are not resized and
    will therefore remain at their size after cropping.

    """

    NO_RESIZE = "NO_RESIZE"
    SAME_AS_IMAGES = "SAME_AS_IMAGES"

    def __init__(self, children,
                 interpolation="cubic",
                 interpolation_heatmaps=SAME_AS_IMAGES,
                 interpolation_segmaps="nearest",
                 name=None, deterministic=False, random_state=None):
        super(KeepSizeByResize, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        self.children = children

        def _validate_param(val, allow_same_as_images):
            valid_ips_and_resize = ia.IMRESIZE_VALID_INTERPOLATIONS \
                                  + [KeepSizeByResize.NO_RESIZE]
            if allow_same_as_images and val == self.SAME_AS_IMAGES:
                return self.SAME_AS_IMAGES
            elif val in valid_ips_and_resize:
                return iap.Deterministic(val)
            elif isinstance(val, list):
                assert len(val) > 0, (
                    "Expected a list of at least one interpolation method. "
                    "Got an empty list.")
                valid_ips_here = valid_ips_and_resize
                if allow_same_as_images:
                    valid_ips_here = valid_ips_here \
                                     + [KeepSizeByResize.SAME_AS_IMAGES]
                only_valid_ips = all([ip in valid_ips_here for ip in val])
                assert only_valid_ips, (
                    "Expected each interpolations to be one of '%s', got "
                    "'%s'." % (str(valid_ips_here), str(val)))
                return iap.Choice(val)
            elif isinstance(val, iap.StochasticParameter):
                return val
            else:
                raise Exception(
                    "Expected interpolation to be one of '%s' or a list of "
                    "these values or a StochasticParameter. Got type %s." % (
                        str(ia.IMRESIZE_VALID_INTERPOLATIONS), type(val)))

        self.children = meta.handle_children_list(children, self.name, "then")
        self.interpolation = _validate_param(interpolation, False)
        self.interpolation_heatmaps = _validate_param(interpolation_heatmaps,
                                                      True)
        self.interpolation_segmaps = _validate_param(interpolation_segmaps,
                                                     True)

    def _augment_batch(self, batch, random_state, parents, hooks):
        with batch.propagation_hooks_ctx(self, hooks, parents):
            images_were_array = None
            if batch.images is not None:
                images_were_array = ia.is_np_array(batch.images)
            shapes_orig = self.get_shapes(batch)

            samples = self._draw_samples(batch.nb_rows, random_state)

            batch = self.children.augment_batch(
                batch, parents=parents + [self], hooks=hooks)

            if batch.images is not None:
                batch.images = self._keep_size_images(
                    batch.images, shapes_orig["images"], images_were_array,
                    samples)

            if batch.heatmaps is not None:
                # dont use shapes_orig["images"] because they might be None
                batch.heatmaps = self._keep_size_maps(
                    batch.heatmaps, shapes_orig["heatmaps"],
                    shapes_orig["heatmaps_arr"], samples[1])

            if batch.segmentation_maps is not None:
                # dont use shapes_orig["images"] because they might be None
                batch.segmentation_maps = self._keep_size_maps(
                    batch.segmentation_maps, shapes_orig["segmentation_maps"],
                    shapes_orig["segmentation_maps_arr"], samples[2])

            for augm_name in ["keypoints", "bounding_boxes", "polygons",
                              "line_strings"]:
                augm_value = getattr(batch, augm_name)
                if augm_value is not None:
                    func = functools.partial(
                        self._keep_size_keypoints,
                        shapes_orig=shapes_orig[augm_name],
                        interpolations=samples[0])
                    cbaois = self._apply_to_cbaois_as_keypoints(augm_value,
                                                                func)
                    setattr(batch, augm_name, cbaois)
        return batch

    @classmethod
    def _keep_size_images(cls, images, shapes_orig, images_were_array,
                          samples):
        interpolations, _, _ = samples

        gen = zip(images, interpolations, shapes_orig)
        result = []
        for image, interpolation, input_shape in gen:
            if interpolation == KeepSizeByResize.NO_RESIZE:
                result.append(image)
            else:
                result.append(
                    ia.imresize_single_image(image, input_shape[0:2],
                                             interpolation))

        if images_were_array:
            # note here that NO_RESIZE can have led to different shapes
            nb_shapes = len(set([image.shape for image in result]))
            if nb_shapes == 1:
                result = np.array(result, dtype=images.dtype)

        return result

    @classmethod
    def _keep_size_maps(cls, augmentables, shapes_orig_images,
                        shapes_orig_arrs, interpolations):
        result = []
        gen = zip(augmentables, interpolations,
                  shapes_orig_arrs, shapes_orig_images)
        for augmentable, interpolation, arr_shape_orig, img_shape_orig in gen:
            if interpolation == "NO_RESIZE":
                result.append(augmentable)
            else:
                augmentable = augmentable.resize(
                    arr_shape_orig[0:2], interpolation=interpolation)
                augmentable.shape = img_shape_orig
                result.append(augmentable)

        return result

    @classmethod
    def _keep_size_keypoints(cls, kps_aug, shapes_orig, interpolations):
        result = []
        gen = zip(kps_aug, interpolations, shapes_orig)
        for kps_aug, interpolation, input_shape in gen:
            if interpolation == KeepSizeByResize.NO_RESIZE:
                result.append(kps_aug)
            else:
                result.append(kps_aug.on(input_shape))

        return result

    @classmethod
    def get_shapes(cls, batch):
        result = dict()
        for column in batch.columns:
            result[column.name] = [cell.shape for cell in column.value]

        if batch.heatmaps is not None:
            result["heatmaps_arr"] = [
                cell.arr_0to1.shape for cell in batch.heatmaps]

        if batch.segmentation_maps is not None:
            result["segmentation_maps_arr"] = [
                cell.arr.shape for cell in batch.segmentation_maps]

        return result

    def _draw_samples(self, nb_images, random_state):
        rngs = random_state.duplicate(3)
        interpolations = self.interpolation.draw_samples((nb_images,),
                                                         random_state=rngs[0])

        if self.interpolation_heatmaps == KeepSizeByResize.SAME_AS_IMAGES:
            interpolations_heatmaps = np.copy(interpolations)
        else:
            interpolations_heatmaps = self.interpolation_heatmaps.draw_samples(
                (nb_images,), random_state=rngs[1]
            )

            # Note that `interpolations_heatmaps == self.SAME_AS_IMAGES`
            # works here only if the datatype of the array is such that it
            # may contain strings. It does not work properly for e.g.
            # integer arrays and will produce a single bool output, even
            # for arrays with more than one entry.
            same_as_imgs_idx = [ip == self.SAME_AS_IMAGES
                                for ip in interpolations_heatmaps]

            interpolations_heatmaps[same_as_imgs_idx] = \
                interpolations[same_as_imgs_idx]

        if self.interpolation_segmaps == KeepSizeByResize.SAME_AS_IMAGES:
            interpolations_segmaps = np.copy(interpolations)
        else:
            # TODO This used previously the same seed as the heatmaps part
            #      leading to the same sampled values. Was that intentional?
            #      Doesn't look like it should be that way.
            interpolations_segmaps = self.interpolation_segmaps.draw_samples(
                (nb_images,), random_state=rngs[2]
            )

            # Note that `interpolations_heatmaps == self.SAME_AS_IMAGES`
            # works here only if the datatype of the array is such that it
            # may contain strings. It does not work properly for e.g.
            # integer arrays and will produce a single bool output, even
            # for arrays with more than one entry.
            same_as_imgs_idx = [ip == self.SAME_AS_IMAGES
                                for ip in interpolations_segmaps]

            interpolations_segmaps[same_as_imgs_idx] = \
                interpolations[same_as_imgs_idx]

        return interpolations, interpolations_heatmaps, interpolations_segmaps

    def _to_deterministic(self):
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self):
        return [self.interpolation, self.interpolation_heatmaps]

    def get_children_lists(self):
        return [self.children]

    def __str__(self):
        pattern = (
            "%s("
            "interpolation=%s, "
            "interpolation_heatmaps=%s, "
            "name=%s, "
            "children=%s, "
            "deterministic=%s"
            ")")
        return pattern % (
            self.__class__.__name__, self.interpolation,
            self.interpolation_heatmaps, self.name, self.children,
            self.deterministic)
