"""
Augmenters that apply color space oriented changes.

Do not import directly from this file, as the categorization is not final.
Use instead
    `from imgaug import augmenters as iaa`
and then e.g. ::

    seq = iaa.Sequential([
        iaa.Grayscale((0.0, 1.0)),
        iaa.AddToHueAndSaturation((-10, 10))
    ])

List of augmenters:
    * InColorspace (deprecated)
    * WithColorspace
    * AddToHueAndSaturation
    * ChangeColorspace
    * Grayscale
"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Deterministic, Choice, Uniform
import numpy as np
import cv2
import six.moves as sm
import warnings

from .meta import Augmenter, Sequential, WithChannels
from .arithmetic import Add

# legacy support
def InColorspace(to_colorspace, from_colorspace="RGB", children=None, name=None, deterministic=False, random_state=None):
    """Deprecated. Use WithColorspace."""
    warnings.warn('InColorspace is deprecated. Use WithColorspace.', DeprecationWarning)
    return WithColorspace(to_colorspace, from_colorspace, children, name, deterministic, random_state)

class WithColorspace(Augmenter):
    """
    Apply child augmenters within a specific colorspace.

    This augumenter takes a source colorspace A and a target colorspace B
    as well as children C. It changes images from A to B, then applies the
    child augmenters C and finally changes the colorspace back from B to A.
    See also ChangeColorspace() for more.

    Parameters
    ----------
    to_colorspace : string
        See `ChangeColorspace.__init__()`

    from_colorspace : string, optional(default="RGB")
        See `ChangeColorspace.__init__()`

    children : None or Augmenter or list of Augmenters, optional(default=None)
        See `ChangeColorspace.__init__()`

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
    >>>                          children=iaa.WithChannels(0, iaa.Add(10)))

    This augmenter will add 10 to Hue value in HSV colorspace,
    then change the colorspace back to the original (RGB).

    """

    def __init__(self, to_colorspace, from_colorspace="RGB", children=None, name=None, deterministic=False, random_state=None):
        super(WithColorspace, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.to_colorspace = to_colorspace
        self.from_colorspace = from_colorspace

        if children is None:
            self.children = Sequential([], name="%s-then" % (self.name,))
        elif ia.is_iterable(children):
            self.children = Sequential(children, name="%s-then" % (self.name,))
        elif isinstance(children, Augmenter):
            self.children = Sequential([children], name="%s-then" % (self.name,))
        else:
            raise Exception("Expected None, Augmenter or list/tuple of Augmenter as children, got %s." % (type(children),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        if hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
            result = ChangeColorspace(
                to_colorspace=self.to_colorspace,
                from_colorspace=self.from_colorspace,
            ).augment_images(images=result)
            result = self.children.augment_images(
                images=result,
                parents=parents + [self],
                hooks=hooks,
            )
            result = ChangeColorspace(
                to_colorspace=self.from_colorspace,
                from_colorspace=self.to_colorspace,
            ).augment_images(images=result)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def _to_deterministic(self):
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = ia.new_random_state()
        return aug

    def get_parameters(self):
        return [self.channels]

    def get_children_lists(self):
        return [self.children]

    def __str__(self):
        return "WithColorspace(from_colorspace=%s, to_colorspace=%s, name=%s, children=[%s], deterministic=%s)" % (self.from_colorspace, self.to_colorspace, self.name, self.children, self.deterministic)

# TODO removed deterministic and random_state here as parameters, because this
# function creates multiple child augmenters. not sure if this is sensible
# (give them all the same random state instead?)
def AddToHueAndSaturation(value=0, per_channel=False, from_colorspace="RGB", channels=[0, 1], name=None): # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
    """
    Augmenter that transforms images into HSV space, selects the H and S
    channels and then adds a given range of values to these.

    Parameters
    ----------
    value : int or iterable of two ints or StochasticParameter, optional(default=0)
        See `Add.__init__()`

    per_channel : bool or float, optional(default=False)
        See `Add.__init__()`

    from_colorspace : string, optional(default="RGB")
        See `ChangeColorspace.__init__()`

    channels : integer or list of integers or None, optional(default=[0, 1])
        See `WithChannels.__init__()`

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >> aug = AddToHueAndSaturation((-20, 20), per_channel=True)

    Adds random values between -20 and 20 to the hue and saturation
    (independently per channel and the same value for all pixels within
    that channel).

    """
    return WithColorspace(
        to_colorspace="HSV",
        from_colorspace=from_colorspace,
        children=WithChannels(
            channels=channels,
            children=Add(value=value, per_channel=per_channel)
        ),
        name=name
    )

# TODO tests
# Note: Not clear whether this class will be kept (for anything aside from grayscale)
# other colorspaces dont really make sense and they also might not work correctly
# due to having no clearly limited range (like 0-255 or 0-1)
# TODO rename to ChangeColorspace3D and then introduce ChangeColorspace, which
# does not enforce 3d images?
class ChangeColorspace(Augmenter):
    """
    Augmenter to change the colorspace of images.

    NOTE: This augmenter is not tested. Some colorspaces might work, others
    might not.

    NOTE: This augmenter tries to project the colorspace value range on 0-255.
    It outputs dtype=uint8 images.

    Parameters
    ----------
    to_colorspace : string or iterable or StochasticParameter
        The target colorspace.
        Allowed are: RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv.
            * If a string, it must be among the allowed colorspaces.
            * If an iterable, it is expected to be a list of strings, each one
              being an allowed colorspace. A random element from the list
              will be chosen per image.
            * If a StochasticParameter, it is expected to return string. A new
              sample will be drawn per image.

    from_colorspace : string, optional(default="RGB")
        The source colorspace (of the input images).
        Allowed are: RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv.

    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=1.0)
        The alpha value of the new colorspace when overlayed over the
        old one. A value close to 1.0 means that mostly the new
        colorspace is visible. A value close to 0.0 means, that mostly the
        old image is visible. Use a tuple (a, b) to use a random value
        `x` with `a <= x <= b` as the alpha value per image.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    """

    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"
    CIE = "CIE"
    YCrCb = "YCrCb"
    HSV = "HSV"
    HLS = "HLS"
    Lab = "Lab"
    Luv = "Luv"
    COLORSPACES = set([
        RGB,
        BGR,
        GRAY,
        CIE,
        YCrCb,
        HSV,
        HLS,
        Lab,
        Luv
    ])
    CV_VARS = {
        # RGB
        #"RGB2RGB": cv2.COLOR_RGB2RGB,
        "RGB2BGR": cv2.COLOR_RGB2BGR,
        "RGB2GRAY": cv2.COLOR_RGB2GRAY,
        "RGB2CIE": cv2.COLOR_RGB2XYZ,
        "RGB2YCrCb": cv2.COLOR_RGB2YCR_CB,
        "RGB2HSV": cv2.COLOR_RGB2HSV,
        "RGB2HLS": cv2.COLOR_RGB2HLS,
        "RGB2LAB": cv2.COLOR_RGB2LAB,
        "RGB2LUV": cv2.COLOR_RGB2LUV,
        # BGR
        "BGR2RGB": cv2.COLOR_BGR2RGB,
        #"BGR2BGR": cv2.COLOR_BGR2BGR,
        "BGR2GRAY": cv2.COLOR_BGR2GRAY,
        "BGR2CIE": cv2.COLOR_BGR2XYZ,
        "BGR2YCrCb": cv2.COLOR_BGR2YCR_CB,
        "BGR2HSV": cv2.COLOR_BGR2HSV,
        "BGR2HLS": cv2.COLOR_BGR2HLS,
        "BGR2LAB": cv2.COLOR_BGR2LAB,
        "BGR2LUV": cv2.COLOR_BGR2LUV,
        # HSV
        "HSV2RGB": cv2.COLOR_HSV2RGB,
        "HSV2BGR": cv2.COLOR_HSV2BGR,
    }

    def __init__(self, to_colorspace, from_colorspace="RGB", alpha=1.0, name=None, deterministic=False, random_state=None):
        super(ChangeColorspace, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(alpha):
            self.alpha = Deterministic(alpha)
        elif ia.is_iterable(alpha):
            ia.do_assert(len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(alpha),))
            self.alpha = Uniform(alpha[0], alpha[1])
        elif isinstance(alpha, StochasticParameter):
            self.alpha = alpha
        else:
            raise Exception("Expected alpha to be int or float or tuple/list of ints/floats or StochasticParameter, got %s." % (type(alpha),))

        if ia.is_string(to_colorspace):
            ia.do_assert(to_colorspace in ChangeColorspace.COLORSPACES)
            self.to_colorspace = Deterministic(to_colorspace)
        elif ia.is_iterable(to_colorspace):
            ia.do_assert(all([ia.is_string(colorspace) for colorspace in to_colorspace]))
            ia.do_assert(all([(colorspace in ChangeColorspace.COLORSPACES) for colorspace in to_colorspace]))
            self.to_colorspace = Choice(to_colorspace)
        elif isinstance(to_colorspace, StochasticParameter):
            self.to_colorspace = to_colorspace
        else:
            raise Exception("Expected to_colorspace to be string, list of strings or StochasticParameter, got %s." % (type(to_colorspace),))

        self.from_colorspace = from_colorspace
        ia.do_assert(self.from_colorspace in ChangeColorspace.COLORSPACES)
        ia.do_assert(from_colorspace != ChangeColorspace.GRAY)

        self.eps = 0.001 # epsilon value to check if alpha is close to 1.0 or 0.0

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        alphas = self.alpha.draw_samples((nb_images,), random_state=ia.copy_random_state(random_state))
        to_colorspaces = self.to_colorspace.draw_samples((nb_images,), random_state=ia.copy_random_state(random_state))
        for i in sm.xrange(nb_images):
            alpha = alphas[i]
            to_colorspace = to_colorspaces[i]
            image = images[i]

            ia.do_assert(0.0 <= alpha <= 1.0)
            ia.do_assert(to_colorspace in ChangeColorspace.COLORSPACES)

            if alpha == 0 or self.from_colorspace == to_colorspace:
                pass # no change necessary
            else:
                # some colorspaces here should use image/255.0 according to the docs,
                # but at least for conversion to grayscale that results in errors,
                # ie uint8 is expected

                if self.from_colorspace in [ChangeColorspace.RGB, ChangeColorspace.BGR]:
                    from_to_var_name = "%s2%s" % (self.from_colorspace, to_colorspace)
                    from_to_var = ChangeColorspace.CV_VARS[from_to_var_name]
                    img_to_cs = cv2.cvtColor(image, from_to_var)
                else:
                    # convert to RGB
                    from_to_var_name = "%s2%s" % (self.from_colorspace, ChangeColorspace.RGB)
                    from_to_var = ChangeColorspace.CV_VARS[from_to_var_name]
                    img_rgb = cv2.cvtColor(image, from_to_var)

                    if to_colorspace == ChangeColorspace.RGB:
                        img_to_cs = img_rgb
                    else:
                        # convert from RGB to desired target colorspace
                        from_to_var_name = "%s2%s" % (ChangeColorspace.RGB, to_colorspace)
                        from_to_var = ChangeColorspace.CV_VARS[from_to_var_name]
                        img_to_cs = cv2.cvtColor(img_rgb, from_to_var)

                # this will break colorspaces that have values outside 0-255 or 0.0-1.0
                if ia.is_integer_array(img_to_cs):
                    img_to_cs = np.clip(img_to_cs, 0, 255).astype(np.uint8)
                else:
                    img_to_cs = np.clip(img_to_cs * 255, 0, 255).astype(np.uint8)

                # for grayscale: covnert from (H, W) to (H, W, 3)
                if len(img_to_cs.shape) == 2:
                    img_to_cs = img_to_cs[:, :, np.newaxis]
                    img_to_cs = np.tile(img_to_cs, (1, 1, 3))

                if alpha >= (1 - self.eps):
                    result[i] = img_to_cs
                elif alpha <= self.eps:
                    result[i] = image
                else:
                    result[i] = (alpha * img_to_cs + (1 - alpha) * image).astype(np.uint8)

        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.to_colorspace, self.alpha]

# TODO tests
# TODO rename to Grayscale3D and add Grayscale that keeps the image at 1D
def Grayscale(alpha=0, from_colorspace="RGB", name=None, deterministic=False, random_state=None):
    """
    Augmenter to convert images to their grayscale versions.

    NOTE: Number of output channels is still 3, i.e. this augmenter just
    "removes" color.

    Parameters
    ----------
    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=0)
        The alpha value of the grayscale image when overlayed over the
        old image. A value close to 1.0 means, that mostly the new grayscale
        image is visible. A value close to 0.0 means, that mostly the
        old image is visible. Use a tuple (a, b) to sample per image a
        random value x with a <= x <= b as the alpha value.

    from_colorspace : string, optional(default="RGB")
        The source colorspace (of the input images).
        Allowed are: RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv.
        Only RGB is decently tested.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Grayscale(alpha=1.0)

    creates an augmenter that turns images to their grayscale versions.

    >>> aug = iaa.Grayscale(alpha=(0.0, 1.0))

    creates an augmenter that turns images to their grayscale versions with
    an alpha value in the range 0 <= alpha <= 1. An alpha value of 0.5 would
    mean, that the output image is 50 percent of the input image and 50
    percent of the grayscale image (i.e. 50 percent of color removed).

    """
    return ChangeColorspace(to_colorspace=ChangeColorspace.GRAY, alpha=alpha, from_colorspace=from_colorspace, name=name, deterministic=deterministic, random_state=random_state)
