"""
Augmenters that apply changes to images based on segmentation methods.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([
        iaa.Superpixels(...)
    ])

List of augmenters:

    * Superpixels
    * Voronoi

"""
from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod

import numpy as np
# use skimage.segmentation instead from ... import segmentation here,
# because otherwise unittest seems to mix up imgaug.augmenters.segmentation
# with skimage.segmentation for whatever reason
import skimage.segmentation
import skimage.measure
import six
import six.moves as sm

from . import meta
import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt


# TODO merge this into imresize?
def _ensure_image_max_size(image, max_size, interpolation):
    """Ensure that images do not exceed a required maximum sidelength.

    This downscales to `max_size` if any side violates that maximum.
    The other side is downscaled too so that the aspect ratio is maintained.

    dtype support::

        See :func:`imgaug.imgaug.imresize_single_image`.

    Parameters
    ----------
    image : ndarray
        Image to potentially downscale.

    max_size : int
        Maximum length of any side of the image.

    interpolation : string or int
        See :func:`imgaug.imgaug.imresize_single_image`.

    """
    if max_size is not None:
        size = max(image.shape[0], image.shape[1])
        if size > max_size:
            resize_factor = max_size / size
            new_height = int(image.shape[0] * resize_factor)
            new_width = int(image.shape[1] * resize_factor)
            image = ia.imresize_single_image(
                image,
                (new_height, new_width),
                interpolation=interpolation)
    return image


# TODO add compactness parameter
class Superpixels(meta.Augmenter):
    """
    Transform images parially/completely to their superpixel representation.

    This implementation uses skimage's version of the SLIC algorithm.

    dtype support::

        if (image size <= max_size)::

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested
            * ``uint64``: limited (1)
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested
            * ``int64``: limited (1)
            * ``float16``: no (2)
            * ``float32``: no (2)
            * ``float64``: no (3)
            * ``float128``: no (2)
            * ``bool``: yes; tested

            - (1) Superpixel mean intensity replacement requires computing
                  these means as float64s. This can cause inaccuracies for
                  large integer values.
            - (2) Error in scikit-image.
            - (3) Loss of resolution in scikit-image.

        if (image size > max_size)::

            minimum of (
                ``imgaug.augmenters.segmentation.Superpixels(image size <= max_size)``,
                :func:`imgaug.augmenters.segmentation._ensure_image_max_size`
            )

    Parameters
    ----------
    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a number, then that number will always be used.
            * If tuple ``(a, b)``, then a random probability will be sampled
              from the interval ``[a, b]`` per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    n_segments : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Rough target number of how many superpixels to generate (the algorithm
        may deviate from this number). Lower value will lead to coarser
        superpixels. Higher values are computationally more intensive and
        will hence lead to a slowdown.

            * If a single int, then that value will always be used as the
              number of segments.
            * If a tuple ``(a, b)``, then a value from the discrete interval
              ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`imgaug.imgaug.imresize_single_image`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Superpixels(p_replace=1.0, n_segments=64)

    Generates around ``64`` superpixels per image and replaces all of them with
    their average color (standard superpixel image).

    >>> aug = iaa.Superpixels(p_replace=0.5, n_segments=64)

    Generates around ``64`` superpixels per image and replaces half of them
    with their average color, while the other half are left unchanged (i.e.
    they still show the input image's content).

    >>> aug = iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128))

    Generates between ``16`` and ``128`` superpixels per image and replaces
    ``25`` to ``100`` percent of them with their average color.

    """

    def __init__(self, p_replace=0, n_segments=100, max_size=128,
                 interpolation="linear",
                 name=None, deterministic=False, random_state=None):
        super(Superpixels, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.p_replace = iap.handle_probability_param(
            p_replace, "p_replace", tuple_to_uniform=True, list_to_choice=True)
        self.n_segments = iap.handle_discrete_param(
            n_segments, "n_segments", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.max_size = max_size
        self.interpolation = interpolation

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool",
                                  "uint8", "uint16", "uint32", "uint64",
                                  "int8", "int16", "int32", "int64"],
                         disallowed=["uint128", "uint256",
                                     "int128", "int256",
                                     "float16", "float32", "float64",
                                     "float96", "float128", "float256"],
                         augmenter=self)

        nb_images = len(images)
        rss = ia.derive_random_states(random_state, 1+nb_images)
        n_segments_samples = self.n_segments.draw_samples(
            (nb_images,), random_state=rss[0])

        # We cant reduce images to 0 or less segments, hence we pick the
        # lowest possible value in these cases (i.e. 1). The alternative
        # would be to not perform superpixel detection in these cases
        # (akin to n_segments=#pixels).
        # TODO add test for this
        n_segments_samples = np.clip(n_segments_samples, 1, None)

        for i, (image, rs) in enumerate(zip(images, rss[1:])):
            replace_samples = self.p_replace.draw_samples(
                (n_segments_samples[i],), random_state=rs)

            if np.max(replace_samples) == 0:
                # not a single superpixel would be replaced by its average
                # color, i.e. the image would not be changed, so just keep it
                continue

            image = images[i]

            orig_shape = image.shape
            image = _ensure_image_max_size(image, self.max_size, self.interpolation)

            segments = skimage.segmentation.slic(
                image, n_segments=n_segments_samples[i], compactness=10)

            image_aug = self._replace_segments(image, segments, replace_samples)

            if orig_shape != image_aug.shape:
                image_aug = ia.imresize_single_image(
                    image_aug,
                    orig_shape[0:2],
                    interpolation=self.interpolation)

            images[i] = image_aug
        return images

    @classmethod
    def _replace_segments(cls, image, segments, replace_samples):
        min_value, _center_value, max_value = \
                iadt.get_value_range_of_dtype(image.dtype)
        image_sp = np.copy(image)

        nb_channels = image.shape[2]
        for c in sm.xrange(nb_channels):
            # segments+1 here because otherwise regionprops always
            # misses the last label
            regions = skimage.measure.regionprops(
                segments+1, intensity_image=image[..., c])
            for ridx, region in enumerate(regions):
                # with mod here, because slic can sometimes create more
                # superpixel than requested. replace_samples then does not
                # have enough values, so we just start over with the first one
                # again.
                if replace_samples[ridx % len(replace_samples)] > 0.5:
                    mean_intensity = region.mean_intensity
                    image_sp_c = image_sp[..., c]

                    if image_sp_c.dtype.kind in ["i", "u", "b"]:
                        # After rounding the value can end up slightly outside
                        # of the value_range. Hence, we need to clip. We do
                        # clip via min(max(...)) instead of np.clip because
                        # the latter one does not seem to keep dtypes for
                        # dtypes with large itemsizes (e.g. uint64).
                        value = int(np.round(mean_intensity))
                        value = min(max(value, min_value), max_value)
                    else:
                        value = mean_intensity

                    image_sp_c[segments == ridx] = value

        return image_sp

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        return keypoints_on_images

    def get_parameters(self):
        return [self.p_replace, self.n_segments, self.max_size,
                self.interpolation]


@six.add_metaclass(ABCMeta)
class PointsSamplerIf(object):
    """Interface for all point samplers.

    Point samplers return coordinate arrays of shape ``Nx2``.
    These coordinates can be used in other augmenters, see e.g. ``Voronoi``.

    """

    @abstractmethod
    def sample_points(self, images, random_state):
        """Generate coordinates of points on images.

        Parameters
        ----------
        images : ndarray or list of ndarray
            One or more images for which to generate points.
            If this is a list of arrays, each one of them is expected to
            have three dimensions.
            If this is an array, it must be four-dimensional and the first
            axis is expected to denote the image index. For ``RGB`` images
            the array would hence have to be of shape ``(N, H, W, 3)``.

        random_state : None or numpy.random.RandomState or int or float
            A random state to use for any probabilistic function required
            during the point sampling.
            See :func:`imgaug.imgaug.normalize_random_state` for details.

        Returns
        -------
        ndarray
            An ``(N,2)`` ``float32`` array containing ``(x,y)`` subpixel
            coordinates, all of which being within the intervals
            ``[0.0, width]`` and ``[0.0, height]``.

        """


def _verify_sample_points_images(images):
    assert len(images) > 0, "Expected at least one image, got zero."
    if isinstance(images, list):
        assert all([ia.is_np_array(image) for image in images]), (
            "Expected list of numpy arrays, got list of types %s." % (
                ", ".join([str(type(image)) for image in images]),))
        assert all([image.ndim == 3 for image in images]), (
            "Expected each image to have three dimensions, "
            "got dimensions %s." % (
                ", ".join([str(image.ndim) for image in images]),))
    else:
        assert ia.is_np_array(images), (
            "Expected either a list of numpy arrays or a single numpy "
            "array of shape NxHxWxC. Got type %s." % (type(images),))
        assert images.ndim == 4, (
            "Expected a four-dimensional array of shape NxHxWxC. "
            "Got shape %d dimensions (shape: %s)." % (
                images.ndim, images.shape))


class RegularGridPointsSampler(PointsSamplerIf):
    """Sampler that generates a regular grid of coordinates on an image.

    'Regular grid' here means that on each axis all coordinates have the
    same distance from each other. Note that the distance may change between
    axis.

    Parameters
    ----------
    n_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of rows of coordinates to place on each image, i.e. the number
        of coordinates on the y-axis. Note that for each image, the sampled
        value is clipped to the interval ``[1..H]``, where ``H`` is the image
        height.

            * If a single int, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete interval
              ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    n_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of columns of coordinates to place on each image, i.e. the number
        of coordinates on the x-axis. Note that for each image, the sampled
        value is clipped to the interval ``[1..W]``, where ``W`` is the image
        width.

            * If a single int, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete interval
              ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.RegularGridPointsSampler(
    >>>     n_rows=(5, 20),
    >>>     n_cols=50)

    Creates a point sampler that generates regular grids of points. These grids
    contain ``r`` points on the y-axis, where ``r`` is sampled
    uniformly from the discrete interval ``[5..20]`` per image.
    On the x-axis, the grids always contain ``50`` points.

    """

    def __init__(self, n_rows, n_cols):
        self.n_rows = iap.handle_discrete_param(
            n_rows, "n_rows", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.n_cols = iap.handle_discrete_param(
            n_cols, "n_cols", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)

    def sample_points(self, images, random_state):
        random_state = ia.normalize_random_state(random_state)
        _verify_sample_points_images(images)

        n_rows_lst, n_cols_lst = self._draw_samples(images, random_state)
        return self._generate_point_grids(images, n_rows_lst, n_cols_lst)

    def _draw_samples(self, images, random_state):
        rss = ia.derive_random_states(random_state, 2)
        n_rows_lst = self.n_rows.draw_samples(len(images), random_state=rss[0])
        n_cols_lst = self.n_cols.draw_samples(len(images), random_state=rss[1])
        return self._clip_rows_and_cols(n_rows_lst, n_cols_lst, images)

    @classmethod
    def _clip_rows_and_cols(cls, n_rows_lst, n_cols_lst, images):
        heights = np.int32([image.shape[0] for image in images])
        widths = np.int32([image.shape[1] for image in images])
        # We clip intentionally not to H-1 or W-1 here. If e.g. an image has
        # a width of 1, we want to get a maximum of 1 column of coordinates.
        n_rows_lst = np.clip(n_rows_lst, 1, heights)
        n_cols_lst = np.clip(n_cols_lst, 1, widths)
        return n_rows_lst, n_cols_lst

    @classmethod
    def _generate_point_grids(cls, images, n_rows_lst, n_cols_lst):
        grids = []
        for image, n_rows_i, n_cols_i in zip(images, n_rows_lst, n_cols_lst):
            grids.append(cls._generate_point_grid(image, n_rows_i, n_cols_i))
        return grids

    @classmethod
    def _generate_point_grid(cls, image, n_rows, n_cols):
        height, width = image.shape[0:2]

        # We do not have to subtract 1 here from height/width as these are
        # subpixel coordinates. Technically, we could also place the cell
        # centers outside of the image plane.
        if n_rows == 1:
            yy = np.float32([float(height)/2])
        else:
            yy = np.linspace(0.0, height, num=n_rows)

        if n_cols == 1:
            xx = np.float32([float(width)/2])
        else:
            xx = np.linspace(0.0, width, num=n_cols)

        xx, yy = np.meshgrid(xx, yy)
        grid = np.vstack([xx.ravel(), yy.ravel()]).T
        return grid


class RelativeRegularGridPointsSampler(PointsSamplerIf):
    """Regular grid coordinate sampler; places more points on larger images.

    This is similar to ``RegularGridPointSampler``, but the number of rows
    and columns is given as fractions of each image's height and width.
    Hence, more coordinates are generated for larger images.

    Parameters
    ----------
    n_rows_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the y-axis. For a value
        ``y`` and image height ``H`` the number of actually placed coordinates
        (i.e. computed rows) is given by ``int(round(y*H))``.
        Note that for each image, the number of coordinates is clipped to the
        interval ``[1,H]``, where ``H`` is the image height.

            * If a single number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    n_cols_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the x-axis. For a value
        ``x`` and image height ``W`` the number of actually placed coordinates
        (i.e. computed columns) is given by ``int(round(x*W))``.
        Note that for each image, the number of coordinates is clipped to the
        interval ``[1,W]``, where ``W`` is the image width.

            * If a single number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.RelativeRegularGridPointsSampler(
    >>>     n_rows_frac=(0.01, 0.1),
    >>>     n_cols_frac=0.2)

    Creates a point sampler that generates regular grids of points. These grids
    contain ``round(y*H)`` points on the y-axis, where ``y`` is sampled
    uniformly from the interval ``[0.01, 0.1]`` per image and ``H`` is the
    image height. On the x-axis, the grids always contain ``0.2*W`` points,
    where ``W`` is the image width.

    """

    def __init__(self, n_rows_frac, n_cols_frac):
        eps = 1e-4
        self.n_rows_frac = iap.handle_continuous_param(
            n_rows_frac, "n_rows_frac", value_range=(0.0+eps, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        self.n_cols_frac = iap.handle_continuous_param(
            n_cols_frac, "n_cols_frac", value_range=(0.0+eps, 1.0),
            tuple_to_uniform=True, list_to_choice=True)

    def sample_points(self, images, random_state):
        random_state = ia.normalize_random_state(random_state)
        _verify_sample_points_images(images)

        n_rows, n_cols = self._draw_samples(images, random_state)
        return RegularGridPointsSampler._generate_point_grids(images,
                                                              n_rows, n_cols)

    def _draw_samples(self, images, random_state):
        n_augmentables = len(images)
        rss = ia.derive_random_states(random_state, 2)
        n_rows_frac = self.n_rows_frac.draw_samples(n_augmentables,
                                                    random_state=rss[0])
        n_cols_frac = self.n_cols_frac.draw_samples(n_augmentables,
                                                    random_state=rss[1])
        heights = np.int32([image.shape[0] for image in images])
        widths = np.int32([image.shape[1] for image in images])

        n_rows = np.round(n_rows_frac * heights)
        n_cols = np.round(n_cols_frac * widths)
        n_rows, n_cols = RegularGridPointsSampler._clip_rows_and_cols(
            n_rows, n_cols, images)

        return n_rows.astype(np.int32), n_cols.astype(np.int32)


class DropoutPointsSampler(PointsSamplerIf):
    """Remove a defined fraction of sampled points.

    Parameters
    ----------
    other_points_sampler : PointSamplerIf
        Another point sampler that is queried to generate a list of points.
        The dropout operation will be applied to that list.

    p_drop : number or tuple of number or imgaug.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. to set it to zero).
        A value of ``1.0`` would mean that (on average) ``100`` percent of all
        coordinates will be dropped, while ``0.0`` denotes ``0`` percent.
        Note that this sampler will always ensure that at least one coordinate
        is left after the dropout operation, i.e. even ``1.0`` will only
        drop all *except one* coordinate.

            * If a float, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
              the interval ``[a, b]`` per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per coordinate whether it should be *kept* (sampled
              value of ``>0.5``) or shouldn't be kept (sampled value of
              ``<=0.5``). If you instead want to provide the probability as
              a stochastic parameter, you can usually do
              ``imgaug.parameters.Binomial(1-p)`` to convert parameter `p` to
              a 0/1 representation.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.DropoutPointsSampler(
    >>>     iaa.RegularGridPointsSampler(10, 20),
    >>>     0.2)

    Creates a point sampler that first generates points following a regular
    grid of 10 rows and 20 columns, then randomly drops ``20`` percent of these
    points.

    """

    def __init__(self, other_points_sampler, p_drop):
        assert isinstance(other_points_sampler, PointsSamplerIf), (
            "Expected to get an instance of PointsSamplerIf as argument "
            "'other_points_sampler', got type %s." % (
                type(other_points_sampler),))
        self.other_points_sampler = other_points_sampler
        self.p_drop = self._convert_p_drop_to_inverted_mask_param(p_drop)

    @classmethod
    def _convert_p_drop_to_inverted_mask_param(cls, p_drop):
        # TODO this is the same as in Dropout, make DRY
        # TODO add list as an option
        if ia.is_single_number(p_drop):
            p_drop = iap.Binomial(1 - p_drop)
        elif ia.is_iterable(p_drop):
            assert len(p_drop) == 2
            assert p_drop[0] < p_drop[1]
            assert 0 <= p_drop[0] <= 1.0
            assert 0 <= p_drop[1] <= 1.0
            p_drop = iap.Binomial(iap.Uniform(1 - p_drop[1], 1 - p_drop[0]))
        elif isinstance(p_drop, iap.StochasticParameter):
            pass
        else:
            raise Exception(
                "Expected p_drop to be float or int or StochasticParameter, "
                "got %s." % (type(p_drop),))
        return p_drop

    def sample_points(self, images, random_state):
        random_state = ia.normalize_random_state(random_state)
        _verify_sample_points_images(images)

        rss = ia.derive_random_states(random_state, 2)
        points_on_images = self.other_points_sampler.sample_points(images,
                                                                   rss[0])
        drop_masks = self._draw_samples(points_on_images, rss[1])
        return self._apply_dropout_masks(points_on_images, drop_masks)

    def _draw_samples(self, points_on_images, random_state):
        rss = ia.derive_random_states(random_state, len(points_on_images))
        drop_masks = [self._draw_samples_for_image(points_on_image, rs)
                      for points_on_image, rs
                      in zip(points_on_images, rss)]
        return drop_masks

    def _draw_samples_for_image(self, points_on_image, random_state):
        drop_samples = self.p_drop.draw_samples((len(points_on_image),),
                                                random_state)
        keep_mask = (drop_samples > 0.5)
        return keep_mask

    @classmethod
    def _apply_dropout_masks(cls, points_on_images, keep_masks):
        points_on_images_dropped = []
        for points_on_image, keep_mask in zip(points_on_images, keep_masks):
            if len(points_on_image) == 0:
                # other sampler didn't provide any points
                poi_dropped = points_on_image
            else:
                if not np.any(keep_mask):
                    # keep at least one point if all were supposed to be
                    # dropped
                    # TODO this could also be moved into its own point sampler,
                    #      like AtLeastOnePoint(...)
                    idx = (len(points_on_image) - 1) // 2
                    keep_mask = np.copy(keep_mask)
                    keep_mask[idx] = True
                poi_dropped = points_on_image[keep_mask, :]
            points_on_images_dropped.append(poi_dropped)
        return points_on_images_dropped


class SubsamplingPointsSampler(PointsSamplerIf):
    """Ensure that the number of sampled points is below a maximum.

    This point sampler will sample points from another sampler and
    then -- in case more points were generated than an allowed maximum --
    will randomly pick `n_points_max` of these.

    Parameters
    ----------
    other_points_sampler : PointsSamplerIf
        Another point sampler that is queried to generate a list of points.
        The dropout operation will be applied to that list.

    n_points_max : int
        Maximum number of allowed points. If `other_points_sampler` generates
        more points than this maximum, a random subset of size `n_points_max`
        will be selected.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.SubsamplingPointsSampler(
    >>>     iaa.RelativeRegularGridPointsSampler(0.1, 0.2),
    >>>     50
    >>> )

    Creates a points sampler that places ``y*H`` points on the y-axis (with
    ``y`` being ``0.1`` and ``H`` being an image's height) and ``x*W`` on
    the x-axis (analogous). Then, if that number of placed points exceeds
    ``50`` (can easily happen for larger images), a random subset of ``50``
    points will be picked and returned.

    """

    def __init__(self, other_points_sampler, n_points_max):
        assert isinstance(other_points_sampler, PointsSamplerIf), (
            "Expected to get an instance of PointsSamplerIf as argument "
            "'other_points_sampler', got type %s." % (
                type(other_points_sampler),))
        self.other_points_sampler = other_points_sampler
        self.n_points_max = np.clip(n_points_max, -1, None)
        if self.n_points_max == 0:
            import warnings
            warnings.warn("Got n_points_max=0 in SubsamplingPointsSampler. "
                          "This will result in no points ever getting "
                          "returned.")

    def sample_points(self, images, random_state):
        random_state = ia.normalize_random_state(random_state)
        _verify_sample_points_images(images)

        rss = ia.derive_random_states(random_state, len(images) + 1)
        points_on_images = self.other_points_sampler.sample_points(
            images, rss[-1])
        return [self._subsample(points_on_image, self.n_points_max, rs)
                for points_on_image, rs
                in zip(points_on_images, rss[:-1])]

    @classmethod
    def _subsample(cls, points_on_image, n_points_max, random_state):
        if len(points_on_image) <= n_points_max:
            return points_on_image
        indices = np.arange(len(points_on_image))
        indices_to_keep = random_state.permutation(indices)[0:n_points_max]
        return points_on_image[indices_to_keep]
