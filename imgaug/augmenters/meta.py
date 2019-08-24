"""
Augmenters that don't apply augmentations themselves, but are needed
for meta usage.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([...])

List of augmenters:

    * Augmenter (base class for all augmenters)
    * Sequential
    * SomeOf
    * OneOf
    * Sometimes
    * WithChannels
    * Noop
    * Lambda
    * AssertLambda
    * AssertShape
    * ChannelShuffle

Note: WithColorspace is in ``color.py``.

"""
from __future__ import print_function, division, absolute_import

import warnings
from abc import ABCMeta, abstractmethod
import copy as copy_module
import re
import itertools
import sys

import numpy as np
import six
import six.moves as sm

import imgaug as ia
from .. import parameters as iap
from .. import random as iarandom
from imgaug.augmentables.batches import Batch, UnnormalizedBatch


@ia.deprecated("imgaug.dtypes.clip_")
def clip_augmented_image_(image, min_value, max_value):
    return clip_augmented_images_(image, min_value, max_value)


@ia.deprecated("imgaug.dtypes.clip_")
def clip_augmented_image(image, min_value, max_value):
    return clip_augmented_images(image, min_value, max_value)


@ia.deprecated("imgaug.dtypes.clip_")
def clip_augmented_images_(images, min_value, max_value):
    if ia.is_np_array(images):
        return np.clip(images, min_value, max_value, out=images)
    else:
        return [np.clip(image, min_value, max_value, out=image)
                for image in images]


@ia.deprecated("imgaug.dtypes.clip_")
def clip_augmented_images(images, min_value, max_value):
    if ia.is_np_array(images):
        images = np.copy(images)
    else:
        images = [np.copy(image) for image in images]
    return clip_augmented_images_(images, min_value, max_value)


def handle_children_list(lst, augmenter_name, lst_name, default="sequential"):
    if lst is None:
        if default == "sequential":
            return Sequential([], name="%s-%s" % (augmenter_name, lst_name))
        else:
            return default
    elif isinstance(lst, Augmenter):
        if ia.is_iterable(lst):
            # TODO why was this assert added here? seems to make no sense
            only_augmenters = all([isinstance(child, Augmenter)
                                   for child in lst])
            assert only_augmenters, (
                "Expected all children to be augmenters, got types %s." % (
                    ", ".join([str(type(v)) for v in lst])))
            return lst
        else:
            return Sequential(lst, name="%s-%s" % (augmenter_name, lst_name))
    elif ia.is_iterable(lst):
        if len(lst) == 0 and default != "sequential":
            return default
        only_augmenters = all([isinstance(child, Augmenter)
                               for child in lst])
        assert only_augmenters, (
            "Expected all children to be augmenters, got types %s." % (
                ", ".join([str(type(v)) for v in lst])))
        return Sequential(lst, name="%s-%s" % (augmenter_name, lst_name))
    else:
        raise Exception(
            "Expected None, Augmenter or list/tuple as children list %s "
            "for augmenter with name %s, got %s." % (
                lst_name, augmenter_name, type(lst),))


def reduce_to_nonempty(objs):
    objs_reduced = []
    ids = []
    for i, obj in enumerate(objs):
        assert hasattr(obj, "empty"), (
            "Expected object with property 'empty'. Got type %s." % (
                type(obj),))
        if not obj.empty:
            objs_reduced.append(obj)
            ids.append(i)
    return objs_reduced, ids


def invert_reduce_to_nonempty(objs, ids, objs_reduced):
    objs_inv = list(objs)
    for idx, obj_from_reduced in zip(ids, objs_reduced):
        objs_inv[idx] = obj_from_reduced
    return objs_inv


def estimate_max_number_of_channels(images):
    if ia.is_np_array(images):
        assert images.ndim == 4, (
            "Expected 'images' to be 4-dimensional if provided as array. "
            "Got %d dimensions." % (images.ndim,))
        return images.shape[3]
    else:
        assert ia.is_iterable(images), (
            "Expected 'images' to be an array or iterable, got %s." % (
                type(images),))
        if len(images) == 0:
            return None
        channels = [el.shape[2] if len(el.shape) >= 3 else 1 for el in images]
        return max(channels)


def copy_arrays(arrays):
    if ia.is_np_array(arrays):
        return np.copy(arrays)
    else:
        assert ia.is_iterable(arrays), (
            "Expected ndarray or iterable of ndarray, got type %s." % (
                type(arrays),))
        return [np.copy(array) for array in arrays]


@six.add_metaclass(ABCMeta)
class Augmenter(object):
    """
    Base class for Augmenter objects.
    All augmenters derive from this class.

    Parameters
    ----------
    name : None or str, optional
        Name given to the Augmenter instance. This name is used when
        converting the instance to a string, e.g. for ``print`` statements.
        It is also used for ``find``, ``remove`` or similar operations
        on augmenters with children.
        If ``None``, ``UnnamedX`` will be used as the name, where ``X``
        is the Augmenter's class name.

    deterministic : bool, optional
        Whether the augmenter instance's random state will be saved before
        augmenting a batch and then reset to that initial saved state
        after the augmentation was finished. I.e. if set to ``True``,
        each batch will be augmented in the same way (e.g. first image
        might always be flipped horizontally, second image will never be
        flipped etc.).
        This is useful when you want to transform multiple batches
        in the same way, or when you want to augment images and
        corresponding data (e.g. keypoints or segmentation maps) on these
        images. Usually, there is no need to set this variable by hand.
        Instead, instantiate the augmenter and then use
        :func:`imgaug.augmenters.Augmenter.to_deterministic`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        The RNG (random number generator) to use for this augmenter.
        Setting this parameter allows to control/influence the random
        number sampling of the augmenter. Usually, there is no need to
        set this parameter.

            * If ``None``: The global RNG is used (shared by all
              augmenters).
            * If ``int``: The value will be used as a seed for a new
              :class:`imgaug.random.RNG` instance.
            * If :class:`imgaug.random.RNG`: The ``RNG`` instance will be
              used without changes.
            * If :class:`imgaug.random.Generator`: A new
              :class:`imgaug.random.RNG` instance will be
              created, containing that generator.
            * If :class:`imgaug.random.bit_generator.BitGenerator`: Will
              be wrapped in a :class:`imgaug.random.Generator`. Then
              similar behaviour to :class:`imgaug.random.Generator`
              parameters.
            * If :class:`imgaug.random.SeedSequence`: Will
              be wrapped in a new bit generator and
              :class:`imgaug.random.Generator`. Then
              similar behaviour to :class:`imgaug.random.Generator`
              parameters.
            * If :class:`imgaug.random.RandomState`: Similar behaviour to
              :class:`imgaug.random.Generator`. Outdated in numpy 1.17+.

        If a new bit generator has to be created, it will be an instance
        of :class:`numpy.random.SFC64`.

    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        """Create a new Augmenter instance."""
        super(Augmenter, self).__init__()

        assert name is None or ia.is_string(name), (
            "Expected name to be None or string-like, got %s." % (
                type(name),))
        if name is None:
            self.name = "Unnamed%s" % (self.__class__.__name__,)
        else:
            self.name = name

        assert ia.is_single_bool(deterministic), (
            "Expected deterministic to be a boolean, got %s." % (
                type(deterministic),))
        self.deterministic = deterministic

        if deterministic and random_state is None:
            # Usually if None is provided, the global RNG will be used.
            # In case of deterministic mode we most likely rather want a local
            # RNG, which is here created.
            self.random_state = iarandom.RNG.create_pseudo_random_()
        else:
            # self.random_state = iarandom.normalize_rng_(random_state)
            self.random_state = iarandom.RNG(random_state)

        self.activated = True

    def augment_batches(self, batches, hooks=None, background=False):
        """Augment multiple batches.

        In contrast to other augment functions, this function **yields**
        batches instead of just returning a full list. This is more suited
        for most training loops. It also supports augmentation on multiple
        cpu cores, activated via the `background` flag.

        Parameters
        ----------
        batches : imgaug.augmentables.batches.Batch or imgaug.augmentables.batches.UnnormalizedBatch or iterable of imgaug.augmentables.batches.Batch or iterable of imgaug.augmentables.batches.UnnormalizedBatch
            A single batch or a list of batches to augment.

        hooks : None or imgaug.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation
            process.

        background : bool, optional
            Whether to augment the batches in background processes.
            If ``True``, hooks can currently not be used as that would require
            pickling functions.
            Note that multicore augmentation distributes the batches onto
            different CPU cores. It does *not* split the data *within* batches.
            It is therefore *not* sensible to use ``background=True`` to
            augment a single batch. Only use it for multiple batches.
            Note also that multicore augmentation needs some time to start. It
            is therefore not recommended to use it for very few batches.

        Yields
        -------
        imgaug.augmentables.batches.Batch or imgaug.augmentables.batches.UnnormalizedBatch or iterable of imgaug.augmentables.batches.Batch or iterable of imgaug.augmentables.batches.UnnormalizedBatch
            Augmented batches.

        """
        if isinstance(batches, (Batch, UnnormalizedBatch)):
            batches = [batches]

        assert (
            (ia.is_iterable(batches)
             and not ia.is_np_array(batches)
             and not ia.is_string(batches))
            or ia.is_generator(batches)), (
                "Expected either (a) an iterable that is not an array or a "
                "string or (b) a generator. Got: %s" % (type(batches),))

        if background:
            assert hooks is None, (
                "Hooks can not be used when background augmentation is "
                "activated.")

        def _normalize_batch(idx, batch):
            if isinstance(batch, Batch):
                batch_copy = batch.deepcopy()
                batch_copy.data = (idx, batch_copy.data)
                batch_normalized = batch_copy
                batch_orig_dt = "imgaug.Batch"
            elif isinstance(batch, UnnormalizedBatch):
                batch_copy = batch.to_normalized_batch()
                batch_copy.data = (idx, batch_copy.data)
                batch_normalized = batch_copy
                batch_orig_dt = "imgaug.UnnormalizedBatch"
            elif ia.is_np_array(batch):
                assert batch.ndim in (3, 4),(
                    "Expected numpy array to have shape (N, H, W) or "
                    "(N, H, W, C), got %s." % (batch.shape,))
                batch_normalized = Batch(images=batch, data=(idx,))
                batch_orig_dt = "numpy_array"
            elif isinstance(batch, list):
                if len(batch) == 0:
                    batch_normalized = Batch(data=(idx,))
                    batch_orig_dt = "empty_list"
                elif ia.is_np_array(batch[0]):
                    batch_normalized = Batch(images=batch, data=(idx,))
                    batch_orig_dt = "list_of_numpy_arrays"
                elif isinstance(batch[0], ia.HeatmapsOnImage):
                    batch_normalized = Batch(heatmaps=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.HeatmapsOnImage"
                elif isinstance(batch[0], ia.SegmentationMapsOnImage):
                    batch_normalized = Batch(segmentation_maps=batch,
                                             data=(idx,))
                    batch_orig_dt = "list_of_imgaug.SegmentationMapsOnImage"
                elif isinstance(batch[0], ia.KeypointsOnImage):
                    batch_normalized = Batch(keypoints=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.KeypointsOnImage"
                elif isinstance(batch[0], ia.BoundingBoxesOnImage):
                    batch_normalized = Batch(bounding_boxes=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.BoundingBoxesOnImage"
                elif isinstance(batch[0], ia.PolygonsOnImage):
                    batch_normalized = Batch(polygons=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.PolygonsOnImage"
                else:
                    raise Exception(
                        "Unknown datatype in batch[0]. Expected numpy array "
                        "or imgaug.HeatmapsOnImage or "
                        "imgaug.SegmentationMapsOnImage or "
                        "imgaug.KeypointsOnImage or "
                        "imgaug.BoundingBoxesOnImage, "
                        "or imgaug.PolygonsOnImage, "
                        "got %s." % (type(batch[0]),))
            else:
                raise Exception(
                    "Unknown datatype of batch. Expected imgaug.Batch or "
                    "imgaug.UnnormalizedBatch or "
                    "numpy array or list of (numpy array or "
                    "imgaug.HeatmapsOnImage or "
                    "imgaug.SegmentationMapsOnImage "
                    "or imgaug.KeypointsOnImage or "
                    "imgaug.BoundingBoxesOnImage or "
                    "imgaug.PolygonsOnImage). Got %s." % (type(batch),))

            if batch_orig_dt not in ["imgaug.Batch",
                                     "imgaug.UnnormalizedBatch"]:
                ia.warn_deprecated(
                    "Received an input in augment_batches() that was not an "
                    "instance of imgaug.augmentables.batches.Batch "
                    "or imgaug.augmentables.batches.UnnormalizedBatch, but "
                    "instead %s. This is deprecated. Use augment() for such "
                    "data or wrap it in a Batch instance." % (
                        batch_orig_dt,))
            return batch_normalized, batch_orig_dt

        # unnormalization of non-Batch/UnnormalizedBatch is for legacy support
        def _unnormalize_batch(batch_aug, batch_orig, batch_orig_dt):
            if batch_orig_dt == "imgaug.Batch":
                batch_unnormalized = batch_aug
                # change (i, .data) back to just .data
                batch_unnormalized.data = batch_unnormalized.data[1]
            elif batch_orig_dt == "imgaug.UnnormalizedBatch":
                # change (i, .data) back to just .data
                batch_aug.data = batch_aug.data[1]

                batch_unnormalized = \
                    batch_orig.fill_from_augmented_normalized_batch(batch_aug)
            elif batch_orig_dt == "numpy_array":
                batch_unnormalized = batch_aug.images_aug
            elif batch_orig_dt == "empty_list":
                batch_unnormalized = []
            elif batch_orig_dt == "list_of_numpy_arrays":
                batch_unnormalized = batch_aug.images_aug
            elif batch_orig_dt == "list_of_imgaug.HeatmapsOnImage":
                batch_unnormalized = batch_aug.heatmaps_aug
            elif batch_orig_dt == "list_of_imgaug.SegmentationMapsOnImage":
                batch_unnormalized = batch_aug.segmentation_maps_aug
            elif batch_orig_dt == "list_of_imgaug.KeypointsOnImage":
                batch_unnormalized = batch_aug.keypoints_aug
            elif batch_orig_dt == "list_of_imgaug.BoundingBoxesOnImage":
                batch_unnormalized = batch_aug.bounding_boxes_aug
            else:  # only option left
                assert batch_orig_dt == "list_of_imgaug.PolygonsOnImage", (
                    "Got an unexpected type %s." % (type(batch_orig_dt),))
                batch_unnormalized = batch_aug.polygons_aug
            return batch_unnormalized

        if not background:
            # singlecore augmentation

            for idx, batch in enumerate(batches):
                batch_normalized, batch_orig_dt = _normalize_batch(idx, batch)
                batch_normalized = self.augment_batch(
                    batch_normalized, hooks=hooks)
                batch_unnormalized = _unnormalize_batch(
                    batch_normalized, batch, batch_orig_dt)

                yield batch_unnormalized
        else:
            # multicore augmentation
            import imgaug.multicore as multicore

            id_to_batch_orig = dict()

            def load_batches():
                for idx, batch in enumerate(batches):
                    batch_normalized, batch_orig_dt = _normalize_batch(
                        idx, batch)
                    id_to_batch_orig[idx] = (batch, batch_orig_dt)
                    yield batch_normalized

            with multicore.Pool(self) as pool:
                for batch_aug in pool.imap_batches(load_batches()):
                    idx = batch_aug.data[0]
                    assert idx in id_to_batch_orig, (
                        "Got idx %d from Pool, which is not known." % (
                            idx))
                    batch_orig, batch_orig_dt = id_to_batch_orig[idx]
                    batch_unnormalized = _unnormalize_batch(
                        batch_aug, batch_orig, batch_orig_dt)
                    del id_to_batch_orig[idx]
                    yield batch_unnormalized

    def augment_batch(self, batch, hooks=None):
        """
        Augment a single batch.

        Parameters
        ----------
        batch : imgaug.augmentables.batches.Batch or imgaug.augmentables.batches.UnnormalizedBatch
            A single batch to augment.

        hooks : None or imgaug.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        imgaug.augmentables.batches.Batch or imgaug.augmentables.batches.UnnormalizedBatch
            Augmented batch.

        """
        batch_orig = batch
        if isinstance(batch, UnnormalizedBatch):
            batch = batch.to_normalized_batch()

        augmentables = [(attr_name[:-len("_unaug")], attr)
                        for attr_name, attr
                        in batch.__dict__.items()
                        if attr_name.endswith("_unaug") and attr is not None]

        augseq = self
        if len(augmentables) > 1 and not self.deterministic:
            augseq = self.to_deterministic()

        # set attribute batch.T_aug with result of self.augment_T() for each
        # batch.T_unaug that was not None
        for attr_name, attr in augmentables:
            aug = getattr(augseq, "augment_%s" % (attr_name,))(
                attr, hooks=hooks)
            setattr(batch, "%s_aug" % (attr_name,), aug)

        if isinstance(batch_orig, UnnormalizedBatch):
            batch = batch_orig.fill_from_augmented_normalized_batch(batch)
        return batch

    # TODO is that used by augment_batches()?
    # TODO should this simply be removed?
    def _is_activated_with_hooks(self, augmentables, parents, hooks):
        is_activated = (
            (hooks is None and self.activated)
            or (
                hooks is not None
                and hooks.is_activated(
                    augmentables, augmenter=self, parents=parents,
                    default=self.activated)
            )
        )
        return is_activated

    def augment_image(self, image, hooks=None):
        """Augment a single image.

        Parameters
        ----------
        image : (H,W,C) ndarray or (H,W) ndarray
            The image to augment.
            Channel-axis is optional, but expected to be the last axis if
            present. In most cases, this array should be of dtype ``uint8``,
            which is supported by all augmenters. Support for other dtypes
            varies by augmenter -- see the respective augmenter-specific
            documentation for more details.

        hooks : None or imgaug.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        img : ndarray
            The corresponding augmented image.

        """
        assert image.ndim in [2, 3], (
            "Expected image to have shape (height, width, [channels]), "
            "got shape %s." % (image.shape,))
        return self.augment_images([image], hooks=hooks)[0]

    def augment_images(self, images, parents=None, hooks=None):
        """Augment a batch of images.

        Parameters
        ----------
        images : (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            Images to augment.
            The input can be a list of numpy arrays or a single array. Each
            array is expected to have shape ``(H, W, C)`` or ``(H, W)``,
            where ``H`` is the height, ``W`` is the width and ``C`` are the
            channels. The number of channels may differ between images.
            If a list is provided, the height, width and channels may differ
            between images within the provided batch.
            In most cases, the image array(s) should be of dtype ``uint8``,
            which is supported by all augmenters. Support for other dtypes
            varies by augmenter -- see the respective augmenter-specific
            documentation for more details.

        parents : None or list of imgaug.augmenters.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug.imgaug.HooksImages, optional
            :class:`imgaug.imgaug.HooksImages` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        ndarray or list
            Corresponding augmented images.
            If the input was an ``ndarray``, the output is also an ``ndarray``,
            unless the used augmentations have led to different output image
            sizes (as can happen in e.g. cropping).

        Examples
        --------
        >>> import imgaug.augmenters as iaa
        >>> import numpy as np
        >>> aug = iaa.GaussianBlur((0.0, 3.0))
        >>> # create empty example images
        >>> images = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        >>> images_aug = aug.augment_images(images)

        Create ``2`` empty (i.e. black) example numpy images and apply
        gaussian blurring to them.

        """
        if parents is not None and len(parents) > 0 and hooks is None:
            # This is a child call. The data has already been validated and
            # copied. We don't need to copy it again for hooks, as these
            # don't exist. So we can augment here fully in-place.
            if not self.activated or len(images) == 0:
                return images

            if self.deterministic:
                state_orig = self.random_state.state

            images_result = self._augment_images(
                images,
                random_state=self.random_state,
                parents=parents,
                hooks=hooks
            )
            # move "forward" the random state, so that the next call to
            # augment_images() will use different random values
            # This is currently deactivated as the RNG is no longer copied
            # for the _augment_* call.
            # self.random_state.advance_()

            if self.deterministic:
                self.random_state.set_state_(state_orig)

            return images_result

        #
        # Everything below is for non-in-place augmentation.
        # It was either the first call (no parents) or hooks were provided.
        #
        if self.deterministic:
            state_orig = self.random_state.state

        if parents is None:
            parents = []

        if ia.is_np_array(images):
            input_type = "array"
            input_added_axis = False

            assert images.ndim in [3, 4], (
                "Expected 3d/4d array of form (N, height, width) or (N, "
                "height, width, channels), got shape %s." % (images.shape,))

            # copy the input, we don't want to augment it in-place
            images_copy = np.copy(images)

            if images_copy.ndim == 3 and images_copy.shape[-1] in [1, 3]:
                ia.warn(
                    "You provided a numpy array of shape %s as input to "
                    "augment_images(), which was interpreted as (N, H, W). "
                    "The last dimension however has value 1 or 3, which "
                    "indicates that you provided a single image with shape "
                    "(H, W, C) instead. If that is the case, you should use "
                    "augment_image(image) or augment_images([image]), "
                    "otherwise you will not get the expected "
                    "augmentations." % (images_copy.shape,))

            # for 2D input images (i.e. shape (N, H, W)), we add a channel
            # axis (i.e. (N, H, W, 1)), so that all augmenters can rely on
            # the input having a channel axis and don't have to add if/else
            # statements for 2D images
            if images_copy.ndim == 3:
                images_copy = images_copy[..., np.newaxis]
                input_added_axis = True
        elif ia.is_iterable(images):
            input_type = "list"
            input_added_axis = []

            if len(images) == 0:
                images_copy = []
            else:
                assert all(image.ndim in [2, 3] for image in images), (
                    "Expected list of images with each image having shape "
                    "(height, width) or (height, width, channels), got "
                    "shapes %s." % ([image.shape for image in images],))

                # copy images and add channel axis for 2D images (see above,
                # as for list inputs each image can have different shape, it
                # is done here on a per images basis)
                images_copy = []
                input_added_axis = []
                for image in images:
                    image_copy = np.copy(image)
                    if image.ndim == 2:
                        image_copy = image_copy[:, :, np.newaxis]
                        input_added_axis.append(True)
                    else:
                        input_added_axis.append(False)
                    images_copy.append(image_copy)
        else:
            raise Exception(
                "Expected images as one numpy array or list/tuple of numpy "
                "arrays, got %s." % (
                    type(images),))

        if hooks is not None:
            images_copy = hooks.preprocess(images_copy, augmenter=self,
                                           parents=parents)

        # the is_activated() call allows to use hooks that selectively
        # deactivate specific augmenters in previously defined augmentation
        # sequences
        if self._is_activated_with_hooks(images_copy, parents, hooks):
            if len(images) > 0:
                images_result = self._augment_images(
                    images_copy,
                    random_state=self.random_state,
                    parents=parents,
                    hooks=hooks
                )
                # move "forward" the random state, so that the next call to
                # augment_images() will use different random values
                # This is currently deactivated as the RNG is no longer copied
                # for the _augment_* call.
                # self.random_state.advance_()
            else:
                images_result = images_copy
        else:
            images_result = images_copy

        if hooks is not None:
            images_result = hooks.postprocess(images_result, augmenter=self,
                                              parents=parents)

        # remove temporarily added channel axis for 2D input images
        output_type = "list" if isinstance(images_result, list) else "array"
        if input_type == "array":
            if input_added_axis is True:
                if output_type == "array":
                    images_result = np.squeeze(images_result, axis=3)
                else:
                    images_result = [np.squeeze(image, axis=2)
                                     for image in images_result]
        else:  # input_type == "list"
            assert len(images_result) == len(images), (
                "INTERNAL ERROR: Expected number of images to be unchanged "
                "after augmentation, but was changed from %d to %d." % (
                    len(images), len(images_result)))
            for i in sm.xrange(len(images_result)):
                if input_added_axis[i] is True:
                    images_result[i] = np.squeeze(images_result[i], axis=2)

        if self.deterministic:
            self.random_state.set_state_(state_orig)

        return images_result

    @abstractmethod
    def _augment_images(self, images, random_state, parents, hooks):
        """Augment a batch of images in-place.

        This is the internal version of :func:`Augmenter.augment_images`.
        It is called from :func:`Augmenter.augment_images` and should usually
        not be called directly.
        It has to be implemented by every augmenter.
        This method may transform the images in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        Parameters
        ----------
        images : (N,H,W,C) ndarray or list of (H,W,C) ndarray
            Images to augment.
            They may be changed in-place.
            Either a list of ``(H, W, C)`` arrays or a single ``(N, H, W, C)``
            array, where ``N`` is the number of images, ``H`` is the height of
            images, ``W`` is the width of images and ``C`` is the number of
            channels of images. In the case of a list as input, ``H``, ``W``
            and ``C`` may change per image.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`imgaug.augmenters.meta.Augmenter.augment_images`.

        hooks : imgaug.imgaug.HooksImages or None
            See :func:`imgaug.augmenters.meta.Augmenter.augment_images`.

        Returns
        ----------
        images : (N,H,W,C) ndarray or list of (H,W,C) ndarray
            The augmented images.

        """
        raise NotImplementedError()

    def augment_heatmaps(self, heatmaps, parents=None, hooks=None):
        """Augment a batch of heatmaps.

        Parameters
        ----------
        heatmaps : imgaug.augmentables.heatmaps.HeatmapsOnImage or list of imgaug.augmentables.heatmaps.HeatmapsOnImage
            Heatmap(s) to augment. Either a single heatmap or a list of
            heatmaps.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``.
            It is set automatically for child augmenters.

        hooks : None or imaug.imgaug.HooksHeatmaps, optional
            :class:`imgaug.imgaug.HooksHeatmaps` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage or list of imgaug.augmentables.heatmaps.HeatmapsOnImage
            Corresponding augmented heatmap(s).

        """
        if self.deterministic:
            state_orig = self.random_state.state

        if parents is None:
            parents = []

        input_was_single_instance = False
        if isinstance(heatmaps, ia.HeatmapsOnImage):
            input_was_single_instance = True
            heatmaps = [heatmaps]

        assert ia.is_iterable(heatmaps), (
            "Expected to get list of imgaug.HeatmapsOnImage() instances, "
            "got %s." % (type(heatmaps),))
        only_heatmaps = all([isinstance(heatmaps_i, ia.HeatmapsOnImage)
                             for heatmaps_i in heatmaps])
        assert only_heatmaps, (
            "Expected to get list of imgaug.HeatmapsOnImage() instances, "
            "got %s." % ([type(el) for el in heatmaps],))

        # copy, but only if topmost call or hooks are provided
        if len(parents) == 0 or hooks is not None:
            heatmaps_copy = [heatmaps_i.deepcopy() for heatmaps_i in heatmaps]
        else:
            heatmaps_copy = heatmaps

        if hooks is not None:
            heatmaps_copy = hooks.preprocess(heatmaps_copy, augmenter=self,
                                             parents=parents)

        if self._is_activated_with_hooks(heatmaps_copy, parents, hooks):
            if len(heatmaps_copy) > 0:
                heatmaps_result = self._augment_heatmaps(
                    heatmaps_copy,
                    random_state=self.random_state,
                    parents=parents,
                    hooks=hooks
                )
                # self.random_state.advance_()
            else:
                heatmaps_result = heatmaps_copy
        else:
            heatmaps_result = heatmaps_copy

        if hooks is not None:
            heatmaps_result = hooks.postprocess(
                heatmaps_result, augmenter=self, parents=parents)

        if self.deterministic:
            self.random_state.set_state_(state_orig)

        if input_was_single_instance:
            return heatmaps_result[0]
        return heatmaps_result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        """Augment a batch of heatmaps in-place.

        This is the internal version of :func:`Augmenter.augment_heatmaps`.
        It is called from :func:`Augmenter.augment_heatmaps` and should
        usually not be called directly.
        This method may augment heatmaps in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        Parameters
        ----------
        heatmaps : list of imgaug.augmentables.heatmaps.HeatmapsOnImage
            Heatmaps to augment. They may be changed in-place.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`imgaug.augmenters.meta.Augmenter.augment_heatmaps`.

        hooks : imgaug.imgaug.HooksHeatmaps or None
            See :func:`imgaug.augmenters.meta.Augmenter.augment_heatmaps`.

        Returns
        ----------
        images : list of imgaug.augmentables.heatmaps.HeatmapsOnImage
            The augmented heatmaps.

        """
        return heatmaps

    def _augment_heatmaps_as_images(self, heatmaps, parents, hooks):
        # TODO documentation
        # TODO keep this? it is afaik not used anywhere
        heatmaps_uint8 = [heatmaps_i.to_uint8() for heatmaps_i in heatmaps]
        heatmaps_uint8_aug = [
            self.augment_images([heatmaps_uint8_i],
                                parents=parents, hooks=hooks)[0]
            for heatmaps_uint8_i
            in heatmaps_uint8
        ]
        return [
            ia.HeatmapsOnImage.from_uint8(
                heatmaps_aug,
                shape=heatmaps_i.shape,
                min_value=heatmaps_i.min_value,
                max_value=heatmaps_i.max_value
            )
            for heatmaps_aug, heatmaps_i
            in zip(heatmaps_uint8_aug, heatmaps)
        ]

    def augment_segmentation_maps(self, segmaps, parents=None, hooks=None):
        """Augment a batch of segmentation maps.

        Parameters
        ----------
        segmaps : imgaug.augmentables.segmaps.SegmentationMapsOnImage or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Segmentation map(s) to augment. Either a single segmentation map
            or a list of segmentation maps.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug.HooksHeatmaps, optional
            :class:`imgaug.imgaug.HooksHeatmaps` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.segmaps.SegmentationMapsOnImage or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Corresponding augmented segmentation map(s).

        """
        if self.deterministic:
            state_orig = self.random_state.state

        if parents is None:
            parents = []

        input_was_single_instance = False
        if isinstance(segmaps, ia.SegmentationMapsOnImage):
            input_was_single_instance = True
            segmaps = [segmaps]

        assert ia.is_iterable(segmaps), (
            "Expected to get list of imgaug.SegmentationMapsOnImage() "
            "instances, got %s." % (type(segmaps),))
        only_segmaps = all(
            [isinstance(segmaps_i, ia.SegmentationMapsOnImage)
             for segmaps_i in segmaps])
        assert only_segmaps, (
            "Expected to get list of imgaug.SegmentationMapsOnImage() "
            "instances, got %s." % ([type(el) for el in segmaps],))

        # copy, but only if topmost call or hooks are provided
        if len(parents) == 0 or hooks is not None:
            segmaps_copy = [segmaps_i.deepcopy() for segmaps_i in segmaps]
        else:
            segmaps_copy = segmaps

        if hooks is not None:
            segmaps_copy = hooks.preprocess(segmaps_copy, augmenter=self, parents=parents)

        if self._is_activated_with_hooks(segmaps_copy, parents, hooks):
            if len(segmaps_copy) > 0:
                segmaps_result = self._augment_segmentation_maps(
                    segmaps_copy,
                    random_state=self.random_state,
                    parents=parents,
                    hooks=hooks
                )
                # self.random_state.advance_()
            else:
                segmaps_result = segmaps_copy
        else:
            segmaps_result = segmaps_copy

        if hooks is not None:
            segmaps_result = hooks.postprocess(segmaps_result, augmenter=self,
                                               parents=parents)

        if self.deterministic:
            self.random_state.set_state_(state_orig)

        if input_was_single_instance:
            return segmaps_result[0]
        return segmaps_result

    def _augment_segmentation_maps(self, segmaps, random_state, parents, hooks):
        """Augment a batch of segmentation in-place.

        This is the internal version of
        :func:`Augmenter.augment_segmentation_maps`.
        It is called from :func:`Augmenter.augment_segmentation_maps` and
        should usually not be called directly.
        This method may augment segmentation maps in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        Parameters
        ----------
        segmaps : list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Segmentation maps to augment. They may be changed in-place.

        parents : list of imgaug.augmenters.meta.Augmenter
            See
            :func:`imgaug.augmenters.meta.Augmenter.augment_segmentation_maps`.

        hooks : imgaug.imgaug.HooksHeatmaps or None
            See
            :func:`imgaug.augmenters.meta.Augmenter.augment_segmentation_maps`.

        Returns
        ----------
        images : list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
            The augmented segmentation maps.

        """
        return segmaps

    def augment_keypoints(self, keypoints_on_images, parents=None, hooks=None):
        """Augment a batch of keypoints/landmarks.

        This is the corresponding function to :func:`Augmenter.augment_images`,
        just for keypoints/landmarks (i.e. points on images).
        Usually you will want to call :func:`Augmenter.augment_images` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_keypoints()`` with the corresponding list of keypoints on
        these images, e.g. ``augment_keypoints([Ak, Bk, Ck])``, where ``Ak``
        are the keypoints on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding keypoints,
        e.g. by

        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.kps import Keypoint
        >>> from imgaug.augmentables.kps import KeypointsOnImage
        >>> A = B = C = np.zeros((10, 10), dtype=np.uint8)
        >>> Ak = Bk = Ck = KeypointsOnImage([Keypoint(2, 2)], (10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> kps_aug = seq_det.augment_keypoints([Ak, Bk, Ck])

        Otherwise, different random values will be sampled for the image
        and keypoint augmentations, resulting in different augmentations (e.g.
        images might be rotated by ``30deg`` and keypoints by ``-10deg``).
        Also make sure to call :func:`Augmenter.to_deterministic` again for
        each new batch, otherwise you would augment all batches in the same
        way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        keypoints_on_images : imgaug.augmentables.kps.KeypointsOnImage or list of imgaug.augmentables.kps.KeypointsOnImage
            The keypoints/landmarks to augment.
            Either a single instance of
            :class:`imgaug.augmentables.kps.KeypointsOnImage` or a list of
            such instances. Each instance must contain the keypoints of a
            single image.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug.imgaug.HooksKeypoints, optional
            :class:`imgaug.imgaug.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage or list of imgaug.augmentables.kps.KeypointsOnImage
            Augmented keypoints.

        """
        if self.deterministic:
            state_orig = self.random_state.state

        if parents is None:
            parents = []

        input_was_single_instance = False
        if isinstance(keypoints_on_images, ia.KeypointsOnImage):
            input_was_single_instance = True
            keypoints_on_images = [keypoints_on_images]

        assert ia.is_iterable(keypoints_on_images), (
            "Expected to get list of imgaug.KeypointsOnImage() "
            "instances, got %s." % (type(keypoints_on_images),))
        only_keypoints = all(
            [isinstance(keypoints_on_image, ia.KeypointsOnImage)
             for keypoints_on_image in keypoints_on_images])
        assert only_keypoints, (
            "Expected to get list of imgaug.KeypointsOnImage() "
            "instances, got %s." % ([type(el) for el in keypoints_on_images],))

        # copy, but only if topmost call or hooks are provided
        if len(parents) == 0 or hooks is not None:
            keypoints_on_images_copy = [keypoints_on_image.deepcopy()
                                        for keypoints_on_image
                                        in keypoints_on_images]
        else:
            keypoints_on_images_copy = keypoints_on_images

        if hooks is not None:
            keypoints_on_images_copy = hooks.preprocess(keypoints_on_images_copy, augmenter=self, parents=parents)

        if self._is_activated_with_hooks(keypoints_on_images_copy, parents,
                                         hooks):
            if len(keypoints_on_images_copy) > 0:
                keypoints_on_images_result = self._augment_keypoints(
                    keypoints_on_images_copy,
                    random_state=self.random_state,
                    parents=parents,
                    hooks=hooks
                )
                # self.random_state.advance_()
            else:
                keypoints_on_images_result = keypoints_on_images_copy
        else:
            keypoints_on_images_result = keypoints_on_images_copy

        if hooks is not None:
            keypoints_on_images_result = hooks.postprocess(
                keypoints_on_images_result, augmenter=self, parents=parents)

        if self.deterministic:
            self.random_state.set_state_(state_orig)

        if input_was_single_instance:
            return keypoints_on_images_result[0]
        return keypoints_on_images_result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        """Augment a batch of keypoints in-place.

        This is the internal version of :func:`Augmenter.augment_keypoints`.
        It is called from :func:`Augmenter.augment_keypoints` and should
        usually not be called directly.
        This method may transform the keypoints in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        Parameters
        ----------
        keypoints_on_images : list of imgaug.augmentables.kps.KeypointsOnImage
            Keypoints to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`imgaug.augmenters.meta.Augmenter.augment_keypoints`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`imgaug.augmenters.meta.Augmenter.augment_keypoints`.

        Returns
        ----------
        list of imgaug.augmentables.kps.KeypointsOnImage
            The augmented keypoints.

        """
        return keypoints_on_images

    def augment_bounding_boxes(self, bounding_boxes_on_images, hooks=None):
        """Augment a batch of bounding boxes.

        This is the corresponding function to
        :func:`Augmenter.augment_images`, just for bounding boxes.
        Usually you will want to call :func:`Augmenter.augment_images` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_bounding_boxes()`` with the corresponding list of bounding
        boxes on these images, e.g.
        ``augment_bounding_boxes([Abb, Bbb, Cbb])``, where ``Abb`` are the
        bounding boxes on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding bounding boxes,
        e.g. by

        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.bbs import BoundingBox
        >>> from imgaug.augmentables.bbs import BoundingBoxesOnImage
        >>> A = B = C = np.ones((10, 10), dtype=np.uint8)
        >>> Abb = Bbb = Cbb = BoundingBoxesOnImage([
        >>>     BoundingBox(1, 1, 9, 9)], (10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> bbs_aug = seq_det.augment_bounding_boxes([Abb, Bbb, Cbb])

        Otherwise, different random values will be sampled for the image
        and bounding box augmentations, resulting in different augmentations
        (e.g. images might be rotated by ``30deg`` and bounding boxes by
        ``-10deg``). Also make sure to call :func:`Augmenter.to_deterministic`
        again for each new batch, otherwise you would augment all batches in
        the same way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        bounding_boxes_on_images : imgaug.augmentables.bbs.BoundingBoxesOnImage or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
            The bounding boxes to augment.
            Either a single instance of
            :class:`imgaug.augmentables.bbs.BoundingBoxesOnImage` or a list of
            such instances, with each one of them containing the bounding
            boxes of a single image.

        hooks : None or imgaug.imgaug.HooksKeypoints, optional
            :class:`imgaug.imgaug.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBoxesOnImage or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
            Augmented bounding boxes.

        """
        input_was_single_instance = False
        if isinstance(bounding_boxes_on_images, ia.BoundingBoxesOnImage):
            input_was_single_instance = True
            bounding_boxes_on_images = [bounding_boxes_on_images]

        kps_ois = []
        for bbs_oi in bounding_boxes_on_images:
            kps = []
            for bb in bbs_oi.bounding_boxes:
                kps.extend(bb.to_keypoints())
            kps_ois.append(ia.KeypointsOnImage(kps, shape=bbs_oi.shape))

        kps_ois_aug = self.augment_keypoints(kps_ois, hooks=hooks)

        result = []
        for img_idx, kps_oi_aug in enumerate(kps_ois_aug):
            bbs_aug = []
            for i in sm.xrange(len(kps_oi_aug.keypoints) // 4):
                bb_kps = kps_oi_aug.keypoints[i*4:i*4+4]
                x1 = min([kp.x for kp in bb_kps])
                x2 = max([kp.x for kp in bb_kps])
                y1 = min([kp.y for kp in bb_kps])
                y2 = max([kp.y for kp in bb_kps])
                bbs_aug.append(
                    bounding_boxes_on_images[img_idx].bounding_boxes[i].copy(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2
                    )
                )
            result.append(
                ia.BoundingBoxesOnImage(
                    bbs_aug,
                    shape=kps_oi_aug.shape
                )
            )
        if input_was_single_instance:
            return result[0]
        return result

    def augment_polygons(self, polygons_on_images, parents=None, hooks=None):
        """Augment a batch of polygons.

        This is the corresponding function to :func:`Augmenter.augment_images`,
        just for polygons.
        Usually you will want to call :func:`Augmenter.augment_images`` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_polygons()`` with the corresponding list of polygons on these
        images, e.g. ``augment_polygons([A_poly, B_poly, C_poly])``, where
        ``A_poly`` are the polygons on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding polygons,
        e.g. by

        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.polys import Polygon, PolygonsOnImage
        >>> A = B = C = np.ones((10, 10), dtype=np.uint8)
        >>> Apoly = Bpoly = Cpoly = PolygonsOnImage(
        >>>     [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        >>>     shape=(10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> polys_aug = seq_det.augment_polygons([Apoly, Bpoly, Cpoly])

        Otherwise, different random values will be sampled for the image
        and polygon augmentations, resulting in different augmentations
        (e.g. images might be rotated by ``30deg`` and polygons by
        ``-10deg``). Also make sure to call ``to_deterministic()`` again for
        each new batch, otherwise you would augment all batches in the same
        way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        polygons_on_images : imgaug.augmentables.polys.PolygonsOnImage or list of imgaug.augmentables.polys.PolygonsOnImage
            The polygons to augment.
            Either a single instance of
            :class:`imgaug.augmentables.polys.PolygonsOnImage` or a list of
            such instances, with each one of them containing the polygons of
            a single image.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug.imgaug.HooksKeypoints, optional
            :class:`imgaug.imgaug.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage or list of imgaug.augmentables.polys.PolygonsOnImage
            Augmented polygons.

        """
        from imgaug.augmentables.polys import PolygonsOnImage

        def _subaugment(polygons_on_images_, random_state_, parents_, hooks_):
            return self._augment_polygons(
                polygons_on_images_,
                random_state=random_state_,
                parents=parents_,
                hooks=hooks_
            )

        return self._augment_coord_augables(
            cls_expected=PolygonsOnImage,
            subaugment_func=_subaugment,
            augables_ois=polygons_on_images,
            parents=parents,
            hooks=hooks
        )

    def augment_line_strings(self, line_strings_on_images, parents=None,
                             hooks=None):
        """Augment a batch of line strings.

        This is the corresponding function to
        :func:`Augmenter.augment_images``, just for line strings.
        Usually you will want to call :func:`Augmenter.augment_images` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_line_strings()`` with the corresponding list of line
        strings on these images, e.g.
        ``augment_line_strings([A_line, B_line, C_line])``, where ``A_line``
        are the line strings on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding line strings,
        e.g. by

        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.lines import LineString
        >>> from imgaug.augmentables.lines import LineStringsOnImage
        >>> A = B = C = np.ones((10, 10), dtype=np.uint8)
        >>> A_line = B_line = C_line = LineStringsOnImage(
        >>>     [LineString([(0, 0), (1, 0), (1, 1), (0, 1)])],
        >>>     shape=(10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> lines_aug = seq_det.augment_line_strings([A_line, B_line, C_line])

        Otherwise, different random values will be sampled for the image
        and line string augmentations, resulting in different augmentations
        (e.g. images might be rotated by ``30deg`` and line strings by
        ``-10deg``). Also make sure to call ``to_deterministic()`` again for
        each new batch, otherwise you would augment all batches in the same
        way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        line_strings_on_images : imgaug.augmentables.lines.LineStringsOnImage or list of imgaug.augmentables.lines.LineStringsOnImage
            The line strings to augment.
            Either a single instance of
            :class:`imgaug.augmentables.lines.LineStringsOnImage` or a list of
            such instances, with each one of them containing the line strings
            of a single image.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : None or imgaug.imgaug.HooksKeypoints, optional
            :class:`imgaug.imgaug.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.lines.LineStringsOnImage or list of imgaug.augmentables.lines.LineStringsOnImage
            Augmented line strings.

        """
        from imgaug.augmentables.lines import LineStringsOnImage

        def _subaugment(line_strings_on_images_, random_state_, parents_, hooks_):
            return self._augment_line_strings(
                line_strings_on_images_,
                random_state=random_state_,
                parents=parents_,
                hooks=hooks_
            )

        return self._augment_coord_augables(
            cls_expected=LineStringsOnImage,
            subaugment_func=_subaugment,
            augables_ois=line_strings_on_images,
            parents=parents,
            hooks=hooks
        )

    def _augment_coord_augables(self, cls_expected, subaugment_func,
                                augables_ois, parents=None,
                                hooks=None):
        """Augment a batch of coordinate-based augmentables.

        This is an generic function called by keypoints, bounding boxes,
        polygons and line strings.
        TODO keypoints, bounding boxes currently missing -- add them

        Parameters
        ----------
        cls_expected : class
            Class type that is expected. `augmentables_ois` will be
            verified to use that class.

        subaugment_func : callable
            Function that will be called to actually augment the data.

        augables_ois : imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.lines.LineStringsOnImage or list of imgaug.augmentables.lines.LineStringsOnImage or list of imgaug.augmentables.polys.PolygonsOnImage
            The augmentables to augment. `augables_ois` is the abbreviation for
            "augmentables_on_images". Expected are the augmentables on a
            single image (single instance) or >=1 images (list of instances).

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : None or imgaug.imgaug.HooksKeypoints, optional
            :class:`imgaug.imgaug.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.lines.LineStringsOnImage or list of imgaug.augmentables.polys.PolygonsOnImage or list of imgaug.augmentables.lines.LineStringsOnImage
            Augmented augmentables.

        """
        if self.deterministic:
            state_orig = self.random_state.state

        if parents is None:
            parents = []

        input_was_single_instance = False
        if isinstance(augables_ois, cls_expected):
            input_was_single_instance = True
            augables_ois = [augables_ois]

        assert ia.is_iterable(augables_ois), (
            "Expected to get list of %s instances, got %s." % (
                cls_expected.__class__.__name__,
                type(augables_ois),))
        only_valid_types = all(
            [isinstance(augable_oi, cls_expected)
             for augable_oi in augables_ois])
        assert only_valid_types, (
            "Expected to get list of %s instances, got %s." % (
                cls_expected.__class__.__name__,
                [type(el) for el in augables_ois],))

        # copy, but only if topmost call or hooks are provided
        augables_ois_copy = augables_ois
        if len(parents) == 0 or hooks is not None:
            augables_ois_copy = [augable_oi.deepcopy()
                                 for augable_oi
                                 in augables_ois]

        if hooks is not None:
            augables_ois_copy = hooks.preprocess(
                augables_ois_copy, augmenter=self, parents=parents)

        augables_ois_result = augables_ois_copy
        if self._is_activated_with_hooks(augables_ois_copy, parents, hooks):
            if len(augables_ois) > 0:
                augables_ois_result = subaugment_func(
                    augables_ois_copy,
                    self.random_state,
                    parents,
                    hooks
                )
                # self.random_state.advance_()

        if hooks is not None:
            augables_ois_result = hooks.postprocess(
                augables_ois_result, augmenter=self, parents=parents)

        if self.deterministic:
            self.random_state.set_state_(state_orig)

        if input_was_single_instance:
            return augables_ois_result[0]
        return augables_ois_result

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        """Augment a batch of polygons on images in-place.

        This is the internal version of :func:`Augmenter.augment_polygons`.
        It is called from :func:`Augmenter.augment_polygons` and should
        usually not be called directly.
        This method may transform the polygons in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        Parameters
        ----------
        polygons_on_images : list of imgaug.PolygonsOnImage
            Polygons to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`imgaug.augmenters.meta.Augmenter.augment_polygons`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`imgaug.augmenters.meta.Augmenter.augment_polygons`.

        Returns
        ----------
        list of imgaug.augmentables.polys.PolygonsOnImage
            The augmented polygons.

        """
        return polygons_on_images

    def _augment_polygons_as_keypoints(self, polygons_on_images, random_state,
                                       parents, hooks, recoverer=None):
        """
        Augment polygons by applying keypoint augmentation to their vertices.

        Parameters
        ----------
        polygons_on_images : list of imgaug.augmentables.polys.PolygonsOnImage
            Polygons to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`imgaug.augmenters.meta.Augmenter.augment_polygons`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`imgaug.augmenters.meta.Augmenter.augment_polygons`.

        recoverer : None or imgaug.augmentables.polys._ConcavePolygonRecoverer
            An instance used to repair invalid polygons after augmentation.
            Must offer the method
            ``recover_from(new_exterior, old_polygon, random_state=0)``.
            If ``None`` then invalid polygons are not repaired.

        Returns
        ----------
        list of imgaug.augmentables.polys.PolygonsOnImage
            The augmented polygons.

        """
        from imgaug.augmentables.kps import KeypointsOnImage
        from imgaug.augmentables.polys import PolygonsOnImage

        kps_ois = []
        kp_counts = []
        for polys_oi in polygons_on_images:
            kps = []
            kp_counts_image = []
            for poly in polys_oi.polygons:
                poly_kps = poly.to_keypoints()
                kps.extend(poly_kps)
                kp_counts_image.append(len(poly_kps))
            kps_ois.append(KeypointsOnImage(kps, shape=polys_oi.shape))
            kp_counts.append(kp_counts_image)

        kps_ois_aug = self._augment_keypoints(kps_ois, random_state, parents,
                                              hooks)

        result = []
        gen = enumerate(zip(kps_ois_aug, kp_counts))
        for img_idx, (kps_oi_aug, kp_counts_image) in gen:
            polys_aug = []
            counter = 0
            for i, count in enumerate(kp_counts_image):
                poly_kps_aug = kps_oi_aug.keypoints[counter:counter+count]
                poly_old = polygons_on_images[img_idx].polygons[i]
                if recoverer is not None:
                    # make sure to not derive random state from random_state
                    # at the start of this function, otherwise random_state
                    # in _augment_keypoints() will be unaligned with images
                    poly_aug = recoverer.recover_from(
                        [(kp.x, kp.y) for kp in poly_kps_aug],
                        poly_old,
                        random_state=random_state)
                else:
                    poly_aug = poly_old.deepcopy(exterior=poly_kps_aug)
                polys_aug.append(poly_aug)
                counter += count
            result.append(PolygonsOnImage(polys_aug, shape=kps_oi_aug.shape))

        return result

    def _augment_line_strings(self, line_strings_on_images, random_state,
                              parents, hooks):
        """Augment a batch of line strings in-place.

        This is the internal version of
        :func:`Augmenter.augment_line_strings`.
        It is called from :func:`Augmenter.augment_line_strings` and should
        usually not be called directly.
        This method may transform the line strings in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        Parameters
        ----------
        line_strings_on_images : list of imgaug.augmentables.lines.LineStringsOnImage
            Line strings to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`imgaug.augmenters.meta.Augmenter.augment_line_strings`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`imgaug.augmenters.meta.Augmenter.augment_line_strings`.

        Returns
        ----------
        list of imgaug.augmentables.lines.LineStringsOnImage
            The augmented line strings.

        """
        # TODO this is very similar to the polygon augmentation method,
        #      merge somehow
        # TODO get rid of this deferred import:
        from imgaug.augmentables.kps import KeypointsOnImage
        from imgaug.augmentables.lines import LineStringsOnImage

        kps_ois = []
        kp_counts = []
        for ls_oi in line_strings_on_images:
            kps = []
            kp_counts_image = []
            for ls in ls_oi.line_strings:
                ls_kps = ls.to_keypoints()
                kps.extend(ls_kps)
                kp_counts_image.append(len(ls_kps))
            kps_ois.append(KeypointsOnImage(kps, shape=ls_oi.shape))
            kp_counts.append(kp_counts_image)

        kps_ois_aug = self._augment_keypoints(kps_ois, random_state, parents,
                                              hooks)

        result = []
        gen = enumerate(zip(kps_ois_aug, kp_counts))
        for img_idx, (kps_oi_aug, kp_counts_image) in gen:
            lss_aug = []
            counter = 0
            for i, count in enumerate(kp_counts_image):
                ls_kps_aug = kps_oi_aug.keypoints[counter:counter+count]
                ls_old = line_strings_on_images[img_idx].line_strings[i]
                ls_aug = ls_old.deepcopy(
                    coords=[(kp.x, kp.y) for kp in ls_kps_aug])
                lss_aug.append(ls_aug)
                counter += count
            result.append(LineStringsOnImage(lss_aug, shape=kps_oi_aug.shape))

        return result

    def augment(self, return_batch=False, hooks=None, **kwargs):
        """Augment a batch.

        This method is a wrapper around
        :class:`imgaug.augmentables.batches.UnnormalizedBatch` and
        :func:`imgaug.augmenters.meta.Augmenter.augment_batch`. Hence, it
        supports the same datatypes as
        :class:`imgaug.augmentables.batches.UnnormalizedBatch`.

        If `return_batch` was set to ``False`` (the default), the method will
        return a tuple of augmentables. It will return the same types of
        augmentables (but in augmented form) as input into the method. This
        behaviour is partly specific to the python version:

        * In **python 3.6+** (if ``return_batch=False``):

            * Any number of augmentables may be provided as input.
            * None of the provided named arguments *has to be* `image` or
              `images` (but of coarse you *may* provide them).
            * The return order matches the order of the named arguments, e.g.
              ``x_aug, y_aug, z_aug = augment(X=x, Y=y, Z=z)``.

        * In **python <3.6** (if ``return_batch=False``):

            * One or two augmentables may be used as input, not more than that.
            * One of the input arguments has to be `image` or `images`.
            * The augmented images are *always* returned first, independent
              of the input argument order, e.g.
              ``a_aug, b_aug = augment(b=b, images=a)``. This also means
              that the output of the function can only be one of the
              following three cases: a batch, list/array of images,
              tuple of images and something (like images + segmentation maps).

        If `return_batch` was set to ``True``, an instance of
        :class:`imgaug.augmentables.batches.UnnormalizedBatch` will be
        returned. The output is the same for all python version and any
        number or combination of augmentables may be provided.

        So, to keep code downward compatible for python <3.6, use one of the
        following three options:

          * Use ``batch = augment(images=X, ..., return_batch=True)``.
          * Call ``images = augment(images=X)``.
          * Call ``images, other = augment(images=X, <something_else>=Y)``.

        All augmentables must be provided as named arguments.
        E.g. ``augment(<array>)`` will crash, but ``augment(images=<array>)``
        will work.

        Parameters
        ----------
        image : None or (H,W,C) ndarray or (H,W) ndarray, optional
            The image to augment. Only this or `images` can be set, not both.
            If `return_batch` is ``False`` and the python version is below 3.6,
            either this or `images` **must** be provided.

        images : None or (N,H,W,C) ndarray or (N,H,W) ndarray or iterable of (H,W,C) ndarray or iterable of (H,W) ndarray, optional
            The images to augment. Only this or `image` can be set, not both.
            If `return_batch` is ``False`` and the python version is below 3.6,
            either this or `image` **must** be provided.

        heatmaps : None or (N,H,W,C) ndarray or imgaug.augmentables.heatmaps.HeatmapsOnImage or iterable of (H,W,C) ndarray or iterable of imgaug.augmentables.heatmaps.HeatmapsOnImage, optional
            The heatmaps to augment.
            If anything else than
            :class:`imgaug.augmentables.heatmaps.HeatmapsOnImage`, then the
            number of heatmaps must match the number of images provided via
            parameter `images`. The number is contained either in ``N`` or the
            first iterable's size.

        segmentation_maps : None or (N,H,W) ndarray or imgaug.augmentables.segmaps.SegmentationMapsOnImage or iterable of (H,W) ndarray or iterable of imgaug.augmentables.segmaps.SegmentationMapsOnImage, optional
            The segmentation maps to augment.
            If anything else than
            :class:`imgaug.augmentables.segmaps.SegmentationMapsOnImage`, then
            the number of segmaps must match the number of images provided via
            parameter `images`. The number is contained either in ``N`` or the
            first iterable's size.

        keypoints : None or list of (N,K,2) ndarray or tuple of number or imgaug.augmentables.kps.Keypoint or iterable of (K,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.kps.KeypointOnImage or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint, optional
            The keypoints to augment.
            If a tuple (or iterable(s) of tuple), then iterpreted as ``(x,y)``
            coordinates and must hence contain two numbers.
            A single tuple represents a single coordinate on one image, an
            iterable of tuples the coordinates on one image and an iterable of
            iterable of tuples the coordinates on several images. Analogous if
            :class:`imgaug.augmentables.kps.Keypoint` instances are used
            instead of tuples.
            If an ndarray, then ``N`` denotes the number of images and ``K``
            the number of keypoints on each image.
            If anything else than
            :class:`imgaug.augmentables.kps.KeypointsOnImage` is provided, then
            the number of keypoint groups must match the number of images
            provided via parameter `images`. The number is contained e.g. in
            ``N`` or in case of "iterable of iterable of tuples" in the first
            iterable's size.

        bounding_boxes : None or (N,B,4) ndarray or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage or iterable of (B,4) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.bbs.BoundingBox or iterable of imgaug.augmentables.bbs.BoundingBoxesOnImage or iterable of iterable of tuple of number or iterable of iterable imgaug.augmentables.bbs.BoundingBox, optional
            The bounding boxes to augment.
            This is analogous to the `keypoints` parameter. However, each
            tuple -- and also the last index in case of arrays -- has size
            ``4``, denoting the bounding box coordinates ``x1``, ``y1``,
            ``x2`` and ``y2``.

        polygons : None or (N,#polys,#points,2) ndarray or imgaug.augmentables.polys.Polygon or imgaug.augmentables.polys.PolygonsOnImage or iterable of (#polys,#points,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.polys.Polygon or iterable of imgaug.augmentables.polys.PolygonsOnImage or iterable of iterable of (#points,2) ndarray or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint or iterable of iterable of imgaug.augmentables.polys.Polygon or iterable of iterable of iterable of tuple of number or iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint, optional
            The polygons to augment.
            This is similar to the `keypoints` parameter. However, each polygon
            may be made up of several ``(x,y) ``coordinates (three or more are
            required for valid polygons).
            The following datatypes will be interpreted as a single polygon on
            a single image:

              * ``imgaug.augmentables.polys.Polygon``
              * ``iterable of tuple of number``
              * ``iterable of imgaug.augmentables.kps.Keypoint``

            The following datatypes will be interpreted as multiple polygons
            on a single image:

              * ``imgaug.augmentables.polys.PolygonsOnImage``
              * ``iterable of imgaug.augmentables.polys.Polygon``
              * ``iterable of iterable of tuple of number``
              * ``iterable of iterable of imgaug.augmentables.kps.Keypoint``
              * ``iterable of iterable of imgaug.augmentables.polys.Polygon``

            The following datatypes will be interpreted as multiple polygons on
            multiple images:

              * ``(N,#polys,#points,2) ndarray``
              * ``iterable of (#polys,#points,2) ndarray``
              * ``iterable of iterable of (#points,2) ndarray``
              * ``iterable of iterable of iterable of tuple of number``
              * ``iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint``

        line_strings : None or (N,#lines,#points,2) ndarray or imgaug.augmentables.lines.LineString or imgaug.augmentables.lines.LineStringOnImage or iterable of (#polys,#points,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.lines.LineString or iterable of imgaug.augmentables.lines.LineStringOnImage or iterable of iterable of (#points,2) ndarray or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint or iterable of iterable of imgaug.augmentables.lines.LineString or iterable of iterable of iterable of tuple of number or iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint, optional
            The line strings to augment.
            See `polygons`, which behaves similarly.

        return_batch : bool, optional
            Whether to return an instance of
            :class:`imgaug.augmentables.batches.UnnormalizedBatch`. If the
            python version is below 3.6 and more than two augmentables were
            provided (e.g. images, keypoints and polygons), then this must be
            set to ``True``. Otherwise an error will be raised.

        hooks : None or imgaug.imgaug.HooksImages, optional
            Hooks object to dynamically interfere with the augmentation process.

        Returns
        -------
        tuple or imgaug.augmentables.batches.UnnormalizedBatch
            If `return_batch` was set to ``True``, a instance of
            ``UnnormalizedBatch`` will be returned.
            If `return_batch` was set to ``False``, a tuple of augmentables
            will be returned, e.g. ``(augmented images, augmented keypoints)``.
            The datatypes match the input datatypes of the corresponding named
            arguments. In python <3.6, augmented images are always the first
            entry in the returned tuple. In python 3.6+ the order matches the
            order of the named arguments.

        Examples
        --------
        >>> import numpy as np
        >>> import imgaug as ia
        >>> import imgaug.augmenters as iaa
        >>> aug = iaa.Affine(rotate=(-25, 25))
        >>> image = np.zeros((64, 64, 3), dtype=np.uint8)
        >>> keypoints = [(10, 20), (30, 32)]  # (x,y) coordinates
        >>> images_aug, keypoints_aug = aug.augment(
        >>>     image=image, keypoints=keypoints)

        Create a single image and a set of two keypoints on it, then
        augment both by applying a random rotation between ``-25`` deg and
        ``+25`` deg. The sampled rotation value is automatically aligned
        between image and keypoints. Note that in python <3.6, augmented
        images will always be returned first, independent of the order of
        the named input arguments. So
        ``keypoints_aug, images_aug = aug.augment(keypoints=keypoints,
        image=image)`` would **not** be correct (but in python 3.6+ it would
        be).

        >>> import numpy as np
        >>> import imgaug as ia
        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.bbs import BoundingBox
        >>> aug = iaa.Affine(rotate=(-25, 25))
        >>> images = [np.zeros((64, 64, 3), dtype=np.uint8),
        >>>           np.zeros((32, 32, 3), dtype=np.uint8)]
        >>> keypoints = [[(10, 20), (30, 32)],  # KPs on first image
        >>>              [(22, 10), (12, 14)]]  # KPs on second image
        >>> bbs = [
        >>>           [BoundingBox(x1=5, y1=5, x2=50, y2=45)],
        >>>           [BoundingBox(x1=4, y1=6, x2=10, y2=15),
        >>>            BoundingBox(x1=8, y1=9, x2=16, y2=30)]
        >>>       ]  # one BB on first image, two BBs on second image
        >>> batch_aug = aug.augment(
        >>>     images=images, keypoints=keypoints, bounding_boxes=bbs,
        >>>     return_batch=True)

        Create two images of size ``64x64`` and ``32x32``, two sets of
        keypoints (each containing two keypoints) and two sets of bounding
        boxes (the first containing one bounding box, the second two bounding
        boxes). These augmentables are then augmented by applying random
        rotations between ``-25`` deg and ``+25`` deg to them. The rotation
        values are sampled by image and aligned between all augmentables on
        the same image. The method finally returns an instance of
        :class:`imgaug.augmentables.batches.UnnormalizedBatch` from which the
        augmented data can be retrieved via ``batch_aug.images_aug``,
        ``batch_aug.keypoints_aug``, and ``batch_aug.bounding_boxes_aug``.
        In python 3.6+, `return_batch` can be kept at ``False`` and the
        augmented data can be retrieved as
        ``images_aug, keypoints_aug, bbs_aug = augment(...)``.

        """
        assert ia.is_single_bool(return_batch), (
            "Expected boolean as argument for 'return_batch', got type %s. "
            "Call augment() only with named arguments, e.g. "
            "augment(images=<array>)." % (str(type(return_batch)),))

        expected_keys = ["images", "heatmaps", "segmentation_maps",
                         "keypoints", "bounding_boxes", "polygons",
                         "line_strings"]
        expected_keys_call = ["image"] + expected_keys

        # at least one augmentable provided?
        assert any([key in kwargs for key in expected_keys_call]), (
            "Expected augment() to be called with one of the following named "
            "arguments: %s. Got none of these." % (
                ", ".join(expected_keys_call),))

        # all keys in kwargs actually known?
        unknown_args = [key for key in kwargs.keys()
                        if key not in expected_keys_call]
        assert len(unknown_args) == 0, (
            "Got the following unknown keyword argument(s) in augment(): %s" % (
                ", ".join(unknown_args)
            ))

        # normalize image=... input to images=...
        # this is not done by Batch.to_normalized_batch()
        if "image" in kwargs:
            assert "images" not in kwargs, (
                "You may only provide the argument 'image' OR 'images' to "
                "augment(), not both of them.")
            images = [kwargs["image"]]
        else:
            images = kwargs.get("images", None)

        # Decide whether to return the final tuple in the order of the kwargs
        # keys or the default order based on python version. Only 3.6+ uses
        # an ordered dict implementation for kwargs.
        order = "standard"
        nb_keys = len(list(kwargs.keys()))
        vinfo = sys.version_info
        is_py36_or_newer = vinfo[0] > 3 or (vinfo[0] == 3 and vinfo[1] >= 6)
        if is_py36_or_newer:
            order = "kwargs_keys"
        elif not return_batch and nb_keys > 2:
            raise ValueError(
                "Requested more than two outputs in augment(), but detected "
                "python version is below 3.6. More than two outputs are only "
                "supported for 3.6+ as earlier python versions offer no way "
                "to retrieve the order of the provided named arguments. To "
                "still use more than two outputs, add 'return_batch=True' as "
                "an argument and retrieve the outputs manually from the "
                "returned UnnormalizedBatch instance, e.g. via "
                "'batch.images_aug' to get augmented images."
            )
        elif not return_batch and nb_keys == 2 and images is None:
            raise ValueError(
                "Requested two outputs from augment() that were not 'images', "
                "but detected python version is below 3.6. For security "
                "reasons, only single-output requests or requests with two "
                "outputs of which one is 'images' are allowed in <3.6. "
                "'images' will then always be returned first. If you don't "
                "want this, use 'return_batch=True' mode in augment(), which "
                "returns a single UnnormalizedBatch instance instead and "
                "supports any combination of outputs."
            )

        # augment batch
        batch = UnnormalizedBatch(
            images=images,
            heatmaps=kwargs.get("heatmaps", None),
            segmentation_maps=kwargs.get("segmentation_maps", None),
            keypoints=kwargs.get("keypoints", None),
            bounding_boxes=kwargs.get("bounding_boxes", None),
            polygons=kwargs.get("polygons", None),
            line_strings=kwargs.get("line_strings", None)
        )

        batch_aug = self.augment_batch(batch, hooks=hooks)

        # return either batch or tuple of augmentables, depending on what
        # was requested by user
        if return_batch:
            return batch_aug

        result = []
        if order == "kwargs_keys":
            for key in kwargs:
                if key == "image":
                    attr = getattr(batch_aug, "images_aug")
                    result.append(attr[0])
                else:
                    result.append(getattr(batch_aug, "%s_aug" % (key,)))
        else:
            for key in expected_keys:
                if key == "images" and "image" in kwargs:
                    attr = getattr(batch_aug, "images_aug")
                    result.append(attr[0])
                elif key in kwargs:
                    result.append(getattr(batch_aug, "%s_aug" % (key,)))

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def __call__(self, *args, **kwargs):
        """Alias for :func:`imgaug.augmenters.meta.Augmenter.augment`."""
        return self.augment(*args, **kwargs)

    def pool(self, processes=None, maxtasksperchild=None, seed=None):
        """Create a pool used for multicore augmentation.

        Parameters
        ----------
        processes : None or int, optional
            Same as in :func:`imgaug.multicore.Pool.__init__`.
            The number of background workers. If ``None``, the number of the
            machine's CPU cores will be used (this counts hyperthreads as CPU
            cores). If this is set to a negative value ``p``, then
            ``P - abs(p)`` will be used, where ``P`` is the number of CPU
            cores. E.g. ``-1`` would use all cores except one (this is useful
            to e.g. reserve one core to feed batches to the GPU).

        maxtasksperchild : None or int, optional
            Same as for :func:`imgaug.multicore.Pool.__init__`.
            The number of tasks done per worker process before the process
            is killed and restarted. If ``None``, worker processes will not
            be automatically restarted.

        seed : None or int, optional
            Same as for :func:`imgaug.multicore.Pool.__init__`.
            The seed to use for child processes. If ``None``, a random seed
            will be used.

        Returns
        -------
        imgaug.multicore.Pool
            Pool for multicore augmentation.

        Examples
        --------
        >>> import numpy as np
        >>> import imgaug as ia
        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.batches import Batch
        >>>
        >>> aug = iaa.Add(1)
        >>> images = np.zeros((16, 128, 128, 3), dtype=np.uint8)
        >>> batches = [Batch(images=np.copy(images)) for _ in range(100)]
        >>> with aug.pool(processes=-1, seed=2) as pool:
        >>>     batches_aug = pool.map_batches(batches, chunksize=8)
        >>> print(np.sum(batches_aug[0].images_aug[0]))
        49152

        Create ``100`` batches of empty images. Each batch contains
        ``16`` images of size ``128x128``. The batches are then augmented on
        all CPU cores except one (``processes=-1``). After augmentation, the
        sum of pixel values from the first augmented image is printed.

        >>> import numpy as np
        >>> import imgaug as ia
        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.batches import Batch
        >>>
        >>> aug = iaa.Add(1)
        >>> images = np.zeros((16, 128, 128, 3), dtype=np.uint8)
        >>> def generate_batches():
        >>>     for _ in range(100):
        >>>         yield Batch(images=np.copy(images))
        >>>
        >>> with aug.pool(processes=-1, seed=2) as pool:
        >>>     batches_aug = pool.imap_batches(generate_batches(), chunksize=8)
        >>>     batch_aug = next(batches_aug)
        >>>     print(np.sum(batch_aug.images_aug[0]))
        49152

        Same as above. This time, a generator is used to generate batches
        of images. Again, the first augmented image's sum of pixels is printed.

        """
        import imgaug.multicore as multicore
        return multicore.Pool(self, processes=processes,
                              maxtasksperchild=maxtasksperchild, seed=seed)

    # TODO most of the code of this function could be replaced with
    #      ia.draw_grid()
    # TODO add parameter for handling multiple images ((a) next to each other
    #      in each row or (b) multiply row count by number of images and put
    #      each one in a new row)
    # TODO "images" parameter deviates from augment_images (3d array is here
    #      treated as one 3d image, in augment_images as (N, H, W))
    # TODO according to the docstring, this can handle (H,W) images, but not
    #      (H,W,1)
    def draw_grid(self, images, rows, cols):
        """Augment images and draw the results as a single grid-like image.

        This method applies this augmenter to the provided images and returns
        a grid image of the results. Each cell in the grid contains a single
        augmented version of an input image.

        If multiple input images are provided, the row count is multiplied by
        the number of images and each image gets its own row.
        E.g. for ``images = [A, B]``, ``rows=2``, ``cols=3``::

            A A A
            B B B
            A A A
            B B B

        for ``images = [A]``, ``rows=2``, ``cols=3``::

            A A A
            A A A

        Parameters
        -------
        images : (N,H,W,3) ndarray or (H,W,3) ndarray or (H,W) ndarray or list of (H,W,3) ndarray or list of (H,W) ndarray
            List of images to augment and draw in the grid.
            If a list, then each element is expected to have shape ``(H, W)``
            or ``(H, W, 3)``. If a single array, then it is expected to have
            shape ``(N, H, W, 3)`` or ``(H, W, 3)`` or ``(H, W)``.

        rows : int
            Number of rows in the grid.
            If ``N`` input images are given, this value will automatically be
            multiplied by ``N`` to create rows for each image.

        cols : int
            Number of columns in the grid.

        Returns
        -------
        (Hg, Wg, 3) ndarray
            The generated grid image with augmented versions of the input
            images. Here, ``Hg`` and ``Wg`` reference the output size of the
            grid, and *not* the sizes of the input images.

        """
        if ia.is_np_array(images):
            if len(images.shape) == 4:
                images = [images[i] for i in range(images.shape[0])]
            elif len(images.shape) == 3:
                images = [images]
            elif len(images.shape) == 2:
                images = [images[:, :, np.newaxis]]
            else:
                raise Exception(
                    "Unexpected images shape, expected 2-, 3- or "
                    "4-dimensional array, got shape %s." % (images.shape,))
        else:
            assert isinstance(images, list), (
                "Expected 'images' to be an ndarray or list of ndarrays. "
                "Got %s." % (type(images),))
            for i, image in enumerate(images):
                if len(image.shape) == 3:
                    continue
                elif len(image.shape) == 2:
                    images[i] = image[:, :, np.newaxis]
                else:
                    raise Exception(
                        "Unexpected image shape at index %d, expected 2- or "
                        "3-dimensional array, got shape %s." % (
                            i, image.shape,))

        det = self if self.deterministic else self.to_deterministic()
        augs = []
        for image in images:
            augs.append(det.augment_images([image] * (rows * cols)))

        augs_flat = list(itertools.chain(*augs))
        cell_height = max([image.shape[0] for image in augs_flat])
        cell_width = max([image.shape[1] for image in augs_flat])
        width = cell_width * cols
        height = cell_height * (rows * len(images))
        grid = np.zeros((height, width, 3), dtype=augs[0][0].dtype)
        for row_idx in range(rows):
            for img_idx, image in enumerate(images):
                for col_idx in range(cols):
                    image_aug = augs[img_idx][(row_idx * cols) + col_idx]
                    cell_y1 = cell_height * (row_idx * len(images) + img_idx)
                    cell_y2 = cell_y1 + image_aug.shape[0]
                    cell_x1 = cell_width * col_idx
                    cell_x2 = cell_x1 + image_aug.shape[1]
                    grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image_aug

        return grid

    # TODO test for 2D images
    # TODO test with C = 1
    def show_grid(self, images, rows, cols):
        """Augment images and plot the results as a single grid-like image.

        This calls :func:`imgaug.augmenters.meta.Augmenter.draw_grid` and
        simply shows the results. See that method for details.

        Parameters
        ----------
        images : (N,H,W,3) ndarray or (H,W,3) ndarray or (H,W) ndarray or list of (H,W,3) ndarray or list of (H,W) ndarray
            List of images to augment and draw in the grid.
            If a list, then each element is expected to have shape ``(H, W)``
            or ``(H, W, 3)``. If a single array, then it is expected to have
            shape ``(N, H, W, 3)`` or ``(H, W, 3)`` or ``(H, W)``.

        rows : int
            Number of rows in the grid.
            If ``N`` input images are given, this value will automatically be
            multiplied by ``N`` to create rows for each image.

        cols : int
            Number of columns in the grid.

        """
        grid = self.draw_grid(images, rows, cols)
        ia.imshow(grid)

    def to_deterministic(self, n=None):
        """Convert this augmenter from a stochastic to a deterministic one.

        A stochastic augmenter samples pseudo-random values for each parameter,
        image and batch.
        A deterministic augmenter also samples new values for each parameter
        and image, but not batch. Instead, for consecutive batches it will
        sample the same values (provided the number of images and their sizes
        don't change).
        From a technical perspective this means that a deterministic augmenter
        starts each batch's augmentation with a random number generator in
        the same state (i.e. same seed), instead of advancing that state from
        batch to batch.

        Using determinism is useful to (a) get the same augmentations for
        two or more image batches (e.g. for stereo cameras), (b) to augment
        images and corresponding data on them (e.g. segmentation maps or
        bounding boxes) in the same way.

        Parameters
        ----------
        n : None or int, optional
            Number of deterministic augmenters to return.
            If ``None`` then only one :class:`imgaug.augmenters.meta.Augmenter`
            instance will be returned.
            If ``1`` or higher, a list containing ``n``
            :class:`imgaug.augmenters.meta.Augmenter` instances will be
            returned.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter
            A single Augmenter object if `n` was None,
            otherwise a list of Augmenter objects (even if `n` was ``1``).

        """
        assert n is None or n >= 1, (
            "Expected 'n' to be None or >=1, got %s." % (n,))
        if n is None:
            return self.to_deterministic(1)[0]
        else:
            return [self._to_deterministic() for _ in sm.xrange(n)]

    def _to_deterministic(self):
        """Convert this augmenter from a stochastic to a deterministic one.

        Augmenter-specific implementation of
        :func:`imgaug.augmenters.meta.to_deterministic`. This function is
        expected to return a single new deterministic
        :class:`imgaug.augmenters.meta.Augmenter` instance of this augmenter.

        Returns
        -------
        det : imgaug.augmenters.meta.Augmenter
            Deterministic variation of this Augmenter object.

        """
        aug = self.copy()

        # This was changed for 0.2.8 from deriving a new random state based on
        # the global random state to deriving it from the augmenter's local
        # random state. This should reduce the risk that re-runs of scripts
        # lead to different results upon small changes somewhere. It also
        # decreases the likelihood of problems when using multiprocessing
        # (the child processes might use the same global random state as the
        # parent process). Note for the latter point that augment_batches()
        # might call to_deterministic() if the batch contains multiply types
        # of augmentables.
        # aug.random_state = iarandom.create_random_rng()
        aug.random_state = self.random_state.derive_rng_()

        aug.deterministic = True
        return aug

    # TODO mark this as in-place
    def reseed(self, random_state=None, deterministic_too=False):
        """Reseed this augmenter and all of its children.

        This method assigns a new random number generator to the
        augmenter and all of its children (if it has any). The new random
        number generator is derived from the provided one or from the
        global random number generator.

        If this augmenter or any child augmenter had a random numer generator
        that pointed to the global random state, it will automatically be
        replaced with a local random state. This is similar to what
        :func:`imgaug.augmenters.meta.Augmenter.localize_random_state`
        does.

        This method is useful when augmentations are run in the
        background (i.e. on multiple cores).
        It should be called before sending this
        :class:`imgaug.augmenters.meta.Augmenter` instance to a
        background worker or once within each worker with different seeds
        (i.e., if ``N`` workers are used, the function should be called
        ``N`` times). Otherwise, all background workers will
        use the same seeds and therefore apply the same augmentations.
        Note that :func:`Augmenter.augment_batches` and :func:`Augmenter.pool`
        already do this automatically.

        Parameters
        ----------
        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            A seed or random number generator that is used to derive new
            random number generators for this augmenter and its children.
            If an ``int`` is provided, it will be interpreted as a seed.
            If ``None`` is provided, the global random number generator will
            be used.

        deterministic_too : bool, optional
            Whether to also change the seed of an augmenter ``A``, if ``A``
            is deterministic. This is the case both when this augmenter
            object is ``A`` or one of its children is ``A``.

        """
        assert isinstance(deterministic_too, bool), (
            "Expected 'deterministic_too' to be a boolean, got type %s." % (
                deterministic_too))

        if random_state is None:
            random_state = iarandom.RNG.create_pseudo_random_()
        else:
            random_state = iarandom.RNG(random_state)

        if not self.deterministic or deterministic_too:
            # note that deriving advances the RNG, so child augmenters get a
            # different RNG state
            self.random_state = random_state.copy()

        for lst in self.get_children_lists():
            for aug in lst:
                aug.reseed(random_state=random_state.derive_rng_(),
                           deterministic_too=deterministic_too)

    def localize_random_state(self, recursive=True):
        """Assign augmenter-specific RNGs to this augmenter and its children.

        See :func:`Augmenter.localize_random_state_` for more details.

        Parameters
        ----------
        recursive : bool, optional
            See
            :func:`imgaug.augmenters.meta.Augmenter.localize_random_state_`.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Copy of the augmenter and its children, with localized RNGs.

        """
        aug = self.deepcopy()
        aug.localize_random_state_(
            recursive=recursive
        )
        return aug

    # TODO rename random_state -> rng
    def localize_random_state_(self, recursive=True):
        """Assign augmenter-specific RNGs to this augmenter and its children.

        This method iterates over this augmenter and all of its children and
        replaces any pointer to the global RNG with a new local (i.e.
        augmenter-specific) RNG.

        A random number generator (RNG) is used for the sampling of random
        values.
        The global random number generator exists exactly once throughout
        the library and is shared by many augmenters.
        A local RNG (usually) exists within exactly one augmenter and is
        only used by that augmenter.

        Usually there is no need to change global into local RNGs.
        The only noteworthy exceptions are

            * Whenever you want to use determinism (so that the global RNG is
              not accidentally reverted).
            * Whenever you want to copy RNGs from one augmenter to
              another. (Copying the global RNG would usually not be useful.
              Copying the global RNG from augmenter A to B, then executing A
              and then B would result in B's (global) RNG's state having
              already changed because of A's sampling. So the samples of
              A and B would differ.)

        The case of determinism is handled automatically by
        :func:`imgaug.augmenters.meta.Augmenter.to_deterministic`.
        Only when you copy RNGs (via
        :func:`imgaug.augmenters.meta.Augmenter.copy_random_state`),
        you need to call this function first.

        Parameters
        ----------
        recursive : bool, optional
            Whether to localize the RNGs of the augmenter's children too.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Returns itself (with localized RNGs).

        """
        if self.random_state.is_global_rng():
            self.random_state = self.random_state.derive_rng_()
        if recursive:
            for lst in self.get_children_lists():
                for child in lst:
                    child.localize_random_state_(recursive=recursive)
        return self

    # TODO adapt random_state -> rng
    def copy_random_state(self, source, recursive=True, matching="position",
                          matching_tolerant=True, copy_determinism=False):
        """Copy the RNGs from a source augmenter sequence.

        Parameters
        ----------
        source : imgaug.augmenters.meta.Augmenter
            See :func:`imgaug.augmenters.meta.Augmenter.copy_random_state_`.

        recursive : bool, optional
            See :func:`imgaug.augmenters.meta.Augmenter.copy_random_state_`.

        matching : {'position', 'name'}, optional
            See :func:`imgaug.augmenters.meta.Augmenter.copy_random_state_`.

        matching_tolerant : bool, optional
            See :func:`imgaug.augmenters.meta.Augmenter.copy_random_state_`.

        copy_determinism : bool, optional
            See :func:`imgaug.augmenters.meta.Augmenter.copy_random_state_`.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Copy of the augmenter itself (with copied RNGs).

        """
        aug = self.deepcopy()
        aug.copy_random_state_(
            source,
            recursive=recursive,
            matching=matching,
            matching_tolerant=matching_tolerant,
            copy_determinism=copy_determinism
        )
        return aug

    def copy_random_state_(self, source, recursive=True, matching="position",
                           matching_tolerant=True, copy_determinism=False):
        """Copy the RNGs from a source augmenter sequence (in-place).

        .. note ::

            The source augmenters are not allowed to use the global RNG.
            Call
            :func:`imgaug.augmenters.meta.Augmenter.localize_random_state_`
            once on the source to localize all random states.

        Parameters
        ----------
        source : imgaug.augmenters.meta.Augmenter
            The source augmenter(s) from where to copy the RNG(s).
            The source may have children (e.g. the source can be a
            :class:`imgaug.augmenters.meta.Sequential`).

        recursive : bool, optional
            Whether to copy the RNGs of the source augmenter *and*
            all of its children (``True``) or just the source
            augmenter (``False``).

        matching : {'position', 'name'}, optional
            Defines the matching mode to use during recursive copy.
            This is used to associate source augmenters with target augmenters.
            If ``position`` then the target and source sequences of augmenters
            are turned into flattened lists and are associated based on
            their list indices. If ``name`` then the target and source
            augmenters are matched based on their names (i.e.
            ``augmenter.name``).

        matching_tolerant : bool, optional
            Whether to use tolerant matching between source and target
            augmenters. If set to ``False``: Name matching will raise an
            exception for any target augmenter which's name does not appear
            among the source augmenters. Position matching will raise an
            exception if source and target augmenter have an unequal number
            of children.

        copy_determinism : bool, optional
            Whether to copy the ``deterministic`` attributes from source to
            target augmenters too.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            The augmenter itself.

        """
        # Note: the target random states are localized, but the source random
        # states don't have to be localized. That means that they can be
        # the global random state. Worse, if copy_random_state() was called,
        # the target random states would have different identities, but
        # same states. If multiple target random states were the global random
        # state, then after deepcopying them, they would all share the same
        # identity that is different to the global random state. I.e., if the
        # state of any random state of them is set in-place, it modifies the
        # state of all other target random states (that were once global),
        # but not the global random state.
        # Summary: Use target = source.copy() here, instead of
        # target.use_state_of_(source).

        source_augs = (
            [source] + source.get_all_children(flat=True)
            if recursive
            else [source])
        target_augs = (
            [self] + self.get_all_children(flat=True)
            if recursive
            else [self])

        global_rs_exc_msg = (
            "You called copy_random_state_() with a source that uses global "
            "RNGs. Call localize_random_state_() on the source "
            "first or initialize your augmenters with local random states, "
            "e.g. via Dropout(..., random_state=1234).")

        if matching == "name":
            source_augs_dict = {aug.name: aug for aug in source_augs}
            target_augs_dict = {aug.name: aug for aug in target_augs}

            different_lengths = (
                len(source_augs_dict) < len(source_augs)
                or len(target_augs_dict) < len(target_augs))
            if different_lengths:
                ia.warn(
                    "Matching mode 'name' with recursive=True was chosen in "
                    "copy_random_state_, but either the source or target "
                    "augmentation sequence contains multiple augmenters with "
                    "the same name."
                )

            for name in target_augs_dict:
                if name in source_augs_dict:
                    if source_augs_dict[name].random_state.is_global_rng():
                        raise Exception(global_rs_exc_msg)
                    # has to be copy(), see above
                    target_augs_dict[name].random_state = \
                        source_augs_dict[name].random_state.copy()
                    if copy_determinism:
                        target_augs_dict[name].deterministic = \
                            source_augs_dict[name].deterministic
                elif not matching_tolerant:
                    raise Exception(
                        "Augmenter name '%s' not found among source "
                        "augmenters." % (name,))
        elif matching == "position":
            if len(source_augs) != len(target_augs) and not matching_tolerant:
                raise Exception(
                    "Source and target augmentation sequences have different "
                    "lengths.")
            for source_aug, target_aug in zip(source_augs, target_augs):
                if source_aug.random_state.is_global_rng():
                    raise Exception(global_rs_exc_msg)
                # has to be copy(), see above
                target_aug.random_state = source_aug.random_state.copy()
                if copy_determinism:
                    target_aug.deterministic = source_aug.deterministic
        else:
            raise Exception(
                "Unknown matching method '%s'. Valid options are 'name' "
                "and 'position'." % (matching,))

        return self

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError()

    def get_children_lists(self):
        """Get a list of lists of children of this augmenter.

        For most augmenters, the result will be a single empty list.
        For augmenters with children it will often be a list with one
        sublist containing all children. In some cases the augmenter will
        contain multiple distinct lists of children, e.g. an if-list and an
        else-list. This will lead to a result consisting of a single list
        with multiple sublists, each representing the respective sublist of
        children.

        E.g. for an if/else-augmenter that executes the children ``A1``,
        ``A2`` if a condition is met and otherwise executes the children
        ``B1``, ``B2``, ``B3`` the result will be
        ``[[A1, A2], [B1, B2, B3]]``.

        IMPORTANT: While the topmost list may be newly created, each of the
        sublist must be editable inplace resulting in a changed children list
        of the augmenter. E.g. if an Augmenter
        ``IfElse(condition, [A1, A2], [B1, B2, B3])`` returns
        ``[[A1, A2], [B1, B2, B3]]``
        for a call to
        :func:`imgaug.augmenters.meta.Augmenter.get_children_lists` and
        ``A2`` is removed inplace from ``[A1, A2]``, then the children lists
        of ``IfElse(...)`` must also change to ``[A1], [B1, B2, B3]``. This
        is used in
        :func:`imgaug.augmeneters.meta.Augmenter.remove_augmenters_inplace`.

        Returns
        -------
        list of list of imgaug.augmenters.meta.Augmenter
            One or more lists of child augmenter.
            Can also be a single empty list.

        """
        return []

    # TODO why does this exist? it seems to be identical to
    #      get_children_lists() for flat=False, aside from returning list
    #      copies instead of the same instances as used by the augmenters.
    # TODO this can be simplified using imgaug.imgaug.flatten()?
    def get_all_children(self, flat=False):
        """Get all children of this augmenter as a list.

        If the augmenter has no children, the returned list is empty.

        Parameters
        ----------
        flat : bool
            If set to ``True``, the returned list will be flat.

        Returns
        -------
        list of imgaug.augmenters.meta.Augmenter
            The children as a nested or flat list.

        """
        result = []
        for lst in self.get_children_lists():
            for aug in lst:
                result.append(aug)
                children = aug.get_all_children(flat=flat)
                if len(children) > 0:
                    if flat:
                        result.extend(children)
                    else:
                        result.append(children)
        return result

    def find_augmenters(self, func, parents=None, flat=True):
        """Find augmenters that match a condition.

        This function will compare this augmenter and all of its children
        with a condition. The condition is a lambda function.

        Parameters
        ----------
        func : callable
            A function that receives a
            :class:`imgaug.augmenters.meta.Augmenter` instance and a list of
            parent :class:`imgaug.augmenters.meta.Augmenter` instances and
            must return ``True``, if that augmenter is valid match or
            ``False`` otherwise.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            List of parent augmenters.
            Intended for nested calls and can usually be left as ``None``.

        flat : bool, optional
            Whether to return the result as a flat list (``True``)
            or a nested list (``False``). In the latter case, the nesting
            matches each augmenters position among the children.

        Returns
        ----------
        list of imgaug.augmenters.meta.Augmenter
            Nested list if `flat` was set to ``False``.
            Flat list if `flat` was set to ``True``.

        Examples
        --------
        >>> import imgaug.augmenters as iaa
        >>> aug = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>     iaa.Flipud(0.5, name="flipud")
        >>> ])
        >>> print(aug.find_augmenters(lambda a, parents: a.name == "fliplr"))

        Return the first child augmenter (``Fliplr`` instance).

        """
        if parents is None:
            parents = []

        result = []
        if func(self, parents):
            result.append(self)

        subparents = parents + [self]
        for lst in self.get_children_lists():
            for aug in lst:
                found = aug.find_augmenters(func, parents=subparents,
                                            flat=flat)
                if len(found) > 0:
                    if flat:
                        result.extend(found)
                    else:
                        result.append(found)
        return result

    def find_augmenters_by_name(self, name, regex=False, flat=True):
        """Find augmenter(s) by name.

        Parameters
        ----------
        name : str
            Name of the augmenter(s) to search for.

        regex : bool, optional
            Whether `name` parameter is a regular expression.

        flat : bool, optional
            See :func:`imgaug.augmenters.meta.Augmenter.find_augmenters`.

        Returns
        -------
        augmenters : list of imgaug.augmenters.meta.Augmenter
            Nested list if `flat` was set to ``False``.
            Flat list if `flat` was set to ``True``.

        """
        return self.find_augmenters_by_names([name], regex=regex, flat=flat)

    def find_augmenters_by_names(self, names, regex=False, flat=True):
        """Find augmenter(s) by names.

        Parameters
        ----------
        names : list of str
            Names of the augmenter(s) to search for.

        regex : bool, optional
            Whether `names` is a list of regular expressions.
            If it is, an augmenter is considered a match if *at least* one
            of these expressions is a match.

        flat : boolean, optional
            See :func:`imgaug.augmenters.meta.Augmenter.find_augmenters`.

        Returns
        -------
        augmenters : list of imgaug.augmenters.meta.Augmenter
            Nested list if `flat` was set to ``False``.
            Flat list if `flat` was set to ``True``.

        """
        if regex:
            def comparer(aug, parents):
                for pattern in names:
                    if re.match(pattern, aug.name):
                        return True
                return False

            return self.find_augmenters(comparer, flat=flat)
        else:
            return self.find_augmenters(
                lambda aug, parents: aug.name in names, flat=flat)

    def remove_augmenters(self, func, copy=True, noop_if_topmost=True):
        """Remove this augmenter or children that match a condition.

        Parameters
        ----------
        func : callable
            Condition to match per augmenter.
            The function must expect the augmenter itself and a list of parent
            augmenters and returns ``True`` if that augmenter is supposed to
            be removed, or ``False`` otherwise.
            E.g. ``lambda a, parents: a.name == "fliplr" and len(parents) == 1``
            removes an augmenter with name ``fliplr`` if it is the direct child
            of the augmenter upon which ``remove_augmenters()`` was initially
            called.

        copy : bool, optional
            Whether to copy this augmenter and all if its children before
            removing. If ``False``, removal is performed in-place.

        noop_if_topmost : bool, optional
            If ``True`` and the condition (lambda function) leads to the
            removal of the topmost augmenter (the one this function is called
            on initially), then that topmost augmenter will be replaced by an
            instance of :class:`imgaug.augmenters.meta.Noop` (i.e. an
            augmenter that doesn't change its inputs). If ``False``, ``None``
            will be returned in these cases.
            This can only be ``False`` if copy is set to ``True``.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter or None
            This augmenter after the removal was performed.
            ``None`` is returned if the condition was matched for the
            topmost augmenter, `copy` was set to ``True`` and `noop_if_topmost`
            was set to ``False``.

        Examples
        --------
        >>> import imgaug.augmenters as iaa
        >>> seq = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>     iaa.Flipud(0.5, name="flipud"),
        >>> ])
        >>> seq = seq.remove_augmenters(lambda a, parents: a.name == "fliplr")

        This removes the augmenter ``Fliplr`` from the ``Sequential``
        object's children.

        """
        if func(self, []):
            if not copy:
                raise Exception(
                    "Inplace removal of topmost augmenter requested, "
                    "which is currently not possible. Set 'copy' to True.")

            if noop_if_topmost:
                return Noop()
            else:
                return None
        else:
            aug = self if not copy else self.deepcopy()
            aug.remove_augmenters_inplace(func, parents=[])
            return aug

    # TODO rename to remove_augmenters_()
    def remove_augmenters_inplace(self, func, parents=None):
        """Remove in-place children of this augmenter that match a condition.

        This is functionally identical to
        :func:`imgaug.augmenters.meta.remove_augmenters` with
        ``copy=False``, except that it does not affect the topmost augmenter
        (the one on which this function is initially called on).

        Parameters
        ----------
        func : callable
            See :func:`imgaug.augmenters.meta.Augmenter.remove_augmenters`.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            List of parent :class:`imgaug.augmenters.meta.Augmenter` instances
            that lead to this augmenter. If ``None``, an empty list will be
            used. This parameter can usually be left empty and will be set
            automatically for children.

        Examples
        --------
        >>> import imgaug.augmenters as iaa
        >>> seq = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>    iaa.Flipud(0.5, name="flipud"),
        >>> ])
        >>> seq.remove_augmenters_inplace(lambda a, parents: a.name == "fliplr")

        This removes the augmenter ``Fliplr`` from the ``Sequential``
        object's children.

        """
        parents = [] if parents is None else parents
        subparents = parents + [self]
        for lst in self.get_children_lists():
            to_remove = []
            for i, aug in enumerate(lst):
                if func(aug, subparents):
                    to_remove.append((i, aug))

            for count_removed, (i, aug) in enumerate(to_remove):
                del lst[i - count_removed]

            for aug in lst:
                aug.remove_augmenters_inplace(func, subparents)

    def copy(self):
        """Create a shallow copy of this Augmenter instance.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Shallow copy of this Augmenter instance.

        """
        return copy_module.copy(self)

    def deepcopy(self):
        """Create a deep copy of this Augmenter instance.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Deep copy of this Augmenter instance.

        """
        # TODO if this augmenter has child augmenters and multiple of them
        #      use the global random state, then after copying, these
        #      augmenters share a single new random state that is a copy of
        #      the global random state (i.e. all use the same *instance*,
        #      not just state). This can lead to confusing bugs.
        # TODO write a custom copying routine?
        return copy_module.deepcopy(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        params = self.get_parameters()
        params_str = ", ".join([param.__str__() for param in params])
        return "%s(name=%s, parameters=[%s], deterministic=%s)" % (
            self.__class__.__name__, self.name, params_str, self.deterministic)


class Sequential(Augmenter, list):
    """List augmenter containing child augmenters to apply to inputs.

    This augmenter is simply a list of other augmenters. To augment an image
    or any other data, it iterates over its children and applies each one
    of them independently to the data. (This also means that the second
    applied augmenter will already receive augmented input data and augment
    it further.)

    This augmenter offers the option to apply its children in random order
    using the `random_order` parameter. This should often be activated as
    it greatly increases the space of possible augmentations.

    .. note ::

        You are *not* forced to use :class:`imgaug.augmenters.meta.Sequential`
        in order to use other augmenters. Each augmenter can be used on its
        own, e.g the following defines an augmenter for horizontal flips and
        then augments a single image:

        >>> import numpy as np
        >>> import imgaug.augmenters as iaa
        >>> image = np.zeros((32, 32, 3), dtype=np.uint8)
        >>> aug = iaa.Fliplr(0.5)
        >>> image_aug = aug.augment_image(image)

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
    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        The augmenters to apply to images.

    random_order : bool, optional
        Whether to apply the child augmenters in random order.
        If ``True``, the order will be randomly sampled once per batch.

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
    >>> imgs = [np.random.rand(10, 10)]
    >>> seq = iaa.Sequential([
    >>>     iaa.Fliplr(0.5),
    >>>     iaa.Flipud(0.5)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Create a :class:`imgaug.augmenters.meta.Sequential` that always first
    applies a horizontal flip augmenter and then a vertical flip augmenter.
    Each of these two augmenters has a ``50%`` probability of actually
    flipping the image.

    >>> seq = iaa.Sequential([
    >>>     iaa.Fliplr(0.5),
    >>>     iaa.Flipud(0.5)
    >>> ], random_order=True)
    >>> imgs_aug = seq.augment_images(imgs)

    Create a :class:`imgaug.augmenters.meta.Sequential` that sometimes first
    applies a horizontal flip augmenter (followed by a vertical flip
    augmenter) and sometimes first a vertical flip augmenter (followed by a
    horizontal flip augmenter). Again, each of them has a ``50%`` probability
    of actually flipping the image.

    """

    def __init__(self, children=None, random_order=False, name=None,
                 deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic,
                           random_state=random_state)

        if children is None:
            list.__init__(self, [])
        elif isinstance(children, Augmenter):
            # this must be separate from `list.__init__(self, children)`,
            # otherwise in `Sequential(OneOf(...))` the OneOf(...) is
            # interpreted as a list and OneOf's children become Sequential's
            # children
            list.__init__(self, [children])
        elif ia.is_iterable(children):
            assert all([isinstance(child, Augmenter) for child in children]), (
                "Expected all children to be augmenters, got types %s." % (
                    ", ".join([str(type(v)) for v in children])))
            list.__init__(self, children)
        else:
            raise Exception("Expected None or Augmenter or list of Augmenter, "
                            "got %s." % (type(children),))

        assert ia.is_single_bool(random_order), (
            "Expected random_order to be boolean, got %s." % (
                type(random_order),))
        self.random_order = random_order

    def _is_propagating(self, augmentables, parents, hooks):
        return (
            hooks is None
            or hooks.is_propagating(augmentables, augmenter=self,
                                    parents=parents, default=True)
        )

    # TODO make the below functions more DRY
    def _augment_images(self, images, random_state, parents, hooks):
        if self._is_propagating(images, parents, hooks):
            if self.random_order:
                for index in random_state.permutation(len(self)):
                    images = self[index].augment_images(
                        images=images,
                        parents=parents + [self],
                        hooks=hooks
                    )
            else:
                for augmenter in self:
                    images = augmenter.augment_images(
                        images=images,
                        parents=parents + [self],
                        hooks=hooks
                    )
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        if self._is_propagating(heatmaps, parents, hooks):
            if self.random_order:
                for index in random_state.permutation(len(self)):
                    heatmaps = self[index].augment_heatmaps(
                        heatmaps=heatmaps,
                        parents=parents + [self],
                        hooks=hooks
                    )
            else:
                for augmenter in self:
                    heatmaps = augmenter.augment_heatmaps(
                        heatmaps=heatmaps,
                        parents=parents + [self],
                        hooks=hooks
                    )
        return heatmaps

    def _augment_segmentation_maps(self, segmaps, random_state, parents, hooks):
        if self._is_propagating(segmaps, parents, hooks):
            if self.random_order:
                for index in random_state.permutation(len(self)):
                    segmaps = self[index].augment_segmentation_maps(
                        segmaps=segmaps,
                        parents=parents + [self],
                        hooks=hooks
                    )
            else:
                for augmenter in self:
                    segmaps = augmenter.augment_segmentation_maps(
                        segmaps=segmaps,
                        parents=parents + [self],
                        hooks=hooks
                    )
        return segmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        if self._is_propagating(keypoints_on_images, parents, hooks):
            if self.random_order:
                for index in random_state.permutation(len(self)):
                    keypoints_on_images = self[index].augment_keypoints(
                        keypoints_on_images=keypoints_on_images,
                        parents=parents + [self],
                        hooks=hooks
                    )
            else:
                for augmenter in self:
                    keypoints_on_images = augmenter.augment_keypoints(
                        keypoints_on_images=keypoints_on_images,
                        parents=parents + [self],
                        hooks=hooks
                    )
        return keypoints_on_images

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        if self._is_propagating(polygons_on_images, parents, hooks):
            if self.random_order:
                for index in random_state.permutation(len(self)):
                    polygons_on_images = self[index].augment_polygons(
                        polygons_on_images=polygons_on_images,
                        parents=parents + [self],
                        hooks=hooks
                    )
            else:
                for augmenter in self:
                    polygons_on_images = augmenter.augment_polygons(
                        polygons_on_images=polygons_on_images,
                        parents=parents + [self],
                        hooks=hooks
                    )
        return polygons_on_images

    def _to_deterministic(self):
        augs = [aug.to_deterministic() for aug in self]
        seq = self.copy()
        seq[:] = augs
        seq.random_state = self.random_state.derive_rng_()
        seq.deterministic = True
        return seq

    def get_parameters(self):
        return [self.random_order]

    def add(self, augmenter):
        """Add an augmenter to the list of child augmenters.

        Parameters
        ----------
        imgaug.augmenters.meta.Augmenter
            The augmenter to add.

        """
        self.append(augmenter)

    def get_children_lists(self):
        return [self]

    def __str__(self):
        augs_str = ", ".join([aug.__str__() for aug in self])
        pattern = (
            "%s("
            "name=%s, random_order=%s, children=[%s], deterministic=%s"
            ")")
        return pattern % (
            self.__class__.__name__, self.name, self.random_order, augs_str,
            self.deterministic)


class SomeOf(Augmenter, list):
    """List augmenter that applies only some of its children to inputs.

    This augmenter is similar to :class:`imgaug.augmenters.meta.Sequential`,
    but may apply only a fixed or random subset of its child augmenters to
    inputs. E.g. the augmenter could be initialized with a list of 20 child
    augmenters and then apply 5 randomly chosen child augmenters to images.

    The subset of augmenters to apply (and their order) is sampled once
    *per image*. If `random_order` is ``True``, the order will be sampled once
    *per batch* (similar to :class:`imgaug.augmenters.meta.Sequential`).

    This augmenter currently does not support replacing (i.e. picking the same
    child multiple times) due to implementation difficulties in connection
    with deterministic augmenters.

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
    n : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or None, optional
        Count of augmenters to apply.

            * If ``int``, then exactly `n` of the child augmenters are applied
              to every image.
            * If tuple of two ``int`` s ``(a, b)``, then a random value will
              be uniformly sampled per image from the discrete interval
              ``[a..b]`` and denote the number of child augmenters to pick
              and apply. ``b`` may be set to ``None``, which is then equivalent
              to ``(a..C)`` with ``C`` denoting the number of children that
              the augmenter has.
            * If ``StochasticParameter``, then ``N`` numbers will be sampled
              for ``N`` images. The parameter is expected to be discrete.
            * If ``None``, then the total number of available children will be
              used (i.e. all children will be applied).

    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        The augmenters to apply to images.
        If this is a list of augmenters, it will be converted to a
        :class:`imgaug.augmenters.meta.Sequential`.

    random_order : boolean, optional
        Whether to apply the child augmenters in random order.
        If ``True``, the order will be randomly sampled once per batch.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> imgs = [np.random.rand(10, 10)]
    >>> seq = iaa.SomeOf(1, [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Apply either ``Fliplr`` or ``Flipud`` to images.

    >>> seq = iaa.SomeOf((1, 3), [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0),
    >>>     iaa.GaussianBlur(1.0)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Apply one to three of the listed augmenters (``Fliplr``, ``Flipud``,
    ``GaussianBlur``) to images. They are always applied in the
    provided order, i.e. first ``Fliplr``, second ``Flipud``, third
    ``GaussianBlur``.

    >>> seq = iaa.SomeOf((1, None), [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0),
    >>>     iaa.GaussianBlur(1.0)
    >>> ], random_order=True)
    >>> imgs_aug = seq.augment_images(imgs)

    Apply one to all of the listed augmenters (``Fliplr``, ``Flipud``,
    ``GaussianBlur``) to images. They are applied in random order, i.e.
    sometimes ``GaussianBlur`` first, followed by ``Fliplr``, sometimes
    ``Fliplr`` followed by ``Flipud`` followed by ``Blur`` etc.
    The order is sampled once per batch.

    """

    def __init__(self, n=None, children=None, random_order=False,
                 name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic,
                           random_state=random_state)

        # TODO use handle_children_list() here?
        if children is None:
            list.__init__(self, [])
        elif isinstance(children, Augmenter):
            # this must be separate from `list.__init__(self, children)`,
            # otherwise in `SomeOf(OneOf(...))` the OneOf(...) is
            # interpreted as a list and OneOf's children become SomeOf's
            # children
            list.__init__(self, [children])
        elif ia.is_iterable(children):
            assert all([isinstance(child, Augmenter) for child in children]), (
                "Expected all children to be augmenters, got types %s." % (
                    ", ".join([str(type(v)) for v in children])))
            list.__init__(self, children)
        else:
            raise Exception("Expected None or Augmenter or list of Augmenter, "
                            "got %s." % (type(children),))

        if ia.is_single_number(n):
            self.n = int(n)
            self.n_mode = "deterministic"
        elif n is None:
            self.n = None
            self.n_mode = "None"
        elif ia.is_iterable(n):
            assert len(n) == 2, (
                "Expected iterable 'n' to contain exactly two values, "
                "got %d." % (len(n),))
            if ia.is_single_number(n[0]) and n[1] is None:
                self.n = (int(n[0]), None)
                self.n_mode = "(int,None)"
            elif ia.is_single_number(n[0]) and ia.is_single_number(n[1]):
                self.n = iap.DiscreteUniform(int(n[0]), int(n[1]))
                self.n_mode = "stochastic"
            else:
                raise Exception("Expected tuple of (int, None) or (int, int), "
                                "got %s" % ([type(el) for el in n],))
        elif isinstance(n, iap.StochasticParameter):
            self.n = n
            self.n_mode = "stochastic"
        else:
            raise Exception("Expected int, (int, None), (int, int) or "
                            "StochasticParameter, got %s" % (type(n),))

        assert ia.is_single_bool(random_order), (
            "Expected random_order to be boolean, got %s." % (
                type(random_order),))
        self.random_order = random_order

    def _get_n(self, nb_images, random_state):
        if self.n_mode == "deterministic":
            return [self.n] * nb_images
        elif self.n_mode == "None":
            return [len(self)] * nb_images
        elif self.n_mode == "(int,None)":
            param = iap.DiscreteUniform(self.n[0], len(self))
            return param.draw_samples((nb_images,), random_state=random_state)
        elif self.n_mode == "stochastic":
            return self.n.draw_samples((nb_images,), random_state=random_state)
        else:
            raise Exception("Invalid n_mode: %s" % (self.n_mode,))

    def _get_augmenter_order(self, random_state):
        if not self.random_order:
            augmenter_order = np.arange(len(self))
        else:
            augmenter_order = random_state.permutation(len(self))
        return augmenter_order

    def _get_augmenter_active(self, nb_rows, random_state):
        nn = self._get_n(nb_rows, random_state)
        nn = [min(n, len(self)) for n in nn]
        augmenter_active = np.zeros((nb_rows, len(self)), dtype=np.bool)
        for row_idx, n_true in enumerate(nn):
            if n_true > 0:
                augmenter_active[row_idx, 0:n_true] = 1
        for row in augmenter_active:
            random_state.shuffle(row)
        return augmenter_active

    def _is_propagating(self, augmentables, parents, hooks):
        return (
            hooks is None
            or hooks.is_propagating(augmentables, augmenter=self,
                                    parents=parents, default=True)
        )

    def _augment_images(self, images, random_state, parents, hooks):
        if self._is_propagating(images, parents, hooks):
            input_is_array = ia.is_np_array(images)

            # This must happen before creating the augmenter_active array,
            # otherwise in case of determinism the number of augmented images
            # would change the random_state's state, resulting in the order
            # being dependent on the number of augmented images (and not be
            # constant). By doing this first, the random state is always the
            # same (when determinism is active), so the order is always the
            # same.
            augmenter_order = self._get_augmenter_order(random_state)

            # create an array of active augmenters per image
            # e.g.
            #  [[0, 0, 1],
            #   [1, 0, 1],
            #   [1, 0, 0]]
            # would signal, that augmenter 3 is active for the first image,
            # augmenter 1 and 3 for the 2nd image and augmenter 1 for the 3rd.
            augmenter_active = self._get_augmenter_active(len(images),
                                                          random_state)

            for augmenter_index in augmenter_order:
                active = augmenter_active[:, augmenter_index].nonzero()[0]
                if len(active) > 0:
                    # pick images to augment, i.e. images for which
                    # augmenter at current index is active
                    if input_is_array:
                        images_to_aug = images[active]
                    else:
                        images_to_aug = [images[idx] for idx in active]

                    # augment the images
                    images_to_aug = self[augmenter_index].augment_images(
                        images=images_to_aug,
                        parents=parents + [self],
                        hooks=hooks
                    )
                    output_is_array = ia.is_np_array(images_to_aug)
                    output_all_same_shape = len(
                        set([img.shape for img in images_to_aug])) == 1

                    # Map them back to their position in the images array/list
                    # But it can happen that the augmented images have
                    # different shape(s) from the input image, as well as
                    # being suddenly a list instead of a numpy array.
                    # This is usually the case if a child augmenter has to
                    # change shapes, e.g. due to cropping (without resize
                    # afterwards). So accomodate here for that possibility.
                    if input_is_array:
                        if not output_is_array and output_all_same_shape:
                            images_to_aug = np.array(
                                images_to_aug, dtype=images.dtype)
                            output_is_array = True

                        if output_is_array and images_to_aug.shape[1:] == images.shape[1:]:
                            images[active] = images_to_aug
                        else:
                            images = list(images)
                            for aug_idx, original_idx in enumerate(active):
                                images[original_idx] = images_to_aug[aug_idx]
                            input_is_array = False
                    else:
                        for aug_idx, original_idx in enumerate(active):
                            images[original_idx] = images_to_aug[aug_idx]

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        def _augfunc(augmenter_, heatmaps_to_aug_, parents_, hooks_):
            return augmenter_.augment_heatmaps(
                heatmaps_to_aug_, parents_, hooks_)
        return self._augment_non_images(heatmaps, random_state,
                                        parents, hooks, _augfunc)

    def _augment_segmentation_maps(self, segmaps, random_state, parents, hooks):
        def _augfunc(augmenter_, segmaps_to_aug_, parents_, hooks_):
            return augmenter_.augment_segmentation_maps(
                segmaps_to_aug_, parents_, hooks_)
        return self._augment_non_images(segmaps, random_state,
                                        parents, hooks, _augfunc)

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        def _augfunc(augmenter_, koi_to_aug_, parents_, hooks_):
            return augmenter_.augment_keypoints(
                koi_to_aug_, parents_, hooks_)
        return self._augment_non_images(keypoints_on_images, random_state,
                                        parents, hooks, _augfunc)

    def _augment_polygons(self, polygons_on_images, random_state, parents, hooks):
        def _augfunc(augmenter_, polys_to_aug_, parents_, hooks_):
            return augmenter_.augment_polygons(
                polys_to_aug_, parents_, hooks_)
        return self._augment_non_images(polygons_on_images, random_state,
                                        parents, hooks, _augfunc)

    def _augment_non_images(self, inputs, random_state, parents, hooks, func):
        if self._is_propagating(inputs, parents, hooks):
            # This must happen before creating the augmenter_active array,
            # otherwise in case of determinism the number of augmented images
            # would change the random_state's state, resulting in the order
            # being dependent on the number of augmented images (and not be
            # constant). By doing this first, the random state is always the
            # same (when determinism is active), so the order is always the
            # same.
            augmenter_order = self._get_augmenter_order(random_state)

            # create an array of active augmenters per image
            # e.g.
            #  [[0, 0, 1],
            #   [1, 0, 1],
            #   [1, 0, 0]]
            # would signal, that augmenter 3 is active for the first image,
            # augmenter 1 and 3 for the 2nd image and augmenter 1 for the 3rd.
            augmenter_active = self._get_augmenter_active(len(inputs),
                                                          random_state)

            for augmenter_index in augmenter_order:
                active = augmenter_active[:, augmenter_index].nonzero()[0]
                if len(active) > 0:
                    # pick images to augment, i.e. images for which
                    # augmenter at current index is active
                    koi_to_aug = [inputs[idx] for idx in active]

                    # augment the image-related objects
                    koi_to_aug = func(
                        self[augmenter_index], koi_to_aug, parents + [self],
                        hooks
                    )

                    # map them back to their position in the images array/list
                    for aug_idx, original_idx in enumerate(active):
                        inputs[original_idx] = koi_to_aug[aug_idx]
        return inputs

    def _to_deterministic(self):
        augs = [aug.to_deterministic() for aug in self]
        seq = self.copy()
        seq[:] = augs
        seq.random_state = self.random_state.derive_rng_()
        seq.deterministic = True
        return seq

    def get_parameters(self):
        return [self.n]

    def add(self, augmenter):
        """Add an augmenter to the list of child augmenters.

        Parameters
        ----------
        augmenter : imgaug.augmenters.meta.Augmenter
            The augmenter to add.

        """
        self.append(augmenter)

    def get_children_lists(self):
        return [self]

    def __str__(self):
        augs_str = ", ".join([aug.__str__() for aug in self])
        pattern = (
            "%s("
            "name=%s, n=%s, random_order=%s, augmenters=[%s], deterministic=%s"
            ")")
        return pattern % (
            self.__class__.__name__, self.name, str(self.n),
            str(self.random_order), augs_str, self.deterministic)


def OneOf(children, name=None, deterministic=False, random_state=None):
    """Augmenter that always executes exactly one of its children.

    dtype support::

        See ``imgaug.augmenters.meta.SomeOf``.

    Parameters
    ----------
    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter
        The choices of augmenters to apply.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> images = [np.ones((10, 10), dtype=np.uint8)]  # dummy example images
    >>> seq = iaa.OneOf([
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0)
    >>> ])
    >>> images_aug = seq.augment_images(images)

    Flip each image either horizontally or vertically.

    >>> images = [np.ones((10, 10), dtype=np.uint8)]  # dummy example images
    >>> seq = iaa.OneOf([
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Sequential([
    >>>         iaa.GaussianBlur(1.0),
    >>>         iaa.Dropout(0.05),
    >>>         iaa.AdditiveGaussianNoise(0.1*255)
    >>>     ]),
    >>>     iaa.Noop()
    >>> ])
    >>> images_aug = seq.augment_images(images)

    Either flip each image horizontally, or add blur+dropout+noise or do
    nothing.

    """
    return SomeOf(n=1, children=children, random_order=False, name=name,
                  deterministic=deterministic, random_state=random_state)


class Sometimes(Augmenter):
    """Apply child augmenter(s) with a probability of `p`.

    Let ``C`` be one or more child augmenters given to
    :class:`imgaug.augmenters.meta.Sometimes`.
    Let ``p`` be the fraction of images (or other data) to augment.
    Let ``I`` be the input images (or other data).
    Let ``N`` be the number of input images (or other entities).
    Then (on average) ``p*N`` images of ``I`` will be augmented using ``C``.

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
    p : float or imgaug.parameters.StochasticParameter, optional
        Sets the probability with which the given augmenters will be applied to
        input images/data. E.g. a value of ``0.5`` will result in ``50%`` of
        all input images (or other augmentables) being augmented.

    then_list : None or imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) to apply to `p%` percent of all images.
        If this is a list of augmenters, it will be converted to a
        :class:`imgaug.augmenters.meta.Sequential`.

    else_list : None or imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) to apply to ``(1-p)`` percent of all images.
        These augmenters will be applied only when the ones in `then_list`
        are *not* applied (either-or-relationship).
        If this is a list of augmenters, it will be converted to a
        :class:`imgaug.augmenters.meta.Sequential`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3))

    Apply ``GaussianBlur`` to ``50%`` of all input images.

    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3), iaa.Fliplr(1.0))

    Apply ``GaussianBlur`` to ``50%`` of all input images. Apply ``Fliplr``
    to the other ``50%`` of all input images.

    """

    def __init__(self, p=0.5, then_list=None, else_list=None, name=None,
                 deterministic=False, random_state=None):
        super(Sometimes, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.p = iap.handle_probability_param(p, "p")

        self.then_list = handle_children_list(then_list, self.name, "then",
                                              default=None)
        self.else_list = handle_children_list(else_list, self.name, "else",
                                              default=None)

    def _is_propagating(self, augmentables, parents, hooks):
        return (
            hooks is None
            or hooks.is_propagating(augmentables, augmenter=self,
                                    parents=parents, default=True)
        )

    def _augment_images(self, images, random_state, parents, hooks):
        if self._is_propagating(images, parents, hooks):
            input_is_np_array = ia.is_np_array(images)
            if input_is_np_array:
                input_dtype = images.dtype

            nb_images = len(images)
            samples = self.p.draw_samples((nb_images,),
                                          random_state=random_state)

            # create lists/arrays of images for if and else lists (one for
            # each)
            # note that np.where returns tuple(array([0, 5, 9, ...])) or
            # tuple(array([]))
            indices_then_list = np.where(samples == 1)[0]
            indices_else_list = np.where(samples == 0)[0]
            if isinstance(images, list):
                images_then_list = [images[i] for i in indices_then_list]
                images_else_list = [images[i] for i in indices_else_list]
            else:
                images_then_list = images[indices_then_list]
                images_else_list = images[indices_else_list]

            # augment according to if and else list
            result_then_list = images_then_list
            result_else_list = images_else_list
            if self.then_list is not None and len(images_then_list) > 0:
                result_then_list = self.then_list.augment_images(
                    images=images_then_list,
                    parents=parents + [self],
                    hooks=hooks
                )
            if self.else_list is not None and len(images_else_list) > 0:
                result_else_list = self.else_list.augment_images(
                    images=images_else_list,
                    parents=parents + [self],
                    hooks=hooks
                )

            # map results of if/else lists back to their initial positions (in
            # "images" variable)
            result = [None] * len(images)
            for idx_result_then_list, idx_images in enumerate(indices_then_list):
                result[idx_images] = result_then_list[idx_result_then_list]
            for idx_result_else_list, idx_images in enumerate(indices_else_list):
                result[idx_images] = result_else_list[idx_result_else_list]

            # If input was a list, keep the output as a list too,
            # otherwise it was a numpy array, so make the output a numpy array
            # too. Note here though that shapes can differ between images,
            # e.g. when using Crop without resizing. In these cases, the
            # output has to be a list.
            all_same_shape = len(set([image.shape for image in result])) == 1
            if input_is_np_array and all_same_shape:
                result = np.array(result, dtype=input_dtype)
        else:
            result = images

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        def _augfunc(augs_, inputs_, parents_, hooks_):
            return augs_.augment_heatmaps(inputs_, parents_, hooks_)
        return self._augment_non_images(heatmaps, random_state,
                                        parents, hooks, _augfunc)

    def _augment_segmentation_maps(self, segmaps, random_state, parents,
                                   hooks):
        def _augfunc(augs_, inputs_, parents_, hooks_):
            return augs_.augment_segmentation_maps(inputs_, parents_, hooks_)
        return self._augment_non_images(segmaps, random_state,
                                        parents, hooks, _augfunc)

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        def _augfunc(augs_, inputs_, parents_, hooks_):
            return augs_.augment_keypoints(inputs_, parents_, hooks_)
        return self._augment_non_images(keypoints_on_images, random_state,
                                        parents, hooks, _augfunc)

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        def _augfunc(augs_, inputs_, parents_, hooks_):
            return augs_.augment_polygons(inputs_, parents_, hooks_)
        return self._augment_non_images(polygons_on_images, random_state,
                                        parents, hooks, _augfunc)

    def _augment_non_images(self, inputs, random_state, parents, hooks, func):
        result = inputs
        if self._is_propagating(inputs, parents, hooks):
            nb_images = len(inputs)
            samples = self.p.draw_samples((nb_images,),
                                          random_state=random_state)

            # create lists/arrays of images for if and else lists (one for
            # each)
            # note that np.where returns tuple(array([0, 5, 9, ...])) or
            # tuple(array([]))
            indices_then_list = np.where(samples == 1)[0]
            indices_else_list = np.where(samples == 0)[0]
            images_then_list = [inputs[i] for i in indices_then_list]
            images_else_list = [inputs[i] for i in indices_else_list]

            # augment according to if and else list
            result_then_list = images_then_list
            result_else_list = images_else_list
            if self.then_list is not None and len(images_then_list) > 0:
                result_then_list = func(self.then_list, images_then_list,
                                        parents + [self], hooks)
            if self.else_list is not None and len(images_else_list) > 0:
                result_else_list = func(self.else_list, images_else_list,
                                        parents + [self], hooks)

            # map results of if/else lists back to their initial positions
            # (in "images" variable)
            result = [None] * len(inputs)

            gen = enumerate(indices_then_list)
            for idx_result_then_list, idx_images in gen:
                result[idx_images] = result_then_list[idx_result_then_list]

            gen = enumerate(indices_else_list)
            for idx_result_else_list, idx_images in gen:
                result[idx_images] = result_else_list[idx_result_else_list]

        return result

    def _to_deterministic(self):
        aug = self.copy()
        aug.then_list = (
            aug.then_list.to_deterministic()
            if aug.then_list is not None
            else aug.then_list)
        aug.else_list = (
            aug.else_list.to_deterministic()
            if aug.else_list is not None
            else aug.else_list)
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self):
        return [self.p]

    def get_children_lists(self):
        result = []
        if self.then_list is not None:
            result.append(self.then_list)
        if self.else_list is not None:
            result.append(self.else_list)
        return result

    def __str__(self):
        pattern = (
            "%s("
            "p=%s, name=%s, then_list=%s, else_list=%s, deterministic=%s"
            ")")
        return pattern % (
            self.__class__.__name__, self.p, self.name, self.then_list,
            self.else_list, self.deterministic)


class WithChannels(Augmenter):
    """Apply child augmenters to specific channels.

    Let ``C`` be one or more child augmenters given to this augmenter.
    Let ``H`` be a list of channels.
    Let ``I`` be the input images.
    Then this augmenter will pick the channels ``H`` from each image
    in ``I`` (resulting in new images) and apply ``C`` to them.
    The result of the augmentation will be merged back into the original
    images.

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
    channels : None or int or list of int, optional
        Sets the channels to be extracted from each image.
        If ``None``, all channels will be used. Note that this is not
        stochastic - the extracted channels are always the same ones.

    children : Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to images, after the channels
        are extracted.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.WithChannels([0], iaa.Add(10))

    Assuming input images are RGB, then this augmenter will add ``10`` only to
    the first channel, i.e. it will make images appear more red.

    """

    def __init__(self, channels=None, children=None,
                 name=None, deterministic=False, random_state=None):
        super(WithChannels, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        # TODO change this to a stochastic parameter
        if channels is None:
            self.channels = None
        elif ia.is_single_integer(channels):
            self.channels = [channels]
        elif ia.is_iterable(channels):
            only_ints = all([
                ia.is_single_integer(channel) for channel in channels])
            assert only_ints, (
                "Expected integers as channels, got %s." % (
                    [type(channel) for channel in channels],))
            self.channels = channels
        else:
            raise Exception("Expected None, int or list of ints as channels, "
                            "got %s." % (type(channels),))

        self.children = handle_children_list(children, self.name, "then")

    def _is_propagating(self, augmentables, parents, hooks):
        return (
            hooks is None
            or hooks.is_propagating(
                augmentables, augmenter=self, parents=parents, default=True)
        )

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        if self._is_propagating(images, parents, hooks):
            if self.channels is None:
                result = self.children.augment_images(
                    images=images,
                    parents=parents + [self],
                    hooks=hooks
                )
            elif len(self.channels) == 0:
                pass
            else:
                # save the shapes as images are augmented below in-place
                shapes_orig = [image.shape for image in images]

                if ia.is_np_array(images):
                    images_then_list = images[..., self.channels]
                else:
                    images_then_list = [image[..., self.channels]
                                        for image in images]

                result_then_list = self.children.augment_images(
                    images=images_then_list,
                    parents=parents + [self],
                    hooks=hooks
                )

                shapes_same = (
                    all([img_out.shape[0:2] == shape_orig[0:2]
                         for img_out, shape_orig
                         in zip(result_then_list, shapes_orig)]))
                assert shapes_same, (
                    "Heights/widths of images changed in WithChannels from "
                    "%s to %s, but expected to be the same." % (
                        str([shape_orig[0:2]
                             for shape_orig in shapes_orig]),
                        str([img_out.shape[0:2]
                             for img_out in result_then_list]),
                    ))

                if ia.is_np_array(images):
                    result[..., self.channels] = result_then_list
                else:
                    for i in sm.xrange(len(images)):
                        result[i][..., self.channels] = result_then_list[i]

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        def _augfunc(children_, inputs_, parents_, hooks_):
            return children_.augment_heatmaps(inputs_, parents_, hooks_)
        return self._augment_non_images(heatmaps, parents, hooks, _augfunc)

    def _augment_segmentation_maps(self, segmaps, random_state, parents,
                                   hooks):
        def _augfunc(children_, inputs_, parents_, hooks_):
            return children_.augment_segmentation_maps(
                inputs_, parents_, hooks_)
        return self._augment_non_images(segmaps, parents, hooks, _augfunc)

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        def _augfunc(children_, inputs_, parents_, hooks_):
            return children_.augment_keypoints(inputs_, parents_, hooks_)
        return self._augment_non_images(keypoints_on_images, parents, hooks,
                                        _augfunc)

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        def _augfunc(children_, inputs_, parents_, hooks_):
            return children_.augment_polygons(inputs_, parents_, hooks_)
        return self._augment_non_images(polygons_on_images, parents, hooks,
                                        _augfunc)

    def _augment_non_images(self, inputs, parents, hooks, func):
        result = inputs
        if self._is_propagating(inputs, parents, hooks):
            # Augment the non-images in the style of the children if all
            # channels or the majority of them are selected by this layer,
            # otherwise don't change the non-images.
            inputs_to_aug = []
            indices = []

            for i, inputs_i in enumerate(inputs):
                nb_channels = (
                    inputs_i.shape[2] if len(inputs_i.shape) >= 3 else 1)
                did_augment_image = (
                    self.channels is None
                    or len(self.channels) > nb_channels*0.5)
                if did_augment_image:
                    inputs_to_aug.append(inputs_i)
                    indices.append(i)

            if len(inputs_to_aug) > 0:
                inputs_aug = func(self.children, inputs_to_aug,
                                  parents + [self], hooks)

                for idx_orig, inputs_i_aug in zip(indices, inputs_aug):
                    result[idx_orig] = inputs_i_aug

        return result

    def _to_deterministic(self):
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self):
        return [self.channels]

    def get_children_lists(self):
        return [self.children]

    def __str__(self):
        pattern = (
            "%s("
            "channels=%s, name=%s, children=%s, deterministic=%s"
            ")")
        return pattern % (self.__class__.__name__, self.channels, self.name,
                          self.children, self.deterministic)


class Noop(Augmenter):
    """Augmenter that never changes input images ("no operation").

    This augmenter is useful when you just want to use a placeholder augmenter
    in some situation, so that you can continue to call augmentation methods
    without actually transforming the input data. This allows to use the
    same code for training and test.

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
    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        super(Noop, self).__init__(name=name, deterministic=deterministic,
                                   random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        return images

    def get_parameters(self):
        return []


class Lambda(Augmenter):
    """Augmenter that calls a lambda function for each input batch.

    This is useful to add missing functions to a list of augmenters.

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
    func_images : None or callable, optional
        The function to call for each batch of images.
        It must follow the form::

            function(images, random_state, parents, hooks)

        and return the changed images (may be transformed in-place).
        This is essentially the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_images`.
        If this is ``None`` instead of a function, the images will not be
        altered.

    func_heatmaps : None or callable, optional
        The function to call for each batch of heatmaps.
        It must follow the form::

            function(heatmaps, random_state, parents, hooks)

        and return the changed heatmaps (may be transformed in-place).
        This is essentially the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_heatmaps`.
        If this is ``None`` instead of a function, the heatmaps will not be
        altered.

    func_segmentation_maps : None or callable, optional
        The function to call for each batch of segmentation maps.
        It must follow the form::

            function(segmaps, random_state, parents, hooks)

        and return the changed segmaps (may be transformed in-place).
        This is essentially the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_segmentation_maps`.
        If this is ``None`` instead of a function, the segmentatio maps will
        not be altered.

    func_keypoints : None or callable, optional
        The function to call for each batch of image keypoints.
        It must follow the form::

            function(keypoints_on_images, random_state, parents, hooks)

        and return the changed keypoints (may be transformed in-place).
        This is essentially the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_keypoints`.
        If this is ``None`` instead of a function, the keypoints will not be
        altered.

    func_polygons : "keypoints" or None or callable, optional
        The function to call for each batch of image polygons.
        It must follow the form::

            function(polygons_on_images, random_state, parents, hooks)

        and return the changed polygons (may be transformed in-place).
        This is essentially the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_polygons`.
        If this is ``None`` instead of a function, the polygons will not be
        altered.
        If this is the string ``"keypoints"`` instead of a function, the
        polygons will automatically be augmented by transforming their corner
        vertices to keypoints and calling `func_keypoints`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>>
    >>> def func_images(images, random_state, parents, hooks):
    >>>     images[:, ::2, :, :] = 0
    >>>     return images
    >>>
    >>> aug = iaa.Lambda(
    >>>     func_images=func_images
    >>> )

    Replace every second row in input images with black pixels. Leave
    other data (e.g. heatmaps, keypoints) unchanged.

    >>> def func_images(images, random_state, parents, hooks):
    >>>     images[:, ::2, :, :] = 0
    >>>     return images
    >>>
    >>> def func_heatmaps(heatmaps, random_state, parents, hooks):
    >>>     for heatmaps_i in heatmaps:
    >>>         heatmaps.arr_0to1[::2, :, :] = 0
    >>>     return heatmaps
    >>>
    >>> def func_keypoints(keypoints_on_images, random_state, parents, hooks):
    >>>     return keypoints_on_images
    >>>
    >>> aug = iaa.Lambda(
    >>>     func_images=func_images,
    >>>     func_heatmaps=func_heatmaps,
    >>>     func_keypoints=func_keypoints
    >>> )

    Replace every second row in images with black pixels, set every second
    row in heatmaps to zero and leave other data (e.g. keypoints)
    unchanged.

    """

    def __init__(self, func_images=None, func_heatmaps=None,
                 func_segmentation_maps=None, func_keypoints=None,
                 func_polygons="keypoints",
                 name=None, deterministic=False, random_state=None):
        super(Lambda, self).__init__(name=name, deterministic=deterministic,
                                     random_state=random_state)
        self.func_images = func_images
        self.func_heatmaps = func_heatmaps
        self.func_segmentation_maps = func_segmentation_maps
        self.func_keypoints = func_keypoints
        self.func_polygons = func_polygons

    def _augment_images(self, images, random_state, parents, hooks):
        if self.func_images is not None:
            return self.func_images(images, random_state, parents, hooks)
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        if self.func_heatmaps is not None:
            result = self.func_heatmaps(heatmaps, random_state, parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for heatmaps to return list of "
                "imgaug.HeatmapsOnImage instances, got %s." % (
                    type(result),))
            only_heatmaps = all([
                isinstance(el, ia.HeatmapsOnImage) for el in result])
            assert only_heatmaps, (
                "Expected callback function for heatmaps to return list of "
                "imgaug.HeatmapsOnImage instances, got %s." % (
                    [type(el) for el in result],))
            return result
        return heatmaps

    def _augment_segmentation_maps(self, segmaps, random_state, parents, hooks):
        if self.func_segmentation_maps is not None:
            result = self.func_segmentation_maps(segmaps, random_state,
                                                 parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for segmentation maps to return "
                "list of imgaug.SegmentationMapsOnImage() instances, "
                "got %s." % (type(result),))
            only_segmaps = all([
                isinstance(el, ia.SegmentationMapsOnImage) for el in result])
            assert only_segmaps, (
                "Expected callback function for segmentation maps to return "
                "list of imgaug.SegmentationMapsOnImage() instances, "
                "got %s." % ([type(el) for el in result],))
            return result
        return segmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        if self.func_keypoints is not None:
            result = self.func_keypoints(keypoints_on_images, random_state,
                                         parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for keypoints to return list of "
                "imgaug.KeypointsOnImage() instances, got %s." % (
                    type(result),))
            only_keypoints = all([
                isinstance(el, ia.KeypointsOnImage) for el in result])
            assert only_keypoints, (
                "Expected callback function for keypoints to return list of "
                "imgaug.KeypointsOnImage() instances, got %s." % (
                    [type(el) for el in result],))
            return result
        return keypoints_on_images

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        from imgaug.augmentables.polys import _ConcavePolygonRecoverer

        if self.func_polygons == "keypoints":
            return self._augment_polygons_as_keypoints(
                polygons_on_images, random_state, parents, hooks,
                recoverer=_ConcavePolygonRecoverer())
        elif self.func_polygons is not None:
            result = self.func_polygons(polygons_on_images, random_state,
                                        parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for polygons to return list of "
                "imgaug.PolygonsOnImage() instances, got %s." % (
                    type(result),))
            only_polygons = all([
                isinstance(el, ia.PolygonsOnImage) for el in result])
            assert only_polygons, (
                "Expected callback function for polygons to return list of "
                "imgaug.PolygonsOnImage() instances, got %s." % (
                    [type(el) for el in result],))
            return result
        return polygons_on_images

    def get_parameters(self):
        return []


def AssertLambda(func_images=None, func_heatmaps=None,
                 func_segmentation_maps=None, func_keypoints=None,
                 func_polygons=None, name=None, deterministic=False,
                 random_state=None):
    """Assert conditions based on lambda-function to be the case for input data.

    This augmenter applies a lambda function to each image or other input.
    The lambda function must return ``True`` or ``False``. If ``False`` is
    returned, an assertion error is produced.

    This is useful to ensure that generic assumption about the input data
    are actually the case and error out early otherwise.

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
    func_images : None or callable, optional
        The function to call for each batch of images.
        It must follow the form::

            function(images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_images`.

    func_heatmaps : None or callable, optional
        The function to call for each batch of heatmaps.
        It must follow the form::

            function(heatmaps, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_heatmaps`.

    func_segmentation_maps : None or callable, optional
        The function to call for each batch of segmentation maps.
        It must follow the form::

            function(segmaps, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_segmentation_maps`.

    func_keypoints : None or callable, optional
        The function to call for each batch of keypoints.
        It must follow the form::

            function(keypoints_on_images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_keypoints`.

    func_polygons : None or callable, optional
        The function to call for each batch of polygons.
        It must follow the form::

            function(polygons_on_images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_polygons`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    """
    def func_images_assert(images, random_state, parents, hooks):
        assert func_images(images, random_state, parents, hooks), (
            "Input images did not fulfill user-defined assertion in "
            "AssertLambda.")
        return images

    def func_heatmaps_assert(heatmaps, random_state, parents, hooks):
        assert func_heatmaps(heatmaps, random_state, parents, hooks), (
            "Input heatmaps did not fulfill user-defined assertion in "
            "AssertLambda.")
        return heatmaps

    def func_segmentation_maps_assert(segmaps, random_state, parents, hooks):
        assert func_segmentation_maps(segmaps, random_state, parents, hooks), (
            "Input segmentation maps did not fulfill user-defined assertion "
            "in AssertLambda.")
        return segmaps

    def func_keypoints_assert(keypoints_on_images, random_state, parents, hooks):
        assert func_keypoints(keypoints_on_images, random_state, parents, hooks), (
            "Input keypoints did not fulfill user-defined assertion in"
            "AssertLambda.")
        return keypoints_on_images

    def func_polygons_assert(polygons_on_images, random_state, parents, hooks):
        assert func_polygons(polygons_on_images, random_state, parents, hooks), (
            "Input polygons did not fulfill user-defined assertion in"
            "AssertLambda.")
        return polygons_on_images

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    func_sm_assert = func_segmentation_maps_assert
    return Lambda(
        func_images_assert if func_images is not None else None,
        func_heatmaps_assert if func_heatmaps is not None else None,
        func_sm_assert if func_segmentation_maps is not None else None,
        func_keypoints_assert if func_keypoints is not None else None,
        func_polygons_assert if func_polygons is not None else None,
        name=name, deterministic=deterministic, random_state=random_state)


# TODO add tests for segmaps
# TODO This evaluates .shape for kps/polys, but the array shape for
#      heatmaps/segmaps. Not very consistent.
def AssertShape(shape, check_images=True, check_heatmaps=True,
                check_segmentation_maps=True, check_keypoints=True,
                check_polygons=True,
                name=None, deterministic=False, random_state=None):
    """Assert that inputs have a specified shape.

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
    shape : tuple
        The expected shape, given as a ``tuple``. The number of entries in
        the ``tuple`` must match the number of dimensions, i.e. it must
        contain four entries for ``(N, H, W, C)``. If only a single entity
        is augmented, e.g. via
        :func:`imgaug.augmenters.meta.Augmenter.augment_image`, then ``N`` is
        ``1`` in the input to this augmenter. Images that don't have
        a channel axis will automatically have one assigned, i.e. ``C`` is
        at least ``1``.
        For each component of the ``tuple`` one of the following datatypes
        may be used:

            * If a component is ``None``, any value for that dimensions is
              accepted.
            * If a component is ``int``, exactly that value (and no other one)
              will be accepted for that dimension.
            * If a component is a ``tuple`` of two ``int`` s with values ``a``
              and ``b``, only a value within the interval ``[a, b)`` will be
              accepted for that dimension.
            * If an entry is a ``list`` of ``int`` s, only a value from that
              ``list`` will be accepted for that dimension.

    check_images : bool, optional
        Whether to validate input images via the given shape.

    check_heatmaps : bool, optional
        Whether to validate input heatmaps via the given shape.
        The number of heatmaps will be verified as ``N``. For each
        :class:`imgaug.augmentables.heatmaps.HeatmapsOnImage` instance
        its array's height and width will be verified as ``H`` and ``W``,
        but not the channel count.

    check_segmentation_maps : bool, optional
        Whether to validate input segmentation maps via the given shape.
        The number of segmentation maps will be verified as ``N``. For each
        :class:`imgaug.augmentables.segmaps.SegmentationMapOnImage` instance
        its array's height and width will be verified as ``H`` and ``W``,
        but not the channel count.

    check_keypoints : bool, optional
        Whether to validate input keypoints via the given shape.
        This will check (a) the number of keypoints and (b) for each
        :class:`imgaug.augmentables.kps.KeypointsOnImage` instance the
        ``.shape`` attribute, i.e. the shape of the corresponding image.

    check_polygons : bool, optional
        Whether to validate input keypoints via the given shape.
        This will check (a) the number of polygons and (b) for each
        :class:`imgaug.augmentables.polys.PolygonsOnImage` instance the
        ``.shape`` attribute, i.e. the shape of the corresponding image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> seq = iaa.Sequential([
    >>>     iaa.AssertShape((None, 32, 32, 3)),
    >>>     iaa.Fliplr(0.5)
    >>> ])

    Verify first for each image batch if it contains a variable number of
    ``32x32`` images with ``3`` channels each. Only if that check succeeds, the
    horizontal flip will be executed. Otherwise an assertion error will be
    raised.

    >>> seq = iaa.Sequential([
    >>>     iaa.AssertShape((None, (32, 64), 32, [1, 3])),
    >>>     iaa.Fliplr(0.5)
    >>> ])

    Similar to the above example, but now the height may be in the interval
    ``[32, 64)`` and the number of channels may be either ``1`` or ``3``.

    """
    assert len(shape) == 4, (
        "Expected shape to have length 4, got %d with shape: %s." % (
            len(shape), str(shape)))

    def compare(observed, expected, dimension, image_index):
        if expected is not None:
            if ia.is_single_integer(expected):
                assert observed == expected, (
                    "Expected dim %d (entry index: %s) to have value %d, "
                    "got %d." % (dimension, image_index, expected, observed))
            elif isinstance(expected, tuple):
                assert len(expected) == 2, (
                    "Expected tuple argument 'expected' to contain exactly 2 "
                    "entries, got %d." % (len(expected),))
                assert expected[0] <= observed < expected[1], (
                    "Expected dim %d (entry index: %s) to have value in "
                    "interval [%d, %d), got %d." % (
                        dimension, image_index, expected[0], expected[1],
                        observed))
            elif isinstance(expected, list):
                assert any([observed == val for val in expected]), (
                    "Expected dim %d (entry index: %s) to have any value "
                    "of %s, got %d." % (
                        dimension, image_index, str(expected), observed))
            else:
                raise Exception(
                    "Invalid datatype for shape entry %d, expected each entry "
                    "to be an integer, a tuple (with two entries) or a list, "
                    "got %s." % (dimension, type(expected),))

    def func_images(images, _random_state, _parents, _hooks):
        if check_images:
            if isinstance(images, list):
                if shape[0] is not None:
                    compare(len(images), shape[0], 0, "ALL")

                for i in sm.xrange(len(images)):
                    image = images[i]
                    assert len(image.shape) == 3, (
                        "Expected image number %d to have a shape of length "
                        "3, got %d (shape: %s)." % (
                            i, len(image.shape), str(image.shape)))
                    for j in sm.xrange(len(shape)-1):
                        expected = shape[j+1]
                        observed = image.shape[j]
                        compare(observed, expected, j, i)
            else:
                assert len(images.shape) == 4, (
                    "Expected image's shape to have length 4, got %d "
                    "(shape: %s)." % (len(images.shape), str(images.shape)))
                for i in range(4):
                    expected = shape[i]
                    observed = images.shape[i]
                    compare(observed, expected, i, "ALL")
        return images

    def func_heatmaps(heatmaps, _random_state, _parents, _hooks):
        if check_heatmaps:
            if shape[0] is not None:
                compare(len(heatmaps), shape[0], 0, "ALL")

            for i in sm.xrange(len(heatmaps)):
                heatmaps_i = heatmaps[i]
                for j in sm.xrange(len(shape[0:2])):
                    expected = shape[j+1]
                    observed = heatmaps_i.arr_0to1.shape[j]
                    compare(observed, expected, j, i)
        return heatmaps

    def func_segmentation_maps(segmaps, _random_state, _parents, _hooks):
        if check_segmentation_maps:
            if shape[0] is not None:
                compare(len(segmaps), shape[0], 0, "ALL")

            for i in sm.xrange(len(segmaps)):
                segmaps_i = segmaps[i]
                for j in sm.xrange(len(shape[0:2])):
                    expected = shape[j+1]
                    observed = segmaps_i.arr.shape[j]
                    compare(observed, expected, j, i)
        return segmaps

    def func_keypoints(keypoints_on_images, _random_state, _parents, _hooks):
        if check_keypoints:
            if shape[0] is not None:
                compare(len(keypoints_on_images), shape[0], 0, "ALL")

            for i in sm.xrange(len(keypoints_on_images)):
                keypoints_on_image = keypoints_on_images[i]
                for j in sm.xrange(len(shape[0:2])):
                    expected = shape[j+1]
                    observed = keypoints_on_image.shape[j]
                    compare(observed, expected, j, i)
        return keypoints_on_images

    def func_polygons(polygons_on_images, _random_state, _parents, _hooks):
        if check_polygons:
            if shape[0] is not None:
                compare(len(polygons_on_images), shape[0], 0, "ALL")

            for i in sm.xrange(len(polygons_on_images)):
                polygons_on_image = polygons_on_images[i]
                for j in sm.xrange(len(shape[0:2])):
                    expected = shape[j+1]
                    observed = polygons_on_image.shape[j]
                    compare(observed, expected, j, i)
        return polygons_on_images

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return Lambda(func_images=func_images,
                  func_heatmaps=func_heatmaps,
                  func_segmentation_maps=func_segmentation_maps,
                  func_keypoints=func_keypoints,
                  func_polygons=func_polygons,
                  name=name, deterministic=deterministic,
                  random_state=random_state)


class ChannelShuffle(Augmenter):
    """Randomize the order of channels in input images.

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
    p : float or imgaug.parameters.StochasticParameter, optional
        Probability of shuffling channels in any given image.
        May be a fixed probability as a ``float``, or a
        :class:`imgaug.parameters.StochasticParameter` that returns ``0`` s
        and ``1`` s.

    channels : None or imgaug.ALL or list of int, optional
        Which channels are allowed to be shuffled with each other.
        If this is ``None`` or ``imgaug.ALL``, then all channels may be
        shuffled. If it is a ``list`` of ``int`` s,
        then only the channels with indices in that list may be shuffled.
        (Values start at ``0``. All channel indices in the list must exist in
        each image.)

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.ChannelShuffle(0.35)

    Shuffle all channels of ``35%`` of all images.

    >>> aug = iaa.ChannelShuffle(0.35, channels=[0, 1])

    Shuffle only channels ``0`` and ``1`` of ``35%`` of all images. As the new
    channel orders ``0, 1`` and ``1, 0`` are both valid outcomes of the
    shuffling, it means that for ``0.35 * 0.5 = 0.175`` or ``17.5%`` of all
    images the order of channels ``0`` and ``1`` is inverted.

    """

    def __init__(self, p=1.0, channels=None,
                 name=None, deterministic=False, random_state=None):
        super(ChannelShuffle, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        self.p = iap.handle_probability_param(p, "p")
        valid_channels = (
            channels is None
            or channels == ia.ALL
            or (
                isinstance(channels, list)
                and all([ia.is_single_integer(v) for v in channels])
            ))
        assert valid_channels, (
            "Expected None or imgaug.ALL or list of int, got %s." % (
                type(channels),))
        self.channels = channels

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        p_samples = self.p.draw_samples((nb_images,),
                                        random_state=random_state)
        rss = random_state.duplicate(nb_images)
        for i in sm.xrange(nb_images):
            if p_samples[i] >= 1-1e-4:
                images[i] = shuffle_channels(images[i], rss[i], self.channels)
        return images

    def get_parameters(self):
        return [self.p, self.channels]


def shuffle_channels(image, random_state, channels=None):
    """Randomize the order of (color) channels in an image.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; indirectly tested (1)
        * ``uint32``: yes; indirectly tested (1)
        * ``uint64``: yes; indirectly tested (1)
        * ``int8``: yes; indirectly tested (1)
        * ``int16``: yes; indirectly tested (1)
        * ``int32``: yes; indirectly tested (1)
        * ``int64``: yes; indirectly tested (1)
        * ``float16``: yes; indirectly tested (1)
        * ``float32``: yes; indirectly tested (1)
        * ``float64``: yes; indirectly tested (1)
        * ``float128``: yes; indirectly tested (1)
        * ``bool``: yes; indirectly tested (1)

        - (1) Indirectly tested via ``ChannelShuffle``.

    Parameters
    ----------
    image : (H,W,[C]) ndarray
        Image of any dtype for which to shuffle the channels.

    random_state : imgaug.random.RNG
        The random state to use for this shuffling operation.

    channels : None or imgaug.ALL or list of int, optional
        Which channels are allowed to be shuffled with each other.
        If this is ``None`` or ``imgaug.ALL``, then all channels may be
        shuffled. If it is a ``list`` of ``int`` s,
        then only the channels with indices in that list may be shuffled.
        (Values start at ``0``. All channel indices in the list must exist in
        the image.)

    Returns
    -------
    ndarray
        The input image with shuffled channels.

    """
    if image.ndim < 3 or image.shape[2] == 1:
        return image
    nb_channels = image.shape[2]
    all_channels = np.arange(nb_channels)
    is_all_channels = (
        channels is None
        or channels == ia.ALL
        or len(set(all_channels).difference(set(channels))) == 0
    )
    if is_all_channels:
        # note that if this is the case, then 'channels' may be None or
        # imgaug.ALL, so don't simply move the assignment outside of the
        # if/else
        channels_perm = random_state.permutation(all_channels)
        return image[..., channels_perm]
    else:
        channels_perm = random_state.permutation(channels)
        channels_perm_full = all_channels
        for channel_source, channel_target in zip(channels, channels_perm):
            channels_perm_full[channel_source] = channel_target
        return image[..., channels_perm_full]
