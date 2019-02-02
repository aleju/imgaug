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

import numpy as np
import six
import six.moves as sm

from .. import imgaug as ia
from .. import parameters as iap


def clip_augmented_image_(image, min_value, max_value):
    return clip_augmented_images_(image, min_value, max_value)


def clip_augmented_image(image, min_value, max_value):
    return clip_augmented_images(image, min_value, max_value)


def clip_augmented_images_(images, min_value, max_value):
    if ia.is_np_array(images):
        return np.clip(images, min_value, max_value, out=images)
    else:
        return [np.clip(image, min_value, max_value, out=image) for image in images]


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
            ia.do_assert(all([isinstance(child, Augmenter) for child in lst]))
            return lst
        else:
            return Sequential(lst, name="%s-%s" % (augmenter_name, lst_name))
    elif ia.is_iterable(lst):
        if len(lst) == 0 and default != "sequential":
            return default
        ia.do_assert(all([isinstance(child, Augmenter) for child in lst]))
        return Sequential(lst, name="%s-%s" % (augmenter_name, lst_name))
    else:
        raise Exception(("Expected None, Augmenter or list/tuple as children list %s for augmenter with name %s, "
                         + "got %s.") % (lst_name, augmenter_name, type(lst),))


def reduce_to_nonempty(objs):
    objs_reduced = []
    ids = []
    for i, obj in enumerate(objs):
        ia.do_assert(hasattr(obj, "empty"))
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
        assert images.ndim == 4
        return images.shape[3]
    else:
        assert ia.is_iterable(images)
        if len(images) == 0:
            return None
        channels = [el.shape[2] if len(el.shape) >= 3 else 1 for el in images]
        return max(channels)


def copy_arrays(arrays):
    if ia.is_np_array(arrays):
        return np.copy(arrays)
    else:
        assert ia.is_iterable(arrays), "Expected ndarray or iterable of ndarray, got type %s." % (type(arrays),)
        return [np.copy(array) for array in arrays]


@six.add_metaclass(ABCMeta)
class Augmenter(object):  # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Base class for Augmenter objects.
    All augmenters derive from this class.

    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        """
        Create a new Augmenter instance.

        Parameters
        ----------
        name : None or str, optional
            Name given to an Augmenter object. This name is used in ``print()``
            statements as well as find and remove functions.
            If None, ``UnnamedX`` will be used as the name, where ``X`` is the
            Augmenter's class name.

        deterministic : bool, optional
            Whether the augmenter instance's random state will be saved before
            augmenting images and then reset to that saved state after an
            augmentation (of multiple images/keypoints) is finished.
            I.e. if set to True, each batch of images will be augmented in the
            same way (e.g. first image might always be flipped horizontally,
            second image will never be flipped etc.).
            This is useful when you want to transform multiple batches of images
            in the same way, or when you want to augment images and keypoints
            on these images.
            Usually, there is no need to set this variable by hand. Instead,
            instantiate the augmenter with the defaults and then use
            :func:`imgaug.augmenters.Augmenter.to_deterministic`.

        random_state : None or int or numpy.random.RandomState, optional
            The random state to use for this augmenter.

                * If int, a new ``numpy.random.RandomState`` will be created using this
                  value as the seed.
                * If ``numpy.random.RandomState`` instance, the instance will be used directly.
                * If None, imgaug's default RandomState will be used, which's state can
                  be controlled using ``imgaug.seed(int)``.

            Usually there is no need to set this variable by hand. Instead,
            instantiate the augmenter with the defaults and then use
            :func:`imgaug.augmenters.Augmenter.to_deterministic`.

        """
        super(Augmenter, self).__init__()

        ia.do_assert(name is None or ia.is_string(name),
                     "Expected name to be None or string-like, got %s." % (type(name),))
        if name is None:
            self.name = "Unnamed%s" % (self.__class__.__name__,)
        else:
            self.name = name

        ia.do_assert(ia.is_single_bool(deterministic),
                     "Expected deterministic to be a boolean, got %s." % (type(deterministic),))
        self.deterministic = deterministic

        if random_state is None:
            if self.deterministic:
                self.random_state = ia.new_random_state()
            else:
                self.random_state = ia.current_random_state()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        self.activated = True

    def augment_batches(self, batches, hooks=None, background=False):
        """
        Augment multiple batches of images.

        In contrast to other augment functions, this function _yields_ batches instead of just
        returning a full list. This is more suited for most training loops. It also supports
        augmentation on multiple cpu cores, activated via the `background` flag.

        Parameters
        ----------
        batches : imgaug.Batch or list of imgaug.Batch or list of imgaug.HeatmapsOnImage\
                  or list of imgaug.SegmentationMapOnImage or list of imgaug.KeypointsOnImage\
                  or list of ([N],H,W,[C]) ndarray
            List of batches to augment.
            The expected input is a list, with each entry having one of the following datatypes:

                * imgaug.Batch
                * []
                * list of imgaug.HeatmapsOnImage
                * list of imgaug.SegmentationMapOnImage
                * list of imgaug.KeypointsOnImage
                * list of imgaug.BoundingBoxesOnImage
                * list of (H,W,C) ndarray
                * list of (H,W) ndarray
                * (N,H,W,C) ndarray
                * (N,H,W) ndarray

            where ``N`` is the number of images, ``H`` is the height, ``W`` is the width, ``C`` is the number of
            channels. Each image is recommended to have dtype uint8 (range 0-255).

        hooks : None or imgaug.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation process.

        background : bool, optional
            Whether to augment the batches in background processes.
            If true, hooks can currently not be used as that would require
            pickling functions.

        Yields
        -------
        augmented_batch : ia.Batch\
                          or list of ia.HeatmapsOnImage\
                          or list of ia.SegmentationMapOnImage\
                          or list of ia.KeypointsOnImage\
                          or list of ia.BoundingBoxesOnImage\
                          or list of (H,W,C) ndarray\
                          or list of (H,W) ndarray\
                          or list of (N,H,W,C) ndarray\
                          or list of (N,H,W) ndarray
            Augmented objects.

        """
        ia.do_assert(isinstance(batches, list))
        ia.do_assert(len(batches) > 0)
        if background:
            ia.do_assert(hooks is None, "Hooks can not be used when background augmentation is activated.")

        batches_normalized = []
        batches_original_dts = []
        for i, batch in enumerate(batches):
            if isinstance(batch, ia.Batch):
                batch_copy = batch.deepcopy()
                batch_copy.data = (i, batch_copy.data)
                batches_normalized.append(batch_copy)
                batches_original_dts.append("imgaug.Batch")
            elif ia.is_np_array(batch):
                ia.do_assert(batch.ndim in (3, 4),
                             "Expected numpy array to have shape (N, H, W) or (N, H, W, C), got %s." % (batch.shape,))
                batches_normalized.append(ia.Batch(images=batch, data=i))
                batches_original_dts.append("numpy_array")
            elif isinstance(batch, list):
                if len(batch) == 0:
                    batches_normalized.append(ia.Batch(data=i))
                    batches_original_dts.append("empty_list")
                elif ia.is_np_array(batch[0]):
                    batches_normalized.append(ia.Batch(images=batch, data=i))
                    batches_original_dts.append("list_of_numpy_arrays")
                elif isinstance(batch[0], ia.HeatmapsOnImage):
                    batches_normalized.append(ia.Batch(heatmaps=batch, data=i))
                    batches_original_dts.append("list_of_imgaug.HeatmapsOnImage")
                elif isinstance(batch[0], ia.SegmentationMapOnImage):
                    batches_normalized.append(ia.Batch(segmentation_maps=batch, data=i))
                    batches_original_dts.append("list_of_imgaug.SegmentationMapOnImage")
                elif isinstance(batch[0], ia.KeypointsOnImage):
                    batches_normalized.append(ia.Batch(keypoints=batch, data=i))
                    batches_original_dts.append("list_of_imgaug.KeypointsOnImage")
                elif isinstance(batch[0], ia.BoundingBoxesOnImage):
                    batches_normalized.append(ia.Batch(bounding_boxes=batch, data=i))
                    batches_original_dts.append("list_of_imgaug.BoundingBoxesOnImage")
                else:
                    raise Exception(
                        "Unknown datatype in batch[0]. Expected numpy array or imgaug.HeatmapsOnImage or "
                        + "imgaug.SegmentationMapOnImage or imgaug.KeypointsOnImage or imgaug.BoundingBoxesOnImage, "
                        + "got %s." % (type(batch[0]),))
            else:
                raise Exception(
                    "Unknown datatype of batch. Expected imgaug.Batch or numpy array or list of (numpy array or "
                    + "imgaug.HeatmapsOnImage or imgaug.SegmentationMapOnImage or imgaug.KeypointsOnImage or "
                    + "imgaug.BoundingBoxesOnImage). "
                    + "Got %s." % (type(batch),))

        def unnormalize_batch(batch_aug):
            i = batch_aug.data
            # if input was ia.Batch, then .data has content (i, .data)
            if isinstance(i, tuple):
                i = i[0]
            dt_orig = batches_original_dts[i]
            if dt_orig == "imgaug.Batch":
                batch_unnormalized = batch_aug
                # change (i, .data) back to just .data
                batch_unnormalized.data = batch_unnormalized.data[1]
            elif dt_orig == "numpy_array":
                batch_unnormalized = batch_aug.images_aug
            elif dt_orig == "empty_list":
                batch_unnormalized = []
            elif dt_orig == "list_of_numpy_arrays":
                batch_unnormalized = batch_aug.images_aug
            elif dt_orig == "list_of_imgaug.HeatmapsOnImage":
                batch_unnormalized = batch_aug.heatmaps_aug
            elif dt_orig == "list_of_imgaug.SegmentationMapOnImage":
                batch_unnormalized = batch_aug.segmentation_maps_aug
            elif dt_orig == "list_of_imgaug.KeypointsOnImage":
                batch_unnormalized = batch_aug.keypoints_aug
            else:  # only option left
                ia.do_assert(dt_orig == "list_of_imgaug.BoundingBoxesOnImage")
                batch_unnormalized = batch_aug.bounding_boxes_aug
            return batch_unnormalized

        if not background:
            # singlecore augmentation

            for batch_normalized in batches_normalized:
                batch_augment_images = batch_normalized.images_unaug is not None
                batch_augment_heatmaps = batch_normalized.heatmaps_unaug is not None
                batch_augment_segmaps = batch_normalized.segmentation_maps_unaug is not None
                batch_augment_keypoints = batch_normalized.keypoints_unaug is not None
                batch_augment_bounding_boxes = batch_normalized.bounding_boxes_unaug is not None

                nb_to_aug = sum([1 if to_aug else 0
                                 for to_aug in [batch_augment_images, batch_augment_heatmaps, batch_augment_segmaps,
                                                batch_augment_keypoints, batch_augment_bounding_boxes]])

                if nb_to_aug > 1:
                    augseq = self.to_deterministic() if not self.deterministic else self
                else:
                    augseq = self

                if batch_augment_images:
                    batch_normalized.images_aug = augseq.augment_images(
                        batch_normalized.images_unaug, hooks=hooks)
                if batch_augment_heatmaps:
                    batch_normalized.heatmaps_aug = augseq.augment_heatmaps(
                        batch_normalized.heatmaps_unaug, hooks=hooks)
                if batch_augment_segmaps:
                    batch_normalized.segmentation_maps_aug = augseq.augment_segmentation_maps(
                        batch_normalized.segmentation_maps_unaug, hooks=hooks)
                if batch_augment_keypoints:
                    batch_normalized.keypoints_aug = augseq.augment_keypoints(
                        batch_normalized.keypoints_unaug, hooks=hooks)
                if batch_augment_bounding_boxes:
                    batch_normalized.bounding_boxes_aug = augseq.augment_bounding_boxes(
                        batch_normalized.bounding_boxes_unaug, hooks=hooks)

                batch_unnormalized = unnormalize_batch(batch_normalized)

                yield batch_unnormalized
        else:
            # multicore augmentation
            import imgaug.multicore as multicore

            # TODO skip this if the input is already a generator
            def load_batches():
                for batch in batches_normalized:
                    yield batch

            with multicore.Pool(self) as pool:
                for batch_aug in pool.imap_batches(load_batches()):
                    yield unnormalize_batch(batch_aug)

    def augment_image(self, image, hooks=None):
        """
        Augment a single image.

        Parameters
        ----------
        image : (H,W,C) ndarray or (H,W) ndarray
            The image to augment.
            Channel-axis is optional, but expected to be the last axis if present.
            In most cases, this array should be of dtype ``uint8``, which is supported by all
            augmenters. Support for other dtypes varies by augmenter -- see the respective
            augmenter-specific documentation for more details.

        hooks : None or imgaug.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation process.

        Returns
        -------
        img : ndarray
            The corresponding augmented image.

        """
        ia.do_assert(image.ndim in [2, 3],
                     "Expected image to have shape (height, width, [channels]), got shape %s." % (image.shape,))
        return self.augment_images([image], hooks=hooks)[0]

    def augment_images(self, images, parents=None, hooks=None):
        """
        Augment multiple images.

        Parameters
        ----------
        images : (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            Images to augment.
            The input can be a list of numpy arrays or a single array. Each array is expected to
            have shape ``(H, W, C)`` or ``(H, W)``, where H is the height, ``W`` is the width and
            ``C`` are the channels. Number of channels may differ between images.
            If a list is chosen, height and width may differ per between images.
            In most cases, this array (or these arrays) should be of dtype ``uint8``, which is
            supported by all augmenters. Support for other dtypes varies by augmenter -- see the
            respective augmenter-specific documentation for more details.

        parents : None or list of imgaug.augmenters.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : None or imgaug.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation process.

        Returns
        -------
        images_result : ndarray or list
            Corresponding augmented images.

        """
        if parents is not None and len(parents) > 0 and hooks is None:
            # This is a child call. The data has already been validated and copied. We don't need to copy it again
            # for hooks, as these don't exist. So we can augment here fully in-place.
            if not self.activated or len(images) == 0:
                return images

            if self.deterministic:
                state_orig = self.random_state.get_state()

            images_result = self._augment_images(
                images,
                random_state=ia.copy_random_state(self.random_state),
                parents=parents,
                hooks=hooks
            )
            # move "forward" the random state, so that the next call to
            # augment_images() will use different random values
            ia.forward_random_state(self.random_state)

            if self.deterministic:
                self.random_state.set_state(state_orig)

            return images_result

        #
        # Everything below is for non-in-place augmentation.
        # It was either the first call (no parents) or hooks were provided.
        #
        if self.deterministic:
            state_orig = self.random_state.get_state()

        if parents is None:
            parents = []

        if ia.is_np_array(images):
            input_type = "array"
            input_added_axis = False

            ia.do_assert(images.ndim in [3, 4],
                         "Expected 3d/4d array of form (N, height, width) or (N, height, width, channels), "
                         "got shape %s." % (images.shape,))

            # copy the input, we don't want to augment it in-place
            images_copy = np.copy(images)

            if images_copy.ndim == 3 and images_copy.shape[-1] in [1, 3]:
                warnings.warn("You provided a numpy array of shape %s as input to augment_images(), "
                              "which was interpreted as (N, H, W). The last dimension however has "
                              "value 1 or 3, which indicates that you provided a single image "
                              "with shape (H, W, C) instead. If that is the case, you should use "
                              "augment_image(image) or augment_images([image]), otherwise "
                              "you will not get the expected augmentations." % (images_copy.shape,))

            # for 2D input images (i.e. shape (N, H, W)), we add a channel axis (i.e. (N, H, W, 1)),
            # so that all augmenters can rely on the input having a channel axis and
            # don't have to add if/else statements for 2D images
            if images_copy.ndim == 3:
                images_copy = images_copy[..., np.newaxis]
                input_added_axis = True
        elif ia.is_iterable(images):
            input_type = "list"
            input_added_axis = []

            if len(images) == 0:
                images_copy = []
            else:
                ia.do_assert(all(image.ndim in [2, 3] for image in images),
                             "Expected list of images with each image having shape (height, width) or "
                             + "(height, width, channels), got shapes %s." % ([image.shape for image in images],))

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
            raise Exception("Expected images as one numpy array or list/tuple of numpy arrays, got %s." % (
                type(images),))

        if hooks is not None:
            images_copy = hooks.preprocess(images_copy, augmenter=self, parents=parents)

        # the is_activated() call allows to use hooks that selectively
        # deactivate specific augmenters in previously defined augmentation
        # sequences
        if (hooks is None and self.activated) \
                or (hooks is not None
                    and hooks.is_activated(images_copy, augmenter=self, parents=parents, default=self.activated)):
            if len(images) > 0:
                images_result = self._augment_images(
                    images_copy,
                    random_state=ia.copy_random_state(self.random_state),
                    parents=parents,
                    hooks=hooks
                )
                # move "forward" the random state, so that the next call to
                # augment_images() will use different random values
                ia.forward_random_state(self.random_state)
            else:
                images_result = images_copy
        else:
            images_result = images_copy

        if hooks is not None:
            images_result = hooks.postprocess(images_result, augmenter=self, parents=parents)

        # remove temporarily added channel axis for 2D input images
        output_type = "list" if isinstance(images_result, list) else "array"
        if input_type == "array":
            if input_added_axis is True:
                if output_type == "array":
                    images_result = np.squeeze(images_result, axis=3)
                else:
                    images_result = [np.squeeze(image, axis=2) for image in images_result]
        else:  # if input_type == "list":
            # This test was removed for now because hooks can change the type
            # ia.do_assert(
            #    isinstance(images_result, list),
            #    "INTERNAL ERROR: Input was list, output was expected to be list too "
            #    "but got %s." % (type(images_result),)
            # )

            ia.do_assert(
                len(images_result) == len(images),
                "INTERNAL ERROR: Expected number of images to be unchanged after augmentation, "
                "but was changed from %d to %d." % (len(images), len(images_result))
            )
            for i in sm.xrange(len(images_result)):
                if input_added_axis[i] is True:
                    images_result[i] = np.squeeze(images_result[i], axis=2)

        if self.deterministic:
            self.random_state.set_state(state_orig)

        return images_result

    @abstractmethod
    def _augment_images(self, images, random_state, parents, hooks):
        """
        Augment multiple images.

        This is the internal variation of `augment_images()`.
        It is called from `augment_images()` and should usually not be called
        directly.
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
            Either a list of ``(H, W, C)`` arrays or a single ``(N, H, W, C)`` array,
            where ``N`` is the number of images, ``H`` is the height of images, ``W``
            is the width of images and ``C`` is the number of channels of images.
            In the case of a list as input, ``H``, ``W`` and ``C`` may change per image.

        random_state : numpy.random.RandomState
            The random state to use for all sampling tasks during the augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`imgaug.augmenters.meta.Augmenter.augment_images`.

        hooks : imgaug.HooksImages or None
            See :func:`imgaug.augmenters.meta.Augmenter.augment_images`.

        Returns
        ----------
        images : (N,H,W,C) ndarray or list of (H,W,C) ndarray
            The augmented images.

        """
        raise NotImplementedError()

    def augment_heatmaps(self, heatmaps, parents=None, hooks=None):
        """
        Augment a heatmap.

        Parameters
        ----------
        heatmaps : imgaug.HeatmapsOnImage or list of imgaug.HeatmapsOnImage
            Heatmap(s) to augment. Either a single heatmap or a list of
            heatmaps.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : None or imaug.HooksHeatmaps, optional
            HooksHeatmaps object to dynamically interfere with the augmentation process.

        Returns
        -------
        heatmap_result : imgaug.HeatmapsOnImage or list of imgaug.HeatmapsOnImage
            Corresponding augmented heatmap(s).

        """
        if self.deterministic:
            state_orig = self.random_state.get_state()

        if parents is None:
            parents = []

        input_was_single_instance = False
        if isinstance(heatmaps, ia.HeatmapsOnImage):
            input_was_single_instance = True
            heatmaps = [heatmaps]

        ia.do_assert(ia.is_iterable(heatmaps),
                     "Expected to get list of imgaug.HeatmapsOnImage() instances, got %s." % (type(heatmaps),))
        ia.do_assert(all([isinstance(heatmaps_i, ia.HeatmapsOnImage) for heatmaps_i in heatmaps]),
                     "Expected to get list of imgaug.HeatmapsOnImage() instances, got %s." % (
                         [type(el) for el in heatmaps],))

        # copy, but only if topmost call or hooks are provided
        if len(parents) == 0 or hooks is not None:
            heatmaps_copy = [heatmaps_i.deepcopy() for heatmaps_i in heatmaps]
        else:
            heatmaps_copy = heatmaps

        if hooks is not None:
            heatmaps_copy = hooks.preprocess(heatmaps_copy, augmenter=self, parents=parents)

        if (hooks is None and self.activated) \
                or (hooks is not None
                    and hooks.is_activated(heatmaps_copy, augmenter=self, parents=parents, default=self.activated)):
            if len(heatmaps_copy) > 0:
                heatmaps_result = self._augment_heatmaps(
                    heatmaps_copy,
                    random_state=ia.copy_random_state(self.random_state),
                    parents=parents,
                    hooks=hooks
                )
                ia.forward_random_state(self.random_state)
            else:
                heatmaps_result = heatmaps_copy
        else:
            heatmaps_result = heatmaps_copy

        if hooks is not None:
            heatmaps_result = hooks.postprocess(heatmaps_result, augmenter=self, parents=parents)

        if self.deterministic:
            self.random_state.set_state(state_orig)

        if input_was_single_instance:
            return heatmaps_result[0]
        return heatmaps_result

    @abstractmethod
    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        """
        Augment heatmaps on multiple images.

        This is the internal version of ``augment_heatmaps()``.
        It is called from ``augment_heatmaps()`` and should usually not be called
        directly.
        This method may heatmaps in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        Parameters
        ----------
        heatmaps : list of imgaug.HeatmapsOnImage
            Heatmaps to augment. They may be changed in-place.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`imgaug.augmenters.meta.Augmenter.augment_heatmaps`.

        hooks : imgaug.HooksHeatmaps
            See :func:`imgaug.augmenters.meta.Augmenter.augment_heatmaps`.

        Returns
        ----------
        images : list of imgaug.HeatmapsOnImage
            The augmented heatmaps.

        """
        raise NotImplementedError()

    def _augment_heatmaps_as_images(self, heatmaps, parents, hooks):
        # TODO documentation
        # TODO keep this? it is afaik not used anywhere
        heatmaps_uint8 = [heatmaps_i.to_uint8() for heatmaps_i in heatmaps]
        heatmaps_uint8_aug = [
            self.augment_images([heatmaps_uint8_i], parents=parents, hooks=hooks)[0]
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
        """
        Augment segmentation maps.

        Parameters
        ----------
        segmaps : imgaug.SegmentationMapOnImage or \
                  list of imgaug.SegmentationMapOnImage
            Segmentation map(s) to augment. Either a single heatmap or a list of
            segmentation maps.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : None or imgaug.HooksHeatmaps, optional
            HooksHeatmaps object to dynamically interfere with the augmentation process.

        Returns
        -------
        segmaps_aug : imgaug.SegmentationMapOnImage or \
                      list of imgaug.SegmentationMapOnImage
            Corresponding augmented segmentation map(s).

        """
        input_was_single_instance = False
        if isinstance(segmaps, ia.SegmentationMapOnImage):
            input_was_single_instance = True
            segmaps = [segmaps]

        heatmaps_with_nonempty = [segmap.to_heatmaps(only_nonempty=True, not_none_if_no_nonempty=True)
                                  for segmap in segmaps]
        heatmaps = [heatmaps_i for heatmaps_i, nonempty_class_indices_i in heatmaps_with_nonempty]
        nonempty_class_indices = [nonempty_class_indices_i
                                  for heatmaps_i, nonempty_class_indices_i in heatmaps_with_nonempty]
        heatmaps_aug = self.augment_heatmaps(heatmaps, parents=parents, hooks=hooks)
        segmaps_aug = []
        for segmap, heatmaps_aug_i, nonempty_class_indices_i in zip(segmaps, heatmaps_aug, nonempty_class_indices):
            segmap_aug = ia.SegmentationMapOnImage.from_heatmaps(heatmaps_aug_i,
                                                                 class_indices=nonempty_class_indices_i,
                                                                 nb_classes=segmap.nb_classes)
            segmap_aug.input_was = segmap.input_was
            segmaps_aug.append(segmap_aug)

        if input_was_single_instance:
            return segmaps_aug[0]
        return segmaps_aug

    def augment_keypoints(self, keypoints_on_images, parents=None, hooks=None):
        """
        Augment image keypoints.

        This is the corresponding function to ``augment_images()``, just for
        keypoints/landmarks (i.e. coordinates on the image).
        Usually you will want to call ``augment_images()`` with a list of images,
        e.g. ``augment_images([A, B, C])`` and then ``augment_keypoints()`` with the
        corresponding list of keypoints on these images, e.g.
        ``augment_keypoints([Ak, Bk, Ck])``, where ``Ak`` are the keypoints on
        image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding keypoints,
        e.g. by

        >>> A = B = C = np.zeros((10, 10), dtype=np.uint8)
        >>> Ak = Bk = Ck = ia.KeypointsOnImage([ia.Keypoint(2, 2)], (10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> kps_aug = seq_det.augment_keypoints([Ak, Bk, Ck])

        Otherwise, different random values will be sampled for the image
        and keypoint augmentations, resulting in different augmentations (e.g.
        images might be rotated by ``30deg`` and keypoints by ``-10deg``).
        Also make sure to call `to_deterministic()` again for each new batch,
        otherwise you would augment all batches in the same way.

        Parameters
        ----------
        keypoints_on_images : imgaug.KeypointsOnImage or \
                              list of imgaug.KeypointsOnImage
            The keypoints/landmarks to augment.
            Expected is an instance of imgaug.KeypointsOnImage or a list of
            imgaug.KeypointsOnImage objects, with each such object containing
            the keypoints of a single image.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : None or imgaug.HooksKeypoints, optional
            HooksKeypoints object to dynamically interfere with the
            augmentation process.

        Returns
        -------
        keypoints_on_images_result : imgaug.KeypointsOnImage or \
                                     list of imgaug.KeypointsOnImage
            Augmented keypoints.

        """
        if self.deterministic:
            state_orig = self.random_state.get_state()

        if parents is None:
            parents = []

        input_was_single_instance = False
        if isinstance(keypoints_on_images, ia.KeypointsOnImage):
            input_was_single_instance = True
            keypoints_on_images = [keypoints_on_images]

        ia.do_assert(ia.is_iterable(keypoints_on_images))
        ia.do_assert(all([isinstance(keypoints_on_image, ia.KeypointsOnImage)
                          for keypoints_on_image in keypoints_on_images]))

        # copy, but only if topmost call or hooks are provided
        if len(parents) == 0 or hooks is not None:
            keypoints_on_images_copy = [keypoints_on_image.deepcopy() for keypoints_on_image in keypoints_on_images]
        else:
            keypoints_on_images_copy = keypoints_on_images

        if hooks is not None:
            keypoints_on_images_copy = hooks.preprocess(keypoints_on_images_copy, augmenter=self, parents=parents)

        if (hooks is None and self.activated) \
                or (hooks is not None
                    and hooks.is_activated(keypoints_on_images_copy,
                                           augmenter=self, parents=parents, default=self.activated)):
            if len(keypoints_on_images_copy) > 0:
                keypoints_on_images_result = self._augment_keypoints(
                    keypoints_on_images_copy,
                    random_state=ia.copy_random_state(self.random_state),
                    parents=parents,
                    hooks=hooks
                )
                ia.forward_random_state(self.random_state)
            else:
                keypoints_on_images_result = keypoints_on_images_copy
        else:
            keypoints_on_images_result = keypoints_on_images_copy

        if hooks is not None:
            keypoints_on_images_result = hooks.postprocess(keypoints_on_images_result, augmenter=self, parents=parents)

        if self.deterministic:
            self.random_state.set_state(state_orig)

        if input_was_single_instance:
            return keypoints_on_images_result[0]
        return keypoints_on_images_result

    @abstractmethod
    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        """
        Augment keypoints on multiple images.

        This is the internal variation of ``augment_keypoints()``.
        It is called from ``augment_keypoints()`` and should usually not be called directly.
        It has to be implemented by every augmenter.
        This method may transform the keypoints in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        Parameters
        ----------
        keypoints_on_images : list of imgaug.KeypointsOnImage
            Keypoints to augment. They may be changed in-place.

        random_state : numpy.random.RandomState
            The random state to use for all sampling tasks during the augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`imgaug.augmenters.meta.Augmenter.augment_keypoints`.

        hooks : imgaug.HooksImages
            See :func:`imgaug.augmenters.meta.Augmenter.augment_keypoints`.

        Returns
        ----------
        images : list of imgaug.KeypointsOnImage
            The augmented keypoints.

        """
        raise NotImplementedError()

    def augment_bounding_boxes(self, bounding_boxes_on_images, hooks=None):
        """
        Augment image bounding boxes.

        This is the corresponding function to ``augment_keypoints()``, just for
        bounding boxes.
        Usually you will want to call ``augment_images()`` with a list of images,
        e.g. ``augment_images([A, B, C])`` and then ``augment_bounding_boxes()``
        with the corresponding list of bounding boxes on these images, e.g.
        ``augment_bounding_boxes([Abb, Bbb, Cbb])``, where ``Abb`` are the
        bounding boxes on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding bounding boxes,
        e.g. by

        >>> A = B = C = np.ones((10, 10), dtype=np.uint8)
        >>> Abb = Bbb = Cbb = ia.BoundingBoxesOnImage([ia.BoundingBox(1, 1, 9, 9)], (10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> bbs_aug = seq_det.augment_bounding_boxes([Abb, Bbb, Cbb])

        Otherwise, different random values will be sampled for the image
        and bounding box augmentations, resulting in different augmentations
        (e.g. images might be rotated by ``30deg`` and bounding boxes by
        ``-10deg``). Also make sure to call ``to_deterministic()`` again for each
        new batch, otherwise you would augment all batches in the same way.

        Parameters
        ----------
        bounding_boxes_on_images : imgaug.BoundingBoxesOnImage or \
                                   list of imgaug.BoundingBoxesOnImage
            The bounding boxes to augment.
            Expected is an instance of imgaug.BoundingBoxesOnImage or a list of
            imgaug.BoundingBoxesOnImage objects, witch each such object
            containing the bounding boxes of a single image.

        hooks : None or imgaug.HooksKeypoints, optional
            HooksKeypoints object to dynamically interfere with the
            augmentation process.

        Returns
        -------
        result : imgaug.BoundingBoxesOnImage or \
                 list of imgaug.BoundingBoxesOnImage
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

    def pool(self, processes=None, maxtasksperchild=None, seed=None):
        """
        Create a pool used for multicore augmentation from this augmenter.

        Parameters
        ----------
        processes : None or int, optional
            Same as for :func:`imgaug.multicore.Pool.__init__`.
            The number of background workers, similar to the same parameter in multiprocessing.Pool.
            If ``None``, the number of the machine's CPU cores will be used (this counts hyperthreads as CPU cores).
            If this is set to a negative value ``p``, then ``P - abs(p)`` will be used, where ``P`` is the number
            of CPU cores. E.g. ``-1`` would use all cores except one (this is useful to e.g. reserve one core to
            feed batches to the GPU).

        maxtasksperchild : None or int, optional
            Same as for :func:`imgaug.multicore.Pool.__init__`.
            The number of tasks done per worker process before the process is killed and restarted, similar to the
            same parameter in multiprocessing.Pool. If ``None``, worker processes will not be automatically restarted.

        seed : None or int, optional
            Same as for :func:`imgaug.multicore.Pool.__init__`.
            The seed to use for child processes. If ``None``, a random seed will be used.

        Returns
        -------
        imgaug.multicore.Pool
            Pool for multicore augmentation.

        Examples
        --------
        >>> import imgaug as ia
        >>> from imgaug import augmenters as iaa
        >>> import numpy as np
        >>> aug = iaa.Add(1)
        >>> images = np.zeros((16, 128, 128, 3), dtype=np.uint8)
        >>> batches = [ia.Batch(images=np.copy(images)) for _ in range(100)]
        >>> with aug.pool(processes=-1, seed=2) as pool:
        >>>     batches_aug = pool.map_batches(batches, chunksize=8)
        >>> print(np.sum(batches_aug[0].images_aug[0]))
        49152

        Creates ``100`` batches of empty images. Each batch contains ``16`` images of size ``128x128``. The batches
        are then augmented on all CPU cores except one (``processes=-1``). After augmentation, the sum of pixel values
        from the first augmented image is printed.

        >>> import imgaug as ia
        >>> from imgaug import augmenters as iaa
        >>> import numpy as np
        >>> aug = iaa.Dropout(0.2)
        >>> images = np.zeros((16, 128, 128, 3), dtype=np.uint8)
        >>> def generate_batches():
        >>>     for _ in range(100):
        >>>         yield ia.Batch(images=np.copy(images))
        >>>
        >>> with aug.pool(processes=-1, seed=2) as pool:
        >>>     batches_aug = pool.imap_batches(generate_batches(), chunksize=8)
        >>>     batch_aug = next(batches_aug)
        >>>     print(np.sum(batch_aug.images_aug[0]))
        0

        Same as above. This time, a generator is used to generate batches of images. Again, the first augmented image's
        sum of pixels is printed.

        """
        import imgaug.multicore as multicore
        return multicore.Pool(self, processes=processes, maxtasksperchild=maxtasksperchild, seed=seed)

    # TODO most of the code of this function could be replaced with ia.draw_grid()
    # TODO add parameter for handling multiple images ((a) next to each other in each row or (b) multiply row count
    # by number of images and put each one in a new row)
    # TODO "images" parameter deviates from augment_images (3d array is here treated as one 3d image, in
    # augment_images as (N, H, W))
    def draw_grid(self, images, rows, cols):
        """
        Apply this augmenter to the given images and return a grid image of the results.
        Each cell in the grid contains a single augmented variation of an input image.

        If multiple images are provided, the row count is multiplied by
        the number of images and each image gets its own row.
        E.g. for ``images = [A, B]``, ``rows=2``, ``cols=3``::

            A A A
            B B B
            A A A
            B B B

        for ``images = [A]``, ``rows=2``,
        ``cols=3``::

            A A A
            A A A

        Parameters
        -------
        images : (N,H,W,3) ndarray or (H,W,3) ndarray or (H,W) ndarray or list of (H,W,3) ndarray\
                 or list of (H,W) ndarray
            List of images of which to show the augmented versions.
            If a list, then each element is expected to have shape ``(H, W)`` or
            ``(H, W, 3)``. If a single array, then it is expected to have
            shape ``(N, H, W, 3)`` or ``(H, W, 3)`` or ``(H, W)``.

        rows : int
            Number of rows in the grid.
            If ``N`` input images are given, this value will automatically be
            multiplied by ``N`` to create rows for each image.

        cols : int
            Number of columns in the grid.

        Returns
        -------
        grid : (Hg, Wg, 3) ndarray
            The generated grid image with augmented versions of the input
            images. Here, ``Hg`` and ``Wg`` reference the output size of the grid,
            and *not* the sizes of the input images.

        """
        if ia.is_np_array(images):
            if len(images.shape) == 4:
                images = [images[i] for i in range(images.shape[0])]
            elif len(images.shape) == 3:
                images = [images]
            elif len(images.shape) == 2:
                images = [images[:, :, np.newaxis]]
            else:
                raise Exception("Unexpected images shape, expected 2-, 3- or 4-dimensional array, "
                                + "got shape %s." % (images.shape,))
        elif isinstance(images, list):
            for i, image in enumerate(images):
                if len(image.shape) == 3:
                    continue
                elif len(image.shape) == 2:
                    images[i] = image[:, :, np.newaxis]
                else:
                    raise Exception(("Unexpected image shape at index %d, expected 2- or 3-dimensional array, "
                                     + "got shape %s.") % (i, image.shape,))
        ia.do_assert(isinstance(images, list))

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
        """
        Apply this augmenter to the given images and show/plot the results as a grid of images.

        If multiple images are provided, the row count is multiplied by
        the number of images and each image gets its own row.
        E.g. for ``images = [A, B]``, ``rows=2``, ``cols=3``::

            A A A
            B B B
            A A A
            B B B

        for ``images = [A]``, ``rows=2``,
        ``cols=3``::

            A A A
            A A A

        Parameters
        ----------
        images : (N,H,W,3) ndarray or (H,W,3) ndarray or (H,W) ndarray or list of (H,W,3) ndarray\
                 or list of (H,W) ndarray
            List of images of which to show the augmented versions.
            If a list, then each element is expected to have shape ``(H, W)`` or ``(H, W, 3)``.
            If a single array, then it is expected to have shape ``(N, H, W, 3)``
            or ``(H, W, 3)`` or ``(H, W)``.

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
        """
        Converts this augmenter from a stochastic to a deterministic one.

        A stochastic augmenter samples new values for each parameter per image.
        Feed a new batch of images into the augmenter and you will get a
        new set of transformations.
        A deterministic augmenter also samples new values for each parameter
        per image, but starts each batch with the same RandomState (i.e. seed).
        Feed two batches of images into the augmenter and you get the same
        transformations both times (same number of images assumed; some
        augmenter's results are also dependend on image height, width and
        channel count).

        Using determinism is useful for keypoint augmentation,
        as you will usually want to augment images and their corresponding
        keypoints in the same way (e.g. if an image is rotated by ``30deg``, then
        also rotate its keypoints by ``30deg``).

        Parameters
        ----------
        n : None or int, optional
            Number of deterministic augmenters to return.
            If None then only one Augmenter object will be returned.
            If 1 or higher, then a list containing `n` Augmenter objects
            will be returned.

        Returns
        -------
        det : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter
            A single Augmenter object if `n` was None,
            otherwise a list of Augmenter objects (even if `n` was 1).

        """
        ia.do_assert(n is None or n >= 1)
        if n is None:
            return self.to_deterministic(1)[0]
        else:
            return [self._to_deterministic() for _ in sm.xrange(n)]

    def _to_deterministic(self):
        """
        Augmenter-specific implementation of ``to_deterministic()``.
        This function is expected to return a single new deterministic
        Augmenter object of this augmenter.

        Returns
        -------
        det : imgaug.augmenters.meta.Augmenter
            Deterministic variation of this Augmenter object.

        """
        aug = self.copy()

        # This was changed for 0.2.8 from deriving a new random state based on the global random state to deriving
        # it from the augmenter's local random state. This should reduce the risk that re-runs of scripts lead to
        # different results upon small changes somewhere. It also decreases the likelihood of problems when using
        # multiprocessing (the child processes might use the same global random state as the parent process).
        # Note for the latter point that augment_batches() might call to_deterministic() if the batch contains
        # multiply types of augmentables.
        # aug.random_state = ia.new_random_state()
        aug.random_state = ia.derive_random_state(self.random_state)

        aug.deterministic = True
        return aug

    def reseed(self, random_state=None, deterministic_too=False):
        """
        Reseed this augmenter and all of its children (if it has any).

        This function is useful, when augmentations are run in the
        background (i.e. on multiple cores).
        It should be called before sending this Augmenter object to a
        background worker or once within each worker with different seeds
        (i.e., if ``N`` workers are used, the function should be called
        ``N`` times). Otherwise, all background workers will
        use the same seeds and therefore apply the same augmentations.

        If this augmenter or any child augmenter had a random state that
        pointed to the global random state, it will automatically be
        replaced with a local random state. This is similar to what
        :func:`imgaug.augmenters.meta.Augmenter.localize_random_state`
        does.

        Parameters
        ----------
        random_state : None or int or numpy.random.RandomState, optional
            A RandomState that is used to sample seeds per augmenter.
            If int, the parameter will be used as a seed for a new RandomState.
            If None, a new RandomState will automatically be created.

        deterministic_too : bool, optional
            Whether to also change the seed of an augmenter ``A``, if ``A``
            is deterministic. This is the case both when this augmenter
            object is ``A`` or one of its children is ``A``.

        """
        ia.do_assert(isinstance(deterministic_too, bool))

        if random_state is None:
            random_state = ia.current_random_state()
        elif isinstance(random_state, np.random.RandomState):
            pass  # just use the provided random state without change
        else:
            random_state = ia.new_random_state(random_state)

        if not self.deterministic or deterministic_too:
            # TODO replace by ia.derive_random_state()
            seed = random_state.randint(0, 10**6, 1)[0]
            self.random_state = ia.new_random_state(seed)

        for lst in self.get_children_lists():
            for aug in lst:
                aug.reseed(random_state=random_state, deterministic_too=deterministic_too)

    def localize_random_state(self, recursive=True):
        """
        Converts global random states to local ones.
        See :func:`Augmenter.localize_random_state_` for more details.

        Parameters
        ----------
        recursive : bool, optional
            See :func:`imgaug.augmenters.meta.Augmenter.localize_random_state_`.

        Returns
        -------
        aug : imgaug.augmenters.meta.Augmenter
            Returns copy of augmenter and children, with localized random states.

        """
        aug = self.deepcopy()
        aug.localize_random_state_(
            recursive=recursive
        )
        return aug

    def localize_random_state_(self, recursive=True):
        """
        Converts global random states to local ones.

        A global random state exists exactly once. Many augmenters can point
        to it (and thereby use it to sample random numbers).
        Local random states usually exist for exactly one augmenter and are
        saved within that augmenter.

        Usually there is no need to change global into local random states.
        The only noteworthy exceptions are

            * whenever you want to use determinism (so that the global random
              state is not accidentally reverted)
            * whenever you want to copy random states from one augmenter to
              another. (Copying the global random state doesn't help very
              much. If you copy the state from A to B, then execute A and then
              B, B's (global) random state has already changed because of A's
              sampling.)

        The case of determinism is handled automatically by
        :func:`imgaug.augmenters.meta.Augmenter.to_deterministic`.
        Only when you copy random states (via :func:`imgaug.augmenters.meta.Augmenter.copy_random_state`),
        you need to call this function first.

        Parameters
        ----------
        recursive : bool, optional
            Whether to localize the random states of children too.

        Returns
        -------
        self : imgaug.augmenters.meta.Augmenter
            Returns itself (with localized random states).

        """
        if self.random_state == ia.current_random_state():
            self.random_state = ia.new_random_state()
        if recursive:
            for lst in self.get_children_lists():
                for child in lst:
                    child.localize_random_state_(recursive=recursive)
        return self

    def copy_random_state(self, source, recursive=True, matching="position", matching_tolerant=True,
                          copy_determinism=False):
        """
        Copy the random states from a source augmenter sequence.

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
        aug : imgaug.augmenters.meta.Augmenter
            Copy of the augmenter(s) with the same random state(s) as in the
            source augmenter(s).

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

    def copy_random_state_(self, source, recursive=True, matching="position", matching_tolerant=True,
                           copy_determinism=False):
        """
        Copy the random states from a source augmenter sequence (inplace).

        Parameters
        ----------
        source : imgaug.augmenters.meta.Augmenter
            The source augmenter from where to copy the random_state(s).
            May have children (e.g. a Sequential).
            May not use the global random state. This is used by default
            by all augmenters. Call :func:`imgaug.augmenters.meta.Augmenter.localize_random_state_`
            once on the source to localize all random states.

        recursive : bool, optional
            Whether to copy the random states of the source augmenter *and*
            all of its children (True) or just the source augmenter (False).

        matching : {'position', 'name'}, optional
            Defines the matching mode to use during recursive copy.
            This is used to associate source augmenters with target augmenters.
            If ``position`` then the target and source sequences of augmenters
            are turned into flattened lists and are associated based on
            their list indices. If ``name`` then the target and source augmenters
            are matched based on their names (i.e. ``augmenter.name``).

        matching_tolerant : bool, optional
            Whether to use tolerant matching between source and target
            augmenters. If set to False: Name matching will raise an exception
            for any target augmenter which's name does not appear among the
            source augmeters. Position matching will raise an exception if
            source and target augmenter have an unequal number of children.

        copy_determinism : bool, optional
            Whether to copy the ``deterministic`` flags from source to target
            augmenters too.

        Returns
        -------
        self : imgaug.augmenters.meta.Augmenter
            Returns itself (after random state copy).

        """
        source_augs = [source] + source.get_all_children(flat=True) if recursive else [source]
        target_augs = [self] + self.get_all_children(flat=True) if recursive else [self]

        global_rs = ia.current_random_state()
        global_rs_exc_msg = "You called copy_random_state_() with a source " \
                            "that uses global random states. Call " \
                            "localize_random_state_() on the source first " \
                            "or initialize your augmenters with local random " \
                            "states, e.g. via Dropout(..., random_state=1234)."

        if matching == "name":
            source_augs_dict = {aug.name: aug for aug in source_augs}
            target_augs_dict = {aug.name: aug for aug in target_augs}

            if len(source_augs_dict) < len(source_augs) or len(target_augs_dict) < len(target_augs):
                warnings.warn(
                    "Matching mode 'name' with recursive=True was chosen in copy_random_state_, "
                    "but either the source or target augmentation sequence contains multiple "
                    "augmenters with the same name."
                )

            for name in target_augs_dict:
                if name in source_augs_dict:
                    if source_augs_dict[name].random_state == global_rs:
                        raise Exception(global_rs_exc_msg)
                    target_augs_dict[name].random_state = ia.copy_random_state(source_augs_dict[name].random_state)
                    if copy_determinism:
                        target_augs_dict[name].deterministic = source_augs_dict[name].deterministic
                elif not matching_tolerant:
                    raise Exception(
                        "Augmenter name '%s' not found among source augmenters." % (name,)
                    )
        elif matching == "position":
            if len(source_augs) != len(target_augs) and not matching_tolerant:
                raise Exception(
                    "Source and target augmentation sequences have different lengths."
                )
            for source_aug, target_aug in zip(source_augs, target_augs):
                if source_aug.random_state == global_rs:
                    raise Exception(global_rs_exc_msg)
                target_aug.random_state = ia.copy_random_state(source_aug.random_state, force_copy=True)
                if copy_determinism:
                    target_aug.deterministic = source_aug.deterministic
        else:
            raise Exception("Unknown matching method '%s'. Valid options are 'name' and 'position'." % (matching,))

        return self

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError()

    def get_children_lists(self):
        """
        Get a list of lists of children of this augmenter.

        For most augmenters, the result will be a single empty list.
        For augmenters with children it will often be a list with one sublist containing all
        children. In some cases the augmenter will contain multiple distinct lists of children,
        e.g. an if-list and an else-list. This will lead to a result consisting of a single
        list with multiple sublists, each representing the respective sublist of children.

        E.g. for an if/else-augmenter that executes the children ``A1``, ``A2`` if a condition is met
        and otherwise executes the children ``B1``, ``B2``, ``B3`` the result will
        be ``[[A1, A2], [B1, B2, B3]]``.

        IMPORTANT: While the topmost list may be newly created, each of the sublist must be
        editable inplace resulting in a changed children list of the augmenter. E.g. if
        an Augmenter ``IfElse(condition, [A1, A2], [B1, B2, B3])`` returns ``[[A1, A2], [B1, B2, B3]]``
        for a call to :func:`imgaug.augmenters.meta.Augmenter.get_children_lists` and
        ``A2`` is removed inplace from ``[A1, A2]``, then the
        children lists of ``IfElse(...)`` must also change to ``[A1], [B1, B2, B3]``. This is used
        in :func:`imgaug.augmeneters.meta.Augmenter.remove_augmenters_inplace`.

        Returns
        -------
        children : list of list of imgaug.augmenters.meta.Augmenter
            One or more lists of child augmenter.
            Can also be a single empty list.

        """
        return []

    def get_all_children(self, flat=False):
        """
        Returns all children of this augmenter as a list.

        If the augmenter has no children, the returned list is empty.

        Parameters
        ----------
        flat : bool
            If set to True, the returned list will be flat.

        Returns
        -------
        result : list of imgaug.augmenters.meta.Augmenter
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
        """
        Find augmenters that match a condition.
        This function will compare this augmenter and all of its children
        with a condition. The condition is a lambda function.

        Parameters
        ----------
        func : callable
            A function that receives an Augmenter instance and a list of
            parent Augmenter instances and must return True, if that
            augmenter is valid match.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            List of parent augmenters.
            Intended for nested calls and can usually be left as None.

        flat : bool, optional
            Whether to return the result as a flat list (True)
            or a nested list (False). In the latter case, the nesting matches
            each augmenters position among the children.

        Returns
        ----------
        augmenters : list of imgaug.augmenters.meta.Augmenter
            Nested list if flat was set to False.
            Flat list if flat was set to True.

        Examples
        --------
        >>> aug = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>     iaa.Flipud(0.5, name="flipud")
        >>> ])
        >>> print(aug.find_augmenters(lambda a, parents: a.name == "fliplr"))

        This will return the first child augmenter (Fliplr instance).

        """
        if parents is None:
            parents = []

        result = []
        if func(self, parents):
            result.append(self)

        subparents = parents + [self]
        for lst in self.get_children_lists():
            for aug in lst:
                found = aug.find_augmenters(func, parents=subparents, flat=flat)
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
            Nested list if flat was set to False.
            Flat list if flat was set to True.

        """
        return self.find_augmenters_by_names([name], regex=regex, flat=flat)

    def find_augmenters_by_names(self, names, regex=False, flat=True):
        """
        Find augmenter(s) by names.

        Parameters
        ----------
        names : list of str
            Names of the augmenter(s) to search for.

        regex : bool, optional
            Whether `names` is a list of regular expressions.

        flat : boolean, optional
            See :func:`imgaug.augmenters.meta.Augmenter.find_augmenters`.

        Returns
        -------
        augmenters : list of imgaug.augmenters.meta.Augmenter
            Nested list if flat was set to False.
            Flat list if flat was set to True.

        """
        if regex:
            def comparer(aug, parents):
                for pattern in names:
                    if re.match(pattern, aug.name):
                        return True
                return False

            return self.find_augmenters(comparer, flat=flat)
        else:
            return self.find_augmenters(lambda aug, parents: aug.name in names, flat=flat)

    def remove_augmenters(self, func, copy=True, noop_if_topmost=True):
        """
        Remove this augmenter or its children that match a condition.

        Parameters
        ----------
        func : callable
            Condition to match per augmenter.
            The function must expect the augmenter itself and a list of parent
            augmenters and returns True if that augmenter is to be removed,
            or False otherwise.
            E.g. ``lambda a, parents: a.name == "fliplr" and len(parents) == 1``
            removes an augmenter with name "fliplr" if it is the direct child
            of the augmenter upon which ``remove_augmenters()`` was initially called.

        copy : bool, optional
            Whether to copy this augmenter and all if its children before
            removing. If False, removal is performed in-place.

        noop_if_topmost : bool, optional
            If True and the condition (lambda function) leads to the removal
            of the topmost augmenter (the one this function is called on
            initially), then that topmost augmenter will be replaced by a
            Noop instance (i.e. an object that will still offer ``augment_images()``,
            but does not change images). If False, None will be returned in
            these cases.
            This can only be False if copy is set to True.

        Returns
        -------
        aug : imgaug.augmenters.meta.Augmenter or None
            This augmenter after the removal was performed.
            Is None iff condition was matched for the topmost augmenter,
            copy was set to True and `noop_if_topmost` was set to False.

        Examples
        --------
        >>> seq = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>     iaa.Flipud(0.5, name="flipud"),
        >>> ])
        >>> seq = seq.remove_augmenters(lambda a, parents: a.name == "fliplr")

        This removes the augmenter Fliplr from the Sequential object's children.

        """
        if func(self, []):
            if not copy:
                raise Exception("Inplace removal of topmost augmenter requested, which is currently not possible.")

            if noop_if_topmost:
                return Noop()
            else:
                return None
        else:
            aug = self if not copy else self.deepcopy()
            aug.remove_augmenters_inplace(func, parents=[])
            return aug

    def remove_augmenters_inplace(self, func, parents=None):
        """
        Remove in-place children of this augmenter that match a condition.

        This is functionally identical to ``remove_augmenters()`` with
        ``copy=False``, except that it does not affect the topmost augmenter
        (the one on which this function is initially called on).

        Parameters
        ----------
        func : callable
            See :func:`imgaug.augmenters.meta.Augmenter.remove_augmenters`.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            List of parent Augmenter instances that lead to this
            Augmenter. If None, an empty list will be used.
            This parameter can usually be left empty and will be set
            automatically for children.

        Examples
        --------
        >>> seq = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>    iaa.Flipud(0.5, name="flipud"),
        >>> ])
        >>> seq.remove_augmenters_inplace(lambda a, parents: a.name == "fliplr")

        This removes the augmenter Fliplr from the Sequential object's children.

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
        """
        Create a shallow copy of this Augmenter instance.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Shallow copy of this Augmenter instance.

        """
        return copy_module.copy(self)

    def deepcopy(self):
        """
        Create a deep copy of this Augmenter instance.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Deep copy of this Augmenter instance.

        """
        return copy_module.deepcopy(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        params = self.get_parameters()
        params_str = ", ".join([param.__str__() for param in params])
        return "%s(name=%s, parameters=[%s], deterministic=%s)" % (
            self.__class__.__name__, self.name, params_str, self.deterministic)


class Sequential(Augmenter, list):
    """
    List augmenter that may contain other augmenters to apply in sequence
    or random order.

    NOTE: You are *not* forced to use `Sequential` in order to use other
    augmenters. Each augmenter can be used on its own, e.g the following
    defines an augmenter for horizontal flips and then augments a single
    image::

        aug = iaa.Fliplr(0.5)
        image_aug = aug.augment_image(image)

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
        Whether to apply the child augmenters in random order per image.
        The order is resampled for each image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> imgs = [np.random.rand(10, 10)]
    >>> seq = iaa.Sequential([
    >>>     iaa.Fliplr(0.5),
    >>>     iaa.Flipud(0.5)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Calls always first the horizontal flip augmenter and then the vertical
    flip augmenter (each having a probability of 50 percent to be used).

    >>> seq = iaa.Sequential([
    >>>     iaa.Fliplr(0.5),
    >>>     iaa.Flipud(0.5)
    >>> ], random_order=True)
    >>> imgs_aug = seq.augment_images(imgs)

    Calls sometimes first the horizontal flip augmenter and sometimes first the
    vertical flip augmenter (each again with 50 percent probability to be used).

    """

    def __init__(self, children=None, random_order=False, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        if children is None:
            list.__init__(self, [])
        elif isinstance(children, Augmenter):
            # this must be separate from `list.__init__(self, children)`,
            # otherwise in `Sequential(OneOf(...))` the OneOf(...) is
            # interpreted as a list and OneOf's children become Sequential's
            # children
            list.__init__(self, [children])
        elif ia.is_iterable(children):
            ia.do_assert(all([isinstance(child, Augmenter) for child in children]))
            list.__init__(self, children)
        else:
            raise Exception("Expected None or Augmenter or list of Augmenter, got %s." % (type(children),))

        ia.do_assert(ia.is_single_bool(random_order),
                     "Expected random_order to be boolean, got %s." % (type(random_order),))
        self.random_order = random_order

    def _augment_images(self, images, random_state, parents, hooks):
        if hooks is None or hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
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
        if hooks is None or hooks.is_propagating(heatmaps, augmenter=self, parents=parents, default=True):
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

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        if hooks is None or hooks.is_propagating(keypoints_on_images,
                                                     augmenter=self, parents=parents, default=True):
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

    def _to_deterministic(self):
        augs = [aug.to_deterministic() for aug in self]
        seq = self.copy()
        seq[:] = augs
        seq.random_state = ia.derive_random_state(self.random_state)
        seq.deterministic = True
        return seq

    def get_parameters(self):
        return [self.random_order]

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
        return "Sequential(name=%s, random_order=%s, children=[%s], deterministic=%s)" % (
            self.name, self.random_order, augs_str, self.deterministic)


class SomeOf(Augmenter, list):
    """
    List augmenter that applies only some of its children to images.

    E.g. this allows to define a list of 20 augmenters, but only apply a
    random selection of 5 of them to each image.

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

            * If int, then exactly `n` of the child augmenters are applied to
              every image.
            * If tuple of two ints ``(a, b)``, then ``a <= x <= b`` augmenters are
              picked and applied to every image. Here, b may be set to None,
              then it will automatically replaced with the total number of
              available children.
            * If StochasticParameter, then ``N`` numbers will be sampled for ``N`` images.
              The parameter is expected to be discrete.
            * If None, then the total number of available children will be
              used (i.e. all children will be applied).

    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        The augmenters to apply to images.

    random_order : boolean, optional
        Whether to apply the child augmenters in random order per image.
        The order is resampled for each image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> imgs = [np.random.rand(10, 10)]
    >>> seq = iaa.SomeOf(1, [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Applies either Fliplr or Flipud to images.

    >>> seq = iaa.SomeOf((1, 3), [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0),
    >>>     iaa.GaussianBlur(1.0)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Applies one to three of the listed augmenters (Fliplr, Flipud,
    GaussianBlur) to images. They are always applied in the
    order (1st) Fliplr, (2nd) Flipud, (3rd) GaussianBlur.

    >>> seq = iaa.SomeOf((1, None), [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0),
    >>>     iaa.GaussianBlur(1.0)
    >>> ], random_order=True)
    >>> imgs_aug = seq.augment_images(imgs)

    Applies one to all of the listed augmenters (Fliplr, Flipud,
    GaussianBlur) to images. They are applied in random order, i.e.
    sometimes Blur first, followed by Fliplr, sometimes Fliplr follow by
    Flipud followed by Blur, sometimes Flipud follow by Blur, etc.

    """

    def __init__(self, n=None, children=None, random_order=False, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        if children is None:
            list.__init__(self, [])
        elif isinstance(children, Augmenter):
            # this must be separate from `list.__init__(self, children)`,
            # otherwise in `SomeOf(OneOf(...))` the OneOf(...) is
            # interpreted as a list and OneOf's children become SomeOf's
            # children
            list.__init__(self, [children])
        elif ia.is_iterable(children):
            ia.do_assert(all([isinstance(child, Augmenter) for child in children]))
            list.__init__(self, children)
        else:
            raise Exception("Expected None or Augmenter or list of Augmenter, got %s." % (type(children),))

        if ia.is_single_number(n):
            self.n = int(n)
            self.n_mode = "deterministic"
        elif n is None:
            self.n = None
            self.n_mode = "None"
        elif ia.is_iterable(n):
            ia.do_assert(len(n) == 2)
            if ia.is_single_number(n[0]) and n[1] is None:
                self.n = (int(n[0]), None)
                self.n_mode = "(int,None)"
            elif ia.is_single_number(n[0]) and ia.is_single_number(n[1]):
                self.n = iap.DiscreteUniform(int(n[0]), int(n[1]))
                self.n_mode = "stochastic"
            else:
                raise Exception("Expected tuple of (int, None) or (int, int), got %s" % ([type(el) for el in n],))
        elif isinstance(n, iap.StochasticParameter):
            self.n = n
            self.n_mode = "stochastic"
        else:
            raise Exception("Expected int, (int, None), (int, int) or StochasticParameter, got %s" % (type(n),))

        ia.do_assert(ia.is_single_bool(random_order),
                     "Expected random_order to be boolean, got %s." % (type(random_order),))
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

    def _augment_images(self, images, random_state, parents, hooks):
        if hooks is None or hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
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
            augmenter_active = self._get_augmenter_active(len(images), random_state)

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
                    output_all_same_shape = len(set([img.shape for img in images_to_aug])) == 1

                    # Map them back to their position in the images array/list
                    # But it can happen that the augmented images have different shape(s) from
                    # the input image, as well as being suddenly a list instead of a numpy array.
                    # This is usually the case if a child augmenter has to change shapes, e.g.
                    # due to cropping (without resize afterwards). So accomodate here for that
                    # possibility.
                    if input_is_array:
                        if not output_is_array and output_all_same_shape:
                            images_to_aug = np.array(images_to_aug, dtype=images.dtype)
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
        if hooks is None or hooks.is_propagating(heatmaps, augmenter=self, parents=parents, default=True):
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
            augmenter_active = self._get_augmenter_active(len(heatmaps), random_state)

            for augmenter_index in augmenter_order:
                active = augmenter_active[:, augmenter_index].nonzero()[0]
                if len(active) > 0:
                    # pick images to augment, i.e. images for which
                    # augmenter at current index is active
                    heatmaps_to_aug = [heatmaps[idx] for idx in active]

                    # augment the images
                    heatmaps_aug = self[augmenter_index].augment_heatmaps(
                        heatmaps_to_aug,
                        parents=parents + [self],
                        hooks=hooks
                    )

                    # Map them back to their position in the images array/list
                    for aug_idx, original_idx in enumerate(active):
                        heatmaps[original_idx] = heatmaps_aug[aug_idx]

        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        if hooks is None or hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents, default=True):
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
            augmenter_active = self._get_augmenter_active(len(keypoints_on_images), random_state)

            for augmenter_index in augmenter_order:
                active = augmenter_active[:, augmenter_index].nonzero()[0]
                if len(active) > 0:
                    # pick images to augment, i.e. images for which
                    # augmenter at current index is active
                    koi_to_aug = [keypoints_on_images[idx] for idx in active]

                    # augment the images
                    koi_to_aug = self[augmenter_index].augment_keypoints(
                        keypoints_on_images=koi_to_aug,
                        parents=parents + [self],
                        hooks=hooks
                    )

                    # map them back to their position in the images array/list
                    for aug_idx, original_idx in enumerate(active):
                        keypoints_on_images[original_idx] = koi_to_aug[aug_idx]

        return keypoints_on_images

    def _to_deterministic(self):
        augs = [aug.to_deterministic() for aug in self]
        seq = self.copy()
        seq[:] = augs
        seq.random_state = ia.derive_random_state(self.random_state)
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
        return "SomeOf(name=%s, n=%s, random_order=%s, augmenters=[%s], deterministic=%s)" % (
            self.name, str(self.n), str(self.random_order), augs_str, self.deterministic)


def OneOf(children, name=None, deterministic=False, random_state=None):
    """
    Augmenter that always executes exactly one of its children.

    dtype support::

        See ``imgaug.augmenters.meta.SomeOf``.

    Parameters
    ----------
    children : list of imgaug.augmenters.meta.Augmenter
        The choices of augmenters to apply.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> imgs = [np.ones((10, 10))]
    >>> seq = iaa.OneOf([
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    flips each image either horizontally or vertically.


    >>> seq = iaa.OneOf([
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Sequential([
    >>>         iaa.GaussianBlur(1.0),
    >>>         iaa.Dropout(0.05),
    >>>         iaa.AdditiveGaussianNoise(0.1*255)
    >>>     ]),
    >>>     iaa.Noop()
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    either flips each image horizontally, or adds blur+dropout+noise or does
    nothing.

    """
    return SomeOf(n=1, children=children, random_order=False, name=name, deterministic=deterministic,
                  random_state=random_state)


class Sometimes(Augmenter):
    """
    Augment only ``p`` percent of all images with one or more augmenters.

    Let ``C`` be one or more child augmenters given to Sometimes.
    Let ``p`` be the percent of images to augment.
    Let ``I`` be the input images.
    Then (on average) ``p`` percent of all images in ``I`` will be augmented using ``C``.

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
        input images. E.g. a value of 0.5 will result in 50 percent of all
        input images being augmented.

    then_list : None or imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) to apply to `p` percent of all images.

    else_list : None or imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) to apply to ``(1-p)`` percent of all images.
        These augmenters will be applied only when the ones in then_list
        are NOT applied (either-or-relationship).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3))

    when calling ``aug.augment_images()``, only (on average) 50 percent of
    all images will be blurred.

    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3), iaa.Fliplr(1.0))

    when calling ``aug.augment_images()``, (on average) 50 percent of all images
    will be blurred, the other (again, on average) 50 percent will be
    horizontally flipped.

    """

    def __init__(self, p=0.5, then_list=None, else_list=None, name=None, deterministic=False, random_state=None):
        super(Sometimes, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.p = iap.handle_probability_param(p, "p")

        self.then_list = handle_children_list(then_list, self.name, "then", default=None)
        self.else_list = handle_children_list(else_list, self.name, "else", default=None)

    def _augment_images(self, images, random_state, parents, hooks):
        if hooks is None or hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
            input_is_np_array = ia.is_np_array(images)
            if input_is_np_array:
                input_dtype = images.dtype

            nb_images = len(images)
            samples = self.p.draw_samples((nb_images,), random_state=random_state)

            # create lists/arrays of images for if and else lists (one for each)
            # note that np.where returns tuple(array([0, 5, 9, ...])) or tuple(array([]))
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

            # map results of if/else lists back to their initial positions (in "images" variable)
            result = [None] * len(images)
            for idx_result_then_list, idx_images in enumerate(indices_then_list):
                result[idx_images] = result_then_list[idx_result_then_list]
            for idx_result_else_list, idx_images in enumerate(indices_else_list):
                result[idx_images] = result_else_list[idx_result_else_list]

            # If input was a list, keep the output as a list too,
            # otherwise it was a numpy array, so make the output a numpy array too.
            # Note here though that shapes can differ between images, e.g. when using Crop
            # without resizing. In these cases, the output has to be a list.
            all_same_shape = len(set([image.shape for image in result])) == 1
            if input_is_np_array and all_same_shape:
                result = np.array(result, dtype=input_dtype)
        else:
            result = images

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        if hooks is None or hooks.is_propagating(heatmaps, augmenter=self, parents=parents, default=True):
            nb_heatmaps = len(heatmaps)
            samples = self.p.draw_samples((nb_heatmaps,), random_state=random_state)

            # create lists of heatmaps for if and else lists (one for each)
            # note that np.where returns tuple(array([0, 5, 9, ...])) or tuple(array([]))
            indices_then_list = np.where(samples == 1)[0]
            indices_else_list = np.where(samples == 0)[0]
            heatmaps_then_list = [heatmaps[i] for i in indices_then_list]
            heatmaps_else_list = [heatmaps[i] for i in indices_else_list]

            # augment according to if and else list
            result_then_list = heatmaps_then_list
            result_else_list = heatmaps_else_list
            if self.then_list is not None and len(heatmaps_then_list) > 0:
                result_then_list = self.then_list.augment_heatmaps(
                    heatmaps_then_list,
                    parents=parents + [self],
                    hooks=hooks
                )
            if self.else_list is not None and len(heatmaps_else_list) > 0:
                result_else_list = self.else_list.augment_heatmaps(
                    heatmaps_else_list,
                    parents=parents + [self],
                    hooks=hooks
                )

            # map results of if/else lists back to their initial positions (in "heatmaps" variable)
            result = [None] * len(heatmaps)
            for idx_result_then_list, idx_heatmaps in enumerate(indices_then_list):
                result[idx_heatmaps] = result_then_list[idx_result_then_list]
            for idx_result_else_list, idx_heatmaps in enumerate(indices_else_list):
                result[idx_heatmaps] = result_else_list[idx_result_else_list]
        else:
            result = heatmaps

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        # TODO this is mostly copy pasted from _augment_images, make dry
        result = keypoints_on_images
        if hooks is None or hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents, default=True):
            nb_images = len(keypoints_on_images)
            samples = self.p.draw_samples((nb_images,), random_state=random_state)

            # create lists/arrays of images for if and else lists (one for each)
            # note that np.where returns tuple(array([0, 5, 9, ...])) or tuple(array([]))
            indices_then_list = np.where(samples == 1)[0]
            indices_else_list = np.where(samples == 0)[0]
            images_then_list = [keypoints_on_images[i] for i in indices_then_list]
            images_else_list = [keypoints_on_images[i] for i in indices_else_list]

            # augment according to if and else list
            result_then_list = images_then_list
            result_else_list = images_else_list
            if self.then_list is not None and len(images_then_list) > 0:
                result_then_list = self.then_list.augment_keypoints(
                    keypoints_on_images=images_then_list,
                    parents=parents + [self],
                    hooks=hooks
                )
            if self.else_list is not None and len(images_else_list) > 0:
                result_else_list = self.else_list.augment_keypoints(
                    keypoints_on_images=images_else_list,
                    parents=parents + [self],
                    hooks=hooks
                )

            # map results of if/else lists back to their initial positions (in "images" variable)
            result = [None] * len(keypoints_on_images)
            for idx_result_then_list, idx_images in enumerate(indices_then_list):
                result[idx_images] = result_then_list[idx_result_then_list]
            for idx_result_else_list, idx_images in enumerate(indices_else_list):
                result[idx_images] = result_else_list[idx_result_else_list]

        return result

    def _to_deterministic(self):
        aug = self.copy()
        aug.then_list = aug.then_list.to_deterministic() if aug.then_list is not None else aug.then_list
        aug.else_list = aug.else_list.to_deterministic() if aug.else_list is not None else aug.else_list
        aug.deterministic = True
        aug.random_state = ia.derive_random_state(self.random_state)
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
        return "Sometimes(p=%s, name=%s, then_list=%s, else_list=%s, deterministic=%s)" % (
            self.p, self.name, self.then_list, self.else_list, self.deterministic)


class WithChannels(Augmenter):
    """
    Apply child augmenters to specific channels.

    Let ``C`` be one or more child augmenters given to this augmenter.
    Let ``H`` be a list of channels.
    Let ``I`` be the input images.
    Then this augmenter will pick the channels ``H`` from each image
    in ``I`` (resulting in new images) and apply ``C`` to them.
    The result of the augmentation will be merged back into the original images.

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
        If None, all channels will be used. Note that this is not
        stochastic - the extracted channels are always the same
        ones.

    children : Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to images, after the channels
        are extracted.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.WithChannels([0], iaa.Add(10))

    assuming input images are RGB, then this augmenter will add 10 only
    to the first channel, i.e. make images more red.

    """

    def __init__(self, channels=None, children=None, name=None, deterministic=False, random_state=None):
        super(WithChannels, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        # TODO change this to a stochastic parameter
        if channels is None:
            self.channels = None
        elif ia.is_single_integer(channels):
            self.channels = [channels]
        elif ia.is_iterable(channels):
            ia.do_assert(all([ia.is_single_integer(channel) for channel in channels]),
                         "Expected integers as channels, got %s." % ([type(channel) for channel in channels],))
            self.channels = channels
        else:
            raise Exception("Expected None, int or list of ints as channels, got %s." % (type(channels),))

        self.children = handle_children_list(children, self.name, "then")

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        if hooks is None or hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
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
                    images_then_list = [image[..., self.channels] for image in images]

                result_then_list = self.children.augment_images(
                    images=images_then_list,
                    parents=parents + [self],
                    hooks=hooks
                )

                ia.do_assert(
                    all([img_out.shape[0:2] == shape_orig[0:2]
                         for img_out, shape_orig in zip(result_then_list, shapes_orig)]),
                    "Heights/widths of images changed in WithChannels from %s to %s, but expected to be the same." % (
                        str([shape_orig[0:2] for shape_orig in shapes_orig]),
                        str([img_out.shape[0:2] for img_out in result_then_list]),
                    )
                )

                if ia.is_np_array(images):
                    result[..., self.channels] = result_then_list
                else:
                    for i in sm.xrange(len(images)):
                        result[i][..., self.channels] = result_then_list[i]

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        result = heatmaps
        if hooks is None or hooks.is_propagating(heatmaps, augmenter=self, parents=parents, default=True):
            # Augment heatmaps in the style of the children if all channels or the majority of
            # them are selected by this layer, otherwise don't change the heatmaps.
            heatmaps_to_aug = []
            indices = []

            for i, heatmaps_i in enumerate(heatmaps):
                nb_channels = heatmaps_i.shape[2] if len(heatmaps_i.shape) >= 3 else 1
                if self.channels is None or len(self.channels) > nb_channels*0.5:
                    heatmaps_to_aug.append(heatmaps_i)
                    indices.append(i)

            if len(heatmaps_to_aug) > 0:
                heatmaps_aug = self.children.augment_heatmaps(
                    heatmaps_to_aug,
                    parents=parents + [self],
                    hooks=hooks
                )

                for idx_orig, heatmaps_i_aug in zip(indices, heatmaps_aug):
                    result[idx_orig] = heatmaps_i_aug

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = keypoints_on_images
        if hooks is None or hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents, default=True):
            # Augment keypoints in the style of the children if all channels or the majority of
            # them are selected by this layer, otherwise don't change the heatmaps.
            # We expect here the images channel number to be 3, but actually can't be fully sure
            # about that.
            if self.channels is None or len(self.channels) > 1:
                result = self.children.augment_keypoints(
                    keypoints_on_images,
                    parents=parents + [self],
                    hooks=hooks
                )

        return result

    def _to_deterministic(self):
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = ia.derive_random_state(self.random_state)
        return aug

    def get_parameters(self):
        return [self.channels]

    def get_children_lists(self):
        return [self.children]

    def __str__(self):
        return "WithChannels(channels=%s, name=%s, children=%s, deterministic=%s)" % (
            self.channels, self.name, self.children, self.deterministic)


class Noop(Augmenter):
    """
    Augmenter that never changes input images ("no operation").

    This augmenter is useful when you just want to use a placeholder augmenter
    in some situation, so that you can continue to call :func:`imgaug.augmenters.meta.Augmenter.augment_images`,
    without actually changing them (e.g. when switching from training to test).

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

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        super(Noop, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []


class Lambda(Augmenter):
    """
    Augmenter that calls a lambda function for each batch of input image.

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
        It must follow the form

            ``function(images, random_state, parents, hooks)``

        and return the changed images (may be transformed in-place).
        This is essentially the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_images`.
        If this is None instead of a function, the images will not be altered.

    func_heatmaps : None or callable, optional
        The function to call for each batch of heatmaps.
        It must follow the form

            ``function(heatmaps, random_state, parents, hooks)``

        and return the changed heatmaps (may be transformed in-place).
        This is essentially the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_heatmaps`.
        If this is None instead of a function, the heatmaps will not be altered.

    func_keypoints : None or callable, optional
        The function to call for each batch of image keypoints.
        It must follow the form

            ``function(keypoints_on_images, random_state, parents, hooks)``

        and return the changed keypoints (may be transformed in-place).
        This is essentially the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_keypoints`.
        If this is None instead of a function, the keypoints will not be altered.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> def func_images(images, random_state, parents, hooks):
    >>>     images[:, ::2, :, :] = 0
    >>>     return images
    >>>
    >>> aug = iaa.Lambda(
    >>>     func_images=func_images
    >>> )

    Replaces every second row in images with black pixels. Leaves heatmaps and keypoints unchanged.

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

    Replaces every second row in images with black pixels, sets every second row in heatmapps to
    zero and leaves keypoints unchanged.

    """

    def __init__(self, func_images=None, func_heatmaps=None, func_keypoints=None, name=None, deterministic=False,
                 random_state=None):
        super(Lambda, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.func_images = func_images
        self.func_heatmaps = func_heatmaps
        self.func_keypoints = func_keypoints

    def _augment_images(self, images, random_state, parents, hooks):
        if self.func_images is not None:
            return self.func_images(images, random_state, parents, hooks)
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        if self.func_heatmaps is not None:
            result = self.func_heatmaps(heatmaps, random_state, parents, hooks)
            ia.do_assert(ia.is_iterable(result),
                         "Expected callback function for heatmaps to return list of imgaug.HeatmapsOnImage() "
                         + "instances, got %s." % (type(result),))
            ia.do_assert(all([isinstance(el, ia.HeatmapsOnImage) for el in result]),
                         "Expected callback function for heatmaps to return list of imgaug.HeatmapsOnImage() "
                         + "instances, got %s." % ([type(el) for el in result],))
            return result
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        if self.func_keypoints is not None:
            result = self.func_keypoints(keypoints_on_images, random_state, parents, hooks)
            ia.do_assert(ia.is_iterable(result),
                         "Expected callback function for keypoints to return list of imgaug.KeypointsOnImage() "
                         + "instances, got %s." % (type(result),))
            ia.do_assert(all([isinstance(el, ia.KeypointsOnImage) for el in result]),
                         "Expected callback function for keypoints to return list of imgaug.KeypointsOnImage() "
                         + "instances, got %s." % ([type(el) for el in result],))
            return result
        return keypoints_on_images

    def get_parameters(self):
        return []


def AssertLambda(func_images=None, func_heatmaps=None, func_keypoints=None, name=None, deterministic=False,
                 random_state=None):
    """
    Augmenter that runs an assert on each batch of input images
    using a lambda function as condition.

    This is useful to make generic assumption about the input images and error
    out early if they aren't met.

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
        It must follow the form ``function(images, random_state, parents, hooks)``
        and return either True (valid input) or False (invalid input).
        It essentially reuses the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_images`.

    func_heatmaps : None or callable, optional
        The function to call for each batch of heatmaps.
        It must follow the form ``function(heatmaps, random_state, parents, hooks)``
        and return either True (valid input) or False (invalid input).
        It essentially reuses the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_heatmaps`.

    func_keypoints : None or callable, optional
        The function to call for each batch of keypoints.
        It must follow the form ``function(keypoints_on_images, random_state, parents, hooks)``
        and return either True (valid input) or False (invalid input).
        It essentially reuses the interface of
        :func:`imgaug.augmenters.meta.Augmenter._augment_keypoints`.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    """
    def func_images_assert(images, random_state, parents, hooks):
        ia.do_assert(func_images(images, random_state, parents, hooks),
                     "Input images did not fulfill user-defined assertion in AssertLambda.")
        return images

    def func_heatmaps_assert(heatmaps, random_state, parents, hooks):
        ia.do_assert(func_heatmaps(heatmaps, random_state, parents, hooks),
                     "Input heatmaps did not fulfill user-defined assertion in AssertLambda.")
        return heatmaps

    def func_keypoints_assert(keypoints_on_images, random_state, parents, hooks):
        ia.do_assert(func_keypoints(keypoints_on_images, random_state, parents, hooks),
                     "Input keypoints did not fulfill user-defined assertion in AssertLambda.")
        return keypoints_on_images

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)
    return Lambda(func_images_assert if func_images is not None else None,
                  func_heatmaps_assert if func_heatmaps is not None else None,
                  func_keypoints_assert if func_keypoints is not None else None,
                  name=name, deterministic=deterministic, random_state=random_state)


def AssertShape(shape, check_images=True, check_heatmaps=True, check_keypoints=True,
                name=None, deterministic=False, random_state=None):
    """
    Augmenter to make assumptions about the shape of input image(s), heatmaps and keypoints.

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
        The expected shape, given as a tuple. The number of entries in the tuple must match the
        number of dimensions, i.e. it must contain four entries for ``(N, H, W, C)``. If only a
        single image is augmented via ``augment_image()``, then ``N`` is viewed as 1 by this
        augmenter. If the input image(s) don't have a channel axis, then ``C`` is viewed as 1
        by this augmenter.
        Each of the four entries may be None or a tuple of two ints or a list of ints.

            * If an entry is None, any value for that dimensions is accepted.
            * If an entry is int, exactly that integer value will be accepted
              or no other value.
            * If an entry is a tuple of two ints with values ``a`` and ``b``, only a
              value ``x`` with ``a <= x < b`` will be accepted for the dimension.
            * If an entry is a list of ints, only a value for the dimension
              will be accepted which is contained in the list.

    check_images : bool, optional
        Whether to validate input images via the given shape.

    check_heatmaps : bool, optional
        Whether to validate input heatmaps via the given shape.
        The number of heatmaps will be checked and for each Heatmaps
        instance its array's height and width, but not the channel
        count as the channel number denotes the expected number of channels
        in images.

    check_keypoints : bool, optional
        Whether to validate input keypoints via the given shape.
        The number of keypoints will be checked and for each KeypointsOnImage
        instance its image's shape, i.e. KeypointsOnImage.shape.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> seq = iaa.Sequential([
    >>>     iaa.AssertShape((None, 32, 32, 3)),
    >>>     iaa.Fliplr(0.5)
    >>> ])

    will first check for each image batch, if it contains a variable number of
    ``32x32`` images with 3 channels each. Only if that check succeeds, the
    horizontal flip will be executed (otherwise an assertion error will be
    thrown).

    >>> seq = iaa.Sequential([
    >>>     iaa.AssertShape((None, (32, 64), 32, [1, 3])),
    >>>     iaa.Fliplr(0.5)
    >>> ])

    like above, but now the height may be in the range ``32 <= H < 64`` and
    the number of channels may be either 1 or 3.

    """
    ia.do_assert(len(shape) == 4, "Expected shape to have length 4, got %d with shape: %s." % (len(shape), str(shape)))

    def compare(observed, expected, dimension, image_index):
        if expected is not None:
            if ia.is_single_integer(expected):
                ia.do_assert(observed == expected,
                             "Expected dim %d (entry index: %s) to have value %d, got %d." % (
                                 dimension, image_index, expected, observed))
            elif isinstance(expected, tuple):
                ia.do_assert(len(expected) == 2)
                ia.do_assert(expected[0] <= observed < expected[1],
                             "Expected dim %d (entry index: %s) to have value in range [%d, %d), got %d." % (
                                 dimension, image_index, expected[0], expected[1], observed))
            elif isinstance(expected, list):
                ia.do_assert(any([observed == val for val in expected]),
                             "Expected dim %d (entry index: %s) to have any value of %s, got %d." % (
                                 dimension, image_index, str(expected), observed))
            else:
                raise Exception(("Invalid datatype for shape entry %d, expected each entry to be an integer, "
                                + "a tuple (with two entries) or a list, got %s.") % (dimension, type(expected),))

    def func_images(images, _random_state, _parents, _hooks):
        if check_images:
            if isinstance(images, list):
                if shape[0] is not None:
                    compare(len(images), shape[0], 0, "ALL")

                for i in sm.xrange(len(images)):
                    image = images[i]
                    ia.do_assert(len(image.shape) == 3,
                                 "Expected image number %d to have a shape of length 3, got %d (shape: %s)." % (
                                     i, len(image.shape), str(image.shape)))
                    for j in sm.xrange(len(shape)-1):
                        expected = shape[j+1]
                        observed = image.shape[j]
                        compare(observed, expected, j, i)
            else:
                ia.do_assert(len(images.shape) == 4,
                             "Expected image's shape to have length 4, got %d (shape: %s)." % (
                                 len(images.shape), str(images.shape)))
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

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return Lambda(func_images, func_heatmaps, func_keypoints,
                  name=name, deterministic=deterministic, random_state=random_state)


class ChannelShuffle(Augmenter):
    """
    Augmenter that randomly shuffles the channels in images.

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
        May be a fixed probability as a float, or a StochasticParameter that returns 0s and 1s.

    channels : None or imgaug.ALL or list of int, optional
        Which channels are allowed to be shuffled with each other.
        If this is ``None`` or ``imgaug.ALL``, then all channels may be shuffled. If it is a list of integers,
        then only the channels with indices in that list may be shuffled. (Values start at 0. All channel indices in
        the list must exist in each image.)

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> aug = iaa.ChannelShuffle(0.25)

    Shuffles channels for 25% of all images.

    >>> aug = iaa.ChannelShuffle(0.25, channels=[0, 1])

    Shuffles channels 0 and 1 with each other for 25% of all images.

    """

    def __init__(self, p=1.0, channels=None, name=None, deterministic=False, random_state=None):
        super(ChannelShuffle, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.p = iap.handle_probability_param(p, "p")
        ia.do_assert(channels is None
                     or channels == ia.ALL
                     or (isinstance(channels, list) and all([ia.is_single_integer(v) for v in channels])),
                     "Expected None or imgaug.ALL or list of int, got %s." % (type(channels),))
        self.channels = channels

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        p_samples = self.p.draw_samples((nb_images,), random_state=random_state)
        rss = ia.derive_random_states(random_state, nb_images)
        for i in sm.xrange(nb_images):
            if p_samples[i] >= 1-1e-4:
                images[i] = shuffle_channels(images[i], rss[i], self.channels)
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.p, self.channels]


def shuffle_channels(image, random_state, channels=None):
    """
    Randomize the order of (color) channels in an image.

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

    random_state : numpy.random.RandomState
        The random state to use for this shuffling operation.

    channels : None or imgaug.ALL or list of int, optional
        Which channels are allowed to be shuffled with each other.
        If this is ``None`` or ``imgaug.ALL``, then all channels may be shuffled. If it is a list of integers,
        then only the channels with indices in that list may be shuffled. (Values start at 0. All channel indices in
        the list must exist in each image.)

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
        # note that if this is the case, then 'channels' may be None or imgaug.ALL, so don't simply move the
        # assignment outside of the if/else
        channels_perm = random_state.permutation(all_channels)
        return image[..., channels_perm]
    else:
        channels_perm = random_state.permutation(channels)
        channels_perm_full = all_channels
        for channel_source, channel_target in zip(channels, channels_perm):
            channels_perm_full[channel_source] = channel_target
        return image[..., channels_perm_full]
