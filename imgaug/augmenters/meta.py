"""
Augmenters that don't apply augmentations themselves, but are needed
for meta usage.

Do not import directly from this file, as the categorization is not final.
Use instead
    `from imgaug import augmenters as iaa`
and then e.g.
    `seq = iaa.Sequential([...])`

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

Note that WithColorspace is in `color.py`.
"""
from __future__ import print_function, division, absolute_import
from .. import imgaug as ia
# TODO replace these imports with iap.XYZ
from ..parameters import StochasticParameter, Binomial, DiscreteUniform
from abc import ABCMeta, abstractmethod
import numpy as np
import copy as copy_module
import re
from scipy import misc
import itertools
import six
import six.moves as sm
import warnings

def copy_dtypes_for_restore(images, force_list=False):
    if ia.is_np_array(images):
        if force_list:
            return [images.dtype for _ in sm.xrange(len(images))]
        else:
            return images.dtype
    else:
        return [image.dtype for image in images]

def restore_augmented_image_dtype_(image, orig_dtype):
    return image.astype(orig_dtype, copy=False)

def restore_augmented_images_dtypes_(images, orig_dtypes):
    if ia.is_np_array(images):
        return images.astype(orig_dtypes, copy=False)
    else:
        return [image.astype(orig_dtype, copy=False) for image, orig_dtype in zip(images, orig_dtypes)]

def restore_augmented_images_dtypes(images, orig_dtypes):
    if ia.is_np_array(images):
        images = np.copy(images)
    else:
        images = [np.copy(image) for image in images]
    return restore_augmented_images_dtypes_(images, orig_dtypes)

def clip_augmented_image_(image, minval, maxval):
    return clip_augmented_images_(image, minval, maxval)

def clip_augmented_images_(images, minval, maxval):
    if ia.is_np_array(images):
        return np.clip(images, minval, maxval, out=images)
    else:
        return [np.clip(image, minval, maxval, out=image) for image in images]

def clip_augmented_images(images, minval, maxval):
    if ia.is_np_array(images):
        images = np.copy(images)
    else:
        images = [np.copy(image) for image in images]
    return clip_augmented_images_(images, minval, maxval)

@six.add_metaclass(ABCMeta)
class Augmenter(object): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Base class for Augmenter objects.
    All augmenters derive from this class.
    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        """
        Create a new Augmenter instance.

        Parameters
        ----------
        name : None or string, optional(default=None)
            Name given to an Augmenter object. This name is used in print()
            statements as well as find and remove functions.
            If None, `UnnamedX` will be used as the name, where X is the
            Augmenter's class name.

        deterministic : bool, optional(default=False)
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
            `augmenter.to_deterministic()`.

        random_state : None or int or np.random.RandomState, optional(default=None)
            The random state to use for this
            augmenter.
                * If int, a new np.random.RandomState will be created using this
                  value as the seed.
                * If np.random.RandomState instance, the instance will be used directly.
                * If None, imgaug's default RandomState will be used, which's state can
                  be controlled using imgaug.seed(int).
            Usually there is no need to set this variable by hand. Instead,
            instantiate the augmenter with the defaults and then use
            `augmenter.to_deterministic()`.

        """
        super(Augmenter, self).__init__()

        if name is None:
            self.name = "Unnamed%s" % (self.__class__.__name__,)
        else:
            self.name = name

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

        Parameters
        ----------
        batches : list
            List of image batches to augment.
            The expected input is a list, with each entry having one of the
            following datatypes:
                * ia.Batch
                * []
                * list of ia.KeypointsOnImage
                * list of (H,W,C) ndarray
                * list of (H,W) ndarray
                * (N,H,W,C) ndarray
                * (N,H,W) ndarray
            where N = number of images, H = height, W = width,
            C = number of channels.
            Each image is recommended to have dtype uint8 (range 0-255).

        hooks : None or ia.HooksImages, optional(default=None)
            HooksImages object to dynamically interfere with the augmentation
            process.

        background : bool, optional(default=False)
            Whether to augment the batches in background processes.
            If true, hooks can currently not be used as that would require
            pickling functions.

        Yields
        -------
        augmented_batch : ia.Batch or list of ia.KeypointsOnImage or list of (H,W,C) ndarray or list of (H,W) ndarray or (N,H,W,C) ndarray or (N,H,W) ndarray
            Augmented images/keypoints.
            Datatype usually matches the input datatypes per list element.

        """
        ia.do_assert(isinstance(batches, list))
        ia.do_assert(len(batches) > 0)
        if background:
            ia.do_assert(hooks is None, "Hooks can not be used when background augmentation is activated.")

        batches_normalized = []
        batches_original_dts = []
        for i, batch in enumerate(batches):
            if isinstance(batch, ia.Batch):
                batch.data = (i, batch.data)
                batches_normalized.append(batch)
                batches_original_dts.append("imgaug.Batch")
            elif ia.is_np_array(batch):
                ia.do_assert(batch.ndim in (3, 4), "Expected numpy array to have shape (N, H, W) or (N, H, W, C), got %s." % (batch.shape,))
                batches_normalized.append(ia.Batch(images=batch, data=i))
                batches_original_dts.append("numpy_array")
            elif isinstance(batch, list):
                if len(batch) == 0:
                    batches_normalized.append(ia.Batch())
                    batches_original_dts.append("empty_list")
                elif ia.is_np_array(batch[0]):
                    batches_normalized.append(ia.Batch(images=batch, data=i))
                    batches_original_dts.append("list_of_numpy_arrays")
                elif isinstance(batch[0], ia.KeypointsOnImage):
                    batches_normalized.append(ia.Batch(keypoints=batch, data=i))
                    batches_original_dts.append("list_of_imgaug.KeypointsOnImage")
                else:
                    raise Exception("Unknown datatype in batch[0]. Expected numpy array or imgaug.KeypointsOnImage, got %s." % (type(batch[0]),))
            else:
                raise Exception("Unknown datatype in of batch. Expected imgaug.Batch or numpy array or list of numpy arrays/imgaug.KeypointsOnImage. Got %s." % (type(batch),))

        def unnormalize_batch(batch_aug):
            #if batch_aug.data is None:
            #    return batch_aug
            #else:
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
            elif dt_orig == "list_of_imgaug.KeypointsOnImage":
                batch_unnormalized = batch_aug.keypoints_aug
            else:
                raise Exception("Internal error. Unexpected value in dt_orig '%s'. This should never happen." % (dt_orig,))
            return batch_unnormalized

        if not background:
            for batch_normalized in batches_normalized:
                batch_augment_images = batch_normalized.images is not None
                batch_augment_keypoints = batch_normalized.keypoints is not None

                if batch_augment_images and batch_augment_keypoints:
                    augseq_det = self.to_deterministic() if not self.deterministic else self
                    batch_normalized.images_aug = augseq_det.augment_images(batch_normalized.images, hooks=hooks)
                    batch_normalized.keypoints_aug = augseq_det.augment_keypoints(batch_normalized.keypoints, hooks=hooks)
                elif batch_augment_images:
                    batch_normalized.images_aug = self.augment_images(batch_normalized.images, hooks=hooks)
                elif batch_augment_keypoints:
                    batch_normalized.keypoints_aug = self.augment_keypoints(batch_normalized.keypoints, hooks=hooks)
                batch_unnormalized = unnormalize_batch(batch_normalized)
                yield batch_unnormalized
        else:
            def load_batches():
                for batch in batches_normalized:
                    yield batch

            batch_loader = ia.BatchLoader(load_batches)
            bg_augmenter = ia.BackgroundAugmenter(batch_loader, self)
            while True:
                batch_aug = bg_augmenter.get_batch()
                if batch_aug is None:
                    break
                else:
                    batch_unnormalized = unnormalize_batch(batch_aug)
                    yield batch_unnormalized
            batch_loader.terminate()
            bg_augmenter.terminate()

    def augment_image(self, image, hooks=None):
        """
        Augment a single image.

        Parameters
        ----------
        image : (H,W,C) ndarray or (H,W) ndarray
            The image to augment. Should have dtype uint8 (range 0-255).

        hooks : None or ia.HooksImages, optional(default=None)
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        img : ndarray
            The corresponding augmented image.

        """
        ia.do_assert(image.ndim in [2, 3], "Expected image to have shape (height, width, [channels]), got shape %s." % (image.shape,))
        return self.augment_images([image], hooks=hooks)[0]

    def augment_images(self, images, parents=None, hooks=None):
        """
        Augment multiple images.

        Parameters
        ----------
        images : (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            Images to augment. The input can be a list of numpy arrays or
            a single array. Each array is expected to have shape (H, W, C)
            or (H, W), where H is the height, W is the width and C are the
            channels. Number of channels may differ between images.
            If a list is chosen, height and width may differ per between images.
            Currently the recommended dtype is uint8 (i.e. integer values in
            the range 0 to 255). Other dtypes are not tested.

        parents : None or list of Augmenter, optional(default=None)
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : None or ia.HooksImages, optional(default=None)
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        images_result : ndarray or list
            Corresponding augmented images.

        """
        if self.deterministic:
            state_orig = self.random_state.get_state()

        if parents is None:
            parents = []

        if hooks is None:
            hooks = ia.HooksImages()

        if ia.is_np_array(images):
            input_type = "array"
            input_added_axis = False

            ia.do_assert(images.ndim in [3, 4], "Expected 3d/4d array of form (N, height, width) or (N, height, width, channels), got shape %s." % (images.shape,))

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
                ia.do_assert(all(image.ndim in [2, 3] for image in images), "Expected list of images with each image having shape (height, width) or (height, width, channels), got shapes %s." % ([image.shape for image in images],))

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
            raise Exception("Expected images as one numpy array or list/tuple of numpy arrays, got %s." % (type(images),))

        images_copy = hooks.preprocess(images_copy, augmenter=self, parents=parents)

        #if ia.is_np_array(images) != ia.is_np_array(images_copy):
        #    print("[WARNING] images vs images_copy", ia.is_np_array(images), ia.is_np_array(images_copy))
        #if ia.is_np_array(images):
            #ia.do_assert(images.shape[0] > 0, images.shape)
        #    print("images.shape", images.shape)
        #if ia.is_np_array(images_copy):
        #    print("images_copy.shape", images_copy.shape)

        # the is_activated() call allows to use hooks that selectively
        # deactivate specific augmenters in previously defined augmentation
        # sequences
        if hooks.is_activated(images_copy, augmenter=self, parents=parents, default=self.activated):
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

        images_result = hooks.postprocess(images_result, augmenter=self, parents=parents)

        # remove temporarily added channel axis for 2D input images
        if input_type == "array":
            if input_added_axis == True:
                images_result = np.squeeze(images_result, axis=3)
        if input_type == "list":
            for i in sm.xrange(len(images_result)):
                if input_added_axis[i] == True:
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
        Augmenter instance's `random_state` variable. The parameter
        `random_state` takes care of both of these.

        Parameters
        ----------
        images : (N,H,W,C) ndarray or list of (H,W,C) ndarray
            Images to augment.
            They may be changed in-place.
            Either a list of (H, W, C) arrays or a single (N, H, W, C) array,
            where N = number of images, H = height of images, W = width of
            images, C = number of channels of images.
            In the case of a list as input, H, W and C may change per image.

        random_state : np.random.RandomState
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of Augmenter
            See augment_images().

        hooks : ia.HooksImages
            See augment_images().

        Returns
        ----------
        images : (N,H,W,C) ndarray or list of (H,W,C) ndarray
            The augmented images.

        """
        raise NotImplementedError()

    def augment_keypoints(self, keypoints_on_images, parents=None, hooks=None):
        """
        Augment image keypoints.

        This is the corresponding function to `augment_images()`, just for
        keypoints/landmarks (i.e. coordinates on the image).
        Usually you will want to call `augment_images()` with a list of images,
        e.g. `augment_images([A, B, C])` and then `augment_keypoints()` with the
        corresponding list of keypoints on these images, e.g.
        `augment_keypoints([Ak, Bk, Ck])`, where `Ak` are the keypoints on
        image `A`.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding keypoints,
        e.g. by
            >>> seq = iaa.Fliplr(0.5)
            >>> seq_det = seq.to_deterministic()
            >>> imgs_aug = seq_det.augment_images([A, B, C])
            >>> kps_aug = seq_det.augment_keypoints([Ak, Bk, Ck])
        Otherwise, different random values will be sampled for the image
        and keypoint augmentations, resulting in different augmentations (e.g.
        images might be rotated by `30deg` and keypoints by `-10deg`).
        Also make sure to call `to_deterministic()` again for each new batch,
        otherwise you would augment all batches in the same way.


        Parameters
        ----------
        keypoints_on_images : list of ia.KeypointsOnImage
            The keypoints/landmarks to augment.
            Expected is a list of ia.KeypointsOnImage objects,
            each containing the keypoints of a single image.

        parents : None or list of Augmenter, optional(default=None)
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : None or ia.HooksKeypoints, optional(default=None)
            HooksKeypoints object to dynamically interfere with the
            augmentation process.

        Returns
        -------
        keypoints_on_images_result : list of ia.KeypointsOnImage
            Augmented keypoints.

        """
        if self.deterministic:
            state_orig = self.random_state.get_state()

        if parents is None:
            parents = []

        if hooks is None:
            hooks = ia.HooksKeypoints()

        ia.do_assert(ia.is_iterable(keypoints_on_images))
        ia.do_assert(all([isinstance(keypoints_on_image, ia.KeypointsOnImage) for keypoints_on_image in keypoints_on_images]))

        keypoints_on_images_copy = [keypoints_on_image.deepcopy() for keypoints_on_image in keypoints_on_images]

        keypoints_on_images_copy = hooks.preprocess(keypoints_on_images_copy, augmenter=self, parents=parents)

        if hooks.is_activated(keypoints_on_images_copy, augmenter=self, parents=parents, default=self.activated):
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

        keypoints_on_images_result = hooks.postprocess(keypoints_on_images_result, augmenter=self, parents=parents)

        if self.deterministic:
            self.random_state.set_state(state_orig)

        return keypoints_on_images_result

    @abstractmethod
    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        """
        Augment keypoints on multiple images.

        This is the internal variation of `augment_keypoints()`.
        It is called from `augment_keypoints()` and should usually not be called
        directly.
        It has to be implemented by every augmenter.
        This method may transform the keypoints in-place.
        This method does not have to care about determinism or the
        Augmenter instance's `random_state` variable. The parameter
        `random_state` takes care of both of these.

        Parameters
        ----------
        keypoints_on_images : list of ia.KeypointsOnImage
            Keypoints to augment. They may be changed in-place.

        random_state : np.random.RandomState
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of Augmenter
            See `augment_keypoints()`.

        hooks : ia.HooksImages
            See `augment_keypoints()`.

        Returns
        ----------
        images : list of ia.KeypointsOnImage
            The augmented keypoints.

        """
        raise NotImplementedError()

    def augment_bounding_boxes(self, bounding_boxes_on_images, hooks=None):
        """
        Augment image bounding boxes.

        This is the corresponding function to `augment_keypoints()`, just for
        bounding boxes.
        Usually you will want to call `augment_images()` with a list of images,
        e.g. `augment_images([A, B, C])` and then `augment_bounding_boxes()`
        with the corresponding list of bounding boxes on these images, e.g.
        `augment_bounding_boxes([Abb, Bbb, Cbb])`, where `Abb` are the
        bounding boxes on image `A`.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding bounding boxes,
        e.g. by
            >>> seq = iaa.Fliplr(0.5)
            >>> seq_det = seq.to_deterministic()
            >>> imgs_aug = seq_det.augment_images([A, B, C])
            >>> bbs_aug = seq_det.augment_keypoints([Abb, Bbb, Cbb])
        Otherwise, different random values will be sampled for the image
        and bounding box augmentations, resulting in different augmentations
        (e.g. images might be rotated by `30deg` and bounding boxes by
        `-10deg`). Also make sure to call `to_deterministic()` again for each
        new batch, otherwise you would augment all batches in the same way.

        Parameters
        ----------
        bounding_boxes_on_images : list of ia.BoundingBoxesOnImage
            The bounding boxes to augment.
            Expected is a list of ia.BoundingBoxesOnImage objects,
            each containing the bounding boxes of a single image.

        hooks : None or ia.HooksKeypoints, optional(default=None)
            HooksKeypoints object to dynamically interfere with the
            augmentation process.

        Returns
        -------
        result : list of ia.BoundingBoxesOnImage
            Augmented bounding boxes.

        """
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
        return result

    # TODO most of the code of this function could be replaced with ia.draw_grid()
    # TODO add parameter for handling multiple images ((a) next to each other
    # in each row or (b) multiply row count by number of images and put each
    # one in a new row)
    # TODO "images" parameter deviates from augment_images (3d array is here
    # treated as one 3d image, in augment_images as (N, H, W))
    def draw_grid(self, images, rows, cols):
        """
        Apply this augmenter to the given images and return a grid
        image of the results.
        Each cell in the grid contains a single augmented variation of
        an input image.

        If multiple images are provided, the row count is multiplied by
        the number of images and each image gets its own row.
        E.g. for `images = [A, B]`, `rows=2`, `cols=3`::
            A A A
            B B B
            A A A
            B B B

        for `images = [A]`, `rows=2`,
        `cols=3`::
            A A A
            A A A

        Parameters
        -------
        images : (N,H,W,3) ndarray or (H,W,3) ndarray or (H,W) ndarray or list of (H,W,3) ndarray or list of (H,W) ndarray
            List of images of which to show the augmented versions.
            If a list, then each element is expected to have shape (H, W) or
            (H, W, 3). If a single array, then it is expected to have
            shape (N, H, W, 3) or (H, W, 3) or (H, W).

        rows : int
            Number of rows in the grid.
            If N input images are given, this value will automatically be
            multiplied by N to create rows for each image.

        cols : int
            Number of columns in the grid.

        Returns
        -------
        grid : (Hg,Wg,3) ndarray
            The generated grid image with augmented versions of the input
            images. Here, Hg and Wg reference the output size of the grid,
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
                raise Exception("Unexpected images shape, expected 2-, 3- or 4-dimensional array, got shape %s." % (images.shape,))
        elif isinstance(images, list):
            for i, image in enumerate(images):
                if len(image.shape) == 3:
                    continue
                elif len(image.shape) == 2:
                    images[i] = image[:, :, np.newaxis]
                else:
                    raise Exception("Unexpected image shape at index %d, expected 2- or 3-dimensional array, got shape %s." % (i, image.shape,))
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
        grid = np.zeros((height, width, 3))
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
        Apply this augmenter to the given images and show/plot the results as
        a grid of images.

        If multiple images are provided, the row count is multiplied by
        the number of images and each image gets its own row.
        E.g. for `images = [A, B]`, `rows=2`, `cols=3`::
            A A A
            B B B
            A A A
            B B B

        for `images = [A]`, `rows=2`,
        `cols=3`::
            A A A
            A A A

        Parameters
        ----------
        images : (N,H,W,3) ndarray or (H,W,3) ndarray or (H,W) ndarray or list of (H,W,3) ndarray or list of (H,W) ndarray
            List of images of which to show the augmented versions.
            If a list, then each element is expected to have shape (H, W)
            or (H, W, 3).
            If a single array, then it is expected to have shape (N, H, W, 3)
            or (H, W, 3) or (H, W).

        rows : int
            Number of rows in the grid.
            If N input images are given, this value will automatically be
            multiplied by N to create rows for each image.

        cols : int
            Number of columns in the grid.

        """
        grid = self.draw_grid(images, rows, cols)
        misc.imshow(grid)

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
        keypoints in the same way (e.g. if an image is rotated by `30deg`, then
        also rotate its keypoints by `30deg`).

        Parameters
        ----------
        n : None or int, optional
            Number of deterministic augmenters to return.
            If None then only one Augmenter object will be returned.
            If 1 or higher, then a list containing n Augmenter objects
            will be returned.

        Returns
        -------
        det : Augmenter or list of Augmenter
            A single Augmenter object if n was None,
            otherwise a list of Augmenter objects (even if n was 1).

        """
        ia.do_assert(n is None or n >= 1)
        if n is None:
            return self.to_deterministic(1)[0]
        else:
            return [self._to_deterministic() for _ in sm.xrange(n)]

    def _to_deterministic(self):
        """
        Augmenter-specific implementation of `to_deterministic()`.
        This function is expected to return a single new deterministic
        Augmenter object of this augmenter.

        Returns
        -------
        det : Augmenter object
            Deterministic variation of this Augmenter object.

        """
        aug = self.copy()
        aug.random_state = ia.new_random_state()
        aug.deterministic = True
        return aug

    def reseed(self, random_state=None, deterministic_too=False):
        """
        Reseed this augmenter and all of its children (if it has any).

        This function is useful, when augmentations are run in the
        background (i.e. on multiple cores).
        It should be called before sending this Augmenter object to a
        background worker (i.e., if N workers are used, the function
        should be called N times). Otherwise, all background workers will
        use the same seeds and therefore apply the same augmentations.

        Parameters
        ----------
        random_state : None or int or np.random.RandomState, optional
            A RandomState that is used to sample seeds per augmenter.
            If int, the parameter will be used as a seed for a new RandomState.
            If None, a new RandomState will automatically be created.

        deterministic_too : bool, optional
            Whether to also change the seed of an augmenter `A`, if `A`
            is deterministic. This is the case both when this augmenter
            object is `A` or one of its children is `A`.

        """
        ia.do_assert(isinstance(deterministic_too, bool))

        if random_state is None:
            random_state = ia.current_random_state()
        elif isinstance(random_state, np.random.RandomState):
            pass # just use the provided random state without change
        else:
            random_state = ia.new_random_state(random_state)

        if not self.deterministic or deterministic_too:
            seed = random_state.randint(0, 10**6, 1)[0]
            self.random_state = ia.new_random_state(seed)

        for lst in self.get_children_lists():
            for aug in lst:
                aug.reseed(random_state=random_state, deterministic_too=deterministic_too)

    def localize_random_state(self, recursive=True):
        """
        Converts global random states to local ones.
        See `Augmenter.localize_random_state_()` for more details.

        Parameters
        ----------
        recursive : bool, optional(default=True)
            See `Augmenter.localize_random_state_()`.

        Returns
        -------
        aug : Augmenter
            Returns copy of augmenter and children, with localized random
            states.

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
        Local random usually exists for exactly one augmenter and are
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
        The case of determinism is handled automatically by to_deterministic().
        Only when you copy random states (via copy_random_state()), you need
        to call this function first.

        Parameters
        ----------
        recursive : bool, optional(default=True)
            Whether to localize the random states of children
            too.

        Returns
        -------
        self : Augmenter
            Returns itself (with localized random states).

        """
        if self.random_state == ia.current_random_state():
            self.random_state = ia.new_random_state()
        if recursive:
            for lst in self.get_children_lists():
                for child in lst:
                    child.localize_random_state_(recursive=recursive)
        return self

    def copy_random_state(self, source, recursive=True, matching="position", matching_tolerant=True, copy_determinism=False):
        """
        Copy the random states from a source augmenter sequence.

        Parameters
        ----------
        source : Augmenter
            See `Augmenter.copy_random_state_()`.

        recursive : bool, optional(default=True)
            See `Augmenter.copy_random_state_()`.

        matching : {'position', 'name'}, optional(default='position')
            See `Augmenter.copy_random_state_()`.

        matching_tolerant : bool, optional(default=True)
            See `Augmenter.copy_random_state_()`.

        copy_determinism : bool, optional(default=False)
            See `Augmenter.copy_random_state_()`.

        Returns
        -------
        aug : Augmenter
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

    def copy_random_state_(self, source, recursive=True, matching="position", matching_tolerant=True, copy_determinism=False):
        """
        Copy the random states from a source augmenter sequence (inplace).

        Parameters
        ----------
        source : Augmenter
            The source augmenter from where to copy the random_state(s).
            May have children (e.g. a Sequential).
            May not use the global random state. This is used by default
            by all augmenters. Call `localize_random_state_()` once on the
            source to localize all random states.

        recursive : bool, optional(default=True)
            Whether to copy the random states of the source augmenter *and*
            all of its children (True) or just the source augmenter (False).

        matching : {'position', 'name'}, optional(default='position')
            Defines the matching mode to use during recursive copy.
            This is used to associate source augmenters with target augmenters.
            If 'position' then the target and source sequences of augmenters
            are turned into flattened lists and are associated based on
            their list indices. If 'name' then the target and source augmenters
            are matched based on their names (i.e. `augmenter.name`).

        matching_tolerant : bool, optional(default=True)
            Whether to use tolerant matching between source and target
            augmenters. If set to False: Name matching will raise an exception
            for any target augmenter which's name does not appear among the
            source augmeters. Position matching will raise an exception if
            source and target augmenter have an unequal number of children.

        copy_determinism : bool, optional(default=False)
            Whether to copy the `deterministic` flags from source to target
            augmenters too.

        Returns
        -------
        self : Augmenter
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
        return []

    def get_all_children(self, flat=False):
        """
        Returns all children of this augmenter as a list.

        If the augmenter has no children, the returned list is empty.

        Parameters
        ----------
        flat : bool
            If set to True, the returned list will be
            flat.

        Returns
        -------
        result : list of Augmenter
            The children as a nested or flat list.

        """
        result = []
        for lst in self.get_children_lists():
            for aug in lst:
                result.append(aug)
                children = aug.get_all_children(flat=flat)
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

        parents : None or list of Augmenter, optional
            List of parent augmenters.
            Intended for nested calls and can usually be left as None.

        flat : bool, optional
            Whether to return the result as a flat list (True)
            or a nested list (False). In the latter case, the nesting matches
            each augmenters position among the children.

        Returns
        ----------
        augmenters : list of Augmenter
            Nested list if flat was set to False.
            Flat list if flat was set to True.

        Examples
        --------
        >>> aug = iaa.Sequential([
        >>>     nn.Fliplr(0.5, name="fliplr"),
        >>>     nn.Flipud(0.5, name="flipud")
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
        name : string
            Name of the augmenter(s) to search for.

        regex : bool, optional
            Whether `name` parameter is a regular expression.

        flat : bool, optional
            See `Augmenter.find_augmenters()`.

        Returns
        -------
        augmenters : list of Augmenter objects
            Nested list if flat was set to False.
            Flat list if flat was set to True.

        """
        return self.find_augmenters_by_names([name], regex=regex, flat=flat)

    def find_augmenters_by_names(self, names, regex=False, flat=True):
        """
        Find augmenter(s) by names.

        Parameters
        ----------
        names : list of string
            Names of the augmenter(s) to search for.

        regex : bool, optional
            Whether `names` is a list of regular expressions.

        flat : boolean, optional
            See `Augmenter.find_augmenters()`.

        Returns
        -------
        augmenters : list of Augmenter
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
            E.g. `lambda a, parents: a.name == "fliplr" and len(parents) == 1`
            removes an augmenter with name "fliplr" if it is the direct child
            of the augmenter upon which `remove_augmenters()` was initially called.

        copy : bool, optional
            Whether to copy this augmenter and all if its children before
            removing. If False, removal is performed in-place.

        noop_if_topmost : bool, optional
            If True and the condition (lambda function) leads to the removal
            of the topmost augmenter (the one this function is called on
            initially), then that topmost augmenter will be replaced by a
            Noop instance (i.e. an object will still knows `augment_images()`,
            but doesnt change images). If False, None will be returned in
            these cases.
            This can only be False if copy is set to True.

        Returns
        -------
        aug : Augmenter or None
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

        This is functionally identical to `remove_augmenters()` with
        `copy=False`, except that it does not affect the topmost augmenter
        (the one on which this function is initially called on).

        Parameters
        ----------
        func : callable
            See `Augmenter.remove_augmenters()`.

        parents : None or list of Augmenter, optional
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
                # self._remove_augmenters_inplace_from_list(lst, aug, i, i - count_removed)
                del lst[i - count_removed]

            for aug in lst:
                aug.remove_augmenters_inplace(func, subparents)

    # TODO
    # def to_json(self):
    #    pass

    def copy(self):
        """
        Create a shallow copy of this Augmenter instance.

        Returns
        -------
        aug : Augmenter
            Shallow copy of this Augmenter instance.

        """
        return copy_module.copy(self)

    def deepcopy(self):
        """
        Create a deep copy of this Augmenter instance.

        Returns
        -------
        aug : Augmenter
            Deep copy of this Augmenter instance.

        """
        return copy_module.deepcopy(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        params = self.get_parameters()
        params_str = ", ".join([param.__str__() for param in params])
        return "%s(name=%s, parameters=[%s], deterministic=%s)" % (self.__class__.__name__, self.name, params_str, self.deterministic)


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

    Parameters
    ----------
    children : Augmenter or list of Augmenter or None, optional(default=None)
        The augmenters to apply to images.

    random_order : bool, optional(default=False)
        Whether to apply the child augmenters in random order per image.
        The order is resampled for each image.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
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
            list.__init__(self, children)
        else:
            raise Exception("Expected None or Augmenter or list of Augmenter, got %s." % (type(children),))
        self.random_order = random_order

    def _augment_images(self, images, random_state, parents, hooks):
        if hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
            if self.random_order:
                # for augmenter in self.children:
                for index in random_state.permutation(len(self)):
                    images = self[index].augment_images(
                        images=images,
                        parents=parents + [self],
                        hooks=hooks
                    )
            else:
                # for augmenter in self.children:
                for augmenter in self:
                    images = augmenter.augment_images(
                        images=images,
                        parents=parents + [self],
                        hooks=hooks
                    )
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        if hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents, default=True):
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
        seq.random_state = ia.new_random_state()
        seq.deterministic = True
        return seq

    def get_parameters(self):
        return []

    def add(self, augmenter):
        """Add an augmenter to the list of child augmenters.

        Parameters
        ----------
        augmenter : Augmenter
            The augmenter to add.
        """
        self.append(augmenter)

    def get_children_lists(self):
        return [self]

    def __str__(self):
        augs_str = ", ".join([aug.__str__() for aug in self])
        return "Sequential(name=%s, augmenters=[%s], deterministic=%s)" % (self.name, augs_str, self.deterministic)

class SomeOf(Augmenter, list):
    """
    List augmenter that applies only some of its children to images.

    E.g. this allows to define a list of 20 augmenters, but only apply a
    random selection of 5 of them to each image.

    This augmenter currently does not support replacing (i.e. picking the same
    child multiple times) due to implementation difficulties in connection
    with deterministic augmenters.

    Parameters
    ----------
    n : int or tuple of two ints or list of ints or StochasticParameter or None, optional(default=None)
        Count of augmenters to
        apply.
            * If int n, then exactly n of the child augmenters are applied to
              every image.
            * If tuple of two ints (a, b), then a <= x <= b augmenters are
              picked and applied to every image. Here, b may be set to None,
              then it will automatically replaced with the total number of
              available children.
            * If StochasticParameter, then N numbers will be sampled for N images.
              The parameter is expected to be discrete.
            * If None, then the total number of available children will be
              used (i.e. all children will be applied).

    children : Augmenter or list of Augmenter or None, optional(default=None)
        The augmenters to apply to images.

    random_order : boolean, optional(default=False)
        Whether to apply the child augmenters in random order per image.
        The order is resampled for each image.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
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
                self.n = DiscreteUniform(int(n[0]), int(n[1]))
                self.n_mode = "stochastic"
            else:
                raise Exception("Expected tuple of (int, None) or (int, int), got %s" % ([type(el) for el in n],))
        elif isinstance(n, StochasticParameter):
            self.n = n
            self.n_mode = "stochastic"
        else:
            raise Exception("Expected int, (int, None), (int, int) or StochasticParameter, got %s" % (type(n),))

        self.random_order = random_order

    def _get_n(self, nb_images, random_state):
        if self.n_mode == "deterministic":
            return [self.n] * nb_images
        elif self.n_mode == "None":
            return [len(self)] * nb_images
        elif self.n_mode == "(int,None)":
            param = DiscreteUniform(self.n[0], len(self))
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
        #if not self.replace:
        #    nn = [min(n, len(self)) for n in nn]
        #augmenter_indices = [
        #    random_state.choice(len(self.children), size=(min(n, len(self)),), replace=False]) for n in nn
        #]
        nn = [min(n, len(self)) for n in nn]
        augmenter_active = np.zeros((nb_rows, len(self)), dtype=np.bool)
        for row_idx, n_true in enumerate(nn):
            if n_true > 0:
                augmenter_active[row_idx, 0:n_true] = 1
        for row in augmenter_active:
            random_state.shuffle(row)
        return augmenter_active

    def _augment_images(self, images, random_state, parents, hooks):
        if hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
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

                    # map them back to their position in the images array/list
                    if input_is_array:
                        images[active] = images_to_aug
                    else:
                        for aug_idx, original_idx in enumerate(active):
                            images[original_idx] = images_to_aug[aug_idx]

        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        if hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents, default=True):
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
        seq.random_state = ia.new_random_state()
        seq.deterministic = True
        return seq

    def get_parameters(self):
        return [self.n]

    def add(self, augmenter):
        """Add an augmenter to the list of child augmenters.

        Parameters
        ----------
        augmenter : Augmenter
            The augmenter to add.

        """
        self.append(augmenter)

    def get_children_lists(self):
        return [self]

    def __str__(self):
        # augs_str = ", ".join([aug.__str__() for aug in self.children])
        augs_str = ", ".join([aug.__str__() for aug in self])
        return "SomeOf(name=%s, n=%s, random_order=%s, augmenters=[%s], deterministic=%s)" % (self.name, str(self.n), str(self.random_order), augs_str, self.deterministic)

def OneOf(children, name=None, deterministic=False, random_state=None):
    """
    Augmenter that always executes exactly one of its children.

    Parameters
    ----------
    children : list of Augmenter
        The choices of augmenters to apply.

    random_order : boolean, optional(default=False)
        Whether to apply the child augmenters in random order per image.
        The order is resampled for each image.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
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

    either flips each image horizontally, or ads blur+dropout+noise or does
    nothing.

    """
    return SomeOf(n=1, children=children, random_order=False, name=name, deterministic=deterministic, random_state=random_state)

class Sometimes(Augmenter):
    """
    Augment only p percent of all images with one or more augmenters.

    Let C be one or more child augmenters given to Sometimes.
    Let p be the percent of images to augment.
    Let I be the input images.
    Then (on average) p percent of all images in I will be augmented using C.

    Parameters
    ----------
    p : float or StochasticParameter, optional(default=0.5)
        Sets the probability with which the given augmenters will be applied to
        input images. E.g. a value of 0.5 will result in 50 percent of all
        input images being augmented.

    then_list : None or Augmenter or list of Augmenters, optional(default=None)
        Augmenter(s) to apply to p percent of all images.

    else_list : None or Augmenter or list of Augmenters, optional(default=None)
        Augmenter(s) to apply to (1-p) percent of all images.
        These augmenters will be applied only when the ones in then_list
        are NOT applied (either-or-relationship).

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3))

    when calling `aug.augment_images()`, only (on average) 50 percent of
    all images will be blurred.

    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3), iaa.Fliplr(1.0))

    when calling `aug.augment_images()`, (on average) 50 percent of all images
    will be blurred, the other (again, on average) 50 percent will be
    horizontally flipped.

    """

    def __init__(self, p=0.5, then_list=None, else_list=None, name=None, deterministic=False, random_state=None):
        super(Sometimes, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_float(p) or ia.is_single_integer(p):
            ia.do_assert(0 <= p <= 1)
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected float/int in range [0, 1] or StochasticParameter as p, got %s." % (type(p),))

        if then_list is None:
            self.then_list = Sequential([], name="%s-then" % (self.name,))
        elif ia.is_iterable(then_list):
            # TODO does this work with SomeOf(), Sequential(), ... ?
            self.then_list = Sequential(then_list, name="%s-then" % (self.name,))
        elif isinstance(then_list, Augmenter):
            self.then_list = Sequential([then_list], name="%s-then" % (self.name,))
        else:
            raise Exception("Expected None, Augmenter or list/tuple as then_list, got %s." % (type(then_list),))

        if else_list is None:
            self.else_list = Sequential([], name="%s-else" % (self.name,))
        elif ia.is_iterable(else_list):
            self.else_list = Sequential(else_list, name="%s-else" % (self.name,))
        elif isinstance(else_list, Augmenter):
            self.else_list = Sequential([else_list], name="%s-else" % (self.name,))
        else:
            raise Exception("Expected None, Augmenter or list/tuple as else_list, got %s." % (type(else_list),))

    def _augment_images(self, images, random_state, parents, hooks):
        if hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
            input_is_np_array = ia.is_np_array(images)
            if input_is_np_array:
                input_dtype = images.dtype

            nb_images = len(images)
            samples = self.p.draw_samples((nb_images,), random_state=random_state)

            # create lists/arrays of images for if and else lists (one for each)
            indices_then_list = np.where(samples == 1)[0] # np.where returns tuple(array([0, 5, 9, ...])) or tuple(array([]))
            indices_else_list = np.where(samples == 0)[0]
            if isinstance(images, list):
                images_then_list = [images[i] for i in indices_then_list]
                images_else_list = [images[i] for i in indices_else_list]
            else:
                images_then_list = images[indices_then_list]
                images_else_list = images[indices_else_list]

            # augment according to if and else list
            result_then_list = self.then_list.augment_images(
                images=images_then_list,
                parents=parents + [self],
                hooks=hooks
            )
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

            # if input was a list, keep the output as a list too,
            # otherwise it was a numpy array, so make the output a numpy array too
            if input_is_np_array:
                result = np.array(result, dtype=input_dtype)
        else:
            result = images

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        # TODO this is mostly copy pasted from _augment_images, make dry
        result = keypoints_on_images
        if hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents, default=True):
            nb_images = len(keypoints_on_images)
            samples = self.p.draw_samples((nb_images,), random_state=random_state)

            # create lists/arrays of images for if and else lists (one for each)
            indices_then_list = np.where(samples == 1)[0] # np.where returns tuple(array([0, 5, 9, ...])) or tuple(array([]))
            indices_else_list = np.where(samples == 0)[0]
            images_then_list = [keypoints_on_images[i] for i in indices_then_list]
            images_else_list = [keypoints_on_images[i] for i in indices_else_list]

            # augment according to if and else list
            result_then_list = self.then_list.augment_keypoints(
                keypoints_on_images=images_then_list,
                parents=parents + [self],
                hooks=hooks
            )
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
        aug.then_list = aug.then_list.to_deterministic()
        aug.else_list = aug.else_list.to_deterministic()
        aug.deterministic = True
        aug.random_state = ia.new_random_state()
        return aug

    def get_parameters(self):
        return [self.p]

    def get_children_lists(self):
        return [self.then_list, self.else_list]

    def __str__(self):
        return "Sometimes(p=%s, name=%s, then_list=[%s], else_list=[%s], deterministic=%s)" % (self.p, self.name, self.then_list, self.else_list, self.deterministic)

class WithChannels(Augmenter):
    """
    Apply child augmenters to specific channels.

    Let C be one or more child augmenters given to this augmenter.
    Let H be a list of channels.
    Let I be the input images.
    Then this augmenter will pick the channels H from each image
    in I (resulting in new images) and apply C to them.
    The result of the augmentation will be merged back into the original
    images.

    Parameters
    ----------
    channels : integer or list of integers or None, optional(default=None)
        Sets the channels to extract from each image.
        If None, all channels will be used.

    children : Augmenter or list of Augmenters or None, optional(default=None)
        One or more augmenters to apply to images, after the channels
        are extracted.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.WithChannels([0], iaa.Add(10))

    assuming input images are RGB, then this augmenter will add 10 only
    to the first channel, i.e. make images more red.

    """

    def __init__(self, channels=None, children=None, name=None, deterministic=False, random_state=None):
        super(WithChannels, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if channels is None:
            self.channels = None
        elif ia.is_single_integer(channels):
            self.channels = [channels]
        elif ia.is_iterable(channels):
            ia.do_assert(all([ia.is_single_integer(channel) for channel in channels]), "Expected integers as channels, got %s." % ([type(channel) for channel in channels],))
            self.channels = channels
        else:
            raise Exception("Expected None, int or list of ints as channels, got %s." % (type(channels),))

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
            if self.channels is None:
                result = self.children.augment_images(
                    images=images,
                    parents=parents + [self],
                    hooks=hooks
                )
            elif len(self.channels) == 0:
                pass
            else:
                if ia.is_np_array(images):
                    images_then_list = images[..., self.channels]
                else:
                    images_then_list = [image[..., self.channels] for image in images]

                result_then_list = self.children.augment_images(
                    images=images_then_list,
                    parents=parents + [self],
                    hooks=hooks
                )

                if ia.is_np_array(images):
                    result[..., self.channels] = result_then_list
                else:
                    for i in sm.xrange(len(images)):
                        result[i][..., self.channels] = result_then_list[i]

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
        return "WithChannels(channels=%s, name=%s, children=[%s], deterministic=%s)" % (self.channels, self.name, self.children, self.deterministic)

class Noop(Augmenter):
    """
    Augmenter that never changes input images ("no operation").

    This augmenter is useful when you just want to use a placeholder augmenter
    in some situation, so that you can continue to call `augment_images()`,
    without actually changing them (e.g. when switching from training to test).

    Parameters
    ----------
    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        #Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        super(Noop, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []


class Lambda(Augmenter):
    """
    Augmenter that calls a lambda function for each batch of input image.

    This is useful to add missing functions to a list of augmenters.

    Parameters
    ----------
    func_images : callable,
        The function to call for each batch of images.
        It must follow the form

            ``function(images, random_state, parents, hooks)``

        and return the changed images (may be transformed in-place).
        This is essentially the interface of `Augmenter._augment_images()`.

    func_keypoints : callable,
        The function to call for each batch of image keypoints.
        It must follow the form

            ``function(keypoints_on_images, random_state, parents, hooks)``

        and return the changed keypoints (may be transformed in-place).
        This is essentially the interface of `Augmenter._augment_keypoints()`.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> def func_images(images, random_state, parents, hooks):
    >>>     images[:, ::2, :, :] = 0
    >>>     return images
    >>>
    >>> def func_keypoints(keypoints_on_images, random_state, parents, hooks):
    >>>     return keypoints_on_images
    >>>
    >>> aug = iaa.Lambda(
    >>>     func_images=func_images,
    >>>     func_keypoints=func_keypoints
    >>> )

    Replaces every second row in images with black pixels and leaves keypoints
    unchanged.

    """

    def __init__(self, func_images, func_keypoints, name=None, deterministic=False, random_state=None):
        #Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        super(Lambda, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.func_images = func_images
        self.func_keypoints = func_keypoints

    def _augment_images(self, images, random_state, parents, hooks):
        return self.func_images(images, random_state, parents=parents, hooks=hooks)

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = self.func_keypoints(keypoints_on_images, random_state, parents=parents, hooks=hooks)
        ia.do_assert(isinstance(result, list))
        ia.do_assert(all([isinstance(el, ia.KeypointsOnImage) for el in result]))
        return result

    def get_parameters(self):
        return []


def AssertLambda(func_images, func_keypoints, name=None, deterministic=False, random_state=None):
    """
    Augmenter that runs an assert on each batch of input images
    using a lambda function as condition.

    This is useful to make generic assumption about the input images and error
    out early if they aren't met.

    Parameters
    ----------
    func_images : callable,
        The function to call for each batch of images.
        It must follow the form
            ``function(images, random_state, parents, hooks)``
        and return either True (valid input) or False (invalid input).
        It essentially reuses the interface of Augmenter._augment_images().

    func_keypoints : callable,
        The function to call for each batch of keypoints.
        It must follow the form
            ``function(keypoints_on_images, random_state, parents, hooks)``
        and return either True (valid input) or False (invalid input).
        It essentially reuses the interface of Augmenter._augment_keypoints().

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`
    """
    def func_images_assert(images, random_state, parents, hooks):
        ia.do_assert(func_images(images, random_state, parents=parents, hooks=hooks), "Input images did not fulfill user-defined assertion in AssertLambda.")
        return images
    def func_keypoints_assert(keypoints_on_images, random_state, parents, hooks):
        ia.do_assert(func_keypoints(keypoints_on_images, random_state, parents=parents, hooks=hooks), "Input keypoints did not fulfill user-defined assertion in AssertLambda.")
        return keypoints_on_images
    if name is None:
        name = "UnnamedAssertLambda"
    return Lambda(func_images_assert, func_keypoints_assert, name=name, deterministic=deterministic, random_state=random_state)


def AssertShape(shape, check_images=True, check_keypoints=True, name=None, deterministic=False, random_state=None):
    """
    Augmenter to make assumptions about the shape of input image(s)
    and keypoints.

    Parameters
    ----------
    shape : tuple with each entry being None or tuple of two ints or list of ints
        The expected shape. Given as a tuple. The number of entries in the tuple
        must match the number of dimensions, i.e. usually four entries for
        (N, H, W, C).
            * If an entry is None, any value for that dimensions is accepted.
            * If an entry is int, exactly that integer value will be accepted
              or no other value.
            * If an entry is a tuple of two ints with values a and b, only a
              value x with a <= x < b will be accepted for the dimension.
            * If an entry is a list of ints, only a value for the dimension
              will be accepted which is contained in the list.

    check_images : bool, optional(default=True)
        Whether to validate input images via the given shape.

    check_keypoints : bool, optional(default=True)
        Whether to validate input keypoints via the given shape.
        The number of keypoints will be checked and for each KeypointsOnImage
        instance its image's shape, i.e. KeypointsOnImage.shape.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> seq = iaa.Sequential([
    >>>     iaa.AssertShape((None, 32, 32, 3)),
    >>>     iaa.Fliplr(0.5)
    >>> ])

    will first check for each image batch, if it contains a variable number of
    32x32 images with 3 channels each. Only if that check succeeds, the
    horizontal flip will be executed (otherwise an assertion error will be
    thrown).

    >>> seq = iaa.Sequential([
    >>>     iaa.AssertShape((None, (32, 64), 32, [1, 3])),
    >>>     iaa.Fliplr(0.5)
    >>> ])

    like above, but now the height may be in the range 32 <= H < 64 and
    the number of channels may be either 1 or 3.
    """
    ia.do_assert(len(shape) == 4, "Expected shape to have length 4, got %d with shape: %s." % (len(shape), str(shape)))

    def compare(observed, expected, dimension, image_index):
        if expected is not None:
            if ia.is_single_integer(expected):
                ia.do_assert(observed == expected, "Expected dim %d (entry index: %s) to have value %d, got %d." % (dimension, image_index, expected, observed))
            elif isinstance(expected, tuple):
                ia.do_assert(len(expected) == 2)
                ia.do_assert(expected[0] <= observed < expected[1], "Expected dim %d (entry index: %s) to have value in range [%d, %d), got %d." % (dimension, image_index, expected[0], expected[1], observed))
            elif isinstance(expected, list):
                ia.do_assert(any([observed == val for val in expected]), "Expected dim %d (entry index: %s) to have any value of %s, got %d." % (dimension, image_index, str(expected), observed))
            else:
                raise Exception("Invalid datatype for shape entry %d, expected each entry to be an integer, a tuple (with two entries) or a list, got %s." % (dimension, type(expected),))

    def func_images(images, random_state, parents, hooks):
        if check_images:
            #ia.do_assert(is_np_array(images), "AssertShape can currently only handle numpy arrays, got ")
            if isinstance(images, list):
                if shape[0] is not None:
                    compare(len(images), shape[0], 0, "ALL")

                for i in sm.xrange(len(images)):
                    image = images[i]
                    ia.do_assert(len(image.shape) == 3, "Expected image number %d to have a shape of length 3, got %d (shape: %s)." % (i, len(image.shape), str(image.shape)))
                    for j in sm.xrange(len(shape)-1):
                        expected = shape[j+1]
                        observed = image.shape[j]
                        compare(observed, expected, j, i)
            else:
                ia.do_assert(len(images.shape) == 4, "Expected image's shape to have length 4, got %d (shape: %s)." % (len(images.shape), str(images.shape)))
                for i in range(4):
                    expected = shape[i]
                    observed = images.shape[i]
                    compare(observed, expected, i, "ALL")
        return images

    def func_keypoints(keypoints_on_images, random_state, parents, hooks):
        if check_keypoints:
            #ia.do_assert(is_np_array(images), "AssertShape can currently only handle numpy arrays, got ")
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
        name = "UnnamedAssertShape"

    return Lambda(func_images, func_keypoints, name=name, deterministic=deterministic, random_state=random_state)
