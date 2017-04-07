from __future__ import print_function, division, absolute_import
from . import imgaug as ia
from .parameters import StochasticParameter, Deterministic, Binomial, Choice, DiscreteUniform, Normal, Uniform
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy as copy_module
import re
import math
from scipy import misc, ndimage
from skimage import transform as tf, segmentation, measure
import itertools
import cv2
import six
import six.moves as sm
import types

"""
TODOs
    - add a version of augment_images that skips validation and copying which
        is called by augmenters with children
    - check if all get_parameters() implementations really return all parameters.
    - Add Alpha augmenter
    - Add SpatialDropout augmenter
    - Add CoarseDropout shortcut function
    - Add Hue and Saturation augmenters
    - Add uniform blurring augmenter
    - Add bilateral filter augmenter
    - Add median filter augmenter
    - Add random blurring shortcut (either uniform or gaussian or bilateral or median)
    - Add Max-Pooling augmenter
    - Add edge pronounce augmenter
    - Add Cartoon augmenter
    - Add OilPainting augmenter
"""

@six.add_metaclass(ABCMeta)
class Augmenter(object):
    """Base class for Augmenter objects.
    All augmenters derive from this class.
    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        """Create a new Augmenter instance.

        Parameters
        ----------
        name : string or None, optional(default=None)
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
            augmenter.to_deterministic().

        random_state : int or np.random.RandomState or None, optional(default=None)
            The random state to use for this augmenter.
            If int, a new np.random.RandomState will be created using this
            value as the seed.
            If np.random.RandomState instance, the instance will be used directly.
            If None, imgaug's default RandomState will be used, which's state can
            be controlled using imgaug.seed(int).
            Usually, there is no need to set this variable by hand. Instead,
            instantiate the augmenter with the defaults and then use
            augmenter.to_deterministic().
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

    def augment_batches(self, batches, hooks=None):
        """Augment multiple batches of images.

        Parameters
        ----------
        batches : list
            List of image batches to augment.
            Expected is a list of array-likes, each one being either
            (a) a list of arrays of shape (H, W) or (H, W, C) or
            (b) an array of shape (N, H, W) or (N, H, W, C),
            where N = number of images, H = height, W = width,
            C = number of channels.
            Each image should have dtype uint8 (range 0-255).

        hooks : ia.HooksImages or None, optional(default=None)
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        augmented_batch : list
            Corresponding list of batches of augmented images.
        """
        assert isinstance(batches, list)
        return [self.augment_images(batch, hooks=hooks) for batch in batches]

    def augment_image(self, image, hooks=None):
        """Augment a single image.

        Parameters
        ----------
        image : (H, W, C) ndarray or (H, W) ndarray
            The image to augment. Should have dtype uint8 (range 0-255).

        hooks : ia.HooksImages or None, optional(default=None)
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        img : ndarray
            The corresponding augmented image.
        """
        assert image.ndim in [2, 3], "Expected image to have shape (height, width, [channels]), got shape %s." % (image.shape,)
        return self.augment_images([image], hooks=hooks)[0]

    def augment_images(self, images, parents=None, hooks=None):
        """Augment multiple images.

        Parameters
        ----------
        images : (N, H, W, C) ndarray or (N, H, W) ndarray
                or list of ndarray (each either (H, W, C) or (H, W))
            Images to augment. The input can be a list of numpy arrays or
            a single array. Each array is expected to have shape (H, W, C)
            or (H, W), where H is the height, W is the width and C are the
            channels. Number of channels may differ between images.
            If a list is chosen, height and width may differ per between images.
            Currently the recommended dtype is uint8 (i.e. integer values in
            the range 0 to 255). Other dtypes are not tested.

        parents : list of Augmenter or None, optional(default=None)
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : ia.HooksImages or None, optional(default=None)
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        images_result : array-like
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

            assert images.ndim in [3, 4], "Expected 3d/4d array of form (N, height, width) or (N, height, width, channels), got shape %s." % (images.shape,)

            # copy the input, we don't want to augment it in-place
            images_copy = np.copy(images)

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
                assert all(image.ndim in [2, 3] for image in images), "Expected list of images with each image having shape (height, width) or (height, width, channels), got shapes %s." % ([image.shape for image in images],)

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
            #assert images.shape[0] > 0, images.shape
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
                self.random_state.uniform()
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
        """Augment multiple images.

        This is the internal variation of augment_images().
        It is called from augment_images() and should usually not be called
        directly.
        It has to be implemented by every augmenter.
        This method may transform the images in-place.
        This method does not have to care about determinism or the
        Augmenter instance's random_state variable. The parameter random_state
        takes care of both of these.

        Parameters
        ----------
        images : list of (H, W, C) ndarray or a single (N, H, W, C) ndarray
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
            (Here never None, but instead an empty list in these cases.)

        hooks : ia.HooksImages
            See augment_images().
            (Here never None, is always an ia.HooksImages instance.)

        Returns
        ----------
        images : list of (H, W, C) ndarray or a single (N, H, W, C) ndarray
            The augmented images.
        """
        raise NotImplementedError()

    def augment_keypoints(self, keypoints_on_images, parents=None, hooks=None):
        """Augment image keypoints.

        This is the corresponding function to augment_images(), just for
        keypoints/landmarks (i.e. coordinates on the image).
        Usually you will want to call augment_images() with a list of images,
        e.g. augment_images([A, B, C]) and then augment_keypoints() with the
        corresponding list of keypoints on these images, e.g.
        augment_keypoints([Ak, Bk, Ck]), where Ak are the keypoints on image A.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting the images and keypoints,
        e.g. by
            seq = iaa.Fliplr(0.5)
            seq_det = seq.to_deterministic()
            imgs_aug = seq_det.augment_images([A, B, C])
            kps_aug = seq_det.augment_keypoints([Ak, Bk, Ck])
        Otherwise, different random values will be sampled for the image
        and keypoint augmentations, resulting in different augmentations (e.g.
        images might be rotated by 30deg and keypoints by -10deg).
        Also make sure to call to_deterministic() again for each new batch,
        otherwise you would augment all batches in the same way.


        Parameters
        ----------
        keypoints_on_images : list of ia.KeypointsOnImage
            The keypoints/landmarks to augment.
            Expected is a list of ia.KeypointsOnImage objects,
            each containing the keypoints of a single image.

        parents : list of Augmenter or None, optional(default=None)
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : ia.HooksKeypoints or None, optional(default=None)
            HooksKeypoints object to dynamically interfere with the
            augmentation process.

        Returns
        -------
        keypoints_on_images_result : list of augmented ia.KeypointsOnImage
            objects.
        """
        if self.deterministic:
            state_orig = self.random_state.get_state()

        if parents is None:
            parents = []

        if hooks is None:
            hooks = ia.HooksKeypoints()

        assert ia.is_iterable(keypoints_on_images)
        assert all([isinstance(keypoints_on_image, ia.KeypointsOnImage) for keypoints_on_image in keypoints_on_images])

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
                self.random_state.uniform()
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
        """Augment keypoints on multiple images.

        This is the internal variation of augment_keypoints().
        It is called from augment_keypoints() and should usually not be called
        directly.
        It has to be implemented by every augmenter.
        This method may transform the keypoints in-place.
        This method does not have to care about determinism or the
        Augmenter instance's random_state variable. The parameter random_state
        takes care of both of these.

        Parameters
        ----------
        keypoints_on_images : list of ia.KeypointsOnImage
            Keypoints to augment. They may be changed in-place.

        random_state : np.random.RandomState
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of Augmenter
            See augment_keypoints().
            (Here never None, but instead an empty list in these cases.)

        hooks : ia.HooksImages
            See augment_keypoints().
            (Here never None, is always an ia.HooksKeypoints instance.)

        Returns
        ----------
        images : list of (H, W, C) ndarray or a single (N, H, W, C) ndarray
            The augmented images.
        """
        raise NotImplementedError()

    # TODO most of the code of this function could be replaced with ia.draw_grid()
    # TODO add parameter for handling multiple images ((a) next to each other
    # in each row or (b) multiply row count by number of images and put each
    # one in a new row)
    def draw_grid(self, images, rows, cols):
        """Apply this augmenter to the given images and return a grid
        image of the results.
        Each cell in the grid contains a single augmented variation of
        an input image.

        If multiple images are provided, the row count is multiplied by
        the number of images and each image gets its own row.
        E.g. for images = [A, B], rows=2, cols=3:
            A A A
            B B B
            A A A
            B B B
        for images = [A], rows=2, cols=3:
            A A A
            A A A

        Parameters
        -------
        images : list of ndarray or ndarray
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
        grid : ndarray of shape (H, W, 3).
            The generated grid image with augmented versions of the input
            images. Here, H and W reference the output size of the grid,
            and _not_ the sizes of the input images.
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
        assert isinstance(images, list)

        det = self if self.deterministic else self.to_deterministic()
        augs = []
        for image in images:
            augs.append(det.augment_images([image] * (rows * cols)))

        augs_flat = list(itertools.chain(*augs))
        cell_height = max([image.shape[0] for image in images] + [image.shape[0] for image in augs_flat])
        cell_width = max([image.shape[1] for image in images] + [image.shape[1] for image in augs_flat])
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
        """Apply this augmenter to the given images and show/plot the results as
        a grid of images.

        If multiple images are provided, the row count is multiplied by
        the number of images and each image gets its own row.
        E.g. for images = [A, B], rows=2, cols=3:
            A A A
            B B B
            A A A
            B B B
        for images = [A], rows=2, cols=3:
            A A A
            A A A

        Parameters
        ----------
        images : list of ndarray or ndarray
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
        """
        grid = self.draw_grid(images, rows, cols)
        misc.imshow(grid)

    def to_deterministic(self, n=None):
        """Converts this augmenter from a stochastic to a deterministic one.

        A stochastic augmenter samples new values for each parameter per image.
        Feed a new batch of images into the augmenter and you will get a
        new set of transformations.
        A deterministic augmenter also samples new values for each parameter
        per image, but starts each batch the same RandomState (i.e. seed).
        Feed two batches of images into the augmenter and you get the same
        transformations both times (same number of images assumed; some
        augmenter's results are also dependend on image height, width and
        channel count).

        Using determinism is useful for keypoint augmentation,
        as you will usually want to augment images and their corresponding
        keypoints in the same way (e.g. if an image is rotated by 30deg, then
        also rotate its keypoints by 30deg).

        Parameters
        ----------
        n : int or None, optional(default=None)
            Number of deterministic augmenters to return.
            If None then only one Augmenter object will be returned.
            If 1 or higher, then a list containing n Augmenter objects
            will be returned.

        Returns
        -------
        det : Augmenter object or list of Augmenter object
            A single Augmenter object if n was None,
            otherwise a list of Augmenter objects (even if n was 1).
        """
        assert n is None or n >= 1
        if n is None:
            return self.to_deterministic(1)[0]
        else:
            return [self._to_deterministic() for _ in sm.xrange(n)]

    def _to_deterministic(self):
        """Augmenter-specific implementation of to_deterministic().
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

    def reseed(self, deterministic_too=False, random_state=None):
        """Reseed this augmenter and all of its children (if it has any).

        This function is useful, when augmentations are run in the
        background (i.e. on multiple cores).
        It should be called before sending this Augmenter object to a
        background worker (i.e., if N workers are used, the function
        should be called N times). Otherwise, all background workers will
        use the same seeds and therefore apply the same augmentations.

        Parameters
        ----------
        deterministic_too : bool, optional(default=False)
            Whether to also change the seed of an augmenter A, if A
            is deterministic. This is the case both when this augmenter
            object is A or one of its children is A.

        random_state : np.random.RandomState or int, optional(default=None)
            Generator that creates new random seeds.
            If int, it will be used as a seed.
            If None, a new RandomState will automatically be created.
        """
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
                aug.reseed(deterministic_too=deterministic_too, random_state=random_state)

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError()

    def get_children_lists(self):
        return []

    def find_augmenters(self, func, parents=None, flat=True):
        """Find augmenters that match a condition.
        This function will compare this augmenter and all of its children
        with a condition. The condition is a lambda function.

        Example:
            aug = iaa.Sequential([
                nn.Fliplr(0.5, name="fliplr"),
                nn.Flipud(0.5, name="flipud")
            ])
            print(aug.find_augmenters(lambda a, parents: a.name == "fliplr"))
        will return the first child augmenter (Fliplr instance).

        Parameters
        ----------
        func : callable
            A function that receives an Augmenter instance and a list of
            parent Augmenter instances and must return True, if that
            augmenter is valid match.
            E.g.
                lambda a, parents: a.name == "fliplr" and any([b.name == "other-augmenter" for b in parents])

        parents : list of Augmenter or None, optional(default=None)
            List of parent augmenters.
            Intended for nested calls and can usually be left as None.

        flat : bool, optional(default=True)
            Whether to return the result as a flat list (True)
            or a nested list (False). In the latter case, the nesting matches
            each augmenters position among the children.

        Returns
        ----------
        augmenters : list of Augmenter objects
            Nested list if flat was set to False.
            Flat list if flat was set to True.
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

        regex : bool, optional(default=False)
            Whether `name` parameter is a regular expression.

        flat : bool, optional(default=True)
            See `Augmenter.find_augmenters()`.

        Returns
        -------
        augmenters : list of Augmenter objects
            Nested list if flat was set to False.
            Flat list if flat was set to True.
        """
        return self.find_augmenters_by_names([name], regex=regex, flat=flat)

    def find_augmenters_by_names(self, names, regex=False, flat=True):
        """Find augmenter(s) by names.

        Parameters
        ----------
        names : list of strings
            Names of the augmenter(s) to search for.

        regex : bool, optional(default=False)
            Whether `names` is a list of regular expressions.

        flat : boolean, optional(default=True)
            See `Augmenter.find_augmenters()`.

        Returns
        -------
        augmenters : list of Augmenter objects
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
        """Remove this augmenter or its children that match a condition.

        Example:
            seq = iaa.Sequential([
                iaa.Fliplr(0.5, name="fliplr"),
                iaa.Flipud(0.5, name="flipud"),
            ])
            seq = seq.remove_augmenters(lambda a, parents: a.name == "fliplr")
        removes the augmenter Fliplr from the Sequential object's children.

        Parameters
        ----------
        func : callable
            Condition to match per augmenter.
            The function must expect the augmenter itself and a list of parent
            augmenters and returns True if that augmenter is to be removed,
            or False otherwise.
            E.g. lambda a, parents: a.name == "fliplr" and len(parents) == 1
            removes an augmenter with name "fliplr" if it is the direct child
            of the augmenter upon which remove_augmenters() was initially called.

        copy : bool, optional(default=True)
            Whether to copy this augmenter and all if its children before
            removing. If False, removal is performed in-place.

        noop_if_topmost : boolean, optional(default=True)
            If True and the condition (lambda function) leads to the removal
            of the topmost augmenter (the one this function is called on
            initially), then that topmost augmenter will be replaced by a
            Noop instance (i.e. an object will still knows augment_images(),
            but doesnt change images). If False, None will be returned in
            these cases.
            This can only be False if copy is set to True.

        Returns
        -------
        aug : Augmenter or None
            This augmenter after the removal was performed.
            Is None iff condition was matched for the topmost augmenter,
            copy was set to True and noop_if_topmost was set to False.
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
        """Remove in-place children of this augmenter that match a condition.

        E.g. seq = iaa.Sequential([
                iaa.Fliplr(0.5, name="fliplr"),
                iaa.Flipud(0.5, name="flipud"),
            ])
            seq.remove_augmenters_inplace(lambda a, parents: a.name == "fliplr")
        removes the augmenter Fliplr from the Sequential object's children.

        This is functionally identical to remove_augmenters() with copy=False,
        except that it does not affect the topmost augmenter (the one on
        which this function is initially called on).

        Parameters
        ----------
        func : lambda function
            See Augmenter.remove_augmenters().

        parents : list of Augmenter or None, optional(default=None)
            List of parent Augmenter instances that lead to this
            Augmenter. If None, an empty list will be used.
            This parameter can usually be left empty and will be set
            automatically for children.
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
        """Create a shallow copy of this Augmenter instance.

        Returns
        -------
        aug : Augmenter
            Shallow copy of this Augmenter instance.
        """
        return copy_module.copy(self)

    def deepcopy(self):
        """Create a deep copy of this Augmenter instance.

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
    """List augmenter that may contain other augmenters to apply in sequence
    or random order.
    """

    def __init__(self, children=None, random_order=False, name=None, deterministic=False, random_state=None):
        """Initialize a new Sequential instance.

        Example:
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])
            imgs_aug = seq.augment_images(imgs)
        Calls always first the horizontal flip augmenter and then the vertical
        flip augmenter (each having a probability of 50 percent to be used).

        Example:
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ], random_order=True)
            imgs_aug = seq.augment_images(imgs)
        Calls sometimes first the horizontal flip augmenter and sometimes first the
        vertical flip augmenter (each again with 50 percent probability to be used).

        Parameters
        ----------
        children : Augmenter or list of Augmenter or None, optional(default=None)
            The augmenters to apply to images.

        random_order : boolean, optional(default=False)
            Whether to apply the child augmenters in random order per image.
            The order is resampled for each image.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : boolean, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        list.__init__(self, children if children is not None else [])
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
        # augs_str = ", ".join([aug.__str__() for aug in self.children])
        augs_str = ", ".join([aug.__str__() for aug in self])
        return "Sequential(name=%s, augmenters=[%s], deterministic=%s)" % (self.name, augs_str, self.deterministic)


class Sometimes(Augmenter):
    """Augment only p percent of all images with one or more augmenters.

    Let C be one or more child augmenters given to Sometimes.
    Let p be the percent of images to augment.
    Let I be the input images.
    Then (on average) p percent of all images in I will be augmented using C.
    """

    def __init__(self, p=0.5, then_list=None, else_list=None, name=None, deterministic=False, random_state=None):
        """Instantiate a new Sometimes instance.

        Example:
            aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3))
        when calling aug.augment_images(...), only (on average) 50 percent of
        all images will be blurred.

        Example:
            aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3), iaa.Fliplr(1.0))
        when calling aug.augment_images(...), (on average) 50 percent of all images
        will be blurred, the other (again, on average) 50 percent will be
        horizontally flipped.

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
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(Sometimes, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_float(p) or ia.is_single_integer(p):
            assert 0 <= p <= 1
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected float/int in range [0, 1] or StochasticParameter as p, got %s." % (type(p),))

        if then_list is None:
            self.then_list = Sequential([], name="%s-then" % (self.name,))
        elif ia.is_iterable(then_list):
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
        result = images
        if hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
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
            if not isinstance(images, list):
                result = np.array(result, dtype=np.uint8)

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


class InColorspace(Augmenter):
    """Select colorspace for augumentation.

    This augumenter applies children augumenters with changing the colorspace to specified one.
    See ~imgaug.augumenters.ChainerColorspace for detail.

    Example:
        aug = iaa.InColorspace(to_colorspace="HSV", from_colorspace="RGB",
                               children=iaa.WithChannels(0, iaa.Add(10)))
    This augmenter will add 10 to Hue value in HSV colorspace,
    then return the colorspace to the original, RGB.
    """

    def __init__(self, to_colorspace, from_colorspace="RGB", children=None, name=None, deterministic=False, random_state=None):
        super(InColorspace, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

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
        return ("InColorspace(from_colorspace=%s, to_colorspace=%s, name=%s, children=[%s], deterministic=%s)" %
                (self.from_colorspace, self.to_colorspace, self.name, self.children, self.deterministic))


class WithChannels(Augmenter):
    """Select channels to augment.

    Let C be one or more child augmenters given to this augmenter.
    Let H be a list of channels.
    Let I be the input images.
    Then this augmenter will pick the channels H from each image
    in I (resulting in new images) and apply C to them.
    The result of the augmentation will be merged back into the original
    images.

    Example:
        aug = iaa.WithChannels([0], iaa.Add(10))
    assuming input images are RGB, then this augmenter will add 10 only
    to the first channel, i.e. make images more red.
    """

    def __init__(self, channels=None, children=None, name=None, deterministic=False, random_state=None):
        """Instantiate a new WithChannels augmenter.

        Parameters
        ----------
        channels : integer, list of integers, None, optional(default=None)
            Sets the channels to extract from each image.
            If None, all channels will be used.

        children : Augmenter, list of Augmenters, None, optional(default=None)
            One or more augmenters to apply to images, after the channels
            are extracted.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(WithChannels, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if channels is None:
            self.channels = None
        elif ia.is_single_integer(channels):
            self.channels = [channels]
        elif ia.is_iterable(channels):
            assert all([ia.is_single_integer(channel) for channel in channels]), "Expected integers as channels, got %s." % ([type(channel) for channel in channels],)
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
    """Augmenter that never changes input images ("no operation").

    This augmenter is useful when you just want to use a placeholder augmenter
    in some situation, so that you can continue to call augment_images(),
    without actually changing them (e.g. when switching from training to test).
    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        """Instantiate a new Noop instance.

        Parameters
        ----------
        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        #Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        super(Noop, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []


class Lambda(Augmenter):
    """Augmenter that calls a lambda function for each batch of input image.

    This is useful to add missing functions to a list of augmenters.
    """

    def __init__(self, func_images, func_keypoints, name=None, deterministic=False, random_state=None):
        """Instantiate a new Lambda instance.

        Parameters
        ----------
        func_images : function,
            The function to call for each batch of images.
            It must follow the form
                `function(images, random_state, parents, hooks)`
            and return the changed images (may be transformed in-place).
            This is essentially the interface of Augmenter._augment_images().

        func_keypoints : function,
            The function to call for each batch of image keypoints.
            It must follow the form
                `function(keypoints_on_images, random_state, parents, hooks)`
            and return the changed keypoints (may be transformed in-place).
            This is essentially the interface of Augmenter._augment_keypoints().

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        #Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        super(Lambda, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.func_images = func_images
        self.func_keypoints = func_keypoints

    def _augment_images(self, images, random_state, parents, hooks):
        return self.func_images(images, random_state, parents=parents, hooks=hooks)

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = self.func_keypoints(keypoints_on_images, random_state, parents=parents, hooks=hooks)
        assert isinstance(result, list)
        assert all([isinstance(el, ia.KeypointsOnImage) for el in result])
        return result

    def get_parameters(self):
        return []


def AssertLambda(func_images, func_keypoints, name=None, deterministic=False, random_state=None):
    """Augmenter that runs an assert on each batch of input images
    using a lambda function as condition.

    This is useful to make generic assumption about the input images and error
    out early if they aren't met.

    Parameters
    ----------
    func_images : callable,
        The function to call for each batch of images.
        It must follow the form
            `function(images, random_state, parents, hooks)`
        and return either True (valid input) or False (invalid input).
        It essentially reuses the interface of Augmenter._augment_images().

    func_keypoints : callable,
        The function to call for each batch of keypoints.
        It must follow the form
            `function(keypoints_on_images, random_state, parents, hooks)`
        and return either True (valid input) or False (invalid input).
        It essentially reuses the interface of Augmenter._augment_keypoints().

    name : string, optional(default=None)
        See Augmenter.__init__()

    deterministic : bool, optional(default=False)
        See Augmenter.__init__()

    random_state : int or np.random.RandomState or None, optional(default=None)
        See Augmenter.__init__()
    """
    def func_images_assert(images, random_state, parents, hooks):
        assert func_images(images, random_state, parents=parents, hooks=hooks)
        return images
    def func_keypoints_assert(keypoints_on_images, random_state, parents, hooks):
        assert func_keypoints(keypoints_on_images, random_state, parents=parents, hooks=hooks)
        return keypoints_on_images
    if name is None:
        name = "UnnamedAssertLambda"
    return Lambda(func_images_assert, func_keypoints_assert, name=name, deterministic=deterministic, random_state=random_state)


def AssertShape(shape, check_images=True, check_keypoints=True, name=None, deterministic=False, random_state=None):
    """Augmenter to make assumptions about the shape of input image(s)
    and keypoints.

    Example:
        seq = iaa.Sequential([
            iaa.AssertShape((None, 32, 32, 3)),
            iaa.Fliplr(0.5)
        ])
    will first check for each image batch, if it contains a variable number of
    32x32 images with 3 channels each. Only if that check succeeds, the
    horizontal flip will be executed (otherwise an assertion error will be
    thrown).

    Example:
        seq = iaa.Sequential([
            iaa.AssertShape((None, (32, 64), 32, [1, 3])),
            iaa.Fliplr(0.5)
        ])
    like above, but now the height may be in the range 32 <= H < 64 and
    the number of channels may be either 1 or 3.

    Parameters
    ----------
    shape : tuple of None|int|tuple of two ints|list of ints,
        The expected shape.
        For number of entries must match the number of dimensions, i.e. usually
        follow (N, H, W, C).
        If an entry is None, any value for that dimensions is accepted.
        If an entry is int, exactly that integer value will be accepted or no
        other value.
        If an entry is a tuple of two ints with values a and b, only a
        value x with a <= x < b will be accepted for the dimension.
        If an entry is a list of ints, only a value for the dimension will
        be accepted which is contained in the list.

    check_images : bool, optional(default=True)
        Whether to validate input images via the given shape.

    check_keypoints : bool, optional(default=True)
        Whether to validate input keypoints via the given shape.
        The number of keypoints will be checked and for each KeypointsOnImage
        instance its image's shape, i.e. KeypointsOnImage.shape.

    name : string, optional(default=None)
        See Augmenter.__init__()

    deterministic : bool, optional(default=False)
        See Augmenter.__init__()

    random_state : int or np.random.RandomState or None, optional(default=None)
        See Augmenter.__init__()
    """
    assert len(shape) == 4, "Expected shape to have length 4, got %d with shape: %s." % (len(shape), str(shape))

    def compare(observed, expected, dimension, image_index):
        if expected is not None:
            if ia.is_single_integer(expected):
                assert observed == expected, "Expected dim %d (entry index: %s) to have value %d, got %d." % (dimension, image_index, expected, observed)
            elif isinstance(expected, tuple):
                assert len(expected) == 2
                assert expected[0] <= observed < expected[1], "Expected dim %d (entry index: %s) to have value in range [%d, %d), got %d." % (dimension, image_index, expected[0], expected[1], observed)
            elif isinstance(expected, list):
                assert any([observed == val for val in expected]), "Expected dim %d (entry index: %s) to have any value of %s, got %d." % (dimension, image_index, str(expected), observed)
            else:
                raise Exception("Invalid datatype for shape entry %d, expected each entry to be an integer, a tuple (with two entries) or a list, got %s." % (dimension, type(expected),))

    def func_images(images, random_state, parents, hooks):
        if check_images:
            #assert is_np_array(images), "AssertShape can currently only handle numpy arrays, got "
            if isinstance(images, list):
                if shape[0] is not None:
                    compare(len(images), shape[0], 0, "ALL")

                for i in sm.xrange(len(images)):
                    image = images[i]
                    assert len(image.shape) == 3, "Expected image number %d to have a shape of length 3, got %d (shape: %s)." % (i, len(image.shape), str(image.shape))
                    for j in sm.xrange(len(shape)-1):
                        expected = shape[j+1]
                        observed = image.shape[j]
                        compare(observed, expected, j, i)
            else:
                assert len(images.shape) == 4, "Expected image's shape to have length 4, got %d (shape: %s)." % (len(images.shape), str(images.shape))
                for i in range(4):
                    expected = shape[i]
                    observed = images.shape[i]
                    compare(observed, expected, i, "ALL")
        return images

    def func_keypoints(keypoints_on_images, random_state, parents, hooks):
        if check_keypoints:
            #assert is_np_array(images), "AssertShape can currently only handle numpy arrays, got "
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


class Crop(Augmenter):
    """Augmenter that crops images, i.e. extracts smaller subareas from them."""

    def __init__(self, px=None, percent=None, keep_size=True, name=None, deterministic=False, random_state=None):
        """Create a new Crop instance.

        Example:
            aug = iaa.Crop(px=(0, 10))
        crops each side by a random value from the range 0px to 10px (the value
        is sampled per side).

        Example:
            aug = iaa.Crop(px=((0, 10), (0, 5), (0, 10), (0, 5)))
        crops the top and bottom by a random value from the range 0px to 10px
        and the left and right by a random value in the range 0px to 5px.

        Example:
            aug = iaa.Crop(percent=(0, 0.1))
        crops each side by a random value from the range 0 percent to
        10 percent. (Percent with respect to the side's size, e.g. for the
        top side it uses the image's height.)

        Example:
            aug = iaa.Crop(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]))
        crops each side by either 5 percent or 10 percent.

        Parameters
        ----------
        px : None or int or StochasticParameter or tuple, optional(default=None)
            The number of pixels to crop away (cut off) on each side of the image.
            Either this or the parameter `percent` may be set, not both at the same
            time.
            If None, then pixel-based cropping will not be used.
            If int, then that exact number of pixels will always be cropped.
            If StochasticParameter, then that parameter will be used for each
              image. Four samples will be drawn per image (top, right, bottom,
              left).
            If a tuple of two ints with values a and b, then each side will be
              cropped by a random amount in the range a <= x <= b.
              x is sampled per image side.
            If a tuple of four entries, then the entries represent top, right,
              bottom, left. Each entry may be a single integer (always crop by exactly
              that value), a tuple of two ints a and b (crop by an
              amount a <= x <= b), a list of ints (crop by a random value that
              is contained in the list) or a StochasticParameter (sample the
              amount to crop from that parameter).

        percent : None or int or float or StochasticParameter or tuple, optional(default=None)
            The number of pixels to crop away (cut off) on each side of the image
            given _in percent_ of the image height/width.
            E.g. if this is set to 0.1, the augmenter will always crop away
            10 percent of the image's height at the top, 10 percent of the width
            on the right, 10 percent of the height at the bottom and 10 percent
            of the width on the left.
            Either this or the parameter `px` may be set, not both at the same
            time.
            If None, then percent-based cropping will not be used.
            If int, then expected to be 0 (no cropping).
            If float, then that percentage will always be cropped away.
            If StochasticParameter, then that parameter will be used for each
              image. Four samples will be drawn per image (top, right, bottom,
              left).
            If a tuple of two floats with values a and b, then each side will be
              cropped by a random percentage in the range a <= x <= b.
              x is sampled per image side.
            If a tuple of four entries, then the entries represent top, right,
              bottom, left. Each entry may be a single float (always crop by exactly
              that percent value), a tuple of two floats a and b (crop by a
              percentage a <= x <= b), a list of floats (crop by a random value that
              is contained in the list) or a StochasticParameter (sample the
              percentage to crop from that parameter).

        keep_input_size : bool, optional(default=True)
            After cropping, the result image has a different height/width than
            the input image. If this parameter is set to True, then the cropped
            image will be resized to the input image's size, i.e. the image size
            is then not changed by the augmenter.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(Crop, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.keep_size = keep_size

        self.all_sides = None
        self.top = None
        self.right = None
        self.bottom = None
        self.left = None
        if px is None and percent is None:
            self.mode = "noop"
        elif px is not None and percent is not None:
            raise Exception("Can only crop by pixels or percent, not both.")
        elif px is not None:
            self.mode = "px"
            if ia.is_single_integer(px):
                assert px >= 0
                #self.top = self.right = self.bottom = self.left = Deterministic(px)
                self.all_sides = Deterministic(px)
            elif isinstance(px, tuple):
                assert len(px) in [2, 4]
                def handle_param(p):
                    if ia.is_single_integer(p):
                        assert p >= 0
                        return Deterministic(p)
                    elif isinstance(p, tuple):
                        assert len(p) == 2
                        assert ia.is_single_integer(p[0])
                        assert ia.is_single_integer(p[1])
                        assert p[0] >= 0
                        assert p[1] >= 0
                        return DiscreteUniform(p[0], p[1])
                    elif isinstance(p, list):
                        assert len(p) > 0
                        assert all([ia.is_single_integer(val) for val in p])
                        assert all([val >= 0 for val in p])
                        return Choice(p)
                    elif isinstance(p, StochasticParameter):
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
            elif isinstance(px, StochasticParameter):
                self.top = self.right = self.bottom = self.left = px
            else:
                raise Exception("Expected int, tuple of 4 ints/tuples/lists/StochasticParameters or StochasticParameter, got type %s." % (type(px),))
        else: # = elif percent is not None:
            self.mode = "percent"
            if ia.is_single_number(percent):
                assert 0 <= percent < 1.0
                #self.top = self.right = self.bottom = self.left = Deterministic(percent)
                self.all_sides = Deterministic(percent)
            elif isinstance(percent, tuple):
                assert len(percent) in [2, 4]
                def handle_param(p):
                    if ia.is_single_number(p):
                        return Deterministic(p)
                    elif isinstance(p, tuple):
                        assert len(p) == 2
                        assert ia.is_single_number(p[0])
                        assert ia.is_single_number(p[1])
                        assert 0 <= p[0] < 1.0
                        assert 0 <= p[1] < 1.0
                        return Uniform(p[0], p[1])
                    elif isinstance(p, list):
                        assert len(p) > 0
                        assert all([ia.is_single_number(val) for val in p])
                        assert all([0 <= val < 1.0 for val in p])
                        return Choice(p)
                    elif isinstance(p, StochasticParameter):
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
            elif isinstance(percent, StochasticParameter):
                self.top = self.right = self.bottom = self.left = percent
            else:
                raise Exception("Expected number, tuple of 4 numbers/tuples/lists/StochasticParameters or StochasticParameter, got type %s." % (type(percent),))


    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            seed = seeds[i]
            height, width = images[i].shape[0:2]
            top, right, bottom, left = self._draw_samples_image(seed, height, width)
            image_cropped = images[i][top:height-bottom, left:width-right, :]
            if self.keep_size:
                image_cropped = ia.imresize_single_image(image_cropped, (height, width))
            result.append(image_cropped)

        if not isinstance(images, list):
            if self.keep_size:
                result = np.array(result, dtype=np.uint8)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            seed = seeds[i]
            height, width = keypoints_on_image.shape[0:2]
            top, right, bottom, left = self._draw_samples_image(seed, height, width)
            shifted = keypoints_on_image.shift(x=-left, y=-top)
            shifted.shape = (height - top - bottom, width - left - right)
            if self.keep_size:
                result.append(shifted.on(keypoints_on_image.shape))
            else:
                result.append(shifted)

        return result

    def _draw_samples_image(self, seed, height, width):
        random_state = ia.new_random_state(seed)

        if self.all_sides is not None:
            samples = self.all_sides.draw_samples((4,), random_state=random_state)
            top, right, bottom, left = samples
        else:
            rs_top = random_state
            rs_right = rs_top
            rs_bottom = rs_top
            rs_left = rs_top
            top = self.top.draw_sample(random_state=rs_top)
            right = self.right.draw_sample(random_state=rs_right)
            bottom = self.bottom.draw_sample(random_state=rs_bottom)
            left = self.left.draw_sample(random_state=rs_left)

        if self.mode == "px":
            # no change necessary for pixel values
            pass
        elif self.mode == "percent":
            # percentage values have to be transformed to pixel values
            top = int(height * top)
            right = int(width * right)
            bottom = int(height * bottom)
            left = int(width * left)
        else:
            raise Exception("Invalid mode")

        remaining_height = height - (top + bottom)
        remaining_width = width - (left + right)
        if remaining_height < 1:
            regain = abs(remaining_height) + 1
            regain_top = regain // 2
            regain_bottom = regain // 2
            if regain_top + regain_bottom < regain:
                regain_top += 1

            if regain_top > top:
                diff = regain_top - top
                regain_top = top
                regain_bottom += diff
            elif regain_bottom > bottom:
                diff = regain_bottom - bottom
                regain_bottom = bottom
                regain_top += diff

            assert regain_top <= top
            assert regain_bottom <= bottom

            top = top - regain_top
            bottom = bottom - regain_bottom

        if remaining_width < 1:
            regain = abs(remaining_width) + 1
            regain_right = regain // 2
            regain_left = regain // 2
            if regain_right + regain_left < regain:
                regain_right += 1

            if regain_right > right:
                diff = regain_right - right
                regain_right = right
                regain_left += diff
            elif regain_left > left:
                diff = regain_left - left
                regain_left = left
                regain_right += diff

            assert regain_right <= right
            assert regain_left <= left

            right = right - regain_right
            left = left - regain_left

        assert top >= 0 and right >= 0 and bottom >= 0 and left >= 0
        assert top + bottom < height
        assert right + left < width

        return top, right, bottom, left

    def get_parameters(self):
        return [self.top, self.right, self.bottom, self.left]


class Fliplr(Augmenter):
    """Flip/mirror input images horizontally."""

    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        """Create a new Fliplr instance.

        Example:
            aug = iaa.Fliplr(0.5)
        would horizontally flip/mirror 50 percent of all input images.

        Example:
            aug = iaa.Fliplr(1.0)
        would horizontally flip/mirror all input images.

        Parameters
        ----------
        p : int or float or StochasticParameter, optional(default=0)
            Probability of each image being flipped.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(Fliplr, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be int or float or StochasticParameter, got %s." % (type(p),))

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            if samples[i] == 1:
                images[i] = np.fliplr(images[i])
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                width = keypoints_on_image.shape[1]
                for keypoint in keypoints_on_image.keypoints:
                    keypoint.x = (width - 1) - keypoint.x
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]


class Flipud(Augmenter):
    """Flip/mirror input images vertically."""

    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        """Create a new Flipud instance.

        Example:
            aug = iaa.Flipud(0.5)
        would vertically flip/mirror 50 percent of all input images.

        Example:
            aug = iaa.Flipud(1.0)
        would vertically flip/mirror all input images.

        Parameters
        ----------
        p : int or float or StochasticParameter, optional(default=0)
            Probability of each image being flipped.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(Flipud, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be int or float or StochasticParameter, got %s." % (type(p),))

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            if samples[i] == 1:
                images[i] = np.flipud(images[i])
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                height = keypoints_on_image.shape[0]
                for keypoint in keypoints_on_image.keypoints:
                    keypoint.y = (height - 1) - keypoint.y
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]

# TODO tests
class Superpixels(Augmenter):
    """Completely or partially transform images to their superpixel representation."""

    def __init__(self, p_replace=0, n_segments=100, max_size=128, interpolation="linear", name=None, deterministic=False, random_state=None):
        """Create a new Superpixels augmenter instance.

        This implementation uses skimage's version of the SLIC algorithm.

        Example:
            aug = iaa.Superpixels(p_replace=1.0, n_segments=64)
        generates ~64 superpixels per image and replaces all of them with
        their average color (standard superpixel image).

        Example:
            aug = iaa.Superpixels(p_replace=0.5, n_segments=64)
        generates always ~64 superpixels per image and replaces half of them
        with their average color, while the other half are left unchanged (i.e.
        they still show the input image's content).

        Example:
            aug = iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128))
        generates between ~16 and ~128 superpixels per image and replaces
        25 to 100 percent of them with their average color.

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
            downscaled so that the longest side matches max_size.
            Though, the final output (superpixel) image has the same size as the
            input image.
            This is done to speed up the superpixel algorithm.
            Use None to apply no downscaling.

        interpolation : int or string, optional(default="linear")
            Interpolation method to use during downscaling when max_size is
            exceeded. Valid methods are the same as in ia.imresize_single_image().

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(Superpixels, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p_replace):
            self.p_replace = Binomial(p_replace)
        elif ia.is_iterable(p_replace):
            assert len(p_replace) == 2
            assert p_replace[0] < p_replace[1]
            assert 0 <= p_replace[0] <= 1.0
            assert 0 <= p_replace[1] <= 1.0
            self.p_replace = p_replace = Binomial(Uniform(p_replace[0], p_replace[1]))
        elif isinstance(p_replace, StochasticParameter):
            self.p_replace = p_replace
        else:
            raise Exception("Expected p_replace to be float, int, list/tuple of floats/ints or StochasticParameter, got %s." % (type(p_replace),))

        if ia.is_single_integer(n_segments):
            self.n_segments = Deterministic(n_segments)
        elif ia.is_iterable(n_segments):
            assert len(n_segments) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(n_segments)),)
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

# TODO tests
# Note: Not clear whether this class will be kept (for anything aside from grayscale)
# other colorspaces dont really make sense and they also might not work correctly
# due to having no clearly limited range (like 0-255 or 0-1)
class ChangeColorspace(Augmenter):
    """Augmenter to change the colorspace of images."""

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
        """Create a ChangeColorspace instance.

        NOTE: This augmenter is not tested. Some colorspaces might work, others
        might not.

        Parameters
        ----------
        to_colorspace : string or iterable or StochasticParameter
            The target colorspace.
            Allowed are: RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv.
            If a string, it must be among the allowed colorspaces.
            If an iterable, it is expected to be a list of strings, each one
              being an allowed colorspace. A random element from the list
              will be chosen per image.
            If a StochasticParameter, it is expected to return string. A new
              sample will be drawn per image.

        from_colorspace : string, optional(default="RGB")
            The source colorspace (of the input images).
            Allowed are: RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv.

        alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=1.0)
            The alpha value of the new colorspace when overlayed over the
            old one. A value close to 1.0 means, that mostly the new
            colorspace is visible. A value close to 0.0 means, that mostlye the
            old image is visible. Use a tuple (a, b) to use a random value
            x with a <= x <= b as the alpha value per image.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(ChangeColorspace, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(alpha):
            self.alpha = Deterministic(alpha)
        elif ia.is_iterable(alpha):
            assert len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(alpha)),)
            self.alpha = Uniform(alpha[0], alpha[1])
        elif isinstance(p, StochasticParameter):
            self.alpha = alpha
        else:
            raise Exception("Expected alpha to be int or float or tuple/list of ints/floats or StochasticParameter, got %s." % (type(alpha),))

        if ia.is_string(to_colorspace):
            assert to_colorspace in ChangeColorspace.COLORSPACES
            self.to_colorspace = Deterministic(to_colorspace)
        elif ia.is_iterable(to_colorspace):
            assert all([ia.is_string(colorspace) for colorspace in to_colorspace])
            assert all([(colorspace in ChangeColorspace.COLORSPACES) for colorspace in to_colorspace])
            self.to_colorspace = Choice(to_colorspace)
        elif isinstance(to_colorspace, StochasticParameter):
            self.to_colorspace = to_colorspace
        else:
            raise Exception("Expected to_colorspace to be string, list of strings or StochasticParameter, got %s." % (type(to_colorspace),))

        self.from_colorspace = from_colorspace
        assert self.from_colorspace in ChangeColorspace.COLORSPACES
        assert from_colorspace != ChangeColorspace.GRAY

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

            assert 0.0 <= alpha <= 1.0
            assert to_colorspace in ChangeColorspace.COLORSPACES

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
def Grayscale(alpha=0, from_colorspace="RGB", name=None, deterministic=False, random_state=None):
    """Augmenter to convert images to their grayscale versions.

    NOTE: Number of output channels is still 3, i.e. this augmenter just
    "removes" color.

    Example:
        aug = iaa.Grayscale(alpha=1.0)
    creates an augmenter that turns images to their grayscale versions.

    Example:
        aug = iaa.Grayscale(alpha=(0.0, 1.0))
    creates an augmenter that turns images to their grayscale versions with
    an alpha value in the range 0 <= alpha <= 1. An alpha value of 0.5 would
    mean, that the output image is 50 percent of the input image and 50
    percent of the grayscale image (i.e. 50 percent of color removed).

    Parameters
    ----------
    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=0)
        The alpha value of the grayscale image when overlayed over the
        old image. A value close to 1.0 means, that mostly the new grayscale
        image is visible. A value close to 0.0 means, that mostly the
        old image is visible. Use a tuple (a, b) to use a random value
        x with a <= x <= b as the alpha value per image.

    from_colorspace : string, optional(default="RGB")
        The source colorspace (of the input images).
        Allowed are: RGB, BGR, GRAY, CIE, YCrCb, HSV, HLS, Lab, Luv.

    name : string, optional(default=None)
        See Augmenter.__init__()

    deterministic : bool, optional(default=False)
        See Augmenter.__init__()

    random_state : int or np.random.RandomState or None, optional(default=None)
        See Augmenter.__init__()
    """
    return ChangeColorspace(to_colorspace=ChangeColorspace.GRAY, alpha=alpha, from_colorspace=from_colorspace, name=name, deterministic=deterministic, random_state=random_state)


class GaussianBlur(Augmenter):
    """Blur an image using a gaussian kernel."""

    def __init__(self, sigma=0, name=None, deterministic=False, random_state=None):
        """Creates a new GaussianBlur instance.

        Example:
            aug = iaa.GaussianBlur(sigma=1.5)
        blurs all images using a gaussian kernel with standard deviation 1.5.

        Example:
            aug = iaa.GaussianBlur(sigma=(0.0, 3.0))
        blurs images using a gaussian kernel with a random standard deviation
        from the range 0.0 <= x <= 3.0. The value is sampled per image.

        Parameters
        ----------
        sigma : float or tuple of two floats or StochasticParameter
            Standard deviation of the gaussian kernel.
            Values in the range 0.0 (no blur) to 3.0 (strong blur) are common.
            If a single float, that value will always be used as the standard
              deviation.
            If a tuple (a, b), then a random value from the range a <= x <= b
              will be picked per image.
            If a StochasticParameter, then N samples will be drawn from
              that parameter per N input images.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(GaussianBlur, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(sigma):
            self.sigma = Deterministic(sigma)
        elif ia.is_iterable(sigma):
            assert len(sigma) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(sigma)),)
            self.sigma = Uniform(sigma[0], sigma[1])
        elif isinstance(sigma, StochasticParameter):
            self.sigma = sigma
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(sigma),))

        self.eps = 0.001 # epsilon value to estimate whether sigma is above 0

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.sigma.draw_samples((nb_images,), random_state=random_state)
        for i in sm.xrange(nb_images):
            nb_channels = images[i].shape[2]
            sig = samples[i]
            if sig > 0 + self.eps:
                # note that while gaussian_filter can be applied to all channels
                # at the same time, that should not be done here, because then
                # the blurring would also happen across channels (e.g. red
                # values might be mixed with blue values in RGB)
                for channel in sm.xrange(nb_channels):
                    result[i][:, :, channel] = ndimage.gaussian_filter(result[i][:, :, channel], sig)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.sigma]

# TODO tests
class Convolve(Augmenter):
    """Apply a Convolution to input images."""

    def __init__(self, matrix=None, name=None, deterministic=False, random_state=None):
        """Instantiate a new Convolve augmenter.

        Example:
            matrix = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])
            aug = iaa.Convolve(matrix=matrix)
        convolves all input images with the kernel shown in the `matrix`
        variable.

        Example:
            def gen_matrix(image, nb_channels, random_state):
                matrix_A = np.array([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]])
                matrix_B = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
                if image.shape[0] % 2 == 0:
                    return matrix_A
                else:
                    return matrix_B
            aug = iaa.Convolve(matrix=gen_matrix)
        convolves images that have an even height with matrix A and images
        with an odd height with matrix B.

        Parameters
        ----------
        matrix : None or (H, W) ndarray or StochasticParameter or callable, optional(default=None)
            The weight matrix of the convolution kernel to apply.
            If None, the input images will not be changed.
            If a numpy array, that array will be used for all images and
              channels as the kernel.
            If a stochastic parameter, C new matrices will be generated
              via param.draw_samples(C) for each image, where C is the number
              of channels.
            If a callable, the parameter will be called for each image
              via param(image, C, random_state). The function must return C
              matrices, one per channel. It may return None, then that channel
              will not be changed.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(Convolve, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if matrix is None:
            self.matrix = None #np.array([[1]], dtype=np.float32)
            self.matrix_type = "None"
        elif ia.is_np_array(matrix):
            assert len(matrix.shape) == 2, "Expected convolution matrix to have 2 axis, got %d (shape %s)." % (len(matrix.shape), matrix.shape)
            self.matrix = matrix
            self.matrix_type = "constant"
        elif isinstance(matrix, StochasticParameter):
            self.matrix = matrix
            self.matrix_type = "stochastic"
        elif isinstance(matrix, types.FunctionType):
            self.matrix = matrix
            self.matrix_type = "function"
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(sigma),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        for i in sm.xrange(nb_images):
            height, width, nb_channels = images[i].shape
            if self.matrix_type == "None":
                matrices = [None] * nb_channels
            elif self.matrix_type == "constant":
                matrices = [self.matrix] * nb_channels
            elif self.matrix_type == "stochastic":
                matrices = self.matrix.draw_samples((nb_channels), random_state=random_state)
            elif self.matrix_type == "function":
                matrices = self.matrix(images[i], nb_channels, random_state)
                assert len(matrices) == nb_channels
            else:
                raise Exception("Invalid matrix type")

            for channel in sm.xrange(nb_channels):
                if matrices[channel] is not None:
                    # ndimage.convolve caused problems here
                    result[i][..., channel] = cv2.filter2D(result[i][..., channel], -1, matrices[channel])
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        # TODO this can fail for some matrices, e.g. [[0, 0, 1]]
        return keypoints_on_images

    def get_parameters(self):
        return [self.matrix, self.matrix_type]

# TODO tests
def Sharpen(alpha=0, lightness=1, name=None, deterministic=False, random_state=None):
    """Creates an augmenter that sharpens images.

    Example:
        aug = Sharpen(alpha=(0.0, 1.0))
    sharpens input images and overlays the sharpened image by a variable
    amount over the old image.

    Example:
        aug = Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
    sharpens input images with a variable lightness in the range
    0.75 <= x <= 2.0 and with a variable alpha.

    Parameters
    ----------
    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=0)
        Visibility of the sharpened image. At 0, only the original image is
        visible, at 1.0 only its sharpened version is visible.
        If an int or float, exactly that value will be used.
        If a tuple (a, b), a random value from the range a <= x <= b will be
          sampled per image.
        If a StochasticParameter, a value will be sampled from the parameter
          per image.

    lightness : int or float or tuple of two ints/floats or StochasticParameter, optional(default=1)
        Parameter that controls the lightness/brightness of the sharped image.
        Sane values are somewhere in the range (0.5, 2).
        The value 0 results in an edge map. Values higher than 1 create bright
        images. Default value is 1.
        If an int or float, exactly that value will be used.
        If a tuple (a, b), a random value from the range a <= x <= b will be
          sampled per image.
        If a StochasticParameter, a value will be sampled from the parameter
          per image.

    name : string, optional(default=None)
        See Augmenter.__init__()

    deterministic : bool, optional(default=False)
        See Augmenter.__init__()

    random_state : int or np.random.RandomState or None, optional(default=None)
        See Augmenter.__init__()
    """
    if ia.is_single_number(alpha):
        alpha_param = Deterministic(alpha)
    elif ia.is_iterable(alpha):
        assert len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(alpha)),)
        alpha_param = Uniform(alpha[0], alpha[1])
    elif isinstance(alpha, StochasticParameter):
        alpha_param = alpha
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

    if ia.is_single_number(lightness):
        lightness_param = Deterministic(lightness)
    elif ia.is_iterable(lightness):
        assert len(lightness) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(lightness)),)
        lightness_param = Uniform(lightness[0], lightness[1])
    elif isinstance(lightness, StochasticParameter):
        lightness_param = lightness
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(lightness),))

    def create_matrices(image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        assert 0 <= alpha_sample <= 1.0
        lightness_sample = lightness_param.draw_sample(random_state=random_state_func)
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [-1, -1, -1],
            [-1, 8+lightness_sample, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        matrix = (1-alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return [matrix] * nb_channels

    return Convolve(create_matrices, name=name, deterministic=deterministic, random_state=random_state)

# TODO tests
def Emboss(alpha=0, strength=1, name=None, deterministic=False, random_state=None):
    """Creates an augmenter that embosses an image.
    The embossed version pronounces highlights and shadows,
    letting the image look as if it was recreated on a metal plate ("embossed").

    Example:
        aug = Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
    embosses an image with a variable strength in the range 0.5 <= x <= 1.5
    and overlays the result with a variable alpha in the range 0.0 <= a <= 1.0
    over the old image.

    Parameters
    ----------
    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=0)
        Visibility of the sharpened image. At 0, only the original image is
        visible, at 1.0 only its sharpened version is visible.
        If an int or float, exactly that value will be used.
        If a tuple (a, b), a random value from the range a <= x <= b will be
          sampled per image.
        If a StochasticParameter, a value will be sampled from the parameter
          per image.

    strength : int or float or tuple of two ints/floats or StochasticParameter, optional(default=1)
        Parameter that controls the strength of the embossing.
        Sane values are somewhere in the range (0, 2) with 1 being the standard
        embossing effect. Default value is 1.
        If an int or float, exactly that value will be used.
        If a tuple (a, b), a random value from the range a <= x <= b will be
          sampled per image.
        If a StochasticParameter, a value will be sampled from the parameter
          per image.

    name : string, optional(default=None)
        See Augmenter.__init__()

    deterministic : bool, optional(default=False)
        See Augmenter.__init__()

    random_state : int or np.random.RandomState or None, optional(default=None)
        See Augmenter.__init__()
    """

    if ia.is_single_number(alpha):
        alpha_param = Deterministic(alpha)
    elif ia.is_iterable(alpha):
        assert len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(alpha)),)
        alpha_param = Uniform(alpha[0], alpha[1])
    elif isinstance(alpha, StochasticParameter):
        alpha_param = alpha
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

    if ia.is_single_number(strength):
        strength_param = Deterministic(strength)
    elif ia.is_iterable(strength):
        assert len(strength) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(strength)),)
        strength_param = Uniform(strength[0], strength[1])
    elif isinstance(strength, StochasticParameter):
        strength_param = strength
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(strength),))

    def create_matrices(image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        assert 0 <= alpha_sample <= 1.0
        strength_sample = strength_param.draw_sample(random_state=random_state_func)
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [-1-strength_sample, 0-strength_sample, 0],
            [0-strength_sample, 1, 0+strength_sample],
            [0, 0+strength_sample, 1+strength_sample]
        ], dtype=np.float32)
        matrix = (1-alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return [matrix] * nb_channels

    return Convolve(create_matrices, name=name, deterministic=deterministic, random_state=random_state)

# TODO tests
def EdgeDetect(alpha=0, name=None, deterministic=False, random_state=None):
    """Creates an augmenter that pronounces all edges in images.

    Example:
        aug = EdgeDetect(alpha=(0.0, 1.0))
    detects edges in an image  and overlays the result with a variable alpha
    in the range 0.0 <= a <= 1.0 over the old image.

    Parameters
    ----------
    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=0)
        Visibility of the sharpened image. At 0, only the original image is
        visible, at 1.0 only its sharpened version is visible.
        If an int or float, exactly that value will be used.
        If a tuple (a, b), a random value from the range a <= x <= b will be
          sampled per image.
        If a StochasticParameter, a value will be sampled from the parameter
          per image.

    name : string, optional(default=None)
        See Augmenter.__init__()

    deterministic : bool, optional(default=False)
        See Augmenter.__init__()

    random_state : int or np.random.RandomState or None, optional(default=None)
        See Augmenter.__init__()
    """
    if ia.is_single_number(alpha):
        alpha_param = Deterministic(alpha)
    elif ia.is_iterable(alpha):
        assert len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(alpha)),)
        alpha_param = Uniform(alpha[0], alpha[1])
    elif isinstance(alpha, StochasticParameter):
        alpha_param = alpha
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

    def create_matrices(image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        assert 0 <= alpha_sample <= 1.0
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        matrix = (1-alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return [matrix] * nb_channels

    return Convolve(create_matrices, name=name, deterministic=deterministic, random_state=random_state)

# TODO tests

def DirectedEdgeDetect(alpha=0, direction=(0.0, 1.0), name=None, deterministic=False, random_state=None):
    """Creates an augmenter that detects edges that have certain directions.

    Example:
        aug = EdgeDetect(alpha=1.0, direction=0)
    turns input images into edge images in which edges are detected from
    top side of the image (i.e. the top sides of horizontal edges are
    added to the output).

    Example:
        aug = EdgeDetect(alpha=1.0, direction=90/360)
    same as before, but detecting edges from the right (right side of each
    vertical edge).

    Example:
        aug = EdgeDetect(alpha=1.0, direction=(0.0, 1.0))
    same as before, but detecting edges from a variable direction (anything
    between 0 and 1.0, i.e. 0 degrees and 360 degrees, starting from the
    top and moving clockwise).

    Example:
        aug = EdgeDetect(alpha=(0.0, 0.3), direction=0)
    generates edge images (edges detected from the top) and overlays them
    with the input images by a variable amount between 0 and 30 percent
    (e.g. for 0.3 then 0.7*old_image + 0.3*edge_image).

    Parameters
    ----------
    alpha : int or float or tuple of two ints/floats or StochasticParameter, optional(default=0)
        Visibility of the sharpened image. At 0, only the original image is
        visible, at 1.0 only its sharpened version is visible.
        If an int or float, exactly that value will be used.
        If a tuple (a, b), a random value from the range a <= x <= b will be
          sampled per image.
        If a StochasticParameter, a value will be sampled from the parameter
          per image.

    direction : int or float or tuple of two ints/floats or StochasticParameter, optional(default=(0.0, 1.0))
        Angle of edges to pronounce, where 0 represents 0 degrees and 1.0
        represents 360 degrees (both clockwise, starting at the top).
        Default value is (0.0, 1.0), i.e. pick a random angle per image.
        If an int or float, exactly that value will be used.
        If a tuple (a, b), a random value from the range a <= x <= b will be
          sampled per image.
        If a StochasticParameter, a value will be sampled from the parameter
          per image.

    name : string, optional(default=None)
        See Augmenter.__init__()

    deterministic : bool, optional(default=False)
        See Augmenter.__init__()

    random_state : int or np.random.RandomState or None, optional(default=None)
        See Augmenter.__init__()
    """
    if ia.is_single_number(alpha):
        alpha_param = Deterministic(alpha)
    elif ia.is_iterable(alpha):
        assert len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(alpha)),)
        alpha_param = Uniform(alpha[0], alpha[1])
    elif isinstance(alpha, StochasticParameter):
        alpha_param = alpha
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

    if ia.is_single_number(direction):
        direction_param = Deterministic(direction)
    elif ia.is_iterable(direction):
        assert len(direction) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(direction)),)
        direction_param = Uniform(direction[0], direction[1])
    elif isinstance(direction, StochasticParameter):
        direction_param = direction
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(direction),))

    def create_matrices(image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        assert 0 <= alpha_sample <= 1.0
        direction_sample = direction_param.draw_sample(random_state=random_state_func)

        deg = int(direction_sample * 360) % 360
        rad = np.deg2rad(deg)
        x = np.cos(rad - 0.5*np.pi)
        y = np.sin(rad - 0.5*np.pi)
        #x = (deg % 90) / 90 if 0 <= deg <= 180 else -(deg % 90) / 90
        #y = (-1) + (deg % 90) / 90 if 90 < deg < 270 else 1 - (deg % 90) / 90
        direction_vector = np.array([x, y])

        #print("direction_vector", direction_vector)

        vertical_vector = np.array([0, 1])

        matrix_effect = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if (x, y) != (0, 0):
                    cell_vector = np.array([x, y])
                    #deg_cell = angle_between_vectors(vertical_vector, vec_cell)
                    distance_deg = np.rad2deg(ia.angle_between_vectors(cell_vector, direction_vector))
                    distance = distance_deg / 180
                    similarity = (1 - distance)**4
                    matrix_effect[y+1, x+1] = similarity
                    #print("cell", y, x, "distance_deg", distance_deg, "distance", distance, "similarity", similarity)
        matrix_effect = matrix_effect / np.sum(matrix_effect)
        matrix_effect = matrix_effect * (-1)
        matrix_effect[1, 1] = 1
        #for y in [0, 1, 2]:
        #    vals = []
        #    for x in [0, 1, 2]:
        #        vals.append("%.2f" % (matrix_effect[y, x],))
        #    print(" ".join(vals))
        #print("matrix_effect", matrix_effect)

        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)

        matrix = (1-alpha_sample) * matrix_nochange + alpha_sample * matrix_effect

        return [matrix] * nb_channels

    return Convolve(create_matrices, name=name, deterministic=deterministic, random_state=random_state)

# TODO tests
class Add(Augmenter):
    """Add a value to all pixels in an image."""

    def __init__(self, value=0, per_channel=False, name=None,
                 deterministic=False, random_state=None):
        """Creates an instance of the Add augmenter.

        Example:
            aug = iaa.Add(10)
        always adds a value of 10 to all pixels in the image.

        Example:
            aug = iaa.Add((-10, 10))
        adds a value from the discrete range [-10 .. 10] to all pixels of
        the input images. The exact value is sampled per image.

        Example:
            aug = iaa.Add((-10, 10), per_channel=True)
        adds a value from the discrete range [-10 .. 10] to all pixels of
        the input images. The exact value is sampled per image AND channel,
        i.e. to a red-channel it might add 5 while subtracting 7 from the
        blue channel of the same image.

        Example:
            aug = iaa.Add((-10, 10), per_channel=0.5)
        same as previous example, but the per_channel feature is only active
        for 50 percent of all images.

        Parameters
        ----------
        value : int or iterable of two ints or StochasticParameter, optional(default=0)
            Value to add to all pixels.
            If an int, then that value will be used for all images.
            If a tuple (a, b), then a value from the discrete range [a .. b]
              will be used.
            If a StochasticParameter, then a value will be sampled per image
              from that parameter.

        per_channel : bool, optional(default=False)
            Whether to use the same value for all channels (False)
            or to sample a new value for each channel (True).
            If this value is a float p, then for p percent of all images
            per_channel will be treated as True, otherwise as False.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(Add, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_integer(value):
            assert -255 <= value <= 255, "Expected value to have range [-255, 255], got value %d." % (value,)
            self.value = Deterministic(value)
        elif ia.is_iterable(value):
            assert len(value) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(value),)
            self.value = DiscreteUniform(value[0], value[1])
        elif isinstance(value, StochasticParameter):
            self.value = value
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(value),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            assert 0 <= per_channel <= 1.0
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.int32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.value.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    assert -255 <= sample <= 255
                    image[..., c] += sample
                np.clip(image, 0, 255, out=image)
                result[i] = image.astype(np.uint8)
            else:
                sample = self.value.draw_sample(random_state=rs_image)
                assert -255 <= sample <= 255
                image += sample
                np.clip(image, 0, 255, out=image)
                result[i] = image.astype(np.uint8)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.value]

# TODO tests
class AddElementwise(Augmenter):
    """Add values to the pixels of images with possibly different values
    for neighbouring pixels.

    While the Add Augmenter adds a constant value per image, this one can
    add different values (sampled per pixel).
    """

    def __init__(self, value=0, per_channel=False, name=None, deterministic=False, random_state=None):
        """Create a new AddElementwise instance.

        Example:
            aug = iaa.AddElementwise(10)
        always adds a value of 10 to all pixels in the image.

        Example:
            aug = iaa.AddElementwise((-10, 10))
        samples per pixel a value from the discrete range [-10 .. 10] and
        adds that value to the pixel.

        Example:
            aug = iaa.AddElementwise((-10, 10), per_channel=True)
        samples per pixel _and channel_ a value from the discrete
        range [-10 .. 10] ands adds it to the pixel's value. Therefore,
        added values may differ between channels of the same pixel.

        Example:
            aug = iaa.AddElementwise((-10, 10), per_channel=0.5)
        same as previous example, but the per_channel feature is only active
        for 50 percent of all images.

        Parameters
        ----------
        value : int or iterable of two ints or StochasticParameter, optional(default=0)
            Value to add to the pixels.
            If an int, then that value will be used for all images.
            If a tuple (a, b), then values from the discrete range [a .. b]
              will be sampled.
            If a StochasticParameter, then values will be sampled per pixel
              (and possibly channel) from that parameter.

        per_channel : bool, optional(default=False)
            Whether to use the same value for all channels (False)
            or to sample a new value for each channel (True).
            If this value is a float p, then for p percent of all images
            per_channel will be treated as True, otherwise as False.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(AddElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_integer(value):
            assert -255 <= value <= 255, "Expected value to have range [-255, 255], got value %d." % (value,)
            self.value = Deterministic(value)
        elif ia.is_iterable(value):
            assert len(value) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(value),)
            self.value = DiscreteUniform(value[0], value[1])
        elif isinstance(value, StochasticParameter):
            self.value = value
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(value),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            assert 0 <= per_channel <= 1.0
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            seed = seeds[i]
            image = images[i].astype(np.int32)
            height, width, nb_channels = image.shape
            rs_image = ia.new_random_state(seed)
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                samples = self.value.draw_samples((height, width, nb_channels), random_state=rs_image)
            else:
                samples = self.value.draw_samples((height, width, 1), random_state=rs_image)
                samples = np.tile(samples, (1, 1, nb_channels))
            after_add = image + samples
            np.clip(after_add, 0, 255, out=after_add)
            result[i] = after_add.astype(np.uint8)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.value]

def AdditiveGaussianNoise(loc=0, scale=0, per_channel=False, name=None, deterministic=False, random_state=None):
    """Add gaussian noise (aka white noise) to images.

    Example:
        aug = iaa.GaussianNoise(scale=0.1*255)
    adds gaussian noise from the distribution N(0, 0.1*255) to images.

    Example:
        aug = iaa.GaussianNoise(scale=(0, 0.1*255))
    adds gaussian noise from the distribution N(0, s) to images,
    where s is sampled per image from the range 0 <= s <= 0.1*255.

    Example:
        aug = iaa.GaussianNoise(scale=0.1*255, per_channel=True)
    adds gaussian noise from the distribution N(0, 0.1*255) to images,
    where the noise value is different per pixel _and_ channel (e.g. a
    different one for red, green and blue channels for the same pixel).

    Example:
        aug = iaa.GaussianNoise(scale=0.1*255, per_channel=0.5)
    adds gaussian noise from the distribution N(0, 0.1*255) to images,
    where the noise value is sometimes (50 percent of all cases) the same
    per pixel for all channels and sometimes different (other 50 percent).

    Parameters
    ----------
    loc : int or float or tupel of two ints/floats or StochasticParameter, optional(default=0)
        Mean of the normal distribution that generates the noise.
        If an int or float, exactly that value will be used.
        If a tuple (a, b), a random value from the range a <= x <= b will be
          sampled per image.
        If a StochasticParameter, a value will be sampled from the parameter
          per image.

    scale : int or float or tupel of two ints/floats or StochasticParameter, optional(default=0)
        Standard deviation of the normal distribution that generates the noise.
        If this value gets too close to zero, the image will not be changed.
        If an int or float, exactly that value will be used.
        If a tuple (a, b), a random value from the range a <= x <= b will be
          sampled per image.
        If a StochasticParameter, a value will be sampled from the parameter
          per image.

    per_channel : bool or float, optional(default=False)
        Whether to use the same noise value per pixel for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images per_channel
        will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See Augmenter.__init__()

    deterministic : bool, optional(default=False)
        See Augmenter.__init__()

    random_state : int or np.random.RandomState or None, optional(default=None)
        See Augmenter.__init__()
    """
    if ia.is_single_number(loc):
        loc2 = Deterministic(loc)
    elif ia.is_iterable(loc):
        assert len(loc) == 2, "Expected tuple/list with 2 entries for argument 'loc', got %d entries." % (str(len(scale)),)
        loc2 = Uniform(loc[0], loc[1])
    elif isinstance(loc, StochasticParameter):
        loc2 = loc
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter for argument 'loc'. Got %s." % (type(loc),))

    if ia.is_single_number(scale):
        scale2 = Deterministic(scale)
    elif ia.is_iterable(scale):
        assert len(scale) == 2, "Expected tuple/list with 2 entries for argument 'scale', got %d entries." % (str(len(scale)),)
        scale2 = Uniform(scale[0], scale[1])
    elif isinstance(scale, StochasticParameter):
        scale2 = scale
    else:
        raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter for argument 'scale'. Got %s." % (type(scale),))

    return AddElementwise(Normal(loc=loc2, scale=scale2), per_channel=per_channel, name=name, deterministic=deterministic, random_state=random_state)

# TODO
#class MultiplicativeGaussianNoise(Augmenter):
#    pass

# TODO
#class ReplacingGaussianNoise(Augmenter):
#    pass


class Multiply(Augmenter):
    """Multiply all pixels in an image with a specific value.

    This augmenter can be used to make images lighter or darker."""

    def __init__(self, mul=1.0, per_channel=False, name=None,
                 deterministic=False, random_state=None):
        """Create a new Multiply instance.

        Example:
            aug = iaa.Multiply(2.0)
        would multiply all images by a factor of 2, making the images
        significantly brighter.

        Example:
            aug = iaa.Multiply((0.5, 1.5))
        would multiply images by a random value from the range 0.5 <= x <= 1.5,
        making some images darker and others brighter.

        Parameters
        ----------
        mul : float or tuple of two floats or StochasticParameter, optional(default=1.0)
            The value by which to multiply the pixel values in the image.
            If a float, then that value will always be used.
            If a tuple (a, b), then a value from the range a <= x <= b will
              be sampled per image and used for all pixels.
            If a StochasticParameter, then that parameter will be used to
              sample a new value per image.

        per_channel : bool, optional(default=False)
            Whether to use the same multiplier per pixel for all channels (False)
            or to sample a new value for each channel (True).
            If this value is a float p, then for p percent of all images per_channel
            will be treated as True, otherwise as False.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(Multiply, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(mul):
            assert mul >= 0.0, "Expected multiplier to have range [0, inf), got value %.4f." % (mul,)
            self.mul = Deterministic(mul)
        elif ia.is_iterable(mul):
            assert len(mul) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(mul),)
            self.mul = Uniform(mul[0], mul[1])
        elif isinstance(mul, StochasticParameter):
            self.mul = mul
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(mul),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            assert 0 <= per_channel <= 1.0
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.float32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.mul.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    assert sample >= 0
                    image[..., c] *= sample
                np.clip(image, 0, 255, out=image)
                result[i] = image.astype(np.uint8)
            else:
                sample = self.mul.draw_sample(random_state=rs_image)
                assert sample >= 0
                image *= sample
                np.clip(image, 0, 255, out=image)
                result[i] = image.astype(np.uint8)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul]


# TODO tests
class MultiplyElementwise(Augmenter):
    """Multiply values of pixels with possibly different values
    for neighbouring pixels.

    While the Multiply Augmenter uses a constant multiplier per image,
    this one can use different multipliers per pixel."""

    def __init__(self, mul=1.0, per_channel=False, name=None, deterministic=False, random_state=None):
        """Create a new MultiplyElementwise instance.

        Example:
            aug = iaa.MultiplyElementwise(2.0)
        multiply all images by a factor of 2.0, making them significantly
        bighter.

        Example:
            aug = iaa.MultiplyElementwise((0.5, 1.5))
        samples per pixel a value from the range 0.5 <= x <= 1.5 and
        multiplies the pixel with that value.

        Example:
            aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=True)
        samples per pixel _and channel_ a value from the range
        0.5 <= x <= 1.5 ands multiplies the pixel by that value. Therefore,
        added multipliers may differ between channels of the same pixel.

        Example:
            aug = iaa.AddElementwise((0.5, 1.5), per_channel=0.5)
        same as previous example, but the per_channel feature is only active
        for 50 percent of all images.

        Parameters
        ----------
        mul : float or iterable of two floats or StochasticParameter, optional(default=1.0)
            The value by which to multiply the pixel values in the image.
            If a float, then that value will always be used.
            If a tuple (a, b), then a value from the range a <= x <= b will
              be sampled per image and pixel.
            If a StochasticParameter, then that parameter will be used to
              sample a new value per image and pixel.

        per_channel : bool, optional(default=False)
            Whether to use the same value for all channels (False)
            or to sample a new value for each channel (True).
            If this value is a float p, then for p percent of all images
            per_channel will be treated as True, otherwise as False.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(MultiplyElementwise, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(mul):
            assert mul >= 0.0, "Expected multiplier to have range [0, inf), got value %.4f." % (mul,)
            self.mul = Deterministic(mul)
        elif ia.is_iterable(mul):
            assert len(mul) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(mul)),)
            self.mul = Uniform(mul[0], mul[1])
        elif isinstance(mul, StochasticParameter):
            self.mul = mul
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(mul),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            assert 0 <= per_channel <= 1.0
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            seed = seeds[i]
            image = images[i].astype(np.float32)
            height, width, nb_channels = image.shape
            rs_image = ia.new_random_state(seed)
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                samples = self.mul.draw_samples((height, width, nb_channels), random_state=rs_image)
            else:
                samples = self.mul.draw_samples((height, width, 1), random_state=rs_image)
                samples = np.tile(samples, (1, 1, nb_channels))
            after_multiply = image * samples
            np.clip(after_multiply, 0, 255, out=after_multiply)
            result[i] = after_multiply.astype(np.uint8)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul]

def Dropout(p=0, per_channel=False, name=None, deterministic=False,
            random_state=None):
    """Create an augmenter to set a certain fraction of pixels in images to zero.

    Examples:
        aug = iaa.Dropout(0.02)
    drops 2 percent of all pixels.

    Examples:
        aug = iaa.Dropout((0.0, 0.05))
    drops in each image a random fraction of all pixels, where the fraction
    is in the range 0.0 <= x <= 0.05.

    Examples:
        aug = iaa.Dropout(0.02, per_channel=True)
    drops 2 percent of all pixels in a channel-wise fashion, i.e. it is unlikely
    for any pixel to have all channels set to zero (black pixels).

    Examples:
        aug = iaa.Dropout(0.02, per_channel=0.5)
    same as previous example, but the per_channel feature is only active
    for 50 percent of all images.

    Parameters
    ----------
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        The probability of any pixel being dropped (i.e. set to zero).
        If a float, then that value will be used for all pixels. A value of 1.0
          would mean, that all pixels will be dropped. A value of 0.0 would
          lead to no pixels being dropped.
        If a tuple (a, b), then a value p will be sampled from the
          range a <= p <= b per image and be used as the pixel's dropout
          probability.
        If a StochasticParameter, then this parameter will be used to determine
          per pixel whether it should be dropped (sampled value of 0)
          or shouldn't (sampled value of 1).

    per_channel : bool, optional(default=False)
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float p, then for p percent of all images
        per_channel will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See Augmenter.__init__()

    deterministic : bool, optional(default=False)
        See Augmenter.__init__()

    random_state : int or np.random.RandomState or None, optional(default=None)
        See Augmenter.__init__()
    """
    if ia.is_single_number(p):
        p2 = Binomial(1 - p)
    elif ia.is_iterable(p):
        assert len(p) == 2
        assert p[0] < p[1]
        assert 0 <= p[0] <= 1.0
        assert 0 <= p[1] <= 1.0
        p2 = Binomial(Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, StochasticParameter):
        p2 = p
    else:
        raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))
    return MultiplyElementwise(p2, per_channel=per_channel, name=name, deterministic=deterministic, random_state=random_state)

# TODO tests
class Invert(Augmenter):
    """Augmenter that inverts all values in images.

    For the standard value range of 0-255 it converts 0 to 255, 255 to 0
    and 10 to (255-10)=245.

    Let M be the maximum value possible, m the minimum value possible,
    v a value. Then the distance of v to m is d=abs(v-m) and the new value
    is given by v'=M-d."""

    def __init__(self, p=0, per_channel=False, min_value=0, max_value=255, name=None,
                 deterministic=False, random_state=None):
        """Create a new Invert instance.



        Parameters
        ----------
        p : float or StochasticParameter, optional(default=0)
            The probability of an image being inverted.
            If a float, then that probability will be used for all images.
            If a StochasticParameter, then that parameter will queried per
              image and is expected too return values in the range [0.0, 1.0],
              where values >0.5 mean that the image/channel is supposed to be
              inverted.

        per_channel : bool, optional(default=False)
            Whether to use the same value for all channels (False)
            or to sample a new value for each channel (True).
            If this value is a float p, then for p percent of all images
            per_channel will be treated as True, otherwise as False.

        min_value : int or float, optional(default=0)
            Minimum of the range of possible pixel values. For uint8 (0-255)
            images, this should be 0.

        max_value : int or float, optional(default=255)
            Maximum of the range of possible pixel values. For uint8 (0-255)
            images, this should be 255.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(Invert, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be int or float or StochasticParameter, got %s." % (type(p),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            assert 0 <= per_channel <= 1.0
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

        self.min_value = min_value
        self.max_value = max_value

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.int32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                p_samples = self.p.draw_samples((nb_channels,), random_state=rs_image)
                for c, p_sample in enumerate(p_samples):
                    assert 0 <= p_sample <= 1
                    if p_sample > 0.5:
                        image_c = image[..., c]
                        distance_from_min = np.abs(image_c - self.min_value) # d=abs(v-m)
                        image[..., c] = -distance_from_min + self.max_value # v'=M-d
                np.clip(image, 0, 255, out=image)
                result[i] = image.astype(np.uint8)
            else:
                p_sample = self.p.draw_sample(random_state=rs_image)
                assert 0 <= p_sample <= 1.0
                if p_sample > 0.5:
                    distance_from_min = np.abs(image - self.min_value) # d=abs(v-m)
                    image = -distance_from_min + self.max_value
                    np.clip(image, 0, 255, out=image)
                    result[i] = image.astype(np.uint8)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.p, self.per_channel, self.min_value, self.max_value]

# TODO tests
class ContrastNormalization(Augmenter):
    """Augmenter that changes the contrast of images."""

    def __init__(self, alpha=1.0, per_channel=False, name=None, deterministic=False, random_state=None):
        """Create a new ContrastNormalization instance.

        Parameters
        ----------
        alpha : float or tuple of two floats or StochasticParameter, optional(default=1.0)
            Strength of the contrast normalization. Higher values than 1.0
            lead to higher contrast, lower values decrease the contrast.
            If a float, then that value will be used for all images.
            If a tuple (a, b), then a value will be sampled per image from the
              range a <= x <= b and be used as the alpha value.
            If a StochasticParameter, then this parameter will be used to
              sample the alpha value per image.

        per_channel : bool, optional(default=False)
            Whether to use the same value for all channels (False)
            or to sample a new value for each channel (True).
            If this value is a float p, then for p percent of all images
            per_channel will be treated as True, otherwise as False.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(ContrastNormalization, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(alpha):
            assert alpha >= 0.0, "Expected alpha to have range (0, inf), got value %.4f." % (alpha,)
            self.alpha = Deterministic(alpha)
        elif ia.is_iterable(alpha):
            assert len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(alpha)),)
            self.alpha = Uniform(alpha[0], alpha[1])
        elif isinstance(alpha, StochasticParameter):
            self.alpha = alpha
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            assert 0 <= per_channel <= 1.0
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.float32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel:
                nb_channels = images[i].shape[2]
                alphas = self.alpha.draw_samples((nb_channels,), random_state=rs_image)
                for c, alpha in enumerate(alphas):
                    image[..., c] = alpha * (image[..., c] - 128) + 128
            else:
                alpha = self.alpha.draw_sample(random_state=rs_image)
                image = alpha * (image - 128) + 128
            np.clip(image, 0, 255, out=image)
            result[i] = image.astype(np.uint8)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.alpha]


class Affine(Augmenter):
    """Augmenter to apply affine transformations to images.

    This is mostly a wrapper around skimage's
    [AffineTransform](http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.AffineTransform)
    class and
    [warp](http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp)
    function.

    Affine transformations involve:
        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a
    defined content, e.g. if the image is translated to the left, pixels
    are created on the right.
    A method has to be defined to deal with these pixel values. The
    parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameter `order`
    deals with the method of interpolation used for this.
    """

    def __init__(self, scale=1.0, translate_percent=None, translate_px=None,
                 rotate=0.0, shear=0.0, order=1, cval=0, mode="constant",
                 name=None, deterministic=False, random_state=None):
        """Create a new Affine instance.

        Example:
            aug = iaa.Affine(scale=2.0)
        zooms all images by a factor of 2.

        Example:
            aug = iaa.Affine(translate_px=16)
        translates all images on the x- and y-axis by 16 pixels (to the
        right/top), fills up any new pixels with zero (black values).

        Example:
            aug = iaa.Affine(translate_percent=0.1)
        translates all images on the x- and y-axis by 10 percent of their
        width/height (to the right/top), fills up any new pixels with zero
        (black values).

        Example:
            aug = iaa.Affine(rotate=35)
        rotates all images by 35 degrees, fills up any new pixels with zero
        (black values).

        Example:
            aug = iaa.Affine(shear=15)
        rotates all images by 15 degrees, fills up any new pixels with zero
        (black values).

        Example:
            aug = iaa.Affine(translate_px=(-16, 16))
        translates all images on the x- and y-axis by a random value
        between -16 and 16 pixels (to the right/top) (same for both axis, i.e.
        sampled once per image), fills up any new pixels with zero (black values).

        Example:
            aug = iaa.Affine(translate_px={"x": (-16, 16), "y": (-4, 4)})
        translates all images on the x-axis by a random value
        between -16 and 16 pixels (to the right) and on the y-axis by a
        random value between -4 and 4 pixels to the top. Even if both ranges
        were the same, both axis could use different samples.
        Fills up any new pixels with zero (black values).

        Example:
            aug = iaa.Affine(scale=2.0, order=[0, 1])
        same as previously, but uses (randomly) either nearest neighbour
        interpolation or linear interpolation.

        Example:
            aug = iaa.Affine(translate_px=16, cval=(0, 255))
        same as previously, but fills up any new pixels with a random
        brightness (same for the whole image).

        Example:
            aug = iaa.Affine(translate_px=16, mode=["constant", "edge"])
        same as previously, but fills up the new pixels in only 50 percent
        of all images with black values. In the other 50 percent of all cases,
        the value of the nearest edge is used.

        Parameters
        ----------
        scale : float or tuple of two floats or StochasticParameter or dict {"x": float/tuple/StochasticParameter, "y": float/tuple/StochasticParameter}, optional(default=1.0)
            Scaling factor to use, where 1.0 represents no change and 0.5 is
            zoomed out to 50 percent of the original size.
            If a single float, then that value will be used for all images.
            If a tuple (a, b), then a value will be sampled from the range
              a <= x <= b per image. That value will be used identically for
              both x- and y-axis.
            If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`scale`). Using a dictionary allows to
              set different values for the axis. If they are set to the same
              ranges, different values may still be sampled per axis.

        translate_percent : float or tuple of two floats or StochasticParameter or dict {"x": float/tuple/StochasticParameter, "y": float/tuple/StochasticParameter}, optional(default=1.0)
            Translation in percent relative to the image
            height/width (x-translation, y-translation) to use,
            where 0 represents no change and 0.5 is half of the image
            height/width.
            If a single float, then that value will be used for all images.
            If a tuple (a, b), then a value will be sampled from the range
              a <= x <= b per image. That percent value will be used identically
              for both x- and y-axis.
            If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`translate_percent`).
              Using a dictionary allows to set different values for the axis.
              If they are set to the same ranges, different values may still
              be sampled per axis.

        translate_px : int or tuple of two ints or StochasticParameter or dict {"x": int/tuple/StochasticParameter, "y": int/tuple/StochasticParameter}, optional(default=1.0)
            Translation in pixels.
            If a single int, then that value will be used for all images.
            If a tuple (a, b), then a value will be sampled from the discrete
              range [a .. b] per image. That number will be used identically
              for both x- and y-axis.
            If a StochasticParameter, then from that parameter a value will
              be sampled per image (again, used for both x- and y-axis).
            If a dictionary, then it is expected to have the keys "x" and/or "y".
              Each of these keys can have the same values as described before
              for this whole parameter (`translate_px`).
              Using a dictionary allows to set different values for the axis.
              If they are set to the same ranges, different values may still
              be sampled per axis.

        rotate : float or int or tuple of two floats/ints or StochasticParameter, optional(default=0)
            Rotation in degrees (NOT radians), i.e. expected value range is
            0 to 360.
            If a float/int, then that value will be used for all images.
            If a tuple (a, b), then a value will be sampled per image from the
              range a <= x <= b and be used as the rotation value.
            If a StochasticParameter, then this parameter will be used to
              sample the rotation value per image.

        shear : float or int or tuple of two floats/ints or StochasticParameter, optional(default=0)
            Shear in degrees (NOT radians), i.e. expected value range is
            0 to 360.
            If a float/int, then that value will be used for all images.
            If a tuple (a, b), then a value will be sampled per image from the
              range a <= x <= b and be used as the rotation value.
            If a StochasticParameter, then this parameter will be used to
              sample the shear value per image.

        order : int or iterable of int or ia.ALL or StochasticParameter, optional(default=1)
            Interpolation order to use. Same meaning as in skimage:
                0: Nearest-neighbor
                1: Bi-linear (default)
                2: Bi-quadratic (not recommended by skimage)
                3: Bi-cubic
                4: Bi-quartic
                5: Bi-quintic
            Method 0 and 1 are fast, 3 is a bit slower, 4 and 5 are very slow.
            If a single int, then that order will be used for all images.
            If an iterable, then for each image a random value will be sampled
              from that iterable (i.e. list of allowed order values).
            If ia.ALL, then equivalant to list [0, 1, 3, 4, 5].
            If StochasticParameter, then that parameter is queried per image
              to sample the order value to use.

        cval : int or float or tuple of two floats or ia.ALL or StochasticParameter, optional(default=0)
            The constant value used for skimage's transform function.
            This is the value used to fill up pixels in the result image that
            didn't exist in the input image (e.g. when translating to the left,
            some new pixels are created at the right). Such a fill-up with a
            constant value only happens, when `mode` is "constant".
            For standard uint8 images (value range 0-255), this value may also
            come from the range 0-255. It may be a float value, even for
            integer image dtypes.
            If this is a single int or float, then that value will be used
              (e.g. 0 results in black pixels).
            If a tuple (a, b), then a random value from the range a <= x <= b
              is picked per image.
            If ia.ALL, a value from the discrete range [0 .. 255] will be
              sampled per image.
            If a StochasticParameter, a new value will be sampled from the
              parameter per image.

        mode : string or list of string or ia.ALL or StochasticParameter, optional(default="constant")
            Parameter that defines the handling of newly created pixels.
            Same meaning as in skimage (and numpy.pad):
                "constant": Pads with a constant value
                "edge": Pads with the edge values of array
                "symmetric": Pads with the reflection of the vector mirrored
                    along the edge of the array.
                "reflect": Pads with the reflection of the vector mirrored on
                    the first and last values of the vector along each axis.
                "wrap": Pads with the wrap of the vector along the axis.
                    The first values are used to pad the end and the end values
                    are used to pad the beginning.
            If a single string, then that mode will be used for all images.
            If a list of strings, then per image a random mode will be picked
              from that list.
            If ia.ALL, then a random mode from all possible modes will be
              picked.
            If StochasticParameter, then the mode will be sampled from that
              parameter per image, i.e. it must return only the above mentioned
              strings.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(Affine, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        # Peformance:
        #  1.0x order 0
        #  1.5x order 1
        #  3.0x order 3
        # 30.0x order 4
        # 60.0x order 5
        # measurement based on 256x256x3 batches, difference is smaller
        # on smaller images (seems to grow more like exponentially with image
        # size)
        if order == ia.ALL:
            # self.order = DiscreteUniform(0, 5)
            self.order = Choice([0, 1, 3, 4, 5]) # dont use order=2 (bi-quadratic) because that is apparently currently not recommended (and throws a warning)
        elif ia.is_single_integer(order):
            assert 0 <= order <= 5, "Expected order's integer value to be in range 0 <= x <= 5, got %d." % (order,)
            self.order = Deterministic(order)
        elif isinstance(order, list):
            assert all([ia.is_single_integer(val) for val in order]), "Expected order list to only contain integers, got types %s." % (str([type(val) for val in order]),)
            assert all([0 <= val <= 5 for val in order]), "Expected all of order's integer values to be in range 0 <= x <= 5, got %s." % (str(order),)
            self.order = Choice(order)
        elif isinstance(order, StochasticParameter):
            self.order = order
        else:
            raise Exception("Expected order to be imgaug.ALL, int or StochasticParameter, got %s." % (type(order),))

        if cval == ia.ALL:
            self.cval = DiscreteUniform(0, 255)
        elif ia.is_single_number(cval):
            self.cval = Deterministic(cval)
        elif ia.is_iterable(cval):
            assert len(cval) == 2
            assert 0 <= cval[0] <= 255
            assert 0 <= cval[1] <= 255
            self.cval = Uniform(cval[0], cval[1])
        elif isinstance(cval, StochasticParameter):
            self.cval = cval
        else:
            raise Exception("Expected cval to be imgaug.ALL, int, float or StochasticParameter, got %s." % (type(cval),))

        # constant, edge, symmetric, reflect, wrap
        if mode == ia.ALL:
            self.mode = Choice(["constant", "edge", "symmetric", "reflect", "wrap"])
        elif ia.is_string(mode):
            self.mode = Deterministic(mode)
        elif isinstance(mode, list):
            assert all([ia.is_string(val) for val in mode])
            self.mode = Choice(mode)
        elif isinstance(mode, StochasticParameter):
            self.mode = mode
        else:
            raise Exception("Expected mode to be imgaug.ALL, a string, a list of strings or StochasticParameter, got %s." % (type(mode),))

        # scale
        # float | (float, float) | [float, float] | StochasticParameter
        def scale_handle_param(param, allow_dict):
            if isinstance(param, StochasticParameter):
                return param
            elif ia.is_single_number(param):
                assert param > 0.0, "Expected scale to have range (0, inf), got value %.4f. Note: The value to _not_ change the scale of images is 1.0, not 0.0." % (param,)
                return Deterministic(param)
            elif ia.is_iterable(param) and not isinstance(param, dict):
                assert len(param) == 2, "Expected scale tuple/list with 2 entries, got %d entries." % (str(len(param)),)
                assert param[0] > 0.0 and param[1] > 0.0, "Expected scale tuple/list to have values in range (0, inf), got values %.4f and %.4f. Note: The value to _not_ change the scale of images is 1.0, not 0.0." % (param[0], param[1])
                return Uniform(param[0], param[1])
            elif allow_dict and isinstance(param, dict):
                assert "x" in param or "y" in param
                x = param.get("x")
                y = param.get("y")

                x = x if x is not None else 1.0
                y = y if y is not None else 1.0

                return (scale_handle_param(x, False), scale_handle_param(y, False))
            else:
                raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(param),))
        self.scale = scale_handle_param(scale, True)

        # translate
        if translate_percent is None and translate_px is None:
            translate_px = 0

        assert translate_percent is None or translate_px is None

        if translate_percent is not None:
            # translate by percent
            def translate_handle_param(param, allow_dict):
                if ia.is_single_number(param):
                    return Deterministic(float(param))
                elif ia.is_iterable(param) and not isinstance(param, dict):
                    assert len(param) == 2, "Expected translate_percent tuple/list with 2 entries, got %d entries." % (str(len(param)),)
                    all_numbers = all([ia.is_single_number(p) for p in param])
                    assert all_numbers, "Expected translate_percent tuple/list to contain only numbers, got types %s." % (str([type(p) for p in param]),)
                    #assert param[0] > 0.0 and param[1] > 0.0, "Expected translate_percent tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (param[0], param[1])
                    return Uniform(param[0], param[1])
                elif allow_dict and isinstance(param, dict):
                    assert "x" in param or "y" in param
                    x = param.get("x")
                    y = param.get("y")

                    x = x if x is not None else 0
                    y = y if y is not None else 0

                    return (translate_handle_param(x, False), translate_handle_param(y, False))
                elif isinstance(param, StochasticParameter):
                    return param
                else:
                    raise Exception("Expected float, int or tuple/list with 2 entries of both floats or ints or StochasticParameter. Got %s." % (type(param),))
            self.translate = translate_handle_param(translate_percent, True)
        else:
            # translate by pixels
            def translate_handle_param(param, allow_dict):
                if ia.is_single_integer(param):
                    return Deterministic(param)
                elif ia.is_iterable(param) and not isinstance(param, dict):
                    assert len(param) == 2, "Expected translate_px tuple/list with 2 entries, got %d entries." % (str(len(param)),)
                    all_integer = all([ia.is_single_integer(p) for p in param])
                    assert all_integer, "Expected translate_px tuple/list to contain only integers, got types %s." % (str([type(p) for p in param]),)
                    return DiscreteUniform(param[0], param[1])
                elif allow_dict and isinstance(param, dict):
                    assert "x" in param or "y" in param
                    x = param.get("x")
                    y = param.get("y")

                    x = x if x is not None else 0
                    y = y if y is not None else 0

                    return (translate_handle_param(x, False), translate_handle_param(y, False))
                elif isinstance(param, StochasticParameter):
                    return param
                else:
                    raise Exception("Expected int or tuple/list with 2 ints or StochasticParameter. Got %s." % (type(param),))
            self.translate = translate_handle_param(translate_px, True)

        # rotate
        # StochasticParameter | float | int | (float or int, float or int) | [float or int, float or int]
        if isinstance(rotate, StochasticParameter):
            self.rotate = rotate
        elif ia.is_single_number(rotate):
            self.rotate = Deterministic(rotate)
        elif ia.is_iterable(rotate):
            assert len(rotate) == 2, "Expected rotate tuple/list with 2 entries, got %d entries." % (str(len(rotate)),)
            assert all([ia.is_single_number(val) for val in rotate]), "Expected floats/ints in rotate tuple/list"
            self.rotate = Uniform(rotate[0], rotate[1])
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(rotate),))

        # shear
        # StochasticParameter | float | int | (float or int, float or int) | [float or int, float or int]
        if isinstance(shear, StochasticParameter):
            self.shear = shear
        elif ia.is_single_number(shear):
            self.shear = Deterministic(shear)
        elif ia.is_iterable(shear):
            assert len(shear) == 2, "Expected rotate tuple/list with 2 entries, got %d entries." % (str(len(shear)),)
            assert all([ia.is_single_number(val) for val in shear]), "Expected floats/ints in shear tuple/list."
            self.shear = Uniform(shear[0], shear[1])
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(shear),))

    def _augment_images(self, images, random_state, parents, hooks):
        #images = images if isinstance(images, list) else [images]
        nb_images = len(images)
        #result = [None] * nb_images
        result = images

        scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, mode_samples, order_samples = self._draw_samples(nb_images, random_state)

        for i in sm.xrange(nb_images):
            height, width = images[i].shape[0], images[i].shape[1]
            shift_x = int(width / 2.0)
            shift_y = int(height / 2.0)
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x, translate_y = translate_samples[0][i], translate_samples[1][i]
            #assert isinstance(translate_x, (float, int))
            #assert isinstance(translate_y, (float, int))
            if ia.is_single_float(translate_y):
                translate_y_px = int(round(translate_y * images[i].shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(round(translate_x * images[i].shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            cval = cval_samples[i]
            mode = mode_samples[i]
            order = order_samples[i]
            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 or shear != 0:
                matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x_px, translate_y_px),
                    rotation=math.radians(rotate),
                    shear=math.radians(shear)
                )
                matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
                matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center)
                #print("before aug", images[i].dtype, np.min(images[i]), np.max(images[i]))
                image_warped = tf.warp(
                    images[i],
                    matrix.inverse,
                    order=order,
                    mode=mode,
                    cval=cval,
                    preserve_range=True,
                )
                #print("after aug", image_warped.dtype, np.min(image_warped), np.max(image_warped))
                # warp changes uint8 to float64, making this necessary
                if image_warped.dtype != images[i].dtype:
                    image_warped = image_warped.astype(images[i].dtype, copy=False)
                #print("after aug2", image_warped.dtype, np.min(image_warped), np.max(image_warped))
                #result[i] = result[i].astype(images[i].dtype, copy=False)
                result[i] = image_warped
            else:
                result[i] = images[i]

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, mode_samples, order_samples = self._draw_samples(nb_images, random_state)

        for i, keypoints_on_image in enumerate(keypoints_on_images):
            height, width = keypoints_on_image.height, keypoints_on_image.width
            shift_x = int(width / 2.0)
            shift_y = int(height / 2.0)
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x, translate_y = translate_samples[0][i], translate_samples[1][i]
            #assert isinstance(translate_x, (float, int))
            #assert isinstance(translate_y, (float, int))
            if ia.is_single_float(translate_y):
                translate_y_px = int(round(translate_y * keypoints_on_image.shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(round(translate_x * keypoints_on_image.shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            #cval = cval_samples[i]
            #mode = mode_samples[i]
            #order = order_samples[i]
            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 or shear != 0:
                matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x_px, translate_y_px),
                    rotation=math.radians(rotate),
                    shear=math.radians(shear)
                )
                matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
                matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center)

                coords = keypoints_on_image.get_coords_array()
                #print("coords", coords)
                #print("matrix", matrix.params)
                coords_aug = tf.matrix_transform(coords, matrix.params)
                #print("coords before", coords)
                #print("coordsa ftre", coords_aug, np.around(coords_aug).astype(np.int32))
                result.append(ia.KeypointsOnImage.from_coords_array(np.around(coords_aug).astype(np.int32), shape=keypoints_on_image.shape))
            else:
                result.append(keypoints_on_image)
        return result

    def get_parameters(self):
        return [self.scale, self.translate, self.rotate, self.shear]

    def _draw_samples(self, nb_samples, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]

        if isinstance(self.scale, tuple):
            scale_samples = (
                self.scale[0].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 10)),
                self.scale[1].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 20)),
            )
        else:
            scale_samples = self.scale.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 30))
            scale_samples = (scale_samples, scale_samples)

        if isinstance(self.translate, tuple):
            translate_samples = (
                self.translate[0].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 40)),
                self.translate[1].draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 50)),
            )
        else:
            translate_samples = self.translate.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 60))
            translate_samples = (translate_samples, translate_samples)

        assert translate_samples[0].dtype in [np.int32, np.int64, np.float32, np.float64]
        assert translate_samples[1].dtype in [np.int32, np.int64, np.float32, np.float64]

        rotate_samples = self.rotate.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 70))
        shear_samples = self.shear.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 80))

        cval_samples = self.cval.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 90))
        mode_samples = self.mode.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 100))
        order_samples = self.order.draw_samples((nb_samples,), random_state=ia.new_random_state(seed + 110))

        return scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, mode_samples, order_samples


# code partially from
# https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
class ElasticTransformation(Augmenter):
    """Augmenter to transform images using elastic transformations/distortions.

    Elastic transformations move pixels around based on displacement fields,
    leading to distorted images.
    See
        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003
    for a detailed explanation."""

    def __init__(self, alpha=0, sigma=0, name=None, deterministic=False,
                 random_state=None):
        """Create a new ElasticTransformation instance.

        Example:
            aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)
        apply elastic transformations with a strength/alpha of 0.5 and
        smoothness of 0.25 to all images.

        Example:
            aug = iaa.ElasticTransformation(alpha=(0.25, 3.0), sigma=0.25)
        apply elastic transformations with a strength/alpha that comes
        from the range 0.25 <= x <= 3.0 (randomly picked per image) and
        smoothness of 0.25.

        Parameters
        ----------
        alpha : float or tuple of two floats or StochasticParameter, optional(default=0)
            Strength of the distortion field. Higher values mean more "movement" of
            pixels.
            If float, then that value will be used for all images.
            If tuple (a, b), then a random value from range a <= x <= b will be
              sampled per image.
            If StochasticParameter, then that parameter will be used to sample
              a value per image.

        sigma : float or tuple of two floats or StochasticParameter, optional(default=0)
            Standard deviation of the gaussian kernel used to smooth the distortion
            fields.
            If float, then that value will be used for all images.
            If tuple (a, b), then a random value from range a <= x <= b will be
              sampled per image.
            If StochasticParameter, then that parameter will be used to sample
              a value per image.

        name : string, optional(default=None)
            See Augmenter.__init__()

        deterministic : bool, optional(default=False)
            See Augmenter.__init__()

        random_state : int or np.random.RandomState or None, optional(default=None)
            See Augmenter.__init__()
        """
        super(ElasticTransformation, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(alpha):
            assert alpha >= 0.0, "Expected alpha to have range [0, inf), got value %.4f." % (alpha,)
            self.alpha = Deterministic(alpha)
        elif ia.is_iterable(alpha):
            assert len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(alpha)),)
            self.alpha = Uniform(alpha[0], alpha[1])
        elif isinstance(alpha, StochasticParameter):
            self.alpha = alpha
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

        if ia.is_single_number(sigma):
            assert sigma >= 0.0, "Expected sigma to have range [0, inf), got value %.4f." % (sigma,)
            self.sigma = Deterministic(sigma)
        elif ia.is_iterable(sigma):
            assert len(sigma) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(sigma)),)
            self.sigma = Uniform(sigma[0], sigma[1])
        elif isinstance(sigma, StochasticParameter):
            self.sigma = sigma
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(sigma),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        seeds = ia.copy_random_state(random_state).randint(0, 10**6, (nb_images,))
        alphas = self.alpha.draw_samples((nb_images,), random_state=ia.copy_random_state(random_state))
        sigmas = self.sigma.draw_samples((nb_images,), random_state=ia.copy_random_state(random_state))
        for i in sm.xrange(nb_images):
            image = images[i]
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = ElasticTransformation.generate_indices(image_first_channel.shape, alpha=alphas[i], sigma=sigmas[i], random_state=ia.new_random_state(seeds[i]))
            result[i] = ElasticTransformation.map_coordinates(images[i], indices_x, indices_y)
        return result

    """
    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        # TODO do keypoints even have to be augmented for elastic transformations?
        # TODO this transforms keypoints to images, augments the images, then transforms
        # back to keypoints - inefficient and keypoints that get outside of the images
        # cannot be recovered
        result = []
        nb_images = len(keypoints_on_images)
        seeds = ia.copy_random_state(random_state).randint(0, 10**6, (nb_images,))
        alphas = self.alpha.draw_samples((nb_images,), random_state=ia.copy_random_state(random_state))
        sigmas = self.sigma.draw_samples((nb_images,), random_state=ia.copy_random_state(random_state))
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            indices_x, indices_y = ElasticTransformation.generate_indices(keypoints_on_image.shape[0:2], alpha=alphas[i], sigma=sigmas[i], random_state=ia.new_random_state(seeds[i]))
            keypoint_image = keypoints_on_image.to_keypoint_image()
            keypoint_image_aug = ElasticTransformation.map_coordinates(keypoint_image, indices_x, indices_y)
            keypoints_aug = ia.KeypointsOnImage.from_keypoint_image(keypoint_image_aug)
            result.append(keypoints_aug)
        return result
    """

    # no transformation of keypoints for this currently,
    # it seems like this is the more appropiate choice overall for this augmentation
    # technique
    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.alpha, self.sigma]

    @staticmethod
    def generate_indices(shape, alpha, sigma, random_state):
        assert len(shape) == 2

        dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    @staticmethod
    def map_coordinates(image, indices_x, indices_y):
        assert len(image.shape) == 3
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in sm.xrange(image.shape[2]):
            remapped_flat = ndimage.interpolation.map_coordinates(image[..., c], (indices_x, indices_y), order=1)
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result
