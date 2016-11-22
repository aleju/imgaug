from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy
import re
import math
from scipy import misc, ndimage
from skimage import transform as tf
import itertools
import pyimgaug as ia
from parameters import StochasticParameter, Deterministic, Binomial, Choice, DiscreteUniform, Normal, Uniform

try:
    xrange
except NameError:  # python3
    xrange = range

class Augmenter(object):
    __metaclass__ = ABCMeta

    def __init__(self, name=None, deterministic=False, random_state=None):
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
        #elif isinstance(random_state, AugmenterRandomState):
        #    self.random_state = random_state
        #elif isinstance(random_state, np.random.RandomState):
        #    self.random_state = AugmenterRandomState(random_state)
        #else:
        #    self.random_state = ia.AugmenterRandomState(np.random.RandomState(random_state))
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        self.activated = True

    def augment_batches(self, batches, hooks=None):
        assert isinstance(batches, list)
        return [self.augment_images(batch, hooks=hooks) for batch in batches]

    def augment_images(self, images, parents=None, hooks=None):
        if self.deterministic:
            state_orig = self.random_state.get_state()

        if parents is None:
            parents = []

        if hooks is None:
            hooks = ia.HooksImages()

        if ia.is_np_array(images):
            assert len(images.shape) == 4, "Expected 4d array of form (N, height, width, channels), got shape %s" % (str(images.shape),)
            assert images.dtype == np.uint8, "Expected dtype uint8 (with value range 0 to 255), got dtype %s." % (str(images.dtype),)
            images_tf = images
        elif ia.is_iterable(images):
            #assert all([len(image.shape) == 3 for image in images])
            if len(images) > 0:
                # dont check all images, only the first one
                # that's faster and usually all are affected by the same problem anyways
                # AssertShape exists for more thorough checks
                assert len(images[0].shape) == 3, "Expected list of images with each image having shape length 3, got length %d." % (len(images[0].shape),)
                assert images[0].dtype == np.uint8, "Expected dtype uint8 (with value range 0 to 255), got dtype %s." % (str(images[0].dtype),)
            images_tf = list(images)
        else:
            raise Exception("Expected list/tuple of numpy arrays or one numpy array, got %s." % (type(images),))

        if isinstance(images_tf, list):
            images_copy = [np.copy(image) for image in images]
        else:
            images_copy = np.copy(images)

        images_copy = hooks.preprocess(images_copy, augmenter=self, parents=parents)

        if hooks.is_activated(images_copy, augmenter=self, parents=parents):
            if len(images) > 0:
                images_result = self._augment_images(
                    images_copy,
                    random_state=ia.copy_random_state(self.random_state),
                    parents=parents,
                    hooks=hooks
                )
                self.random_state.uniform()
            else:
                images_result = images_copy
        else:
            images_result = images_copy

        images_result = hooks.postprocess(images_result, augmenter=self, parents=parents)

        if self.deterministic:
            self.random_state.set_state(state_orig)

        if isinstance(images_result, list):
            assert all([image.dtype == np.uint8 for image in images_result]), "Expected list of dtype uint8 as augmenter result, got %s." % ([image.dtype for image in images_result],)
        else:
            assert images_result.dtype == np.uint8, "Expected dtype uint8 as augmenter result, got %s." % (images_result.dtype,)

        return images_result

    @abstractmethod
    def _augment_images(self, images, random_state, parents, hooks):
        raise NotImplemented()

    def augment_keypoints(self, keypoints_on_images, parents=None, hooks=None):
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

        if hooks.is_activated(keypoints_on_images_copy, augmenter=self, parents=parents):
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
        raise NotImplemented()

    def draw_grid(self, images, rows, cols):
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

    def show_grid(self, images, rows, cols):
        grid = self.draw_grid(images, rows, cols)
        misc.imshow(grid)

    """
    def transform_augjob(self, augjob):
        has_activator = (augjob.activator is not None)
        if (has_activator and augjob.activator(self, augjob)) or (not has_activator and self.activator):
            augjob.add_history(self, augjob.images, augjob.images, {})
            return augjob
        else:
            before = augjob.images

            if augjob.preprocessor is not None:
                augjob = augjob.preprocessor(self, augjob)

            augjob_transformed, changes = self._transform_augjob(augjob)

            if augjob.postprocessor is not None:
                augjob = augjob.postprocessor(self, augjob, augjob_transformed, changes)
            augjob.add_history(self, augjob.images, augjob_transformed.images, changes)

            return augjob_transformed
    """

    def to_deterministic(self, n=None):
        if n is None:
            return self.to_deterministic(1)[0]
        else:
            return [self._to_deterministic() for _ in xrange(n)]

    def _to_deterministic(self):
        aug = copy.copy(self)
        aug.random_state = ia.new_random_state()
        aug.deterministic = True
        return aug

    def reseed(self, deterministic_too=False, random_state=None):
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
        raise NotImplemented()

    def get_children_lists(self):
        return []

    def find_augmenters(self, func, flat=True):
        return self._find_augmenters(func, parents=[], flat=flat)

    def _find_augmenters(self, func, parents, flat):
        result = []
        if func(self, parents):
            result.append(self)

        subparents = parents + [self]
        for lst in self.get_children_lists():
            for aug in lst:
                found = aug._find_augmenters(func, parents=subparents, flat=flat)
                if len(found) > 0:
                    if flat:
                        result.extend(found)
                    else:
                        result.append(found)
        return result

    def find_augmenters_by_name(self, name, regex=False, flat=True):
        return self.find_augmenters_by_names([name], regex=regex, flat=flat)

    def find_augmenters_by_names(self, names, regex=False, flat=True):
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
        if func(self, []):
            if not copy:
                raise Exception("Inplace removal of topmost augmenter requested, which is currently not possible.")

            if noop_if_topmost:
                return Noop()
            else:
                return None
        else:
            aug = self if not copy else self.deepcopy()
            aug._remove_augmenters_inplace(func, parents=[])
            return aug

    def _remove_augmenters_inplace(self, func, parents):
        subparents = parents + [self]
        for lst in self.get_children_lists():
            to_remove = []
            for i, aug in enumerate(lst):
                if func(aug, subparents):
                    to_remove.append((i, aug))

            for count_removed, (i, aug) in enumerate(to_remove):
                self._remove_augmenters_inplace_from_list(lst, aug, i, i - count_removed)

            for aug in lst:
                aug._remove_augmenters_inplace(func, subparents)

    def _remove_augmenters_inplace_from_list(self, lst, aug, index, index_adjusted):
        del lst[index_adjusted]

    # TODO
    #def to_json(self):
    #    pass

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        params = self.get_parameters()
        params_str = ", ".join([param.__str__() for param in params])
        return "%s(name=%s, parameters=[%s], deterministic=%s)" % (self.__class__.__name__, self.name, params_str, self.deterministic)

class Sequential(Augmenter, list):
    def __init__(self, children=None, random_order=False, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        list.__init__(self, children if children is not None else [])
        #self.children = children if children is not None else []
        self.random_order = random_order

    def _augment_images(self, images, random_state, parents, hooks):
        if hooks.is_propagating(images, augmenter=self, parents=parents):
            if self.random_order:
                #for augmenter in self.children:
                for index in random_state.permutation(len(self)):
                    images = self[index].augment_images(
                        images=images,
                        parents=parents + [self],
                        hooks=hooks
                    )
            else:
                #for augmenter in self.children:
                for augmenter in self:
                    images = augmenter.augment_images(
                        images=images,
                        parents=parents + [self],
                        hooks=hooks
                    )
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        if hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents):
            if self.random_order:
                #for augmenter in self.children:
                for index in random_state.permutation(len(self)):
                    keypoints_on_images = self[index].augment_keypoints(
                        keypoints_on_images=keypoints_on_images,
                        parents=parents + [self],
                        hooks=hooks
                    )
            else:
                #for augmenter in self.children:
                for augmenter in self:
                    keypoints_on_images = augmenter.augment_keypoints(
                        keypoints_on_images=keypoints_on_images,
                        parents=parents + [self],
                        hooks=hooks
                    )
        return keypoints_on_images

    def _to_deterministic(self):
        #augs = [aug.to_deterministic() for aug in self.children]
        augs = [aug.to_deterministic() for aug in self]
        seq = copy.copy(self)
        seq[:] = augs
        seq.random_state = ia.new_random_state()
        seq.deterministic = True
        return seq

    def get_parameters(self):
        return []

    def add(self, augmenter):
        self.append(augmenter)

    #def append(self, augmenter):
    #    self.children.append(augmenter)
    #    return self

    #def extend(self, augmenters):
    #    self.children.extend(augmenters)
    #    return self

    def get_children_lists(self):
        return [self]

    def __str__(self):
        #augs_str = ", ".join([aug.__str__() for aug in self.children])
        augs_str = ", ".join([aug.__str__() for aug in self])
        return "Sequential(name=%s, augmenters=[%s], deterministic=%s)" % (self.name, augs_str, self.deterministic)

class Sometimes(Augmenter):
    def __init__(self, p, then_list=None, else_list=None, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
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
        if hooks.is_propagating(images, augmenter=self, parents=parents):
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
        if hooks.is_propagating(keypoints_on_images, augmenter=self, parents=parents):
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

    """
    def _to_deterministic(self, n):
        seqs = []
        then_lists = self.then_list.to_deterministic(n)
        else_lists = self.else_list.to_deterministic(n)
        for i in xrange(n):
            seqs.append(Sometimes(Deterministic(samples[i]), then_list=then_lists[i], else_list=else_lists[i], name=self.name))
        return seqs
    """
    def _to_deterministic(self):
        aug = copy.copy(self)
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

class Noop(Augmenter):
    def __init__(self, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []

class Lambda(Augmenter):
    def __init__(self, func_images, func_keypoints, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
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

                for i in xrange(len(images)):
                    image = images[i]
                    assert len(image.shape) == 3, "Expected image number %d to have a shape of length 3, got %d (shape: %s)." % (i, len(image.shape), str(image.shape))
                    for j in xrange(len(shape)-1):
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

            for i in xrange(len(keypoints_on_images)):
                keypoints_on_image = keypoints_on_images[i]
                for j in xrange(len(shape[0:2])):
                    expected = shape[j+1]
                    observed = keypoints_on_image.shape[j]
                    compare(observed, expected, j, i)
        return keypoints_on_images

    if name is None:
        name = "UnnamedAssertShape"

    return Lambda(func_images, func_keypoints, name=name, deterministic=deterministic, random_state=random_state)

class Crop(Augmenter):
    def __init__(self, px=None, percent=None, resize=True, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

        self.resize = resize

        if px is None and percent is None:
            self.mode = "noop"
        elif px is not None and percent is not None:
            raise Exception("Can only crop by pixels or percent, not both.")
        elif px is not None:
            self.mode = "px"
            if ia.is_single_integer(px):
                assert px >= 0
                self.top = self.right = self.bottom = self.left = Deterministic(px)
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
                    self.top = self.right = self.bottom = self.left = handle_param(px)
                else: # len == 4
                    self.top = handle_param(px[0])
                    self.right = handle_param(px[1])
                    self.bottom = handle_param(px[2])
                    self.left = handle_param(px[3])
            elif isinstance(px, StochasticParameter):
                self.top = self.right = self.bottom = self.left = px
            else:
                raise Exception("Expected int, tuple of 4 ints/lists/StochasticParameters or StochasticParameter, git type %s." % (type(px),))
        else: # = elif percent is not None:
            self.mode = "percent"
            if ia.is_single_number(percent):
                assert 0 <= percent < 1.0
                self.top = self.right = self.bottom = self.left = Deterministic(percent)
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
                    self.top = self.right = self.bottom = self.left = handle_param(percent)
                else: # len == 4
                    self.top = handle_param(percent[0])
                    self.right = handle_param(percent[1])
                    self.bottom = handle_param(percent[2])
                    self.left = handle_param(percent[3])
            elif isinstance(percent, StochasticParameter):
                self.top = self.right = self.bottom = self.left = percent
            else:
                raise Exception("Expected number, tuple of 4 numbers/lists/StochasticParameters or StochasticParameter, got type %s." % (type(percent),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in xrange(nb_images):
            seed = seeds[i]
            height, width = images[i].shape[0:2]
            top, right, bottom, left = self._draw_samples_image(seed, height, width)
            image_cropped = images[i][top:height-bottom, left:width-right, :]
            if self.resize:
                image_cropped = ia.imresize_single_image(image_cropped, (height, width))
            result.append(image_cropped)

        if not isinstance(images, list):
            if self.resize:
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
            if self.resize:
                result.append(shifted.on(keypoints_on_image.shape))
            else:
                result.append(shifted)

        return result

    def _draw_samples_image(self, seed, height, width):
        random_state = ia.new_random_state(seed)

        top = self.top.draw_samples((1,), random_state=ia.copy_random_state(random_state))[0]
        right = self.right.draw_samples((1,), random_state=ia.copy_random_state(random_state))[0]
        bottom = self.bottom.draw_samples((1,), random_state=ia.copy_random_state(random_state))[0]
        left = self.left.draw_samples((1,), random_state=ia.copy_random_state(random_state))[0]

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
    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be int or float or StochasticParameter, got %s." % (type(p),))

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in xrange(nb_images):
            if samples[i] == 1:
                images[i] = np.fliplr(images[i])
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                height, width = keypoints_on_image.shape[0:2]
                for keypoint in keypoints_on_image.keypoints:
                    keypoint.x = (width - 1) - keypoint.x
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]

class Flipud(Augmenter):
    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be int or float or StochasticParameter, got %s." % (type(p),))

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in xrange(nb_images):
            if samples[i] == 1:
                images[i] = np.flipud(images[i])
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if samples[i] == 1:
                height, width = keypoints_on_image.shape[0:2]
                for keypoint in keypoints_on_image.keypoints:
                    keypoint.y = (height - 1) - keypoint.y
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]

class GaussianBlur(Augmenter):
    def __init__(self, sigma=0, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
        if ia.is_single_number(sigma):
            self.sigma = Deterministic(sigma)
        elif ia.is_iterable(sigma):
            assert len(sigma) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(sigma)),)
            self.sigma = Uniform(sigma[0], sigma[1])
        elif isinstance(sigma, StochasticParameter):
            self.sigma = sigma
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(sigma),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.sigma.draw_samples((nb_images,), random_state=random_state)
        for i in xrange(nb_images):
            nb_channels = images[i].shape[2]
            sig = samples[i]
            if sig > 0:
                for channel in xrange(nb_channels):
                    result[i][:, :, channel] = ndimage.gaussian_filter(result[i][:, :, channel], sig)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.sigma]

class AdditiveGaussianNoise(Augmenter):
    def __init__(self, loc=0, scale=0, per_channel=False, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(loc):
            self.loc = Deterministic(loc)
        elif ia.is_iterable(loc):
            assert len(loc) == 2, "Expected tuple/list with 2 entries for argument 'loc', got %d entries." % (str(len(scale)),)
            self.loc = Uniform(loc[0], loc[1])
        elif isinstance(loc, StochasticParameter):
            self.loc = loc
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter for argument 'loc'. Got %s." % (type(loc),))

        if ia.is_single_number(scale):
            self.scale = Deterministic(scale)
        elif ia.is_iterable(scale):
            assert len(scale) == 2, "Expected tuple/list with 2 entries for argument 'scale', got %d entries." % (str(len(scale)),)
            self.scale = Uniform(scale[0], scale[1])
        elif isinstance(scale, StochasticParameter):
            self.scale = scale
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter for argument 'scale'. Got %s." % (type(scale),))

        self.per_channel = per_channel

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples_seeds = ia.copy_random_state(random_state).randint(0, 10**6, size=(nb_images,))
        samples_loc = self.loc.draw_samples(nb_images, random_state=ia.copy_random_state(random_state))
        samples_scale = self.scale.draw_samples(nb_images, random_state=ia.copy_random_state(random_state))
        for i in xrange(nb_images):
            nb_channels = images[i].shape[2]
            sample_seed = samples_seeds[i]
            sample_loc = samples_loc[i]
            sample_scale = samples_scale[i]
            assert sample_scale >= 0
            if sample_loc != 0 or sample_scale > 0:
                rs = ia.new_random_state(sample_seed)
                if self.per_channel:
                    noise = rs.normal(sample_loc, sample_scale, size=images[i].shape)
                else:
                    height, width, nb_channels = images[i].shape
                    noise = rs.normal(sample_loc, sample_scale, size=(height, width, 1))
                    noise = np.tile(noise, (1, 1, nb_channels))
                noise = (noise * 255).astype(np.int32)
                after_noise = images[i] + noise
                np.clip(after_noise, 0, 255, out=after_noise)
                result[i] = after_noise.astype(np.uint8)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.loc, self.scale]

# TODO
#class MultiplicativeGaussianNoise(Augmenter):
#    pass

# TODO
#class ReplacingGaussianNoise(Augmenter):
#    pass

class Dropout(Augmenter):
    def __init__(self, p=0, per_channel=False, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(p):
            self.p = Binomial(p)
        elif ia.is_iterable(p):
            assert len(p) == 2
            assert p[0] < p[1]
            assert 0 <= p[0] <= 1.0
            assert 0 <= p[1] <= 1.0
            self.p = Binomial(Uniform(p[0], p[1]))
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))

        self.per_channel = per_channel

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples_seeds = random_state.randint(0, 10**6, size=(nb_images,))
        for i in range(nb_images):
            height, width, nb_channels = images[i].shape
            seed = samples_seeds[i]
            rs_image = ia.new_random_state(seed)
            if self.per_channel:
                samples = self.p.draw_samples((height, width, nb_channels), random_state=rs_image)
            else:
                samples = self.p.draw_samples((height, width, 1), random_state=rs_image)
                samples = np.tile(samples, (1, 1, nb_channels))
            image_drop = result[i] * (1 - samples)
            result[i] = image_drop.astype(np.uint8) # for some reason image_drop is int64
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]

class Multiply(Augmenter):
    def __init__(self, mul=1.0, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)
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

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.mul.draw_samples((nb_images,), random_state=random_state)
        for i in xrange(nb_images):
            after_multiply = images[i] * samples[i]
            np.clip(after_multiply, 0, 255, out=after_multiply)
            result[i] = after_multiply.astype(np.uint8)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul]

class Affine(Augmenter):
    def __init__(self, scale=1.0, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0, order=1, cval=0.0, mode="constant", name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

        if order == ia.ALL:
            #self.order = DiscreteUniform(0, 5)
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
            self.cval = Uniform(0, 1.0)
        elif ia.is_single_number(cval):
            assert 0 <= cval <= 1.0
            self.cval = Deterministic(cval)
        elif ia.is_iterable(cval):
            assert len(cval) == 2
            assert 0 <= cval[0] <= 1.0
            assert 0 <= cval[1] <= 1.0
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
                    assert all_numbers, "Expected translate_percent tuple/list to contain only numbers, got types %s." % (str([type(p) in param]),)
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
                    assert all_integer, "Expected translate_px tuple/list to contain only integers, got types %s." % (str([type(p) in param]),)
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

        # float | int | (float, float) | (int, int) | [float, float] | [int, int] | StochasticParameter
        """
        def translate_handle_param(param, allow_dict):
            if ia.is_single_number(param):
                return Deterministic(param)
            elif ia.is_iterable(param) and not isinstance(param, dict):
                assert len(param) == 2, "Expected translate tuple/list with 2 entries, got %d entries." % (str(len(param)),)
                all_integer = all([ia.is_single_integer(p) for p in param])
                all_float = all([ia.is_single_float(p) for p in param])
                assert all_integer or all_float, "Expected translate tuple/list to have either int or float datatype, got types %s." % (str([type(p) in param]),)

                if all_integer:
                    return DiscreteUniform(param[0], param[1])
                else: # float
                    assert param[0] > 0.0 and param[1] > 0.0, "Expected translate tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (param[0], param[1])
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
        self.translate = translate_handle_param(translate, True)
        """

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
        # skimage's warp() converts to 0-1 range, so we use float here and then convert
        # at the end
        # float images are expected by skimage's warp() to be in range 0-1, so we divide by 255
        if isinstance(images, list):
            result = [image.astype(np.float32, copy=False) for image in images]
            result = [image / 255.0 for image in images]
        else:
            result = images.astype(np.float32, copy=False)
            result = result / 255.0

        nb_images = len(images)

        scale_samples, translate_samples, rotate_samples, shear_samples, cval_samples, mode_samples, order_samples = self._draw_samples(nb_images, random_state)

        for i in xrange(nb_images):
            height, width = result[i].shape[0], result[i].shape[1]
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
                result[i] = tf.warp(
                    result[i],
                    matrix.inverse,
                    order=order,
                    mode=mode,
                    cval=cval
                )

            result[i] *= 255.0
            np.clip(result[i], 0, 255, out=result[i])

        if isinstance(images, list):
            result = [image.astype(np.uint8, copy=False) for image in result]
        else:
            result = result.astype(np.uint8, copy=False)

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

# code partially from https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
class ElasticTransformation(Augmenter):
    def __init__(self, alpha=0, sigma=0, name=None, deterministic=False, random_state=None):
        Augmenter.__init__(self, name=name, deterministic=deterministic, random_state=random_state)

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
        for i in xrange(nb_images):
            seed = seeds[i]
            alpha = alphas[i]
            sigma = sigmas[i]
            image = images[i]
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=alpha, sigma=sigma, random_state=ia.new_random_state(seed))
            result[i] = self._map_coordinates(images[i], indices_x, indices_y)
        return result

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
        for keypoints_on_image in keypoints_on_images:
            seed = seeds[i]
            alpha = alphas[i]
            sigma = sigmas[i]
            indices_x, indices_y = self._generate_indices(keypoints_on_image.shape, alpha=alpha, sigma=sigma, random_state=ia.new_random_state(seed))
            keypoint_image = keypoints_on_image.to_keypoint_image()
            keypoint_image_aug = self._map_coordinates(keypoint_image, indices_x, indices_y)
            keypoints_aug = KeypointsOnImage.from_keypoint_image(keypoint_image_aug)
            result.append(keypoints_aug)
        return result

    def get_parameters(self):
        return [self.alpha, self.sigma]

    def _generate_indices(self, shape, alpha, sigma, random_state):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        assert len(shape) == 2

        dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y):
        assert len(image.shape) == 3
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in xrange(image.shape[2]):
            remapped_flat = ndimage.interpolation.map_coordinates(image[..., c], (indices_x, indices_y), order=1)
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result
