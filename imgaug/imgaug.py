from __future__ import print_function, division, absolute_import
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy
import numbers
import cv2
import math
from scipy import misc
import six
import six.moves as sm

"""
try:
    xrange
except NameError:  # python3
    xrange = range
"""

ALL = "ALL"

# We instantiate a current/global random state here once.
# One can also call np.random, but that is (in contrast to np.random.RandomState)
# a module and hence cannot be copied via deepcopy. That's why we use RandomState
# here (and in all augmenters) instead of np.random.
CURRENT_RANDOM_STATE = np.random.RandomState(42)


def is_np_array(val):
    return isinstance(val, (np.ndarray, np.generic))


def is_single_integer(val):
    return isinstance(val, numbers.Integral)


def is_single_float(val):
    return isinstance(val, numbers.Real) and not is_single_integer(val)


def is_single_number(val):
    return is_single_integer(val) or is_single_float(val)


def is_iterable(val):
    return isinstance(val, (tuple, list))


def is_string(val):
    return isinstance(val, str) or isinstance(val, unicode)


def is_integer_array(val):
    return issubclass(val.dtype.type, np.integer)


def current_random_state():
    return CURRENT_RANDOM_STATE


def new_random_state(seed=None, fully_random=False):
    if seed is None:
        if not fully_random:
            # sample manually a seed instead of just RandomState(),
            # because the latter one
            # is way slower.
            seed = CURRENT_RANDOM_STATE.randint(0, 10**6, 1)[0]
    return np.random.RandomState(seed)


def dummy_random_state():
    return np.random.RandomState(1)


def copy_random_state(random_state, force_copy=False):
    if random_state == np.random and not force_copy:
        return random_state
    else:
        rs_copy = dummy_random_state()
        orig_state = random_state.get_state()
        rs_copy.set_state(orig_state)
        return rs_copy

# TODO
# def from_json(json_str):
#    pass


def imresize_many_images(images, sizes=None, interpolation=None):
    s = images.shape
    assert len(s) == 4, s
    nb_images = s[0]
    im_height, im_width = s[1], s[2]
    nb_channels = s[3]
    height, width = sizes[0], sizes[1]

    if height == im_height and width == im_width:
        return np.copy(images)

    ip = interpolation
    assert ip is None or ip in ["nearest", "linear", "area", "cubic", cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]
    if ip is None:
        if height > im_height or width > im_width:
            ip = cv2.INTER_AREA
        else:
            ip = cv2.INTER_LINEAR
    elif ip in ["nearest", cv2.INTER_NEAREST]:
        ip = cv2.INTER_NEAREST
    elif ip in ["linear", cv2.INTER_LINEAR]:
        ip = cv2.INTER_LINEAR
    elif ip in ["area", cv2.INTER_AREA]:
        ip = cv2.INTER_AREA
    elif ip in ["cubic", cv2.INTER_CUBIC]:
        ip = cv2.INTER_CUBIC
    else:
        raise Exception("Invalid interpolation order")

    result = np.zeros((nb_images, height, width, nb_channels), dtype=np.uint8)
    for img_idx in sm.xrange(nb_images):
        result_img = cv2.resize(images[img_idx], (width, height), interpolation=ip)
        if len(result_img.shape) == 2:
            result_img = result_img[:, :, np.newaxis]
        result[img_idx] = result_img
    return result


def imresize_single_image(image, sizes, interpolation=None):
    grayscale = False
    if image.shape == 2:
        grayscale = True
        image = image[:, :, np.newaxis]
    assert len(image.shape) == 3, image.shape
    rs = imresize_many_images(image[np.newaxis, :, :, :], sizes, interpolation=interpolation)
    if grayscale:
        return np.squeeze(rs[0, :, :, 0])
    else:
        return rs[0, ...]


def draw_grid(images, rows=None, cols=None):
    if is_np_array(images):
        assert len(images.shape) == 4
    else:
        assert is_iterable(images)

    nb_images = len(images)
    cell_height = max([image.shape[0] for image in images])
    cell_width = max([image.shape[1] for image in images])
    channels = set([image.shape[2] for image in images])
    assert len(channels) == 1
    nb_channels = list(channels)[0]
    if rows is None and cols is None:
        rows = cols = int(math.ceil(math.sqrt(nb_images)))
    elif rows is not None:
        cols = int(math.ceil(nb_images / rows))
    elif cols is not None:
        rows = int(math.ceil(nb_images / cols))
    assert rows * cols >= nb_images

    width = cell_width * cols
    height = cell_height * rows
    grid = np.zeros((height, width, nb_channels))
    cell_idx = 0
    for row_idx in sm.xrange(rows):
        for col_idx in sm.xrange(cols):
            if cell_idx < nb_images:
                image = images[cell_idx]
                cell_y1 = cell_height * row_idx
                cell_y2 = cell_y1 + image.shape[0]
                cell_x1 = cell_width * col_idx
                cell_x2 = cell_x1 + image.shape[1]
                grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image
            cell_idx += 1

    return grid


def show_grid(images, rows=None, cols=None):
    grid = draw_grid(images, rows=rows, cols=cols)
    misc.imshow(grid)


class HooksImages(object):
    """
    # TODO
    """
    def __init__(self, activator=None, propagator=None, preprocessor=None, postprocessor=None):
        self.activator = activator
        self.propagator = propagator
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def is_activated(self, images, augmenter, parents, default):
        if self.activator is None:
            return default
        else:
            return self.activator(images, augmenter, parents, default)

    # TODO is a propagating hook necessary? seems to be covered by activated
    # hook already
    def is_propagating(self, images, augmenter, parents, default):
        if self.propagator is None:
            return default
        else:
            return self.propagator(images, augmenter, parents, default)

    def preprocess(self, images, augmenter, parents):
        if self.preprocessor is None:
            return images
        else:
            return self.preprocessor(images, augmenter, parents)

    def postprocess(self, images, augmenter, parents):
        if self.postprocessor is None:
            return images
        else:
            return self.postprocessor(images, augmenter, parents)


class HooksKeypoints(HooksImages):
    pass


class Keypoint(object):
    """
    # TODO
    """

    def __init__(self, x, y):
        # these checks are currently removed because they are very slow for some
        # reason
        #assert is_single_integer(x), type(x)
        #assert is_single_integer(y), type(y)
        self.x = x
        self.y = y

    def project(self, from_shape, to_shape):
        if from_shape[0:2] == to_shape[0:2]:
            return Keypoint(x=self.x, y=self.y)
        else:
            from_height, from_width = from_shape[0:2]
            to_height, to_width = to_shape[0:2]
            x = int(round((self.x / from_width) * to_width))
            y = int(round((self.y / from_height) * to_height))
            return Keypoint(x=x, y=y)

    def shift(self, x, y):
        return Keypoint(self.x + x, self.y + y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Keypoint(x=%d, y=%d)" % (self.x, self.y)


class KeypointsOnImage(object):
    def __init__(self, keypoints, shape):
        self.keypoints = keypoints
        if is_np_array(shape):
            self.shape = shape.shape
        else:
            assert isinstance(shape, (tuple, list))
            self.shape = tuple(shape)

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    def on(self, image):
        if is_np_array(image):
            shape = image.shape
        else:
            shape = image

        if shape[0:2] == self.shape[0:2]:
            return self.deepcopy()
        else:
            keypoints = [kp.project(self.shape, shape) for kp in self.keypoints]
            return KeypointsOnImage(keypoints, shape)

    def draw_on_image(self, image, color=[0, 255, 0], size=3, copy=True, raise_if_out_of_image=False):
        if copy:
            image = np.copy(image)

        height, width = image.shape[0:2]

        for keypoint in self.keypoints:
            y, x = keypoint.y, keypoint.x
            if 0 <= y < height and 0 <= x < width:
                x1 = max(x - size//2, 0)
                x2 = min(x + 1 + size//2, width - 1)
                y1 = max(y - size//2, 0)
                y2 = min(y + 1 + size//2, height - 1)
                image[y1:y2, x1:x2] = color
            else:
                if raise_if_out_of_image:
                    raise Exception("Cannot draw keypoint x=%d, y=%d on image with shape %s." % (y, x, image.shape))

        return image

    def shift(self, x, y):
        keypoints = [keypoint.shift(x=x, y=y) for keypoint in self.keypoints]
        return KeypointsOnImage(keypoints, self.shape)

    def get_coords_array(self):
        result = np.zeros((len(self.keypoints), 2), np.int32)
        for i, keypoint in enumerate(self.keypoints):
            result[i, 0] = keypoint.x
            result[i, 1] = keypoint.y
        return result

    @staticmethod
    def from_coords_array(coords, shape):
        assert is_integer_array(coords), coords.dtype
        keypoints = [Keypoint(x=coords[i, 0], y=coords[i, 1]) for i in sm.xrange(coords.shape[0])]
        return KeypointsOnImage(keypoints, shape)

    def to_keypoint_image(self):
        assert len(self.keypoints) > 0
        height, width = self.shape[0:2]
        image = np.zeros((height, width, len(self.keypoints)), dtype=np.uint8)
        for i, keypoint in enumerate(self.keypoints):
            y = keypoint.y
            x = keypoint.x
            if 0 <= y < height and 0 <= x < width:
                image[y, x, i] = 255
        return image

    @staticmethod
    def from_keypoint_image(image, if_not_found_coords={"x": -1, "y": -1}, threshold=1):
        assert len(image.shape) == 3
        height, width, nb_keypoints = image.shape

        drop_if_not_found = False
        if if_not_found_coords is None:
            drop_if_not_found = True
            if_not_found_x = -1
            if_not_found_y = -1
        elif isinstance(if_not_found_coords, (tuple, list)):
            assert len(if_not_found_coords) == 2
            if_not_found_x = if_not_found_coords[0]
            if_not_found_y = if_not_found_coords[1]
        elif isinstance(if_not_found_coords, dict):
            if_not_found_x = if_not_found_coords["x"]
            if_not_found_y = if_not_found_coords["y"]
        else:
            raise Exception("Expected if_not_found_coords to be None or tuple or list or dict, got %s." % (type(if_not_found_coords),))

        keypoints = []
        for i in sm.xrange(nb_keypoints):
            maxidx_flat = np.argmax(image[..., i])
            maxidx_ndim = np.unravel_index(maxidx_flat, (height, width))
            found = (image[maxidx_ndim[0], maxidx_ndim[1], i] >= threshold)
            if found:
                keypoints.append(Keypoint(x=maxidx_ndim[1], y=maxidx_ndim[0]))
            else:
                if drop_if_not_found:
                    pass # dont add the keypoint to the result list, i.e. drop it
                else:
                    keypoints.append(Keypoint(x=if_not_found_x, y=if_not_found_y))

        return KeypointsOnImage(keypoints, shape=(height, width))

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        # for some reason deepcopy is way slower here than manual copy
        #return copy.deepcopy(self)
        kps = [Keypoint(x=kp.x, y=kp.y) for kp in self.keypoints]
        return KeypointsOnImage(kps, tuple(self.shape))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        #print(type(self.keypoints), type(self.shape))
        return "KeypointOnImage(%s, shape=%s)" % (str(self.keypoints), self.shape)

# TODO
"""
class BackgroundAugmenter(object):
    def __init__(self, image_source, augmenter, maxlen, nb_workers=1):
        self.augmenter = augmenter
        self.maxlen = maxlen
        self.result_queue = multiprocessing.Queue(maxlen)
        self.batch_workers = []
        for i in range(nb_workers):
            worker = multiprocessing.Process(target=self._augment, args=(image_source, augmenter, self.result_queue))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)

    def join(self):
        for worker in self.batch_workers:
            worker.join()

    def get_batch(self):
        return self.result_queue.get()

    def _augment(self, image_source, augmenter, result_queue):
        batch = next(image_source)
        self.result_queue.put(augmenter.transform(batch))
"""
