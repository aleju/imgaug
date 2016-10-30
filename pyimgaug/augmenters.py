from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy
from parameters import StochasticParameter, Deterministic, Binomial, DiscreteUniform, Normal, Uniform

class Augmenter(object):
    __metaclass__ = ABCMeta

    def __init__(self, name=None):
        if name is None:
            self.name = "Unnamed%s" % (self.__class__.__name__,)
        else:
            self.name = name

    def transform(self, images):
        if isinstance(images, (list, tuple)):
            return self.transform(np.array(images))
        elif is_np_array(images):
            assert len(images.shape) == 4, "Expected 4d array of form (N, height, width, rgb), got shape %s" % (str(images.shape),)
            #assert images.shape[3] == 3, "Expected RGB images, i.e. shape[3] == 3, got shape %s" % (str(images.shape),)
            assert images.dtype == np.uint8, "Expected dtype uint8 (with value range 0 to 255), got dtype %s." % (str(images.dtype),)
            return self._transform(images)
        else:
            raise Exception("Expected list/tuple of numpy arrays or one numpy array, got %s." % (type(images),))

    @abstractmethod
    def _transform(self, images):
        raise NotImplemented()

    def to_deterministic(self, n=None):
        if n is None:
            return self.to_deterministic(1)[0]
        else:
            return self._to_deterministic(n)

    @abstractmethod
    def _to_deterministic(self, n):
        raise NotImplemented()

    @abstractmethod
    def get_parameters(self):
        raise NotImplemented()

    def __str__(self):
        params = self.get_parameters()
        params_str = ", ".join([param.__str__() for param in params])
        return "%s(name=%s, parameters=[%s])" % (self.__class__.__name__, self.name, params_str)

class AugmenterSequence(Augmenter):
    def __init__(self, augmenters=None, name=None):
        Augmenter.__init__(self, name=name)
        self.augmenters = augmenters if augmenters is not None else []

    def _transform(self, images):
        result = images
        for augmenter in self.augmenters:
            result = augmenter._transform(result)
        return result

    def _to_deterministic(self, n):
        seqs = []
        for i in xrange(n):
            augs = [aug.to_deterministic() for aug in self.augmenters]
            seqs.append(AugmenterSequence(augs, name=name))
        return seqs

    def get_parameters(self):
        return []

    def append(self, augmenter):
        self.augmenters.append(augmenter)
        return self

    def extend(self, augmenters):
        self.augmenters.extend(augmenters)
        return self

    def __str__(self):
        augs_str = ", ".join([aug.__str__() for aug in self.augmenters])
        return "AugmenterSequence(name=%s, augmenters=[%s])" % (self.name, augs_str,)

class Sometimes(Augmenter):
    def __init__(self, p, then_list=None, else_list=None, name=None):
        Augmenter.__init__(self, name=name)
        self.p = Binomial(p)

        assert then_list is None or isinstance(then_list, (AugmenterSequence, list, tuple))
        assert else_list is None or isinstance(else_list, (AugmenterSequence, list, tuple))
        then_list = then_list if then_list is not None else []
        else_list = else_list if else_list is not None else []AugmenterSequence(name="%s-else" % (self.name,))
        then_list = AugmenterSequence(augmenters=then_list, name="%s-then" % (self.name,)) if isinstance(then_list, (list, tuple)) else then_list
        else_list = AugmenterSequence(augmenters=else_list, name="%s-else" % (self.name,)) if isinstance(else_list, (list, tuple)) else else_list
        self.then_list = then_list
        self.else_list = else_list

    def _transform(self, images):
        result = np.copy(images)
        samples = self.p.draw_samples(images.shape[0])
        for i in xrange(images.shape[0]):
            subimages = images[i][np.newaxis, ...]
            lst = self.then_list if samples[i] == 1 else self.else_list
            result[i] = lst._transform(subimages)
        return result

    def _to_deterministic(self, n):
        seqs = []
        samples = self.p.draw_samples(n)
        then_lists = self.then_list.to_deterministic(n)
        else_lists = self.else_list.to_deterministic(n)
        for i in xrange(n):
            seqs.append(Sometimes(Deterministic(samples[i]), then_list=then_lists[i], else_list=else_lists[i], name=self.name))
        return seqs

    def get_parameters(self):
        return [self.p]

    def __str__(self):
        return "Sometimes(p=%s, name=%s, then_list=[%s], else_list=[%s])" % (self.p.__str__(), self.name, self.then_list.__str__(), self.else_list.__str__())

class Noop(Augmenter):
    def __init__(self, name=None):
        Augmenter.__init__(self, name=name)

    def _transform(self, images):
        return images

    def _to_deterministic(self, n):
        return [Noop(name=name) for _ in xrange(n)]

    def get_parameters(self):
        return []

class Fliplr(Augmenter):
    def __init__(self, p=0.5, name=None):
        Augmenter.__init__(self, name=name)
        self.p = Binomial(p)

    def _transform(self, images):
        result = np.copy(images)
        samples = self.p.draw_samples(images.shape[0])
        for i in xrange(images.shape[0]):
            if samples[i] == 1:
                result[i] = np.fliplr(images[i])
        return result

    def _to_deterministic(self, n):
        samples = self.p.draw_samples(n)
        return [Fliplr(Deterministic(sample), name=name) for sample in samples]

    def get_parameters(self):
        return [self.p]

class Flipud(Augmenter):
    def __init__(self, p=0.5, name=None):
        Augmenter.__init__(self, name=name)
        self.p = Binomial(p)

    def _transform(self, images):
        result = np.copy(images)
        samples = self.p.draw_samples(images.shape[0])
        for i in xrange(images.shape[0]):
            if samples[i] == 1:
                result[i] = np.flipud(images[i])
        return result

    def _to_deterministic(self, n):
        samples = self.p.draw_samples(n)
        return [Flipud(Deterministic(sample), name=self.name) for sample in samples]

    def get_parameters(self):
        return [self.p]

class GaussianBlur(Augmenter):
    def __init__(self, sigma=0, name=None):
        Augmenter.__init__(self, name=name)
        if isinstance(sigma, (float, int)):
            self.sigma = Deterministic(sigma)
        elif isinstance(sigma, (tuple, list)):
            assert len(sigma) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(sigma)),)
            self.sigma = Uniform(sigma[0], sigma[1])
        elif isinstance(sigma, StochasticParameter):
            self.sigma = sigma
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(sigma),))

    def _transform(self, images):
        result = np.copy(images)
        nb_channels = images.shape[3]
        samples = self.sigma.draw_samples(images.shape[0])
        for i in xrange(images.shape[0]):
            sig = self.samples[i]
            if sig > 0:
                for channel in range(nb_channels):
                    result[i, :, :, channel] = ndimage.gaussian_filter(result[i, :, :, channel], sig)
        return result

    def _to_deterministic(self, n):
        samples = self.sigma.draw_samples(n)
        return [GaussianBlur(Deterministic(sample), name=self.name) for sample in samples]

    def get_parameters(self):
        return [self.sigma]

def GaussianNoise(Augmenter):
    def __init__(self, loc=0, scale=0, clip=True, name=None):
        Augmenter.__init__(self, name=name)

        if isinstance(loc, float):
            self.loc = Deterministic(loc)
        elif isinstance(loc, (tuple, list)):
            assert len(loc) == 2, "Expected tuple/list with 2 entries for argument 'loc', got %d entries." % (str(len(scale)),)
            self.loc = Uniform(loc[0], loc[1])
        elif isinstance(loc, StochasticParameter):
            self.loc = loc
        else:
            raise Exception("Expected float, tuple/list with 2 entries or StochasticParameter for argument 'loc'. Got %s." % (type(loc),))

        if isinstance(scale, float):
            self.scale = Deterministic(scale)
        elif isinstance(scale, (tuple, list)):
            assert len(scale) == 2, "Expected tuple/list with 2 entries for argument 'scale', got %d entries." % (str(len(scale)),)
            self.scale = Uniform(scale[0], scale[1])
        elif isinstance(scale, StochasticParameter):
            self.scale = scale
        else:
            raise Exception("Expected float, tuple/list with 2 entries or StochasticParameter for argument 'scale'. Got %s." % (type(scale),))

        self.clip = clip

    def _transform(self, images):
        result = np.copy(images)
        nb_channels = images.shape[3]
        samples_loc = self.loc.draw_samples(images.shape[0])
        samples_scale = self.scale.draw_samples(images.shape[0])
        for i in xrange(images.shape[0]):
            sample_loc = self.samples_loc[i]
            sample_scale = self.samples_scale[i]
            assert sample_scale >= 0
            if sample_loc != 0 or sample_scale > 0:
                result[i] += (255 * np.random.normal(sample_loc, sample_scale, size=images[i].shape))
        if self.clip:
            np.clip(result, 0, 255, out=result)
        return result

    def _to_deterministic(self, n):
        samples = self.p.draw_samples(n)
        return [GaussianNoise(Deterministic(sample), clip=self.clip, name=self.name) for sample in samples]

    def get_parameters(self):
        return [self.loc, self.scale]

def Multiply(Augmenter):
    def __init__(self, mul=1.0, clip=True, name=None):
        Augmenter.__init__(self, name=name)
        if isinstance(mul, float):
            assert mul >= 0.0, "Expected multiplier to have range [0, inf), got value %.4f." % (mul,)
            self.mul = Deterministic(mul)
        elif isinstance(mul, (tuple, list)):
            assert len(mul) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(mul)),)
            self.mul = Uniform(mul[0], mul[1])
        elif isinstance(mul, StochasticParameter):
            self.mul = mul
        else:
            raise Exception("Expected float, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(mul),))
        self.clip = clip

    def _transform(self, images):
        result = np.copy(images)
        samples = self.mul.draw_samples(images.shape[0])
        result = result * samples
        if self.clip:
            np.clip(result, 0, 255, out=result)
        return result

    def _to_deterministic(self, n):
        samples = self.mul.draw_samples(n)
        return [Multiply(Deterministic(sample), clip=self.clip, name=self.name) for sample in samples]

    def get_parameters(self):
        return [self.mul]

def Affine(Augmenter):
    def __init__(self, scale=1.0, translate=0, rotate=0.0, shear=0.0, name=None):
        Augmenter.__init__(self, name=name)
        self.warp_args = warp_args if warp_args is not None else dict()

        # scale
        # float | (float, float) | [float, float] | StochasticParameter
        def scale_handle_param(param, allow_dict):
            if isinstance(param, StochasticParameter):
                return param
            elif isinstance(param, float):
                assert param > 0.0, "Expected scale to have range (0, inf), got value %.4f." % (param,)
                return Deterministic(param)
            elif isinstance(param, (tuple, list)):
                assert len(param) == 2, "Expected scale tuple/list with 2 entries, got %d entries." % (str(len(param)),)
                assert param[0] > 0.0 and param[1] > 0.0, "Expected scale tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (param[0], param[1])
                return Uniform(param[0], param[1])
            elif allow_dict and isinstance(param, dict):
                assert "x" in param or "y" in param
                x = param.get("x")
                y = param.get("y")

                x = x if x is not None else y
                y = y if y is not None else x

                return (scale_handle_param(x, Fale), scale_handle_param(y, False))
            else:
                raise Exception("Expected float, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(param),))
        self.scale = scale_handle_param(scale, True)

        # translate
        # float | int | (float, float) | (int, int) | [float, float] | [int, int] | StochasticParameter
        def translate_handle_param(param, allow_dict):
            if isinstance(param, float):
                assert param > 0.0, "Expected translate to have range (0, inf), got value %.4f." % (param,)
                self.param = Deterministic(param)
            elif isinstance(param, int):
                self.param = Deterministic(param)
            elif isinstance(param, (tuple, list)):
                assert len(param) == 2, "Expected translate tuple/list with 2 entries, got %d entries." % (str(len(param)),)
                types_unique = set([type(val) for val in param])
                assert len(types_unique) == 1, "Expected translate tuple/list to have either int or float datatype, got %s." % (str(types_unique),)
                assert types_unique in ["int", "float"], "Expected translate tuple/list to have either int or float datatype, got %s." % (str(types_unique),)

                if types_unique[0] == "int":
                    self.translate = DiscreteUniform(param[0], param[1])
                else: # float
                    assert param[0] > 0.0 and param[1] > 0.0, "Expected translate tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (param[0], param[1])
                    self.translate = Uniform(param[0], param[1])
            elif allow_dict and isinstance(parm, dict):
                assert "x" in param or "y" in param
                x = param.get("x")
                y = param.get("y")

                x = x if x is not None else y
                y = y if y is not None else x

                return (translate_handle_param(x, Fale), translate_handle_param(y, False))
            elif isinstance(param, StochasticParameter):
                self.translate = param
            else:
                raise Exception("Expected float, or int or tuple/list with 2 entries of both floats or ints or StochasticParameter. Got %s." % (type(param),))
        self.translate = translate_handle_param(translate, True)

        # rotate
        # StochasticParameter | float | int | (float or int, float or int) | [float or int, float or int]
        if isinstance(rotate, StochasticParameter):
            self.rotate = rotate
        elif isinstance(rotate, (float, int)):
            self.rotate = rotate
        elif isinstance(rotate, (tuple, list)):
            assert len(rotate) == 2, "Expected rotate tuple/list with 2 entries, got %d entries." % (str(len(rotate)),)
            types = [type(r) for r in rotate]
            assert all([val in ["float", "int"] for val in types), "Expected floats/ints in rotate tuple/list, got %s." % (str(types),)
            self.rotate = Uniform(rotate[0], rotate[1])
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(param),))

        # shear
        # StochasticParameter | float | int | (float or int, float or int) | [float or int, float or int]
        if isinstance(shear, StochasticParameter):
            self.shear = shear
        elif isinstance(shear, (float, int)):
            self.shear = shear
        elif isinstance(shear, (tuple, list)):
            assert len(shear) == 2, "Expected rotate tuple/list with 2 entries, got %d entries." % (str(len(shear)),)
            types = [type(r) for r in rotate]
            assert all([val in ["float", "int"] for val in types), "Expected floats/ints in shear tuple/list, got %s." % (str(types),)
            self.shear = Uniform(shear[0], shear[1])
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(param),))

    def _transform(self, images):
        # skimage's warp() converts to 0-1 range, so we use float here and then convert
        # at the end
        result = np.copy(images).astype(np.float32, copy=False)

        nb_images, height, width = images.shape[0], images.shape[1], images.shape[2]

        scale_samples, translate_samples_px, rotate_samples, shear_samples = self._draw_samples(nb_images)

        shift_x = int(width / 2.0)
        shift_y = int(height / 2.0)

        for i in xrange(nb_images):
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x_px, translate_y_px = translate_samples_px[0][i], translate_samples_px[1][i]
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            if scale_x != 1.0 or scale_y != 1.0 or translate_x_px != 0 or translate_y_px != 0 or rotate != 0 or shear != 0:
                matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x, translate_y),
                    rotation=rotate,
                    shear=shear
                )
                matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
                matrix = (matrix_to_topleft + matrix_transforms + matrix_to_center).inverse
                result[i, ...] = tf.warp(result[i, ...], matrix, **self.warp_args)

        return (result * 255.0).astype(copy=False)

    def _to_deterministic(self, n):
        scale_samples, translate_samples_px, rotate_samples, shear_samples = self._draw_samples(n)
        augs = []
        for i in range(n):
            augs.append(
                Affine(
                    scale=(scale_samples[0][i], scale_samples[1][i]),
                    translate=(translate_samples_px[0][i], translate_samples_px[1][i]),
                    rotate=rotate_samples[i],
                    shear=shear_samples[i],
                    name=self.name,
                    warp_args=self.warp_args
                )
            )
        return augs

    def get_parameters(self):
        return [self.scale, self.translate, self.rotate, self.shear]

    def _draw_samples(self, nb_samples):
        if isinstance(self.scale, tuple):
            scale_samples = (self.scale[0].draw_samples(nb_samples), self.scale[1].draw_samples(nb_samples))
        else:
            scale_samples = self.scale.draw_samples(nb_samples)
            scale_samples = (scale_samples, scale_samples)

        if isinstance(self.translate, tuple):
            translate_samples = (self.translate[0].draw_samples(nb_samples), self.translate[1].draw_samples(nb_samples))
        else:
            translate_samples = self.translate.draw_samples(nb_samples)
            translate_samples = (translate_samples, translate_samples)

        assert translate_samples[0].dtype in [np.int32, np.int64, np.float32, np.float64]
        assert translate_samples[1].dtype in [np.int32, np.int64, np.float32, np.float64]
        translate_samples_px = [None, None]
        if translate_samples[0].dtype in [np.float32, np.float64]:
            translate_samples_px[0] = translate_samples[0] * width
        else:
            translate_samples_px[0] = translate_samples[0]
        if translate_samples[1].dtype in [np.float32, np.float64]:
            translate_samples_px[1] = translate_samples[1] * height
        else:
            translate_samples_px[1] = translate_samples[1]

        rotate_samples = self.rotate.draw_samples(nb_images)
        shear_samples = self.shear.draw_samples(nb_images)

        return scale_samples, translate_samples_px, rotate_samples, shear_samples
