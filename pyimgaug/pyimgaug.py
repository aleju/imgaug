from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy

try:
    xrange
except NameError:  # python3
    xrange = range

def is_np_array(val):
    return isinstance(val, (np.ndarray, np.generic))

class StochasticParameter(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def draw_sample(self, random_state=None):
        return self.draw_samples(1, random_state=random_state)[0]

    @abstractmethod
    def draw_samples(self, n, random_state=None):
        raise NotImplemented()

class Binomial(StochasticParameter):
    def __init__(self, p):
        if isinstance(p, float):
            assert 0 <= p <= 1.0
            self.p = p
        elif isinstance(p, BinomialParameter):
            self.p = p.p
        else:
            raise Exception("Expected float value or Binomial object, got %s." % (type(p),))

    def draw_samples(self, n, random_state=None):
        rng = random_state if random_state is not None else np.random
        return rng.binomial(1, self.p, n)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Binomial(%.4f)" % (self.p,)

class Uniform(StochasticParameter):
    def __init__(self, a, b=None):
        if isinstance(a, Uniform):
            self.a = a.a
            self.b = a.b
        elif isinstance(a, (float, int)):
            assert b is not None and isinstance(b, (float, int))
            assert a <= b
            self.a = a
            self.b = b
        else:
            raise Exception("Expected two float values or Uniform object, got %s, %s." % (type(a), type(b)))

    def draw_samples(self, n, random_state=None):
        rng = random_state if random_state is not None else np.random
        return rng.uniform(self.a, self.b, n)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Uniform(%.6f, %.6f)" % (self.a, self.b)

class Deterministic(StochasticParameter):
    def __init__(self, value):
        if isinstance(value, StochasticParameter):
            self.value = value.draw_sample()
        elif isinstance(value, (float, int)):
            self.value = value
        else:
            raise Exception("Expected StochasticParameter object or float or int as value, got %s." % (type(value),))

    def draw_samples(self, n, random_state=None):
        return np.repeat(np.array([self.value]), n)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.value, int):
            return "Deterministic(int %d)" % (self.value,)
        else:
            return "Deterministic(float %.8f)" % (self.value,)

class Augmenter(object):
    __metaclass__ = ABCMeta

    def __init__(self, name):
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
        return [Noop(name=name) for i in n]

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
            self.p = Deterministic(sigma)
        elif isinstance(sigma, (tuple, list)):
            assert len(sigma) == 2, "Expected tuple/list with 2 entries, got %d entries." % (str(len(sigma)),)
            self.p = Uniform(p[0], p[1])
        elif isinstance(sigma, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(p),))

    def _transform(self, images):
        result = np.copy(images)
        nb_channels = images.shape[3]
        samples = self.p.draw_samples(images.shape[0])
        for i in xrange(images.shape[0]):
            sigma = self.samples[i]
            if sigma > 0:
                for channel in range(nb_channels):
                    result[i, :, :, channel] = ndimage.gaussian_filter(result[i, :, :, channel], sigma)
        return result

    def _to_deterministic(self, n):
        samples = self.p.draw_samples(n)
        return [GaussianNoise(Deterministic(sample), name=self.name) for sample in samples]

    def get_parameters(self):
        return [self.p]

# TODO
# mit RNG
def GaussianNoise(Augmenter):
    pass

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
            result = np.clip(result, 0, 255)
        return result

    def _to_deterministic(self, n):
        samples = self.mul.draw_samples(n)
        return [Multiply(Deterministic(sample), clip=self.clip, name=self.name) for sample in samples]

    def get_parameters(self):
        return [self.mul]

def AffineTransformation(Augmenter):
    def __init__(self, scale=1.0, translate=0, rotate=0.0, shear=0.0, name=None):
        # scale
        # float | (float, float) | [float, float] | StochasticParameter
        def scale_handle_param(param, allow_dict):
            if isinstance(scale, StochasticParameter):
                self.scale = scale
            elif isinstance(scale, float):
                assert scale > 0.0, "Expected scale to have range (0, inf), got value %.4f." % (scale,)
                self.scale = Deterministic(scale)
            elif isinstance(scale, (tuple, list)):
                assert len(scale) == 2, "Expected scale tuple/list with 2 entries, got %d entries." % (str(len(scale)),)
                assert scale[0] > 0.0 and scale[1] > 0.0, "Expected scale tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (scale[0], scale[1])
                self.scale = Uniform(scale[0], scale[1])
            elif allow_dict and isinstance(scale, dict):
                assert "x" in scale or "y" in scale
                x = scale.get("x")
                y = scale.get("y")

                if x is None and y is None:
                    x = y = 1.0
                else:
                    x = x if x is not None else y
                    y = y if y is not None else x

                if isinstance(scale, float):
                    assert scale > 0.0, "Expected scale to have range (0, inf), got value %.4f." % (scale,)
                    self.scale = Deterministic(scale)
                elif isinstance(scale, (tuple, list)):
                    assert len(scale) == 2, "Expected scale tuple/list with 2 entries, got %d entries." % (str(len(scale)),)
                    assert scale[0] > 0.0 and scale[1] > 0.0, "Expected scale tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (scale[0], scale[1])
                    self.scale = Uniform(scale[0], scale[1])
                elif isinstance(scale, StochasticParameter):
                    self.scale = scale
            else:
                raise Exception("Expected float, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(scale),))

        """
        if isinstance(scale, float):
            assert scale > 0.0, "Expected scale to have range (0, inf), got value %.4f." % (scale,)
            self.scale = Deterministic(scale)
        elif isinstance(scale, (tuple, list)):
            assert len(scale) == 2, "Expected scale tuple/list with 2 entries, got %d entries." % (str(len(scale)),)
            assert scale[0] > 0.0 and scale[1] > 0.0, "Expected scale tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (scale[0], scale[1])
            self.scale = Uniform(scale[0], scale[1])
        elif isinstance(scale, dict):
            assert "x" in scale or "y" in scale
            x = scale.get("x")
            y = scale.get("y")
            if x is None and y is None:
                x = y = 1.0
            else:
                x = x if x is not None else y
                y = y if y is not None else x

            if isinstance(scale, float):
                assert scale > 0.0, "Expected scale to have range (0, inf), got value %.4f." % (scale,)
                self.scale = Deterministic(scale)
            elif isinstance(scale, (tuple, list)):
                assert len(scale) == 2, "Expected scale tuple/list with 2 entries, got %d entries." % (str(len(scale)),)
                assert scale[0] > 0.0 and scale[1] > 0.0, "Expected scale tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (scale[0], scale[1])
                self.scale = Uniform(scale[0], scale[1])
            elif isinstance(scale, StochasticParameter):
                self.scale = scale

        elif isinstance(scale, StochasticParameter):
            self.scale = scale
        else:
            raise Exception("Expected float, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(scale),))
        """

        # translate
        # float | int | (float, float) | (int, int) | [float, float] | [int, int] | StochasticParameter
        if isinstance(translate, float):
            assert translate > 0.0, "Expected translate to have range (0, inf), got value %.4f." % (translate,)
            self.translate = Deterministic(translate)
        elif isinstance(translate, int):
            self.translate = Deterministic(translate)
        elif isinstance(translate, (tuple, list)):
            assert len(translate) == 2, "Expected translate tuple/list with 2 entries, got %d entries." % (str(len(translate)),)
            types_unique = set([type(val) for val in translate])
            assert len(types_unique) == 1, "Expected translate tuple/list to have either int or float datatype, got %s." % (str(types_unique),)
            assert types_unique in ["int", "float"], "Expected translate tuple/list to have either int or float datatype, got %s." % (str(types_unique),)

            if types_unique[0] == "int":
                self.translate = DiscreteUniform(translate[0], translate[1])
            else: # float
                assert translate[0] > 0.0 and translate[1] > 0.0, "Expected translate tuple/list to have values in range (0, inf), got values %.4f and %.4f." % (translate[0], translate[1])
                self.translate = Uniform(translate[0], translate[1])
        elif isinstance(translate, StochasticParameter):
            self.translate = translate
        else:
            raise Exception("Expected float, or int or tuple/list with 2 entries of both floats or ints or StochasticParameter. Got %s." % (type(translate),))



    def
