from __future__ import print_function, division, absolute_import
from . import imgaug as ia
from abc import ABCMeta, abstractmethod
import numpy as np
import copy as copy_module
import six

@six.add_metaclass(ABCMeta)
class StochasticParameter(object):
    def __init__(self):
        super(StochasticParameter, self).__init__()

    def draw_sample(self, random_state=None):
        return self.draw_samples(1, random_state=random_state)[0]

    def draw_samples(self, size, random_state=None):
        random_state = random_state if random_state is not None else ia.current_random_state()
        return self._draw_samples(size, random_state)

    @abstractmethod
    def _draw_samples(self, size, random_state):
        raise NotImplementedError()

    def copy(self):
        return copy_module.copy(self)

    def deepcopy(self):
        return copy_module.deepcopy(self)

class Binomial(StochasticParameter):
    def __init__(self, p):
        super(Binomial, self).__init__()

        if isinstance(p, StochasticParameter):
            self.p = p
        elif ia.is_single_number(p):
            assert 0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,)
            self.p = Deterministic(float(p))
        else:
            raise Exception("Expected StochasticParameter or float/int value, got %s." % (type(p),))

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        assert 0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,)
        return random_state.binomial(1, p, size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.p, float):
            return "Binomial(%.4f)" % (self.p,)
        else:
            return "Binomial(%s)" % (self.p,)

class Choice(StochasticParameter):
    def __init__(self, a, replace=True, p=None):
        super(Choice, self).__init__()

        self.a = a
        self.replace = replace
        self.p = p

    def _draw_samples(self, size, random_state):
        return random_state.choice(self.a, size, replace=self.replace, p=self.p)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Choice(a=%s, replace=%s, p=%s)" % (str(self.a), str(self.replace), str(self.p),)

class DiscreteUniform(StochasticParameter):
    def __init__(self, a, b):
        StochasticParameter.__init__(self)

        # for two ints the samples will be from range a <= x <= b
        assert isinstance(a, (int, StochasticParameter)), "Expected a to be int or StochasticParameter, got %s" % (type(a),)
        assert isinstance(b, (int, StochasticParameter)), "Expected b to be int or StochasticParameter, got %s" % (type(b),)

        if ia.is_single_integer(a):
            self.a = Deterministic(a)
        else:
            self.a = a

        if ia.is_single_integer(b):
            self.b = Deterministic(b)
        else:
            self.b = b

    def _draw_samples(self, size, random_state):
        a = self.a.draw_sample(random_state=random_state)
        b = self.b.draw_sample(random_state=random_state)
        if a > b:
            a, b = b, a
        elif a == b:
            return np.tile(np.array([a]), size)
        return random_state.randint(a, b + 1, size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DiscreteUniform(%s, %s)" % (self.a, self.b)

class Normal(StochasticParameter):
    def __init__(self, loc, scale):
        super(Normal, self).__init__()

        if isinstance(loc, StochasticParameter):
            self.loc = loc
        elif ia.is_single_number(loc):
            self.loc = Deterministic(loc)
        else:
            raise Exception("Expected float, int or StochasticParameter as loc, got %s, %s." % (type(loc),))

        if isinstance(scale, StochasticParameter):
            self.scale = scale
        elif ia.is_single_number(scale):
            assert scale >= 0, "Expected scale to be in range [0, inf) got %s (type %s)." % (scale, type(scale))
            self.scale = Deterministic(scale)
        else:
            raise Exception("Expected float, int or StochasticParameter as scale, got %s, %s." % (type(scale),))

    def _draw_samples(self, size, random_state):
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        assert scale >= 0, "Expected scale to be in rnage [0, inf), got %s." % (scale,)
        if scale == 0:
            return np.tile(loc, size)
        else:
            return random_state.normal(loc, scale, size=size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Normal(loc=%s, scale=%s)" % (self.loc, self.scale)

class Uniform(StochasticParameter):
    def __init__(self, a, b):
        super(Uniform, self).__init__()

        assert isinstance(a, (int, float, StochasticParameter)), "Expected a to be int, float or StochasticParameter, got %s" % (type(a),)
        assert isinstance(b, (int, float, StochasticParameter)), "Expected b to be int, float or StochasticParameter, got %s" % (type(b),)

        if ia.is_single_number(a):
            self.a = Deterministic(a)
        else:
            self.a = a

        if ia.is_single_number(b):
            self.b = Deterministic(b)
        else:
            self.b = b

    def _draw_samples(self, size, random_state):
        a = self.a.draw_sample(random_state=random_state)
        b = self.b.draw_sample(random_state=random_state)
        if a > b:
            a, b = b, a
        elif a == b:
            return np.tile(np.array([a]), size)
        return random_state.uniform(a, b, size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Uniform(%s, %s)" % (self.a, self.b)

class Deterministic(StochasticParameter):
    def __init__(self, value):
        super(Deterministic, self).__init__()

        if isinstance(value, StochasticParameter):
            self.value = value.draw_sample()
        elif ia.is_single_number(value) or ia.is_string(value):
            self.value = value
        else:
            raise Exception("Expected StochasticParameter object or number or string, got %s." % (type(value),))

    def _draw_samples(self, size, random_state):
        return np.tile(np.array([self.value]), size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.value, int):
            return "Deterministic(int %d)" % (self.value,)
        else:
            return "Deterministic(float %.8f)" % (self.value,)

class FromLowerResolution(StochasticParameter):
    def __init__(self, other_param, size_percent=None, size_px=None, method="nearest", min_size=1):
        super(StochasticParameter, self).__init__()

        assert size_percent is not None or size_px is not None

        if size_percent is not None:
            self.size_method = "percent"
            self.size_px = None
            if ia.is_single_number(size_percent):
                self.size_percent = Deterministic(size_percent)
            elif ia.is_iterable(size_percent):
                assert len(size_percent) == 2
                self.size_percent = Uniform(size_percent[0], size_percent[1])
            elif isinstance(size_percent, StochasticParameter):
                self.size_percent = size_percent
            else:
                raise Exception("Expected int, float, tuple of two ints/floats or StochasticParameter for size_percent, got %s." % (type(size_percent),))
        else: # = elif size_px is not None:
            self.size_method = "px"
            self.size_percent = None
            if ia.is_single_integer(size_px):
                self.size_px = Deterministic(size_px)
            elif ia.is_iterable(size_px):
                assert len(size_px) == 2
                self.size_px = DiscreteUniform(size_px[0], size_px[1])
            elif isinstance(size_px, StochasticParameter):
                self.size_px = size_px
            else:
                raise Exception("Expected int, float, tuple of two ints/floats or StochasticParameter for size_px, got %s." % (type(size_px),))

        self.other_param = other_param

        if ia.is_string(method):
            self.method = Deterministic(method)
        elif isinstance(method, StochasticParameter):
            self.method = method
        else:
            raise Exception("Expected string or StochasticParameter, got %s." % (type(method),))

        self.min_size = min_size

    def _draw_samples(self, size, random_state):
        if len(size) == 3:
            n = 1
            h, w, c = size
        elif len(size) == 4:
            n, h, w, c = size
        else:
            raise Exception("FromLowerResolution can only generate samples of shape (H, W, C) or (N, H, W, C), requested was %s." % (str(size),))

        if self.size_method == "percent":
            hw_percents = self.size_percent.draw_samples((n, 2), random_state=random_state)
            hw_pxs = (hw_percents * np.array([h, w])).astype(np.int32)
        else:
            hw_pxs = self.size_px.draw_samples((n, 2), random_state=random_state)

        methods = self.method.draw_samples((n,), random_state=random_state)
        result = None
        #for i, (size_factor, method) in enumerate(zip(size_factors, methods)):
        for i, (hw_px, method) in enumerate(zip(hw_pxs, methods)):
            #h_small = max(int(h * size_factor), self.min_size)
            #w_small = max(int(w * size_factor), self.min_size)
            h_small = max(hw_px[0], self.min_size)
            w_small = max(hw_px[1], self.min_size)
            samples = self.other_param.draw_samples((1, h_small, w_small, c))
            samples_upscaled = ia.imresize_many_images(samples, (h, w), interpolation=method)
            if result is None:
                result = np.zeros((n, h, w, c), dtype=samples.dtype)
            result[i] = samples_upscaled

        if len(size) == 3:
            return result[0]
        else:
            return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.size_method == "percent":
            return "FromLowerResolution(size_percent=%s, method=%s, other_param=%s)" % (self.size_percent, self.method, self.other_param)
        else:
            return "FromLowerResolution(size_px=%s, method=%s, other_param=%s)" % (self.size_px, self.method, self.other_param)

class Clip(StochasticParameter):
    def __init__(self, other_param, minval=None, maxval=None):
        super(Clip, self).__init__()

        assert isinstance(other_param, StochasticParameter)
        assert minval is None or ia.is_single_number(minval)
        assert maxval is None or ia.is_single_number(maxval)

        self.other_param = other_param
        self.minval = minval
        self.maxval = maxval

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(size, random_state=random_state)
        if self.minval is not None and self.maxval is not None:
            np.clip(samples, self.minval, self.maxval, out=samples)
        elif self.minval is not None:
            np.clip(samples, self.minval, np.max(samples), out=samples)
        elif self.maxval is not None:
            np.clip(samples, np.min(samples), self.maxval, out=samples)
        else:
            pass
        return samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        if self.minval is not None and self.maxval is not None:
            return "Clip(%s, %.6f, %.6f)" % (opstr, float(self.minval), float(self.maxval))
        elif self.minval is not None:
            return "Clip(%s, %.6f, None)" % (opstr, float(self.minval))
        elif self.maxval is not None:
            return "Clip(%s, None, %.6f)" % (opstr, float(self.maxval))
        else:
            return "Clip(%s, None, None)" % (opstr,)
