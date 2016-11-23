from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np
import imgaug as ia
import copy as copy_module

try:
    xrange
except NameError:  # python3
    xrange = range

class StochasticParameter(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

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
        StochasticParameter.__init__(self)

        if isinstance(p, StochasticParameter):
            self.p = p
        elif isinstance(p, (float, int)):
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
        StochasticParameter.__init__(self)

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

        if isinstance(a, int):
            self.a = Deterministic(a)
        else:
            self.a = a

        if isinstance(b, int):
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
        StochasticParameter.__init__(self)

        if isinstance(loc, StochasticParameter):
            self.loc = loc
        elif isinstance(loc, (float, int)):
            self.loc = Deterministic(loc)
        else:
            raise Exception("Expected float, int or StochasticParameter as loc, got %s, %s." % (type(loc),))

        if isinstance(scale, StochasticParameter):
            self.scale = scale
        elif isinstance(scale, (float, int)):
            assert scale > 0, "Expected scale to be higher than 0, got %s (type %s)." % (scale, type(scale))
            self.scale = Deterministic(scale)
        else:
            raise Exception("Expected float, int or StochasticParameter as scale, got %s, %s." % (type(scale),))

    def _draw_samples(self, size, random_state):
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        assert scale > 0, "Expected scale to be higher than 0, got %s." % (scale,)
        return random_state.normal(loc, scale, size=size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Normal(loc=%s, scale=%s)" % (self.loc, self.scale)

class Uniform(StochasticParameter):
    def __init__(self, a, b):
        StochasticParameter.__init__(self)

        assert isinstance(a, (int, float, StochasticParameter)), "Expected a to be int, float or StochasticParameter, got %s" % (type(a),)
        assert isinstance(b, (int, float, StochasticParameter)), "Expected b to be int, float or StochasticParameter, got %s" % (type(b),)

        if isinstance(a, (int, float)):
            self.a = Deterministic(a)
        else:
            self.a = a

        if isinstance(b, (int, float)):
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
        StochasticParameter.__init__(self)

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

class Clip(StochasticParameter):
    def __init__(self, other_param, minval=None, maxval=None):
        StochasticParameter.__init__(self)

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
