from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy

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

class DiscreteUniform(StochasticParameter):
    def __init__(self, a, b=None):
        if isinstance(a, DiscreteUniform):
            self.a = a.a
            self.b = a.b
        elif isinstance(a, int):
            assert b is not None and isinstance(b, int)
            assert a <= b
            self.a = a
            self.b = b
        else:
            raise Exception("Expected two int values or DiscreteUniform object, got %s, %s." % (type(a), type(b)))

    def draw_samples(self, n, random_state=None):
        rng = random_state if random_state is not None else np.random
        return rng.randint(self.a, self.b, n)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DiscreteUniform(%d, %d)" % (self.a, self.b)

class Normal(StochasticParameter):
    def __init__(self, mean, std=None):
        if isinstance(mean, Normal):
            self.mean = mean.mean
            self.std = mean.std
        elif isinstance(mean, (float, int)):
            assert std is not None and isinstance(std, (float, int))
            assert std >= 0
            self.mean = mean
            self.std = std
        else:
            raise Exception("Expected two float/int values or Normal object, got %s, %s." % (type(mean), type(std)))

    def draw_samples(self, n, random_state=None):
        rng = random_state if random_state is not None else np.random
        return rng.normal(self.mean, self.std, size=n)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Normal(mean=%.6f, std=%.6f)" % (self.mean, self.std)

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
            raise Exception("Expected two float/int values or Uniform object, got %s, %s." % (type(a), type(b)))

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

class Clip(StochasticParameter):
    def __init__(self, other_param, minval=None, maxval=None):
        self.other_param = other_param
        self.minval = minval
        self.maxval = maxval

    def draw_samples(self, n, random_state=None):
        samples = self.other_param.draw_samples(n, random_state=random_state)
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
        opstr = self.other_param.__str()__
        if self.minval is not None and self.maxval is not None:
            return "Clip(%s, %.6f, %.6f)" % (opstr, float(self.minval), float(self.maxval))
        elif self.minval is not None:
            return "Clip(%s, %.6f, None)" % (opstr, float(self.minval))
        elif self.maxval is not None:
            return "Clip(%s, None, %.6f)" % (opstr, float(self.maxval))
        else:
            return "Clip(%s, None, None)" % (opstr,)
