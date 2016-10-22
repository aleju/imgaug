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

    def __init__(self):
        pass

    def transform(self, images):
        if isinstance(images, (list, tuple)):
            return self.transform(np.array(images))
        elif is_np_array(images):
            assert len(images.shape) == 4, "Expected 4d array of form (N, height, width, rgb), got shape %s" % (str(images.shape),)
            assert images.shape[3] == 3, "Expected RGB images, i.e. shape[3] == 3, got shape %s" % (str(images.shape),)
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

class AugmenterSequence(Augmenter):
    def __init__(self, augmenters=None):
        Augmenter.__init__(self)
        self.augmenters = [] if augmenters is None else augmenters

    def _transform(self, images):
        result = images
        for augmenter in self.augmenters:
            result = augmenter._transform(result)
        return result

    #@profile
    def _to_deterministic(self, n):
        seqs = []
        for i in xrange(n):
            augs = [aug.to_deterministic() for aug in self.augmenters]
            seqs.append(AugmenterSequence(augs))
        return seqs

class Fliplr(Augmenter):
    def __init__(self, p=0.5):
        Augmenter.__init__(self)
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
        return [Fliplr(Binomial(sample)) for sample in samples]

class Flipud(Augmenter):
    def __init__(self, p=0.5):
        Augmenter.__init__(self)
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
        return [Flipud(Binomial(sample)) for sample in samples]
