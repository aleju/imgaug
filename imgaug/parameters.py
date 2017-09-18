from __future__ import print_function, division, absolute_import
from . import imgaug as ia
from .external.opensimplex import OpenSimplex
from abc import ABCMeta, abstractmethod
import numpy as np
import copy as copy_module
import six
import six.moves as sm
import scipy

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
            samples = self.other_param.draw_samples((1, h_small, w_small, c), random_state=random_state)
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

class Multiply(StochasticParameter):
    def __init__(self, other_param, val):
        super(Multiply, self).__init__()

        assert isinstance(other_param, StochasticParameter)
        assert ia.is_single_number(val)

        self.other_param = other_param
        self.val = val

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(size, random_state=random_state)
        return samples * self.val

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "Multiply(%s, %s)" % (opstr, str(self.val))

# TODO this always aggregates the result in high resolution space,
# instead of aggregating them in low resolution and then only upscaling the
# final image (for N iterations that would save up to N-1 upscales)
class IterativeNoiseAggregator(StochasticParameter):
    def __init__(self, other_param, iterations=(1, 3), aggregation_method=["max", "avg"]):
        assert isinstance(other_param, StochasticParameter)
        self.other_param = other_param

        if ia.is_single_integer(iterations):
            assert 1 <= iterations <= 1000
            self.iterations = Deterministic(iterations)
        elif ia.is_iterable(iterations):
            assert len(iterations) == 2
            assert all([ia.is_single_integer(val) for val in iterations])
            assert all([1 <= val <= 10000 for val in iterations])
            self.iterations = DiscreteUniform(iterations[0], iterations[1])
        elif ia.is_iterable(iterations):
            assert len(iterations) > 0
            assert all([1 <= val <= 10000 for val in iterations])
            self.iterations = Choice(iterations)
        elif isinstance(iterations, StochasticParameter):
            self.iterations = iterations
        else:
            raise Exception("Expected iterations to be int or tuple of two ints or StochasticParameter, got %s." % (type(iterations),))

        if aggregation_method == ia.ALL:
            self.aggregation_method = Choice(["min", "max", "avg"])
        elif ia.is_string(aggregation_method):
            self.aggregation_method = Deterministic(aggregation_method)
        elif isinstance(aggregation_method, list):
            assert len(aggregation_method) >= 1
            assert all([ia.is_string(val) for val in aggregation_method])
            self.aggregation_method = Choice(aggregation_method)
        elif isinstance(aggregation_method, StochasticParameter):
            self.aggregation_method = aggregation_method
        else:
            raise Exception("Expected aggregation_method to be string or list of strings or StochasticParameter, got %s." % (type(aggregation_method),))

    def _draw_samples(self, size, random_state):
        assert len(size) == 2, "Expected requested other_param to have shape (H, W), got shape %s." % (size,)
        h, w = size

        seed = random_state.randint(0, 10**6)
        aggregation_method = self.aggregation_method.draw_sample(random_state=ia.new_random_state(seed))
        iterations = self.iterations.draw_sample(random_state=ia.new_random_state(seed+1))
        assert iterations > 0

        result = np.zeros((h, w), dtype=np.float32)
        for i in sm.xrange(iterations):
            noise_iter = self.other_param.draw_samples((h, w), random_state=ia.new_random_state(seed+2+i))
            if aggregation_method == "avg":
                result += noise_iter
            elif aggregation_method == "min":
                if i == 0:
                    result = noise_iter
                else:
                    result = np.minimum(result, noise_iter)
            else: # self.aggregation_method == "max"
                if i == 0:
                    result = noise_iter
                else:
                    result = np.maximum(result, noise_iter)

        if aggregation_method == "avg":
            result = result / iterations

        return result

class Sigmoid(StochasticParameter):
    def __init__(self, other_param, threshold=(-10, 10), activated=True, mul=1, add=0):
        assert isinstance(other_param, StochasticParameter)
        self.other_param = other_param

        if ia.is_single_number(threshold):
            self.threshold = Deterministic(threshold)
        elif isinstance(threshold, tuple):
            assert len(threshold) == 2
            assert all([ia.is_single_number(val) for val in threshold])
            self.threshold = Uniform(threshold[0], threshold[1])
        elif ia.is_iterable(threshold):
            assert len(threshold) > 0
            self.threshold = Choice(threshold)
        elif isinstance(threshold, StochasticParameter):
            self.threshold = threshold
        else:
            raise Exception("Expected threshold to be number or tuple of two numbers or StochasticParameter, got %s." % (type(threshold),))

        if activated in [True, False, 0, 1, 0.0, 1.0]:
            self.activated = Deterministic(int(activated))
        elif ia.is_single_number(activated):
            assert 0 <= activated <= 1.0
            self.activated = Binomial(activated)
        else:
            raise Exception("Expected activated to be boolean or number or StochasticParameter, got %s." % (type(activated),))

        assert ia.is_single_number(mul)
        assert mul > 0
        self.mul = mul

        assert ia.is_single_number(add)
        self.add = add

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6)
        result = self.other_param.draw_samples(size, random_state=ia.new_random_state(seed))
        activated = self.activated.draw_sample(random_state=ia.new_random_state(seed+1))
        threshold = self.threshold.draw_sample(random_state=ia.new_random_state(seed+2))
        if activated > 0.5:
            # threshold must be subtracted here, not added
            # higher threshold = move threshold of sigmoid towards the right
            #                  = make it harder to pass the threshold
            #                  = more 0.0s / less 1.0s
            # by subtracting a high value, it moves each x towards the left,
            # leading to more values being left of the threshold, leading
            # to more 0.0s
            return 1 / (1 + np.exp(-(result * self.mul + self.add - threshold)))
        else:
            return result

"""
class SimplexNoise(StochasticParameter):
    def __init__(self, iterations=(1, 3), size_px_max=(2, 16), upscale_method=["linear", "nearest"], aggregation_method=["max", "avg"], sigmoid=0.5, sigmoid_thresh=(-10, 10)):
        if ia.is_single_integer(iterations):
            assert 1 <= iterations <= 1000
            self.iterations = Deterministic(iterations)
        elif ia.is_iterable(iterations):
            assert len(iterations) == 2
            assert all([ia.is_single_integer(val) for val in iterations])
            assert all([1 <= val <= 10000 for val in iterations])
            self.iterations = DiscreteUniform(iterations[0], iterations[1])
        elif ia.is_iterable(iterations):
            assert len(iterations) > 0
            assert all([1 <= val <= 10000 for val in iterations])
            self.iterations = Choice(iterations)
        elif isinstance(iterations, StochasticParameter):
            self.iterations = iterations
        else:
            raise Exception("Expected iterations to be int or tuple of two ints or StochasticParameter, got %s." % (type(iterations),))

        if ia.is_single_integer(size_px_max):
            assert 1 <= size_px_max <= 10000
            self.size_px_max = Deterministic(size_px_max)
        elif isinstance(size_px_max, tuple):
            assert len(size_px_max) == 2
            assert all([ia.is_single_integer(val) for val in size_px_max])
            assert all([1 <= val <= 10000 for val in size_px_max])
            self.size_px_max = DiscreteUniform(size_px_max[0], size_px_max[1])
        elif ia.is_iterable(size_px_max):
            assert len(size_px_max) > 0
            assert all([1 <= val <= 10000 for val in size_px_max])
            self.size_px_max = Choice(size_px_max)
        elif isinstance(size_px_max, StochasticParameter):
            self.size_px_max = size_px_max
        else:
            raise Exception("Expected size_px_max to be int or tuple of two ints or StochasticParameter, got %s." % (type(size_px_max),))

        if upscale_method == ia.ALL:
            self.upscale_method = Choice(["nearest", "linear", "area", "cubic"])
        elif ia.is_string(upscale_method):
            self.upscale_method = Deterministic(upscale_method)
        elif isinstance(upscale_method, list):
            assert len(upscale_method) >= 1
            assert all([ia.is_string(val) for val in upscale_method])
            self.upscale_method = Choice(upscale_method)
        elif isinstance(upscale_method, StochasticParameter):
            self.upscale_method = upscale_method
        else:
            raise Exception("Expected upscale_method to be string or list of strings or StochasticParameter, got %s." % (type(upscale_method),))

        if aggregation_method == ia.ALL:
            self.aggregation_method = Choice(["min", "max", "avg"])
        elif ia.is_string(aggregation_method):
            self.aggregation_method = Deterministic(aggregation_method)
        elif isinstance(aggregation_method, list):
            assert len(aggregation_method) >= 1
            assert all([ia.is_string(val) for val in aggregation_method])
            self.aggregation_method = Choice(aggregation_method)
        elif isinstance(aggregation_method, StochasticParameter):
            self.aggregation_method = aggregation_method
        else:
            raise Exception("Expected aggregation_method to be string or list of strings or StochasticParameter, got %s." % (type(aggregation_method),))

        if sigmoid in [True, False, 0, 1, 0.0, 1.0]:
            self.sigmoid = Deterministic(int(sigmoid))
        elif ia.is_single_number(sigmoid):
            assert 0 <= sigmoid <= 1.0
            self.sigmoid = Binomial(sigmoid)
        else:
            raise Exception("Expected sigmoid to be boolean or number or StochasticParameter, got %s." % (type(sigmoid),))

        if ia.is_single_number(sigmoid_thresh):
            self.sigmoid_thresh = Deterministic(sigmoid_thresh)
        elif isinstance(sigmoid_thresh, tuple):
            assert len(sigmoid_thresh) == 2
            assert all([ia.is_single_number(val) for val in sigmoid_thresh])
            self.sigmoid_thresh = Uniform(sigmoid_thresh[0], sigmoid_thresh[1])
        elif ia.is_iterable(sigmoid_thresh):
            assert len(sigmoid_thresh) > 0
            self.sigmoid_thresh = Choice(sigmoid_thresh)
        elif isinstance(sigmoid_thresh, StochasticParameter):
            self.sigmoid_thresh = sigmoid_thresh
        else:
            raise Exception("Expected sigmoid_thresh to be number or tuple of two numbers or StochasticParameter, got %s." % (type(sigmoid_thresh),))

    def _draw_samples(self, size, random_state):
        assert len(size) == 2, "Expected requested noise to have shape (H, W), got shape %s." % (size,)
        h, w = size
        seed = random_state.randint(0, 10**6)
        aggregation_method = self.aggregation_method.draw_sample(random_state=ia.new_random_state(seed))
        iterations = self.iterations.draw_sample(random_state=ia.new_random_state(seed+1))
        upscale_methods = self.upscale_method.draw_samples((iterations,), random_state=ia.new_random_state(seed+2))
        result = np.zeros((h, w), dtype=np.float32)
        for i in sm.xrange(iterations):
            noise_iter = self._draw_samples_iteration(h, w, seed + 10 + i, upscale_methods[i])
            if aggregation_method == "avg":
                result += noise_iter
            elif aggregation_method == "min":
                if i == 0:
                    result = noise_iter
                else:
                    result = np.minimum(result, noise_iter)
            else: # self.aggregation_method == "max"
                if i == 0:
                    result = noise_iter
                else:
                    result = np.maximum(result, noise_iter)

        if aggregation_method == "avg":
            result = result / iterations

        sigmoid = self.sigmoid.draw_sample(random_state=ia.new_random_state(seed+3))
        sigmoid_thresh = self.sigmoid_thresh.draw_sample(random_state=ia.new_random_state(seed+4))
        if sigmoid > 0.5:
            # yes, threshold must be subtracted here, not added
            # higher threshold = move threshold of sigmoid towards the right
            #                  = make it harder to pass the threshold
            #                  = more 0.0s / less 1.0s
            # by subtracting a high value, it moves each x towards the left,
            # leading to more values being left of the threshold, leading
            # to more 0.0s
            result = 1 / (1 + np.exp(-(result * 20 - 10 - sigmoid_thresh)))

        #from scipy import misc
        #misc.imshow((result * 255).astype(np.uint8))

        return result

    def _draw_samples_iteration(self, h, w, seed, upscale_method):
        maxlen = max(h, w)
        size_px_max = self.size_px_max.draw_sample(random_state=ia.new_random_state(seed))
        if maxlen > size_px_max:
            downscale_factor = size_px_max / maxlen
            h_small = int(h * downscale_factor)
            w_small = int(w * downscale_factor)
        else:
            h_small = h
            w_small = w

        # don't go below Hx1 or 1xW
        h_small = max(h_small, 1)
        w_small = max(w_small, 1)

        generator = OpenSimplex(seed=seed)
        noise = np.zeros((h_small, w_small), dtype=np.float32)
        for y in sm.xrange(h_small):
            for x in sm.xrange(w_small):
                noise[y, x] = generator.noise2d(y=y, x=x)
        noise_0to1 = (noise + 0.5) / 2

        if noise_0to1.shape != (h, w):
            noise_0to1_uint8 = (noise_0to1 * 255).astype(np.uint8)
            noise_0to1_3d = np.tile(noise_0to1_uint8[..., np.newaxis], (1, 1, 3))
            noise_0to1 = ia.imresize_single_image(noise_0to1_3d, (h, w), interpolation=upscale_method)
            noise_0to1 = (noise_0to1[..., 0] / 255.0).astype(np.float32)

        #from scipy import misc
        #print(noise_0to1.shape, h_small, w_small, self.size_percent, self.size_px_max, maxlen)
        #misc.imshow((noise_0to1 * 255).astype(np.uint8))

        return noise_0to1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "SimplexNoise(%s, %s, %s, %s, %s, %s, %s)" % (
            str(self.iterations),
            str(self.size_px_max),
            str(self.upscale_method),
            str(self.aggregation_method),
            str(self.sigmoid),
            str(self.sigmoid_thresh)
        )
"""

class SimplexNoise(StochasticParameter):
    def __init__(self, size_px_max=(2, 16), upscale_method=["linear", "nearest"]):
        if ia.is_single_integer(size_px_max):
            assert 1 <= size_px_max <= 10000
            self.size_px_max = Deterministic(size_px_max)
        elif isinstance(size_px_max, tuple):
            assert len(size_px_max) == 2
            assert all([ia.is_single_integer(val) for val in size_px_max])
            assert all([1 <= val <= 10000 for val in size_px_max])
            self.size_px_max = DiscreteUniform(size_px_max[0], size_px_max[1])
        elif ia.is_iterable(size_px_max):
            assert len(size_px_max) > 0
            assert all([1 <= val <= 10000 for val in size_px_max])
            self.size_px_max = Choice(size_px_max)
        elif isinstance(size_px_max, StochasticParameter):
            self.size_px_max = size_px_max
        else:
            raise Exception("Expected size_px_max to be int or tuple of two ints or StochasticParameter, got %s." % (type(size_px_max),))

        if upscale_method == ia.ALL:
            self.upscale_method = Choice(["nearest", "linear", "area", "cubic"])
        elif ia.is_string(upscale_method):
            self.upscale_method = Deterministic(upscale_method)
        elif isinstance(upscale_method, list):
            assert len(upscale_method) >= 1
            assert all([ia.is_string(val) for val in upscale_method])
            self.upscale_method = Choice(upscale_method)
        elif isinstance(upscale_method, StochasticParameter):
            self.upscale_method = upscale_method
        else:
            raise Exception("Expected upscale_method to be string or list of strings or StochasticParameter, got %s." % (type(upscale_method),))

    def _draw_samples(self, size, random_state):
        assert len(size) == 2, "Expected requested noise to have shape (H, W), got shape %s." % (size,)
        h, w = size
        seed = random_state.randint(0, 10**6)
        iterations = 1
        aggregation_method = "max"
        upscale_methods = self.upscale_method.draw_samples((iterations,), random_state=ia.new_random_state(seed))
        result = np.zeros((h, w), dtype=np.float32)
        for i in sm.xrange(iterations):
            noise_iter = self._draw_samples_iteration(h, w, seed + 10 + i, upscale_methods[i])
            if aggregation_method == "avg":
                result += noise_iter
            elif aggregation_method == "min":
                if i == 0:
                    result = noise_iter
                else:
                    result = np.minimum(result, noise_iter)
            else: # self.aggregation_method == "max"
                if i == 0:
                    result = noise_iter
                else:
                    result = np.maximum(result, noise_iter)

        if aggregation_method == "avg":
            result = result / iterations

        return result

    def _draw_samples_iteration(self, h, w, seed, upscale_method):
        maxlen = max(h, w)
        size_px_max = self.size_px_max.draw_sample(random_state=ia.new_random_state(seed))
        if maxlen > size_px_max:
            downscale_factor = size_px_max / maxlen
            h_small = int(h * downscale_factor)
            w_small = int(w * downscale_factor)
        else:
            h_small = h
            w_small = w

        # don't go below Hx1 or 1xW
        h_small = max(h_small, 1)
        w_small = max(w_small, 1)

        generator = OpenSimplex(seed=seed)
        noise = np.zeros((h_small, w_small), dtype=np.float32)
        for y in sm.xrange(h_small):
            for x in sm.xrange(w_small):
                noise[y, x] = generator.noise2d(y=y, x=x)
        noise_0to1 = (noise + 0.5) / 2

        if noise_0to1.shape != (h, w):
            noise_0to1_uint8 = (noise_0to1 * 255).astype(np.uint8)
            noise_0to1_3d = np.tile(noise_0to1_uint8[..., np.newaxis], (1, 1, 3))
            noise_0to1 = ia.imresize_single_image(noise_0to1_3d, (h, w), interpolation=upscale_method)
            noise_0to1 = (noise_0to1[..., 0] / 255.0).astype(np.float32)

        #from scipy import misc
        #print(noise_0to1.shape, h_small, w_small, self.size_percent, self.size_px_max, maxlen)
        #misc.imshow((noise_0to1 * 255).astype(np.uint8))

        return noise_0to1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "SimplexNoise(%s, %s)" % (
            str(self.size_px_max),
            str(self.upscale_method)
        )

class FrequencyNoise(StochasticParameter):
    def __init__(self, exponent=(-4, 4), size_px_max=(4, 32), upscale_method=["linear", "nearest"]):
        if ia.is_single_number(exponent):
            self.exponent = Deterministic(exponent)
        elif isinstance(exponent, tuple):
            assert len(exponent) == 2
            assert all([ia.is_single_number(val) for val in exponent])
            self.exponent = Uniform(exponent[0], exponent[1])
        elif ia.is_iterable(exponent):
            assert len(exponent) > 0
            self.exponent = Choice(exponent)
        elif isinstance(exponent, StochasticParameter):
            self.exponent = exponent
        else:
            raise Exception("Expected exponent to be int or tuple of two ints or StochasticParameter, got %s." % (type(exponent),))

        if ia.is_single_integer(size_px_max):
            assert 1 <= size_px_max <= 10000
            self.size_px_max = Deterministic(size_px_max)
        elif isinstance(size_px_max, tuple):
            assert len(size_px_max) == 2
            assert all([ia.is_single_integer(val) for val in size_px_max])
            assert all([1 <= val <= 10000 for val in size_px_max])
            self.size_px_max = DiscreteUniform(size_px_max[0], size_px_max[1])
        elif ia.is_iterable(sigmoid_thresh):
            assert len(size_px_max) > 0
            assert all([1 <= val <= 10000 for val in size_px_max])
            self.size_px_max = Choice(size_px_max)
        elif isinstance(size_px_max, StochasticParameter):
            self.size_px_max = size_px_max
        else:
            raise Exception("Expected size_px_max to be int or tuple of two ints or StochasticParameter, got %s." % (type(size_px_max),))

        if upscale_method == ia.ALL:
            self.upscale_method = Choice(["nearest", "linear", "area", "cubic"])
        elif ia.is_string(upscale_method):
            self.upscale_method = Deterministic(upscale_method)
        elif isinstance(upscale_method, list):
            assert len(upscale_method) >= 1
            assert all([ia.is_string(val) for val in upscale_method])
            self.upscale_method = Choice(upscale_method)
        elif isinstance(upscale_method, StochasticParameter):
            self.upscale_method = upscale_method
        else:
            raise Exception("Expected upscale_method to be string or list of strings or StochasticParameter, got %s." % (type(upscale_method),))

    def _draw_samples(self, size, random_state):
        # code here is similar to:
        #   http://www.redblobgames.com/articles/noise/2d/
        #   http://www.redblobgames.com/articles/noise/2d/2d-noise.js

        assert len(size) == 2, "Expected requested noise to have shape (H, W), got shape %s." % (size,)

        seed = random_state.randint(0, 10**6)

        h, w = size
        maxlen = max(h, w)
        size_px_max = self.size_px_max.draw_sample(random_state=ia.new_random_state(seed))
        if maxlen > size_px_max:
            downscale_factor = size_px_max / maxlen
            h_small = int(h * downscale_factor)
            w_small = int(w * downscale_factor)
        else:
            h_small = h
            w_small = w

        # don't go below Hx4 or 4xW
        h_small = max(h_small, 4)
        w_small = max(w_small, 4)

        # generate random base matrix
        wn_r = ia.new_random_state(seed+1).rand(h_small, w_small)
        wn_a = ia.new_random_state(seed+2).rand(h_small, w_small)

        wn_r = wn_r * (max(h_small, w_small) ** 2)
        wn_a = wn_a * 2 * np.pi

        wn_r = wn_r * np.cos(wn_a)
        wn_a = wn_r * np.sin(wn_a)

        # pronounce some frequencies
        exponent = self.exponent.draw_sample(random_state=ia.new_random_state(seed+3))
        # this has some similarity with a distance map from the center, but looks a bit more like a cross
        f = self._create_distance_matrix((h_small, w_small))
        f[0, 0] = 1 # necessary to prevent -inf from appearing
        scale = f ** exponent
        scale[0, 0] = 0
        tr = wn_r * scale
        ti = wn_a * scale

        """
        Fmin = 1
        Fmax = 64
        tr = np.zeros(wn_r.shape, dtype=np.float32)
        ti = np.zeros(wn_r.shape, dtype=np.float32)
        for i in range(h_small):
            for j in range(w_small):
                if i==0 and j==0:
                    continue
                f1 = min(i, h_small-i)
                f2 = min(j, w_small-j)
                f = np.sqrt(f1**2 + f2**2)
                #scale = (Fmin <= f <= Fmax) * (f**(2*-2))
                scale = (f**(2*-2))
                x = wn_r[i, j] * scale
                y = wn_a[i, j] * scale
                tr[i, j] = x
                ti[i, j] = y
        """

        wn_freqs_mul = np.zeros(tr.shape, dtype=np.complex)
        wn_freqs_mul.real = tr
        wn_freqs_mul.imag = ti

        wn_inv = np.fft.ifft2(wn_freqs_mul).real

        # normalize to 0 to 1
        wn_inv_min = np.min(wn_inv)
        wn_inv_max = np.max(wn_inv)
        noise_0to1 = (wn_inv - wn_inv_min) / (wn_inv_max - wn_inv_min)

        # upscale from low resolution to image size
        upscale_method = self.upscale_method.draw_sample(random_state=ia.new_random_state(seed+1))
        if noise_0to1.shape != (size[0], size[1]):
            noise_0to1_uint8 = (noise_0to1 * 255).astype(np.uint8)
            noise_0to1_3d = np.tile(noise_0to1_uint8[..., np.newaxis], (1, 1, 3))
            noise_0to1 = ia.imresize_single_image(noise_0to1_3d, (size[0], size[1]), interpolation=upscale_method)
            noise_0to1 = (noise_0to1[..., 0] / 255.0).astype(np.float32)

        return noise_0to1

    def _create_distance_matrix(self, size):
        h, w = size
        def freq(yy, xx):
            f1 = np.minimum(yy, h-yy)
            f2 = np.minimum(xx, w-xx)
            return np.sqrt(f1**2 + f2**2)
        return scipy.fromfunction(freq, (h, w))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "FrequencyNoise(%s, %s, %s)" % (
            str(self.exponent),
            str(self.size_px_max),
            str(self.upscale_method)
        )
