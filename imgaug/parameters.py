from __future__ import print_function, division, absolute_import
from . import imgaug as ia
from .external.opensimplex import OpenSimplex
from abc import ABCMeta, abstractmethod
import numpy as np
import copy as copy_module
import six
import six.moves as sm
import scipy
import numbers
from collections import defaultdict
from scipy import misc

NP_FLOAT_TYPES = set(np.sctypes["float"])

def handle_continuous_param(param, name, value_range=None, tuple_to_uniform=True, list_to_choice=True):
    def check_value_range(v):
        if value_range is None:
            return True
        elif isinstance(value_range, tuple):
            ia.do_assert(len(value_range) == 2)
            if value_range[0] is None and value_range[1] is None:
                return True
            elif value_range[0] is None:
                ia.do_assert(v <= value_range[1], "Parameter '%s' is outside of the expected value range (x <= %.4f)" % (name, value_range[1]))
                return True
            elif value_range[1] is None:
                ia.do_assert(value_range[0] <= v, "Parameter '%s' is outside of the expected value range (%.4f <= x)" % (name, value_range[0]))
                return True
            else:
                ia.do_assert(value_range[0] <= v <= value_range[1], "Parameter '%s' is outside of the expected value range (%.4f <= x <= %.4f)" % (name, value_range[0], value_range[1]))
                return True
        elif ia.is_callable(value_range):
            value_range(v)
            return True
        else:
            raise Exception("Unexpected input for value_range, got %s." % (str(value_range),))

    if ia.is_single_number(param):
        check_value_range(param)
        return Deterministic(param)
    elif tuple_to_uniform and isinstance(param, tuple):
        ia.do_assert(len(param) == 2)
        check_value_range(param[0])
        check_value_range(param[1])
        return Uniform(param[0], param[1])
    elif list_to_choice and ia.is_iterable(param):
        for param_i in param:
            check_value_range(param_i)
        return Choice(param)
    elif isinstance(param, StochasticParameter):
        return param
    else:
        raise Exception("Expected number, tuple of two number, list of number or StochasticParameter for %s, got %s." % (name, type(param),))

def handle_discrete_param(param, name, value_range=None, tuple_to_uniform=True, list_to_choice=True, allow_floats=True):
    def check_value_range(v):
        if value_range is None:
            return True
        elif isinstance(value_range, tuple):
            ia.do_assert(len(value_range) == 2)
            if value_range[0] is None and value_range[1] is None:
                return True
            elif value_range[0] is None:
                ia.do_assert(v <= value_range[1], "Parameter '%s' is outside of the expected value range (x <= %.4f)" % (name, value_range[1]))
                return True
            elif value_range[1] is None:
                ia.do_assert(value_range[0] <= v, "Parameter '%s' is outside of the expected value range (%.4f <= x)" % (name, value_range[0]))
                return True
            else:
                ia.do_assert(value_range[0] <= v <= value_range[1], "Parameter '%s' is outside of the expected value range (%.4f <= x <= %.4f)" % (name, value_range[0], value_range[1]))
                return True
        elif ia.is_callable(value_range):
            value_range(v)
            return True
        else:
            raise Exception("Unexpected input for value_range, got %s." % (str(value_range),))

    if ia.is_single_integer(param) or (allow_floats and ia.is_single_float(param)):
        check_value_range(param)
        return Deterministic(int(param))
    elif tuple_to_uniform and isinstance(param, tuple):
        ia.do_assert(len(param) == 2)
        if allow_floats:
            ia.do_assert(ia.is_single_number(param[0]), "Expected number, got %s." % (type(param[0]),))
            ia.do_assert(ia.is_single_number(param[1]), "Expected number, got %s." % (type(param[1]),))
        else:
            ia.do_assert(ia.is_single_integer(param[0]), "Expected integer, got %s." % (type(param[0]),))
            ia.do_assert(ia.is_single_integer(param[1]), "Expected integer, got %s." % (type(param[1]),))
        check_value_range(param[0])
        check_value_range(param[1])
        return DiscreteUniform(int(param[0]), int(param[1]))
    elif list_to_choice and ia.is_iterable(param):
        for param_i in param:
            check_value_range(param_i)
        return Choice([int(param_i) for param_i in param])
    elif isinstance(param, StochasticParameter):
        return param
    else:
        if allow_floats:
            raise Exception("Expected number, tuple of two number, list of number or StochasticParameter for %s, got %s." % (name, type(param),))
        else:
            raise Exception("Expected int, tuple of two int, list of int or StochasticParameter for %s, got %s." % (name, type(param),))

def force_np_float_dtype(val):
    if val.dtype in NP_FLOAT_TYPES:
        return val
    else:
        return val.astype(np.float32)

def both_np_float_if_one_is_float(a, b):
    a_f = a.dtype in NP_FLOAT_TYPES
    b_f = b.dtype in NP_FLOAT_TYPES
    if a_f and b_f:
        return a, b
    elif a_f:
        return a, b.astype(np.float32)
    elif b_f:
        return a.astype(np.float32), b
    else:
        return a.astype(np.float32), b.astype(np.float32)

def draw_distributions_grid(params, rows=None, cols=None, graph_sizes=(350, 350), sample_sizes=None, titles=None):
    if titles is None:
        titles = [None] * len(params)
    elif titles == False:
        titles = [False] * len(params)

    if sample_sizes is not None:
        images = [param_i.draw_distribution_graph(size=size_i, title=title_i) for param_i, size_i, title_i in zip(params, sample_sizes, titles)]
    else:
        images = [param_i.draw_distribution_graph(title=title_i) for param_i, title_i in zip(params, titles)]

    images_rs = ia.imresize_many_images(np.array(images), sizes=graph_sizes)
    grid = ia.draw_grid(images_rs, rows=rows, cols=cols)
    return grid

def show_distributions_grid(params, rows=None, cols=None, graph_sizes=(350, 350), sample_sizes=None, titles=None):
    misc.imshow(
        draw_distributions_grid(
            params,
            graph_sizes=graph_sizes,
            sample_sizes=sample_sizes,
            rows=rows,
            cols=cols,
            titles=titles
        )
    )

@six.add_metaclass(ABCMeta)
class StochasticParameter(object): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Abstract parent class for all stochastic parameters.

    Stochastic parameters are here all parameters from which values are
    supposed to be sampled. Usually the sampled values are to a degree random.
    E.g. a stochastic parameter may be the range [-10, 10], with sampled
    values being 5.2, -3.7, -9.7 and 6.4.

    """

    def __init__(self):
        super(StochasticParameter, self).__init__()

    def draw_sample(self, random_state=None):
        """
        Draws a single sample value from this parameter.

        Parameters
        ----------
        random_state : None or np.random.RandomState, optional(default=None)
            A random state to use during the sampling process.
            If None, the libraries global random state will be used.

        Returns
        -------
        out : anything
            A single sample value.

        """
        return self.draw_samples(1, random_state=random_state)[0]

    def draw_samples(self, size, random_state=None):
        """
        Draws one or more sample values from the parameter.

        Parameters
        ----------
        size : tuple of int
            Number of sample values by
            dimension.

        random_state : None or np.random.RandomState, optional(default=None)
            A random state to use during the sampling process.
            If None, the libraries global random state will be used.

        Returns
        -------
        out : (size) iterable
            Sampled values. Usually a numpy ndarray of basically any dtype,
            though not strictly limited to numpy arrays.

        """
        random_state = random_state if random_state is not None else ia.current_random_state()
        samples = self._draw_samples(size, random_state)
        ia.forward_random_state(random_state)

        return samples

    @abstractmethod
    def _draw_samples(self, size, random_state):
        raise NotImplementedError()

    def __add__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Add(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter + %s. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __sub__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Subtract(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter - %s. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __mul__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Multiply(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter * %s. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __pow__(self, other, z=None):
        if z is not None:
            raise NotImplementedError("Modulo power is currently not supported by StochasticParameter.")
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Power(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter ** %s. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __div__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter / %s. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __truediv__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter / %s (truediv). Expected second argument to be number or StochasticParameter." % (type(other),))

    def __radd__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Add(other, self)
        else:
            raise Exception("Invalid datatypes in: %s + StochasticParameter. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __rsub__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Subtract(other, self)
        else:
            raise Exception("Invalid datatypes in: %s - StochasticParameter. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __rmul__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Multiply(other, self)
        else:
            raise Exception("Invalid datatypes in: %s * StochasticParameter. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __rpow__(self, other, z=None):
        if z is not None:
            raise NotImplementedError("Modulo power is currently not supported by StochasticParameter.")
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Power(other, self)
        else:
            raise Exception("Invalid datatypes in: %s ** StochasticParameter. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __rdiv__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(other, self)
        else:
            raise Exception("Invalid datatypes in: %s / StochasticParameter. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __rtruediv__(self, other):
        if ia.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(other, self)
        else:
            raise Exception("Invalid datatypes in: %s / StochasticParameter (truediv). Expected second argument to be number or StochasticParameter." % (type(other),))

    def copy(self):
        """
        Create a shallow copy of this parameter.

        Returns
        -------
        out : StochasticParameter
            Shallow copy.

        """
        return copy_module.copy(self)

    def deepcopy(self):
        """
        Create a deep copy of this parameter.

        Returns
        -------
        out : StochasticParameter
            Deep copy.

        """
        return copy_module.deepcopy(self)

    def draw_distribution_graph(self, title=None, size=(1000, 1000), bins=100):
        """
        Generate a plot (image) that shows the parameter's distribution of
        values.

        Parameters
        ----------
        title : None or False or string, optional(default=None)
            Title of the plot. None is automatically replaced by a title
            derived from `str(param)`. If set to False, no title will be
            shown.

        size : tuple of int
            Number of points to sample. This is always expected to have at
            least two values. The first defines the number of sampling runs,
            the second (and further) dimensions define the size assigned
            to each `draw_samples()` call. E.g. `(10, 20, 15)` will lead
            to `10` calls of `draw_samples(size=(20, 15))`. The results
            will be merged to a single 1d array.

        bins : int
            Number of bins in the plot histograms.

        Returns
        -------
        data : (H,W,3) ndarray
            Image of the plot.

        """
        import matplotlib.pyplot as plt

        points = []
        for _ in sm.xrange(size[0]):
            points.append(self.draw_samples(size[1:]).flatten())
        points = np.concatenate(points)

        fig = plt.figure()
        fig.add_subplot(111)
        ax = fig.gca()
        heights, bins = np.histogram(points, bins=bins)
        heights = heights / sum(heights)
        ax.bar(
            bins[:-1],
            heights,
            width=(max(bins) - min(bins))/len(bins),
            color="blue",
            alpha=0.75
        )
        #print("[draw_distribution_graph] points", points[0:100])
        #print("[draw_distribution_graph] min/max/avg", np.min(points), np.max(points), np.average(points))
        #print("[draw_distribution_graph] bins", len(bins), bins[0:10], heights[0:10])

        if title is None:
            title = str(self)
        if title != False:
            # split long titles - otherwise matplotlib generates errors
            title_fragments = [title[i:i+50] for i in sm.xrange(0, len(title), 50)]
            ax.set_title("\n".join(title_fragments))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

class Binomial(StochasticParameter):
    """
    Binomial distribution.

    Parameters
    ----------
    p : number or tuple of two number or list of number or StochasticParameter
        Probability of the binomial distribution. Expected to be in the
        range [0, 1]. If this is a StochasticParameter, the value will be
        sampled once per call to _draw_samples().

    Examples
    --------
    >>> param = Binomial(Uniform(0.01, 0.2))

    Uses a varying probability `p` between 0.01 and 0.2 per sampling.

    """

    def __init__(self, p):
        super(Binomial, self).__init__()

        """
        if isinstance(p, StochasticParameter):
            self.p = p
        elif ia.is_single_number(p):
            ia.do_assert(0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,))
            self.p = Deterministic(float(p))
        else:
            raise Exception("Expected StochasticParameter or float/int value, got %s." % (type(p),))
        """

        self.p = handle_continuous_param(p, "p")

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        ia.do_assert(0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,))
        return random_state.binomial(1, p, size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.p, float):
            return "Binomial(%.4f)" % (self.p,)
        else:
            return "Binomial(%s)" % (self.p,)

class Choice(StochasticParameter):
    """
    Parameter that samples value from a list of allowed values.

    Parameters
    ----------
    a : iterable
        List of allowed values.
        Usually expected to be integers, floats or strings.

    replace : bool, optional(default=True)
        Whether to perform sampling with or without
        replacing.

    p : None or iterable, optional(default=None)
        Optional probabilities of each element in `a`.
        Must have the same length as `a` (if provided).

    Examples
    --------
    >>> param = Choice([0.25, 0.5, 0.75], p=[0.25, 0.5, 0.25])

    Parameter of which 50 pecent of all sampled values will be 0.5.
    The other 50 percent will be either 0.25 or 0.75.

    """
    def __init__(self, a, replace=True, p=None):
        super(Choice, self).__init__()

        self.a = a
        self.replace = replace
        self.p = p

    def _draw_samples(self, size, random_state):
        if any([isinstance(a_i, StochasticParameter) for a_i in self.a]):
            seed = random_state.randint(0, 10**6, 1)[0]
            samples = ia.new_random_state(seed).choice(self.a, np.prod(size), replace=self.replace, p=self.p)

            # collect the sampled parameters and how many samples must be taken
            # from each of them
            params_counter = defaultdict(lambda: 0)
            #params_keys = set()
            for sample in samples:
                if isinstance(sample, StochasticParameter):
                    key = str(sample)
                    params_counter[key] += 1
                    #params_keys.add(key)

            # collect per parameter once the required number of samples
            # iterate here over self.a to always use the same seed for
            # the same parameter
            # TODO this might fail if the same parameter is added
            # multiple times to self.a?
            # TODO this will fail if a parameter cant handle size=(N,)
            param_to_samples = dict()
            for i, param in enumerate(self.a):
                key = str(param)
                if key in params_counter:
                    #print("[Choice] sampling %d from %s" % (params_counter[key], key))
                    param_to_samples[key] = param.draw_samples(
                        size=(params_counter[key],),
                        random_state=ia.new_random_state(seed+1+i)
                    )

            # assign the values sampled from the parameters to the `samples`
            # array by replacing the respective parameter
            param_to_readcount = defaultdict(lambda: 0)
            for i, sample in enumerate(samples):
                #if i%10 == 0:
                #    print("[Choice] assigning sample %d" % (i,))
                if isinstance(sample, StochasticParameter):
                    key = str(sample)
                    readcount = param_to_readcount[key]
                    #if readcount%10==0:
                    #    print("[Choice] readcount %d for %s" % (readcount, key))
                    samples[i] = param_to_samples[key][readcount]
                    param_to_readcount[key] += 1

            samples = samples.reshape(size)
        else:
            samples = random_state.choice(self.a, size, replace=self.replace, p=self.p)
        return samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Choice(a=%s, replace=%s, p=%s)" % (str(self.a), str(self.replace), str(self.p),)

class DiscreteUniform(StochasticParameter):
    """
    Parameter that resembles a discrete range of values [a .. b].

    Parameters
    ----------
    {a, b} : int or StochasticParameter
        Lower and upper bound of the sampling range. Values will be sampled
        from a <= x <= b. All sampled values will be discrete. If a or b is
        a StochasticParameter, it will be queried once per sampling to
        estimate the value of a/b. If a>b, the values will automatically be
        flipped. If a==b, all generated values will be identical to a.

    Examples
    --------
    >>> param = DiscreteUniform(10, Choice([20, 30, 40]))

    Sampled values will be discrete and come from the either [10..20] or
    [10..30] or [10..40].

    """

    def __init__(self, a, b):
        super(DiscreteUniform, self).__init__()

        """
        # for two ints the samples will be from range a <= x <= b
        ia.do_assert(isinstance(a, (int, StochasticParameter)), "Expected a to be int or StochasticParameter, got %s" % (type(a),))
        ia.do_assert(isinstance(b, (int, StochasticParameter)), "Expected b to be int or StochasticParameter, got %s" % (type(b),))

        if ia.is_single_integer(a):
            self.a = Deterministic(a)
        else:
            self.a = a

        if ia.is_single_integer(b):
            self.b = Deterministic(b)
        else:
            self.b = b
        """
        self.a = handle_discrete_param(a, "a")
        self.b = handle_discrete_param(b, "b")

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

class Poisson(StochasticParameter):
    """
    Parameter that resembles a poisson distribution.

    A poisson distribution with lambda=0 has its highest probability at
    point 0 and decreases quickly from there.
    Poisson distributions are discrete and never negative.

    Parameters
    ----------
    lam : number or tuple of two number or list of number or StochasticParameter
        Lambda parameter of the poisson
        distribution.
            * If a number, this number will be used as a constant value.
            * If a tuple of two numbers (a, b), the value will be sampled
              once per call to `_draw_samples()` from the range [a, b).
            * If a list of numbers, a random value will be picked from the
              list per call to `_draw_samples()`.
            * If a StochasticParameter, that parameter will be queried once
              per call to `_draw_samples()`.

    Examples
    --------
    >>> param = Poisson(1)

    Sample from a poisson distribution with lambda=1.

    """

    def __init__(self, lam):
        super(Poisson, self).__init__()

        if ia.is_single_number(lam):
            self.lam = Deterministic(lam)
        elif isinstance(lam, tuple):
            ia.do_assert(len(lam) == 2)
            self.lam = Uniform(lam[0], lam[1])
        elif ia.is_iterable(lam):
            self.lam = Choice(lam)
        elif isinstance(lam, StochasticParameter):
            self.lam = lam
        else:
            raise Exception("Expected number, tuple of two number, list of number or StochasticParameter for lam, got %s." % (type(lam),))

    def _draw_samples(self, size, random_state):
        lam = self.lam.draw_sample(random_state=random_state)
        lam = max(lam, 0)

        return random_state.poisson(lam=lam, size=size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Poisson(%s)" % (self.lam,)

class Normal(StochasticParameter):
    """
    Parameter that resembles a (continuous) normal distribution.

    This is a wrapper around numpy's random.normal().

    Parameters
    ----------
    loc : number or StochasticParameter
        The mean of the normal distribution.
        If StochasticParameter, the mean will be sampled once per call
        to `_draw_samples()`.

    scale : number or StochasticParameter
        The standard deviation of the normal distribution.
        If StochasticParameter, the scale will be sampled once per call
        to `_draw_samples()`.

    Examples
    --------
    >>> param = Normal(Choice([-1.0, 1.0]), 1.0)

    A standard normal distribution, which's mean is shifted either 1.0 to
    the left or 1.0 to the right.

    """
    def __init__(self, loc, scale):
        super(Normal, self).__init__()

        if isinstance(loc, StochasticParameter):
            self.loc = loc
        elif ia.is_single_number(loc):
            self.loc = Deterministic(loc)
        else:
            raise Exception("Expected float, int or StochasticParameter as loc, got %s." % (type(loc),))

        if isinstance(scale, StochasticParameter):
            self.scale = scale
        elif ia.is_single_number(scale):
            ia.do_assert(scale >= 0, "Expected scale to be in range [0, inf) got %s (type %s)." % (scale, type(scale)))
            self.scale = Deterministic(scale)
        else:
            raise Exception("Expected float, int or StochasticParameter as scale, got %s." % (type(scale),))

    def _draw_samples(self, size, random_state):
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        ia.do_assert(scale >= 0, "Expected scale to be in range [0, inf), got %s." % (scale,))
        if scale == 0:
            return np.tile(loc, size)
        else:
            return random_state.normal(loc, scale, size=size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Normal(loc=%s, scale=%s)" % (self.loc, self.scale)

class Laplace(StochasticParameter):
    """
    Parameter that resembles a (continuous) laplace distribution.

    This is a wrapper around numpy's random.laplace().

    Parameters
    ----------
    loc : number or tuple of two number or list of number or StochasticParameter
        The position of the distribution peak, similar to the mean in normal
        distributions.
            * If a single number, this number will be used as a constant value.
            * If a tuple of two numbers (a, b), the value will be sampled
              once per call to `_draw_samples()` from the continuous
              range [a, b).
            * If a list of numbers, a random value will be picked from the
              list per call to `_draw_samples()`.
            * If a StochasticParameter, that parameter will be queried once
              per call to `_draw_samples()`.

    scale : number or tuple of two number or list of number or StochasticParameter
        The exponential decay factor, similar to standard deviation in
        normal distributions.
            * If a single number, this number will be used as a constant value.
            * If a tuple of two numbers (a, b), the value will be sampled
              once per call to `_draw_samples()` from the continuous
              range [a, b).
            * If a list of numbers, a random value will be picked from the
              list per call to `_draw_samples()`.
            * If a StochasticParameter, that parameter will be queried once
              per call to `_draw_samples()`.

    Examples
    --------
    >>> param = Laplace(0, 1.0)

    A laplace distribution, which's peak is at 0 and decay is 1.0.

    """
    def __init__(self, loc, scale):
        super(Laplace, self).__init__()

        self.loc = handle_continuous_param(loc, "loc")
        self.scale = handle_continuous_param(scale, "scale")

    def _draw_samples(self, size, random_state):
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        ia.do_assert(scale >= 0, "Expected scale to be in range [0, inf), got %s." % (scale,))
        if scale == 0:
            return np.tile(loc, size)
        else:
            return random_state.laplace(loc, scale, size=size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Laplace(loc=%s, scale=%s)" % (self.loc, self.scale)

class ChiSquare(StochasticParameter):
    """
    Parameter that resembles a (continuous) chi-square distribution.

    This is a wrapper around numpy's random.chisquare().

    Parameters
    ----------
    df : int or tuple of two int or list of int or StochasticParameter
        Degrees of freedom (must be 1 or
        higher).
            * If a single int, this int will be used as a constant value.
            * If a tuple of two ints (a, b), the value will be sampled
              once per call to `_draw_samples()` from the discrete
              range [a..b].
            * If a list of ints, a random value will be picked from the
              list per call to `_draw_samples()`.
            * If a StochasticParameter, that parameter will be queried once
              per call to `_draw_samples()`.

    Examples
    --------
    >>> param = ChiSquare(df=2)

    A chi-square distribution with two degrees of freedom.

    """
    def __init__(self, df):
        super(ChiSquare, self).__init__()

        self.df = handle_discrete_param(df, "df", value_range=(1, None))

    def _draw_samples(self, size, random_state):
        df = self.df.draw_sample(random_state=random_state)
        ia.do_assert(df >= 1, "Expected df to be in range [1, inf), got %s." % (df,))
        return random_state.chisquare(df, size=size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "ChiSquare(df=%s)" % (self.df,)

class Weibull(StochasticParameter):
    """
    Parameter that resembles a (continuous) weibull distribution.

    This is a wrapper around numpy's random.weibull().

    Parameters
    ----------
    a : number or tuple of two number or list of number or StochasticParameter
        Shape parameter of the
        distribution.
            * If a single number, this number will be used as a constant value.
            * If a tuple of two numbers (a, b), the value will be sampled
              once per call to `_draw_samples()` from the continuous
              range [a, b).
            * If a list of numbers, a random value will be picked from the
              list per call to `_draw_samples()`.
            * If a StochasticParameter, that parameter will be queried once
              per call to `_draw_samples()`.

    Examples
    --------
    >>> param = Weibull(a=0.5)

    A weibull distribution with shape 0.5.

    """
    def __init__(self, a):
        super(Weibull, self).__init__()

        self.a = handle_continuous_param(a, "a", value_range=(0.0001, None))

    def _draw_samples(self, size, random_state):
        a = self.a.draw_sample(random_state=random_state)
        ia.do_assert(a > 0, "Expected a to be in range (0, inf), got %s." % (a,))
        return random_state.weibull(a, size=size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Weibull(a=%s)" % (self.a,)

class Uniform(StochasticParameter):
    """
    Parameter that resembles a (continuous) uniform range [a, b).

    Parameters
    ----------
    {a, b} : number or tuple of two number or list of number or StochasticParameter
        Lower and upper bound of the sampling range. Values will be sampled
        from a <= x < b. All sampled values will be continuous. If a or b is
        a StochasticParameter, it will be queried once per sampling to
        estimate the value of a/b. If a>b, the values will automatically be
        flipped. If a==b, all generated values will be identical to a.

    Examples
    --------
    >>> param = Uniform(0, 10.0)

    Samples random values from the range [0, 10.0).

    """
    def __init__(self, a, b):
        super(Uniform, self).__init__()

        """
        ia.do_assert(isinstance(a, (int, float, StochasticParameter)), "Expected a to be int, float or StochasticParameter, got %s" % (type(a),))
        ia.do_assert(isinstance(b, (int, float, StochasticParameter)), "Expected b to be int, float or StochasticParameter, got %s" % (type(b),))

        if ia.is_single_number(a):
            self.a = Deterministic(a)
        else:
            self.a = a

        if ia.is_single_number(b):
            self.b = Deterministic(b)
        else:
            self.b = b
        """

        self.a = handle_continuous_param(a, "a")
        self.b = handle_continuous_param(b, "b")

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

class Beta(StochasticParameter):
    """
    Parameter that resembles a (continuous) beta distribution.

    Parameters
    ----------
    {alpha, beta} : number or tuple of two number or list of number or StochasticParameter
        alpha and beta parameters of the beta
        distribution.
            * If number, that number will always be used.
            * If tuple of two number, a random value will be sampled per
              call to `_draw_samples()` from the range [a, b).
            * If list of number, a random element from that list will be
              sampled per call to `_draw_samples()`.
            * If a StochasticParameter, a random value will be sampled
              from that parameter per call to `_draw_samples()`.
        alpha and beta have to be values above 0. If they end up <=0 they
        are automatically clipped to 0+epsilon.

    epsilon : number
        Clipping parameter. If alpha or beta end up <=0, they are clipped to
        0+epsilon.

    Examples
    --------
    >>> param = Beta(0.5, 0.5)

    Samples random values from the beta distribution with alpha=beta=0.5.

    """
    def __init__(self, alpha, beta, epsilon=0.0001):
        super(Beta, self).__init__()

        def handle_param(param, name):
            if ia.is_single_number(param):
                return Deterministic(param)
            elif isinstance(param, tuple):
                ia.do_assert(len(param) == 2)
                return Uniform(param[0], param[1])
            elif ia.is_iterable(param):
                return Choice(param)
            elif isinstance(param, StochasticParameter):
                return param
            else:
                raise Exception("Expected number, tuple of two number, list of number or StochasticParameter for %s, got %s." % (name, type(param),))

        self.alpha = handle_param(alpha, "alpha")
        self.beta = handle_param(beta, "beta")

        ia.do_assert(ia.is_single_number(epsilon))
        self.epsilon = epsilon

    def _draw_samples(self, size, random_state):
        alpha = self.alpha.draw_sample(random_state=random_state)
        beta = self.beta.draw_sample(random_state=random_state)
        alpha = max(alpha, self.epsilon)
        beta = max(beta, self.epsilon)
        return random_state.beta(alpha, beta, size=size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Beta(%s, %s)" % (self.alpha, self.beta)

class Deterministic(StochasticParameter):
    """
    Parameter that resembles a constant value.

    If N values are sampled from this parameter, it will return N times V,
    where V is the constant value.

    Parameters
    ----------
    value : number or string or StochasticParameter
        A constant value to use.
        A string may be provided to generate arrays of strings.
        If this is a StochasticParameter, a single value will be sampled
        from it exactly once and then used as the constant value.

    Examples
    --------
    >>> param = Deterministic(10)

    Will always sample the value 10.

    """
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
        if ia.is_single_integer(self.value):
            return "Deterministic(int %d)" % (self.value,)
        elif ia.is_single_float(self.value):
            return "Deterministic(float %.8f)" % (self.value,)
        else:
            return "Deterministic(%s)" % (str(self.value),)

class FromLowerResolution(StochasticParameter):
    """
    A meta parameter used to sample other parameter values on a low resolution
    2d plane (where 2d means of size (H,W,C)).

    This is intended to be used with parameters that would usually sample
    once value per pixel (or one value per pixel and channel). With this
    parameter, the sampling can be made more coarse, i.e. the result will
    become rectangles instead of single pixels.

    Parameters
    ----------
    other_param : StochasticParameter
        The other parameter which is to be sampled on a coarser
        image.

    size_percent : None or number or iterable of two numbers or StochasticParameter, optional(default=None)
        Size of the 2d sampling plane in percent of the requested size.
        I.e. this is relative to the size provided in the call to
        `_draw_samples(size, ...)`. Lower values will result in smaller
        sampling planes, which are then upsampled to `size`. This means that
        lower values will result in larger rectangles.
        The size may be provided as a constant value or a tuple (a, b), which
        will automatically be converted to the continuous uniform range [a, b)
        or a StochasticParameter, which will be queried per call to
        `_draw_samples()`.

    size_px : None or number or iterable of two numbers or StochasticParameter, optional(default=None)
        Size of the 2d sampling plane in pixels.
        Lower values will result in smaller sampling planes, which are then
        upsampled to the input `size` of `draw_samples(size, ...)`.
        This means that lower values will result in larger rectangles.
        The size may be provided as a constant value or a tuple (a, b), which
        will automatically be converted to the discrete uniform range [a..b]
        or a StochasticParameter, which will be queried per call to
        `_draw_samples()`.

    method : string or int or StochasticParameter, optional(default="nearest")
        Upsampling/interpolation method to use. This is used after the sampling
        is finished and the low resolution plane has to be upsampled to the
        requested `size` in `_draw_samples(size, ...)`. The method may be
        the same as in `imgaug.imresize_many_images()`. Usually `nearest`
        or `linear` are good choices. `nearest` will result in rectangles
        with sharp edges and `linear` in rectangles with blurry and round
        edges. The method may be provided as a StochasticParameter, which
        will be queried per call to `_draw_samples()`.

    min_size : int, optional(default=1)
        Minimum size in pixels of the low resolution sampling
        plane.

    Examples
    --------
    >>> param = FromLowerResolution(Binomial(0.05), size_px=(2, 16), method=Choice(["nearest", "linear"]))

    Samples from a binomial distribution with p=0.05. The sampling plane
    will always have a size HxWxC with H and W being independently sampled
    from [2..16] (i.e. it may range from 2x2xC up to 16x16xC max, but may
    also be e.g. 4x8xC). The upsampling method will be "nearest" in 50 percent
    of all cases and "linear" in the other 50 percent. The result will
    sometimes be rectangular patches of sharp 1s surrounded by 0s and
    sometimes blurry blobs of 1s, surrounded by values <1.0.

    """
    def __init__(self, other_param, size_percent=None, size_px=None, method="nearest", min_size=1):
        super(FromLowerResolution, self).__init__()

        ia.do_assert(size_percent is not None or size_px is not None)

        if size_percent is not None:
            self.size_method = "percent"
            self.size_px = None
            if ia.is_single_number(size_percent):
                self.size_percent = Deterministic(size_percent)
            elif ia.is_iterable(size_percent):
                ia.do_assert(len(size_percent) == 2)
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
                ia.do_assert(len(size_px) == 2)
                self.size_px = DiscreteUniform(size_px[0], size_px[1])
            elif isinstance(size_px, StochasticParameter):
                self.size_px = size_px
            else:
                raise Exception("Expected int, float, tuple of two ints/floats or StochasticParameter for size_px, got %s." % (type(size_px),))

        self.other_param = other_param

        if ia.is_string(method) or ia.is_single_integer(method):
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
    """
    Clips another parameter to a defined value range.

    Parameters
    ----------
    other_param : StochasticParameter
        The other parameter, which's values are to be
        clipped.

    minval : None or number, optional(default=None)
        The minimum value to use.
        If None, no minimum will be used.

    maxval : None or number, optional(default=None)
        The maximum value to use.
        If None, no maximum will be used.

    Examples
    --------
    >>> param = Clip(Normal(0, 1.0), minval=-2.0, maxval=2.0)

    Defines a standard normal distribution, which's values never go below -2.0
    or above 2.0. Note that this will lead to small "bumps" of higher
    probability at -2.0 and 2.0, as values below/above these will be clipped
    to them.

    """
    def __init__(self, other_param, minval=None, maxval=None):
        super(Clip, self).__init__()

        ia.do_assert(isinstance(other_param, StochasticParameter))
        ia.do_assert(minval is None or ia.is_single_number(minval))
        ia.do_assert(maxval is None or ia.is_single_number(maxval))

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

class Discretize(StochasticParameter):
    """
    Convert values sampled from a continuous distribution into discrete values.

    This will round the values and then cast them to integers.
    Values sampled from discrete distributions are not changed.

    Parameters
    ----------
    other_param : StochasticParameter
        The other parameter, which's values are to be
        discretized.

    Examples
    --------
    >>> param = Discretize(Normal(0, 1.0))

    Generates a discrete standard normal distribution.

    """
    def __init__(self, other_param):
        super(Discretize, self).__init__()
        ia.do_assert(isinstance(other_param, StochasticParameter))
        self.other_param = other_param

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(
            size, random_state=random_state
        )
        if isinstance(samples.dtype, numbers.Integral):
            # integer array, already discrete
            return samples
        else:
            return np.round(samples).astype(np.int32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "Discretize(%s)" % (opstr,)

class Multiply(StochasticParameter):
    """
    Parameter to multiply other parameter's results with.

    Parameters
    ----------
    other_param : number or tuple of two number or list of number or StochasticParameter
        Other parameter which's sampled values are to be
        multiplied.

    val : number or tuple of two number or list of number or StochasticParameter
        Multiplier to use. If this is a StochasticParameter, either
        a single or multiple values will be sampled and used as the
        multiplier(s).

    elementwise : bool, optional(default=False)
        Controls the sampling behaviour when `val` is a StochasticParameter.
        If set to False, a single value will be sampled from val and used as
        the constant multiplier.
        If set to True and `_draw_samples(size=S)` is called, `S` values will
        be sampled from `val` and multiplied elementwise with the results
        of `other_param`.

    Examples
    --------
    >>> param = Multiply(Uniform(0.0, 1.0), -1)

    Converts a uniform range [0.0, 1.0) to (-1.0, 0.0].

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Multiply, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(size, random_state=ia.new_random_state(seed))

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=ia.new_random_state(seed+1))
        else:
            val_samples = self.val.draw_sample(random_state=ia.new_random_state(seed+1))

        if elementwise:
            return np.multiply(samples, val_samples)
        else:
            return samples * val_samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Multiply(%s, %s, %s)" % (str(self.other_param), str(self.val), self.elementwise)

class Divide(StochasticParameter):
    """
    Parameter to divide other parameter's results with.

    This parameter will automatically prevent division by zero (uses 1.0)
    as the denominator in these cases.

    Parameters
    ----------
    other_param : number or tuple of two number or list of number or StochasticParameter
        Other parameter which's sampled values are to be
        divided.

    val : number or tuple of two number or list of number or StochasticParameter
        Denominator to use. If this is a StochasticParameter, either
        a single or multiple values will be sampled and used as the
        denominator(s).

    elementwise : bool, optional(default=False)
        Controls the sampling behaviour when `val` is a StochasticParameter.
        If set to False, a single value will be sampled from val and used as
        the constant denominator.
        If set to True and `_draw_samples(size=S)` is called, `S` values will
        be sampled from `val` and used as the elementwise denominators for the
        results of `other_param`.

    Examples
    --------
    >>> param = Divide(Uniform(0.0, 1.0), 2)

    Converts a uniform range [0.0, 1.0) to [0, 0.5).

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Divide, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(size, random_state=ia.new_random_state(seed))

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(
                size,
                random_state=ia.new_random_state(seed+1)
            )

            # prevent division by zero
            val_samples[val_samples == 0] = 1

            return np.multiply(
                force_np_float_dtype(samples),
                force_np_float_dtype(val_samples)
            )
        else:
            val_sample = self.val.draw_sample(
                random_state=ia.new_random_state(seed+1)
            )

            # prevent division by zero
            if val_sample == 0:
                val_sample = 1

            return force_np_float_dtype(samples) / float(val_sample)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Divide(%s, %s, %s)" % (str(self.other_param), str(self.val), self.elementwise)

class Add(StochasticParameter):
    """
    Parameter to add to other parameter's results.

    Parameters
    ----------
    other_param : number or tuple of two number or list of number or StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    val : number or tuple of two number or list of number or StochasticParameter
        Value to add to the other parameter's results. If this is a
        StochasticParameter, either a single or multiple values will be
        sampled and added.

    elementwise : bool, optional(default=False)
        Controls the sampling behaviour when `val` is a StochasticParameter.
        If set to False, a single value will be sampled from val and added
        to all values generated by `other_param`.
        If set to True and `_draw_samples(size=S)` is called, `S` values will
        be sampled from `val` and added to the results of `other_param`.

    Examples
    --------
    >>> param = Add(Uniform(0.0, 1.0), 1.0)

    Converts a uniform range [0.0, 1.0) to [1.0, 2.0).

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Add, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(size, random_state=ia.new_random_state(seed))

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=ia.new_random_state(seed+1))
        else:
            val_samples = self.val.draw_sample(random_state=ia.new_random_state(seed+1))

        if elementwise:
            return np.add(samples, val_samples)
        else:
            return samples + val_samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Add(%s, %s, %s)" % (str(self.other_param), str(self.val), self.elementwise)

class Subtract(StochasticParameter):
    """
    Parameter to subtract from another parameter's results.

    Parameters
    ----------
    other_param : number or tuple of two number or list of number or StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    val : number or tuple of two number or list of number or StochasticParameter
        Value to add to the other parameter's results. If this is a
        StochasticParameter, either a single or multiple values will be
        sampled and subtracted.

    elementwise : bool, optional(default=False)
        Controls the sampling behaviour when `val` is a StochasticParameter.
        If set to False, a single value will be sampled from val and subtracted
        from all values generated by `other_param`.
        If set to True and `_draw_samples(size=S)` is called, `S` values will
        be sampled from `val` and subtracted from the results of `other_param`.

    Examples
    --------
    >>> param = Add(Uniform(0.0, 1.0), 1.0)

    Converts a uniform range [0.0, 1.0) to [1.0, 2.0).

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Subtract, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(size, random_state=ia.new_random_state(seed))

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(size, random_state=ia.new_random_state(seed+1))
        else:
            val_samples = self.val.draw_sample(random_state=ia.new_random_state(seed+1))

        if elementwise:
            return np.subtract(samples, val_samples)
        else:
            return samples - val_samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Subtract(%s, %s, %s)" % (str(self.other_param), str(self.val), self.elementwise)

class Power(StochasticParameter):
    """
    Parameter to exponentiate another parameter's results with.

    Parameters
    ----------
    other_param : number or tuple of two number or list of number or StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    val : number or tuple of two number or list of number or StochasticParameter
        Value to use exponentiate the other parameter's results with. If this
        is a StochasticParameter, either a single or multiple values will be
        sampled and used as the exponents.

    elementwise : bool, optional(default=False)
        Controls the sampling behaviour when `val` is a StochasticParameter.
        If set to False, a single value will be sampled from val and used as
        the exponent for all values generated by `other_param`.
        If set to True and `_draw_samples(size=S)` is called, `S` values will
        be sampled from `val` and used as the exponents for the results of
        `other_param`.

    Examples
    --------
    >>> param = Power(Uniform(0.0, 1.0), 2)

    Converts a uniform range [0.0, 1.0) to a distribution that is peaked
    towards 1.0.

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Power, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(size, random_state=ia.new_random_state(seed))

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            exponents = self.val.draw_samples(size, random_state=ia.new_random_state(seed+1))
        else:
            exponents = self.val.draw_sample(random_state=ia.new_random_state(seed+1))

        # without this we get int results in the case of
        # Power(<int>, <stochastic float param>)
        samples, exponents = both_np_float_if_one_is_float(samples, exponents)
        samples_dtype = samples.dtype

        # float_power requires numpy>=1.12
        #result = np.float_power(samples, exponents)
        # TODO why was float32 type here replaced with complex number
        # formulation?
        result = np.power(samples.astype(np.complex), exponents).real
        if result.dtype != samples_dtype:
            result = result.astype(samples_dtype)

        return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Power(%s, %s, %s)" % (str(self.other_param), str(self.val), self.elementwise)

class Absolute(StochasticParameter):
    """
    Converts another parameter's results to absolute values.

    Parameters
    ----------
    other_param : StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    Examples
    --------
    >>> param = Absolute(Uniform(-1.0, 1.0))

    Converts a uniform range [-1.0, 1.0) to [0.0, 1.0].

    """
    def __init__(self, other_param):
        super(Absolute, self).__init__()

        ia.do_assert(isinstance(other_param, StochasticParameter))

        self.other_param = other_param

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(size, random_state=random_state)
        return np.absolute(samples)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "Absolute(%s)" % (opstr,)

class RandomSign(StochasticParameter):
    """
    Converts another parameter's results randomly to positive or negative
    values.

    Parameters
    ----------
    other_param : StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    p_positive : number
        Fraction of values that are supposed to be turned to positive
        values.

    Examples
    --------
    >>> param = RandomSign(Poisson(1))

    Generates a poisson distribution with alpha=1 that is mirrored at the
    y-axis.

    """
    def __init__(self, other_param, p_positive=0.5):
        super(RandomSign, self).__init__()

        ia.do_assert(isinstance(other_param, StochasticParameter))
        ia.do_assert(ia.is_single_number(p_positive))
        ia.do_assert(0 <= p_positive <= 1)

        self.other_param = other_param
        self.p_positive = p_positive

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(
            size,
            random_state=ia.copy_random_state(random_state)
        )
        coinflips = ia.copy_random_state(random_state).binomial(
            1, self.p_positive, size=size
        ).astype(np.int32)
        signs = coinflips * 2 - 1
        # Add absolute here to guarantee that we get p_positive percent of
        # positive values. Otherwise we would merely flip p_positive percent
        # of all signs.
        result = np.absolute(samples) * signs
        #print("[RandomSign] ", size, coinflips.shape, signs.shape, result.shape, result[0:10])
        return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "RandomSign(%s, %.2f)" % (opstr, self.p_positive)

class ForceSign(StochasticParameter):
    """
    Converts another parameter's results to positive or negative values.

    Parameters
    ----------
    other_param : StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    positive : bool
        Whether to force all signs to be positive/+ (True) or
        negative/- (False).

    mode : string, optional(default="invert")
        How to change the signs. Valid values are "invert" and "reroll".
        "invert" means that wrong signs are simply flipped.
        "reroll" means that all samples with wrong signs are sampled again,
        optionally many times, until they randomly end up having the correct
        sign.

    reroll_count_max : int, optional(default=2)
        If `mode` is set to "reroll", this determines how often values may
        be rerolled before giving up and simply flipping the sign (as in
        mode="invert"). This shouldn't be set too high, as rerolling is
        expensive.

    Examples
    --------
    >>> param = ForceSign(Poisson(1), positive=False)

    Generates a poisson distribution with alpha=1 that is flipped towards
    negative values.

    """
    def __init__(self, other_param, positive, mode="invert", reroll_count_max=2):
        super(ForceSign, self).__init__()

        ia.do_assert(isinstance(other_param, StochasticParameter))

        self.other_param = other_param

        ia.do_assert(positive in [True, False])
        self.positive = positive

        ia.do_assert(mode in ["invert", "reroll"])
        self.mode = mode

        ia.do_assert(ia.is_single_integer(reroll_count_max))
        self.reroll_count_max = reroll_count_max

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(
            size,
            random_state=ia.new_random_state(seed)
        )

        if self.mode == "invert":
            if self.positive:
                samples[samples < 0] *= (-1)
            else:
                samples[samples > 0] *= (-1)
        else:
            if self.positive:
                bad_samples = np.where(samples < 0)[0]
            else:
                bad_samples = np.where(samples > 0)[0]

            reroll_count = 0
            while len(bad_samples) > 0 and reroll_count < self.reroll_count_max:
                # This rerolls the full input size, even when only a tiny
                # fraction of the values were wrong. That is done, because not
                # all parameters necessarily support any number of dimensions
                # for `size`, so we cant just resample size=N for N values
                # with wrong signs.
                # There is still quite some room for improvement here.
                samples_reroll = self.other_param.draw_samples(
                    size,
                    random_state=ia.new_random_state(seed+1+reroll_count)
                )
                samples[bad_samples] = samples_reroll[bad_samples]

                reroll_count += 1
                if self.positive:
                    bad_samples = np.where(samples < 0)[0]
                else:
                    bad_samples = np.where(samples > 0)[0]

            if len(bad_samples) > 0:
                samples[bad_samples] *= (-1)

        return samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "ForceSign(%s, %s, %s, %d)" % (opstr, str(self.positive), self.mode, self.reroll_count_max)

def Positive(other_param, mode="invert", reroll_count_max=2):
    """
    Converts another parameter's results to positive values.

    Parameters
    ----------
    other_param : StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    mode : string, optional(default="invert")
        How to change the signs. Valid values are "invert" and "reroll".
        "invert" means that wrong signs are simply flipped.
        "reroll" means that all samples with wrong signs are sampled again,
        optionally many times, until they randomly end up having the correct
        sign.

    reroll_count_max : int, optional(default=2)
        If `mode` is set to "reroll", this determines how often values may
        be rerolled before giving up and simply flipping the sign (as in
        mode="invert"). This shouldn't be set too high, as rerolling is
        expensive.

    Examples
    --------
    >>> param = Positive(Normal(0, 1), mode="reroll")

    Generates a normal distribution that has only positive values.

    """
    return ForceSign(
        other_param=other_param,
        positive=True,
        mode=mode,
        reroll_count_max=reroll_count_max
    )

def Negative(other_param, mode="invert", reroll_count_max=2):
    """
    Converts another parameter's results to negative values.

    Parameters
    ----------
    other_param : StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    mode : string, optional(default="invert")
        How to change the signs. Valid values are "invert" and "reroll".
        "invert" means that wrong signs are simply flipped.
        "reroll" means that all samples with wrong signs are sampled again,
        optionally many times, until they randomly end up having the correct
        sign.

    reroll_count_max : int, optional(default=2)
        If `mode` is set to "reroll", this determines how often values may
        be rerolled before giving up and simply flipping the sign (as in
        mode="invert"). This shouldn't be set too high, as rerolling is
        expensive.

    Examples
    --------
    >>> param = Negative(Normal(0, 1), mode="reroll")

    Generates a normal distribution that has only negative values.

    """
    return ForceSign(
        other_param=other_param,
        positive=False,
        mode=mode,
        reroll_count_max=reroll_count_max
    )

# TODO this always aggregates the result in high resolution space,
# instead of aggregating them in low resolution and then only upscaling the
# final image (for N iterations that would save up to N-1 upscales)
class IterativeNoiseAggregator(StochasticParameter):
    """
    Parameter to generate noise maps in multiple iterations and aggregate
    their results.

    This is supposed to be used in conjunction with SimplexNoise or
    FrequencyNoise.

    Parameters
    ----------
    other_param : StochasticParameter
        The noise parameter to iterate multiple
        times.

    iterations : int or iterable of two ints or list of ints or StochasticParameter, optional(default=(1, 3))
        The number of iterations. This may be a single integer or a tuple
        of integers (a, b), which will result in [a..b] iterations or
        a list of integers [a, b, c, ...], which will result in a or b or
        c, ... iterations. It may also be a StochasticParameter, in which case
        the number of iterations will be sampled once per call
        to `_draw_samples()`.

    aggregation_method : ia.ALL or string or list of string or StochasticParameter, optional(default=["max", "avg"])
        The method to use to aggregate the results of multiple iterations.
        If a string, it must have the value "min" or "max" or "avg".
        If "min" is chosen, the elementwise minimum will be computed over
        all iterations (pushing the noise towards zeros). "max" will result
        in the elementwise maximum and "avg" in the average over all
        iterations. If `ia.ALL` is used, it will be randomly either min or max
        or avg (per call to `_draw_samples()`). If a list is chosen, it must
        contain the mentioned strings and a random one will be picked per call
        to `_draw_samples()`. If a StochasticParameter is used, a value will
        be sampled from it per call to `_draw_samples()`.

    Examples
    --------
    >>> noise = IterativeNoiseAggregator(SimplexNoise(), iterations=(2, 5), aggregation_method="max")

    Generates per call 2 to 5 times simplex noise of a given size. Then
    combines these noise maps to a single map using elementwise maximum.

    """
    def __init__(self, other_param, iterations=(1, 3), aggregation_method=["max", "avg"]): # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
        ia.do_assert(isinstance(other_param, StochasticParameter))
        self.other_param = other_param

        if ia.is_single_integer(iterations):
            ia.do_assert(1 <= iterations <= 1000)
            self.iterations = Deterministic(iterations)
        elif ia.is_iterable(iterations):
            ia.do_assert(len(iterations) == 2)
            ia.do_assert(all([ia.is_single_integer(val) for val in iterations]))
            ia.do_assert(all([1 <= val <= 10000 for val in iterations]))
            self.iterations = DiscreteUniform(iterations[0], iterations[1])
        elif isinstance(iterations, list):
            ia.do_assert(len(iterations) > 0)
            ia.do_assert(all([1 <= val <= 10000 for val in iterations]))
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
            ia.do_assert(len(aggregation_method) >= 1)
            ia.do_assert(all([ia.is_string(val) for val in aggregation_method]))
            self.aggregation_method = Choice(aggregation_method)
        elif isinstance(aggregation_method, StochasticParameter):
            self.aggregation_method = aggregation_method
        else:
            raise Exception("Expected aggregation_method to be string or list of strings or StochasticParameter, got %s." % (type(aggregation_method),))

    def _draw_samples(self, size, random_state):
        ia.do_assert(len(size) == 2, "Expected requested other_param to have shape (H, W), got shape %s." % (size,))
        h, w = size

        seed = random_state.randint(0, 10**6)
        aggregation_method = self.aggregation_method.draw_sample(random_state=ia.new_random_state(seed))
        iterations = self.iterations.draw_sample(random_state=ia.new_random_state(seed+1))
        ia.do_assert(iterations > 0)

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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "IterativeNoiseAggregator(%s, %s, %s)" % (opstr, str(self.iterations), str(self.aggregation_method))

class Sigmoid(StochasticParameter):
    """
    Applies a sigmoid function to the outputs of another parameter.

    This is intended to be used in combination with SimplexNoise or
    FrequencyNoise. It pushes the noise values away from ~0.5 and towards
    0.0 or 1.0, making the noise maps more binary.

    Parameters
    ----------
    other_param : StochasticParameter
        The other parameter to which the sigmoid will be
        applied.

    threshold : number or tuple of two numbers or iterable of numbers or StochasticParameter, optional(default=(-10, 10))
        Sets the value of the sigmoid's saddle point, i.e. where values
        start to quickly shift from 0.0 to 1.0.
        This may be set using a single number, a tuple (a, b) (will result in
        a random threshold a<=x<b per call), a list of numbers (will
        result in a random threshold drawn from the list per call) or a
        StochasticParameter (will be queried once per call to determine the
        threshold).

    activated : bool or number, optional(default=True)
        Defines whether the sigmoid is activated. If this is False, the
        results of other_param will not be altered. This may be set to a
        float value p with 0<=p<=1.0, which will result in `activated` being
        True in p percent of all calls.

    mul : number, optional(default=1)
        The results of other_param will be multiplied with this value before
        applying the sigmoid. For noise values (range [0.0, 1.0]) this should
        be set to about 20.

    add : number, optional(default=0)
        This value will be added to the results of other_param before applying
        the sigmoid. For noise values (range [0.0, 1.0]) this should be set
        to about -10.0, provided `mul` was set to 20.

    Examples
    --------
    >>> param = Sigmoid(SimplexNoise(), activated=0.5, mul=20, add=-10)

    Applies a sigmoid to simplex noise in 50 percent of all calls. The noise
    results are modified to match the sigmoid's expected value range. The
    sigmoid's outputs are in the range [0.0, 1.0].

    """
    def __init__(self, other_param, threshold=(-10, 10), activated=True, mul=1, add=0):
        ia.do_assert(isinstance(other_param, StochasticParameter))
        self.other_param = other_param

        if ia.is_single_number(threshold):
            self.threshold = Deterministic(threshold)
        elif isinstance(threshold, tuple):
            ia.do_assert(len(threshold) == 2)
            ia.do_assert(all([ia.is_single_number(val) for val in threshold]))
            self.threshold = Uniform(threshold[0], threshold[1])
        elif ia.is_iterable(threshold):
            ia.do_assert(len(threshold) > 0)
            self.threshold = Choice(threshold)
        elif isinstance(threshold, StochasticParameter):
            self.threshold = threshold
        else:
            raise Exception("Expected threshold to be number or tuple of two numbers or StochasticParameter, got %s." % (type(threshold),))

        if activated in [True, False, 0, 1, 0.0, 1.0]:
            self.activated = Deterministic(int(activated))
        elif ia.is_single_number(activated):
            ia.do_assert(0 <= activated <= 1.0)
            self.activated = Binomial(activated)
        else:
            raise Exception("Expected activated to be boolean or number or StochasticParameter, got %s." % (type(activated),))

        ia.do_assert(ia.is_single_number(mul))
        ia.do_assert(mul > 0)
        self.mul = mul

        ia.do_assert(ia.is_single_number(add))
        self.add = add

    @staticmethod
    def create_for_noise(other_param, threshold=(-10, 10), activated=True):
        """
        Creates a Sigmoid that is adjusted to be used with noise parameters,
        i.e. with parameters which's output values are in the range [0.0, 1.0].

        Parameters
        ----------
        other_param : StochasticParameter
            See `Sigmoid`.

        threshold : number or tuple of two numbers or iterable of numbers or StochasticParameter, optional(default=(-10, 10))
            See `Sigmoid`.

        activated : bool or number, optional(default=True)
            See `Sigmoid`.

        Returns
        -------
        out : Sigmoid
            A sigmoid adjusted to be used with noise.

        """
        return Sigmoid(other_param, threshold, activated, mul=20, add=-10)

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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "Sigmoid(%s, %s, %s, %s, %s)" % (opstr, str(self.threshold), str(self.activated), str(self.mul), str(self.add))

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
    """
    A parameter that generates simplex noise of varying resolutions.

    This parameter expects to sample noise for 2d planes, i.e. for
    sizes (H, W) and will return a value in the range [0.0, 1.0] per location
    in that plane.

    The noise is sampled from low resolution planes and
    upscaled to the requested height and width. The size of the low
    resolution plane may be defined (high values can be slow) and the
    interpolation method for upscaling can be set.

    Parameters
    ----------
    size_px_max : int or tuple of two int or list of int or StochasticParameter, optional(default=(2, 16))
        Size in pixels of the low resolution plane.
        A single int will be used as a constant value. A tuple of two
        ints (a, b) will result in random values sampled from [a..b].
        A list of ints will result in random values being sampled from that
        list. A StochasticParameter will be queried once per call
        to `_draw_samples()`.

    upscale_method : string or int or StochasticParameter, optional(default="nearest")
        Upsampling/interpolation method to use. This is used after the sampling
        is finished and the low resolution plane has to be upsampled to the
        requested `size` in `_draw_samples(size, ...)`. The method may be
        the same as in `imgaug.imresize_many_images()`. Usually `nearest`
        or `linear` are good choices. `nearest` will result in rectangles
        with sharp edges and `linear` in rectangles with blurry and round
        edges. The method may be provided as a StochasticParameter, which
        will be queried per call to `_draw_samples()`.

    Examples
    --------
    >>> param = SimplexNoise(upscale_method="linear")

    Results in smooth simplex noise of varying sizes.

    >>> param = SimplexNoise(size_px_max=(8, 16), upscale_method="nearest")

    Results in rectangular simplex noise of rather high detail.

    """
    def __init__(self, size_px_max=(2, 16), upscale_method=["linear", "nearest"]): # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
        if ia.is_single_integer(size_px_max):
            ia.do_assert(1 <= size_px_max <= 10000)
            self.size_px_max = Deterministic(size_px_max)
        elif isinstance(size_px_max, tuple):
            ia.do_assert(len(size_px_max) == 2)
            ia.do_assert(all([ia.is_single_integer(val) for val in size_px_max]))
            ia.do_assert(all([1 <= val <= 10000 for val in size_px_max]))
            self.size_px_max = DiscreteUniform(size_px_max[0], size_px_max[1])
        elif ia.is_iterable(size_px_max):
            ia.do_assert(len(size_px_max) > 0)
            ia.do_assert(all([1 <= val <= 10000 for val in size_px_max]))
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
            ia.do_assert(len(upscale_method) >= 1)
            ia.do_assert(all([ia.is_string(val) for val in upscale_method]))
            self.upscale_method = Choice(upscale_method)
        elif isinstance(upscale_method, StochasticParameter):
            self.upscale_method = upscale_method
        else:
            raise Exception("Expected upscale_method to be string or list of strings or StochasticParameter, got %s." % (type(upscale_method),))

    def _draw_samples(self, size, random_state):
        ia.do_assert(len(size) == 2, "Expected requested noise to have shape (H, W), got shape %s." % (size,))
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
    """
    Parameter to generate noise of varying frequencies.

    This parameter expects to sample noise for 2d planes, i.e. for
    sizes (H, W) and will return a value in the range [0.0, 1.0] per location
    in that plane.

    The exponent controls the frequencies and therefore noise patterns.
    Low values (around -4.0) will result in large blobs. High values (around
    4.0) will result in small, repetitive patterns.

    The noise is sampled from low resolution planes and
    upscaled to the requested height and width. The size of the low
    resolution plane may be defined (high values can be slow) and the
    interpolation method for upscaling can be set.

    Parameters
    ----------
    exponent : number or tuple of numbers of list of numbers or StochasticParameter, optional(default=(-4, 4))
        Exponent to use when scaling in the frequency domain.
        Sane values are in the range -4 (large blobs) to 4 (small patterns).
        To generate cloud-like structures, use roughly -2.
            * If number, then that number will be used as the exponent for all
              iterations.
            * If tuple of two numbers (a, b), then a value will be sampled
              per iteration from the range [a, b].
            * If a list of numbers, then a value will be picked per iteration
              at random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per iteration.

    size_px_max : int or tuple of ints or list of ints or StochasticParameter, optional(default=(4, 16))
        The frequency noise is generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).
            * If int, then that number will be used as the size for all
              iterations.
            * If tuple of two ints (a, b), then a value will be sampled
              per iteration from the discrete range [a..b].
            * If a list of ints, then a value will be picked per iteration at
              random from that list.
            * If a StochasticParameter, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or ia.ALL or string or list of string or StochasticParameter, optional(default=None)
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.
            * If None, then either 'nearest' or 'linear' or 'cubic' is picked.
              Most weight is put on linear, followed by cubic.
            * If ia.ALL, then either 'nearest' or 'linear' or 'area' or 'cubic'
              is picked per iteration (all same probability).
            * If string, then that value will be used as the method (must be
              'nearest' or 'linear' or 'area' or 'cubic').
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If StochasticParameter, then a random value will be sampled
              from that parameter per iteration.

    Examples
    --------
    >>> param = FrequencyNoise(exponent=-2, size_px_max=(16, 32), upscale_method="linear")

    Generates noise with cloud-like patterns.

    """
    def __init__(self, exponent=(-4, 4), size_px_max=(4, 32), upscale_method=["linear", "nearest"]): # pylint: disable=locally-disabled, dangerous-default-value, line-too-long
        if ia.is_single_number(exponent):
            self.exponent = Deterministic(exponent)
        elif isinstance(exponent, tuple):
            ia.do_assert(len(exponent) == 2)
            ia.do_assert(all([ia.is_single_number(val) for val in exponent]))
            self.exponent = Uniform(exponent[0], exponent[1])
        elif ia.is_iterable(exponent):
            ia.do_assert(len(exponent) > 0)
            self.exponent = Choice(exponent)
        elif isinstance(exponent, StochasticParameter):
            self.exponent = exponent
        else:
            raise Exception("Expected exponent to be number or tuple of two numbers or StochasticParameter, got %s." % (type(exponent),))

        if ia.is_single_integer(size_px_max):
            ia.do_assert(1 <= size_px_max <= 10000)
            self.size_px_max = Deterministic(size_px_max)
        elif isinstance(size_px_max, tuple):
            ia.do_assert(len(size_px_max) == 2)
            ia.do_assert(all([ia.is_single_integer(val) for val in size_px_max]))
            ia.do_assert(all([1 <= val <= 10000 for val in size_px_max]))
            self.size_px_max = DiscreteUniform(size_px_max[0], size_px_max[1])
        elif ia.is_iterable(size_px_max):
            ia.do_assert(len(size_px_max) > 0)
            ia.do_assert(all([1 <= val <= 10000 for val in size_px_max]))
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
            ia.do_assert(len(upscale_method) >= 1)
            ia.do_assert(all([ia.is_string(val) for val in upscale_method]))
            self.upscale_method = Choice(upscale_method)
        elif isinstance(upscale_method, StochasticParameter):
            self.upscale_method = upscale_method
        else:
            raise Exception("Expected upscale_method to be string or list of strings or StochasticParameter, got %s." % (type(upscale_method),))

    def _draw_samples(self, size, random_state):
        # code here is similar to:
        #   http://www.redblobgames.com/articles/noise/2d/
        #   http://www.redblobgames.com/articles/noise/2d/2d-noise.js

        ia.do_assert(len(size) == 2, "Expected requested noise to have shape (H, W), got shape %s." % (size,))

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
