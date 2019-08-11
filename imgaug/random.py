from __future__ import print_function, division, absolute_import

import copy as copylib

import numpy as np
import six.moves as sm


# Check if numpy is version 1.17 or later. In that version, the new random
# number interface was added.
# Note that a valid version number can also be "1.18.0.dev0+285ab1d",
# in which the last component cannot easily be converted to an int. Hence we
# only pick the first two components.
SUPPORTS_NEW_NP_RNG_STYLE = False
np_version = list(map(int, np.__version__.split(".")[0:2]))
if np_version[0] > 1 or np_version[1] >= 17:
    SUPPORTS_NEW_NP_RNG_STYLE = True


# We instantiate a current/global random state here once.
GLOBAL_RNG = None

# Deprecated name for current RNG
CURRENT_RANDOM_STATE = GLOBAL_RNG

# use 2**31 instead of 2**32 as the maximum here, because 2**31 errored on
# some systems
SEED_MIN_VALUE = 0
SEED_MAX_VALUE = 2**31-1

# TODO decrease pool_size in SeedSequence to 2 or 1?
# TODO add 'with resetted_rng(...)'
# TODO rename _rng functions to _generator
# TODO update augmenter docstrings
# TODO change random_state to rng
# TODO use use_state_of_() in augment_* functions
# TODO use duplicate() function in augmenters


class RNG(object):
    """

    Not supported:

    * :func:`numpy.random.RandomState.rand`
    * :func:`numpy.random.RandomState.randn`
    * :func:`numpy.random.RandomState.randint`
    * :func:`numpy.random.RandomState.random_integers`
    * :func:`numpy.random.RandomState.random_sample`
    * :func:`numpy.random.RandomState.ranf`
    * :func:`numpy.random.RandomState.sample`
    * :func:`numpy.random.RandomState.seed`
    * :func:`numpy.random.RandomState.get_state`
    * :func:`numpy.random.RandomState.set_state`

    """

    # TODO add maybe a __new__ here that feeds-through an RNG input without
    #      wrapping it in RNG(rng_input)?
    def __init__(self, generator):
        if isinstance(generator, RNG):
            self.generator = generator.generator
        else:
            self.generator = normalize_generator_(generator)
        self._is_new_rng_style = (
            not isinstance(self.generator, np.random.RandomState))

    @property
    def state(self):
        return get_generator_state(self.generator)

    @state.setter
    def state(self, value):
        self.set_state_(value)

    def set_state_(self, value):
        set_rng_state_(self.generator, value)
        return self

    def use_state_of_(self, other):
        return self.set_state_(other.state)

    def use_state_copy_global_rng_(self, other):
        if self.is_global_rng():
            return other.copy()
        return self.set_state_(other.state)

    def is_global_rng(self):
        # We use .generator here, because otherwise RNG(global_rng) would be
        # viewed as not-identical to the global RNG, even though its generator
        # and bit generator are identical.
        return get_global_rng().generator is self.generator

    def equals_global_rng(self):
        return get_global_rng().equals(self)

    def generate_seed_(self):
        return generate_seed_(self.generator)

    def generate_seeds_(self, n):
        return generate_seeds_(self.generator, n)

    def reset_cache_(self):
        reset_generator_cache_(self.generator)
        return self

    def derive_rng_(self):
        return self.derive_rngs_(1)[0]

    def derive_rngs_(self, n):
        return [RNG(gen) for gen in derive_generators_(self.generator, n)]

    def equals(self, other):
        assert isinstance(other, RNG)
        return is_generator_equal_to(self.generator, other.generator)

    def advance_(self):
        advance_generator_(self.generator)
        return self

    def copy(self):
        return RNG(copy_generator(self.generator))

    def copy_unless_global_rng(self):
        return RNG(copy_generator_unless_global_rng(self.generator))

    def duplicate(self, n):
        return [self for _ in sm.xrange(n)]

    @classmethod
    def create_fully_random(cls):
        return RNG(create_fully_random_generator())

    @classmethod
    def create_pseudo_random_(cls):
        return get_global_rng().derive_rng_()

    ##################################
    # numpy.random.Generator functions
    ##################################

    def integers(self, low, high=None, size=None, dtype="int32",
                 endpoint=False):
        """Call numpy's ``integers()`` or ``randint()``.

        .. note ::

            Changed `dtype` argument default value from numpy's ``int64`` to
            ``int32``.

        """
        return polyfill_integers(
            self.generator, low=low, high=high, size=size, dtype=dtype,
            endpoint=endpoint)

    def random(self, size, dtype="float32", out=None):
        """Call numpy's ``random()`` or ``random_sample()``.

        .. note ::

            Changed `dtype` argument default value from numpy's ``d`` to
            ``float32``.

        """
        return polyfill_random(
            self.generator, size=size, dtype=dtype, out=out)

    # TODO add support for Generator's 'axis' argument
    def choice(self, a, size=None, replace=True, p=None):
        """Call :func:`numpy.random.Generator.choice`."""
        return self.generator.choice(a=a, size=size, replace=replace, p=p)

    def bytes(self, length):
        """Call :func:`numpy.random.Generator.bytes`."""
        return self.generator.bytes(length=length)

    # TODO mark in-place
    def shuffle(self, x):
        """Call :func:`numpy.random.Generator.shuffle`."""
        # note that shuffle() does not allow keyword arguments
        # note that shuffle() works in-place
        self.generator.shuffle(x)

    def permutation(self, x):
        """Call :func:`numpy.random.Generator.permutation`."""
        # note that permutation() does not allow keyword arguments
        return self.generator.permutation(x)

    def beta(self, a, b, size=None):
        """Call :func:`numpy.random.Generator.beta`."""
        return self.generator.beta(a=a, b=b, size=size)

    def binomial(self, n, p, size=None):
        """Call :func:`numpy.random.Generator.binomial`."""
        return self.generator.binomial(n=n, p=p, size=size)

    def chisquare(self, df, size=None):
        """Call :func:`numpy.random.Generator.chisquare`."""
        return self.generator.chisquare(df=df, size=size)

    def dirichlet(self, alpha, size=None):
        """Call :func:`numpy.random.Generator.dirichlet`."""
        return self.generator.dirichlet(alpha=alpha, size=size)

    def exponential(self, scale=1.0, size=None):
        """Call :func:`numpy.random.Generator.exponential`."""
        return self.generator.exponential(scale=scale, size=size)

    def f(self, dfnum, dfden, size=None):
        """Call :func:`numpy.random.Generator.f`."""
        return self.generator.f(dfnum=dfnum, dfden=dfden, size=size)

    def gamma(self, shape, scale=1.0, size=None):
        """Call :func:`numpy.random.Generator.gamma`."""
        return self.generator.gamma(shape=shape, scale=scale, size=size)

    def geometric(self, p, size=None):
        """Call :func:`numpy.random.Generator.geometric`."""
        return self.generator.geometric(p=p, size=size)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        """Call :func:`numpy.random.Generator.gumbel`."""
        return self.generator.gumbel(loc=loc, scale=scale, size=size)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        """Call :func:`numpy.random.Generator.hypergeometric`."""
        return self.generator.hypergeometric(
            ngood=ngood, nbad=nbad, nsample=nsample, size=size)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        """Call :func:`numpy.random.Generator.laplace`."""
        return self.generator.laplace(loc=loc, scale=scale, size=size)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        """Call :func:`numpy.random.Generator.logistic`."""
        return self.generator.logistic(loc=loc, scale=scale, size=size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        """Call :func:`numpy.random.Generator.lognormal`."""
        return self.generator.lognormal(mean=mean, sigma=sigma, size=size)

    def logseries(self, p, size=None):
        """Call :func:`numpy.random.Generator.logseries`."""
        return self.generator.logseries(p=p, size=size)

    def multinomial(self, n, pvals, size=None):
        """Call :func:`numpy.random.Generator.multinomial`."""
        return self.generator.multinomial(n=n, pvals=pvals, size=size)

    def multivariate_normal(self, mean, cov, size=None, check_valid="warn",
                            tol=1e-8):
        """Call :func:`numpy.random.Generator.multivariate_normal`."""
        return self.generator.multivariate_normal(
            mean=mean, cov=cov, size=size, check_valid=check_valid, tol=tol)

    def negative_binomial(self, n, p, size=None):
        """Call :func:`numpy.random.Generator.negative_binomial`."""
        return self.generator.negative_binomial(n=n, p=p, size=size)

    def noncentral_chisquare(self, df, nonc, size=None):
        """Call :func:`numpy.random.Generator.noncentral_chisquare`."""
        return self.generator.noncentral_chisquare(df=df, nonc=nonc, size=size)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        """Call :func:`numpy.random.Generator.noncentral_f`."""
        return self.generator.noncentral_f(
            dfnum=dfnum, dfden=dfden, nonc=nonc, size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Call :func:`numpy.random.Generator.normal`."""
        return self.generator.normal(loc=loc, scale=scale, size=size)

    def pareto(self, a, size=None):
        """Call :func:`numpy.random.Generator.pareto`."""
        return self.generator.pareto(a=a, size=size)

    def poisson(self, lam=1.0, size=None):
        """Call :func:`numpy.random.Generator.poisson`."""
        return self.generator.poisson(lam=lam, size=size)

    def power(self, a, size=None):
        """Call :func:`numpy.random.Generator.power`."""
        return self.generator.power(a=a, size=size)

    def rayleigh(self, scale=1.0, size=None):
        """Call :func:`numpy.random.Generator.rayleigh`."""
        return self.generator.rayleigh(scale=scale, size=size)

    def standard_cauchy(self, size=None):
        """Call :func:`numpy.random.Generator.standard_cauchy`."""
        return self.generator.standard_cauchy(size=size)

    def standard_exponential(self, size=None, dtype="float32", method='zig',
                             out=None):
        """Call :func:`numpy.random.Generator.standard_exponential`.

        .. note ::

            Changed `dtype` argument default value from numpy's ``d`` to
            ``float32``.

        """
        if self._is_new_rng_style:
            return self.generator.standard_exponential(
                size=size, dtype=dtype, method=method, out=out)
        return self.generator.standard_exponential(size=size).astype(dtype)

    def standard_gamma(self, shape, size=None, dtype="float32", out=None):
        """Call :func:`numpy.random.Generator.standard_gamma`.

        .. note ::

            Changed `dtype` argument default value from numpy's ``d`` to
            ``float32``.

        """
        if self._is_new_rng_style:
            return self.generator.standard_gamma(
                shape=shape, size=size, dtype=dtype, out=out)
        return self.generator.standard_gamma(
            shape=shape, size=size).astype(dtype)

    def standard_normal(self, size=None, dtype="float32", out=None):
        """Call :func:`numpy.random.Generator.standard_normal`.

        .. note ::

            Changed `dtype` argument default value from numpy's ``d`` to
            ``float32``.

        """
        if self._is_new_rng_style:
            return self.generator.standard_normal(
                size=size, dtype=dtype, out=out)
        return self.generator.standard_normal(size=size).astype(dtype)

    def standard_t(self, df, size=None):
        """Call :func:`numpy.random.Generator.standard_t`."""
        return self.generator.standard_t(df=df, size=size)

    def triangular(self, left, mode, right, size=None):
        """Call :func:`numpy.random.Generator.triangular`."""
        return self.generator.triangular(
            left=left, mode=mode, right=right, size=size)

    def uniform(self, low=0.0, high=1.0, size=None):
        """Call :func:`numpy.random.Generator.uniform`."""
        return self.generator.uniform(low=low, high=high, size=size)

    def vonmises(self, mu, kappa, size=None):
        """Call :func:`numpy.random.Generator.vonmises`."""
        return self.generator.vonmises(mu=mu, kappa=kappa, size=size)

    def wald(self, mean, scale, size=None):
        """Call :func:`numpy.random.Generator.wald`."""
        return self.generator.wald(mean=mean, scale=scale, size=size)

    def weibull(self, a, size=None):
        """Call :func:`numpy.random.Generator.weibull`."""
        return self.generator.weibull(a=a, size=size)

    def zipf(self, a, size=None):
        """Call :func:`numpy.random.Generator.zipf`."""
        return self.generator.zipf(a=a, size=size)


# TODO rename to supports_* or is_supporting_*
def supports_new_numpy_rng_style():
    """
    Determine whether numpy supports the new ``random`` interface (v1.17+).

    Returns
    -------
    bool
        ``True`` if the new ``random`` interface is supported by numpy, i.e.
        if numpy has version 1.17 or later. Otherwise ``False``, i.e.
        numpy has version 1.16 or older and ``numpy.random.RandomState``
        should be used instead.

    """
    return SUPPORTS_NEW_NP_RNG_STYLE


def get_global_rng():
    """
    Get or create the current global RNG of imgaug.

    Note that the first call to this function will create a global RNG.

    Returns
    -------
    RNG
        The global RNG to use.
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    global GLOBAL_RNG
    if GLOBAL_RNG is None:
        # TODO replace seed by constant
        GLOBAL_RNG = RNG(convert_seed_to_generator(42))
    return GLOBAL_RNG


# TODO replace by constructor
def get_bit_generator_class():
    """
    Get the bit generator class used by imgaug.

    Returns
    -------
    numpy.random.bit_generator.BitGenerator
        The class to use as the bit generator. (**Not** an instance of that
        class!)

    """
    assert SUPPORTS_NEW_NP_RNG_STYLE
    return np.random.SFC64


# TODO mark as in-place
def seed(entropy):
    """
    Set the seed of imgaug's global random number generator.

    The global RNG controls most of the "randomness" in imgaug.

    The global RNG is the default one used by all augmenters. Under special
    circumstances (e.g. when an augmenter is switched to deterministic mode),
    the global RNG is replaced by a local one. The state of that replacement
    may be dependent on the global RNG's state at the time of creating the
    child RNG.

    Parameters
    ----------
    entropy : int
        The seed value to use.

    """
    if SUPPORTS_NEW_NP_RNG_STYLE:
        _seed_np117(entropy)
    else:
        _seed_np116(entropy)


def _seed_np117(entropy):
    global GLOBAL_RNG, CURRENT_RANDOM_STATE
    # TODO any way to seed the Generator object instead of creating a new one?
    GLOBAL_RNG = RNG(entropy)
    CURRENT_RANDOM_STATE = GLOBAL_RNG


def _seed_np116(entropy):
    get_global_rng().generator.seed(entropy)


def normalize_generator(rng):
    return normalize_generator_(copylib.deepcopy(rng))


# TODO add tests
def normalize_generator_(generator):
    """
    Normalize various inputs to a numpy (random number) generator.

    Parameters
    ----------
    generator : None or int or numpy.random.SeedSequence or numpy.random.bit_generator.BitGenerator or numpy.random.Generator
        The input to normalize.
        If this is ``None``, the global RNG will be returned.
        If this is an instance of ``numpy.random.Generator`` or
        ``numpy.random.RandomState`` it will be returned
        without any change. Seed sequences or bit generators will be wrapped
        to return a Generator. Integers will result in a new generator
        (wrapping a seed sequence and bit generator) being returned.

    Returns
    -------
    numpy.random.Generator
        Normalized RNG.
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    if not SUPPORTS_NEW_NP_RNG_STYLE:
        return _normalize_generator_np116_(generator)
    return _normalize_generator_np117_(generator)


def _normalize_generator_np117_(generator):
    if generator is None:
        return get_global_rng().generator
    elif isinstance(generator, np.random.SeedSequence):
        return np.random.Generator(
            get_bit_generator_class()(generator)
        )
    elif isinstance(generator, np.random.bit_generator.BitGenerator):
        generator = np.random.Generator(generator)
        reset_generator_cache_(generator)
        return generator
    elif isinstance(generator, np.random.Generator):
        reset_generator_cache_(generator)
        return generator
    elif isinstance(generator, np.random.RandomState):
        # TODO warn
        return convert_seed_to_generator(generate_seed_(generator))
    # seed given
    seed_ = generator
    return convert_seed_to_generator(seed_)


def _normalize_generator_np116_(random_state):
    if random_state is None:
        return get_global_rng().generator
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    # seed given
    seed_ = random_state
    return convert_seed_to_generator(seed_)


def convert_seed_to_generator(entropy):
    """
    Convert a seed value to a numpy (random number) generator.

    Parameters
    ----------
    entropy : int
        The seed value to use.

    Returns
    -------
    numpy.random.Generator
        RNG initialized with the provided seed.
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    if not SUPPORTS_NEW_NP_RNG_STYLE:
        return _convert_seed_to_generator_np116(entropy)
    return _convert_seed_to_generator_np117(entropy)


def _convert_seed_to_generator_np117(entropy):
    seed_sequence = np.random.SeedSequence(entropy)
    return convert_seed_sequence_to_generator(seed_sequence)


def _convert_seed_to_generator_np116(entropy):
    return np.random.RandomState(entropy)


def convert_seed_sequence_to_generator(seed_sequence):
    """
    Convert a seed sequence to a numpy (random number) generator.

    Parameters
    ----------
    seed_sequence : numpy.random.bit_generator.SeedSequence
        The seed value to use.

    Returns
    -------
    numpy.random.Generator
        RNG initialized with the provided seed.
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    bit_gen = get_bit_generator_class()
    return np.random.Generator(bit_gen(seed_sequence))


def create_pseudo_random_generator_():
    # could also use derive_rng(get_global_rng()) here
    random_seed = generate_seed_(get_global_rng())
    return convert_seed_to_generator(random_seed)


def create_fully_random_generator():
    if not SUPPORTS_NEW_NP_RNG_STYLE:
        return _create_fully_random_generator_np116()
    return _create_fully_random_generator_np117()


def _create_fully_random_generator_np117():
    # TODO need entropy here?
    return np.random.Generator(np.random.SFC64())


def _create_fully_random_generator_np116():
    return np.random.RandomState()


def generate_seed_(generator):
    return generate_seeds_(generator, 1)[0]


def generate_seeds_(generator, n_seeds):
    return polyfill_integers(generator, SEED_MIN_VALUE, SEED_MAX_VALUE,
                             size=(n_seeds,))


def copy_generator(generator):
    """
    Copy an existing numpy (random number) generator.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The RNG to copy.

    Returns
    -------
    numpy.random.Generator
        The copied RNG.
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    if isinstance(generator, np.random.RandomState):
        return _copy_generator_np116(generator)
    return _copy_generator_np117(generator)


def _copy_generator_np117(generator):
    # TODO not sure if it is enough to only copy the state
    # TODO initializing a bit gen and then copying the state might be slower
    #      then just deepcopying the whole thing
    old_bit_gen = generator.bit_generator
    new_bit_gen = old_bit_gen.__class__(1)
    new_bit_gen.state = copylib.deepcopy(old_bit_gen.state)
    return np.random.Generator(new_bit_gen)


def _copy_generator_np116(random_state):
    rs_copy = np.random.RandomState(1)
    state = random_state.get_state()
    rs_copy.set_state(state)
    return rs_copy


def copy_generator_unless_global_rng(generator):
    """
    Copy a numpy generator unless it is the current global generator.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The RNG to copy.

    Returns
    -------
    numpy.random.Generator
        Either a copy of the RNG (if it wasn't identical to imgaug's global
        RNG) or the input RNG (if it was identical).
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    if generator is get_global_rng().generator:
        return generator
    return copy_generator(generator)


def reset_generator_cache_(generator):
    """
    Reset a numpy (random number) generator's internal cache.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The RNG to reset.

    Returns
    -------
    numpy.random.Generator
        The same RNG, with its cache being reset.
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    if isinstance(generator, np.random.RandomState):
        return _reset_generator_cache_np116_(generator)
    return _reset_generator_cache_np117_(generator)


def _reset_generator_cache_np117_(generator):
    # This deactivates usage of the cache. We could also remove the cached
    # value itself in "uinteger", but setting the RNG to ignore the cached
    # value should be enough.
    generator.bit_generator.state["has_uint32"] = 0
    return generator


def _reset_generator_cache_np116_(random_state):
    # State tuple content:
    #   'MT19937', array of ints, unknown int, cache flag, cached value
    # The cache flag only affects the standard_normal() method.
    state = list(random_state.get_state())
    state[-2] = 0
    random_state.set_state(tuple(state))
    return random_state


def derive_generator_(generator):
    """
    Create a new numpy (random number) generator based on an existing generator.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        RNG from which to derive a new RNG.

    Returns
    -------
    numpy.random.Generator
        Derived RNG.
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    return derive_generators_(generator, n=1)[0]


# TODO does this advance the RNG in 1.17? It should advance it for security
#      reasons
def derive_generators_(generator, n=1):
    """
    Create new numpy (random number) generators based on an existing generator.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        RNG from which to derive new RNGs.

    n : int, optional
        Number of RNGs to derive.

    Returns
    -------
    list of numpy.random.Generator
        Derived RNGs.
        In numpy 1.16 or older, a list of ``numpy.random.RandomState``
        will be returned.

    """
    if isinstance(generator, np.random.RandomState):
        return _derive_generators_np116_(generator, n=n)
    return _derive_generators_np117_(generator, n=n)


def _derive_generators_np117_(generator, n=1):
    # TODO possible to get the SeedSequence from 'rng'?
    """
    advance_rng_(rng)
    rng = copylib.deepcopy(rng)
    reset_rng_cache_(rng)
    state = rng.bit_generator.state
    rngs = []
    for i in sm.xrange(n):
        state["state"]["state"] += (i * 100003 + 17)
        rng.bit_generator.state = state
        rngs.append(rng)
        rng = copylib.deepcopy(rng)
    return rngs
    """

    # We generate here two integers instead of one, because the internal state
    # of the RNG might have one 32bit integer still cached up, which would
    # then be returned first when calling integers(). This should usually be
    # fine, but there is some risk involved that this will lead to sampling
    # many times the same seed in loop constructions (if the internal state
    # is not properly advanced and the cache is then also not reset). Adding
    # 'size=(2,)' decreases that risk. (It is then enough to e.g. call once
    # random() to advance the internal state. No resetting of caches is
    # needed.)
    seed_ = generator.integers(SEED_MIN_VALUE, SEED_MAX_VALUE, dtype="int32",
                               size=(2,))[-1]

    seed_seq = np.random.SeedSequence(seed_)
    seed_seqs = seed_seq.spawn(n)
    return [convert_seed_sequence_to_generator(seed_seq)
            for seed_seq in seed_seqs]


def _derive_generators_np116_(random_state, n=1):
    seed_ = random_state.randint(SEED_MIN_VALUE, SEED_MAX_VALUE)
    return [_convert_seed_to_generator_np116(seed_ + i) for i in sm.xrange(n)]


def get_generator_state(generator):
    if isinstance(generator, np.random.RandomState):
        return _get_generator_state_np116(generator)
    return _get_generator_state_np117(generator)


def _get_generator_state_np117(generator):
    return generator.bit_generator.state


def _get_generator_state_np116(random_state):
    return random_state.get_state()


def set_rng_state_(generator, state):
    if isinstance(generator, np.random.RandomState):
        _set_rng_state_np116_(generator, state)
    else:
        _set_rng_state_np117_(generator, state)


def _set_rng_state_np117_(generator, state):
    generator.bit_generator.state = state


def _set_rng_state_np116_(random_state, state):
    random_state.set_state(state)


def is_generator_equal_to(generator, other_generator):
    if isinstance(generator, np.random.RandomState):
        return _is_generator_equal_to_np116(generator, other_generator)
    return _is_generator_equal_to_np117(generator, other_generator)


# TODO rework this method
def _is_generator_equal_to_np117(generator, other_generator):
    assert generator.__class__ is other_generator.__class__, (
        "Expected both rngs to have the same class, "
        "got types '%s' and '%s'." % (type(generator), type(other_generator)))

    state1 = get_generator_state(generator)["state"]
    state2 = get_generator_state(other_generator)["state"]

    if isinstance(state1, (list, tuple)):
        for a, b in zip(state1, state2):
            if a.dtype.kind != b.dtype.kind:
                return False
            assert a.dtype.kind in ["i", "u"]
            if not np.array_equal(a, b):
                return False
    elif isinstance(state1, dict):
        keys1 = set(state1.keys())
        keys2 = set(state2.keys())
        assert len(keys1.union(keys2)) == len(keys1)

        for key in state1:
            a, b = state1[key], state2[key]
            if a.dtype.kind != b.dtype.kind:
                return False
            assert a.dtype.kind in ["i", "u"]
            if not np.array_equal(a, b):
                return False
    else:
        raise ValueError("Unknown state type. Expected list, tuple or dict, "
                         "got type %s" % (type(state1),))

    return True


def _is_generator_equal_to_np116(random_state, other_random_state):
    state1 = _get_generator_state_np116(random_state)
    state2 = _get_generator_state_np116(other_random_state)
    return np.array_equal(state1[1:4+1], state2[1:4+1])


def advance_generator_(generator):
    """
    Forward the internal state of an RNG by one step.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        RNG to forward.

    """
    if isinstance(generator, np.random.RandomState):
        _advance_generator_np116_(generator)
    else:
        _advance_generator_np117_(generator)


def _advance_generator_np117_(generator):
    _reset_generator_cache_np117_(generator)
    generator.random()


def _advance_generator_np116_(generator):
    _reset_generator_cache_np116_(generator)
    generator.uniform()


def polyfill_integers(generator, low, high=None, size=None, dtype="int32",
                      endpoint=False):
    """
    Sample integers from an RNG in different numpy versions.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The RNG to sample from. If it is a ``RandomState``, ``randint()`` will
        be called, otherwise ``integers()``.

    low : int or array-like of ints
        See :func:`numpy.random.Generator.integers`.

    high : int or array-like of ints, optional
        See :func:`numpy.random.Generator.integers`.

    size : int or tuple of ints, optional
        See :func:`numpy.random.Generator.integers`.

    dtype : {str, dtype}, optional
        See :func:`numpy.random.Generator.integers`.

    endpoint : bool, optional
        See :func:`numpy.random.Generator.integers`.

    Returns
    -------
    int or ndarray of ints
        See :func:`numpy.random.Generator.integers`.

    """
    if isinstance(generator, np.random.RandomState):
        return generator.randint(low=low, high=high, size=size, dtype=dtype)
    return generator.integers(low=low, high=high, size=size, dtype=dtype,
                              endpoint=endpoint)


def polyfill_random(generator, size, dtype="float32", out=None):
    """
    Sample random floats from an RNG in different numpy versions.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The RNG to sample from. Both ``RandomState`` and ``Generator``
        suppert ``random()``, but with different interfaces.

    size : int or tuple of ints, optional
        See :func:`numpy.random.Generator.random`.

    dtype : {str, dtype}, optional
        See :func:`numpy.random.Generator.random`.

    out : ndarray, optional
        See :func:`numpy.random.Generator.random`.


    Returns
    -------
    float or ndarray of floats
        See :func:`numpy.random.Generator.random`.

    """
    if isinstance(generator, np.random.RandomState):
        # note that numpy.random in <=1.16 supports random(), but
        # numpy.random.RandomState does not
        return generator.random_sample(size=size).astype(dtype)
    return generator.random(size=size, dtype=dtype, out=out)
