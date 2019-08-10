from __future__ import print_function, division, absolute_import

import copy as copylib

import numpy as np
import six.moves as sm


# Check if numpy is version 1.17 or later. In that version, the new random
# number interface was added.
# Note that a valid version number can also be "1.18.0.dev0+285ab1d",
# in which the last component cannot easily be converted to an int. Hence we
# only pick the first two components.
# TODO rename to SUPPORTS_* or IS_SUPPORTING_*
IS_NEW_NP_RNG_STYLE = False
np_version = list(map(int, np.__version__.split(".")[0:2]))
if np_version[0] > 1 or np_version[1] >= 17:
    IS_NEW_NP_RNG_STYLE = True
IS_OLD_NP_RNG_STYLE = not IS_NEW_NP_RNG_STYLE


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
            self.generator = normalize_rng_(generator)
        self._is_new_rng_style = (
            not isinstance(self.generator, np.random.RandomState))

    @property
    def state(self):
        return get_rng_state(self.generator)

    @state.setter
    def state(self, value):
        self.set_state_(value)

    def set_state_(self, value):
        set_rng_state(self.generator, value)
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
        return generate_seed(self.generator)

    def generate_seeds_(self, n):
        return generate_seeds(self.generator, n)

    def reset_cache_(self):
        reset_rng_cache_(self.generator)
        return self

    def derive_rng_(self):
        return self.derive_rngs_(1)[0]

    def derive_rngs_(self, n):
        return [RNG(gen) for gen in derive_rngs(self.generator, n)]

    def equals(self, other):
        assert isinstance(other, RNG)
        return is_rng_identical_with(self.generator, other.generator)

    def advance_(self):
        advance_rng_(self.generator)
        return self

    def copy(self):
        return RNG(copy_rng(self.generator))

    def copy_unless_global_rng(self):
        return RNG(copy_rng_unless_global_rng(self.generator))

    def duplicate(self, n):
        return [self for _ in sm.xrange(n)]

    @classmethod
    def create_fully_random(cls):
        return RNG(create_fully_random_rng())

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
def is_new_numpy_rng_style():
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
    return IS_NEW_NP_RNG_STYLE


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
        GLOBAL_RNG = RNG(convert_seed_to_rng(42))
    return GLOBAL_RNG


def get_bit_generator_class():
    """
    Get the bit generator class used by imgaug.

    Returns
    -------
    numpy.random.bit_generator.BitGenerator
        The class to use as the bit generator. (**Not** an instance of that
        class!)

    """
    assert IS_NEW_NP_RNG_STYLE
    return np.random.SFC64


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
    if IS_NEW_NP_RNG_STYLE:
        _seed_np117(entropy)
    else:
        _seed_np116(entropy)


def _seed_np117(entropy):
    global GLOBAL_RNG, CURRENT_RANDOM_STATE
    # TODO any way to seed the Generator object instead of creating a new one?
    GLOBAL_RNG = RNG(entropy)
    CURRENT_RANDOM_STATE = GLOBAL_RNG


def _seed_np116(entropy):
    get_global_rng().seed(entropy)


def normalize_rng(rng):
    return normalize_rng_(copylib.deepcopy(rng))


# TODO add tests
def normalize_rng_(rng):
    """
    Normalize various inputs to a numpy random number generator.

    Parameters
    ----------
    rng : None or int or numpy.random.SeedSequence or numpy.random.bit_generator.BitGenerator or numpy.random.Generator
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
    if IS_OLD_NP_RNG_STYLE:
        return _normalize_rng_np116_(rng)
    return _normalize_rng_np117_(rng)


def _normalize_rng_np117_(rng):
    if rng is None:
        return get_global_rng().generator
    elif isinstance(rng, np.random.SeedSequence):
        return np.random.Generator(
            get_bit_generator_class()(rng)
        )
    elif isinstance(rng, np.random.bit_generator.BitGenerator):
        rng = np.random.Generator(rng)
        reset_rng_cache_(rng)
        return rng
    elif isinstance(rng, np.random.Generator):
        reset_rng_cache_(rng)
        return rng
    elif isinstance(rng, np.random.RandomState):
        # TODO warn
        return convert_seed_to_rng(generate_seed(rng))
    # seed given
    seed_ = rng
    return convert_seed_to_rng(seed_)


def _normalize_rng_np116_(random_state):
    if random_state is None:
        return get_global_rng().generator
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    # seed given
    seed_ = random_state
    return convert_seed_to_rng(seed_)


def convert_seed_to_rng(entropy):
    """
    Convert a seed value to an RNG.

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
    if IS_OLD_NP_RNG_STYLE:
        return _convert_seed_to_rng_np116(entropy)
    return _convert_seed_to_rng_np117(entropy)


def _convert_seed_to_rng_np117(entropy):
    seed_sequence = np.random.SeedSequence(entropy)
    return convert_seed_sequence_to_rng(seed_sequence)


def _convert_seed_to_rng_np116(entropy):
    return np.random.RandomState(entropy)


def convert_seed_sequence_to_rng(seed_sequence):
    """
    Convert a seed sequence to an RNG.

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


# TODO rename to create_pseudo_random_rng()?
def create_random_rng():
    # could also use derive_rng(get_global_rng()) here
    random_seed = generate_seed(get_global_rng())
    return convert_seed_to_rng(random_seed)


def create_fully_random_rng():
    if IS_NEW_NP_RNG_STYLE:
        return _create_fully_random_rng_np117()
    return _create_fully_random_rng_np116()


def _create_fully_random_rng_np117():
    # TODO need entropy here?
    return np.random.Generator(np.random.SFC64())


def _create_fully_random_rng_np116():
    return np.random.RandomState()


def generate_seed(rng):
    return generate_seeds(rng, 1)[0]


def generate_seeds(rng, n_seeds):
    return polyfill_integers(rng, SEED_MIN_VALUE, SEED_MAX_VALUE,
                             size=(n_seeds,))


def copy_rng(rng):
    """
    Copy an existing RNG.

    Parameters
    ----------
    rng : numpy.random.Generator or numpy.random.RandomState
        The RNG to copy.

    Returns
    -------
    numpy.random.Generator
        The copied RNG.
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    if isinstance(rng, np.random.RandomState):
        return _copy_rng_np116(rng)
    return _copy_rng_np117(rng)


def _copy_rng_np117(rng):
    # TODO not sure if it is enough to only copy the state
    # TODO initializing a bit gen and then copying the state might be slower
    #      then just deepcopying the whole thing
    old_bit_gen = rng.bit_generator
    new_bit_gen = old_bit_gen.__class__(1)
    new_bit_gen.state = copylib.deepcopy(old_bit_gen.state)
    return np.random.Generator(new_bit_gen)


def _copy_rng_np116(random_state):
    rs_copy = np.random.RandomState(1)
    state = random_state.get_state()
    rs_copy.set_state(state)
    return rs_copy


def copy_rng_unless_global_rng(rng):
    """
    Copy an RNG unless it is the current global RNG.

    Parameters
    ----------
    rng : numpy.random.Generator or numpy.random.RandomState
        The RNG to copy.

    Returns
    -------
    numpy.random.Generator
        Either a copy of the RNG (if it wasn't identical to imgaug's global
        RNG) or the input RNG (if it was identical).
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    if rng is get_global_rng():
        return rng
    return copy_rng(rng)


def reset_rng_cache_(rng):
    """
    Reset an RNG's internal cache.

    Parameters
    ----------
    rng : numpy.random.Generator or numpy.random.RandomState
        The RNG to reset.

    Returns
    -------
    numpy.random.Generator
        The same RNG, with its cache being reset.
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    if isinstance(rng, np.random.RandomState):
        return _reset_rng_cache_np116_(rng)
    return _reset_rng_cache_np117_(rng)


def _reset_rng_cache_np117_(rng):
    # This deactivates usage of the cache. We could also remove the cached
    # value itself in "uinteger", but setting the RNG to ignore the cached
    # value should be enough.
    rng.bit_generator.state["has_uint32"] = 0
    return rng


def _reset_rng_cache_np116_(rng):
    # State tuple content:
    #   'MT19937', array of ints, unknown int, cache flag, cached value
    # The cache flag only affects the standard_normal() method.
    state = list(rng.get_state())
    state[-2] = 0
    rng.set_state(tuple(state))
    return rng


def derive_rng(rng):
    """
    Create a new RNG based on an existing RNG.

    Parameters
    ----------
    rng : numpy.random.Generator or numpy.random.RandomState
        RNG from which to derive a new RNG.

    Returns
    -------
    numpy.random.Generator
        Derived RNG.
        In numpy 1.16 or older, an instance of ``numpy.random.RandomState``
        will be returned.

    """
    return derive_rngs(rng, n=1)[0]


# TODO does this advance the RNG in 1.17? It should advance it for security
#      reasons
def derive_rngs(rng, n=1):
    """
    Create new RNGs based on an existing RNG.

    Parameters
    ----------
    rng : numpy.random.Generator or numpy.random.RandomState
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
    if isinstance(rng, np.random.RandomState):
        return _derive_rngs_np116(rng, n=n)
    return _derive_rngs_np117(rng, n=n)


def _derive_rngs_np117(rng, n=1):
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
    seed_ = rng.integers(SEED_MIN_VALUE, SEED_MAX_VALUE, dtype="int32",
                         size=(2,))[-1]

    seed_seq = np.random.SeedSequence(seed_)
    seed_seqs = seed_seq.spawn(n)
    return [convert_seed_sequence_to_rng(seed_seq)
            for seed_seq in seed_seqs]


def _derive_rngs_np116(random_state, n=1):
    seed_ = random_state.randint(SEED_MIN_VALUE, SEED_MAX_VALUE)
    return [_convert_seed_to_rng_np116(seed_+i) for i in sm.xrange(n)]


def get_rng_state(rng):
    if isinstance(rng, np.random.RandomState):
        return _get_rng_state_np116(rng)
    return _get_rng_state_np117(rng)


def _get_rng_state_np117(rng):
    return rng.bit_generator.state


def _get_rng_state_np116(rng):
    return rng.get_state()


# TODO mark as inplace
def set_rng_state(rng, state):
    if isinstance(rng, np.random.RandomState):
        _set_rng_state_np116(rng, state)
    else:
        _set_rng_state_np117(rng, state)


def _set_rng_state_np117(rng, state):
    rng.bit_generator.state = state


def _set_rng_state_np116(rng, state):
    rng.set_state(state)


def is_rng_identical_with(rng, other_rng):
    if isinstance(rng, np.random.RandomState):
        return _is_rng_identical_with_np116(rng, other_rng)
    return _is_rng_identical_with_np117(rng, other_rng)


def _is_rng_identical_with_np117(rng, other_rng):
    assert rng.__class__ is other_rng.__class__, (
        "Expected both rngs to have the same class, "
        "got types '%s' and '%s'." % (type(rng), type(other_rng)))

    state1 = get_rng_state(rng)["state"]
    state2 = get_rng_state(other_rng)["state"]

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


def _is_rng_identical_with_np116(rng, other_rng):
    state1 = get_rng_state(rng)
    state2 = get_rng_state(other_rng)
    return np.array_equal(state1[1:4+1], state2[1:4+1])


def advance_rng_(rng):
    """
    Forward the internal state of an RNG by one step.

    Parameters
    ----------
    rng : numpy.random.Generator or numpy.random.RandomState
        RNG to forward.

    """
    if isinstance(rng, np.random.RandomState):
        _advance_rng_np116_(rng)
    else:
        _advance_rng_np117_(rng)


def _advance_rng_np117_(rng):
    _reset_rng_cache_np117_(rng)
    rng.random()


def _advance_rng_np116_(rng):
    _reset_rng_cache_np116_(rng)
    rng.uniform()


def polyfill_integers(rng, low, high=None, size=None, dtype="int32",
                      endpoint=False):
    """
    Sample integers from an RNG in different numpy versions.

    Parameters
    ----------
    rng : numpy.random.Generator or numpy.random.RandomState
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
    if isinstance(rng, np.random.RandomState):
        return rng.randint(low=low, high=high, size=size, dtype=dtype)
    return rng.integers(low=low, high=high, size=size, dtype=dtype,
                        endpoint=endpoint)


def polyfill_random(rng, size, dtype="float32", out=None):
    """
    Sample random floats from an RNG in different numpy versions.

    Parameters
    ----------
    rng : numpy.random.Generator or numpy.random.RandomState
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
    if isinstance(rng, np.random.RandomState):
        # note that numpy.random in <=1.16 supports random(), but
        # numpy.random.RandomState does not
        return rng.random_sample(size=size).astype(dtype)
    return rng.random(size=size, dtype=dtype, out=out)
