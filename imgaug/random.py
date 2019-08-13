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
BIT_GENERATOR = None
np_version = list(map(int, np.__version__.split(".")[0:2]))
if np_version[0] > 1 or np_version[1] >= 17:
    SUPPORTS_NEW_NP_RNG_STYLE = True
    BIT_GENERATOR = np.random.SFC64

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
# TODO update augmenter docstrings
# TODO change random_state to rng


class RNG(object):
    """
    Random number generator for imgaug.

    This class is a wrapper around ``numpy.random.Generator`` and
    automatically falls back to ``numpy.random.RandomState`` in case of
    numpy version 1.16 or lower. It allows to use numpy 1.17's sampling
    functions in 1.16 too and supports a variety of useful functions on
    the wrapped sampler, e.g. gettings its state or copying it.

    Not supported sampling functions of 1.16:

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

    In :func:`imgaug.random.RNG.choice`, the `axis` argument is not yet
    supported.

    Parameters
    ----------
    generator : None or int or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState
        The numpy random number generator to use. In case of numpy
        version 1.17 or later, this shouldn't be a ``RandomState`` as that
        class is outdated.
        Behaviour for different datatypes:

          * If ``None``: The global RNG is wrapped by this RNG (they are then
            effectively identical, any sampling on this RNG will affect the
            global RNG).
          * If ``int``: In numpy 1.17+, the value is used as a seed for a
            ``Generator`` wrapped by this RNG. I.e. it will be provided as the
            entropy to a ``SeedSequence``, which will then be used for an
            ``SFC64`` bit generator and wrapped by a ``Generator``.
            In numpy <=1.16, the value is used as a seed for a ``RandomState``,
            which is then wrapped by this RNG.
          * If :class:`numpy.random.Generator`: That generator will be wrapped.
          * If :class:`numpy.random.bit_generator.BitGenerator`: A numpy
            generator will be created (and wrapped by this RNG) that contains
            the bit generator.
          * If :class:`numpy.random.SeedSequence`: A numpy
            generator will be created (and wrapped by this RNG) that contains
            an ``SFC64`` bit generator initialized with the given
            ``SeedSequence``.
          * If :class:`numpy.random.RandomState`: In numpy <=1.16, this
            ``RandomState`` will be wrapped and used to sample random values.
            In numpy 1.17+, a seed will be derived from this ``RandomState``
            and a new ``numpy.generator.Generator`` based on an ``SFC64``
            bit generator will be created and wrapped by this RNG.

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
        """Get the state of this RNG.

        Returns
        -------
        tuple or dict
            The state of the RNG.
            In numpy 1.17+, the bit generator's state will be returned.
            In numpy <=1.16, the ``RandomState`` 's state is returned.
            In both cases the state is a copy. In-place changes will not affect
            the RNG.

        """
        return get_generator_state(self.generator)

    @state.setter
    def state(self, value):
        """Set the state if the RNG in-place.

        Parameters
        ----------
        value : tuple or dict
            The new state of the RNG.
            Should correspond to the output of the ``state`` property.

        """
        self.set_state_(value)

    def set_state_(self, value):
        """Set the state if the RNG in-place.

        Parameters
        ----------
        value : tuple or dict
            The new state of the RNG.
            Should correspond to the output of the ``state`` property.

        Returns
        -------
        RNG
            The RNG itself.

        """
        set_generator_state_(self.generator, value)
        return self

    def use_state_of_(self, other):
        """Copy and use (in-place) the state of another RNG.

        .. note ::

            It is often sensible to first verify that neither this RNG nor
            `other` are identical to the global RNG.


        Parameters
        ----------
        other : RNG
            The other RNG, which's state will be copied.

        Returns
        -------
        RNG
            The RNG itself.

        """
        return self.set_state_(other.state)

    def is_global_rng(self):
        """Estimate whether this RNG is identical to the global RNG.

        Returns
        -------
        bool
            ``True`` is this RNG's underlying generator is identical to the
            global RNG's underlying generator. The RNGs themselves may
            be different, only the wrapped generator matters.
            ``False`` otherwise.

        """
        # We use .generator here, because otherwise RNG(global_rng) would be
        # viewed as not-identical to the global RNG, even though its generator
        # and bit generator are identical.
        return get_global_rng().generator is self.generator

    def equals_global_rng(self):
        """Estimate whether this RNG has the same state as the global RNG.

        Returns
        -------
        bool
            ``True`` is this RNG has the same state as the global RNG, i.e.
            it will lead to the same sampled values given the same sampling
            method calls. The RNGs *don't* have to be identical object
            instances, which protects against e.g. copy effects.
            ``False`` otherwise.

        """
        return get_global_rng().equals(self)

    def generate_seed_(self):
        """Sample a random seed.

        This advances the underlying generator's state.

        See ``SEED_MIN_VALUE`` and ``SEED_MAX_VALUE`` for the seed's value
        range.

        Returns
        -------
        int
            The sampled seed.

        """
        return generate_seed_(self.generator)

    def generate_seeds_(self, n):
        """Generate `n` random seed values.

        This advances the underlying generator's state.

        See ``SEED_MIN_VALUE`` and ``SEED_MAX_VALUE`` for the seed's value
        range.

        Parameters
        ----------
        n : int
            Number of seeds to sample.

        Returns
        -------
        ndarray
            1D-array of ``int32`` seeds.

        """
        return generate_seeds_(self.generator, n)

    def reset_cache_(self):
        """Reset all cache of this RNG.

        Returns
        -------
        RNG
            The RNG itself.

        """
        reset_generator_cache_(self.generator)
        return self

    def derive_rng_(self):
        """Create a child RNG.

        This advances the underlying generator's state.

        Returns
        -------
        RNG
            A child RNG.

        """
        return self.derive_rngs_(1)[0]

    def derive_rngs_(self, n):
        """Create `n` child RNGs.

        This advances the underlying generator's state.

        Parameters
        ----------
        n : int
            Number of child RNGs to derive.

        Returns
        -------
        list of RNG
            Child RNGs.

        """
        return [RNG(gen) for gen in derive_generators_(self.generator, n)]

    def equals(self, other):
        """Estimate whether this RNG and `other` have the same state.

        Returns
        -------
        bool
            ``True`` if this RNG's generator and the generator of `other`
            have equal internal states. ``False`` otherwise.

        """
        assert isinstance(other, RNG)
        return is_generator_equal_to(self.generator, other.generator)

    def advance_(self):
        """Advance the RNG's internal state in-place by one step.

        This advances the underlying generator's state.

        .. note ::

            This simply samples one or more random values. This means that
            a call of this method will not completely change the outputs of
            the next called sampling method. To achieve more drastic output
            changes, call :func:`imgaug.random.RNG.derive_rng_`.

        Returns
        -------
        RNG
            The RNG itself.

        """
        advance_generator_(self.generator)
        return self

    def copy(self):
        """Create a copy of this RNG.

        Returns
        -------
        RNG
            Copy of this RNG. The copy will produce the same random samples.

        """
        return RNG(copy_generator(self.generator))

    def copy_unless_global_rng(self):
        """Create a copy of this RNG unless it is the global RNG.

        Returns
        -------
        RNG
            Copy of this RNG unless it is the global RNG. In the latter case
            the RNG instance itself will be returned without any changes.

        """
        if self.is_global_rng():
            return self
        return self.copy()

    def duplicate(self, n):
        """Create a list containing `n` times this RNG.

        This method was mainly introduced as a replacement for previous
        calls of :func:`imgaug.random.RNG.derive_rngs_`. These calls
        turned out to be very slow in numpy 1.17+ and were hence replaced
        by simple duplication (except for the cases where child RNGs
        absolutely *had* to be created).
        This RNG duplication method doesn't help very much against code
        repetition, but it does *mark* the points where it would be desirable
        to create child RNGs for various reasons. Once deriving child RNGs
        is somehow sped up in the future, these calls can again be
        easily found and replaced.

        Parameters
        ----------
        n : int
            Length of the output list.

        Returns
        -------
        list of RNG
            List containing `n` times this RNG (same instances, no copies).

        """
        return [self for _ in sm.xrange(n)]

    @classmethod
    def create_fully_random(cls):
        """Create a new RNG, based on entropy provided from the OS.

        Returns
        -------
        RNG
            A new RNG. It is not derived from any other previously created
            RNG, nor does it depend on the seeding of imgaug or numpy.

        """
        return RNG(create_fully_random_generator())

    @classmethod
    def create_pseudo_random_(cls):
        """Create a new RNG in pseudo-random fashion.

        A seed will be sampled from the current global RNG and used to
        initialize the new RNG.

        This advandes the global RNG's state.

        Returns
        -------
        RNG
            A new RNG, derived from the current global RNG.

        """
        return get_global_rng().derive_rng_()

    ###########################################################################
    # Below:
    #   Aliases for methods of numpy.random.Generator functions
    #
    # The methods below could also be handled with less code using some magic
    # methods. Explicitly writing things down here has the advantage that
    # the methods actually appear in the autogenerated API.
    ###########################################################################

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

    """
    global GLOBAL_RNG
    if GLOBAL_RNG is None:
        # TODO replace seed by constant
        GLOBAL_RNG = RNG(convert_seed_to_generator(42))
    return GLOBAL_RNG


# TODO mark as in-place
def seed(entropy):
    """Set the seed of imgaug's global RNG.

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


def normalize_generator(generator):
    """Normalize various inputs to a numpy (random number) generator.

    This function will first copy the provided argument, i.e. it never returns
    a provided instance itself.

    Parameters
    ----------
    generator : None or int or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState
        The numpy random number generator to normalize. In case of numpy
        version 1.17 or later, this shouldn't be a ``RandomState`` as that
        class is outdated.
        Behaviour for different datatypes:

          * If ``None``: The global RNG's generator is returned.
          * If ``int``: In numpy 1.17+, the value is used as a seed for a
            ``Generator``, i.e. it will be provided as the entropy to a
            ``SeedSequence``, which will then be used for an ``SFC64`` bit
            generator and wrapped by a ``Generator``, which is then returned.
            In numpy <=1.16, the value is used as a seed for a ``RandomState``,
            which will then be returned.
          * If :class:`numpy.random.Generator`: That generator will be
            returned.
          * If :class:`numpy.random.bit_generator.BitGenerator`: A numpy
            generator will be created and returned that contains the bit
            generator.
          * If :class:`numpy.random.SeedSequence`: A numpy
            generator will be created and returned that contains an ``SFC64``
            bit generator initialized with the given ``SeedSequence``.
          * If :class:`numpy.random.RandomState`: In numpy <=1.16, this
            ``RandomState`` will be returned. In numpy 1.17+, a seed will be
            derived from this ``RandomState`` and a new
            ``numpy.generator.Generator`` based on an ``SFC64`` bit generator
            will be created and returned.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator`` (even if
        the input was a ``RandomState``).

    """
    return normalize_generator_(copylib.deepcopy(generator))


def normalize_generator_(generator):
    """Normalize in-place various inputs to a numpy (random number) generator.

    This function will try to return the provided instance itself.

    Parameters
    ----------
    generator : None or int or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState
        See :func:`imgaug.random.normalize_generator`.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator`` (even if
        the input was a ``RandomState``).

    """
    if not SUPPORTS_NEW_NP_RNG_STYLE:
        return _normalize_generator_np116_(generator)
    return _normalize_generator_np117_(generator)


def _normalize_generator_np117_(generator):
    if generator is None:
        return get_global_rng().generator
    elif isinstance(generator, np.random.SeedSequence):
        return np.random.Generator(
            BIT_GENERATOR(generator)
        )
    elif isinstance(generator, np.random.bit_generator.BitGenerator):
        generator = np.random.Generator(generator)
        # TODO is it necessary/sensible here to reset the cache?
        reset_generator_cache_(generator)
        return generator
    elif isinstance(generator, np.random.Generator):
        # TODO is it necessary/sensible here to reset the cache?
        reset_generator_cache_(generator)
        return generator
    elif isinstance(generator, np.random.RandomState):
        # TODO warn
        # TODO reset the cache here too?
        return convert_seed_to_generator(generate_seed_(generator))
    # seed given
    seed_ = generator
    return convert_seed_to_generator(seed_)


def _normalize_generator_np116_(random_state):
    if random_state is None:
        return get_global_rng().generator
    elif isinstance(random_state, np.random.RandomState):
        # TODO reset the cache here, like in np117?
        return random_state
    # seed given
    seed_ = random_state
    return convert_seed_to_generator(seed_)


def convert_seed_to_generator(entropy):
    """Convert a seed value to a numpy (random number) generator.

    Parameters
    ----------
    entropy : int
        The seed value to use.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are initialized with the provided seed.

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
    """Convert a seed sequence to a numpy (random number) generator.

    Parameters
    ----------
    seed_sequence : numpy.random.SeedSequence
        The seed value to use.

    Returns
    -------
    numpy.random.Generator
        Generator initialized with the provided seed sequence.

    """
    return np.random.Generator(BIT_GENERATOR(seed_sequence))


def create_pseudo_random_generator_():
    """Create a new numpy (random) generator, derived from the global RNG.

    This function advances the global RNG's state.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are initialized with a seed sampled from the global RNG.

    """
    # could also use derive_rng(get_global_rng()) here
    random_seed = generate_seed_(get_global_rng().generator)
    return convert_seed_to_generator(random_seed)


def create_fully_random_generator():
    """Create a new numpy (random) generator, derived from OS's entropy.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are initialized with entropy requested from the OS. They are
        hence independent of entered seeds or the library's global RNG.

    """
    if not SUPPORTS_NEW_NP_RNG_STYLE:
        return _create_fully_random_generator_np116()
    return _create_fully_random_generator_np117()


def _create_fully_random_generator_np117():
    # TODO need entropy here?
    return np.random.Generator(np.random.SFC64())


def _create_fully_random_generator_np116():
    return np.random.RandomState()


def generate_seed_(generator):
    """Sample a seed from the provided generator.

    This function advances the generator's state.

    See ``SEED_MIN_VALUE`` and ``SEED_MAX_VALUE`` for the seed's value
    range.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator from which to sample the seed.

    Returns
    -------
    int
        The sampled seed.

    """
    return generate_seeds_(generator, 1)[0]


def generate_seeds_(generator, n):
    """Sample `n` seeds from the provided generator.

    This function advances the generator's state.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator from which to sample the seed.

    n : int
        Number of seeds to sample.

    Returns
    -------
    ndarray
        1D-array of ``int32`` seeds.

    """
    return polyfill_integers(generator, SEED_MIN_VALUE, SEED_MAX_VALUE,
                             size=(n,))


def copy_generator(generator):
    """Copy an existing numpy (random number) generator.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator to copy.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are copies of the input argument.

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
    """Copy a numpy generator unless it is the current global generator.

    "global generator" here denotes the generator contained in the
    global RNG's ``.generator`` attribute.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator to copy.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        Both are copies of the input argument, unless that input is
        identical to the global generator. If it is identical, the
        instance itself will be returned without copying it.

    """
    if generator is get_global_rng().generator:
        return generator
    return copy_generator(generator)


def reset_generator_cache_(generator):
    """Reset a numpy (random number) generator's internal cache.

    This function modifies the generator's state in-place.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator of which to reset the cache.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        In both cases the input argument itself.

    """
    if isinstance(generator, np.random.RandomState):
        return _reset_generator_cache_np116_(generator)
    return _reset_generator_cache_np117_(generator)


def _reset_generator_cache_np117_(generator):
    # This deactivates usage of the cache. We could also remove the cached
    # value itself in "uinteger", but setting the RNG to ignore the cached
    # value should be enough.
    state = _get_generator_state_np117(generator)
    state["has_uint32"] = 0
    _set_generator_state_np117_(generator, state)
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
    """Create a child numpy (random number) generator from an existing one.

    This advances the generator's state.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator from which to derive a new child generator.

    Returns
    -------
    numpy.random.Generator or numpy.random.RandomState
        In numpy <=1.16 a ``RandomState``, in 1.17+ a ``Generator``.
        In both cases a derived child generator.

    """
    return derive_generators_(generator, n=1)[0]


# TODO does this advance the RNG in 1.17? It should advance it for security
#      reasons
def derive_generators_(generator, n=1):
    """Create child numpy (random number) generators from an existing one.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator from which to derive new child generators.

    n : int, optional
        Number of child generators to derive.

    Returns
    -------
    list of numpy.random.Generator or list of numpy.random.RandomState
        In numpy <=1.16 a list of  ``RandomState`` s,
        in 1.17+ a list of ``Generator`` s.
        In both cases lists of derived child generators.

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
    """Get the state of this provided generator.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator, which's state is supposed to be extracted.

    Returns
    -------
    tuple or dict
        The state of the generator.
        In numpy 1.17+, the bit generator's state will be returned.
        In numpy <=1.16, the ``RandomState`` 's state is returned.
        In both cases the state is a copy. In-place changes will not affect
        the RNG.

    """
    if isinstance(generator, np.random.RandomState):
        return _get_generator_state_np116(generator)
    return _get_generator_state_np117(generator)


def _get_generator_state_np117(generator):
    return generator.bit_generator.state


def _get_generator_state_np116(random_state):
    return random_state.get_state()


def set_generator_state_(generator, state):
    """Set the state of a numpy (random number) generator in-place.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator, which's state is supposed to be modified.

    state : tuple or dict
        The new state of the generator.
        Should correspond to the output of
        :func:`imgaug.random.get_generator_state`.

    """
    if isinstance(generator, np.random.RandomState):
        _set_generator_state_np116_(generator, state)
    else:
        _set_generator_state_np117_(generator, state)


def _set_generator_state_np117_(generator, state):
    generator.bit_generator.state = state


def _set_generator_state_np116_(random_state, state):
    random_state.set_state(state)


def is_generator_equal_to(generator, other_generator):
    """Estimate whether two generator have the same class and state.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        First generator used in the comparison.

    other_generator : numpy.random.Generator or numpy.random.RandomState
        Second generator used in the comparison.

    Returns
    -------
    bool
        ``True`` if `generator` 's class and state are the same as the
        class and state of `other_generator`. ``False`` otherwise.

    """
    if isinstance(generator, np.random.RandomState):
        return _is_generator_equal_to_np116(generator, other_generator)
    return _is_generator_equal_to_np117(generator, other_generator)


def _is_generator_equal_to_np117(generator, other_generator):
    assert generator.__class__ is other_generator.__class__, (
        "Expected both rngs to have the same class, "
        "got types '%s' and '%s'." % (type(generator), type(other_generator)))

    state1 = get_generator_state(generator)
    state2 = get_generator_state(other_generator)
    assert state1["bit_generator"] == "SFC64"
    assert state2["bit_generator"] == "SFC64"

    if state1["has_uint32"] != state2["has_uint32"]:
        return False
    elif state1["has_uint32"] == state2["has_uint32"] == 1:
        if state1["uinteger"] != state2["uinteger"]:
            return False

    return np.array_equal(state1["state"]["state"], state2["state"]["state"])


def _is_generator_equal_to_np116(random_state, other_random_state):
    state1 = _get_generator_state_np116(random_state)
    state2 = _get_generator_state_np116(other_random_state)
    # Note that state1 and state2 are tuples with the value at index 1 being
    # a numpy array and the values at 2-4 being ints/floats, so we can't just
    # apply array_equal to state1[1:4+1] and state2[1:4+1]. We need a loop
    # here.
    for i in sm.xrange(1, 4+1):
        if not np.array_equal(state1[i], state2[i]):
            return False
    return True


def advance_generator_(generator):
    """Advance a numpy random generator's internal state in-place by one step.

    This advances the generator's state.

    .. note ::

        This simply samples one or more random values. This means that
        a call of this method will not completely change the outputs of
        the next called sampling method. To achieve more drastic output
        changes, call :func:`imgaug.random.derive_generator_`.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        Generator of which to advance the internal state.

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
    """Sample integers from a generator in different numpy versions.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator to sample from. If it is a ``RandomState``,
        :func:`numpy.random.RandomState.randint` will be called,
        otherwise :func:`numpy.random.Generator.integers`.

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
    if hasattr(generator, "randint"):
        if endpoint:
            if high is None:
                high = low + 1
                low = 0
            else:
                high = high + 1
        return generator.randint(low=low, high=high, size=size, dtype=dtype)
    return generator.integers(low=low, high=high, size=size, dtype=dtype,
                              endpoint=endpoint)


def polyfill_random(generator, size, dtype="float32", out=None):
    """Sample random floats from a generator in different numpy versions.

    Parameters
    ----------
    generator : numpy.random.Generator or numpy.random.RandomState
        The generator to sample from. Both ``RandomState`` and ``Generator``
        support ``random()``, but with different interfaces.

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
    if hasattr(generator, "random_sample"):
        # note that numpy.random in <=1.16 supports random(), but
        # numpy.random.RandomState does not
        result = generator.random_sample(size=size).astype(dtype)
        if out is not None:
            assert out.dtype.name == result.dtype.name, (
                "Expected out array to have the same dtype as "
                "random_sample()'s result array. Got %s (out) and %s (result) "
                "instead." % (out.dtype.name, result.dtype.name))
            out[...] = result
        return result
    return generator.random(size=size, dtype=dtype, out=out)
