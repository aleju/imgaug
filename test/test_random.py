from __future__ import print_function, division, absolute_import

import copy as copylib
import sys
# unittest only added in 3.4 self.subTest()
if sys.version_info[0] < 3 or sys.version_info[1] < 4:
    import unittest2 as unittest
else:
    import unittest
# unittest.mock is not available in 2.7 (though unittest2 might contain it?)
try:
    import unittest.mock as mock
except ImportError:
    import mock

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np

import imgaug as ia
from imgaug.testutils import reseed
import imgaug.random as iarandom

NP_VERSION = np.__version__
IS_NP_117_OR_HIGHER = (
    NP_VERSION.startswith("2.")
    or NP_VERSION.startswith("1.25")
    or NP_VERSION.startswith("1.24")
    or NP_VERSION.startswith("1.23")
    or NP_VERSION.startswith("1.22")
    or NP_VERSION.startswith("1.21")
    or NP_VERSION.startswith("1.20")
    or NP_VERSION.startswith("1.19")
    or NP_VERSION.startswith("1.18")
    or NP_VERSION.startswith("1.17")
)


class _Base(unittest.TestCase):
    def setUp(self):
        reseed()


class TestConstants(_Base):
    def test_supports_new_np_rng_style_is_true(self):
        assert iarandom.SUPPORTS_NEW_NP_RNG_STYLE is IS_NP_117_OR_HIGHER

    def test_global_rng(self):
        iarandom.get_global_rng()  # creates global RNG upon first call
        assert iarandom.GLOBAL_RNG is not None


class Test_supports_new_numpy_rng_style(_Base):
    def test_call(self):
        assert iarandom.supports_new_numpy_rng_style() is IS_NP_117_OR_HIGHER


class Test_get_global_rng(_Base):
    def test_call(self):
        iarandom.seed(0)

        rng = iarandom.get_global_rng()

        expected = iarandom.RNG(0)
        assert rng is not None
        assert rng.equals(expected)


class Test_seed(_Base):
    @mock.patch("imgaug.random._seed_np117")
    @mock.patch("imgaug.random._seed_np116")
    def test_mocked_call(self, mock_np116, mock_np117):
        iarandom.seed(1)

        if IS_NP_117_OR_HIGHER:
            mock_np117.assert_called_once_with(1)
            assert mock_np116.call_count == 0
        else:
            mock_np116.assert_called_once_with(1)
            assert mock_np117.call_count == 0

    def test_integrationtest(self):
        iarandom.seed(1)
        assert iarandom.GLOBAL_RNG.equals(iarandom.RNG(1))
        assert iarandom.CURRENT_RANDOM_STATE is iarandom.GLOBAL_RNG


class Test_normalize_generator(_Base):
    @mock.patch("imgaug.random.normalize_generator_")
    def test_mocked_call(self, mock_subfunc):
        mock_subfunc.return_value = "foo"
        inputs = ["bar"]

        result = iarandom.normalize_generator(inputs)

        assert mock_subfunc.call_count == 1
        assert mock_subfunc.call_args[0][0] is not inputs
        assert mock_subfunc.call_args[0][0] == inputs
        assert result == "foo"


class Test_normalize_generator_(_Base):
    @mock.patch("imgaug.random._normalize_generator_np117_")
    @mock.patch("imgaug.random._normalize_generator_np116_")
    def test_mocked_call(self, mock_np116, mock_np117):
        mock_np116.return_value = "np116"
        mock_np117.return_value = "np117"

        result = iarandom.normalize_generator_(None)

        if IS_NP_117_OR_HIGHER:
            assert result == "np117"
            mock_np117.assert_called_once_with(None)
            assert mock_np116.call_count == 0
        else:
            assert result == "np116"
            mock_np116.assert_called_once_with(None)
            assert mock_np117.call_count == 0

    def test_called_with_none(self):
        result = iarandom.normalize_generator_(None)
        assert result is iarandom.get_global_rng().generator

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "SeedSequence does not exist in numpy <=1.16")
    def test_called_with_seed_sequence(self):
        seedseq = np.random.SeedSequence(0)

        result = iarandom.normalize_generator_(seedseq)

        expected = np.random.Generator(
            iarandom.BIT_GENERATOR(np.random.SeedSequence(0)))
        assert iarandom.is_generator_equal_to(result, expected)

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "BitGenerator does not exist in numpy <=1.16")
    def test_called_with_generator(self):
        bgen = iarandom.BIT_GENERATOR(np.random.SeedSequence(0))

        result = iarandom.normalize_generator_(bgen)

        assert result.bit_generator is bgen

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Generator does not exist in numpy <=1.16")
    def test_called_with_generator(self):
        gen = np.random.Generator(
            iarandom.BIT_GENERATOR(np.random.SeedSequence(0))
        )

        result = iarandom.normalize_generator_(gen)

        assert result is gen

    def test_called_with_random_state(self):
        rs = np.random.RandomState(0)

        result = iarandom.normalize_generator_(rs)

        if IS_NP_117_OR_HIGHER:
            seed = iarandom.generate_seed_(np.random.RandomState(0))
            expected = iarandom.convert_seed_to_generator(seed)
            assert iarandom.is_generator_equal_to(result, expected)
        else:
            assert result is rs

    def test_called_int(self):
        seed = 0

        result = iarandom.normalize_generator_(seed)

        expected = iarandom.convert_seed_to_generator(seed)
        assert iarandom.is_generator_equal_to(result, expected)


class Test_convert_seed_to_generator(_Base):
    @mock.patch("imgaug.random._convert_seed_to_generator_np117")
    @mock.patch("imgaug.random._convert_seed_to_generator_np116")
    def test_mocked_call(self, mock_np116, mock_np117):
        mock_np116.return_value = "np116"
        mock_np117.return_value = "np117"

        result = iarandom.convert_seed_to_generator(1)

        if IS_NP_117_OR_HIGHER:
            assert result == "np117"
            mock_np117.assert_called_once_with(1)
            assert mock_np116.call_count == 0
        else:
            assert result == "np116"
            mock_np116.assert_called_once_with(1)
            assert mock_np117.call_count == 0

    def test_call(self):
        gen = iarandom.convert_seed_to_generator(1)

        if IS_NP_117_OR_HIGHER:
            expected = np.random.Generator(
                iarandom.BIT_GENERATOR(np.random.SeedSequence(1)))

            assert iarandom.is_generator_equal_to(gen, expected)
        else:
            expected = np.random.RandomState(1)
            assert iarandom.is_generator_equal_to(gen, expected)


class Test_convert_seed_sequence_to_generator(_Base):
    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "SeedSequence does not exist in numpy <=1.16")
    def test_call(self):
        seedseq = np.random.SeedSequence(1)

        gen = iarandom.convert_seed_sequence_to_generator(seedseq)

        expected = np.random.Generator(
            iarandom.BIT_GENERATOR(np.random.SeedSequence(1)))
        assert iarandom.is_generator_equal_to(gen, expected)


class Test_create_pseudo_random_generator_(_Base):
    def test_call(self):
        global_gen = copylib.deepcopy(iarandom.get_global_rng().generator)

        gen = iarandom.create_pseudo_random_generator_()

        expected = iarandom.convert_seed_to_generator(
            iarandom.generate_seed_(global_gen))
        assert iarandom.is_generator_equal_to(gen, expected)


class Test_create_fully_random_generator(_Base):
    @mock.patch("imgaug.random._create_fully_random_generator_np117")
    @mock.patch("imgaug.random._create_fully_random_generator_np116")
    def test_mocked_call(self, mock_np116, mock_np117):
        mock_np116.return_value = "np116"
        mock_np117.return_value = "np117"

        result = iarandom.create_fully_random_generator()

        if IS_NP_117_OR_HIGHER:
            assert result == "np117"
            mock_np117.assert_called_once()
            assert mock_np116.call_count == 0
        else:
            assert result == "np116"
            mock_np116.assert_called_once()
            assert mock_np117.call_count == 0

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_np117_mocked(self):
        dummy_bitgen = np.random.SFC64(1)

        with mock.patch("numpy.random.SFC64") as mock_bitgen:
            mock_bitgen.return_value = dummy_bitgen

            result = iarandom._create_fully_random_generator_np117()

        mock_bitgen.assert_called_once()
        assert iarandom.is_generator_equal_to(
            result, np.random.Generator(dummy_bitgen))

    def test_np116_mocked(self):
        dummy_rs = np.random.RandomState(1)

        with mock.patch("numpy.random.RandomState") as mock_rs:
            mock_rs.return_value = dummy_rs

            result = iarandom._create_fully_random_generator_np116()

        mock_rs.assert_called_once()
        assert iarandom.is_generator_equal_to(result, np.random.RandomState(1))


class Test_generate_seed_(_Base):
    @mock.patch("imgaug.random.generate_seeds_")
    def test_mocked_call(self, mock_seeds):
        gen = iarandom.convert_seed_to_generator(0)

        _ = iarandom.generate_seed_(gen)

        mock_seeds.assert_called_once_with(gen, 1)


class Test_generate_seeds_(_Base):
    @mock.patch("imgaug.random.polyfill_integers")
    def test_mocked_call(self, mock_integers):
        gen = iarandom.convert_seed_to_generator(0)

        _ = iarandom.generate_seeds_(gen, 10)

        mock_integers.assert_called_once_with(
            gen, iarandom.SEED_MIN_VALUE, iarandom.SEED_MAX_VALUE, size=(10,))

    def test_call(self):
        gen = iarandom.convert_seed_to_generator(0)

        seeds = iarandom.generate_seeds_(gen, 2)

        assert len(seeds) == 2
        assert ia.is_np_array(seeds)
        assert seeds.dtype.name == "int32"


class Test_copy_generator(_Base):
    @mock.patch("imgaug.random._copy_generator_np116")
    def test_mocked_call_with_random_state(self, mock_np116):
        mock_np116.return_value = "np116"
        gen = np.random.RandomState(1)

        gen_copy = iarandom.copy_generator(gen)

        assert gen_copy == "np116"
        mock_np116.assert_called_once_with(gen)

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    @mock.patch("imgaug.random._copy_generator_np117")
    def test_mocked_call_with_generator(self, mock_np117):
        mock_np117.return_value = "np117"
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))

        gen_copy = iarandom.copy_generator(gen)

        assert gen_copy == "np117"
        mock_np117.assert_called_once_with(gen)

    def test_call_with_random_state(self):
        gen = np.random.RandomState(1)

        gen_copy = iarandom._copy_generator_np116(gen)

        assert gen is not gen_copy
        assert iarandom.is_generator_equal_to(gen, gen_copy)

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_call_with_generator(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))

        gen_copy = iarandom._copy_generator_np117(gen)

        assert gen is not gen_copy
        assert iarandom.is_generator_equal_to(gen, gen_copy)


class Test_copy_generator_unless_global_rng(_Base):
    @mock.patch("imgaug.random.get_global_rng")
    @mock.patch("imgaug.random.copy_generator")
    def test_mocked_gen_is_global(self, mock_copy, mock_get_global_rng):
        gen = iarandom.convert_seed_to_generator(1)
        mock_copy.return_value = "foo"
        mock_get_global_rng.return_value = iarandom.RNG(gen)

        result = iarandom.copy_generator_unless_global_rng(gen)

        mock_get_global_rng.assert_called_once()
        assert mock_copy.call_count == 0
        assert result is gen

    @mock.patch("imgaug.random.get_global_rng")
    @mock.patch("imgaug.random.copy_generator")
    def test_mocked_gen_is_not_global(self, mock_copy, mock_get_global_rng):
        gen1 = iarandom.convert_seed_to_generator(1)
        gen2 = iarandom.convert_seed_to_generator(2)
        mock_copy.return_value = "foo"
        mock_get_global_rng.return_value = iarandom.RNG(gen2)

        result = iarandom.copy_generator_unless_global_rng(gen1)

        mock_get_global_rng.assert_called_once()
        mock_copy.assert_called_once()
        assert result == "foo"


class Test_reset_generator_cache_(_Base):
    @mock.patch("imgaug.random._reset_generator_cache_np117_")
    @mock.patch("imgaug.random._reset_generator_cache_np116_")
    def test_mocked_call(self, mock_np116, mock_np117):
        mock_np116.return_value = "np116"
        mock_np117.return_value = "np117"
        gen = iarandom.convert_seed_to_generator(1)

        result = iarandom.reset_generator_cache_(gen)

        if IS_NP_117_OR_HIGHER:
            assert result == "np117"
            mock_np117.assert_called_once()
            assert mock_np116.call_count == 0
        else:
            assert result == "np116"
            mock_np116.assert_called_once()
            assert mock_np117.call_count == 0

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_call_np117(self):
        gen = iarandom.convert_seed_to_generator(1)
        gen_without_cache_copy = copylib.deepcopy(gen)

        state = iarandom._get_generator_state_np117(gen)
        state["has_uint32"] = 1
        gen_with_cache = copylib.deepcopy(gen)
        iarandom.set_generator_state_(gen_with_cache, state)
        gen_with_cache_copy = copylib.deepcopy(gen_with_cache)

        gen_cache_reset = iarandom.reset_generator_cache_(gen_with_cache)

        assert iarandom.is_generator_equal_to(gen_cache_reset,
                                              gen_without_cache_copy)
        assert not iarandom.is_generator_equal_to(gen_cache_reset,
                                                  gen_with_cache_copy)

    def test_call_np116(self):
        gen = np.random.RandomState(1)
        gen_without_cache_copy = copylib.deepcopy(gen)

        state = iarandom._get_generator_state_np116(gen)
        state = list(state)
        state[-2] = 1
        gen_with_cache = copylib.deepcopy(gen)
        iarandom.set_generator_state_(gen_with_cache, tuple(state))
        gen_with_cache_copy = copylib.deepcopy(gen_with_cache)

        gen_cache_reset = iarandom.reset_generator_cache_(gen_with_cache)

        assert iarandom.is_generator_equal_to(gen_cache_reset,
                                              gen_without_cache_copy)
        assert not iarandom.is_generator_equal_to(gen_cache_reset,
                                                  gen_with_cache_copy)


class Test_derive_generator_(_Base):
    @mock.patch("imgaug.random.derive_generators_")
    def test_mocked_call(self, mock_derive_gens):
        mock_derive_gens.return_value = ["foo"]
        gen = iarandom.convert_seed_to_generator(1)

        gen_derived = iarandom.derive_generator_(gen)

        mock_derive_gens.assert_called_once_with(gen, n=1)
        assert gen_derived == "foo"

    def test_integration(self):
        gen = iarandom.convert_seed_to_generator(1)
        gen_copy = copylib.deepcopy(gen)

        gen_derived = iarandom.derive_generator_(gen)

        assert not iarandom.is_generator_equal_to(gen_derived, gen_copy)
        # should have advanced the state
        assert not iarandom.is_generator_equal_to(gen_copy, gen)


class Test_derive_generators_(_Base):
    @mock.patch("imgaug.random._derive_generators_np117_")
    @mock.patch("imgaug.random._derive_generators_np116_")
    def test_mocked_call(self, mock_np116, mock_np117):
        mock_np116.return_value = "np116"
        mock_np117.return_value = "np117"
        gen = iarandom.convert_seed_to_generator(1)

        result = iarandom.derive_generators_(gen)

        if isinstance(gen, np.random.RandomState):
            assert result == "np116"
            mock_np116.assert_called_once()
            assert mock_np117.call_count == 0
        else:
            assert result == "np117"
            mock_np117.assert_called_once()
            assert mock_np116.call_count == 0

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_call_np117(self):
        gen = iarandom.convert_seed_to_generator(1)
        gen_copy = copylib.deepcopy(gen)

        result = iarandom.derive_generators_(gen, 2)

        assert len(result) == 2
        assert np.all([isinstance(gen, np.random.Generator)
                       for gen in result])
        assert not iarandom.is_generator_equal_to(result[0], gen_copy)
        assert not iarandom.is_generator_equal_to(result[1], gen_copy)
        assert not iarandom.is_generator_equal_to(result[0], result[1])
        # derive should advance state
        assert not iarandom.is_generator_equal_to(gen, gen_copy)

    def test_call_np116(self):
        gen = np.random.RandomState(1)
        gen_copy = copylib.deepcopy(gen)

        result = iarandom.derive_generators_(gen, 2)

        assert len(result) == 2
        assert np.all([isinstance(gen, np.random.RandomState)
                       for gen in result])
        assert not iarandom.is_generator_equal_to(result[0], gen_copy)
        assert not iarandom.is_generator_equal_to(result[1], gen_copy)
        assert not iarandom.is_generator_equal_to(result[0], result[1])
        # derive should advance state
        assert not iarandom.is_generator_equal_to(gen, gen_copy)


class Test_get_generator_state(_Base):
    @mock.patch("imgaug.random._get_generator_state_np117")
    @mock.patch("imgaug.random._get_generator_state_np116")
    def test_mocked_call(self, mock_np116, mock_np117):
        mock_np116.return_value = "np116"
        mock_np117.return_value = "np117"
        gen = iarandom.convert_seed_to_generator(1)

        result = iarandom.get_generator_state(gen)

        if isinstance(gen, np.random.RandomState):
            assert result == "np116"
            mock_np116.assert_called_once()
            assert mock_np117.call_count == 0
        else:
            assert result == "np117"
            mock_np117.assert_called_once()
            assert mock_np116.call_count == 0

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_call_np117(self):
        gen = iarandom.convert_seed_to_generator(1)
        state = iarandom.get_generator_state(gen)
        assert str(state) == str(gen.bit_generator.state)

    def test_call_np116(self):
        gen = iarandom.convert_seed_to_generator(1)
        state = iarandom.get_generator_state(gen)
        assert str(state) == str(gen.bit_generator.state)


class Test_set_generator_state_(_Base):
    @mock.patch("imgaug.random._set_generator_state_np117_")
    @mock.patch("imgaug.random._set_generator_state_np116_")
    def test_mocked_call(self, mock_np116, mock_np117):
        gen = iarandom.convert_seed_to_generator(1)
        state = {"state": 0}

        iarandom.set_generator_state_(gen, state)

        if isinstance(gen, np.random.RandomState):
            mock_np116.assert_called_once_with(gen, state)
            assert mock_np117.call_count == 0
        else:
            mock_np117.assert_called_once_with(gen, state)
            assert mock_np116.call_count == 0

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_call_np117(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen2 = np.random.Generator(iarandom.BIT_GENERATOR(2))
        gen1_copy = copylib.deepcopy(gen1)
        gen2_copy = copylib.deepcopy(gen2)

        iarandom._set_generator_state_np117_(
            gen2, iarandom.get_generator_state(gen1))

        assert iarandom.is_generator_equal_to(gen2, gen1)
        assert iarandom.is_generator_equal_to(gen1, gen1_copy)
        assert not iarandom.is_generator_equal_to(gen2, gen2_copy)

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_call_np117_via_samples(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen2 = np.random.Generator(iarandom.BIT_GENERATOR(2))
        gen1_copy = copylib.deepcopy(gen1)
        gen2_copy = copylib.deepcopy(gen2)

        iarandom._set_generator_state_np117_(
            gen2, iarandom.get_generator_state(gen1))

        samples1 = gen1.random(size=(100,))
        samples2 = gen2.random(size=(100,))
        samples1_copy = gen1_copy.random(size=(100,))
        samples2_copy = gen2_copy.random(size=(100,))

        assert np.allclose(samples1, samples2)
        assert np.allclose(samples1, samples1_copy)
        assert not np.allclose(samples2, samples2_copy)

    def test_call_np116(self):
        gen1 = np.random.RandomState(1)
        gen2 = np.random.RandomState(2)
        gen1_copy = copylib.deepcopy(gen1)
        gen2_copy = copylib.deepcopy(gen2)

        iarandom._set_generator_state_np116_(
            gen2, iarandom.get_generator_state(gen1))

        assert iarandom.is_generator_equal_to(gen2, gen1)
        assert iarandom.is_generator_equal_to(gen1, gen1_copy)
        assert not iarandom.is_generator_equal_to(gen2, gen2_copy)

    def test_call_np116_via_samples(self):
        gen1 = np.random.RandomState(1)
        gen2 = np.random.RandomState(2)
        gen1_copy = copylib.deepcopy(gen1)
        gen2_copy = copylib.deepcopy(gen2)

        iarandom._set_generator_state_np116_(
            gen2, iarandom.get_generator_state(gen1))

        samples1 = gen1.random(size=(100,))
        samples2 = gen2.random(size=(100,))
        samples1_copy = gen1_copy.random(size=(100,))
        samples2_copy = gen2_copy.random(size=(100,))

        assert np.allclose(samples1, samples2)
        assert np.allclose(samples1, samples1_copy)
        assert not np.allclose(samples2, samples2_copy)


class Test_is_generator_equal_to(_Base):
    @mock.patch("imgaug.random._is_generator_equal_to_np117")
    @mock.patch("imgaug.random._is_generator_equal_to_np116")
    def test_mocked_call(self, mock_np116, mock_np117):
        mock_np116.return_value = "np116"
        mock_np117.return_value = "np117"
        gen = iarandom.convert_seed_to_generator(1)

        result = iarandom.is_generator_equal_to(gen, gen)

        if isinstance(gen, np.random.RandomState):
            assert result == "np116"
            mock_np116.assert_called_once_with(gen, gen)
            assert mock_np117.call_count == 0
        else:
            assert result == "np117"
            mock_np117.assert_called_once_with(gen, gen)
            assert mock_np116.call_count == 0

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_generator_is_identical_np117(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))

        result = iarandom._is_generator_equal_to_np117(gen, gen)

        assert result is True

    def test_generator_is_identical_np116(self):
        gen = np.random.RandomState(1)

        result = iarandom._is_generator_equal_to_np116(gen, gen)

        assert result is True

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_generator_created_with_same_seed_np117(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen2 = np.random.Generator(iarandom.BIT_GENERATOR(1))

        result = iarandom._is_generator_equal_to_np117(gen1, gen2)

        assert result is True

    def test_generator_created_with_same_seed_np116(self):
        gen1 = np.random.RandomState(1)
        gen2 = np.random.RandomState(1)

        result = iarandom._is_generator_equal_to_np116(gen1, gen2)

        assert result is True

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_generator_is_copy_of_itself_np117(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))

        result = iarandom._is_generator_equal_to_np117(gen1,
                                                       copylib.deepcopy(gen1))

        assert result is True

    def test_generator_is_copy_of_itself_np116(self):
        gen1 = np.random.RandomState(1)

        result = iarandom._is_generator_equal_to_np116(gen1,
                                                       copylib.deepcopy(gen1))

        assert result is True

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_generator_created_with_different_seed_np117(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen2 = np.random.Generator(iarandom.BIT_GENERATOR(2))

        result = iarandom._is_generator_equal_to_np117(gen1, gen2)

        assert result is False

    def test_generator_created_with_different_seed_np116(self):
        gen1 = np.random.RandomState(1)
        gen2 = np.random.RandomState(2)

        result = iarandom._is_generator_equal_to_np116(gen1, gen2)

        assert result is False

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_generator_modified_to_have_same_state_np117(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen2 = np.random.Generator(iarandom.BIT_GENERATOR(2))
        iarandom.set_generator_state_(gen2, iarandom.get_generator_state(gen1))

        result = iarandom._is_generator_equal_to_np117(gen1, gen2)

        assert result is True

    def test_generator_modified_to_have_same_state_np116(self):
        gen1 = np.random.RandomState(1)
        gen2 = np.random.RandomState(2)
        iarandom.set_generator_state_(gen2, iarandom.get_generator_state(gen1))

        result = iarandom._is_generator_equal_to_np116(gen1, gen2)

        assert result is True


class Test_advance_generator_(_Base):
    @mock.patch("imgaug.random._advance_generator_np117_")
    @mock.patch("imgaug.random._advance_generator_np116_")
    def test_mocked_call(self, mock_np116, mock_np117):
        gen = iarandom.convert_seed_to_generator(1)

        iarandom.advance_generator_(gen)

        if isinstance(gen, np.random.RandomState):
            mock_np116.assert_called_once_with(gen)
            assert mock_np117.call_count == 0
        else:
            mock_np117.assert_called_once_with(gen)
            assert mock_np116.call_count == 0

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_call_np117(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen_copy1 = copylib.deepcopy(gen)

        iarandom._advance_generator_np117_(gen)
        gen_copy2 = copylib.deepcopy(gen)

        iarandom._advance_generator_np117_(gen)

        assert iarandom.is_generator_equal_to(gen, copylib.deepcopy(gen))
        assert not iarandom.is_generator_equal_to(gen_copy1, gen_copy2)
        assert not iarandom.is_generator_equal_to(gen_copy2, gen)
        assert not iarandom.is_generator_equal_to(gen_copy1, gen)

    def test_call_np116(self):
        gen = np.random.RandomState(1)
        gen_copy1 = copylib.deepcopy(gen)

        iarandom._advance_generator_np116_(gen)
        gen_copy2 = copylib.deepcopy(gen)

        iarandom._advance_generator_np116_(gen)

        assert iarandom.is_generator_equal_to(gen, copylib.deepcopy(gen))
        assert not iarandom.is_generator_equal_to(gen_copy1, gen_copy2)
        assert not iarandom.is_generator_equal_to(gen_copy2, gen)
        assert not iarandom.is_generator_equal_to(gen_copy1, gen)

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_samples_different_after_advance_np117(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen_copy1 = copylib.deepcopy(gen)

        # first advance
        iarandom._advance_generator_np117_(gen)
        gen_copy2 = copylib.deepcopy(gen)

        # second advance
        iarandom._advance_generator_np117_(gen)

        sample_before = gen_copy1.random()
        sample_after = gen_copy2.random()
        sample_after_after = gen.random()
        assert not np.isclose(sample_after, sample_before, rtol=0)
        assert not np.isclose(sample_after_after, sample_after, rtol=0)
        assert not np.isclose(sample_after_after, sample_before, rtol=0)

    def test_samples_different_after_advance_np116(self):
        gen = np.random.RandomState(1)
        gen_copy1 = copylib.deepcopy(gen)

        # first advance
        iarandom._advance_generator_np116_(gen)
        gen_copy2 = copylib.deepcopy(gen)

        # second advance
        iarandom._advance_generator_np116_(gen)

        sample_before = gen_copy1.random()
        sample_after = gen_copy2.random()
        sample_after_after = gen.random()
        assert not np.isclose(sample_after, sample_before, rtol=0)
        assert not np.isclose(sample_after_after, sample_after, rtol=0)
        assert not np.isclose(sample_after_after, sample_before, rtol=0)


class Test_polyfill_integers(_Base):
    def test_mocked_standard_call_np116(self):
        def side_effect(low, high=None, size=None, dtype='l'):
            return "np116"

        gen = mock.MagicMock()
        gen.randint.side_effect = side_effect

        result = iarandom.polyfill_integers(gen, 2, 2000, size=(10,),
                                            dtype="int8")

        assert result == "np116"
        gen.randint.assert_called_once_with(low=2, high=2000, size=(10,),
                                            dtype="int8")

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_mocked_standard_call_np117(self):
        def side_effect(low, high=None, size=None, dtype='int64',
                        endpoint=False):
            return "np117"

        gen = mock.MagicMock()
        gen.integers.side_effect = side_effect
        del gen.randint

        result = iarandom.polyfill_integers(gen, 2, 2000, size=(10,),
                                            dtype="int8", endpoint=True)

        assert result == "np117"
        gen.integers.assert_called_once_with(low=2, high=2000, size=(10,),
                                             dtype="int8", endpoint=True)

    def test_mocked_call_with_default_values_np116(self):
        def side_effect(low, high=None, size=None, dtype='l'):
            return "np116"

        gen = mock.MagicMock()
        gen.randint.side_effect = side_effect

        result = iarandom.polyfill_integers(gen, 2)

        assert result == "np116"
        gen.randint.assert_called_once_with(low=2, high=None, size=None,
                                            dtype="int32")

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_mocked_call_with_default_values_np117(self):
        def side_effect(low, high=None, size=None, dtype='int64',
                        endpoint=False):
            return "np117"

        gen = mock.MagicMock()
        gen.integers.side_effect = side_effect
        del gen.randint

        result = iarandom.polyfill_integers(gen, 2)

        assert result == "np117"
        gen.integers.assert_called_once_with(low=2, high=None, size=None,
                                             dtype="int32", endpoint=False)

    def test_mocked_call_with_default_values_and_endpoint_np116(self):
        def side_effect(low, high=None, size=None, dtype='l'):
            return "np116"

        gen = mock.MagicMock()
        gen.randint.side_effect = side_effect

        result = iarandom.polyfill_integers(gen, 2, endpoint=True)

        assert result == "np116"
        gen.randint.assert_called_once_with(low=0, high=3, size=None,
                                            dtype="int32")

    def test_mocked_call_with_low_high_and_endpoint_np116(self):
        def side_effect(low, high=None, size=None, dtype='l'):
            return "np116"

        gen = mock.MagicMock()
        gen.randint.side_effect = side_effect

        result = iarandom.polyfill_integers(gen, 2, 5, endpoint=True)

        assert result == "np116"
        gen.randint.assert_called_once_with(low=2, high=6, size=None,
                                            dtype="int32")

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_sampled_values_np117(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))

        result = iarandom.polyfill_integers(gen, 1, 10, size=(1000,),
                                            endpoint=False)

        assert result.dtype.name == "int32"
        assert np.all(result >= 1)
        assert np.all(result < 10)

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_sampled_values_with_endpoint_np117(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))

        result = iarandom.polyfill_integers(gen, 1, 10, size=(1000,),
                                            endpoint=True)

        assert result.dtype.name == "int32"
        assert np.all(result >= 1)
        assert np.all(result <= 10)

    def test_sampled_values_np116(self):
        gen = np.random.RandomState(1)

        result = iarandom.polyfill_integers(gen, 1, 10, size=(1000,),
                                            endpoint=False)

        assert result.dtype.name == "int32"
        assert np.all(result >= 1)
        assert np.all(result < 10)

    def test_sampled_values_with_endpoint_np116(self):
        gen = np.random.RandomState(1)

        result = iarandom.polyfill_integers(gen, 1, 10, size=(1000,),
                                            endpoint=True)

        assert result.dtype.name == "int32"
        assert np.all(result >= 1)
        assert np.all(result <= 10)


class Test_polyfill_random(_Base):
    def test_mocked_standard_call_np116(self):
        def side_effect(size=None):
            return np.zeros((1,), dtype="float64")

        gen = mock.MagicMock()
        gen.random_sample.side_effect = side_effect

        result = iarandom.polyfill_random(gen, size=(10,), dtype="float16")

        assert result.dtype.name == "float16"
        gen.random_sample.assert_called_once_with(
            size=(10,))

    def test_mocked_standard_call_np117(self):
        def side_effect(size=None, dtype='d', out=None):
            return "np117"

        gen = mock.MagicMock()
        gen.random.side_effect = side_effect
        del gen.random_sample

        result = iarandom.polyfill_random(gen, size=(10,), dtype="float16")

        assert result == "np117"
        gen.random.assert_called_once_with(
            size=(10,), dtype="float16", out=None)

    def test_mocked_call_with_out_arg_np116(self):
        def side_effect(size=None):
            return np.zeros((1,), dtype="float64")

        gen = mock.MagicMock()
        gen.random_sample.side_effect = side_effect

        out = np.empty((10,), dtype="float16")
        result = iarandom.polyfill_random(gen, size=(10,), dtype="float16",
                                          out=out)

        assert result.dtype.name == "float16"
        # np1.16 does not support an out arg, hence it is not part of the
        # expected call
        gen.random_sample.assert_called_once_with(size=(10,))

    def test_mocked_call_with_out_arg_np117(self):
        def side_effect(size=None, dtype='d', out=None):
            return "np117"

        gen = mock.MagicMock()
        gen.random.side_effect = side_effect
        del gen.random_sample

        out = np.empty((10,), dtype="float16")
        result = iarandom.polyfill_random(gen, size=(10,), dtype="float16",
                                          out=out)

        assert result == "np117"
        gen.random.assert_called_once_with(size=(10,), dtype="float16",
                                           out=out)

    def test_sampled_values_np116(self):
        gen = np.random.RandomState(1)

        result = iarandom.polyfill_random(gen, size=(1000,))

        assert result.dtype.name == "float32"
        assert np.all(result >= 0)
        assert np.all(result < 1.0)

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_sampled_values_np117(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))

        result = iarandom.polyfill_random(gen, size=(1000,))

        assert result.dtype.name == "float32"
        assert np.all(result >= 0)
        assert np.all(result < 1.0)

    def test_sampled_values_with_out_arg_np116(self):
        gen = np.random.RandomState(1)

        out = np.zeros((1000,), dtype="float32")
        result = iarandom.polyfill_random(gen, size=(1000,), out=out)

        assert result.dtype.name == "float32"
        assert np.all(result >= 0)
        assert np.all(result < 1.0)

        assert np.any(out > 0.9)
        assert np.all(out >= 0)
        assert np.all(out < 1.0)

    @unittest.skipIf(not IS_NP_117_OR_HIGHER,
                     "Function uses classes from numpy 1.17+")
    def test_sampled_values_with_out_arg_np117(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))

        out = np.zeros((1000,), dtype="float32")
        result = iarandom.polyfill_random(gen, size=(1000,), out=out)

        assert result.dtype.name == "float32"
        assert np.all(result >= 0)
        assert np.all(result < 1.0)

        assert np.any(out > 0.9)
        assert np.all(out >= 0)
        assert np.all(out < 1.0)
