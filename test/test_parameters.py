from __future__ import print_function, division, absolute_import

import itertools
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
import six.moves as sm
import skimage
import skimage.data
import skimage.morphology
import scipy
import scipy.special

import imgaug as ia
from imgaug import parameters as iap
from imgaug.testutils import reseed


def _eps(arr):
    if ia.is_np_array(arr) and arr.dtype.kind == "f":
        return np.finfo(arr.dtype).eps
    return 1e-4


class Test_handle_continuous_param(unittest.TestCase):
    def test_value_range_is_none(self):
        result = iap.handle_continuous_param(
            1, "[test1]",
            value_range=None, tuple_to_uniform=True, list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_value_range_is_tuple_of_nones(self):
        result = iap.handle_continuous_param(
            1, "[test1b]",
            value_range=(None, None),
            tuple_to_uniform=True,
            list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_param_is_stochastic_parameter(self):
        result = iap.handle_continuous_param(
            iap.Deterministic(1), "[test2]",
            value_range=None, tuple_to_uniform=True, list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_value_range_is_tuple_of_integers(self):
        result = iap.handle_continuous_param(
            1, "[test3]",
            value_range=(0, 10),
            tuple_to_uniform=True,
            list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_param_is_outside_of_value_range(self):
        with self.assertRaises(Exception) as context:
            _ = iap.handle_continuous_param(
                1, "[test4]",
                value_range=(2, 12),
                tuple_to_uniform=True,
                list_to_choice=True)
        self.assertTrue("[test4]" in str(context.exception))

    def test_param_is_inside_value_range_and_no_lower_bound(self):
        # value within value range (without lower bound)
        result = iap.handle_continuous_param(
            1, "[test5]",
            value_range=(None, 12),
            tuple_to_uniform=True,
            list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_param_is_outside_of_value_range_and_no_lower_bound(self):
        # value outside of value range (without lower bound)
        with self.assertRaises(Exception) as context:
            _ = iap.handle_continuous_param(
                1, "[test6]",
                value_range=(None, 0),
                tuple_to_uniform=True,
                list_to_choice=True)
        self.assertTrue("[test6]" in str(context.exception))

    def test_param_is_inside_value_range_and_no_upper_bound(self):
        # value within value range (without upper bound)
        result = iap.handle_continuous_param(
            1, "[test7]",
            value_range=(-1, None),
            tuple_to_uniform=True,
            list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_param_is_outside_of_value_range_and_no_upper_bound(self):
        # value outside of value range (without upper bound)
        with self.assertRaises(Exception) as context:
            _ = iap.handle_continuous_param(
                1, "[test8]",
                value_range=(2, None),
                tuple_to_uniform=True,
                list_to_choice=True)
        self.assertTrue("[test8]" in str(context.exception))

    def test_tuple_as_value_but_no_tuples_allowed(self):
        # tuple as value, but no tuples allowed
        with self.assertRaises(Exception) as context:
            _ = iap.handle_continuous_param(
                (1, 2), "[test9]",
                value_range=None,
                tuple_to_uniform=False,
                list_to_choice=True)
        self.assertTrue("[test9]" in str(context.exception))

    def test_tuple_as_value_and_tuples_allowed(self):
        # tuple as value and tuple allowed
        result = iap.handle_continuous_param(
            (1, 2), "[test10]",
            value_range=None,
            tuple_to_uniform=True,
            list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Uniform))

    def test_tuple_as_value_and_tuples_allowed_and_inside_value_range(self):
        # tuple as value and tuple allowed and tuple within value range
        result = iap.handle_continuous_param(
            (1, 2), "[test11]",
            value_range=(0, 10),
            tuple_to_uniform=True,
            list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Uniform))

    def test_tuple_value_and_allowed_and_partially_outside_value_range(self):
        # tuple as value and tuple allowed and tuple partially outside of
        # value range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_continuous_param(
                (1, 2), "[test12]",
                value_range=(1.5, 13),
                tuple_to_uniform=True,
                list_to_choice=True)
        self.assertTrue("[test12]" in str(context.exception))

    def test_tuple_value_and_allowed_and_fully_outside_value_range(self):
        # tuple as value and tuple allowed and tuple fully outside of value
        # range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_continuous_param(
                (1, 2), "[test13]",
                value_range=(3, 13),
                tuple_to_uniform=True,
                list_to_choice=True)
        self.assertTrue("[test13]" in str(context.exception))

    def test_list_as_value_but_no_lists_allowed(self):
        # list as value, but no list allowed
        with self.assertRaises(Exception) as context:
            _ = iap.handle_continuous_param(
                [1, 2, 3], "[test14]",
                value_range=None,
                tuple_to_uniform=True,
                list_to_choice=False)
        self.assertTrue("[test14]" in str(context.exception))

    def test_list_as_value_and_lists_allowed(self):
        # list as value and list allowed
        result = iap.handle_continuous_param(
            [1, 2, 3], "[test15]",
            value_range=None,
            tuple_to_uniform=True,
            list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Choice))

    def test_list_value_and_allowed_and_partially_outside_value_range(self):
        # list as value and list allowed and list partially outside of value
        # range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_continuous_param(
                [1, 2], "[test16]",
                value_range=(1.5, 13),
                tuple_to_uniform=True,
                list_to_choice=True)
        self.assertTrue("[test16]" in str(context.exception))

    def test_list_value_and_allowed_and_fully_outside_of_value_range(self):
        # list as value and list allowed and list fully outside of value range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_continuous_param(
                [1, 2], "[test17]",
                value_range=(3, 13),
                tuple_to_uniform=True,
                list_to_choice=True)
        self.assertTrue("[test17]" in str(context.exception))

    def test_value_inside_value_range_and_value_range_given_as_callable(self):
        # single value within value range given as callable
        def _value_range(x):
            return -1 < x < 1

        result = iap.handle_continuous_param(
            1, "[test18]",
            value_range=_value_range,
            tuple_to_uniform=True,
            list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_bad_datatype_as_value_range(self):
        # bad datatype for value range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_continuous_param(
                1, "[test19]",
                value_range=False,
                tuple_to_uniform=True,
                list_to_choice=True)
        self.assertTrue(
            "Unexpected input for value_range" in str(context.exception))


class Test_handle_discrete_param(unittest.TestCase):
    def test_float_value_inside_value_range_but_no_floats_allowed(self):
        # float value without value range when no float value is allowed
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                1.5, "[test0]",
                value_range=None,
                tuple_to_uniform=True,
                list_to_choice=True, allow_floats=False)
        self.assertTrue("[test0]" in str(context.exception))

    def test_value_range_is_none(self):
        # value without value range
        result = iap.handle_discrete_param(
            1, "[test1]", value_range=None, tuple_to_uniform=True,
            list_to_choice=True, allow_floats=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_value_range_is_tuple_of_nones(self):
        # value without value range as (None, None)
        result = iap.handle_discrete_param(
            1, "[test1b]", value_range=(None, None), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_value_is_stochastic_parameter(self):
        # stochastic parameter
        result = iap.handle_discrete_param(
            iap.Deterministic(1), "[test2]", value_range=None,
            tuple_to_uniform=True, list_to_choice=True, allow_floats=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_value_inside_value_range(self):
        # value within value range
        result = iap.handle_discrete_param(
            1, "[test3]", value_range=(0, 10), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_value_outside_value_range(self):
        # value outside of value range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                1, "[test4]", value_range=(2, 12), tuple_to_uniform=True,
                list_to_choice=True, allow_floats=True)
        self.assertTrue("[test4]" in str(context.exception))

    def test_value_inside_value_range_no_lower_bound(self):
        # value within value range (without lower bound)
        result = iap.handle_discrete_param(
            1, "[test5]", value_range=(None, 12), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_value_outside_value_range_no_lower_bound(self):
        # value outside of value range (without lower bound)
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                1, "[test6]", value_range=(None, 0), tuple_to_uniform=True,
                list_to_choice=True, allow_floats=True)
        self.assertTrue("[test6]" in str(context.exception))

    def test_value_inside_value_range_no_upper_bound(self):
        # value within value range (without upper bound)
        result = iap.handle_discrete_param(
            1, "[test7]", value_range=(-1, None), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_value_outside_value_range_no_upper_bound(self):
        # value outside of value range (without upper bound)
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                1, "[test8]", value_range=(2, None), tuple_to_uniform=True,
                list_to_choice=True, allow_floats=True)
        self.assertTrue("[test8]" in str(context.exception))

    def test_value_is_tuple_but_no_tuples_allowed(self):
        # tuple as value, but no tuples allowed
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                (1, 2), "[test9]", value_range=None, tuple_to_uniform=False,
                list_to_choice=True, allow_floats=True)
        self.assertTrue("[test9]" in str(context.exception))

    def test_value_is_tuple_and_tuples_allowed(self):
        # tuple as value and tuple allowed
        result = iap.handle_discrete_param(
            (1, 2), "[test10]", value_range=None, tuple_to_uniform=True,
            list_to_choice=True, allow_floats=True)
        self.assertTrue(isinstance(result, iap.DiscreteUniform))

    def test_value_tuple_and_allowed_and_inside_value_range(self):
        # tuple as value and tuple allowed and tuple within value range
        result = iap.handle_discrete_param(
            (1, 2), "[test11]", value_range=(0, 10), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=True)
        self.assertTrue(isinstance(result, iap.DiscreteUniform))

    def test_value_tuple_and_allowed_and_inside_vr_allow_floats_false(self):
        # tuple as value and tuple allowed and tuple within value range with
        # allow_floats=False
        result = iap.handle_discrete_param(
            (1, 2), "[test11b]", value_range=(0, 10),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.assertTrue(isinstance(result, iap.DiscreteUniform))

    def test_value_tuple_and_allowed_and_partially_outside_value_range(self):
        # tuple as value and tuple allowed and tuple partially outside of
        # value range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                (1, 3), "[test12]", value_range=(2, 13), tuple_to_uniform=True,
                list_to_choice=True, allow_floats=True)
        self.assertTrue("[test12]" in str(context.exception))

    def test_value_tuple_and_allowed_and_fully_outside_value_range(self):
        # tuple as value and tuple allowed and tuple fully outside of value
        # range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                (1, 2), "[test13]", value_range=(3, 13), tuple_to_uniform=True,
                list_to_choice=True, allow_floats=True)
        self.assertTrue("[test13]" in str(context.exception))

    def test_value_list_but_not_allowed(self):
        # list as value, but no list allowed
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                [1, 2, 3], "[test14]", value_range=None, tuple_to_uniform=True,
                list_to_choice=False, allow_floats=True)
        self.assertTrue("[test14]" in str(context.exception))

    def test_value_list_and_allowed(self):
        # list as value and list allowed
        result = iap.handle_discrete_param(
            [1, 2, 3], "[test15]", value_range=None, tuple_to_uniform=True,
            list_to_choice=True, allow_floats=True)
        self.assertTrue(isinstance(result, iap.Choice))

    def test_value_list_and_allowed_and_partially_outside_value_range(self):
        # list as value and list allowed and list partially outside of value range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                [1, 3], "[test16]", value_range=(2, 13), tuple_to_uniform=True,
                list_to_choice=True, allow_floats=True)
        self.assertTrue("[test16]" in str(context.exception))

    def test_value_list_and_allowed_and_fully_outside_value_range(self):
        # list as value and list allowed and list fully outside of value range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                [1, 2], "[test17]", value_range=(3, 13), tuple_to_uniform=True,
                list_to_choice=True, allow_floats=True)
        self.assertTrue("[test17]" in str(context.exception))

    def test_value_inside_value_range_given_as_callable(self):
        # single value within value range given as callable
        def _value_range(x):
            return -1 < x < 1

        result = iap.handle_discrete_param(
            1, "[test18]",
            value_range=_value_range,
            tuple_to_uniform=True,
            list_to_choice=True)
        self.assertTrue(isinstance(result, iap.Deterministic))

    def test_bad_datatype_as_value_range(self):
        # bad datatype for value range
        with self.assertRaises(Exception) as context:
            _ = iap.handle_discrete_param(
                1, "[test19]", value_range=False, tuple_to_uniform=True,
                list_to_choice=True)
        self.assertTrue(
            "Unexpected input for value_range" in str(context.exception))


class Test_handle_probability_param(unittest.TestCase):
    def test_bool_like_values(self):
        for val in [True, False, 0, 1, 0.0, 1.0]:
            with self.subTest(param=val):
                p = iap.handle_probability_param(val, "[test1]")
                assert isinstance(p, iap.Deterministic)
                assert p.value == int(val)

    def test_float_probabilities(self):
        for val in [0.0001, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999, 0.9999]:
            with self.subTest(param=val):
                p = iap.handle_probability_param(val, "[test2]")
                assert isinstance(p, iap.Binomial)
                assert isinstance(p.p, iap.Deterministic)
                assert val-1e-8 < p.p.value < val+1e-8

    def test_probability_is_stochastic_parameter(self):
        det = iap.Deterministic(1)
        p = iap.handle_probability_param(det, "[test3]")
        assert p == det

    def test_probability_has_bad_datatype(self):
        with self.assertRaises(Exception) as context:
            _p = iap.handle_probability_param("test", "[test4]")
        self.assertTrue("Expected " in str(context.exception))

    def test_probability_is_negative(self):
        with self.assertRaises(AssertionError):
            _p = iap.handle_probability_param(-0.01, "[test5]")

    def test_probability_is_above_100_percent(self):
        with self.assertRaises(AssertionError):
            _p = iap.handle_probability_param(1.01, "[test6]")


class Test_force_np_float_dtype(unittest.TestCase):
    def test_common_dtypes(self):
        dtypes = [
            ("float16", "float16"),
            ("float32", "float32"),
            ("float64", "float64"),
            ("uint8", "float64"),
            ("int32", "float64")
        ]
        for dtype_in, expected in dtypes:
            with self.subTest(dtype_in=dtype_in):
                arr = np.zeros((1,), dtype=dtype_in)
                observed = iap.force_np_float_dtype(arr).dtype
                assert observed.name == expected


class Test_both_np_float_if_one_is_float(unittest.TestCase):
    def test_float16_float32(self):
        a1 = np.zeros((1,), dtype=np.float16)
        b1 = np.zeros((1,), dtype=np.float32)
        a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
        assert a2.dtype.name == "float16"
        assert b2.dtype.name == "float32"

    def test_float16_int32(self):
        a1 = np.zeros((1,), dtype=np.float16)
        b1 = np.zeros((1,), dtype=np.int32)
        a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
        assert a2.dtype.name == "float16"
        assert b2.dtype.name == "float64"

    def test_int32_float16(self):
        a1 = np.zeros((1,), dtype=np.int32)
        b1 = np.zeros((1,), dtype=np.float16)
        a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
        assert a2.dtype.name == "float64"
        assert b2.dtype.name == "float16"

    def test_int32_uint8(self):
        a1 = np.zeros((1,), dtype=np.int32)
        b1 = np.zeros((1,), dtype=np.uint8)
        a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
        assert a2.dtype.name == "float64"
        assert b2.dtype.name == "float64"


class Test_draw_distributions_grid(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_basic_functionality(self):
        params = [mock.Mock(), mock.Mock()]
        params[0].draw_distribution_graph.return_value = \
            np.zeros((1, 1, 3), dtype=np.uint8)
        params[1].draw_distribution_graph.return_value = \
            np.zeros((1, 1, 3), dtype=np.uint8)

        draw_grid_mock = mock.Mock()
        draw_grid_mock.return_value = np.zeros((4, 3, 2), dtype=np.uint8)
        with mock.patch('imgaug.imgaug.draw_grid', draw_grid_mock):
            grid_observed = iap.draw_distributions_grid(
                params, rows=2, cols=3, graph_sizes=(20, 21),
                sample_sizes=[(1, 2), (3, 4)], titles=["A", "B"])

        assert grid_observed.shape == (4, 3, 2)
        assert params[0].draw_distribution_graph.call_count == 1
        assert params[1].draw_distribution_graph.call_count == 1
        assert params[0].draw_distribution_graph.call_args[1]["size"] == (1, 2)
        assert params[0].draw_distribution_graph.call_args[1]["title"] == "A"
        assert params[1].draw_distribution_graph.call_args[1]["size"] == (3, 4)
        assert params[1].draw_distribution_graph.call_args[1]["title"] == "B"
        assert draw_grid_mock.call_count == 1
        assert draw_grid_mock.call_args[0][0][0].shape == (20, 21, 3)
        assert draw_grid_mock.call_args[0][0][1].shape == (20, 21, 3)
        assert draw_grid_mock.call_args[1]["rows"] == 2
        assert draw_grid_mock.call_args[1]["cols"] == 3


class Test_draw_distributions_graph(unittest.TestCase):
    def test_basic_functionality(self):
        # this test is very rough as we get a not-very-well-defined image out
        # of the function
        param = iap.Uniform(0.0, 1.0)

        graph_img = param.draw_distribution_graph(title=None, size=(10000,),
                                                  bins=100)

        # at least 10% of the image should be white-ish (background)
        nb_white = np.sum(graph_img[..., :] > [200, 200, 200])
        nb_all = np.prod(graph_img.shape)

        graph_img_title = param.draw_distribution_graph(title="test",
                                                        size=(10000,),
                                                        bins=100)

        assert graph_img.ndim == 3
        assert graph_img.shape[2] == 3
        assert nb_white > 0.1 * nb_all
        assert graph_img_title.ndim == 3
        assert graph_img_title.shape[2] == 3
        assert not np.array_equal(graph_img_title, graph_img)


class TestStochasticParameter(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_copy(self):
        other_param = iap.Uniform(1.0, 10.0)
        param = iap.Discretize(other_param)
        other_param.a = [1.0]
        param_copy = param.copy()

        param.other_param.a[0] += 1

        assert isinstance(param_copy, iap.Discretize)
        assert isinstance(param_copy.other_param, iap.Uniform)
        assert param_copy.other_param.a[0] == param.other_param.a[0]

    def test_deepcopy(self):
        other_param = iap.Uniform(1.0, 10.0)
        param = iap.Discretize(other_param)
        other_param.a = [1.0]
        param_copy = param.deepcopy()

        param.other_param.a[0] += 1

        assert isinstance(param_copy, iap.Discretize)
        assert isinstance(param_copy.other_param, iap.Uniform)
        assert param_copy.other_param.a[0] != param.other_param.a[0]


class TestStochasticParameterOperators(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_multiply_stochasic_params(self):
        param1 = iap.Normal(0, 1)
        param2 = iap.Uniform(-1.0, 1.0)

        param3 = param1 * param2

        assert isinstance(param3, iap.Multiply)
        assert param3.other_param == param1
        assert param3.val == param2

    def test_multiply_stochastic_param_with_integer(self):
        param1 = iap.Normal(0, 1)

        param3 = param1 * 2

        assert isinstance(param3, iap.Multiply)
        assert param3.other_param == param1
        assert isinstance(param3.val, iap.Deterministic)
        assert param3.val.value == 2

    def test_multiply_integer_with_stochastic_param(self):
        param1 = iap.Normal(0, 1)

        param3 = 2 * param1

        assert isinstance(param3, iap.Multiply)
        assert isinstance(param3.other_param, iap.Deterministic)
        assert param3.other_param.value == 2
        assert param3.val == param1

    def test_multiply_string_with_stochastic_param_should_fail(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = "test" * param1

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_multiply_stochastic_param_with_string_should_fail(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = param1 * "test"

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_divide_stochastic_params(self):
        # Divide (__truediv__)
        param1 = iap.Normal(0, 1)
        param2 = iap.Uniform(-1.0, 1.0)

        param3 = param1 / param2

        assert isinstance(param3, iap.Divide)
        assert param3.other_param == param1
        assert param3.val == param2

    def test_divide_stochastic_param_by_integer(self):
        param1 = iap.Normal(0, 1)

        param3 = param1 / 2

        assert isinstance(param3, iap.Divide)
        assert param3.other_param == param1
        assert isinstance(param3.val, iap.Deterministic)
        assert param3.val.value == 2

    def test_divide_integer_by_stochastic_param(self):
        param1 = iap.Normal(0, 1)

        param3 = 2 / param1

        assert isinstance(param3, iap.Divide)
        assert isinstance(param3.other_param, iap.Deterministic)
        assert param3.other_param.value == 2
        assert param3.val == param1

    def test_divide_string_by_stochastic_param_should_fail(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = "test" / param1

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_divide_stochastic_param_by_string_should_fail(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = param1 / "test"

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_div_stochastic_params(self):
        # Divide (__div__)
        param1 = iap.Normal(0, 1)
        param2 = iap.Uniform(-1.0, 1.0)

        param3 = param1.__div__(param2)

        assert isinstance(param3, iap.Divide)
        assert param3.other_param == param1
        assert param3.val == param2

    def test_div_stochastic_param_by_integer(self):
        param1 = iap.Normal(0, 1)

        param3 = param1.__div__(2)

        assert isinstance(param3, iap.Divide)
        assert param3.other_param == param1
        assert isinstance(param3.val, iap.Deterministic)
        assert param3.val.value == 2

    def test_div_stochastic_param_by_string_should_fail(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = param1.__div__("test")

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_rdiv_stochastic_param_by_integer(self):
        # Divide (__rdiv__)
        param1 = iap.Normal(0, 1)

        param3 = param1.__rdiv__(2)

        assert isinstance(param3, iap.Divide)
        assert isinstance(param3.other_param, iap.Deterministic)
        assert param3.other_param.value == 2
        assert param3.val == param1

    def test_rdiv_stochastic_param_by_string_should_fail(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = param1.__rdiv__("test")

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_floordiv_stochastic_params(self):
        # Divide (__floordiv__)
        param1_int = iap.DiscreteUniform(0, 10)
        param2_int = iap.Choice([1, 2])

        param3 = param1_int // param2_int

        assert isinstance(param3, iap.Discretize)
        assert isinstance(param3.other_param, iap.Divide)
        assert param3.other_param.other_param == param1_int
        assert param3.other_param.val == param2_int

    def test_floordiv_symbol_stochastic_param_by_integer(self):
        param1_int = iap.DiscreteUniform(0, 10)

        param3 = param1_int // 2

        assert isinstance(param3, iap.Discretize)
        assert isinstance(param3.other_param, iap.Divide)
        assert param3.other_param.other_param == param1_int
        assert isinstance(param3.other_param.val, iap.Deterministic)
        assert param3.other_param.val.value == 2

    def test_floordiv_symbol_integer_by_stochastic_param(self):
        param1_int = iap.DiscreteUniform(0, 10)

        param3 = 2 // param1_int

        assert isinstance(param3, iap.Discretize)
        assert isinstance(param3.other_param, iap.Divide)
        assert isinstance(param3.other_param.other_param, iap.Deterministic)
        assert param3.other_param.other_param.value == 2
        assert param3.other_param.val == param1_int

    def test_floordiv_symbol_string_by_stochastic_should_fail(self):
        param1_int = iap.DiscreteUniform(0, 10)

        with self.assertRaises(Exception) as context:
            _ = "test" // param1_int

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_floordiv_symbol_stochastic_param_by_string_should_fail(self):
        param1_int = iap.DiscreteUniform(0, 10)

        with self.assertRaises(Exception) as context:
            _ = param1_int // "test"

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_add_stochastic_params(self):
        param1 = iap.Normal(0, 1)
        param2 = iap.Uniform(-1.0, 1.0)

        param3 = param1 + param2

        assert isinstance(param3, iap.Add)
        assert param3.other_param == param1
        assert param3.val == param2

    def test_add_integer_to_stochastic_param(self):
        param1 = iap.Normal(0, 1)

        param3 = param1 + 2

        assert isinstance(param3, iap.Add)
        assert param3.other_param == param1
        assert isinstance(param3.val, iap.Deterministic)
        assert param3.val.value == 2

    def test_add_stochastic_param_to_integer(self):
        param1 = iap.Normal(0, 1)

        param3 = 2 + param1

        assert isinstance(param3, iap.Add)
        assert isinstance(param3.other_param, iap.Deterministic)
        assert param3.other_param.value == 2
        assert param3.val == param1

    def test_add_stochastic_param_to_string(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = "test" + param1

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_add_string_to_stochastic_param(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = param1 + "test"

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_subtract_stochastic_params(self):
        param1 = iap.Normal(0, 1)
        param2 = iap.Uniform(-1.0, 1.0)

        param3 = param1 - param2

        assert isinstance(param3, iap.Subtract)
        assert param3.other_param == param1
        assert param3.val == param2

    def test_subtract_integer_from_stochastic_param(self):
        param1 = iap.Normal(0, 1)

        param3 = param1 - 2

        assert isinstance(param3, iap.Subtract)
        assert param3.other_param == param1
        assert isinstance(param3.val, iap.Deterministic)
        assert param3.val.value == 2

    def test_subtract_stochastic_param_from_integer(self):
        param1 = iap.Normal(0, 1)

        param3 = 2 - param1

        assert isinstance(param3, iap.Subtract)
        assert isinstance(param3.other_param, iap.Deterministic)
        assert param3.other_param.value == 2
        assert param3.val == param1

    def test_subtract_stochastic_param_from_string_should_fail(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = "test" - param1

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_subtract_string_from_stochastic_param_should_fail(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = param1 - "test"

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_exponentiate_stochastic_params(self):
        param1 = iap.Normal(0, 1)
        param2 = iap.Uniform(-1.0, 1.0)

        param3 = param1 ** param2

        assert isinstance(param3, iap.Power)
        assert param3.other_param == param1
        assert param3.val == param2

    def test_exponentiate_stochastic_param_by_integer(self):
        param1 = iap.Normal(0, 1)

        param3 = param1 ** 2

        assert isinstance(param3, iap.Power)
        assert param3.other_param == param1
        assert isinstance(param3.val, iap.Deterministic)
        assert param3.val.value == 2

    def test_exponentiate_integer_by_stochastic_param(self):
        param1 = iap.Normal(0, 1)

        param3 = 2 ** param1

        assert isinstance(param3, iap.Power)
        assert isinstance(param3.other_param, iap.Deterministic)
        assert param3.other_param.value == 2
        assert param3.val == param1

    def test_exponentiate_string_by_stochastic_param(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = "test" ** param1

        self.assertTrue("Invalid datatypes" in str(context.exception))

    def test_exponentiate_stochastic_param_by_string(self):
        param1 = iap.Normal(0, 1)

        with self.assertRaises(Exception) as context:
            _ = param1 ** "test"

        self.assertTrue("Invalid datatypes" in str(context.exception))


class TestBinomial(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___p_is_zero(self):
        param = iap.Binomial(0)
        assert (
            param.__str__()
            == param.__repr__()
            == "Binomial(Deterministic(int 0))"
        )

    def test___init___p_is_one(self):
        param = iap.Binomial(1.0)
        assert (
            param.__str__()
            == param.__repr__()
            == "Binomial(Deterministic(float 1.00000000))"
        )

    def test_p_is_zero(self):
        param = iap.Binomial(0)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample == 0
        assert np.all(samples == 0)

    def test_p_is_one(self):
        param = iap.Binomial(1.0)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample == 1
        assert np.all(samples == 1)

    def test_p_is_50_percent(self):
        param = iap.Binomial(0.5)

        sample = param.draw_sample()
        samples = param.draw_samples((10000,))
        unique, counts = np.unique(samples, return_counts=True)

        assert sample.shape == tuple()
        assert samples.shape == (10000,)
        assert sample in [0, 1]
        assert len(unique) == 2
        for val, count in zip(unique, counts):
            if val == 0:
                assert 5000 - 500 < count < 5000 + 500
            elif val == 1:
                assert 5000 - 500 < count < 5000 + 500
            else:
                assert False

    def test_p_is_list(self):
        param = iap.Binomial(iap.Choice([0.25, 0.75]))
        for _ in sm.xrange(10):
            samples = param.draw_samples((1000,))
            p = np.sum(samples) / samples.size
            assert (
                (0.25 - 0.05 < p < 0.25 + 0.05)
                or (0.75 - 0.05 < p < 0.75 + 0.05)
            )

    def test_p_is_tuple(self):
        param = iap.Binomial((0.0, 1.0))

        last_p = 0.5
        diffs = []
        for _ in sm.xrange(30):
            samples = param.draw_samples((1000,))
            p = np.sum(samples).astype(np.float32) / samples.size
            diffs.append(abs(p - last_p))
            last_p = p
        nb_p_changed = sum([diff > 0.05 for diff in diffs])

        assert nb_p_changed > 15

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Binomial(0.5)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.array_equal(samples1, samples2)


class TestChoice(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Choice([0, 1, 2])
        assert (
            param.__str__()
            == param.__repr__()
            == "Choice(a=[0, 1, 2], replace=True, p=None)"
        )

    def test_value_is_list(self):
        param = iap.Choice([0, 1, 2])

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample in [0, 1, 2]
        assert np.all(
            np.logical_or(
                np.logical_or(samples == 0, samples == 1),
                samples == 2
            )
        )

    def test_sampled_values_match_expected_counts(self):
        param = iap.Choice([0, 1, 2])

        samples = param.draw_samples((10000,))
        expected = 10000/3
        expected_tolerance = expected * 0.05
        for v in [0, 1, 2]:
            count = np.sum(samples == v)
            assert (
                expected - expected_tolerance
                < count <
                expected + expected_tolerance
            )

    def test_value_is_list_containing_negative_number(self):
        param = iap.Choice([-1, 1])

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample in [-1, 1]
        assert np.all(np.logical_or(samples == -1, samples == 1))

    def test_value_is_list_of_floats(self):
        param = iap.Choice([-1.2, 1.7])

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert (
            (
                -1.2 - _eps(sample)
                < sample <
                -1.2 + _eps(sample)
            )
            or
            (
                1.7 - _eps(sample)
                < sample <
                1.7 + _eps(sample)
            )
        )
        assert np.all(
            np.logical_or(
                np.logical_and(
                    -1.2 - _eps(sample) < samples,
                    samples < -1.2 + _eps(sample)
                ),
                np.logical_and(
                    1.7 - _eps(sample) < samples,
                    samples < 1.7 + _eps(sample)
                )
            )
        )

    def test_value_is_list_of_strings(self):
        param = iap.Choice(["first", "second", "third"])

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample in ["first", "second", "third"]
        assert np.all(
            np.logical_or(
                np.logical_or(
                    samples == "first",
                    samples == "second"
                ),
                samples == "third"
            )
        )

    def test_sample_without_replacing(self):
        param = iap.Choice([1+i for i in sm.xrange(100)], replace=False)

        samples = param.draw_samples((50,))
        seen = [0 for _ in sm.xrange(100)]
        for sample in samples:
            seen[sample-1] += 1

        assert all([count in [0, 1] for count in seen])

    def test_non_uniform_probabilities_over_elements(self):
        param = iap.Choice([0, 1], p=[0.25, 0.75])

        samples = param.draw_samples((10000,))
        unique, counts = np.unique(samples, return_counts=True)

        assert len(unique) == 2
        for val, count in zip(unique, counts):
            if val == 0:
                assert 2500 - 500 < count < 2500 + 500
            elif val == 1:
                assert 7500 - 500 < count < 7500 + 500
            else:
                assert False

    def test_list_contains_stochastic_parameter(self):
        param = iap.Choice([iap.Choice([0, 1]), 2])

        samples = param.draw_samples((10000,))
        unique, counts = np.unique(samples, return_counts=True)

        assert len(unique) == 3
        for val, count in zip(unique, counts):
            if val in [0, 1]:
                assert 2500 - 500 < count < 2500 + 500
            elif val == 2:
                assert 5000 - 500 < count < 5000 + 500
            else:
                assert False

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Choice([-1, 0, 1, 2, 3])

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.array_equal(samples1, samples2)

    def test_value_is_bad_datatype(self):
        with self.assertRaises(Exception) as context:
            _ = iap.Choice(123)

        self.assertTrue(
            "Expected a to be an iterable" in str(context.exception))

    def test_p_is_bad_datatype(self):
        with self.assertRaises(Exception) as context:
            _ = iap.Choice([1, 2], p=123)

        self.assertTrue("Expected p to be" in str(context.exception))

    def test_value_and_p_have_unequal_lengths(self):
        with self.assertRaises(Exception) as context:
            _ = iap.Choice([1, 2], p=[1])

        self.assertTrue("Expected lengths of" in str(context.exception))


class TestDiscreteUniform(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.DiscreteUniform(0, 2)
        assert (
            param.__str__()
            == param.__repr__()
            == "DiscreteUniform(Deterministic(int 0), Deterministic(int 2))"
        )

    def test_bounds_are_ints(self):
        param = iap.DiscreteUniform(0, 2)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample in [0, 1, 2]
        assert np.all(
            np.logical_or(
                np.logical_or(samples == 0, samples == 1),
                samples == 2
            )
        )

    def test_samples_match_expected_counts(self):
        param = iap.DiscreteUniform(0, 2)

        samples = param.draw_samples((10000,))
        expected = 10000/3
        expected_tolerance = expected * 0.05
        for v in [0, 1, 2]:
            count = np.sum(samples == v)
            assert (
                expected - expected_tolerance
                < count <
                expected + expected_tolerance
            )

    def test_lower_bound_is_negative(self):
        param = iap.DiscreteUniform(-1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample in [-1, 0, 1]
        assert np.all(
            np.logical_or(
                np.logical_or(samples == -1, samples == 0),
                samples == 1
            )
        )

    def test_bounds_are_floats(self):
        param = iap.DiscreteUniform(-1.2, 1.2)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample in [-1, 0, 1]
        assert np.all(
            np.logical_or(
                np.logical_or(
                    samples == -1, samples == 0
                ),
                samples == 1
            )
        )

    def test_lower_and_upper_bound_have_wrong_order(self):
        param = iap.DiscreteUniform(1, -1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample in [-1, 0, 1]
        assert np.all(
            np.logical_or(
                np.logical_or(
                    samples == -1, samples == 0
                ),
                samples == 1
            )
        )

    def test_lower_and_upper_bound_are_the_same(self):
        param = iap.DiscreteUniform(1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((100,))

        assert sample == 1
        assert np.all(samples == 1)

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Uniform(-1, 1)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.array_equal(samples1, samples2)


class TestPoisson(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Poisson(1)
        assert (
            param.__str__()
            == param.__repr__()
            == "Poisson(Deterministic(int 1))"
        )

    def test_draw_sample(self):
        param = iap.Poisson(1)

        sample = param.draw_sample()

        assert sample.shape == tuple()
        assert 0 <= sample

    def test_via_comparison_to_np_poisson(self):
        param = iap.Poisson(1)

        samples = param.draw_samples((100, 1000))
        samples_direct = np.random.RandomState(1234).poisson(
            lam=1, size=(100, 1000))
        assert samples.shape == (100, 1000)

        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            count_direct = int(np.sum(samples_direct == i))
            count = np.sum(samples == i)
            tolerance = max(count_direct * 0.1, 250)
            assert count_direct - tolerance < count < count_direct + tolerance

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Poisson(1)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.array_equal(samples1, samples2)


class TestNormal(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Normal(0, 1)
        assert (
            param.__str__()
            == param.__repr__()
            == "Normal(loc=Deterministic(int 0), scale=Deterministic(int 1))"
        )

    def test_draw_sample(self):
        param = iap.Normal(0, 1)

        sample = param.draw_sample()

        assert sample.shape == tuple()

    def test_via_comparison_to_np_normal(self):
        param = iap.Normal(0, 1)

        samples = param.draw_samples((100, 1000))
        samples_direct = np.random.RandomState(1234).normal(loc=0, scale=1,
                                                            size=(100, 1000))
        samples = np.clip(samples, -1, 1)
        samples_direct = np.clip(samples_direct, -1, 1)
        nb_bins = 10
        hist, _ = np.histogram(samples, bins=nb_bins, range=(-1.0, 1.0),
                               density=False)
        hist_direct, _ = np.histogram(samples_direct, bins=nb_bins,
                                      range=(-1.0, 1.0), density=False)
        tolerance = 0.05
        for nb_samples, nb_samples_direct in zip(hist, hist_direct):
            density = nb_samples / samples.size
            density_direct = nb_samples_direct / samples_direct.size
            assert (
                density_direct - tolerance
                < density <
                density_direct + tolerance
            )

    def test_loc_is_stochastic_parameter(self):
        param = iap.Normal(iap.Choice([-100, 100]), 1)

        seen = [0, 0]
        for _ in sm.xrange(1000):
            samples = param.draw_samples((100,))
            exp = np.mean(samples)

            if -100 - 10 < exp < -100 + 10:
                seen[0] += 1
            elif 100 - 10 < exp < 100 + 10:
                seen[1] += 1
            else:
                assert False
        assert 500 - 100 < seen[0] < 500 + 100
        assert 500 - 100 < seen[1] < 500 + 100

    def test_scale(self):
        param1 = iap.Normal(0, 1)
        param2 = iap.Normal(0, 100)

        samples1 = param1.draw_samples((1000,))
        samples2 = param2.draw_samples((1000,))

        assert np.std(samples1) < np.std(samples2)
        assert 100 - 10 < np.std(samples2) < 100 + 10

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Normal(0, 1)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.allclose(samples1, samples2)


class TestTruncatedNormal(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.TruncatedNormal(0, 1)
        expected = (
            "TruncatedNormal("
            "loc=Deterministic(int 0), "
            "scale=Deterministic(int 1), "
            "low=Deterministic(float -inf), "
            "high=Deterministic(float inf)"
            ")"
        )
        assert (
            param.__str__()
            == param.__repr__()
            == expected
        )

    def test___init___custom_range(self):
        param = iap.TruncatedNormal(0, 1, low=-100, high=50.0)
        expected = (
            "TruncatedNormal("
            "loc=Deterministic(int 0), "
            "scale=Deterministic(int 1), "
            "low=Deterministic(int -100), "
            "high=Deterministic(float 50.00000000)"
            ")"
        )
        assert (
            param.__str__()
            == param.__repr__()
            == expected
        )

    def test_scale_is_zero(self):
        param = iap.TruncatedNormal(0.5, 0, low=-10, high=10)
        samples = param.draw_samples((100,))
        assert np.allclose(samples, 0.5)

    def test_scale(self):
        param1 = iap.TruncatedNormal(0.0, 0.1, low=-100, high=100)
        param2 = iap.TruncatedNormal(0.0, 5.0, low=-100, high=100)
        samples1 = param1.draw_samples((1000,))
        samples2 = param2.draw_samples((1000,))
        assert np.std(samples1) < np.std(samples2)
        assert np.isclose(np.std(samples1), 0.1, rtol=0, atol=0.20)
        assert np.isclose(np.std(samples2), 5.0, rtol=0, atol=0.40)

    def test_loc_is_stochastic_parameter(self):
        param = iap.TruncatedNormal(iap.Choice([-100, 100]), 0.01,
                                    low=-1000, high=1000)

        seen = [0, 0]
        for _ in sm.xrange(200):
            samples = param.draw_samples((5,))
            observed = np.mean(samples)

            dist1 = np.abs(-100 - observed)
            dist2 = np.abs(100 - observed)

            if dist1 < 1:
                seen[0] += 1
            elif dist2 < 1:
                seen[1] += 1
            else:
                assert False
        assert np.isclose(seen[0], 100, rtol=0, atol=20)
        assert np.isclose(seen[1], 100, rtol=0, atol=20)

    def test_samples_are_within_bounds(self):
        param = iap.TruncatedNormal(0, 10.0, low=-5, high=7.5)

        samples = param.draw_samples((1000,))

        # are all within bounds
        assert np.all(samples >= -5.0 - 1e-4)
        assert np.all(samples <= 7.5 + 1e-4)

        # at least some samples close to bounds
        assert np.any(samples <= -4.5)
        assert np.any(samples >= 7.0)

        # at least some samples close to loc
        assert np.any(np.abs(samples) < 0.5)

    def test_samples_same_values_for_same_seeds(self):
        param = iap.TruncatedNormal(0, 1)

        samples1 = param.draw_samples((10, 5), random_state=1234)
        samples2 = param.draw_samples((10, 5), random_state=1234)

        assert np.allclose(samples1, samples2)

    def test_samples_different_values_for_different_seeds(self):
        param = iap.TruncatedNormal(0, 1)

        samples1 = param.draw_samples((10, 5), random_state=1234)
        samples2 = param.draw_samples((10, 5), random_state=2345)

        assert not np.allclose(samples1, samples2)


class TestLaplace(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Laplace(0, 1)
        assert (
            param.__str__()
            == param.__repr__()
            == "Laplace(loc=Deterministic(int 0), scale=Deterministic(int 1))"
        )

    def test_draw_sample(self):
        param = iap.Laplace(0, 1)

        sample = param.draw_sample()

        assert sample.shape == tuple()

    def test_via_comparison_to_np_laplace(self):
        param = iap.Laplace(0, 1)

        samples = param.draw_samples((100, 1000))
        samples_direct = np.random.RandomState(1234).laplace(loc=0, scale=1,
                                                             size=(100, 1000))

        assert samples.shape == (100, 1000)

        samples = np.clip(samples, -1, 1)
        samples_direct = np.clip(samples_direct, -1, 1)
        nb_bins = 10
        hist, _ = np.histogram(samples, bins=nb_bins, range=(-1.0, 1.0),
                               density=False)
        hist_direct, _ = np.histogram(samples_direct, bins=nb_bins,
                                      range=(-1.0, 1.0), density=False)
        tolerance = 0.05
        for nb_samples, nb_samples_direct in zip(hist, hist_direct):
            density = nb_samples / samples.size
            density_direct = nb_samples_direct / samples_direct.size
            assert (
                density_direct - tolerance
                < density <
                density_direct + tolerance
            )

    def test_loc_is_stochastic_parameter(self):
        param = iap.Laplace(iap.Choice([-100, 100]), 1)

        seen = [0, 0]
        for _ in sm.xrange(1000):
            samples = param.draw_samples((100,))
            exp = np.mean(samples)

            if -100 - 10 < exp < -100 + 10:
                seen[0] += 1
            elif 100 - 10 < exp < 100 + 10:
                seen[1] += 1
            else:
                assert False

        assert 500 - 100 < seen[0] < 500 + 100
        assert 500 - 100 < seen[1] < 500 + 100

    def test_scale(self):
        param1 = iap.Laplace(0, 1)
        param2 = iap.Laplace(0, 100)

        samples1 = param1.draw_samples((1000,))
        samples2 = param2.draw_samples((1000,))

        assert np.var(samples1) < np.var(samples2)

    def test_scale_is_zero(self):
        param1 = iap.Laplace(1, 0)

        samples = param1.draw_samples((100,))

        assert np.all(np.logical_and(
            samples > 1 - _eps(samples),
            samples < 1 + _eps(samples)
        ))

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Laplace(0, 1)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.allclose(samples1, samples2)


class TestChiSquare(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.ChiSquare(1)

        assert (
            param.__str__()
            == param.__repr__()
            == "ChiSquare(df=Deterministic(int 1))"
        )

    def test_draw_sample(self):
        param = iap.ChiSquare(1)

        sample = param.draw_sample()

        assert sample.shape == tuple()
        assert 0 <= sample

    def test_via_comparison_to_np_chisquare(self):
        param = iap.ChiSquare(1)

        samples = param.draw_samples((100, 1000))
        samples_direct = np.random.RandomState(1234).chisquare(df=1,
                                                               size=(100, 1000))

        assert samples.shape == (100, 1000)
        assert np.all(0 <= samples)

        samples = np.clip(samples, 0, 3)
        samples_direct = np.clip(samples_direct, 0, 3)
        nb_bins = 10
        hist, _ = np.histogram(samples, bins=nb_bins, range=(0, 3.0),
                               density=False)
        hist_direct, _ = np.histogram(samples_direct, bins=nb_bins,
                                      range=(0, 3.0), density=False)
        tolerance = 0.05
        for nb_samples, nb_samples_direct in zip(hist, hist_direct):
            density = nb_samples / samples.size
            density_direct = nb_samples_direct / samples_direct.size
            assert (
                density_direct - tolerance
                < density <
                density_direct + tolerance
            )

    def test_df_is_stochastic_parameter(self):
        param = iap.ChiSquare(iap.Choice([1, 10]))

        seen = [0, 0]
        for _ in sm.xrange(1000):
            samples = param.draw_samples((100,))
            exp = np.mean(samples)

            if 1 - 1.0 < exp < 1 + 1.0:
                seen[0] += 1
            elif 10 - 4.0 < exp < 10 + 4.0:
                seen[1] += 1
            else:
                assert False

        assert 500 - 100 < seen[0] < 500 + 100
        assert 500 - 100 < seen[1] < 500 + 100

    def test_larger_df_leads_to_more_variance(self):
        param1 = iap.ChiSquare(1)
        param2 = iap.ChiSquare(10)

        samples1 = param1.draw_samples((1000,))
        samples2 = param2.draw_samples((1000,))

        assert np.var(samples1) < np.var(samples2)
        assert 2*1 - 1.0 < np.var(samples1) < 2*1 + 1.0
        assert 2*10 - 5.0 < np.var(samples2) < 2*10 + 5.0

    def test_samples_same_values_for_same_seeds(self):
        param = iap.ChiSquare(1)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.allclose(samples1, samples2)


class TestWeibull(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Weibull(1)

        assert (
            param.__str__()
            == param.__repr__()
            == "Weibull(a=Deterministic(int 1))"
        )

    def test_draw_sample(self):
        param = iap.Weibull(1)

        sample = param.draw_sample()

        assert sample.shape == tuple()
        assert 0 <= sample

    def test_via_comparison_to_np_weibull(self):
        param = iap.Weibull(1)

        samples = param.draw_samples((100, 1000))
        samples_direct = np.random.RandomState(1234).weibull(a=1,
                                                             size=(100, 1000))

        assert samples.shape == (100, 1000)
        assert np.all(0 <= samples)

        samples = np.clip(samples, 0, 2)
        samples_direct = np.clip(samples_direct, 0, 2)
        nb_bins = 10
        hist, _ = np.histogram(samples, bins=nb_bins, range=(0, 2.0),
                               density=False)
        hist_direct, _ = np.histogram(samples_direct, bins=nb_bins,
                                      range=(0, 2.0), density=False)
        tolerance = 0.05
        for nb_samples, nb_samples_direct in zip(hist, hist_direct):
            density = nb_samples / samples.size
            density_direct = nb_samples_direct / samples_direct.size
            assert (
                density_direct - tolerance
                < density <
                density_direct + tolerance
            )

    def test_argument_is_stochastic_parameter(self):
        param = iap.Weibull(iap.Choice([1, 0.5]))

        expected_first = scipy.special.gamma(1 + 1/1)
        expected_second = scipy.special.gamma(1 + 1/0.5)
        seen = [0, 0]
        for _ in sm.xrange(100):
            samples = param.draw_samples((50000,))
            observed = np.mean(samples)

            matches_first = (
                expected_first - 0.2 * expected_first
                < observed <
                expected_first + 0.2 * expected_first
            )
            matches_second = (
                expected_second - 0.2 * expected_second
                < observed <
                expected_second + 0.2 * expected_second
            )

            if matches_first:
                seen[0] += 1
            elif matches_second:
                seen[1] += 1
            else:
                assert False

        assert 50 - 25 < seen[0] < 50 + 25
        assert 50 - 25 < seen[1] < 50 + 25

    def test_different_strengths(self):
        param1 = iap.Weibull(1)
        param2 = iap.Weibull(0.5)

        samples1 = param1.draw_samples((10000,))
        samples2 = param2.draw_samples((10000,))
        expected_first = (
            scipy.special.gamma(1 + 2/1)
            - (scipy.special.gamma(1 + 1/1))**2
        )
        expected_second = (
            scipy.special.gamma(1 + 2/0.5)
            - (scipy.special.gamma(1 + 1/0.5))**2
        )

        assert np.var(samples1) < np.var(samples2)
        assert (
            expected_first - 0.2 * expected_first
            < np.var(samples1) <
            expected_first + 0.2 * expected_first
        )
        assert (
            expected_second - 0.2 * expected_second
            < np.var(samples2) <
            expected_second + 0.2 * expected_second
        )

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Weibull(1)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.allclose(samples1, samples2)


class TestUniform(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Uniform(0, 1.0)
        assert (
            param.__str__()
            == param.__repr__()
            == "Uniform(Deterministic(int 0), Deterministic(float 1.00000000))"
        )

    def test_draw_sample(self):
        param = iap.Uniform(0, 1.0)

        sample = param.draw_sample()

        assert sample.shape == tuple()
        assert 0 - _eps(sample) < sample < 1.0 + _eps(sample)

    def test_draw_samples(self):
        param = iap.Uniform(0, 1.0)

        samples = param.draw_samples((10, 5))

        assert samples.shape == (10, 5)
        assert np.all(
            np.logical_and(
                0 - _eps(samples) < samples,
                samples < 1.0 + _eps(samples)
            )
        )

    def test_via_density_histogram(self):
        param = iap.Uniform(0, 1.0)

        samples = param.draw_samples((10000,))
        nb_bins = 10
        hist, _ = np.histogram(samples, bins=nb_bins, range=(0.0, 1.0),
                               density=False)
        density_expected = 1.0/nb_bins
        density_tolerance = 0.05
        for nb_samples in hist:
            density = nb_samples / samples.size
            assert (
                density_expected - density_tolerance
                < density <
                density_expected + density_tolerance
            )

    def test_negative_value(self):
        param = iap.Uniform(-1.0, 1.0)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert -1.0 - _eps(sample) < sample < 1.0 + _eps(sample)
        assert np.all(
            np.logical_and(
                -1.0 - _eps(samples) < samples,
                samples < 1.0 + _eps(samples)
            )
        )

    def test_wrong_argument_order(self):
        param = iap.Uniform(1.0, -1.0)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert -1.0 - _eps(sample) < sample < 1.0 + _eps(sample)
        assert np.all(
            np.logical_and(
                -1.0 - _eps(samples) < samples,
                samples < 1.0 + _eps(samples)
            )
        )

    def test_arguments_are_integers(self):
        param = iap.Uniform(-1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert -1.0 - _eps(sample) < sample < 1.0 + _eps(sample)
        assert np.all(
            np.logical_and(
                -1.0 - _eps(samples) < samples,
                samples < 1.0 + _eps(samples)
            )
        )

    def test_arguments_are_identical(self):
        param = iap.Uniform(1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert 1.0 - _eps(sample) < sample < 1.0 + _eps(sample)
        assert np.all(
            np.logical_and(
                1.0 - _eps(samples) < samples,
                samples < 1.0 + _eps(samples)
            )
        )

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Uniform(-1.0, 1.0)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.allclose(samples1, samples2)


class TestBeta(unittest.TestCase):
    @classmethod
    def _mean(cls, alpha, beta):
        return alpha / (alpha + beta)

    @classmethod
    def _var(cls, alpha, beta):
        return (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Beta(0.5, 0.5)
        assert (
            param.__str__()
            == param.__repr__()
            == "Beta("
               "Deterministic(float 0.50000000), "
               "Deterministic(float 0.50000000)"
               ")"
        )

    def test_draw_sample(self):
        param = iap.Beta(0.5, 0.5)

        sample = param.draw_sample()

        assert sample.shape == tuple()
        assert 0 - _eps(sample) < sample < 1.0 + _eps(sample)

    def test_draw_samples(self):
        param = iap.Beta(0.5, 0.5)

        samples = param.draw_samples((100, 1000))

        assert samples.shape == (100, 1000)
        assert np.all(
            np.logical_and(
                0 - _eps(samples) <= samples,
                samples <= 1.0 + _eps(samples)
            )
        )

    def test_via_comparison_to_np_beta(self):
        param = iap.Beta(0.5, 0.5)

        samples = param.draw_samples((100, 1000))
        samples_direct = np.random.RandomState(1234).beta(
            a=0.5, b=0.5, size=(100, 1000))

        nb_bins = 10
        hist, _ = np.histogram(samples, bins=nb_bins, range=(0, 1.0),
                               density=False)
        hist_direct, _ = np.histogram(samples_direct, bins=nb_bins,
                                      range=(0, 1.0), density=False)
        tolerance = 0.05
        for nb_samples, nb_samples_direct in zip(hist, hist_direct):
            density = nb_samples / samples.size
            density_direct = nb_samples_direct / samples_direct.size
            assert (
                density_direct - tolerance
                < density <
                density_direct + tolerance
            )

    def test_argument_is_stochastic_parameter(self):
        param = iap.Beta(iap.Choice([0.5, 2]), 0.5)

        expected_first = self._mean(0.5, 0.5)
        expected_second = self._mean(2, 0.5)
        seen = [0, 0]
        for _ in sm.xrange(100):
            samples = param.draw_samples((10000,))
            observed = np.mean(samples)

            if expected_first - 0.05 < observed < expected_first + 0.05:
                seen[0] += 1
            elif expected_second - 0.05 < observed < expected_second + 0.05:
                seen[1] += 1
            else:
                assert False

        assert 50 - 25 < seen[0] < 50 + 25
        assert 50 - 25 < seen[1] < 50 + 25

    def test_compare_curves_of_different_arguments(self):
        param1 = iap.Beta(2, 2)
        param2 = iap.Beta(0.5, 0.5)

        samples1 = param1.draw_samples((10000,))
        samples2 = param2.draw_samples((10000,))

        expected_first = self._var(2, 2)
        expected_second = self._var(0.5, 0.5)

        assert np.var(samples1) < np.var(samples2)
        assert (
            expected_first - 0.1 * expected_first
            < np.var(samples1) <
            expected_first + 0.1 * expected_first
        )
        assert (
            expected_second - 0.1 * expected_second
            < np.var(samples2) <
            expected_second + 0.1 * expected_second
        )

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Beta(0.5, 0.5)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.allclose(samples1, samples2)


class TestDeterministic(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        pairs = [
            (0, "Deterministic(int 0)"),
            (1.0, "Deterministic(float 1.00000000)"),
            ("test", "Deterministic(test)")
        ]
        for value, expected in pairs:
            with self.subTest(value=value):
                param = iap.Deterministic(value)
                assert (
                    param.__str__()
                    == param.__repr__()
                    == expected
                )

    def test_samples_same_values_for_same_seeds(self):
        values = [
            -100, -54, -1, 0, 1, 54, 100,
            -100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0
        ]
        for value in values:
            with self.subTest(value=value):
                param = iap.Deterministic(value)

                rs1 = np.random.RandomState(123456)
                rs2 = np.random.RandomState(123456)

                samples1 = param.draw_samples(20, random_state=rs1)
                samples2 = param.draw_samples(20, random_state=rs2)

                assert np.array_equal(samples1, samples2)

    def test_draw_sample_int(self):
        values = [-100, -54, -1, 0, 1, 54, 100]
        for value in values:
            with self.subTest(value=value):
                param = iap.Deterministic(value)

                sample1 = param.draw_sample()
                sample2 = param.draw_sample()

                assert sample1.shape == tuple()
                assert sample1 == sample2

    def test_draw_sample_float(self):
        values = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]
        for value in values:
            with self.subTest(value=value):
                param = iap.Deterministic(value)

                sample1 = param.draw_sample()
                sample2 = param.draw_sample()

                assert sample1.shape == tuple()
                assert np.isclose(
                    sample1, sample2, rtol=0, atol=_eps(sample1))

    def test_draw_samples_int(self):
        values = [-100, -54, -1, 0, 1, 54, 100]
        shapes = [10, 10, (5, 3), (5, 3), (4, 5, 3), (4, 5, 3)]
        for value, shape in itertools.product(values, shapes):
            with self.subTest(value=value, shape=shape):
                param = iap.Deterministic(value)

                samples = param.draw_samples(shape)
                shape_expected = (
                    shape
                    if isinstance(shape, tuple)
                    else tuple([shape]))

                assert samples.shape == shape_expected
                assert np.all(samples == value)

    def test_draw_samples_float(self):
        values = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]
        shapes = [10, 10, (5, 3), (5, 3), (4, 5, 3), (4, 5, 3)]
        for value, shape in itertools.product(values, shapes):
            with self.subTest(value=value, shape=shape):
                param = iap.Deterministic(value)

                samples = param.draw_samples(shape)
                shape_expected = (
                    shape
                    if isinstance(shape, tuple)
                    else tuple([shape]))

                assert samples.shape == shape_expected
                assert np.allclose(samples, value, rtol=0, atol=_eps(samples))

    def test_argument_is_stochastic_parameter(self):
        seen = [0, 0]
        for _ in sm.xrange(200):
            param = iap.Deterministic(iap.Choice([0, 1]))
            seen[param.value] += 1

        assert 100 - 50 < seen[0] < 100 + 50
        assert 100 - 50 < seen[1] < 100 + 50

    def test_argument_has_invalid_type(self):
        with self.assertRaises(Exception) as context:
            _ = iap.Deterministic([1, 2, 3])

        self.assertTrue(
            "Expected StochasticParameter object or number or string"
            in str(context.exception))


class TestFromLowerResolution(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init___size_percent(self):
        param = iap.FromLowerResolution(other_param=iap.Deterministic(0),
                                        size_percent=1, method="nearest")

        assert (
            param.__str__()
            == param.__repr__()
            == "FromLowerResolution("
               "size_percent=Deterministic(int 1), "
               "method=Deterministic(nearest), "
               "other_param=Deterministic(int 0)"
               ")"
        )

    def test___init___size_px(self):
        param = iap.FromLowerResolution(other_param=iap.Deterministic(0),
                                        size_px=1, method="nearest")

        assert (
            param.__str__()
            == param.__repr__()
            == "FromLowerResolution("
               "size_px=Deterministic(int 1), "
               "method=Deterministic(nearest), "
               "other_param=Deterministic(int 0)"
               ")"
        )

    def test_binomial_hwc(self):
        param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=8)

        samples = param.draw_samples((8, 8, 1))
        uq = np.unique(samples)

        assert samples.shape == (8, 8, 1)
        assert len(uq) == 2
        assert 0 in uq
        assert 1 in uq

    def test_binomial_nhwc(self):
        param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=8)

        samples_nhwc = param.draw_samples((1, 8, 8, 1))
        uq = np.unique(samples_nhwc)

        assert samples_nhwc.shape == (1, 8, 8, 1)
        assert len(uq) == 2
        assert 0 in uq
        assert 1 in uq

    def test_draw_samples_with_too_many_dimensions(self):
        # (N, H, W, C, something) causing error
        param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=8)

        with self.assertRaises(Exception) as context:
            _ = param.draw_samples((1, 8, 8, 1, 1))

        self.assertTrue(
            "FromLowerResolution can only generate samples of shape"
            in str(context.exception)
        )

    def test_binomial_hw3(self):
        # C=3
        param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=8)

        samples = param.draw_samples((8, 8, 3))
        uq = np.unique(samples)

        assert samples.shape == (8, 8, 3)
        assert len(uq) == 2
        assert 0 in uq
        assert 1 in uq

    def test_different_size_px_arguments(self):
        # different sizes in px
        param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
        param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=16)

        seen_components = [0, 0]
        seen_pixels = [0, 0]
        for _ in sm.xrange(100):
            samples1 = param1.draw_samples((16, 16, 1))
            samples2 = param2.draw_samples((16, 16, 1))
            _, num1 = skimage.morphology.label(samples1, neighbors=4,
                                               background=0, return_num=True)
            _, num2 = skimage.morphology.label(samples2, neighbors=4,
                                               background=0, return_num=True)
            seen_components[0] += num1
            seen_components[1] += num2
            seen_pixels[0] += np.sum(samples1 == 1)
            seen_pixels[1] += np.sum(samples2 == 1)

        assert seen_components[0] < seen_components[1]
        assert (
            seen_pixels[0] / seen_components[0]
            > seen_pixels[1] / seen_components[1]
        )

    def test_different_size_px_arguments_with_tuple(self):
        # different sizes in px, one given as tuple (a, b)
        param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
        param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=(2, 16))

        seen_components = [0, 0]
        seen_pixels = [0, 0]
        for _ in sm.xrange(400):
            samples1 = param1.draw_samples((16, 16, 1))
            samples2 = param2.draw_samples((16, 16, 1))
            _, num1 = skimage.morphology.label(samples1, neighbors=4,
                                               background=0, return_num=True)
            _, num2 = skimage.morphology.label(samples2, neighbors=4,
                                               background=0, return_num=True)
            seen_components[0] += num1
            seen_components[1] += num2
            seen_pixels[0] += np.sum(samples1 == 1)
            seen_pixels[1] += np.sum(samples2 == 1)

        assert seen_components[0] < seen_components[1]
        assert (
            seen_pixels[0] / seen_components[0]
            > seen_pixels[1] / seen_components[1]
        )

    def test_different_size_px_argument_with_stochastic_parameters(self):
        # different sizes in px, given as StochasticParameter
        param1 = iap.FromLowerResolution(iap.Binomial(0.5),
                                         size_px=iap.Deterministic(1))
        param2 = iap.FromLowerResolution(iap.Binomial(0.5),
                                         size_px=iap.Choice([8, 16]))

        seen_components = [0, 0]
        seen_pixels = [0, 0]
        for _ in sm.xrange(100):
            samples1 = param1.draw_samples((16, 16, 1))
            samples2 = param2.draw_samples((16, 16, 1))
            _, num1 = skimage.morphology.label(samples1, neighbors=4,
                                               background=0, return_num=True)
            _, num2 = skimage.morphology.label(samples2, neighbors=4,
                                               background=0, return_num=True)
            seen_components[0] += num1
            seen_components[1] += num2
            seen_pixels[0] += np.sum(samples1 == 1)
            seen_pixels[1] += np.sum(samples2 == 1)

        assert seen_components[0] < seen_components[1]
        assert (
            seen_pixels[0] / seen_components[0]
            > seen_pixels[1] / seen_components[1]
        )

    def test_size_px_has_invalid_datatype(self):
        # bad datatype for size_px
        with self.assertRaises(Exception) as context:
            _ = iap.FromLowerResolution(iap.Binomial(0.5), size_px=False)

        self.assertTrue("Expected " in str(context.exception))

    def test_min_size(self):
        # min_size
        param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
        param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=1,
                                         min_size=16)

        seen_components = [0, 0]
        seen_pixels = [0, 0]
        for _ in sm.xrange(100):
            samples1 = param1.draw_samples((16, 16, 1))
            samples2 = param2.draw_samples((16, 16, 1))
            _, num1 = skimage.morphology.label(samples1, neighbors=4,
                                               background=0, return_num=True)
            _, num2 = skimage.morphology.label(samples2, neighbors=4,
                                               background=0, return_num=True)
            seen_components[0] += num1
            seen_components[1] += num2
            seen_pixels[0] += np.sum(samples1 == 1)
            seen_pixels[1] += np.sum(samples2 == 1)

        assert seen_components[0] < seen_components[1]
        assert (
            seen_pixels[0] / seen_components[0]
            > seen_pixels[1] / seen_components[1]
        )

    def test_size_percent(self):
        # different sizes in percent
        param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=0.01)
        param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=0.8)

        seen_components = [0, 0]
        seen_pixels = [0, 0]
        for _ in sm.xrange(100):
            samples1 = param1.draw_samples((16, 16, 1))
            samples2 = param2.draw_samples((16, 16, 1))
            _, num1 = skimage.morphology.label(samples1, neighbors=4,
                                               background=0, return_num=True)
            _, num2 = skimage.morphology.label(samples2, neighbors=4,
                                               background=0, return_num=True)
            seen_components[0] += num1
            seen_components[1] += num2
            seen_pixels[0] += np.sum(samples1 == 1)
            seen_pixels[1] += np.sum(samples2 == 1)

        assert seen_components[0] < seen_components[1]
        assert (
            seen_pixels[0] / seen_components[0]
            > seen_pixels[1] / seen_components[1]
        )

    def test_size_percent_as_stochastic_parameters(self):
        # different sizes in percent, given as StochasticParameter
        param1 = iap.FromLowerResolution(iap.Binomial(0.5),
                                         size_percent=iap.Deterministic(0.01))
        param2 = iap.FromLowerResolution(iap.Binomial(0.5),
                                         size_percent=iap.Choice([0.4, 0.8]))

        seen_components = [0, 0]
        seen_pixels = [0, 0]
        for _ in sm.xrange(100):
            samples1 = param1.draw_samples((16, 16, 1))
            samples2 = param2.draw_samples((16, 16, 1))
            _, num1 = skimage.morphology.label(samples1, neighbors=4,
                                               background=0, return_num=True)
            _, num2 = skimage.morphology.label(samples2, neighbors=4,
                                               background=0, return_num=True)
            seen_components[0] += num1
            seen_components[1] += num2
            seen_pixels[0] += np.sum(samples1 == 1)
            seen_pixels[1] += np.sum(samples2 == 1)

        assert seen_components[0] < seen_components[1]
        assert (
            seen_pixels[0] / seen_components[0]
            > seen_pixels[1] / seen_components[1]
        )

    def test_size_percent_has_invalid_datatype(self):
        # bad datatype for size_percent
        with self.assertRaises(Exception) as context:
            _ = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=False)

        self.assertTrue("Expected " in str(context.exception))

    def test_method(self):
        # method given as StochasticParameter
        param = iap.FromLowerResolution(
            iap.Binomial(0.5), size_px=4,
            method=iap.Choice(["nearest", "linear"]))

        seen = [0, 0]
        for _ in sm.xrange(200):
            samples = param.draw_samples((16, 16, 1))
            nb_in_between = np.sum(
                np.logical_and(0.05 < samples, samples < 0.95))
            if nb_in_between == 0:
                seen[0] += 1
            else:
                seen[1] += 1

        assert 100 - 50 < seen[0] < 100 + 50
        assert 100 - 50 < seen[1] < 100 + 50

    def test_method_has_invalid_datatype(self):
        # bad datatype for method
        with self.assertRaises(Exception) as context:
            _ = iap.FromLowerResolution(iap.Binomial(0.5), size_px=4,
                                        method=False)

        self.assertTrue("Expected " in str(context.exception))

    def test_samples_same_values_for_same_seeds(self):
        # multiple calls with same random_state
        param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)

        samples1 = param.draw_samples((10, 5, 1),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5, 1),
                                      random_state=np.random.RandomState(1234))

        assert np.allclose(samples1, samples2)


class TestClip(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Clip(iap.Deterministic(0), -1, 1)
        assert (
            param.__str__()
            == param.__repr__()
            == "Clip(Deterministic(int 0), -1.000000, 1.000000)"
        )

    def test_value_within_bounds(self):
        param = iap.Clip(iap.Deterministic(0), -1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample == 0
        assert np.all(samples == 0)

    def test_value_exactly_at_upper_bound(self):
        param = iap.Clip(iap.Deterministic(1), -1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample == 1
        assert np.all(samples == 1)

    def test_value_exactly_at_lower_bound(self):
        param = iap.Clip(iap.Deterministic(-1), -1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample == -1
        assert np.all(samples == -1)

    def test_value_is_within_bounds_and_float(self):
        param = iap.Clip(iap.Deterministic(0.5), -1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert 0.5 - _eps(sample) < sample < 0.5 + _eps(sample)
        assert np.all(
            np.logical_and(
                0.5 - _eps(sample) <= samples,
                samples <= 0.5 + _eps(sample)
            )
        )

    def test_value_is_above_upper_bound(self):
        param = iap.Clip(iap.Deterministic(2), -1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample == 1
        assert np.all(samples == 1)

    def test_value_is_below_lower_bound(self):
        param = iap.Clip(iap.Deterministic(-2), -1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample == -1
        assert np.all(samples == -1)

    def test_value_is_sometimes_without_bounds_sometimes_beyond(self):
        param = iap.Clip(iap.Choice([0, 2]), -1, 1)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample in [0, 1]
        assert np.all(np.logical_or(samples == 0, samples == 1))

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Clip(iap.Choice([0, 2]), -1, 1)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.array_equal(samples1, samples2)

    def test_lower_bound_is_none(self):
        param = iap.Clip(iap.Deterministic(0), None, 1)

        sample = param.draw_sample()

        assert sample == 0
        assert (
            param.__str__()
            == param.__repr__()
            == "Clip(Deterministic(int 0), None, 1.000000)"
        )

    def test_upper_bound_is_none(self):
        param = iap.Clip(iap.Deterministic(0), 0, None)

        sample = param.draw_sample()

        assert sample == 0
        assert (
            param.__str__()
            == param.__repr__()
            == "Clip(Deterministic(int 0), 0.000000, None)"
        )

    def test_both_bounds_are_none(self):
        param = iap.Clip(iap.Deterministic(0), None, None)

        sample = param.draw_sample()

        assert sample == 0
        assert (
            param.__str__()
            == param.__repr__()
            == "Clip(Deterministic(int 0), None, None)"
        )


class TestDiscretize(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Discretize(iap.Deterministic(0))

        assert (
            param.__str__()
            == param.__repr__()
            == "Discretize(Deterministic(int 0))"
        )

    def test_applied_to_deterministic(self):
        values = [-100.2, -54.3, -1.0, -1, -0.7, -0.00043,
                  0,
                  0.00043, 0.7, 1.0, 1, 54.3, 100.2]
        for value in values:
            with self.subTest(value=value):
                param = iap.Discretize(iap.Deterministic(value))
                value_expected = np.round(
                    np.float64([value])
                ).astype(np.int32)[0]

                sample = param.draw_sample()
                samples = param.draw_samples((10, 5))

                assert sample.shape == tuple()
                assert samples.shape == (10, 5)
                assert sample == value_expected
                assert np.all(samples == value_expected)

    # TODO why are these tests applied to DiscreteUniform instead of Uniform?
    def test_applied_to_discrete_uniform(self):
        param_orig = iap.DiscreteUniform(0, 1)
        param = iap.Discretize(param_orig)

        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))

        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample in [0, 1]
        assert np.all(np.logical_or(samples == 0, samples == 1))

    def test_applied_to_discrete_uniform_with_wider_range(self):
        param_orig = iap.DiscreteUniform(0, 2)
        param = iap.Discretize(param_orig)

        samples1 = param_orig.draw_samples((10000,))
        samples2 = param.draw_samples((10000,))

        assert np.all(np.abs(samples1 - samples2) < 0.2*(10000/3))

    def test_samples_same_values_for_same_seeds(self):
        param_orig = iap.DiscreteUniform(0, 2)
        param = iap.Discretize(param_orig)

        samples1 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((10, 5),
                                      random_state=np.random.RandomState(1234))

        assert np.array_equal(samples1, samples2)


class TestMultiply(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Multiply(iap.Deterministic(0), 1, elementwise=False)
        assert (
            param.__str__()
            == param.__repr__()
            == "Multiply(Deterministic(int 0), Deterministic(int 1), False)"
        )

    def test_multiply_example_integer_values(self):
        values_int = [-100, -54, -1, 0, 1, 54, 100]

        for v1, v2 in itertools.product(values_int, values_int):
            with self.subTest(left=v1, right=v2):
                p = iap.Multiply(iap.Deterministic(v1), v2)

                samples = p.draw_samples((2, 3))

                assert p.draw_sample() == v1 * v2
                assert samples.dtype.kind == "i"
                assert np.array_equal(
                    samples,
                    np.zeros((2, 3), dtype=np.int64) + v1 * v2
                )

    def test_multiply_example_integer_values_both_deterministic(self):
        values_int = [-100, -54, -1, 0, 1, 54, 100]

        for v1, v2 in itertools.product(values_int, values_int):
            with self.subTest(left=v1, right=v2):
                p = iap.Multiply(iap.Deterministic(v1), iap.Deterministic(v2))

                samples = p.draw_samples((2, 3))

                assert p.draw_sample() == v1 * v2
                assert samples.dtype.name == "int32"
                assert np.array_equal(
                    samples,
                    np.zeros((2, 3), dtype=np.int32) + v1 * v2
                )

    def test_multiply_example_float_values(self):
        values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]
        for v1, v2 in itertools.product(values_float, values_float):
            with self.subTest(left=v1, right=v2):
                p = iap.Multiply(iap.Deterministic(v1), v2)

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert np.isclose(sample, v1 * v2, atol=1e-3, rtol=0)
                assert samples.dtype.kind == "f"
                assert np.allclose(
                    samples,
                    np.zeros((2, 3), dtype=np.float32) + v1 * v2
                )

    def test_multiply_example_float_values_both_deterministic(self):
        values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]
        for v1, v2 in itertools.product(values_float, values_float):
            with self.subTest(left=v1, right=v2):
                p = iap.Multiply(iap.Deterministic(v1), iap.Deterministic(v2))

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert np.isclose(sample, v1 * v2, atol=1e-3, rtol=0)
                assert samples.dtype.kind == "f"
                assert np.allclose(
                    samples,
                    np.zeros((2, 3), dtype=np.float32) + v1 * v2
                )

    def test_multiply_by_stochastic_parameter(self):
        param = iap.Multiply(iap.Deterministic(1.0),
                             (1.0, 2.0),
                             elementwise=False)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.0 * 1.0 - _eps(samples))
        assert np.all(samples < 1.0 * 2.0 + _eps(samples))
        assert (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_multiply_by_stochastic_parameter_elementwise(self):
        param = iap.Multiply(iap.Deterministic(1.0),
                             (1.0, 2.0),
                             elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.0 * 1.0 - _eps(samples))
        assert np.all(samples < 1.0 * 2.0 + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_multiply_stochastic_parameter_by_fixed_value(self):
        param = iap.Multiply(iap.Uniform(1.0, 2.0),
                             1.0,
                             elementwise=False)
        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.0 * 1.0 - _eps(samples))
        assert np.all(samples < 2.0 * 1.0 + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_multiply_stochastic_parameter_by_fixed_value_elementwise(self):
        param = iap.Multiply(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.0 * 1.0 - _eps(samples))
        assert np.all(samples < 2.0 * 1.0 + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )


class TestDivide(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Divide(iap.Deterministic(0), 1, elementwise=False)
        assert (
            param.__str__()
            == param.__repr__()
            == "Divide(Deterministic(int 0), Deterministic(int 1), False)"
        )

    def test_divide_integers(self):
        values_int = [-100, -54, -1, 0, 1, 54, 100]

        for v1, v2 in itertools.product(values_int, values_int):
            if v2 == 0:
                v2 = 1

            with self.subTest(left=v1, right=v2):
                p = iap.Divide(iap.Deterministic(v1), v2)

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert sample == (v1 / v2)
                assert samples.dtype.kind == "f"
                assert np.array_equal(
                    samples,
                    np.zeros((2, 3), dtype=np.float64) + (v1 / v2)
                )

    def test_divide_integers_both_deterministic(self):
        values_int = [-100, -54, -1, 0, 1, 54, 100]

        for v1, v2 in itertools.product(values_int, values_int):
            if v2 == 0:
                v2 = 1

            with self.subTest(left=v1, right=v2):
                p = iap.Divide(iap.Deterministic(v1), iap.Deterministic(v2))

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert sample == (v1 / v2)
                assert samples.dtype.kind == "f"
                assert np.array_equal(
                    samples,
                    np.zeros((2, 3), dtype=np.float64) + (v1 / v2)
                )

    def test_divide_floats(self):
        values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

        for v1, v2 in itertools.product(values_float, values_float):
            if v2 == 0:
                v2 = 1

            with self.subTest(left=v1, right=v2):
                p = iap.Divide(iap.Deterministic(v1), v2)

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert (
                    (v1 / v2) - _eps(sample)
                    <= sample <=
                    (v1 / v2) + _eps(sample)
                )
                assert samples.dtype.kind == "f"
                assert np.allclose(
                    samples,
                    np.zeros((2, 3), dtype=np.float64) + (v1 / v2)
                )

    def test_divide_floats_both_deterministic(self):
        values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

        for v1, v2 in itertools.product(values_float, values_float):
            if v2 == 0:
                v2 = 1

            with self.subTest(left=v1, right=v2):
                p = iap.Divide(iap.Deterministic(v1), iap.Deterministic(v2))

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert (
                    (v1 / v2) - _eps(sample)
                    <= sample <=
                    (v1 / v2) + _eps(sample)
                )
                assert samples.dtype.kind == "f"
                assert np.allclose(
                    samples,
                    np.zeros((2, 3), dtype=np.float64) + (v1 / v2)
                )

    def test_divide_by_stochastic_parameter(self):
        param = iap.Divide(iap.Deterministic(1.0),
                           (1.0, 2.0),
                           elementwise=False)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > (1.0 / 2.0) - _eps(samples))
        assert np.all(samples < (1.0 / 1.0) + _eps(samples))
        assert (
            samples_sorted[0] - _eps(samples)
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples)
        )

    def test_divide_by_stochastic_parameter_elementwise(self):
        param = iap.Divide(iap.Deterministic(1.0),
                           (1.0, 2.0),
                           elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > (1.0 / 2.0) - _eps(samples))
        assert np.all(samples < (1.0 / 1.0) + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples)
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples)
        )

    def test_divide_stochastic_parameter_by_float(self):
        param = iap.Divide(iap.Uniform(1.0, 2.0),
                           1.0,
                           elementwise=False)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > (1.0 / 1.0) - _eps(samples))
        assert np.all(samples < (2.0 / 1.0) + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples)
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples)
        )

    def test_divide_stochastic_parameter_by_float_elementwise(self):
        param = iap.Divide(iap.Uniform(1.0, 2.0),
                           1.0,
                           elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > (1.0 / 1.0) - _eps(samples))
        assert np.all(samples < (2.0 / 1.0) + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted)
            < samples_sorted[-1]
            < samples_sorted[-1]
            < samples_sorted[0] + _eps(samples_sorted)
        )

    def test_divide_by_stochastic_parameter_that_can_by_zero(self):
        # test division by zero automatically being converted to division by 1
        param = iap.Divide(2,
                           iap.Choice([0, 2]),
                           elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_unique = np.sort(np.unique(samples.flatten()))

        assert samples_unique[0] == 1 and samples_unique[1] == 2

    def test_divide_by_zero(self):
        param = iap.Divide(iap.Deterministic(1), 0, elementwise=False)

        sample = param.draw_sample()

        assert sample == 1


class TestAdd(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Add(iap.Deterministic(0), 1, elementwise=False)
        assert (
            param.__str__()
            == param.__repr__()
            == "Add(Deterministic(int 0), Deterministic(int 1), False)"
        )

    def test_add_integers(self):
        values_int = [-100, -54, -1, 0, 1, 54, 100]

        for v1, v2 in itertools.product(values_int, values_int):
            with self.subTest(left=v1, right=v2):
                p = iap.Add(iap.Deterministic(v1), v2)

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert sample == v1 + v2
                assert samples.dtype.kind == "i"
                assert np.array_equal(
                    samples,
                    np.zeros((2, 3), dtype=np.int32) + v1 + v2
                )

    def test_add_integers_both_deterministic(self):
        values_int = [-100, -54, -1, 0, 1, 54, 100]

        for v1, v2 in itertools.product(values_int, values_int):
            with self.subTest(left=v1, right=v2):
                p = iap.Add(iap.Deterministic(v1), iap.Deterministic(v2))

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert sample == v1 + v2
                assert samples.dtype.kind == "i"
                assert np.array_equal(
                    samples,
                    np.zeros((2, 3), dtype=np.int32) + v1 + v2
                )

    def test_add_floats(self):
        values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]
        for v1, v2 in itertools.product(values_float, values_float):
            with self.subTest(left=v1, right=v2):
                p = iap.Add(iap.Deterministic(v1), v2)

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert np.isclose(sample, v1 + v2, atol=1e-3, rtol=0)
                assert samples.dtype.kind == "f"
                assert np.allclose(
                    samples,
                    np.zeros((2, 3), dtype=np.float32) + v1 + v2
                )

    def test_add_floats_both_deterministic(self):
        values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]
        for v1, v2 in itertools.product(values_float, values_float):
            with self.subTest(left=v1, right=v2):
                p = iap.Add(iap.Deterministic(v1), iap.Deterministic(v2))

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert np.isclose(sample, v1 + v2, atol=1e-3, rtol=0)
                assert samples.dtype.kind == "f"
                assert np.allclose(
                    samples,
                    np.zeros((2, 3), dtype=np.float32) + v1 + v2
                )

    def test_add_stochastic_parameter(self):
        param = iap.Add(iap.Deterministic(1.0), (1.0, 2.0), elementwise=False)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples >= 1.0 + 1.0 - _eps(samples))
        assert np.all(samples <= 1.0 + 2.0 + _eps(samples))
        assert (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1]
            < samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_add_stochastic_parameter_elementwise(self):
        param = iap.Add(iap.Deterministic(1.0), (1.0, 2.0), elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples >= 1.0 + 1.0 - _eps(samples))
        assert np.all(samples <= 1.0 + 2.0 + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1]
            < samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_add_to_stochastic_parameter(self):
        param = iap.Add(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples >= 1.0 + 1.0 - _eps(samples))
        assert np.all(samples <= 2.0 + 1.0 + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1]
            < samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_add_to_stochastic_parameter_elementwise(self):
        param = iap.Add(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples >= 1.0 + 1.0 - _eps(samples))
        assert np.all(samples <= 2.0 + 1.0 + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1]
            < samples_sorted[0] + _eps(samples_sorted[0])
        )


class TestSubtract(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Subtract(iap.Deterministic(0), 1, elementwise=False)
        assert (
            param.__str__()
            == param.__repr__()
            == "Subtract(Deterministic(int 0), Deterministic(int 1), False)"
        )

    def test_subtract_integers(self):
        values_int = [-100, -54, -1, 0, 1, 54, 100]

        for v1, v2 in itertools.product(values_int, values_int):
            with self.subTest(left=v1, right=v2):
                p = iap.Subtract(iap.Deterministic(v1), v2)

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert sample == v1 - v2
                assert samples.dtype.kind == "i"
                assert np.array_equal(
                    samples,
                    np.zeros((2, 3), dtype=np.int64) + v1 - v2
                )

    def test_subtract_integers_both_deterministic(self):
        values_int = [-100, -54, -1, 0, 1, 54, 100]

        for v1, v2 in itertools.product(values_int, values_int):
            with self.subTest(left=v1, right=v2):
                p = iap.Subtract(iap.Deterministic(v1), iap.Deterministic(v2))

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert sample == v1 - v2
                assert samples.dtype.kind == "i"
                assert np.array_equal(
                    samples,
                    np.zeros((2, 3), dtype=np.int64) + v1 - v2
                )

    def test_subtract_floats(self):
        values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]
        for v1, v2 in itertools.product(values_float, values_float):
            with self.subTest(left=v1, right=v2):
                p = iap.Subtract(iap.Deterministic(v1), v2)

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert v1 - v2 - _eps(sample) < sample < v1 - v2 + _eps(sample)
                assert samples.dtype.kind == "f"
                assert np.allclose(
                    samples,
                    np.zeros((2, 3), dtype=np.float64) + v1 - v2
                )

    def test_subtract_floats_both_deterministic(self):
        values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]
        for v1, v2 in itertools.product(values_float, values_float):
            with self.subTest(left=v1, right=v2):
                p = iap.Subtract(iap.Deterministic(v1), iap.Deterministic(v2))

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert v1 - v2 - _eps(sample) < sample < v1 - v2 + _eps(sample)
                assert samples.dtype.kind == "f"
                assert np.allclose(
                    samples,
                    np.zeros((2, 3), dtype=np.float64) + v1 - v2
                )

    def test_subtract_stochastic_parameter(self):
        param = iap.Subtract(iap.Deterministic(1.0),
                             (1.0, 2.0),
                             elementwise=False)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.0 - 2.0 - _eps(samples))
        assert np.all(samples < 1.0 - 1.0 + _eps(samples))
        assert (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_subtract_stochastic_parameter_elementwise(self):
        param = iap.Subtract(iap.Deterministic(1.0),
                             (1.0, 2.0),
                             elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.0 - 2.0 - _eps(samples))
        assert np.all(samples < 1.0 - 1.0 + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_subtract_from_stochastic_parameter(self):
        param = iap.Subtract(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.0 - 1.0 - _eps(samples))
        assert np.all(samples < 2.0 - 1.0 + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_subtract_from_stochastic_parameter_elementwise(self):
        param = iap.Subtract(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.0 - 1.0 - _eps(samples))
        assert np.all(samples < 2.0 - 1.0 + _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )


class TestPower(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Power(iap.Deterministic(0), 1, elementwise=False)
        assert (
            param.__str__()
            == param.__repr__()
            == "Power(Deterministic(int 0), Deterministic(int 1), False)"
        )

    def test_pairs(self):
        values = [
            -100, -54, -1, 0, 1, 54, 100,
            -100.0, -54.0, -1.0, 0.0, 1.0, 54.0, 100.0
        ]
        exponents = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

        for base, exponent in itertools.product(values, exponents):
            if base < 0 and ia.is_single_float(exponent):
                continue
            if base == 0 and exponent < 0:
                continue

            with self.subTest(base=base, exponent=exponent):
                p = iap.Power(iap.Deterministic(base), exponent)

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert (
                    base ** exponent - _eps(sample)
                    < sample <
                    base ** exponent + _eps(sample)
                )
                assert samples.dtype.kind == "f"
                assert np.allclose(
                    samples,
                    np.zeros((2, 3), dtype=np.float64) + base ** exponent
                )

    def test_pairs_both_deterministic(self):
        values = [
            -100, -54, -1, 0, 1, 54, 100,
            -100.0, -54.0, -1.0, 0.0, 1.0, 54.0, 100.0
        ]
        exponents = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

        for base, exponent in itertools.product(values, exponents):
            if base < 0 and ia.is_single_float(exponent):
                continue
            if base == 0 and exponent < 0:
                continue

            with self.subTest(base=base, exponent=exponent):
                p = iap.Power(iap.Deterministic(base), iap.Deterministic(exponent))

                sample = p.draw_sample()
                samples = p.draw_samples((2, 3))

                assert (
                    base ** exponent - _eps(sample)
                    < sample <
                    base ** exponent + _eps(sample)
                )
                assert samples.dtype.kind == "f"
                assert np.allclose(
                    samples,
                    np.zeros((2, 3), dtype=np.float64) + base ** exponent
                )

    def test_exponent_is_stochastic_parameter(self):
        param = iap.Power(iap.Deterministic(1.5),
                          (1.0, 2.0),
                          elementwise=False)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.5 ** 1.0 - 2 * _eps(samples))
        assert np.all(samples < 1.5 ** 2.0 + 2 * _eps(samples))
        assert (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_exponent_is_stochastic_parameter_elementwise(self):
        param = iap.Power(iap.Deterministic(1.5),
                          (1.0, 2.0),
                          elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.5 ** 1.0 - 2 * _eps(samples))
        assert np.all(samples < 1.5 ** 2.0 + 2 * _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_value_is_uniform(self):
        param = iap.Power(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.0 ** 1.0 - 2 * _eps(samples))
        assert np.all(samples < 2.0 ** 1.0 + 2 * _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )

    def test_value_is_uniform_elementwise(self):
        param = iap.Power(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)

        samples = param.draw_samples((10, 20))
        samples_sorted = np.sort(samples.flatten())

        assert samples.shape == (10, 20)
        assert np.all(samples > 1.0 ** 1.0 - 2 * _eps(samples))
        assert np.all(samples < 2.0 ** 1.0 + 2 * _eps(samples))
        assert not (
            samples_sorted[0] - _eps(samples_sorted[0])
            < samples_sorted[-1] <
            samples_sorted[0] + _eps(samples_sorted[0])
        )


class TestAbsolute(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Absolute(iap.Deterministic(0))
        assert (
            param.__str__()
            == param.__repr__()
            == "Absolute(Deterministic(int 0))"
        )

    def test_fixed_values(self):
        simple_values = [-1.5, -1, -1.0, -0.1, 0, 0.0, 0.1, 1, 1.0, 1.5]

        for value in simple_values:
            with self.subTest(value=value):
                param = iap.Absolute(iap.Deterministic(value))

                sample = param.draw_sample()
                samples = param.draw_samples((10, 5))

                assert sample.shape == tuple()
                assert samples.shape == (10, 5)
                if ia.is_single_float(value):
                    assert (
                        abs(value) - _eps(sample)
                        < sample <
                        abs(value) + _eps(sample)
                    )
                    assert np.all(abs(value) - _eps(samples) < samples)
                    assert np.all(samples < abs(value) + _eps(samples))
                else:
                    assert sample == abs(value)
                    assert np.all(samples == abs(value))

    def test_value_is_stochastic_parameter(self):
        param = iap.Absolute(iap.Choice([-3, -1, 1, 3]))

        sample = param.draw_sample()
        samples = param.draw_samples((10, 10))
        samples_uq = np.sort(np.unique(samples))

        assert sample.shape == tuple()
        assert sample in [3, 1]
        assert samples.shape == (10, 10)
        assert len(samples_uq) == 2
        assert samples_uq[0] == 1 and samples_uq[1] == 3


class TestRandomSign(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.RandomSign(iap.Deterministic(0), 0.5)
        assert (
            param.__str__()
            == param.__repr__()
            == "RandomSign(Deterministic(int 0), 0.50)"
        )

    def test_value_is_deterministic(self):
        param = iap.RandomSign(iap.Deterministic(1))

        samples = param.draw_samples((1000,))
        n_positive = np.sum(samples == 1)
        n_negative = np.sum(samples == -1)

        assert samples.shape == (1000,)
        assert n_positive + n_negative == 1000
        assert 350 < n_positive < 750

    def test_value_is_deterministic_many_samples(self):
        param = iap.RandomSign(iap.Deterministic(1))

        seen = [0, 0]
        for _ in sm.xrange(1000):
            sample = param.draw_sample()
            assert sample.shape == tuple()
            if sample == 1:
                seen[1] += 1
            else:
                seen[0] += 1
        n_negative, n_positive = seen

        assert n_positive + n_negative == 1000
        assert 350 < n_positive < 750

    def test_value_is_stochastic_parameter(self):
        param = iap.RandomSign(iap.Choice([1, 2]))

        samples = param.draw_samples((4000,))
        seen = [0, 0, 0, 0]
        seen[0] = np.sum(samples == -2)
        seen[1] = np.sum(samples == -1)
        seen[2] = np.sum(samples == 1)
        seen[3] = np.sum(samples == 2)

        assert np.sum(seen) == 4000
        assert all([700 < v < 1300 for v in seen])

    def test_samples_same_values_for_same_seeds(self):
        param = iap.RandomSign(iap.Choice([1, 2]))

        samples1 = param.draw_samples((100, 10),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((100, 10),
                                      random_state=np.random.RandomState(1234))

        assert samples1.shape == (100, 10)
        assert samples2.shape == (100, 10)
        assert np.array_equal(samples1, samples2)
        assert np.sum(samples1 == -2) > 50
        assert np.sum(samples1 == -1) > 50
        assert np.sum(samples1 == 1) > 50
        assert np.sum(samples1 == 2) > 50


class TestForceSign(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.ForceSign(iap.Deterministic(0), True, "invert", 1)
        assert (
            param.__str__()
            == param.__repr__()
            == "ForceSign(Deterministic(int 0), True, invert, 1)"
        )

    def test_single_sample_positive(self):
        param = iap.ForceSign(iap.Deterministic(1), positive=True,
                              mode="invert")

        sample = param.draw_sample()

        assert sample.shape == tuple()
        assert sample == 1

    def test_single_sample_negative(self):
        param = iap.ForceSign(iap.Deterministic(1), positive=False,
                              mode="invert")

        sample = param.draw_sample()

        assert sample.shape == tuple()
        assert sample == -1

    def test_many_samples_positive(self):
        param = iap.ForceSign(iap.Deterministic(1), positive=True,
                              mode="invert")

        samples = param.draw_samples(100)

        assert samples.shape == (100,)
        assert np.all(samples == 1)

    def test_many_samples_negative(self):
        param = iap.ForceSign(iap.Deterministic(1), positive=False,
                              mode="invert")

        samples = param.draw_samples(100)

        assert samples.shape == (100,)
        assert np.all(samples == -1)

    def test_many_samples_negative_value_to_positive(self):
        param = iap.ForceSign(iap.Deterministic(-1), positive=True,
                              mode="invert")

        samples = param.draw_samples(100)

        assert samples.shape == (100,)
        assert np.all(samples == 1)

    def test_many_samples_negative_value_to_negative(self):
        param = iap.ForceSign(iap.Deterministic(-1), positive=False,
                              mode="invert")

        samples = param.draw_samples(100)

        assert samples.shape == (100,)
        assert np.all(samples == -1)

    def test_many_samples_stochastic_value_to_positive(self):
        param = iap.ForceSign(iap.Choice([-2, 1]), positive=True,
                              mode="invert")

        samples = param.draw_samples(1000)
        n_twos = np.sum(samples == 2)
        n_ones = np.sum(samples == 1)

        assert samples.shape == (1000,)
        assert n_twos + n_ones == 1000
        assert 200 < n_twos < 700
        assert 200 < n_ones < 700

    def test_many_samples_stochastic_value_to_positive_reroll(self):
        param = iap.ForceSign(iap.Choice([-2, 1]), positive=True,
                              mode="reroll")

        samples = param.draw_samples(1000)
        n_twos = np.sum(samples == 2)
        n_ones = np.sum(samples == 1)

        assert samples.shape == (1000,)
        assert n_twos + n_ones == 1000
        assert n_twos > 0
        assert n_ones > 0

    def test_many_samples_stochastic_value_to_positive_reroll_max_count(self):
        param = iap.ForceSign(iap.Choice([-2, 1]), positive=True,
                              mode="reroll", reroll_count_max=100)

        samples = param.draw_samples(100)
        n_twos = np.sum(samples == 2)
        n_ones = np.sum(samples == 1)

        assert samples.shape == (100,)
        assert n_twos + n_ones == 100
        assert n_twos < 5

    def test_samples_same_values_for_same_seeds(self):
        param = iap.ForceSign(iap.Choice([-2, 1]),
                              positive=True,
                              mode="invert")

        samples1 = param.draw_samples((100, 10),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((100, 10),
                                      random_state=np.random.RandomState(1234))

        assert samples1.shape == (100, 10)
        assert samples2.shape == (100, 10)
        assert np.array_equal(samples1, samples2)


class TestPositive(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_many_samples_reroll(self):
        param = iap.Positive(iap.Deterministic(-1),
                             mode="reroll",
                             reroll_count_max=1)

        samples = param.draw_samples((100,))

        assert samples.shape == (100,)
        assert np.all(samples == 1)


class TestNegative(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_many_samples_reroll(self):
        param = iap.Negative(iap.Deterministic(1),
                             mode="reroll",
                             reroll_count_max=1)

        samples = param.draw_samples((100,))

        assert samples.shape == (100,)
        assert np.all(samples == -1)


class TestIterativeNoiseAggregator(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.IterativeNoiseAggregator(iap.Deterministic(0),
                                             iterations=(1, 3),
                                             aggregation_method="max")
        assert (
            param.__str__()
            == param.__repr__()
            == (
                "IterativeNoiseAggregator("
                "Deterministic(int 0), "
                "DiscreteUniform(Deterministic(int 1), "
                "Deterministic(int 3)"
                "), "
                "Deterministic(max)"
                ")"
            )
        )

    def test_value_is_deterministic_max_1_iter(self):
        param = iap.IterativeNoiseAggregator(iap.Deterministic(1),
                                             iterations=1,
                                             aggregation_method="max")

        sample = param.draw_sample()
        samples = param.draw_samples((2, 4))

        assert sample.shape == tuple()
        assert samples.shape == (2, 4)
        assert sample == 1
        assert np.all(samples == 1)

    def test_value_is_stochastic_avg_200_iter(self):
        param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]),
                                             iterations=200,
                                             aggregation_method="avg")

        sample = param.draw_sample()
        samples = param.draw_samples((2, 4))

        assert sample.shape == tuple()
        assert samples.shape == (2, 4)
        assert 25 - 10 < sample < 25 + 10
        assert np.all(np.logical_and(25 - 10 < samples, samples < 25 + 10))

    def test_value_is_stochastic_max_100_iter(self):
        param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]),
                                             iterations=100,
                                             aggregation_method="max")

        sample = param.draw_sample()
        samples = param.draw_samples((2, 4))

        assert sample.shape == tuple()
        assert samples.shape == (2, 4)
        assert sample == 50
        assert np.all(samples == 50)

    def test_value_is_stochastic_min_100_iter(self):
        param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]),
                                             iterations=100,
                                             aggregation_method="min")

        sample = param.draw_sample()
        samples = param.draw_samples((2, 4))

        assert sample.shape == tuple()
        assert samples.shape == (2, 4)
        assert sample == 0
        assert np.all(samples == 0)

    def test_value_is_stochastic_avg_or_max_100_iter_evaluate_counts(self):
        seen = [0, 0, 0, 0]
        for _ in sm.xrange(100):
            param = iap.IterativeNoiseAggregator(
                iap.Choice([0, 50]),
                iterations=100,
                aggregation_method=["avg", "max"])
            samples = param.draw_samples((1, 1))
            diff_0 = abs(0 - samples[0, 0])
            diff_25 = abs(25 - samples[0, 0])
            diff_50 = abs(50 - samples[0, 0])
            if diff_25 < 10.0:
                seen[0] += 1
            elif diff_50 < _eps(samples):
                seen[1] += 1
            elif diff_0 < _eps(samples):
                seen[2] += 1
            else:
                seen[3] += 1

        assert seen[2] <= 2  # around 0.0
        assert seen[3] <= 2  # 0.0+eps <= x < 15.0 or 35.0 < x < 50.0 or >50.0
        assert 50 - 20 < seen[0] < 50 + 20
        assert 50 - 20 < seen[1] < 50 + 20

    def test_value_is_stochastic_avg_tuple_as_iter_evaluate_histograms(self):
        # iterations as tuple
        param = iap.IterativeNoiseAggregator(
            iap.Uniform(-1.0, 1.0),
            iterations=(1, 100),
            aggregation_method="avg")

        diffs = []
        for _ in sm.xrange(100):
            samples = param.draw_samples((1, 1))
            diff = abs(samples[0, 0] - 0.0)
            diffs.append(diff)

        nb_bins = 3
        hist, _ = np.histogram(diffs, bins=nb_bins, range=(-1.0, 1.0),
                               density=False)

        assert hist[1] > hist[0]
        assert hist[1] > hist[2]

    def test_value_is_stochastic_max_list_as_iter_evaluate_counts(self):
        # iterations as list
        seen = [0, 0]
        for _ in sm.xrange(400):
            param = iap.IterativeNoiseAggregator(
                iap.Choice([0, 50]),
                iterations=[1, 100],
                aggregation_method=["max"])
            samples = param.draw_samples((1, 1))
            diff_0 = abs(0 - samples[0, 0])
            diff_50 = abs(50 - samples[0, 0])
            if diff_50 < _eps(samples):
                seen[0] += 1
            elif diff_0 < _eps(samples):
                seen[1] += 1
            else:
                assert False

        assert 300 - 50 < seen[0] < 300 + 50
        assert 100 - 50 < seen[1] < 100 + 50

    def test_value_is_stochastic_all_100_iter(self):
        # test ia.ALL as aggregation_method
        # note that each method individually and list of methods are already
        # tested, so no in depth test is needed here
        param = iap.IterativeNoiseAggregator(
            iap.Choice([0, 50]), iterations=100, aggregation_method=ia.ALL)

        assert isinstance(param.aggregation_method, iap.Choice)
        assert len(param.aggregation_method.a) == 3
        assert [v in param.aggregation_method.a for v in ["min", "avg", "max"]]

    def test_value_is_stochastic_max_2_iter(self):
        param = iap.IterativeNoiseAggregator(
            iap.Choice([0, 50]), iterations=2, aggregation_method="max")

        samples = param.draw_samples((2, 1000))
        nb_0 = np.sum(samples == 0)
        nb_50 = np.sum(samples == 50)

        assert nb_0 + nb_50 == 2 * 1000
        assert 0.25 - 0.05 < nb_0 / (2 * 1000) < 0.25 + 0.05

    def test_samples_same_values_for_same_seeds(self):
        param = iap.IterativeNoiseAggregator(
            iap.Choice([0, 50]), iterations=5, aggregation_method="avg")

        samples1 = param.draw_samples((100, 10),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((100, 10),
                                      random_state=np.random.RandomState(1234))

        assert samples1.shape == (100, 10)
        assert samples2.shape == (100, 10)
        assert np.allclose(samples1, samples2)

    def test_stochastic_param_as_aggregation_method(self):
        param = iap.IterativeNoiseAggregator(
            iap.Choice([0, 50]),
            iterations=5,
            aggregation_method=iap.Deterministic("max"))

        assert isinstance(param.aggregation_method, iap.Deterministic)
        assert param.aggregation_method.value == "max"

    def test_bad_datatype_for_aggregation_method(self):
        with self.assertRaises(Exception) as context:
            _ = iap.IterativeNoiseAggregator(
                iap.Choice([0, 50]), iterations=5, aggregation_method=False)

        self.assertTrue(
            "Expected aggregation_method to be" in str(context.exception))

    def test_bad_datatype_for_iterations(self):
        with self.assertRaises(Exception) as context:
            _ = iap.IterativeNoiseAggregator(
                iap.Choice([0, 50]),
                iterations=False,
                aggregation_method="max")

        self.assertTrue("Expected iterations to be" in str(context.exception))


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        param = iap.Sigmoid(
            iap.Deterministic(0),
            threshold=(-10, 10),
            activated=True,
            mul=1,
            add=0)
        assert (
            param.__str__()
            == param.__repr__()
            == (
                "Sigmoid("
                "Deterministic(int 0), "
                "Uniform("
                "Deterministic(int -10), "
                "Deterministic(int 10)"
                "), "
                "Deterministic(int 1), "
                "1, "
                "0)"
            )
        )

    def test_activated_is_true(self):
        param = iap.Sigmoid(
            iap.Deterministic(5),
            add=0,
            mul=1,
            threshold=0.5,
            activated=True)

        expected = 1 / (1 + np.exp(-(5 * 1 + 0 - 0.5)))
        sample = param.draw_sample()
        samples = param.draw_samples((5, 10))

        assert sample.shape == tuple()
        assert samples.shape == (5, 10)
        assert expected - _eps(sample) < sample < expected + _eps(sample)
        assert np.all(
            np.logical_and(
                expected - _eps(samples) < samples,
                samples < expected + _eps(samples)
            )
        )

    def test_activated_is_false(self):
        param = iap.Sigmoid(
            iap.Deterministic(5),
            add=0,
            mul=1,
            threshold=0.5,
            activated=False)

        expected = 5
        sample = param.draw_sample()
        samples = param.draw_samples((5, 10))

        assert sample.shape == tuple()
        assert samples.shape == (5, 10)
        assert expected - _eps(sample) < sample < expected + _eps(sample)
        assert np.all(
            np.logical_and(
                expected - _eps(sample) < samples,
                samples < expected + _eps(sample)
            )
        )

    def test_activated_is_probabilistic(self):
        param = iap.Sigmoid(
            iap.Deterministic(5),
            add=0,
            mul=1,
            threshold=0.5,
            activated=0.5)

        expected_first = 5
        expected_second = 1 / (1 + np.exp(-(5 * 1 + 0 - 0.5)))
        seen = [0, 0]
        for _ in sm.xrange(1000):
            sample = param.draw_sample()
            diff_first = abs(sample - expected_first)
            diff_second = abs(sample - expected_second)
            if diff_first < _eps(sample):
                seen[0] += 1
            elif diff_second < _eps(sample):
                seen[1] += 1
            else:
                assert False

        assert 500 - 150 < seen[0] < 500 + 150
        assert 500 - 150 < seen[1] < 500 + 150

    def test_value_is_stochastic_param(self):
        param = iap.Sigmoid(
            iap.Choice([1, 10]),
            add=0,
            mul=1,
            threshold=0.5,
            activated=True)

        expected_first = 1 / (1 + np.exp(-(1 * 1 + 0 - 0.5)))
        expected_second = 1 / (1 + np.exp(-(10 * 1 + 0 - 0.5)))
        seen = [0, 0]
        for _ in sm.xrange(1000):
            sample = param.draw_sample()
            diff_first = abs(sample - expected_first)
            diff_second = abs(sample - expected_second)
            if diff_first < _eps(sample):
                seen[0] += 1
            elif diff_second < _eps(sample):
                seen[1] += 1
            else:
                assert False

        assert 500 - 150 < seen[0] < 500 + 150
        assert 500 - 150 < seen[1] < 500 + 150

    def test_mul_add_threshold_with_various_fixed_values(self):
        muls = [0.1, 1, 10.3]
        adds = [-5.7, -0.0734, 0, 0.0734, 5.7]
        vals = [-1, -0.7, 0, 0.7, 1]
        threshs = [-5.7, -0.0734, 0, 0.0734, 5.7]
        for mul, add, val, thresh in itertools.product(muls, adds, vals,
                                                       threshs):
            with self.subTest(mul=mul, add=add, val=val, threshold=thresh):
                param = iap.Sigmoid(
                    iap.Deterministic(val),
                    add=add,
                    mul=mul,
                    threshold=thresh)

                sample = param.draw_sample()
                samples = param.draw_samples((2, 3))
                dt = sample.dtype
                val_ = np.array([val], dtype=dt)
                mul_ = np.array([mul], dtype=dt)
                add_ = np.array([add], dtype=dt)
                thresh_ = np.array([thresh], dtype=dt)
                expected = (
                    1 / (
                        1 + np.exp(
                            -(val_ * mul_ + add_ - thresh_)
                        )
                    )
                )

                assert sample.shape == tuple()
                assert samples.shape == (2, 3)
                assert (
                    expected - 5*_eps(sample)
                    < sample <
                    expected + 5*_eps(sample)
                )
                assert np.all(
                    np.logical_and(
                        expected - 5*_eps(sample) < samples,
                        samples < expected + 5*_eps(sample)
                    )
                )

    def test_samples_same_values_for_same_seeds(self):
        param = iap.Sigmoid(
            iap.Choice([1, 10]),
            add=0,
            mul=1,
            threshold=0.5,
            activated=True)

        samples1 = param.draw_samples((100, 10),
                                      random_state=np.random.RandomState(1234))
        samples2 = param.draw_samples((100, 10),
                                      random_state=np.random.RandomState(1234))

        assert samples1.shape == (100, 10)
        assert samples2.shape == (100, 10)
        assert np.array_equal(samples1, samples2)
