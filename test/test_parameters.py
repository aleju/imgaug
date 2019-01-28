from __future__ import print_function, division, absolute_import

import time
import sys

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm
import skimage
import skimage.data
import scipy
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

import imgaug as ia
from imgaug import parameters as iap
from imgaug.testutils import reseed


def main():
    time_start = time.time()

    test_parameters_handle_continuous_param()
    test_parameters_handle_discrete_param()
    test_parameters_handle_probability_param()
    test_parameters_force_np_float_dtype()
    test_parameters_both_np_float_if_one_is_float()
    test_parameters_draw_distribution_graph()
    test_parameters_Biomial()
    test_parameters_Choice()
    test_parameters_DiscreteUniform()
    test_parameters_Poisson()
    test_parameters_Normal()
    test_parameters_Laplace()
    test_parameters_ChiSquare()
    test_parameters_Weibull()
    test_parameters_Uniform()
    test_parameters_Beta()
    test_parameters_Deterministic()
    test_parameters_FromLowerResolution()
    test_parameters_Clip()
    test_parameters_Discretize()
    test_parameters_Multiply()
    test_parameters_Divide()
    test_parameters_Add()
    test_parameters_Subtract()
    test_parameters_Power()
    test_parameters_Absolute()
    test_parameters_RandomSign()
    test_parameters_ForceSign()
    test_parameters_Positive()
    test_parameters_Negative()
    test_parameters_IterativeNoiseAggregator()
    test_parameters_Sigmoid()
    # test_parameters_SimplexNoise()
    # test_parameters_FrequencyNoise()
    test_parameters_operators()
    test_parameters_copy()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def _eps(arr):
    if ia.is_np_array(arr) and arr.dtype.kind == "f":
        return np.finfo(arr.dtype).eps
    return 1e-4


def test_parameters_handle_continuous_param():
    # value without value range
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test1]", value_range=None, tuple_to_uniform=True, list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test1]" in str(e)
    assert got_exception == False

    # value without value range as (None, None)
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test1b]", value_range=(None, None), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test1b]" in str(e)
    assert got_exception == False

    # stochastic parameter
    got_exception = False
    try:
        result = iap.handle_continuous_param(iap.Deterministic(1), "[test2]", value_range=None, tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test2]" in str(e)
    assert got_exception == False

    # value within value range
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test3]", value_range=(0, 10), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test3]" in str(e)
    assert got_exception == False

    # value outside of value range
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test4]", value_range=(2, 12), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test4]" in str(e)
    assert got_exception == True

    # value within value range (without lower bound)
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test5]", value_range=(None, 12), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test5]" in str(e)
    assert got_exception == False

    # value outside of value range (without lower bound)
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test6]", value_range=(None, 0), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test6]" in str(e)
    assert got_exception == True

    # value within value range (without upper bound)
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test7]", value_range=(-1, None), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test7]" in str(e)
    assert got_exception == False

    # value outside of value range (without upper bound)
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test8]", value_range=(2, None), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test8]" in str(e)
    assert got_exception == True

    # tuple as value, but no tuples allowed
    got_exception = False
    try:
        result = iap.handle_continuous_param((1, 2), "[test9]", value_range=None, tuple_to_uniform=False,
                                             list_to_choice=True)
        assert isinstance(result, iap.Uniform)
    except Exception as e:
        got_exception = True
        assert "[test9]" in str(e)
    assert got_exception == True

    # tuple as value and tuple allowed
    got_exception = False
    try:
        result = iap.handle_continuous_param((1, 2), "[test10]", value_range=None, tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Uniform)
    except Exception as e:
        got_exception = True
        assert "[test10]" in str(e)
    assert got_exception == False

    # tuple as value and tuple allowed and tuple within value range
    got_exception = False
    try:
        result = iap.handle_continuous_param((1, 2), "[test11]", value_range=(0, 10), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Uniform)
    except Exception as e:
        got_exception = True
        assert "[test11]" in str(e)
    assert got_exception == False

    # tuple as value and tuple allowed and tuple partially outside of value range
    got_exception = False
    try:
        result = iap.handle_continuous_param((1, 2), "[test12]", value_range=(1.5, 13), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Uniform)
    except Exception as e:
        got_exception = True
        assert "[test12]" in str(e)
    assert got_exception == True

    # tuple as value and tuple allowed and tuple fully outside of value range
    got_exception = False
    try:
        result = iap.handle_continuous_param((1, 2), "[test13]", value_range=(3, 13), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Uniform)
    except Exception as e:
        got_exception = True
        assert "[test13]" in str(e)
    assert got_exception == True

    # list as value, but no list allowed
    got_exception = False
    try:
        result = iap.handle_continuous_param([1, 2, 3], "[test14]", value_range=None, tuple_to_uniform=True,
                                             list_to_choice=False)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test14]" in str(e)
    assert got_exception == True

    # list as value and list allowed
    got_exception = False
    try:
        result = iap.handle_continuous_param([1, 2, 3], "[test15]", value_range=None, tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test15]" in str(e)
    assert got_exception == False

    # list as value and list allowed and list partially outside of value range
    got_exception = False
    try:
        result = iap.handle_continuous_param([1, 2], "[test16]", value_range=(1.5, 13), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test16]" in str(e)
    assert got_exception == True

    # list as value and list allowed and list fully outside of value range
    got_exception = False
    try:
        result = iap.handle_continuous_param([1, 2], "[test17]", value_range=(3, 13), tuple_to_uniform=True,
                                             list_to_choice=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test17]" in str(e)
    assert got_exception == True

    # single value within value range given as callable
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test18]", value_range=lambda x: -1 < x < 1, tuple_to_uniform=True,
                                             list_to_choice=True)
    except Exception as e:
        got_exception = True
        assert "[test18]" in str(e)
    assert got_exception == False

    # bad datatype for value range
    got_exception = False
    try:
        result = iap.handle_continuous_param(1, "[test19]", value_range=False, tuple_to_uniform=True,
                                             list_to_choice=True)
    except Exception as e:
        got_exception = True
        assert "Unexpected input for value_range" in str(e)
    assert got_exception == True


def test_parameters_handle_discrete_param():
    # float value without value range when no float value is allowed
    got_exception = False
    try:
        result = iap.handle_discrete_param(1.5, "[test0]", value_range=None, tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=False)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test0]" in str(e)
    assert got_exception == True

    # value without value range
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test1]", value_range=None, tuple_to_uniform=True, list_to_choice=True,
                                           allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test1]" in str(e)
    assert got_exception is False

    # value without value range as (None, None)
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test1b]", value_range=(None, None), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test1b]" in str(e)
    assert got_exception is False

    # stochastic parameter
    got_exception = False
    try:
        result = iap.handle_discrete_param(iap.Deterministic(1), "[test2]", value_range=None, tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test2]" in str(e)
    assert got_exception is False

    # value within value range
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test3]", value_range=(0, 10), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test3]" in str(e)
    assert got_exception is False

    # value outside of value range
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test4]", value_range=(2, 12), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test4]" in str(e)
    assert got_exception is True

    # value within value range (without lower bound)
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test5]", value_range=(None, 12), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test5]" in str(e)
    assert got_exception is False

    # value outside of value range (without lower bound)
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test6]", value_range=(None, 0), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test6]" in str(e)
    assert got_exception is True

    # value within value range (without upper bound)
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test7]", value_range=(-1, None), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test7]" in str(e)
    assert got_exception is False

    # value outside of value range (without upper bound)
    got_exception = False
    try:
        result = iap.handle_discrete_param(1, "[test8]", value_range=(2, None), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Deterministic)
    except Exception as e:
        got_exception = True
        assert "[test8]" in str(e)
    assert got_exception is True

    # tuple as value, but no tuples allowed
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 2), "[test9]", value_range=None, tuple_to_uniform=False,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test9]" in str(e)
    assert got_exception is True

    # tuple as value and tuple allowed
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 2), "[test10]", value_range=None, tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test10]" in str(e)
    assert got_exception is False

    # tuple as value and tuple allowed and tuple within value range
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 2), "[test11]", value_range=(0, 10), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test11]" in str(e)
    assert got_exception is False

    # tuple as value and tuple allowed and tuple within value range with allow_floats=False
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 2), "[test11b]", value_range=(0, 10), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=False)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test11b]" in str(e)
    assert got_exception is False

    # tuple as value and tuple allowed and tuple partially outside of value range
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 3), "[test12]", value_range=(2, 13), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test12]" in str(e)
    assert got_exception is True

    # tuple as value and tuple allowed and tuple fully outside of value range
    got_exception = False
    try:
        result = iap.handle_discrete_param((1, 2), "[test13]", value_range=(3, 13), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.DiscreteUniform)
    except Exception as e:
        got_exception = True
        assert "[test13]" in str(e)
    assert got_exception is True

    # list as value, but no list allowed
    got_exception = False
    try:
        result = iap.handle_discrete_param([1, 2, 3], "[test14]", value_range=None, tuple_to_uniform=True,
                                           list_to_choice=False, allow_floats=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test14]" in str(e)
    assert got_exception is True

    # list as value and list allowed
    got_exception = False
    try:
        result = iap.handle_discrete_param([1, 2, 3], "[test15]", value_range=None, tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test15]" in str(e)
    assert got_exception is False

    # list as value and list allowed and list partially outside of value range
    got_exception = False
    try:
        result = iap.handle_discrete_param([1, 3], "[test16]", value_range=(2, 13), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test16]" in str(e)
    assert got_exception is True

    # list as value and list allowed and list fully outside of value range
    got_exception = False
    try:
        result = iap.handle_discrete_param([1, 2], "[test17]", value_range=(3, 13), tuple_to_uniform=True,
                                           list_to_choice=True, allow_floats=True)
        assert isinstance(result, iap.Choice)
    except Exception as e:
        got_exception = True
        assert "[test17]" in str(e)
    assert got_exception is True

    # single value within value range given as callable
    got_exception = False
    try:
        _ = iap.handle_discrete_param(1, "[test18]", value_range=lambda x: -1 < x < 1, tuple_to_uniform=True,
                                      list_to_choice=True)
    except Exception as e:
        got_exception = True
        assert "[test18]" in str(e)
    assert got_exception is False

    # bad datatype for value range
    got_exception = False
    try:
        _ = iap.handle_discrete_param(1, "[test19]", value_range=False, tuple_to_uniform=True,
                                      list_to_choice=True)
    except Exception as e:
        got_exception = True
        assert "Unexpected input for value_range" in str(e)
    assert got_exception is True


def test_parameters_handle_probability_param():
    for val in [True, False, 0, 1, 0.0, 1.0]:
        p = iap.handle_probability_param(val, "[test1]")
        assert isinstance(p, iap.Deterministic)
        assert p.value == int(val)

    for val in [0.0001, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999, 0.9999]:
        p = iap.handle_probability_param(val, "[test2]")
        assert isinstance(p, iap.Binomial)
        assert isinstance(p.p, iap.Deterministic)
        assert val-1e-8 < p.p.value < val+1e-8

    det = iap.Deterministic(1)
    p = iap.handle_probability_param(det, "[test3]")
    assert p == det

    got_exception = False
    try:
        _p = iap.handle_probability_param("test", "[test4]")
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _p = iap.handle_probability_param(-0.01, "[test5]")
    except AssertionError:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _p = iap.handle_probability_param(1.01, "[test6]")
    except AssertionError:
        got_exception = True
    assert got_exception


def test_parameters_force_np_float_dtype():
    dtypes = [
        (np.float16, np.float16),
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.uint8, np.float64),
        (np.int32, np.float64)
    ]
    for i, (dtype_in, dtype_out) in enumerate(dtypes):
        assert iap.force_np_float_dtype(np.zeros((1,), dtype=dtype_in)).dtype == dtype_out,\
            "force_np_float_dtype() failed at %d" % (i,)


def test_parameters_both_np_float_if_one_is_float():
    a1 = np.zeros((1,), dtype=np.float16)
    b1 = np.zeros((1,), dtype=np.float32)
    a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
    assert a2.dtype.type == np.float16, a2.dtype.type
    assert b2.dtype.type == np.float32, b2.dtype.type

    a1 = np.zeros((1,), dtype=np.float16)
    b1 = np.zeros((1,), dtype=np.int32)
    a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
    assert a2.dtype.type == np.float16, a2.dtype.type
    assert b2.dtype.type == np.float64, b2.dtype.type

    a1 = np.zeros((1,), dtype=np.int32)
    b1 = np.zeros((1,), dtype=np.float16)
    a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
    assert a2.dtype.type == np.float64, a2.dtype.type
    assert b2.dtype.type == np.float16, b2.dtype.type

    a1 = np.zeros((1,), dtype=np.int32)
    b1 = np.zeros((1,), dtype=np.uint8)
    a2, b2 = iap.both_np_float_if_one_is_float(a1, b1)
    assert a2.dtype.type == np.float64, a2.dtype.type
    assert b2.dtype.type == np.float64, b2.dtype.type


class TestDrawDistributionsGrid(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_basic_functionality(self):
        params = [mock.Mock(), mock.Mock()]
        params[0].draw_distribution_graph.return_value = np.zeros((1, 1, 3), dtype=np.uint8)
        params[1].draw_distribution_graph.return_value = np.zeros((1, 1, 3), dtype=np.uint8)

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


def test_parameters_draw_distribution_graph():
    # this test is very rough as we get a not-very-well-defined image out of the function
    param = iap.Uniform(0.0, 1.0)

    graph_img = param.draw_distribution_graph(title=None, size=(10000,), bins=100)
    assert graph_img.ndim == 3
    assert graph_img.shape[2] == 3

    # at least 10% of the image should be white-ish (background)
    nb_white = np.sum(graph_img[..., :] > [200, 200, 200])
    nb_all = np.prod(graph_img.shape)
    assert nb_white > 0.1 * nb_all

    graph_img_title = param.draw_distribution_graph(title="test", size=(10000,), bins=100)
    assert graph_img_title.ndim == 3
    assert graph_img_title.shape[2] == 3
    assert not np.array_equal(graph_img_title, graph_img)


def test_parameters_Biomial():
    reseed()

    param = iap.Binomial(0)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == 0
    assert np.all(samples == 0)
    assert param.__str__() == param.__repr__() == "Binomial(Deterministic(int 0))"

    param = iap.Binomial(1.0)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == 1
    assert np.all(samples == 1)
    assert param.__str__() == param.__repr__() == "Binomial(Deterministic(float 1.00000000))"

    param = iap.Binomial(0.5)
    sample = param.draw_sample()
    samples = param.draw_samples((10000,))
    assert sample.shape == tuple()
    assert samples.shape == (10000,)
    assert sample in [0, 1]
    unique, counts = np.unique(samples, return_counts=True)
    assert len(unique) == 2
    for val, count in zip(unique, counts):
        if val == 0:
            assert 5000 - 500 < count < 5000 + 500
        elif val == 1:
            assert 5000 - 500 < count < 5000 + 500
        else:
            assert False

    param = iap.Binomial(iap.Choice([0.25, 0.75]))
    for _ in sm.xrange(10):
        samples = param.draw_samples((1000,))
        p = np.sum(samples) / samples.size
        assert (0.25 - 0.05 < p < 0.25 + 0.05) or (0.75 - 0.05 < p < 0.75 + 0.05)

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

    param = iap.Binomial(0.5)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)


def test_parameters_Choice():
    reseed()

    param = iap.Choice([0, 1, 2])
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [0, 1, 2]
    assert np.all(np.logical_or(np.logical_or(samples == 0, samples == 1), samples==2))
    assert param.__str__() == param.__repr__() == "Choice(a=[0, 1, 2], replace=True, p=None)"

    samples = param.draw_samples((10000,))
    expected = 10000/3
    expected_tolerance = expected * 0.05
    for v in [0, 1, 2]:
        count = np.sum(samples == v)
        assert expected - expected_tolerance < count < expected + expected_tolerance

    param = iap.Choice([-1, 1])
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [-1, 1]
    assert np.all(np.logical_or(samples == -1, samples == 1))

    param = iap.Choice([-1.2, 1.7])
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert -1.2 - _eps(sample) < sample < -1.2 + _eps(sample) or 1.7 - _eps(sample) < sample < 1.7 + _eps(sample)
    assert np.all(
        np.logical_or(
            np.logical_and(-1.2 - _eps(sample) < samples, samples < -1.2 + _eps(sample)),
            np.logical_and(1.7 - _eps(sample) < samples, samples < 1.7 + _eps(sample))
        )
    )

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

    param = iap.Choice([1+i for i in sm.xrange(100)], replace=False)
    samples = param.draw_samples((50,))
    seen = [0 for _ in sm.xrange(100)]
    for sample in samples:
        seen[sample-1] += 1
    assert all([count in [0, 1] for count in seen])

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

    param = iap.Choice([-1, 0, 1, 2, 3])
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)

    got_exception = False
    try:
        _ = iap.Choice(123)
    except Exception as exc:
        assert "Expected a to be an iterable" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iap.Choice([1, 2], p=123)
    except Exception as exc:
        assert "Expected p to be" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iap.Choice([1, 2], p=[1])
    except Exception as exc:
        assert "Expected lengths of" in str(exc)
        got_exception = True
    assert got_exception


def test_parameters_DiscreteUniform():
    reseed()

    param = iap.DiscreteUniform(0, 2)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [0, 1, 2]
    assert np.all(np.logical_or(np.logical_or(samples == 0, samples == 1), samples==2))
    assert param.__str__() == param.__repr__() == "DiscreteUniform(Deterministic(int 0), Deterministic(int 2))"

    samples = param.draw_samples((10000,))
    expected = 10000/3
    expected_tolerance = expected * 0.05
    for v in [0, 1, 2]:
        count = np.sum(samples == v)
        assert expected - expected_tolerance < count < expected + expected_tolerance

    param = iap.DiscreteUniform(-1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [-1, 0, 1]
    assert np.all(np.logical_or(np.logical_or(samples == -1, samples == 0), samples==1))

    param = iap.DiscreteUniform(-1.2, 1.2)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [-1, 0, 1]
    assert np.all(np.logical_or(np.logical_or(samples == -1, samples == 0), samples==1))

    param = iap.DiscreteUniform(1, -1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [-1, 0, 1]
    assert np.all(np.logical_or(np.logical_or(samples == -1, samples == 0), samples==1))

    param = iap.DiscreteUniform(1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((100,))
    assert sample == 1
    assert np.all(samples == 1)

    param = iap.Uniform(-1, 1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)


def test_parameters_Poisson():
    reseed()

    param = iap.Poisson(1)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).poisson(lam=1, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert 0 < sample
    assert param.__str__() == param.__repr__() == "Poisson(Deterministic(int 1))"

    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        count_direct = int(np.sum(samples_direct == i))
        count = np.sum(samples == i)
        tolerance = max(count_direct * 0.1, 250)
        assert count_direct - tolerance < count < count_direct + tolerance

    param = iap.Poisson(1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)


def test_parameters_Normal():
    reseed()

    param = iap.Normal(0, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).normal(loc=0, scale=1, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert param.__str__() == param.__repr__() == "Normal(loc=Deterministic(int 0), scale=Deterministic(int 1))"

    samples = np.clip(samples, -1, 1)
    samples_direct = np.clip(samples_direct, -1, 1)
    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(-1.0, 1.0), density=False)
    hist_direct, _ = np.histogram(samples_direct, bins=nb_bins, range=(-1.0, 1.0), density=False)
    tolerance = 0.05
    for nb_samples, nb_samples_direct in zip(hist, hist_direct):
        density = nb_samples / samples.size
        density_direct = nb_samples_direct / samples_direct.size
        assert density_direct - tolerance < density < density_direct + tolerance

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

    param1 = iap.Normal(0, 1)
    param2 = iap.Normal(0, 100)
    samples1 = param1.draw_samples((1000,))
    samples2 = param2.draw_samples((1000,))
    assert np.std(samples1) < np.std(samples2)
    assert 100 - 10 < np.std(samples2) < 100 + 10

    param = iap.Normal(0, 1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_Laplace():
    reseed()

    param = iap.Laplace(0, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).laplace(loc=0, scale=1, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert param.__str__() == param.__repr__() == "Laplace(loc=Deterministic(int 0), scale=Deterministic(int 1))"

    samples = np.clip(samples, -1, 1)
    samples_direct = np.clip(samples_direct, -1, 1)
    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(-1.0, 1.0), density=False)
    hist_direct, _ = np.histogram(samples_direct, bins=nb_bins, range=(-1.0, 1.0), density=False)
    tolerance = 0.05
    for nb_samples, nb_samples_direct in zip(hist, hist_direct):
        density = nb_samples / samples.size
        density_direct = nb_samples_direct / samples_direct.size
        assert density_direct - tolerance < density < density_direct + tolerance

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

    param1 = iap.Laplace(0, 1)
    param2 = iap.Laplace(0, 100)
    samples1 = param1.draw_samples((1000,))
    samples2 = param2.draw_samples((1000,))
    assert np.var(samples1) < np.var(samples2)

    param1 = iap.Laplace(1, 0)
    samples = param1.draw_samples((100,))
    assert np.all(np.logical_and(
        samples > 1 - _eps(samples),
        samples < 1 + _eps(samples)
    ))

    param = iap.Laplace(0, 1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_ChiSquare():
    reseed()

    param = iap.ChiSquare(1)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).chisquare(df=1, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert 0 <= sample
    assert np.all(0 <= samples)
    assert param.__str__() == param.__repr__() == "ChiSquare(df=Deterministic(int 1))"

    samples = np.clip(samples, 0, 3)
    samples_direct = np.clip(samples_direct, 0, 3)
    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(0, 3.0), density=False)
    hist_direct, _ = np.histogram(samples_direct, bins=nb_bins, range=(0, 3.0), density=False)
    tolerance = 0.05
    for nb_samples, nb_samples_direct in zip(hist, hist_direct):
        density = nb_samples / samples.size
        density_direct = nb_samples_direct / samples_direct.size
        assert density_direct - tolerance < density < density_direct + tolerance

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

    param1 = iap.ChiSquare(1)
    param2 = iap.ChiSquare(10)
    samples1 = param1.draw_samples((1000,))
    samples2 = param2.draw_samples((1000,))
    assert np.var(samples1) < np.var(samples2)
    assert 2*1 - 1.0 < np.var(samples1) < 2*1 + 1.0
    assert 2*10 - 5.0 < np.var(samples2) < 2*10 + 5.0

    param = iap.ChiSquare(1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_Weibull():
    reseed()

    param = iap.Weibull(1)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).weibull(a=1, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert 0 <= sample
    assert np.all(0 <= samples)
    assert param.__str__() == param.__repr__() == "Weibull(a=Deterministic(int 1))"

    samples = np.clip(samples, 0, 2)
    samples_direct = np.clip(samples_direct, 0, 2)
    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(0, 2.0), density=False)
    hist_direct, _ = np.histogram(samples_direct, bins=nb_bins, range=(0, 2.0), density=False)
    tolerance = 0.05
    for nb_samples, nb_samples_direct in zip(hist, hist_direct):
        density = nb_samples / samples.size
        density_direct = nb_samples_direct / samples_direct.size
        assert density_direct - tolerance < density < density_direct + tolerance

    param = iap.Weibull(iap.Choice([1, 0.5]))
    expected_first = scipy.special.gamma(1 + 1/1)
    expected_second = scipy.special.gamma(1 + 1/0.5)
    seen = [0, 0]
    for _ in sm.xrange(100):
        samples = param.draw_samples((50000,))
        observed = np.mean(samples)

        if expected_first - 0.2 * expected_first < observed < expected_first + 0.2 * expected_first:
            seen[0] += 1
        elif expected_second - 0.2 * expected_second < observed < expected_second + 0.2 * expected_second:
            seen[1] += 1
        else:
            assert False

    assert 50 - 25 < seen[0] < 50 + 25
    assert 50 - 25 < seen[1] < 50 + 25

    param1 = iap.Weibull(1)
    param2 = iap.Weibull(0.5)
    samples1 = param1.draw_samples((10000,))
    samples2 = param2.draw_samples((10000,))
    assert np.var(samples1) < np.var(samples2)
    expected_first = scipy.special.gamma(1 + 2/1) - (scipy.special.gamma(1 + 1/1))**2
    expected_second = scipy.special.gamma(1 + 2/0.5) - (scipy.special.gamma(1 + 1/0.5))**2
    assert expected_first - 0.2 * expected_first < np.var(samples1) < expected_first + 0.2 * expected_first
    assert expected_second - 0.2 * expected_second < np.var(samples2) < expected_second + 0.2 * expected_second

    param = iap.Weibull(1)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_Uniform():
    reseed()

    param = iap.Uniform(0, 1.0)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert 0 - _eps(sample) < sample < 1.0 + _eps(sample)
    assert np.all(np.logical_and(0 - _eps(samples) < samples, samples < 1.0 + _eps(samples)))
    assert param.__str__() == param.__repr__() == "Uniform(Deterministic(int 0), Deterministic(float 1.00000000))"

    samples = param.draw_samples((10000,))
    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / samples.size
        assert density_expected - density_tolerance < density < density_expected + density_tolerance

    param = iap.Uniform(-1.0, 1.0)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert -1.0 - _eps(sample) < sample < 1.0 + _eps(sample)
    assert np.all(np.logical_and(-1.0 - _eps(samples) < samples, samples < 1.0 + _eps(samples)))

    param = iap.Uniform(1.0, -1.0)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert -1.0 - _eps(sample) < sample < 1.0 + _eps(sample)
    assert np.all(np.logical_and(-1.0 - _eps(samples) < samples, samples < 1.0 + _eps(samples)))

    param = iap.Uniform(-1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert -1.0 - _eps(sample) < sample < 1.0 + _eps(sample)
    assert np.all(np.logical_and(-1.0 - _eps(samples) < samples, samples < 1.0 + _eps(samples)))

    param = iap.Uniform(1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert 1.0 - _eps(sample) < sample < 1.0 + _eps(sample)
    assert np.all(np.logical_and(1.0 - _eps(samples) < samples, samples < 1.0 + _eps(samples)))

    param = iap.Uniform(-1.0, 1.0)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_Beta():
    def _mean(alpha, beta):
        return alpha / (alpha + beta)

    def _var(alpha, beta):
        return (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

    reseed()

    param = iap.Beta(0.5, 0.5)
    sample = param.draw_sample()
    samples = param.draw_samples((100, 1000))
    samples_direct = np.random.RandomState(1234).beta(a=0.5, b=0.5, size=(100, 1000))
    assert sample.shape == tuple()
    assert samples.shape == (100, 1000)
    assert 0 - _eps(sample) < sample < 1.0 + _eps(sample)
    assert np.all(np.logical_and(0 - _eps(sample) <= samples, samples <= 1.0 + _eps(sample)))
    assert param.__str__() == param.__repr__() == \
        "Beta(Deterministic(float 0.50000000), Deterministic(float 0.50000000))"

    nb_bins = 10
    hist, _ = np.histogram(samples, bins=nb_bins, range=(0, 1.0), density=False)
    hist_direct, _ = np.histogram(samples_direct, bins=nb_bins, range=(0, 1.0), density=False)
    tolerance = 0.05
    for nb_samples, nb_samples_direct in zip(hist, hist_direct):
        density = nb_samples / samples.size
        density_direct = nb_samples_direct / samples_direct.size
        assert density_direct - tolerance < density < density_direct + tolerance

    param = iap.Beta(iap.Choice([0.5, 2]), 0.5)
    expected_first = _mean(0.5, 0.5)
    expected_second = _mean(2, 0.5)
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

    param1 = iap.Beta(2, 2)
    param2 = iap.Beta(0.5, 0.5)
    samples1 = param1.draw_samples((10000,))
    samples2 = param2.draw_samples((10000,))
    assert np.var(samples1) < np.var(samples2)
    expected_first = _var(2, 2)
    expected_second = _var(0.5, 0.5)
    assert expected_first - 0.1 * expected_first < np.var(samples1) < expected_first + 0.1 * expected_first
    assert expected_second - 0.1 * expected_second < np.var(samples2) < expected_second + 0.1 * expected_second

    param = iap.Beta(0.5, 0.5)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)


def test_parameters_Deterministic():
    reseed()

    values_int = [-100, -54, -1, 0, 1, 54, 100]
    values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

    for value in values_int:
        param = iap.Deterministic(value)

        sample1 = param.draw_sample()
        sample2 = param.draw_sample()
        assert sample1.shape == tuple()
        assert sample1 == sample2

        samples1 = param.draw_samples(10)
        samples2 = param.draw_samples(10)
        samples3 = param.draw_samples((5, 3))
        samples4 = param.draw_samples((5, 3))
        samples5 = param.draw_samples((4, 5, 3))
        samples6 = param.draw_samples((4, 5, 3))

        samples1_unique = np.unique(samples1)
        samples2_unique = np.unique(samples2)
        samples3_unique = np.unique(samples3)
        samples4_unique = np.unique(samples4)
        samples5_unique = np.unique(samples5)
        samples6_unique = np.unique(samples6)

        assert samples1.shape == (10,)
        assert samples2.shape == (10,)
        assert samples3.shape == (5, 3)
        assert samples4.shape == (5, 3)
        assert samples5.shape == (4, 5, 3)
        assert samples6.shape == (4, 5, 3)
        assert len(samples1_unique) == 1 and samples1_unique[0] == value
        assert len(samples2_unique) == 1 and samples2_unique[0] == value
        assert len(samples3_unique) == 1 and samples3_unique[0] == value
        assert len(samples4_unique) == 1 and samples4_unique[0] == value
        assert len(samples5_unique) == 1 and samples5_unique[0] == value
        assert len(samples6_unique) == 1 and samples6_unique[0] == value

        rs1 = np.random.RandomState(123456)
        rs2 = np.random.RandomState(123456)
        assert np.array_equal(
            param.draw_samples(20, random_state=rs1),
            param.draw_samples(20, random_state=rs2)
        )

    for value in values_float:
        param = iap.Deterministic(value)

        sample1 = param.draw_sample()
        sample2 = param.draw_sample()
        assert sample1.shape == tuple()
        assert sample1 - _eps(sample1) < sample2 < sample1 + _eps(sample1)

        samples1 = param.draw_samples(10)
        samples2 = param.draw_samples(10)
        samples3 = param.draw_samples((5, 3))
        samples4 = param.draw_samples((5, 3))
        samples5 = param.draw_samples((4, 5, 3))
        samples6 = param.draw_samples((4, 5, 3))

        samples1_sorted = np.sort(samples1)
        samples2_sorted = np.sort(samples2)
        samples3_sorted = np.sort(samples3.flatten())
        samples4_sorted = np.sort(samples4.flatten())
        samples5_sorted = np.sort(samples5.flatten())
        samples6_sorted = np.sort(samples6.flatten())

        assert samples1.shape == (10,)
        assert samples2.shape == (10,)
        assert samples3.shape == (5, 3)
        assert samples4.shape == (5, 3)
        assert samples5.shape == (4, 5, 3)
        assert samples6.shape == (4, 5, 3)
        assert samples1_sorted[0] - _eps(samples1_sorted[0]) < samples1_sorted[-1] < samples1_sorted[0] + _eps(samples1_sorted[0])
        assert samples2_sorted[0] - _eps(samples2_sorted[0]) < samples2_sorted[-1] < samples2_sorted[0] + _eps(samples2_sorted[0])
        assert samples3_sorted[0] - _eps(samples3_sorted[0]) < samples3_sorted[-1] < samples3_sorted[0] + _eps(samples3_sorted[0])
        assert samples4_sorted[0] - _eps(samples4_sorted[0]) < samples4_sorted[-1] < samples4_sorted[0] + _eps(samples4_sorted[0])
        assert samples5_sorted[0] - _eps(samples5_sorted[0]) < samples5_sorted[-1] < samples5_sorted[0] + _eps(samples5_sorted[0])
        assert samples6_sorted[0] - _eps(samples6_sorted[0]) < samples6_sorted[-1] < samples6_sorted[0] + _eps(samples6_sorted[0])

        rs1 = np.random.RandomState(123456)
        rs2 = np.random.RandomState(123456)
        assert np.allclose(
            param.draw_samples(20, random_state=rs1),
            param.draw_samples(20, random_state=rs2)
        )

    param = iap.Deterministic(0)
    assert param.__str__() == param.__repr__() == "Deterministic(int 0)"
    param = iap.Deterministic(1.0)
    assert param.__str__() == param.__repr__() == "Deterministic(float 1.00000000)"
    param = iap.Deterministic("test")
    assert param.__str__() == param.__repr__() == "Deterministic(test)"

    seen = [0, 0]
    for _ in sm.xrange(200):
        param = iap.Deterministic(iap.Choice([0, 1]))
        seen[param.value] += 1
    assert 100 - 50 < seen[0] < 100 + 50
    assert 100 - 50 < seen[1] < 100 + 50

    got_exception = False
    try:
        _ = iap.Deterministic([1, 2, 3])
    except Exception as exc:
        assert "Expected StochasticParameter object or number or string" in str(exc)
        got_exception = True
    assert got_exception


def test_parameters_FromLowerResolution():
    reseed()

    # (H, W, C)
    param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=8)
    samples = param.draw_samples((8, 8, 1))
    assert samples.shape == (8, 8, 1)
    uq = np.unique(samples)
    assert len(uq) == 2 and (0 in uq or 1 in uq)

    # (N, H, W, C)
    samples_nhwc = param.draw_samples((1, 8, 8, 1))
    assert samples_nhwc.shape == (1, 8, 8, 1)
    uq = np.unique(samples_nhwc)
    assert len(uq) == 2 and (0 in uq or 1 in uq)

    # (N, H, W, C, something) causing error
    got_exception = False
    try:
        _ = param.draw_samples((1, 8, 8, 1, 1))
    except Exception as exc:
        assert "FromLowerResolution can only generate samples of shape" in str(exc)
        got_exception = True
    assert got_exception

    # C=3
    param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=8)
    samples = param.draw_samples((8, 8, 3))
    assert samples.shape == (8, 8, 3)
    uq = np.unique(samples)
    assert len(uq) == 2 and (0 in uq or 1 in uq)

    # different sizes in px
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=16)
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(100):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # different sizes in px, one given as tuple (a, b)
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=(2, 16))
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(400):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # different sizes in px, given as StochasticParameter
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=iap.Deterministic(1))
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=iap.Choice([8, 16]))
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(100):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # bad datatype for size_px
    got_exception = False
    try:
        _ = iap.FromLowerResolution(iap.Binomial(0.5), size_px=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # min_size
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_px=1, min_size=16)
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(100):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # different sizes in percent
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=0.01)
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=0.8)
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(100):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # different sizes in percent, given as StochasticParameter
    param1 = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=iap.Deterministic(0.01))
    param2 = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=iap.Choice([0.4, 0.8]))
    seen_components = [0, 0]
    seen_pixels = [0, 0]
    for _ in sm.xrange(100):
        samples1 = param1.draw_samples((16, 16, 1))
        samples2 = param2.draw_samples((16, 16, 1))
        _, num1 = skimage.morphology.label(samples1, neighbors=4, background=0, return_num=True)
        _, num2 = skimage.morphology.label(samples2, neighbors=4, background=0, return_num=True)
        seen_components[0] += num1
        seen_components[1] += num2
        seen_pixels[0] += np.sum(samples1 == 1)
        seen_pixels[1] += np.sum(samples2 == 1)

    assert seen_components[0] < seen_components[1]
    assert seen_pixels[0] / seen_components[0] > seen_pixels[1] / seen_components[1]

    # bad datatype for size_percent
    got_exception = False
    try:
        _ = iap.FromLowerResolution(iap.Binomial(0.5), size_percent=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # method given as StochasticParameter
    param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=4, method=iap.Choice(["nearest", "linear"]))
    seen = [0, 0]
    for _ in sm.xrange(200):
        samples = param.draw_samples((16, 16, 1))
        nb_in_between = np.sum(np.logical_and(0.05 < samples, samples < 0.95))
        if nb_in_between == 0:
            seen[0] += 1
        else:
            seen[1] += 1
    assert 100 - 50 < seen[0] < 100 + 50
    assert 100 - 50 < seen[1] < 100 + 50

    # bad datatype for method
    got_exception = False
    try:
        _ = iap.FromLowerResolution(iap.Binomial(0.5), size_px=4, method=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # multiple calls with same random_state
    param = iap.FromLowerResolution(iap.Binomial(0.5), size_px=2)
    samples1 = param.draw_samples((10, 5, 1), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5, 1), random_state=np.random.RandomState(1234))
    assert np.allclose(samples1, samples2)

    # str / repr
    param = iap.FromLowerResolution(other_param=iap.Deterministic(0), size_percent=1, method="nearest")
    assert param.__str__() == param.__repr__() == "FromLowerResolution(size_percent=Deterministic(int 1), method=Deterministic(nearest), other_param=Deterministic(int 0))"
    param = iap.FromLowerResolution(other_param=iap.Deterministic(0), size_px=1, method="nearest")
    assert param.__str__() == param.__repr__() == "FromLowerResolution(size_px=Deterministic(int 1), method=Deterministic(nearest), other_param=Deterministic(int 0))"


def test_parameters_Clip():
    reseed()

    param = iap.Clip(iap.Deterministic(0), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == 0
    assert np.all(samples == 0)
    assert param.__str__() == param.__repr__() == "Clip(Deterministic(int 0), -1.000000, 1.000000)"

    param = iap.Clip(iap.Deterministic(1), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == 1
    assert np.all(samples == 1)

    param = iap.Clip(iap.Deterministic(-1), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == -1
    assert np.all(samples == -1)

    param = iap.Clip(iap.Deterministic(0.5), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert 0.5 - _eps(sample) < sample < 0.5 + _eps(sample)
    assert np.all(np.logical_and(0.5 - _eps(sample) <= samples, samples <= 0.5 + _eps(sample)))

    param = iap.Clip(iap.Deterministic(2), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == 1
    assert np.all(samples == 1)

    param = iap.Clip(iap.Deterministic(-2), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample == -1
    assert np.all(samples == -1)

    param = iap.Clip(iap.Choice([0, 2]), -1, 1)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [0, 1]
    assert np.all(np.logical_or(samples == 0, samples == 1))

    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)

    param = iap.Clip(iap.Deterministic(0), None, 1)
    sample = param.draw_sample()
    assert sample == 0
    assert param.__str__() == param.__repr__() == "Clip(Deterministic(int 0), None, 1.000000)"

    param = iap.Clip(iap.Deterministic(0), 0, None)
    sample = param.draw_sample()
    assert sample == 0
    assert param.__str__() == param.__repr__() == "Clip(Deterministic(int 0), 0.000000, None)"

    param = iap.Clip(iap.Deterministic(0), None, None)
    sample = param.draw_sample()
    assert sample == 0
    assert param.__str__() == param.__repr__() == "Clip(Deterministic(int 0), None, None)"


def test_parameters_Discretize():
    reseed()

    values = [-100.2, -54.3, -1.0, -1, -0.7, -0.00043, 0, 0.00043, 0.7, 1.0, 1, 54.3, 100.2]
    for value in values:
        value_expected = np.round(np.float64([value])).astype(np.int32)[0]
        param = iap.Discretize(iap.Deterministic(value))
        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))
        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        assert sample == value_expected
        assert np.all(samples == value_expected)

    param_orig = iap.DiscreteUniform(0, 1)
    param = iap.Discretize(param_orig)
    sample = param.draw_sample()
    samples = param.draw_samples((10, 5))
    assert sample.shape == tuple()
    assert samples.shape == (10, 5)
    assert sample in [0, 1]
    assert np.all(np.logical_or(samples == 0, samples == 1))

    param_orig = iap.DiscreteUniform(0, 2)
    param = iap.Discretize(param_orig)
    samples1 = param_orig.draw_samples((10000,))
    samples2 = param.draw_samples((10000,))
    assert np.all(np.abs(samples1 - samples2) < 0.2*(10000/3))

    param_orig = iap.DiscreteUniform(0, 2)
    param = iap.Discretize(param_orig)
    samples1 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((10, 5), random_state=np.random.RandomState(1234))
    assert np.array_equal(samples1, samples2)

    param = iap.Discretize(iap.Deterministic(0))
    assert param.__str__() == param.__repr__() == "Discretize(Deterministic(int 0))"


def test_parameters_Multiply():
    reseed()

    values_int = [-100, -54, -1, 0, 1, 54, 100]
    values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

    for v1 in values_int:
        for v2 in values_int:
            p = iap.Multiply(iap.Deterministic(v1), v2)
            assert p.draw_sample() == v1 * v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "i"
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 * v2)

            p = iap.Multiply(iap.Deterministic(v1), iap.Deterministic(v2))
            assert p.draw_sample() == v1 * v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype == np.int64
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 * v2)

    for v1 in values_float:
        for v2 in values_float:
            p = iap.Multiply(iap.Deterministic(v1), v2)
            sample = p.draw_sample()
            assert v1 * v2 - _eps(sample) < sample < v1 * v2 + _eps(sample)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 * v2)

            p = iap.Multiply(iap.Deterministic(v1), iap.Deterministic(v2))
            sample = p.draw_sample()
            assert v1 * v2 - _eps(sample) < sample < v1 * v2 + _eps(sample)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 * v2)

    param = iap.Multiply(iap.Deterministic(1.0), (1.0, 2.0), elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 * 1.0 - _eps(samples))
    assert np.all(samples < 1.0 * 2.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0])

    param = iap.Multiply(iap.Deterministic(1.0), (1.0, 2.0), elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 * 1.0 - _eps(samples))
    assert np.all(samples < 1.0 * 2.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Multiply(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 * 1.0 - _eps(samples))
    assert np.all(samples < 2.0 * 1.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Multiply(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 * 1.0 - _eps(samples))
    assert np.all(samples < 2.0 * 1.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Multiply(iap.Deterministic(0), 1, elementwise=False)
    assert param.__str__() == param.__repr__() == "Multiply(Deterministic(int 0), Deterministic(int 1), False)"


def test_parameters_Divide():
    reseed()

    values_int = [-100, -54, -1, 0, 1, 54, 100]
    values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

    for v1 in values_int:
        for v2 in values_int:
            if v2 == 0:
                v2 = 1

            p = iap.Divide(iap.Deterministic(v1), v2)
            assert p.draw_sample() == (v1 / v2)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.float64) + (v1 / v2))

            p = iap.Divide(iap.Deterministic(v1), iap.Deterministic(v2))
            assert p.draw_sample() == (v1 / v2)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.float64) + (v1 / v2))

    for v1 in values_float:
        for v2 in values_float:
            if v2 == 0:
                v2 = 1

            p = iap.Divide(iap.Deterministic(v1), v2)
            sample = p.draw_sample()
            assert (v1 / v2) - _eps(sample) <= sample <= (v1 / v2) + _eps(sample)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + (v1 / v2))

            p = iap.Divide(iap.Deterministic(v1), iap.Deterministic(v2))
            sample = p.draw_sample()
            assert (v1 / v2) - _eps(sample) <= sample <= (v1 / v2) + _eps(sample)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + (v1 / v2))

    param = iap.Divide(iap.Deterministic(1.0), (1.0, 2.0), elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > (1.0 / 2.0) - _eps(samples))
    assert np.all(samples < (1.0 / 1.0) + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert samples_sorted[0] - _eps(samples) < samples_sorted[-1] < samples_sorted[0] + _eps(samples)

    param = iap.Divide(iap.Deterministic(1.0), (1.0, 2.0), elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > (1.0 / 2.0) - _eps(samples))
    assert np.all(samples < (1.0 / 1.0) + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples) < samples_sorted[-1] < samples_sorted[0] + _eps(samples))

    param = iap.Divide(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > (1.0 / 1.0) - _eps(samples))
    assert np.all(samples < (2.0 / 1.0) + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples) < samples_sorted[-1] < samples_sorted[0] + _eps(samples))

    param = iap.Divide(iap.Deterministic(1), 0, elementwise=False)
    sample = param.draw_sample()
    assert sample == 1

    param = iap.Divide(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > (1.0 / 1.0) - _eps(samples))
    assert np.all(samples < (2.0 / 1.0) + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted) < samples_sorted[-1]
                < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted))

    # test division by zero automatically being converted to division by 1
    param = iap.Divide(2, iap.Choice([0, 2]), elementwise=True)
    samples = param.draw_samples((10, 20))
    samples_unique = np.sort(np.unique(samples.flatten()))
    assert samples_unique[0] == 1 and samples_unique[1] == 2

    param = iap.Divide(iap.Deterministic(0), 1, elementwise=False)
    assert param.__str__() == param.__repr__() == "Divide(Deterministic(int 0), Deterministic(int 1), False)"


def test_parameters_Add():
    reseed()

    values_int = [-100, -54, -1, 0, 1, 54, 100]
    values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

    for v1 in values_int:
        for v2 in values_int:
            p = iap.Add(iap.Deterministic(v1), v2)
            assert p.draw_sample() == v1 + v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "i"
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 + v2)

            p = iap.Add(iap.Deterministic(v1), iap.Deterministic(v2))
            assert p.draw_sample() == v1 + v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "i"
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 + v2)

    for v1 in values_float:
        for v2 in values_float:
            p = iap.Add(iap.Deterministic(v1), v2)
            sample = p.draw_sample()
            assert v1 + v2 - _eps(sample) < sample < v1 + v2 + _eps(sample)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 + v2)

            p = iap.Add(iap.Deterministic(v1), iap.Deterministic(v2))
            assert v1 + v2 - _eps(sample) < sample < v1 + v2 + _eps(sample)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 + v2)

    param = iap.Add(iap.Deterministic(1.0), (1.0, 2.0), elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples >= 1.0 + 1.0 - _eps(samples))
    assert np.all(samples <= 1.0 + 2.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0])

    param = iap.Add(iap.Deterministic(1.0), (1.0, 2.0), elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples >= 1.0 + 1.0 - _eps(samples))
    assert np.all(samples <= 1.0 + 2.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Add(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples >= 1.0 + 1.0 - _eps(samples))
    assert np.all(samples <= 2.0 + 1.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Add(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples >= 1.0 + 1.0 - _eps(samples))
    assert np.all(samples <= 2.0 + 1.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Add(iap.Deterministic(0), 1, elementwise=False)
    assert param.__str__() == param.__repr__() == "Add(Deterministic(int 0), Deterministic(int 1), False)"


def test_parameters_Subtract():
    reseed()

    values_int = [-100, -54, -1, 0, 1, 54, 100]
    values_float = [-100.0, -54.3, -1.0, 0.1, 0.0, 0.1, 1.0, 54.4, 100.0]

    for v1 in values_int:
        for v2 in values_int:
            p = iap.Subtract(iap.Deterministic(v1), v2)
            assert p.draw_sample() == v1 - v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "i"
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 - v2)

            p = iap.Subtract(iap.Deterministic(v1), iap.Deterministic(v2))
            assert p.draw_sample() == v1 - v2
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "i"
            assert np.array_equal(samples, np.zeros((2, 3), dtype=np.int64) + v1 - v2)

    for v1 in values_float:
        for v2 in values_float:
            p = iap.Subtract(iap.Deterministic(v1), v2)
            sample = p.draw_sample()
            assert v1 - v2 - _eps(sample) < sample < v1 - v2 + _eps(sample)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 - v2)

            p = iap.Subtract(iap.Deterministic(v1), iap.Deterministic(v2))
            sample = p.draw_sample()
            assert v1 - v2 - _eps(sample) < sample < v1 - v2 + _eps(sample)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 - v2)

    param = iap.Subtract(iap.Deterministic(1.0), (1.0, 2.0), elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 - 2.0 - _eps(samples))
    assert np.all(samples < 1.0 - 1.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0])

    param = iap.Subtract(iap.Deterministic(1.0), (1.0, 2.0), elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 - 2.0 - _eps(samples))
    assert np.all(samples < 1.0 - 1.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Subtract(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 - 1.0 - _eps(samples))
    assert np.all(samples < 2.0 - 1.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Subtract(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 - 1.0 - _eps(samples))
    assert np.all(samples < 2.0 - 1.0 + _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Subtract(iap.Deterministic(0), 1, elementwise=False)
    assert param.__str__() == param.__repr__() == "Subtract(Deterministic(int 0), Deterministic(int 1), False)"


def test_parameters_Power():
    reseed()

    values = [-100, -54, -1, 0, 1, 54, 100]
    values = values + [float(v) for v in values]
    exponents = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    for v1 in values:
        for v2 in exponents:
            if v1 < 0 and ia.is_single_float(v2):
                continue
            if v1 == 0 and v2 < 0:
                continue
            p = iap.Power(iap.Deterministic(v1), v2)
            sample = p.draw_sample()
            assert v1 ** v2 - _eps(sample) < sample < v1 ** v2 + _eps(sample)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 ** v2)

            p = iap.Power(iap.Deterministic(v1), iap.Deterministic(v2))
            sample = p.draw_sample()
            assert v1 ** v2 - _eps(sample) < sample < v1 ** v2 + _eps(sample)
            samples = p.draw_samples((2, 3))
            assert samples.dtype.kind == "f"
            assert np.allclose(samples, np.zeros((2, 3), dtype=np.float64) + v1 ** v2)

    param = iap.Power(iap.Deterministic(1.5), (1.0, 2.0), elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.5 ** 1.0 - 2 * _eps(samples))
    assert np.all(samples < 1.5 ** 2.0 + 2 * _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0])

    param = iap.Power(iap.Deterministic(1.5), (1.0, 2.0), elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.5 ** 1.0 - 2 * _eps(samples))
    assert np.all(samples < 1.5 ** 2.0 + 2 * _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Power(iap.Uniform(1.0, 2.0), 1.0, elementwise=False)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 ** 1.0 - 2 * _eps(samples))
    assert np.all(samples < 2.0 ** 1.0 + 2 * _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Power(iap.Uniform(1.0, 2.0), 1.0, elementwise=True)
    samples = param.draw_samples((10, 20))
    assert samples.shape == (10, 20)
    assert np.all(samples > 1.0 ** 1.0 - 2 * _eps(samples))
    assert np.all(samples < 2.0 ** 1.0 + 2 * _eps(samples))
    samples_sorted = np.sort(samples.flatten())
    assert not (samples_sorted[0] - _eps(samples_sorted[0]) < samples_sorted[-1] < samples_sorted[0] + _eps(samples_sorted[0]))

    param = iap.Power(iap.Deterministic(0), 1, elementwise=False)
    assert param.__str__() == param.__repr__() == "Power(Deterministic(int 0), Deterministic(int 1), False)"


def test_parameters_Absolute():
    reseed()

    simple_values = [-1.5, -1, -1.0, -0.1, 0, 0.0, 0.1, 1, 1.0, 1.5]

    for value in simple_values:
        param = iap.Absolute(iap.Deterministic(value))
        sample = param.draw_sample()
        samples = param.draw_samples((10, 5))
        assert sample.shape == tuple()
        assert samples.shape == (10, 5)
        if ia.is_single_float(value):
            assert abs(value) - _eps(sample) < sample < abs(value) + _eps(sample)
            assert np.all(abs(value) - _eps(samples) < samples)
            assert np.all(samples < abs(value) + _eps(samples))
        else:
            assert sample == abs(value)
            assert np.all(samples == abs(value))

    param = iap.Absolute(iap.Choice([-3, -1, 1, 3]))
    sample = param.draw_sample()
    samples = param.draw_samples((10, 10))
    samples_uq = np.sort(np.unique(samples))
    assert sample.shape == tuple()
    assert sample in [3, 1]
    assert samples.shape == (10, 10)
    assert len(samples_uq) == 2
    assert samples_uq[0] == 1 and samples_uq[1] == 3

    param = iap.Absolute(iap.Deterministic(0))
    assert param.__str__() == param.__repr__() == "Absolute(Deterministic(int 0))"


def test_parameters_RandomSign():
    reseed()

    param = iap.RandomSign(iap.Deterministic(1))
    samples = param.draw_samples((1000,))
    n_positive = np.sum(samples == 1)
    n_negative = np.sum(samples == -1)
    assert samples.shape == (1000,)
    assert n_positive + n_negative == 1000
    assert 350 < n_positive < 750

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

    param = iap.RandomSign(iap.Choice([1, 2]))
    samples = param.draw_samples((4000,))
    seen = [0, 0, 0, 0]
    seen[0] = np.sum(samples == -2)
    seen[1] = np.sum(samples == -1)
    seen[2] = np.sum(samples == 1)
    seen[3] = np.sum(samples == 2)
    assert np.sum(seen) == 4000
    assert all([700 < v < 1300 for v in seen])

    param = iap.RandomSign(iap.Choice([1, 2]))
    samples1 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    assert samples1.shape == (100, 10)
    assert samples2.shape == (100, 10)
    assert np.array_equal(samples1, samples2)
    assert np.sum(samples == -2) > 50
    assert np.sum(samples == -1) > 50
    assert np.sum(samples == 1) > 50
    assert np.sum(samples == 2) > 50

    param = iap.RandomSign(iap.Deterministic(0), 0.5)
    assert param.__str__() == param.__repr__() == "RandomSign(Deterministic(int 0), 0.50)"


def test_parameters_ForceSign():
    reseed()

    param = iap.ForceSign(iap.Deterministic(1), positive=True, mode="invert")
    sample = param.draw_sample()
    assert sample.shape == tuple()
    assert sample == 1

    param = iap.ForceSign(iap.Deterministic(1), positive=False, mode="invert")
    sample = param.draw_sample()
    assert sample.shape == tuple()
    assert sample == -1

    param = iap.ForceSign(iap.Deterministic(1), positive=True, mode="invert")
    samples = param.draw_samples(100)
    assert samples.shape == (100,)
    assert np.all(samples == 1)

    param = iap.ForceSign(iap.Deterministic(1), positive=False, mode="invert")
    samples = param.draw_samples(100)
    assert samples.shape == (100,)
    assert np.all(samples == -1)

    param = iap.ForceSign(iap.Deterministic(-1), positive=True, mode="invert")
    samples = param.draw_samples(100)
    assert samples.shape == (100,)
    assert np.all(samples == 1)

    param = iap.ForceSign(iap.Deterministic(-1), positive=False, mode="invert")
    samples = param.draw_samples(100)
    assert samples.shape == (100,)
    assert np.all(samples == -1)

    param = iap.ForceSign(iap.Choice([-2, 1]), positive=True, mode="invert")
    samples = param.draw_samples(1000)
    assert samples.shape == (1000,)
    n_twos = np.sum(samples == 2)
    n_ones = np.sum(samples == 1)
    assert n_twos + n_ones == 1000
    assert 200 < n_twos < 700
    assert 200 < n_ones < 700

    param = iap.ForceSign(iap.Choice([-2, 1]), positive=True, mode="reroll")
    samples = param.draw_samples(1000)
    assert samples.shape == (1000,)
    n_twos = np.sum(samples == 2)
    n_ones = np.sum(samples == 1)
    assert n_twos + n_ones == 1000
    assert n_twos > 0
    assert n_ones > 0

    param = iap.ForceSign(iap.Choice([-2, 1]), positive=True, mode="reroll", reroll_count_max=100)
    samples = param.draw_samples(100)
    assert samples.shape == (100,)
    n_twos = np.sum(samples == 2)
    n_ones = np.sum(samples == 1)
    assert n_twos + n_ones == 100
    assert n_twos < 5

    param = iap.ForceSign(iap.Choice([-2, 1]), positive=True, mode="invert")
    samples1 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    assert samples1.shape == (100, 10)
    assert samples2.shape == (100, 10)
    assert np.array_equal(samples1, samples2)

    param = iap.ForceSign(iap.Deterministic(0), True, "invert", 1)
    assert param.__str__() == param.__repr__() == "ForceSign(Deterministic(int 0), True, invert, 1)"


def test_parameters_Positive():
    reseed()

    param = iap.Positive(iap.Deterministic(-1), mode="reroll", reroll_count_max=1)
    samples = param.draw_samples((100,))
    assert samples.shape == (100,)
    assert np.all(samples == 1)


def test_parameters_Negative():
    reseed()

    param = iap.Negative(iap.Deterministic(1), mode="reroll", reroll_count_max=1)
    samples = param.draw_samples((100,))
    assert samples.shape == (100,)
    assert np.all(samples == -1)


def test_parameters_IterativeNoiseAggregator():
    reseed()

    param = iap.IterativeNoiseAggregator(iap.Deterministic(1), iterations=1, aggregation_method="max")
    sample = param.draw_sample()
    samples = param.draw_samples((2, 4))
    assert sample.shape == tuple()
    assert samples.shape == (2, 4)
    assert sample == 1
    assert np.all(samples == 1)

    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=200, aggregation_method="avg")
    sample = param.draw_sample()
    samples = param.draw_samples((2, 4))
    assert sample.shape == tuple()
    assert samples.shape == (2, 4)
    assert 25 - 10 < sample < 25 + 10
    assert np.all(np.logical_and(25 - 10 < samples, samples < 25 + 10))

    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=100, aggregation_method="max")
    sample = param.draw_sample()
    samples = param.draw_samples((2, 4))
    assert sample.shape == tuple()
    assert samples.shape == (2, 4)
    assert sample == 50
    assert np.all(samples == 50)

    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=100, aggregation_method="min")
    sample = param.draw_sample()
    samples = param.draw_samples((2, 4))
    assert sample.shape == tuple()
    assert samples.shape == (2, 4)
    assert sample == 0
    assert np.all(samples == 0)

    seen = [0, 0, 0]
    for _ in sm.xrange(100):
        param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=100, aggregation_method=["avg", "max"])
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
            assert False
    assert seen[2] < 5
    assert 50 - 20 < seen[0] < 50 + 20
    assert 50 - 20 < seen[1] < 50 + 20

    # iterations as tuple
    param = iap.IterativeNoiseAggregator(iap.Uniform(-1.0, 1.0), iterations=(1, 100), aggregation_method="avg")
    diffs = []
    for _ in sm.xrange(100):
        samples = param.draw_samples((1, 1))
        diff = abs(samples[0, 0] - 0.0)
        diffs.append(diff)

    nb_bins = 3
    hist, _ = np.histogram(diffs, bins=nb_bins, range=(-1.0, 1.0), density=False)
    assert hist[1] > hist[0]
    assert hist[1] > hist[2]

    # iterations as list
    seen = [0, 0]
    for _ in sm.xrange(400):
        param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=[1, 100], aggregation_method=["max"])
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

    # test ia.ALL as aggregation_method
    # note that each method individually and list of methods are already tested, so no in depth
    # test is needed here
    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=100, aggregation_method=ia.ALL)
    assert isinstance(param.aggregation_method, iap.Choice)
    assert len(param.aggregation_method.a) == 3
    assert [v in param.aggregation_method.a for v in ["min", "avg", "max"]]

    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=2, aggregation_method="max")
    samples = param.draw_samples((2, 1000))
    nb_0 = np.sum(samples == 0)
    nb_50 = np.sum(samples == 50)
    assert nb_0 + nb_50 == 2 * 1000
    assert 0.25 - 0.05 < nb_0 / (2 * 1000) < 0.25 + 0.05

    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=5, aggregation_method="avg")
    samples1 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    assert samples1.shape == (100, 10)
    assert samples2.shape == (100, 10)
    assert np.allclose(samples1, samples2)

    # StochasticParameter as aggregation_method
    param = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=5, aggregation_method=iap.Deterministic("max"))
    assert isinstance(param.aggregation_method, iap.Deterministic)
    assert param.aggregation_method.value == "max"

    # bad datatype as aggregation_method
    got_exception = False
    try:
        _ = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=5, aggregation_method=False)
    except Exception as exc:
        assert "Expected aggregation_method to be" in str(exc)
        got_exception = True
    assert got_exception

    # bad datatype as for iterations
    got_exception = False
    try:
        _ = iap.IterativeNoiseAggregator(iap.Choice([0, 50]), iterations=False, aggregation_method="max")
    except Exception as exc:
        assert "Expected iterations to be" in str(exc)
        got_exception = True
    assert got_exception

    param = iap.IterativeNoiseAggregator(iap.Deterministic(0), iterations=(1, 3), aggregation_method="max")
    assert param.__str__() == param.__repr__() == "IterativeNoiseAggregator(Deterministic(int 0), DiscreteUniform(Deterministic(int 1), Deterministic(int 3)), Deterministic(max))"


def test_parameters_Sigmoid():
    reseed()

    param = iap.Sigmoid(iap.Deterministic(5), add=0, mul=1, threshold=0.5, activated=True)
    expected = 1 / (1 + np.exp(-(5 * 1 + 0 - 0.5)))
    sample = param.draw_sample()
    samples = param.draw_samples((5, 10))
    assert sample.shape == tuple()
    assert samples.shape == (5, 10)
    assert expected - _eps(sample) < sample < expected + _eps(sample)
    assert np.all(np.logical_and(expected - _eps(samples) < samples, samples < expected + _eps(samples)))

    param = iap.Sigmoid(iap.Deterministic(5), add=0, mul=1, threshold=0.5, activated=False)
    expected = 5
    sample = param.draw_sample()
    samples = param.draw_samples((5, 10))
    assert sample.shape == tuple()
    assert samples.shape == (5, 10)
    assert expected - _eps(sample) < sample < expected + _eps(sample)
    assert np.all(np.logical_and(expected - _eps(sample) < samples, samples < expected + _eps(sample)))

    param = iap.Sigmoid(iap.Deterministic(5), add=0, mul=1, threshold=0.5, activated=0.5)
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

    param = iap.Sigmoid(iap.Choice([1, 10]), add=0, mul=1, threshold=0.5, activated=True)
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

    muls = [0.1, 1, 10.3]
    adds = [-5.7, -1, -0.0734, 0, 0.0734, 1, 5.7]
    vals = [-1, -0.7, 0, 0.7, 1]
    threshs = [-5.7, -1, -0.0734, 0, 0.0734, 1, 5.7]
    for mul in muls:
        for add in adds:
            for val in vals:
                for thresh in threshs:
                    param = iap.Sigmoid(iap.Deterministic(val), add=add, mul=mul, threshold=thresh)
                    sample = param.draw_sample()
                    samples = param.draw_samples((2, 3))
                    assert sample.shape == tuple()
                    assert samples.shape == (2, 3)
                    dt = sample.dtype
                    val_ = np.array([val], dtype=dt)
                    mul_ = np.array([mul], dtype=dt)
                    add_ = np.array([add], dtype=dt)
                    thresh_ = np.array([thresh], dtype=dt)
                    expected = 1 / (1 + np.exp(-(val_ * mul_ + add_ - thresh_)))
                    assert expected - 5*_eps(sample) < sample < expected + 5*_eps(sample)
                    assert np.all(np.logical_and(expected - 5*_eps(sample) < samples, samples < expected + 5*_eps(sample)))

    param = iap.Sigmoid(iap.Choice([1, 10]), add=0, mul=1, threshold=0.5, activated=True)
    samples1 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    samples2 = param.draw_samples((100, 10), random_state=np.random.RandomState(1234))
    assert samples1.shape == (100, 10)
    assert samples2.shape == (100, 10)
    assert np.array_equal(samples1, samples2)

    param = iap.Sigmoid(iap.Deterministic(0), threshold=(-10, 10), activated=True, mul=1, add=0)
    assert param.__str__() == param.__repr__() == \
        "Sigmoid(Deterministic(int 0), Uniform(Deterministic(int -10), Deterministic(int 10)), " \
        + "Deterministic(int 1), 1, 0)"


def test_parameters_operators():
    reseed()

    param1 = iap.Normal(0, 1)
    param2 = iap.Uniform(-1.0, 1.0)

    # Multiply
    param3 = param1 * param2
    assert isinstance(param3, iap.Multiply)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1 * 2
    assert isinstance(param3, iap.Multiply)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    param3 = 2 * param1
    assert isinstance(param3, iap.Multiply)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        _ = "test" * param1
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = param1 * "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Divide (__truediv__)
    param3 = param1 / param2
    assert isinstance(param3, iap.Divide)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1 / 2
    assert isinstance(param3, iap.Divide)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    param3 = 2 / param1
    assert isinstance(param3, iap.Divide)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        _ = "test" / param1
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = param1 / "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Divide (__div__)
    param3 = param1.__div__(param2)
    assert isinstance(param3, iap.Divide)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1.__div__(2)
    assert isinstance(param3, iap.Divide)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    got_exception = False
    try:
        _ = param1.__div__("test")
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Divide (__rdiv__)
    param3 = param1.__rdiv__(2)
    assert isinstance(param3, iap.Divide)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        _ = param1.__rdiv__("test")
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Divide (__floordiv__)
    param1_int = iap.DiscreteUniform(0, 10)
    param2_int = iap.Choice([1, 2])
    param3 = param1_int // param2_int
    assert isinstance(param3, iap.Discretize)
    assert isinstance(param3.other_param, iap.Divide)
    assert param3.other_param.other_param == param1_int
    assert param3.other_param.val == param2_int

    param3 = param1_int // 2
    assert isinstance(param3, iap.Discretize)
    assert isinstance(param3.other_param, iap.Divide)
    assert param3.other_param.other_param == param1_int
    assert isinstance(param3.other_param.val, iap.Deterministic)
    assert param3.other_param.val.value == 2

    param3 = 2 // param1_int
    assert isinstance(param3, iap.Discretize)
    assert isinstance(param3.other_param, iap.Divide)
    assert isinstance(param3.other_param.other_param, iap.Deterministic)
    assert param3.other_param.other_param.value == 2
    assert param3.other_param.val == param1_int

    got_exception = False
    try:
        _ = "test" // param1_int
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = param1_int // "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Add
    param3 = param1 + param2
    assert isinstance(param3, iap.Add)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1 + 2
    assert isinstance(param3, iap.Add)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    param3 = 2 + param1
    assert isinstance(param3, iap.Add)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        _ = "test" + param1
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = param1 + "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Subtract
    param3 = param1 - param2
    assert isinstance(param3, iap.Subtract)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1 - 2
    assert isinstance(param3, iap.Subtract)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    param3 = 2 - param1
    assert isinstance(param3, iap.Subtract)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        _ = "test" - param1
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = param1 - "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    # Power
    param3 = param1 ** param2
    assert isinstance(param3, iap.Power)
    assert param3.other_param == param1
    assert param3.val == param2

    param3 = param1 ** 2
    assert isinstance(param3, iap.Power)
    assert param3.other_param == param1
    assert isinstance(param3.val, iap.Deterministic)
    assert param3.val.value == 2

    param3 = 2 ** param1
    assert isinstance(param3, iap.Power)
    assert isinstance(param3.other_param, iap.Deterministic)
    assert param3.other_param.value == 2
    assert param3.val == param1

    got_exception = False
    try:
        _ = "test" ** param1
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = param1 ** "test"
    except Exception as exc:
        assert "Invalid datatypes" in str(exc)
        got_exception = True
    assert got_exception


def test_parameters_copy():
    reseed()
    other_param = iap.Uniform(1.0, 10.0)
    param = iap.Discretize(other_param)
    other_param.a = [1.0]
    param_copy = param.copy()
    assert isinstance(param_copy, iap.Discretize)
    assert isinstance(param_copy.other_param, iap.Uniform)
    param.other_param.a[0] += 1
    assert param_copy.other_param.a[0] == param.other_param.a[0]

    other_param = iap.Uniform(1.0, 10.0)
    param = iap.Discretize(other_param)
    other_param.a = [1.0]
    param_copy = param.deepcopy()
    assert isinstance(param_copy, iap.Discretize)
    assert isinstance(param_copy.other_param, iap.Uniform)
    param.other_param.a[0] += 1
    assert param_copy.other_param.a[0] != param.other_param.a[0]


if __name__ == "__main__":
    main()
