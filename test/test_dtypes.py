from __future__ import print_function, division, absolute_import

import warnings
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

import numpy as np

from imgaug import dtypes as iadt


class Test_normalize_dtypes(unittest.TestCase):
    @mock.patch("imgaug.dtypes.normalize_dtype")
    def test_single_non_list(self, mock_nd):
        mock_nd.return_value = "foo"
        dtypes = iadt.normalize_dtypes("int16")
        assert dtypes == ["foo"]
        assert mock_nd.call_count == 1
        assert mock_nd.call_args_list[0][0][0] == "int16"

    def test_single_dtype(self):
        dtypes = iadt.normalize_dtypes(np.dtype("int16"))
        assert isinstance(dtypes, list)
        assert len(dtypes) == 1
        assert isinstance(dtypes[0], np.dtype)
        assert dtypes[0].name == "int16"

    def test_empty_list(self):
        dtypes = iadt.normalize_dtypes([])
        assert isinstance(dtypes, list)
        assert len(dtypes) == 0

    @mock.patch("imgaug.dtypes.normalize_dtype")
    def test_list_of_dtype_names(self, mock_nd):
        mock_nd.return_value = "foo"
        dtypes = iadt.normalize_dtypes(["int16", "int32"])
        assert dtypes == ["foo", "foo"]
        assert mock_nd.call_count == 2
        assert mock_nd.call_args_list[0][0][0] == "int16"
        assert mock_nd.call_args_list[1][0][0] == "int32"


class Test_normalize_dtype(unittest.TestCase):
    def test_dtype(self):
        dtype = iadt.normalize_dtype(np.dtype("int16"))
        assert isinstance(dtype, np.dtype)
        assert dtype.name == "int16"

    def test_dtype_name(self):
        dtype = iadt.normalize_dtype("int16")
        assert isinstance(dtype, np.dtype)
        assert dtype.name == "int16"

    def test_dtype_name_short(self):
        dtype = iadt.normalize_dtype("i2")
        assert isinstance(dtype, np.dtype)
        assert dtype.name == "int16"

    def test_dtype_function(self):
        dtype = iadt.normalize_dtype(np.int16)
        assert isinstance(dtype, np.dtype)
        assert dtype.name == "int16"

    def test_ndarray(self):
        arr = np.zeros((1,), dtype=np.int16)
        dtype = iadt.normalize_dtype(arr)
        assert isinstance(dtype, np.dtype)
        assert dtype.name == "int16"

    def test_numpy_scalar(self):
        scalar = np.int16(0)
        dtype = iadt.normalize_dtype(scalar)
        assert isinstance(dtype, np.dtype)
        assert dtype.name == "int16"


# change_dtype_() is already indirectly tested via Test_change_dtypes_(),
# so were don't have to be very thorough here
class Test_change_dtype_(unittest.TestCase):
    def test_no_clip_no_round(self):
        arr = np.array([[0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0]],
                       dtype=np.float32)
        dtype = np.int8

        observed = iadt.change_dtype_(np.copy(arr), dtype,
                                      clip=False, round=False)

        expected = np.array([[0, 0, 0, -128+1-1, 127-1+1]], dtype=np.int8)
        assert observed.dtype.name == "int8"
        assert np.array_equal(observed, expected)

    def test_clip_and_round(self):
        arr = np.array([[0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0]],
                       dtype=np.float32)
        dtype = np.int8

        observed = iadt.restore_dtypes_(np.copy(arr), dtype)

        expected = np.array([[0, 0, 1, 127, -128]], dtype=np.int8)
        assert observed.dtype.name == "int8"
        assert np.array_equal(observed, expected)

    def test_dtype_not_changed(self):
        arr = np.array([[-128, -1, 0, 1, 127]], dtype=np.int8)
        dtype = np.int8

        observed = iadt.restore_dtypes_(arr, dtype,
                                        clip=False, round=False)

        assert observed is arr

    @mock.patch('numpy.round')
    def test_no_round_if_dtype_not_changed(self, mock_round):
        arr = np.array([[-128, -1, 0, 1, 127]], dtype=np.int8)
        dtype = np.int8

        observed = iadt.restore_dtypes_(arr, dtype, clip=False)

        assert observed is arr
        assert mock_round.call_count == 0

    def test_round_float_dtypes(self):
        arr = np.array([[-128, -1.1, 0.7, 1.1, 127]], dtype=np.float32)
        dtype = np.int8

        observed = iadt.restore_dtypes_(np.copy(arr), dtype, clip=False)

        expected = np.array([[-128, -1, 1, 1, 127]], dtype=np.int8)
        assert observed.dtype.name == "int8"
        assert np.array_equal(observed, expected)

    @mock.patch('numpy.round')
    def test_dont_round_non_float_dtypes(self, mock_round):
        arr = np.array([[-128, -1, 0, 1, 127]], dtype=np.int8)
        dtype = np.float32

        _ = iadt.restore_dtypes_(np.copy(arr), dtype, clip=False)

        assert mock_round.call_count == 0

    def test_int16_to_int8(self):
        arr = np.zeros((1,), dtype=np.int16) + 1
        observed = iadt.change_dtype_(arr, np.int8, clip=False, round=False)
        assert observed.shape == (1,)
        assert observed.dtype.name == "int8"
        assert np.all(observed == 1)

    def test_int16_to_int8_with_overflow(self):
        arr = np.zeros((1,), dtype=np.int16) + 128
        observed = iadt.change_dtype_(arr, np.int8, clip=False, round=False)
        assert observed.shape == (1,)
        assert observed.dtype.name == "int8"
        assert np.all(observed == -128)

    def test_float32_to_int8(self):
        arr = np.zeros((1,), dtype=np.int32) + 1
        observed = iadt.change_dtype_(arr, np.int8, clip=False, round=False)
        assert observed.shape == (1,)
        assert observed.dtype.name == "int8"
        assert np.all(observed == 1)

    def test_float32_to_int8_with_overflow(self):
        arr = np.zeros((1,), dtype=np.int32) + 1
        observed = iadt.change_dtype_(arr, np.int8, clip=False, round=False)
        assert observed.shape == (1,)
        assert observed.dtype.name == "int8"
        assert np.all(observed == 1)

    def test_dtype_given_as_string(self):
        arr = np.zeros((1,), dtype=np.int8) + 1
        observed = iadt.change_dtype_(arr, "int16", clip=False, round=False)
        assert observed.shape == (1,)
        assert observed.dtype.name == "int16"
        assert np.all(observed == 1)


class Test_change_dtypes_(unittest.TestCase):
    def test_array_input_single_dtype_no_clip_no_round(self):
        arr = np.array([[0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0]],
                       dtype=np.float32)
        dtype = np.int8

        observed = iadt.restore_dtypes_(np.copy(arr), dtype,
                                        clip=False, round=False)

        expected = np.array([[0, 0, 0, -128+1-1, 127-1+1]], dtype=np.int8)
        assert observed.dtype.name == "int8"
        assert np.array_equal(observed, expected)

    def test_array_input_single_dtype_with_clip_no_round(self):
        arr = np.array([[0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0]],
                       dtype=np.float32)
        dtype = np.int8

        observed = iadt.restore_dtypes_(np.copy(arr), dtype,
                                        clip=True, round=False)

        expected = np.array([[0, 0, 0, 127, -128]], dtype=np.int8)
        assert observed.dtype.name == "int8"
        assert np.array_equal(observed, expected)

    def test_array_input_single_dtype_no_clip_with_round(self):
        arr = np.array([[0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0]],
                       dtype=np.float32)
        dtype = np.int8

        observed = iadt.restore_dtypes_(np.copy(arr), dtype,
                                        clip=False, round=True)

        expected = np.array([[0, 0, 1, -128+1-1, 127-1+1]], dtype=np.int8)
        assert observed.dtype.name == "int8"
        assert np.array_equal(observed, expected)

    def test_array_input_fail_if_many_different_dtypes(self):
        arr = np.array([
            [0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0],
            [0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0],
        ], dtype=np.float32)
        dtypes = [np.int8, np.int16]

        with self.assertRaises(AssertionError) as context:
            _observed = iadt.restore_dtypes_(np.copy(arr), dtypes,
                                             clip=False, round=False)

        assert (
            "or an iterable of N times the *same* dtype"
            in str(context.exception)
        )

    def test_array_input_many_dtypes_no_clip_no_round(self):
        arr = np.array([
            [0.0, 0.1, 0.9, 127.0+0.0, -128.0-0.0],
            [0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0],
        ], dtype=np.float32)
        dtypes = [np.int8, np.int8]

        observed = iadt.restore_dtypes_(np.copy(arr), dtypes,
                                        clip=False, round=False)

        expected = np.array([
            [0, 0, 0, 127, -128],
            [0, 0, 0, -128+1-1, 127-1+1]
        ], dtype=np.int8)
        assert observed.dtype.name == "int8"
        assert np.array_equal(observed, expected)

    def test_empty_array_input(self):
        arr = np.zeros((0, 5), dtype=np.float32)
        dtypes = np.int8

        observed = iadt.restore_dtypes_(np.copy(arr), dtypes,
                                        clip=False, round=False)

        assert observed.dtype.name == "int8"
        assert observed.shape == (0, 5)

    def test_empty_list_input(self):
        arrs = []
        dtypes = np.int8

        observed = iadt.restore_dtypes_(arrs, dtypes,
                                        clip=False, round=False)

        assert len(observed) == 0

    def test_many_items_list_input_single_dtype(self):
        arrs = [
            np.array([0.0, 0.1, 0.9, 127.0+0.0, -128.0-0.0], dtype=np.float32),
            np.array([0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0], dtype=np.float32)
        ]
        dtypes = np.int8

        observed = iadt.restore_dtypes_(
            [np.copy(arr) for arr in arrs],
            dtypes,
            clip=False,
            round=False)

        expected = [
            np.array([0, 0, 0, 127, -128], dtype=np.int8),
            np.array([0, 0, 0, -128+1-1, 127-1+1], dtype=np.int8)
        ]
        assert len(observed) == 2
        assert observed[0].dtype.name == "int8"
        assert observed[1].dtype.name == "int8"
        assert np.array_equal(observed[0], expected[0])
        assert np.array_equal(observed[1], expected[1])

    def test_many_items_list_input_many_dtypes(self):
        arrs = [
            np.array([0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0], dtype=np.float32),
            np.array([0.0, 0.1, 0.9, 127.0+1.0, -128.0-1.0], dtype=np.float32)
        ]
        dtypes = [np.int8, np.int16]

        observed = iadt.restore_dtypes_(
            [np.copy(arr) for arr in arrs],
            dtypes,
            clip=False,
            round=False)

        expected = [
            np.array([0, 0, 0, -128+1-1, 127-1+1], dtype=np.int8),
            np.array([0, 0, 0, 127+1, -128-1], dtype=np.int16)
        ]
        assert len(observed) == 2
        assert observed[0].dtype.name == "int8"
        assert observed[1].dtype.name == "int16"
        assert np.array_equal(observed[0], expected[0])
        assert np.array_equal(observed[1], expected[1])

    def test_invalid_input(self):
        arr = False

        with self.assertRaises(Exception) as context:
            _ = iadt.restore_dtypes_(arr, np.int8)

        assert "Expected numpy array or " in str(context.exception)

    def test_int_to_float(self):
        arr = np.array([[-100, -1, 0, 1, 100]], dtype=np.int8)
        dtype = np.float32

        observed = iadt.restore_dtypes_(np.copy(arr), dtype,
                                        clip=False, round=False)

        expected = np.array([[-100.0, -1.0, 0.0, 1.0, 100.0]],
                            dtype=np.float32)
        assert observed.dtype.name == "float32"
        assert np.allclose(observed, expected)

    def test_increase_float_resolution(self):
        arr = np.array([[-100.0, -1.0, 0.0, 1.0, 100.0]], dtype=np.float32)
        dtype = np.float64

        observed = iadt.restore_dtypes_(np.copy(arr), dtype,
                                        clip=False, round=False)

        expected = np.array([[-100.0, -1.0, 0.0, 1.0, 100.0]],
                            dtype=np.float32)
        assert observed.dtype.name == "float64"
        assert np.allclose(observed, expected)

    def test_int_to_uint(self):
        arr = np.array([[-100, -1, 0, 1, 100]], dtype=np.int8)
        dtype = np.uint8

        observed = iadt.restore_dtypes_(np.copy(arr), dtype,
                                        clip=False, round=False)

        expected = np.array([[255-100+1, 255-1+1, 0, 1, 100]],
                            dtype=np.uint8)
        assert observed.dtype.name == "uint8"
        assert np.allclose(observed, expected)

    def test_int_to_uint_with_clip(self):
        arr = np.array([[-100, -1, 0, 1, 100]], dtype=np.int8)
        dtype = np.uint8

        observed = iadt.restore_dtypes_(np.copy(arr), dtype,
                                        clip=True, round=False)

        expected = np.array([[0, 0, 0, 1, 100]], dtype=np.uint8)
        assert observed.dtype.name == "uint8"
        assert np.allclose(observed, expected)


# TODO is the copy_* function still used anywhere
class Test_copy_dtypes_for_restore(unittest.TestCase):
    def test_images_as_list(self):
        # TODO using dtype=np.bool is causing this to fail as it ends up
        #      being <type bool> instead of <type 'numpy.bool_'>.
        #      Any problems from that for the library?
        images = [
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((10, 16, 3), dtype=np.float32),
            np.zeros((20, 10, 6), dtype=np.int32)
        ]

        dtypes_copy = iadt.copy_dtypes_for_restore(images, force_list=False)
        assert np.all([
            dtype_observed.name == dtype_expected
            for dtype_observed, dtype_expected
            in zip(
                dtypes_copy,
                ["uint8", "float32", "int32"]
            )
        ])

    def test_images_as_single_array(self):
        dts = ["uint8", "float32", "int32"]
        for dt in dts:
            with self.subTest(dtype=dt):
                images = np.zeros((10, 16, 32, 3), dtype=dt)
                dtypes_copy = iadt.copy_dtypes_for_restore(images)
                assert isinstance(dtypes_copy, np.dtype)
                assert dtypes_copy.name == dt

    def test_images_as_single_array_force_list(self):
        dts = ["uint8", "float32", "int32"]
        for dt in dts:
            with self.subTest(dtype=dt):
                images = np.zeros((10, 16, 32, 3), dtype=dt)
                dtypes_copy = iadt.copy_dtypes_for_restore(images,
                                                           force_list=True)
                assert isinstance(dtypes_copy, list)
                assert np.all([dtype_i.name == dt for dtype_i in dtypes_copy])


class Test_increase_itemsize_of_dtype(unittest.TestCase):
    def test_factor_is_1(self):
        dts = [
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64
        ]
        for dt in dts:
            dt = np.dtype(dt)
            with self.subTest(dtype=dt.name):
                dt_increased = iadt.increase_itemsize_of_dtype(dt, 1)
                assert dt_increased.name == dt.name

    def test_factor_is_2(self):
        dts = [
            np.int8, np.int16, np.int32,
            np.uint8, np.uint16, np.uint32,
            np.float16, np.float32
        ]
        expecteds = [
            np.int16, np.int32, np.int64,
            np.uint16, np.uint32, np.uint64,
            np.float32, np.float64
        ]
        for dt, expected in zip(dts, expecteds):
            dt = np.dtype(dt)
            expected = np.dtype(expected)
            with self.subTest(dtype=dt.name):
                dt_increased = iadt.increase_itemsize_of_dtype(dt, 2)
                assert dt_increased.name == expected.name

    def test_dtype_as_string(self):
        dt_names = [
            "int8", "int16", "int32",
            "uint8", "uint16", "uint32",
            "float16", "float32"
        ]
        expecteds = [
            np.int16, np.int32, np.int64,
            np.uint16, np.uint32, np.uint64,
            np.float32, np.float64
        ]
        for dt_name, expected in zip(dt_names, expecteds):
            expected = np.dtype(expected)
            with self.subTest(dtype=dt_name):
                dt_increased = iadt.increase_itemsize_of_dtype(dt_name, 2)
                assert dt_increased.name == expected.name

    def test_unknown_dtype(self):
        with self.assertRaises(TypeError) as context:
            _ = iadt.increase_itemsize_of_dtype(np.uint64, 2)

        assert (
            "Unable to create a numpy dtype matching"
            in str(context.exception))


class Test_get_minimal_dtype(unittest.TestCase):
    def test_with_dtype_function(self):
        dt_funcs = [
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64,
            np.bool_
        ]

        for dt_func in dt_funcs:
            with self.subTest(dtype=np.dtype(dt_func).name):
                inputs = [dt_func]
                promoted_dt = iadt.get_minimal_dtype(inputs)
                assert promoted_dt.name == np.dtype(dt_func).name

    def test_with_lists_of_identical_dtypes(self):
        dts = [
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64,
            np.bool_
        ]

        for dt in dts:
            dt = np.dtype(dt)
            for length in [1, 2, 3]:
                with self.subTest(dtype=dt.name, length=length):
                    inputs = [dt for _ in range(length)]
                    promoted_dt = iadt.get_minimal_dtype(inputs)
                    assert promoted_dt.name == dt.name

    def test_with_lists_of_identical_dtype_arrays(self):
        dts = [
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64,
            np.bool_
        ]

        for dt in dts:
            dt = np.dtype(dt)
            for length in [1, 2, 3]:
                with self.subTest(dtype=dt.name, length=length):
                    inputs = [np.zeros((1, 1, 3), dtype=dt)
                              for _ in range(length)]
                    promoted_dt = iadt.get_minimal_dtype(inputs)
                    assert promoted_dt.name == dt.name

    def test_with_lists_of_different_arrays(self):
        dt_lists = [
            [np.uint8, np.uint16],
            [np.uint8, np.uint32],
            [np.uint8, np.int8],
            [np.uint8, np.bool_],
            [np.int8, np.int16],
            [np.float16, np.float32],
            [np.uint8, np.float32],
            [np.uint8, np.int8, np.int16],
            [np.uint8, np.int8, np.bool_],
            [np.uint8, np.int8, np.float32],
        ]
        expecteds = [
            np.uint16,
            np.uint32,
            np.int16,
            np.uint8,
            np.int16,
            np.float32,
            np.float32,
            np.int16,
            np.int16,
            np.float32
        ]
        for dt_list, expected in zip(dt_lists, expecteds):
            expected = np.dtype(expected)
            dt_list = [np.dtype(dt) for dt in dt_list]
            dt_names = ", ".join([dt.name for dt in dt_list])
            with self.subTest(dtypes=dt_names):
                promoted_dt = iadt.get_minimal_dtype(dt_list)
                assert promoted_dt.name == expected.name

    @mock.patch("imgaug.dtypes.increase_itemsize_of_dtype")
    def test_calls_increase_itemsize_factor(self, mock_iibf):
        dt = np.int8
        factor = 2

        _ = iadt.get_minimal_dtype([dt], factor)

        assert mock_iibf.call_count == 1


class Test_promote_array_dtypes_(unittest.TestCase):
    @mock.patch("imgaug.dtypes.get_minimal_dtype")
    @mock.patch("imgaug.dtypes.change_dtypes_")
    def test_calls_subfunctions(self, mock_cd, mock_gmd):
        mock_gmd.return_value = np.dtype("int16")
        mock_cd.return_value = "foo"
        arrays = [np.zeros((1,), dtype=np.int8)]

        observed = iadt.promote_array_dtypes_(arrays)

        assert mock_gmd.call_count == 1
        assert mock_cd.call_count == 1
        # call 0, args, arg 0, dtype 0
        assert mock_gmd.call_args_list[0][0][0][0].name == "int8"
        assert mock_gmd.call_args_list[0][1]["increase_itemsize_factor"] == 1
        assert mock_cd.call_args_list[0][0][0] is arrays
        assert observed == "foo"

    @mock.patch("imgaug.dtypes.get_minimal_dtype")
    @mock.patch("imgaug.dtypes.change_dtypes_")
    def test_calls_subfunctions_dtypes_set(self, mock_cd, mock_gmd):
        mock_gmd.return_value = np.dtype("int16")
        mock_cd.return_value = "foo"
        arrays = [np.zeros((1,), dtype=np.int8)]

        observed = iadt.promote_array_dtypes_(
            arrays,
            dtypes=["float32"])

        assert mock_gmd.call_count == 1
        assert mock_cd.call_count == 1
        # call 0, args, arg 0, dtype 0
        assert mock_gmd.call_args_list[0][0][0][0] == "float32"
        assert mock_gmd.call_args_list[0][1]["increase_itemsize_factor"] == 1
        assert mock_cd.call_args_list[0][0][0] is arrays
        assert observed == "foo"

    @mock.patch("imgaug.dtypes.get_minimal_dtype")
    @mock.patch("imgaug.dtypes.change_dtypes_")
    def test_calls_subfunctions_increase_itemsize_factor_set(self, mock_cd,
                                                             mock_gmd):
        mock_gmd.return_value = np.dtype("int16")
        mock_cd.return_value = "foo"
        arrays = [np.zeros((1,), dtype=np.int8)]

        observed = iadt.promote_array_dtypes_(
            arrays,
            increase_itemsize_factor=2)

        assert mock_gmd.call_count == 1
        assert mock_cd.call_count == 1
        # call 0, args, arg 0, dtype 0
        assert mock_gmd.call_args_list[0][0][0][0].name == "int8"
        assert mock_gmd.call_args_list[0][1]["increase_itemsize_factor"] == 2
        assert mock_cd.call_args_list[0][0][0] is arrays
        assert observed == "foo"

    def test_promote_single_array(self):
        arr = np.zeros((1,), dtype=np.int8)
        observed = iadt.promote_array_dtypes_(arr)
        assert observed.dtype.name == "int8"

    def test_promote_single_array_single_dtype_set(self):
        arr = np.zeros((1,), dtype=np.int8)
        observed = iadt.promote_array_dtypes_(arr, np.int16)
        assert observed.dtype.name == "int16"

    def test_promote_single_array_dtypes_set(self):
        arr = np.zeros((1,), dtype=np.int8)
        observed = iadt.promote_array_dtypes_(arr, [np.int16])
        assert observed.dtype.name == "int16"

    def test_promote_single_array_increase_itemsize_factor_set(self):
        arr = np.zeros((1,), dtype=np.int8)
        observed = iadt.promote_array_dtypes_(arr, increase_itemsize_factor=2)
        assert observed.dtype.name == "int16"

    def test_promote_list_of_single_array(self):
        arrays = [np.zeros((1,), dtype=np.int8)]
        observed = iadt.promote_array_dtypes_(arrays,
                                              increase_itemsize_factor=2)
        assert observed[0].dtype.name == "int16"

    def test_promote_list_of_two_arrays(self):
        arrays = [np.zeros((1,), dtype=np.int8),
                  np.zeros((1,), dtype=np.int16)]
        observed = iadt.promote_array_dtypes_(arrays,
                                              increase_itemsize_factor=2)
        assert observed[0].dtype.name == "int32"
        assert observed[1].dtype.name == "int32"

    def test_promote_list_of_two_arrays_dtypes_set(self):
        arrays = [np.zeros((1,), dtype=np.int8),
                  np.zeros((1,), dtype=np.int16)]
        observed = iadt.promote_array_dtypes_(arrays,
                                              dtypes=[np.float32, np.float64])
        assert observed[0].dtype.name == "float64"
        assert observed[1].dtype.name == "float64"

    def test_promote_list_of_three_arrays(self):
        arrays = [np.zeros((1,), dtype=np.int8),
                  np.zeros((1,), dtype=np.int16),
                  np.zeros((1,), dtype=np.uint8)]
        observed = iadt.promote_array_dtypes_(arrays,
                                              increase_itemsize_factor=2)
        assert observed[0].dtype.name == "int32"
        assert observed[1].dtype.name == "int32"
        assert observed[2].dtype.name == "int32"


class Test_increase_array_resolutions_(unittest.TestCase):
    def test_single_array_factor_1(self):
        arr = np.zeros((1,), dtype=np.int8)
        observed = iadt.increase_array_resolutions_(arr, 1)
        assert observed.dtype.name == "int8"

    def test_single_array_factor_2(self):
        arr = np.zeros((1,), dtype=np.int8)
        observed = iadt.increase_array_resolutions_(arr, 2)
        assert observed.dtype.name == "int16"

    def test_list_of_one_array(self):
        arr = np.zeros((1,), dtype=np.int8)
        observed = iadt.increase_array_resolutions_([arr], 2)
        assert observed[0].dtype.name == "int16"

    def test_list_of_two_arrays(self):
        arrays = [
            np.zeros((1,), dtype=np.int8),
            np.zeros((1,), dtype=np.int16)
        ]
        observed = iadt.increase_array_resolutions_(arrays, 2)
        assert observed[0].dtype.name == "int16"
        assert observed[1].dtype.name == "int32"


class Test_get_value_range_of_dtype(unittest.TestCase):
    def test_bool(self):
        minv, center, maxv = iadt.get_value_range_of_dtype(np.dtype(bool))
        assert minv == 0
        assert center is None
        assert maxv == 1

    def test_uint8_string_name(self):
        assert (
            iadt.get_value_range_of_dtype("uint8")
            == iadt.get_value_range_of_dtype(np.dtype("uint8"))
        )

    def test_uint8(self):
        minv, center, maxv = iadt.get_value_range_of_dtype(np.dtype("uint8"))
        assert minv == 0
        assert np.isclose(center, 0.5*255)
        assert maxv == 255

    def test_uint16(self):
        minv, center, maxv = iadt.get_value_range_of_dtype(np.dtype("uint16"))
        assert minv == 0
        assert np.isclose(center, 0.5*65535)
        assert maxv == 65535

    def test_int8(self):
        minv, center, maxv = iadt.get_value_range_of_dtype(np.dtype("int8"))
        assert minv == -128
        assert np.isclose(center, -0.5)
        assert maxv == 127

    def test_int16(self):
        minv, center, maxv = iadt.get_value_range_of_dtype(np.dtype("int16"))
        assert minv == -32768
        assert np.isclose(center, -0.5)
        assert maxv == 32767

    def test_float16(self):
        minv, center, maxv = iadt.get_value_range_of_dtype(np.dtype("float16"))
        assert minv < 100.0
        assert np.isclose(center, 0.0)
        assert maxv > 100.0


# TODO extend tests towards all dtypes and actual minima/maxima of value ranges
# TODO what happens if both bounds are negative, but input dtype is uint*?
class Test_clip_(unittest.TestCase):
    def test_values_hit_lower_bound_int32(self):
        arr = np.int32([0, 1, 2, 3, 4, 5])
        observed = iadt.clip_(arr, 0, 10)
        assert np.array_equal(observed, np.int32([0, 1, 2, 3, 4, 5]))

    def test_values_hit_lower_and_upper_bound_int32(self):
        arr = np.int32([0, 1, 2, 3, 4, 5])
        observed = iadt.clip_(arr, 0, 5)
        assert np.array_equal(observed, np.int32([0, 1, 2, 3, 4, 5]))

    def test_values_hit_lower_bound_exceed_upper_bound_int32(self):
        arr = np.int32([0, 1, 2, 3, 4, 5])
        observed = iadt.clip_(arr, 0, 4)
        assert np.array_equal(observed, np.int32([0, 1, 2, 3, 4, 4]))

    def test_values_exceed_lower_bound_float32(self):
        arr = np.float32([-1.0])
        observed = iadt.clip_(arr, 0, 1)
        assert np.allclose(observed, np.float32([0.0]))

    def test_values_hit_lower_bound_float32(self):
        arr = np.float32([-1.0])
        observed = iadt.clip_(arr, -1.0, 1)
        assert np.allclose(observed, np.float32([-1.0]))

    def test_values_hit_lower_bound_uint32(self):
        arr = np.uint32([0])
        observed = iadt.clip_(arr, 0, 1)
        assert np.array_equal(observed, np.uint32([0]))

    def test_values_hit_upper_bound_uint32(self):
        arr = np.uint32([1])
        observed = iadt.clip_(arr, 0, 1)
        assert np.array_equal(observed, np.uint32([1]))

    def test_values_exceed_upper_bound_uint32(self):
        arr = np.uint32([2])
        observed = iadt.clip_(arr, 0, 1)
        assert np.array_equal(observed, np.uint32([1]))

    def test_values_hit_upper_bound_negative_lower_bound_uint32(self):
        arr = np.uint32([1])
        observed = iadt.clip_(arr, -1, 1)
        assert np.array_equal(observed, np.uint32([1]))

    def test_values_exceed_upper_bound_negative_lower_bound_uint32(self):
        arr = np.uint32([10])
        observed = iadt.clip_(arr, -1, 1)
        assert np.array_equal(observed, np.uint32([1]))

    def test_values_hit_upper_bound_int8(self):
        arr = np.int8([127])
        observed = iadt.clip_(arr, 0, 127)
        assert np.array_equal(observed, np.int8([127]))

    def test_values_within_bounds_upper_bound_is_dtype_limit_int8(self):
        arr = np.int8([127])
        observed = iadt.clip_(arr, 0, 128)
        assert np.array_equal(observed, np.int8([127]))

    def test_values_hit_upper_bound_negative_lower_bound_int8(self):
        arr = np.int8([127])
        observed = iadt.clip_(arr, -1, 127)
        assert np.array_equal(observed, np.int8([127]))

    def test_both_bounds_are_none_int8(self):
        arr = np.int8([1])
        observed = iadt.clip_(arr, None, None)
        assert np.array_equal(observed, np.int8([1]))

    def test_lower_bound_is_none_int8(self):
        arr = np.int8([1])
        observed = iadt.clip_(arr, None, 10)
        assert np.array_equal(observed, np.int8([1]))

    def test_upper_bound_is_none_int8(self):
        arr = np.int8([1])
        observed = iadt.clip_(arr, -10, None)
        assert np.array_equal(observed, np.int8([1]))

    def test_values_exceed_upper_bound_and_lower_bound_is_none_int8(self):
        arr = np.int8([10])
        observed = iadt.clip_(arr, None, 1)
        assert np.array_equal(observed, np.int8([1]))

    def test_values_exceed_lower_bound_and_upper_bound_is_none_int8(self):
        arr = np.int8([-10])
        observed = iadt.clip_(arr, -1, None)
        assert np.array_equal(observed, np.int8([-1]))

    def test_numpy_scalar_hits_lower_bound_int8(self):
        # single value arrays, shape == tuple()
        arr = np.int8(-10)
        observed = iadt.clip_(arr, -10, 10)
        assert np.array_equal(observed, np.int8(-10))

    def test_numpy_scalar_exceeds_lower_bound_int8(self):
        arr = np.int8(-10)
        observed = iadt.clip_(arr, -1, 10)
        assert np.array_equal(observed, np.int8(-1))

    def test_numpy_scalar_exceeds_upper_bound_int8(self):
        arr = np.int8(10)
        observed = iadt.clip_(arr, -10, 1)
        assert np.array_equal(observed, np.int8(1))


class Test_clip_to_dtype_value_range(unittest.TestCase):
    def test_clip_to_wider_dtype(self):
        arr = np.array([-10, -1, 0, 1, 10, 255, 256], dtype=np.int16)

        arr_clipped = iadt.clip_to_dtype_value_range_(
            np.copy(arr), np.int32, validate=False)

        assert np.array_equal(arr_clipped, arr)
        assert arr_clipped.dtype.name == "int16"

    def test_clip_to_wider_dtype_given_by_name(self):
        arr = np.array([-10, -1, 0, 1, 10, 255, 256], dtype=np.int16)

        arr_clipped = iadt.clip_to_dtype_value_range_(
            np.copy(arr), "int32", validate=False)

        assert np.array_equal(arr_clipped, arr)
        assert arr_clipped.dtype.name == "int16"

    def test_clip_to_wider_dtype_different_kind(self):
        arr = np.array([-10, -1, 0, 1, 10, 255, 256], dtype=np.int16)

        arr_clipped = iadt.clip_to_dtype_value_range_(
            np.copy(arr), np.float64, validate=False)

        assert np.array_equal(arr_clipped, arr)
        assert arr_clipped.dtype.name == "int16"

    def test_clip_to_same_dtype(self):
        arr = np.array([-10, -1, 0, 1, 10, 255, 256], dtype=np.int16)

        arr_clipped = iadt.clip_to_dtype_value_range_(
            np.copy(arr), np.int16, validate=False)

        assert np.array_equal(arr_clipped, arr)
        assert arr_clipped.dtype.name == "int16"

    def test_clip_to_narrower_dtype(self):
        arr = np.array([-10, -1, 0, 1, 10, 255, 256], dtype=np.int16)

        arr_clipped = iadt.clip_to_dtype_value_range_(
            np.copy(arr), np.int8, validate=False)

        expected = np.array([-10, -1, 0, 1, 10, 127, 127], dtype=np.int16)
        assert np.array_equal(arr_clipped, expected)
        assert arr_clipped.dtype.name == "int16"

    def test_dtype_is_array(self):
        arr = np.array([-10, -1, 0, 1, 10, 255, 256], dtype=np.int16)
        dt_arr = np.array([1], dtype=np.int32)

        arr_clipped = iadt.clip_to_dtype_value_range_(
            np.copy(arr), dt_arr, validate=False)

        assert np.array_equal(arr_clipped, arr)
        assert arr_clipped.dtype.name == "int16"

    def test_validate_true_all_values_within_value_range(self):
        arr = np.array([-10, -1, 0, 1, 10, 126, 127], dtype=np.int16)

        arr_clipped = iadt.clip_to_dtype_value_range_(
            np.copy(arr), np.int8, validate=True)

        assert np.array_equal(arr_clipped, arr)
        assert arr_clipped.dtype.name == "int16"

    def test_validate_true_min_value_outside_value_range(self):
        arr = np.array([-200, -1, 0, 1, 10, 126, 127], dtype=np.int16)

        with self.assertRaises(AssertionError) as context:
            _ = iadt.clip_to_dtype_value_range_(
                np.copy(arr), np.int8, validate=True)

        assert (
            "Minimum value of array is outside of allowed value range "
            "(-200.0000 vs -128.0000 to 127.0000)." in str(context.exception))

    def test_validate_true_max_value_outside_value_range(self):
        arr = np.array([-10, -1, 0, 1, 10, 126, 200], dtype=np.int16)

        with self.assertRaises(AssertionError) as context:
            _ = iadt.clip_to_dtype_value_range_(
                np.copy(arr), np.int8, validate=True)

        assert (
            "Maximum value of array is outside of allowed value range "
            "(200.0000 vs -128.0000 to 127.0000)." in str(context.exception))

    def test_validate_too_few_values(self):
        arr = np.array([-10, 0, 200], dtype=np.int16)

        _ = iadt.clip_to_dtype_value_range_(np.copy(arr), np.int8, validate=2)

    def test_validate_enough_values(self):
        arr = np.array([-10, 0, 200], dtype=np.int16)

        with self.assertRaises(AssertionError) as context:
            _ = iadt.clip_to_dtype_value_range_(
                np.copy(arr), np.int8, validate=3)

        assert (
            "Maximum value of array is outside of allowed value range "
            "(200.0000 vs -128.0000 to 127.0000)." in str(context.exception))

    def test_validate_too_many_values(self):
        arr = np.array([-10, 0, 200], dtype=np.int16)

        with self.assertRaises(AssertionError) as context:
            _ = iadt.clip_to_dtype_value_range_(
                np.copy(arr), np.int8, validate=100)

        assert (
            "Maximum value of array is outside of allowed value range "
            "(200.0000 vs -128.0000 to 127.0000)." in str(context.exception))

    def test_validate_values_set(self):
        arr = np.array([-10, -1, 0, 1, 10, 126, 200], dtype=np.int16)

        with self.assertRaises(AssertionError) as context:
            _ = iadt.clip_to_dtype_value_range_(
                np.copy(arr), np.int8, validate=True,
                validate_values=(-5, 201))

        assert (
            "Maximum value of array is outside of allowed value range "
            "(201.0000 vs -128.0000 to 127.0000)." in str(context.exception))


class Test_gate_dtypes(unittest.TestCase):
    def test_single_array_allowed(self):
        arr = np.zeros((1, 1, 3), dtype=np.int8)

        iadt.gate_dtypes(arr, ["int8"], [])

    def test_single_array_disallowed(self):
        arr = np.zeros((1, 1, 3), dtype=np.int8)

        with self.assertRaises(ValueError) as context:
            iadt.gate_dtypes(arr, ["uint8"], ["int8"])

        assert "Got dtype 'int8'" in str(context.exception)

    def test_list_of_single_array(self):
        arr = np.zeros((1, 1, 3), dtype=np.int8)

        iadt.gate_dtypes([arr], ["int8"], [])

    def test_list_of_two_arrays_same_dtypes(self):
        arrays = [
            np.zeros((1, 1, 3), dtype=np.int8),
            np.zeros((1, 1, 3), dtype=np.int8)
        ]

        iadt.gate_dtypes(arrays, ["int8"], [])

    def test_list_of_two_arrays_different_dtypes(self):
        arrays = [
            np.zeros((1, 1, 3), dtype=np.int8),
            np.zeros((1, 1, 3), dtype=np.uint8)
        ]

        iadt.gate_dtypes(arrays, ["int8", "uint8"], [])

    def test_list_of_two_arrays_same_dtypes_one_disallowed(self):
        arrays = [
            np.zeros((1, 1, 3), dtype=np.int8),
            np.zeros((1, 1, 3), dtype=np.uint8)
        ]

        with self.assertRaises(ValueError) as context:
            iadt.gate_dtypes(arrays, ["int8"], ["uint8"])

        assert "Got dtype 'uint8', which" in str(context.exception)

    def test_single_dtype_allowed(self):
        dtype = np.dtype("int8")

        iadt.gate_dtypes(dtype, ["int8"], [])

    def test_single_dtype_disallowed(self):
        dtype = np.dtype("int8")

        with self.assertRaises(ValueError) as context:
            iadt.gate_dtypes(dtype, ["uint8"], ["int8"])

        assert "Got dtype 'int8', which" in str(context.exception)

    def test_single_dtype_disallowed_augmenter_set(self):
        class _DummyAugmenter(object):
            def __init__(self):
                self.name = "foo"

        dtype = np.dtype("int8")
        dummy_augmenter = _DummyAugmenter()

        with self.assertRaises(ValueError) as context:
            iadt.gate_dtypes(dtype,
                             ["uint8"],
                             ["int8"],
                             augmenter=dummy_augmenter)

        assert "Got dtype 'int8' in augmenter 'foo'" in str(context.exception)

    def test_single_dtype_function(self):
        dtype = np.int8

        iadt.gate_dtypes(dtype, ["int8"], [])

    def test_single_dtype_name(self):
        dtype = "int8"

        iadt.gate_dtypes(dtype, ["int8"], [])

    def test_list_of_two_dtypes_both_same(self):
        dtypes = [
            np.dtype("int8"),
            np.dtype("int8")
        ]

        iadt.gate_dtypes(dtypes, ["int8"], [])

    def test_list_of_two_dtypes_both_different(self):
        dtypes = [
            np.dtype("int8"),
            np.dtype("uint8")
        ]

        iadt.gate_dtypes(dtypes, ["int8", "uint8"], [])

    def test_list_of_two_dtypes_both_different_one_disallowed(self):
        dtypes = [
            np.dtype("int8"),
            np.dtype("uint8")
        ]

        with self.assertRaises(ValueError) as context:
            iadt.gate_dtypes(dtypes, ["int8"], ["uint8"])

        assert "Got dtype 'uint8', which" in str(context.exception)

    def test_dtype_not_in_allowed_or_disallowed(self):
        dtypes = [
            np.dtype("int8"),
            np.dtype("float32")
        ]

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            iadt.gate_dtypes(dtypes, ["int8"], ["uint8"])

        assert len(caught_warnings) == 1
        assert (
            "Got dtype 'float32', which" in str(caught_warnings[-1].message))

    def test_dtype_not_in_allowed_or_disallowed_augmenter_set(self):
        class _DummyAugmenter(object):
            def __init__(self):
                self.name = "foo"

        dtypes = [
            np.dtype("int8"),
            np.dtype("float32")
        ]
        dummy_augmenter = _DummyAugmenter()

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            iadt.gate_dtypes(dtypes,
                             ["int8"],
                             ["uint8"],
                             augmenter=dummy_augmenter)

        assert len(caught_warnings) == 1
        assert (
            "Got dtype 'float32' in augmenter 'foo'"
            in str(caught_warnings[-1].message))
