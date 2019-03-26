from __future__ import print_function, division
import warnings
import numpy as np
import six.moves as sm
import imgaug as ia

KIND_TO_DTYPES = {
    "i": ["int8", "int16", "int32", "int64"],
    "u": ["uint8", "uint16", "uint32", "uint64"],
    "b": ["bool"],
    "f": ["float16", "float32", "float64", "float128"]
}


def restore_dtypes_(images, dtypes, clip=True, round=True):
    if ia.is_np_array(images):
        if ia.is_iterable(dtypes):
            assert len(dtypes) > 0

            if len(dtypes) > 1:
                assert all([dtype_i == dtypes[0] for dtype_i in dtypes])

            dtypes = dtypes[0]

        dtypes = np.dtype(dtypes)

        dtype_to = dtypes
        if images.dtype.type == dtype_to:
            result = images
        else:
            if round and dtype_to.kind in ["u", "i", "b"]:
                images = np.round(images)
            if clip:
                min_value, _, max_value = get_value_range_of_dtype(dtype_to)
                images = clip_(images, min_value, max_value)
            result = images.astype(dtype_to, copy=False)
    elif ia.is_iterable(images):
        result = images
        dtypes = dtypes if not isinstance(dtypes, np.dtype) else [dtypes] * len(images)
        for i, (image, dtype) in enumerate(zip(images, dtypes)):
            assert ia.is_np_array(image)
            result[i] = restore_dtypes_(image, dtype, clip=clip)
    else:
        raise Exception("Expected numpy array or iterable of numpy arrays, got type '%s'." % (type(images),))
    return result


def copy_dtypes_for_restore(images, force_list=False):
    if ia.is_np_array(images):
        if force_list:
            return [images.dtype for _ in sm.xrange(len(images))]
        else:
            return images.dtype
    else:
        return [image.dtype for image in images]


def get_minimal_dtype(arrays, increase_itemsize_factor=1):
    input_dts = [array.dtype if not isinstance(array, np.dtype) else array
                 for array in arrays]
    promoted_dt = np.promote_types(*input_dts)
    if increase_itemsize_factor > 1:
        promoted_dt_highres = "%s%d" % (promoted_dt.kind, promoted_dt.itemsize * increase_itemsize_factor)
        try:
            promoted_dt_highres = np.dtype(promoted_dt_highres)
            return promoted_dt_highres
        except TypeError:
            raise TypeError(
                ("Unable to create a numpy dtype matching the name '%s'. "
                 + "This error was caused when trying to find a minimal dtype covering the dtypes '%s' (which was "
                 + "determined to be '%s') and then increasing its resolution (aka itemsize) by a factor of %d. "
                 + "This error can be avoided by choosing arrays with lower resolution dtypes as inputs, e.g. by "
                 + "reducing float32 to float16.") % (
                    promoted_dt_highres,
                    ", ".join([input_dt.name for input_dt in input_dts]),
                    promoted_dt.name,
                    increase_itemsize_factor
                )
            )
    return promoted_dt


def get_minimal_dtype_for_values(values, allowed_kinds, default, allow_bool_as_intlike=True):
    values_normalized = []
    for value in values:
        if ia.is_np_array(value):
            values_normalized.extend([np.min(values), np.max(values)])
        else:
            values_normalized.append(value)
    vmin = np.min(values_normalized)
    vmax = np.max(values_normalized)
    possible_kinds = []
    if ia.is_single_float(vmin) or ia.is_single_float(vmax):
        # at least one is a float
        possible_kinds.append("f")
    elif ia.is_single_bool(vmin) and ia.is_single_bool(vmax):
        # both are bools
        possible_kinds.extend(["b", "u", "i"])
    else:
        # at least one of them is an integer and none is float
        if vmin >= 0:
            possible_kinds.append("u")
        possible_kinds.append("i")
        # vmin and vmax are already guarantueed to not be float due to if-statement above
        if allow_bool_as_intlike and 0 <= vmin <= 1 and 0 <= vmax <= 1:
            possible_kinds.append("b")

    for allowed_kind in allowed_kinds:
        if allowed_kind in possible_kinds:
            dt = get_minimal_dtype_by_value_range(vmin, vmax, allowed_kind, default=None)
            if dt is not None:
                return dt

    if ia.is_string(default) and default == "raise":
        raise Exception(("Did not find matching dtypes for vmin=%s (type %s) and vmax=%s (type %s). "
                         + "Got %s input values of types %s.") % (
            vmin, type(vmin), vmax, type(vmax), ", ".join([str(type(value)) for value in values])))
    return default


def get_minimal_dtype_by_value_range(low, high, kind, default):
    assert low <= high, "Expected low to be less or equal than high, got %s and %s." % (low, high)
    for dt in KIND_TO_DTYPES[kind]:
        min_value, _center_value, max_value = get_value_range_of_dtype(dt)
        if min_value <= low and high <= max_value:
            return np.dtype(dt)
    if ia.is_string(default) and default == "raise":
        raise Exception("Could not find dtype of kind '%s' within value range [%s, %s]" % (kind, low, high))
    return default


def promote_array_dtypes_(arrays, dtypes=None, increase_itemsize_factor=1, affects=None):
    if dtypes is None:
        dtypes = [array.dtype for array in arrays]
    dt = get_minimal_dtype(dtypes, increase_itemsize_factor=increase_itemsize_factor)
    if affects is None:
        affects = arrays
    result = []
    for array in affects:
        if array.dtype.type != dt:
            array = array.astype(dt, copy=False)
        result.append(array)
    return result


def increase_array_resolutions_(arrays, factor):
    assert ia.is_single_integer(factor)
    assert factor in [1, 2, 4, 8]
    if factor == 1:
        return arrays

    for i, array in enumerate(arrays):
        dtype = array.dtype
        dtype_target = np.dtype("%s%d" % (dtype.kind, dtype.itemsize * factor))
        arrays[i] = array.astype(dtype_target, copy=False)

    return arrays


def get_value_range_of_dtype(dtype):
    # normalize inputs, makes it work with strings (e.g. "uint8"), types like np.uint8 and also proper dtypes, like
    # np.dtype("uint8")
    dtype = np.dtype(dtype)

    # This check seems to fail sometimes, e.g. get_value_range_of_dtype(np.int8)
    # assert isinstance(dtype, np.dtype), "Expected instance of numpy.dtype, got %s." % (type(dtype),)

    if dtype.kind == "f":
        finfo = np.finfo(dtype)
        return finfo.min, 0.0, finfo.max
    elif dtype.kind == "u":
        iinfo = np.iinfo(dtype)
        return iinfo.min, int(iinfo.min + 0.5 * iinfo.max), iinfo.max
    elif dtype.kind == "i":
        iinfo = np.iinfo(dtype)
        return iinfo.min, -0.5, iinfo.max
    elif dtype.kind == "b":
        return 0, None, 1
    else:
        raise Exception("Cannot estimate value range of dtype '%s' (type: %s)" % (str(dtype), type(dtype)))


# TODO call this function wherever data is clipped
def clip_(array, min_value, max_value):
    # If the min of the input value range is above the allowed min, we do not
    # have to clip to the allowed min as we cannot exceed it anyways.
    # Analogous for max. In fact, we must not clip then to min/max as that can
    # lead to errors in numpy's clip. E.g.
    #     >>> arr = np.zeros((1,), dtype=np.int32)
    #     >>> np.clip(arr, 0, np.iinfo(np.dtype("uint32")).max)
    # will return
    #     array([-1], dtype=int32)
    # (observed on numpy version 1.15.2).
    min_value_arrdt, _, max_value_arrdt = get_value_range_of_dtype(array.dtype)
    if min_value is not None and min_value < min_value_arrdt:
        min_value = None
    if max_value is not None and max_value_arrdt < max_value:
        max_value = None

    if min_value is not None or max_value is not None:
        # for scalar arrays, i.e. with shape = (), "out" is not a valid
        # argument
        if len(array.shape) == 0:
            array = np.clip(array, min_value, max_value)
        else:
            array = np.clip(array, min_value, max_value, out=array)
    return array


def clip_to_dtype_value_range_(array, dtype, validate=True, validate_values=None):
    # for some reason, using 'out' did not work for uint64 (would clip max value to 0)
    # but removing out then results in float64 array instead of uint64
    assert array.dtype.name not in ["uint64", "uint128"]

    dtype = np.dtype(dtype)
    min_value, _, max_value = get_value_range_of_dtype(dtype)
    if validate:
        array_val = array
        if ia.is_single_integer(validate):
            assert validate_values is None
            array_val = array.flat[0:validate]
        if validate_values is not None:
            min_value_found, max_value_found = validate_values
        else:
            min_value_found = np.min(array_val)
            max_value_found = np.max(array_val)
        assert min_value <= min_value_found <= max_value
        assert min_value <= max_value_found <= max_value

    return clip_(array, min_value, max_value)


def gate_dtypes(dtypes, allowed, disallowed, augmenter=None):
    assert len(allowed) > 0
    assert ia.is_string(allowed[0])
    if len(disallowed) > 0:
        assert ia.is_string(disallowed[0])

    if ia.is_np_array(dtypes):
        dtypes = [dtypes.dtype]
    else:
        dtypes = [np.dtype(dtype) if not ia.is_np_array(dtype) else dtype.dtype for dtype in dtypes]
    for dtype in dtypes:
        if dtype.name in allowed:
            pass
        elif dtype.name in disallowed:
            if augmenter is None:
                raise ValueError("Got dtype '%s', which is a forbidden dtype (%s)." % (
                    dtype.name, ", ".join(disallowed)
                ))
            else:
                raise ValueError("Got dtype '%s' in augmenter '%s' (class '%s'), which is a forbidden dtype (%s)." % (
                    dtype.name, augmenter.name, augmenter.__class__.__name__, ", ".join(disallowed)
                ))
        else:
            if augmenter is None:
                warnings.warn(("Got dtype '%s', which was neither explicitly allowed "
                               + "(%s), nor explicitly disallowed (%s). Generated outputs may contain errors.") % (
                        dtype.name, ", ".join(allowed), ", ".join(disallowed)
                    ))
            else:
                warnings.warn(("Got dtype '%s' in augmenter '%s' (class '%s'), which was neither explicitly allowed "
                               + "(%s), nor explicitly disallowed (%s). Generated outputs may contain errors.") % (
                    dtype.name, augmenter.name, augmenter.__class__.__name__, ", ".join(allowed), ", ".join(disallowed)
                ))

