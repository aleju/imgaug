from __future__ import print_function, absolute_import, division
import numpy as np
import imgaug as ia


# TODO integrate into polygons
# TODO integrate into keypoints
def normalize_shape(shape):
    """
    Normalize a shape tuple or array to a shape tuple.

    Parameters
    ----------
    shape : tuple of int or ndarray
        The input to normalize. May optionally be an array.

    Returns
    -------
    tuple of int
        Shape tuple.

    """
    if isinstance(shape, tuple):
        return shape
    assert ia.is_np_array(shape), (
        "Expected tuple of ints or array, got %s." % (type(shape),))
    return shape.shape


# TODO intergrate into polygons
# TODO integrate into keypoints
def project_coords(coords, from_shape, to_shape):
    """
    Project coordinates from one image shape to another.

    This performs a relative projection, e.g. a point at 60% of the old
    image width will be at 60% of the new image width after projection.

    Parameters
    ----------
    coords : ndarray or tuple of number
        Coordinates to project. Either a ``(N,2)`` numpy array or a tuple
        of `(x,y)` coordinates.

    from_shape : tuple of int or ndarray
        Old image shape.

    to_shape : tuple of int or ndarray
        New image shape.

    Returns
    -------
    ndarray
        Projected coordinates as ``(N,2)`` ``float32`` numpy array.

    """
    from_shape = normalize_shape(from_shape)
    to_shape = normalize_shape(to_shape)
    if from_shape[0:2] == to_shape[0:2]:
        return coords

    from_height, from_width = from_shape[0:2]
    to_height, to_width = to_shape[0:2]
    assert all([v > 0 for v in [from_height, from_width, to_height, to_width]])

    # make sure to not just call np.float32(coords) here as the following lines
    # perform in-place changes and np.float32(.) only copies if the input
    # was *not* a float32 array
    coords_proj = np.array(coords).astype(np.float32)
    coords_proj[:, 0] = (coords_proj[:, 0] / from_width) * to_width
    coords_proj[:, 1] = (coords_proj[:, 1] / from_height) * to_height
    return coords_proj
