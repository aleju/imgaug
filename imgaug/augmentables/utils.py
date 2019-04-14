from __future__ import print_function, absolute_import, division
import numpy as np
import six.moves as sm
import imgaug as ia


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


def interpolate_point_pair(point_a, point_b, nb_steps):
    if nb_steps < 1:
        return []
    x1, y1 = point_a
    x2, y2 = point_b
    vec = np.float32([x2 - x1, y2 - y1])
    step_size = vec / (1 + nb_steps)
    return [(x1 + (i + 1) * step_size[0], y1 + (i + 1) * step_size[1]) for i in sm.xrange(nb_steps)]


def interpolate_points(points, nb_steps, closed=True):
    if len(points) <= 1:
        return points
    if closed:
        points = list(points) + [points[0]]
    points_interp = []
    for point_a, point_b in zip(points[:-1], points[1:]):
        points_interp.extend([point_a] + interpolate_point_pair(point_a, point_b, nb_steps))
    if not closed:
        points_interp.append(points[-1])
    # close does not have to be reverted here, as last point is not included in the extend()
    return points_interp


def interpolate_points_by_max_distance(points, max_distance, closed=True):
    ia.do_assert(max_distance > 0, "max_distance must have value greater than 0, got %.8f" % (max_distance,))
    if len(points) <= 1:
        return points
    if closed:
        points = list(points) + [points[0]]
    points_interp = []
    for point_a, point_b in zip(points[:-1], points[1:]):
        dist = np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
        nb_steps = int((dist / max_distance) - 1)
        points_interp.extend([point_a] + interpolate_point_pair(point_a, point_b, nb_steps))
    if not closed:
        points_interp.append(points[-1])
    return points_interp
