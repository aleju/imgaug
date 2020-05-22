"""Functions to generate example data, e.g. example images or segmaps.

Added in 0.5.0.

"""
from __future__ import print_function, division, absolute_import

import os
import json

import imageio
import numpy as np

# filepath to the quokka image, its annotations and depth map
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_QUOKKA_FP = os.path.join(_FILE_DIR, "quokka.jpg")
_QUOKKA_ANNOTATIONS_FP = os.path.join(_FILE_DIR, "quokka_annotations.json")
_QUOKKA_DEPTH_MAP_HALFRES_FP = os.path.join(
    _FILE_DIR, "quokka_depth_map_halfres.png")


def _quokka_normalize_extract(extract):
    """Generate a normalized rectangle for the standard quokka image.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    extract : 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Unnormalized representation of the image subarea to be extracted.

            * If ``str`` ``square``, then a squared area
              ``(x: 0 to max 643, y: 0 to max 643)`` will be extracted from
              the image.
            * If a ``tuple``, then expected to contain four ``number`` s
              denoting ``(x1, y1, x2, y2)``.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBox`, then that
              bounding box's area will be extracted from the image.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBoxesOnImage`,
              then expected to contain exactly one bounding box and a shape
              matching the full image dimensions (i.e. ``(643, 960, *)``).
              Then the one bounding box will be used similar to
              ``BoundingBox`` above.

    Returns
    -------
    imgaug.augmentables.bbs.BoundingBox
        Normalized representation of the area to extract from the standard
        quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

    if extract == "square":
        bb = BoundingBox(x1=0, y1=0, x2=643, y2=643)
    elif isinstance(extract, tuple) and len(extract) == 4:
        bb = BoundingBox(x1=extract[0], y1=extract[1],
                         x2=extract[2], y2=extract[3])
    elif isinstance(extract, BoundingBox):
        bb = extract
    elif isinstance(extract, BoundingBoxesOnImage):
        assert len(extract.bounding_boxes) == 1, (
            "Provided BoundingBoxesOnImage instance may currently only "
            "contain a single bounding box.")
        assert extract.shape[0:2] == (643, 960), (
            "Expected BoundingBoxesOnImage instance on an image of shape "
            "(643, 960, ?). Got shape %s." % (extract.shape,))
        bb = extract.bounding_boxes[0]
    else:
        raise Exception(
            "Expected 'square' or tuple of four entries or BoundingBox or "
            "BoundingBoxesOnImage for parameter 'extract', "
            "got %s." % (type(extract),)
        )
    return bb


# TODO is this the same as the project functions in augmentables?
def _compute_resized_shape(from_shape, to_shape):
    """Compute the intended new shape of an image-like array after resizing.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    from_shape : tuple or ndarray
        Old shape of the array. Usually expected to be a ``tuple`` of form
        ``(H, W)`` or ``(H, W, C)`` or alternatively an array with two or
        three dimensions.

    to_shape : None or tuple of ints or tuple of floats or int or float or ndarray
        New shape of the array.

            * If ``None``, then `from_shape` will be used as the new shape.
            * If an ``int`` ``V``, then the new shape will be ``(V, V, [C])``,
              where ``C`` will be added if it is part of `from_shape`.
            * If a ``float`` ``V``, then the new shape will be
              ``(H*V, W*V, [C])``, where ``H`` and ``W`` are the old
              height/width.
            * If a ``tuple`` ``(H', W', [C'])`` of ints, then ``H'`` and ``W'``
              will be used as the new height and width.
            * If a ``tuple`` ``(H', W', [C'])`` of floats (except ``C``), then
              ``H'`` and ``W'`` will be used as the new height and width.
            * If a numpy array, then the array's shape will be used.

    Returns
    -------
    tuple of int
        New shape.

    """
    from . import imgaug as ia

    if ia.is_np_array(from_shape):
        from_shape = from_shape.shape
    if ia.is_np_array(to_shape):
        to_shape = to_shape.shape

    to_shape_computed = list(from_shape)

    if to_shape is None:
        pass
    elif isinstance(to_shape, tuple):
        assert len(from_shape) in [2, 3]
        assert len(to_shape) in [2, 3]

        if len(from_shape) == 3 and len(to_shape) == 3:
            assert from_shape[2] == to_shape[2]
        elif len(to_shape) == 3:
            to_shape_computed.append(to_shape[2])

        is_to_s_valid_values = all(
            [v is None or ia.is_single_number(v) for v in to_shape[0:2]])
        assert is_to_s_valid_values, (
            "Expected the first two entries in to_shape to be None or "
            "numbers, got types %s." % (
                str([type(v) for v in to_shape[0:2]]),))

        for i, from_shape_i in enumerate(from_shape[0:2]):
            if to_shape[i] is None:
                to_shape_computed[i] = from_shape_i
            elif ia.is_single_integer(to_shape[i]):
                to_shape_computed[i] = to_shape[i]
            else:  # float
                to_shape_computed[i] = int(np.round(from_shape_i * to_shape[i]))
    elif ia.is_single_integer(to_shape) or ia.is_single_float(to_shape):
        to_shape_computed = _compute_resized_shape(
            from_shape, (to_shape, to_shape))
    else:
        raise Exception(
            "Expected to_shape to be None or ndarray or tuple of floats or "
            "tuple of ints or single int or single float, "
            "got %s." % (type(to_shape),))

    return tuple(to_shape_computed)


def quokka(size=None, extract=None):
    """Return an image of a quokka as a numpy array.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int, optional
        Size of the output image. Input into
        :func:`~imgaug.imgaug.imresize_single_image`. Usually expected to be a
        ``tuple`` ``(H, W)``, where ``H`` is the desired height and ``W`` is
        the width. If ``None``, then the image will not be resized.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea of the quokka image to extract:

            * If ``None``, then the whole image will be used.
            * If ``str`` ``square``, then a squared area
              ``(x: 0 to max 643, y: 0 to max 643)`` will be extracted from
              the image.
            * If a ``tuple``, then expected to contain four ``number`` s
              denoting ``(x1, y1, x2, y2)``.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBox`, then that
              bounding box's area will be extracted from the image.
            * If a :class:`~imgaug.augmentables.bbs.BoundingBoxesOnImage`,
              then expected to contain exactly one bounding box and a shape
              matching the full image dimensions (i.e. ``(643, 960, *)``).
              Then the one bounding box will be used similar to
              ``BoundingBox`` above.

    Returns
    -------
    (H,W,3) ndarray
        The image array of dtype ``uint8``.

    """
    from . import imgaug as ia

    img = imageio.imread(_QUOKKA_FP, pilmode="RGB")
    if extract is not None:
        bb = _quokka_normalize_extract(extract)
        img = bb.extract_from_image(img)
    if size is not None:
        shape_resized = _compute_resized_shape(img.shape, size)
        img = ia.imresize_single_image(img, shape_resized[0:2])
    return img


def quokka_square(size=None):
    """Return an (square) image of a quokka as a numpy array.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int, optional
        Size of the output image. Input into
        :func:`~imgaug.imgaug.imresize_single_image`. Usually expected to be a
        ``tuple`` ``(H, W)``, where ``H`` is the desired height and ``W`` is
        the width. If ``None``, then the image will not be resized.

    Returns
    -------
    (H,W,3) ndarray
        The image array of dtype ``uint8``.

    """
    return quokka(size=size, extract="square")


def quokka_heatmap(size=None, extract=None):
    """Return a heatmap (here: depth map) for the standard example quokka image.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int, optional
        See :func:`~imgaug.imgaug.quokka`.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.heatmaps.HeatmapsOnImage
        Depth map as an heatmap object. Values close to ``0.0`` denote objects
        that are close to the camera. Values close to ``1.0`` denote objects
        that are furthest away (among all shown objects).

    """
    # TODO get rid of this deferred import
    from . import imgaug as ia
    from imgaug.augmentables.heatmaps import HeatmapsOnImage

    img = imageio.imread(_QUOKKA_DEPTH_MAP_HALFRES_FP, pilmode="RGB")
    img = ia.imresize_single_image(img, (643, 960), interpolation="cubic")

    if extract is not None:
        bb = _quokka_normalize_extract(extract)
        img = bb.extract_from_image(img)
    if size is None:
        size = img.shape[0:2]

    shape_resized = _compute_resized_shape(img.shape, size)
    img = ia.imresize_single_image(img, shape_resized[0:2])
    img_0to1 = img[..., 0]  # depth map was saved as 3-channel RGB
    img_0to1 = img_0to1.astype(np.float32) / 255.0
    img_0to1 = 1 - img_0to1  # depth map was saved as 0 being furthest away

    return HeatmapsOnImage(img_0to1, shape=img_0to1.shape[0:2] + (3,))


def quokka_segmentation_map(size=None, extract=None):
    """Return a segmentation map for the standard example quokka image.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int, optional
        See :func:`~imgaug.imgaug.quokka`.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.segmaps.SegmentationMapsOnImage
        Segmentation map object.

    """
    # pylint: disable=invalid-name
    import skimage.draw
    # TODO get rid of this deferred import
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    with open(_QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)

    xx = []
    yy = []
    for kp_dict in json_dict["polygons"][0]["keypoints"]:
        x = kp_dict["x"]
        y = kp_dict["y"]
        xx.append(x)
        yy.append(y)

    img_seg = np.zeros((643, 960, 1), dtype=np.int32)
    rr, cc = skimage.draw.polygon(
        np.array(yy), np.array(xx), shape=img_seg.shape)
    img_seg[rr, cc, 0] = 1

    if extract is not None:
        bb = _quokka_normalize_extract(extract)
        img_seg = bb.extract_from_image(img_seg)

    segmap = SegmentationMapsOnImage(img_seg, shape=img_seg.shape[0:2] + (3,))

    if size is not None:
        shape_resized = _compute_resized_shape(img_seg.shape, size)
        segmap = segmap.resize(shape_resized[0:2])
        segmap.shape = tuple(shape_resized[0:2]) + (3,)

    return segmap


def quokka_keypoints(size=None, extract=None):
    """Return example keypoints on the standard example quokke image.

    The keypoints cover the eyes, ears, nose and paws.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the keypoints are placed. If
        ``None``, then the keypoints are not projected to any new size
        (positions on the original image are used). ``float`` s lead to
        relative size changes, ``int`` s to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.kps.KeypointsOnImage
        Example keypoints on the quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

    left, top = 0, 0
    if extract is not None:
        bb_extract = _quokka_normalize_extract(extract)
        left = bb_extract.x1
        top = bb_extract.y1
    with open(_QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    keypoints = []
    for kp_dict in json_dict["keypoints"]:
        keypoints.append(Keypoint(x=kp_dict["x"] - left, y=kp_dict["y"] - top))
    if extract is not None:
        shape = (bb_extract.height, bb_extract.width, 3)
    else:
        shape = (643, 960, 3)
    kpsoi = KeypointsOnImage(keypoints, shape=shape)
    if size is not None:
        shape_resized = _compute_resized_shape(shape, size)
        kpsoi = kpsoi.on(shape_resized)
    return kpsoi


def quokka_bounding_boxes(size=None, extract=None):
    """Return example bounding boxes on the standard example quokke image.

    Currently only a single bounding box is returned that covers the quokka.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the BBs are placed. If ``None``, then
        the BBs are not projected to any new size (positions on the original
        image are used). ``float`` s lead to relative size changes, ``int`` s
        to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.bbs.BoundingBoxesOnImage
        Example BBs on the quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

    left, top = 0, 0
    if extract is not None:
        bb_extract = _quokka_normalize_extract(extract)
        left = bb_extract.x1
        top = bb_extract.y1
    with open(_QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    bbs = []
    for bb_dict in json_dict["bounding_boxes"]:
        bbs.append(
            BoundingBox(
                x1=bb_dict["x1"] - left,
                y1=bb_dict["y1"] - top,
                x2=bb_dict["x2"] - left,
                y2=bb_dict["y2"] - top
            )
        )
    if extract is not None:
        shape = (bb_extract.height, bb_extract.width, 3)
    else:
        shape = (643, 960, 3)
    bbsoi = BoundingBoxesOnImage(bbs, shape=shape)
    if size is not None:
        shape_resized = _compute_resized_shape(shape, size)
        bbsoi = bbsoi.on(shape_resized)
    return bbsoi


def quokka_polygons(size=None, extract=None):
    """
    Returns example polygons on the standard example quokke image.

    The result contains one polygon, covering the quokka's outline.

    Added in 0.5.0. (Moved from ``imgaug.imgaug``.)

    Parameters
    ----------
    size : None or float or tuple of int or tuple of float, optional
        Size of the output image on which the polygons are placed. If ``None``,
        then the polygons are not projected to any new size (positions on the
        original image are used). ``float`` s lead to relative size changes,
        ``int`` s to absolute sizes in pixels.

    extract : None or 'square' or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage
        Subarea to extract from the image. See :func:`~imgaug.imgaug.quokka`.

    Returns
    -------
    imgaug.augmentables.polys.PolygonsOnImage
        Example polygons on the quokka image.

    """
    # TODO get rid of this deferred import
    from imgaug.augmentables.polys import Polygon, PolygonsOnImage

    left, top = 0, 0
    if extract is not None:
        bb_extract = _quokka_normalize_extract(extract)
        left = bb_extract.x1
        top = bb_extract.y1
    with open(_QUOKKA_ANNOTATIONS_FP, "r") as f:
        json_dict = json.load(f)
    polygons = []
    for poly_json in json_dict["polygons"]:
        polygons.append(
            Polygon([(point["x"] - left, point["y"] - top)
                     for point in poly_json["keypoints"]])
        )
    if extract is not None:
        shape = (bb_extract.height, bb_extract.width, 3)
    else:
        shape = (643, 960, 3)
    psoi = PolygonsOnImage(polygons, shape=shape)
    if size is not None:
        shape_resized = _compute_resized_shape(shape, size)
        psoi = psoi.on(shape_resized)
    return psoi
