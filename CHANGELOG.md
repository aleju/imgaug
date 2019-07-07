# master (will be 0.3.0)

* Added argument `output_buffer_size` to `multicore.Pool.imap_batches()`
  and `multicore.Pool.imap_batches_unordered()` to control the maximum number
  of batches in the background augmentation pipeline (allows to limit
  maximum RAM demands).
* Increased `max_distance` thresholds for `almost_equals()`,
  `exterior_almost_equals()` and `coords_almost_equals()` in `Polygon` and
  `LineString` from `1e-6` to `1e-4`.
  This should fix false-negative problems related to float inaccuracies.
* Added module `imgaug.augmenters.edges`.
* Added interface `augmenters.edges.BinaryImageColorizerIf`, which
  contains the interface for classes used to convert binary images to RGB
  images.
* Added `augmenters.edges.RandomColorsBinaryImageColorizer`, which
  converts binary images to RGB images by sampling uniformly RGB colors for
  `True` and `False` values.
* Added `augmenters.edges.Canny`, which applies canny edge detection with alpha
  blending and random coloring to images.
* Renamed `imgaug/external/poly_point_isect.py` to
  `imgaug/external/poly_point_isect_py3.py.bak`.
  The file is in the library only for completeness and contains python3 syntax.
  `poly_point_isect_py2py3.py` is actually used.
* Added dtype gating to `dtypes.clip_()`.
* Added module `augmenters.pooling`. #317
    * Added `augmenters.pooling._AbstractPoolingBase`. #317
    * Added `augmenters.pooling.AveragePooling`. #317
    * Added `augmenters.pooling.MaxPooling`. #317
    * Added `augmenters.pooling.MinPooling`. #317
    * Added `augmenters.pooling.MedianPooling`. #317
* `augmenters.color.AddToHueAndSaturation`
    * [rarely breaking] Refactored `AddToHueAndSaturation` to clean it up.
      Re-running old code with the same seeds will now produce different
      images. #319
    * [rarely breaking] The `value` parameter is now interpreted by the
      augmenter to return first the hue and then the saturation value to add,
      instead of the other way round.
      (This shouldn't affect anybody.) #319
    * [rarely breaking] Added `value_hue` and `value_saturation` arguments,
      which allow to set individual parameters for hue and saturation
      instead of having to use one parameter for both (they may not be set
      if `value` is already set).
      This changes the order of arguments of the augmenter and code that relied
      on that order will now break.
      This also changes the output of
      `AddToHueAndSaturation.get_parameters()`. #319
* Added `augmenters.color.AddToHue`, a shortcut for
  `AddToHueAndSaturation(value_hue=...)`. #319
* Added `augmenters.color.AddToSaturation`, a shortcut for
  `AddToHueAndSaturation(value_saturation=...)`. #319
* Added `augmenters.color.WithHueAndSaturation`. #319
* Added `augmenters.color.MultiplyHueAndSaturation`. #319
* Added `augmenters.color.MultiplyHue`. #319
* Added `augmenters.color.MultiplySaturation`. #319
* Refactored `augmenters/weather.py` (general code and docstring cleanup). #336
* [rarely breaking] Refactored `augmenters/convolutional.py`
  (general code and docstring cleanup).
  This involved changing the random state handling.
  Old seeds might now produce different result images for convolutional
  augmenters (`Convolve`, `Sharpen`, `Emboss`, `EdgeDetect`,
  `DirectedEdgeDetect`). #335
* [rarely breaking] Added argument `polygon_recoverer` to
  `augmenters.geometric.PerspectiveTransform`. This changes the order of
  arguments of the augmenter and code that relied on that order will now
  break. #338
* Changed `_ConcavePolygonRecoverer` to not search for segment intersection
  points in polygons with very large absolute coordinate values.
  This prevents rare errors due to floating point inaccuracies. #338
* Changed `_ConcavePolygonRecoverer` to raise warnings instead of throwing
  exceptions when the underlying search for segment intersection points
  crashes. #338
* Added the library to `conda-forge` so it can now be installed via
  `conda install imgaug` (provided the conda-forge channel was added
  before that). #320 #339
* Changed dependency `opencv-python` to `opencv-python-headless`.
  This should improve support for some system without GUIs.
* Refactored code in `augmenters.segmentation` (general code and docstring cleanup). #334
* Refactored code in `augmenters.arithmetic` (general code and docstring cleanup). #328
* Added check to `dtypes.gate_dtypes()` verifying that arguments `allowed`
  and `disallowed` have no intersection. #346
* Added dependency `pytest-subtests` for the library's unittests. #366
* Added `imgaug.is_np_scalar()`, analogous to `imgaug.is_np_array()`. #366
* Reworked and refactored code in `dtypes.py` (general code cleanup). #366
  * Added `dtypes.normalize_dtypes()`.
  * Added `dtypes.normaliz_dtypes()`.
  * Refactored `dtypes.promote_array_dtypes_()` to use
    `dtypes.change_dtypes_()`.
  * Reworked dtype normalization. All functions in the module that required
    dtype inputs accept now dtypes, dtype functions, dtype names, ndarrays
    or numpy scalars.
  * [rarely breaking] `dtypes.restore_dtypes_()`
    * Improved error messages.
    * Changed so that if `images` is a list of length `N` and `dtypes` is a
      list of length `M` and `N!=M`, the function now raises an
      `AssertionError`.
    * The argument `round` is now ignored if the input array does not have
      a float dtype.
  * Renamed `dtypes.restore_dtypes_()` to `dtypes.change_dtypes_()` (old name
    still exists too and redirects to new name).
  * Added `dtypes.change_dtype_()`, analogous to `dtypes.change_dtypes_()`.
  * Added `dtypes.increase_itemsize_of_dtype()`.
      * Refactored `dtypes.get_minimal_dtype()` to use that new function.
  * [rarely breaking] Removed `dtypes.get_minimal_dtype_for_values()`. The
    function was not used anywhere in the library.
  * [rarely breaking] Removed `dtypes.get_minimal_dtype_by_value_range()`. The
    function was not used anywhere in the library.
  * Changed `dtypes.get_value_range_of_dtype()` to return a float as the center
    value of `uint` dtypes.
  * [rarely breaking] Removed argument `affects` from
    `dtypes.promote_array_dtypes_()` as it was unnecessary and not used anywhere
    in the library. #366
* Added `imgaug.warn()` function. #367
* Changed `multicore.Pool` to produce a warning if it cannot find or call the
  function `multiprocessing.cpu_count()` instead of silently failing.
  (In both cases it falls back to a default value.) #367
* Refactored code in `multicore` (general code and docstring cleanup). #367
  * Improved error messages in `multicore`.
* Added `imgaug.min_pool()`.
  * Refactored `augmenters.pooling.MinPooling` to use `imgaug.min_pool()`.
* Added `imgaug.median_pool()`.
  * Refactored `augmenters.pooling.MedianPooling` to use `imgaug.median_pool()`.
* Added `imgaug.compute_paddings_to_reach_multiples_of()`.
* Added `imgaug.pad_to_multiples_of()`.
* Refactored `imgaug.pool()` to use `imgaug.pad()` for image padding.
* [rarely breaking] Added a `pad_mode` argument to `imgaug.pool()`,
  `imgaug.avg_pool()`, `imgaug.max_pool()`, `imgaug.min_pool()` and
  `imgaug.median_pool()`. This breaks code relying on the order of the
  functions arguments.s
  * Changed the default `pad_mode` of `avg_pool` from `constant` (`cval=128`)
    to `reflect`.
  * Changed the default `pad_mode` of `max_pool` from `constant` (`cval=0`)
    to `edge`.
  * Changed the default `pad_mode` of `min_pool` from `constant` (`cval=255`)
    to `edge`.
  * Changed the default `pad_mode` of `median_pool` from `constant`
    (`cval=128`) to `reflect`.
* Renamed argument `cval` to `pad_cval` in `imgaug.pool()`,
  `imgaug.avg_pool()` and `imgaug.max_pool()`. The old name `cval` is now
  deprecated.
* Added `augmenters.color._AbstractColorQuantization`. #347
* Added `augmenters.color.KMeansColorQuantization` and corresponding
  `augmenters.color.quantize_colors_kmeans()`. Both deal with quantizing
  similar colors using k-Means clustering. #347
    * Added a check script for `KMeansColorQuantization` under
      `checks/check_kmeans_color_quantization.py`. #347
* Added `augmenters.color.UniformColorQuantization` and corresponding
  `augmenters.color.quantize_colors_uniform()`. Both deal with quantizing
  similar colors using k-Means clustering. #347
    * Added a check script for `UniformColorQuantization` under
      `checks/check_uniform_color_quantization.py`. #347
* Added `imgaug.imgaug.normalize_random_state()`.
* Added `imgaug.augmenters.segmentation._ensure_image_max_size()`.
* Added `imgaug.augmenters.segmentation.PointsSamplerIf`. An interface for
  classes used for sampling (usually random) coordinate arrays on images.
* Added `imgaug.augmenters.segmentation._verify_sample_points_images()`.
* Added `imgaug.augmenters.segmentation.RegularGridPointsSampler`. A class
  used to generate regular grids of `rows x columns` points on images.
* Added `imgaug.augmenters.segmentation.RelativeRegularGridPointsSampler`.
  Similar to `RegularGridPointsSampler`, but number of rows/columns is set
  as fractions of image sizes, leading to more rows/columns for larger images.
* Added `imgaug.augmenters.segmentation.DropoutPointsSampler`. A class
  used to randomly drop `p` percent of all coordinates sampled by another
  another points sampler.
* Added `imgaug.augmenters.segmentation.SubsamplingPointsSampler`. A class
  that ensures that another points sampler does not produce more than
  `N` points by subsampling a random subset of the produced points if `N`
  is exceeded.
* Added `imgaug.augmenters.segmentation.segment_voronoi()`. A function that
  converts an image into a voronoi image, i.e. averages the colors within
  voronoi cells placed on the image.
    * Also added in the same module the functions
      `_match_pixels_with_voronoi_cells()`, `_generate_pixel_coords()`,
      `_compute_avg_segment_colors()`, `_render_segments()`.
* Added `imgaug.augmenters.segmentation.Voronoi`. An augmenter that converts
  an image to a voronoi image. 
    * Added a check script for `Voronoi` in `checks/check_voronoi.py`.

## Fixes
 
* Fixed an issue with `Polygon.clip_out_of_image()`,
  which would lead to exceptions if a polygon had overlap with an image,
  but not a single one of its points was inside that image plane. 
* Fixed `multicore` methods falsely not accepting
  `augmentables.batches.UnnormalizedBatch`.
* `Rot90` now uses subpixel-based coordinate remapping.
  I.e. any coordinate `(x, y)` will be mapped to `(H-y, x)` for a rotation by
  90deg.
  Previously, an integer-based remapping to `(H-y-1, x)` was used.
  Coordinates are e.g. used by keypoints, bounding boxes or polygons.
* `augmenters.arithmetic.Invert`
    * [rarely breaking] If `min_value` and/or `max_value` arguments were
      set, `uint64` is no longer a valid input array dtype for `Invert`.
      This is due to a conversion to `float64` resulting in loss of resolution.
    * Fixed `Invert` in rare cases restoring dtypes improperly.
* Fixed `dtypes.gate_dtypes()` crashing if the input was one or more numpy
  scalars instead of numpy arrays or dtypes.
* Fixed `augmenters.geometric.PerspectiveTransform` producing invalid
  polygons (more often with higher `scale` values). #338
* Fixed errors caused by `external/poly_point_isect_py2py3.py` related to
  floating point inaccuracies (changed an epsilon from `1e-10` to `1e-4`,
  rounded some floats). #338
* Fixed `Superpixels` breaking when a sampled `n_segments` was `<=0`.
  `n_segments` is now treated as `1` in these cases.
* Fixed `ReplaceElementwise` both allowing and disallowing dtype `int64`. #346
* Fixed `BoundingBox.deepcopy()` creating only shallow copies of labels. #356
* Fixed `dtypes.change_dtypes_()` #366
    * Fixed argument `round` being ignored if input images were a list.
    * Fixed failure if input images were a list and dtypes a single numpy
      dtype function.
* Fixed `dtypes.get_minimal_dtype()` failing if argument `arrays` contained
  not *exactly* two items. #366


# 0.2.9

This update mainly covers the following topics:
* Moved classes/methods related to augmentable data to their own modules.
* Added polygon augmentation methods.
* Added line strings and line string augmentation methods.
* Added easier augmentation interface.

## New 'augmentables' Modules

For the Polygon and Line String augmentation, new classes and methods had to be
added. The previous file for that was `imgaug/imgaug.py`, which however was
already fairly large. Therefore, all classes and methods related to augmentable
data were split off and moved to `imgaug/augmentables/<type>.py`. The new
modules and their main contents are:
* `imgaug.augmentables.batches`: Contains `Batch`, `UnnormalizedBatch`.
* `imgaug.augmentables.utils`: Contains utility functions.
* `imgaug.augmentables.bbs`: Contains `BoundingBox`, `BoundingBoxesOnImage`.
* `imgaug.augmentables.kps`: Contains `Keypoint`, `KeypointsOnImage`.
* `imgaug.augmentables.polys`: Contains `Polygon`, `PolygonsOnImage`.
* `imgaug.augmentables.lines`: Contains `LineString`, `LineStringsOnImage`.
* `imgaug.augmentables.heatmaps`: Contains `HeatmapsOnImage`.
* `imgaug.augmentables.segmaps`: Contains `SegmentationMapOnImage`.

Currently, all augmentable classes can still be created via `imgaug.<type>`,
e.g. `imgaug.BoundingBox` still works.

Changes related to the new modules:
* Moved `Keypoint`, `KeypointsOnImage` and `imgaug.imgaug.compute_geometric_median` to `augmentables/kps.py`.
* Moved `BoundingBox`, `BoundingBoxesOnImage` to `augmentables/bbs.py`.
* Moved `Polygon`, `PolygonsOnImage` and related classes/functions to `augmentables/polys.py`.
* Moved `HeatmapsOnImage` to `augmentables/heatmaps.py`.
* Moved `SegmentationMapOnImage` to `augmentables/segmaps.py`.
* Moved `Batch` to `augmentables/batches.py`.
* Added module `imgaug.augmentables.utils`.
    * Added function `normalize_shape()`.
    * Added function `project_coords()`.
* Moved line interpolation functions `_interpolate_points()`, `_interpolate_point_pair()` and `_interpolate_points_by_max_distance()` to `imgaug.augmentables.utils` and made them public functions.
* Refactored `__init__()` of `PolygonsOnImage`, `BoundingBoxesOnImage`, `KeypointsOnImage` to make use of `imgaug.augmentables.utils.normalize_shape()`.
* Refactored `KeypointsOnImage.on()` to use `imgaug.augmentables.utils.normalize_shape()`.
* Refactored `Keypoint.project()` to use `imgaug.augmentables.utils.project_coords()`.

## Polygon Augmentation

Polygons were already part of `imgaug` for quite a while, but couldn't be
augmented yet. This version adds methods to perform such augmentations.
It also makes some changes to the `Polygon` class, see the list of changes
below.

Example for polygon augmentation:
```python
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

image = ia.quokka(size=0.2)
psoi = PolygonsOnImage([
    Polygon([(0, 0), (20, 0), (20, 20)])
], shape=image.shape)

image_aug, psoi_aug = iaa.Affine(rotate=45).augment(
    images=[image],
    polygons=[psoi]
)
``` 

See [imgaug-doc/notebooks](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
for a jupyter notebook with many more examples.

Changes related to polygon augmentation:
* Added `_ConcavePolygonRecoverer` to `imgaug.augmentables.polys`.
* Added `PolygonsOnImage` to `imgaug.augmentables.polys`.
* Added polygon augmentation methods:
    * Added `augment_polygons()` to `Augmenter`.
    * Added `_augment_polygons()` to `Augmenter`.
    * Added `_augment_polygons_as_keypoints()` to `Augmenter`.
    * Added argument `polygons` to `imgaug.augmentables.batches.Batch`.
    * Added attributes `polygons_aug` and `polygons_unaug` to `imgaug.augmentables.batches.Batch`.
    * Added polygon handling to `Augmenter.augment_batches()`.
* Added property `Polygon.height`.
* Added property `Polygon.width`.
* Added method `Polygon.to_keypoints()`.
* Added optional drawing of corner points to `Polygon.draw_on_image()` and `PolygonsOnImage.draw_on_image()`.
* Added argument `raise_if_too_far_away=True` to `Polygon.change_first_point_by_coords()`.
* Added `imgaug.quokka_polygons()` function to generate example polygon data.
* [rarely breaking] `Polygon.draw_on_image()`, `PolygonsOnImage.draw_on_image()`
    * Refactored to make partial use `LineString` methods.
    * Added arguments `size` and `size_perimeter` to control polygon line thickness.
    * Renamed arguments `alpha_perimeter` to `alpha_line`, `color_perimeter` to `color_line` to align with `LineStrings`.
    * Renamed arguments `alpha_fill` to `alpha_face` and `color_fill` to `color_face`.
* [rarely breaking] Changed the output of `Polygon.clip_out_of_image()` from `MultiPolygon` to `list` of `Polygon`.
  This breaks for anybody who has already used `Polygon.clip_out_of_image()`.
* Changed `Polygon.exterior_almost_equals()` to accept lists of tuples as argument `other_polygon`.
* Changed arguments `color` and `alpha` in `Polygon.draw_on_image()` and `PolygonsOnImage.draw_on_image()` to represent
  the general color and alpha of the polygon. The colors/alphas of the inner area, perimeter and points are derived from
  `color` and `alpha` (unless `color_inner`, `color_perimeter` or `color_points` are set (analogous for alpha)).
* Refactored `Polygon.project()` to use `LineString.project()`.
* Refactored `Polygon.shift()` to use `LineString.shift()`.
* [rarely breaking] `Polygon.exterior_almost_equals()`, `Polygon.almost_equals()`
    * Refactored to make use of `LineString.coords_almost_equals()`.
    * Renamed argument `interpolate` to `points_per_edge`.
    * Renamed argument `other_polygon` to `other`.
* Renamed `color_line` to `color_lines`, `alpha_line` to `alpha_lines` in `Polygon.draw_on_image()` and `PolygonsOnImage.draw_on_image()`.
* Fixed `Polygon.clip_out_of_image(image)` not handling `image` being a tuple.
* Fixed `Polygon.is_out_of_image()` falsely only checking the corner points of the polygon.

## LineString Augmentation

This version adds Line String augmentation. Line Strings are simply lines made
up of consecutive corner points that are connected by straight lines.
Line strings have similarity with polygons, but do not have a filled inner area
and are not closed (i.e. first and last coordinate differ).

Similar to other augmentables, line string are represented with the classes
`LineString(<iterable of xy-coords>)` and
`LineStringsOnImage(<iterable of LineString>, <shape of image>)`.
They are augmented e.g. via `Augmenter.augment_line_strings(<iterable of LineStringsOnImage>)`
or `Augmenter.augment(images=..., line_strings=...)`.  

Example:
```python
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage

image = ia.quokka(size=0.2)
lsoi = LineStringsOnImage([
    LineString([(0, 0), (20, 0), (20, 20)])
], shape=image.shape)

image_aug, lsoi_aug = iaa.Affine(rotate=45).augment(
    images=[image],
    line_strings=[lsoi]
)
``` 

See [imgaug-doc/notebooks](https://github.com/aleju/imgaug-doc/tree/master/notebooks)
for a jupyter notebook with many more examples.

## Simplified Augmentation Interface

Augmentation of different data corresponding to the same image(s) has been
a bit convoluted in the past, as each data type had to be augmented on its own.
E.g. to augment an image and its bounding boxes, one had to first switch the
augmenters to deterministic mode, then augment the images, then the bounding
boxes. This version adds methods that perform these steps in one call.
Specifically, `Augmenter.augment(...)` is used for that, which has the alias
`Augmenter.__call__(...)`. One argument can be used for each augmentable,
e.g. `bounding_boxes=<bounding box data>`. 
Example:
```python
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

image = ia.quokka(size=0.2)
kpsoi = KeypointsOnImage([Keypoint(x=0, y=10), Keypoint(x=10, y=5)],
                         shape=image.shape)

image_aug, kpsoi_aug = iaa.Affine(rotate=(-45, 45)).augment(
    image=image,
    keypoints=kpsoi
)
```
This will automatically make sure that image and keypoints are rotated by
corresponding amounts.

Normalization methods have been added to that class, which allow it to
process many more different inputs than just variations of `*OnImage`.
Example:
```python
import imgaug as ia
import imgaug.augmenters as iaa

image = ia.quokka(size=0.2)
kps = [(0, 10), (10, 5)]

image_aug, kps_aug = iaa.Affine(rotate=(-45, 45)).augment(
    image=image, keypoints=kps)
```
Examples for other inputs that are automatically handled by `augment()`:
* Integer arrays as segmentation maps. 
* Float arrays for heatmaps.
* `list([N,4] ndarray)` for bounding boxes. (One list for images,
  then `N` bounding boxes in `(x1,y1,x2,y2)` form.) 
* `list(list(list(tuple)))` for line strings. (One list for images,
  one list for line strings on the image, one list for coordinates within
  the line string. Each tuple must contain two values for xy-coordinates.)
* `list(list(imgaug.augmentables.polys.Polygon))` for polygons.
  Note that this "skips" `imgaug.augmentables.polys.PolygonsOnImage`.

In **python <3.6**, `augment()` is limited to a maximum of two
inputs/outputs *and* if two inputs/outputs are used, then one of them must be
image data *and* such (augmented) image data will always be returned first,
independent of the argument's order.
E.g. `augment(line_strings=<data>, polygons=<data>)` would be invalid due to
not containing image data. `augment(polygons=<data>, images=<data>)` would
still return the images first, even though they are the second argument.

In **python >=3.6**, `augment()` may be called with more than two
arguments and will respect their order.
Example:
```python
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

image = ia.quokka(size=0.2)
kps = [(0, 10), (10, 5)]
heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
rotate = iaa.Affine(rotate=(-45, 45))

heatmaps_aug, images_aug, kps_aug = rotate(
    heatmaps=[heatmap],
    images=[image],
    keypoints=[kps]
)
```

To use more than two inputs/outputs in python <3.6, add the argument `return_batch=True`,
which will return an instance of `imgaug.augmentables.batches.UnnormalizedBatch`.

Changes related to the augmentation interface:
* Added `Augmenter.augment()` method.
* Added `Augmenter.augment_batch()` method.
    * This method is now called by `Augmenter.augment_batches()` and multicore routines.
* Added `imgaug.augmentables.batches.UnnormalizedBatch`.
* Added module `imgaug.augmentables.normalization` for data normalization routines.
* Changed `augment_batches()`:
  * Accepts now `UnnormalizedBatch` as input. It is automatically normalized before augmentation and unnormalized afterwards.
    This allows to use `Batch` instances with non-standard datatypes.
  * Accepts now single instances of `Batch` (and `UnnormalizedBatch`).
  * The input may now also be a generator.
  * The input may now be any iterable instead of just list (arrays or strings are not allowed).
* Marked support for non-`Batch` (and non-`UnnormalizedBatch`) inputs to `augment_batches()` as deprecated.
* Refactored `Batch.deepcopy()`
    * Does no longer verify attribute datatypes.
    * Allows now to directly change attributes of created copies, e.g. via `batch.deepcopy(images_aug=...)`.

## Other Additions, Changes and Refactorings

Keypoint augmentation
* Added method `Keypoint.draw_on_image()`.
* [mildly breaking] Added an `alpha` argument to `KeypointsOnImage.draw_on_image()`. This can break code that relied on
  the order of arguments of the method (though will usually only have visual consequences).
* `KeypointsOnImage` and `Keypoint` copying:
    * Added optional arguments `keypoints` and `shape` to `KeypointsOnImage.deepcopy()`.
    * Added optional arguments `keypoints` and `shape` to `KeypointsOnImage.copy()`.
    * Added method `Keypoint.copy()`.
    * Added method `Keypoint.deepcopy()`.
        * Refactored methods in `Keypoint` to use `deepcopy()` to create copies of itself (instead of instantiating new instances via `Keypoint(...)`).
    * `KeypointsOnImage.deepcopy()` now uses `Keypoint.deepcopy()` to create Keypoint copies, making it more flexible.
    * Refactored `KeypointsOnImage` to use `KeypointsOnImage.deepcopy()` in as many methods as possible to create copies of itself.
    * Refactored `Affine`, `AffineCv2`, `PiecewiseAffine`, `PerspectiveTransform`, `ElasticTransformation`, `Rot90` to use `KeypointsOnImage.deepcopy()` and `Keypoint.deepcopy()` during keypoint augmentation.
* Changed `Keypoint.draw_on_image()` to draw a rectangle for the keypoint so long as *any* part of that rectangle is within the image plane.
  (Previously, the rectangle was only drawn if the integer xy-coordinate of the point was inside the image plane.)
* Changed `KeypointsOnImage.draw_on_image()` to raise an error if an input image has shape `(H,W)`.
* Changed `KeypointsOnImage.draw_on_image()` to handle single-number inputs for `color`.
* `KeypointsOnImage.from_coords_array()`
    * Marked as deprecated.
    * Renamed to `from_xy_array()`.
    * Renamed arg `coords` to `xy`.
    * Changed the method from `staticmethod` to `classmethod`.
    * Refactored to make code simpler.
* `KeypointsOnImage.get_coords_array()`
    * Marked as deprecated.
    * Renamed to `to_xy_array()`.
* Refactored `KeypointsOnImage.draw_on_image()` to use `Keypoint.draw_on_image()`.

Heatmap augmentation
* Changed `Affine`, `PiecewiseAffine`, `ElasticTransformation` to always use `order=3` for heatmap augmentation.
* Changed check in `HeatmapsOnImage` that validates whether the input array is within the desired value range `[min_value, max_value]` 
  from a hard exception to a soft warning (with clipping). Also improved the error message a bit.

Deprecation warnings:
* Added `imgaug.imgaug.DeprecationWarning`. The builtin python `DeprecationWarning` is silent since 2.7, which is why now a separate deprecation warning is used.
* Added `imgaug.imgaug.warn_deprecated()`.
    * Refactored deprecation warnings to use this function.
* Added `imgaug.imgaug.deprecated` decorator.
    * Refactored deprecation warnings to use this decorator.

Bounding Boxes:
* Added to `BoundingBox.extract_from_image()` the arguments `pad` and `pad_max`.
* Changed `BoundingBox.contains()` to also accept `Keypoint`.
* Changed `BoundingBox.project(from, to)` to also accept images instead of shapes.
* Renamed argument `thickness` in `BoundingBox.draw_on_image()` to `size` in order to match the name used for keypoints, polygons and line strings.
  The argument `thickness` will still be accepted, but raises a deprecation warning.
* Renamed argument `thickness` in `BoundingBoxesOnImage.draw_on_image()` to `size` in order to match the name used for keypoints, polygons and line strings.
  The argument `thickness` will still be accepted, but raises a deprecation warning.
* Refactored `BoundingBox` to reduce code repetition.
* Refactored `BoundingBox.extract_from_image()`. Improved some code fragments that looked wrong.
* Refactored `BoundingBoxesOnImage.draw_on_image()` to improve efficiency by evading unnecessary array copies.

Other:
* [rarely breaking] Added arguments `cval` and `mode` to `PerspectiveTransform` (PR #301).
  This breaks code that relied on the order of the arguments and used `keep_size`, `name`, `deterministic` or `random_state` as positional arguments.
* Added `dtypes.clip_()` function.
* Added function `imgaug.imgaug.flatten()` that flattens nested lists/tuples.
* Changed `PerspectiveTransform` to ensure minimum height and width of output images (by default `2x2`).
  This prevents errors in polygon augmentation (possibly also in keypoint augmentation).
* Refactored `imgaug.augmenters.blend.blend_alpha()` to no longer enforce a channel axis for foreground and background image.
* Refactored `imgaug/parameters.py` to reorder classes within the file.
* Re-allowed numpy 1.16 in `requirements.txt`.


## Fixes

* Fixed possible crash in `blend.blend_alpha()` if dtype numpy.float128 does not exist.
* Fixed a crash in `ChangeColorspace` when `cv2.COLOR_Lab2RGB` was actually called `cv2.COLOR_LAB2RGB` in the local OpenCV installation (analogous for BGR). (PR #263)
* Fixed `ReplaceElementwise` always sampling replacement per channel.
* Fixed an error in `draw_text()` due to arrays that could not be set to writeable after drawing the text via PIL.  
* Fixed errors in docstring of `parameters.Subtract`.
* Fixed a division by zero bug in `angle_between_vectors()`.
* Augmentation of empty `KeypointsOnImage` instances
    * Fixed `Rot90` not changing `KeypointsOnImage.shape` if `.keypoints` was empty.
    * Fixed `Affine` not changing `KeypointsOnImage.shape` if `.keypoints` was empty.
    * Fixed `PerspectiveTransform` not changing `KeypointsOnImage.shape` if `.keypoints` was empty.
    * Fixed `Resize` not changing `KeypointsOnImage.shape` if `.keypoints` was empty.
    * Fixed `CropAndPad` not changing `KeypointsOnImage.shape` if `.keypoints` was empty. (Same for `Crop`, `Pad`.)
    * Fixed `PadToFixedSize` not changing `KeypointsOnImage.shape` if `.keypoints` was empty.
    * Fixed `CropToFixedSize` not changing `KeypointsOnImage.shape` if `.keypoints` was empty.
    * Fixed `KeepSizeByResize` not changing `KeypointsOnImage.shape` if `.keypoints` was empty.
* Fixed `Affine` heatmap augmentation producing arrays with values outside the range `[0.0, 1.0]` when `order` was set to `3`.
* Fixed `PiecewiseAffine` heatmap augmentation producing arrays with values outside the range `[0.0, 1.0]` when `order` was set to `3`.
* Fixed assert in `SegmentationMapOnImage` falsely checking if max class index is `<= nb_classes` instead of `< nb_classes`. 
* Fixed an issue in `dtypes.clip_to_value_range_()` and `dtypes.restore_dtypes_()` causing errors when clip value range exceeded array dtype's value range.
* Fixed an issue in `dtypes.clip_to_value_range_()` and `dtypes.restore_dtypes_()` when the input array was scalar, i.e. had shape `()`.
* Fixed a Permission Denied error when using `JpegCompression` on windows (possibly also affected other systems). #297


# 0.2.8

This update focused on extending and documenting the library's dtype support, improving the performance and reworking multicore augmentation.


## dtype support

Previous versions of `imgaug` were primarily geared towards `uint8`.
In this version, all augmenters and helper functions were refactored to be more tolerant towards non-uint8 dtypes.
Additionally all augmenters were tested with non-uint8 dtypes and an overview of the expected support-level
is now listed in the documentation on page [dtype support](https://imgaug.readthedocs.io/en/latest/source/dtype_support.html).
Further details are listed in the docstrings of each individual augmenter or helper function.


## Performance Improvements

Below are some numbers for the achieved performance improvements compared to 0.2.7.
The measurements were taken using realistic 224x224x3 uint8 images and batch size 128.
The percentage values denote the increase in bandwidth (i.e. mbyte/sec) of the respective
augmenter given the described input. Improvements for smaller images, smaller batch sizes
and non-uint8 dtypes may differ. Augmenters with less than roughly 10% improvement are not
listed. While the numbers here are exact, there is some measurement error involved as they
were calculated based on a rather low number of 100 repetitions.

* Sequential (with 2x Noop as children) +184% to +276%
* SomeOf (with 3x Noop as children) +24% to +49%
* OneOf (with 3x Noop as children) +21%
* Sometimes (with Noop as child) +23%
* WithChannels +32%
* Add +216%
* AddElementwise +49%
* AdditiveGaussianNoise +26%
* AdditiveLaplaceNoise +20%
* AdditivePoissonNoise +18%
* Multiply +206%
* MultiplyElementwise +74%
* Dropout +154%
* CoarseDropout +246%
* ReplaceElementwise +119%
* ImpulseNoise +333%
* SaltAndPepper +184%
* CoarseSaltAndPepper +227%
* Salt +204%
* CoarseSalt +260%
* Pepper +208%
* CoarsePepper +276%
* Invert +1192%
* GaussianBlur +885%
* AddToHueAndSaturation +48%
* GammaContrast +2988%
* SigmoidContrast +519%
* LogContrast +1048%
* LinearContrast +448%
* Convolve +47%
* Sharpen +29%
* Emboss +18%
* EdgeDetect +41%
* DirectedEdgeDetect +53%
* Fliplr +75%
* Flipud +25%
* Affine +7% to +33%
* ElasticTransformation +650 to +680%
* CropAndPad +30% to +77% (from improved padding)
* Pad +40 to +140%
* PadToFixedSize +288%
* KeepSizeByResize (with CropToFixedSize as child) +58%
* Snowflakes +44%
* SnowflakesLayer +42%


## multicore augmentation

The implementation for multicore augmentation was completely rewritten and is now a wrapper around python's `multiprocessing.Pool`. Compared to the old version, it is by far less fragile and faster. It is also easier to use. Every augmenter now offers a simple `pool()` method, which can be used to quickly spawn a pool of child workers on multiple CPU cores. Example:
```python
aug = iaa.PiecewiseAffine(0.2)
with aug.pool(processes=-1, seed=123) as pool:
    batches_aug = pool.imap_batches(batches_generator, chunksize=32)
    for batch_aug in batches_aug:
        # do something
```
Here, `batches_generator` is a generator that yields instances of `imgaug.Batch`, e.g. something like `imgaug.Batch(images=<numpy array>, keypoints=[imgaug.KeypointsOnImage(...), imgaug.KeypointsOnImage(...), ...])`. The arguement `processes=-1` spawns `N-1` workers, where `N` is the number of CPU cores (includes hyperthreads).

Note that `Augmenter.augment_batches(batches, background=True)` still works and now uses the above `pool()` method.


## imgaug.imgaug

* Added constants that control the min/max values for seed generation
* Improved performance of `pad()`
    * this change also improves the performance of:
        * `imgaug.imgaug.pad_to_aspect_ratio()`,
        * `imgaug.imgaug.HeatmapsOnImage.pad()`,
        * `imgaug.imgaug.HeatmapsOnImage.pad_to_aspect_ratio()`,
        * `imgaug.imgaug.SegmentationMapOnImage.pad()`,
        * `imgaug.imgaug.SegmentationMapOnImage.pad_to_aspect_ratio()`,
        * `imgaug.augmenters.size.PadToFixedSize`,
        * `imgaug.augmenters.size.Pad`,
        * `imgaug.augmenters.size.CropAndPad`
* Changed `imshow()` to explicitly make the plot figure size dependent on the input image size.
* Refactored `SegmentationMapOnImage` to have simplified dtype handling in `__init__`
* Fixed an issue with `SEED_MAX_VALUE` exceeding the `int32` maximum on some systems, causing crashes related to
  RandomState.
* Moved BatchLoader to `multicore.py` and replaced the class with an alias pointing to `imgaug.multicore.BatchLoader`.
* Moved BackgroundAugmenter to `multicore.py` and replaced the class with an alias pointing to `imgaug.multicore.BatchLoader`.
* Renamed `HeatmapsOnImage.scale()` to `HeatmapsOnImage.resize()`.
* Marked `HeatmapsOnImage.scale()` as deprecated.
* Renamed `SegmentationMapOnImage.scale()` to `SegmentationMapOnImage.resize()`.
* Marked `SegmentationMapOnImage.scale()` as deprecated.
* Renamed `BoundingBox.cut_out_of_image()` to `BoundingBox.clip_out_of_image()`.
* Marked `BoundingBox.cut_out_of_image()` as deprecated.
* Renamed `BoundingBoxesOnImage.cut_out_of_image()` to `BoundingBoxesOnImage.clip_out_of_image()`.
* Marked `BoundingBoxesOnImage.cut_out_of_image()` as deprecated.
* Marked `Polygon.cut_out_of_image()` as deprecated. (The analogous clip function existed already.)
* Renamed in `imgaug.Batch` the attributes storing input data `<attribute>_unaug`, e.g. `imgaug.Batch.images` to `imgaug.Batch.images_unaug` or `imgaug.Batch.keypoints` to `imgaug.Batch.keypoints_unaug`. The old attributes are still accessible, but will raise a DeprecatedWarning.


## imgaug.multicore

* Created this file.
* Moved `BatchLoader` here from `imgaug.py`.
* Moved `BackgroudAugmenter` here from `imgaug.py`.
* Marked `BatchLoader` as deprecated.
* Marked `BackgroundAugmenter` as deprecated.
* Added class `Pool`. This is the new recommended way for multicore augmentation. `BatchLoader`/`BackgroundAugmenter` should not be used anymore. Example:
  ```python
  import imgaug as ia
  from imgaug import augmenters as iaa
  from imgaug import multicore
  import numpy as np
  aug = iaa.Add(1)
  images = np.zeros((16, 128, 128, 3), dtype=np.uint8)
  batches = [ia.Batch(images=np.copy(images)) for _ in range(100)]
  with multicore.Pool(aug, processes=-1, seed=2) as pool:
      batches_aug = pool.map_batches(batches, chunksize=8)
  print(np.sum(batches_aug[0].images_aug[0]))
  ```
  The example starts a pool with N-1 workers (N=number of CPU cores) and augments 100 batches using these workers.
  Use `imap_batches()` to feed in and get out a generator.


## imgaug.parameters

* Added `TruncatedNormal`
* Added `handle_discrete_kernel_size_param()`
* Improved dtype-related interplay of `FromLowerResolution` and `imresize_many_images()`
* Improved performance for sampling from `Deterministic` by about 2x
* Improved performance for sampling from `Uniform` with `a == b`
* Improved performance for sampling from `DiscreteUniform` with `a == b`
* Improved performance for sampling from `Laplace` with `scale=0`
* Improved performance for sampling from `Normal` with `scale=0`
* Improved performance of `Clip` and improved code style
* Refactored `float check in force_np_float_dtype()`
* Refactored `RandomSign`
* Refactored various unittests to be more flexible with regards to returned dtypes
* Refactored `StochasticParameter.draw_distribution_graph()` to use internally tempfile-based drawing. Should result in higher-quality outputs.
* Refactored unittest for `draw_distributions_grid()` to improve performance
* Fixed in `draw_distributions_grid()` a possible error from arrays with unequal shapes being combined to one array
* Fixed a problem with `Sigmoid` not returning floats
* Fixed noise produced by `SimplexNoise` having values below 0.0 or above 1.0
* Fixed noise produced by `SimplexNoise` being more biased towards 0 than it should be


## imgaug.dtypes

* Added new file `imgaug/dtypes.py` and respective test file `test_dtypes.py`.
* Added `clip_to_dtype_value_range_()`
* Added `get_value_range_of_dtype()`
* Added `promote_array_dtypes_()`
* Added `get_minimal_dtype()`
* Added `get_minimal_dtypes_for_values()`
* Added `get_minimal_dtype_by_value_range()`
* Added `restore_dtypes_()`
* Added `gate_dtypes()`
* Added `increase_array_resolutions()`
* Added `copy_dtypes_for_restore()`


## imgaug.augmenters.meta

* Added `estimate_max_number_of_channels()`
* Added `copy_arrays()`
* Added an optional parameter `default` to `handle_children_lists()`
* Enabled `None` as arguments for `Lambda` and made all arguments optional
* Enabled `None` as arguments for `AssertLambda` and made all arguments optional
* Improved dtype support of `AssertShape`
* Improved dtype support of `AssertLambda`
* Improved dtype support of `Lambda`
* Improved dtype support of `ChannelShuffle`
* Improved dtype support of `WithChannels`
* Improved dtype support of `Sometimes`
* Improved dtype support of `SomeOf`
* Improved dtype support of `Sequential`
* Improved dtype support of `Noop`
* [breaking, mostly internal] Removed `restore_augmented_images_dtypes()`
* [breaking, mostly internal] Removed `restore_augmented_images_dtypes_()`
* [breaking, mostly internal] Removed `restore_augmented_images_dtype()`
* [breaking, mostly internal] Removed `restore_augmented_images_dtype_()`
* [breaking, mostly internal] Refactored `Augmenter.augment_images()` and `Augmenter._augment_images()` to default
  `hooks` to `None`
    * This will affect any custom implemented augmenters that try to access the hooks argument.
* [breaking, mostly internal] Refactored `Augmenter.augment_heatmaps()` and `Augmenter._augment_heatmaps()` to default
  `hooks` to `None`
    * Same as above for images.
* [breaking, mostly internal] Refactored `Augmenter.augment_keypoints()` and `Augmenter._augment_keypoints()` to default
  `hooks` to None
    * Same as above for images.
* [breaking, mostly internal] Improved performance of image augmentation for augmenters with children
    * For calls to `augment_image()`, the validation, normalization and copying steps are skipped if the call is
      a child call (e.g. a `Sequential` calling `augment_images()` on a child `Add`). Hence, such child calls augment
      now fully in-place (the top-most call still creates a copy though, so from the user perspective nothing
      changes). Custom implemented augmenters that rely on child calls to `augment_images()` creating copies will
      break from this change.
      For an example `Sequential` containing two `Noop` augmenters, this change improves the performance by roughly 2x
* [breaking, mostly internal] Improved performance of heatmap augmentation for augmenters with children
    * Same as above for images.
    * Speedup is around 2-3x for an exemplary `Sequential` containing two `Noop`s.
    * This will similarly affect segmentation map augmentation too.
* [breaking, mostly internal] Improved performance of keypoint augmentation for augmenters with children
    * Same as above for images.
    * Speedup is around 1.5-2x for an exemplary `Sequential` containing two Noops.
    * This will similarly affect bounding box augmentation.
* [critical] Fixed a bug in the augmentation of empty `KeypointsOnImage` instances that would lead image and keypoint
  augmentation to be un-aligned within a batch after the first empty `KeypointsOnImage` instance. (#231)
* Added `pool()` to `Augmenter`. This is a helper to start a `imgaug.multicore.Pool` via `with augmenter.pool() as pool: ...`.
* Refactored `Augmenter.augment_batches(..., background=True)` to use `imgaug.multicore.Pool`.
* Changed `to_deterministic()` in `Augmenter` and various child classes to derive its new random state from the augmenter's local random state instead of the global random state.
* Enabled support for non-list `HeatmapsOnImage` inputs in `Augmenter.augment_heatmaps()`. (Before, only lists were supported.)
* Enabled support for non-list `SegmentationMapOnImage` inputs in `Augmenter.augment_segmentation_maps()`. (Before, only lists were supported.)
* Enabled support for non-list `KeypointsOnImage` inputs in `Augmenter.augment_keypoints()`. (Before, only lists were supported.)
* Enabled support for non-list `BoundingBoxesOnImage` inputs in `Augmenter.augment_bounding_boxes()`. (Before, only lists were supported.)


## imgaug.augmenters.arithmetic

* `ContrastNormalization` is now an alias for `LinearContrast`
* Restricted `JpegCompression` to uint8 inputs. Other dtypes will now produce errors early on.
* Changed in `Add` the parameter `value` to be continuous and removed its `value_range`


## imgaug.augmenters.blend

* Renamed `imgaug.augmenters.overlay` to `imgaug.augmenters.blend`. Functions and classes in `imgaug.augmenters.overlay` are still accessible, but will now raise a DeprecatedWarning.
* Added `blend_alpha()`.
* Refactored `Alpha` to be simpler and use `blend_alpha()`.
* Fixed `Alpha` not having its own `__str__` method.
* Improved dtype support of `AlphaElementwise`.


## imgaug.augmenters.blur

* Added function `blur_gaussian()`


## imgaug.augmenters.color

* Added `Lab2RGB` and `Lab2BGR` to `ChangeColorspace`
* Refactored the main loop in `AddToHueAndSaturation` to make it simpler and faster
* Fixed `ChangeColorspace` not being able to convert from RGB/BGR to Lab/Luv


## imgaug.augmenters.contrast

* Added `AllChannelsCLAHE`
* Added `CLAHE`
* Added `AllChannelsHistogramEqualization`
* Added `HistogramEqualization`
* Added `_IntensityChannelBasedApplier`
* Added function `adjust_contrast_gamma()`
* Added function `adjust_contrast_sigmoid()`
* Added function `adjust_contrast_log()`
* Refactored random state handling in `_ContrastFuncWrapper`
* [breaking, internal] Removed `_PreserveDtype`
* [breaking, internal] Renamed `_adjust_linear` to `adjust_contrast_linear()`


## imgaug.augmenters.convolutional

* Refactored `AverageBlur` to have improved random state handling
* Refactored `GaussianBlur` to only overwrite input images when that is necessary
* Refactored `GaussianBlur` to have a simplified main loop
* Refactored `AverageBlur` to have a simplified main loop
* Refactored `MedianBlur` to have a simplified main loop
* Refactored `BilateralBlur` to have a simplified main loop
* Improved dtype support of `GaussianBlur`
* Improved dtype support of `AverageBlur`
* Improved dtype support of `Convolve`


## imgaug.augmenters.flip

* Improved dtype support of `Fliplr`
* Improved dtype support of `Flipud`
* Refactored `Fliplr` main loop to be more elegant and tolerant
* Refactored `Flipud` main loop to be more elegant and tolerant
* Added alias `HorizontalFlip` for `Fliplr`.
* Added alias `VerticalFlip` for `Flipud`.


## imgaug.augmenters.geometric

* `ElasticTransformation`
    * [breaking, mostly internal] `generate_indices()` now returns only the pixelwise shift as a tuple of x and y
    * [breaking, mostly internal] `generate_indices()` has no longer a `reshape` argument
    * [breaking, mostly internal] `renamed generate_indices()` to `generate_shift_maps()`
    * [breaking, mostly internal] `map_coordinates()` now expects to get the pixelwise shift as its input, instead of
      the target coordinates


## imgaug.augmenters.segmentation

* Improved dtype support of `Superpixels`


## imgaug.augmenters.size

* Removed the restriction to `uint8` in `Scale`. The augmenter now supports the same dtypes as `imresize_many_images()`.
* Fixed missing pad mode `mean` in `Pad` and `CropAndPad`.
* Improved error messages related to pad mode.
* Improved and fixed docstrings of `CropAndPad`, `Crop`, `Pad`.
* Renamed `Scale` to `Resize`.
* Marked `Scale` as deprecated.


## other

* Improved descriptions of the library in `setup.py`
* `matplotlib` is now an optional dependency of the library and loaded lazily when needed
* `Shapely` is now an optional dependency of the library and loaded lazily when needed
* `opencv-python` is now a dependency of the library
* `setup.py` no longer enforces `cv2` to be installed (to allow installing libraries in random order)
* Minimum required `numpy` version is now 1.15
