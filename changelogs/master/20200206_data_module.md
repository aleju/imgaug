# Add New `data` Module #606

This patch moves the example data functions from `imgaug.imgaug` to
the new module `imgaug.data`.

Add Modules:
* `imgaug.data`

Add Functions:
* `imgaug.data.quokka`
* `imgaug.data.quokka_square`
* `imgaug.data.quokka_heatmap`
* `imgaug.data.quokka_segmentation_map`
* `imgaug.data.quokka_keypoints`
* `imgaug.data.quokka_bounding_boxes`
* `imgaug.data.quokka_polygons`

Deprecated Functions:
* `imgaug.imgaug.quokka`.
  Use `imgaug.data.quokka` instead.
* `imgaug.imgaug.quokka_square`.
  Use `imgaug.data.quokka_square` instead.
* `imgaug.imgaug.quokka_heatmap`.
  Use `imgaug.data.quokka_heatmap` instead.
* `imgaug.imgaug.quokka_segmentation_map`.
  Use `imgaug.data.quokka_segmentation_map` instead.
* `imgaug.imgaug.quokka_keypoints`.
  Use `imgaug.data.quokka_keypoints` instead.
* `imgaug.imgaug.quokka_bounding_boxes`.
  Use `imgaug.data.quokka_bounding_boxes` instead.
* `imgaug.imgaug.quokka_polygons`.
  Use `imgaug.data.quokka_polygons` instead.

Removed Constants:
* `imgaug.imgaug.FILE_DIR`
* `imgaug.imgaug.QUOKKA_FP`
* `imgaug.imgaug.QUOKKA_ANNOTATIONS_FP`
* `imgaug.imgaug.QUOKKA_DEPTH_MAP_HALFRES_FP`
