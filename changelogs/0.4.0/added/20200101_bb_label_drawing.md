# Drawing Bounding Box Labels #545

When drawing bounding boxes on images via `BoundingBox.draw_on_image()`
or `BoundingBoxesOnImage.draw_on_image()`, a box containing the label will now
be drawn over each bounding box's rectangle. If the bounding box's label is
set to `None`, the label box will not be drawn. For more detailed control,
use `BoundingBox.draw_label_on_image()`.

* Added method `imgaug.augmentables.BoundingBox.draw_label_on_image()`.
* Added method `imgaug.augmentables.BoundingBox.draw_box_on_image()`.
* Changed method `imgaug.augmentables.BoundingBox.draw_on_image()`
  to automatically draw a bounding box's label.
