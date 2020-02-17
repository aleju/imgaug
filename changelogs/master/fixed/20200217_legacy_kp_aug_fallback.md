* Fix legacy augmenters (i.e. no `_augment_batch_()`
  implemented) not automatically falling back to
  `_augment_keypoints()` for the augmentation of bounding
  boxes, polygons and line strings. #617 #618