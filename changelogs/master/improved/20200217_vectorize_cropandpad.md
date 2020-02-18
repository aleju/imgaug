# Vectorize `CropAndPad` #619

This patch vectorizes parts of `CropAndPad`, especially the
sampling process, leading to an improved performance for large
batches.

Previously, cropping an image below a height and/or width of
`1` would be prevented by `CropAndPad` *and* a warning was
raised if it was tried. That warning was now removed, but
height/width of at least `1` is still ensured.
