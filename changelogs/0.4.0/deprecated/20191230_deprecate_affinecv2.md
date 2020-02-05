# Deprecated AffineCv2 in Favor of Affine #540

The augmenter `imgaug.augmenters.geometric.AffineCv2` was not properly
maintained for quite a while and its functionality is already covered
by `imgaug.augmenters.geometric.Affine` using parameter
`backend='cv2'`. Hence, it was now deprecated. Use `Affine` instead.
