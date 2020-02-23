# Improved Performance of `ElasticTransformation` #624

This patch applies various performance-related changes to
`ElasticTransformation`. These cover: (a) the re-use of
generated random samples for multiple images in the same
batch (with some adjustments so that they are not identical),
(b) the caching of generated and re-useable arrays,
(c) a performance-optimized smoothing method for the
underlying displacement maps and (d) the use of nearest
neighbour interpolation (`order=0`) instead of cubic
interpolation (`order=3`) as the new default parameter
for `order`.

These changes lead to a speedup of about 3x to 4x (more
for larger images) at a slight loss of visual
quality (mainly from `order=0`) and variety (due to the
re-use of random samples within each batch).
The new smoothing method leads to slightly stronger
displacements for larger `sigma` values.
