# Create imagecorruptions fork

Since `scikit-image >= 0.19.0` the determinism of some 
functions cannot be controlled via manipulating the global
NumPy random generator seed. This results in random behavior
for tests involving `imagecorruptions` functions. 
The only solutions is to add a seed parameter to the repective
functions. A corresponding upstream pull request is pending but
the changes are low such that a `imagecorruptions` fork is required.

Another problem is the broken support of the `imagecorruptions`
main branch for older `scikit-image` versions.

In the future versions will merge the fork into the `imgaug` codebase and
also require newer `scikit-image` versions.