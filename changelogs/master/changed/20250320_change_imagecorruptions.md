# Switch imagecorruptions dependency

The existing PyPi `imagecorruptions` package introduces test
failures due to broken determinism and broken `scikit-image`
compatability.

A [fix](https://github.com/bethgelab/imagecorruptions/pull/27)
was provided, yet ignored for several months. Therefore a
[fork](https://github.com/imaug/imagecorruptions) was necessary which
fixes the issues and provides the PyPi `imagecorruptions-imaug` package.