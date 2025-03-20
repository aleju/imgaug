# Delete different test routines for float128 systems 

The test `test_unusual_channel_numbers` in 
`test/augmenters/test_contrast.py` ran different test routines
depending on the float128 support of the system.

Yet, the main reason for the different behavior is the `params.draw_samples` call in `contrast.py:71` which converts a regular python int into a numpy array of int dtype and thus resulted in different floating point rounding errors.

This behavior was not mirrored in the expected test output and thus resulted in different outcomes on systems with different floating point support.