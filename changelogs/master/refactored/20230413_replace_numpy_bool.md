# Replace numpy bool with standard bool #832

This commit replaces all occurences of the 
dtype `np.bool` with `bool` from the python
standard library. `np.bool` is marked as
depreciated in `numpy >= 1.20`. 