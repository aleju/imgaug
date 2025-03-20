# Fix broken dtypes for numpy 2.0 #856

`np.sctypes` was removed in the NumPy 2.0.
All occurences were replaced with a set of 
the specific NumPy types.

