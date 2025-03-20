# Changed github actions

Version 1 of `actions/cache` is now depreciated. 
The pipeline switched version 4.

Since `requirements.txt` was deleted, the caching
not relies on `setup.py` instead.

Python 3.7 is not available on `ubuntu-latest` runners.
