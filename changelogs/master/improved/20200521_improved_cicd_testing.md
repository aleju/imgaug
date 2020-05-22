# Improved CI/CD Testing #670 #678

This patch improves the CI/CD environment by adding
github actions. The library is now automatically tested
in Ubuntu with python 2.7, 3.5, 3.6, 3.7 and 3.8,
as well as MacOS and Windows with the same python
versions (except for 2.7 in Windows).
Previously, only Ubuntu with python <=3.7 was
automatically tested in the CI/CD chain.

Additionally, the CI/CD pipeline now also generates
wheel files (sdist, bdist) for every patch merged
into master.
