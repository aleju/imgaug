This folder contains documentation files for imgaug.
To generate the docs, first install the necessary packages, e.g. via:
```bash
pip install Sphinx numpydoc sphinx_rtd_theme
```
Then generate the documentation via
```bash
make html
```
Note that you will probably get lots of warnings saying
`WARNING: toctree contains reference to nonexisting document (...)`. That is normal. Just ignore
these messages.
