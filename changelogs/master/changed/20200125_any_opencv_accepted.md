# `setup.py` Now Accepts Any `opencv-*` Installation #586

`setup.py` was changed so that it now accepts `opencv-python`,
`opencv-python-headless`, `opencv-contrib-python` and
`opencv-contrib-python-headless`. Previously, only
`opencv-python-headless` was accepted, which could easily cause
conflicts when another one of the mentioned libraries was already
installed.
