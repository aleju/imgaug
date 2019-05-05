from __future__ import absolute_import

# this contains some deprecated classes/functions pointing to the new
# classes/functions, hence always place the other imports below this so that
# the deprecated stuff gets overwritten as much as possible
from imgaug.imgaug import *

import imgaug.augmentables as augmentables
from imgaug.augmentables import *
import imgaug.augmenters as augmenters
import imgaug.parameters as parameters
import imgaug.dtypes as dtypes

__version__ = '0.2.9'
