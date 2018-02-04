from __future__ import print_function, division
from imgaug import augmenters as iaa
from scipy import misc
import numpy as np
from skimage import data

def main():
    image = data.astronaut()
    print("image shape:", image.shape)

    aug = iaa.WithColorspace(
        from_colorspace="RGB",
        to_colorspace="HSV",
        children=iaa.WithChannels(0, iaa.Add(50))
    )

    aug_no_colorspace = iaa.WithChannels(0, iaa.Add(50))

    img_show = np.hstack([
        image,
        aug.augment_image(image),
        aug_no_colorspace.augment_image(image)
    ])
    misc.imshow(img_show)

if __name__ == "__main__":
    main()
