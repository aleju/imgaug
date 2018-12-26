from __future__ import print_function, division

import numpy as np

from imgaug import augmenters as iaa


def main():
    def z(shape):
        return np.zeros(shape, dtype=np.uint8)

    seq = iaa.Noop()

    print("This should generate NO warning:")
    _image_aug = seq.augment_images(z((1, 16, 16, 3)))

    print("This should generate NO warning:")
    _image_aug = seq.augment_images(z((16, 16, 8)))

    print("This should generate NO warning:")
    _image_aug = seq.augment_images([z((16, 16, 3))])

    print("This should generate NO warning:")
    _image_aug = seq.augment_images([z((16, 16))])

    print("This should generate a warning:")
    _image_aug = seq.augment_images(z((16, 16, 3)))

    print("This should generate a warning:")
    for _ in range(2):
        _image_aug = seq.augment_images(z((16, 16, 1)))

    print("This should fail:")
    _image_aug = seq.augment_images(z((16, 16)))


if __name__ == "__main__":
    main()
