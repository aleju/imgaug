from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import numpy as np
from skimage import data

def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (128, 128))

    images = []
    params = [
        (0.25, 0.25),
        (1.0, 0.25),
        (2.0, 0.25),
        (3.0, 0.25),
        (0.25, 0.50),
        (1.0, 0.50),
        (2.0, 0.50),
        (3.0, 0.50),
        (0.25, 0.75),
        (1.0, 0.75),
        (2.0, 0.75),
        (3.0, 0.75)
    ]

    for (alpha, sigma) in params:
        images_row = []
        seqs_row = [
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=0, order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=128, order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=255, order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=0, order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=128, order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=255, order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=0, order=3),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=128, order=3),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=255, order=3),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest", order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest", order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest", order=2),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest", order=3),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="reflect", order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="reflect", order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="reflect", order=2),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="reflect", order=3),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="wrap", order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="wrap", order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="wrap", order=2),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="wrap", order=3)
        ]

        for seq in seqs_row:
            images_row.append(
                seq.augment_image(image)
            )

        images.append(np.hstack(images_row))

    misc.imshow(np.vstack(images))
    misc.imsave("elastic_transformations.jpg", np.vstack(images))

if __name__ == "__main__":
    main()
