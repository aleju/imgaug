from __future__ import print_function, division

import imageio
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (128, 128))

    print("alpha=vary, sigma=0.25")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=0.25) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=vary, sigma=1.0")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=1.0) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=vary, sigma=3.0")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=3.0) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=vary, sigma=5.0")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=5.0) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=1.0, sigma=vary")
    augs = [iaa.ElasticTransformation(alpha=1.0, sigma=sigma) for sigma in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=10.0, sigma=vary")
    augs = [iaa.ElasticTransformation(alpha=10.0, sigma=sigma) for sigma in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    kps = ia.KeypointsOnImage(
        [ia.Keypoint(x=1, y=1),
         ia.Keypoint(x=50, y=24), ia.Keypoint(x=42, y=96), ia.Keypoint(x=88, y=106), ia.Keypoint(x=88, y=53),
         ia.Keypoint(x=0, y=0), ia.Keypoint(x=128, y=128), ia.Keypoint(x=-20, y=30), ia.Keypoint(x=20, y=-30),
         ia.Keypoint(x=-20, y=-30)],
        shape=image.shape
    )

    images = []
    params = [
        (0.0, 0.0),
        (0.2, 0.2),
        (2.0, 0.25),
        (0.25, 3.0),
        (2.0, 3.0),
        (6.0, 3.0),
        (12.0, 3.0),
        (50.0, 5.0),
        (100.0, 5.0),
        (100.0, 10.0)
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
            seq_det = seq.to_deterministic()
            image_aug = seq_det.augment_image(image)
            kps_aug = seq_det.augment_keypoints([kps])[0]
            image_aug_kp = np.copy(image_aug)
            image_aug_kp = kps_aug.draw_on_image(image_aug_kp, size=3)
            images_row.append(image_aug_kp)

        images.append(np.hstack(images_row))

    ia.imshow(np.vstack(images))
    imageio.imwrite("elastic_transformations.jpg", np.vstack(images))


if __name__ == "__main__":
    main()
