from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
from skimage import data

def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (128, 128))

    #image = ia.imresize_single_image(np.tile(data.checkerboard()[:,:,np.newaxis], (1, 1, 3)), (124, 124))
    #image = ia.pad(image, top=2, right=2, bottom=2, left=2)

    #image = np.tile(np.linspace(0, 255, 128).astype(np.float32).reshape(1, 128), (128, 1))
    #image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
    #image = ia.pad(image, top=2, right=2, bottom=2, left=2)
    #image = np.rot90(image)

    #image = np.zeros((126, 126+20, 3), dtype=np.uint8)
    #image[5:10, :, :] = 255
    #image = ia.pad(image, top=2, right=2, bottom=2, left=2, cval=128)

    #image = np.tile(np.arange(0, 100, 1).reshape(10, 10)[:,:,np.newaxis], (1, 1, 3))

    print("alpha=vary, sigma=0.25")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=0.25) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    misc.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=vary, sigma=1.0")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=1.0) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    misc.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=vary, sigma=3.0")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=3.0) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    misc.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=vary, sigma=5.0")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=5.0) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    misc.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=1.0, sigma=vary")
    augs = [iaa.ElasticTransformation(alpha=1.0, sigma=sigma) for sigma in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    misc.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=10.0, sigma=vary")
    augs = [iaa.ElasticTransformation(alpha=10.0, sigma=sigma) for sigma in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    misc.imshow(ia.draw_grid(images_aug, cols=10))

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
        #(0.25, 0.25),
        (2.0, 0.25),
        #(6.0, 0.25),
        #(12.0, 0.25),
        #(0.25, 1.5),
        #(2.0, 1.50),
        #(6.0, 1.50),
        #(12.0, 1.50),
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
        kps_row = []
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
            #image_aug_kp = kps.draw_on_image(image_aug_kp, color=[0, 0, 255])
            image_aug_kp = kps_aug.draw_on_image(image_aug_kp, size=3)
            images_row.append(image_aug_kp)
            #exit()

        images.append(np.hstack(images_row))

    ia.imshow(np.vstack(images))
    imageio.imwrite("elastic_transformations.jpg", np.vstack(images))

if __name__ == "__main__":
    main()
