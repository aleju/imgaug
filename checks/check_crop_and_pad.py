from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np


def main():
    image = ia.quokka(size=0.5)
    kps = [ia.KeypointsOnImage(
        [ia.Keypoint(x=245, y=203), ia.Keypoint(x=365, y=195), ia.Keypoint(x=313, y=269)],
        shape=(image.shape[0]*2, image.shape[1]*2)
    )]
    kps[0] = kps[0].on(image.shape)
    print("image shape:", image.shape)

    augs = [
        iaa.CropAndPad(px=50, name="pad-by-50px"),
        iaa.CropAndPad(px=(10, 20, 30, 40), name="pad-by-10-20-30-40px"),
        iaa.CropAndPad(percent=0.1, name="pad-by-01percent"),
        iaa.CropAndPad(percent=(0.01, 0.02, 0.03, 0.04), name="pad-by-001-002-003-004percent"),
        iaa.CropAndPad(px=-20, name="crop-by-20px"),
        iaa.CropAndPad(px=(-10, -20, -30, -40), name="crop-by-10-20-30-40px"),
        iaa.CropAndPad(percent=-0.1, name="crop-by-01percent"),
        iaa.CropAndPad(percent=(-0.01, -0.02, -0.03, -0.04), name="crop-by-001-002-003-004percent")
    ]

    augs_many = [
        iaa.Crop(px=(0, 50), name="native-crop-0-to-50px"),
        iaa.Crop(px=iap.DiscreteUniform(0, 50), name="native-crop-0-to-50px-iap"),
        iaa.Pad(px=(0, 50), pad_mode="linear_ramp", pad_cval=(0, 255), name="native-pad-0-to-50px-pad-modes"),
        iaa.CropAndPad(px=(0, 50), sample_independently=False, name="pad-by-0-to-50px-same"),
        iaa.CropAndPad(px=(0, 50), name="pad-by-0-to-50px"),
        iaa.CropAndPad(px=(0, 50), pad_mode=ia.ALL, pad_cval=(0, 255), name="pad-by-0-to-50px-random-pad-modes-cvals"),
        iaa.CropAndPad(px=((0, 50), (0, 50), (0, 50), (0, 50)), name="pad-by-0-to-50px-each"),
        iaa.CropAndPad(percent=(0, 0.1), sample_independently=False, name="pad-by-0-to-01percent-same"),
        iaa.CropAndPad(percent=(0, 0.1), name="pad-by-0-to-01percent"),
        iaa.CropAndPad(percent=(0, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255),
                       name="pad-by-0-to-01percent-random-pad-modes-cvals"),
        iaa.CropAndPad(percent=((0, 0.1), (0, 0.1), (0, 0.1), (0, 0.1)), name="pad-by-0-to-01percent-each"),
        iaa.CropAndPad(px=(-50, 0), name="crop-by-50-to-0px"),
        iaa.CropAndPad(px=((-50, 0), (-50, 0), (-50, 0), (-50, 0)), name="crop-by-50-to-0px-each"),
        iaa.CropAndPad(percent=(-0.1, 0), name="crop-by-01-to-0percent"),
        iaa.CropAndPad(percent=((-0.1, 0), (-0.1, 0), (-0.1, 0), (-0.1, 0)), name="crop-by-01-to-0percent-each"),
        iaa.CropAndPad(px=(-50, 50), name="pad-and-crop-by-50px")
    ]

    print("original", image.shape)
    ia.imshow(kps[0].draw_on_image(image))

    print("-----------------")
    print("Same aug per image")
    print("-----------------")
    for aug in augs:
        img_aug = aug.augment_image(image)
        kps_aug = aug.augment_keypoints(kps)[0]
        img_aug_kps = kps_aug.draw_on_image(img_aug)
        print(aug.name, img_aug_kps.shape, img_aug_kps.shape[1]/img_aug_kps.shape[0])
        ia.imshow(img_aug_kps)

    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    for aug in augs_many:
        images_aug = []
        for _ in range(64):
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            img_aug_kps = kps_aug.draw_on_image(img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            images_aug.append(img_aug_kps)
        print(aug.name)
        ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
