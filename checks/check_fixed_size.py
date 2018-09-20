from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data
import cv2

def main():
    image = ia.quokka(size=0.5)
    kps = [ia.KeypointsOnImage(
        [ia.Keypoint(x=245, y=203), ia.Keypoint(x=365, y=195), ia.Keypoint(x=313, y=269)],
        shape=(image.shape[0]*2, image.shape[1]*2)
    )]
    kps[0] = kps[0].on(image.shape)
    print("image shape:", image.shape)
    
    augs_many = [
        iaa.CropFixedSize(200,200, name="width200-height200"),
        iaa.CropFixedSize(200,322, name="width200-height322"),
        iaa.CropFixedSize(480,200, name="width480-height200"),
        iaa.CropFixedSize(480,322, name="width480-height322"),
        iaa.Sequential([
            iaa.PadUptoFixedSize(480,400),
            iaa.CropFixedSize(480,400)
        ], name="width480-height400"),
        iaa.Sequential([
            iaa.PadUptoFixedSize(600,322),
            iaa.CropFixedSize(600,322)
        ], name="width600-height322"),
        iaa.Sequential([
            iaa.PadUptoFixedSize(600,400),
            iaa.CropFixedSize(600,400)
        ], name="width600-height400"),
    ]

    print("original", image.shape)
    ia.imshow(kps[0].draw_on_image(image))

    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    for aug in augs_many:
        images_aug = []
        for _ in range(36):
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            img_aug_kps = kps_aug.draw_on_image(img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            #print(aug.name, img_aug_kps.shape, img_aug_kps.shape[1]/img_aug_kps.shape[0])
            images_aug.append(img_aug_kps)
        print(aug.name)
        ia.imshow(ia.draw_grid(images_aug))

if __name__ == "__main__":
    main()
