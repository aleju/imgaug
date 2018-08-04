from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import numpy as np

ia.seed(3)

def main():
    image = ia.quokka(size=0.5)
    print(image.shape)
    kps = [
        ia.KeypointsOnImage(
            [
                ia.Keypoint(x=123, y=102),
                ia.Keypoint(x=182, y=98),
                ia.Keypoint(x=155, y=134),

                #ia.Keypoint(x=255, y=213),
                #ia.Keypoint(x=375, y=205),
                #ia.Keypoint(x=323, y=279),

                #ia.Keypoint(x=265, y=223),
                #ia.Keypoint(x=385, y=215),
                #ia.Keypoint(x=333, y=289),

                #ia.Keypoint(x=275, y=233),
                #ia.Keypoint(x=395, y=225),
                #ia.Keypoint(x=343, y=299),

                ia.Keypoint(x=-20, y=20)
            ],
            shape=(image.shape[0], image.shape[1])
        )
    ]
    #kps[0] = kps[0].on(image.shape)
    print("image shape:", image.shape)

    augs = [
        #iaa.PiecewiseAffine(scale=0),
        iaa.PiecewiseAffine(scale=0.05),
        iaa.PiecewiseAffine(scale=0.1),
        iaa.PiecewiseAffine(scale=0.2)
    ]

    #print("original", image.shape)
    misc.imshow(kps[0].draw_on_image(image))

    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    for aug in augs:
        images_aug = []
        for _ in range(16):
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            #img_aug_kps = kps_aug.draw_on_image(img_aug)
            img_aug_kps = keypoints_draw_on_image(kps_aug, img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            #print(aug.name, img_aug_kps.shape, img_aug_kps.shape[1]/img_aug_kps.shape[0])
            images_aug.append(img_aug_kps)
            #misc.imshow(img_aug_kps)
        print(aug.name)
        misc.imshow(ia.draw_grid(images_aug))

def keypoints_draw_on_image(kps, image, color=[0, 255, 0], size=3, copy=True, raise_if_out_of_image=False, border=50):
    if copy:
        image = np.copy(image)

    image = np.pad(
        image,
        ((border, border), (border, border), (0, 0)),
        mode="constant",
        constant_values=0
    )

    height, width = image.shape[0:2]

    for keypoint in kps.keypoints:
        y, x = keypoint.y + border, keypoint.x + border
        if 0 <= y < height and 0 <= x < width:
            x1 = max(x - size//2, 0)
            x2 = min(x + 1 + size//2, width - 1)
            y1 = max(y - size//2, 0)
            y2 = min(y + 1 + size//2, height - 1)
            image[y1:y2, x1:x2] = color
        else:
            if raise_if_out_of_image:
                raise Exception("Cannot draw keypoint x=%d, y=%d on image with shape %s." % (y, x, image.shape))

    return image

if __name__ == "__main__":
    main()
