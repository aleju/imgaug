from __future__ import print_function, division

import cv2
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

TIME_PER_STEP = 10000


def main():
    image = data.astronaut()
    print("image shape:", image.shape)
    print("Press ENTER or wait %d ms to proceed to the next image." % (TIME_PER_STEP,))

    children_all = [
        ("hflip", iaa.Fliplr(1)),
        ("add", iaa.Add(50)),
        ("dropout", iaa.Dropout(0.2)),
        ("affine", iaa.Affine(rotate=35))
    ]

    channels_all = [
        None,
        0,
        [],
        [0],
        [0, 1],
        [1, 2],
        [0, 1, 2]
    ]

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.imshow("aug", image[..., ::-1])
    cv2.waitKey(TIME_PER_STEP)

    for children_title, children in children_all:
        for channels in channels_all:
            aug = iaa.WithChannels(channels=channels, children=children)
            img_aug = aug.augment_image(image)
            print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))

            title = "children=%s | channels=%s" % (children_title, channels)
            img_aug = ia.draw_text(img_aug, x=5, y=5, text=title)

            cv2.imshow("aug", img_aug[..., ::-1])  # here with rgb2bgr
            cv2.waitKey(TIME_PER_STEP)


if __name__ == "__main__":
    main()
