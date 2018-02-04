from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data
import cv2

TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10

def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (128, 128))
    #image = ia.quokka(size=(128, 128))
    print("image shape:", image.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (TIME_PER_STEP,))

    configs = [
        (1, 75, 75),
        (3, 75, 75),
        (5, 75, 75),
        (10, 75, 75),
        (10, 25, 25),
        (10, 250, 150),
        (15, 75, 75),
        (15, 150, 150),
        (15, 250, 150),
        (20, 75, 75),
        (40, 150, 150),
        ((1, 5), 75, 75),
        (5, (10, 250), 75),
        (5, 75, (10, 250)),
        (5, (10, 250), (10, 250)),
        (10, (10, 250), (10, 250)),
    ]

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", 128*NB_AUGS_PER_IMAGE, 128)
    #cv2.imshow("aug", image[..., ::-1])
    #cv2.waitKey(TIME_PER_STEP)

    for (d, sigma_color, sigma_space) in configs:
        aug = iaa.BilateralBlur(d=d, sigma_color=sigma_color, sigma_space=sigma_space)

        img_aug = [aug.augment_image(image) for _ in range(NB_AUGS_PER_IMAGE)]
        img_aug = np.hstack(img_aug)
        print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))
        #print("dtype", img_aug.dtype, "averages", img_aug.mean(axis=range(1, img_aug.ndim)))

        title = "d=%s, sc=%s, ss=%s" % (str(d), str(sigma_color), str(sigma_space))
        img_aug = ia.draw_text(img_aug, x=5, y=5, text=title)

        cv2.imshow("aug", img_aug[..., ::-1]) # here with rgb2bgr
        cv2.waitKey(TIME_PER_STEP)

if __name__ == "__main__":
    main()
