from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc, ndimage
import numpy as np
from skimage import data
import cv2

NB_ROWS = 10
NB_COLS = 10
HEIGHT = 200
WIDTH = 256
BB_X1 = 64
BB_X2 = WIDTH - 64
BB_Y1 = 64
BB_Y2 = HEIGHT - 64

def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (HEIGHT, WIDTH))

    kps = []
    for y in range(NB_ROWS):
        ycoord = BB_Y1 + int(y * (BB_Y2 - BB_Y1) / (NB_COLS - 1))
        for x in range(NB_COLS):
            xcoord = BB_X1 + int(x * (BB_X2 - BB_X1) / (NB_ROWS - 1))
            kp = (xcoord, ycoord)
            kps.append(kp)
    kps = set(kps)
    kps = [ia.Keypoint(x=xcoord, y=ycoord) for (xcoord, ycoord) in kps]
    kps = ia.KeypointsOnImage(kps, shape=image.shape)

    bb = ia.BoundingBox(x1=BB_X1, x2=BB_X2, y1=BB_Y1, y2=BB_Y2)
    bbs = ia.BoundingBoxesOnImage([bb], shape=image.shape)

    pairs = []
    seqs = [
        iaa.AffineOpenCV(rotate=45),
        iaa.AffineOpenCV(translate_px=20),
        iaa.AffineOpenCV(translate_percent=0.1),
        iaa.AffineOpenCV(scale=1.2),
        iaa.AffineOpenCV(scale=0.8),
        iaa.AffineOpenCV(shear=45),
        iaa.AffineOpenCV(rotate=45, cval=256),
        iaa.AffineOpenCV(translate_px=20, mode=cv2.BORDER_CONSTANT),
        iaa.AffineOpenCV(translate_px=20, mode=cv2.BORDER_REPLICATE),
        iaa.AffineOpenCV(translate_px=20, mode=cv2.BORDER_REFLECT),
        iaa.AffineOpenCV(translate_px=20, mode=cv2.BORDER_REFLECT_101),
        iaa.AffineOpenCV(translate_px=20, mode=cv2.BORDER_WRAP),
        iaa.AffineOpenCV(translate_px=20, mode="constant"),
        iaa.AffineOpenCV(translate_px=20, mode="replicate"),
        iaa.AffineOpenCV(translate_px=20, mode="reflect"),
        iaa.AffineOpenCV(translate_px=20, mode="reflect_101"),
        iaa.AffineOpenCV(translate_px=20, mode="wrap"),
        iaa.AffineOpenCV(scale=0.5, order=cv2.INTER_NEAREST),
        iaa.AffineOpenCV(scale=0.5, order=cv2.INTER_LINEAR),
        iaa.AffineOpenCV(scale=0.5, order=cv2.INTER_CUBIC),
        iaa.AffineOpenCV(scale=0.5, order=cv2.INTER_LANCZOS4),
        iaa.AffineOpenCV(scale=0.5, order="nearest"),
        iaa.AffineOpenCV(scale=0.5, order="linear"),
        iaa.AffineOpenCV(scale=0.5, order="cubic"),
        iaa.AffineOpenCV(scale=0.5, order="lanczos4"),
        iaa.AffineOpenCV(rotate=45, translate_px=20, scale=1.2),
        iaa.AffineOpenCV(rotate=45, translate_px=20, scale=0.8),
        iaa.AffineOpenCV(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL),
        iaa.AffineOpenCV(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL),
        iaa.AffineOpenCV(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL),
        iaa.AffineOpenCV(rotate=(-45, 45), translate_px=(-20, 20), scale=(0.8, 1.2), order=ia.ALL, mode=ia.ALL, cval=ia.ALL)
    ]

    for seq in seqs:
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_image(image)
        #print(image_aug.dtype, np.min(image_aug), np.max(image_aug))
        kps_aug = seq_det.augment_keypoints([kps])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        image_before = np.copy(image)
        image_before = kps.draw_on_image(image_before)
        image_before = bbs.draw_on_image(image_before)

        image_after = np.copy(image_aug)
        image_after = kps_aug.draw_on_image(image_after)
        image_after = bbs_aug.draw_on_image(image_after)

        pairs.append(np.hstack((image_before, image_after)))

    misc.imshow(np.vstack(pairs))
    misc.imsave("affineopencv.jpg", np.vstack(pairs))

if __name__ == "__main__":
    main()
