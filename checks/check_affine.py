from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import numpy as np
from skimage import data

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
    params = [
        {"rotate":45},
        {"translate_px":20},
        {"translate_percent":0.1},
        {"scale":1.2},
        {"scale":0.8},
        {"shear":45},
        {"rotate":45, "cval":255},
        {"translate_px":20, "mode":"constant"},
        {"translate_px":20, "mode":"edge"},
        {"translate_px":20, "mode":"symmetric"},
        {"translate_px":20, "mode":"reflect"},
        {"translate_px":20, "mode":"wrap"},
        {"scale":0.5, "order":0},
        {"scale":0.5, "order":1},
        {"scale":0.5, "order":2},
        {"scale":0.5, "order":3},
        {"scale":0.5, "order":4},
        {"scale":0.5, "order":5},
        {"rotate":45, "translate_px":20, "scale":1.2},
        {"rotate":45, "translate_px":20, "scale":0.8},
        {"rotate":(-45, 45), "translate_px":(-20, 20), "scale":(0.8, 1.2), "order":ia.ALL, "mode":ia.ALL, "cval":ia.ALL},
        {"rotate":(-45, 45), "translate_px":(-20, 20), "scale":(0.8, 1.2), "order":ia.ALL, "mode":ia.ALL, "cval":ia.ALL},
        {"rotate":(-45, 45), "translate_px":(-20, 20), "scale":(0.8, 1.2), "order":ia.ALL, "mode":ia.ALL, "cval":ia.ALL},
        {"rotate":(-45, 45), "translate_px":(-20, 20), "scale":(0.8, 1.2), "order":ia.ALL, "mode":ia.ALL, "cval":ia.ALL}
    ]
    seqs_skimage = [iaa.Affine(backend="skimage", **p) for p in params]
    seqs_cv2 = [iaa.Affine(backend="auto", **p) for p in params]
    #seqs_cv2 = []
    #for p in params:
    #    seqs_cv2 = [iaa.Affine(backend="cv2", **p) for p in params]

    for seq_skimage, seq_cv2 in zip(seqs_skimage, seqs_cv2):
        #seq_skimage.localize_random_state_()
        #seq_cv2.localize_random_state_()
        #seq_cv2.copy_random_state_(seq_skimage)

        seq_skimage_det = seq_skimage.to_deterministic()
        seq_cv2_det = seq_cv2.to_deterministic()

        seq_cv2_det.copy_random_state_(seq_skimage_det)

        image_aug_skimage = seq_skimage_det.augment_image(image)
        image_aug_cv2 = seq_cv2_det.augment_image(image)
        #print(image_aug.dtype, np.min(image_aug), np.max(image_aug))
        kps_aug_skimage = seq_skimage_det.augment_keypoints([kps])[0]
        kps_aug_cv2     = seq_cv2_det.augment_keypoints([kps])[0]
        bbs_aug_skimage = seq_skimage_det.augment_bounding_boxes([bbs])[0]
        bbs_aug_cv2     = seq_cv2_det.augment_bounding_boxes([bbs])[0]

        image_before_skimage = np.copy(image)
        image_before_cv2     = np.copy(image)
        image_before_skimage = kps.draw_on_image(image_before_skimage)
        image_before_cv2     = kps.draw_on_image(image_before_cv2)
        image_before_skimage = bbs.draw_on_image(image_before_skimage)
        image_before_cv2     = bbs.draw_on_image(image_before_cv2)

        image_after_skimage = np.copy(image_aug_skimage)
        image_after_cv2     = np.copy(image_aug_cv2)
        image_after_skimage = kps_aug_skimage.draw_on_image(image_after_skimage)
        image_after_cv2     = kps_aug_cv2.draw_on_image(image_after_cv2)
        image_after_skimage = bbs_aug_skimage.draw_on_image(image_after_skimage)
        image_after_cv2     = bbs_aug_cv2.draw_on_image(image_after_cv2)

        pairs.append(np.hstack((image_before_skimage, image_after_skimage, image_after_cv2)))

    misc.imshow(np.vstack(pairs))
    misc.imsave("affine.jpg", np.vstack(pairs))

if __name__ == "__main__":
    main()
