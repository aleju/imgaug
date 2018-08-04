from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import numpy as np
from skimage import data

def main():
    augseq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.CoarseDropout(p=0.1, size_percent=0.1)
    ])

    print("------------------")
    print("augseq.augment_batches(batches, background=True)")
    print("------------------")
    batches = list(load_images())
    batches_aug = augseq.augment_batches(batches, background=True)
    images_aug = []
    keypoints_aug = []
    for batch_aug in batches_aug:
        images_aug.append(batch_aug.images_aug)
        keypoints_aug.append(batch_aug.keypoints_aug)
    misc.imshow(draw_grid(images_aug, keypoints_aug))

    print("------------------")
    print("augseq.augment_batches(batches, background=True) -> only images")
    print("------------------")
    batches = list(load_images())
    batches = [batch.images for batch in batches]
    batches_aug = augseq.augment_batches(batches, background=True)
    images_aug = []
    keypoints_aug = None
    for batch_aug in batches_aug:
        images_aug.append(batch_aug)
    misc.imshow(draw_grid(images_aug, keypoints_aug))

    print("------------------")
    print("BackgroundAugmenter")
    print("------------------")
    batch_loader = ia.BatchLoader(load_images)
    bg_augmenter = ia.BackgroundAugmenter(batch_loader, augseq)
    images_aug = []
    keypoints_aug = []
    while True:
        print("Next batch...")
        batch = bg_augmenter.get_batch()
        if batch is None:
            print("Finished.")
            break
        images_aug.append(batch.images_aug)
        keypoints_aug.append(batch.keypoints_aug)
    misc.imshow(draw_grid(images_aug, keypoints_aug))

def load_images():
    batch_size = 4
    astronaut = data.astronaut()
    astronaut = ia.imresize_single_image(astronaut, (64, 64))
    kps = ia.KeypointsOnImage([ia.Keypoint(x=15, y=25)], shape=astronaut.shape)
    counter = 0
    for i in range(10):
        batch_images = []
        batch_kps = []
        for b in range(batch_size):
            astronaut_text = ia.draw_text(astronaut, x=0, y=0, text="%d" % (counter,), color=[0, 255, 0], size=16)
            batch_images.append(astronaut_text)
            batch_kps.append(kps)
            counter += 1
        batch = ia.Batch(
            images=np.array(batch_images, dtype=np.uint8),
            keypoints=batch_kps
        )
        yield batch

def draw_grid(images_aug, keypoints_aug):
    if keypoints_aug is None:
        keypoints_aug = []
        for bidx in range(len(images_aug)):
            keypoints_aug.append([None for image in images_aug[bidx]])

    images_kps_batches = []
    for bidx in range(len(images_aug)):
        images_kps_batch = []
        for image, kps in zip(images_aug[bidx], keypoints_aug[bidx]):
            if kps is None:
                image_kps = image
            else:
                image_kps = kps.draw_on_image(image, size=5, color=[255, 0, 0])
            images_kps_batch.append(image_kps)
        images_kps_batches.extend(images_kps_batch)

    grid = ia.draw_grid(images_kps_batches, cols=len(images_aug[0]))
    return grid

if __name__ == "__main__":
    main()
