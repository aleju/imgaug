from __future__ import print_function, division

import time

import numpy as np
from skimage import data

import imgaug as ia
import imgaug.multicore as multicore
from imgaug import augmenters as iaa


def main():
    augseq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.CoarseDropout(p=0.1, size_percent=0.1)
    ])

    def func_images(images, random_state, parents, hooks):
        time.sleep(0.2)
        return images

    def func_heatmaps(heatmaps, random_state, parents, hooks):
        return heatmaps

    def func_keypoints(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    augseq_slow = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Lambda(
            func_images=func_images,
            func_heatmaps=func_heatmaps,
            func_keypoints=func_keypoints
        )
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
    ia.imshow(draw_grid(images_aug, keypoints_aug))

    print("------------------")
    print("augseq.augment_batches(batches, background=True) -> only images")
    print("------------------")
    batches = list(load_images())
    batches = [batch.images_unaug for batch in batches]
    batches_aug = augseq.augment_batches(batches, background=True)
    images_aug = []
    keypoints_aug = None
    for batch_aug in batches_aug:
        images_aug.append(batch_aug)
    ia.imshow(draw_grid(images_aug, keypoints_aug))

    print("------------------")
    print("BackgroundAugmenter")
    print("------------------")
    batch_loader = multicore.BatchLoader(load_images)
    bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq)
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
    ia.imshow(draw_grid(images_aug, keypoints_aug))

    print("------------------")
    print("BackgroundAugmenter with generator in BL")
    print("------------------")
    batch_loader = multicore.BatchLoader(load_images())
    bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq)
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
    ia.imshow(draw_grid(images_aug, keypoints_aug))

    print("------------------")
    print("Long running BackgroundAugmenter at BL-queue_size=12")
    print("------------------")
    batch_loader = multicore.BatchLoader(load_images(n_batches=1000), queue_size=12)
    bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq)
    i = 0
    while True:
        if i % 100 == 0:
            print("batch=%d..." % (i,))
        batch = bg_augmenter.get_batch()
        if batch is None:
            print("Finished.")
            break
        i += 1

    print("------------------")
    print("Long running BackgroundAugmenter at BL-queue_size=2")
    print("------------------")
    batch_loader = multicore.BatchLoader(load_images(n_batches=1000), queue_size=2)
    bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq)
    i = 0
    while True:
        if i % 100 == 0:
            print("batch=%d..." % (i,))
        batch = bg_augmenter.get_batch()
        if batch is None:
            print("Finished.")
            break
        i += 1

    print("------------------")
    print("Long running BackgroundAugmenter (slow loading)")
    print("------------------")
    batch_loader = multicore.BatchLoader(load_images(n_batches=100, sleep=0.2))
    bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq)
    i = 0
    while True:
        if i % 10 == 0:
            print("batch=%d..." % (i,))
        batch = bg_augmenter.get_batch()
        if batch is None:
            print("Finished.")
            break
        i += 1

    print("------------------")
    print("Long running BackgroundAugmenter (slow aug) at BL-queue_size=12")
    print("------------------")
    batch_loader = multicore.BatchLoader(load_images(n_batches=100), queue_size=12)
    bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq_slow)
    i = 0
    while True:
        if i % 10 == 0:
            print("batch=%d..." % (i,))
        batch = bg_augmenter.get_batch()
        if batch is None:
            print("Finished.")
            break
        i += 1

    print("------------------")
    print("Long running BackgroundAugmenter (slow aug) at BL-queue_size=2")
    print("------------------")
    batch_loader = multicore.BatchLoader(load_images(n_batches=100), queue_size=2)
    bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq_slow)
    i = 0
    while True:
        if i % 10 == 0:
            print("batch=%d..." % (i,))
        batch = bg_augmenter.get_batch()
        if batch is None:
            print("Finished.")
            break
        i += 1

    for augseq_i in [augseq, augseq_slow]:
        print("------------------")
        print("Many very small runs (batches=1)")
        print("------------------")
        for i in range(100):
            batch_loader = multicore.BatchLoader(load_images(n_batches=1), queue_size=100)
            bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq_i)
            while True:
                batch = bg_augmenter.get_batch()
                if batch is None:
                    print("Finished (%d/%d)." % (i+1, 100))
                    break

        print("------------------")
        print("Many very small runs (batches=2)")
        print("------------------")
        for i in range(100):
            batch_loader = multicore.BatchLoader(load_images(n_batches=2), queue_size=100)
            bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq_i)
            while True:
                batch = bg_augmenter.get_batch()
                if batch is None:
                    print("Finished (%d/%d)." % (i+1, 100))
                    break

        print("------------------")
        print("Many very small runs, separate function (batches=1)")
        print("------------------")

        def _augment_small_1():
            batch_loader = multicore.BatchLoader(load_images(n_batches=1), queue_size=100)
            bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq_i)
            i = 0
            while True:
                batch = bg_augmenter.get_batch()
                if batch is None:
                    break
                i += 1

        for i in range(100):
            _augment_small_1()
            print("Finished (%d/%d)." % (i+1, 100))

        print("------------------")
        print("Many very small runs, separate function (batches=2)")
        print("------------------")

        def _augment_small_2():
            batch_loader = multicore.BatchLoader(load_images(n_batches=2), queue_size=100)
            bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq_i)
            i = 0
            while True:
                batch = bg_augmenter.get_batch()
                if batch is None:
                    break
                i += 1

        for i in range(100):
            _augment_small_2()
            print("Finished (%d/%d)." % (i+1, 100))

        print("------------------")
        print("Many very small runs, separate function, incomplete fetching (batches=2)")
        print("------------------")

        def _augment_small_3():
            batch_loader = multicore.BatchLoader(load_images(n_batches=2), queue_size=100)
            bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq_i)
            batch = bg_augmenter.get_batch()

        for i in range(100):
            _augment_small_3()
            print("Finished (%d/%d)." % (i+1, 100))

    #for augseq_i in [augseq, augseq_slow]:
        print("------------------")
        print("Many very small runs, separate function, incomplete fetching (batches=10)")
        print("------------------")

        def _augment_small_4():
            batch_loader = multicore.BatchLoader(load_images(n_batches=10), queue_size=100)
            bg_augmenter = multicore.BackgroundAugmenter(batch_loader, augseq_i)
            batch = bg_augmenter.get_batch()
            #bg_augmenter.terminate()

        for i in range(100):
            _augment_small_4()
            print("Finished (%d/%d)." % (i+1, 100))


def load_images(n_batches=10, sleep=0.0):
    batch_size = 4
    astronaut = data.astronaut()
    astronaut = ia.imresize_single_image(astronaut, (64, 64))
    kps = ia.KeypointsOnImage([ia.Keypoint(x=15, y=25)], shape=astronaut.shape)
    counter = 0
    for i in range(n_batches):
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
        if sleep > 0:
            time.sleep(sleep)


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
