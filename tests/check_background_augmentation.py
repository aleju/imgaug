from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc, ndimage
import numpy as np
from skimage import data

TIME_PER_STEP = 5000
NB_AUGS_PER_IMAGE = 10

def main():
    augseq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.CoarseDropout(p=0.1, size_percent=0.1)
    ])
    batch_loader = ia.BatchLoader(load_images)
    bg_augmenter = ia.BackgroundAugmenter(augseq, batch_loader, nb_workers=5)
    augs = []
    while True:
        print("Next batch...")
        batch = bg_augmenter.get_batch()
        if batch is None:
            print("Finished.")
            break
        augs.append(np.hstack(list(batch.images_aug)))
    misc.imshow(np.vstack(augs))

def load_images():
    batch_size = 4
    astronaut = data.astronaut()
    astronaut = ia.imresize_single_image(astronaut, (64, 64))
    counter = 0
    for i in range(10):
        batch = []
        for b in range(batch_size):
            astronaut_text = ia.draw_text(astronaut, x=0, y=0, text="%d" % (counter,), color=[0, 255, 0], size=16)
            batch.append(astronaut_text)
            counter += 1
        batch = ia.Batch(images=np.array(batch, dtype=np.uint8))
        yield batch

if __name__ == "__main__":
    main()
