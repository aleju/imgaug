from __future__ import print_function, division

import time
import multiprocessing

import numpy as np
from skimage import data

import imgaug as ia
import imgaug.multicore as multicore
from imgaug import augmenters as iaa


class PoolWithMarkedWorker(multicore.Pool):
    def __init__(self, *args, **kwargs):
        super(PoolWithMarkedWorker, self).__init__(*args, **kwargs)

    @classmethod
    def _worker(cls, batch_idx, batch):
        process_name = multiprocessing.current_process().name
        # print("[_worker] called %s. images in batch: %d" % (process_name, len(batch.images_unaug),))
        if "-1" in process_name:
            for image in batch.images_unaug:
                image[::4, ::4, :] = [255, 255, 255]
        return multicore.Pool._worker(batch_idx, batch)


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
    print(".pool()")
    print("------------------")
    with augseq.pool() as pool:
        time_start = time.time()
        batches = list(load_images())
        batches_aug = pool.map_batches(batches)
        images_aug = []
        keypoints_aug = []
        for batch_aug in batches_aug:
            images_aug.append(batch_aug.images_aug)
            keypoints_aug.append(batch_aug.keypoints_aug)
        print("Done in %.4fs" % (time.time() - time_start,))
    # ia.imshow(draw_grid(images_aug, keypoints_aug))

    print("------------------")
    print("Pool.map_batches(batches)")
    print("------------------")
    with multicore.Pool(augseq) as pool:
        time_start = time.time()
        batches = list(load_images())
        batches_aug = pool.map_batches(batches)
        images_aug = []
        keypoints_aug = []
        for batch_aug in batches_aug:
            images_aug.append(batch_aug.images_aug)
            keypoints_aug.append(batch_aug.keypoints_aug)
        print("Done in %.4fs" % (time.time() - time_start,))
    # ia.imshow(draw_grid(images_aug, keypoints_aug))

    print("------------------")
    print("Pool.imap_batches(batches)")
    print("------------------")
    with multicore.Pool(augseq) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images())
        images_aug = []
        keypoints_aug = []
        for batch in batches_aug:
            images_aug.append(batch.images_aug)
            keypoints_aug.append(batch.keypoints_aug)
        print("Done in %.4fs" % (time.time() - time_start,))
    # ia.imshow(draw_grid(images_aug, keypoints_aug))

    print("------------------")
    print("Pool.imap_batches(batches, chunksize=32)")
    print("------------------")
    with multicore.Pool(augseq) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images(n_batches=1000), chunksize=32)
        count = 0
        for batch in batches_aug:
            count += 1
        assert count == 1000
        print("Done in %.4fs" % (time.time() - time_start,))

    print("------------------")
    print("Pool.imap_batches(batches, chunksize=2)")
    print("------------------")
    with multicore.Pool(augseq) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images(n_batches=1000), chunksize=2)
        count = 0
        for batch in batches_aug:
            count += 1
        assert count == 1000
        print("Done in %.4fs" % (time.time() - time_start,))

    print("------------------")
    print("Pool.imap_batches(batches, chunksize=1)")
    print("------------------")
    with multicore.Pool(augseq) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images(n_batches=1000), chunksize=1)
        count = 0
        for batch in batches_aug:
            count += 1
        assert count == 1000
        print("Done in %.4fs" % (time.time() - time_start,))

    print("------------------")
    print("Pool.map_batches(batches, chunksize=32)")
    print("------------------")
    with multicore.Pool(augseq) as pool:
        time_start = time.time()
        batches_aug = pool.map_batches(list(load_images(n_batches=1000)), chunksize=32)
        assert len(batches_aug) == 1000
        print("Done in %.4fs" % (time.time() - time_start,))

    print("------------------")
    print("Pool.map_batches chunksize with fast aug")
    print("------------------")
    def test_fast(processes, chunksize):
        augseq = iaa.Dropout(0.1)
        with multicore.Pool(augseq, processes=processes) as pool:
            batches = list(load_images(n_batches=10000, draw_text=False))
            time_start = time.time()
            batches_aug = pool.map_batches(batches, chunksize=chunksize)
            assert len(batches_aug) == 10000
            print("chunksize=%d, worker=%s, time=%.4fs" % (chunksize, processes, time.time() - time_start))

    test_fast(-4, 1)
    test_fast(1, 1)
    test_fast(None, 1)
    test_fast(1, 4)
    test_fast(None, 4)
    test_fast(1, 32)
    test_fast(None, 32)

    print("------------------")
    print("Pool.imap_batches chunksize with fast aug")
    print("------------------")
    def test_fast_imap(processes, chunksize):
        augseq = iaa.Dropout(0.1)
        with multicore.Pool(augseq, processes=processes) as pool:
            time_start = time.time()
            batches_aug = pool.imap_batches(load_images(n_batches=10000, draw_text=False), chunksize=chunksize)
            batches_aug = list(batches_aug)
            assert len(batches_aug) == 10000
            print("chunksize=%d, worker=%s, time=%.4fs" % (chunksize, processes, time.time() - time_start))

    test_fast_imap(-4, 1)
    test_fast_imap(1, 1)
    test_fast_imap(None, 1)
    test_fast_imap(1, 4)
    test_fast_imap(None, 4)
    test_fast_imap(1, 32)
    test_fast_imap(None, 32)

    print("------------------")
    print("Pool.map_batches with computationally expensive aug")
    print("------------------")
    def test_heavy(processes, chunksize):
        augseq_heavy = iaa.PiecewiseAffine(scale=0.2, nb_cols=8, nb_rows=8)
        with multicore.Pool(augseq_heavy, processes=processes) as pool:
            batches = list(load_images(n_batches=500, draw_text=False))
            time_start = time.time()
            batches_aug = pool.map_batches(batches, chunksize=chunksize)
            assert len(batches_aug) == 500
            print("chunksize=%d, worker=%s, time=%.4fs" % (chunksize, processes, time.time() - time_start))

    test_heavy(-4, 1)
    test_heavy(1, 1)
    test_heavy(None, 1)
    test_heavy(1, 4)
    test_heavy(None, 4)
    test_heavy(1, 32)
    test_heavy(None, 32)

    print("------------------")
    print("Pool.imap_batches(batches), slow loading")
    print("------------------")
    with multicore.Pool(augseq) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images(n_batches=100, sleep=0.2))
        images_aug = []
        keypoints_aug = []
        for batch in batches_aug:
            images_aug.append(batch.images_aug)
            keypoints_aug.append(batch.keypoints_aug)
        print("Done in %.4fs" % (time.time() - time_start,))

    print("------------------")
    print("Pool.imap_batches(batches), maxtasksperchild=4")
    print("------------------")
    with multicore.Pool(augseq, maxtasksperchild=4) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images(n_batches=100))
        images_aug = []
        keypoints_aug = []
        for batch in batches_aug:
            images_aug.append(batch.images_aug)
            keypoints_aug.append(batch.keypoints_aug)
        print("Done in %.4fs" % (time.time() - time_start,))
    ia.imshow(draw_grid(images_aug, keypoints_aug))

    print("------------------")
    print("Pool.imap_batches(batches), seed=1")
    print("------------------")
    # we color here the images of the first worker to see in the grids which images belong to one worker
    with PoolWithMarkedWorker(augseq, seed=1) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images(n_batches=4))
        images_aug = []
        keypoints_aug = []
        for batch in batches_aug:
            images_aug.append(batch.images_aug)
            keypoints_aug.append(batch.keypoints_aug)
        print("Done in %.4fs" % (time.time() - time_start,))
    grid_a = draw_grid(images_aug, keypoints_aug)

    with multicore.Pool(augseq, seed=1) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images(n_batches=4))
        images_aug = []
        keypoints_aug = []
        for batch in batches_aug:
            images_aug.append(batch.images_aug)
            keypoints_aug.append(batch.keypoints_aug)
        print("Done in %.4fs" % (time.time() - time_start,))
    grid_b = draw_grid(images_aug, keypoints_aug)

    grid_b[:, 0:2, 0] = 0
    grid_b[:, 0:2, 1] = 255
    grid_b[:, 0:2, 2] = 0
    ia.imshow(np.hstack([grid_a, grid_b]))

    print("------------------")
    print("Pool.imap_batches(batches), seed=None")
    print("------------------")
    with multicore.Pool(augseq, seed=None) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images(n_batches=4))
        images_aug = []
        keypoints_aug = []
        for batch in batches_aug:
            images_aug.append(batch.images_aug)
            keypoints_aug.append(batch.keypoints_aug)
        print("Done in %.4fs" % (time.time() - time_start,))
    grid_a = draw_grid(images_aug, keypoints_aug)

    with multicore.Pool(augseq, seed=None) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images(n_batches=4))
        images_aug = []
        keypoints_aug = []
        for batch in batches_aug:
            images_aug.append(batch.images_aug)
            keypoints_aug.append(batch.keypoints_aug)
        print("Done in %.4fs" % (time.time() - time_start,))
    grid_b = draw_grid(images_aug, keypoints_aug)

    ia.imshow(np.hstack([grid_a, grid_b]))

    print("------------------")
    print("Pool.imap_batches(batches), maxtasksperchild=4, seed=1")
    print("------------------")
    with multicore.Pool(augseq, maxtasksperchild=4, seed=1) as pool:
        time_start = time.time()
        batches_aug = pool.imap_batches(load_images(n_batches=100))
        images_aug = []
        keypoints_aug = []
        for batch in batches_aug:
            images_aug.append(batch.images_aug)
            keypoints_aug.append(batch.keypoints_aug)
        print("Done in %.4fs" % (time.time() - time_start,))
    ia.imshow(draw_grid(images_aug, keypoints_aug))

    for augseq_i in [augseq, augseq_slow]:
        print("------------------")
        print("Many very small runs (batches=1)")
        print("------------------")
        with multicore.Pool(augseq_i) as pool:
            time_start = time.time()
            for i in range(100):
                _ = pool.map_batches(list(load_images(n_batches=1)))
            print("Done in %.4fs" % (time.time() - time_start,))

        print("------------------")
        print("Many very small runs (batches=2)")
        print("------------------")
        with multicore.Pool(augseq_i) as pool:
            time_start = time.time()
            for i in range(100):
                _ = pool.map_batches(list(load_images(n_batches=2)))
            print("Done in %.4fs" % (time.time() - time_start,))


def load_images(n_batches=10, sleep=0.0, draw_text=True):
    batch_size = 4
    astronaut = data.astronaut()
    astronaut = ia.imresize_single_image(astronaut, (64, 64))
    kps = ia.KeypointsOnImage([ia.Keypoint(x=15, y=25)], shape=astronaut.shape)

    counter = 0
    for i in range(n_batches):
        if draw_text:
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
        else:
            if i == 0:
                batch_images = np.array([np.copy(astronaut) for _ in range(batch_size)], dtype=np.uint8)

            batch = ia.Batch(
                images=np.copy(batch_images),
                keypoints=[kps.deepcopy() for _ in range(batch_size)]
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
