from __future__ import print_function, division, absolute_import

import time
import multiprocessing
import pickle

import numpy as np
import six.moves as sm
import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis

import imgaug as ia
import imgaug.multicore as multicore
from imgaug import augmenters as iaa
from imgaug.testutils import reseed


def main():
    time_start = time.time()

    test_BatchLoader()
    # test_BackgroundAugmenter.get_batch()
    test_BackgroundAugmenter__augment_images_worker()
    # test_BackgroundAugmenter.terminate()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_BatchLoader():
    reseed()

    def _load_func():
        for _ in sm.xrange(20):
            yield ia.Batch(images=np.zeros((2, 4, 4, 3), dtype=np.uint8))

    for nb_workers in [1, 2]:
        # repeat these tests many times to catch rarer race conditions
        for _ in sm.xrange(5):
            loader = multicore.BatchLoader(_load_func, queue_size=2, nb_workers=nb_workers, threaded=True)
            loaded = []
            counter = 0
            while (not loader.all_finished() or not loader.queue.empty()) and counter < 1000:
                try:
                    batch = loader.queue.get(timeout=0.001)
                    loaded.append(batch)
                except:
                    pass
                counter += 1
            assert len(loaded) == 20*nb_workers, \
                "Expected %d to be loaded by threads, got %d for %d workers at counter %d." % (
                    20*nb_workers, len(loaded), nb_workers, counter
                )

            loader = multicore.BatchLoader(_load_func, queue_size=200, nb_workers=nb_workers, threaded=True)
            loader.terminate()
            assert loader.all_finished()

            loader = multicore.BatchLoader(_load_func, queue_size=2, nb_workers=nb_workers, threaded=False)
            loaded = []
            counter = 0
            while (not loader.all_finished() or not loader.queue.empty()) and counter < 1000:
                try:
                    batch = loader.queue.get(timeout=0.001)
                    loaded.append(batch)
                except:
                    pass
                counter += 1
            assert len(loaded) == 20*nb_workers, \
                "Expected %d to be loaded by background processes, got %d for %d workers at counter %d." % (
                    20*nb_workers, len(loaded), nb_workers, counter
                )

            loader = multicore.BatchLoader(_load_func, queue_size=200, nb_workers=nb_workers, threaded=False)
            loader.terminate()
            assert loader.all_finished()


def test_BackgroundAugmenter__augment_images_worker():
    reseed()

    def gen():
        yield ia.Batch(images=np.zeros((1, 4, 4, 3), dtype=np.uint8))
    bl = multicore.BatchLoader(gen(), queue_size=2)
    bgaug = multicore.BackgroundAugmenter(bl, iaa.Noop(), queue_size=1, nb_workers=1)

    queue_source = multiprocessing.Queue(2)
    queue_target = multiprocessing.Queue(2)
    queue_source.put(
        pickle.dumps(
            ia.Batch(images=np.zeros((1, 4, 8, 3), dtype=np.uint8)),
            protocol=-1
        )
    )
    queue_source.put(pickle.dumps(None, protocol=-1))
    bgaug._augment_images_worker(iaa.Add(1), queue_source, queue_target, 1)

    batch_aug = pickle.loads(queue_target.get())
    assert isinstance(batch_aug, ia.Batch)
    assert batch_aug.images is not None
    assert batch_aug.images.dtype == np.uint8
    assert batch_aug.images.shape == (1, 4, 8, 3)
    assert np.array_equal(batch_aug.images, np.zeros((1, 4, 8, 3), dtype=np.uint8))
    assert batch_aug.images_aug is not None
    assert batch_aug.images_aug.dtype == np.uint8
    assert batch_aug.images_aug.shape == (1, 4, 8, 3)
    assert np.array_equal(batch_aug.images_aug, np.zeros((1, 4, 8, 3), dtype=np.uint8) + 1)

    finished_signal = pickle.loads(queue_target.get())
    assert finished_signal is None

    source_finished_signal = pickle.loads(queue_source.get())
    assert source_finished_signal is None

    assert queue_source.empty()
    assert queue_target.empty()

    queue_source.close()
    queue_target.close()
    queue_source.join_thread()
    queue_target.join_thread()
    bl.terminate()
    bgaug.terminate()


if __name__ == "__main__":
    main()
