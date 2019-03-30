from __future__ import print_function, division, absolute_import

import time
import multiprocessing
import pickle
from collections import defaultdict
import warnings
import sys
# unittest only added in 3.4 self.subTest()
if sys.version_info[0] < 3 or sys.version_info[1] < 4:
    import unittest2 as unittest
else:
    import unittest
# unittest.mock is not available in 2.7 (though unittest2 might contain it?)
try:
    import unittest.mock as mock
except ImportError:
    import mock

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


class TestPool(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_property_pool(self):
        mock_Pool = mock.MagicMock()
        mock_Pool.return_value = mock_Pool
        mock_Pool.close.return_value = None
        mock_Pool.join.return_value = None
        with mock.patch("multiprocessing.Pool", mock_Pool):
            augseq = iaa.Noop()
            with multicore.Pool(augseq, processes=1, maxtasksperchild=4, seed=123) as pool:
                assert pool.processes == 1
            assert pool._pool is None
        assert mock_Pool.call_count == 1
        assert mock_Pool.close.call_count == 1
        assert mock_Pool.join.call_count == 1
        assert mock_Pool.call_args[0][0] == 1  # processes
        assert mock_Pool.call_args[1]["initargs"] == (augseq, 123)
        assert mock_Pool.call_args[1]["maxtasksperchild"] == 4

    def test_processes(self):
        augseq = iaa.Noop()
        mock_Pool = mock.MagicMock()
        mock_cpu_count = mock.Mock()
        with mock.patch("multiprocessing.Pool", mock_Pool), mock.patch("multiprocessing.cpu_count", mock_cpu_count):
            combos = [
                (1, 1, 1),
                (2, 1, 1),
                (3, 1, 1),
                (1, 2, 2),
                (3, 2, 2),
                (1, None, None),
                (2, None, None),
                (3, None, None),
                (1, -1, 1),
                (2, -1, 1),
                (3, -1, 2),
                (4, -2, 2)
            ]

            for ret_val, inputs, expected in combos:
                mock_cpu_count.return_value = ret_val
                with multicore.Pool(augseq, processes=inputs) as _pool:
                    pass

                if expected is None:
                    assert mock_Pool.call_args[0][0] is None
                else:
                    assert mock_Pool.call_args[0][0] == expected

    def _test_map_batches_both(self, call_async):
        augseq = iaa.Noop()
        mock_Pool = mock.MagicMock()
        mock_Pool.return_value = mock_Pool
        mock_Pool.map.return_value = "X"
        mock_Pool.map_async.return_value = "X"
        with mock.patch("multiprocessing.Pool", mock_Pool):
            batches = [ia.Batch(images=[ia.quokka()]), ia.Batch(images=[ia.quokka()+1])]
            with multicore.Pool(augseq, processes=1) as pool:
                if call_async:
                    _ = pool.map_batches_async(batches)
                else:
                    _ = pool.map_batches(batches)

            if call_async:
                to_check = mock_Pool.map_async
            else:
                to_check = mock_Pool.map

            assert to_check.call_count == 1
            # args, arg 0
            assert to_check.call_args[0][0] == multicore._Pool_starworker
            # args, arg 1 (batches with ids), tuple 0, entry 0 in tuple (=> batch id)
            assert to_check.call_args[0][1][0][0] == 0
            # args, arg 1 (batches with ids), tuple 0, entry 1 in tuple (=> batch)
            assert np.array_equal(to_check.call_args[0][1][0][1].images_unaug, batches[0].images_unaug)
            # args, arg 1 (batches with ids), tuple 1, entry 0 in tuple (=> batch id)
            assert to_check.call_args[0][1][1][0] == 1
            # args, arg 1 (batches with ids), tuple 1, entry 1 in tuple (=> batch)
            assert np.array_equal(to_check.call_args[0][1][1][1].images_unaug, batches[1].images_unaug)

    def test_map_batches(self):
        self._test_map_batches_both(call_async=False)

    def test_map_batches_async(self):
        self._test_map_batches_both(call_async=True)

    def _test_imap_batches_both(self, call_unordered):
        batches = [ia.Batch(images=[ia.quokka()]), ia.Batch(images=[ia.quokka()+1])]

        def _generate_batches():
            for batch in batches:
                yield batch

        augseq = iaa.Noop()
        mock_Pool = mock.MagicMock()
        mock_Pool.return_value = mock_Pool
        mock_Pool.imap.return_value = batches
        mock_Pool.imap_unordered.return_value = batches
        with mock.patch("multiprocessing.Pool", mock_Pool):
            with multicore.Pool(augseq, processes=1) as pool:
                gen = _generate_batches()
                if call_unordered:
                    _ = list(pool.imap_batches_unordered(gen))
                else:
                    _ = list(pool.imap_batches(gen))

            if call_unordered:
                to_check = mock_Pool.imap_unordered
            else:
                to_check = mock_Pool.imap

            assert to_check.call_count == 1
            assert to_check.call_args[0][0] == multicore._Pool_starworker
            arg_batches = list(to_check.call_args[0][1])  # convert generator to list, make it subscriptable
            # args, arg 1 (batches with ids), tuple 0, entry 0 in tuple (=> batch id)
            assert arg_batches[0][0] == 0
            # tuple 0, entry 1 in tuple (=> batch)
            assert np.array_equal(arg_batches[0][1].images_unaug, batches[0].images_unaug)
            # tuple 1, entry 0 in tuple (=> batch id)
            assert arg_batches[1][0] == 1
            # tuple 1, entry 1 in tuple (=> batch)
            assert np.array_equal(arg_batches[1][1].images_unaug, batches[1].images_unaug)

    def test_imap_batches(self):
        self._test_imap_batches_both(call_unordered=False)

    def test_imap_batches_unordered(self):
        self._test_imap_batches_both(call_unordered=True)

    def _assert_each_augmentation_not_more_than_once(self, batches_aug):
        sum_to_vecs = defaultdict(list)
        for batch in batches_aug:
            assert not np.array_equal(batch.images_aug[0], batch.images_aug[1])

            vec = batch.images_aug.flatten()
            vecsum = int(np.sum(vec))
            if vecsum in sum_to_vecs:
                for other_vec in sum_to_vecs[vecsum]:
                    assert not np.array_equal(vec, other_vec)
            else:
                sum_to_vecs[vecsum].append(vec)

    def test_augmentations_with_seed_match(self):
        augseq = iaa.AddElementwise((0, 255))
        image = np.zeros((10, 10, 1), dtype=np.uint8)
        batch = ia.Batch(images=np.uint8([image, image]))
        batches = [batch.deepcopy() for _ in sm.xrange(60)]

        # seed=1
        with multicore.Pool(augseq, processes=2, maxtasksperchild=30, seed=1) as pool:
            batches_aug1 = pool.map_batches(batches, chunksize=2)
        # seed=1
        with multicore.Pool(augseq, processes=2, seed=1) as pool:
            batches_aug2 = pool.map_batches(batches, chunksize=1)
        # seed=2
        with multicore.Pool(augseq, processes=2, seed=2) as pool:
            batches_aug3 = pool.map_batches(batches, chunksize=1)

        assert len(batches_aug1) == 60
        assert len(batches_aug2) == 60
        assert len(batches_aug3) == 60

        for b1, b2, b3 in zip(batches_aug1, batches_aug2, batches_aug3):
            # images were augmented
            assert not np.array_equal(b1.images_unaug, b1.images_aug)
            assert not np.array_equal(b2.images_unaug, b2.images_aug)
            assert not np.array_equal(b3.images_unaug, b3.images_aug)

            # original images still the same
            assert np.array_equal(b1.images_unaug, batch.images_unaug)
            assert np.array_equal(b2.images_unaug, batch.images_unaug)
            assert np.array_equal(b3.images_unaug, batch.images_unaug)

            # augmentations for same seed are the same
            assert np.array_equal(b1.images_aug, b2.images_aug)

            # augmentations for different seeds are different
            assert not np.array_equal(b1.images_aug, b3.images_aug)

        # make sure that batches for the two pools with same seed did not repeat within results (only between the
        # results of the two pools)
        for batches_aug in [batches_aug1, batches_aug2, batches_aug3]:
            self._assert_each_augmentation_not_more_than_once(batches_aug)

    def test_augmentations_with_seed_match_for_images_and_keypoints(self):
        augseq = iaa.AddElementwise((0, 255))
        image = np.zeros((10, 10, 1), dtype=np.uint8)
        # keypoints here will not be changed by augseq, but they will induce deterministic mode to start in
        # augment_batches() as each batch contains images AND keypoints
        kps = ia.KeypointsOnImage([ia.Keypoint(x=2, y=0)], shape=(10, 10, 1))
        batch = ia.Batch(images=np.uint8([image, image]), keypoints=[kps, kps])
        batches = [batch.deepcopy() for _ in sm.xrange(60)]

        # seed=1
        with multicore.Pool(augseq, processes=2, maxtasksperchild=30, seed=1) as pool:
            batches_aug1 = pool.map_batches(batches, chunksize=2)
        # seed=1
        with multicore.Pool(augseq, processes=2, seed=1) as pool:
            batches_aug2 = pool.map_batches(batches, chunksize=1)
        # seed=2
        with multicore.Pool(augseq, processes=2, seed=2) as pool:
            batches_aug3 = pool.map_batches(batches, chunksize=1)

        assert len(batches_aug1) == 60
        assert len(batches_aug2) == 60
        assert len(batches_aug3) == 60

        for batches_aug in [batches_aug1, batches_aug2, batches_aug3]:
            for batch in batches_aug:
                for keypoints_aug in batch.keypoints_aug:
                    assert keypoints_aug.keypoints[0].x == 2
                    assert keypoints_aug.keypoints[0].y == 0

        for b1, b2, b3 in zip(batches_aug1, batches_aug2, batches_aug3):
            # images were augmented
            assert not np.array_equal(b1.images_unaug, b1.images_aug)
            assert not np.array_equal(b2.images_unaug, b2.images_aug)
            assert not np.array_equal(b3.images_unaug, b3.images_aug)

            # original images still the same
            assert np.array_equal(b1.images_unaug, batch.images_unaug)
            assert np.array_equal(b2.images_unaug, batch.images_unaug)
            assert np.array_equal(b3.images_unaug, batch.images_unaug)

            # augmentations for same seed are the same
            assert np.array_equal(b1.images_aug, b2.images_aug)

            # augmentations for different seeds are different
            assert not np.array_equal(b1.images_aug, b3.images_aug)

        # make sure that batches for the two pools with same seed did not repeat within results (only between the
        # results of the two pools)
        for batches_aug in [batches_aug1, batches_aug2, batches_aug3]:
            self._assert_each_augmentation_not_more_than_once(batches_aug)

    def test_augmentations_without_seed_differ(self):
        augseq = iaa.AddElementwise((0, 255))
        image = np.zeros((10, 10, 1), dtype=np.uint8)
        batch = ia.Batch(images=np.uint8([image, image]))
        batches = [batch.deepcopy() for _ in sm.xrange(20)]
        with multicore.Pool(augseq, processes=2, maxtasksperchild=5) as pool:
            batches_aug = pool.map_batches(batches, chunksize=2)
        with multicore.Pool(augseq, processes=2) as pool:
            batches_aug.extend(pool.map_batches(batches, chunksize=1))

        assert len(batches_aug) == 2*20

        self._assert_each_augmentation_not_more_than_once(batches_aug)

    def test_augmentations_without_seed_differ_for_images_and_keypoints(self):
        augseq = iaa.AddElementwise((0, 255))
        image = np.zeros((10, 10, 1), dtype=np.uint8)
        # keypoints here will not be changed by augseq, but they will induce deterministic mode to start in
        # augment_batches() as each batch contains images AND keypoints
        kps = ia.KeypointsOnImage([ia.Keypoint(x=2, y=0)], shape=(10, 10, 1))
        batch = ia.Batch(images=np.uint8([image, image]), keypoints=[kps, kps])
        batches = [batch.deepcopy() for _ in sm.xrange(20)]
        with multicore.Pool(augseq, processes=2, maxtasksperchild=5) as pool:
            batches_aug = pool.map_batches(batches, chunksize=2)
        with multicore.Pool(augseq, processes=2) as pool:
            batches_aug.extend(pool.map_batches(batches, chunksize=1))

        assert len(batches_aug) == 2*20

        for batch in batches_aug:
            for keypoints_aug in batch.keypoints_aug:
                assert keypoints_aug.keypoints[0].x == 2
                assert keypoints_aug.keypoints[0].y == 0

        self._assert_each_augmentation_not_more_than_once(batches_aug)

    def test_inputs_not_lost(self):
        """Test to make sure that inputs (e.g. images) are never lost."""
        def _assert_contains_all_ids(batches_aug):
            # batch.images_unaug
            ids = set()
            for batch_aug in batches_aug:
                ids.add(int(batch_aug.images_unaug.flat[0]))
                ids.add(int(batch_aug.images_unaug.flat[1]))
            for idx in sm.xrange(2*100):
                assert idx in ids
            assert len(ids) == 200

            # batch.images_aug
            ids = set()
            for batch_aug in batches_aug:
                ids.add(int(batch_aug.images_aug.flat[0]))
                ids.add(int(batch_aug.images_aug.flat[1]))
            for idx in sm.xrange(2*100):
                assert idx in ids
            assert len(ids) == 200

        augseq = iaa.Noop()
        image = np.zeros((1, 1, 1), dtype=np.uint8)
        # creates batches containing images with ids from 0 to 199 (one pair of consecutive ids per batch)
        batches = [ia.Batch(images=np.uint8([image + b_idx*2, image + b_idx*2+1]))
                   for b_idx in sm.xrange(100)]

        with multicore.Pool(augseq, processes=2, maxtasksperchild=25) as pool:
            batches_aug = pool.map_batches(batches)
            _assert_contains_all_ids(batches_aug)

        with multicore.Pool(augseq, processes=2, maxtasksperchild=25, seed=1) as pool:
            batches_aug = pool.map_batches(batches)
            _assert_contains_all_ids(batches_aug)

        with multicore.Pool(augseq, processes=3, seed=2) as pool:
            batches_aug = pool.map_batches(batches)
            _assert_contains_all_ids(batches_aug)

        with multicore.Pool(augseq, processes=2, seed=None) as pool:
            batches_aug = pool.map_batches(batches)
            _assert_contains_all_ids(batches_aug)

            batches_aug = pool.map_batches(batches)
            _assert_contains_all_ids(batches_aug)

    def test_close(self):
        augseq = iaa.Noop()
        with multicore.Pool(augseq, processes=2) as pool:
            pool.close()

    def test_terminate(self):
        augseq = iaa.Noop()
        with multicore.Pool(augseq, processes=2) as pool:
            pool.terminate()

    def test_join(self):
        augseq = iaa.Noop()
        with multicore.Pool(augseq, processes=2) as pool:
            pool.close()
            pool.join()


def test_BatchLoader():
    reseed()

    def _load_func():
        for _ in sm.xrange(20):
            yield ia.Batch(images=np.zeros((2, 4, 4, 3), dtype=np.uint8))

    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as caught_warnings:
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

        assert len(caught_warnings) > 0
        for warning in caught_warnings:
            assert "is deprecated" in str(warning.message)


def test_BackgroundAugmenter__augment_images_worker():
    reseed()

    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as caught_warnings:
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
        assert batch_aug.images_unaug is not None
        assert batch_aug.images_unaug.dtype == np.uint8
        assert batch_aug.images_unaug.shape == (1, 4, 8, 3)
        assert np.array_equal(batch_aug.images_unaug, np.zeros((1, 4, 8, 3), dtype=np.uint8))
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

    assert len(caught_warnings) > 0
    for warning in caught_warnings:
        assert "is deprecated" in str(warning.message)


if __name__ == "__main__":
    main()
