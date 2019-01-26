from __future__ import print_function, division, absolute_import
import sys
import multiprocessing
import threading
import traceback
import time
import random

import numpy as np

from . import imgaug as ia

if sys.version_info[0] == 2:
    import cPickle as pickle
    from Queue import Empty as QueueEmpty, Full as QueueFull
    import socket
    BrokenPipeError = socket.error
elif sys.version_info[0] == 3:
    import pickle
    from queue import Empty as QueueEmpty, Full as QueueFull


class BatchLoader(object):
    """
    Class to load batches in the background.

    Loaded batches can be accesses using :attr:`imgaug.BatchLoader.queue`.

    Parameters
    ----------
    load_batch_func : callable or generator
        Generator or generator function (i.e. function that yields Batch objects)
        or a function that returns a list of Batch objects.
        Background loading automatically stops when the last batch was yielded or the
        last batch in the list was reached.

    queue_size : int, optional
        Maximum number of batches to store in the queue. May be set higher
        for small images and/or small batches.

    nb_workers : int, optional
        Number of workers to run in the background.

    threaded : bool, optional
        Whether to run the background processes using threads (True) or full processes (False).

    """

    def __init__(self, load_batch_func, queue_size=50, nb_workers=1, threaded=True):
        ia.do_assert(queue_size >= 2, "Queue size for BatchLoader must be at least 2, got %d." % (queue_size,))
        ia.do_assert(nb_workers >= 1, "Number of workers for BatchLoader must be at least 1, got %d" % (nb_workers,))
        self._queue_internal = multiprocessing.Queue(queue_size//2)
        self.queue = multiprocessing.Queue(queue_size//2)
        self.join_signal = multiprocessing.Event()
        self.workers = []
        self.threaded = threaded
        seeds = ia.current_random_state().randint(0, 10**6, size=(nb_workers,))
        for i in range(nb_workers):
            if threaded:
                worker = threading.Thread(
                    target=self._load_batches,
                    args=(load_batch_func, self._queue_internal, self.join_signal, None)
                )
            else:
                worker = multiprocessing.Process(
                    target=self._load_batches,
                    args=(load_batch_func, self._queue_internal, self.join_signal, seeds[i])
                )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        self.main_worker_thread = threading.Thread(
            target=self._main_worker,
            args=()
        )
        self.main_worker_thread.daemon = True
        self.main_worker_thread.start()

    def count_workers_alive(self):
        return sum([int(worker.is_alive()) for worker in self.workers])

    def all_finished(self):
        """
        Determine whether the workers have finished the loading process.

        Returns
        -------
        out : bool
            True if all workers have finished. Else False.

        """
        return self.count_workers_alive() == 0

    def _main_worker(self):
        workers_running = self.count_workers_alive()

        while workers_running > 0 and not self.join_signal.is_set():
            # wait for a new batch in the source queue and load it
            try:
                batch_str = self._queue_internal.get(timeout=0.1)
                if batch_str == "":
                    workers_running -= 1
                else:
                    self.queue.put(batch_str)
            except QueueEmpty:
                time.sleep(0.01)
            except (EOFError, BrokenPipeError):
                break

            workers_running = self.count_workers_alive()

        # All workers have finished, move the remaining entries from internal to external queue
        while True:
            try:
                batch_str = self._queue_internal.get(timeout=0.005)
                if batch_str != "":
                    self.queue.put(batch_str)
            except QueueEmpty:
                break
            except (EOFError, BrokenPipeError):
                break

        self.queue.put(pickle.dumps(None, protocol=-1))
        time.sleep(0.01)

    def _load_batches(self, load_batch_func, queue_internal, join_signal, seedval):
        if seedval is not None:
            random.seed(seedval)
            np.random.seed(seedval)
            ia.seed(seedval)

        try:
            gen = load_batch_func() if not ia.is_generator(load_batch_func) else load_batch_func
            for batch in gen:
                ia.do_assert(isinstance(batch, ia.Batch),
                             "Expected batch returned by load_batch_func to be of class imgaug.Batch, got %s." % (
                                 type(batch),))
                batch_pickled = pickle.dumps(batch, protocol=-1)
                while not join_signal.is_set():
                    try:
                        queue_internal.put(batch_pickled, timeout=0.005)
                        break
                    except QueueFull:
                        pass
                if join_signal.is_set():
                    break
        except Exception:
            traceback.print_exc()
        finally:
            queue_internal.put("")
        time.sleep(0.01)

    def terminate(self):
        """Stop all workers."""
        if not self.join_signal.is_set():
            self.join_signal.set()
        # give minimal time to put generated batches in queue and gracefully shut down
        time.sleep(0.01)

        if self.main_worker_thread.is_alive():
            self.main_worker_thread.join()

        if self.threaded:
            for worker in self.workers:
                if worker.is_alive():
                    worker.join()
        else:
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

            # wait until all workers are fully terminated
            while not self.all_finished():
                time.sleep(0.001)

        # empty queue until at least one element can be added and place None as signal that BL finished
        if self.queue.full():
            self.queue.get()
        self.queue.put(pickle.dumps(None, protocol=-1))
        time.sleep(0.01)

        # clean the queue, this reportedly prevents hanging threads
        while True:
            try:
                self._queue_internal.get(timeout=0.005)
            except QueueEmpty:
                break

        if not self._queue_internal._closed:
            self._queue_internal.close()
        if not self.queue._closed:
            self.queue.close()
        self._queue_internal.join_thread()
        self.queue.join_thread()
        time.sleep(0.025)

    def __del__(self):
        if not self.join_signal.is_set():
            self.join_signal.set()


class BackgroundAugmenter(object):
    """
    Class to augment batches in the background (while training on the GPU).

    This is a wrapper around the multiprocessing module.

    Parameters
    ----------
    batch_loader : BatchLoader or multiprocessing.Queue
        BatchLoader object that loads the data fed into the BackgroundAugmenter, or alternatively a Queue.
        If a Queue, then it must be made sure that a final ``None`` in the Queue signals that the loading is
        finished and no more batches will follow. Otherwise the BackgroundAugmenter will wait forever for the next
        batch.

    augseq : Augmenter
        An augmenter to apply to all loaded images.
        This may be e.g. a Sequential to apply multiple augmenters.

    queue_size : int
        Size of the queue that is used to temporarily save the augmentation
        results. Larger values offer the background processes more room
        to save results when the main process doesn't load much, i.e. they
        can lead to smoother and faster training. For large images, high
        values can block a lot of RAM though.

    nb_workers : 'auto' or int
        Number of background workers to spawn.
        If ``auto``, it will be set to ``C-1``, where ``C`` is the number of CPU cores.

    """
    def __init__(self, batch_loader, augseq, queue_size=50, nb_workers="auto"):
        ia.do_assert(queue_size > 0)
        self.augseq = augseq
        self.queue_source = batch_loader if isinstance(batch_loader, multiprocessing.queues.Queue) else batch_loader.queue
        self.queue_result = multiprocessing.Queue(queue_size)

        if nb_workers == "auto":
            try:
                nb_workers = multiprocessing.cpu_count()
            except (ImportError, NotImplementedError):
                nb_workers = 1
            # try to reserve at least one core for the main process
            nb_workers = max(1, nb_workers - 1)
        else:
            ia.do_assert(nb_workers >= 1)

        self.nb_workers = nb_workers
        self.workers = []
        self.nb_workers_finished = 0

        seeds = ia.current_random_state().randint(0, 10**6, size=(nb_workers,))
        for i in range(nb_workers):
            worker = multiprocessing.Process(
                target=self._augment_images_worker,
                args=(augseq, self.queue_source, self.queue_result, seeds[i])
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def all_finished(self):
        return self.nb_workers_finished == self.nb_workers

    def get_batch(self):
        """
        Returns a batch from the queue of augmented batches.

        If workers are still running and there are no batches in the queue,
        it will automatically wait for the next batch.

        Returns
        -------
        out : None or imgaug.Batch
            One batch or None if all workers have finished.

        """
        if self.all_finished():
            return None

        batch_str = self.queue_result.get()
        batch = pickle.loads(batch_str)
        if batch is not None:
            return batch
        else:
            self.nb_workers_finished += 1
            if self.nb_workers_finished >= self.nb_workers:
                try:
                    self.queue_source.get(timeout=0.001)  # remove the None from the source queue
                except QueueEmpty:
                    pass
                return None
            else:
                return self.get_batch()

    def _augment_images_worker(self, augseq, queue_source, queue_result, seedval):
        """
        Augment endlessly images in the source queue.

        This is a worker function for that endlessly queries the source queue (input batches),
        augments batches in it and sends the result to the output queue.

        """
        np.random.seed(seedval)
        random.seed(seedval)
        augseq.reseed(seedval)
        ia.seed(seedval)

        loader_finished = False

        while not loader_finished:
            # wait for a new batch in the source queue and load it
            try:
                batch_str = queue_source.get(timeout=0.1)
                batch = pickle.loads(batch_str)
                if batch is None:
                    loader_finished = True
                    # put it back in so that other workers know that the loading queue is finished
                    queue_source.put(pickle.dumps(None, protocol=-1))
                else:
                    batch_aug = list(augseq.augment_batches([batch], background=False))[0]

                    # send augmented batch to output queue
                    batch_str = pickle.dumps(batch_aug, protocol=-1)
                    queue_result.put(batch_str)
            except QueueEmpty:
                time.sleep(0.01)

        queue_result.put(pickle.dumps(None, protocol=-1))
        time.sleep(0.01)

    def terminate(self):
        """
        Terminates all background processes immediately.

        This will also free their RAM.

        """
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
        self.nb_workers_finished = len(self.workers)

        if not self.queue_result._closed:
            self.queue_result.close()
        time.sleep(0.01)

    def __del__(self):
        time.sleep(0.1)
        self.terminate()
