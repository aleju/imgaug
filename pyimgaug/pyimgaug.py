from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import random
import numpy as np
import copy

try:
    xrange
except NameError:  # python3
    xrange = range

def is_np_array(val):
    return isinstance(val, (np.ndarray, np.generic))


class BackgroundAugmenter(object):
    def __init__(self, image_source, augmenter, maxlen, nb_workers=1):
        self.augmenter = augmenter
        self.maxlen = maxlen
        self.result_queue = multiprocessing.Queue(maxlen)
        self.batch_workers = []
        for i in range(nb_workers):
            worker = multiprocessing.Process(target=self._augment, args=(image_source, augmenter, self.result_queue))
            worker.daemon = True
            worker.start()
            self.batch_workers.append(worker)

    def join(self):
        for worker in self.batch_workers:
            worker.join()

    def get_batch(self):
        return self.result_queue.get()

    def _augment(self, image_source, augmenter, result_queue):
        batch = next(image_source)
        self.result_queue.put(augmenter.transform(batch))
