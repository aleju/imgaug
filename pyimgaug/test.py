from __future__ import print_function, division
from pyimgaug import AugmenterSequence, Fliplr, Flipud, BinomialParameter, DeterministicParameter
#from skimage import
import numpy as np
import time

def main():
    seq = AugmenterSequence([Fliplr(0.5), Flipud(0.5)])
    imgs = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    imgs[0, 0, :, :] = 255
    imgs[0, 1, 1, :] = 255

    print("imgs", imgs[:, :, :, 0])
    for i in range(10):
        auged = seq.transform(imgs)
        print("C1", auged[:, :, :, 0])
        print("C2", auged[:, :, :, 1])
        print("C3", auged[:, :, :, 2])

    dseq = seq.to_deterministic(4)
    for di in range(4):
        print("-----------------------")
        print("dseq ", di)
        print([augmenter.params for augmenter in dseq[di].augmenters])
        print("-----------------------")
        for i in range(4):
            auged = dseq[di].transform(imgs)
            print("C1", auged[:, :, :, 0])

    times = []
    for i in range(100):
        start = time.time()
        dseq = seq.to_deterministic(1000)
        times.append(time.time() - start)
    print("[Time 1] avg=%.4f, var=%.4f, range=[%.4f, %.4f]" % (np.average(times), np.var(times), np.min(times), np.max(times)))

    param = BinomialParameter(0.5)
    start = time.time()
    for i in range(1000 * 1000):
        DeterministicParameter(param.draw_sample())
    req = time.time() - start
    print("[Time 2] %.4f per 1M, %.4f per 1k" % (req, req/1000))

    param = 1
    start = time.time()
    for i in range(1000 * 1000):
        DeterministicParameter(1)
    req = time.time() - start
    print("[Time 3] %.4f per 1M, %.4f per 1k" % (req, req/1000))

if __name__ == "__main__":
    main()
