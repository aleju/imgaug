from __future__ import print_function, division
import augmenters2 as iaa
import parameters as iap
#from skimage import
import numpy as np
import time

def main():
    seq = iaa.Sequence([iaa.Fliplr(0.5), iaa.Flipud(0.5)])
    imgs = np.zeros((1, 2, 2, 1), dtype=np.uint8)
    imgs[0, 0, :, 0] = 255
    imgs[0, 1, 1, 0] = 255

    print("[Test 1] random hflips/vflips")
    print("imgs", imgs[:, :, :, 0])
    for i in range(10):
        auged = seq.transform(imgs)
        print("#%02d" % (i,), auged[:, :, :, 0])

    print("[Test 2] deterministic hflips/vflips")
    dseqs = seq.to_deterministic(4)
    for dseq_i, dseq in enumerate(dseqs):
        for i in range(10):
            auged = dseq.transform(imgs)
            print("#%02d/%02d" % (dseq_i, i,), auged[:, :, :, 0])

    print("[Time Measurements]")
    times = []
    for i in range(1):
        start = time.time()
        dseq = seq.to_deterministic(1000)
        times.append(time.time() - start)
    print("[Time 1] avg=%.4f, var=%.4f, range=[%.4f, %.4f]" % (np.average(times), np.var(times), np.min(times), np.max(times)))

    param = 1
    start = time.time()
    for i in range(1000 * 1000):
        iaa.Deterministic(1)
    req = time.time() - start
    print("[Time 2] %.4f per 1M, %.4f per 1k" % (req, req/1000))

    param = iap.Binomial(0.5)
    start = time.time()
    for i in range(1000 * 1000):
        param.draw_sample(random_state=np.random)
    req = time.time() - start
    print("[Time 3] %.4f per 1M, %.4f per 1k" % (req, req/1000))

    param = iap.Binomial(0.5)
    start = time.time()
    for i in range(1000 * 1000):
        param.draw_sample()
    req = time.time() - start
    print("[Time 4] %.4f per 1M, %.4f per 1k" % (req, req/1000))

if __name__ == "__main__":
    main()
