from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from scipy import misc
import numpy as np

def main():
    nb_rows = 8
    nb_cols = 8
    h, w = (128, 128)
    sample_size = 128

    noise_gens = [
        iap.SimplexNoise(),
        iap.FrequencyNoise(exponent=-4, size_px_max=sample_size, upscale_method="cubic"),
        iap.FrequencyNoise(exponent=-2, size_px_max=sample_size, upscale_method="cubic"),
        iap.FrequencyNoise(exponent=0, size_px_max=sample_size, upscale_method="cubic"),
        iap.FrequencyNoise(exponent=2, size_px_max=sample_size, upscale_method="cubic"),
        iap.FrequencyNoise(exponent=4, size_px_max=sample_size, upscale_method="cubic"),
        iap.FrequencyNoise(exponent=(-4, 4), size_px_max=(4, sample_size), upscale_method=["nearest", "linear", "cubic"]),
        iap.IterativeNoiseAggregator(
            other_param=iap.FrequencyNoise(exponent=(-4, 4), size_px_max=(4, sample_size), upscale_method=["nearest", "linear", "cubic"]),
            iterations=(1, 3),
            aggregation_method=["max", "avg"]
        ),
        iap.IterativeNoiseAggregator(
            other_param=iap.Sigmoid(
                iap.FrequencyNoise(exponent=(-4, 4), size_px_max=(4, sample_size), upscale_method=["nearest", "linear", "cubic"]),
                threshold=(-10, 10),
                activated=0.33,
                mul=20,
                add=-10
            ),
            iterations=(1, 3),
            aggregation_method=["max", "avg"]
        )
    ]

    samples = [[] for _ in range(len(noise_gens))]
    for _ in range(nb_rows * nb_cols):
        for i, noise_gen in enumerate(noise_gens):
            samples[i].append(noise_gen.draw_samples((h, w)))

    rows = [np.hstack(row) for row in samples]
    grid = np.vstack(rows)
    misc.imshow((grid*255).astype(np.uint8))

    images = [ia.quokka_square(size=(128, 128)) for _ in range(16)]
    seqs = [
        iaa.SimplexNoiseAlpha(first=iaa.EdgeDetect(1.0)),
        iaa.SimplexNoiseAlpha(first=iaa.EdgeDetect(1.0), per_channel=True),
        iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(1.0)),
        iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(1.0), per_channel=True)
    ]
    images_aug = []

    for seq in seqs:
        images_aug.append(np.hstack(seq.augment_images(images)))
    images_aug = np.vstack(images_aug)
    misc.imshow(images_aug)

if __name__ == "__main__":
    main()
