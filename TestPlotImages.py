import numpy as np
from ImageAugmenter import ImageAugmenter
import random
from scipy import misc # to show images
from skimage import data

def main():
    image = data.lena()
    ia = ImageAugmenter(image.shape[0], image.shape[1],
                        hflip=True, vflip=True,
                        scale_to_percent=1.3, scale_axis_equally=False,
                        rotation_deg=25, shear_deg=10,
                        translation_x_px=5, translation_y_px=5)

    ia.plot_image(image, 100)

if __name__ == "__main__":
    main()
