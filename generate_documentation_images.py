from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
from scipy import ndimage, misc
#from skimage import data
#import matplotlib.pyplot as plt
#from matplotlib import gridspec
#import six
#import six.moves as sm
import os
import PIL.Image
import math
from skimage import data

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

DOCS_IMAGES_BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "docs",
    "images"
)

PARAMETERS_DEFAULT_SIZE = (350, 350)
PARAMETER_DEFAULT_QUALITY = 25

def main():
    chapter_examples_basics()
    chapter_examples_keypoints()
    chapter_examples_bounding_boxes()
    chapter_augmenters()
    chapter_parameters()
    chapter_alpha()

def save(chapter_dir, filename, image, quality=None):
    dir_fp = os.path.join(DOCS_IMAGES_BASE_PATH, chapter_dir)
    if not os.path.exists(dir_fp):
        os.makedirs(dir_fp)
    file_fp = os.path.join(dir_fp, filename)

    image_jpg = compress_to_jpg(image, quality=quality)
    image_jpg_decompressed = decompress_jpg(image_jpg)

    # If the image file already exists and is (practically) identical,
    # then don't save it again to avoid polluting the repository with tons
    # of image updates.
    # Not that we have to compare here the results AFTER jpg compression
    # and then decompression. Otherwise we compare two images of which
    # image (1) has never been compressed while image (2) was compressed and
    # then decompressed.
    if os.path.isfile(file_fp):
        image_saved = ndimage.imread(file_fp, mode="RGB")
        #print("arrdiff", arrdiff(image_jpg_decompressed, image_saved))
        same_shape = (image_jpg_decompressed.shape == image_saved.shape)
        d_avg = arrdiff(image_jpg_decompressed, image_saved) if same_shape else -1
        if same_shape and d_avg <= 1.0:
            print("[INFO] Did not save image '%s/%s', because the already saved image is basically identical (d_avg=%.4f)" % (chapter_dir, filename, d_avg,))
            return

    with open(file_fp, "w") as f:
        f.write(image_jpg)

def arrdiff(arr1, arr2):
    nb_cells = np.prod(arr2.shape)
    d_avg = np.sum(np.power(np.abs(arr1 - arr2), 2)) / nb_cells
    return d_avg

def compress_to_jpg(image, quality=75):
    quality = quality if quality is not None else 75
    im = PIL.Image.fromarray(image)
    out = BytesIO()
    im.save(out, format="JPEG", quality=quality)
    jpg_string = out.getvalue()
    out.close()
    return jpg_string

def decompress_jpg(image_compressed):
    img_compressed_buffer = BytesIO()
    img_compressed_buffer.write(image_compressed)
    img = ndimage.imread(img_compressed_buffer, mode="RGB")
    img_compressed_buffer.close()
    return img

def grid(images, rows, cols, border=1, border_color=255):
    nb_images = len(images)
    cell_height = max([image.shape[0] for image in images])
    cell_width = max([image.shape[1] for image in images])
    channels = set([image.shape[2] for image in images])
    assert len(channels) == 1
    nb_channels = list(channels)[0]
    if rows is None and cols is None:
        rows = cols = int(math.ceil(math.sqrt(nb_images)))
    elif rows is not None:
        cols = int(math.ceil(nb_images / rows))
    elif cols is not None:
        rows = int(math.ceil(nb_images / cols))
    assert rows * cols >= nb_images

    cell_height = cell_height + 1 * border
    cell_width = cell_width + 1 * border

    width = cell_width * cols
    height = cell_height * rows
    grid = np.zeros((height, width, nb_channels), dtype=np.uint8)
    cell_idx = 0
    for row_idx in range(rows):
        for col_idx in range(cols):
            if cell_idx < nb_images:
                image = images[cell_idx]
                border_top = border_right = border_bottom = border_left = border
                #if row_idx > 1:
                border_top = 0
                #if col_idx > 1:
                border_left = 0
                image = np.pad(image, ((border_top, border_bottom), (border_left, border_right), (0, 0)), mode="constant", constant_values=border_color)

                cell_y1 = cell_height * row_idx
                cell_y2 = cell_y1 + image.shape[0]
                cell_x1 = cell_width * col_idx
                cell_x2 = cell_x1 + image.shape[1]
                grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image
            cell_idx += 1

    grid = np.pad(grid, ((border, 0), (border, 0), (0, 0)), mode="constant", constant_values=border_color)

    return grid

def checkerboard(size):
    img = data.checkerboard()
    img3d = np.tile(img[..., np.newaxis], (1, 1, 3))
    return misc.imresize(img3d, size)

###############################
# Examples: Basics
###############################

def chapter_examples_basics():
    """Generate all example images for the chapter `Examples: Basics`
    in the documentation."""
    chapter_examples_basics_simple()
    chapter_examples_basics_heavy()

def chapter_examples_basics_simple():
    import imgaug as ia
    from imgaug import augmenters as iaa

    # Example batch of images.
    # The array has shape (32, 64, 64, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(64, 64)) for _ in range(32)],
        dtype=np.uint8
    )

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    ia.seed(1)
    images_aug = seq.augment_images(images)

    # ------------

    save(
        "examples_basics",
        "simple.jpg",
        grid(images_aug, cols=8, rows=4)
    )

def chapter_examples_basics_heavy():
    import imgaug as ia
    from imgaug import augmenters as iaa
    import numpy as np

    # Example batch of images.
    # The array has shape (32, 64, 64, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(64, 64)) for _ in range(32)],
        dtype=np.uint8
    )

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                [
                    # Convert some images into their superpixel representation,
                    # sample between 20 and 200 superpixels per image, but do
                    # not replace all superpixels with their average, only
                    # some of them (p_replace).
                    sometimes(
                        iaa.Superpixels(
                            p_replace=(0, 1.0),
                            n_segments=(20, 200)
                        )
                    ),

                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),

                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                    # Search in some images either for all edges or for
                    # directed edges. These edges are then marked in a black
                    # and white image and overlayed with the original image
                    # using an alpha of 0 to 0.7.
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.7)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.7), direction=(0.0, 1.0)
                        ),
                    ])),

                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled per
                    # channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                    ),

                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5% percent
                    # of the original size, leading to large dropped
                    # rectangles.
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15), size_percent=(0.02, 0.05),
                            per_channel=0.2
                        ),
                    ]),

                    # Invert each image's chanell with 5% probability.
                    # This sets each pixel value v to 255-v.
                    iaa.Invert(0.05, per_channel=True), # invert color channels

                    # Add a value of -10 to 10 to each pixel.
                    iaa.Add((-10, 10), per_channel=0.5),

                    # Change brightness of images (50-150% of original value).
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),

                    # Improve or worsen the contrast of images.
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e. remove
                    # colors with varying strengths.
                    iaa.Grayscale(alpha=(0.0, 1.0)),

                    # In some images move pixels locally around (with random
                    # strengths).
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                    ),

                    # In some images distort local areas with varying strength.
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ],
                # do all of the above augmentations in random order
                random_order=True
            )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    ia.seed(1)
    images_aug = seq.augment_images(images)

    # ------------

    save(
        "examples_basics",
        "heavy.jpg",
        grid(images_aug, cols=8, rows=4)
    )

###############################
# Examples: Keypoints
###############################

def chapter_examples_keypoints():
    """Generate all example images for the chapter `Examples: Keypoints`
    in the documentation."""
    chapter_examples_keypoints_simple()

def chapter_examples_keypoints_simple():
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    keypoints = ia.KeypointsOnImage([
        ia.Keypoint(x=65, y=100),
        ia.Keypoint(x=75, y=200),
        ia.Keypoint(x=100, y=100),
        ia.Keypoint(x=200, y=80)
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
        iaa.Affine(
            rotate=10,
            scale=(0.5, 0.7)
        ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    ])

    # Make our sequence deterministic.
    # We can now apply it to the image and then to the keypoints and it will
    # lead to the same augmentations.
    # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
    # exactly same augmentations for every batch!
    seq_det = seq.to_deterministic()

    # augment keypoints and images
    image_aug = seq_det.augment_images([image])[0]
    keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

    # print coordinates before/after augmentation (see below)
    for i in range(len(keypoints.keypoints)):
        before = keypoints.keypoints[i]
        after = keypoints_aug.keypoints[i]
        print("Keypoint %d: (%d, %d) -> (%d, %d)" % (
            i, before.x, before.y, after.x, after.y)
        )

    # image with keypoints before/after augmentation (shown below)
    image_before = keypoints.draw_on_image(image, size=7)
    image_after = keypoints_aug.draw_on_image(image_aug, size=7)

    # ------------

    save(
        "examples_keypoints",
        "simple.jpg",
        grid([image_before, image_after], cols=2, rows=1),
        quality=90
    )

###############################
# Examples: Bounding Boxes
###############################

def chapter_examples_bounding_boxes():
    """Generate all example images for the chapter `Examples: Bounding Boxes`
    in the documentation."""
    chapter_examples_bounding_boxes_simple()
    chapter_examples_bounding_boxes_rotation()
    chapter_examples_bounding_boxes_ooi()
    chapter_examples_bounding_boxes_shift()
    chapter_examples_bounding_boxes_projection()
    chapter_examples_bounding_boxes_iou()

def chapter_examples_bounding_boxes_simple():
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=65, y1=100, x2=200, y2=150),
        ia.BoundingBox(x1=150, y1=80, x2=200, y2=130)
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    # Make our sequence deterministic.
    # We can now apply it to the image and then to the BBs and it will
    # lead to the same augmentations.
    # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
    # exactly same augmentations for every batch!
    seq_det = seq.to_deterministic()

    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the
    # functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # print coordinates before/after augmentation (see below)
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%d, %d, %d, %d) -> (%d, %d, %d, %d)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
        )

    # image with BBs before/after augmentation (shown below)
    image_before = bbs.draw_on_image(image, thickness=2)
    image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

    # ------------

    save(
        "examples_bounding_boxes",
        "simple.jpg",
        grid([image_before, image_after], cols=2, rows=1),
        quality=75
    )

def chapter_examples_bounding_boxes_rotation():
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=65, y1=100, x2=200, y2=150),
        ia.BoundingBox(x1=150, y1=80, x2=200, y2=130)
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        iaa.Affine(
            rotate=45,
        )
    ])

    # Make our sequence deterministic.
    # We can now apply it to the image and then to the BBs and it will
    # lead to the same augmentations.
    # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
    # exactly same augmentations for every batch!
    seq_det = seq.to_deterministic()

    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the
    # functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # print coordinates before/after augmentation (see below)
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%d, %d, %d, %d) -> (%d, %d, %d, %d)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
        )

    # image with BBs before/after augmentation (shown below)
    image_before = bbs.draw_on_image(image, thickness=2)
    image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

    # ------------

    save(
        "examples_bounding_boxes",
        "rotation.jpg",
        grid([image_before, image_after], cols=2, rows=1),
        quality=75
    )

def chapter_examples_bounding_boxes_ooi():
    import imgaug as ia
    from imgaug import augmenters as iaa
    import numpy as np

    ia.seed(1)

    GREEN = [0, 255, 0]
    ORANGE = [255, 140, 0]
    RED = [255, 0, 0]

    # Pad image with a 1px white and (BY-1)px black border
    def pad(image, by):
        if by <= 0:
            return image
        image_border1 = np.pad(
            image, ((1, 1), (1, 1), (0, 0)),
            mode="constant", constant_values=255
        )
        image_border2 = np.pad(
            image_border1, ((by-1, by-1), (by-1, by-1), (0, 0)),
            mode="constant", constant_values=0
        )

        return image_border2

    # Draw BBs on an image
    # and before doing that, extend the image plane by BORDER pixels.
    # Mark BBs inside the image plane with green color, those partially inside
    # with orange and those fully outside with red.
    def draw_bbs(image, bbs, border):
        image_border = pad(image, border)
        for bb in bbs.bounding_boxes:
            if bb.is_fully_within_image(image.shape):
                color = GREEN
            elif bb.is_partly_within_image(image.shape):
                color = ORANGE
            else:
                color = RED
            image_border = bb.shift(left=border, top=border)\
                             .draw_on_image(image_border, thickness=2, color=color)

        return image_border

    # Define example image with three small square BBs next to each other.
    # Augment these BBs by shifting them to the right.
    image = ia.quokka(size=(256, 256))
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=25, x2=75, y1=25, y2=75),
        ia.BoundingBox(x1=100, x2=150, y1=25, y2=75),
        ia.BoundingBox(x1=175, x2=225, y1=25, y2=75)
    ], shape=image.shape)

    seq = iaa.Affine(translate_px={"x": 120})
    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # Draw the BBs (a) in their original form, (b) after augmentation,
    # (c) after augmentation and removing those fully outside the image,
    # (d) after augmentation and removing those fully outside the image and
    # cutting those partially inside the image so that they are fully inside.
    image_before = draw_bbs(image, bbs, 100)
    image_after1 = draw_bbs(image_aug, bbs_aug, 100)
    image_after2 = draw_bbs(image_aug, bbs_aug.remove_out_of_image(), 100)
    image_after3 = draw_bbs(image_aug, bbs_aug.remove_out_of_image().cut_out_of_image(), 100)

    # ------------

    save(
        "examples_bounding_boxes",
        "ooi.jpg",
        grid([image_before, image_after1, np.zeros_like(image_before), image_after2, np.zeros_like(image_before), image_after3], cols=2, rows=3),
        #grid([image_before, image_after1], cols=2, rows=1),
        quality=75
    )

def chapter_examples_bounding_boxes_shift():
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Define image and two bounding boxes
    image = ia.quokka(size=(256, 256))
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=25, x2=75, y1=25, y2=75),
        ia.BoundingBox(x1=100, x2=150, y1=25, y2=75)
    ], shape=image.shape)

    # Move both BBs 25px to the right and the second BB 25px down
    bbs_shifted = bbs.shift(left=25)
    bbs_shifted.bounding_boxes[1] = bbs_shifted.bounding_boxes[1].shift(top=25)

    # Draw images before/after moving BBs
    image = bbs.draw_on_image(image, color=[0, 255, 0], thickness=2, alpha=0.75)
    image = bbs_shifted.draw_on_image(image, color=[0, 0, 255], thickness=2, alpha=0.75)

    # ------------

    save(
        "examples_bounding_boxes",
        "shift.jpg",
        grid([image], cols=1, rows=1),
        quality=75
    )

def chapter_examples_bounding_boxes_projection():
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Define image with two bounding boxes
    image = ia.quokka(size=(256, 256))
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=25, x2=75, y1=25, y2=75),
        ia.BoundingBox(x1=100, x2=150, y1=25, y2=75)
    ], shape=image.shape)

    # Rescale image and bounding boxes
    image_rescaled = ia.imresize_single_image(image, (512, 512))
    bbs_rescaled = bbs.on(image_rescaled)

    # Draw image before/after rescaling and with rescaled bounding boxes
    image_bbs = bbs.draw_on_image(image, thickness=2)
    image_rescaled_bbs = bbs_rescaled.draw_on_image(image_rescaled, thickness=2)

    # ------------

    save(
        "examples_bounding_boxes",
        "projection.jpg",
        grid([image_bbs, image_rescaled_bbs], cols=2, rows=1),
        quality=75
    )

def chapter_examples_bounding_boxes_iou():
    import imgaug as ia
    from imgaug import augmenters as iaa
    import numpy as np

    ia.seed(1)

    # Define image with two bounding boxes.
    image = ia.quokka(size=(256, 256))
    bb1 = ia.BoundingBox(x1=50, x2=100, y1=25, y2=75)
    bb2 = ia.BoundingBox(x1=75, x2=125, y1=50, y2=100)

    # Compute intersection, union and IoU value
    # Intersection and union are both bounding boxes. They are here
    # decreased/increased in size purely for better visualization.
    bb_inters = bb1.intersection(bb2).extend(all_sides=-1)
    bb_union = bb1.union(bb2).extend(all_sides=2)
    iou = bb1.iou(bb2)

    # Draw bounding boxes, intersection, union and IoU value on image.
    image_bbs = np.copy(image)
    image_bbs = bb1.draw_on_image(image_bbs, thickness=2, color=[0, 255, 0])
    image_bbs = bb2.draw_on_image(image_bbs, thickness=2, color=[0, 255, 0])
    image_bbs = bb_inters.draw_on_image(image_bbs, thickness=2, color=[255, 0, 0])
    image_bbs = bb_union.draw_on_image(image_bbs, thickness=2, color=[0, 0, 255])
    image_bbs = ia.draw_text(
        image_bbs, text="IoU=%.2f" % (iou,),
        x=bb_union.x2+10, y=bb_union.y1+bb_union.height//2,
        color=[255, 255, 255], size=13
    )

    # ------------

    save(
        "examples_bounding_boxes",
        "iou.jpg",
        grid([image_bbs], cols=1, rows=1),
        quality=75
    )

###############################
# Overview of augmenters
###############################

def run_and_save_augseq(filename, augseq, images, cols, rows, quality=75, seed=1):
    ia.seed(seed)
    # augseq may be a single seq (applied to all images) or a list (one seq per
    # image).
    # use type() here instead of isinstance, because otherwise Sequential is
    # also interpreted as a list
    if type(augseq) == list:
        # one augmenter per image specified
        assert len(augseq) == len(images)
        images_aug = [augseq[i].augment_image(images[i]) for i in range(len(images))]
    else:
        # calling N times augment_image() is here critical for random order in
        # Sequential
        images_aug = [augseq.augment_image(images[i]) for i in range(len(images))]
    save(
        "overview_of_augmenters",
        filename,
        grid(images_aug, cols=cols, rows=rows),
        quality=quality
    )

def chapter_augmenters():
    chapter_augmenters_sequential()
    chapter_augmenters_someof()
    chapter_augmenters_oneof()
    chapter_augmenters_sometimes()
    chapter_augmenters_withcolorspace()
    chapter_augmenters_withchannels()
    chapter_augmenters_noop()
    chapter_augmenters_lambda()
    chapter_augmenters_assertlambda()
    chapter_augmenters_assertshape()
    chapter_augmenters_scale()
    chapter_augmenters_cropandpad()
    chapter_augmenters_pad()
    chapter_augmenters_crop()
    chapter_augmenters_fliplr()
    chapter_augmenters_flipud()
    chapter_augmenters_superpixels()
    chapter_augmenters_changecolorspace()
    chapter_augmenters_grayscale()
    chapter_augmenters_gaussianblur()
    chapter_augmenters_averageblur()
    chapter_augmenters_medianblur()
    chapter_augmenters_convolve()
    chapter_augmenters_sharpen()
    chapter_augmenters_emboss()
    chapter_augmenters_edgedetect()
    chapter_augmenters_directededgedetect()
    chapter_augmenters_add()
    chapter_augmenters_addelementwise()
    chapter_augmenters_additivegaussiannoise()
    chapter_augmenters_multiply()
    chapter_augmenters_multiplyelementwise()
    chapter_augmenters_dropout()
    chapter_augmenters_coarsedropout()
    chapter_augmenters_invert()
    chapter_augmenters_contrastnormalization()
    chapter_augmenters_affine()
    chapter_augmenters_piecewiseaffine()
    chapter_augmenters_elastictransformation()

def chapter_augmenters_sequential():
    aug = iaa.Sequential([
        iaa.Affine(translate_px={"x":-40}),
        iaa.AdditiveGaussianNoise(scale=0.2*255)
    ])
    run_and_save_augseq(
        "sequential.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Sequential([
        iaa.Affine(translate_px={"x":-40}),
        iaa.AdditiveGaussianNoise(scale=0.2*255)
    ], random_order=True)
    run_and_save_augseq(
        "sequential_random_order.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

def chapter_augmenters_someof():
    aug = iaa.SomeOf(2, [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])
    run_and_save_augseq(
        "someof.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.SomeOf((0, None), [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])
    run_and_save_augseq(
        "someof_0_to_none.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.SomeOf(2, [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ], random_order=True)
    run_and_save_augseq(
        "someof_random_order.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

def chapter_augmenters_oneof():
    aug = iaa.OneOf([
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])
    run_and_save_augseq(
        "oneof.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

def chapter_augmenters_sometimes():
    aug = iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=2.0))
    run_and_save_augseq(
        "sometimes.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2,
        seed=2
    )

    aug = iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=2.0),
        iaa.Sequential([iaa.Affine(rotate=45), iaa.Sharpen(alpha=1.0)])
    )
    run_and_save_augseq(
        "sometimes_if_else.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

def chapter_augmenters_withcolorspace():
    aug = iaa.WithColorspace(
        to_colorspace="HSV",
        from_colorspace="RGB",
        children=iaa.WithChannels(0, iaa.Add((10, 50)))
    )
    run_and_save_augseq(
        "withcolorspace.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

def chapter_augmenters_withchannels():
    aug = iaa.WithChannels(0, iaa.Add((10, 100)))
    run_and_save_augseq(
        "withchannels.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.WithChannels(0, iaa.Affine(rotate=(0, 45)))
    run_and_save_augseq(
        "withchannels_affine.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

def chapter_augmenters_noop():
    aug = iaa.Noop()
    run_and_save_augseq(
        "noop.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

def chapter_augmenters_lambda():
    def img_func(images, random_state, parents, hooks):
        for img in images:
            img[::4] = 0
        return images

    def keypoint_func(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    aug = iaa.Lambda(img_func, keypoint_func)
    run_and_save_augseq(
        "lambda.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

def chapter_augmenters_assertlambda():
    pass

def chapter_augmenters_assertshape():
    pass

def chapter_augmenters_scale():
    aug = iaa.Scale({"height": 32, "width": 64})
    run_and_save_augseq(
        "scale_32x64.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Scale({"height": 32, "width": "keep-aspect-ratio"})
    run_and_save_augseq(
        "scale_32xkar.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Scale((0.5, 1.0))
    run_and_save_augseq(
        "scale_50_to_100_percent.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Scale({"height": (0.5, 0.75), "width": [16, 32, 64]})
    run_and_save_augseq(
        "scale_h_uniform_w_choice.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

def chapter_augmenters_cropandpad():
    aug = iaa.CropAndPad(percent=(-0.25, 0.25))
    run_and_save_augseq(
        "cropandpad_percent.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.CropAndPad(
        percent=(0, 0.2),
        pad_mode=["constant", "edge"],
        pad_cval=(0, 128)
    )
    run_and_save_augseq(
        "cropandpad_mode_cval.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.CropAndPad(
        px=((0, 30), (0, 10), (0, 30), (0, 10)),
        pad_mode=ia.ALL,
        pad_cval=(0, 128)
    )
    run_and_save_augseq(
        "cropandpad_pad_complex.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(32)], cols=8, rows=4
    )

    aug = iaa.CropAndPad(
        px=(-10, 10),
        sample_independently=False
    )
    run_and_save_augseq(
        "cropandpad_correlated.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

def chapter_augmenters_pad():
    pass

def chapter_augmenters_crop():
    pass

def chapter_augmenters_fliplr():
    aug = iaa.Fliplr(0.5)
    run_and_save_augseq(
        "fliplr.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

def chapter_augmenters_flipud():
    aug = iaa.Flipud(0.5)
    run_and_save_augseq(
        "flipud.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

def chapter_augmenters_superpixels():
    aug = iaa.Superpixels(p_replace=0.5, n_segments=64)
    run_and_save_augseq(
        "superpixels_50_64.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Superpixels(p_replace=(0.1, 1.0), n_segments=(16, 128))
    run_and_save_augseq(
        "superpixels.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    #ps = [1/8*i for i in range(8)]
    ps = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "superpixels_vary_p.jpg",
        [iaa.Superpixels(p_replace=p, n_segments=64) for p in ps],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1,
        quality=75
    )

    ns = [16*i for i in range(1, 9)]
    run_and_save_augseq(
        "superpixels_vary_n.jpg",
        [iaa.Superpixels(p_replace=1.0, n_segments=n) for n in ns],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1,
        quality=75
    )

def chapter_augmenters_changecolorspace():
    aug = iaa.Sequential([
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((50, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
    ])
    run_and_save_augseq(
        "changecolorspace.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

def chapter_augmenters_grayscale():
    aug = iaa.Grayscale(alpha=(0.0, 1.0))
    run_and_save_augseq(
        "grayscale.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    #alphas = [1/8*i for i in range(8)]
    alphas = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "grayscale_vary_alpha.jpg",
        [iaa.Grayscale(alpha=alpha) for alpha in alphas],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1,
        quality=75
    )

def chapter_augmenters_gaussianblur():
    aug = iaa.GaussianBlur(sigma=(0.0, 3.0))
    run_and_save_augseq(
        "gaussianblur.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4,
        quality=75
    )

def chapter_augmenters_averageblur():
    aug = iaa.AverageBlur(k=(2, 11))
    run_and_save_augseq(
        "averageblur.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4,
        quality=75
    )

    aug = iaa.AverageBlur(k=((5, 11), (1, 3)))
    run_and_save_augseq(
        "averageblur_mixed.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4,
        quality=75
    )

def chapter_augmenters_medianblur():
    aug = iaa.MedianBlur(k=(3, 11))
    run_and_save_augseq(
        "medianblur.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4,
        quality=75
    )

    # median doesnt support this
    #aug = iaa.MedianBlur(k=((5, 11), (1, 3)))
    #run_and_save_augseq(
    #    "medianblur_mixed.jpg", aug,
    #    [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2,
    #    quality=75
    #)

def chapter_augmenters_convolve():
    matrix = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]])
    aug = iaa.Convolve(matrix=matrix)
    run_and_save_augseq(
        "convolve.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=50
    )

    def gen_matrix(image, nb_channels, random_state):
        matrix_A = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
        matrix_B = np.array([[0, 0, 0],
                             [0, -4, 1],
                             [0, 2, 1]])
        if random_state.rand() < 0.5:
            return [matrix_A] * nb_channels
        else:
            return [matrix_B] * nb_channels
    aug = iaa.Convolve(matrix=gen_matrix)
    run_and_save_augseq(
        "convolve_callable.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

def chapter_augmenters_sharpen():
    aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
    run_and_save_augseq(
        "sharpen.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    #alphas = [1/8*i for i in range(8)]
    alphas = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "sharpen_vary_alpha.jpg",
        [iaa.Sharpen(alpha=alpha, lightness=1.0) for alpha in alphas],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1,
        quality=90
    )

    #lightnesses = [1/8*i for i in range(8)]
    lightnesses = np.linspace(0.75, 1.5, num=8)
    run_and_save_augseq(
        "sharpen_vary_lightness.jpg",
        [iaa.Sharpen(alpha=1.0, lightness=lightness) for lightness in lightnesses],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1,
        quality=90
    )

def chapter_augmenters_emboss():
    aug = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
    run_and_save_augseq(
        "emboss.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    #alphas = [1/8*i for i in range(8)]
    alphas = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "emboss_vary_alpha.jpg",
        [iaa.Emboss(alpha=alpha, strength=1.0) for alpha in alphas],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )

    #strengths = [0.5+(0.5/8)*i for i in range(8)]
    strengths = np.linspace(0.5, 1.5, num=8)
    run_and_save_augseq(
        "emboss_vary_strength.jpg",
        [iaa.Emboss(alpha=1.0, strength=strength) for strength in strengths],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )

def chapter_augmenters_edgedetect():
    aug = iaa.EdgeDetect(alpha=(0.0, 1.0))
    run_and_save_augseq(
        "edgedetect.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    #alphas = [1/8*i for i in range(8)]
    alphas = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "edgedetect_vary_alpha.jpg",
        [iaa.EdgeDetect(alpha=alpha) for alpha in alphas],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )

def chapter_augmenters_directededgedetect():
    aug = iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0))
    run_and_save_augseq(
        "directededgedetect.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    #alphas = [1/8*i for i in range(8)]
    alphas = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "directededgedetect_vary_alpha.jpg",
        [iaa.DirectedEdgeDetect(alpha=alpha, direction=0) for alpha in alphas],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )

    #strength = [0.5+(0.5/8)*i for i in range(8)]
    directions = np.linspace(0.0, 1.0, num=8)
    run_and_save_augseq(
        "directededgedetect_vary_direction.jpg",
        [iaa.DirectedEdgeDetect(alpha=1.0, direction=direction) for direction in directions],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )

def chapter_augmenters_add():
    aug = iaa.Add((-40, 40))
    run_and_save_augseq(
        "add.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=75
    )

    aug = iaa.Add((-40, 40), per_channel=0.5)
    run_and_save_augseq(
        "add_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=75
    )

def chapter_augmenters_addelementwise():
    aug = iaa.AddElementwise((-40, 40))
    run_and_save_augseq(
        "addelementwise.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=90
    )

    aug = iaa.AddElementwise((-40, 40), per_channel=0.5)
    run_and_save_augseq(
        "addelementwise_per_channel.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=90
    )

def chapter_augmenters_additivegaussiannoise():
    aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
    run_and_save_augseq(
        "additivegaussiannoise.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=90
    )

    aug = iaa.AdditiveGaussianNoise(scale=0.2*255)
    run_and_save_augseq(
        "additivegaussiannoise_large.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=90
    )

    aug = iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=True)
    run_and_save_augseq(
        "additivegaussiannoise_per_channel.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=90
    )

def chapter_augmenters_multiply():
    aug = iaa.Multiply((0.5, 1.5))
    run_and_save_augseq(
        "multiply.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Multiply((0.5, 1.5), per_channel=0.5)
    run_and_save_augseq(
        "multiply_per_channel.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

def chapter_augmenters_multiplyelementwise():
    aug = iaa.MultiplyElementwise((0.5, 1.5))
    run_and_save_augseq(
        "multiplyelementwise.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=90
    )

    aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=True)
    run_and_save_augseq(
        "multiplyelementwise_per_channel.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=90
    )

def chapter_augmenters_dropout():
    aug = iaa.Dropout(p=(0, 0.2))
    run_and_save_augseq(
        "dropout.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=75
    )

    aug = iaa.Dropout(p=(0, 0.2), per_channel=0.5)
    run_and_save_augseq(
        "dropout_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=75
    )

def chapter_augmenters_coarsedropout():
    aug = iaa.CoarseDropout(0.02, size_percent=0.5)
    run_and_save_augseq(
        "coarsedropout.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=75
    )

    aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))
    run_and_save_augseq(
        "coarsedropout_both_uniform.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=75,
        seed=2
    )

    aug = iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5)
    run_and_save_augseq(
        "coarsedropout_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=75,
        seed=2
    )

def chapter_augmenters_invert():
    aug = iaa.Invert(0.5)
    run_and_save_augseq(
        "invert.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Invert(0.25, per_channel=0.5)
    run_and_save_augseq(
        "invert_per_channel.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

def chapter_augmenters_contrastnormalization():
    aug = iaa.ContrastNormalization((0.5, 1.5))
    run_and_save_augseq(
        "contrastnormalization.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)
    run_and_save_augseq(
        "contrastnormalization_per_channel.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

def chapter_augmenters_affine():
    aug = iaa.Affine(scale=(0.5, 1.5))
    run_and_save_augseq(
        "affine_scale.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})
    run_and_save_augseq(
        "affine_scale_independently.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
    run_and_save_augseq(
        "affine_translate_percent.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)})
    run_and_save_augseq(
        "affine_translate_px.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(rotate=(-45, 45))
    run_and_save_augseq(
        "affine_rotate.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(shear=(-16, 16))
    run_and_save_augseq(
        "affine_shear.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(translate_percent={"x": -0.20}, mode=ia.ALL, cval=(0, 255))
    run_and_save_augseq(
        "affine_fill.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

def chapter_augmenters_piecewiseaffine():
    aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))
    run_and_save_augseq(
        "piecewiseaffine.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=75
    )

    aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))
    run_and_save_augseq(
        "piecewiseaffine_checkerboard.jpg", aug,
        [checkerboard(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=75
    )

    scales = np.linspace(0.0, 0.3, num=8)
    run_and_save_augseq(
        "piecewiseaffine_vary_scales.jpg",
        [iaa.PiecewiseAffine(scale=scale) for scale in scales],
        [checkerboard(size=(128, 128)) for _ in range(8)], cols=8, rows=1,
        quality=75
    )

    gridvals = [2, 4, 6, 8, 10, 12, 14, 16]
    run_and_save_augseq(
        "piecewiseaffine_vary_grid.jpg",
        [iaa.PiecewiseAffine(scale=0.05, nb_rows=g, nb_cols=g) for g in gridvals],
        [checkerboard(size=(128, 128)) for _ in range(8)], cols=8, rows=1,
        quality=75
    )

def chapter_augmenters_elastictransformation():
    aug = iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
    run_and_save_augseq(
        "elastictransformations.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=75
    )

    alphas = np.linspace(0.0, 5.0, num=8)
    run_and_save_augseq(
        "elastictransformations_vary_alpha.jpg",
        [iaa.ElasticTransformation(alpha=alpha, sigma=0.25) for alpha in alphas],
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=8, rows=1,
        quality=75
    )

    sigmas = np.linspace(0.01, 1.0, num=8)
    run_and_save_augseq(
        "elastictransformations_vary_sigmas.jpg",
        [iaa.ElasticTransformation(alpha=2.5, sigma=sigma) for sigma in sigmas],
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=8, rows=1,
        quality=75
    )

###############################
# Parameters
###############################

def draw_distributions_grid(params, rows=None, cols=None, graph_sizes=PARAMETERS_DEFAULT_SIZE, sample_sizes=None, titles=False):
    return iap.draw_distributions_grid(
        params, rows=rows, cols=cols, graph_sizes=graph_sizes,
        sample_sizes=sample_sizes, titles=titles
    )

def chapter_parameters():
    chapter_parameters_introduction()
    chapter_parameters_continuous()
    chapter_parameters_discrete()
    chapter_parameters_arithmetic()
    chapter_parameters_special()

def chapter_parameters_introduction():
    ia.seed(1)
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    seq = iaa.Sequential([
        iaa.GaussianBlur(
            sigma=iap.Uniform(0.0, 1.0)
        ),
        iaa.ContrastNormalization(
            iap.Choice(
                [1.0, 1.5, 3.0],
                p=[0.5, 0.3, 0.2]
            )
        ),
        iaa.Affine(
            rotate=iap.Normal(0.0, 30),
            translate_px=iap.RandomSign(iap.Poisson(3))
        ),
        iaa.AddElementwise(
            iap.Discretize(
                (iap.Beta(0.5, 0.5) * 2 - 1.0) * 64
            )
        ),
        iaa.Multiply(
            iap.Positive(iap.Normal(0.0, 0.1)) + 1.0
        )
    ])

    images = np.array([ia.quokka_square(size=(128, 128)) for i in range(16)])
    images_aug = [seq.augment_image(images[i]) for i in range(len(images))]
    save(
        "parameters",
        "introduction.jpg",
        grid(images_aug, cols=4, rows=4),
        quality=25
    )

def chapter_parameters_continuous():
    ia.seed(1)

    # -----------------------
    # Normal
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Normal(0, 1),
        iap.Normal(5, 3),
        iap.Normal(iap.Choice([-3, 3]), 1),
        iap.Normal(iap.Uniform(-3, 3), 1)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_normal.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Laplace
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Laplace(0, 1),
        iap.Laplace(5, 3),
        iap.Laplace(iap.Choice([-3, 3]), 1),
        iap.Laplace(iap.Uniform(-3, 3), 1)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_laplace.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # ChiSquare
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.ChiSquare(1),
        iap.ChiSquare(3),
        iap.ChiSquare(iap.Choice([1, 5])),
        iap.RandomSign(iap.ChiSquare(3))
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_chisquare.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Weibull
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Weibull(0.5),
        iap.Weibull(1),
        iap.Weibull(1.5),
        iap.Weibull((0.5, 1.5))
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_weibull.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Uniform
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Uniform(0, 1),
        iap.Uniform(iap.Normal(-3, 1), iap.Normal(3, 1)),
        iap.Uniform([-1, 0], 1),
        iap.Uniform((-1, 0), 1)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_uniform.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Beta
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Beta(0.5, 0.5),
        iap.Beta(2.0, 2.0),
        iap.Beta(1.0, 0.5),
        iap.Beta(0.5, 1.0)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_beta.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

def chapter_parameters_discrete():
    ia.seed(1)

    # -----------------------
    # Binomial
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Binomial(0.5),
        iap.Binomial(0.9)
    ]
    gridarr = draw_distributions_grid(params, rows=1)
    save(
        "parameters",
        "continuous_binomial.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # DiscreteUniform
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.DiscreteUniform(0, 10),
        iap.DiscreteUniform(-10, 10),
        iap.DiscreteUniform([-10, -9, -8, -7], 10),
        iap.DiscreteUniform((-10, -7), 10)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_discreteuniform.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Poisson
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Poisson(1),
        iap.Poisson(2.5),
        iap.Poisson((1, 2.5)),
        iap.RandomSign(iap.Poisson(2.5))
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_poisson.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

def chapter_parameters_arithmetic():
    ia.seed(1)

    # -----------------------
    # Add
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Uniform(0, 1) + 1, # identical to: Add(Uniform(0, 1), 1)
        iap.Add(iap.Uniform(0, 1), iap.Choice([0, 1], p=[0.7, 0.3])),
        iap.Normal(0, 1) + iap.Uniform(-5.5, -5) + iap.Uniform(5, 5.5),
        iap.Normal(0, 1) + iap.Uniform(-7, 5) + iap.Poisson(3),
        iap.Add(iap.Normal(-3, 1), iap.Normal(3, 1)),
        iap.Add(iap.Normal(-3, 1), iap.Normal(3, 1), elementwise=True)
    ]
    gridarr = draw_distributions_grid(
        params,
        rows=2,
        sample_sizes=[ # (iterations, samples per iteration)
            (1000, 1000), (1000, 1000), (1000, 1000),
            (1000, 1000), (1, 100000), (1, 100000)
        ]
    )
    save(
        "parameters",
        "arithmetic_add.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Multiply
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Uniform(0, 1) * 2, # identical to: Multiply(Uniform(0, 1), 2)
        iap.Multiply(iap.Uniform(0, 1), iap.Choice([0, 1], p=[0.7, 0.3])),
        (iap.Normal(0, 1) * iap.Uniform(-5.5, -5)) * iap.Uniform(5, 5.5),
        (iap.Normal(0, 1) * iap.Uniform(-7, 5)) * iap.Poisson(3),
        iap.Multiply(iap.Normal(-3, 1), iap.Normal(3, 1)),
        iap.Multiply(iap.Normal(-3, 1), iap.Normal(3, 1), elementwise=True)
    ]
    gridarr = draw_distributions_grid(
        params,
        rows=2,
        sample_sizes=[ # (iterations, samples per iteration)
            (1000, 1000), (1000, 1000), (1000, 1000),
            (1000, 1000), (1, 100000), (1, 100000)
        ]
    )
    save(
        "parameters",
        "arithmetic_multiply.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Divide
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Uniform(0, 1) / 2, # identical to: Divide(Uniform(0, 1), 2)
        iap.Divide(iap.Uniform(0, 1), iap.Choice([0, 2], p=[0.7, 0.3])),
        (iap.Normal(0, 1) / iap.Uniform(-5.5, -5)) / iap.Uniform(5, 5.5),
        (iap.Normal(0, 1) * iap.Uniform(-7, 5)) / iap.Poisson(3),
        iap.Divide(iap.Normal(-3, 1), iap.Normal(3, 1)),
        iap.Divide(iap.Normal(-3, 1), iap.Normal(3, 1), elementwise=True)
    ]
    gridarr = draw_distributions_grid(
        params,
        rows=2,
        sample_sizes=[ # (iterations, samples per iteration)
            (1000, 1000), (1000, 1000), (1000, 1000),
            (1000, 1000), (1, 100000), (1, 100000)
        ]
    )
    save(
        "parameters",
        "arithmetic_divide.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Power
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Uniform(0, 1) ** 2, # identical to: Power(Uniform(0, 1), 2)
        iap.Clip(iap.Uniform(-1, 1) ** iap.Normal(0, 1), -4, 4)
    ]
    gridarr = draw_distributions_grid(
        params,
        rows=1
    )
    save(
        "parameters",
        "arithmetic_power.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

def chapter_parameters_special():
    ia.seed(1)

    # -----------------------
    # Choice
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Choice([0, 1, 2]),
        iap.Choice([0, 1, 2], p=[0.15, 0.5, 0.35]),
        iap.Choice([iap.Normal(-3, 1), iap.Normal(3, 1)]),
        iap.Choice([iap.Normal(-3, 1), iap.Poisson(3)])
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "special_choice.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Clip
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Clip(iap.Normal(0, 1), -2, 2),
        iap.Clip(iap.Normal(0, 1), -2, None)
    ]
    gridarr = draw_distributions_grid(params, rows=1)
    save(
        "parameters",
        "special_clip.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Discretize
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Discretize(iap.Normal(0, 1)),
        iap.Discretize(iap.ChiSquare(3))
    ]
    gridarr = draw_distributions_grid(params, rows=1)
    save(
        "parameters",
        "special_discretize.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Absolute
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Absolute(iap.Normal(0, 1)),
        iap.Absolute(iap.Laplace(0, 1))
    ]
    gridarr = draw_distributions_grid(params, rows=1)
    save(
        "parameters",
        "special_absolute.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # RandomSign
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.ChiSquare(3),
        iap.RandomSign(iap.ChiSquare(3)),
        iap.RandomSign(iap.ChiSquare(3), p_positive=0.75),
        iap.RandomSign(iap.ChiSquare(3), p_positive=0.9)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "special_randomsign.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # ForceSign
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.ForceSign(iap.Normal(0, 1), positive=True),
        iap.ChiSquare(3) - 3.0,
        iap.ForceSign(iap.ChiSquare(3) - 3.0, positive=True, mode="invert"),
        iap.ForceSign(iap.ChiSquare(3) - 3.0, positive=True, mode="reroll")
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "special_forcesign.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

###############################
# Alpha
###############################

def chapter_alpha():
    """Generate all example images for the chapter `Alpha`
    in the documentation."""
    chapter_alpha_masks_introduction()
    chapter_alpha_constant()
    chapter_alpha_masks_simplex()
    chapter_alpha_masks_frequency()
    chapter_alpha_masks_iterative()
    chapter_alpha_masks_sigmoid()

def chapter_alpha_masks_introduction():
    # -----------------------------------------
    # example introduction
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(2)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seqs = [
        iaa.Alpha(
            (0.0, 1.0),
            first=iaa.MedianBlur(11),
            per_channel=True
        ),
        iaa.SimplexNoiseAlpha(
            first=iaa.EdgeDetect(1.0),
            per_channel=False
        ),
        iaa.SimplexNoiseAlpha(
            first=iaa.EdgeDetect(1.0),
            second=iaa.ContrastNormalization((0.5, 2.0)),
            per_channel=0.5
        ),
        iaa.FrequencyNoiseAlpha(
            first=iaa.Affine(
                rotate=(-10, 10),
                translate_px={"x": (-4, 4), "y": (-4, 4)}
            ),
            second=iaa.AddToHueAndSaturation((-40, 40)),
            per_channel=0.5
        ),
        iaa.SimplexNoiseAlpha(
            first=iaa.SimplexNoiseAlpha(
                first=iaa.EdgeDetect(1.0),
                second=iaa.ContrastNormalization((0.5, 2.0)),
                per_channel=True
            ),
            second=iaa.FrequencyNoiseAlpha(
                exponent=(-2.5, -1.0),
                first=iaa.Affine(
                    rotate=(-10, 10),
                    translate_px={"x": (-4, 4), "y": (-4, 4)}
                ),
                second=iaa.AddToHueAndSaturation((-40, 40)),
                per_channel=True
            ),
            per_channel=True,
            aggregation_method="max",
            sigmoid=False
        )
    ]

    cells = []
    for seq in seqs:
        images_aug = seq.augment_images(images)
        cells.extend(images_aug)

    # ------------

    save(
        "alpha",
        "introduction.jpg",
        grid(cells, cols=8, rows=5)
    )

def chapter_alpha_constant():
    # -----------------------------------------
    # example 1 (sharpen + dropout)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.Alpha(
        factor=(0.2, 0.8),
        first=iaa.Sharpen(1.0, lightness=2),
        second=iaa.CoarseDropout(p=0.1, size_px=8)
    )

    images_aug = seq.augment_images(images)

    # ------------

    save(
        "alpha",
        "alpha_constant_example_basic.jpg",
        grid(images_aug, cols=4, rows=2)
    )


    # -----------------------------------------
    # example 2 (per channel)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.Alpha(
        factor=(0.2, 0.8),
        first=iaa.Sharpen(1.0, lightness=2),
        second=iaa.CoarseDropout(p=0.1, size_px=8),
        per_channel=True
    )

    images_aug = seq.augment_images(images)

    # ------------

    save(
        "alpha",
        "alpha_constant_example_per_channel.jpg",
        grid(images_aug, cols=4, rows=2)
    )


    # -----------------------------------------
    # example 3 (affine + per channel)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.Alpha(
        factor=(0.2, 0.8),
        first=iaa.Affine(rotate=(-20, 20)),
        per_channel=True
    )

    images_aug = seq.augment_images(images)

    # ------------

    save(
        "alpha",
        "alpha_constant_example_affine.jpg",
        grid(images_aug, cols=4, rows=2)
    )

def chapter_alpha_masks_simplex():
    # -----------------------------------------
    # example 1 (basic)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.SimplexNoiseAlpha(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
    )

    images_aug = seq.augment_images(images)

    # ------------

    save(
        "alpha",
        "alpha_simplex_example_basic.jpg",
        grid(images_aug, cols=4, rows=2)
    )


    # -----------------------------------------
    # example 1 (per_channel)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.SimplexNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        per_channel=True
    )

    images_aug = seq.augment_images(images)

    # ------------

    save(
        "alpha",
        "alpha_simplex_example_per_channel.jpg",
        grid(images_aug, cols=4, rows=2)
    )


    # -----------------------------------------
    # noise masks
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    seed = 1
    ia.seed(seed)

    seq = iaa.SimplexNoiseAlpha(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
    )

    masks = [seq.factor.draw_samples((64, 64), random_state=ia.new_random_state(seed+1+i)) for i in range(16)]
    masks = np.hstack(masks)
    masks = np.tile(masks[:, :, np.newaxis], (1, 1, 1, 3))
    masks = (masks * 255).astype(np.uint8)

    # ------------

    save(
        "alpha",
        "alpha_simplex_noise_masks.jpg",
        grid(masks, cols=16, rows=1)
    )


    # -----------------------------------------
    # noise masks, upscale=nearest
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    seed = 1
    ia.seed(seed)

    seq = iaa.SimplexNoiseAlpha(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
        upscale_method="nearest"
    )

    masks = [seq.factor.draw_samples((64, 64), random_state=ia.new_random_state(seed+1+i)) for i in range(16)]
    masks = np.hstack(masks)
    masks = np.tile(masks[:, :, np.newaxis], (1, 1, 1, 3))
    masks = (masks * 255).astype(np.uint8)

    # ------------

    save(
        "alpha",
        "alpha_simplex_noise_masks_nearest.jpg",
        grid(masks, cols=16, rows=1)
    )


    # -----------------------------------------
    # noise masks linear
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    seed = 1
    ia.seed(seed)

    seq = iaa.SimplexNoiseAlpha(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
        upscale_method="linear"
    )

    masks = [seq.factor.draw_samples((64, 64), random_state=ia.new_random_state(seed+1+i)) for i in range(16)]
    masks = np.hstack(masks)
    masks = np.tile(masks[:, :, np.newaxis], (1, 1, 1, 3))
    masks = (masks * 255).astype(np.uint8)

    # ------------

    save(
        "alpha",
        "alpha_simplex_noise_masks_linear.jpg",
        grid(masks, cols=16, rows=1)
    )

def chapter_alpha_masks_frequency():
    # -----------------------------------------
    # example 1 (basic)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 64, 64, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.FrequencyNoiseAlpha(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
    )

    images_aug = seq.augment_images(images)

    # ------------

    save(
        "alpha",
        "alpha_frequency_example_basic.jpg",
        grid(images_aug, cols=4, rows=2)
    )


    # -----------------------------------------
    # example 1 (per_channel)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.FrequencyNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        per_channel=True
    )

    images_aug = seq.augment_images(images)

    # ------------

    save(
        "alpha",
        "alpha_frequency_example_per_channel.jpg",
        grid(images_aug, cols=4, rows=2)
    )


    # -----------------------------------------
    # noise masks
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    seq = iaa.FrequencyNoiseAlpha(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
    )

    masks = [seq.factor.draw_samples((64, 64), random_state=ia.new_random_state(seed+1+i)) for i in range(16)]
    masks = [np.tile(mask[:, :, np.newaxis], (1, 1, 3)) for mask in masks]
    masks = [(mask * 255).astype(np.uint8) for mask in masks]

    # ------------

    save(
        "alpha",
        "alpha_frequency_noise_masks.jpg",
        grid(masks, cols=8, rows=2)
    )


    # -----------------------------------------
    # noise masks, varying exponent
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    masks = []
    nb_rows = 4
    exponents = np.linspace(-4.0, 4.0, 16)

    for i, exponent in enumerate(exponents):
        seq = iaa.FrequencyNoiseAlpha(
            exponent=exponent,
            first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
            size_px_max=32,
            upscale_method="linear",
            iterations=1,
            sigmoid=False
        )

        group = []
        for row in range(nb_rows):
            mask = seq.factor.draw_samples((64, 64), random_state=ia.new_random_state(seed+1+i*10+row))
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
            mask = (mask * 255).astype(np.uint8)
            if row == nb_rows - 1:
                mask = np.pad(mask, ((0, 20), (0, 0), (0, 0)), mode="constant", constant_values=255)
                mask = ia.draw_text(mask, y=64+2, x=6, text="%.2f" % (exponent,), size=10, color=[0, 0, 0])
            group.append(mask)
        masks.append(np.vstack(group))

    # ------------

    save(
        "alpha",
        "alpha_frequency_noise_masks_exponents.jpg",
        grid(masks, cols=16, rows=1)
    )


    # -----------------------------------------
    # noise masks, upscale=nearest
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    seq = iaa.FrequencyNoiseAlpha(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
        upscale_method="nearest"
    )

    masks = [seq.factor.draw_samples((64, 64), random_state=ia.new_random_state(seed+1+i)) for i in range(16)]
    masks = [np.tile(mask[:, :, np.newaxis], (1, 1, 3)) for mask in masks]
    masks = [(mask * 255).astype(np.uint8) for mask in masks]

    # ------------

    save(
        "alpha",
        "alpha_frequency_noise_masks_nearest.jpg",
        grid(masks, cols=8, rows=2)
    )


    # -----------------------------------------
    # noise masks linear
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    seq = iaa.FrequencyNoiseAlpha(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
        upscale_method="linear"
    )

    masks = [seq.factor.draw_samples((64, 64), random_state=ia.new_random_state(seed+1+i)) for i in range(16)]
    masks = [np.tile(mask[:, :, np.newaxis], (1, 1, 3)) for mask in masks]
    masks = [(mask * 255).astype(np.uint8) for mask in masks]

    # ------------

    save(
        "alpha",
        "alpha_frequency_noise_masks_linear.jpg",
        grid(masks, cols=8, rows=2)
    )

def chapter_alpha_masks_iterative():
    # -----------------------------------------
    # IterativeNoiseAggregator varying number of iterations
    # -----------------------------------------
    import imgaug as ia
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    masks = []
    iterations_all = [1, 2, 3, 4]

    for iterations in iterations_all:
        noise = iap.IterativeNoiseAggregator(
            other_param=iap.FrequencyNoise(
                exponent=(-4.0, 4.0),
                upscale_method=["linear", "nearest"]
            ),
            iterations=iterations,
            aggregation_method="max"
        )

        row = [noise.draw_samples((64, 64), random_state=ia.new_random_state(seed+1+i)) for i in range(16)]
        row = np.hstack(row)
        row = np.tile(row[:, :, np.newaxis], (1, 1, 3))
        row = (row * 255).astype(np.uint8)
        row = np.pad(row, ((0, 0), (50, 0), (0, 0)), mode="constant", constant_values=255)
        row = ia.draw_text(row, y=24, x=2, text="%d iter." % (iterations,), size=14, color=[0, 0, 0])
        masks.append(row)

    # ------------

    save(
        "alpha",
        "iterative_vary_iterations.jpg",
        grid(masks, cols=1, rows=len(iterations_all))
    )


    # -----------------------------------------
    # IterativeNoiseAggregator varying methods
    # -----------------------------------------
    import imgaug as ia
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    masks = []
    iterations_all = [1, 2, 3, 4, 5, 6]
    methods = ["min", "avg", "max"]
    cell_idx = 0
    rows = []

    for method_idx, method in enumerate(methods):
        row = []
        for iterations in iterations_all:
            noise = iap.IterativeNoiseAggregator(
                other_param=iap.FrequencyNoise(
                    exponent=-2.0,
                    size_px_max=32,
                    upscale_method=["linear", "nearest"]
                ),
                iterations=iterations,
                aggregation_method=method
            )

            cell = noise.draw_samples((64, 64), random_state=ia.new_random_state(seed+1+method_idx))
            cell = np.tile(cell[:, :, np.newaxis], (1, 1, 3))
            cell = (cell * 255).astype(np.uint8)

            if iterations == 1:
                cell = np.pad(cell, ((0, 0), (40, 0), (0, 0)), mode="constant", constant_values=255)
                cell = ia.draw_text(cell, y=27, x=2, text="%s" % (method,), size=14, color=[0, 0, 0])
            if method_idx == 0:
                cell = np.pad(cell, ((20, 0), (0, 0), (0, 0)), mode="constant", constant_values=255)
                cell = ia.draw_text(cell, y=0, x=12+40*(iterations==1), text="%d iter." % (iterations,), size=14, color=[0, 0, 0])
            cell = np.pad(cell, ((0, 1), (0, 1), (0, 0)), mode="constant", constant_values=255)

            row.append(cell)
            cell_idx += 1
        rows.append(np.hstack(row))
    gridarr = np.vstack(rows)

    # ------------

    save(
        "alpha",
        "iterative_vary_methods.jpg",
        gridarr
    )

def chapter_alpha_masks_sigmoid():
    # -----------------------------------------
    # Sigmoid varying on/off
    # -----------------------------------------
    import imgaug as ia
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    masks = []

    for activated in [False, True]:
        noise = iap.Sigmoid.create_for_noise(
            other_param=iap.FrequencyNoise(
                exponent=(-4.0, 4.0),
                upscale_method="linear"
            ),
            activated=activated
        )

        row = [noise.draw_samples((64, 64), random_state=ia.new_random_state(seed+1+i)) for i in range(16)]
        row = np.hstack(row)
        row = np.tile(row[:, :, np.newaxis], (1, 1, 3))
        row = (row * 255).astype(np.uint8)
        row = np.pad(row, ((0, 0), (90, 0), (0, 0)), mode="constant", constant_values=255)
        row = ia.draw_text(row, y=17, x=2, text="activated=\n%s" % (activated,), size=14, color=[0, 0, 0])
        masks.append(row)

    # ------------

    save(
        "alpha",
        "sigmoid_vary_activated.jpg",
        grid(masks, cols=1, rows=2)
    )

    # -----------------------------------------
    # Sigmoid varying on/off
    # -----------------------------------------
    import imgaug as ia
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    masks = []
    nb_rows = 3

    class ConstantNoise(iap.StochasticParameter):
        def __init__(self, noise, seed):
            self.noise = noise
            self.seed = seed

        def _draw_samples(self, size, random_state):
            return self.noise.draw_samples(size, random_state=ia.new_random_state(self.seed))

    for rowidx in range(nb_rows):
        row = []
        for tidx, threshold in enumerate(np.linspace(-10.0, 10.0, 10)):
            noise = iap.Sigmoid.create_for_noise(
                other_param=ConstantNoise(
                    iap.FrequencyNoise(
                        exponent=(-4.0, 4.0),
                        upscale_method="linear"
                    ),
                    seed=seed+100+rowidx
                ),
                activated=True,
                threshold=threshold
            )

            cell = noise.draw_samples((64, 64), random_state=ia.new_random_state(seed+tidx))
            cell = np.tile(cell[:, :, np.newaxis], (1, 1, 3))
            cell = (cell * 255).astype(np.uint8)
            if rowidx == 0:
                cell = np.pad(cell, ((20, 0), (0, 0), (0, 0)), mode="constant", constant_values=255)
                cell = ia.draw_text(cell, y=2, x=15, text="%.1f" % (threshold,), size=14, color=[0, 0, 0])
            row.append(cell)
        row = np.hstack(row)
        masks.append(row)

    gridarr = np.vstack(masks)
    # ------------

    save(
        "alpha",
        "sigmoid_vary_threshold.jpg",
        gridarr
    )

if __name__ == "__main__":
    main()
