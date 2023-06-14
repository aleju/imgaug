import cv2
import imgaug.augmenters as iaa
import glob
import os

# define augmentation pipeline
augmentation = iaa.Sequential([
    iaa.Grayscale(alpha=(0.2, 1.0)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
])

# get paths of all images in directory
images_path = glob.glob("./imgaug-master/augment_sample/*")

# create outputs directory if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# apply augmentation to each image and save to outputs directory
for img_path in images_path:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image: {img_path}")
        continue
    img_aug = augmentation(image=img)
    if img_aug is None or img_aug.shape[0] == 0 or img_aug.shape[1] == 0:
        print(f"Error applying augmentation to image: {img_path}")
        continue
    filename = os.path.basename(img_path)
    new_filename = f"aug_gray_{filename}"
    cv2.imwrite(f"outputs/{new_filename}", img_aug)

print("Augmentation complete!")


