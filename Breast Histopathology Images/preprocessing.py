"""
Python 3.6.3
script for preprocessing a dataset and creating images and corresponding labels
created only because Jupyter is too messy for my taste
!!no actual need to use it!!
"""
import glob
import numpy as np
import cv2
import warnings

def read_data():
    all_patches = glob.glob('./data/**/*.png', recursive=True)
    class0 = []
    class1 = []
    for name in glob.glob('./data/**/*class0.png', recursive=True):
        class0.append(name)
    for name in glob.glob('./data/**/*class1.png', recursive=True):
        class1.append(name)
    return all_patches, class0, class1

def prepare_data_resize(patches, class0, class1):
    """ divide initial data into images and labels
        this function won't be used in a final version, because it has been decided
        against interpolating images, since it changes the image
        Min-Max scaling all images
    Args: 
        patches : list of file path names
        class0 : list of file names with negative examples
        class1 : list of file names with positive examples
    Returns:
        scaled_X : array with resized images
        y : array with corresponding labels
        (0: non-IDC, 1: IDC)
    """
    warnings.warn("this function uses interpolation and is deprecated", DeprecationWarning,
              stacklevel=2)
    X = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    for img_name in patches:
        image = cv2.imread(img_name)
        image_resized = cv2.resize(image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC) 
        X.append(image_resized)
        if img_name in class0:
            y.append(0)
        else:
            y.append(1)
    scaled_X = np.array(X)/255.0
    return scaled_X, y

def prepare_data_pad(patches, class0, class1):
    """ divides data into images and labels
        pads images that are smaller than 50x50 with white (note: this isn't zero-padding,
        decided against black, because cancer parts of the images are dark)
        Min-Max scales all images 
    Args: 
        patches : list of file path names
        class0 : list of file names with negative examples
        class1 : list of file names with positive examples
    Returns:
        scaled_X : array with padded images
        y : array with corresponding labels
        (0 for non-IDC, 1 for IDC)
    """
    X = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    for img_name in patches:
        image = cv2.imread(img_name)
        orig_height, orig_width, _ = image.shape
        if (orig_height != 50 or orig_width != 50):
            height_diff = HEIGHT - orig_height
            width_diff = WIDTH - orig_width
            top = height_diff // 2 # floor division
            bottom = height_diff - top
            left = width_diff // 2
            right = width_diff - left
            constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])
            X.append(constant)
        else:
            X.append(image)
        y.append(0) if img_name in class0 else y.append(1)
    scaled_X = np.array(X)/255.0
    return scaled_X, y

def main():
    patches, class0, class1 = read_data()
    X, y = prepare_data_pad(patches, class0, class1)
    np.save('padded_imgs.npy', X) 
    np.save('labels.npy', y)

if __name__ == "__main__":
    main()
