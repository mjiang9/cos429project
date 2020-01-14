"""
 Princeton University, COS 429, Fall 2019
"""
import os
import cv2
import random
from glob import glob
import numpy as np
from skimage.feature import hog
from logistic_predict import logistic_predict

def get_false_positives(n, orientations):
    """Uses saved model to test randomly sampled patches in nonfaces, and
    save the false positives.

    Args:
        orientations: the number of HoG gradient orientations to use
        wrap180: if true, the HoG orientations cover 180 degrees, else 360

    Returns:
        descriptors: matrix of descriptors for all 2*n training examples, where
                     each row contains the HoG descriptor for one face or nonface
        classes: vector indicating whether each example is a face (1) or nonface (0)
    """
    hog_input_size = 36
    hog_descriptor_size = 100 * orientations
    saved = np.load('face_classifier.npz')
    params, orientations = saved['params'], saved['orientations']
    nonfaces_dir = '../data/training_nonfaces'

    # Get the filenames within the training faces directory
    nonface_filenames = sorted(glob(os.path.join(nonfaces_dir, '*.jpg')))
    num_nonface_filenames = len(nonface_filenames)

    # directory to save images
    folder_name = "false_pos/"

    # Loop over images
    count = 0
    while count < n:
        # Read a random nonface file
        j = random.randint(0, num_nonface_filenames - 1)
        nonface = cv2.imread(nonface_filenames[j], cv2.IMREAD_GRAYSCALE)

        # Crop out a random square at least hog_input_size
        k = random.randint(hog_input_size, min(nonface.shape))
        row = random.randint(0, nonface.shape[0] - k)
        col = random.randint(0, nonface.shape[1] - k)
        crop = nonface[row:row+k , col:col+k]

        # Resize to be the right size
        crop = cv2.resize(crop, (hog_input_size, hog_input_size))

        # Compute descriptor and calculated model probability
        descriptor = np.zeros(hog_descriptor_size + 1)
        hog_o = hog(crop, orientations, pixels_per_cell=(6, 6), cells_per_block=(2, 2))

        descriptor[0] = 1
        descriptor[1:] = hog_o
        predicted = logistic_predict(descriptor, params)
        if predicted != 0:
            print(count)
            # save image
            name = "wrong_" + str(count) + '.jpg'
            cv2.imwrite(os.path.join(folder_name, name), crop)
            count += 1

def get_false_negatives(n, orientations):
    """Uses saved model to test randomly sampled patches in nonfaces, and
    save the false positives.

    Args:
        orientations: the number of HoG gradient orientations to use
        wrap180: if true, the HoG orientations cover 180 degrees, else 360

    Returns:
        descriptors: matrix of descriptors for all 2*n training examples, where
                     each row contains the HoG descriptor for one face or nonface
        classes: vector indicating whether each example is a face (1) or nonface (0)
    """
    hog_input_size = 36
    hog_descriptor_size = 100 * orientations
    saved = np.load('face_classifier.npz')
    params, orientations = saved['params'], saved['orientations']
    testing_dir = '../data/train_50'

    # Get the filenames within the training faces directory
    testing_filenames = sorted(glob(os.path.join(testing_dir, '*.jpg')))
    num_filenames = len(testing_filenames)

    # directory to save images
    folder_name = "false_neg/"

    # Loop over images
    count = 0
    while count < n:
        # Read a random face file
        j = random.randint(0, num_filenames - 1)
        face = cv2.imread(testing_filenames[j], cv2.IMREAD_GRAYSCALE)

        # Resize to be the right size
        crop = cv2.resize(face, (hog_input_size, hog_input_size))

        # Compute descriptor and calculated model probability
        descriptor = np.zeros(hog_descriptor_size + 1)
        hog_o = hog(crop, orientations, pixels_per_cell=(6, 6), cells_per_block=(2, 2))

        descriptor[0] = 1
        descriptor[1:] = hog_o
        predicted = logistic_predict(descriptor, params)
        if predicted != 1:
            print(count)
            # save image
            name = "hard_" + str(count) + '.jpg'
            cv2.imwrite(os.path.join(folder_name, name), crop)
            count += 1



def main():
    get_false_positives(20, 9)
    get_false_negatives(20, 9)

if __name__ == '__main__':
    main()
