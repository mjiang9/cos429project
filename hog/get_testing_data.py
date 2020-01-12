"""
 Princeton University, COS 429, Fall 2019
"""
import os
import cv2
from glob import glob
import numpy as np
from skimage.feature import hog


def get_testing_data(n, orientations):
    """Reads in examples of faces and nonfaces, and builds a matrix of HoG
       descriptors, ready to pass in to logistic_predict

    Args:
        n: number of face and nonface testing examples (n of each)
        orientations: the number of HoG gradient orientations to use

    Returns:
        descriptors: matrix of descriptors for all 2*n testing examples, where
                     each row contains the HoG descriptor for one face or nonface
        classes: vector indicating whether each example is a face (1) or nonface (0)
        races: vector indicating the race of each example. -1 if nonface
    """
    testing_faces_dir = '../data/testing_faces'
    testing_nonfaces_dir = '../data/testing_nonfaces'
    hog_input_size = 36
    hog_descriptor_size = 100 * orientations

    # Get the names of the first n testing faces
    face_filenames = sorted(glob(os.path.join(testing_faces_dir, '*.jpg')))
    num_face_filenames = len(face_filenames)
    if num_face_filenames > n:
        face_filenames = face_filenames[:n]
    elif num_face_filenames < n:
        n = num_face_filenames

    # Initialize descriptors, classes
    descriptors = np.zeros([2 * n, hog_descriptor_size + 1])
    classes = np.zeros([2 * n])
    races = np.zeros([2 * n])

    # Loop over faces
    for i in range(n):
        # Read the next face file
        face = cv2.imread(face_filenames[i], cv2.IMREAD_GRAYSCALE)
        face = cv2.resize(face, (hog_input_size, hog_input_size))

        # Compute HoG descriptor
        face_descriptor = (
            hog(face, 9, pixels_per_cell=(6, 6), cells_per_block=(2, 2)))

        # Fill in descriptors and classes
        descriptors[i, 0] = 1
        descriptors[i, 1:] = face_descriptor
        classes[i] = 1
        races[i] = face_filenames[i].split('_')[2]

    # Loop over nonfaces
    nonface_filenames = sorted(glob(os.path.join(testing_nonfaces_dir, '*.jpg')))
    for i in range(n, 2 * n):
        # Fill in here
        # Note that unlike in get_training_data, you are not sampling
        # random patches from the nonface images, just reading them in
        # and using them directly.

        # Read the next nonface file
        nonface = cv2.imread(nonface_filenames[i - n], cv2.IMREAD_GRAYSCALE)
        nonface = cv2.resize(nonface, (hog_input_size, hog_input_size))

        # Compute descriptor, and fill in descriptors and classes
        # Fill in here
        nonface_descriptor = (
            hog(nonface, 9, pixels_per_cell=(6, 6), cells_per_block=(2, 2)))
        descriptors[i, 0] = 1
        descriptors[i, 1:] = nonface_descriptor
        classes[i] = 0
        races[i] = -1

    return descriptors, classes, races
