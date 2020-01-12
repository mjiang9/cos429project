"""
 Princeton University, COS 429, Fall 2019
"""
import os
import cv2
import random
from glob import glob
import numpy as np
from logistic_prob import logistic_prob
from logistic_fit import logistic_fit
from skimage.feature import hog

def get_hog_images(n, orientations, face_dir):
    """Reads in examples of faces and nonfaces, and builds a matrix of HoG
       descriptors, ready to pass in to logistic_fit

    Args:
        orientations: the number of HoG gradient orientations to use

    Returns:
        descriptors: matrix of descriptors for all 2*n training examples, where
                     each row contains the HoG descriptor for one face or nonface
        classes: vector indicating whether each example is a face (1) or nonface (0)
    """
    faces_dir = ('../data/' + face_dir)
    hog_input_size = 36
    hog_descriptor_size = 100 * orientations

    # Get the filenames within the training faces directory
    face_filenames = sorted(glob(os.path.join(faces_dir, '*.jpg')))
    num_face_filenames = len(face_filenames)

    if num_face_filenames > n:
        face_filenames = face_filenames[:n]
    elif num_face_filenames < n:
        n = num_face_filenames

    # Initialize hog_images, races
    hog_images = np.empty(n, dtype=object)
    races = np.zeros([n])

    # Loop over faces
    for i in range(n):
        # Read the next face file
        face = cv2.imread(face_filenames[i], cv2.IMREAD_GRAYSCALE)
        # Resize to be the right size
        face = cv2.resize(face, (hog_input_size, hog_input_size))

        # Compute HoG descriptor
        #face_descriptor = hog36(face, orientations, wrap180)
        _, hog_image = (
            hog(face, 9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), visualize=True))
        hog_images[i] = hog_image
        races[i] = face_filenames[i].split('_')[2]

    return hog_images, races
