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

""" Returns subset of n filenames from train_face_dir, where the distribution of
    black/white faces matches the overall distribution of black/white faces in
    train_face_dir.
"""
def get_imageset(n, train_face_dir):
    training_faces_dir = ('../data/' + train_face_dir)
    # Get the filenames within the training faces directory
    face_filenames = sorted(glob(os.path.join(training_faces_dir, '*.jpg')))

    percent = float(train_face_dir.split('_')[1]) / 100
    total_b = int(percent * n)
    total_w = n - total_b
    num_b = 0
    num_w = 0
    all_b = False
    all_w = False

    filenames = []

    for i, face_file in enumerate(face_filenames):
        if (all_w and all_b):
            break
        race = face_file.split('_')[3]
        if race == '0':  # white
            if num_w < total_w:
                filenames.append(face_file)
                num_w += 1
            else:
                all_w = True
        elif race == '1': # black
            if num_b < total_b:
                filenames.append(face_file)
                num_b += 1
            else:
                all_b = True
    return filenames

def get_training_data(n, orientations, train_face_dir, train_nonface_dir):
    """Reads in examples of faces and nonfaces, and builds a matrix of HoG
       descriptors, ready to pass in to logistic_fit

    Args:
        n: number of training and testing
        orientations: the number of HoG gradient orientations to use

    Returns:
        descriptors: matrix of descriptors for all 2*n training examples, where
                     each row contains the HoG descriptor for one face or nonface
        classes: vector indicating whether each example is a face (1) or nonface (0)
    """
    training_nonfaces_dir = '../data/' + train_nonface_dir
    hog_input_size = 36
    hog_descriptor_size = 100 * orientations

    # Get n faces, with same black/white distribution as the training set
    face_files = get_imageset(n, train_face_dir)

    # Initialize descriptors, classes
    descriptors = np.zeros([2 * n, hog_descriptor_size + 1])
    classes = np.zeros([2 * n])

    # Loop over faces
    white = 0
    black = 0
    for i, file in enumerate(face_files):
        # count number of white and black
        race = file.split('_')[3]
        if race == '0':
            white += 1
        elif race == '1':
            black += 1
        # Read the face file
        face = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # Resize to be the right size
        face = cv2.resize(face, (hog_input_size, hog_input_size))

        # Compute HoG descriptor
        #face_descriptor = hog36(face, orientations, wrap180)
        face_descriptor = (
            hog(face, 9, pixels_per_cell=(6, 6), cells_per_block=(2, 2)))

        # Fill in descriptors and classes
        descriptors[i, 0] = 1
        descriptors[i, 1:] = face_descriptor
        classes[i] = 1

    # print black/white counts
    #print("number of black = " + str(black))
    #print("number of white = " + str(white))
    # Get the names of the nonfaces
    nonface_filenames = sorted(glob(os.path.join(training_nonfaces_dir, '*.jpg')))
    num_nonface_filenames = len(nonface_filenames)

    # Loop over all nonface samples we want
    for i in range(n, 2 * n):
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

        # Compute descriptor, and fill in descriptors and classes
        # Fill in here
        #nonface_descriptor = hog36(crop, orientations, wrap180)
        nonface_descriptor = (
            hog(crop, 9, pixels_per_cell=(6, 6), cells_per_block=(2, 2)))
        descriptors[i, 0] = 1
        descriptors[i, 1:] = nonface_descriptor
        classes[i] = 0

    return descriptors, classes
