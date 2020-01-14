"""
 Princeton University, COS 429, Fall 2019
"""
import os
import cv2
import random
from glob import glob
import numpy as np
from random_tree import random_forest_fit
from skimage.feature import hog

"""
Classifier using HOG features for predicting whether a face is dark-skinned
or light-skinned.
"""

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

def get_training_data(n, orientations, train_face_dir):
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
    hog_input_size = 36
    hog_descriptor_size = 100 * orientations

    # Get n faces, with same black/white distribution as the training set
    face_files = get_imageset(n, train_face_dir)

    # Initialize descriptors, classes
    descriptors = np.zeros([n, hog_descriptor_size + 1])
    classes = np.zeros([n])

    # Loop over faces
    for i, file in enumerate(face_files):
        # count number of white and black
        race = int(file.split('_')[3])
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
        classes[i] = race

    return descriptors, classes
"""
 Princeton University, COS 429, Fall 2019
"""
import os
import cv2
from glob import glob
import numpy as np
from skimage.feature import hog


def get_testing_data(n, orientations):
    testing_faces_dir = '../data/testing_faces'
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
    descriptors = np.zeros([n, hog_descriptor_size + 1])
    classes = np.zeros([n])
    races = np.zeros([n])

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
    return descriptors, classes, races

def test_race_classifier(ntrain, ntest, orientations, train_face_dir):
    """Train and test a face classifier with logistic function.

    Args:
        ntrain: number of face and nonface training examples (ntrain of each)
        ntest: number of face and nonface testing examples (ntest of each)
        orientations: the number of HoG gradient orientations to use
    """
    # Get some training data
    descriptors, classes = get_training_data(ntrain, orientations, train_face_dir)
    # Train a classifier
    params = logistic_fit(descriptors, classes, 0.001)

    # Evaluate the classifier on the training data
    # predicted = logistic_prob(descriptors, params)
    #plot_errors(predicted, classes, 'Performance on training set for varying threshold', 1)

    # Get some test data
    tdescriptors, tclasses, races = get_testing_data(ntest, orientations)

    # Evaluate the classifier on the test data
    # Add in prediction accuracy??
    tpredicted = logistic_predict(tdescriptors, params)
    overall_testing_accuracy = np.sum(tpredicted == races) / len(races)
    print(overall_testing_accuracy)
    print('overall accuracy = ' + str(overall_testing_accuracy))

    # Evaluate classifier on races separately
    # white
    w_descriptors = tdescriptors[races == 0, :]
    w_classes = races[races == 0]
    w_predicted = logistic_predict(w_descriptors, params)
    white_testing_accuracy = np.sum(w_predicted == w_classes) / len(w_classes)
    print('white accuracy = ' + str(white_testing_accuracy))

    # black
    b_descriptors = tdescriptors[races == 1, :]
    b_classes = tclasses[races == 1]
    b_predicted = logistic_predict(b_descriptors, params)
    black_testing_accuracy = np.sum(b_predicted == b_classes) / len(b_classes)
    print('black accuracy = ' + str(black_testing_accuracy))

    #np.savez('face_classifier_hard.npz', params=params, orientations=orientations, wrap180=wrap180)
    return overall_testing_accuracy, white_testing_accuracy, black_testing_accuracy

def main():
    test_race_classifier(4020, 300, 9, 'train_50')

if __name__ == "__main__":
    main()
