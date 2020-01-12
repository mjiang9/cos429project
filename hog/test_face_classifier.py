"""
 Princeton University, COS 429, Fall 2019
"""
import numpy as np
from get_training_data import get_training_data
from get_testing_data import get_testing_data
from logistic_prob import logistic_prob
from logistic_fit import logistic_fit
from logistic_predict import logistic_predict # added this to test
from random_tree import random_forest_fit
import matplotlib.pyplot as plt


def test_face_classifier_l(ntrain, ntest, orientations, train_face_dir, train_nonface_dir):
    """Train and test a face classifier with logistic function.

    Args:
        ntrain: number of face and nonface training examples (ntrain of each)
        ntest: number of face and nonface testing examples (ntest of each)
        orientations: the number of HoG gradient orientations to use
    """
    # Get some training data
    descriptors, classes = get_training_data(ntrain, orientations, train_face_dir, train_nonface_dir)
    # Train a classifier
    params = logistic_fit(descriptors, classes, 0.001)

    # Evaluate the classifier on the training data
    # predicted = logistic_prob(descriptors, params)
    #plot_errors(predicted, classes, 'Performance on training set for varying threshold', 1)

    # Get some test data
    tdescriptors, tclasses, races = get_testing_data(ntest, orientations)

    # Evaluate the classifier on the test data
    # Add in prediction accuracy??
    tpredicted2 = logistic_predict(tdescriptors, params)
    overall_testing_accuracy = np.sum(tpredicted2 == tclasses) / len(tclasses)
    #print('overall accuracy = ' + str(overall_testing_accuracy))

    # Evaluate classifier on races separately
    # white
    w_descriptors = tdescriptors[races == 0, :]
    w_classes = tclasses[races == 0]
    w_predicted = logistic_predict(w_descriptors, params)
    white_testing_accuracy = np.sum(w_predicted == w_classes) / len(w_classes)
    #print('white accuracy = ' + str(white_testing_accuracy))

    # black
    b_descriptors = tdescriptors[races == 1, :]
    b_classes = tclasses[races == 1]
    b_predicted = logistic_predict(b_descriptors, params)
    black_testing_accuracy = np.sum(b_predicted == b_classes) / len(b_classes)
    #print('black accuracy = ' + str(black_testing_accuracy))

    # false positives with prob > 0.5
    npts = tpredicted2.shape[0]
    falsepos = np.sum(np.logical_and(tpredicted2 == 1, tclasses == 0)) / npts

    #tpredicted = logistic_prob(tdescriptors, params)
    #plot_errors(tpredicted, tclasses, 'Performance on test set for varying threshold', 2)

    #np.savez('face_classifier_hard.npz', params=params, orientations=orientations, wrap180=wrap180)
    return overall_testing_accuracy, white_testing_accuracy, black_testing_accuracy, falsepos

def test_face_classifier_rf(ntrain, ntest, orientations, train_face_dir, train_nonface_dir):
    """Train and test a face classifier with logistic function.

    Args:
        ntrain: number of face and nonface training examples (ntrain of each)
        ntest: number of face and nonface testing examples (ntest of each)
        orientations: the number of HoG gradient orientations to use
    """
    # Get some training data
    descriptors, classes = get_training_data(ntrain, orientations, train_face_dir, train_nonface_dir)
    # Train a classifier
    clf = random_forest_fit(descriptors, classes)

    # Get some test data
    tdescriptors, tclasses, races = get_testing_data(ntest, orientations)

    # Evaluate the classifier on the test data
    # Add in prediction accuracy??
    tpredicted2 = clf.predict(tdescriptors)
    overall_testing_accuracy = np.sum(tpredicted2 == tclasses) / len(tclasses)
    #print('overall accuracy = ' + str(overall_testing_accuracy))

    # Evaluate classifier on races separately
    # white
    w_descriptors = tdescriptors[races == 0, :]
    w_classes = tclasses[races == 0]
    w_predicted = clf.predict(w_descriptors)
    white_testing_accuracy = np.sum(w_predicted == w_classes) / len(w_classes)
    #print('white accuracy = ' + str(white_testing_accuracy))

    # black
    b_descriptors = tdescriptors[races == 1, :]
    b_classes = tclasses[races == 1]
    b_predicted = clf.predict(b_descriptors)
    black_testing_accuracy = np.sum(b_predicted == b_classes) / len(b_classes)
    #print('black accuracy = ' + str(black_testing_accuracy))

    # false positives
    npts = tpredicted2.shape[0]
    falsepos = np.sum(np.logical_and(tpredicted2 == 1, tclasses == 0)) / npts

    #tpredicted = logistic_prob(tdescriptors, params)
    #plot_errors(tpredicted, tclasses, 'Performance on test set for varying threshold', 2)

    #np.savez('face_classifier_hard.npz', params=params, orientations=orientations, wrap180=wrap180)
    return overall_testing_accuracy, white_testing_accuracy, black_testing_accuracy, falsepos
