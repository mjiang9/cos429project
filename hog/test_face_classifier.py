"""
 Princeton University, COS 429, Fall 2019
"""
import numpy as np
from get_training_data import get_training_data
from get_testing_data import get_testing_data
from logistic_prob import logistic_prob
from logistic_fit import logistic_fit
from logistic_predict import logistic_predict # added this to test
import matplotlib.pyplot as plt


def test_face_classifier(ntrain, ntest, orientations, wrap180, dataset_name):
    """Train and test a face classifier.

    Args:
        ntrain: number of face and nonface training examples (ntrain of each)
        ntest: number of face and nonface testing examples (ntest of each)
        orientations: the number of HoG gradient orientations to use
        wrap180: if true, the HoG orientations cover 180 degrees, else 360
    """
    # Get some training data
    descriptors, classes = get_training_data(ntrain, orientations, wrap180, dataset_name)
    # Train a classifier
    params = logistic_fit(descriptors, classes, 0.001)

    # Evaluate the classifier on the training data
    predicted = logistic_prob(descriptors, params)
    #plot_errors(predicted, classes, 'Performance on training set for varying threshold', 1)

    # Get some test data
    tdescriptors, tclasses, races = get_testing_data(ntest, orientations, wrap180)

    # Evaluate the classifier on the test data
    # Add in prediction accuracy??
    tpredicted2 = logistic_predict(tdescriptors, params)
    testing_accuracy = np.sum(tpredicted2 == tclasses) / len(tclasses)
    print('overall testing_accuracy =', testing_accuracy)

    # Evaluate classifier on races separately
    # white
    w_descriptors = tdescriptors[races == 0, :]
    w_classes = tclasses[races == 0]
    w_predicted = logistic_predict(w_descriptors, params)
    testing_accuracy = np.sum(w_predicted == w_classes) / len(w_classes)
    print('white testing_accuracy =', testing_accuracy)

    # black
    b_descriptors = tdescriptors[races == 1, :]
    b_classes = tclasses[races == 1]
    b_predicted = logistic_predict(b_descriptors, params)
    testing_accuracy = np.sum(b_predicted == b_classes) / len(b_classes)
    print('black testing_accuracy =', testing_accuracy)

    #tpredicted = logistic_prob(tdescriptors, params)
    #plot_errors(tpredicted, tclasses, 'Performance on test set for varying threshold', 2)

    np.savez('face_classifier.npz', params=params, orientations=orientations, wrap180=wrap180)


def plot_errors(predicted, classes, name, num):
    """Plot a log/log graph of miss rate (false negatives) vs false positives
       for a variety of thresholds on probability.

    Args:
        predicted: probabilities that the class is 1
        classes: ground-truth class labels (0/1)
        name: name of the figure
        num: number of the figure
    """
    nthresh = 99
    npts = predicted.shape[0]

    falsepos = np.zeros([nthresh])
    falseneg = np.zeros([nthresh])

    stepsize = 1. / (nthresh + 1)
    for i in range(nthresh):
        thresh = (i + 1) * stepsize
        falsepos[i] = np.sum(np.logical_and(predicted >= thresh, classes == 0)) / npts
        falseneg[i] = np.sum(np.logical_and(predicted < thresh, classes == 1)) / npts

    limit = 1e-4
    plt.figure(num)
    plt.title(name)
    plt.loglog(np.maximum(falsepos, limit), np.maximum(falseneg, limit))
    plt.axis([limit, 1, limit, 1])
    plt.xlabel('False positive rate')
    plt.ylabel('False negative rate')

    plt.show()
