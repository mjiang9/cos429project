"""
 Princeton University, COS 429, Fall 2019
"""
import numpy as np

def logistic_fit(X, z, l):
    """Performs L2-regularized logistic regression via Gauss-Newton iteration

    Args:
        X: datapoints (one per row, should include a column of ones
                       if the model is to have a constant)
        z: labels (0/1)
        l: lambda, regularization parameter (will be scaled by the number of examples)

    Returns:
        params: vector of parameters
    """
    num_pts, num_vars = X.shape

    # Linear regression to compute initial estimate.
    # We need to apply a correction to z for just the first
    # linear fit, since the nonlinearity isn't being applied.
    z_corr = 2 * z - 1
    params = np.linalg.inv(X.T @ X + l * num_pts * np.identity(num_vars)) @ (X.T @ z_corr)

    I = np.identity(num_vars)
    # Now iterate to improve params
    for iter in range(10):
        # Fill in here
        prediction = logistic(X @ params)
        r = z - prediction
        W = np.diag(prediction * (1 - prediction))
        J = W @ X
        delta = np.linalg.solve(J.T @ J + l * num_pts * I, J.T @ r)
        params = params + delta
    return params


def logistic(x):
    """The logistic "sigmoid" function
    """
    pass
    return 1 / (1 + np.exp(-x))

def logistic_prob(X, params):
    """Given a logistic model and some new data, predicts probability that
       the class is 1.

    Args:
        X: datapoints (one per row, should include a column of ones
                       if the model is to have a constant)
        params: vector of parameters

    Returns:
        z: predicted probabilities (0..1)
    """
    # Fill in here
    return logistic(X @ params)
