import numpy as np

def MyPerceptron(X, y, w0):
    # Get num samples (m) & features (n) from input data
    m, n = np.shape(X)
    # Initialize the weight vector with w0 and a bias term (0)
    w = np.append(w0, 0)
    # Add a column of ones to the input data for the bias term
    X = np.hstack((X, np.ones((m, 1))))
    # Calculate the initial classification error rate
    err = sum(np.sign(X @ w) != y) / m
    step = 0

    # While classification errors still exist
    while err > 0:
        for i in range(m):
            # If misclassified sample found, update weight vector
            if X[i, :] @ w * y[i] <= 0:
                w = w + y[i] * X[i, :].T
        step = step + 1
        # Calculate the classification error rate after the update
        err = sum(np.sign(X @ w) != y) / m

    # Return updated weight vector & #stepsTaken
    return w, step
