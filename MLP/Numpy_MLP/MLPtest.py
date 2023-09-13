import numpy as np
from MLPtrain import normalize, load

# Tests mutli-layer perceptron on data. Gets weights & biases from vectors W and V
def MLPtest(test_data, W, V):
    # Init
    Xtest, ytest = load(test_data)
    Xtest, mean, std = normalize(Xtest)
    N = Xtest.shape[0]

    # Add bias term to input
    Xtest = np.hstack((Xtest, np.ones((N, 1))))

    # Hidden layer activations
    Z = np.maximum(np.dot(Xtest, W), 0)
    Z = np.hstack((Z, np.ones((N, 1))))

    # Compute output layer activations, test error rate, print
    preds = np.argmax(np.dot(Z, V), axis=1)
    err = np.mean(preds != ytest)
    print("Test error rate: {:.2f}%".format(err * 100))
    return Z
