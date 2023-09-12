import numpy as np
from myLDA import runLDA

# Get training & testing data
training = np.loadtxt("optdigits_train.txt", delimiter=",")
testing = np.loadtxt("optdigits_test.txt", delimiter=",")
Xtrain, Ytrain = training[:, :-1], training[:, -1]
Xtest, Ytest = testing[:, :-1], testing[:, -1]

# Run LDA w/ l=2, plot results in 2D
runLDA(2, Xtrain, Xtest, Ytrain, Ytest, True)
