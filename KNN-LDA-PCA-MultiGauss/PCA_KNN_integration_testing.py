import numpy as np
from myPCA import myPCA, getEigVecs, projectData
from myKNN import KNN

# Get data
training = np.loadtxt("optdigits_train.txt", delimiter=",")
testing = np.loadtxt("optdigits_test.txt", delimiter=",")

Xtrain, ytrain = training[:, :-1], training[:, -1]
Xtest, ytest = testing[:, :-1], testing[:, -1]

# Plot & run with two principal components
eigVecs = getEigVecs(Xtrain)
projectData(Xtrain, Xtest, ytrain, ytest, eigVecs, 2, True)

