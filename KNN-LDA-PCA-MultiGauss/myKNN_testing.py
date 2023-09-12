import numpy as np
from myKNN import KNN

# Extract data
training = np.loadtxt("optdigits_train.txt", delimiter=",")
testing = np.loadtxt("optdigits_test.txt", delimiter=",")

Xtrain, ytrain = training[:, :-1], training[:, -1]
Xtest, ytest = testing[:, :-1], testing[:, -1]

# For each k, predict, get error rate, print k and arror rate
for k in [1, 3, 5, 7]:
    knn = KNN(k)
    knn.fit(Xtrain, ytrain)
    preds, labels = knn.predict(Xtest)
    error = np.mean(preds != ytest)
    print(f'k = {k}: error rate = {error:5f}')

