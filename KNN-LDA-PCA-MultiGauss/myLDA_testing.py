import numpy as np
from myLDA import runLDA
from myKNN import KNN

# Get training & testing data
training = np.loadtxt("optdigits_train.txt", delimiter=",")
testing = np.loadtxt("optdigits_test.txt", delimiter=",")
Xtrain, Ytrain = training[:, :-1], training[:, -1]
Xtest, Ytest = testing[:, :-1], testing[:, -1]


# Run LDA for new x given l, run KNN for each k with that l, calculate & print error rate
for l in [2, 4, 9]:
    XtrainLDA, XtestLDA = runLDA(l, Xtrain, Xtest, Ytrain, Ytest, False)
    for k in [1, 3, 5]:
        knn = KNN(k)
        knn.fit(XtrainLDA, Ytrain)
        preds, labels = knn.predict(XtestLDA)
        error = np.mean(preds != Ytest)
        print(f"l= {l}, k = {k}, error = {error:5f}")

