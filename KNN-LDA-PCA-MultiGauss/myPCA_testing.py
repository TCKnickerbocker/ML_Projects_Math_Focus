import numpy as np
from myPCA import myPCA, getEigVecs, projectData
from myKNN import KNN

# Load data, get ks
training = np.loadtxt("optdigits_train.txt", delimiter=",")
testing = np.loadtxt("optdigits_test.txt", delimiter=",")

Xtrain, ytrain = training[:, :-1], training[:, -1]
Xtest, ytest = testing[:, :-1], testing[:, -1]

print("K with 90`% of explained variance:")
ks = np.asarray(myPCA(Xtrain, True))
k = np.amin(ks)

# Get projected data, get error rates for data using k values:
eigVecs = getEigVecs(Xtrain)

newXtrain, newXtest = projectData(Xtrain, Xtest, ytrain, ytest, eigVecs, 20, False) # 20 is best

# Test each k-value
for k in [1, 3, 5, 7]:
    knn = KNN(k)
    knn.fit(newXtrain, ytrain)
    preds, labels = knn.predict(newXtest)
    error = np.mean(preds != ytest)
    print(f'k = {k}, error rate = {error:5f}')
