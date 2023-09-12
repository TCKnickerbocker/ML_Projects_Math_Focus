import numpy as np
import matplotlib.pyplot as plt
from myKNN import KNN
from myPCA import facesPCA, myPCA

# Load data, extract training/testing values
train = np.loadtxt('face_train_data_960.txt')
test = np.loadtxt('face_test_data_960.txt')
Xtrain, ytrain = train[:, :-1], train[:, -1]
Xtest, ytest = test[:, :-1], test[:, -1]

# Combine data, get a good k-value, re-run PCA on X data
allX = np.vstack([Xtrain, Xtest])
labels = np.concatenate((ytrain, ytest), axis=0)
ks = np.asarray(myPCA(Xtrain, False)) # Get k-value accounting for 90% of variance
k = np.amin(ks)
newFaces = facesPCA(allX, k)

# First 5 faces. Uses 960 principal components, since 30X32 = 960
for i in range(5):
    curFace = newFaces[i, :]
    plt.imshow(np.reshape(curFace, (30, 32))) 
    plt.title('Label: ' + str(ytrain[i]))
    plt.show()

# Use KNN on the sorted eigenvectors
knn = KNN(5)
knn.fit(newFaces, labels)


for i in range(5):
    print("Actual: ", labels[i])
    preds, kLabels = knn.predict(Xtest[i])
    print("Prediction labels: ", kLabels)
    print("Prediction: ", np.bincount(kLabels).argmax())


