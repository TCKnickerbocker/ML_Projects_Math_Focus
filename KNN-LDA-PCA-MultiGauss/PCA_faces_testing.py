import numpy as np
import matplotlib.pyplot as plt
from myPCA import myPCA, facesPCA, combinedPCA

# Load & Extract data
train_data = np.loadtxt('face_train_data_960.txt')
test_data = np.loadtxt('face_test_data_960.txt')
Xtrain, ytrain = train_data[:, :-1], train_data[:, -1]
Xtest, ytest = test_data[:, :-1], test_data[:, -1]

# Combine training & testing X data, get eigVecs from PCA
allX = np.vstack([Xtrain, Xtest])
labels = np.concatenate((ytrain, ytest), axis=0)
ks = np.asarray(myPCA(Xtrain, False)) # Get k-value accounting for 90% of variance
k = np.amin(ks)
newFaces = facesPCA(allX, k)


# Original images
for i in range(5):
    curFace = allX[i, :]
    plt.imshow(np.reshape(curFace, (30, 32)))
    plt.title("Original with label " + str(labels[i]))
    plt.show()

# Post-PCA Images
for i in range(5):
    curFace = newFaces[i, :]
    plt.imshow(np.real(np.reshape(curFace, (30, 32)))) # Have to do np.real because my returned values include imaginary numbers
    plt.title("Post-PCA with label " + str(labels[i]))
    plt.show()

