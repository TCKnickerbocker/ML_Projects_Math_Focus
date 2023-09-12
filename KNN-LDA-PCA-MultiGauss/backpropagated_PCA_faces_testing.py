import numpy as np
import matplotlib.pyplot as plt

# Load data
train = np.loadtxt('face_train_data_960.txt') 
Xtrain = train[:, :-1]

# Center/normalize, get covMat
averageFace = np.mean(Xtrain, axis=0)
centeredX = Xtrain - np.mean(Xtrain, axis=0)
covMat = np.cov(centeredX.T)

# Get eigs of covMat, sort
eigVals, eigVecs = np.linalg.eigh(covMat)
sortIndexes = np.argsort(eigVals)[::-1]
eigVecs = eigVecs[:, sortIndexes]

# Project data onto k components, reconstruct original via backprojection
for k in [10, 50, 100]:
    # Get eigs, project, backproject
    firstKEigs = eigVecs[:, :k]
    projectedFaces = np.dot(centeredX, firstKEigs)
    newFaces = np.dot(projectedFaces, firstKEigs.T) + averageFace

    # Plot the reconstructed images
    for i in range(5):
        # Get current image, reshape & display
        origImg = np.reshape(Xtrain[i, :], (30, 32))
        backProjectedImage = newFaces[i, :]
        backProjectedImage = (np.reshape(backProjectedImage, (30, 32))) 

        # Show current image, backprojected image
        plt.imshow(origImg)
        plt.title("Original")
        plt.show()

        plt.imshow(backProjectedImage)
        plt.title('K= ' + str(k))
        plt.show()
