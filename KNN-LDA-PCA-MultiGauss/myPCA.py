from sklearn import *
import numpy as np
import matplotlib.pyplot as plt


# Returns best k for PCA, and plots if plot=True
def myPCA(Xtrain, plot):
    # Normalize, get covariance, eigs
    normalized = Xtrain - np.mean(Xtrain, axis=0)
    cov = np.cov(normalized.T)
    eigVals, eigVecs = np.linalg.eigh(cov)

    # Get list of explained variance for values & cumulative variance explained
    totalVar = sum(eigVals)
    explainedVar = []
    for eigval in sorted(eigVals, reverse=True):
        explainedVar.append(eigval / totalVar)
    cumVarExpl = np.cumsum(explainedVar)

    # Pick minimum k value explaining 90% of variance, plot
    minK = min(np.where(cumVarExpl >= .9))
    if plot:
        plt.plot(range(1, len(explainedVar)+1), cumVarExpl)
        plt.xlabel('K-value')
        plt.ylabel('Proportion of variance explained')
        plt.show()
    return minK

# Gets eigenvectors of training data, returns them in sorted order
def getEigVecs(Xtrain):
    # Center data, get cov mat
    Xtrain = Xtrain - np.mean(Xtrain, axis=0)
    cov = np.cov(Xtrain, rowvar=False)

    # Compute the eigenvalues and eigenvectors of the covariance matrix, sort eigenVectors
    eigVals, eigVecs = np.linalg.eigh(cov)
    sortIndexes = np.argsort(eigVals)[::-1]
    eigVecs = eigVecs[:, sortIndexes]

    return eigVecs

# Projects data onto top k eigenvectors, returns projected data
def projectData(Xtrain, Xtest, Ytrain, Ytest, sortedEigVecs, k, plot):
    # Use first k eigVecs as dot product on X data
    topEigVecs = sortedEigVecs[:, :k]
    newXTrain = np.dot(Xtrain, topEigVecs)
    newXTest = np.dot(Xtest, topEigVecs)

    # Plot PCA for training & testing onto 2 dimensions with 10 different colors
    if plot:
        dotColors = ['red', 'orange', 'yellow', 'green', 'blue', 'pink', 'purple', 'lime', 'maroon', 'cyan']
        plt.figure(figsize=(7.5, 7.5))
        for i in range(10):
            plt.scatter(newXTrain[Ytrain==i, 0], newXTrain[Ytrain==i, 1], color=dotColors[i], label=str(i), alpha=0.4)
        plt.legend()
        plt.title('PCA Projection on Training Data')
        plt.xlabel('PCA Dim 1')
        plt.ylabel('PCA Dim 2')
        plt.show()

        plt.figure(figsize=(7.5, 7.5))
        for i in range(10):
            plt.scatter(newXTest[Ytest==i, 0], newXTest[Ytest==i, 1], color=dotColors[i], label=str(i), alpha=0.4)
        plt.legend()
        plt.title('PCA Projection on Test Data')
        plt.xlabel('PCA Dim 1')
        plt.ylabel('PCA Dim 2')
        plt.show()
    return newXTrain, newXTest


# Computes dot-product of top-k eigenvectors
def combinedPCA(allXData, sortedEigVecs, k):
    topEigVecs = sortedEigVecs[:, :k]
    return np.dot(allXData, topEigVecs)

# Computes PCA on data using k-components
def facesPCA(X, k):
    # Center data, get covMat
    centeredX = X - np.mean(X, axis=0)
    covMat = np.cov(centeredX.T)

    # Get eigs of covMat, sort
    eigVals, eigVecs = np.linalg.eigh(covMat)
    sortIndexes = np.argsort(eigVals)[::-1]
    eigVecs = eigVecs[:, sortIndexes]

    # 
    firstKEigs = eigVecs[:, :k]
    projectedFaces = np.dot(centeredX, firstKEigs)
    newFaces = np.dot(projectedFaces, firstKEigs.T)
    return newFaces

