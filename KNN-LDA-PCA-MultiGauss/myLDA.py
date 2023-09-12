import numpy as np
import matplotlib.pyplot as plt



def runLDA(l, Xtrain, Xtest, Ytrain, Ytest, plot):
    # Get mean, between-scatter, within-scatter matricies 
    Xmean = np.mean(Xtrain, axis=0)
    betweenScatter = np.zeros((Xtrain.shape[1], Xtrain.shape[1])) # square matrix
    inScatter = np.zeros((Xtrain.shape[1], Xtrain.shape[1])) # square matrix
    cMeans = []
    # Get overall mean, means for each class
    for clas in np.unique(Ytrain):
        cMeans.append(np.mean(Xtrain[Ytrain == clas], axis=0))

    # Get within-class scatterMat
    for clas, cMean in zip(np.unique(Ytrain), cMeans):
        XofClass = Xtrain[Ytrain == clas]
        centered = XofClass - cMean
        inScatter += np.dot(centered.T, centered) # Square centered class matrix
    
    # Get between-class scatterMat
    for clas, cMean in zip(np.unique(Ytrain), cMeans):
        numXs = Xtrain[Ytrain == clas].shape[0]
        betweenScatter += numXs * np.outer(cMean - Xmean, cMean - Xmean)

    # Get eigs of inScatter^-1 dot betweenScatter, sort eigVecs
    eigVals, eigVecs = np.linalg.eig(np.linalg.pinv(inScatter).dot(betweenScatter))
    sortInd = np.argsort(eigVals)[::-1] # sort descending order of eigVals
    eigVecs = eigVecs[:, sortInd]

    # Project Xtrain, Xtest onto top l eigVecs
    Xtrain_lda = Xtrain.dot(eigVecs[:, :l])
    Xtest_lda = Xtest.dot(eigVecs[:, :l])

    # Plot LDA Projections
    if plot:
        dotColors = ['red', 'orange', 'yellow', 'green', 'blue', 'pink', 'purple', 'lime', 'maroon', 'cyan']
        plt.figure(figsize=(7.5, 7.5))
        for i in range(10):
            plt.scatter(Xtrain_lda[Ytrain==i, 0], Xtrain_lda[Ytrain==i, 1], color=dotColors[i], label=str(i), alpha=0.4)
        plt.legend()
        plt.title('LDA Projection on Training Data')
        plt.xlabel('LDA Dim 1')
        plt.ylabel('LDA Dim 2')
        plt.show()

        plt.figure(figsize=(7.5, 7.5))
        for i in range(10):
            plt.scatter(Xtest_lda[Ytest==i, 0], Xtest_lda[Ytest==i, 1], color=dotColors[i], label=str(i), alpha=0.4)
        plt.legend()
        plt.title('LDA Projection on Test Data')
        plt.xlabel('LDA Dim 1')
        plt.ylabel('LDA Dim 2')
        plt.show()

    return Xtrain_lda, Xtest_lda

