import numpy as np

def Bayes_Testing(test_data, p1, p2, pc1, pc2):
    Xtest, ytest = test_data[:, :-1], test_data[:, -1]  # <--- extract features, class labels

    c1Like = np.prod((p1 ** Xtest) * ((1 - p1) ** (1 - Xtest)), axis=1)  # <--- product of bernouli for every feature set belonging to class 1
    c2Like = np.prod((p2 ** Xtest) * ((1 - p2) ** (1 - Xtest)), axis=1)  # <--- product of bernouli for every feature set belonging to class 2

    c1Posterior = (1 - np.exp(-pc1) ) * c2Like  # <--- get posteriors for each class
    c2Posterior = np.exp(-pc2) * c1Like 
    
    preds = np.where(c1Posterior > c2Posterior, 1, 2)  # <---- make array of class predictions
    errorRate = np.mean(preds != ytest)

    print(f'Error rate on test set: {errorRate}')
