import numpy as np

def Bayes_Learning(training_data , validation_data):
    Xtrain, ytrain = training_data[:, :-1], training_data[:, -1] # extract & sever features, classes from data
    Xvalid, yvalid = validation_data[:, :-1], validation_data[:, -1]

    p1 = np.mean(Xtrain[ytrain == 1], axis=0)  # <--- get MLE for p1 (class 1), p2 (class 2)
    p2 = np.mean(Xtrain[ytrain == 2], axis=0)
    p1[p1 == 0] = 1e-10  # <---replace zeroes w/ small constant
    p2[p2 == 0] = 1e-10

    priors = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6]  # <---all priors (sigma) tested
    errors = [] # <--- error rates (for each prior)

    for prior in priors:
        c1 = np.prod((p1 ** Xvalid) * ((1 - p1) ** (1 - Xvalid)), axis=1)  # <--- product of bernouli for every feature set belonging to class 1
        c2 = np.prod((p2 ** Xvalid) * ((1 - p2) ** (1 - Xvalid)), axis=1)  # <--- product of bernouli for every feature set belonging to class 2

        c1Posterior = (1 - np.exp(-prior) ) * c1  # since math.e is unallowed
        c2Posterior = np.exp(-prior) * c2 
        
        preds = np.where(c1Posterior > c2Posterior, 1, 2)  # predictions array
        errorRate = np.mean(preds != yvalid)  # error formula in numpy
        errors.append(errorRate) # append error rate for prior

    idx = np.argmin(errors)
    bestPrior = priors[idx]  # best prior
    pc1 = 1 - np.exp(-bestPrior) # calculate pc1, pc2 (1-e^sig)
    pc2 = np.exp(-bestPrior)

    print('Prior    Error rate')  # display prior : error rate for each prior
    for i, prior in enumerate(priors):
        print(f'{prior:.5f}   {errors[i]:.5f}')

    return p1, p2, pc1, pc2 
