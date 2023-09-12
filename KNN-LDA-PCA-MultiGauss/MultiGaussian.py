import numpy as np
from scipy.linalg import inv, det


def MultiGaussian(training_data, testing_data, Model):
    # Extract training & testing data
    train = np.loadtxt(training_data, delimiter=",")
    test = np.loadtxt(testing_data, delimiter=",")

    Xtrain, ytrain = train[:, :-1], train[:, -1]
    Xtest, ytest = test[:, :-1], test[:, -1]

    X1 = Xtrain[ytrain == 1]
    X2 = Xtrain[ytrain == 2]
    # Get means, priors
    m1 = np.mean(Xtrain[ytrain == 1], axis=0) 
    m2 = np.mean(Xtrain[ytrain == 2], axis=0)
    unique_elements, counts = np.unique(ytrain, return_counts=True)
    pc1, pc2 = (counts/len(ytrain))

    # Adjust S-matricies based on user's selected model
    if Model==1:
        S1 = np.cov(X1.T)
        S2 = np.cov(X2.T)
    elif Model==2:
        S1 = np.cov(X1.T)
        S2 = S1
    elif Model==3:
        S1 = np.diag(np.var(X1, axis=0)) # aka sigma12
        S2 = np.diag(np.var(X2, axis=0)) # aka sigma22
    else:
        return

    print("Model 1 Data:")
    print(f"M1: {m1}")
    print(f"M2: {m2}")
    print(f"PC1\n{pc1}")
    print(f"PC2\n{pc2}")
    print(f"S1/sigma11\n{S1}")
    print(f"S2/sigma22\n{S2}")

    # Get predictions
    preds = np.zeros_like(ytest)
    for i, x in enumerate(Xtest):
        probC1 = np.log(pc1) - 0.5 * np.log(det(S1)) - 0.5 * np.dot(np.dot((x - m1).T, inv(S1)), (x - m1))
        probC2 = np.log(pc2) - 0.5 * np.log(det(S2)) - 0.5 * np.dot(np.dot((x - m2).T, inv(S2)), (x - m2))
        preds[i] = 1 if probC1 > probC2 else 2
    
    # Get & print error rate for model on data
    errors = preds != ytest
    error = np.mean(errors)
    print(f"Error rate model {Model}, training data {training_data[-5]}, testing data {testing_data[-5]}): {error}")
    return m1, m2, pc1, pc2, S1, S2

# Independant S model:
MultiGaussian('training_data1.txt', 'test_data1.txt', 1)
MultiGaussian('training_data2.txt', 'test_data1.txt', 1)
MultiGaussian('training_data3.txt', 'test_data1.txt', 1)

MultiGaussian('training_data1.txt', 'test_data2.txt', 1)
MultiGaussian('training_data2.txt', 'test_data2.txt', 1)
MultiGaussian('training_data3.txt', 'test_data2.txt', 1)

MultiGaussian('training_data1.txt', 'test_data3.txt', 1)
MultiGaussian('training_data2.txt', 'test_data3.txt', 1)
MultiGaussian('training_data3.txt', 'test_data3.txt', 1)

# S1=S2 Model
MultiGaussian('training_data1.txt', 'test_data1.txt', 2)
MultiGaussian('training_data2.txt', 'test_data1.txt', 2)
MultiGaussian('training_data3.txt', 'test_data1.txt', 2)

MultiGaussian('training_data1.txt', 'test_data2.txt', 2)
MultiGaussian('training_data2.txt', 'test_data2.txt', 2)
MultiGaussian('training_data3.txt', 'test_data2.txt', 2)

MultiGaussian('training_data1.txt', 'test_data3.txt', 2)
MultiGaussian('training_data2.txt', 'test_data3.txt', 2)
MultiGaussian('training_data3.txt', 'test_data3.txt', 2)

# Diagnol S Model:
MultiGaussian('training_data1.txt', 'test_data1.txt', 3)
MultiGaussian('training_data2.txt', 'test_data1.txt', 3)
MultiGaussian('training_data3.txt', 'test_data1.txt', 3)

MultiGaussian('training_data1.txt', 'test_data2.txt', 3)
MultiGaussian('training_data2.txt', 'test_data2.txt', 3)
MultiGaussian('training_data3.txt', 'test_data2.txt', 3)

MultiGaussian('training_data1.txt', 'test_data3.txt', 3)
MultiGaussian('training_data2.txt', 'test_data3.txt', 3)
MultiGaussian('training_data3.txt', 'test_data3.txt', 3)
