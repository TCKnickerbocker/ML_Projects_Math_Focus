import numpy as np
from MLPtrain import MLPtrain
from MLPtest import MLPtest

from matplotlib import pyplot as plt


# Declare error arrays, hidden unit numbers to test
trainErrs, validErrs, Ws, Vs = [], [], [], []
Hs = [3, 6, 9, 12, 15, 18]

# Test different numbers of hidden units
for i, num_hid in enumerate(Hs):
    # call model update based on training data, record best val accuracy
    print(f"Running with H={num_hid}")
    trainAcc, validAcc, Z, W, V = MLPtrain("../Data/optdigits_train.txt", "../Data/optdigits_test.txt", num_hid, 10)
    trainErrs.append(trainAcc)
    validErrs.append(validAcc)
    Ws.append(W)
    Vs.append(V)
# Plot results
plt.plot(Hs, trainErrs)
plt.scatter(Hs, trainErrs, c='r', label='data')
plt.xlabel("Number of Hidden Units")
plt.ylabel("Training Error Rate (%)")
plt.show()
plt.plot(Hs, validErrs)
plt.scatter(Hs, validErrs, c='r', label='data')
plt.xlabel("Number of Hidden Units")
plt.ylabel("Validation Error Rate (%)")
plt.show()

# find the model with the best validation accuracy, get values
bestInd = np.argmin(validErrs)
bestH = Hs[bestInd]
bestW = Ws[bestInd]
bestV = Vs[bestInd]
print(f"Best number of hidden units (we will test using this number): {bestH}")

# Run best model on test data
predictions = MLPtest("../Data/optdigits_train.txt", bestW, bestV)

