import numpy as np
import matplotlib.pyplot as plt
from kernPercGD import kernPercGD, polyKern


### PART A:
print("Running part A: ")
# Init data
np.random.seed(1)
r1 = np.sqrt(np.random.rand(100, 1)) # Radius
t1 = 2*np.pi*np.random.rand(100, 1) # Angle
data1 = np.hstack((r1*np.cos(t1), r1*np.sin(t1))) # Points
# np.random.seed(2)
r2 = np.sqrt(3*np.random.rand(100, 1)+2) # Radius
t2 = 2*np.pi*np.random.rand(100, 1) # Angle
data2 = np.hstack((r2*np.cos(t2), r2*np.sin(t2))) # Points

# Combine data and labels
data3 = np.vstack((data1, data2))
labels = np.ones((200, 1))
labels[0:100, :] = -1


# Train
d=3 # Degree of polynomial
alpha, b = kernPercGD(data3, labels, d)

# Test
preds=[]
for t in range(len(data3)): # for each sample
    net=0
    for s in range(len(alpha)):
        net += alpha[s] * labels[s] * polyKern(data3[s], data3[t], d)
    preds.append(np.sign(net))
err = np.sum(preds != labels) / len(labels)
print(f"Part A error rate: {err}")


# Graph
gridx = np.arange(np.min(data3[:, 0]),np.max(data3[:, 0]), .01)
gridy = np.arange(np.min(data3[:, 1]),np.max(data3[:, 1]), .01)

gridxx, gridyy = np.meshgrid(gridx, gridy)
grid = np.vstack((gridxx.ravel(), gridyy.ravel())).T

decisionBoundaryA = np.zeros(grid.shape[0])
print("Calculating decision boundary & graphing (may take awhile)")
print(f"Num grid outputs to calculate: {grid.shape[0]}")
for i in range(grid.shape[0]):
    point = grid[i]
    net = 0
    for j in range(len(alpha)):
        net += alpha[j] * labels[j] * polyKern(data3[j], point, d)
    decisionBoundaryA[i] = np.sign(net)

decisionBoundaryA = np.reshape(decisionBoundaryA, np.shape(gridxx))


# prepare to part A and decision boundary from part B on same plot
fig, ax = plt.subplots()
ax.contour(gridx, gridy, decisionBoundaryA, 0, colors='green')
ax.scatter(data3[np.where(labels==1)[0],0], data3[np.where(labels==1)[0],1], c='r')
ax.scatter(data3[np.where(labels==-1)[0],0], data3[np.where(labels==-1)[0],1], c='b')
# plt.show()
### uncommenting the above line results in plotting A alone



# PART B:
from sklearn.svm import SVC
# Higher C = tighter margin around "inner" data, lower = larger margin (and more misclassifications for the "outer" data)
# SVM with polynomial kernel
C = 3  # margin penalty param
print(f"Running part B: SVC with C={C}")
clf = SVC(C=C, kernel='poly', degree=2)
clf.fit(data3, labels.flatten())

# Plot decision bound
gridxB, gridyB = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
decisionBoundaryB = clf.decision_function(np.c_[gridxB.ravel(), gridyB.ravel()]) # flatten
decisionBoundaryB = decisionBoundaryB.reshape(gridxB.shape)
plt.contour(gridxB, gridyB, decisionBoundaryB, 0, colors='purple')

plt.show()

# Get & display error rate
preds = clf.predict(data3)
err = np.sum(preds != labels.flatten()) / len(labels)
print("Part B: SVC'd Error rate:", err)




### PART C:

# Part C.1: Optdigits 49
print("Running part C for 49...")
train_data = np.genfromtxt('optdigits49_train.txt', delimiter=',')
train_X = train_data[:,:-1] 
train_y = train_data[:,-1] 
test_data = np.genfromtxt('optdigits49_test.txt', delimiter=',')
test_X = test_data[:,:-1]   
test_y = test_data[:,-1]  

# Train
d = 3 # Degree of polynomial
alpha, b = kernPercGD(train_X, train_y, d)

# Test (testing data)
preds=[]
for t in range(len(test_X)): # for each sample
    net=0
    for s in range(len(alpha)):
        net += alpha[s] * train_y[s] * polyKern(train_X[s], test_X[t], d)
    preds.append(np.sign(net))
err = np.sum(preds != test_y) / len(test_y)
print(f"Part C: 49 error rate (testing data): {err}")

# Test (training data)
preds=[]
for t in range(len(train_X)): # for each sample
    net=0
    for s in range(len(alpha)):
        net += alpha[s] * train_y[s] * polyKern(train_X[s], train_X[t], d)
    preds.append(np.sign(net))
err = np.sum(preds != train_y) / len(train_y)
print(f"Part C: 49 error rate (training data): {err}")



# Part C.2: Optdigits 79
train_data = np.genfromtxt('optdigits79_train.txt', delimiter=',')
train_X = train_data[:,:-1] 
train_y = train_data[:,-1] 
test_data = np.genfromtxt('optdigits79_test.txt', delimiter=',')
test_X = test_data[:,:-1]   
test_y = test_data[:,-1]  

# Train
d = 3 # Degree of polynomial
alpha, b = kernPercGD(train_X, train_y, d)

# Test (testing data)
preds=[]
for t in range(len(test_X)): # for each sample
    net=0
    for s in range(len(alpha)):
        net += alpha[s] * train_y[s] * polyKern(train_X[s], test_X[t], d)
    preds.append(np.sign(net))
err = np.sum(preds != test_y) / len(test_y)
print(f"Part C: 79 error rate (testing data): {err}")
# Test (training data)
preds=[]
for t in range(len(train_X)): # for each sample
    net=0
    for s in range(len(alpha)):
        net += alpha[s] * train_y[s] * polyKern(train_X[s], train_X[t], d)
    preds.append(np.sign(net))
err = np.sum(preds != train_y) / len(train_y)
print(f"Part C: 79 error rate (training data): {err}")

