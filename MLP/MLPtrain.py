import numpy as np
epsilon = 1e-12

# Trains a Multi-Layer Perceptron on data with num_hid hidden layers and num_out output values
def MLPtrain(trainPath, validPath, num_hid, num_out):
    # Init data
    train_x, train_y = load(trainPath)
    valid_x, valid_y = load(validPath)

    train_x, mean, std = normalize(train_x)
    valid_x = normalize(valid_x, mean, std)
    # Init weights
    weight_1 = np.random.random([64, num_hid]) / 100
    hiddenBias = np.random.random([1, num_hid]) / 100
    weight_2 = np.random.random([num_hid, num_out]) / 100
    inputBias = np.random.random([1, num_out]) / 100

    old_train_y = train_y
    # turn training labels into one-hot vectors
    train_y = oneHot(train_y)
    # Init
    lr = 10e-5
    epoch = 0
    best_valid_acc = 0
    valid_acc_hist = []
    converged = False

    while epoch <= 10000 and not converged:
        # Forward pass
        hidden_input = train_x.dot(weight_1) + hiddenBias
        hidden_output = computeHidden(train_x, weight_1, hiddenBias)

        output_layer_input = hidden_output.dot(weight_2) + inputBias
        y_pred = softmax(output_layer_input)

        # Backward pass (aka backpropagation)
        # Get gradients for each param
        gradInputWeightLoss = np.multiply(lossGrad(train_y, y_pred),softmaxGrad(output_layer_input))
        gradWeight2 = hidden_output.T.dot(gradInputWeightLoss)
        gradBias2 = np.sum(gradInputWeightLoss, axis=0, keepdims=True)

        gradHiddenInputWeight = np.multiply(gradInputWeightLoss.dot(weight_2.T),LReLUGrad(hidden_input))
        gradWeight1 = train_x.T.dot(gradHiddenInputWeight)
        gradBias1 = np.sum(gradHiddenInputWeight,axis=0, keepdims=True)

        # update params based on gradients, step size
        weight_2 -= lr * gradWeight2
        inputBias -= lr * gradBias2
        weight_1 -= lr * gradWeight1
        hiddenBias -= lr * gradBias1

        # get validation acc
        predictions = predict(valid_x, weight_1, hiddenBias, weight_2, inputBias)
        cur_valid_acc = (predictions.reshape(-1) == valid_y.reshape(-1)).sum() / len(valid_x)
        # compare the current validation accuracy, if cur_valid_acc > best_valid_acc, we will increase count by it
        if cur_valid_acc > best_valid_acc:
            best_valid_acc = cur_valid_acc
            epoch = 0
            lr = 5e-4
        else:
            epoch += 1

        # convergence check, adjust learning rate
        if len(valid_acc_hist) == 80:
            if abs(np.mean(valid_acc_hist) - cur_valid_acc) <= lr:
                converged = True
            valid_acc_hist = []
            lr /= 10

        valid_acc_hist.append(cur_valid_acc)

    # Get training error rate w/ this 
    trainPred = predict(train_x, weight_1, hiddenBias, weight_2, inputBias)
    trainAcc = (trainPred.reshape(-1) == old_train_y.reshape(-1)).sum() / len(train_x)
    trainErr = 100*(1-trainAcc)
    validErr = 100*(1-best_valid_acc)
    hidden_input = train_x.dot(weight_1) + hiddenBias
    Z = computeHidden(train_x, weight_1, hiddenBias)
    W = np.concatenate((weight_1, hiddenBias), axis=0)
    V = np.concatenate((weight_2, inputBias), axis=0)
    print("TrainErr, ValidErr: {:.2f}%, {:.2f}%".format(trainErr, validErr))
    # Return train & validation accuracies as well for plotting
    return trainErr, validErr, Z, W, V



## Helper functions
def LReLU(x):
    return np.maximum(x, x*.01)

def softmax(x):
    xi = np.exp(x - np.max(x, axis=-1, keepdims=True))
    xj = np.sum(xi, axis=-1, keepdims=True)
    return xi / xj

def softmaxGrad(x):
    p = softmax(x)
    return p * (1 - p)

def LReLUGrad(x):
    return np.clip(x,epsilon,1-epsilon)

def lossGrad(y, p):
    return -(y / p) + (1 - y) / (1 - p)

# Gets weight prob of each class, convert to most likely label
def predict(x, weight_1, hiddenBias, weight_2, inputBias):
    hidden_output = computeHidden(x, weight_1, hiddenBias)
    output_layer_input = hidden_output.dot(weight_2) + inputBias
    pred = softmax(output_layer_input)
    return np.argmax(pred, axis=1)

# Gets hidden layer features after applying activation func
def computeHidden(x, w1, hiddenBias):
    hidden_input = x.dot(w1) + hiddenBias
    return LReLU(hidden_input)

# Normalizes data
def normalize(data, m=None, std=None):
    if (m is None and std is None):
        # get mean & stdev
        m = np.mean(data, axis=0)
        std = np.std(data + epsilon, axis=0)
        data = np.subtract(data, m)
        data = np.divide(data, std + epsilon)
        return data, m, std
    else:
        # Normalize data -> 0 mean & var
        data = np.subtract(data, m)
        data = np.divide(data, std + epsilon)
        return data
    
# Loads data from file
def load(path):
    data = np.genfromtxt(path, delimiter=",")
    x = data[:, :-1]
    y = data[:, -1].astype('int')
    return x, y

# Performs one-hot vector encoding on our outputs
def oneHot(y):
    # convert y to oneHot for loss computation
    oneHotY = np.zeros([len(y), 10])
    oneHotY[np.arange(y.shape[0]), y] = 1
    return oneHotY