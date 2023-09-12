import numpy as np
import matplotlib.pyplot as plt


def kernPercGD(train_data, train_label, poly_degree):
    alpha = np.zeros(len(train_data))
    b = 0
    for i in range(100):  # Setting cap to 100 for iterations
        converged=True
        for t in range(len(train_data)): # for each sample
            net=0
            for s in range(len(train_data)):
                net += alpha[s] * train_label[s] * polyKern(train_data[s], train_data[t], poly_degree)
            net *= train_label[t]
            if net <= 0:
                alpha[t] += 1
                b += train_label[t]
                converged=False
        if converged:
            print(f"Algorithm converged after {i+1} total iterations")
            break
    return alpha, b

def polyKern(s, t, d): # Helper func
    return (1 + np.dot(s, t)) ** d
