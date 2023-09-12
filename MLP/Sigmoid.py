from tensorflow import keras

# Performs sigmoid function on data
def Sigmoid(x):
    activation_vals = keras.backend.sigmoid(x)
    return activation_vals