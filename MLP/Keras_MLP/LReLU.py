from tensorflow import keras

# Performs linear ReLU function on data
def LReLU(x):
    return keras.backend.maximum(.01 * x, x)
