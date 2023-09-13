import keras
from keras import backend as K
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from datetime import datetime
from LReLU import LReLU
from ReadNormalizedOptdigitsDataset import ReadNormalizedOptdigitsDataset

# Params
batch_size = 64
epochs = 20
num_classes = 10
img_rows, img_cols = 8, 8  # input image dimensions

# Load & normalize data
x_train, y_train, x_valid, y_valid, x_test, y_test = ReadNormalizedOptdigitsDataset('../Data/optdigits_train.txt', '../Data/optdigits_valid.txt', '../Data/optdigits_test.txt')

# Convert data format to channel last format
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Declare model
model = Sequential()


### STRUCTURE 1 ###
## 2D conv layer w/ linear activation followed by a batchnorm, LReLU to introduce nonlinearity, 
## flattening, a dense layer with 10 units, and a softmax activation layer
model.add(Conv2D(1, kernel_size=(4, 4),
                 activation='linear',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation(LReLU)) # use your leaky relu function
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))


### STRUCTURE 2 ###
# model.add(Conv2D(20, kernel_size=(3, 3),
#                  activation='linear',
#                  input_shape=input_shape,
#                  padding='same'))
# model.add(BatchNormalization())
# model.add(Activation(LReLU))
#
# model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# model.add(Conv2D(32, kernel_size=(3, 3), activation='linear'))
# model.add(BatchNormalization())
# model.add(Activation(LReLU))
#
# model.add(Flatten())
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))


model.summary()



# Store results & use visualize training loss
logdir = "./logs/myResults" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Compile model using crossentropy loss function, adadelta optimizer, and accuracy as a metric
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[tensorboard_callback])

# Evaluate model on test dataset
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])
