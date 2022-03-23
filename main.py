import keras
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils, to_categorical
from keras.optimizers import Adam

# load data
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images.reshape(training_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)
input_shape = (28,28,1)

training_img = training_images.astype('float32')
test_img = test_images.astype('float32')

training_img /= 225
test_img /= 225

training_labels = to_categorical(training_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Conv2D(32,(5,5), padding="same", input_shape=(input_shape),activation='relu'))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, kernel_initializer="normal", activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, batch_size=150, verbose=1)

eval = model.evaluate(test_img, test_labels, verbose=1)
print(f"we got {eval[1]} accuracy in our testing set.")
print(f"Our error rate was {eval[0]}.")

model.save('mnist.h5')