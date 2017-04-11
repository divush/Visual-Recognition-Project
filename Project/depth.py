import tensorflow as tf
import numpy as np
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layer.core import Dense, Dropout, Activation, Flatten
from keras.layer import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import tensorflow as tf
#-------------------------------------------------
image_rows = 100
image_cols = 100
#-------------------------------------------------
#
# Logistics of taking inputs.
#

# Implement conv1 layer here.
def build_coarse_model():
	print("Now we build model\n")
	model = Sequential()
	model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(image_rows, image_cols, 1)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (5, 5), strides=(1, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu'))
	model.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu'))
	model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu'))
	model.add(Dense(4096))
	model.add(Activation('relu'))
	model.add(Dense(1))
	sgd = SGD(lr=0.01)
	model.compile(loss='mse',optimizer=sgd)
	print("Modelling has finished\n")
	return model
