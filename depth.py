#-------------------------------------------------------------------------------
#	VISUAL RECOGNITION PROJECT FOR DEPTH ESTIMATION
#			BY
#		  DIVYANSHU SHENDE
#			AND
#		    RAHUL TUDU
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import random
from collections import deque
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Merge
from keras.optimizers import SGD, Adam
from keras import backend as K
import tensorflow as tf
import json
import glob
from PIL import Image
import argparse
#-------------------------------------------------
# FOLDER_PATH = PATH TO IMAGE FOLDER
image_rows = 576
image_cols = 172
fin_rows = 142
fin_cols = 41
lamda = 0.5
n = 142*41
memory = 200
BATCH = 32
image_path = "training/image_2"
label_path = "training/viz_flow_occ"
#--------------------------------------------------
# defining custom loss function
# which takes input the predicted and actual depth values
# takes their logarithms and do further processing
def loss_function(_true, _pred):
	_true = K.log(_true)
	_pred = K.log(_pred)
	first = np.subtract(_true, _pred)
	second = np.subtract(_pred, _true)
	first = np.square(first)
	f_sum = np.sum(first)
	s_sum = np.sum(second)
	s_sum = s_sum * s_sum;
	f_sum = f_sum/n
	s_sum = lamda * (s_sum/(n * n))
	return (f_sum - s_sum)

# builds a model of the CNN network by concatenating a coarse and a fine network
# uses custom loss function defined
# returns the model

def build_coarse_model():
	print("Now we build model\n")
	first = Sequential()
	first.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', border_mode='same',  input_shape=(image_rows, image_cols, 1)))
	first.add(MaxPooling2D(pool_size=(2, 2)))
	first.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'))
	first.add(MaxPooling2D(pool_size=(2, 2)))
	first.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	first.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
	first.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu'))
	first.add(Flatten())
	first.add(Dense(5822))
	first.add(Activation('relu'))
	first.add(Reshape((142,41, 1)))

	second = Sequential()
	second.add(Conv2D(63, (9, 9), strides=(2, 2), activation='relu', border_mode='valid', input_shape=(image_rows, image_cols, 1)))
	second.add(MaxPooling2D(pool_size=(2, 2)))
	model = Sequential()
	model.add(Merge([first, second], mode = 'concat'))
	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(Conv2D(1, (5, 5), activation='relu'))
	adam = Adam(lr=1e-4)
	model.compile(loss=loss_function, optimizer=adam)
	print("Modelling has finished\n")
	return model

# takes  the model as input
# and predicts the output for an image

def test(model):
	model.load_weights("model.h5")
	sgd = SGD(lr=0.01)
	model.compile(loss=loss_function, optimizer=sgd)
	#ima = path to an input image
	q = model.predict(ima)
	im = Image.fromarray(q)
	img.save('outputs/myphoto.jpg', 'JPEG')

def create_dataset():
	pairs = deque()
	image_set = []
	label_set = []
	for filename in glob.glob(image_path + "/*.png"):
		im = Image.open(filename)
		im = im.convert('1')
		im = im.resize((image_cols, image_rows), Image.ANTIALIAS)
		im.load()
		data = np.asarray( im, dtype="int32")
		image_set.append(data)
	
	for filename in glob.glob(label_path + "/*.png"):
		im = Image.open(filename)
		im = im.convert('1')
		im = im.resize((fin_cols, fin_rows), Image.ANTIALIAS)
		im.load()
		data = np.asarray( im, dtype="int32")
		label_set.append(data)
	for i in range(0,200):
		pairs.append((image_set[i], label_set[i]))
	print("Dataset created !")
	return pairs

# decides whether to train or test
# calls a function to generate a model
# training is done on a	set of 32 images taken radomly

def __run(args):
	model = build_coarse_model()	
	pairs = create_dataset()
	if(args['mode'] == 'Train'):
		for it in range(0,1000):
			print("Batch number: ", it, "; ")
			minibatch = random.sample(pairs, BATCH)
			inputs = np.zeros((BATCH, image_rows, image_cols))
			outputs = np.zeros((BATCH, fin_rows, fin_cols))
			for i in range(0, len(minibatch)):
				inputs[i:i+1] = minibatch[i][0].reshape(1, 1, image_rows, image_cols)
				outputs[i:i+1] = minibatch[i][1].reshape(1, 1, fin_rows, fin_cols)
				prediction = model.predict([minibatch[i][0].reshape(1, image_rows, image_cols, 1), minibatch[i][0].reshape(1, image_rows, image_cols, 1)])
			loss = model.train_on_batch([inputs, inputs], outputs)
			print("Loss: ", loss, "\n" )
			if (it % 100 == 0):
				print("Saving model !\n")
				model.save_weights("model.h5", overwrite=True)
				with open("model.json", "w") as outfile:
					json.dump(model.to_json(), outfile)
	else:
		test(model)
# the main function

def main():
	parser = argparse.ArgumentParser(description='Depth prediction')
	parser.add_argument('-m','--mode', help='Train / Run', required=True)
	args = vars(parser.parse_args())
	__run(args)

if __name__ == '__main__':
	main()
