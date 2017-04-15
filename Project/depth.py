#-------------------------------------------------------------------------------
#	VISUAL RECOGNITION PROJECT FOR DEPTH ESTIMATION
#			BY
#		DIVYANSHU SHENDE
#			AND
#		RAHUL TUDU
#-------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layer.core import Dense, Dropout, Activation, Flatten
from keras.layer import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import tensorflow as tf
import json
from PIL import image
#-------------------------------------------------
# FOLDER_PATH = PATH TO IMAGE FOLDER
image_rows = 304
image_cols = 228
lamda = 0.5;
n = 100;
#-------------------------------------------------
#
# Logistics of taking inputs.
#

# Implement conv1 layer here.

# defining custom loss function
# which takes input the predicted and actual depth values
# takes their logarithms and do further processing

def loss_function(_true, _pred):
	_true = np.log(_true)
	_pred = np.log(_pred)
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
	first.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(image_rows, image_cols, 1)))
	first.add(MaxPooling2D(pool_size=(2, 2)))
	first.add(Conv2D(256, (5, 5), strides=(1, 1), activation='relu'))
	first.add(MaxPooling2D(pool_size=(2, 2)))
	first.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu'))
	first.add(Conv2D(384, (3, 3), strides=(1, 1), activation='relu'))
	first.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu'))
	first.add(Dense(4096))
	first.add(Activation('relu'))
	first.add(Dense(1))

	second = Sequential()
	second.add(Conv2D(63, (9, 9), strides=(2, 2), activation='relu', input_shape=(image_rows, image_cols, 1)))
	second.add(MaxPooling2D(pool_size=(2, 2)))
	merged = Concatenate([first, second])
	model = Sequential()
	model.add(merged)
	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(Conv2D(1, (5, 5), activation='relu'))
	sgd = SGD(lr=0.01)
	model.compile(loss=loss_function, optimizer=sgd)
	print("Modelling has finished\n")
	return model
<<<<<<< HEAD

# takes  the model, images and their labels as inputs
# saves the weights of the network in model.h5
# evaluates the final score of the training

def train(model, X_train, Y_train):
	model.fit(X_train, Y_train, batch_size=32, epochs=20)
	model.save_weights("model.h5", overwrite=True)
	with open("model.json", "w") as outfile:
		json.dump(model.to_json(), outfile)
	score = model.evaluate(X_train, Y_train)
	print("Score is {}\n".format(score))

# takes  the model as input
# and predicts the output for an image
	
def test(model)
	model.load_weights("model.h5")
	sgd = SGD(lr=0.01)
	model.compile(loss=loss_function, optimizer=sgd)
	#ima = path to an input image
	q = model.predict(ima)
	im = Image.fromarray(q)
	img.save('outputs/myphoto.jpg', 'JPEG')

# processes an image before using it for training
# takes the path of  the image as input	 
def process_image(filename):
	im = Image.open(filename)
	im = im.convert('1') #convert the image to black and white
	im = im.resize((image_cols, image_rows), PIL.Image.ANTIALIAS) # resize the image
	im.load()
	data = np.asarray( im, dtype="int32")
	return data

# dummy function for creating the dataset after processing each image

def create_dataset(path)
	# definition of function to create dataset

# decides whether to train or test
# calls a function to generate a model
def __run(args)
	model = build_coarse_model()
	if(args['mode'] == 'train'):
		images, label = create_dataset(FOLDER_PATH)
		train(model, images, labels)
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
