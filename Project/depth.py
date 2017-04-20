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
import glob
from PIL import image
#-------------------------------------------------
# FOLDER_PATH = PATH TO IMAGE FOLDER
image_rows = 304
image_cols = 228
#fin_rows = 
#fin_cols = 
lamda = 0.5;
n = 100;
memory = 398
BATCH = 32
image path = ""
label_path = ""
#--------------------------------------------------
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

# a pair of image--label is inserted in a deque

def create_dataset()
	pairs = deque()
	image_set = []
	label_set = []
	for filename in glob.glob(image_path + "/*.png"):
		im = Image.open(filename)
		im = im.convert('1')
		im = im.resize((image_cols, image_rows), PIL.Image.ANTIALIAS) # resize the image
		im.load()
		data = np.asarray( im, dtype="int32")
		image_set.append(data)
	
	for filename in glob.glob(label_path + "/*.png"):
		im = Image.open(filename)
		im = im.convert('1')
		im = im.resize((fin_cols, fin_rows), PIL.Image.ANTIALIAS) # resize the image
		im.load()
		data = np.asarray( im, dtype="int32")
		label_set.append(data)
	for i in range(0,398):
		pairs.append(image_set[i], label_set[i])

	return pairs

# decides whether to train or test
# calls a function to generate a model
# training is done on a	set of 32 images taken radomly

def __run(args)
	model = build_coarse_model()
	
	pairs = create_dataset(PATH)
	if(args['mode'] == 'train'):
		for it in range(0,1000):
			minibatch = random.sample(pairs, BATCH)
			inputs = np.zeros(BATCH, image_rows, image_cols)
			outputs = np.zeros(BATCH, image_rows, image_cols)
			for i in range(0, len(minibatch)):
				inputs[i:i+1] = minibatch[i][0]
				outputs[i:i+1] = minibatch[i][1]
				prediction = model.predict(inputs)
			loss = model.train_on_batch(inputs, outputs)
			print("loss is: {}\n".format(loss))
			if (it % 100 == 0):
				print("Saving model !\n")
				model.save_weights("model.h5", overwrite=True)
				with open("model.json", "w") as outfile:
					json.dump(model.to_json(), outfile)
	else:
		#test(model)
# the main function

def main():
	parser = argparse.ArgumentParser(description='Depth prediction')
	parser.add_argument('-m','--mode', help='Train / Run', required=True)
	args = vars(parser.parse_args())
	__run(args)

if __name__ == '__main__':
	main()
