import cv2, os
import numpy as np
from matplotlib import pyplot as plt

lblfile=open('labels', 'r')
dat = lblfile.read()
labels = dat.split('\n')
labels.remove('')
lblfile.close()

# Check - the following should give 84!
# print(len(labels))

flist=[]
for label in labels:
	path = 'Dataset/'+label+'/'
	fn = os.listdir(path)
	for i in range(len(fn)):
		fn[i] = path+fn[i]
	flist.append(fn)

# At this point, flist contains names of all files, 
# e.g. 'Dataset/white_rain_sensations_ocean_mist_hydrating_body_wash/N1_354.jpg'
# Can do name.split('/')

tr_names, test_names = [], []
for fnames in flist:
	# Use 60 images for train, rest for test!
	tr_names.append(fnames[0:60])
	test_names.append(fnames[60:len(fnames)])
