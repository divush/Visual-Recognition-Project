import cv2, os
import numpy as np
from matplotlib import pyplot as plt
import scipy, pickle
from sklearn.cluster import KMeans
import time

lblfile=open('labels', 'r')
dat = lblfile.read()
labels = dat.split('\n')
labels = [x for x in labels if x != '']
lblfile.close()

# Check - the following should give 84!
# print(len(labels))

flist=[]
for label in labels:
	path = 'Dataset/'+label+'/'
	fn = os.listdir(path)
	for i in range(len(fn)):
		fn[i] = path+fn[i]
		flist.append(fn[i])

# At this point, flist contains names of all files, 
# e.g. 'Dataset/white_rain_sensations_ocean_mist_hydrating_body_wash/N1_354.jpg'
# Can do name.split('/')

tr_names, test_names = [], []
train_mask = []
i=0
for fname in flist:
	# Use 70 images for train, rest for test!
	if i<=70:
		tr_names.append(fname)
		train_mask.append(1)
	else:
		test_names.append(fname)
		train_mask.append(0)
	i = i + 1
	if i==72:
		i=0

descriptors=[]
desc_by_file = {}
# For each image in the training set!
# Extract descriptors and store in inverted dictionary.

count = 1
for name in tr_names:
	print("Looking at image"+(str(name)))
	img = cv2.imread(name, 0)

	# Compute SIFT descriptors
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img,None)
	# print(type(des))
	desc_by_file[name] = des
	# print(len(descriptors))
	descriptors.extend(des)
	# for x in des:
	# 	if np.any(descriptors == x) == True:
	# 		continue
	# 	else:
	# 		descriptors.extend(x)

	# DEBUGGING :
	count = count + 1
	if count > 10:
		break

# print(type(descriptors))
# print(len(descriptors))
# descriptors = list(set(descriptors))
# desc_dict = {}
# for i in range(descriptors):
# 	desc_dict[i] = descriptors[i]
# count = len(tr_names)
count = 10
print("On to K-Means!\n")
# Train k-means classifier
kmeans = KMeans(n_clusters=count, max_iter=100).fit(descriptors)
print("K-Means completed!")

ifl = {}
cluster_labels = kmeans.labels_
# assert(1<0)
temp_count = 0
# Now make the inverted file list. Essentially store the labels associated with each point
for name in tr_names:
	if temp_count >= 4:
		break
	temp_count = temp_count + 1
	print("Looking at image"+(str(name)))
	img = cv2.imread(name, 0)

	# Compute SIFT descriptors
	# print("computing descriptors")
	# sift = cv2.xfeatures2d.SIFT_create()
	# kp, des = sift.detectAndCompute(img,None)
	# print("done computing")
	hist = [0]*count
	# print(len(des))
	# print("histogram building")
	# this step is taking time!
	des = desc_by_file[name]
	ticks_old = time.time()
	for x in des:
		# Find index of x in pooled descriptor list
		# ind = np.where(descriptors == x)
		# print(ind[0])
		# print("Finding....")
		ind = [np.array_equal(x, t) for t in descriptors].index(True)
		# print("Found.....")
		# print(type(ind))
		# if(len(ind) > 1):
		# 	print("Error!")
		# ind = descriptors.index(x)
		# Find the "word" which this belongs to.
		cluster_x = cluster_labels[ind]
		# Increase frequency of the word in the training image
		hist[cluster_x] = hist[cluster_x] + 1
	ticks_new = time.time()
	print("time taken = " + str(ticks_new - ticks_old))
	# Image representation is a histogram!
	ifl[name] = hist
	# print("done copying!")


# At this point, I have the inverted file list which tells me the set and number of each word in a file. Eg.
# ifl['img001'] = [1, 2, 3, 2, 5, 4, 4]
# means that the image 'img001' has 7 words, with two 2's, two 4's and one of each 1, 3, 5
print("Onto testing!\n")
fd = open('answers', 'w')
for name in test_names:
	print("Testing image " + str(name))
	img = cv2.imread(name, 0)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img,None)

	# Now comes the matching part! First get the cluster centeres corresponding to each descriptor!
	hist = [0]*count
	test_labels = kmeans.predict(des)
	
	# Now that I have the labels, I can assign make a histogram to represent this image!
	# This histogram is the representation of my image w.r.t. my vocabulary
	for x in test_labels:
		hist[x] = hist[x] + 1
	# print(hist)
	# Reference http://vision.cs.utexas.edu/378h-fall2015/slides/lecture17.pdf
	# The image similiarity score of query (q) and image (i) is <q, i>/(norm(i))
	similiarity = {}
	temp = list(ifl.keys())[0:4]
	for image in temp:
		score = np.dot(hist, ifl[image])
		norm = np.dot(ifl[image], ifl[image])
		similiarity[str(image)] = round((float(score)/norm), 3)

	# Reverse sort by similiarity index!
	d = sorted(similiarity.items(), key=lambda x: x[1], reverse=True)
	print(d)
	towrite = str(name) + " : "
	for i in range(0,3):
		towrite = towrite + " " + str(d[i][0])
	towrite = towrite + "\n"
	fd.write(towrite)

fd.close()















