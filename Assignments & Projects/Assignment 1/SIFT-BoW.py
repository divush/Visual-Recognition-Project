import cv2, os
import numpy as np
from matplotlib import pyplot as plt
import scipy, pickle
from sklearn.cluster import KMeans

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
ifl = {}
# For each image in the training set!
# Extract descriptors and store in inverted dictionary.
count = 1
for name in tr_names:
	print("Looking at image"+(str(name)))
	img = cv2.imread(name, 0)

	# Compute SIFT descriptors
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img,None)
	print(des.shape)
	print(type(des[0]))
	print(des[0].shape)
	u=des[0].tolist
	print(len(u))
	# print(len(descriptors))
	for x in des:
		# Pool descriptors for all images!
		type(x)
		if x not in descriptors:
			descriptors.append(x)

		# Inverted file list stores file names with descriptors as index!
		ind = descriptors.index(x)
		if ind in ifl:
			ifl[ind].append(name)
		else:
			ifl[ind]=[name]

	# DEBUGGING :
	count = count + 1
	if count > 10:
		break

# count = len(tr_names)
count = 10
print("On to K-Means!\n")
# Train k-means classifier
kmeans = KMeans(n_clusters=count, max_iter=100).fit(descriptors)
print("K-Means completed!")

cluster_imgs = []
for i in range(count):
	cluster_imgs.append([])

cluster_labels = kmeans.labels_
for i in range(len(descriptors)):
	des = descriptors[i]
	lbl = cluster_labels[i]
	img_files = ifl[i]
	for name in img_files:
		if name not in cluster_imgs[lbl]:
			cluster_imgs[lbl].append(name)


test_dict = dict.fromkeys(tr_names, 0)
for name in test_names:
	print("Testing image " + str(name))
	img = cv2.imread(name, 0)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img,None)

	# Now comes the matching part! First get the cluster centeres corresponding to each descriptor!
	test_labels = kmeans.predict(des)
	test_dict = {}
	for lbl in test_labels:
		for img_name in cluster_imgs[lbl]:
			test_dict[img_name] = test_dict[img_name] + 1
	# We now have in a dictionary all the labels


	for x in test_dict:
		if test_dict[x] == 0:
			t = test_dict.pop(x)
	# Now the dictionary contains only those labels that are non-zero
	# What this means is that it contains only those filenames that have non-zero descriptors 
	# matching with the cluster centers

	plt.bar(range(len(test_dict)), test_dict.values(), align='center')
	plt.xticks(range(len(test_dict)), list(test_dict.keys()))
	fig = plt.figure()
	fig.savefig(name)























