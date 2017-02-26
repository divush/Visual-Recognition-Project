import cv2, os
import numpy as np
from matplotlib import pyplot as plt
import scipy, pickle
from sklearn.cluster import KMeans, MiniBatchKMeans
import time
import operator

total_start = time.time()
lblfile=open('labels', 'r')
dat = lblfile.read()
labels = dat.split('\n')
labels = [x for x in labels if x != '']
lblfile.close()

# Check - the following should give 84!
# print(len(labels))
leaf_dict = {}
n_child = 4
n_levels = 5
class node(object):
	"A tree node in the vocabulary tree."
	def __init__(self):
		self.name = 0
		self.children = {}
		self.centers = []
		self.imglist = {}
		self.leaf = 0
		self.weight = 0

img_score = {}

def find_node(root, name):
	if root.name == name:
		return root
	flag = 0
	for x in root.children.values():
		if x.name == name:
			flag = 1
			break
		else:
			p = find_node(x, name)
			if p != None:
				return p
	if flag == 1:
		return x

def base_change(name):
	base_nc = name[4:]
	base_nc = base_nc[::-1]
	base10 = 0
	cnt = 0
	for x in base_nc:
		cnt = cnt + 1
		base10 = base10 + (n_child**cnt)*int(x)
	return base10

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
# n_images = len(tr_names)
train_mask = []
i=0
for fname in flist:
	# Use 63 images for train, rest for test!
	if i<=63:
		tr_names.append(fname)
		train_mask.append(1)
	else:
		test_names.append(fname)
		train_mask.append(0)
	i = i + 1
	if i==72:
		i=0

descriptors=[]
n_images = len(tr_names)
# desc_by_file = {}
# For each image in the training set!
# Extract descriptors and store in inverted dictionary.
cnt = 1
start = time.time()
for name in tr_names:
	# print("Looking at image"+(str(name)))
	img = cv2.imread(name, 0)

	# Compute SIFT descriptors
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img, None)
	# print(type(des))
	# print(len(descriptors))
	for x in des:
		descriptors.append(x)
	# for x in des:
	# 	if np.any(descriptors == x) == True:
	# 		continue
	# 	else:
	# 		descriptors.extend(x)

	# DEBUGGING :
	cnt = cnt + 1
	if cnt%100 == 0:
		# break
		print(str(cnt) + " images done!")
# print(type(descriptors))
# print(len(descriptors))
# Convert to array to avoid stupid numpy comparison drama.
end = time.time()
print("Done getting descriptors for " + str(len(tr_names)) +" images in time " + str(end - start))
descriptors = np.array(descriptors)
tree = node()
# desc_dict = {}
# for i in range(descriptors):
# 	desc_dict[i] = descriptors[i]
count = len(tr_names)
# count = 100
# fd = open('treedata2', 'w')
def build_tree(tree_node, name, descriptor_indexes, level):
	# start = time.time()
	tree_node.name = name
	tree_node.name  = base_change(name)
	p = []
	[p.append([]) for i in range(0,n_child)]
	# I have reached the maximum I wanted to branch! I now label this node as a leaf.
	if level == n_levels:
		# print("Going to declare a leaf!")
		tree_node.leaf = 1
		if tree_node.name not in leaf_dict.keys():
			adding = len(leaf_dict)
			leaf_dict[tree_node.name] = adding
		return
	# I want a minimum of 40 descriptors per node of my tree!
	# So if I have <40, this node is my leaf!
	if len(descriptor_indexes) < 40:
		# print("Going to declare a leaf!")
		tree_node.leaf = 1
		if tree_node.name not in leaf_dict.keys():
			adding = len(leaf_dict)
			leaf_dict[tree_node.name] = adding
		return
		
	# print(tree_node.children[x].leaf)
	# cluster my descriptors into n_child many clusters
	points = [descriptors[index] for index in descriptor_indexes]
	kmeans =  MiniBatchKMeans(n_clusters=n_child, max_iter=100).fit(points)	#KMeans clustering
	tree_node.centers = kmeans.cluster_centers_ 
	tree_node.centers = np.array(tree_node.centers)
	for x in range(0, len(points)):
		child_index = kmeans.labels_[x]
		# Note that we store the descriptor index, not the descriptor itself!
		p[child_index].append(x)
	# we have run it once, now we need to initialize for next iteration!
	for x in range(0, n_child):
		child_name = name + str(int(x)+1)
		tree_node.children[x] = node()
		build_tree(tree_node.children[x], child_name, p[x], level+1)
	# str(tree_node.name) + "\t\t\t" +
	# towrite = str(tree_node.leaf) + "\t" + str(len(tree_node.children)) + str(len(tree_node.centers)) + "\n" 
	# # towrite = towrite + "\t" + str(len(tree_node.centers)) + "\t" + str(level) + "\t" + str(len(descriptor_indexes)) + "\n"
	# fd.write(towrite)
	# end = time.time()
	# print("Ran build_tree() for level " + str(level) + " for node " + str(name) + " for time " + str(end - start))

initial_desc = list(range(0, len(descriptors)))
print("On to K-Means!\n")
start = time.time()
build_tree(tree, 'root', initial_desc, 0)
# fd.close()
end = time.time()
print("K-Means completed in time " + str(end - start))
# We don't need the descriptors vector anymore! It will only take up memory and be a nuisance!
# Unfortunately, python has no sure shot way of freeing the memory, this is the best we can do.

del descriptors

# At this point, we have the vocabulary tree with us, now we train by propogating 
# the test image descriptors down the tree! We store the image's index in the
# nodes it goes through!
# fd = open('treedata1', 'w')
def propogate(desc, im_name, tree_node):
	# Calculate distance between descriptor and each of my childrens cluster center.
	# Propogate image through child with cluster center with minimum euclidean distance!
	if tree_node.centers == []:
		tree_node.leaf = 1
	else:
		dist = []
		for x in tree_node.centers:
			temp = np.linalg.norm(x - desc)
			dist.append(temp)
		min_dist = min(dist)
		ind_dist = dist.index(min_dist)
		# Propogate through that!
		propogate(desc, im_name, tree_node.children[ind_dist])

	# If I am at a leaf, I update the inverted document index and the weight of the node!
	# The weight is given by w_i = ln (N/ N_i)
	# Note : I update the weight in the else part because I add a new index there.
	if tree_node.leaf == 1:
		if im_name in tree_node.imglist.keys():
			tree_node.imglist[im_name] = tree_node.imglist[im_name] + 1
			# tree_node.weight = np.log(n_images/len(tree_node.imglist))
			# if len(tree_node.imglist) == 0:
			# 	p = find_node(tree_node)
			# 	if p==None:
			# 		print("None hai!")
			# 	else:
			# 		print("None nahi hai!")
			# else:
			# 	print("Zero nahi hai!")
			# 	if n_images == 0:
			# 		print("ni = 0")
			# 	else:
			# 		print("What the fuck?")
		if im_name not in tree_node.imglist.keys():
			tree_node.imglist[im_name] = 1
			tree_node.weight = np.log(n_images/len(tree_node.imglist))
			# if len(tree_node.imglist) == 0:
			# 	p = find_node(tree_node)
			# 	if p==None:
			# 		print("None hai!")
			# 	else:
			# 		print("None nahi hai!")
			# else:
			# 	print("Zero nahi hai!")
			# 	if n_images == 0:
			# 		print("ni = 0")
			# 	else:
			# 		print("What the fuck?")
	return 
	# towrite = str(tree_node.name) + "\t\t\t" + str(tree_node.leaf) + "\t" +  str(len(tree_node.children)) + str(len(tree_node.centers)) + "\n" 
	# towrite = towrite + "\t" + str(len(tree_node.centers)) + "\t" + str(level) + "\t" + str(len(descriptor_indexes)) + "\n"
	# fd.write(towrite)
	# flag = 0
	# ch = 0
	# assign = -1
	
	# for child in tree_node.children.values():
	# 	if child.leaf == 1:
	# 		continue
	# 	if flag==0:
	# 		d = np.linalg.norm(child.centers[0] - desc)
	# 	for x in child.centers:
	# 		temp = np.linalg.norm(x - desc) 
	# 		# Note : default norm is l2 norm, which is Euclidean distance
	# 		if temp < d:
	# 			assign = ch
	# 	# On to other child!
	# 	ch = ch + 1
	
	# if assign == -1:
	# 	tree_node.leaf = 1
	# 	return
	# if tree_node.leaf == 1:
	# 	if im_name in tree_node.imglist.keys():
	# 		tree_node.imglist[im_name] = tree_node.imglist[im_name] + 1
	# 	else:
	# 		tree_node.imglist[im_name] = 1
	# 		tree_node.weight = np.log(n_images/len(tree_node.imglist))

	# propogate(desc, im_name, tree_node.children[assign])

start = time.time()
count = len(tr_names)
# count = 100
ct = 0
for ind in range(0, count):
	# print("Looking at image"+(str(ind)))
	img = cv2.imread(tr_names[ind], 0)

	# Compute SIFT descriptors
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img, None)
	# print(type(des))
	# print(len(descriptors))
	for x in des:
		propogate(x, ind, tree)

	# DEBUGGING :
	ct = ct + 1
	if ct%100 == 0:
		print(str(ct) + " images propogated!")

end = time.time()
# fd.close()
print("Image propogation completed in time " + str(end - start))

# remove all the zero weighted nodes!
ld = list(leaf_dict.keys())
for lf in ld:
	p = find_node(tree, lf)
	if p.weight == 0:
		leaf_dict.pop(lf)

# Reset leaf_dict numbering!
temp_leaf = {}
ctr = 0
for x in leaf_dict.keys():
	temp_leaf[x] = ctr
	ctr = ctr + 1
leaf_dict = temp_leaf

start = time.time()
train_scores = dict.fromkeys(list(range(0, count)))
for x in train_scores.keys():
	train_scores[x] = [0]*len(leaf_dict)

# This function will compute the score  vector of images in the training set!
def score_train(tree_node):
	if tree_node.name in leaf_dict:
		imgl = tree_node.imglist.keys()
		# print(imgl)
		for img_ind in imgl:
			# Number of descriptors through this node!
			mi = tree_node.imglist[img_ind]
			# weight of this node
			wi = tree_node.weight
			temp = mi/wi
			# Find the "i" of this leaf!
			index = leaf_dict[tree_node.name]
			p = train_scores[img_ind]
			# print(p)
			# Modify the "i"th index
			p[index] = temp
			# print(temp)
			train_scores[img_ind] = p
			# print(train_scores[img_ind])
	# if len(tree_node.children) == 0:
	# 	# I am a leaf, update score here!
	# 	imgl = tree_node.imglist.keys()
	# 	print(imgl)
	# 	for img_ind in imgl:
	# 		# Number of descriptors through this node!
	# 		mi = tree_node.imglist[img_ind]
	# 		# weight of this node
	# 		wi = tree_node.weight
	# 		temp = mi/wi
	# 		# Find the "i" of this leaf!
	# 		index = leaf_dict[tree_node.name]
	# 		p = train_scores[img_ind]
	# 		print(p)
	# 		# Modify the "i"th index
	# 		p[index] = temp
	# 		print(temp)
	# 		train_scores[img_ind] = p
	# 		print(train_scores[img_ind])
	else:
		for child in tree_node.children.values():
			score_train(child)

# Now we compute the score vectors of the train images!
score_train(tree)

end = time.time()

for x in train_scores.keys():
	vec = train_scores[x]
	# Normalize vector!
	vec = vec/np.linalg.norm(vec)
	train_scores[x] = vec

print("Image scoring completed in time " + str(end - start))

# Now we have propogated the images, we have the idf and weights! Now assign scores to test images!
def find_leaves(desc, tree_node, qscore):
	# If I am a leaf, I simply score this query descriptor
	# print(tree_node.name)
	if tree_node.name in leaf_dict:
		# find the number of descriptors I (query) have through this node.
		wi = tree_node.weight
		index = leaf_dict[tree_node.name]
		# Get the "i"th index
		p = qscore[index]
		# Basically, add wi to it.
		p = p + wi
		qscore[index] = p
		
	# I am not a leaf, so I need to send this image to one!
	# Calculate distance between descriptor and each of my childrens cluster center.
	# Propogate image through child with cluster center with minimum euclidean distance!
	if tree_node.centers == []:
		# No children, therefore no centers! This branch is done.
		return 
	else:
		dist = []
		for x in tree_node.centers:
			temp = np.linalg.norm(x - desc)
			dist.append(temp)
		min_dist = min(dist)
		ind_dist = dist.index(min_dist)
		# Propogate through that!
		find_leaves(desc, tree_node.children[ind_dist], qscore)
		return 

correct_pred = 0
cntr = 0
print("Onto testing!\n")
fd = open('answers', 'w')
start = time.time()
for name in test_names:
	# print("Testing image " + str(name))
	img = cv2.imread(name, 0)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(img,None)
	query_score=[0]*len(leaf_dict)
	for x in des:
		find_leaves(x, tree, query_score)
	sim = {}
	for x in train_scores.keys():
		sim[x] = np.linalg.norm(query_score/np.linalg.norm(query_score) - train_scores[x])
	max_score = max(sim.values())
	max_matches = []
	for p in sim.keys():
		if sim[p] == max_score:
			max_matches.append(p)
	max_lbls = [tr_names[i].split('/')[1] for i in max_matches]
	# print(sim)
	test_lbl = name.split('/')[1]
	if test_lbl in max_lbls:
		correct_pred = correct_pred + 1
		# print("Correct!")
	towrite = str(test_lbl) + "\t\t" + str(max_lbls) + "\n"
	fd.write(towrite)
	if cntr%100 == 0:
		print(str(cntr) + " images tested!")
	cntr = cntr + 1

end = time.time()
print("Testing of " + str(len(test_names)) +" images completed in time " + str(end - start))
print("Accuracy = " + str(correct_pred/len(test_names)))
fd.close()

total_end = time.time()
print("Total time taken = " + str(total_end - total_start))
