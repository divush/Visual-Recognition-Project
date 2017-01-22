import cv2
from tkinter import *
import matplotlib.pyplot as plt
img = cv2.imread("messi.jpg")
for alpha in range(0,4):
	num=alpha*0.25
	newim = num*img
	imname="messi"+str(alpha*0.25)+".jpg"
	cv2.imwrite(imname, newim)
	#cv2.waitKey()

remim=img+20
red, blue, green= [], [], []
cv2.imwrite("adding.jpg", remim)
for row in img:
	for vec in row:
		red.append(vec[0])
		green.append(vec[1])
		blue.append(vec[2])
plt.figure(1)                # the first figure
plt.plot(red, 'r')
plt.plot(green, 'g')
plt.plot(blue, 'b')
plt.show()
