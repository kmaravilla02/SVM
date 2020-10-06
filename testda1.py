import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from skimage.filters import prewitt_v, prewitt_h
from skimage.feature import hog
from skimage import color
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import metrics
from sklearn.svm import SVC


dir = 'C:\\Users\\Keenu\\Desktop\\desk\\COE200L\\Thsis\\BLOOD CELLS TESTING IMAGE\\Data_Echistomato\\Pos_Echino\\Echi (57).jpg'

img = cv2.imread(dir)
print(img.shape)
blood_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)

prew_v = prewitt_v(blood_img_gray)

cv2.imshow("img", prew_v)
cv2.waitKey(0)