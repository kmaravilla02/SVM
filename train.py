import os
import numpy as np
import cv2
import pickle
import random
import imutils
import time
import matplotlib.pyplot as plt
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import measure, color, io
from skimage.segmentation import clear_border


def create_data_WS(trdir, categories):
	global dataFile

	data = []

	for category in categories:
		path = os.path.join(trdir, category)
		label = categories.index(category)

		for img in os.listdir(path):
			imgpath = os.path.join(path, img)
			blood_img = cv2.imread(imgpath)
			blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)
			ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
			blood_imgf = cv2.resize(blood_img_otsu, (48, 48))


def output_total(Discocyte, Echinocyte, Stomatocyte, Other): #Stomatocyte,Other):

    print('Total Discocytes:', Discocyte)
    print('Total Echinocytes:', Echinocyte)
    print('Total Stomatocytes: ', Stomatocyte)
    print('Total Others: ', Other)

    cv2.waitKey(0)


def main():
	# location directory of training images
	train_dir = 'trainingbed_\\trbed_main'

	# location directory of test image/s
	test_dir = 'test_image\\Im083_02.jpg'

	# set dataset and model file
	dataFile = 'dataset_dir\\OTSU\\Data_DiscoEchinoStomatoOTSU3.pickle'
	modelFile = 'model_dir\\OTSU\\ModelSVM_DiscoEchinoStomatoOTSU3.sav'

	categories = ['Negative', 'Pos_Disco', 'Pos_Echino', 'Pos_Stomato']  # 0, 1, 2
	# displayLabels = ["Others", "Echinocytes", "Stomatocytes"]
	# categories = ['Neg_Stomatocyte', 'Pos_Stomatocyte']
	# categories = ['Discocyte', 'Echinocyte', 'Negative', 'Others', 'Stomatocyte']

	# initialize variables
	total_Others = 0
	total_Discocyte = 0
	total_Echinocyte = 0
	total_Stomatocyte = 0
	prediction = None

	# resize for imread
	# dsize = (896, 672)

	# set window width and height
	#     winW = 48
	#     winH = 48
	#     (winW, winH) = (80, 80)
	#     ss = 80

	# HOG--------------------------------------------------
	# create_data_HOG(train_dir)
	# xtrain, xtest, ytrain, ytest = prepare_data_HOG()
	# train_svm(xtrain, ytrain)

	# preClassification_report(ytest, xtest)

	# run_svm_HOG(modelFile, test_dir)

	# -----------------------------------------------------
	# create_data_OTSU(train_dir)
	# xtrain, xtest, ytrain, ytest = prepare_data_OTSU()
	# train_svm(xtrain, ytrain)

	run_svm_OTSU(modelFile, test_dir)

	# single_cell_test_OTSU(test_dir)

	# output---------------------------------
	output_total(total_Discocyte, total_Echinocyte, total_Stomatocyte, total_Others)  # total_Stomatocyte, total_Others)

	# print_report(ytest, xtest)

	# single_cell_test(test_dir)