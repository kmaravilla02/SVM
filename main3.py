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

# output_file = open('Central Pallor.csv', 'x')

# def create_data_WS(trdir):
# 	global dataFile
#
# 	data = []
#
# 	for category in categories:
# 		path = os.path.join(trdir, category)
# 		label = categories.index(category)
#
# 		for img in os.listdir(path):
# 			imgpath = os.path.join(path, img)
# 			blood_img = cv2.imread(imgpath)
# 			blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)
# 			ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
# 			blood_imgf = cv2.resize(blood_img_otsu, (48, 48))


if __name__ == "__main__":

	propList = ['Area',
				'equivalent_diameter',
				'orientation',
				'MajorAxisLength',
				'MinorAxisLength',
				'Perimeter',
				'MinIntensity',
				'MeanIntensity',
				'MaxIntensity']

	output_file = open('Central Pallor.csv', 'w')
	output_file.write('Cell #' + "," + "," + ",".join(propList) + '\n')
	output_file.close()

	global dataFile

	# location directory of training images
	train_dir = 'trainingbed_\\trbed_main'

	# location directory of test image/s
	# test_dir = 'test_image\\Im083_02.jpg'

	dataFile = 'dataset_dir\\OTSU\\Data_DiscoEchinoStomatoWS0.pickle'
	modelFile = 'model_dir\\OTSU\\ModelSVM_DiscoEchinoStomatoWS0.sav'
	# set dataset and model file

	categories = ['Negative', 'Pos_Disco', 'Pos_Echino', 'Pos_Stomato']  # 0, 1, 2
	# displayLabels = ["Others", "Echinocytes", "Stomatocytes"]
	# categories = ['Neg_Stomatocyte', 'Pos_Stomatocyte']
	# categories = ['Discocyte', 'Echinocyte', 'Negative', 'Others', 'Stomatocyte']

	global total_Discocyte
	global total_Echinocyte
	global total_Stomatocyte
	global total_Others
	global prediction

	# # load model
	# pick = open(modelFile, 'rb')  # change model depending on SVC parameter
	# model = pickle.load(pick)
	# pick.close()

	data = []

	for category in categories:
		path = os.path.join(train_dir, category)
		label = categories.index(category)

		for img in os.listdir(path):
			imgpath = os.path.join(path, img)
			blood_img = cv2.imread(imgpath)
			blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)
			ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
			blood_imgf = cv2.resize(blood_img_otsu, (48, 48))

			# noise removal
			kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
			opening = cv2.morphologyEx(blood_img_otsu, cv2.MORPH_OPEN, kernel1, iterations=1)
			cv2.imshow("erode-dilate", opening)

			cv2.waitKey(0)

			# sure background area
			sure_bg = cv2.dilate(opening, kernel1, iterations=3)
			cv2.imshow("bg", sure_bg)

			cv2.waitKey(0)

			# Finding sure foreground area
			dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
			cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
			cv2.imshow("normalized", dist_transform)

			cv2.waitKey(0)

			ret, sure_fg = cv2.threshold(dist_transform, 0.0001 * dist_transform.max(), 255, 0)
			cv2.imshow("fg", sure_fg)

			# sure_fg = cv2.dilate(sure_fg, kernel1)
			# cv2.imshow("peaks", sure_fg)

			# Finding unknown region
			sure_fg = np.uint8(sure_fg)
			unknown = cv2.subtract(sure_bg, sure_fg)

			cv2.waitKey(0)

			# Marker labelling
			ret, markers = cv2.connectedComponents(sure_fg)

			# Add one to all labels so that sure background is not 0, but 1
			markers = markers + 1

			# Now, mark the region of unknown with zero
			markers[unknown == 255] = 0

			markers = cv2.watershed(blood_img, markers)
			blood_img[markers == -1] = [0, 255, 255]

			ws = color.label2rgb(markers, bg_label=0)
			cv2.imshow("overlay on original image", blood_img)
			cv2.imshow("watershed", ws)

			cv2.waitKey(0)

			regions = measure.regionprops(markers, intensity_image=blood_img)

			propList = ['Area',
						'equivalent_diameter',
						'orientation',
						'MajorAxisLength',
						'MinorAxisLength',
						'Perimeter',
						'MinIntensity',
						'MeanIntensity',
						'MaxIntensity']

			output_file = open('Central Pallor.csv', 'w')
			output_file.write('Cell #' + "," + "," + ",".join(propList) + '\n')
			output_file.close()
	#
	# # image pipeline
	# # img1 = glob.glob("test_image/*.jpg")
	# # for images in img1:
	# #     # load image
	# #     bld_img = cv2.imread(images)
	# #     bld_img = cv2.resize(bld_img, dsize)
	# #     scan_image(bld_img, winW, winH, model)
	#
	# # load single image
	# # blood_img = cv2.imread(test_dir)
	#
	# # sharpen
	# # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype = np.float32)
	# # laplacian = cv2.filter2D(blood_img, cv2.CV_32F, kernel)
	# # sharp = np.float32(blood_img)
	# # blood_img = sharp - laplacian
	# # blood_img = np.clip(blood_img, 0, 255)
	# # blood_img = blood_img.astype('uint8')
	# # laplacian = np.clip(laplacian, 0, 255)
	# # laplacian = np.uint8(laplacian)
	#
	# #cv2.imshow("sharpened", blood_img)
	#
	# cv2.waitKey(0)
	#
	# # nagcreate lang ako ng copy sa line na to
	# #blood_img2 = blood_img
	#
	# # load image as grayscale
	# blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)
	#
	# # perform OTSU's binarization method
	# ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
	# # cv2.imshow("otsu", blood_img_otsu)
	#
	# # inverse otsu (black bg, white foreground)
	# invblood_otsu = cv2.bitwise_not(blood_img_otsu)
	# # cv2.imshow("inverse otsu", invblood_otsu)
	#
	# # noise removal
	# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
	# opening = cv2.morphologyEx(blood_img_otsu, cv2.MORPH_OPEN, kernel1, iterations=1)
	# cv2.imshow("erode-dilate", opening)
	#
	# cv2.waitKey(0)
	#
	# # sure background area
	# sure_bg = cv2.dilate(opening, kernel1, iterations=3)
	# cv2.imshow("bg", sure_bg)
	#
	# cv2.waitKey(0)
	#
	# # Finding sure foreground area
	# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
	# cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
	# cv2.imshow("normalized", dist_transform)
	#
	# cv2.waitKey(0)
	#
	# ret, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
	# cv2.imshow("fg", sure_fg)
	#
	# # sure_fg = cv2.dilate(sure_fg, kernel1)
	# # cv2.imshow("peaks", sure_fg)
	#
	# # Finding unknown region
	# sure_fg = np.uint8(sure_fg)
	# unknown = cv2.subtract(sure_bg, sure_fg)
	#
	# cv2.waitKey(0)
	#
	# # Marker labelling
	# ret, markers = cv2.connectedComponents(sure_fg)
	#
	# # Add one to all labels so that sure background is not 0, but 1
	# markers = markers + 1
	#
	# # Now, mark the region of unknown with zero
	# markers[unknown == 255] = 0
	#
	# markers = cv2.watershed(blood_img, markers)
	# blood_img[markers == -1] = [0, 255, 255]
	#
	# ws = color.label2rgb(markers, bg_label=0)
	# cv2.imshow("overlay on original image", blood_img)
	# cv2.imshow("watershed", ws)
	#
	# cv2.waitKey(0)
	#
	# regions = measure.regionprops(markers, intensity_image=blood_img)
	#
	# propList = ['Area',
	# 			'equivalent_diameter',
	# 			'orientation',
	# 			'MajorAxisLength',
	# 			'MinorAxisLength',
	# 			'Perimeter',
	# 			'MinIntensity',
	# 			'MeanIntensity',
	# 			'MaxIntensity']
	#
	# output_file = open('Central Pallor.csv', 'w')
	# output_file.write('Cell #' + "," + "," + ",".join(propList) + '\n')
	# output_file.close()
	# #
	# # # # inverse otsu (black bg, white foreground)
	# # # invblood_otsu = cv2.bitwise_not(blood_img_otsu)
	# # #
	# # # cv2.imshow("inverse otsu", invblood_otsu)
	# #
	# # sure_fg_8u = sure_fg.astype('uint8')
	# #
	# # contours, hierarchy = cv2.findContours(sure_fg_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# #
	# # markers = np.zeros(sure_fg.shape, dtype=np.int32)
	# #
	# # for i in range(len(contours)):
	# #     cv2.drawContours((markers, contours, i, (i+1), -1))
	# #
	# # cv2.circle(markers, (5,5), 3, (255,255,255), -1)
	# # cv2.imshow('Markers', markers * 10000)
	# #
	# # cv2.waitKey(0)
	#
	# # contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# # for cnt in contours:
	# #     cv2.drawContours(opening, [cnt], 0, 255, -1)
	# #
	# # cv2.imshow("filled", opening)
	# #
	# # img = opening
	# #
	# # # # fill holes found on Inverse Otsu
	# # # contours, hierarchy = cv2.findContours(invblood_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# # # for cnt in contours:
	# # #     cv2.drawContours(invblood_otsu, [cnt], 0, 255, -1)
	# # #
	# # # cv2.imshow("filled", invblood_otsu)
	# # #
	# # # # Clean cell edges
	# # # kernel = np.ones((2, 2), np.uint8)
	# # # img = cv2.morphologyEx(invblood_otsu, cv2.MORPH_OPEN, kernel, iterations=3)
	# # # #cv2.imshow('Morphological', img)
	# #
	# # # BlobDetector Parameters
	# # params = cv2.SimpleBlobDetector_Params()
	# # params.minThreshold = 0
	# # params.maxThreshold = 256
	# # params.filterByArea = True
	# # params.maxArea = 13000
	# # params.minArea = 2000
	# # params.filterByColor = True
	# # params.blobColor = 255
	# # params.filterByCircularity = False
	# # params.maxCircularity = 1.00
	# # params.minCircularity = 0.75
	# # params.filterByConvexity = True
	# # params.maxConvexity = 1.00
	# # params.minConvexity = 0.00
	# # params.filterByInertia = True
	# # params.maxInertiaRatio = 1.00
	# # params.minInertiaRatio = 0.00
	# #
	# # # BlobDetector
	# # detector = cv2.SimpleBlobDetector_create(params)
	# # keypoints = detector.detect(img)
	# #
	# # draw = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# # # for keypoints in keypoints:
	# # #     print(keypoints.size)
	# #
	# # cv2.imshow("Keypoints", draw)
	# #
	# # cv2.waitKey(0)
