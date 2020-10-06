import os
import numpy as np
import cv2
import pickle
import random
import imutils
import time
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.feature import hog


#path where output images will be stored
#(Hindi pa ko nag-ssave ng output image so di ko pa ginagamit to. Nakalagay lang siya ngayon dito sa code)
out_dir = os.path.join(os.getcwd(), "output")
os.makedirs(out_dir, exist_ok = True)

def create_data(train_dir):
    data = []

    for category in categories:
        path = os.path.join(train_dir,category)
        label = categories.index(category)


        for img in os.listdir(path):
            imgpath = os.path.join(path,img)
            blood_img=cv2.imread(imgpath)
            blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)
            blood_img=cv2.resize(blood_img_gray,(48, 48))


            image=np.array(blood_img).flatten()

            data.append([image,label])

    # print(len(data))

#save dataset into pickle file
    pick_in = open(dataFile,'wb')
    pickle.dump(data, pick_in)
    pick_in.close()

def prepare_data():
#load dataset from pickle file
    pick_in = open(dataFile, 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    random.shuffle(data)
    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.25)  # 90% training and 10% test

    return xtrain, xtest, ytrain, ytest

def train_SVM(x, y):

    model = SVC(C=10, class_weight=None, decision_function_shape='ovo', gamma=0.001, kernel='sigmoid')
    # model = SVC(C=1, kernel='poly', gamma=0.1)  # adjust SVC parameter to create new SVM Model
    model.fit(x, y)

#save model into .SAV file
    pick = open(modelFile, 'wb')
    pickle.dump(model, pick)
    pick.close()

def plot_confusionMatrix(xtest, ytest, displayLabels):

    pick = open(modelFile, 'rb')  # change model depending on SVC parameter
    model = pickle.load(pick)
    pick.close()

    prediction = model.predict(xtest)
    # accuracy = model.score(xtest, ytest)
    #
    # y_true = xtest
    # y_pred = prediction

    matrix = plot_confusion_matrix(model, xtest, ytest, display_labels=displayLabels, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show(matrix)
    plt.show




if __name__ == "__main__":

#location directory of training images
    train_dir = 'rawdata_echistomato'

#location directory of test image/s
    test_dir = 'test_image\\Im083_02.jpg'

#set dataset and model file
    dataFile = 'dataset_dir\\DataEchinoStomatoSampleGRAY1.pickle'
    modelFile = 'model_dir\\ModelSVMEchinoStomatoSample1.sav'

    categories = ['Negative', 'Pos_Disco', 'Pos_Echino', 'Pos_Stomato'] #0, 1, 2
    # displayLabels = ["Others", "Echinocytes", "Stomatocytes"]
    # categories = ['Neg_Stomatocyte', 'Pos_Stomatocyte']
    # categories = ['Discocyte', 'Echinocyte', 'Negative', 'Others', 'Stomatocyte']

#initialize variables
    total_Discocyte = 0
    total_Stomatocyte = 0
    total_Echinocyte = 0
    total_Others = 0

#resize for imread
    # dsize = (896, 672)

# set window width and height
#     winW = 48
#     winH = 48
#     (winW, winH) = (80, 80)
#     ss = 80

    # create_data(train_dir)
    # xtrain, xtest, ytrain, ytest = prepare_data()
    # train_SVM(xtrain, ytrain)


# load model
    pick = open(modelFile, 'rb')  # change model depending on SVC parameter
    model = pickle.load(pick)
    pick.close()

#image pipeline
    # img1 = glob.glob("test_image/*.jpg")
    # for images in img1:
    #     # load image
    #     bld_img = cv2.imread(images)
    #     bld_img = cv2.resize(bld_img, dsize)
    #     scan_image(bld_img, winW, winH, model)

#load single image
    blood_img = cv2.imread(test_dir)

#nagcreate lang ako ng copy sa line na to
    blood_img2 = blood_img

#load image as grayscale
    blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)

#perform OTSU's binarization method
    ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)

#inverse otsu (black bg, white foreground)
    invblood_otsu = cv2.bitwise_not(blood_img_otsu)

#fill holes found on Inverse Otsu
    contours, hierarchy = cv2.findContours(invblood_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(invblood_otsu, [cnt], 0, 255, -1)

#Clean cell edges
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(invblood_otsu, cv2.MORPH_OPEN, kernel, iterations = 3)
    cv2.imshow('Morphological', img)

#BlobDetector Parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 256
    params.filterByArea = True
    params.maxArea = 12000
    params.minArea = 0
    params.filterByColor = True
    params.blobColor = 255
    params.filterByCircularity = True
    params.minCircularity = 0.0
    # params.maxCircularity = 0.785
    params.filterByConvexity = True
    params.minConvexity = 0.00
    # params.maxConvexity = 0.785
    params.filterByInertia = True
    params.minInertiaRatio = 0.00

#BlobDetector
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    draw = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", draw)

#find center of each blob
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0,0

        cv2.circle(blood_img, (cX, cY), 5, (255, 0, 0), -1)
        cv2.imshow("Centroids", blood_img)
        cv2.putText(blood_img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    #Position offset variables
        ul = 56
        jl = 120

    #Position variables
        #initial location variables (x1.y1)
        #final location variables (x2,y2)

    #variables used to create ROI on the parts of the image
        x1 = cX - ul
        y1 = cY + ul

        x2 = x1 + jl
        y2 = y1 - jl

    #create rectangle on each identified blob forming around the center of the blob
        cv2.rectangle(blood_img2, (x1, y1), (x2, y2), (0, 255, 255), 1)
        # cv2.imshow("img2", bld_img2)


        # cv2.imshow("Image", blood)

    # pwedeng icomment muna itong since ung currently trained dataset, nakabase sa grayscale lang.
    #  #HOG Parameters
    #     ppc = 2
    #     cpb = 2
        # num = random.randint(0, 302)
        # print(num)
        # random.shuffle(data)
        # hog_images = []
        # hog_features = []
        # # labels = []

    #If patch of image to be extracted exceeds the image size (640x640),
        # ignore patch, then move on to next patch
        if x1 < 0 or x2 > 640 or y1 > 640 or y2 < 0:
            continue


        try:
            clone = blood_img_gray.copy()
            patch1 = clone[x1:x1+jl, y1:y1+jl]
            patch2 = cv2.resize(patch1, (48, 48))

            patch2 = np.array(patch2)


            #Apply HOG for each image on each iteration
            # fd, hog_image = hog(patch2, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb),
            #                     block_norm='L2', visualize=True)

            patch_final = patch2.flatten()
            patch_final = patch_final.reshape(-1,2304)
            prediction = model.predict(patch_final)
            # extImg = np.array(window).flatten()

            if prediction[0] == 2:
                total_Echinocyte = total_Echinocyte + 1
                cv2.rectangle(blood_img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(blood_img2, "echinocyte", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            elif prediction[0] == 3:
                total_Stomatocyte = total_Stomatocyte + 1
                cv2.rectangle(blood_img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(blood_img2, "stomatocyte", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            elif prediction[0] == 1:
                total_Others = total_Others + 1

            cv2.imshow("Image", blood_img2)

        except:
            continue


    print('Total Echinocytes:', total_Echinocyte)
    print('Total Stomatocytes: ', total_Stomatocyte)
    print('Total Others: ', total_Others)

    cv2.waitKey(0)