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


#path where output images will be stored
#(Hindi pa ko nag-ssave ng output image so di ko pa ginagamit to. Nakalagay lang siya ngayon dito sa code)
out_dir = os.path.join(os.getcwd(), "output")
os.makedirs(out_dir, exist_ok = True)


def create_data_OTSU(trdir):
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

            print(blood_imgf.shape)

            image = np.array(blood_imgf).flatten()

            print(image.shape)

            data.append([image, label])

    print(len(data))


#save dataset into pickle file
    pick_in = open(dataFile,'wb')
    pickle.dump(data, pick_in)
    pick_in.close()

def prepare_data_OTSU():

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

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.001)  # 90% training and 10% test

    return xtrain, xtest, ytrain, ytest

def create_data_WS(trdir):
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

def train_svm(xtr, ytr):
    global modelFile

    model = SVC(C=1, class_weight='balanced', gamma=1, kernel='poly') #OTSU
    # model = SVC(C=10, class_weight='balanced', gamma=1, kernel='linear') #HOG
    # model = SVC(C=10, class_weight='balanced', decision_function_shape='ovo', gamma='auto', kernel='sigmoid')
    # model = SVC(C=1, kernel='poly', gamma=0.1)  # adjust SVC parameter to create new SVM Model
    model.fit(xtr, ytr)

#save model into .SAV file
    pick = open(modelFile, 'wb')
    pickle.dump(model, pick)
    pick.close()

def preClassification_report(xtest, ytest): #displayLabels):

    print(classification_report(ytest, xtest))
    print(confusion_matrix(ytest, xtest))

def run_svm_OTSU(modelFile, test_dir):

    global total_Discocyte
    global total_Echinocyte
    global total_Stomatocyte
    global total_Others
    global prediction

    # load model
    pick = open(modelFile, 'rb')  # change model depending on SVC parameter
    model = pickle.load(pick)
    pick.close()

    # image pipeline
    # img1 = glob.glob("test_image/*.jpg")
    # for images in img1:
    #     # load image
    #     bld_img = cv2.imread(images)
    #     bld_img = cv2.resize(bld_img, dsize)
    #     scan_image(bld_img, winW, winH, model)

    # load single image
    blood_img = cv2.imread(test_dir)

    # sharpen
    # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype = np.float32)
    # laplacian = cv2.filter2D(blood_img, cv2.CV_32F, kernel)
    # sharp = np.float32(blood_img)
    # blood_img = sharp - laplacian
    # blood_img = np.clip(blood_img, 0, 255)
    # blood_img = blood_img.astype('uint8')
    # laplacian = np.clip(laplacian, 0, 255)
    # laplacian = np.uint8(laplacian)

    #cv2.imshow("sharpened", blood_img)

    cv2.waitKey(0)

    # nagcreate lang ako ng copy sa line na to
    #blood_img2 = blood_img

    # load image as grayscale
    blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)

    # perform OTSU's binarization method
    ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("otsu", blood_img_otsu)

    # inverse otsu (black bg, white foreground)
    invblood_otsu = cv2.bitwise_not(blood_img_otsu)
    # cv2.imshow("inverse otsu", invblood_otsu)

    # noise removal
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) #7,7
    opening = cv2.morphologyEx(invblood_otsu, cv2.MORPH_OPEN, kernel1, iterations=1)
    cv2.imshow("erode-dilate", opening)

    cv2.waitKey(0)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel1, iterations=10)
    cv2.imshow("bg", sure_bg)

    cv2.waitKey(0)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow("normalized", dist_transform)

    cv2.waitKey(0)

    ret, sure_fg = cv2.threshold(dist_transform, 0.157 * dist_transform.max(), 255, 0) #0.157
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
    #
    # # # inverse otsu (black bg, white foreground)
    # # invblood_otsu = cv2.bitwise_not(blood_img_otsu)
    # #
    # # cv2.imshow("inverse otsu", invblood_otsu)
    #
    # sure_fg_8u = sure_fg.astype('uint8')
    #
    # contours, hierarchy = cv2.findContours(sure_fg_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # markers = np.zeros(sure_fg.shape, dtype=np.int32)
    #
    # for i in range(len(contours)):
    #     cv2.drawContours((markers, contours, i, (i+1), -1))
    #
    # cv2.circle(markers, (5,5), 3, (255,255,255), -1)
    # cv2.imshow('Markers', markers * 10000)
    #
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(opening, [cnt], 0, 255, -1)

    cv2.imshow("filled", opening)

    img = opening

    # # fill holes found on Inverse Otsu
    # contours, hierarchy = cv2.findContours(invblood_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     cv2.drawContours(invblood_otsu, [cnt], 0, 255, -1)
    #
    # cv2.imshow("filled", invblood_otsu)
    #
    # # Clean cell edges
    # kernel = np.ones((2, 2), np.uint8)
    # img = cv2.morphologyEx(invblood_otsu, cv2.MORPH_OPEN, kernel, iterations=3)
    # #cv2.imshow('Morphological', img)

    # BlobDetector Parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 256
    params.filterByArea = True
    params.maxArea = 13000
    params.minArea = 1000 #2000
    params.filterByColor = True
    params.blobColor = 255
    params.filterByCircularity = False
    params.maxCircularity = 1.00
    params.minCircularity = 0.75
    params.filterByConvexity = True
    params.maxConvexity = 1.00
    params.minConvexity = 0.00
    params.filterByInertia = True
    params.maxInertiaRatio = 1.00
    params.minInertiaRatio = 0.00

    # BlobDetector
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    draw = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # for keypoints in keypoints:
    #     print(keypoints.size)

    cv2.imshow("Keypoints", draw)

    cv2.waitKey(0)

    #General Test Param
    # # Position offset variables
    # ul = 50
    # jl = 100
    #
    # #patch size
    # PTCH_SZ = 55 #52


    #Stomatest_Param
    # Position offset variables
    ul = 35
    jl = 70

    # patch size
    PTCH_SZ = 30  # 52

    # #Echinotest Param
    # # Position offset variables
    # ul = 40
    # jl = 70
    #
    # # patch size
    # PTCH_SZ = 40  # 52

    # # find center of each blob
    # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # contours, hierarchy = cv2.findContours(blood_img_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # cv2.imshow("pre-moment", img)
    #
    # for c in contours:
    #     M = cv2.moments(c)
    #
    #     if M["m00"] != 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #     else:
    #         cX, cY = 0, 0
    #
    #     cv2.circle(blood_img, (cX, cY), 5, (255, 0, 0), -1)
        # cv2.imshow("Centroids", blood_img)
        #cv2.putText(blood_img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for kp in keypoints:
        cX = int(kp.pt[0])
        cY = int(kp.pt[1])

        cv2.circle(blood_img, (cX, cY), 3, (255, 0, 0), -1)

        # cv2.waitKey(0)

        # Position variables
        # initial location variables (x1.y1)
        # final location variables (x2,y2)

        # variables used to create ROI on the parts of the image
        x1 = cX - ul
        y1 = cY + ul

        x2 = x1 + jl
        y2 = y1 - jl

        # x1 = cX - w
        # y1 = cY + h
        #
        # x2 = x1 + w
        # y2 = y1 - h

        # create rectangle on each identified blob forming around the center of the blob
        cv2.rectangle(blood_img, (x1, y1), (x2, y2), (0, 255, 200), 1)
        cv2.imshow("Moments", blood_img)

        cv2.waitKey(0)
        # cv2.imshow("Image", blood)

        # pwedeng icomment muna itong since ung currently trained dataset, nakabase sa grayscale lang.


        # If patch of image to be extracted exceeds the image size (640x640),
        # ignore patch, then move on to next patch
        if x1 < 0 or x2 > 640 or y1 > 640 or y2 < 0:
            continue

        try:
            clone = blood_img_otsu.copy()

            patch1 = clone[cY-PTCH_SZ:cY+PTCH_SZ, cX-PTCH_SZ:cX+PTCH_SZ]
            # patch1 = clone[y1:y1 + jl, x1:x1 + jl]
            # patch1 = clone[x1:x2, y1:y2]
            # patch1 = clone[y1:y2, x1,x2]

            # patch1 = clone[x1:x1 + jl, y1:y1 + jl]
            # patch2 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
            patch3 = cv2.resize(patch1, (48, 48))

            cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("cropped", 48 * 6, 48 * 6)
            cv2.imshow("cropped", patch3)

            # cv2.waitKey(0)

            patch4 = np.array(patch3).flatten()
            # patch_final = patch4[0]
            patch_final = patch4.reshape(-1, 2304)

            # prediction = None

            prediction = model.predict(patch_final)

            # prediction[0] = 1

            if prediction[0] == 0:
                total_Others = total_Others + 1
                cv2.rectangle(blood_img, (x1, y1), (x2, y2), (255, 200, 100), 2)
                cv2.putText(blood_img, "others", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)

            elif prediction[0] == 1:
                total_Discocyte = total_Discocyte + 1
                cv2.rectangle(blood_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(blood_img, "discocyte", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            elif prediction[0] == 2:
                total_Echinocyte = total_Echinocyte + 1
                cv2.rectangle(blood_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(blood_img, "echinocyte", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif prediction[0] == 3:
                total_Stomatocyte = total_Stomatocyte + 1
                cv2.rectangle(blood_img, (x1, y1), (x2, y2), (255, 80, 0), 2)
                cv2.putText(blood_img, "stomatocyte", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0),2)

            cv2.imshow("Image", blood_img)

        except BaseException as ex:
            #print(ex.message, ex.args)
            continue

def output_total(Discocyte, Echinocyte, Stomatocyte, Other): #Stomatocyte,Other):

    print('Total Discocytes:', Discocyte)
    print('Total Echinocytes:', Echinocyte)
    print('Total Stomatocytes: ', Stomatocyte)
    print('Total Others: ', Other)

    cv2.waitKey(0)

def single_cell_test_OTSU(tstdir):

    # load model
    pick = open(modelFile, 'rb')  # change model depending on SVC parameter
    model = pickle.load(pick)
    pick.close()

    blood_img = cv2.imread(tstdir)### testdir -- local variable for single_cell_test,
                                  ### please refer to test_dir when selecting another single cell image file

    blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)
    ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
    blood_imgf = cv2.resize(blood_img_otsu, (48, 48))

    image = np.array(blood_imgf).flatten()

    imgf= image.reshape(-1,2304)

    prediction = model.predict(imgf)
    print('RBC Type Prediction: ', categories[prediction[0]])

    cv2.namedWindow("otsu", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("otsu", 48 * 6, 48 * 6)

    cv2.namedWindow("48x48", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("48x48", 48 * 6, 48 * 6)

    cv2.namedWindow("imggr", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("imggr", 48 * 6, 48 * 6)

    cv2.imshow("otsu", blood_img_otsu)
    cv2.imshow("48x48", blood_imgf)
    cv2.imshow("imggr", blood_img_gray)

    plt.imshow(blood_imgf) #, cmap='gray')
    plt.show()


if __name__ == "__main__":

#location directory of training images
    train_dir = 'trainingbed_\\trbed_main'

#location directory of test image/s
    test_dir = 'test_image\\Echinotest.jpg'

#set dataset and model file
    dataFile = 'dataset_dir\\OTSU\\Data_DiscoEchinoStomatoOTSU4.pickle'
    modelFile = 'model_dir\\OTSU\\ModelSVM_DiscoEchinoStomatoOTSU4.sav'

    categories = ['Negative', 'Pos_Disco', 'Pos_Echino', 'Pos_Stomato'] #0, 1, 2
    # displayLabels = ["Others", "Echinocytes", "Stomatocytes"]
    # categories = ['Neg_Stomatocyte', 'Pos_Stomatocyte']
    # categories = ['Discocyte', 'Echinocyte', 'Negative', 'Others', 'Stomatocyte']

#initialize variables
    total_Others = 0
    total_Discocyte = 0
    total_Echinocyte = 0
    total_Stomatocyte = 0
    prediction = None

#resize for imread
    # dsize = (896, 672)

# set window width and height
#     winW = 48
#     winH = 48
#     (winW, winH) = (80, 80)
#     ss = 80

#HOG--------------------------------------------------
    # create_data_HOG(train_dir)
    # xtrain, xtest, ytrain, ytest = prepare_data_HOG()
    # train_svm(xtrain, ytrain)

    # preClassification_report(ytest, xtest)

    # run_svm_HOG(modelFile, test_dir)

#-----------------------------------------------------
    # create_data_OTSU(train_dir)
    # xtrain, xtest, ytrain, ytest = prepare_data_OTSU()
    # train_svm(xtrain, ytrain)

    run_svm_OTSU(modelFile, test_dir)

    # single_cell_test_OTSU(test_dir)

#output---------------------------------
    output_total(total_Discocyte, total_Echinocyte, total_Stomatocyte, total_Others) #total_Stomatocyte, total_Others)

    #print_report(ytest, xtest)

    # single_cell_test(test_dir)



