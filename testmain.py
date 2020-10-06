import os
# from PIL import Image
import glob
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
from skimage.filters import prewitt_h, prewitt_v


#path where output images will be stored
out_dir = os.path.join(os.getcwd(), "output")
os.makedirs(out_dir, exist_ok = True)

# total_Stomatocyte = 0
# total_Echinocyte = 0
# total_Others = 0



def create_data():
    data = []

    for category in categories:
        path = os.path.join(dir,category)
        label = categories.index(category)


        for img in os.listdir(path):
            imgpath = os.path.join(path,img)
            blood_img=cv2.imread(imgpath, 0)
            blood_img=cv2.resize(blood_img,(48, 48))


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

    model = SVC(C=1, kernel='poly', gamma=0.1)  # adjust SVC parameter to create new SVM Model
    model.fit(xtrain, ytrain)

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

def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[0], x:x + windowSize[1]])


def scan_image(bld_img, winW, winH, model, ss):

    categories = ['Negative', 'Pos_Echino', 'Pos_Stomato']
    total_Stomatocyte = 0
    total_Echinocyte = 0
    total_Others = 0

    bbox = []
    # for resized in pyramid(bld_img, scale=1.5):
    for (x, y, window) in sliding_window(bld_img, stepSize=ss, windowSize=(winW, winH)):
        if window.shape[1] != winW or window.shape[0] != winH:
            continue

        # if prediction == 0:
        #     cv2.rectangle(out_img)
        #     cv2.imwrite(os.path.join(out_dir, "detected_"+os.path.basename()))
        #
        # return prediction
        # clone = resized.copy()

        blood_img_gray = cv2.cvtColor(bld_img, cv2.COLOR_BGR2GRAY)
        ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)

        crop = blood_img_otsu[x:x + winW, y:y + winH]
        try:
            patch_image = cv2.resize(crop, (48, 48))
        except:
            continue

        patch_image = np.array(patch_image)
        patch = patch_image.flatten()

        if patch.shape[0] < 2304:
            continue
        elif patch.shape[0] == 2304:
            patch = patch.reshape(-1, 2304)
        # if patch.shape[1] != 2304:
        #     continue
        # else:
        #     continue

        cv2.rectangle(bld_img, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", bld_img)
        prediction = model.predict(patch)
        # extImg = np.array(window).flatten()

        if prediction[0] == 1:
            total_Echinocyte = total_Echinocyte + 1
            # total_Echinocyte = total_Echinocyte.append(patch)

        elif prediction[0] == 2:
            total_Stomatocyte = total_Stomatocyte + 1
            # total_Stomatocyte = total_Stomatocyte.append(patch)

        elif prediction[0] == 0:
            total_Others = total_Others + 1
            # total_Others = []

        # clone = resized.copy()
        # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        # blood_img_gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
        # patch_image = blood_img_gray[x:x+winW, y:y+winH]
        # ret, blood_img_otsu = cv2.threshold(patch_image, 0, 255, cv2.THRESH_OTSU)
        # patch = np.array(blood_img_otsu).flatten()
        # patch = patch.reshape(-1, 2304)
        # prediction = model.predict(patch)
        # # extImg = np.array(window).flatten()
        # cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)

    return total_Echinocyte, total_Stomatocyte, total_Others


# def detect(img, window_list):
#     # classify all windows within image
#
#     windows = []
#     for bbox in window_list:
#         window = extract_window(img, bbox)
#         windows.append(window)
#
#     windows = np.stack(windows)
#     detections = pipeline.predict(windows)
#
#     return detections

def count_Echino(img):

    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
    #                            param1=100, param2=30, minRadius=1, maxRadius=90)
    #
    # circles = np.uint16(np.around(circles))
    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 3)
    #
    # cv2.imshow('detected circles', img)
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
    # params.maxCircularity = 0.3
    params.filterByConvexity = True
    params.minConvexity = 0.00
    # params.maxConvexity = 0.2
    params.filterByInertia = True
    params.minInertiaRatio = 0.00

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)

    #image with keypoints =
    draw = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", draw)

    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Image", img)

    cv2.waitKey(0)
    time.sleep(0.025)





# cv2::rectangle( image,
#                cvPoint(x-w/2,y-h/2),
#                cvPoint(x+w/2,y+h/2),
#                CV_RGB(r,g,b), 1, 8
#              );

if __name__ == "__main2__":


    dir = 'C:\\Users\\Keenu\\Desktop\\desk\\COE200L\\Thsis\\BLOOD CELLS TESTING IMAGE\\Tests1\\Im083_02.jpg'

    dataFile = 'DataEchinoStomatoOTSU1.pickle'
    modelFile = 'ModelSVMEchinoStomato2.sav'

    categories = ['Negative', 'Pos_Echino', 'Pos_Stomato']
    # displayLabels = ["Others", "Echinocytes", "Stomatocytes"]
    # categories = ['Neg_Stomatocyte', 'Pos_Stomatocyte']
    # categories = ['Discocyte', 'Echinocyte', 'Negative', 'Others', 'Stomatocyte']

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
    #
    bld_img = cv2.imread(dir)
    bld_img2 = bld_img

    # top = int(0.2 * bld_img.shape[0])  # shape[0] = rows
    # bottom = top
    # left = int(0.2 * bld_img.shape[1])  # shape[1] = cols
    # right = left
    #
    # bld_img = cv2.copyMakeBorder(bld_img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)

    # bld_img = cv2.resize(bld_img, dsize)

    blood_img_gray = cv2.cvtColor(bld_img, cv2.COLOR_BGR2GRAY)
    #
    ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
    invblood_otsu = cv2.bitwise_not(blood_img_otsu)

    contours, hierarchy = cv2.findContours(invblood_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(invblood_otsu, [cnt], 0, 255, -1)

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(invblood_otsu, cv2.MORPH_OPEN, kernel, iterations = 3)
    cv2.imshow('Morphological', img)

    # scan_image(bld_img, winW, winH, model, total_Echinocyte, total_Stomatocyte, total_Others, categories)



    # count_Echino(blood_img_otsu)
    # count_Echino(opening)
    # count_Echino(invblood_otsu)

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

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)

    # image with keypoints =
    draw = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", draw)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0,0

        # cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
        # cv2.imshow("Centroids", img)


        # crop = img[cX-128/2:cX+128, cY+128/2:cY+128]
        # cv2.putText(bld_img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        x1 = cX - 56
        y1 = cY + 56

        x2 = x1 + 120
        y2 = y1 - 120

        # cv2.imshow("Image", bld_img)

        # xf = 0
        # yf = 0
        # xf1 = 0
        # yf1 = 0

        # while xf1 != 128:
        #     xf += 1
        #
        # xf1 = x1+xf
        #
        # while yf1 != 128:
        #     yf += 1
        #
        # yf1 = y1 + yf

        cv2.imshow("Image", bld_img)

        if x1 < 0 or x2 > 639 or y1 > 639 or y2 < 0:
            continue


        # try:
        #
        # except:
        #

        # patch_image = cv2.resize(crop, (48, 48))
        #
        # # try:
        # #     perimeter = cv2.arcLength(crop, True)
        # #     patch_image = cv2.resize(crop, (48, 48))
        # # except:
        # #     continue
        #
        # patch_image = np.array(crop)
        # patch = patch_image.flatten()

        try:
            clone = blood_img_otsu.copy()
            patch1 = clone[x1:x1+120, y1:y1+120]
            patch_image = cv2.resize(patch1, (48, 48))

            # try:
            #     perimeter = cv2.arcLength(crop, True)
            #     patch_image = cv2.resize(crop, (48, 48))
            # except:
            #     continue

            patch_image = np.array(patch_image)
            patch = patch_image.flatten()

            patch = patch.reshape(-1,2304)
            prediction = model.predict(patch)
            # extImg = np.array(window).flatten()

            if prediction[0] == 1:
                total_Echinocyte = total_Echinocyte + 1
                cv2.rectangle(bld_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # total_Echinocyte = total_Echinocyte.append(patch)

            elif prediction[0] == 2:
                total_Stomatocyte = total_Stomatocyte + 1
                cv2.rectangle(bld_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # total_Stomatocyte = total_Stomatocyte.append(patch)

            elif prediction[0] == 0:
                total_Others = total_Others + 1

            # cv2.imshow("Image", img)

        except:
            continue


        # cv2.waitKey(0)

    # cv2.waitKey(0)
    # time.sleep(0.025)




    # total_Echinocyte, total_Stomatocyte, total_Others = scan_image(bld_img, winW, winH, model, ss)


# #visualize windows to be search for blood cells
#     out_img = np.array(bld_img)

    # create_data()
    # xtrain, xtest, ytrain, ytest = prepare_data()
    # train_SVM(xtrain, ytrain)
    # plot_confusionMatrix(xtest, ytest, displayLabels)\


    # for resized in pyramid(bld_img, scale=1.5):
    #     for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
    #         if window.shape[0] != winW or window.shape[1] != winH:
    #             continue
    #
    #         # if prediction == 0:
    #         #     cv2.rectangle(out_img)
    #         #     cv2.imwrite(os.path.join(out_dir, "detected_"+os.path.basename()))
    #         #
    #         # return prediction
    #         clone = resized.copy()
    #
    #         blood_img_gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
    #         ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
    #         blood_img_otsu = np.array(blood_img_otsu)
    #         patch_image = blood_img_otsu[x:x + winW, y:y + winH]
    #
    #         patch = np.array(patch_image).flatten()
    #         if patch.shape[0] < 2304:
    #             continue
    #         elif patch.shape[0] == 2304:
    #             patch = patch.reshape(-1, 2304)
    #         # if patch.shape[1] != 2304:
    #         #     continue
    #         # else:
    #         #     continue
    #
    #         cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    #         cv2.imshow("Window", clone)
    #         prediction = model.predict(patch)
    #         # extImg = np.array(window).flatten()
    #
    #         if categories[prediction[0]] == '1':
    #             total_Echinocyte = total_Echinocyte.append(patch)
    #
    #         elif categories[prediction[0]] == '2':
    #             total_Stomatocyte = total_Stomatocyte.append(patch)
    #
    #         elif categories[prediction[0]] == '0':
    #             total_Others = []
    #
    #
    #         # clone = resized.copy()
    #         # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    #         # blood_img_gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
    #         # patch_image = blood_img_gray[x:x+winW, y:y+winH]
    #         # ret, blood_img_otsu = cv2.threshold(patch_image, 0, 255, cv2.THRESH_OTSU)
    #         # patch = np.array(blood_img_otsu).flatten()
    #         # patch = patch.reshape(-1, 2304)
    #         # prediction = model.predict(patch)
    #         # # extImg = np.array(window).flatten()
    #         # cv2.imshow("Window", clone)
    #         cv2.waitKey(1)
    #         # time.sleep(0.025)



    # print('Total Echinocytes:', len(total_Echinocyte))
    # print('Total Stomatocytes: ', len(total_Stomatocyte))
    # print('Total Others: ', len(total_Others))

    print('Total Echinocytes:', total_Echinocyte)
    print('Total Stomatocytes: ', total_Stomatocyte)
    print('Total Others: ', total_Others)

    cv2.waitKey(0)