import os
import numpy as np
import cv2
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage import measure, color, io

def create_data(trdir):
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

    xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.001)  # 90% training and 10% test

    return xtrain, xtest, ytrain, ytest

def train_svm(xtr, ytr):
    global modelFile

    model = SVC(C=1, class_weight='balanced', gamma=1, kernel='poly') #OTSU
    model.fit(xtr, ytr)

#save model into .SAV file
    pick = open(modelFile, 'wb')
    pickle.dump(model, pick)
    pick.close()

def run_svm(modelFile, test_dir):

    global total_Discocyte
    global total_Echinocyte
    global total_Stomatocyte
    global total_Others
    global prediction

    # load model
    pick = open(modelFile, 'rb')  # change model depending on SVC parameter
    model = pickle.load(pick)
    pick.close()

    # load single image
    blood_img = cv2.imread(test_dir)
    cv2.waitKey(0)

    # load image as grayscale
    blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)

    # perform OTSU's binarization method
    ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("otsu", blood_img_otsu)

    # inverse otsu (black bg, white foreground)
    invblood_otsu = cv2.bitwise_not(blood_img_otsu)
    # cv2.imshow("inverse otsu", invblood_otsu)

    # noise removal
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) #7,7 ###CHANGE VALUE per VARYING IMAGE IF NOT FROM SAME SET OF IMAGE
    opening = cv2.morphologyEx(invblood_otsu, cv2.MORPH_OPEN, kernel1, iterations=1)
    cv2.imshow("erode-dilate", opening)
    # cv2.waitKey(0)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel1, iterations=10)
    # cv2.imshow("bg", sure_bg)
    # cv2.waitKey(0)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    # cv2.imshow("normalized", dist_transform)
    # cv2.waitKey(0)

    ret, sure_fg = cv2.threshold(dist_transform, 0.23 * dist_transform.max(), 255, 0) #0.23 ##DEFAULT FOR IM083_XX IMAGES
    cv2.imshow("fg", sure_fg)

    # sure_fg = cv2.dilate(sure_fg, kernel1)
    # cv2.imshow("peaks", sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(opening, [cnt], 0, 255, -1)

    cv2.imshow("filled", opening)

    img = opening

    # BlobDetector Parameters
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 256
    params.filterByArea = True
    params.maxArea = 13000 #13000
    params.minArea = 2000 #2000 default #CHANGE VALUE per VARYING IMAGE IF NOT FROM SAME SET OF IMAGE
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
    cv2.imshow("Keypoints", draw)

    #General Test Param
    # Position offset variables
    ul = 50
    jl = 100

    #patch size
    PTCH_SZ = 55 #52
#-----------------------------------------

    # #Stomatest_Param
    # # Position offset variables
    # ul = 35
    # jl = 70
    #
    # # patch size
    # PTCH_SZ = 30  # 52
#-----------------------------------------

    # #Echinotest Param
    # # Position offset variables
    # ul = 40
    # jl = 70
    #
    # # patch size
    # PTCH_SZ = 40  # 52

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

        # create rectangle on each identified blob forming around the center of the blob
        cv2.rectangle(blood_img, (x1, y1), (x2, y2), (0, 255, 200), 1)
        cv2.imshow("Image", blood_img)

        # If patch of image to be extracted exceeds the image size (640x640),
        # ignore patch, then move on to next patch
        if x1 < 0 or x2 > 640 or y1 > 640 or y2 < 0:
            continue

        try:
            clone = blood_img_otsu.copy()

            patch1 = clone[cY-PTCH_SZ:cY+PTCH_SZ, cX-PTCH_SZ:cX+PTCH_SZ]
            patch3 = cv2.resize(patch1, (48, 48))

            cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("cropped", 48 * 6, 48 * 6)
            cv2.imshow("cropped", patch3)

            # cv2.waitKey(0)

            patch4 = np.array(patch3).flatten()
            patch_final = patch4.reshape(-1, 2304)

            prediction = model.predict(patch_final)

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
            continue

def output_total(Discocyte, Echinocyte, Stomatocyte, Other):


    print('\nNumber of Identified Cells\n')

    print('Discocytes:', Discocyte)
    print('Echinocytes:', Echinocyte)
    print('Stomatocytes:', Stomatocyte)
    print('Other Cells:', Other)

    cv2.waitKey(0)

if __name__ == "__main__":

#location directory of training images
    train_dir = 'trainingbed_\\trbed_main'

#location directory of test image/s
    test_dir = 'test_image\\Im083_04.jpg'

#set dataset and model file
    dataFile = 'dataset_dir\\OTSU\\Data_DiscoEchinoStomatoOTSU4.pickle'
    modelFile = 'model_dir\\OTSU\\ModelSVM_DiscoEchinoStomatoOTSU4.sav'

    categories = ['Negative', 'Pos_Disco', 'Pos_Echino', 'Pos_Stomato'] #0, 1, 2

#initialize variables
    total_Others = 0
    total_Discocyte = 0
    total_Echinocyte = 0
    total_Stomatocyte = 0
    prediction = None

#-----------------------------------------------------
    # create_data(train_dir)
    # xtrain, xtest, ytrain, ytest = prepare_data()
    # train_svm(xtrain, ytrain)

    run_svm(modelFile, test_dir)

#output---------------------------------
    output_total(total_Discocyte, total_Echinocyte, total_Stomatocyte, total_Others) #total_Stomatocyte, total_Others)



