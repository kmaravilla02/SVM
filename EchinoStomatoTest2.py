import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from skimage.feature import hog
from skimage import color
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools
from sklearn import metrics
from sklearn.svm import SVC

def sanity_check():
    # sanity check
    scores = []
    for i in range(0, 50):
        xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.25)

        model = SVC(C=100, kernel='poly', gamma=10)
        model = model.fit(xtrain, ytrain)

        scores.append(accuracy_score(model.predict(xtest), ytest))

    plt.hist(scores)

dir = 'C:\\Users\\Keenu\\PycharmProjects\\SVM\\Program1\\rawdata_echistomato'
#dir = 'C:\\Users\\Keenu\\Desktop\\desk\\COE200L\\Thsis\\BLOOD CELLS TESTING IMAGE\\rawdata_echistomato'


# categories = ['Neg_Echinocyte', 'Pos_Echinocyte']
categories = ['Negative', 'Pos_Echino', 'Pos_Stomato']  # 0, 1, 2
#categories = ['Negative', 'Pos_Disco','Pos_Echino','Pos_Stomato']
# categories = ['Discocyte', 'Echinocyte', 'Others', 'Stomatocyte']

# kernel = np.ones((5,5), np.uint8)
data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        blood_img = cv2.imread(imgpath)
        blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)
        # ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
        blood_img = cv2.resize(blood_img_gray, (48, 48))


        # img_erosion = cv2.erode(blood_img_otsu, kernel, iterations=1)
        # img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

        # blood_img = color.rgb2grey(blood_img)

        image = np.array(blood_img)

        data.append([image, label])

print(len(data))


# kernel = np.ones((5,5), np.uint8)

#load single image
# blood_img = cv2.imread(dir)
# blood_img = cv2.resize(blood_img, (48, 48))
# blood_img_gray = cv2.cvtColor(blood_img, cv2.COLOR_BGR2GRAY)
# ret, blood_img_otsu = cv2.threshold(blood_img_gray, 0, 255, cv2.THRESH_OTSU)
#
# img_erosion = cv2.erode(blood_img_otsu, kernel, iterations=1)
# img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
# blood_img = color.rgb2grey(blood_img)

# image = np.array(blood_img_gray)

# cv2.imshow('win1', img_dilation)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------HOGTreatment-----------------------
ppc = 2
cpb = 2
# num = random.randint(0, 302)
# print(num)
# random.shuffle(data)
hog_images = []
hog_features = []
labels = []

for images, label in data:
    fd, hog_image = hog(images, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb), block_norm='L2', visualize=True)
    hog_images.append(hog_image)
    # hog_features = np.array(hog_images)
    hog_features.append(fd)
    labels.append(label)


#-----------------------REGULAR DATA EXTRACTION--------------------------
# features = []
# labels = []
#
# for feature, label in data:
#     features.append(feature)
#     labels.append(label)

# pick_in = open('DataEchinoStomato1.pickle', 'wb')
# pickle.dump(data,pick_in)
# pick_in.close()

xtrain, xtest, ytrain, ytest = train_test_split(hog_features, labels, test_size=0.25)

# model = SVC(C= 10, class_weight= None, decision_function_shape= 'ovo', gamma= 0.001, kernel= 'sigmoid')
#
# # model = SVC(C=1, class_weight='balanced', decision_function_shape='ovo', gamma=1, kernel='poly')
# # model = SVC(C=100, kernel='poly', gamma='scale', degree=4)
#
# model = model.fit(xtrain, ytrain)
#
# # pick = open('ModelSVMEchinoStomato5.sav','wb')
# # pickle.dump(model, pick)
# # pick.close()
#
# prediction = model.predict(xtest)
#
#
# print(classification_report(ytest,prediction))
# print(confusion_matrix(ytest, prediction))

# accuracy = model.score(xtest, ytest)

# y_true = xtrain
# y_pred = prediction

# acc1 = accuracy_score(y_true, y_pred)
# print(acc1)

# percent_acc = accuracy * 100
# print('Accuracy: ', accuracy)
# print('% Accuracy:', percent_acc)
# print('RBC Type Prediction: ', categories[prediction[0]])
#
# matrix = plot_confusion_matrix(model, xtest, ytest, display_labels=categories, cmap = plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.show(matrix)
# plt.show

param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001,100,1000], 'kernel':['linear','rbf', 'poly','sigmoid'], 'class_weight':['balanced', None], 'decision_function_shape': ['ovo', 'ovr']}

grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)

grid.fit(xtrain,ytrain)

print(grid.best_params_)

predic = grid.predict(xtest)

print(classification_report(ytest, predic))
print(confusion_matrix(ytest, predic))

#{'C': 1, 'class_weight': 'balanced', 'decision_function_shape': 'ovo', 'gamma': 1, 'kernel': 'linear'}