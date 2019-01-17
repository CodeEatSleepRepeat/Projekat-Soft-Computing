import os
import cv2
import math
import dlib
import operator
import datetime
import itertools
import statistics
import matplotlib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Declaration of functions
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

# Declaration of variables
image_list = {}

pos_features = []
neg_features = []
labels = []

image_dir_pos = 'slike/positive'
image_dir_neg = 'slike/negative'

nbins = 9 # broj binova (unutar histograma ima 9 elemenata)
cell_size = (8, 8) # broj piksela po celiji
block_size = (3, 3) # broj celija po bloku

pos_imgs = []
neg_imgs = []

print('\n*** An algorithm that inserts HOG image features in SVM and KNN for \'stupid\' detection of drug addicts ***')
start_time = datetime.datetime.utcnow()
print('\nProcess started at ', start_time,'\n')

# Algorithm 1: Download all images from the folder
for img_name in os.listdir(image_dir_pos):
    img_path = os.path.join(image_dir_pos, img_name)
    img = load_image(img_path)
    img = cv2.resize(img,(50,25))
    pos_imgs.append(img)

for img_name in os.listdir(image_dir_neg):
    img_path = os.path.join(image_dir_neg, img_name)
    img = load_image(img_path)
    img = cv2.resize(img,(50,25))
    neg_imgs.append(img)

hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)


# Algorithm 2: Calculating HOG image features and filling lists
for img in pos_imgs:
    pos_features.append(hog.compute(img))
    labels.append(1)

for img in neg_imgs:
    neg_features.append(hog.compute(img))
    labels.append(0)

# Algorithm 3: Preparation for classification
pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = reshape_data(x_train)
x_test = reshape_data(x_test)

# Algorithm 4: SVM classification
svm = SVC(kernel='linear', probability=True) 
svm.fit(x_train, y_train)
y_train_pred = svm.predict(x_train)
y_test_pred = svm.predict(x_test)
print("SVM Train accuracy: ", accuracy_score(y_train, y_train_pred))
print("SVM Validation accuracy: ", accuracy_score(y_test, y_test_pred))

# Algorithm 5: KNN classification
knn = KNeighborsClassifier(n_neighbors=10)
knn = knn.fit(x_train, y_train)
y_train_pred = knn.predict(x_train)
y_test_pred = knn.predict(x_test)
print("KNN Train accuracy: ", accuracy_score(y_train, y_train_pred))
print("KNN Validation accuracy: ", accuracy_score(y_test, y_test_pred))

end_time = datetime.datetime.utcnow()
print('\nProcess ended at ', end_time)
print('The process lasted ', end_time-start_time) 

