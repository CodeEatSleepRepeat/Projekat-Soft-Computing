import os
import cv2
import math
import dlib
import PIL
import operator
import datetime
import itertools
import statistics
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from imutils import face_utils
from keras.layers import Dense
from sklearn.svm import SVC

matplotlib.rcParams['figure.figsize'] = 20,16


# Declaration of functions
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def print_image(image,title=''):
    if verbosity==False:
        return
    figure = plt.figure()
    plt.title(title)
    plt.imshow(image)

def line_centre(x,y):
    return [int(round((x[0]+y[0])/2)),int(round((x[1]+y[1])/2))]

def golden_ratio(shape):
    GR = 1.62
    err_list = [] # sum of square errors
    
    # odnos debljina usne - sirina nosa
    x1 = get_lips_height(shape)
    x2 = get_nose_width(shape)
    err_list.append((GR-x1/x2)**2)
    
    # odnos sirina nosa i razmak usana od nosa
    x1 = get_nose_width(shape)
    mouth_centre = line_centre(shape[62],shape[66])
    x2 = euclidean_distance(shape[33][0],shape[33][1],mouth_centre[0],mouth_centre[1])
    err_list.append((GR-x1/x2)**2)
    
    # odnos razmaka od vrha nosa do brade i razmak usana od vrha brade
    x1 = euclidean_distance(shape[33][0],shape[33][1],shape[8][0],shape[8][1])
    x2 = euclidean_distance(shape[8][0],shape[8][1],mouth_centre[0],mouth_centre[1])
    err_list.append((GR-x1/x2)**2)
    
    # jednak odnos sirine ociju i sirine medjuocnog prostora
    x1 = euclidean_distance(shape[36][0],shape[36][1],shape[39][0],shape[39][1])
    x2 = euclidean_distance(shape[42][0],shape[42][1],shape[45][0],shape[45][1])
    eye_width = (x1+x2)/2
    x3 = get_eye_distance(shape)
    
    err = eye_width/x3
    if (err<0):
        err = x3/eye_width
    
    err_list.append(err**2)
    
    # jednak odnos po visini
    left_eye = line_centre(shape[36],shape[39])
    right_eye = line_centre(shape[42],shape[45])
    eyes = line_centre(left_eye,right_eye)
    
    x1 = euclidean_distance(shape[33][0],shape[33][1],eyes[0],eyes[1])
    x2 = euclidean_distance(shape[33][0],shape[33][1],shape[8][0],shape[8][1])
    
    err = x1/x2
    if (err<0):
        err = x2/x1
    
    err_list.append(err**2)
    
    return sum(err_list)

def get_nose_width(shape):
    shape = shape.tolist()
    high_point = shape[31]
    low_point = shape[35]
    return euclidean_distance(high_point[0],high_point[1],low_point[0],low_point[1])

def get_lips_height(shape):
    shape = shape.tolist()
    high_point = shape[51]
    low_point = shape[57]
    return euclidean_distance(high_point[0],high_point[1],low_point[0],low_point[1])

def get_eye_distance(shape):
    shape = shape.tolist()
    left_point = shape[39]
    right_point = shape[42]
    return euclidean_distance(left_point[0],left_point[1],right_point[0],right_point[1])

def euclidean_distance(x1,y1,x2,y2):
    return math.hypot(x2 - x1, y2 - y1)

def average(l): 
    return sum(l)/len(l)
    
    
# Declaration of variables
verbosity=False

image_list = {}
pos_list_golden_ratio = []
neg_list_golden_ratio = []

pos_features = []
neg_features = []
labels = []

image_dir_pos = 'slike/positive'
image_dir_neg = 'slike/negative'

print('\n*** An algorithm that extracts the features from face image and inserts them in SVM and KNN for \'smart\' detection of drug addicts ***')
start_time = datetime.datetime.utcnow()
print('\nProcess started at ', start_time,'\n')

# Algorithm 1: Download all images from the folder
for img_name in os.listdir(image_dir_pos):
    img_path = os.path.join(image_dir_pos, img_name)
    img = load_image(img_path)
    img_name = "pos_" + img_name
    image_list[img_name]=img
    
for img_name in os.listdir(image_dir_neg):
    img_path = os.path.join(image_dir_neg, img_name)
    img = load_image(img_path)
    img_name = "neg_" + img_name
    image_list[img_name]=img

# Algorithm 2: Image preprocessing
#for img in image_list:
#    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    temp_list.append(image_hsv)

#image_list = temp_list

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

for name in image_list:
    image = image_list[name]
    original = image.copy()
    features = []
    
    if verbosity==True:
        print('\nThe image being processed is ' + name)
    
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Algorithm 3: Face landmarks extraction
    rects = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    # Draw on our image, all the finded cordinate points (x,y) 
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)        
    
    # Show the image with landmarks
    print_image(image,name)
    
	# Algorithm 5: Golden ratio
    gr = golden_ratio(shape)
    if verbosity==True:
        print('The deviation from the Golden Ratio is ' + str(gr) + ' (less is better, ideally 1)')
    features.append(gr)
    
    # Algorithm 8: Preparation for classification
    img_type = name.split("_", 1)[0]
    if (img_type=="pos"):
        pos_features.append(features)
        labels.append(1)
        pos_list_golden_ratio.append(gr)
    else:
        neg_features.append(features)
        labels.append(0)
        neg_list_golden_ratio.append(gr)

print('\nThe results (negative):')
print('The average deviation from the Golden Ratio is ' + str(average(neg_list_golden_ratio)) + ' (less is better, ideally 1)')

print('\nThe results (positive):')
print('The average deviation from the Golden Ratio is ' + str(average(pos_list_golden_ratio)) + ' (less is better, ideally 1)')


# Algorithm 8: Preparation for classification
pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

end_prep_time = datetime.datetime.utcnow()

#  Algorithm 9: SVM classification
start_svm_time = datetime.datetime.utcnow()
svm = SVC(kernel='linear', probability=True) 
svm.fit(x_train, y_train)
y_train_pred = svm.predict(x_train)
y_test_pred = svm.predict(x_test)
print("\nSVM Train accuracy: ", accuracy_score(y_train, y_train_pred))
print("SVM Validation accuracy: ", accuracy_score(y_test, y_test_pred))
end_svm_time = datetime.datetime.utcnow()

# Algorithm 10: KNN classification
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(x_train, y_train)
y_train_pred = knn.predict(x_train)
y_test_pred = knn.predict(x_test)
print("\nKNN Train accuracy: ", accuracy_score(y_train, y_train_pred))
print("KNN Validation accuracy: ", accuracy_score(y_test, y_test_pred))
end_knn_time = datetime.datetime.utcnow()


# Algorithm 11: Deep learning
model = Sequential()
n_cols = x_train.shape[1]

model.add(Dense(250, activation='relu', input_shape=(n_cols,)))
model.add(Dense(250, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

early_stopping_monitor = EarlyStopping(patience=3)
model.fit(x_train, y_train, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
print("\nDeep learning Train accuracy: ", accuracy_score(y_train, y_train_pred))
print("Deep learning Validation accuracy: ", accuracy_score(y_test, y_test_pred))


end_time = datetime.datetime.utcnow()
print('\nProcess ended at ', end_time)
print('The process lasted ', end_time-start_time)

print('\nThe preparation of images lasted ', end_prep_time-start_time)
print('The SVM classification of images lasted ', end_svm_time-start_svm_time)
print('The KNN classification of images lasted ', end_knn_time-end_svm_time)
print('Deep learning classification of images lasted ', end_time-end_knn_time)


time_array = []
avg_array = []

for j in range(1000):
    for i in range(20):
        x = x_train[i].reshape(1, -1)

        start_time = datetime.datetime.utcnow()
        y = svm.predict(x)
        end_time = datetime.datetime.utcnow()

        time = end_time-start_time
        time_array.append(time.microseconds)
        
    avg_array.append(average(time_array))

print('The classification for SVM lasted in average (ms)', str(average(avg_array)/1000))

time_array = []
avg_array = []

for j in range(1000):
    for i in range(20):
        x = x_train[i].reshape(1, -1)

        start_time = datetime.datetime.utcnow()
        y = knn.predict(x)
        end_time = datetime.datetime.utcnow()

        time = end_time-start_time
        time_array.append(time.microseconds)
        
    avg_array.append(average(time_array))

print('The classification for KNN lasted in average (ms)', str(average(avg_array)/1000))

time_array = []
avg_array = []

for j in range(1000):
    for i in range(20):
        x = x_train[i].reshape(1, -1)

        start_time = datetime.datetime.utcnow()
        y = model.predict(x)
        end_time = datetime.datetime.utcnow()

        time = end_time-start_time
        time_array.append(time.microseconds)
        
    avg_array.append(average(time_array))

print('The classification for Deep learning lasted in average (ms)', str(average(avg_array)/1000))
