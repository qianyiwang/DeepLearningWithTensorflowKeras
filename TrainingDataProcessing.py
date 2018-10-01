from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

trainDataDir = 'data/train/'
testDataDir = 'data/test/'
IMG_SIZE = 50
categories = ['cat','dog']

def createTrainingData():
    training_data = []
    for fileName in tqdm(os.listdir(trainDataDir)):
        className = fileName.split('.')[0] # find class name from image name
        path = os.path.join(trainDataDir,fileName)
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            resizeArr = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([resizeArr,categories.index(className)]) # append into training data set as a list of image array and label
        except Exception as e:
            print('Error Happened: ',e)

    random.shuffle(training_data) # random order
    x = []
    y = []
    for feature, label in training_data:
        x.append(feature)
        y.append(label)
    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # gray pic only has one chanel
    return x, y

x, y = createTrainingData()

import pickle
pickle_out = open("catvsdog_feature.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("catvsdog_label.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# if want to load data
# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)
#
# pickle_in = open("y.pickle","rb")
# y = pickle.load(pickle_in)
