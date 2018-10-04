from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

class CreateTrainingData:

    testString = 'hi there'
    x = []
    y = []

    def __init__(self, trainDataDir, testDataDir, imageSize, categories):
        self.trainDataDir = trainDataDir
        self.testDataDir = testDataDir
        self.IMG_SIZE = imageSize
        self.categories = categories

    def createTrainingData(self):
        training_data = []
        for fileName in tqdm(os.listdir(self.trainDataDir)):
            className = fileName.split('.')[0] # find class name from image name
            path = os.path.join(self.trainDataDir,fileName)
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                resizeArr = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                training_data.append([resizeArr,self.categories.index(className)]) # append into training data set as a list of image array and label
            except Exception as e:
                print('Error Happened: ',e)

        random.shuffle(training_data) # random order

        for feature, label in training_data:
            self.x.append(feature)
            self.y.append(label)
        self.x = np.array(self.x).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1) # gray pic only has one chanel
        self.y = np.array(self.y)
        return self.x, self.y

    def saveData(self):
        import pickle
        pickle_out = open("catvsdog_feature.pickle","wb")
        pickle.dump(self.x, pickle_out)
        pickle_out.close()

        pickle_out = open("catvsdog_label.pickle","wb")
        pickle.dump(self.y, pickle_out)
        pickle_out.close()

    def isCreated(self):
        for name in os.listdir():
            if '.pickle' in name:
                return True
        return False

    def fetchData(self, path):
        import pickle
        pickle_in = open(path,'rb')
        x = pickle.load(pickle_in)
        return x

    def pin(self):
        print(self.testString)
