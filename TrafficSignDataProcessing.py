
def prepareData(dir):
    import os
    from tqdm import tqdm
    import cv2
    import random
    import numpy as np

    IMAGE_SIZE = 70
    dataSet = []
    x = []
    y = []
    for fileName in tqdm(os.listdir(dir)):
        filePath = os.path.join(dir,fileName)
        try:
            label = fileName.split('_')[0]
            image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            dataSet.append([image,label])

        except Exception as e:
            print('Error occured when process image:',e)

    random.shuffle(dataSet)
    for features, label in dataSet:
        x.append(features)
        y.append(label)
    x = np.array(x).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    y = np.array(y)

    return x, y

def saveData(data, name):
    import pickle
    pickle_out = open(name,"wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

def test():
    print('test success')

def createDataSet(trainPath, testPath):
    x_train, y_train = prepareData(trainPath)
    saveData(x_train, 'pickle/traffic_sign_x_training.pickle')
    saveData(y_train, 'pickle/traffic_sign_y_training.pickle')
    x_test, y_test = prepareData(testPath)
    saveData(x_test, 'pickle/traffic_sign_x_testing.pickle')
    saveData(y_test, 'pickle/traffic_sign_y_testing.pickle')
