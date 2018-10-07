import TrafficSignDataProcessing as dataProc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle

TRAINING_PATH = 'data/tsrd-train/'
TESTING_PATH = 'data/TSRD-Test/'
feature_train_path = 'pickle/traffic_sign_x_training.pickle'
label_train_path = 'pickle/traffic_sign_y_training.pickle'
feature_test_path = 'pickle/traffic_sign_x_testing.pickle'
label_test_path = 'pickle/traffic_sign_y_testing.pickle'

# dataProc.createDataSet(TRAINING_PATH, TESTING_PATH)

def readData(path):
	pickle_in = open(path,'rb')
	data = pickle.load(pickle_in)
	return data

def train():
	feature = readData(feature_train_path)
	label = readData(label_train_path)
	feature = feature/255
	model = Sequential()
	model.add(Conv2D(256, (3, 3), input_shape=feature.shape[1:], activation=tf.nn.relu)) # feature.shape = (25000, 50, 50, 1); input_shape should be 50 of 50 dementions vectors
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), activation=tf.nn.relu))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors for the Dense layers

	model.add(Dense(64, activation=tf.nn.relu))

	model.add(Dense(len(set(label)), activation=tf.nn.softmax)) # output layer

	model.compile(loss='sparse_categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	model.fit(feature, label, batch_size=40, epochs=3, validation_split=0.3)

train()