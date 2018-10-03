from TrainingDataProcessing import CreateTrainingData
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

TRAINING = 'catvsdog_feature.pickle'
LABEL = 'catvsdog_label.pickle'

createTraining = CreateTrainingData('data/train/', 'data/test/', 50, ['cat','dog'])
if not createTraining.isCreated():
    createTraining.createTrainingData()
    createTraining.saveData()
    print('Trainind data has been created and saved')
else:
    print('Trainind data already exist')

x = createTraining.fetchData(TRAINING)
y = createTraining.fetchData(LABEL)
x_norm = tf.keras.utils.normalize(x, axis=-1)

# print(x_norm.shape)
# print(x_norm.shape[1:])
model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=x_norm.shape[1:], activation='relu')) # x_norm.shape = (25000, 50, 50, 1); input_shape should be 50 of 50 dementions vectors
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors for the Dense layers

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid')) # output layer

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_norm, y, batch_size=32, epochs=3, validation_split=0.3)
model.save('catvsdog_cnn.model')
