import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# scales data between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1) 

# declear a model from tensorflow keras API
model = tf.keras.models.Sequential()
# add input layer to model
model.add(tf.keras.layers.Flatten())
# add hidden layers to model, with 123 nodes
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# add output layer to model, 10 nodes since there are 10 classes
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# config the model about its optimization method, and lost function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# train the model
model.fit(x_train, y_train, epochs=3)

# test the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

if val_acc>=0.95:
    model.save('epic_num_reader.model')
    print('model saved!')
# load model
# new_model = tf.keras.models.load_model('epic_num_reader.model')
