import csv
import cv2
import numpy as np



# Use of python csv library to read and store the lines from the driving_log.csv file
lines = []
with open('../training_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# Loading the images and steering measurements
images = [] 
measurements = []
for line in lines:
	for i in range(3):
		path = line[i]
		image = cv2.imread(path) # Image Measurements --> Inputs
		images.append(image)
	
	# Create adjusted steering measurements for the side cameras images by implementing a correction
	correction = 0.2
	measurement = float(line[3]) # Steering Measurements --> Output Labels
	measurements.append(measurement)
	measurements.append(measurement + correction)
	measurements.append(measurement - correction)
	
# Expand the model and help it generalize better by augmenting the dataset 
# Flipping the images horizontally and inverting steering angles 
# More data
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1)) # 1 for flipping around y-axis
	augmented_measurements.append(measurement*(-1.0))

# Converting images and steering measurements to numpy arrays, since Keras needs numpy arrays as inputs
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Building neural network with Keras
import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Normalization of the data by adding a lambda layer
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3))) # this makes training and validation loss much smaller
# Cropping 50 rows pixels from the top and 20 rows pixels from the bottom
model.add(Cropping2D(cropping=((50,25),(0,0))))																																																																																																																																																																																																																																																																																																																										
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))	
model.add(Dense(1))

# Using Mean Squared Error for the loss function, this is a regression network and not a classification network
model.compile(loss = 'mse', optimizer = 'adam')

# Shuffle the data and use 20% of them for validation
# Default number of epochs in Keras is 10, but in order to prevent overfitting I used 3 epochs 
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)

# Save the model, so as to use it for driving my car in the simulator 
model.save('model_nvidia_1counterclock.h5')

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
