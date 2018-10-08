from keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc

np.random.seed(100)

img = cv2.imread('Data/0.jpeg')
img_rows, img_cols = img.shape[0], img.shape[1]

def load_train():
	X_t = []
	Y_t = []
	x = 0
	mxs = 0
	while x < 506:
		img = cv2.imread('Data/'+str(x) + '.jpeg')
		X_t.append(img)
		Y_t.append(x)
		x += 1
	return train_test_split(X_t, Y_t, test_size = 0.2, random_state = 71)

X_train, X_test, y_train, y_test  = load_train()

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

X_test = X_test.astype(float)
X_test /= 255

X_train = X_train.astype(float)
X_train /= 255

y_train = y_train.astype(float)
y_train /= 506

y_test = y_test.astype(float)
y_test /= 506

input_shape = (img_rows, img_cols, 3)
print input_shape
if os.path.exists("./model_weights/model_t.json") and os.path.exists("./model_weights/model_t.h5"):
	print("Load model")
	json_file = open('model_weights/model_t.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model_weights/model_t.h5")
	model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
else:
	model = Sequential()
	model.add(Conv2D(20, kernel_size=(5, 5),
			activation='relu',
			input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(20, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Conv2D(20, (5, 5), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(200, activation='sigmoid'))
	model.add(Dense(100, activation='sigmoid'))
	model.add(Dense(30, activation='sigmoid'))
	model.add(Dense(5, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
	print(model.summary())
	model.fit(X_train, y_train, epochs=38)

	model_json = model.to_json()
	with open("model_weights/model_t.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model_weights/model_t.h5")
	print("Saved model to disk")
model.fit(X_train, y_train, epochs=38)

predicted = model.predict(X_test)
alls = len(predicted)
i = 0
while i < alls:
	p = int(predicted[i] * 506)
	y = int(y_test[i] * 506)
	print p, ' == ',y
	i += 1

y_test = y_test.astype(float)
print 'mean_squared_error = ', mean_squared_error(y_test, predicted)
if raw_input('save:') == 'y':
	model_json = model.to_json()
	with open("model_weights/model_t.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model_weights/model_t.h5")
	print("Saved model to disk")
