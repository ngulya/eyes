from keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
import cv2
import os
# import curses
# import table

# 	#  MAIN LOOP
# try:
# 	table.myscreen = curses.initscr()
# 	table.curses.curs_set(0)
# 	table.curses.cbreak(); 
# 	table.curses.noecho(); 
# 	table.MainInKey()
# finally:
# 	table.curses.endwin()

answer = np.arange(0, 506, 1)
batch_size = 32
nb_classes = 506
nb_epoch = 2
img_rows, img_cols = 48, 96
nb_filters = 32
nb_pool = 2
nb_conv = 3

def load_train():
	X_t = []
	Y_t = []
	x = 0
	while x < 506:
		img = cv2.imread('Data/'+str(x) + '.jpeg', 0)
		X_t.append(cv2.resize(img, (96, 48)))
		Y_t.append(x)
		x+=1
	return X_t, Y_t

train_data, train_target = load_train()

batch_size = 46
nb_classes = 506
nb_epoch = 10#20 min 30 ep
img_rows, img_cols = 48, 96
nb_filters = 13
nb_pool = 2
nb_conv = 6

train_data = np.array(train_data, dtype=np.uint8)
train_target = np.array(train_target, dtype=np.uint8)

train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
train_target = np_utils.to_categorical(train_target, nb_classes)

train_data = train_data.astype('float32')
train_data /= 255
model = Sequential()
if os.path.exists("./model.json") and os.path.exists("./model.h5"):
	print("Load model")
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model.h5")
else:
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
			border_mode='valid',
			input_shape=(1, img_rows, img_cols)))
	model.add(Activation('relu'))
	model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1012))
	model.add(Activation('relu'))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')
# model.fit(train_data, train_target, batch_size=batch_size, epochs=nb_epoch, validation_data=(train_data, train_target))
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# model.save_weights("model.h5")
# print("Saved model to disk")



def return_num(ar):
	max_ = ar.max()
	i = 0
	for x in ar:
		if x == max_:
			return i
		i+= 1
	return 0
img = cv2.imread('x.jpeg', 0)
xax = []
xax.append(cv2.resize(img, (96, 48)))

xax = np.array(xax, dtype=np.uint8)
xax = xax.reshape(xax.shape[0], 1, img_rows, img_cols)
xax = xax.astype('float32')
xax /= 255


res = model.predict(xax)
# print(res[0])
# g = res[0].max()
# print(g)
print(return_num(res[0]))
# print(return_num(res[1]))