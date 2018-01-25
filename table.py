import curses
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
import time
from PIL import Image

def return_num(ar):
	max_ = ar.max()
	i = 0
	for x in ar:
		if x == max_:
			return i
		i+= 1
	return 0

def paint_num(m):
	x = 0
	y = 0
	z = 0
	curses.start_color()
	curses.init_pair(2, curses.COLOR_WHITE,curses.COLOR_GREEN)
	if m < 10:
		l = '| ' + str(m) + ' |'
	elif m < 100:
		l = '| ' + str(m) + '|'
	elif m < 1000:
		l = '|' + str(m) + '|'
	else:
		l = str(z)

	while y < 43:
		x = 0
		while x < 138:
			if z == m:
				myscreen.addstr(y,x,l,curses.color_pair(2))
				return
			x += 6
			z += 1
		y += 2
	

def paint_a():
	x = 0
	y = 0
	z = 0
	curses.start_color()
	curses.init_pair(1, curses.COLOR_WHITE,curses.COLOR_BLACK)
	while y < 43:
		x = 0
		while x < 138:
			if z < 10:
				l = '| ' + str(z) + ' |'
			elif z < 100:
				l = '| ' + str(z) + '|'
			elif z < 1000:
				l = '|' + str(z) + '|'
			else:
				l = str(z)
			myscreen.addstr(y,x,l,curses.color_pair(1))
			x += 6
			z += 1
		y += 2
def MainInKey():
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	cap = cv2.VideoCapture(0)
	xl = 0
	yl = 0
	yr = 0
	xr = 0
	ew = 50
	eh = 50
	w = 250
	h = 250
	Ox = 0
	img_rows, img_cols = 48, 96
	Oy = 0
	num = 0
	ok = 0
	# if os.path.exists("./model.json") and os.path.exists("./model.h5"):
	# 	print("Load model")
	# 	json_file = open('model.json', 'r')
	# 	loaded_model_json = json_file.read()
	# 	json_file.close()
	# 	model = model_from_json(loaded_model_json)
	# 	model.load_weights("model.h5")
	# 	model.compile(loss='categorical_crossentropy', optimizer='adadelta')
	# else:
	# 	print("No model")
	# 	exit()
	key = 'X'
	paint_a()
	myscreen.refresh()  
	while key != ord('q') and key != ord('Q'):
		ret, img = cap.read()
		cv2.flip(src = img,  dst = img, flipCode = 0)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.6, 5, 150)
		roi_color = img[Oy:Oy+h, Ox:Ox+w]
		if ok < 50:
			for (x,y,wdel,hdel) in faces:
				ok += 1
				if (not (Ox - 2 < x and Ox + 2 > x)) or (not (Oy - 2 < y and Oy + 2 > y)):

					Ox = x
					Oy = y
					cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
					roi_gray = gray[y:y+h, x:x+w]
					roi_color = img[y:y+h, x:x+w]
				
					eyes = eye_cascade.detectMultiScale(roi_gray)
					j = 0
					poy = w / 4
					pox = w / 5

					for (ex,ey,ewdel,ehdel) in eyes:
						if (w / 2) > ex:#LEFT
							if ex + 20 > pox and ex - 20 < pox and ey + 20 > poy and ey - 20 < poy:#BIG DELTA
								if not (xl - 10 < ex and xl + 10 > ex):#If move
									xl = ex
									yl = ey
						else:#RIGTH
							poxr = (w - w / 5) - ew
							if ex + 30 > poxr and ex - 30 < poxr and ey + 30 > poy and ey - 30 < poy:#BIG DELTA
								if not (xr - 5 < ex and xr + 5 > ex):
									xr = ex
									yr = ey
				
		cv2.rectangle(img,(Ox,Oy),(Ox+w,Oy+h),(255,0,0),2)	
		cv2.rectangle(roi_color,(xl,yl),(xl+ew,yl+eh),(0,255,0),1)
		cv2.rectangle(roi_color,(xr,yr),(xr+ew,yr+eh),(0,255,0),1)

		cv2.imshow('img',roi_color)
		key = myscreen.getch()
		# k = cv2.waitKey(30) & 0xff
		if key == ord('\n'):
			paint_a()
			# takephoto
			cv2.imwrite('l.jpeg',roi_color[yl+1:yl-1 + eh,xl+1:xl-1 + ew])
			cv2.imwrite('r.jpeg',roi_color[yr+1:yr-1 + eh,xr+1:xr-1 + ew])
			img1 = Image.open('l.jpeg')
			img2 = Image.open('r.jpeg')
			img_to_save = Image.new('RGB', (96, 48))
			img_to_save.paste(img1, (0,0))
			img_to_save.paste(img2, (48,0))
			img_to_save.save('x.jpeg',"JPEG")
			#endphoto
			img_to_save = cv2.imread('x.jpeg', 0)
			xax = []
			xax.append(cv2.resize(img_to_save, (96, 48)))
			xax = np.array(xax, dtype=np.uint8)
			xax = xax.reshape(xax.shape[0], 1, img_rows, img_cols)
			xax = xax.astype('float32')
			xax /= 255
			res = model.predict(xax)
			paint_num(return_num(res[0]))
			myscreen.refresh()

	curses.endwin()
	cap.release()
	cv2.destroyAllWindows() 
	exit()

#  MAIN LOOP
try:
	#init cam
	myscreen = curses.initscr()
	curses.curs_set(0)
	curses.cbreak(); 
	curses.noecho();
	MainInKey()
finally:
	curses.endwin()