import numpy as np
import time
import cv2
from PIL import Image

# num = 0
# while(1):
# 	k = cv2.waitKey(30) & 0xff
# 	if k == 27:
# 		break
	# img1 = Image.open('left/l'+str(num)+'.jpeg')
	# img2 = Image.open('right/r'+str(num)+'.jpeg')
	# img.paste(img1, (0,0))
	# img.paste(img2, (48,0))
	# img.save('Data/'+str(num)+'.jpeg',"JPEG")
# 	if num == 505:
# 		break
# 	print(num)
# 	num += 1


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
Oy = 0
num = 0
ok = 0
while 1:
	ret, img = cap.read()

	cv2.flip(src = img,  dst = img, flipCode = 0)
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# faces = face_cascade.detectMultiScale(gray, 1.6, 5, 150)
	# roi_color = img[Oy:Oy+h, Ox:Ox+w]
	
	# if ok < 50:
	# 	for (x,y,wdel,hdel) in faces:
	# 		ok += 1
	# 		if (not (Ox - 2 < x and Ox + 2 > x)) or (not (Oy - 2 < y and Oy + 2 > y)):

	# 			Ox = x
	# 			Oy = y
	# 			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	# 			roi_gray = gray[y:y+h, x:x+w]
	# 			roi_color = img[y:y+h, x:x+w]
			
	# 			eyes = eye_cascade.detectMultiScale(roi_gray)
	# 			j = 0
	# 			poy = w / 4
	# 			pox = w / 5

	# 			for (ex,ey,ewdel,ehdel) in eyes:
	# 				if (w / 2) > ex:#LEFT
	# 					if ex + 20 > pox and ex - 20 < pox and ey + 20 > poy and ey - 20 < poy:#BIG DELTA
	# 						if not (xl - 10 < ex and xl + 10 > ex):#If move
	# 							xl = ex
	# 							yl = ey
	# 				else:#RIGTH
	# 					poxr = (w - w / 5) - ew
	# 					if ex + 30 > poxr and ex - 30 < poxr and ey + 30 > poy and ey - 30 < poy:#BIG DELTA
	# 						if not (xr - 5 < ex and xr + 5 > ex):
	# 							xr = ex
	# 							yr = ey
			
	# cv2.rectangle(img,(Ox,Oy),(Ox+w,Oy+h),(255,0,0),2)	
	# print(xl,yl, xr, yr)
	# xl = 60
	# xr = 140
	# yl = 75
	# yr = 75

	# cv2.rectangle(roi_color,(xl,yl),(xl+ew,yl+eh),(0,255,0),1)
	# cv2.rectangle(roi_color,(xr,yr),(xr+ew,yr+eh),(0,255,0),1)
	cv2.imshow('img',img)
	# cv2.imshow('img',roi_color)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	if k == 13:
		cv2.imwrite('l.jpeg',roi_color[yl+1:yl-1 + eh,xl+1:xl-1 + ew])
		cv2.imwrite('r.jpeg',roi_color[yr+1:yr-1 + eh,xr+1:xr-1 + ew])
		img1 = Image.open('l.jpeg')
		img2 = Image.open('r.jpeg')
		img = Image.new('RGB', (96, 48))
		img.paste(img1, (0,0))
		img.paste(img2, (48,0))
		img.save('x.jpeg',"JPEG")

cap.release()
cv2.destroyAllWindows()