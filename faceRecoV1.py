import cv2 
import sys
import os
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np


# Load the model
model = tensorflow.keras.models.load_model('model/model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

class_names = ['Rauchdi', 'Mohammed','Youness']

cam = cv2.VideoCapture(0)

face_dete = cv2.CascadeClassifier('haarcascade_frontalface.xml')

while True :
	ret, img = cam.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces= face_dete.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		
		cv2.rectangle(img, (x-20,y-20), (x+w+20, y+h+20), (155,155,155), 2)
		# Extract the frame of each card 
		face = img[y:y+h,x:x+w]
		# Save cards 
		cv2.imwrite(f"Faces/face.jpg", face)

		image = Image.open('Faces/face.jpg')

		size = (224, 224)
		image = ImageOps.fit(image, size, Image.ANTIALIAS)

		#turn the image into a numpy array
		image_array = np.asarray(image)

		# Normalize the image
		normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

		# Load the image into the array
		data[0] = normalized_image_array

		# run the inference
		prediction = model.predict(data)
		# print(prediction, type(prediction))
		pred_id=np.argmax(prediction,axis=-1)
		# print(pred_id)
		pred_label=class_names[int(pred_id)]

		PC=int(prediction[0][pred_id]*100)
	    	
		if PC >60: 
			name= str(pred_label) + ":" + str(PC)    	
		else:
			name = 'Unkown'
		cv2.putText(img,name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2,cv2.LINE_AA)
	          





	cv2.imshow('image', img)

	k = cv2.waitKey(1) & 0xff

	if k == 27:
		break

cam.release()
cv2.destroyAllWindows()