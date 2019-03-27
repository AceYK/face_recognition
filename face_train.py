import os
import numpy as np
import cv2
import pickle
from PIL import Image, ImageEnhance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

# Change the path when you are free
face_cascade = cv2.CascadeClassifier('{}/cascades/data/haarcascade_frontalface_default.xml'.format(os.getcwd()))
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_label = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			if label in label_ids:
				pass
			else:
				label_ids[label] = current_id
				current_id += 1

			id_ = label_ids[label]
			# Convert image into number (numpy array)
			pil_image = Image.open(path)
			pil_image = ImageEnhance.Brightness(pil_image).enhance(1.8)
			pil_image = pil_image.convert("L") # Convert into grayscale
			## Resize
			# size = (550, 550)
			# pil_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(pil_image, "uint8") # Convert into numpy array

			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_label.append(id_)

with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_label))
recognizer.save("trainner.yml")