import numpy as np
import cv2
import pickle
import os

# Change the path when you are free
face_cascade = cv2.CascadeClassifier('{}/cascades/data/haarcascade_frontalface_default.xml'.format(os.getcwd()))
# eye_cascade = cv2.CascadeClassifier('{}/cascades/data/haarcascade_eye.xml'.format(os.getcwd()))
# smile_cascade = cv2.CascadeClassifier('{}/cascades/data/haarcascade_smile.xml'.format(os.getcwd()))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

label = {}
with open("labels.pickle", 'rb') as f:
	og_label = pickle.load(f)
	label = {v:k for k,v in og_label.items()}

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x, y, w, h) in faces:
		# print(x, y, w, h)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]

		id_, conf = recognizer.predict(roi_gray)
		if conf >= 0 and conf <= 85:
			# print(conf, label[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = "{} {}".format(label[id_], conf)
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		color = (255, 0, 0) #BGR
		stroke = 2 #thick of line
		x_end = x + w
		y_end = y + h
		cv2.rectangle(frame, (x, y), (x_end, y_end), color, stroke)

	 #    #for eyes
		# eyes = eye_cascade.detectMultiScale(roi_gray)
		# for (ex, ey, ew, eh) in eyes:
		# 	cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)

		# #for smile
		# smile = smile_cascade.detectMultiScale(roi_gray)
		# for (ex, ey, ew, eh) in smile:
		# 	cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)

	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()