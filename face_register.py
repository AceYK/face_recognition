import os
import cv2

name = input("What's your name? ")
path = '{}/images/{}'.format(os.getcwd(), name)

if(not(os.path.isdir(path))):
    os.mkdir(path)

face_cascade = cv2.CascadeClassifier('{}/cascades/data/haarcascade_frontalface_default.xml'.format(os.getcwd()))

cap = cv2.VideoCapture(0)

counter = 1
while(counter <= 5):
    scale = 80

    if cv2.waitKey(20) & 0xFF == ord('y'):
        print("Taking photo {}".format(counter))
        img_item = "{}/images/{}/{}.png".format(os.getcwd(), name, counter)
        counter += 1
        cv2.imwrite(img_item, roi_color)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y-scale:y+h+scale, x:x+w]
        roi_color = frame[y-scale:y+h+scale, x:x+w]

        color = (255, 0, 0) #BGR
        stroke = 2 #thick of line
        cv2.rectangle(frame, (x, y-scale), (x+w, y+h+scale), color, stroke)

    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()