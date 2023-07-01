import os
import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['AishwaryaRai', 'BlakeLively', 'JimCarrey', 'MorganFreeman', 'RyanReynolds',
          'ShahRukhKhan', 'ShraddhaKapoor']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('/Users/pranayjaju/Desktop/Coding/Python/OpenCV/Faces/Test/BlakeLively/4.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', img)

faces_rect = haar_cascade.detectMultiScale(gray, 1.2, 5)

print(f'faces rect: {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label: {people[label]} with confidence of: {confidence}')

    cv.putText(img, str(people[label]), (x,y), cv.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

cv.imshow('Detected', img)
cv.waitKey(0)