import re
import os
import cv2 as cv
import numpy as np

people = ['AishwaryaRai', 'BlakeLively', 'JimCarrey', 'MorganFreeman', 'RyanReynolds',
          'ShahRukhKhan', 'ShraddhaKapoor']

DIR = r'/Users/pranayjaju/Desktop/Coding/Python/OpenCV/Faces/Train' #*Base directory/path

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = [] #* Image list
labels = [] #* Images corresponding label aka person's name/index

def train():
    i = 0
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            #* few folders have .DS_Store files which were giving empty image issues
            if re.match(r'\S+DS_Store$', img_path):
                continue
            img_array = cv.imread(img_path)
            print(f'img_path({i}): {img_path}')
            i += 1
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, 1.2, 5)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)

train()
print(f'********Training finished********')

features = np.array(features, dtype='object')
labels = np.array(labels)

print(f'features: {len(features)}')
print(f'labels: {len(labels)}')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')

cv.imshow(str(labels[120])+str(120), features[120])
cv.imshow(str(labels[6])+str(6), features[6])
cv.imshow(str(labels[21])+str(21), features[21])
cv.imshow(str(labels[95])+str(95), features[95])
cv.imshow(str(labels[69])+str(69), features[69])
cv.waitKey(0)