import face_recognition as fr
import cv2 as cv
import os
import re

known_faces_dir = '/Users/pranayjaju/Desktop/Coding/Python/face_rec/Faces/Known'
unknown_faces_dir = '/Users/pranayjaju/Desktop/Coding/Python/face_rec/Faces/Unknow'

known_faces_encoding = []
known_names = []

print(f'Encoding Known Faces')
for name in os.listdir(known_faces_dir):
    #* Mac folders have a hidden folder - '.DS_Store' which has to be skipped
    if re.match(r'\S+DS_Store$', name):
        continue
    img_path = os.path.join(known_faces_dir, name)
    img = fr.load_image_file(img_path)
    encoding = fr.face_encodings(img)[0]
    known_faces_encoding.append(encoding)
    known_names.append(name.split('.')[0])
    # print(f'{name.split(".")[0]} encoded!!!')

print(f'Processing Unknown Faces')
for file in os.listdir(unknown_faces_dir):
    #* Mac folders have a hidden folder - '.DS_Store' which has to be skipped
    if re.match(r'\S+DS_Store$', file):
        continue
    img_path = os.path.join(unknown_faces_dir, file)
    img = fr.load_image_file(img_path)
    locations = fr.face_locations(img)
    encodings = fr.face_encodings(img, locations)
    img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    # print(f'location of {file}: {locations}')
    # print(f'Distance of {file}: {fr.face_distance(known_faces_encoding, encodings[0])}')

    for (top,right,bottom,left), face_encoding in zip(locations, encodings):
        result = fr.compare_faces(known_faces_encoding, face_encoding)
        match = 'Unknown Person'
        if True in result:
            match = known_names[result.index(True)]
            print(f'Match Found: {match}')

            cv.rectangle(img_bgr, (left,top), (right,bottom), (255,0,255), 2)
            cv.putText(img_bgr, match, (left+15,bottom+15), cv.FONT_HERSHEY_COMPLEX, 1, (210,210,210), 2)

        # else:
        #     cv.rectangle(img_bgr, (left,top), (right,bottom), (0,255,0), 2)
        #     cv.putText(img_bgr, match, (left+15,bottom+15), cv.FONT_HERSHEY_COMPLEX, 1, (210,210,210), 2)
    
    cv.imshow(file, img_bgr)
    cv.waitKey(10000)