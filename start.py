import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#*Creating a blank image
# blank = np.zeros((600,750,3), dtype='uint8')
# cv.imshow('Blank', blank)
# cv.waitKey(0)

#*Reading and displaying an image
# img = cv.imread('PhotoCV/cat.jpg')
# cv.imshow('Cat', img)

# cv.waitKey(0)

#*Reading and displaying a video
# capture = cv.VideoCapture('PhotoCV/dog.mp4')

# while True:
#     isTrue, frame = capture.read()
    
#     cv.imshow('Dog Video',frame)

#     if cv.waitKey(20) and 0xFF == ord('d'):
#         break

# capture.release()
# cv.destroyAllWindows()

#*Resizing an img/video frame
# img = cv.imread('PhotoCV/dog.jpg')
# cv.imshow('Dog', img)

# img_resize = cv.resize(img, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
# cv.imshow('Dog resized',img_resize)

# cv.waitKey(0)

#*Resizing a video
# capture = cv.VideoCapture('PhotoCV/cat.mp4')

# while True:
#     isTrue, frame = capture.read()
#     frame_resized = cv.resize(frame, [1000,1000], interpolation=cv.INTER_AREA)
    
#     cv.imshow('Cat Video',frame)
#     cv.imshow('Cat Video Resized', frame_resized)

#     if cv.waitKey(20) and 0xFF == ord('d'):
#         break

# capture.release()
# cv.destroyAllWindows()

#!-------------------------------------------------------------------------------------
#!Drawing
#~OpenCV reads the color in BGR channels(not the traditional RGB)
# #*Painting an blank image a certain color
# blank[:] = 255,0,0 #*Painting the image Blue
# cv.imshow('Blank->Blue', blank)

# #~rows1:rows2, columns1:columns2
# blank[250:350, 300:400] = 0,255,0 #*Painting the image Green
# cv.imshow('Blue+Green Sq', blank)

# blank[150:200, 0:750] = 0,0,255 #*Painting the image Red
# cv.imshow('Blue+Green Sq+Red Block', blank)

# cv.waitKey(0)

#*Drawing a line
# cv.line(blank, (250,45), (700,550), color=(255,0,255), thickness=1, lineType=16)
# cv.arrowedLine(blank, (0, 400), (700,150), color=(255,0,0), thickness=3, line_type=16)
# cv.imshow('Line+ArrowedLine', blank)

# cv.waitKey(0)

#*Drawing a rectangle
#~(x1,y1), (x2,y2) but origin is top left, so assume quadrant 4
# cv.rectangle(blank, (0,0), (375,300), color=(255,255,0), thickness=cv.FILLED)
# cv.rectangle(blank, (375,300), (750,600), color=(0,255,255), thickness=4)
# cv.imshow('Rectangle', blank)

# cv.waitKey(0)

#*Drawing a circle and ellipse
# cv.circle(blank, center=(200,500), radius=150, color=(239,64,87), lineType=16)
# cv.ellipse(blank, center=(450,175), axes=(100,30), angle=90, startAngle=30, endAngle=270, color=(43,231,93), lineType=16)
# cv.imshow('Circle', blank)

# cv.waitKey(0)

#*Putting text in an image
# cv.putText(blank, 'Hi there', (100,200), fontFace=cv.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=2, color=(0,255,127))
# cv.imshow('Text', blank)

# cv.waitKey(0)

#!------------------------------------------------------------------------------------
#*Finding the edges
# img = cv.imread('PhotoCV/ny_sl.jpg')
# cv.imshow('Statue of Liberty',img)

# canny = cv.Canny(img, 125, 175)
# cv.imshow('Canny Edges of Statue of Liberty', canny)
# #~to reduce the no.of edges pass in the blurred image

# cv.waitKey(0)

#*Dilating an image
#~Increase the object area and accentuate the features
# img = cv.imread('PhotoCV/ny_sl.jpg')
# cv.imshow('Staute of Liberty', img)

# dilated = cv.dilate(img, (9,9), iterations=5)
# cv.imshow('Dialted Statue of Liberty',dilated)

# cv.waitKey(0)

#*Eroding an image
#~Erode away the boundary of foreground object and diminishes the features
# img = cv.imread('PhotoCV/ny_sl.jpg')
# cv.imshow('Staute of Liberty', img)

# eroded = cv.erode(img, (5,5), iterations=3)
# cv.imshow('Eroded Statue of Liberty',eroded)

# cv.waitKey(0)

#!------------------------------------------------------------------------------------
#!Transformations
#*Crop
# img = cv.imread('PhotoCV/lion.jpg')
# cv.imshow('Lion', img)

# cropped = img[(img.shape)[1]//2:(img.shape)[1], 0:(img.shape)[1]//2]
# cv.imshow('Cropped Lion', cropped)

# cv.waitKey(0)

#*Translation
# img = cv.imread('PhotoCV/ny_cp.jpg')
# cv.imshow('Central Park', img)

# def translate(img, x, y):
#     transMat = np.float32([[1,0,x],[0,1,y]])
#     dimensions = (img.shape[1], img.shape[0])
#     return cv.warpAffine(img,transMat,dimensions)

# #~ -x = left; +x = right
# #~ -y = up ; +y = down

# translated = translate(img, 500, 100)
# cv.imshow('Translated Central Park',translated)

# cv.waitKey(0)

#*Rotation
# img = cv.imread('PhotoCV/dog.jpg')
# cv.imshow('Dog', img)

# def rotate(img, angle, rotatePoint=None):
#     (height,width) = img.shape[:2]

#     if rotatePoint is None:
#         rotatePoint = (width//2,height//2)

#     rotMat = cv.getRotationMatrix2D(rotatePoint, angle, scale=1.0)

#     return cv.warpAffine(img, rotMat, (width,height))

# #~ +angle = anti-clockwise; -angle = clockwise
# rotated = rotate(img, 45)
# cv.imshow('Rotated Dog', rotated)

# cv.waitKey(0)

#*Flip
# img =cv.imread('PhotoCV/hedgehog.jpg')
# cv.imshow('Hedgehog', img)

# ver_flip = cv.flip(img, 0)
# cv.imshow('Vertical Flip', ver_flip)

# hori_flip = cv.flip(img, 1)
# cv.imshow('Horizontal Flip', hori_flip)

# both_flip = cv.flip(img, -1)
# cv.imshow('Ver+Hori Flip', both_flip)

# cv.waitKey(0)

#!Contours
#?From the perspective of basic CV, these could be understood as edges. But not exactly
# img = cv.imread('PhotoCV/ny_ts.jpg')
# cv.imshow('Times Square', img)

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray Times Square', gray)

# #~ this is one way of finding the contours
# blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
# cv.imshow('Blurred Times Square', blur)

# canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Edges of Times Square', canny)

# #~ this is another way of finding the contours
# #? threshold binarizing the image. In above case <125 = 0/black and >125 = 255/white
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('Binarized image of Times Square: Threshold of 125', thresh)

# contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# print(f'{len(contours)} contours found')

# #~ we can visualize the contours of the image on a blank image, if threshold is used
# blank = np.zeros(img.shape, dtype='uint8')
# cv.drawContours(blank, contours, -1, color=[255,0,255], thickness=1)
# cv.imshow('Contours', blank)

# cv.waitKey(0)

#!Color Spaces
#~ All the below shown conversions can be reversed too
# img = cv.imread('PhotoCV/ny_ts.jpg')
# cv.imshow('Times Square', img)

#*BGR to Gray Scale
# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray Times Square', gray_img)

#*BGR to HSV(Hue-Saturation Value)
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('HSV of Times Square', hsv)

# #*BGR to LAB/L*a*b
# lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('LAB of Times Square', lab)

# #*BGR to RGB
# rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.imshow('RGB of Times Square', rgb) 

#? the following code is just to show how the reading of an image in BGR gives
#? a different output when using a library that expects RGB image
# #~ use matplotlib to see how it displays the image differently
# fig = plt.figure(figsize=(10,5))

# fig.add_subplot(1,2,1)
# plt.imshow(img)
# plt.axis('off')
# plt.title('BGR of CV')

# fig.add_subplot(1,2,2)
# plt.imshow(rgb)
# plt.axis('off')
# plt.title('RGB of CV')
# plt.show()

# cv.waitKey(0)

#!Color Channels
#?The 3 colors used to make up every photo aka Blue, Green, Red
# img =cv.imread('PhotoCV/ny_ts.jpg')
# cv.imshow('Times Square', img)

# b,g,r = cv.split(img)

# cv.imshow('Blue',b)
# cv.imshow('Green',g)
# cv.imshow('Red',r)

# print(f'shape of image: {img.shape}')
# print(f'shape of blue: {b.shape}')
# print(f'shape of green: {g.shape}')
# print(f'shape of red: {r.shape}')

# merged = cv.merge([b,g,r])
# cv.imshow('Merged',merged)

#~ This helps us visualize the respective color channels
# blank = np.zeros(img.shape[:2], dtype='uint8')

# blue = cv.merge([b,blank,blank])
# green = cv.merge([blank,g,blank])
# red = cv.merge([blank,blank,r])

# cv.imshow('Blue',blue)
# cv.imshow('Green',green)
# cv.imshow('Red',red)

# cv.waitKey(0)

#!------------------------------------------------------------------------------------
#! Blurring
# img = cv.imread('PhotoCV/ny_sl.jpg')
# cv.imshow('Statue of Liberty',img)

# #*Average Blur
# average = cv.blur(img, (7,7))
# cv.imshow('Average Blur', average)

# # cv.waitKey(0)

# #*Gaussian Blur
# gaussian = cv.GaussianBlur(img, (7,7), 0)
# cv.imshow('Gaussian Blur',gaussian)

# # cv.waitKey(0)

# #*Median Blur
# median = cv.medianBlur(img, 7)
# cv.imshow('Median Blur', median)

# # cv.waitKey(0)

# #*Bilateral Blur
# bilateral = cv.bilateralFilter(img, 20, 60, 60)
# cv.imshow('Bilateral Blur', bilateral)

# # cv.waitKey(0)

#!------------------------------------------------------------------------------------
#!Bitwise Operations
# blank = np.zeros((400,400), dtype="uint8")

# rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
# circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

# cv.imshow('Rectangle', rectangle)
# cv.imshow('Circle', circle)

# #*Bitwise AND
# bitwise_and = cv.bitwise_and(rectangle, circle)
# cv.imshow('Bitwise AND', bitwise_and)

# #*Bitwise OR
# bitwise_or = cv.bitwise_or(rectangle, circle)
# cv.imshow('Bitwise OR', bitwise_or)

# #*Bitwise XOR
# bitwise_xor = cv.bitwise_xor(rectangle, circle)
# cv.imshow('Bitwise XOR', bitwise_xor)
# #~ OR = XOR + AND

# #*Bitwise NOT
# rec_not = cv.bitwise_not(rectangle)
# cir_not = cv.bitwise_not(circle)
# cv.imshow('Rectangle NOT', rec_not)
# cv.imshow('Circle NOT', cir_not)

# cv.waitKey(0)

#!------------------------------------------------------------------------------------
#!Masking
# img = cv.imread('PhotoCV/lion.jpg')
# cv.imshow('Lion', img)

# blank = np.zeros(img.shape[:2], dtype='uint8') #shape has to be same as that of img

# mask = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 500, 255, -1)
# cv.imshow('Mask', mask)

# masked = cv.bitwise_and(img,img,mask=mask)
# cv.imshow('Masked', masked)

# cv.waitKey(0)

#!------------------------------------------------------------------------------------
#!Histograms Computation
#? Can be computer for gray scale and RGB images
#~ the mask used has to be a single channel, binary format
# img = cv.imread('PhotoCV/cat.jpg')
# cv.imshow('Cat', img)

# blank = np.zeros(img.shape[:2], dtype='uint8')
# circle = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 500, 255, -1)

#*Grayscale Histogram
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# mask = cv.bitwise_and(gray,gray,mask=circle)
# cv.imshow('Mask', mask)

# gray_hist = cv.calcHist([gray],[0],mask,[256],[0,256])
# plt.figure()
# plt.plot(gray_hist)
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# pixels')
# plt.xlim([0,256])
# plt.show()

# cv.waitKey(0)

#*Color Histogram(RGB)
# mask = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 500, 255, -1)
# masked = cv.bitwise_and(img,img,mask=mask)
# cv.imshow('Masked', masked)

# plt.figure()
# plt.title('Color Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# pixels')

# colors = ('b','g','r')
# for i,col in enumerate(colors):
#     hist = cv.calcHist([img], [i], mask, [256], [0,256])
#     plt.plot(hist, color=col)
#     plt.xlim([0,256])

# plt.show()

# cv.waitKey(0)

#!------------------------------------------------------------------------------------
#!Thresholding
#? Binarization of an image- pixels are either 0/black or 255/white
# img = cv.imread('PhotoCV/ny_cp.jpg')
# cv.imshow('Central Park', img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

#*Simple Thresholding
# threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
# cv.imshow('Simple Threshold', thresh)

# threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
# cv.imshow('Simple Threshold Inverse', thresh_inv)

# cv.waitKey(0)

#*Adaptive Thresholding
# adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv.THRESH_BINARY, 11, 4)
# cv.imshow('Adaptive Thresholding', adaptive_thresh)

# cv.waitKey(0)

#!------------------------------------------------------------------------------------
#!Edges
# img = cv.imread('PhotoCV/ny_cp.jpg')
# # cv.imshow('Central Park', img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # cv.imshow('Gray', gray)

# #*Laplacian
# lap = cv.Laplacian(gray, cv.CV_64F)
# lap = np.uint8(np.absolute(lap))
# cv.imshow('Laplacian', lap)

# #*Sobel
# sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
# sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
# combined_sobel = cv.bitwise_or(sobelx,sobely)

# # cv.imshow('Sobel X', sobelx)
# # cv.imshow('Sobel Y', sobely)
# cv.imshow('Combined Sobel', combined_sobel)

# canny = cv.Canny(gray, 150, 175)
# cv.imshow('Canny', canny)

# cv.waitKey(0)