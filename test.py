import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands= 1)
classifier = Classifier("model/keras_model.h5","model/labels.txt")
offset = 20
imgsize = 300
folder = "data/c"
c=0
labels=["A","B","C"]
while True:
    success, img = cap.read()
    imgoutput=img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]

        x, y, w, h = hand["bbox"]

        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        imgCrop = img[y-offset: y+h+offset,  x-offset:x + w+offset]


        assertRatio = h/w
        if assertRatio>1:
            k = imgsize/h
            wcal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wcal,imgsize))
            imgresizeshape = imgResize.shape

            wgap = math.ceil((imgsize-wcal)/2)

            imgwhite[:, wgap:wcal+wgap] = imgResize
            pred,index=classifier.getPrediction(imgwhite,draw=False)
            print(pred,index)



        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize,hcal))
            imgresizeshape = imgResize.shape
            hgap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hgap:hcal + hgap,:] = imgResize
            pred, index = classifier.getPrediction(imgwhite,draw=False)
        cv2.rectangle(imgoutput, (x - offset, y - offset-50),
                      (x-offset+90,y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgoutput,labels[index],(x,y-25),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgoutput,(x-offset,y-offset),(x+w+offset,y+h+offset),
                      (255,0,255),4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Imagewhite", imgwhite)

    cv2.imshow("Image", imgoutput)
    cv2.waitKey(1)
