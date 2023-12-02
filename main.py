import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands= 1)

offset = 20
imgsize = 300
folder = "data/c"
c=0
while True:
    success, img = cap.read()
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
        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize,hcal))
            imgresizeshape = imgResize.shape
            hgap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hgap:hcal + hgap,:] = imgResize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Imagewhite", imgwhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        c+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(c)